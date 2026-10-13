#!/usr/bin/env python3
"""Dump c_1048 decisions from the Step-1 Python 3.7 / TF 1.15 environment.

Do not run this with the Step-3 PyTorch interpreter. The two dependency sets
cannot share a process.

    conda activate alphanl
    export PYTHONPATH=src
    python scripts/dump_c1048_bc.py --hands 200 --opponent self --out data/c1048_bc.npz

``--opponent pool`` plays c_1048 against other pickles in ``--pool``. Each
extra pickle is its own TF graph. ``--opponent random`` uses a uniform legal
opponent and records only the teacher seat.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.inference import AlphaNLHoldemAgent  # noqa: E402
from alphanl.nl_obs import encode_observation  # noqa: E402
from alphanl.paths import DEFAULT_WEIGHTS  # noqa: E402
from alphanl.tf_teacher import teacher_distribution  # noqa: E402


def _raw(env):
    last = env.last_obs
    state = last[0] if isinstance(last, (tuple, list)) else last
    if isinstance(state, dict) and "raw_obs" in state:
        return state["raw_obs"]
    raise RuntimeError("rlcard reset did not leave a raw_obs on the wrapper")


def _stage(raw):
    stage = raw["stage"]
    return int(stage.value if hasattr(stage, "value") else stage)


def _legal(raw):
    actions = raw["legal_actions"]
    if isinstance(actions, dict):
        actions = list(actions.keys())
    out = []
    for action in actions:
        value = action.value if hasattr(action, "value") else action
        out.append(int(value))
    return out


def _infer_sb(all_chips):
    matches = [index for index, chips in enumerate(all_chips) if int(chips) == 1]
    if len(matches) != 1:
        raise RuntimeError("expected one small blind, all_chips=%s" % (list(all_chips),))
    return int(matches[0])


def _meta(env, sb_seat):
    raw = _raw(env)
    return {
        "hand": [str(card) for card in raw["hand"]],
        "public_cards": [str(card) for card in (raw.get("public_cards") or [])],
        "all_chips": [float(chip) for chip in raw["all_chips"]],
        "stakes": [float(chip) for chip in raw["stakes"]],
        "pot": float(raw["pot"]),
        "stage": _stage(raw),
        "legal_actions": _legal(raw),
        "current_player": int(env.my_agent()),
        "history": copy.deepcopy(env.history),
        "sb_seat": int(sb_seat),
        "big_blind": 2,
        "small_blind": 1,
        "starting_stack": 100,
    }


def _sample(probs, rng):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        raise RuntimeError("teacher distribution is empty")
    probs = probs / total
    return int(rng.choice(len(probs), p=probs))


def _random_legal(obs, rng):
    legal = np.flatnonzero(np.asarray(obs["legal_moves"]) > 0.5)
    if len(legal) == 0:
        raise RuntimeError("opponent saw no legal action")
    return int(rng.choice(legal))


def dump_hands(agent, env, hands, opponent, pool_agents, rng):
    card, action, extra, legal, probs, actions, meta = [], [], [], [], [], [], []
    for hand_index in range(int(hands)):
        obs = env.reset()
        sb_seat = _infer_sb(_raw(env)["all_chips"])
        teacher_seat = int(hand_index % 2) if opponent != "self" else None
        pool_agent = None
        if opponent == "pool":
            if not pool_agents:
                raise RuntimeError("opponent=pool needs at least one --pool pickle")
            pool_agent = pool_agents[int(rng.integers(0, len(pool_agents)))]
        done = False
        guard = 0
        while not done:
            guard += 1
            if guard > 80:
                raise RuntimeError("hand exceeded 80 decisions")
            seat = int(env.my_agent())
            record = opponent == "self" or seat == teacher_seat
            if record:
                distribution, _greedy = teacher_distribution(agent, obs)
                chosen = _sample(distribution, rng)
                snapshot = _meta(env, sb_seat)
                encoded = encode_observation(_raw(env), env.history, seat)
                # The wrapper observation is what the teacher saw. Keep that copy.
                card.append(np.asarray(obs["card_info"], dtype=np.float32))
                action.append(np.asarray(obs["action_info"], dtype=np.float32))
                extra.append(np.asarray(obs["extra_info"], dtype=np.float32))
                legal.append(np.asarray(obs["legal_moves"], dtype=np.float32))
                probs.append(np.asarray(distribution, dtype=np.float32))
                actions.append(chosen)
                meta.append(json.dumps(snapshot))
                # encoded is checked against the wrapper so a drift fails the dump.
                if not np.array_equal(encoded["legal_mask"], legal[-1]):
                    raise RuntimeError("nl_obs legal mask does not match the TF wrapper")
                step_action = chosen
            elif opponent == "random":
                step_action = _random_legal(obs, rng)
            else:
                distribution, _greedy = teacher_distribution(pool_agent, obs)
                step_action = _sample(distribution, rng)
            obs, _reward, done, _info = env.step(int(step_action))
    if not actions:
        raise RuntimeError("dump produced no decisions")
    return {
        "card_info": np.stack(card).astype(np.float32),
        "action_info": np.stack(action).astype(np.float32),
        "extra_info": np.stack(extra).astype(np.float32),
        "legal_mask": np.stack(legal).astype(np.float32),
        "probs": np.stack(probs).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "meta_json": np.asarray(meta, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--hands", type=int, default=50)
    parser.add_argument("--opponent", choices=("self", "random", "pool"), default="self")
    parser.add_argument("--pool", type=Path, nargs="*", default=())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    agent = AlphaNLHoldemAgent(weights_path=args.weights)
    env = agent.env
    pool_agents = []
    for path in args.pool:
        pool_agents.append(AlphaNLHoldemAgent(weights_path=path))
    payload = dump_hands(agent, env, args.hands, args.opponent, pool_agents, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.out), **payload)
    print("wrote %s decisions=%d" % (args.out, int(payload["actions"].shape[0])))


if __name__ == "__main__":
    main()

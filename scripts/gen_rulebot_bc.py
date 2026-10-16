#!/usr/bin/env python3
"""Generate behavior-cloning rows from the Botzone rule bot.

The rule bot (``alphanl.botzone.rule_bot.BotzoneRuleBot``) is not modified.
Each recorded decision stores the botzone transformer observation PPO encodes
(``BettingHistoryTokenizer`` on ``BotzoneEnvTable.snapshot``), the maskfold
policy legal mask, and the rule bot's target action id.

Action-id mapping
-----------------
``BotzoneRuleBot.action_id`` returns one of the five policy ids. This script
reads that id's Botzone response (the chip code from ``response_for_action``)
and maps the code onto the five ids. The same rule applies to any chip amount,
not only the five packed responses.

1. Fold (``-1``). If ``mask_free_fold`` is on and the spot is a free check
   (``to_call == 0`` and check/call is legal), the target is rewritten to
   check_call (id 1). Otherwise the target is fold (id 0). This is the same
   spot ``policy_action_mask`` clears: fold is dropped only when nothing is
   owed and check/call is available. The rule bot itself is not edited; the
   rewrite happens here, and the data-policy seat plays the rewritten id.
2. Check/call (``0``) maps to check_call (id 1).
3. Any other code is a wager. ``-2`` (all-in) has chip amount equal to the
   actor's remaining stack. A positive code's chip amount is that extra
   (the response is extra chips, not raise-to). Candidates are the legal
   subset of raise_min (id 2), raise_large (id 3), and all_in (id 4). Raise
   chip amounts are the packed positive extras from ``action_responses``.
   All-in chip amount is the remaining stack, not the code ``-2``. The target
   is the candidate with the smallest absolute difference. A tie goes to the
   smaller action id: raise_min, then raise_large, then all_in.

The live rule bot only emits a packed response, so step 3 is the identity on
those five codes. An amount that sits between the sizes is what the nearest
rule is for. Decisions whose policy mask has fewer than two legal ids are
not counted (``skipped_single_legal``).

``raise_large`` stays in the map, but this rule bot's priority tables do not
emit it: ``action_responses`` only marks raise_large legal when raise_min is
legal too, all-in is legal whenever the seat can act, and every strength
band ranks all-in or raise_min ahead of raise_large. Do not retune the bot
to fill that bucket.

Short form stored in metadata ``action_mapping``: Fold (-1): if mask_free_fold
and the spot is a free check (to_call == 0 and check/call is legal), rewrite
the target to check_call (id 1); otherwise the target is fold (id 0).
Check/call (0) maps to check_call (id 1). Any other code is a wager: -2
(all-in) has chip amount equal to the actor's remaining stack; a positive
code's chip amount is that extra. Candidates are the legal subset of
raise_min (id 2), raise_large (id 3), and all_in (id 4). The target is the
candidate with the smallest absolute chip difference. A tie goes to the
smaller action id (raise_min, then raise_large, then all_in).

Opponents
---------
One opponent per hand, equal weight: another rule bot, uniform random over
environment-legal ids, or this same mapped policy with probability
``epsilon`` (default 0.2) replaced by that uniform draw. Random and epsilon
draws use environment legality, matching the training note that rule and
random opponents do not use the policy mask. Only the data-policy seat
(``hand_index % 2``) is recorded. That seat always plays the mapped target.

Output
------
``<out>/decisions.npz`` and ``<out>/metadata.json``. Default out is
``runs/bc_rulebot/``. Writes refuse any other directory under ``runs/``.
Existing files in the output directory are left in place. ``decisions.npz``
and ``metadata.json`` are replaced only with ``--force``.

``legal_mask`` is the mask the PPO forward sees when ``mask_free_fold`` is
on (environment mask, then ``policy_action_mask``). ``env_legal_mask`` is
the environment mask before that drop. Token fields match the tokenizer.
``probs`` is a one-hot of the target id so a later KL run can treat the rule
bot as a deterministic teacher. ``train_bc.py``'s current transformer loader
still expects c_1048 ``meta_json``; this file does not call it.

Legality is whatever ``src/alphanl/botzone/legal.py`` is on the tree you run.
``metadata.env_version.legal_py_sha256`` is the sha256 of that file.
``load_dataset`` refuses a directory whose hash is not the current file
(BC training calls the same check). Regenerate after ``legal.py`` changes.
Do not mix two hashes.

Laptop command for at least 200k counted decisions (seed 42). The last hand
is finished, so the count can be slightly above 200k::

    PYTHONPATH=src python3 scripts/gen_rulebot_bc.py \\
        --decisions 200000 --seed 42 --out runs/bc_rulebot

Small sample::

    PYTHONPATH=src python3 scripts/gen_rulebot_bc.py \\
        --hands 24 --seed 42 --out tests/fixtures/bc_rulebot --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.botzone.actions import (  # noqa: E402
    ACTION_NAMES,
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    NUM_ACTIONS,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    action_responses,
    response_for_action,
)
from alphanl.botzone.legal import ALL_IN, CHECK_CALL, FOLD, HandState, to_call  # noqa: E402
from alphanl.botzone.rule_bot import BotzoneRuleBot  # noqa: E402

# Identical to the "Action-id mapping" section of the module docstring.
ACTION_MAPPING_RULE = (
    "Fold (-1): if mask_free_fold and the spot is a free check "
    "(to_call == 0 and check/call is legal), rewrite the target to check_call "
    "(id 1); otherwise the target is fold (id 0). "
    "Check/call (0) maps to check_call (id 1). "
    "Any other code is a wager: -2 (all-in) has chip amount equal to the "
    "actor's remaining stack; a positive code's chip amount is that extra. "
    "Candidates are the legal subset of raise_min (id 2), raise_large (id 3), "
    "and all_in (id 4). The target is the candidate with the smallest absolute "
    "chip difference. A tie goes to the smaller action id "
    "(raise_min, then raise_large, then all_in)."
)

DEFAULT_SEED = 42
DEFAULT_EPSILON = 0.2
DEFAULT_OUT = REPO_ROOT / "runs" / "bc_rulebot"
FIXTURE_HANDS = 24
FIXTURE_SEED = 42
FORMAT_VERSION = 1
OPPONENT_NAMES: Tuple[str, ...] = ("rule", "random", "epsilon")
WAGER_ACTIONS: Tuple[int, ...] = (RAISE_MIN_ACTION, RAISE_LARGE_ACTION, ALL_IN_ACTION)
OWNED_FILES = ("decisions.npz", "metadata.json")
REQUIRED_METADATA_KEYS = (
    "format_version",
    "seed",
    "num_hands",
    "num_decisions",
    "num_hands_with_decisions",
    "skipped_single_legal",
    "action_id_distribution",
    "action_id_fraction",
    "mask_free_fold",
    "epsilon",
    "opponent_mix",
    "opponent_hand_counts",
    "env_version",
    "max_seq_len",
    "data_policy",
    "recorded_seat",
    "mapped_differs_from_raw",
    "action_mapping",
    "arrays",
    "elapsed_seconds",
)

MASKFOLD_CONFIG = REPO_ROOT / "configs" / "transformer" / "botzone_laptop_8gb_maskfold.json"


def default_max_seq_len() -> int:
    """Sequence length of the laptop maskfold transformer, without widening it."""

    with MASKFOLD_CONFIG.open(encoding="utf-8") as handle:
        return int(json.load(handle)["model"]["max_seq_len"])


def environment_version() -> dict:
    """Git commit and ``legal.py`` hash of the tree that dealt these hands."""

    legal_path = REPO_ROOT / "src" / "alphanl" / "botzone" / "legal.py"
    digest = hashlib.sha256(legal_path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "backend": "botzone",
        "game": "TexasHoldem2p",
        "stacks": 20000,
        "small_blind": 50,
        "big_blind": 100,
        "git_commit": commit,
        "legal_py": "src/alphanl/botzone/legal.py",
        "legal_py_sha256": digest,
        "note": (
            "Raise sizes and legality come from legal.py at git_commit. "
            "BC training refuses a dataset whose legal_py_sha256 is not the "
            "hash of src/alphanl/botzone/legal.py on the tree that loads it. "
            "Regenerate after legal.py changes. Do not mix two hashes."
        ),
    }


def is_free_check(state: HandState) -> bool:
    """Nothing owed, and check/call is an environment-legal policy id."""

    if to_call(state) != 0:
        return False
    responses = action_responses(state)
    return responses[CHECK_CALL_ACTION] is not None


def wager_chips(state: HandState, response: int) -> int:
    """Chip amount used for nearest-wager distance.

    All-in (``-2``) spends the remaining stack. A positive response is the
    extra chips put in on this action.
    """

    if response == ALL_IN:
        return int(state.stacks[state.actor])
    if isinstance(response, bool) or not isinstance(response, int) or response <= 0:
        raise ValueError(f"wager response must be -2 or a positive extra, got {response!r}")
    return int(response)


def wager_candidates(state: HandState) -> Tuple[Tuple[int, int], ...]:
    """Legal ``(action_id, chip_amount)`` for raise_min, raise_large, all_in."""

    responses = action_responses(state)
    found = []
    for action_id in WAGER_ACTIONS:
        response = responses[action_id]
        if response is None:
            continue
        found.append((action_id, wager_chips(state, int(response))))
    return tuple(found)


def nearest_wager_action(state: HandState, response: int) -> int:
    """Legal wager id closest in chips to ``response``. Ties: smaller id."""

    chips = wager_chips(state, response)
    candidates = wager_candidates(state)
    if not candidates:
        raise ValueError("no legal raise_min, raise_large, or all_in to map onto")
    action_id, _amount = min(candidates, key=lambda item: (abs(item[1] - chips), item[0]))
    return int(action_id)


def map_response_to_action_id(
    state: HandState,
    response: int,
    *,
    mask_free_fold: bool,
) -> int:
    """Map a Botzone chip code to a policy id. See the module docstring."""

    if isinstance(response, bool) or not isinstance(response, int):
        raise ValueError(f"response must be an int, got {response!r}")
    if response == FOLD:
        if mask_free_fold and is_free_check(state):
            return CHECK_CALL_ACTION
        if action_responses(state)[FOLD_ACTION] is None:
            raise ValueError("fold is not legal")
        return FOLD_ACTION
    if response == CHECK_CALL:
        if action_responses(state)[CHECK_CALL_ACTION] is None:
            raise ValueError("check/call is not legal")
        return CHECK_CALL_ACTION
    if response == ALL_IN or response > 0:
        return nearest_wager_action(state, response)
    raise ValueError(f"response {response} is not fold, check/call, all-in, or a positive extra")


def decision_is_counted(legal_mask: np.ndarray) -> bool:
    """True when the policy mask has at least two legal ids."""

    return int(np.count_nonzero(np.asarray(legal_mask) > 0.5)) >= 2


def rulebot_target(
    table,
    bot: BotzoneRuleBot,
    *,
    mask_free_fold: bool,
) -> Tuple[int, int, int]:
    """``(raw action id, raw response, mapped action id)`` for the actor.

    ``table`` is a ``BotzoneTable``. The bot's policy function is called
    as-is; only the response is mapped.
    """

    raw = int(bot.action_id(table))
    response = response_for_action(table.state, raw)
    if response is None:
        raise RuntimeError(f"rule bot returned action {raw} with no response")
    mapped = map_response_to_action_id(table.state, int(response), mask_free_fold=mask_free_fold)
    return raw, int(response), int(mapped)


def _botzone_table(env_table):
    """Inner ``BotzoneTable``. The env table does not expose it publicly."""

    return env_table._table


def encode_policy_observation(env_table, seat: int, tokenizer, *, mask_free_fold: bool) -> dict:
    """One PPO observation. ``legal_mask`` is the maskfold policy mask."""

    import torch
    from alphanl.transformer.decision_stats import policy_action_mask

    snapshot = env_table.snapshot(int(seat))
    batch = tokenizer.encode(snapshot)
    env_mask = batch["legal_mask"]
    owed = float(snapshot.to_call)
    if mask_free_fold:
        policy = policy_action_mask(env_mask, owed, True)
    else:
        policy = env_mask
    if not isinstance(policy, torch.Tensor):
        raise RuntimeError("policy mask was not a tensor")
    numeric = batch["numeric"][0].detach().cpu().numpy().astype(np.float32)
    legal = policy[0].detach().cpu().numpy().astype(np.float32)
    env_legal = env_mask[0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(numeric).all() or not np.isfinite(legal).all():
        raise RuntimeError("non-finite observation")
    return {
        "token_ids": batch["token_ids"][0].detach().cpu().numpy().astype(np.int64),
        "type_ids": batch["type_ids"][0].detach().cpu().numpy().astype(np.int64),
        "numeric": numeric,
        "pad_mask": batch["pad_mask"][0].detach().cpu().numpy().astype(np.bool_),
        "legal_mask": legal,
        "env_legal_mask": env_legal,
        "to_call": np.float32(owed),
    }


def _opponent_action(kind: str, mapped: int, legal: Sequence[int], rng: np.random.Generator, epsilon: float) -> int:
    """Action the opponent actually plays. Environment-legal ids only."""

    allowed = tuple(int(action) for action in legal)
    if not allowed:
        raise RuntimeError("opponent has no legal action")
    if kind == "rule":
        if mapped not in allowed:
            raise RuntimeError(f"mapped rule action {mapped} is not environment-legal")
        return int(mapped)
    if kind == "random":
        return int(allowed[int(rng.integers(0, len(allowed)))])
    if kind == "epsilon":
        if float(rng.random()) < float(epsilon):
            return int(allowed[int(rng.integers(0, len(allowed)))])
        if mapped not in allowed:
            raise RuntimeError(f"mapped rule action {mapped} is not environment-legal")
        return int(mapped)
    raise ValueError(f"unknown opponent kind {kind!r}")


def _check_limits(hands: Optional[int], decisions: Optional[int]) -> None:
    if hands is None and decisions is None:
        raise ValueError("set hands or decisions")
    if hands is not None and (isinstance(hands, bool) or int(hands) < 1):
        raise ValueError(f"hands must be >= 1, got {hands!r}")
    if decisions is not None and (isinstance(decisions, bool) or int(decisions) < 1):
        raise ValueError(f"decisions must be >= 1, got {decisions!r}")


def collect_decisions(
    *,
    seed: int = DEFAULT_SEED,
    hands: Optional[int] = None,
    decisions: Optional[int] = None,
    epsilon: float = DEFAULT_EPSILON,
    mask_free_fold: bool = True,
    max_seq_len: Optional[int] = None,
    progress_every: int = 5000,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Play hands and return ``(arrays, metadata)``. Does not write files."""

    from alphanl.transformer.botzone_table import BotzoneEnvTable
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    _check_limits(hands, decisions)
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
    seq_len = default_max_seq_len() if max_seq_len is None else int(max_seq_len)
    if seq_len < 8:
        raise ValueError(f"max_seq_len must be >= 8, got {seq_len}")

    started = time.perf_counter()
    env_table = BotzoneEnvTable(seed=int(seed))
    tokenizer = BettingHistoryTokenizer(max_seq_len=seq_len)
    bot = BotzoneRuleBot()
    # Independent of the deck's random.Random(seed) inside BotzoneEnvTable.
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    rows: List[dict] = []
    skipped = 0
    played = 0
    hand_counts = {name: 0 for name in OPPONENT_NAMES}
    mapped_differs = 0
    hand_cap = None
    if decisions is not None:
        hand_cap = int(decisions) * 2

    while True:
        if hands is not None and played >= int(hands):
            break
        if decisions is not None and len(rows) >= int(decisions):
            break
        if hand_cap is not None and played > hand_cap and len(rows) < int(decisions):
            raise RuntimeError(
                f"played {played} hands and only recorded {len(rows)} decisions; "
                "refusing to spin toward the decision target"
            )
        kind_index = int(rng.integers(0, len(OPPONENT_NAMES)))
        kind = OPPONENT_NAMES[kind_index]
        hand_counts[kind] += 1
        expert_seat = played % 2
        env_table.reset()
        guard = 0
        while not env_table.is_over():
            guard += 1
            if guard > 80:
                raise RuntimeError("hand exceeded 80 decisions")
            seat = int(env_table.current_player())
            inner = _botzone_table(env_table)
            raw_id, raw_response, mapped = rulebot_target(
                inner, bot, mask_free_fold=bool(mask_free_fold)
            )
            encoded = encode_policy_observation(
                env_table, seat, tokenizer, mask_free_fold=bool(mask_free_fold)
            )
            if float(encoded["legal_mask"][mapped]) <= 0.5:
                raise RuntimeError(
                    f"mapped action {mapped} is outside the policy mask "
                    f"{encoded['legal_mask'].tolist()} (raw {raw_id} response {raw_response})"
                )
            if seat == expert_seat:
                if decision_is_counted(encoded["legal_mask"]):
                    if raw_id != mapped:
                        mapped_differs += 1
                    encoded["actions"] = np.int64(mapped)
                    encoded["raw_action_id"] = np.int64(raw_id)
                    encoded["raw_response"] = np.int64(raw_response)
                    encoded["game_id"] = np.int64(played)
                    encoded["seat"] = np.int64(seat)
                    encoded["opponent_id"] = np.int64(kind_index)
                    rows.append(encoded)
                    if progress_every and len(rows) % int(progress_every) == 0:
                        print(f"recorded {len(rows)} decisions ({played + 1} hands)", flush=True)
                else:
                    skipped += 1
            legal = env_table.legal_action_ids()
            if seat == expert_seat:
                play = mapped
            else:
                play = _opponent_action(kind, mapped, legal, rng, float(epsilon))
            if play not in legal:
                raise RuntimeError(f"play {play} is not environment-legal {legal}")
            env_table.step(int(play))
        played += 1

    if not rows:
        raise RuntimeError("no counted decisions; refuse to write an empty dataset")

    arrays = _stack_rows(rows)
    elapsed = time.perf_counter() - started
    metadata = _metadata(
        arrays,
        seed=int(seed),
        hands=played,
        skipped=skipped,
        epsilon=float(epsilon),
        mask_free_fold=bool(mask_free_fold),
        max_seq_len=seq_len,
        hand_counts=hand_counts,
        mapped_differs=mapped_differs,
        elapsed=elapsed,
    )
    return arrays, metadata


def _stack_rows(rows: Sequence[dict]) -> Dict[str, np.ndarray]:
    actions = np.asarray([int(row["actions"]) for row in rows], dtype=np.int64)
    probs = np.zeros((len(rows), NUM_ACTIONS), dtype=np.float32)
    probs[np.arange(len(rows)), actions] = 1.0
    arrays = {
        "token_ids": np.stack([row["token_ids"] for row in rows]).astype(np.int64),
        "type_ids": np.stack([row["type_ids"] for row in rows]).astype(np.int64),
        "numeric": np.stack([row["numeric"] for row in rows]).astype(np.float32),
        "pad_mask": np.stack([row["pad_mask"] for row in rows]).astype(np.bool_),
        "legal_mask": np.stack([row["legal_mask"] for row in rows]).astype(np.float32),
        "env_legal_mask": np.stack([row["env_legal_mask"] for row in rows]).astype(np.float32),
        "actions": actions,
        "probs": probs,
        "to_call": np.asarray([row["to_call"] for row in rows], dtype=np.float32),
        "game_id": np.asarray([row["game_id"] for row in rows], dtype=np.int64),
        "seat": np.asarray([row["seat"] for row in rows], dtype=np.int64),
        "opponent_id": np.asarray([row["opponent_id"] for row in rows], dtype=np.int64),
        "raw_action_id": np.asarray([row["raw_action_id"] for row in rows], dtype=np.int64),
        "raw_response": np.asarray([row["raw_response"] for row in rows], dtype=np.int64),
    }
    if not np.isfinite(arrays["numeric"]).all() or not np.isfinite(arrays["legal_mask"]).all():
        raise RuntimeError("stacked dataset has NaN or Inf")
    if not np.isfinite(arrays["to_call"]).all() or not np.isfinite(arrays["probs"]).all():
        raise RuntimeError("stacked dataset has NaN or Inf")
    return arrays


def _metadata(
    arrays: Dict[str, np.ndarray],
    *,
    seed: int,
    hands: int,
    skipped: int,
    epsilon: float,
    mask_free_fold: bool,
    max_seq_len: int,
    hand_counts: Dict[str, int],
    mapped_differs: int,
    elapsed: float,
) -> dict:
    actions = arrays["actions"]
    counts = {name: int(np.count_nonzero(actions == index)) for index, name in enumerate(ACTION_NAMES)}
    total = int(actions.shape[0])
    fractions = {name: (counts[name] / total if total else 0.0) for name in ACTION_NAMES}
    games_with = int(len(set(int(game) for game in arrays["game_id"].tolist())))
    return {
        "format_version": FORMAT_VERSION,
        "seed": int(seed),
        "deck_rng": "random.Random(seed) inside BotzoneEnvTable",
        "opponent_rng": "numpy.random.Generator(PCG64(seed))",
        "num_hands": int(hands),
        "num_decisions": total,
        "num_hands_with_decisions": games_with,
        "skipped_single_legal": int(skipped),
        "min_legal_actions": 2,
        "action_id_distribution": counts,
        "action_id_fraction": fractions,
        "mask_free_fold": bool(mask_free_fold),
        "epsilon": float(epsilon),
        "opponent_mix": {
            "kinds": list(OPPONENT_NAMES),
            "weights": [1, 1, 1],
            "selection": "one kind per hand, uniform over kinds, drawn before the hand",
            "epsilon": float(epsilon),
            "epsilon_meaning": (
                "With probability epsilon the opponent plays uniform over "
                "environment-legal action ids; otherwise it plays the mapped rule-bot target."
            ),
            "random_meaning": (
                "Uniform over environment-legal action ids. A random opponent may "
                "fold a free check. Those seats are not recorded."
            ),
            "rule_meaning": (
                "The mapped rule-bot target, including the free-check fold rewrite."
            ),
        },
        "opponent_hand_counts": {name: int(hand_counts[name]) for name in OPPONENT_NAMES},
        "env_version": environment_version(),
        "max_seq_len": int(max_seq_len),
        "network": "laptop_8gb botzone transformer (configs/transformer/botzone_laptop_8gb_maskfold.json); not widened",
        "data_policy": "BotzoneRuleBot.action_id, unchanged, then map_response_to_action_id",
        "recorded_seat": "hand_index % 2 (the data-policy seat). The other seat is the opponent mix.",
        "mapped_differs_from_raw": int(mapped_differs),
        "action_mapping": ACTION_MAPPING_RULE,
        "arrays": {
            "token_ids": "int64 [N, max_seq_len], tokenizer ids",
            "type_ids": "int64 [N, max_seq_len]",
            "numeric": "float32 [N, max_seq_len, 4]",
            "pad_mask": "bool [N, max_seq_len], True where padded",
            "legal_mask": "float32 [N, 5], policy mask after mask_free_fold",
            "env_legal_mask": "float32 [N, 5], environment mask before the free-fold drop",
            "actions": "int64 [N], mapped target action id",
            "probs": "float32 [N, 5], one-hot of actions (deterministic teacher for KL)",
            "to_call": "float32 [N], chips owed at the decision",
            "game_id": "int64 [N], hand index, for a later holdout by games",
            "seat": "int64 [N]",
            "opponent_id": "int64 [N], index into opponent_mix.kinds",
            "raw_action_id": "int64 [N], BotzoneRuleBot.action_id before mapping",
            "raw_response": "int64 [N], response_for_action of raw_action_id",
        },
        "elapsed_seconds": float(elapsed),
        "bc_training": (
            "Not run by this script. A later KL run should load these token fields "
            "and probs. scripts/train_bc.py still rebuilds transformer rows from "
            "c_1048 meta_json and will not load this npz unchanged."
        ),
    }


def assert_output_dir(out: Path, force: bool) -> Path:
    """Refuse any ``runs/`` path other than ``runs/bc_rulebot``."""

    out = Path(out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out = out.resolve()
    runs = (REPO_ROOT / "runs").resolve()
    allowed = (runs / "bc_rulebot").resolve()
    if out == runs or (runs in out.parents and allowed != out and allowed not in out.parents):
        raise SystemExit(
            f"refusing to write {out}. Under runs/, only runs/bc_rulebot/ is allowed, "
            "and nothing already in another runs/ directory is touched."
        )
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} exists and is not a directory")
    if out.exists():
        for name in OWNED_FILES:
            path = out / name
            if path.exists() and not force:
                raise SystemExit(
                    f"{path} already exists. Pass --force to replace {', '.join(OWNED_FILES)} "
                    "only. Other files in the directory are never deleted."
                )
    return out


def write_dataset(out: Path, arrays: Dict[str, np.ndarray], metadata: dict, *, force: bool) -> Path:
    """Write ``decisions.npz`` and ``metadata.json``. Does not delete other files."""

    out = assert_output_dir(out, force)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metadata, indent=2, sort_keys=True)
    payload = text.encode("utf-8")
    np.savez_compressed(
        out / "decisions.npz",
        metadata_json=np.frombuffer(payload, dtype=np.uint8).copy(),
        **arrays,
    )
    (out / "metadata.json").write_text(text + "\n", encoding="utf-8")
    return out


class LegalPyMismatch(ValueError):
    """Dataset legality hash is not the ``legal.py`` on this tree."""


def current_legal_py_sha256() -> str:
    """sha256 of ``src/alphanl/botzone/legal.py`` as it is on disk now."""

    return environment_version()["legal_py_sha256"]


def assert_legal_py_sha256(recorded: str, *, where: str) -> None:
    """Refuse to train or load when the dataset was dealt by another ``legal.py``."""

    current = current_legal_py_sha256()
    if str(recorded) != current:
        raise LegalPyMismatch(
            f"refusing {where}: legal_py_sha256 {recorded} != current legal.py {current}. "
            "Regenerate with scripts/gen_rulebot_bc.py on this tree. Do not mix two hashes."
        )


def load_dataset(out: Path, *, check_legal: bool = True) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load a directory written by :func:`write_dataset`.

    ``check_legal`` is on by default. A hash that is not the current
    ``legal.py`` raises :class:`LegalPyMismatch` before any training step.
    """

    out = Path(out)
    file_meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    with np.load(out / "decisions.npz", allow_pickle=False) as blob:
        embedded = json.loads(bytes(blob["metadata_json"]).decode("utf-8"))
        arrays = {key: blob[key] for key in blob.files if key != "metadata_json"}
    if embedded != file_meta:
        raise ValueError("metadata.json does not match decisions.npz metadata_json")
    if check_legal:
        recorded = file_meta.get("env_version", {}).get("legal_py_sha256")
        assert_legal_py_sha256("" if recorded is None else str(recorded), where=str(out))
    return arrays, file_meta


def _print_summary(metadata: dict, out: Path) -> None:
    dist = metadata["action_id_distribution"]
    print(f"wrote {out}")
    print(
        f"hands={metadata['num_hands']} decisions={metadata['num_decisions']} "
        f"skipped_single_legal={metadata['skipped_single_legal']} "
        f"mapped_differs_from_raw={metadata['mapped_differs_from_raw']} "
        f"elapsed_seconds={metadata['elapsed_seconds']:.3f}"
    )
    print("action_id_distribution " + " ".join(f"{name}={dist[name]}" for name in ACTION_NAMES))
    print(
        "opponent_hand_counts "
        + " ".join(f"{name}={metadata['opponent_hand_counts'][name]}" for name in OPPONENT_NAMES)
    )
    print(f"mask_free_fold={metadata['mask_free_fold']} seed={metadata['seed']}")
    version = metadata["env_version"]
    print(f"git_commit={version['git_commit']} legal_py_sha256={version['legal_py_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", type=int, default=None, help="Stop once at least this many counted decisions are stored")
    parser.add_argument("--hands", type=int, default=None, help="Stop after this many hands (full hands)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Default: maskfold laptop config")
    parser.add_argument("--mask-free-fold", dest="mask_free_fold", action="store_true", default=True)
    parser.add_argument("--no-mask-free-fold", dest="mask_free_fold", action="store_false")
    parser.add_argument("--force", action="store_true", help="Replace decisions.npz and metadata.json if they exist")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    if args.decisions is None and args.hands is None:
        raise SystemExit("pass --decisions and/or --hands")
    arrays, metadata = collect_decisions(
        seed=args.seed,
        hands=args.hands,
        decisions=args.decisions,
        epsilon=args.epsilon,
        mask_free_fold=args.mask_free_fold,
        max_seq_len=args.max_seq_len,
        progress_every=args.progress_every,
    )
    out = write_dataset(args.out, arrays, metadata, force=args.force)
    _print_summary(metadata, out)


if __name__ == "__main__":
    main()

"""Bounded, full 200BB external evaluation of a frozen CFR trajectory mixture.

Replays exact public chip amounts through HunlTable. Unknown cards are filled
only to construct the engine; no invented card reaches a policy observation.
No training state is loaded and no 50BB cap or legacy AlphaNL encoder is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import requests
import torch

from alphanl.cfr.evaluation import CFRActor, mean_interval
from alphanl.transformer.hunl import HunlTable
from alphanl.transformer.schema import normalize_cards
from alphanl.transformer.tables import aligned_decks, standard_deck
from slumbot_client import State, post


def decision_table(response, *, bet_telemetry=None):
    seat = response["client_pos"]
    parsed = State(response["action"])
    if parsed.pos != seat or parsed.over:
        raise ValueError("Slumbot response is not a hero decision")
    hole, board = normalize_cards(response["hole_cards"]), normalize_cards(response["board"])
    if len(hole) != 2 or len(board) != (0, 3, 4, 5)[parsed.street]:
        raise ValueError("Slumbot card count disagrees with street")
    known = hole + board
    if len(set(known)) != len(known):
        raise ValueError("duplicate observed cards")
    remaining = [card for card in standard_deck() if card not in known]
    other = remaining[:2]
    filled_board = list(board) + remaining[2:7 - len(board)]
    holes = [hole, hole]
    holes[1 - seat] = other
    _, deck = aligned_decks(holes[0], holes[1], filled_board)
    table = HunlTable(seed=0)
    table.reset(deck)
    street_start, last_street = [0, 0], 0
    for event_index, (street, player, kind, before, after) in enumerate(parsed.events):
        if street != last_street:
            street_start, last_street = before, street
        if table.street != street or table.current_player() != player:
            raise ValueError("Slumbot action order disagrees with standard HUNL")
        if kind == "b":
            target = after[player] - street_start[player]
            if bet_telemetry is not None:
                abstract_targets = {int(table.state.bets[player]) + table.action_chips(a)
                    for a in table.legal_action_ids() if a in (2, 3, 4)}
                bet_telemetry[event_index] = dict(player=player, off_tree=target not in abstract_targets)
            table.step_raise_to(target)
        else:
            table.step(0 if kind == "f" else 1)
    snapshot = table.snapshot(seat)
    if (snapshot.board_cards != board or snapshot.hole_cards != hole
            or snapshot.street != parsed.street
            or snapshot.pot != sum(parsed.committed(p) for p in (0, 1))
            or snapshot.stacks != tuple(20000 - parsed.committed(p) for p in (0, 1))):
        raise ValueError("Slumbot replay does not match observed state")
    return table, parsed


def api_action(table, parsed, action):
    chips = table.action_chips(action)
    if chips is None:
        raise ValueError("CFR selected an illegal Slumbot action")
    if action == 0:
        return "f"
    if action == 1:
        return "c" if table.snapshot(parsed.pos).to_call > 0 else "k"
    return f"b{parsed.contrib[parsed.pos] + chips}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--cache-size", type=int, default=100)
    parser.add_argument("--spr-action-guard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--guard-max-pot-ratio", type=float)
    args = parser.parse_args(argv)
    if min(args.hands, args.iteration) < 1:
        parser.error("hands and iteration must be positive")
    torch.set_num_threads(1)
    actor = CFRActor(args.strategies, seed=args.seed, through=args.iteration,
                     cache_size=args.cache_size, spr_guard=args.spr_action_guard,
                     max_pot_ratio=args.guard_max_pot_ratio)
    if actor.policy.iterations.tolist() != list(range(1, args.iteration + 1)):
        raise ValueError("full average requires every strategy snapshot 1..T")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"format": "cfr_slumbot_200bb_v1", "iteration": args.iteration,
              "requested_hands": args.hands, "seed": args.seed, "stack_bb": 200,
              "spr_guard_override": args.spr_action_guard,
              "max_pot_ratio_override": args.guard_max_pot_ratio,
              "policy": "linear-weighted mixture; one snapshot per hand; optional mask intervention",
              "snapshot_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in actor.policy.paths},
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / "eval/slumbot_client.py",
                                          *sorted((ROOT / "src/alphanl").rglob("*.py"))]}}
    winnings = []
    def publish():
        report["completed_hands"] = len(winnings)
        report["audit"] = actor.audit.summary()
        if winnings:
            report["performance"] = mean_interval(winnings)
            report["performance"]["unit"] = "mbb/hand"
            report["performance"]["ci_method"] = "normal_1.96_sample_std_per_hand"
        temporary = args.out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out / "report.json")
    publish()
    token = None
    with requests.Session() as session, (args.out / "hands.jsonl").open("w") as log:
        try:
            for hand in range(args.hands):
                response = post(session, "new_hand", {"token": token} if token else {}, retries=2)
                actor.begin_hand(response["client_pos"])
                decisions = []
                while "winnings" not in response:
                    token = response.get("token", token)
                    table, parsed = decision_table(response)
                    action = actor.act(table.snapshot(parsed.pos))
                    sent = api_action(table, parsed, action)
                    decisions.append({"history": response["action"], "action_id": action, "sent": sent})
                    response = post(session, "act", {"token": token, "incr": sent}, retries=2)
                token = response.get("token", token)
                winnings.append(response["winnings"] * 10.)
                # API tokens are session secrets; never persist them.
                log.write(json.dumps({"hand": hand, "snapshot": actor.policy.hand_iteration,
                                      "decisions": decisions,
                                      "result": {k: v for k, v in response.items() if k != "token"}}) + "\n")
                log.flush()
                publish()
                print(f"slumbot {hand + 1}/{args.hands}", flush=True)
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            publish()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent full-200BB Slumbot evaluation for teacher, student or residual.

Online deals are server-selected, not duplicate deals. No blind retries of
state-changing requests: after an ambiguous network failure, stop and preserve
the partial report instead of risking duplicated/omitted actions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import requests
import torch
from alphanl.cfr.gauntlet import Candidate, sha256
from alphanl.cfr.evaluation import mean_interval
from alphanl.transformer.schema import ActionEvent
from eval_cfr_slumbot import api_action, decision_table
from slumbot_client import HOST, State


def public_event(row, seat):
    street, player, kind, before, after = row
    call, chips = max(0, before[1-player] - before[player]), after[player] - before[player]
    action_type = {"f": "fold", "k": "check", "c": "call", "b": "raise" if call else "bet"}[kind]
    if chips > 0 and after[player] == 20000:
        action_type = "all_in"
    return ActionEvent(street=street, actor_seat=player, position="BB" if player == 0 else "SB",
        actor_is_me=player == seat, action_id=None, action_type=action_type,
        pot_before=sum(before), chips_added=chips, pot_fraction=chips / max(1, sum(before)),
        size_bucket=0, to_call_before=call, stack_before=20000-before[player], legal_action_ids=())


def safe_post(session, path, data):
    response = session.post(f"{HOST}/api/{path}", json=data, timeout=(10, 30))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or "error_msg" in result:
        # Server error text can contain session tokens. Never persist it.
        raise RuntimeError("Slumbot API returned an error or invalid object")
    return result


def evaluate_slumbot(candidate, out, *, hands, seed, session=None, delay=.2):
    if hands < 1:
        raise ValueError("hands must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(format="pro_slumbot_200bb_v1", candidate=candidate.metadata,
        requested_hands=hands, completed_hands=0, complete=False, stack_bb=200,
        ci_unit="50_hand_context_block", online_duplicate_deals=False,
        notes=["Full 200BB exact-chip public-history replay; no 50BB cap.",
               "HUD resets every 50 server hands. API seat order is server-selected.",
               "Intervals cluster 50 hands; online Slumbot scores are separate from local duplicate matrix."],
        telemetry=dict(decisions=0, off_tree_opponent_bets=0, opponent_bets=0),
        source="https://slumbot.com/",
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), ROOT / "scripts/eval_cfr_slumbot.py", ROOT / "eval/slumbot_client.py",
             *sorted((ROOT / "src/alphanl").rglob("*.py"))]})
    values, blocks = [], []
    actor, token, started = candidate.actor(seed), None, time.perf_counter()
    own_session = session is None
    session = session or requests.Session()
    def publish():
        report["completed_hands"] = len(values)
        report["wall_seconds"] = time.perf_counter() - started
        if len(blocks) >= 2:
            report["performance"] = {**mean_interval(blocks), "unit": "mbb/hand",
                "completed_blocks": len(blocks), "ci_method": "normal_1.96_sample_std_of_50_hand_block_means"}
        if values:
            report["all_completed_hand_mean_mbb"] = sum(values) / len(values)
        report["audit"] = actor.audit.summary()
        temporary = out / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "report.json")
    publish()
    try:
        with (out / "hands.jsonl").open("x") as log:
            for hand in range(hands):
                if hand % 50 == 0 and hasattr(actor, "begin_match"):
                    actor.begin_match()
                time.sleep(delay)
                response = safe_post(session, "new_hand", {"token": token} if token else {})
                seat = response["client_pos"]
                actor.begin_hand(seat)
                seen, decisions, bets = 0, [], {}
                for step in range(1000):
                    token = response.get("token", token)
                    parsed = State(response["action"])
                    for row in parsed.events[seen:]:
                        if hasattr(actor, "observe_action"):
                            actor.observe_action(public_event(row, seat))
                    seen = len(parsed.events)
                    if "winnings" in response:
                        break
                    table, parsed = decision_table(response, bet_telemetry=bets)
                    action = actor.act(table.snapshot(seat))
                    sent = api_action(table, parsed, action)
                    decisions.append(dict(history=response["action"], action_id=action, sent=sent))
                    report["telemetry"]["decisions"] += 1
                    time.sleep(delay)
                    response = safe_post(session, "act", {"token": token, "incr": sent})
                else:
                    raise RuntimeError("Slumbot hand exceeded action limit")
                winnings = float(response["winnings"])
                if not math.isfinite(winnings) or abs(winnings) > 20000:
                    raise ValueError("invalid Slumbot winnings")
                if hasattr(actor, "finish_hand"):
                    actor.finish_hand(winnings / 100, showdown=not any(r[2] == "f" for r in parsed.events))
                values.append(winnings * 10)
                report["telemetry"]["opponent_bets"] += sum(b["player"] != seat for b in bets.values())
                report["telemetry"]["off_tree_opponent_bets"] += sum(b["player"] != seat and b["off_tree"] for b in bets.values())
                if len(values) % 50 == 0:
                    blocks.append(sum(values[-50:]) / 50)
                # Explicit allowlist excludes token, session metadata and opponent cards.
                log.write(json.dumps(dict(hand=hand, client_pos=seat, winnings_chips=winnings,
                    action=response["action"], hole_cards=response["hole_cards"], board=response["board"],
                    decisions=decisions, snapshot=getattr(getattr(actor, "policy", None), "hand_iteration", None))) + "\n")
                log.flush()
                publish()
                print(f"Slumbot {candidate.name}: {hand+1}/{hands}", flush=True)
        report["complete"] = True
    except Exception as exc:
        # Exception strings can expose tokens/HTTP request contents; store the type only.
        report["error"] = dict(type=type(exc).__name__, reason="Request or replay failed; no automatic retry")
        publish()
        raise
    finally:
        publish()
        if own_session:
            session.close()
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("teacher", "student", "stage2"), required=True)
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--name", default="candidate")
    p.add_argument("--start", type=int, default=150)
    p.add_argument("--end", type=int, default=350)
    p.add_argument("--update", type=int)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hands", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261029)
    args = p.parse_args(argv)
    if args.hands < 1:
        p.error("hands must be positive")
    torch.set_num_threads(1)
    spec = dict(name=args.name, kind=args.kind, path=str(args.path))
    if args.kind == "teacher":
        spec.update(start=args.start, end=args.end)
    if args.update is not None:
        spec["update"] = args.update
    candidate = Candidate(spec)
    evaluate_slumbot(candidate, args.out_dir, hands=args.hands, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

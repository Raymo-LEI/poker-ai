"""Trace frozen policies versus Equity without loading or changing training state.

Terminal-street buckets attribute whole-hand returns to the last betting street,
not counterfactual action EV. Oracle equity uses villain cards only after the
hand for descriptive audit; no actor receives those cards or this RNG.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from alphanl.cfr.evaluation import CFRActor, RuleActor, paired_summary
from alphanl.cfr.strategy import SnapshotPolicy
from alphanl.cfr.study import publish, sha256
from alphanl.transformer.hunl import HunlTable
from alphanl.transformer.tables import standard_deck


def oracle_equity(hole, villain, board, *, seed, samples=128):
    from phevaluator import evaluate_cards
    convert = lambda c: c[1] + c[0].lower()
    known = set(hole + villain + board)
    remaining = [convert(c) for c in standard_deck() if c not in known]
    own, opp, public = [[convert(c) for c in cards] for cards in (hole, villain, board)]
    rng = np.random.default_rng(seed)
    score = 0.
    for _ in range(samples if len(board) < 5 else 1):
        runout = public + [remaining[i] for i in rng.choice(len(remaining), size=5-len(board), replace=False)]
        a, b = evaluate_cards(*(own + runout)), evaluate_cards(*(opp + runout))
        score += (a < b) + .5 * (a == b)
    return float(score / (samples if len(board) < 5 else 1))


def trace_hand(hero, opponent, *, deck_seed, hero_seat):
    table = HunlTable(seed=deck_seed)
    table.reset()
    # This information is stored for post-hand analysis, never passed to actors.
    hole, villain = tuple(table.hole_cards(hero_seat)), tuple(table.hole_cards(1-hero_seat))
    hero.begin_hand(hero_seat)
    opponent.begin_hand(1-hero_seat)
    events = []
    while not table.is_over():
        seat = table.current_player()
        snapshot = table.snapshot(seat)
        actor = hero if seat == hero_seat else opponent
        action = actor.act(snapshot)
        table.step(action)
        event = table.events_for(hero_seat)[-1]
        events.append(dict(seat=seat, street=snapshot.street, action=action,
            pot=snapshot.pot, to_call=snapshot.to_call, chips_added=event.chips_added,
            board=list(snapshot.board_cards)))
        if len(events) > 1000:
            raise RuntimeError("unterminated hand")
    payoff = float(table.payoffs()[hero_seat])
    if sum(table.payoffs()) != 0 or table.chips_in_play() != 40000:
        raise RuntimeError("chip conservation failed")
    last = events[-1]
    aggressions = [e for e in events if e["chips_added"] > e["to_call"]]
    aggressor = aggressions[-1] if aggressions else None
    large_calls = [e for e in events if e["seat"] == hero_seat and e["street"] > 0
                   and e["action"] == 1 and e["to_call"] > 0 and e["to_call"] >= .5 * e["pot"]]
    sb_open_fold = hero_seat == 1 and events[0]["seat"] == hero_seat and events[0]["action"] == 0
    if sb_open_fold:
        category = "sb_open_fold"
    elif last["street"] == 0:
        category = "other_preflop_terminal"
    elif last["action"] == 0:
        category = "hero_postflop_fold" if last["seat"] == hero_seat else "villain_postflop_fold"
    elif aggressor and aggressor["seat"] == hero_seat:
        category = "showdown_hero_last_aggressor"
    elif large_calls:
        category = "showdown_villain_last_aggressor_large_hero_call"
    else:
        category = "other_showdown"
    audit = {}
    if large_calls:
        e = large_calls[-1]
        eq = oracle_equity(hole, villain, tuple(e["board"]), seed=deck_seed ^ 0x91AC)
        audit["last_large_call_oracle_equity"] = eq
        audit["last_large_call_pot_odds"] = min(e["to_call"], e["chips_added"]) / (e["pot"] + min(e["to_call"], e["chips_added"]))
    if aggressor and aggressor["seat"] == hero_seat and aggressor["street"] > 0:
        audit["last_hero_aggression_oracle_equity"] = oracle_equity(hole, villain, tuple(aggressor["board"]), seed=deck_seed ^ 0x2B17)
    return dict(hero_seat=hero_seat, payoff_chips=payoff, snapshot=hero.policy.hand_iteration,
        category=category, terminal_betting_street=last["street"], sb_open_fold=sb_open_fold,
        large_postflop_calls=len(large_calls), events=events, post_hand_oracle_audit=audit)


def summarize(hands):
    buckets = defaultdict(lambda: dict(hands=0, chips=0., gross_loss_chips=0., gross_win_chips=0.))
    tags = defaultdict(lambda: dict(hands=0, chips=0.))
    counts = defaultdict(int)
    for hand in hands:
        seat = "sb" if hand["hero_seat"] == 1 else "bb"
        for key in (f"position:{seat}", f"terminal_street:{hand['terminal_betting_street']}",
                    f"position_terminal_street:{seat}:{hand['terminal_betting_street']}", f"category:{hand['category']}"):
            b = buckets[key]; value = hand["payoff_chips"]
            b["hands"] += 1; b["chips"] += value
            b["gross_loss_chips"] += min(0., value); b["gross_win_chips"] += max(0., value)
        counts["sb_open_folds"] += int(hand["sb_open_fold"])
        counts["large_postflop_calls"] += hand["large_postflop_calls"]
        audit = hand["post_hand_oracle_audit"]
        selected_tags = []
        if hand["category"] == "showdown_hero_last_aggressor":
            aggression = [e for e in hand["events"] if e["chips_added"] > e["to_call"]][-1]
            if aggression["action"] == 4:
                selected_tags.append("showdown_hero_last_aggression_all_in")
                selected_tags.append(f"showdown_hero_last_aggression_all_in_{seat}")
        if hand["payoff_chips"] < 0:
            if audit.get("last_large_call_oracle_equity", 1) < audit.get("last_large_call_pot_odds", 0):
                counts["losing_hands_with_last_large_call_oracle_equity_below_odds"] += 1
            if hand["category"] == "showdown_hero_last_aggressor" and audit.get("last_hero_aggression_oracle_equity", 1) < .35:
                counts["losing_showdown_hero_last_aggressor_oracle_equity_below_35pct"] += 1
                selected_tags.append("losing_showdown_hero_last_aggressor_oracle_equity_below_35pct")
        for key in selected_tags:
            tags[key]["hands"] += 1
            tags[key]["chips"] += hand["payoff_chips"]
    for b in buckets.values():
        b["contribution_mbb_per_all_hands"] = b["chips"] / 100 * 1000 / len(hands)
        b["conditional_mbb_per_hand"] = b["chips"] / 100 * 1000 / b["hands"]
    for tag in tags.values():
        tag["contribution_mbb_per_all_hands"] = tag["chips"] / 100 * 1000 / len(hands)
    return dict(buckets=dict(buckets), overlapping_tags=dict(tags), counts=dict(counts),
        sb_open_fold_rate=counts["sb_open_folds"] / (len(hands) / 2),
        notes=["Category buckets partition all hands; terminal-street buckets are a separate partition.",
               "Large call means a postflop check/call facing >=0.5 times the pot BEFORE calling.",
               "Oracle equity is 128-runout equity versus the actual holding, evaluated after play; it is not range equity or proof of a mistake.",
               "Weak-equity aggression is a descriptive bluff proxy, not an inference of intent. Overlapping audit tags must not be added as loss shares."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategies", type=Path, required=True)
    p.add_argument("--through", type=int, required=True)
    p.add_argument("--selection", choices=("full", "latest", "burnin_quadratic"), default="full")
    p.add_argument("--pairs", type=int, default=2500)
    p.add_argument("--seed", type=int, default=1567594708)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.pairs < 2 or args.seed < 0:
        p.error("requires >=2 pairs and a nonnegative seed")
    torch.set_num_threads(1)
    archive = SnapshotPolicy(args.strategies, through_iteration=args.through)
    if args.selection != "latest" and archive.iterations.tolist() != list(range(1, args.through+1)):
        raise ValueError("full/burn-in evaluation requires every historical snapshot 1..T; history is incomplete")
    selected = ([args.through] if args.selection == "latest" else list(range(
        args.through // 2 + 1 if args.selection == "burnin_quadratic" else 1, args.through+1)))
    hero = CFRActor(args.strategies, seed=args.seed+11, through=args.through, iterations=selected, cache_size=len(selected))
    hero.policy.sampling_power = 2. if args.selection == "burnin_quadratic" else 1.
    opponent = RuleActor("equity", seed=args.seed+19, equity_samples=64)
    manifest = {path.name: sha256(path) for path in hero.policy.paths}
    args.out.mkdir(parents=True, exist_ok=False)
    hands, pairs = [], []
    rng = np.random.default_rng(args.seed)
    with (args.out/"hands.jsonl").open("x") as raw, (args.out/"pairs.jsonl").open("x") as pair_log:
        for i in range(args.pairs):
            seed = int(rng.integers(0, 2**63-1))
            pair = []
            for seat in (1, 0):
                hand = trace_hand(hero, opponent, deck_seed=seed, hero_seat=seat)
                hand.update(pair=i, deck_seed=seed)
                raw.write(json.dumps(hand, allow_nan=False)+"\n")
                hands.append(hand); pair.append(hand["payoff_chips"])
            pairs.append(pair)
            pair_log.write(json.dumps(dict(pair=i, deck_seed=seed, chips_sb_bb=pair))+"\n")
            if (i+1) % 250 == 0:
                print(f"{args.selection} {i+1}/{args.pairs}", flush=True)
    if any(sha256(path) != manifest[path.name] for path in hero.policy.paths):
        raise RuntimeError("archive changed during diagnosis")
    report = dict(format="cfr_equity_loss_trace_v1", selection=args.selection,
        selected_iterations=selected, sampling_power=hero.policy.sampling_power,
        seed=args.seed, snapshot_sha256=manifest, config=hero.policy.config.to_dict(),
        summary=paired_summary(pairs), breakdown=summarize(hands), hero_audit=hero.audit.summary(),
        hands_sha256=sha256(args.out/"hands.jsonl"), pairs_sha256=sha256(args.out/"pairs.jsonl"),
        source_sha256={str(q.relative_to(ROOT)):sha256(q) for q in [Path(__file__),*sorted((ROOT/"src/alphanl").rglob("*.py"))]})
    publish(args.out/"report.json", report)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()

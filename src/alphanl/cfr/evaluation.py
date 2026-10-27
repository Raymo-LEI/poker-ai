"""Reproducible, seat-swapped evaluation on the locked standard HUNL game.

The unit of uncertainty is a deck pair, never either correlated hand.
Rule opponents see only HandSnapshot (own hole cards and public history).
These diagnostics are not a measurement of Nash exploitability.
"""
from __future__ import annotations

from collections import Counter
import math
import time

import numpy as np
import torch

from .model import regret_matching
from .strategy import SnapshotPolicy
from alphanl.transformer.hunl import BIG_BLIND, HunlTable, HunlTokenizer
from alphanl.transformer.tables import standard_deck


OPPONENTS = ("random", "call", "maniac", "overbet", "nit", "equity")


def mean_interval(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected a nonempty finite vector")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    margin = 1.96 * std / math.sqrt(len(values)) if std is not None else None
    return {"n": len(values), "mean": mean, "sample_std": std,
            "ci95": [mean - margin, mean + margin] if margin is not None else None}


def paired_summary(chip_pairs) -> dict:
    pairs = np.asarray(chip_pairs, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("expected [pairs, 2] chip payoffs")
    result = mean_interval(pairs.mean(axis=1) / BIG_BLIND * 1000)
    return {"pairs": result.pop("n"), "hands": len(pairs) * 2,
            "mbb_per_hand": result.pop("mean"), **result,
            "ci_unit": "mbb/hand", "ci_method": "normal_1.96_sample_std_of_pair_means",
            "sb_mbb_per_hand": float(pairs[:, 0].mean() / BIG_BLIND * 1000),
            "bb_mbb_per_hand": float(pairs[:, 1].mean() / BIG_BLIND * 1000)}


def wilson(wins: int, total: int) -> list[float]:
    p, z = wins / total, 1.96
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - margin), min(1., center + margin)]


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.actions = Counter()
        self.entropy_sum = 0.
        self.latency_ms = []
        self.snapshot_counts = Counter()
        self.seat_actions = Counter()
        self.seat_counts = Counter()
        self.positive_regret_sum = 0
        self.positive_regret_decisions = 0
        self._hand = None

    def begin_hand(self, seat):
        """Count dealt hands, including BB walks; reset once-per-hand flags."""
        self.counts[f"hands_{'bb' if seat == 0 else 'sb'}"] += 1
        self._hand = {"seat": seat, "vpip": False, "pfr": False, "threebet": False}

    def observe_guard(self, snapshot, guard, action):
        """Record the intervention and residual early deep shoves by street."""
        prefix = f"spr_{snapshot.me_seat}_{snapshot.street}"
        self.counts[prefix + "_decisions"] += 1
        blocked = guard.blocks_all_in(snapshot)
        self.counts[prefix + "_blocked"] += int(blocked)
        high = guard.all_in_pot_ratio(snapshot) > guard.max_pot_ratio
        early = guard.street_raises(snapshot) < guard.unlock_raises
        self.counts[prefix + "_high_spr_shoves"] += int(action == 4 and high)
        self.counts[prefix + "_early_high_spr_shoves"] += int(action == 4 and high and early)

    @staticmethod
    def _raises(snapshot, action):
        # A short-stack all-in call is not a raise.
        return action in (2, 3) or (action == 4 and snapshot.me_stack > snapshot.to_call)

    def observe(self, snapshot, probs, action, elapsed_ms, *, regrets=None):
        c = self.counts
        c["decisions"] += 1
        self.latency_ms.append(elapsed_ms)
        self.actions[f"{snapshot.street}:{action}"] += 1
        self.seat_actions[f"{snapshot.me_seat}:{snapshot.street}:{action}"] += 1
        if snapshot.street == 0:
            self.seat_counts[f"{snapshot.me_seat}:preflop_decisions"] += 1
            self.seat_counts[f"{snapshot.me_seat}:preflop_fold"] += int(action == 0)
            self.seat_counts[f"{snapshot.me_seat}:preflop_all_in"] += int(action == 4)
        legal = snapshot.legal_action_ids
        if action not in legal:
            c["illegal_selections"] += 1
            raise ValueError("policy selected an illegal action")
        illegal = [i for i in range(5) if i not in legal]
        if any(probs[i] != 0 for i in illegal):
            c["mask_leaks"] += 1
            raise ValueError("nonzero illegal probability")
        if regrets is not None:
            self.positive_regret_sum += int(np.sum(np.asarray(regrets)[list(legal)] > 0))
            self.positive_regret_decisions += 1
        if self._hand is not None and snapshot.street == 0:
            position = "sb" if snapshot.me_seat == snapshot.sb_seat else "bb"
            voluntary = action in (2, 3, 4) or (action == 1 and snapshot.to_call > 0)
            if voluntary and not self._hand["vpip"]:
                c[f"vpip_hands_{position}"] += 1
                self._hand["vpip"] = True
            if self._raises(snapshot, action) and not self._hand["pfr"]:
                c[f"pfr_hands_{position}"] += 1
                self._hand["pfr"] = True
            raises = [e for e in snapshot.events if e.street == 0
                      and e.action_type in ("bet", "raise", "all_in")
                      and e.chips_added > e.to_call_before]
            can_raise = any(self._raises(snapshot, a) for a in legal)
            if (position == "bb" and snapshot.to_call > 0 and len(raises) == 1
                    and raises[0].actor_seat == snapshot.sb_seat and can_raise
                    and not self._hand["threebet"]):
                c["bb_3bet_opportunities"] += 1
                c["bb_3bets"] += int(self._raises(snapshot, action))
                self._hand["threebet"] = True
        if len(legal) > 1:
            c["nonforced_decisions"] += 1
            positive = probs[probs > 0]
            self.entropy_sum -= float(np.sum(positive * np.log(positive)))
            c["near_deterministic"] += int(float(probs.max()) >= .999)
            if snapshot.to_call == 0 and 0 in legal and 1 in legal:
                c["free_check_opportunities"] += 1
                c["free_check_folds"] += int(action == 0)
        if snapshot.street == 0 and snapshot.me_seat == snapshot.sb_seat and not snapshot.events:
            c["sb_open_opportunities"] += 1
            c["sb_vpip"] += int(action != 0)
            c["sb_open_shoves"] += int(action == 4)
        if snapshot.street == 0 and snapshot.me_seat != snapshot.sb_seat:
            # Count the first BB response to an SB raise, excluding limp pots.
            history = [e for e in snapshot.events if e.street == 0]
            if (len(history) == 1 and history[0].actor_seat == snapshot.sb_seat
                    and history[0].action_type in ("raise", "bet", "all_in")):
                c["bb_vs_open_raise"] += 1
                c["bb_fold_vs_open_raise"] += int(action == 0)

    def summary(self):
        c = self.counts
        def ratio(numerator, denominator):
            return c[numerator] / c[denominator] if c[denominator] else None
        counts = {**{key: 0 for key in ("illegal_selections", "mask_leaks", "nonfinite_regrets")}, **dict(c)}
        return {"counts": counts, "action_counts_by_street": dict(self.actions),
                "action_counts_by_seat_street": dict(self.seat_actions),
                "counts_by_seat": dict(self.seat_counts),
                "positive_regret_actions_sum": self.positive_regret_sum,
                "positive_regret_decisions": self.positive_regret_decisions,
                "positive_regret_actions_mean": self.positive_regret_sum / self.positive_regret_decisions
                if self.positive_regret_decisions else None,
                "vpip_bb": ratio("vpip_hands_bb", "hands_bb"),
                "vpip_sb": ratio("vpip_hands_sb", "hands_sb"),
                "pfr_sb": ratio("pfr_hands_sb", "hands_sb"),
                "threebet_bb": ratio("bb_3bets", "bb_3bet_opportunities"),
                "free_check_fold_ratio": ratio("free_check_folds", "free_check_opportunities"),
                "sampled_snapshot_counts": dict(self.snapshot_counts),
                "sb_vpip": ratio("sb_vpip", "sb_open_opportunities"),
                "sb_open_shove": ratio("sb_open_shoves", "sb_open_opportunities"),
                "bb_fold_to_open_raise": ratio("bb_fold_vs_open_raise", "bb_vs_open_raise"),
                "mean_nonforced_entropy_nats": self.entropy_sum / c["nonforced_decisions"]
                if c["nonforced_decisions"] else None,
                "near_deterministic_fraction": ratio("near_deterministic", "nonforced_decisions"),
                "decision_p99_ms": float(np.quantile(self.latency_ms, .99)) if self.latency_ms else None,
                "decision_p50_ms": float(np.median(self.latency_ms)) if self.latency_ms else None}


class CFRActor:
    def __init__(self, directory, *, seed, through=None, iterations=None, cache_size=64,
                 spr_guard=None, max_pot_ratio=None):
        self.policy = SnapshotPolicy(directory, seed=seed, device="cpu",
                                     through_iteration=through, snapshot_iterations=iterations,
                                     cache_size=cache_size)
        self.audit = Audit()
        self.tokenizer = None
        if spr_guard is not None and not isinstance(spr_guard, bool):
            raise ValueError("spr_guard must be boolean or None")
        if max_pot_ratio is not None and (not math.isfinite(max_pot_ratio) or max_pot_ratio <= 0):
            raise ValueError("max_pot_ratio must be positive and finite")
        self.spr_guard_override = spr_guard
        self.max_pot_ratio_override = max_pot_ratio

    def begin_hand(self, seat):
        self.audit.begin_hand(seat)
        iteration = self.policy.begin_hand(seat)
        self.audit.snapshot_counts[iteration] += 1
        self.tokenizer = HunlTokenizer(self.policy.config.model.max_seq_len,
                                       position_conditioning=self.policy.config.position_conditioning)
        from .actions import CFRActionGuard
        self.action_guard = CFRActionGuard.from_config(self.policy.config)
        from dataclasses import replace
        overrides = {}
        if self.spr_guard_override is not None:
            overrides["spr_enabled"] = self.spr_guard_override
        if self.max_pot_ratio_override is not None:
            overrides["max_pot_ratio"] = self.max_pot_ratio_override
        self.action_guard = replace(self.action_guard, **overrides)

    @torch.inference_mode()
    def act(self, snapshot):
        started = time.perf_counter()
        original = snapshot
        snapshot = self.action_guard.apply(snapshot)
        batch = self.tokenizer.encode(snapshot)
        raw = self.policy.model(batch)
        if not torch.isfinite(raw).all():
            self.audit.counts["nonfinite_regrets"] += 1
            raise ValueError("NaN/Inf regret prediction")
        probs = regret_matching(raw, batch["legal_mask"], fallback=self.policy.config.fallback)[0].double().numpy()
        action = int(self.policy.rng.choice(5, p=probs / probs.sum()))
        self.audit.observe(snapshot, probs, action, (time.perf_counter() - started) * 1000,
                           regrets=raw[0].double().numpy())
        self.audit.observe_guard(original, self.action_guard, action)
        return action


def random_equity(snapshot, rng, samples):
    """Showdown equity versus uniform unknown cards, without reading the table."""
    from phevaluator import evaluate_cards
    known = set(snapshot.hole_cards + snapshot.board_cards)
    remaining = [c for c in standard_deck() if c not in known]
    convert = lambda c: c[1] + c[0].lower()
    hole = [convert(c) for c in snapshot.hole_cards]
    board = [convert(c) for c in snapshot.board_cards]
    remaining = [convert(c) for c in remaining]
    need = 5 - len(board)
    score = 0.
    for _ in range(samples):
        draw = rng.choice(len(remaining), size=need + 2, replace=False)
        runout = board + [remaining[i] for i in draw[:need]]
        own = evaluate_cards(*(hole + runout))
        opp = evaluate_cards(*([remaining[i] for i in draw[need:]] + runout))
        score += float(own < opp) + .5 * float(own == opp)
    return score / samples


class RuleActor:
    def __init__(self, name, *, seed, equity_samples=64):
        if name not in OPPONENTS or equity_samples < 1:
            raise ValueError("invalid rule opponent or equity sample count")
        self.name, self.rng, self.equity_samples = name, np.random.default_rng(seed), equity_samples

    def begin_hand(self, seat):
        pass

    def act(self, snapshot):
        legal = snapshot.legal_action_ids
        prefer = lambda choices: next(a for a in choices if a in legal)
        if self.name == "random":
            return int(self.rng.choice(legal))
        if self.name == "call":
            return prefer((1, 0, 4, 3, 2))
        if self.name == "maniac":
            return prefer((4, 3, 2, 1, 0))
        if self.name == "overbet":
            return prefer((3, 4, 2, 1, 0))
        equity = random_equity(snapshot, self.rng, self.equity_samples)
        odds = snapshot.to_call / max(1., snapshot.pot + snapshot.to_call)
        if self.name == "nit":
            if snapshot.to_call > 0 and equity < max(.62, odds + .12):
                return prefer((0, 1, 4, 3, 2))
            return prefer((1, 0, 4, 3, 2))
        if snapshot.to_call > 0 and equity < odds + .03:
            return prefer((0, 1, 4, 3, 2))
        if equity > .8:
            return prefer((3, 2, 4, 1, 0))
        if equity > .65:
            return prefer((2, 3, 1, 4, 0))
        return prefer((1, 0, 2, 3, 4))


def play_hand(hero, opponent, *, deck_seed, hero_seat):
    table = HunlTable(seed=int(deck_seed))
    table.reset()
    hero.begin_hand(hero_seat)
    opponent.begin_hand(1 - hero_seat)
    steps = 0
    while not table.is_over():
        seat = table.current_player()
        actor = hero if seat == hero_seat else opponent
        action = actor.act(table.snapshot(seat))
        if action not in table.legal_action_ids():
            raise ValueError(f"illegal action {action}")
        table.step(action)
        steps += 1
        if steps > 1000:
            raise RuntimeError("hand failed to terminate")
    payoff = table.payoffs()
    if sum(payoff) != 0 or table.chips_in_play() != 40000:
        raise RuntimeError("chip conservation failed")
    return float(payoff[hero_seat])


def duplicate_match(hero, opponent, *, pairs, seed, log, progress=None):
    if pairs < 2:
        raise ValueError("duplicate match requires at least two pairs")
    rng = np.random.default_rng(seed)
    chips = []
    started = time.perf_counter()
    for index in range(pairs):
        deck_seed = int(rng.integers(0, 2**63 - 1))
        # Same seat-specific cards and full deck in both legs; identities swap.
        pair, selected = [], []
        for seat in (1, 0):
            pair.append(play_hand(hero, opponent, deck_seed=deck_seed, hero_seat=seat))
            selected.append(hero.policy.hand_iteration if hasattr(hero, "policy") else None)
        chips.append(pair)
        log({"pair": index, "deck_seed": deck_seed, "chips_sb_bb": pair,
             "hero_snapshot_sb_bb": selected})
        if progress and (index + 1) % 250 == 0:
            progress(index + 1, pairs)
    return {**paired_summary(chips), "wall_seconds": time.perf_counter() - started,
            "hero_audit": hero.audit.summary()}


def independent_matches(hero, opponent, *, matches, seed, log, progress=None):
    if matches < 1:
        raise ValueError("matches must be positive")
    rng = np.random.default_rng(seed)
    totals, cpu_seconds = [], []
    for match in range(matches):
        # No duplicated decks in these independent 50-hand match blocks.
        seeds = rng.integers(0, 2**63 - 1, size=50)
        start_seat = int(rng.integers(0, 2))
        started = time.process_time()
        chips = [play_hand(hero, opponent, deck_seed=int(s), hero_seat=(h + start_seat) % 2)
                 for h, s in enumerate(seeds)]
        elapsed = time.process_time() - started
        totals.append(sum(chips))
        cpu_seconds.append(elapsed)
        log({"match": match, "start_seat": start_seat, "deck_seeds": seeds.tolist(),
             "chips": chips, "cpu_seconds_including_opponent_and_referee": elapsed})
        if progress and (match + 1) % 25 == 0:
            progress(match + 1, matches)
    totals = np.array(totals)
    wins = int(np.sum(totals > 0))
    interval = mean_interval(totals / (50 * BIG_BLIND) * 1000)
    return {"matches": matches, "hands": matches * 50, "wins": wins,
            "ties": int(np.sum(totals == 0)), "win_fraction": wins / matches,
            "win_fraction_ci95_wilson": wilson(wins, matches),
            "mbb_per_hand": interval["mean"], "mbb_per_hand_ci95_match_clusters": interval["ci95"],
            "match_cpu_p99_seconds_including_opponent_and_referee": float(np.quantile(cpu_seconds, .99)),
            "hero_audit": hero.audit.summary()}


def health_check(hero, opponent, *, hands, seed, log, progress=None):
    if hands < 1:
        raise ValueError("health hands must be positive")
    rng = np.random.default_rng(seed)
    for hand in range(hands):
        deck_seed = int(rng.integers(0, 2**63 - 1))
        result = play_hand(hero, opponent, deck_seed=deck_seed, hero_seat=hand % 2)
        log({"hand": hand, "deck_seed": deck_seed, "hero_seat": hand % 2, "chips": result})
        if progress and (hand + 1) % 500 == 0:
            progress(hand + 1, hands)
    return {"hands": hands, "hero": hero.audit.summary(), "opponent": opponent.audit.summary()}


def launch_gates(report):
    """Missing evidence never passes. Thresholds are experimental user targets."""
    results = report.get("results", {})
    target = report["hero_iteration"]
    full = target == 1000
    def gate(ready, ok, reason):
        return {"status": "PASS" if ready and ok else "FAIL" if ready else "INCOMPLETE", "reason": reason}
    improve = results.get("duplicate:1000_vs_100", {})
    ci = improve.get("ci95")
    gates = {"self_improvement": gate(full and improve.get("pairs", 0) >= 2500,
                  improve.get("mbb_per_hand", -math.inf) >= 35 and ci is not None and ci[0] > 0,
                  "T=1000 vs T=100; >=2500 pairs, mean>=35, CI lower>0 mbb/hand")}
    gauntlet = {}
    for name, threshold, win_fraction in (("random", 1500, .95), ("call", 250, .75),
                                         ("nit", 250, .75), ("maniac", 400, None), ("equity", 80, .6)):
        d = results.get(f"duplicate:{target}_vs_{name}", {})
        m = results.get(f"matches:{target}_vs_{name}", {})
        ready = d.get("pairs", 0) >= 2500 and (win_fraction is None or m.get("matches", 0) >= 500)
        ok = d.get("mbb_per_hand", -math.inf) > threshold and (win_fraction is None or m.get("win_fraction", 0) > win_fraction)
        gauntlet[name] = gate(ready, ok, f">{threshold} mbb/hand; MatchWinRate@50>{win_fraction}" if win_fraction else f">{threshold} mbb/hand")
    states = [v["status"] for v in gauntlet.values()]
    gates["gauntlet"] = {"status": "FAIL" if "FAIL" in states else "INCOMPLETE" if "INCOMPLETE" in states else "PASS", "opponents": gauntlet}
    gates["repo_baseline"] = gate(False, False, "Requires a rule-compatible PPO/conv baseline and paired CI lower>0")
    gates["slumbot"] = gate(False, False, "Requires >=5000 external 200BB hands, CI and off-tree telemetry")
    health = results.get("health", {})
    counts = [health.get(actor, {}).get("counts", {}) for actor in ("hero", "opponent")]
    health_fields = ("illegal_selections", "mask_leaks", "nonfinite_regrets",
                     "sb_open_opportunities", "sb_open_shoves")
    complete = all(all(key in c for key in health_fields) for c in counts)
    safe = not any(c.get(k, 0) for c in counts for k in health_fields[:3])
    opportunities = sum(c.get("sb_open_opportunities", 0) for c in counts)
    shove = sum(c.get("sb_open_shoves", 0) for c in counts) / opportunities if opportunities else None
    gates["behavior"] = gate(complete and health.get("hands", 0) >= 10000 and shove is not None,
                             safe and shove is not None and shove < .015,
                             "10000 self-play hands; SB open shove<1.5%; zero illegal/NaN/Inf")
    gates["sandbox"] = gate(False, False, "Evaluation PyTorch process is not a Botzone deployment artifact; packaged CPU sandbox still required")
    overall = "NO_GO" if report.get("error") or any(g["status"] == "FAIL" for g in gates.values()) else "INCOMPLETE" if any(g["status"] == "INCOMPLETE" for g in gates.values()) else "GO"
    return {"verdict": overall, "criteria": gates}

"""Independent frozen-policy tournament with symmetric public match context."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .actions import CFRActionGuard
from .distillation import load_student
from .evaluation import Audit, CFRActor, OPPONENTS, RuleActor, mean_interval
from .pro_bots import PRO_OPPONENTS, ProScriptActor
from alphanl.stage2.context import MatchCtx
from alphanl.stage2.evaluation import summarize_match_pairs
from alphanl.stage2.train import load_checkpoint
from alphanl.transformer.hunl import BIG_BLIND, HunlTable, HunlTokenizer

GAUNTLET_OPPONENTS = OPPONENTS + PRO_OPPONENTS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelActor:
    """Actor-local RNG and HUD; the frozen module can be shared across tasks."""
    def __init__(self, model, *, seed, residual=False):
        self.model, self.residual = model, residual
        self.rng, self.audit = np.random.default_rng(seed), Audit()
        self.guard = CFRActionGuard.from_config(model.config)
        self.tokenizer = HunlTokenizer(model.config.model.max_seq_len, **model.config.tokenizer_kwargs())
        self.begin_match()

    def begin_match(self):
        self.ctx = MatchCtx()

    def begin_hand(self, seat):
        self.audit.begin_hand(seat)
        self.ctx.begin_hand()

    def observe_action(self, event):
        self.ctx.observe_action(event)

    def finish_hand(self, net_bb, *, showdown):
        self.ctx.finish_hand(net_bb, showdown=showdown)

    @torch.inference_mode()
    def act(self, snapshot):
        began, original = time.perf_counter(), snapshot
        snapshot = self.guard.apply(snapshot)
        batch = self.tokenizer.encode(snapshot)
        if self.residual:
            batch["match_ctx"] = self.ctx.tensor()
        batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
        probs = (self.model(batch)[0].softmax(-1) if self.residual else self.model.strategy(batch))[0].cpu().double().numpy()
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            raise FloatingPointError("nonfinite/empty policy distribution")
        action = int(self.rng.choice(5, p=probs / probs.sum()))
        self.audit.observe(snapshot, probs, action, (time.perf_counter() - began) * 1000)
        self.audit.observe_guard(original, self.guard, action)
        return action


class Candidate:
    def __init__(self, spec, *, device="cpu", cache_size=16):
        self.spec, self.name = dict(spec), spec["name"]
        self.kind, self.path = spec["kind"], Path(spec["path"]).resolve()
        self.model = None
        self.metadata = dict(spec, path=str(self.path))
        if self.kind == "teacher":
            start, end = spec["start"], spec["end"]
            if not 1 <= start <= end:
                raise ValueError("teacher requires a positive contiguous iteration window")
            self.iterations = list(range(start, end + 1))
            # Construction validates that every requested snapshot exists.
            actor = CFRActor(self.path, seed=1, iterations=self.iterations, cache_size=cache_size)
            actor.begin_hand(1)
            self.metadata["snapshot_sha256"] = {p.name: sha256(p) for p in actor.policy.paths}
            self.config = actor.policy.config
            self.cache_size = cache_size
        elif self.kind in ("student", "stage2"):
            self.metadata["sha256"] = sha256(self.path)
            self.model, blob = (load_student(self.path, device=device) if self.kind == "student"
                                else load_checkpoint(self.path, device=device))
            if self.kind == "stage2":
                self.metadata["update"] = blob["update"]
                if "update" in spec and spec["update"] != blob["update"]:
                    raise ValueError(f"{self.name}: requested update {spec['update']}, got {blob['update']}")
            self.config = self.model.config
            self.model.requires_grad_(False)
        else:
            raise ValueError("kind must be teacher, student or stage2; legacy models need an explicit compatible format")
        self.metadata["numeric_version"] = self.config.numeric_version
        self.metadata["guards"] = {k: v for k, v in self.config.to_dict().items() if "guard" in k}

    def actor(self, seed):
        if self.kind == "teacher":
            return CFRActor(self.path, seed=seed, iterations=self.iterations, cache_size=self.cache_size)
        return ModelActor(self.model, seed=seed, residual=self.kind == "stage2")


def rule_actor(name, seed, equity_samples):
    return (ProScriptActor if name in PRO_OPPONENTS else RuleActor)(name, seed=seed, equity_samples=equity_samples)


def play_block(left, right, deck_seeds, first_left_seat):
    """Exactly 50 independent hands; context sees both players' public events."""
    if len(deck_seeds) != 50:
        raise ValueError("a context block must contain 50 hands")
    for actor in (left, right):
        if hasattr(actor, "begin_match"):
            actor.begin_match()
    total, hands = 0., []
    for h, deck_seed in enumerate(deck_seeds):
        seat = first_left_seat ^ (h % 2)
        actors = {seat: left, 1 - seat: right}
        table = HunlTable(seed=int(deck_seed))
        table.reset()
        for player, actor in actors.items():
            actor.begin_hand(player)
        steps = 0
        while not table.is_over():
            player = table.current_player()
            action = actors[player].act(table.snapshot(player))
            if action not in table.legal_action_ids():
                raise ValueError("illegal tournament action")
            table.step(action)
            for p, actor in actors.items():
                if hasattr(actor, "observe_action"):
                    actor.observe_action(table.events_for(p)[-1])
            steps += 1
            if steps > 1000:
                raise RuntimeError("hand action limit exceeded")
        payoffs = table.payoffs()
        if sum(payoffs) != 0 or table.chips_in_play() != 40000:
            raise RuntimeError("chip conservation failed")
        net = float(payoffs[seat]) / BIG_BLIND
        showdown = not any(e.action_type == "fold" for e in table.events_for(seat))
        for p, actor in actors.items():
            if hasattr(actor, "finish_hand"):
                actor.finish_hand(float(payoffs[p]) / BIG_BLIND, showdown=showdown)
        total += net
        hands.append(dict(hand=h, left_seat=seat, net_bb=net,
                          left_snapshot=getattr(getattr(left, "policy", None), "hand_iteration", None),
                          right_snapshot=getattr(getattr(right, "policy", None), "hand_iteration", None)))
    return total, hands


def match_factories(left, right, *, schedule, seed, log, progress=None):
    totals = []
    started = time.perf_counter()
    for pair, decks in enumerate(schedule):
        legs = []
        for first_seat in (1, 0):
            # Factories reset RNG at each leg. Seeds are identical across candidates.
            a, b = left(seed + pair * 4 + first_seat), right(seed + pair * 4 + first_seat + 2)
            total, hands = play_block(a, b, decks, first_seat)
            legs.append(total)
            log(dict(pair=pair, first_left_seat=first_seat, deck_seeds=list(map(int, decks)),
                     net_bb=total, hands=hands,
                     left_audit=a.audit.summary() if hasattr(a, "audit") else dict(getattr(a, "counts", {})),
                     right_audit=b.audit.summary() if hasattr(b, "audit") else dict(getattr(b, "counts", {}))))
        totals.append(legs)
        if progress:
            progress(pair + 1, len(schedule))
    return {**summarize_match_pairs(totals), "net_bb_pairs": totals,
            "wall_seconds": time.perf_counter() - started}


def paired_candidate_gain(left, right):
    """Common-deal candidate comparison, clustered by 50-deal match pair."""
    diff = (np.asarray(left["net_bb_pairs"]) - np.asarray(right["net_bb_pairs"])).mean(axis=1) / 50 * 1000
    summary = mean_interval(diff)
    return dict(mbb_per_hand=summary["mean"], ci95=summary["ci95"], match_pairs=summary["n"])


def select_candidate(matrix, names, *, full_budget, all_required_present):
    """Equal-opponent ranks prevent huge wins against maniacs dominating selection.

    A recommendation is descriptive. A champion additionally needs simultaneous
    nonnegative lower bounds against all scripts and positive direct round-robin
    lower bounds. These normal intervals are approximate, not exploitability.
    """
    from statistics import NormalDist
    rows = []
    comparisons = max(1, len(names) * (len(GAUNTLET_OPPONENTS) + len(names) - 1))
    z = NormalDist().inv_cdf(1 - .05 / (2 * comparisons))
    for name in names:
        rank_scores = []
        for opponent in GAUNTLET_OPPONENTS:
            value = matrix[name][opponent]["mbb_per_hand"]
            others = [matrix[n][opponent]["mbb_per_hand"] for n in names if n != name]
            rank_scores.append(sum(float(value > x) + .5 * float(value == x) for x in others) / max(1, len(others)))
        # Round-robin is one equally weighted axis, not N extra votes.
        rr = [matrix[name][n]["mbb_per_hand"] for n in names if n != name]
        if rr:
            rank_scores.append(sum(float(v > 0) + .5 * float(v == 0) for v in rr) / len(rr))
        certified = []
        for opponent, cell in matrix[name].items():
            values = np.asarray(cell["net_bb_pairs"]).mean(axis=1) / 50 * 1000
            margin = z * float(values.std(ddof=1)) / len(values) ** .5
            lower = float(values.mean()) - margin
            certified.append(lower > 0 if opponent in names else lower >= 0)
        rows.append(dict(name=name, mean_rank_score=float(np.mean(rank_scores)),
                         weakest_three_rank_score=float(np.mean(sorted(rank_scores)[:3])),
                         uniform_script_mbb=float(np.mean([matrix[name][o]["mbb_per_hand"] for o in GAUNTLET_OPPONENTS])),
                         all_simultaneous_bounds_pass=all(certified)))
    rows.sort(key=lambda r: (-r["weakest_three_rank_score"], -r["mean_rank_score"], r["name"]))
    eligible = full_budget and all_required_present and len(names) >= 2
    champions = [r["name"] for r in rows if r["all_simultaneous_bounds_pass"]] if eligible else []
    return dict(recommendation=rows[0]["name"] if eligible else None,
                champion=champions[0] if len(champions) == 1 else None, ranking=rows,
                status="complete" if eligible else "smoke_or_missing_candidates",
                familywise_alpha=.05, approximate_bonferroni_z=z,
                method="weakest-three equal-opponent rank scores; mean ranks tie-break; round-robin one axis",
                caveat="Selection-set result only. Confirm on a fresh seed before deployment; no ladder-strength or Nash claim.")

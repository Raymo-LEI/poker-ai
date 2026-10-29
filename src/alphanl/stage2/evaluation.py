"""Held-out duplicate 50-hand matches, public HUDs and common-deal gains.

2,500 duplicate deck pairs = 50 paired matches = 100 complete matches,
5,000 played hands per policy/opponent. Confidence units are match pairs.
"""
import copy
import numpy as np
import torch
from alphanl.cfr.evaluation import RuleActor, mean_interval
from alphanl.transformer.tables import standard_deck
from .env import BotzoneMatchEnv
from .opponents import NeuralMatchActor
from .policy import ResidualExploiterPolicy

EVAL_OPPONENTS = ("equity", "nit", "call", "maniac", "overbet", "reference")


@torch.inference_mode()
def play_match(model, opponent, *, decks, first_hero_seat, hand_trace=None):
    env = BotzoneMatchEnv(model.config, opponent, decks=decks, first_hero_seat=first_hero_seat, hand_trace=hand_trace)
    obs = env.reset()
    device = next(model.parameters()).device
    while not env.done:
        batch = {k: v.to(device) for k, v in obs.items()}
        action, *_ = model.act(batch)
        obs, _, _, _ = env.step(int(action))
    return env.ctx.hero_net_bb


def summarize_match_pairs(totals):
    values = np.asarray(totals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected finite [match pairs, 2] net BB totals")
    rate = mean_interval(values.mean(axis=1) / 50 * 1000)
    wins = mean_interval((values > 0).mean(axis=1))
    return {"mbb_per_hand": rate["mean"], "mbb_ci95": rate["ci95"],
            "match_win": wins["mean"], "match_win_ci95": wins["ci95"],
            "ties": int((values == 0).sum()), "matches": int(values.size), "match_pairs": len(values),
            "deck_pairs": len(values) * 50, "hands": int(values.size) * 50}


class MatchEvaluator:
    def __init__(self, model, *, match_pairs=50, seed=92017, equity_samples=64):
        if isinstance(match_pairs, bool) or not isinstance(match_pairs, int) or match_pairs < 2:
            raise ValueError("evaluation needs at least two match pairs")
        if equity_samples < 1:
            raise ValueError("invalid evaluation equity sample count")
        self.match_pairs, self.seed, self.equity_samples = match_pairs, seed, equity_samples
        with torch.random.fork_rng(devices=[]):
            self.baseline = ResidualExploiterPolicy(model.reference, **model.settings).to(next(model.parameters()).device).eval()
        self.baseline.requires_grad_(False)
        # Same held-out deals in candidate/base comparisons and across updates.
        rng = np.random.default_rng(seed)
        self.decks = [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(match_pairs)]
        self.baseline_totals = {}

    def _totals(self, model, name, *, trace_writer=None, trace_match_pairs=0):
        totals = []
        for pair, decks in enumerate(self.decks):
            legs = []
            for first_seat in (1, 0):
                seed = self.seed + pair * 2 + first_seat
                torch.manual_seed(seed)
                opponent = (NeuralMatchActor(self.baseline.reference, kind="reference", snapshot_id="reference", seed=seed)
                            if name == "reference" else RuleActor(name, seed=seed, equity_samples=self.equity_samples))
                sink = trace_writer.sink(opponent=name, pair=pair, first_hero_seat=first_seat,
                    policy='candidate') if trace_writer is not None and pair < trace_match_pairs else None
                legs.append(play_match(model, opponent, decks=decks, first_hero_seat=first_seat, hand_trace=sink))
            totals.append(legs)
        return totals

    def evaluate(self, model, *, update, progress=None, trace_writer=None, trace_match_pairs=0):
        was_training = model.training
        devices = [next(model.parameters()).device.index or 0] if next(model.parameters()).is_cuda else []
        report = {"format": "stage2_duplicate_matches_v1", "update": update, "seed": self.seed,
                  "equity_samples": self.equity_samples, "jitter": False,
                  "ci_unit": "paired_50_hand_match", "win_definition": "strictly_positive_net_bb; ties are not wins",
                  "opponents": {}}
        try:
            model.eval()
            # Evaluation must not consume the training action RNG stream.
            with torch.random.fork_rng(devices=devices):
                for name in EVAL_OPPONENTS:
                    if progress:
                        progress(name)
                    if name not in self.baseline_totals:
                        self.baseline_totals[name] = self._totals(self.baseline, name)
                    candidate = self._totals(model, name, trace_writer=trace_writer, trace_match_pairs=trace_match_pairs)
                    baseline = self.baseline_totals[name]
                    gain = mean_interval((np.asarray(candidate) - np.asarray(baseline)).mean(axis=1) / 50 * 1000)
                    report["opponents"][name] = {"candidate": summarize_match_pairs(candidate),
                        "baseline": summarize_match_pairs(baseline), "gain_mbb_per_hand": gain["mean"],
                        "gain_ci95": gain["ci95"], "candidate_net_bb_pairs": candidate,
                        "baseline_net_bb_pairs": copy.deepcopy(baseline)}
        finally:
            model.train(was_training)
        report["retention_pass"] = report["opponents"]["reference"]["candidate"]["mbb_per_hand"] >= -20
        report["equity_positive"] = report["opponents"]["equity"]["candidate"]["mbb_per_hand"] > 0
        return report


def evaluation_metrics(report):
    metrics = {"stage2/eval/update": report["update"],
               "stage2/eval/retention_pass": int(report["retention_pass"]),
               "stage2/eval/equity_positive": int(report["equity_positive"])}
    for name, row in report["opponents"].items():
        for key, value in row["candidate"].items():
            if isinstance(value, (int, float)):
                metrics[f"stage2/eval/{name}/{key}"] = value
        metrics[f"stage2/eval/{name}/base_mbb_per_hand"] = row["baseline"]["mbb_per_hand"]
        metrics[f"stage2/eval/{name}/gain_mbb_per_hand"] = row["gain_mbb_per_hand"]
        for suffix, interval in (("mbb", row["candidate"]["mbb_ci95"]), ("gain", row["gain_ci95"])):
            metrics[f"stage2/eval/{name}/{suffix}_ci95_low"] = interval[0]
            metrics[f"stage2/eval/{name}/{suffix}_ci95_high"] = interval[1]
    return metrics

"""120-update fine-tune of u250_pro update 40 on the Botzone native tree.

The frozen feature backbone stays the golden200k student, because that is the
residual contract Botzone loads. The KL anchor and the 10% reference opponent
are the frozen update-40 policy, so the sprint does not practice against the
student it has already left behind.
"""
import copy
import json
import math
import random
import shutil
import time
import types
from pathlib import Path

import numpy as np
import torch

from alphanl.cfr.evaluation import mean_interval
from alphanl.stage2.allin_mask import SETTINGS_KEY, mask_config
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.evaluation import MatchEvaluator, evaluation_metrics, summarize_match_pairs
from alphanl.stage2.opponents import NeuralMatchActor
from alphanl.stage2.pool import DEFENSE, NATIVE_WEIGHTS, MatchOpponentPool
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.traces import trace_settings
from alphanl.stage2.train import (
    collect_matches,
    load_checkpoint,
    save_checkpoint,
    train_matches,
    warm_start_policy,
)
from alphanl.transformer.model import masked_log_probs
from alphanl.transformer.tables import standard_deck
from experiment_metadata import experiment_tags, write_experiment

OUT = Path("/root/autodl-tmp/stage2_u250_gen2_20261103")
BEST_PT = Path("/root/autodl-tmp/stage2_u250_pro_20261103/u250_pro_best.pt")
U40 = Path("/root/autodl-tmp/stage2_u250_pro_20261103/update_0000040.pt")
C200 = Path("/root/autodl-tmp/stage2_champion_v2_cand200_20261029/checkpoints/champion_stage2_v2_cand200.pt")
REFERENCE = "/root/autodl-tmp/distill_offtree_w150_350_golden200k_20261030/policy_best.pt"
STRATEGIES = "/root/autodl-tmp/stage2_botzone_native_code_20261030/runs/frozen_teacher/strategies"
SEED = 20261030
UPDATES = 120
EVAL_EVERY = 20
EVAL_PAIRS = 25  # 25 pairs x 2 seats x 50 hands = 2,500 hands


def pool_weights():
    """10/15/10 defense, then TAG 20, Trapper 20, equity 15, call 5, other scripts 5."""
    parts = {
        "reference": 1000,
        "history": 1500,
        "self": 1000,
        "TAG_Pro": 2000,
        "Trapper_CR": 2000,
        "equity": 1500,
        "call": 500,
        "OffTree_Aggro_Boss": 80,
        "Check_Shove_Trapper": 80,
        "Flop_Raise_War_Bot": 80,
        "nit": 65,
        "maniac": 65,
        "overbet": 65,
        "push_fold": 65,
    }
    if sum(parts.values()) != 10000:
        raise RuntimeError("pool parts must sum to 10000")
    if frozenset(parts) != frozenset(NATIVE_WEIGHTS):
        raise RuntimeError("pool keys are not the native Botzone mix: %s" % sorted(set(NATIVE_WEIGHTS) ^ set(parts)))
    weights = {key: value / 10000.0 for key, value in parts.items()}
    if not math.isclose(sum(weights.values()), 1.0) or any(
        not math.isclose(weights[key], quota) for key, quota in DEFENSE.items()
    ):
        raise RuntimeError("pool weights failed the defense quota check")
    return weights


def mask_block():
    return {
        "enabled": True,
        "spr_above": 1.2,
        "preflop_equity": 0.72,
        "postflop_equity": 0.8,
        "equity_samples": 64,
        "equity_seed": 17,
    }


def build_settings():
    mask = mask_block()
    return {
        "stage": "stage2",
        "device": "cuda",
        "seed": 44,
        "reference": REFERENCE,
        "strategies": STRATEGIES,
        "updates": UPDATES,
        "matches_per_update": 20,
        "learning_rate": 3.5e-5,
        "botzone_native_allin_mask": mask,
        "policy": {
            "alpha": 1.0,
            "hidden": 64,
            "beta_kl": 0.05,
            "late_beta_kl": 0.03,
            "context_fields": [
                "hand_progress", "bankroll_diff", "vpip", "pfr", "three_bet",
                "fold_to_preflop_raise", "flop_aggression", "turn_aggression",
                "river_aggression", "fold_to_postflop_bet", "all_in_frequency",
                "showdown_win", "check_raise_freq", "cbet_freq",
            ],
            "botzone_native_allin_mask": mask,
        },
        "ppo": {"epochs": 4, "minibatch_size": 256, "gamma": 1.0, "gae_lambda": 1.0},
        "pool": {
            "weights": pool_weights(),
            "equity_samples": 32,
            "window_size": 100,
            "history_cache_size": 16,
            "history_refresh_matches": 100,
            "history_buckets": [[140, 160], [240, 260], [320, 350], [100, 350]],
            "self_snapshot_every": 20,
            "self_history_size": 16,
        },
        "evaluation": {
            "every_updates": EVAL_EVERY,
            "match_pairs": EVAL_PAIRS,
            "seed": SEED,
            "equity_samples": 32,
            "stop_on_retention_failure": False,
            "opponents": ["equity", "TAG_Pro", "Trapper_CR", "reference"],
        },
        "traces": {
            "enabled": False,
            "train_every_updates": 25,
            "train_matches": 0,
            "eval_match_pairs_per_opponent": 0,
            "max_bytes": 16777216,
        },
        "checkpoint_every": EVAL_EVERY,
        "notes": {
            "start": "Warm start from u250_pro_best.pt, which is update 40.",
            "anchor": "KL target and the 10% reference opponent are the frozen update-40 policy. The feature backbone remains golden200k so the checkpoint stays exportable.",
            "pool": "Defense stays 10/15/10. Of the other 65%: TAG_Pro 20, Trapper_CR 20, equity 15, call 5, ladder replicas and the four small scripts share 5.",
            "select": "Every 20 updates, 25 match pairs on seed 20261030. Keep the highest min(TAG_Pro, Trapper_CR) among checkpoints with equity > 1000.",
        },
    }


def sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def bind_anchor(model, anchor):
    def kl_penalty(self, batch):
        batch = self.prepared(batch)
        logits, _ = self(batch)
        with torch.no_grad():
            anchor_logits, _ = anchor(batch)
        _, logp, probs, legal = masked_log_probs(logits, batch["legal_mask"])
        _, ref_logp, _, _ = masked_log_probs(anchor_logits, batch["legal_mask"])
        kl = (probs * (logp - ref_logp).masked_fill(~legal, 0.0)).sum(-1).clamp_min(0.0)
        progress = ((batch["match_ctx"][:, 0] * 50 - 10) / 40).clamp(0, 1)
        beta = self.settings["beta_kl"] + progress * (self.settings["late_beta_kl"] - self.settings["beta_kl"])
        return (beta * kl).mean(), kl, beta

    model.kl_penalty = types.MethodType(kl_penalty, model)


def patch_reference_opponent(pool, anchor, seed):
    original = pool.sample
    seen = {"n": 0}

    def sample():
        name, actor = original()
        if name == "reference":
            seen["n"] += 1
            actor = NeuralMatchActor(anchor, kind="residual", snapshot_id="u40_anchor", seed=seed + seen["n"])
        return name, actor

    pool.sample = sample


def load_model(path, device):
    try:
        return load_checkpoint(path, device=device)
    except Exception as exc:
        print("LOAD_RETRY", path.name, type(exc).__name__, flush=True)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob.pop("astra", None)
        clean = path.with_suffix(".clean.pt")
        torch.save(blob, clean)
        return load_checkpoint(clean, device=device)


def cell(report, name):
    row = report["opponents"][name]["candidate"]
    return {
        "mbb_per_hand": row["mbb_per_hand"],
        "mbb_ci95": list(row["mbb_ci95"]),
        "hands": row["hands"],
        "match_pairs": row["match_pairs"],
    }


def script_eval(model, update, pairs):
    evaluator = MatchEvaluator(
        model, match_pairs=pairs, seed=SEED, equity_samples=32,
        opponents=("equity", "TAG_Pro", "Trapper_CR", "reference"),
    )
    report = evaluator.evaluate(
        model, update=update, progress=lambda name: print("confirm", update, name, flush=True)
    )
    return {name: cell(report, name) for name in ("equity", "TAG_Pro", "Trapper_CR", "reference")}


class GreedyResidualActor(NeuralMatchActor):
    def act(self, snapshot):
        batch = self.encode(snapshot)
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        logits, _ = self.model(batch)
        return int(torch.argmax(logits, dim=-1).item())


@torch.inference_mode()
def play_h2h(hero, villain, *, decks, first_hero_seat, seed):
    torch.manual_seed(seed)
    opponent = GreedyResidualActor(villain, kind="residual", snapshot_id="villain", seed=seed)
    env = BotzoneMatchEnv(
        hero.config, opponent, decks=decks, first_hero_seat=first_hero_seat,
        context_fields=hero.context_fields, allin_mask=hero.settings.get(SETTINGS_KEY),
    )
    obs = env.reset()
    device = next(hero.parameters()).device
    while not env.done:
        batch = {key: value.to(device) for key, value in obs.items()}
        action, *_ = hero.act(batch, deterministic=True)
        obs, _, _, _ = env.step(int(action))
    return env.ctx.hero_net_bb


def h2h_direction(hero, villain, decks, seed):
    totals = []
    for pair, pair_decks in enumerate(decks):
        legs = []
        for first_seat in (1, 0):
            legs.append(play_h2h(
                hero, villain, decks=pair_decks, first_hero_seat=first_seat,
                seed=seed + pair * 2 + first_seat,
            ))
        totals.append(legs)
    return totals


def combined(forward, reverse):
    left = np.asarray(forward, float).mean(axis=1) / 50 * 1000
    right = (-np.asarray(reverse, float)).mean(axis=1) / 50 * 1000
    return mean_interval((left + right) / 2)


def train(settings):
    device = "cuda"
    count = settings["matches_per_update"]
    updates = settings["updates"]
    evaluation = settings["evaluation"]
    traces = trace_settings(settings)
    settings["traces"] = traces
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    OUT.mkdir(parents=True, exist_ok=False)
    experiment = write_experiment(OUT, kind="formal", name="u250_pro_gen2", stage="stage2")
    identity = experiment_tags(experiment)
    settings["experiment"] = experiment
    (OUT / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    pool = MatchOpponentPool(settings["strategies"], seed=settings["seed"], **settings["pool"])
    model = ResidualExploiterPolicy.from_reference(settings["reference"], **settings["policy"]).to(device)
    initialization = warm_start_policy(model, BEST_PT)
    if int(initialization["source_update"]) != 40:
        raise RuntimeError("warm start is update %s, expected 40" % initialization["source_update"])
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    live = next(model.parameters())
    frozen = next(anchor.parameters())
    if live.data_ptr() == frozen.data_ptr():
        raise RuntimeError("anchor shares storage with the live policy")
    bind_anchor(model, anchor)
    pool.initialize(model)
    patch_reference_opponent(pool, anchor, settings["seed"] * 1000003)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"])
    evaluator = MatchEvaluator(
        model, match_pairs=evaluation["match_pairs"], seed=evaluation["seed"],
        equity_samples=evaluation["equity_samples"], opponents=evaluation["opponents"],
    )
    (OUT / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
    save_checkpoint(
        OUT / "initial.pt", model, optimizer=optimizer, pool=pool, update=0,
        metadata={**identity, "initialization": initialization, "matches": 0, "timesteps": 0,
                  "anchor": "frozen update 40"},
    )
    timesteps = matches = 0
    every = evaluation["every_updates"]
    for update in range(1, updates + 1):
        began = time.perf_counter()
        print("UPDATE_START", update, flush=True)
        mask = model.settings.get("botzone_native_allin_mask")
        envs = [
            BotzoneMatchEnv(
                model.config, pool=pool, seed=settings["seed"] + matches + i,
                first_hero_seat=i % 2, defer_opponents=True,
                context_fields=model.context_fields, allin_mask=mask,
            )
            for i in range(count)
        ]
        with (OUT / "matches.jsonl").open("a") as match_stream:
            def completed(index, report, update=update):
                pool.record(report["opponent"], report["hero_net_bb"])
                record = {key: value for key, value in report.items() if key not in ("hands", "context")}
                record.update(identity)
                record.update(format="stage2_training_match_v1", update=update, match=matches + index + 1)
                match_stream.write(json.dumps(record, allow_nan=False) + "\n")
                match_stream.flush()

            rollouts = collect_matches(model, envs, on_match_end=completed)
        metrics = train_matches(model, optimizer, rollouts, **settings["ppo"])
        if update % pool.self_snapshot_every == 0:
            pool.add_self_snapshot(model, update=update)
        timesteps += sum(len(item.actions) for item in rollouts)
        matches += count
        row = {
            **identity, "update": update, "stage2": 1, "timesteps": timesteps, **metrics,
            "stage2/matches": matches, "stage2/hands": matches * 50,
            "stage2/opponent": "mixed",
            "stage2/hero_net_bb": sum(env.ctx.hero_net_bb for env in envs) / count,
            "stage2/defense_fraction": sum(env.opponent_name in ("reference", "history", "self") for env in envs) / count,
            "stage2/anchor": "u40",
        }
        report = None
        if update % every == 0:
            report = evaluator.evaluate(
                model, update=update,
                progress=lambda name, update=update: print("eval2500", update, name, flush=True),
            )
            report.update(identity)
            report_dir = OUT / "stage2_evaluations"
            report_dir.mkdir(exist_ok=True)
            (report_dir / f"update_{update:07d}.json").write_text(
                json.dumps(sanitize(report), indent=2, allow_nan=False) + "\n"
            )
            slim = {name: cell(report, name) for name in ("equity", "TAG_Pro", "Trapper_CR", "reference")}
            with (OUT / "selection_live.jsonl").open("a") as stream:
                stream.write(json.dumps({"update": update, "scores": slim}) + "\n")
            print("EVAL2500", update, json.dumps(slim), flush=True)
            row.update(evaluation_metrics(report))
        row["seconds"] = time.perf_counter() - began
        row = sanitize(row)
        with (OUT / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        metadata = {
            **identity, "reference": settings["reference"], "stage": "stage2",
            "timesteps": timesteps, "initialization": initialization, "matches": matches,
            "evaluation_settings": evaluation, "anchor": "frozen update 40",
        }
        save_checkpoint(OUT / "train.pt", model, optimizer=optimizer, pool=pool, update=update, metadata=metadata)
        if settings["checkpoint_every"] and update % settings["checkpoint_every"] == 0:
            save_checkpoint(OUT / f"update_{update:07d}.pt", model, update=update, metadata=metadata)
        print(json.dumps({"update": update, "seconds": row["seconds"], "loss": row.get("loss")}), flush=True)


def choose():
    rows = []
    for path in sorted((OUT / "stage2_evaluations").glob("update_*.json")):
        report = json.loads(path.read_text())
        scores = {name: cell(report, name) for name in ("equity", "TAG_Pro", "Trapper_CR", "reference")}
        equity = scores["equity"]["mbb_per_hand"]
        style = min(scores["TAG_Pro"]["mbb_per_hand"], scores["Trapper_CR"]["mbb_per_hand"])
        rows.append({"update": report["update"], "scores": scores, "equity": equity, "style_min": style,
                     "checkpoint": str(OUT / f"update_{report['update']:07d}.pt")})
    gated = [row for row in rows if row["equity"] > 1000]
    chosen = max(gated or rows, key=lambda row: (row["style_min"], row["equity"]))
    payload = {
        "rule": "max min(TAG_Pro, Trapper_CR) among 2500-hand rows with equity > 1000",
        "equity_gate_hit": bool(gated),
        "both_styles_positive": chosen["style_min"] > 0,
        "chosen": chosen,
        "table": rows,
    }
    (OUT / "selection_2500.json").write_text(json.dumps(payload, indent=2) + "\n")
    dest = OUT / "u250_gen2_best.pt"
    shutil.copy2(chosen["checkpoint"], dest)
    print("SELECTED", chosen["update"], chosen["equity"], chosen["style_min"], flush=True)
    return dest, chosen


def confirm(candidate_path):
    device = "cuda"
    candidate, blob = load_model(candidate_path, device)
    u40, _ = load_model(U40, device)
    c200, _ = load_model(C200, device)
    for model in (candidate, u40, c200):
        model.eval()
    print("CONFIRM_5000", int(blob["update"]), flush=True)
    scores = {
        "gen2": script_eval(candidate, int(blob["update"]), 50),
        "u40": script_eval(u40, 40, 50),
    }
    (OUT / "confirm_5000.json").write_text(json.dumps({
        "seed": SEED, "match_pairs": 50, "hands_per_opponent": 5000, "scores": scores,
    }, indent=2) + "\n")
    rng = np.random.default_rng(SEED)
    decks = [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(50)]
    matchups = {}
    for name, villain in (("u40", u40), ("c200", c200)):
        print("H2H", name, flush=True)
        forward = h2h_direction(candidate, villain, decks, SEED)
        reverse = h2h_direction(villain, candidate, decks, SEED + 100000)
        both = combined(forward, reverse)
        matchups["gen2_vs_%s" % name] = {
            "left_as_hero": summarize_match_pairs(forward),
            "right_as_hero": summarize_match_pairs(reverse),
            "combined_left_view": {"mbb_per_hand": both["mean"], "mbb_ci95": list(both["ci95"])},
        }
        (OUT / "h2h_gen2.json").write_text(json.dumps({"seed": SEED, "matchups": matchups}, indent=2) + "\n")
    gen2 = scores["gen2"]
    base = scores["u40"]
    style = min(gen2["TAG_Pro"]["mbb_per_hand"], gen2["Trapper_CR"]["mbb_per_hand"])
    base_style = min(base["TAG_Pro"]["mbb_per_hand"], base["Trapper_CR"]["mbb_per_hand"])
    h2h = matchups["gen2_vs_u40"]["combined_left_view"]["mbb_per_hand"]
    verdict = {
        "equity_above_1000": gen2["equity"]["mbb_per_hand"] > 1000,
        "style_min_beats_u40": style > base_style,
        "h2h_beats_u40": h2h > 0,
        "equity_beats_u40": gen2["equity"]["mbb_per_hand"] > base["equity"]["mbb_per_hand"],
        "gen2_equity": gen2["equity"]["mbb_per_hand"],
        "gen2_tag": gen2["TAG_Pro"]["mbb_per_hand"],
        "gen2_trapper": gen2["Trapper_CR"]["mbb_per_hand"],
        "u40_equity": base["equity"]["mbb_per_hand"],
        "u40_tag": base["TAG_Pro"]["mbb_per_hand"],
        "u40_trapper": base["Trapper_CR"]["mbb_per_hand"],
        "h2h_u40": h2h,
        "h2h_c200": matchups["gen2_vs_c200"]["combined_left_view"]["mbb_per_hand"],
    }
    verdict["beats_v8"] = bool(
        verdict["equity_above_1000"] and verdict["style_min_beats_u40"] and verdict["h2h_beats_u40"]
    )
    (OUT / "gen2_result.json").write_text(json.dumps({
        "checkpoint": str(candidate_path), "update": int(blob["update"]), "verdict": verdict,
    }, indent=2) + "\n")
    print("GEN2_DONE", json.dumps(verdict), flush=True)


def main():
    settings = build_settings()
    train(settings)
    path, _chosen = choose()
    confirm(path)


if __name__ == "__main__":
    main()

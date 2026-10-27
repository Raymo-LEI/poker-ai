"""Measure a frozen CFR checkpoint and average policy; append real monitor metrics.

No optimizer steps, trainer resume, network requests or writes to model files.
Use only a trusted checkpoint, after its training/checkpoint publication finishes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.diagnostics import regression_probe, probe_comparison
from alphanl.cfr.evaluation import CFRActor, RuleActor, duplicate_match, health_check, OPPONENTS
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.model import PokerTransformer


def append_metrics(path, row):
    if path.name == "eval.jsonl":
        row = {**row, "diagnostic/eval_average_policy": 1}
    clean = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in row.items()}
    with path.open("a") as handle:
        handle.write(json.dumps(clean, allow_nan=False) + "\n")
    print(json.dumps(clean, allow_nan=False), flush=True)


def checkpoint_probes(run, *, samples, device, seed):
    # The caller explicitly chooses its own trusted training artifact.
    blob = torch.load(run / "train.pt", map_location="cpu", weights_only=False)
    cfg = CFRConfig.from_checkpoint(blob["config"])
    result = {"iteration": int(blob["completed_iteration"]), "diagnostic/probe_is_training_data": 1}
    for player in (0, 1):
        model_key = player if cfg.separate_player_models else "shared"
        saved = blob["buffers"][model_key]
        if saved.get("storage") == "memmap_snapshot_v1":
            meta = saved["metadata"]
            directory = (run / saved["directory"]).resolve()
            arrays = {key: np.memmap(directory / f"{key}.dat", mode="r", dtype=field["dtype"],
                       shape=(meta["size"], *field["shape"])) for key, field in meta["schema"].items()}
        else:
            meta, arrays = saved, saved["data"]
        candidates = (np.arange(meta["size"]) if cfg.separate_player_models
                      else np.flatnonzero(arrays["hero_seat"] == player))
        count = min(samples, len(candidates))
        if not count:
            continue
        indices = np.random.default_rng(np.random.SeedSequence([seed, player, 7843])).choice(candidates, count, replace=False)
        batch = {}
        for key in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask"):
            value = torch.from_numpy(np.array(arrays[key][indices], copy=True))
            batch[key] = (value.long() if key in ("token_ids", "type_ids") else value).to(device)
        target = torch.from_numpy(np.array(arrays["regrets"][indices], copy=True)).to(device) / cfg.regret_target_scale
        weights = torch.from_numpy(np.array(arrays["iterations"][indices], copy=True)).to(device)
        model = RegretNetwork(PokerTransformer(**asdict(cfg.model)), fallback=cfg.fallback).to(device).eval()
        model.load_state_dict(blob["models"][model_key])
        with torch.inference_mode():
            predicted = model(batch)
            end = regression_probe(predicted, target, batch["legal_mask"], weights)
            zero = regression_probe(torch.zeros_like(predicted), target, batch["legal_mask"], weights)
            values = probe_comparison(zero, end)
        # This is a post-hoc end/zero comparison, not an observed historical start.
        values["probe_end_zero_ratio"] = values.pop("probe_loss_ratio")
        values.pop("probe_start_loss")
        values["probe_samples"] = count
        result.update({f"p{player}_{key}": value for key, value in values.items()})
        del model, arrays
    append_metrics(run / "diagnostics.jsonl", result)
    return result["iteration"]


def health_metrics(result, iteration):
    row = {"iteration": iteration, "hud/eval_hands": result["hands"]}
    audits = [result["hero"], result["opponent"]]
    all_counts = {}
    for audit in audits:
        for key, value in audit["counts"].items():
            all_counts[key] = all_counts.get(key, 0) + value
    denominator = all_counts.get("sb_open_opportunities", 0)
    if denominator:
        row["hud/preflop_vpip_sb"] = all_counts.get("sb_vpip", 0) / denominator
        row["hud/sb_open_all_in"] = all_counts.get("sb_open_shoves", 0) / denominator
    # VPIP/PFR are once-per-dealt-hand statistics, not per-decision action rates.
    for position in ("sb", "bb"):
        hands = all_counts.get(f"hands_{position}", 0)
        if hands:
            row[f"hud/preflop_vpip_{position}"] = all_counts.get(f"vpip_hands_{position}", 0) / hands
            row[f"hud/preflop_vpip_{position}_hands"] = hands
            if position == "sb":
                row["hud/preflop_pfr_sb"] = all_counts.get("pfr_hands_sb", 0) / hands
                row["hud/preflop_pfr_sb_hands"] = hands
    if "hands_bb" in all_counts:
        opportunities = all_counts.get("bb_3bet_opportunities", 0)
        row["hud/preflop_3bet_bb_opportunities"] = opportunities
        row["hud/preflop_3bet_bb"] = all_counts.get("bb_3bets", 0) / opportunities if opportunities else None
        free = all_counts.get("free_check_opportunities", 0)
        row["hud/free_check_opportunities"] = free
        row["hud/free_check_fold_ratio"] = all_counts.get("free_check_folds", 0) / free if free else None
    regret_decisions = sum(a.get("positive_regret_decisions", 0) for a in audits)
    if regret_decisions:
        row["hud/positive_regret_actions_mean"] = sum(a.get("positive_regret_actions_sum", 0) for a in audits) / regret_decisions
        row["hud/positive_regret_decisions"] = regret_decisions
    denominator = all_counts.get("bb_vs_open_raise", 0)
    if denominator:
        row["hud/bb_fold_to_open_raise"] = all_counts.get("bb_fold_vs_open_raise", 0) / denominator
    nonforced = sum(a["counts"].get("nonforced_decisions", 0) for a in audits)
    if nonforced:
        row["hud/policy_entropy_mean"] = sum((a.get("mean_nonforced_entropy_nats") or 0) * a["counts"].get("nonforced_decisions", 0) for a in audits) / nonforced
    for street_id, street in enumerate(("preflop", "flop", "turn", "river")):
        combined = [0] * 5
        for player in (0, 1):
            prefix = f"spr_{player}_{street_id}"
            for metric in ("decisions", "blocked", "high_spr_shoves", "early_high_spr_shoves"):
                key = prefix + "_" + metric
                if key in all_counts:
                    row[f"hud/p{player}_{street}_{metric}"] = all_counts[key]
            counts = [sum(a.get("action_counts_by_seat_street", {}).get(f"{player}:{street_id}:{action}", 0) for a in audits) for action in range(5)]
            total = sum(counts)
            if total:
                row[f"hud/p{player}_{street}_decisions"] = total
                for name, value in zip(("fold", "call", "half_pot", "pot", "all_in"), counts):
                    row[f"hud/action_freq_p{player}_{street}/{name}"] = value / total
            combined = [x + y for x, y in zip(combined, counts)]
        if sum(combined):
            for name, value in zip(("fold", "call", "half_pot", "pot", "all_in"), combined):
                row[f"hud/action_freq_{street}/{name}"] = value / sum(combined)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--health-hands", type=int, default=4000)
    parser.add_argument("--probe-samples", type=int, default=2048)
    parser.add_argument("--probe-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--health-only", action="store_true", help="refresh only HUD; preserve opponent evaluation and checkpoint")
    args = parser.parse_args()
    if min(args.iteration, args.health_hands, args.probe_samples) < 1 or args.pairs < 2:
        parser.error("positive budgets and pairs >= 2 required")
    names = args.opponents.split(",")
    if any(name not in OPPONENTS for name in names):
        parser.error("unsupported opponent")
    torch.set_num_threads(1)
    run = args.run_dir.resolve()
    archive = json.loads((run / "strategies/index.json").read_text())
    if [i for i in archive["iterations"] if i <= args.iteration] != list(range(1, args.iteration + 1)):
        raise ValueError("full average-policy evaluation requires all snapshots 1..T")
    output = run / ("monitor_hud" if args.health_only else "monitor_diagnostics") / f"t{args.iteration:07d}_{int(time.time())}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": "overwatch_diagnostics_v1", "iteration": args.iteration,
              "seed": args.seed, "pairs": args.pairs, "health_hands": args.health_hands,
              "policy": "linear-weighted trajectory mixture sigma_1..sigma_T", "results": {}}

    def publish(key, result):
        report["results"][key] = result
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2))
        temporary.replace(output / "report.json")

    if not args.health_only:
        measured = checkpoint_probes(run, samples=args.probe_samples, device=args.probe_device, seed=args.seed)
        if measured != args.iteration:
            print(f"Probe checkpoint T={measured}; policy evaluation T={args.iteration}", flush=True)

    def actor(seed):
        return CFRActor(run / "strategies", seed=seed, through=args.iteration, cache_size=args.iteration)

    hero, opponent = actor(args.seed + 1), actor(args.seed + 2)
    with (output / "health.jsonl").open("w") as log:
        result = health_check(hero, opponent, hands=args.health_hands, seed=args.seed,
            log=lambda row: log.write(json.dumps(row) + "\n"),
            progress=lambda done, total: print(f"health {done}/{total}", flush=True))
    metrics = health_metrics(result, args.iteration)
    if args.health_only and (run / "eval.jsonl").is_file():
        previous = [json.loads(line) for line in (run / "eval.jsonl").read_text().splitlines()
                    if line.strip()]
        previous = [r for r in previous if r.get("iteration") == args.iteration
                    and r.get("hud/eval_hands") == args.health_hands]
        if previous:
            # Audit additions must not change the frozen-policy replay.
            for key, old in previous[-1].items():
                if key.startswith("hud/") and key in metrics and old is not None and metrics[key] is not None:
                    if not math.isclose(old, metrics[key], abs_tol=1e-12, rel_tol=1e-12):
                        raise ValueError(f"HUD replay changed existing measurement {key}: {old} -> {metrics[key]}")
    metrics["diagnostic/hud_schema_version"] = 2
    append_metrics(run / "eval.jsonl", metrics)
    publish("health", result)
    del hero, opponent
    if args.health_only:
        print(f"Saved: {output / 'report.json'}", flush=True)
        return
    for index, name in enumerate(names):
        hero = actor(args.seed + 11 + index)
        opponent = RuleActor(name, seed=args.seed + 101 + index)
        with (output / f"vs_{name}.jsonl").open("w") as log:
            result = duplicate_match(hero, opponent, pairs=args.pairs, seed=args.seed + 1001 + index,
                log=lambda row: log.write(json.dumps(row) + "\n"),
                progress=lambda done, total: print(f"{name} {done}/{total}", flush=True))
        tag = {"call": "calling_station", "maniac": "aggressive"}.get(name, name)
        key = f"eval/winrate_vs_{tag}_mbb"
        row = {"iteration": args.iteration, key: result["mbb_per_hand"], key + "_pairs": result["pairs"],
               key + "_ci_low": result["ci95"][0], key + "_ci_high": result["ci95"][1]}
        if name == "random":
            row.update({"eval/positional_sb_mbb": result["sb_mbb_per_hand"], "eval/positional_bb_mbb": result["bb_mbb_per_hand"]})
        append_metrics(run / "eval.jsonl", row)
        publish(name, result)
        del hero, opponent
    print(f"Saved: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

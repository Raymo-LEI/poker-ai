#!/usr/bin/env python3
"""Six-opponent paired evaluation with decks independent of arm and iteration."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from alphanl.cfr.evaluation import CFRActor, OPPONENTS, RuleActor, duplicate_match, mean_interval
from alphanl.cfr.study import publish, sha256
from alphanl.transformer.hunl import BIG_BLIND


def opponent_seed(seed, name):
    return seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def task(settings, name):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed = opponent_seed(settings["seed"], name)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    hero = CFRActor(settings["archive"]["path"], seed=seed + 11,
                    through=settings["archive"]["through"], cache_size=64)
    opponent = RuleActor(name, seed=seed + 19, equity_samples=64)
    raw = Path(settings["out"]) / f"vs_{name}.jsonl"
    with raw.open("x") as handle:
        def log(record):
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        result = duplicate_match(hero, opponent, pairs=settings["pairs"], seed=seed, log=log,
                    progress=lambda n, total: print(f"{settings['tag']}/{name} {n}/{total}", flush=True))
    result.update(raw_log=str(raw), raw_sha256=sha256(raw), task_seed=seed,
                  hero_iteration=settings["archive"]["through"])
    return name, result


def paired_delta(left, right):
    a = [json.loads(row) for row in Path(left).read_text().splitlines()]
    b = [json.loads(row) for row in Path(right).read_text().splitlines()]
    if len(a) != len(b):
        raise ValueError("paired difference requires equal budgets")
    values = []
    for x, y in zip(a, b):
        if (x["pair"], x["deck_seed"]) != (y["pair"], y["deck_seed"]):
            raise ValueError("paired difference requires identical deck seeds")
        values.append((np.mean(y["chips_sb_bb"]) - np.mean(x["chips_sb_bb"])) / BIG_BLIND * 1000)
    return {**mean_interval(values), "unit": "mbb/hand", "positive_means": "2400 improves on 1200"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.study / "manifest.json").read_text())
    arms = {s: json.loads((args.study / f"arm{s}.json").read_text()) for s in (1200, 2400)}
    report = {"format": "sd_cfr_stage1_paired_evaluation_v1", "seed": manifest["seed"],
              "pairs_per_opponent": manifest["pairs"], "results": {}, "paired_deltas": {},
              "notes": ["95% pointwise CIs use independent duplicate deck pairs, not individual hands.",
                        "Each snapshot is sampled with its absolute CFR iteration weight t, once per hand.",
                        "Equal-time uses completed iterations only; exact elapsed times and unused budget retained.",
                        "Heuristic opponents are not a best-response exploitability certificate."]}
    path = args.study / "evaluation_report.json"
    publish(path, report)
    jobs = {}
    # Baseline's endpoint is also the common wall-time budget reference.
    for steps in (1200, 2400):
        for mode in ("same_iteration", "equal_time"):
            archive = arms[steps][f"{mode}_archive"]
            key = (archive["path"], archive["through"])
            if key not in jobs:
                tag = f"steps{steps}_{mode}"
                out = Path(arms[steps]["run"]) / "evaluations" / f"six_opponents_{mode}"
                out.mkdir(parents=True, exist_ok=False)
                jobs[key] = {"tag": tag, "archive": archive, "out": str(out),
                             "pairs": manifest["pairs"], "seed": manifest["seed"], "aliases": []}
            jobs[key]["aliases"].append(f"steps{steps}_{mode}")
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in
               [Path(__file__), *sorted((ROOT / "src/alphanl").rglob("*.py"))]}
    report["source_sha256"] = sources
    for settings in jobs.values():
        archive = Path(settings["archive"]["path"])
        snapshots = sorted(archive.glob("iteration_*.pt"))
        fingerprints = {p.name: sha256(p) for p in snapshots}
        group = {"archive": settings["archive"], "snapshots_sha256": fingerprints,
                 "pairs": manifest["pairs"], "results": {}}
        with ProcessPoolExecutor(max_workers=manifest["eval_workers"], max_tasks_per_child=1,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(task, settings, name) for name in OPPONENTS]
            for future in as_completed(pending):
                name, result = future.result()
                group["results"][name] = result
                for alias in settings["aliases"]:
                    report["results"][alias] = group
                publish(path, report)
                steps = 1200 if settings["tag"].startswith("steps1200_") else 2400
                with (Path(arms[steps]["run"]) / "eval.jsonl").open("a") as handle:
                    handle.write(json.dumps({"iteration": settings["archive"]["through"],
                        "eval/pairs": manifest["pairs"],
                        "diagnostic/eval_average_policy": 1,
                        f"eval/winrate_vs_{'calling_station' if name == 'call' else 'aggressive' if name == 'maniac' else name}_mbb": result["mbb_per_hand"],
                        f"eval/winrate_vs_{'calling_station' if name == 'call' else 'aggressive' if name == 'maniac' else name}_mbb_ci_low": result["ci95"][0],
                        f"eval/winrate_vs_{'calling_station' if name == 'call' else 'aggressive' if name == 'maniac' else name}_mbb_ci_high": result["ci95"][1]}, allow_nan=False) + "\n")
        if any(sha256(archive / filename) != fingerprint for filename, fingerprint in fingerprints.items()):
            raise RuntimeError("evaluation archive changed")
        publish(Path(settings["out"]) / "report.json", group)
    for mode in ("same_iteration", "equal_time"):
        left = report["results"][f"steps1200_{mode}"]["results"]
        right = report["results"][f"steps2400_{mode}"]["results"]
        report["paired_deltas"][mode] = {name: paired_delta(left[name]["raw_log"], right[name]["raw_log"])
                                          for name in OPPONENTS}
    report["complete"] = True
    # No arbitrary scalar score silently trades off losses against an opponent.
    report["training_budget_winner"] = None
    report["selection_note"] = "Review matched CIs and same-time tradeoffs before selecting Config3 train_steps."
    publish(path, report)
    for arm in arms.values():
        publish(Path(arm["run"]) / "study_status.json", {"phase": "complete", "target": manifest["target"]})


if __name__ == "__main__":
    main()

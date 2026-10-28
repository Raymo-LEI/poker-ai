#!/usr/bin/env python3
"""Evaluate frozen SD-CFR mixtures without modifying or loading training state.

Default budgets are 2500 pairs/opponent, 500 independent 50-hand matches,
and 10000 self-play hands. Smaller budgets are diagnostics, never launch
clearance. Progress and raw outcomes are persisted after every task.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from alphanl.cfr.evaluation import (CFRActor, OPPONENTS, RuleActor, duplicate_match,
                                    health_check, independent_matches, launch_gates)
from alphanl.cfr.evaluation_io import atomic_json, output_lock, sha256_file
from alphanl.cfr.strategy import SnapshotPolicy
from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION


def csv_ints(value):
    try:
        result = sorted({int(x) for x in value.split(",")})
        if not result or min(result) < 1:
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated positive iterations")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hero-iteration", type=int)
    parser.add_argument("--milestones", type=csv_ints, default=csv_ints("50,100,200,400,500,600,800,1000"))
    parser.add_argument("--opponents", default=",".join(OPPONENTS))
    parser.add_argument("--stages", default="duplicate,matches,health")
    parser.add_argument("--pairs", type=int, default=2500, help="Independent pairs; 2 hands each")
    parser.add_argument("--matches", type=int, default=500, help="Independent 50-hand matches/opponent")
    parser.add_argument("--health-hands", type=int, default=10000)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=64, help="LRU CPU snapshot count per actor")
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--spr-action-guard", action=argparse.BooleanOptionalAction, default=None,
                        help="Explicit evaluation-only intervention; archive is unchanged")
    parser.add_argument("--guard-max-pot-ratio", type=float)
    parser.add_argument("--resume", action="store_true", help="Reuse completed tasks only with identical inputs and code")
    args = parser.parse_args(argv)
    args.opponents = [x for x in args.opponents.split(",") if x]
    args.stages = [x for x in args.stages.split(",") if x]
    if set(args.opponents) - set(OPPONENTS) or len(set(args.opponents)) != len(args.opponents):
        parser.error(f"opponents must be distinct names from {OPPONENTS}")
    if set(args.stages) - {"duplicate", "matches", "health"} or not args.stages:
        parser.error("stages must be duplicate,matches,health or a subset")
    if "matches" in args.stages and not args.opponents:
        parser.error("matches require at least one opponent")
    if (args.pairs < 2 or args.matches < 1 or args.health_hands < 1 or args.threads < 1
            or args.cache_size < 0 or args.equity_samples < 1
            or (args.hero_iteration is not None and args.hero_iteration < 1)):
        parser.error("positive budgets required; pairs>=2 and cache-size>=0")
    if args.guard_max_pot_ratio is not None and (not np.isfinite(args.guard_max_pot_ratio) or args.guard_max_pot_ratio <= 0):
        parser.error("guard-max-pot-ratio must be positive and finite")
    return args


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    archive = SnapshotPolicy(args.strategies)
    latest = int(max(archive.iterations))
    target = args.hero_iteration or latest
    if target not in archive.iterations:
        raise ValueError(f"iteration {target} is not archived (latest={latest})")
    selected = [int(i) for i in archive.iterations if i <= target]
    if selected != list(range(1, target + 1)):
        raise ValueError("full-mixture evaluation requires every snapshot 1..T; archive has gaps")
    manifest = [{"iteration": int(i), "file": str(p.resolve()), "sha256": sha256_file(p)}
                for i, p in zip(archive.iterations, archive.paths) if i <= target]
    metadata = SnapshotPolicy(args.strategies, snapshot_iterations=[target])
    metadata.begin_hand(0)
    source_paths = [Path(__file__), *sorted((ROOT / "src/alphanl").rglob("*.py"))]
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths}
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("out", "resume")}
    settings["hero_iteration"] = target
    runtime = {"python": sys.version, "torch": str(torch.__version__),
               "numpy": np.__version__, "platform": platform.platform(),
               "pokerkit": version("pokerkit"), "phevaluator": version("phevaluator"),
               "device": "cpu", "threads": torch.get_num_threads(),
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    identity = hashlib.sha256(json.dumps([settings, manifest, source_hashes, runtime], sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.json"
    if path.exists():
        if not args.resume:
            raise ValueError("output already has report.json; use --resume or a fresh directory")
        report = json.loads(path.read_text())
        if report["identity"] != identity:
            raise ValueError("resume rejected: code, settings, or strategy fingerprints changed")
    else:
        report = {"format": "sd_cfr_readiness_v1", "identity": identity,
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "hero_iteration": target, "settings": settings, "snapshots": manifest,
                  "source_sha256": source_hashes,
                  "rules_version": RULES_VERSION, "numeric_version": metadata.config.numeric_version,
                  "runtime": runtime,
                  "notes": ["All tiers here use standard HUNL; not the legacy Botzone simple referee.",
                            "Mixtures sample t proportional to iteration once per player per hand.",
                            "An explicit guard override evaluates a mask-modified mixture, not a retrained policy.",
                            "Guard overrides affect the hero only, except health self-play where both use the override.",
                            "Latest single archive is sigma_T (pre-fit), not post-fit theta_T.",
                            "Rule opponents are reproducible heuristics, not a local best response.",
                            "Intervals are pointwise; no guarantee of monotonic CFR improvement.",
                            "Thresholds are project targets, not certified GTO ranges."],
                  "results": {}}
        try:
            report["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            report["git_revision"] = None

    def publish():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["evaluation_process_peak_rss_mb"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
        report["launch_gate"] = launch_gates(report)
        atomic_json(path, report)

    publish()
    def actor(iteration, seed, single=False, intervention=False):
        return CFRActor(args.strategies, seed=seed, through=iteration,
                        iterations=[iteration] if single else None, cache_size=args.cache_size,
                        spr_guard=args.spr_action_guard if intervention else None,
                        max_pot_ratio=args.guard_max_pot_ratio if intervention else None)

    def run_task(key, kind, hero_iteration, opponent_name, single=False):
        if key in report["results"]:
            previous = report["results"][key]
            raw = args.out / previous["raw_log"]
            if not raw.is_file() or sha256_file(raw) != previous.get("raw_log_sha256"):
                raise ValueError(f"resume rejected: missing or changed raw log for {key}")
            print(f"reuse {key}", flush=True)
            return
        task_seed = args.seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        hero = actor(hero_iteration, task_seed + 11, intervention=True)
        opponent = (actor(opponent_name, task_seed + 19, single=single) if isinstance(opponent_name, int)
                    else RuleActor(opponent_name, seed=task_seed + 19, equity_samples=args.equity_samples))
        if kind == "health":
            opponent = actor(hero_iteration, task_seed + 19, intervention=True)
        print(f"start {key}", flush=True)
        raw = args.out / (key.replace(":", "_") + ".jsonl")
        with raw.open("w") as handle:
            def log(record):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
            def progress(done, total):
                print(f"{key}: {done}/{total}", flush=True)
            try:
                if kind == "duplicate":
                    result = duplicate_match(hero, opponent, pairs=args.pairs, seed=task_seed, log=log, progress=progress)
                elif kind == "matches":
                    result = independent_matches(hero, opponent, matches=args.matches, seed=task_seed, log=log, progress=progress)
                else:
                    result = health_check(hero, opponent, hands=args.health_hands, seed=task_seed, log=log, progress=progress)
            except Exception as exc:
                report["error"] = {"task": key, "type": type(exc).__name__, "message": str(exc),
                                   "hero_audit": hero.audit.summary()}
                publish()
                raise
        result["hero_guard_override"] = {"enabled": args.spr_action_guard,
                                          "max_pot_ratio": args.guard_max_pot_ratio}
        result["raw_log"] = raw.name
        result["raw_log_sha256"] = sha256_file(raw)
        report["results"][key] = result
        report.pop("error", None)
        publish()
        print(f"done {key}: {json.dumps({k: v for k,v in result.items() if k in ('mbb_per_hand','ci95','win_fraction')})}", flush=True)

    if "duplicate" in args.stages:
        if not args.skip_ladder:
            milestones = sorted({target} | {m for m in args.milestones if m <= target})
            matchups = {(target, m) for m in milestones if m < target}
            if target > 1:
                matchups.update({(target, 1), (target, target - 1)})
            matchups.update(zip(milestones[1:], milestones[:-1]))
            for later, earlier in sorted(matchups):
                run_task(f"duplicate:{later}_vs_{earlier}", "duplicate", later, earlier)
            run_task(f"duplicate:{target}_vs_single_{target}", "duplicate", target, target, single=True)
        for name in args.opponents:
            run_task(f"duplicate:{target}_vs_{name}", "duplicate", target, name)
    if "health" in args.stages:
        run_task("health", "health", target, target)
    if "matches" in args.stages:
        for name in args.opponents:
            run_task(f"matches:{target}_vs_{name}", "matches", target, name)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    publish()
    print(f"report: {path.resolve()} | launch: {report['launch_gate']['verdict']}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    with output_lock(args.out):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sequential, isolated 1200/2400 full-state branches and paired evaluation.

Training wall time includes probe/checkpoint overhead, excludes preparation,
queue waits and opponent evaluation. Equal-time comparison uses the last
completed iteration at or below the baseline's time budget (no interpolation).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from alphanl.cfr import CFRConfig, SDCFRTrainer
from alphanl.cfr.study import (check_memory, evaluate_probe, fixed_probe, freeze_prefix,
                               latency_probe, publish, sha256)
from alphanl.cfr.model import RegretNetwork
from alphanl.cfr.retention import archive_due
from alphanl.transformer.model import PokerTransformer, count_parameters
from alphanl.transformer.tokenizer import BettingHistoryTokenizer


def append(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def prepare(args):
    study = args.study
    study.mkdir(parents=True, exist_ok=False)
    provenance = "existing_full_checkpoint"
    if args.bootstrap_config is not None:
        # Missing historical reservoirs cannot be recreated from sigma_t alone.
        # This is a NEW baseline trajectory, labeled separately from old v2.
        bootstrap = CFRConfig.from_json(args.bootstrap_config)
        if bootstrap.separate_player_models or bootstrap.train_steps != 1200:
            raise ValueError("bootstrap requires a unified 1200-step baseline config")
        bootstrap.iterations = args.source_iteration
        bootstrap.checkpoint_archive_from = 100
        bootstrap.checkpoint_archive_every = 10
        bootstrap.checkpoint_archive_extra_count = 2 * ((args.target - args.source_iteration) // 10)
        bootstrap.out_dir = str(args.run_root / f"{study.name}_bootstrap")
        bootstrap.buffer_directory = None
        bootstrap.__post_init__()
        bootstrap_run = Path(bootstrap.out_dir)
        bootstrap_run.mkdir(parents=True, exist_ok=False)
        publish(bootstrap_run / "ops/run_config.json", bootstrap.to_dict())
        publish(study / "status.json", {"phase": "bootstrap_training", "run": str(bootstrap_run),
            "target": args.source_iteration, "note": "New baseline trajectory, not historical v2 replay."})
        with (study / "bootstrap.log").open("x") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/train_cfr.py"),
                "--config", str(bootstrap_run / "ops/run_config.json")],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        args.checkpoint = bootstrap_run / "train.pt"
        provenance = "new_baseline_rebuilt_from_seed"
    publish(study / "status.json", {"phase": "preparing_source", "source_iteration": args.source_iteration})
    source = study / "source"
    source.mkdir()
    # Atomic checkpoint publication replaces files; links pin immutable bytes.
    # The source run must be quiescent during this one-time freeze.
    before = args.checkpoint.stat()
    os.link(args.checkpoint, source / "train.pt")
    shutil.copytree(args.checkpoint.parent / "checkpoint_data", source / "checkpoint_data", copy_function=os.link)
    after = args.checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("source checkpoint changed while freezing; retry with a quiescent source")
    fingerprint = sha256(source / "train.pt")
    blob = torch.load(source / "train.pt", map_location="cpu", weights_only=False)
    cfg = CFRConfig.from_checkpoint(blob["config"])
    completed = int(blob["completed_iteration"])
    if completed != args.source_iteration:
        raise ValueError(f"expected full checkpoint T={args.source_iteration}, got T={completed}; cannot rewind a later reservoir")
    if cfg.separate_player_models or args.target <= completed:
        raise ValueError("study requires unified checkpoint and target after its completed iteration")
    source_config = cfg.to_dict()
    del blob
    cfg.out_dir = str(study / "ops" / "probe_source")
    cfg.buffer_directory = None
    cfg.iterations = args.target
    cfg.__post_init__()
    check_memory()
    with SDCFRTrainer(cfg) as trainer:
        trainer.load_checkpoint(source / "train.pt", branch_train_steps=True)
        reservoir = trainer.shared_reservoir
        pool_state = {"size": len(reservoir), "seen": reservoir.seen, "capacity": reservoir.capacity,
                      "full": len(reservoir) == reservoir.capacity}
        if args.require_full_pool and not pool_state["full"]:
            raise ValueError(f"source T={completed} reservoir is not full: {pool_state}; no branch started")
        probe = fixed_probe(trainer, rows=args.probe_rows, seed=args.seed)
        torch.save(probe, study / "fixed_training_probe.pt")
        initial = evaluate_probe(trainer, probe)
        publish(study / "initial_probe.json", initial)
    big = CFRConfig.from_checkpoint(source_config)
    big.iterations = 300
    big.model.d_model, big.model.nhead, big.model.dim_feedforward = 192, 6, 512
    big.buffer_directory = None
    big.out_dir = str(args.run_root / "cfr_stage1_config3_149m_PENDING")
    parameters = {}
    # These two files are alternatives, not two queued large-model jobs.
    for steps in (1200, 2400):
        big.train_steps = steps
        publish(study / f"config3_149m_steps{steps}.json", big.to_dict())
    for name, model_cfg in (("baseline", cfg.model), ("sweet_spot", big.model)):
        model = PokerTransformer(**asdict(model_cfg))
        regret = RegretNetwork(model)
        latency = latency_probe(regret, BettingHistoryTokenizer(model_cfg.max_seq_len).blank())
        parameters[name] = {"model": asdict(model_cfg), "backbone_parameters": count_parameters(model),
                            "regret_trainable_parameters": count_parameters(regret),
                            "stored_parameters": sum(p.numel() for p in regret.parameters()),
                            "cpu_synthetic_max_length_latency": latency}
    manifest = {"format": "sd_cfr_stage1_budget_study_v1", "source_iteration": completed,
                "source_provenance": provenance, "source_pool": pool_state,
                "target": args.target, "source_checkpoint": str((source / "train.pt").resolve()),
                "source_sha256": fingerprint, "source_config": source_config,
                "probe_sha256": sha256(study / "fixed_training_probe.pt"), "seed": args.seed,
                "pairs": args.pairs, "eval_workers": args.eval_workers, "parameters": parameters,
                "arms": {str(s): str((args.run_root / f"{study.name}_steps{s}").resolve()) for s in (1200, 2400)},
                "config3": {"status": "prepared_not_started", "initial_learning_rate": big.learning_rate,
                            "train_steps_selection": "pending_stage1_results", "target": 300},
                "notes": ["Config2 inherits the baseline prefix; it is NOT 2400 steps from scratch.",
                          "Fixed probe is sampled from the common source TRAINING reservoir, never called holdout.",
                          "Common deck seeds do not depend on candidate, training step or evaluated iteration.",
                          "Equal-time checkpoints contain only completed CFR iterations and retain actual elapsed times.",
                          "No Stage2 winner is selected until cross-play, external and CPU gates have evidence."],
                "created_utc": datetime.now(timezone.utc).isoformat()}
    publish(study / "manifest.json", manifest)
    return manifest


def train_arm(args):
    study = args.study
    manifest = json.loads((study / "manifest.json").read_text())
    cfg = CFRConfig.from_checkpoint(manifest["source_config"])
    cfg.out_dir = manifest["arms"][str(args.steps)]
    cfg.train_steps, cfg.iterations = args.steps, manifest["target"]
    cfg.checkpoint_archive_from, cfg.checkpoint_archive_every = 100, 10
    cfg.checkpoint_archive_extra_count = ((cfg.iterations - manifest["source_iteration"]) // 10
                                        if args.steps == 1200 else 0)
    cfg.buffer_directory = None
    cfg.__post_init__()
    run = Path(cfg.out_dir)
    run.mkdir(parents=True, exist_ok=False)
    publish(run / "ops/run_config.json", cfg.to_dict())
    publish(run / "ops/environment.json", {"device": cfg.device,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "study": str(study), "branch_iteration": manifest["source_iteration"]})
    probe_path = study / "fixed_training_probe.pt"
    if sha256(probe_path) != manifest["probe_sha256"]:
        raise ValueError("common probe fingerprint changed")
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    budget = None
    if args.steps == 2400:
        budget = json.loads((study / "arm1200.json").read_text())["training_wall_seconds"]
    report = {"steps": args.steps, "run": str(run), "start_iteration": manifest["source_iteration"],
              "target": cfg.iterations, "wallclock_budget_seconds": budget,
              "training_wall_seconds": 0., "iterations": []}
    with SDCFRTrainer(cfg) as trainer:
        trainer.load_checkpoint(manifest["source_checkpoint"], branch_train_steps=True)
        append(run / "metrics.jsonl", evaluate_probe(trainer, probe))
        last_under = trainer.iteration
        last_under_seconds = 0.
        wall_archive = None
        while trainer.iteration < cfg.iterations:
            resources = check_memory()
            started = time.perf_counter()
            row = trainer.run_iteration()
            measurement = evaluate_probe(trainer, probe)
            # Saving at every 10 rounds preserves a full restart point. No archival
            # strategy snapshot is ever mistaken for a historical reservoir.
            if (trainer.iteration % cfg.checkpoint_every == 0 or trainer.iteration == cfg.iterations
                    or archive_due(cfg, trainer.iteration)):
                trainer.save_checkpoint(run / "train.pt")
            report["training_wall_seconds"] += time.perf_counter() - started
            measurement["study/branch_wall_seconds"] = report["training_wall_seconds"]
            measurement["study/train_steps"] = args.steps
            append(run / "metrics.jsonl", measurement)
            resources.update(check_memory())
            report["iterations"].append({"iteration": trainer.iteration,
                "wall_seconds": report["training_wall_seconds"],
                "regret_samples": row["p0_regret_samples"] + row["p1_regret_samples"], **resources})
            if budget is not None and wall_archive is None:
                if report["training_wall_seconds"] <= budget:
                    last_under, last_under_seconds = trainer.iteration, report["training_wall_seconds"]
                else:
                    # The most recent profile is omitted if its full iteration
                    # exceeded the budget. Keep a true prefix; no interpolation.
                    paths = trainer.strategy_paths
                    trainer.strategy_paths = {t: p for t, p in paths.items() if t <= last_under}
                    original = trainer.iteration
                    trainer.iteration = last_under
                    try:
                        wall_archive = freeze_prefix(trainer, run / "evaluations/equal_time/strategies")
                    finally:
                        trainer.strategy_paths, trainer.iteration = paths, original
                    wall_archive["training_wall_seconds"] = last_under_seconds
                    wall_archive["unused_budget_seconds"] = budget - last_under_seconds
            publish(study / f"arm{args.steps}.json", report)
            publish(run / "study_status.json", {"phase": "training", "iteration": trainer.iteration,
                     "target": cfg.iterations, "wall_seconds": report["training_wall_seconds"]})
            print(f"steps={args.steps} T={trainer.iteration} wall={report['training_wall_seconds']:.1f}s", flush=True)
        report["same_iteration_archive"] = freeze_prefix(trainer, run / "evaluations/same_iteration/strategies")
        if wall_archive is None:
            wall_archive = dict(report["same_iteration_archive"], training_wall_seconds=report["training_wall_seconds"])
            if budget is not None:
                wall_archive["unused_budget_seconds"] = budget - report["training_wall_seconds"]
        report["equal_time_archive"] = wall_archive
        report["completed"] = True
        publish(study / f"arm{args.steps}.json", report)
        publish(run / "study_status.json", {"phase": "awaiting_evaluation", "iteration": trainer.iteration})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "train-arm"))
    parser.add_argument("--study", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="trusted complete checkpoint at --source-iteration")
    source.add_argument("--bootstrap-config", type=Path, help="rebuild missing source state with a fresh baseline run")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-iteration", type=int, default=150)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--require-full-pool", action=argparse.BooleanOptionalAction, default=True,
                        help="require source reservoir to be full; disable only for diagnostic toy runs")
    parser.add_argument("--steps", type=int, choices=(1200, 2400))
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--probe-rows", type=int, default=4096)
    parser.add_argument("--pairs", type=int, default=2500)
    parser.add_argument("--eval-workers", type=int, default=6)
    args = parser.parse_args()
    args.study = args.study.resolve()
    if min(args.source_iteration, args.target, args.probe_rows, args.eval_workers) < 1 or args.pairs < 2:
        parser.error("positive budgets required (pairs>=2)")
    if args.target <= args.source_iteration:
        parser.error("--target must be after --source-iteration")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == "train-arm":
        if args.steps is None:
            parser.error("train-arm requires --steps")
        train_arm(args)
        return
    if (args.checkpoint is None and args.bootstrap_config is None) or args.run_root is None:
        parser.error("run requires --checkpoint or --bootstrap-config, plus --run-root")
    args.run_root = args.run_root.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.bootstrap_config is not None:
        args.bootstrap_config = args.bootstrap_config.resolve()
    if args.study.exists():
        parser.error("study already exists; inspect its status before creating a fresh study")
    # One training orchestrator per output root; independent readers are allowed.
    lock_path = args.run_root / ".stage1-training.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(args)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            publish(args.study / "status.json", {"phase": "training", "steps": 1200})
            for steps in (1200, 2400):
                publish(args.study / "status.json", {"phase": "training", "steps": steps})
                with (args.study / f"train{steps}.log").open("x") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "train-arm",
                        "--study", str(args.study), "--steps", str(steps)], stdout=log, stderr=subprocess.STDOUT, check=True)
            publish(args.study / "status.json", {"phase": "evaluation"})
            with (args.study / "evaluation.log").open("x") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/eval_cfr_stage1.py"),
                    "--study", str(args.study)], stdout=log, stderr=subprocess.STDOUT, check=True)
            publish(args.study / "status.json", {"phase": "complete", "winner": None,
                "note": "Budget study complete. Model/capacity trials and Stage2 selection are still pending."})
        except BaseException:
            publish(args.study / "status.json", {"phase": "failed", "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()

"""Measure the existing SD-CFR pipeline in a fresh, isolated output directory.

This benchmarks the configured traversal workers and full training pool,
including configured BF16/fused-optimizer options and pool upload cost.
CUDA synchronization occurs at phase boundaries, not at every operation.
No existing run is resumed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer
from alphanl.transformer.config import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cfr/hunl_200bb.json"))
    parser.add_argument("--out-dir", type=Path, required=True, help="new, empty benchmark directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--traversals", type=int, default=16, help="roots per player per iteration")
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tree-batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--inference-batch", type=int)
    parser.add_argument("--inference-wait-ms", type=float)
    parser.add_argument("--train-buffer-device", choices=("cpu", "cuda"))
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"))
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--baseline", type=Path, help="benchmark.json from the same host/configuration")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    for name in ("iterations", "traversals", "train_steps", "batch_size", "tree_batch", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    out = args.out_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--out-dir must be a new or empty directory")

    config = CFRConfig.from_json(args.config)
    config.device = resolve_device(args.device)
    config.iterations = args.iterations
    config.traversals_per_player = args.traversals
    config.traversal_batch_size = args.tree_batch
    config.train_steps = args.train_steps
    config.minibatch_size = args.batch_size
    config.out_dir = str(out)
    # A benchmark must never open a production memmap from its input config.
    config.buffer_directory = None
    for argument, field in (("workers", "traversal_workers"), ("inference_batch", "inference_batch_size"),
                            ("inference_wait_ms", "inference_wait_ms"), ("train_buffer_device", "train_buffer_device"),
                            ("amp_dtype", "amp_dtype"), ("fused_optimizer", "fused_optimizer"), ("allow_tf32", "allow_tf32")):
        value = getattr(args, argument)
        if value is not None:
            setattr(config, field, value)
    config.__post_init__()
    torch.set_num_threads(args.threads)
    cuda = torch.device(config.device).type == "cuda"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    environment = {
        "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": config.device, "torch_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "gpu": torch.cuda.get_device_name() if cuda else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with SDCFRTrainer(config) as trainer:
        sync()
        setup_s = time.perf_counter() - start
        phase_rows: list[dict] = []
        for _ in range(args.iterations):
            sync()
            began = time.perf_counter()
            metrics = trainer.run_iteration()
            sync()
            current = {key: value for key, value in trainer.last_timing.items()}
            current["iteration_s"] = time.perf_counter() - began
            collect_s = sum(current[f"p{p}_collect_s"] for p in (0, 1))
            fit_s = (sum(current[f"p{p}_fit_s"] for p in (0, 1)) if config.separate_player_models
                     else current["shared_fit_s"])
            samples = sum(metrics[f"p{p}_regret_samples"] for p in (0, 1))
            draws = (sum(metrics[f"p{p}_train_steps"] for p in (0, 1)) if config.separate_player_models
                     else metrics["shared_train_steps"]) * config.minibatch_size
            calls = sum(metrics[f"p{p}_inference_calls"] for p in (0, 1))
            rows = sum(metrics[f"p{p}_inference_rows"] for p in (0, 1))
            current.update(
                iteration=trainer.iteration,
                regret_samples_per_s=samples / collect_s,
                training_samples_per_s=draws / fit_s,
                average_inference_batch=rows / calls if calls else None,
                metrics=metrics,
            )
            phase_rows.append(current)
            print(json.dumps({k: v for k, v in current.items() if k != "metrics"}), flush=True)
        sync()
        began = time.perf_counter()
        trainer.save_checkpoint(out / "train.pt")
        sync()
        checkpoint_s = time.perf_counter() - began
        elapsed_s = time.perf_counter() - start
        fields = next(iter(trainer.buffers.values()))._schema.values()
        row_bytes = sum(dtype.itemsize * math.prod(tail) for dtype, tail in fields)
        report = {
            "environment": environment, "config": asdict(config),
            "notes": [
                "RAM reservoir is authoritative; a configured full GPU copy exists only during one player's fit.",
                "A single final checkpoint is timed separately.",
                "First iteration includes lazy initialization; subsequent iterations still reinitialize networks as configured.",
                "fit_s includes reinitialization, full-pool upload (if enabled), sampling, forward/backward and optimizer.",
                "Small fresh reservoirs do not represent random sampling from a mature, full reservoir.",
                "First collect includes spawned worker startup when configured; peak RSS is per process, not simultaneous total.",
            ],
            "setup_s": setup_s, "iterations": phase_rows,
            "checkpoint_s": checkpoint_s, "elapsed_including_setup_and_checkpoint_s": elapsed_s,
            "reservoir_bytes_per_sample": row_bytes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
            "cpu_parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "checkpoint": trainer.last_checkpoint_timing,
            "effective_options": {"amp": trainer.amp_enabled, "fused_adam": trainer.optimizer_fused,
                                  "full_gpu_pool": trainer.gpu_pool_enabled},
        }
        if baseline is not None:
            reasons = []
            for key in ("platform", "python", "torch", "torch_cuda", "device", "gpu", "torch_threads"):
                if baseline["environment"].get(key) != environment.get(key):
                    reasons.append(f"environment.{key}")
            for key in ("seed", "iterations", "traversals_per_player", "buffer_capacity", "train_steps",
                        "minibatch_size", "learning_rate", "max_grad_norm", "max_traversal_nodes",
                        "reinitialize", "fallback", "model", "separate_player_models", "position_conditioning",
                        "regret_target_scale", "loss", "huber_beta", "preflop_action_guard",
                        "guard_short_stack_bb", "guard_unlock_invested_bb", "guard_unlock_raises",
                        "spr_action_guard", "guard_max_pot_ratio"):
                if baseline["config"].get(key) != report["config"].get(key):
                    reasons.append(f"config.{key}")
            comparison = {"baseline": str(args.baseline.resolve()), "comparable": not reasons,
                          "mismatches": reasons,
                          "note": "Same host/software and root/training budgets; different collector RNG/order may change tree sizes and regrets."}
            if not reasons:
                comparison["iterations"] = [
                    {"iteration": old["iteration"],
                     "collect_wall_speedup": sum(old[f"p{p}_collect_s"] for p in (0, 1)) / sum(new[f"p{p}_collect_s"] for p in (0, 1)),
                     "regret_throughput_speedup": new["regret_samples_per_s"] / old["regret_samples_per_s"],
                     "fit_wall_speedup": (sum(old[f"p{p}_fit_s"] for p in (0, 1)) if config.separate_player_models else old["shared_fit_s"]) /
                                         (sum(new[f"p{p}_fit_s"] for p in (0, 1)) if config.separate_player_models else new["shared_fit_s"]),
                     "iteration_wall_speedup": old["iteration_s"] / new["iteration_s"],
                     "same_node_count": sum(old["metrics"][f"p{p}_nodes"] for p in (0, 1)) == sum(new["metrics"][f"p{p}_nodes"] for p in (0, 1)),
                     "same_regret_count": sum(old["metrics"][f"p{p}_regret_samples"] for p in (0, 1)) == sum(new["metrics"][f"p{p}_regret_samples"] for p in (0, 1))}
                    for old, new in zip(baseline["iterations"], phase_rows, strict=True)]
            report["comparison"] = comparison
        shutdown_started = time.perf_counter()
    report["worker_shutdown_s"] = time.perf_counter() - shutdown_started
    report["elapsed_total_s"] = time.perf_counter() - start
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {out / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()

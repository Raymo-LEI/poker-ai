"""Measure the existing SD-CFR pipeline in a fresh, isolated output directory.

This benchmarks the current algorithm and configured BF16/fused-optimizer
path. It does not enable compilation, GPU reservoirs, or multiprocessing.
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
    args = parser.parse_args()
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
    trainer = SDCFRTrainer(config)
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
        fit_s = sum(current[f"p{p}_fit_s"] for p in (0, 1))
        samples = sum(metrics[f"p{p}_regret_samples"] for p in (0, 1))
        draws = sum(metrics[f"p{p}_train_steps"] * config.minibatch_size for p in (0, 1))
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
    fields = trainer.buffers[0]._schema.values()
    row_bytes = sum(dtype.itemsize * math.prod(tail) for dtype, tail in fields)
    report = {
        "environment": environment, "config": asdict(config),
        "notes": [
            "Current pipeline; RAM reservoir; configured BF16/fused flags are included in the report.",
            "A single final checkpoint is timed separately.",
            "First iteration includes lazy initialization; subsequent iterations still reinitialize networks as configured.",
            "fit_s includes model reinitialization, CPU sampling, H2D, forward/backward and optimizer.",
            "Small fresh reservoirs do not represent random sampling from a mature, full reservoir.",
            "No CPU RSS measurement; use pidstat or /usr/bin/time -v on the server.",
        ],
        "setup_s": setup_s, "iterations": phase_rows,
        "checkpoint_s": checkpoint_s, "elapsed_including_setup_and_checkpoint_s": elapsed_s,
        "reservoir_bytes_per_sample": row_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
    }
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {out / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()

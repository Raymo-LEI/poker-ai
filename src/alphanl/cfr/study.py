"""Auditable helpers for paired training-budget experiments (no PPO changes)."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostics import regression_probe


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fixed_probe(trainer, *, rows=4096, seed=20261028):
    """Use separate RNGs; these are retained TRAINING samples, not a holdout."""
    seats = {}
    for seat in (0, 1):
        batch, target, iterations = trainer.shared_reservoir.sample(
            rows, seat=seat, rng=np.random.default_rng(np.random.SeedSequence([seed, seat])))
        seats[seat] = ({k: v.cpu() for k, v in batch.items()}, target.cpu(), iterations.cpu())
    return {"format": "sd_cfr_common_training_probe_v1", "seats": seats,
            "source_iteration": trainer.iteration, "seed": seed,
            "target_scale": trainer.config.regret_target_scale,
            "model": asdict(trainer.config.model),
            "note": "Fixed retained training rows shared across arms/iterations; NOT independent validation."}


@torch.no_grad()
def evaluate_probe(trainer, probe, *, batch_size=256):
    if asdict(trainer.config.model) != probe["model"] or trainer.config.regret_target_scale != probe["target_scale"]:
        raise ValueError("fixed probe feature/target configuration mismatch")
    model = trainer.shared_regret_net
    model.eval()
    result = {"iteration": trainer.iteration, "study/fixed_probe_source_iteration": probe["source_iteration"]}
    for seat, (batch, target, iterations) in probe["seats"].items():
        predictions = []
        for start in range(0, len(target), batch_size):
            x = {k: v[start:start + batch_size].to(trainer.device) for k, v in batch.items()}
            # FP32 measurement for both arms, outside training autocast.
            predictions.append(model(x).float().cpu())
        values = regression_probe(torch.cat(predictions), target / probe["target_scale"],
                                  batch["legal_mask"], iterations)
        for field in ("r2", "normalized_mse", "unweighted_mse"):
            value = float(values[field])
            result[f"train/fixed_probe_{field}_p{seat}"] = value if np.isfinite(value) else None
        result[f"train/fixed_probe_samples_p{seat}"] = len(target)
    return result


def freeze_prefix(trainer, destination):
    """Immutable weight-only evaluation archive. Never claimed to be restart state."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    selected = sorted(trainer.strategy_paths)
    for iteration in selected:
        source = trainer.strategy_paths[iteration]
        os.link(source, destination / source.name)
    publish(destination / "index.json", {"format": "sd_cfr_archive_v1", "iterations": selected})
    return {"path": str(destination.resolve()), "through": trainer.iteration,
            "count": len(selected), "restart_checkpoint": False}


def memory_usage():
    """Linux cgroup accounting includes worker RSS and charged page cache."""
    values = {}
    for field, name in (("ram_current_bytes", "memory.current"), ("ram_limit_bytes", "memory.max")):
        path = Path("/sys/fs/cgroup") / name
        if path.exists() and path.read_text().strip() != "max":
            values[field] = int(path.read_text().strip())
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        values.update(vram_used_bytes=total - free, vram_total_bytes=total,
                      vram_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      vram_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return values


def check_memory(*, ram_limit=90_000_000_000, vram_limit=24_000_000_000):
    values = memory_usage()
    if values.get("ram_current_bytes", 0) >= ram_limit:
        raise RuntimeError(f"study RAM safety limit exceeded: {values}")
    if values.get("vram_used_bytes", 0) >= vram_limit:
        raise RuntimeError(f"study VRAM safety limit exceeded: {values}")
    return values


@torch.no_grad()
def latency_probe(model, batch, *, repeats=100):
    """Single-thread, batch=1 FP32 strategy latency, excluding snapshot I/O."""
    if torch.get_num_threads() != 1 or len(batch["token_ids"]) != 1:
        raise ValueError("latency probe requires one torch thread and batch size one")
    model.eval()
    for _ in range(10):
        model.strategy(batch)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.strategy(batch)
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.quantile(elapsed, .5)), "p95_ms": float(np.quantile(elapsed, .95)),
            "p99_ms": float(np.quantile(elapsed, .99)), "max_ms": max(elapsed),
            "repeats": repeats, "threads": 1, "batch_size": 1,
            "scope": "FP32 model.strategy only; excludes tokenizer, archive I/O and process startup"}

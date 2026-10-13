"""Training checkpoints: model, optimizer, GradScaler, opponent pool, update, all RNG states,
and (optionally) the self-play tables, so a resumed run continues the same rollouts.

Writes are atomic (``<name>.tmp`` then ``os.replace``) and only the newest ``keep_last``
``update_*.pt`` files are kept.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .pool import OpponentPool

FORMAT_VERSION = 2
_CKPT_RE = re.compile(r"^update_(\d+)\.pt$")


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


def checkpoint_path(ckpt_dir: Path | str, update: int) -> Path:
    return Path(ckpt_dir) / f"update_{int(update):07d}.pt"


def list_checkpoints(ckpt_dir: Path | str) -> List[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in ckpt_dir.iterdir() if (m := _CKPT_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(ckpt_dir: Path | str) -> Optional[Path]:
    found = list_checkpoints(ckpt_dir)
    return found[-1] if found else None


def prune_checkpoints(ckpt_dir: Path | str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    for old in list_checkpoints(ckpt_dir)[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    update: int,
    config: Dict[str, Any],
    pool: Optional[OpponentPool] = None,
    scaler: Optional["torch.amp.GradScaler"] = None,
    envs: Optional[Sequence[Any]] = None,
    actor_rng: Optional[Dict[str, Any]] = None,
    rng: Optional[Dict[str, Any]] = None,
    keep_last: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": FORMAT_VERSION,
        "update": int(update),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": config,
        "pool": None if pool is None else pool.state_dict(),
        "rng": capture_rng_state() if rng is None else rng,
        "envs": None if envs is None else [env.state_dict() for env in envs],
        "actor_rng": actor_rng,
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(blob, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)
    if keep_last > 0:
        prune_checkpoints(path.parent, keep_last)
    return path


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    pool: Optional[OpponentPool] = None,
    map_location: str = "cpu",
    scaler: Optional["torch.amp.GradScaler"] = None,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    fmt = int(blob.get("format", 0))
    if fmt not in (1, FORMAT_VERSION):
        raise ValueError(f"unsupported checkpoint format: {blob.get('format')}")
    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if pool is not None and blob.get("pool") is not None:
        pool.load_state_dict(blob["pool"])
    if scaler is not None and blob.get("scaler"):
        scaler.load_state_dict(blob["scaler"])
    if restore_rng:
        restore_rng_state(blob.get("rng"))
    return blob

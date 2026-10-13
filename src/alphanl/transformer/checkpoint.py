"""Save and load a training snapshot (model, optimizer, opponent pool, update)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .pool import OpponentPool

FORMAT_VERSION = 1


def save_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    update: int,
    config: Dict[str, Any],
    pool: Optional[OpponentPool] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": FORMAT_VERSION,
        "update": int(update),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "config": config,
        "pool": None if pool is None else pool.state_dict(),
    }
    torch.save(blob, path)


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    pool: Optional[OpponentPool] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    if int(blob.get("format", 0)) != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format: {blob.get('format')}")
    model.load_state_dict(blob["model"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if pool is not None and blob.get("pool") is not None:
        pool.load_state_dict(blob["pool"])
    return blob

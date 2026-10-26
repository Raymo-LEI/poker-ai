"""Standalone SD-CFR configuration; existing PPO presets are unchanged."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alphanl.transformer.config import ModelConfig


@dataclass
class CFRConfig:
    seed: int = 1
    device: str = "cpu"
    iterations: int = 1000
    traversals_per_player: int = 128
    traversal_batch_size: int = 32
    buffer_capacity: int = 2_000_000
    buffer_directory: str | None = None
    train_steps: int = 512
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    # Deep CFR fits each player's accumulated advantage memory from scratch.
    reinitialize: bool = True
    fallback: str = "uniform"
    snapshot_every: int = 1
    checkpoint_every: int = 10
    out_dir: str = "runs/sd_cfr_hunl"
    max_traversal_nodes: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        for name in ("iterations", "traversals_per_player", "traversal_batch_size",
                     "buffer_capacity", "train_steps", "minibatch_size",
                     "snapshot_every", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning rate and gradient norm must be positive")
        if self.fallback not in ("uniform", "argmax"):
            raise ValueError("fallback must be uniform or argmax")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        if self.max_traversal_nodes is not None and self.max_traversal_nodes < 1:
            raise ValueError("max_traversal_nodes must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "CFRConfig":
        return cls(**json.loads(Path(path).read_text()))

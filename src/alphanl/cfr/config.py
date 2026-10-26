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
    traversal_workers: int = 0
    inference_batch_size: int = 256
    inference_wait_ms: float = 1.0
    worker_sample_batch: int = 128
    worker_timeout_s: float = 120.0
    buffer_capacity: int = 2_000_000
    buffer_directory: str | None = None
    train_steps: int = 512
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    # GPU-only training options. Defaults preserve the original FP32/Adam path.
    amp_dtype: str = "none"
    fused_optimizer: bool = False
    allow_tf32: bool = False
    train_buffer_device: str = "cpu"
    gpu_pool_reserve_bytes: int = 2 * 1024 ** 3
    # Deep CFR fits each player's accumulated advantage memory from scratch.
    reinitialize: bool = True
    fallback: str = "uniform"
    snapshot_every: int = 1
    checkpoint_every: int = 10
    checkpoint_min_free_bytes: int = 1024 ** 3
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
        for name in ("inference_batch_size", "worker_sample_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.traversal_workers, bool) or not isinstance(self.traversal_workers, int)
                or self.traversal_workers < 0):
            raise ValueError("traversal_workers must be a nonnegative integer")
        if self.traversal_workers and self.inference_batch_size < self.traversal_batch_size:
            raise ValueError("inference_batch_size must cover one worker tree batch")
        if not 0 <= self.inference_wait_ms <= 1000:
            raise ValueError("inference_wait_ms must be between 0 and 1000")
        if not 0 < self.worker_timeout_s < float("inf"):
            raise ValueError("worker_timeout_s must be positive and finite")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning rate and gradient norm must be positive")
        if self.fallback not in ("uniform", "argmax"):
            raise ValueError("fallback must be uniform or argmax")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        if self.amp_dtype not in ("none", "bfloat16"):
            raise ValueError("amp_dtype must be 'none' or 'bfloat16'")
        if self.train_buffer_device not in ("cpu", "cuda"):
            raise ValueError("train_buffer_device must be cpu or cuda")
        if (isinstance(self.gpu_pool_reserve_bytes, bool) or not isinstance(self.gpu_pool_reserve_bytes, int)
                or self.gpu_pool_reserve_bytes < 0):
            raise ValueError("gpu_pool_reserve_bytes must be a nonnegative integer")
        if not isinstance(self.fused_optimizer, bool) or not isinstance(self.allow_tf32, bool):
            raise ValueError("fused_optimizer and allow_tf32 must be booleans")
        if self.max_traversal_nodes is not None and self.max_traversal_nodes < 1:
            raise ValueError("max_traversal_nodes must be positive")
        if (isinstance(self.checkpoint_min_free_bytes, bool)
                or not isinstance(self.checkpoint_min_free_bytes, int)
                or self.checkpoint_min_free_bytes < 0):
            raise ValueError("checkpoint_min_free_bytes must be a nonnegative integer")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "CFRConfig":
        return cls(**json.loads(Path(path).read_text()))

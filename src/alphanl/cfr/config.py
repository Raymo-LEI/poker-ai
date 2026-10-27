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
    train_steps: int = 1200
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    # CUDA acceleration; CPU transparently retains FP32 and ordinary Adam.
    amp_dtype: str = "bfloat16"
    fused_optimizer: bool = True
    allow_tf32: bool = True
    train_buffer_device: str = "cpu"
    gpu_pool_reserve_bytes: int = 2 * 1024 ** 3
    # Warm-start by default; reinitialization remains an explicit ablation.
    reinitialize: bool = False
    separate_player_models: bool = False
    position_conditioning: bool = True
    # Targets arrive in BB, NOT chips. A 200BB outlier becomes 2.0.
    regret_target_scale: float = 100.0
    loss: str = "huber"
    huber_beta: float = 1.0
    fallback: str = "softmax"
    preflop_action_guard: bool = True
    # New runs use the SPR guard on every street. Old archives opt out on load.
    spr_action_guard: bool = True
    guard_max_pot_ratio: float = 3.0
    guard_short_stack_bb: float = 20.0
    guard_unlock_invested_bb: float = 15.0
    guard_unlock_raises: int = 2  # open + 3-bet already in the history
    snapshot_every: int = 1
    checkpoint_every: int = 10
    checkpoint_min_free_bytes: int = 1024 ** 3
    checkpoint_archive_from: int | None = None
    checkpoint_archive_every: int = 10
    checkpoint_archive_min_free_bytes: int = 2 * 1024 ** 3
    checkpoint_archive_extra_count: int = 0
    out_dir: str = "runs/sd_cfr_hunl"
    max_traversal_nodes: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        for name in ("iterations", "traversals_per_player", "traversal_batch_size",
                     "buffer_capacity", "train_steps", "minibatch_size",
                     "snapshot_every", "checkpoint_every", "checkpoint_archive_every"):
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
        if self.fallback not in ("uniform", "argmax", "softmax"):
            raise ValueError("fallback must be uniform, argmax or softmax")
        import math
        for name in ("regret_target_scale", "huber_beta", "guard_short_stack_bb", "guard_unlock_invested_bb", "guard_max_pot_ratio"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.loss not in ("huber", "mse"):
            raise ValueError("loss must be huber or mse")
        if not self.separate_player_models and self.minibatch_size % 2:
            raise ValueError("shared training requires an even minibatch_size for exact 1:1 seat sampling")
        if not self.separate_player_models and not self.position_conditioning:
            raise ValueError("shared training requires explicit hero position conditioning")
        for name in ("separate_player_models", "position_conditioning", "preflop_action_guard", "spr_action_guard", "reinitialize"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if isinstance(self.guard_unlock_raises, bool) or not isinstance(self.guard_unlock_raises, int) or self.guard_unlock_raises < 1:
            raise ValueError("guard_unlock_raises must be a positive integer")
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
        if self.checkpoint_archive_from is not None and (isinstance(self.checkpoint_archive_from, bool)
                or not isinstance(self.checkpoint_archive_from, int) or self.checkpoint_archive_from < 1):
            raise ValueError("checkpoint_archive_from must be positive or None")
        for name in ("checkpoint_archive_min_free_bytes", "checkpoint_archive_extra_count"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "CFRConfig":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_checkpoint(cls, data: dict) -> "CFRConfig":
        """Old archives retain their original encoding, units and action tree."""
        data = dict(data)
        data.setdefault("spr_action_guard", False)
        if "separate_player_models" not in data:
            data.update(separate_player_models=True, position_conditioning=False,
                        regret_target_scale=1.0, loss="mse", huber_beta=1.0,
                        preflop_action_guard=False)
        for key, value in (("amp_dtype", "none"), ("fused_optimizer", False), ("allow_tf32", False)):
            data.setdefault(key, value)
        return cls(**data)

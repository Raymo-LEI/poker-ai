"""JSON configs for the laptop 8GB default and the larger Colab run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

T = TypeVar("T")


def _fill(cls: Type[T], payload: Mapping[str, Any]) -> T:
    known = {item.name for item in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise KeyError(f"{cls.__name__} got unknown keys: {sorted(unknown)}")
    defaults = cls()  # type: ignore[call-arg]
    data = asdict(defaults)
    data.update(payload)
    return cls(**data)  # type: ignore[call-arg]


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 64


@dataclass
class PPOConfig:
    lr: float = 3.0e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 128
    num_envs: int = 8
    adam_eps: float = 1.0e-5
    # T-C2: >0 = multiprocess rollout ('spawn'), each worker owns a contiguous slice of
    # envs with 1 torch thread. 0 = single-process path (unchanged). Needs env.per_env_rng.
    rollout_workers: int = 0
    # T-C2: seconds to wait for any single worker reply before raising (never hang).
    rollout_worker_timeout_s: float = 600.0


@dataclass
class PoolConfig:
    max_size: int = 8
    selfplay_prob: float = 0.25
    random_prob: float = 0.1
    snapshot_every: int = 10
    sampling: str = "pfsp"
    # T-C2: buffer pool.record() during a rollout, apply after it (PFSP win rates frozen
    # within one rollout). Implied by ppo.rollout_workers > 0; set it with
    # rollout_workers=0 for the single-process reference of the workers path.
    defer_record: bool = False


@dataclass
class TrainConfig:
    num_updates: int = 100000
    checkpoint_every: int = 50
    log_every: int = 1
    eval_every: int = 25
    eval_hands: int = 20
    out_dir: str = "runs/transformer_laptop"
    amp: bool = False
    checkpoint_dir: str = "runs/ckpt"
    keep_last: int = 3


@dataclass
class EnvConfig:
    backend: str = "rlcard"
    big_blind: float = 2.0
    small_blind: float = 1.0
    starting_stack: float = 100.0
    # T-C1c: each table gets its own RNG streams spawned from SeedSequence(seed):
    # a DEAL stream (deck, hero seat, opponent pick) and a SAMPLE stream (hero and
    # opponent action sampling). Draw order then no longer depends on batching.
    per_env_rng: bool = False
    # T-C1c: old global-RNG sampling, but decks / seats / opponent picks from the
    # per-table deal stream (identical cards to per_env_rng, for duplicate eval).
    shared_deal_stream: bool = False


@dataclass
class ExperimentConfig:
    name: str = "laptop_8gb"
    seed: int = 1
    device: str = "auto"
    rollout_device: str = "cpu"
    # ``transformer`` is the step-3 token encoder. ``alphanl_conv`` is the
    # PyTorch port of TF1.15 ``NlHoldemNet`` (the control group). PPO, the
    # 5-way mask, and this training script stay the same either way.
    encoder: str = "transformer"
    model: ModelConfig = None  # type: ignore[assignment]
    ppo: PPOConfig = None  # type: ignore[assignment]
    pool: PoolConfig = None  # type: ignore[assignment]
    train: TrainConfig = None  # type: ignore[assignment]
    env: EnvConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = ModelConfig()
        elif isinstance(self.model, Mapping):
            self.model = _fill(ModelConfig, self.model)
        if self.ppo is None:
            self.ppo = PPOConfig()
        elif isinstance(self.ppo, Mapping):
            self.ppo = _fill(PPOConfig, self.ppo)
        if self.pool is None:
            self.pool = PoolConfig()
        elif isinstance(self.pool, Mapping):
            self.pool = _fill(PoolConfig, self.pool)
        if self.train is None:
            self.train = TrainConfig()
        elif isinstance(self.train, Mapping):
            self.train = _fill(TrainConfig, self.train)
        if self.env is None:
            self.env = EnvConfig()
        elif isinstance(self.env, Mapping):
            self.env = _fill(EnvConfig, self.env)
        if self.encoder not in ("transformer", "alphanl_conv"):
            raise ValueError("encoder must be 'transformer' or 'alphanl_conv'")
        if self.encoder == "transformer" and self.model.d_model % self.model.nhead != 0:
            raise ValueError("model.d_model must be divisible by model.nhead")
        if self.env.backend not in ("rlcard", "pokerkit"):
            raise ValueError("env.backend must be 'rlcard' or 'pokerkit'")
        if self.ppo.minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")
        if self.ppo.num_envs < 1 or self.ppo.rollout_steps < 1:
            raise ValueError("num_envs and rollout_steps must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        known = {item.name for item in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise KeyError(f"ExperimentConfig got unknown keys: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path | str) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    def apply_overrides(
        self,
        *,
        entropy_coef: Optional[float] = None,
        num_updates: Optional[int] = None,
        out_dir: Optional[str] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        amp: Optional[bool] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: Optional[int] = None,
        encoder: Optional[str] = None,
    ) -> "ExperimentConfig":
        if encoder is not None:
            self.encoder = str(encoder)
            if self.encoder not in ("transformer", "alphanl_conv"):
                raise ValueError("encoder must be 'transformer' or 'alphanl_conv'")
        if amp is not None:
            self.train.amp = bool(amp)
        if checkpoint_dir is not None:
            self.train.checkpoint_dir = str(checkpoint_dir)
        if checkpoint_every is not None:
            self.train.checkpoint_every = int(checkpoint_every)
        if entropy_coef is not None:
            self.ppo.entropy_coef = float(entropy_coef)
        if num_updates is not None:
            self.train.num_updates = int(num_updates)
        if out_dir is not None:
            self.train.out_dir = str(out_dir)
        if device is not None:
            self.device = str(device)
        if seed is not None:
            self.seed = int(seed)
        return self


def resolve_device(name: str) -> str:
    if name == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {name!r}")
    return name

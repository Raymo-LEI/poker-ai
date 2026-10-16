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
    # Reward units, not a ratio: 0.2 is 0.2bb on rlcard/pokerkit and 40bb on
    # botzone. ``None`` disables value clipping.
    value_clip_eps: Optional[float] = None
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 128
    num_envs: int = 8
    adam_eps: float = 1.0e-5


@dataclass
class PoolConfig:
    max_size: int = 8
    selfplay_prob: float = 0.25
    random_prob: float = 0.1
    snapshot_every: int = 10
    sampling: str = "pfsp"


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
    # Policy-only. False keeps the historical mask (fold stays available on a
    # free check). True removes fold from the policy mask when nothing is
    # owed and check/call is legal. Environment legality is unchanged.
    # Checkpoints written before this field existed mean false.
    mask_free_fold: bool = False


# rlcard / the historical SelfPlayEnv signature. Botzone simple is not this game.
_RLCARD_LIMITS = (1.0, 2.0, 100.0)


def resolve_env_limits(
    backend: str,
    small_blind: float,
    big_blind: float,
    starting_stack: float,
) -> tuple:
    """Blinds and stack for ``backend``.

    ``botzone`` plays the wiki simple game (50 / 100, both seats start at
    20000). Passing that triple, or omitting it and leaving the rlcard
    1 / 2 / 100 defaults, selects those blinds. Any other triple is an
    error: :class:`alphanl.botzone.table.BotzoneTable` cannot post them.
    rlcard and pokerkit are returned unchanged.
    """

    current = (float(small_blind), float(big_blind), float(starting_stack))
    if str(backend) != "botzone":
        return current
    from alphanl.botzone.legal import BIG_BLIND, SMALL_BLIND, STARTING_STACK

    botzone = (float(SMALL_BLIND), float(BIG_BLIND), float(STARTING_STACK))
    if current == botzone or current == _RLCARD_LIMITS:
        return botzone
    raise ValueError(
        "botzone requires small_blind=50, big_blind=100, starting_stack=20000, "
        f"got small_blind={small_blind}, big_blind={big_blind}, starting_stack={starting_stack}"
    )


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
    # Documented path of a behavior-cloning checkpoint for a later PPO run.
    # PPO does not load this field. Pass the same path as ``--init-weights``
    # (with ``--allow-cross-backend-init`` on botzone) or as ``--resume``.
    # ``None`` on every config that is not that continuation.
    bc_init_checkpoint: Optional[str] = None
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
        if self.env.backend not in ("rlcard", "pokerkit", "botzone"):
            raise ValueError("env.backend must be 'rlcard', 'pokerkit', or 'botzone'")
        small_blind, big_blind, starting_stack = resolve_env_limits(
            self.env.backend,
            self.env.small_blind,
            self.env.big_blind,
            self.env.starting_stack,
        )
        self.env.small_blind = small_blind
        self.env.big_blind = big_blind
        self.env.starting_stack = starting_stack
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
        keep_last: Optional[int] = None,
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
        if keep_last is not None:
            self.train.keep_last = int(keep_last)
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

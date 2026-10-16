"""Training checkpoints: model, optimizer, GradScaler, opponent pool, update, all RNG states,
and (optionally) the self-play tables, so a resumed run continues the same rollouts.

Writes are atomic (``<name>.tmp`` in the same directory, flush, fsync, then ``os.replace``)
and only the newest ``keep_last`` ``update_*.pt`` files are kept. A kill during the temp
write leaves the previous ``update_*.pt`` untouched.

``--resume auto`` uses :func:`latest_complete_checkpoint`, which skips unreadable files.
``config_hash`` is a sha256 of the config dict. ``env_rngs`` is the per-env deal RNG
(and the button cursor the next ``reset`` needs). Mid-hand cards are still stored in
``envs`` for ``--resume latest``; ``--resume auto`` ignores that hand and resets.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .pool import OpponentPool

FORMAT_VERSION = 2
_CKPT_RE = re.compile(r"^update_(\d+)\.pt$")


class ConfigMismatchError(ValueError):
    """``--resume auto`` refused because the config hash does not match."""

    exit_code = 2


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


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 of a config dict. Key order does not matter."""

    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> str:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"config hash cannot encode {type(obj).__name__}")


def _diff_config(saved: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    diffs: List[str] = []

    def walk(left: Any, right: Any, prefix: str) -> None:
        if len(diffs) >= 8:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key, "<missing>"), right.get(key, "<missing>"), f"{prefix}.{key}" if prefix else str(key))
            return
        if left != right:
            diffs.append(f"{prefix}: checkpoint={left!r} current={right!r}")

    walk(dict(saved), dict(current), "")
    if not diffs:
        return "(hash differs; no field-level difference was found)"
    return "; ".join(diffs)


def assert_resume_config(
    saved_config: Optional[Mapping[str, Any]],
    saved_hash: Optional[str],
    current_config: Mapping[str, Any],
    *,
    allow: bool,
    path: Path | str,
) -> None:
    """Refuse ``--resume auto`` when the checkpoint hash is not this config.

    Checkpoints written before ``config_hash`` existed are accepted. ``allow``
    is the ``--allow-config-mismatch`` override and is off unless the caller
    passes it.
    """

    if allow or not saved_hash:
        return
    current = config_hash(current_config)
    if current == saved_hash:
        return
    detail = _diff_config(saved_config or {}, current_config)
    raise ConfigMismatchError(
        "refusing --resume auto: config hash does not match "
        f"{path}. checkpoint={saved_hash} current={current}. "
        f"Differences: {detail}. "
        "Pass --allow-config-mismatch to resume anyway."
    )


def _checkpoint_readable(path: Path) -> bool:
    try:
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
    except Exception:
        return False
    if not isinstance(blob, dict) or "model" not in blob or "update" not in blob:
        return False
    try:
        fmt = int(blob.get("format", 0))
        int(blob["update"])
    except (TypeError, ValueError):
        return False
    return fmt in (1, FORMAT_VERSION)


def latest_complete_checkpoint(ckpt_dir: Path | str, *, verbose: bool = True) -> Optional[Path]:
    """Newest ``update_*.pt`` that ``torch.load`` can read.

    Unreadable or half-written files are skipped. ``*.tmp`` is not a
    checkpoint name, so a kill during :func:`save_checkpoint` does not hide
    the previous complete file. Returns ``None`` when every file is bad or
    the directory is empty. ``verbose=False`` skips the stdout note so a
    launcher can read the result without an extra log line.
    """

    chosen: Optional[Path] = None
    skipped: List[Path] = []
    for path in reversed(list_checkpoints(ckpt_dir)):
        if _checkpoint_readable(path):
            chosen = path
            break
        skipped.append(path)
    if verbose:
        for path in skipped:
            print(f"[resume] skipping unreadable checkpoint {path}")
    return chosen


def choose_launch_mode(ckpt_dir: Path | str) -> str:
    """``resume`` if any complete PPO checkpoint exists, otherwise ``fresh``.

    The readability check is :func:`latest_complete_checkpoint`, the same one
    ``--resume auto`` uses. A corrupt newest file with an older complete file
    is still ``resume`` (auto falls back). An empty directory, or a directory
    whose ``update_*.pt`` files are all unreadable, is ``fresh``.
    """

    if latest_complete_checkpoint(ckpt_dir, verbose=False) is None:
        return "fresh"
    return "resume"


def ppo_launch_args(ckpt_dir: Path | str, bc_checkpoint: Path | str) -> List[str]:
    """Flags for one autoresume launch. Never mixes resume and init-weights.

    ``fresh`` loads BC weights only (``--init-weights`` plus
    ``--allow-cross-backend-init``) and does not pass ``--resume``.
    ``resume`` passes only ``--resume auto``.
    """

    if choose_launch_mode(ckpt_dir) == "resume":
        return ["--resume", "auto"]
    return ["--init-weights", str(bc_checkpoint), "--allow-cross-backend-init"]


def capture_env_rngs(envs: Sequence[Any]) -> List[Dict[str, Any]]:
    """Per-env RNGs, not the in-progress hand.

    ``env`` is the numpy generator that picks the hero seat. ``table`` is
    whatever :meth:`rng_state` the backend exposes (deck RNG, and the button
    cursor the next ``reset`` consumes).
    """

    snaps: List[Dict[str, Any]] = []
    for env in envs:
        snap: Dict[str, Any] = {"env": env.rng.bit_generator.state}
        table = getattr(env, "table", None)
        if table is not None and hasattr(table, "rng_state"):
            snap["table"] = table.rng_state()
        snaps.append(snap)
    return snaps


def _detensor(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {key: _detensor(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_detensor(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_detensor(value) for value in obj)
    return obj


def restore_env_rngs(envs: Sequence[Any], snaps: Optional[Sequence[Mapping[str, Any]]]) -> None:
    """Put each env's deal RNG back. Does not deal and does not load the hand."""

    if snaps is None:
        return
    if len(snaps) != len(envs):
        raise ValueError(
            f"checkpoint has {len(snaps)} env RNG snapshots, this run has {len(envs)} envs"
        )
    for env, snap in zip(envs, snaps):
        state = snap.get("env")
        if state is not None:
            env.rng.bit_generator.state = _detensor(state)
        table_state = snap.get("table")
        table = getattr(env, "table", None)
        if table_state is not None and table is not None and hasattr(table, "set_rng_state"):
            table.set_rng_state(_detensor(table_state))


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
    digest = config_hash(config) if isinstance(config, Mapping) else None
    blob = {
        "format": FORMAT_VERSION,
        "update": int(update),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "config": config,
        "config_hash": digest,
        "pool": None if pool is None else pool.state_dict(),
        "rng": capture_rng_state() if rng is None else rng,
        "envs": None if envs is None else [env.state_dict() for env in envs],
        "env_rngs": None if envs is None else capture_env_rngs(envs),
        "actor_rng": actor_rng,
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        # Test-only. Unset in normal training, so the bytes written below are
        # the same path as before. A kill here never calls os.replace.
        if os.environ.get("ALPHANL_CKPT_FAULT") == "kill-during-write":
            handle.write(b"PARTIAL-CHECKPOINT")
            handle.flush()
            os.fsync(handle.fileno())
            os.kill(os.getpid(), signal.SIGKILL)
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

"""Bounded-memory reservoir checkpoints with immutable disk generations.

The caller gives each checkpoint its own fresh generation directory. Disk
reservoirs are copied as compact retained prefixes, never referenced through
their mutable live files and never materialized as full NumPy arrays. RAM
reservoirs retain the portable embedded state dictionary format.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import re
import warnings
import os
from contextlib import contextmanager
import fcntl
from typing import Any

import numpy as np
import torch

from .reservoir import ReservoirBuffer


COPY_CHUNK_BYTES = 8 * 1024 * 1024
DISK_STORAGE = "memmap_snapshot_v1"


@contextmanager
def checkpoint_lock(parent: Path):
    """Serialize publishers/GC in one run (Linux/macOS training hosts)."""
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".checkpoint.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _references(payload: dict, parent: Path) -> set[Path]:
    """Validate all external references before authorizing collection."""
    paths = list(payload.get("strategy_archive", {}).values())
    paths += [payload[key] for key in ("metrics_path", "performance_path") if payload.get(key)]
    for buffer in payload.get("buffers", {}).values():
        if buffer.get("storage") == DISK_STORAGE:
            directory = (parent / buffer["directory"]).resolve()
            metadata = buffer["metadata"]
            if json.loads((directory / "metadata.json").read_text()) != metadata:
                raise ValueError("checkpoint reservoir metadata mismatch")
            for key, field in metadata["schema"].items():
                field_path = directory / f"{key}.dat"
                expected = metadata["size"] * int(np.prod(field["shape"])) * np.dtype(field["dtype"]).itemsize
                if field_path.stat().st_size != expected:
                    raise ValueError(f"checkpoint reservoir size mismatch: {field_path}")
            paths.append(str(directory))
    references = set()
    data = (parent / "checkpoint_data").resolve()
    for filename in paths:
        target = (parent / filename).resolve()
        if not target.exists():
            raise ValueError(f"missing checkpoint reference: {target}")
        relative = target.relative_to(data)
        references.add(data / relative.parts[0])
    return references


def _fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_references(path: Path, payload: dict) -> None:
    """Small sidecar avoids repeatedly loading multi-GB embedded RAM pools."""
    refs = _references(payload, path.parent)
    sidecar = path.with_name(path.name + ".refs.json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"fingerprint": _fingerprint(path),
                                     "generations": [p.name for p in sorted(refs)]}))
    temporary.replace(sidecar)


def prune_generations(path: Path) -> None:
    """Collect only unreferenced generations, failing closed on unreadable refs.

    Checkpoints live beside checkpoint_data. Discover legacy .pt/.pth/.ckpt
    files once; subsequent saves use fingerprinted, small reference sidecars.
    Copies of a checkpoint in this directory retain their generations too.
    A whole run must be copied with checkpoint_data when moved elsewhere.
    """
    parent = path.parent
    data = parent / "checkpoint_data"
    protected = set()
    candidates = {p for pattern in ("*.pt", "*.pth", "*.ckpt") for p in parent.glob(pattern)}
    candidates.add(path)
    candidates.update(parent / p.name.removesuffix(".refs.json") for p in parent.glob("*.refs.json")
                      if (parent / p.name.removesuffix(".refs.json")).is_file())
    try:
        for checkpoint in candidates:
            sidecar = checkpoint.with_name(checkpoint.name + ".refs.json")
            cached = json.loads(sidecar.read_text()) if sidecar.exists() else None
            if cached is None or cached["fingerprint"] != _fingerprint(checkpoint):
                blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if blob.get("format") != "sd_cfr_train_v2":
                    raise ValueError(f"unrecognized checkpoint beside checkpoint_data: {checkpoint}")
                publish_references(checkpoint, blob)
                cached = json.loads(sidecar.read_text())
                del blob
            for name in cached["generations"]:
                if not re.fullmatch(r"iteration_\d{7,}_[0-9a-f]{32}", name):
                    raise ValueError("invalid generation reference")
                if not (data / name).is_dir():
                    raise ValueError("referenced checkpoint generation is missing")
                protected.add(name)
    except Exception as exc:
        warnings.warn(f"checkpoint published; generation cleanup skipped: {exc}", RuntimeWarning)
        return
    for generation in data.iterdir():
        if (re.fullmatch(r"iteration_\d{7,}_[0-9a-f]{32}", generation.name)
                and generation.name not in protected and generation.is_dir() and not generation.is_symlink()):
            shutil.rmtree(generation)


def _field_elements(size: int, tail: tuple[int, ...]) -> int:
    return size * int(np.prod(tail, dtype=np.int64))


def _chunk_elements(dtype: np.dtype) -> int:
    if COPY_CHUNK_BYTES < dtype.itemsize:
        raise ValueError("copy chunk size must accommodate one scalar")
    return COPY_CHUNK_BYTES // dtype.itemsize


def save_buffer(buffer: ReservoirBuffer, directory: str | Path) -> dict[str, Any]:
    """Save a portable RAM state or an immutable compact disk snapshot.

    ``directory`` must not exist for a disk snapshot. Its parent may exist;
    the trainer should put it beneath a unique checkpoint generation. Returned
    disk paths are absolute so callers may explicitly relativize them against
    their checkpoint file for moving a complete run to another machine.
    """
    if buffer.directory is None:
        return buffer.state_dict()
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=False)
    try:
        buffer.flush()
        metadata = buffer.state_dict(include_data=False)
        for key, (dtype, _) in buffer._schema.items():
            source = buffer._maps[key][:len(buffer)].reshape(-1)
            field_path = target / f"{key}.dat"
            with field_path.open("wb") as handle:
                step = _chunk_elements(dtype)
                for start in range(0, source.size, step):
                    chunk = np.array(source[start:start + step], copy=True)
                    chunk.tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            field_path.chmod(0o444)
        metadata_path = target / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        metadata_path.chmod(0o444)
        with metadata_path.open("rb") as handle:
            os.fsync(handle.fileno())
        sync_directory(target)
    except BaseException:
        # Only remove the fresh generation created by this call.
        shutil.rmtree(target)
        raise
    return {"storage": DISK_STORAGE, "directory": str(target), "metadata": metadata}


def _write_memory_chunk(buffer: ReservoirBuffer, key: str, offset: int,
                        values: np.ndarray) -> None:
    """Write a flat input chunk across independently allocated RAM chunks."""
    _, tail = buffer._schema[key]
    elements_per_chunk = _field_elements(buffer.chunk_size, tail)
    consumed = 0
    while consumed < len(values):
        chunk_id, local = divmod(offset + consumed, elements_per_chunk)
        destination = buffer._chunks[chunk_id][key].reshape(-1)
        count = min(len(values) - consumed, destination.size - local)
        destination[local:local + count] = values[consumed:consumed + count]
        consumed += count


def restore_buffer(buffer: ReservoirBuffer, payload: dict[str, Any]) -> None:
    """Restore embedded RAM data or stream an immutable disk snapshot.

    Disk snapshots may restore into either a live memmap or a RAM reservoir.
    The latter necessarily allocates the retained data, but the copy itself
    still uses bounded temporary memory. Resolve any relative reference paths
    against the training checkpoint before calling this function.
    """
    if "storage" not in payload:
        buffer.load_state_dict(payload)
        return
    if payload["storage"] != DISK_STORAGE:
        raise ValueError("unsupported reservoir checkpoint storage")
    source_dir = Path(payload["directory"])
    if not source_dir.is_absolute():
        raise ValueError("resolve reservoir snapshot paths against their checkpoint first")
    source_dir = source_dir.resolve()
    if source_dir == buffer.directory:
        raise ValueError("immutable reservoir snapshots must be separate from live files")
    metadata = payload["metadata"]
    buffer._validate_signature(metadata)
    buffer._validate_counters(metadata)
    if json.loads((source_dir / "metadata.json").read_text()) != metadata:
        raise ValueError("reservoir snapshot metadata does not match its checkpoint")
    size = int(metadata["size"])
    for key, (dtype, tail) in buffer._schema.items():
        path = source_dir / f"{key}.dat"
        expected = _field_elements(size, tail) * dtype.itemsize
        if not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"reservoir snapshot field has missing or invalid size: {path}")
    if buffer.directory is None:
        buffer._chunks = []
        if size:
            buffer._ensure_slot(size - 1)
    for key, (dtype, tail) in buffer._schema.items():
        count = _field_elements(size, tail)
        destination = buffer._maps[key].reshape(-1) if buffer.directory is not None else None
        with (source_dir / f"{key}.dat").open("rb") as handle:
            step = _chunk_elements(dtype)
            for start in range(0, count, step):
                chunk = np.fromfile(handle, dtype=dtype, count=min(step, count - start))
                if chunk.size != min(step, count - start):
                    raise ValueError(f"reservoir snapshot field was truncated during restore: {key}")
                if destination is None:
                    _write_memory_chunk(buffer, key, start, chunk)
                else:
                    destination[start:start + chunk.size] = chunk
    buffer.size = size
    buffer.seen = int(metadata["seen"])
    buffer._rng.bit_generator.state = copy.deepcopy(metadata["rng_state"])
    buffer.flush()

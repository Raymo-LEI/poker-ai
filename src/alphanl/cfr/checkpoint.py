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
from typing import Any

import numpy as np

from .reservoir import ReservoirBuffer


COPY_CHUNK_BYTES = 8 * 1024 * 1024
DISK_STORAGE = "memmap_snapshot_v1"


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
            field_path.chmod(0o444)
        metadata_path = target / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        metadata_path.chmod(0o444)
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

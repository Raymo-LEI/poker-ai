"""Uniform infoset reservoir with lazy RAM chunks or persistent NumPy memmaps.

Every traverser infoset has the same retention probability, irrespective of
its CFR iteration. The trainer applies linear iteration weighting to the
sampled loss instead. Sampling minibatches is with replacement.

``directory`` opts into disk storage: arrays have their full logical size but
only accessed pages need resident RAM. Call :meth:`flush` at a consistent
training boundary and reopen the same directory to continue, including RNG
state. A full ``state_dict()`` is a portable snapshot; it copies all retained
data and can be large. Metadata-only state dictionaries require a separate,
matching snapshot of the disk files, not a subsequently modified directory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphanl.transformer.schema import NUM_ACTIONS
from alphanl.transformer.tokenizer import NUM_NUMERIC, TensorBatch


class ReservoirBuffer:
    """Algorithm R over batched infosets, retaining FP32 numerical features.

    ``iteration`` starts at one. IDs use compact integer storage and are
    restored as PyTorch ``long`` tensors. Illegal regret targets are zeroed.
    The buffer is single-writer; callers must synchronize parallel collectors.
    """

    VERSION = 1
    _BATCH_KEYS = ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")

    def __init__(
        self,
        capacity: int,
        max_seq_len: int,
        *,
        seed: int = 1,
        directory: str | Path | None = None,
        chunk_size: int = 4096,
    ) -> None:
        for name, value in (("capacity", capacity), ("max_seq_len", max_seq_len),
                            ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.capacity = int(capacity)
        self.max_seq_len = int(max_seq_len)
        self.chunk_size = min(int(chunk_size), self.capacity)
        self.directory = None if directory is None else Path(directory).resolve()
        self.seen = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)
        length = self.max_seq_len
        self._schema = {
            "token_ids": (np.dtype("uint16"), (length,)),
            "type_ids": (np.dtype("uint8"), (length,)),
            "numeric": (np.dtype("float32"), (length, NUM_NUMERIC)),
            "pad_mask": (np.dtype("bool"), (length,)),
            "legal_mask": (np.dtype("bool"), (NUM_ACTIONS,)),
            "regrets": (np.dtype("float32"), (NUM_ACTIONS,)),
            "iterations": (np.dtype("int64"), ()),
        }
        if isinstance(self, UnifiedReservoir):
            self._schema["hero_seat"] = (np.dtype("uint8"), ())
            self._seat_cache = None
        self._chunks: list[dict[str, np.ndarray]] = []
        self._maps: dict[str, np.memmap] = {}
        if self.directory is not None:
            self._open_disk()

    def __len__(self) -> int:
        return self.size

    def _signature(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "capacity": self.capacity,
            "max_seq_len": self.max_seq_len,
            "schema": {key: {"dtype": dtype.str, "shape": list(shape)}
                       for key, (dtype, shape) in self._schema.items()},
        }

    def _open_disk(self) -> None:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = self.directory / "metadata.json"
        if metadata.exists():
            state = json.loads(metadata.read_text())
            self._validate_signature(state)
            self._validate_counters(state)
            mode = "r+"
        else:
            if any((self.directory / f"{key}.dat").exists() for key in self._schema):
                raise ValueError("reservoir data exists without metadata; refusing to overwrite")
            state = None
            mode = "w+"
        for key, (dtype, tail) in self._schema.items():
            path = self.directory / f"{key}.dat"
            shape = (self.capacity,) + tail
            if mode == "r+" and (not path.exists() or path.stat().st_size != int(np.prod(shape)) * dtype.itemsize):
                raise ValueError(f"reservoir file has missing or invalid size: {path}")
            self._maps[key] = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        if state is not None:
            self.size = int(state["size"])
            self.seen = int(state["seen"])
            self._rng.bit_generator.state = state["rng_state"]
        else:
            self.flush()

    def _validate_signature(self, state: dict[str, Any]) -> None:
        for key, expected in self._signature().items():
            if state.get(key) != expected:
                raise ValueError(f"reservoir {key} does not match this buffer")

    def _validate_counters(self, state: dict[str, Any]) -> None:
        size, seen = int(state["size"]), int(state["seen"])
        if size < 0 or seen < 0 or size != min(seen, self.capacity):
            raise ValueError("invalid reservoir size/seen counters")

    def _ensure_slot(self, index: int) -> None:
        if self.directory is not None:
            return
        while len(self._chunks) <= index // self.chunk_size:
            offset = len(self._chunks) * self.chunk_size
            count = min(self.chunk_size, self.capacity - offset)
            self._chunks.append({key: np.empty((count,) + tail, dtype=dtype)
                                 for key, (dtype, tail) in self._schema.items()})

    def _write_row(self, index: int, arrays: dict[str, np.ndarray], row: int) -> None:
        if "hero_seat" in self._schema:
            self._seat_cache = None
        self._ensure_slot(index)
        if self.directory is not None:
            storage, offset = self._maps, index
        else:
            storage, offset = self._chunks[index // self.chunk_size], index % self.chunk_size
        for key in self._schema:
            storage[key][offset] = arrays[key][row]

    def add(self, batch: TensorBatch, regrets: torch.Tensor, iteration: int, *, hero_seat=None) -> None:
        """Add all rows atomically after validation, with uniform retention."""
        if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)) or iteration < 1:
            raise ValueError("iteration must be a positive integer")
        if iteration > np.iinfo(np.int64).max:
            raise ValueError("iteration exceeds int64 storage")
        if set(batch) != set(self._BATCH_KEYS):
            raise ValueError(f"batch must contain exactly {self._BATCH_KEYS}")
        count = int(batch["token_ids"].shape[0])
        arrays: dict[str, np.ndarray] = {}
        if "hero_seat" in self._schema:
            if hero_seat not in (0, 1):
                raise ValueError("unified reservoir requires hero_seat=0 (BB) or 1 (SB)")
            arrays["hero_seat"] = np.full(count, hero_seat, dtype=np.uint8)
        for key in self._BATCH_KEYS:
            dtype, tail = self._schema[key]
            value = batch[key].detach().cpu().numpy()
            if value.shape != (count,) + tail:
                raise ValueError(f"{key} has invalid shape {value.shape}; expected {(count,) + tail}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite")
            if key in ("token_ids", "type_ids"):
                if ((value < 0) | (value > np.iinfo(dtype).max) | (value != np.floor(value))).any():
                    raise ValueError(f"{key} cannot be represented as {dtype}")
            elif key in ("pad_mask", "legal_mask"):
                if ((value != 0) & (value != 1)).any():
                    raise ValueError(f"{key} must contain boolean or zero/one values")
            arrays[key] = value.astype(dtype, copy=False)
        if not arrays["legal_mask"].any(axis=-1).all():
            raise ValueError("every infoset must have at least one legal action")
        targets = regrets.detach().cpu().numpy()
        if targets.shape != (count, NUM_ACTIONS) or not np.isfinite(targets).all():
            raise ValueError(f"regrets must be finite with shape {(count, NUM_ACTIONS)}")
        arrays["regrets"] = np.where(arrays["legal_mask"], targets, 0).astype(np.float32)
        if not np.isfinite(arrays["regrets"]).all():
            raise ValueError("regrets overflow float32 storage")
        if not np.isfinite(arrays["numeric"]).all():
            raise ValueError("numeric features overflow float32 storage")
        arrays["iterations"] = np.full(count, iteration, dtype=np.int64)
        for row in range(count):
            self.seen += 1
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                index = int(self._rng.integers(self.seen))
                if index >= self.capacity:
                    continue
            self._write_row(index, arrays, row)

    def _read_rows(self, indexes: np.ndarray) -> dict[str, np.ndarray]:
        if self.directory is not None:
            return {key: np.array(value[indexes], copy=True) for key, value in self._maps.items()}
        result = {key: np.empty((len(indexes),) + tail, dtype=dtype)
                  for key, (dtype, tail) in self._schema.items()}
        chunk_ids = indexes // self.chunk_size
        for chunk_id in np.unique(chunk_ids):
            selected = np.flatnonzero(chunk_ids == chunk_id)
            offsets = indexes[selected] % self.chunk_size
            for key, value in self._chunks[int(chunk_id)].items():
                result[key][selected] = value[offsets]
        return result

    def _retained_data(self) -> dict[str, np.ndarray]:
        """Copy a full snapshot in linear time, without a capacity-sized index."""
        if self.directory is not None:
            return {key: np.array(value[:self.size], copy=True)
                    for key, value in self._maps.items()}
        result = {key: np.empty((self.size,) + tail, dtype=dtype)
                  for key, (dtype, tail) in self._schema.items()}
        for chunk_id, chunk in enumerate(self._chunks):
            start = chunk_id * self.chunk_size
            count = min(self.chunk_size, self.size - start)
            for key, value in chunk.items():
                result[key][start:start + count] = value[:count]
        return result

    def sample(
        self, batch_size: int, device: str | torch.device = "cpu", *,
        rng: np.random.Generator | None = None,
    ) -> tuple[TensorBatch, torch.Tensor, torch.Tensor]:
        """Uniform minibatch with replacement; return inputs, targets, iterations."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self.size:
            raise ValueError("cannot sample an empty reservoir")
        sampler = self._rng if rng is None else rng
        arrays = self._read_rows(sampler.integers(self.size, size=int(batch_size)))
        return self._tensor_sample(arrays, device)

    def _tensor_sample(self, arrays, device):
        batch = {}
        for key in self._BATCH_KEYS:
            value = torch.from_numpy(arrays[key])
            if key in ("token_ids", "type_ids"):
                value = value.long()
            elif key == "legal_mask":
                value = value.float()
            batch[key] = value.to(device)
        return batch, torch.from_numpy(arrays["regrets"]).to(device), torch.from_numpy(arrays["iterations"]).to(device)

    def flush(self) -> None:
        """Persist current disk data and metadata, replacing metadata atomically."""
        if self.directory is None:
            return
        for value in self._maps.values():
            value.flush()
        temporary = self.directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.state_dict(include_data=False)))
        temporary.replace(self.directory / "metadata.json")

    def state_dict(self, *, include_data: bool = True) -> dict[str, Any]:
        """Capture counters/RNG and optionally portable copies of retained data.

        Metadata-only snapshots can be restored only if the destination buffer
        already has the matching data (for example, a copied memmap directory).
        """
        result = self._signature()
        result.update(size=self.size, seen=self.seen,
                      rng_state=copy.deepcopy(self._rng.bit_generator.state))
        if include_data:
            result["data"] = self._retained_data()
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a portable snapshot or metadata for already matching data."""
        self._validate_signature(state)
        self._validate_counters(state)
        size, seen = int(state["size"]), int(state["seen"])
        if "data" not in state:
            if self.size != size or self.seen != seen:
                raise ValueError("metadata-only restore requires matching reservoir data")
        else:
            arrays = state["data"]
            if set(arrays) != set(self._schema):
                raise ValueError("snapshot has invalid data fields")
            for key, (dtype, tail) in self._schema.items():
                value = np.asarray(arrays[key])
                if value.shape != (size,) + tail or value.dtype != dtype:
                    raise ValueError(f"snapshot has invalid {key} shape/dtype")
            if self.directory is None:
                self._chunks = []
                if size:
                    self._ensure_slot(size - 1)
                for chunk_id, chunk in enumerate(self._chunks):
                    start = chunk_id * self.chunk_size
                    count = min(self.chunk_size, size - start)
                    for key, value in chunk.items():
                        value[:count] = arrays[key][start:start + count]
            else:
                for key, value in self._maps.items():
                    value[:size] = arrays[key]
        self.size = size
        self.seen = seen
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        if "hero_seat" in self._schema:
            self._seat_cache = None
        self.flush()


class UnifiedReservoir(ReservoirBuffer):
    """One global Algorithm-R memory, with exact 1:1 seat-stratified draws.

    Retention is uniform across ALL infosets, not two seat-local reservoirs.
    Seat indexes are built once per fit phase, never scanned per minibatch.
    Sampling within each stratum is uniform with replacement. An absent
    stratum is an explicit error; the trainer defers fitting until both exist.
    """

    VERSION = 2

    def seat_indices(self):
        if self._seat_cache is None:
            if self.directory is not None:
                seats = self._maps["hero_seat"][:self.size]
            else:
                seats = np.concatenate([c["hero_seat"][:min(self.chunk_size, self.size - i * self.chunk_size)]
                                        for i, c in enumerate(self._chunks)]) if self.size else np.empty(0, dtype=np.uint8)
            self._seat_cache = {seat: np.flatnonzero(seats == seat) for seat in (0, 1)}
        return self._seat_cache

    def seat_counts(self):
        return {seat: len(indexes) for seat, indexes in self.seat_indices().items()}

    def sample(self, batch_size, device="cpu", *, rng=None, seat=None, balanced=True):
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sampler = self._rng if rng is None else rng
        if seat is not None:
            if seat not in (0, 1):
                raise ValueError("seat must be 0 or 1")
            candidates = self.seat_indices()[seat]
            if not len(candidates):
                raise ValueError(f"no retained samples for seat {seat}")
            indexes = sampler.choice(candidates, int(batch_size), replace=True)
        elif balanced:
            if batch_size % 2:
                raise ValueError("balanced sampling requires an even batch size")
            strata = self.seat_indices()
            if any(not len(strata[p]) for p in (0, 1)):
                raise ValueError("balanced sampling requires retained samples from both seats")
            indexes = np.concatenate([sampler.choice(strata[p], batch_size // 2, replace=True) for p in (0, 1)])
            sampler.shuffle(indexes)
        else:
            return super().sample(batch_size, device, rng=rng)
        return self._tensor_sample(self._read_rows(indexes), device)


class SeatReservoirWriter:
    """Tag collection at its traverser boundary, including parallel workers."""

    def __init__(self, reservoir, seat):
        self.reservoir, self.seat = reservoir, seat

    def add(self, batch, regrets, iteration):
        self.reservoir.add(batch, regrets, iteration, hero_seat=self.seat)

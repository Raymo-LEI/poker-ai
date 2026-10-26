import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from alphanl.cfr import checkpoint
from alphanl.cfr.checkpoint import restore_buffer, save_buffer
from alphanl.cfr.reservoir import ReservoirBuffer


def _batch(count, start=0):
    ids = torch.arange(start, start + count)
    return {"token_ids": ids[:, None].repeat(1, 8),
            "type_ids": torch.ones(count, 8, dtype=torch.long),
            "numeric": ids[:, None, None].repeat(1, 8, 4).float() / 17,
            "pad_mask": torch.zeros(count, 8, dtype=torch.bool),
            "legal_mask": torch.tensor([[1., 1, 0, 0, 1]]).repeat(count, 1)}


def _assert_equal(left, right):
    assert len(left) == len(right)
    assert left.seen == right.seen
    first, second = left.sample(25), right.sample(25)
    for key in first[0]:
        assert torch.equal(first[0][key], second[0][key])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize("restore_to_disk", [True, False])
def test_disk_generation_is_immutable_and_restores_exact_rng(tmp_path, restore_to_disk):
    original = ReservoirBuffer(7, 8, seed=41, directory=tmp_path / "live")
    original.add(_batch(20), torch.arange(100).reshape(20, 5).float(), 3)
    expected = ReservoirBuffer(7, 8, chunk_size=2)
    expected.load_state_dict(original.state_dict())
    payload = save_buffer(original, tmp_path / "generation_a" / "player_0")
    assert Path(payload["directory"]).is_absolute()
    original.add(_batch(100, 20), torch.zeros(100, 5), 4)
    original.flush()
    # A later checkpoint gets its own files, keeping the old retained samples.
    later = save_buffer(original, tmp_path / "generation_b" / "player_0")
    assert later["directory"] != payload["directory"]
    restored = ReservoirBuffer(7, 8, seed=999, chunk_size=3,
                               directory=tmp_path / "restored" if restore_to_disk else None)
    restore_buffer(restored, payload)
    _assert_equal(expected, restored)
    expected.add(_batch(20, 120), torch.zeros(20, 5), 5)
    restored.add(_batch(20, 120), torch.zeros(20, 5), 5)
    _assert_equal(expected, restored)
    with pytest.raises(FileExistsError):
        save_buffer(original, tmp_path / "generation_a" / "player_0")


def test_disk_copy_only_retained_rows_and_bounds_temporary_memory(tmp_path, monkeypatch):
    original = ReservoirBuffer(1000, 8, directory=tmp_path / "live")
    original.add(_batch(3), torch.zeros(3, 5), 2)
    monkeypatch.setattr(checkpoint, "COPY_CHUNK_BYTES", 31)
    original_array = np.array
    original_fromfile = np.fromfile
    copied_sizes = []
    read_sizes = []

    def bounded_array(value, *args, **kwargs):
        if kwargs.get("copy") is True:
            copied_sizes.append(value.nbytes)
            assert value.nbytes <= 31
        return original_array(value, *args, **kwargs)

    def bounded_read(handle, dtype, count):
        read_sizes.append(count * np.dtype(dtype).itemsize)
        assert count * np.dtype(dtype).itemsize <= 31
        return original_fromfile(handle, dtype=dtype, count=count)

    def refuse_full_state(*, include_data=True):
        assert not include_data, "disk checkpoint must never materialize its full reservoir"
        return ReservoirBuffer.state_dict(original, include_data=False)

    monkeypatch.setattr(np, "array", bounded_array)
    monkeypatch.setattr(np, "fromfile", bounded_read)
    monkeypatch.setattr(original, "state_dict", refuse_full_state)
    payload = save_buffer(original, tmp_path / "generation")
    for key, (dtype, tail) in original._schema.items():
        size = (Path(payload["directory"]) / f"{key}.dat").stat().st_size
        assert size == 3 * int(np.prod(tail)) * dtype.itemsize
    restored = ReservoirBuffer(1000, 8, directory=tmp_path / "restored")
    restore_buffer(restored, payload)
    assert copied_sizes and read_sizes
    monkeypatch.setattr(np, "array", original_array)
    monkeypatch.setattr(np, "fromfile", original_fromfile)
    _assert_equal(original, restored)


def test_empty_memmap_snapshot_and_ram_embedded_state(tmp_path):
    disk = ReservoirBuffer(1000, 8, directory=tmp_path / "empty")
    payload = save_buffer(disk, tmp_path / "generation")
    restored = ReservoirBuffer(1000, 8, directory=tmp_path / "restored")
    restore_buffer(restored, payload)
    assert len(restored) == restored.seen == 0
    assert all(path.stat().st_size == 0 for path in Path(payload["directory"]).glob("*.dat"))
    memory = ReservoirBuffer(5, 8)
    memory.add(_batch(3), torch.ones(3, 5), 1)
    portable = save_buffer(memory, tmp_path / "unused")
    assert "storage" not in portable
    target = ReservoirBuffer(5, 8)
    restore_buffer(target, portable)
    _assert_equal(memory, target)


def test_disk_restore_rejects_incompatible_or_incomplete_snapshot(tmp_path):
    disk = ReservoirBuffer(5, 8, directory=tmp_path / "live")
    disk.add(_batch(2), torch.zeros(2, 5), 1)
    payload = save_buffer(disk, tmp_path / "generation")
    with pytest.raises(ValueError, match="capacity"):
        restore_buffer(ReservoirBuffer(6, 8), payload)
    changed = copy.deepcopy(payload)
    changed["metadata"]["rng_state"]["state"]["state"] += 1
    with pytest.raises(ValueError, match="metadata"):
        restore_buffer(ReservoirBuffer(5, 8), changed)
    (Path(payload["directory"]) / "numeric.dat").unlink()
    with pytest.raises(ValueError, match="missing or invalid size"):
        restore_buffer(ReservoirBuffer(5, 8), payload)

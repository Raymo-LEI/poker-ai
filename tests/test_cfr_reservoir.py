import numpy as np
import pytest
import torch

from alphanl.cfr.reservoir import ReservoirBuffer


def _batch(count, *, start=0, length=8):
    identities = torch.arange(start, start + count)
    return {
        "token_ids": identities[:, None].repeat(1, length),
        "type_ids": torch.ones(count, length, dtype=torch.long),
        "numeric": identities[:, None, None].repeat(1, length, 4).float() / 100,
        "pad_mask": torch.zeros(count, length, dtype=torch.bool),
        "legal_mask": torch.tensor([[1, 1, 0, 0, 1]]).repeat(count, 1).float(),
    }


def _assert_samples_equal(left, right):
    for key in left[0]:
        assert torch.equal(left[0][key], right[0][key])
    assert torch.equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def test_lazy_capacity_and_lossless_batch_storage():
    buffer = ReservoirBuffer(10_000_000, 8, chunk_size=2)
    assert not buffer._chunks
    batch = _batch(3)
    batch["numeric"][1, 0, 0] = 0.12345679
    buffer.add(batch, torch.arange(15).view(3, 5).float(), iteration=7)
    assert len(buffer._chunks) == 2
    assert len(buffer) == buffer.seen == 3
    samples, targets, iterations = buffer.sample(100)
    ids = samples["token_ids"][:, 0]
    for row, identity in enumerate(ids):
        assert torch.equal(samples["numeric"][row], batch["numeric"][identity])
        assert targets[row, 2] == targets[row, 3] == 0
        assert targets[row, 0] == identity * 5
    assert samples["token_ids"].dtype == samples["type_ids"].dtype == torch.int64
    assert samples["numeric"].dtype == targets.dtype == torch.float32
    assert samples["pad_mask"].dtype == torch.bool
    assert iterations.dtype == torch.int64
    assert (iterations == 7).all()


def test_algorithm_r_matches_reference_and_retains_early_and_late_samples():
    capacity, total, seed = 13, 1000, 17
    buffer = ReservoirBuffer(capacity, 8, seed=seed, chunk_size=4)
    buffer.add(_batch(total), torch.zeros(total, 5), 1)
    rng = np.random.default_rng(seed)
    expected = list(range(capacity))
    for index in range(capacity, total):
        selected = int(rng.integers(index + 1))
        if selected < capacity:
            expected[selected] = index
    retained = buffer.state_dict()["data"]["token_ids"][:, 0]
    np.testing.assert_array_equal(retained, expected)
    assert len(buffer) == capacity
    assert buffer.seen == total
    assert min(retained) < total // 2 < max(retained)


def test_independent_runs_approximate_uniform_retention():
    counts = np.zeros(20)
    for seed in range(400):
        buffer = ReservoirBuffer(4, 8, seed=seed)
        buffer.add(_batch(20), torch.zeros(20, 5), 1)
        indexes = buffer.state_dict()["data"]["token_ids"][:, 0]
        counts[indexes] += 1
    # Expected 80 appearances each; deterministic broad bounds catch FIFO or
    # off-by-one implementations without a fragile significance threshold.
    assert counts.min() > 50
    assert counts.max() < 115


def test_checkpoint_restores_sampling_and_future_replacements():
    original = ReservoirBuffer(7, 8, seed=10, chunk_size=3)
    original.add(_batch(30), torch.zeros(30, 5), 1)
    restored = ReservoirBuffer(7, 8, seed=999, chunk_size=2)
    restored.load_state_dict(original.state_dict())
    _assert_samples_equal(original.sample(12), restored.sample(12))
    original.add(_batch(20, start=30), torch.ones(20, 5), 2)
    restored.add(_batch(20, start=30), torch.ones(20, 5), 2)
    _assert_samples_equal(original.sample(30), restored.sample(30))


def test_memmap_reopen_restores_data_counters_rng_and_allows_more_samples(tmp_path):
    original = ReservoirBuffer(9, 8, seed=10, directory=tmp_path)
    original.add(_batch(20), torch.arange(100).view(20, 5).float(), 3)
    original.flush()
    reopened = ReservoirBuffer(9, 8, seed=999, directory=tmp_path)
    assert len(reopened) == 9
    assert reopened.seen == 20
    _assert_samples_equal(original.sample(30), reopened.sample(30))
    reopened.add(_batch(3, start=20), torch.zeros(3, 5), 4)
    reopened.flush()
    final = ReservoirBuffer(9, 8, directory=tmp_path)
    assert final.seen == 23
    _assert_samples_equal(reopened.sample(30), final.sample(30))
    with pytest.raises(ValueError, match="capacity"):
        ReservoirBuffer(10, 8, directory=tmp_path)


def test_memmap_can_restore_portable_snapshot_into_ram(tmp_path):
    disk = ReservoirBuffer(5, 8, directory=tmp_path)
    disk.add(_batch(20), torch.zeros(20, 5), 1)
    memory = ReservoirBuffer(5, 8)
    memory.load_state_dict(disk.state_dict())
    _assert_samples_equal(disk.sample(20), memory.sample(20))


def test_invalid_inputs_do_not_change_seen_or_size():
    buffer = ReservoirBuffer(5, 8)
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)
    batch = _batch(1)
    batch["legal_mask"][:] = 0
    with pytest.raises(ValueError, match="legal"):
        buffer.add(batch, torch.zeros(1, 5), 1)
    with pytest.raises(ValueError, match="iteration"):
        buffer.add(_batch(1), torch.zeros(1, 5), 0)
    bad = _batch(1)
    bad["token_ids"][0, 0] = 100_000
    with pytest.raises(ValueError, match="represented"):
        buffer.add(bad, torch.zeros(1, 5), 1)
    with pytest.raises(ValueError, match="shape"):
        buffer.add(_batch(2), torch.zeros(1, 5), 1)
    assert buffer.seen == len(buffer) == 0


def test_metadata_only_restore_rejects_nonmatching_data():
    buffer = ReservoirBuffer(5, 8)
    buffer.add(_batch(1), torch.zeros(1, 5), 1)
    empty = ReservoirBuffer(5, 8)
    with pytest.raises(ValueError, match="matching"):
        empty.load_state_dict(buffer.state_dict(include_data=False))

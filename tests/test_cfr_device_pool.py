"""Full retained data and uniform draw semantics; optional CUDA coverage."""

import copy

import numpy as np
import pytest
import torch

from alphanl.cfr.device_pool import DeviceTrainingPool
from alphanl.cfr.reservoir import ReservoirBuffer
from test_cfr_buffer_checkpoint import _batch


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_pool_keeps_full_retained_data_and_sampler_rng(tmp_path, disk, device):
    buffer = ReservoirBuffer(11, 8, chunk_size=3, directory=tmp_path / "live" if disk else None)
    buffer.add(_batch(40, 32760), torch.arange(200).reshape(40, 5).float(), 17)
    before = copy.deepcopy(buffer.state_dict(include_data=False))
    pool = DeviceTrainingPool(buffer, device, chunk_rows=3)
    try:
        expected = buffer._retained_data()
        for key, values in expected.items():
            np.testing.assert_array_equal(pool.arrays[key].cpu().numpy(), values)
        generator = torch.Generator(device=device).manual_seed(88)
        state = generator.get_state()
        first = pool.sample(4096, generator)
        generator.set_state(state)
        second = pool.sample(4096, generator)
        for key in first[0]:
            assert torch.equal(first[0][key], second[0][key])
        assert torch.equal(first[1], second[1]) and torch.equal(first[2], second[2])
        unique, counts = first[0]["token_ids"][:, 0].unique(return_counts=True)
        assert len(unique) == 11 and counts.min() > 270 and counts.max() < 470
        assert first[0]["numeric"].dtype == first[1].dtype == torch.float32
        assert first[0]["token_ids"].dtype == first[0]["type_ids"].dtype == torch.long
        assert buffer.state_dict(include_data=False) == before
    finally:
        pool.close()
    assert not pool.arrays


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_training_bf16_fused_full_pool_checkpoint(tmp_path):
    from dataclasses import replace
    from alphanl.cfr import SDCFRTrainer
    from test_cfr_trainer import config, TinyGame

    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 required")
    cfg = replace(config(tmp_path / "run"), device="cuda", amp_dtype="bfloat16", fused_optimizer=True,
                  train_buffer_device="cuda", gpu_pool_reserve_bytes=0)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        row = trainer.run_iteration()
        assert np.isfinite(row["p0_loss"]) and trainer.optimizer_fused and trainer.amp_enabled
        assert trainer.last_fit_timing["pool_bytes"] > 0
        assert all(p.dtype == torch.float32 for m in trainer.models.values() for p in m.parameters())
        trainer.save_checkpoint(tmp_path / "train.pt")
        expected = trainer.run_iteration()
        expected_weights = {p: copy.deepcopy(m.state_dict()) for p, m in trainer.models.items()}
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed")),
                       root_factory=lambda seed: TinyGame()) as resumed:
        resumed.load_checkpoint(tmp_path / "train.pt")
        actual = resumed.run_iteration()
        assert actual == expected
        for player in (0, 1):
            for key, value in expected_weights[player].items():
                torch.testing.assert_close(resumed.models[player].state_dict()[key], value, rtol=0, atol=0)

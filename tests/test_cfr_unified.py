"""Unified architecture invariants and failure paths (CPU and optional CUDA)."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer, SnapshotPolicy, UnifiedReservoir, iteration_weighted_huber
from alphanl.cfr.actions import PreflopActionGuard
from alphanl.cfr.checkpoint import save_buffer, restore_buffer
from alphanl.cfr.device_pool import DeviceTrainingPool
from alphanl.cfr.model import regret_matching
from alphanl.cfr.traversal import HunlState
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.tokenizer import BettingHistoryTokenizer
from test_cfr_trainer import TinyGame, config


def unified_config(path, **kwargs):
    return config(path, separate_player_models=False, regret_target_scale=100., loss="huber", **kwargs)


def seat_batch(seat, count=1):
    row = TinyGame().observation()
    row["numeric"][0, 1, seat] = 1
    return {key: value.repeat((count,) + (1,) * (value.ndim - 1)) for key, value in row.items()}


@pytest.mark.parametrize("disk", [False, True])
def test_global_retention_and_exact_balanced_sampling(tmp_path, disk):
    buffer = UnifiedReservoir(500, 8, chunk_size=13, directory=tmp_path / "live" if disk else None)
    buffer.add(seat_batch(0, 1000), torch.zeros(1000, 5), 1, hero_seat=0)
    buffer.add(seat_batch(1, 9000), torch.zeros(9000, 5), 2, hero_seat=1)
    counts = buffer.seat_counts()
    assert buffer.seen == 10000 and len(buffer) == 500
    assert 20 < counts[0] < 80  # uniform GLOBAL retention, not seat-local quotas
    for _ in range(10):
        batch, _, _ = buffer.sample(1024)
        assert tuple(batch["numeric"][:, 1, :2].sum(0).tolist()) == (512., 512.)
    before = buffer.state_dict(include_data=False)["rng_state"]
    buffer.sample(8, rng=np.random.default_rng(55))
    assert buffer.state_dict(include_data=False)["rng_state"] == before
    with pytest.raises(ValueError, match="even"):
        buffer.sample(3)
    saved = save_buffer(buffer, tmp_path / "saved")
    restored = UnifiedReservoir(500, 8)
    restore_buffer(restored, saved)
    assert restored.seat_counts() == counts
    for key, value in buffer._retained_data().items():
        np.testing.assert_array_equal(restored._retained_data()[key], value)
    assert torch.equal(restored.sample(64)[0]["numeric"], buffer.sample(64)[0]["numeric"])
    if disk:
        buffer.flush()
        reopened = UnifiedReservoir(500, 8, directory=tmp_path / "live")
        assert reopened.seat_counts() == counts


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_device_pool_balances_each_batch_and_restores_rng(device):
    buffer = UnifiedReservoir(200, 8)
    for seat, count in ((0, 5), (1, 195)):
        buffer.add(seat_batch(seat, count), torch.zeros(count, 5), 1, hero_seat=seat)
    pool = DeviceTrainingPool(buffer, device)
    try:
        rng = torch.Generator(device=device).manual_seed(42)
        state = rng.get_state()
        first = pool.sample(1024, rng)
        assert first[0]["numeric"][:, 1, :2].sum(0).tolist() == [512., 512.]
        rng.set_state(state)
        second = pool.sample(1024, rng)
        assert all(torch.equal(first[0][key], second[0][key]) for key in first[0])
    finally:
        pool.close()


def test_missing_seat_defers_shared_fit(tmp_path):
    with SDCFRTrainer(unified_config(tmp_path)) as trainer:
        buffer = trainer.shared_reservoir
        buffer.add(seat_batch(1), torch.zeros(1, 5), 1, hero_seat=1)
        before = {k: v.clone() for k, v in trainer.shared_regret_net.state_dict().items()}
        with pytest.raises(ValueError, match="both seats"):
            buffer.sample(8)
        assert trainer.train_player("shared") == 0
        assert trainer.last_fit_diagnostics["fit_deferred_missing_seat"] == 1
        assert all(torch.equal(v, trainer.shared_regret_net.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("disk", [False, True])
def test_shared_model_single_update_warm_start_and_exact_resume(tmp_path, disk):
    cfg = unified_config(tmp_path / "run", buffer_directory=str(tmp_path / "live") if disk else None)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        assert list(trainer.models) == list(trainer.optimizers) == list(trainer.buffers) == ["shared"]
        assert trainer.shared_reservoir.capacity == 200
        assert trainer.policies[0] is trainer.policies[1] is trainer.shared_regret_net
        model, optimizer = trainer.shared_regret_net, trainer.optimizers["shared"]
        with patch.object(trainer, "train_player", wraps=trainer.train_player) as fit:
            row = trainer.run_iteration()
            assert fit.call_count == 1
        assert row["p0_training_draws"] == row["p1_training_draws"] == 12
        assert all(f"p{p}_probe_r2" in row and f"p{p}_probe_loss_ratio" in row for p in (0, 1))
        assert trainer.shared_regret_net is model and trainer.optimizers["shared"] is optimizer
        snapshot = torch.load(tmp_path / "run/strategies/iteration_0000001.pt", weights_only=True)
        assert list(snapshot["models"]) == ["shared"]
        assert not snapshot["models"]["shared"]["regret_head.weight"].any()  # sigma_1 used for traversal
        trainer.save_checkpoint(tmp_path / "train.pt")
        expected = trainer.run_iteration()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        steps = next(iter(optimizer.state.values()))["step"]
        assert steps == 6
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "restored")), root_factory=lambda seed: TinyGame()) as restored:
        restored.load_checkpoint(tmp_path / "train.pt")
        assert restored.run_iteration() == expected
        assert all(torch.equal(v, restored.shared_regret_net.state_dict()[k]) for k, v in weights.items())
    policy = SnapshotPolicy(tmp_path / "run/strategies")
    for player in (0, 1):
        policy.begin_hand(player)
        assert torch.isfinite(policy.strategy(TinyGame().observation())).all()


def test_position_survives_empty_history_and_truncation():
    table = HunlTable(seed=41)
    table.reset()
    tokenizer = HunlTokenizer(8)
    sb = table.snapshot(1)
    bb = replace(sb, me_seat=0)  # identical public fields/cards isolates position signal
    encoded_sb, encoded_bb = tokenizer.encode(sb), tokenizer.encode(bb)
    assert encoded_sb["numeric"][0, 1].tolist() == [0., 1., 0., 0.]
    assert encoded_bb["numeric"][0, 1].tolist() == [1., 0., 0., 0.]
    assert not tokenizer.encode(replace(sb, events=sb.events * 100))["pad_mask"][0, 1]
    legacy = HunlTokenizer(8, position_conditioning=False).encode(sb)
    assert not legacy["numeric"][0, 1].any()


def test_preflop_guard_consistent_mask_and_unlocks():
    table = HunlTable(seed=44)
    table.reset()
    guard = PreflopActionGuard()
    state = HunlState(table, HunlTokenizer(64), guard)
    assert 4 in table.legal_action_ids() and 4 not in state.legal_actions()
    assert state.observation()["legal_mask"][0, 4] == 0
    table.step(3)  # SB open
    assert 4 not in guard.apply(table.snapshot(0)).legal_action_ids
    table.step(3)  # BB 3-bet: SB may now shove
    assert 4 in guard.apply(table.snapshot(1)).legal_action_ids
    for snapshot in (replace(table.snapshot(1), street=1),
                     replace(table.snapshot(1), events=(), stacks=(1900., 1900.)),
                     replace(table.snapshot(1), events=(), stacks=(18000., 18000.))):
        assert 4 in guard.apply(snapshot).legal_action_ids
    fresh = HunlTable(seed=5)
    fresh.reset()
    assert 4 in PreflopActionGuard(enabled=False).apply(fresh.snapshot(1)).legal_action_ids
    fresh.step_raise_to(20000)  # unrestricted evaluation opponent can open shove
    assert 4 in guard.apply(fresh.snapshot(0)).legal_action_ids or 1 in fresh.legal_action_ids()


def test_huber_outlier_gradient_bounded_and_illegal_targets_ignored():
    prediction = torch.zeros(1, 5, requires_grad=True)
    # +/-200BB is scaled to +/-2, not scaled twice from chips.
    target = torch.tensor([[200., -200., 999., 999., 999.]]) / 100
    mask = torch.tensor([[1., 1., 0., 0., 0.]])
    loss = iteration_weighted_huber(prediction, target, mask, torch.tensor([7]))
    assert loss.item() == 21
    loss.backward()
    assert prediction.grad.tolist() == [[-7., 7., 0., 0., 0.]]


def test_all_negative_smoothing_has_no_fixed_index_bias():
    raw = torch.tensor([[-4., -1., -2., -3., -1000.], [0., 0., 0., 0., 0.]])
    mask = torch.tensor([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1]])
    probs = regret_matching(raw, mask, fallback="softmax")
    assert probs[0].argmax() == 1 and probs[0, 4] > 0 and probs[0, 3] == 0
    torch.testing.assert_close(probs[1], torch.full((5,), .2))
    torch.testing.assert_close(probs.sum(-1), torch.ones(2))


def test_config_defaults_and_legacy_metadata():
    cfg = CFRConfig()
    assert not cfg.separate_player_models and not cfg.reinitialize
    assert cfg.train_steps == 1200 and cfg.minibatch_size == 1024
    assert cfg.amp_dtype == "bfloat16" and cfg.fused_optimizer
    old = {k: v for k, v in cfg.to_dict().items() if k not in (
        "separate_player_models", "position_conditioning", "preflop_action_guard", "loss", "regret_target_scale")}
    restored = CFRConfig.from_checkpoint(old)
    assert restored.separate_player_models and not restored.position_conditioning
    assert not restored.preflop_action_guard and restored.regret_target_scale == 1
    with pytest.raises(ValueError, match="even"):
        CFRConfig(minibatch_size=3)
    with pytest.raises(ValueError, match="position"):
        CFRConfig(position_conditioning=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_cuda_bf16_fused_checkpoint(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 required")
    cfg = unified_config(tmp_path / "run", device="cuda", amp_dtype="bfloat16", fused_optimizer=True,
                         train_buffer_device="cuda", gpu_pool_reserve_bytes=0)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        assert trainer.amp_enabled and trainer.optimizer_fused
        assert trainer.last_fit_timing["pool_bytes"] > 0
        trainer.save_checkpoint(tmp_path / "train.pt")
        expected = trainer.run_iteration()
        weights = {k: v.clone() for k, v in trainer.shared_regret_net.state_dict().items()}
        assert trainer.shared_regret_net(TinyGame().observation() if cfg.device == "cpu" else
            {k: v.cuda() for k, v in TinyGame().observation().items()}).dtype == torch.float32
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed")), root_factory=lambda seed: TinyGame()) as restored:
        restored.load_checkpoint(tmp_path / "train.pt")
        assert restored.run_iteration() == expected
        for key, value in weights.items():
            torch.testing.assert_close(value, restored.shared_regret_net.state_dict()[key], rtol=0, atol=0)


def test_shared_parallel_collects_tagged_samples(tmp_path):
    from test_cfr_parallel import tiny_root
    cfg = unified_config(tmp_path, traversal_workers=2, worker_sample_batch=1)
    with SDCFRTrainer(cfg, root_factory=tiny_root) as trainer:
        row = trainer.run_iteration()
        assert trainer.shared_reservoir.seen == 8
        assert trainer.shared_reservoir.seat_counts() == {0: 4, 1: 4}
        assert row["p0_training_draws"] == row["p1_training_draws"]


def test_pool_fixed_capacity_samples_only_initialized_rows():
    buffer = UnifiedReservoir(100, 8)
    buffer.add(seat_batch(0, 1), torch.zeros(1, 5), 1, hero_seat=0)
    buffer.add(seat_batch(1, 3), torch.ones(3, 5), 2, hero_seat=1)
    pool = DeviceTrainingPool(buffer, 'cpu', capacity=100)
    try:
        assert pool.capacity == 100 and pool.size == 4
        assert pool.arrays['hero_seat'].shape == (100,)
        assert [len(pool.seat_indexes[p]) for p in (0, 1)] == [1, 3]
        batch, _, iterations = pool.sample(1024, torch.Generator().manual_seed(3))
        assert batch['numeric'][:, 1, :2].sum(0).tolist() == [512., 512.]
        assert set(iterations.tolist()) == {1, 2}
    finally:
        pool.close()

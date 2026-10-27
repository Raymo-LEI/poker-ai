"""Neural fitting, simultaneous updates, snapshot semantics and exact resume."""

from dataclasses import replace
import csv
import shutil
from unittest.mock import patch

import pytest
import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer, SnapshotPolicy, iteration_weighted_mse
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.tokenizer import BettingHistoryTokenizer


class TinyGame:
    """Two decision nodes, signed utilities, no hidden state in observations."""
    def __init__(self, actions=()):
        self.actions = actions

    def is_terminal(self):
        return len(self.actions) == 2

    def current_player(self):
        return len(self.actions)

    def legal_actions(self):
        return (1, 2)

    def observation(self):
        batch = BettingHistoryTokenizer(8).blank()
        batch["legal_mask"][0, 2] = 1
        batch["numeric"][0, 0, 0] = len(self.actions)
        return batch

    def child(self, action):
        return TinyGame(self.actions + (action,))

    def returns(self):
        value = 1.0 if self.actions[0] == self.actions[1] else -1.0
        return (value, -value)


def config(tmp_path, **kwargs):
    kwargs = {"separate_player_models": True, "regret_target_scale": 1., "loss": "mse",
              "amp_dtype": "none", "fused_optimizer": False, "allow_tf32": False, **kwargs}
    return CFRConfig(
        iterations=2, traversals_per_player=4, traversal_batch_size=4,
        buffer_capacity=100, train_steps=3, minibatch_size=8,
        model=ModelConfig(d_model=16, nhead=2, num_layers=1,
                          dim_feedforward=32, max_seq_len=8, dropout=0),
        out_dir=str(tmp_path), **kwargs,
    )


def test_weighted_mse_keeps_signed_targets_and_exact_iteration_weights():
    prediction = torch.zeros(2, 5, requires_grad=True)
    target = torch.tensor([[1., -2, 999, 999, 999], [3., 999, 999, 999, 999]])
    mask = torch.tensor([[1., 1, 0, 0, 0], [1., 0, 0, 0, 0]])
    loss = iteration_weighted_mse(prediction, target, mask, torch.tensor([1, 3]))
    assert float(loss.detach()) == (5 + 3 * 9) / 2
    loss.backward()
    assert torch.equal(prediction.grad[:, 2:], torch.zeros(2, 3))
    # A minibatch of size one must still retain the iteration multiplier.
    assert iteration_weighted_mse(prediction[:1], target[:1], mask[:1], torch.tensor([7])) == 35
    with pytest.raises(ValueError):
        iteration_weighted_mse(prediction, target, mask * 0, torch.tensor([1, 3]))


def test_simultaneous_collection_and_snapshot_profile(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path, reinitialize=False), root_factory=lambda seed: TinyGame())
    initial = {p: {k: v.clone() for k, v in model.state_dict().items()}
               for p, model in trainer.models.items()}
    row = trainer.run_iteration()
    assert row["p0_regret_samples"] == 4
    assert row["p1_regret_samples"] == 4
    assert trainer.buffers[0] is not trainer.buffers[1]
    path = tmp_path / "strategies" / "iteration_0000001.pt"
    snapshot = torch.load(path, weights_only=True)
    for player in (0, 1):
        assert snapshot["weight"] == 1
        for key, value in initial[player].items():
            assert torch.equal(snapshot["models"][player][key], value)
        # Unused legacy heads are preserved exactly by the new optimizer.
        for key in ("backbone.policy_head.weight", "backbone.value_head.weight"):
            assert torch.equal(trainer.models[player].state_dict()[key], initial[player][key])


def test_player_with_no_visited_infoset_keeps_current_network(tmp_path):
    class OnlySmallBlind(TinyGame):
        def current_player(self):
            return 1

        def child(self, action):
            return TinyGame((action, action))

    trainer = SDCFRTrainer(config(tmp_path), root_factory=lambda seed: OnlySmallBlind())
    before = {k: v.clone() for k, v in trainer.models[0].state_dict().items()}
    row = trainer.run_iteration()
    assert row["p0_regret_samples"] == row["p0_train_steps"] == row["p0_loss"] == 0
    assert row["p1_regret_samples"] > 0
    assert all(torch.equal(value, trainer.models[0].state_dict()[key]) for key, value in before.items())


def test_cuda_training_options_are_noops_on_cpu_and_phase_metrics_are_written(tmp_path):
    cfg = config(tmp_path, amp_dtype="bfloat16", fused_optimizer=True,
                 allow_tf32=True, reinitialize=False)
    cfg.train_steps = 1
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    assert trainer.amp_enabled is False
    assert trainer.optimizer_fused is False
    trainer.run_iteration()
    rows = list(csv.DictReader((tmp_path / "performance.csv").open()))
    assert len(rows) == 1
    for key in ("p0_collect_s", "p1_collect_s", "p0_fit_s", "p1_fit_s",
                "p0_regret_samples_per_s", "p1_training_draws_per_s", "iteration_s"):
        assert float(rows[0][key]) >= 0


def test_legacy_checkpoint_without_gpu_option_fields_still_resumes(tmp_path):
    cfg = config(tmp_path / "original", reinitialize=False)
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    checkpoint = tmp_path / "legacy.pt"
    trainer.save_checkpoint(checkpoint)
    blob = torch.load(checkpoint, weights_only=False)
    for key in ("amp_dtype", "fused_optimizer", "allow_tf32"):
        blob["config"].pop(key)
    torch.save(blob, checkpoint)
    restored = SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "restored")),
                            root_factory=lambda seed: TinyGame())
    restored.load_checkpoint(checkpoint)
    assert restored.iteration == 1
    profile_rows = list(csv.DictReader((tmp_path / "restored" / "performance.csv").open()))
    assert len(profile_rows) == 1


@pytest.mark.parametrize("disk", [False, True])
def test_checkpoint_exact_continuation(tmp_path, disk):
    cfg = config(tmp_path / "original", buffer_directory=str(tmp_path / "buffer") if disk else None)
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    checkpoint = tmp_path / "train.pt"
    trainer.save_checkpoint(checkpoint)
    expected = trainer.run_iteration()
    expected_weights = {p: {k: v.clone() for k, v in model.state_dict().items()}
                        for p, model in trainer.models.items()}
    restored_cfg = replace(cfg, out_dir=str(tmp_path / "restored"))
    restored = SDCFRTrainer(restored_cfg, root_factory=lambda seed: TinyGame())
    restored.load_checkpoint(checkpoint)
    actual = restored.run_iteration()
    assert actual == expected
    assert sorted(restored.strategy_paths) == [1, 2]
    snapshot_policy = SnapshotPolicy(tmp_path / "restored" / "strategies")
    assert snapshot_policy.iterations.tolist() == [1, 2]
    for player in (0, 1):
        for key, value in expected_weights[player].items():
            assert torch.equal(restored.models[player].state_dict()[key], value)


def test_restoring_old_checkpoint_excludes_future_profiles_and_metrics(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path), root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    checkpoint = tmp_path / "iteration1.pt"
    trainer.save_checkpoint(checkpoint)
    trainer.run_iteration()
    assert SnapshotPolicy(tmp_path / "strategies").iterations.tolist() == [1, 2]
    trainer.load_checkpoint(checkpoint)
    assert SnapshotPolicy(tmp_path / "strategies").iterations.tolist() == [1]
    lines = (tmp_path / "metrics.csv").read_text().splitlines()
    assert len(lines) == 2
    assert list((tmp_path / "history").glob("*.csv"))


def test_checkpoint_rejects_feature_version_mismatch_before_loading(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path), root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    path = tmp_path / "train.pt"
    trainer.save_checkpoint(path)
    blob = torch.load(path, weights_only=False)
    blob["numeric_version"] = "legacy_clipped"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="feature version mismatch"):
        trainer.load_checkpoint(path)


def test_disk_checkpoint_directory_can_be_moved_with_its_data(tmp_path):
    cfg = config(tmp_path / "source", buffer_directory=str(tmp_path / "live"))
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    trainer.save_checkpoint(tmp_path / "source" / "train.pt")
    shutil.copytree(tmp_path / "source", tmp_path / "moved")
    shutil.rmtree(tmp_path / "source")
    loaded_cfg = replace(cfg, out_dir=str(tmp_path / "resumed"),
                         buffer_directory=str(tmp_path / "new_live"))
    loaded = SDCFRTrainer(loaded_cfg, root_factory=lambda seed: TinyGame())
    loaded.load_checkpoint(tmp_path / "moved" / "train.pt")
    assert loaded.iteration == 1
    assert SnapshotPolicy(tmp_path / "resumed" / "strategies").iterations.tolist() == [1]
    for player in (0, 1):
        assert loaded.buffers[player].seen == trainer.buffers[player].seen
        assert len(loaded.buffers[player]) == len(trainer.buffers[player])


def test_checkpoint_gc_bounds_disk_and_preserves_other_checkpoints(tmp_path):
    cfg = config(tmp_path / "run", buffer_directory=str(tmp_path / "live"))
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    path = tmp_path / "train.pt"
    trainer.save_checkpoint(path)
    # An older checkpoint copy without a sidecar must protect its data.
    pinned = tmp_path / "pinned.pt"
    shutil.copyfile(path, pinned)
    first = set((tmp_path / "checkpoint_data").iterdir())
    trainer.run_iteration()
    for _ in range(3):
        trainer.save_checkpoint(path)
        assert len(list((tmp_path / "checkpoint_data").iterdir())) == 2
        assert all(p.exists() for p in first)
    loaded = SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed")),
                          root_factory=lambda seed: TinyGame())
    loaded.load_checkpoint(pinned)
    assert loaded.iteration == 1
    pinned.unlink()
    trainer.save_checkpoint(path)
    assert len(list((tmp_path / "checkpoint_data").iterdir())) == 1
    assert sorted(trainer.strategy_paths) == [1, 2]
    loaded.load_checkpoint(path)
    assert loaded.iteration == 2


def test_checkpoint_failure_keeps_published_checkpoint_and_removes_partial_data(tmp_path):
    cfg = config(tmp_path / "run", buffer_directory=str(tmp_path / "live"))
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    path = tmp_path / "train.pt"
    trainer.save_checkpoint(path)
    expected = path.read_bytes()
    generations = set((tmp_path / "checkpoint_data").iterdir())
    with patch("alphanl.cfr.trainer.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            trainer.save_checkpoint(path)
    assert path.read_bytes() == expected
    assert set((tmp_path / "checkpoint_data").iterdir()) == generations
    assert not path.with_suffix(".pt.tmp").exists()
    with patch("alphanl.cfr.trainer.shutil.disk_usage", return_value=shutil._ntuple_diskusage(10, 9, 1)):
        with pytest.raises(OSError, match="previous checkpoint retained"):
            trainer.save_checkpoint(path)
    assert path.read_bytes() == expected


def test_resumed_legacy_csv_headers_are_upgraded_without_losing_history(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path), root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    for name in ("metrics.csv", "performance.csv"):
        path = tmp_path / name
        rows = list(csv.DictReader(path.open()))
        # Model the earlier release's shorter CSV schema.
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration"])
            writer.writeheader()
            writer.writerow({"iteration": rows[0]["iteration"]})
    trainer.run_iteration()
    for name in ("metrics.csv", "performance.csv"):
        rows = list(csv.DictReader((tmp_path / name).open()))
        assert [row["iteration"] for row in rows] == ["1", "2"]
        assert None not in rows[1]
        assert rows[1]["p0_collect_s" if name == "performance.csv" else "p0_regret_samples"]


def test_checkpoint_post_publish_sync_failure_preserves_referenced_data(tmp_path):
    import alphanl.cfr.trainer as module
    cfg = config(tmp_path / "run", buffer_directory=str(tmp_path / "live"))
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    path = tmp_path / "train.pt"
    trainer.save_checkpoint(path)
    trainer.run_iteration()
    sync = module.sync_directory

    def fail_after_publish(directory):
        if directory == path.parent:
            raise OSError("directory sync failure")
        sync(directory)

    with patch.object(module, "sync_directory", side_effect=fail_after_publish):
        with pytest.raises(OSError, match="directory sync failure"):
            trainer.save_checkpoint(path)
    restored = SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed")), root_factory=lambda seed: TinyGame())
    restored.load_checkpoint(path)
    assert restored.iteration == 2


def test_gc_refuses_to_delete_when_another_checkpoint_cannot_be_read(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path / "run"), root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    path = tmp_path / "train.pt"
    trainer.save_checkpoint(path)
    original = set((tmp_path / "checkpoint_data").iterdir())
    (tmp_path / "unreadable.pt").write_text("incomplete checkpoint")
    with pytest.warns(RuntimeWarning, match="cleanup skipped"):
        trainer.save_checkpoint(path)
    assert all(p.exists() for p in original)


def test_checkpoint_copy_fallback_preserves_complete_strategy_history(tmp_path):
    cfg = config(tmp_path / "run", buffer_directory=str(tmp_path / "live"))
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    trainer.run_iteration()
    with patch("alphanl.cfr.trainer.os.link", side_effect=OSError("hardlink unsupported")):
        trainer.save_checkpoint(tmp_path / "train.pt")
        trainer.save_checkpoint(tmp_path / "train.pt")
    assert len(list((tmp_path / "checkpoint_data").iterdir())) == 1
    restored = SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "restored")), root_factory=lambda seed: TinyGame())
    restored.load_checkpoint(tmp_path / "train.pt")
    assert sorted(restored.strategy_paths) == [1, 2]


def test_neural_regret_fit_reduces_loss(tmp_path):
    cfg = config(tmp_path)
    cfg.train_steps = 60
    cfg.learning_rate = 0.01
    cfg.reinitialize = False
    trainer = SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame())
    batch = TinyGame().observation()
    target = torch.tensor([[0., -2, 3, 0, 0]])
    trainer.buffers[0].add(batch, target, 1)
    model = trainer.models[0].eval()
    with torch.no_grad():
        before = iteration_weighted_mse(model(batch), target, batch["legal_mask"], torch.ones(1))
    trainer.train_player(0)
    with torch.no_grad():
        after = iteration_weighted_mse(model(batch), target, batch["legal_mask"], torch.ones(1))
    assert after < before * 0.02
    assert model.strategy(batch)[0, 2] > 0.99


def test_snapshot_policy_keeps_one_sampled_model_for_entire_hand(tmp_path):
    trainer = SDCFRTrainer(config(tmp_path), root_factory=lambda seed: TinyGame())
    trainer.run_iteration()
    trainer.run_iteration()
    policy = SnapshotPolicy(tmp_path / "strategies", seed=5)
    batch = TinyGame().observation()
    with pytest.raises(RuntimeError):
        policy.strategy(batch)
    iteration = policy.begin_hand(1)
    weights = {k: v.clone() for k, v in policy.model.state_dict().items()}
    for _ in range(3):
        assert policy.act(batch).item() in (1, 2)
        assert policy.hand_iteration == iteration
        assert all(torch.equal(value, policy.model.state_dict()[key]) for key, value in weights.items())
    # Iteration 2 carries twice iteration 1's probability.
    draws = [policy.begin_hand(1) for _ in range(150)]
    assert 0.55 < draws.count(2) / len(draws) < 0.8


def test_hunl_real_table_collects_and_fits_both_players(tmp_path):
    pytest.importorskip("phevaluator")
    pytest.importorskip("pokerkit")
    cfg = config(tmp_path)
    cfg.model.max_seq_len = 32
    cfg.traversals_per_player = 1
    cfg.traversal_batch_size = 1
    cfg.train_steps = 1
    trainer = SDCFRTrainer(cfg)
    row = trainer.run_iteration()
    assert row["p0_regret_samples"] > 0 and row["p1_regret_samples"] > 0
    assert row["p0_terminal_nodes"] > 0 and row["p1_terminal_nodes"] > 0

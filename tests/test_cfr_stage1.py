"""Branch intervention restores full state and leaves ordinary resume strict."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphanl.cfr import SDCFRTrainer, SnapshotPolicy
from alphanl.cfr.study import evaluate_probe, fixed_probe, freeze_prefix
from test_cfr_trainer import TinyGame, config


@pytest.mark.parametrize("disk", [False, True])
def test_budget_branch_restores_full_state_and_changes_only_future_fit(tmp_path, disk):
    cfg = config(tmp_path / "original", separate_player_models=False,
                 buffer_directory=str(tmp_path / "source_buffer") if disk else None)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as source:
        source.run_iteration()
        source.save_checkpoint(tmp_path / "original/train.pt")
        expected_probe = fixed_probe(source, rows=16)
        rng = copy.deepcopy(source.rng.bit_generator.state)
        traversal_rng = copy.deepcopy(source.traverser.rng.bit_generator.state)
        reservoir_rng = copy.deepcopy(source.shared_reservoir._rng.bit_generator.state)
        expected_data = source.shared_reservoir._retained_data()
        expected_weights = {k: v.clone() for k, v in source.shared_regret_net.state_dict().items()}
        expected_optimizer = source.optimizers["shared"].state_dict()
    branch_cfg = replace(cfg, out_dir=str(tmp_path / "branch"), train_steps=6,
                         buffer_directory=str(tmp_path / "branch_buffer") if disk else None)
    with SDCFRTrainer(branch_cfg, root_factory=lambda seed: TinyGame()) as branch:
        with pytest.raises(ValueError, match="train_steps"):
            branch.load_checkpoint(tmp_path / "original/train.pt")
        branch.load_checkpoint(tmp_path / "original/train.pt", branch_train_steps=True)
        assert branch.iteration == 1 and branch.config.train_steps == 6
        assert branch.rng.bit_generator.state == rng
        assert branch.traverser.rng.bit_generator.state == traversal_rng
        assert branch.shared_reservoir._rng.bit_generator.state == reservoir_rng
        for key, value in expected_data.items():
            np.testing.assert_array_equal(value, branch.shared_reservoir._retained_data()[key])
        for key, value in expected_weights.items():
            assert torch.equal(value, branch.shared_regret_net.state_dict()[key])
        for key, state in expected_optimizer["state"].items():
            for field, value in state.items():
                assert torch.equal(value, branch.optimizers["shared"].state_dict()["state"][key][field])
        assert sorted(branch.strategy_paths) == [1]
        assert fixed_probe(branch, rows=16)["source_iteration"] == 1
        assert branch.shared_reservoir._rng.bit_generator.state == reservoir_rng
        scores = evaluate_probe(branch, expected_probe, batch_size=5)
        assert all(np.isfinite(v) for k, v in scores.items() if "normalized_mse" in k)
        row = branch.run_iteration()
        assert row["shared_train_steps"] == 6
        archive = freeze_prefix(branch, tmp_path / "frozen")
        assert SnapshotPolicy(archive["path"]).iterations.tolist() == [1, 2]
        lineage = json.loads((tmp_path / "branch/ops/branch.json").read_text())
        assert lineage["intervention"]["train_steps"] == {"before": 3, "after": 6}


def test_branch_cannot_overwrite_source_or_existing_run_or_change_capacity(tmp_path):
    cfg = config(tmp_path / "source")
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as source:
        source.run_iteration()
        source.save_checkpoint(tmp_path / "source/train.pt")
        with pytest.raises(ValueError, match="new output"):
            source.load_checkpoint(tmp_path / "source/train.pt", branch_train_steps=True)
    changed = replace(cfg, out_dir=str(tmp_path / "new"), buffer_capacity=200)
    with SDCFRTrainer(changed, root_factory=lambda seed: TinyGame()) as branch:
        with pytest.raises(ValueError, match="buffer_capacity"):
            branch.load_checkpoint(tmp_path / "source/train.pt", branch_train_steps=True)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "metrics.csv").write_text("existing data")
    with SDCFRTrainer(replace(cfg, out_dir=str(destination)), root_factory=lambda seed: TinyGame()) as branch:
        with pytest.raises(ValueError, match="already contains"):
            branch.load_checkpoint(tmp_path / "source/train.pt", branch_train_steps=True)
    assert (destination / "metrics.csv").read_text() == "existing data"


def test_branch_reencodes_trusted_protocol4_snapshots_without_mutating_source(tmp_path):
    cfg = config(tmp_path / "source")
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as source:
        source.run_iteration()
        profile_path = source.strategy_paths[1]
        profile = torch.load(profile_path, weights_only=True)
        torch.save(profile, profile_path, pickle_protocol=4)
        source.save_checkpoint(tmp_path / "source/train.pt")
        original_bytes = profile_path.read_bytes()
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "branch")), root_factory=lambda seed: TinyGame()) as branch:
        branch.load_checkpoint(tmp_path / "source/train.pt", branch_train_steps=True)
        converted = torch.load(branch.strategy_paths[1], weights_only=True)
        assert converted["iteration"] == 1
        for player in (0, 1):
            for key, value in profile["models"][player].items():
                assert torch.equal(value, converted["models"][player][key])
    assert profile_path.read_bytes() == original_bytes


def test_evaluation_seeds_and_paired_deltas_are_candidate_independent(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/eval_cfr_stage1.py"
    spec = importlib.util.spec_from_file_location("stage1_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.opponent_seed(1, "equity") == module.opponent_seed(1, "equity")
    assert module.opponent_seed(1, "equity") != module.opponent_seed(1, "nit")
    left, right = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = [{"pair": n, "deck_seed": n + 100, "chips_sb_bb": [0, 0]} for n in range(4)]
    left.write_text("\n".join(json.dumps(r) for r in rows))
    right.write_text("\n".join(json.dumps(dict(r, chips_sb_bb=[100, 100])) for r in rows))
    result = module.paired_delta(left, right)
    assert result["mean"] == 1000 and result["ci95"] == [1000, 1000]
    rows[0]["deck_seed"] = 999
    right.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="identical deck"):
        module.paired_delta(left, right)


def test_sweet_spot_config_is_exact_requested_architecture():
    from alphanl.cfr import CFRConfig
    from alphanl.transformer.model import PokerTransformer, count_parameters
    from dataclasses import asdict
    root = Path(__file__).resolve().parents[1]
    cfg = CFRConfig.from_json(root / "configs/cfr/stage1_149m_4m_1200.json")
    assert (cfg.model.d_model, cfg.model.nhead, cfg.model.num_layers, cfg.model.dim_feedforward) == (192, 6, 4, 512)
    assert count_parameters(PokerTransformer(**asdict(cfg.model))) == 1491206
    assert not cfg.separate_player_models and cfg.buffer_capacity * 2 == 4000000


def test_real_orchestration_freezes_source_and_publishes_both_time_views(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / "scripts/run_cfr_stage1.py"
    spec = importlib.util.spec_from_file_location("stage1_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_path = tmp_path / "source/train.pt"
    cfg = config(source_path.parent, separate_player_models=False)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as source:
        source.run_iteration()
        source.save_checkpoint(source_path)
    source_bytes = source_path.read_bytes()
    monkeypatch.setattr(module, "SDCFRTrainer", lambda c: SDCFRTrainer(c, root_factory=lambda seed: TinyGame()))
    args = SimpleNamespace(study=tmp_path / "study", checkpoint=source_path,
                           bootstrap_config=None, source_iteration=1, require_full_pool=False,
                           run_root=tmp_path, target=2, probe_rows=8, seed=2, pairs=2, eval_workers=1)
    manifest = module.prepare(args)
    assert manifest["source_iteration"] == 1
    for steps in (1200, 2400):
        args.steps = steps
        module.train_arm(args)
        report = json.loads((args.study / f"arm{steps}.json").read_text())
        assert report["completed"] and report["same_iteration_archive"]["through"] == 2
        assert 1 <= report["equal_time_archive"]["through"] <= 2
        assert not report["equal_time_archive"]["restart_checkpoint"]
        assert report["training_wall_seconds"] > 0
        archive = SnapshotPolicy(report["same_iteration_archive"]["path"])
        assert archive.iterations.tolist() == [1, 2]
        rows = [json.loads(r) for r in (Path(report["run"]) / "metrics.jsonl").read_text().splitlines()]
        assert rows[-1]["iteration"] == 2
    assert source_path.read_bytes() == source_bytes


@pytest.mark.parametrize("wrong_iteration,require_full", [(True, False), (False, True)])
def test_study_refuses_wrong_source_iteration_or_nonfull_pool(tmp_path, wrong_iteration, require_full):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("study_guards", Path(__file__).resolve().parents[1] / "scripts/run_cfr_stage1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "source/train.pt"
    cfg = config(checkpoint.parent, separate_player_models=False)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(checkpoint)
    args = SimpleNamespace(study=tmp_path / "study", checkpoint=checkpoint, bootstrap_config=None,
        source_iteration=150 if wrong_iteration else 1, require_full_pool=require_full,
        run_root=tmp_path, target=200, probe_rows=8, seed=2, pairs=2, eval_workers=1)
    with pytest.raises(ValueError, match="expected full checkpoint" if wrong_iteration else "not full"):
        module.prepare(args)
    assert not list(tmp_path.glob("study_steps*"))


def test_missing_source_can_be_rebuilt_and_is_labeled_new_trajectory(tmp_path, monkeypatch):
    from types import SimpleNamespace
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("study_bootstrap", Path(__file__).resolve().parents[1] / "scripts/run_cfr_stage1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = config(tmp_path / "unused", separate_player_models=False)
    cfg.train_steps = 1200
    config_path = tmp_path / "bootstrap.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    def toy_trainer(c):
        return SDCFRTrainer(c, root_factory=lambda seed: TinyGame())
    def train_bootstrap(command, **kwargs):
        from alphanl.cfr import CFRConfig
        created = CFRConfig.from_json(command[command.index("--config") + 1])
        assert created.iterations == 1 and created.train_steps == 1200
        with toy_trainer(created) as trainer:
            trainer.run()
    monkeypatch.setattr(module, "SDCFRTrainer", toy_trainer)
    monkeypatch.setattr(module.subprocess, "run", train_bootstrap)
    args = SimpleNamespace(study=tmp_path / "study", checkpoint=None, bootstrap_config=config_path,
        source_iteration=1, require_full_pool=False, run_root=tmp_path, target=2,
        probe_rows=8, seed=2, pairs=2, eval_workers=1)
    manifest = module.prepare(args)
    assert manifest["source_provenance"] == "new_baseline_rebuilt_from_seed"
    assert manifest["source_iteration"] == 1
    assert (tmp_path / "study_bootstrap/train.pt").is_file()

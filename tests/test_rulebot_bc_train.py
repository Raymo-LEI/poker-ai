"""Rule-bot BC training: game holdout, legal hash, PPO-loadable checkpoint, continuation config."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphanl.transformer.checkpoint import load_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.train import build_model, build_optimizer

ROOT = Path(__file__).resolve().parents[1]
MASKFOLD = ROOT / "configs" / "transformer" / "botzone_laptop_8gb_maskfold.json"
BC_CONFIG = ROOT / "configs" / "transformer" / "botzone_bc_rulebot.json"
BC_INIT = ROOT / "configs" / "transformer" / "botzone_laptop_8gb_bc_init.json"


def _load(name: str, path: Path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load("train_bc_rulebot", ROOT / "scripts" / "train_bc_rulebot.py")


@pytest.fixture(scope="module")
def checker():
    return _load("check_bc_rulebot", ROOT / "scripts" / "check_bc_rulebot.py")


def test_holdout_is_by_games_not_rows(trainer):
    # Three rows in game 0, one row in each of nine other games: 10 games.
    game_ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    mask, val_games = trainer.split_game_mask(game_ids, seed=42, fraction=0.1)
    assert len(val_games) == 1
    assert int(mask.sum()) == int(np.isin(game_ids, val_games).sum())
    # A game is entirely in or entirely out.
    for game in range(10):
        rows = game_ids == game
        assert len(set(mask[rows].tolist())) == 1
    train_games = set(range(10)) - set(val_games)
    assert train_games.isdisjoint(val_games)
    again, again_games = trainer.split_game_mask(game_ids, seed=42, fraction=0.1)
    assert again_games == val_games
    assert np.array_equal(again, mask)


def test_continuation_config_differs_only_in_lr_init_and_output_dirs():
    mask = json.loads(MASKFOLD.read_text())
    init = json.loads(BC_INIT.read_text())
    assert init.pop("bc_init_checkpoint") == "runs/bc_rulebot_train/bc.pt"
    assert "bc_init_checkpoint" not in mask
    assert init["ppo"].pop("lr") == 0.0001
    assert mask["ppo"].pop("lr") == 0.0003
    assert init["train"].pop("out_dir") == "runs/transformer_botzone_laptop_bc_init"
    assert init["train"].pop("checkpoint_dir") == "runs/transformer_botzone_laptop_bc_init/ckpt"
    mask["train"].pop("out_dir")
    mask["train"].pop("checkpoint_dir")
    assert init == mask
    loaded = ExperimentConfig.from_json(BC_INIT)
    assert loaded.ppo.lr == 1e-4
    assert loaded.bc_init_checkpoint == "runs/bc_rulebot_train/bc.pt"
    assert loaded.env.mask_free_fold is True
    assert loaded.model.d_model == ExperimentConfig.from_json(MASKFOLD).model.d_model
    assert ExperimentConfig.from_json(MASKFOLD).bc_init_checkpoint is None
    bc = ExperimentConfig.from_json(BC_CONFIG)
    assert bc.model.d_model == loaded.model.d_model
    assert bc.model.num_layers == loaded.model.num_layers
    assert bc.env.mask_free_fold is True
    assert bc.env.backend == "botzone"
    assert bc.ppo.lr == 0.001
    assert bc.ppo.epochs == 8
    assert bc.ppo.minibatch_size == 256


def test_bc_checkpoint_loads_on_the_ppo_resume_path(trainer, checker, tmp_path: Path):
    gen = _load("gen_rulebot_bc_for_train", ROOT / "scripts" / "gen_rulebot_bc.py")
    data = tmp_path / "data"
    arrays, meta = gen.collect_decisions(seed=11, hands=6, epsilon=0.2, mask_free_fold=True)
    gen.write_dataset(data, arrays, meta, force=False)
    ckpt = tmp_path / "bc.pt"
    trained = trainer.train_rulebot_bc(
        config=BC_CONFIG,
        data=data,
        out=ckpt,
        epochs=1,
        batch_size=16,
        device="cpu",
    )
    assert trained["legal_py_sha256"] == gen.current_legal_py_sha256()
    assert trained["mask_free_fold"] is True
    assert trained["data_seed"] == 11
    assert trained["kl_curve"] and all(
        np.isfinite(row["train_kl"]) and np.isfinite(row["val_kl"]) for row in trained["kl_curve"]
    )
    assert set(trained["val_game_ids"]).isdisjoint(trained["train_game_ids"])
    assert len(trained["val_game_ids"]) == 1 or len(trained["val_game_ids"]) == max(
        1, int(round(0.1 * (len(trained["val_game_ids"]) + len(trained["train_game_ids"]))))
    )

    cfg = ExperimentConfig.from_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["config"])
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    blob = load_checkpoint(ckpt, model, optimizer, map_location="cpu", restore_rng=False)
    assert int(blob["update"]) == -1
    assert int(blob["update"]) + 1 == 0
    assert blob["bc_metadata"]["legal_py_sha256"] == gen.current_legal_py_sha256()
    assert blob["bc_metadata"]["env_version"]["legal_py_sha256"] == blob["bc_metadata"]["legal_py_sha256"]
    assert blob["bc_metadata"]["data_seed"] == 11
    assert blob["bc_metadata"]["mask_free_fold"] is True
    ExperimentConfig.from_dict(blob["config"])
    for param in model.parameters():
        assert torch.isfinite(param).all()

    collapse = _load("eval_botzone_collapse_train_test", ROOT / "eval" / "eval_botzone_collapse.py")
    loaded, _tokenizer, loaded_cfg, path, update = collapse.load_botzone_model(
        ckpt.parent, torch.device("cpu"), ckpt_path=ckpt
    )
    assert update == -1
    assert path == ckpt
    assert loaded_cfg.env.backend == "botzone"
    assert loaded_cfg.env.mask_free_fold is True
    assert torch.allclose(
        next(model.parameters()).cpu(),
        next(loaded.parameters()).cpu(),
    )

    report = checker.evaluate_checkpoint(
        ckpt=ckpt,
        data=data,
        on_policy_hands=2,
        seed=11,
        device="cpu",
    )
    assert report["nan_inf"] is False
    assert report["kl_curve"]
    assert "legal_entropy" in report["validation"]
    assert "legal_entropy" in report["on_policy"]
    assert "free_check_fold_ratio" in report["validation"]
    assert report["gates"]["validation"]["legal_entropy_gt_0.2"]["value"] == report["validation"]["legal_entropy"]
    # The frequency gates are not required to pass on one epoch. The free-check
    # mask is: fold is not a legal sample there.
    for split in ("validation", "on_policy"):
        block = report[split]
        if block["free_check_count"]:
            assert block["free_check_fold_ratio"] == 0.0
            assert block["free_check_mask_allows_fold"] is False

    stale = tmp_path / "stale"
    arrays, meta = gen.load_dataset(data)
    meta["env_version"]["legal_py_sha256"] = "f" * 64
    gen.write_dataset(stale, arrays, meta, force=False)
    with pytest.raises(trainer._gen().LegalPyMismatch, match="refusing"):
        trainer.train_rulebot_bc(config=BC_CONFIG, data=stale, out=tmp_path / "no.pt", epochs=1, device="cpu")

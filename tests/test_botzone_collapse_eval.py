"""M9 collapse eval: free-check folds, legal entropy, sample and argmax passes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.model import masked_log_probs
from alphanl.transformer.tables import make_table
from alphanl.transformer.tokenizer import BettingHistoryTokenizer
from alphanl.transformer.train import build_model

ROOT = Path(__file__).resolve().parents[1]
LAPTOP_CONFIG = ROOT / "configs" / "transformer" / "botzone_laptop_8gb.json"
RUN_COMMAND = (
    "PYTHONPATH=src python eval/eval_botzone_collapse.py "
    "--ckpt-dir runs/smoke_botzone_m9/ckpt --decisions 2000 --seed 42"
)


def _load_eval():
    path = ROOT / "eval" / "eval_botzone_collapse.py"
    spec = importlib.util.spec_from_file_location("eval_botzone_collapse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclass looks the class up in sys.modules while the body is executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval()


def test_is_free_check_is_the_training_helper():
    from alphanl.transformer.decision_stats import is_free_check

    assert EVAL.is_free_check is is_free_check


def test_botzone_laptop_keeps_last_20_checkpoints():
    cfg = ExperimentConfig.from_json(LAPTOP_CONFIG)
    assert cfg.train.keep_last == 20
    assert ExperimentConfig().train.keep_last == 3


PASS_KEYS = (
    "n_decisions",
    "n_hands",
    "n_skipped_forced",
    "mean_legal_entropy",
    "mean_legal_entropy_free_check",
    "free_check_count",
    "free_check_fold_count",
    "free_check_fold_fraction",
    "free_check_fold_wilson95",
    "action_count",
    "action_share",
    "mode",
)
ACTION_NAMES = ("fold", "check_call", "raise_min", "raise_large", "all_in")


def _free_check_snapshot():
    """Big blind after the small blind limps: nothing owed, fold still legal."""

    table = make_table("botzone", seed=1)
    table.reset()
    assert table.current_player() == table.sb_seat
    table.step(1)
    snap = table.snapshot(table.current_player())
    assert snap.to_call == 0.0
    assert 0 in snap.legal_action_ids
    assert 1 in snap.legal_action_ids
    batch = BettingHistoryTokenizer(max_seq_len=64).encode(snap)
    return snap, batch["legal_mask"]


def _assert_pass(block: dict, decisions: int) -> None:
    for key in PASS_KEYS:
        assert key in block
    assert block["n_decisions"] == decisions
    assert block["n_skipped_forced"] >= 0
    assert block["mean_legal_entropy"] >= 0.0
    shares = block["action_share"]
    assert set(shares) == set(ACTION_NAMES)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)
    for name in ACTION_NAMES:
        assert 0.0 <= shares[name] <= 1.0
        assert block["action_count"][name] >= 0
    fraction = block["free_check_fold_fraction"]
    if block["free_check_count"] == 0:
        assert fraction is None
        assert block["free_check_fold_wilson95"] is None
        assert block["mean_legal_entropy_free_check"] is None
        return
    assert 0.0 <= fraction <= 1.0
    assert fraction == pytest.approx(block["free_check_fold_count"] / block["free_check_count"])
    low, high = block["free_check_fold_wilson95"]
    assert 0.0 <= low <= fraction <= high <= 1.0
    assert block["mean_legal_entropy_free_check"] >= 0.0


def test_script_reports_both_passes(tmp_path: Path):
    assert RUN_COMMAND in EVAL.__doc__
    # Dry-run construction: laptop botzone config, random init, no training.
    cfg = ExperimentConfig.from_json(LAPTOP_CONFIG)
    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint_path(ckpt_dir, 0),
        model=model,
        optimizer=None,
        update=0,
        config=cfg.to_dict(),
    )
    out = tmp_path / "collapse.json"
    payload = EVAL.main(
        [
            "--ckpt-dir",
            str(ckpt_dir),
            "--decisions",
            "50",
            "--seed",
            "1",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    saved = json.loads(out.read_text())
    assert saved == payload
    for key in (
        "ckpt",
        "update",
        "backend",
        "small_blind",
        "big_blind",
        "starting_stack",
        "opponent",
        "seed",
        "decisions",
        "device",
        "entropy_thr",
        "fold_thr",
        "sample",
        "argmax",
        "verdict",
    ):
        assert key in saved
    assert saved["backend"] == "botzone"
    assert saved["small_blind"] == 50.0
    assert saved["big_blind"] == 100.0
    assert saved["starting_stack"] == 20000.0
    assert saved["opponent"] == "rule"
    assert saved["seed"] == 1
    assert saved["decisions"] == 50
    assert saved["device"] == "cpu"
    assert saved["entropy_thr"] == 0.2
    assert saved["fold_thr"] == 0.5
    assert saved["verdict"] in ("COLLAPSED", "NOT COLLAPSED")
    _assert_pass(saved["sample"], 50)
    _assert_pass(saved["argmax"], 50)
    fold = saved["sample"]["free_check_fold_fraction"]
    entropy = saved["sample"]["mean_legal_entropy"]
    collapsed = fold is not None and entropy < saved["entropy_thr"] and fold > saved["fold_thr"]
    assert saved["verdict"] == ("COLLAPSED" if collapsed else "NOT COLLAPSED")


def test_fake_policy_free_check_fold_fraction():
    snap, mask = _free_check_snapshot()
    fold_logits = torch.full((1, 5), -1.0e9)
    fold_logits[0, 0] = 0.0
    _masked, _log_probs, fold_probs, legal = masked_log_probs(fold_logits, mask)
    assert bool(legal[0, 0].item())
    assert float(fold_probs[0, 0]) == pytest.approx(1.0)
    assert float(mask[0, 0]) > 0.5

    stay_logits = torch.full((1, 5), -1.0e9)
    stay_logits[0, 1] = 0.0
    _masked, _log_probs, stay_probs, _legal = masked_log_probs(stay_logits, mask)
    assert float(stay_probs[0, 0]) == pytest.approx(0.0)
    assert float(stay_probs[0, 1]) == pytest.approx(1.0)

    for mode in ("sample", "argmax"):
        always = EVAL.evaluate_policy_on_mask(fold_logits, mask, snap.to_call, mode=mode)
        always_summary = EVAL.summarize_decisions([always])
        assert always.free_check
        assert always.action == 0
        assert always_summary["free_check_fold_fraction"] == 1.0
        assert always_summary["free_check_count"] == 1

        never = EVAL.evaluate_policy_on_mask(stay_logits, mask, snap.to_call, mode=mode)
        never_summary = EVAL.summarize_decisions([never])
        assert never.free_check
        assert never.action != 0
        assert never_summary["free_check_fold_fraction"] == 0.0
        assert never_summary["free_check_fold_count"] == 0


def test_same_seed_writes_identical_json(tmp_path: Path):
    cfg = ExperimentConfig.from_json(LAPTOP_CONFIG)
    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint_path(ckpt_dir, 3),
        model=model,
        optimizer=None,
        update=3,
        config=cfg.to_dict(),
    )
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    args = [
        "--ckpt-dir",
        str(ckpt_dir),
        "--decisions",
        "50",
        "--seed",
        "1",
        "--device",
        "cpu",
    ]
    EVAL.main([*args, "--out", str(first)])
    EVAL.main([*args, "--out", str(second)])
    assert first.read_text() == second.read_text()
    saved = json.loads(first.read_text())
    assert saved["seed"] == 1
    assert saved["update"] == 3
    assert saved["sample"]["n_decisions"] == 50
    assert saved["argmax"]["n_decisions"] == 50

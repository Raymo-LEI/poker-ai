"""M11b checkpoint eval: bb/100, 50-hand matches, rule and random opponents."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from alphanl.botzone.rule_eval import bb_per_100_interval, match_scores
from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.train import build_model

ROOT = Path(__file__).resolve().parents[1]
BOTZONE_CONFIG = ROOT / "configs" / "transformer" / "botzone_laptop_8gb.json"
RLCARD_CONFIG = ROOT / "configs" / "transformer" / "laptop_8gb.json"
RUN_COMMAND = (
    "PYTHONPATH=src python eval/eval_botzone_checkpoint.py "
    "--ckpt-dir runs/ckpt_botzone --hands 5000 --seed 42 --mode both"
)

TOP_KEYS = (
    "ckpt",
    "ckpt_dir",
    "ckpt_file",
    "update",
    "backend",
    "small_blind",
    "big_blind",
    "starting_stack",
    "seed",
    "hands",
    "device",
    "mode",
    "modes",
    "amp",
    "reference_R",
    "reference_R_hands",
    "reference_R_seed",
    "reference_R_hero_seat",
    "reference_R_matchup",
    "reference_R_note",
    "match_length",
    "min_matches",
    "ci_z",
    "seat_schedule",
    "opponents",
    "pass_lines",
)
BLOCK_KEYS = (
    "opponent",
    "mode",
    "n_hands",
    "big_blind",
    "mean_bb_per_hand",
    "std_bb_per_hand",
    "std_divisor",
    "bb_per_100",
    "ci95_low",
    "ci95_high",
    "ci95",
    "match_length",
    "n_matches",
    "mean_match_score",
    "match_scores",
    "n_match_wins",
    "n_match_ties",
    "n_match_losses",
    "match_win_fraction",
    "match_score_unit",
    "tail_hands_dropped",
    "chip_deltas",
)


def _load_eval():
    path = ROOT / "eval" / "eval_botzone_checkpoint.py"
    spec = importlib.util.spec_from_file_location("eval_botzone_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval()


def _tiny_botzone_config() -> ExperimentConfig:
    cfg = ExperimentConfig.from_json(BOTZONE_CONFIG)
    cfg.name = "m11b_tiny"
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 32
    cfg.model.dropout = 0.0
    cfg.model.max_seq_len = 64
    return cfg


def _save_tiny(tmp_path: Path, update: int = 7) -> Path:
    cfg = _tiny_botzone_config()
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint_path(ckpt_dir, update),
        model=build_model(cfg),
        optimizer=None,
        update=update,
        config=cfg.to_dict(),
    )
    return ckpt_dir


def _hand_ci(deltas):
    """mean ± 1.96 * sample std / sqrt(n), std with divisor n-1, in bb/100."""

    blind = 100.0
    bbs = [delta / blind for delta in deltas]
    n = len(bbs)
    mean = sum(bbs) / n
    var = sum((value - mean) ** 2 for value in bbs) / (n - 1)
    std = math.sqrt(var)
    half = 1.96 * std / math.sqrt(n)
    return mean, std, mean * 100.0, (mean - half) * 100.0, (mean + half) * 100.0


def test_ci_formula_matches_a_hand_computed_interval():
    deltas = [0, 200]
    mean, std, bb100, low, high = _hand_ci(deltas)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(math.sqrt(2.0))
    assert bb100 == pytest.approx(100.0)
    assert low == pytest.approx(-96.0)
    assert high == pytest.approx(296.0)

    summary = EVAL.summarize_chip_deltas(deltas)
    assert summary["std_divisor"] == "n-1"
    assert summary["mean_bb_per_hand"] == pytest.approx(mean)
    assert summary["std_bb_per_hand"] == pytest.approx(std)
    assert summary["bb_per_100"] == pytest.approx(bb100)
    assert summary["ci95_low"] == pytest.approx(low)
    assert summary["ci95_high"] == pytest.approx(high)
    assert summary["ci95"] == pytest.approx([low, high])

    same = bb_per_100_interval(deltas, big_blind=100)
    assert summary["bb_per_100"] == pytest.approx(same[2])
    assert summary["ci95_low"] == pytest.approx(same[3])
    assert summary["ci95_high"] == pytest.approx(same[4])
    assert summary["std_bb_per_hand"] == pytest.approx(same[1])


def test_match_score_slicing_keeps_two_blocks_from_120_hands():
    deltas = [100] * 50 + [-100] * 50 + [50] * 20
    assert len(deltas) == 120
    assert sum(deltas[:50]) / 100 == 50
    assert sum(deltas[50:100]) / 100 == -50
    assert match_scores(deltas) == [50.0, -50.0]

    summary = EVAL.summarize_chip_deltas(deltas)
    assert summary["n_hands"] == 120
    assert summary["match_length"] == 50
    assert summary["n_matches"] == 2
    assert summary["tail_hands_dropped"] == 20
    assert summary["match_scores"] == [50.0, -50.0]
    assert summary["mean_match_score"] == pytest.approx(0.0)
    assert summary["n_match_wins"] == 1
    assert summary["n_match_ties"] == 0
    assert summary["n_match_losses"] == 1
    assert summary["match_win_fraction"] == pytest.approx(0.5)

    tied = EVAL.summarize_chip_deltas([0] * 50 + [1])
    assert tied["n_matches"] == 1
    assert tied["tail_hands_dropped"] == 1
    assert tied["match_scores"] == [0.0]
    assert tied["mean_match_score"] == 0.0
    assert tied["n_match_wins"] == 0
    assert tied["n_match_ties"] == 1
    assert tied["match_win_fraction"] == 0.0


def test_pass_line_thresholds():
    def opponents(rule_ci, random_bb, rule_matches, random_matches):
        def block(ci_low, bb, n_matches):
            return {"ci95_low": ci_low, "bb_per_100": bb, "n_matches": n_matches}

        return {
            "rule": {"argmax": block(rule_ci, 0.0, rule_matches)},
            "random": {"argmax": block(0.0, random_bb, random_matches)},
        }

    on_the_line = EVAL.pass_line(opponents(0.0, 540.95, 100, 99), "argmax", 540.95)
    assert on_the_line["vs_rule_ci_low_gt_0"]["pass"] is False
    assert on_the_line["vs_rule_ci_low_gt_0"]["result"] == "FAIL"
    assert on_the_line["vs_random_bb_per_100_ge_R"]["pass"] is True
    assert on_the_line["vs_random_bb_per_100_ge_R"]["result"] == "PASS"
    assert on_the_line["n_matches_ge_100"]["rule"]["pass"] is True
    assert on_the_line["n_matches_ge_100"]["random"]["pass"] is False
    assert on_the_line["n_matches_ge_100"]["random"]["result"] == "FAIL"

    clear = EVAL.pass_line(opponents(0.01, 540.94, 100, 100), "argmax", 540.95)
    assert clear["vs_rule_ci_low_gt_0"]["pass"] is True
    assert clear["vs_random_bb_per_100_ge_R"]["pass"] is False
    assert clear["n_matches_ge_100"]["rule"]["result"] == "PASS"
    assert clear["n_matches_ge_100"]["random"]["result"] == "PASS"


def test_seat_rotation_matches_the_collapse_schedule():
    assert EVAL.seats_for_hand(0) == (0, 0)
    assert EVAL.seats_for_hand(1) == (0, 1)
    assert EVAL.seats_for_hand(2) == (1, 0)
    assert EVAL.seats_for_hand(3) == (1, 1)
    assert EVAL.seats_for_hand(4) == (0, 0)
    blinds = []
    for index in range(4):
        hero, dealer = EVAL.seats_for_hand(index)
        blinds.append("BB" if hero == dealer else "SB")
    assert blinds == ["BB", "SB", "SB", "BB"]


def test_loader_picks_newest_or_ckpt_file_and_rejects_other_games(tmp_path: Path):
    cfg = _tiny_botzone_config()
    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    older = checkpoint_path(ckpt_dir, 1)
    newer = checkpoint_path(ckpt_dir, 5)
    save_checkpoint(older, model=model, optimizer=None, update=1, config=cfg.to_dict())
    save_checkpoint(newer, model=model, optimizer=None, update=5, config=cfg.to_dict())
    device = torch.device("cpu")

    _model, _tok, loaded_cfg, path, update = EVAL.load_botzone_model(ckpt_dir, device)
    assert update == 5
    assert path == newer
    assert loaded_cfg.env.backend == "botzone"

    _model, _tok, _cfg, picked, picked_update = EVAL.load_botzone_model(
        ckpt_dir, device, ckpt_path=older
    )
    assert picked_update == 1
    assert picked == older

    rlcard = ExperimentConfig.from_json(RLCARD_CONFIG)
    rlcard_dir = tmp_path / "rlcard"
    save_checkpoint(
        checkpoint_path(rlcard_dir, 0),
        model=build_model(rlcard),
        optimizer=None,
        update=0,
        config=rlcard.to_dict(),
    )
    with pytest.raises(ValueError, match="botzone backend"):
        EVAL.load_botzone_model(rlcard_dir, device)

    bad = tmp_path / "bad_blinds.pt"
    torch.save(
        {
            "format": 2,
            "update": 0,
            "config": {
                "env": {
                    "backend": "botzone",
                    "small_blind": 5.0,
                    "big_blind": 10.0,
                    "starting_stack": 1000.0,
                }
            },
        },
        bad,
    )
    with pytest.raises(ValueError, match="20000"):
        EVAL.load_botzone_model(tmp_path, device, ckpt_path=bad)


def test_main_reports_both_opponents_and_replays(tmp_path: Path):
    assert RUN_COMMAND in EVAL.__doc__
    ckpt_dir = _save_tiny(tmp_path, update=7)
    args = [
        "--ckpt-dir",
        str(ckpt_dir),
        "--hands",
        "60",
        "--seed",
        "1",
        "--device",
        "cpu",
    ]
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    payload = EVAL.main([*args, "--out", str(first)])
    EVAL.main([*args, "--out", str(second)])
    assert first.read_text() == second.read_text()
    saved = json.loads(first.read_text())
    assert saved == payload

    for key in TOP_KEYS:
        assert key in saved
    assert saved["seed"] == 1
    assert saved["hands"] == 60
    assert saved["device"] == "cpu"
    assert saved["mode"] == "argmax"
    assert saved["modes"] == ["argmax"]
    assert saved["amp"] is False
    assert saved["update"] == 7
    assert saved["backend"] == "botzone"
    assert saved["small_blind"] == 50.0
    assert saved["big_blind"] == 100.0
    assert saved["starting_stack"] == 20000.0
    assert saved["ckpt_file"] is None
    assert saved["reference_R"] == pytest.approx(540.95)
    assert saved["reference_R_hands"] == 10000
    assert saved["reference_R_seed"] == 42
    assert saved["reference_R_hero_seat"] == 0
    assert saved["reference_R_matchup"] == "rule_vs_random"
    assert saved["match_length"] == 50
    assert saved["min_matches"] == 100
    assert saved["ci_z"] == pytest.approx(1.96)
    assert set(saved["opponents"]) == {"rule", "random"}
    assert saved["opponents"]["rule"]["name"] == "botzone_rule_bot"
    assert saved["opponents"]["random"]["name"] == "uniform_legal_action_ids"

    for opponent in ("rule", "random"):
        block = saved["opponents"][opponent]["argmax"]
        for key in BLOCK_KEYS:
            assert key in block
        assert block["n_hands"] == 60
        assert block["n_matches"] == 1
        assert block["tail_hands_dropped"] == 10
        assert len(block["chip_deltas"]) == 60
        assert len(block["match_scores"]) == 1
        assert block["match_win_fraction"] == pytest.approx(
            block["n_match_wins"] / block["n_matches"]
        )
        mean, std, bb100, low, high = _hand_ci(block["chip_deltas"])
        assert block["mean_bb_per_hand"] == pytest.approx(mean)
        assert block["std_bb_per_hand"] == pytest.approx(std)
        assert block["bb_per_100"] == pytest.approx(bb100)
        assert block["ci95_low"] == pytest.approx(low)
        assert block["ci95_high"] == pytest.approx(high)
        assert block["ci95_low"] <= block["bb_per_100"] <= block["ci95_high"]
        same = bb_per_100_interval(block["chip_deltas"], big_blind=100)
        assert block["bb_per_100"] == pytest.approx(same[2])
        assert block["ci95"] == pytest.approx([same[3], same[4]])

    rule_check = saved["pass_lines"]["argmax"]["vs_rule_ci_low_gt_0"]
    random_check = saved["pass_lines"]["argmax"]["vs_random_bb_per_100_ge_R"]
    assert rule_check["pass"] is (saved["opponents"]["rule"]["argmax"]["ci95_low"] > 0.0)
    assert random_check["pass"] is (
        saved["opponents"]["random"]["argmax"]["bb_per_100"] >= saved["reference_R"]
    )
    assert rule_check["result"] == ("PASS" if rule_check["pass"] else "FAIL")
    assert random_check["result"] == ("PASS" if random_check["pass"] else "FAIL")
    for opponent in ("rule", "random"):
        item = saved["pass_lines"]["argmax"]["n_matches_ge_100"][opponent]
        assert item["n_matches"] == 1
        assert item["pass"] is False
        assert item["result"] == "FAIL"

    report = EVAL.render_report(saved)
    assert "540.95" in report
    assert "vs rule bot [argmax]" in report
    assert "vs random [argmax]" in report
    assert "matches >= 100 [rule argmax]" in report
    assert "matches >= 100 [random argmax]" in report
    assert "PASS" in report or "FAIL" in report
    assert "comparison only" in report


def test_same_seed_sample_and_both_modes(tmp_path: Path):
    ckpt_dir = _save_tiny(tmp_path, update=2)
    base = [
        "--ckpt-dir",
        str(ckpt_dir),
        "--hands",
        "4",
        "--seed",
        "1",
        "--device",
        "cpu",
    ]
    sample_a = tmp_path / "sample_a.json"
    sample_b = tmp_path / "sample_b.json"
    EVAL.main([*base, "--mode", "sample", "--out", str(sample_a)])
    EVAL.main([*base, "--mode", "sample", "--out", str(sample_b)])
    assert sample_a.read_text() == sample_b.read_text()
    sample = json.loads(sample_a.read_text())
    assert sample["modes"] == ["sample"]
    assert sample["opponents"]["rule"]["sample"]["n_hands"] == 4
    assert sample["opponents"]["random"]["sample"]["n_matches"] == 0
    assert sample["opponents"]["random"]["sample"]["mean_match_score"] is None

    both_path = tmp_path / "both.json"
    both = EVAL.main([*base, "--mode", "both", "--ckpt-file", str(checkpoint_path(ckpt_dir, 2)), "--out", str(both_path)])
    assert both["modes"] == ["sample", "argmax"]
    assert both["update"] == 2
    assert both["ckpt_file"] == str(checkpoint_path(ckpt_dir, 2))
    for opponent in ("rule", "random"):
        assert both["opponents"][opponent]["sample"]["n_hands"] == 4
        assert both["opponents"][opponent]["argmax"]["n_hands"] == 4
    assert set(both["pass_lines"]) == {"sample", "argmax"}
    report = EVAL.render_report(both)
    assert "M11 §5.4.4 pass line is the sample pass" in report

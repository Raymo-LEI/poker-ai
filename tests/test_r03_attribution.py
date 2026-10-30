"""R03 window manifest, unreachable teacher labels, and the V6 attribution suite."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "eval"))
sys.path.insert(0, str(_REPO / "eval" / "r03"))

from alphanl.cfr.fidelity import own_reach_mixture, posterior_mixture
from alphanl.cfr.teacher_window import (
    archive_kwargs,
    assert_window_matches,
    load_teacher_window,
    missing_teacher_record,
)
from run_attribution import absolute_nuts, card_id

_JUDGE = _REPO / "tests" / "fixtures" / "botzone_judge.py"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"
_NPZ_SHA = "dfd57d307c6d4d9eb5d789ba145370792752ea85f5f43ba57b5dd787e3b4faaa"


def test_teacher_window_manifest_is_the_150_through_350_range():
    window = load_teacher_window(_REPO / "eval" / "r03" / "teacher_window_manifest.json")
    assert archive_kwargs(window) == {
        "after": 149,
        "through": 350,
        "stride": 1,
        "sampling_power": 1.0,
    }
    assert window["inclusive_iterations"][0] == 150
    assert window["inclusive_iterations"][-1] == 350
    assert window["unweighted_fill"] is False
    assert len(window["inclusive_iterations"]) == 201


def test_shifted_window_is_refused(tmp_path):
    path = tmp_path / "window.json"
    path.write_text(json.dumps({
        "format": "route1_r03_teacher_window_v1",
        "after_exclusive": 1,
        "through": 3,
        "stride": 1,
        "sampling_power": 1,
    }), encoding="utf-8")
    window = load_teacher_window(path)
    manifest = {
        "selection": {"after_exclusive": 0, "through": 3, "stride": 1},
        "sampling_power": 1.0,
        "iterations": [1, 2, 3],
    }
    try:
        assert_window_matches(manifest, window)
    except ValueError as exc:
        assert "MISSING" in str(exc)
    else:
        raise AssertionError("shifted window was accepted")


def test_zero_own_reach_is_not_filled_with_the_unweighted_mean():
    probs = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    reach = torch.tensor([[0.1], [0.9]])
    weights = torch.tensor([1.0, 3.0])
    labelled = own_reach_mixture(probs, reach, weights)
    assert labelled["status"] == "OK"
    assert labelled["unweighted_fill"] is False
    torch.testing.assert_close(labelled["target_probs"], posterior_mixture(probs, reach, weights))
    unweighted = probs.double().mean(0)
    assert not torch.allclose(labelled["target_probs"], unweighted)
    missing = own_reach_mixture(probs, torch.zeros(2, 1), weights)
    assert missing["status"] == "UNREACHABLE/OOD"
    assert missing["target_probs"].sum() == 0
    assert missing["unweighted_fill"] is False
    record = missing_teacher_record(load_teacher_window(_REPO / "eval" / "r03" / "teacher_window_manifest.json"), "absent")
    assert record["status"] == "MISSING"
    assert record["target_probs"] is None


def test_river_king_spade_flush_is_absolute_nuts():
    hole = [card_id("Ks"), card_id("3s")]
    board = [card_id(card) for card in ("As", "9s", "4s", "2h", "7d")]
    assert absolute_nuts(hole, board)
    assert not absolute_nuts([card_id("Qs"), card_id("3s")], board)


def test_attribution_suite_matches_the_hard_checks():
    root = _REPO / "eval" / "r03"
    summary = json.loads((root / "attribution_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["pt_np_max_abs"] <= 1e-4
    assert summary["illegal_mass_deploy_max"] == 0
    assert summary["illegal_mass_raw_max"] <= 1e-12
    assert summary["discrete_diagnostics_identical"] is True
    assert summary["teacher_status"] == "MISSING"
    assert summary["frozen_student_status"] == "MISSING"
    assert summary["gpu_hours_used"] == 0
    assert hashlib.sha256(_JUDGE.read_bytes()).hexdigest() == _JUDGE_SHA
    assert hashlib.sha256((root / "weights" / "V6_R6_MODEL.npz").read_bytes()).hexdigest() == _NPZ_SHA
    lines = (root / "attribution.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summary["cases"] == 29
    index = json.loads((root / "counterexample_suite" / "index.json").read_text(encoding="utf-8"))
    assert len(index) == len(lines)
    grades = set()
    for line, discrete in zip(lines, index):
        record = json.loads(line)
        assert record["source"]
        assert record["proof_grade"] in ("hard_dominated", "research_pending")
        grades.add(record["proof_grade"])
        assert record["teacher"]["status"] == "MISSING"
        assert record["teacher"]["target_probs"] is None
        assert record["teacher"]["unweighted_fill"] is False
        assert record["frozen_student"]["status"] == "MISSING"
        assert record["frozen_student"]["probs"] is None
        assert abs(sum(record["ppo_raw_probs"]) - 1) < 1e-9
        assert abs(sum(record["deploy_probs"]) - 1) < 1e-9
        assert record["illegal_mass_deploy"] == 0
        assert record["pt_np_max_abs"] <= 1e-4
        assert "opponent" not in json.dumps(record["hero_hole"])
        path = root / "counterexample_suite" / f"{record['case_id']}.json"
        assert json.loads(path.read_text(encoding="utf-8"))["case_id"] == discrete["case_id"]
        if record["baseline_defect"]:
            assert record["proof_grade"] == "hard_dominated"
            assert record["ppo_raw_probs"][0] > 0.01
            assert record["root_cause_layer"] == "ppo"
    assert grades == {"hard_dominated", "research_pending"}
    assert summary["baseline_defects"]
    assert summary["unattributed"]
    assert any(item["case_id"] == "river_ks_nut_flush_facing_bet" for item in summary["baseline_defects"])

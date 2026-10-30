"""R07 plans the Teacher without inventing curves or dropping quarter-pot."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "eval" / "r07"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"


def test_plan_keeps_three_seeds_quarter_pot_and_no_fake_curves():
    frozen = json.loads((_ROOT / "frozen_teacher_manifest.json").read_text(encoding="utf-8"))
    assert frozen["quarter_pot_retained"] is True
    assert frozen["seeds"] == [101, 202, 303]
    assert frozen["milestones"] == [50, 100, 200, 400, 800]
    assert frozen["a6_primary"]["num_actions"] == 6
    names = []
    for seed in (101, 202, 303):
        manifest = json.loads((_ROOT / "manifests" / f"a6_seed{seed}.json").read_text(encoding="utf-8"))
        assert manifest["quarter_pot_required"] is True
        assert "quarter_pot" in manifest["action_names"]
        assert manifest["loss_iteration_weight_power"] == 1
        assert manifest["do_not_treat_t_squared_loss_as_t_squared_average"] is True
        assert manifest["burn_in_frozen_before_scores"] is True
        assert manifest["distill_authorized"] is False
        assert manifest["max_gpu_hours_this_seed"] <= 6
        names.append(manifest["seed"])
    assert names == [101, 202, 303]
    curves = json.loads((_ROOT / "learning_curves.json").read_text(encoding="utf-8"))
    assert curves["status"] == "NOT_MEASURED"
    assert curves["t_squared_loss_used_as_mixture"] is False
    holdout = json.loads((_ROOT / "holdout_report.json").read_text(encoding="utf-8"))
    assert holdout["status"] == "NOT_RUN"
    assert holdout["lbr_abr_is_full_exploitability"] is False
    assert holdout["auto_extend_training"] is False
    selection = json.loads((_ROOT / "selection_rationale.json").read_text(encoding="utf-8"))
    assert selection["selected_teacher"] is None
    assert selection["best_seed_only"] is False
    assert selection["distill_authorized"] is False
    debit = json.loads((_ROOT / "budget_debit.json").read_text(encoding="utf-8"))
    assert debit["gpu_hours_used_this_task"] == 0
    assert debit["total_gpu_hours_cap"] == 1000
    assert debit["cap_raised"] is False
    assert debit["distill_hours"] == 0
    assert debit["per_milestone_hours_measured"] is False
    assert debit["worst_case_serial_hours_if_every_seed_hits_the_stop"] == 36
    assert debit["production_width5_trainer"] == "FORBIDDEN"
    assert selection["certified_river_raw_fold_status"] == "NOT_MEASURED"
    assert selection["a6_vs_a5_cost_benefit"] == "NOT_MEASURED"
    assert selection["thresholds"]["primary_point_estimate_min_mbb_per_hand"] == 20
    assert len(debit["jobs_serial"]) == 6
    digest = hashlib.sha256((_REPO / "tests" / "fixtures" / "botzone_judge.py").read_bytes()).hexdigest()
    assert digest == _JUDGE_SHA


def test_dry_run_and_cpu_gate_for_a6_and_a5():
    for name, width, quarter in (
        ("a6_seed101.json", 6, True),
        ("a5_seed101.json", 5, False),
    ):
        manifest = _ROOT / "manifests" / name
        dry = subprocess.run(
            [sys.executable, str(_ROOT / "launch_teacher.py"), "--manifest", str(manifest)],
            cwd=_REPO,
            capture_output=True,
            text=True,
        )
        assert dry.returncode == 0, dry.stderr
        payload = json.loads(dry.stdout)
        assert payload["status"] == "DRY_RUN_OK"
        assert payload["head_width"] == width
        assert payload["quarter_pot"] is quarter
        executed = subprocess.run(
            [sys.executable, str(_ROOT / "launch_teacher.py"), "--manifest", str(manifest), "--execute"],
            cwd=_REPO,
            capture_output=True,
            text=True,
        )
        assert executed.returncode == 0, executed.stderr
        gate = json.loads(executed.stdout)
        assert gate["cpu_gate_ok"] is True
        assert gate["head_width"] == width
        assert gate["rejected_botzone_5action_v1"] is True
        assert gate["started_gpu"] is False
        assert gate["gpu_hours_used"] == 0
        assert gate["status"] == "PENDING_AUTODL_PARENT"

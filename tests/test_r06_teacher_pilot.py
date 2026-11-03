"""R06 Kuhn evidence is real, and the A6 pilot is not a renamed five-wide run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "eval" / "r06"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"


def test_kuhn_report_separates_approximation_from_hold_em():
    report = json.loads((_ROOT / "teacher_pilot_report.json").read_text(encoding="utf-8"))
    tabular = report["kuhn_tabular"]
    assert tabular["final_exploitability"] < tabular["initial_exploitability"] / 8
    assert abs(tabular["player_zero_value"] - tabular["equilibrium_value"]) < 0.02
    assert tabular["payoff"]["zero_sum"] is True
    assert tabular["own_reach"]["matches"] is True
    assert report["hunl_convergence_claimed"] is False
    assert report["deep_cfr_approximation"]["not_a_hunl_result"] is True
    assert report["strength_claim"] is False
    assert report["gpu_hours_used"] == 0
    restore = report["snapshot_restore"]
    assert restore["resume_metrics_match_continuation"] is True
    assert restore["iteration_equals_weight"] is True
    assert restore["both_seats_sampled"] is True
    assert restore["losses_finite"] is True


def _launch(name: str, execute: bool) -> subprocess.CompletedProcess:
    command = [sys.executable, str(_ROOT / "launch_pilot.py"), "--manifest", str(_ROOT / "manifests" / name)]
    if execute:
        command.append("--execute")
    return subprocess.run(command, cwd=_REPO, capture_output=True, text=True)


def test_a5_and_a6_dry_run_and_cpu_execute_match_schema_width():
    manifest = json.loads((_ROOT / "manifests" / "a6.json").read_text(encoding="utf-8"))
    assert manifest["quarter_pot_required"] is True
    assert "quarter_pot" in manifest["action_names"]
    assert manifest["num_actions"] == 6
    assert manifest["max_gpu_hours"] <= 0.25
    assert manifest["total_gpu_hours_cap"] == 1000
    for name, width, quarter in (("a5.json", 5, False), ("a6.json", 6, True)):
        dry = _launch(name, execute=False)
        assert dry.returncode == 0, dry.stderr
        dry_payload = json.loads(dry.stdout)
        assert dry_payload["status"] == "DRY_RUN_OK"
        assert dry_payload["head_width"] == width
        assert dry_payload["quarter_pot"] is quarter
        assert dry_payload["gpu_hours_used"] == 0
        executed = _launch(name, execute=True)
        assert executed.returncode == 0, executed.stderr
        payload = json.loads(executed.stdout)
        assert payload["cpu_execute_ok"] is True
        assert payload["head_width"] == width
        assert payload["quarter_pot_in_head"] is quarter
        assert payload["started_gpu"] is False
        assert payload["gpu_hours_used"] == 0
        assert payload["rejected_botzone_5action_v1"] is True
        assert payload["status"] == "PENDING_AUTODL_PARENT"
    selection = json.loads((_ROOT / "candidate_spec_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_candidate"] is None
    assert selection["quarter_pot_retained"] is True
    digest = hashlib.sha256((_REPO / "tests" / "fixtures" / "botzone_judge.py").read_bytes()).hexdigest()
    assert digest == _JUDGE_SHA

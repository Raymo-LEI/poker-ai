"""A timer must not mix live profiles, repeat completed work, or race writers."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.evaluation_io import EvaluationBusy, atomic_json, output_lock, sha256_file
from alphanl.cfr.evaluation_schedule import (ArchiveChanged, freeze_prefix, load_config,
                                            read_index, run_pending)
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer


ROOT = Path(__file__).resolve().parents[1]


def archive(path, count=3):
    path.mkdir()
    for i in range(1, count + 1):
        (path / f"iteration_{i:07d}.pt").write_bytes(f"snapshot-{i}".encode())
    atomic_json(path / "index.json", {"format": "sd_cfr_archive_v1",
                "completed_iteration": count, "iterations": list(range(1, count + 1))})
    return path


def config(tmp_path, source, **evaluation):
    path = tmp_path / "periodic.json"
    atomic_json(path, {"strategies": str(source), "out_dir": "evaluations",
                       "milestones": [1, 2, 3], "max_evaluations": 1,
                       "evaluation": evaluation})
    return load_config(path)


def fake_evaluator(calls):
    def execute(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        raw = output / "raw.jsonl"
        raw.write_text('{"chips": 0}\n')
        atomic_json(output / "report.json", {"format": "sd_cfr_readiness_v1",
                    "identity": str(output), "completed_utc": "2026-10-27T00:00:00Z",
                    "results": {"test": {"raw_log": raw.name, "raw_log_sha256": sha256_file(raw)}},
                    "launch_gate": {"verdict": "NO_GO"}})
        return SimpleNamespace(returncode=0)
    return execute


def test_poll_is_bounded_skips_completed_and_keeps_failure_verdicts(tmp_path):
    source = archive(tmp_path / "source")
    cfg = config(tmp_path, source)
    calls = []
    for _ in range(4):
        result = run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    assert len(calls) == 3
    assert [c[c.index("--hero-iteration") + 1] for c in calls] == ["1", "2", "3"]
    assert all("--resume" in c for c in calls)
    assert all(j["verdict"] == "NO_GO" and j["state"] == "complete" for j in result["jobs"].values())
    assert result["state"] == "idle"


def test_partial_run_is_retried_then_completed_once(tmp_path):
    source = archive(tmp_path / "source", 1)
    cfg = config(tmp_path, source)
    with pytest.raises(RuntimeError, match="exited 7"):
        run_pending(cfg, repository=ROOT, execute=lambda *a, **k: SimpleNamespace(returncode=7))
    calls = []
    result = run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    assert len(calls) == 1 and "--resume" in calls[0]
    assert result["state"] == "idle" and "error" not in result
    run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    assert len(calls) == 1


def test_freeze_does_not_follow_future_snapshot_replacements(tmp_path):
    source = archive(tmp_path / "source")
    frozen = freeze_prefix(source, tmp_path / "freeze", 2)
    assert read_index(frozen)["iterations"] == [1, 2]
    assert not (frozen / "iteration_0000003.pt").exists()
    replacement = source / "replacement.tmp"
    replacement.write_bytes(b"a-different-training-history")
    replacement.replace(source / "iteration_0000001.pt")
    assert (frozen / "iteration_0000001.pt").read_bytes() == b"snapshot-1"
    with pytest.raises(ValueError, match="training history changed"):
        freeze_prefix(source, tmp_path / "freeze", 2)


def test_freeze_copies_across_filesystems_and_detects_publisher_races(tmp_path):
    source = archive(tmp_path / "source")
    with patch("alphanl.cfr.evaluation_schedule.os.link", side_effect=OSError("cross-device")):
        frozen = freeze_prefix(source, tmp_path / "copy", 2)
    assert (frozen / "iteration_0000002.pt").read_bytes() == b"snapshot-2"
    assert (frozen / "iteration_0000002.pt").stat().st_ino != (source / "iteration_0000002.pt").stat().st_ino
    original_link = os.link
    def replace_during_link(src, dst):
        original_link(src, dst)
        replacement = source / "new.tmp"
        replacement.write_bytes(b"replacement")
        replacement.replace(src)
    with patch("alphanl.cfr.evaluation_schedule.os.link", side_effect=replace_during_link):
        with pytest.raises(ArchiveChanged):
            freeze_prefix(source, tmp_path / "race", 2)
    assert not (tmp_path / "race").exists()
    assert not list(tmp_path.glob(".freeze-*"))


def test_unpublished_or_sparse_snapshots_do_not_get_evaluated(tmp_path):
    source = archive(tmp_path / "source", 1)
    (source / "iteration_0000002.pt").write_bytes(b"unpublished")
    cfg = config(tmp_path, source)
    calls = []
    run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    assert len(calls) == 1
    atomic_json(source / "index.json", {"format": "sd_cfr_archive_v1",
                "completed_iteration": 3, "iterations": [1, 3]})
    with pytest.raises(ValueError, match="snapshot_every=1"):
        run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))


def test_changed_budgets_get_a_new_result_and_corrupt_logs_do_not_skip(tmp_path):
    source = archive(tmp_path / "source", 1)
    cfg = config(tmp_path, source)
    calls = []
    first = run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    cfg["evaluation"]["pairs"] += 1
    second = run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))
    assert len(calls) == 2 and len(second["jobs"]) == 2
    assert calls[0][calls[0].index("--out") + 1] != calls[1][calls[1].index("--out") + 1]
    report = Path(list(first["jobs"].values())[0]["report"])
    report.with_name("raw.jsonl").unlink()
    cfg["evaluation"]["pairs"] -= 1
    with pytest.raises(ValueError, match="missing or changed raw log"):
        run_pending(cfg, repository=ROOT, execute=fake_evaluator(calls))


def test_locks_prevent_parallel_publishers_and_release_on_failure(tmp_path):
    source = archive(tmp_path / "source", 1)
    cfg = config(tmp_path, source)
    with output_lock(cfg["out_dir"]):
        with pytest.raises(EvaluationBusy):
            run_pending(cfg, repository=ROOT)
    with pytest.raises(RuntimeError):
        with output_lock(cfg["out_dir"]):
            raise RuntimeError("interrupted")
    with output_lock(cfg["out_dir"]):
        pass


@pytest.mark.parametrize("field,value", [("milestones", [True]), ("max_evaluations", 0),
                                         ("evaluation", {"typo": 3})])
def test_configuration_rejects_invalid_inputs(tmp_path, field, value):
    path = tmp_path / "config.json"
    data = {"strategies": "source", "out_dir": "results", field: value}
    atomic_json(path, data)
    with pytest.raises(ValueError):
        load_config(path)


def test_real_cli_poll_resume_and_completed_skip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cfg = CFRConfig(separate_player_models=True, model=ModelConfig(d_model=16, nhead=2, num_layers=1,
                                     dim_feedforward=32, dropout=0, max_seq_len=32))
    model = RegretNetwork(PokerTransformer(**asdict(cfg.model)))
    torch.save({"format": "sd_cfr_strategy_v1", "iteration": 1, "weight": 1,
                "rules_version": RULES_VERSION, "numeric_version": NUMERIC_VERSION,
                "config": cfg.to_dict(), "models": {0: model.state_dict(), 1: model.state_dict()}},
               source / "iteration_0000001.pt")
    atomic_json(source / "index.json", {"format": "sd_cfr_archive_v1",
                "completed_iteration": 1, "iterations": [1]})
    config(tmp_path, source, pairs=2, matches=1, health_hands=4,
           opponents=["call"], stages=["duplicate", "matches", "health"], equity_samples=4)
    command = [sys.executable, str(ROOT / "scripts/eval_cfr_periodic.py"),
               "--config", str(tmp_path / "periodic.json")]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    report_path = next((tmp_path / "evaluations/results").rglob("report.json"))
    first = json.loads(report_path.read_text())
    assert first["completed_utc"] and first["hero_iteration"] == 1
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(report_path.read_text())["completed_utc"] == first["completed_utc"]
    # A crash after one task preserves completed raw logs; CLI checks them on resume.
    first.pop("completed_utc")
    report_path.write_text(json.dumps(first))
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "reuse duplicate:1_vs_call" in report_path.with_name("evaluation.log").read_text()

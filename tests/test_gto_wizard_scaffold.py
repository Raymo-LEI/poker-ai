"""Offline GTO Wizard scaffold and R08 handoff stubs. No network and no weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.gto_wizard.adapter_hunl import (  # noqa: E402
    AdapterError,
    FakeSnapshot,
    action_for_street_total,
    new_table,
    prepare_spot,
    refuse_weight_source,
    to_wizard_action,
)
from eval.gto_wizard.client import ApiError, GtoWizardClient, live_refusal  # noqa: E402
from eval.gto_wizard.compare import CompareError, compare_distributions  # noqa: E402
from eval.gto_wizard.schema import SCHEMA_A5, SCHEMA_A6  # noqa: E402

FIXTURE = ROOT / "eval/gto_wizard/fixtures/spots.json"
OFFLINE = ROOT / "scripts/eval_gto_wizard_offline.py"
FAKE = ROOT / "scripts/compare_teacher_fixture.py"
MANIFEST = ROOT / "eval/r08/manifests/export_from_teacher.example.json"
PYTHON = sys.executable


def _run(script: Path, args: list[str], *, env=None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    merged.pop("GTOWIZARD_API_KEY", None)
    merged.pop("PYTHONPATH", None)
    if env:
        merged.update(env)
    return subprocess.run(
        [PYTHON, str(script), *args],
        cwd=ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def test_known_two_action_math_via_a5_padding():
    policy = [0.0, 1.0, 0.0, 0.0, 0.0]
    reference = [0.0, 0.5, 0.0, 0.5, 0.0]
    metrics = compare_distributions(policy, reference, SCHEMA_A5, legal_action_ids=(0, 1, 3, 4))
    assert metrics["kl"] == pytest.approx(1.0 * __import__("math").log(1 / 0.5))
    assert metrics["l1"] == pytest.approx(1.0)
    assert metrics["same_bucket"] is True
    assert metrics["policy_bucket"] == "check_call"


def test_same_bucket_false_and_undefined_kl_fails_closed():
    moved = compare_distributions(
        [0.05, 0.5, 0.0, 0.35, 0.1],
        [0.4, 0.2, 0.0, 0.3, 0.1],
        SCHEMA_A5,
        legal_action_ids=(0, 1, 3, 4),
    )
    assert moved["same_bucket"] is False
    assert moved["policy_bucket"] == "check_call"
    assert moved["reference_bucket"] == "fold"
    with pytest.raises(CompareError, match="undefined"):
        compare_distributions([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], SCHEMA_A5)
    with pytest.raises(CompareError, match="illegal"):
        compare_distributions(
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            SCHEMA_A5,
            legal_action_ids=(0, 1, 3, 4),
        )


def test_quarter_pot_maps_on_hunl_and_off_tree_fails(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    flop = next(spot for spot in document["spots"] if spot["spot_id"] == "a6_flop_cbet")
    prepared = prepare_spot(flop)
    assert prepared["snapshot"].me_seat == 0
    assert 5 in prepared["legal_action_ids"]
    assert to_wizard_action(prepared["table"], SCHEMA_A6, 5) == {"action": "b", "amount": 100}
    assert to_wizard_action(prepared["table"], SCHEMA_A6, 1) == {"action": "k"}
    assert action_for_street_total(prepared["table"], SCHEMA_A6, 100) == 5
    with pytest.raises(AdapterError, match="off-tree"):
        action_for_street_total(prepared["table"], SCHEMA_A6, 333)
    preflop = new_table(3)
    assert to_wizard_action(preflop, SCHEMA_A5, 0) == {"action": "f"}
    assert to_wizard_action(preflop, SCHEMA_A5, 1) == {"action": "c"}
    assert to_wizard_action(preflop, SCHEMA_A5, 3) == {"action": "b", "amount": 200}
    with pytest.raises(AdapterError, match="illegal"):
        to_wizard_action(preflop, SCHEMA_A5, 5)
    with pytest.raises(AdapterError, match="illegal"):
        to_wizard_action(preflop, SCHEMA_A6, 5)
    fake = FakeSnapshot(SCHEMA_A6, flop["policy_probs"])
    assert fake.probabilities(prepared["snapshot"])[5] == pytest.approx(0.3)
    refuse_weight_source(fake)
    with pytest.raises(AdapterError, match="filesystem"):
        refuse_weight_source(tmp_path / "autodl")
    from alphanl.cfr.strategy import SnapshotPolicy
    with pytest.raises(AdapterError, match="SnapshotPolicy"):
        refuse_weight_source(object.__new__(SnapshotPolicy))


def test_fixture_cli_writes_kl_l1_without_network(tmp_path):
    out = tmp_path / "report"
    result = _run(OFFLINE, ["--fixture", str(FIXTURE), "--out-dir", str(out)])
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["live"] is False and report["network_used"] is False
    assert report["botzone_used"] is False and report["training_modified"] is False
    assert report["leaderboard_submitted"] is False
    assert report["hands"] == 3
    assert report["chips"] is None and report["aivat_bb_per_100"] is None
    assert report["kl"] > 0 and report["l1"] > 0
    assert "mbb" not in json.dumps(report)
    flop = next(row for row in report["spots"] if row["spot_id"] == "a6_flop_cbet")
    assert flop["argmax_action"] == {"action": "b", "amount": 100}
    assert flop["quarter_pot_legal"] is True and flop["same_bucket"] is True
    preflop = next(row for row in report["spots"] if row["spot_id"] == "a5_preflop_sb")
    assert preflop["same_bucket"] is False
    assert preflop["argmax_action"] == {"action": "c"}
    blocked = next(row for row in report["spots"] if row["spot_id"] == "a6_preflop_sb_quarter_illegal")
    assert blocked["quarter_pot_legal"] is False and 5 not in blocked["legal_action_ids"]


def test_illegal_fixture_and_live_do_not_call_http(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    preflop = next(spot for spot in document["spots"] if spot["spot_id"] == "a6_preflop_sb_quarter_illegal")
    preflop["policy_probs"][5] = 0.1
    preflop["policy_probs"][1] = 0.3
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    refused = _run(OFFLINE, ["--fixture", str(path), "--out-dir", str(tmp_path / "bad")])
    assert refused.returncode == 2
    assert "SECRET" not in refused.stderr
    live = _run(OFFLINE, ["--live"], env={"GTOWIZARD_API_KEY": "SECRET-KEY"})
    assert live.returncode == 2
    body = json.loads(live.stderr)
    assert body["reason"] == "live_not_a_gate" and body["network_used"] is False
    assert "SECRET" not in live.stderr
    missing = _run(OFFLINE, ["--live"])
    assert missing.returncode == 2
    assert json.loads(missing.stderr)["reason"] == "missing_api_key"


def test_client_shape_and_fake_snapshot_script(tmp_path):
    class Transport:
        def __init__(self):
            self.calls = []
        def post(self, path, payload, headers):
            self.calls.append((path, payload, headers))
            return {"hand_id": 7, "ok": True}
    transport = Transport()
    client = GtoWizardClient(transport, "local-test-key")
    client.new_hand()
    client.act(7, "b", 200)
    assert transport.calls[0][0] == "/hands"
    assert transport.calls[0][1] == {"game_name": "HUNL 200BB"}
    assert transport.calls[0][2]["X-API-Key"] == "local-test-key"
    assert transport.calls[1] == ("/hands/7/act", {"action": "b", "amount": 200}, {"X-API-Key": "local-test-key"})
    with pytest.raises(ApiError):
        client.act(7, "f", 1)
    assert live_refusal("") == "missing_api_key"
    assert live_refusal("abc") == "live_not_a_gate"
    result = _run(FAKE, ["--fixture", str(FIXTURE), "--out-dir", str(tmp_path / "fake")])
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "fake" / "report.json").read_text(encoding="utf-8"))
    assert report["weights_read"] is False and report["autodl_outputs_read"] is False
    assert report["kl"] > 0 and report["l1"] > 0
    assert report["chips"] is None
    assert "mbb" not in json.dumps(report)


def test_r08_stubs_stay_pending_and_reject_execute(tmp_path):
    text = (ROOT / "eval/r08/result.json").read_text(encoding="utf-8")
    assert "PENDING_TEACHER" in text and "NOT_RUN" in text
    assert "mbb" not in text.lower()
    selection = _run(
        ROOT / "eval/r08/build_selection_report.py",
        ["--manifest", str(MANIFEST), "--out", str(tmp_path / "selection.json")],
    )
    assert selection.returncode == 0, selection.stderr
    report = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert report["status"] == "PENDING_TEACHER" and report["selected_teacher"] is None
    assert report["weights_read"] is False and report["metrics"] == "NOT_RUN"
    assert report["num_actions"] == 6 and report["quarter_pot_required"] is True
    assert report["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert "mbb" not in json.dumps(report)
    executed = _run(
        ROOT / "eval/r08/build_selection_report.py",
        ["--manifest", str(MANIFEST), "--out", str(tmp_path / "nope.json"), "--execute"],
    )
    assert executed.returncode == 2 and not (tmp_path / "nope.json").exists()
    claimed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    claimed["mbb_per_hand"] = 12
    claimed_path = tmp_path / "claimed.json"
    claimed_path.write_text(json.dumps(claimed), encoding="utf-8")
    rejected = _run(
        ROOT / "eval/r08/build_selection_report.py",
        ["--manifest", str(claimed_path), "--out", str(tmp_path / "claimed-out.json")],
    )
    assert rejected.returncode == 2 and not (tmp_path / "claimed-out.json").exists()
    filled = json.loads(MANIFEST.read_text(encoding="utf-8"))
    filled["teacher_out_dir"] = str(tmp_path / "weights")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "iteration_0000001.pt").write_bytes(b"not-a-weight")
    filled_path = tmp_path / "filled.json"
    filled_path.write_text(json.dumps(filled), encoding="utf-8")
    bundle = _run(
        ROOT / "eval/r08/export_teacher_bundle.py",
        ["--manifest", str(filled_path), "--out-dir", str(tmp_path / "bundle")],
    )
    assert bundle.returncode == 2 and not (tmp_path / "bundle").exists()
    empty = _run(
        ROOT / "eval/r08/export_teacher_bundle.py",
        ["--manifest", str(MANIFEST), "--out-dir", str(tmp_path / "bundle-ok")],
    )
    assert empty.returncode == 0, empty.stderr
    body = json.loads((tmp_path / "bundle-ok" / "bundle.json").read_text(encoding="utf-8"))
    assert body["status"] == "PENDING_TEACHER" and body["weights"] == [] and body["files_copied"] == 0
    assert "mbb" not in json.dumps(body)
    gate = _run(ROOT / "eval/r08/distill_gate.py", ["--execute"])
    assert gate.returncode == 2
    assert "scripts/distill_cfr.py" in gate.stdout
    assert "distill_started" in gate.stdout
    payload = json.loads(gate.stdout.strip().splitlines()[-1])
    assert payload["distill_started"] is False and payload["command_executed"] is False
    assert "mbb" not in gate.stdout.lower()

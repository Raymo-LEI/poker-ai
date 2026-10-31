"""Live GTO Wizard runner with a mock transport. No researcher socket."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.gto_wizard.adapter_hunl import AdapterError  # noqa: E402
from eval.gto_wizard.client import ApiError, GtoWizardClient, RequestsTransport  # noqa: E402
from eval.gto_wizard.live import project_hand, run_session, select_wizard_action  # noqa: E402
from eval.gto_wizard.policy import PolicyLoadError, load_frozen_policy  # noqa: E402

LIVE = ROOT / "scripts/eval_gto_wizard_live.py"
PYTHON = sys.executable
SECRET = "SECRET-KEY"


def _load_cli():
    spec = importlib.util.spec_from_file_location("gto_wizard_live_cli", LIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args, *, env=None):
    merged = os.environ.copy()
    merged.pop("GTOWIZARD_API_KEY", None)
    merged.pop("PYTHONPATH", None)
    if env:
        merged.update(env)
    return subprocess.run(
        [PYTHON, str(LIVE), *args],
        cwd=ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def _opening(**overrides):
    position = overrides.pop("position", "SB")
    hole = overrides.pop("hole", "AhKd")
    history = overrides.pop("history", [])
    board = overrides.pop("board", "")
    street = overrides.pop("street", "preflop")
    pot = overrides.pop("pot", 150)
    common = overrides.pop("common", 0)
    stacks = overrides.pop("stacks", {"SB": 19950, "BB": 19900})
    legal = overrides.pop("legal", ["f", "c", "b"])
    raise_range = overrides.pop("raise_range", {"min": 200, "max": 20000})
    over = overrides.pop("over", False)
    winnings = overrides.pop("winnings", None)
    aivat = overrides.pop("aivat", None)
    to_act = overrides.pop("to_act", "user")
    folded = overrides.pop("folded", False)
    hand_id = overrides.pop("hand_id", 1)
    if overrides:
        raise AssertionError(f"unused {sorted(overrides)}")
    villain = "BB" if position == "SB" else "SB"
    return {
        "hand_id": hand_id,
        "game": {
            "game_id": 1,
            "game_name": "HUNL 200BB",
            "game_format": "HUNL",
            "starting_stack": 20000,
            "blinds": [50, 100],
            "stack_reset_per_hand": True,
        },
        "game_state": {
            "street": street,
            "common_pot": common,
            "total_pot": pot,
            "board_cards": board,
            "is_hand_over": over,
            "player_to_act": to_act,
            "players": [
                {"name": "user", "stack": stacks[position], "position": position, "hole_cards": hole},
                {"name": "gto_wizard", "stack": stacks[villain], "position": villain, "hole_cards": None},
            ],
            "legal_actions": legal,
            "raise_range": raise_range,
            "action_history": history,
            "has_gto_wizard_folded": folded,
            "winnings": winnings,
            "aivat_score": aivat,
        },
    }


def _terminal(hand_id, *, winnings, aivat, hole="AhKd"):
    return _opening(
        hand_id=hand_id,
        hole=hole,
        history=["f"],
        over=True,
        to_act=None,
        legal=[],
        raise_range=None,
        winnings=winnings,
        aivat=aivat,
        pot=0,
    )


class ScriptTransport:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    def post(self, path, payload, headers):
        self.calls.append((path, dict(payload), dict(headers)))
        if "leaderboard" in path or path in ("/results", "/winnings"):
            raise AssertionError(path)
        if not self.steps:
            raise AssertionError("unexpected post " + path)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def test_help_documents_thousand_hand_command():
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    text = result.stdout
    assert "u250_retained.pt" in text
    assert "--hands 1000" in text
    assert "GTOWIZARD_API_KEY" in text
    assert "leaderboard" in text.lower()
    assert "model_structure" in text


def test_missing_key_and_over_cap_do_not_construct_transport(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.delenv("GTOWIZARD_API_KEY", raising=False)

    def boom(_key):
        raise AssertionError("socket")

    missing = cli.main(
        ["--policy", "missing.pt", "--config", "missing.json", "--hands", "1", "--out-dir", str(tmp_path / "a")],
        transport_factory=boom,
    )
    assert missing == 2
    monkeypatch.setenv("GTOWIZARD_API_KEY", SECRET)
    capped = cli.main(
        ["--policy", "missing.pt", "--config", "missing.json", "--hands", "1001", "--out-dir", str(tmp_path / "b")],
        transport_factory=boom,
    )
    assert capped == 2
    zero = cli.main(
        ["--policy", "missing.pt", "--config", "missing.json", "--hands", "0", "--out-dir", str(tmp_path / "c")],
        transport_factory=boom,
    )
    assert zero == 2
    refused = _run(
        ["--policy", "missing.pt", "--config", "missing.json", "--hands", "1", "--out-dir", str(tmp_path / "d")],
    )
    assert refused.returncode == 2
    assert json.loads(refused.stderr)["reason"] == "missing_api_key"
    assert SECRET not in refused.stderr
    present = _run(
        ["--policy", str(tmp_path / "no.pt"), "--config", str(tmp_path / "no.json"), "--hands", "1",
         "--out-dir", str(tmp_path / "e")],
        env={"GTOWIZARD_API_KEY": SECRET},
    )
    assert present.returncode == 2
    body = json.loads(present.stderr)
    assert body["reason"] == "policy_refused" and body["network_used"] is False
    assert SECRET not in present.stderr and SECRET not in present.stdout


def test_client_refuses_leaderboard_and_hides_the_key():
    transport = ScriptTransport([])
    client = GtoWizardClient(transport, SECRET)
    for path in ("/leaderboard", "/results", "/winnings", "/hands/in-progress"):
        with pytest.raises(ApiError, match="endpoint"):
            client._post(path, {})
    assert transport.calls == []
    assert SECRET not in repr(client)
    http = RequestsTransport(SECRET)
    assert SECRET not in repr(http)
    live = GtoWizardClient(ScriptTransport([{"hand_id": 7}, {"hand_id": 7, "ok": True}]), SECRET)
    live.new_hand()
    live.act(7, "c")
    assert live.transport.calls[0][0] == "/hands"
    assert live.transport.calls[1][0] == "/hands/7/act"
    assert live.transport.calls[1][1] == {"action": "c"}


def test_projection_maps_check_and_rejects_short_raise():
    projected = project_hand(_opening())
    assert projected["hero_seat"] == 1
    assert projected["table"].hole_cards(1) == ("HA", "DK")
    assert select_wizard_action(projected, 0) == {"action": "f"}
    assert select_wizard_action(projected, 1) == {"action": "c"}
    assert select_wizard_action(projected, 3) == {"action": "b", "amount": 200}
    tight = project_hand(_opening(raise_range={"min": 400, "max": 20000}))
    with pytest.raises(AdapterError, match="off-tree"):
        select_wizard_action(tight, 3)
    checked = project_hand(_opening(
        position="BB",
        hole="AsKd",
        history=["c"],
        pot=200,
        stacks={"SB": 19900, "BB": 19900},
        legal=["k", "b"],
        raise_range={"min": 200, "max": 20000},
    ))
    assert checked["hero_seat"] == 0
    assert select_wizard_action(checked, 1) == {"action": "k"}
    raised = project_hand(_opening(
        history=["b200", "b800"],
        pot=1000,
        stacks={"SB": 19800, "BB": 19200},
        legal=["f", "c", "b"],
        raise_range={"min": 1400, "max": 20000},
    ))
    assert raised["hero_seat"] == 1
    assert int(raised["table"].state.total_pot_amount) == 1000
    with pytest.raises(AdapterError, match="off-tree|illegal"):
        project_hand(_opening(history=["b150"]))
    with pytest.raises(AdapterError, match="board"):
        project_hand(_opening(board="AhKd3c"))


def test_mock_session_records_metrics_and_does_not_retry_or_submit(tmp_path):
    transport = ScriptTransport([
        _opening(hand_id=1),
        _terminal(1, winnings=-50, aivat=-40),
        _opening(hand_id=2, hole="QcJd"),
        _terminal(2, winnings=-50, aivat=80, hole="QcJd"),
    ])
    client = GtoWizardClient(transport, SECRET)
    out = tmp_path / "session"
    report, code = run_session(
        client,
        lambda table, hero, ctx: 0,
        hands=2,
        out_dir=out,
        policy_name="u250_retained.pt",
        config_name="config.json",
        redact=SECRET,
    )
    assert code == 0
    assert [call[0] for call in transport.calls] == ["/hands", "/hands/1/act", "/hands", "/hands/2/act"]
    assert transport.calls[1][1] == {"action": "f"}
    assert transport.calls[1][2]["X-API-Key"] == SECRET
    assert report["live"] is True and report["leaderboard_submitted"] is False
    assert report["training_modified"] is False and report["botzone_used"] is False
    assert report["score_claimed"] is False and report["network_used"] is True
    assert report["hands"] == 2 and report["chips"] == -100 and report["bb"] == -1
    assert report["bb_per_100"] == -50
    assert report["aivat_score"] == 40 and report["aivat_bb_per_100"] == 20
    assert report["schema_id"] == "botzone_a5_v2"
    saved = (out / "report.json").read_text(encoding="utf-8")
    assert SECRET not in saved
    with pytest.raises(AdapterError, match="already exists"):
        run_session(client, lambda table, hero, ctx: 0, hands=1, out_dir=out, redact=SECRET)
    assert len(transport.calls) == 4

    blocked = ScriptTransport([_opening(raise_range={"min": 400, "max": 20000})])
    blocked_out = tmp_path / "blocked"
    stopped, status = run_session(
        GtoWizardClient(blocked, SECRET),
        lambda table, hero, ctx: 3,
        hands=1,
        out_dir=blocked_out,
        redact=SECRET,
    )
    assert status == 2 and stopped["stop_reason"] == "fail_closed"
    assert stopped["hands"] == 0 and stopped["leaderboard_submitted"] is False
    assert stopped["live"] is True and "off-tree" in stopped["error"]
    assert [call[0] for call in blocked.calls] == ["/hands"]
    assert SECRET not in (blocked_out / "report.json").read_text(encoding="utf-8")

    flaky = ScriptTransport([_opening(), ApiError("researcher API request failed: Timeout")])
    flaky_out = tmp_path / "flaky"
    failed, failed_code = run_session(
        GtoWizardClient(flaky, SECRET),
        lambda table, hero, ctx: 1,
        hands=1,
        out_dir=flaky_out,
        redact=SECRET,
    )
    assert failed_code == 2 and failed["hands"] == 0
    assert [call[0] for call in flaky.calls] == ["/hands", "/hands/1/act"]
    assert sum(path.endswith("/act") for path, _payload, _headers in flaky.calls) == 1
    assert failed["leaderboard_submitted"] is False and failed["training_modified"] is False


def test_frozen_policy_uses_stage2_loader_and_refuses_other_heads(tmp_path, monkeypatch):
    source = (ROOT / "eval/gto_wizard/policy.py").read_text(encoding="utf-8")
    assert "load_checkpoint" in source
    assert "PolicyNetwork(" not in source
    assert "u250_model_structure" not in source
    sidecar = tmp_path / "config.json"
    sidecar.write_text(json.dumps({"schema_id": "botzone_a6_v2"}), encoding="utf-8")
    dummy = tmp_path / "weights.pt"
    dummy.write_bytes(b"unused")
    with pytest.raises(PolicyLoadError, match="schema"):
        load_frozen_policy(dummy, sidecar)
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not-a-checkpoint")
    sidecar.write_text("{}", encoding="utf-8")
    with pytest.raises(PolicyLoadError, match="Stage 2"):
        load_frozen_policy(junk, sidecar)
    with pytest.raises(PolicyLoadError, match="files"):
        load_frozen_policy(tmp_path, sidecar)

    import alphanl.stage2.train as train

    class SnapshotPolicy:
        pass

    def snapshot_loader(path, device="cpu"):
        return SnapshotPolicy(), {}

    monkeypatch.setattr(train, "load_checkpoint", snapshot_loader)
    with pytest.raises(PolicyLoadError, match="SnapshotPolicy"):
        load_frozen_policy(dummy, sidecar)

    class ResidualExploiterPolicy:
        def __init__(self):
            self.delta_head = type("Head", (), {"out_features": 6})()
            self.settings = {}

        def eval(self):
            return self

    monkeypatch.setattr(train, "load_checkpoint", lambda path, device="cpu": (ResidualExploiterPolicy(), {}))
    with pytest.raises(PolicyLoadError, match="5-action"):
        load_frozen_policy(dummy, sidecar)

    import torch
    from alphanl.cfr.config import CFRConfig
    from alphanl.cfr.distillation import PolicyNetwork
    from alphanl.stage2.context import MatchCtx
    from alphanl.stage2.policy import ResidualExploiterPolicy as RealPolicy
    from alphanl.stage2.train import save_checkpoint
    from alphanl.transformer.config import ModelConfig
    from eval.gto_wizard.policy import Stage2Decider

    monkeypatch.undo()
    torch.manual_seed(5)
    tiny = CFRConfig(model=ModelConfig(d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                                       max_seq_len=24, dropout=0.0))
    mask = {"enabled": True, "spr_above": 2.5, "preflop_equity": 0.72, "postflop_equity": 0.80,
            "equity_samples": 4, "equity_seed": 17}
    model = RealPolicy(PolicyNetwork(tiny), botzone_native_allin_mask=mask).eval()
    path = tmp_path / "u250_retained.pt"
    save_checkpoint(path, model, update=250)
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "botzone_native_allin_mask": mask,
        "policy": {"botzone_native_allin_mask": mask},
    }), encoding="utf-8")
    loaded, blob = load_frozen_policy(path, good)
    assert type(loaded).__name__ == "ResidualExploiterPolicy"
    assert blob["update"] == 250 and loaded.delta_head.out_features == 5
    projected = project_hand(_opening())
    ctx = MatchCtx()
    ctx.begin_hand()
    action = Stage2Decider(loaded)(projected["table"], projected["hero_seat"], ctx)
    spec = select_wizard_action(projected, action)
    assert spec["action"] in {"f", "c", "b"}
    bad_mask = tmp_path / "mask.json"
    bad_mask.write_text(json.dumps({"botzone_native_allin_mask": {"enabled": False}}), encoding="utf-8")
    with pytest.raises(PolicyLoadError, match="mask"):
        load_frozen_policy(path, bad_mask)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "botzone_native_allin_mask": mask,
        "policy": {"botzone_native_allin_mask": {"enabled": False}},
    }), encoding="utf-8")
    with pytest.raises(PolicyLoadError, match="disagree"):
        load_frozen_policy(path, split)

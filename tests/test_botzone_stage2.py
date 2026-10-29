"""Stage 2 Botzone bundle: zip forward parity, Python 3.6 shim, judge matches."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.cfr.traversal import HunlState
from alphanl.stage2 import MatchCtx, ResidualExploiterPolicy
from alphanl.stage2.export import export_botzone_weights, pack_botzone_zip
from alphanl.stage2.train import save_checkpoint
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import HunlTable, HunlTokenizer

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {"num_players": 2, "dealer_id": 0, "my_id": 1, "my_chips": 19950, "my_cards": [51, 23],
           "public_cards": [], "history": [], "hand": 0, "max_hand": 50,
           "total_win_chips": [0, 0], "total_win_games": [0, 0]}
FLOP_REQUEST = {"num_players": 2, "dealer_id": 0, "my_id": 0, "my_chips": 19900, "my_cards": [51, 23],
                "public_cards": [3, 8, 41], "hand": 3, "max_hand": 50,
                "total_win_chips": [1200, -1200], "total_win_games": [2, 1],
                "history": [{"round": 0, "player_id": 1, "action": 0, "action_type": "call"},
                            {"round": 0, "player_id": 0, "action": 0, "action_type": "check"},
                            {"round": 1, "player_id": 1, "action": 100, "action_type": "raise"}]}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.manual_seed(3)
    root = tmp_path_factory.mktemp("stage2_botzone")
    config = CFRConfig(equity_features=True, model=ModelConfig(d_model=32, nhead=2, num_layers=1,
                       dim_feedforward=64, max_seq_len=24, dropout=.1))
    model = ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    with torch.no_grad():
        model.delta_head.weight.normal_(0, .1)
    checkpoint = root / "stage2.pt"
    save_checkpoint(checkpoint, model)
    workdir = root / "work"
    (workdir / "data").mkdir(parents=True)
    weights = workdir / "data" / "stage2_model.npz"
    export_botzone_weights(checkpoint, weights)
    zip_path = root / "stage2.zip"
    pack_botzone_zip(zip_path)
    return {"config": config, "model": model, "zip": zip_path, "weights": weights, "workdir": workdir}


def _probe(bundle, script, *, shim=False, python=sys.executable):
    """Run ``script`` against the zip's runtime in a fresh interpreter."""
    code = ("import sys, json\nsys.path.insert(0, %r)\n" % str(bundle["zip"])
            + ("import dataclasses_compat\nsys.modules['dataclasses'] = dataclasses_compat\n" if shim else "")
            + "import botzone_entry as entry\nentry._install_dataclasses()\nfrom stage2_runtime import runtime\n"
            + script)
    env = dict(os.environ, POKER_STAGE2_WEIGHTS=str(bundle["weights"]))
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, env=env, check=True,
                            cwd=str(bundle["workdir"]))
    return json.loads(result.stdout.strip().splitlines()[-1])


LOGITS_SCRIPT = """
agent, safety, path = entry.load_agent(runtime)
out = []
for request in %s:
    snap, responses = runtime.request_snapshot(request)
    ctx, _ = runtime.context_from_payload({"requests": [request]})
    obs = agent.observation(snap, ctx)
    out.append({"logits": agent.logits(obs)[0].tolist(), "responses": {str(k): v for k, v in responses.items()},
                "ctx": ctx.state_dict(), "legal": list(snap.legal_action_ids)})
print(json.dumps(out))
"""


def test_zip_forward_matches_torch_on_all_streets(bundle):
    config, model = bundle["config"], bundle["model"]
    table = HunlTable(seed=5)
    table.reset()
    ctx = MatchCtx()
    ctx.begin_hand()
    tokenizer = HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs())
    batches, snapshots = [], []
    while not table.is_over():
        batch = HunlState(table, tokenizer, CFRActionGuard.from_config(config)).observation()
        batch["match_ctx"] = ctx.tensor()
        batches.append(batch)
        snapshots.append(table.snapshot(table.current_player()))
        table.step(1)
    script = """
agent, safety, path = entry.load_agent(runtime)
schema = sys.modules['stage2_runtime.schema']
ctx = sys.modules['stage2_runtime.context'].MatchCtx(); ctx.begin_hand()
rows = []
for raw in %s:
    events = tuple(schema.ActionEvent(*e) for e in raw.pop('events'))
    snap = schema.HandSnapshot(events, *raw['rest'])
    obs = agent.observation(snap, ctx)
    rows.append({'obs': {k: v.tolist() for k, v in obs.items()}, 'logits': agent.logits(obs)[0].tolist()})
print(json.dumps(rows))
""" % repr([{"events": [list(e.__dict__.values()) for e in s.events],
             "rest": [s.hole_cards, s.board_cards, s.me_seat, s.sb_seat, s.street, s.pot, s.stacks,
                      s.to_call, s.legal_action_ids, s.big_blind, s.starting_stack]} for s in snapshots])
    rows = _probe(bundle, script)
    assert len(rows) == len(batches) >= 4
    for row, batch in zip(rows, batches):
        for key in batch:
            assert np.array_equal(np.asarray(row["obs"][key], dtype=batch[key].numpy().dtype), batch[key].numpy()), key
        expected = model(batch)[0].detach().numpy()[0]
        np.testing.assert_allclose(np.asarray(row["logits"], dtype=np.float32), expected, atol=2e-5, rtol=2e-5)


def test_dataclasses_shim_reproduces_real_dataclasses(bundle):
    script = LOGITS_SCRIPT % repr([REQUEST, FLOP_REQUEST])
    real = _probe(bundle, script)
    shimmed = _probe(bundle, script + "\nassert type(sys.modules['dataclasses']).__name__ == 'module'\n"
                     "assert sys.modules['dataclasses'].__name__ == 'dataclasses_compat'\n", shim=True)
    assert real == shimmed
    assert real[1]["ctx"]["hands_played"] == 3 and real[1]["ctx"]["hero_net_bb"] == 12.0


def test_entry_outputs_one_json_line_and_keeps_data_small(bundle):
    env = dict(os.environ, POKER_STAGE2_WEIGHTS=str(bundle["weights"]))
    payload = {"requests": [REQUEST], "responses": [], "data": "", "globaldata": "",
               "time_limit": 6, "memory_limit": 256}
    result = subprocess.run([sys.executable, str(bundle["zip"])], input=json.dumps(payload) + "\n",
                            capture_output=True, text=True, env=env, check=True, cwd=str(bundle["workdir"]))
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1 and not result.stderr
    body = json.loads(lines[0])
    assert isinstance(body["response"], int) and " ok " in body["debug"]
    assert len(body["debug"].encode()) <= 1024
    assert len(json.dumps(body["data"]).encode()) < 100 * 1024
    # Botzone may hand data back as a string; the entry accepts both and stays idempotent.
    payload["data"] = json.dumps(body["data"])
    again = subprocess.run([sys.executable, str(bundle["zip"])], input=json.dumps(payload) + "\n",
                           capture_output=True, text=True, env=env, check=True, cwd=str(bundle["workdir"]))
    assert json.loads(again.stdout)["response"] == body["response"]


@pytest.mark.parametrize("request_, expected", [(REQUEST, 0), (FLOP_REQUEST, 0)])
def test_missing_weights_fall_back_to_judge_legal_action(bundle, tmp_path, request_, expected):
    env = dict(os.environ, POKER_STAGE2_WEIGHTS=str(tmp_path / "missing.npz"))
    result = subprocess.run([sys.executable, str(bundle["zip"])], input=json.dumps({"requests": [request_]}) + "\n",
                            capture_output=True, text=True, env=env, check=True, cwd=str(tmp_path))
    body = json.loads(result.stdout)
    assert body["response"] == expected and "fallback" in body["debug"]
    garbage = subprocess.run([sys.executable, str(bundle["zip"])], input="not json\n",
                             capture_output=True, text=True, env=env, check=True, cwd=str(tmp_path))
    assert json.loads(garbage.stdout)["response"] == -1


def test_bundle_avoids_python37_syntax_and_torch(bundle):
    with zipfile.ZipFile(bundle["zip"]) as archive:
        names = archive.namelist()
        assert names[0] == "__main__.py" and "stage2_runtime/runtime.py" in names
        for name in names:
            text = archive.read(name).decode("utf-8")
            assert "from __future__ import annotations" not in text, name
            assert "import torch" not in text.replace("from . import arraytorch as torch", ""), name
            assert "np.matmul(" not in text, name


def test_committed_zip_is_current(tmp_path):
    fresh = tmp_path / "stage2.zip"
    pack_botzone_zip(fresh)
    with zipfile.ZipFile(fresh) as new, zipfile.ZipFile(ROOT / "botzone_bots" / "stage2.zip") as old:
        assert new.namelist() == old.namelist()
        for name in new.namelist():
            assert new.read(name) == old.read(name), "%s changed; run tools/pack_botzone_stage2.py" % name


def test_full_judge_match_under_platform_limits(bundle):
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "accept_botzone_stage2.py"),
                             "--zip", str(bundle["zip"]), "--weights", str(bundle["weights"]),
                             "--matches", "1", "--seed", "5"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    summary = json.loads(result.stdout[:result.stdout.rindex("}") + 1])
    assert summary["errors"] == 0 and summary["hero_decisions"] > 0
    assert summary["max_turn_s"] < 6.0


@pytest.mark.skipif(not os.environ.get("POKER_PY36"), reason="set POKER_PY36 to a Python 3.6 interpreter")
def test_python36_interpreter_runs_bundle(bundle):
    script = LOGITS_SCRIPT % repr([REQUEST, FLOP_REQUEST])
    old, new = _probe(bundle, script, python=os.environ["POKER_PY36"]), _probe(bundle, script)
    for a, b in zip(old, new):
        np.testing.assert_allclose(a.pop("logits"), b.pop("logits"), atol=1e-5, rtol=1e-5)
        assert a == b

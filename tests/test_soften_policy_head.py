"""pytest for soften_policy_head.py. Run from the repo root: PYTHONPATH=src python -m pytest -q test_soften_policy_head.py

Works both as ``tests/test_soften_policy_head.py`` in the repo (script at scripts/) and next to the script.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "scripts", HERE / "scripts"):
    if (cand / "soften_policy_head.py").is_file():
        sys.path.insert(0, str(cand))
        break

import soften_policy_head as sph  # noqa: E402

from alphanl.transformer.bc import save_bc_checkpoint  # noqa: E402
from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.model import PokerTransformer  # noqa: E402
from alphanl.transformer.tokenizer import NUM_NUMERIC, NUM_TYPES, VOCAB_SIZE  # noqa: E402
from alphanl.transformer.train import build_model, build_optimizer  # noqa: E402

REPO = sph.REPO_ROOT
FIXTURE = REPO / "tests" / "fixtures" / "bc_rulebot"


def _tiny_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig.from_json(REPO / "configs" / "transformer" / "botzone_bc_rulebot.json")
    cfg.device = "cpu"
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    return cfg


def _random_batch(n: int = 64, seq: int = 12, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    legal = (torch.rand(n, 5, generator=g) > 0.4).float()
    legal[:, 1] = 1.0  # check/call always legal
    legal[: n // 4, :] = 0.0
    legal[: n // 4, 1] = 1.0  # a quarter of the rows have a single legal action
    return {
        "token_ids": torch.randint(1, VOCAB_SIZE, (n, seq), generator=g),
        "type_ids": torch.randint(0, NUM_TYPES, (n, seq), generator=g),
        "numeric": torch.randn(n, seq, NUM_NUMERIC, generator=g),
        "pad_mask": torch.zeros(n, seq, dtype=torch.bool),
        "legal_mask": legal,
    }


def _model_with_big_head(seed: int = 0) -> PokerTransformer:
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    model = build_model(cfg)
    with torch.no_grad():  # make the head non-trivial so entropies are spread
        model.policy_head.weight.normal_(0, 1.0)
        model.policy_head.bias.normal_(0, 0.5)
    return model.eval()


def _write_ckpt(path: Path, model, cfg=None) -> Path:
    cfg = cfg or _tiny_cfg()
    opt = build_optimizer(model, cfg)
    save_bc_checkpoint(path, model=model, optimizer=opt, config=cfg.to_dict())
    return path


def test_head_is_plain_linear_and_key_names():
    model = _model_with_big_head()
    assert type(model.policy_head) is torch.nn.Linear and type(model.value_head) is torch.nn.Linear
    assert model.policy_head is not model.value_head
    keys = [k for k in model.state_dict() if k.startswith("policy_head")]
    assert keys == [sph.POLICY_WEIGHT, sph.POLICY_BIAS]


@pytest.mark.parametrize("T", [0.5, 2.0, 7.3])
def test_logits_over_T_equals_scaled_head(T):
    model = _model_with_big_head()
    batch = _random_batch()
    with torch.no_grad():
        raw, v0 = model(batch, mask_illegal=False)
    new_state = sph.soften_state_dict(model.state_dict(), T)
    soft = PokerTransformer(d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1, max_seq_len=64)
    soft.load_state_dict(new_state)
    soft.eval()
    with torch.no_grad():
        raw1, v1 = soft(batch, mask_illegal=False)
    assert torch.allclose(raw1, raw / T, rtol=1e-5, atol=1e-6)
    assert torch.equal(v0, v1)
    for k, v in model.state_dict().items():
        if k not in (sph.POLICY_WEIGHT, sph.POLICY_BIAS):
            assert torch.equal(v, new_state[k]), k
    assert torch.equal(new_state[sph.POLICY_WEIGHT], model.state_dict()[sph.POLICY_WEIGHT] / T)
    assert torch.equal(new_state[sph.POLICY_BIAS], model.state_dict()[sph.POLICY_BIAS] / T)


def test_bisection_hits_target_random_logits():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2000, 5, generator=g) * 6.0
    legal = (torch.rand(2000, 5, generator=g) > 0.35).float()
    legal[:, 1] = 1.0
    h1 = sph.mean_entropy(logits, legal, 1.0)
    hmax = sph.max_reachable_entropy(legal)
    for target in (h1 + 0.05, 0.5 * (h1 + hmax), hmax - 0.02):
        T, h = sph.find_temperature(logits, legal, target)
        assert T > 1.0
        assert abs(sph.mean_entropy(logits, legal, T) - target) < 1e-3
        assert abs(h - target) < 1e-3
    # sharpening only when explicitly allowed
    with pytest.raises(ValueError, match="allow-sharpen"):
        sph.find_temperature(logits, legal, 0.5 * h1)
    T, h = sph.find_temperature(logits, legal, 0.5 * h1, allow_sharpen=True)
    assert T < 1.0 and abs(h - 0.5 * h1) < 1e-3
    # unreachable
    with pytest.raises(ValueError, match="not reachable"):
        sph.find_temperature(logits, legal, hmax + 0.1)


def test_single_legal_row_has_zero_entropy():
    logits = torch.tensor([[3.0, -1.0, 2.0, 0.0, 1.0]])
    legal = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    assert float(sph.masked_entropy_rows(logits, legal, 5.0)) == 0.0


def test_argmax_invariance_under_T():
    model = _model_with_big_head(3)
    batch = _random_batch(256, seed=5)
    base, _ = sph.forward_all(model, batch)
    for T in (1.5, 4.0, 30.0):
        soft = PokerTransformer(d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1, max_seq_len=64)
        soft.load_state_dict(sph.soften_state_dict(model.state_dict(), T))
        soft.eval()
        out, _ = sph.forward_all(soft, batch)
        assert torch.equal(base.argmax(-1), out.argmax(-1))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_refuse_bad_temperature(bad):
    with pytest.raises(ValueError):
        sph.check_temperature(bad)
    with pytest.raises(ValueError):
        sph.soften_state_dict(_model_with_big_head().state_dict(), bad)


def test_cli_refusals(tmp_path):
    ckpt = _write_ckpt(tmp_path / "bc.pt", _model_with_big_head())
    out = tmp_path / "soft.pt"
    with pytest.raises(SystemExit):  # neither mode
        sph.main(["--ckpt", str(ckpt), "--out", str(out)])
    with pytest.raises(SystemExit):  # both modes
        sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "2", "--target-entropy", "0.3", "--val-data", str(FIXTURE)])
    with pytest.raises(SystemExit):  # target without val data
        sph.main(["--ckpt", str(ckpt), "--out", str(out), "--target-entropy", "0.3"])
    with pytest.raises(SystemExit):  # T <= 0
        sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "0"])
    with pytest.raises(SystemExit):  # nan
        sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "nan"])
    with pytest.raises(SystemExit):  # same file
        sph.main(["--ckpt", str(ckpt), "--out", str(ckpt), "--temperature", "2", "--force"])
    bad_out = REPO / "runs" / "some_other_run" / "x.pt"
    with pytest.raises(SystemExit):  # runs/ elsewhere
        sph.main(["--ckpt", str(ckpt), "--out", str(bad_out), "--temperature", "2"])
    assert not bad_out.exists()
    sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "2"])
    assert out.exists()
    with pytest.raises(SystemExit):  # no overwrite without --force
        sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "3"])
    sph.main(["--ckpt", str(ckpt), "--out", str(out), "--temperature", "3", "--force"])
    blob = sph.load_blob(out)
    assert blob["soften_metadata"]["temperature"] == 3.0


def test_allowed_runs_dir_is_bc_rulebot_train():
    sph.assert_out_path(REPO / "runs" / "bc_rulebot_train" / "never_written.pt", False)
    with pytest.raises(SystemExit):
        sph.assert_out_path(REPO / "runs" / "transformer_x" / "a.pt", False)


@pytest.mark.skipif(not (FIXTURE / "decisions.npz").is_file(), reason="fixture missing")
def test_end_to_end_fixture(tmp_path):
    ckpt = _write_ckpt(tmp_path / "bc.pt", _model_with_big_head(7))
    src = sph.load_blob(ckpt)
    out = tmp_path / "soft.pt"
    probe = sph.soften_checkpoint(ckpt, tmp_path / "probe.pt", temperature=1.0, val_data=FIXTURE, val_all=True, verbose=False)
    target = probe["stats"]["entropy_before"] + 0.1
    res = sph.soften_checkpoint(ckpt, out, target_entropy=target, val_data=FIXTURE, val_all=True, verbose=False)
    st = res["stats"]
    assert res["temperature"] > 1.0
    assert st["argmax_changed_rows"] == 0
    assert st["value_max_abs_diff"] == 0.0
    assert abs(st["entropy_after"] - target) < 1e-3
    # target below the current entropy: refused unless --allow-sharpen
    with pytest.raises(SystemExit):
        sph.soften_checkpoint(ckpt, tmp_path / "s.pt", target_entropy=0.5 * st["entropy_before"], val_data=FIXTURE, val_all=True, verbose=False)
    r2 = sph.soften_checkpoint(ckpt, tmp_path / "s.pt", target_entropy=0.5 * st["entropy_before"], val_data=FIXTURE,
                               val_all=True, allow_sharpen=True, verbose=False)
    assert r2["temperature"] < 1.0 and r2["stats"]["argmax_changed_rows"] == 0
    blob = sph.load_blob(out)
    assert blob["update"] == src["update"] == -1
    assert blob["soften_metadata"]["source_ckpt"] == str(ckpt)
    assert set(blob) == set(src) | {"soften_metadata"}
    assert json.dumps(blob["config"], sort_keys=True, default=str) == json.dumps(src["config"], sort_keys=True, default=str)
    # held-out split path (not --val-all) also runs
    sph.soften_checkpoint(ckpt, tmp_path / "soft2.pt", temperature=2.0, val_data=FIXTURE, verbose=False)

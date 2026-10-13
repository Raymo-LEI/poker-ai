"""C1 (AMP path) and C2 (checkpoint resume equivalence) tests. CPU only."""

import importlib.util
from pathlib import Path

import pytest
import torch

HAS_RLCARD = importlib.util.find_spec("rlcard") is not None


def _tiny_cfg(tmp_path: Path, num_updates: int):
    from alphanl.transformer.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.1  # dropout on, so torch RNG restore is exercised
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 3
    cfg.ppo.num_envs = 2
    cfg.ppo.rollout_steps = 4
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 4
    cfg.train.num_updates = num_updates
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 2
    cfg.train.log_every = 1
    cfg.train.amp = True  # must auto-disable on CPU
    cfg.pool.snapshot_every = 1
    cfg.pool.max_size = 3
    cfg.train.out_dir = str(tmp_path / "out")
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def test_amp_helpers_off_on_cpu():
    from alphanl.transformer.ppo import amp_enabled, make_grad_scaler

    assert amp_enabled(True, "cpu") is False
    assert amp_enabled(False, "cuda") is False
    assert make_grad_scaler(True, "cpu").is_enabled() is False


def test_masked_log_probs_fp32_no_nan_from_fp16_logits():
    from alphanl.transformer.model import masked_log_probs

    logits = torch.randn(3, 5).half()
    legal = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.float32)
    masked, logp, probs, _ = masked_log_probs(logits, legal)
    assert masked.dtype == torch.float32 and logp.dtype == torch.float32
    assert torch.isfinite(logp[legal > 0.5]).all()
    assert not torch.isnan(probs).any()
    assert torch.allclose(probs.sum(-1), torch.ones(3))


def test_ppo_update_amp_path_cpu():
    """Import the AMP code path; run with amp off, and amp=True on CPU (auto-off)."""
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.ppo import make_grad_scaler, ppo_update
    from alphanl.transformer.tokenizer import NUM_NUMERIC

    torch.manual_seed(0)
    model = PokerTransformer(d_model=32, nhead=4, num_layers=1, dim_feedforward=64, max_seq_len=16)
    n, seq = 8, 16
    batch = {
        "token_ids": torch.randint(1, 10, (n, seq)),
        "type_ids": torch.zeros(n, seq, dtype=torch.long),
        "numeric": torch.zeros(n, seq, NUM_NUMERIC),
        "pad_mask": torch.zeros(n, seq, dtype=torch.bool),
        "legal_mask": torch.ones(n, 5),
    }
    for amp in (False, True):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        m = ppo_update(
            model, opt, batch, actions=torch.randint(0, 5, (n,)), old_logp=torch.full((n,), -1.6),
            old_values=torch.zeros(n), advantages=torch.randn(n), returns=torch.randn(n),
            clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, epochs=1, minibatch_size=4,
            max_grad_norm=0.5, amp=amp, scaler=make_grad_scaler(amp, "cpu"),
        )
        assert all(torch.isfinite(torch.tensor(float(v))) for v in m.values())


def test_atomic_save_prune_and_latest(tmp_path: Path):
    from alphanl.transformer.checkpoint import checkpoint_path, latest_checkpoint, list_checkpoints, save_checkpoint

    model = torch.nn.Linear(2, 2)
    for u in range(5):
        save_checkpoint(checkpoint_path(tmp_path, u), model=model, optimizer=None, update=u, config={}, keep_last=3)
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["update_0000002.pt", "update_0000003.pt", "update_0000004.pt"]
    assert latest_checkpoint(tmp_path).name == "update_0000004.pt"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_resume_matches_continuous(tmp_path: Path):
    """2 updates + save, then resume 2 more == 4 continuous updates.
    Compares params, Adam state, pool/env/actor RNG and global torch RNG (CPU)."""
    from alphanl.transformer.checkpoint import latest_checkpoint
    from alphanl.transformer.train import run_training

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    cont = _tiny_cfg(tmp_path / "cont", 4)
    run_training(cont)
    a = load(latest_checkpoint(cont.train.checkpoint_dir))

    split = _tiny_cfg(tmp_path / "split", 2)
    run_training(split)
    assert latest_checkpoint(split.train.checkpoint_dir).name == "update_0000001.pt"
    split.train.num_updates = 4
    run_training(split, resume="latest")
    b = load(latest_checkpoint(split.train.checkpoint_dir))

    assert a["update"] == b["update"] == 3
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    assert sa.keys() == sb.keys() and len(sa) > 0
    for pid in sa:
        for k in sa[pid]:
            assert torch.equal(torch.as_tensor(sa[pid][k]), torch.as_tensor(sb[pid][k])), (pid, k)
    assert a["pool"]["rng"] == b["pool"]["rng"]
    assert [e["rng"] for e in a["envs"]] == [e["rng"] for e in b["envs"]]
    assert a["actor_rng"] == b["actor_rng"]
    assert torch.equal(a["rng"]["torch_cpu"], b["rng"]["torch_cpu"])

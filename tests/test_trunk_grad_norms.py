"""T-L1: shared-trunk policy/value gradient norms on metrics.csv.

The switch is ``alphanl.transformer.ppo._log_grad_norms_enabled``. Tests set
it false to skip ``torch.autograd.grad``. The three columns are still
written (as NaN), the same way M11a keeps its columns when its switch is off.

Definition, also in :func:`alphanl.transformer.ppo.trunk_grad_norms`:

* ``grad_norm_policy`` — global L2 of d(policy_side)/d(trunk), where
  ``policy_side = policy_loss - entropy_coef * entropy`` (the policy-side
  term of the total loss, entropy included with its minus sign and
  ``entropy_coef``). ``value_coef`` is not applied.
* ``grad_norm_value`` — global L2 of d(value_loss)/d(trunk). ``value_loss``
  is the unscaled 0.5 * squared error. ``value_coef`` is not applied.
* ``grad_norm_ratio = grad_norm_value / grad_norm_policy``.

Trunk parameters come only from :func:`shared_trunk_parameters`: embeddings
(``token_emb``, ``type_emb``, ``pos_emb``, ``numeric_proj``), transformer
``encoder``, and the shared post-encoder ``trunk`` MLP. ``policy_head`` and
``value_head`` are excluded. Measured once per update, on the first
minibatch of the first epoch, from that forward, before the real backward.
GradScaler scale is not applied.
"""

from __future__ import annotations

import csv
import math
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from alphanl.transformer.decision_stats import DECISION_STAT_COLUMNS
from alphanl.transformer.model import PokerTransformer, shared_trunk_parameters
from alphanl.transformer.ppo import (
    GRAD_NORM_COLUMNS,
    _global_l2_norm,
    ppo_losses,
    ppo_update,
    trunk_grad_norms,
)
from alphanl.transformer.tokenizer import NUM_NUMERIC, slice_batch
from alphanl.transformer.train import _append_csv

_LOSS_COLUMNS = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
)


def _model() -> PokerTransformer:
    return PokerTransformer(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=8,
    )


def _batch(n: int = 8, seq: int = 8) -> dict:
    return {
        "token_ids": torch.randint(1, 20, (n, seq)),
        "type_ids": torch.randint(0, 4, (n, seq)),
        "numeric": torch.randn(n, seq, NUM_NUMERIC),
        "pad_mask": torch.zeros(n, seq, dtype=torch.bool),
        "legal_mask": torch.ones(n, 5),
    }


def _targets(n: int):
    return (
        torch.randint(0, 5, (n,)),
        torch.full((n,), -1.6),
        torch.zeros(n),
        torch.randn(n),
        torch.randn(n),
    )


def _trunk_excluding_heads(model: torch.nn.Module):
    """Reference set: every parameter that is not a policy or value head."""

    banned = set()
    for name in ("policy_head", "value_head"):
        for param in getattr(model, name).parameters():
            banned.add(id(param))
    return [param for param in model.parameters() if id(param) not in banned]


def _l2_of_param_grads(params) -> float:
    total = torch.zeros(())
    saw = False
    for param in params:
        if param.grad is None:
            continue
        saw = True
        total = total + param.grad.detach().float().pow(2).sum()
    if not saw:
        return float("nan")
    return float(total.sqrt())


def _normalize(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)


def test_shared_trunk_parameters_exclude_only_the_heads():
    model = _model()
    selected = shared_trunk_parameters(model)
    names = [
        name
        for name, param in model.named_parameters()
        if any(param is chosen for chosen in selected)
    ]
    prefixes = {name.split(".")[0] for name in names}
    assert prefixes == {
        "token_emb",
        "type_emb",
        "pos_emb",
        "numeric_proj",
        "encoder",
        "trunk",
    }
    assert not any(name.startswith("policy_head") or name.startswith("value_head") for name in names)
    assert {id(param) for param in selected} == {id(param) for param in _trunk_excluding_heads(model)}

    from alphanl.transformer.conv_net import AlphaNLConvNet

    conv = AlphaNLConvNet()
    conv_names = [
        name
        for name, param in conv.named_parameters()
        if any(param is chosen for chosen in shared_trunk_parameters(conv))
    ]
    assert any(name.startswith("fc_1.") for name in conv_names)
    assert any(name.startswith("card.") for name in conv_names)
    assert not any(
        name.startswith("conv_fuse.") or name.startswith("value_out.") for name in conv_names
    )


def test_nonfinite_and_missing_grads_are_nan():
    assert math.isnan(_global_l2_norm([]))
    assert math.isnan(_global_l2_norm([None, None]))
    assert math.isnan(_global_l2_norm([torch.tensor([float("inf")])]))
    assert math.isnan(_global_l2_norm([torch.tensor([float("nan")], dtype=torch.float16)]))
    assert math.isnan(_global_l2_norm([torch.tensor([float("inf")], dtype=torch.float16)]))
    # None is a zero contribution. 3-4-5 is exact in fp16.
    assert _global_l2_norm([torch.tensor([3.0, 4.0], dtype=torch.float16), None]) == pytest.approx(5.0)

    model = _model()
    nan_loss = model.token_emb.weight.sum() * float("nan")
    got = trunk_grad_norms(nan_loss, nan_loss, model)
    assert set(got) == set(GRAD_NORM_COLUMNS)
    assert all(math.isnan(got[key]) for key in GRAD_NORM_COLUMNS)
    assert all(param.grad is None for param in model.parameters())


def test_value_coef_is_not_applied_and_entropy_term_is_inside_policy():
    """value_coef does not scale grad_norm_value. entropy_coef is inside policy_side."""

    torch.manual_seed(0)
    model = _model().eval()
    batch = _batch()
    actions, old_logp, old_values, advantages, returns = _targets(8)
    common = dict(
        actions=actions,
        old_logp=old_logp,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
    )

    def _measure(value_coef: float, entropy_coef: float):
        _loss, metrics, policy_side, value_side = ppo_losses(
            model,
            batch,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            **common,
        )
        norms = trunk_grad_norms(policy_side, value_side, model)
        assert all(param.grad is None for param in model.parameters())
        return metrics, norms

    metrics_lo, norms_lo = _measure(0.5, 0.01)
    metrics_hi, norms_hi = _measure(2.0, 0.01)
    assert abs(metrics_lo["loss"] - metrics_hi["loss"]) > 1e-6
    assert metrics_lo["value_loss"] == pytest.approx(metrics_hi["value_loss"], abs=1e-6)
    assert norms_lo["grad_norm_value"] == pytest.approx(norms_hi["grad_norm_value"], abs=1e-6)
    assert norms_lo["grad_norm_policy"] == pytest.approx(norms_hi["grad_norm_policy"], abs=1e-6)
    assert norms_lo["grad_norm_ratio"] == pytest.approx(
        norms_lo["grad_norm_value"] / norms_lo["grad_norm_policy"], abs=1e-6
    )

    _metrics_off, norms_off = _measure(0.5, 0.0)
    _metrics_on, norms_on = _measure(0.5, 1.0)
    assert abs(norms_off["grad_norm_policy"] - norms_on["grad_norm_policy"]) > 1e-4


def test_logged_norms_match_an_independent_backward():
    """ppo_update's columns match a separate backward on a cloned model."""

    torch.manual_seed(0)
    model = _model()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    batch = _batch()
    actions, old_logp, old_values, advantages, returns = _targets(8)
    rng = torch.get_rng_state()
    metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        batch,
        actions=actions,
        old_logp=old_logp,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=2,
        minibatch_size=4,
        max_grad_norm=0.5,
    )

    ref = _model()
    ref.load_state_dict(initial)
    ref.eval()
    torch.set_rng_state(rng)
    normalized = _normalize(advantages)
    index = torch.randperm(actions.shape[0])[:4]
    mb = slice_batch(batch, index)
    _loss, _metrics, policy_side, value_side = ppo_losses(
        ref,
        mb,
        actions=actions.index_select(0, index),
        old_logp=old_logp.index_select(0, index),
        old_values=old_values.index_select(0, index),
        advantages=normalized.index_select(0, index),
        returns=returns.index_select(0, index),
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
    )
    trunk = _trunk_excluding_heads(ref)
    ref.zero_grad(set_to_none=True)
    policy_side.backward(retain_graph=True)
    policy_norm = _l2_of_param_grads(trunk)
    full_policy = _l2_of_param_grads(list(ref.parameters()))
    ref.zero_grad(set_to_none=True)
    value_side.backward()
    value_norm = _l2_of_param_grads(trunk)

    assert math.isfinite(policy_norm) and math.isfinite(value_norm)
    assert abs(full_policy - policy_norm) > 1e-8
    assert metrics["grad_norm_policy"] == pytest.approx(policy_norm, abs=1e-6)
    assert metrics["grad_norm_value"] == pytest.approx(value_norm, abs=1e-6)
    assert metrics["grad_norm_ratio"] == pytest.approx(value_norm / policy_norm, abs=1e-6)
    assert metrics["grad_norm_ratio"] == pytest.approx(
        metrics["grad_norm_value"] / metrics["grad_norm_policy"], abs=1e-6
    )


def test_unused_trunk_parameter_is_a_zero_and_dot_grad_stays_empty():
    """``token_emb`` is in the trunk and unused. ``None`` counts as zero.

    The reference backward reads only the parameters that actually received a
    gradient. ``.grad`` on the probed model stays ``None``.
    """

    class _Partial(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.trunk = torch.nn.Linear(2, 2, bias=False)
            self.token_emb = torch.nn.Linear(2, 2, bias=False)
            self.policy_head = torch.nn.Linear(2, 1, bias=False)
            self.value_head = torch.nn.Linear(2, 1, bias=False)

        def forward(self, x):
            hidden = self.trunk(torch.relu(self.encoder(x)))
            return self.policy_head(hidden).squeeze(-1), self.value_head(hidden).squeeze(-1)

    torch.manual_seed(1)
    model = _Partial()
    x = torch.randn(4, 2)
    policy, value = model(x)
    model.zero_grad(set_to_none=True)
    norms = trunk_grad_norms(policy.pow(2).mean(), (value - 0.5).pow(2).mean(), model)
    assert all(param.grad is None for param in model.parameters())

    ref = _Partial()
    ref.load_state_dict({key: value.detach().clone() for key, value in model.state_dict().items()})
    policy_r, _value_r = ref(x)
    policy_r.pow(2).mean().backward()
    assert ref.token_emb.weight.grad is None
    expected = _l2_of_param_grads([ref.encoder.weight, ref.trunk.weight])
    assert norms["grad_norm_policy"] == pytest.approx(expected, abs=1e-6)
    assert math.isfinite(norms["grad_norm_ratio"])
    assert norms["grad_norm_ratio"] == pytest.approx(
        norms["grad_norm_value"] / norms["grad_norm_policy"], abs=1e-6
    )


def test_only_the_first_minibatch_of_the_first_epoch_is_probed(monkeypatch):
    import alphanl.transformer.ppo as ppo_mod

    calls = {"n": 0, "widths": []}
    real = torch.autograd.grad

    def _wrapped(outputs, inputs, *args, **kwargs):
        calls["n"] += 1
        calls["widths"].append(len(list(inputs)))
        assert kwargs.get("retain_graph") is True
        assert kwargs.get("allow_unused") is True
        return real(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _wrapped)
    torch.manual_seed(0)
    model = _model()
    n = 8
    batch = _batch(n)
    actions, old_logp, old_values, advantages, returns = _targets(n)
    ppo_update(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        batch,
        actions=actions,
        old_logp=old_logp,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=2,
        minibatch_size=4,
        max_grad_norm=0.5,
    )
    # Two losses, once. epochs=2 and two minibatches would be 8 calls if every
    # minibatch were measured.
    assert calls["n"] == 2
    trunk_width = len(shared_trunk_parameters(model))
    assert calls["widths"] == [trunk_width, trunk_width]
    assert trunk_width < sum(1 for _ in model.parameters())

    calls["n"] = 0
    monkeypatch.setattr(ppo_mod, "_log_grad_norms_enabled", lambda: False)
    torch.manual_seed(0)
    model = _model()
    metrics = ppo_update(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        batch,
        actions=actions,
        old_logp=old_logp,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=2,
        minibatch_size=4,
        max_grad_norm=0.5,
    )
    assert calls["n"] == 0
    assert all(math.isnan(metrics[key]) for key in GRAD_NORM_COLUMNS)


def test_probe_on_matches_probe_off_for_one_update():
    """Same seed, CPU, no AMP. Loss metrics and the final weights match."""

    import alphanl.transformer.ppo as ppo_mod

    torch.manual_seed(0)
    base = _model()
    initial = {key: value.detach().clone() for key, value in base.state_dict().items()}
    batch = _batch()
    actions, old_logp, old_values, advantages, returns = _targets(8)
    original = ppo_mod._log_grad_norms_enabled

    def _run(enabled: bool):
        ppo_mod._log_grad_norms_enabled = (lambda: True) if enabled else (lambda: False)
        model = _model()
        model.load_state_dict(initial)
        torch.manual_seed(1)
        metrics = ppo_update(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            batch,
            actions=actions,
            old_logp=old_logp,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=2,
            minibatch_size=4,
            max_grad_norm=0.5,
        )
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        return metrics, state

    try:
        on_metrics, on_state = _run(True)
        off_metrics, off_state = _run(False)
    finally:
        ppo_mod._log_grad_norms_enabled = original

    for key in _LOSS_COLUMNS:
        assert on_metrics[key] == off_metrics[key], key
    for key, value in on_state.items():
        assert torch.equal(value, off_state[key]), key
    assert all(math.isfinite(on_metrics[key]) for key in GRAD_NORM_COLUMNS)
    assert all(math.isnan(off_metrics[key]) for key in GRAD_NORM_COLUMNS)


def test_amp_branch_logs_unscaled_norms(monkeypatch):
    """AMP branch without multiplying the reported norms by the scaler scale.

    CUDA autocast cannot run on CPU. This test forces ``amp_enabled`` so
    ``ppo_update`` takes the GradScaler branch, and replaces ``torch.autocast``
    with a no-op. The fake scaler's scale is 32: the real backward is
    multiplied by 32 and then unscaled. The logged norms must match a
    non-AMP run of the same batch, so they were taken from the unscaled
    losses and were not multiplied by that scale. A separate test covers a
    real CUDA GradScaler when CUDA is present.
    """

    import alphanl.transformer.ppo as ppo_mod

    monkeypatch.setattr(ppo_mod, "amp_enabled", lambda amp, device: bool(amp))

    @contextmanager
    def _noop_autocast(*args, **kwargs):
        yield

    monkeypatch.setattr(torch, "autocast", _noop_autocast)

    class _FakeScaler:
        def __init__(self, module):
            self.module = module
            self.factor = 32.0
            self.backward_calls = 0
            self.unscale_calls = 0
            self.step_calls = 0
            self.update_calls = 0
            self.grad_was_none = []

        def scale(self, loss):
            scaler = self

            class _Scaled:
                def backward(self_inner):
                    scaler.grad_was_none.append(
                        all(param.grad is None for param in scaler.module.parameters())
                    )
                    (loss * scaler.factor).backward()
                    scaler.backward_calls += 1

            return _Scaled()

        def unscale_(self, optimizer):
            self.unscale_calls += 1
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.div_(self.factor)

        def step(self, optimizer):
            self.step_calls += 1
            optimizer.step()

        def update(self):
            self.update_calls += 1

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.factor

    torch.manual_seed(0)
    initial_model = _model()
    initial = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
    batch = _batch()
    actions, old_logp, old_values, advantages, returns = _targets(8)

    def _run(amp: bool, scaler):
        model = _model()
        model.load_state_dict(initial)
        if scaler is not None:
            scaler.module = model
        torch.manual_seed(2)
        return ppo_update(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            batch,
            actions=actions,
            old_logp=old_logp,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=2,
            minibatch_size=8,
            max_grad_norm=0.5,
            amp=amp,
            scaler=scaler,
        )

    plain = _run(False, None)
    fake = _FakeScaler(initial_model)
    amp_metrics = _run(True, fake)
    assert fake.backward_calls == 2
    assert fake.unscale_calls == 2 and fake.step_calls == 2 and fake.update_calls == 2
    assert fake.grad_was_none == [True, True]
    for key in GRAD_NORM_COLUMNS:
        assert key in amp_metrics
    for key in ("grad_norm_policy", "grad_norm_value"):
        assert plain[key] > 0.0
        assert amp_metrics[key] == pytest.approx(plain[key], abs=1e-6)
        assert amp_metrics[key] / plain[key] == pytest.approx(1.0, abs=1e-5)
    assert amp_metrics["grad_norm_ratio"] == pytest.approx(
        amp_metrics["grad_norm_value"] / amp_metrics["grad_norm_policy"], abs=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_real_cuda_amp_writes_grad_columns():
    """Real GradScaler + CUDA autocast. fp16 overflow is NaN, not an exception."""

    from alphanl.transformer.ppo import make_grad_scaler

    torch.manual_seed(0)
    model = _model().cuda()
    n = 8
    batch = {key: value.cuda() for key, value in _batch(n).items()}
    actions, old_logp, old_values, advantages, returns = _targets(n)
    metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        batch,
        actions=actions.cuda(),
        old_logp=old_logp.cuda(),
        old_values=old_values.cuda(),
        advantages=advantages.cuda(),
        returns=returns.cuda(),
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=1,
        minibatch_size=n,
        max_grad_norm=0.5,
        amp=True,
        scaler=make_grad_scaler(True, "cuda"),
    )
    for key in GRAD_NORM_COLUMNS:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_old_metrics_header_keeps_m13a_columns_and_appends_grad_norms(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    old_header = (
        "update,entropy,free_check_fold_ratio,legal_entropy,free_check_count,fold_facing_ratio"
    )
    path.write_text(old_header + "\n0,1.0,0.5,0.4,3,0.2\n")
    _append_csv(
        path,
        {
            "update": 1,
            "entropy": 1.1,
            "free_check_fold_ratio": 0.0,
            "legal_entropy": 0.2,
            "free_check_count": 4,
            "fold_facing_ratio": 0.25,
            "grad_norm_policy": 0.1,
            "grad_norm_value": 0.2,
            "grad_norm_ratio": 2.0,
        },
    )
    lines = path.read_text().splitlines()
    assert lines[0] == old_header
    assert lines[1] == "0,1.0,0.5,0.4,3,0.2"
    assert lines[2].endswith(",0.1,0.2,2.0")
    parsed = next(csv.DictReader([lines[0], lines[2]]))
    assert parsed["fold_facing_ratio"] == "0.25"
    assert "grad_norm_policy" not in parsed


def _tiny_cfg(tmp_path: Path, updates: int):
    from alphanl.transformer.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.0
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 0
    cfg.env.backend = "botzone"
    cfg.train.amp = False
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 4
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 4
    cfg.train.num_updates = updates
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 0
    cfg.train.log_every = 1
    cfg.pool.snapshot_every = 0
    cfg.pool.max_size = 2
    cfg.train.out_dir = str(tmp_path)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def _rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _equal_tree(left, right, path: str) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert torch.equal(left.cpu(), right.cpu()), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _equal_tree(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (item_l, item_r) in enumerate(zip(left, right)):
            _equal_tree(item_l, item_r, f"{path}[{index}]")
        return
    try:
        import numpy as np

        if isinstance(left, np.ndarray):
            assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
            return
    except ImportError:
        pass
    assert left == right, path


def test_logging_on_matches_training_with_the_probe_disabled(tmp_path: Path):
    """Enabled probe vs the switch off. Same seed, 3 updates, CPU, no AMP.

    Same approach as M11a's ``test_logging_matches_training_with_the_reduction_disabled``.
    The switch is ``ppo._log_grad_norms_enabled``. Loss columns, including
    ``policy_loss``, ``value_loss``, and ``approx_kl``, are compared as CSV
    strings. The final model ``state_dict`` is compared with ``torch.equal``.
    """

    import alphanl.transformer.botzone_table  # noqa: F401
    import alphanl.transformer.ppo as ppo_mod
    import alphanl.transformer.train as train_mod

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pass

    original = ppo_mod._log_grad_norms_enabled

    def _run(out: Path, enabled: bool):
        ppo_mod._log_grad_norms_enabled = (lambda: True) if enabled else (lambda: False)
        train_mod.run_training(_tiny_cfg(out, 3))
        rows, header = _rows(out / "metrics.csv")
        blob = _load_checkpoint(out / "ckpt" / "update_0000002.pt")
        return rows, header, blob

    try:
        rows_on, header_on, blob_on = _run(tmp_path / "on", True)
        rows_off, header_off, blob_off = _run(tmp_path / "off", False)
    finally:
        ppo_mod._log_grad_norms_enabled = original

    assert header_on == header_off
    assert header_on[-len(GRAD_NORM_COLUMNS) :] == list(GRAD_NORM_COLUMNS)
    decision_end = -len(GRAD_NORM_COLUMNS)
    decision_start = decision_end - len(DECISION_STAT_COLUMNS)
    assert header_on[decision_start:decision_end] == list(DECISION_STAT_COLUMNS)
    assert len(rows_on) == 3 and len(rows_off) == 3
    skip = {"seconds", *GRAD_NORM_COLUMNS}
    for update, (left, right) in enumerate(zip(rows_on, rows_off)):
        for key in _LOSS_COLUMNS:
            assert left[key] == right[key], f"update {update} {key}: {left[key]!r} != {right[key]!r}"
        for key in left:
            if key not in skip:
                assert left[key] == right[key], f"update {update} {key}"
        assert all(math.isfinite(float(left[key])) for key in GRAD_NORM_COLUMNS)
        assert all(math.isnan(float(right[key])) for key in GRAD_NORM_COLUMNS)
        assert float(left["grad_norm_ratio"]) == pytest.approx(
            float(left["grad_norm_value"]) / float(left["grad_norm_policy"]), abs=1e-6
        )
    _equal_tree(blob_on["model"], blob_off["model"], "model")
    _equal_tree(blob_on["optimizer"], blob_off["optimizer"], "optimizer")
    _equal_tree(blob_on["scaler"], blob_off["scaler"], "scaler")
    _equal_tree(blob_on["rng"], blob_off["rng"], "rng")

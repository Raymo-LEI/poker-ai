"""Regret matching, Transformer compatibility, and regret-head training."""

import pytest
import torch

from alphanl.cfr.model import RegretNetwork, regret_matching
from alphanl.transformer.model import PokerTransformer, count_parameters
from alphanl.transformer.tokenizer import BettingHistoryTokenizer


def _backbone(**kwargs):
    config = dict(
        d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, max_seq_len=16,
    )
    config.update(kwargs)
    return PokerTransformer(**config)


def _batch():
    return BettingHistoryTokenizer(max_seq_len=16).blank()


def test_regret_matching_masks_positive_and_nonpositive_cases():
    regrets = torch.tensor([
        [1.0, 3.0, 1000.0, -4.0, 5.0],
        [-4.0, -2.0, 1000.0, -3.0, -1.0],
    ])
    legal = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 0, 1, 0]])
    matched = regret_matching(regrets, legal)
    assert torch.allclose(matched[0], torch.tensor([0.25, 0.75, 0, 0, 0]))
    assert torch.allclose(matched[1], torch.tensor([1 / 3, 1 / 3, 0, 1 / 3, 0]))
    greedy = regret_matching(regrets, legal, fallback="argmax")
    assert torch.equal(greedy[1], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))


def test_regret_matching_single_legal_and_extreme_finite_values():
    half = regret_matching(torch.tensor([[-100.0, -1.0]], dtype=torch.float16),
                           torch.tensor([[0, 1]]))
    assert half.dtype == torch.float32
    assert torch.equal(half, torch.tensor([[0.0, 1.0]]))
    huge = regret_matching(torch.tensor([[3.0e38, 3.0e38, -1.0]]),
                           torch.ones(1, 3))
    assert torch.equal(huge, torch.tensor([[0.5, 0.5, 0.0]]))
    tiny = regret_matching(torch.tensor([[1.0e-44, 0.0]]), torch.ones(1, 2))
    assert torch.equal(tiny, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("regrets,legal,match", [
    (torch.zeros(1, 5), torch.zeros(1, 5), "at least one legal"),
    (torch.zeros(1, 5), torch.ones(5), "same non-scalar shape"),
    (torch.tensor([[float("nan"), 0.0]]), torch.ones(1, 2), "finite"),
])
def test_regret_matching_rejects_invalid_decisions(regrets, legal, match):
    with pytest.raises(ValueError, match=match):
        regret_matching(regrets, legal)


def test_regret_matching_has_finite_positive_branch_gradients():
    raw = torch.tensor([[1.0, 2.0, -1.0, 500.0, 0.0]], requires_grad=True)
    probs = regret_matching(raw, torch.tensor([[1, 1, 1, 0, 0]]))
    probs[0, 0].backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.allclose(raw.grad[0, :2], torch.tensor([2 / 9, -1 / 9]))
    assert torch.equal(raw.grad[0, 2:], torch.zeros(3))


def test_wrapper_preserves_backbone_and_original_ppo_checkpoint():
    torch.set_num_threads(1)
    backbone = _backbone().eval()
    batch = _batch()
    expected = backbone(batch)
    old_state = {k: v.clone() for k, v in backbone.state_dict().items()}
    model = RegretNetwork(backbone).eval()
    actual = model.backbone(batch)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert old_state.keys() == model.backbone.state_dict().keys()
    fresh = _backbone().eval()
    fresh.load_state_dict(model.backbone.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(expected, fresh(batch)))
    assert backbone.policy_head.weight.requires_grad
    assert backbone.value_head.weight.requires_grad
    assert not model.backbone.policy_head.weight.requires_grad
    assert not model.backbone.value_head.weight.requires_grad
    assert model.backbone.token_emb.weight.data_ptr() != backbone.token_emb.weight.data_ptr()
    assert all(torch.equal(old_state[k], v) for k, v in backbone.state_dict().items())


def test_laptop_architecture_keeps_592390_parameters():
    backbone = PokerTransformer()
    assert count_parameters(backbone) == 592390
    original_shapes = {k: v.shape for k, v in backbone.state_dict().items()}
    wrapper = RegretNetwork(backbone)
    assert {k: v.shape for k, v in wrapper.backbone.state_dict().items()} == original_shapes
    assert count_parameters(backbone) == 592390
    assert sum(p.numel() for p in wrapper.regret_head.parameters()) == 645


def test_zero_initialization_and_legacy_initialization():
    backbone = _backbone()
    wrapped = RegretNetwork.from_ppo_state_dict(
        backbone.state_dict(), d_model=16, nhead=4, num_layers=1,
        dim_feedforward=32, dropout=0.0, max_seq_len=16,
    ).eval()
    batch = _batch()
    batch["legal_mask"][:] = torch.tensor([[1, 1, 0, 1, 0]])
    assert torch.equal(wrapped(batch), torch.zeros(1, 5))
    assert torch.allclose(wrapped.strategy(batch), torch.tensor([[1 / 3, 1 / 3, 0, 1 / 3, 0]]))


def test_regret_training_reaches_encoder_without_changing_ppo_model():
    torch.manual_seed(3)
    original = _backbone().eval()
    weights = {k: v.clone() for k, v in original.state_dict().items()}
    model = RegretNetwork(original).eval()
    batch = _batch()
    target = torch.tensor([[1.0, -0.5, 0.0, 0.0, 0.0]])
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    initial = torch.nn.functional.mse_loss(model(batch), target).item()
    # The first step learns the zero head; subsequent steps reach the encoder.
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = model(batch)
            assert prediction.dtype == torch.float32
            loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        assert torch.isfinite(model.regret_head.weight.grad).all()
        optimizer.step()
    assert torch.nn.functional.mse_loss(model(batch), target).item() < initial
    assert model.backbone.token_emb.weight.grad.abs().sum() > 0
    assert model.backbone.policy_head.weight.grad is None
    assert model.backbone.value_head.weight.grad is None
    assert all(torch.equal(weights[k], v) for k, v in original.state_dict().items())


def test_strategy_preserves_mode_and_weight_roundtrip():
    model = RegretNetwork(_backbone(dropout=0.2)).eval()
    batch = _batch()
    with torch.no_grad():
        model.regret_head.bias[:] = torch.tensor([-1, 2, 3, 4, 5])
    probs = model.strategy(batch)
    assert not model.training
    restored = RegretNetwork(_backbone(dropout=0.2)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(probs, restored.strategy(batch))
    model.train()
    model.strategy(batch)
    assert model.training

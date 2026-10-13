"""Shapes, illegal-action masking, GAE, and a few PPO steps on random data."""

import torch

from alphanl.transformer.model import PokerTransformer, masked_log_probs
from alphanl.transformer.ppo import compute_gae, ppo_losses, ppo_update
from alphanl.transformer.tokenizer import NUM_NUMERIC, NUM_TYPES, VOCAB_SIZE


def _model() -> PokerTransformer:
    return PokerTransformer(
        vocab_size=VOCAB_SIZE,
        num_types=NUM_TYPES,
        num_numeric=NUM_NUMERIC,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=16,
    )


def _batch(batch_size: int = 8, legal=None) -> dict:
    seq = 16
    token_ids = torch.randint(1, 40, (batch_size, seq))
    type_ids = torch.randint(1, NUM_TYPES, (batch_size, seq))
    numeric = torch.randn(batch_size, seq, NUM_NUMERIC)
    pad_mask = torch.zeros(batch_size, seq, dtype=torch.bool)
    pad_mask[:, 10:] = True
    token_ids[:, 10:] = 0
    type_ids[:, 10:] = 0
    if legal is None:
        legal_mask = torch.ones(batch_size, 5)
    else:
        legal_mask = legal.expand(batch_size, -1).clone()
    return {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "numeric": numeric,
        "pad_mask": pad_mask,
        "legal_mask": legal_mask,
    }


def test_forward_shapes_and_illegal_mask():
    model = _model().eval()
    legal = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])
    batch = _batch(legal=legal)
    logits, value = model(batch)
    assert logits.shape == (8, 5)
    assert value.shape == (8,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    probs = torch.softmax(logits, dim=-1)
    assert float(probs[:, 1:4].max().detach()) < 1e-6
    assert torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5)
    for _ in range(30):
        actions, logp, _, entropy = model.act(batch, deterministic=False)
        assert set(actions.tolist()).issubset({0, 4})
        assert torch.isfinite(logp).all()
        assert torch.isfinite(entropy).all()
        assert float(entropy.min()) >= -1e-5


def test_padding_does_not_change_cls():
    model = _model().eval()
    batch = _batch(batch_size=2)
    logits, value = model(batch)
    mutated = {key: value.clone() for key, value in batch.items()}
    mutated["token_ids"][:, 12] = 17
    mutated["type_ids"][:, 12] = 3
    logits_pad, value_pad = model(mutated)
    assert torch.allclose(logits, logits_pad, atol=1e-5)
    assert torch.allclose(value, value_pad, atol=1e-5)
    mutated["pad_mask"][:, 12] = False
    logits_seen, _ = model(mutated)
    assert not torch.allclose(logits, logits_seen, atol=1e-4)


def test_gae_terminal_and_bootstrap():
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5]])
    dones = torch.tensor([[0.0, 1.0]])
    bootstrap = torch.tensor([999.0])
    adv, ret = compute_gae(rewards, values, dones, bootstrap, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(adv, torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(ret, torch.tensor([[1.0, 1.0]]))

    adv_b, ret_b = compute_gae(
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([5.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(adv_b, torch.tensor([[5.0]]))
    assert torch.allclose(ret_b, torch.tensor([[5.0]]))


def test_entropy_coef_changes_only_the_entropy_term():
    torch.manual_seed(0)
    model = _model().eval()
    batch = _batch(batch_size=16)
    with torch.no_grad():
        logits, values = model(batch)
        _, log_probs, _, _ = masked_log_probs(logits, batch["legal_mask"])
        actions = torch.distributions.Categorical(logits=logits).sample()
        old_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    advantages = torch.randn(16)
    returns = values.detach() + advantages
    loss0, metrics0 = ppo_losses(
        model,
        batch,
        actions=actions,
        old_logp=old_logp,
        old_values=values.detach(),
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
    )
    loss1, metrics1 = ppo_losses(
        model,
        batch,
        actions=actions,
        old_logp=old_logp,
        old_values=values.detach(),
        advantages=advantages,
        returns=returns,
        clip_eps=0.2,
        entropy_coef=1.0,
        value_coef=0.5,
    )
    assert abs(metrics0["entropy"] - metrics1["entropy"]) < 1e-5
    assert abs(float((loss1 - loss0).detach()) + metrics0["entropy"]) < 1e-4
    assert metrics0["policy_loss"] == metrics1["policy_loss"] or abs(
        metrics0["policy_loss"] - metrics1["policy_loss"]
    ) < 1e-5


def test_ppo_update_steps_are_finite_and_move_weights():
    torch.manual_seed(1)
    model = _model()
    batch = _batch(batch_size=32)
    model.eval()
    with torch.no_grad():
        logits, values = model(batch)
        _, log_probs, _, _ = masked_log_probs(logits, batch["legal_mask"])
        actions = torch.distributions.Categorical(logits=logits).sample()
        old_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).detach()
        old_values = values.detach()
    returns = old_values + torch.linspace(-1, 1, 32)
    advantages = returns - old_values
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = None
    for _ in range(3):
        metrics = ppo_update(
            model,
            optimizer,
            batch,
            actions=actions,
            old_logp=old_logp,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            clip_eps=0.2,
            entropy_coef=0.05,
            value_coef=0.5,
            epochs=2,
            minibatch_size=16,
            max_grad_norm=0.5,
        )
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics is not None
    assert metrics["entropy"] > 0
    moved = any(not torch.allclose(a, b) for a, b in zip(before, model.parameters()))
    assert moved

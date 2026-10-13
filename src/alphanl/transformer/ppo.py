"""Clipped PPO with a separate entropy coefficient.

AlphaNLHoldem's published training script uses IMPALA/V-trace. The project
README and the AlphaHoldem recipe keep PPO, a probabilistic policy, and an
entropy bonus so the 5-way policy does not collapse to always-fold. This is
that update. Trinal-clip is not implemented (see docs/step3_transformer.md).
"""

from __future__ import annotations

from typing import Dict

from typing import Optional

import torch

from .model import PokerTransformer, entropy_from_probs, masked_log_probs
from .tokenizer import TensorBatch, slice_batch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """GAE(γ, λ).

    ``rewards``, ``values``, and ``dones`` have shape ``[num_envs, time]``.
    ``dones`` is 1 when the episode ended on that step. ``bootstrap_value``
    is the value of the state after the last step; it is ignored on a done.
    Returns advantages and returns, both ``[num_envs, time]``.
    """

    num_envs, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_value = bootstrap_value
    next_advantage = torch.zeros(num_envs, device=rewards.device, dtype=rewards.dtype)
    for t in range(steps - 1, -1, -1):
        mask = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_value * mask - values[:, t]
        next_advantage = delta + gamma * gae_lambda * next_advantage * mask
        advantages[:, t] = next_advantage
        next_value = values[:, t]
    return advantages, advantages + values


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_ret = torch.var(returns)
    if float(var_ret) < 1e-8:
        return 0.0
    return float(1.0 - torch.var(returns - values) / var_ret)


def ppo_losses(
    model: PokerTransformer,
    batch: TensorBatch,
    *,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple:
    """One forward and the clipped PPO objective.

    ``advantages`` must already be normalized by the caller when that is
    desired. Entropy is subtracted, so raising ``entropy_coef`` pushes the
    masked policy toward uniform over the legal actions.
    """

    logits, values = model(batch, mask_illegal=True)
    _, log_probs, probs, legal = masked_log_probs(logits, batch["legal_mask"])
    new_logp = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    log_ratio = (new_logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    surr_unclipped = ratio * advantages
    surr_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr_unclipped, surr_clipped).mean()

    value_delta = (values - old_values).clamp(-clip_eps, clip_eps)
    value_clipped = old_values + value_delta
    value_unclipped_loss = (values - returns).pow(2)
    value_clipped_loss = (value_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_unclipped_loss, value_clipped_loss).mean()

    entropy = entropy_from_probs(probs, log_probs, legal).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    with torch.no_grad():
        approx_kl = (old_logp - new_logp).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
    metrics = {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "entropy_coef": float(entropy_coef),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "explained_variance": explained_variance(values.detach(), returns.detach()),
    }
    return loss, metrics


def amp_enabled(amp: bool, device) -> bool:
    """AMP only on CUDA; silently off on CPU."""
    return bool(amp) and torch.device(device).type == "cuda"


def make_grad_scaler(amp: bool, device) -> "torch.amp.GradScaler":
    """A GradScaler that is a no-op (enabled=False) when AMP is off. Its state is checkpointed."""
    return torch.amp.GradScaler("cuda", enabled=amp_enabled(amp, device))


def ppo_update(
    model: PokerTransformer,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
    *,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    amp: bool = False,
    scaler: "Optional[torch.amp.GradScaler]" = None,
) -> Dict[str, float]:
    """Several epochs of minibatch PPO. Storage stays on ``batch``'s device.

    Pass a CPU batch; each minibatch is what ``ppo_losses`` reads, so the
    caller should move the minibatch onto the model device before calling
    this if the batch is not already there. This function runs in-place on
    whatever device the tensors and the model share.
    """

    num_samples = int(actions.shape[0])
    if num_samples == 0:
        raise ValueError("empty PPO batch")
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    device = actions.device
    use_amp = amp_enabled(amp, device)
    if use_amp and scaler is None:
        scaler = make_grad_scaler(True, device)

    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": 0.0,
    }
    updates = 0
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, int(minibatch_size)):
            index = order[start : start + int(minibatch_size)]
            mb_batch = slice_batch(batch, index)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    loss, metrics = ppo_losses(
                        model,
                        mb_batch,
                        actions=actions.index_select(0, index),
                        old_logp=old_logp.index_select(0, index),
                        old_values=old_values.index_select(0, index),
                        advantages=advantages.index_select(0, index),
                        returns=returns.index_select(0, index),
                        clip_eps=clip_eps,
                        entropy_coef=entropy_coef,
                        value_coef=value_coef,
                    )
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, metrics = ppo_losses(
                    model,
                    mb_batch,
                    actions=actions.index_select(0, index),
                    old_logp=old_logp.index_select(0, index),
                    old_values=old_values.index_select(0, index),
                    advantages=advantages.index_select(0, index),
                    returns=returns.index_select(0, index),
                    clip_eps=clip_eps,
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            for key in totals:
                totals[key] += metrics[key]
            updates += 1
    if updates == 0:
        raise ValueError("minibatch_size produced no updates")
    averaged = {key: value / updates for key, value in totals.items()}
    averaged["entropy_coef"] = float(entropy_coef)
    return averaged

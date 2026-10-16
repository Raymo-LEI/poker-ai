"""Clipped PPO with a separate entropy coefficient.

AlphaNLHoldem's published training script uses IMPALA/V-trace. The project
README and the AlphaHoldem recipe keep PPO, a probabilistic policy, and an
entropy bonus so the 5-way policy does not collapse to always-fold. This is
that update. Trinal-clip is not implemented (see docs/step3_transformer.md).

Trunk gradient norms for metrics.csv are a logging probe only
(:func:`trunk_grad_norms`). They do not change this objective.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

from .decision_stats import policy_action_mask
from .model import entropy_from_probs, masked_log_probs, shared_trunk_parameters
from .tokenizer import TensorBatch, slice_batch

# Appended at the end of metrics.csv, after the M11a/M13a decision columns.
GRAD_NORM_COLUMNS = (
    "grad_norm_policy",
    "grad_norm_value",
    "grad_norm_ratio",
)


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
    model: torch.nn.Module,
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
    value_clip_eps: Optional[float] = None,
    mask_free_fold: bool = False,
    to_call: Optional[torch.Tensor] = None,
) -> tuple:
    """One forward and the clipped PPO objective.

    ``advantages`` must already be normalized by the caller when that is
    desired. Entropy is subtracted, so raising ``entropy_coef`` pushes the
    masked policy toward uniform over the legal actions.

    ``value_clip_eps`` is in reward units, not a ratio. ``None`` turns value
    clipping off and fits the value head with a plain squared error.

    ``mask_free_fold`` only replaces the legal mask before the existing
    forward. The clip, value loss, and entropy terms are unchanged. When
    the flag is false the mask tensor the forward sees is the one in
    ``batch``.

    Returns ``loss``, the float metrics dict, ``policy_side``, and
    ``value_loss``. ``policy_side`` is ``policy_loss - entropy_coef * entropy``:
    the policy-side term inside ``loss``, with the entropy term included as
    it is defined there (minus sign and ``entropy_coef``). ``value_loss`` is
    the unscaled value term (``value_coef`` is not applied). Neither tensor
    is multiplied by a GradScaler scale. The total ``loss`` expression is
    unchanged; the extra tensors are for the logging probe only.
    """

    if mask_free_fold:
        if to_call is None:
            raise ValueError("mask_free_fold requires to_call aligned with the batch")
        batch = dict(batch)
        batch["legal_mask"] = policy_action_mask(batch["legal_mask"], to_call, True)
    logits, values = model(batch, mask_illegal=True)
    _, log_probs, probs, legal = masked_log_probs(logits, batch["legal_mask"])
    new_logp = log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    log_ratio = (new_logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    surr_unclipped = ratio * advantages
    surr_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr_unclipped, surr_clipped).mean()

    value_unclipped_loss = (values - returns).pow(2)
    if value_clip_eps is None:
        value_loss = 0.5 * value_unclipped_loss.mean()
    else:
        value_delta = (values - old_values).clamp(-value_clip_eps, value_clip_eps)
        value_clipped_loss = (old_values + value_delta - returns).pow(2)
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
    # Built after the metrics floats so this line cannot change a logged loss.
    # policy_side is the policy-side term of ``loss`` (entropy included).
    # value_loss is unscaled: value_coef is applied only on ``loss`` above.
    policy_side = policy_loss - entropy_coef * entropy
    return loss, metrics, policy_side, value_loss


def amp_enabled(amp: bool, device) -> bool:
    """AMP only on CUDA; silently off on CPU."""
    return bool(amp) and torch.device(device).type == "cuda"


def make_grad_scaler(amp: bool, device) -> "torch.amp.GradScaler":
    """A GradScaler that is a no-op (enabled=False) when AMP is off. Its state is checkpointed."""
    return torch.amp.GradScaler("cuda", enabled=amp_enabled(amp, device))


def _log_grad_norms_enabled() -> bool:
    """Tests set this false to skip the trunk-gradient probe.

    The probe is ``torch.autograd.grad`` on the first minibatch of the first
    epoch only. It does not write ``.grad``. When this returns false the
    probe is not called and ``GRAD_NORM_COLUMNS`` are NaN. The PPO loss,
    backward, and optimizer step do not read this flag.
    """

    return True


def _nan_grad_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in GRAD_NORM_COLUMNS}


def _global_l2_norm(grads: Sequence[Optional[torch.Tensor]]) -> float:
    """Global L2. ``None`` is a zero gradient and does not add to the sum.

    An empty sequence or an all-``None`` sequence is "no gradient" and
    returns NaN. Any non-finite component (including fp16 inf/NaN) returns
    NaN. Squares are accumulated in fp32 so a finite fp16 gradient does not
    overflow the reduction itself.
    """

    if not grads or all(grad is None for grad in grads):
        return float("nan")
    total = None
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return float("nan")
        square = grad.detach().float().pow(2).sum()
        total = square if total is None else total + square
    if total is None:
        return float("nan")
    value = float(total.sqrt())
    if not math.isfinite(value):
        return float("nan")
    return value


def trunk_grad_norms(
    policy_side: torch.Tensor,
    value_side: torch.Tensor,
    model: torch.nn.Module,
) -> Dict[str, float]:
    """Global L2 norms of the unscaled per-term losses on the shared trunk.

    Called once per update, from the first minibatch of the first epoch,
    after that minibatch's forward and before the real backward. No second
    forward is run. ``retain_graph=True`` keeps the graph for the real
    backward. ``allow_unused=True``; a ``None`` gradient is a zero.

    Definition (also the metrics.csv columns):

    * ``grad_norm_policy`` is the global L2 of ``d(policy_side)/d(trunk)``.
      ``policy_side = policy_loss - entropy_coef * entropy``. That is the
      policy-side term inside
      ``loss = policy_loss + value_coef * value_loss - entropy_coef * entropy``:
      the clipped surrogate plus the entropy term with the minus sign and
      ``entropy_coef`` as that term is defined in the total loss.
      ``value_coef`` is not applied.
    * ``grad_norm_value`` is the global L2 of ``d(value_loss)/d(trunk)``.
      ``value_loss`` is the existing ``0.5 *`` squared error (clipped when
      ``value_clip_eps`` is set). ``value_coef`` is not applied.
    * ``grad_norm_ratio = grad_norm_value / grad_norm_policy``. NaN when
      either norm is non-finite, when there is no gradient, or when the
      policy norm is 0.

    Trunk parameters are :func:`shared_trunk_parameters` only (embeddings,
    transformer layers, and the shared post-encoder MLP; not the policy or
    value head).

    Both tensors are unscaled: this does not call ``GradScaler.scale`` and
    does not multiply by the scaler's scale, so the numbers stay in the
    unscaled magnitude under AMP. A non-finite loss or gradient (fp16
    overflow included) is NaN. Nothing is written to ``.grad``.
    """

    params = [param for param in shared_trunk_parameters(model) if param.requires_grad]
    if not params:
        return _nan_grad_metrics()
    if (not torch.isfinite(policy_side).all()) or (not torch.isfinite(value_side).all()):
        return _nan_grad_metrics()
    if not policy_side.requires_grad or not value_side.requires_grad:
        return _nan_grad_metrics()
    policy_grads = torch.autograd.grad(
        policy_side,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    value_grads = torch.autograd.grad(
        value_side,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    policy_norm = _global_l2_norm(policy_grads)
    value_norm = _global_l2_norm(value_grads)
    if (
        not math.isfinite(policy_norm)
        or not math.isfinite(value_norm)
        or policy_norm == 0.0
    ):
        ratio = float("nan")
    else:
        ratio = value_norm / policy_norm
        if not math.isfinite(ratio):
            ratio = float("nan")
    return {
        "grad_norm_policy": policy_norm,
        "grad_norm_value": value_norm,
        "grad_norm_ratio": ratio,
    }


def ppo_update(
    model: torch.nn.Module,
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
    value_clip_eps: Optional[float] = None,
    mask_free_fold: bool = False,
    to_call: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Several epochs of minibatch PPO. Storage stays on ``batch``'s device.

    Pass a CPU batch; each minibatch is what ``ppo_losses`` reads, so the
    caller should move the minibatch onto the model device before calling
    this if the batch is not already there. This function runs in-place on
    whatever device the tensors and the model share.

    The model is left in eval mode: dropout is off for every PPO forward.

    ``mask_free_fold`` and ``to_call`` are forwarded to :func:`ppo_losses`.
    When the flag is false, ``to_call`` is ignored and the mask in ``batch``
    is the one the forward uses.

    The returned dict also carries :data:`GRAD_NORM_COLUMNS`. Those three
    values are measured on the first minibatch of the first epoch only,
    from that forward, before the real backward. They are not averaged
    across minibatches. See :func:`trunk_grad_norms`.
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
    # NaN until (and unless) the first minibatch of the first epoch is probed.
    grad_metrics = _nan_grad_metrics()
    # Rollouts sample in eval mode. Dropout here would make the PPO ratio
    # differ from 1 before any step and push samples outside the clip range.
    model.eval()
    for epoch_index in range(int(epochs)):
        order = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, int(minibatch_size)):
            index = order[start : start + int(minibatch_size)]
            mb_batch = slice_batch(batch, index)
            mb_to_call = None if to_call is None else to_call.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            loss_kwargs = dict(
                actions=actions.index_select(0, index),
                old_logp=old_logp.index_select(0, index),
                old_values=old_values.index_select(0, index),
                advantages=advantages.index_select(0, index),
                returns=returns.index_select(0, index),
                clip_eps=clip_eps,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                value_clip_eps=value_clip_eps,
                mask_free_fold=mask_free_fold,
                to_call=mb_to_call,
            )
            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    loss, metrics, policy_side, value_side = ppo_losses(
                        model, mb_batch, **loss_kwargs
                    )
            else:
                loss, metrics, policy_side, value_side = ppo_losses(
                    model, mb_batch, **loss_kwargs
                )
            # Same forward as the step below. Unscaled losses, before
            # scaler.scale / loss.backward. Other minibatches are not measured.
            if epoch_index == 0 and start == 0 and _log_grad_norms_enabled():
                grad_metrics = trunk_grad_norms(policy_side, value_side, model)
            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
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
    averaged.update(grad_metrics)
    return averaged

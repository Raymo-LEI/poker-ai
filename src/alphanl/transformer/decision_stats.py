"""Free-check fold ratio, facing-bet fold ratio, and legal-action entropy.

The free-check definition is the one :mod:`eval.eval_botzone_collapse` uses.
``is_free_check`` is that helper. Entropy is the Shannon entropy, in nats,
of the masked legal-action distribution (:func:`entropy_from_probs`).
A decision with fewer than two legal actions is left out of every stat,
the same exclusion as the collapse eval's ``len(legal_ids) <= 1``.

``policy_action_mask`` is the optional policy-only mask. It does not change
environment legality. ``fold_facing_ratio`` is the sampled fold rate on
decisions that face a bet, fold is legal, and the decision is not forced.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from alphanl.botzone.actions import CHECK_CALL_ACTION, FOLD_ACTION

# Appended to metrics.csv in this order. Existing columns stay in front.
DECISION_STAT_COLUMNS = (
    "free_check_fold_ratio",
    "legal_entropy",
    "free_check_count",
    "fold_facing_ratio",
)


def is_free_check(to_call: float, legal_mask: torch.Tensor) -> bool:
    """Check is free: nothing is owed, and the mask the policy sees allows fold."""

    mask = legal_mask.detach().reshape(-1)
    fold_legal = float(mask[FOLD_ACTION]) > 0.5
    return float(to_call) == 0.0 and fold_legal


def policy_action_mask(
    legal_mask: torch.Tensor,
    to_call: Union[torch.Tensor, float],
    mask_free_fold: bool,
) -> torch.Tensor:
    """Mask the policy may sample. Environment legality is not this tensor.

    ``legal_mask`` is the environment mask (fold may stay legal on a free
    check). When ``mask_free_fold`` is false, that same tensor is returned
    and nothing is copied. When true, fold is cleared only where
    ``to_call == 0`` and check/call is available. Rows that face a bet, and
    rows where check/call is absent (all-in leaves fold and all-in), are
    unchanged. A non-finite ``to_call`` is not treated as zero.
    """

    if not mask_free_fold:
        return legal_mask
    owed = torch.as_tensor(to_call, device=legal_mask.device, dtype=torch.float32)
    if owed.ndim == legal_mask.ndim and owed.shape[-1] == 1:
        owed = owed.squeeze(-1)
    free = torch.isfinite(owed) & (owed == 0)
    check_open = legal_mask[..., CHECK_CALL_ACTION] > 0.5
    drop = free & check_open
    if drop.ndim == 0:
        if not bool(drop.item()):
            return legal_mask
    elif not bool(drop.any().item()):
        return legal_mask
    out = legal_mask.clone()
    fold = out[..., FOLD_ACTION]
    out[..., FOLD_ACTION] = torch.where(drop, torch.zeros_like(fold), fold)
    return out


def empty_decision_stats() -> dict:
    """No measurement. Callers log these as NaN rather than a guessed zero."""

    nan = float("nan")
    return {
        "free_check_fold_ratio": nan,
        "legal_entropy": nan,
        "free_check_count": nan,
        "fold_facing_ratio": nan,
    }


def summarize_rollout_decisions(
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    entropy: torch.Tensor,
    to_call: Optional[torch.Tensor],
) -> dict:
    """Ratio and mean legal entropy for hero decisions already sampled.

    ``actions`` is the action the rollout stored (the sampled choice).
    ``entropy`` is the masked-policy entropy already returned with that
    sample, one value per decision. ``to_call`` is chips owed at the same
    decisions, or ``None`` when the backend did not provide it. A missing
    ``to_call`` logs NaN for the fold ratio and its count. It does not
    guess from normalized observations.

    Decisions with fewer than two legal ids are excluded from the entropy
    mean, from the free-check denominator, and from ``fold_facing_ratio``.
    When a denominator is zero that ratio is NaN. A missing or non-finite
    ``to_call`` makes both ratios and the free-check count NaN.

    ``fold_facing_ratio`` is the fraction of those non-forced decisions
    with ``to_call > 0`` and fold legal whose sampled action is fold.
    """

    action_ids = actions.detach().reshape(-1).tolist()
    entropies = entropy.detach().reshape(-1).tolist()
    mask = legal_mask.detach().reshape(-1, int(legal_mask.shape[-1]))
    n = len(action_ids)
    if len(entropies) != n or int(mask.shape[0]) != n:
        raise ValueError(
            f"decision tensors disagree: actions={n}, entropy={len(entropies)}, mask={tuple(mask.shape)}"
        )
    owed: Optional[list] = None
    if to_call is not None:
        owed = [float(value) for value in to_call.detach().reshape(-1).tolist()]
        if len(owed) != n:
            raise ValueError(f"to_call length {len(owed)} does not match {n} decisions")

    kept_entropy = []
    free_folds = 0
    free_count = 0
    facing_folds = 0
    facing_count = 0
    to_call_unknown = owed is None
    for index, action in enumerate(action_ids):
        flags = mask[index].tolist()
        n_legal = sum(1 for flag in flags if float(flag) > 0.5)
        if n_legal <= 1:
            continue
        kept_entropy.append(float(entropies[index]))
        if to_call_unknown:
            continue
        value = owed[index]
        if not math.isfinite(value):
            to_call_unknown = True
            continue
        if is_free_check(value, mask[index]):
            free_count += 1
            if int(action) == FOLD_ACTION:
                free_folds += 1
        # Facing a bet: fold is still a real choice. Forced one-action
        # spots were already skipped. ``value > 0`` is the same chip count
        # ``is_free_check`` treats as zero when it is exactly 0.
        if value > 0.0 and float(flags[FOLD_ACTION]) > 0.5:
            facing_count += 1
            if int(action) == FOLD_ACTION:
                facing_folds += 1

    if kept_entropy:
        legal_entropy = sum(kept_entropy) / len(kept_entropy)
    else:
        legal_entropy = float("nan")
    if to_call_unknown:
        return {
            "free_check_fold_ratio": float("nan"),
            "legal_entropy": legal_entropy,
            "free_check_count": float("nan"),
            "fold_facing_ratio": float("nan"),
        }
    if free_count == 0:
        ratio = float("nan")
        count: float | int = 0
    else:
        ratio = free_folds / free_count
        count = free_count
    if facing_count == 0:
        facing_ratio = float("nan")
    else:
        facing_ratio = facing_folds / facing_count
    return {
        "free_check_fold_ratio": ratio,
        "legal_entropy": legal_entropy,
        "free_check_count": count,
        "fold_facing_ratio": facing_ratio,
    }

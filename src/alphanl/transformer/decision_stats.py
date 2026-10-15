"""Free-check fold ratio and legal-action entropy for one rollout.

The definitions are the ones :mod:`eval.eval_botzone_collapse` uses.
``is_free_check`` is that helper. Entropy is the Shannon entropy, in nats,
of the masked legal-action distribution (:func:`entropy_from_probs`).
A decision with fewer than two legal actions is left out of every stat,
the same exclusion as the collapse eval's ``len(legal_ids) <= 1``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from alphanl.botzone.actions import FOLD_ACTION

# Appended to metrics.csv in this order. Existing columns stay in front.
DECISION_STAT_COLUMNS = (
    "free_check_fold_ratio",
    "legal_entropy",
    "free_check_count",
)


def is_free_check(to_call: float, legal_mask: torch.Tensor) -> bool:
    """Check is free: nothing is owed, and the mask the policy sees allows fold."""

    mask = legal_mask.detach().reshape(-1)
    fold_legal = float(mask[FOLD_ACTION]) > 0.5
    return float(to_call) == 0.0 and fold_legal


def empty_decision_stats() -> dict:
    """No measurement. Callers log these as NaN rather than a guessed zero."""

    nan = float("nan")
    return {
        "free_check_fold_ratio": nan,
        "legal_entropy": nan,
        "free_check_count": nan,
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
    mean and from the free-check denominator. When that denominator is
    zero the ratio is NaN and the count is 0. A non-finite ``to_call`` on
    any counted decision makes the ratio and the count NaN.
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

    if kept_entropy:
        legal_entropy = sum(kept_entropy) / len(kept_entropy)
    else:
        legal_entropy = float("nan")
    if to_call_unknown:
        return {
            "free_check_fold_ratio": float("nan"),
            "legal_entropy": legal_entropy,
            "free_check_count": float("nan"),
        }
    if free_count == 0:
        ratio = float("nan")
        count: float | int = 0
    else:
        ratio = free_folds / free_count
        count = free_count
    return {
        "free_check_fold_ratio": ratio,
        "legal_entropy": legal_entropy,
        "free_check_count": count,
    }

"""KL, L1, and same-bucket comparison of two action distributions.

Natural-log KL is undefined when the reference gives zero mass to an action
the policy uses, and that case raises. No strength number is computed.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .schema import SchemaError, action_names


class CompareError(ValueError):
    """The two distributions cannot be compared on this schema."""


def _vector(values: Sequence[float], width: int) -> tuple[float, ...]:
    if len(values) != width:
        raise CompareError(f"expected {width} probabilities, got {len(values)}")
    probs = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise CompareError("probabilities must be finite and non-negative")
        probs.append(number)
    if abs(sum(probs) - 1.0) > 1e-6:
        raise CompareError("probabilities must sum to 1")
    return tuple(probs)


def _argmax(probs: Sequence[float]) -> int:
    best = 0
    for index, value in enumerate(probs):
        if value > probs[best]:
            best = index
    return best


def kl_divergence(policy: Sequence[float], reference: Sequence[float]) -> float:
    total = 0.0
    for left, right in zip(policy, reference):
        if left == 0.0:
            continue
        if right <= 0.0:
            raise CompareError("KL is undefined where the reference bucket is empty")
        total += left * math.log(left / right)
    return total


def l1_distance(policy: Sequence[float], reference: Sequence[float]) -> float:
    return float(sum(abs(left - right) for left, right in zip(policy, reference)))


def same_bucket(policy: Sequence[float], reference: Sequence[float], names: Sequence[str]) -> bool:
    return names[_argmax(policy)] == names[_argmax(reference)]


def compare_distributions(
    policy: Sequence[float],
    reference: Sequence[float],
    schema_id: str,
    *,
    legal_action_ids: Sequence[int] | None = None,
) -> dict:
    names = action_names(schema_id)
    left = _vector(policy, len(names))
    right = _vector(reference, len(names))
    if legal_action_ids is not None:
        legal = set(int(action) for action in legal_action_ids)
        for index, (p_mass, r_mass) in enumerate(zip(left, right)):
            if index not in legal and (p_mass > 1e-12 or r_mass > 1e-12):
                raise CompareError(f"mass on illegal action {names[index]}")
    policy_name = names[_argmax(left)]
    reference_name = names[_argmax(right)]
    return {
        "schema_id": schema_id,
        "num_actions": len(names),
        "kl": kl_divergence(left, right),
        "l1": l1_distance(left, right),
        "same_bucket": policy_name == reference_name,
        "policy_bucket": policy_name,
        "reference_bucket": reference_name,
    }


def compare_spot(spot: Mapping, prepared: Mapping) -> dict:
    schema_id = prepared["schema_id"]
    if schema_id != spot.get("schema_id"):
        raise SchemaError("prepared schema does not match the fixture")
    metrics = compare_distributions(
        spot["policy_probs"],
        spot["reference_probs"],
        schema_id,
        legal_action_ids=prepared["legal_action_ids"],
    )
    names = action_names(schema_id)
    action_id = names.index(metrics["policy_bucket"])
    from .adapter_hunl import to_wizard_action

    metrics.update(
        spot_id=spot["spot_id"],
        legal_action_ids=list(prepared["legal_action_ids"]),
        quarter_pot_legal=5 in prepared["legal_action_ids"],
        argmax_action=to_wizard_action(prepared["table"], schema_id, action_id),
        reference_label=spot.get("reference_label", "recorded_fixture_not_a_live_result"),
    )
    return metrics

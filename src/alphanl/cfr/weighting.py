"""Polynomial sample weighting and a deterministic schedule within each fit.

Polynomial loss weighting does not implement DCFR's cumulative-regret discount
recurrence. A scripted-opponent mixture likewise changes the self-play objective.
"""
import math

import torch


def polynomial_weights(iterations, *, power=1.0, reference=1.0):
    if not math.isfinite(power) or power <= 0 or not math.isfinite(reference) or reference <= 0:
        raise ValueError("weight power and reference must be positive and finite")
    if not torch.isfinite(iterations).all() or (iterations < 1).any():
        raise ValueError("sample iterations must be positive and finite")
    weights = (iterations.to(dtype=torch.float32) / reference).pow(power)
    if not torch.isfinite(weights).all():
        raise ValueError("polynomial iteration weights overflowed")
    return weights


def fit_learning_rate(config, step):
    """Restart warmup/cosine per CFR fit; checkpoints occur at fit barriers."""
    if not 0 <= step < config.train_steps:
        raise ValueError("fit step out of range")
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = config.train_steps - config.warmup_steps
    phase = (step - config.warmup_steps) / max(1, span - 1)
    ratio = config.min_learning_rate_ratio + (1 - config.min_learning_rate_ratio) * .5 * (1 + math.cos(math.pi * phase))
    return config.learning_rate * ratio

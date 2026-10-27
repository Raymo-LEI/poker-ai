"""Scale-independent regression probes, evaluated only at fit boundaries."""

from __future__ import annotations

import torch


def regression_probe(prediction, target, legal_mask, iterations):
    """Return device scalars; mask illegal actions and center each action separately.

    This is a retained-training-data probe, not a held-out validation score.
    R² uses an iteration-weighted, per-action constant-mean baseline.
    Undefined ratios/constant-target R² remain NaN rather than a guessed zero.
    """
    legal = legal_mask > .5
    weight = iterations.to(prediction.device, dtype=torch.float64)[:, None] * legal
    target = target.to(torch.float64)
    error = torch.where(legal, prediction.to(torch.float64) - target, 0.)
    safe_target = torch.where(legal, target, 0.)
    sse = (weight * error.square()).sum()
    mean = (weight * safe_target).sum(0) / weight.sum(0).clamp_min(1.)
    sst = (weight * torch.where(legal, target - mean, 0.).square()).sum()
    err_mean = (weight * error).sum(0) / weight.sum(0).clamp_min(1.)
    centered_sse = (weight * torch.where(legal, error - err_mean, 0.).square()).sum()
    nan = torch.full_like(sse, float("nan"))
    return {
        "weighted_mse": sse / prediction.shape[0],
        "normalized_mse": sse / iterations.to(sse).sum().clamp_min(1.),
        "unweighted_mse": error.square().sum() / prediction.shape[0],
        "r2": torch.where(sst > 0, 1 - sse / sst.clamp_min(1e-30), nan),
        "explained_variance": torch.where(sst > 0, 1 - centered_sse / sst.clamp_min(1e-30), nan),
    }


def probe_comparison(start, end):
    """Copy the complete diagnostic packet to CPU once, after the fit."""
    ratio = torch.where(start["weighted_mse"] > 0,
                        end["weighted_mse"] / start["weighted_mse"].clamp_min(1e-30),
                        torch.full_like(start["weighted_mse"], float("nan")))
    values = {"probe_start_loss": start["weighted_mse"], "probe_end_loss": end["weighted_mse"],
              "probe_loss_ratio": ratio, "probe_r2": end["r2"],
              "explained_variance": end["explained_variance"],
              "probe_normalized_mse": end["normalized_mse"],
              "probe_unweighted_mse": end["unweighted_mse"]}
    scalars = torch.stack(list(values.values())).detach().cpu().tolist()
    return dict(zip(values, scalars))

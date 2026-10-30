#!/usr/bin/env python3
"""Soften (or sharpen) the policy head of a PPO-format checkpoint by a temperature T.

The policy of ``PokerTransformer`` is ``logits = policy_head(features)`` (one plain
``nn.Linear``), then illegal actions are filled with -1e9 and a softmax is taken.
Dividing the head's ``weight`` AND ``bias`` by T therefore gives exactly ``logits / T``
before the mask. The value head and every other tensor are copied bit for bit. The
argmax over legal actions does not change for T > 0.

Two modes::

    # direct
    PYTHONPATH=src python scripts/soften_policy_head.py \\
        --ckpt runs/bc_rulebot_train/bc_v2.pt --temperature 3.0 \\
        --out runs/bc_rulebot_train/bc_v2_soft.pt

    # find T by bisection so the mean legal-action entropy on validation rows hits a target (nats)
    PYTHONPATH=src python scripts/soften_policy_head.py \\
        --ckpt runs/bc_rulebot_train/bc_v2.pt --val-data runs/bc_rulebot/merged_600k \\
        --target-entropy 0.3 --out runs/bc_rulebot_train/bc_v2_soft.pt

Validation rows are the held-out games exactly as ``scripts/train_bc_rulebot.py`` picks them
(``split_game_mask`` with the data's metadata seed and VAL_FRACTION). ``--val-all`` uses every
row of ``--val-data`` instead. The output is loaded with
``train_transformer.py --init-weights <out> --allow-cross-backend-init`` (not ``--resume``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    env = os.environ.get("SOFTEN_REPO_ROOT")
    candidates = ([Path(env)] if env else []) + [here.parent, Path.cwd(), here / "repo", here]
    for cand in candidates:
        if (cand / "src" / "alphanl").is_dir() and (cand / "scripts" / "train_bc_rulebot.py").is_file():
            return cand.resolve()
    raise SystemExit(
        "cannot find the repo root (needs src/alphanl and scripts/train_bc_rulebot.py). "
        "Run from the repo root or set SOFTEN_REPO_ROOT."
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_bc_rulebot as tbr  # noqa: E402  (split_game_mask, assert_train_out, _gen, tensors)
from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.train import build_model  # noqa: E402

POLICY_WEIGHT = "policy_head.weight"
POLICY_BIAS = "policy_head.bias"
ACTION_NAMES = ("fold", "check_call", "raise_min", "raise_large", "all_in")
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- math

def masked_entropy_rows(logits: torch.Tensor, legal: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-row Shannon entropy (nats) of softmax(logits / T) restricted to ``legal``.

    ``logits`` are the raw (unmasked) head outputs. One legal action -> 0. Rows with no legal
    action are treated as all legal, like ``masked_log_probs``.
    """

    z = logits.double() / float(temperature)
    legal_b = legal > 0.5 if legal.dtype != torch.bool else legal
    legal_b = legal_b | ~legal_b.any(dim=-1, keepdim=True)
    z = z.masked_fill(~legal_b, float("-inf"))
    logp = torch.log_softmax(z, dim=-1)
    p = logp.exp()
    ent = -(p * torch.where(legal_b, logp, torch.zeros_like(logp))).sum(dim=-1)
    return ent.clamp_min(0.0)


def mean_entropy(logits: torch.Tensor, legal: torch.Tensor, temperature: float) -> float:
    return float(masked_entropy_rows(logits, legal, temperature).mean())


def max_reachable_entropy(legal: torch.Tensor) -> float:
    """Mean entropy as T -> infinity (uniform over legal actions)."""

    legal_b = legal > 0.5
    n = legal_b.sum(dim=-1).clamp_min(1).double()
    return float(n.log().mean())


def find_temperature(
    logits: torch.Tensor,
    legal: torch.Tensor,
    target: float,
    *,
    allow_sharpen: bool = False,
    t_max: float = 1000.0,
    t_min: float = 1e-3,
    tol: float = 1e-5,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """Log-space bisection for T with ``mean_entropy(logits, legal, T) == target``.

    Mean entropy is non-decreasing in T. Returns ``(T, entropy_at_T)``. Raises ``ValueError``
    if the target is out of reach, or if it is below the current entropy and
    ``allow_sharpen`` is false.
    """

    if not math.isfinite(target) or target < 0:
        raise ValueError(f"target entropy must be finite and >= 0, got {target}")
    h1 = mean_entropy(logits, legal, 1.0)
    if abs(h1 - target) <= tol:
        return 1.0, h1
    if target > h1:
        lo, hi = 1.0, float(t_max)
        h_hi = mean_entropy(logits, legal, hi)
        if h_hi < target:
            raise ValueError(
                f"target {target:.4f} is not reachable for T <= {t_max:g}: entropy there is {h_hi:.4f} "
                f"(uniform-over-legal limit {max_reachable_entropy(legal):.4f})"
            )
    else:
        if not allow_sharpen:
            raise ValueError(
                f"current mean entropy {h1:.4f} is already <= target {target:.4f}; softening would need T < 1 "
                "(sharpening). Pass --allow-sharpen to allow that."
            )
        lo, hi = float(t_min), 1.0
        h_lo = mean_entropy(logits, legal, lo)
        if h_lo > target:
            raise ValueError(
                f"target {target:.4f} is not reachable for T >= {t_min:g}: entropy there is {h_lo:.4f}"
            )
    for _ in range(max_iter):
        mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
        h = mean_entropy(logits, legal, mid)
        if abs(h - target) <= tol:
            return mid, h
        if h < target:
            lo = mid
        else:
            hi = mid
    mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
    return mid, mean_entropy(logits, legal, mid)


def check_temperature(temperature: float) -> float:
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")
    return t


def soften_state_dict(state: Dict[str, torch.Tensor], temperature: float) -> Dict[str, torch.Tensor]:
    """Copy of ``state`` with policy_head.weight and .bias divided by T. Nothing else changes."""

    t = check_temperature(temperature)
    for key in (POLICY_WEIGHT, POLICY_BIAS):
        if key not in state:
            raise KeyError(f"state_dict has no {key!r}; is this a PokerTransformer checkpoint?")
    out = dict(state)
    out[POLICY_WEIGHT] = state[POLICY_WEIGHT].detach().clone() / t
    out[POLICY_BIAS] = state[POLICY_BIAS].detach().clone() / t
    for key in (POLICY_WEIGHT, POLICY_BIAS):
        if not torch.isfinite(out[key]).all():
            raise ValueError(f"non-finite values in {key} after dividing by T={t}")
    return out


# ----------------------------------------------------------------------------- data / model

def load_blob(path: Path) -> dict:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "model" not in blob or "config" not in blob:
        raise SystemExit(f"{path} is not a PPO-format checkpoint (needs 'model' and 'config')")
    return blob


def build_from_blob(blob: dict, state: Optional[Dict[str, torch.Tensor]] = None) -> torch.nn.Module:
    cfg = ExperimentConfig.from_dict(blob["config"])
    if cfg.encoder != "transformer":
        raise SystemExit(f"encoder is {cfg.encoder!r}; only the 'transformer' policy head is supported")
    model = build_model(cfg)
    model.load_state_dict(blob["model"] if state is None else state)
    model.eval()
    return model


def load_val_batch(val_dir: Path, val_all: bool) -> Tuple[Dict[str, torch.Tensor], dict, List[int]]:
    gen = tbr._gen()
    try:
        arrays, meta = gen.load_dataset(val_dir, check_legal=True)
    except gen.LegalPyMismatch as exc:
        raise SystemExit(str(exc)) from exc
    n = int(np.asarray(arrays["actions"]).shape[0])
    if val_all:
        mask = np.ones(n, dtype=bool)
        val_games = sorted({int(g) for g in np.asarray(arrays["game_id"]).tolist()})
    else:
        mask, val_games = tbr.split_game_mask(arrays["game_id"], int(meta["seed"]), tbr.VAL_FRACTION)
    sub = {k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v) for k, v in arrays.items()}
    tensors = tbr._observation_tensors(sub)
    batch = {k: tensors[k] for k in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")}
    return batch, meta, val_games


@torch.no_grad()
def forward_all(model: torch.nn.Module, batch: Dict[str, torch.Tensor], chunk: int = 4096):
    """Same call as bc_losses: model(batch, mask_illegal=True), eval, no_grad. Returns logits, value."""

    n = int(batch["legal_mask"].shape[0])
    logits, values = [], []
    for s in range(0, n, chunk):
        mb = {k: v[s : s + chunk] for k, v in batch.items()}
        lg, val = model(mb, mask_illegal=True)
        logits.append(lg.float())
        values.append(val.float())
    return torch.cat(logits), torch.cat(values)


def action_stats(masked_logits: torch.Tensor, legal: torch.Tensor, temperature: float = 1.0):
    """(mean prob per action, argmax frequency per action) on already-masked logits."""

    lg = masked_logits.double()
    legal_b = legal > 0.5
    legal_b = legal_b | ~legal_b.any(dim=-1, keepdim=True)
    lg = torch.where(legal_b, lg / float(temperature), torch.full_like(lg, float("-inf")))
    probs = torch.softmax(lg, dim=-1)
    am = torch.argmax(masked_logits, dim=-1)
    freq = torch.bincount(am, minlength=probs.shape[1]).double() / max(1, am.numel())
    return probs.mean(dim=0), freq


def _fmt(vec: torch.Tensor) -> str:
    return " ".join(f"{name}={float(v):.4f}" for name, v in zip(ACTION_NAMES, vec.tolist()))


# ----------------------------------------------------------------------------- output

def assert_out_path(out: Path, force: bool) -> Path:
    out = tbr.assert_train_out(out)  # only runs/bc_rulebot_train/ allowed under runs/
    if out.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing {out} (pass --force)")
    return out


def write_atomic(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(blob, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def verify_written(src_blob: dict, out_path: Path, temperature: float) -> None:
    new = load_blob(out_path)
    a, b = src_blob["model"], new["model"]
    if set(a) != set(b):
        raise RuntimeError("written checkpoint has a different key set")
    for key in a:
        if key in (POLICY_WEIGHT, POLICY_BIAS):
            if not torch.equal(b[key], a[key].detach().clone() / float(temperature)):
                raise RuntimeError(f"{key} is not source / T")
        elif not torch.equal(a[key], b[key]):
            raise RuntimeError(f"{key} changed but should be bit-identical")
    for key in set(src_blob) | set(new):
        if key in ("model", "soften_metadata"):
            continue
        if key not in src_blob or key not in new:
            raise RuntimeError(f"top-level key {key!r} added or lost")


def soften_checkpoint(
    ckpt: Path,
    out: Path,
    *,
    temperature: Optional[float] = None,
    target_entropy: Optional[float] = None,
    val_data: Optional[Path] = None,
    val_all: bool = False,
    allow_sharpen: bool = False,
    t_max: float = 1000.0,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    if (temperature is None) == (target_entropy is None):
        raise SystemExit("pass exactly one of --temperature and --target-entropy")
    if temperature is not None:
        try:
            temperature = check_temperature(temperature)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if target_entropy is not None:
        if val_data is None:
            raise SystemExit("--target-entropy needs --val-data")
        if not math.isfinite(target_entropy) or target_entropy < 0:
            raise SystemExit(f"--target-entropy must be finite and >= 0, got {target_entropy}")
    out = assert_out_path(Path(out), force)
    ckpt = Path(ckpt)
    if out.resolve() == ckpt.resolve():
        raise SystemExit("--out must differ from --ckpt")
    blob = load_blob(ckpt)
    state = blob["model"]
    for key in (POLICY_WEIGHT, POLICY_BIAS):
        if key not in state:
            raise SystemExit(f"{ckpt} has no {key!r}")
    for key, value in state.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise SystemExit(f"non-finite values in checkpoint tensor {key}")
    model = build_from_blob(blob)

    batch = meta = None
    val_games: List[int] = []
    if val_data is not None:
        batch, meta, val_games = load_val_batch(Path(val_data), val_all)
        bcm = blob.get("bc_metadata") or {}
        if not val_all and bcm.get("val_game_ids") is not None and sorted(bcm["val_game_ids"]) != val_games:
            print(
                "WARNING: validation games differ from the BC run's val_game_ids "
                f"({len(val_games)} here vs {len(bcm['val_game_ids'])} recorded). Some rows may be BC training "
                "games, so the entropy is measured partly on training data.",
                flush=True,
            )
    stats: dict = {}
    legal = None
    if batch is not None:
        legal = batch["legal_mask"]
        masked0, value0 = forward_all(model, batch)
        # Legal entries of the masked output are the raw head outputs; illegal ones are -1e9.
        h_rows0 = masked_entropy_rows(masked0, legal, 1.0)
        multi = (legal > 0.5).sum(dim=-1) >= 2
        stats["n_rows"] = int(legal.shape[0])
        stats["n_rows_multi"] = int(multi.sum())
        stats["entropy_before"] = float(h_rows0.mean())
        stats["entropy_before_multi"] = float(h_rows0[multi].mean()) if bool(multi.any()) else 0.0
        if target_entropy is not None:
            try:
                temperature, _ = find_temperature(
                    masked0, legal, float(target_entropy), allow_sharpen=allow_sharpen, t_max=float(t_max)
                )
            except ValueError as exc:
                raise SystemExit(f"cannot reach target entropy: {exc}") from exc
    assert temperature is not None
    new_state = soften_state_dict(state, temperature)

    if batch is not None:
        new_model = build_from_blob(blob, state=new_state)
        masked1, value1 = forward_all(new_model, batch)
        h_rows1 = masked_entropy_rows(masked1, legal, 1.0)
        multi = (legal > 0.5).sum(dim=-1) >= 2
        stats["entropy_after"] = float(h_rows1.mean())
        stats["entropy_after_multi"] = float(h_rows1[multi].mean()) if bool(multi.any()) else 0.0
        stats["argmax_changed_rows"] = int((masked0.argmax(-1) != masked1.argmax(-1)).sum())
        stats["value_max_abs_diff"] = float((value0 - value1).abs().max())
        p0, f0 = action_stats(masked0, legal, 1.0)
        p1, f1 = action_stats(masked1, legal, 1.0)
        stats.update(mean_prob_before=p0.tolist(), mean_prob_after=p1.tolist(),
                     argmax_freq_before=f0.tolist(), argmax_freq_after=f1.tolist())

    metadata = {
        "temperature": float(temperature),
        "target_entropy": None if target_entropy is None else float(target_entropy),
        "val_data": None if val_data is None else str(val_data),
        "val_all": bool(val_all),
        "val_games": len(val_games),
        "source_ckpt": str(ckpt),
        "entropy_before": stats.get("entropy_before"),
        "entropy_after": stats.get("entropy_after"),
        "tool": "soften_policy_head.py",
    }
    new_blob = dict(blob)
    new_blob["model"] = new_state
    new_blob["soften_metadata"] = metadata
    write_atomic(out, new_blob)
    verify_written(blob, out, temperature)

    if verbose:
        print(f"source ckpt: {ckpt}  (update={blob.get('update')}, bc_metadata={'yes' if 'bc_metadata' in blob else 'no'})")
        if blob.get("optimizer") is not None:
            print("note: optimizer state kept as-is (unused by --init-weights). Do NOT use --resume on the softened file.")
        if batch is not None:
            print(f"validation rows: {stats['n_rows']} ({stats['n_rows_multi']} with >=2 legal actions), games: {len(val_games)}")
            hb, ha = stats["entropy_before"], stats["entropy_after"]
            print(f"mean entropy (all rows)   before {hb:.4f} nats / {hb / LN2:.4f} bits   after {ha:.4f} nats / {ha / LN2:.4f} bits")
            hb, ha = stats["entropy_before_multi"], stats["entropy_after_multi"]
            print(f"mean entropy (>=2 legal)  before {hb:.4f} nats / {hb / LN2:.4f} bits   after {ha:.4f} nats / {ha / LN2:.4f} bits")
            print("mean prob   before " + _fmt(torch.tensor(stats["mean_prob_before"])))
            print("mean prob   after  " + _fmt(torch.tensor(stats["mean_prob_after"])))
            print("argmax freq before " + _fmt(torch.tensor(stats["argmax_freq_before"])))
            print("argmax freq after  " + _fmt(torch.tensor(stats["argmax_freq_after"])))
            if stats["argmax_changed_rows"] == 0:
                print(f"argmax unchanged on all {stats['n_rows']} validation rows")
            else:
                print(f"WARNING: ARGMAX CHANGED on {stats['argmax_changed_rows']} of {stats['n_rows']} rows (near-ties)")
            print(f"value head max abs diff on validation rows: {stats['value_max_abs_diff']:.3g}")
            if target_entropy is not None and abs(stats["entropy_after"] - target_entropy) > 1e-3:
                print(f"WARNING: entropy after ({stats['entropy_after']:.4f}) differs from target {target_entropy:.4f} by > 1e-3")
        else:
            print("no --val-data: no entropy or argmax statistics")
        print(f"chosen T = {temperature:.6g}  (policy_head.weight and .bias divided by T)")
        print(f"wrote {out}")
    return {"temperature": float(temperature), "stats": stats, "metadata": metadata, "out": str(out)}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="Input BC / PPO-format checkpoint (.pt)")
    ap.add_argument("--out", type=Path, required=True, help="Output checkpoint (under runs/ only runs/bc_rulebot_train/)")
    ap.add_argument("--temperature", type=float, default=None, help="Divide the policy head by this T (> 0)")
    ap.add_argument("--target-entropy", type=float, default=None, help="Find T so mean legal-action entropy (nats) hits this")
    ap.add_argument("--val-data", type=Path, default=None, help="Dir with decisions.npz + metadata.json (gen_rulebot_bc.py)")
    ap.add_argument("--val-all", action="store_true", help="Use every row of --val-data instead of the held-out game split")
    ap.add_argument("--allow-sharpen", action="store_true", help="Allow T < 1 when the target is below the current entropy")
    ap.add_argument("--max-temperature", type=float, default=1000.0, help="Upper end of the bisection range (default 1000)")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing --out")
    args = ap.parse_args(argv)
    soften_checkpoint(
        args.ckpt,
        args.out,
        temperature=args.temperature,
        target_entropy=args.target_entropy,
        val_data=args.val_data,
        val_all=args.val_all,
        allow_sharpen=args.allow_sharpen,
        t_max=args.max_temperature,
        force=args.force,
    )


if __name__ == "__main__":
    main()

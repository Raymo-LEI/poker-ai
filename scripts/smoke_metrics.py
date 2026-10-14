#!/usr/bin/env python3
"""AMP / memory smoke for the step-3 Transformer (T-A).

Runs the normal training loop (``alphanl.transformer.train.run_training``)
in-process for a few hundred PPO updates and records, per update:

  * rollout seconds (CPU self-play) and update seconds (PPO step on GPU)
  * ``torch.cuda.max_memory_allocated`` (GB, cumulative peak)
  * GradScaler current scale (AMP only)
  * loss / policy_loss / value_loss / entropy (from the existing metrics row)

Training code is not modified: this script only wraps ``collect_rollout``,
``ppo_update`` and ``make_grad_scaler`` inside the train module and reads the
rows it already writes to metrics.csv. Every ``--log-every`` updates it
prints a window summary. At the end it writes ``smoke_summary.csv``,
``smoke_summary.json`` and ``smoke_summary.md`` into ``--out-dir`` and exits
non-zero on FAIL (NaN/inf loss, CUDA OOM, or peak memory >= --max-peak-gb).

    PYTHONPATH=src python scripts/smoke_metrics.py \
        --config configs/transformer/laptop_8gb.json --steps 300 --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

import importlib  # noqa: E402

T = importlib.import_module("alphanl.transformer.train")
from alphanl.transformer.config import ExperimentConfig, resolve_device  # noqa: E402

GB = 1024 ** 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/transformer/laptop_8gb.json")
    p.add_argument("--steps", type=int, default=300, help="PPO updates to run")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="runs/smoke_3070")
    p.add_argument("--device", type=str, default=None, help="auto/cpu/cuda (default: config)")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-envs", type=int, default=None, help="override ppo.num_envs (smaller batch)")
    p.add_argument("--rollout-steps", type=int, default=None, help="override ppo.rollout_steps")
    p.add_argument("--minibatch", type=int, default=None, help="override ppo.minibatch_size")
    p.add_argument("--with-eval", action="store_true", help="keep the config's periodic eval vs random")
    p.add_argument("--max-peak-gb", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    out = Path(a.out_dir)
    if (out / "metrics.csv").exists():
        (out / "metrics.csv").unlink()  # fresh smoke each run; never touches real runs
    cfg = ExperimentConfig.from_json(a.config)
    cfg.apply_overrides(
        num_updates=a.steps, out_dir=str(out), device=a.device, seed=a.seed, amp=a.amp,
        checkpoint_dir=str(out / "ckpt"), checkpoint_every=a.steps,
    )
    if a.num_envs:
        cfg.ppo.num_envs = a.num_envs
    if a.rollout_steps:
        cfg.ppo.rollout_steps = a.rollout_steps
    if a.minibatch:
        cfg.ppo.minibatch_size = a.minibatch
    if not a.with_eval:
        cfg.train.eval_every = 0
    cfg.train.log_every = 1
    dev = torch.device(resolve_device(cfg.device))
    cuda = dev.type == "cuda"
    print(T.describe_run(cfg))
    print(f"[smoke] device={dev} amp_requested={a.amp} amp_active={T.amp_enabled(cfg.train.amp, dev)} "
          f"steps={a.steps} out={out}", flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    cur = {}
    scalers = []
    rows = []
    orig_rollout, orig_update, orig_scaler, orig_csv = T.collect_rollout, T.ppo_update, T.make_grad_scaler, T._append_csv

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    def rollout(*args, **kw):
        _sync(); t = time.perf_counter()
        r = orig_rollout(*args, **kw)
        _sync(); cur["rollout_s"] = time.perf_counter() - t
        return r

    def update(*args, **kw):
        _sync(); t = time.perf_counter()
        r = orig_update(*args, **kw)
        _sync(); cur["update_s"] = time.perf_counter() - t
        return r

    def scaler(*args, **kw):
        s = orig_scaler(*args, **kw)
        scalers.append(s)
        return s

    def append_csv(path, row):
        orig_csv(path, row)
        s = scalers[0] if scalers else None
        rec = {
            "update": int(row["update"]),
            "loss": float(row.get("loss", float("nan"))),
            "policy_loss": float(row.get("policy_loss", float("nan"))),
            "value_loss": float(row.get("value_loss", float("nan"))),
            "entropy": float(row.get("entropy", float("nan"))),
            "seconds": float(row["seconds"]),
            "rollout_s": cur.get("rollout_s", float("nan")),
            "update_s": cur.get("update_s", float("nan")),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / GB if cuda else 0.0,
            "grad_scale": float(s.get_scale()) if (s is not None and s.is_enabled()) else float("nan"),
        }
        rows.append(rec)
        n = len(rows)
        if n % a.log_every == 0 or n == a.steps:
            _report(rows[-(((n - 1) % a.log_every) + 1):], n)

    T.collect_rollout, T.ppo_update, T.make_grad_scaler, T._append_csv = rollout, update, scaler, append_csv
    error = ""
    started = time.perf_counter()
    try:
        T.run_training(cfg)
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        error = f"CUDA OOM: {exc}"
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        error = f"CUDA OOM: {exc}"
    finally:
        T.collect_rollout, T.ppo_update, T.make_grad_scaler, T._append_csv = orig_rollout, orig_update, orig_scaler, orig_csv
    wall = time.perf_counter() - started
    return _finish(rows, a, out, cuda, error, wall, dev, cfg)


def _window(chunk, end):
    def m(k):
        v = [r[k] for r in chunk if not math.isnan(r[k])]
        return sum(v) / len(v) if v else float("nan")
    sec = sum(r["seconds"] for r in chunk)
    return {
        "steps": f"{chunk[0]['update'] + 1}-{end}",
        "steps_per_sec": len(chunk) / sec if sec > 0 else float("nan"),
        "rollout_s": m("rollout_s"),
        "update_s": m("update_s"),
        "loss": m("loss"),
        "peak_mem_gb": max(r["peak_mem_gb"] for r in chunk),
        "grad_scale": chunk[-1]["grad_scale"],
    }


def _report(chunk, n):
    w = _window(chunk, chunk[-1]["update"] + 1)
    print(f"[smoke] steps {w['steps']}: {w['steps_per_sec']:.3f} steps/s, rollout {w['rollout_s']:.2f}s, "
          f"update {w['update_s']:.2f}s, loss {w['loss']:.4f}, peak {w['peak_mem_gb']:.3f} GB, "
          f"scale {w['grad_scale']:g}", flush=True)


def _finish(rows, a, out, cuda, error, wall, dev, cfg) -> int:
    out.mkdir(parents=True, exist_ok=True)
    windows = [_window(rows[i:i + a.log_every], min(i + a.log_every, len(rows))) for i in range(0, len(rows), a.log_every)]
    nan_steps = [r["update"] for r in rows if not math.isfinite(r["loss"])]
    peak = max((r["peak_mem_gb"] for r in rows), default=0.0)
    checks = {
        "completed_steps": len(rows) == a.steps and not error,
        "no_nan_loss": not nan_steps,
        "no_oom": not error.startswith("CUDA OOM"),
        "peak_mem_lt_limit": peak < a.max_peak_gb,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    summary = {
        "status": status, "checks": checks, "device": str(dev), "gpu": gpu,
        "amp_active": T.amp_enabled(cfg.train.amp, dev), "config": str(a.config),
        "steps_requested": a.steps, "steps_done": len(rows), "wall_seconds": wall,
        "peak_mem_gb": peak, "max_peak_gb": a.max_peak_gb, "nan_steps": nan_steps[:20], "error": error,
        "batch": {"num_envs": cfg.ppo.num_envs, "rollout_steps": cfg.ppo.rollout_steps, "minibatch": cfg.ppo.minibatch_size},
        "windows": windows,
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "smoke_steps.csv").open("w", newline="") as h:
        if rows:
            wr = csv.DictWriter(h, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with (out / "smoke_summary.csv").open("w", newline="") as h:
        if windows:
            wr = csv.DictWriter(h, fieldnames=list(windows[0].keys())); wr.writeheader(); wr.writerows(windows)
    md = [
        f"**3070 smoke {status}** - {gpu}, amp={summary['amp_active']}, {len(rows)}/{a.steps} steps, "
        f"wall {wall / 60:.1f} min, peak {peak:.3f} GB (limit {a.max_peak_gb} GB)",
        "",
        "| steps | steps/s | rollout s | update s | loss | peak GB | grad scale |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in windows:
        md.append(f"| {w['steps']} | {w['steps_per_sec']:.3f} | {w['rollout_s']:.2f} | {w['update_s']:.2f} | "
                  f"{w['loss']:.4f} | {w['peak_mem_gb']:.3f} | {w['grad_scale']:g} |")
    md.append("")
    md.append("checks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items())
              + (f"; error: {error}" if error else ""))
    text = "\n".join(md)
    (out / "smoke_summary.md").write_text(text + "\n")
    print("\n" + text)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

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


class RolloutProfiler:
    """Exclusive-time breakdown of ``collect_rollout`` (measurement only).

    Wraps, at class level, without changing behaviour:
      * ``SelfPlayEnv.step`` / ``SelfPlayEnv.reset``      -> env_step_s (table rules,
        tokenizer, pool.sample; opponent time nested inside is subtracted)
      * ``PolicyActor.act_batch``                          -> opp_infer_s (opponent
        forward incl. kind=self/random; nested load time subtracted)
      * ``PolicyActor.prepare_pool``                       -> opp_load_s / opp_load_count
        (``shadow.load_state_dict`` of a pool snapshot)
      * learner ``forward``/``act`` outside the opponent    -> net_infer_s
    other_s = rollout_total_s - the four buckets (batching, copies into storage).
    """

    KEYS = ("env_step_s", "net_infer_s", "opp_infer_s", "opp_load_s")

    def __init__(self, cuda_sync: bool):
        self.sync = cuda_sync and torch.cuda.is_available()
        self.stack = []  # [category, start, child_time]
        self.reset()

    def reset(self):
        self.acc = {k: 0.0 for k in self.KEYS}
        self.opp_load_count = 0
        self.opp_calls = 0

    def _wrap(self, owner, name, cat):
        orig = getattr(owner, name)
        prof = self

        def wrapped(*args, **kw):
            if cat == "net_infer_s" and prof.stack and prof.stack[-1][0] in ("net_infer_s", "opp_infer_s", "opp_load_s"):
                return orig(*args, **kw)  # nested call: already counted by the outer section
            if prof.sync and cat != "env_step_s":
                torch.cuda.synchronize()
            prof.stack.append([cat, time.perf_counter(), 0.0])
            try:
                return orig(*args, **kw)
            finally:
                if prof.sync and cat != "env_step_s":
                    torch.cuda.synchronize()
                c, t0, child = prof.stack.pop()
                total = time.perf_counter() - t0
                prof.acc[c] += total - child
                if c == "opp_load_s":
                    prof.opp_load_count += 1
                if c == "opp_infer_s":
                    prof.opp_calls += 1
                if prof.stack:
                    prof.stack[-1][2] += total

        setattr(owner, name, wrapped)
        self._undo.append((owner, name, orig))

    def install(self):
        from alphanl.transformer import env as E
        from alphanl.transformer.model import PokerTransformer

        self._undo = []
        self._wrap(E.SelfPlayEnv, "step", "env_step_s")
        self._wrap(E.SelfPlayEnv, "reset", "env_step_s")
        if hasattr(E.SelfPlayEnv, "begin_step"):
            # T-C1b batched rollout calls begin_step/finish_step instead of step
            self._wrap(E.SelfPlayEnv, "begin_step", "env_step_s")
            self._wrap(E.SelfPlayEnv, "finish_step", "env_step_s")
        if hasattr(E.PolicyActor, "prefetch"):
            # T-C1b: batched opponent forward (+ encoding the opponent obs)
            self._wrap(E.PolicyActor, "prefetch", "opp_infer_s")
        self._wrap(E.PolicyActor, "act_batch", "opp_infer_s")
        if getattr(E.PolicyActor, "cache_opponents", False):
            # T-C1a: a load is building one frozen per-snapshot opponent.
            self._wrap(E.PolicyActor, "_build_opponent", "opp_load_s")
        else:
            self._wrap(E.PolicyActor, "prepare_pool", "opp_load_s")
        classes = [PokerTransformer]
        try:
            from alphanl.transformer.conv_net import AlphaNLConvNet
            classes.append(AlphaNLConvNet)
        except Exception:  # pragma: no cover
            pass
        for cls in classes:
            self._wrap(cls, "forward", "net_infer_s")
            self._wrap(cls, "act", "net_infer_s")

    def uninstall(self):
        for owner, name, orig in reversed(self._undo):
            setattr(owner, name, orig)


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
    p.add_argument("--legacy-opponents", action="store_true",
                   help="T-C1a off: one shared shadow reloaded before every opponent move (pre-T-C1a timing)")
    p.add_argument("--no-batch-rollout", action="store_true",
                   help="T-C1b off: opponent forward one move at a time (T-C1a timing)")
    p.add_argument("--per-env-rng", action="store_true",
                   help="T-C1c: env.per_env_rng=true (per-table deal + sample streams)")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch.set_num_threads(N) (+ set_num_interop_threads(N) when still allowed) before training")
    p.add_argument("--rollout-device", choices=("cpu", "cuda", "auto"), default=None,
                   help="override rollout_device (where learner/opponent inference runs during rollout)")
    p.add_argument("--profile-rollout", action="store_true",
                   help="time env step / learner inference / opponent inference / opponent loads")
    p.add_argument("--profile-csv", type=str, default=None,
                   help="per-step profile CSV (default <out-dir>/profile_steps.csv; implies --profile-rollout)")
    a = p.parse_args()

    out = Path(a.out_dir)
    if (out / "metrics.csv").exists():
        (out / "metrics.csv").unlink()  # fresh smoke each run; never touches real runs
    if a.legacy_opponents:
        from alphanl.transformer import env as _E
        _E.PolicyActor.cache_opponents = False
    if a.no_batch_rollout:
        from alphanl.transformer import env as _E
        _E.PolicyActor.batch_rollout = False
    if a.torch_threads:
        torch.set_num_threads(int(a.torch_threads))
        try:
            torch.set_num_interop_threads(int(a.torch_threads))
        except RuntimeError:
            pass  # interop pool already started; intra-op count still applies
    print(f"[smoke] torch threads intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}", flush=True)
    cfg = ExperimentConfig.from_json(a.config)
    cfg.apply_overrides(
        num_updates=a.steps, out_dir=str(out), device=a.device, seed=a.seed, amp=a.amp,
        checkpoint_dir=str(out / "ckpt"), checkpoint_every=a.steps,
    )
    if a.per_env_rng:
        cfg.env.per_env_rng = True
    if a.num_envs:
        cfg.ppo.num_envs = a.num_envs
    if a.rollout_steps:
        cfg.ppo.rollout_steps = a.rollout_steps
    if a.minibatch:
        cfg.ppo.minibatch_size = a.minibatch
    if not a.with_eval:
        cfg.train.eval_every = 0
    cfg.train.log_every = 1
    if a.rollout_device:
        cfg.rollout_device = a.rollout_device
    if a.profile_csv:
        a.profile_rollout = True
    prof = RolloutProfiler(cuda_sync=True) if a.profile_rollout else None
    prof_csv = Path(a.profile_csv) if a.profile_csv else out / "profile_steps.csv"
    if prof is not None and prof_csv.exists():
        prof_csv.unlink()
    dev = torch.device(resolve_device(cfg.device))
    cuda = dev.type == "cuda"
    print(T.describe_run(cfg))
    print(f"[smoke] device={dev} amp_requested={a.amp} amp_active={T.amp_enabled(cfg.train.amp, dev)} "
          f"steps={a.steps} out={out} rollout_device={cfg.rollout_device} profile={bool(prof)}", flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    cur = {}
    scalers = []
    rows = []
    orig_rollout, orig_update, orig_scaler, orig_csv = T.collect_rollout, T.ppo_update, T.make_grad_scaler, T._append_csv

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    # weight_sync_s: learner -> rollout model copy (state_dict transfer), only when devices differ.
    orig_wsync = T._sync_weights
    cur["weight_sync_s"] = 0.0
    def wsync(src, dst):
        _sync(); t = time.perf_counter()
        orig_wsync(src, dst)
        _sync(); cur["weight_sync_s"] += time.perf_counter() - t
    T._sync_weights = wsync
    # opp_batchable_frac: model opponent moves served from a batched prefetch / all model opponent moves.
    from alphanl.transformer import env as _E2
    orig_ab = _E2.PolicyActor.act_batch
    cur["opp_model"] = 0; cur["opp_batched"] = 0
    def act_batch(self, batch, choice, logits=None, **kw):
        if choice["kind"] != "random":
            cur["opp_model"] += 1
            cur["opp_batched"] += logits is not None
        return orig_ab(self, batch, choice, logits=logits, **kw)
    _E2.PolicyActor.act_batch = act_batch

    def rollout(*args, **kw):
        if prof is not None:
            prof.reset()
        _sync(); t = time.perf_counter()
        r = orig_rollout(*args, **kw)
        _sync(); cur["rollout_s"] = time.perf_counter() - t
        if prof is not None:
            cur["prof"] = dict(prof.acc, opp_load_count=prof.opp_load_count, opp_calls=prof.opp_calls)
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
        if prof is not None:
            pr = cur.get("prof", {})
            tot = rec["rollout_s"]
            known = sum(pr.get(k, 0.0) for k in RolloutProfiler.KEYS)
            prow = {
                "step": rec["update"] + 1,
                "env_step_s": pr.get("env_step_s", 0.0),
                "net_infer_s": pr.get("net_infer_s", 0.0),
                "opp_infer_s": pr.get("opp_infer_s", 0.0),
                "other_s": tot - known,
                "rollout_total_s": tot,
                "pool_size": int(row.get("pool_size", 0)),
                "opp_load_count": int(pr.get("opp_load_count", 0)),
                "opp_load_s": pr.get("opp_load_s", 0.0),
                "opp_calls": int(pr.get("opp_calls", 0)),
                "weight_sync_s": cur.get("weight_sync_s", 0.0),
                "opp_batchable_frac": cur["opp_batched"] / max(1, cur["opp_model"]),
            }
            cur["weight_sync_s"] = 0.0; cur["opp_model"] = 0; cur["opp_batched"] = 0
            rec.update({k: v for k, v in prow.items() if k != "step"})
            orig_csv(prof_csv, prow)
        rows.append(rec)
        n = len(rows)
        if n % a.log_every == 0 or n == a.steps:
            _report(rows[-(((n - 1) % a.log_every) + 1):], n)

    T.collect_rollout, T.ppo_update, T.make_grad_scaler, T._append_csv = rollout, update, scaler, append_csv
    if prof is not None:
        prof.install()
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
        if prof is not None:
            prof.uninstall()
        T._sync_weights = orig_wsync
        _E2.PolicyActor.act_batch = orig_ab
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
        **({
            "env_step_s": m("env_step_s"), "net_infer_s": m("net_infer_s"), "opp_infer_s": m("opp_infer_s"),
            "other_s": m("other_s"), "pool_size": chunk[-1]["pool_size"],
            "opp_load_count": m("opp_load_count"), "opp_load_s": m("opp_load_s"),
            "weight_sync_s": m("weight_sync_s"), "opp_batchable_frac": m("opp_batchable_frac"),
        } if "env_step_s" in chunk[0] else {}),
    }


def _report(chunk, n):
    w = _window(chunk, chunk[-1]["update"] + 1)
    print(f"[smoke] steps {w['steps']}: {w['steps_per_sec']:.3f} steps/s, rollout {w['rollout_s']:.2f}s, "
          f"update {w['update_s']:.2f}s, loss {w['loss']:.4f}, peak {w['peak_mem_gb']:.3f} GB, "
          f"scale {w['grad_scale']:g}", flush=True)
    if "env_step_s" in w:
        print(f"[profile] steps {w['steps']}: env {w['env_step_s']:.3f}s, net {w['net_infer_s']:.3f}s, "
              f"opp {w['opp_infer_s']:.3f}s, other {w['other_s']:.3f}s, pool {w['pool_size']}, "
              f"opp loads {w['opp_load_count']:.1f}/step in {w['opp_load_s']:.3f}s, "
              f"weight_sync {w['weight_sync_s']:.4f}s, opp batchable {100 * w['opp_batchable_frac']:.1f}%", flush=True)


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
        "rollout_device": cfg.rollout_device,
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
    if windows and "env_step_s" in windows[0]:
        md += ["", f"rollout profile (avg seconds per step; rollout_device={cfg.rollout_device})", "",
               "| steps | env_step_s | net_infer_s | opp_infer_s | other_s | pool_size | opp_load_count | opp_load_s |",
               "|---|---|---|---|---|---|---|---|"]
        for w in windows:
            md.append(f"| {w['steps']} | {w['env_step_s']:.3f} | {w['net_infer_s']:.3f} | {w['opp_infer_s']:.3f} | "
                      f"{w['other_s']:.3f} | {w['pool_size']} | {w['opp_load_count']:.1f} | {w['opp_load_s']:.3f} |")
    md.append("")
    md.append("checks: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items())
              + (f"; error: {error}" if error else ""))
    text = "\n".join(md)
    (out / "smoke_summary.md").write_text(text + "\n")
    print("\n" + text)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

"""T-C2 acceptance: multiprocess rollout == single-process rollout, bit for bit.

    PYTHONPATH=src python scripts/test_t_c2_equivalence.py [--updates 5] [--device cpu|cuda] [--workers 1,2,3,8] --deterministic

Fresh subprocess per mode, same seed, env.per_env_rng=true, pool.defer_record=true:
  ref  ppo.rollout_workers=0 (single process; learner on --device, rollout on cpu)
  wN   ppo.rollout_workers=N ('spawn' workers, 1 torch thread each), 8 envs -> splits incl. uneven (3 -> 3/3/2)
  sN   single process, forwards batched over the same N env slices as the workers
CPU matmul/conv results depend on batch size (~1e-8), so bitwise is checked wN vs sN (w1 vs ref).
PASS iff for every N, bit-identical: every rollout buffer, post-update params after each of K updates,
optimizer state, final table state_dicts (incl. deal/sample streams) and pool state (ids + win/game stats);
and wN vs ref: obs/actions/rewards/dones and streams/pool stats bit-identical, logp/values <= 1e-5;
and every worker reports torch.get_num_threads() == 1.
Also checks: workers=N without per_env_rng raises; a crashed worker and a hung worker raise RolloutWorkerError
(with worker id) instead of hanging (--skip-fault to skip).
On cuda, --deterministic needs CUBLAS_WORKSPACE_CONFIG=':4096:8' set BEFORE python starts.
"""
from __future__ import annotations

import argparse, copy, subprocess, sys, tempfile, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import test_t_c1a_equivalence as A  # noqa: E402


def _cfg(out: Path, updates: int, device: str, workers: int):
    cfg = A._cfg(out, updates, device)
    cfg.rollout_device = "cpu"
    cfg.env.per_env_rng = True
    cfg.pool.defer_record = True
    cfg.ppo.rollout_workers = workers
    cfg.ppo.rollout_worker_timeout_s = 300.0
    return cfg


def child(mode: str, out: Path, updates: int, device: str) -> None:
    from alphanl.transformer import train as T
    from alphanl.transformer.checkpoint import latest_checkpoint
    workers = int(mode[1:]) if mode.startswith("w") else 0
    rec = {"storage": [], "params": [], "threads": None, "startup_s": None, "bcast": [], "poolsync": []}

    o1, o2 = T.collect_rollout, T.collect_rollout_workers
    def c1(*a, **k):
        r = o1(*a, **k); rec["storage"].append({kk: v.clone() for kk, v in r[0].items()}); return r
    def c2(w, pool):
        if rec["threads"] is None:
            rec["threads"], rec["startup_s"] = list(w.worker_threads), w.startup_s
        r = o2(w, pool); rec["storage"].append({kk: v.clone() for kk, v in r[0].items()})
        rec["bcast"].append(w.last["weight_broadcast_s"]); return r
    if mode.startswith("s"):
        # Single process, but the hero/opponent forwards batched over the same contiguous
        # env slices the workers use (CPU kernels are batch-size dependent at ~1e-8).
        from alphanl.transformer.rollout_workers import split_envs
        def c1(envs, obs, model, device, steps):  # noqa: F811
            parts, stats = [], {"episodes": 0, "passive_list": []}
            for sl in split_envs(len(envs), int(mode[1:])):
                sub = [obs[i] for i in sl]
                st, ss = o1([envs[i] for i in sl], sub, model, device, steps)
                for j, i in enumerate(sl):
                    obs[i] = sub[j]
                parts.append(st); stats["episodes"] += ss["episodes"]; stats["passive_list"] += ss["passive_list"]
            st = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
            term = st["dones"] > 0.5
            stats.update(mean_reward_bb=float(st["rewards"][term].mean()) if int(term.sum()) else 0.0,
                         passive_hands=len(stats["passive_list"]), passive_mean_bb=0.0)
            rec["storage"].append({kk: v.clone() for kk, v in st.items()})
            return st, stats
    T.collect_rollout, T.collect_rollout_workers = c1, c2
    ou = T.ppo_update
    def upd(model, opt, *a, **k):
        m = ou(model, opt, *a, **k)
        rec["params"].append({kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()})
        rec["opt"] = _cpu(opt.state_dict())
        return m
    T.ppo_update = upd
    cfg = _cfg(out, updates, device, workers)
    T.run_training(cfg, resume=None)
    blob = torch.load(latest_checkpoint(Path(cfg.train.checkpoint_dir)), weights_only=False, map_location="cpu")
    rec["envs"] = blob["envs"]
    rec["pool"] = blob["pool"]
    torch.save(rec, out / f"{mode}.pt")


def _cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    return copy.deepcopy(x)


def _spawn(mode, d: Path, a):
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--child", mode, "--out", str(d), "--updates", str(a.updates), "--device", a.device]
    if a.deterministic:
        cmd.append("--deterministic")
    t = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"[c2] mode {mode}: {time.perf_counter() - t:.1f}s", flush=True)
    return torch.load(d / f"{mode}.pt", weights_only=False, map_location="cpu")


FLOAT_KEYS = ("old_logp", "old_values", "bootstrap")
TOL_ABS, TOL_REL = 1e-5, 1e-3


def compare_tol(name, X, Y) -> bool:
    """Decisions bit-identical, float outputs within tolerance (different batch composition)."""
    dec = all(A._eq({k: v for k, v in x.items() if k not in FLOAT_KEYS},
                    {k: v for k, v in y.items() if k not in FLOAT_KEYS}) == []
              for x, y in zip(X["storage"], Y["storage"])) and len(X["storage"]) == len(Y["storage"])
    ferr = max(float((x[k] - y[k]).abs().max()) for x, y in zip(X["storage"], Y["storage"]) for k in FLOAT_KEYS)
    streams = A._eq([e["streams"] for e in X["envs"]], [e["streams"] for e in Y["envs"]]) == []
    pstats = ({k: (v["wins"], v["games"]) for k, v in X["pool"]["entries"].items()}
              == {k: (v["wins"], v["games"]) for k, v in Y["pool"]["entries"].items()})
    rel, _ = A._max_rel(X["params"][-1], Y["params"][-1])
    ok = dec and streams and pstats and ferr <= TOL_ABS and rel <= TOL_REL
    print(f"[c2] {name}: obs/actions/rewards/dones bit-identical={dec} streams={streams} pool_stats={pstats} "
          f"logp/value max abs err={ferr:.2e} max_rel_param_err={rel:.2e} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def _deep(a, b, path, seen) -> list:
    """Semantic equality: unpickles byte blobs, walks objects' __dict__, RNGs by state."""
    import pickle
    import numpy as np
    if isinstance(a, bytes) and isinstance(b, bytes):
        if a == b:
            return []
        try:
            a, b = pickle.loads(a), pickle.loads(b)
        except Exception:
            return [path]
    if id(a) in seen:
        return []
    if torch.is_tensor(a) or isinstance(a, np.ndarray):
        return A._eq(a, b, path)
    if isinstance(a, np.random.Generator):
        return [] if a.bit_generator.state == b.bit_generator.state else [path]
    if isinstance(a, np.random.RandomState):
        sa, sb = a.get_state(), b.get_state()
        return [] if sa[0] == sb[0] and np.array_equal(sa[1], sb[1]) and sa[2:] == sb[2:] else [path]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [path + " keys"]
        return [d for k in a for d in _deep(a[k], b[k], f"{path}.{k}", seen)]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [path + " len"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _deep(x, y, f"{path}[{i}]", seen)]
    if hasattr(a, "__dict__") and not isinstance(a, type):
        if type(a) is not type(b):
            return [path + " type"]
        seen.add(id(a))
        return _deep(vars(a), vars(b), path, seen)
    try:
        return [] if a == b else [path]
    except Exception:
        return [path]


def compare(name, X, Y) -> bool:
    bad = {}
    for k in ("storage", "params", "opt", "envs", "pool"):
        diff = _deep(X[k], Y[k], k, set()) if k == "envs" else A._eq(X[k], Y[k], k)
        bad[k] = not diff
        if diff:
            print(f"[c2]   {name} {k} differs at: {diff[:5]}")
    rel, _ = A._max_rel(X["params"][-1], Y["params"][-1])
    ok = all(bad.values()) and len(X["storage"]) == len(Y["storage"])
    print(f"[c2] {name}: bit-identical {bad}  rollouts={len(Y['storage'])}  max_rel_param_err={rel:.3e} "
          f"-> {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def fault_checks(device: str) -> bool:
    from alphanl.transformer.rollout_workers import RolloutWorkers, RolloutWorkerError
    from alphanl.transformer.train import build_model
    from alphanl.transformer.pool import OpponentPool
    out = Path(tempfile.mkdtemp(prefix="tc2f_"))
    ok = True
    cfg = _cfg(out, 1, "cpu", 2)
    cfg.env.per_env_rng = False
    from alphanl.transformer import train as T
    try:
        T.run_training(cfg)
        print("[c2] fault: no per_env_rng -> no error  FAIL"); ok = False
    except ValueError as exc:
        print(f"[c2] fault: no per_env_rng -> ValueError OK ({exc})")
    cfg = _cfg(out, 1, "cpu", 2)
    model = build_model(cfg)
    pool = OpponentPool(cfg.pool.max_size, cfg.pool.selfplay_prob, cfg.pool.random_prob, cfg.pool.sampling, cfg.seed)
    pool.add(model.state_dict())
    for hook, timeout in (("crash", 60.0), ("hang", 8.0)):
        w = RolloutWorkers(cfg, model.state_dict(), pool, timeout_s=timeout)
        w.conns[1].send((hook,))
        t = time.perf_counter()
        try:
            w.collect(pool)
            print(f"[c2] fault: {hook} -> no error  FAIL"); ok = False
        except RolloutWorkerError as exc:
            msg = str(exc).splitlines()[0]
            good = "worker 1" in msg and not any(p.is_alive() for p in w.procs)
            ok &= good
            print(f"[c2] fault: {hook} -> RolloutWorkerError in {time.perf_counter() - t:.1f}s: {msg} "
                  f"(all workers stopped={not any(p.is_alive() for p in w.procs)}) {'OK' if good else 'FAIL'}")
        finally:
            w.close()
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--device", default="cpu", help="learner/update device; rollout always cpu")
    p.add_argument("--workers", default="1,2,3,8")
    p.add_argument("--child")
    p.add_argument("--out", type=Path)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--skip-fault", action="store_true")
    a = p.parse_args()
    A._setup_determinism(a.deterministic, a.device)
    if a.child:
        child(a.child, a.out, a.updates, a.device)
        return 0
    out = Path(tempfile.mkdtemp(prefix="tc2_"))
    ns = [int(n) for n in a.workers.split(",")]
    modes = ["ref"] + [f"s{n}" for n in ns if n > 1] + [f"w{n}" for n in ns]
    R = {m: _spawn(m, out / m, a) for m in modes}
    print(f"[c2] updates={a.updates} device={a.device} rollout=cpu deterministic={a.deterministic} "
          f"per_env_rng=True defer_record=True envs=8")
    ok = True
    for m in [x for x in modes if x.startswith("w")]:
        n = int(m[1:])
        th = R[m]["threads"]
        good = th is not None and all(t == 1 for t in th)
        ok &= good
        print(f"[c2] {m}: worker torch threads={th} {'OK' if good else 'FAIL'}  startup={R[m]['startup_s']:.2f}s  "
              f"weight_broadcast mean={sum(R[m]['bcast']) / max(1, len(R[m]['bcast'])):.4f}s")
        base = "ref" if n == 1 else f"s{n}"
        ok &= compare(f"{base} vs {m} (bitwise)", R[base], R[m])
        if n > 1:
            ok &= compare_tol(f"ref vs {m} (unsliced ref)", R["ref"], R[m])
    if not a.skip_fault:
        ok &= fault_checks(a.device)
    print(f"[c2] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

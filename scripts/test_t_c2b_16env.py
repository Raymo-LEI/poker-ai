"""T-C2b: 16 envs x 64 steps vs 8 envs x 128 steps (same samples/update), GAE bootstrap, uneven-split warning.

    PYTHONPATH=src python scripts/test_t_c2b_16env.py [--device cpu|cuda] [--updates 3] [--deterministic] [--skip a,b,c,d]

(a) configs laptop_8gb.json (8x128) vs laptop_8gb_16env.json (16x64), one real update each: same per-update
    sample count (1024), same flat storage tensor shapes, same minibatch split sizes (ppo_losses batch sizes
    over all epochs); per-env storage is [16,64] vs [8,128]; every other hyperparameter identical.
(b) GAE bootstrap at rollout truncation: hand-built compute_gae case (non-terminal truncated env uses
    gamma*V(next obs); terminal env ignores it), and in the real collect_rollout path storage["bootstrap"]
    == value head of the obs each env holds after the rollout (the next hero decision).
(c) 16 envs, rollout_workers 8 (2 envs/worker) vs single process, deterministic, --updates updates (default 3),
    reusing test_t_c2_equivalence: w8 bit-identical to s8 (single process, forwards batched over the same 8
    slices) in every buffer/param/optimizer/env/pool state; w8 vs ref (w0, unsliced 16-env batch):
    obs/actions/rewards/dones + RNG streams + pool stats bit-identical, logp/value <= 1e-5 (CPU kernels
    are batch-size dependent at ~1e-7, same as T-C2).
(d) uneven-split warning printed for 8 envs / 3 workers, not for 16 / 8 (real RolloutWorkers startup).
"""
from __future__ import annotations

import argparse, contextlib, io, subprocess, sys, tempfile, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import test_t_c1a_equivalence as A  # noqa: E402
import test_t_c2_equivalence as C2  # noqa: E402

CFG8 = REPO / "configs/transformer/laptop_8gb.json"
CFG16 = REPO / "configs/transformer/laptop_8gb_16env.json"
C_ENVS, C_STEPS, C_MB = 16, 16, 64   # (c): small but 2 envs/worker at w8


# ------------------------------------------------------------------ (a) + real-path bootstrap
def _run_one(cfg_path: Path, device: str, out: Path) -> dict:
    from alphanl.transformer import train as T, ppo as P
    from alphanl.transformer.config import ExperimentConfig
    cfg = ExperimentConfig.from_json(cfg_path)
    cfg.device, cfg.rollout_device = device, "cpu"
    cfg.env.per_env_rng = True
    cfg.ppo.rollout_workers = 0  # (a)/(b) instrument the in-process path; (c) covers workers
    cfg.train.num_updates, cfg.train.eval_every, cfg.train.checkpoint_every = 1, 0, 0
    cfg.train.out_dir, cfg.train.checkpoint_dir = str(out / "run"), str(out / "ckpt")
    rec = {"mb": [], "boot_err": None}
    oc, ol, of = T.collect_rollout, P.ppo_losses, T.flatten_rollout

    def c(envs, obs, model, dev, steps):
        st, ss = oc(envs, obs, model, dev, steps)
        rec["storage_shapes"] = {k: tuple(v.shape) for k, v in st.items()}
        with torch.no_grad():
            _, v = model(T.batch_to_device(T.cat_batches(obs), dev), mask_illegal=True)
        rec["boot_err"] = float((v.cpu() - st["bootstrap"]).abs().max())
        rec["nonterm_last"] = int((st["dones"][:, -1] < 0.5).sum())
        return st, ss

    def f(storage, g, l):
        flat = of(storage, g, l)
        rec["flat_shapes"] = {k: tuple(v.shape) for k, v in flat.items()}
        return flat

    def lo(model, batch, *a, **k):
        rec["mb"].append(int(k["actions"].shape[0]))
        return ol(model, batch, *a, **k)
    T.collect_rollout, P.ppo_losses, T.flatten_rollout = c, lo, f
    try:
        T.run_training(cfg, resume=None)
    finally:
        T.collect_rollout, P.ppo_losses, T.flatten_rollout = oc, ol, of
    rec["cfg"] = cfg.to_dict()
    return rec


def test_a(device: str) -> bool:
    out = Path(tempfile.mkdtemp(prefix="tc2b_a_"))
    R8, R16 = _run_one(CFG8, device, out / "8"), _run_one(CFG16, device, out / "16")
    n8, n16 = R8["flat_shapes"]["actions"][0], R16["flat_shapes"]["actions"][0]
    d8 = {k: v for k, v in R8["cfg"].items() if k != "name"}
    d16 = {k: v for k, v in R16["cfg"].items() if k != "name"}
    p8, p16 = dict(d8.pop("ppo")), dict(d16.pop("ppo"))
    for p in (p8, p16):
        p.pop("num_envs"); p.pop("rollout_steps")
    for d in (d8, d16):  # per-run temp dirs
        d["train"] = {k: v for k, v in d["train"].items() if k not in ("out_dir", "checkpoint_dir")}
    hyp = d8 == d16 and p8 == p16
    ok = (n8 == n16 == 1024 and R8["flat_shapes"] == R16["flat_shapes"] and R8["mb"] == R16["mb"]
          and R8["storage_shapes"]["rewards"] == (8, 128) and R16["storage_shapes"]["rewards"] == (16, 64) and hyp)
    print(f"[c2b] (a) samples/update 8x128={n8} 16x64={n16}; flat shapes equal={R8['flat_shapes'] == R16['flat_shapes']}; "
          f"minibatches 8x128={len(R8['mb'])}x{sorted(set(R8['mb']))} 16x64={len(R16['mb'])}x{sorted(set(R16['mb']))} "
          f"equal={R8['mb'] == R16['mb']}; per-env rewards {R8['storage_shapes']['rewards']} vs "
          f"{R16['storage_shapes']['rewards']}; other hyperparams identical={hyp} -> {'OK' if ok else 'FAIL'}", flush=True)
    okb = all(r["boot_err"] == 0.0 for r in (R8, R16))
    print(f"[c2b] (b) real path: bootstrap == V(obs after rollout) max abs err 8x128={R8['boot_err']:.1e} "
          f"16x64={R16['boot_err']:.1e} (non-terminal truncated envs: {R8['nonterm_last']}/8, {R16['nonterm_last']}/16) "
          f"-> {'OK' if okb else 'FAIL'}", flush=True)
    return ok and okb


# ------------------------------------------------------------------ (b) hand-built GAE
def test_b_unit() -> bool:
    from alphanl.transformer.ppo import compute_gae
    g, lam = 0.9, 0.5
    r = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    v = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    d = torch.tensor([[0.0, 0.0, 0.0],    # env0: truncated mid-hand -> bootstrap
                      [0.0, 0.0, 1.0],    # env1: hand ends on last step -> no bootstrap
                      [0.0, 1.0, 0.0]])   # env2: hand ended at t=1, new hand truncated at t=2
    boot = torch.tensor([5.0, 7.0, 11.0])
    adv, ret = compute_gae(r, v, d, boot, g, lam)
    exp = torch.zeros(3, 3)
    for e in range(3):
        nv, na = float(boot[e]), 0.0
        for t in (2, 1, 0):
            m = 1.0 - float(d[e, t])
            delta = float(r[e, t]) + g * nv * m - float(v[e, t])
            na = delta + g * lam * na * m
            exp[e, t] = na
            nv = float(v[e, t])
    last_ok = (abs(float(adv[0, 2]) - (1.0 + g * 5.0 - 0.3)) < 1e-6 and abs(float(adv[1, 2]) - (2.0 - 0.6)) < 1e-6
               and abs(float(adv[2, 2]) - (0.0 + g * 11.0 - 0.9)) < 1e-6
               and abs(float(adv[2, 1]) - (3.0 - 0.8)) < 1e-6)
    ok = torch.allclose(adv, exp, atol=1e-6) and torch.allclose(ret, adv + v) and last_ok
    print(f"[c2b] (b) hand-built GAE: truncated env uses r+g*V(next)-V, terminal ignores bootstrap, "
          f"done mid-rollout cuts the trace -> {'OK' if ok else 'FAIL'}", flush=True)
    return ok


# ------------------------------------------------------------------ (c) 16 envs w0/s8/w8
def _cfg16(out: Path, updates: int, device: str, workers: int):
    cfg = C2._ORIG_CFG(out, updates, device, workers)
    cfg.ppo.num_envs, cfg.ppo.rollout_steps, cfg.ppo.minibatch_size = C_ENVS, C_STEPS, C_MB
    return cfg


def _spawn(mode, d: Path, a):
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--child", mode, "--out", str(d), "--updates", str(a.updates), "--device", a.device]
    if a.deterministic:
        cmd.append("--deterministic")
    t = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"[c2b] (c) mode {mode}: {time.perf_counter() - t:.1f}s", flush=True)
    return torch.load(d / f"{mode}.pt", weights_only=False, map_location="cpu")


def test_c(a) -> bool:
    out = Path(tempfile.mkdtemp(prefix="tc2b_c_"))
    R = {m: _spawn(m, out / m, a) for m in ("ref", "s8", "w8")}
    shp = tuple(R["w8"]["storage"][0]["rewards"].shape)
    th = R["w8"]["threads"]
    ok = shp == (C_ENVS, C_STEPS) and th is not None and len(th) == 8 and all(t == 1 for t in th)
    print(f"[c2b] (c) envs={C_ENVS} steps={C_STEPS} updates={a.updates} workers=8 (2 envs each) threads={th} "
          f"storage {shp}", flush=True)
    ok &= C2.compare("s8 vs w8 (bitwise)", R["s8"], R["w8"])
    ok &= C2.compare_tol("ref(w0) vs w8 (unsliced ref)", R["ref"], R["w8"])
    return ok


# ------------------------------------------------------------------ (d) warning
def test_d() -> bool:
    from alphanl.transformer.rollout_workers import RolloutWorkers, uneven_split_warning
    from alphanl.transformer.train import build_model
    from alphanl.transformer.pool import OpponentPool
    ok = True
    for envs, w, want in ((8, 3, True), (16, 8, False)):
        cfg = C2._ORIG_CFG(Path(tempfile.mkdtemp(prefix="tc2b_d_")), 1, "cpu", w)
        cfg.ppo.num_envs = envs
        model = build_model(cfg)
        pool = OpponentPool(cfg.pool.max_size, cfg.pool.selfplay_prob, cfg.pool.random_prob, cfg.pool.sampling, cfg.seed)
        pool.add(model.state_dict())
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            rw = RolloutWorkers(cfg, model.state_dict(), pool)
        rw.close()
        got = "WARNING" in buf.getvalue() and "not divisible" in buf.getvalue()
        good = got == want and (uneven_split_warning(envs, w) is not None) == want
        ok &= good
        print(f"[c2b] (d) envs={envs} workers={w}: warning={'yes' if got else 'no'} (expected {'yes' if want else 'no'}) "
              f"{'OK' if good else 'FAIL'}" + (f"\n        {buf.getvalue().strip()}" if got else ""), flush=True)
    return ok


C2._ORIG_CFG = C2._cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--skip", default="")
    p.add_argument("--child")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    A._setup_determinism(a.deterministic, a.device)
    if a.child:
        C2._cfg = _cfg16
        C2.child(a.child, a.out, a.updates, a.device)
        return 0
    skip = set(a.skip.split(",")) - {""}
    ok = True
    if "b" not in skip:
        ok &= test_b_unit()
    if "a" not in skip:
        ok &= test_a(a.device)
    if "c" not in skip:
        ok &= test_c(a)
    if "d" not in skip:
        ok &= test_d()
    print(f"[c2b] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

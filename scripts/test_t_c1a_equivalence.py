"""T-C1a acceptance: per-snapshot opponent cache vs legacy shared shadow.

    python3 scripts/test_t_c1a_equivalence.py [--updates 5] [--device cpu] [--deterministic]
    python3 scripts/test_t_c1a_equivalence.py --legacy-twice [--device cuda] [--deterministic]
    python3 scripts/test_t_c1a_equivalence.py --per-step 5 [--device cuda] [--deterministic]

--deterministic: torch.use_deterministic_algorithms(True) + cudnn deterministic (test only).
  On cuda, CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' BEFORE python starts.
--legacy-twice: run the legacy path twice, compare bitwise (control: nondeterminism w/o cache).
--per-step K: K trials; trial t resumes both paths from the same checkpoint (after t updates,
  incl. full python/numpy/torch cpu+cuda RNG, tables, actor RNG), runs 1 rollout + 1 update;
  pass iff actions identical and max relative param error <= 1e-5.

Runs run_training twice in fresh subprocesses (legacy, cached) with the same
seed/device, capturing every rollout buffer, every opponent action, post-update
params, optimizer state, pool and all RNG states, then asserts torch.equal on all.
Also checks opp builds == 1 per snapshot id and the frozen-copy guarantees.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _setup_determinism(enabled: bool, device: str) -> None:
    if not enabled:
        return
    if torch.device(device).type == "cuda":
        cfgv = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cfgv not in (":4096:8", ":16:8"):
            sys.exit(f"ERROR: --deterministic on cuda needs CUBLAS_WORKSPACE_CONFIG=':4096:8' (or ':16:8') "
                     f"set in the shell BEFORE python starts; got {cfgv!r}. "
                     "PowerShell: $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cfg(out: Path, updates: int, device: str, ckpt_every: int = 0):
    from alphanl.transformer.config import ExperimentConfig
    cfg = ExperimentConfig.from_json(REPO / "configs/transformer/laptop_8gb.json")
    cfg.device = device
    cfg.rollout_device = device
    cfg.ppo.num_envs = 8
    cfg.ppo.rollout_steps = 32
    cfg.ppo.minibatch_size = 64
    cfg.pool.snapshot_every = 1   # pool grows every update
    cfg.pool.max_size = 3         # ...and evicts from update 3 on
    cfg.pool.selfplay_prob = 0.25
    cfg.pool.random_prob = 0.1
    cfg.train.num_updates = updates
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = ckpt_every
    cfg.train.keep_last = 0       # keep every checkpoint (per-step trials resume from each)
    cfg.train.out_dir = str(out / "run")
    cfg.train.checkpoint_dir = str(out / "ckpt")
    return cfg


def child(mode: str, out: Path, updates: int, device: str, resume=None, ckpt_every: int = 0,
          tag: str = "") -> None:
    from alphanl.transformer import env as E, train as T
    E.PolicyActor.cache_opponents = (mode == "cached")
    rec = {"storage": [], "params": [], "opp": [], "builds": {}, "loads": 0}

    orig_collect = T.collect_rollout
    def collect(*a, **k):
        res = orig_collect(*a, **k)
        rec["storage"].append({kk: v.clone() for kk, v in res[0].items() if torch.is_tensor(v)})
        return res
    T.collect_rollout = collect

    orig_update = T.ppo_update
    def update(model, opt, *a, **k):
        m = orig_update(model, opt, *a, **k)
        rec["params"].append({kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()})
        rec["opt"] = copy.deepcopy(opt.state_dict())
        return m
    T.ppo_update = update

    orig_act = E.PolicyActor.act_batch
    def act_batch(self, batch, choice):
        a = orig_act(self, batch, choice)
        rec["opp"].append((choice["kind"], choice.get("policy_id"), a))
        rec["actor_rng"] = self.rng_state()
        return a
    E.PolicyActor.act_batch = act_batch

    orig_prep = E.PolicyActor.prepare_pool
    def prep(self, weights, *a, **k):
        if not E.PolicyActor.cache_opponents:
            rec["loads"] += 1
        r = orig_prep(self, weights, *a, **k)
        rec["cache_ids"] = sorted(self._opponents)
        return r
    E.PolicyActor.prepare_pool = prep
    if hasattr(E.PolicyActor, "_build_opponent"):
        orig_build = E.PolicyActor._build_opponent
        def build(self, weights):
            m = orig_build(self, weights)
            rec["loads"] += 1
            return m
        E.PolicyActor._build_opponent = build
        orig_opp = E.PolicyActor._opponent
        def opp(self, choice):
            pid = choice.get("policy_id")
            if pid is not None and int(pid) not in self._opponents:
                rec["builds"][int(pid)] = rec["builds"].get(int(pid), 0) + 1
            return orig_opp(self, choice)
        E.PolicyActor._opponent = opp

    T.run_training(_cfg(out, updates, device, ckpt_every), resume=resume)
    rec["torch_rng"] = torch.get_rng_state()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        rec["cuda_rng"] = torch.cuda.get_rng_state_all()
    rec["np_rng"] = np.random.get_state()[1].copy()
    torch.save(rec, out / f"{tag or mode}.pt")


def _eq(a, b, path="") -> list:
    bad = []
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
            bad.append(path)
    elif isinstance(a, dict):
        if set(a) != set(b):
            bad.append(path + " keys")
        for k in a:
            if k in b:
                bad += _eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            bad.append(path + " len")
        for i, (x, y) in enumerate(zip(a, b)):
            bad += _eq(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            bad.append(path)
    elif a != b:
        bad.append(path)
    return bad


def _assert_on(tensors: dict, device: str) -> None:
    want = torch.device(device)
    off = {k: str(v.device) for k, v in tensors.items()
           if torch.is_tensor(v) and v.device.type != want.type}
    assert not off, f"tensors not on {device}: {off}"


def _hash(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def frozen_test(device: str) -> None:
    """Cached opponent is a frozen deep copy: unchanged by a PPO update."""
    from alphanl.transformer import train as T
    from alphanl.transformer.env import PolicyActor
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.ppo import ppo_update
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer
    from alphanl.transformer.env import SelfPlayEnv
    cfg = _cfg(Path(tempfile.mkdtemp()), 1, device)
    T.set_seed(0)
    learner = T.build_model(cfg).to(device)
    opt = T.build_optimizer(learner, cfg)
    pool = OpponentPool(3, 0.0, 0.0, seed=0)
    pid = pool.add(learner.state_dict())
    tok = BettingHistoryTokenizer(max_seq_len=cfg.model.max_seq_len)
    actor = PolicyActor(learner, T.build_model(cfg).to(device), tok, np.random.default_rng(0))
    choice = pool.sample()
    assert choice["kind"] == "pool" and choice["policy_id"] == pid
    actor.prepare_pool(choice["weights"], policy_id=pid, live_ids=pool.live_ids())
    opp = actor._opponents[pid]
    assert not opp.training and all(not p.requires_grad for p in opp.parameters())
    learner_ptrs = {p.data_ptr() for p in learner.parameters()}
    pool_ptrs = {v.data_ptr() for v in choice["weights"].values()}
    assert not ({p.data_ptr() for p in opp.parameters()} & (learner_ptrs | pool_ptrs)), "opponent aliases live tensors"
    h_opp, h_pool = _hash(opp), hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in choice["weights"].values())).hexdigest()
    # one real PPO update on the learner from a tiny rollout
    envs = [SelfPlayEnv(tok, actor, pool, seed=10 + i, observer=T.observer_for(cfg)) for i in range(4)]
    obs = [e.reset() for e in envs]
    storage, _ = T.collect_rollout(envs, obs, learner, torch.device(device), 16)
    flat = T.flatten_rollout(storage, 1.0, 0.95)
    # collect_rollout builds its buffers on CPU; train.py moves them to the learner
    # device before ppo_update (train.py: `flat = {k: v.to(train_device) ...}`). Mirror it.
    flat = {k: v.to(device) for k, v in flat.items()}
    _assert_on(flat, device)
    before = _hash(learner)
    # guard: every tensor fed to the learner during ppo_update must be on its device
    def _dev_hook(mod, args, kwargs):
        dev = next(mod.parameters()).device
        for x in list(args) + list(kwargs.values()):
            for t in (x.values() if isinstance(x, dict) else [x]):
                assert not torch.is_tensor(t) or t.device == dev, f"input on {t.device}, model on {dev}"
    hook = learner.register_forward_pre_hook(_dev_hook, with_kwargs=True)
    learner.train()
    ppo_update(learner, opt, {k: v for k, v in flat.items() if k not in T._PPO_SKIP},
               actions=flat["actions"], old_logp=flat["old_logp"], old_values=flat["old_values"],
               advantages=flat["advantages"], returns=flat["returns"], clip_eps=0.2, entropy_coef=0.01,
               value_coef=0.5, epochs=2, minibatch_size=32, max_grad_norm=0.5)
    hook.remove()
    assert _hash(learner) != before, "PPO update did not change learner (test is vacuous)"
    assert _hash(opp) == h_opp, "cached opponent changed after PPO update"
    assert hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in choice["weights"].values())).hexdigest() == h_pool
    # eviction: while a table is mid-hand vs pid it must stay cached ...
    pool.add(learner.state_dict()); pool.add(learner.state_dict()); pool.add(learner.state_dict())
    assert pid not in pool.live_ids()
    actor.prune_opponents(pool.live_ids())
    assert pid in actor._opponents, "evicted while a table still plays it"
    # ... and is dropped once no table plays it
    for e in envs:
        e.choice = {"kind": "random", "policy_id": None, "weights": None}
    c2 = {"kind": "pool", "policy_id": pool.live_ids()[0], "weights": pool._entries[pool.live_ids()[0]]["weights"]}
    actor.prepare_pool(c2["weights"], policy_id=c2["policy_id"], live_ids=pool.live_ids())
    assert pid not in actor._opponents and set(actor._opponents) <= set(pool.live_ids())
    print(f"[frozen] OK: opponent hash unchanged across PPO update ({h_opp[:12]}), no aliasing, eval+no-grad, eviction OK")


def _spawn(mode, d: Path, a, updates=None, resume=None, ckpt_every=0, tag=""):
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--child", mode, "--out", str(d),
           "--updates", str(a.updates if updates is None else updates), "--device", a.device,
           "--ckpt-every", str(ckpt_every)]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if tag:
        cmd += ["--tag", tag]
    if a.deterministic:
        cmd.append("--deterministic")
    subprocess.run(cmd, check=True, cwd=REPO)
    return torch.load(d / f"{tag or mode}.pt", weights_only=False, map_location="cpu")


def _compare(L, C) -> list:
    bad = []
    for key in ("storage", "params", "opt", "opp", "actor_rng", "torch_rng", "np_rng", "cuda_rng"):
        if key not in L and key not in C:
            continue
        bad += _eq(L[key], C[key], key)
    return bad


def _report_bad(prefix: str, bad: list) -> None:
    print(f"{prefix} mismatching items: {len(bad)}")
    for b in bad[:20]:
        print(f"{prefix}   {b}")
    if len(bad) > 20:
        print(f"{prefix}   ... (+{len(bad) - 20} more)")


def legacy_twice(a) -> int:
    out = Path(tempfile.mkdtemp(prefix="tc1a_l2_"))
    A = _spawn("legacy", out / "a", a, tag="legacy_a")
    B = _spawn("legacy", out / "b", a, tag="legacy_b")
    bad = _compare(A, B)
    print(f"[legacy-twice] updates={a.updates} device={a.device} deterministic={a.deterministic} "
          f"opp moves={len(A['opp'])}/{len(B['opp'])}")
    if bad:
        _report_bad("[legacy-twice]", bad)
        print(f"[legacy-twice] DIVERGED n={len(bad)}")
    else:
        print("[legacy-twice] IDENTICAL")
    return 0


def _actions(rec) -> tuple:
    acts = [s["actions"] for s in rec["storage"] if "actions" in s]
    opp = [(k, p, (x.tolist() if torch.is_tensor(x) else (x.tolist() if isinstance(x, np.ndarray) else x)))
           for k, p, x in rec["opp"]]
    return acts, opp


def _max_rel(pa: dict, pb: dict):
    worst, name = 0.0, None
    for k in pa:
        x, y = pa[k].double(), pb[k].double()
        if not x.is_floating_point():
            r = 0.0 if torch.equal(pa[k], pb[k]) else float("inf")
        else:
            r = float((x - y).abs().max() / max(float(x.abs().max()), 1e-12)) if x.numel() else 0.0
        if r > worst:
            worst, name = r, k
    return worst, name


def per_step(a) -> int:
    K = a.per_step
    out = Path(tempfile.mkdtemp(prefix="tc1a_ps_"))
    # base trajectory (legacy) saving a checkpoint after every update; trial t resumes from update t-1
    if K > 1:
        _spawn("legacy", out / "base", a, updates=K - 1, ckpt_every=1, tag="base")
    from alphanl.transformer.checkpoint import checkpoint_path
    ok_all = True
    for t in range(K):
        src = None if t == 0 else checkpoint_path(out / "base" / "ckpt", t - 1)
        recs = {}
        for mode in ("legacy", "cached"):
            d = out / f"t{t}" / mode
            resume = None
            if src is not None:
                (d / "ckpt").mkdir(parents=True, exist_ok=True)
                resume = d / "ckpt" / Path(src).name
                shutil.copy(src, resume)
            recs[mode] = _spawn(mode, d, a, updates=t + 1, resume=resume, tag=mode)
        L, C = recs["legacy"], recs["cached"]
        la, lo = _actions(L); ca, co = _actions(C)
        n_act = sum(int(x.numel()) for x in la)
        act_diff = sum(int((x != y).sum()) for x, y in zip(la, ca)) if len(la) == len(ca) else n_act
        opp_diff = sum(1 for x, y in zip(lo, co) if x != y) + abs(len(lo) - len(co))
        rel, worst = _max_rel(L["params"][-1], C["params"][-1])
        ok = act_diff == 0 and opp_diff == 0 and len(la) == len(ca) == 1 and rel <= 1e-5
        ok_all &= ok
        print(f"[per-step] trial {t}: start={'fresh' if src is None else 'update_' + str(t - 1)} "
              f"learner actions={n_act} diff={act_diff} opp moves={len(co)} diff={opp_diff} "
              f"max_rel_param_err={rel:.3e} ({worst}) bitwise_params={not _eq(L['params'], C['params'])} "
              f"-> {'ok' if ok else 'FAIL'}")
    print(f"[per-step] {'PASS' if ok_all else 'FAIL'} trials={K} device={a.device} deterministic={a.deterministic}")
    return 0 if ok_all else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--child", choices=("legacy", "cached"))
    p.add_argument("--out", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--legacy-twice", action="store_true")
    p.add_argument("--per-step", type=int, default=0, metavar="K")
    a = p.parse_args()
    _setup_determinism(a.deterministic, a.device)
    if a.child:
        child(a.child, a.out, a.updates, a.device, a.resume, a.ckpt_every, a.tag)
        return 0
    if a.legacy_twice:
        return legacy_twice(a)
    if a.per_step:
        return per_step(a)
    frozen_test(a.device)
    out = Path(tempfile.mkdtemp(prefix="tc1a_"))
    # loaded with map_location=cpu so torch.equal never compares across devices
    L = _spawn("legacy", out / "legacy", a)
    C = _spawn("cached", out / "cached", a)
    bad = _compare(L, C)
    kinds = {}
    for k, _, _ in C["opp"]:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[equiv] updates={a.updates} device={a.device} opp moves={len(C['opp'])} by kind={kinds}")
    print(f"[equiv] rollout buffers={len(C['storage'])} param sets={len(C['params'])}")
    print(f"[equiv] legacy loads={L['loads']}  cached builds={C['loads']}  builds per snapshot id={C['builds']}  cache at end={C.get('cache_ids')}")
    assert not bad, f"MISMATCH: {bad[:20]}"
    assert all(v == 1 for v in C["builds"].values()), C["builds"]
    print("[equiv] PASS: storage, opponent actions, post-update params, optimizer, actor/torch/numpy RNG all bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())

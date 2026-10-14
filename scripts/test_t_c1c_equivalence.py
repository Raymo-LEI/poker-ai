"""T-C1c acceptance (a): per-env RNG determinism and batching-order independence.

    PYTHONPATH=src python scripts/test_t_c1c_equivalence.py [--updates 5] [--device cpu|cuda] --deterministic

Three fresh subprocesses, same seed, env.per_env_rng=true:
  batched   (PolicyActor.batch_rollout=True)
  batched2  (same again: same seed twice)
  unbatched (--no-batch-rollout path)
PASS iff every pair is bit-identical in: rollout buffers, opponent moves (kind, id, action),
post-update params, optimizer state, and every table's final deal/sample stream state.
Opponent logits batched vs single-row: max abs err <= 1e-5 (informational for actions).
Also checks per_env_rng sampling uses no global torch RNG during rollout (draws confined to streams).
On cuda, --deterministic needs CUBLAS_WORKSPACE_CONFIG=':4096:8' set BEFORE python starts.
"""
from __future__ import annotations

import argparse, copy, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import test_t_c1a_equivalence as A  # noqa: E402

TOL = 1e-5
MODES = ("batched", "batched2", "unbatched")


def child(mode: str, out: Path, updates: int, device: str) -> None:
    from alphanl.transformer import env as E, train as T
    E.PolicyActor.cache_opponents = True
    E.PolicyActor.batch_rollout = mode != "unbatched"
    rec = {"storage": [], "params": [], "opp": [], "logits": [], "prefetched": 0, "global_draws": 0}
    holder = {}

    orig_collect = T.collect_rollout
    def collect(envs, *a, **k):
        holder["envs"] = envs
        before = torch.get_rng_state()
        res = orig_collect(envs, *a, **k)
        if not torch.equal(before, torch.get_rng_state()):
            rec["global_draws"] += 1  # rollout touched the global torch RNG
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
    def act_batch(self, batch, choice, logits=None, streams=None):
        assert streams is not None, "per_env_rng run sampled without streams"
        if choice["kind"] != "random":
            if logits is None:
                model = self._model_for(choice)
                dev = next(model.parameters()).device
                with torch.no_grad():
                    lg, _ = model(E.batch_to_device(batch, dev), mask_illegal=True)
            else:
                lg = logits
                rec["prefetched"] += 1
            rec["logits"].append(lg.detach().float().cpu().clone())
        a = orig_act(self, batch, choice, logits=logits, streams=streams)
        rec["opp"].append((choice["kind"], choice.get("policy_id"), a))
        return a
    E.PolicyActor.act_batch = act_batch

    cfg = A._cfg(out, updates, device)
    cfg.env.per_env_rng = True
    T.run_training(cfg, resume=None)
    rec["streams"] = [env.streams.state_dict() for env in holder["envs"]]
    torch.save(rec, out / f"{mode}.pt")


def _spawn(mode, d: Path, a):
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--child", mode, "--out", str(d), "--updates", str(a.updates),
           "--device", a.device]
    if a.deterministic:
        cmd.append("--deterministic")
    subprocess.run(cmd, check=True, cwd=REPO)
    return torch.load(d / f"{mode}.pt", weights_only=False, map_location="cpu")


def _streams_eq(x, y) -> bool:
    if len(x) != len(y):
        return False
    for s, t in zip(x, y):
        if s["deal"] != t["deal"] or s["sample_np"] != t["sample_np"]:
            return False
        if not torch.equal(s["sample_torch"], t["sample_torch"]):
            return False
    return True


def compare(name, X, Y) -> bool:
    bit = {k: not A._eq(X[k], Y[k], f"{name}:{k}") for k in ("storage", "params", "opt")}
    bit["opp"] = X["opp"] == Y["opp"]
    bit["streams"] = _streams_eq(X["streams"], Y["streams"])
    if len(X["logits"]) == len(Y["logits"]) and X["logits"]:
        lerr = max(float((p - q).abs().max()) for p, q in zip(X["logits"], Y["logits"]))
    else:
        lerr = float("inf") if X["logits"] or Y["logits"] else 0.0
    rel, worst = A._max_rel(X["params"][-1], Y["params"][-1])
    ok = all(bit.values()) and lerr <= TOL
    print(f"[c1c] {name}: bit-identical {bit}  opp logits max abs err={lerr:.3e}  "
          f"max_rel_param_err={rel:.3e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--child", choices=MODES)
    p.add_argument("--out", type=Path)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    A._setup_determinism(a.deterministic, a.device)
    if a.child:
        child(a.child, a.out, a.updates, a.device)
        return 0
    out = Path(tempfile.mkdtemp(prefix="tc1c_"))
    R = {m: _spawn(m, out / m, a) for m in MODES}
    B = R["batched"]
    kinds = {}
    for k, _, _ in B["opp"]:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[c1c] updates={a.updates} device={a.device} deterministic={a.deterministic} per_env_rng=True")
    print(f"[c1c] learner actions={sum(int(s['actions'].numel()) for s in B['storage'])} "
          f"opp moves={len(B['opp'])} by kind={kinds}; served from batch: batched={B['prefetched']} "
          f"unbatched={R['unbatched']['prefetched']}")
    print(f"[c1c] rollouts touching global torch RNG: " + ", ".join(f"{m}={R[m]['global_draws']}" for m in MODES))
    ok = compare("seed-twice (batched vs batched2)", B, R["batched2"])
    ok &= compare("batched vs unbatched", B, R["unbatched"])
    ok &= all(R[m]["global_draws"] == 0 for m in MODES)
    ok &= B["prefetched"] > 0 and R["unbatched"]["prefetched"] == 0
    print(f"[c1c] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

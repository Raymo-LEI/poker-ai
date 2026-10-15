"""T-C1b acceptance: batched opponent forward vs T-C1a unbatched (per-move forward).

    PYTHONPATH=src python scripts/test_t_c1b_equivalence.py [--updates 5] [--device cpu|cuda] --deterministic

Both runs use the T-C1a opponent cache; ``batched`` sets PolicyActor.batch_rollout=True,
``unbatched`` sets it False. Each run happens in a fresh subprocess with the same seed and
records: every rollout buffer, every opponent move (kind, policy_id, action, logits row),
post-update params, optimizer state and the actor / torch / numpy RNG states.

PASS iff
  * learner actions and opponent actions are 100% identical (same count, same order),
  * opponent logits max abs err <= 1e-5 (batched forward vs single-row forward),
  * actor / torch(cpu,cuda) RNG states identical (same number and order of draws).
Also prints (informational) whether storage / params / optimizer are bit-identical and
the max relative param error, plus how many opponent moves were served from a batch.
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

import test_t_c1a_equivalence as A  # noqa: E402  (reuses _cfg, _eq, _max_rel, _setup_determinism)

TOL = 1e-5


def child(mode: str, out: Path, updates: int, device: str) -> None:
    from alphanl.transformer import env as E, train as T
    E.PolicyActor.cache_opponents = True
    E.PolicyActor.batch_rollout = (mode == "batched")
    rec = {"storage": [], "params": [], "opp": [], "logits": [], "prefetched": 0, "batches": 0}

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
    def act_batch(self, batch, choice, logits=None):
        if choice["kind"] != "random":
            if logits is None:
                # unbatched: recompute the single-row logits the policy_act call sees (no RNG use)
                model = self._model_for(choice)
                dev = next(model.parameters()).device
                with torch.no_grad():
                    lg, _ = model(E.batch_to_device(batch, dev), mask_illegal=True)
            else:
                lg = logits
                rec["prefetched"] += 1
            rec["logits"].append(lg.detach().float().cpu().clone())
        a = orig_act(self, batch, choice, logits=logits)
        rec["opp"].append((choice["kind"], choice.get("policy_id"), a))
        rec["actor_rng"] = self.rng_state()
        return a
    E.PolicyActor.act_batch = act_batch

    orig_pf = E.PolicyActor.prefetch
    def prefetch(self, envs):
        n = orig_pf(self, envs)
        rec["batches"] += 1
        return n
    E.PolicyActor.prefetch = prefetch

    T.run_training(A._cfg(out, updates, device), resume=None)
    rec["torch_rng"] = torch.get_rng_state()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        rec["cuda_rng"] = torch.cuda.get_rng_state_all()
    rec["np_rng"] = np.random.get_state()[1].copy()
    torch.save(rec, out / f"{mode}.pt")


def _spawn(mode, d: Path, a):
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--child", mode, "--out", str(d), "--updates", str(a.updates),
           "--device", a.device]
    if a.deterministic:
        cmd.append("--deterministic")
    subprocess.run(cmd, check=True, cwd=REPO)
    return torch.load(d / f"{mode}.pt", weights_only=False, map_location="cpu")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--child", choices=("batched", "unbatched"))
    p.add_argument("--out", type=Path)
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    A._setup_determinism(a.deterministic, a.device)
    if a.child:
        child(a.child, a.out, a.updates, a.device)
        return 0
    out = Path(tempfile.mkdtemp(prefix="tc1b_"))
    U = _spawn("unbatched", out / "unbatched", a)
    B = _spawn("batched", out / "batched", a)

    la = [s["actions"] for s in U["storage"]]; ba = [s["actions"] for s in B["storage"]]
    n_learn = sum(int(x.numel()) for x in la)
    learn_diff = sum(int((x != y).sum()) for x, y in zip(la, ba)) if len(la) == len(ba) else n_learn
    opp_diff = sum(1 for x, y in zip(U["opp"], B["opp"]) if x != y) + abs(len(U["opp"]) - len(B["opp"]))
    if len(U["logits"]) == len(B["logits"]) and U["logits"]:
        lerr = max(float((x - y).abs().max()) for x, y in zip(U["logits"], B["logits"]))
    else:
        lerr = float("inf")
    rng_bad = A._eq(U.get("actor_rng"), B.get("actor_rng"), "actor_rng") + A._eq(U["torch_rng"], B["torch_rng"], "torch_rng")
    if "cuda_rng" in U or "cuda_rng" in B:
        rng_bad += A._eq(U.get("cuda_rng"), B.get("cuda_rng"), "cuda_rng")
    rel, worst = A._max_rel(U["params"][-1], B["params"][-1])
    bit = {k: not A._eq(U[k], B[k], k) for k in ("storage", "params", "opt", "np_rng")}
    kinds = {}
    for k, _, _ in B["opp"]:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"[c1b] updates={a.updates} device={a.device} deterministic={a.deterministic}")
    print(f"[c1b] learner actions={n_learn} diff={learn_diff}  opp moves={len(B['opp'])} (unbatched {len(U['opp'])}) diff={opp_diff} by kind={kinds}")
    print(f"[c1b] model opp moves={len(B['logits'])}, served from batch={B['prefetched']} "
          f"({100.0 * B['prefetched'] / max(1, len(B['logits'])):.1f}%), batched calls={B['batches']}")
    print(f"[c1b] opp logits max abs err={lerr:.3e} (tol {TOL:g})  rng mismatches={rng_bad}")
    print(f"[c1b] bit-identical: {bit}  max_rel_param_err={rel:.3e} ({worst})")
    ok = learn_diff == 0 and opp_diff == 0 and lerr <= TOL and not rng_bad and U["prefetched"] == 0 and B["prefetched"] > 0
    print(f"[c1b] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

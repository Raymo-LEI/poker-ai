"""T-C1c acceptance (b): duplicate statistical comparison, per-env RNG vs global-RNG sampling.

    PYTHONPATH=src python scripts/eval_t_c1c_duplicate.py --deals 10000 [--tables 64] [--out res.json]

Fixed "checkpoint" (hero) and fixed opponent: two PokerTransformers built from fixed torch
seeds, policy head scaled by --head-scale so the policy is state-dependent rather than
near-uniform (no training run is needed or done). Same weights for both paths.

Each table has the T-C1c per-env streams (SeedSequence(master).spawn(tables)). For every deal
the DEAL stream draws one deck; the deal is played twice, hero in seat 0 then seat 1 (pokerkit
backend: fixed button, so the two orientations swap cards and position = duplicate). Both
paths use the same master seed, so they play the identical list of decks.
  new: every action sampled from the table's own SAMPLE stream (per_env_rng)
  old: every action sampled with the global torch RNG (Categorical(probs).sample(), table order)
Forwards are batched across tables in both paths (same logits).
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from alphanl.transformer import env as E, train as T  # noqa: E402
from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.model import masked_log_probs  # noqa: E402
from alphanl.transformer.pool import OpponentPool  # noqa: E402
from alphanl.transformer.tokenizer import BettingHistoryTokenizer, batch_to_device, cat_batches  # noqa: E402

STREETS = ("preflop", "flop", "turn", "river")


def _model(cfg, seed, scale):
    torch.manual_seed(seed)
    m = T.build_model(cfg)
    with torch.no_grad():
        m.policy_head.weight.mul_(scale)
    m.eval()
    return m


def run_path(path: str, a, cfg, hero, opp):
    torch.manual_seed(a.sample_seed)  # global RNG for the old path
    tok = BettingHistoryTokenizer(cfg.model.max_seq_len)
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=0.0, seed=0)
    actor = E.PolicyActor(hero, opp, tok, np.random.default_rng(0))
    ss = np.random.SeedSequence(a.master_seed).spawn(a.tables)
    envs = [E.SelfPlayEnv(tok, actor, pool, seed=1000 + i, backend="pokerkit", record_results=False,
                          seed_seq=ss[i], per_env_rng=(path == "new"), shared_deal_stream=True)
            for i in range(a.tables)]
    deals_per_table = math.ceil(a.deals / a.tables)
    # per hand records
    payoff = np.zeros((a.tables, deals_per_table, 2))
    counts = np.zeros((4, 5), dtype=np.int64)  # hero actions by street
    opp_counts = np.zeros((4, 5), dtype=np.int64)
    state = [{"d": 0, "o": 0, "deck": None} for _ in envs]

    def start(i):
        env, st = envs[i], state[i]
        if st["o"] == 0:
            st["deck"] = env.streams.draw_deck()  # DEAL stream: one deck per deal
        env.table.reset(st["deck"])
        env.hero = st["o"]
        env.choice = {"kind": "self", "policy_id": None, "weights": None}

    active = []
    for i in range(a.tables):
        start(i); active.append(i)
    while active:
        hero_rows, opp_rows = [], []
        for i in active:
            env = envs[i]
            seat = int(env.table.current_player())
            (hero_rows if seat == env.hero else opp_rows).append((i, seat, env._observe_seat(seat)))
        sampled = {}
        for rows, model in ((hero_rows, hero), (opp_rows, opp)):
            if not rows:
                continue
            b = cat_batches([r[2] for r in rows])
            with torch.no_grad():
                logits, _ = model(b, mask_illegal=True)
            _, _, probs, _ = masked_log_probs(logits, b["legal_mask"])
            for k, (i, seat, _) in enumerate(rows):
                sampled[i] = (probs[k], seat)
        nxt = []
        for i in active:  # sample in table order
            env, st = envs[i], state[i]
            pr, seat = sampled[i]
            if path == "new":
                act = env.streams.sample_action(pr)
            else:
                act = int(torch.distributions.Categorical(probs=pr.unsqueeze(0)).sample().item())
            street = int(env.table.street)
            (counts if seat == env.hero else opp_counts)[min(street, 3), act] += 1
            env._push_and_step(act)
            if env.table.is_over():
                payoff[i, st["d"], st["o"]] = env._hero_bb()
                if st["o"] == 0:
                    st["o"] = 1
                else:
                    st["o"] = 0; st["d"] += 1
                if st["d"] < deals_per_table:
                    start(i); nxt.append(i)
            else:
                nxt.append(i)
        active = nxt
    return payoff.reshape(-1, 2)[: a.deals], counts, opp_counts


def ci_mean(x):
    x = np.asarray(x, float); m = x.mean(); h = 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - h, m + h


def chi2(c1, c2):
    tab = np.stack([c1, c2]).astype(float)
    keep = tab.sum(0) > 0
    tab = tab[:, keep]
    if tab.shape[1] < 2 or (tab.sum(1) == 0).any():
        return 0.0, 0, 1.0
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    x = float(((tab - exp) ** 2 / exp).sum()); df = tab.shape[1] - 1
    p = float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64), torch.tensor(x / 2.0, dtype=torch.float64)))
    return x, df, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=10000)
    p.add_argument("--tables", type=int, default=64)
    p.add_argument("--master-seed", type=int, default=12345)
    p.add_argument("--sample-seed", type=int, default=777)
    p.add_argument("--hero-seed", type=int, default=100)
    p.add_argument("--opp-seed", type=int, default=200)
    p.add_argument("--head-scale", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = ExperimentConfig.from_json(REPO / "configs/transformer/laptop_8gb.json")
    hero, opp = _model(cfg, a.hero_seed, a.head_scale), _model(cfg, a.opp_seed, a.head_scale)
    res, t = {}, {}
    for path in ("new", "old"):
        t0 = time.time(); res[path] = run_path(path, a, cfg, hero, opp); t[path] = time.time() - t0
        print(f"[dup] {path}: {a.deals} deals x 2 seats = {2 * a.deals} hands in {t[path]:.0f}s", flush=True)
    out = {"deals": a.deals, "hands_per_path": 2 * a.deals, "tables": a.tables, "seconds": t, "args": vars(a) | {"out": str(a.out)}}
    (Pn, Cn, On), (Po, Co, Oo) = res["new"], res["old"]
    lines = []
    for name, P in (("new", Pn), ("old", Po)):
        dup = P.mean(1); m, lo, hi = ci_mean(dup)
        hands = P.reshape(-1); w = (hands > 0).mean(); wh = 1.96 * math.sqrt(w * (1 - w) / len(hands))
        out[name] = {"bb_per_hand_dup": [m, lo, hi], "win_rate": [w, w - wh, w + wh],
                     "hero_action_counts": Cn.tolist() if name == "new" else Co.tolist()}
        lines.append(f"{name}: mean payoff {m:+.4f} bb/hand [95% CI {lo:+.4f}, {hi:+.4f}] (duplicate, n={len(dup)} deals); "
                     f"win rate {w:.4f} [{w - wh:.4f}, {w + wh:.4f}] (n={len(hands)} hands)")
    d = Pn.mean(1) - Po.mean(1); m, lo, hi = ci_mean(d)
    out["diff_bb_paired"] = [m, lo, hi]
    sig_pay = not (lo <= 0 <= hi)
    lines.append(f"new-old payoff (paired by deal): {m:+.4f} bb/hand [95% CI {lo:+.4f}, {hi:+.4f}] -> {'SIGNIFICANT' if sig_pay else 'not significant'}")
    wn, wo = (Pn.reshape(-1) > 0), (Po.reshape(-1) > 0)
    dw = wn.astype(float) - wo.astype(float); m, lo, hi = ci_mean(dw)
    sig_win = not (lo <= 0 <= hi)
    out["diff_win_paired"] = [m, lo, hi]
    lines.append(f"new-old win rate (paired by hand): {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] -> {'SIGNIFICANT' if sig_win else 'not significant'}")
    sig_act = []
    out["chi2"] = {}
    for who, A_, B_ in (("hero", Cn, Co), ("opp", On, Oo)):
        for s in range(4):
            x, df, pv = chi2(A_[s], B_[s])
            out["chi2"][f"{who}_{STREETS[s]}"] = [x, df, pv]
            na, nb = A_[s].sum(), B_[s].sum()
            props = " ".join(f"a{k}:{A_[s,k]/max(na,1):.3f}/{B_[s,k]/max(nb,1):.3f}" for k in range(5))
            lines.append(f"{who} {STREETS[s]:7s} n={na}/{nb} chi2={x:.2f} df={df} p={pv:.3f}  new/old {props}")
            sig_act.append(pv < 0.05)
    nsig = sum(sig_act)
    lines.append(f"action-distribution tests at p<0.05: {nsig}/{len(sig_act)} significant "
                 f"(expected ~{0.05 * len(sig_act):.1f} by chance; Bonferroni threshold p<{0.05 / len(sig_act):.4f}: "
                 f"{sum(v[2] < 0.05 / len(sig_act) for v in out['chi2'].values())} significant)")
    ok = (not sig_pay) and (not sig_win) and all(v[2] >= 0.05 / len(sig_act) for v in out["chi2"].values())
    lines.append(f"RESULT: {'PASS - no significant difference' if ok else 'FAIL - significant difference'}")
    out["pass"] = ok
    out["summary"] = lines
    print("\n".join(lines))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

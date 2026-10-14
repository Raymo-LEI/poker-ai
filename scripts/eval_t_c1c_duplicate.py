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


def run_path(path: str, a, cfg, hero, opp, master_seed=None, deals=None):
    master_seed = a.master_seed if master_seed is None else master_seed
    deals = a.deals if deals is None else deals
    torch.manual_seed(a.sample_seed + master_seed)  # global RNG for the old path
    tok = BettingHistoryTokenizer(cfg.model.max_seq_len)
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=0.0, seed=0)
    actor = E.PolicyActor(hero, opp, tok, np.random.default_rng(0))
    ss = np.random.SeedSequence(master_seed).spawn(a.tables)
    envs = [E.SelfPlayEnv(tok, actor, pool, seed=1000 + i, backend="pokerkit", record_results=False,
                          seed_seq=ss[i], per_env_rng=(path == "new"), shared_deal_stream=True)
            for i in range(a.tables)]
    deals_per_table = math.ceil(deals / a.tables)
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
    return payoff.reshape(-1, 2)[: deals], counts, opp_counts


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


def _load(cfg, path, seed, scale):
    if path is None:
        return _model(cfg, seed, scale)
    m = T.build_model(cfg)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    m.eval()
    return m


def paired_stats(Pn, Po, boot=2000, seed=0, target=0.1):
    """Pairing unit = same deck + same seat: diff_ds = new(d,s) - old(d,s)."""
    d = (Pn - Po).reshape(-1)
    n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1))
    hw = 1.96 * sd / math.sqrt(n)  # t ~ normal for n in the thousands
    rng = np.random.default_rng(seed)
    bm = np.array([d[rng.integers(0, n, n)].mean() for _ in range(boot)])
    per_deal = (Pn - Po).mean(1)
    return {
        "hands": n, "mean_diff_bb": m, "sd_hand": sd, "ci_t": [m - hw, m + hw], "halfwidth": hw,
        "ci_boot": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
        "var_hand_diff": sd ** 2, "var_per_deal_diff": float(per_deal.var(ddof=1)),
        "var_new": float(Pn.reshape(-1).var(ddof=1)), "var_old": float(Po.reshape(-1).var(ddof=1)),
        "corr_new_old": float(np.corrcoef(Pn.reshape(-1), Po.reshape(-1))[0, 1]),
        "hands_needed_for_target": int(math.ceil((1.96 * sd / target) ** 2)),
    }


class _NoLoadActor:
    """Deal check only: the pool pick is recorded, no opponent model is built."""
    def prepare_pool(self, weights):
        pass


def _deal_records(path: str, n: int, tables: int, master_seed: int):
    tok = BettingHistoryTokenizer(64)
    pool = OpponentPool(max_size=4, selfplay_prob=0.25, random_prob=0.1, seed=0)
    w = {"x": torch.zeros(1)}
    for _ in range(4):
        pool.add(w)
    for pid, wins in zip(list(pool._order), (1, 5, 9, 3)):
        for k in range(10):
            pool.record(pid, 1.0 if k < wins else -1.0)  # uneven win rates -> pfsp picks vary
    ss = np.random.SeedSequence(master_seed).spawn(tables)
    envs = [E.SelfPlayEnv(tok, _NoLoadActor(), pool, seed=1000 + i, backend="pokerkit", record_results=False,
                          seed_seq=ss[i], per_env_rng=(path == "new"), shared_deal_stream=True) for i in range(tables)]
    recs = []
    for d in range(n):
        env = envs[d % tables]
        env._deal(None)  # real deal code: deck, hero seat, pool pick from the DEAL stream
        st = env.table.state
        rest = [E_card(c) for c in st.deck_cards]  # undealt deck after hole cards
        board = tuple(rest[1:4] + rest[5:6] + rest[7:8])  # burn, flop, burn, turn, burn, river
        holes = (env.table.hole_cards(0), env.table.hole_cards(1))
        recs.append((d, d % tables, holes, board, int(env.hero), env.choice["kind"], env.choice.get("policy_id")))
    return recs


def E_card(c):
    from alphanl.transformer.tables import _card_text
    return _card_text(c)


def check_deals(n: int, tables: int, master_seed: int) -> int:
    t0 = time.time()
    A_, B_ = _deal_records("new", n, tables, master_seed), _deal_records("old", n, tables, master_seed)
    bad = []
    for ra, rb in zip(A_, B_):
        d, tab, holes_a, board_a, hero_a, kind_a, pid_a = ra
        _, _, holes_b, board_b, hero_b, kind_b, pid_b = rb
        for s in (0, 1):  # both seat orientations: hero in seat s, opponent in 1-s
            for what, x, y in (("hero hole", holes_a[s], holes_b[s]), ("opp hole", holes_a[1 - s], holes_b[1 - s])):
                if x != y:
                    bad.append(f"deal {d} table {tab} seat {s}: {what} new={x} old={y}")
        if board_a != board_b:
            bad.append(f"deal {d} table {tab}: board new={board_a} old={board_b}")
        if len(board_a) != 5 or len(set(board_a) | set(holes_a[0]) | set(holes_a[1])) != 9:
            bad.append(f"deal {d} table {tab}: malformed cards {holes_a} {board_a}")
        if hero_a != hero_b:
            bad.append(f"deal {d} table {tab}: hero seat new={hero_a} old={hero_b}")
        if (kind_a, pid_a) != (kind_b, pid_b):
            bad.append(f"deal {d} table {tab}: opponent pick new={(kind_a, pid_a)} old={(kind_b, pid_b)}")
    picks = {}
    for r in A_:
        picks[(r[5], r[6])] = picks.get((r[5], r[6]), 0) + 1
    dt = time.time() - t0
    if bad:
        print("\n".join(bad[:20]))
        print(f"[check-deals] FAIL n={n} mismatches={len(bad)} ({dt:.1f}s)")
        return 1
    print(f"[check-deals] opponent picks: {dict(sorted(picks.items(), key=str))}; hero seat 1 share "
          f"{sum(r[4] for r in A_) / n:.3f}")
    print(f"[check-deals] PASS n={n} (hole cards x2 seats, 5 board cards, hero seat, opponent pick identical; {dt:.1f}s)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=10000, help="deals per batch (x2 seats = hands)")
    p.add_argument("--tables", type=int, default=64)
    p.add_argument("--master-seed", type=int, default=12345)
    p.add_argument("--sample-seed", type=int, default=777)
    p.add_argument("--hero-seed", type=int, default=100)
    p.add_argument("--opp-seed", type=int, default=200)
    p.add_argument("--head-scale", type=float, default=5.0)
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=Path, default=None,
                   help="hero checkpoint (repo .pt with 'model', e.g. runs/.../ckpt/update_0000019.pt, or a state_dict)")
    p.add_argument("--opponent", type=Path, default=None, help="fixed opponent checkpoint (default: same as --ckpt)")
    p.add_argument("--config", type=Path, default=REPO / "configs/transformer/laptop_8gb.json")
    p.add_argument("--target-halfwidth", type=float, default=None,
                   help="keep adding --deals batches until paired 95%% half-width <= this (bb/hand)")
    p.add_argument("--max-hands", type=int, default=200000, help="cap per side for --target-halfwidth")
    p.add_argument("--accept", type=float, default=0.1, help="strict half-width criterion")
    p.add_argument("--relaxed", type=float, default=0.2, help="fallback half-width if cap reached")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--check-deals", type=int, default=None,
                   help="only verify N deals are identical between new and old paths (no play), then exit")
    a = p.parse_args()
    if a.check_deals:
        sys.exit(check_deals(a.check_deals, a.tables, a.master_seed))
    cfg = ExperimentConfig.from_json(a.config)
    if a.opponent is None:
        a.opponent = a.ckpt
    hero = _load(cfg, a.ckpt, a.hero_seed, a.head_scale)
    opp = _load(cfg, a.opponent, a.opp_seed, a.head_scale)
    Pn, Po, Cn, Co, On, Oo = [], [], 0, 0, 0, 0
    t = {"new": 0.0, "old": 0.0}
    batch = 0
    while True:
        seed = a.master_seed + 1000003 * batch
        for path in ("new", "old"):
            t0 = time.time(); P, C, O = run_path(path, a, cfg, hero, opp, master_seed=seed); t[path] += time.time() - t0
            if path == "new":
                Pn.append(P); Cn = Cn + C; On = On + O
            else:
                Po.append(P); Co = Co + C; Oo = Oo + O
        batch += 1
        st = paired_stats(np.concatenate(Pn), np.concatenate(Po), target=a.accept)
        print(f"[dup] batch {batch}: {st['hands']} hands/side, paired diff {st['mean_diff_bb']:+.4f} "
              f"+-{st['halfwidth']:.4f} bb/hand ({t['new']:.0f}s new, {t['old']:.0f}s old)", flush=True)
        if a.target_halfwidth is None or st["halfwidth"] <= a.target_halfwidth:
            break
        if st["hands"] + 2 * a.deals > a.max_hands:
            print(f"[dup] cap {a.max_hands} hands/side reached", flush=True)
            break
    Pn, Po = np.concatenate(Pn), np.concatenate(Po)
    st = paired_stats(Pn, Po, target=a.accept)
    lo, hi = st["ci_t"]; has0 = lo <= 0 <= hi
    lines = [
        f"hands/side={st['hands']} (deals={len(Pn)} x 2 seats), pairing unit = same deck + same seat",
        f"paired diff new-old: {st['mean_diff_bb']:+.4f} bb/hand, 95% CI t [{lo:+.4f}, {hi:+.4f}] (half-width {st['halfwidth']:.4f}), "
        f"bootstrap [{st['ci_boot'][0]:+.4f}, {st['ci_boot'][1]:+.4f}]",
        f"variance: per-hand diff {st['var_hand_diff']:.2f} bb^2 (sd {st['sd_hand']:.2f}), per-deal diff (mean of 2 seats) {st['var_per_deal_diff']:.2f}, "
        f"new {st['var_new']:.2f}, old {st['var_old']:.2f}; corr(new, old) = {st['corr_new_old']:.4f}",
        f"hands needed for half-width <= {a.accept}: {st['hands_needed_for_target']}",
    ]
    for name, P in (("new", Pn), ("old", Po)):
        h = P.reshape(-1); w = (h > 0).mean(); wh = 1.96 * math.sqrt(w * (1 - w) / len(h))
        m, l2, h2 = ci_mean(h)
        lines.append(f"{name}: payoff {m:+.4f} bb/hand [{l2:+.4f}, {h2:+.4f}], win rate {w:.4f} [{w - wh:.4f}, {w + wh:.4f}]")
    dw = (Pn.reshape(-1) > 0).astype(float) - (Po.reshape(-1) > 0).astype(float); m, l2, h2 = ci_mean(dw)
    lines.append(f"win rate diff (paired by deck+seat): {m:+.4f} [{l2:+.4f}, {h2:+.4f}]")
    chi = {}
    for who, A_, B_ in (("hero", Cn, Co), ("opp", On, Oo)):
        for s_ in range(4):
            x, df, pv = chi2(A_[s_], B_[s_]); chi[f"{who}_{STREETS[s_]}"] = [x, df, pv]
            lines.append(f"{who} {STREETS[s_]:7s} n={A_[s_].sum()}/{B_[s_].sum()} chi2={x:.2f} df={df} p={pv:.3f}")
    nsig = sum(v[2] < 0.05 / len(chi) for v in chi.values())
    lines.append(f"action chi2: {sum(v[2] < 0.05 for v in chi.values())}/{len(chi)} p<0.05, {nsig} below Bonferroni {0.05 / len(chi):.4f}")
    if st["halfwidth"] <= a.accept:
        crit, ok = f"strict (half-width <= {a.accept})", has0
    elif a.target_halfwidth is not None and st["hands"] + 2 * a.deals > a.max_hands:
        crit, ok = f"relaxed (cap {a.max_hands} reached; half-width <= {a.relaxed})", has0 and st["halfwidth"] <= a.relaxed
    else:
        crit, ok = f"strict (half-width <= {a.accept}) - NOT MET, run with --target-halfwidth {a.accept}", False
    verdict = "FAIL" if not ok else ("PASS(relaxed)" if crit.startswith("relaxed") else "PASS")
    lines.append(f"criterion: {crit}; CI contains 0: {has0}; action dists not significant: {nsig == 0} -> {verdict}")
    print("\n".join(lines))
    if a.out:
        a.out.write_text(json.dumps({"stats": st, "chi2": chi, "criterion": crit, "verdict": verdict,
                                     "seconds": t, "summary": lines, "args": {k: str(v) for k, v in vars(a).items()}},
                                    indent=1, default=float))


if __name__ == "__main__":
    main()

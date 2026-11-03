"""Public-belief subgame search. NumPy only.

One street, both ranges as equity-quantile buckets, warm-started discounted CFR,
then a KL fuse back into the frozen blueprint. Exact chip amounts are the root
of the subgame, so an off-tree bet is not rounded into an action bucket.
"""
import hashlib
import time
from math import erf

import numpy as np

B = 12
HANDS = 12
RUNOUTS = 1
DCFR_ITERS = 24
BETA = 0.5
DECK = tuple(suit + rank for suit in "SHDC" for rank in "23456789TJQKA")
HUD = (
    ("vpip", (0.45, 6.0)),
    ("pfr", (0.25, 6.0)),
    ("three_bet", (0.10, 10.0)),
    ("fold_to_preflop_raise", (0.5, 6.0)),
    ("flop_aggression", (0.35, 6.0)),
    ("turn_aggression", (0.35, 6.0)),
    ("river_aggression", (0.35, 6.0)),
    ("fold_to_postflop_bet", (0.5, 6.0)),
    ("all_in_frequency", (0.05, 10.0)),
    ("showdown_win", (0.5, 6.0)),
)
EVALUATE = None


def bind(evaluate_cards):
    global EVALUATE
    EVALUATE = evaluate_cards


def _evaluate():
    if EVALUATE is not None:
        return EVALUATE
    from .portable_evaluator import evaluate_cards
    return evaluate_cards


def _gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def _norm(card):
    text = str(card).strip().upper()
    if text[0] in "SHDC" and text[1] in "23456789TJQKA":
        return text
    return text[1] + text[0]


def _evcard(card):
    card = _norm(card)
    return card[1] + card[0].lower()


def _rng(snapshot):
    key = repr((tuple(snapshot.hole_cards), tuple(snapshot.board_cards))).encode()
    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


def _hud(context, name, prior):
    if not hasattr(context, "successes") or name not in context.successes:
        return prior[0]
    mean, count = prior
    return (context.successes[name] + mean * count) / (context.opportunities[name] + count)


def _trap(context):
    if not hasattr(context, "trap_successes"):
        return 0.15
    successes = context.trap_successes.get("check_raise_freq", 0.0)
    chances = context.trap_opportunities.get("check_raise_freq", 0)
    return (successes + 0.15 * 2.0) / (chances + 2.0)


def _texture(board):
    if len(board) < 3:
        return np.zeros(4, dtype=np.float64)
    ranks = ["23456789TJQKA".index(_norm(card)[1]) for card in board]
    suits = [_norm(card)[0] for card in board]
    paired = float(len(ranks) != len(set(ranks)))
    flush = float(max(suits.count(suit) for suit in set(suits)) >= 3)
    return np.array([paired, flush, (max(ranks) - min(ranks)) / 12.0, max(ranks) / 12.0], dtype=np.float64)


def _context_vector(context):
    progress = getattr(context, "hands_played", 0) / 50.0
    bank = 2.0 * getattr(context, "hero_net_bb", 0.0) / 200.0
    return np.array([progress, bank] + [_hud(context, name, prior) for name, prior in HUD], dtype=np.float64)


def _win(evaluate, hero, opp, board):
    left = evaluate(*(hero + board))
    right = evaluate(*(opp + board))
    if left < right:
        return 1.0
    if left == right:
        return 0.5
    return 0.0


def _active(snapshot):
    return snapshot.street > 0 and (
        float(snapshot.pot) >= 6.0 * float(snapshot.big_blind) or float(snapshot.to_call) > 0
    )


def _chips(action, snapshot, amounts):
    """Exact chips for this abstract action, including an off-tree facing bet."""
    if amounts is not None and action in amounts:
        return float(amounts[action])
    owed = float(snapshot.to_call)
    stack = float(snapshot.me_stack)
    pot = float(snapshot.pot)
    if action == 0:
        return 0.0
    if action == 1:
        return min(owed, stack)
    if action == 4:
        return stack
    extra = max(pot * (0.5 if action == 2 else 1.0), owed)
    return min(stack, extra)


def _legal(snapshot, probs):
    mask = np.zeros(5, dtype=np.float64)
    for action in snapshot.legal_action_ids:
        if probs[action] > 1e-8:
            mask[action] = 1.0
    if mask.sum() <= 0:
        mask[:] = 0.0
        mask[int(np.argmax(probs))] = 1.0
    prior = np.where(mask > 0, np.asarray(probs, dtype=np.float64), 0.0)
    prior /= prior.sum()
    return mask, prior


def _sample_hands(snapshot):
    """Pairwise equity. Runout cards may sit in other sampled hands."""
    rng = _rng(snapshot)
    board_cards = [_norm(card) for card in snapshot.board_cards]
    hero0 = [_norm(card) for card in snapshot.hole_cards]
    deck = [card for card in DECK if card not in set(board_cards + hero0)]
    rng.shuffle(deck)
    heroes = [hero0]
    cursor = 0
    for _ in range(HANDS - 1):
        heroes.append(deck[cursor:cursor + 2])
        cursor += 2
    opps = []
    for _ in range(HANDS):
        opps.append(deck[cursor:cursor + 2])
        cursor += 2
    board = [_evcard(card) for card in board_cards]
    need = 5 - len(board)
    evaluate = _evaluate()
    wins = np.zeros((HANDS, HANDS), dtype=np.float64)
    hero_ev = [[_evcard(card) for card in hand] for hand in heroes]
    opp_ev = [[_evcard(card) for card in hand] for hand in opps]
    for i in range(HANDS):
        for j in range(HANDS):
            if need == 0:
                runs = [board]
            else:
                blocked = set(heroes[i] + opps[j] + board_cards)
                pool = [card for card in DECK if card not in blocked]
                runs = []
                for _run in range(RUNOUTS):
                    draw = rng.choice(len(pool), size=need, replace=False)
                    runs.append(board + [_evcard(pool[int(k)]) for k in draw])
            wins[i, j] = sum(_win(evaluate, hero_ev[i], opp_ev[j], run) for run in runs) / len(runs)
    return wins


def _likelihood(kind, fraction, strength, fold_rate, aggro, trap, prior):
    if kind == "fold":
        base = prior[0]
        tilt = 1.35 - strength
    elif kind == "check":
        base = prior[1]
        tilt = 1.15 - 0.45 * strength + trap * strength
    elif kind == "call":
        base = max(prior[1], 0.15)
        tilt = 0.85 + 0.2 * (1.0 - np.abs(strength - 0.45))
    else:
        base = prior[2] + prior[3] + prior[4]
        big = fraction >= 0.75 or kind == "all_in"
        tilt = (0.25 + 1.15 * strength) * (1.15 if big else 1.0)
        tilt = tilt + 0.15 * (strength < 0.22) * aggro * (0.4 if big else 1.0)
        tilt = tilt + (fold_rate - 0.5) * 0.1
    return np.clip(base * tilt + 0.02, 0.04, 0.96)


def _posterior(snapshot, wins, axis, prior, fold_rate, aggro, trap):
    strength = wins.mean(axis=axis)
    if axis == 0:
        strength = 1.0 - strength
    post = np.ones(HANDS, dtype=np.float64)
    mine = axis == 1
    events = [event for event in snapshot.events if event.street > 0 and (event.actor_is_me == mine)]
    for event in events:
        post *= _likelihood(event.action_type, float(event.pot_fraction), strength, fold_rate, aggro, trap, prior)
    post = np.clip(post, 1e-6, None)
    post /= post.sum()
    return strength, post


def _buckets(strength, posterior, wins, hero_axis):
    order = np.argsort(strength)
    mass = posterior[order]
    mass = mass / mass.sum()
    if hero_axis:
        matrix = wins[order][:, :]
    else:
        matrix = wins[:, order]
    bucket = int(np.where(order == 0)[0][0]) if hero_axis else None
    return mass, matrix, bucket, order


def build_pbs(snapshot, context, prior):
    """Both ranges, the showdown matrix, and the acting hand's bucket."""
    wins = _sample_hands(snapshot)
    fold_rate = _hud(context, "fold_to_postflop_bet", (0.5, 6.0))
    aggro = float(np.mean([
        _hud(context, "flop_aggression", (0.35, 6.0)),
        _hud(context, "turn_aggression", (0.35, 6.0)),
        _hud(context, "river_aggression", (0.35, 6.0)),
    ]))
    trap = _trap(context)
    hero_strength, hero_post = _posterior(snapshot, wins, 1, prior, fold_rate, aggro, trap)
    opp_strength, opp_post = _posterior(snapshot, wins, 0, prior, fold_rate, aggro, trap)
    r_hero, hero_rows, hero_bucket, _hero_order = _buckets(hero_strength, hero_post, wins, True)
    r_opp, _opp_cols, _none, opp_order = _buckets(opp_strength, opp_post, wins, False)
    matrix = hero_rows[:, opp_order]
    call = min(float(snapshot.to_call), float(snapshot.me_stack))
    pot = max(float(snapshot.pot), 1.0)
    blind = float(snapshot.big_blind)
    odds = call / (pot + call)
    equity = float(np.dot(r_opp, matrix[hero_bucket]))
    spr = min(float(snapshot.me_stack), float(snapshot.opp_stack)) / pot
    feat = np.concatenate([
        np.array([
            equity, equity - odds, pot / (blind * 200.0), min(spr, 20.0) / 20.0,
            call / pot, snapshot.street / 3.0,
            1.0 if snapshot.to_call > 0 else 0.0,
            min(float(snapshot.me_stack), float(snapshot.opp_stack)) / (blind * 200.0),
        ], dtype=np.float64),
        _texture(snapshot.board_cards),
        r_hero, r_opp, _context_vector(context),
    ])
    if feat.size != 48:
        raise RuntimeError("PBS feature width is %d" % feat.size)
    return {
        "r_hero": r_hero, "r_opp": r_opp, "matrix": matrix, "hero_bucket": hero_bucket,
        "feat": feat, "equity": equity, "fold_rate": fold_rate, "aggro": aggro, "trap": trap,
    }


def observe(snapshot, context):
    pbs = build_pbs(snapshot, context, np.full(5, 0.2))
    return pbs["feat"], pbs["hero_bucket"]


def leaf_vector(weights, feat):
    key = "vleaf.0.weight"
    if key not in weights or weights[key].shape[0] == 0:
        return None
    if weights[key].shape[1] != feat.size or weights["vleaf.1.weight"].shape[0] != B:
        return None
    x = feat.astype(np.float32)
    x = _gelu(x @ weights[key].T + weights["vleaf.0.bias"])
    return (x @ weights["vleaf.1.weight"].T + weights["vleaf.1.bias"]).astype(np.float64)


def _showdown(pot, invest, equity, blind):
    return (equity * (pot + invest) - invest) / blind


def _payoffs(snapshot, pbs, amounts, leaf):
    blind = float(snapshot.big_blind)
    pot = float(snapshot.pot)
    owed = float(snapshot.to_call)
    me = float(snapshot.me_stack)
    opp = float(snapshot.opp_stack)
    street = int(snapshot.street)
    matrix = pbs["matrix"]
    pay = np.zeros((5, B, B, 3), dtype=np.float64)
    opp0 = np.zeros((5, B, 3), dtype=np.float64)
    strength = (np.arange(B) + 0.5) / B
    for action in range(5):
        invest = _chips(action, snapshot, amounts)
        matched = min(invest, opp)
        ends = street >= 3 or invest >= me - 1.0 or matched >= opp - 1.0
        def value(equity):
            realized = _showdown(pot + matched, invest, equity, blind)
            if ends or leaf is None:
                return realized
            return leaf
        if action == 0:
            opp0[action, :, 1] = 1.0
            continue
        if action == 1 and owed > 0:
            settled = value(matrix)
            if np.ndim(leaf) and not ends:
                settled = np.broadcast_to(leaf[:, None], matrix.shape).copy()
            pay[action, :, :, :] = settled[:, :, None]
            opp0[action, :, 1] = 1.0
            continue
        if action == 1:
            back = matrix * pot / blind if street >= 3 or leaf is None else np.broadcast_to(leaf[:, None], matrix.shape)
            their = min(opp, max(pot * 0.66, blind))
            respond = np.maximum(_showdown(pot + their, their, matrix, blind), 0.0)
            pay[action, :, :, 1] = back
            pay[action, :, :, 2] = respond
            bet_p = np.clip(0.10 + 0.72 * strength + 0.08 * pbs["aggro"] * (strength < 0.25), 0.05, 0.9)
            opp0[action, :, 2] = bet_p
            opp0[action, :, 1] = 1.0 - bet_p
            continue
        pay[action, :, :, 0] = pot / blind
        called = value(matrix)
        if np.ndim(leaf) and not ends:
            called = 0.5 * _showdown(pot + matched, invest, matrix, blind) + 0.5 * leaf[:, None]
        pay[action, :, :, 1] = called
        extra = max(0.0, opp - invest)
        shove = _showdown(pot + opp, invest + extra, matrix, blind)
        pay[action, :, :, 2] = np.maximum(shove, -invest / blind)
        big = invest >= 0.75 * max(pot, blind)
        fold_p = np.clip(0.58 - 0.55 * strength + 0.16 * (pbs["fold_rate"] - 0.5) + (0.10 if big else 0.0), 0.04, 0.9)
        raise_p = np.clip(0.03 + 0.40 * strength * strength + pbs["trap"] * 0.18 * strength, 0.02, 0.55)
        over = np.maximum(0.0, fold_p + raise_p - 0.96)
        fold_p = fold_p - over * 0.5
        raise_p = raise_p - over * 0.5
        opp0[action, :, 0] = fold_p
        opp0[action, :, 2] = raise_p
        opp0[action, :, 1] = 1.0 - fold_p - raise_p
    return pay, opp0


def _dcfr(pay, r_hero, r_opp, legal, prior, opp0):
    hero = np.broadcast_to(prior, (B, 5)).copy()
    opp = opp0.copy()
    hero_regret = np.zeros((B, 5), dtype=np.float64)
    opp_regret = np.zeros_like(pay[0])  # (B, 3) is wrong
    opp_regret = np.zeros_like(opp)
    hero_sum = hero.copy()
    opp_sum = opp.copy()
    for step in range(1, DCFR_ITERS + 1):
        pos_w = (step ** 1.5) / (step ** 1.5 + 1.0)
        strat_w = float(step ** 2)
        cfv = np.einsum("o,aor,ahor->ha", r_opp, opp, pay)
        base = np.sum(hero * cfv, axis=1, keepdims=True)
        instant = cfv - base
        hero_regret = np.where(hero_regret > 0.0, hero_regret * pos_w, 0.0) + instant
        hero_regret *= legal
        hero_regret = np.maximum(hero_regret, 0.0)
        total = hero_regret.sum(axis=1, keepdims=True)
        hero = np.where(total > 1e-8, hero_regret / np.maximum(total, 1e-8), prior)
        opp_cfv = np.einsum("h,ha,ahor->aor", r_hero, hero, -pay)
        opp_base = np.sum(opp * opp_cfv, axis=-1, keepdims=True)
        opp_regret = np.where(opp_regret > 0.0, opp_regret * pos_w, 0.0) + (opp_cfv - opp_base)
        opp_regret = np.maximum(opp_regret, 0.0)
        denom = opp_regret.sum(axis=-1, keepdims=True)
        opp = np.where(denom > 1e-8, opp_regret / np.maximum(denom, 1e-8), opp0)
        hero_sum += strat_w * hero
        opp_sum += strat_w * opp
    hero_avg = hero_sum / hero_sum.sum(axis=1, keepdims=True)
    opp_avg = opp_sum / np.maximum(opp_sum.sum(axis=-1, keepdims=True), 1e-8)
    return np.einsum("o,aor,ahor->ha", r_opp, opp_avg, pay), hero_avg


def _beta(weights):
    if "wizard_beta" in weights:
        return float(np.asarray(weights["wizard_beta"]).reshape(-1)[0])
    return float(BETA)


def fuse(agent, snapshot, context, probs, amounts):
    started = time.perf_counter()
    try:
        probs = np.asarray(probs, dtype=np.float64)
        if not _active(snapshot):
            agent.last_wizard = "wiz=0"
            return probs
        legal, prior = _legal(snapshot, probs)
        pbs = build_pbs(snapshot, context, prior)
        leaf = leaf_vector(agent.weights, pbs["feat"])
        pay, opp0 = _payoffs(snapshot, pbs, amounts, leaf)
        cfv, _avg = _dcfr(pay, pbs["r_hero"], pbs["r_opp"], legal, prior, opp0)
        values = cfv[pbs["hero_bucket"]]
        center = float(np.sum(prior * values))
        pot_bb = max(float(snapshot.pot) / float(snapshot.big_blind), 1.0)
        logits = np.log(prior + 1e-12) + _beta(agent.weights) * (values - center) / pot_bb
        logits = np.where(legal > 0, logits, -1e9)
        fused = np.exp(logits - logits.max())
        fused /= fused.sum()
        elapsed = int((time.perf_counter() - started) * 1000)
        agent.last_wizard = "wiz=1/%dms" % elapsed
        return fused
    except Exception:
        agent.last_wizard = "wiz=err"
        return probs


if __name__ == "__main__":
    class Event:
        def __init__(self, street, kind, fraction, mine=False):
            self.actor_is_me = mine
            self.street = street
            self.action_type = kind
            self.pot_fraction = fraction

    class Snap:
        hole_cards = ("SA", "HA")
        board_cards = ("DK", "CQ", "HJ", "ST")
        events = (Event(1, "check", 0.0), Event(2, "raise", 1.1),)
        street = 2
        pot = 1800.0
        stacks = (9000.0, 8200.0)
        to_call = 624.0
        legal_action_ids = (0, 1, 2, 3, 4)
        big_blind = 100.0
        me_stack = 9000.0
        opp_stack = 8200.0

    class Ctx:
        hands_played = 10
        hero_net_bb = 4.0
        successes = {name: 1.0 for name, _ in HUD}
        opportunities = {name: 4 for name, _ in HUD}
        trap_successes = {"check_raise_freq": 1.0}
        trap_opportunities = {"check_raise_freq": 3}

    def fake_eval(*cards):
        return (sum(ord(card[0]) for card in cards),)

    bind(fake_eval)
    rng = np.random.default_rng(0)

    class Agent:
        pass

    agent = Agent()
    agent.weights = {
        "vleaf.0.weight": rng.normal(0, 0.02, (32, 48)).astype(np.float32),
        "vleaf.0.bias": np.zeros(32, np.float32),
        "vleaf.1.weight": rng.normal(0, 0.02, (12, 32)).astype(np.float32),
        "vleaf.1.bias": np.zeros(12, np.float32),
        "wizard_beta": np.float32(0.5),
    }
    probs = np.array([0.05, 0.4, 0.25, 0.2, 0.1])
    out = fuse(agent, Snap(), Ctx(), probs, {0: 0.0, 1: 624.0, 2: 900.0, 3: 1800.0, 4: 9000.0})
    assert agent.last_wizard.startswith("wiz=1"), agent.last_wizard
    assert abs(float(out.sum()) - 1) < 1e-6 and (out >= 0).all()
    started = time.perf_counter()
    for _ in range(5):
        fuse(agent, Snap(), Ctx(), probs, None)
    print("smoke", agent.last_wizard, "mean_ms", round((time.perf_counter() - started) / 5 * 1000, 2))

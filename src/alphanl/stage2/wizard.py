"""Blueprint-anchored online subgame search. NumPy only.

The 1.49M blueprint weights are read, never written. A ~10KB leaf head
supplies the street-end continuation. Search runs only postflop, and only
when the pot is at least 6BB or the hero is facing a bet.
"""
import hashlib
import time
from math import erf

import numpy as np

B = 8
K = 8
RUNOUTS = 2
CFR_ITERS = 4
TAU = 4.0
LEAF_MIX = 0.5
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
    return np.array([paired, flush, (max(ranks) - min(ranks)) / 12.0, max(ranks) / 12.0])


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


def _belief(snapshot, context):
    evaluate = _evaluate()
    rng = _rng(snapshot)
    known = {_norm(card) for card in tuple(snapshot.hole_cards) + tuple(snapshot.board_cards)}
    deck = [card for card in DECK if card not in known]
    hero = [_evcard(card) for card in snapshot.hole_cards]
    board = [_evcard(card) for card in snapshot.board_cards]
    need = 5 - len(board)
    hero_eq = np.zeros(K, dtype=np.float64)
    for index in range(K):
        take = rng.choice(len(deck), size=2, replace=False)
        opp = [deck[int(i)] for i in take]
        rest = [card for card in deck if card not in opp]
        if need == 0:
            runs = [board]
        else:
            runs = []
            for _run in range(RUNOUTS):
                draw = rng.choice(len(rest), size=need, replace=False)
                runs.append(board + [_evcard(rest[int(i)]) for i in draw])
        hero_eq[index] = sum(_win(evaluate, hero, [_evcard(card) for card in opp], run) for run in runs) / len(runs)
    strength = 1.0 - hero_eq
    post = np.ones(K, dtype=np.float64)
    events = [event for event in snapshot.events if not event.actor_is_me and event.street > 0]
    fold_rate = _hud(context, "fold_to_postflop_bet", (0.5, 6.0))
    aggro = np.mean([
        _hud(context, "flop_aggression", (0.35, 6.0)),
        _hud(context, "turn_aggression", (0.35, 6.0)),
        _hud(context, "river_aggression", (0.35, 6.0)),
    ])
    trap = _trap(context)
    for street in (1, 2, 3):
        street_events = [event for event in events if event.street == street]
        if street_events and all(event.action_type == "check" for event in street_events):
            post *= np.where(strength > 0.75, 0.45 + 0.35 * trap, 1.0)
        kinds = [event.action_type for event in street_events]
        if "check" in kinds and any(kind in ("bet", "raise", "all_in") for kind in kinds[kinds.index("check") + 1:]):
            post *= np.where(strength < 0.25, 0.25, 1.0)
    for event in events:
        post *= _likelihood(event.action_type, float(event.pot_fraction), strength, fold_rate, aggro, trap)
    post = np.clip(post, 1e-6, None)
    post /= post.sum()
    weights = np.zeros(B, dtype=np.float64)
    equity = np.zeros(B, dtype=np.float64)
    for hand_eq, hand_s, weight in zip(hero_eq, strength, post):
        bucket = min(B - 1, int(hand_s * B))
        weights[bucket] += weight
        equity[bucket] += weight * hand_eq
    equity = np.divide(equity, weights, out=np.full(B, 0.5), where=weights > 0)
    weights /= weights.sum()
    return float(np.dot(post, hero_eq)), equity, weights, fold_rate, aggro, trap


def _likelihood(kind, fraction, strength, fold_rate, aggro, trap):
    if kind == "fold":
        chance = 0.82 - 0.68 * strength + 0.16 * (fold_rate - 0.5)
    elif kind == "check":
        chance = 0.60 - 0.34 * strength + trap * 0.28 * strength
        chance = np.where(strength > 0.8, chance * (0.50 + trap), chance)
    elif kind == "call":
        chance = 0.42 + 0.16 * (1.0 - np.abs(strength - 0.45))
    else:
        big = fraction >= 0.75 or kind == "all_in"
        chance = 0.10 + (0.72 if big else 0.38) * strength * strength
        chance = chance + 0.18 * (strength < 0.22) * aggro * (0.45 if big else 1.0)
    return np.clip(chance, 0.04, 0.96)


def observe(snapshot, context):
    hero_equity, _bucket_eq, weights, *_rest = _belief(snapshot, context)
    return features(snapshot, context, hero_equity, weights)


def features(snapshot, context, hero_equity, weights):
    call = min(float(snapshot.to_call), float(snapshot.me_stack))
    pot = max(float(snapshot.pot), 1.0)
    blind = float(snapshot.big_blind)
    odds = call / (pot + call)
    spr = min(float(snapshot.me_stack), float(snapshot.opp_stack)) / pot
    values = [
        hero_equity,
        hero_equity - odds,
        pot / (blind * 200.0),
        min(spr, 20.0) / 20.0,
        call / pot,
        snapshot.street / 3.0,
    ]
    vector = np.concatenate([
        np.asarray(values, dtype=np.float64),
        _texture(snapshot.board_cards),
        weights,
        _context_vector(context),
        np.array([
            min(float(snapshot.me_stack), float(snapshot.opp_stack)) / (blind * 200.0),
            1.0 if snapshot.to_call > 0 else 0.0,
        ]),
    ])
    if vector.size != 32:
        raise RuntimeError("leaf feature width is %d" % vector.size)
    return vector


def leaf_values(weights, vector):
    if "vleaf.0.weight" not in weights:
        return np.zeros(6, dtype=np.float64)
    x = vector.astype(np.float32)
    x = _gelu(x @ weights["vleaf.0.weight"].T + weights["vleaf.0.bias"])
    x = _gelu(x @ weights["vleaf.1.weight"].T + weights["vleaf.1.bias"])
    return (x @ weights["vleaf.2.weight"].T + weights["vleaf.2.bias"]).astype(np.float64)


def _active(snapshot):
    return snapshot.street > 0 and (
        float(snapshot.pot) >= 6.0 * float(snapshot.big_blind) or float(snapshot.to_call) > 0
    )


def _chips(action, snapshot, amounts):
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


def _showdown(pot, invest, equity, blind):
    return (equity * (pot + invest) - invest) / blind


def _search(snapshot, context, blueprint, amount_map, model_weights):
    hero_equity, bucket_eq, bucket_w, fold_rate, aggro, trap = _belief(snapshot, context)
    vector = features(snapshot, context, hero_equity, bucket_w)
    leaf = float(leaf_values(model_weights, vector)[0])
    blind = float(snapshot.big_blind)
    pot = float(snapshot.pot)
    owed = float(snapshot.to_call)
    me = float(snapshot.me_stack)
    opp = float(snapshot.opp_stack)
    street = int(snapshot.street)
    legal = np.zeros(5, dtype=np.float64)
    for action in snapshot.legal_action_ids:
        if blueprint[action] > 1e-8:
            legal[action] = 1.0
    if legal.sum() <= 0:
        return blueprint, vector
    prior = np.where(legal > 0, blueprint, 0.0)
    prior /= prior.sum()
    pay = np.zeros((5, B, 3), dtype=np.float64)
    sigma0 = np.zeros((5, B, 3), dtype=np.float64)
    strength = 1.0 - bucket_eq
    for action in range(5):
        invest = _chips(action, snapshot, amount_map)
        matched = min(invest, opp)
        closes = street >= 3 or invest >= me - 1.0 or matched >= opp - 1.0
        def settle(equity, put, add):
            realized = _showdown(pot + add, put, equity, blind)
            if closes or street >= 3:
                return realized
            return (1.0 - LEAF_MIX) * realized + LEAF_MIX * leaf
        if action == 0 or legal[action] == 0:
            sigma0[action, :, 1] = 1.0
            continue
        if action == 1 and owed > 0:
            pay[action, :, :] = settle(bucket_eq, invest, 0.0)[:, None]
            sigma0[action, :, 1] = 1.0
            continue
        if action == 1:
            back = bucket_eq * pot / blind if street >= 3 else leaf
            their = min(opp, max(pot * 0.66, blind))
            respond = np.maximum(settle(bucket_eq, their, their), 0.0)
            pay[action, :, 1] = back
            pay[action, :, 2] = respond
            bet_p = np.clip(0.10 + 0.72 * strength + 0.10 * (strength < 0.2) * aggro, 0.05, 0.9)
            sigma0[action, :, 2] = bet_p
            sigma0[action, :, 1] = 1.0 - bet_p
            continue
        pay[action, :, 0] = pot / blind
        pay[action, :, 1] = settle(bucket_eq, invest, matched)
        extra = np.maximum(0.0, opp - invest)
        call_ev = _showdown(pot + opp, invest + extra, bucket_eq, blind)
        pay[action, :, 2] = np.maximum(call_ev, -invest / blind)
        big = invest >= 0.75 * max(pot, blind)
        fold_p = np.clip(0.60 - 0.58 * strength + 0.18 * (fold_rate - 0.5) + (0.12 if big else 0.0), 0.04, 0.90)
        raise_p = np.clip(0.03 + 0.42 * strength * strength + trap * 0.16 * strength, 0.02, 0.55)
        over = np.maximum(0.0, fold_p + raise_p - 0.96)
        fold_p -= over * 0.5
        raise_p -= over * 0.5
        sigma0[action, :, 0] = fold_p
        sigma0[action, :, 2] = raise_p
        sigma0[action, :, 1] = 1.0 - fold_p - raise_p
    hero = prior.copy()
    regret = np.zeros(5, dtype=np.float64)
    opp_regret = np.zeros_like(pay)
    opp = sigma0
    for _ in range(CFR_ITERS):
        ev = np.einsum("b,abo,abo->a", bucket_w, opp, pay)
        regret = np.maximum(0.0, regret + (ev - np.sum(hero * ev)))
        regret *= legal
        total = regret.sum()
        if total > 0:
            hero = regret / total
        opp_value = -pay
        base = np.sum(opp * opp_value, axis=-1, keepdims=True)
        opp_regret = np.maximum(0.0, opp_regret + opp_value - base)
        denom = opp_regret.sum(axis=-1, keepdims=True)
        opp = np.where(denom > 1e-8, opp_regret / np.maximum(denom, 1e-8), sigma0)
    ev = np.einsum("b,abo,abo->a", bucket_w, opp, pay)
    center = float(np.sum(prior * ev))
    logits = np.log(prior + 1e-12) + (ev - center) / TAU
    logits = np.where(legal > 0, logits, -1e9)
    fused = np.exp(logits - logits.max())
    fused /= fused.sum()
    return fused, vector


def fuse(agent, snapshot, context, probs, amounts):
    """KL-regularized mix of the blueprint and one-street search values."""
    started = time.perf_counter()
    try:
        if not _active(snapshot) or "vleaf.0.weight" not in agent.weights:
            agent.last_wizard = "wiz=0"
            return probs
        fused, _ = _search(snapshot, context, np.asarray(probs, dtype=np.float64), amounts, agent.weights)
        elapsed = int((time.perf_counter() - started) * 1000)
        agent.last_wizard = "wiz=1/%dms" % elapsed
        return fused
    except Exception:
        agent.last_wizard = "wiz=err"
        return probs


if __name__ == "__main__":
    class Event:
        def __init__(self, street, kind, fraction):
            self.actor_is_me = False
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
        to_call = 600.0
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
    weights = {
        "vleaf.0.weight": rng.normal(0, 0.02, (48, 32)).astype(np.float32),
        "vleaf.0.bias": np.zeros(48, np.float32),
        "vleaf.1.weight": rng.normal(0, 0.02, (16, 48)).astype(np.float32),
        "vleaf.1.bias": np.zeros(16, np.float32),
        "vleaf.2.weight": rng.normal(0, 0.02, (6, 16)).astype(np.float32),
        "vleaf.2.bias": np.zeros(6, np.float32),
    }

    class Agent:
        pass

    agent = Agent()
    agent.weights = weights
    probs = np.array([0.05, 0.4, 0.25, 0.2, 0.1])
    out = fuse(agent, Snap(), Ctx(), probs, None)
    assert agent.last_wizard.startswith("wiz=1"), agent.last_wizard
    assert abs(out.sum() - 1) < 1e-6 and (out >= 0).all(), out
    started = time.perf_counter()
    for _ in range(20):
        fuse(agent, Snap(), Ctx(), probs, None)
    print("smoke", agent.last_wizard, "mean_ms", round((time.perf_counter() - started) / 20 * 1000, 2), out.round(3))

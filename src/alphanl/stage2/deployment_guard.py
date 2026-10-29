"""Visible-state-only v10 deployment intervention, shared by NumPy and eval.

This is a conservative short-match heuristic, not range equity or a training
action abstraction. No deck/opponent/policy RNG is consumed here. Python 3.6.
"""
import math
import numpy as np
from .portable_evaluator import strength


V10_GUARD = {
    "enabled": True,
    "prune_below": 0.10,
    "active_shove_min_bb": 30.0,
    "active_shove_pot_ratio": 2.0,
    "overbet_call_min_bb": 50.0,
    "overbet_call_pot_ratio": 1.2,
    "overbet_min_equity": 0.82,
    "require_made_hand": True,
    # Earlier 40BB confidence lock remains an opt-in ablation. v10 defaults
    # implement the latest requested overbet fuse without this extra gate.
    "big_call_enabled": False,
    "big_call_min_bb": 40.0,
    "big_call_min_probability": 0.35,
    "big_call_equity_margin": 0.05,
    "equity_samples": 128,
    "equity_seed": 17,
}


def guard_config(config=None):
    result = dict(V10_GUARD)
    # Absent metadata preserves the behavior of existing exports.
    if config is None:
        result["enabled"] = False
    else:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("unknown deployment guard fields: " + str(sorted(unknown)))
        result.update(config)
    for key in ("enabled", "require_made_hand", "big_call_enabled"):
        if not isinstance(result[key], bool):
            raise ValueError(key + " must be boolean")
    for key in ("prune_below", "overbet_min_equity", "big_call_min_probability", "big_call_equity_margin"):
        if not isinstance(result[key], (int, float)) or not 0 <= result[key] <= 1:
            raise ValueError(key + " must be in [0,1]")
    for key in ("active_shove_min_bb", "active_shove_pot_ratio", "overbet_call_min_bb",
                "overbet_call_pot_ratio", "big_call_min_bb"):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    for key in ("equity_samples", "equity_seed"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < (1 if key == "equity_samples" else 0):
            raise ValueError(key + " must be a valid integer")
    return result


def strong_made_hand(snapshot):
    """Two pair+ with private contribution; board-only strength is insufficient.

    Preflop has no made-hand category: the separate .82 equity gate applies.
    With a five-card board, kicker-only improvement to board trips/two pair
    does not qualify. This is deliberately stricter than random-hand equity.
    """
    board = snapshot.board_cards
    if not board:
        return True
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + board])
    if own[0] < 2:
        return False
    if len(board) < 5:
        # Board trips/quads alone are not a privately made monster.
        ranks = [c[1] for c in board]
        pairs = sorted(["23456789TJQKA".index(r) + 2 for r in set(ranks) if ranks.count(r) >= 2], reverse=True)
        if own[0] == 2 and len(pairs) == 2 and own[:3] <= (2, *pairs):
            return False
        if own[0] in (3, 7) and max(ranks.count(r) for r in ranks) >= (3 if own[0] == 3 else 4):
            return False
        return True
    public = strength([convert(c) for c in board])
    if own[0] != public[0]:
        return own[0] > public[0]
    primary = {2: 3, 3: 2, 7: 2}.get(own[0], len(own))
    return own[:primary] > public[:primary]


def pot_before_bet(snapshot):
    for event in reversed(snapshot.events):
        if (event.street == snapshot.street and not event.actor_is_me
                and event.chips_added > event.to_call_before):
            return max(float(event.pot_before), 1.0), "history"
    return max(float(snapshot.pot) - float(snapshot.to_call), 1.0), "pot_minus_call_fallback"


def apply_guard(snapshot, probabilities, config, equity_fn, amounts=None):
    """Return normalized probabilities and audit; confidence uses RAW policy.

    ``amounts`` optionally supplies exact legal judge chip investments. A pot
    raise whose minimum legal amount breaches the shove cap is not a safe
    remapping target. Any big-call fuse forbids ALL continuing actions.
    """
    raw = np.asarray(probabilities, dtype=np.float64).copy()
    if raw.shape != (5,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("invalid five-action probabilities")
    legal = set(snapshot.legal_action_ids)
    for action in range(5):
        if action not in legal:
            raw[action] = 0.0
    if raw.sum() <= 0:
        raise ValueError("empty legal policy")
    raw /= raw.sum()
    info = {"pruned": [], "active_cap": False, "call_fuse": None}
    if not config["enabled"]:
        return raw, info
    probs = raw.copy()
    for action in legal:
        if 0 < raw[action] < config["prune_below"]:
            probs[action] = 0.0
            info["pruned"].append(action)
    if probs.sum() <= 0:
        probs[int(np.argmax(raw))] = 1.0

    blind = float(snapshot.big_blind)
    call = min(float(snapshot.to_call), snapshot.me_stack)
    before, source = pot_before_bet(snapshot)
    giant = (call > config["overbet_call_min_bb"] * blind
             and call > config["overbet_call_pot_ratio"] * before)
    big = config["big_call_enabled"] and call > config["big_call_min_bb"] * blind
    if (giant or big) and 0 in legal:
        equity = float(equity_fn(snapshot, samples=config["equity_samples"], seed=config["equity_seed"]))
        if not math.isfinite(equity) or not 0 <= equity <= 1:
            raise ValueError("invalid estimated equity")
        odds = call / (snapshot.pot + call)
        made = strong_made_hand(snapshot)
        info.update(equity=equity, pot_odds=odds, strong_made_hand=made,
                    pot_before_bet=before, pot_before_source=source)
        if giant and (equity < config["overbet_min_equity"]
                      or (config["require_made_hand"] and not made)):
            info["call_fuse"] = "overbet_shove"
        elif big and (raw[1] < config["big_call_min_probability"]
                      or equity < odds + config["big_call_equity_margin"]):
            info["call_fuse"] = "big_call"
        if info["call_fuse"]:
            probs[:] = 0.0
            probs[0] = 1.0
            return probs, info

    amount = {1: call, 2: int(snapshot.pot * .5), 3: int(snapshot.pot), 4: snapshot.me_stack}
    if amounts is not None:
        amount.update(amounts)
    def excessive(chips):
        return (chips > config["active_shove_min_bb"] * blind
                and chips > config["active_shove_pot_ratio"] * snapshot.pot)
    if 4 in legal and snapshot.me_stack > snapshot.to_call and excessive(amount[4]) and probs[4] > 0:
        # Prefer the standard full-pot action, then half-pot. If a large
        # minimum raise makes both unsafe, keep only check/call or fold.
        targets = [a for a in (3, 2) if a in legal and not excessive(amount[a])]
        target = targets[0] if targets else 1 if 1 in legal else 0
        if target not in legal:
            raise ValueError("no safe legal shove remapping")
        info["active_cap"] = True
        info["remapped_to"] = target
        probs[target] += probs[4]
        probs[4] = 0.0
    probs /= probs.sum()
    return probs, info

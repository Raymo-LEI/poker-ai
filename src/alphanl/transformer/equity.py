"""Visible-card-only, reproducible equity versus a uniform random hand.

The preflop artifact is a Monte Carlo estimate, NOT exact enumeration.
Postflop sampling has a separate per-infoset RNG: features never consume
training/deck/opponent randomness or inspect a pre-sampled future board.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .schema import RANKS, normalize_cards
from .tables import standard_deck

TABLE_PATH = Path(__file__).with_name("data") / "preflop_equity_v1.json"
# Lazy loading below permits rebuilding a missing table with the builder.
def equity_numeric_version():
    return "hunl_random_equity_odds_v1:" + hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()


def preflop_key(hole):
    a, b = normalize_cards(hole)
    high, low = sorted((a[1], b[1]), key=RANKS.index, reverse=True)
    return high + low + ("" if high == low else "s" if a[0] == b[0] else "o")


@lru_cache(maxsize=1)
def preflop_table():
    data = json.loads(TABLE_PATH.read_text())
    if data["format"] != "random_equity_preflop_mc_v1" or len(data["equities"]) != 169:
        raise ValueError("invalid preflop equity table")
    return data["equities"]


def sample_equity(hole_cards, board_cards, rng, samples):
    """Reuse the rule baseline's phevaluator algorithm and tie=half convention."""
    from phevaluator import evaluate_cards
    known = set(hole_cards + board_cards)
    remaining = [c for c in standard_deck() if c not in known]
    convert = lambda c: c[1] + c[0].lower()
    hole = [convert(c) for c in hole_cards]
    board = [convert(c) for c in board_cards]
    remaining = [convert(c) for c in remaining]
    need = 5 - len(board)
    score = 0.
    for _ in range(samples):
        draw = rng.choice(len(remaining), size=need + 2, replace=False)
        runout = board + [remaining[i] for i in draw[:need]]
        own = evaluate_cards(*(hole + runout))
        opp = evaluate_cards(*([remaining[i] for i in draw[need:]] + runout))
        score += float(own < opp) + .5 * float(own == opp)
    return score / samples


@lru_cache(maxsize=65536)
def _postflop(hole, board, samples, seed):
    digest = hashlib.blake2b(repr((hole, board, samples, seed)).encode(), digest_size=8).digest()
    return sample_equity(hole, board, np.random.default_rng(int.from_bytes(digest, "little")), samples)


def estimated_equity(snapshot, *, samples=32, seed=17):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("equity samples must be a positive integer")
    hole = tuple(sorted(normalize_cards(snapshot.hole_cards)))
    board = tuple(sorted(normalize_cards(snapshot.board_cards)))
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board):
        raise ValueError("equity requires two distinct hole cards and 0/3/4/5 visible board cards")
    if not board:
        return float(preflop_table()[preflop_key(hole)])
    return _postflop(hole, board, samples, seed)


def equity_features(snapshot, *, samples=32, seed=17):
    equity = estimated_equity(snapshot, samples=samples, seed=seed)
    # The short all-in call cannot spend more than the remaining stack.
    call = max(0., min(float(snapshot.to_call), snapshot.me_stack))
    denominator = float(snapshot.pot) + call
    odds = call / denominator if denominator > 0 else 0.
    return equity, odds, equity - odds

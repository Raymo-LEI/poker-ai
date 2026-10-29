"""Ladder-replica scripts for the Stage 2 pool. Visible cards only.

These are style probes inspired by Botzone ladder habits, not certified
copies of any named bot. The five-action hero policy is unchanged. Only
``OffTree_Aggro_Boss`` spends chips off that tree, through ``PotFractionBet``.
"""
from collections import Counter

import numpy as np

from alphanl.cfr.pro_bots import raises, texture
from alphanl.transformer.actor_actions import PotFractionBet
from alphanl.transformer.equity import estimated_equity, preflop_key
from alphanl.transformer.schema import RANKS

from .allin_mask import PREMIUM_PREFLOP
from .deployment_guard import strong_made_hand

LADDER_OPPONENTS = ("OffTree_Aggro_Boss", "Check_Shove_Trapper", "Flop_Raise_War_Bot")
LADDER_VERSION = "ladder_replicas_v1"
DEFAULT_LADDER_PARAMS = dict(
    open_raise=.85, three_bet=.70, five_bet_shove=.80, postflop_pressure=.85,
    size_low=.35, size_high=1.15, cold_shove=.18, value_shove_share=.75, flop_raise=.85,
)


def top_pair(snapshot):
    """Hero pairs the highest rank showing on a flop or later board."""
    board = snapshot.board_cards
    if len(board) < 3:
        return False
    rank = lambda card: RANKS.index(card[1])
    top = max(rank(card) for card in board)
    return any(rank(card) == top for card in snapshot.hole_cards)


def flop_pressure_hand(snapshot):
    """Flop top pair, true two pair or better, or a strong draw."""
    if snapshot.street != 1 or len(snapshot.board_cards) < 3:
        return False
    if top_pair(snapshot) or strong_made_hand(snapshot):
        return True
    return bool(texture(snapshot)["strong_draw"])


def is_value_hand(snapshot, *, equity_samples, seed=17):
    """Value side of the check/shove mix. Bluffs are the complement."""
    if snapshot.street == 0:
        if preflop_key(snapshot.hole_cards) in PREMIUM_PREFLOP:
            return True
        return estimated_equity(snapshot, samples=equity_samples, seed=seed) >= 0.62
    if snapshot.board_cards and (strong_made_hand(snapshot) or top_pair(snapshot)):
        return True
    return estimated_equity(snapshot, samples=equity_samples, seed=seed) >= 0.70


class LadderReplicaActor(object):
    """Fixed parameters. Evaluation and the gauntlet use this class."""

    def __init__(self, name, *, seed, equity_samples=64):
        if name not in LADDER_OPPONENTS or isinstance(equity_samples, bool) or equity_samples < 1:
            raise ValueError("invalid ladder replica or equity sample count")
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.equity_samples = int(equity_samples)
        self.variant = "ladder_replica_v1"
        self.params = dict(DEFAULT_LADDER_PARAMS)
        self.counts = Counter()

    def begin_hand(self, seat):
        return None

    def act(self, snapshot):
        if self.name == "OffTree_Aggro_Boss":
            return self._aggro(snapshot)
        if self.name == "Check_Shove_Trapper":
            return self._trap(snapshot)
        return self._flop_war(snapshot)

    def _prefer(self, snapshot, *choices):
        legal = snapshot.legal_action_ids
        return next(action for action in choices if action in legal)

    def _aggro(self, snapshot):
        if snapshot.street == 0:
            seen = len(raises(snapshot.events))
            if seen >= 2:
                self.counts["five_bet_spots"] += 1
                if self.rng.random() < self.params["five_bet_shove"]:
                    self.counts["five_bet_shoves"] += 1
                    return self._prefer(snapshot, 4, 3, 2, 1, 0)
                return self._prefer(snapshot, 1, 0, 4, 3, 2)
            frequency = self.params["open_raise"] if seen == 0 else self.params["three_bet"]
            if self.rng.random() < frequency:
                self.counts["preflop_raises"] += 1
                return self._prefer(snapshot, 3, 2, 4, 1, 0)
            return self._prefer(snapshot, 1, 0, 2, 3, 4)
        if self.rng.random() < self.params["postflop_pressure"]:
            low, high = self.params["size_low"], self.params["size_high"]
            fraction = float(self.rng.uniform(low, high))
            self.counts["off_tree_bets"] += 1
            return PotFractionBet(fraction)
        return self._prefer(snapshot, 1, 0, 2, 3, 4)

    def _trap(self, snapshot):
        if snapshot.to_call > 0:
            if is_value_hand(snapshot, equity_samples=self.equity_samples):
                return self._prefer(snapshot, 1, 0, 4, 2, 3)
            return self._prefer(snapshot, 0, 1, 2, 3, 4)
        if self.rng.random() >= self.params["cold_shove"]:
            self.counts["checks"] += 1
            return self._prefer(snapshot, 1, 0, 2, 3, 4)
        value = is_value_hand(snapshot, equity_samples=self.equity_samples)
        mode = "value" if self.rng.random() < self.params["value_shove_share"] else "bluff"
        if (mode == "value" and value) or (mode == "bluff" and not value):
            self.counts[mode + "_shoves"] += 1
            return self._prefer(snapshot, 4, 3, 2, 1, 0)
        self.counts["checks"] += 1
        return self._prefer(snapshot, 1, 0, 2, 3, 4)

    def _flop_war(self, snapshot):
        if flop_pressure_hand(snapshot):
            street = [event for event in snapshot.events if event.street == snapshot.street]
            can_raise = any(action in snapshot.legal_action_ids for action in (2, 3, 4))
            if can_raise and len(raises(street)) < 4 and self.rng.random() < self.params["flop_raise"]:
                self.counts["flop_raises"] += 1
                return self._prefer(snapshot, 3, 2, 4, 1, 0)
        return self._prefer(snapshot, 1, 0, 2, 3, 4)


class JitteredLadderReplicaActor(LadderReplicaActor):
    """Training-only frequencies drawn once and held for the whole match.

    The off-tree size band stays inside 0.35x-1.15x pot. The value/bluff
    shove share stays near 75/25.
    """

    def __init__(self, name, *, seed, equity_samples=64):
        super(JitteredLadderReplicaActor, self).__init__(name, seed=seed, equity_samples=equity_samples)
        self.variant = "jittered_ladder_v1"
        draw = self.rng.uniform
        self.params = dict(self.params)
        self.params.update(
            open_raise=float(draw(.75, .95)),
            three_bet=float(draw(.60, .85)),
            five_bet_shove=float(draw(.70, .90)),
            postflop_pressure=float(draw(.75, .95)),
            cold_shove=float(draw(.12, .24)),
            value_shove_share=float(draw(.70, .80)),
            flop_raise=float(draw(.75, .95)),
        )

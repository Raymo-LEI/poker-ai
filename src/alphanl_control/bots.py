"""Baseline opponents for step-1 eval (same observation API as AlphaNLHoldem)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

RANK_ORDER = "23456789TJQKA"


def _card_rank(card: str) -> int:
    return RANK_ORDER.index(card[1])


def _has_straight(ranks) -> bool:
    present = set(ranks)
    if 12 in present:
        present.add(-1)
    return any(all(top - k in present for k in range(5)) for top in range(3, 13))


class RandomBot:
    def make_action(self, obs: Dict[str, Any]) -> int:
        legal = np.where(obs["legal_moves"])[0]
        return int(np.random.choice(legal))


class RuleBot:
    """
    Simple heuristic bot on rlcard raw state (tight preflop, strength-based postflop).
    Not strong; intended as a step above uniform random.
    """

    def __init__(self, rlcard_env) -> None:
        self._game = rlcard_env

    def make_action(self, obs: Dict[str, Any]) -> int:
        legal = list(np.where(obs["legal_moves"])[0])
        pid = self._game.get_player_id()
        state = self._game.get_state(pid)
        raw = state["raw_obs"]
        hole = raw["hand"]
        board = raw["public_cards"]
        strength = self._hand_strength(hole, board)

        if strength >= 7:
            return self._prefer(legal, [4, 3, 2, 1, 0])
        if strength >= 4:
            return self._prefer(legal, [2, 3, 1, 4, 0])
        if strength >= 2:
            return self._prefer(legal, [1, 2, 0, 3, 4])
        return self._prefer(legal, [1, 0, 2, 3, 4])

    @staticmethod
    def _prefer(legal: list, order: list) -> int:
        for a in order:
            if a in legal:
                return int(a)
        return int(legal[0])

    def _hand_strength(self, hole: list, board: list) -> int:
        if len(board) == 0:
            return self._preflop_strength(hole)
        if len(board) >= 3:
            return self._postflop_category(list(hole) + list(board))
        return self._made_hand_category(hole, board)

    @staticmethod
    def _postflop_category(cards: list) -> int:
        """Best five-card category of 5 to 7 real cards, on rlcard's ``Hand.category`` scale.

        rlcard's ``Hand.evaluateHand`` only accepts exactly seven cards, and
        padding the flop or turn with filler cards invents pairs, straights
        and flushes, so the category is computed here instead.
        """

        ranks = [_card_rank(c) for c in cards]
        counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
        by_suit: Dict[str, list] = {}
        for card in cards:
            by_suit.setdefault(card[0], []).append(_card_rank(card))
        flush_ranks = next((rs for rs in by_suit.values() if len(rs) >= 5), None)
        if flush_ranks is not None and _has_straight(flush_ranks):
            return 9
        if counts[0] >= 4:
            return 8
        if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
            return 7
        if flush_ranks is not None:
            return 6
        if _has_straight(ranks):
            return 5
        if counts[0] == 3:
            return 4
        if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
            return 3
        if counts[0] == 2:
            return 2
        return 1

    @staticmethod
    def _preflop_strength(hole: list) -> int:
        r1, r2 = _card_rank(hole[0]), _card_rank(hole[1])
        high, low = max(r1, r2), min(r1, r2)
        suited = hole[0][0] == hole[1][0]
        if r1 == r2:
            return 4 + (high // 3)
        if high >= 12 and low >= 9:
            return 6
        if high >= 12:
            return 5
        if suited and high - low <= 2 and low >= 4:
            return 4
        if high >= 10:
            return 3
        return 1

    @staticmethod
    def _made_hand_category(hole: list, board: list) -> int:
        ranks = [_card_rank(c) for c in hole + board]
        counts = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1
        mx = max(counts.values())
        if mx >= 4:
            return 8
        if mx == 3 and 2 in counts.values():
            return 7
        if mx == 3:
            return 4
        pairs = sum(1 for v in counts.values() if v >= 2)
        if pairs >= 2:
            return 3
        if pairs == 1:
            return 2
        return 1

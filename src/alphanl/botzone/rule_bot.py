"""Weak Botzone rule opponent on the five policy ids.

Strength bands are the ones in ``alphanl.bots.RuleBot``. This module does
not call ``RuleBot.make_action``: that method speaks rlcard's observation
and its own bet sizes. Here the same integers select among

``fold / check_call / raise_min / raise_large / all_in``.

Preflop points are that bot's pair / high-card / suited-connector table.
Postflop uses the made-hand category on the same 1..9 scale (high card
through straight flush). Design §5.3 names that scale: high is ≥ 7, which
already includes pocket aces and kings, mid is ≥ 4, low is ≥ 2, and the
rest check when a check is free and fold when facing a bet.

The four priority orders are the design table. They are not retuned.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from .actions import (
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    legal_action_ids,
)
from .legal import to_call
from .table import BotzoneTable, _local_rank

# RuleBot.Hand.category / ``_preflop_strength`` cut points. Do not shift them.
HIGH_AT = 7
MID_AT = 4
LOW_AT = 2

# Design §5.3, in policy-id order.
HIGH_PRIORITY: Tuple[int, ...] = (
    ALL_IN_ACTION,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
)
MID_PRIORITY: Tuple[int, ...] = (
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    CHECK_CALL_ACTION,
    ALL_IN_ACTION,
    FOLD_ACTION,
)
LOW_PRIORITY: Tuple[int, ...] = (
    CHECK_CALL_ACTION,
    RAISE_MIN_ACTION,
    FOLD_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)
# Rest, when a check is free. Check is ``check_call`` with nothing owed.
REST_CHECK_PRIORITY: Tuple[int, ...] = (
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)
# Rest, facing a bet: fold, then call. Raises stay behind both.
REST_FACING_PRIORITY: Tuple[int, ...] = (
    FOLD_ACTION,
    CHECK_CALL_ACTION,
    RAISE_MIN_ACTION,
    RAISE_LARGE_ACTION,
    ALL_IN_ACTION,
)


def preflop_strength(hole: Sequence[int]) -> int:
    """``RuleBot._preflop_strength`` on wiki card ids.

    Rank index 0 is a deuce and 12 is an ace, matching ``"23456789TJQKA"``.
    A pair scores ``4 + high // 3``, so jacks through aces are ≥ 7.
    """

    if len(hole) != 2:
        raise ValueError(f"hole must be two cards, got {hole!r}")
    r1, r2 = hole[0] // 4, hole[1] // 4
    high, low = max(r1, r2), min(r1, r2)
    suited = hole[0] % 4 == hole[1] % 4
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


def postflop_strength(hole: Sequence[int], board: Sequence[int]) -> int:
    """Made-hand category on the RuleBot 1..9 scale.

    ``BotzoneTable``'s local evaluator ranks a straight flush as 8 and a
    high card as 0. Adding 1 is the limitholdem ``Hand.category`` numbering
    the original thresholds were written against: 9 straight flush, 8
    quads, 7 full house, 6 flush, 5 straight, 4 trips, 3 two pair, 2 pair,
    1 high card. The best five-card subset is used, so a flush or a
    straight is not collapsed into the pair counter.
    """

    if len(hole) != 2 or len(board) < 3:
        raise ValueError(f"postflop needs two hole cards and a board, got {hole!r} {board!r}")
    ranked = _local_rank((*hole, *board))
    return int(ranked[0]) + 1


def hand_strength(hole: Sequence[int], board: Sequence[int]) -> int:
    """Preflop points, or the postflop category once three board cards are out."""

    if len(board) == 0:
        return preflop_strength(hole)
    return postflop_strength(hole, board)


def action_priority(strength: int, to_call_chips: int) -> Tuple[int, ...]:
    """Policy-id order for this band. First legal id in the order is played."""

    if strength >= HIGH_AT:
        return HIGH_PRIORITY
    if strength >= MID_AT:
        return MID_PRIORITY
    if strength >= LOW_AT:
        return LOW_PRIORITY
    if to_call_chips <= 0:
        return REST_CHECK_PRIORITY
    return REST_FACING_PRIORITY


def choose_action(
    legal: Sequence[int],
    strength: int,
    to_call_chips: int,
) -> int:
    """First id from :func:`action_priority` that is currently legal."""

    allowed = {int(action) for action in legal}
    if not allowed:
        raise ValueError("no legal action")
    for action in action_priority(strength, to_call_chips):
        if action in allowed:
            return action
    raise ValueError(f"priority missed every legal id {tuple(legal)}")


class BotzoneRuleBot:
    """Pick a 5-id action from the seat that is about to act.

    Only that seat's hole cards are read. The other seat's hole cards
    stay unused until showdown, which this bot does not call.
    """

    def action_id(self, table: BotzoneTable) -> int:
        state = table.state
        if state is None or state.hand_over:
            raise ValueError("no seat is acting")
        seat = state.actor
        hole = table.holes[seat]
        owed = to_call(state, seat)
        return choose_action(
            legal_action_ids(state),
            hand_strength(hole, table.board),
            owed,
        )

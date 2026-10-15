"""Five Botzone policy ids, packed through M1 legality.

The policy head stays width 5. This module is the only place that turns
those ids into a Botzone integer (``-1`` / ``0`` / ``safe`` /
``max(safe * 2, pot)`` / ``-2``). ``action_responses`` both builds the
candidate and drops it unless ``legal_response`` accepts that integer,
so an id is unmasked exactly when its response is legal.

``pot`` is the pot before the action. After anyone is all-in, M1's set
is only fold and all-in, and the two raise ids have no candidate.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .legal import ALL_IN, CHECK_CALL, FOLD, HandState, is_legal, safe_extra

NUM_ACTIONS = 5

FOLD_ACTION = 0
CHECK_CALL_ACTION = 1
RAISE_MIN_ACTION = 2
RAISE_LARGE_ACTION = 3
ALL_IN_ACTION = 4

ACTION_NAMES: Tuple[str, ...] = (
    "fold",
    "check_call",
    "raise_min",
    "raise_large",
    "all_in",
)


def action_responses(state: HandState) -> Tuple[Optional[int], ...]:
    """Response for each id, or ``None`` when that id is masked.

    Index ``i`` is policy id ``i``. ``legal_action_ids`` and
    ``response_for_action`` only read this tuple.
    """

    safe = safe_extra(state)
    large = None if safe is None else max(safe * 2, state.pot)
    candidates = (FOLD, CHECK_CALL, safe, large, ALL_IN)
    return tuple(
        response if response is not None and is_legal(state, response) else None
        for response in candidates
    )


def legal_action_ids(state: HandState) -> Tuple[int, ...]:
    """Policy ids whose packed response passed ``legal_response``."""

    return tuple(
        action_id
        for action_id, response in enumerate(action_responses(state))
        if response is not None
    )


def response_for_action(state: HandState, action_id: int) -> Optional[int]:
    """Packed response for one id, or ``None`` when that id is masked."""

    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError(f"action id must be an int, got {action_id!r}")
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"action id must be in 0..{NUM_ACTIONS - 1}, got {action_id}")
    return action_responses(state)[action_id]

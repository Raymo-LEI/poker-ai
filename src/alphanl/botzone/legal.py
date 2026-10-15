"""Pure legality for Botzone simple TexasHoldem2p.

Responses are integers, matching the wiki:

- ``-1`` fold
- ``-2`` all-in
- ``0`` check or call
- a positive int is the extra chips put in on this action (not raise-to)

Two lower bounds disagree once a raise has happened. The extra that may
be sent is ``max(A, B)``:

- A, by street commitment. No chips in on this street yet: 100. Otherwise
  ``2 *`` the largest street commitment. Preflop blinds count, so the
  opening bound is ``2 * 100 = 200``.
- B, by the wiki sample. Preflop starts at 200. Each real bet or raise
  sets it to ``max(B, 2 * extra)``. A new street resets it to 100.

A fold is always offered, including when a check is free. After anyone
is all-in, the only responses are fold and all-in. Calling requires the
stack to be strictly greater than the chips owed. A positive extra must
be strictly less than the stack. There are no side pots: both seats
reset to 20000, and remaining chips plus the pot stay 40000.

``apply`` only advances betting. A fold pays the pot to the other seat.
If betting ends without a fold, the pot stays where it is and the hand
is over. Dealing and showdown are not this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

STARTING_STACK = 20000
SMALL_BLIND = 50
BIG_BLIND = 100
TOTAL_CHIPS = STARTING_STACK * 2

FOLD = -1
ALL_IN = -2
CHECK_CALL = 0

_PREFLOP = 0
_RIVER = 3


@dataclass(frozen=True)
class HandState:
    """Betting state after blinds, with no cards.

    ``stacks`` and ``street_commit`` are indexed by seat. ``dealer`` posts
    the big blind; the other seat posts the small blind and acts first on
    every street.
    """

    dealer: int
    street: int
    stacks: Tuple[int, int]
    street_commit: Tuple[int, int]
    pot: int
    bound_b: int
    actor: int
    all_in: Tuple[bool, bool]
    folded: Tuple[bool, bool]
    needs_action: Tuple[bool, bool]
    hand_over: bool


def new_hand(dealer: int = 0) -> HandState:
    """Post blinds and give the small blind the first action."""

    if dealer not in (0, 1):
        raise ValueError(f"dealer must be 0 or 1, got {dealer}")
    sb = _sb_seat(dealer)
    bb = dealer
    stacks = [STARTING_STACK, STARTING_STACK]
    commit = [0, 0]
    stacks[sb] -= SMALL_BLIND
    stacks[bb] -= BIG_BLIND
    commit[sb] = SMALL_BLIND
    commit[bb] = BIG_BLIND
    return _checked(
        HandState(
            dealer=dealer,
            street=_PREFLOP,
            stacks=(stacks[0], stacks[1]),
            street_commit=(commit[0], commit[1]),
            pot=SMALL_BLIND + BIG_BLIND,
            bound_b=2 * BIG_BLIND,
            actor=sb,
            all_in=(False, False),
            folded=(False, False),
            needs_action=(True, True),
            hand_over=False,
        )
    )


def lower_bound_a(state: HandState) -> int:
    """Minimum extra from the current street's largest commitment."""

    max_commit = max(state.street_commit)
    if max_commit == 0:
        return BIG_BLIND
    return 2 * max_commit


def lower_bound_b(state: HandState) -> int:
    """Minimum extra tracked the way the wiki sample tracks ``round_raise``."""

    return state.bound_b


def to_call(state: HandState, seat: Optional[int] = None) -> int:
    """Chips the seat must put in to match the current street bet."""

    who = state.actor if seat is None else seat
    return max(state.street_commit) - state.street_commit[who]


def safe_extra(state: HandState) -> Optional[int]:
    """Smallest legal positive extra, or ``None`` when that raise does not exist.

    The number is ``max(A, B)``. It exists only when nobody is all-in and
    it is strictly less than the actor's remaining stack (and, with these
    bounds, strictly more than the call).
    """

    if state.hand_over or any(state.all_in):
        return None
    safe = max(lower_bound_a(state), lower_bound_b(state))
    remaining = state.stacks[state.actor]
    if to_call(state) < safe < remaining:
        return safe
    return None


def legal_response(state: HandState) -> set:
    """Legal responses for the seat about to act.

    ``{-1, -2, 0}`` plus every positive extra that clears the safe minimum
    and is strictly less than the remaining stack. Fold is always included
    while the hand is still asking this seat to act.
    """

    if state.hand_over:
        return set()
    remaining = state.stacks[state.actor]
    if remaining <= 0:
        return set()

    legal = {FOLD, ALL_IN}
    if any(state.all_in):
        return legal

    if remaining > to_call(state):
        legal.add(CHECK_CALL)
    safe = safe_extra(state)
    if safe is not None:
        legal.update(range(safe, remaining))
    return legal


def apply(state: HandState, response: int) -> HandState:
    """Return the next betting state. Does not mutate ``state``."""

    if state.hand_over:
        raise ValueError("hand is over")
    response = _require_response(response)
    if response not in legal_response(state):
        raise ValueError(f"illegal response {response}")

    actor = state.actor
    other = 1 - actor
    stacks = [state.stacks[0], state.stacks[1]]
    commit = [state.street_commit[0], state.street_commit[1]]
    all_in = [state.all_in[0], state.all_in[1]]
    folded = [state.folded[0], state.folded[1]]
    needs = [state.needs_action[0], state.needs_action[1]]
    pot = state.pot
    bound_b = state.bound_b

    if response == FOLD:
        folded[actor] = True
        stacks[other] += pot
        pot = 0
        return _checked(
            HandState(
                dealer=state.dealer,
                street=state.street,
                stacks=(stacks[0], stacks[1]),
                street_commit=(commit[0], commit[1]),
                pot=pot,
                bound_b=bound_b,
                actor=actor,
                all_in=(all_in[0], all_in[1]),
                folded=(folded[0], folded[1]),
                needs_action=(False, False),
                hand_over=True,
            )
        )

    if response == ALL_IN:
        chips = stacks[actor]
    elif response == CHECK_CALL:
        chips = to_call(state, actor)
    else:
        chips = response

    prev_max = max(commit)
    stacks[actor] -= chips
    commit[actor] += chips
    pot += chips
    raised = commit[actor] > prev_max
    if stacks[actor] == 0:
        all_in[actor] = True
    if raised:
        bound_b = max(bound_b, 2 * chips)
        if not folded[other] and not all_in[other] and stacks[other] > 0:
            needs[other] = True
    needs[actor] = False

    advanced = HandState(
        dealer=state.dealer,
        street=state.street,
        stacks=(stacks[0], stacks[1]),
        street_commit=(commit[0], commit[1]),
        pot=pot,
        bound_b=bound_b,
        actor=actor,
        all_in=(all_in[0], all_in[1]),
        folded=(folded[0], folded[1]),
        needs_action=(needs[0], needs[1]),
        hand_over=False,
    )
    if needs[other] and not all_in[other] and not folded[other] and stacks[other] > 0:
        return _checked(_replace(advanced, actor=other))
    return _end_betting_round(advanced)


def _sb_seat(dealer: int) -> int:
    return (dealer + 1) % 2


def _require_response(response: int) -> int:
    # ``True`` is an ``int``, and ``200.0 == 200``, so equality cannot
    # tell those from a real response code.
    if isinstance(response, bool) or not isinstance(response, int):
        raise ValueError(f"response must be an int, got {response!r}")
    return response


def _end_betting_round(state: HandState) -> HandState:
    """Close the street, or the hand when no further betting can happen."""

    if any(state.folded) or any(state.all_in) or state.street >= _RIVER:
        return _checked(
            _replace(state, needs_action=(False, False), hand_over=True)
        )
    return _checked(
        _replace(
            state,
            street=state.street + 1,
            street_commit=(0, 0),
            bound_b=BIG_BLIND,
            actor=_sb_seat(state.dealer),
            needs_action=(True, True),
            hand_over=False,
        )
    )


def _replace(state: HandState, **changes) -> HandState:
    data = {
        "dealer": state.dealer,
        "street": state.street,
        "stacks": state.stacks,
        "street_commit": state.street_commit,
        "pot": state.pot,
        "bound_b": state.bound_b,
        "actor": state.actor,
        "all_in": state.all_in,
        "folded": state.folded,
        "needs_action": state.needs_action,
        "hand_over": state.hand_over,
    }
    data.update(changes)
    return HandState(**data)


def _checked(state: HandState) -> HandState:
    total = state.stacks[0] + state.stacks[1] + state.pot
    if total != TOTAL_CHIPS:
        raise RuntimeError(f"chip invariant broken: {total} != {TOTAL_CHIPS}")
    if state.pot < 0 or state.stacks[0] < 0 or state.stacks[1] < 0:
        raise RuntimeError("negative chips")
    return state

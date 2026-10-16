"""Training-environment betting, in the Python 3.6 subset.

This is the chip and mask behaviour of ``alphanl.botzone.legal`` plus the
five-wide packing in ``alphanl.botzone.actions``. The zip cannot import
those modules: ``legal.py`` uses ``dataclasses`` and
``from __future__ import annotations``. The observation tests compare this
copy to those modules on replayed hands.
"""

STARTING_STACK = 20000
SMALL_BLIND = 50
BIG_BLIND = 100
TOTAL_CHIPS = STARTING_STACK * 2

FOLD = -1
ALL_IN = -2
CHECK_CALL = 0

FOLD_ACTION = 0
CHECK_CALL_ACTION = 1
RAISE_MIN_ACTION = 2
RAISE_LARGE_ACTION = 3
ALL_IN_ACTION = 4
NUM_ACTIONS = 5


class HandState(object):
    """Betting state after blinds. Stacks are indexed by seat."""

    def __init__(
        self,
        dealer,
        street,
        stacks,
        street_commit,
        pot,
        bound_b,
        actor,
        all_in,
        folded,
        needs_action,
        hand_over,
    ):
        self.dealer = dealer
        self.street = street
        self.stacks = stacks
        self.street_commit = street_commit
        self.pot = pot
        self.bound_b = bound_b
        self.actor = actor
        self.all_in = all_in
        self.folded = folded
        self.needs_action = needs_action
        self.hand_over = hand_over


def new_hand(dealer):
    """Post blinds and give the small blind the first action."""

    if dealer not in (0, 1):
        raise ValueError("dealer must be 0 or 1")
    sb = (dealer + 1) % 2
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
            street=0,
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


def to_call(state, seat=None):
    """Chips the seat must put in to match the current street bet."""

    who = state.actor if seat is None else seat
    return max(state.street_commit) - state.street_commit[who]


def safe_extra(state):
    """Smallest legal positive extra, or None when that raise does not exist."""

    if state.hand_over or any(state.all_in):
        return None
    extra = state.bound_b - state.street_commit[state.actor]
    if extra < 1:
        extra = 1
    remaining = state.stacks[state.actor]
    if extra < remaining:
        return extra
    return None


def legal_response(state):
    """Legal judge integers for the seat about to act."""

    if state.hand_over:
        return set()
    remaining = state.stacks[state.actor]
    if remaining <= 0:
        return set()
    legal = set((FOLD, ALL_IN))
    if any(state.all_in):
        return legal
    if remaining > to_call(state):
        legal.add(CHECK_CALL)
    safe = safe_extra(state)
    if safe is not None:
        legal.update(range(safe, remaining))
    return legal


def action_responses(state):
    """Response for each policy id, or None when that id is masked."""

    legal = legal_response(state)
    safe = safe_extra(state)
    large = None if safe is None else max(safe * 2, state.pot)
    candidates = (FOLD, CHECK_CALL, safe, large, ALL_IN)
    out = []
    for response in candidates:
        if response is not None and response in legal:
            out.append(response)
        else:
            out.append(None)
    return tuple(out)


def legal_action_ids(state):
    """Policy ids whose packed response passed ``legal_response``."""

    return tuple(
        action_id
        for action_id, response in enumerate(action_responses(state))
        if response is not None
    )


def action_response(state, action_id):
    """Packed response for one id, or None when that id is masked."""

    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError("action id must be an int")
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError("action id out of range")
    return action_responses(state)[action_id]


def apply(state, response):
    """Return the next betting state. Does not mutate ``state``."""

    if state.hand_over:
        raise ValueError("hand is over")
    if isinstance(response, bool) or not isinstance(response, int):
        raise ValueError("response must be an int")
    if response not in legal_response(state):
        raise ValueError("illegal response")

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
    if response > 0:
        bound_b = max(bound_b, 2 * chips)
    if raised:
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


def _end_betting_round(state):
    if any(state.folded) or any(state.all_in) or state.street >= 3:
        return _checked(_replace(state, needs_action=(False, False), hand_over=True))
    return _checked(
        _replace(
            state,
            street=state.street + 1,
            street_commit=(0, 0),
            bound_b=BIG_BLIND,
            actor=(state.dealer + 1) % 2,
            needs_action=(True, True),
            hand_over=False,
        )
    )


def _replace(state, **changes):
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


def _checked(state):
    total = state.stacks[0] + state.stacks[1] + state.pot
    if total != TOTAL_CHIPS:
        raise RuntimeError("chip invariant broken")
    if state.pot < 0 or state.stacks[0] < 0 or state.stacks[1] < 0:
        raise RuntimeError("negative chips")
    return state

"""M3: five policy ids pack through legal_response, mask included.

dealer = 0: seat 0 is the big blind, seat 1 is the small blind.
"""

import random

import pytest

from alphanl.botzone.actions import (
    ACTION_NAMES,
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    NUM_ACTIONS,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    action_responses,
    legal_action_ids,
    response_for_action,
)
from alphanl.botzone.legal import HandState, apply, legal_response, new_hand, safe_extra
from alphanl.botzone.table import BotzoneTable


def _flop_open(stack: int, pot: int) -> HandState:
    """Flop, no bet yet. ``stack`` is the actor; the other seat holds the rest."""

    return HandState(
        dealer=0,
        street=1,
        stacks=(stack, 40_000 - pot - stack),
        street_commit=(0, 0),
        pot=pot,
        bound_b=100,
        actor=0,
        all_in=(False, False),
        folded=(False, False),
        needs_action=(True, True),
        hand_over=False,
    )


def _assert_one_source(state: HandState) -> None:
    """Unmasked ids are exactly those whose packed int passes legality."""

    packed = action_responses(state)
    allowed = legal_response(state)
    ids = legal_action_ids(state)
    assert len(packed) == NUM_ACTIONS == len(ACTION_NAMES)
    assert ids == tuple(
        action_id for action_id, response in enumerate(packed) if response is not None
    )
    for action_id in range(NUM_ACTIONS):
        response = response_for_action(state, action_id)
        assert response == packed[action_id]
        if response is None:
            assert action_id not in ids
        else:
            assert action_id in ids
            assert response in allowed
    if allowed:
        assert response_for_action(state, FOLD_ACTION) == -1
        assert FOLD_ACTION in ids


def test_action_ids_stay_width_five():
    assert NUM_ACTIONS == 5
    assert ACTION_NAMES == (
        "fold",
        "check_call",
        "raise_min",
        "raise_large",
        "all_in",
    )
    assert (
        FOLD_ACTION,
        CHECK_CALL_ACTION,
        RAISE_MIN_ACTION,
        RAISE_LARGE_ACTION,
        ALL_IN_ACTION,
    ) == (0, 1, 2, 3, 4)


def test_hand1_sb_open_raise_min_150_large_300():
    state = new_hand(dealer=0)
    assert state.pot == 150
    assert safe_extra(state) == 150

    _assert_one_source(state)
    assert legal_action_ids(state) == (0, 1, 2, 3, 4)
    assert response_for_action(state, FOLD_ACTION) == -1
    assert response_for_action(state, CHECK_CALL_ACTION) == 0
    assert response_for_action(state, RAISE_MIN_ACTION) == 150
    assert response_for_action(state, RAISE_LARGE_ACTION) == max(150 * 2, 150) == 300
    assert response_for_action(state, ALL_IN_ACTION) == -2
    # legal_action_ids does not collapse equal responses. These two differ,
    # so both ids stay. raise_large = max(safe * 2, pot) is above safe
    # whenever safe >= 1, so the new minimum cannot make them the same bet.
    assert response_for_action(state, RAISE_LARGE_ACTION) > response_for_action(
        state, RAISE_MIN_ACTION
    )


def test_hand2_bb_safe_300_raise_large_600():
    state = apply(new_hand(dealer=0), 200)
    assert state.actor == 0
    assert state.pot == 350
    assert safe_extra(state) == 300

    _assert_one_source(state)
    assert response_for_action(state, CHECK_CALL_ACTION) == 0
    assert response_for_action(state, RAISE_MIN_ACTION) == 300
    assert response_for_action(state, RAISE_LARGE_ACTION) == max(300 * 2, 350) == 600
    assert 600 in legal_response(state)
    assert 299 not in legal_response(state)
    assert 400 in legal_response(state)


def test_hand3_flop_min_bet_100_large_is_pot():
    state = apply(apply(new_hand(dealer=0), 0), 0)
    assert state.street == 1
    assert state.pot == 200
    assert safe_extra(state) == 100

    _assert_one_source(state)
    assert response_for_action(state, CHECK_CALL_ACTION) == 0
    assert response_for_action(state, RAISE_MIN_ACTION) == 100
    assert response_for_action(state, RAISE_LARGE_ACTION) == max(100 * 2, 200) == 200


def test_hand4_all_in_leaves_only_fold_and_all_in():
    state = apply(new_hand(dealer=0), -2)
    assert state.all_in[1] is True

    _assert_one_source(state)
    assert legal_action_ids(state) == (FOLD_ACTION, ALL_IN_ACTION)
    assert response_for_action(state, FOLD_ACTION) == -1
    assert response_for_action(state, ALL_IN_ACTION) == -2
    for action_id in (CHECK_CALL_ACTION, RAISE_MIN_ACTION, RAISE_LARGE_ACTION):
        assert response_for_action(state, action_id) is None
        assert action_id not in legal_action_ids(state)
    assert action_responses(state) == (-1, None, None, None, -2)


def test_hand5_exact_stack_masks_call_and_both_raises():
    facing = apply(new_hand(dealer=0), -2)
    _assert_one_source(facing)
    assert 0 not in legal_response(facing)
    assert response_for_action(facing, CHECK_CALL_ACTION) is None

    # A 500 commit is a bet of at least 500, so round_raise >= 500 and
    # bound_b >= 1000. The old fixture left bound_b at the postflop reset
    # (100) and relied on bound A. The minimum is now 1000, which does not
    # fit in a 500 stack, so both raise ids stay masked.
    exact = HandState(
        dealer=0,
        street=1,
        stacks=(500, 10_000),
        street_commit=(0, 500),
        pot=500,
        bound_b=1000,
        actor=0,
        all_in=(False, False),
        folded=(False, False),
        needs_action=(True, False),
        hand_over=False,
    )
    assert safe_extra(exact) is None
    _assert_one_source(exact)
    assert legal_action_ids(exact) == (FOLD_ACTION, ALL_IN_ACTION)
    assert response_for_action(exact, CHECK_CALL_ACTION) is None
    assert response_for_action(exact, RAISE_MIN_ACTION) is None
    assert response_for_action(exact, RAISE_LARGE_ACTION) is None

    one_chip_more = HandState(
        dealer=0,
        street=1,
        stacks=(501, 10_000),
        street_commit=(0, 500),
        pot=500,
        bound_b=1000,
        actor=0,
        all_in=(False, False),
        folded=(False, False),
        needs_action=(True, False),
        hand_over=False,
    )
    _assert_one_source(one_chip_more)
    assert response_for_action(one_chip_more, CHECK_CALL_ACTION) == 0
    assert response_for_action(one_chip_more, RAISE_MIN_ACTION) is None
    assert response_for_action(one_chip_more, RAISE_LARGE_ACTION) is None
    assert ALL_IN_ACTION in legal_action_ids(one_chip_more)

    short = apply(new_hand(dealer=0), 19949)
    _assert_one_source(short)
    assert response_for_action(short, CHECK_CALL_ACTION) == 0
    assert response_for_action(short, RAISE_MIN_ACTION) is None
    assert response_for_action(short, RAISE_LARGE_ACTION) is None


def test_raise_large_uses_pot_when_pot_exceeds_twice_safe():
    state = _flop_open(stack=5000, pot=1000)
    assert safe_extra(state) == 100
    _assert_one_source(state)
    assert response_for_action(state, RAISE_MIN_ACTION) == 100
    assert response_for_action(state, RAISE_LARGE_ACTION) == 1000
    assert response_for_action(state, ALL_IN_ACTION) == -2


def test_raise_large_masked_when_amount_reaches_stack():
    masked = _flop_open(stack=250, pot=400)
    assert safe_extra(masked) == 100
    assert max(100 * 2, masked.pot) == 400
    assert 400 >= masked.stacks[0]
    _assert_one_source(masked)
    assert response_for_action(masked, RAISE_MIN_ACTION) == 100
    assert response_for_action(masked, RAISE_LARGE_ACTION) is None
    assert RAISE_LARGE_ACTION not in legal_action_ids(masked)
    assert 100 in legal_response(masked)
    assert 400 not in legal_response(masked)

    inside = _flop_open(stack=401, pot=400)
    _assert_one_source(inside)
    assert response_for_action(inside, RAISE_LARGE_ACTION) == 400
    assert response_for_action(inside, ALL_IN_ACTION) == -2

    exact = _flop_open(stack=400, pot=400)
    _assert_one_source(exact)
    assert response_for_action(exact, RAISE_LARGE_ACTION) is None
    assert response_for_action(exact, ALL_IN_ACTION) == -2


def test_finished_hand_masks_every_id():
    state = apply(new_hand(dealer=0), -1)
    assert state.hand_over
    _assert_one_source(state)
    assert legal_action_ids(state) == ()
    assert action_responses(state) == (None, None, None, None, None)


def test_response_for_action_rejects_unknown_id():
    state = new_hand(dealer=0)
    with pytest.raises(ValueError):
        response_for_action(state, 5)
    with pytest.raises(ValueError):
        response_for_action(state, -1)
    with pytest.raises(ValueError):
        response_for_action(state, True)


def test_table_accepts_every_packed_response():
    table = BotzoneTable(rng=random.Random(0), dealer=0)
    table.start_hand(deck=tuple(range(52)))
    rng = random.Random(1)
    acted = 0
    while acted < 40:
        if table.hand_over:
            table.start_hand()
            continue
        state = table.state
        assert state is not None
        _assert_one_source(state)
        action_id = rng.choice(legal_action_ids(state))
        response = response_for_action(state, action_id)
        assert response in table.legal_responses()
        table.act(response)
        acted += 1

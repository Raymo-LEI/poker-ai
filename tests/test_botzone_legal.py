"""Section 1.4 acceptance hands for Botzone raise bounds.

dealer = 0: seat 0 is the big blind, seat 1 is the small blind.
"""

import pytest

from alphanl.botzone.legal import (
    HandState,
    apply,
    legal_response,
    lower_bound_a,
    lower_bound_b,
    new_hand,
    safe_extra,
    to_call,
)


def test_hand1_sb_first_action_safe_raise_is_150():
    """Judge lines 299-303: 2 * 100 - 50 = 150 after the blinds."""

    state = new_hand(dealer=0)
    assert state.actor == 1
    assert to_call(state) == 50
    assert state.bound_b == 200  # 2 * round_raise, round_raise == 100

    legal = legal_response(state)
    assert -1 in legal
    assert 0 in legal
    assert -2 in legal
    assert safe_extra(state) == 150
    assert 150 in legal
    assert 149 not in legal
    # The old max(A, B) minimum is still a legal larger raise.
    assert 200 in legal
    # A sized response must be strictly less than the remaining stack.
    assert state.stacks[1] - 1 in legal
    assert state.stacks[1] not in legal

    opened = apply(state, 150)
    assert state.street_commit[1] == 50
    assert state.pot == 150
    assert opened.street_commit[1] == 200
    assert opened.pot == 300
    assert opened.bound_b == 300  # 2 * max(100, 150)
    assert opened.actor == 0

    after = apply(state, 200)
    assert after.street_commit[1] == 250
    assert after.pot == 350
    assert after.bound_b == 400  # 2 * max(100, 200)
    assert lower_bound_b(after) == 400
    assert after.actor == 0


def test_hand2_bb_min_is_judge_formula_not_bound_a():
    """After SB raises 200, BB's minimum is 2 * 200 - 100 = 300.

    The old gate was max(A, B) = max(2 * 250, 400) = 500, so 400 was
    rejected. Bound A is still 500; it is no longer the minimum.
    """

    state = apply(new_hand(dealer=0), 200)
    assert state.actor == 0
    assert to_call(state) == 150

    legal = legal_response(state)
    assert 0 in legal
    assert lower_bound_a(state) == 500
    assert lower_bound_b(state) == 400
    assert safe_extra(state) == 300
    assert 300 in legal
    assert 299 not in legal
    assert 400 in legal
    assert 500 in legal

    after = apply(state, 300)
    assert after.street_commit[0] == 400
    assert after.pot == 650
    assert lower_bound_b(after) == 600  # 2 * max(200, 300)
    with pytest.raises(ValueError):
        apply(state, 299)


def test_hand3_flop_min_bet_is_100_not_200():
    state = apply(apply(new_hand(dealer=0), 0), 0)
    assert state.street == 1
    assert state.actor == 1
    assert state.street_commit == (0, 0)
    assert to_call(state) == 0
    assert lower_bound_a(state) == 100
    assert lower_bound_b(state) == 100

    legal = legal_response(state)
    assert -1 in legal
    assert 0 in legal
    positives = [response for response in legal if response > 0]
    assert min(positives) == 100
    assert 100 in legal
    assert 99 not in legal


def test_hand4_all_in_leaves_only_fold_and_all_in():
    state = apply(new_hand(dealer=0), -2)
    assert state.all_in[1] is True
    assert state.actor == 0
    assert state.hand_over is False

    legal = legal_response(state)
    assert legal == {-1, -2}
    assert 0 not in legal
    assert all(response not in legal for response in (1, 100, 200, 400, 500, 19900))
    with pytest.raises(ValueError):
        apply(state, 0)


def test_hand5_call_requires_stack_strictly_greater_than_owed():
    facing = apply(new_hand(dealer=0), -2)
    owed = to_call(facing)
    assert owed == facing.stacks[facing.actor]
    assert owed > 0

    legal = legal_response(facing)
    assert 0 not in legal
    assert -2 in legal
    assert -1 in legal

    # Nobody is all-in and the opponent still has chips behind. ``0`` is
    # illegal only because the stack is not strictly greater than the call.
    # Facing a 500 bet: round_raise is at least 500 (bound_b 1000). The
    # call does not fit in a 500 stack. The minimum raise does not either.
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
    assert to_call(exact) == exact.stacks[exact.actor] == 500
    assert 0 not in legal_response(exact)
    assert -2 in legal_response(exact)
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
    assert 0 in legal_response(one_chip_more)

    # One chip short of all-in: the big blind's stack is exactly one more
    # than the call, so check/call is legal again.
    short = apply(new_hand(dealer=0), 19949)
    assert short.all_in == (False, False)
    assert to_call(short) + 1 == short.stacks[short.actor]
    assert 0 in legal_response(short)


def test_call_does_not_change_round_raise_and_postflop_resets_to_50():
    called = apply(new_hand(dealer=0), 0)
    assert called.bound_b == 200
    flop = apply(called, 0)
    assert flop.street == 1
    assert flop.street_commit == (0, 0)
    assert flop.bound_b == 100  # 2 * (big_blind // 2)
    assert safe_extra(flop) == 100


def test_all_in_does_not_change_round_raise():
    """Judge lines 285-290 set round_bet to -2 and leave round_raise alone."""

    state = apply(new_hand(dealer=0), -2)
    assert state.all_in[1] is True
    assert state.bound_b == 200
    assert legal_response(state) == {-1, -2}


def test_positive_bet_updates_round_raise_without_a_new_street_max():
    """A legal extra above round_raise still updates it when the total does not.

    Shape taken from a judge-reachable window: own 421, street max 723,
    round_raise 209. Bet 210 stays under 723 and becomes the new max bet.
    """

    state = HandState(
        dealer=0,
        street=0,
        stacks=(20000 - 723, 20000 - 421),
        street_commit=(723, 421),
        pot=723 + 421,
        bound_b=2 * 209,
        actor=1,
        all_in=(False, False),
        folded=(False, False),
        needs_action=(True, True),
        hand_over=False,
    )
    assert safe_extra(state) == 1
    assert 210 in legal_response(state)
    after = apply(state, 210)
    assert after.street == 0
    assert after.street_commit[1] == 631
    assert max(after.street_commit) == 723
    assert after.bound_b == 2 * 210


def test_undercall_minimum_is_three_after_the_scripted_raises():
    """Judge sequence 0, 100, 101, 102. Minimum extra is 2 * 102 - 201 = 3."""

    state = new_hand(dealer=0)
    state = apply(state, 0)
    state = apply(state, 100)
    state = apply(state, 101)
    state = apply(state, 102)
    assert state.actor == 1
    assert state.street_commit[1] == 201
    assert state.bound_b == 2 * 102
    assert to_call(state) == 101
    assert safe_extra(state) == 3
    assert 3 in legal_response(state)
    assert 2 not in legal_response(state)
    assert 0 in legal_response(state)

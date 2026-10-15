"""Section 1.4 acceptance hands for Botzone raise bounds.

dealer = 0: seat 0 is the big blind, seat 1 is the small blind.
"""

import random

import pytest

from alphanl.botzone.legal import (
    HandState,
    apply,
    is_legal,
    legal_response,
    lower_bound_a,
    lower_bound_b,
    new_hand,
    safe_extra,
    to_call,
)


def test_hand1_sb_first_action_safe_raise_is_200():
    state = new_hand(dealer=0)
    assert state.actor == 1
    assert to_call(state) == 50

    legal = legal_response(state)
    assert -1 in legal
    assert 0 in legal
    assert -2 in legal
    assert safe_extra(state) == 200
    assert 200 in legal
    assert 199 not in legal
    # A sized response must be strictly less than the remaining stack.
    assert state.stacks[1] - 1 in legal
    assert state.stacks[1] not in legal

    after = apply(state, 200)
    assert state.street_commit[1] == 50
    assert state.pot == 150
    assert after.street_commit[1] == 250
    assert after.pot == 350
    assert after.bound_b == 400
    assert lower_bound_b(after) == 400
    assert after.actor == 0


def test_hand2_bb_emits_stricter_bound_500_not_400():
    state = apply(new_hand(dealer=0), 200)
    assert state.actor == 0
    assert to_call(state) == 150

    legal = legal_response(state)
    assert 0 in legal
    assert lower_bound_a(state) == 500
    assert lower_bound_b(state) == 400
    assert safe_extra(state) == 500
    assert 500 in legal
    assert 400 not in legal
    assert 499 not in legal

    after = apply(state, 500)
    assert after.street_commit[0] == 600
    assert after.pot == 850
    assert lower_bound_b(after) == 1000
    with pytest.raises(ValueError):
        apply(state, 400)


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
    exact = HandState(
        dealer=0,
        street=1,
        stacks=(500, 10_000),
        street_commit=(0, 500),
        pot=500,
        bound_b=100,
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
        bound_b=100,
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


def test_is_legal_matches_legal_response_on_random_hands():
    rng = random.Random(7)
    checked = 0
    for hand in range(300):
        state = new_hand(dealer=hand % 2)
        while not state.hand_over:
            legal = legal_response(state)
            remaining = state.stacks[state.actor]
            safe = safe_extra(state) or 0
            probes = {-3, -2, -1, 0, 1, safe - 1, safe, safe + 1, remaining - 1,
                      remaining, remaining + 1, to_call(state), rng.randrange(1, 40001)}
            for response in probes:
                assert is_legal(state, response) == (response in legal), (state, response)
                checked += 1
            choices = sorted(legal)
            pick = rng.choice([-1, -2, 0, safe, rng.choice(choices)])
            state = apply(state, pick if pick in legal else rng.choice(choices))
        assert not is_legal(state, -1)
    assert checked > 3000

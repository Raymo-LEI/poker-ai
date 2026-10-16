"""Harness tests only. They do not exercise a changed trainer.

The oracle is the unmodified judge in tests/fixtures/botzone_judge.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "eval"))

from botzone_referee_adapter import (  # noqa: E402
    new_dealt_holdem,
    referee_is_legal,
    spec_is_legal,
    spec_label,
    state_from_holdem,
)
from botzone_referee_diff import (  # noqa: E402
    replay_fixture,
    run_fuzz,
    scenario_tables,
    undercall_sequence,
)

from alphanl.botzone.actions import response_for_action  # noqa: E402
from alphanl.botzone.legal import apply, legal_response, new_hand, safe_extra  # noqa: E402


def test_spec_checker_is_labelled_as_not_the_referee():
    assert "NOT the real referee" in spec_label()


def test_opening_raise_150_is_the_shared_minimum():
    game = new_dealt_holdem(0)
    state = state_from_holdem(game)
    assert state.actor == 1
    assert state.round_raise == 100
    assert referee_is_legal(state, 150)
    assert referee_is_legal(state, 150.0)
    assert not referee_is_legal(state, 149)
    assert not referee_is_legal(state, 150.5)
    assert not referee_is_legal(state, True)

    ours = new_hand(0)
    assert safe_extra(ours) == 150
    assert 149 not in legal_response(ours)
    assert 150 in legal_response(ours)
    assert 200 in legal_response(ours)
    assert response_for_action(ours, 2) == 150
    assert referee_is_legal(state, response_for_action(ours, 2))
    assert referee_is_legal(state, response_for_action(ours, 3))


def test_postflop_non_dealer_acts_first_and_min_bet_is_100():
    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    state = state_from_holdem(game)
    assert state.street == 1
    assert state.actor == 1  # dealer 0, non-dealer acts first
    assert state.round_raise == 50
    assert referee_is_legal(state, 100)
    assert not referee_is_legal(state, 99)

    ours = apply(apply(new_hand(0), 0), 0)
    assert ours.street == 1 and ours.actor == 1
    assert safe_extra(ours) == 100


def test_undercall_raise_is_accepted_by_the_judge_and_by_us():
    state = undercall_sequence()
    assert state.round_bet == 302
    assert state.round_raise == 102
    assert state.own_round_bet == 201
    assert referee_is_legal(state, 3)
    assert not referee_is_legal(state, 2)
    assert 3 + state.own_round_bet < state.round_bet

    ours = new_hand(0)
    for bet in (0, 100, 101, 102):
        ours = apply(ours, bet)
    assert safe_extra(ours) == 3
    assert 3 in legal_response(ours)
    assert 2 not in legal_response(ours)


def test_street_total_spec_disagrees_at_a_zero_bet_postflop():
    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    state = state_from_holdem(game)
    assert spec_is_legal(state, 100, "action_amount")
    assert referee_is_legal(state, 100)
    # Largest street total is 0 here, so that reading accepts 99. The judge does not.
    assert spec_is_legal(state, 99, "street_total")
    assert not referee_is_legal(state, 99)
    assert not spec_is_legal(state, 99, "action_amount")


def test_scenario_gaps_match_the_judge_formula():
    rows = scenario_tables()
    opening = rows["preflop"][0]
    assert opening["judge_min"] == 150
    assert opening["our_safe"] == 150
    assert opening["gap"] == 0
    owner = rows["reraise"][0]
    assert owner["judge_min"] == 500
    assert owner["our_safe"] == 500
    assert owner["gap"] == 0
    post = rows["postflop"][0]
    assert post["judge_min"] == post["our_safe"] == 100
    under = rows["reraise"][1]
    assert under["judge_min"] == under["our_safe"] == 3
    for bucket in rows.values():
        for row in bucket:
            assert row["our_safe"] == row["judge_min"]
            assert row["gap"] == 0


def test_fixture_replay_whole_hands_match():
    """The 41 raises that used to sit under the old safe now replay cleanly."""

    report = replay_fixture()
    assert report["hands"] == 2050
    assert report["matches"] == 41
    assert report["decisions"] == 7785
    assert report["order_mismatches"] == 0
    assert report["judge_rejected_real_actions"] == 0
    assert report["mapped_illegal"] == 0
    assert report["our_rejected_real_actions"] == 0
    assert report["our_rejected_patterns"] == []
    assert report["engine_mismatches"] == 0


def test_fuzz_2000_hands_never_sends_a_judge_illegal_bet():
    summary = run_fuzz(2000, seed=7, explore_hands=400, use_policy=True)
    assert summary["we_legal_judge_illegal"] == 0
    assert summary["mapped_raise_illegal"] == 0
    assert summary["desyncs"] == 0
    assert summary["min_raise_mismatch"] == 0
    assert summary["raise_min_and_raise_large_always_judge_legal"] is True
    assert summary["spec_action_amount_disagree"] == 0
    assert summary["hands_our_action_set"] == 2000

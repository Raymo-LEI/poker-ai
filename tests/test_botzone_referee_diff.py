"""Harness tests only. They do not exercise a changed trainer.

The oracle is the unmodified judge in tests/fixtures/botzone_judge.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def test_opening_raise_150_is_judge_legal_and_our_minimum_is_200():
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
    assert safe_extra(ours) == 200
    assert 150 not in legal_response(ours)
    assert 200 in legal_response(ours)
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


def test_undercall_raise_is_accepted_by_the_judge():
    state = undercall_sequence()
    assert state.round_bet == 302
    assert state.round_raise == 102
    assert state.own_round_bet == 201
    assert referee_is_legal(state, 3)
    assert not referee_is_legal(state, 2)
    assert 3 + state.own_round_bet < state.round_bet


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
    assert opening["our_safe"] == 200
    assert opening["gap"] == 50
    owner = rows["reraise"][0]
    assert owner["judge_min"] == 500
    assert owner["our_safe"] == 600
    assert owner["gap"] == 100
    post = rows["postflop"][0]
    assert post["judge_min"] == post["our_safe"] == 100
    for bucket in rows.values():
        for row in bucket:
            assert row["our_safe"] >= row["judge_min"]


# Acceptance "replay of 41 real matches with 0 mismatch" FAILED.
# Each row is a real raise the judge accepts (response >= 2*round_raise - own)
# and our legal.py safe rejects (response < safe). Expected until T-M14b.
# Do not delete rows or lower the counts to make the replay look clean.
# count, street, response, our_safe, judge_min, to_call
EXPECTED_STRICTER_MIN_RAISES = (
    (15, 0, 150, 200, 150, 50),
    (5, 0, 400, 500, 300, 150),
    (3, 0, 800, 900, 700, 350),
    (2, 0, 450, 500, 300, 150),
    (2, 0, 900, 1100, 650, 300),
    (2, 0, 820, 920, 720, 360),
    (2, 0, 410, 500, 300, 150),
    (1, 0, 100, 200, 100, 0),
    (1, 0, 3320, 3520, 2870, 1310),
    (1, 0, 1600, 1800, 1350, 650),
    (1, 1, 14600, 16040, 12899, 6319),
    (1, 0, 1800, 2300, 1250, 600),
    (1, 1, 4210, 4720, 2940, 1380),
    (1, 1, 8420, 10380, 6060, 2830),
    (1, 0, 1700, 1800, 1150, 450),
    (1, 0, 870, 1000, 550, 250),
    (1, 0, 870, 920, 720, 360),
)


def test_fixture_replay_records_expected_stricter_min_raises():
    report = replay_fixture()
    assert report["hands"] == 2050
    assert report["matches"] == 41
    assert report["decisions"] == 7785
    assert report["order_mismatches"] == 0
    assert report["judge_rejected_real_actions"] == 0
    assert report["mapped_illegal"] == 0
    patterns = tuple(
        (
            row["count"],
            row["street"],
            row["response"],
            row["our_safe"],
            row["judge_min"],
            row["to_call"],
        )
        for row in report["our_rejected_patterns"]
    )

    def _order(row: tuple) -> tuple:
        count, street, response, our_safe, _judge_min, _call = row
        return (-count, street, response, our_safe)

    assert tuple(sorted(patterns, key=_order)) == tuple(
        sorted(EXPECTED_STRICTER_MIN_RAISES, key=_order)
    )
    assert sum(row[0] for row in patterns) == 41
    assert report["our_rejected_real_actions"] == 41
    for count, _street, response, our_safe, judge_min, call_amount in patterns:
        assert count >= 1
        assert judge_min <= response < our_safe
        assert response > call_amount


def test_fuzz_2000_hands_never_sends_a_judge_illegal_bet():
    summary = run_fuzz(2000, seed=7, explore_hands=400, use_policy=True)
    assert summary["we_legal_judge_illegal"] == 0
    assert summary["mapped_raise_illegal"] == 0
    assert summary["desyncs"] == 0
    assert summary["raise_min_and_raise_large_always_judge_legal"] is True
    assert summary["spec_action_amount_disagree"] == 0
    assert summary["conservative_decisions"] > 0
    assert summary["hands_our_action_set"] == 2000

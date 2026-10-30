"""Branchable kernel against the unmodified Botzone referee."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from alphanl.botzone.actions import ALL_IN_ACTION, RAISE_LARGE_ACTION, RAISE_MIN_ACTION
from alphanl.botzone.kernel import POLICY_FEATURE_NAMES, BranchableKernel, LegacyFiveActionBridge
from alphanl.botzone.legal import apply, new_hand

_JUDGE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "botzone_judge.py"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"


def _deck(seed: int = 0):
    cards = list(range(52))
    random.Random(seed).shuffle(cards)
    return cards


def test_judge_fixture_hash_is_unchanged():
    assert hashlib.sha256(_JUDGE.read_bytes()).hexdigest() == _JUDGE_SHA


def test_opening_matches_a_separately_constructed_judge():
    import random as py_random
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from botzone_referee_adapter import JUDGE

    deck = _deck(1)
    kernel = BranchableKernel.start_hand(deck, dealer=0)
    state = py_random.getstate()
    try:
        game = JUDGE.Holdem([20000, 20000], dealer_idx=0)
        game.set_deck_array(list(deck))
        game.deal_cards_and_blind()
    finally:
        py_random.setstate(state)
    audit = kernel.referee_audit_state()
    assert audit["actor"] == 1
    assert audit["dealer"] == 0
    assert audit["street"] == 0
    assert audit["pot"] == 150
    assert audit["chips"] == (19900, 19950)
    assert audit["round_raise"] == 100
    assert audit["round_player_bet"] == (100, 50)
    assert audit["holes"] == (
        tuple(card.to_int() for card in game.player_cards[0]),
        tuple(card.to_int() for card in game.player_cards[1]),
    )
    assert audit["deck"] == tuple(card.to_int() for card in game.deck)
    assert kernel.min_positive_extra() == 150
    assert kernel.is_legal(150)
    assert kernel.is_legal(150.0)
    assert not kernel.is_legal(149)
    assert not kernel.is_legal(150.5)
    assert not kernel.is_legal(True)


def test_illegal_response_does_not_change_state():
    kernel = BranchableKernel.start_hand(_deck(2), dealer=0)
    before = kernel.referee_audit_state()
    with pytest.raises(ValueError):
        kernel.step(149)
    assert kernel.referee_audit_state() == before


def test_undercall_stays_on_the_street_the_judge_keeps_open():
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    for bet in (0, 100, 101, 102):
        kernel.step(bet)
    assert kernel.min_positive_extra() == 3
    assert kernel.is_legal(3)
    assert not kernel.is_legal(2)
    kernel.step(3)
    assert kernel.street == 0
    assert kernel.actor == 0
    assert not kernel.hand_over
    # BB already has 302 in. The new minimum extra is 1, and the street stays open.
    assert kernel.min_positive_extra() == 1

    state = new_hand(0)
    for bet in (0, 100, 101, 102, 3):
        state = apply(state, bet)
    assert state.street == 1


def test_clone_branches_do_not_share_mutations():
    kernel = BranchableKernel.start_hand(_deck(3), dealer=0)
    other = kernel.branch()
    assert other.referee_audit_state() == kernel.referee_audit_state()
    kernel.step(150)
    assert other.pot == 150
    assert other.actions_this_hand == 0
    assert kernel.pot == 300
    other.step(0)
    assert kernel.actions_this_hand == 1
    assert other.actions_this_hand == 1
    assert kernel.referee_audit_state()["history"] != other.referee_audit_state()["history"]

    fresh = BranchableKernel.start_hand(_deck(4), dealer=0)
    twin = fresh.clone()
    twin.set_deal(_deck(5), dealer=1)
    assert fresh.dealer == 0
    assert twin.dealer == 1
    assert fresh.referee_audit_state()["holes"] != twin.referee_audit_state()["holes"]


def test_policy_view_hides_opponent_holes_and_undealt_cards():
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    audit = kernel.referee_audit_state()
    view = kernel.policy_view()
    assert view["seat"] == 1
    assert view["hole_cards"] == list(audit["holes"][1])
    assert view["public_cards"] == []
    hidden = set(audit["holes"][0]) | set(audit["deck"])
    assert hidden.isdisjoint(view["hole_cards"])
    assert hidden.isdisjoint(view["public_cards"])
    assert "deck" not in view
    kernel.step(0)
    kernel.step(0)
    flop = kernel.policy_view()
    assert len(flop["public_cards"]) == 3
    still_hidden = set(kernel.referee_audit_state()["deck"])
    assert still_hidden.isdisjoint(flop["public_cards"])
    assert still_hidden.isdisjoint(flop["hole_cards"])
    assert flop["feature_names"] == list(POLICY_FEATURE_NAMES)
    assert len(flop["feature_vector"]) == len(POLICY_FEATURE_NAMES)
    assert flop["features"]["pot"] == kernel.pot
    assert flop["action_events"][-1]["action"] == 0


def test_legacy_bridge_sends_chip_responses_not_action_ids():
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    bridge = LegacyFiveActionBridge(kernel)
    assert bridge.response_for(RAISE_MIN_ACTION) == 150
    assert bridge.response_for(ALL_IN_ACTION) == -2
    assert 150 != RAISE_MIN_ACTION
    assert -2 != ALL_IN_ACTION
    assert not kernel.is_legal(RAISE_MIN_ACTION)
    assert not kernel.is_legal(ALL_IN_ACTION)
    other = bridge.clone()
    stepped = bridge.step(RAISE_MIN_ACTION)
    assert stepped.response == 150
    assert kernel.pot == 300
    assert other.kernel.pot == 150
    assert other.response_for(RAISE_MIN_ACTION) == 150

    facing = BranchableKernel.start_hand(list(range(52)), dealer=0)
    facing.step(-2)
    mask = LegacyFiveActionBridge(facing)
    assert mask.legal_ids() == (0, ALL_IN_ACTION)
    assert mask.response_for(RAISE_MIN_ACTION) is None
    assert mask.response_for(RAISE_LARGE_ACTION) is None
    with pytest.raises(ValueError):
        mask.step(RAISE_MIN_ACTION)
    observed = mask.observation()
    assert observed["legacy_action_ids"]["chip_responses"][ALL_IN_ACTION] == -2
    assert observed["hole_cards"] == list(facing.referee_audit_state()["holes"][facing.actor])


def test_showdown_payoff_and_chip_conservation():
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    kernel.step(-2)
    result = kernel.step(-2)
    assert result.hand_over
    assert kernel.public_cards and len(kernel.public_cards) == 5
    assert kernel.conserved()
    assert kernel.settled_chips is not None
    assert kernel.settled_chips[0] + kernel.settled_chips[1] == 40000
    assert kernel.payoffs[0] + kernel.payoffs[1] == 0
    # This deck chops the board, so each seat is returned to 20000.
    assert kernel.payoffs == (0.0, 0.0)
    assert kernel.winners == (0, 1)


def test_fifty_hand_score_alternates_the_button_and_resets_stacks():
    decks = [_deck(1000 + index) for index in range(50)]
    kernel = BranchableKernel.start_match(decks, initial_dealer=0, max_hands=50)
    chips = [0.0, 0.0]
    for hand in range(50):
        assert kernel.dealer == hand % 2
        # Blinds are already posted: the dealer (big blind) has 19900.
        if kernel.dealer == 0:
            assert kernel.stacks == (19900, 19950)
        else:
            assert kernel.stacks == (19950, 19900)
        assert kernel.pot == 150
        kernel.step(-1)
        assert kernel.hand_over
        assert kernel.conserved()
        assert kernel.payoffs[0] + kernel.payoffs[1] == 0
        chips[0] += kernel.payoffs[0]
        chips[1] += kernel.payoffs[1]
        if hand < 49:
            assert not kernel.match_over
            kernel.next_hand()
        else:
            assert kernel.match_over
            with pytest.raises(ValueError):
                kernel.next_hand()
    assert kernel.scores == (chips[0] / 100, chips[1] / 100)
    assert kernel.scores == (0.0, 0.0)


def test_set_match_deals_before_the_first_action():
    first = [_deck(1), _deck(2)]
    kernel = BranchableKernel.start_match(first, initial_dealer=0)
    holes = kernel.referee_audit_state()["holes"]
    kernel.set_match_deals([_deck(3), _deck(4)], initial_dealer=1)
    assert kernel.dealer == 1
    assert kernel.max_hands == 2
    assert kernel.referee_audit_state()["holes"] != holes
    kernel.step(-1)
    with pytest.raises(ValueError):
        kernel.set_match_deals([_deck(5)], initial_dealer=0)


def test_fixed_deal_replays_the_same_runout_on_two_branches():
    kernel = BranchableKernel.start_hand(_deck(9), dealer=1)
    kernel.step(0)
    kernel.step(0)
    left = kernel.clone()
    right = kernel.clone()
    for _ in range(6):
        left.step(0)
        right.step(0)
    assert left.hand_over and right.hand_over
    assert left.public_cards == right.public_cards
    assert left.payoffs == right.payoffs
    assert len(left.public_cards) == 5


def test_boundary_and_short_trajectories_match_the_judge():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import botzone_kernel_diff as diff

    boundary = diff.run_boundary(80, seed=7)
    trajectories = diff.run_trajectories(20, seed=7, match_length=5)
    assert boundary["states"] >= 80
    assert boundary["legal_reject_mismatches"] == 0
    assert boundary["min_positive_mismatches"] == 0
    assert boundary["step_mismatches"] == 0
    assert boundary["policy_view_leaks"] == 0
    assert boundary["conservation_failures"] == 0
    assert trajectories["hands"] == 20
    assert trajectories["showdown_payoff_mismatches"] == 0
    assert trajectories["match_score_mismatches"] == 0
    assert trajectories["clone_isolation_failures"] == 0
    assert trajectories["legacy_bridge_mismatches"] == 0


def test_replay_fixture_is_step_identical():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import botzone_kernel_diff as diff

    report = diff.run_replay()
    assert report["matches"] == 41
    assert report["hands"] == 2050
    assert report["decisions"] == 7785
    assert report["mac_33_match_tree"] == "not_attached"
    for key in (
        "step_mismatches",
        "legal_reject_mismatches",
        "payoff_mismatches",
        "showdown_payoff_mismatches",
        "conservation_failures",
        "policy_view_leaks",
        "legacy_bridge_mismatches",
        "match_score_mismatches",
        "clone_isolation_failures",
    ):
        assert report[key] == 0, key

"""Standard 200BB CFR rules, exact off-tree history, and lookup settlement."""

import random
import copy

import pytest
import torch

pytest.importorskip("pokerkit")
pytest.importorskip("phevaluator")

from alphanl.transformer.buckets import nearest_pot_fraction_bucket
from alphanl.transformer.hunl import HunlTable, HunlTokenizer, hand_rank
from alphanl.transformer.tables import PokerkitTable, aligned_decks
from alphanl.transformer.tokenizer import BettingHistoryTokenizer, TYPE_ACTION, VOCAB_SIZE


def test_200bb_blinds_and_all_street_action_orders():
    table = HunlTable(seed=7)
    table.reset()
    assert table.starting_stacks == (20_000, 20_000)
    assert table.big_blind == 100 and table.small_blind == 50
    assert tuple(table.state.bets) == (100, 50)
    assert tuple(table.state.stacks) == (19_900, 19_950)
    assert table.sb_seat == table.current_player() == 1
    table.step(1)  # SB completes.
    assert table.current_player() == 0
    table.step(1)  # BB exercises the option to check.
    for street in (1, 2, 3):
        assert table.street == street
        assert table.current_player() == 0  # BB opens postflop.
        assert 0 not in table.legal_action_ids()  # No folding a free check.
        table.step(1)
        assert table.current_player() == 1  # SB/button acts last.
        table.step(1)
    assert table.is_over()
    assert sum(table.payoffs()) == 0
    assert table.chips_in_play() == 40_000


def test_off_tree_raise_preserves_exact_unclipped_features():
    table = HunlTable(seed=9)
    table.reset()
    table.step_raise_to(12_345)
    snapshot = table.snapshot(table.current_player())
    event = snapshot.events[-1]
    assert event.action_id is None
    assert event.chips_added == 12_295
    assert event.pot_before == 150
    assert event.pot_fraction == pytest.approx(12_295 / 150)
    assert event.to_call_before == 50
    assert snapshot.to_call == 12_245
    assert event.size_bucket == 8
    exact = HunlTokenizer().encode(snapshot)
    legacy = BettingHistoryTokenizer().encode(snapshot)
    at = exact["type_ids"] == TYPE_ACTION
    assert exact["numeric"][at][0, 0].item() == pytest.approx(event.pot_fraction / 10)
    assert legacy["numeric"][at][0, 0].item() == 1
    torch.testing.assert_close(exact["numeric"][at][:, 1:], legacy["numeric"][at][:, 1:])
    assert exact["token_ids"].max().item() < VOCAB_SIZE == 275
    assert exact["numeric"].shape[-1] == 4


def test_nearest_buckets_use_existing_ids_and_tie_rule():
    assert nearest_pot_fraction_bucket(0) == 0
    assert nearest_pot_fraction_bucket(0.001) == 1
    assert nearest_pot_fraction_bucket(0.62) == 2
    assert nearest_pot_fraction_bucket(0.625) == 2
    assert nearest_pot_fraction_bucket(0.63) == 3
    assert nearest_pot_fraction_bucket(100) == 8


def test_clone_preserves_sampled_deck_and_isolates_branches():
    table = HunlTable(seed=123)
    table.reset()
    table.step(1)
    copy = table.clone()
    assert copy.snapshot(0) == table.snapshot(0)
    copy.step(3)
    assert table.current_player() == 0 and len(table.snapshot(0).events) == 1
    assert copy.current_player() == 1 and len(copy.snapshot(1).events) == 2
    # Reset RNG state is independently cloned too.
    table.reset()
    copy.reset()
    assert copy.hole_cards(0) == table.hole_cards(0)
    assert copy.hole_cards(1) == table.hole_cards(1)


def test_light_clone_matches_deepcopy_at_every_branch_and_street():
    chooser = random.Random(711)
    streets = set()
    for seed in range(40):
        table = HunlTable(seed=seed)
        table.reset()
        while not table.is_over():
            streets.add(table.street)
            before = copy.deepcopy(table.state_dict())
            for action in table.legal_action_ids():
                light, oracle = table.clone(), copy.deepcopy(table)
                light.step(action)
                oracle.step(action)
                assert vars(light.state) == vars(oracle.state)
                assert light._records == oracle._records
                assert light.rng_state() == oracle.rng_state()
                assert table.state_dict() == before
                if not light.is_over():
                    assert light.snapshot(0) == oracle.snapshot(0)
                    assert light.snapshot(1) == oracle.snapshot(1)
                else:
                    assert light.payoffs() == oracle.payoffs()
            # Include complete check-downs as well as random betting/all-ins.
            table.step(1 if seed < 8 else chooser.choice(table.legal_action_ids()))
    assert streets == {0, 1, 2, 3}


def test_clone_reset_rng_is_independent_in_both_directions():
    parent = HunlTable(seed=10)
    parent.reset()
    sibling = parent.clone()
    oracle = copy.deepcopy(parent)
    parent.reset()
    parent.reset()
    sibling.reset()
    oracle.reset()
    assert vars(sibling.state) == vars(oracle.state)
    assert sibling.rng_state() == oracle.rng_state()


def test_packed_hunl_observations_match_original_padding_and_own_storage():
    table = HunlTable(seed=93)
    table.reset()
    for length in (8, 32, 64):
        tokenizer = HunlTokenizer(length)
        while not table.is_over():
            snapshot = table.snapshot(table.current_player())
            sequences = [tokenizer._tokens(snapshot)]
            original = BettingHistoryTokenizer._pad(tokenizer, sequences)
            packed = tokenizer._pad(sequences)
            assert all(torch.equal(packed[key], original[key]) for key in original)
            before = {key: value.clone() for key, value in packed.items()}
            tokenizer.encode(snapshot)
            assert all(torch.equal(packed[key], before[key]) for key in packed)
            table.step(1)
        table.reset()


def test_checkpoint_round_trip_and_rule_validation():
    table = HunlTable(seed=11)
    table.reset()
    table.step_raise_to(777)
    blob = table.state_dict()
    loaded = HunlTable(seed=100)
    loaded.load_state_dict(blob)
    assert loaded.snapshot(0) == table.snapshot(0)
    assert loaded.state.hand_types == table.state.hand_types
    with pytest.raises(ValueError, match="evaluator"):
        HunlTable(seed=11, fast_evaluator=False).load_state_dict(blob)
    old = PokerkitTable(seed=1, small_blind=50, big_blind=100, starting_stack=20_000)
    old.reset()
    with pytest.raises(ValueError, match="standard 200BB"):
        loaded.load_state_dict(old.state_dict())


def test_lookup_ranking_strength_ties_and_wheel():
    assert hand_rank(("SA", "SK", "SQ", "SJ", "ST", "H2", "D3")) > hand_rank(
        ("SA", "HA", "DA", "CA", "SK", "H2", "D3")
    )
    assert hand_rank(("SA", "H2", "D3", "C4", "S5", "H8", "D9")) > hand_rank(
        ("SA", "HA", "D3", "C4", "S6", "H8", "D9")
    )
    assert hand_rank(("SA", "SK", "SQ", "SJ", "ST", "H2", "D3")) == hand_rank(
        ("SA", "SK", "SQ", "SJ", "ST", "H4", "D5")
    )


def test_native_lookup_is_used_instead_of_pokerkit_combination_loop(monkeypatch):
    from pokerkit import StandardHighHand

    def fail(*args, **kwargs):
        raise AssertionError("slow combination evaluator was called")

    monkeypatch.setattr(StandardHighHand, "from_game", fail)
    table = HunlTable(seed=77)
    table.reset()
    table.step(4)
    table.step(1)
    assert table.is_over() and sum(table.payoffs()) == 0
    assert table.chips_in_play() == 40_000


def test_lookup_matches_legacy_settlement_for_random_games():
    chooser = random.Random(4)
    for seed in range(24):
        fast = HunlTable(seed=seed)
        slow = HunlTable(seed=seed, fast_evaluator=False)
        fast.reset()
        slow.reset()
        while not fast.is_over():
            assert fast.current_player() == slow.current_player()
            assert fast.legal_action_ids() == slow.legal_action_ids()
            action = chooser.choice(fast.legal_action_ids())
            fast.step(action)
            slow.step(action)
        assert slow.is_over()
        assert fast.payoffs() == slow.payoffs()
        assert fast.chips_in_play() == slow.chips_in_play() == 40_000


def test_board_plays_chop_allin_and_raise_limits():
    _, deck = aligned_decks(("H2", "D3"), ("H4", "D5"), ("SA", "SK", "SQ", "SJ", "ST"))
    table = HunlTable(seed=1)
    table.reset(deck)
    with pytest.raises(ValueError):
        table.step_raise_to(199)
    with pytest.raises(ValueError):
        table.step_raise_to(200.5)
    with pytest.raises(ValueError):
        table.step_raise_to(20_001)
    table.step_raise_to(20_000)
    assert table.events_for(0)[-1].action_type == "all_in"
    table.step(1)
    assert table.payoffs() == (0, 0)
    assert table.chips_in_play() == 40_000

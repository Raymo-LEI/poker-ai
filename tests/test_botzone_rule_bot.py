"""M7: Botzone rule-bot priorities and the bb/100 interval.

Priorities are design §5.3. Strength cut points are RuleBot's ≥7 / ≥4 / ≥2.
"""

import itertools
import math
import random

import pytest

from alphanl.bots import RuleBot
from alphanl.botzone.actions import (
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    legal_action_ids,
    response_for_action,
)
from alphanl.botzone.legal import BIG_BLIND, to_call
from alphanl.botzone.rule_bot import (
    HIGH_PRIORITY,
    LOW_PRIORITY,
    MID_PRIORITY,
    REST_CHECK_PRIORITY,
    REST_FACING_PRIORITY,
    BotzoneRuleBot,
    action_priority,
    choose_action,
    hand_strength,
    postflop_strength,
    preflop_strength,
)
from alphanl.botzone.rule_eval import (
    bb_per_100_interval,
    match_scores,
    play_rule_vs_random,
    uniform_legal_action,
)
from alphanl.botzone.table import BotzoneTable, parse_card, preset_deck

RANKS = "23456789TJQKA"
SUITS = "hdsc"


def _cards(*names: str) -> tuple:
    return tuple(parse_card(name) for name in names)


def _rlcard(card: int) -> str:
    return SUITS[card % 4].upper() + RANKS[card // 4]


def test_priority_tables_match_the_design():
    assert HIGH_PRIORITY == (
        ALL_IN_ACTION,
        RAISE_LARGE_ACTION,
        RAISE_MIN_ACTION,
        CHECK_CALL_ACTION,
        FOLD_ACTION,
    )
    assert MID_PRIORITY == (
        RAISE_MIN_ACTION,
        RAISE_LARGE_ACTION,
        CHECK_CALL_ACTION,
        ALL_IN_ACTION,
        FOLD_ACTION,
    )
    assert LOW_PRIORITY == (
        CHECK_CALL_ACTION,
        RAISE_MIN_ACTION,
        FOLD_ACTION,
        RAISE_LARGE_ACTION,
        ALL_IN_ACTION,
    )
    assert REST_CHECK_PRIORITY[0] == CHECK_CALL_ACTION
    assert REST_FACING_PRIORITY[:2] == (FOLD_ACTION, CHECK_CALL_ACTION)


def test_preflop_points_match_rulebot_for_every_holding():
    for left, right in itertools.combinations(range(52), 2):
        expected = RuleBot._preflop_strength([_rlcard(left), _rlcard(right)])
        assert preflop_strength((left, right)) == expected


def test_pocket_aces_and_kings_are_the_high_band():
    assert preflop_strength(_cards("Ah", "Ad")) == 8
    assert preflop_strength(_cards("Kh", "Kd")) == 7
    assert preflop_strength(_cards("Qh", "Qd")) == 7
    assert preflop_strength(_cards("Jh", "Jd")) == 7
    assert action_priority(8, to_call_chips=50) is HIGH_PRIORITY
    assert action_priority(7, to_call_chips=50) is HIGH_PRIORITY
    assert action_priority(6, to_call_chips=50) is MID_PRIORITY


@pytest.mark.parametrize(
    ("names", "score"),
    [
        (("Ah", "Kd"), 6),
        (("Ah", "Kh"), 6),
        (("As", "2d"), 5),
        (("7h", "6h"), 4),
        (("2c", "2d"), 4),
        (("Kh", "Qd"), 3),
        (("6h", "5h"), 1),
        (("7c", "2d"), 1),
    ],
)
def test_preflop_examples(names, score):
    assert preflop_strength(_cards(*names)) == score


@pytest.mark.parametrize(
    ("hole", "board", "score"),
    [
        (("Ah", "Kd"), ("2c", "7s", "9d"), 1),
        (("Ah", "Ad"), ("2c", "7s", "9d"), 2),
        (("Ah", "Ad"), ("Kc", "Ks", "9d"), 3),
        (("Ah", "Ad"), ("Ac", "7s", "9d"), 4),
        (("9h", "8d"), ("7c", "6s", "5d"), 5),
        (("Ah", "3h"), ("Qh", "8h", "2h"), 6),
        (("Ah", "Ad"), ("Ac", "Kc", "Ks"), 7),
        (("Ah", "Ad"), ("Ac", "As", "9d"), 8),
        (("Ah", "Kh"), ("Qh", "Jh", "Th"), 9),
    ],
)
def test_postflop_category_uses_the_rulebot_scale(hole, board, score):
    assert postflop_strength(_cards(*hole), _cards(*board)) == score


def test_straight_and_flush_are_not_collapsed_to_the_pair_counter():
    flush = (_cards("Ah", "3h"), _cards("Qh", "8h", "2h", "9c", "4d"))
    straight = (_cards("9h", "8d"), _cards("7c", "6s", "5d", "Kc", "2h"))
    assert postflop_strength(*flush) == 6
    assert postflop_strength(*straight) == 5
    assert RuleBot._made_hand_category(
        [_rlcard(card) for card in flush[0]],
        [_rlcard(card) for card in flush[1]],
    ) == 1
    assert RuleBot._made_hand_category(
        [_rlcard(card) for card in straight[0]],
        [_rlcard(card) for card in straight[1]],
    ) == 1


def test_choose_walks_past_masked_ids():
    everything = (0, 1, 2, 3, 4)
    assert choose_action(everything, 8, 50) == ALL_IN_ACTION
    assert choose_action((0, 1, 2, 3), 8, 50) == RAISE_LARGE_ACTION
    assert choose_action((0, 1), 8, 50) == CHECK_CALL_ACTION
    assert choose_action(everything, 6, 0) == RAISE_MIN_ACTION
    assert choose_action((0, 1, 4), 6, 0) == CHECK_CALL_ACTION
    assert choose_action((0, 4), 4, 100) == ALL_IN_ACTION
    assert choose_action(everything, 2, 100) == CHECK_CALL_ACTION
    assert choose_action((0, 3, 4), 2, 100) == FOLD_ACTION
    assert choose_action((3, 4), 2, 100) == RAISE_LARGE_ACTION
    assert choose_action(everything, 1, 0) == CHECK_CALL_ACTION
    assert choose_action(everything, 1, 100) == FOLD_ACTION
    assert choose_action((1, 2, 3, 4), 1, 100) == CHECK_CALL_ACTION


def test_choose_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        choose_action((), 1, 0)


def test_aces_shove_and_trash_folds_the_blind():
    bot = BotzoneRuleBot()
    aces = BotzoneTable()
    aces.start_hand(
        0,
        preset_deck(
            0,
            (_cards("2c", "3d"), _cards("Ah", "Ad")),
            _cards("Kc", "Qs", "7d", "4h", "9c"),
        ),
    )
    assert aces.actor == 1
    assert to_call(aces.state) == 50
    assert bot.action_id(aces) == ALL_IN_ACTION

    trash = BotzoneTable()
    trash.start_hand(
        0,
        preset_deck(
            0,
            (_cards("Ah", "Kd"), _cards("7c", "2d")),
            _cards("Kc", "Qs", "4h", "9c", "3s"),
        ),
    )
    assert trash.actor == 1
    assert hand_strength(trash.holes[1], trash.board) == 1
    assert bot.action_id(trash) == FOLD_ACTION


def test_trash_checks_for_free_then_folds_to_a_bet():
    bot = BotzoneRuleBot()
    table = BotzoneTable()
    table.start_hand(
        1,
        preset_deck(
            1,
            (_cards("As", "Kd"), _cards("7c", "2d")),
            _cards("Kc", "9s", "4h", "8c", "3s"),
        ),
    )
    assert table.actor == 0
    table.act(0)
    assert table.actor == 1
    assert to_call(table.state) == 0
    assert bot.action_id(table) == CHECK_CALL_ACTION
    table.act(response_for_action(table.state, CHECK_CALL_ACTION))
    assert table.street == 1
    assert table.actor == 0
    assert RAISE_MIN_ACTION in legal_action_ids(table.state)
    table.act(response_for_action(table.state, RAISE_MIN_ACTION))
    assert table.actor == 1
    assert to_call(table.state) > 0
    assert hand_strength(table.holes[1], table.board) == 1
    assert bot.action_id(table) == FOLD_ACTION


def test_full_house_shoves_and_a_bare_flush_minraises():
    bot = BotzoneRuleBot()
    boat = BotzoneTable()
    boat.start_hand(
        0,
        preset_deck(
            0,
            (_cards("Ah", "Ad"), _cards("2c", "3d")),
            _cards("Ac", "Kc", "Ks", "4s", "5c"),
        ),
    )
    boat.act(0)
    boat.act(0)
    boat.act(0)
    assert boat.actor == 0
    assert boat.street == 1
    assert hand_strength(boat.holes[0], boat.board) == 7
    assert bot.action_id(boat) == ALL_IN_ACTION

    flush = BotzoneTable()
    flush.start_hand(
        0,
        preset_deck(
            0,
            (_cards("Ah", "3h"), _cards("2c", "3d")),
            _cards("Qh", "8h", "2h", "9c", "4d"),
        ),
    )
    flush.act(0)
    flush.act(0)
    flush.act(0)
    assert flush.actor == 0
    assert hand_strength(flush.holes[0], flush.board) == 6
    assert bot.action_id(flush) == RAISE_MIN_ACTION


def test_action_ignores_the_other_seat_hole_cards():
    bot = BotzoneRuleBot()
    boards = _cards("Kc", "9s", "4h", "8c", "3s")
    first = BotzoneTable()
    first.start_hand(1, preset_deck(1, (_cards("As", "Kd"), _cards("7c", "2d")), boards))
    first.act(0)
    second = BotzoneTable()
    second.start_hand(1, preset_deck(1, (_cards("2c", "3d"), _cards("7c", "2d")), boards))
    second.act(0)
    assert first.holes[1] == second.holes[1]
    assert first.holes[0] != second.holes[0]
    assert bot.action_id(first) == bot.action_id(second) == CHECK_CALL_ACTION


def test_bb100_interval_is_mean_plus_minus_1_96_se():
    mean, std, bb100, low, high = bb_per_100_interval([0, 200], big_blind=100)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(math.sqrt(2.0))
    assert bb100 == pytest.approx(100.0)
    assert low == pytest.approx(100.0 - 1.96 * 100.0)
    assert high == pytest.approx(100.0 + 1.96 * 100.0)

    flat = bb_per_100_interval([100, 100, 100, 100])
    assert flat[2:] == pytest.approx((100.0, 100.0, 100.0))


def test_match_score_is_fifty_hands_of_chips_over_100():
    chips = [100] * 50 + [-100] * 50 + [50]
    assert match_scores(chips) == [50.0, -50.0]
    assert match_scores([100] * 49) == []


def test_uniform_draw_stays_inside_the_five_id_mask():
    table = BotzoneTable(rng=random.Random(0))
    table.start_hand()
    rng = random.Random(1)
    seen = {uniform_legal_action(table.state, rng) for _ in range(200)}
    legal = set(legal_action_ids(table.state))
    assert seen == legal
    assert seen <= {0, 1, 2, 3, 4}


def test_short_match_alternates_the_button_and_scores_complete_blocks():
    result = play_rule_vs_random(n_hands=100, seed=7, hero_seat=0)
    assert result.dealers == tuple(index % 2 for index in range(100))
    assert result.n_matches == 2
    assert result.match_length == 50
    assert result.ci95_low <= result.R <= result.ci95_high
    assert result.big_blind == BIG_BLIND
    other = play_rule_vs_random(n_hands=100, seed=7, hero_seat=1)
    assert other.hero_seat == 1
    assert other.dealers == result.dealers

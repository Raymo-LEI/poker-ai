"""Step-1 ``RuleBot`` postflop strength on the flop, turn, and river."""

import random

import pytest

from alphanl.bots import RuleBot
from alphanl.botzone.rule_bot import postflop_strength
from alphanl.botzone.table import parse_card

RANKS = "23456789TJQKA"
SUITS = "hdsc"
DECK = [suit.upper() + rank for suit in SUITS for rank in RANKS]


def _strength(hole, board):
    return RuleBot.__new__(RuleBot)._hand_strength(list(hole), list(board))


def _wiki(card: str) -> int:
    return parse_card(card[1] + card[0].lower())


@pytest.mark.parametrize(
    ("hole", "board", "score"),
    [
        (("HA", "DK"), ("C7", "S9", "HJ"), 1),
        (("HA", "DK"), ("C7", "S9", "HJ", "D4"), 1),
        (("S3", "S4"), ("S5", "S6", "HK"), 1),
        (("H2", "DK"), ("C7", "S9", "HJ", "D4"), 1),
        (("S3", "S4"), ("S5", "S6", "HK", "D2"), 5),
        (("SA", "S2"), ("S3", "S4", "S5"), 9),
        (("HA", "DA"), ("C7", "S9", "HJ"), 2),
    ],
)
def test_flop_and_turn_are_not_padded_with_fake_cards(hole, board, score):
    assert _strength(hole, board) == score


def test_seven_cards_match_rlcard_hand_category():
    hand_cls = pytest.importorskip("rlcard.games.limitholdem.utils").Hand
    rng = random.Random(0)
    for _ in range(3000):
        cards = rng.sample(DECK, 7)
        hand = hand_cls(list(cards))
        hand.evaluateHand()
        assert _strength(cards[:2], cards[2:]) == int(hand.category), cards


@pytest.mark.parametrize("n_board", [3, 4, 5])
def test_every_street_matches_the_best_five_card_evaluator(n_board):
    rng = random.Random(n_board)
    for _ in range(3000):
        cards = rng.sample(DECK, 2 + n_board)
        expected = postflop_strength(
            tuple(_wiki(c) for c in cards[:2]),
            tuple(_wiki(c) for c in cards[2:]),
        )
        assert _strength(cards[:2], cards[2:]) == expected, cards

"""BotzoneTable smoke: chips, showdown ranks, odd-chip chop.

The 1000-hand conservation suite is a later check. This file plays a
short random run and the scripted showdowns from the design.
"""

import random

import pytest

from alphanl.botzone.legal import TOTAL_CHIPS, safe_extra, to_call
from alphanl.botzone.table import (
    BotzoneTable,
    parse_card,
    preset_deck,
    split_pot,
)


def _cards(*names: str) -> tuple:
    return tuple(parse_card(name) for name in names)


def _checkdown(table: BotzoneTable) -> None:
    while not table.hand_over:
        assert 0 in table.legal_responses()
        table.act(0)
        assert table.stacks[0] + table.stacks[1] + table.pot == TOTAL_CHIPS


def _sample(rng: random.Random, state, legal: set) -> int:
    options = [response for response in (-1, 0, -2) if response in legal]
    extra = safe_extra(state)
    if extra is not None:
        options.append(extra)
        larger = max(extra * 2, state.pot)
        if larger in legal and larger != extra:
            options.append(larger)
    assert options
    return rng.choice(options)


def test_blinds_reset_and_button_rotates():
    table = BotzoneTable(dealer=0)
    table.start_hand()
    assert table.dealer == 0
    assert table.actor == 1
    assert table.stacks == (19900, 19950)
    assert table.pot == 150
    assert table.stacks[0] + table.stacks[1] + table.pot == TOTAL_CHIPS

    table.start_hand()
    assert table.dealer == 1
    assert table.actor == 0
    assert table.stacks == (19950, 19900)
    assert table.pot == 150


def test_conventional_burn_skips_three_cards():
    deck = tuple(range(52))
    table = BotzoneTable()
    table.start_hand(0, deck)
    assert table.holes == ((1, 3), (0, 2))
    assert table.board == ()

    table.act(0)
    table.act(0)
    assert table.street == 1
    assert table.board == (5, 6, 7)

    table.act(0)
    table.act(0)
    assert table.board == (5, 6, 7, 9)

    table.act(0)
    table.act(0)
    assert table.board == (5, 6, 7, 9, 11)
    assert 4 not in table.board
    assert 8 not in table.board
    assert 10 not in table.board

    other = BotzoneTable()
    other.start_hand(1, deck)
    assert other.holes == ((0, 2), (1, 3))


def test_all_in_leaves_fold_or_all_in_then_runs_the_board():
    table = BotzoneTable()
    table.start_hand(0, tuple(range(52)))
    table.act(-2)
    assert table.hand_over is False
    assert table.legal_responses() == {-1, -2}
    assert table.board == ()

    table.act(-2)
    assert table.hand_over is True
    assert table.board == (5, 6, 7, 9, 11)
    assert table.pot == 0
    assert table.stacks[0] + table.stacks[1] == TOTAL_CHIPS
    assert table.payoffs[0] + table.payoffs[1] == 0


def test_free_fold_is_legal_when_check_is_free():
    table = BotzoneTable()
    table.start_hand(
        0,
        preset_deck(
            0,
            (_cards("Ah", "Kd"), _cards("Qs", "Jc")),
            _cards("2c", "3d", "4h", "7s", "9c"),
        ),
    )
    table.act(0)
    table.act(0)
    assert table.street == 1
    assert to_call(table.state) == 0
    legal = table.legal_responses()
    assert -1 in legal
    assert 0 in legal

    table.act(-1)
    assert table.hand_over is True
    assert table.winners == (0,)
    assert table.pot == 0
    assert table.stacks == (20100, 19900)
    assert table.payoffs == (100, -100)


@pytest.mark.parametrize(
    ("hole0", "hole1", "board", "winner"),
    [
        pytest.param(
            ("Ah", "3h"),
            ("Kd", "9d"),
            ("Qh", "8h", "2h", "Js", "Tc"),
            0,
            id="flush",
        ),
        pytest.param(
            ("9h", "8d"),
            ("Ah", "Ad"),
            ("7c", "6s", "5d", "Kc", "2h"),
            0,
            id="straight",
        ),
        pytest.param(
            ("As", "Ad"),
            ("Jh", "3h"),
            ("Ah", "Kh", "7h", "2c", "2d"),
            0,
            id="full-house",
        ),
        pytest.param(
            ("Ac", "3c"),
            ("Ks", "Qd"),
            ("As", "Ah", "Ad", "Kc", "Kd"),
            0,
            id="quads",
        ),
        pytest.param(
            ("Ah", "2d"),
            ("Kc", "Kd"),
            ("5c", "4h", "3s", "9d", "8c"),
            0,
            id="wheel",
        ),
        pytest.param(
            ("Ah", "Kc"),
            ("6d", "8c"),
            ("5d", "4h", "3s", "2h", "9c"),
            1,
            id="wheel-plays-low",
        ),
    ],
)
def test_showdown_rankings(hole0, hole1, board, winner):
    table = BotzoneTable()
    table.start_hand(0, preset_deck(0, (_cards(*hole0), _cards(*hole1)), _cards(*board)))
    _checkdown(table)
    assert table.hand_over is True
    assert len(table.board) == 5
    assert table.winners == (winner,)
    assert table.pot == 0
    assert table.payoffs == ((100, -100) if winner == 0 else (-100, 100))
    assert table.stacks[0] + table.stacks[1] == TOTAL_CHIPS


def test_board_chop_splits_with_no_remainder():
    table = BotzoneTable()
    table.start_hand(
        0,
        preset_deck(
            0,
            (_cards("2c", "3d"), _cards("4c", "5d")),
            _cards("Ah", "Ad", "Ac", "Kd", "Qc"),
        ),
    )
    awarded_from_pot = False
    while not table.hand_over:
        assert 0 in table.legal_responses()
        if table.street == 3 and table.actor == 0:
            pot = table.pot
            stacks = table.stacks
            table.act(0)
            assert table.winners == (0, 1)
            awarded = split_pot(pot, (0, 1))
            assert table.stacks == (stacks[0] + awarded[0], stacks[1] + awarded[1])
            awarded_from_pot = True
            break
        table.act(0)
    assert awarded_from_pot
    assert table.pot == 0
    assert table.payoffs == (0, 0)
    assert table.stacks == (20000, 20000)


def test_odd_chip_on_chop_goes_to_lower_seat():
    assert split_pot(151, (0, 1)) == (76, 75)
    assert split_pot(151, (1, 0)) == (76, 75)
    assert split_pot(1, (0, 1)) == (1, 0)
    assert split_pot(150, (0, 1)) == (75, 75)
    assert split_pot(151, (0,)) == (151, 0)
    assert split_pot(151, (1,)) == (0, 151)


def test_short_random_run_conserves_chips_and_is_zero_sum():
    rng = random.Random(7186)
    table = BotzoneTable(rng=rng, dealer=0)
    for hand_index in range(12):
        table.start_hand()
        assert table.dealer == hand_index % 2
        assert table.stacks[0] + table.stacks[1] + table.pot == TOTAL_CHIPS
        steps = 0
        while not table.hand_over:
            legal = table.legal_responses()
            if any(table.state.all_in):
                assert legal <= {-1, -2}
            response = _sample(rng, table.state, legal)
            assert response in legal
            table.act(response)
            assert table.stacks[0] + table.stacks[1] + table.pot == TOTAL_CHIPS
            steps += 1
            assert steps < 40
        assert table.pot == 0
        assert table.stacks[0] + table.stacks[1] == TOTAL_CHIPS
        assert table.payoffs[0] + table.payoffs[1] == 0

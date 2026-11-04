"""Behavioral counterexamples; first run intentionally fails against the old API."""
import numpy as np
from repair_runtime.search_math import rank, ShowdownKernel, Terminal, Node, DCFR


def test_straight_beats_high_card():
    board = ['2s', '3h', '4d', '9c', 'Ks']
    assert rank(['5s', '6h'] + board) > rank(['As', 'Qh'] + board)


def test_forced_terminal_cannot_escape_ten_bb():
    result = DCFR(Terminal(np.full((2, 2), 10.)), np.ones((2, 2))).solve(40)
    assert np.allclose(result['values'][0], 10.)


def test_hidden_card_does_not_select_later_action():
    # One hero type; two hidden opponent types. Clairvoyant max gives 1,
    # but either legal shared response has expected payoff 0.
    root = Node(0, {'left': Terminal(np.array([[1., -1.]])),
                    'right': Terminal(np.array([[-1., 1.]]))})
    result = DCFR(root, np.array([[.5, .5]])).solve(100)
    assert abs(result['value']) < 1e-12
    assert result['strategy'][root.key].shape == (1, 2)


def test_nash_matching_pennies():
    opponent0 = Node(1, {0: Terminal(np.array([[1.]])), 1: Terminal(np.array([[-1.]]))}, key='o')
    opponent1 = Node(1, {0: Terminal(np.array([[-1.]])), 1: Terminal(np.array([[1.]]))}, key='o')
    root = Node(0, {0: opponent0, 1: opponent1})
    result = DCFR(root, np.ones((1, 1))).solve(2000)
    assert abs(result['value']) < .01
    assert np.allclose(result['strategy'][root.key], .5, atol=.01)

import numpy as np
import pytest
import torch

from alphanl.cfr.kuhn import (
    HISTORIES,
    KuhnRootFactory,
    KuhnState,
    best_response_value,
    exploitability,
    game_value,
    infoset_key,
    nash_conv,
    nash_strategy,
    train_kuhn,
    uniform_strategy,
)


def test_kuhn_rules_and_terminal_net_utilities():
    root = KuhnState((0, 2))
    assert root.current_player() == 0
    assert root.legal_actions() == (1, 2)
    assert root.child(1).current_player() == 1
    assert root.child(2).legal_actions() == (0, 1)
    assert root.child(2).child(0).returns() == (1, -1)
    assert root.child(1).child(2).child(0).returns() == (-1, 1)
    assert root.child(1).child(1).returns() == (-1, 1)
    assert root.child(2).child(1).returns() == (-2, 2)
    assert root.child(1).child(2).child(1).returns() == (-2, 2)
    with pytest.raises(ValueError, match="illegal"):
        root.child(0)


def test_infoset_encoding_never_reveals_other_private_card():
    for history in HISTORIES:
        actor = 0 if history in ("", "cb") else 1
        cards_a = (0, 1) if actor == 0 else (1, 0)
        cards_b = (0, 2) if actor == 0 else (2, 0)
        first = KuhnState(cards_a, history).observation()
        second = KuhnState(cards_b, history).observation()
        assert all(torch.equal(first[key], second[key]) for key in first)
        assert infoset_key(first) == (0, history)
        assert first["legal_mask"][0, 3:].sum() == 0


def test_factory_reproducible_uniform_distinct_deals():
    first, second = KuhnRootFactory(17), KuhnRootFactory(17)
    deals = [first().cards for _ in range(300)]
    assert deals == [second().cards for _ in range(300)]
    assert len(set(deals)) == 6
    assert all(len(set(cards)) == 2 for cards in deals)


def test_exact_imperfect_information_best_response_recognizes_nash():
    equilibrium = nash_strategy()
    assert game_value(equilibrium) == pytest.approx(-1 / 18)
    assert best_response_value(equilibrium, 0) == pytest.approx(-1 / 18)
    assert best_response_value(equilibrium, 1) == pytest.approx(1 / 18)
    assert nash_conv(equilibrium) == pytest.approx(0, abs=1e-12)
    assert exploitability(uniform_strategy()) == pytest.approx(11 / 24)


def test_production_external_sampling_reduces_kuhn_exploitability():
    # Stochastic MCCFR need not improve monotonically each iteration. Compare
    # a fixed-seed linearly averaged strategy with its untrained uniform prior.
    result = train_kuhn(iterations=1000, traversals_per_player=6, seed=1)
    assert result.regret_samples > 10_000
    assert result.final_exploitability < 0.04
    assert result.final_exploitability < result.initial_exploitability / 8
    assert result.value == pytest.approx(-1 / 18, abs=0.02)
    for (rank, history), strategy in result.strategy.items():
        assert rank in range(3)
        assert strategy.sum() == pytest.approx(1)
        assert np.isfinite(strategy).all()
        assert strategy[3:].sum() == 0

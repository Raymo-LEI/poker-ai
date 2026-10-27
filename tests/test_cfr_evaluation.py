"""Statistical units, private information boundaries and trajectory sampling."""
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.evaluation import (Audit, RuleActor, duplicate_match, launch_gates,
                                    independent_matches, paired_summary, random_equity)
from alphanl.cfr.model import RegretNetwork
from alphanl.cfr.strategy import SnapshotPolicy
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import HunlTable, NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer


def archive(tmp_path):
    config = CFRConfig(model=ModelConfig(d_model=16, nhead=2, num_layers=1,
                                        dim_feedforward=32, max_seq_len=32, dropout=0))
    model = RegretNetwork(PokerTransformer(**asdict(config.model)))
    for iteration in (1, 2, 3):
        torch.save({"format": "sd_cfr_strategy_v1", "iteration": iteration,
                    "weight": iteration, "rules_version": RULES_VERSION,
                    "numeric_version": NUMERIC_VERSION, "config": config.to_dict(),
                    "models": {0: model.state_dict(), 1: model.state_dict()}},
                   tmp_path / f"iteration_{iteration:07d}.pt")
    return tmp_path


def test_ci_uses_pair_means_and_chip_to_mbb_units():
    result = paired_summary([[100, -100], [300, -100], [500, -100]])
    expected = np.array([0, 1000, 2000])
    assert result["mbb_per_hand"] == 1000
    assert result["sample_std"] == expected.std(ddof=1)
    assert result["ci95"] == pytest.approx([1000 - 1.96 * 1000 / np.sqrt(3),
                                             1000 + 1.96 * 1000 / np.sqrt(3)])
    assert result["pairs"] == 3 and result["hands"] == 6


def test_cache_preserves_exact_draws_and_player_profiles(tmp_path):
    directory = archive(tmp_path)
    uncached = SnapshotPolicy(directory, seed=23, through_iteration=2)
    cached = SnapshotPolicy(directory, seed=23, through_iteration=2, cache_size=2)
    with patch("alphanl.cfr.strategy.torch.load", wraps=torch.load) as load:
        draws = [cached.begin_hand(i % 2) for i in range(100)]
        assert load.call_count == 2
    assert draws == [uncached.begin_hand(i % 2) for i in range(100)]
    assert set(draws) == {1, 2} and 55 < draws.count(2) < 80
    assert cached.player == 1 and cached.disk_loads == 2


def test_cache_eviction_and_selection_never_add_unarchived_iterations(tmp_path):
    directory = archive(tmp_path)
    policy = SnapshotPolicy(directory, snapshot_iterations=[2], cache_size=1)
    assert all(policy.begin_hand(p) == 2 for p in (0, 1, 0, 1))
    assert policy.disk_loads == 1
    policy = SnapshotPolicy(directory, cache_size=1)
    for i in range(30):
        policy.begin_hand(i % 2)
        assert len(policy._cache) == 1
    assert policy.disk_loads > 1
    with pytest.raises(ValueError, match="not in the archive"):
        SnapshotPolicy(directory, snapshot_iterations=[4])
    for kwargs in ({"through_iteration": True}, {"snapshot_iterations": [1.5]},
                   {"snapshot_iterations": [True]}, {"cache_size": -1}):
        with pytest.raises(ValueError):
            SnapshotPolicy(directory, **kwargs)


def test_duplicate_reuses_holes_and_deck_and_swaps_identities():
    class RecordingCaller(RuleActor):
        def __init__(self):
            super().__init__("call", seed=1)
            self.audit = Audit()
            self.views = []
            self.seats = []

        def begin_hand(self, seat):
            self.seats.append(seat)

        def act(self, snapshot):
            self.views.append(snapshot)
            return super().act(snapshot)

    hero, opponent = RecordingCaller(), RecordingCaller()
    rows = []
    result = duplicate_match(hero, opponent, pairs=5, seed=13, log=rows.append)
    assert hero.seats == [1, 0] * 5 and opponent.seats == [0, 1] * 5
    assert result["mbb_per_hand"] == 0 and result["ci95"] == [0, 0]
    assert all(sum(row["chips_sb_bb"]) == 0 for row in rows)
    # Both legs reach the river. The same seat keeps its cards and board.
    hero_rivers = [s for s in hero.views if s.street == 3]
    opp_rivers = [s for s in opponent.views if s.street == 3]
    for i in range(0, len(hero_rivers), 2):
        assert hero_rivers[i].hole_cards == opp_rivers[i + 1].hole_cards
        assert hero_rivers[i].board_cards == opp_rivers[i + 1].board_cards


def test_rule_equity_sees_only_snapshot_and_is_seed_reproducible():
    table = HunlTable(seed=7)
    table.reset()
    snapshot = table.snapshot(1)
    first = random_equity(snapshot, np.random.default_rng(17), 100)
    second = random_equity(snapshot, np.random.default_rng(17), 100)
    assert first == second and 0 <= first <= 1
    assert not hasattr(snapshot, "opponent_hole_cards")


@pytest.mark.parametrize("payoff,win_fraction,ties", [(100, 1., 0), (0, 0., 2)])
def test_match_blocks_are_independent_and_ties_are_not_wins(payoff, win_fraction, ties):
    class Actor:
        audit = Audit()
    records = []
    with patch("alphanl.cfr.evaluation.play_hand", return_value=payoff) as hand:
        result = independent_matches(Actor(), Actor(), matches=2, seed=17, log=records.append)
    calls = hand.call_args_list
    assert len(calls) == 100
    assert len({call.kwargs["deck_seed"] for call in calls}) == 100
    for offset in (0, 50):
        seats = [call.kwargs["hero_seat"] for call in calls[offset:offset + 50]]
        assert seats.count(0) == seats.count(1) == 25
        assert all(a != b for a, b in zip(seats, seats[1:]))
    assert result["win_fraction"] == win_fraction and result["ties"] == ties
    assert result["mbb_per_hand"] == payoff * 10
    assert all(len(row["chips"]) == 50 for row in records)


def test_gate_never_passes_missing_or_smoke_evidence():
    report = {"hero_iteration": 50, "results": {
        "duplicate:1000_vs_100": {"pairs": 2500, "mbb_per_hand": 100, "ci95": [50, 150]},
        "duplicate:50_vs_random": {"pairs": 2500, "mbb_per_hand": 2000},
        "matches:50_vs_random": {"matches": 500, "win_fraction": 1}}}
    assert launch_gates(report)["verdict"] == "INCOMPLETE"
    report["hero_iteration"] = 1000
    report["results"]["duplicate:1000_vs_100"]["ci95"] = [-1, 150]
    result = launch_gates(report)
    assert result["criteria"]["self_improvement"]["status"] == "FAIL"
    assert result["verdict"] == "NO_GO"
    assert launch_gates({"hero_iteration": 50, "results": {}, "error": {"message": "NaN"}})["verdict"] == "NO_GO"


def test_early_candidate_behavior_failure_checks_both_selfplay_actors():
    def counts(shoves):
        return {"sb_open_shoves": shoves, "sb_open_opportunities": 5000,
                "illegal_selections": 0, "mask_leaks": 0, "nonfinite_regrets": 0}
    health = {"hands": 10000, "hero": {"counts": counts(0)}, "opponent": {"counts": counts(1000)}}
    report = {"hero_iteration": 50, "results": {"health": health}}
    result = launch_gates(report)
    assert result["criteria"]["behavior"]["status"] == "FAIL"
    assert result["verdict"] == "NO_GO"
    health["opponent"]["counts"].pop("nonfinite_regrets")
    assert launch_gates(report)["criteria"]["behavior"]["status"] == "INCOMPLETE"

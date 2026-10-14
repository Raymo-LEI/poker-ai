"""Heads-up match simulation and mbb/hand statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .paths import BIG_BLIND_CHIPS


@dataclass
class MatchupResult:
    n_hands: int
    mean_mbb_per_hand: float
    std_mbb_per_hand: float
    ci95_low: float
    ci95_high: float
    mean_chip_per_hand: float

    def as_dict(self):
        return {
            "n_hands": self.n_hands,
            "mean_mbb_per_hand": self.mean_mbb_per_hand,
            "std_mbb_per_hand": self.std_mbb_per_hand,
            "ci95": [self.ci95_low, self.ci95_high],
            "mean_chip_per_hand": self.mean_chip_per_hand,
        }


def _one_hand(env, seat0_agent, seat1_agent, payoff_index: int) -> float:
    obs = env.reset()
    done = False
    while not done:
        seat = env.my_agent()
        if seat == 0:
            action = seat0_agent.make_action(obs)
        else:
            action = seat1_agent.make_action(obs)
        obs, reward, done, _info = env.step(action)
    return float(reward[payoff_index])


def _play_balanced(env, hero, villain, n_per_side: int, progress: bool) -> List[float]:
    """Mirror agi.evaluation_tools.death_match seating balance."""
    profits: List[float] = []
    total = n_per_side * 2
    iterator = range(total)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="hands")

    for i in iterator:
        if i < n_per_side:
            profits.append(_one_hand(env, hero, villain, 0))
        else:
            profits.append(_one_hand(env, villain, hero, 1))
    return profits


def run_matchup(env, hero, villain, n_hands: int, progress: bool = False) -> MatchupResult:
    """
    Play n_hands total with balanced seating (same structure as upstream death_match).

    Each block of n_hands//2 uses hero as seat0_agent then swapped mapping.
    """
    n_per_side = max(1, n_hands // 2)
    chip_profits = _play_balanced(env, hero, villain, n_per_side, progress)

    arr = np.asarray(chip_profits, dtype=np.float64)
    profits_bb = arr / BIG_BLIND_CHIPS
    mean_bb = float(profits_bb.mean())
    std_bb = float(profits_bb.std(ddof=1)) if len(profits_bb) > 1 else 0.0
    se = std_bb / np.sqrt(len(profits_bb)) if len(profits_bb) else 0.0
    z = 1.96
    return MatchupResult(
        n_hands=len(profits_bb),
        mean_mbb_per_hand=1000.0 * mean_bb,
        std_mbb_per_hand=1000.0 * std_bb,
        ci95_low=1000.0 * (mean_bb - z * se),
        ci95_high=1000.0 * (mean_bb + z * se),
        mean_chip_per_hand=float(arr.mean()),
    )

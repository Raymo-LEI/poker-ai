"""Rule bot versus uniform random legal ids, with the §5.3 scores.

``bb/100`` is the mean big blinds per hand times 100. The 95% interval is
the normal interval on that mean: mean ± 1.96 × sample std / √n, then
scaled by 100 the same way. Sample std is the one with divisor ``n - 1``.

A match score is 50 hands with the button alternating, summed chips
divided by 100 (the big blind). Hands left over after complete matches
are not a match. The yardstick R is the bb/100 point estimate, not the
interval and not the match score.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .actions import legal_action_ids, response_for_action
from .legal import BIG_BLIND
from .rule_bot import BotzoneRuleBot
from .table import BotzoneTable

MATCH_LENGTH = 50
Z_95 = 1.96


@dataclass(frozen=True)
class RuleEvalResult:
    n_hands: int
    seed: int
    hero_seat: int
    big_blind: int
    mean_bb_per_hand: float
    std_bb_per_hand: float
    bb_per_100: float
    ci95_low: float
    ci95_high: float
    match_length: int
    n_matches: int
    mean_match_score: float
    dealers: Tuple[int, ...]

    @property
    def R(self) -> float:
        """Point estimate of rule-versus-random bb/100."""

        return self.bb_per_100

    def as_dict(self) -> dict:
        return {
            "hero": "botzone_rule_bot",
            "opponent": "uniform_legal_action_ids",
            "hero_seat": self.hero_seat,
            "dealer": "alternating",
            "n_hands": self.n_hands,
            "seed": self.seed,
            "big_blind": self.big_blind,
            "mean_bb_per_hand": self.mean_bb_per_hand,
            "std_bb_per_hand": self.std_bb_per_hand,
            "bb_per_100": self.bb_per_100,
            "ci95": [self.ci95_low, self.ci95_high],
            "R": self.R,
            "match_length": self.match_length,
            "n_matches": self.n_matches,
            "mean_match_score": self.mean_match_score,
            "match_score_unit": "cumulative_chips_per_100",
        }


def bb_per_100_interval(
    chip_profits: Sequence[int],
    big_blind: int = BIG_BLIND,
) -> Tuple[float, float, float, float, float]:
    """Return mean bb/hand, sample std, bb/100, ci low, ci high.

    The interval is ``(mean ± 1.96 * std / sqrt(n)) * 100`` with ``mean``
    and ``std`` in big blinds per hand.
    """

    if big_blind <= 0:
        raise ValueError(f"big blind must be positive, got {big_blind}")
    n = len(chip_profits)
    if n < 2:
        raise ValueError(f"need at least 2 hands for a sample std, got {n}")
    bbs = [profit / big_blind for profit in chip_profits]
    mean = statistics.fmean(bbs)
    std = statistics.stdev(bbs)
    half = Z_95 * std / math.sqrt(n)
    return mean, std, mean * 100.0, (mean - half) * 100.0, (mean + half) * 100.0


def match_scores(
    chip_profits: Sequence[int],
    match_length: int = MATCH_LENGTH,
    per: int = BIG_BLIND,
) -> List[float]:
    """Cumulative chips / ``per`` for each complete block of hands.

    ``per`` is the big blind (100). The design writes this as cumulative
    chips / 100.
    """

    if match_length < 1:
        raise ValueError(f"match length must be positive, got {match_length}")
    if per <= 0:
        raise ValueError(f"divisor must be positive, got {per}")
    scores = []
    for start in range(0, len(chip_profits) - match_length + 1, match_length):
        block = chip_profits[start : start + match_length]
        scores.append(sum(block) / per)
    return scores


def uniform_legal_action(state, rng: random.Random) -> int:
    """Uniform draw over the five-id mask, not over raw raise extras."""

    legal = legal_action_ids(state)
    if not legal:
        raise ValueError("no legal action")
    return legal[rng.randrange(len(legal))]


def play_rule_vs_random(
    n_hands: int,
    seed: int = 42,
    hero_seat: int = 0,
    match_length: int = MATCH_LENGTH,
    table_rng: Optional[random.Random] = None,
    action_rng: Optional[random.Random] = None,
) -> RuleEvalResult:
    """Rule bot at ``hero_seat`` against uniform legal ids.

    The button starts on seat 0 and then alternates, including inside each
    50-hand match. ``table_rng`` only shuffles. ``action_rng`` only picks
    the random seat's action, so a fixed seed replays both.
    """

    if n_hands < 2:
        raise ValueError(f"need at least 2 hands, got {n_hands}")
    if hero_seat not in (0, 1):
        raise ValueError(f"hero seat must be 0 or 1, got {hero_seat}")
    deck_rng = random.Random(seed) if table_rng is None else table_rng
    pick_rng = random.Random(seed + 1) if action_rng is None else action_rng
    table = BotzoneTable(rng=deck_rng, dealer=0)
    bot = BotzoneRuleBot()
    chips: List[int] = []
    dealers: List[int] = []
    for hand_index in range(n_hands):
        table.start_hand()
        if table.dealer != hand_index % 2:
            raise RuntimeError(
                f"dealer did not alternate: hand {hand_index} dealt by {table.dealer}"
            )
        dealers.append(table.dealer)
        while not table.hand_over:
            state = table.state
            assert state is not None
            if state.actor == hero_seat:
                action = bot.action_id(table)
            else:
                action = uniform_legal_action(state, pick_rng)
            response = response_for_action(state, action)
            if response is None:
                raise RuntimeError(f"action {action} was chosen but is masked")
            table.act(response)
        chips.append(table.payoffs[hero_seat])
    mean, std, bb100, low, high = bb_per_100_interval(chips, BIG_BLIND)
    scores = match_scores(chips, match_length)
    return RuleEvalResult(
        n_hands=n_hands,
        seed=seed,
        hero_seat=hero_seat,
        big_blind=BIG_BLIND,
        mean_bb_per_hand=mean,
        std_bb_per_hand=std,
        bb_per_100=bb100,
        ci95_low=low,
        ci95_high=high,
        match_length=match_length,
        n_matches=len(scores),
        mean_match_score=statistics.fmean(scores) if scores else float("nan"),
        dealers=tuple(dealers),
    )

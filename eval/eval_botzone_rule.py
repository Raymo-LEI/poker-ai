#!/usr/bin/env python3
"""Botzone rule bot versus uniform random legal actions.

Reports the design §5.3 yardstick R: bb/100 of the rule bot, with the
normal 95% interval (mean ± 1.96 × sample std / √n). Also reports the
50-hand match score (button alternating, cumulative chips / 100).

This is not a Slumbot number and not a 50bb rlcard mbb/hand number.
Stacks are 200bb, so a few hundred hands already print an interval and a
few thousand narrow it enough to keep as the yardstick R.

    python3 eval/eval_botzone_rule.py --hands 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.botzone.rule_eval import MATCH_LENGTH, play_rule_vs_random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hands",
        type=int,
        default=2000,
        help="Hands in the matchup. A few hundred already yields an interval; "
        "50-hand matches use only complete blocks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deck seed. Actions use seed+1.")
    parser.add_argument("--hero-seat", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--match-length",
        type=int,
        default=MATCH_LENGTH,
        help="Hands per match score. Default 50, button alternating.",
    )
    args = parser.parse_args()

    result = play_rule_vs_random(
        n_hands=args.hands,
        seed=args.seed,
        hero_seat=args.hero_seat,
        match_length=args.match_length,
    )
    print(
        f"rule vs random: R = {result.bb_per_100:.2f} bb/100 "
        f"(95% CI [{result.ci95_low:.2f}, {result.ci95_high:.2f}], "
        f"n={result.n_hands}, seed={result.seed}, hero_seat={result.hero_seat})"
    )
    print(
        f"50-hand match score: mean {result.mean_match_score:.2f} "
        f"cumulative chips/100 over {result.n_matches} matches "
        f"(length {result.match_length}, dealer alternating)"
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()

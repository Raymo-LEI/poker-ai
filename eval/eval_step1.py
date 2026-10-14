#!/usr/bin/env python3
"""Step 1: AlphaNLHoldem vs random / rule bot at 50bb (rlcard env, unchanged weights)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.bots import RandomBot, RuleBot
from alphanl.inference import AlphaNLHoldemAgent
from alphanl.matchup import run_matchup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=2000, help="Hands per opponent matchup")
    parser.add_argument("--weights", type=Path, default=None, help="Path to c_*.pkl checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="NumPy / rlcard RNG seed")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Allow TensorFlow GPU (default CPU). Set CUDA_VISIBLE_DEVICES as needed.",
    )
    args = parser.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    from rlcard.utils import set_seed

    np.random.seed(args.seed)
    set_seed(args.seed)

    agent = AlphaNLHoldemAgent(weights_path=args.weights)
    env = agent.env
    random_bot = RandomBot()
    rule_bot = RuleBot(env.env)

    results = {}
    for name, villain in [("random", random_bot), ("rule", rule_bot)]:
        res = run_matchup(env, agent, villain, n_hands=args.hands, progress=args.progress)
        results[name] = res.as_dict()
        print(
            f"AlphaNL vs {name}: {res.mean_mbb_per_hand:.1f} mbb/hand "
            f"(95% CI [{res.ci95_low:.1f}, {res.ci95_high:.1f}], n={res.n_hands})"
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

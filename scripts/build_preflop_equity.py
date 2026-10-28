"""Build 169 uniform-range Monte Carlo estimates; no training or hidden cards."""
import argparse
import json
from pathlib import Path
import numpy as np
from alphanl.transformer.equity import TABLE_PATH, sample_equity
from alphanl.transformer.schema import RANKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=16384)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--out", type=Path, default=TABLE_PATH)
    args = p.parse_args()
    if args.samples < 1 or args.seed < 0:
        p.error("positive samples and nonnegative seed required")
    equities = {}
    for i, high in enumerate(RANKS):
        for low in RANKS[:i+1]:
            for suffix in (("",) if high == low else ("s", "o")):
                key = high + low + suffix
                hole = ("C" + high, ("C" if suffix == "s" else "D") + low)
                seed = np.random.SeedSequence([args.seed, len(equities)])
                equities[key] = sample_equity(hole, (), np.random.default_rng(seed), args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"format": "random_equity_preflop_mc_v1",
        "samples_per_class": args.samples, "seed": args.seed, "evaluator": "phevaluator==0.6.0",
        "method": "uniform unknown opponent and board; ties=0.5; not exact",
        "worst_case_mc_standard_error": .5 / args.samples ** .5,
        "equities": equities}, indent=2) + "\n")
    print(f"Wrote {len(equities)} classes to {args.out}")

if __name__ == "__main__":
    main()

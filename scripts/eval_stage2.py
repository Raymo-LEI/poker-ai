"""Explicit independent Stage 2 acceptance evaluation; never launches training."""
import argparse
import json
from pathlib import Path
import torch
from alphanl.stage2.train import load_checkpoint
from alphanl.stage2.evaluation import MatchEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--match-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=92017)
    parser.add_argument("--equity-samples", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(1)
    model, blob = load_checkpoint(args.checkpoint, device=args.device)
    evaluator = MatchEvaluator(model, match_pairs=args.match_pairs, seed=args.seed, equity_samples=args.equity_samples)
    report = evaluator.evaluate(model, update=blob["update"], progress=lambda name: print(f"Evaluating {name}", flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "retention_pass": report["retention_pass"],
                      "equity_positive": report["equity_positive"]}))
    if not report["retention_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

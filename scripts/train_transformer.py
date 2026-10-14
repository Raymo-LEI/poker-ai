#!/usr/bin/env python3
"""Self-play PPO for the step-3 betting-history Transformer.

Does not train unless you run it. ``--dry-run`` only prints the model size.

    export PYTHONPATH=src
    python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json
    python scripts/train_transformer.py --config configs/transformer/colab_l4.json --entropy-coef 0.02
    python scripts/train_transformer.py --config configs/transformer/laptop_8gb.json --resume   # newest ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.train import describe_run, run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "transformer" / "laptop_8gb.json",
        help="JSON experiment config. Default is the 8GB laptop preset.",
    )
    parser.add_argument("--entropy-coef", type=float, default=None, help="Override ppo.entropy_coef")
    parser.add_argument("--updates", type=int, default=None, help="Override train.num_updates")
    parser.add_argument("--out-dir", type=str, default=None, help="Override train.out_dir")
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="Resume. Bare --resume picks the newest checkpoint in train.checkpoint_dir; or pass a path.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", default=None, help="Mixed precision (CUDA only; ignored on CPU)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override train.checkpoint_dir")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override train.checkpoint_every (updates)")
    parser.add_argument("--dry-run", action="store_true", help="Print the model summary and exit")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    cfg.apply_overrides(
        entropy_coef=args.entropy_coef,
        num_updates=args.updates,
        out_dir=args.out_dir,
        device=args.device,
        seed=args.seed,
        amp=args.amp,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )
    print(describe_run(cfg))
    if args.dry_run:
        return
    out_dir = run_training(cfg, resume=args.resume)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

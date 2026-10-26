"""Train the standard 200BB SD-CFR base without touching the PPO pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer
from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cfr/hunl_200bb.json"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"))
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", type=Path, help="trusted local train.pt (includes reservoirs)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = CFRConfig.from_json(args.config)
    for arg, field in (("iterations", "iterations"), ("traversals", "traversals_per_player"),
                       ("train_steps", "train_steps"), ("out_dir", "out_dir"), ("device", "device")):
        value = getattr(args, arg)
        if value is not None:
            setattr(config, field, value)
    if args.amp_dtype is not None:
        config.amp_dtype = args.amp_dtype
    if args.fused_optimizer is not None:
        config.fused_optimizer = args.fused_optimizer
    if args.allow_tf32 is not None:
        config.allow_tf32 = args.allow_tf32
    config.__post_init__()
    if args.dry_run:
        from dataclasses import asdict
        report = config.to_dict()
        resolved_device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                                       else ("cpu" if config.device == "auto" else config.device))
        bf16_supported = bool(resolved_device.type == "cuda" and torch.cuda.is_available()
                              and torch.cuda.is_bf16_supported())
        cuda_available = torch.cuda.is_available()
        report.update({"effective_device": str(resolved_device),
                       "cuda_available": cuda_available,
                       "effective_amp_dtype": config.amp_dtype if bf16_supported else "none",
                       "effective_fused_optimizer": bool(config.fused_optimizer and resolved_device.type == "cuda" and cuda_available),
                       "cuda_bf16_supported": bf16_supported})
        report.update({"backbone_parameters": count_parameters(PokerTransformer(**asdict(config.model))),
                       "rules_version": RULES_VERSION, "numeric_version": NUMERIC_VERSION,
                       "players": 2, "small_blind": 50, "big_blind": 100, "starting_stack": 20000})
        print(json.dumps(report, indent=2))
        return
    if args.resume is None:
        if (Path(config.out_dir) / "metrics.csv").exists():
            parser.error("output directory already contains a run; use --resume or a new --out-dir")
        if config.buffer_directory and any(Path(config.buffer_directory).glob("player_*/metadata.json")):
            parser.error("buffer directory contains a previous run; use --resume or a new directory")
    trainer = SDCFRTrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()

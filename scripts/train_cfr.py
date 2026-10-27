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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--separate-player-models", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reinitialize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preflop-action-guard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--spr-action-guard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--guard-max-pot-ratio", type=float)
    parser.add_argument("--target-scale", type=float)
    parser.add_argument("--loss", choices=("huber", "mse"))
    parser.add_argument("--out-dir")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16"))
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--workers", type=int, help="spawned CPU collectors; 0 uses the single-process path")
    parser.add_argument("--tree-batch", type=int)
    parser.add_argument("--inference-batch", type=int)
    parser.add_argument("--inference-wait-ms", type=float)
    parser.add_argument("--train-buffer-device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", type=Path, help="trusted local train.pt (includes reservoirs)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    raw_config = json.loads(args.config.read_text())
    config = (CFRConfig.from_checkpoint(raw_config) if args.resume
              else CFRConfig(**raw_config))
    for arg, field in (("iterations", "iterations"), ("traversals", "traversals_per_player"),
                       ("train_steps", "train_steps"), ("out_dir", "out_dir"), ("device", "device"),
                       ("workers", "traversal_workers"), ("tree_batch", "traversal_batch_size"),
                       ("inference_batch", "inference_batch_size"), ("inference_wait_ms", "inference_wait_ms"),
                       ("train_buffer_device", "train_buffer_device"), ("batch_size", "minibatch_size"),
                       ("separate_player_models", "separate_player_models"), ("reinitialize", "reinitialize"),
                       ("preflop_action_guard", "preflop_action_guard"), ("spr_action_guard", "spr_action_guard"),
                       ("guard_max_pot_ratio", "guard_max_pot_ratio"), ("target_scale", "regret_target_scale"), ("loss", "loss")):
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
                       "effective_train_buffer_device": "cuda" if (config.train_buffer_device == "cuda"
                           and resolved_device.type == "cuda" and cuda_available) else "cpu",
                       "cuda_bf16_supported": bf16_supported})
        report.update({"backbone_parameters": count_parameters(PokerTransformer(**asdict(config.model))),
                       "rules_version": RULES_VERSION, "numeric_version": NUMERIC_VERSION,
                       "players": 2, "small_blind": 50, "big_blind": 100, "starting_stack": 20000})
        report.update(network_count=2 if config.separate_player_models else 1,
                      effective_buffer_capacity=config.buffer_capacity * (1 if config.separate_player_models else 2),
                      seat_balanced_sampling=not config.separate_player_models)
        print(json.dumps(report, indent=2))
        return
    if args.resume is None:
        if (Path(config.out_dir) / "metrics.csv").exists():
            parser.error("output directory already contains a run; use --resume or a new --out-dir")
        if config.buffer_directory and any(Path(config.buffer_directory).glob("*/metadata.json")):
            parser.error("buffer directory contains a previous run; use --resume or a new directory")
    with SDCFRTrainer(config) as trainer:
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
        trainer.run()


if __name__ == "__main__":
    main()

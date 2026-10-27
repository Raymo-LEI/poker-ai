"""Prepare or run offline SD-CFR trajectory distillation; never resumes training."""

import argparse
import json
from pathlib import Path
import time

import torch

from alphanl.cfr.distillation import (
    PolicyNetwork, collect_trajectories, cpu_latency, evaluate_student,
    fit_student, load_student, prepare_archive, save_student,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="must be a new directory")
    parser.add_argument("--through", type=int)
    parser.add_argument("--after", type=int, default=0, help="exclusive; default preserves early history")
    parser.add_argument("--stride", type=int, default=1, help="retain iteration ids divisible by stride")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-hands", type=int, default=10_000)
    parser.add_argument("--validation-hands", type=int, default=2_000)
    parser.add_argument("--max-decisions", type=int, default=200_000, help="memory bound per split")
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    for name in ("train_hands", "validation_hands", "max_decisions", "steps", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config, manifest = prepare_archive(args.strategies, through=args.through,
                                       after=args.after, stride=args.stride)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "teacher.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    started = time.perf_counter()
    common = dict(directory=args.strategies, config=config, manifest=manifest, seed=args.seed,
                  max_decisions=args.max_decisions, teacher_device=args.device)
    train, train_report = collect_trajectories(**common, hands=args.train_hands)
    validation, validation_report = collect_trajectories(
        **common, hands=args.validation_hands, hand_offset=args.train_hands)
    # Persist the sampled data for reproducibility. Split by complete hands.
    torch.save({"train": train, "validation": validation}, args.out_dir / "dataset.pt")
    model = PolicyNetwork(config).to(args.device)
    before = evaluate_student(model, validation, batch_size=args.batch_size)
    last_loss = fit_student(model, train, steps=args.steps, batch_size=args.batch_size,
                            seed=args.seed, learning_rate=args.learning_rate)
    report = {"status": "offline_candidate_requires_playing_strength_and_deployment_acceptance",
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
              "train": train_report, "validation": validation_report,
              "validation_before": before,
              "validation_after": evaluate_student(model, validation, batch_size=args.batch_size),
              "last_minibatch_loss": last_loss,
              "parameters": sum(p.numel() for p in model.parameters()),
              "elapsed_collect_fit_validate_s": time.perf_counter() - started}
    path = args.out_dir / "policy.pt"
    save_student(path, model, manifest, report)
    restored, _ = load_student(path)
    report["cpu_latency"] = cpu_latency(restored, validation)
    save_student(path, restored, manifest, report)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

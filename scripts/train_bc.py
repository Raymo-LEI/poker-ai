#!/usr/bin/env python3
"""Behavior-cloning warm start, then a checkpoint PPO can load.

The dataset is produced by ``scripts/dump_c1048_bc.py`` in the Python 3.7
environment. This script stays in the PyTorch environment. ``--dump`` shells
out to that interpreter when the npz is missing; it does not import TensorFlow
into this process.

    export PYTHONPATH=src
    python scripts/train_bc.py --config configs/transformer/laptop_8gb_conv.json \\
        --data data/c1048_bc.npz --init-c1048 third_party/AlphaNLHoldem/weights/c_1048.pkl \\
        --loss kl --steps 2000 --out runs/bc_conv/bc.pt

    python scripts/train_bc.py --config configs/transformer/laptop_8gb.json \\
        --data data/c1048_bc.npz --loss kl --steps 2000 --out runs/bc_transformer/bc.pt
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.transformer.bc import (  # noqa: E402
    bc_losses,
    bc_update,
    dataset_for_encoder,
    load_bc_npz,
    save_bc_checkpoint,
)
from alphanl.transformer.config import ExperimentConfig, resolve_device  # noqa: E402
from alphanl.transformer.tokenizer import slice_batch  # noqa: E402
from alphanl.transformer.train import build_model, build_optimizer  # noqa: E402


def kl_value(model, batch, teacher, *, minibatch_size: int) -> float:
    """Row-weighted KL from :func:`bc_losses`. No optimizer step.

    ``train_bc.py``'s KL. Illegal teacher mass is dropped inside
    ``bc_losses`` the same way a training step drops it.
    """

    count = int(teacher.shape[0])
    if count == 0:
        raise ValueError("KL requires at least one row")
    width = max(1, int(minibatch_size))
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for start in range(0, count, width):
                index = torch.arange(start, min(start + width, count), device=teacher.device)
                _loss, metrics = bc_losses(
                    model,
                    slice_batch(batch, index),
                    teacher_probs=teacher.index_select(0, index),
                    kind="kl",
                )
                value = float(metrics["kl"])
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite KL {value}")
                total += value * int(index.shape[0])
    finally:
        if was_training:
            model.train()
    return total / float(count)


def train_kl_epoch(model, optimizer, batch, teacher, *, minibatch_size: int, max_grad_norm: float) -> float:
    """One pass of :func:`bc_update` with ``kind='kl'``. Returns that pass's mean loss."""

    metrics = bc_update(
        model,
        optimizer,
        batch,
        teacher_probs=teacher,
        actions=None,
        kind="kl",
        epochs=1,
        minibatch_size=int(minibatch_size),
        max_grad_norm=float(max_grad_norm),
    )
    value = float(metrics["loss"])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite training KL {value}")
    return value


def _dump_if_needed(args) -> None:
    if args.data.is_file() and not args.force_dump:
        return
    if not args.dump:
        raise SystemExit(
            "BC data %s is missing. Generate it in the Python 3.7 env "
            "(scripts/dump_c1048_bc.py) or pass --dump." % args.data
        )
    python = args.python or os.environ.get("ALPHANL_PYTHON") or shutil.which("python3.7")
    if not python:
        raise SystemExit(
            "python3.7 was not found. Set ALPHANL_PYTHON to the Step-1 interpreter. "
            "Do not install TensorFlow into this PyTorch environment."
        )
    cmd = [
        python,
        str(REPO_ROOT / "scripts" / "dump_c1048_bc.py"),
        "--hands",
        str(args.dump_hands),
        "--opponent",
        args.dump_opponent,
        "--out",
        str(args.data),
        "--seed",
        str(args.seed or 0),
    ]
    if args.weights is not None:
        cmd.extend(["--weights", str(args.weights)])
    for path in args.pool:
        cmd.extend(["--pool", str(path)])
    print("dump:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "transformer" / "laptop_8gb.json")
    parser.add_argument("--encoder", choices=("transformer", "alphanl_conv"), default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss", choices=("kl", "ce"), default="kl")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None, help="Default is the config PPO learning rate")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-c1048",
        type=Path,
        default=None,
        help="Port c_1048 into the conv control net before BC. Refused for the transformer.",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Passed through to the dump script")
    parser.add_argument("--dump", action="store_true", help="If --data is missing, call the py3.7 dump script")
    parser.add_argument("--force-dump", action="store_true")
    parser.add_argument("--python", type=str, default=None, help="Step-1 interpreter (default: python3.7)")
    parser.add_argument("--dump-hands", type=int, default=50)
    parser.add_argument("--dump-opponent", choices=("self", "random", "pool"), default="self")
    parser.add_argument("--pool", type=Path, nargs="*", default=())
    args = parser.parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    cfg.apply_overrides(encoder=args.encoder, device=args.device, seed=args.seed)
    _dump_if_needed(args)
    if args.init_c1048 is not None and cfg.encoder != "alphanl_conv":
        raise SystemExit(
            "--init-c1048 only fits encoder=alphanl_conv. The transformer does not share "
            "NlHoldemNet's layers; warm-start it with this BC run instead."
        )

    torch.manual_seed(cfg.seed)
    device = torch.device(resolve_device(cfg.device))
    model = build_model(cfg).to(device)
    if args.init_c1048 is not None:
        from alphanl.transformer.conv_net import load_tf_weights

        missing = load_tf_weights(model, args.init_c1048)
        print("ported c_1048, missing layers:", missing)
    arrays = load_bc_npz(args.data)
    data = dataset_for_encoder(cfg, arrays)
    data = {key: value.to(device) for key, value in data.items()}
    lr = cfg.ppo.lr if args.lr is None else float(args.lr)
    optimizer = build_optimizer(model, cfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    batch = {key: data[key] for key in data if key not in ("probs", "actions")}
    teacher = data["probs"]
    actions = data["actions"]
    minibatch = min(int(args.batch_size), int(actions.shape[0]))
    first = None
    last = None
    for step in range(int(args.steps)):
        metrics = bc_update(
            model,
            optimizer,
            batch,
            teacher_probs=teacher,
            actions=actions,
            kind=args.loss,
            epochs=1,
            minibatch_size=minibatch,
            max_grad_norm=cfg.ppo.max_grad_norm,
        )
        if first is None:
            first = metrics["loss"]
        last = metrics["loss"]
        if step % 50 == 0 or step + 1 == int(args.steps):
            print("bc step %d loss %.6f" % (step, metrics["loss"]))
    path = save_bc_checkpoint(args.out, model=model, optimizer=optimizer, config=cfg.to_dict())
    print("wrote %s first_loss=%s last_loss=%s" % (path, first, last))


if __name__ == "__main__":
    main()

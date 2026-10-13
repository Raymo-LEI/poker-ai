#!/usr/bin/env python3
"""Compare the PyTorch control net with TF NlHoldemNet on the same observations.

The two stacks do not share a process. Dump probabilities in the Step-1
Python 3.7 env, then compare in the PyTorch env. Target: max abs diff < 1e-4.

    # Python 3.7
    export PYTHONPATH=src
    python scripts/compare_c1048_probs.py dump-tf \
        --weights third_party/AlphaNLHoldem/weights/c_1048.pkl \
        --hands 20 --out data/c1048_probs.npz

    # Python 3.10+
    export PYTHONPATH=src
    python scripts/compare_c1048_probs.py compare \
        --weights third_party/AlphaNLHoldem/weights/c_1048.pkl \
        --data data/c1048_probs.npz

With no subcommand the script prints UNVERIFIED and exits 0. That is what
happens in an environment that cannot start TensorFlow 1.15.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _unverified(reason):
    print("UNVERIFIED: %s" % reason)
    print("Target when both sides have run: max abs diff < 1e-4.")
    print("Dump with `dump-tf` in the Python 3.7 env, then `compare` in the PyTorch env.")
    return 0


def pytorch_probs(model, card, action, extra, legal):
    import torch

    batch = {
        "card_info": torch.from_numpy(np.asarray(card, dtype=np.float32)),
        "action_info": torch.from_numpy(np.asarray(action, dtype=np.float32)),
        "extra_info": torch.from_numpy(np.asarray(extra, dtype=np.float32)),
        "legal_mask": torch.from_numpy(np.asarray(legal, dtype=np.float32)),
    }
    was = model.training
    model.eval()
    with torch.no_grad():
        logits, _value = model(batch, mask_illegal=True)
        probs = torch.softmax(logits, dim=-1)
    if was:
        model.train()
    return probs.cpu().numpy()


def dump_tf(args) -> int:
    if not args.weights.is_file():
        return _unverified("weights not found at %s" % args.weights)
    try:
        from alphanl.inference import AlphaNLHoldemAgent
        from alphanl.tf_teacher import teacher_distribution
    except Exception as exc:
        return _unverified("TF teacher import failed (%s)" % exc)
    try:
        agent = AlphaNLHoldemAgent(weights_path=args.weights)
    except Exception as exc:
        return _unverified("TF agent failed to start (%s)" % exc)

    env = agent.env
    rng = np.random.default_rng(args.seed)
    card, action, extra, legal, probs = [], [], [], [], []
    for _ in range(int(args.hands)):
        obs = env.reset()
        done = False
        guard = 0
        while not done and len(probs) < int(args.hands) * 8:
            guard += 1
            if guard > 80:
                break
            distribution, _greedy = teacher_distribution(agent, obs)
            card.append(np.asarray(obs["card_info"], dtype=np.float32))
            action.append(np.asarray(obs["action_info"], dtype=np.float32))
            extra.append(np.asarray(obs["extra_info"], dtype=np.float32))
            legal.append(np.asarray(obs["legal_moves"], dtype=np.float32))
            probs.append(np.asarray(distribution, dtype=np.float32))
            choices = np.flatnonzero(legal[-1] > 0.5)
            obs, _reward, done, _info = env.step(int(rng.choice(choices)))
    if not probs:
        return _unverified("TF dump produced no decisions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(args.out),
        card_info=np.stack(card),
        action_info=np.stack(action),
        extra_info=np.stack(extra),
        legal_mask=np.stack(legal),
        probs=np.stack(probs),
    )
    print("wrote %s decisions=%d" % (args.out, len(probs)))
    return 0


def compare(args) -> int:
    if not args.weights.is_file():
        return _unverified("weights not found at %s" % args.weights)
    if not args.data.is_file():
        return _unverified("TF probability dump not found at %s" % args.data)
    try:
        from alphanl.transformer.conv_net import AlphaNLConvNet, load_tf_weights
    except Exception as exc:
        return _unverified("PyTorch control net failed to import (%s)" % exc)
    model = AlphaNLConvNet()
    try:
        load_tf_weights(model, args.weights)
    except Exception as exc:
        print("FAIL: could not port weights (%s)" % exc)
        return 1
    with np.load(args.data) as blob:
        tf_probs = np.asarray(blob["probs"], dtype=np.float32)
        pt_probs = pytorch_probs(
            model, blob["card_info"], blob["action_info"], blob["extra_info"], blob["legal_mask"]
        )
    if pt_probs.shape != tf_probs.shape:
        print("FAIL: shape %s vs %s" % (pt_probs.shape, tf_probs.shape))
        return 1
    worst = float(np.max(np.abs(pt_probs - tf_probs)))
    print("decisions=%d max_abs_diff=%.3e threshold=%.1e" % (tf_probs.shape[0], worst, args.threshold))
    if worst >= float(args.threshold):
        print("FAIL: action distributions differ by more than the target")
        return 1
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    dump = sub.add_parser("dump-tf", help="Python 3.7: write TF probabilities for shared observations")
    dump.add_argument("--weights", type=Path, default=REPO_ROOT / "third_party" / "AlphaNLHoldem" / "weights" / "c_1048.pkl")
    dump.add_argument("--hands", type=int, default=10)
    dump.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "c1048_probs.npz")
    dump.add_argument("--seed", type=int, default=0)

    check = sub.add_parser("compare", help="PyTorch env: port c_1048 and diff against dump-tf")
    check.add_argument("--weights", type=Path, default=REPO_ROOT / "third_party" / "AlphaNLHoldem" / "weights" / "c_1048.pkl")
    check.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "c1048_probs.npz")
    check.add_argument("--threshold", type=float, default=1e-4)

    args = parser.parse_args()
    if args.cmd == "dump-tf":
        return dump_tf(args)
    if args.cmd == "compare":
        return compare(args)
    return _unverified(
        "this environment was not asked to start TensorFlow. "
        "No in-process comparison is attempted, because TF 1.15 and the training env must stay separate."
    )


if __name__ == "__main__":
    sys.exit(main())

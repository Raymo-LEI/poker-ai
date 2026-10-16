"""Compare a checkpoint's torch forward with the NumPy Botzone forward.

The owner runs this on a laptop that has the real Botzone BC/PPO
checkpoint. It does not train. Argmax of the masked logits must match on
every row. Sampling is not used.

    python3 tools/verify_numpy_vs_torch_ckpt.py \
        --checkpoint path/to/update.pt \
        --npz model.npz \
        --states 1000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "botzone_bots" / "model"))

from alphanl.transformer.botzone_table import BotzoneEnvTable  # noqa: E402
from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.decision_stats import policy_action_mask  # noqa: E402
from alphanl.transformer.tokenizer import BettingHistoryTokenizer  # noqa: E402
from alphanl.transformer.train import build_model  # noqa: E402
from forward import forward_logits, raw_forward  # noqa: E402


def _load_torch_checkpoint(path):
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "model" not in blob or "config" not in blob:
        raise SystemExit("FAIL checkpoint is missing model or config: %s" % path)
    config = ExperimentConfig.from_dict(blob["config"])
    model = build_model(config)
    model.load_state_dict(blob["model"])
    model.eval()
    return blob, config, model


def _load_npz(path):
    archive = np.load(str(path))
    try:
        if "meta_json" not in archive.files:
            raise SystemExit("FAIL npz has no meta_json: %s" % path)
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        weights = {}
        for key in archive.files:
            if key == "meta_json":
                continue
            weights[key] = np.array(archive[key], dtype=np.float32, copy=True)
    finally:
        archive.close()
    return weights, meta


def _collect_states(n_states, seed):
    """Distinct self-play observations. Same walk as the parity test."""

    import torch

    tokenizer = BettingHistoryTokenizer(64)
    rng = random.Random(seed)
    seen = set()
    states = []
    table = BotzoneEnvTable(seed=seed + 2, dealer=0)
    hands = 0
    while len(states) < n_states and hands < max(4000, n_states * 4):
        hands += 1
        if hands % 2 == 0:
            table = BotzoneEnvTable(seed=seed + hands, dealer=hands % 2)
        table.reset()
        guard = 0
        while not table.is_over() and guard < 40 and len(states) < n_states:
            guard += 1
            snap = table.snapshot(table.current_player())
            batch = tokenizer.encode(snap)
            key = batch["token_ids"].numpy().tobytes() + batch["numeric"].numpy().tobytes()
            if key not in seen:
                seen.add(key)
                states.append((batch, float(snap.to_call)))
            legal = [i for i, flag in enumerate(batch["legal_mask"][0].tolist()) if flag > 0.5]
            table.step(int(rng.choice(legal)))
    if len(states) < n_states:
        raise SystemExit("FAIL only gathered %s states" % len(states))
    return states


def _near_tie_count(masked):
    """Rows whose best and second-best logits differ by at most 1e-4."""

    array = np.asarray(masked, dtype=np.float64)
    count = 0
    for row in array:
        ordered = np.partition(row, -2)
        margin = float(np.max(row) - ordered[-2])
        if margin <= 1e-4:
            count += 1
    return count


def verify(checkpoint, npz_path, n_states, seed):
    import torch

    blob, config, model = _load_torch_checkpoint(checkpoint)
    weights, meta = _load_npz(npz_path)
    flag = bool(config.env.mask_free_fold)
    problems = []
    if bool(meta.get("mask_free_fold")) != flag:
        problems.append("mask_free_fold npz=%s checkpoint=%s" % (meta.get("mask_free_fold"), flag))
    if meta.get("sample"):
        problems.append("npz sample is on; this check uses argmax only")
    missing = [name for name in model.state_dict() if name not in weights]
    if missing:
        problems.append("npz missing %s tensors" % len(missing))
    weight_err = 0.0
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name not in weights:
                continue
            ref = tensor.detach().cpu().float().numpy()
            if ref.shape != weights[name].shape:
                problems.append("shape %s" % name)
                continue
            weight_err = max(weight_err, float(np.max(np.abs(ref - weights[name]))))
    if weight_err > 0.0:
        problems.append("weight max abs %s" % weight_err)

    states = _collect_states(n_states, seed)
    free = sum(1 for _batch, owed in states if owed == 0.0)
    facing = sum(1 for _batch, owed in states if owed > 0.0)
    if free == 0 or facing == 0:
        problems.append("states are not diverse free=%s facing=%s" % (free, facing))

    raw_err = 0.0
    masked_err = 0.0
    argmax_match = 0
    argmax_mismatch = 0
    mismatch_near_tie = 0
    near_ties = 0
    rows = 0
    chunk = 64
    for start in range(0, len(states), chunk):
        block = states[start : start + chunk]
        token_ids = torch.cat([item[0]["token_ids"] for item in block], dim=0)
        type_ids = torch.cat([item[0]["type_ids"] for item in block], dim=0)
        numeric = torch.cat([item[0]["numeric"] for item in block], dim=0)
        pad_mask = torch.cat([item[0]["pad_mask"] for item in block], dim=0)
        legal_mask = torch.cat([item[0]["legal_mask"] for item in block], dim=0)
        owed = torch.tensor([item[1] for item in block], dtype=torch.float32)
        torch_batch = {
            "token_ids": token_ids,
            "type_ids": type_ids,
            "numeric": numeric,
            "pad_mask": pad_mask,
            "legal_mask": legal_mask,
        }
        with torch.no_grad():
            torch_raw, _value = model(torch_batch, mask_illegal=False)
            policy = dict(torch_batch)
            policy["legal_mask"] = policy_action_mask(legal_mask, owed, flag)
            torch_masked, _value = model(policy, mask_illegal=True)
        np_batch = {key: value.numpy() for key, value in torch_batch.items()}
        np_raw, _np_value = raw_forward(weights, meta, np_batch)
        np_masked = forward_logits(weights, meta, np_batch, owed.numpy())
        raw_err = max(raw_err, float(np.max(np.abs(np_raw - torch_raw.numpy()))))
        masked_np = np.asarray(np_masked, dtype=np.float32)
        masked_th = torch_masked.numpy()
        masked_err = max(masked_err, float(np.max(np.abs(masked_np - masked_th))))
        np_arg = np.argmax(masked_np, axis=-1)
        th_arg = np.argmax(masked_th, axis=-1)
        same = np_arg == th_arg
        argmax_match += int(np.sum(same))
        argmax_mismatch += int(np.sum(~same))
        rows += int(same.shape[0])
        near_ties += _near_tie_count(masked_th)
        if np.any(~same):
            mismatch_near_tie += _near_tie_count(masked_th[~same])

    if raw_err >= 1e-4:
        problems.append("raw logit max abs %s" % raw_err)
    if masked_err >= 1e-4:
        problems.append("masked logit max abs %s" % masked_err)
    if argmax_mismatch != 0 or argmax_match != rows:
        problems.append(
            "argmax mismatches %s (near-tie among them %s)" % (argmax_mismatch, mismatch_near_tie)
        )
    report = {
        "checkpoint": str(checkpoint),
        "npz": str(npz_path),
        "update": blob.get("update"),
        "mask_free_fold": flag,
        "states": rows,
        "to_call_zero": free,
        "to_call_positive": facing,
        "weight_max_abs": weight_err,
        "raw_logit_max_abs": raw_err,
        "masked_logit_max_abs": masked_err,
        "argmax_match": argmax_match,
        "argmax_mismatch": argmax_mismatch,
        "near_tie_rows_margin_le_1e-4": near_ties,
        "mismatch_rows_that_are_near_ties": mismatch_near_tie,
        "problems": problems,
    }
    report["ok"] = not problems
    return report


def main():
    parser = argparse.ArgumentParser(description="NumPy vs torch argmax check for one checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.states < 1:
        raise SystemExit("FAIL --states must be positive")
    report = verify(args.checkpoint, args.npz, args.states, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ok"]:
        print("PASS")
        return
    print("FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

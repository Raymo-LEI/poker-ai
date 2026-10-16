#!/usr/bin/env python3
"""CPU-side check of a rule-bot BC checkpoint. Does not start PPO.

Loads the checkpoint the same way ``eval/eval_botzone_collapse.py`` loads a
botzone checkpoint (``ExperimentConfig.from_dict`` then ``load_checkpoint``).
Reports, on the held-out games and on states visited by sampling the policy
against itself:

- mean legal entropy (gate: > 0.2)
- ``free_check_fold_ratio`` (gate: exactly 0; a split with no free check fails)
- ``fold_facing_ratio`` versus the rule bot on the same states (gate: within
  5 percentage points)
- per-action frequencies versus the rule bot (gate: each within 5 percentage
  points). Policy frequencies are the mean masked probabilities, not one
  noisy sample. The rule-bot side is the label rate.
- the train / val KL curve stored at training time
- NaN / Inf in that curve, the probabilities, and the parameters

A tiny run often misses the frequency gates. The thresholds are not tuned.
Exit status is 0 only when every gate passes.

    PYTHONPATH=src python3 scripts/check_bc_rulebot.py \\
        --ckpt runs/bc_rulebot_train/bc.pt \\
        --data runs/bc_rulebot \\
        --on-policy-hands 40 --seed 42 --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from alphanl.botzone.actions import (  # noqa: E402
    ACTION_NAMES,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    NUM_ACTIONS,
)
from alphanl.transformer.botzone_table import BotzoneEnvTable  # noqa: E402
from alphanl.transformer.decision_stats import policy_action_mask  # noqa: E402
from alphanl.transformer.model import entropy_from_probs, masked_log_probs  # noqa: E402
from alphanl.transformer.tokenizer import batch_to_device  # noqa: E402

ENTROPY_MIN = 0.2
FREQ_TOL = 0.05


def _load(name: str, path: Path):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_entropy_and_probs(model, batches: List[dict], device: torch.device):
    """Concatenated per-row entropy and masked probabilities. Model stays in eval."""

    entropies = []
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moved = batch_to_device(batch, device)
            logits, _value = model(moved, mask_illegal=True)
            _masked, log_probs, prob, legal = masked_log_probs(logits, moved["legal_mask"])
            ent = entropy_from_probs(prob, log_probs, legal)
            if not torch.isfinite(ent).all() or not torch.isfinite(prob).all():
                raise RuntimeError("non-finite policy distribution")
            entropies.append(ent.detach().cpu())
            probs.append(prob.detach().cpu())
    if not entropies:
        raise RuntimeError("no rows to score")
    return torch.cat(entropies), torch.cat(probs)


def _chunks(rows: dict, width: int) -> List[dict]:
    n = int(rows["legal_mask"].shape[0])
    out = []
    for start in range(0, n, width):
        stop = min(start + width, n)
        out.append({key: value[start:stop] for key, value in rows.items()})
    return out


def _rates(actions: np.ndarray, mask: np.ndarray) -> Optional[float]:
    chosen = np.asarray(mask, dtype=bool)
    if int(chosen.sum()) == 0:
        return None
    return float(np.mean(np.asarray(actions)[chosen] == FOLD_ACTION))


def _pack_split(
    *,
    name: str,
    entropy: torch.Tensor,
    probs: torch.Tensor,
    labels: np.ndarray,
    to_call: np.ndarray,
    env_legal: np.ndarray,
    policy_legal: np.ndarray,
) -> dict:
    label_freq = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for action in labels:
        label_freq[int(action)] += 1.0
    label_freq /= float(len(labels))
    policy_freq = probs.mean(dim=0).numpy().astype(np.float64)
    freq_diff = {ACTION_NAMES[i]: abs(float(policy_freq[i]) - float(label_freq[i])) for i in range(NUM_ACTIONS)}
    free = (to_call == 0.0) & (env_legal[:, CHECK_CALL_ACTION] > 0.5)
    facing = (to_call > 0.0) & (env_legal[:, FOLD_ACTION] > 0.5)
    # One draw from the masked distribution, only to measure free-check folds.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    drawn = torch.multinomial(probs, num_samples=1, generator=generator).reshape(-1).numpy()
    if int(facing.sum()) == 0:
        policy_facing = None
        label_facing = None
    else:
        policy_facing = float(probs[facing, FOLD_ACTION].mean())
        label_facing = float(np.mean(labels[facing] == FOLD_ACTION))
    free_fold = _rates(drawn, free)
    mask_allows_free_fold = bool(np.any(policy_legal[free, FOLD_ACTION] > 0.5)) if int(free.sum()) else False
    return {
        "split": name,
        "n": int(len(labels)),
        "legal_entropy": float(entropy.mean()),
        "free_check_count": int(free.sum()),
        "free_check_fold_ratio": free_fold,
        "free_check_mask_allows_fold": mask_allows_free_fold,
        "facing_count": int(facing.sum()),
        "fold_facing_ratio_policy": policy_facing,
        "fold_facing_ratio_rule": label_facing,
        "action_freq_policy": {ACTION_NAMES[i]: float(policy_freq[i]) for i in range(NUM_ACTIONS)},
        "action_freq_rule": {ACTION_NAMES[i]: float(label_freq[i]) for i in range(NUM_ACTIONS)},
        "action_freq_abs_diff": freq_diff,
    }


def _gates(block: dict) -> Dict[str, dict]:
    entropy = block["legal_entropy"]
    free = block["free_check_fold_ratio"]
    free_count = block["free_check_count"]
    facing_policy = block["fold_facing_ratio_policy"]
    facing_rule = block["fold_facing_ratio_rule"]
    gates = {
        "legal_entropy_gt_0.2": {
            "value": entropy,
            "ok": _finite(entropy) and float(entropy) > ENTROPY_MIN,
        },
        "free_check_fold_ratio_eq_0": {
            "value": free,
            "count": free_count,
            "ok": free_count > 0 and free == 0.0 and not block["free_check_mask_allows_fold"],
        },
        "fold_facing_within_5pp": {
            "policy": facing_policy,
            "rule": facing_rule,
            "ok": (
                facing_policy is not None
                and facing_rule is not None
                and _finite(facing_policy)
                and _finite(facing_rule)
                and abs(float(facing_policy) - float(facing_rule)) <= FREQ_TOL
            ),
        },
    }
    freq_ok = True
    for name, diff in block["action_freq_abs_diff"].items():
        ok = _finite(diff) and float(diff) <= FREQ_TOL
        gates[f"freq_{name}_within_5pp"] = {"value": diff, "ok": ok}
        freq_ok = freq_ok and ok
    gates["all_action_freqs_within_5pp"] = {"ok": freq_ok}
    return gates


def _val_rows(arrays: dict, val_games: Sequence[int]) -> dict:
    games = set(int(game) for game in val_games)
    mask = np.asarray([int(game) in games for game in arrays["game_id"].tolist()], dtype=bool)
    if int(mask.sum()) == 0:
        raise RuntimeError("validation game ids match no rows")

    def take(key, dtype=None):
        value = np.asarray(arrays[key])
        value = value[mask]
        if dtype is not None:
            value = value.astype(dtype)
        return value

    return {
        "token_ids": torch.from_numpy(take("token_ids", np.int64)),
        "type_ids": torch.from_numpy(take("type_ids", np.int64)),
        "numeric": torch.from_numpy(take("numeric", np.float32)),
        "pad_mask": torch.from_numpy(take("pad_mask").astype(bool)),
        "legal_mask": torch.from_numpy(take("legal_mask", np.float32)),
        "labels": take("actions", np.int64),
        "to_call": take("to_call", np.float32),
        "env_legal": take("env_legal_mask", np.float32),
    }


def _on_policy_rows(model, tokenizer, *, hands: int, seed: int, device: torch.device, mask_free_fold: bool) -> dict:
    gen = _load("gen_rulebot_bc", REPO_ROOT / "scripts" / "gen_rulebot_bc.py")
    from alphanl.botzone.rule_bot import BotzoneRuleBot

    table = BotzoneEnvTable(seed=int(seed))
    bot = BotzoneRuleBot()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    stored = []
    for _hand in range(int(hands)):
        table.reset()
        guard = 0
        while not table.is_over():
            guard += 1
            if guard > 80:
                raise RuntimeError("on-policy hand exceeded 80 decisions")
            seat = int(table.current_player())
            encoded = gen.encode_policy_observation(
                table, seat, tokenizer, mask_free_fold=bool(mask_free_fold)
            )
            batch = {
                "token_ids": torch.from_numpy(encoded["token_ids"]).unsqueeze(0),
                "type_ids": torch.from_numpy(encoded["type_ids"]).unsqueeze(0),
                "numeric": torch.from_numpy(encoded["numeric"]).unsqueeze(0),
                "pad_mask": torch.from_numpy(encoded["pad_mask"]).unsqueeze(0),
                "legal_mask": torch.from_numpy(encoded["legal_mask"]).unsqueeze(0),
            }
            # Re-apply through the same helper PPO uses, in case the stored mask drifted.
            owed = float(encoded["to_call"])
            if mask_free_fold:
                batch["legal_mask"] = policy_action_mask(batch["legal_mask"], owed, True)
            moved = batch_to_device(batch, device)
            with torch.no_grad():
                logits, _value = model(moved, mask_illegal=True)
                _masked, log_probs, prob, legal = masked_log_probs(logits, moved["legal_mask"])
                ent = entropy_from_probs(prob, log_probs, legal)
            if not torch.isfinite(ent).all() or not torch.isfinite(prob).all():
                raise RuntimeError("non-finite on-policy distribution")
            drawn = torch.multinomial(prob.detach().cpu(), num_samples=1, generator=generator)
            action = int(drawn.reshape(-1)[0])
            _raw, _response, target = gen.rulebot_target(
                table._table, bot, mask_free_fold=bool(mask_free_fold)
            )
            if not gen.decision_is_counted(batch["legal_mask"].reshape(-1).numpy()):
                table.step(action)
                continue
            stored.append(
                {
                    "entropy": float(ent.reshape(-1)[0]),
                    "probs": prob.detach().cpu().reshape(-1).numpy().astype(np.float32),
                    "label": int(target),
                    "to_call": np.float32(owed),
                    "env_legal": encoded["env_legal_mask"].astype(np.float32),
                    "legal_mask": batch["legal_mask"].reshape(-1).numpy().astype(np.float32),
                }
            )
            table.step(action)
    if not stored:
        raise RuntimeError("on-policy rollout recorded no decisions")
    return {
        "entropy": torch.tensor([row["entropy"] for row in stored], dtype=torch.float32),
        "probs": torch.from_numpy(np.stack([row["probs"] for row in stored])),
        "labels": np.asarray([row["label"] for row in stored], dtype=np.int64),
        "to_call": np.asarray([row["to_call"] for row in stored], dtype=np.float32),
        "env_legal": np.stack([row["env_legal"] for row in stored]),
        "legal_mask": np.stack([row["legal_mask"] for row in stored]),
    }


def evaluate_checkpoint(
    *,
    ckpt: Path,
    data: Path,
    on_policy_hands: int,
    seed: int,
    device: str = "cpu",
) -> dict:
    gen = _load("gen_rulebot_bc", REPO_ROOT / "scripts" / "gen_rulebot_bc.py")
    collapse = _load("eval_botzone_collapse_for_bc", REPO_ROOT / "eval" / "eval_botzone_collapse.py")
    arrays, meta = gen.load_dataset(data, check_legal=True)
    device_t = torch.device(device)
    model, tokenizer, cfg, path, update = collapse.load_botzone_model(
        Path(ckpt).parent, device_t, ckpt_path=Path(ckpt)
    )
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    bc_meta = blob.get("bc_metadata")
    if not isinstance(bc_meta, dict):
        raise RuntimeError(f"{path} has no bc_metadata")
    gen.assert_legal_py_sha256(str(bc_meta.get("legal_py_sha256", "")), where=str(path))
    if str(bc_meta.get("legal_py_sha256")) != str(meta["env_version"]["legal_py_sha256"]):
        raise gen.LegalPyMismatch("checkpoint legal_py_sha256 does not match the dataset")
    non_finite_param = False
    for param in model.parameters():
        if not torch.isfinite(param).all():
            non_finite_param = True
    curve = list(bc_meta.get("kl_curve") or [])
    curve_finite = all(_finite(row.get("train_kl")) and _finite(row.get("val_kl")) for row in curve) and bool(curve)
    val = _val_rows(arrays, bc_meta["val_game_ids"])
    obs = {
        "token_ids": val["token_ids"],
        "type_ids": val["type_ids"],
        "numeric": val["numeric"],
        "pad_mask": val["pad_mask"],
        "legal_mask": val["legal_mask"],
    }
    ent, prob = _mean_entropy_and_probs(model, _chunks(obs, 64), device_t)
    validation = _pack_split(
        name="validation",
        entropy=ent,
        probs=prob,
        labels=val["labels"],
        to_call=val["to_call"],
        env_legal=val["env_legal"],
        policy_legal=val["legal_mask"].numpy(),
    )
    on_policy = _on_policy_rows(
        model,
        tokenizer,
        hands=int(on_policy_hands),
        seed=int(seed),
        device=device_t,
        mask_free_fold=bool(cfg.env.mask_free_fold),
    )
    on_block = _pack_split(
        name="on_policy",
        entropy=on_policy["entropy"],
        probs=on_policy["probs"],
        labels=on_policy["labels"],
        to_call=on_policy["to_call"],
        env_legal=on_policy["env_legal"],
        policy_legal=on_policy["legal_mask"],
    )
    gates = {
        "validation": _gates(validation),
        "on_policy": _gates(on_block),
        "kl_curve_finite": {"ok": curve_finite and not non_finite_param, "n_epochs": len(curve)},
    }
    ok = curve_finite and not non_finite_param
    for group in (gates["validation"], gates["on_policy"]):
        for item in group.values():
            ok = ok and bool(item["ok"])
    train_first = curve[0]["train_kl"] if curve else None
    train_last = curve[-1]["train_kl"] if curve else None
    val_first = curve[0]["val_kl"] if curve else None
    val_last = curve[-1]["val_kl"] if curve else None
    return {
        "ckpt": str(path),
        "update": int(update),
        "mask_free_fold": bool(bc_meta.get("mask_free_fold")),
        "legal_py_sha256": bc_meta.get("legal_py_sha256"),
        "data_seed": bc_meta.get("data_seed"),
        "nan_inf": bool(non_finite_param or not curve_finite),
        "kl_curve": curve,
        "train_kl_first": train_first,
        "train_kl_last": train_last,
        "val_kl_first": val_first,
        "val_kl_last": val_last,
        "train_kl_decreased": (
            train_first is not None and train_last is not None and float(train_last) < float(train_first)
        ),
        "val_kl_not_up": (
            val_first is not None and val_last is not None and float(val_last) <= float(val_first)
        ),
        "validation": validation,
        "on_policy": on_block,
        "gates": gates,
        "ok": bool(ok),
        "thresholds": {"legal_entropy_min": ENTROPY_MIN, "frequency_abs_tol": FREQ_TOL},
    }


def render_report(payload: dict) -> str:
    lines = [
        f"ckpt: {payload['ckpt']} update={payload['update']} ok={payload['ok']}",
        f"nan_inf: {payload['nan_inf']}",
        (
            f"KL train {payload['train_kl_first']} -> {payload['train_kl_last']} "
            f"decreased={payload['train_kl_decreased']}"
        ),
        (
            f"KL val   {payload['val_kl_first']} -> {payload['val_kl_last']} "
            f"not_up={payload['val_kl_not_up']}"
        ),
    ]
    for row in payload["kl_curve"]:
        lines.append(f"  epoch {row['epoch']} train_kl {row['train_kl']:.6f} val_kl {row['val_kl']:.6f}")
    for name in ("validation", "on_policy"):
        block = payload[name]
        lines.append(
            f"{name}: n={block['n']} entropy={block['legal_entropy']:.4f} "
            f"free_check_fold={block['free_check_fold_ratio']} "
            f"(n_free={block['free_check_count']}) "
            f"fold_facing policy={block['fold_facing_ratio_policy']} "
            f"rule={block['fold_facing_ratio_rule']} (n_facing={block['facing_count']})"
        )
        lines.append("  policy " + " ".join(f"{k}={v:.3f}" for k, v in block["action_freq_policy"].items()))
        lines.append("  rule   " + " ".join(f"{k}={v:.3f}" for k, v in block["action_freq_rule"].items()))
        failed = [key for key, item in payload["gates"][name].items() if not item["ok"]]
        lines.append("  failed: " + (", ".join(failed) if failed else "none"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--on-policy-hands", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="Default: <ckpt dir>/bc_check.json")
    args = parser.parse_args()
    payload = evaluate_checkpoint(
        ckpt=args.ckpt,
        data=args.data,
        on_policy_hands=args.on_policy_hands,
        seed=args.seed,
        device=args.device,
    )
    text = render_report(payload)
    print(text)
    out = args.out if args.out is not None else Path(args.ckpt).with_name("bc_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

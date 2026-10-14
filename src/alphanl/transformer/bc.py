"""Behavior-cloning warm start for either encoder.

The loss is cross-entropy to a teacher action, or KL from the teacher
distribution to the student. Illegal actions are removed from the teacher and
from the student softmax (the same ``masked_log_probs`` PPO uses). A teacher
that put mass on an illegal action is renormalized over the legal set; a row
with no legal teacher mass falls back to uniform over the legal actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .model import masked_log_probs
from .tokenizer import TensorBatch, slice_batch


def legalize_teacher(teacher: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Zero illegal mass and renormalize. Uniform over legal actions if nothing remains."""

    teacher = teacher.float().clamp_min(0)
    legal_f = legal.float()
    teacher = torch.where(legal, teacher, torch.zeros_like(teacher))
    denom = teacher.sum(dim=-1, keepdim=True)
    uniform = legal_f / legal_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cleaned = teacher / denom.clamp_min(1e-8)
    return torch.where(denom > 0, cleaned, uniform)


def bc_losses(
    model: torch.nn.Module,
    batch: TensorBatch,
    *,
    teacher_probs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    kind: str = "kl",
) -> tuple:
    """One forward and a masked BC loss. ``kind`` is ``kl`` or ``ce``."""

    if kind not in ("kl", "ce"):
        raise ValueError("bc loss kind must be 'kl' or 'ce'")
    logits, _values = model(batch, mask_illegal=True)
    _, log_probs, probs, legal = masked_log_probs(logits, batch["legal_mask"])
    if kind == "kl":
        if teacher_probs is None:
            raise ValueError("kl loss needs teacher_probs")
        teacher = legalize_teacher(teacher_probs, legal)
        log_t = torch.where(teacher > 0, teacher.log(), torch.zeros_like(teacher))
        # Illegal student log-probs are large and negative. Multiply only where legal.
        kl = torch.where(legal, teacher * (log_t - log_probs), torch.zeros_like(log_probs)).sum(dim=-1)
        loss = kl.mean()
        with torch.no_grad():
            ent = -(teacher * log_t).sum(dim=-1).mean()
        metrics = {"loss": float(loss.detach()), "kl": float(loss.detach()), "teacher_entropy": float(ent)}
        return loss, metrics
    if actions is None:
        raise ValueError("ce loss needs actions")
    nll = -log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    loss = nll.mean()
    metrics = {"loss": float(loss.detach()), "nll": float(loss.detach())}
    # Touch probs so a caller can assert the mask without a second forward.
    metrics["illegal_prob_max"] = float(probs.masked_fill(legal, 0).max().detach())
    return loss, metrics


def bc_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TensorBatch,
    *,
    teacher_probs: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
    kind: str,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """Minibatch BC. Tensors and the model share a device."""

    num_samples = int(batch["legal_mask"].shape[0])
    if num_samples == 0:
        raise ValueError("empty BC batch")
    device = batch["legal_mask"].device
    totals = {"loss": 0.0}
    updates = 0
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, int(minibatch_size)):
            index = order[start : start + int(minibatch_size)]
            mb = slice_batch(batch, index)
            mb_probs = None if teacher_probs is None else teacher_probs.index_select(0, index)
            mb_actions = None if actions is None else actions.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = bc_losses(
                model, mb, teacher_probs=mb_probs, actions=mb_actions, kind=kind
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
    if updates == 0:
        raise ValueError("minibatch_size produced no updates")
    return {key: value / updates for key, value in totals.items()}


def load_bc_npz(path: Path | str) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as blob:
        return {key: blob[key] for key in blob.files}


def conv_tensors_from_npz(arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    required = ("card_info", "action_info", "extra_info", "legal_mask", "probs", "actions")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError("BC npz is missing %s" % missing)
    return {
        "card_info": torch.from_numpy(np.asarray(arrays["card_info"], dtype=np.float32)),
        "action_info": torch.from_numpy(np.asarray(arrays["action_info"], dtype=np.float32)),
        "extra_info": torch.from_numpy(np.asarray(arrays["extra_info"], dtype=np.float32)),
        "legal_mask": torch.from_numpy(np.asarray(arrays["legal_mask"], dtype=np.float32)),
        "probs": torch.from_numpy(np.asarray(arrays["probs"], dtype=np.float32)),
        "actions": torch.from_numpy(np.asarray(arrays["actions"], dtype=np.int64)),
    }


def transformer_tensors_from_npz(arrays: Dict[str, np.ndarray], max_seq_len: int) -> Dict[str, torch.Tensor]:
    """Rebuild token batches from the dump's per-decision JSON.

    The dump runs in the TF 1.15 env and does not import this package. It stores
    the wrapper history and the rlcard raw fields; this function replays them
    with the same tokenizer PPO uses.
    """

    if "meta_json" not in arrays:
        raise KeyError(
            "transformer BC needs meta_json in the npz (re-run scripts/dump_c1048_bc.py)"
        )
    from .history import build_snapshot
    from .tokenizer import BettingHistoryTokenizer, cat_batches

    tokenizer = BettingHistoryTokenizer(max_seq_len=max_seq_len)
    rows = []
    for raw_text in arrays["meta_json"]:
        meta = json.loads(str(raw_text))
        raw_obs = {
            "hand": meta["hand"],
            "public_cards": meta.get("public_cards") or [],
            "all_chips": meta["all_chips"],
            "stakes": meta["stakes"],
            "pot": meta["pot"],
            "stage": meta["stage"],
            "legal_actions": meta["legal_actions"],
            "current_player": meta["current_player"],
        }
        snapshot = build_snapshot(
            meta["history"],
            raw_obs,
            int(meta["current_player"]),
            int(meta["sb_seat"]),
            big_blind=float(meta.get("big_blind", 2)),
            small_blind=int(meta.get("small_blind", 1)),
            starting_stack=float(meta.get("starting_stack", 100)),
        )
        rows.append(tokenizer.encode(snapshot))
    if not rows:
        raise ValueError("BC npz has no decisions")
    batch = cat_batches(rows)
    batch["probs"] = torch.from_numpy(np.asarray(arrays["probs"], dtype=np.float32))
    batch["actions"] = torch.from_numpy(np.asarray(arrays["actions"], dtype=np.int64))
    if batch["probs"].shape[0] != batch["legal_mask"].shape[0]:
        raise ValueError("meta_json length does not match probs")
    return batch


def dataset_for_encoder(cfg: ExperimentConfig, arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    if cfg.encoder == "alphanl_conv":
        return conv_tensors_from_npz(arrays)
    if cfg.encoder == "transformer":
        return transformer_tensors_from_npz(arrays, cfg.model.max_seq_len)
    raise ValueError("unknown encoder %r" % cfg.encoder)


def save_bc_checkpoint(
    path: Path | str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> Path:
    """A PPO-compatible checkpoint.

    ``update`` is -1 so ``--resume`` of this file starts PPO at update 0.
    ``--init-weights`` loads the model only and also starts at 0.
    """

    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        update=-1,
        config=config,
        pool=None,
        scaler=None,
        envs=None,
        actor_rng=None,
        keep_last=0,
    )

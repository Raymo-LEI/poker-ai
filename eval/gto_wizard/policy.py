"""Load a frozen Stage 2 residual checkpoint and turn one HUNL decision into an action id.

The network is whatever ``alphanl.stage2.train.load_checkpoint`` already
builds for ``stage2_match_residual_v1``. This module does not read
``model_structure`` JSON and does not construct a second encoder or PPO net.
A file that is not that checkpoint, including a CFR ``SnapshotPolicy``
archive, is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import SCHEMA_A5


class PolicyLoadError(ValueError):
    """The weights or sidecar config cannot be used as the frozen Stage 2 policy."""


_MISSING = object()


def _mask_block(config: dict):
    root = config.get("botzone_native_allin_mask", _MISSING)
    policy = config.get("policy")
    nested = _MISSING
    if isinstance(policy, dict) and "botzone_native_allin_mask" in policy:
        nested = policy["botzone_native_allin_mask"]
    if root is not _MISSING and nested is not _MISSING and root != nested:
        raise PolicyLoadError("config all-in mask blocks disagree")
    if root is not _MISSING:
        return root
    if nested is not _MISSING:
        return nested
    return _MISSING


def load_frozen_policy(policy_path, config_path, *, device: str = "cpu"):
    """Return ``(model, blob)`` from the existing Stage 2 checkpoint loader."""

    if device not in ("cpu", "cuda"):
        raise PolicyLoadError("device must be cpu or cuda")
    weights = Path(policy_path)
    sidecar = Path(config_path)
    if not weights.is_file() or not sidecar.is_file():
        raise PolicyLoadError("policy and config must be files")
    try:
        config = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyLoadError("config must be a JSON object") from exc
    if not isinstance(config, dict):
        raise PolicyLoadError("config must be a JSON object")
    if "schema_id" in config and config["schema_id"] != SCHEMA_A5:
        raise PolicyLoadError("config schema does not match the 5-action head")
    if "num_actions" in config and config["num_actions"] != 5:
        raise PolicyLoadError("config num_actions does not match the 5-action head")
    block = _mask_block(config)

    from alphanl.stage2.train import load_checkpoint

    try:
        model, blob = load_checkpoint(weights, device=device)
        name = type(model).__name__
        if name == "SnapshotPolicy":
            raise PolicyLoadError("refusing SnapshotPolicy weights")
        if name != "ResidualExploiterPolicy":
            raise PolicyLoadError("refusing a checkpoint that is not a Stage 2 residual policy")
        width = int(model.delta_head.out_features)
    except PolicyLoadError:
        raise
    except Exception as exc:
        raise PolicyLoadError("refusing unrecognized Stage 2 checkpoint") from exc
    if width != 5:
        raise PolicyLoadError("refusing a policy head that is not the frozen 5-action Stage 2 head")
    if block is not _MISSING:
        from alphanl.stage2.allin_mask import mask_config

        try:
            left = mask_config(block)
            right = mask_config(model.settings.get("botzone_native_allin_mask"))
        except ValueError as exc:
            raise PolicyLoadError("config all-in mask does not match the checkpoint") from exc
        if left != right:
            raise PolicyLoadError("config all-in mask does not match the checkpoint")
    model.eval()
    return model, blob


class Stage2Decider:
    """Sample the frozen policy the same way Stage 2 evaluation calls ``model.act``."""

    def __init__(self, model) -> None:
        import torch
        from alphanl.cfr.actions import CFRActionGuard
        from alphanl.transformer.hunl import HunlTokenizer

        self._torch = torch
        self.model = model.eval()
        self.guard = CFRActionGuard.from_config(model.config)
        self.mask = model.settings.get("botzone_native_allin_mask")
        self.tokenizer = HunlTokenizer(
            model.config.model.max_seq_len, **model.config.tokenizer_kwargs()
        )
        try:
            self.device = next(model.parameters()).device
        except StopIteration as exc:
            raise PolicyLoadError("Stage 2 policy has no parameters") from exc

    def __call__(self, table, hero_seat: int, ctx) -> int:
        from alphanl.stage2.allin_mask import mask_snapshot

        from .adapter_hunl import AdapterError

        snap = mask_snapshot(self.guard.apply(table.snapshot(int(hero_seat))), self.mask)
        batch = self.tokenizer.encode(snap)
        batch["match_ctx"] = ctx.tensor(self.model.context_fields)
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if self._torch.is_tensor(value) else value
        action, *_rest = self.model.act(moved)
        if self._torch.is_tensor(action):
            action_id = int(action.detach().cpu().reshape(-1)[0].item())
        else:
            action_id = int(action)
        if action_id not in snap.legal_action_ids:
            raise AdapterError(f"policy chose illegal action {action_id}")
        return action_id

"""Regret estimation using the unchanged step-3 Transformer backbone.

The PPO model keeps its five-way policy head, value head, and checkpoint
format. This wrapper reads its existing CLS/GELU features and adds one
five-way regret head. It predicts signed sampled advantages; the positive
part is used only when constructing a strategy, never as a regression
target. Positive-regret matching alone does not implement CFR+'s clipped
cumulative-regret updates.
"""

from __future__ import annotations

import copy
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.tokenizer import TensorBatch


def regret_matching(
    regrets: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    fallback: str = "uniform",
) -> torch.Tensor:
    """Normalize positive legal regrets in fp32, with exact zero illegal mass.

    Inputs have the same shape ``[..., num_actions]``. When every legal
    regret is nonpositive, ``uniform`` explores all legal actions; ``argmax``
    chooses the largest raw legal regret (ties use the first action).
    An empty legal set is an invalid decision state and raises an error.
    """

    if fallback not in ("uniform", "argmax", "softmax"):
        raise ValueError("fallback must be uniform, argmax or softmax")
    if regrets.ndim < 1 or regrets.shape != legal_mask.shape:
        raise ValueError("regrets and legal_mask must have the same non-scalar shape")
    if regrets.shape[-1] == 0:
        raise ValueError("regrets must have at least one action")
    if regrets.device != legal_mask.device:
        raise ValueError("regrets and legal_mask must be on the same device")
    if not torch.isfinite(legal_mask).all():
        raise ValueError("legal_mask must be finite")
    legal = legal_mask > 0.5
    if not legal.any(dim=-1).all():
        raise ValueError("every decision must have at least one legal action")
    raw = regrets.float()
    if not torch.isfinite(raw[legal]).all():
        raise ValueError("legal regrets must be finite in fp32")

    # Scaling before summation avoids overflow for large finite regrets.
    # One scaled entry is 1 whenever a positive legal regret exists.
    positive = raw.masked_fill(~legal, 0.0).clamp_min(0.0)
    scale = positive.amax(dim=-1, keepdim=True)
    scaled = positive / torch.where(scale > 0, scale, torch.ones_like(scale))
    total = scaled.sum(dim=-1, keepdim=True)
    matched = scaled / total.clamp_min(1.0)
    if fallback == "uniform":
        default = legal.float() / legal.sum(dim=-1, keepdim=True)
    elif fallback == "argmax":
        best = raw.masked_fill(~legal, -torch.inf).argmax(dim=-1)
        default = F.one_hot(best, num_classes=raw.shape[-1]).float()
    else:
        # Unit-temperature smoothing + 5% exploration floor. Subtracting
        # the legal maximum avoids overflow; equal/zero regrets are uniform.
        logits = raw.masked_fill(~legal, -torch.inf)
        logits = logits - logits.amax(-1, keepdim=True)
        default = .95 * logits.softmax(-1) + .05 * legal.float() / legal.sum(-1, keepdim=True)
    return torch.where(total > 0, matched, default)


class RegretNetwork(nn.Module):
    """An independent Transformer copy plus a raw five-action regret head.

    Passing a PPO backbone copies its weights and architecture so training
    or freezing this wrapper cannot change the caller's PPO model. The
    original policy/value heads remain in ``backbone.state_dict()`` for
    strict checkpoint compatibility, but are unused and frozen here.

    ``forward`` returns signed predictions, without masking or softmax.
    ``strategy`` applies regret matching and respects the module's current
    training mode. Traversal callers should use ``eval()`` and ``no_grad()``.
    """

    def __init__(
        self,
        backbone: PokerTransformer | None = None,
        *,
        fallback: str = "uniform",
    ) -> None:
        super().__init__()
        if fallback not in ("uniform", "argmax", "softmax"):
            raise ValueError("fallback must be uniform, argmax or softmax")
        if backbone is not None and not isinstance(backbone, PokerTransformer):
            raise TypeError("backbone must be a PokerTransformer")
        self.backbone = PokerTransformer() if backbone is None else copy.deepcopy(backbone)
        self.fallback = fallback
        if self.backbone.num_actions != 5:
            raise ValueError("SD-CFR requires the existing five-action abstraction")
        for head in (self.backbone.policy_head, self.backbone.value_head):
            head.requires_grad_(False)
        reference = self.backbone.policy_head.weight
        self.regret_head = nn.Linear(
            self.backbone.d_model,
            self.backbone.num_actions,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.zeros_(self.regret_head.weight)
        nn.init.zeros_(self.regret_head.bias)

    @classmethod
    def from_ppo_state_dict(
        cls,
        weights: Mapping[str, torch.Tensor],
        *,
        fallback: str = "uniform",
        **backbone_kwargs,
    ) -> "RegretNetwork":
        """Strictly load a legacy PPO state dictionary, then add a zero head.

        Pass the original model configuration as ``backbone_kwargs`` when
        loading a preset other than the default laptop 8GB architecture.
        Optimizer, value targets, and PPO logits are not converted to regrets.
        """

        backbone = PokerTransformer(**backbone_kwargs)
        backbone.load_state_dict(weights, strict=True)
        return cls(backbone, fallback=fallback)

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        cls = self.backbone.encode_tokens(
            batch["token_ids"],
            batch["type_ids"],
            batch["numeric"],
            batch["pad_mask"],
        )
        features = self.backbone.trunk(cls)
        # Match the PPO heads' fp32 precision under automatic mixed precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.regret_head(features.float())

    def strategy(self, batch: TensorBatch) -> torch.Tensor:
        return regret_matching(self(batch), batch["legal_mask"], fallback=self.fallback)


class SchemaRegretNetwork(nn.Module):
    """Regret head whose width is the ActionSpec table, not a fixed five.

    The production ``RegretNetwork`` stays on ``botzone_5action_v1``. This
    class is the A5/A6 pilot head. A checkpoint from another schema, including
    a five-wide head, is rejected instead of reshaped.
    """

    def __init__(
        self,
        schema,
        *,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 32,
        max_seq_len: int = 8,
        dropout: float = 0.0,
        fallback: str = "uniform",
    ) -> None:
        super().__init__()
        from alphanl.botzone.action_spec import artifact_stamp

        if fallback not in ("uniform", "argmax", "softmax"):
            raise ValueError("fallback must be uniform, argmax or softmax")
        self.schema_stamp = artifact_stamp(schema)
        self.num_actions = int(schema.num_actions)
        self.fallback = fallback
        self.backbone = PokerTransformer(
            num_actions=self.num_actions,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        for head in (self.backbone.policy_head, self.backbone.value_head):
            head.requires_grad_(False)
        reference = self.backbone.policy_head.weight
        self.regret_head = nn.Linear(
            d_model,
            self.num_actions,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.zeros_(self.regret_head.weight)
        nn.init.zeros_(self.regret_head.bias)

    def forward(self, batch: TensorBatch) -> torch.Tensor:
        from alphanl.botzone.action_spec import SchemaMismatch

        legal = batch["legal_mask"]
        if legal.shape[-1] != self.num_actions:
            raise SchemaMismatch(
                f"legal mask width {legal.shape[-1]} != schema width {self.num_actions}"
            )
        cls = self.backbone.encode_tokens(
            batch["token_ids"],
            batch["type_ids"],
            batch["numeric"],
            batch["pad_mask"],
        )
        features = self.backbone.trunk(cls)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.regret_head(features.float())

    def strategy(self, batch: TensorBatch) -> torch.Tensor:
        return regret_matching(self(batch), batch["legal_mask"], fallback=self.fallback)

    def checkpoint_payload(self) -> dict:
        return {**self.schema_stamp, "state_dict": self.state_dict()}

    def load_matching_checkpoint(self, blob: Mapping) -> None:
        from alphanl.botzone.action_spec import EXPERIMENT_SCHEMAS, SchemaMismatch, require_schema

        schema_id = blob.get("action_schema_id")
        if schema_id not in EXPERIMENT_SCHEMAS:
            raise SchemaMismatch(
                f"fail closed: checkpoint schema {schema_id!r} is not an A5/A6 head"
            )
        require_schema(blob, EXPERIMENT_SCHEMAS[str(schema_id)])
        if EXPERIMENT_SCHEMAS[str(schema_id)].num_actions != self.num_actions:
            raise SchemaMismatch("fail closed: refusing to reshape the regret head")
        state = blob.get("state_dict")
        if not isinstance(state, Mapping) or "regret_head.weight" not in state:
            raise SchemaMismatch("fail closed: checkpoint has no regret head")
        weight = state["regret_head.weight"]
        if tuple(weight.shape) != tuple(self.regret_head.weight.shape):
            raise SchemaMismatch(
                f"fail closed: regret head shape {tuple(weight.shape)} "
                f"!= {tuple(self.regret_head.weight.shape)}; not reshaping"
            )
        self.load_state_dict(state, strict=True)

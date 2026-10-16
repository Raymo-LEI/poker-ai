"""Transformer encoder plus the two heads AlphaNLHoldem already separates.

The encoder consumes betting-history tokens. A linear policy head emits logits
over the same 5 abstract actions as ``NlHoldemNet``, with illegal actions
masked before the softmax. A linear value head reads the same CLS vector.
"""

from __future__ import annotations

import inspect
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import NUM_ACTIONS
from .tokenizer import NUM_NUMERIC, NUM_TYPES, VOCAB_SIZE, TensorBatch

ILLEGAL_LOGIT = -1.0e9


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def masked_log_probs(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked logits, log-probs, probs, and a boolean legal mask.

    Rows with no legal action are treated as fully legal so softmax stays
    finite. Callers that build observations should not produce that row.
    """

    # Always fp32: ILLEGAL_LOGIT (-1e9) overflows fp16 to -inf and an all -inf row
    # gives NaN after softmax. This also runs outside autocast.
    logits = logits.float()
    legal = legal_mask > 0.5
    none_legal = ~legal.any(dim=-1, keepdim=True)
    legal = legal | none_legal
    masked = logits.masked_fill(~legal, ILLEGAL_LOGIT)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_probs = F.log_softmax(masked, dim=-1)
    probs = log_probs.exp()
    return masked, log_probs, probs, legal


def entropy_from_probs(probs: torch.Tensor, log_probs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the masked distribution. Illegal mass is zero, not NaN."""

    safe_log = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    return -(probs * safe_log).sum(dim=-1)


@torch.no_grad()
def policy_act(
    model: nn.Module,
    batch: TensorBatch,
    *,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (or argmax) a legal action from any encoder that returns logits and value.

    Returns actions, log-probs, values, and per-row entropy. All detached.
    Both the Transformer and the AlphaNL conv control net use this so the
    action distribution, the mask, and the PPO storage stay the same.
    """

    was_training = model.training
    model.eval()
    logits, value = model(batch, mask_illegal=True)
    _, log_probs, probs, legal = masked_log_probs(logits, batch["legal_mask"])
    if deterministic:
        actions = torch.argmax(logits, dim=-1)
    else:
        actions = torch.distributions.Categorical(probs=probs).sample()
    chosen_logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    ent = entropy_from_probs(probs, log_probs, legal)
    if was_training:
        model.train()
    return actions, chosen_logp, value, ent


class PokerTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        num_types: int = NUM_TYPES,
        num_numeric: int = NUM_NUMERIC,
        num_actions: int = NUM_ACTIONS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_actions = num_actions
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(num_types, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.numeric_proj = nn.Linear(num_numeric, d_model)
        layer_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(**layer_kwargs)
        encoder_kwargs = {"num_layers": num_layers}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(encoder_layer, **encoder_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 1)
        self._init_heads()

    def _init_heads(self) -> None:
        """Small policy head so the initial masked policy is nearly uniform."""

        def _linear(layer: nn.Linear, gain: float) -> None:
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

        _linear(self.numeric_proj, 1.0)
        _linear(self.trunk[0], 2.0 ** 0.5)
        _linear(self.policy_head, 0.01)
        _linear(self.value_head, 1.0)

    def encode_tokens(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        numeric: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        hidden = (
            self.token_emb(token_ids)
            + self.type_emb(type_ids)
            + self.pos_emb(positions)
            + self.numeric_proj(numeric)
        )
        hidden = self.dropout(hidden)
        encoded = self.encoder(hidden, src_key_padding_mask=pad_mask)
        return encoded[:, 0]

    def forward(
        self,
        batch: TensorBatch,
        *,
        mask_illegal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls = self.encode_tokens(
            batch["token_ids"], batch["type_ids"], batch["numeric"], batch["pad_mask"]
        )
        features = self.trunk(cls)
        # Policy/value heads in fp32 even under AMP (encoder stays in autocast dtype).
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            logits = self.policy_head(features)
            value = self.value_head(features).squeeze(-1)
        if mask_illegal:
            logits, _, _, _ = masked_log_probs(logits, batch["legal_mask"])
        return logits, value

    def act(
        self,
        batch: TensorBatch,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return policy_act(self, batch, deterministic=deterministic)


# Heads are not part of the shared trunk. ``conv_fuse`` / ``value_out`` are
# the AlphaNL conv net's policy and value heads; the transformer uses
# ``policy_head`` / ``value_head``. One tuple, so the probe cannot grow a
# second filter.
_HEAD_PARAM_PREFIXES = (
    "policy_head.",
    "value_head.",
    "conv_fuse.",
    "value_out.",
)


def shared_trunk_parameters(model: nn.Module) -> list:
    """Parameters of the shared trunk. This is the only definition of that set.

    Gradient-norm logging differentiates with respect to this list only.

    ``PokerTransformer`` trunk (everything except the two heads):

    * embeddings ``token_emb``, ``type_emb``, ``pos_emb``
    * numeric embedding projection ``numeric_proj``
    * transformer layers ``encoder``
    * shared post-encoder MLP ``trunk`` (Linear + GELU). Both heads read it.
      Dropout has no parameters.

    Excluded: ``policy_head`` and ``value_head``.

    ``AlphaNLConvNet`` uses the same rule: exclude the policy head
    ``conv_fuse`` and the value head ``value_out``. The conv towers and the
    shared 256-128-64 stack stay in the trunk.
    """

    selected = []
    for name, param in model.named_parameters():
        if name.startswith(_HEAD_PARAM_PREFIXES):
            continue
        selected.append(param)
    return selected


"""Frozen distilled policy, bounded zero-init residual and explicit KL anchor."""
import copy
import math
import torch
from torch import nn
from alphanl.cfr.distillation import load_student
from alphanl.transformer.model import masked_log_probs, policy_act
from alphanl.transformer.ppo import ppo_losses
from .allin_mask import SETTINGS_KEY, mask_config
from .context import CONTEXT_FIELDS, PRO_CONTEXT_FIELDS

class ResidualExploiterPolicy(nn.Module):
    def __init__(self, reference, *, alpha=1., hidden=64, beta_kl=1., late_beta_kl=.1,
                 context_fields=CONTEXT_FIELDS, botzone_native_allin_mask=None):
        super().__init__()
        if any(not math.isfinite(v) or v < 0 for v in (alpha, beta_kl, late_beta_kl)) or hidden < 1:
            raise ValueError("invalid residual hyperparameters")
        self.reference = copy.deepcopy(reference).requires_grad_(False).eval()
        self.config = reference.config
        self.settings = dict(alpha=alpha, hidden=hidden, beta_kl=beta_kl, late_beta_kl=late_beta_kl)
        self.context_fields = tuple(context_fields)
        if self.context_fields not in (CONTEXT_FIELDS, PRO_CONTEXT_FIELDS):
            raise ValueError("unsupported policy context fields")
        if self.context_fields != CONTEXT_FIELDS:
            self.settings["context_fields"] = list(self.context_fields)
        # Additive and absent on old checkpoints. The observation path turns it
        # into a legal-mask bit; forward still masks that bit before softmax.
        if botzone_native_allin_mask is not None:
            self.settings[SETTINGS_KEY] = mask_config(botzone_native_allin_mask)
        self.context_encoder = nn.Sequential(nn.Linear(len(self.context_fields), hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(reference.backbone.d_model + hidden, hidden), nn.GELU())
        self.delta_head = nn.Linear(hidden, 5)
        self.value_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.to(next(reference.parameters()).device)

    @classmethod
    def from_reference(cls, path, **kwargs):
        reference, _ = load_student(path)
        return cls(reference, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    @torch.no_grad()
    def prepare(self, batch):
        backbone = self.reference.backbone
        features = backbone.trunk(backbone.encode_tokens(
            batch["token_ids"], batch["type_ids"], batch["numeric"], batch["pad_mask"])).float()
        ref = backbone.policy_head(features)
        return {**batch, "poker_features": features.detach(), "ref_logits": ref.detach()}

    def prepared(self, batch):
        return batch if "poker_features" in batch and "ref_logits" in batch else self.prepare(batch)

    def forward(self, batch, *, mask_illegal=True):
        batch = self.prepared(batch)
        if not (batch["legal_mask"] > .5).any(-1).all():
            raise ValueError("empty legal mask")
        context = batch["match_ctx"]
        if context.ndim != 2 or context.shape[1] != len(self.context_fields) or not torch.isfinite(context).all():
            raise ValueError("invalid match_ctx")
        fused = self.fusion(torch.cat((batch["poker_features"].detach(), self.context_encoder(context)), -1))
        logits = batch["ref_logits"].detach() + self.settings["alpha"] * self.delta_head(fused).tanh()
        if mask_illegal:
            # legal_mask already includes the optional native all-in mask.
            # Illegal logits become ILLEGAL_LOGIT (-1e9) before this softmax.
            logits = masked_log_probs(logits, batch["legal_mask"])[0]
        return logits, self.value_head(fused).squeeze(-1)

    def act(self, batch, *, deterministic=False):
        return policy_act(self, batch, deterministic=deterministic)

    def kl_penalty(self, batch):
        batch = self.prepared(batch)
        logits, _ = self(batch)
        _, logp, probs, legal = masked_log_probs(logits, batch["legal_mask"])
        _, ref_logp, _, _ = masked_log_probs(batch["ref_logits"].detach(), batch["legal_mask"])
        kl = (probs * (logp - ref_logp).masked_fill(~legal, 0.)).sum(-1).clamp_min(0.)
        # Strong anchor for first ten hands, then anneal; no claim of automatic strength detection.
        progress = ((batch["match_ctx"][:, 0] * 50 - 10) / 40).clamp(0, 1)
        beta = self.settings["beta_kl"] + progress * (self.settings["late_beta_kl"] - self.settings["beta_kl"])
        return (beta * kl).mean(), kl, beta


def residual_ppo_loss(model, batch, **kwargs):
    batch = model.prepared(batch)
    loss, metrics, *_ = ppo_losses(model, batch, **kwargs)
    penalty, kl, beta = model.kl_penalty(batch)
    total = loss + penalty
    metrics.update(loss=float(total.detach()), base_kl=float(kl.mean().detach()),
                   kl_penalty=float(penalty.detach()), beta_kl=float(beta.mean()))
    return total, metrics


def safe_all_in(probs, snapshot, *, enabled=False, max_spr=3., probability_threshold=.05):
    """Optional inference intervention; downgrade rare deep shoves to legal call."""
    from alphanl.cfr.actions import CFRActionGuard
    result = probs.clone()
    if (enabled and 4 in snapshot.legal_action_ids and 1 in snapshot.legal_action_ids
            and CFRActionGuard.all_in_pot_ratio(snapshot) > max_spr
            and 0 < float(result[4]) < probability_threshold):
        result[1] += result[4]
        result[4] = 0.
    return result

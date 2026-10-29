"""Convex blending of bounded residual outputs on an identical frozen base."""
import math
import torch
from torch import nn
from alphanl.transformer.model import masked_log_probs


class ResidualOutputBlend(nn.Module):
    def __init__(self, members, weights):
        super().__init__()
        if (len(members) < 2 or len(members) != len(weights)
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.)):
            raise ValueError("blend requires normalized nonnegative weights")
        first = members[0]
        for other in members[1:]:
            if first.config.to_dict() != other.config.to_dict() or first.settings != other.settings:
                raise ValueError("blend policy contracts differ")
            for key, value in first.reference.state_dict().items():
                if not torch.equal(value.cpu(), other.reference.state_dict()[key].cpu()):
                    raise ValueError("blend references differ")
        self.members = nn.ModuleList(members)
        self.weights = tuple(weights)
        self.config = first.config
        self.context_fields = getattr(first, "context_fields", None)
        self.requires_grad_(False).eval()

    @property
    def reference(self):
        return self.members[0].reference

    def forward(self, batch, *, mask_illegal=True):
        # One shared reference trunk; blend after tanh, before the final softmax.
        prepared = self.members[0].prepared(batch)
        outputs = [model(prepared, mask_illegal=False) for model in self.members]
        logits = sum(w * output[0] for w, output in zip(self.weights, outputs))
        values = sum(w * output[1] for w, output in zip(self.weights, outputs))
        if mask_illegal:
            logits = masked_log_probs(logits, batch["legal_mask"])[0]
        return logits, values

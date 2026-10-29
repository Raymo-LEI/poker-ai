"""Match-local rule parameters and shared, frozen neural opponent inference."""
from collections import defaultdict
import numpy as np
import torch
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.evaluation import RuleActor, random_equity
from alphanl.cfr.model import regret_matching
from alphanl.transformer.hunl import HunlTokenizer
from alphanl.transformer.tokenizer import cat_batches
from .context import MatchCtx


class JitteredRuleActor(RuleActor):
    """Parameters are drawn once, never exposed in the policy observation."""
    def __init__(self, name, *, seed, equity_samples=64, variant=None):
        super().__init__("maniac" if name == "push_fold" else name, seed=seed, equity_samples=equity_samples)
        self.name = name
        self.variant = variant or (str(self.rng.choice(["standard", "tight", "aggressive"])) if name == "equity" else name)
        self.params = {
            "margin": float(self.rng.uniform(.03, .08)) if self.variant == "tight" else 0.,
            "nit_tightness": float(self.rng.uniform(-.02, .02)),
            "value_threshold": float(self.rng.uniform(.82, .90)),
            "aggression": float(self.rng.uniform(.88, .98)),
            "push_frequency": float(self.rng.uniform(.80, .98)),
            "large_threshold": float(self.rng.uniform(.65, .70)),
        }
        if self.variant == "standard":
            self.rng = np.random.default_rng(seed)

    def act(self, snapshot):
        # This branch is literally the unchanged evaluation baseline.
        if self.name == "equity" and self.variant == "standard":
            return super().act(snapshot)
        legal = snapshot.legal_action_ids
        def prefer(*choices):
            return next(a for a in choices if a in legal)
        if self.name == "push_fold":
            if self.rng.random() < self.params["push_frequency"]:
                return prefer(4, 3, 2, 1, 0)
            return prefer(0, 1, 4, 3, 2) if snapshot.to_call else prefer(1, 0, 4, 3, 2)
        if self.name in ("maniac", "overbet"):
            if self.rng.random() < self.params["aggression"]:
                return prefer(*( (4, 3, 2, 1, 0) if self.name == "maniac" else (3, 4, 2, 1, 0)))
            return prefer(1, 0, 4, 3, 2)
        equity = random_equity(snapshot, self.rng, self.equity_samples)
        odds = snapshot.to_call / max(1., snapshot.pot + snapshot.to_call)
        if self.name == "call":
            # Still a calling station; only rare strong value raises are jittered.
            if equity > self.params["value_threshold"] and self.rng.random() < .2:
                return prefer(2, 3, 1, 4, 0)
            return prefer(1, 0, 4, 3, 2)
        if self.name == "nit":
            jitter = self.params["nit_tightness"]
            if snapshot.to_call > 0 and equity < max(.62 + jitter, odds + .12 + jitter):
                return prefer(0, 1, 4, 3, 2)
            return prefer(1, 0, 4, 3, 2)
        margin = self.params["margin"]
        if snapshot.to_call > 0 and equity < odds + .03 + margin:
            return prefer(0, 1, 4, 3, 2)
        if self.variant == "aggressive" and equity > self.params["large_threshold"]:
            # The shared five-action tree supports pot and all-in; no invented size.
            return prefer(4, 3, 2, 1, 0) if equity > .90 else prefer(3, 2, 4, 1, 0)
        if equity > .8 + margin:
            return prefer(3, 2, 4, 1, 0)
        if equity > .65 + margin:
            return prefer(2, 3, 1, 4, 0)
        return prefer(1, 0, 2, 3, 4)


class NeuralMatchActor:
    """Independent public HUD/RNG per match, shared immutable model per id."""
    def __init__(self, model, *, kind, snapshot_id, seed):
        self.model, self.kind, self.snapshot_id = model, kind, snapshot_id
        self.rng = np.random.default_rng(seed)
        self.ctx = MatchCtx()
        self.config = model.config
        self.tokenizer = HunlTokenizer(self.config.model.max_seq_len, **self.config.tokenizer_kwargs())
        self.guard = CFRActionGuard.from_config(self.config)

    def begin_match(self):
        self.ctx = MatchCtx()

    def begin_hand(self, seat):
        self.ctx.begin_hand()

    def observe_action(self, event):
        self.ctx.observe_action(event)

    def finish_hand(self, net_bb, *, showdown):
        self.ctx.finish_hand(net_bb, showdown=showdown)

    def encode(self, snapshot):
        snap = self.guard.apply(snapshot)
        # Self-play copies use the same voluntary all-in mask as the learner.
        if self.kind == "residual":
            from .allin_mask import mask_snapshot
            snap = mask_snapshot(snap, self.model.settings.get("botzone_native_allin_mask"))
        batch = self.tokenizer.encode(snap)
        if self.kind == "residual":
            batch["match_ctx"] = self.ctx.tensor(self.model.context_fields)
        return batch

    def act(self, snapshot):
        return neural_actions([(self, snapshot)])[0]


@torch.inference_mode()
def neural_actions(items):
    """Same grouping pattern as transformer.env.PolicyActor.prefetch."""
    groups = defaultdict(list)
    result = [None] * len(items)
    for i, (actor, snapshot) in enumerate(items):
        groups[id(actor.model)].append((i, actor, actor.encode(snapshot)))
    for group in groups.values():
        model, kind = group[0][1].model, group[0][1].kind
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in cat_batches([b for _, _, b in group]).items()}
        if kind == "cfr":
            probs = regret_matching(model(batch), batch["legal_mask"], fallback=model.config.fallback)
        elif kind == "residual":
            probs = model(batch)[0].softmax(-1)
        else:
            probs = model(batch).softmax(-1)
        rows = probs.cpu().double().numpy()
        if not np.isfinite(rows).all():
            raise FloatingPointError("nonfinite opponent probabilities")
        for (index, actor, _), row in zip(group, rows):
            result[index] = int(actor.rng.choice(5, p=row / row.sum()))
    return result

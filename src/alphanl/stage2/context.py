"""Public, opportunity-based Bayesian HUD. No opponent cards enter this state."""
from dataclasses import dataclass, field, asdict
import math

CONTEXT_FIELDS = (
    "hand_progress", "bankroll_diff", "vpip", "pfr", "three_bet",
    "fold_to_preflop_raise", "flop_aggression", "turn_aggression",
    "river_aggression", "fold_to_postflop_bet", "all_in_frequency", "showdown_win",
)
# Beta priors: mean and effective observation count, explicitly serialized.
PRIORS = {
    "vpip": (.45, 6.), "pfr": (.25, 6.), "three_bet": (.10, 10.),
    "fold_to_preflop_raise": (.5, 6.), "flop_aggression": (.35, 6.),
    "turn_aggression": (.35, 6.), "river_aggression": (.35, 6.),
    "fold_to_postflop_bet": (.5, 6.), "all_in_frequency": (.05, 10.),
    "showdown_win": (.5, 6.),
}

@dataclass
class MatchCtx:
    hands_played: int = 0
    hero_net_bb: float = 0.
    successes: dict = field(default_factory=lambda: dict.fromkeys(PRIORS, 0.))
    opportunities: dict = field(default_factory=lambda: dict.fromkeys(PRIORS, 0))
    active: bool = False
    vpip: bool = False
    pfr: bool = False
    preflop_raises: int = 0
    three_bet_seen: bool = False
    preflop_fold_seen: bool = False

    def begin_hand(self):
        if self.active or self.hands_played >= 50:
            raise ValueError("finish hand or reset MatchCtx at match boundary")
        self.active = True
        self.vpip = self.pfr = self.three_bet_seen = self.preflop_fold_seen = False
        self.preflop_raises = 0

    def _record(self, name, success):
        self.opportunities[name] += 1
        self.successes[name] += float(success)

    def observe_action(self, event):
        if not self.active or event.street not in range(4):
            raise ValueError("action requires an active hand and valid street")
        aggressive = event.action_type in ("bet", "raise", "all_in") and event.chips_added > event.to_call_before
        if not event.actor_is_me:
            if event.street == 0:
                self.vpip |= event.chips_added > 0
                self.pfr |= aggressive
                # First response to exactly one open raise; exclude subsequent 4-bets.
                if self.preflop_raises == 1 and event.to_call_before > 0 and not self.three_bet_seen:
                    self._record("three_bet", aggressive)
                    self.three_bet_seen = True
                if self.preflop_raises > 0 and event.to_call_before > 0 and not self.preflop_fold_seen:
                    self._record("fold_to_preflop_raise", event.action_type == "fold")
                    self.preflop_fold_seen = True
            else:
                # Aggression frequency = bets/raises / all voluntary actions, folds included.
                self._record(CONTEXT_FIELDS[5 + event.street], aggressive)
                if event.to_call_before > 0:
                    self._record("fold_to_postflop_bet", event.action_type == "fold")
            self._record("all_in_frequency", event.chips_added > 0 and event.chips_added == event.stack_before)
        if event.street == 0 and aggressive:
            self.preflop_raises += 1

    def finish_hand(self, hero_net_bb, *, showdown=False):
        if not self.active or not math.isfinite(hero_net_bb):
            raise ValueError("finish requires active hand and finite payoff")
        self._record("vpip", self.vpip)
        self._record("pfr", self.pfr)
        if showdown:
            self._record("showdown_win", .5 if hero_net_bb == 0 else float(hero_net_bb < 0))
        self.hands_played += 1
        self.hero_net_bb += float(hero_net_bb)
        self.active = False

    def as_dict(self):
        # Score is hero minus opponent bankroll, i.e. twice zero-sum hero net.
        result = {"hand_progress": self.hands_played / 50., "bankroll_diff": 2 * self.hero_net_bb / 200.}
        for name, (mean, count) in PRIORS.items():
            successes, opportunities = self.successes[name], self.opportunities[name]
            if self.active and name in ("vpip", "pfr"):
                successes += float(getattr(self, name))
                opportunities += 1
            result[name] = (successes + mean * count) / (opportunities + count)
        return result

    def tensor(self):
        import torch
        return torch.tensor([list(self.as_dict().values())], dtype=torch.float32)

    def state_dict(self):
        return {"format": "match_ctx_v1", **asdict(self)}

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        if state.pop("format", None) != "match_ctx_v1":
            raise ValueError("incompatible match context")
        value = cls(**state)
        if (not 0 <= value.hands_played <= 50 or not math.isfinite(value.hero_net_bb)
                or set(value.successes) != set(PRIORS) or set(value.opportunities) != set(PRIORS)
                or any(not math.isfinite(value.successes[k]) or not 0 <= value.successes[k] <= value.opportunities[k] for k in PRIORS)):
            raise ValueError("invalid match context counters")
        return value

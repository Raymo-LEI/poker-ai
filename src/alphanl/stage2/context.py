"""Public, opportunity-based Bayesian HUD. No opponent cards enter this state."""
from dataclasses import dataclass, field, asdict
import math
from typing import Optional

CONTEXT_FIELDS = (
    "hand_progress", "bankroll_diff", "vpip", "pfr", "three_bet",
    "fold_to_preflop_raise", "flop_aggression", "turn_aggression",
    "river_aggression", "fold_to_postflop_bet", "all_in_frequency", "showdown_win",
)
PRO_CONTEXT_FIELDS = CONTEXT_FIELDS + ("check_raise_freq", "cbet_freq")
TRAP_PRIORS = {"check_raise_freq": (.15, 2.), "cbet_freq": (.55, 2.)}
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
    trap_successes: dict = field(default_factory=lambda: dict.fromkeys(TRAP_PRIORS, 0.))
    trap_opportunities: dict = field(default_factory=lambda: dict.fromkeys(TRAP_PRIORS, 0))
    opponent_checks: set = field(default_factory=set)
    check_raise_seen: set = field(default_factory=set)
    postflop_bets: set = field(default_factory=set)
    last_preflop_aggressor_is_me: Optional[bool] = None
    flop_opponent_acted: bool = False

    def begin_hand(self):
        if self.active or self.hands_played >= 50:
            raise ValueError("finish hand or reset MatchCtx at match boundary")
        self.active = True
        self.vpip = self.pfr = self.three_bet_seen = self.preflop_fold_seen = False
        self.preflop_raises = 0
        self.opponent_checks, self.check_raise_seen, self.postflop_bets = set(), set(), set()
        self.last_preflop_aggressor_is_me = None
        self.flop_opponent_acted = False

    def _record(self, name, success):
        self.opportunities[name] += 1
        self.successes[name] += float(success)

    def observe_action(self, event):
        if not self.active or event.street not in range(4):
            raise ValueError("action requires an active hand and valid street")
        aggressive = event.action_type in ("bet", "raise", "all_in") and event.chips_added > event.to_call_before
        if event.street > 0 and not event.actor_is_me:
            can_raise = event.stack_before > event.to_call_before and (
                not event.legal_action_ids or any(a in event.legal_action_ids for a in (2,3,4)))
            if event.street in self.opponent_checks and event.to_call_before > 0 and event.street not in self.check_raise_seen and can_raise:
                self.trap_opportunities["check_raise_freq"] += 1
                self.trap_successes["check_raise_freq"] += float(aggressive)
                self.check_raise_seen.add(event.street)
            if event.street == 1 and not self.flop_opponent_acted:
                if self.last_preflop_aggressor_is_me is False and 1 not in self.postflop_bets and event.to_call_before == 0:
                    self.trap_opportunities["cbet_freq"] += 1
                    self.trap_successes["cbet_freq"] += float(aggressive)
                self.flop_opponent_acted = True
            if event.action_type == "check":
                self.opponent_checks.add(event.street)
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
            self.last_preflop_aggressor_is_me = bool(event.actor_is_me)
        if event.street > 0 and aggressive:
            self.postflop_bets.add(event.street)

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

    def as_dict(self, fields=CONTEXT_FIELDS):
        # Score is hero minus opponent bankroll, i.e. twice zero-sum hero net.
        result = {"hand_progress": self.hands_played / 50., "bankroll_diff": 2 * self.hero_net_bb / 200.}
        for name, (mean, count) in PRIORS.items():
            successes, opportunities = self.successes[name], self.opportunities[name]
            if self.active and name in ("vpip", "pfr"):
                successes += float(getattr(self, name))
                opportunities += 1
            result[name] = (successes + mean * count) / (opportunities + count)
        fields = tuple(fields or CONTEXT_FIELDS)
        if fields not in (CONTEXT_FIELDS, PRO_CONTEXT_FIELDS):
            raise ValueError("unsupported match context fields")
        if fields == PRO_CONTEXT_FIELDS:
            for name, (mean, count) in TRAP_PRIORS.items():
                result[name] = (self.trap_successes[name] + mean * count) / (self.trap_opportunities[name] + count)
        return result

    def tensor(self, fields=CONTEXT_FIELDS):
        import torch
        return torch.tensor([list(self.as_dict(fields).values())], dtype=torch.float32)

    def state_dict(self):
        state = asdict(self)
        for name in ("opponent_checks", "check_raise_seen", "postflop_bets"):
            state[name] = sorted(state[name])
        return {"format": "match_ctx_v1", **state}

    @classmethod
    def from_state_dict(cls, state):
        state = dict(state)
        if state.pop("format", None) != "match_ctx_v1":
            raise ValueError("incompatible match context")
        for name in ("opponent_checks", "check_raise_seen", "postflop_bets"):
            if name in state:
                state[name] = set(state[name])
        value = cls(**state)
        if (not 0 <= value.hands_played <= 50 or not math.isfinite(value.hero_net_bb)
                or set(value.successes) != set(PRIORS) or set(value.opportunities) != set(PRIORS)
                or any(not math.isfinite(value.successes[k]) or not 0 <= value.successes[k] <= value.opportunities[k] for k in PRIORS)):
            raise ValueError("invalid match context counters")
        if (set(value.trap_successes) != set(TRAP_PRIORS) or set(value.trap_opportunities) != set(TRAP_PRIORS)
                or any(not math.isfinite(value.trap_successes[k]) or not 0 <= value.trap_successes[k] <= value.trap_opportunities[k] for k in TRAP_PRIORS)):
            raise ValueError("invalid trap context counters")
        return value

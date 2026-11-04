"""Public, opportunity-based Bayesian HUD. No opponent cards enter this state."""
from dataclasses import dataclass, field, asdict
import math
from typing import Optional
from .shove_threat import shove_threat

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
# Counters from a hand with an inconsistent/unknown settlement are rolled back.
PROFILE_STATE_FIELDS = (
    "successes", "opportunities", "trap_successes", "trap_opportunities",
    "opp_shove_count", "opp_effective_shove_count", "opp_preflop_effective_shove_count",
    "opp_preflop_action_opportunities", "opp_postflop_bet_responses", "opp_postflop_bet_folds",
    "opp_regular_postflop_actions", "opp_regular_postflop_bets", "opp_postflop_checks",
    "sb_open_count", "sb_open_mean_bb", "sb_open_m2_bb", "postflop_size_count",
    "postflop_small_bets", "postflop_overbets", "reconstructed_folds", "reconstructed_calls",
    "response_by_size",
)

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
    # Deployment-only evidence. These are NOT extra neural HUD input columns.
    opp_shove_count: int = 0
    opp_effective_shove_count: int = 0
    opp_preflop_effective_shove_count: int = 0
    opp_preflop_action_opportunities: int = 0
    opp_postflop_bet_responses: int = 0
    opp_postflop_bet_folds: int = 0
    station_evidence_complete: bool = True
    opp_regular_postflop_actions: int = 0
    opp_regular_postflop_bets: int = 0
    opp_postflop_checks: int = 0
    # Public terminal ledger, carried by Botzone data rather than hand tokens.
    last_hand_idx: int = -1
    my_committed_before_act: float = 0.
    my_committed_after_act: float = 0.
    opp_committed_at_my_act: float = 0.
    my_last_action_was_bet_or_raise: bool = False
    last_street: int = 0
    last_action_type: str = ""
    last_opp_stack: float = 0.
    last_me_stack_after: float = 0.
    terminal_pending: bool = False
    reconstructed_folds: int = 0
    reconstructed_calls: int = 0
    # Welford moments of SB open extra chips, expressed in BB.
    sb_open_count: int = 0
    sb_open_mean_bb: float = 0.
    sb_open_m2_bb: float = 0.
    postflop_size_count: int = 0
    postflop_small_bets: int = 0
    postflop_overbets: int = 0
    last_check_closes: bool = False
    hand_fold_seen: bool = False
    hand_statistics_baseline: dict = field(default_factory=dict)
    unknown_hands: int = 0
    last_terminal_status: str = "none"
    frequency_frozen: bool = False
    frozen_at_hand: int = -1
    freeze_reason: str = ""
    response_by_size: dict = field(default_factory=dict)
    hero_bet_fractions: dict = field(default_factory=dict)

    def begin_hand(self):
        if self.active or self.hands_played >= 50:
            raise ValueError("finish hand or reset MatchCtx at match boundary")
        self.active = True
        self.vpip = self.pfr = self.three_bet_seen = self.preflop_fold_seen = False
        self.preflop_raises = 0
        self.opponent_checks, self.check_raise_seen, self.postflop_bets = set(), set(), set()
        self.last_preflop_aggressor_is_me = None
        self.flop_opponent_acted = False
        self.terminal_pending = False
        self.hand_fold_seen = False
        self.hero_bet_fractions = {}
        self.hand_statistics_baseline = {k: {n: list(v) if isinstance(v, list) else v for n, v in getattr(self, k).items()} if isinstance(getattr(self, k), dict)
                                         else getattr(self, k) for k in PROFILE_STATE_FIELDS}

    def _record(self, name, success):
        if self.frequency_frozen:
            return
        self.opportunities[name] += 1
        self.successes[name] += float(success)

    def observe_action(self, event):
        if not self.active or event.street not in range(4):
            raise ValueError("action requires an active hand and valid street")
        aggressive = event.action_type in ("bet", "raise", "all_in") and event.chips_added > event.to_call_before
        # Any subsequently observed action supersedes the pending snapshot.
        self.terminal_pending = False
        self.hand_fold_seen |= event.action_type == "fold"
        if self.frequency_frozen:
            return
        if event.actor_is_me and aggressive:
            self.hero_bet_fractions[str(event.street)] = float(event.pot_fraction)
        if not event.actor_is_me and event.to_call_before > 0:
            fraction = self.hero_bet_fractions.get(str(event.street))
            if fraction is not None:
                bucket = "small" if fraction <= .4 else "overbet" if fraction > 1. else "regular"
                key = "%d:%s" % (event.street, bucket)
                counts = self.response_by_size.setdefault(key, [0, 0, 0])
                counts[0] += 1
                counts[1] += int(event.action_type == "fold")
                counts[2] += int(aggressive)
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
            if aggressive and event.street == 0 and event.position == "SB" and self.preflop_raises == 0:
                self.sb_open_count += 1
                delta = event.chips_added / 100. - self.sb_open_mean_bb
                self.sb_open_mean_bb += delta / self.sb_open_count
                self.sb_open_m2_bb += delta * (event.chips_added / 100. - self.sb_open_mean_bb)
            if aggressive and event.street > 0:
                self.postflop_size_count += 1
                self.postflop_small_bets += int(event.pot_fraction <= .4)
                self.postflop_overbets += int(event.pot_fraction > 1.)
            all_in = event.chips_added > 0 and event.chips_added == event.stack_before
            if (all_in and aggressive and event.chips_added > 5000.
                    and event.chips_added > event.pot_before):
                self.opp_shove_count += 1
            if shove_threat(event, allow_near=True):
                self.opp_effective_shove_count += 1
                if event.street == 0:
                    self.opp_preflop_effective_shove_count += 1
            if event.street == 0:
                self.opp_preflop_action_opportunities += 1
            if event.street > 0 and not all_in:
                self.opp_regular_postflop_actions += 1
                self.opp_regular_postflop_bets += int(aggressive)
                self.opp_postflop_checks += int(event.action_type == "check")
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
                if event.to_call_before > 0:
                    self.opp_postflop_bet_responses += 1
                    self.opp_postflop_bet_folds += int(event.action_type == "fold")
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

    def remember_action(self, hand_idx, snapshot, chips, kind):
        self.last_hand_idx = int(hand_idx)
        self.my_committed_before_act = snapshot.starting_stack - snapshot.me_stack
        self.my_committed_after_act = self.my_committed_before_act + chips
        self.opp_committed_at_my_act = snapshot.starting_stack - snapshot.opp_stack
        self.my_last_action_was_bet_or_raise = chips > snapshot.to_call and kind in ("bet", "raise", "all_in")
        self.last_street, self.last_action_type = snapshot.street, kind
        self.last_opp_stack = snapshot.opp_stack
        self.last_me_stack_after = snapshot.me_stack - chips
        self.last_check_closes = (kind == "check" and snapshot.street == 3
            and snapshot.me_seat != getattr(snapshot, 'postflop_first_seat', snapshot.sb_seat))
        self.terminal_pending = True

    def terminal_response(self, hand_idx, net_chips):
        """Return a provable missing (kind, chips), or None; consume once.

        The caller must validate adjacent hands, seats and the zero-sum score
        ledger first. A call is proved only when matching ends the hand and
        its effective contribution agrees exactly, including short-call refunds.
        A river CHECK can still be followed by a judge-legal free Fold, so a
        zero payoff alone is deliberately insufficient evidence there.
        """
        pending = self.terminal_pending and self.last_hand_idx == hand_idx
        self.terminal_pending = False
        if not pending or not self.my_last_action_was_bet_or_raise:
            return None
        before = self.opp_committed_at_my_act
        owed = self.my_committed_after_act - before
        if owed <= 0 or self.last_opp_stack <= 0:
            return None
        if net_chips == before and net_chips > 0:
            self.reconstructed_folds += 1
            return "fold", 0.
        call = min(owed, self.last_opp_stack)
        matched = before + call
        ends = self.last_street == 3 or self.last_me_stack_after == 0 or call == self.last_opp_stack
        if ends and (net_chips in (-matched, matched) or abs(net_chips) <= 1):
            self.reconstructed_calls += 1
            return "call", call
        return None

    @property
    def sb_open_variance_bb(self):
        return self.sb_open_m2_bb / self.sb_open_count if self.sb_open_count else 0.

    def freeze_frequencies(self, hand_idx, reason):
        if not self.frequency_frozen:
            self.frequency_frozen = True
            self.frozen_at_hand = int(hand_idx)
            self.freeze_reason = reason

    def finish_hand(self, hero_net_bb, *, showdown=False, split=False, statistics_known=True):
        if not self.active or not math.isfinite(hero_net_bb):
            raise ValueError("finish requires active hand and finite payoff")
        if not statistics_known:
            for key, value in self.hand_statistics_baseline.items():
                setattr(self, key, dict(value) if isinstance(value, dict) else value)
            self.unknown_hands += 1
            self.last_terminal_status = "unknown"
        else:
            self._record("vpip", self.vpip)
            self._record("pfr", self.pfr)
            if showdown:
                self._record("showdown_win", .5 if split or hero_net_bb == 0 else float(hero_net_bb < 0))
        self.hands_played += 1
        self.hero_net_bb += float(hero_net_bb)
        self.active = False

    def as_dict(self, fields=CONTEXT_FIELDS):
        # Score is hero minus opponent bankroll, i.e. twice zero-sum hero net.
        result = {"hand_progress": self.hands_played / 50., "bankroll_diff": 2 * self.hero_net_bb / 200.}
        for name, (mean, count) in PRIORS.items():
            successes, opportunities = self.successes[name], self.opportunities[name]
            if self.active and not self.frequency_frozen and name in ("vpip", "pfr"):
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
        counters = (value.opp_shove_count, value.opp_effective_shove_count, value.opp_regular_postflop_actions,
                    value.opp_regular_postflop_bets, value.opp_postflop_checks)
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counters)
                or value.opp_regular_postflop_bets + value.opp_postflop_checks > value.opp_regular_postflop_actions):
            raise ValueError("invalid deployment context counters")
        counts = (value.sb_open_count, value.postflop_size_count, value.postflop_small_bets,
                  value.postflop_overbets, value.reconstructed_folds, value.reconstructed_calls, value.unknown_hands)
        finite = (value.my_committed_before_act, value.my_committed_after_act, value.opp_committed_at_my_act,
                  value.last_opp_stack, value.last_me_stack_after, value.sb_open_mean_bb, value.sb_open_m2_bb)
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts)
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in finite)
                or value.postflop_small_bets + value.postflop_overbets > value.postflop_size_count
                or value.last_street not in range(4) or not -1 <= value.last_hand_idx < 50
                or value.my_committed_after_act < value.my_committed_before_act):
            raise ValueError("invalid terminal ledger or size counters")
        for key, counts in value.response_by_size.items():
            if (key not in {"%d:%s" % (s, b) for s in range(4) for b in ("small", "regular", "overbet")}
                    or len(counts) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts)
                    or counts[1] + counts[2] > counts[0]):
                raise ValueError("invalid size response counters")
        return value

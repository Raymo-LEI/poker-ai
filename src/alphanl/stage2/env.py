"""50-hand match schedule on the *same* standard HUNL rules as Stage 1.

The name describes the match schedule. Legacy TexasHoldem2p judge rules
have different action order/min-raises; deployment uses an explicit adapter.
"""
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.traversal import HunlState
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from .context import MatchCtx, CONTEXT_FIELDS

class BotzoneMatchEnv:
    def __init__(self, config, opponent=None, *, pool=None, seed=1, first_hero_seat=1,
                 defer_opponents=False, decks=None, hand_trace=None, context_fields=CONTEXT_FIELDS):
        if config.separate_player_models or not config.position_conditioning:
            raise ValueError("Stage 2 requires unified position-conditioned Stage 1")
        if first_hero_seat not in (0, 1):
            raise ValueError("invalid first hero seat")
        self.hand_trace = hand_trace
        self.context_fields = tuple(context_fields)
        self.config, self.opponent = config, opponent
        if (pool is None) == (opponent is None):
            raise ValueError("provide exactly one fixed opponent or MatchPool")
        self.pool = pool
        self.opponent_name = (getattr(opponent, "name", None) or
                              {"cfr": "history"}.get(getattr(opponent, "kind", None),
                                  getattr(opponent, "kind", None)))
        self.defer_opponents, self.decks = defer_opponents, decks
        if decks is not None and len(decks) != 50:
            raise ValueError("a duplicate match needs exactly 50 decks")
        self.first_hero_seat = first_hero_seat
        self.table = HunlTable(seed=seed)
        self.tokenizer = HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs())
        self.guard = CFRActionGuard.from_config(config)
        self.ctx = MatchCtx()
        self.done = True
        self.hands = []

    def observe_poker(self):
        if self.table.is_over() or self.table.current_player() != self.hero:
            raise RuntimeError("observation requires hero decision")
        # Exact Stage 1 traversal boundary: tokenizer, numeric features, action mask.
        return HunlState(self.table, self.tokenizer, self.guard).observation()

    def observe(self):
        return {**self.observe_poker(), "match_ctx": self.ctx.tensor(self.context_fields)}

    def reset(self):
        if not self.done:
            raise RuntimeError("cannot reset an unfinished match")
        self.ctx, self.hands, self.done = MatchCtx(), [], False
        if self.pool is not None:
            self.opponent_name, self.opponent = self.pool.sample()
        if hasattr(self.opponent, "begin_match"):
            self.opponent.begin_match()
        self._deal()
        reward = self._advance()
        # A match may contain passive hands before the first decision.
        self.initial_reward = reward
        return self._next_observation()

    def _next_observation(self):
        if self.done:
            return self._blank()
        return None if self.pending_opponent() else self.observe()

    def pending_opponent(self):
        return not self.done and self.table.current_player() != self.hero

    def opponent_step(self, action):
        if not self.defer_opponents or not self.pending_opponent():
            raise RuntimeError("no deferred opponent decision")
        if action not in self.table.legal_action_ids():
            raise ValueError("illegal opponent action")
        self._push(int(action))
        return self._advance()

    def _deal(self):
        self.hero = self.first_hero_seat ^ (self.ctx.hands_played % 2)
        self.table.reset(None if self.decks is None else self.decks[self.ctx.hands_played])
        if self.hand_trace is not None:
            self._trace_events = []
            self._trace_hole = list(self.table.hole_cards(self.hero))
        self.ctx.begin_hand()
        self.opponent.begin_hand(1 - self.hero)

    def _push(self, action):
        if self.hand_trace is not None:
            seat = self.table.current_player()
            snapshot = self.table.snapshot(seat)
        self.table.step(action)
        if self.hand_trace is not None:
            event = self.table.events_for(self.hero)[-1]
            self._trace_events.append(dict(seat=seat, street=snapshot.street, action=action,
                board=list(snapshot.board_cards), pot=snapshot.pot, to_call=snapshot.to_call,
                chips_added=event.chips_added, stack_before=event.stack_before))
        self.ctx.observe_action(self.table.events_for(self.hero)[-1])
        if hasattr(self.opponent, "observe_action"):
            self.opponent.observe_action(self.table.events_for(1 - self.hero)[-1])

    def _advance(self):
        reward = 0.
        for _ in range(50000):
            if self.table.is_over():
                payoff = float(self.table.payoffs()[self.hero]) / 100.
                if sum(self.table.payoffs()) != 0 or self.table.chips_in_play() != 40000:
                    raise RuntimeError("chip conservation failed")
                showdown = not any(e.action_type == "fold" for e in self.table.events_for(self.hero))
                self.ctx.finish_hand(payoff, showdown=showdown)
                if hasattr(self.opponent, "finish_hand"):
                    self.opponent.finish_hand(-payoff, showdown=showdown)
                reward += payoff / 200.
                self.hands.append({"hand": self.ctx.hands_played, "hero_seat": self.hero, "hero_net_bb": payoff})
                if self.hand_trace is not None:
                    from .traces import category
                    self.hand_trace(dict(hand=self.ctx.hands_played, hero_seat=self.hero,
                        hero_hole=self._trace_hole, payoff_chips=payoff * 100,
                        terminal_board=list(self.table.board_cards()),
                        category=category(self._trace_events, self.hero),
                        terminal_betting_street=self._trace_events[-1]['street'],
                        events=self._trace_events))
                if self.ctx.hands_played == 50:
                    self.done = True
                    return reward
                self._deal()
            if self.table.current_player() == self.hero:
                return reward
            if self.defer_opponents:
                return reward
            # Rule bots retain their full engine-legal actions as in Stage 1 evaluation.
            action = int(self.opponent.act(self.table.snapshot(1 - self.hero)))
            if action not in self.table.legal_action_ids():
                raise ValueError("opponent selected an illegal action")
            self._push(action)
        raise RuntimeError("match action budget exceeded")

    def step(self, action):
        if self.done:
            raise RuntimeError("match has ended")
        if self.pending_opponent():
            raise RuntimeError("wait for the deferred opponent decision")
        if action not in self.guard.apply(self.table.snapshot(self.hero)).legal_action_ids:
            raise ValueError("hero action violates Stage 1 abstraction")
        self._push(int(action))
        reward = self._advance()
        return (self._next_observation(), reward, self.done,
                {"hands_played": self.ctx.hands_played, "hero_net_bb": self.ctx.hero_net_bb})

    def _blank(self):
        return {**self.tokenizer.blank(), "match_ctx": self.ctx.tensor(self.context_fields)}

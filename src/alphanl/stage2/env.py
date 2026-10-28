"""50-hand match schedule on the *same* standard HUNL rules as Stage 1.

The name describes the match schedule. Legacy TexasHoldem2p judge rules
have different action order/min-raises; deployment uses an explicit adapter.
"""
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.traversal import HunlState
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from .context import MatchCtx

class BotzoneMatchEnv:
    def __init__(self, config, opponent, *, seed=1, first_hero_seat=1):
        if config.separate_player_models or not config.position_conditioning:
            raise ValueError("Stage 2 requires unified position-conditioned Stage 1")
        if first_hero_seat not in (0, 1):
            raise ValueError("invalid first hero seat")
        self.config, self.opponent = config, opponent
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
        return {**self.observe_poker(), "match_ctx": self.ctx.tensor()}

    def reset(self):
        if not self.done:
            raise RuntimeError("cannot reset an unfinished match")
        self.ctx, self.hands, self.done = MatchCtx(), [], False
        self._deal()
        reward = self._advance()
        # A match may contain passive hands before the first decision.
        self.initial_reward = reward
        return self._blank() if self.done else self.observe()

    def _deal(self):
        self.hero = self.first_hero_seat ^ (self.ctx.hands_played % 2)
        self.table.reset()
        self.ctx.begin_hand()
        self.opponent.begin_hand(1 - self.hero)

    def _push(self, action):
        self.table.step(action)
        self.ctx.observe_action(self.table.events_for(self.hero)[-1])

    def _advance(self):
        reward = 0.
        for _ in range(50000):
            if self.table.is_over():
                payoff = float(self.table.payoffs()[self.hero]) / 100.
                if sum(self.table.payoffs()) != 0 or self.table.chips_in_play() != 40000:
                    raise RuntimeError("chip conservation failed")
                showdown = not any(e.action_type == "fold" for e in self.table.events_for(self.hero))
                self.ctx.finish_hand(payoff, showdown=showdown)
                reward += payoff / 200.
                self.hands.append({"hand": self.ctx.hands_played, "hero_seat": self.hero, "hero_net_bb": payoff})
                if self.ctx.hands_played == 50:
                    self.done = True
                    return reward
                self._deal()
            if self.table.current_player() == self.hero:
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
        if action not in self.guard.apply(self.table.snapshot(self.hero)).legal_action_ids:
            raise ValueError("hero action violates Stage 1 abstraction")
        self._push(int(action))
        reward = self._advance()
        return (self._blank() if self.done else self.observe(), reward, self.done,
                {"hands_played": self.ctx.hands_played, "hero_net_bb": self.ctx.hero_net_bb})

    def _blank(self):
        return {**self.tokenizer.blank(), "match_ctx": self.ctx.tensor()}

"""Heads-up self-play on the same rlcard 1.1.0 game AlphaNLHoldem wraps.

The learning agent sees one decision at a time. The opponent (random, the
current net, or a frozen pool checkpoint) plays its own seat until the hero
is to act again or the hand ends. Observations are ``HandSnapshot`` values,
so a pokerkit env can replace ``make_rlcard_env`` later without touching the
tokenizer or the policy head.

This module does not import TensorFlow or Ray. rlcard is imported only when
a hand is actually started.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .history import (
    BIG_BLIND_CHIPS,
    SMALL_BLIND_CHIPS,
    STARTING_STACK,
    build_snapshot,
    infer_sb_seat,
    legal_action_ids,
    stage_value,
)
from .model import PokerTransformer
from .pool import OpponentPool
from .tokenizer import BettingHistoryTokenizer, TensorBatch, batch_to_device


def make_rlcard_env(seed: int, starting_stack: int):
    try:
        import rlcard
    except ImportError as exc:
        raise ImportError(
            "rlcard is required for self-play. Install requirements-transformer.txt "
            "(rlcard==1.1.0 matches AlphaNLHoldem's 5-action 50bb game)."
        ) from exc
    return rlcard.make(
        "no-limit-holdem",
        config={
            "seed": int(seed),
            "game_num_players": 2,
            "chips_for_each": int(starting_stack),
        },
    )


class PolicyActor:
    """Stochastic legal actions for the opponent during a rollout."""

    def __init__(
        self,
        rollout_model: PokerTransformer,
        shadow: PokerTransformer,
        tokenizer: BettingHistoryTokenizer,
        rng: np.random.Generator,
    ) -> None:
        self.rollout_model = rollout_model
        self.shadow = shadow
        self.tokenizer = tokenizer
        self.rng = rng

    def rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state) -> None:
        self.rng.bit_generator.state = state

    def prepare_pool(self, weights: Dict[str, torch.Tensor]) -> None:
        device = next(self.shadow.parameters()).device
        self.shadow.load_state_dict({key: value.to(device) for key, value in weights.items()})

    def act(self, snapshot, choice: Dict[str, Any]) -> int:
        legal = list(snapshot.legal_action_ids)
        if not legal:
            raise RuntimeError("opponent saw no legal actions")
        if choice["kind"] == "random":
            return int(self.rng.choice(legal))
        if choice["kind"] == "pool":
            # One shared shadow serves every table, so reload this hand's weights
            # immediately before the forward. Copies are tiny relative to 8GB.
            self.prepare_pool(choice["weights"])
            model = self.shadow
        else:
            model = self.rollout_model
        device = next(model.parameters()).device
        batch = batch_to_device(self.tokenizer.encode(snapshot), device)
        actions, _, _, _ = model.act(batch, deterministic=False)
        action = int(actions.item())
        if action not in legal:
            return int(legal[0])
        return action


class SelfPlayEnv:
    """One heads-up table. ``reset`` / ``step`` speak in hero decisions."""

    def __init__(
        self,
        tokenizer: BettingHistoryTokenizer,
        actor: PolicyActor,
        pool: OpponentPool,
        *,
        seed: int,
        big_blind: float = BIG_BLIND_CHIPS,
        small_blind: float = SMALL_BLIND_CHIPS,
        starting_stack: float = STARTING_STACK,
        backend: str = "rlcard",
        record_results: bool = True,
    ) -> None:
        if backend != "rlcard":
            raise NotImplementedError(
                "Only the rlcard backend is implemented. A pokerkit backend should "
                "emit HandSnapshot events via make_action_event and reuse SelfPlayEnv's "
                "hero/opponent loop. See docs/step3_transformer.md."
            )
        self.tokenizer = tokenizer
        self.actor = actor
        self.pool = pool
        self.big_blind = float(big_blind)
        self.small_blind = int(small_blind)
        self.starting_stack = float(starting_stack)
        self.record_results = record_results
        self.rng = np.random.default_rng(seed)
        self.rl = make_rlcard_env(seed, int(starting_stack))
        self.force_opponent: Optional[Dict[str, Any]] = None
        self.passive_rewards_bb: list = []
        self.history: list = [[], [], [], []]
        self.state: Dict[str, Any] = {}
        self.player = 0
        self.hero = 0
        self.sb_seat = 0
        self.choice: Dict[str, Any] = {"kind": "random", "policy_id": None, "weights": None}

    _STATE_FIELDS = ("history", "state", "player", "hero", "sb_seat", "choice", "passive_rewards_bb", "force_opponent")

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to continue this table bit-for-bit: our RNG, the rlcard env
        (its game, dealer deck and np_random) and the hand in progress."""
        import pickle

        blob = {name: getattr(self, name) for name in self._STATE_FIELDS}
        return {"rng": self.rng.bit_generator.state, "rl": pickle.dumps(self.rl), "fields": pickle.dumps(blob)}

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import pickle

        self.rng.bit_generator.state = blob["rng"]
        self.rl = pickle.loads(blob["rl"])
        for name, value in pickle.loads(blob["fields"]).items():
            setattr(self, name, value)

    def reset(self) -> TensorBatch:
        for _ in range(1000):
            self._deal()
            if not self._play_until_hero():
                return self.observe()
            reward = self._hero_bb()
            self.passive_rewards_bb.append(reward)
            self._record(reward)
        raise RuntimeError("hero received no decision in 1000 hands")

    def step(self, action: int):
        if self.rl.game.is_over():
            raise RuntimeError("step() called on a finished hand")
        if int(self.player) != int(self.hero):
            raise RuntimeError("step() called off the hero's turn")
        self._push_and_step(int(action))
        ended = bool(self.rl.game.is_over()) or self._play_until_hero()
        reward = self._hero_bb() if ended else 0.0
        if ended:
            self._record(reward)
        info = {"policy_id": self.choice.get("policy_id"), "kind": self.choice.get("kind")}
        obs = self.tokenizer.blank() if ended else self.observe()
        return obs, float(reward), bool(ended), info

    def observe(self) -> TensorBatch:
        return self.tokenizer.encode(self._snapshot(self.hero))

    def drain_passive_rewards(self) -> list:
        rewards = list(self.passive_rewards_bb)
        self.passive_rewards_bb.clear()
        return rewards

    def _deal(self) -> None:
        self.state, self.player = self.rl.reset()
        self.player = int(self.player)
        self.history = [[], [], [], []]
        raw = self.state["raw_obs"]
        self.sb_seat = infer_sb_seat(raw["all_chips"], small_blind=self.small_blind)
        self.hero = int(self.rng.integers(0, 2))
        if self.force_opponent is not None:
            self.choice = dict(self.force_opponent)
        else:
            self.choice = self.pool.sample()
        if self.choice["kind"] == "pool":
            self.actor.prepare_pool(self.choice["weights"])

    def _play_until_hero(self) -> bool:
        for _ in range(64):
            if self.rl.game.is_over():
                return True
            if int(self.player) == int(self.hero):
                return False
            action = self.actor.act(self._snapshot(int(self.player)), self.choice)
            self._push_and_step(action)
        raise RuntimeError("opponent played more than 64 actions in one hand")

    def _push_and_step(self, action: int) -> None:
        raw = self.state["raw_obs"]
        stage = stage_value(raw["stage"])
        legal = legal_action_ids(raw["legal_actions"])
        actor = int(raw["current_player"])
        if action not in legal:
            if not legal:
                raise RuntimeError("no legal action to push")
            action = int(legal[0])
        if stage <= 3:
            self.history[stage].append([actor, int(action), list(legal)])
        self.state, self.player = self.rl.step(int(action))
        self.player = int(self.player)

    def _snapshot(self, me_seat: int):
        return build_snapshot(
            self.history,
            self.state["raw_obs"],
            int(me_seat),
            int(self.sb_seat),
            big_blind=self.big_blind,
            small_blind=self.small_blind,
            starting_stack=self.starting_stack,
        )

    def _hero_bb(self) -> float:
        payoffs = self.rl.get_payoffs()
        return float(payoffs[int(self.hero)]) / self.big_blind

    def _record(self, reward_bb: float) -> None:
        if self.record_results:
            self.pool.record(self.choice.get("policy_id"), reward_bb)

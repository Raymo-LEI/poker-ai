"""Heads-up self-play for the PyTorch trainer.

The learning agent sees one decision at a time. The opponent (random, the
current net, or a frozen pool checkpoint) plays its own seat until the hero
is to act again or the hand ends. Observations are ``HandSnapshot`` values.

``backend`` selects the table:

- ``rlcard`` — rlcard 1.1.0, the AlphaNLHoldem game (small blind acts first
  on every street).
- ``pokerkit`` — PokerKit heads-up no-limit. The button / small blind acts
  first preflop and the big blind acts first postflop.
- ``botzone`` — Botzone TexasHoldem2p simple (50 / 100, stacks of 20000).
  The small blind acts first on every street. Policy ids are applied only
  through ``alphanl.botzone.actions``. The reward that enters GAE is the
  hero's chip delta divided by the big blind and then by 200. Training logs
  multiply that reward by 200 and report bb/hand. rlcard and pokerkit stay
  in big blinds.

The gym methods (``reset``, ``step``, ``observe``, ``drain_passive_rewards``,
``state_dict`` / ``load_state_dict``) are the same for all three. Rules live
in ``tables.py``; this module only runs the hero / opponent loop.

This module does not import TensorFlow or Ray. rlcard and pokerkit are
imported only when a hand is actually started.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import resolve_env_limits
from .decision_stats import policy_action_mask
from .history import BIG_BLIND_CHIPS, SMALL_BLIND_CHIPS, STARTING_STACK
from .pool import OpponentPool
from .tables import make_table
from .tokenizer import BettingHistoryTokenizer, TensorBatch, batch_to_device, cat_batches

# Botzone simple starts 200 big blinds deep (20000 / 100). The value head
# fits a fraction of that depth. Other backends are not divided, so their
# logged bb/hand stays equal to the reward.
BOTZONE_REWARD_DEPTH = 200.0


def gae_reward_depth(backend: str) -> float:
    """Big blinds represented by one unit of reward entering GAE.

    ``botzone`` uses 200. ``rlcard`` and ``pokerkit`` use 1.
    """

    if str(backend) == "botzone":
        return BOTZONE_REWARD_DEPTH
    return 1.0


def logged_bb_from_reward(reward: float, backend: str) -> float:
    """bb/hand for logs. Botzone multiplies the GAE reward by 200."""

    return float(reward) * gae_reward_depth(backend)


class PolicyActor:
    """Stochastic legal actions for the opponent during a rollout."""

    def __init__(
        self,
        rollout_model: nn.Module,
        shadow: nn.Module,
        tokenizer: BettingHistoryTokenizer,
        rng: np.random.Generator,
    ) -> None:
        self.rollout_model = rollout_model
        self.shadow = shadow
        self.tokenizer = tokenizer
        self.rng = rng
        self._opponents: Dict[int, nn.Module] = {}
        # Tables sharing this actor; a snapshot evicted from the pool stays
        # cached while any table is still mid-hand against it.
        self._tables: list = []

    def rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state) -> None:
        self.rng.bit_generator.state = state

    # T-C1a: one frozen opponent per pool snapshot id, resident on the rollout
    # device. Set ``cache_opponents = False`` to get the legacy behaviour (one
    # shared shadow reloaded before every opponent move); the equivalence test
    # uses it to compare both paths.
    cache_opponents: bool = True

    def _build_opponent(self, weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Frozen deep copy of the shadow architecture holding ``weights``.

        The weights are copied (``load_state_dict`` copies into fresh
        parameters), so the model never aliases the pool's CPU tensors or the
        live learner.
        """
        model = copy.deepcopy(self.shadow)
        device = next(model.parameters()).device
        model.load_state_dict({key: value.detach().to(device).clone() for key, value in weights.items()})
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def prune_opponents(self, live_ids: Iterable[int]) -> None:
        """Drop cached opponents whose snapshot has left the pool."""
        keep = {int(pid) for pid in live_ids}
        for table in self._tables:
            choice = getattr(table, "choice", None) or {}
            if choice.get("kind") == "pool" and choice.get("policy_id") is not None:
                keep.add(int(choice["policy_id"]))
        for pid in list(self._opponents):
            if pid not in keep:
                del self._opponents[pid]

    def _opponent(self, choice: Dict[str, Any]) -> nn.Module:
        pid = choice.get("policy_id")
        if pid is None:
            # No id to key on: build a private frozen copy every time.
            return self._build_opponent(choice["weights"])
        model = self._opponents.get(int(pid))
        if model is None:
            # First sample of this snapshot (or a hand still running against a
            # snapshot that was evicted mid-hand).
            model = self._build_opponent(choice["weights"])
            self._opponents[int(pid)] = model
        return model

    def prepare_pool(self, weights: Dict[str, torch.Tensor], policy_id: Optional[int] = None,
                     live_ids: Optional[Iterable[int]] = None) -> None:
        if not self.cache_opponents:
            device = next(self.shadow.parameters()).device
            self.shadow.load_state_dict({key: value.to(device) for key, value in weights.items()})
            return
        if live_ids is not None:
            self.prune_opponents(set(live_ids) | ({int(policy_id)} if policy_id is not None else set()))
        self._opponent({"policy_id": policy_id, "weights": weights})

    # T-C1b: batch the opponent forward pass across envs (one forward per
    # opponent model). Only the logits are batched; each env still samples
    # its own action with a [1]-shaped draw, in the original env order, so the
    # torch RNG stream is consumed exactly as in the unbatched path. Needs
    # ``cache_opponents`` (the legacy shared shadow changes weights per move).
    # Set ``batch_rollout = False`` to get the T-C1a per-move forward.
    batch_rollout: bool = True

    def batching_enabled(self) -> bool:
        return bool(self.batch_rollout and self.cache_opponents)

    def _model_for(self, choice: Dict[str, Any]) -> Optional[nn.Module]:
        if choice["kind"] == "random":
            return None
        if choice["kind"] == "pool":
            return self._opponent(choice)
        return self.rollout_model

    def prefetch(self, envs: Sequence["SelfPlayEnv"]) -> int:
        """One batched forward per opponent model for every env whose opponent
        is to act now. Stores (obs, logits row) on the env; consumed by the
        env's next ``_actor_choose``. Draws no random numbers. Returns rows."""
        groups: Dict[int, list] = {}
        models: Dict[int, nn.Module] = {}
        for env in envs:
            env._prefetched = None
            seat = env.pending_opponent_seat()
            if seat is None:
                continue
            model = self._model_for(env.choice)
            if model is None:
                continue
            groups.setdefault(id(model), []).append((env, env._observe_seat(seat)))
            models[id(model)] = model
        rows = 0
        for key, items in groups.items():
            model = models[key]
            device = next(model.parameters()).device
            stacked = batch_to_device(cat_batches([b for _, b in items]), device)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                logits, _ = model(stacked, mask_illegal=True)
            if was_training:
                model.train()
            for i, (env, b) in enumerate(items):
                env._prefetched = (b, logits[i:i + 1])
            rows += len(items)
        return rows

    def act(self, snapshot, choice: Dict[str, Any]) -> int:
        return self.act_batch(self.tokenizer.encode(snapshot), choice)

    def act_batch(self, batch: TensorBatch, choice: Dict[str, Any], logits: Optional[torch.Tensor] = None) -> int:
        legal = [index for index, flag in enumerate(batch["legal_mask"][0].tolist()) if flag > 0.5]
        if not legal:
            raise RuntimeError("opponent saw no legal actions")
        if choice["kind"] == "random":
            return int(self.rng.choice(legal))
        if logits is not None:
            # T-C1b: same sampling as policy_act on a single row.
            from .model import masked_log_probs
            mask = batch["legal_mask"].to(logits.device)
            _, _, probs, _ = masked_log_probs(logits, mask)
            action = int(torch.distributions.Categorical(probs=probs).sample().item())
            if action not in legal:
                return int(legal[0])
            return action
        if choice["kind"] == "pool":
            if self.cache_opponents:
                model = self._opponent(choice)
            else:
                # Legacy: one shared shadow serves every table, reloaded before each move.
                self.prepare_pool(choice["weights"])
                model = self.shadow
        else:
            model = self.rollout_model
        device = next(model.parameters()).device
        moved = batch_to_device(batch, device)
        actions, _, _, _ = model.act(moved, deterministic=False)
        action = int(actions.item())
        if action not in legal:
            return int(legal[0])
        return action


class SelfPlayEnv:
    """One heads-up table. ``reset`` / ``step`` speak in hero decisions."""

    def __init__(
        self,
        tokenizer: BettingHistoryTokenizer,
        actor: Any,
        pool: OpponentPool,
        *,
        seed: int,
        big_blind: float = BIG_BLIND_CHIPS,
        small_blind: float = SMALL_BLIND_CHIPS,
        starting_stack: float = STARTING_STACK,
        backend: str = "rlcard",
        record_results: bool = True,
        observer: Optional[Any] = None,
        mask_free_fold: bool = False,
    ) -> None:
        small_blind, big_blind, starting_stack = resolve_env_limits(
            backend, small_blind, big_blind, starting_stack
        )
        self.tokenizer = tokenizer
        self.actor = actor
        self.pool = pool
        self.backend = str(backend)
        self.observer = observer
        self.big_blind = float(big_blind)
        self.small_blind = int(small_blind)
        self.starting_stack = float(starting_stack)
        self.record_results = record_results
        # Policy copies (self-play and the pool) use this. Random opponents
        # keep the environment mask. The flag is not part of the table blob:
        # a resume builds the env from the config and then loads the hand.
        self.mask_free_fold = bool(mask_free_fold)
        self.rng = np.random.default_rng(seed)
        self.table = make_table(
            self.backend,
            seed=int(seed),
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        self.force_opponent: Optional[Dict[str, Any]] = None
        # Same units as the reward ``step`` returns (bb, or bb/200 on botzone).
        self.passive_rewards_bb: list = []
        self.hero = 0
        self.choice: Dict[str, Any] = {"kind": "random", "policy_id": None, "weights": None}
        self._prefetched = None
        if isinstance(actor, PolicyActor):
            actor._tables.append(self)

    _STATE_FIELDS = ("hero", "choice", "passive_rewards_bb", "force_opponent")

    @property
    def sb_seat(self) -> int:
        return int(self.table.sb_seat)

    def state_dict(self) -> Dict[str, Any]:
        """RNG, the in-progress hand, and who the hero is.

        The table blob is backend-specific. An rlcard checkpoint still
        unpickles the rlcard env; a pokerkit checkpoint unpickles the
        PokerKit ``State`` plus that table's deck RNG; a botzone checkpoint
        unpickles ``BotzoneTable`` plus the policy-id history. Loading one
        backend into another raises ``ValueError``. Blinds and
        ``starting_stack`` are stored beside ``backend`` and must match the
        env that loads them. A checkpoint written before the table wrapper
        (keys ``rl`` and ``fields`` only) still loads into an rlcard env.
        A checkpoint written before blinds were stored (no ``small_blind``
        key) still loads into the same non-botzone backend.
        """

        import pickle

        blob = {name: getattr(self, name) for name in self._STATE_FIELDS}
        return {
            "rng": self.rng.bit_generator.state,
            "backend": self.backend,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stack": self.starting_stack,
            "table": self.table.state_dict(),
            "fields": pickle.dumps(blob),
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import pickle

        if "table" not in blob and "rl" in blob:
            self._load_legacy_rlcard(blob)
            return
        backend = blob.get("backend")
        if backend != self.backend:
            raise ValueError(
                f"checkpoint backend is {backend!r}, this env is {self.backend!r}"
            )
        self._check_limits(blob)
        self.rng.bit_generator.state = blob["rng"]
        self.table.load_state_dict(blob["table"])
        for name, value in pickle.loads(blob["fields"]).items():
            setattr(self, name, value)

    def _load_legacy_rlcard(self, blob: Dict[str, Any]) -> None:
        import pickle

        if self.backend != "rlcard":
            raise ValueError(
                f"legacy checkpoint is an rlcard table and cannot load into {self.backend}"
            )
        self.rng.bit_generator.state = blob["rng"]
        fields = pickle.loads(blob["fields"])
        self.table.load_state_dict(
            {
                "version": 1,
                "rl": blob["rl"],
                "history": fields["history"],
                "state": fields["state"],
                "player": fields["player"],
                "sb_seat": fields["sb_seat"],
            }
        )
        for name in self._STATE_FIELDS:
            setattr(self, name, fields[name])

    def reset(self, deck: Optional[Sequence[str]] = None) -> TensorBatch:
        """Start a hand and return the hero's first decision.

        ``deck`` is the deal order for that first hand only. Later hands
        in the same call (when the hero was dealt out without a decision)
        are shuffled normally.
        """

        first = True
        for _ in range(1000):
            self._deal(deck if first else None)
            first = False
            if not self._play_until_hero():
                return self.observe()
            reward = self._gae_reward()
            self.passive_rewards_bb.append(reward)
            self._record(reward)
        raise RuntimeError("hero received no decision in 1000 hands")

    def step(self, action: int):
        self.begin_step(action)
        return self.finish_step()

    # T-C1b: ``step`` split in two so a rollout can push every env's hero
    # action, batch the opponents' next forward pass, then finish each env in
    # order. ``begin_step`` only advances this env's own table.
    def begin_step(self, action: int) -> None:
        if self.table.is_over():
            raise RuntimeError("step() called on a finished hand")
        if int(self.table.current_player()) != int(self.hero):
            raise RuntimeError("step() called off the hero's turn")
        self._prefetched = None
        self._push_and_step(int(action))

    def pending_opponent_seat(self) -> Optional[int]:
        if self.table.is_over():
            return None
        seat = int(self.table.current_player())
        return None if seat == int(self.hero) else seat

    def finish_step(self):
        ended = bool(self.table.is_over()) or self._play_until_hero()
        reward = self._gae_reward() if ended else 0.0
        if ended:
            self._record(reward)
        info = {"policy_id": self.choice.get("policy_id"), "kind": self.choice.get("kind")}
        obs = self._blank() if ended else self.observe()
        return obs, float(reward), bool(ended), info

    def observe(self) -> TensorBatch:
        return self._observe_seat(self.hero)

    def _observe_seat(self, seat: int) -> TensorBatch:
        if self.observer is not None:
            return self.observer.observe_table(self.table, int(seat))
        return self.tokenizer.encode(self.table.snapshot(int(seat)))

    def _blank(self) -> TensorBatch:
        if self.observer is not None:
            return self.observer.blank()
        return self.tokenizer.blank()

    def drain_passive_rewards(self) -> list:
        rewards = list(self.passive_rewards_bb)
        self.passive_rewards_bb.clear()
        return rewards

    def _check_limits(self, blob: Dict[str, Any]) -> None:
        keys = ("small_blind", "big_blind", "starting_stack")
        present = [key for key in keys if key in blob]
        if self.backend == "botzone":
            if len(present) != len(keys):
                missing = [key for key in keys if key not in blob]
                raise ValueError(f"botzone checkpoint is missing {missing}")
        elif len(present) == 0:
            return
        elif len(present) != len(keys):
            missing = [key for key in keys if key not in blob]
            raise ValueError(f"checkpoint is missing {missing}")
        saved = tuple(float(blob[key]) for key in keys)
        current = (float(self.small_blind), float(self.big_blind), float(self.starting_stack))
        if any(abs(left - right) > 1e-6 for left, right in zip(saved, current)):
            raise ValueError(
                "checkpoint blinds/stack "
                f"(small_blind={saved[0]}, big_blind={saved[1]}, starting_stack={saved[2]}) "
                "do not match this env "
                f"(small_blind={current[0]}, big_blind={current[1]}, starting_stack={current[2]})"
            )

    def _deal(self, deck: Optional[Sequence[str]]) -> None:
        self.table.reset(deck)
        self.hero = int(self.rng.integers(0, 2))
        if self.force_opponent is not None:
            self.choice = dict(self.force_opponent)
        else:
            self.choice = self.pool.sample()
        if self.choice["kind"] == "pool":
            if isinstance(self.actor, PolicyActor):
                self.actor.prepare_pool(
                    self.choice["weights"],
                    policy_id=self.choice.get("policy_id"),
                    live_ids=self.pool.live_ids(),
                )
            else:
                self.actor.prepare_pool(self.choice["weights"])

    def _policy_batch(self, batch: TensorBatch, seat: int) -> TensorBatch:
        """Environment mask for a random opponent. Policy mask otherwise.

        Fold at a free check stays in the batch a random opponent draws from.
        A neural opponent sees the same mask the learner samples from.
        """

        if not self.mask_free_fold or self.choice.get("kind") == "random":
            return batch
        owed = float(self.table.snapshot(int(seat)).to_call)
        masked = dict(batch)
        masked["legal_mask"] = policy_action_mask(batch["legal_mask"], owed, True)
        return masked

    def _actor_choose(self, seat: int) -> int:
        pre, self._prefetched = self._prefetched, None
        if pre is not None:
            batch = self._policy_batch(pre[0], seat)
            return self.actor.act_batch(batch, self.choice, logits=pre[1])
        batch = self._policy_batch(self._observe_seat(seat), seat)
        if hasattr(self.actor, "act_batch"):
            return self.actor.act_batch(batch, self.choice)
        return self.actor.act(self.table.snapshot(int(seat)), self.choice)

    def _play_until_hero(self) -> bool:
        for _ in range(64):
            if self.table.is_over():
                return True
            if int(self.table.current_player()) == int(self.hero):
                return False
            action = self._actor_choose(int(self.table.current_player()))
            self._push_and_step(action)
        raise RuntimeError("opponent played more than 64 actions in one hand")

    def _push_and_step(self, action: int) -> None:
        legal = self.table.legal_action_ids()
        if action not in legal:
            if not legal:
                raise RuntimeError("no legal action to push")
            action = int(legal[0])
        self.table.step(int(action))

    def _hero_bb(self) -> float:
        payoffs = self.table.payoffs()
        return float(payoffs[int(self.hero)]) / self.big_blind

    def _gae_reward(self) -> float:
        """Reward that enters GAE. Botzone is ``_hero_bb() / 200``."""

        return self._hero_bb() / gae_reward_depth(self.backend)

    def _record(self, reward_bb: float) -> None:
        if self.record_results:
            self.pool.record(self.choice.get("policy_id"), reward_bb)

"""External-sampling MCCFR with batched inference across independent trees.

Every traverser action is enumerated, including zero-probability actions.
Only opponent actions and chance are sampled. Chance can be pre-sampled at
the root (as a shuffled HUNL deck); sharing that deck across siblings is a
common-random-number estimator, not a look-ahead feature for the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence
import time

import numpy as np
import torch

from alphanl.transformer.tokenizer import TensorBatch, batch_to_device, cat_batches


class GameState(Protocol):
    """Decision state after sampled chance; observations hide other hands."""

    def is_terminal(self) -> bool: ...
    def current_player(self) -> int: ...
    def legal_actions(self) -> Sequence[int]: ...
    def observation(self) -> TensorBatch: ...
    def child(self, action: int) -> "GameState": ...
    def returns(self) -> Sequence[float]: ...


class HunlState:
    """Adapter for the locked 200BB table; terminal utilities are in BB."""

    def __init__(self, table, tokenizer, action_guard=None) -> None:
        self.table = table
        self.tokenizer = tokenizer
        self.action_guard = action_guard
        self._decision_snapshot = None

    def is_terminal(self) -> bool:
        return self.table.is_over()

    def current_player(self) -> int:
        return self.table.current_player()

    def legal_actions(self) -> Sequence[int]:
        return self._snapshot().legal_action_ids

    def _snapshot(self):
        if self._decision_snapshot is None:
            snapshot = self.table.snapshot(self.current_player())
            self._decision_snapshot = self.action_guard.apply(snapshot) if self.action_guard else snapshot
        return self._decision_snapshot

    def observation(self) -> TensorBatch:
        return self.tokenizer.encode(self._snapshot())

    def scripted_action(self, actor) -> int:
        # Only the acting seat's visible HandSnapshot crosses this boundary.
        # Match evaluation's unguarded rule bot; the learned traverser retains
        # its configured guard. This also covers bot branches outside that guard.
        return actor.act(self.table.snapshot(self.current_player()))

    def scripted_legal_actions(self):
        return self.table.legal_action_ids()

    def child(self, action: int) -> "HunlState":
        table = self.table.clone()
        table.step(action)
        return HunlState(table, self.tokenizer, self.action_guard)

    def returns(self) -> Sequence[float]:
        return tuple(value / self.table.big_blind for value in self.table.payoffs())


@dataclass
class TraversalStats:
    selfplay_roots: int = 0
    scripted_roots: int = 0
    scripted_equity_roots: int = 0
    scripted_nit_roots: int = 0
    scripted_opponent_actions: int = 0
    scripted_off_guard_actions: int = 0
    scripted_regret_samples: int = 0
    nodes: int = 0
    terminal_nodes: int = 0
    regret_samples: int = 0
    inference_calls: int = 0
    inference_rows: int = 0
    inference_singleton_calls: int = 0
    inference_max_batch: int = 0
    inference_batches_2_7: int = 0
    inference_batches_8_31: int = 0
    inference_batches_32_127: int = 0
    inference_batches_128_plus: int = 0

    def record_inference(self, count: int) -> None:
        self.inference_calls += 1
        self.inference_rows += count
        self.inference_max_batch = max(self.inference_max_batch, count)
        field = ("inference_singleton_calls" if count == 1 else "inference_batches_2_7" if count < 8
                 else "inference_batches_8_31" if count < 32 else "inference_batches_32_127" if count < 128
                 else "inference_batches_128_plus")
        setattr(self, field, getattr(self, field) + 1)


@dataclass
class _Query:
    player: int
    batch: TensorBatch


class ExternalSamplingTraverser:
    """Cooperative DFS workers with a replenished pool of independent trees.

    This vectorizes inference without multiprocessing/global engine patches.
    ``batch_size`` controls concurrent trees. Inference is further grouped by
    player because each player has its own advantage network and memory.
    No reach factor is multiplied into the sampled regret: opponent/chance
    reach is already represented by the probability of visiting this node.
    """

    def __init__(self, *, seed: int = 1, batch_size: int = 32,
                 max_nodes: int | None = None, scripted_opponent_prob: float = 0.0,
                 scripted_opponent_pool: dict[str, float] | None = None,
                 scripted_opponent_equity_samples: int = 64) -> None:
        if batch_size < 1 or (max_nodes is not None and max_nodes < 1):
            raise ValueError("batch_size and max_nodes must be positive")
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.max_nodes = max_nodes
        import math
        pool = {"equity": .75, "nit": .25} if scripted_opponent_pool is None else dict(scripted_opponent_pool)
        if (not math.isfinite(scripted_opponent_prob) or not 0 <= scripted_opponent_prob <= 1
                or not pool or set(pool) - {"equity", "nit"}
                or any(not math.isfinite(w) or w <= 0 for w in pool.values())
                or scripted_opponent_equity_samples < 1):
            raise ValueError("invalid scripted opponent mixture")
        self.scripted_opponent_prob = scripted_opponent_prob
        self.scripted_names = tuple(sorted(pool))
        self.scripted_probabilities = np.array([pool[name] for name in self.scripted_names], dtype=np.float64)
        self.scripted_probabilities /= self.scripted_probabilities.sum()
        self.scripted_opponent_equity_samples = scripted_opponent_equity_samples
        self.last_timing = {}

    def _walk(self, state: GameState, traverser: int, iteration: int,
              buffer, stats: TraversalStats, scripted_opponent=None):
        stats.nodes += 1
        if self.max_nodes is not None and stats.nodes > self.max_nodes:
            raise RuntimeError("traversal node budget exceeded; no truncated utility is used")
        if state.is_terminal():
            stats.terminal_nodes += 1
            return float(state.returns()[traverser])
        player = state.current_player()
        if player not in (0, 1):
            raise ValueError("external sampling requires a two-player game")
        legal = tuple(int(a) for a in state.legal_actions())
        if player != traverser and scripted_opponent is not None:
            if not hasattr(state, "scripted_action"):
                raise TypeError("scripted opponents require a visible-state scripted_action adapter")
            action = int(state.scripted_action(scripted_opponent))
            scripted_legal = state.scripted_legal_actions() if hasattr(state, "scripted_legal_actions") else legal
            if action not in scripted_legal:
                raise ValueError("scripted opponent selected an illegal action")
            stats.scripted_opponent_actions += 1
            stats.scripted_off_guard_actions += int(action not in legal)
            return (yield from self._walk(state.child(action), traverser, iteration, buffer, stats, scripted_opponent))
        began = time.perf_counter()
        batch = state.observation()
        self.last_timing["observation_s"] += time.perf_counter() - began
        mask_ids = tuple(torch.nonzero(batch["legal_mask"][0] > 0.5).flatten().tolist())
        if not legal or len(legal) != len(set(legal)) or tuple(sorted(legal)) != mask_ids:
            raise ValueError("game legal actions must match its observation mask")
        strategy = yield _Query(player, batch)
        if player != traverser:
            action = int(self.rng.choice(len(strategy), p=strategy))
            return (yield from self._walk(state.child(action), traverser, iteration, buffer, stats, scripted_opponent))
        action_values = np.zeros(len(strategy), dtype=np.float64)
        for action in legal:
            action_values[action] = yield from self._walk(
                state.child(action), traverser, iteration, buffer, stats, scripted_opponent
            )
        value = float(np.dot(strategy, action_values))
        regret = np.zeros(len(strategy), dtype=np.float32)
        regret[list(legal)] = action_values[list(legal)] - value
        began = time.perf_counter()
        buffer.add(batch, torch.from_numpy(regret).unsqueeze(0), iteration)
        self.last_timing["buffer_insert_s"] += time.perf_counter() - began
        stats.regret_samples += 1
        stats.scripted_regret_samples += int(scripted_opponent is not None)
        return value

    @torch.no_grad()
    def collect(self, roots: Sequence[GameState], *, traverser: int, iteration: int,
                policies: Mapping[int, torch.nn.Module], buffer) -> TraversalStats:
        if traverser not in (0, 1) or iteration < 1:
            raise ValueError("traverser must be 0/1 and iteration must be >= 1")
        stats = TraversalStats()
        self.last_timing = {"observation_s": 0.0, "buffer_insert_s": 0.0,
                            "packing_s": 0.0, "inference_s": 0.0, "root_s": 0.0}
        training = {p: policies[p].training for p in (0, 1)}
        for policy in policies.values():
            policy.eval()
        try:
            root_iter = iter(roots)
            active = []
            roots_exhausted = False
            while active or not roots_exhausted:
                # Start the next roots as soon as earlier trees finish. This
                # keeps a wave full when tree depths vary substantially.
                while len(active) < self.batch_size and not roots_exhausted:
                    try:
                        began = time.perf_counter()
                        root = next(root_iter)
                        self.last_timing["root_s"] += time.perf_counter() - began
                    except StopIteration:
                        roots_exhausted = True
                        break
                    opponent = None
                    # One Bernoulli/identity draw per root, never per decision.
                    # Zero leaves the historical RNG stream exactly unchanged.
                    if self.scripted_opponent_prob and self.rng.random() < self.scripted_opponent_prob:
                        from .evaluation import RuleActor
                        name = str(self.rng.choice(self.scripted_names, p=self.scripted_probabilities))
                        opponent = RuleActor(name, seed=int(self.rng.integers(0, 2**63 - 1)),
                                             equity_samples=self.scripted_opponent_equity_samples)
                        opponent.begin_hand(1 - traverser)
                        stats.scripted_roots += 1
                        setattr(stats, f"scripted_{name}_roots", getattr(stats, f"scripted_{name}_roots") + 1)
                    else:
                        stats.selfplay_roots += 1
                    worker = self._walk(root, traverser, iteration, buffer, stats, opponent)
                    try:
                        active.append((worker, next(worker)))
                    except StopIteration:
                        continue
                if not active:
                    continue
                responses = {}
                for player in (0, 1):
                    indexed = [(i, query) for i, (_, query) in enumerate(active)
                               if query.player == player]
                    if not indexed:
                        continue
                    policy = policies[player]
                    device = next(policy.parameters()).device
                    began = time.perf_counter()
                    cpu_batch = cat_batches([q.batch for _, q in indexed])
                    batch = batch_to_device(cpu_batch, device)
                    self.last_timing["packing_s"] += time.perf_counter() - began
                    began = time.perf_counter()
                    probs = policy.strategy(batch).detach().double().cpu().numpy()
                    self.last_timing["inference_s"] += time.perf_counter() - began
                    if probs.shape != tuple(batch["legal_mask"].shape):
                        raise ValueError("policy strategy shape must match legal mask")
                    if (not np.isfinite(probs).all() or (probs < 0).any()
                            or not np.allclose(probs.sum(-1), 1, atol=1e-6)):
                        raise ValueError("policy returned invalid strategy probabilities")
                    legal = cpu_batch["legal_mask"].numpy() > 0.5
                    if (probs[~legal] != 0).any():
                        raise ValueError("policy assigns mass to an illegal action")
                    probs /= probs.sum(-1, keepdims=True)
                    stats.record_inference(len(indexed))
                    for (i, _), row in zip(indexed, probs):
                        responses[i] = row
                pending = []
                for i, (worker, _) in enumerate(active):
                    try:
                        pending.append((worker, worker.send(responses[i])))
                    except StopIteration:
                        pass
                active = pending
        finally:
            for player, was_training in training.items():
                policies[player].train(was_training)
        return stats

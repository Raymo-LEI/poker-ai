"""External-sampling MCCFR with batched inference across independent trees.

Every traverser action is enumerated, including zero-probability actions.
Only opponent actions and chance are sampled. Chance can be pre-sampled at
the root (as a shuffled HUNL deck); sharing that deck across siblings is a
common-random-number estimator, not a look-ahead feature for the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

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

    def __init__(self, table, tokenizer) -> None:
        self.table = table
        self.tokenizer = tokenizer

    def is_terminal(self) -> bool:
        return self.table.is_over()

    def current_player(self) -> int:
        return self.table.current_player()

    def legal_actions(self) -> Sequence[int]:
        return self.table.legal_action_ids()

    def observation(self) -> TensorBatch:
        return self.tokenizer.encode(self.table.snapshot(self.current_player()))

    def child(self, action: int) -> "HunlState":
        table = self.table.clone()
        table.step(action)
        return HunlState(table, self.tokenizer)

    def returns(self) -> Sequence[float]:
        return tuple(value / self.table.big_blind for value in self.table.payoffs())


@dataclass
class TraversalStats:
    nodes: int = 0
    terminal_nodes: int = 0
    regret_samples: int = 0
    inference_calls: int = 0
    inference_rows: int = 0


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
                 max_nodes: int | None = None) -> None:
        if batch_size < 1 or (max_nodes is not None and max_nodes < 1):
            raise ValueError("batch_size and max_nodes must be positive")
        self.rng = np.random.default_rng(seed)
        self.batch_size = int(batch_size)
        self.max_nodes = max_nodes

    def _walk(self, state: GameState, traverser: int, iteration: int,
              buffer, stats: TraversalStats):
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
        batch = state.observation()
        mask_ids = tuple(torch.nonzero(batch["legal_mask"][0] > 0.5).flatten().tolist())
        if not legal or len(legal) != len(set(legal)) or tuple(sorted(legal)) != mask_ids:
            raise ValueError("game legal actions must match its observation mask")
        strategy = yield _Query(player, batch)
        if player != traverser:
            action = int(self.rng.choice(len(strategy), p=strategy))
            return (yield from self._walk(state.child(action), traverser, iteration, buffer, stats))
        action_values = np.zeros(len(strategy), dtype=np.float64)
        for action in legal:
            action_values[action] = yield from self._walk(
                state.child(action), traverser, iteration, buffer, stats
            )
        value = float(np.dot(strategy, action_values))
        regret = np.zeros(len(strategy), dtype=np.float32)
        regret[list(legal)] = action_values[list(legal)] - value
        buffer.add(batch, torch.from_numpy(regret).unsqueeze(0), iteration)
        stats.regret_samples += 1
        return value

    @torch.no_grad()
    def collect(self, roots: Sequence[GameState], *, traverser: int, iteration: int,
                policies: Mapping[int, torch.nn.Module], buffer) -> TraversalStats:
        if traverser not in (0, 1) or iteration < 1:
            raise ValueError("traverser must be 0/1 and iteration must be >= 1")
        stats = TraversalStats()
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
                        root = next(root_iter)
                    except StopIteration:
                        roots_exhausted = True
                        break
                    worker = self._walk(root, traverser, iteration, buffer, stats)
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
                    batch = batch_to_device(cat_batches([q.batch for _, q in indexed]), device)
                    probs = policy.strategy(batch).detach().double().cpu().numpy()
                    if probs.shape != tuple(batch["legal_mask"].shape):
                        raise ValueError("policy strategy shape must match legal mask")
                    if (not np.isfinite(probs).all() or (probs < 0).any()
                            or not np.allclose(probs.sum(-1), 1, atol=1e-6)):
                        raise ValueError("policy returned invalid strategy probabilities")
                    legal = batch["legal_mask"].detach().cpu().numpy() > 0.5
                    if (probs[~legal] != 0).any():
                        raise ValueError("policy assigns mass to an illegal action")
                    probs /= probs.sum(-1, keepdims=True)
                    stats.inference_calls += 1
                    stats.inference_rows += len(indexed)
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

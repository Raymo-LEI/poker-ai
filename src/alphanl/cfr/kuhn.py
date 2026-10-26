"""Exact Kuhn Poker sanity check for the production external sampler.

This uses tabular regret estimates to isolate traversal, masking, signed
targets, and linear strategy averaging from neural function approximation.
It is not evidence that a Transformer has converged on no-limit Hold'em.

Run ``python -m alphanl.cfr.kuhn --iterations 1000``. Exploitability is half
NashConv, evaluated by enumerating deterministic *infoset* best responses.
An opponent's private rank never appears in an observation or a BR decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations, product
import json
from typing import Mapping

import numpy as np
import torch
from torch import nn

from alphanl.transformer.tokenizer import (
    CARD_ID_BASE,
    CLS_ID,
    HOLE_MARKER_ID,
    STREET_ID_BASE,
    TYPE_ACTION,
    TYPE_CLS,
    TYPE_HOLE_CARD,
    TYPE_HOLE_MARKER,
    TYPE_STREET,
    TensorBatch,
    action_token_id,
)

from .model import regret_matching
from .traversal import ExternalSamplingTraverser

HISTORIES = ("", "c", "b", "cb")
TERMINALS = ("cc", "bc", "bf", "cbc", "cbf")
DEALS = tuple(permutations(range(3), 2))
Infoset = tuple[int, str]
Strategy = dict[Infoset, np.ndarray]


def _player(history: str) -> int:
    return 0 if history in ("", "cb") else 1


def _legal(history: str) -> tuple[int, ...]:
    return (0, 1) if history in ("b", "cb") else (1, 2)


def _history_tokens(history: str) -> tuple[int, ...]:
    observer = _player(history)
    tokens = []
    for index, symbol in enumerate(history):
        actor = index % 2
        tokens.append(action_token_id(
            "check" if symbol == "c" else "bet",
            "SB" if actor == 0 else "BB",
            actor == observer,
            0 if symbol == "c" else 2,
        ))
    return tuple(tokens)


_HISTORY_FROM_TOKENS = {_history_tokens(history): history for history in HISTORIES}


def infoset_key(batch: TensorBatch, row: int = 0) -> Infoset:
    """Decode only own card and public history from the observation."""
    tokens = batch["token_ids"][row].detach().cpu().tolist()
    active = (~batch["pad_mask"][row]).sum().item()
    rank = int(tokens[2]) - CARD_ID_BASE
    history = _HISTORY_FROM_TOKENS[tuple(tokens[4:int(active)])]
    if rank not in range(3):
        raise ValueError("not a Kuhn infoset observation")
    return rank, history


@dataclass(frozen=True)
class KuhnState:
    """Two-player Kuhn Poker, after the chance deal of distinct J/Q/K ranks.

    Each player antes one unit; a single bet is one unit. Action ids share
    the Hold'em five-way head: 0 fold, 1 check/call, 2 bet. Slots 3/4 are
    always illegal. The state may know the full deal; its observation does not.
    """

    cards: tuple[int, int]
    history: str = ""

    def __post_init__(self) -> None:
        if len(self.cards) != 2 or len(set(self.cards)) != 2 or any(card not in range(3) for card in self.cards):
            raise ValueError("Kuhn requires two distinct ranks from 0/1/2")
        if self.history not in HISTORIES + TERMINALS:
            raise ValueError(f"invalid Kuhn history {self.history!r}")

    def is_terminal(self) -> bool:
        return self.history in TERMINALS

    def current_player(self) -> int:
        if self.is_terminal():
            raise RuntimeError("terminal states have no actor")
        return _player(self.history)

    def legal_actions(self) -> tuple[int, ...]:
        return () if self.is_terminal() else _legal(self.history)

    @property
    def infoset(self) -> Infoset:
        return self.cards[self.current_player()], self.history

    def observation(self) -> TensorBatch:
        actor = self.current_player()
        tokens = [CLS_ID, HOLE_MARKER_ID, CARD_ID_BASE + self.cards[actor], STREET_ID_BASE]
        tokens.extend(_history_tokens(self.history))
        token_ids = torch.zeros(1, 8, dtype=torch.long)
        token_ids[0, :len(tokens)] = torch.tensor(tokens)
        type_ids = torch.zeros_like(token_ids)
        type_ids[0, :4] = torch.tensor([TYPE_CLS, TYPE_HOLE_MARKER, TYPE_HOLE_CARD, TYPE_STREET])
        type_ids[0, 4:len(tokens)] = TYPE_ACTION
        pad_mask = torch.ones(1, 8, dtype=torch.bool)
        pad_mask[0, :len(tokens)] = False
        legal_mask = torch.zeros(1, 5)
        legal_mask[0, list(self.legal_actions())] = 1
        numeric = torch.zeros(1, 8, 4)
        # Public pot, own committed ante, opponent committed ante, call amount.
        # No numerical field depends on the other player's rank.
        facing_bet = self.history in ("b", "cb")
        numeric[0, 0] = torch.tensor([0.75 if facing_bet else 0.5,
                                    0.5, 1.0 if facing_bet else 0.5,
                                    1.0 if facing_bet else 0.0])
        return {"token_ids": token_ids, "type_ids": type_ids, "numeric": numeric,
                "pad_mask": pad_mask, "legal_mask": legal_mask}

    def child(self, action: int) -> "KuhnState":
        if action not in self.legal_actions():
            raise ValueError(f"illegal Kuhn action {action}")
        symbol = {0: "f", 1: "c", 2: "b"}[action]
        return KuhnState(self.cards, self.history + symbol)

    def returns(self) -> tuple[float, float]:
        if not self.is_terminal():
            raise RuntimeError("returns require a terminal state")
        if self.history == "bf":
            value = 1.0
        elif self.history == "cbf":
            value = -1.0
        else:
            stake = 1.0 if self.history == "cc" else 2.0
            value = stake if self.cards[0] > self.cards[1] else -stake
        return value, -value


class KuhnRootFactory:
    """Independent reproducible uniform chance deals."""

    def __init__(self, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self) -> KuhnState:
        cards = self.rng.choice(3, size=2, replace=False)
        return KuhnState((int(cards[0]), int(cards[1])))


class TabularRegretPolicy(nn.Module):
    """Exact table plugged into the same ``strategy`` API as RegretNetwork."""

    def __init__(self) -> None:
        super().__init__()
        # The production scheduler uses a parameter to select its device.
        self.device_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer("regrets", torch.zeros(12, 5, dtype=torch.float64))

    @staticmethod
    def index(key: Infoset) -> int:
        rank, history = key
        return 3 * HISTORIES.index(history) + rank

    def strategy(self, batch: TensorBatch) -> torch.Tensor:
        indexes = torch.tensor([self.index(infoset_key(batch, row))
                                for row in range(batch["token_ids"].shape[0])],
                               device=self.regrets.device)
        return regret_matching(self.regrets[indexes], batch["legal_mask"])

    def at(self, key: Infoset) -> np.ndarray:
        rank, history = key
        other = (rank + 1) % 3
        cards = (rank, other) if _player(history) == 0 else (other, rank)
        with torch.no_grad():
            return self.strategy(KuhnState(cards, history).observation())[0].cpu().numpy().astype(np.float64)


class _IterationRegrets:
    """Adapt the production reservoir interface to an exact cumulative table."""

    def __init__(self, policy: TabularRegretPolicy) -> None:
        self.policy = policy
        self.pending = torch.zeros_like(policy.regrets)
        self.samples = 0

    def add(self, batch: TensorBatch, regrets: torch.Tensor, iteration: int) -> None:
        for row in range(len(regrets)):
            index = self.policy.index(infoset_key(batch, row))
            self.pending[index] += iteration * regrets[row].double().cpu()
            self.samples += 1

    def apply(self) -> None:
        self.policy.regrets += self.pending
        self.pending.zero_()


def uniform_strategy() -> Strategy:
    result = {}
    for rank, history in product(range(3), HISTORIES):
        row = np.zeros(5)
        row[list(_legal(history))] = 0.5
        result[(rank, history)] = row
    return result


def nash_strategy() -> Strategy:
    """One known Kuhn equilibrium (first player's J bluff frequency 1/3)."""
    result = uniform_strategy()
    bets = {"": (1 / 3, 0, 1), "c": (1 / 3, 0, 1)}
    calls = {"b": (0, 1 / 3, 1), "cb": (0, 2 / 3, 1)}
    for history, probabilities in bets.items():
        for rank, probability in enumerate(probabilities):
            result[(rank, history)] = np.array([0, 1 - probability, probability, 0, 0])
    for history, probabilities in calls.items():
        for rank, probability in enumerate(probabilities):
            result[(rank, history)] = np.array([1 - probability, probability, 0, 0, 0])
    return result


def _state_value(state: KuhnState, strategy: Mapping[Infoset, np.ndarray]) -> float:
    if state.is_terminal():
        return state.returns()[0]
    row = strategy[state.infoset]
    return sum(float(row[action]) * _state_value(state.child(action), strategy)
               for action in state.legal_actions())


def game_value(strategy: Mapping[Infoset, np.ndarray]) -> float:
    """Expected utility of player zero across all six equally likely deals."""
    return sum(_state_value(KuhnState(cards), strategy) for cards in DEALS) / len(DEALS)


def best_response_value(strategy: Mapping[Infoset, np.ndarray], player: int) -> float:
    """Best response utility, using one action per private-rank/public-history infoset.

    Enumerating the 2^6 pure behavioral policies makes information constraints
    explicit. A per-deal maximization would incorrectly grant hidden-card
    knowledge and even report positive exploitability for Nash equilibrium.
    """
    if player not in (0, 1):
        raise ValueError("player must be zero or one")
    infosets = [(rank, history) for rank, history in product(range(3), HISTORIES)
                if _player(history) == player]
    best = -np.inf
    for choices in product((0, 1), repeat=len(infosets)):
        candidate = dict(strategy)
        for key, choice in zip(infosets, choices):
            action = _legal(key[1])[choice]
            candidate[key] = np.eye(5)[action]
        value = game_value(candidate)
        best = max(best, value if player == 0 else -value)
    return float(best)


def nash_conv(strategy: Mapping[Infoset, np.ndarray]) -> float:
    return max(0.0, best_response_value(strategy, 0) + best_response_value(strategy, 1))


def exploitability(strategy: Mapping[Infoset, np.ndarray]) -> float:
    return nash_conv(strategy) / 2


class _AverageStrategy:
    def __init__(self) -> None:
        self.sums = {key: np.zeros(5) for key in uniform_strategy()}

    def add(self, policies: Mapping[int, TabularRegretPolicy], iteration: int) -> None:
        # Average using each player's own reach, not opponent/chance reach.
        # The only nontrivial own reach is P0's check-then-face-bet infoset.
        for rank, history in self.sums:
            player = _player(history)
            reach = policies[0].at((rank, ""))[1] if history == "cb" else 1.0
            self.sums[(rank, history)] += iteration * reach * policies[player].at((rank, history))

    def strategy(self) -> Strategy:
        default = uniform_strategy()
        return {key: value / value.sum() if value.sum() > 0 else default[key]
                for key, value in self.sums.items()}


@dataclass
class KuhnTrainingResult:
    strategy: Strategy
    initial_exploitability: float
    final_exploitability: float
    value: float
    regret_samples: int


def train_kuhn(
    iterations: int = 1000,
    *,
    traversals_per_player: int = 6,
    seed: int = 1,
) -> KuhnTrainingResult:
    """Exercise production vectorized MCCFR with exact linear regret estimates."""
    if iterations < 1 or traversals_per_player < 1:
        raise ValueError("iterations and traversals_per_player must be positive")
    policies = {player: TabularRegretPolicy() for player in (0, 1)}
    memories = {player: _IterationRegrets(policies[player]) for player in (0, 1)}
    factory = KuhnRootFactory(seed)
    sampler = ExternalSamplingTraverser(seed=seed + 1, batch_size=traversals_per_player)
    average = _AverageStrategy()
    initial = exploitability(uniform_strategy())
    for iteration in range(1, iterations + 1):
        average.add(policies, iteration)
        # Keep the policies fixed throughout this iteration, as in the neural
        # collect-then-regress loop. Apply both players' signed targets later.
        for player in (0, 1):
            sampler.collect([factory() for _ in range(traversals_per_player)],
                            traverser=player, iteration=iteration,
                            policies=policies, buffer=memories[player])
        for memory in memories.values():
            memory.apply()
    strategy = average.strategy()
    return KuhnTrainingResult(strategy, initial, exploitability(strategy), game_value(strategy),
                              sum(memory.samples for memory in memories.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--traversals-per-player", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    result = train_kuhn(args.iterations, traversals_per_player=args.traversals_per_player,
                        seed=args.seed)
    print(json.dumps({"iterations": args.iterations,
                      "initial_exploitability": result.initial_exploitability,
                      "final_exploitability": result.final_exploitability,
                      "nash_conv": 2 * result.final_exploitability,
                      "player_zero_value": result.value,
                      "equilibrium_value": -1 / 18,
                      "regret_samples": result.regret_samples}, indent=2))


if __name__ == "__main__":
    main()

"""In-process opponent pool for self-play.

Sampling is prioritized fictitious self-play on a win/tie/loss score, plus
explicit probabilities of playing the current net and a uniform random legal
policy. Checkpoints are full ``state_dict`` copies on CPU. This is not the
Ray league in AlphaNLHoldem and it is not K-best self-play.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


class OpponentPool:
    def __init__(
        self,
        max_size: int,
        selfplay_prob: float,
        random_prob: float,
        sampling: str = "pfsp",
        seed: int = 0,
    ) -> None:
        if max_size < 1:
            raise ValueError("pool max_size must be >= 1")
        if not 0.0 <= selfplay_prob <= 1.0 or not 0.0 <= random_prob <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
        if selfplay_prob + random_prob > 1.0:
            raise ValueError("selfplay_prob + random_prob must be <= 1")
        if sampling not in ("pfsp", "uniform"):
            raise ValueError("sampling must be 'pfsp' or 'uniform'")
        self.max_size = int(max_size)
        self.selfplay_prob = float(selfplay_prob)
        self.random_prob = float(random_prob)
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self._next_id = 0
        self._order: List[int] = []
        self._entries: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._order)

    def add(self, state_dict: Dict[str, torch.Tensor]) -> int:
        policy_id = self._next_id
        self._next_id += 1
        self._entries[policy_id] = {
            "weights": _clone_state_dict(state_dict),
            "wins": 0.0,
            "games": 0.0,
        }
        self._order.append(policy_id)
        while len(self._order) > self.max_size:
            evicted = self._order.pop(0)
            self._entries.pop(evicted, None)
        return policy_id

    def win_rate(self, policy_id: int) -> float:
        entry = self._entries.get(policy_id)
        if entry is None or entry["games"] <= 0:
            return 0.5
        return float(entry["wins"] / entry["games"])

    def sample(self) -> Dict[str, Any]:
        """Return kind ``random``, ``self``, or ``pool``.

        ``policy_id`` is stable across later evictions: ``record`` ignores an
        id that has already been dropped. Weights are the CPU tensors to load
        into a frozen opponent, and are ``None`` unless kind is ``pool``.
        """

        roll = float(self.rng.random())
        if roll < self.random_prob or not self._order:
            return {"kind": "random", "policy_id": None, "weights": None}
        if roll < self.random_prob + self.selfplay_prob:
            return {"kind": "self", "policy_id": None, "weights": None}
        policy_id = self._sample_pool_id()
        return {
            "kind": "pool",
            "policy_id": policy_id,
            "weights": self._entries[policy_id]["weights"],
        }

    def record(self, policy_id: Optional[int], reward_bb: float) -> None:
        if policy_id is None:
            return
        entry = self._entries.get(int(policy_id))
        if entry is None:
            return
        if reward_bb > 0:
            score = 1.0
        elif reward_bb < 0:
            score = 0.0
        else:
            score = 0.5
        entry["games"] += 1.0
        entry["wins"] += score

    def state_dict(self) -> Dict[str, Any]:
        return {
            "next_id": self._next_id,
            "order": list(self._order),
            "entries": {
                pid: {
                    "weights": _clone_state_dict(entry["weights"]),
                    "wins": float(entry["wins"]),
                    "games": float(entry["games"]),
                }
                for pid, entry in self._entries.items()
            },
            "max_size": self.max_size,
            "selfplay_prob": self.selfplay_prob,
            "random_prob": self.random_prob,
            "sampling": self.sampling,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        self._next_id = int(blob["next_id"])
        self._order = [int(pid) for pid in blob["order"]]
        if blob.get("rng") is not None:
            self.rng.bit_generator.state = blob["rng"]
        self._entries = {}
        for pid, entry in blob["entries"].items():
            self._entries[int(pid)] = {
                "weights": _clone_state_dict(entry["weights"]),
                "wins": float(entry["wins"]),
                "games": float(entry["games"]),
            }

    def _sample_pool_id(self) -> int:
        ids = list(self._order)
        if self.sampling == "uniform":
            return int(self.rng.choice(ids))
        weights = []
        for pid in ids:
            rate = min(self.win_rate(pid), 0.95)
            weights.append((1.0 - rate) ** 2)
        total = float(sum(weights))
        if total <= 1e-12:
            return int(self.rng.choice(ids))
        probs = [w / total for w in weights]
        return int(self.rng.choice(ids, p=np.asarray(probs)))

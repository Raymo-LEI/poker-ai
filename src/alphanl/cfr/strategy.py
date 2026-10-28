"""SD-CFR average strategy by trajectory sampling, without a policy net."""

from __future__ import annotations

from dataclasses import asdict
from collections import OrderedDict
import math
import json
from pathlib import Path

import numpy as np
import torch

from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.tokenizer import TensorBatch
from .config import CFRConfig
from .model import RegretNetwork


class SnapshotPolicy:
    """Draw an iteration proportional to t once at the start of a hand.

    Keep that network for *every* decision in the hand. Resampling at each
    infoset, or averaging local probabilities without own-reach weights, is
    not the SD-CFR average strategy. Sparse archives approximate the average
    over retained iterations only; snapshot_every=1 preserves every profile.
    ``through_iteration`` freezes a prefix; ``snapshot_iterations`` selects
    an explicit subset. ``cache_size`` bounds an optional LRU of CPU weights.
    ``sampling_power`` defaults to the historical linear mixture. Power 2
    is a quadratic evaluation intervention; power 0 samples uniformly.
    """

    def __init__(self, directory: str | Path, *, seed: int = 1, device="cpu",
                 through_iteration: int | None = None,
                 snapshot_iterations: list[int] | None = None,
                 cache_size: int = 0, sampling_power: float = 1.0) -> None:
        if not math.isfinite(sampling_power) or sampling_power < 0:
            raise ValueError("sampling_power must be nonnegative and finite")
        self.sampling_power = sampling_power
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        directory = Path(directory)
        index_path = directory / "index.json"
        if index_path.exists():
            archive = json.loads(index_path.read_text())
            if archive.get("format") != "sd_cfr_archive_v1":
                raise ValueError("unsupported strategy archive index")
            paths = [directory / f"iteration_{iteration:07d}.pt"
                     for iteration in archive["iterations"]]
            if any(not path.is_file() for path in paths):
                raise ValueError("strategy archive is missing an indexed snapshot")
        else:
            paths = sorted(directory.glob("iteration_*.pt"))
        if through_iteration is not None:
            if (isinstance(through_iteration, bool) or not isinstance(through_iteration, int)
                    or through_iteration < 1):
                raise ValueError("through_iteration must be positive")
            paths = [p for p in paths if int(p.stem.split("_")[-1]) <= through_iteration]
        if snapshot_iterations is not None:
            selected = set(snapshot_iterations)
            if (not selected or len(selected) != len(snapshot_iterations)
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in selected)):
                raise ValueError("snapshot_iterations must be distinct positive integers")
            paths = [p for p in paths if int(p.stem.split("_")[-1]) in selected]
            if {int(p.stem.split("_")[-1]) for p in paths} != selected:
                raise ValueError("requested strategy snapshot is not in the archive")
        if not paths:
            raise ValueError("no strategy snapshots in directory")
        self.paths = paths
        self.iterations = np.array([int(p.stem.split("_")[-1]) for p in paths], dtype=np.int64)
        if (self.iterations < 1).any() or len(set(self.iterations)) != len(paths):
            raise ValueError("snapshot iterations must be distinct positive integers")
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.model: RegretNetwork | None = None
        self.config: CFRConfig | None = None
        self.hand_iteration: int | None = None
        self.player: int | None = None
        # Cache CPU state dictionaries, not one inference module per iteration.
        # Eviction never changes the sampling distribution or the active hand.
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.disk_loads = 0

    def begin_hand(self, player: int) -> int:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
        weights = self.iterations.astype(np.float64)
        if self.sampling_power != 1.0:
            weights = (weights / weights.max()) ** self.sampling_power
        index = int(self.rng.choice(len(self.paths), p=weights / weights.sum()))
        blob = self._cache.pop(index, None)
        if blob is None:
            blob = torch.load(self.paths[index], map_location="cpu", weights_only=True)
            self.disk_loads += 1
        config = CFRConfig.from_checkpoint(blob["config"])
        if (blob.get("format") not in ("sd_cfr_strategy_v1", "sd_cfr_strategy_v2")
                or blob["iteration"] != int(self.iterations[index])
                or blob["weight"] != blob["iteration"]
                or blob.get("rules_version") != RULES_VERSION
                or blob.get("numeric_version") != config.numeric_version):
            raise ValueError("incompatible SD-CFR strategy snapshot")
        if self.cache_size:
            self._cache[index] = blob
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        if self.model is None:
            self.model = RegretNetwork(PokerTransformer(**asdict(config.model)),
                                       fallback=config.fallback).to(self.device)
            self.config = config
        elif (asdict(config.model) != asdict(self.config.model)
              or any(getattr(config, key) != getattr(self.config, key) for key in (
                  "fallback", "separate_player_models", "position_conditioning", "regret_target_scale",
                  "preflop_action_guard", "guard_short_stack_bb", "guard_unlock_invested_bb", "guard_unlock_raises",
                  "spr_action_guard", "guard_max_pot_ratio", "equity_features", "equity_samples", "equity_seed"))):
            raise ValueError("snapshots use different network configurations")
        self.model.load_state_dict(blob["models"][player if config.separate_player_models else "shared"], strict=True)
        self.model.eval()
        self.hand_iteration = int(blob["iteration"])
        self.player = player
        return self.hand_iteration

    @torch.no_grad()
    def strategy(self, batch: TensorBatch) -> torch.Tensor:
        if self.model is None or self.hand_iteration is None:
            raise RuntimeError("call begin_hand(player) before acting")
        return self.model.strategy({key: value.to(self.device) for key, value in batch.items()})

    @torch.no_grad()
    def act(self, batch: TensorBatch) -> torch.Tensor:
        probs = self.strategy(batch)
        rows = probs.cpu().double().numpy()
        return torch.tensor([self.rng.choice(len(row), p=row / row.sum()) for row in rows],
                            dtype=torch.long, device=self.device)

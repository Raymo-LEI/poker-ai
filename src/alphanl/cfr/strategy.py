"""SD-CFR average strategy by trajectory sampling, without a policy net."""

from __future__ import annotations

from dataclasses import asdict
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
    """

    def __init__(self, directory: str | Path, *, seed: int = 1, device="cpu") -> None:
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

    def begin_hand(self, player: int) -> int:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
        weights = self.iterations.astype(np.float64)
        index = int(self.rng.choice(len(self.paths), p=weights / weights.sum()))
        blob = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        if (blob.get("format") != "sd_cfr_strategy_v1"
                or blob["iteration"] != int(self.iterations[index])
                or blob["weight"] != blob["iteration"]
                or blob.get("rules_version") != RULES_VERSION
                or blob.get("numeric_version") != NUMERIC_VERSION):
            raise ValueError("incompatible SD-CFR strategy snapshot")
        config = CFRConfig(**blob["config"])
        if self.model is None:
            self.model = RegretNetwork(PokerTransformer(**asdict(config.model)),
                                       fallback=config.fallback).to(self.device)
            self.config = config
        elif (asdict(config.model) != asdict(self.config.model)
              or config.fallback != self.config.fallback):
            raise ValueError("snapshots use different network configurations")
        self.model.load_state_dict(blob["models"][player], strict=True)
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

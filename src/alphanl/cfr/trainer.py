"""Position-conditioned external-sampling SD-CFR with a shared regret model.

No average-policy network, PPO modification, or synthetic terminal values.
The fitted network estimates a weighted average of signed sampled regrets;
its common positive scale cancels under regret matching. This is not the
tabular CFR+ operation that clips a running cumulative regret after updates.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable
import time
from uuid import uuid4

import numpy as np
import torch

from alphanl.transformer.config import resolve_device
from alphanl.transformer.model import PokerTransformer
from .config import CFRConfig
from .checkpoint import (restore_buffer, save_buffer, _references, publish_references,
                         prune_generations, sync_directory, checkpoint_lock)
from .model import RegretNetwork
from .reservoir import ReservoirBuffer, UnifiedReservoir, SeatReservoirWriter
from .actions import CFRActionGuard
from .retention import archive_due, archive_checkpoint
from .weighting import polynomial_weights, fit_learning_rate
from .traversal import ExternalSamplingTraverser, GameState, HunlState


def iteration_weighted_mse(predicted: torch.Tensor, target: torch.Tensor,
                           legal_mask: torch.Tensor, iterations: torch.Tensor, *,
                           power: float = 1.0, reference: float = 1.0) -> torch.Tensor:
    """E[(t/reference)**power * sum_legal (prediction-target)^2].

    Do not normalize weights by a random minibatch mean (a batch of size one
    would lose all iteration weighting). Do not divide rows by action count:
    each sampled infoset carries the same iteration weight, as in the
    Deep CFR advantage objective.
    """
    if predicted.shape != target.shape or predicted.shape != legal_mask.shape:
        raise ValueError("predictions, targets and legal mask must have the same shape")
    if iterations.shape != predicted.shape[:-1] or (iterations < 1).any():
        raise ValueError("one positive iteration weight is required per sample")
    legal = legal_mask > 0.5
    if not legal.any(-1).all():
        raise ValueError("regret training requires a legal action in every row")
    weights = polynomial_weights(iterations.to(predicted.device), power=power, reference=reference)
    errors = torch.where(legal, predicted.float() - target.float(), 0.0)
    return (weights * errors.square().sum(-1)).mean()


def iteration_weighted_huber(predicted, target, legal_mask, iterations, *, beta=1.0, power=1.0, reference=1.0):
    """FP32 polynomial-iteration weighted Smooth L1 over legal scaled targets."""
    if predicted.shape != target.shape or predicted.shape != legal_mask.shape:
        raise ValueError("predictions, targets and legal mask must have the same shape")
    if iterations.shape != predicted.shape[:-1] or (iterations < 1).any():
        raise ValueError("one positive iteration weight is required per sample")
    legal = legal_mask > .5
    if not legal.any(-1).all():
        raise ValueError("regret training requires a legal action in every row")
    errors = torch.where(legal, predicted.float() - target.float(), 0.)
    losses = torch.nn.functional.smooth_l1_loss(errors, torch.zeros_like(errors), reduction="none", beta=beta)
    weights = polynomial_weights(iterations.to(predicted.device), power=power, reference=reference)
    return (weights * losses.sum(-1)).mean()


def _atomic_save(payload: dict, path: Path, *, after_replace=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        # Full RAM reservoirs can contain NumPy byte strings over 4 GiB.
        # Strategy snapshots keep the weights-only-compatible default protocol.
        protocol = 4 if payload.get("format") in ("sd_cfr_train_v2", "sd_cfr_train_v3") else 2
        torch.save(payload, temp, pickle_protocol=protocol)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        temp.replace(path)
        if after_replace is not None:
            after_replace()
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _link_or_copy(source: Path, target: Path, *, reserve_bytes: int = 0) -> None:
    """Immutable strategy files can share disk blocks with saved generations."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".{uuid4().hex}.tmp")
    try:
        os.link(source, temp)
    except OSError:
        if shutil.disk_usage(target.parent).free < source.stat().st_size + reserve_bytes:
            raise OSError("insufficient space for checkpoint strategy copy and reserve")
        shutil.copyfile(source, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    temp.replace(target)


def _append_csv(path: Path, row: dict) -> None:
    """Upgrade older log headers when new performance counters are added."""
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            extra = [key for key in row if key not in fields]
            if extra:
                history = list(reader)
        if extra:
            fields += extra
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                with temporary.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(history)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
    else:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)


class SDCFRTrainer:
    def __init__(self, config: CFRConfig, *,
                 root_factory: Callable[[int], GameState] | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.device = torch.device(resolve_device(config.device))
        self.amp_enabled = self.device.type == "cuda" and config.amp_dtype == "bfloat16"
        self.optimizer_fused = self.device.type == "cuda" and config.fused_optimizer
        if self.amp_enabled and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 autocast was requested, but this CUDA device does not support BF16")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.iteration = 0
        self.last_timing: dict[str, float | int] = {}
        self.last_fit_timing: dict[str, float | int] = {}
        self.gpu_pool_enabled = self.device.type == "cuda" and config.train_buffer_device == "cuda"
        self.model_keys = (0, 1) if config.separate_player_models else ("shared",)
        self.gpu_sample_generators = {}
        if self.gpu_pool_enabled:
            self.gpu_sample_generators = {
                p: torch.Generator(device=self.device).manual_seed(config.seed + 3000 + i)
                for i, p in enumerate(self.model_keys)
            }
        self.strategy_paths: dict[int, Path] = {}
        self.models = {p: self._new_model() for p in self.model_keys}
        self.optimizers = {p: self._new_optimizer(p) for p in self.model_keys}
        self.buffers = {
            p: (ReservoirBuffer if config.separate_player_models else UnifiedReservoir)(
                               config.buffer_capacity * (1 if config.separate_player_models else 2),
                               config.model.max_seq_len, seed=config.seed + 100 + i,
                               directory=(Path(config.buffer_directory) / (f"player_{p}" if p != "shared" else "shared")
                                          if config.buffer_directory else None))
            for i, p in enumerate(self.model_keys)
        }
        self.traverser = ExternalSamplingTraverser(
            seed=config.seed + 200, batch_size=config.traversal_batch_size,
            max_nodes=config.max_traversal_nodes,
            scripted_opponent_prob=config.scripted_opponent_prob,
            scripted_opponent_pool=config.scripted_opponent_pool,
            scripted_opponent_equity_samples=config.scripted_opponent_equity_samples,
        )
        self.root_factory = root_factory or self._hunl_root
        from alphanl.transformer.hunl import HunlTokenizer
        self._hunl_tokenizer = HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs())
        self.action_guard = CFRActionGuard.from_config(config)
        self.parallel_collector = None
        if config.traversal_workers:
            from .parallel import ParallelCollector
            self.parallel_collector = ParallelCollector(self.config, root_factory)

    @property
    def shared_regret_net(self):
        return self.models.get("shared")

    @property
    def shared_reservoir(self):
        return self.buffers.get("shared")

    @property
    def policies(self):
        return self.models if self.config.separate_player_models else {0: self.shared_regret_net, 1: self.shared_regret_net}

    def _new_model(self) -> RegretNetwork:
        backbone = PokerTransformer(**asdict(self.config.model))
        return RegretNetwork(backbone, fallback=self.config.fallback).to(self.device)

    def _new_optimizer(self, player: int):
        return torch.optim.Adam(
            (p for p in self.models[player].parameters() if p.requires_grad),
            lr=self.config.learning_rate,
            fused=self.optimizer_fused,
        )

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _hunl_root(self, seed: int) -> HunlState:
        from alphanl.transformer.hunl import HunlTable
        table = HunlTable(seed=seed)
        table.reset()
        return HunlState(table, self._hunl_tokenizer, self.action_guard)

    def _snapshot(self, iteration: int) -> dict:
        from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
        return {
            "format": "sd_cfr_strategy_v2", "iteration": iteration,
            "weight": iteration, "config": self.config.to_dict(),
            "rules_version": RULES_VERSION, "numeric_version": self.config.numeric_version,
            "models": {p: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                       for p, model in self.models.items()},
        }

    def _write_strategy_index(self) -> None:
        directory = Path(self.config.out_dir) / "strategies"
        directory.mkdir(parents=True, exist_ok=True)
        index = {"format": "sd_cfr_archive_v1", "completed_iteration": self.iteration,
                 "iterations": sorted(self.strategy_paths)}
        temp = directory / "index.json.tmp"
        temp.write_text(json.dumps(index, indent=2) + "\n")
        temp.replace(directory / "index.json")

    def train_player(self, player: int | str) -> float:
        self.last_fit_timing = {"pool_upload_s": 0.0, "pool_bytes": 0}
        self.last_fit_diagnostics = {}
        if not len(self.buffers[player]):
            # For example, every sampled SB folds before the BB can act.
            # The empty player's uniform/current network remains unchanged.
            return 0.0
        shared = player == "shared"
        buffer = self.buffers[player]
        if shared and any(not n for n in buffer.seat_counts().values()):
            # Never silently violate the 1:1 contract if collection produced
            # no BB decisions. Log the deferral instead of inventing samples.
            self.last_fit_diagnostics = {"fit_deferred_missing_seat": 1}
            return 0.0
        if self.config.reinitialize:
            self.models[player] = self._new_model()
            self.optimizers[player] = self._new_optimizer(player)
        model = self.models[player]
        # As in the PPO path, suppress dropout during deterministic fitting.
        # eval() preserves gradients through the full encoder and regret head.
        model.eval()
        # Independent RNG preserves reservoir draws and exact resume semantics.
        # Both measurements use the SAME retained-data batch, without dropout.
        from .diagnostics import regression_probe, probe_comparison
        probes = {}
        for seat in ((0, 1) if shared else (player,)):
            count = min(1024, buffer.seat_counts()[seat] if shared else len(buffer))
            probes[seat] = buffer.sample(count, self.device, **({"seat": seat} if shared else {}),
                rng=np.random.default_rng(np.random.SeedSequence([self.config.seed, self.iteration + 1, seat, 91827])))
        scale = self.config.regret_target_scale
        weight_reference = self.iteration + 1 if self.config.normalize_iteration_weights else 1.0
        def probe_weights(iterations):
            return polynomial_weights(iterations, power=self.config.iteration_weight_power, reference=weight_reference)
        with torch.no_grad():
            start_probes = {seat: regression_probe(model(probe[0]), probe[1] / scale, probe[0]["legal_mask"], probe_weights(probe[2]))
                            for seat, probe in probes.items()}
        pool = None
        if self.gpu_pool_enabled:
            from .device_pool import DeviceTrainingPool
            self._sync_device()
            began = time.perf_counter()
            pool = DeviceTrainingPool(self.buffers[player], self.device,
                                      reserve_bytes=self.config.gpu_pool_reserve_bytes,
                                      capacity=self.buffers[player].capacity)
            self._sync_device()
            self.last_fit_timing = {"pool_upload_s": time.perf_counter() - began, "pool_bytes": pool.nbytes}
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        try:
            for fit_step in range(self.config.train_steps):
                if pool is not None:
                    batch, target, iterations = pool.sample(self.config.minibatch_size, self.gpu_sample_generators[player])
                else:
                    batch, target, iterations = self.buffers[player].sample(
                        self.config.minibatch_size, device=self.device
                    )
                if self.amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted = model(batch)
                else:
                    predicted = model(batch)
                target = target.float() / scale
                loss = (iteration_weighted_huber(predicted, target, batch["legal_mask"], iterations, beta=self.config.huber_beta,
                                                power=self.config.iteration_weight_power, reference=weight_reference)
                        if self.config.loss == "huber" else
                        iteration_weighted_mse(predicted, target, batch["legal_mask"], iterations,
                                              power=self.config.iteration_weight_power, reference=weight_reference))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite regret regression loss")
                optimizer = self.optimizers[player]
                for group in optimizer.param_groups:
                    group["lr"] = fit_learning_rate(self.config, fit_step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    self.config.max_grad_norm, error_if_nonfinite=True,
                )
                optimizer.step()
                total.add_(loss.detach().to(total.dtype))
            with torch.no_grad():
                for seat, probe in probes.items():
                    end = regression_probe(model(probe[0]), probe[1] / scale, probe[0]["legal_mask"], probe_weights(probe[2]))
                    values = probe_comparison(start_probes[seat], end)
                    values["probe_samples"] = len(probe[1])
                    values["fit_acceptance"] = int(values["probe_r2"] > .5 and values["probe_loss_ratio"] < .3)
                    self.last_fit_diagnostics.update({(f"p{seat}_{k}" if shared else k): v for k, v in values.items()})
            self.last_fit_diagnostics.update(iteration_weight_power=self.config.iteration_weight_power,
                iteration_weight_reference=weight_reference, learning_rate_first=fit_learning_rate(self.config, 0),
                learning_rate_last=fit_learning_rate(self.config, self.config.train_steps - 1))
            return float((total / self.config.train_steps).cpu())
        finally:
            if pool is not None:
                pool.close()
                del pool
                if self.device.type == "cuda":
                    # Return unequally-sized cache blocks to the driver when a
                    # prior build staged growing prefixes; fixed-capacity
                    # staging makes this cheap / usually a no-op.
                    torch.cuda.empty_cache()

    def run_iteration(self) -> dict:
        if self.iteration == 0 and any(buffer.seen > 0 for buffer in self.buffers.values()):
            raise RuntimeError("existing regret memory requires load_checkpoint before continuation")
        iteration = self.iteration + 1
        iteration_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        # sigma_t comes from theta_(t-1). Save exactly the profile used to
        # generate t's regrets, including uniform sigma_1 (zero regret heads).
        snapshot = self._snapshot(iteration)
        metrics = {"iteration": iteration}
        timing: dict[str, float | int] = {"iteration": iteration}
        for player in (0, 1):
            self._sync_device()
            collect_started = time.perf_counter()
            seeds = self.rng.integers(0, 2**31 - 1, self.config.traversals_per_player)
            source = seeds if self.parallel_collector else (self.root_factory(int(seed)) for seed in seeds)
            collector = self.parallel_collector or self.traverser
            stats = collector.collect(
                source, traverser=player, iteration=iteration,
                policies=self.policies, buffer=(self.buffers[player] if self.config.separate_player_models
                                               else SeatReservoirWriter(self.shared_reservoir, player)),
            )
            self._sync_device()
            collect_seconds = time.perf_counter() - collect_started
            metrics.update({f"p{player}_{k}": v for k, v in asdict(stats).items()})
            metrics[f"p{player}_buffer_size"] = (len(self.buffers[player]) if self.config.separate_player_models
                                                else self.shared_reservoir.seat_counts()[player])
            metrics[f"p{player}_samples_per_root"] = stats.regret_samples / self.config.traversals_per_player
            timing[f"p{player}_collect_s"] = collect_seconds
            timing.update({f"p{player}_{key}": value
                           for key, value in collector.last_timing.items()})
            timing[f"p{player}_regret_samples_per_s"] = (
                stats.regret_samples / collect_seconds if collect_seconds > 0 else 0.0
            )
        # Simultaneous CFR: neither player's policy changes during collection.
        for player in self.model_keys:
            self._sync_device()
            fit_started = time.perf_counter()
            prefix = "shared" if player == "shared" else f"p{player}"
            metrics[f"{prefix}_loss"] = self.train_player(player)
            metrics.update({(key if player == "shared" else f"p{player}_{key}"): value
                            for key, value in self.last_fit_diagnostics.items()})
            self._sync_device()
            fit_seconds = time.perf_counter() - fit_started
            metrics[f"{prefix}_train_steps"] = (
                self.config.train_steps if len(self.buffers[player]) and not self.last_fit_diagnostics.get("fit_deferred_missing_seat") else 0
            )
            draws = metrics[f"{prefix}_train_steps"] * self.config.minibatch_size
            timing[f"{prefix}_fit_s"] = fit_seconds
            timing.update({f"{prefix}_{key}": value for key, value in self.last_fit_timing.items()})
            timing[f"{prefix}_training_draws_per_s"] = draws / fit_seconds if fit_seconds > 0 else 0.0
            if player == "shared":
                counts = self.shared_reservoir.seat_counts()
                metrics["shared_buffer_size"] = len(self.shared_reservoir)
                for seat in (0, 1):
                    metrics[f"p{seat}_buffer_size"] = counts[seat]
                    metrics[f"p{seat}_buffer_fraction"] = counts[seat] / max(1, len(self.shared_reservoir))
                    metrics[f"p{seat}_training_draws"] = draws // 2
                    # Compatibility labels refer to the SAME shared fit.
                    metrics[f"p{seat}_loss"] = metrics["shared_loss"]
                    metrics[f"p{seat}_train_steps"] = metrics["shared_train_steps"]
                metrics["seat_balanced_sampling"] = 1
        self.iteration = iteration
        from .study import memory_usage
        metrics.update(memory_usage())
        out = Path(self.config.out_dir)
        if iteration % self.config.snapshot_every == 0 or iteration == 1:
            path = out / "strategies" / f"iteration_{iteration:07d}.pt"
            _atomic_save(snapshot, path)
            self.strategy_paths[iteration] = path
        self._write_strategy_index()
        out.mkdir(parents=True, exist_ok=True)
        _append_csv(out / "metrics.csv", metrics)
        timing["iteration_s"] = time.perf_counter() - iteration_started
        self.last_timing = timing
        timing_path = out / "performance.csv"
        _append_csv(timing_path, timing)
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path).resolve()
        with checkpoint_lock(path.parent):
            self._save_checkpoint(path)

    def _save_checkpoint(self, path: Path) -> None:
        """Restart state plus immutable sibling checkpoint_data generation.

        Disk reservoirs are copied in bounded chunks, and past strategies are
        hardlinked (copied across filesystems). Keep the sibling data directory
        with train.pt. Mutable live memmaps never back an older checkpoint.
        """
        path = Path(path).resolve()
        checkpoint_started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Old + new generations coexist until atomic publication. Refuse
        # early rather than consuming the filesystem's final free blocks.
        retained_bytes = sum(len(b) * sum(dtype.itemsize * int(np.prod(tail))
                             for dtype, tail in b._schema.values()) for b in self.buffers.values())
        model_bytes = sum(t.numel() * t.element_size() for m in self.models.values()
                          for t in m.state_dict().values())
        archive_bytes = sum(p.stat().st_size for p in self.strategy_paths.values()
                            if p.stat().st_dev != path.parent.stat().st_dev)
        required = retained_bytes + 4 * model_bytes + archive_bytes + self.config.checkpoint_min_free_bytes
        free_before = shutil.disk_usage(path.parent).free
        if free_before < required:
            raise OSError(f"checkpoint requires {required} free bytes including reserve; previous checkpoint retained")
        generation = path.parent / "checkpoint_data" / f"iteration_{self.iteration:07d}_{uuid4().hex}"
        generation.mkdir(parents=True)
        published = False

        def mark_published():
            nonlocal published
            published = True

        try:
            buffers = {}
            for player, buffer in self.buffers.items():
                saved = save_buffer(buffer, generation / f"player_{player}")
                if saved.get("storage") == "memmap_snapshot_v1":
                    saved["directory"] = str(Path(saved["directory"]).relative_to(path.parent))
                buffers[player] = saved
            strategies = {}
            remaining_payload_bytes = sum(
                len(b) * sum(dtype.itemsize * int(np.prod(tail)) for dtype, tail in b._schema.values())
                for b in self.buffers.values() if b.directory is None)
            for iteration, source in self.strategy_paths.items():
                target = generation / "strategies" / source.name
                _link_or_copy(source, target, reserve_bytes=self.config.checkpoint_min_free_bytes
                              + remaining_payload_bytes + 4 * model_bytes)
                strategies[iteration] = str(target.relative_to(path.parent))
            metrics_source = Path(self.config.out_dir) / "metrics.csv"
            metrics_path = None
            if metrics_source.exists():
                target = generation / "metrics.csv"
                shutil.copyfile(metrics_source, target)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                metrics_path = str(target.relative_to(path.parent))
            profile_source = Path(self.config.out_dir) / "performance.csv"
            profile_path = None
            if profile_source.exists():
                target = generation / "performance.csv"
                shutil.copyfile(profile_source, target)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
                profile_path = str(target.relative_to(path.parent))
            payload = self._snapshot(self.iteration + 1)
            payload.update({
                "format": "sd_cfr_train_v3", "completed_iteration": self.iteration,
                "optimizers": {p: opt.state_dict() for p, opt in self.optimizers.items()},
                "buffers": buffers, "strategy_archive": strategies,
                "metrics_path": metrics_path, "performance_path": profile_path,
                "rng": self.rng.bit_generator.state,
                "traverser_rng": self.traverser.rng.bit_generator.state,
                "worker_rng": self.parallel_collector.rng_states if self.parallel_collector else None,
                "gpu_sample_rng": {p: g.get_state() for p, g in self.gpu_sample_generators.items()},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            })
            _references(payload, path.parent)
            # Ensure external files/directories reach disk before publishing
            # the checkpoint that makes earlier generations collectible.
            for directory in (generation / "strategies", generation, generation.parent):
                if directory.is_dir():
                    sync_directory(directory)
            _atomic_save(payload, path, after_replace=mark_published)
        except BaseException:
            if not published:
                shutil.rmtree(generation)
            raise
        publish_references(path, payload)
        prune_generations(path)
        if archive_due(self.config, self.iteration):
            capacity_bytes = sum(b.capacity * sum(dtype.itemsize * int(np.prod(tail))
                                 for dtype, tail in b._schema.values()) for b in self.buffers.values())
            archive_checkpoint(path, payload, iteration=self.iteration,
                remaining=max(0, (self.config.iterations - self.iteration) // self.config.checkpoint_archive_every)
                          + self.config.checkpoint_archive_extra_count,
                min_free_bytes=self.config.checkpoint_archive_min_free_bytes,
                growth_factor=max(1., capacity_bytes / max(1, retained_bytes),
                                  self.config.iterations / max(1, self.iteration)))
        self.last_checkpoint_timing = {
            "iteration": self.iteration, "checkpoint_s": time.perf_counter() - checkpoint_started,
            "retained_buffer_bytes": retained_bytes, "free_before_bytes": free_before,
            "free_after_bytes": shutil.disk_usage(path.parent).free,
        }
        with (path.parent / "checkpoint_performance.jsonl").open("a") as handle:
            handle.write(json.dumps(self.last_checkpoint_timing) + "\n")

    def load_checkpoint(self, path: str | Path, *, branch_train_steps: bool = False) -> None:
        # Only load locally trusted training files (contains reservoir arrays).
        path = Path(path).resolve()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") not in ("sd_cfr_train_v2", "sd_cfr_train_v3"):
            raise ValueError("expected an SD-CFR training checkpoint")
        from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
        if (blob.get("rules_version") != RULES_VERSION
                or blob.get("numeric_version") != self.config.numeric_version):
            raise ValueError("CFR checkpoint rules or numeric feature version mismatch")
        previous = CFRConfig.from_checkpoint(blob["config"]).to_dict()
        current = self.config.to_dict()
        if branch_train_steps:
            destination = Path(self.config.out_dir).resolve()
            if destination == path.parent or destination == Path(previous["out_dir"]).resolve():
                raise ValueError("training-budget branch requires a new output directory")
            if any((destination / name).exists() for name in ("metrics.csv", "train.pt", "strategies")):
                raise ValueError("training-budget branch output already contains a run")
            if self.config.buffer_directory and any(b.seen for b in self.buffers.values()):
                raise ValueError("training-budget branch requires an empty destination reservoir")
        legacy_config_defaults = {"amp_dtype": "none", "fused_optimizer": False, "allow_tf32": False,
                                  "traversal_workers": 0, "train_buffer_device": "cpu"}
        for key in current:
            if branch_train_steps and key == "train_steps":
                continue
            if key not in ("iterations", "out_dir", "device", "buffer_directory", "checkpoint_every",
                           "checkpoint_min_free_bytes", "gpu_pool_reserve_bytes", "traversal_workers",
                           "train_buffer_device", "traversal_batch_size"):
                if key.startswith("checkpoint_archive_"):
                    continue  # Operational retention policy does not change training semantics.
                if key in ("inference_batch_size", "inference_wait_ms", "worker_sample_batch", "worker_timeout_s"):
                    continue
                previous_value = previous.get(key, legacy_config_defaults.get(key))
                if key not in previous and key not in legacy_config_defaults:
                    raise ValueError(f"CFR checkpoint config is missing: {key}")
                if previous_value != current[key]:
                    raise ValueError(f"CFR checkpoint config mismatch: {key}")
        completed = int(blob["completed_iteration"])
        expected = [t for t in range(1, completed + 1)
                    if t == 1 or t % self.config.snapshot_every == 0]
        if sorted(blob["strategy_archive"]) != expected:
            raise ValueError("checkpoint is missing historical strategy profiles")
        for filename in blob["strategy_archive"].values():
            if not (path.parent / filename).is_file():
                raise ValueError("checkpoint strategy data is missing; keep checkpoint_data with train.pt")
        for player in self.model_keys:
            self.models[player].load_state_dict(blob["models"][player], strict=True)
            self.optimizers[player].load_state_dict(blob["optimizers"][player])
            buffer_state = blob["buffers"][player]
            if buffer_state.get("storage") == "memmap_snapshot_v1":
                buffer_state = dict(buffer_state)
                buffer_state["directory"] = str(path.parent / buffer_state["directory"])
            restore_buffer(self.buffers[player], buffer_state)
        self.iteration = completed
        self.rng.bit_generator.state = blob["rng"]
        self.traverser.rng.bit_generator.state = blob["traverser_rng"]
        if self.parallel_collector:
            states = blob.get("worker_rng")
            if states is not None and len(states) == self.config.traversal_workers:
                self.parallel_collector.rng_states = copy.deepcopy(states)
            else:
                warnings.warn("collector mode/count changed on resume; initializing deterministic worker RNG streams", RuntimeWarning)
                self.parallel_collector.rng_states = [
                    np.random.default_rng(np.random.SeedSequence([self.config.seed, completed, i, 1000])).bit_generator.state
                    for i in range(self.config.traversal_workers)]
        if self.gpu_pool_enabled:
            states = blob.get("gpu_sample_rng", {})
            if set(states) == set(self.model_keys):
                for player, generator in self.gpu_sample_generators.items():
                    generator.set_state(states[player])
            else:
                warnings.warn("GPU sampling enabled on resume; initializing deterministic GPU sampler RNG streams", RuntimeWarning)
                for i, generator in enumerate(self.gpu_sample_generators.values()):
                    generator.manual_seed(self.config.seed + 3000 + i + completed * 100003)
        torch.set_rng_state(blob["torch_rng"])
        if blob["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        # Publish only profiles belonging to the restored history, and carry
        # that complete history when resuming into a different output folder.
        self.strategy_paths = {}
        for iteration, filename in blob["strategy_archive"].items():
            source = path.parent / filename
            target = Path(self.config.out_dir) / "strategies" / source.name
            if branch_train_steps:
                # Legacy trusted v2 runs wrote protocol-4 strategy files while
                # fixing >4 GiB RAM checkpoints. Torch 2.8's weights-only reader
                # cannot decode those files. Re-encode ONLY this new branch's
                # copies, never loosen the ordinary SnapshotPolicy loader.
                profile = torch.load(source, map_location="cpu", weights_only=False)
                if (profile.get("format") not in ("sd_cfr_strategy_v1", "sd_cfr_strategy_v2")
                        or profile.get("iteration") != iteration
                        or profile.get("weight") != iteration
                        or profile.get("rules_version") != RULES_VERSION
                        or profile.get("numeric_version") != NUMERIC_VERSION):
                    raise ValueError("invalid trusted branch strategy profile")
                _atomic_save(profile, target)
            else:
                _link_or_copy(source, target)
            self.strategy_paths[iteration] = target
        self._write_strategy_index()
        target = Path(self.config.out_dir) / "metrics.csv"
        if target.exists():
            archive = target.parent / "history" / f"metrics_before_resume_{uuid4().hex}.csv"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, archive)
            target.unlink()
        if blob["metrics_path"] is not None:
            shutil.copyfile(path.parent / blob["metrics_path"], target)
        profile_target = Path(self.config.out_dir) / "performance.csv"
        if profile_target.exists():
            archive = profile_target.parent / "history" / f"performance_before_resume_{uuid4().hex}.csv"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(profile_target, archive)
            profile_target.unlink()
        if blob.get("performance_path") is not None:
            shutil.copyfile(path.parent / blob["performance_path"], profile_target)
        if branch_train_steps:
            provenance = Path(self.config.out_dir) / "ops" / "branch.json"
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_text(json.dumps({
                "format": "sd_cfr_training_budget_branch_v1",
                "source_checkpoint": str(path), "completed_iteration": completed,
                "source_config": previous, "branch_config": current,
                "intervention": {"train_steps": {"before": previous["train_steps"],
                                                  "after": current["train_steps"]}},
                "note": "Full model, optimizer, reservoir and RNG restore; inherited strategy prefix retained."
            }, indent=2) + "\n")

    def run(self) -> list[dict]:
        metrics = []
        while self.iteration < self.config.iterations:
            row = self.run_iteration()
            metrics.append(row)
            print(f"iteration={self.iteration} "
                  f"samples={row['p0_regret_samples']}/{row['p1_regret_samples']} "
                  f"loss={row['p0_loss']:.4g}/{row['p1_loss']:.4g}", flush=True)
            if (self.iteration % self.config.checkpoint_every == 0
                    or self.iteration == self.config.iterations
                    or archive_due(self.config, self.iteration)):
                self.save_checkpoint(Path(self.config.out_dir) / "train.pt")
        return metrics

    def close(self) -> None:
        if self.parallel_collector is not None:
            self.parallel_collector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Real spawned workers, metadata IPC, signed regrets and failure cleanup."""

from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from alphanl.cfr.parallel import ParallelCollector
from test_cfr_traversal import _analytic_tree, _RecordingBuffer
from test_cfr_trainer import TinyGame, config
from alphanl.cfr import SDCFRTrainer


def analytic_root(seed):
    return _analytic_tree()[0]


def tiny_root(seed):
    return TinyGame()


class BatchedRecordingBuffer(_RecordingBuffer):
    def add(self, batch, regrets, iteration):
        for row in range(len(regrets)):
            super().add({key: value[row:row + 1] for key, value in batch.items()},
                        regrets[row:row + 1], iteration)


def test_spawn_service_keeps_exact_regrets_and_releases_slots(tmp_path):
    cfg = replace(config(tmp_path), traversal_workers=2, traversal_batch_size=2,
                  inference_batch_size=4, inference_wait_ms=2, worker_sample_batch=2)
    collector = ParallelCollector(cfg, analytic_root)
    _, policies = _analytic_tree()
    buffer = BatchedRecordingBuffer()
    try:
        stats = collector.collect(np.arange(8), traverser=0, iteration=7, policies=policies, buffer=buffer)
        assert stats.nodes == 56 and stats.terminal_nodes == 32
        assert stats.regret_samples == 16
        assert len(buffer.records) == 16
        assert stats.inference_rows == 24
        assert max(policies[0].batch_sizes) <= 4
        for node, regret, iteration in buffer.records:
            assert iteration == 7
            expected = [-6.75, -3.75, 0, 0, 1.25] if node == 0 else [0, 3, 0, -1, 0]
            assert regret[0].tolist() == expected
        names = [s.shm.name for s in collector.slots]
        processes = list(collector.processes)
        assert policies[0].training and policies[1].training
        # Reuse the same processes with a new policy version.
        second = collector.collect(np.arange(3), traverser=0, iteration=8, policies=policies, buffer=buffer)
        assert second.regret_samples == 6
        assert collector.processes == processes
    finally:
        collector.close()
    assert all(not p.is_alive() for p in processes)
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_worker_exception_propagates_and_closes_pool(tmp_path):
    cfg = replace(config(tmp_path), traversal_workers=2, max_traversal_nodes=1)
    collector = ParallelCollector(cfg, analytic_root)
    _, policies = _analytic_tree()
    with pytest.raises(RuntimeError, match="no truncated utility"):
        collector.collect(np.arange(4), traverser=0, iteration=1, policies=policies, buffer=_RecordingBuffer())
    assert not collector.processes and not collector.slots
    assert policies[0].training and policies[1].training


def test_parallel_trainer_checkpoint_restores_worker_streams(tmp_path):
    cfg = replace(config(tmp_path / "run"), traversal_workers=2, worker_sample_batch=1)
    with SDCFRTrainer(cfg, root_factory=tiny_root) as trainer:
        row = trainer.run_iteration()
        assert row["p0_regret_samples"] == row["p1_regret_samples"] == 4
        assert trainer.buffers[0].seen == trainer.buffers[1].seen == 4
        assert len(trainer.buffers[0]) == len(trainer.buffers[1]) == 4
        assert row["p0_train_steps"] == row["p1_train_steps"] == cfg.train_steps
        trainer.save_checkpoint(tmp_path / "train.pt")
        states = trainer.parallel_collector.rng_states
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed")), root_factory=tiny_root) as resumed:
        resumed.load_checkpoint(tmp_path / "train.pt")
        assert resumed.parallel_collector.rng_states == states
        assert resumed.run_iteration()["iteration"] == 2


def test_parallel_rejects_unpicklable_root_factory_before_spawning(tmp_path):
    with pytest.raises(ValueError, match="spawn-picklable"):
        ParallelCollector(replace(config(tmp_path), traversal_workers=1), lambda seed: TinyGame())


def test_single_process_checkpoint_can_migrate_to_spawned_collectors(tmp_path):
    cfg = config(tmp_path / "run")
    with SDCFRTrainer(cfg, root_factory=tiny_root) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(tmp_path / "train.pt")
    target = replace(cfg, out_dir=str(tmp_path / "parallel"), traversal_workers=2, traversal_batch_size=2)
    with SDCFRTrainer(target, root_factory=tiny_root) as resumed:
        with pytest.warns(RuntimeWarning, match="mode/count changed"):
            resumed.load_checkpoint(tmp_path / "train.pt")
        assert resumed.run_iteration()["iteration"] == 2

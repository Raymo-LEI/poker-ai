"""Spawned CPU traversals with one parent-owned inference/training device.

Each worker owns one shared-memory slot until the parent acknowledges its
request. Queues/pipes carry metadata only. The parent is the sole writer of
the global Algorithm-R reservoir; no worker-local reservoirs are combined.
"""

from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
import resource
import sys
from queue import Empty
import time
import traceback

import numpy as np
import torch

from alphanl.transformer.tokenizer import cat_batches, batch_to_device
from .traversal import ExternalSamplingTraverser, HunlState, TraversalStats
from .actions import CFRActionGuard


class _Slot:
    def __init__(self, rows, length, *, name=None):
        specs = {"token_ids": ("int64", (rows, length)),
                 "type_ids": ("int64", (rows, length)),
                 "numeric": ("float32", (rows, length, 4)),
                 "pad_mask": ("bool", (rows, length)),
                 "legal_mask": ("float32", (rows, 5)),
                 "regrets": ("float32", (rows, 5)),
                 "probs": ("float64", (rows, 5))}
        offsets, total = {}, 0
        for key, (dtype, shape) in specs.items():
            total = (total + 7) // 8 * 8
            offsets[key] = total
            total += int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.shm = SharedMemory(create=name is None, size=total, name=name)
        self.arrays = {key: np.ndarray(shape, dtype=dtype, buffer=self.shm.buf, offset=offsets[key])
                       for key, (dtype, shape) in specs.items()}

    def write(self, batch, regrets=None):
        count = len(batch["token_ids"])
        for key, value in batch.items():
            self.arrays[key][:count] = value.detach().numpy()
        if regrets is not None:
            self.arrays["regrets"][:count] = regrets.detach().numpy()
        return count

    def batch(self, count):
        return {key: torch.from_numpy(self.arrays[key][:count]) for key in
                ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")}

    def close(self, *, unlink=False):
        self.arrays.clear()
        self.shm.close()
        if unlink:
            self.shm.unlink()


class _Channel:
    def __init__(self, worker_id, slot, requests, connection, version):
        self.worker_id, self.slot = worker_id, slot
        self.requests, self.connection = requests, connection
        self.version, self.serial = version, 0

    def exchange(self, kind, player, count):
        self.serial += 1
        self.requests.put({"kind": kind, "worker": self.worker_id, "id": self.serial,
                           "version": self.version, "player": player, "count": count})
        response = self.connection.recv()
        if response != (self.version, self.serial):
            raise RuntimeError("stale or mismatched inference acknowledgement")


class _RemotePolicy(torch.nn.Module):
    def __init__(self, channel, player):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.channel, self.player = channel, player

    def strategy(self, batch):
        count = self.channel.slot.write(batch)
        self.channel.exchange("infer", self.player, count)
        return torch.from_numpy(self.channel.slot.arrays["probs"][:count].copy())


class _SampleSink:
    def __init__(self, channel, player, limit):
        self.channel, self.player, self.limit = channel, player, limit
        self.batches, self.regrets = [], []
        self.iteration = None

    def add(self, batch, regrets, iteration):
        self.batches.append(batch)
        self.regrets.append(regrets)
        self.iteration = iteration
        if len(self.batches) >= self.limit:
            self.flush()

    def flush(self):
        if not self.batches:
            return
        count = self.channel.slot.write(cat_batches(self.batches), torch.cat(self.regrets))
        self.channel.exchange("samples", self.player, count)
        self.batches.clear()
        self.regrets.clear()


def _cpu_worker(worker_id, name, rows, length, tree_batch, max_nodes,
                sample_batch, requests, connection, root_factory):
    # This process never receives a model or constructs a CUDA tensor.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    slot = _Slot(rows, length, name=name)
    try:
        while True:
            job = connection.recv()
            if job is None:
                return
            try:
                from alphanl.transformer.hunl import HunlTable, HunlTokenizer
                tokenizer = HunlTokenizer(length, position_conditioning=job["position_conditioning"])
                action_guard = CFRActionGuard(**job["action_guard"])

                def roots():
                    for seed in job["seeds"]:
                        if root_factory is not None:
                            yield root_factory(int(seed))
                        else:
                            table = HunlTable(seed=int(seed))
                            table.reset()
                            yield HunlState(table, tokenizer, action_guard)

                channel = _Channel(worker_id, slot, requests, connection, job["version"])
                sink = _SampleSink(channel, job["player"], sample_batch)
                traverser = ExternalSamplingTraverser(batch_size=tree_batch, max_nodes=max_nodes)
                traverser.rng.bit_generator.state = job["rng"]
                stats = traverser.collect(roots(), traverser=job["player"], iteration=job["iteration"],
                                          policies={p: _RemotePolicy(channel, p) for p in (0, 1)}, buffer=sink)
                sink.flush()
                requests.put({"kind": "done", "worker": worker_id, "version": job["version"],
                              "stats": asdict(stats), "rng": traverser.rng.bit_generator.state,
                              "timing": traverser.last_timing,
                              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                              "cuda_initialized": torch.cuda.is_initialized()})
            except BaseException:
                requests.put({"kind": "error", "worker": worker_id, "version": job["version"],
                              "error": traceback.format_exc()})
    finally:
        slot.close()
        connection.close()


class ParallelCollector:
    """Persistent CPU workers; collect() services frozen policies in the parent.

    Fixed worker RNG streams are saved at iteration barriers. Parallel sample
    arrival order may differ between runs, but each arriving infoset receives
    exactly the global reservoir's uniform retention probability.
    """

    def __init__(self, config, root_factory=None):
        if root_factory is not None:
            try:
                pickle.dumps(root_factory)
            except (TypeError, pickle.PicklingError, AttributeError) as exc:
                raise ValueError("parallel root_factory must be spawn-picklable") from exc
        self.config, self.root_factory = config, root_factory
        self.rng_states = [np.random.default_rng(config.seed + 1000 + i).bit_generator.state
                           for i in range(config.traversal_workers)]
        self.processes, self.slots, self.connections = [], [], []
        self.requests = None
        self.version = 0
        self.last_timing = {}

    def _start(self):
        if self.processes:
            return
        context = mp.get_context("spawn")
        self.requests = context.Queue()
        rows = max(self.config.traversal_batch_size, self.config.worker_sample_batch)
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        old = {key: os.environ.get(key) for key in keys}
        try:
            os.environ.update({key: "1" for key in keys})
            for worker in range(self.config.traversal_workers):
                slot = _Slot(rows, self.config.model.max_seq_len)
                self.slots.append(slot)
                parent, child = context.Pipe()
                self.connections.append(parent)
                process = context.Process(target=_cpu_worker, args=(
                    worker, slot.shm.name, rows, self.config.model.max_seq_len,
                    self.config.traversal_batch_size, self.config.max_traversal_nodes,
                    self.config.worker_sample_batch, self.requests, child, self.root_factory))
                process.start()
                child.close()
                self.processes.append(process)
        except BaseException:
            self.close()
            raise
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _receive(self, timeout):
        try:
            return self.requests.get(timeout=timeout)
        except Empty:
            if any(not p.is_alive() for p in self.processes):
                raise RuntimeError("CPU traversal worker exited unexpectedly")
            return None

    @torch.no_grad()
    def collect(self, seeds, *, traverser, iteration, policies, buffer):
        if traverser not in (0, 1) or iteration < 1:
            raise ValueError("traverser must be 0/1 and iteration must be >= 1")
        if not len(seeds):
            self.last_timing = {}
            return TraversalStats()
        stats, pending, completed = TraversalStats(), [], set()
        timing = {"inference_s": 0.0, "packing_s": 0.0, "queue_wait_s": 0.0,
                  "buffer_insert_s": 0.0, "worker_root_s_sum": 0.0,
                  "worker_observation_s_sum": 0.0, "worker_peak_rss_bytes_sum": 0}
        training = {p: policy.training for p, policy in policies.items()}
        for policy in policies.values():
            policy.eval()
        try:
            self._start()
            self.version += 1
            version = self.version
            workers = min(len(seeds), len(self.processes))
            for worker, chunk in enumerate(np.array_split(seeds, workers)):
                self.connections[worker].send({"seeds": chunk.tolist(), "player": traverser,
                                               "iteration": iteration, "version": version,
                                               "position_conditioning": self.config.position_conditioning,
                                               "action_guard": asdict(CFRActionGuard.from_config(self.config)),
                                               "rng": self.rng_states[worker]})
            last_activity = time.perf_counter()
            while len(completed) < workers:
                wait = self.config.inference_wait_ms / 1000
                remaining = max(0.0, pending[0][0] + wait - time.perf_counter()) if pending else 0.2
                began = time.perf_counter()
                request = self._receive(remaining)
                timing["queue_wait_s"] += time.perf_counter() - began
                if request is not None:
                    last_activity = time.perf_counter()
                    if request["version"] != version:
                        raise RuntimeError("stale traversal policy version")
                    worker = request["worker"]
                    kind = request["kind"]
                    if kind == "error":
                        raise RuntimeError(f"CPU worker {worker}:\n{request['error']}")
                    if kind == "done":
                        if request["cuda_initialized"]:
                            raise RuntimeError("CPU traversal worker initialized CUDA")
                        completed.add(worker)
                        self.rng_states[worker] = request["rng"]
                        timing["worker_root_s_sum"] += request["timing"]["root_s"]
                        timing["worker_observation_s_sum"] += request["timing"]["observation_s"]
                        timing["worker_peak_rss_bytes_sum"] += request["peak_rss_bytes"]
                        for key in ("nodes", "terminal_nodes", "regret_samples"):
                            setattr(stats, key, getattr(stats, key) + request["stats"][key])
                        if self.config.max_traversal_nodes and stats.nodes > self.config.max_traversal_nodes:
                            raise RuntimeError("traversal node budget exceeded; no truncated utility is used")
                    elif kind == "samples":
                        began = time.perf_counter()
                        slot, count = self.slots[worker], request["count"]
                        buffer.add(slot.batch(count), torch.from_numpy(slot.arrays["regrets"][:count]), iteration)
                        timing["buffer_insert_s"] += time.perf_counter() - began
                        self.connections[worker].send((version, request["id"]))
                    elif kind == "infer":
                        pending.append((time.perf_counter(), request))
                    else:
                        raise RuntimeError("unknown worker request")
                elif time.perf_counter() - last_activity > self.config.worker_timeout_s:
                    raise RuntimeError("CPU traversal worker made no progress before worker_timeout_s")
                if not pending:
                    continue
                player = pending[0][1]["player"]
                available = sum(r["count"] for _, r in pending if r["player"] == player)
                if available < self.config.inference_batch_size and time.perf_counter() < pending[0][0] + wait:
                    continue
                selected, rest, count = [], [], 0
                for item in pending:
                    r = item[1]
                    if r["player"] == player and count + r["count"] <= self.config.inference_batch_size:
                        selected.append(r)
                        count += r["count"]
                    else:
                        rest.append(item)
                pending = rest
                began = time.perf_counter()
                cpu_batch = cat_batches([self.slots[r["worker"]].batch(r["count"]) for r in selected])
                policy = policies[player]
                batch = batch_to_device(cpu_batch, next(policy.parameters()).device)
                timing["packing_s"] += time.perf_counter() - began
                began = time.perf_counter()
                probs = policy.strategy(batch).detach().double().cpu().numpy()
                timing["inference_s"] += time.perf_counter() - began
                legal = cpu_batch["legal_mask"].numpy() > 0.5
                if (probs.shape != legal.shape or not np.isfinite(probs).all() or (probs < 0).any()
                        or not np.allclose(probs.sum(-1), 1, atol=1e-6) or (probs[~legal] != 0).any()):
                    raise ValueError("inference service returned invalid strategy probabilities")
                offset = 0
                for r in selected:
                    self.slots[r["worker"]].arrays["probs"][:r["count"]] = probs[offset:offset + r["count"]]
                    offset += r["count"]
                    self.connections[r["worker"]].send((version, r["id"]))
                stats.record_inference(count)
        except BaseException:
            self.close()
            raise
        finally:
            for player, was_training in training.items():
                policies[player].train(was_training)
        self.last_timing = timing
        return stats

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for connection in self.connections:
            connection.close()
        for slot in self.slots:
            slot.close(unlink=True)
        if self.requests is not None:
            self.requests.close()
            self.requests.join_thread()
        self.processes, self.connections, self.slots = [], [], []
        self.requests = None

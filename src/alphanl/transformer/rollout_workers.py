"""T-C2: multiprocess rollout with the 'spawn' start method (Windows-safe).

Each worker owns a contiguous slice of tables (``env.per_env_rng`` required, so a
table's draws do not depend on which process runs it), runs 1 torch thread, and
keeps its own rollout model, opponent pool mirror and ``PolicyActor``.
The main process does the PPO update and talks to workers over one Pipe each:

  init (in Process args) -> "ready" (num_threads, startup)
  ("weights", sd)        -> learner state_dict as numpy arrays, after every update
  ("pool_add", sd)       -> opponent pool snapshot, mirrored in every worker
  ("rollout", stats)     -> pool win/game stats in, (storage numpy, extras) out
  ("get_state",)         -> table state_dicts (for checkpoints)
  ("close",)

Pool results during a rollout are buffered (``pool.defer_record``) and merged by
the main process after the rollout, so PFSP win rates are frozen within one rollout
in both the single-process reference (``pool.defer_record=true``) and here.
Any worker death, pipe EOF, error reply or timeout raises ``RolloutWorkerError``
naming the worker; workers are then terminated.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


class RolloutWorkerError(RuntimeError):
    pass


def split_envs(num_envs: int, workers: int) -> List[List[int]]:
    """Contiguous, as even as possible: first ``num_envs % workers`` slices get one extra."""
    if workers < 1 or workers > num_envs:
        raise ValueError(f"rollout_workers must be in [1, num_envs={num_envs}], got {workers}")
    base, extra = divmod(num_envs, workers)
    out, start = [], 0
    for w in range(workers):
        n = base + (1 if w < extra else 0)
        out.append(list(range(start, start + n)))
        start += n
    return out


def uneven_split_warning(num_envs: int, workers: int) -> Optional[str]:
    """T-C2b: message when envs don't divide evenly over workers (None if they do)."""
    if workers < 1 or num_envs % workers == 0:
        return None
    sizes = sorted({len(s) for s in split_envs(num_envs, workers)}, reverse=True)
    return (f"[rollout_workers] WARNING: num_envs={num_envs} is not divisible by rollout_workers={workers}; "
            f"envs per worker uneven ({'/'.join(map(str, sizes))}); rollout step time is bound by the slowest "
            f"(largest) worker. Prefer num_envs a multiple of rollout_workers.")


def sd_to_numpy(state_dict) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in state_dict.items()}


def sd_from_numpy(blob):
    import torch

    return {k: torch.from_numpy(np.asarray(v).copy()) for k, v in blob.items()}


def _storage_to_numpy(storage) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in storage.items()}


# ----------------------------------------------------------------- worker side
def worker_main(conn, init: Dict[str, Any]) -> None:  # top-level: importable under spawn
    started = time.perf_counter()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        if init.get("deterministic"):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        from . import env as E
        from . import train as T
        from .config import ExperimentConfig
        from .pool import OpponentPool
        from .tokenizer import BettingHistoryTokenizer

        for name, value in init.get("actor_flags", {}).items():
            setattr(E.PolicyActor, name, value)
        cfg = ExperimentConfig.from_dict(init["cfg"])
        indices: List[int] = init["indices"]
        device = torch.device("cpu")
        tokenizer = BettingHistoryTokenizer(max_seq_len=cfg.model.max_seq_len)
        model = T.build_model(cfg).to(device)
        model.load_state_dict(sd_from_numpy(init["weights"]))
        model.eval()
        shadow = T.build_model(cfg).to(device)
        pool = OpponentPool(cfg.pool.max_size, cfg.pool.selfplay_prob, cfg.pool.random_prob,
                            sampling=cfg.pool.sampling, seed=cfg.seed)
        pool.load_state_dict(init["pool"])
        pool.defer_record = True
        actor = E.PolicyActor(model, shadow, tokenizer, np.random.default_rng(cfg.seed + 17))
        seqs = np.random.SeedSequence(int(cfg.seed)).spawn(cfg.ppo.num_envs)
        envs = [
            E.SelfPlayEnv(
                tokenizer, actor, pool, seed=cfg.seed + 1000 + i,
                big_blind=cfg.env.big_blind, small_blind=cfg.env.small_blind,
                starting_stack=cfg.env.starting_stack, backend=cfg.env.backend,
                observer=T.observer_for(cfg), seed_seq=seqs[i], per_env_rng=True,
                shared_deal_stream=False,
            )
            for i in indices
        ]
        if init.get("env_states") is not None:
            for env, st in zip(envs, init["env_states"]):
                env.load_state_dict(st)
            obs = [env.observe() for env in envs]
        else:
            obs = [env.reset() for env in envs]
            for env in envs:
                env.drain_passive_rewards()
        conn.send(("ready", {"num_threads": torch.get_num_threads(),
                             "startup_s": time.perf_counter() - started, "pid": os.getpid()}))
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "close":
                conn.send(("ok", None))
                return
            if cmd == "weights":
                model.load_state_dict(sd_from_numpy(msg[1]))
                conn.send(("ok", None))
            elif cmd == "pool_add":
                pool.add(sd_from_numpy(msg[1]))
                conn.send(("ok", None))
            elif cmd == "rollout":
                pool.set_stats(msg[1])
                t0 = time.perf_counter()
                model.eval()
                storage, stats = T.collect_rollout(envs, obs, model, device, cfg.ppo.rollout_steps)
                extra = {"episodes": stats["episodes"], "passive": stats.get("passive_list", []),
                         "deltas": pool.take_pending(), "compute_s": time.perf_counter() - t0}
                conn.send(("ok", (_storage_to_numpy(storage), extra)))
            elif cmd == "get_state":
                conn.send(("ok", [env.state_dict() for env in envs]))
            elif cmd == "crash":  # test hook
                os._exit(int(msg[1]) if len(msg) > 1 else 3)
            elif cmd == "hang":  # test hook
                time.sleep(1e9)
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        return
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
        raise SystemExit(1)


# ------------------------------------------------------------------ main side
class RolloutWorkers:
    def __init__(self, cfg, weights_sd, pool, *, env_states: Optional[Sequence[Any]] = None,
                 deterministic: Optional[bool] = None, timeout_s: Optional[float] = None,
                 actor_flags: Optional[Dict[str, Any]] = None) -> None:
        import torch

        n = int(cfg.ppo.rollout_workers)
        if not getattr(cfg.env, "per_env_rng", False):
            raise ValueError("ppo.rollout_workers > 0 requires env.per_env_rng: true")
        if getattr(cfg.env, "shared_deal_stream", False):
            raise ValueError("ppo.rollout_workers > 0 is incompatible with env.shared_deal_stream")
        if str(cfg.rollout_device) not in ("cpu", "auto") and torch.device(cfg.rollout_device).type != "cpu":
            raise ValueError("ppo.rollout_workers > 0 runs rollouts on cpu; set rollout_device cpu")
        self.slices = split_envs(int(cfg.ppo.num_envs), n)
        msg = uneven_split_warning(int(cfg.ppo.num_envs), n)
        if msg:
            print(msg, file=sys.stderr, flush=True)
        self.timeout = float(timeout_s if timeout_s is not None else cfg.ppo.rollout_worker_timeout_s)
        if deterministic is None:
            deterministic = torch.are_deterministic_algorithms_enabled()
        ctx = mp.get_context("spawn")
        cfg_dict = cfg.to_dict()
        wnp = sd_to_numpy(weights_sd)
        pool_state = pool.state_dict()
        self.conns, self.procs = [], []
        t0 = time.perf_counter()
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        try:
            for k in _THREAD_VARS:
                os.environ[k] = "1"  # inherited by spawned children before they import torch
            for w, idx in enumerate(self.slices):
                parent, child = ctx.Pipe(duplex=True)
                init = {"cfg": cfg_dict, "indices": idx, "weights": wnp, "pool": pool_state,
                        "deterministic": bool(deterministic), "actor_flags": dict(actor_flags or {}),
                        "env_states": None if env_states is None else [env_states[i] for i in idx]}
                p = ctx.Process(target=worker_main, args=(child, init), name=f"rollout-worker-{w}", daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        try:
            self.ready = [self._recv(w, "ready") for w in range(n)]
        except BaseException:
            self.terminate()
            raise
        self.worker_threads = [r["num_threads"] for r in self.ready]
        self.startup_s = time.perf_counter() - t0
        self.last = {"weight_broadcast_s": 0.0, "pool_sync_s": 0.0, "rollout_total_s": 0.0}

    # -- plumbing
    def _recv(self, w: int, expect: str = "ok"):
        conn, proc = self.conns[w], self.procs[w]
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if conn.poll(0.2):
                    tag, payload = conn.recv()
                    break
            except (EOFError, OSError, ConnectionError) as exc:
                proc.join(1.0)
                self.terminate()
                raise RolloutWorkerError(f"rollout worker {w} (envs {self._span(w)}): pipe closed "
                                         f"(exitcode={proc.exitcode}): {exc!r}") from None
            if not proc.is_alive():
                if conn.poll(0):
                    continue  # drain a final error message
                code = proc.exitcode
                self.terminate()
                raise RolloutWorkerError(f"rollout worker {w} (envs {self._span(w)}) died, exitcode={code}")
            if time.monotonic() > deadline:
                self.terminate()
                raise RolloutWorkerError(f"rollout worker {w} (envs {self._span(w)}) timed out after "
                                         f"{self.timeout:.0f}s waiting for '{expect}'")
        if tag == "error":
            self.terminate()
            raise RolloutWorkerError(f"rollout worker {w} (envs {self._span(w)}) raised:\n{payload}")
        if tag != expect:
            self.terminate()
            raise RolloutWorkerError(f"rollout worker {w}: expected '{expect}', got '{tag}'")
        return payload

    def _span(self, w: int) -> str:
        s = self.slices[w]
        return f"{s[0]}-{s[-1]}" if s else "-"

    def _send_all(self, msg) -> None:
        for w, conn in enumerate(self.conns):
            try:
                conn.send(msg)
            except (BrokenPipeError, EOFError, OSError) as exc:
                code = self.procs[w].exitcode
                self.terminate()
                raise RolloutWorkerError(f"rollout worker {w} (envs {self._span(w)}): send failed "
                                         f"(exitcode={code}): {exc!r}") from None

    def _all(self, msg):
        self._send_all(msg)
        return [self._recv(w) for w in range(len(self.conns))]

    # -- API
    def broadcast_weights(self, state_dict) -> float:
        t = time.perf_counter()
        self._all(("weights", sd_to_numpy(state_dict)))
        self.last["weight_broadcast_s"] = time.perf_counter() - t
        return self.last["weight_broadcast_s"]

    def pool_add(self, state_dict) -> float:
        t = time.perf_counter()
        self._all(("pool_add", sd_to_numpy(state_dict)))
        dt = time.perf_counter() - t
        self.last["pool_sync_s"] += dt
        return dt

    def collect(self, pool):
        """Run one rollout on all workers; returns (storage, stats) like collect_rollout."""
        import torch

        t = time.perf_counter()
        replies = self._all(("rollout", pool.stats()))
        parts = [r[0] for r in replies]
        storage = {k: torch.from_numpy(np.concatenate([p[k] for p in parts], axis=0)) for k in parts[0]}
        passive: List[float] = []
        episodes = 0
        for _, extra in replies:  # worker order == env order
            pool.apply_deltas(extra["deltas"])
            passive.extend(extra["passive"])
            episodes += int(extra["episodes"])
        rewards, dones = storage["rewards"], storage["dones"]
        terminal = dones > 0.5
        stats = {
            "episodes": episodes,
            "mean_reward_bb": float(rewards[terminal].mean()) if int(terminal.sum()) else 0.0,
            "passive_hands": len(passive),
            "passive_mean_bb": float(np.mean(passive)) if passive else 0.0,
            "worker_compute_s": [float(r[1]["compute_s"]) for r in replies],
        }
        self.last["rollout_total_s"] = time.perf_counter() - t
        return storage, stats

    def env_states(self) -> List[Any]:
        out: List[Any] = []
        for part in self._all(("get_state",)):
            out.extend(part)
        return out

    def close(self) -> None:
        for w, conn in enumerate(self.conns):
            try:
                if self.procs[w].is_alive():
                    conn.send(("close",))
                    if conn.poll(5.0):
                        conn.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(5.0)
        self.terminate()

    def terminate(self) -> None:
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join(2.0)
        for c in self.conns:
            try:
                c.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

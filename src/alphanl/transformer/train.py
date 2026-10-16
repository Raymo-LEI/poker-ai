"""Self-play PPO loop. Importing it does not start training or import rlcard."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .checkpoint import (
    ConfigMismatchError,
    assert_resume_config,
    checkpoint_path,
    config_hash,
    latest_checkpoint,
    latest_complete_checkpoint,
    load_checkpoint,
    restore_env_rngs,
    restore_rng_state,
    save_checkpoint,
)
from .config import ExperimentConfig, resolve_device, resolve_env_limits
from .decision_stats import (
    DECISION_STAT_COLUMNS,
    empty_decision_stats,
    policy_action_mask,
    summarize_rollout_decisions,
)
from .env import PolicyActor, SelfPlayEnv, gae_reward_depth, logged_bb_from_reward
from .model import PokerTransformer, count_parameters
from .pool import OpponentPool
from .ppo import GRAD_NORM_COLUMNS, amp_enabled, compute_gae, make_grad_scaler, ppo_update
from .tokenizer import BettingHistoryTokenizer, batch_to_device, cat_batches

_PPO_SKIP = {
    "actions",
    "old_logp",
    "old_values",
    "advantages",
    "returns",
    "rewards",
    "dones",
    "to_call",
}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    if cfg.encoder == "alphanl_conv":
        from .conv_net import AlphaNLConvNet

        return AlphaNLConvNet()
    if cfg.encoder != "transformer":
        raise ValueError("encoder must be 'transformer' or 'alphanl_conv'")
    model_cfg = cfg.model
    return PokerTransformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        max_seq_len=model_cfg.max_seq_len,
    )


def observer_for(cfg: ExperimentConfig):
    """Conv runs see the TF card/history tensors. The transformer keeps tokens."""

    if cfg.encoder == "alphanl_conv":
        from .conv_obs import AlphaNLObserver

        return AlphaNLObserver()
    return None


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=cfg.ppo.adam_eps)


def _sync_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    if src is dst:
        return
    device = next(dst.parameters()).device
    dst.load_state_dict({key: value.detach().to(device) for key, value in src.state_dict().items()})


def _log_decision_stats_enabled() -> bool:
    """Tests set this false to replay the pre-log rollout path.

    The flag skips the to_call read and the reduction only. ``model.act``
    still returns the entropy it already computed, and nothing is resampled.
    """

    return True


def mask_free_fold_enabled(flag: bool) -> bool:
    """Whether this process should mask free folds for the policy.

    Tests monkeypatch this to false to replay the branch from before the
    switch existed. The config field is still recorded on the checkpoint.
    """

    return bool(flag)


def _read_to_call(env: Any) -> Optional[float]:
    """Chips the hero owes right now, or ``None`` if this env cannot say.

    Only a table ``snapshot`` counts. Normalized observation features are
    not decoded: a backend without ``to_call`` must log NaN, not a guess.
    """

    table = getattr(env, "table", None)
    if table is None or not hasattr(table, "snapshot"):
        return None
    hero = getattr(env, "hero", None)
    if hero is None:
        return None
    return float(table.snapshot(int(hero)).to_call)


def _batching(envs: Sequence[Any]) -> bool:
    actor = getattr(envs[0], "actor", None) if envs else None
    if actor is None or not hasattr(actor, "prefetch") or not actor.batching_enabled():
        return False
    return all(getattr(env, "actor", None) is actor and hasattr(env, "begin_step") for env in envs)


def collect_rollout(
    envs: Sequence[Any],
    obs: List[Dict[str, torch.Tensor]],
    model: torch.nn.Module,
    device: torch.device,
    rollout_steps: int,
    *,
    mask_free_fold: bool = False,
) -> tuple:
    """Step each env ``rollout_steps`` hero decisions.

    ``obs`` is updated in place to the states after the rollout. Returns the
    per-env tensors (including advantages' inputs), the next observations, and
    a small stats dict. Does not run the optimizer.
    """

    num_envs = len(envs)
    if len(obs) != num_envs:
        raise ValueError("obs and envs length mismatch")
    sample = obs[0]
    obs_keys = list(sample.keys())
    columns = {
        key: torch.zeros(
            (num_envs, rollout_steps, *sample[key].shape[1:]),
            dtype=sample[key].dtype,
        )
        for key in obs_keys
    }
    actions = torch.zeros(num_envs, rollout_steps, dtype=torch.long)
    old_logp = torch.zeros(num_envs, rollout_steps)
    old_values = torch.zeros(num_envs, rollout_steps)
    rewards = torch.zeros(num_envs, rollout_steps)
    dones = torch.zeros(num_envs, rollout_steps)
    passive_bb: List[float] = []
    episodes = 0
    # Read before begin_step. The reduction uses the sampled action, the mask
    # already stored on the rollout, and the entropy act() already returned.
    log_stats = _log_decision_stats_enabled()
    decision_entropy = torch.zeros(num_envs, rollout_steps) if log_stats else None
    decision_to_call = torch.zeros(num_envs, rollout_steps) if log_stats else None
    to_call_known = True
    # Chips owed at the hero decision, kept so the PPO re-forward can rebuild
    # the same policy mask. Absent when the switch is off, so that path does
    # not add a tensor to storage.
    mask_to_call = (
        torch.full((num_envs, rollout_steps), float("nan"), dtype=torch.float32)
        if mask_free_fold
        else None
    )

    model.eval()
    for t in range(rollout_steps):
        stacked = batch_to_device(cat_batches(obs), device)
        owed_now = None
        if mask_free_fold:
            # Read before the step. ``act`` does not change the table, so this
            # is the chips owed at the decision the policy is about to take.
            owed_now = [_read_to_call(env) for env in envs]
            owed_tensor = torch.tensor(
                [float("nan") if value is None else float(value) for value in owed_now],
                dtype=stacked["legal_mask"].dtype,
                device=stacked["legal_mask"].device,
            )
            stacked = dict(stacked)
            stacked["legal_mask"] = policy_action_mask(stacked["legal_mask"], owed_tensor, True)
            for env_index, value in enumerate(owed_now):
                if value is not None:
                    mask_to_call[env_index, t] = float(value)
        with torch.no_grad():
            act, logp, values, ent = model.act(stacked, deterministic=False)
        act = act.cpu()
        logp = logp.cpu()
        values = values.cpu()
        if log_stats:
            ent = ent.detach().cpu()
            if owed_now is None:
                owed_now = [_read_to_call(env) for env in envs]
        batched = _batching(envs)
        if batched:
            # T-C1b: push every hero action, then one opponent forward per
            # model across envs; sampling stays per env, in env order.
            for e, env in enumerate(envs):
                env.begin_step(int(act[e]))
            envs[0].actor.prefetch(envs)
        for e, env in enumerate(envs):
            for key in obs_keys:
                columns[key][e, t] = obs[e][key][0]
            actions[e, t] = act[e]
            old_logp[e, t] = logp[e]
            old_values[e, t] = values[e]
            if log_stats:
                decision_entropy[e, t] = ent[e]
                if owed_now[e] is None:
                    to_call_known = False
                else:
                    decision_to_call[e, t] = owed_now[e]
            if batched:
                next_obs, reward, done, _info = env.finish_step()
            else:
                next_obs, reward, done, _info = env.step(int(act[e]))
            rewards[e, t] = float(reward)
            dones[e, t] = 1.0 if done else 0.0
            if done:
                episodes += 1
                next_obs = env.reset()
                backend = _env_backend(env)
                passive_bb.extend(
                    logged_bb_from_reward(x, backend) for x in env.drain_passive_rewards()
                )
            obs[e] = next_obs

    bootstrap_batch = batch_to_device(cat_batches(obs), device)
    with torch.no_grad():
        _, bootstrap = model(bootstrap_batch, mask_illegal=True)
    storage = dict(columns)
    storage.update(
        {
            "actions": actions,
            "old_logp": old_logp,
            "old_values": old_values,
            "rewards": rewards,
            "dones": dones,
            "bootstrap": bootstrap.detach().cpu(),
        }
    )
    if mask_to_call is not None:
        storage["to_call"] = mask_to_call
    terminal = dones > 0.5
    if int(terminal.sum()) == 0:
        mean_reward = 0.0
    else:
        # ``rewards`` is the GAE scale. Botzone logs multiply by 200.
        scales = torch.tensor(
            [gae_reward_depth(_env_backend(env)) for env in envs],
            dtype=rewards.dtype,
        ).unsqueeze(1)
        mean_reward = float((rewards * scales)[terminal].mean())
    stats = {
        "episodes": episodes,
        "mean_reward_bb": mean_reward,
        "passive_hands": len(passive_bb),
        "passive_mean_bb": float(np.mean(passive_bb)) if passive_bb else 0.0,
    }
    if log_stats and "legal_mask" in columns:
        stats.update(
            summarize_rollout_decisions(
                actions,
                columns["legal_mask"],
                decision_entropy,
                decision_to_call if to_call_known else None,
            )
        )
    else:
        stats.update(empty_decision_stats())
    return storage, stats


def _env_backend(env: Any) -> str:
    return str(getattr(env, "backend", "") or "")


def flatten_rollout(storage: Dict[str, torch.Tensor], gamma: float, gae_lambda: float) -> Dict[str, torch.Tensor]:
    advantages, returns = compute_gae(
        storage["rewards"],
        storage["old_values"],
        storage["dones"],
        storage["bootstrap"],
        gamma,
        gae_lambda,
    )
    num_envs, steps = storage["rewards"].shape

    def flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(num_envs * steps, *tensor.shape[2:])

    flat_storage = {
        key: flat(tensor) for key, tensor in storage.items() if key != "bootstrap"
    }
    flat_storage["advantages"] = flat(advantages)
    flat_storage["returns"] = flat(returns)
    return flat_storage


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def truncate_metrics_csv(path: Path, start_update: int) -> None:
    """Drop metrics rows whose ``update`` is at or after ``start_update``.

    Rows from the crashed process that sit past the checkpoint are removed
    before the resumed run appends them again. The header and earlier rows
    stay byte-for-byte. The rewrite is atomic (temp file, fsync, replace).
    """

    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return
    lines = text.splitlines(keepends=True)
    header = lines[0]
    try:
        fieldnames = [name.strip() for name in next(csv.reader([header.strip("\r\n")]))]
    except StopIteration:
        return
    if "update" not in fieldnames:
        return
    data_lines = lines[1:]
    parsed_rows = []
    for line in data_lines:
        stripped = line.strip("\r\n")
        if not stripped:
            parsed_rows.append({})
            continue
        parsed_rows.append(next(csv.DictReader([header.strip("\r\n"), stripped])))
    kept = [header if header.endswith(("\n", "\r\n")) else header + "\n"]
    dropped = False
    for line, row in zip(data_lines, parsed_rows):
        if not row:
            piece = line if line.endswith(("\n", "\r\n")) else line + "\n"
            kept.append(piece)
            continue
        raw = row.get("update", "")
        try:
            update = int(str(raw).strip())
        except (TypeError, ValueError):
            piece = line if line.endswith(("\n", "\r\n")) else line + "\n"
            kept.append(piece)
            continue
        if update < int(start_update):
            piece = line if line.endswith(("\n", "\r\n")) else line + "\n"
            kept.append(piece)
        else:
            dropped = True
    if not dropped:
        return
    payload = "".join(kept)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def reset_envs_from_rng_snapshot(
    envs: Sequence[Any],
    actor: Any,
    blob: Dict[str, Any],
) -> List[Dict[str, torch.Tensor]]:
    """Restore RNGs, then deal a new hand on every table.

    The in-progress hand in ``blob["envs"]`` is not loaded. ``--resume auto``
    and the equality-test control both call this, so the next rollout starts
    from the same deal. The rollout net is set to eval first: PPO leaves the
    learner in eval, and a freshly built resume model would otherwise still
    be in train mode.
    """

    rollout = getattr(actor, "rollout_model", None)
    if rollout is not None:
        rollout.eval()
    shadow = getattr(actor, "shadow", None)
    if shadow is not None:
        shadow.eval()
    opponents = getattr(actor, "_opponents", None)
    if isinstance(opponents, dict):
        opponents.clear()
    restore_rng_state(blob.get("rng"))
    actor_rng = blob.get("actor_rng")
    if actor_rng is not None and hasattr(actor, "set_rng_state"):
        actor.set_rng_state(actor_rng)
    restore_env_rngs(envs, blob.get("env_rngs"))
    obs = []
    for env in envs:
        obs.append(env.reset())
        env.drain_passive_rewards()
    return obs


def _make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


@torch.no_grad()
def evaluate_vs_random(
    model: torch.nn.Module,
    tokenizer: BettingHistoryTokenizer,
    pool: OpponentPool,
    cfg: ExperimentConfig,
    device: torch.device,
    n_hands: int,
    seed: int,
) -> float:
    """Greedy hero vs a uniform legal opponent. Returns mean bb/hand."""

    shadow = build_model(cfg).to(device)
    actor = PolicyActor(model, shadow, tokenizer, np.random.default_rng(seed))
    env = SelfPlayEnv(
        tokenizer,
        actor,
        pool,
        seed=seed,
        big_blind=cfg.env.big_blind,
        small_blind=cfg.env.small_blind,
        starting_stack=cfg.env.starting_stack,
        backend=cfg.env.backend,
        record_results=False,
        observer=observer_for(cfg),
        mask_free_fold=mask_free_fold_enabled(cfg.env.mask_free_fold),
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    total = 0.0
    played = 0
    use_mask = mask_free_fold_enabled(cfg.env.mask_free_fold)
    while played < int(n_hands):
        env.reset()
        backend = cfg.env.backend
        passive = [logged_bb_from_reward(reward, backend) for reward in env.drain_passive_rewards()]
        total += sum(passive)
        played += len(passive)
        if played >= int(n_hands):
            break
        done = False
        reward = 0.0
        while not done:
            batch = batch_to_device(env.observe(), device)
            if use_mask:
                owed = _read_to_call(env)
                batch = dict(batch)
                batch["legal_mask"] = policy_action_mask(
                    batch["legal_mask"],
                    float("nan") if owed is None else float(owed),
                    True,
                )
            actions, _, _, _ = model.act(batch, deterministic=True)
            _, reward, done, _info = env.step(int(actions.item()))
        total += logged_bb_from_reward(reward, backend)
        played += 1
    return total / float(played)


def refuse_botzone_init_weights(
    backend: str,
    init_weights: Optional[Path | str],
    allow_cross_backend_init: bool = False,
) -> None:
    """Reject ``--init-weights`` on the botzone backend unless opted in.

    The five policy ids on botzone are not the rlcard 50bb sizes. Loading a
    behavior-cloning checkpoint into that head is allowed only when
    ``allow_cross_backend_init`` is true (the ``--allow-cross-backend-init``
    flag). rlcard and pokerkit are unchanged.
    """

    if init_weights is None or allow_cross_backend_init:
        return
    if str(backend) == "botzone":
        raise ValueError(
            "refusing --init-weights when env.backend is 'botzone'. "
            "Botzone action ids are not the rlcard 50bb sizes. "
            "Pass --allow-cross-backend-init to load those weights anyway."
        )


def _resolve_resume(resume: Optional[Path | str], ckpt_dir: Path) -> tuple:
    """Return ``(mode, path)``. ``path`` is ``None`` when this run starts fresh.

    ``latest`` is the newest filename and is not opened here. ``auto`` skips
    unreadable files and is ``None`` when the directory has no complete
    checkpoint. A filesystem path is returned as-is.
    """

    if resume is None:
        return None, None
    label = str(resume)
    if label == "auto":
        path = latest_complete_checkpoint(ckpt_dir)
        if path is None:
            print(f"[resume] auto: no complete checkpoint in {ckpt_dir}, starting fresh")
        else:
            print(f"[resume] auto: {path}")
        return "auto", path
    if label == "latest":
        path = latest_checkpoint(ckpt_dir)
        print(f"[resume] {'starting fresh, no checkpoint in ' + str(ckpt_dir) if path is None else path}")
        return "latest", path
    return "path", Path(resume)


def run_training(
    cfg: ExperimentConfig,
    *,
    resume: Optional[Path | str] = None,
    init_weights: Optional[Path | str] = None,
    allow_cross_backend_init: bool = False,
    allow_config_mismatch: bool = False,
    reset_envs_after_update: Optional[int] = None,
    stop_after_update: Optional[int] = None,
) -> Path:
    """Run ``cfg.train.num_updates`` PPO updates. Returns the output directory.

    ``resume`` is a checkpoint path, ``"latest"`` (newest ``update_*.pt``,
    fresh if the directory is empty), or ``"auto"`` (newest file that
    actually loads; corrupt files are skipped; fresh if none load).
    ``"auto"`` refuses when ``config_hash`` differs unless
    ``allow_config_mismatch`` is set. It restores RNGs and the opponent
    pool, then resets every table instead of loading the mid-hand.
    ``"latest"`` and a path still restore the mid-hand when the checkpoint
    has one.

    ``init_weights`` loads model weights only. It is refused when
    ``cfg.env.backend`` is ``botzone`` unless ``allow_cross_backend_init``.

    ``reset_envs_after_update`` and ``stop_after_update`` are test seams.
    Both default to off. The first reruns the auto-resume reset after that
    update's checkpoint so an in-process control matches ``--resume auto``.
    The second returns after that update without the usual final-save, which
    is how tests stand in for a kill between updates.
    """

    # A botzone config that still carries the rlcard 1/2/100 defaults is
    # rewritten to 50/100/20000 before tables exist and before the checkpoint
    # stores ``config``.
    small_blind, big_blind, starting_stack = resolve_env_limits(
        cfg.env.backend, cfg.env.small_blind, cfg.env.big_blind, cfg.env.starting_stack
    )
    cfg.env.small_blind = small_blind
    cfg.env.big_blind = big_blind
    cfg.env.starting_stack = starting_stack
    refuse_botzone_init_weights(cfg.env.backend, init_weights, allow_cross_backend_init)
    set_seed(cfg.seed)
    train_device = torch.device(resolve_device(cfg.device))
    rollout_name = cfg.rollout_device
    if rollout_name == "auto":
        rollout_name = resolve_device("auto")
    rollout_device = torch.device(rollout_name)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BettingHistoryTokenizer(max_seq_len=cfg.model.max_seq_len)
    learner = build_model(cfg).to(train_device)
    optimizer = build_optimizer(learner, cfg)
    use_amp = amp_enabled(cfg.train.amp, train_device)  # auto-off on CPU
    scaler = make_grad_scaler(use_amp, train_device)
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    resume_mode, resume_path = _resolve_resume(resume, ckpt_dir)
    pool = OpponentPool(
        max_size=cfg.pool.max_size,
        selfplay_prob=cfg.pool.selfplay_prob,
        random_prob=cfg.pool.random_prob,
        sampling=cfg.pool.sampling,
        seed=cfg.seed,
    )
    start_update = 0
    blob = None
    if resume_path is not None and init_weights is not None:
        raise ValueError("pass only one of resume and init_weights")
    if resume_path is not None:
        # RNG states are restored after the tables are built (building them consumes RNG).
        blob = load_checkpoint(
            resume_path, learner, optimizer, pool, map_location=str(train_device), scaler=scaler, restore_rng=False
        )
        if resume_mode == "auto":
            assert_resume_config(
                blob.get("config"),
                blob.get("config_hash"),
                cfg.to_dict(),
                allow=allow_config_mismatch,
                path=resume_path,
            )
            if allow_config_mismatch and blob.get("config_hash") not in (None, ""):
                if config_hash(cfg.to_dict()) != blob.get("config_hash"):
                    print(f"[resume] auto: config hash mismatch ignored for {resume_path}")
            if not blob.get("config_hash"):
                print(f"[resume] auto: {resume_path} has no config hash; continuing")
            if blob.get("env_rngs") is None:
                print(f"[resume] auto: {resume_path} has no per-env RNG snapshot; tables reset from their seeds")
        start_update = int(blob["update"]) + 1
    elif init_weights is not None:
        # BC warm start: model weights only. Fresh Adam, fresh tables, update 0.
        blob = load_checkpoint(
            init_weights, learner, optimizer=None, pool=None, map_location=str(train_device), restore_rng=False
        )
        print(f"[init-weights] loaded {init_weights}")
    if len(pool) == 0:
        pool.add(learner.state_dict())

    if rollout_device == train_device:
        rollout_model: torch.nn.Module = learner
    else:
        rollout_model = build_model(cfg).to(rollout_device)
        _sync_weights(learner, rollout_model)
    shadow = build_model(cfg).to(rollout_device)
    actor_rng = np.random.default_rng(cfg.seed + 17)
    actor = PolicyActor(rollout_model, shadow, tokenizer, actor_rng)
    use_mask = mask_free_fold_enabled(cfg.env.mask_free_fold)
    envs = [
        SelfPlayEnv(
            tokenizer,
            actor,
            pool,
            seed=cfg.seed + 1000 + index,
            big_blind=cfg.env.big_blind,
            small_blind=cfg.env.small_blind,
            starting_stack=cfg.env.starting_stack,
            backend=cfg.env.backend,
            observer=observer_for(cfg),
            mask_free_fold=use_mask,
        )
        for index in range(cfg.ppo.num_envs)
    ]
    resumed_tables = (
        resume_mode in ("latest", "path")
        and blob is not None
        and blob.get("envs") is not None
        and len(blob["envs"]) == len(envs)
    )
    if resume_mode == "auto" and blob is not None:
        obs = reset_envs_from_rng_snapshot(envs, actor, blob)
    elif resumed_tables:
        for env, env_state in zip(envs, blob["envs"]):
            env.load_state_dict(env_state)
        if blob.get("actor_rng") is not None:
            actor.set_rng_state(blob["actor_rng"])
        obs = [env.observe() for env in envs]
    else:
        obs = [env.reset() for env in envs]
        for env in envs:
            env.drain_passive_rewards()
    if resume_mode in ("latest", "path") and blob is not None:
        restore_rng_state(blob.get("rng"))
    writer = _make_writer(out_dir / "tb")
    csv_path = out_dir / "metrics.csv"
    if resume_path is not None:
        truncate_metrics_csv(csv_path, start_update)

    def _save(update: int) -> None:
        save_checkpoint(
            checkpoint_path(ckpt_dir, update),
            model=learner,
            optimizer=optimizer,
            update=update,
            config=cfg.to_dict(),
            pool=pool,
            scaler=scaler,
            envs=envs,
            actor_rng=actor.rng_state(),
            keep_last=cfg.train.keep_last,
        )

    last_saved = None
    finished = True

    for update in range(start_update, cfg.train.num_updates):
        started = time.perf_counter()
        _sync_weights(learner, rollout_model)
        rollout_model.eval()
        storage, stats = collect_rollout(
            envs,
            obs,
            rollout_model,
            rollout_device,
            cfg.ppo.rollout_steps,
            mask_free_fold=use_mask,
        )
        flat = flatten_rollout(storage, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        flat = {key: value.to(train_device) for key, value in flat.items()}
        decision_to_call = flat.get("to_call")
        metrics = ppo_update(
            learner,
            optimizer,
            {key: value for key, value in flat.items() if key not in _PPO_SKIP},
            actions=flat["actions"],
            old_logp=flat["old_logp"],
            old_values=flat["old_values"],
            advantages=flat["advantages"],
            returns=flat["returns"],
            clip_eps=cfg.ppo.clip_eps,
            entropy_coef=cfg.ppo.entropy_coef,
            value_coef=cfg.ppo.value_coef,
            epochs=cfg.ppo.epochs,
            minibatch_size=min(cfg.ppo.minibatch_size, int(flat["actions"].shape[0])),
            max_grad_norm=cfg.ppo.max_grad_norm,
            amp=use_amp,
            scaler=scaler,
            value_clip_eps=cfg.ppo.value_clip_eps,
            mask_free_fold=use_mask,
            to_call=decision_to_call if use_mask else None,
        )
        # Keep these off the middle of the row. They are appended after the
        # M11a/M13a decision columns so an existing metrics.csv header stays
        # put and the new fields land at the end of the line.
        grad_cols = {key: metrics.pop(key) for key in GRAD_NORM_COLUMNS}
        if cfg.pool.snapshot_every > 0 and (update + 1) % cfg.pool.snapshot_every == 0:
            pool.add(learner.state_dict())
        row = {
            "update": update,
            "timesteps": (update + 1) * cfg.ppo.num_envs * cfg.ppo.rollout_steps,
            "episodes": stats["episodes"],
            "mean_reward_bb": stats["mean_reward_bb"],
            "passive_hands": stats["passive_hands"],
            "pool_size": len(pool),
            "entropy_coef": cfg.ppo.entropy_coef,
            "seconds": time.perf_counter() - started,
            **metrics,
        }
        if cfg.train.eval_every > 0 and cfg.train.eval_hands > 0 and (update + 1) % cfg.train.eval_every == 0:
            _sync_weights(learner, rollout_model)
            row["eval_random_bb"] = evaluate_vs_random(
                rollout_model,
                tokenizer,
                pool,
                cfg,
                rollout_device,
                cfg.train.eval_hands,
                seed=cfg.seed + update,
            )
        else:
            row["eval_random_bb"] = ""
        for key in DECISION_STAT_COLUMNS:
            row[key] = stats[key]
        for key in GRAD_NORM_COLUMNS:
            row[key] = grad_cols[key]
        if cfg.train.log_every > 0 and update % cfg.train.log_every == 0:
            _append_csv(csv_path, row)
            if writer is not None:
                for key, value in row.items():
                    if key == "eval_random_bb" and value == "":
                        continue
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, update)
                writer.flush()
        if cfg.train.checkpoint_every > 0 and (update + 1) % cfg.train.checkpoint_every == 0:
            _save(update)
            last_saved = update
        if reset_envs_after_update is not None and update == int(reset_envs_after_update):
            # Same bytes --resume auto will load: RNG snapshot from this update,
            # then a fresh deal. The hand inside the checkpoint is not restored.
            saved = checkpoint_path(ckpt_dir, update)
            if not saved.is_file():
                raise RuntimeError(
                    f"reset_envs_after_update={update} but {saved} was not written. "
                    "checkpoint_every must save that update."
                )
            try:
                snap = torch.load(saved, map_location="cpu", weights_only=False)
            except TypeError:
                snap = torch.load(saved, map_location="cpu")
            obs = reset_envs_from_rng_snapshot(envs, actor, snap)
        if stop_after_update is not None and update >= int(stop_after_update):
            finished = False
            break
    final_update = cfg.train.num_updates - 1
    if finished and final_update >= start_update and last_saved != final_update:
        _save(final_update)
    if writer is not None:
        writer.close()
    return out_dir


def describe_run(cfg: ExperimentConfig) -> str:
    model = build_model(cfg)
    params = count_parameters(model)
    batch = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    if cfg.encoder == "alphanl_conv":
        body = "alphanl_conv (NlHoldemNet port, fixed 16/32/64 channels)"
    else:
        body = (
            f"transformer d_model={cfg.model.d_model}, layers={cfg.model.num_layers}, "
            f"seq={cfg.model.max_seq_len}"
        )
    return (
        f"{cfg.name}: encoder={cfg.encoder}, {params:,} parameters, {body}, "
        f"envs={cfg.ppo.num_envs}, "
        f"rollout={cfg.ppo.rollout_steps} ({batch} hero decisions/update), "
        f"minibatch={cfg.ppo.minibatch_size}, entropy_coef={cfg.ppo.entropy_coef}, "
        f"device={cfg.device}, rollout_device={cfg.rollout_device}, "
        f"env={cfg.env.backend} {cfg.env.starting_stack / cfg.env.big_blind:g}bb"
    )

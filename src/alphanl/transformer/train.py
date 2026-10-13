"""Self-play PPO loop. Importing it does not start training or import rlcard."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .checkpoint import checkpoint_path, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import ExperimentConfig, resolve_device
from .env import PolicyActor, SelfPlayEnv
from .model import PokerTransformer, count_parameters
from .pool import OpponentPool
from .ppo import amp_enabled, compute_gae, make_grad_scaler, ppo_update
from .tokenizer import BettingHistoryTokenizer, batch_to_device, cat_batches


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: ExperimentConfig) -> PokerTransformer:
    model_cfg = cfg.model
    return PokerTransformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dim_feedforward=model_cfg.dim_feedforward,
        dropout=model_cfg.dropout,
        max_seq_len=model_cfg.max_seq_len,
    )


def build_optimizer(model: PokerTransformer, cfg: ExperimentConfig) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=cfg.ppo.adam_eps)


def _sync_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    if src is dst:
        return
    device = next(dst.parameters()).device
    dst.load_state_dict({key: value.detach().to(device) for key, value in src.state_dict().items()})


def collect_rollout(
    envs: Sequence[Any],
    obs: List[Dict[str, torch.Tensor]],
    model: PokerTransformer,
    device: torch.device,
    rollout_steps: int,
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
    seq_len = int(sample["token_ids"].shape[-1])
    num_actions = int(sample["legal_mask"].shape[-1])
    token_ids = torch.zeros(num_envs, rollout_steps, seq_len, dtype=torch.long)
    type_ids = torch.zeros(num_envs, rollout_steps, seq_len, dtype=torch.long)
    numeric = torch.zeros(num_envs, rollout_steps, seq_len, sample["numeric"].shape[-1])
    pad_mask = torch.ones(num_envs, rollout_steps, seq_len, dtype=torch.bool)
    legal_mask = torch.zeros(num_envs, rollout_steps, num_actions)
    actions = torch.zeros(num_envs, rollout_steps, dtype=torch.long)
    old_logp = torch.zeros(num_envs, rollout_steps)
    old_values = torch.zeros(num_envs, rollout_steps)
    rewards = torch.zeros(num_envs, rollout_steps)
    dones = torch.zeros(num_envs, rollout_steps)
    passive_bb: List[float] = []
    episodes = 0

    model.eval()
    for t in range(rollout_steps):
        stacked = batch_to_device(cat_batches(obs), device)
        with torch.no_grad():
            act, logp, values, _ = model.act(stacked, deterministic=False)
        act = act.cpu()
        logp = logp.cpu()
        values = values.cpu()
        for e, env in enumerate(envs):
            token_ids[e, t] = obs[e]["token_ids"][0]
            type_ids[e, t] = obs[e]["type_ids"][0]
            numeric[e, t] = obs[e]["numeric"][0]
            pad_mask[e, t] = obs[e]["pad_mask"][0]
            legal_mask[e, t] = obs[e]["legal_mask"][0]
            actions[e, t] = act[e]
            old_logp[e, t] = logp[e]
            old_values[e, t] = values[e]
            next_obs, reward, done, _info = env.step(int(act[e]))
            rewards[e, t] = float(reward)
            dones[e, t] = 1.0 if done else 0.0
            if done:
                episodes += 1
                next_obs = env.reset()
                passive_bb.extend(float(x) for x in env.drain_passive_rewards())
            obs[e] = next_obs

    bootstrap_batch = batch_to_device(cat_batches(obs), device)
    with torch.no_grad():
        _, bootstrap = model(bootstrap_batch, mask_illegal=True)
    storage = {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "numeric": numeric,
        "pad_mask": pad_mask,
        "legal_mask": legal_mask,
        "actions": actions,
        "old_logp": old_logp,
        "old_values": old_values,
        "rewards": rewards,
        "dones": dones,
        "bootstrap": bootstrap.detach().cpu(),
    }
    terminal = dones > 0.5
    if int(terminal.sum()) == 0:
        mean_reward = 0.0
    else:
        mean_reward = float(rewards[terminal].mean())
    stats = {
        "episodes": episodes,
        "mean_reward_bb": mean_reward,
        "passive_hands": len(passive_bb),
        "passive_mean_bb": float(np.mean(passive_bb)) if passive_bb else 0.0,
    }
    return storage, stats


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

    return {
        "token_ids": flat(storage["token_ids"]),
        "type_ids": flat(storage["type_ids"]),
        "numeric": flat(storage["numeric"]),
        "pad_mask": flat(storage["pad_mask"]),
        "legal_mask": flat(storage["legal_mask"]),
        "actions": flat(storage["actions"]),
        "old_logp": flat(storage["old_logp"]),
        "old_values": flat(storage["old_values"]),
        "advantages": flat(advantages),
        "returns": flat(returns),
    }


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


@torch.no_grad()
def evaluate_vs_random(
    model: PokerTransformer,
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
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    total = 0.0
    played = 0
    while played < int(n_hands):
        env.reset()
        passive = [float(reward) for reward in env.drain_passive_rewards()]
        total += sum(passive)
        played += len(passive)
        if played >= int(n_hands):
            break
        done = False
        reward = 0.0
        while not done:
            batch = batch_to_device(env.observe(), device)
            actions, _, _, _ = model.act(batch, deterministic=True)
            _, reward, done, _info = env.step(int(actions.item()))
        total += float(reward)
        played += 1
    return total / float(played)


def run_training(
    cfg: ExperimentConfig,
    *,
    resume: Optional[Path | str] = None,
) -> Path:
    """Run ``cfg.train.num_updates`` PPO updates. Returns the output directory.

    ``resume`` is a checkpoint path, or ``"latest"`` to pick the newest
    ``update_*.pt`` in ``cfg.train.checkpoint_dir`` (starts fresh if none).
    """

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
    if resume is not None and str(resume) == "latest":
        resume = latest_checkpoint(ckpt_dir)
        print(f"[resume] {'starting fresh, no checkpoint in ' + str(ckpt_dir) if resume is None else resume}")
    pool = OpponentPool(
        max_size=cfg.pool.max_size,
        selfplay_prob=cfg.pool.selfplay_prob,
        random_prob=cfg.pool.random_prob,
        sampling=cfg.pool.sampling,
        seed=cfg.seed,
    )
    start_update = 0
    if resume is not None:
        # RNG states are restored after the tables are built (building them consumes RNG).
        blob = load_checkpoint(
            resume, learner, optimizer, pool, map_location=str(train_device), scaler=scaler, restore_rng=False
        )
        start_update = int(blob["update"]) + 1
    if len(pool) == 0:
        pool.add(learner.state_dict())

    if rollout_device == train_device:
        rollout_model: PokerTransformer = learner
    else:
        rollout_model = build_model(cfg).to(rollout_device)
        _sync_weights(learner, rollout_model)
    shadow = build_model(cfg).to(rollout_device)
    actor_rng = np.random.default_rng(cfg.seed + 17)
    actor = PolicyActor(rollout_model, shadow, tokenizer, actor_rng)
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
        )
        for index in range(cfg.ppo.num_envs)
    ]
    resumed_tables = resume is not None and blob.get("envs") is not None and len(blob["envs"]) == len(envs)
    if resumed_tables:
        for env, env_state in zip(envs, blob["envs"]):
            env.load_state_dict(env_state)
        if blob.get("actor_rng") is not None:
            actor.set_rng_state(blob["actor_rng"])
        obs = [env.observe() for env in envs]
    else:
        obs = [env.reset() for env in envs]
        for env in envs:
            env.drain_passive_rewards()
    if resume is not None:
        from .checkpoint import restore_rng_state

        restore_rng_state(blob.get("rng"))
    writer = _make_writer(out_dir / "tb")
    csv_path = out_dir / "metrics.csv"

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

    for update in range(start_update, cfg.train.num_updates):
        started = time.perf_counter()
        _sync_weights(learner, rollout_model)
        rollout_model.eval()
        storage, stats = collect_rollout(envs, obs, rollout_model, rollout_device, cfg.ppo.rollout_steps)
        flat = flatten_rollout(storage, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        flat = {key: value.to(train_device) for key, value in flat.items()}
        learner.train()
        metrics = ppo_update(
            learner,
            optimizer,
            {key: flat[key] for key in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")},
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
        )
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
    final_update = cfg.train.num_updates - 1
    if final_update >= start_update and last_saved != final_update:
        _save(final_update)
    if writer is not None:
        writer.close()
    return out_dir


def describe_run(cfg: ExperimentConfig) -> str:
    model = build_model(cfg)
    params = count_parameters(model)
    batch = cfg.ppo.num_envs * cfg.ppo.rollout_steps
    return (
        f"{cfg.name}: {params:,} parameters, "
        f"d_model={cfg.model.d_model}, layers={cfg.model.num_layers}, "
        f"seq={cfg.model.max_seq_len}, envs={cfg.ppo.num_envs}, "
        f"rollout={cfg.ppo.rollout_steps} ({batch} hero decisions/update), "
        f"minibatch={cfg.ppo.minibatch_size}, entropy_coef={cfg.ppo.entropy_coef}, "
        f"device={cfg.device}, rollout_device={cfg.rollout_device}, "
        f"env={cfg.env.backend} {cfg.env.starting_stack / cfg.env.big_blind:g}bb"
    )

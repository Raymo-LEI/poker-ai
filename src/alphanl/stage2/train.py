"""Match rollout, minibatched PPO and portable resumable checkpoints."""
from dataclasses import dataclass
from pathlib import Path
import random
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.transformer.ppo import compute_gae
from alphanl.transformer.tokenizer import cat_batches
from alphanl.transformer.hunl import RULES_VERSION
from .context import CONTEXT_FIELDS
from .policy import ResidualExploiterPolicy, residual_ppo_loss

FORMAT = "stage2_match_residual_v1"

@dataclass
class MatchRollout:
    batch: dict
    actions: torch.Tensor
    old_logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    report: dict

@torch.no_grad()
def collect_match(model, env):
    model.eval()
    device = next(model.parameters()).device
    obs = env.reset()
    batches, actions, logps, values, rewards, dones = [], [], [], [], [], []
    while not env.done:
        batch = model.prepare({k: v.to(device) for k, v in obs.items()})
        action, logp, value, _ = model.act(batch)
        obs, reward, done, _ = env.step(int(action.item()))
        batches.append(batch)
        actions.append(action)
        logps.append(logp)
        values.append(value)
        rewards.append(reward)
        dones.append(float(done))
    if not actions:
        raise RuntimeError("match has no hero decisions")
    # Reward preceding the first decision is policy-independent but retained for accounting.
    rewards[0] += env.initial_reward
    return MatchRollout(cat_batches(batches), torch.cat(actions), torch.cat(logps), torch.cat(values),
                        torch.tensor(rewards, device=device), torch.tensor(dones, device=device),
                        {"hands": env.hands, "hands_played": env.ctx.hands_played,
                         "hero_net_bb": env.ctx.hero_net_bb, "context": env.ctx.as_dict()})


def train_match(model, optimizer, rollout, *, epochs=4, minibatch_size=256, clip_eps=.2,
                entropy_coef=.01, value_coef=.5, max_grad_norm=.5, gamma=1., gae_lambda=1.):
    if epochs < 1 or minibatch_size < 1 or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("invalid PPO settings")
    advantages, returns = compute_gae(rollout.rewards[None], rollout.values[None], rollout.dones[None],
                                     torch.zeros(1, device=rollout.values.device), gamma, gae_lambda)
    advantages, returns = advantages.flatten(), returns.flatten()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    model.eval()  # deterministic features throughout PPO epochs
    metrics_rows = []
    for _ in range(epochs):
        for ids in torch.randperm(len(advantages), device=advantages.device).split(minibatch_size):
            batch = {k: v[ids] for k, v in rollout.batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = residual_ppo_loss(model, batch, actions=rollout.actions[ids], old_logp=rollout.old_logp[ids],
                old_values=rollout.values[ids], advantages=advantages[ids], returns=returns[ids],
                clip_eps=clip_eps, entropy_coef=entropy_coef, value_coef=value_coef)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Stage 2 PPO loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite Stage 2 gradients")
            optimizer.step()
            metrics_rows.append((len(ids), metrics))
    return {key: sum(n * m[key] for n, m in metrics_rows) / sum(n for n, _ in metrics_rows)
            for key in metrics_rows[0][1]}


def save_checkpoint(path, model, *, optimizer=None, pool=None, update=0, metadata=None):
    blob = {"format": FORMAT, "config": model.config.to_dict(), "settings": model.settings,
            "rules_version": RULES_VERSION, "numeric_version": model.config.numeric_version,
            "context_fields": list(CONTEXT_FIELDS), "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "update": update, "metadata": metadata or {}, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate()}
    if next(model.parameters()).device.type == "cuda":
        blob["cuda_rng"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    if pool is not None:
        blob["pool"] = pool.state_dict()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    config = CFRConfig.from_checkpoint(blob["config"])
    if (blob.get("format") != FORMAT or blob.get("rules_version") != RULES_VERSION
            or blob.get("numeric_version") != config.numeric_version
            or blob.get("context_fields") != list(CONTEXT_FIELDS)):
        raise ValueError("incompatible Stage 2 checkpoint")
    model = ResidualExploiterPolicy(PolicyNetwork(config), **blob["settings"])
    model.load_state_dict(blob["model"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite Stage 2 checkpoint")
    return model.to(device).eval(), blob

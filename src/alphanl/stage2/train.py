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


def match_report(env):
    """One completed match, with identity kept out of the policy observation."""
    if not env.done or env.ctx.hands_played != 50:
        raise ValueError("cannot report an unfinished match")
    return {"hands": env.hands, "hands_played": env.ctx.hands_played,
            "hero_net_bb": env.ctx.hero_net_bb, "context": env.ctx.as_dict(),
            "opponent": env.opponent_name,
            "opponent_snapshot": getattr(env.opponent, "snapshot_id", None),
            "opponent_variant": getattr(env.opponent, "variant", None)}

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
                        match_report(env))


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
    metrics = {key: sum(n * m[key] for n, m in metrics_rows) / sum(n for n, _ in metrics_rows)
               for key in metrics_rows[0][1]}
    metrics.update(final_policy_diagnostics(model, rollout.batch, minibatch_size=minibatch_size))
    return metrics


@torch.no_grad()
def final_policy_diagnostics(model, batch, *, minibatch_size=256):
    """Measure the final update on collected states, separately from epoch means."""
    from alphanl.transformer.model import masked_log_probs
    totals = dict(final_base_kl=0., final_policy_tv=0., final_argmax_changed=0.)
    count = len(batch['legal_mask'])
    for start in range(0, count, minibatch_size):
        part = {k: v[start:start + minibatch_size] for k, v in batch.items()}
        part = model.prepared(part)
        logits, _ = model(part)
        _, logp, probs, legal = masked_log_probs(logits, part['legal_mask'])
        _, ref_logp, ref, _ = masked_log_probs(part['ref_logits'], part['legal_mask'])
        totals['final_base_kl'] += float((probs * (logp-ref_logp).masked_fill(~legal, 0.)).sum(-1).clamp_min(0.).sum())
        totals['final_policy_tv'] += float((probs-ref).abs().sum(-1).sum() / 2)
        totals['final_argmax_changed'] += float((probs.argmax(-1) != ref.argmax(-1)).sum())
    return {k: v / count for k, v in totals.items()}


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


@torch.no_grad()
def collect_matches(model, envs, *, on_match_end=None):
    """Advance concurrent matches, grouping hero and snapshot opponent inference.

    Each env owns its actor for all 50 hands. Rewards before/between hero
    decisions, including passive hands, belong to that same complete match.
    """
    from .opponents import NeuralMatchActor, neural_actions
    if not envs or any(not env.defer_opponents for env in envs):
        raise ValueError("batched collection requires deferred opponent environments")
    model.eval()
    device = next(model.parameters()).device
    records = []
    for env in envs:
        env.reset()
        records.append({"batches": [], "actions": [], "logps": [], "values": [], "rewards": [],
                        "initial": env.initial_reward})

    reported = set()
    def publish_completed():
        for index, env in enumerate(envs):
            if env.done and index not in reported:
                if on_match_end is not None:
                    on_match_end(index, match_report(env))
                reported.add(index)

    publish_completed()

    def credit(index, reward):
        row = records[index]
        if row["rewards"]:
            row["rewards"][-1] += reward
        else:
            row["initial"] += reward

    while any(not env.done for env in envs):
        pending = [i for i, env in enumerate(envs) if env.pending_opponent()]
        neural = [i for i in pending if isinstance(envs[i].opponent, NeuralMatchActor)]
        actions = neural_actions([(envs[i].opponent, envs[i].table.snapshot(1 - envs[i].hero)) for i in neural])
        selected = dict(zip(neural, actions))
        for i in pending:
            env = envs[i]
            action = selected[i] if i in selected else env.opponent.act(env.table.snapshot(1 - env.hero))
            credit(i, env.opponent_step(action))
        ready = [i for i, env in enumerate(envs) if not env.done and not env.pending_opponent()]
        if not ready:
            publish_completed()
            continue
        batch = model.prepare({k: v.to(device) for k, v in cat_batches([envs[i].observe() for i in ready]).items()})
        actions, logps, values, _ = model.act(batch)
        for j, i in enumerate(ready):
            row, env = records[i], envs[i]
            row["batches"].append({k: v[j:j+1] for k, v in batch.items()})
            row["actions"].append(actions[j:j+1])
            row["logps"].append(logps[j:j+1])
            row["values"].append(values[j:j+1])
            row["rewards"].append(0.)
            _, reward, _, _ = env.step(int(actions[j]))
            credit(i, reward)
        publish_completed()
    rollouts = []
    for env, row in zip(envs, records):
        if not row["actions"]:
            raise RuntimeError("match has no hero decisions")
        row["rewards"][0] += row["initial"]
        dones = torch.zeros(len(row["actions"]), device=device)
        dones[-1] = 1.
        rollouts.append(MatchRollout(cat_batches(row["batches"]), torch.cat(row["actions"]), torch.cat(row["logps"]),
            torch.cat(row["values"]), torch.tensor(row["rewards"], device=device), dones,
            match_report(env)))
    return rollouts


def train_matches(model, optimizer, rollouts, **kwargs):
    """GAE resets at each match's done; one PPO batch covers the quota block."""
    merged = MatchRollout(cat_batches([r.batch for r in rollouts]),
        *[torch.cat([getattr(r, key) for r in rollouts])
          for key in ("actions", "old_logp", "values", "rewards", "dones")], report={})
    return train_match(model, optimizer, merged, **kwargs)

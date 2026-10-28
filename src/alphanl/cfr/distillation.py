"""Offline compression of a frozen SD-CFR trajectory mixture.

Select t with probability proportional to t, then keep sigma_t for the hand.
Collect only hero-reached infosets against a snapshot-independent opponent.
The resulting soft-label cross entropy has the own-reach weighted average
as its population optimum. Never average local outputs or network weights.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .actions import CFRActionGuard
from .config import CFRConfig
from .model import RegretNetwork
from .strategy import SnapshotPolicy
from alphanl.transformer.hunl import HunlTable, HunlTokenizer, NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer


FORMAT = "sd_cfr_distilled_policy_v1"
SEMANTIC_FIELDS = (
    "model", "fallback", "separate_player_models", "position_conditioning",
    "regret_target_scale", "preflop_action_guard", "spr_action_guard",
    "guard_max_pot_ratio", "guard_short_stack_bb", "guard_unlock_invested_bb",
    "guard_unlock_raises", "equity_features", "equity_samples", "equity_seed",
)


def semantics_match(semantics, config):
    legacy = {"equity_features": False, "equity_samples": 32, "equity_seed": 17}
    actual = {**legacy, **semantics}
    return actual == {key: config.to_dict()[key] for key in SEMANTIC_FIELDS}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def prepare_archive(directory, *, through=None, after=0, stride=1, sampling_power=1.0):
    """Validate and fingerprint every selected profile without touching training.

    Default is all indexed history. An explicit boundary must exist; missing
    future snapshots cannot silently produce a shorter teacher. Sparse and
    tail selections are deliberately labeled approximations.
    """
    if sampling_power not in (1., 2.):
        raise ValueError("distillation sampling_power must be 1 or 2")
    _positive_integer(stride, "stride")
    if isinstance(after, bool) or not isinstance(after, int) or after < 0:
        raise ValueError("after must be a nonnegative integer")
    archive = SnapshotPolicy(directory, through_iteration=through)
    available = archive.iterations.tolist()
    if through is not None and through not in available:
        raise ValueError("through iteration is not present in the archive")
    selected = [t for t in available if t > after and t % stride == 0]
    if not selected:
        raise ValueError("snapshot selection is empty")
    records, reference, validator = [], None, None
    for t, path in zip(available, archive.paths):
        if t not in selected:
            continue
        data = path.read_bytes()
        blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        config = CFRConfig.from_checkpoint(blob["config"])
        if (blob.get("format") not in ("sd_cfr_strategy_v1", "sd_cfr_strategy_v2")
                or blob.get("iteration") != t or blob.get("weight") != t
                or blob.get("rules_version") != RULES_VERSION
                or blob.get("numeric_version") != config.numeric_version):
            raise ValueError(f"incompatible strategy snapshot: {path.name}")
        if config.separate_player_models or not config.position_conditioning:
            raise ValueError("single-student distillation requires unified position-conditioned snapshots")
        semantics = {key: config.to_dict()[key] for key in SEMANTIC_FIELDS}
        if reference is None:
            reference = semantics
            validator = RegretNetwork(PokerTransformer(**asdict(config.model)), fallback=config.fallback)
            first_config = config
        elif semantics != reference:
            raise ValueError("selected snapshots have incompatible policy semantics")
        validator.load_state_dict(blob["models"]["shared"], strict=True)
        if not all(torch.isfinite(tensor).all() for tensor in validator.state_dict().values()):
            raise ValueError("snapshot contains nonfinite weights")
        records.append({"iteration": t, "weight": t, "sampling_weight": t ** sampling_power, "file": path.name,
                        "sha256": hashlib.sha256(data).hexdigest()})
    full_prefix = list(range(1, max(available) + 1))
    return first_config, {
        "format": "sd_cfr_distillation_manifest_v1",
        "rules_version": RULES_VERSION, "numeric_version": config.numeric_version,
        "selection": {"through": through, "after_exclusive": after, "stride": stride},
        "sampling_power": sampling_power,
        "iterations": selected, "snapshots": records,
        "approximation": selected != full_prefix,
        "sampling": "iteration_proportional_once_per_hand_hero_on_policy" if sampling_power == 1 else "iteration_squared_once_per_hand_hero_on_policy",
        "semantics": reference,
    }


def verify_archive(directory, manifest):
    """Fail if training resume or another publisher replaced a teacher file."""
    for record in manifest["snapshots"]:
        data = (Path(directory) / record["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("teacher archive changed since preparation")


class PolicyNetwork(nn.Module):
    """One position-conditioned Transformer with a five-way policy head.

    Reuse the backbone implementation but remove its unused value head.
    This checkpoint is intentionally distinct from PPO and regret checkpoints.
    """

    def __init__(self, config: CFRConfig):
        super().__init__()
        if config.separate_player_models or not config.position_conditioning:
            raise ValueError("PolicyNetwork requires unified position conditioning")
        self.config = config
        self.backbone = PokerTransformer(**asdict(config.model))
        del self.backbone.value_head

    def forward(self, batch):
        legal = batch["legal_mask"] > .5
        if not legal.any(-1).all():
            raise ValueError("every decision must have a legal action")
        features = self.backbone.trunk(self.backbone.encode_tokens(
            batch["token_ids"], batch["type_ids"], batch["numeric"], batch["pad_mask"],
        ))
        return self.backbone.policy_head(features.float()).masked_fill(~legal, -torch.inf)

    def strategy(self, batch):
        return self(batch).softmax(-1)


def soft_cross_entropy(logits, targets, legal_mask):
    """Distribution targets; zero illegal gradient, including masked -inf."""
    legal = legal_mask > .5
    if (logits.shape != targets.shape or logits.shape != legal.shape
            or logits.ndim != 2 or not len(logits) or not legal.any(-1).all()):
        raise ValueError("expected matching nonempty [batch, actions] with legal actions")
    if (not torch.isfinite(targets).all() or (targets < 0).any()
            or (targets[~legal] != 0).any()
            or not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-5)
            or not torch.isfinite(logits[legal]).all()):
        raise ValueError("invalid legal logits or target distribution")
    logp = F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
    return -(targets * logp.masked_fill(~legal, 0.)).sum(-1).mean()


@torch.inference_mode()
def collect_trajectories(directory, config, manifest, *, hands, seed, hand_offset=0,
                         max_decisions=200_000, teacher_device="cpu"):
    """Bounded in-memory dataset, uniform legal opponent, alternating hero seats.

    Teacher labels and hero actions use the same fixed snapshot. There is no
    extra t weighting: trajectory sampling already supplies it. Hand ids and
    the deck seed recipe are retained for disjoint, reproducible splits.
    """
    _positive_integer(hands, "hands")
    _positive_integer(max_decisions, "max_decisions")
    for name, value in (("seed", seed), ("hand_offset", hand_offset)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if hand_offset + hands >= 2**63:
        raise ValueError("hand ids must fit in int64")
    if not semantics_match(manifest["semantics"], config):
        raise ValueError("collector and teacher semantics disagree")
    verify_archive(directory, manifest)
    policy_seed = int(np.random.SeedSequence([seed, hand_offset, 1]).generate_state(1)[0])
    policy = SnapshotPolicy(directory, seed=policy_seed, device=teacher_device,
                            snapshot_iterations=manifest["iterations"], cache_size=8,
                            sampling_power=manifest.get("sampling_power", 1.))
    tokenizer = HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs())
    guard = CFRActionGuard.from_config(config)
    rows, counts = [], Counter()
    for hand in range(hand_offset, hand_offset + hands):
        hero = hand % 2
        # Injective across hand ids, avoiding collisions from 32-bit hashes
        # when a real run collects tens of thousands of train/held-out hands.
        deck_seed = (seed << 64) + hand
        opponent_seed = int(np.random.SeedSequence([seed, hand]).generate_state(1)[0])
        opponent = np.random.default_rng(opponent_seed)
        table = HunlTable(seed=deck_seed)
        table.reset()
        iteration = policy.begin_hand(hero)
        counts[iteration] += 1
        steps = 0
        while not table.is_over():
            seat = table.current_player()
            snapshot = guard.apply(table.snapshot(seat))
            if seat == hero:
                batch = tokenizer.encode(snapshot)
                probs = policy.strategy(batch).cpu()
                rows.append({**batch, "target_probs": probs,
                             "hand_id": torch.tensor([hand]),
                             "hero_seat": torch.tensor([hero]),
                             "iteration": torch.tensor([iteration]),
                             "street": torch.tensor([snapshot.street])})
                if len(rows) > max_decisions:
                    raise ValueError("max_decisions exceeded; lower hands or increase the explicit memory bound")
                row = probs[0].double().numpy()
                action = int(policy.rng.choice(5, p=row / row.sum()))
            else:
                action = int(opponent.choice(snapshot.legal_action_ids))
            table.step(action)
            steps += 1
            if steps > 1000:
                raise RuntimeError("distillation hand failed to terminate")
        if sum(table.payoffs()) != 0 or table.chips_in_play() != 40000:
            raise RuntimeError("chip conservation failed")
    if not rows:
        raise ValueError("no hero decisions collected; increase hands")
    verify_archive(directory, manifest)
    dataset = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    return dataset, {"hands": hands, "decisions": len(rows), "snapshot_hand_counts": dict(counts),
                     "opponent": "uniform_over_guarded_legal_actions", "seed": seed,
                     "deck_seed_scheme": "(seed << 64) + hand_id",
                     "hand_range": [hand_offset, hand_offset + hands]}


def fit_student(model, dataset, *, steps, batch_size, seed, learning_rate=3e-4):
    _positive_integer(steps, "steps")
    _positive_integer(batch_size, "batch_size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.eval()  # Train with gradients, but disable dropout as in CFR fitting.
    for _ in range(steps):
        ids = torch.randint(len(dataset["target_probs"]), (batch_size,), generator=generator)
        batch = {key: value[ids].to(device) for key, value in dataset.items()}
        loss = soft_cross_entropy(model(batch), batch["target_probs"], batch["legal_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return float(loss.detach())


@torch.inference_mode()
def evaluate_student(model, dataset, *, batch_size=256):
    """Held-out proper scoring loss; component KL is NOT mixture exploitability."""
    _positive_integer(batch_size, "batch_size")
    model.eval()
    device = next(model.parameters()).device
    count, total_ce, total_entropy = len(dataset["target_probs"]), 0., 0.
    coverage = Counter(zip(dataset["hero_seat"].tolist(), dataset["street"].tolist()))
    slice_ce = Counter()
    for start in range(0, count, batch_size):
        batch = {key: value[start:start + batch_size].to(device) for key, value in dataset.items()}
        logits, targets = model(batch), batch["target_probs"]
        n = len(targets)
        total_ce += float(soft_cross_entropy(logits, targets, batch["legal_mask"])) * n
        total_entropy += float(-(targets * targets.clamp_min(1e-30).log()).sum())
        probs = logits.softmax(-1)
        if not torch.isfinite(probs).all() or (probs[batch["legal_mask"] <= .5] != 0).any():
            raise ValueError("student produced invalid or illegal probabilities")
        logp = logits.log_softmax(-1).masked_fill(batch["legal_mask"] <= .5, 0.)
        row_ce = (-(targets * logp).sum(-1)).cpu().tolist()
        for seat, street, ce in zip(batch["hero_seat"].cpu().tolist(),
                                    batch["street"].cpu().tolist(), row_ce):
            slice_ce[(seat, street)] += ce
    if count == 0:
        raise ValueError("empty validation dataset")
    return {"decisions": count, "soft_cross_entropy_nats": total_ce / count,
            "sampled_component_kl_nats": max(0., (total_ce - total_entropy) / count),
            "seat_street_counts": {f"{seat}:{street}": n for (seat, street), n in coverage.items()},
            "seat_street_cross_entropy_nats": {
                f"{seat}:{street}": slice_ce[(seat, street)] / n
                for (seat, street), n in coverage.items()},
            "illegal_probability_mass": 0.,
            "metric_scope": "held_out_teacher_reached_states_not_exploitability"}


@torch.inference_mode()
def cpu_latency(model, dataset, *, decisions=100):
    """Warm batch=1 single-thread forward+softmax; excludes startup/tokenization."""
    _positive_integer(decisions, "decisions")
    if next(model.parameters()).device.type != "cpu" or torch.get_num_threads() != 1:
        raise ValueError("latency measurement requires a CPU model and one torch thread")
    model.eval()
    rows = len(dataset["target_probs"])
    for i in range(10):
        model.strategy({key: value[i % rows:i % rows + 1] for key, value in dataset.items()})
    times = []
    for i in range(decisions):
        batch = {key: value[i % rows:i % rows + 1] for key, value in dataset.items()}
        started = time.perf_counter()
        model.strategy(batch)
        times.append((time.perf_counter() - started) * 1000)
    return {"batch_size": 1, "torch_threads": 1, "decisions": decisions,
            "p50_ms": float(np.median(times)), "p99_ms": float(np.quantile(times, .99)),
            "scope": "warm_forward_plus_softmax_excludes_load_tokenize_and_protocol"}


def save_student(path, model, manifest, report):
    if not semantics_match(manifest["semantics"], model.config):
        raise ValueError("student and teacher semantics disagree")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "config": model.config.to_dict(),
                "rules_version": RULES_VERSION, "numeric_version": model.config.numeric_version,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "teacher": manifest, "report": report}, path)


def load_student(path, *, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if (blob.get("format") != FORMAT or blob.get("rules_version") != RULES_VERSION
            or blob.get("numeric_version") != CFRConfig.from_checkpoint(blob["config"]).numeric_version):
        raise ValueError("incompatible distilled policy checkpoint")
    config = CFRConfig.from_checkpoint(blob["config"])
    if not semantics_match(blob["teacher"]["semantics"], config):
        raise ValueError("student checkpoint semantics disagree with teacher manifest")
    model = PolicyNetwork(config).to(device)
    model.load_state_dict(blob["model"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("nonfinite student weights")
    return model.eval(), blob

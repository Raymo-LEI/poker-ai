"""Train the frozen-blueprint street-end leaf on realized chip deltas.

The policy parameters are not stepped. Only the 32-48-16-6 head learns.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

import wizard
from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.opponents import NeuralMatchActor
from alphanl.transformer.tables import standard_deck
from phevaluator import evaluate_cards
from surgery import load_model

OUT = Path("/root/autodl-tmp/sprint24/wizard")
SURGICAL = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt")
COLLECT_SEED = 20261104


def choose_blueprint():
    return {
        "kind": "surgical", "pt": str(SURGICAL),
        "equity": 1958.1, "tag": -50.6, "trapper": -0.4, "h2h_v9": 1.0,
    }


def collect(model, device):
    wizard.bind(evaluate_cards)
    torch.set_grad_enabled(False)
    mask = model.settings.get("botzone_native_allin_mask")
    rows, targets, buckets = [], [], []
    names = ("equity", "TAG_Pro", "Trapper_CR", "self")
    match_id = 0
    for name in names:
        for seat in (1, 0):
            for replica in range(8):
                rng = np.random.default_rng(COLLECT_SEED + match_id)
                decks = [rng.permutation(standard_deck()).tolist() for _ in range(50)]
                seed = COLLECT_SEED + match_id
                if name == "self":
                    opponent = NeuralMatchActor(model, kind="residual", snapshot_id="self", seed=seed)
                else:
                    opponent = RuleActor(name, seed=seed, equity_samples=16)
                env = BotzoneMatchEnv(
                    model.config, opponent, decks=decks, first_hero_seat=seat, seed=seed,
                    context_fields=model.context_fields, allin_mask=mask,
                )
                env.reset()
                seen = []
                while not env.done:
                    snap = env._hero_decision_snapshot()
                    hand_index = env.ctx.hands_played
                    if snap.street > 0:
                        feat, bucket = wizard.observe(snap, env.ctx)
                        seen.append((hand_index, feat, int(bucket), snap.me_stack, snap.big_blind))
                    batch = {key: value.to(device) for key, value in env.observe().items()}
                    action, *_ = model.act(batch)
                    seen_action = int(action.item()) if hasattr(action, "item") else int(action)
                    if seen and seen[-1][0] == hand_index and snap.street > 0:
                        seen[-1] = seen[-1] + (seen_action,)
                    env.step(seen_action)
                for item in seen:
                    if len(item) != 6:
                        continue
                    hand_index, feat, bucket, stack, blind, _action = item
                    net = env.hands[hand_index]["hero_net_bb"]
                    delta = (snap_start(blind) + net * blind - stack) / blind
                    rows.append(feat)
                    targets.append(float(np.clip(delta, -200, 200)))
                    buckets.append(bucket)
                match_id += 1
                if match_id % 8 == 0:
                    print("COLLECT", match_id, len(rows), flush=True)
    return np.stack(rows).astype(np.float32), np.asarray(targets, np.float32), np.asarray(buckets, np.int64)


def snap_start(blind):
    return 200.0 * blind


def fit(features, targets, buckets):
    torch.set_grad_enabled(True)
    torch.manual_seed(20261104)
    net = nn.Sequential(nn.Linear(48, 32), nn.GELU(), nn.Linear(32, 12)).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.tensor(features, device="cuda")
    y = torch.tensor(targets, device="cuda")
    bucket = torch.tensor(buckets, device="cuda")
    loss_value = None
    for epoch in range(8):
        order = torch.randperm(x.shape[0], device="cuda")
        total, seen = 0.0, 0
        for start in range(0, x.shape[0], 1024):
            batch = order[start:start + 1024]
            pred = net(x[batch])
            taken = pred[torch.arange(batch.shape[0], device="cuda"), bucket[batch]]
            loss = (taken - y[batch]).pow(2).mean() + 0.02 * pred.var(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * batch.shape[0]
            seen += batch.shape[0]
        loss_value = total / seen
        print("EPOCH", epoch, round(loss_value, 4), flush=True)
    weights = dict(net.named_parameters())
    arrays = {
        "vleaf.0.weight": weights["0.weight"].detach().cpu().numpy(),
        "vleaf.0.bias": weights["0.bias"].detach().cpu().numpy(),
        "vleaf.1.weight": weights["2.weight"].detach().cpu().numpy(),
        "vleaf.1.bias": weights["2.bias"].detach().cpu().numpy(),
    }
    return arrays, loss_value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wizard.bind(evaluate_cards)
    choice = choose_blueprint()
    print("BLUEPRINT", json.dumps(choice), flush=True)
    model, _blob = load_model(Path(choice["pt"]), "cuda")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    features, targets, buckets = collect(model, "cuda")
    print("SAMPLES", len(targets), "mean", round(float(targets.mean()), 3), flush=True)
    arrays, loss_value = fit(features, targets, buckets)
    leaf = OUT / "vleaf.npz"
    np.savez(leaf, **{key: np.ascontiguousarray(value.astype(np.float32)) for key, value in arrays.items()})
    (OUT / "blueprint.json").write_text(json.dumps({**choice, "samples": int(len(targets)), "loss": loss_value}) + "\n")
    print("VLEAF_DONE", leaf, round(loss_value, 4), flush=True)


if __name__ == "__main__":
    main()

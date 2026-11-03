"""Search-guided distillation. V10 wizard labels teach the surgical blueprint."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.evaluation import MatchEvaluator
from alphanl.stage2.opponents import NeuralMatchActor
from alphanl.stage2.train import save_checkpoint
from alphanl.transformer.model import masked_log_probs
from alphanl.transformer.tables import standard_deck
from surgery import load_model

OUT = Path("/root/autodl-tmp/sprint24/rebel")
SURGICAL = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt")
WIZARD_ZIP = Path("/root/autodl-tmp/sprint24/deliver/V10_WIZARD/V10_WIZARD_SOURCE.zip")
WIZARD_NPZ = Path("/root/autodl-tmp/sprint24/deliver/V10_WIZARD/V10_WIZARD_MODEL.npz")
TARGET = 100000
COLLECT_SEED = 20261104
KL_BETA = 0.05
LR = 3e-5
EPOCHS = 4


def load_wizard():
    tree = OUT / "runtime"
    if not (tree / "stage2_runtime" / "wizard.py").is_file():
        tree.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(WIZARD_ZIP) as archive:
            archive.extractall(tree)
    sys.path.insert(0, str(tree))
    from stage2_runtime.deployment_guard import apply_guard
    from stage2_runtime.equity import estimated_equity
    from stage2_runtime.runtime import Agent
    from stage2_runtime.wizard import fuse

    with np.load(WIZARD_NPZ, allow_pickle=False) as archive:
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        weights = {key: archive[key] for key in archive.files if key != "meta_json"}
    agent = Agent(weights, meta["config"], meta["settings"], meta["numeric_version"], meta.get("deployment_guard"))
    return agent, apply_guard, estimated_equity, fuse


def distribution(agent, snapshot, context, apply_guard, estimated_equity, fuse):
    row = agent.logits(agent.observation(snapshot, context))[0].astype(np.float64)
    probs = np.exp(row - row.max())
    probs /= probs.sum()
    probs, agent.last_guard = apply_guard(snapshot, probs, agent.deployment_guard, estimated_equity, None)
    return np.asarray(fuse(agent, snapshot, context, probs, None), dtype=np.float64)


def freeze(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        out[key] = np.array(value)
    return out


def collect(model, agent, helpers):
    apply_guard, estimated_equity, fuse = helpers
    rows, targets = [], []
    names = ("equity", "TAG_Pro", "Trapper_CR", "self")
    match_id = 0
    while len(rows) < TARGET:
        name = names[match_id % len(names)]
        seat = 1 if (match_id // len(names)) % 2 == 0 else 0
        seed = COLLECT_SEED + match_id
        rng = np.random.default_rng(seed)
        decks = [rng.permutation(standard_deck()).tolist() for _ in range(50)]
        if name == "self":
            opponent = NeuralMatchActor(model, kind="residual", snapshot_id="self", seed=seed)
        else:
            opponent = RuleActor(name, seed=seed, equity_samples=16)
        env = BotzoneMatchEnv(
            model.config, opponent, decks=decks, first_hero_seat=seat, seed=seed,
            context_fields=model.context_fields,
            allin_mask=model.settings.get("botzone_native_allin_mask"),
        )
        env.reset()
        while not env.done:
            snap = env._hero_decision_snapshot()
            probs = distribution(agent, snap, env.ctx, apply_guard, estimated_equity, fuse)
            if str(getattr(agent, "last_wizard", "")).startswith("wiz=1"):
                rows.append(freeze(env.observe()))
                targets.append(probs.astype(np.float32))
            action = int(rng.choice(5, p=probs / probs.sum()))
            env.step(action)
        match_id += 1
        if match_id % 4 == 0:
            print("REBEL_COLLECT", match_id, len(rows), name, flush=True)
    return rows[:TARGET], np.stack(targets[:TARGET])


def batch_at(rows, index, device):
    chosen = [rows[int(i)] for i in index]
    batch = {}
    for key in chosen[0]:
        value = np.stack([item[key] for item in chosen])
        if value.ndim >= 2 and value.shape[1] == 1:
            value = value[:, 0]
        batch[key] = torch.as_tensor(value, device=device)
    return batch


def fit(model, rows, targets):
    frozen, _ = load_model(SURGICAL, "cuda")
    frozen.eval()
    for param in frozen.parameters():
        param.requires_grad_(False)
    model.train()
    trainable = [param for name, param in model.named_parameters() if not name.startswith("reference.")]
    opt = torch.optim.Adam(trainable, lr=LR)
    device = "cuda"
    order = np.arange(len(rows))
    last = None
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(COLLECT_SEED + epoch)
        rng.shuffle(order)
        total, seen = 0.0, 0
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            batch = batch_at(rows, index, device)
            target = torch.as_tensor(targets[index], device=device)
            legal = batch["legal_mask"] > 0.5
            target = torch.where(legal, target, torch.zeros_like(target))
            target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
            logits, _ = model(batch)
            _, logp, probs, _ = masked_log_probs(logits, batch["legal_mask"])
            ce = -(target * logp).sum(-1).mean()
            with torch.no_grad():
                ref_logits, _ = frozen(batch)
                _, ref_logp, _, _ = masked_log_probs(ref_logits, batch["legal_mask"])
            kl = (probs * (logp - ref_logp)).sum(-1).clamp_min(0).mean()
            loss = ce + KL_BETA * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(index)
            seen += len(index)
        last = total / seen
        print("REBEL_EPOCH", epoch, round(last, 5), flush=True)
    model.eval()
    return last


def confirm(model):
    evaluator = MatchEvaluator(
        model, match_pairs=50, seed=20261030, equity_samples=32,
        opponents=("equity", "TAG_Pro", "Trapper_CR", "reference"),
    )
    report = evaluator.evaluate(model, update=0, progress=lambda name: print("REBEL_CONFIRM", name, flush=True))
    scores = {}
    for name in ("equity", "TAG_Pro", "Trapper_CR"):
        row = report["opponents"][name]["candidate"]
        scores[name] = round(row["mbb_per_hand"], 1)
        print("REBEL_NET", name, scores[name], flush=True)
    return scores


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(SURGICAL, "cuda")
    model.eval()
    agent, *helpers = load_wizard()
    rows, targets = collect(model, agent, helpers)
    print("REBEL_SAMPLES", len(rows), flush=True)
    loss = fit(model, rows, targets)
    dest = OUT / "u250_rebel_blueprint.pt"
    save_checkpoint(dest, model, update=0, metadata={"loss": loss, "samples": len(rows), "beta_kl": KL_BETA})
    print("REBEL_TRAIN_DONE", dest, round(loss, 5), flush=True)
    scores = confirm(model)
    (OUT / "rebel_net.json").write_text(json.dumps(scores, indent=2) + "\n")
    print("REBEL_NET_DONE", json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()

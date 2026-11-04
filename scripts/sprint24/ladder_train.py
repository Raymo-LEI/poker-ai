"""120-update ladder-slayer fine-tune, then one Astra cut and an unconditional V12 pack."""
import copy
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

import pack_wizard
import surgery
from alphanl.cfr.pro_bots import PRO_OPPONENTS, JitteredProScriptActor
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.evaluation import MatchEvaluator, evaluation_metrics
from alphanl.stage2.ladder_bots import LADDER_OPPONENTS, JitteredLadderReplicaActor
from alphanl.stage2.opponents import JitteredRuleActor, NeuralMatchActor
from alphanl.stage2.pool import DEFENSE, MatchOpponentPool
from alphanl.stage2.train import collect_matches, save_checkpoint, train_matches
from ladder_actors import SLAYER_NAMES, SlayerActor
from surgery import load_model

OUT = Path("/root/autodl-tmp/sprint24/ladder")
REBEL = Path("/root/autodl-tmp/sprint24/rebel/u250_rebel_blueprint.pt")
SURGICAL = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt")
STRATEGIES = "/root/autodl-tmp/stage2_botzone_native_code_20261030/runs/frozen_teacher/strategies"
RELEASE = "V12_R12_LADDER_20261104"
WEIGHTS = "V12_MODEL.npz"
SOURCE_NAME = "V12_SOURCE.zip"


def install_slayers():
    import alphanl.stage2.pool as poolmod

    weights = {
        "reference": 0.10, "history": 0.15, "self": 0.10,
        "TAG_Pro": 0.15, "Trapper_CR": 0.15, "Bot17_Probe3Bet": 0.15,
        "NearShove_19949": 0.05, "CheckShove_Trapper": 0.05, "equity": 0.10,
    }
    poolmod._ALLOWED_POOLS = poolmod._ALLOWED_POOLS + (frozenset(weights),)

    def sample(self):
        if self.template is None:
            raise RuntimeError("initialize the pool before sampling")
        if not self.schedule:
            if self.sampled and self.sampled % self.history_refresh_matches == 0:
                self._rotate_history()
            self.schedule = ["reference"] * 2 + ["history"] * 3 + ["self"] * 2 + ["exploit"] * 13
            self.rng.shuffle(self.schedule)
        name = self.schedule.pop()
        if name == "exploit":
            probabilities = {key: value for key, value in self.probabilities().items() if key not in DEFENSE}
            name = str(self.rng.choice(list(probabilities), p=np.array(list(probabilities.values())) / 0.65))
        seed = int(self.rng.integers(0, 2**31))
        self.sampled += 1
        if name == "reference":
            actor = NeuralMatchActor(self.reference, kind="reference", snapshot_id="reference", seed=seed)
        elif name == "history":
            bucket = self.resident_buckets[int(self.rng.integers(len(self.resident_buckets)))]
            actor = NeuralMatchActor(self.history_models[int(self.rng.choice(bucket))], kind="cfr", snapshot_id="history", seed=seed)
        elif name == "self":
            update = int(self.rng.choice(list(self.self_weights)))
            actor = NeuralMatchActor(self._self_model(update), kind="residual", snapshot_id=update, seed=seed)
        elif name in SLAYER_NAMES:
            actor = SlayerActor(name, seed=seed, equity_samples=self.equity_samples)
        elif name in LADDER_OPPONENTS:
            actor = JitteredLadderReplicaActor(name, seed=seed, equity_samples=self.equity_samples)
        elif name in PRO_OPPONENTS:
            actor = JitteredProScriptActor(name, seed=seed, equity_samples=self.equity_samples)
        else:
            actor = JitteredRuleActor(name, seed=seed, equity_samples=self.equity_samples)
        return name, actor

    poolmod.MatchOpponentPool.sample = sample
    return weights


def bind_anchor(model, anchor):
    import types
    from alphanl.transformer.model import masked_log_probs

    def kl_penalty(self, batch):
        batch = self.prepared(batch)
        logits, _ = self(batch)
        with torch.no_grad():
            anchor_logits, _ = anchor(batch)
        _, logp, probs, legal = masked_log_probs(logits, batch["legal_mask"])
        _, ref_logp, _, _ = masked_log_probs(anchor_logits, batch["legal_mask"])
        kl = (probs * (logp - ref_logp)).sum(-1).clamp_min(0.0)
        progress = ((batch["match_ctx"][:, 0] * 50 - 10) / 40).clamp(0, 1)
        beta = self.settings["beta_kl"] + progress * (self.settings["late_beta_kl"] - self.settings["beta_kl"])
        return (beta * kl).mean(), kl, beta

    model.kl_penalty = types.MethodType(kl_penalty, model)


def train(start):
    weights = install_slayers()
    device = "cuda"
    model, _blob = load_model(start, device)
    anchor = copy.deepcopy(model).eval()
    for param in anchor.parameters():
        param.requires_grad_(False)
    bind_anchor(model, anchor)
    pool = MatchOpponentPool(
        STRATEGIES, seed=45, weights=weights, equity_samples=16,
        history_buckets=((140, 160), (240, 260), (320, 350), (100, 350)),
    )
    pool.initialize(model)
    original = pool.sample

    def sample_anchor():
        name, actor = original()
        if name == "reference":
            actor = NeuralMatchActor(anchor, kind="residual", snapshot_id="rebel_anchor", seed=45)
        return name, actor

    pool.sample = sample_anchor
    opt = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=3e-5)
    evaluator = MatchEvaluator(
        model, match_pairs=25, seed=20261030, equity_samples=32,
        opponents=("equity", "TAG_Pro", "Trapper_CR", "reference"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    matches = 0
    best = None
    for update in range(1, 121):
        began = time.perf_counter()
        print("LADDER_UPDATE", update, flush=True)
        mask = model.settings.get("botzone_native_allin_mask")
        envs = [
            BotzoneMatchEnv(
                model.config, pool=pool, seed=45 + matches + index, first_hero_seat=index % 2,
                defer_opponents=True, context_fields=model.context_fields, allin_mask=mask,
            )
            for index in range(20)
        ]
        rollouts = collect_matches(model, envs)
        metrics = train_matches(model, opt, rollouts, epochs=4, minibatch_size=256, gamma=1.0, gae_lambda=1.0)
        if update % pool.self_snapshot_every == 0:
            pool.add_self_snapshot(model, update=update)
        matches += 20
        row = {"update": update, "loss": metrics.get("loss"), "seconds": round(time.perf_counter() - began, 1)}
        if update % 30 == 0:
            report = evaluator.evaluate(model, update=update, progress=lambda name: print("LADDER_EVAL", update, name, flush=True))
            scores = {
                name: round(report["opponents"][name]["candidate"]["mbb_per_hand"], 1)
                for name in ("equity", "TAG_Pro", "Trapper_CR")
            }
            style = min(scores["TAG_Pro"], scores["Trapper_CR"])
            rank = (style, scores["equity"])
            path = OUT / ("update_%07d.pt" % update)
            save_checkpoint(path, model, update=update, metadata={"scores": scores})
            row["scores"] = scores
            if best is None or rank > best[0]:
                best = (rank, path, scores)
            print("LADDER_EVAL_DONE", update, json.dumps(scores), flush=True)
            model.train()
        print("LADDER_ROW", json.dumps(row), flush=True)
    chosen = best[1]
    print("LADDER_SELECTED", chosen, json.dumps(best[2]), flush=True)
    return chosen, best[2]


def force_keep(full, base):
    return True


def cut(path):
    surgery.beats = force_keep
    decks = surgery.official_decks()
    result = surgery.run_one("ladder", path, decks, "cuda")
    (OUT / "surgery.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    saved = result.get("file")
    print("LADDER_SURGERY", json.dumps({"kept": result.get("kept"), "file": saved}), flush=True)
    return Path(saved) if saved else path


def pack(pt):
    folder = pack_wizard.DELIVER / "V12_LADDER"
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / WEIGHTS
    source = folder / SOURCE_NAME
    blueprint = OUT / "ladder_blueprint.npz"
    # Reuse the rebel exporter's weight splice by calling pack_rebel.export_blueprint.
    import pack_rebel
    pack_rebel.export_blueprint(pt, blueprint)
    with np.load(blueprint, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    meta = json.loads(payload.pop("meta_json").tobytes().decode("utf-8"))
    meta["deployment_release"] = RELEASE
    np.savez(model_path, meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **payload)
    tree = pack_rebel.net_tree()
    entry = (tree / "botzone_entry.py").read_text().replace("V11_MODEL.npz", WEIGHTS).replace(pack_rebel.RELEASE, RELEASE)
    (tree / "botzone_entry.py").write_text(entry)
    pack_rebel.zip_tree(tree, source)
    note = {
        "version": 12, "release": RELEASE, "checkpoint": str(pt),
        "model_sha256": pack_wizard._digest(model_path), "source_sha256": pack_wizard._digest(source),
        "model_bytes": model_path.stat().st_size, "source_bytes": source.stat().st_size,
    }
    (folder / "manifest.json").write_text(json.dumps(note, indent=2) + "\n")
    print("V12_PACK_DONE", json.dumps(note), flush=True)


def main():
    torch.set_num_threads(1)
    start = REBEL if REBEL.is_file() else SURGICAL
    print("LADDER_START", start, flush=True)
    chosen, _scores = train(start)
    final = cut(chosen)
    shutil.copy2(final, OUT / "u250_ladder_slayer.pt")
    pack(final)
    print("V12_DONE", flush=True)


if __name__ == "__main__":
    main()

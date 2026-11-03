"""2500-hand Astra scan on the official seed-20261030 deck.

Direction comes from the first 25 match pairs against TAG_Pro and Trapper_CR.
The grid is scored on pairs 26-50 only. A cut is kept only when the same
weights also beat that model's own untouched scores on all 50 pairs.
The knife touches context_encoder, fusion, and delta_head.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")

from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.evaluation import play_match, summarize_match_pairs
from alphanl.stage2.train import load_checkpoint, save_checkpoint
from alphanl.transformer.tables import standard_deck

OUT = Path("/root/autodl-tmp/sprint24/surgery")
GEN2 = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_best.pt")
U40 = Path("/root/autodl-tmp/stage2_u250_pro_20261103/update_0000040.pt")
KEPT = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt")
SEED = 20261030
OPPONENTS = ("equity", "TAG_Pro", "Trapper_CR")
ALPHAS = (-0.25, -0.15, 0.15, 0.25, 0.4)
KS = (8, 16, 32)
SITES = (
    ("ctx", "context_encoder.0.weight", "ctx"),
    ("fusion", "fusion.0.weight", "fusion"),
    ("delta", "delta_head.weight", "delta"),
)


class Probe:
    def __init__(self):
        self.cache = {}
        self.hooks = []

    def hook_head(self, layer, name):
        def _head(_module, incoming, out):
            act = incoming[0]
            written = out[0] if isinstance(out, tuple) else out
            self.cache[f"{name}.a"] = (act[0] if act.ndim == 2 else act[0, 0]).detach().float().cpu().reshape(-1).clone()
            self.cache[f"{name}.h"] = (written[0] if written.ndim == 2 else written[0, 0]).detach().float().cpu().reshape(-1).clone()

        self.hooks.append(layer.register_forward_hook(_head))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def snapshot(self):
        return {key: value.numpy().copy() for key, value in self.cache.items()}


def direction(bad, good):
    bad_t = torch.tensor(np.stack(bad), dtype=torch.float64)
    good_t = torch.tensor(np.stack(good), dtype=torch.float64)
    raw = bad_t.mean(0) - good_t.mean(0)
    cleaned = raw.clone()
    pcs = 0
    if good_t.shape[0] >= 2:
        centered = good_t - good_t.mean(0, keepdim=True)
        _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        energy = singular.square()
        total = energy.sum().clamp_min(1e-18)
        n_keep = int((torch.cumsum(energy, 0) / total < 0.5).sum().item()) + 1
        pcs = max(1, min(n_keep, 8, vh.shape[0]))
        basis = vh[:pcs]
        cleaned = raw - basis.T @ (basis @ raw)
    used = True
    if float(cleaned.norm()) < 1e-8 or float(cleaned.norm()) < 0.25 * float(raw.norm()):
        cleaned = raw
        used = False
    unit = cleaned / cleaned.norm().clamp_min(1e-12)
    return unit.float(), {"pcs": pcs, "orthogonalized": used, "bad": len(bad), "good": len(good)}


def column_scores(bad_a, good_a, unit, weight):
    delta = torch.tensor(np.stack(bad_a).mean(0) - np.stack(good_a).mean(0), dtype=torch.float32)
    align = weight.detach().float().cpu().T @ unit.cpu()
    if delta.numel() != align.numel():
        raise RuntimeError("activation width %s != columns %s" % (delta.numel(), align.numel()))
    return delta * align


def apply_edit(param, unit, scores, k, alpha):
    positive = torch.nonzero(scores > 0, as_tuple=False).flatten()
    if positive.numel() == 0:
        return 0
    order = positive[torch.argsort(scores[positive], descending=True)]
    chosen = order[: min(k, int(order.numel()))]
    mask = torch.zeros(scores.numel(), dtype=torch.float32)
    mask[chosen] = 1.0
    with torch.no_grad():
        u = unit.to(param.device, param.dtype)
        coeff = mask.to(param.device, param.dtype) * (param.data.T @ u)
        param.data.sub_(alpha * torch.outer(u, coeff))
    return int(chosen.numel())


def backup_of(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore(model, backup):
    with torch.no_grad():
        current = model.state_dict()
        for key, value in backup.items():
            current[key].copy_(value.to(current[key].device))


def official_decks():
    rng = np.random.default_rng(SEED)
    return [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(50)]


def load_model(path, device):
    try:
        return load_checkpoint(path, device=device)
    except Exception as exc:
        print("LOAD_RETRY", path.name, type(exc).__name__, flush=True)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob.pop("astra", None)
        clean = path.with_suffix(".clean.pt")
        torch.save(blob, clean)
        return load_checkpoint(clean, device=device)


def flat(scores):
    return {name: round(scores[name]["mbb_per_hand"], 1) for name in OPPONENTS}


def style_min(scores):
    return min(scores["TAG_Pro"]["mbb_per_hand"], scores["Trapper_CR"]["mbb_per_hand"])


def score(model, decks, pair_offset):
    totals = {name: [] for name in OPPONENTS}
    for local, pair_decks in enumerate(decks):
        pair = pair_offset + local
        legs = {name: [] for name in OPPONENTS}
        for first_seat in (1, 0):
            seed = SEED + pair * 2 + first_seat
            for name in OPPONENTS:
                torch.manual_seed(seed)
                opponent = RuleActor(name, seed=seed, equity_samples=32)
                legs[name].append(play_match(
                    model, opponent, decks=pair_decks, first_hero_seat=first_seat,
                ))
        for name in OPPONENTS:
            totals[name].append(legs[name])
    return {name: summarize_match_pairs(rows) for name, rows in totals.items()}


def collect(model, decks, device):
    probe = Probe()
    probe.hook_head(model.context_encoder[0], "ctx")
    probe.hook_head(model.fusion[0], "fusion")
    probe.hook_head(model.delta_head, "delta")
    raw = []
    try:
        for local, pair_decks in enumerate(decks):
            for first_seat in (1, 0):
                seed = SEED + local * 2 + first_seat
                for name in ("TAG_Pro", "Trapper_CR"):
                    torch.manual_seed(seed)
                    opponent = RuleActor(name, seed=seed, equity_samples=32)
                    from alphanl.stage2.env import BotzoneMatchEnv
                    env = BotzoneMatchEnv(
                        model.config, opponent, decks=pair_decks, first_hero_seat=first_seat,
                        context_fields=model.context_fields,
                        allin_mask=model.settings.get("botzone_native_allin_mask"),
                    )
                    obs = env.reset()
                    pending = []
                    seen = len(env.hands)
                    while not env.done:
                        snap = env._hero_decision_snapshot()
                        street, to_call = int(snap.street), float(snap.to_call)
                        batch = {key: value.to(device) for key, value in obs.items()}
                        with torch.inference_mode():
                            action, *_ = model.act(batch)
                        pending.append({
                            "street": street, "to_call": to_call, "action": int(action),
                            "acts": probe.snapshot(),
                        })
                        obs, _reward, _done, _info = env.step(int(action))
                        if len(env.hands) != seen:
                            raw.append((name, float(env.hands[-1]["hero_net_bb"]), pending))
                            pending = []
                            seen = len(env.hands)
            print("collect", local + 1, "hands_logged", len(raw), flush=True)
    finally:
        probe.close()
    return raw


def split_contrast(raw):
    bad, good = [], []
    for _name, reward, rows in raw:
        tag = None
        if reward <= -20.0:
            tag = "bad"
        elif reward > 0:
            tag = "good"
        if tag is None:
            continue
        bucket = bad if tag == "bad" else good
        bucket.extend(rows)
    rung = "loss20_vs_win"
    if len(bad) < 24 or len(good) < 24:
        bad, good = [], []
        for _name, reward, rows in raw:
            (bad if reward < 0 else good).extend(rows)
        rung = "any_loss_vs_win"
    return bad, good, rung


def prepare(model, bad, good):
    named = dict(model.named_parameters())
    ready = []
    for name, path, key in SITES:
        unit, info = direction([row["acts"][f"{key}.h"] for row in bad], [row["acts"][f"{key}.h"] for row in good])
        weight = named[path]
        scores = column_scores(
            [row["acts"][f"{key}.a"] for row in bad],
            [row["acts"][f"{key}.a"] for row in good],
            unit, weight,
        )
        info = dict(info, name=name, positive=int((scores > 0).sum()), columns=int(scores.numel()))
        ready.append({"name": name, "path": path, "unit": unit, "scores": scores, "info": info})
    return ready


def edit(model, sites, alpha, k):
    named = dict(model.named_parameters())
    return {site["name"]: apply_edit(named[site["path"]], site["unit"], site["scores"], k, alpha) for site in sites}


def beats(after, before):
    return style_min(after) > style_min(before) and after["equity"]["mbb_per_hand"] > before["equity"]["mbb_per_hand"]


def run_one(label, path, decks, device):
    print("SURGERY", label, flush=True)
    model, blob = load_model(path, device)
    model.eval()
    raw = collect(model, decks[:25], device)
    bad, good, rung = split_contrast(raw)
    print(json.dumps({"model": label, "bad": len(bad), "good": len(good), "rung": rung, "hands": len(raw)}), flush=True)
    if len(bad) < 24 or len(good) < 24:
        return {"label": label, "kept": False, "reason": "contrast too small", "bad": len(bad), "good": len(good)}
    sites = prepare(model, bad, good)
    for site in sites:
        print(label, site["name"], json.dumps(site["info"]), flush=True)
    backup = backup_of(model)
    blind_base = score(model, decks[25:], 25)
    print("BLIND_BASE", label, json.dumps(flat(blind_base)), flush=True)
    rows = []
    for alpha in ALPHAS:
        for k in KS:
            restore(model, backup)
            touched = edit(model, sites, alpha, k)
            blind = score(model, decks[25:], 25)
            row = {
                "alpha": alpha, "k": k, "columns": touched, "blind": flat(blind),
                "style_min": style_min(blind), "equity": blind["equity"]["mbb_per_hand"],
                "scores": blind,
            }
            rows.append(row)
            print("GRID", label, alpha, k, json.dumps(flat(blind)), flush=True)
            (OUT / f"{label}_grid.json").write_text(json.dumps({
                "blind_base": flat(blind_base), "rows": [
                    {key: value for key, value in item.items() if key != "scores"} for item in rows
                ],
            }, indent=2) + "\n")
    restore(model, backup)
    protected = [row for row in rows if row["equity"] >= blind_base["equity"]["mbb_per_hand"] - 150]
    winner = max(protected or rows, key=lambda row: (row["style_min"], row["equity"]))
    edit(model, sites, float(winner["alpha"]), int(winner["k"]))
    full = score(model, decks, 0)
    restore(model, backup)
    full_base = score(model, decks, 0)
    print("FULL", label, "cut", json.dumps(flat(full)), "base", json.dumps(flat(full_base)), flush=True)
    kept = beats(full, full_base)
    dest = None
    if kept:
        edit(model, sites, float(winner["alpha"]), int(winner["k"]))
        dest = OUT / f"{label}_surgical.pt"
        save_checkpoint(dest, model, update=int(blob.get("update", 0)), metadata={
            "astra": {
                "alpha": winner["alpha"], "k": winner["k"], "rung": rung,
                "knife": "context_encoder,fusion,delta_head",
                "source": str(path),
            },
            "full_base": flat(full_base), "full": flat(full),
        })
    return {
        "label": label, "source": str(path), "bad": len(bad), "good": len(good), "rung": rung,
        "sites": [site["info"] for site in sites], "blind_base": flat(blind_base),
        "winner": {key: winner[key] for key in ("alpha", "k", "blind", "style_min", "equity")},
        "full": flat(full), "full_base": flat(full_base), "kept": kept, "file": None if dest is None else str(dest),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = "cuda"
    decks = official_decks()
    jobs = [("u40", U40)]
    if GEN2.is_file():
        jobs.append(("gen2", GEN2))
    results = [run_one(label, path, decks, device) for label, path in jobs]
    kept = [row for row in results if row.get("kept")]
    chosen = None
    if kept:
        chosen = max(kept, key=lambda row: (min(row["full"]["TAG_Pro"], row["full"]["Trapper_CR"]), row["full"]["equity"]))
        src = Path(chosen["file"])
        KEPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(src, map_location="cpu", weights_only=False), KEPT)
        chosen["saved"] = str(KEPT)
    h2h = None
    beats_v8 = False
    if chosen is not None:
        import gen2_train as g
        from alphanl.transformer.tables import standard_deck
        from alphanl.stage2.evaluation import summarize_match_pairs
        hero, _ = load_model(Path(chosen["saved"]), device)
        u40, _ = load_model(U40, device)
        hero.eval()
        u40.eval()
        rng = np.random.default_rng(SEED)
        decks = [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(50)]
        print("H2H surgery u40", flush=True)
        forward = g.h2h_direction(hero, u40, decks, SEED)
        reverse = g.h2h_direction(u40, hero, decks, SEED + 100000)
        both = g.combined(forward, reverse)
        h2h = {"mbb_per_hand": both["mean"], "mbb_ci95": list(both["ci95"]),
               "left_as_hero": summarize_match_pairs(forward), "right_as_hero": summarize_match_pairs(reverse)}
        base_path = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/confirm_5000.json")
        if base_path.is_file():
            base = json.loads(base_path.read_text())["scores"]["u40"]
            base_style = min(base["TAG_Pro"]["mbb_per_hand"], base["Trapper_CR"]["mbb_per_hand"])
            base_equity = base["equity"]["mbb_per_hand"]
        else:
            base_style = min(chosen["full_base"]["TAG_Pro"], chosen["full_base"]["Trapper_CR"])
            base_equity = chosen["full_base"]["equity"]
        style = min(chosen["full"]["TAG_Pro"], chosen["full"]["Trapper_CR"])
        beats_v8 = bool(chosen["full"]["equity"] > 1000 and style > base_style and h2h["mbb_per_hand"] > 0)
    payload = {"kept": bool(kept), "chosen": chosen, "h2h_u40": h2h, "beats_v8": beats_v8, "results": results}
    (OUT / "surgery_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("SURGERY_DONE", json.dumps({"kept": bool(kept), "beats_v8": beats_v8,
                                      "chosen": None if chosen is None else chosen["label"]}), flush=True)


if __name__ == "__main__":
    main()

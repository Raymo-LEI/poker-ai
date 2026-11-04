"""Versioned public CFV contract. NumPy only; values are zero-sum root BB.

0:32 hero bucket mass; 32:64 opponent bucket mass; 64:116 board bits.
The remaining named features are normalized public state, never placeholders.
Buckets contain all board-legal combinations, uniformly within each bucket.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SCHEMA = "public32-combo136-res3-bb-v3"
BUCKETS = 32
FEATURE_DIM = 136
TAIL_NAMES = (
    "hero_stack_200bb", "opponent_stack_200bb", "effective_spr_20",
    "pot_200bb", "call_pot", "street_3", "board_count_5",
    "effective_stack_200bb", "call_odds", "hero_first_on_new_street",
    "paired_board", "trips_board", "max_suit_fraction", "unique_ranks_5",
    "highest_rank_14", "lowest_rank_14", "rank_span_12", "broadway_fraction",
    "hero_entropy_log32", "opponent_entropy_log32",
)
DECK = tuple(rank + suit for suit in "shdc" for rank in "23456789TJQKA")
CARD_ID = {card: i for i, card in enumerate(DECK)}
CALLS = {"cfv_river.": 0, "cfv_turn.": 0}


def card(value):
    t = str(value).strip().upper()
    if len(t) != 2:
        raise ValueError("invalid card")
    out = t[1] + t[0].lower() if t[0] in "SHDC" else t[0] + t[1].lower()
    if out not in CARD_ID:
        raise ValueError("invalid card " + t)
    return out


def normalize_range(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (32,) or not np.isfinite(a).all() or np.any(a < 0) or a.sum() <= 0:
        raise ValueError("expected nonnegative nonempty 32-bucket range")
    return a / a.sum()


def spot_features(board, r_hero, r_opp, pot, call, stack, blind, street,
                  opp_stack=None, first_player=0):
    board = [card(c) for c in board]
    if len(set(board)) != len(board) or len(board) not in (3, 4, 5):
        raise ValueError("invalid public board")
    rh, ro = normalize_range(r_hero), normalize_range(r_opp)
    pot, call, stack, blind = map(float, (pot, call, stack, blind))
    os_ = stack if opp_stack is None else float(opp_stack)
    if pot <= 0 or blind <= 0 or min(call, stack, os_) < 0:
        raise ValueError("invalid chips")
    ranks = ["23456789TJQKA".index(c[0]) + 2 for c in board]
    suits = [c[1] for c in board]
    entropy = lambda r: -np.sum(r * np.log(np.maximum(r, 1e-300))) / np.log(32.)
    out = np.zeros(136, dtype=np.float64)
    out[:32], out[32:64] = rh, ro
    for c in board:
        out[64 + CARD_ID[c]] = 1
    out[116:] = (
        stack / (200 * blind), os_ / (200 * blind), min(min(stack, os_) / pot, 20) / 20,
        pot / (200 * blind), call / pot, street / 3., len(board) / 5.,
        min(stack, os_) / (200 * blind), min(call, stack) / (pot + min(call, stack)),
        float(first_player == 0), float(len(set(ranks)) < len(ranks)),
        float(max(ranks.count(r) for r in ranks) >= 3),
        max(suits.count(s) for s in suits) / len(board), len(set(ranks)) / 5.,
        max(ranks) / 14., min(ranks) / 14., (max(ranks) - min(ranks)) / 12.,
        sum(r >= 10 for r in ranks) / len(board), entropy(rh), entropy(ro),
    )
    return out


def gelu(x):
    # Vectorized erf approximation (max erf error <1.5e-7), checked against
    # Torch's exact GELU at export. Topology and BB output contract are unchanged.
    y=x/np.sqrt(2.)
    t=1./(1.+.3275911*np.abs(y))
    poly=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t
    erf=np.sign(y)*(1.-poly*np.exp(-y*y))
    return .5*x*(1.+erf)


def validate_weights(weights, prefixes=("cfv_river.", "cfv_turn.")):
    if "cfv_schema" not in weights or bytes(np.asarray(weights["cfv_schema"], dtype=np.uint8)).decode() != SCHEMA:
        raise ValueError("CFV schema mismatch or missing")
    for prefix in prefixes:
        for i, shape in enumerate(((256,136), (256,256), (256,256), (256,256), (32,256))):
            w, b = weights[prefix + "%d.weight" % i], weights[prefix + "%d.bias" % i]
            if w.shape != shape or b.shape != (shape[0],) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("invalid CFV layer " + prefix + str(i))
    return weights


def predict_residual_cfv(weights, prefix, feat):
    x = np.asarray(feat, dtype=np.float64)
    if x.shape[-1:] != (136,) or not np.isfinite(x).all():
        raise ValueError("CFV feature schema mismatch")
    hidden = gelu(x @ weights[prefix + "0.weight"].T + weights[prefix + "0.bias"])
    for i in (1,2,3):
        hidden = hidden + gelu(hidden @ weights[prefix + "%d.weight" % i].T + weights[prefix + "%d.bias" % i])
    CALLS[prefix] += int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    return hidden @ weights[prefix + "4.weight"].T + weights[prefix + "4.bias"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(paths=None):
    paths = paths or [Path(__file__), Path(__file__).with_name("deep_solver_v2.py"),
                      Path(__file__).resolve().parents[3] / "scripts/sprint24/evening_pipeline.py"]
    return {Path(p).name: digest(p) for p in paths}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


def publish(path, payload, samples, epochs, kind, extra=None):
    path = Path(path)
    if "x" in payload and (len(payload["x"]) != int(samples) or not np.isfinite(payload["x"]).all()):
        raise ValueError("dataset sample count or features invalid")
    if "y" in payload and (payload["y"].shape != (int(samples),32) or not np.isfinite(payload["y"]).all()):
        raise ValueError("invalid CFV targets")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))
    manifest = dict(arrays_sha256=array_digest(payload), schema=SCHEMA, samples=int(samples), epochs=int(epochs), kind=kind,
                    units="zero_sum_root_bb", complete=True, sha256=digest(path), sources=source_hashes())
    manifest.update(extra or {})
    atomic_json(str(path) + ".manifest.json", manifest)
    return manifest


def load_artifact(path, kind=None, min_samples=1, min_epochs=0):
    path = Path(path)
    m = json.loads(Path(str(path) + ".manifest.json").read_text())
    if (m.get("schema") != SCHEMA or m.get("units") != "zero_sum_root_bb" or
        not m.get("complete") or m.get("sha256") != digest(path) or
        m.get("samples",0) < min_samples or m.get("epochs",0) < min_epochs or
        m.get("sources") != source_hashes() or (kind and m.get("kind") != kind)):
        raise ValueError("stale, incomplete or incompatible CFV artifact: " + str(path))
    with np.load(path, allow_pickle=False) as blob:
        arrays = {k: np.array(blob[k]) for k in blob.files}
    if m.get("arrays_sha256") != array_digest(arrays):
        raise ValueError("CFV array digest mismatch")
    return arrays, m


def load_nets(path):
    arrays, manifest = load_artifact(path, "nets", min_epochs=1)
    validate_weights(arrays)
    if (manifest.get("river_samples",0)<=0 or manifest.get("turn_samples",0)<=0 or
        manifest.get("river_samples",0)+manifest.get("turn_samples",0)!=manifest["samples"]):
        raise ValueError("CFV training sample manifest mismatch")
    return arrays, manifest


def array_digest(weights):
    h=hashlib.sha256()
    for k in sorted(weights):
        if not k.startswith("cfv_") and not any(x in weights for x in ("x", "target")):
            continue
        v=np.ascontiguousarray(weights[k])
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(repr(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def validate_embedded(weights,manifest):
    validate_weights(weights)
    if (manifest.get("schema")!=SCHEMA or manifest.get("units")!="zero_sum_root_bb" or
        not manifest.get("complete") or manifest.get("epochs",0)<1 or manifest.get("samples",0)<1 or
        manifest.get("arrays_sha256")!=array_digest(weights)):
        raise ValueError("embedded CFV manifest mismatch")
    from . import deep_solver_v2
    for module in (deep_solver_v2,):
        source=module.__loader__.get_data(module.__file__)
        if hashlib.sha256(source).hexdigest()!=manifest["sources"]["deep_solver_v2.py"]:
            raise ValueError("embedded solver source mismatch")
    source=globals()["__loader__"].get_data(__file__)
    if hashlib.sha256(source).hexdigest()!=manifest["sources"]["cfv.py"]:
        raise ValueError("embedded CFV source mismatch")

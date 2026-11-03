"""Re-apply the gen2 knife alpha=-0.15, k=16 and save it.

The 5000-hand cut was 1958.1 / -50.6 / -0.4. Direction is collected again on
the same first 25 pairs, then that one edit is kept. Same-deck H2H is against
the uncut V9 gen2 checkpoint. A stronger Stage-3 model gets the same alpha/k
on its own direction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

import gen2_train as g
import surgery as s
from alphanl.stage2.evaluation import summarize_match_pairs
from alphanl.transformer.tables import standard_deck

GEN2 = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_best.pt")
KEPT = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_surgical.pt")
NATIVE_RESULT = Path("/root/autodl-tmp/stage2_native_v2_20261103/native_result.json")
NATIVE_OUT = Path("/root/autodl-tmp/stage2_native_v2_20261103/native_v2_surgical.pt")
REPORT = Path("/root/autodl-tmp/sprint24/surgery/reapply_knife.json")
ALPHA = -0.15
K = 16
V9_EQUITY = 2056.3
V9_TAG = -171.9
V9_TRAP = -83.4


def cut(path, device):
    model, blob = s.load_model(path, device)
    model.eval()
    decks = s.official_decks()
    raw = s.collect(model, decks[:25], device)
    bad, good, rung = s.split_contrast(raw)
    print(json.dumps({"source": path.name, "bad": len(bad), "good": len(good), "rung": rung}), flush=True)
    if len(bad) < 24 or len(good) < 24:
        raise RuntimeError("contrast too small to reapply the knife")
    sites = s.prepare(model, bad, good)
    touched = s.edit(model, sites, ALPHA, K)
    print("EDIT", path.name, touched, flush=True)
    return model, blob, rung, [site["info"] for site in sites]


def h2h(left, right, label):
    rng = np.random.default_rng(s.SEED)
    decks = [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(50)]
    print("H2H", label, flush=True)
    forward = g.h2h_direction(left, right, decks, s.SEED)
    reverse = g.h2h_direction(right, left, decks, s.SEED + 100000)
    both = g.combined(forward, reverse)
    return {
        "combined_left_view": {"mbb_per_hand": both["mean"], "mbb_ci95": list(both["ci95"])},
        "left_as_hero": summarize_match_pairs(forward),
        "right_as_hero": summarize_match_pairs(reverse),
    }


def native_is_stronger():
    if not NATIVE_RESULT.is_file():
        return False, None
    payload = json.loads(NATIVE_RESULT.read_text())
    verdict = payload["verdict"]
    style = min(verdict["native_tag"], verdict["native_trapper"])
    equity = verdict["native_equity"]
    three = equity + verdict["native_tag"] + verdict["native_trapper"]
    v9_three = V9_EQUITY + V9_TAG + V9_TRAP
    stronger = bool(equity > 1000 and style > min(V9_TAG, V9_TRAP) and three > v9_three)
    return stronger, {"equity": equity, "tag": verdict["native_tag"], "trapper": verdict["native_trapper"],
                      "style_min": style, "three": three, "checkpoint": payload["checkpoint"], "stronger": stronger}


def main():
    torch.set_num_threads(1)
    device = "cuda"
    model, blob, rung, sites = cut(GEN2, device)
    s.save_checkpoint(KEPT, model, update=int(blob.get("update", 40)), metadata={
        "astra": {"alpha": ALPHA, "k": K, "rung": rung, "knife": "context_encoder,fusion,delta_head",
                  "source": str(GEN2), "known_full_5000": {"equity": 1958.1, "TAG_Pro": -50.6, "Trapper_CR": -0.4}},
        "sites": sites,
    })
    print("SAVED", KEPT, flush=True)
    uncut, _ = s.load_model(GEN2, device)
    model.eval()
    uncut.eval()
    versus = h2h(model, uncut, "gen2_surgical_vs_v9")
    report = {"gen2_surgical": str(KEPT), "alpha": ALPHA, "k": K, "h2h_vs_v9": versus}
    stronger, native_scores = native_is_stronger()
    report["native"] = native_scores
    if stronger:
        native_model, native_blob, native_rung, native_sites = cut(Path(native_scores["checkpoint"]), device)
        s.save_checkpoint(NATIVE_OUT, native_model, update=int(native_blob.get("update", 0)), metadata={
            "astra": {"alpha": ALPHA, "k": K, "rung": native_rung, "source": native_scores["checkpoint"]},
            "sites": native_sites,
        })
        base, _ = s.load_model(Path(native_scores["checkpoint"]), device)
        native_model.eval()
        base.eval()
        report["native_surgical"] = str(NATIVE_OUT)
        report["h2h_native_surgical_vs_uncut"] = h2h(native_model, base, "native_surgical_vs_uncut")
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("KNIFE_DONE", json.dumps({
        "saved": str(KEPT),
        "h2h_vs_v9": versus["combined_left_view"]["mbb_per_hand"],
        "native_cut": stronger,
    }), flush=True)


if __name__ == "__main__":
    main()

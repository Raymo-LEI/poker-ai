"""Score the rebel net against a second search pass, then pack V11 unconditionally."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/sprint24")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/src")
sys.path.insert(0, "/root/autodl-tmp/stage2_botzone_native_code_20261030/scripts")

import eval_wizard
import pack_wizard
from surgery import load_model

OUT = Path("/root/autodl-tmp/sprint24/rebel")
REBEL_PT = OUT / "u250_rebel_blueprint.pt"
RELEASE = "V11_R11_REBEL_20261104"
WEIGHTS = "V11_MODEL.npz"
SOURCE = "V11_SOURCE.zip"
C200 = Path("/root/autodl-tmp/stage2_champion_v2_cand200_20261029/checkpoints/champion_stage2_v2_cand200.pt")
V9 = Path("/root/autodl-tmp/stage2_u250_gen2_20261103/u250_gen2_best.pt")


def export_blueprint(pt, dest):
    model, _blob = load_model(pt, "cpu")
    fresh = {}
    for key, value in model.state_dict().items():
        if key.startswith("reference.backbone."):
            key = key[len("reference.backbone."):]
        fresh[key] = np.ascontiguousarray(value.detach().cpu().numpy())
    with np.load(pack_wizard.V9_NPZ, allow_pickle=False) as archive:
        meta = np.array(archive["meta_json"])
        base = {key: np.array(archive[key]) for key in archive.files if key != "meta_json"}
    replaced = 0
    for key, value in fresh.items():
        if key not in base or base[key].shape != value.shape:
            continue
        base[key] = value
        replaced += 1
    if replaced < 10:
        raise RuntimeError("rebel export replaced only %d tensors" % replaced)
    np.savez(dest, meta_json=meta, **{key: np.ascontiguousarray(value) for key, value in base.items()})
    print("REBEL_NPZ", dest, replaced, flush=True)
    return dest


def zip_tree(tree, dest):
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(str(path.relative_to(tree)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())


def net_tree():
    """V9 runtime with the V11 names and no online search."""
    tree = pack_wizard.prepare_tree(11, RELEASE, WEIGHTS)
    runtime = (tree / "stage2_runtime" / "runtime.py").read_text()
    runtime = runtime.replace(pack_wizard.ACT_NEW, pack_wizard.ACT_OLD, 1)
    (tree / "stage2_runtime" / "runtime.py").write_text(runtime)
    entry = (tree / "botzone_entry.py").read_text()
    (tree / "botzone_entry.py").write_text(entry.replace(pack_wizard.NOTE_NEW, pack_wizard.NOTE_OLD, 1))
    return tree


def wizard_exam(blueprint):
    model_path = OUT / "rebel_wizard.npz"
    pack_wizard.write_model(blueprint, Path("/root/autodl-tmp/sprint24/wizard/vleaf.npz"), model_path, RELEASE)
    tree = pack_wizard.prepare_tree(11, RELEASE, WEIGHTS)
    sys.path.insert(0, str(tree))
    agent = eval_wizard.load_agent(model_path)
    agent.force_argmax = False
    torch_model, _ = load_model(REBEL_PT, "cuda")
    torch_model.eval()
    eval_wizard.attach_config(agent, torch_model)
    decks = eval_wizard.decks()
    times = []
    scores = eval_wizard.exam(agent, decks, times)
    v9, _ = load_model(V9, "cuda")
    v9.eval()
    versus_v9 = eval_wizard.h2h(agent, v9, decks, "cuda", times)
    c200, _ = load_model(C200, "cuda")
    c200.eval()
    versus_c200 = eval_wizard.h2h(agent, c200, decks, "cuda", times)
    result = {
        "equity": round(scores["equity"]["mbb_per_hand"], 1),
        "tag": round(scores["TAG_Pro"]["mbb_per_hand"], 1),
        "trapper": round(scores["Trapper_CR"]["mbb_per_hand"], 1),
        "h2h_v9": round(versus_v9["mbb_per_hand"], 1),
        "h2h_c200": round(versus_c200["mbb_per_hand"], 1),
        "p99_ms": round(eval_wizard.percentile(times, 99), 2),
        "mean_ms": round(eval_wizard.percentile(times, 50), 2),
    }
    result["style_min"] = round(min(result["tag"], result["trapper"]), 1)
    print("REBEL_WIZARD", json.dumps(result), flush=True)
    return result


def ship(kind, blueprint, tree_builder):
    folder = pack_wizard.DELIVER / "V11_REBEL"
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / WEIGHTS
    source = folder / SOURCE
    if kind == "wizard":
        pack_wizard.write_model(blueprint, Path("/root/autodl-tmp/sprint24/wizard/vleaf.npz"), model, RELEASE)
        tree = pack_wizard.prepare_tree(11, RELEASE, WEIGHTS)
    else:
        with np.load(blueprint, allow_pickle=False) as archive:
            payload = {key: np.array(archive[key]) for key in archive.files}
        meta = json.loads(payload.pop("meta_json").tobytes().decode("utf-8"))
        meta["deployment_release"] = RELEASE
        meta["wizard"] = {"enabled": False}
        np.savez(model, meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **payload)
        tree = tree_builder()
    zip_tree(tree, source)
    note = {
        "version": 11, "release": RELEASE, "name": "rebel", "kind": kind,
        "weights": WEIGHTS, "source": SOURCE,
        "model_sha256": pack_wizard._digest(model), "source_sha256": pack_wizard._digest(source),
        "model_bytes": model.stat().st_size, "source_bytes": source.stat().st_size,
    }
    (folder / "manifest.json").write_text(json.dumps(note, indent=2) + "\n")
    print("V11_PACK_DONE", json.dumps(note), flush=True)
    return note


def main():
    torch.set_num_threads(1)
    net = json.loads((OUT / "rebel_net.json").read_text())
    net_rank = (min(net["TAG_Pro"], net["Trapper_CR"]), net["equity"] + net["TAG_Pro"] + net["Trapper_CR"])
    blueprint = export_blueprint(REBEL_PT, OUT / "rebel_blueprint.npz")
    wizard = wizard_exam(blueprint)
    wiz_rank = (wizard["style_min"], wizard["equity"] + wizard["tag"] + wizard["trapper"])
    kind = "wizard" if wiz_rank > net_rank else "net"
    print("V11_CHOICE", kind, "net", json.dumps(net), "wizard", json.dumps(wizard), flush=True)
    note = ship(kind, blueprint, net_tree)
    payload = {"choice": kind, "net": net, "wizard": wizard, "pack": note}
    (OUT / "v11_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("V11_DONE", kind, flush=True)


if __name__ == "__main__":
    main()

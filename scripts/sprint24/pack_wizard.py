"""Assemble V10_WIZARD from the frozen blueprint npz plus the leaf head."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

DELIVER = Path("/root/autodl-tmp/sprint24/deliver")
V9_ZIP = DELIVER / "V9_gen2" / "V9_SOURCE.zip"
V9_NPZ = DELIVER / "V9_gen2" / "V9_MODEL.npz"
OUT = Path("/root/autodl-tmp/sprint24/wizard")
TREE = OUT / "src"
WIZARD_PY = Path("/root/autodl-tmp/sprint24/wizard.py")

ACT_OLD = (
    "        probs, self.last_guard = apply_guard(snapshot, probs, self.deployment_guard, "
    "estimated_equity, amounts, context=context)\n"
    "        rng = random if rng is None else rng\n"
)
ACT_NEW = (
    "        probs, self.last_guard = apply_guard(snapshot, probs, self.deployment_guard, "
    "estimated_equity, amounts, context=context)\n"
    "        from .wizard import fuse as _wizard_fuse\n"
    "        probs = _wizard_fuse(self, snapshot, context, probs, amounts)\n"
    "        if getattr(self, \"force_argmax\", False):\n"
    "            return int(np.argmax(probs))\n"
    "        rng = random if rng is None else rng\n"
)
NOTE_OLD = (
    "        if agent.deployment_guard[\"enabled\"]:\n"
    "            note += \" guard=\" + agent.deployment_guard[\"version\"] + \" \" + "
    "json.dumps(agent.last_guard, separators=(\",\", \":\"))\n"
)
NOTE_NEW = NOTE_OLD + (
    "        if getattr(agent, \"last_wizard\", \"\"):\n"
    "            note += \" \" + agent.last_wizard\n"
)


def _meta(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["meta_json"].tobytes().decode("utf-8"))


def prepare_tree(version, release, weights_name):
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir(parents=True)
    with zipfile.ZipFile(V9_ZIP) as archive:
        archive.extractall(TREE)
    runtime = (TREE / "stage2_runtime" / "runtime.py").read_text()
    if ACT_OLD not in runtime:
        raise RuntimeError("blueprint act() no longer matches the V9 source")
    (TREE / "stage2_runtime" / "runtime.py").write_text(runtime.replace(ACT_OLD, ACT_NEW, 1))
    (TREE / "stage2_runtime" / "wizard.py").write_text(WIZARD_PY.read_text())
    entry = (TREE / "botzone_entry.py").read_text()
    entry = entry.replace('WEIGHTS_NAME = "V9_MODEL.npz"', 'WEIGHTS_NAME = "%s"' % weights_name, 1)
    entry = entry.replace("RELEASE_ID = 'V9_R9_GEN2_20261103'", "RELEASE_ID = '%s'" % release, 1)
    entry = entry.replace("``V9_MODEL.npz``", "``%s``" % weights_name)
    if NOTE_OLD not in entry:
        raise RuntimeError("blueprint entry debug no longer matches the V9 source")
    (TREE / "botzone_entry.py").write_text(entry.replace(NOTE_OLD, NOTE_NEW, 1))
    return TREE


def write_model(blueprint_npz, leaf_npz, destination, release):
    with np.load(blueprint_npz, allow_pickle=False) as archive:
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        weights = {key: np.ascontiguousarray(archive[key]) for key in archive.files if key != "meta_json"}
    before = {key: hashlib.sha256(weights[key].tobytes()).hexdigest() for key in weights}
    with np.load(leaf_npz, allow_pickle=False) as leaf:
        for key in leaf.files:
            weights[key] = np.ascontiguousarray(leaf[key])
    after = {key: hashlib.sha256(weights[key].tobytes()).hexdigest() for key in before}
    if after != before:
        raise RuntimeError("blueprint arrays changed while attaching the leaf")
    meta["deployment_release"] = release
    meta["wizard"] = {"leaf": "vleaf", "search": "street_cfr_kl", "buckets": 8, "tau_bb": 4.0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez(handle, meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **weights)
    return meta


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(result):
    state_path = DELIVER / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"shipped": []}
    version = 10
    release = "V10_R10_WIZARD_20261104"
    weights_name = "V10_WIZARD_MODEL.npz"
    folder = DELIVER / "V10_WIZARD"
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / weights_name
    source = folder / "V10_WIZARD_SOURCE.zip"
    write_model(Path(result["blueprint_npz"]), OUT / "vleaf.npz", model, release)
    tree = prepare_tree(version, release, weights_name)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(str(path.relative_to(tree)), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
    note = {
        "version": version, "release": release, "name": "wizard",
        "checkpoint": result["checkpoint"], "equity": result["equity"], "tag": result["tag"],
        "trapper": result["trapper"], "style_min": result["style_min"],
        "h2h_u40": result["h2h_v9"], "h2h_label": "V9_R9_GEN2",
        "h2h_c200": result["h2h_c200"], "p99_ms": result["p99_ms"],
        "model_sha256": _digest(model), "source_sha256": _digest(source),
        "model_bytes": model.stat().st_size, "source_bytes": source.stat().st_size,
    }
    (folder / "manifest.json").write_text(json.dumps(note, indent=2) + "\n")
    state.setdefault("shipped", []).append(note)
    state["latest"] = note
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    (DELIVER / "LATEST.json").write_text(json.dumps(note, indent=2) + "\n")
    print("WIZARD_PACK_DONE", json.dumps(note), flush=True)
    return note

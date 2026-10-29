"""Recover frozen evaluation-only policies from a retained self-play bank.

Reference tensors are immutable throughout Stage 2. Historical bank entries
store every non-reference policy tensor, so combining them is exact. These
artifacts deliberately omit optimizer, RNG and pool state and cannot resume
the original training experiment.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from alphanl.cfr.gauntlet import sha256
from alphanl.stage2.train import load_checkpoint


def extract(source, out, updates):
    model, blob = load_checkpoint(source)
    bank = blob.get("pool", {}).get("self_weights", {})
    expected = {k for k in model.state_dict() if not k.startswith("reference.")}
    if any(update not in bank or set(bank[update]) != expected for update in updates):
        raise ValueError("self-play bank lacks the exact complete requested update")
    if any(update > blob["update"] for update in updates):
        raise ValueError("historical update exceeds source update")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(source=str(Path(source).resolve()), source_sha256=sha256(source),
                      source_update=blob["update"], artifacts={})
    for update in updates:
        weights = {k: v.clone() for k, v in blob["model"].items() if k.startswith("reference.")}
        weights.update(bank[update])
        artifact = {k: blob[k] for k in ("format", "config", "settings", "rules_version", "numeric_version", "context_fields")}
        artifact.update(model=weights, update=update, metadata=dict(evaluation_only=True,
            extraction_source_sha256=provenance["source_sha256"], source_update=blob["update"],
            source_self_snapshot=update, reference_immutable=True))
        path = out / f"update_{update:07d}.pt"
        torch.save(artifact, path)
        recovered, _ = load_checkpoint(path)
        for k, tensor in recovered.state_dict().items():
            torch.testing.assert_close(tensor, weights[k], rtol=0, atol=0)
        if update == blob["update"]:
            for k, tensor in recovered.state_dict().items():
                torch.testing.assert_close(tensor, blob["model"][k], rtol=0, atol=0)
        provenance["artifacts"][str(update)] = dict(path=path.name, sha256=sha256(path))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--updates", nargs="+", type=int, default=[100, 200, 300])
    args = p.parse_args()
    print(json.dumps(extract(args.checkpoint, args.out_dir, args.updates)), flush=True)


if __name__ == "__main__":
    main()

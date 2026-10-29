"""Change deployment metadata without Torch or any modification of weights.

Example: python configure_botzone_guard.py --zip botzone_source.zip
  --weights stage2_model.npz --config guard_config.json --out configured.npz
Upload the resulting file with the user-storage name stage2_model.npz.
"""
import argparse
import json
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.zip)
    from stage2_runtime.deployment_guard import guard_config
    with open(args.config) as handle:
        config = guard_config(json.load(handle))
    with np.load(args.weights, allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    meta = json.loads(weights["meta_json"].tobytes().decode("utf8"))
    if meta.get("format") != "stage2_botzone_v1":
        raise ValueError("not a Stage 2 Botzone weight archive")
    meta["deployment_guard"] = config
    weights["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf8"), dtype=np.uint8)
    with open(args.out, "wb") as handle:
        np.savez(handle, **weights)
    print(json.dumps({"path":args.out,"deployment_guard":config}, indent=2))


if __name__ == "__main__":
    main()

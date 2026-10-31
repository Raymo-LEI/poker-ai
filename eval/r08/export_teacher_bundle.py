"""Write an R08 bundle manifest from the export JSON only.

Does not copy checkpoints. A non-empty ``teacher_out_dir`` and ``--execute``
both exit 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from build_selection_report import build, validate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        print("refusing --execute: R08 prep cannot export weights or start training", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate(manifest)
        selection = build(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"refusing manifest: {exc.__class__.__name__}", file=sys.stderr)
        return 2
    bundle = {
        "format": "route1_r08_teacher_bundle_v1",
        "status": "PENDING_TEACHER",
        "weights": [],
        "weights_read": False,
        "files_copied": 0,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selection": selection,
        "metrics": "NOT_RUN",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PENDING_TEACHER", "weights_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

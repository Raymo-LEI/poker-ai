"""Hash and schema-check an R08 export manifest. Do not read weights.

``--execute`` is refused. A manifest that claims a pass, includes weights, or
carries a strength number is rejected. The written report stays
``PENDING_TEACHER``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _reject_execute(enabled: bool) -> None:
    if enabled:
        print("refusing --execute: R08 prep cannot start training", file=sys.stderr)
        raise SystemExit(2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(manifest: dict) -> None:
    if manifest.get("format") != "route1_r08_export_from_teacher_v1":
        raise ValueError("unexpected export manifest format")
    if manifest.get("status") != "PENDING_TEACHER" or manifest.get("teacher_gate") == "PASSED":
        raise ValueError("prep stub cannot certify a teacher")
    if manifest.get("weights_included") or manifest.get("distill_authorized") or manifest.get("strength_claim"):
        raise ValueError("weights, distill, and strength claims are closed")
    if manifest.get("teacher_out_dir"):
        raise ValueError("refusing a non-empty teacher_out_dir")
    schema = manifest.get("action_schema_id")
    names = tuple(manifest.get("action_names") or ())
    width = manifest.get("num_actions")
    if schema == "botzone_a6_v2":
        expected = ("fold", "check_call", "half_pot", "pot", "all_in", "quarter_pot")
        if width != 6 or names != expected or not manifest.get("quarter_pot_required"):
            raise ValueError("A6 export dropped quarter_pot")
    elif schema == "botzone_a5_v2":
        expected = ("fold", "check_call", "half_pot", "pot", "all_in")
        if width != 5 or names != expected or manifest.get("quarter_pot_required"):
            raise ValueError("A5 export does not match the five-action table")
    else:
        raise ValueError("schema must be botzone_a5_v2 or botzone_a6_v2")
    for key, value in manifest.items():
        if "mbb" in key.lower() and value is not None:
            raise ValueError("refusing a strength number")


def build(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate(manifest)
    return {
        "format": "route1_r08_selection_report_v1",
        "status": "PENDING_TEACHER",
        "teacher_gate": "NOT_PASSED",
        "selected_teacher": None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "action_schema_id": manifest["action_schema_id"],
        "num_actions": manifest["num_actions"],
        "quarter_pot_required": bool(manifest["quarter_pot_required"]),
        "teacher_manifest_hint": manifest.get("teacher_manifest_hint", ""),
        "weights_read": False,
        "strength_claim": False,
        "metrics": "NOT_RUN",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    _reject_execute(args.execute)
    try:
        report = build(args.manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"refusing manifest: {exc.__class__.__name__}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "weights_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate an R06 pilot manifest. Do not start a five-wide run as A6.

``--execute`` is refused while the production regret head width disagrees
with the manifest. That keeps QuarterPot from being dropped in name only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from alphanl.botzone.action_spec import A5, A6, SchemaMismatch, check_head_width  # noqa: E402
from alphanl.transformer.schema import NUM_ACTIONS  # noqa: E402

_CAP = 1000
_USED = 0


def _schema(manifest: dict):
    if manifest["action_schema_id"] == A5.schema_id:
        return A5
    if manifest["action_schema_id"] == A6.schema_id:
        return A6
    raise SchemaMismatch(f"unknown pilot schema {manifest['action_schema_id']!r}")


def validate(manifest: dict) -> None:
    schema = _schema(manifest)
    document = schema.document()
    if manifest["action_schema_sha256"] != document["sha256"]:
        raise SchemaMismatch("pilot manifest schema hash does not match the action table")
    if manifest["num_actions"] != schema.num_actions:
        raise SchemaMismatch("pilot manifest width does not match the action table")
    if schema is A6 and "quarter_pot" not in schema.names:
        raise SchemaMismatch("A6 manifest is missing quarter_pot")
    if manifest.get("quarter_pot_required") and "quarter_pot" not in schema.names:
        raise SchemaMismatch("quarter_pot was required and is absent")
    hours = float(manifest["max_gpu_hours"])
    if hours <= 0 or _USED + hours > _CAP:
        raise SchemaMismatch("pilot hours are outside the registered 1000 hour cap")
    if int(manifest["max_concurrent_training_jobs"]) != 1:
        raise SchemaMismatch("only one training job is authorized")
    if schema.num_actions != NUM_ACTIONS:
        check_head_width(schema, NUM_ACTIONS)
    raise SchemaMismatch(
        "refusing to start: the production regret trainer is botzone_5action_v1, "
        f"not {schema.schema_id}. A width match is not the A5/A6 chip contract. "
        "Quarter-pot stays required on A6 and is not dropped."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        validate(manifest)
    except SchemaMismatch as exc:
        print(json.dumps({
            "status": "PENDING_AUTODL_PARENT",
            "started": False,
            "gpu_hours_used": 0,
            "reason": str(exc),
        }))
        raise SystemExit(2)
    if not args.execute:
        print(json.dumps({"status": "PENDING_AUTODL_PARENT", "started": False, "gpu_hours_used": 0}))
        return
    raise SystemExit("execute passed validation but this dispatch does not start GPU training")


if __name__ == "__main__":
    main()

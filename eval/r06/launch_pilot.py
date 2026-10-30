"""Start an A5 or A6 teacher pilot only when the head matches ActionSpec.

Dry-run checks the manifest and builds the schema-width regret head.
``--execute`` on this CPU host runs one finite backward step and a
checkpoint round-trip. It does not spend GPU hours. A 4090 run is still
``PENDING_AUTODL_PARENT`` and must use the same command, one job at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from alphanl.botzone.action_spec import (  # noqa: E402
    A5,
    A6,
    SCHEMA_V1,
    SchemaMismatch,
    artifact_stamp,
    legacy_v1_document,
)
from alphanl.cfr.model import SchemaRegretNetwork  # noqa: E402
from alphanl.transformer.tokenizer import (  # noqa: E402
    CLS_ID,
    NUM_NUMERIC,
    TYPE_CLS,
)

_CAP = 1000.0
_USED = 0.0


def _schema(manifest: dict):
    if manifest.get("action_schema_id") == A5.schema_id:
        return A5
    if manifest.get("action_schema_id") == A6.schema_id:
        return A6
    raise SchemaMismatch(
        f"fail closed: {manifest.get('action_schema_id')!r} is not an A5/A6 pilot schema"
    )


def validate(manifest: dict):
    schema = _schema(manifest)
    document = schema.document()
    if manifest.get("action_schema_sha256") != document["sha256"]:
        raise SchemaMismatch("pilot manifest schema hash does not match the action table")
    if manifest.get("num_actions") != schema.num_actions:
        raise SchemaMismatch("pilot manifest width does not match the action table")
    if schema is A6 and manifest.get("action_names", [None])[-1] != "quarter_pot":
        raise SchemaMismatch("A6 manifest dropped quarter_pot")
    if manifest.get("quarter_pot_required") and "quarter_pot" not in schema.names:
        raise SchemaMismatch("quarter_pot was required and is absent")
    hours = float(manifest["max_gpu_hours"])
    if hours <= 0 or hours > 0.25 or _USED + hours > _CAP:
        raise SchemaMismatch("pilot hours are outside the 0.25 soft cap or the 1000 hour total")
    if int(manifest["max_concurrent_training_jobs"]) != 1:
        raise SchemaMismatch("only one training job is authorized")
    if manifest.get("action_schema_id") == SCHEMA_V1:
        raise SchemaMismatch("fail closed: botzone_5action_v1 cannot start an A5/A6 pilot")
    model = SchemaRegretNetwork(schema)
    if model.regret_head.out_features != schema.num_actions:
        raise SchemaMismatch("regret head width drifted from the action table")
    if schema is A6 and model.num_actions != 6:
        raise SchemaMismatch("A6 head is not width 6")
    return schema, model


def _batch(schema, model: SchemaRegretNetwork):
    length = model.backbone.max_seq_len
    token_ids = torch.zeros(1, length, dtype=torch.long)
    type_ids = torch.zeros(1, length, dtype=torch.long)
    token_ids[0, 0] = CLS_ID
    type_ids[0, 0] = TYPE_CLS
    numeric = torch.zeros(1, length, NUM_NUMERIC)
    pad_mask = torch.ones(1, length, dtype=torch.bool)
    pad_mask[0, 0] = False
    legal = torch.zeros(1, schema.num_actions)
    legal[0, 1] = 1.0
    if schema.num_actions == 6:
        legal[0, 5] = 1.0
    return {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "numeric": numeric,
        "pad_mask": pad_mask,
        "legal_mask": legal,
    }


def cpu_execute(schema, model: SchemaRegretNetwork, manifest: dict) -> dict:
    """One CPU step proving the head width. Not a strength result and not GPU time."""

    batch = _batch(schema, model)
    logits = model(batch)
    if logits.shape[-1] != schema.num_actions:
        raise SchemaMismatch("forward width does not match the schema")
    loss = logits.square().mean()
    if not torch.isfinite(loss):
        raise SchemaMismatch("non-finite pilot loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise SchemaMismatch("non-finite pilot gradient")
    payload = model.checkpoint_payload()
    reloaded = SchemaRegretNetwork(schema)
    reloaded.load_matching_checkpoint(payload)
    foreign = dict(artifact_stamp(A5 if schema is A6 else A6))
    foreign["state_dict"] = payload["state_dict"]
    rejected_other = False
    try:
        SchemaRegretNetwork(A6 if schema is A5 else A5).load_matching_checkpoint(foreign)
    except SchemaMismatch:
        rejected_other = True
    legacy = {
        "action_schema_id": SCHEMA_V1,
        "action_schema_sha256": legacy_v1_document()["sha256"],
        "num_actions": 5,
        "state_dict": payload["state_dict"],
    }
    rejected_v1 = False
    try:
        model.load_matching_checkpoint(legacy)
    except SchemaMismatch:
        rejected_v1 = True
    if not rejected_other or not rejected_v1:
        raise SchemaMismatch("schema mismatch was accepted")
    return {
        "status": "PENDING_AUTODL_PARENT",
        "cpu_execute_ok": True,
        "started_gpu": False,
        "gpu_hours_used": 0,
        "schema_id": schema.schema_id,
        "head_width": int(model.regret_head.out_features),
        "quarter_pot_in_head": schema.num_actions == 6 and "quarter_pot" in schema.names,
        "loss_finite": True,
        "rejected_other_schema": True,
        "rejected_botzone_5action_v1": True,
        "strength_claim": False,
        "manifest": str(manifest.get("isolated_output_dir")),
        "autodl_command": manifest.get("command"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        schema, model = validate(manifest)
    except SchemaMismatch as exc:
        print(json.dumps({
            "status": "FAIL",
            "started": False,
            "gpu_hours_used": 0,
            "reason": str(exc),
        }))
        raise SystemExit(2)
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN_OK",
            "started": False,
            "gpu_hours_used": 0,
            "schema_id": schema.schema_id,
            "head_width": schema.num_actions,
            "quarter_pot": "quarter_pot" in schema.names,
            "gpu_pilot_status": "PENDING_AUTODL_PARENT",
        }))
        return
    if torch.cuda.is_available():
        print(json.dumps({
            "status": "FAIL",
            "started_gpu": False,
            "gpu_hours_used": 0,
            "reason": "CUDA is visible. This dispatch does not start a GPU job from the cloud VM.",
        }))
        raise SystemExit(2)
    result = cpu_execute(schema, model, manifest)
    out = Path(__file__).resolve().parent / f"cpu_execute_{schema.schema_id}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

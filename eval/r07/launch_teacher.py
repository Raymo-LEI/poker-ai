"""Formal Teacher plan for A6 and the A5 control.

This host has no CUDA. ``--execute`` checks the schema-bound head and
rejects a five-action checkpoint. It does not invent learning curves or
debit GPU time. The 4090 commands are printed for a later serial run.
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
from alphanl.transformer.tokenizer import CLS_ID, NUM_NUMERIC, TYPE_CLS  # noqa: E402

_CAP = 1000.0
_USED = 0.0003791426171341704
_REMAINING = _CAP - _USED
_SEEDS = (101, 202, 303)
_MILESTONES = (50, 100, 200, 400, 800)
_PER_SEED_STOP_HOURS = 6.0
_SOFT_HINT_HOURS = 400.0
_WORST_CASE_SERIAL_HOURS = _PER_SEED_STOP_HOURS * 6
_OPPONENTS = {
    "Equity": 0.25,
    "TAG_Pro": 0.25,
    "Trapper_CR": 0.25,
    "FrozenHistory": 0.25,
}
_THRESHOLDS = {
    "noninferiority_mbb_per_hand_lower_bound_gt": -20,
    "primary_lower_bound_gt": 0,
    "primary_point_estimate_min_mbb_per_hand": 20,
    "multiplicity": "Bonferroni n=5 familywise alpha 0.05",
    "source": "evaluation_contract.default_final_strength_gate_proposal",
    "status": "REGISTERED_NOT_CLAIMED",
}


def _schema(name: str):
    if name == A6.schema_id:
        return A6
    if name == A5.schema_id:
        return A5
    raise SchemaMismatch(f"fail closed: {name!r} is not an A5/A6 teacher schema")


def manifest_for(schema, seed: int) -> dict:
    document = schema.document()
    tag = "a6" if schema is A6 else "a5"
    return {
        "format": "route1_r07_teacher_manifest_v1",
        "role": "primary" if schema is A6 else "a5_control",
        "action_schema_id": schema.schema_id,
        "action_schema_sha256": document["sha256"],
        "num_actions": schema.num_actions,
        "action_names": list(schema.names),
        "quarter_pot_required": schema is A6,
        "seed": seed,
        "milestones": list(_MILESTONES),
        "optional_t1000": False,
        "burn_in_iterations_excluded_from_scoring": list(range(1, 50)),
        "burn_in_frozen_before_scores": True,
        "loss_iteration_weight_power": 1,
        "strategy_mixture_weight": "iteration * own_reach",
        "loss_weight_is_not_mixture_weight": True,
        "do_not_treat_t_squared_loss_as_t_squared_average": True,
        "new_reservoir": True,
        "isolated_output_dir": f"runs/r07_{tag}_seed{seed}",
        "max_gpu_hours_this_seed": _PER_SEED_STOP_HOURS,
        "stop_if_first_iteration_exceeds_hours": 1.0,
        "soft_formal_teacher_hint_hours": _SOFT_HINT_HOURS,
        "soft_hint_is_not_a_target": True,
        "total_gpu_hours_cap": _CAP,
        "gpu_hours_used_before": _USED,
        "remaining_gpu_hours_before": _REMAINING,
        "max_concurrent_training_jobs": 1,
        "dev_eval_seed": 404,
        "opponents": _OPPONENTS,
        "frozen_history_internal_weights": "REGISTERED_PLACEHOLDER_NOT_LOCKED",
        "strength_thresholds": _THRESHOLDS,
        "hour_caps": {
            "per_seed_wall_stop_hours": _PER_SEED_STOP_HOURS,
            "stop_if_first_iteration_exceeds_hours": 1.0,
            "milestones": list(_MILESTONES),
            "milestones_share_one_seed_wall": True,
            "per_milestone_hour_budget_hours": None,
            "per_milestone_hours_measured": False,
            "optional_t1000_hours": 0,
            "note": "6h is a stop guard for the whole seed, not a measured time per milestone and not permission to spend the 400h hint.",
        },
        "production_width5_trainer": "FORBIDDEN",
        "production_width5_reason": "RegretNetwork and SDCFRTrainer stay on botzone_5action_v1. Starting them would drop quarter_pot and the A5 size table.",
        "cpu_smoke_is_not_the_formal_body": True,
        "certified_river_raw_fold_max": 0.01,
        "certified_river_raw_fold_status": "NOT_MEASURED",
        "lbr_is_not_full_exploitability": True,
        "distill_authorized": False,
        "strength_claim": False,
        "status": "PENDING_AUTODL_PARENT",
        "command": (
            "python3 eval/r07/launch_teacher.py --manifest "
            f"eval/r07/manifests/{tag}_seed{seed}.json --execute"
        ),
    }


def validate(manifest: dict):
    schema = _schema(manifest["action_schema_id"])
    document = schema.document()
    if manifest["action_schema_sha256"] != document["sha256"]:
        raise SchemaMismatch("teacher manifest hash does not match ActionSpec")
    if manifest["num_actions"] != schema.num_actions:
        raise SchemaMismatch("teacher manifest width does not match ActionSpec")
    if schema is A6 and "quarter_pot" not in manifest["action_names"]:
        raise SchemaMismatch("A6 teacher manifest dropped quarter_pot")
    if manifest["seed"] not in _SEEDS:
        raise SchemaMismatch("seed is outside the frozen 101/202/303 set")
    if tuple(manifest["milestones"]) != _MILESTONES:
        raise SchemaMismatch("milestones are not 50/100/200/400/800")
    if manifest.get("optional_t1000"):
        raise SchemaMismatch("T=1000 is not scheduled")
    hours = float(manifest["max_gpu_hours_this_seed"])
    if hours <= 0 or hours > _PER_SEED_STOP_HOURS:
        raise SchemaMismatch("per-seed stop is above the 6 hour wall")
    if _USED + hours > _CAP:
        raise SchemaMismatch("per-seed stop exceeds remaining GPU hours")
    if int(manifest["max_concurrent_training_jobs"]) != 1:
        raise SchemaMismatch("only one training job is authorized")
    if manifest.get("loss_iteration_weight_power") != 1:
        raise SchemaMismatch("loss weight power must stay 1 and must not be t^2")
    if manifest.get("distill_authorized"):
        raise SchemaMismatch("distill is not authorized before the teacher passes")
    if manifest.get("production_width5_trainer") != "FORBIDDEN":
        raise SchemaMismatch("the width-5 trainer must stay forbidden")
    if manifest.get("certified_river_raw_fold_status") != "NOT_MEASURED":
        raise SchemaMismatch("river fold was not measured; do not claim the 1% gate")
    caps = manifest.get("hour_caps") or {}
    if caps.get("per_milestone_hours_measured") or caps.get("optional_t1000_hours"):
        raise SchemaMismatch("milestone hours are not measured and T=1000 is not scheduled")
    model = SchemaRegretNetwork(schema)
    if model.regret_head.out_features != schema.num_actions:
        raise SchemaMismatch("regret head width drifted from ActionSpec")
    return schema, model


def _reject_v1(model: SchemaRegretNetwork) -> None:
    legacy = {
        "action_schema_id": SCHEMA_V1,
        "action_schema_sha256": legacy_v1_document()["sha256"],
        "num_actions": 5,
        "state_dict": model.checkpoint_payload()["state_dict"],
    }
    try:
        model.load_matching_checkpoint(legacy)
    except SchemaMismatch:
        return
    raise SchemaMismatch("botzone_5action_v1 checkpoint was accepted")


def cpu_gate(schema, model: SchemaRegretNetwork) -> dict:
    length = model.backbone.max_seq_len
    token_ids = torch.zeros(1, length, dtype=torch.long)
    type_ids = torch.zeros(1, length, dtype=torch.long)
    token_ids[0, 0] = CLS_ID
    type_ids[0, 0] = TYPE_CLS
    pad_mask = torch.ones(1, length, dtype=torch.bool)
    pad_mask[0, 0] = False
    legal = torch.zeros(1, schema.num_actions)
    legal[0, 0] = 1.0
    legal[0, 1] = 1.0
    if schema.num_actions == 6:
        legal[0, 5] = 1.0
    batch = {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "numeric": torch.zeros(1, length, NUM_NUMERIC),
        "pad_mask": pad_mask,
        "legal_mask": legal,
    }
    logits = model(batch)
    if tuple(logits.shape) != (1, schema.num_actions):
        raise SchemaMismatch("forward width does not match ActionSpec")
    loss = logits.square().mean()
    if not torch.isfinite(loss):
        raise SchemaMismatch("non-finite teacher smoke loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise SchemaMismatch("non-finite teacher smoke gradient")
    _reject_v1(model)
    return {
        "cpu_gate_ok": True,
        "head_width": schema.num_actions,
        "quarter_pot": "quarter_pot" in schema.names,
        "rejected_botzone_5action_v1": True,
        "gpu_hours_used": 0,
        "strength_claim": False,
        "metrics_are_training_curves": False,
    }


def write_plan(root: Path) -> None:
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for schema, tag in ((A6, "a6"), (A5, "a5")):
        for seed in _SEEDS:
            payload = manifest_for(schema, seed)
            path = manifest_dir / f"{tag}_seed{seed}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            jobs.append(payload["command"])
    curves = {
        "status": "NOT_MEASURED",
        "reason": "No formal Teacher iteration has been run on this host. Curves are not filled in.",
        "milestones": list(_MILESTONES),
        "seeds": list(_SEEDS),
        "tracks": [A6.schema_id, A5.schema_id],
        "loss_iteration_weight_power": 1,
        "strategy_mixture_weight": "iteration * own_reach",
        "t_squared_loss_used_as_mixture": False,
        "burn_in_frozen": "iterations 1..49 excluded from scoring before any formal look",
        "strength_claim": False,
        "values": None,
    }
    (root / "learning_curves.json").write_text(json.dumps(curves, indent=2) + "\n", encoding="utf-8")
    selection = {
        "selected_teacher": None,
        "reason": "No dev scores exist. Thin evidence is INCOMPLETE, not a reason to train longer automatically.",
        "distill_authorized": False,
        "a6_primary_schema": A6.schema_id,
        "a5_control_schema": A5.schema_id,
        "quarter_pot_retained": True,
        "seeds_reported_separately": list(_SEEDS),
        "best_seed_only": False,
        "strength_claim": False,
        "opponents": _OPPONENTS,
        "frozen_history_internal_weights": "REGISTERED_PLACEHOLDER_NOT_LOCKED",
        "thresholds": _THRESHOLDS,
        "certified_river_raw_fold_max": 0.01,
        "certified_river_raw_fold_status": "NOT_MEASURED",
        "a6_vs_a5_cost_benefit": "NOT_MEASURED",
        "final_set_used": False,
        "auto_extend_training": False,
    }
    (root / "selection_rationale.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    holdout = {
        "status": "NOT_RUN",
        "reason": "Independent holdout waits for a finished Teacher. CI is insufficient.",
        "dev_eval_seed": 404,
        "final_set_used": False,
        "lbr_abr_is_full_exploitability": False,
        "auto_extend_training": False,
    }
    (root / "holdout_report.json").write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    weakness = {
        "status": "NOT_RUN",
        "slices": ["unseen_pool", "unseen_sizes", "history_strategies", "approx_response"],
        "lbr_is_not_full_exploitability": True,
    }
    (root / "weakness_slices.json").write_text(json.dumps(weakness, indent=2) + "\n", encoding="utf-8")
    debit = {
        "task": "R07",
        "gpu_hours_used_this_task": 0,
        "gpu_hours_used_total": _USED,
        "remaining_gpu_hours": _REMAINING,
        "total_gpu_hours_cap": _CAP,
        "cap_raised": False,
        "distill_hours": 0,
        "train_hours": 0,
        "label_hours": 0,
        "eval_hours": 0,
        "per_seed_stop_hours": _PER_SEED_STOP_HOURS,
        "worst_case_serial_hours_if_every_seed_hits_the_stop": _WORST_CASE_SERIAL_HOURS,
        "soft_formal_teacher_hint_hours": _SOFT_HINT_HOURS,
        "soft_hint_is_not_a_target": True,
        "per_milestone_hour_budget_hours": None,
        "per_milestone_hours_measured": False,
        "optional_t1000_scheduled": False,
        "jobs_serial": jobs,
        "max_concurrent_training_jobs": 1,
        "production_width5_trainer": "FORBIDDEN",
        "note": "6 hour stop per seed is a launch guard, not a measured Teacher iteration time and not approval to spend the 400 hour hint. Six seeds times 6 hours is 36 hours, still under the remaining budget. Stop a seed if its first iteration exceeds 1 hour.",
    }
    (root / "budget_debit.json").write_text(json.dumps(debit, indent=2) + "\n", encoding="utf-8")
    frozen = {
        "format": "route1_r07_frozen_teacher_plan_v1",
        "a6_primary": artifact_stamp(A6),
        "a5_control": artifact_stamp(A5),
        "seeds": list(_SEEDS),
        "milestones": list(_MILESTONES),
        "quarter_pot_retained": True,
        "production_width5_trainer": "FORBIDDEN",
        "hunl_iterations_run": 0,
        "jobs": jobs,
        "parent_must_run_autodl": True,
    }
    (root / "frozen_teacher_manifest.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.write_plan:
        write_plan(root)
        print(json.dumps({"wrote_plan": True, "gpu_hours_used": 0}))
        return
    if args.manifest is None:
        raise SystemExit("pass --manifest or --write-plan")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        schema, model = validate(manifest)
    except SchemaMismatch as exc:
        print(json.dumps({"status": "FAIL", "started_gpu": False, "gpu_hours_used": 0, "reason": str(exc)}))
        raise SystemExit(2)
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN_OK",
            "started_gpu": False,
            "gpu_hours_used": 0,
            "schema_id": schema.schema_id,
            "head_width": schema.num_actions,
            "quarter_pot": "quarter_pot" in schema.names,
            "seed": manifest["seed"],
            "milestones": manifest["milestones"],
        }))
        return
    if torch.cuda.is_available():
        print(json.dumps({
            "status": "FAIL",
            "started_gpu": False,
            "gpu_hours_used": 0,
            "reason": "CUDA is visible. This cloud VM must not start the 4090 job.",
        }))
        raise SystemExit(2)
    gate = cpu_gate(schema, model)
    gate.update({
        "status": "PENDING_AUTODL_PARENT",
        "started_gpu": False,
        "seed": manifest["seed"],
        "schema_id": schema.schema_id,
        "command": manifest["command"],
    })
    out = root / f"cpu_gate_{schema.schema_id}_seed{manifest['seed']}.json"
    out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate))


if __name__ == "__main__":
    main()

"""CPU-only unit-cost probe for R05.

No CUDA, no Teacher fit, no distillation, no PPO update, no solve.
GPU rows stay unmeasured because the training GPU budget is unregistered.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

_JUDGE = _REPO / "tests" / "fixtures" / "botzone_judge.py"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"
_NPZ = _REPO / "eval" / "r03" / "weights" / "V6_R6_MODEL.npz"
_NPZ_SHA = "dfd57d307c6d4d9eb5d789ba145370792752ea85f5f43ba57b5dd787e3b4faaa"
_SCHEMA = _REPO / "eval" / "r04" / "action_schema_v2.json"
_SEED = 7
_HANDS = 200


def _rss_kb() -> dict:
    values = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, amount, unit = line.split()
                values[key[:-1]] = {"kb": int(amount), "unit": unit}
    values["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def _cpu_model() -> str:
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _meminfo() -> dict:
    wanted = {"MemTotal", "MemAvailable", "SwapTotal"}
    out = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, rest = line.split(":", 1)
            if key in wanted:
                amount, unit = rest.split()
                out[key] = {"kb": int(amount), "unit": unit}
    return out


def _hardware() -> dict:
    import numpy
    torch_info = {"installed": False, "cuda_used": False}
    try:
        import torch

        torch.set_num_threads(1)
        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_used": False,
        }
    except ImportError:
        pass
    return {
        "host_class": "cloud_vm_not_a_mac",
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "memory": _meminfo(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch_info,
        "process_limits": {
            "processes": 1,
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "torch_num_threads": 1,
            "cuda_visible_devices": "",
        },
    }


def _shuffled(rng: random.Random):
    cards = list(range(52))
    rng.shuffle(cards)
    return cards


def _play(schema, n_hands: int, seed: int) -> dict:
    from alphanl.botzone.action_spec import chip_responses
    from alphanl.botzone.kernel import BranchableKernel

    rng = random.Random(seed)
    started = time.perf_counter()
    rss_before = _rss_kb()
    decisions = 0
    for hand in range(n_hands):
        kernel = BranchableKernel.start_hand(_shuffled(rng), dealer=hand % 2)
        guard = 0
        while not kernel.hand_over and guard < 16:
            state = kernel.as_hand_state()
            legal = [response for response in chip_responses(state, schema) if response is not None]
            kernel.step(legal[rng.randrange(len(legal))] if legal else -1)
            decisions += 1
            guard += 1
    elapsed = time.perf_counter() - started
    return {
        "schema_id": schema.schema_id,
        "num_actions": schema.num_actions,
        "hands": n_hands,
        "decisions": decisions,
        "wall_seconds": elapsed,
        "seconds_per_decision": elapsed / decisions if decisions else None,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "rss_before": rss_before,
        "rss_after": _rss_kb(),
        "source": "derived_from_cpu_probe",
        "cuda_used": False,
    }


def _cold_import() -> dict:
    code = """
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
started = time.perf_counter()
import sys
sys.path.insert(0, 'src')
from alphanl.botzone.kernel import BranchableKernel
from alphanl.botzone.action_spec import A5, A6
elapsed = time.perf_counter() - started
print(elapsed)
"""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO),
        check=True,
        capture_output=True,
        text=True,
    )
    wrapper = time.perf_counter() - started
    return {
        "command": "python3 -c 'import BranchableKernel, A5, A6'",
        "cwd": str(_REPO),
        "import_wall_seconds": float(completed.stdout.strip()),
        "subprocess_wall_seconds": wrapper,
        "cuda_used": False,
        "source": "derived_from_cpu_probe",
    }


def _checkpoint_copy() -> dict:
    if hashlib.sha256(_NPZ.read_bytes()).hexdigest() != _NPZ_SHA:
        raise SystemExit("V6 npz hash changed; refusing to use it as a size sample")
    size = _NPZ.stat().st_size
    destination = Path("/tmp/r05_checkpoint_copy.npz")
    started = time.perf_counter()
    shutil.copyfile(_NPZ, destination)
    elapsed = time.perf_counter() - started
    copied = destination.stat().st_size
    destination.unlink()
    rollback = {
        "assumption": "one live export plus one previous copy plus one rollback copy",
        "bytes": size * 3,
        "source": "assumption:3x_measured_v6_npz",
        "not_a_future_training_checkpoint": True,
    }
    return {
        "sample": "eval/r03/weights/V6_R6_MODEL.npz",
        "sha256": _NPZ_SHA,
        "bytes": size,
        "copy_wall_seconds": elapsed,
        "copied_bytes": copied,
        "rollback_headroom": rollback,
        "cuda_used": False,
        "source": "derived_from_cpu_probe",
    }


def _teacher_label() -> dict:
    return {
        "status": "NOT_RUN",
        "reason": "Teacher snapshots and policy_best.pt are not in this tree. No labels were invented.",
        "wall_seconds": None,
        "decisions_per_second": None,
        "cuda_used": False,
        "source": "not_measured",
    }


def measure() -> dict:
    from alphanl.botzone.action_spec import A5, A6

    if hashlib.sha256(_JUDGE.read_bytes()).hexdigest() != _JUDGE_SHA:
        raise SystemExit("botzone_judge.py hash changed")
    hardware = _hardware()
    if hardware["torch"].get("cuda_available"):
        raise SystemExit("CUDA is visible; this probe must not run on a GPU")
    cold = _cold_import()
    a5 = _play(A5, _HANDS, _SEED)
    a6 = _play(A6, _HANDS, _SEED)
    checkpoint = _checkpoint_copy()
    return {
        "task_id": "R05",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": _SEED,
        "hardware": hardware,
        "authorization": {
            "training_gpu_budget": "UNREGISTERED",
            "gpu_probe": "NOT_RUN",
            "formal_training": "NOT_RUN",
            "gpu_hours_used": 0,
        },
        "cpu": {
            "cold_import": cold,
            "a5_kernel_play": a5,
            "a6_kernel_play": a6,
            "checkpoint_copy": checkpoint,
            "teacher_label": _teacher_label(),
        },
        "gpu": {
            "status": "PENDING_USER_APPROVED_PROBE",
            "peak_vram_bytes": None,
            "step_seconds": None,
            "cuda_used": False,
            "requested_probe": {
                "minutes": 20,
                "gpu_model": "USER_MUST_NAME",
                "claims_approval": False,
                "commands": [
                    "Do not start this until a user-approved probe budget exists.",
                    "One CUDA forward of the existing Stage2 policy, batch 32, no optimizer step.",
                    "Record torch.cuda.max_memory_allocated and wall time, then exit.",
                ],
            },
        },
        "judge_sha256": _JUDGE_SHA,
        "action_schema_path": "eval/r04/action_schema_v2.json",
    }


def _proposal(probe: dict) -> dict:
    a5 = probe["cpu"]["a5_kernel_play"]
    a6 = probe["cpu"]["a6_kernel_play"]
    checkpoint = probe["cpu"]["checkpoint_copy"]
    per_decision = max(a5["seconds_per_decision"], a6["seconds_per_decision"])

    def cpu_play(name: str, decisions: int, note: str) -> dict:
        return {
            "id": name,
            "cpu_wall_seconds_estimate": per_decision * decisions,
            "decisions_assumed": decisions,
            "seconds_per_decision_used": per_decision,
            "source": "derived_from_cpu_probe",
            "assumption": note,
            "gpu_hours_cap": None,
            "gpu_status": "PENDING_USER_APPROVED_PROBE",
            "claims_approval": False,
        }

    return {
        "format": "route1_r05_budget_proposal_v1",
        "status": "PROPOSAL_NOT_APPROVED",
        "claims_user_approval": False,
        "training_gpu_budget": "UNREGISTERED",
        "total_gpu_hours_cap": None,
        "total_gpu_hours_cap_status": "PENDING_USER_APPROVED_PROBE",
        "process_limit": probe["hardware"]["process_limits"],
        "checkpoint_rollback_bytes": checkpoint["rollback_headroom"],
        "measured_seconds_per_decision": {
            "a5": a5["seconds_per_decision"],
            "a6": a6["seconds_per_decision"],
            "source": "derived_from_cpu_probe",
        },
        "phases": [
            cpu_play(
                "a5_pilot_kernel_replay",
                1000,
                "1000 decision states, same order of magnitude as the R04 CPU smoke, priced at the slower of the two measured schema rates",
            ),
            cpu_play(
                "a6_pilot_kernel_replay",
                1000,
                "same 1000-state CPU replay with the six-action schema",
            ),
            {
                "id": "formal_teacher",
                "cpu_wall_seconds_estimate": None,
                "gpu_hours_cap": None,
                "gpu_status": "PENDING_USER_APPROVED_PROBE",
                "source": "requires_gpu_probe",
                "reason": "Teacher snapshots are absent and no GPU probe is authorized.",
                "claims_approval": False,
            },
            {
                "id": "distill",
                "cpu_wall_seconds_estimate": None,
                "gpu_hours_cap": None,
                "gpu_status": "PENDING_USER_APPROVED_PROBE",
                "source": "requires_gpu_probe",
                "claims_approval": False,
            },
            {
                "id": "ppo",
                "cpu_wall_seconds_estimate": None,
                "gpu_hours_cap": None,
                "gpu_status": "PENDING_USER_APPROVED_PROBE",
                "source": "requires_gpu_probe",
                "claims_approval": False,
            },
            {
                "id": "river_turn_solve",
                "cpu_wall_seconds_estimate": None,
                "gpu_hours_cap": None,
                "gpu_status": "PENDING_USER_APPROVED_PROBE",
                "source": "requires_gpu_probe",
                "claims_approval": False,
            },
            {
                "id": "final_acceptance",
                "registered_sample": "200 paired segments / strategy / opponent / training seed = 20000 hands",
                "sample_shrunk": False,
                "cpu_wall_seconds_estimate": None,
                "gpu_hours_cap": None,
                "gpu_status": "PENDING_USER_APPROVED_PROBE",
                "source": "user_must_approve",
                "note": "R01 sample size is kept. Variance recomputation waits for a measured eval rate.",
                "claims_approval": False,
            },
        ],
        "requested_gpu_probe_minutes": probe["gpu"]["requested_probe"]["minutes"],
        "requested_gpu_probe_commands": probe["gpu"]["requested_probe"]["commands"],
    }


def _registry() -> dict:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    return {
        "format": "route1_r05_experiment_registry_v1",
        "status": "PROPOSED_WITH_R01_FREEZES",
        "claims_user_approval": False,
        "action_schemas": {
            "a5": {"schema_id": schema["a5"]["schema_id"], "sha256": schema["a5"]["sha256"]},
            "a6": {"schema_id": schema["a6"]["schema_id"], "sha256": schema["a6"]["sha256"]},
            "retained_v1": {
                "schema_id": schema["retained_v1"]["schema_id"],
                "sha256": schema["retained_v1"]["sha256"],
                "role": "frozen_contrast_not_an_experiment_track",
            },
        },
        "seeds": {
            "status": "FROZEN_IN_R01",
            "source": "evaluation_contract.frozen_seeds",
            "evaluation_master_seed": 20261031,
            "final_training_seeds": [101, 202, 303],
            "dev_eval_seed": 404,
            "technical_smoke_seed": 7,
            "pro_gauntlet_seed_not_for_final": 20261030,
        },
        "primary_opponents": {
            "status": "FROZEN_IN_R01_EXCEPT_FROZEN_HISTORY_WEIGHTS",
            "families": ["Equity", "TAG_Pro", "Trapper_CR", "FrozenHistory"],
            "weights": {"Equity": 0.25, "TAG_Pro": 0.25, "Trapper_CR": 0.25, "FrozenHistory": 0.25},
            "frozen_history_internal_weights": {
                "status": "REGISTERED_PLACEHOLDER_NOT_LOCKED",
                "reason": "R01 requires concrete history weights before the final look. They are not invented here.",
            },
        },
        "thresholds": {
            "status": "FROZEN_DEFAULT_PROPOSAL_pre_training",
            "source": "evaluation_contract.default_final_strength_gate_proposal",
            "noninferiority_mbb_per_hand_lower_bound_gt": -20,
            "primary_lower_bound_gt": 0,
            "primary_point_estimate_min_mbb_per_hand": 20,
            "multiplicity": "Bonferroni n=5 familywise alpha 0.05",
            "cluster": "paired_match_segment",
        },
        "sample_sizes": {
            "status": "REGISTERED_DEFAULT_NOT_SHRUNK",
            "dev_paired_segments_per_opponent": 20,
            "final_paired_segments_per_strategy_opponent_seed": 200,
            "final_hands": 20000,
            "variance_recompute": "PENDING_UNTIL_MEASURED_EVAL_RATE",
        },
        "selection": {
            "max_model_selection_count": {
                "status": "PROPOSAL_NOT_FROZEN",
                "source": "user_must_approve",
                "dev_looks_per_training_seed": 1,
                "final_looks": 0,
            }
        },
        "final_test_access": {
            "status": "FROZEN_IN_R01",
            "final_set_not_used_for_model_selection": True,
            "after_peeking_finals_and_changing_candidate": "build_new_final_set",
            "already_viewed_online_replays": "diagnostic_only_never_final_blind",
        },
        "concurrency": {
            "status": "MEASURED_SINGLE_PROCESS",
            "training_processes_authorized": 0,
            "probe_processes": 1,
            "threads_per_process": 1,
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    probe = measure()
    proposal = _proposal(probe)
    registry = _registry()
    (root / "cost_probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "budget_proposal.json").write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "experiment_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "a5_decisions_per_second": probe["cpu"]["a5_kernel_play"]["decisions_per_second"],
        "a6_decisions_per_second": probe["cpu"]["a6_kernel_play"]["decisions_per_second"],
        "checkpoint_bytes": probe["cpu"]["checkpoint_copy"]["bytes"],
        "gpu_status": probe["gpu"]["status"],
        "proposal_approved": proposal["claims_user_approval"],
    }))


if __name__ == "__main__":
    main()

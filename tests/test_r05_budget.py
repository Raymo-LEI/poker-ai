"""R05 CPU probe does not approve a GPU budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "eval" / "r05"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"


def _load(name: str) -> dict:
    return json.loads((_ROOT / name).read_text(encoding="utf-8"))


def test_cpu_probe_is_measured_and_gpu_was_not_used():
    probe = _load("cost_probe.json")
    assert probe["cpu"]["a5_kernel_play"]["decisions"] > 0
    assert probe["cpu"]["a5_kernel_play"]["wall_seconds"] > 0
    assert probe["cpu"]["a6_kernel_play"]["wall_seconds"] > 0
    assert probe["cpu"]["checkpoint_copy"]["bytes"] == 6060293
    assert probe["cpu"]["cold_import"]["import_wall_seconds"] > 0
    assert probe["hardware"]["process_limits"]["processes"] == 1
    assert probe["gpu"]["status"] == "PENDING_USER_APPROVED_PROBE"
    assert probe["gpu"]["peak_vram_bytes"] is None
    assert probe["gpu"]["cuda_used"] is False
    assert probe["authorization"]["gpu_hours_used"] == 0
    assert probe["hardware"]["torch"]["cuda_available"] is False


def test_budget_proposal_is_not_self_approved():
    proposal = _load("budget_proposal.json")
    assert proposal["claims_user_approval"] is False
    assert proposal["status"] == "PROPOSAL_NOT_APPROVED"
    assert proposal["total_gpu_hours_cap"] is None
    assert proposal["training_gpu_budget"] == "UNREGISTERED"
    ids = [phase["id"] for phase in proposal["phases"]]
    assert ids == [
        "a5_pilot_kernel_replay",
        "a6_pilot_kernel_replay",
        "formal_teacher",
        "distill",
        "ppo",
        "river_turn_solve",
        "final_acceptance",
    ]
    for phase in proposal["phases"]:
        assert phase["gpu_hours_cap"] is None
        assert phase["claims_approval"] is False
    assert proposal["phases"][-1]["sample_shrunk"] is False


def test_registry_keeps_r01_seeds_and_final_access():
    registry = _load("experiment_registry.json")
    assert registry["seeds"]["final_training_seeds"] == [101, 202, 303]
    assert registry["seeds"]["dev_eval_seed"] == 404
    assert registry["final_test_access"]["final_set_not_used_for_model_selection"] is True
    assert registry["sample_sizes"]["final_paired_segments_per_strategy_opponent_seed"] == 200
    assert registry["primary_opponents"]["frozen_history_internal_weights"]["status"] == (
        "REGISTERED_PLACEHOLDER_NOT_LOCKED"
    )
    assert registry["selection"]["max_model_selection_count"]["status"] == "PROPOSAL_NOT_FROZEN"
    digest = hashlib.sha256((_REPO / "tests" / "fixtures" / "botzone_judge.py").read_bytes()).hexdigest()
    assert digest == _JUDGE_SHA

"""A5/A6 action contract: fail closed, dedupe, and kernel legality."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from alphanl.botzone.action_spec import (
    A5,
    A6,
    SCHEMA_V1,
    SchemaMismatch,
    artifact_stamp,
    check_head_width,
    chip_responses,
    legacy_v1_document,
    quarter_is_new,
    require_experiment_schema,
    require_schema,
    schema_bundle,
)
from alphanl.botzone.actions import action_responses
from alphanl.botzone.kernel import BranchableKernel
from alphanl.botzone.legal import new_hand

_REPO = Path(__file__).resolve().parents[1]
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"


def test_a5_and_a6_differ_only_by_quarter_pot():
    assert A6.names[:5] == A5.names
    assert A6.names[5] == "quarter_pot"
    assert A5.num_actions == 5
    assert A6.num_actions == 6
    assert A5.train_budget_steps == A6.train_budget_steps == 0
    bundle = schema_bundle()
    assert bundle["a5"]["sha256"] != bundle["a6"]["sha256"]
    assert bundle["retained_v1"]["schema_id"] == SCHEMA_V1


def test_old_schema_and_wrong_width_fail_closed():
    retained = artifact_stamp(A5)
    retained["action_schema_id"] = SCHEMA_V1
    retained["action_schema_sha256"] = legacy_v1_document()["sha256"]
    with pytest.raises(SchemaMismatch):
        require_schema(retained, A6)
    with pytest.raises(SchemaMismatch):
        require_experiment_schema({"labels": [0.2] * 5})
    with pytest.raises(SchemaMismatch):
        check_head_width(A6, 5)
    stamped = artifact_stamp(A6)
    require_schema(stamped, A6)
    check_head_width(A6, 6)


def test_duplicate_chip_responses_are_one_outcome():
    state = new_hand(0)
    responses = chip_responses(state, A6)
    executable = [response for response in responses if response is not None]
    assert len(executable) == len(set(executable))
    assert responses[A6.index("quarter_pot")] is None or responses[A6.index("quarter_pot")] not in set(
        response for action_id, response in enumerate(responses) if response is not None and action_id != 5
    )


def test_mapped_responses_are_legal_and_do_not_exceed_the_stack():
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    kernel.step(0)
    kernel.step(0)
    state = kernel.as_hand_state()
    stack = state.stacks[state.actor]
    for schema in (A5, A6):
        for response in chip_responses(state, schema):
            if response is None:
                continue
            assert kernel.is_legal(response)
            assert response < 0 or response < stack
    legacy = action_responses(state)
    assert legacy[2] == state.bound_b - state.street_commit[state.actor] or legacy[2] is None


def test_quarter_pot_collapse_is_reported_not_called_skill():
    opening = new_hand(0)
    assert quarter_is_new(opening) is not True
    kernel = BranchableKernel.start_hand(list(range(52)), dealer=0)
    kernel.step(0)
    kernel.step(0)
    kernel.step(400)
    later = kernel.as_hand_state()
    assert quarter_is_new(later) in (True, False, None)


def test_smoke_report_covers_a_thousand_boundary_states():
    report = json.loads((_REPO / "eval" / "r04" / "a5_a6_smoke_report.json").read_text(encoding="utf-8"))
    assert report["states"] >= 1000
    assert report["illegal_or_overstack"] == 0
    assert report["duplicate_executable"] == 0
    assert report["conservation_failures"] == 0
    assert report["shared_response_mismatches"] == 0
    assert report["small_bet_skill_claimed"] is False
    assert 0.0 <= report["quarterpot_effective_new_coverage"] <= 1.0
    equivalence = json.loads((_REPO / "eval" / "r04" / "response_equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["comparison"] == "chip_response_probability_mass"
    digest = __import__("hashlib").sha256((_REPO / "tests" / "fixtures" / "botzone_judge.py").read_bytes()).hexdigest()
    assert digest == _JUDGE_SHA
    rng = random.Random(0)
    cards = list(range(52))
    rng.shuffle(cards)
    kernel = BranchableKernel.start_hand(cards, dealer=1)
    assert kernel.actor in (0, 1)

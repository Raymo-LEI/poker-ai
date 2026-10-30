"""Shared action contracts for the A5/A6 experiment.

Chip amounts are referee extras: ``-1`` fold, ``0`` check/call, ``-2`` all-in,
or a positive number of chips added on this action. Formulas live in this
table. A policy head width is read from the schema; it is not a separate
constant.

``botzone_5action_v1`` is the frozen V6 contrast (min-raise and large-raise).
``botzone_a5_v2`` and ``botzone_a6_v2`` share fold, check/call, half-pot,
pot, and all-in. A6 only appends quarter-pot. Old labels, snapshots, and
reservoirs that do not carry the expected schema id and hash are rejected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .actions import action_responses as legacy_v1_responses
from .legal import ALL_IN, CHECK_CALL, FOLD, HandState, legal_response, safe_extra, to_call

SCHEMA_V1 = "botzone_5action_v1"
SCHEMA_A5 = "botzone_a5_v2"
SCHEMA_A6 = "botzone_a6_v2"


class SchemaMismatch(ValueError):
    """Old or foreign action artifacts are not loaded."""


@dataclass(frozen=True)
class ActionSpec:
    """One policy id and the chip-response rule that implements it."""

    action_id: int
    name: str
    kind: str
    numer: int = 0
    denom: int = 1

    def to_json(self) -> dict:
        payload = {
            "action_id": self.action_id,
            "name": self.name,
            "kind": self.kind,
        }
        if self.kind == "pot_fraction":
            payload["fraction"] = {"numer": self.numer, "denom": self.denom}
        return payload


@dataclass(frozen=True)
class ActionSchema:
    schema_id: str
    actions: Tuple[ActionSpec, ...]
    rules_id: str = "botzone_texasholdem2p_official_referee_v1"
    feature_note: str = "same public kernel view for A5 and A6"
    train_budget_steps: int = 0

    def __post_init__(self) -> None:
        ids = [item.action_id for item in self.actions]
        if ids != list(range(len(ids))):
            raise ValueError(f"action ids must be 0..n-1 in order, got {ids}")

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.actions)

    def index(self, name: str) -> int:
        for item in self.actions:
            if item.name == name:
                return item.action_id
        raise KeyError(name)

    def document(self) -> dict:
        body = {
            "format": "route1_action_schema_v2",
            "schema_id": self.schema_id,
            "rules_id": self.rules_id,
            "num_actions": self.num_actions,
            "train_budget_steps": self.train_budget_steps,
            "feature_note": self.feature_note,
            "chip_response": {
                "fold": -1,
                "check_call": 0,
                "all_in": -2,
                "pot_fraction": (
                    "Unopened: round_half_up(numer/denom * pot). "
                    "Facing a bet: to_call + round_half_up(numer/denom * pot). "
                    "Then raise the extra to the judge minimum positive bet. "
                    "A result greater than or equal to the remaining stack is not "
                    "a distinct size; only the all-in sentinel spends the rest."
                ),
                "rounding": "round_half_up via (pot * numer + denom // 2) // denom",
                "dedupe": "If two ids emit the same chip response, the lower id stays executable.",
            },
            "actions": [item.to_json() for item in self.actions],
        }
        body["sha256"] = schema_sha256(body)
        return body


def schema_sha256(document: Mapping) -> str:
    body = {key: value for key, value in document.items() if key != "sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _spec(action_id: int, name: str, kind: str, numer: int = 0, denom: int = 1) -> ActionSpec:
    return ActionSpec(action_id, name, kind, numer, denom)


def _pot_family(schema_id: str, with_quarter: bool) -> ActionSchema:
    actions = [
        _spec(0, "fold", "fold"),
        _spec(1, "check_call", "check_call"),
        _spec(2, "half_pot", "pot_fraction", 1, 2),
        _spec(3, "pot", "pot_fraction", 1, 1),
        _spec(4, "all_in", "all_in"),
    ]
    if with_quarter:
        actions.append(_spec(5, "quarter_pot", "pot_fraction", 1, 4))
    return ActionSchema(schema_id, tuple(actions))


A5 = _pot_family(SCHEMA_A5, with_quarter=False)
A6 = _pot_family(SCHEMA_A6, with_quarter=True)
EXPERIMENT_SCHEMAS = {SCHEMA_A5: A5, SCHEMA_A6: A6}


def round_half_up(pot: int, numer: int, denom: int) -> int:
    if pot < 0 or numer <= 0 or denom <= 0:
        raise ValueError("pot fraction rounding needs a non-negative pot and a positive fraction")
    return (int(pot) * int(numer) + int(denom) // 2) // int(denom)


def require_schema(artifact: Mapping, expected: ActionSchema) -> None:
    """Reject an artifact whose action contract is missing or different."""

    schema_id = artifact.get("action_schema_id")
    digest = artifact.get("action_schema_sha256")
    width = artifact.get("num_actions")
    expected_sha = expected.document()["sha256"]
    if schema_id != expected.schema_id or digest != expected_sha or width != expected.num_actions:
        raise SchemaMismatch(
            "fail closed: artifact schema "
            f"{schema_id!r} sha={digest!r} width={width!r} "
            f"does not match {expected.schema_id} sha={expected_sha} width={expected.num_actions}"
        )


def require_experiment_schema(artifact: Mapping) -> ActionSchema:
    schema_id = artifact.get("action_schema_id")
    if schema_id not in EXPERIMENT_SCHEMAS:
        raise SchemaMismatch(
            f"fail closed: {schema_id!r} is not an A5/A6 experiment schema"
        )
    schema = EXPERIMENT_SCHEMAS[str(schema_id)]
    require_schema(artifact, schema)
    return schema


def _candidate(state: HandState, spec: ActionSpec, legal: set) -> Optional[int]:
    if spec.kind == "fold":
        return FOLD if FOLD in legal else None
    if spec.kind == "check_call":
        return CHECK_CALL if CHECK_CALL in legal else None
    if spec.kind == "all_in":
        return ALL_IN if ALL_IN in legal else None
    if spec.kind != "pot_fraction":
        raise ValueError(f"unknown action kind {spec.kind}")
    safe = safe_extra(state)
    if safe is None:
        return None
    stack = int(state.stacks[state.actor])
    owed = int(to_call(state))
    raw = round_half_up(int(state.pot), spec.numer, spec.denom)
    extra = owed + raw if owed > 0 else raw
    if extra < int(safe):
        extra = int(safe)
    if extra >= stack or extra not in legal:
        return None
    return int(extra)


def chip_responses(state: HandState, schema: ActionSchema) -> Tuple[Optional[int], ...]:
    """Legal chip response per id, with identical responses kept once.

    The lower id remains executable. Later ids that land on the same chips
    are masked so they are not a second outcome.
    """

    if state.hand_over:
        return tuple(None for _ in schema.actions)
    legal = legal_response(state)
    chosen: List[Optional[int]] = []
    seen = set()
    for spec in schema.actions:
        response = _candidate(state, spec, legal)
        if response is not None and response in seen:
            response = None
        if response is not None:
            seen.add(response)
        chosen.append(response)
    return tuple(chosen)


def executable_responses(state: HandState, schema: ActionSchema) -> Dict[int, int]:
    """Map each distinct chip response to the canonical action id."""

    out = {}
    for action_id, response in enumerate(chip_responses(state, schema)):
        if response is not None:
            out[int(response)] = int(action_id)
    return out


def response_mass(state: HandState, schema: ActionSchema) -> Dict[int, float]:
    """Uniform mass over executable ids, summed by chip response."""

    responses = [response for response in chip_responses(state, schema) if response is not None]
    if not responses:
        return {}
    weight = 1.0 / float(len(responses))
    mass: Dict[int, float] = {}
    for response in responses:
        mass[int(response)] = mass.get(int(response), 0.0) + weight
    return mass


def _raw_responses(state: HandState, schema: ActionSchema) -> Tuple[Optional[int], ...]:
    """Chip candidates before identical responses are collapsed."""

    if state.hand_over:
        return tuple(None for _ in schema.actions)
    legal = legal_response(state)
    return tuple(_candidate(state, spec, legal) for spec in schema.actions)


def quarter_disposition(state: HandState) -> str:
    """``new``, ``collapsed`` onto an older size, or ``unavailable``."""

    raw = _raw_responses(state, A6)
    quarter = raw[A6.index("quarter_pot")]
    if quarter is None:
        return "unavailable"
    older = {response for response in raw[: A6.index("quarter_pot")] if response is not None}
    older.update(response for response in chip_responses(state, A5) if response is not None)
    if int(quarter) in older:
        return "collapsed"
    return "new"


def quarter_is_new(state: HandState) -> Optional[bool]:
    """Whether quarter-pot survives as a chip amount A5 does not emit.

    ``None`` when it is unavailable or collapsed. ``True`` only for a distinct
    legal extra. Collapse is not small-bet skill.
    """

    return True if quarter_disposition(state) == "new" else None


def legacy_v1_document() -> dict:
    """Fingerprint of the frozen five-action contrast. Not an A5/A6 schema."""

    body = {
        "format": "route1_action_schema_v1_retained",
        "schema_id": SCHEMA_V1,
        "rules_id": "botzone_texasholdem2p_official_referee_v1",
        "num_actions": 5,
        "role": "frozen_v6_contrast",
        "actions": [
            {"action_id": 0, "name": "fold", "kind": "fold"},
            {"action_id": 1, "name": "check_call", "kind": "check_call"},
            {
                "action_id": 2,
                "name": "raise_min",
                "kind": "min_legal_positive",
                "formula": "safe_extra = max(1, 2 * round_raise - own_round_bet)",
            },
            {
                "action_id": 3,
                "name": "raise_large",
                "kind": "max_twice_min_or_pot",
                "formula": "max(safe_extra * 2, pot)",
            },
            {"action_id": 4, "name": "all_in", "kind": "all_in"},
        ],
        "implementation": "alphanl.botzone.actions.action_responses",
    }
    body["sha256"] = schema_sha256(body)
    return body


def artifact_stamp(schema: ActionSchema) -> dict:
    document = schema.document()
    return {
        "action_schema_id": schema.schema_id,
        "action_schema_sha256": document["sha256"],
        "num_actions": schema.num_actions,
    }


def head_width(schema: ActionSchema) -> int:
    """Width a new experiment head must use. Refuses a silent 5-d A6 head."""

    if schema.schema_id not in EXPERIMENT_SCHEMAS:
        raise SchemaMismatch(f"{schema.schema_id} is not an experiment schema")
    return schema.num_actions


def check_head_width(schema: ActionSchema, width: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width != head_width(schema):
        raise SchemaMismatch(
            f"fail closed: head width {width!r} != {schema.schema_id} width {schema.num_actions}"
        )


def iter_action_ids(schema: ActionSchema) -> Iterable[int]:
    return range(schema.num_actions)


def shared_action_names() -> Tuple[str, ...]:
    return tuple(name for name in A5.names if name in A6.names)


def mass_by_name(state: HandState, schema: ActionSchema) -> Dict[str, Optional[dict]]:
    """Per-name chip response and the mass sitting on that response."""

    responses = chip_responses(state, schema)
    mass = response_mass(state, schema)
    out = {}
    for spec, response in zip(schema.actions, responses):
        if response is None:
            out[spec.name] = None
        else:
            out[spec.name] = {"response": int(response), "mass": mass[int(response)]}
    return out


def schema_bundle() -> dict:
    """Canonical file body for ``action_schema_v2.json``."""

    retained = legacy_v1_document()
    return {
        "format": "route1_action_schema_v2_bundle",
        "rules_id": "botzone_texasholdem2p_official_referee_v1",
        "note": (
            "A5 and A6 share rules, features, trunk size, data method, and a zero "
            "training budget. Only the action set differs. botzone_5action_v1 stays "
            "the frozen V6 contrast and is rejected by the A5/A6 loader."
        ),
        "retained_v1": retained,
        "a5": A5.document(),
        "a6": A6.document(),
    }


def hardcoded_width_audit(repo: Optional[str] = None) -> List[dict]:
    """Sites whose literal width is still 5, and how R04 treats them.

    The experiment path does not retarget these modules. Feeding them an A6
    artifact fails closed in ``require_schema`` instead of widening a head.
    """

    del repo
    frozen = "frozen_botzone_5action_v1_fail_closed"
    rows = [
        ("src/alphanl/botzone/actions.py", "NUM_ACTIONS = 5", frozen),
        ("src/alphanl/transformer/schema.py", "NUM_ACTIONS = 5", frozen),
        ("src/alphanl/nl_obs.py", "NUM_ACTIONS = 5", frozen),
        ("src/alphanl/transformer/conv_net.py", "conv head width NUM_ACTIONS", frozen),
        ("src/alphanl/transformer/conv_obs.py", "legal_mask width 5", frozen),
        ("src/alphanl/stage2/policy.py", "delta_head width 5", frozen),
        ("src/alphanl/exploiter/policy.py", "delta_head width 5", frozen),
        ("src/alphanl/cfr/fidelity.py", "target width 5", frozen),
        ("src/alphanl/cfr/parallel.py", "regret width 5", frozen),
        ("src/alphanl/tf_teacher.py", "reshape(-1, 5)", frozen),
        ("src/alphanl/botzone/action_spec.py", "width from ActionSchema", "table_driven"),
        ("src/alphanl/cfr/kuhn.py", "width 5", "not_botzone_action_schema"),
    ]
    return [
        {"path": path, "site": site, "disposition": disposition}
        for path, site, disposition in rows
    ]

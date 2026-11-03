"""Action schemas and GTO Wizard request fields for the offline scaffold.

A5 and A6 match the Route 1 names: fold, check/call, half-pot, pot, all-in,
and A6 appends quarter-pot. Chip amounts are resolved by ``HunlTable`` in the
adapter. This module does not change the trainer's five-action head.
"""

from __future__ import annotations

from typing import Mapping

SCHEMA_A5 = "botzone_a5_v2"
SCHEMA_A6 = "botzone_a6_v2"
GAME_NAME = "HUNL 200BB"
API_URL = "https://researcher.gtowizard.com"

A5_NAMES = ("fold", "check_call", "half_pot", "pot", "all_in")
A6_NAMES = A5_NAMES + ("quarter_pot",)
QUARTER_POT_ID = 5

_NAMES = {SCHEMA_A5: A5_NAMES, SCHEMA_A6: A6_NAMES}


class SchemaError(ValueError):
    """The fixture or request does not match the A5/A6 contract."""


def action_names(schema_id: str) -> tuple[str, ...]:
    try:
        return _NAMES[schema_id]
    except KeyError as exc:
        raise SchemaError(f"unsupported action schema {schema_id!r}") from exc


def action_index(schema_id: str, name: str) -> int:
    names = action_names(schema_id)
    try:
        return names.index(name)
    except ValueError as exc:
        raise SchemaError(f"{schema_id} has no action {name!r}") from exc


def require_spot(spot: Mapping) -> str:
    schema_id = spot.get("schema_id")
    names = action_names(str(schema_id))
    if tuple(spot.get("action_names") or ()) != names:
        raise SchemaError("fixture action_names do not match the schema")
    width = len(names)
    for key in ("policy_probs", "reference_probs"):
        probs = spot.get(key)
        if not isinstance(probs, list) or len(probs) != width:
            raise SchemaError(f"{key} must be a length-{width} list")
    if not isinstance(spot.get("actions_before"), list):
        raise SchemaError("actions_before must be a list")
    if "mbb_per_hand" in spot or "aivat_bb_per_100" in spot:
        raise SchemaError("fixture must not carry a strength number")
    return str(schema_id)

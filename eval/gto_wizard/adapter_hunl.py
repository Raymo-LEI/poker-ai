"""Map A5/A6 action ids onto the locked standard HUNL table.

``HunlTable`` stays width 5. Quarter-pot is an evaluation-only id (5) that
uses the same floor pot-fraction rule as half-pot and pot: add
``pot // 4`` chips, then accept the raise only when PokerKit does. Illegal
and off-tree sizes fail closed. ``SnapshotPolicy`` weight files are refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from alphanl.transformer.hunl import HunlTable
from alphanl.transformer.schema import HandSnapshot

from .schema import QUARTER_POT_ID, SCHEMA_A6, SchemaError, action_index, action_names


class AdapterError(ValueError):
    """The policy action cannot be sent as a legal HUNL action."""


class FakeSnapshot:
    """In-memory distribution. This is not a checkpoint and has no weights."""

    def __init__(self, schema_id: str, probs: Sequence[float]) -> None:
        names = action_names(schema_id)
        if len(probs) != len(names):
            raise SchemaError("fake snapshot width does not match the schema")
        self.schema_id = schema_id
        self.probs = tuple(float(p) for p in probs)

    def probabilities(self, snapshot: HandSnapshot) -> tuple[float, ...]:
        if not isinstance(snapshot, HandSnapshot):
            raise AdapterError("FakeSnapshot requires a HandSnapshot")
        return self.probs


def refuse_weight_source(policy) -> None:
    """Fail before any strategy archive or AutoDL directory can be opened."""

    if isinstance(policy, (str, Path)):
        raise AdapterError("refusing a filesystem policy path")
    module = type(policy).__module__
    if type(policy).__name__ == "SnapshotPolicy" and module.startswith("alphanl.cfr"):
        raise AdapterError("refusing SnapshotPolicy weights in the prep scaffold")


def new_table(seed: int) -> HunlTable:
    table = HunlTable(seed=int(seed), fast_evaluator=False)
    table.reset()
    return table


def _quarter_added(table: HunlTable) -> int | None:
    state = table.state
    actor = state.actor_index
    if actor is None:
        return None
    pot = int(state.total_pot_amount)
    added = pot // 4
    if added <= 0 or added > int(state.stacks[actor]):
        return None
    target = int(state.bets[actor]) + added
    if target <= max(int(bet) for bet in state.bets):
        return None
    if not table.can_raise_to(target):
        return None
    return added


def chips_added(table: HunlTable, schema_id: str, action_id: int) -> int | None:
    names = action_names(schema_id)
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        return None
    if not 0 <= action_id < len(names):
        return None
    if action_id == QUARTER_POT_ID:
        if schema_id != SCHEMA_A6:
            return None
        return _quarter_added(table)
    return table.action_chips(action_id)


def legal_action_ids(table: HunlTable, schema_id: str) -> tuple[int, ...]:
    legal = list(table.legal_action_ids())
    if schema_id == SCHEMA_A6 and _quarter_added(table) is not None:
        legal.append(QUARTER_POT_ID)
    return tuple(legal)


def to_wizard_action(table: HunlTable, schema_id: str, action_id: int) -> dict:
    """Return ``{"action": "f"|"k"|"c"|"b", "amount": ...}`` without applying it."""

    added = chips_added(table, schema_id, action_id)
    if added is None or action_id not in legal_action_ids(table, schema_id):
        raise AdapterError(f"illegal {schema_id} action {action_id}")
    if action_id == 0:
        return {"action": "f"}
    if action_id == 1:
        seat = table.current_player()
        kind = "c" if table.snapshot(seat).to_call > 0 else "k"
        return {"action": kind}
    street_total = int(table.state.bets[table.current_player()]) + int(added)
    return {"action": "b", "amount": street_total}


def apply_action_name(table: HunlTable, schema_id: str, name: str) -> None:
    action_id = action_index(schema_id, name)
    added = chips_added(table, schema_id, action_id)
    if added is None:
        raise AdapterError(f"cannot replay illegal action {name}")
    if action_id == QUARTER_POT_ID:
        target = int(table.state.bets[table.current_player()]) + int(added)
        table.step_raise_to(target)
        return
    table.step(action_id)


def action_for_street_total(table: HunlTable, schema_id: str, street_total: int) -> int:
    """Map one cumulative street bet onto a single legal bucket, or fail closed."""

    if isinstance(street_total, bool) or not isinstance(street_total, int):
        raise AdapterError("street total must be an int")
    matches = []
    for action_id in legal_action_ids(table, schema_id):
        spec = to_wizard_action(table, schema_id, action_id)
        if spec.get("action") == "b" and spec.get("amount") == street_total:
            matches.append(action_id)
    if len(matches) != 1:
        raise AdapterError("off-tree or ambiguous bet")
    return matches[0]


def prepare_spot(spot: Mapping) -> dict:
    schema_id = str(spot["schema_id"])
    action_names(schema_id)
    table = new_table(int(spot["seed"]))
    for name in spot["actions_before"]:
        apply_action_name(table, schema_id, str(name))
    if table.is_over():
        raise AdapterError("fixture ended the hand before a decision")
    actor = table.current_player()
    if actor != int(spot["actor_seat"]):
        raise AdapterError("actor seat does not match the replay")
    snapshot = table.snapshot(actor)
    return {
        "table": table,
        "snapshot": snapshot,
        "schema_id": schema_id,
        "legal_action_ids": legal_action_ids(table, schema_id),
    }

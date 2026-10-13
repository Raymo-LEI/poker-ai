"""Rebuild a ``HandSnapshot`` from AlphaNLHoldem's rlcard history.

``NlHoldemEnvWrapper.history`` stores ``[seat, action_id, legal_ids]`` per
street and does not store chip amounts. This module replays the rlcard 1.1.0
heads-up chip rules (including ``int(pot / 2)`` and the cumulative ``in_chips``
pot) so bet sizes become pot fractions.

``PokerkitTable`` does not reuse the replay. It calls ``make_action_event``
with the real chip delta and passes the resulting events to ``HandSnapshot``.
The tokenizer only reads that snapshot.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .buckets import pot_fraction_bucket
from .schema import (
    NUM_ACTIONS,
    ActionEvent,
    HandSnapshot,
    normalize_cards,
)

SMALL_BLIND_CHIPS = 1
BIG_BLIND_CHIPS = 2
STARTING_STACK = 100


def stage_value(stage: Any) -> int:
    if hasattr(stage, "value"):
        return int(stage.value)
    return int(stage)


def legal_action_ids(legal: Any) -> Tuple[int, ...]:
    """Accept enum lists, int lists, or the OrderedDict rlcard puts on observations."""

    if legal is None:
        return tuple()
    if isinstance(legal, Mapping):
        items = list(legal.keys())
    else:
        items = list(legal)
    out = []
    for action in items:
        out.append(int(action.value if hasattr(action, "value") else action))
    return tuple(a for a in out if 0 <= a < NUM_ACTIONS)


def infer_sb_seat(all_chips: Sequence[float], small_blind: int = SMALL_BLIND_CHIPS) -> int:
    """Seat that has posted exactly the small blind. Valid at hand start only."""

    matches = [i for i, chips in enumerate(all_chips) if int(chips) == int(small_blind)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one seat with {small_blind} chips posted, got all_chips={list(all_chips)}"
        )
    return int(matches[0])


def infer_action_type(
    action_id: Optional[int],
    *,
    to_call_before: float,
    chips_added: float,
    stack_after: float,
) -> str:
    """Map an abstract action plus the chip result onto a semantic action type."""

    if action_id == 0:
        return "fold"
    if chips_added > 0 and stack_after <= 0:
        return "all_in"
    if action_id == 4:
        return "all_in"
    if action_id == 1 and to_call_before <= 0:
        return "check"
    if action_id == 1:
        return "call"
    if action_id in (2, 3) and to_call_before <= 0:
        return "bet"
    if action_id in (2, 3):
        return "raise"
    if to_call_before <= 0 and chips_added <= 0:
        return "check"
    if to_call_before > 0 and chips_added <= to_call_before:
        return "call"
    if to_call_before <= 0:
        return "bet"
    return "raise"


def make_action_event(
    *,
    street: int,
    actor_seat: int,
    me_seat: int,
    sb_seat: int,
    action_id: Optional[int],
    action_type: Optional[str],
    pot_before: float,
    chips_added: float,
    to_call_before: float,
    stack_before: float,
    legal_action_ids: Sequence[int] = (),
) -> ActionEvent:
    """Single constructor for rlcard replay and a future pokerkit adapter."""

    pot_before_f = float(pot_before)
    chips_added_f = float(chips_added)
    fraction = 0.0 if pot_before_f <= 0 else chips_added_f / pot_before_f
    stack_after = float(stack_before) - chips_added_f
    kind = action_type or infer_action_type(
        action_id,
        to_call_before=float(to_call_before),
        chips_added=chips_added_f,
        stack_after=stack_after,
    )
    if kind not in ("fold", "check", "call", "bet", "raise", "all_in"):
        raise ValueError(f"unknown action_type {kind!r}")
    return ActionEvent(
        street=int(street),
        actor_seat=int(actor_seat),
        position="SB" if int(actor_seat) == int(sb_seat) else "BB",
        actor_is_me=int(actor_seat) == int(me_seat),
        action_id=None if action_id is None else int(action_id),
        action_type=kind,
        pot_before=pot_before_f,
        chips_added=chips_added_f,
        pot_fraction=fraction,
        size_bucket=pot_fraction_bucket(fraction),
        to_call_before=float(to_call_before),
        stack_before=float(stack_before),
        legal_action_ids=tuple(int(a) for a in legal_action_ids),
    )


def _apply_abstract_action(
    action_id: int,
    seat: int,
    raised: List[int],
    in_chips: List[int],
    remained: List[int],
) -> int:
    """Replay one rlcard 1.1.0 action. Returns chips actually added."""

    pot_before = in_chips[0] + in_chips[1]
    if action_id == 0:
        return 0
    if action_id == 1:
        diff = max(raised) - raised[seat]
        raised[seat] = max(raised)
        added = min(diff, remained[seat])
    elif action_id == 2:
        quantity = int(pot_before / 2)
        raised[seat] = raised[seat] + quantity
        added = min(quantity, remained[seat])
    elif action_id == 3:
        quantity = int(pot_before)
        raised[seat] = raised[seat] + quantity
        added = min(quantity, remained[seat])
    elif action_id == 4:
        added = remained[seat]
        raised[seat] = raised[seat] + added
    else:
        raise ValueError(f"action id {action_id} is outside the 5-way abstraction")
    in_chips[seat] += added
    remained[seat] -= added
    return added


def replay_events(
    history: Sequence[Sequence[Sequence[Any]]],
    *,
    me_seat: int,
    sb_seat: int,
    starting_stack: int = STARTING_STACK,
    small_blind: int = SMALL_BLIND_CHIPS,
    big_blind: int = BIG_BLIND_CHIPS,
) -> Tuple[ActionEvent, ...]:
    """Replay wrapper-format history into semantic events.

    ``history[street]`` entries are ``[seat, action_id, legal_ids]``, which is
    what ``NlHoldemEnvWrapper`` appends.
    """

    if len(history) < 4:
        raise ValueError("history must have four streets (empty lists allowed)")
    bb_seat = 1 - int(sb_seat)
    in_chips = [0, 0]
    remained = [int(starting_stack), int(starting_stack)]
    in_chips[int(sb_seat)] = int(small_blind)
    remained[int(sb_seat)] -= int(small_blind)
    in_chips[bb_seat] = int(big_blind)
    remained[bb_seat] -= int(big_blind)

    events: List[ActionEvent] = []
    for street, actions in enumerate(list(history)[:4]):
        raised = list(in_chips) if street == 0 else [0, 0]
        for record in actions:
            seat = int(record[0])
            action_id = int(record[1])
            legal = tuple(int(a) for a in record[2]) if len(record) > 2 else tuple()
            pot_before = in_chips[0] + in_chips[1]
            to_call = max(0, max(raised) - raised[seat])
            stack_before = remained[seat]
            added = _apply_abstract_action(action_id, seat, raised, in_chips, remained)
            events.append(
                make_action_event(
                    street=street,
                    actor_seat=seat,
                    me_seat=me_seat,
                    sb_seat=sb_seat,
                    action_id=action_id,
                    action_type=None,
                    pot_before=pot_before,
                    chips_added=added,
                    to_call_before=to_call,
                    stack_before=stack_before,
                    legal_action_ids=legal,
                )
            )
    return tuple(events)


def _raw_obs_from_env(env: Any) -> Mapping[str, Any]:
    last = env.last_obs
    extracted = last[0] if isinstance(last, tuple) else last
    if isinstance(extracted, Mapping) and "raw_obs" in extracted:
        return extracted["raw_obs"]
    if isinstance(extracted, Mapping):
        return extracted
    raise TypeError("env.last_obs does not contain a raw observation dict")


def build_snapshot(
    history: Sequence[Sequence[Sequence[Any]]],
    raw_obs: Mapping[str, Any],
    me_seat: int,
    sb_seat: int,
    *,
    big_blind: float = BIG_BLIND_CHIPS,
    small_blind: int = SMALL_BLIND_CHIPS,
    starting_stack: float = STARTING_STACK,
) -> HandSnapshot:
    """Decision snapshot for ``me_seat`` from wrapper history plus current raw obs."""

    events = replay_events(
        history,
        me_seat=me_seat,
        sb_seat=sb_seat,
        starting_stack=int(starting_stack),
        small_blind=int(small_blind),
        big_blind=int(big_blind),
    )
    hole = normalize_cards(list(raw_obs["hand"]))
    if len(hole) != 2:
        raise ValueError(f"expected two hole cards, got {hole}")
    board = normalize_cards(list(raw_obs.get("public_cards") or []))
    all_chips = [float(c) for c in raw_obs["all_chips"]]
    stakes = tuple(float(c) for c in raw_obs["stakes"])
    if len(stakes) != 2 or len(all_chips) != 2:
        raise ValueError("this step only supports heads-up stacks")
    street = stage_value(raw_obs["stage"])
    if street > 3:
        street = 3
    to_call = max(0.0, max(all_chips) - all_chips[int(me_seat)])
    return HandSnapshot(
        events=events,
        hole_cards=(hole[0], hole[1]),
        board_cards=board,
        me_seat=int(me_seat),
        sb_seat=int(sb_seat),
        street=int(street),
        pot=float(raw_obs["pot"]),
        stacks=(float(stakes[0]), float(stakes[1])),
        to_call=to_call,
        legal_action_ids=legal_action_ids(raw_obs["legal_actions"]),
        big_blind=float(big_blind),
        starting_stack=float(starting_stack),
    )


class NlEnvTracker:
    """Read an AlphaNL ``NlHoldemEnvWrapper`` (or a duck-typed stand-in).

    Call ``start`` after ``reset``, while the history is empty and only the
    blinds are in the pot. ``snapshot`` can then be called on any decision.
    The wrapper itself appends to ``env.history`` inside ``step``.
    """

    def __init__(self) -> None:
        self.sb_seat: Optional[int] = None
        self.big_blind = float(BIG_BLIND_CHIPS)
        self.small_blind = SMALL_BLIND_CHIPS
        self.starting_stack = float(STARTING_STACK)

    def start(
        self,
        env: Any,
        *,
        small_blind: int = SMALL_BLIND_CHIPS,
        big_blind: float = BIG_BLIND_CHIPS,
        starting_stack: float = STARTING_STACK,
    ) -> None:
        raw = _raw_obs_from_env(env)
        self.small_blind = int(small_blind)
        self.big_blind = float(big_blind)
        self.starting_stack = float(starting_stack)
        self.sb_seat = infer_sb_seat(raw["all_chips"], small_blind=self.small_blind)

    def snapshot(self, env: Any, me_seat: Optional[int] = None) -> HandSnapshot:
        if self.sb_seat is None:
            raise RuntimeError("call start() after reset before snapshot()")
        raw = _raw_obs_from_env(env)
        if me_seat is None:
            me_seat = int(raw["current_player"])
        return build_snapshot(
            env.history,
            raw,
            int(me_seat),
            int(self.sb_seat),
            big_blind=self.big_blind,
            small_blind=self.small_blind,
            starting_stack=self.starting_stack,
        )

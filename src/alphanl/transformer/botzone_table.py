"""SelfPlayEnv surface over :class:`alphanl.botzone.table.BotzoneTable`.

Policy ids stay the five-wide head. The integer that is applied comes only
from :func:`alphanl.botzone.actions.response_for_action`, and the mask comes
only from :func:`alphanl.botzone.actions.legal_action_ids`. Both of those
already drop an id unless ``legal_response`` accepts it. This module does
not invent a second legality check, and it does not hide a free fold the
way PokerKit does.

Cards enter ``HandSnapshot`` as the existing suit-then-rank strings
(``SA``). Wiki ids are ``rank = n // 4 + 2`` and ``suit = n % 4`` in the
order hearts, diamonds, spades, clubs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from alphanl.botzone import actions as botzone_actions
from alphanl.botzone.legal import (
    ALL_IN,
    BIG_BLIND,
    CHECK_CALL,
    FOLD,
    SMALL_BLIND,
    STARTING_STACK,
    to_call,
)
from alphanl.botzone.table import BotzoneTable

from .history import make_action_event
from .schema import ActionEvent, HandSnapshot, normalize_card

# n % 4. Wiki order: hearts, diamonds, spades, clubs.
_SUITS = "HDSC"
_RANKS = "23456789TJQKA"


def botzone_id_to_card(card: int) -> str:
    """Wiki card id ``0..51`` to suit-then-rank (``H2`` for id 0)."""

    if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card <= 51:
        raise ValueError(f"card id must be an int in 0..51, got {card!r}")
    return _SUITS[card % 4] + _RANKS[card // 4]


def card_to_botzone_id(card: str) -> int:
    """Suit-then-rank or rank-then-suit text to a wiki card id."""

    text = normalize_card(card)
    return _RANKS.index(text[1]) * 4 + _SUITS.index(text[0])


@dataclass(frozen=True)
class _TakenAction:
    street: int
    actor_seat: int
    action_id: int
    pot_before: int
    chips_added: int
    to_call_before: int
    stack_before: int
    legal_action_ids: Tuple[int, ...]


class BotzoneEnvTable:
    """Heads-up Botzone simple game, with the same methods as the other tables.

    Blinds are the wiki simple game: 50 / 100, both seats start at 20000.
    ``BotzoneTable`` posts those amounts itself; a different triple is rejected
    so a checkpoint cannot describe a game this table does not play.
    """

    def __init__(
        self,
        *,
        seed: int,
        starting_stack: float = STARTING_STACK,
        small_blind: float = SMALL_BLIND,
        big_blind: float = BIG_BLIND,
        dealer: int = 0,
    ) -> None:
        self.small_blind = _whole_chips(small_blind, "small_blind")
        self.big_blind = _whole_chips(big_blind, "big_blind")
        self.starting_stack = _whole_chips(starting_stack, "starting_stack")
        if (self.small_blind, self.big_blind, self.starting_stack) != (
            SMALL_BLIND,
            BIG_BLIND,
            STARTING_STACK,
        ):
            raise ValueError(
                "botzone requires small_blind=50, big_blind=100, starting_stack=20000, "
                f"got small_blind={self.small_blind}, big_blind={self.big_blind}, "
                f"starting_stack={self.starting_stack}"
            )
        if dealer not in (0, 1):
            raise ValueError(f"dealer must be 0 or 1, got {dealer}")
        self.seed = int(seed)
        self._table = BotzoneTable(rng=random.Random(self.seed), dealer=dealer)
        self._records: list[_TakenAction] = []

    def rng_state(self) -> Dict[str, Any]:
        return self._table.rng_state()

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        self._table.set_rng_state(state)

    def reset(self, deck: Optional[Sequence] = None) -> None:
        self._table.start_hand(deck=_coerce_deck(deck))
        self._records = []

    def is_over(self) -> bool:
        return bool(self._require().hand_over)

    def current_player(self) -> int:
        state = self._require()
        if state.hand_over:
            raise RuntimeError("hand is over")
        return int(state.actor)

    @property
    def sb_seat(self) -> int:
        return (int(self._table.dealer) + 1) % 2

    @property
    def street(self) -> int:
        return int(self._require().street)

    def legal_action_ids(self) -> Tuple[int, ...]:
        state = self._require()
        if state.hand_over:
            return ()
        return tuple(botzone_actions.legal_action_ids(state))

    def step(self, action_id: int) -> None:
        state = self._require()
        if state.hand_over:
            raise ValueError("hand is over")
        action = int(action_id)
        legal = botzone_actions.legal_action_ids(state)
        if action not in legal:
            raise ValueError(f"action {action} is not legal ({legal})")
        response = botzone_actions.response_for_action(state, action)
        if response is None:
            raise RuntimeError(f"action {action} was listed legal but has no Botzone response")
        actor = int(state.actor)
        owed = int(to_call(state, actor))
        stack_before = int(state.stacks[actor])
        record = _TakenAction(
            street=int(state.street),
            actor_seat=actor,
            action_id=action,
            pot_before=int(state.pot),
            chips_added=_chips_put_in(int(response), owed, stack_before),
            to_call_before=owed,
            stack_before=stack_before,
            legal_action_ids=tuple(legal),
        )
        self._table.act(int(response))
        self._records.append(record)

    def snapshot(self, me_seat: int) -> HandSnapshot:
        state = self._require()
        seat = int(me_seat)
        if seat not in (0, 1):
            raise ValueError(f"me_seat must be 0 or 1, got {me_seat}")
        hole = self.hole_cards(seat)
        street = int(state.street)
        if street > 3:
            street = 3
        owed = 0 if state.hand_over else int(to_call(state, seat))
        return HandSnapshot(
            events=self.events_for(seat),
            hole_cards=(hole[0], hole[1]),
            board_cards=self.board_cards(),
            me_seat=seat,
            sb_seat=int(self.sb_seat),
            street=street,
            pot=float(state.pot),
            stacks=(float(state.stacks[0]), float(state.stacks[1])),
            to_call=float(owed),
            legal_action_ids=self.legal_action_ids(),
            big_blind=float(self.big_blind),
            starting_stack=float(self.starting_stack),
        )

    def events_for(self, me_seat: int) -> Tuple[ActionEvent, ...]:
        self._require()
        seat = int(me_seat)
        sb_seat = int(self.sb_seat)
        return tuple(
            make_action_event(
                street=record.street,
                actor_seat=record.actor_seat,
                me_seat=seat,
                sb_seat=sb_seat,
                action_id=record.action_id,
                action_type=None,
                pot_before=record.pot_before,
                chips_added=record.chips_added,
                to_call_before=record.to_call_before,
                stack_before=record.stack_before,
                legal_action_ids=record.legal_action_ids,
            )
            for record in self._records
        )

    def payoffs(self) -> Tuple[float, float]:
        values = self._table.payoffs
        return (float(values[0]), float(values[1]))

    def hole_cards(self, seat: int) -> Tuple[str, str]:
        self._require()
        cards = self._table.holes[int(seat)]
        return (botzone_id_to_card(cards[0]), botzone_id_to_card(cards[1]))

    def board_cards(self) -> Tuple[str, ...]:
        self._require()
        return tuple(botzone_id_to_card(card) for card in self._table.board)

    def state_dict(self) -> Dict[str, Any]:
        import pickle

        self._require()
        return {
            "version": 1,
            "table": pickle.dumps(self._table),
            "records": [_record_dict(record) for record in self._records],
            "blinds": (self.small_blind, self.big_blind),
            "starting_stack": self.starting_stack,
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import pickle

        if int(blob.get("version", 0)) != 1 or "table" not in blob:
            raise ValueError(f"unsupported botzone table state: {blob.get('version')}")
        if "blinds" not in blob or "starting_stack" not in blob:
            raise ValueError("botzone table checkpoint is missing blinds or starting_stack")
        if tuple(blob["blinds"]) != (self.small_blind, self.big_blind):
            raise ValueError(
                f"checkpoint blinds {tuple(blob['blinds'])} do not match this table "
                f"({self.small_blind}, {self.big_blind})"
            )
        if int(blob["starting_stack"]) != int(self.starting_stack):
            raise ValueError(
                f"checkpoint starting_stack {blob['starting_stack']} does not match "
                f"this table ({self.starting_stack})"
            )
        self._table = pickle.loads(blob["table"])
        self._records = [_record_from(item) for item in blob["records"]]

    def _require(self):
        if self._table.state is None:
            raise RuntimeError("call reset() before using the table")
        return self._table.state


def _chips_put_in(response: int, owed: int, stack_before: int) -> int:
    """Chips ``apply`` moves from the actor's stack for this response.

    Fold adds nothing. Check/call adds the chips owed. All-in adds the
    whole remaining stack. A positive response is itself the extra chips.
    """

    if response == FOLD:
        return 0
    if response == CHECK_CALL:
        return int(owed)
    if response == ALL_IN:
        return int(stack_before)
    if response <= 0:
        raise ValueError(f"response {response} does not move a known chip amount")
    return int(response)


def _whole_chips(value: float, name: str) -> int:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) > 1e-6:
        raise ValueError(f"{name} must be a whole number of chips, got {value}")
    chips = int(rounded)
    if chips <= 0:
        raise ValueError(f"{name} must be positive, got {chips}")
    return chips


def _coerce_deck(deck: Optional[Sequence]) -> Optional[Tuple[int, ...]]:
    if deck is None:
        return None
    cards = tuple(deck)
    if not cards:
        raise ValueError("deck is empty")
    first = cards[0]
    if isinstance(first, str):
        return tuple(card_to_botzone_id(card) for card in cards)
    return tuple(int(card) for card in cards)


def _record_dict(record: _TakenAction) -> Dict[str, Any]:
    return {
        "street": record.street,
        "actor_seat": record.actor_seat,
        "action_id": record.action_id,
        "pot_before": record.pot_before,
        "chips_added": record.chips_added,
        "to_call_before": record.to_call_before,
        "stack_before": record.stack_before,
        "legal_action_ids": list(record.legal_action_ids),
    }


def _record_from(item: Dict[str, Any]) -> _TakenAction:
    return _TakenAction(
        street=int(item["street"]),
        actor_seat=int(item["actor_seat"]),
        action_id=int(item["action_id"]),
        pot_before=int(item["pot_before"]),
        chips_added=int(item["chips_added"]),
        to_call_before=int(item["to_call_before"]),
        stack_before=int(item["stack_before"]),
        legal_action_ids=tuple(int(action) for action in item["legal_action_ids"]),
    )

"""Backend-neutral hand state for the betting-history tokenizer.

rlcard (AlphaNLHoldem's env) and ``PokerkitTable`` both produce
``HandSnapshot`` values. Chip amounts are in chips, not big blinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

NUM_ACTIONS = 5
ACTION_NAMES: Tuple[str, ...] = (
    "fold",
    "check_call",
    "raise_half_pot",
    "raise_pot",
    "all_in",
)
ACTION_TYPES: Tuple[str, ...] = ("fold", "check", "call", "bet", "raise", "all_in")
POSITIONS: Tuple[str, ...] = ("SB", "BB")
STREETS: Tuple[str, ...] = ("preflop", "flop", "turn", "river")

SUITS = "CDHS"
RANKS = "23456789TJQKA"


@dataclass(frozen=True)
class ActionEvent:
    """One voluntary action already taken in the hand.

    ``action_id`` is the rlcard 5-way id when the source is that abstraction,
    and ``None`` when a backend only knows a semantic ``action_type`` plus a
    chip amount (the pokerkit path). ``pot_fraction`` is chips added divided
    by the pot before the action.
    """

    street: int
    actor_seat: int
    position: str
    actor_is_me: bool
    action_id: Optional[int]
    action_type: str
    pot_before: float
    chips_added: float
    pot_fraction: float
    size_bucket: int
    to_call_before: float
    stack_before: float
    legal_action_ids: Tuple[int, ...]


@dataclass(frozen=True)
class HandSnapshot:
    """Decision-time view for the seat about to act (``me_seat``).

    Hole cards are that seat's cards only. ``stacks`` is remaining chips by
    absolute seat. ``legal_action_ids`` indexes the 5-way policy head.
    """

    events: Tuple[ActionEvent, ...]
    hole_cards: Tuple[str, str]
    board_cards: Tuple[str, ...]
    me_seat: int
    sb_seat: int
    street: int
    pot: float
    stacks: Tuple[float, float]
    to_call: float
    legal_action_ids: Tuple[int, ...]
    big_blind: float
    starting_stack: float

    @property
    def bb_seat(self) -> int:
        return 1 - self.sb_seat

    @property
    def me_stack(self) -> float:
        return float(self.stacks[self.me_seat])

    @property
    def opp_stack(self) -> float:
        return float(self.stacks[1 - self.me_seat])

    @property
    def me_position(self) -> str:
        return "SB" if self.me_seat == self.sb_seat else "BB"


def normalize_card(card: str) -> str:
    """Return suit-then-rank (``SA``). Accepts rlcard ``SA`` and pokerkit ``As``."""

    text = card.strip().upper().replace(" ", "")
    if len(text) != 2:
        raise ValueError(f"card must be two characters, got {card!r}")
    a, b = text[0], text[1]
    suits = set("SHDC")
    ranks = set(RANKS)
    if a in suits and b in ranks:
        return a + b
    if b in suits and a in ranks:
        return b + a
    raise ValueError(f"unrecognized card {card!r}")


def normalize_cards(cards: Sequence[str]) -> Tuple[str, ...]:
    return tuple(normalize_card(c) for c in cards)

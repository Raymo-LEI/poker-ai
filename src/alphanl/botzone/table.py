"""Botzone TexasHoldem2p table: deal, button, blinds, showdown.

Betting stays in :mod:`alphanl.botzone.legal`. This module shuffles, burns,
posts the 50/100 blinds, and awards the pot. It does not speak to PPO or
to the rlcard / PokerKit backends.

Deal order, from a 52-card deck of wiki ids (``rank = n // 4 + 2``,
``suit = n % 4`` in hearts, diamonds, spades, clubs):

- small blind, big blind, small blind, big blind
- burn, flop, burn, turn, burn, river

The wiki does not mention a burn. Burning inside the table does not change
the distribution of a shuffled deck. Each hand starts both seats at 20000.
The button moves to the other seat after every hand. A chopped pot's
remainder chip goes to the lower seat index.
"""

from __future__ import annotations

import itertools
import random
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional, Sequence, Tuple

from .legal import STARTING_STACK, TOTAL_CHIPS, HandState, apply, legal_response, new_hand

_BOARD_AT = (5, 6, 7, 9, 11)
_REVEAL = (0, 3, 4, 5)
_SUITS = "hdsc"
_RANKS = "23456789TJQKA"
_UNSET = object()
_pokerkit_high = _UNSET


def card_text(card: int) -> str:
    """Wiki id to ``Ah``-style text. Ten is ``T``."""

    _check_card(card)
    return _RANKS[card // 4] + _SUITS[card % 4]


def parse_card(text: str) -> int:
    """``Ah``-style text to a wiki id. Ten is ``T``."""

    if len(text) != 2:
        raise ValueError(f"card must look like 'Ah', got {text!r}")
    rank = _RANKS.find(text[0].upper())
    suit = _SUITS.find(text[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"card must look like 'Ah', got {text!r}")
    return rank * 4 + suit


def preset_deck(
    dealer: int,
    holes: Sequence[Sequence[int]],
    board: Sequence[int],
) -> Tuple[int, ...]:
    """Deal order that gives these hole cards and this board.

    ``holes`` is indexed by seat. The small blind is dealt first. Burn
    cards and the undealt tail are the unused ids, lowest first.
    """

    if dealer not in (0, 1):
        raise ValueError(f"dealer must be 0 or 1, got {dealer}")
    if len(holes) != 2 or any(len(hole) != 2 for hole in holes) or len(board) != 5:
        raise ValueError("need two hole cards for each seat and five board cards")
    for card in (*holes[0], *holes[1], *board):
        _check_card(card)
    sb = (dealer + 1) % 2
    placed: list[Optional[int]] = [None] * 52
    placed[0] = holes[sb][0]
    placed[2] = holes[sb][1]
    placed[1] = holes[1 - sb][0]
    placed[3] = holes[1 - sb][1]
    for index, card in zip(_BOARD_AT, board):
        placed[index] = card
    specified = [card for card in placed if card is not None]
    if len(specified) != 9 or len(set(specified)) != 9:
        raise ValueError("hole cards and board must be nine distinct cards")
    unused = [card for card in range(52) if card not in set(specified)]
    filled = []
    cursor = 0
    for card in placed:
        if card is None:
            filled.append(unused[cursor])
            cursor += 1
        else:
            filled.append(card)
    return tuple(filled)


def split_pot(pot: int, winners: Tuple[int, ...] = (0, 1)) -> Tuple[int, int]:
    """Chips each seat receives from ``pot``.

    A single winner takes all of it. Both winners chop. The remainder
    chip goes to the lower seat index.
    """

    if isinstance(pot, bool) or not isinstance(pot, int) or pot < 0:
        raise ValueError(f"pot must be a non-negative int, got {pot!r}")
    seats = tuple(winners)
    if sorted(seats) not in ([0], [1], [0, 1]) or len(seats) != len(set(seats)):
        raise ValueError(f"winners must be one or both seats, got {winners!r}")
    if len(seats) == 1:
        awards = [0, 0]
        awards[seats[0]] = pot
        return (awards[0], awards[1])
    share, remainder = divmod(pot, 2)
    awards = [share, share]
    awards[min(seats)] += remainder
    return (awards[0], awards[1])


def hand_rank(hole: Sequence[int], board: Sequence[int]):
    """Comparable high-hand rank. Larger wins. Equal hands chop.

    Uses PokerKit 0.7.7 ``StandardHighHand`` when that exact version
    imports. Otherwise a local seven-card high-hand evaluator. Ace-to-five
    is the lowest straight.
    """

    cards = (*hole, *board)
    if len(hole) != 2 or len(board) != 5 or len(set(cards)) != 7:
        raise ValueError("showdown needs two hole cards and five distinct board cards")
    for card in cards:
        _check_card(card)
    ranked = _pokerkit_rank(hole, board)
    if ranked is not None:
        return ranked
    return _local_rank(cards)


class BotzoneTable:
    """One Botzone simple heads-up hand at a time.

    ``start_hand`` deals and posts the blinds. ``act`` forwards a response
    to :func:`alphanl.botzone.legal.apply`. Fold pays the pot immediately.
    If betting ends without a fold, the table runs the rest of the board
    and awards the pot. ``stacks[0] + stacks[1] + pot`` stays 40000.
    """

    def __init__(self, rng: Optional[random.Random] = None, dealer: int = 0) -> None:
        if dealer not in (0, 1):
            raise ValueError(f"dealer must be 0 or 1, got {dealer}")
        self._rng = random.Random() if rng is None else rng
        self._next_dealer = dealer
        self.state: Optional[HandState] = None
        self._holes: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self._board: Optional[Tuple[int, int, int, int, int]] = None
        self._winners: Tuple[int, ...] = ()
        self._showed = False

    def rng_state(self) -> Dict[str, Any]:
        """Deck RNG and the dealer used when ``start_hand`` is not given one."""

        return {"random": self._rng.getstate(), "next_dealer": int(self._next_dealer)}

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        version, seq, gauss = state["random"]
        self._rng.setstate((int(version), tuple(int(item) for item in seq), gauss))
        self._next_dealer = int(state["next_dealer"])

    def start_hand(
        self,
        dealer: Optional[int] = None,
        deck: Optional[Sequence[int]] = None,
    ) -> HandState:
        """Deal a new hand. Stacks reset to 20000. Omit ``dealer`` to rotate."""

        if dealer is None:
            dealer = self._next_dealer
        elif dealer not in (0, 1):
            raise ValueError(f"dealer must be 0 or 1, got {dealer}")
        cards = _shuffled(self._rng) if deck is None else _validate_deck(deck)
        self._holes, self._board = _holes_and_board(dealer, cards)
        self._showed = False
        self._winners = ()
        self.state = new_hand(dealer)
        self._next_dealer = 1 - dealer
        _conserved(self.state)
        return self.state

    def legal_responses(self) -> set:
        """Responses :func:`legal_response` currently allows. Empty if over."""

        state = self._require()
        if state.hand_over:
            return set()
        return legal_response(state)

    def act(self, response: int) -> HandState:
        """Apply one legal response. Showdown runs when betting is finished."""

        state = self._require()
        if state.hand_over:
            raise ValueError("hand is over")
        self.state = apply(state, response)
        if self.state.hand_over:
            self._settle()
        _conserved(self.state)
        return self.state

    @property
    def hand_over(self) -> bool:
        return self.state is not None and self.state.hand_over

    @property
    def dealer(self) -> int:
        return self._require().dealer

    @property
    def actor(self) -> int:
        return self._require().actor

    @property
    def street(self) -> int:
        return self._require().street

    @property
    def stacks(self) -> Tuple[int, int]:
        return self._require().stacks

    @property
    def pot(self) -> int:
        return self._require().pot

    @property
    def holes(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        self._require()
        assert self._holes is not None
        return self._holes

    @property
    def board(self) -> Tuple[int, ...]:
        """Community cards already dealt. The full five after showdown."""

        state = self._require()
        assert self._board is not None
        if self._showed:
            return self._board
        return self._board[: _REVEAL[state.street]]

    @property
    def winners(self) -> Tuple[int, ...]:
        """Seats who were awarded chips. Both seats when the pot is chopped."""

        if not self._require().hand_over:
            raise ValueError("hand is not over")
        return self._winners

    @property
    def payoffs(self) -> Tuple[int, int]:
        """Chip change from the 20000 each seat started with."""

        state = self._require()
        if not state.hand_over:
            raise ValueError("hand is not over")
        return (state.stacks[0] - STARTING_STACK, state.stacks[1] - STARTING_STACK)

    def _require(self) -> HandState:
        if self.state is None:
            raise ValueError("no hand in progress")
        return self.state

    def _settle(self) -> None:
        state = self._require()
        if state.folded[0] or state.folded[1]:
            if state.pot != 0 or (state.folded[0] and state.folded[1]):
                raise RuntimeError("folded hand was not paid out")
            self._winners = (0 if state.folded[1] else 1,)
            self._showed = False
            return
        assert self._holes is not None and self._board is not None
        self._showed = True
        left = hand_rank(self._holes[0], self._board)
        right = hand_rank(self._holes[1], self._board)
        if left > right:
            winners: Tuple[int, ...] = (0,)
        elif right > left:
            winners = (1,)
        else:
            winners = (0, 1)
        awarded = split_pot(state.pot, winners)
        stacks = (state.stacks[0] + awarded[0], state.stacks[1] + awarded[1])
        self._winners = winners
        self.state = HandState(
            dealer=state.dealer,
            street=state.street,
            stacks=stacks,
            street_commit=state.street_commit,
            pot=0,
            bound_b=state.bound_b,
            actor=state.actor,
            all_in=state.all_in,
            folded=state.folded,
            needs_action=(False, False),
            hand_over=True,
        )


def _holes_and_board(
    dealer: int,
    deck: Sequence[int],
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int, int, int, int]]:
    sb = (dealer + 1) % 2
    holes: list[Tuple[int, int]] = [(-1, -1), (-1, -1)]
    holes[sb] = (deck[0], deck[2])
    holes[1 - sb] = (deck[1], deck[3])
    board = (
        deck[_BOARD_AT[0]],
        deck[_BOARD_AT[1]],
        deck[_BOARD_AT[2]],
        deck[_BOARD_AT[3]],
        deck[_BOARD_AT[4]],
    )
    return (holes[0], holes[1]), board


def _shuffled(rng: random.Random) -> Tuple[int, ...]:
    cards = list(range(52))
    rng.shuffle(cards)
    return tuple(cards)


def _validate_deck(deck: Sequence[int]) -> Tuple[int, ...]:
    cards = tuple(deck)
    if len(cards) != 52 or len(set(cards)) != 52:
        raise ValueError("deck must be 52 distinct card ids in deal order")
    for card in cards:
        _check_card(card)
    return cards


def _check_card(card: int) -> None:
    if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card <= 51:
        raise ValueError(f"card id must be an int in 0..51, got {card!r}")


def _conserved(state: HandState) -> None:
    total = state.stacks[0] + state.stacks[1] + state.pot
    if total != TOTAL_CHIPS:
        raise RuntimeError(f"chip invariant broken: {total} != {TOTAL_CHIPS}")


def _load_pokerkit():
    global _pokerkit_high
    if _pokerkit_high is not _UNSET:
        return _pokerkit_high
    try:
        if version("pokerkit") != "0.7.7":
            _pokerkit_high = None
            return None
        from pokerkit import StandardHighHand
    except (ImportError, PackageNotFoundError):
        _pokerkit_high = None
        return None
    _pokerkit_high = StandardHighHand
    return _pokerkit_high


def _pokerkit_rank(hole: Sequence[int], board: Sequence[int]):
    hand_type = _load_pokerkit()
    if hand_type is None:
        return None
    return hand_type.from_game(
        "".join(card_text(card) for card in hole),
        "".join(card_text(card) for card in board),
    )


def _local_rank(cards: Sequence[int]) -> Tuple:
    best = None
    for combo in itertools.combinations(cards, 5):
        ranked = _score_five(combo)
        if best is None or ranked > best:
            best = ranked
    assert best is not None
    return best


def _score_five(cards: Sequence[int]) -> Tuple:
    ranks = [card // 4 + 2 for card in cards]
    flush = len({card % 4 for card in cards}) == 1
    straight, top = _straight_high(ranks)
    counts = sorted(
        ((ranks.count(rank), rank) for rank in set(ranks)),
        reverse=True,
    )
    if straight and flush:
        return (8, top)
    if counts[0][0] == 4:
        return (7, counts[0][1], counts[1][1])
    if counts[0][0] == 3 and counts[1][0] == 2:
        return (6, counts[0][1], counts[1][1])
    ordered = tuple(sorted(ranks, reverse=True))
    if flush:
        return (5, ordered)
    if straight:
        return (4, top)
    if counts[0][0] == 3:
        kickers = tuple(rank for rank in ordered if rank != counts[0][1])
        return (3, counts[0][1], kickers)
    if counts[0][0] == 2 and len(counts) > 1 and counts[1][0] == 2:
        high, low = counts[0][1], counts[1][1]
        kicker = next(rank for rank in ordered if rank != high and rank != low)
        return (2, high, low, kicker)
    if counts[0][0] == 2:
        kickers = tuple(rank for rank in ordered if rank != counts[0][1])
        return (1, counts[0][1], kickers)
    return (0, ordered)


def _straight_high(ranks: Sequence[int]) -> Tuple[bool, int]:
    unique = sorted(set(ranks))
    if len(unique) != 5:
        return False, 0
    if unique[-1] - unique[0] == 4:
        return True, unique[-1]
    if unique == [2, 3, 4, 5, 14]:
        return True, 5
    return False, 0

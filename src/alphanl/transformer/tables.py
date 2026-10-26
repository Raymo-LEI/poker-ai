"""Heads-up no-limit tables behind ``SelfPlayEnv``.

Both tables speak the same 5 abstract actions as rlcard 1.1.0 /
``NlHoldemEnvWrapper``:

=======  =====================================================
id       action
=======  =====================================================
0        fold
1        check or call
2        bet/raise by ``int(pot / 2)`` chips (rlcard half-pot)
3        bet/raise by ``pot`` chips (rlcard pot)
4        all-in
=======  =====================================================

``RlcardTable`` is the existing rlcard 1.1.0 game, including its
heads-up mistake: the small blind acts first on every street.
``PokerkitTable`` uses PokerKit. Seat 0 is the big blind and seat 1 is
the button / small blind. The small blind acts first preflop; the big
blind acts first on every later street. A sized action is legal only
when that chip amount is a legal PokerKit bet or raise, so an
under-minimum raise that rlcard would still offer is masked here.
History is recorded with ``make_action_event`` from the chips actually
put in, not from the rlcard replay.

``reset(deck=...)`` takes cards in deal order (first card is dealt
first). PokerKit burns one card before the flop, the turn, and the
river. rlcard does not. ``aligned_decks`` builds both orders from the
same hole cards and board.

Stacks are in chips. With the default blind of 2, 50bb / 100bb / 200bb
are 100 / 200 / 400 chips (``chips_for_depth``).

``backend="botzone"`` is a third table (:class:`alphanl.transformer.botzone_table.BotzoneEnvTable`).
It keeps these five policy ids, but the chip amount for each id comes from
``alphanl.botzone.actions`` (safe min-raise / large raise), not from
``int(pot / 2)``. rlcard and pokerkit sizing is unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .history import (
    BIG_BLIND_CHIPS,
    SMALL_BLIND_CHIPS,
    STARTING_STACK,
    build_snapshot,
    infer_sb_seat,
    legal_action_ids,
    make_action_event,
    replay_events,
    stage_value,
)
from .schema import NUM_ACTIONS, SUITS, RANKS, ActionEvent, HandSnapshot, normalize_card

# Abstract-action ids. Kept next to ACTION_NAMES so the head width cannot drift.
_FOLD = 0
_CHECK_CALL = 1
_HALF = 2
_POT = 3
_ALL_IN = 4

_SIZED_NUM = {_HALF: 1, _POT: 1}
_SIZED_DEN = {_HALF: 2, _POT: 1}

SUPPORTED_STACK_BB: Tuple[int, ...] = (50, 100, 200)


def chips_for_depth(stack_bb: int, big_blind: int = BIG_BLIND_CHIPS) -> int:
    """Chips for a 50 / 100 / 200 big-blind starting stack."""

    depth = int(stack_bb)
    if depth not in SUPPORTED_STACK_BB:
        raise ValueError(f"stack depth must be one of {SUPPORTED_STACK_BB} bb, got {stack_bb}")
    blind = _whole_chips(big_blind, "big_blind")
    return depth * blind


def standard_deck() -> Tuple[str, ...]:
    """52-card deck in suit-then-rank form (``SA``)."""

    return tuple(suit + rank for suit in SUITS for rank in RANKS)


def aligned_decks(
    hole_bb: Sequence[str],
    hole_sb: Sequence[str],
    board: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Deal orders that give both engines the same hole cards and board.

    Seat 0 is the big blind and seat 1 is the small blind. That is PokerKit's
    heads-up layout, and rlcard's layout when ``dealer_id`` is 0. Cards are
    dealt one at a time, big blind then small blind. Returns
    ``(rlcard_order, pokerkit_order)``. PokerKit inserts a burn before each
    board street; rlcard deals the board with no burns.
    """

    if len(hole_bb) != 2 or len(hole_sb) != 2 or len(board) != 5:
        raise ValueError("need two hole cards for each seat and five board cards")
    bb = tuple(normalize_card(card) for card in hole_bb)
    sb = tuple(normalize_card(card) for card in hole_sb)
    dealt_board = tuple(normalize_card(card) for card in board)
    hole_order = [bb[0], sb[0], bb[1], sb[1]]
    used = set(hole_order) | set(dealt_board)
    if len(used) != 9:
        raise ValueError("hole cards and board must be nine distinct cards")
    rest = [card for card in standard_deck() if card not in used]
    burns, tail = rest[:3], rest[3:]
    rlcard_order = tuple(hole_order + list(dealt_board) + rest)
    pokerkit_order = tuple(
        hole_order
        + [burns[0], dealt_board[0], dealt_board[1], dealt_board[2]]
        + [burns[1], dealt_board[3]]
        + [burns[2], dealt_board[4]]
        + tail
    )
    if len(rlcard_order) != 52 or len(pokerkit_order) != 52:
        raise RuntimeError("aligned deck was not 52 cards")
    return rlcard_order, pokerkit_order


def _whole_chips(value: float, name: str) -> int:
    number = float(value)
    rounded = round(number)
    if abs(number - rounded) > 1e-6:
        raise ValueError(f"{name} must be a whole number of chips, got {value}")
    chips = int(rounded)
    if chips <= 0:
        raise ValueError(f"{name} must be positive, got {chips}")
    return chips


def _coerce_numpy_random_state(state: Any) -> Tuple[Any, ...]:
    """``RandomState.get_state`` after a torch round-trip, ready for ``set_state``."""

    import numpy as np

    kind, keys, pos, has_gauss, cached = state
    array = np.asarray(keys)
    if array.dtype != np.uint32:
        array = array.astype(np.uint32)
    return (str(kind), array, int(pos), int(has_gauss), float(cached))


def _coerce_py_random_state(state: Any) -> Tuple[Any, ...]:
    version, seq, gauss = state
    return (int(version), tuple(int(item) for item in seq), gauss)


def _normalize_deck(deck: Sequence[str]) -> Tuple[str, ...]:
    cards = tuple(normalize_card(card) for card in deck)
    if len(cards) != 52 or len(set(cards)) != 52:
        raise ValueError("deck must be 52 distinct cards in deal order")
    return cards


@dataclass(frozen=True)
class _TakenAction:
    street: int
    actor_seat: int
    action_id: Optional[int]
    pot_before: int
    chips_added: int
    to_call_before: int
    stack_before: int
    legal_action_ids: Tuple[int, ...]


class RlcardTable:
    """rlcard 1.1.0 heads-up no-limit game.

    Blinds are the game's hardcoded 1 / 2. ``small_blind`` / ``big_blind``
    are stored for the snapshot and must stay 1 / 2 to match the chips.
    ``dealer_id=None`` lets rlcard move the button; ``dealer_id=0`` fixes
    seat 0 as the big blind so a hand can be lined up with PokerKit.
    """

    def __init__(
        self,
        *,
        seed: int,
        starting_stack: float = STARTING_STACK,
        small_blind: float = SMALL_BLIND_CHIPS,
        big_blind: float = BIG_BLIND_CHIPS,
        dealer_id: Optional[int] = None,
    ) -> None:
        self.small_blind = _whole_chips(small_blind, "small_blind")
        self.big_blind = _whole_chips(big_blind, "big_blind")
        if self.big_blind <= self.small_blind:
            raise ValueError("big_blind must be larger than small_blind")
        self.starting_stack = float(_whole_chips(starting_stack, "starting_stack"))
        self.starting_stacks = (int(self.starting_stack), int(self.starting_stack))
        self.dealer_id = None if dealer_id is None else int(dealer_id)
        self.seed = int(seed)
        self.rl = _make_rlcard_env(self.seed, int(self.starting_stack), self.dealer_id)
        self.history: List[list] = [[], [], [], []]
        self.state: Dict[str, Any] = {}
        self.player = 0
        self.sb_seat = 0
        self._dealt = False

    def reset(self, deck: Optional[Sequence[str]] = None) -> None:
        if self.dealer_id is not None:
            self.rl.game.dealer_id = int(self.dealer_id)
        if deck is None:
            self.state, self.player = self.rl.reset()
        else:
            self.state, self.player = _rlcard_reset_with_deck(self.rl, _normalize_deck(deck))
        self.player = int(self.player)
        self.history = [[], [], [], []]
        raw = self.state["raw_obs"]
        self.sb_seat = infer_sb_seat(raw["all_chips"], small_blind=self.small_blind)
        self._dealt = True

    def is_over(self) -> bool:
        self._require_hand()
        return bool(self.rl.game.is_over())

    def current_player(self) -> int:
        self._require_hand()
        if self.is_over():
            raise RuntimeError("hand is over")
        return int(self.player)

    @property
    def street(self) -> int:
        self._require_hand()
        return stage_value(self.state["raw_obs"]["stage"])

    def legal_action_ids(self) -> Tuple[int, ...]:
        self._require_hand()
        if self.is_over():
            return ()
        return legal_action_ids(self.state["raw_obs"]["legal_actions"])

    def step(self, action_id: int) -> None:
        legal = self.legal_action_ids()
        action = int(action_id)
        if action not in legal:
            raise ValueError(f"action {action} is not legal ({legal})")
        raw = self.state["raw_obs"]
        stage = stage_value(raw["stage"])
        actor = int(raw["current_player"])
        if stage <= 3:
            self.history[stage].append([actor, action, list(legal)])
        self.state, self.player = self.rl.step(action)
        self.player = int(self.player)

    def snapshot(self, me_seat: int) -> HandSnapshot:
        self._require_hand()
        return build_snapshot(
            self.history,
            self.state["raw_obs"],
            int(me_seat),
            int(self.sb_seat),
            big_blind=float(self.big_blind),
            small_blind=self.small_blind,
            starting_stack=self.starting_stack,
        )

    def events_for(self, me_seat: int) -> Tuple[ActionEvent, ...]:
        self._require_hand()
        return replay_events(
            self.history,
            me_seat=int(me_seat),
            sb_seat=int(self.sb_seat),
            starting_stack=int(self.starting_stack),
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

    def payoffs(self) -> Tuple[float, float]:
        if not self.is_over():
            raise RuntimeError("payoffs are available after the hand ends")
        values = self.rl.get_payoffs()
        return (float(values[0]), float(values[1]))

    def hole_cards(self, seat: int) -> Tuple[str, ...]:
        self._require_hand()
        return tuple(card.get_index() for card in self.rl.game.players[int(seat)].hand)

    def board_cards(self) -> Tuple[str, ...]:
        self._require_hand()
        return tuple(card.get_index() for card in self.rl.game.public_cards)

    def chips_in_play(self) -> int:
        """Remained chips plus chips committed. rlcard never moves the pot back."""

        self._require_hand()
        return int(
            sum(player.remained_chips + player.in_chips for player in self.rl.game.players)
        )

    def rng_state(self) -> Dict[str, Any]:
        """Deck RNG and the button. Not the cards of the current hand.

        rlcard only draws ``dealer_id`` when it is still ``None``. A fresh
        table starts at ``None``, so the next ``reset`` would consume an
        extra integer unless this cursor is put back first.
        """

        dealer = self.rl.game.dealer_id
        if dealer is not None:
            dealer = int(dealer)
        return {"np_random": self.rl.np_random.get_state(), "dealer_id": dealer}

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        np_state = _coerce_numpy_random_state(state["np_random"])
        self.rl.np_random.set_state(np_state)
        game = self.rl.game
        if game.np_random is not self.rl.np_random:
            game.np_random.set_state(np_state)
        dealer = state["dealer_id"]
        game.dealer_id = None if dealer is None else int(dealer)

    def state_dict(self) -> Dict[str, Any]:
        import pickle

        self._require_hand()
        return {
            "version": 1,
            "rl": pickle.dumps(self.rl),
            "history": self.history,
            "state": self.state,
            "player": self.player,
            "sb_seat": self.sb_seat,
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import pickle

        if int(blob.get("version", 1)) != 1:
            raise ValueError(f"unsupported rlcard table state: {blob.get('version')}")
        self.rl = pickle.loads(blob["rl"])
        self.history = blob["history"]
        self.state = blob["state"]
        self.player = int(blob["player"])
        self.sb_seat = int(blob["sb_seat"])
        self._dealt = True

    def _require_hand(self) -> None:
        if not self._dealt:
            raise RuntimeError("call reset() before using the table")


class PokerkitTable:
    """Heads-up no-limit hold'em dealt by PokerKit.

    Seat 0 posts the big blind. Seat 1 is the button and posts the small
    blind, and is first to act preflop. Postflop the big blind is first.
    The button does not rotate: ``SelfPlayEnv`` already assigns the hero
    to a random seat each hand.

    Sized actions use the rlcard chip amounts (add ``int(pot / 2)`` or
    ``pot``), then PokerKit's minimum-raise rule decides whether that
    amount can be offered. Fold is absent when a check is free. The real
    chip delta is what ``make_action_event`` stores.
    """

    def __init__(
        self,
        *,
        seed: int,
        starting_stack: float = STARTING_STACK,
        small_blind: float = SMALL_BLIND_CHIPS,
        big_blind: float = BIG_BLIND_CHIPS,
        stacks: Optional[Sequence[float]] = None,
    ) -> None:
        self.small_blind = _whole_chips(small_blind, "small_blind")
        self.big_blind = _whole_chips(big_blind, "big_blind")
        if self.big_blind <= self.small_blind:
            raise ValueError("big_blind must be larger than small_blind")
        nominal = _whole_chips(starting_stack, "starting_stack")
        if stacks is None:
            self.starting_stacks = (nominal, nominal)
        else:
            if len(stacks) != 2:
                raise ValueError("heads-up tables take two stacks")
            self.starting_stacks = (
                _whole_chips(stacks[0], "stacks[0]"),
                _whole_chips(stacks[1], "stacks[1]"),
            )
        for stack in self.starting_stacks:
            if stack < self.big_blind:
                raise ValueError(
                    f"each stack must cover the big blind ({self.big_blind}), got {stack}"
                )
        self.starting_stack = float(nominal)
        self.seed = int(seed)
        self._py_random = random.Random(self.seed)
        self.state: Any = None
        self._records: List[_TakenAction] = []
        self.sb_seat = 1

    def reset(self, deck: Optional[Sequence[str]] = None) -> None:
        cards = None if deck is None else _pokerkit_cards(_normalize_deck(deck))
        self.state = _new_pokerkit_state(
            self._py_random,
            blinds=(self.small_blind, self.big_blind),
            min_bet=self.big_blind,
            stacks=self.starting_stacks,
            deck=cards,
        )
        self._records = []
        posted = [index for index, bet in enumerate(self.state.bets) if int(bet) == self.small_blind]
        if len(posted) != 1:
            raise RuntimeError(f"expected one small blind, bets={list(self.state.bets)}")
        self.sb_seat = int(posted[0])

    def is_over(self) -> bool:
        self._require_hand()
        return not bool(self.state.status)

    def current_player(self) -> int:
        self._require_hand()
        actor = self.state.actor_index
        if actor is None:
            raise RuntimeError("hand is over")
        return int(actor)

    @property
    def street(self) -> int:
        self._require_hand()
        index = self.state.street_index
        if index is None:
            return 0
        return int(index)

    def legal_action_ids(self) -> Tuple[int, ...]:
        self._require_hand()
        if self.is_over():
            return ()
        return tuple(action for action in range(NUM_ACTIONS) if self._chips_if_legal(action) is not None)

    def action_chips(self, action_id: int) -> Optional[int]:
        """Chips this abstract action would add, or ``None`` if it is illegal."""

        self._require_hand()
        if self.is_over():
            return None
        return self._chips_if_legal(int(action_id))

    def min_raise_to(self) -> Optional[int]:
        """PokerKit's minimum bet-to / raise-to, or ``None`` if a raise is closed."""

        self._require_hand()
        if self.is_over():
            return None
        amount = self.state.min_completion_betting_or_raising_to_amount
        return None if amount is None else int(amount)

    def can_raise_to(self, amount: int) -> bool:
        self._require_hand()
        if self.is_over():
            return False
        return bool(self.state.can_complete_bet_or_raise_to(int(amount)))

    def step(self, action_id: int) -> None:
        legal = self.legal_action_ids()
        action = int(action_id)
        if action not in legal:
            raise ValueError(f"action {action} is not legal ({legal})")
        state = self.state
        actor = int(state.actor_index)
        pot_before = int(state.total_pot_amount)
        bet_before = int(state.bets[actor])
        stack_before = int(state.stacks[actor])
        to_call = int(max(state.bets) - bet_before)
        street = 0 if state.street_index is None else int(state.street_index)
        added = self._apply(action, bet_before, stack_before, pot_before)
        self._records.append(
            _TakenAction(
                street=street,
                actor_seat=actor,
                action_id=action,
                pot_before=pot_before,
                chips_added=added,
                to_call_before=to_call,
                stack_before=stack_before,
                legal_action_ids=legal,
            )
        )

    def snapshot(self, me_seat: int) -> HandSnapshot:
        self._require_hand()
        hole = self.hole_cards(me_seat)
        if len(hole) != 2:
            raise ValueError(f"expected two hole cards for seat {me_seat}, got {hole}")
        stacks = (float(self.state.stacks[0]), float(self.state.stacks[1]))
        bets = [int(bet) for bet in self.state.bets]
        actor = int(me_seat)
        to_call = float(max(0, max(bets) - bets[actor]))
        street = self.street
        if street > 3:
            street = 3
        return HandSnapshot(
            events=self.events_for(me_seat),
            hole_cards=(hole[0], hole[1]),
            board_cards=self.board_cards(),
            me_seat=int(me_seat),
            sb_seat=int(self.sb_seat),
            street=int(street),
            pot=float(self.state.total_pot_amount),
            stacks=stacks,
            to_call=to_call,
            legal_action_ids=self.legal_action_ids(),
            big_blind=float(self.big_blind),
            starting_stack=float(self.starting_stack),
        )

    def events_for(self, me_seat: int) -> Tuple[ActionEvent, ...]:
        self._require_hand()
        return tuple(
            make_action_event(
                street=record.street,
                actor_seat=record.actor_seat,
                me_seat=int(me_seat),
                sb_seat=int(self.sb_seat),
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

    def wrapper_history(self) -> List[list]:
        """``NlHoldemEnvWrapper``-shaped history for the conv encoder."""

        history: List[list] = [[], [], [], []]
        for record in self._records:
            if 0 <= int(record.street) <= 3:
                history[int(record.street)].append(
                    [int(record.actor_seat), int(record.action_id), list(record.legal_action_ids)]
                )
        return history

    def rlcard_raw_obs(self, me_seat: int) -> Dict[str, Any]:
        """Fields ``encode_observation`` expects, for the seat about to act."""

        self._require_hand()
        seat = int(me_seat)
        if int(self.current_player()) != seat:
            raise RuntimeError("rlcard_raw_obs requires the acting seat")
        snap = self.snapshot(seat)
        bets = [int(bet) for bet in self.state.bets]
        return {
            "hand": list(snap.hole_cards),
            "public_cards": list(snap.board_cards),
            "stakes": [int(self.state.stacks[0]), int(self.state.stacks[1])],
            "all_chips": bets,
            "pot": float(self.state.total_pot_amount),
            "stage": int(snap.street),
            "legal_actions": list(self.legal_action_ids()),
            "current_player": seat,
        }

    def payoffs(self) -> Tuple[float, float]:
        if not self.is_over():
            raise RuntimeError("payoffs are available after the hand ends")
        return (float(self.state.payoffs[0]), float(self.state.payoffs[1]))

    def hole_cards(self, seat: int) -> Tuple[str, ...]:
        self._require_hand()
        return tuple(_card_text(card) for card in self.state.hole_cards[int(seat)])

    def board_cards(self) -> Tuple[str, ...]:
        self._require_hand()
        return tuple(_card_text(card) for card in self.state.get_board_cards(0))

    def chips_in_play(self) -> int:
        self._require_hand()
        if self.is_over():
            return int(sum(self.state.stacks))
        return int(sum(self.state.stacks) + self.state.total_pot_amount)

    def rng_state(self) -> Dict[str, Any]:
        """PokerKit deck RNG. The button does not rotate on this table."""

        return {"random": self._py_random.getstate()}

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        self._py_random.setstate(_coerce_py_random_state(state["random"]))

    def state_dict(self) -> Dict[str, Any]:
        import pickle

        self._require_hand()
        return {
            "version": 1,
            "state": pickle.dumps(self.state),
            "records": list(self._records),
            "sb_seat": self.sb_seat,
            "rng": self._py_random.getstate(),
            "starting_stacks": self.starting_stacks,
            "blinds": (self.small_blind, self.big_blind),
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import pickle

        if int(blob.get("version", 0)) != 1:
            raise ValueError(f"unsupported pokerkit table state: {blob.get('version')}")
        if tuple(blob["starting_stacks"]) != tuple(self.starting_stacks):
            raise ValueError("checkpoint stacks do not match this table")
        if tuple(blob["blinds"]) != (self.small_blind, self.big_blind):
            raise ValueError("checkpoint blinds do not match this table")
        self.state = pickle.loads(blob["state"])
        self._records = list(blob["records"])
        self.sb_seat = int(blob["sb_seat"])
        self._py_random.setstate(blob["rng"])

    def _chips_if_legal(self, action_id: int) -> Optional[int]:
        state = self.state
        actor = state.actor_index
        if actor is None:
            return None
        pot = int(state.total_pot_amount)
        bet = int(state.bets[actor])
        stack = int(state.stacks[actor])
        if action_id == _FOLD:
            return 0 if state.can_fold() else None
        if action_id == _CHECK_CALL:
            amount = state.checking_or_calling_amount
            return None if amount is None else int(amount)
        if action_id == _ALL_IN:
            if stack <= 0:
                return None
            target = bet + stack
            if target <= max(int(item) for item in state.bets):
                return None
            if not state.can_complete_bet_or_raise_to(target):
                return None
            return stack
        if action_id not in _SIZED_NUM:
            return None
        add = (pot * _SIZED_NUM[action_id]) // _SIZED_DEN[action_id]
        if add <= 0 or add > stack:
            return None
        target = bet + add
        if target <= max(int(item) for item in state.bets):
            return None
        if not state.can_complete_bet_or_raise_to(target):
            return None
        return add

    def _apply(self, action_id: int, bet_before: int, stack_before: int, pot_before: int) -> int:
        if action_id == _FOLD:
            self.state.fold()
            return 0
        if action_id == _CHECK_CALL:
            operation = self.state.check_or_call()
            return int(operation.amount)
        if action_id == _ALL_IN:
            target = bet_before + stack_before
        else:
            target = bet_before + (pot_before * _SIZED_NUM[action_id]) // _SIZED_DEN[action_id]
        self.state.complete_bet_or_raise_to(target)
        return int(target - bet_before)

    def _require_hand(self) -> None:
        if self.state is None:
            raise RuntimeError("call reset() before using the table")


def make_table(
    backend: str,
    *,
    seed: int,
    starting_stack: float = STARTING_STACK,
    small_blind: float = SMALL_BLIND_CHIPS,
    big_blind: float = BIG_BLIND_CHIPS,
    dealer_id: Optional[int] = None,
    stacks: Optional[Sequence[float]] = None,
):
    """Build the table ``SelfPlayEnv`` drives.

    ``backend`` is ``rlcard``, ``pokerkit``, or ``botzone``. Botzone blinds
    default to 50 / 100 with a 20000 starting stack (see
    ``resolve_env_limits``).
    """

    from .config import resolve_env_limits

    small_blind, big_blind, starting_stack = resolve_env_limits(
        backend, small_blind, big_blind, starting_stack
    )
    if backend == "rlcard":
        if stacks is not None:
            raise ValueError("rlcard heads-up uses one starting_stack for both seats")
        return RlcardTable(
            seed=seed,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            dealer_id=dealer_id,
        )
    if backend == "pokerkit":
        return PokerkitTable(
            seed=seed,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            stacks=stacks,
        )
    if backend == "botzone":
        if stacks is not None:
            raise ValueError("botzone heads-up resets both seats to 20000")
        from .botzone_table import BotzoneEnvTable

        return BotzoneEnvTable(
            seed=seed,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            dealer=0 if dealer_id is None else int(dealer_id),
        )
    raise ValueError("backend must be 'rlcard', 'pokerkit', or 'botzone'")


def _make_rlcard_env(seed: int, starting_stack: int, dealer_id: Optional[int]):
    try:
        import rlcard
    except ImportError as exc:
        raise ImportError(
            "rlcard is required for the rlcard backend. Install requirements-transformer.txt "
            "(rlcard==1.1.0 matches AlphaNLHoldem's 5-action game)."
        ) from exc
    config: Dict[str, Any] = {
        "seed": int(seed),
        "game_num_players": 2,
        "chips_for_each": int(starting_stack),
    }
    if dealer_id is not None:
        config["dealer_id"] = int(dealer_id)
    return rlcard.make("no-limit-holdem", config=config)


def _rlcard_reset_with_deck(rl_env: Any, deck: Sequence[str]):
    from rlcard.games.base import Card
    from rlcard.games.nolimitholdem.dealer import NolimitholdemDealer

    pile = [Card(card[0], card[1]) for card in reversed(deck)]
    had_override = "shuffle" in NolimitholdemDealer.__dict__
    original = NolimitholdemDealer.shuffle

    def _shuffle(dealer, _pile=pile) -> None:
        dealer.deck = list(_pile)

    NolimitholdemDealer.shuffle = _shuffle
    try:
        return rl_env.reset()
    finally:
        if had_override:
            NolimitholdemDealer.shuffle = original
        else:
            del NolimitholdemDealer.shuffle


def _pokerkit_cards(deck: Sequence[str]):
    from pokerkit import Card

    cards = []
    for card in deck:
        # normalize_card emits suit-then-rank; PokerKit parses rank-then-suit.
        rank_suit = card[1] + card[0].lower()
        cards.append(next(Card.parse(rank_suit)))
    return cards


def _card_text(card: Any) -> str:
    return f"{str(card.suit.value).upper()}{str(card.rank.value).upper()}"


def _new_pokerkit_state(rng: random.Random, *, blinds, min_bet: int, stacks, deck):
    import pokerkit.state as pokerkit_state
    from pokerkit import Automation, NoLimitTexasHoldem

    automations = (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.CARD_BURNING,
        Automation.HOLE_DEALING,
        Automation.BOARD_DEALING,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )
    original = pokerkit_state.shuffle

    def _shuffle(cards, _deck=deck, _rng=rng) -> None:
        if _deck is None:
            _rng.shuffle(cards)
            return
        cards.clear()
        cards.extend(_deck)

    pokerkit_state.shuffle = _shuffle
    try:
        return NoLimitTexasHoldem.create_state(
            automations,
            True,
            0,
            (int(blinds[0]), int(blinds[1])),
            int(min_bet),
            (int(stacks[0]), int(stacks[1])),
            2,
        )
    finally:
        pokerkit_state.shuffle = original

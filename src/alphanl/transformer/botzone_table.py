"""Heads-up NLHE table with Botzone TexasHoldem2p rules (``env.rules = "botzone"``).

Selected with ``env.backend = "pokerkit"`` + ``env.rules = "botzone"``. PokerKit's
state machine hard-codes standard heads-up order (BB first postflop) and its
own min-raise rule, so the betting loop here is plain Python driven by
``alphanl.botzone.rules`` (the same functions the Botzone bot adapter uses);
PokerKit is used only to evaluate showdowns.

Layout matches ``PokerkitTable``: seat 0 = big blind (= Botzone dealer), seat 1
= small blind. Botzone order: the SB acts first preflop AND on every postflop
street (``action_order="sb_first"``). ``action_order="standard"`` keeps
Botzone betting rules but lets the BB act first postflop.

Rules (all verified on real Botzone logs):
  * raise = EXTRA chips; min raise = ``rules.min_raise_amount`` (conservative)
  * raise must be strictly below remaining chips, else only all-in
  * check/call only if remaining chips > amount to call
  * after any all-in only fold / all-in; all-in vs all-in -> showdown
  * illegal response -> fold

Deal order is PokerKit's (``aligned_decks(...)[1]``): hole cards seat0, seat1,
seat0, seat1, then burn+flop, burn+turn, burn+river. Stacks and blinds are
parameters (default from config); nothing assumes 200bb.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from alphanl.botzone.rules import ALLIN, CALL, FOLD, classify_response, min_raise_amount

from .history import BIG_BLIND_CHIPS, SMALL_BLIND_CHIPS, STARTING_STACK, make_action_event
from .schema import NUM_ACTIONS, ActionEvent, HandSnapshot, normalize_card
from .tables import (
    _ALL_IN,
    _CHECK_CALL,
    _FOLD,
    _HALF,
    _POT,
    _SIZED_DEN,
    _SIZED_NUM,
    _TakenAction,
    _normalize_deck,
    _whole_chips,
    standard_deck,
)

ACTION_ORDERS = ("sb_first", "standard")


def _pk(card: str) -> str:
    c = normalize_card(card)  # suit-then-rank, e.g. "SA"
    return c[1] + c[0].lower()


def _best_hand(hole: Sequence[str], board: Sequence[str]):
    from pokerkit import StandardHighHand

    return StandardHighHand.from_game("".join(_pk(c) for c in hole), "".join(_pk(c) for c in board))


class BotzoneTable:
    def __init__(
        self,
        *,
        seed: int,
        starting_stack: float = STARTING_STACK,
        small_blind: float = SMALL_BLIND_CHIPS,
        big_blind: float = BIG_BLIND_CHIPS,
        stacks: Optional[Sequence[float]] = None,
        action_order: str = "sb_first",
    ) -> None:
        if action_order not in ACTION_ORDERS:
            raise ValueError(f"action_order must be one of {ACTION_ORDERS}")
        self.action_order = action_order
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
            self.starting_stacks = (_whole_chips(stacks[0], "stacks[0]"), _whole_chips(stacks[1], "stacks[1]"))
        for stack in self.starting_stacks:
            if stack <= self.big_blind:
                raise ValueError(f"each stack must exceed the big blind ({self.big_blind}), got {stack}")
        self.starting_stack = float(nominal)
        self.seed = int(seed)
        self._py_random = random.Random(self.seed)
        self.sb_seat = 1
        self._hand: Optional[Dict[str, Any]] = None
        self._records: List[_TakenAction] = []

    # ------------------------------------------------------------------ hand state
    def reset(self, deck: Optional[Sequence[str]] = None) -> None:
        if deck is None:
            cards = list(standard_deck())
            self._py_random.shuffle(cards)
        else:
            cards = list(_normalize_deck(deck))
        holes = ((cards[0], cards[2]), (cards[1], cards[3]))
        board = (cards[5], cards[6], cards[7], cards[9], cards[11])  # burns at 4, 8, 10
        bb, sb = 1 - self.sb_seat, self.sb_seat
        bets = [0, 0]
        bets[sb], bets[bb] = self.small_blind, self.big_blind
        self._hand = {
            "holes": holes, "board": board, "street": 0, "bets": bets, "committed": list(bets),
            "folded": [False, False], "allin": [False, False], "round_bet": self.big_blind,
            "last_raise": self.big_blind, "count": 0, "to_act": sb, "over": False, "showdown": False,
        }
        self._records = []
        self._bz: List[Dict[str, Any]] = []

    def _require_hand(self) -> Dict[str, Any]:
        if self._hand is None:
            raise RuntimeError("call reset() before using the table")
        return self._hand

    def is_over(self) -> bool:
        return bool(self._require_hand()["over"])

    def current_player(self) -> int:
        h = self._require_hand()
        if h["over"]:
            raise RuntimeError("hand is over")
        return int(h["to_act"])

    @property
    def street(self) -> int:
        return int(self._require_hand()["street"])

    def chips_behind(self, seat: int) -> int:
        h = self._require_hand()
        return int(self.starting_stacks[seat] - h["committed"][seat])

    def pot(self) -> int:
        return int(sum(self._require_hand()["committed"]))

    def to_call(self, seat: int) -> int:
        h = self._require_hand()
        return int(max(0, h["round_bet"] - h["bets"][seat]))

    def min_raise(self, seat: Optional[int] = None) -> int:
        """Conservative Botzone minimum raise (extra chips) for the seat to act."""
        h = self._require_hand()
        seat = h["to_act"] if seat is None else seat
        return min_raise_amount(h["round_bet"], h["bets"][seat], h["last_raise"], self.big_blind)

    def classify(self, response, seat: Optional[int] = None) -> str:
        h = self._require_hand()
        seat = h["to_act"] if seat is None else int(seat)
        return classify_response(
            response, round_bet=h["round_bet"], my_bet=h["bets"][seat], chips_behind=self.chips_behind(seat),
            last_raise=h["last_raise"], big_blind=self.big_blind, allin_locked=any(h["allin"]),
        )

    # ------------------------------------------------------------- abstract actions
    def _response_for(self, action_id: int) -> Optional[int]:
        h = self._require_hand()
        seat = h["to_act"]
        if action_id == _FOLD:
            return FOLD if (self.to_call(seat) > 0 or any(h["allin"])) else None
        if action_id == _CHECK_CALL:
            return CALL if self.classify(CALL) in ("check", "call") else None
        if action_id == _ALL_IN:
            return ALLIN if self.chips_behind(seat) > 0 else None
        if action_id in _SIZED_NUM:
            add = (self.pot() * _SIZED_NUM[action_id]) // _SIZED_DEN[action_id]
            return add if add > 0 and self.classify(add) == "raise" else None
        return None

    def legal_action_ids(self) -> Tuple[int, ...]:
        if self.is_over():
            return ()
        return tuple(a for a in range(NUM_ACTIONS) if self._response_for(a) is not None)

    def action_chips(self, action_id: int) -> Optional[int]:
        if self.is_over():
            return None
        r = self._response_for(int(action_id))
        if r is None:
            return None
        seat = self._require_hand()["to_act"]
        return {FOLD: 0, CALL: self.to_call(seat), ALLIN: self.chips_behind(seat)}.get(r, r)

    def response_for(self, action_id: int) -> Optional[int]:
        """Botzone response int for an abstract action (``None`` if masked)."""
        return None if self.is_over() else self._response_for(int(action_id))

    def step(self, action_id: int) -> None:
        legal = self.legal_action_ids()
        action = int(action_id)
        if action not in legal:
            raise ValueError(f"action {action} is not legal ({legal})")
        self._apply_response(self._response_for(action), action, legal)

    def step_response(self, response) -> str:
        """Apply a raw Botzone response (illegal -> fold). Returns the action type."""
        if self.is_over():
            raise RuntimeError("hand is over")
        legal = self.legal_action_ids()
        typ = self.classify(response)
        if typ == "illegal":
            response = FOLD
            typ = "fold"
        r = int(response)
        if r == FOLD:
            aid = _FOLD
        elif r == CALL:
            aid = _CHECK_CALL
        elif r == ALLIN:
            aid = _ALL_IN
        else:
            half = self.pot() // 2
            aid = _HALF if abs(r - half) <= abs(r - self.pot()) else _POT
        self._apply_response(r, aid, legal)
        return typ

    def _apply_response(self, r: int, action_id: int, legal: Tuple[int, ...]) -> None:
        h = self._require_hand()
        seat, other = h["to_act"], 1 - h["to_act"]
        pot_before, bet_before = self.pot(), h["bets"][seat]
        stack_before, to_call = self.chips_behind(seat), self.to_call(seat)
        street = h["street"]
        typ = self.classify(r)
        if typ == "illegal":
            raise ValueError(f"response {r} is illegal")
        added = 0
        if typ == "fold":
            h["folded"][seat] = True
            h["over"] = True
        elif typ == "allin":
            added = stack_before
            h["allin"][seat] = True
            h["bets"][seat] += added
            h["committed"][seat] += added
            h["round_bet"] = max(h["round_bet"], h["bets"][seat])
            if h["allin"][other]:
                h["over"] = h["showdown"] = True
            else:
                h["to_act"] = other
        elif typ in ("check", "call"):
            added = to_call
            h["bets"][seat] += added
            h["committed"][seat] += added
            h["count"] += 1
            if h["count"] >= 2 and h["bets"][0] == h["bets"][1]:
                if h["street"] == 3:
                    h["over"] = h["showdown"] = True
                else:
                    h["street"] += 1
                    h["bets"] = [0, 0]
                    h["round_bet"], h["last_raise"], h["count"] = 0, self.small_blind, 0
                    h["to_act"] = self.sb_seat if self.action_order == "sb_first" else 1 - self.sb_seat
            else:
                h["to_act"] = other
        else:  # raise: r extra chips
            added = int(r)
            h["bets"][seat] += added
            h["committed"][seat] += added
            h["round_bet"] = max(h["round_bet"], h["bets"][seat])
            h["last_raise"] = added
            h["count"] += 1
            h["to_act"] = other
        self._bz.append({"round": street, "player_id": seat, "action": int(r),
                         "action_type": "check" if typ == "check" else typ})
        self._records.append(
            _TakenAction(street=street, actor_seat=seat, action_id=int(action_id), pot_before=pot_before,
                         chips_added=int(added), to_call_before=to_call, stack_before=stack_before,
                         legal_action_ids=tuple(legal))
        )

    def botzone_request(self, seat: Optional[int] = None) -> Dict[str, Any]:
        """The Botzone request JSON the seat to act would receive (player_id = seat,
        dealer_id = BB seat). Cards are table strings, not Botzone ints."""
        h = self._require_hand()
        seat = h["to_act"] if seat is None else int(seat)
        return {"num_players": 2, "dealer_id": 1 - self.sb_seat, "my_id": seat,
                "my_chips": self.chips_behind(seat), "my_cards": list(self.hole_cards(seat)),
                "public_cards": list(self.board_cards()), "history": [dict(x) for x in self._bz]}

    # ---------------------------------------------------------------- observations
    def hole_cards(self, seat: int) -> Tuple[str, ...]:
        return tuple(self._require_hand()["holes"][int(seat)])

    def board_cards(self) -> Tuple[str, ...]:
        h = self._require_hand()
        street = 3 if h["showdown"] else min(h["street"], 3)
        return tuple(h["board"][: (0, 3, 4, 5)[street]])

    def events_for(self, me_seat: int) -> Tuple[ActionEvent, ...]:
        self._require_hand()
        return tuple(
            make_action_event(
                street=rec.street, actor_seat=rec.actor_seat, me_seat=int(me_seat), sb_seat=int(self.sb_seat),
                action_id=rec.action_id, action_type=None, pot_before=rec.pot_before,
                chips_added=rec.chips_added, to_call_before=rec.to_call_before,
                stack_before=rec.stack_before, legal_action_ids=rec.legal_action_ids,
            )
            for rec in self._records
        )

    def snapshot(self, me_seat: int) -> HandSnapshot:
        hole = self.hole_cards(me_seat)
        return HandSnapshot(
            events=self.events_for(me_seat), hole_cards=(hole[0], hole[1]), board_cards=self.board_cards(),
            me_seat=int(me_seat), sb_seat=int(self.sb_seat), street=min(self.street, 3), pot=float(self.pot()),
            stacks=(float(self.chips_behind(0)), float(self.chips_behind(1))),
            to_call=float(self.to_call(int(me_seat))), legal_action_ids=self.legal_action_ids(),
            big_blind=float(self.big_blind), starting_stack=float(self.starting_stack),
        )

    def wrapper_history(self) -> List[list]:
        history: List[list] = [[], [], [], []]
        for rec in self._records:
            history[int(rec.street)].append([int(rec.actor_seat), int(rec.action_id), list(rec.legal_action_ids)])
        return history

    def rlcard_raw_obs(self, me_seat: int) -> Dict[str, Any]:
        seat = int(me_seat)
        if self.current_player() != seat:
            raise RuntimeError("rlcard_raw_obs requires the acting seat")
        snap = self.snapshot(seat)
        h = self._require_hand()
        return {
            "hand": list(snap.hole_cards), "public_cards": list(snap.board_cards),
            "stakes": [self.chips_behind(0), self.chips_behind(1)], "all_chips": list(h["bets"]),
            "pot": float(self.pot()), "stage": int(snap.street), "legal_actions": list(self.legal_action_ids()),
            "current_player": seat,
        }

    # -------------------------------------------------------------------- results
    def payoffs(self) -> Tuple[float, float]:
        h = self._require_hand()
        if not h["over"]:
            raise RuntimeError("payoffs are available after the hand ends")
        c = h["committed"]
        if h["folded"][0] or h["folded"][1]:
            loser = 0 if h["folded"][0] else 1
            won = min(c[loser], c[1 - loser])  # uncalled excess goes back
            net = [0.0, 0.0]
            net[loser], net[1 - loser] = -float(won), float(won)
            return (net[0], net[1])
        matched = min(c)
        r0 = _best_hand(h["holes"][0], h["board"])
        r1 = _best_hand(h["holes"][1], h["board"])
        if r0 == r1:
            return (0.0, 0.0)  # heads-up pots are always even: no odd chip
        return (float(matched), -float(matched)) if r0 > r1 else (-float(matched), float(matched))

    def chips_in_play(self) -> int:
        return int(sum(self.starting_stacks))

    def state_dict(self) -> Dict[str, Any]:
        import copy

        return {
            "version": 1, "kind": "botzone", "hand": copy.deepcopy(self._require_hand()),
            "records": list(self._records), "bz": [dict(x) for x in self._bz], "sb_seat": self.sb_seat, "rng": self._py_random.getstate(),
            "starting_stacks": self.starting_stacks, "blinds": (self.small_blind, self.big_blind),
            "action_order": self.action_order,
        }

    def load_state_dict(self, blob: Dict[str, Any]) -> None:
        import copy

        if blob.get("kind") != "botzone" or int(blob.get("version", 0)) != 1:
            raise ValueError("not a botzone table state")
        if tuple(blob["starting_stacks"]) != tuple(self.starting_stacks):
            raise ValueError("checkpoint stacks do not match this table")
        if tuple(blob["blinds"]) != (self.small_blind, self.big_blind):
            raise ValueError("checkpoint blinds do not match this table")
        self._hand = copy.deepcopy(blob["hand"])
        self._records = list(blob["records"])
        self._bz = [dict(x) for x in blob.get("bz", [])]
        self.sb_seat = int(blob["sb_seat"])
        self.action_order = blob["action_order"]
        self._py_random.setstate(blob["rng"])

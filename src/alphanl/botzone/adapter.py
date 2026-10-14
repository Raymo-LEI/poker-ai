"""Botzone TexasHoldem2p adapter: request JSON -> game state, abstract action -> legal int.

Rules: ``alphanl.botzone.rules`` (verified on 41 real Botzone matches).

Response ints: -1 fold, -2 allin, 0 call/check, >0 raise = EXTRA chips put in
by this action (sample: ``player_bets[id] += action``), not "raise to".

Stack and blinds are parameters (``GameParams``); defaults are Botzone's
20000 / 50 / 100. ``effective_stack_bb`` lets a model trained on shallower
stacks (e.g. 50bb) see a capped view.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .rules import ALLIN, CALL, FOLD, min_raise_amount  # noqa: F401  (single shared rule)
SUITS = "hdsc"  # wiki: 0 红桃(hearts) 1 方块(diamonds) 2 黑桃(spades) 3 草花(clubs)
RANKS = "23456789TJQKA"


def card_rank(card: int) -> int:
    """2..14 (A=14). Wiki/sample: number = card // 4 + 2."""
    return card // 4 + 2


def card_suit(card: int) -> int:
    return card % 4


def card_str(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def card_from_str(text: str) -> int:
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


@dataclass
class GameParams:
    stack: int = 20000
    small_blind: int = 50
    big_blind: int = 100
    num_players: int = 2

    @property
    def stack_bb(self) -> float:
        return self.stack / self.big_blind


def sb_seat(dealer: int, n: int = 2) -> int:
    return (dealer + 1) % n


def bb_seat(dealer: int, n: int = 2) -> int:
    # wiki: BB = dealer's next-next. Heads-up that is the dealer itself.
    return (dealer + 2) % n


@dataclass
class State:
    my_id: int
    dealer_id: int
    street: int  # 0 preflop 1 flop 2 turn 3 river
    my_cards: List[int]
    board: List[int]
    my_chips: int  # chips behind (not yet committed this hand)
    opp_chips: int
    my_bet: int  # committed on this street
    opp_bet: int
    round_bet: int  # max bet on this street
    min_raise: int  # minimum EXTRA chips for a raise (see min_raise_amount)
    pot: int  # all chips committed this hand (both players, all streets)
    allin_seen: bool  # someone went all-in this hand -> only fold/allin legal
    to_call: int
    can_check: bool
    can_call: bool
    can_raise: bool
    max_raise: int  # largest legal raise amount (my_chips - 1); allin is -2
    params: GameParams = field(default_factory=GameParams)
    hand: int = 0
    max_hand: int = 50

    @property
    def legal_responses_summary(self) -> Dict[str, object]:
        return {
            "fold": True, "allin": True, "check": self.can_check, "call": self.can_call,
            "raise": (self.min_raise, self.max_raise) if self.can_raise else None,
        }

    def effective_stack_bb(self, cap_bb: Optional[float] = None) -> float:
        start = self.params.stack / self.params.big_blind
        return start if cap_bb is None else min(start, float(cap_bb))


def replay(req: Dict, params: Optional[GameParams] = None) -> State:
    """Rebuild the state from one Botzone request (full hand history each turn).

    Betting reconstruction follows the sample bot's street logic; the min-raise
    rule is ``alphanl.botzone.rules.min_raise_amount``.
    """
    p = params or GameParams()
    n = int(req.get("num_players", p.num_players))
    me, dealer = int(req["my_id"]), int(req["dealer_id"])
    bets = [0] * n
    bets[sb_seat(dealer, n)] = p.small_blind
    bets[bb_seat(dealer, n)] = p.big_blind
    committed = list(bets)
    folded = [False] * n
    allin = False
    street, round_bet, last_raise, count = 0, p.big_blind, p.big_blind, 0
    for h in req.get("history", []):
        pid, act, typ = int(h["player_id"]), int(h["action"]), h["action_type"]
        if typ == "fold":
            folded[pid] = True
        elif typ == "allin":
            allin = True
            add = p.stack - committed[pid]
            bets[pid] += add
            committed[pid] = p.stack
            round_bet = max(round_bet, bets[pid])
        elif typ in ("call", "check"):
            committed[pid] += round_bet - bets[pid]
            bets[pid] = round_bet
            count += 1
            live = [b for i, b in enumerate(bets) if not folded[i]]
            if not allin and count >= len(live) and all(b == round_bet for b in live):
                street += 1
                round_bet, last_raise, count = 0, p.small_blind, 0
                bets = [0] * n
        elif typ == "raise":
            bets[pid] += act
            committed[pid] += act
            round_bet = max(round_bet, bets[pid])
            last_raise = act
            count += 1
    opp = 1 - me
    my_chips = int(req.get("my_chips", p.stack - committed[me]))
    to_call = max(0, round_bet - bets[me])
    can_call = (not allin) and to_call < my_chips
    round_raise = min_raise_amount(round_bet, bets[me], last_raise, p.big_blind)
    can_raise = (not allin) and round_raise < my_chips and to_call < my_chips
    return State(
        my_id=me, dealer_id=dealer, street=street, my_cards=list(req.get("my_cards", [])),
        board=list(req.get("public_cards", [])), my_chips=my_chips,
        opp_chips=max(0, p.stack - committed[opp]), my_bet=bets[me], opp_bet=bets[opp],
        round_bet=round_bet, min_raise=round_raise, pot=sum(committed), allin_seen=allin,
        to_call=to_call, can_check=can_call and to_call == 0, can_call=can_call,
        can_raise=can_raise, max_raise=my_chips - 1, params=p,
        hand=int(req.get("hand", 0)), max_hand=int(req.get("max_hand", 50)),
    )


def legalize(state: State, response: int) -> int:
    """Clamp any int to a legal response. Never returns an illegal value."""
    r = int(response)
    if r == FOLD:
        return CALL if state.can_check else FOLD  # never fold when checking is free
    if r == ALLIN:
        return ALLIN
    if state.allin_seen:
        return ALLIN if r >= 0 else FOLD  # call/raise vs all-in -> all-in (the only "call")
    if r == CALL or r < -2:
        return CALL if state.can_call else ALLIN  # can't afford call -> all-in
    # raise
    if r >= state.my_chips:
        return ALLIN
    if not state.can_raise:
        return ALLIN  # min raise >= stack: the only aggressive option left
    return max(r, state.min_raise)


ABSTRACT = ("fold", "check_call", "raise_min", "half_pot", "pot", "two_pot", "allin")


def abstract_to_response(state: State, action: str, cap_bb: Optional[float] = None) -> int:
    """Map an abstract action to a legal int.

    Pot-fraction raises: extra = to_call + f * (pot + to_call), i.e. call then
    bet f of the pot after calling. ``cap_bb`` (e.g. 50 for a 50bb model)
    turns any raise that would commit more than cap_bb*BB this hand into allin.
    """
    if action == "fold":
        return legalize(state, FOLD)
    if action == "check_call":
        return legalize(state, CALL)
    if action == "allin":
        return ALLIN
    if action == "raise_min":
        extra = state.min_raise
    else:
        frac = {"half_pot": 0.5, "pot": 1.0, "two_pot": 2.0}[action]
        extra = int(state.to_call + frac * (state.pot + state.to_call))
    if cap_bb is not None:
        cap = int(cap_bb * state.params.big_blind)
        committed_me = state.params.stack - state.my_chips
        if committed_me + extra >= cap:
            return ALLIN
    return legalize(state, max(extra, 1))


def bot_main(policy) -> None:
    """Botzone entry point: read one JSON line, answer {"response": int}."""
    data = json.loads(sys.stdin.readline())
    req = data["requests"][-1]
    if isinstance(req, str):
        req = json.loads(req)
    state = replay(req)
    out = legalize(state, policy(state, req))
    print(json.dumps({"response": int(out)}))

"""Botzone request -> the observation AlphaNLHoldem's c_1048 consumes, and back.

c_1048 (bupticybee/AlphaNLHoldem @ d847dbc) was trained in
``agi/nl_holdem_env.py:NlHoldemEnvWrapper`` over rlcard 1.1.0
``no-limit-holdem``: 100-chip stacks (50bb), blinds 1/2, dealer = BB, SB acts
first on every street, 5 abstract actions (fold, check/call, raise
``int(pot/2)``, raise ``pot``, all-in; pot = all chips in the hand). Botzone
uses the same seat order, so seats map 1:1 (Botzone player_id == rlcard seat,
Botzone dealer_id == rlcard dealer_id).

``RlcardShadow`` is a pure-Python port of rlcard 1.1.0's betting logic
(``games/nolimitholdem/round.py`` + ``game.py:step``); only the observation
encoding (``alphanl.nl_obs.encode_observation``, a port of the wrapper's
``_get_observation``) is reused. No TF, no rlcard, no Ray needed at runtime.

Scaling: one rlcard chip = ``botzone_bb / 2`` Botzone chips (50 on Botzone).
The shadow game is capped at 50bb (100 rlcard chips). Botzone stacks are 200bb,
so Botzone states past the cap are clamped: a Botzone raise whose size exceeds
the shadow stack is read as the shadow's all-in, and when the shadow hand is
finished (both "all-in" at the cap) while Botzone still asks for a decision,
the view is flagged ``beyond_cap`` (see ``action_to_response``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alphanl.nl_obs import encode_observation

from .adapter import GameParams, State, legalize, replay
from .rules import ALLIN, CALL, FOLD

FOLD_A, CALL_A, HALF_A, POT_A, ALLIN_A = 0, 1, 2, 3, 4
RL_STACK, RL_SB, RL_BB = 100, 1, 2
BZ_SUITS = "HDSC"  # Botzone card % 4: hearts, diamonds, spades, clubs
RANKS = "23456789TJQKA"


def rl_card(card: int) -> str:
    """Botzone int -> rlcard suit-then-rank string (``SA``)."""
    return BZ_SUITS[int(card) % 4] + RANKS[int(card) // 4]


class RlcardShadow:
    """rlcard 1.1.0 heads-up no-limit betting, chips only (no cards)."""

    ALIVE, FOLDED, ALLIN = 0, 1, 2

    def __init__(self, dealer_id: int, stack: int = RL_STACK, sb: int = RL_SB, bb: int = RL_BB):
        self.dealer_id = int(dealer_id)
        self.in_chips = [0, 0]
        self.remained = [int(stack), int(stack)]
        self.status = [self.ALIVE, self.ALIVE]
        s, b = (self.dealer_id + 1) % 2, (self.dealer_id + 2) % 2
        self._bet(b, bb)
        self._bet(s, sb)
        self.game_pointer = (b + 1) % 2
        self.raised = list(self.in_chips)
        self.not_raise_num = 0
        self.not_playing_num = 0
        self.round_counter = 0
        self.stage = 0
        self.history: List[List[list]] = [[], [], [], []]

    def _bet(self, p: int, chips: int) -> int:
        q = min(int(chips), self.remained[p])
        self.in_chips[p] += q
        self.remained[p] -= q
        return q

    @property
    def pot(self) -> int:
        return int(sum(self.in_chips))

    def is_over(self) -> bool:
        alive = sum(1 for st in self.status if st in (self.ALIVE, self.ALLIN))
        return alive == 1 or self.round_counter >= 4

    def legal_actions(self) -> List[int]:
        """``Round.get_nolimit_legal_actions`` (round.py:126-162)."""
        p = self.game_pointer
        full = [FOLD_A, CALL_A, HALF_A, POT_A, ALLIN_A]
        diff = max(self.raised) - self.raised[p]
        rem = self.remained[p]
        if diff > 0 and diff >= rem:
            for a in (HALF_A, POT_A, ALLIN_A):
                full.remove(a)
        else:
            if self.pot > rem:
                full.remove(POT_A)
            if int(self.pot / 2) > rem:
                full.remove(HALF_A)
            if HALF_A in full and int(self.pot / 2) + self.raised[p] <= max(self.raised):
                full.remove(HALF_A)
        return full

    def chips_for(self, action: int) -> int:
        p = self.game_pointer
        if action == CALL_A:
            return min(max(self.raised) - self.raised[p], self.remained[p])
        if action == HALF_A:
            return int(self.pot / 2)
        if action == POT_A:
            return self.pot
        if action == ALLIN_A:
            return self.remained[p]
        return 0

    def step(self, action: int) -> None:
        """``Game.step`` + ``Round.proceed_round`` (game.py:124-188, round.py:66-124)."""
        action = int(action)
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"action {action} not legal {legal}")
        p = self.game_pointer
        self.history[self.stage].append([p, action, list(legal)])
        pot = self.pot
        if action == CALL_A:
            diff = max(self.raised) - self.raised[p]
            self.raised[p] = max(self.raised)
            self._bet(p, diff)
            self.not_raise_num += 1
        elif action == ALLIN_A:
            q = self.remained[p]
            self.raised[p] += q
            self._bet(p, q)
            self.not_raise_num = 1
        elif action == POT_A:
            self.raised[p] += pot
            self._bet(p, pot)
            self.not_raise_num = 1
        elif action == HALF_A:
            q = int(pot / 2)
            self.raised[p] += q
            self._bet(p, q)
            self.not_raise_num = 1
        elif action == FOLD_A:
            self.status[p] = self.FOLDED
        if self.remained[p] == 0 and self.status[p] != self.FOLDED:
            self.status[p] = self.ALLIN
        self.game_pointer = (p + 1) % 2
        if self.status[p] == self.ALLIN:
            self.not_playing_num += 1
            self.not_raise_num -= 1
        if self.status[p] == self.FOLDED:
            self.not_playing_num += 1
        while self.status[self.game_pointer] == self.FOLDED:
            self.game_pointer = (self.game_pointer + 1) % 2
        bypass = [1 if st in (self.FOLDED, self.ALLIN) else 0 for st in self.status]
        if 2 - sum(bypass) == 1:
            last = bypass.index(0)
            if self.raised[last] >= max(self.raised):
                bypass[last] = 1
        if self.not_raise_num + self.not_playing_num >= 2:  # round.is_over
            self.game_pointer = (self.dealer_id + 1) % 2
            if sum(bypass) < 2:
                while bypass[self.game_pointer]:
                    self.game_pointer = (self.game_pointer + 1) % 2
            if self.round_counter == 0:
                self.stage = 1
                if sum(bypass) == 2:
                    self.round_counter += 1
            if self.round_counter == 1:
                self.stage = 2
                if sum(bypass) == 2:
                    self.round_counter += 1
            if self.round_counter == 2:
                self.stage = 3
                if sum(bypass) == 2:
                    self.round_counter += 1
            self.round_counter += 1
            self.raised = [0, 0]
            self.not_raise_num = 0


@dataclass
class C1048View:
    obs: Optional[Dict[str, Any]]  # encode_observation output (card_info, action_info, extra_info, legal_mask)
    raw_obs: Optional[Dict[str, Any]]
    player_id: int
    shadow: RlcardShadow
    state: State  # adapter view of the real Botzone state
    scale: float  # Botzone chips per rlcard chip
    beyond_cap: bool = False
    translated: List[tuple] = field(default_factory=list)  # (botzone action, shadow action id)
    notes: List[str] = field(default_factory=list)


def _nearest_raise(shadow: RlcardShadow, extra_bz: int, scale: float) -> int:
    legal = shadow.legal_actions()
    sized = [a for a in (HALF_A, POT_A, ALLIN_A) if a in legal]
    if not sized:
        return CALL_A
    want = extra_bz / scale
    for a in sized:  # exact size first (pot raise that happens to use the whole stack stays "pot")
        if abs(shadow.chips_for(a) * scale - extra_bz) < 1e-9:
            return a
    if want >= shadow.remained[shadow.game_pointer] and ALLIN_A in legal:
        return ALLIN_A
    best, best_d = sized[0], float("inf")
    for a in sized:
        c = max(shadow.chips_for(a), 1)
        d = abs(math.log(max(want, 1e-9) / c))
        if d < best_d - 1e-12:
            best, best_d = a, d
    return best


def convert(request: Dict[str, Any], params: Optional[GameParams] = None) -> C1048View:
    """Build c_1048's observation for the Botzone request's acting player."""
    p = params or GameParams()
    scale = p.big_blind / RL_BB
    state = replay(request, p)
    me = int(request["my_id"])
    shadow = RlcardShadow(int(request["dealer_id"]))
    view = C1048View(obs=None, raw_obs=None, player_id=me, shadow=shadow, state=state, scale=scale)
    for h in request.get("history", []):
        pid, act, typ = int(h["player_id"]), int(h["action"]), h["action_type"]
        if shadow.is_over() or shadow.game_pointer != pid:
            view.beyond_cap = True
            view.notes.append(f"shadow finished/desynced before {typ} by {pid}")
            break
        legal = shadow.legal_actions()
        if typ == "fold":
            a = FOLD_A
        elif typ in ("check", "call"):
            a = CALL_A
        elif typ == "allin":
            a = ALLIN_A if ALLIN_A in legal else CALL_A
        else:
            a = _nearest_raise(shadow, act, scale)
        view.translated.append((typ, act, a))
        shadow.step(a)
    if not view.beyond_cap and (shadow.is_over() or shadow.game_pointer != me or shadow.stage != state.street):
        view.beyond_cap = True
        view.notes.append("shadow hand over or out of step with Botzone (past the 50bb cap)")
    public = [rl_card(c) for c in request.get("public_cards", [])]
    hand = [rl_card(c) for c in request.get("my_cards", [])]
    legal = [CALL_A] if view.beyond_cap else shadow.legal_actions()
    view.raw_obs = {
        "hand": hand,
        "public_cards": public,
        "legal_actions": legal,
        "stakes": list(shadow.remained),
        "current_player": me,
        "stage": shadow.stage,
        "pot": shadow.pot,
    }
    view.obs = encode_observation(view.raw_obs, shadow.history, me)
    return view


def action_to_response(view: C1048View, action: int, cap_bb: float = RL_STACK / RL_BB) -> int:
    """Map c_1048's abstract action to a legal Botzone int (via ``legalize``).

    fold -> -1 (legalize turns it into a check when checking is free);
    check/call -> 0 (legalize: call of an all-in or a call costing all our
    chips becomes -2); raise half-pot / pot -> extra = scale * rlcard chips
    for that action, -2 if it would commit more than ``cap_bb``; all-in -> -2.
    ``beyond_cap`` views (the 50bb shadow hand is already all-in/finished) ->
    check if free, otherwise -2 (beyond the cap counts as all-in).
    """
    s = view.state
    a = int(action)
    if view.beyond_cap:
        return legalize(s, CALL if s.to_call == 0 else ALLIN)
    if a == FOLD_A:
        return legalize(s, FOLD)
    if a == CALL_A:
        return legalize(s, CALL)
    if a == ALLIN_A:
        return ALLIN
    if a in (HALF_A, POT_A):
        extra = int(round(view.shadow.chips_for(a) * view.scale))
        committed = s.params.stack - s.my_chips
        if committed + extra > cap_bb * s.params.big_blind:
            return ALLIN
        return legalize(s, max(extra, 1))
    raise ValueError(f"action must be 0..4, got {action}")

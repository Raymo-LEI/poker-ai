"""Branchable training kernel for the official Botzone TexasHoldem2p referee.

Transitions are ``Holdem.player_action`` on the unmodified judge in
``tests/fixtures/botzone_judge.py``. This module does not reimplement
min-raises, street closure, showdown, or the 50-hand score. ``legal.apply``
is intentionally not the transition: a legal under-call that does not beat
the street maximum ends the street there, and the judge leaves the street
open until the live bets match.

Deals follow the judge, not ``BotzoneTable``. The deck list is popped from
the end: seat 0 receives two cards, seat 1 receives two cards, then the
flop, turn, and river with no burn. The dealer posts the big blind. Stacks
reset to 20000 every hand. Blinds are 50/100.

``step`` takes a referee chip response: ``-1`` fold, ``0`` check/call,
``-2`` all-in, or a positive extra (chips added now, not raise-to). Integral
floats such as ``150.0`` are parsed the way the judge parses them. Legacy
five-action ids are not that encoding; :class:`LegacyFiveActionBridge`
turns an id into the chip response ``action_responses`` would send.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .actions import (
    ACTION_NAMES,
    NUM_ACTIONS,
    action_responses,
    legal_action_ids,
    response_for_action,
)
from .legal import (
    BIG_BLIND,
    SMALL_BLIND,
    STARTING_STACK,
    TOTAL_CHIPS,
    HandState,
)

_REPO = Path(__file__).resolve().parents[3]
_EVAL = _REPO / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from botzone_referee_adapter import JUDGE  # noqa: E402

POLICY_FEATURE_NAMES: Tuple[str, ...] = (
    "street",
    "pot",
    "own_stack",
    "opponent_stack",
    "to_call",
    "min_positive_extra",
    "round_raise",
    "own_round_bet",
    "opponent_round_bet",
    "round_bet",
    "pot_fraction",
    "own_stack_fraction",
    "to_call_fraction_of_pot",
    "is_dealer",
    "hand_index",
    "hand_progress",
)

_MISSING = -1.0


@dataclass(frozen=True)
class KernelStep:
    """Public result of one accepted chip response. No hole cards."""

    response: int
    hand_over: bool
    match_over: bool
    street: int
    actor: int
    pot: int
    stacks: Tuple[int, int]
    public_cards: Tuple[int, ...]
    winners: Tuple[int, ...]
    settled_chips: Optional[Tuple[float, float]]
    payoffs: Optional[Tuple[float, float]]
    scores: Tuple[float, float]


def _require_seat(seat: int) -> int:
    if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return seat


def _validate_deck(deck: Sequence[int]) -> List[int]:
    cards = list(deck)
    if len(cards) != 52:
        raise ValueError(f"deck must contain 52 cards, got {len(cards)}")
    out: List[int] = []
    for card in cards:
        if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card <= 51:
            raise ValueError(f"card id must be an int in 0..51, got {card!r}")
        out.append(card)
    if len(set(out)) != 52:
        raise ValueError("deck must be 52 distinct card ids")
    return out


def _new_game(dealer: int):
    """A judge hand before the deal, without touching the global RNG."""

    game = JUDGE.Holdem.__new__(JUDGE.Holdem)
    game.num_players = 2
    game.player_chips = [STARTING_STACK, STARTING_STACK]
    game.dealer_idx = int(dealer)
    game.small_blind = SMALL_BLIND
    game.big_blind = BIG_BLIND
    game.player_cards = [[], []]
    game.public_cards = []
    game.pot = 0
    game.round = JUDGE.Holdem.PRE_FLOP
    game.round_idx = int(dealer)
    game.round_bet = 0
    game.round_raise = 0
    game.round_action_left = 4
    game.round_player_bet = [0, 0]
    game.history = []
    game.deck = []
    return game


def _clone_game(game):
    """Deep copy. Card objects and chip lists are not shared with ``game``."""

    return copy.deepcopy(game)


def _card_ids(cards) -> Tuple[int, ...]:
    return tuple(int(card.to_int()) for card in cards)


class BranchableKernel:
    """One Botzone match whose betting, cards, and score are the judge's.

    ``clone`` / ``branch`` copy the hand, the remaining deck, and the match
    score so later steps on one copy leave the other alone. ``set_deal``
    fixes the deck for a hand that has not been acted on yet. ``step``
    applies one chip response. ``policy_view`` is what a seat is allowed to
    see: that seat's hole cards and the cards already dealt to the board.
    """

    def __init__(self) -> None:
        raise RuntimeError("use start_hand or start_match")

    @classmethod
    def start_hand(cls, deck: Sequence[int], *, dealer: int = 0) -> "BranchableKernel":
        """One hand. ``dealer`` posts the big blind. ``deck`` is judge order."""

        return cls.start_match([deck], initial_dealer=dealer, max_hands=1)

    @classmethod
    def start_match(
        cls,
        decks: Sequence[Sequence[int]],
        *,
        initial_dealer: int = 0,
        max_hands: Optional[int] = None,
    ) -> "BranchableKernel":
        """A match of fixed decks. Hand ``i`` uses dealer ``(initial + i) % 2``.

        The default length is the number of decks. Botzone matches are 50
        hands; pass 50 decks to get that score.
        """

        stored = [_validate_deck(deck) for deck in decks]
        if max_hands is None:
            max_hands = len(stored)
        if isinstance(max_hands, bool) or not isinstance(max_hands, int) or max_hands < 1:
            raise ValueError(f"max_hands must be a positive int, got {max_hands!r}")
        if len(stored) < max_hands:
            raise ValueError(
                f"need at least {max_hands} decks, got {len(stored)}"
            )
        initial_dealer = _require_seat(initial_dealer)
        kernel = cls.__new__(cls)
        kernel._decks = stored
        kernel._dealer_override = {}
        kernel._initial_dealer = initial_dealer
        kernel._hand_index = 0
        kernel._max_hands = int(max_hands)
        kernel._total_win_chips = [0.0, 0.0]
        kernel._total_win_games = [0, 0]
        kernel._hand_over = False
        kernel._match_over = False
        kernel._winners = ()
        kernel._settled = None
        kernel._actions_this_hand = 0
        kernel._start_current_hand()
        return kernel

    def clone(self) -> "BranchableKernel":
        """Deep copy. Later mutation of either kernel is invisible to the other."""

        other = BranchableKernel.__new__(BranchableKernel)
        other._game = _clone_game(self._game)
        other._decks = [list(deck) for deck in self._decks]
        other._dealer_override = dict(self._dealer_override)
        other._initial_dealer = self._initial_dealer
        other._hand_index = self._hand_index
        other._max_hands = self._max_hands
        other._total_win_chips = [float(chips) for chips in self._total_win_chips]
        other._total_win_games = [int(count) for count in self._total_win_games]
        other._hand_over = self._hand_over
        other._match_over = self._match_over
        other._winners = tuple(self._winners)
        other._settled = None if self._settled is None else (float(self._settled[0]), float(self._settled[1]))
        other._actions_this_hand = self._actions_this_hand
        return other

    def branch(self) -> "BranchableKernel":
        """Alias of :meth:`clone`."""

        return self.clone()

    def set_deal(
        self,
        deck: Sequence[int],
        *,
        dealer: Optional[int] = None,
        hand_index: Optional[int] = None,
    ) -> None:
        """Fix one hand's deck before that hand is acted on.

        The current hand, if it has not been stepped, is redealt from the
        blinds. A later hand in the schedule can be replaced before it
        starts. A hand that already has a step, or that has been settled,
        is left unchanged and this raises.
        """

        index = self._hand_index if hand_index is None else hand_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"hand_index must be an int, got {index!r}")
        if not 0 <= index < self._max_hands:
            raise ValueError(f"hand_index must be in 0..{self._max_hands - 1}, got {index}")
        started = index < self._hand_index or (
            index == self._hand_index and (self._actions_this_hand or self._hand_over)
        )
        if started:
            raise ValueError("set_deal cannot rewrite a hand that has already started")
        self._decks[index] = _validate_deck(deck)
        if dealer is not None:
            self._dealer_override[index] = _require_seat(dealer)
        if index == self._hand_index:
            self._start_current_hand()

    def set_match_deals(
        self,
        decks: Sequence[Sequence[int]],
        *,
        initial_dealer: Optional[int] = None,
    ) -> None:
        """Replace the whole deck schedule before the first action."""

        if self._hand_index != 0 or self._actions_this_hand or self._hand_over:
            raise ValueError("set_match_deals must run before the match starts")
        stored = [_validate_deck(deck) for deck in decks]
        if not stored:
            raise ValueError("match needs at least one deck")
        self._decks = stored
        self._max_hands = len(stored)
        self._dealer_override = {}
        if initial_dealer is not None:
            self._initial_dealer = _require_seat(initial_dealer)
        self._start_current_hand()

    def step(self, response) -> KernelStep:
        """Apply one chip response. Illegal responses do not change state.

        ``response`` is ``-1``, ``0``, ``-2``, a positive extra, or an
        integral float the judge would parse as one of those. A legacy
        action id is not accepted here unless that integer is also the
        chip response.
        """

        if self._hand_over:
            raise ValueError("hand is over")
        parsed = JUDGE.parse_bot_response(response)
        if parsed is None or not self.is_legal(parsed):
            raise ValueError(f"illegal response {response!r}")
        result = self._game.player_action(parsed)
        self._actions_this_hand += 1
        if result:
            self._finish_hand(result)
        return self._step_result(int(parsed))

    def next_hand(self) -> None:
        """Start the next scheduled hand after the current one has settled."""

        if not self._hand_over:
            raise ValueError("hand is not over")
        if self._match_over or self._hand_index + 1 >= self._max_hands:
            self._match_over = True
            raise ValueError("match is over")
        self._hand_index += 1
        self._start_current_hand()

    def is_legal(self, response) -> bool:
        """Whether the live judge accepts ``response`` on a throwaway copy.

        The copy is discarded, so a probe does not change this kernel.
        Parsing matches ``parse_bot_response``: integral floats are the
        integer, and ``True`` / ``150.5`` / strings are not bets.
        """

        if self._hand_over:
            return False
        parsed = JUDGE.parse_bot_response(response)
        if parsed is None:
            return False
        probe = _clone_game(self._game)
        try:
            probe.player_action(parsed)
        except ValueError:
            return False
        return True

    def min_positive_extra(self) -> Optional[int]:
        """Judge minimum positive extra for the actor, or ``None``.

        ``2 * round_raise - own_round_bet``, at least 1, and strictly below
        the remaining stack. ``None`` when a positive bet cannot exist
        (all-in lock, short stack, or the hand is over).
        """

        return self._min_positive_for(int(self._game.round_idx))

    def policy_view(self, seat: Optional[int] = None) -> dict:
        """Observation for ``seat`` (default: the actor).

        Contains that seat's hole cards and the community cards already
        dealt. It does not contain the other seat's hole cards or any
        card still in the deck. Chip fields stay continuous: pot, stacks,
        and the real extra amounts, not a snapped size bucket.
        """

        game = self._game
        actor = int(game.round_idx)
        who = actor if seat is None else _require_seat(seat)
        other = 1 - who
        pot = int(game.pot)
        own_stack = int(game.player_chips[who])
        opp_stack = int(game.player_chips[other])
        own_bet = int(game.round_player_bet[who])
        opp_bet = int(game.round_player_bet[other])
        round_bet = int(game.round_bet)
        round_raise = int(game.round_raise)
        to_call = self._to_call(who)
        minimum = self._min_positive_for(who)
        acting = (not self._hand_over) and who == actor
        public = list(_card_ids(game.public_cards))
        holes = list(_card_ids(game.player_cards[who]))
        pot_fraction = pot / float(TOTAL_CHIPS)
        own_fraction = own_stack / float(STARTING_STACK)
        if to_call is None or pot <= 0:
            call_fraction = 0.0
        else:
            call_fraction = to_call / float(pot)
        progress = self._hand_index / float(self._max_hands)
        features = {
            "street": int(game.round),
            "pot": pot,
            "own_stack": own_stack,
            "opponent_stack": opp_stack,
            "to_call": to_call,
            "min_positive_extra": minimum,
            "round_raise": round_raise,
            "own_round_bet": own_bet,
            "opponent_round_bet": opp_bet,
            "round_bet": round_bet,
            "pot_fraction": pot_fraction,
            "own_stack_fraction": own_fraction,
            "to_call_fraction_of_pot": call_fraction,
            "is_dealer": 1 if who == int(game.dealer_idx) else 0,
            "hand_index": int(self._hand_index),
            "hand_progress": progress,
        }
        vector = [
            float(features["street"]),
            float(pot),
            float(own_stack),
            float(opp_stack),
            _MISSING if to_call is None else float(to_call),
            _MISSING if minimum is None else float(minimum),
            float(round_raise),
            float(own_bet),
            float(opp_bet),
            float(round_bet),
            float(pot_fraction),
            float(own_fraction),
            float(call_fraction),
            float(features["is_dealer"]),
            float(self._hand_index),
            float(progress),
        ]
        return {
            "seat": who,
            "hole_cards": holes,
            "public_cards": public,
            "action_events": [dict(item) for item in game.history],
            "features": features,
            "feature_names": list(POLICY_FEATURE_NAMES),
            "feature_vector": vector,
            "legal": {
                "fold": bool(acting and self.is_legal(-1)),
                "check_call": bool(acting and self.is_legal(0)),
                "all_in": bool(acting and self.is_legal(-2)),
                "min_positive_extra": minimum if acting else None,
                "max_positive_extra": (own_stack - 1) if acting and minimum is not None else None,
            },
            "hand_over": bool(self._hand_over),
            "match_over": bool(self._match_over),
            "winners": list(self._winners),
            "payoffs": None if self.payoffs is None else [self.payoffs[0], self.payoffs[1]],
            "scores": [self.scores[0], self.scores[1]],
            "dealer": int(game.dealer_idx),
            "actor": actor,
            "street": int(game.round),
        }

    def as_hand_state(self) -> HandState:
        """Betting snapshot for the legacy five-action mask.

        All-in seats store the judge sentinel rather than the chips they
        put in. ``legal_response`` only needs the all-in flag in that case.
        Do not pass this object to ``legal.apply``: street closure there
        is not the judge's.
        """

        game = self._game
        commits = []
        all_in = []
        folded = []
        for seat in (0, 1):
            bet = int(game.round_player_bet[seat])
            folded.append(bet == -1)
            all_in.append(bet == -2)
            commits.append(0 if bet < 0 else bet)
        return HandState(
            dealer=int(game.dealer_idx),
            street=int(game.round),
            stacks=(int(game.player_chips[0]), int(game.player_chips[1])),
            street_commit=(commits[0], commits[1]),
            pot=int(game.pot),
            bound_b=2 * int(game.round_raise),
            actor=int(game.round_idx),
            all_in=(all_in[0], all_in[1]),
            folded=(folded[0], folded[1]),
            needs_action=(False, False) if self._hand_over else (True, True),
            hand_over=bool(self._hand_over),
        )

    def referee_audit_state(self) -> dict:
        """Full judge fields, including hidden cards. Not a policy observation."""

        game = self._game
        return {
            "chips": (int(game.player_chips[0]), int(game.player_chips[1])),
            "pot": int(game.pot),
            "dealer": int(game.dealer_idx),
            "street": int(game.round),
            "actor": int(game.round_idx),
            "round_bet": int(game.round_bet),
            "round_raise": int(game.round_raise),
            "round_player_bet": (
                int(game.round_player_bet[0]),
                int(game.round_player_bet[1]),
            ),
            "round_action_left": int(game.round_action_left),
            "public_cards": _card_ids(game.public_cards),
            "holes": (
                _card_ids(game.player_cards[0]),
                _card_ids(game.player_cards[1]),
            ),
            "deck": _card_ids(game.deck),
            "history": [dict(item) for item in game.history],
            "hand_over": bool(self._hand_over),
            "winners": tuple(self._winners),
            "settled_chips": self._settled,
            "payoffs": self.payoffs,
            "scores": self.scores,
            "total_win_chips": (float(self._total_win_chips[0]), float(self._total_win_chips[1])),
            "total_win_games": (int(self._total_win_games[0]), int(self._total_win_games[1])),
            "hand_index": int(self._hand_index),
            "match_over": bool(self._match_over),
            "actions_this_hand": int(self._actions_this_hand),
        }

    def conserved(self) -> bool:
        """Stacks plus pot stay 40000, and a settled hand still sums to 40000."""

        game = self._game
        if int(game.player_chips[0]) + int(game.player_chips[1]) + int(game.pot) != TOTAL_CHIPS:
            return False
        if self._settled is None:
            return True
        return self._settled[0] + self._settled[1] == float(TOTAL_CHIPS)

    @property
    def hand_over(self) -> bool:
        return self._hand_over

    @property
    def match_over(self) -> bool:
        return self._match_over

    @property
    def street(self) -> int:
        return int(self._game.round)

    @property
    def actor(self) -> int:
        return int(self._game.round_idx)

    @property
    def dealer(self) -> int:
        return int(self._game.dealer_idx)

    @property
    def pot(self) -> int:
        return int(self._game.pot)

    @property
    def stacks(self) -> Tuple[int, int]:
        return (int(self._game.player_chips[0]), int(self._game.player_chips[1]))

    @property
    def public_cards(self) -> Tuple[int, ...]:
        return _card_ids(self._game.public_cards)

    @property
    def winners(self) -> Tuple[int, ...]:
        return tuple(self._winners)

    @property
    def settled_chips(self) -> Optional[Tuple[float, float]]:
        return self._settled

    @property
    def payoffs(self) -> Optional[Tuple[float, float]]:
        if self._settled is None:
            return None
        return (
            float(self._settled[0] - STARTING_STACK),
            float(self._settled[1] - STARTING_STACK),
        )

    @property
    def scores(self) -> Tuple[float, float]:
        blind = float(self._game.big_blind)
        return (
            float(self._total_win_chips[0]) / blind,
            float(self._total_win_chips[1]) / blind,
        )

    @property
    def hand_index(self) -> int:
        return int(self._hand_index)

    @property
    def max_hands(self) -> int:
        return int(self._max_hands)

    @property
    def actions_this_hand(self) -> int:
        return int(self._actions_this_hand)

    def __repr__(self) -> str:
        return (
            f"BranchableKernel(hand={self._hand_index}, street={self.street}, "
            f"actor={self.actor}, over={self._hand_over})"
        )

    def _start_current_hand(self) -> None:
        dealer = self._dealer_for(self._hand_index)
        game = _new_game(dealer)
        game.set_deck_array(list(self._decks[self._hand_index]))
        game.deal_cards_and_blind()
        self._game = game
        self._hand_over = False
        self._winners = ()
        self._settled = None
        self._actions_this_hand = 0

    def _dealer_for(self, index: int) -> int:
        if index in self._dealer_override:
            return int(self._dealer_override[index])
        return (int(self._initial_dealer) + int(index)) % 2

    def _finish_hand(self, winners) -> None:
        finals = self._game.get_player_final_chips(winners)
        settled = (float(finals[0]), float(finals[1]))
        mean = sum(settled) / float(len(settled))
        for seat, chips in enumerate(settled):
            self._total_win_chips[seat] += chips - mean
            if chips > mean:
                self._total_win_games[seat] += 1
        self._settled = settled
        self._winners = tuple(int(seat) for seat in winners)
        self._hand_over = True
        if self._hand_index + 1 >= self._max_hands:
            self._match_over = True

    def _step_result(self, response: int) -> KernelStep:
        payoffs = self.payoffs
        return KernelStep(
            response=int(response),
            hand_over=bool(self._hand_over),
            match_over=bool(self._match_over),
            street=self.street,
            actor=self.actor,
            pot=self.pot,
            stacks=self.stacks,
            public_cards=self.public_cards,
            winners=self.winners,
            settled_chips=self._settled,
            payoffs=payoffs,
            scores=self.scores,
        )

    def _to_call(self, seat: int) -> Optional[int]:
        game = self._game
        own = int(game.round_player_bet[seat])
        if int(game.round_bet) < 0 or own < 0:
            return None
        return int(game.round_bet) - own

    def _min_positive_for(self, seat: int) -> Optional[int]:
        game = self._game
        if self._hand_over or seat != int(game.round_idx):
            return None
        if int(game.round_bet) < 0:
            return None
        own = int(game.round_player_bet[seat])
        if own < 0:
            return None
        chips = int(game.player_chips[seat])
        if chips <= 1:
            return None
        extra = 2 * int(game.round_raise) - own
        if extra < 1:
            extra = 1
        if extra >= chips:
            return None
        return int(extra)


class LegacyFiveActionBridge:
    """Play a frozen five-action policy by the chip response it maps to.

    Policy id 2 is the minimum raise. On the opening small blind that chip
    response is 150, not 2. Id 4 is all-in, which is the sentinel ``-2``,
    not a 4-chip bet. ``step`` forwards the mapped integer and never the
    raw id.
    """

    def __init__(self, kernel: BranchableKernel) -> None:
        self.kernel = kernel

    def clone(self) -> "LegacyFiveActionBridge":
        return LegacyFiveActionBridge(self.kernel.clone())

    def responses(self) -> Tuple[Optional[int], ...]:
        if self.kernel.hand_over:
            return tuple(None for _ in range(NUM_ACTIONS))
        return action_responses(self.kernel.as_hand_state())

    def legal_ids(self) -> Tuple[int, ...]:
        if self.kernel.hand_over:
            return ()
        return legal_action_ids(self.kernel.as_hand_state())

    def response_for(self, action_id: int) -> Optional[int]:
        if self.kernel.hand_over:
            return None
        return response_for_action(self.kernel.as_hand_state(), action_id)

    def step(self, action_id: int) -> KernelStep:
        response = self.response_for(action_id)
        if response is None:
            raise ValueError(
                f"legacy action {action_id} is masked; refusing to send the raw id"
            )
        return self.kernel.step(int(response))

    def observation(self) -> dict:
        """Policy view plus the five chip responses. Still hides hidden cards."""

        view = self.kernel.policy_view()
        mapped = self.responses()
        view["legacy_action_ids"] = {
            "names": list(ACTION_NAMES),
            "legal_ids": list(self.legal_ids()),
            "chip_responses": [None if response is None else int(response) for response in mapped],
            "note": "chip_responses are referee extras or sentinels, not action ids",
        }
        return view

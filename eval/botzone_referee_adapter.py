"""Drive the official Botzone TexasHoldem2p judge.

``referee_is_legal`` builds a judge ``Holdem`` and calls ``player_action``.
It does not reimplement the raise rule. The spec-based checker below is a
secondary cross-check labelled as such: it is not the referee.

The judge file is imported from ``tests/fixtures/botzone_judge.py`` and is
not modified.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_JUDGE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "botzone_judge.py"
_SPEC_LABEL = "spec-based reference, NOT the real referee"


def load_judge_module():
    """Import the unmodified judge file under a private module name."""

    name = "botzone_judge_fixture"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(_JUDGE_PATH):
        return cached
    spec = importlib.util.spec_from_file_location(name, _JUDGE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load judge from {_JUDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JUDGE = load_judge_module()


@dataclass(frozen=True)
class RefereeState:
    """Fields ``Holdem.player_action`` reads, copied off a live judge.

    ``round_player_bet`` uses the judge's sentinels: ``-1`` fold, ``-2``
    all-in, otherwise the chips committed on this street.
    ``round_bet`` is ``-2`` after an all-in, otherwise the street maximum.
    """

    chips: Tuple[int, int]
    dealer: int
    street: int
    actor: int
    round_bet: int
    round_raise: int
    round_player_bet: Tuple[int, int]
    pot: int

    @property
    def own_round_bet(self) -> int:
        return int(self.round_player_bet[self.actor])

    @property
    def actor_chips(self) -> int:
        return int(self.chips[self.actor])


def state_from_holdem(game) -> RefereeState:
    """Snapshot a live judge without calling into its betting rules."""

    return RefereeState(
        chips=(int(game.player_chips[0]), int(game.player_chips[1])),
        dealer=int(game.dealer_idx),
        street=int(game.round),
        actor=int(game.round_idx),
        round_bet=int(game.round_bet),
        round_raise=int(game.round_raise),
        round_player_bet=(int(game.round_player_bet[0]), int(game.round_player_bet[1])),
        pot=int(game.pot),
    )


def new_dealt_holdem(dealer: int, deck: Optional[Tuple[int, ...]] = None):
    """A judge hand after blinds, history cleared, small blind to act."""

    game = JUDGE.Holdem([20000, 20000], dealer_idx=int(dealer))
    cards = list(range(52)) if deck is None else list(deck)
    game.set_deck_array(cards)
    game.deal_cards_and_blind()
    return game


def _probe_game(state: RefereeState):
    """Fresh judge object whose betting fields are ``state``.

    ``round_action_left`` stays above 1 so a probe cannot fall through into
    the next street. ``player_action`` checks the bet before that counter
    matters, so this does not change accept versus ``ValueError``.
    """

    game = JUDGE.Holdem.__new__(JUDGE.Holdem)
    game.num_players = 2
    game.player_chips = [int(state.chips[0]), int(state.chips[1])]
    game.dealer_idx = int(state.dealer)
    game.small_blind = 50
    game.big_blind = 100
    game.player_cards = [[], []]
    game.public_cards = []
    game.pot = int(state.pot)
    game.round = int(state.street)
    game.round_idx = int(state.actor)
    game.round_bet = int(state.round_bet)
    game.round_raise = int(state.round_raise)
    # One action brings this to 0. If every seat is then all-in, the judge
    # ends the round instead of calling ``_next_player``. ``_next_player``
    # loops forever when every ``round_player_bet`` is negative, which is
    # exactly the both-all-in probe. Legality is decided before this counter.
    game.round_action_left = 1
    game.round_player_bet = [int(state.round_player_bet[0]), int(state.round_player_bet[1])]
    game.history = []
    game.deck = [JUDGE.Card.from_int(i) for i in range(10)]
    return game


def referee_is_legal(state: RefereeState, response) -> bool:
    """True when the official judge accepts this response.

    Integral floats such as ``100.0`` are parsed the way the judge parses
    them. A non-integer parses to "not a bet" and is not legal (the judge
    would then fold; that recovery is not legality).
    """

    parsed = JUDGE.parse_bot_response(response)
    if parsed is None:
        return False
    probe = _probe_game(state)
    try:
        probe.player_action(parsed)
    except ValueError:
        return False
    return True


def judge_min_positive(state: RefereeState) -> Optional[int]:
    """Smallest positive bet ``player_action`` accepts, or ``None``.

    The judge's own formula is ``2 * round_raise - own``. One accept and one
    reject probe confirm it. A disagreement falls back to a search over the
    real ``player_action`` and is reported by the caller via ``verified``.
    """

    if int(state.round_bet) < 0:
        return None
    chips = int(state.actor_chips)
    if chips <= 1:
        return None
    own = int(state.own_round_bet)
    if own < 0:
        return None
    formula = 2 * int(state.round_raise) - own
    guess = formula if formula >= 1 else 1
    if guess >= chips:
        if referee_is_legal(state, chips - 1):
            return _search_min_positive(state, chips)
        return None
    if referee_is_legal(state, guess) and (guess == 1 or not referee_is_legal(state, guess - 1)):
        return guess
    return _search_min_positive(state, chips)


def _search_min_positive(state: RefereeState, chips: int) -> Optional[int]:
    if not referee_is_legal(state, chips - 1):
        return None
    lo, hi = 1, chips - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if referee_is_legal(state, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# --- spec-based reference, NOT the real referee --------------------------------


SPEC_MODES = ("action_amount", "street_total")


def spec_round_raise(state: RefereeState, mode: str) -> int:
    """Phase-1 reading of ``round_raise``. Not used by ``referee_is_legal``.

    ``action_amount``: largest single-action chip amount this street
    (blinds count). This is the quantity the judge stores, kept here only
    so the cross-check can name it.
    ``street_total``: largest total commitment this street (``round_bet``).
    """

    if mode not in SPEC_MODES:
        raise ValueError(f"unknown spec mode {mode!r}; label={_SPEC_LABEL}")
    if mode == "action_amount":
        return int(state.round_raise)
    if int(state.round_bet) < 0:
        return 0
    return int(state.round_bet)


def spec_is_legal(state: RefereeState, response, mode: str = "action_amount") -> bool:
    """Secondary checker. ``_SPEC_LABEL``.

    Same shape as the judge formula ``bet + own >= 2 * round_raise``, with
    ``mode`` selecting which number is called ``round_raise``. All-in lock,
    strict chip check, and call ``0`` match the written spec. This function
    does not call ``player_action``.
    """

    if mode not in SPEC_MODES:
        raise ValueError(f"unknown spec mode {mode!r}; label={_SPEC_LABEL}")
    parsed = JUDGE.parse_bot_response(response)
    if parsed is None:
        return False
    if parsed == -1 or parsed == -2:
        return True
    if int(state.round_bet) < 0 or int(state.own_round_bet) < 0:
        return False
    chips = int(state.actor_chips)
    own = int(state.own_round_bet)
    if parsed == 0:
        inc = int(state.round_bet) - own
        return chips > inc
    if parsed <= 0 or chips <= parsed:
        return False
    return parsed + own >= 2 * spec_round_raise(state, mode)


def spec_label() -> str:
    return _SPEC_LABEL

"""Standard heads-up 200BB game for the CFR pipeline.

Legacy ``rlcard`` and ``botzone`` PPO backends deliberately retain their
historical rules. This path uses SB/button first preflop and BB first after
the flop, with 50/100 blinds and 20,000 starting chips every hand.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np
import torch

from .buckets import nearest_pot_fraction_bucket
from .schema import ActionEvent, normalize_card
from .tables import PokerkitTable, _TakenAction, _whole_chips, _card_text
from .tokenizer import BettingHistoryTokenizer

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 20_000
NUMERIC_VERSION = "hunl_exact_fraction_v1"
RULES_VERSION = "standard_hunl_200bb_v1"


def _branch_container(value):
    """Copy PokerKit's mutable containers, sharing its frozen value objects.

    Cards, streets, operations and rule tuples are immutable in PokerKit
    0.7.7. Lists may nest lists/deques (hole cards and dealing statuses).
    No engine method mutates a historical operation or a card in place.
    """
    if isinstance(value, list):
        return [_branch_container(item) for item in value]
    if isinstance(value, deque):
        return deque(value)
    if isinstance(value, set):
        return value.copy()
    if isinstance(value, dict):
        return {key: _branch_container(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=1)
def _evaluator():
    try:
        from phevaluator import evaluate_cards
    except ImportError as exc:
        raise ImportError(
            "HUNL CFR requires the fast phevaluator lookup evaluator; "
            "install requirements-cfr.txt. Use fast_evaluator=False only for diagnostics."
        ) from exc
    return evaluate_cards


def hand_rank(cards: Sequence[str]) -> int:
    """5–7 card lookup rank, with larger values representing stronger hands."""

    normalized = tuple(normalize_card(card) for card in cards)
    if not 5 <= len(normalized) <= 7 or len(set(normalized)) != len(normalized):
        raise ValueError("hand_rank requires five to seven distinct cards")
    return -int(_evaluator()(*(card[1] + card[0].lower() for card in normalized)))


@dataclass(frozen=True, order=True)
class _LookupHand:
    """Comparable adapter for PokerKit State's hand_types protocol.

    PokerKit 0.7.7 settlement calls ``from_game`` and compares the results;
    it never needs to enumerate five-card combinations or inspect labels.
    This adapter is local to HUNL states and is pickle/deepcopy compatible.
    """

    score: int

    @classmethod
    def from_game(cls, hole_cards, board_cards=()):
        cards = tuple(hole_cards) + tuple(board_cards)
        return cls(hand_rank(tuple(_card_text(card) for card in cards)))


class HunlTokenizer(BettingHistoryTokenizer):
    """Same tokens and four channels, retaining ratios above 10x pot.

    Legacy PPO tokenizers still clip their first action channel to 1. This
    explicit CFR format divides the true fraction by 10 without clipping.
    Checkpoints must record :attr:`numeric_version` to distinguish them.
    """

    numeric_version = NUMERIC_VERSION

    def __init__(self, max_seq_len=64, *, position_conditioning=True):
        super().__init__(max_seq_len)
        self.position_conditioning = position_conditioning

    def _tokens(self, snapshot):
        tokens = super()._tokens(snapshot)
        if self.position_conditioning:
            if snapshot.me_seat not in (0, 1) or snapshot.sb_seat != 1:
                raise ValueError("HUNL position encoding requires seat 0=BB, seat 1=SB/button")
            token, kind, _ = tokens[1]
            # HOLE is never truncated. One-hot position even at an empty
            # preflop history; no opponent private information is added.
            tokens[1] = (token, kind, (float(snapshot.me_seat == 0), float(snapshot.me_seat == 1), 0., 0.))
        return tokens

    def _action_numeric(self, event: ActionEvent, blind: float) -> Tuple[float, float, float, float]:
        legacy = super()._action_numeric(event, blind)
        return (float(event.pot_fraction) / 10.0, *legacy[1:])

    def _pad(self, sequences):
        # A DFS frame retains its observation while visiting descendants, so
        # outputs must own their storage. Pack entire sequences in NumPy,
        # avoiding a small Tensor allocation and four indexed writes/token.
        shape = (len(sequences), self.max_seq_len)
        arrays = {"token_ids": np.zeros(shape, dtype=np.int64),
                  "type_ids": np.zeros(shape, dtype=np.int64),
                  "numeric": np.zeros((*shape, self.num_numeric), dtype=np.float32),
                  "pad_mask": np.ones(shape, dtype=bool),
                  "legal_mask": np.zeros((len(sequences), self.num_actions), dtype=np.float32)}
        for row, sequence in enumerate(sequences):
            if not sequence:
                continue
            ids, types, numeric = zip(*sequence)
            count = len(sequence)
            arrays["token_ids"][row, :count] = ids
            arrays["type_ids"][row, :count] = types
            arrays["numeric"][row, :count] = numeric
            arrays["pad_mask"][row, :count] = False
        return {key: torch.from_numpy(value) for key, value in arrays.items()}


class HunlTable(PokerkitTable):
    """Locked standard HUNL rules with branchable, pre-sampled chance.

    Seat 0 is BB, seat 1 is SB/button. ``reset`` shuffles the complete deck;
    clones preserve that sampled chance outcome while action branches are
    evaluated. ``payoffs`` inherits chip deltas (divide by 100 for BB).
    """

    rules_version = RULES_VERSION

    def __init__(self, *, seed: int, fast_evaluator: bool = True) -> None:
        super().__init__(
            seed=seed, small_blind=SMALL_BLIND, big_blind=BIG_BLIND,
            starting_stack=STARTING_STACK,
        )
        self.fast_evaluator = bool(fast_evaluator)
        if self.fast_evaluator:
            _evaluator()

    def reset(self, deck=None) -> None:
        # RNG is read-only during a pre-sampled hand. Detach on reset so
        # sibling clones can reset independently without copying 625 RNG
        # integers at every action branch.
        rng = copy.copy(self._py_random)
        self._py_random = rng
        super().reset(deck)
        if self.fast_evaluator:
            self.state.hand_types = (_LookupHand,)
        if self.sb_seat != 1 or self.current_player() != self.sb_seat:
            raise RuntimeError("standard HUNL requires SB/button to act first preflop")

    def clone(self) -> "HunlTable":
        self._require_hand()
        branch = copy.copy(self)
        branch._records = self._records.copy()  # frozen _TakenAction values
        branch.state = copy.copy(self.state)
        branch.state.__dict__ = {
            key: _branch_container(value) for key, value in vars(self.state).items()
        }
        return branch

    def set_rng_state(self, state) -> None:
        self._py_random = copy.copy(self._py_random)
        super().set_rng_state(state)

    def events_for(self, me_seat: int) -> Tuple[ActionEvent, ...]:
        return tuple(
            replace(event, size_bucket=nearest_pot_fraction_bucket(event.pot_fraction))
            for event in super().events_for(me_seat)
        )

    def step_raise_to(self, amount: int) -> None:
        """Execute an arbitrary legal street raise-to, preserving true chips.

        Only the history token is bucketed; the actual betting state, pot
        odds, and numeric features use the exact accepted amount. This also
        accepts an all-in raise-to. Calls and folds use ``step(1/0)``.
        """

        self._require_hand()
        target = _whole_chips(amount, "raise_to")
        if not self.can_raise_to(target):
            raise ValueError(f"illegal HUNL raise-to {target}")
        state = self.state
        actor = self.current_player()
        bet = int(state.bets[actor])
        record = _TakenAction(
            street=self.street, actor_seat=actor, action_id=None,
            pot_before=int(state.total_pot_amount), chips_added=target - bet,
            to_call_before=int(max(state.bets) - bet),
            stack_before=int(state.stacks[actor]),
            legal_action_ids=self.legal_action_ids(),
        )
        state.complete_bet_or_raise_to(target)
        self._records.append(record)

    def wrapper_history(self):
        """A coarse five-action history for diagnostics with a conv encoder.

        Off-tree amounts cannot be replayed exactly from action ids. CFR
        uses snapshots/events, which always retain exact chip deltas.
        """

        history = [[], [], [], []]
        for record in self._records:
            action = record.action_id
            if action is None:
                if record.chips_added == record.stack_before:
                    action = 4
                elif record.chips_added <= 0.75 * record.pot_before:
                    action = 2
                else:
                    action = 3
            history[record.street].append([record.actor_seat, action, list(record.legal_action_ids)])
        return history

    def state_dict(self):
        blob = super().state_dict()
        blob.update(rules_version=self.rules_version, fast_evaluator=self.fast_evaluator)
        return blob

    def load_state_dict(self, blob) -> None:
        if blob.get("rules_version") != self.rules_version:
            raise ValueError("checkpoint is not a standard 200BB HUNL state")
        if bool(blob.get("fast_evaluator")) != self.fast_evaluator:
            raise ValueError("checkpoint hand evaluator differs from this HUNL table")
        super().load_state_dict(blob)

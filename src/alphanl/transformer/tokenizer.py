"""Betting-history tokenizer.

Sequence layout (CLS is always index 0, which the encoder reads):

    [CLS] [HOLE] [card] [card]
    [STREET_preflop] [action] ...
    [STREET_flop] [board cards] [action] ...
    ...

Each action token packs action type, SB/BB, whether the actor is ``me``, and
the pot-fraction bucket. A 4-d numeric vector carries the unbucketed fraction
and stack context so a later pokerkit backend can pass sizes the 5-way
abstraction never used.

Padding is on the right. If the hand is longer than ``max_seq_len``, the
oldest actions are dropped first. CLS, hole cards, and street markers are kept
while any action from that street remains.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .buckets import NUM_BUCKETS
from .schema import (
    ACTION_TYPES,
    NUM_ACTIONS,
    RANKS,
    SUITS,
    HandSnapshot,
    normalize_cards,
)

PAD_ID = 0
CLS_ID = 1
HOLE_MARKER_ID = 2
STREET_ID_BASE = 3  # + street index → 4 ids
CARD_ID_BASE = 7  # 52 cards
ACTION_ID_BASE = CARD_ID_BASE + 52

NUM_POSITIONS = 2
NUM_ACTOR_FLAGS = 2
VOCAB_SIZE = ACTION_ID_BASE + (len(ACTION_TYPES) * NUM_POSITIONS * NUM_ACTOR_FLAGS * NUM_BUCKETS)

TYPE_PAD = 0
TYPE_CLS = 1
TYPE_HOLE_MARKER = 2
TYPE_HOLE_CARD = 3
TYPE_STREET = 4
TYPE_BOARD_FLOP = 5
TYPE_BOARD_TURN = 6
TYPE_BOARD_RIVER = 7
TYPE_ACTION = 8
NUM_TYPES = 9

NUM_NUMERIC = 4

TensorBatch = Dict[str, torch.Tensor]


def _suit_rank_index(card: str) -> int:
    return SUITS.index(card[0]) * len(RANKS) + RANKS.index(card[1])


def action_token_id(action_type: str, position: str, actor_is_me: bool, size_bucket: int) -> int:
    type_index = ACTION_TYPES.index(action_type)
    pos_index = 0 if position == "SB" else 1
    me_index = 1 if actor_is_me else 0
    packed = ((type_index * NUM_POSITIONS + pos_index) * NUM_ACTOR_FLAGS + me_index) * NUM_BUCKETS
    return ACTION_ID_BASE + packed + int(size_bucket)


def _norm_chips(amount: float, starting_stack: float) -> float:
    if starting_stack <= 0:
        return 0.0
    return float(amount) / float(starting_stack)


def _norm_fraction(fraction: float) -> float:
    return max(0.0, min(float(fraction), 10.0)) / 10.0


_Token = Tuple[int, int, Tuple[float, float, float, float]]


class BettingHistoryTokenizer:
    def __init__(self, max_seq_len: int = 64) -> None:
        if max_seq_len < 8:
            raise ValueError("max_seq_len must be at least 8 (CLS + hole + one street)")
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = VOCAB_SIZE
        self.num_types = NUM_TYPES
        self.num_numeric = NUM_NUMERIC
        self.num_actions = NUM_ACTIONS

    def blank(self) -> TensorBatch:
        """Padded batch used only as a placeholder after a terminal step."""

        token_ids = torch.zeros(1, self.max_seq_len, dtype=torch.long)
        type_ids = torch.zeros(1, self.max_seq_len, dtype=torch.long)
        numeric = torch.zeros(1, self.max_seq_len, NUM_NUMERIC, dtype=torch.float32)
        pad_mask = torch.ones(1, self.max_seq_len, dtype=torch.bool)
        pad_mask[0, 0] = False
        token_ids[0, 0] = CLS_ID
        type_ids[0, 0] = TYPE_CLS
        legal = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)
        legal[0, 1] = 1.0
        return {
            "token_ids": token_ids,
            "type_ids": type_ids,
            "numeric": numeric,
            "pad_mask": pad_mask,
            "legal_mask": legal,
        }

    def describe(self, token_id: int) -> str:
        if token_id == PAD_ID:
            return "PAD"
        if token_id == CLS_ID:
            return "CLS"
        if token_id == HOLE_MARKER_ID:
            return "HOLE"
        if STREET_ID_BASE <= token_id < CARD_ID_BASE:
            return f"STREET_{token_id - STREET_ID_BASE}"
        if CARD_ID_BASE <= token_id < ACTION_ID_BASE:
            index = token_id - CARD_ID_BASE
            return f"CARD_{SUITS[index // 13]}{RANKS[index % 13]}"
        if ACTION_ID_BASE <= token_id < VOCAB_SIZE:
            packed = token_id - ACTION_ID_BASE
            bucket = packed % NUM_BUCKETS
            packed //= NUM_BUCKETS
            me = packed % NUM_ACTOR_FLAGS
            packed //= NUM_ACTOR_FLAGS
            pos = packed % NUM_POSITIONS
            type_index = packed // NUM_POSITIONS
            who = "me" if me else "opp"
            return f"ACT_{ACTION_TYPES[type_index]}_{'SB' if pos == 0 else 'BB'}_{who}_b{bucket}"
        return f"UNK_{token_id}"

    def _tokens(self, snapshot: HandSnapshot) -> List[_Token]:
        start = float(snapshot.starting_stack)
        zero = (0.0, 0.0, 0.0, 0.0)
        hole_cards = normalize_cards(snapshot.hole_cards)
        if len(hole_cards) != 2:
            raise ValueError(f"expected two hole cards, got {hole_cards}")
        board_cards = normalize_cards(snapshot.board_cards)
        cls_num = (
            _norm_chips(snapshot.pot, start),
            _norm_chips(snapshot.me_stack, start),
            _norm_chips(snapshot.opp_stack, start),
            _norm_chips(snapshot.to_call, start),
        )
        prefix: List[_Token] = [
            (CLS_ID, TYPE_CLS, cls_num),
            (HOLE_MARKER_ID, TYPE_HOLE_MARKER, zero),
            (CARD_ID_BASE + _suit_rank_index(hole_cards[0]), TYPE_HOLE_CARD, zero),
            (CARD_ID_BASE + _suit_rank_index(hole_cards[1]), TYPE_HOLE_CARD, zero),
        ]

        board_by_street: List[List[_Token]] = [[], [], [], []]
        for index, card in enumerate(board_cards[:5]):
            if index < 3:
                street, type_id = 1, TYPE_BOARD_FLOP
            elif index == 3:
                street, type_id = 2, TYPE_BOARD_TURN
            else:
                street, type_id = 3, TYPE_BOARD_RIVER
            board_by_street[street].append((CARD_ID_BASE + _suit_rank_index(card), type_id, zero))

        actions_by_street: List[List[_Token]] = [[], [], [], []]
        for event in snapshot.events:
            if not 0 <= event.street <= 3:
                continue
            token_id = action_token_id(
                event.action_type, event.position, event.actor_is_me, event.size_bucket
            )
            numeric = (
                _norm_fraction(event.pot_fraction),
                _norm_chips(event.pot_before, start),
                _norm_chips(event.stack_before, start),
                _norm_chips(event.to_call_before, start),
            )
            actions_by_street[event.street].append((token_id, TYPE_ACTION, numeric))

        last_street = max(0, min(3, int(snapshot.street)))
        blocks: List[List[_Token]] = []
        for street in range(last_street + 1):
            block = [(STREET_ID_BASE + street, TYPE_STREET, zero)]
            block.extend(board_by_street[street])
            block.extend(actions_by_street[street])
            blocks.append(block)

        flat = prefix + _fit_blocks(blocks, self.max_seq_len - len(prefix))
        if len(flat) > self.max_seq_len:
            flat = flat[: self.max_seq_len]
        return flat

    def _pad(self, sequences: Sequence[Sequence[_Token]]) -> TensorBatch:
        batch = len(sequences)
        token_ids = torch.zeros(batch, self.max_seq_len, dtype=torch.long)
        type_ids = torch.zeros(batch, self.max_seq_len, dtype=torch.long)
        numeric = torch.zeros(batch, self.max_seq_len, NUM_NUMERIC, dtype=torch.float32)
        pad_mask = torch.ones(batch, self.max_seq_len, dtype=torch.bool)
        for row, seq in enumerate(sequences):
            for col, (token_id, type_id, nums) in enumerate(seq):
                token_ids[row, col] = token_id
                type_ids[row, col] = type_id
                numeric[row, col] = torch.tensor(nums, dtype=torch.float32)
                pad_mask[row, col] = False
        return {
            "token_ids": token_ids,
            "type_ids": type_ids,
            "numeric": numeric,
            "pad_mask": pad_mask,
            "legal_mask": torch.zeros(batch, NUM_ACTIONS, dtype=torch.float32),
        }

    def encode(self, snapshot: HandSnapshot) -> TensorBatch:
        """Return a batch of size 1 on CPU."""

        batch = self._pad([self._tokens(snapshot)])
        for action_id in snapshot.legal_action_ids:
            if 0 <= action_id < NUM_ACTIONS:
                batch["legal_mask"][0, action_id] = 1.0
        if int(batch["legal_mask"].sum()) == 0:
            raise ValueError("snapshot has no legal actions in the 5-way head")
        return batch

    def encode_many(self, snapshots: Sequence[HandSnapshot]) -> TensorBatch:
        if not snapshots:
            raise ValueError("encode_many received no snapshots")
        batch = self._pad([self._tokens(s) for s in snapshots])
        for row, snapshot in enumerate(snapshots):
            for action_id in snapshot.legal_action_ids:
                if 0 <= action_id < NUM_ACTIONS:
                    batch["legal_mask"][row, action_id] = 1.0
            if int(batch["legal_mask"][row].sum()) == 0:
                raise ValueError("snapshot has no legal actions in the 5-way head")
        return batch


def _fit_blocks(blocks: List[List[_Token]], budget: int) -> List[_Token]:
    """Drop the oldest actions first. Then drop the oldest street blocks."""

    blocks = [list(block) for block in blocks]

    def length() -> int:
        return sum(len(block) for block in blocks)

    guard = 0
    while length() > budget and guard < 10000:
        guard += 1
        dropped_action = False
        for block in blocks:
            action_indexes = [i for i, tok in enumerate(block) if tok[1] == TYPE_ACTION]
            if action_indexes:
                del block[action_indexes[0]]
                dropped_action = True
                break
        if dropped_action:
            continue
        if len(blocks) > 1:
            blocks.pop(0)
            continue
        if blocks and len(blocks[0]) > 1:
            # Keep the street marker at position 0 of the only remaining block.
            del blocks[0][1]
            continue
        break
    flat: List[_Token] = []
    for block in blocks:
        flat.extend(block)
    if len(flat) > budget:
        flat = flat[-budget:]
    return flat


def cat_batches(batches: Sequence[TensorBatch]) -> TensorBatch:
    if not batches:
        raise ValueError("cat_batches received no batches")
    keys = tuple(batches[0].keys())
    for batch in batches[1:]:
        if tuple(batch.keys()) != keys:
            raise ValueError("cat_batches received batches with different keys")
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def slice_batch(batch: TensorBatch, index: torch.Tensor) -> TensorBatch:
    return {key: value.index_select(0, index) for key, value in batch.items()}


def batch_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    return {key: value.to(device) for key, value in batch.items()}

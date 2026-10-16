"""Betting-history observation, matching the training tokenizer.

Token layout, pot-fraction buckets, and ``log1p`` chip channels are the
ones in ``alphanl.transformer.tokenizer`` and ``buckets``. Action types
go through ``infer_action_type`` with an action id so a fold is not
stored as a check. The legal mask is the environment mask: fold stays
available on a free check. ``mask_free_fold`` is applied later, in the
forward pass.
"""

import math

from betting import (
    BIG_BLIND,
    NUM_ACTIONS,
    STARTING_STACK,
    apply,
    legal_action_ids,
    new_hand,
    to_call,
)

PAD_ID = 0
CLS_ID = 1
HOLE_MARKER_ID = 2
STREET_ID_BASE = 3
CARD_ID_BASE = 7
ACTION_ID_BASE = CARD_ID_BASE + 52

NUM_POSITIONS = 2
NUM_ACTOR_FLAGS = 2
BUCKET_RIGHT_EDGES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
NUM_BUCKETS = 1 + len(BUCKET_RIGHT_EDGES) + 1
ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
VOCAB_SIZE = ACTION_ID_BASE + (
    len(ACTION_TYPES) * NUM_POSITIONS * NUM_ACTOR_FLAGS * NUM_BUCKETS
)

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

_BZ_SUITS = "HDSC"
_TOK_SUITS = "CDHS"
_RANKS = "23456789TJQKA"
_REVEAL = (0, 3, 4, 5)
_LOG1P_200 = math.log1p(200.0)
_ZERO = (0.0, 0.0, 0.0, 0.0)


class Observation(object):
    """One decision, already padded to ``max_seq_len``."""

    def __init__(self, batch, to_call_chips, state):
        self.batch = batch
        self.to_call = to_call_chips
        self.state = state


def build_observation(request, max_seq_len=64):
    """Arrays the training ``BettingHistoryTokenizer.encode`` would emit."""

    state, records = replay_training_state(request)
    if len(request["public_cards"]) != _REVEAL[state.street]:
        raise ValueError("public_cards length does not match the training street")
    hole = [_card_index(card) for card in request["my_cards"]]
    if len(hole) != 2:
        raise ValueError("expected two hole cards")
    board = [_card_index(card) for card in request["public_cards"]]
    tokens = _tokens(state, records, hole, board, int(request["my_id"]), max_seq_len)
    batch = _pad(tokens, max_seq_len)
    for action_id in legal_action_ids(state):
        batch["legal_mask"][0, action_id] = 1.0
    if int(batch["legal_mask"].sum()) == 0:
        raise ValueError("snapshot has no legal actions in the 5-way head")
    owed = float(to_call(state))
    return Observation(batch, owed, state)


def replay_training_state(request):
    """Replay ``history`` with the training betting rules."""

    dealer = request["dealer_id"]
    my_id = request["my_id"]
    if dealer not in (0, 1) or my_id not in (0, 1):
        raise ValueError("dealer_id and my_id must be 0 or 1")
    state = new_hand(dealer)
    records = []
    for entry in request["history"]:
        if state.hand_over:
            raise ValueError("history continues after the hand ended")
        response = _coerce_action(entry["action"])
        pid = entry["player_id"]
        if pid not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        if pid != state.actor:
            raise ValueError("history actor is not the training actor")
        is_fold = response == -1 or entry.get("action_type") == "fold"
        if "round" in entry:
            if int(entry["round"]) != state.street:
                raise ValueError("history street does not match the training street")
        elif not is_fold:
            raise ValueError("non-fold history row has no round")
        owed = int(to_call(state))
        stack_before = int(state.stacks[state.actor])
        chips = _chips_added(state, response)
        records.append(
            (
                int(state.street),
                int(state.actor),
                int(state.pot),
                chips,
                owed,
                stack_before,
                _action_id(response, owed, chips, stack_before),
            )
        )
        state = apply(state, response)
    if state.hand_over:
        raise ValueError("decision requested on a finished hand")
    if state.actor != my_id:
        raise ValueError("my_id is not the training actor")
    return state, records


def _tokens(state, records, hole, board, my_id, max_seq_len):
    blind = float(BIG_BLIND)
    sb_seat = (int(state.dealer) + 1) % 2
    cls_num = (
        _norm_chips(state.pot, blind),
        _norm_chips(state.stacks[my_id], blind),
        _norm_chips(state.stacks[1 - my_id], blind),
        _norm_chips(to_call(state), blind),
    )
    prefix = [
        (CLS_ID, TYPE_CLS, cls_num),
        (HOLE_MARKER_ID, TYPE_HOLE_MARKER, _ZERO),
        (CARD_ID_BASE + hole[0], TYPE_HOLE_CARD, _ZERO),
        (CARD_ID_BASE + hole[1], TYPE_HOLE_CARD, _ZERO),
    ]
    board_by_street = [[], [], [], []]
    for index, card_index in enumerate(board[:5]):
        if index < 3:
            street, type_id = 1, TYPE_BOARD_FLOP
        elif index == 3:
            street, type_id = 2, TYPE_BOARD_TURN
        else:
            street, type_id = 3, TYPE_BOARD_RIVER
        board_by_street[street].append((CARD_ID_BASE + card_index, type_id, _ZERO))

    actions_by_street = [[], [], [], []]
    for street, actor, pot_before, chips, owed, stack_before, action_id in records:
        if not 0 <= street <= 3:
            continue
        kind = _infer_action_type(
            action_id, owed, chips, float(stack_before) - float(chips)
        )
        fraction = 0.0 if pot_before <= 0 else float(chips) / float(pot_before)
        token_id = _action_token_id(kind, actor == sb_seat, actor == my_id, _bucket(fraction))
        numeric = (
            _norm_fraction(fraction),
            _norm_chips(pot_before, blind),
            _norm_chips(stack_before, blind),
            _norm_chips(owed, blind),
        )
        actions_by_street[street].append((token_id, TYPE_ACTION, numeric))

    last_street = state.street
    if last_street < 0:
        last_street = 0
    if last_street > 3:
        last_street = 3
    blocks = []
    for street in range(last_street + 1):
        block = [(STREET_ID_BASE + street, TYPE_STREET, _ZERO)]
        block.extend(board_by_street[street])
        block.extend(actions_by_street[street])
        blocks.append(block)
    flat = prefix + _fit_blocks(blocks, max_seq_len - len(prefix))
    if len(flat) > max_seq_len:
        flat = flat[:max_seq_len]
    return flat


def _pad(tokens, max_seq_len):
    # Imported here so a missing NumPy fails the model, not process import
    # order in tests that only want the fallback path's syntax checked.
    # Thread caps first: this can be the first NumPy import in a process.
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import numpy as np

    token_ids = np.zeros((1, max_seq_len), dtype=np.int64)
    type_ids = np.zeros((1, max_seq_len), dtype=np.int64)
    numeric = np.zeros((1, max_seq_len, NUM_NUMERIC), dtype=np.float32)
    pad_mask = np.ones((1, max_seq_len), dtype=np.bool_)
    for col, (token_id, type_id, nums) in enumerate(tokens):
        token_ids[0, col] = token_id
        type_ids[0, col] = type_id
        numeric[0, col, 0] = np.float32(nums[0])
        numeric[0, col, 1] = np.float32(nums[1])
        numeric[0, col, 2] = np.float32(nums[2])
        numeric[0, col, 3] = np.float32(nums[3])
        pad_mask[0, col] = False
    legal = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
    return {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "numeric": numeric,
        "pad_mask": pad_mask,
        "legal_mask": legal,
    }


def _fit_blocks(blocks, budget):
    """Drop the oldest actions first. Then drop the oldest street blocks."""

    blocks = [list(block) for block in blocks]

    def length():
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
            del blocks[0][1]
            continue
        break
    flat = []
    for block in blocks:
        flat.extend(block)
    if len(flat) > budget:
        flat = flat[-budget:]
    return flat


def _action_token_id(action_type, is_sb, actor_is_me, size_bucket):
    type_index = ACTION_TYPES.index(action_type)
    pos_index = 0 if is_sb else 1
    me_index = 1 if actor_is_me else 0
    packed = ((type_index * NUM_POSITIONS + pos_index) * NUM_ACTOR_FLAGS + me_index) * NUM_BUCKETS
    return ACTION_ID_BASE + packed + int(size_bucket)


def _infer_action_type(action_id, to_call_before, chips_added, stack_after):
    if action_id == 0:
        return "fold"
    if chips_added > 0 and stack_after <= 0:
        return "all_in"
    if action_id == 4:
        return "all_in"
    if action_id == 1 and to_call_before <= 0:
        return "check"
    if action_id == 1:
        return "call"
    if action_id in (2, 3) and to_call_before <= 0:
        return "bet"
    if action_id in (2, 3):
        return "raise"
    if to_call_before <= 0 and chips_added <= 0:
        return "check"
    if to_call_before > 0 and chips_added <= to_call_before:
        return "call"
    if to_call_before <= 0:
        return "bet"
    return "raise"


def _action_id(response, to_call_before, chips_added, stack_before):
    """Id ``infer_action_type`` would see for this judge integer.

    Fold must be id 0. A chip-only guess would call a fold a check when
    nothing is owed. Ids 2 and 3 share a token type; the size bucket
    carries the amount.
    """

    if response == -1:
        return 0
    if chips_added > 0 and stack_before - chips_added <= 0:
        return 4
    if response == -2:
        return 4
    if response == 0:
        return 1
    if response > 0:
        return 2
    raise ValueError("unrecognized action")


def _chips_added(state, response):
    if response == -1:
        return 0
    if response == -2:
        return int(state.stacks[state.actor])
    if response == 0:
        return int(to_call(state))
    return int(response)


def _bucket(pot_fraction):
    if pot_fraction <= 0.0:
        return 0
    for index, edge in enumerate(BUCKET_RIGHT_EDGES):
        if pot_fraction <= edge:
            return index + 1
    return NUM_BUCKETS - 1


def _norm_chips(amount, big_blind):
    if big_blind <= 0:
        return 0.0
    return math.log1p(float(amount) / float(big_blind)) / _LOG1P_200


def _norm_fraction(fraction):
    return max(0.0, min(float(fraction), 10.0)) / 10.0


def _card_index(card):
    if isinstance(card, bool) or not isinstance(card, int) or not 0 <= card <= 51:
        raise ValueError("card id must be an int in 0..51")
    suit = _BZ_SUITS[card % 4]
    rank = _RANKS[card // 4]
    return _TOK_SUITS.index(suit) * len(_RANKS) + _RANKS.index(rank)


def _coerce_action(action):
    if isinstance(action, bool) or action is None:
        raise ValueError("action is not an integer")
    if isinstance(action, int):
        return action
    if isinstance(action, float) and action.is_integer():
        return int(action)
    raise ValueError("action is not an integer")


# Starting stack is part of the training snapshot. The tokenizer does not
# read it; the parity test does.
TRAINING_STARTING_STACK = STARTING_STACK

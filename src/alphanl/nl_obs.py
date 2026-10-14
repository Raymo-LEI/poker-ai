"""AlphaNLHoldem observation tensors. NumPy only, so the Python 3.7 dump can import it.

This is a direct port of ``NlHoldemEnvWrapper._get_observation`` in
``third_party/AlphaNLHoldem/agi/nl_holdem_env.py``. Do not import torch,
tensorflow, or ``alphanl.transformer`` from here.
"""

from __future__ import annotations

import numpy as np

SUITS = "CDHS"
RANKS = "23456789TJQKA"
NUM_ACTIONS = 5


def card_suit_rank(card):
    """Suit-then-rank index. Matches the TF wrapper (``card[0]`` suit, ``card[1]`` rank)."""

    text = str(card).strip()
    if len(text) != 2:
        raise ValueError("card must be two characters, got %r" % (card,))
    suit = SUITS.find(text[0])
    rank = RANKS.find(text[1])
    if suit < 0 or rank < 0:
        raise ValueError("unrecognized card %r" % (card,))
    return suit, rank


def legal_ids(legal):
    if legal is None:
        return []
    if isinstance(legal, dict):
        items = list(legal.keys())
    else:
        items = list(legal)
    out = []
    for action in items:
        value = action.value if hasattr(action, "value") else action
        action_id = int(value)
        if 0 <= action_id < NUM_ACTIONS:
            out.append(action_id)
    return out


def encode_observation(raw_obs, history, player_id):
    """Return float32 arrays shaped like the TF env, before the network's cast.

    ``history[street]`` entries are ``[seat, action_id, legal_ids]``, the same
    list ``NlHoldemEnvWrapper`` appends. Only the first 6 actions of each
    street are written, matching ``one_history[:6]``.

    ``extra_info`` is the two stakes, truncated to uint8 the way the TF env
    stores them. ``NlHoldemNet`` does not read this vector; see
    ``docs/bc_and_baseline.md``.
    """

    card_info = np.zeros((4, 13, 6), np.float32)
    action_info = np.zeros((4, NUM_ACTIONS, 25), np.float32)
    extra_info = np.zeros((2,), np.float32)
    legal_mask = np.zeros((NUM_ACTIONS,), np.float32)

    hand = list(raw_obs["hand"])
    public = list(raw_obs.get("public_cards") or [])

    def mark(cards, channel):
        for card in cards:
            suit, rank = card_suit_rank(card)
            card_info[suit, rank, channel] = 1.0

    mark(hand, 0)
    mark(public[:3], 1)
    mark(public[3:4], 2)
    mark(public[4:5], 3)
    mark(public, 4)
    mark(list(public) + list(hand), 5)

    for action_id in legal_ids(raw_obs.get("legal_actions")):
        legal_mask[action_id] = 1.0

    for street, actions in enumerate(list(history)[:4]):
        for index, record in enumerate(list(actions)[:6]):
            seat = int(record[0])
            action_id = int(record[1])
            channel = street * 6 + index
            if 0 <= seat <= 1 and 0 <= action_id < NUM_ACTIONS:
                action_info[seat, action_id, channel] = 1.0
                action_info[2, action_id, channel] = 1.0
            legal_then = record[2] if len(record) > 2 else []
            for action in legal_ids(legal_then):
                action_info[3, action, channel] = 1.0

    # The TF wrapper assigns the current seat id to the whole last channel.
    action_info[:, :, -1] = float(int(player_id))

    stakes = raw_obs.get("stakes")
    if stakes is not None and len(stakes) >= 2:
        extra_info[0] = float(np.uint8(int(stakes[0])))
        extra_info[1] = float(np.uint8(int(stakes[1])))

    return {
        "card_info": card_info,
        "action_info": action_info,
        "extra_info": extra_info,
        "legal_mask": legal_mask,
    }


def masked_softmax_tf(logits, legal):
    """Softmax after the TF mask ``max(log(legal), float32.min)``.

    ``NlHoldemNet.forward`` adds that mask to the 5 logits and lets the
    categorical distribution softmax the sum. Used by the dump and the
    comparison script so both sides apply the published mask.
    """

    logits = np.asarray(logits, dtype=np.float64).reshape(-1, NUM_ACTIONS)
    legal = np.asarray(legal, dtype=np.float64).reshape(-1, NUM_ACTIONS)
    with np.errstate(divide="ignore"):
        inf_mask = np.log(legal)
    floor = float(np.finfo(np.float32).min)
    inf_mask = np.maximum(inf_mask, floor)
    masked = logits + inf_mask
    masked = masked - masked.max(axis=-1, keepdims=True)
    exp = np.exp(masked)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.astype(np.float32)

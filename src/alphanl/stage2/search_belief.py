"""Combo-level Bayes with an explicit frozen-policy observation model."""
import copy
import numpy as np
from .search_math import combos, card_text, hand_masks
from .search_game import PublicState


class RangeError(ValueError):
    pass


def normalized_policy(probs, legal):
    p = np.asarray(probs, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or (p < 0).any():
        raise RangeError('invalid conditional action probabilities')
    mask = np.zeros(5, dtype=bool)
    mask[list(legal)] = True
    p[:, ~mask] = 0.
    z = p.sum(1, keepdims=True)
    if (z <= 0).any():
        raise RangeError('empty conditional legal policy')
    return p/z


class ComboBelief:
    def __init__(self, hands, weights=None):
        self.hands = tuple(tuple(card_text(c) for c in h) for h in hands)
        hand_masks(self.hands)
        self.weights = np.ones(len(hands)) if weights is None else np.array(weights, dtype=float, copy=True)
        if self.weights.shape != (len(hands),) or (self.weights < 0).any() or not np.isfinite(self.weights).all():
            raise RangeError('invalid range weights')
        self._normalize()

    def _normalize(self):
        z = self.weights.sum()
        if z <= 0:
            raise RangeError('zero posterior range')
        self.weights /= z

    def update(self, likelihood):
        p = np.asarray(likelihood)
        if p.shape != self.weights.shape or (p < 0).any() or (p > 1+1e-12).any() or not np.isfinite(p).all():
            raise RangeError('invalid event likelihood')
        self.weights *= p
        self._normalize()

    def reveal(self, board):
        blocked = set(card_text(c) for c in board)
        self.weights *= np.array([not (set(h) & blocked) for h in self.hands])
        self._normalize()

    def aggregate(self, bucket_ids, size):
        ids = np.asarray(bucket_ids)
        if ids.shape != self.weights.shape or (ids < 0).any() or (ids >= size).any():
            raise RangeError('invalid independent bucket map')
        return np.bincount(ids, weights=self.weights, minlength=size)


def with_hole(snapshot, hand):
    snap = copy.copy(snapshot)
    object.__setattr__(snap, 'hole_cards', tuple(c[1].upper()+c[0] for c in map(card_text, hand)))
    return snap


def policy_rows(agent, snapshot, context, hands, chunk=64):
    rows = []
    for start in range(0, len(hands), chunk):
        batches = [agent.raw_observation(with_hole(snapshot, h), context) for h in hands[start:start+chunk]]
        batch = {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}
        logits = agent.logits(batch).astype(np.float64)
        p = np.exp(logits-logits.max(axis=1, keepdims=True))
        rows.append(normalized_policy(p, snapshot.legal_action_ids))
    return np.concatenate(rows)


def event_snapshot(snapshot, index):
    """Reconstruct only fields available immediately BEFORE this action."""
    event = snapshot.events[index]
    seat = event.actor_seat
    stacks = [int(snapshot.starting_stack)]*2
    stacks[snapshot.sb_seat] -= 50
    stacks[1-snapshot.sb_seat] -= 100
    for previous in snapshot.events[:index]:
        stacks[previous.actor_seat] -= int(previous.chips_added)
    board_len = (0,3,4,5)[event.street]
    kwargs = dict(events=tuple(snapshot.events[:index]), hole_cards=snapshot.hole_cards,
                  board_cards=snapshot.board_cards[:board_len], me_seat=seat, sb_seat=snapshot.sb_seat,
                  street=event.street, pot=event.pot_before, stacks=tuple(stacks),
                  to_call=event.to_call_before, legal_action_ids=(0,1,2,3,4),
                  big_blind=snapshot.big_blind, starting_stack=snapshot.starting_stack)
    # actor_is_me is relative to the new observer, not the current hero.
    rewritten = []
    for previous in kwargs['events']:
        prev = copy.copy(previous)
        object.__setattr__(prev, 'actor_is_me', prev.actor_seat == seat)
        rewritten.append(prev)
    kwargs['events'] = tuple(rewritten)
    out = snapshot.__class__(**kwargs)
    legal = tuple(PublicState.from_snapshot(out).actions())
    object.__setattr__(out, 'legal_action_ids', legal)
    return out


def history_beliefs(agent, snapshot, context):
    """Replay from the same combo support; no street resampling or epsilon mass.

    Neutral match context is intentional: this is a declared observation model,
    not a claim that the opponent follows our blueprint or actual historical HUD.
    """
    hands = combos(snapshot.board_cards)
    beliefs = [ComboBelief(hands), ComboBelief(hands)]
    neutral = context.__class__()
    for i, event in enumerate(snapshot.events):
        state = event_snapshot(snapshot, i)
        rows = policy_rows(agent, state, neutral, hands)
        if event.action_type == 'fold':
            likelihood = rows[:, 0]
        elif event.chips_added <= event.to_call_before:
            likelihood = rows[:, 1]  # includes an all-in CALL
        else:
            likelihood = rows[:, 2:].sum(1)  # explicit size-coarsened observation
        beliefs[0 if event.actor_is_me else 1].update(likelihood)
    return hands, tuple(b.weights for b in beliefs)


def sample_ranges(hands, weights, hero, count=8, seed=20261104):
    """Uniform without-replacement support, inverse-inclusion mass correction.

    The actual hero row has inclusion 1. All other hero rows have count/(N-1);
    opponent rows have count/N. No deterministic strength stride or partition.
    This finite-support game is approximate, not an unbiased equilibrium.
    """
    rng = np.random.RandomState(seed)
    hero_set = set(map(card_text, hero))
    at = next(i for i,h in enumerate(hands) if set(h) == hero_set)
    n = len(hands)
    k = min(count, n-1)
    rest = np.array([i for i in range(n) if i != at])
    hi = np.r_[at, rng.choice(rest, k, replace=False)]
    oi = rng.choice(n, min(count+1,n), replace=False)
    hw = weights[0][hi].copy()
    hw[1:] *= (n-1)/float(k)
    ow = weights[1][oi] * n/float(len(oi))
    if hw.sum() <= 0 or ow.sum() <= 0:
        raise RangeError('sample missed all posterior mass; increase support')
    return (tuple(hands[i] for i in hi), tuple(hands[i] for i in oi)), (hw,ow)

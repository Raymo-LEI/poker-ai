"""NumPy-only public-tree DCFR and exact blocker-aware showdown operators.

See docs/math_repair/CONTRACT.md. Larger fixed-width ranks are stronger.
"""
import itertools
import numpy as np
from .portable_evaluator import strength

DECK = tuple(r + s for s in 'shdc' for r in '23456789TJQKA')
CARD_ID = {card: i for i, card in enumerate(DECK)}
CONTRACT = 'public_tree_dcfr_v1'


def card_text(card):
    text = card.strip()
    if len(text) != 2:
        raise ValueError('card must have two characters')
    result = text[0].upper() + text[1].lower()
    if result not in CARD_ID:
        result = text[1].upper() + text[0].lower()
    if result not in CARD_ID:
        raise ValueError('invalid card: ' + card)
    return result


def rank(cards):
    cards = [card_text(c) for c in cards]
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError('rank requires 5-7 distinct cards')
    key = strength(cards)
    value = 0
    for digit in key + (0,) * (6 - len(key)):
        value = value * 16 + digit
    return value


def combos(board=()):
    blocked = set(card_text(c) for c in board)
    return tuple(itertools.combinations([c for c in DECK if c not in blocked], 2))


def hand_masks(hands):
    result = []
    for hand in hands:
        if len(hand) != 2 or hand[0] == hand[1]:
            raise ValueError('each hand must contain two distinct cards')
        result.append(sum(1 << CARD_ID[card_text(c)] for c in hand))
    return np.asarray(result, dtype=np.int64)


def conditional(numerator, mass):
    out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    np.divide(numerator, mass, out=out, where=mass > 0)
    return out


class ShowdownKernel:
    """Full public combo domain; group-prefix sums plus single-card exclusion.

    contract(w) returns legal mass Z and winning plus half-tied mass S,
    both UNNORMALIZED. Empty rows have Z=0 (equity is undefined).
    """
    def __init__(self, board):
        self.board = tuple(card_text(c) for c in board)
        if len(self.board) != 5 or len(set(self.board)) != 5:
            raise ValueError('showdown kernel requires five distinct board cards')
        self.hands = combos(self.board)
        self.masks = hand_masks(self.hands)
        self.ranks = np.array([rank(list(h) + list(self.board)) for h in self.hands])
        self.groups, self.g = np.unique(self.ranks, return_inverse=True)
        self.c1 = np.array([CARD_ID[h[0]] for h in self.hands])
        self.c2 = np.array([CARD_ID[h[1]] for h in self.hands])
        self.gsize = len(self.groups)

    def contract(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.hands),) or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('invalid combo weights')
        mass = np.bincount(self.g, weights=w, minlength=self.gsize)
        cm = np.bincount(np.r_[self.g * 52 + self.c1, self.g * 52 + self.c2],
                         weights=np.r_[w, w], minlength=self.gsize * 52).reshape(-1, 52)
        prefix = np.r_[0., np.cumsum(mass)]
        cp = np.vstack([np.zeros((1, 52)), np.cumsum(cm, axis=0)])
        lower = prefix[self.g] - cp[self.g, self.c1] - cp[self.g, self.c2]
        tied = mass[self.g] - cm[self.g, self.c1] - cm[self.g, self.c2] + w
        total_card = cm.sum(axis=0)
        z = w.sum() - total_card[self.c1] - total_card[self.c2] + w
        # Cancellation can leave roundoff in a genuinely empty sparse row.
        support = (w > 0).astype(float)
        card_count = np.bincount(np.r_[self.c1,self.c2],weights=np.r_[support,support],minlength=52)
        count = support.sum()-card_count[self.c1]-card_count[self.c2]+support
        z = np.where(count > 0,np.maximum(z,0.),0.)
        won = lower + .5 * tied
        # The standard n-term floating-sum error bound identifies cancellation,
        # not a probability floor. Recompute only ill-conditioned rows densely.
        eps=np.finfo(np.float64).eps
        bound=len(w)*eps/(1-len(w)*eps)*w.sum()
        for i in np.flatnonzero((count>0)&(z<=bound)):
            legal=(self.masks & self.masks[i])==0
            z[i]=w[legal].sum()
            won[i]=w[legal & (self.ranks<self.ranks[i])].sum()+.5*w[legal & (self.ranks==self.ranks[i])].sum()
        won = np.clip(won, 0., z)
        return z, won

    def equity(self, weights):
        z, won = self.contract(weights)
        return conditional(won, z), z > 0


class Terminal:
    def __init__(self, payoff=None, coefficients=None):
        self.payoff = None if payoff is None else np.asarray(payoff, dtype=np.float64)
        self.coefficients = coefficients


class Node:
    def __init__(self, player, children, key=None):
        if player not in (0, 1) or not children:
            raise ValueError('decision requires a player and legal children')
        self.player, self.children = player, dict(children)
        self.actions = tuple(self.children)
        self.key = id(self) if key is None else key


class Chance:
    """Public chance branches. Probabilities may be pair-dependent matrices."""
    def __init__(self, children):
        self.children = tuple(children)  # (probability, child)


class DCFR:
    def __init__(self, root, joint=None, *, kernel=None, ranges=None,
                 alpha=1.5, beta=0., gamma=2.):
        self.root, self.kernel = root, kernel
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.regret, self.sums, self.nodes = {}, {}, {}
        if kernel is None:
            self.joint = np.asarray(joint, dtype=np.float64)
            if self.joint.ndim != 2 or not np.isfinite(self.joint).all() or (self.joint < 0).any():
                raise ValueError('invalid joint deal')
            self.mass = (self.joint.sum(1), self.joint.sum(0))
        else:
            self.joint = None
            self.ranges = tuple(np.asarray(w, dtype=np.float64) for w in ranges)
            for w in self.ranges:
                if w.shape != (len(kernel.hands),) or (w < 0).any() or not np.isfinite(w).all():
                    raise ValueError('invalid range')
            self.mass = tuple(self.ranges[p] * kernel.contract(self.ranges[1-p])[0] for p in (0, 1))
        if self.mass[0].sum() <= 0:
            raise ValueError('zero compatible joint range')
        self.shape = tuple(len(m) for m in self.mass)
        self._collect(root)
        self.iterations = 0

    def _collect(self, node):
        if isinstance(node, Terminal):
            if self.kernel is None and (node.payoff.shape != self.shape or not np.isfinite(node.payoff).all()):
                raise ValueError('terminal payoff must cover the actual joint domain')
            return
        if isinstance(node, Chance):
            if self.kernel is not None:
                raise ValueError('chance trees use the dense reference operator')
            probs = sum(np.broadcast_to(p, self.shape) for p, _ in node.children)
            if not np.allclose(probs[self.joint > 0], 1., atol=1e-12):
                raise ValueError('chance probabilities do not normalize on each legal deal')
            if any((np.asarray(p) < 0).any() for p, _ in node.children):
                raise ValueError('negative chance probability')
            for _, child in node.children:
                self._collect(child)
            return
        if node.key in self.nodes:
            previous = self.nodes[node.key]
            if previous.player != node.player or previous.actions != node.actions:
                raise ValueError('inconsistent information set')
        else:
            self.nodes[node.key] = node
            size = (self.shape[node.player], len(node.actions))
            self.regret[node.key], self.sums[node.key] = np.zeros(size), np.zeros(size)
        for child in node.children.values():
            self._collect(child)

    @staticmethod
    def matching(regret):
        positive = np.maximum(regret, 0.)
        total = positive.sum(1, keepdims=True)
        out = np.full(regret.shape, 1. / regret.shape[1])
        np.divide(positive, total, out=out, where=total > 0)
        return out

    @staticmethod
    def discount(regret, t, alpha=1.5, beta=0.):
        return np.where(regret >= 0, regret * t**alpha / (t**alpha + 1.),
                        regret * t**beta / (t**beta + 1.))

    def _terminal(self, node, reach, joint):
        if self.kernel is None:
            utility = joint * node.payoff
            return utility.dot(reach[1]), -utility.T.dot(reach[0])
        # u0 = a*equity0 + b, u1=-u0; card-removal mass enters constants too.
        a, b = node.coefficients
        z1, s1 = self.kernel.contract(self.ranges[1] * reach[1])
        z0, s0 = self.kernel.contract(self.ranges[0] * reach[0])
        return (self.ranges[0] * (a*s1 + b*z1),
                self.ranges[1] * (a*s0 - (a+b)*z0))

    def _walk(self, node, reach, strategy, joint, update=False):
        if isinstance(node, Terminal):
            return self._terminal(node, reach, joint)
        if isinstance(node, Chance):
            values = [self._walk(child, reach, strategy, joint*p, update) for p, child in node.children]
            return tuple(sum(v[p] for v in values) for p in (0, 1))
        p = node.player
        sigma = strategy[node.key]
        children = []
        for a, child in enumerate(node.children.values()):
            next_reach = list(reach)
            next_reach[p] = reach[p] * sigma[:, a]
            children.append(self._walk(child, next_reach, strategy, joint, update))
        av = np.stack([v[p] for v in children], axis=1)
        own = (sigma * av).sum(1)
        other = sum(v[1-p] for v in children)
        if update:
            self.instant[node.key] += av - own[:, None]
            if node.key not in self.visited:
                self.sums[node.key] += self.t**self.gamma * reach[p][:, None] * sigma
                self.visited[node.key] = reach[p].copy()
            elif not np.allclose(self.visited[node.key],reach[p],atol=1e-14,rtol=0.):
                raise ValueError('information set merges distinct own action histories')
        return (own, other) if p == 0 else (other, own)

    def solve(self, iterations=40):
        if iterations < 1:
            raise ValueError('positive iteration budget required')
        for _ in range(iterations):
            self.t = self.iterations + 1
            strategy = {k: self.matching(r) for k, r in self.regret.items()}
            self.instant = {k: np.zeros_like(r) for k, r in self.regret.items()}
            self.visited = {}
            self._walk(self.root, [np.ones(n) for n in self.shape], strategy, self.joint, True)
            for k in self.regret:
                self.regret[k] = self.discount(self.regret[k], self.t, self.alpha, self.beta) + self.instant[k]
            self.iterations += 1
        average = {k: self.matching(s) for k, s in self.sums.items()}
        cfv = self._walk(self.root, [np.ones(n) for n in self.shape], average, self.joint)
        return dict(strategy=average, values=tuple(conditional(cfv[p], self.mass[p]) for p in (0, 1)),
                    reachable=tuple(m > 0 for m in self.mass), mass=self.mass,
                    value=float(cfv[0].sum()/self.mass[0].sum()), iterations=self.iterations,
                    contract=CONTRACT, output='dcfr_average',
                    zero_sum_error=float(abs(cfv[0].sum()+cfv[1].sum())/self.mass[0].sum()))

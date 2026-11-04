"""Executable chip tree, reusing the portable judge's legality and chip writes.

The public abstraction excludes undercalls accepted by the unusual platform
positive-bet test. Unequal-stack heads-up all-ins use explicit refunds.
"""
import copy
import numpy as np
try:
    from . import reconstruct as judge
except ImportError:
    from botzone_bots.model import reconstruct as judge
from .search_math import Node, Terminal


class PublicState:
    def __init__(self, pot, stacks, to_call=0, street=3, street_put=None,
                 round_raise=None, actor=0, checked=False, first=0, pending=None):
        self.view = judge.HandView()
        self.view.pot = int(pot)
        self.view.chips = list(map(int, stacks))
        self.view.street = int(street)
        self.view.street_put = list(street_put if street_put is not None else (0, int(to_call)))
        self.view.status = [self.view.street_put[p] if self.view.chips[p] else -2 for p in (0, 1)]
        self.view.round_raise = int(round_raise if round_raise is not None else (100 if street == 0 else 50))
        self.view.round_bet = -2 if min(stacks) == 0 else max(self.view.street_put)
        self.view.p_commit = list(self.view.street_put)
        self.actor, self.checked, self.first = actor, checked, first
        self.pending = (1 if to_call or checked else 2) if pending is None else pending
        self.terminal, self.winner = None, None
        if pot < sum(self.view.street_put) or min(stacks) < 0:
            raise ValueError('inconsistent public chip state')

    @classmethod
    def from_snapshot(cls, snap):
        # Accumulate actual per-street contributions, never infer them from pot fractions.
        puts = [0, 0]
        if snap.street == 0:
            puts = [50 if snap.me_seat == snap.sb_seat else 100,
                    100 if snap.me_seat == snap.sb_seat else 50]
        rr = 100 if snap.street == 0 else 50
        current = [e for e in snap.events if e.street == snap.street]
        for e in current:
            p = 0 if e.actor_is_me else 1
            puts[p] += int(e.chips_added)
            if e.action_type in ('bet', 'raise'):
                rr = max(rr, int(e.chips_added))
        # Synthetic snapshots may omit events but still declare the outstanding bet.
        if not current and snap.to_call and max(puts) - puts[0] != int(snap.to_call):
            puts = [0, int(snap.to_call)]
        first = 0 if snap.me_seat == snap.sb_seat else 1
        return cls(snap.pot, (snap.me_stack, snap.opp_stack), snap.to_call,
                   snap.street, puts, rr, checked=bool(current and current[-1].action_type == 'check'), first=first,
                   pending=2 if not current and snap.street==0 and snap.me_seat==snap.sb_seat else
                           1 if current or snap.to_call else 2)

    @property
    def owed(self):
        return max(self.view.street_put) - self.view.street_put[self.actor]

    def actions(self, amounts=None, allow_raise=True):
        if self.terminal:
            return {}
        v, p = self.view, self.actor
        v.my_id = p
        owed, stack = self.owed, v.chips[p]
        if stack <= 0:
            raise ValueError('all-in player cannot act')
        out = {0: 0, 1: min(owed, stack)} if owed else {1: 0}
        # Action 1 is a call including a short/all-in call, never an aggressive jam.
        if allow_raise and v.chips[1-p] > 0 and stack > owed:
            if amounts is None:
                minimum = judge.min_raise_extra(v)
                for a, f in ((2, .5), (3, 1.)):
                    x = max(int(v.pot*f), minimum or stack)
                    if x > owed and judge.judge_accepts(v, x):
                        out[a] = x
                out[4] = stack
            else:
                for a, x in amounts.items():
                    if a not in (2, 3, 4):
                        continue
                    if int(x) != x:
                        raise ValueError('fractional executable chips')
                    x = int(x)
                    if x <= owed or x > stack or (a == 4 and x != stack):
                        raise ValueError('invalid executable amount')
                    if x == stack or judge.judge_accepts(v, x):
                        out[a] = x
                    else:
                        raise ValueError('executable amount fails judge minimum')
        if amounts is not None:
            out = {a: x for a, x in out.items() if a in amounts}
            for a in out:
                if int(amounts[a]) != out[a]:
                    raise ValueError('supplied call amount is not its actual chip cost')
        if not out:
            raise ValueError('empty executable action domain')
        return out

    def step(self, action, amount):
        if self.terminal:
            raise ValueError('terminal has no responses')
        child = copy.deepcopy(self)
        v, p = child.view, child.actor
        owed, before = child.owed, max(v.street_put)
        if action == 0:
            child.terminal, child.winner = 'fold', 1-p
            return child
        if action == 1 and amount != min(owed, v.chips[p]):
            raise ValueError('wrong call amount')
        if not 0 <= amount <= v.chips[p]:
            raise ValueError('invalid investment')
        kind = 'allin' if amount == v.chips[p] else 'call' if action == 1 else 'raise'
        judge._put(v, p, amount, kind)
        raised = v.street_put[p] > before
        child.pending = 1 if raised else child.pending-1
        if min(v.chips) == 0 and (not raised or v.chips[1-p] == 0):
            child.terminal = 'showdown'
        elif not raised and child.pending <= 0:
            child.terminal = 'round'
        else:
            child.actor = 1-p
            child.checked = not raised
        return child

    def settlement(self, equity0):
        v = self.view
        if self.terminal == 'fold':
            result = np.array(v.chips, dtype=float)
            result[self.winner] += v.pot
            return result
        if self.terminal not in ('showdown', 'round'):
            raise ValueError('cannot settle an active decision')
        refund = np.array([max(v.street_put[0]-v.street_put[1], 0),
                           max(v.street_put[1]-v.street_put[0], 0)], dtype=float)
        contested = v.pot - refund.sum()
        return np.asarray(v.chips) + refund + contested * np.array([equity0, 1.-equity0])


def betting_tree(state, terminal_builder, root_amounts=None, raise_cap=2, history=()):
    """All later decisions are public nodes with per-own-hand strategy rows."""
    if state.terminal:
        return terminal_builder(state, history)
    since = next((i+1 for i in range(len(history)-1,-1,-1) if history[i][0] == 'river'), 0)
    raises = sum(isinstance(a, int) and a >= 2 for a, _ in history[since:])
    actions = state.actions(root_amounts if not history else None, allow_raise=raises < raise_cap)
    children = {}
    for a, x in actions.items():
        children[a] = betting_tree(state.step(a, x), terminal_builder, None, raise_cap, history+((a, x),))
    return Node(state.actor, children, key=history)


def utility_coefficients(root, leaf, blind):
    """u0=a*equity0+b, including refunds and root centering."""
    v, r = leaf.view, root.view
    if leaf.terminal == 'fold':
        return 0., (v.chips[0] + (v.pot if leaf.winner == 0 else 0) - r.chips[0] - r.pot/2.)/blind
    refund0 = max(v.street_put[0]-v.street_put[1], 0)
    refund1 = max(v.street_put[1]-v.street_put[0], 0)
    return ((v.pot-refund0-refund1)/blind,
            (v.chips[0]+refund0-r.chips[0]-r.pot/2.)/blind)


def build_tree(state, blind, equity=None, kernel=None, root_amounts=None, raise_cap=2):
    def terminal(leaf, history):
        a, b = utility_coefficients(state, leaf, blind)
        return Terminal(coefficients=(a, b)) if kernel is not None else Terminal(a*equity+b)
    return betting_tree(state, terminal, root_amounts, raise_cap)

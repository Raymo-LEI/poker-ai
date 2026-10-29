"""Abstract action ids, plus an off-tree pot-fraction wager.

Integer decisions keep the historical ``table.step`` path. ``PotFractionBet``
is a chips-added target of ``fraction * pot`` (added on top of a call when
facing a bet). The HUNL table executes that size with ``step_raise_to`` when
the judge accepts it; otherwise the closest legal abstract wager is used.
"""
import math


class PotFractionBet(object):
    def __init__(self, fraction):
        fraction = float(fraction)
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("pot fraction must be positive and finite")
        self.fraction = fraction

    def __repr__(self):
        return "PotFractionBet(%s)" % self.fraction


def apply_actor_decision(table, decision):
    """Apply ``decision`` and return a JSON-safe record of what was used."""
    if isinstance(decision, bool):
        raise ValueError("invalid action")
    if isinstance(decision, PotFractionBet):
        return _apply_fraction(table, decision.fraction)
    if isinstance(decision, int):
        action = int(decision)
        if action not in table.legal_action_ids():
            raise ValueError("illegal action %s" % action)
        table.step(action)
        return action
    raise ValueError("unsupported action %r" % (decision,))


def _apply_fraction(table, fraction):
    actor = table.current_player()
    bet = int(table.state.bets[actor])
    stack = int(table.state.stacks[actor])
    pot = max(int(table.state.total_pot_amount), 1)
    owed = int(round(table.snapshot(actor).to_call))
    extra = int(round(float(fraction) * pot))
    desired = owed + extra if owed > 0 else extra
    desired = max(0, min(stack, desired))
    target = bet + desired
    if desired > owed and table.can_raise_to(target):
        table.step_raise_to(target)
        return {"kind": "pot_fraction", "fraction": float(fraction), "raise_to": int(target)}
    legal = table.legal_action_ids()
    ranked = []
    for action in (2, 3, 4, 1, 0):
        chips = table.action_chips(action)
        if action in legal and chips is not None:
            ranked.append((abs(int(chips) - desired), action))
    if not ranked:
        raise ValueError("no legal fallback for off-tree bet")
    action = sorted(ranked)[0][1]
    table.step(action)
    return action

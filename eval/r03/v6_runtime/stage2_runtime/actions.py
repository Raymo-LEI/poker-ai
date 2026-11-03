"""Configurable CFR action abstraction; the underlying poker rules stay full NL."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PreflopActionGuard:
    enabled: bool = True
    short_stack_bb: float = 20.0
    unlock_invested_bb: float = 15.0
    unlock_raises: int = 2

    @classmethod
    def from_config(cls, config):
        return cls(config.preflop_action_guard, config.guard_short_stack_bb,
                   config.guard_unlock_invested_bb, config.guard_unlock_raises)

    def apply(self, snapshot):
        if not self.enabled or snapshot.street != 0 or 4 not in snapshot.legal_action_ids:
            return snapshot
        blind = snapshot.big_blind
        invested = snapshot.starting_stack - min(snapshot.stacks)
        raises = sum(e.street == 0 and e.action_type in ("bet", "raise", "all_in")
                     for e in snapshot.events)
        if (min(snapshot.stacks) / blind <= self.short_stack_bb
                or invested / blind >= self.unlock_invested_bb
                or raises >= self.unlock_raises):
            return snapshot
        # Existing half-pot/pot raises remain; only the shallow deep-stack
        # shove branch is removed. Facing an off-tree large bet unlocks it.
        return replace(snapshot, legal_action_ids=tuple(a for a in snapshot.legal_action_ids if a != 4))


@dataclass(frozen=True)
class CFRActionGuard(PreflopActionGuard):
    """All-street SPR abstraction, with the original guard for old archives.

    Index 4 remains a true max bet. At matched commitments, effective
    remaining chips are min(stacks). Facing a bet, add the pending call to
    the opponent's remainder: their bet is already in the current pot.
    Otherwise a large off-tree bet could spuriously make SPR look small.
    """

    spr_enabled: bool = True
    max_pot_ratio: float = 3.0

    @classmethod
    def from_config(cls, config):
        return cls(config.preflop_action_guard, config.guard_short_stack_bb,
                   config.guard_unlock_invested_bb, config.guard_unlock_raises,
                   config.spr_action_guard, config.guard_max_pot_ratio)

    @staticmethod
    def all_in_pot_ratio(snapshot):
        effective = min(snapshot.me_stack, snapshot.opp_stack + snapshot.to_call)
        return effective / snapshot.pot if snapshot.pot > 0 else float("inf")

    @staticmethod
    def street_raises(snapshot):
        # All-in calls and checks must not unlock the raise branch.
        return sum(e.street == snapshot.street
                   and e.action_type in ("bet", "raise", "all_in")
                   and e.chips_added > e.to_call_before for e in snapshot.events)

    def blocks_all_in(self, snapshot):
        return (self.spr_enabled and 4 in snapshot.legal_action_ids
                and self.all_in_pot_ratio(snapshot) > self.max_pot_ratio
                and self.street_raises(snapshot) < self.unlock_raises)

    def apply(self, snapshot):
        if not self.spr_enabled:
            return super().apply(snapshot)
        if not self.blocks_all_in(snapshot):
            return snapshot
        return replace(snapshot, legal_action_ids=tuple(a for a in snapshot.legal_action_ids if a != 4))

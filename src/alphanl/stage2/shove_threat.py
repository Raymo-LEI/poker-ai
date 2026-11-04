"""Shared public-event predicate for legacy shoves and generalized overbets.

No stacks, event types, legal actions or neural HUD inputs are rewritten.
The compatibility keyword allow_near includes both >50-BB initiated overbets
and a bet/raise that crosses from above 5 BB to at most 5 BB remaining.
Already tiny residual stacks and passive Calls are not fresh threats.
"""


def shove_threat(event, big_blind=100., allow_near=False):
    if (event is None or event.actor_is_me
            or event.action_type not in ("bet", "raise", "all_in")
            or event.chips_added <= event.to_call_before):
        return False
    giant = event.chips_added > 50. * big_blind and event.chips_added > event.pot_before
    if not allow_near:
        return bool(giant and event.chips_added == event.stack_before)
    remaining = event.stack_before-event.chips_added
    near = event.stack_before > 5. * big_blind and 0 <= remaining <= 5. * big_blind
    return bool(giant or near)

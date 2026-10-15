"""Botzone TexasHoldem2p rules, kept apart from rlcard and PokerKit.

M1 is betting legality. M2 deals, rotates the button, and awards the pot.
The training loop is a later milestone.
"""

from .legal import (
    BIG_BLIND,
    SMALL_BLIND,
    STARTING_STACK,
    HandState,
    apply,
    legal_response,
    lower_bound_a,
    lower_bound_b,
    new_hand,
    safe_extra,
    to_call,
)
from .table import BotzoneTable

__all__ = [
    "BIG_BLIND",
    "SMALL_BLIND",
    "STARTING_STACK",
    "BotzoneTable",
    "HandState",
    "apply",
    "legal_response",
    "lower_bound_a",
    "lower_bound_b",
    "new_hand",
    "safe_extra",
    "to_call",
]

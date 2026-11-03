"""Botzone TexasHoldem2p rules, kept apart from rlcard and PokerKit.

M1 is betting legality. M2 deals, rotates the button, and awards the pot.
M3 packs the five policy ids through that legality. M7 is the weak rule
opponent on those ids. The branchable training kernel in ``kernel`` steps
the unmodified official referee. The training adapter that speaks
``SelfPlayEnv`` lives in ``alphanl.transformer.botzone_table`` and does not
change these rules.
"""

from .actions import (
    ACTION_NAMES,
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    NUM_ACTIONS,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    action_responses,
    legal_action_ids,
    response_for_action,
)
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
from .kernel import (
    POLICY_FEATURE_NAMES,
    BranchableKernel,
    KernelStep,
    LegacyFiveActionBridge,
)
from .rule_bot import BotzoneRuleBot
from .table import BotzoneTable

__all__ = [
    "ACTION_NAMES",
    "ALL_IN_ACTION",
    "BIG_BLIND",
    "CHECK_CALL_ACTION",
    "FOLD_ACTION",
    "NUM_ACTIONS",
    "RAISE_LARGE_ACTION",
    "RAISE_MIN_ACTION",
    "SMALL_BLIND",
    "STARTING_STACK",
    "BotzoneRuleBot",
    "BotzoneTable",
    "BranchableKernel",
    "HandState",
    "KernelStep",
    "LegacyFiveActionBridge",
    "POLICY_FEATURE_NAMES",
    "action_responses",
    "apply",
    "legal_action_ids",
    "legal_response",
    "lower_bound_a",
    "lower_bound_b",
    "new_hand",
    "response_for_action",
    "safe_extra",
    "to_call",
]

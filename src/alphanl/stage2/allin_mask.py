"""Pre-softmax voluntary all-in mask for Stage 2 and the Botzone export.

Call ``mask_snapshot`` on the guarded ``HandSnapshot`` before tokenization.
The resulting legal mask is what ``masked_log_probs`` and the NumPy runtime
turn into the illegal logit before softmax. Disabled configs return the
snapshot unchanged, so older runs stay reproducible.

SPR is ``CFRActionGuard.all_in_pot_ratio`` (effective stack / pot). The
all-in logit is cleared only when that ratio is strictly above ``spr_above``,
the hand is not premium, and continuing does not already require a call
that covers the stack. Premium hands are the user rule: preflop QQ+/AK or
equity >= 0.72; postflop a privately improved two pair or better
(``strong_made_hand``) or equity >= 0.80. Equity uses the fixed table/seed
estimator and does not draw from the match RNG.
"""
import math
from dataclasses import replace

from alphanl.cfr.actions import CFRActionGuard
from alphanl.transformer.equity import estimated_equity, preflop_key

from .deployment_guard import strong_made_hand

SETTINGS_KEY = "botzone_native_allin_mask"
PREMIUM_PREFLOP = frozenset(("AA", "KK", "QQ", "AKs", "AKo"))
ALL_IN = 4

DEFAULTS = {
    "enabled": False,
    "spr_above": 2.5,
    "preflop_equity": 0.72,
    "postflop_equity": 0.80,
    "equity_samples": 64,
    "equity_seed": 17,
}


def mask_config(config=None):
    """Validate an optional mask block. ``None`` is the disabled default."""
    result = dict(DEFAULTS)
    if config is None:
        return result
    if not isinstance(config, dict):
        raise ValueError("botzone_native_allin_mask must be an object")
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown all-in mask fields: " + str(sorted(unknown)))
    result.update(config)
    if not isinstance(result["enabled"], bool):
        raise ValueError("botzone_native_allin_mask.enabled must be boolean")
    for key in ("spr_above",):
        if not isinstance(result[key], (int, float)) or isinstance(result[key], bool):
            raise ValueError(key + " must be a positive number")
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    for key in ("preflop_equity", "postflop_equity"):
        if not isinstance(result[key], (int, float)) or isinstance(result[key], bool):
            raise ValueError(key + " must be a number in [0, 1]")
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(key + " must be in [0, 1]")
    for key, minimum in (("equity_samples", 1), ("equity_seed", 0)):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < minimum:
            raise ValueError(key + " must be an integer >= %d" % minimum)
    return result


def forced_call_allin(snapshot):
    """All-in is the call, not a voluntary shove.

    A covered stack (``me_stack <= to_call``) must still be allowed to
    continue. An all-in that is the only listed action is treated the same
    way so the mask cannot empty the legal set.
    """
    legal = tuple(snapshot.legal_action_ids)
    if ALL_IN not in legal:
        return False
    if float(snapshot.me_stack) <= float(snapshot.to_call):
        return True
    return not any(action != ALL_IN for action in legal)


def premium_hand(snapshot, config):
    """User premium rule. Categorical hits skip the equity estimator."""
    if snapshot.street == 0:
        if preflop_key(snapshot.hole_cards) in PREMIUM_PREFLOP:
            return True
        equity = estimated_equity(snapshot, samples=config["equity_samples"], seed=config["equity_seed"])
        return equity >= config["preflop_equity"]
    if snapshot.board_cards and strong_made_hand(snapshot):
        return True
    equity = estimated_equity(snapshot, samples=config["equity_samples"], seed=config["equity_seed"])
    return equity >= config["postflop_equity"]


def blocks_voluntary_allin(snapshot, config):
    """True when action 4 must be removed before the softmax."""
    config = mask_config(config)
    if not config["enabled"] or ALL_IN not in snapshot.legal_action_ids:
        return False
    if forced_call_allin(snapshot):
        return False
    if not CFRActionGuard.all_in_pot_ratio(snapshot) > config["spr_above"]:
        return False
    return not premium_hand(snapshot, config)


def mask_snapshot(snapshot, config):
    """Drop voluntary all-in from ``legal_action_ids`` when the mask is on."""
    if config is None:
        return snapshot
    config = mask_config(config)
    if not blocks_voluntary_allin(snapshot, config):
        return snapshot
    kept = tuple(action for action in snapshot.legal_action_ids if action != ALL_IN)
    if not kept:
        return snapshot
    return replace(snapshot, legal_action_ids=kept)

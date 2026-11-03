"""Visible-state-only v10 deployment intervention, shared by NumPy and eval.

This is a conservative short-match heuristic, not range equity or a training
action abstraction. No deck/opponent/policy RNG is consumed here. Python 3.6.
"""
import math
import numpy as np
from .portable_evaluator import strength
from .blind_lock import blind_budget
from .shove_threat import shove_threat


V10_GUARD = {
    "enabled": True,
    "prune_below": 0.10,
    "active_shove_min_bb": 30.0,
    "active_shove_pot_ratio": 2.0,
    "overbet_call_min_bb": 50.0,
    "overbet_call_pot_ratio": 1.2,
    "overbet_min_equity": 0.82,
    "require_made_hand": True,
    # Earlier 40BB confidence lock remains an opt-in ablation. v10 defaults
    # implement the latest requested overbet fuse without this extra gate.
    "big_call_enabled": False,
    "big_call_min_bb": 40.0,
    "big_call_min_probability": 0.35,
    "big_call_equity_margin": 0.05,
    "equity_samples": 128,
    "equity_seed": 17,
    "version": "v10",
    "passive_probe_enabled": False,
    "adaptive_anti_shove": False,
    "passive_min_actions": 4,
    "passive_min_checks": 2,
    "passive_max_aggression": 0.10,
    "passive_min_equity": 0.56,
    "passive_min_check_probability": 0.50,
    "passive_check_transfer": 0.50,
    "rare_preflop_equity": 0.77,
    "rare_flop_made_equity": 0.74,
    "rare_top_pair_equity": 0.80,
    "rare_river_equity": 0.80,
    "repeat_preflop_equity": 0.64,
    "repeat_postflop_equity": 0.66,
    "repeat_force_call_equity": 0.72,
    "frequent_force_call_equity": 0.60,
    "pot_control_enabled": False,
    "preflop_raise_cap_bb": 6.0,
    "preflop_control_call_max_bb": 10.0,
    "preflop_raise_equity": 0.65,
    "postflop_raise_equity": 0.74,
    "safe_shove_remap": False,
    "shove_value_min_equity": 0.65,
    "dominant_fold_enabled": False,
    "dominant_preflop_fold": 0.60,
    "dominant_preflop_tail": 0.10,
    "dominant_postflop_fold": 0.65,
    "dominant_postflop_tail": 0.20,
    "dominant_postflop_min_call_bb": 4.0,
    "dominant_fold_equity_margin": 0.03,
    "late_pot_control_enabled": False,
    "late_pot_min_bb": 12.0,
    "late_pot_equity": 0.80,
    "late_pot_max_bet_fraction": 0.50,
    # Revision 4 is opt-in so historical R3 metadata reconstructs exactly.
    "raise_redirect_enabled": False,
    "weak_raise_call_min_bb": 5.0,
    "weak_raise_call_min_odds": 0.28,
    "late_weak_call_enabled": False,
    "late_weak_call_min_bb": 10.0,
    "late_weak_call_min_odds": 0.33,
    "late_weak_call_equity": 0.72,
    "small_probe_defense_enabled": False,
    "small_probe_max_bb": 6.0,
    "small_probe_max_odds": 0.22,
    "small_probe_equity": 0.37,
    "small_probe_equity_max_bb": 4.0,
    "preflop_facing_raise_fold_enabled": False,
    "preflop_facing_raise_min_bb": 1.0,
    "preflop_facing_raise_equity": 0.50,
    "preflop_facing_raise_call": 0.30,
    # R5 A/B/C/D/E: absent metadata preserves the complete R4 policy.
    "victory_blind_lock_enabled": False,
    "preflop_weak_reraise_enabled": False,
    "weak_reraise_equity": 0.62,
    "weak_reraise_min_fold": 0.30,
    "weak_reraise_max_mass": 0.35,
    "weak_reraise_min_call_bb": 1.0,
    "flop_highcard_fold_enabled": False,
    "flop_highcard_min_call_bb": 4.0,
    "flop_highcard_min_odds": 0.28,
    "flop_highcard_max_equity": 0.55,
    "flop_highcard_min_fold": 0.65,
    "flop_highcard_max_call": 0.25,
    "strong_draw_min_outs": 8,
    "free_flop_weak_bluff_enabled": False,
    "free_flop_min_pot_bb": 8.0,
    "free_flop_max_equity": 0.50,
    "free_flop_min_check": 0.60,
    "free_flop_max_response_mass": 0.20,
    "strong_hand_price_exception_enabled": False,
    "strong_hand_min_equity": 0.62,
    "strong_hand_min_call_bb": 1.0,
    "strong_hand_max_call_bb": 3.0,
    "good_hand_blind_protection_enabled": False,
    "sb_good_hand_equity": 0.53,
    "bb_good_hand_equity": 0.56,
    "bb_good_hand_max_call_bb": 2.50,
    "bb_good_hand_max_odds": 0.36,
    # R5 G/H/I/J, independently switchable and off for old metadata.
    "effective_shove_enabled": False,
    "adaptive_call_protection_enabled": False,
    "micro_stack_commitment_enabled": False,
    "micro_min_pot_bb": 80.0,
    "micro_min_invested_bb": 30.0,
    "micro_max_call_bb": 2.0,
    "micro_max_pot_odds": 0.05,
    "micro_postflop_min_pot_bb": 100.0,
    "micro_postflop_max_call_bb": 5.0,
    "shove_runway_enabled": False,
    "shove_sb_discipline_enabled": False,
    "runway_min_shoves": 2,
    "runway_safe_slack_bb": 8.0,
    "runway_warning_slack_bb": 4.0,
    "runway_safe_equity": 0.65,
    "runway_warning_equity": 0.62,
    "runway_critical_equity": 0.54,
    "shove_sb_min_preflop_shoves": 3,
    "shove_sb_min_frequency": 0.50,
    "free_weak_tail_enabled": False,
    "free_tail_min_pot_bb": 2.0,
    "free_tail_preflop_equity": 0.50,
    "free_tail_max_equity": 0.55,
    "free_tail_min_check": 0.60,
    "free_tail_max_response_mass": 0.25,
    "station_control_enabled": False,
    "station_min_responses": 3,
    "station_min_actions": 8,
    "station_max_aggression": 0.25,
    "station_max_fold_rate": 0.25,
    "station_bluff_max_equity": 0.60,
    "station_bluff_min_check": 0.35,
    "station_thin_max_equity": 0.75,
    "station_thin_min_pot_bb": 4.0,
    "postflop_raise_gate_enabled": False,
    "raise_gate_min_call_bb": 4.0,
    "raise_gate_max_equity": 0.75,
    "raise_gate_min_fold": 0.50,
    "raise_gate_max_call": 0.30,
    # R6 strong-hand fold relief, independently switchable and off for old metadata.
    "strong_cap_no_fold_enabled": False,
    "strong_cap_min_equity": 0.70,
    "private_two_pair_strong_enabled": False,
    "strong_fold_floor_enabled": False,
    "strong_floor_min_equity": 0.70,
    "strong_floor_max_odds": 0.35,
    "strong_floor_max_pot_ratio": 1.0,
    "purify_threshold": 0.0,
}

V101_GUARD = dict(V10_GUARD, version="v10.1", prune_below=0.03,
                 passive_probe_enabled=True, adaptive_anti_shove=True)
V102_GUARD = dict(V101_GUARD, version="v10.2", pot_control_enabled=True,
                 safe_shove_remap=True, dominant_fold_enabled=True,
                 postflop_raise_equity=0.85, late_pot_control_enabled=True)

V102_R4_GUARD = dict(V102_GUARD, raise_redirect_enabled=True,
                     late_weak_call_enabled=True, small_probe_defense_enabled=True,
                     preflop_facing_raise_fold_enabled=True)

V102_R5_GUARD = dict(V102_R4_GUARD, victory_blind_lock_enabled=True,
                     preflop_weak_reraise_enabled=True, flop_highcard_fold_enabled=True,
                     free_flop_weak_bluff_enabled=True, strong_hand_price_exception_enabled=True,
                     good_hand_blind_protection_enabled=True,
                     effective_shove_enabled=True, adaptive_call_protection_enabled=True,
                     micro_stack_commitment_enabled=True, shove_runway_enabled=True,
                     shove_sb_discipline_enabled=True, free_weak_tail_enabled=True,
                     station_control_enabled=True, postflop_raise_gate_enabled=True)

V102_R6_GUARD = dict(V102_R5_GUARD, strong_cap_no_fold_enabled=True,
                     private_two_pair_strong_enabled=True, strong_fold_floor_enabled=True)


def guard_config(config=None):
    result = dict(V10_GUARD)
    # Absent metadata preserves the behavior of existing exports.
    if config is None:
        result["enabled"] = False
    else:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("unknown deployment guard fields: " + str(sorted(unknown)))
        result.update(config)
    if result["version"] not in ("v10", "v10.1", "v10.2"):
        raise ValueError("unsupported deployment guard version")
    for key in ("enabled", "require_made_hand", "big_call_enabled", "passive_probe_enabled", "adaptive_anti_shove",
                "pot_control_enabled", "safe_shove_remap", "dominant_fold_enabled", "late_pot_control_enabled",
                "raise_redirect_enabled", "late_weak_call_enabled", "small_probe_defense_enabled",
                "preflop_facing_raise_fold_enabled", "victory_blind_lock_enabled",
                "preflop_weak_reraise_enabled", "flop_highcard_fold_enabled",
                "free_flop_weak_bluff_enabled", "strong_hand_price_exception_enabled",
                "good_hand_blind_protection_enabled", "effective_shove_enabled",
                "adaptive_call_protection_enabled", "micro_stack_commitment_enabled",
                "shove_runway_enabled", "shove_sb_discipline_enabled",
                "free_weak_tail_enabled", "station_control_enabled", "postflop_raise_gate_enabled",
                "strong_cap_no_fold_enabled", "private_two_pair_strong_enabled", "strong_fold_floor_enabled"):
        if not isinstance(result[key], bool):
            raise ValueError(key + " must be boolean")
    for key in ("prune_below", "overbet_min_equity", "big_call_min_probability", "big_call_equity_margin",
                "passive_max_aggression", "passive_min_equity", "passive_min_check_probability", "passive_check_transfer",
                "rare_preflop_equity", "rare_flop_made_equity", "rare_top_pair_equity", "rare_river_equity",
                "repeat_preflop_equity", "repeat_postflop_equity", "repeat_force_call_equity", "frequent_force_call_equity",
                "preflop_raise_equity", "postflop_raise_equity", "shove_value_min_equity", "dominant_preflop_fold",
                "dominant_preflop_tail", "dominant_postflop_fold", "dominant_postflop_tail", "dominant_fold_equity_margin",
                "late_pot_equity", "late_pot_max_bet_fraction", "weak_raise_call_min_odds",
                "late_weak_call_min_odds", "late_weak_call_equity", "small_probe_max_odds",
                "small_probe_equity", "preflop_facing_raise_equity", "preflop_facing_raise_call", "weak_reraise_equity",
                "weak_reraise_min_fold", "weak_reraise_max_mass", "flop_highcard_min_odds",
                "flop_highcard_max_equity", "flop_highcard_min_fold", "flop_highcard_max_call",
                "free_flop_max_equity", "free_flop_min_check", "free_flop_max_response_mass",
                "micro_max_pot_odds", "runway_safe_equity", "runway_warning_equity",
                "runway_critical_equity", "shove_sb_min_frequency", "strong_hand_min_equity", "sb_good_hand_equity", "bb_good_hand_equity", "bb_good_hand_max_odds",
                "free_tail_preflop_equity", "free_tail_max_equity", "free_tail_min_check",
                "free_tail_max_response_mass", "station_max_fold_rate", "station_bluff_max_equity",
                "station_bluff_min_check", "station_thin_max_equity", "station_max_aggression", "raise_gate_max_equity",
                "raise_gate_min_fold", "raise_gate_max_call", "strong_cap_min_equity",
                "strong_floor_min_equity", "strong_floor_max_odds", "purify_threshold"):
        if not isinstance(result[key], (int, float)) or not 0 <= result[key] <= 1:
            raise ValueError(key + " must be in [0,1]")
    for key in ("active_shove_min_bb", "active_shove_pot_ratio", "overbet_call_min_bb",
                "overbet_call_pot_ratio", "big_call_min_bb", "preflop_raise_cap_bb", "preflop_control_call_max_bb",
                "dominant_postflop_min_call_bb", "late_pot_min_bb", "weak_raise_call_min_bb",
                "late_weak_call_min_bb", "small_probe_max_bb", "small_probe_equity_max_bb",
                "preflop_facing_raise_min_bb", "weak_reraise_min_call_bb", "flop_highcard_min_call_bb",
                "free_flop_min_pot_bb", "strong_hand_min_call_bb", "strong_hand_max_call_bb", "bb_good_hand_max_call_bb",
                "free_tail_min_pot_bb", "station_thin_min_pot_bb", "raise_gate_min_call_bb",
                "micro_min_pot_bb", "micro_min_invested_bb", "micro_max_call_bb",
                "micro_postflop_min_pot_bb", "micro_postflop_max_call_bb",
                "runway_safe_slack_bb", "runway_warning_slack_bb", "strong_floor_max_pot_ratio"):
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    for key in ("equity_samples", "equity_seed", "passive_min_actions", "passive_min_checks", "strong_draw_min_outs",
                "station_min_responses", "station_min_actions", "runway_min_shoves", "shove_sb_min_preflop_shoves"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < (1 if key == "equity_samples" else 0):
            raise ValueError(key + " must be a valid integer")
    if not 1 <= result["strong_draw_min_outs"] <= 47:
        raise ValueError("strong_draw_min_outs must be in [1,47]")
    if result["station_min_responses"] < 1:
        raise ValueError("station_min_responses must be positive")
    if result["strong_hand_min_call_bb"] >= result["strong_hand_max_call_bb"]:
        raise ValueError("strong-hand price window must be increasing")
    if result["runway_min_shoves"] < 2 or result["shove_sb_min_preflop_shoves"] < 3:
        raise ValueError("runway needs >=2 threats; SB discipline needs >=3 preflop threats")
    if result["runway_safe_slack_bb"] <= result["runway_warning_slack_bb"]:
        raise ValueError("runway slack thresholds must be descending")
    return result


def strong_made_hand(snapshot):
    """Two pair+ with private contribution; board-only strength is insufficient.

    Preflop has no made-hand category: the separate .82 equity gate applies.
    With a five-card board, kicker-only improvement to board trips/two pair
    does not qualify. This is deliberately stricter than random-hand equity.
    """
    board = snapshot.board_cards
    if not board:
        return True
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + board])
    if own[0] < 2:
        return False
    if len(board) < 5:
        # Board trips/quads alone are not a privately made monster.
        ranks = [c[1] for c in board]
        pairs = sorted(["23456789TJQKA".index(r) + 2 for r in set(ranks) if ranks.count(r) >= 2], reverse=True)
        if own[0] == 2 and len(pairs) == 2 and own[:3] <= (2, *pairs):
            return False
        if own[0] in (3, 7) and max(ranks.count(r) for r in ranks) >= (3 if own[0] == 3 else 4):
            return False
        return True
    public = strength([convert(c) for c in board])
    if own[0] != public[0]:
        return own[0] > public[0]
    primary = {2: 3, 3: 2, 7: 2}.get(own[0], len(own))
    return own[:primary] > public[:primary]


def pot_before_bet(snapshot):
    for event in reversed(snapshot.events):
        if (event.street == snapshot.street and not event.actor_is_me
                and event.chips_added > event.to_call_before):
            return max(float(event.pot_before), 1.0), "history"
    return max(float(snapshot.pot) - float(snapshot.to_call), 1.0), "pot_minus_call_fallback"


def _apply_v10_guard(snapshot, probabilities, config, equity_fn, amounts=None):
    """Return normalized probabilities and audit; confidence uses RAW policy.

    ``amounts`` optionally supplies exact legal judge chip investments. A pot
    raise whose minimum legal amount breaches the shove cap is not a safe
    remapping target. Any big-call fuse forbids ALL continuing actions.
    """
    raw = np.asarray(probabilities, dtype=np.float64).copy()
    if raw.shape != (5,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("invalid five-action probabilities")
    legal = set(snapshot.legal_action_ids)
    for action in range(5):
        if action not in legal:
            raw[action] = 0.0
    if raw.sum() <= 0:
        raise ValueError("empty legal policy")
    raw /= raw.sum()
    info = {"pruned": [], "active_cap": False, "call_fuse": None}
    if not config["enabled"]:
        return raw, info
    probs = raw.copy()
    for action in legal:
        if 0 < raw[action] < config["prune_below"]:
            probs[action] = 0.0
            info["pruned"].append(action)
    if probs.sum() <= 0:
        probs[int(np.argmax(raw))] = 1.0

    blind = float(snapshot.big_blind)
    call = min(float(snapshot.to_call), snapshot.me_stack)
    before, source = pot_before_bet(snapshot)
    giant = (call > config["overbet_call_min_bb"] * blind
             and call > config["overbet_call_pot_ratio"] * before)
    big = config["big_call_enabled"] and call > config["big_call_min_bb"] * blind
    if (giant or big) and 0 in legal:
        equity = float(equity_fn(snapshot, samples=config["equity_samples"], seed=config["equity_seed"]))
        if not math.isfinite(equity) or not 0 <= equity <= 1:
            raise ValueError("invalid estimated equity")
        odds = call / (snapshot.pot + call)
        made = strong_made_hand(snapshot)
        info.update(equity=equity, pot_odds=odds, strong_made_hand=made,
                    pot_before_bet=before, pot_before_source=source)
        if giant and (equity < config["overbet_min_equity"]
                      or (config["require_made_hand"] and not made)):
            info["call_fuse"] = "overbet_shove"
        elif big and (raw[1] < config["big_call_min_probability"]
                      or equity < odds + config["big_call_equity_margin"]):
            info["call_fuse"] = "big_call"
        if info["call_fuse"]:
            probs[:] = 0.0
            probs[0] = 1.0
            return probs, info

    amount = {1: call, 2: int(snapshot.pot * .5), 3: int(snapshot.pot), 4: snapshot.me_stack}
    if amounts is not None:
        amount.update(amounts)
    def excessive(chips):
        return (chips > config["active_shove_min_bb"] * blind
                and chips > config["active_shove_pot_ratio"] * snapshot.pot)
    if 4 in legal and snapshot.me_stack > snapshot.to_call and excessive(amount[4]) and probs[4] > 0:
        # Prefer the standard full-pot action, then half-pot. If a large
        # minimum raise makes both unsafe, keep only check/call or fold.
        targets = [a for a in (3, 2) if a in legal and not excessive(amount[a])]
        target = targets[0] if targets else 1 if 1 in legal else 0
        if target not in legal:
            raise ValueError("no safe legal shove remapping")
        info["active_cap"] = True
        info["remapped_to"] = target
        probs[target] += probs[4]
        probs[4] = 0.0
    probs /= probs.sum()
    return probs, info


def private_pair_or_better(snapshot):
    """Require a hole rank in the made pair, or a privately improved stronger hand."""
    if not snapshot.board_cards:
        return False
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + snapshot.board_cards])
    if own[0] >= 2:
        return strong_made_hand(snapshot)
    hole_ranks = ["23456789TJQKA".index(c[1]) + 2 for c in snapshot.hole_cards]
    return own[0] == 1 and own[1] in hole_ranks


def private_top_pair_or_overpair(snapshot):
    if not snapshot.board_cards or not private_pair_or_better(snapshot):
        return False
    ranks = lambda cards: ["23456789TJQKA".index(c[1]) + 2 for c in cards]
    hole, board = ranks(snapshot.hole_cards), ranks(snapshot.board_cards)
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + snapshot.board_cards])
    return own[0] == 1 and (own[1] == max(board) or (hole[0] == hole[1] and hole[0] > max(board)))


def runway_status(snapshot, context, config):
    """Configured public-score waiting heuristic; no opposing cards or future deals."""
    meta = getattr(context, "public_match", None)
    if not meta or snapshot.big_blind != 100.:
        return None
    opponent = 1-meta["me"]
    slack = blind_budget(meta, opponent) - meta["scores"][opponent]
    if slack > config["runway_safe_slack_bb"] * snapshot.big_blind:
        threshold, tier = config["runway_safe_equity"], "safe"
    elif slack > config["runway_warning_slack_bb"] * snapshot.big_blind:
        threshold, tier = config["runway_warning_equity"], "warning"
    else:
        # If an already lock-eligible opponent still shoves, waiting is no remedy.
        threshold, tier = config["runway_critical_equity"], "critical" if slack > 0 else "locked_active"
    return dict(runway_slack=slack, runway_threshold=threshold, runway_tier=tier)


def runway_hand_allowed(snapshot, equity, status):
    a,b = snapshot.hole_cards
    pair88 = a[1] == b[1] and a[1] in "89TJQKA"
    return equity >= status["runway_threshold"] or pair88


def adaptive_shove_rule(snapshot, count, equity, config, context=None):
    """The observed current shove is included in count; return allowed/force/tier."""
    if (config["shove_runway_enabled"] and snapshot.street == 0 and count >= config["runway_min_shoves"]):
        status = runway_status(snapshot, context, config)
        if status is not None:
            allowed = runway_hand_allowed(snapshot, equity, status)
            return allowed, allowed, "runway_" + status["runway_tier"]
    if count >= 4:
        allowed = equity >= config["frequent_force_call_equity"]
        return allowed, allowed, "4+"
    if count >= 2:
        if snapshot.street == 0:
            return equity >= config["repeat_preflop_equity"], False, "2-3"
        allowed = private_pair_or_better(snapshot) and equity >= config["repeat_postflop_equity"]
        return allowed, allowed and equity >= config["repeat_force_call_equity"], "2-3"
    if snapshot.street == 0:
        return equity >= config["rare_preflop_equity"], False, "1"
    made = strong_made_hand(snapshot)
    if snapshot.street == 3:
        return made and equity >= config["rare_river_equity"], False, "1"
    allowed = ((made and equity >= config["rare_flop_made_equity"])
               or (private_top_pair_or_overpair(snapshot) and equity >= config["rare_top_pair_equity"]))
    return allowed, False, "1"


def premium_preflop(snapshot):
    a, b = snapshot.hole_cards
    ranks = {a[1], b[1]}
    return ((a[1] == b[1] and a[1] in "QKA") or ranks == {"A", "K"}
            or (ranks == {"A", "Q"} and a[0] == b[0]))


def genuine_two_pair_or_better(snapshot):
    """Exclude public pairs plus a single private pair from true two pair."""
    if not snapshot.board_cards or not strong_made_hand(snapshot):
        return False
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + snapshot.board_cards])
    if own[0] != 2:
        return True
    private = {"23456789TJQKA".index(c[1]) + 2 for c in snapshot.hole_cards}
    return own[1] in private and own[2] in private


def top_pair_or_better(snapshot):
    """Private top pair/overpair includes top pair on a publicly paired board."""
    if genuine_two_pair_or_better(snapshot):
        return True
    if not snapshot.board_cards:
        return False
    ranks = lambda cards: ["23456789TJQKA".index(c[1]) + 2 for c in cards]
    hole, board = ranks(snapshot.hole_cards), ranks(snapshot.board_cards)
    top = max(board)
    return ((hole[0] == hole[1] and hole[0] >= top)
            or any(rank == top and board.count(rank) == 1 for rank in hole))


def private_top_two_pair(snapshot):
    """Two pair from a private top pair/overpair plus a public board pair.

    Top pair is measured against the highest unpaired board rank; a pocket
    pair must be above every board card.
    """
    if not snapshot.board_cards:
        return False
    convert = lambda c: c[1] + c[0].lower()
    own = strength([convert(c) for c in snapshot.hole_cards + snapshot.board_cards])
    if own[0] != 2:
        return False
    ranks = lambda cards: ["23456789TJQKA".index(c[1]) + 2 for c in cards]
    hole, board = ranks(snapshot.hole_cards), ranks(snapshot.board_cards)
    if hole[0] == hole[1]:
        return hole[0] > max(board) and hole[0] in own[1:3]
    unpaired = [rank for rank in board if board.count(rank) == 1]
    return bool(unpaired) and any(rank == max(unpaired) and rank in own[1:3] for rank in hole)


def _redirect_raise_mass(snapshot, raw, probs, mass, target, config, info, protected):
    """Apply the same controlled routing to regular raises and intercepted shoves."""
    if config["raise_redirect_enabled"] and snapshot.street > 0 and snapshot.to_call > 0 and target == 1:
        call = min(float(snapshot.to_call), snapshot.me_stack)
        odds = call / (snapshot.pot + call)
        if (not top_pair_or_better(snapshot)
                and (call >= config["weak_raise_call_min_bb"] * snapshot.big_blind
                     or odds >= config["weak_raise_call_min_odds"])):
            target = 0
            info["raise_redirect"] = "weak_large_bet_fold"
        elif raw[0] > raw[1] and 0 in snapshot.legal_action_ids:
            fraction = float(raw[0] / (raw[0] + raw[1]))
            probs[0] += mass * fraction
            probs[1] += mass * (1. - fraction)
            protected.update((0, 1))
            info["raise_redirect"] = "raw_fold_call_ratio"
            return "split"
    if target not in snapshot.legal_action_ids:
        raise ValueError("no legal controlled raise fallback")
    probs[target] += mass
    protected.add(target)
    return target


def _r4_continuation_guards(snapshot, raw, probs, legal, config, equity, info, protected):
    """Final visible-state gates; run after transfers, before pruning."""
    call = min(float(snapshot.to_call), snapshot.me_stack)
    blind = float(snapshot.big_blind)
    if 0 not in legal or call <= 0 or info.get("adaptive_call_protected"):
        return
    reason = None
    if (config["preflop_facing_raise_fold_enabled"] and snapshot.street == 0
            and snapshot.to_call > config["preflop_facing_raise_min_bb"] * blind
            and raw[0] >= config["dominant_preflop_fold"]
            and (raw[1] < config["preflop_facing_raise_call"]
                 or equity() < config["preflop_facing_raise_equity"])):
        if good_hand_blind_side(snapshot, config, equity) == "bb":
            info["r5_blind_price_exception"] = True
        elif (config["strong_hand_price_exception_enabled"]
                and config["strong_hand_min_call_bb"] * blind < snapshot.to_call
                <= config["strong_hand_max_call_bb"] * blind
                and equity() >= config["strong_hand_min_equity"]):
            info["r5_strong_price_exception"] = True
        else:
            reason = "preflop_facing_raise"
    odds = call / (snapshot.pot + call)
    if config["late_weak_call_enabled"] and snapshot.street >= 2:
        before, source = pot_before_bet(snapshot)
        large = (call >= config["late_weak_call_min_bb"] * blind
                 or odds >= config["late_weak_call_min_odds"])
        # snapshot.pot already includes the opponent's bet. Compare against
        # the public pot BEFORE that bet to recognize an actual overbet.
        made = genuine_two_pair_or_better(snapshot)
        if not made and config["private_two_pair_strong_enabled"] and private_top_two_pair(snapshot):
            made = True
            if call > before:
                info["r6_private_two_pair"] = True
        if ((large and not top_pair_or_better(snapshot) and equity() < config["late_weak_call_equity"])
                or (not made and call > before)):
            reason = "late_weak_overbet" if call > before else "late_weak_large_call"
            info.update(pot_before_bet=before, pot_before_source=source)
    if reason:
        probs[:] = 0.; probs[0] = 1.
        protected.clear(); protected.add(0)
        info.update(continuation_fold=reason, pot_odds=odds)
        return
    if (config["small_probe_defense_enabled"] and snapshot.street in (1, 2) and 1 in legal
            and call <= config["small_probe_max_bb"] * blind
            and odds <= config["small_probe_max_odds"]
            and (private_pair_or_better(snapshot)
                 or (call <= config["small_probe_equity_max_bb"] * blind
                     and equity() >= config["small_probe_equity"]))):
        mass = float(probs[0])
        probs[0] = 0.; probs[1] += mass
        protected.discard(0); protected.add(1)
        info.update(small_probe_defense=True, small_probe_mass=mass, pot_odds=odds)


def good_hand_blind_side(snapshot, config, equity):
    """F: public unopened-SB / first BB response to exactly one SB open."""
    if not config["good_hand_blind_protection_enabled"] or snapshot.street != 0:
        return None
    blind = float(snapshot.big_blind)
    if (snapshot.me_seat == snapshot.sb_seat and not snapshot.events
            and snapshot.to_call == .5 * blind and equity() >= config["sb_good_hand_equity"]):
        return "sb"
    if (snapshot.me_seat != snapshot.sb_seat and len(snapshot.events) == 1
            and blind < snapshot.to_call <= config["bb_good_hand_max_call_bb"] * blind):
        event = snapshot.events[0]
        call = min(float(snapshot.to_call), snapshot.me_stack)
        if (not event.actor_is_me and event.chips_added > event.to_call_before
                and event.action_type in ("raise", "bet", "all_in")
                and call / (snapshot.pot + call) <= config["bb_good_hand_max_odds"]
                and equity() >= config["bb_good_hand_equity"]):
            return "bb"
    return None


def _r5_blind_protection(snapshot, probs, legal, config, equity, info, protected):
    side = good_hand_blind_side(snapshot, config, equity)
    if side is None or 1 not in legal or 0 not in legal:
        return
    mass = float(probs[0]); probs[0] = 0.; probs[1] += mass
    protected.discard(0); protected.add(1)
    info.update(r5_blind_protection=side, r5_blind_fold_mass=mass)


def straight_flush_outs(snapshot):
    """Unique unseen next cards making a straight/flush, using only own+board.

    Includes straight-flush overlap once; excludes full houses and quads.
    No opposing cards, observed future cards or sampling RNG are consulted.
    """
    if snapshot.street not in (1, 2):
        return 0
    known = [c[1] + c[0].lower() for c in snapshot.hole_cards + snapshot.board_cards]
    return sum(strength(known + [rank + suit])[0] in (4, 5, 8)
               for rank in "23456789TJQKA" for suit in "hdsc" if rank + suit not in known)


def _r5_weak_reraise(snapshot, raw, probs, legal, amount, config, equity, info, protected):
    # Before shove/pot-control transfers: intercepted aggression must not enter Call.
    if (not config["preflop_weak_reraise_enabled"] or snapshot.street != 0 or 0 not in legal
            or snapshot.to_call <= config["weak_reraise_min_call_bb"] * snapshot.big_blind
            or premium_preflop(snapshot) or raw[0] < config["weak_reraise_min_fold"]):
        return
    call = min(float(snapshot.to_call), snapshot.me_stack)
    raises = [a for a in (2, 3, 4) if a in legal and amount[a] > call]
    mass = float(sum(raw[a] for a in raises))
    if (not raises or mass >= config["weak_reraise_max_mass"]
            or equity() >= config["weak_reraise_equity"]):
        return
    moved = float(sum(probs[a] for a in raises))
    for action in raises:
        probs[action] = 0.; protected.discard(action)
    probs[0] += moved; protected.add(0)
    info.update(r5_weak_reraise=True, r5_reraise_raw_mass=mass, r5_reraise_fold_mass=moved)


def _r5_final_guards(snapshot, raw, probs, legal, amount, config, equity, info, protected):
    if info.get("adaptive_call_protected") or snapshot.street != 1 or private_pair_or_better(snapshot):
        return
    call = min(float(snapshot.to_call), snapshot.me_stack)
    odds = call / (snapshot.pot + call) if call else 0.
    if (config["flop_highcard_fold_enabled"] and call > 0 and 0 in legal
            and (call >= config["flop_highcard_min_call_bb"] * snapshot.big_blind
                 or odds >= config["flop_highcard_min_odds"])
            and raw[0] >= config["flop_highcard_min_fold"]
            and raw[1] < config["flop_highcard_max_call"]
            and equity() < config["flop_highcard_max_equity"]):
        outs = straight_flush_outs(snapshot)
        if outs < config["strong_draw_min_outs"]:
            probs[:] = 0.; probs[0] = 1.; protected.clear(); protected.add(0)
            info.update(r5_highcard_fold=True, r5_draw_outs=outs, pot_odds=odds)
            return
    if (not config["free_flop_weak_bluff_enabled"] or snapshot.to_call != 0 or 1 not in legal
            or snapshot.pot < config["free_flop_min_pot_bb"] * snapshot.big_blind
            or raw[1] < config["free_flop_min_check"] or equity() >= config["free_flop_max_equity"]):
        return
    bets = [a for a in (2, 3, 4) if a in legal and amount[a] > 0]
    responses = {}
    for action in bets:
        responses[amount[action]] = responses.get(amount[action], 0.) + raw[action]
    if not responses or any(mass >= config["free_flop_max_response_mass"] for mass in responses.values()):
        return
    outs = straight_flush_outs(snapshot)
    if outs >= config["strong_draw_min_outs"]:
        return
    mass = float(sum(probs[a] for a in bets))
    for action in bets:
        probs[action] = 0.; protected.discard(action)
    probs[1] += mass; protected.add(1)
    info.update(r5_weak_bluff_check=True, r5_check_mass=mass, r5_draw_outs=outs)


def _r5_supplement(snapshot, raw, probs, legal, amount, config, equity, info, protected, context):
    """Seven-loss controls: supported tails, public station evidence and raises."""
    if info.get("adaptive_call_protected"):
        return
    if not any(config[k] for k in ("free_weak_tail_enabled", "station_control_enabled", "postflop_raise_gate_enabled")):
        return
    blind = float(snapshot.big_blind)
    call = min(float(snapshot.to_call), snapshot.me_stack)
    if (config["postflop_raise_gate_enabled"] and snapshot.street > 0 and call > 0
            and 0 in legal and call >= config["raise_gate_min_call_bb"] * blind
            and not top_pair_or_better(snapshot)
            and raw[0] >= config["raise_gate_min_fold"] and raw[1] < config["raise_gate_max_call"]
            and equity() < config["raise_gate_max_equity"] and snapshot.events):
        event = snapshot.events[-1]
        raised = (event.street == snapshot.street and not event.actor_is_me
                  and event.to_call_before > 0 and event.chips_added > event.to_call_before
                  and any(e.street == snapshot.street and e.actor_is_me
                          and e.chips_added > e.to_call_before for e in snapshot.events[:-1]))
        if raised and straight_flush_outs(snapshot) < config["strong_draw_min_outs"]:
            probs[:] = 0.; probs[0] = 1.; protected.clear(); protected.add(0)
            info.update(r5_raise_gate=True, r5_raise_gate_reason="weak_response_to_raise")
            return
    if snapshot.to_call != 0 or 1 not in legal:
        return
    bets = [a for a in (2, 3, 4) if a in legal and amount[a] > 0]
    if not bets:
        return
    private_pair = private_pair_or_better(snapshot) if snapshot.street else False
    cached_outs = []
    def draw_outs():
        if not cached_outs:
            cached_outs.append(straight_flush_outs(snapshot))
        return cached_outs[0]
    blocked = set()
    preflop_limp = (snapshot.street == 0 and snapshot.me_seat != snapshot.sb_seat
                    and len(snapshot.events) == 1 and not snapshot.events[0].actor_is_me
                    and snapshot.events[0].action_type == "call"
                    and snapshot.events[0].chips_added == .5 * blind
                    and snapshot.events[0].to_call_before == .5 * blind)
    responses = {}
    for action in bets:
        responses[amount[action]] = responses.get(amount[action], 0.) + raw[action]
    if (config["free_weak_tail_enabled"] and (snapshot.street > 0 or preflop_limp)
            and snapshot.pot >= config["free_tail_min_pot_bb"] * blind and not private_pair
            and raw[1] >= config["free_tail_min_check"]
            and equity() < config["free_tail_max_equity" if snapshot.street else "free_tail_preflop_equity"]
            and all(mass < config["free_tail_max_response_mass"] for mass in responses.values())
            and draw_outs() < config["strong_draw_min_outs"]):
        blocked.update(bets)
        info["r5_free_weak_tail"] = True
    opportunities = getattr(context, "opp_postflop_bet_responses", 0)
    folds = getattr(context, "opp_postflop_bet_folds", 0)
    actions = getattr(context, "opp_regular_postflop_actions", 0)
    aggression = getattr(context, "opp_regular_postflop_bets", 0)
    station = (config["station_control_enabled"] and snapshot.street > 0
               and getattr(context, "station_evidence_complete", True)
               and actions >= config["station_min_actions"]
               and aggression / max(actions, 1) <= config["station_max_aggression"]
               and opportunities >= config["station_min_responses"]
               and folds / max(opportunities, 1) <= config["station_max_fold_rate"])
    if station and draw_outs() < config["strong_draw_min_outs"]:
        if (not private_pair and raw[1] >= config["station_bluff_min_check"]
                and equity() < config["station_bluff_max_equity"]):
            blocked.update(bets)
            info["r5_station_bluff_check"] = True
        elif (snapshot.street >= 2 and private_pair and not top_pair_or_better(snapshot)
                and snapshot.pot >= config["station_thin_min_pot_bb"] * blind
                and equity() < config["station_thin_max_equity"]):
            blocked.update(a for a in bets if amount[a] >= snapshot.pot)
            if blocked:
                info["r5_station_thin_cap"] = True
        if blocked:
            info.update(r5_station_responses=opportunities, r5_station_folds=folds,
                        r5_station_actions=actions, r5_station_aggression=aggression/max(actions,1))
    if blocked:
        mass = float(sum(probs[a] for a in blocked))
        for action in blocked:
            probs[action] = 0.; protected.discard(action)
        probs[1] += mass; protected.add(1)
        info.update(r5_supplement_blocked=sorted(blocked), r5_supplement_check_mass=mass,
                    r5_supplement_draw_outs=draw_outs())


def _pot_control(snapshot, raw, probs, legal, amount, config, equity, info, protected):
    """Run after all aggression transfers, before pruning; never revive raises."""
    if not config["pot_control_enabled"]:
        return
    blind = float(snapshot.big_blind)
    call = min(float(snapshot.to_call), snapshot.me_stack)
    invested = max(0., snapshot.starting_stack - snapshot.me_stack)
    raises = [a for a in (2, 3, 4) if a in legal and amount[a] > call]
    if snapshot.street == 0:
        raised = invested > blind or invested + snapshot.to_call > blind
        candidates = [a for a in raises if invested + amount[a] > config["preflop_raise_cap_bb"] * blind]
        if not raised or premium_preflop(snapshot) or not any(probs[a] > 0 for a in candidates):
            return
        if equity() >= config["preflop_raise_equity"]:
            return
        reason = "preflop_non_premium_raise_cap"
        call_size_ok = snapshot.to_call <= config["preflop_control_call_max_bb"] * blind
    else:
        candidates = raises
        if snapshot.to_call <= 0 or not any(probs[a] > 0 for a in candidates):
            return
        if genuine_two_pair_or_better(snapshot) or equity() >= config["postflop_raise_equity"]:
            return
        reason = "postflop_non_nut_raise"
        call_size_ok = True
    odds = call / (snapshot.pot + call) if call else 0.
    target = 1 if 1 in legal and call_size_ok and equity() >= odds else 0
    if target not in legal:
        raise ValueError("no safe legal pot-control fallback")
    blocked = [a for a in candidates if probs[a] > 0]
    mass = float(sum(probs[a] for a in blocked))
    for action in blocked:
        probs[action] = 0.
        protected.discard(action)
    target = _redirect_raise_mass(snapshot, raw, probs, mass, target, config, info, protected)
    info.update(pot_control=reason, pot_control_blocked=blocked, pot_control_to=target,
                pot_control_mass=mass, pot_control_invested=invested, pot_odds=odds)


def _late_pot_control(snapshot, probs, legal, amount, config, equity, info, protected):
    """Check back intercepted large bets; legal min-raises cannot evade half-pot."""
    if (not config["late_pot_control_enabled"] or snapshot.street < 2 or snapshot.to_call != 0
            or snapshot.pot < config["late_pot_min_bb"] * snapshot.big_blind):
        return
    cap = config["late_pot_max_bet_fraction"] * snapshot.pot
    candidates = [a for a in (2,3,4) if a in legal and probs[a] > 0
                  and (a in (3,4) or amount[a] > cap or amount[a] >= snapshot.me_stack)]
    if not candidates or genuine_two_pair_or_better(snapshot) or equity() >= config["late_pot_equity"]:
        return
    if 1 not in legal:
        raise ValueError("no legal check for late-street pot control")
    mass = float(sum(probs[a] for a in candidates))
    for action in candidates:
        probs[action] = 0.; protected.discard(action)
    probs[1] += mass; protected.add(1)
    info.update(late_pot_control=True, late_pot_blocked=candidates, late_pot_to=1,
                late_pot_mass=mass, late_pot_max_chips=cap)


def _safe_shove_transfer(snapshot, raw, probs, legal, amount, excessive, config, equity, info, protected):
    """Facing action never receives new raise mass from an intercepted shove."""
    mass = float(probs[4]); probs[4] = 0.
    call = min(float(snapshot.to_call), snapshot.me_stack)
    odds = call / (snapshot.pot + call) if call else 0.
    info.update(active_cap=True, remapped_mass=mass)
    if snapshot.to_call > 0:
        if (config["strong_cap_no_fold_enabled"] and snapshot.street > 0
                and (equity() >= config["strong_cap_min_equity"] or top_pair_or_better(snapshot))):
            targets = [a for a in (1, 3, 2) if a in legal and (a == 1 or (amount[a] > call and not excessive(amount[a])))]
            if targets:
                target = targets[0]
                probs[target] += mass; protected.add(target)
                info.update(remapped_to=target, shove_route="strong_no_fold", r6_strong_cap=True, pot_odds=odds)
                return
        affordable = snapshot.street != 0 or snapshot.to_call <= config["preflop_control_call_max_bb"] * snapshot.big_blind
        target = 1 if 1 in legal and affordable and equity() >= odds else 0
        if target not in legal:
            raise ValueError("no legal non-raising shove fallback")
        target = _redirect_raise_mass(snapshot, raw, probs, mass, target, config, info, protected)
        info.update(remapped_to=target, shove_route="facing_split" if target == "split" else "facing_call" if target == 1 else "facing_fold", pot_odds=odds)
        return
    value = (equity() >= config["shove_value_min_equity"] and
             (snapshot.street == 0 or genuine_two_pair_or_better(snapshot) or private_top_pair_or_overpair(snapshot)))
    targets = [a for a in (2,3) if a in legal and amount[a] > call and not excessive(amount[a])]
    if value and targets:
        target = targets[0]
        probs[target] += mass; protected.add(target)
        info.update(remapped_to=target, shove_route="free_value")
    else:
        # Removing only the shove preserves every other action's relative mass.
        if probs.sum() <= 0:
            target = 1 if 1 in legal else 0
            if target not in legal:raise ValueError("no legal free-shove fallback")
            probs[target] = 1.; protected.add(target)
        probs /= probs.sum()
        info["shove_route"] = "free_renormalize"
        if not targets:info["no_safe_bet_target"] = True


def _r6_strong_fold_floor(snapshot, probs, legal, config, equity, info, protected):
    """Postflop non-shove bet at most one pot: strong equity never folds."""
    if (not config["strong_fold_floor_enabled"] or snapshot.street == 0 or 0 not in legal
            or 1 not in legal or probs[0] <= 0 or snapshot.to_call <= 0):
        return
    call = min(float(snapshot.to_call), snapshot.me_stack)
    odds = call / (snapshot.pot + call)
    before, _ = pot_before_bet(snapshot)
    current = next((event for event in reversed(snapshot.events)
                    if event.street == snapshot.street and not event.actor_is_me
                    and event.chips_added > event.to_call_before), None)
    if (odds > config["strong_floor_max_odds"] or call > config["strong_floor_max_pot_ratio"] * before
            or shove_threat(current, float(snapshot.big_blind), allow_near=True)
            or equity() < config["strong_floor_min_equity"]):
        return
    mass = float(probs[0]); probs[0] = 0.; probs[1] += mass
    protected.discard(0); protected.add(1)
    info.update(r6_strong_fold_floor=True, r6_floor_fold_mass=mass, pot_odds=odds)


def _r6_purify(snapshot, probs, legal, config, info, protected):
    """Postflop: zero unprotected actions below tau of the current total."""
    tau = config["purify_threshold"]
    total = float(probs.sum())
    if tau <= 0 or snapshot.street == 0 or total <= 0:
        return
    removed = [a for a in sorted(legal) if a not in protected and 0 < probs[a] < tau * total]
    if removed and sum(probs[a] for a in legal if a not in removed) > 0:
        for action in removed:
            probs[action] = 0.
        info["r6_purified"] = removed


def _dominant_fold_filter(snapshot, raw, probs, legal, amount, config, equity, info, protected):
    """Use original legal policy confidence, never inflated remapping probabilities."""
    if (not config["dominant_fold_enabled"] or 0 not in legal
            or (snapshot.street > 0 and info.get("adaptive_call_protected"))):
        return
    # Minimum-raise adjustment can make two abstract slots the SAME response.
    # Its physical-action probability is their sum, not either slot alone.
    def below(action, threshold):
        return sum(raw[a] for a in legal if a != 0 and amount[a] == amount[action]) < threshold
    if snapshot.street == 0 and raw[0] >= config["dominant_preflop_fold"]:
        blocked = [a for a in (2,3,4) if a in legal and below(a,config["dominant_preflop_tail"]) and probs[a] > 0]
        reason = "preflop_raise_tail"
    elif (snapshot.street > 0 and raw[0] >= config["dominant_postflop_fold"]
          and snapshot.to_call >= config["dominant_postflop_min_call_bb"] * snapshot.big_blind):
        call = min(float(snapshot.to_call), snapshot.me_stack)
        odds = call / (snapshot.pot + call)
        if equity() >= odds + config["dominant_fold_equity_margin"]:
            return
        blocked = [a for a in (1,2,3,4) if a in legal and below(a,config["dominant_postflop_tail"]) and probs[a] > 0]
        reason = "postflop_weak_tail"
        info["pot_odds"] = odds
    else:
        return
    if blocked:
        for action in blocked:
            probs[action] = 0.; protected.discard(action)
        protected.add(0)
        info.update(dominant_fold=reason, dominant_fold_blocked=blocked,
                    dominant_fold_forced=bool(sum(probs[a] for a in (1,2,3,4)) == 0))


def apply_guard(snapshot, probabilities, config, equity_fn, amounts=None, context=None):
    """Keep legacy v10 exact; adaptive guards remap BEFORE tail pruning."""
    if config["version"] not in ("v10.1", "v10.2") or not config["enabled"]:
        return _apply_v10_guard(snapshot, probabilities, config, equity_fn, amounts)
    raw = np.asarray(probabilities, dtype=np.float64).copy()
    legal = set(snapshot.legal_action_ids)
    if raw.shape != (5,) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("invalid five-action probabilities")
    for action in range(5):
        if action not in legal:
            raw[action] = 0.
    if raw.sum() <= 0:
        raise ValueError("empty legal policy")
    raw /= raw.sum()
    probs = raw.copy()
    info = {"pruned": [], "active_cap": False, "call_fuse": None, "passive_probe": False,
            "forced_call": False, "opp_shove_count": getattr(context, "opp_shove_count", 0)}
    if config["effective_shove_enabled"]:
        info["opp_shove_count"] = max(getattr(context, "opp_effective_shove_count", 0), info["opp_shove_count"])
    cached = []
    def equity():
        if not cached:
            value = float(equity_fn(snapshot, samples=config["equity_samples"], seed=config["equity_seed"]))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid estimated equity")
            cached.append(value)
            info["equity"] = value
        return cached[0]
    blind = float(snapshot.big_blind)
    call = min(float(snapshot.to_call), snapshot.me_stack)
    odds = call / (snapshot.pot + call) if call else 0.
    invested = max(0., snapshot.starting_stack-snapshot.me_stack)
    postflop_tail = (snapshot.street > 0 and snapshot.pot >= config["micro_postflop_min_pot_bb"] * blind
                     and call <= config["micro_postflop_max_call_bb"] * blind)
    if (config["micro_stack_commitment_enabled"] and call > 0 and 1 in legal
            and invested >= config["micro_min_invested_bb"] * blind
            and (postflop_tail or (snapshot.pot >= config["micro_min_pot_bb"] * blind
                 and (call <= config["micro_max_call_bb"] * blind or odds <= config["micro_max_pot_odds"])
                 and equity() > odds))):
        probs[:] = 0.; probs[1] = 1.
        info.update(micro_commitment=True, micro_postflop_tail=postflop_tail,
                    pot_odds=odds, micro_invested_bb=invested/blind)
        return probs, info
    status = runway_status(snapshot, context, config) if config["shove_runway_enabled"] else None
    if status is not None and snapshot.street == 0:
        info.update(status)
    pre_shoves = getattr(context, "opp_preflop_effective_shove_count", 0)
    pre_actions = getattr(context, "opp_preflop_action_opportunities", 0)
    if (config["shove_sb_discipline_enabled"] and status is not None and snapshot.street == 0
            and snapshot.me_seat == snapshot.sb_seat and not snapshot.events and call == .5*blind
            and pre_shoves >= config["shove_sb_min_preflop_shoves"]
            and pre_shoves / max(pre_actions, 1) > config["shove_sb_min_frequency"]
            and not runway_hand_allowed(snapshot, equity(), status)):
        probs[:] = 0.; probs[0] = 1.
        info.update(shove_sb_discipline=True, preflop_shove_frequency=pre_shoves/max(pre_actions,1))
        return probs, info
    before, source = pot_before_bet(snapshot)
    giant = call > config["overbet_call_min_bb"] * blind and call > config["overbet_call_pot_ratio"] * before
    # Count/grade initiated >50BB shoves above 1x the pot, including the
    # 1.0..1.2x interval that the legacy v10 fuse did not cover.
    current_shove = next((event for event in reversed(snapshot.events)
                          if event.street == snapshot.street and not event.actor_is_me
                          and event.chips_added > event.to_call_before), None)
    if config["effective_shove_enabled"]:
        # Neural HUD fields and the legacy count stay unchanged. Only the
        # deployment decision reads the additional public-event counter.
        info["opp_shove_count"] = max(getattr(context, "opp_effective_shove_count", 0), info["opp_shove_count"])
        info["effective_shove"] = shove_threat(current_shove, blind, allow_near=True)
    adaptive = (config["adaptive_anti_shove"] and snapshot.opp_stack == 0 and call > 0
                and current_shove is not None and current_shove.action_type == "all_in"
                and current_shove.chips_added > 50. * blind
                and current_shove.chips_added > current_shove.pot_before)
    if config["effective_shove_enabled"] and config["adaptive_anti_shove"]:
        adaptive = call > 0 and info["effective_shove"]
    big = config["big_call_enabled"] and call > config["big_call_min_bb"] * blind
    protected = set()
    if (giant or adaptive or big) and 0 in legal:
        value = equity()
        info.update(pot_odds=call / (snapshot.pot + call), strong_made_hand=strong_made_hand(snapshot),
                    pot_before_bet=before, pot_before_source=source)
        if giant or adaptive:
            if adaptive:
                allowed, force, tier = adaptive_shove_rule(snapshot, info["opp_shove_count"], value, config, context)
                info["anti_shove_tier"] = tier
                if allowed and 1 in legal:
                    if config["adaptive_call_protection_enabled"]:
                        info["adaptive_call_protected"] = True
                    protected.add(1)  # An unlocked call must not be relocked by tail pruning.
                if force and 1 in legal:
                    probs[:] = 0.; probs[1] = 1.; info["forced_call"] = True
                    if config["adaptive_call_protection_enabled"]:
                        return probs, info
                    _r4_continuation_guards(snapshot, raw, probs, legal, config, equity, info, protected)
                    _r5_final_guards(snapshot, raw, probs, legal, {1:call}, config, equity, info, protected)
                    if probs[0] == 1.:
                        info["forced_call"] = False
                    return probs, info
            else:
                allowed = (value >= config["overbet_min_equity"]
                           and (not config["require_made_hand"] or strong_made_hand(snapshot)))
            if not allowed:
                info["call_fuse"] = "overbet_shove"
        if not info["call_fuse"] and big and not info.get("adaptive_call_protected") and (raw[1] < config["big_call_min_probability"]
                or value < info["pot_odds"] + config["big_call_equity_margin"]):
            info["call_fuse"] = "big_call"
        if info["call_fuse"]:
            probs[:] = 0.; probs[0] = 1.
            return probs, info
    amount = {1: call, 2: int(snapshot.pot * .5), 3: int(snapshot.pot), 4: snapshot.me_stack}
    if amounts is not None:
        amount.update(amounts)
    _r5_weak_reraise(snapshot, raw, probs, legal, amount, config, equity, info, protected)
    def excessive(chips):
        return chips > config["active_shove_min_bb"] * blind and chips > config["active_shove_pot_ratio"] * snapshot.pot
    if 4 in legal and snapshot.me_stack > snapshot.to_call and excessive(amount[4]) and probs[4] > 0:
        if config["safe_shove_remap"]:
            _safe_shove_transfer(snapshot, raw, probs, legal, amount, excessive, config, equity, info, protected)
        else:
            targets = [a for a in (3, 2) if a in legal and not excessive(amount[a])]
            target = targets[0] if targets else 1 if 1 in legal else 0
            if target not in legal:
                raise ValueError("no safe legal shove remapping")
            info.update(active_cap=True, remapped_to=target, remapped_mass=float(probs[4]))
            if not targets:
                info["no_safe_bet_target"] = True
            probs[target] += probs[4]; probs[4] = 0.
            protected.add(target)
    actions = getattr(context, "opp_regular_postflop_actions", 0)
    checks = getattr(context, "opp_postflop_checks", 0)
    regular_bets = getattr(context, "opp_regular_postflop_bets", 0)
    if (config["passive_probe_enabled"] and snapshot.street in (1, 2) and snapshot.to_call == 0
            and 1 in legal and 2 in legal and not excessive(amount[2])
            and int(np.argmax(raw)) == 1 and raw[1] >= config["passive_min_check_probability"]
            and actions >= config["passive_min_actions"] and checks >= config["passive_min_checks"]
            and regular_bets / max(actions, 1) <= config["passive_max_aggression"]
            and equity() >= config["passive_min_equity"]):
        transfer = probs[1] * config["passive_check_transfer"]
        probs[1] -= transfer; probs[2] += transfer
        info.update(passive_probe=True, probe_mass=float(transfer), regular_aggression=regular_bets / max(actions, 1))
        protected.add(2)
    _pot_control(snapshot, raw, probs, legal, amount, config, equity, info, protected)
    _late_pot_control(snapshot, probs, legal, amount, config, equity, info, protected)
    _dominant_fold_filter(snapshot, raw, probs, legal, amount, config, equity, info, protected)
    _r4_continuation_guards(snapshot, raw, probs, legal, config, equity, info, protected)
    _r5_final_guards(snapshot, raw, probs, legal, amount, config, equity, info, protected)
    _r5_supplement(snapshot, raw, probs, legal, amount, config, equity, info, protected, context)
    _r5_blind_protection(snapshot, probs, legal, config, equity, info, protected)
    _r6_strong_fold_floor(snapshot, probs, legal, config, equity, info, protected)
    for action in sorted(legal):
        if action not in protected and 0 < probs[action] < config["prune_below"]:
            probs[action] = 0.; info["pruned"].append(action)
    _r6_purify(snapshot, probs, legal, config, info, protected)
    if probs.sum() <= 0:
        probs[int(np.argmax(raw))] = 1.
    probs /= probs.sum()
    return probs, info

"""R5 public-score blind locks, independent of neural HUD columns (Python 3.6).

Inference assumes the judge's complete per-player request stream: if a skipped
hand gives the BB no request at all, the SB folded at its first opportunity.
Never infer a terminal result merely from a missing response or score sign.
"""
import copy
import math
import json


def public_meta(request):
    try:
        hand, maximum = request["hand"], request["max_hand"]
        me, dealer = request["my_id"], request["dealer_id"]
        scores = request["total_win_chips"]
        integers = (hand, maximum, me, dealer)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in integers):
            return None
        if not (0 <= hand < maximum <= 50 and me in (0, 1) and dealer in (0, 1)):
            return None
        if request.get("num_players") != 2 or len(scores) != 2:
            return None
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
            return None
        if sum(scores) != 0:
            return None
        return dict(hand=hand, maximum=maximum, me=me, dealer=dealer, scores=list(scores))
    except (KeyError, TypeError, ValueError):
        return None


def blind_budget(meta, seat, include_current=True):
    start = meta["hand"] + (0 if include_current else 1)
    return sum(100 if (meta["dealer"] + index - meta["hand"]) % 2 == seat else 50
               for index in range(start, meta["maximum"]))


def _fresh(meta):
    return dict(format="r5_blind_lock_v1", meta=meta, seen=0, confirmed=False,
                fold_streak=0, probe_used=False, rejected=False, eligible=False,
                last_infer=None, last_response=None, evidence=[], revoked=False)


def _clear(state):
    state.update(confirmed=False, fold_streak=0, probe_used=False, rejected=False)


def _revoke(state):
    state.update(confirmed=False, fold_streak=0, revoked=True)
    if state["eligible"]:
        # One bounded probe per uninterrupted score-eligible episode.
        state["rejected"] = True
        state["probe_used"] = True


def _fold(state, eligible, evidence, free=False):
    if not eligible or state["rejected"]:
        return
    state["fold_streak"] += 1
    state["confirmed"] = bool(free or state["confirmed"] or state["fold_streak"] >= 2)
    state["evidence"].append(evidence)


def observe_request(request, previous=None):
    """Pure copy-on-update observation; never commits a hypothetical action."""
    meta = public_meta(request)
    if meta is None:
        return None
    history = request.get("history", [])
    if not isinstance(history, list):
        return None
    state = copy.deepcopy(previous) if previous and previous.get("format") == "r5_blind_lock_v1" else _fresh(meta)
    old = state["meta"]
    gap = meta["hand"] - old["hand"]
    incompatible = (old["me"] != meta["me"] or old["maximum"] != meta["maximum"] or gap < 0 or gap > 2
                    or (old["dealer"] + gap) % 2 != meta["dealer"]
                    or (gap == 0 and (old["scores"] != meta["scores"] or len(history) < state["seen"]))
                    or (gap == 2 and old["me"] == old["dealer"]))
    if incompatible:
        same_match_forward = (old["me"] == meta["me"] and old["maximum"] == meta["maximum"] and gap >= 0)
        state = _fresh(meta); old = meta; gap = 0
        if same_match_forward:
            # Stale/incomplete forward history must not buy another probe.
            state.update(rejected=True, probe_used=True)
    state["evidence"] = []; state["revoked"] = False
    if gap:
        # At most one skipped hand can be proved from alternating positions:
        # our previous and current SB hands surround one BB hand with no query.
        skipped_gain = 50 if gap == 2 else 0
        gain = meta["scores"][meta["me"]] - old["scores"][old["me"]] - skipped_gain
        if state["last_infer"] is not None:
            if gain == 100:
                _fold(state, state["eligible"], "bb_free_fold" if state["last_infer"] == "free" else "bb_fold_to_open",
                      free=state["last_infer"] == "free")
            else:
                # A result inconsistent with the only safe fold proof cannot confirm a lock.
                _revoke(state)
        if gap == 2:
            skipped = dict(meta, hand=meta["hand"] - 1, dealer=1-meta["dealer"])
            opponent = 1-meta["me"]
            score_before_skipped = meta["scores"][opponent] + 50
            _fold(state, score_before_skipped > blind_budget(skipped, opponent), "skipped_sb_first_fold")
        state["seen"] = 0
    state["meta"] = meta
    opponent = 1-meta["me"]
    state["eligible"] = meta["scores"][opponent] > blind_budget(meta, opponent)
    if not state["eligible"]:
        _clear(state)
    for event in history[state["seen"]:]:
        if event.get("player_id") == opponent:
            if event.get("action") == -1:
                _fold(state, state["eligible"], "observed_fold")
            else:
                # Check, Call, Bet, Raise and All-in all invalidate an all-fold lock.
                _revoke(state)
    state["seen"] = len(history)
    state["last_infer"] = None; state["last_response"] = None
    return state


def commit_response(request, response, state):
    """Record only the action actually returned/observed, never a counterfactual."""
    if state is None:
        return None
    result = copy.deepcopy(state); meta = result["meta"]
    result["last_response"] = response
    unopened_sb = (meta["me"] != meta["dealer"] and not request.get("history")
                   and not request.get("public_cards") and request.get("my_chips") == 19950)
    if unopened_sb and (response == 0 or (isinstance(response, int) and 50 < response < 19950)):
        result["last_infer"] = "free" if response == 0 else "open"
        if result["eligible"] and response == 0 and not result["confirmed"]:
            result["probe_used"] = True
    else:
        result["last_infer"] = None
    return result


def state_from_payload(payload):
    """Resume saved state, or reconstruct from complete requests+responses."""
    requests = payload.get("requests") or []
    if not requests:
        return None
    requests = [json.loads(r) if isinstance(r, str) else r for r in requests]
    saved = payload.get("data") or {}
    if isinstance(saved, str):
        saved = json.loads(saved)
    previous = saved.get("blind_lock")
    if previous:
        return observe_request(requests[-1], previous)
    responses = payload.get("responses") or []
    # If historical responses are unavailable, do not invent them or folds.
    if len(responses) != len(requests)-1:
        return observe_request(requests[-1])
    state = None
    for index, request in enumerate(requests):
        state = observe_request(request, state)
        if index < len(responses):
            state = commit_response(request, responses[index], state)
    return state


def choice(request, snapshot, state):
    """Return own-lock / one SB probe / confirmed SB transfer, or None."""
    meta = public_meta(request)
    if meta is None:
        return None, {}
    own_budget = max(0., snapshot.starting_stack-snapshot.me_stack) + blind_budget(meta, meta["me"], False)
    own_score = meta["scores"][meta["me"]]
    if own_score > own_budget:
        return "own", dict(blind_lock="own", lock_W=own_score, lock_B=own_budget)
    if (state is None or not state["eligible"] or state["rejected"] or snapshot.street != 0
            or meta["me"] == meta["dealer"] or request.get("history") or snapshot.to_call != 50):
        return None, {"lock_revoked": True} if state and state["revoked"] else {}
    mode = "opponent_confirmed" if state["confirmed"] else "opponent_probe" if not state["probe_used"] else None
    if mode is None:
        return None, {}
    return mode, dict(blind_lock=mode, lock_W=meta["scores"][1-meta["me"]],
                      lock_B=blind_budget(meta, 1-meta["me"]))

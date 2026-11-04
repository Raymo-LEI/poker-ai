"""Source bundled by export_stage2_agent.py; runtime requires only NumPy."""
import hashlib
import base64
import io
import json
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from .schema import HandSnapshot, ActionEvent
from .buckets import nearest_pot_fraction_bucket
from .hunl import HunlTokenizer
from .context import MatchCtx, CONTEXT_FIELDS
from .actions import CFRActionGuard
from .allin_mask import mask_snapshot
from .deployment_guard import guard_config, apply_guard, apply_surgical_guard, record_guard_rule
from .equity import estimated_equity
from . import reconstruct as judge
from . import blind_lock
from .forward import features_forward, _linear, _gelu

class Agent:
    def __init__(self, weights=None, config=None, settings=None, numeric_version=None, deployment_guard=None):
        if weights is None:
            archive = np.load(io.BytesIO(base64.b64decode(WEIGHT_DATA)), allow_pickle=False)
            weights = {k: archive[k] for k in archive.files}
            archive.close()
        config = CONFIG if config is None else config
        self.weights = weights
        self.config = config
        self.settings = SETTINGS if settings is None else settings
        self.context_fields = tuple(self.settings.get("context_fields", CONTEXT_FIELDS))
        self.tokenizer = HunlTokenizer(config["model"]["max_seq_len"], **{k:config[k] for k in
            ("position_conditioning", "equity_features", "equity_samples", "equity_seed")})
        if numeric_version is not None and self.tokenizer.numeric_version != numeric_version:
            raise ValueError("bundled tokenizer %s does not match weights %s"
                             % (self.tokenizer.numeric_version, numeric_version))
        self.guard = CFRActionGuard.from_config(SimpleNamespace(**config))
        self.deployment_guard = guard_config(deployment_guard)
        self.last_guard = {}
        self.release_id = self.settings.get("release_id", "development_unreleased")
        identity = hashlib.sha256(json.dumps([self.config,self.settings,self.deployment_guard],sort_keys=True).encode())
        from .search_math import CONTRACT
        identity.update(CONTRACT.encode())
        for name in sorted(self.weights):
            array = np.ascontiguousarray(self.weights[name])
            identity.update(name.encode());identity.update(str((array.dtype,array.shape)).encode());identity.update(array.tobytes())
        self.decision_identity = identity.hexdigest()
        self.search_audit = {"status":"not_started"}
        self.last_wizard = "search=not_started"
        self.meta = {"arch": config["model"]}
        # Charge BLAS initialization to model loading, not the first decision.
        self.logits({**self.tokenizer.blank(), "match_ctx":np.zeros((1,len(self.context_fields)), dtype=np.float32)})

    def observation(self, snapshot, context):
        # Same pre-softmax voluntary all-in mask as Stage 2 training/eval.
        # Absent settings leave the legal mask unchanged.
        snap = snapshot if self.settings.get('evening_online',{}).get('legal_only',False) else mask_snapshot(self.guard.apply(snapshot), self.settings.get("botzone_native_allin_mask"))
        batch = self.tokenizer.encode(snap)
        batch["match_ctx"] = np.asarray([list(context.as_dict(self.context_fields).values())], dtype=np.float32)
        return batch

    def logits(self, batch):
        features = features_forward(self.weights, self.meta, batch)
        ref = _linear(features, self.weights["policy_head.weight"], self.weights["policy_head.bias"])
        encoded = _gelu(_linear(batch["match_ctx"], self.weights["context_encoder.0.weight"], self.weights["context_encoder.0.bias"]))
        fused = _gelu(_linear(np.concatenate((features, encoded), -1), self.weights["fusion.0.weight"], self.weights["fusion.0.bias"]))
        delta = _linear(fused, self.weights["delta_head.weight"], self.weights["delta_head.bias"])
        logits = ref + self.settings["alpha"] * np.tanh(delta)
        if "preflop_anchor.context_encoder.0.weight" in self.weights:
            a = {k[len("preflop_anchor."):]: v for k, v in self.weights.items() if k.startswith("preflop_anchor.")}
            ac = _gelu(_linear(batch["match_ctx"], a["context_encoder.0.weight"], a["context_encoder.0.bias"]))
            af = _gelu(_linear(np.concatenate((features, ac), -1), a["fusion.0.weight"], a["fusion.0.bias"]))
            al = ref + self.settings["alpha"] * np.tanh(_linear(af, a["delta_head.weight"], a["delta_head.bias"]))
            post = np.any((batch["token_ids"] >= 4) & (batch["token_ids"] <= 6) & (batch["type_ids"] == 4), axis=-1)
            logits = np.where(post[:, None], logits, al)
        return np.where(batch["legal_mask"] > .5, logits, np.float32(-1e9))

    def raw_observation(self, snapshot, context):
        """Rule-legal policy for candidate-specific event likelihoods."""
        batch = self.tokenizer.encode(snapshot)
        batch["match_ctx"] = np.asarray([list(context.as_dict(self.context_fields).values())],dtype=np.float32)
        return batch

    def policy(self, snapshot, context, *, safety=False, amounts=None, request=None, lock_state=None):
        """One final pipeline for respond(), Agent.act() and local evaluation.

        Lock -> search -> one named empirical guard -> final domain validation.
        A configured lock does not skip ordinary search. There is no post-guard fusion.
        """
        from .search_game import PublicState
        from .search_belief import RangeError
        evening=bool(self.settings.get('evening_v2',False))
        profile=self.settings.get('evening_online',{})
        guard_mode=profile.get('guard_mode','slim' if evening else 'legacy_full')
        if guard_mode=='legacy_full' and evening and not (profile.get('comparison_only',False) or profile.get('server_arena_release',False)):
            raise ValueError('legacy full guard is comparison-only for Evening')
        if guard_mode not in ('slim','legacy_full','surgical'):raise ValueError('invalid guard mode')
        self.search_request=request
        if amounts is None:
            public_state=PublicState.from_request(request,snapshot) if request is not None else PublicState.from_snapshot(snapshot)
            amounts = {a:x for a,x in public_state.actions().items()
                       if a in snapshot.legal_action_ids}
        if request is None:
            request = dict(num_players=2,hand=int(context.hands_played),max_hand=50,
                my_id=snapshot.me_seat,dealer_id=1-snapshot.sb_seat,
                total_win_chips=[context.hero_net_bb*100 if p==snapshot.me_seat else -context.hero_net_bb*100 for p in (0,1)],
                history=list(snapshot.events),public_cards=list(snapshot.board_cards),my_chips=int(snapshot.me_stack))
        use_lock=(evening and profile.get('mathematical_locks',True)) or (self.deployment_guard['enabled'] and self.deployment_guard['victory_blind_lock_enabled'])
        mode,audit = blind_lock.choice(request,snapshot,lock_state) if use_lock else (None,{})
        confirmed_sb=(mode=='opponent_confirmed' and snapshot.street==0 and snapshot.to_call==50 and not request.get('history') and 1 in amounts)
        legacy_probe=(not evening and mode=='opponent_probe' and snapshot.street==0 and 1 in amounts)
        if mode=='own' or confirmed_sb or legacy_probe:
            action = 0 if mode=='own' else 1
            probs=np.zeros(5);probs[action]=1.
            self.guard_before=probs.copy(); self.guard_after=probs.copy()
            self.last_guard=dict(audit,rules={})
            record_guard_rule(self.last_guard,'lock_'+mode,probs,probs,fired=True)
            self.last_evening=dict(search=False,reason='blind_lock_'+mode,cfv_calls=0)
            self.search_audit=dict(status="skipped",reason="blind_lock_"+mode)
            self.last_wizard="search=skipped reason=blind_lock_"+mode
            return probs
        row=self.logits(self.observation(snapshot,context))[0].astype(np.float64)
        probs=np.exp(row-row.max());probs/=probs.sum()
        from .wizard import fuse
        try:
            probs=fuse(self,snapshot,context,probs,amounts)
        except (RangeError,FloatingPointError) as exc:
            # Expected numeric/range failure is observable and NEVER a search pass.
            self.search_audit=dict(status="error_blueprint",error=type(exc).__name__,detail=str(exc))
            self.last_wizard="search=error_blueprint error="+type(exc).__name__+":"+str(exc)
            if self.settings.get("strict_search",False) or (evening and snapshot.street==3):
                raise
        active_evening=(self.settings.get('evening_v2',False) and snapshot.street>0 and
                        (self.search_audit.get('status')=='active' or self.search_audit.get('reason')=='current_hand_guaranteed_lock'))
        if (safety and not evening and not active_evening and not self.deployment_guard["enabled"] and CFRActionGuard.all_in_pot_ratio(snapshot)>3.
                and 1 in snapshot.legal_action_ids and 0<probs[4]<.05):
            probs[1]+=probs[4];probs[4]=0.
        before_guard=probs.copy()
        if evening and guard_mode=='surgical':
            probs,self.last_guard=apply_surgical_guard(snapshot,probs,self.deployment_guard,estimated_equity,amounts,context=context)
        elif evening and guard_mode=='slim':
            probs,self.last_guard=apply_guard(snapshot,probs,self.deployment_guard,estimated_equity,amounts,context=context,search_active=True)
            self.last_guard['postflop_guard_mode']=guard_mode
        else:
            probs,self.last_guard=apply_guard(snapshot,probs,self.deployment_guard,estimated_equity,amounts,context=context)
        self.guard_before=before_guard.copy(); self.guard_after=probs.copy()
        self.last_guard['pipeline_net_mass_moved']=float(.5*np.abs(before_guard-probs).sum())
        if guard_mode!='surgical' and not self.last_guard.get('rules'):
            record_guard_rule(self.last_guard,'legacy_pipeline' if guard_mode=='legacy_full' else 'slim_safety',before_guard,probs)
        self.last_guard.update(audit)
        if not evening and mode == "opponent_confirmed" and snapshot.street==0 and 1 in amounts:
            probs[1]+=probs[0];probs[0]=0.
            self.last_guard["opponent_lock_transfer"] = True
        changed=not np.allclose(before_guard,probs,atol=1e-14,rtol=0.)
        base_output="blueprint_fallback" if self.search_audit.get("status")=="error_blueprint" else self.search_audit.get("output","dcfr_average")
        self.search_audit.update(final_output=("slim_safety_policy" if evening and guard_mode=='slim' else "empirical_guard_policy") if changed else base_output,
                                 guard_changed=changed)
        self.last_wizard += " final="+("empirical_guard" if changed else base_output)
        allowed=set(amounts)
        if any(p>0 and a not in allowed for a,p in enumerate(probs)) or not np.isclose(probs.sum(),1.):
            raise ValueError("final policy violates executable domain")
        return probs

    def act(self, snapshot, context, *, safety=False, rng=None, amounts=None, request=None, lock_state=None):
        probs=self.policy(snapshot,context,safety=safety,amounts=amounts,request=request,lock_state=lock_state)
        if getattr(self,"force_argmax",False):
            return int(np.argmax(probs))
        draw=(random if rng is None else rng).random()
        active=np.flatnonzero(probs>0)
        cumulative=np.cumsum(probs[active]);cumulative[-1]=1.
        return int(active[np.searchsorted(cumulative,min(draw,np.nextafter(1.,0.)),side="right")])


def card(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 52:
        raise ValueError("invalid Botzone card")
    return "HDSC"[value % 4] + "23456789TJQKA"[value // 4]


def request_snapshot(request):
    """Project actual judge public chips/history into Stage 1 token semantics.

    Canonical seat 0=BB, 1=SB. Judge min-raises/order remain judge rules.
    Output amounts are *extra chips*, not PokerKit street raise-to totals.
    """
    blank = dict(request, history=[], public_cards=[])
    view = judge.reconstruct(blank)
    canonical = lambda pid: 0 if pid == view.dealer else 1
    me = request["my_id"]
    events = []
    for entry in request["history"]:
        street = int(entry.get("round", view.street))
        while view.street < street:
            judge._advance_street(view)
        pid = entry["player_id"]
        owed = max(view.street_put) - view.street_put[pid]
        raw = int(entry["action"])
        chips = 0 if raw == -1 else view.chips[pid] if raw == -2 else owed if raw == 0 else raw
        kind = ("fold" if raw == -1 else "all_in" if chips == view.chips[pid] else
                "check" if chips == 0 else "call" if chips <= owed else "bet" if owed == 0 else "raise")
        fraction = chips / view.pot if view.pot else 0.
        events.append(ActionEvent(street, canonical(pid), "BB" if canonical(pid) == 0 else "SB", pid == me,
            None, kind, view.pot, chips, fraction, nearest_pot_fraction_bucket(fraction), owed, view.chips[pid], ()))
        judge._apply_history(view, entry)
    target = {0:0,3:1,4:2,5:3}[len(request["public_cards"])]
    while view.street < target:
        judge._advance_street(view)
    view.my_id = me
    if view.chips[me] != request["my_chips"]:
        raise ValueError("my_chips disagrees with history")
    owed = max(view.street_put) - view.street_put[me]
    responses = {0:-1, 1:0 if judge.judge_accepts(view, 0) else -2}
    if judge.judge_accepts(view, 0) and view.chips[1-me] > 0:
        for action, fraction in ((2,.5), (3,1.)):
            # Same pot-fraction chips as HunlTable; adjust only to judge minimum.
            extra = max(int(view.pot * fraction), int(judge.min_raise_extra(view) or view.chips[me]))
            if extra > owed and judge.judge_accepts(view, extra):
                responses[action] = extra
        if view.chips[me] > owed:
            responses[4] = -2
    if owed == 0:
        responses.pop(0, None)  # standard HUNL does not offer free folds
    stacks = (view.chips[view.dealer], view.chips[1-view.dealer])
    snap = HandSnapshot(tuple(events), tuple(card(c) for c in request["my_cards"]),
        tuple(card(c) for c in request["public_cards"]), canonical(me), 1, view.street,
        view.pot, stacks, owed, tuple(sorted(responses)), 100., 20000.)
    object.__setattr__(snap, "postflop_first_seat", 1)
    return snap, responses


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def _handoff_delta(previous, current):
    """Completed-hand score only; judge total_win_chips excludes new blinds.

    my_chips is a different ledger (20000 less current-hand contributions).
    Only a wholly unqueried BB hand proves an opposing SB first-action Fold.
    Alternating seats make any larger ambiguous gap fail closed.
    """
    old, new = blind_lock.public_meta(previous), blind_lock.public_meta(current)
    if old is None or new is None or old["me"] != new["me"] or old["maximum"] != new["maximum"]:
        return None, []
    gap = new["hand"] - old["hand"]
    if gap < 1 or (old["dealer"] + gap) % 2 != new["dealer"]:
        return None, []
    skipped = []
    for hand in range(old["hand"] + 1, new["hand"]):
        dealer = (old["dealer"] + hand - old["hand"]) % 2
        if old["me"] != dealer:  # Hero SB necessarily receives the first query.
            return None, []
        skipped.append(hand)
    net = new["scores"][new["me"]] - old["scores"][old["me"]] - 50 * len(skipped)
    return net, skipped


def _freeze_lock_profile(ctx, request):
    meta = blind_lock.public_meta(request)
    if meta is None:
        return
    for seat in (0, 1):
        # Freeze near the proven all-fold blind budget, before the cliff itself.
        if meta["scores"][seat] >= .9 * blind_lock.blind_budget(meta, seat):
            ctx.freeze_frequencies(meta["hand"], "near_blind_lock_seat_%d" % seat)


def _commit_context_action(ctx, request, snap, response):
    """Commit the exact executable action once, for live and cumulative replay."""
    chips = 0 if response == -1 else snap.me_stack if response == -2 else min(snap.to_call, snap.me_stack) if response == 0 else response
    kind = ("fold" if response == -1 else "all_in" if chips == snap.me_stack else "check" if chips == 0
            else "call" if chips <= snap.to_call else "bet" if snap.to_call == 0 else "raise")
    ctx.observe_action(ActionEvent(snap.street, snap.me_seat, snap.me_position, True, None, kind,
        snap.pot, chips, chips / snap.pot, nearest_pot_fraction_bucket(chips / snap.pot),
        snap.to_call, snap.me_stack, snap.legal_action_ids))
    ctx.remember_action(request.get("hand", 0), snap, chips, kind)
    return kind != "fold" and chips > 0 and chips <= snap.to_call and (
        snap.street == 3 or snap.opp_stack == 0 or chips == snap.me_stack)


def _complete_terminal(ctx, previous, response, current):
    """Reconcile only a validated complete Botzone per-player request stream."""
    net, skipped = _handoff_delta(previous, current)
    if net is None:
        return None, None
    old = blind_lock.public_meta(previous)
    # Migrate older data checkpoints without replaying the already counted hero action.
    if ctx.last_hand_idx != old["hand"] and response is not None:
        snap, _ = request_snapshot(previous)
        if judge.judge_accepts(judge.reconstruct(previous), response):
            chips = 0 if response == -1 else snap.me_stack if response == -2 else snap.to_call if response == 0 else response
            kind = "fold" if response == -1 else "check" if chips == 0 else "call" if chips <= snap.to_call else "raise"
            ctx.remember_action(old["hand"], snap, chips, kind)
    pending = ctx.terminal_pending and ctx.last_hand_idx == old["hand"]
    if pending and ctx.last_action_type == "fold":
        ctx.terminal_pending = False
        return ("fold" if net == -ctx.my_committed_after_act else None), net
    call_closes = (ctx.last_action_type in ("call", "all_in") and not ctx.my_last_action_was_bet_or_raise
                   and (ctx.last_street == 3 or min(ctx.last_me_stack_after, ctx.last_opp_stack) == 0))
    if pending and (ctx.last_check_closes or call_closes):
        ctx.terminal_pending = False
        matched = min(ctx.my_committed_after_act, ctx.opp_committed_at_my_act)
        return ("showdown" if net in (-matched, matched) or abs(net) <= 1 else None), net
    if not pending:
        snap, _ = request_snapshot(previous)
        if snap.events:
            end = snap.events[-1]
            mine = snap.starting_stack - snap.me_stack
            theirs = snap.starting_stack - snap.opp_stack
            if end.action_type == "fold":
                expected = -mine if end.actor_is_me else theirs
                return ("fold" if net == expected else None), net
            call = end.chips_added > 0 and end.chips_added <= end.to_call_before
            checks = len(snap.events) >= 2 and all(e.street == 3 and e.action_type == "check" for e in snap.events[-2:])
            if checks or (call and (end.street == 3 or min(snap.stacks) == 0)):
                matched = min(mine, theirs)
                return ("showdown" if net in (-matched, matched) or abs(net) <= 1 else None), net
    result = ctx.terminal_response(old["hand"], net)
    if result is None:
        return None, net
    kind, chips = result
    street, owed, stack = ctx.last_street, ctx.my_committed_after_act - ctx.opp_committed_at_my_act, ctx.last_opp_stack
    # Reuse the opportunity accounting, including 3-bet and trap denominators.
    ctx.observe_action(ActionEvent(street, 1 - (0 if old["me"] == old["dealer"] else 1),
        "SB" if old["me"] == old["dealer"] else "BB", False, None, kind,
        ctx.my_committed_after_act + ctx.opp_committed_at_my_act, chips, 0., 0, owed, stack, ()))
    return kind, net


def context_from_payload(payload):
    """Replay cumulative judge requests, deduplicating per-hand histories.

    Botzone requests carry zero-based hand and total_win_chips. data is used
    as a compact cross-process checkpoint. Missing terminal actions enter the
    HUD only after exact public chip-ledger reconciliation.
    """
    requests = [_object(r) for r in payload["requests"]]
    historical = [(_object(r), response) for r, response in zip(payload["requests"], payload.get("responses") or [])]
    saved = _object(payload.get("data") or {})
    if (saved.get("format") == "stage2_agent_data_v1"
            and int(requests[-1].get("hand", 0)) < saved.get("hand", -1)):
        saved = {}  # A new 50-hand match must not inherit the previous opponent.
    if saved.get("format") == "stage2_agent_data_v1":
        ctx = MatchCtx.from_state_dict(saved["context"])
        if "opp_postflop_bet_responses" not in saved["context"]:
            ctx.station_evidence_complete = False
        hand, seen, last = saved["hand"], saved["seen"], saved["last"]
        known_showdown = saved.get("known_showdown", False)
        requests = [r for r in requests if int(r.get("hand", 0)) >= hand]
    else:
        ctx, hand, seen, last = MatchCtx(), -1, 0, None
        first = requests[0]
        meta = blind_lock.public_meta(first)
        complete_start = (meta is not None and not first.get("public_cards")
            and ((meta["hand"]==0 and len(first.get("history",[]))==(0 if meta["me"]!=meta["dealer"] else 1))
                 or (meta["hand"]==1 and meta["me"]!=meta["dealer"] and not first.get("history"))))
        if not complete_start:
            ctx.station_evidence_complete = False
        elif meta["hand"] == 1:
            # A complete stream starting at our second-hand SB query proves
            # the opponent folded its initial SB without giving our BB a turn.
            _freeze_lock_profile(ctx, first)
            if not ctx.frequency_frozen:
                ctx.opp_preflop_action_opportunities += 1
                ctx._record("vpip", False)
                ctx._record("pfr", False)
        known_showdown = False
    for req in requests:
        current = int(req.get("hand", 0))
        if current != hand:
            if ctx.active:
                response = saved.get("response") if saved.get("last") == last else None
                if response is None:
                    historical_responses = payload.get("responses") or []
                    for index, original in enumerate(payload["requests"]):
                        if index < len(historical_responses) and _object(original) == last:
                            response = historical_responses[index]
                settled_net, skipped = _handoff_delta(last, req)
                terminal, _ = _complete_terminal(ctx, last, response, req)
                if response is None or terminal is None:
                    ctx.station_evidence_complete = False
                old_total = last.get("total_win_chips", [0,0])[last["my_id"]]
                new_total = req.get("total_win_chips", [0,0])[req["my_id"]]
                net = new_total - old_total if settled_net is None else settled_net
                showdown = terminal in ("call", "showdown")
                ctx.last_terminal_status = "split" if showdown and abs(net) <= 1 else terminal or "unknown"
                ctx.finish_hand(net / 100., showdown=showdown, split=abs(net) <= 1,
                                statistics_known=terminal is not None)
                for skipped_hand in skipped:
                    score_before = new_total - 50 * (current - skipped_hand)
                    skipped_req = dict(req, hand=skipped_hand, dealer_id=(last["dealer_id"] + skipped_hand - hand) % 2,
                                       total_win_chips=[score_before if seat == req["my_id"] else -score_before for seat in (0, 1)])
                    _freeze_lock_profile(ctx, skipped_req)
                    if not ctx.frequency_frozen:
                        ctx.opp_preflop_action_opportunities += 1
                        ctx._record("vpip", False)
                        ctx._record("pfr", False)
            ctx.hands_played = current
            ctx.hero_net_bb = req.get("total_win_chips", [0,0])[req["my_id"]] / 100.
            _freeze_lock_profile(ctx, req)
            ctx.begin_hand()
            hand, seen, known_showdown = current, 0, False
        if len(req["history"]) < seen:
            continue
        snap, _ = request_snapshot(req)
        for event in snap.events[seen:]:
            ctx.observe_action(event)
        if len(snap.events) >= seen:
            seen, last = len(snap.events), req
            if snap.events:
                end = snap.events[-1]
                call = end.chips_added > 0 and end.chips_added <= end.to_call_before
                known_showdown |= call and (end.street == 3 or min(snap.stacks) == 0)
            for original, response in historical:
                if original == req and judge.judge_accepts(judge.reconstruct(req), response):
                    known_showdown |= _commit_context_action(ctx, req, snap, response)
                    seen += 1
                    break
    # Deployment-only public score/schedule; not serialized into neural HUD columns.
    ctx.public_match = blind_lock.public_meta(requests[-1])
    return ctx, {"format":"stage2_agent_data_v1", "context":ctx.state_dict(), "hand":hand, "seen":seen, "last":last, "known_showdown":known_showdown, "opp_shove_count":ctx.opp_shove_count}


def respond(agent, payload, safety=None):
    safety = SAFETY if safety is None else safety
    request = _object(payload["requests"][-1])
    decision_key = json.dumps([agent.decision_identity,agent.release_id,request],sort_keys=True)
    saved = _object(payload.get("data") or {})
    if saved.get("decision_key") == decision_key and "response" in saved:
        if not judge.judge_accepts(judge.reconstruct(request),saved["response"]):
            raise ValueError("cached response is no longer legal")
        agent.last_guard=saved.get("decision_guard",{})
        agent.search_audit=dict(status="cached",original=saved.get("search_audit",{}))
        agent.last_wizard="search=cached original="+saved.get("search_audit",{}).get("status","unknown")
        return {"response":saved["response"],"data":saved}
    snap, responses = request_snapshot(request)
    context, data = context_from_payload(payload)
    amounts = {a: snap.me_stack if r == -2 else min(snap.to_call, snap.me_stack) if a == 1 else max(r, 0)
               for a, r in responses.items()}
    settings = getattr(agent,'settings',{})
    use_lock = (settings.get('evening_v2',False) and settings.get('evening_online',{}).get('mathematical_locks',True)) or (agent.deployment_guard["enabled"] and agent.deployment_guard["victory_blind_lock_enabled"])
    lock_state = blind_lock.state_from_payload(payload) if use_lock else None
    action = agent.act(snap,context,safety=safety,amounts=amounts,request=request,lock_state=lock_state)
    if agent.last_guard.get("blind_lock") == "own":
        responses[0] = -1  # judge-legal free fold is a match-objective constraint
    response = responses[action]
    if not judge.judge_accepts(judge.reconstruct(request), response):
        raise ValueError("export selected judge-illegal response")
    # Our own chosen action is known even if it ends this hand before another request.
    showdown = _commit_context_action(context, request, snap, response)
    data.update(context=context.state_dict(), seen=len(snap.events)+1, decision_key=decision_key,
                response=int(response), known_showdown=showdown)
    data["search_audit"] = agent.search_audit
    data["decision_guard"] = agent.last_guard
    if use_lock:
        data["blind_lock"] = blind_lock.commit_response(request, int(response), lock_state)
        data["decision_guard"] = agent.last_guard
    return {"response":int(response), "data":data}


def run():
    try:
        payload = json.loads(sys.stdin.read())
        result = respond(Agent(), payload)
    except Exception as exc:
        # Keep a visible diagnostic for local acceptance, never silently claim success.
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        result = {"response":-1}
    print(json.dumps(result, separators=(",", ":")))

"""Source bundled by export_stage2_agent.py; runtime requires only NumPy."""
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
from .deployment_guard import guard_config, apply_guard
from .equity import estimated_equity
from . import reconstruct as judge
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
        self.meta = {"arch": config["model"]}
        # Charge BLAS initialization to model loading, not the first decision.
        self.logits({**self.tokenizer.blank(), "match_ctx":np.zeros((1,len(self.context_fields)), dtype=np.float32)})

    def observation(self, snapshot, context):
        # Same pre-softmax voluntary all-in mask as Stage 2 training/eval.
        # Absent settings leave the legal mask unchanged.
        snap = mask_snapshot(self.guard.apply(snapshot), self.settings.get("botzone_native_allin_mask"))
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
        return np.where(batch["legal_mask"] > .5, logits, np.float32(-1e9))

    def act(self, snapshot, context, *, safety=False, rng=None, amounts=None):
        row = self.logits(self.observation(snapshot, context))[0].astype(np.float64)
        probs = np.exp(row - row.max()); probs /= probs.sum()
        if (safety and not self.deployment_guard["enabled"] and CFRActionGuard.all_in_pot_ratio(snapshot) > 3. and 1 in snapshot.legal_action_ids
                and 0 < probs[4] < .05):
            probs[1] += probs[4]; probs[4] = 0.
        probs, self.last_guard = apply_guard(snapshot, probs, self.deployment_guard, estimated_equity, amounts)
        rng = random if rng is None else rng
        draw, cumulative = rng.random(), 0.
        for index, prob in enumerate(probs):
            cumulative += prob
            if draw < cumulative:
                return index
        return int(np.argmax(probs))


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
    return snap, responses


def _object(value):
    return json.loads(value) if isinstance(value, str) else value


def context_from_payload(payload):
    """Replay cumulative judge requests, deduplicating per-hand histories.

    Botzone requests carry zero-based hand and total_win_chips. data is used
    as a compact cross-process checkpoint. Unseen terminal actions are never
    invented; only observed actions/opportunities contribute to HUD.
    """
    requests = [_object(r) for r in payload["requests"]]
    saved = _object(payload.get("data") or {})
    if saved.get("format") == "stage2_agent_data_v1":
        ctx = MatchCtx.from_state_dict(saved["context"])
        hand, seen, last = saved["hand"], saved["seen"], saved["last"]
        known_showdown = saved.get("known_showdown", False)
        requests = [r for r in requests if int(r.get("hand", 0)) >= hand]
    else:
        ctx, hand, seen, last = MatchCtx(), -1, 0, None
        known_showdown = False
    for req in requests:
        current = int(req.get("hand", 0))
        if current != hand:
            if ctx.active:
                old_total = last.get("total_win_chips", [0,0])[last["my_id"]]
                new_total = req.get("total_win_chips", [0,0])[req["my_id"]]
                # We cannot infer showdown from missing terminal actions.
                ctx.finish_hand((new_total - old_total) / 100., showdown=known_showdown)
            ctx.hands_played = current
            ctx.hero_net_bb = req.get("total_win_chips", [0,0])[req["my_id"]] / 100.
            ctx.begin_hand()
            hand, seen, known_showdown = current, 0, False
        if len(req["history"]) < seen:
            continue
        snap, _ = request_snapshot(req)
        for event in snap.events[seen:]:
            ctx.observe_action(event)
        if len(snap.events) >= seen:
            seen, last, known_showdown = len(snap.events), req, False
    return ctx, {"format":"stage2_agent_data_v1", "context":ctx.state_dict(), "hand":hand, "seen":seen, "last":last, "known_showdown":known_showdown}


def respond(agent, payload, safety=None):
    safety = SAFETY if safety is None else safety
    request = _object(payload["requests"][-1])
    decision_key = json.dumps([request.get("hand", 0), request["my_id"], request["my_cards"], request["history"], request["public_cards"]], sort_keys=True)
    saved = _object(payload.get("data") or {})
    if saved.get("decision_key") == decision_key and "response" in saved:
        return {"response":saved["response"], "data":saved}
    snap, responses = request_snapshot(request)
    context, data = context_from_payload(payload)
    amounts = {a: snap.me_stack if r == -2 else min(snap.to_call, snap.me_stack) if a == 1 else max(r, 0)
               for a, r in responses.items()}
    action = agent.act(snap, context, safety=safety, amounts=amounts)
    response = responses[action]
    if not judge.judge_accepts(judge.reconstruct(request), response):
        raise ValueError("export selected judge-illegal response")
    # Our own chosen action is known even if it ends this hand before another request.
    chips = 0 if action == 0 else min(snap.to_call, snap.me_stack) if action == 1 else snap.me_stack if action == 4 else response
    kind = ("fold" if action == 0 else "all_in" if chips == snap.me_stack else "check" if chips == 0
            else "call" if chips <= snap.to_call else "bet" if snap.to_call == 0 else "raise")
    context.observe_action(ActionEvent(snap.street, snap.me_seat, snap.me_position, True, action, kind,
        snap.pot, chips, chips / snap.pot, nearest_pot_fraction_bucket(chips / snap.pot),
        snap.to_call, snap.me_stack, snap.legal_action_ids))
    data.update(context=context.state_dict(), seen=len(snap.events)+1, decision_key=decision_key,
                response=int(response), known_showdown=action == 1 and (snap.street == 3 or snap.opp_stack == 0 or chips == snap.me_stack))
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

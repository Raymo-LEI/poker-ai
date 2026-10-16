"""Map the NumPy policy onto one judge-legal Botzone response.

Normal path: reconstruct the last request, build the training observation,
and send the packed integer for the chosen policy id when the judge would
accept it. An illegal target becomes a call (0) when a call is legal,
otherwise all-in (-2), otherwise fold (-1).

Any model failure (missing file, exception, non-finite logits) after a
consistent reconstruction sends 0 when the judge accepts 0, and -1
otherwise. 0 is a check when nothing is owed and a call when the stack
covers the chips owed. A request that cannot be read sends -1.
"""

import json
import os
import sys

# Before any sibling import. ``forward`` imports NumPy at module scope.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from betting import action_response
from forward import forward_logits, select_action
from observe import VOCAB_SIZE, build_observation
from reconstruct import judge_accepts, reconstruct
from weights import load_weights


def map_target(view, target):
    """Judge-legal integer for a policy target. Same order as the rule bot."""

    if judge_accepts(view, target):
        return target
    if judge_accepts(view, 0):
        return 0
    if judge_accepts(view, -2):
        return -2
    return -1


def want_sample(meta):
    """Argmax unless the export or ``POKER_SAMPLE`` asks for a draw."""

    flag = os.environ.get("POKER_SAMPLE", "")
    if flag in ("1", "true", "True", "yes"):
        return True
    return bool(meta.get("sample"))


def response_from_logits(logits, obs, view, deterministic, rng):
    """One judge integer from one logit row. Raises on a bad row."""

    action_id = select_action(logits, deterministic, rng)
    response = action_response(obs.state, action_id)
    if response is None:
        raise ValueError("policy id is masked")
    return map_target(view, int(response))


def choose_response(request, deterministic=None, rng=None):
    """One action for a single request object."""

    view = None
    try:
        view = reconstruct(request)
        reported = request.get("my_chips")
        if reported is None or int(reported) != view.chips[view.my_id]:
            raise ValueError("my_chips does not match the rebuilt stack")
        weights, meta = load_weights()
        arch = meta.get("arch") or {}
        if int(arch.get("vocab_size", -1)) != VOCAB_SIZE:
            raise ValueError("weights vocab does not match the observation")
        max_seq = int(arch.get("max_seq_len", 64))
        obs = build_observation(request, max_seq_len=max_seq)
        if int(obs.state.stacks[view.my_id]) != int(reported):
            raise ValueError("training stack does not match my_chips")
        logits = forward_logits(weights, meta, obs.batch, obs.to_call)
        if deterministic is None:
            deterministic = not want_sample(meta)
        if not deterministic and rng is None:
            import random

            rng = random.Random()
        return response_from_logits(logits[0], obs, view, deterministic, rng)
    except Exception:
        return _fallback(view, request)


def action_from_payload(payload, deterministic=None, rng=None):
    """Read Botzone's ``requests`` array and act on the last element."""

    requests = payload["requests"]
    if not requests:
        raise ValueError("no requests")
    current = requests[-1]
    if isinstance(current, str):
        current = json.loads(current)
    return choose_response(current, deterministic=deterministic, rng=rng)


def run():
    """Read stdin, write one JSON line. Does not raise."""

    action = -1
    try:
        payload = json.loads(sys.stdin.read())
        action = action_from_payload(payload)
    except Exception:
        action = -1
    sys.stdout.write(json.dumps({"response": int(action)}))
    sys.stdout.write("\n")


def _fallback(view, request):
    try:
        if view is None or request is None:
            return -1
        reported = request.get("my_chips")
        if reported is None or int(reported) != view.chips[view.my_id]:
            return -1
        if judge_accepts(view, 0):
            return 0
    except Exception:
        return -1
    return -1

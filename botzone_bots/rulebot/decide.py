"""Map the rule bot onto one judge-legal Botzone response.

Normal path: reconstruct the last request, ask the rule bot for its
integer, and send it when the judge would accept it. An illegal target
becomes a call (0) when a call is legal, otherwise all-in (-2), otherwise
fold (-1).

Any exception after a consistent reconstruction sends 0 when a check is
legal and -1 otherwise. A request that cannot be read sends -1. Call is
not used on the exception path: 0 would be a call when chips are owed,
and a short stack makes that INSUFFICIENT_CHIPS.
"""

import json
import sys

from policy import policy_response
from reconstruct import check_possible, judge_accepts, reconstruct


def map_target(view, target):
    """Judge-legal integer for a rule-bot target."""

    if judge_accepts(view, target):
        return target
    if judge_accepts(view, 0):
        return 0
    if judge_accepts(view, -2):
        return -2
    return -1


def choose_response(request, policy_fn=None):
    """One action for a single request object."""

    view = None
    try:
        view = reconstruct(request)
        reported = request.get("my_chips")
        if reported is None or int(reported) != view.chips[view.my_id]:
            raise ValueError("my_chips does not match the rebuilt stack")
        fn = policy_response if policy_fn is None else policy_fn
        return map_target(view, fn(view, request))
    except Exception:
        return _exception_action(view, request)


def action_from_payload(payload, policy_fn=None):
    """Read Botzone's ``requests`` array and act on the last element."""

    requests = payload["requests"]
    if not requests:
        raise ValueError("no requests")
    current = requests[-1]
    if isinstance(current, str):
        current = json.loads(current)
    return choose_response(current, policy_fn=policy_fn)


def run():
    """Read stdin, write one JSON line. Does not raise."""

    action = -1
    try:
        payload = json.loads(sys.stdin.read())
        action = action_from_payload(payload)
    except Exception:
        action = -1
    sys.stdout.write(json.dumps({"response": action}))
    sys.stdout.write("\n")


def _exception_action(view, request):
    try:
        if view is None or request is None:
            return -1
        reported = request.get("my_chips")
        if reported is None or int(reported) != view.chips[view.my_id]:
            return -1
        if check_possible(view):
            return 0
    except Exception:
        return -1
    return -1

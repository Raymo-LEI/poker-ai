"""Botzone TexasHoldem2p entry for the Stage 2 NumPy forward.

JSON interaction only, one process per decision. Weights are not in the
zip: upload ``stage2_model.npz`` through Botzone user storage; the bot
reads it from the ``data`` path. ``POKER_STAGE2_WEIGHTS`` overrides the
path for local tests. stdout is one line with ``response`` (judge int),
``data`` (match context, well under the 100KB cap) and ``debug`` (<1KB).
"""

import json
import os
import sys
import time

WEIGHTS_NAME = "stage2_model.npz"
WEIGHTS_FORMAT = "stage2_botzone_v1"
DEBUG_LIMIT = 1000


def _install_dataclasses():
    try:
        import dataclasses  # noqa: F401
    except ImportError:
        import dataclasses_compat
        sys.modules["dataclasses"] = dataclasses_compat


def _weight_candidates():
    override = os.environ.get("POKER_STAGE2_WEIGHTS")
    if override:
        return [override]
    here = os.path.dirname(os.path.abspath(sys.argv[0] or "."))
    return [os.path.join("data", WEIGHTS_NAME),
            os.path.join(here, "data", WEIGHTS_NAME),
            os.path.join("/data", WEIGHTS_NAME)]


def load_agent(runtime):
    import numpy as np
    for path in _weight_candidates():
        if os.path.isfile(path):
            break
    else:
        raise IOError("weights not found: " + ", ".join(_weight_candidates()))
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta_json"].tobytes().decode("utf-8"))
        weights = {key: archive[key] for key in archive.files if key != "meta_json"}
    if meta.get("format") != WEIGHTS_FORMAT:
        raise ValueError("unsupported weights format %r" % meta.get("format"))
    config = meta.get("deployment_guard")
    override = os.environ.get("POKER_STAGE2_GUARD_CONFIG")
    if override:
        with open(override) as handle:
            config = json.load(handle)
    agent = runtime.Agent(weights, meta["config"], meta["settings"], meta["numeric_version"], config)
    return agent, bool(meta.get("safety", False)), path


def fallback_response(reconstruct, payload):
    """Never call a >35BB bet when inference/config/weights fail."""
    try:
        request = payload["requests"][-1]
        if isinstance(request, str):
            request = json.loads(request)
        view = reconstruct.reconstruct(request)
        owed = max(view.street_put) - view.street_put[view.my_id]
        if owed > 3500:
            return -1
        if view is not None and view.chips[view.my_id] == request["my_chips"] and reconstruct.judge_accepts(view, 0):
            return 0
    except Exception:
        pass
    return -1


def _environment():
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "missing"
    return "py%d.%d.%d np%s" % (sys.version_info[:3] + (numpy_version,))


def main():
    started = time.time()
    payload = None
    result = None
    note = ""
    try:
        payload = json.loads(sys.stdin.readline())
        _install_dataclasses()
        from stage2_runtime import runtime
        agent, safety, path = load_agent(runtime)
        result = runtime.respond(agent, payload, safety)
        note = "ok " + os.path.basename(path)
        if agent.deployment_guard["enabled"]:
            note += " guard=v10 " + json.dumps(agent.last_guard, separators=(",", ":"))
    except Exception as exc:
        note = "fallback %s: %s" % (type(exc).__name__, exc)
        print(note, file=sys.stderr)
        response = -1
        if payload is not None:
            try:
                from stage2_runtime import reconstruct
                response = fallback_response(reconstruct, payload)
            except Exception:
                pass
        result = {"response": response}
        if payload is not None and payload.get("data"):
            result["data"] = payload["data"]
    debug = "%s %s %dms" % (_environment(), note, int((time.time() - started) * 1000))
    result["debug"] = debug[:DEBUG_LIMIT]
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    sys.stdout.flush()

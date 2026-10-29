"""Platform-style acceptance for the Stage 2 Botzone zip.

Drives complete matches through the checked-in TexasHoldem2p judge
(``tests/fixtures/botzone_judge.py``'s ``judge``), keeping the judge log the
way Botzone does. Each Stage 2 decision is a fresh ``python stage2.zip``
process on one core under an address-space cap, fed the cumulative JSON
input (``requests``, ``responses``, ``data`` as a string, ``globaldata``,
limits). Any judge verdict other than OK, judge-rejected response, model
fallback, oversized ``data``/``debug`` or slow turn is reported as an error.
"""

import argparse
import importlib.util
import json
import os
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "botzone_bots" / "stage2.zip"
RULE_MAIN = ROOT / "botzone_bots" / "rulebot" / "__main__.py"
DATA_LIMIT = 100 * 1024
DEBUG_LIMIT = 1024
# Botzone: 1s base, x6 for Python, first turn of a match doubled.
TURN_LIMIT_S = 6.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spawn(cmd, payload, memory_mb, timeout, env):
    def _limit():
        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    started = time.perf_counter()
    try:
        proc = subprocess.run(["taskset", "-c", "0"] + cmd, input=json.dumps(payload) + "\n",
                              capture_output=True, text=True, timeout=timeout, env=env,
                              preexec_fn=_limit if memory_mb else None)
    except subprocess.TimeoutExpired:
        return "TLE", None, timeout, ""
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        return "RE", None, elapsed, proc.stderr[-400:]
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        body = json.loads(lines[0]) if len(lines) == 1 else None
    except ValueError:
        body = None
    if not isinstance(body, dict) or "response" not in body:
        return "NJ", None, elapsed, proc.stdout[-400:]
    return "OK", body, elapsed, proc.stderr[-400:]


def _random_action(judge_module, request, rng):
    view = judge_module.reconstruct(request)
    options = [a for a in (-1, -2, 0) if judge_module.judge_accepts(view, a)]
    low = judge_module.min_raise_extra(view)
    if low is not None:
        options += [x for x in {low, max(low, view.pot)} if judge_module.judge_accepts(view, x)]
    return rng.choice(options) if options else -1


def play_match(args, judge, reconstruct, hero_seat, rng, env):
    log, requests, responses, data = [], {0: [], 1: []}, {0: [], 1: []}, {0: "", 1: ""}
    report = {"hero_seat": hero_seat, "hero_decisions": 0, "errors": [], "turn_s": [],
              "max_data_bytes": 0, "max_debug_bytes": 0, "debug_samples": []}
    initdata = None
    while True:
        query = {"log": log}
        if initdata is not None:
            query["initdata"] = initdata
        output = json.loads(judge.judge(json.dumps(query)))
        if log and log[-1].get(str(hero_seat)) is not None and output["display"].get("error"):
            report["errors"].append("judge rejected hero: %s" % output["display"]["error"])
        if output["command"] == "finish":
            report["hero_score_bb"] = output["content"][str(hero_seat)]
            break
        initdata = output.get("initdata", initdata)
        (seat, request), = ((int(k), v) for k, v in output["content"].items())
        requests[seat].append(request)
        payload = {"requests": requests[seat], "responses": responses[seat], "data": data[seat],
                   "globaldata": "", "time_limit": TURN_LIMIT_S, "memory_limit": args.memory_mb}
        if seat == hero_seat:
            first = not responses[seat]
            limit = TURN_LIMIT_S * (2 if first else 1)
            verdict, body, elapsed, stderr = _spawn([args.python, str(args.zip)], payload,
                                                    args.memory_mb, limit + 1, env)
            report["hero_decisions"] += 1
            report["turn_s"].append(elapsed)
            if elapsed > limit:
                verdict = "TLE"
            if verdict != "OK":
                report["errors"].append("%s %s" % (verdict, stderr))
                response = None
            else:
                response = body["response"]
                saved = body.get("data", "")
                saved = saved if isinstance(saved, str) else json.dumps(saved, separators=(",", ":"))
                debug = body.get("debug", "")
                report["max_data_bytes"] = max(report["max_data_bytes"], len(saved.encode("utf-8")))
                report["max_debug_bytes"] = max(report["max_debug_bytes"], len(debug.encode("utf-8")))
                if len(saved.encode("utf-8")) > DATA_LIMIT or len(debug.encode("utf-8")) > DEBUG_LIMIT:
                    report["errors"].append("data/debug over Botzone limit")
                if not debug.startswith("py") or " ok " not in debug:
                    report["errors"].append("model fallback: " + debug)
                if len(report["debug_samples"]) < 2:
                    report["debug_samples"].append(debug)
                if isinstance(saved, str) and saved:
                    context = json.loads(saved).get("context", {})
                    if context.get("hands_played") != request["hand"]:
                        report["errors"].append("context hand %r != %r" % (context.get("hands_played"), request["hand"]))
                data[seat] = saved
        elif args.opponent == "rule":
            verdict, body, _, stderr = _spawn([sys.executable, str(RULE_MAIN)], payload, 0, TURN_LIMIT_S, env)
            response = body["response"] if verdict == "OK" else None
        else:
            verdict, response = "OK", _random_action(reconstruct, request, rng)
        responses[seat].append(response)
        log.append({"output": output})
        log.append({str(seat): {"verdict": verdict, "response": response}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--weights", type=Path, required=True, help="stage2_model.npz")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the bot (e.g. Python 3.6)")
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--opponent", choices=("random", "rule"), default="random")
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.zip = args.zip.resolve()
    judge = _load("botzone_judge_fixture", ROOT / "tests" / "fixtures" / "botzone_judge.py")
    judge.random = random.Random(args.seed)
    reconstruct = _load("botzone_reconstruct", ROOT / "botzone_bots" / "model" / "reconstruct.py")
    rng = random.Random(args.seed + 1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("POKER_")}
    reports = []
    with tempfile.TemporaryDirectory() as workdir:
        # Botzone exposes user storage under the relative ``data`` path.
        os.makedirs(os.path.join(workdir, "data"))
        os.symlink(str(args.weights.resolve()), os.path.join(workdir, "data", "stage2_model.npz"))
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            for index in range(args.matches):
                reports.append(play_match(args, judge, reconstruct, index % 2, rng, env))
        finally:
            os.chdir(cwd)
    turns = [t for r in reports for t in r.pop("turn_s")]
    summary = {
        "python": subprocess.run([args.python, "-c", "import sys,numpy;print(sys.version.split()[0], numpy.__version__)"],
                                 capture_output=True, text=True).stdout.strip(),
        "opponent": args.opponent, "matches": reports, "memory_cap_mb": args.memory_mb,
        "hero_decisions": len(turns), "max_turn_s": max(turns) if turns else None,
        "mean_turn_s": sum(turns) / len(turns) if turns else None,
        "children_max_rss_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "errors": sum(len(r["errors"]) for r in reports),
    }
    print(json.dumps(summary, indent=2))
    if summary["errors"] or not turns:
        raise SystemExit("Stage 2 Botzone acceptance failed")


if __name__ == "__main__":
    main()

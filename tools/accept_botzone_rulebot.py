"""Acceptance runs for the Botzone rule-bot entry.

Replay is in-process. Matches spawn ``python __main__.py`` once per
decision, with the full request/response history on stdin. Resource
numbers use one core and a 256MB address-space cap.
"""

import argparse
import copy
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT_DIR = ROOT / "botzone_bots" / "rulebot"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(BOT_DIR))

import decide  # noqa: E402
from botzone_referee_adapter import (  # noqa: E402
    new_dealt_holdem,
    referee_is_legal,
    state_from_holdem,
)
from test_botzone_rulebot_entry import (  # noqa: E402
    _request_from_game,
    replay_fixture,
)

MEMORY_BYTES = 256 * 1024 * 1024


def _random_legal(game, rng):
    state = state_from_holdem(game)
    options = []
    for action in (-1, -2, 0):
        if referee_is_legal(state, action):
            options.append(action)
    chips = int(state.actor_chips)
    if state.round_bet >= 0 and chips > 1:
        need = 2 * int(state.round_raise) - int(state.own_round_bet)
        if need < 1:
            need = 1
        if need < chips and referee_is_legal(state, need):
            options.append(need)
        bigger = chips - 1
        if state.pot > bigger:
            bigger = state.pot
        if bigger >= chips:
            bigger = chips - 1
        if bigger > need and referee_is_legal(state, bigger):
            options.append(bigger)
    if not options:
        return -1
    return int(rng.choice(options))


def _spawn(main, requests, responses, limit_memory):
    payload = json.dumps({"requests": requests, "responses": responses}) + "\n"
    cmd = ["taskset", "-c", "0", sys.executable, str(main)]

    def _limit():
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        text=True,
        timeout=2,
        preexec_fn=_limit if limit_memory else None,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError("entry exited %s: %s" % (proc.returncode, proc.stderr[-400:]))
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("expected one output line, got %r" % (proc.stdout,))
    body = json.loads(lines[0])
    action = body["response"]
    if isinstance(action, bool) or not isinstance(action, int):
        raise RuntimeError("response is not an int: %r" % (action,))
    return action, elapsed


def play_hands(n_hands, opponent, seed, main, limit_memory):
    """Play ``n_hands``. Entry is seat 0. ``opponent`` is ``random`` or ``rule``."""

    rng = random.Random(seed)
    errors = []
    decisions = 0
    times = []
    entry_decisions = 0
    for index in range(n_hands):
        dealer = index % 2
        deck = list(range(52))
        rng.shuffle(deck)
        game = new_dealt_holdem(dealer, deck=tuple(deck))
        per_seat = {0: [], 1: []}
        per_resp = {0: [], 1: []}
        guard = 0
        while True:
            guard += 1
            if guard > 80:
                errors.append("LOOP")
                break
            actor = int(game.round_idx)
            hole = [card.to_int() for card in game.player_cards[actor]]
            public = [card.to_int() for card in game.public_cards]
            request = _request_from_game(game, hole, public)
            per_seat[actor].append(copy.deepcopy(request))
            use_entry = actor == 0 or opponent == "rule"
            try:
                if use_entry:
                    action, elapsed = _spawn(
                        main, per_seat[actor], per_resp[actor], limit_memory
                    )
                    times.append(elapsed)
                    entry_decisions += 1
                else:
                    action = _random_legal(game, rng)
            except Exception as exc:
                errors.append("SPAWN %s" % exc)
                action = -1
            try:
                result = game.player_action(int(action))
            except ValueError as exc:
                errors.append(str(exc))
                result = game.player_action(-1)
                action = -1
            per_resp[actor].append(int(action))
            decisions += 1
            if result:
                break
    return {
        "hands": n_hands,
        "opponent": opponent,
        "decisions": decisions,
        "entry_decisions": entry_decisions,
        "errors": errors,
        "max_turn_s": max(times) if times else None,
        "mean_turn_s": (sum(times) / len(times)) if times else None,
    }


def _rss_sample(main):
    """One capped turn: poll VmPeak / VmHWM, then confirm it printed."""

    game = new_dealt_holdem(0)
    request = _request_from_game(
        game, [card.to_int() for card in game.player_cards[1]], []
    )
    payload = json.dumps({"requests": [request], "responses": []}) + "\n"
    cmd = ["taskset", "-c", "0", sys.executable, str(main)]

    def _limit():
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_limit,
    )
    vm_peak = 0
    vm_hwm = 0
    assert proc.stdin is not None
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.close()
    while True:
        try:
            status = open("/proc/%d/status" % proc.pid, "r").read()
        except IOError:
            status = ""
        for line in status.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "VmPeak:" and parts[1].isdigit():
                vm_peak = max(vm_peak, int(parts[1]))
            if len(parts) >= 2 and parts[0] == "VmHWM:" and parts[1].isdigit():
                vm_hwm = max(vm_hwm, int(parts[1]))
        if proc.poll() is not None:
            break
    stdout = proc.stdout.read() if proc.stdout is not None else b""
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    return {
        "returncode": proc.returncode,
        "vm_peak_kb": vm_peak,
        "vm_hwm_kb": vm_hwm,
        "stdout": stdout.decode("utf-8").strip(),
        "stderr": stderr.decode("utf-8")[-200:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--main", type=str, default=str(BOT_DIR / "__main__.py"))
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    report = {}
    if not args.skip_replay:
        replay = replay_fixture()
        report["replay"] = {
            "decisions": replay["decisions"],
            "mismatches": len(replay["mismatches"]),
            "illegal": len(replay["illegal"]),
        }
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["vs_random"] = play_hands(args.hands, "random", args.seed, args.main, True)
    report["vs_rule"] = play_hands(args.hands, "rule", args.seed + 1, args.main, True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report["child_max_rss_kb"] = after
    report["child_max_rss_before_kb"] = before
    report["rss_sample"] = _rss_sample(args.main)
    # Drop the long error lists when empty so the print stays readable.
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    bad = []
    if not args.skip_replay and (
        report["replay"]["mismatches"] or report["replay"]["illegal"]
    ):
        bad.append("replay")
    for key in ("vs_random", "vs_rule"):
        if report[key]["errors"]:
            bad.append(key)
        if report[key]["max_turn_s"] is None or report[key]["max_turn_s"] >= 1.0:
            bad.append(key + " time")
    if report["rss_sample"]["returncode"] != 0:
        bad.append("rss")
    peak = report["rss_sample"]["vm_peak_kb"]
    hwm = report["child_max_rss_kb"]
    if not peak or peak >= 256 * 1024 or not hwm or hwm >= 256 * 1024:
        bad.append("rss size")
    if bad:
        raise SystemExit("acceptance failed: %s" % ", ".join(bad))


if __name__ == "__main__":
    main()

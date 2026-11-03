"""Play up to 1000 hands against the GTO Wizard researcher API.

The transport is injected. ``RequestsTransport`` is the only HTTP
implementation, and it is constructed by the CLI after the environment key
is present. ``/act`` is sent at most once per decision. A timeout or any
other failure does not retry that post. Leaderboard routes are not called.

Chip amounts stay on the standard 200BB ``HunlTable``. An abstract A5 size
that the table rejects, or that falls outside the API ``raise_range``, is
not snapped to a neighboring size.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from alphanl.stage2.context import MatchCtx
from alphanl.transformer.hunl import HunlTable
from alphanl.transformer.schema import normalize_card
from alphanl.transformer.tables import aligned_decks, standard_deck

from .adapter_hunl import AdapterError, legal_action_ids, to_wizard_action
from .client import ApiError
from .schema import GAME_NAME, SCHEMA_A5

MAX_HANDS = 1000
MAX_DECISIONS = 40
BIG_BLIND = 100
_BET = re.compile(r"^b([0-9]+)$")
_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_BOARD_LENGTH = {0: 0, 1: 3, 2: 4, 3: 5}


def _whole(value, name: str, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"{name} must be whole chips")
    number = float(value)
    if not math.isfinite(number) or abs(number - round(number)) > 1e-6:
        raise AdapterError(f"{name} must be whole chips")
    return int(round(number))


def _finite_number(value, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise AdapterError(f"{name} must be a finite number")
    return number


def _parse_cards(text, *, count: int | None = None) -> tuple[str, ...]:
    if text is None:
        return ()
    if not isinstance(text, str):
        raise AdapterError("cards must be a string")
    raw = text.strip()
    if not raw:
        return ()
    if any(ch.isspace() for ch in raw) or "," in raw:
        parts = [part for part in raw.replace(",", " ").split() if part]
    else:
        if len(raw) % 2:
            raise AdapterError("card string has an odd length")
        parts = [raw[index:index + 2] for index in range(0, len(raw), 2)]
    try:
        cards = tuple(normalize_card(part) for part in parts)
    except ValueError as exc:
        raise AdapterError("unrecognized card") from exc
    if count is not None and len(cards) != count:
        raise AdapterError("unexpected card count")
    if len(set(cards)) != len(cards):
        raise AdapterError("duplicate cards")
    return cards


def _require_game(game) -> None:
    if not isinstance(game, dict):
        raise AdapterError("game payload is missing")
    if game.get("game_name") != GAME_NAME:
        raise AdapterError("only HUNL 200BB is supported")
    if game.get("stack_reset_per_hand") is not True:
        raise AdapterError("HUNL 200BB resets stacks every hand")
    if _whole(game.get("starting_stack"), "starting_stack") != 20_000:
        raise AdapterError("starting stack must be 20000 chips")
    blinds = game.get("blinds")
    if not isinstance(blinds, (list, tuple)) or len(blinds) != 2:
        raise AdapterError("blinds must be 50 and 100")
    if {_whole(blinds[0], "small blind"), _whole(blinds[1], "big blind")} != {50, 100}:
        raise AdapterError("blinds must be 50 and 100")


def _players(state: dict) -> tuple[dict, dict]:
    players = state.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise AdapterError("expected two players")
    by_position = {}
    for player in players:
        if not isinstance(player, dict) or not isinstance(player.get("name"), str) or not player["name"]:
            raise AdapterError("player name is missing")
        position = player.get("position")
        if position not in ("SB", "BB") or position in by_position:
            raise AdapterError("positions must be SB and BB")
        by_position[position] = player
    if set(by_position) != {"SB", "BB"}:
        raise AdapterError("positions must be SB and BB")
    return by_position["SB"], by_position["BB"]


def _deal_order(hero_seat: int, hero_cards: tuple[str, str], villain_cards: tuple[str, ...], board: tuple[str, ...]):
    used = set(hero_cards) | set(villain_cards) | set(board)
    if len(used) != len(hero_cards) + len(villain_cards) + len(board):
        raise AdapterError("duplicate cards")
    pool = [card for card in standard_deck() if card not in used]

    def take(count: int) -> list[str]:
        if len(pool) < count:
            raise AdapterError("deck ran out of cards")
        chosen = pool[:count]
        del pool[:count]
        return chosen

    villain = list(villain_cards) if len(villain_cards) == 2 else take(2)
    full_board = list(board) + take(5 - len(board))
    if hero_seat == 0:
        hole_bb, hole_sb = hero_cards, villain
    else:
        hole_bb, hole_sb = villain, hero_cards
    try:
        _rlcard, pokerkit_order = aligned_decks(hole_bb, hole_sb, full_board)
    except ValueError as exc:
        raise AdapterError("cannot deal this board") from exc
    return pokerkit_order


def _parse_token(token: str) -> tuple[str, int | None]:
    if token == "_":
        return "end", None
    if token in ("f", "k", "c"):
        return token, None
    matched = _BET.fullmatch(token)
    if matched is None or str(int(matched.group(1))) != matched.group(1):
        raise AdapterError("unrecognized action history token")
    return "b", int(matched.group(1))


def _apply_token(table: HunlTable, kind: str, amount: int | None) -> None:
    if table.is_over():
        raise AdapterError("action after the hand ended")
    seat = table.current_player()
    to_call = int(table.snapshot(seat).to_call)
    if kind == "f":
        if 0 not in table.legal_action_ids():
            raise AdapterError("illegal fold")
        table.step(0)
        return
    if kind == "k":
        if to_call != 0 or 1 not in table.legal_action_ids():
            raise AdapterError("illegal check")
        table.step(1)
        return
    if kind == "c":
        if to_call <= 0 or 1 not in table.legal_action_ids():
            raise AdapterError("illegal call")
        table.step(1)
        return
    if kind != "b" or amount is None:
        raise AdapterError("unrecognized action history token")
    matches = []
    for action_id in legal_action_ids(table, SCHEMA_A5):
        spec = to_wizard_action(table, SCHEMA_A5, action_id)
        if spec.get("action") == "b" and spec.get("amount") == amount:
            matches.append(action_id)
    if len(matches) > 1:
        raise AdapterError("ambiguous bet")
    if len(matches) == 1:
        table.step(matches[0])
        return
    if not table.can_raise_to(amount):
        raise AdapterError("off-tree or illegal bet")
    table.step_raise_to(amount)


def _raise_bounds(state: dict, legal: set[str]) -> tuple[int, int] | None:
    raw = state.get("raise_range")
    if "b" not in legal:
        if raw is not None:
            raise AdapterError("raise_range is set when a bet is illegal")
        return None
    if not isinstance(raw, dict) or "min" not in raw or "max" not in raw:
        raise AdapterError("raise_range is missing")
    low = _whole(raw.get("min"), "raise_range.min")
    high = _whole(raw.get("max"), "raise_range.max")
    if low > high:
        raise AdapterError("raise_range is inverted")
    return low, high


def project_hand(body: dict, *, hero_name: str | None = None) -> dict:
    """Replay one researcher payload onto a 200BB table. Fail closed on disagreement."""

    if not isinstance(body, dict):
        raise AdapterError("researcher payload must be an object")
    hand_id = body.get("hand_id")
    if isinstance(hand_id, bool) or not isinstance(hand_id, int):
        raise AdapterError("hand_id must be an int")
    _require_game(body.get("game"))
    state = body.get("game_state")
    if not isinstance(state, dict):
        raise AdapterError("game_state is missing")
    street_name = state.get("street")
    if street_name not in _STREETS:
        raise AdapterError("unknown street")
    street_id = _STREETS[street_name]
    board = _parse_cards(state.get("board_cards"))
    if len(board) != _BOARD_LENGTH[street_id]:
        raise AdapterError("board does not match the street")
    history = state.get("action_history")
    if not isinstance(history, list) or any(not isinstance(token, str) for token in history):
        raise AdapterError("action_history must be a list of strings")
    over = state.get("is_hand_over")
    if not isinstance(over, bool):
        raise AdapterError("is_hand_over must be a bool")
    actor = state.get("player_to_act")
    if over:
        if actor is not None:
            raise AdapterError("a finished hand has no actor")
    elif actor != "user":
        raise AdapterError("refusing to act when it is not the user's turn")
    legal_list = state.get("legal_actions")
    if not isinstance(legal_list, list) or any(item not in ("f", "k", "c", "b") for item in legal_list):
        raise AdapterError("legal_actions must be f, k, c, or b")
    if len(set(legal_list)) != len(legal_list):
        raise AdapterError("duplicate legal action")
    if over and legal_list:
        raise AdapterError("a finished hand has no legal actions")
    if not over and not legal_list:
        raise AdapterError("missing legal actions")
    legal = set(legal_list)
    if not over and ("k" in legal) == ("c" in legal):
        raise AdapterError("check and call are mutually exclusive")
    bounds = _raise_bounds(state, legal)
    small, big = _players(state)
    visible = [
        player for player in (small, big)
        if isinstance(player.get("hole_cards"), str) and player["hole_cards"].strip()
    ]
    if hero_name is None:
        if len(visible) != 1:
            raise AdapterError("expected exactly one visible hand")
        hero = visible[0]
        hero_name = hero["name"]
    else:
        named = [player for player in (small, big) if player["name"] == hero_name]
        if len(named) != 1:
            raise AdapterError("hero name changed during the hand")
        hero = named[0]
    villain = big if hero is small else small
    hero_cards = _parse_cards(hero.get("hole_cards"), count=2)
    villain_cards = _parse_cards(villain.get("hole_cards"))
    if villain_cards and len(villain_cards) != 2:
        raise AdapterError("villain hole cards must be empty or two cards")
    hero_seat = 1 if hero["position"] == "SB" else 0
    table = HunlTable(seed=1, fast_evaluator=False)
    table.reset(_deal_order(hero_seat, hero_cards, villain_cards, board))
    if table.sb_seat != 1:
        raise AdapterError("HUNL seat 1 must be the small blind")
    for token in history:
        kind, amount = _parse_token(token)
        if kind == "end":
            continue
        _apply_token(table, kind, amount)
    if table.is_over() != over:
        raise AdapterError("hand-over flag does not match the replay")
    dealt = tuple(table.hole_cards(hero_seat))
    if not over and dealt != hero_cards:
        raise AdapterError("dealt hole cards do not match the payload")
    if over and dealt and dealt != hero_cards:
        raise AdapterError("dealt hole cards do not match the payload")
    if tuple(table.board_cards()) != board:
        raise AdapterError("dealt board does not match the payload")
    events = table.events_for(hero_seat)
    villain_folded = any(event.action_type == "fold" and not event.actor_is_me for event in events)
    if state.get("has_gto_wizard_folded") is not villain_folded:
        raise AdapterError("fold flag does not match the replay")
    if not over:
        if table.current_player() != hero_seat:
            raise AdapterError("actor does not match the replay")
        if table.street != street_id:
            raise AdapterError("street does not match the replay")
        pot = _whole(state.get("total_pot"), "total_pot")
        common = _whole(state.get("common_pot"), "common_pot")
        if pot != int(table.state.total_pot_amount):
            raise AdapterError("pot does not match the replay")
        bets = [int(bet) for bet in table.state.bets]
        if common != pot - sum(bets):
            raise AdapterError("common pot does not match the replay")
        stacks = {
            "SB": _whole(small.get("stack"), "SB stack"),
            "BB": _whole(big.get("stack"), "BB stack"),
        }
        actual = {"BB": int(table.state.stacks[0]), "SB": int(table.state.stacks[1])}
        if stacks != actual:
            raise AdapterError("stacks do not match the replay")
        kinds = set()
        to_call = int(table.snapshot(hero_seat).to_call)
        for action_id in legal_action_ids(table, SCHEMA_A5):
            kinds.add(to_wizard_action(table, SCHEMA_A5, action_id)["action"])
        if kinds != legal:
            raise AdapterError("legal actions do not match the table")
        if ("k" in legal) != (to_call == 0):
            raise AdapterError("check does not match the amount to call")
    winnings = _whole(state.get("winnings"), "winnings", allow_none=True)
    if over and winnings is not None and (villain_folded or any(event.action_type == "fold" for event in events) or len(villain_cards) == 2):
        payoff = int(table.payoffs()[hero_seat])
        if payoff != winnings:
            raise AdapterError("winnings do not match the replayed hand")
    return {
        "hand_id": hand_id,
        "table": table,
        "hero_seat": hero_seat,
        "hero_name": hero_name,
        "over": over,
        "events": events,
        "legal": legal,
        "raise_bounds": bounds,
        "winnings": winnings,
        "aivat_score": _finite_number(state.get("aivat_score"), "aivat_score"),
        "history": tuple(history),
        "hero_cards": hero_cards,
    }


def select_wizard_action(projected: dict, action_id: int) -> dict:
    """Map one A5 id to ``f/k/c/b``. Off-tree sizes are not rewritten."""

    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise AdapterError("action id must be an int")
    spec = to_wizard_action(projected["table"], SCHEMA_A5, action_id)
    if spec["action"] not in projected["legal"]:
        raise AdapterError("illegal wizard action")
    if spec["action"] == "b":
        amount = spec.get("amount")
        bounds = projected["raise_bounds"]
        if bounds is None or not isinstance(amount, int) or isinstance(amount, bool):
            raise AdapterError("off-tree bet")
        low, high = bounds
        if not low <= amount <= high:
            raise AdapterError("off-tree bet")
    elif "amount" in spec:
        raise AdapterError("amount is only valid for a bet")
    return spec


def _empty_report(hands_requested: int, policy_name: str | None, config_name: str | None) -> dict:
    return {
        "format": "gto_wizard_live_smoke_v1",
        "live": True,
        "network_used": False,
        "leaderboard_submitted": False,
        "training_modified": False,
        "botzone_used": False,
        "score_claimed": False,
        "game_name": GAME_NAME,
        "schema_id": SCHEMA_A5,
        "policy_name": policy_name,
        "config_name": config_name,
        "hands_requested": hands_requested,
        "hands": 0,
        "chips": None,
        "bb": None,
        "bb_per_100": None,
        "aivat_score": None,
        "aivat_bb_per_100": None,
        "stop_reason": None,
        "metrics_status": "PARTIAL",
        "notes": [
            "POST /hands and POST /hands/{id}/act only. No leaderboard write.",
            "Per-hand winnings and aivat_score are chips. bb/100 is the mean of those chip values; the big blind is 100.",
            "An act that fails is not retried. Off-tree sizes are not snapped.",
            "This file is a session log, not a claimed live score.",
        ],
    }


def _recompute(report: dict, winnings: list, aivats: list) -> None:
    report["hands"] = len(winnings)
    if winnings and all(value is not None for value in winnings):
        chips = int(sum(winnings))
        report["chips"] = chips
        report["bb"] = chips / BIG_BLIND
        report["bb_per_100"] = chips / len(winnings)
    else:
        report["chips"] = None
        report["bb"] = None
        report["bb_per_100"] = None
    if winnings and len(aivats) == len(winnings) and all(value is not None for value in aivats):
        score = float(sum(aivats))
        report["aivat_score"] = score
        report["aivat_bb_per_100"] = score / len(aivats)
    else:
        report["aivat_score"] = None
        report["aivat_bb_per_100"] = None


def _scrub(value, secret: str | None):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, list):
        return [_scrub(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item, secret) for key, item in value.items()}
    return value


def _write_report(path: Path, report: dict, secret: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _scrub(report, secret)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _play_hand(client, decider, ctx: MatchCtx, winnings: list, aivats: list) -> str | None:
    body = client.new_hand()
    hero_name = None
    hero_cards = None
    seen = 0
    previous = None
    for _decision in range(MAX_DECISIONS + 1):
        projected = project_hand(body, hero_name=hero_name)
        hero_name = projected["hero_name"]
        if hero_cards is None:
            hero_cards = projected["hero_cards"]
        elif projected["hero_cards"] != hero_cards:
            raise AdapterError("hole cards changed during the hand")
        events = projected["events"]
        if len(events) < seen:
            raise AdapterError("replayed history shrank")
        for event in events[seen:]:
            ctx.observe_action(event)
        seen = len(events)
        if projected["over"]:
            if projected["winnings"] is None:
                winnings.append(None)
                aivats.append(projected["aivat_score"])
                raise AdapterError("completed hand did not include winnings")
            showdown = not any(event.action_type == "fold" for event in events)
            ctx.finish_hand(projected["winnings"] / BIG_BLIND, showdown=showdown)
            winnings.append(projected["winnings"])
            aivats.append(projected["aivat_score"])
            return hero_name
        if previous == (projected["hand_id"], projected["history"]):
            raise AdapterError("hand did not advance")
        if _decision == MAX_DECISIONS:
            raise AdapterError("decision budget exceeded")
        action_id = decider(projected["table"], projected["hero_seat"], ctx)
        if hasattr(action_id, "item"):
            action_id = action_id.item()
        spec = select_wizard_action(projected, action_id)
        amount = spec.get("amount")
        body = client.act(projected["hand_id"], spec["action"], amount)
        advanced = project_hand(body, hero_name=hero_name)
        if advanced["hand_id"] != projected["hand_id"]:
            raise AdapterError("hand_id changed after act")
        if len(advanced["history"]) <= len(projected["history"]) or advanced["history"][:len(projected["history"])] != projected["history"]:
            raise AdapterError("act did not advance the hand")
        previous = (projected["hand_id"], projected["history"])
    raise AdapterError("decision budget exceeded")


def run_session(
    client,
    decider,
    *,
    hands: int,
    out_dir,
    policy_name: str | None = None,
    config_name: str | None = None,
    redact: str | None = None,
) -> tuple[dict, int]:
    """Play ``hands`` and write ``report.json``. Returns ``(report, exit_code)``."""

    if isinstance(hands, bool) or not isinstance(hands, int) or not 1 <= hands <= MAX_HANDS:
        raise AdapterError("hands must be an integer from 1 to 1000")
    out = Path(out_dir)
    report_path = out / "report.json"
    if report_path.exists():
        raise AdapterError("report.json already exists")
    report = _empty_report(hands, policy_name, config_name)
    winnings: list = []
    aivats: list = []
    ctx = MatchCtx()

    def flush() -> None:
        _recompute(report, winnings, aivats)
        _write_report(report_path, report, redact)

    def stop(reason: str, exc: BaseException | None, code: int) -> tuple[dict, int]:
        report["stop_reason"] = reason
        report["metrics_status"] = "INTERRUPTED" if reason == "interrupted" else "PARTIAL"
        if exc is not None:
            report["error"] = type(exc).__name__ + ": " + str(exc)
        flush()
        return report, code

    try:
        report["network_used"] = True
        flush()
        for index in range(hands):
            if ctx.hands_played >= 50:
                ctx = MatchCtx()
            ctx.begin_hand()
            _play_hand(client, decider, ctx, winnings, aivats)
            flush()
        report["stop_reason"] = "completed"
        report["metrics_status"] = "COMPLETE"
        flush()
        return report, 0
    except KeyboardInterrupt as exc:
        return stop("interrupted", exc, 2)
    except (AdapterError, ApiError, OSError, RuntimeError, ValueError, TypeError) as exc:
        return stop("fail_closed", exc, 2)

"""T-M12b-lite: rule-bot Botzone entry, reconstruction, downgrade, fallback.

The oracle is the unmodified judge. ``legal.py`` is not the oracle and is
not modified.
"""

import copy
import gzip
import itertools
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOT_DIR = ROOT / "botzone_bots" / "rulebot"
sys.path.insert(0, str(BOT_DIR))
sys.path.insert(0, str(ROOT / "eval"))

import decide  # noqa: E402
import policy  # noqa: E402
import reconstruct  # noqa: E402
from botzone_referee_adapter import (  # noqa: E402
    new_dealt_holdem,
    referee_is_legal,
    state_from_holdem,
)
from alphanl.botzone.actions import response_for_action  # noqa: E402
from alphanl.botzone.legal import apply, new_hand  # noqa: E402
from alphanl.botzone.rule_bot import (  # noqa: E402
    BotzoneRuleBot,
    postflop_strength,
    preflop_strength,
)
from alphanl.botzone.table import BotzoneTable  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "botzone_real.jsonl.gz"
PUBLIC_AT = {0: 0, 1: 3, 2: 4, 3: 5}


def _load_hands():
    hands = []
    with gzip.open(str(FIXTURE), "rt") as handle:
        for line in handle:
            hands.append(json.loads(line))
    return hands


def _request_from_game(game, hole, public):
    return {
        "num_players": int(game.num_players),
        "dealer_id": int(game.dealer_idx),
        "my_id": int(game.round_idx),
        "my_chips": int(game.player_chips[game.round_idx]),
        "my_cards": list(hole),
        "public_cards": list(public),
        "history": copy.deepcopy(game.history),
    }


def _judge_min(game):
    if game.round_bet < 0:
        return None
    actor = game.round_idx
    own = game.round_player_bet[actor]
    chips = game.player_chips[actor]
    if own < 0 or chips <= 1:
        return None
    need = 2 * game.round_raise - own
    if need < 1:
        need = 1
    if need >= chips:
        return None
    return need


def replay_fixture():
    """Compare every real decision to the live judge. Returns a report dict."""

    mismatches = []
    illegal = []
    decisions = 0
    for hand in _load_hands():
        game = new_dealt_holdem(int(hand["d"]))
        street_base = [20000, 20000]
        for pid, response, street in hand["a"]:
            decisions += 1
            if int(game.round_idx) != int(pid) or int(game.round) != int(street):
                mismatches.append(
                    {
                        "kind": "order",
                        "match": hand["m"],
                        "hand": hand["h"],
                        "want_pid": pid,
                        "want_street": street,
                        "got_pid": game.round_idx,
                        "got_street": game.round,
                    }
                )
                break
            need = PUBLIC_AT[int(game.round)]
            public = list(hand["b"][:need])
            request = _request_from_game(game, hand["c"][pid], public)
            view = reconstruct.reconstruct(request)
            opp = 1 - int(pid)
            contrib = street_base[opp] - int(game.player_chips[opp])
            problems = []
            if list(view.chips) != [int(c) for c in game.player_chips]:
                problems.append("chips")
            if int(view.opp_street) != contrib:
                problems.append("opp_street")
            if int(view.round_raise) != int(game.round_raise):
                problems.append("round_raise")
            if reconstruct.min_raise_extra(view) != _judge_min(game):
                problems.append("min_raise")
            if problems:
                mismatches.append(
                    {
                        "kind": "state",
                        "fields": problems,
                        "match": hand["m"],
                        "hand": hand["h"],
                        "index": decisions,
                    }
                )
            action = decide.choose_response(request)
            state = state_from_holdem(game)
            if not referee_is_legal(state, action):
                illegal.append(
                    {
                        "match": hand["m"],
                        "hand": hand["h"],
                        "action": action,
                        "index": decisions,
                    }
                )
            old_round = int(game.round)
            result = game.player_action(int(response))
            if int(game.round) != old_round:
                street_base = [int(c) for c in game.player_chips]
            if result:
                break
    return {
        "decisions": decisions,
        "mismatches": mismatches,
        "illegal": illegal,
    }


def test_replay_41_matches_match_the_judge_and_are_legal():
    report = replay_fixture()
    assert report["mismatches"] == []
    assert report["illegal"] == []
    assert report["decisions"] > 7000


def test_opening_state_matches_the_judge():
    game = new_dealt_holdem(0)
    request = _request_from_game(game, [48, 49], [])
    view = reconstruct.reconstruct(request)
    assert view.my_id == 1
    assert view.chips == [19900, 19950]
    assert view.street_put == [100, 50]
    assert view.round_raise == 100
    assert view.round_bet == 100
    assert reconstruct.min_raise_extra(view) == 150
    assert view.bound_b == 200
    assert policy.safe_extra(view) == 200


def test_allin_does_not_change_round_raise_and_derives_chips():
    game = new_dealt_holdem(0)
    game.player_action(-2)
    request = _request_from_game(game, [0, 4], [])
    view = reconstruct.reconstruct(request)
    assert view.my_id == 0
    assert view.status[1] == -2
    assert view.round_bet == -2
    assert view.round_raise == 100
    assert view.street_put[1] == 20000
    assert view.chips[1] == 0
    assert view.chips[0] == 19900
    assert reconstruct.min_raise_extra(view) is None
    assert decide.choose_response(request) in (-1, -2)
    assert referee_is_legal(state_from_holdem(game), decide.choose_response(request))


def test_postflop_round_raise_resets_to_50_and_non_dealer_acts():
    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    assert game.round == 1
    assert game.round_idx == 1
    request = _request_from_game(game, [8, 12], [0, 1, 2])
    view = reconstruct.reconstruct(request)
    assert view.street == 1
    assert view.round_raise == 50
    assert view.round_bet == 0
    assert view.street_put == [0, 0]
    assert view.bound_b == 100
    assert reconstruct.min_raise_extra(view) == 100
    assert view.chips[0] == 19900
    assert view.chips[1] == 19900


def test_fold_history_has_no_round_and_is_not_required_to_act():
    game = new_dealt_holdem(0)
    game.player_action(200)
    row = game.history[-1]
    assert "round" in row
    game.player_action(-1)
    fold = game.history[-1]
    assert "round" not in fold
    assert fold["action"] == -1


def test_illegal_raise_downgrades_to_call():
    game = new_dealt_holdem(0)
    request = _request_from_game(game, [0, 8], [])
    view = reconstruct.reconstruct(request)
    # 149 is below the judge minimum of 150. Call is legal.
    assert not reconstruct.judge_accepts(view, 149)
    assert reconstruct.judge_accepts(view, 0)
    assert decide.map_target(view, 149) == 0
    assert decide.choose_response(request, policy_fn=lambda view, request: 149) == 0


def test_illegal_raise_with_no_call_becomes_allin():
    game = new_dealt_holdem(0)
    game.player_action(-2)
    request = _request_from_game(game, [0, 4], [])
    view = reconstruct.reconstruct(request)
    assert not reconstruct.judge_accepts(view, 0)
    assert not reconstruct.judge_accepts(view, 100)
    assert decide.map_target(view, 100) == -2


def test_exception_checks_when_checking_is_possible_else_folds():
    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    check_request = _request_from_game(game, [8, 12], [0, 1, 2])

    def boom(view, request):
        raise RuntimeError("policy failed")

    assert decide.choose_response(check_request, policy_fn=boom) == 0

    facing = new_dealt_holdem(0)
    facing_request = _request_from_game(facing, [0, 8], [])
    assert decide.choose_response(facing_request, policy_fn=boom) == -1
    with pytest.raises(ValueError):
        decide.action_from_payload({"requests": []})


def test_unreadable_stdin_emits_fold_json(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(BOT_DIR / "__main__.py")],
        input="this is not json\n",
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == {"response": -1}


def test_policy_strength_and_responses_match_the_source_rule_bot():
    for left, right in itertools.combinations(range(52), 2):
        hole = (left, right)
        assert policy.preflop_strength(hole) == preflop_strength(hole)
    rng = random.Random(0)
    for n_board in (3, 4, 5):
        for _ in range(200):
            cards = rng.sample(range(52), 2 + n_board)
            hole = tuple(cards[:2])
            board = tuple(cards[2:])
            assert policy.postflop_strength(hole, board) == postflop_strength(hole, board)
    bot = BotzoneRuleBot()
    table = BotzoneTable(random.Random(1), dealer=0)
    compared = 0
    for dealer in range(30):
        table.start_hand(dealer=dealer % 2)
        while not table.hand_over:
            state = table.state
            view = _view_from_our_state(state, table.holes[state.actor], table.board)
            action_id = bot.action_id(table)
            expected = response_for_action(state, action_id)
            assert policy.policy_response(view) == expected
            compared += 1
            table.act(expected)
    assert compared > 50


def test_policy_tracker_follows_legal_apply_on_a_full_hand():
    """Calls and raises our mask accepts keep bound_b aligned with legal.py."""

    game = new_dealt_holdem(0)
    state = new_hand(0)
    script = [0, 0, 100, 0, 200, 400, 0, 0, 0]
    for response in script:
        request = _request_from_game(
            game,
            [0, 4],
            [8, 9, 10, 12, 16][: PUBLIC_AT[int(game.round)]],
        )
        view = reconstruct.reconstruct(request)
        assert view.chips == [state.stacks[0], state.stacks[1]]
        assert view.p_commit == [state.street_commit[0], state.street_commit[1]]
        assert view.bound_b == state.bound_b
        assert view.pot == state.pot
        assert tuple(view.p_all_in) == state.all_in
        assert int(game.round_idx) == state.actor
        state = apply(state, response)
        result = game.player_action(response)
        if result:
            break
    assert state.hand_over or game.round == 3


def _view_from_our_state(state, hole, board):
    view = reconstruct.HandView()
    view.chips = [state.stacks[0], state.stacks[1]]
    view.p_commit = [state.street_commit[0], state.street_commit[1]]
    view.p_all_in = [state.all_in[0], state.all_in[1]]
    view.p_folded = [state.folded[0], state.folded[1]]
    view.bound_b = state.bound_b
    view.pot = state.pot
    view.my_id = state.actor
    view.dealer = state.dealer
    view.street = state.street
    view.hole = list(hole)
    view.board = list(board)
    return view


def test_zip_roundtrip_prints_one_legal_action(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import pack_botzone_rulebot

    zip_path = pack_botzone_rulebot.pack(tmp_path / "rulebot.zip")
    with zipfile.ZipFile(str(zip_path)) as archive:
        names = archive.namelist()
    assert "__main__.py" in names
    assert all(not name.startswith("/") for name in names)
    extracted = tmp_path / "unpacked"
    with zipfile.ZipFile(str(zip_path)) as archive:
        archive.extractall(str(extracted))
    game = new_dealt_holdem(0)
    request = _request_from_game(game, [48, 49], [])
    payload = json.dumps({"requests": [request], "responses": []}) + "\n"
    proc = subprocess.run(
        [sys.executable, str(extracted / "__main__.py")],
        input=payload,
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert isinstance(body["response"], int)
    assert referee_is_legal(state_from_holdem(game), body["response"])
    zipped = subprocess.run(
        [sys.executable, str(zip_path)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert zipped.returncode == 0, zipped.stderr
    assert json.loads(zipped.stdout)["response"] == body["response"]

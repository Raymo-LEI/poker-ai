"""Botzone request -> c_1048 observation converter (alphanl.botzone.c1048)."""

import gzip
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from alphanl.botzone.c1048 import (ALLIN_A, CALL_A, FOLD_A, HALF_A, POT_A, RlcardShadow, action_to_response,
                                   convert, rl_card)
from alphanl.botzone.rules import ALLIN, CALL, FOLD
from alphanl.transformer.botzone_table import BotzoneTable
from alphanl.transformer.tables import aligned_decks

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "botzone_real.jsonl.gz"
ALPHANL_ROOT = Path(os.environ.get("ALPHANL_ROOT", ROOT / "third_party" / "AlphaNLHoldem"))
BZ_SUITS = "HDSC"
RANKS = "23456789TJQKA"


def bz_int(card: str) -> int:
    card = str(card)
    return RANKS.index(card[1]) * 4 + BZ_SUITS.index(card[0])


def table_card(c: int) -> str:
    return rl_card(c)  # suit-then-rank, also the BotzoneTable format


def request_from_table(t: BotzoneTable, dealer_pid: int) -> dict:
    """Botzone request JSON for the seat to act (pid: dealer = BB = table seat 0)."""
    pid_of = lambda seat: dealer_pid if seat == 0 else 1 - dealer_pid
    seat = t.current_player()
    req = t.botzone_request(seat)
    for h in req["history"]:
        h["player_id"] = pid_of(h["player_id"])
    req["dealer_id"] = dealer_pid
    req["my_id"] = pid_of(seat)
    req["my_cards"] = [bz_int(c) for c in t.hole_cards(seat)]
    req["public_cards"] = [bz_int(c) for c in t.board_cards()]
    return req


def load_original_wrapper():
    """Import the unmodified upstream NlHoldemEnvWrapper with gym/ray/TF stubbed."""
    path = ALPHANL_ROOT / "agi" / "nl_holdem_env.py"
    if not path.exists():
        pytest.skip(f"AlphaNLHoldem source not found at {path} (set ALPHANL_ROOT)")
    pytest.importorskip("rlcard")

    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules.setdefault(name, mod)
        return sys.modules[name]

    spaces = stub("gym.spaces", Box=lambda **k: None, Dict=lambda d: d, Discrete=lambda n: n)
    stub("gym", spaces=spaces)
    for name in ("ray", "ray.rllib", "ray.rllib.agents", "ray.rllib.agents.impala"):
        stub(name)
    stub("ray.rllib.agents.impala.vtrace_policy", VTraceTFPolicy=object)
    stub("ray.rllib.models", ModelCatalog=object)
    stub("ray.rllib.utils", try_import_tf=lambda: None)
    spec = importlib.util.spec_from_file_location("anl_nl_holdem_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NlHoldemEnvWrapper


def test_shadow_matches_rlcard_legality_random_play():
    rlcard = pytest.importorskip("rlcard")
    for seed in range(300):
        dealer = seed % 2
        env = rlcard.make("no-limit-holdem", config={"seed": seed, "dealer_id": dealer})
        env.reset()
        sh = RlcardShadow(dealer)
        rng = random.Random(seed)
        while not env.game.is_over():
            g = env.game
            assert sh.game_pointer == g.game_pointer and sh.stage == g.stage.value
            legal = sorted(a.value for a in g.get_legal_actions())
            assert sorted(sh.legal_actions()) == legal
            a = rng.choice(legal)
            env.step(a)
            sh.step(a)
            assert sh.in_chips == [p.in_chips for p in g.players]
        assert sh.is_over()


def test_observation_bit_identical_to_original_wrapper():
    import rlcard

    Wrapper = load_original_wrapper()
    compared = hands = 0
    for seed in range(400):
        dealer = seed % 2
        w = Wrapper.__new__(Wrapper)
        w.action_num = 5
        w.env = rlcard.make("no-limit-holdem", config={"seed": seed, "dealer_id": dealer})
        obs = w.reset()
        g = w.env.game
        deck = list(g.dealer.deck)
        board = [deck.pop() for _ in range(5)]
        cards = [[bz_int(c.get_index()) for c in p.hand] for p in g.players]
        board = [bz_int(c.get_index()) for c in board]
        _, pk_deck = aligned_decks([table_card(c) for c in cards[dealer]], [table_card(c) for c in cards[1 - dealer]],
                                   [table_card(c) for c in board])
        t = BotzoneTable(seed=0, starting_stack=20000, small_blind=50, big_blind=100)
        t.reset(pk_deck)
        rng = random.Random(seed)
        hands += 1
        while True:
            req = request_from_table(t, dealer)
            assert req["my_id"] == w.my_agent()
            view = convert(req)
            assert not view.beyond_cap
            for key, mine in (("card_info", "card_info"), ("action_info", "action_info"),
                              ("legal_moves", "legal_mask"), ("extra_info", "extra_info")):
                np.testing.assert_array_equal(np.asarray(obs[key], np.float32), view.obs[mine], err_msg=key)
            compared += 1
            legal = [int(i) for i in np.where(obs["legal_moves"])[0]]
            # only sequences legal (and identical in meaning) under both rule sets
            ok = []
            for a in legal:
                r = action_to_response(view, a)
                if a == FOLD_A and r != FOLD:
                    continue  # rlcard may fold when checking is free; Botzone adapter checks
                if a in (HALF_A, POT_A) and r != view.shadow.chips_for(a) * view.scale:
                    continue  # clamped to Botzone min raise / all-in: not the same action
                ok.append((a, r))
            a, r = rng.choice(ok)
            obs, _, done, _ = w.step(a)
            t.step_response(r)
            if done or t.is_over():
                assert done and t.is_over()
                break
    assert hands == 400 and compared > 1000


def load_hands():
    with gzip.open(FIXTURE, "rt") as fh:
        return [json.loads(line) for line in fh]


def test_real_botzone_hands_convert_and_map_legally():
    stats = {"decisions": 0, "beyond_cap": 0, "desync_notes": 0}
    diffs = {}
    for hand in load_hands():
        dealer = hand["d"]
        seat = lambda pid: 0 if pid == dealer else 1
        used = set(hand["c"][0] + hand["c"][1] + hand["b"])
        board = hand["b"] + [c for c in range(52) if c not in used][: 5 - len(hand["b"])]
        _, deck = aligned_decks([table_card(c) for c in hand["c"][dealer]],
                                [table_card(c) for c in hand["c"][1 - dealer]], [table_card(c) for c in board])
        t = BotzoneTable(seed=0, starting_stack=20000, small_blind=50, big_blind=100)
        t.reset(deck)
        for pid, resp, _street in hand["a"]:
            req = request_from_table(t, dealer)
            assert req["my_id"] == pid
            view = convert(req)
            stats["decisions"] += 1
            stats["beyond_cap"] += view.beyond_cap
            assert view.obs["card_info"].shape == (4, 13, 6) and view.obs["legal_mask"].sum() >= 1
            for a in range(5):
                r = action_to_response(view, a)
                assert t.classify(r) != "illegal", (hand["m"], hand["h"], a, r)
                # categorize rlcard-vs-Botzone legality differences
                if not view.beyond_cap and view.obs["legal_mask"][a]:
                    raw = {FOLD_A: FOLD, CALL_A: CALL, ALLIN_A: ALLIN}.get(a)
                    if raw is None:
                        raw = int(round(view.shadow.chips_for(a) * view.scale))
                    if r != raw:
                        key = f"{['fold','call','half','pot','allin'][a]} {raw if raw<0 else 'x'}->{'raise' if r > 0 else r}"
                        diffs[key] = diffs.get(key, 0) + 1
            t.step_response(resp)
    print("STATS", stats, "DIFFS", dict(sorted(diffs.items())))
    assert stats["decisions"] == 7785

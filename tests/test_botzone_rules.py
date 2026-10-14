"""Botzone rules variant of the pokerkit backend (env.rules = "botzone")."""

import gzip
import json
import random
from pathlib import Path

import pytest

from alphanl.botzone.adapter import GameParams, legalize, replay
from alphanl.botzone.rules import ALLIN, CALL, FOLD, min_raise_amount
from alphanl.transformer.botzone_table import BotzoneTable
from alphanl.transformer.tables import PokerkitTable, aligned_decks, make_table

FIXTURE = Path(__file__).parent / "fixtures" / "botzone_real.jsonl.gz"
BZ_SUITS = "HDSC"  # Botzone suit order: hearts, diamonds, spades, clubs
RANKS = "23456789TJQKA"


def bz_card(c: int) -> str:
    return BZ_SUITS[c % 4] + RANKS[c // 4]


def load_hands():
    with gzip.open(FIXTURE, "rt") as fh:
        return [json.loads(line) for line in fh]


def test_fixture_shape():
    hands = load_hands()
    assert len(hands) == 2050 and len({h["m"] for h in hands}) == 41


def test_replay_all_real_hands_zero_mismatches():
    mism = []
    for hand in load_hands():
        dealer = hand["d"]
        seat = lambda pid: 0 if pid == dealer else 1  # Botzone dealer = BB = seat 0
        used = set(hand["c"][0] + hand["c"][1] + hand["b"])
        board = hand["b"] + [c for c in range(52) if c not in used][: 5 - len(hand["b"])]
        _, deck = aligned_decks([bz_card(c) for c in hand["c"][dealer]],
                                [bz_card(c) for c in hand["c"][1 - dealer]], [bz_card(c) for c in board])
        t = BotzoneTable(seed=0, starting_stack=20000, small_blind=50, big_blind=100)
        t.reset(deck)
        for k, (pid, resp, street) in enumerate(hand["a"]):
            where = (hand["m"], hand["h"], k)
            if t.is_over():
                mism.append((where, "table ended early")); break
            if t.current_player() != seat(pid):
                mism.append((where, "to_act", t.current_player(), seat(pid)))
            if t.street != street:
                mism.append((where, "street", t.street, street))
            typ = t.classify(resp)
            if typ == "illegal":
                mism.append((where, "real accepted response judged illegal", resp))
            want = hand["t"][k]
            if want is not None and want != typ:
                mism.append((where, "action_type", typ, want))
            t.step_response(resp)
        if not t.is_over():
            mism.append(((hand["m"], hand["h"]), "hand not over"))
            continue
        pay = t.payoffs()
        got = [pay[seat(0)], pay[seat(1)]]
        if got != [float(x) for x in hand["r"]]:
            mism.append(((hand["m"], hand["h"]), "result", got, hand["r"]))
    assert mism == [], mism[:10]


@pytest.mark.parametrize("stack_bb,sb,bb", [(50, 1, 2), (100, 1, 2), (200, 1, 2), (200, 50, 100), (37, 5, 10)])
def test_every_generated_raise_is_legal_and_above_lower_bound(stack_bb, sb, bb):
    params = GameParams(stack=stack_bb * bb, small_blind=sb, big_blind=bb)
    checked = 0
    for seed in range(150):
        rng = random.Random(seed)
        t = BotzoneTable(seed=seed, starting_stack=params.stack, small_blind=sb, big_blind=bb)
        t.reset()
        while not t.is_over():
            req = t.botzone_request()
            state = replay(req, params)
            assert state.my_chips == t.chips_behind(t.current_player())
            assert state.to_call == t.to_call(t.current_player())
            legal = t.legal_action_ids()
            assert legal, "no legal action"
            for aid in range(5):
                r = t.response_for(aid)
                if r is None:
                    continue
                assert legalize(state, r) == r or (r == FOLD and state.can_check is False), (aid, r)
                assert t.classify(r) != "illegal"
                if r > 0:
                    lb = min_raise_amount(state.round_bet, state.my_bet, t._hand["last_raise"], bb)
                    assert r >= lb and r >= state.min_raise and r < state.my_chips
                    assert not state.allin_seen
                    checked += 1
                if r == CALL:
                    assert state.to_call < state.my_chips and not state.allin_seen
            t.step(rng.choice(legal))
        p = t.payoffs()
        assert p[0] + p[1] == 0
    assert checked > 0


def test_botzone_order_sb_first_every_street_and_standard_option():
    for order, post_first in (("sb_first", 1), ("standard", 0)):
        t = BotzoneTable(seed=1, starting_stack=400, small_blind=1, big_blind=2, action_order=order)
        t.reset()
        assert t.current_player() == 1  # SB first preflop either way
        t.step(1); t.step(1)  # SB calls, BB checks
        assert t.street == 1 and t.current_player() == post_first


def test_after_allin_only_fold_or_allin_and_short_raise_is_allin():
    t = BotzoneTable(seed=2, starting_stack=400, small_blind=1, big_blind=2)
    t.reset()
    t.step(4)
    assert set(t.legal_action_ids()) == {0, 4}
    assert t.classify(CALL) == "illegal" and t.classify(10) == "illegal"
    t2 = BotzoneTable(seed=3, starting_stack=10, small_blind=1, big_blind=2)
    t2.reset()
    assert t2.classify(9) == "illegal" and t2.classify(8) == "raise"  # raise < 9 behind
    assert t2.classify(2) == "illegal" and t2.classify(3) == "raise"  # min = max(4-1, 1+2, 2) = 3


def test_illegal_response_is_fold():
    t = BotzoneTable(seed=4, starting_stack=400, small_blind=1, big_blind=2)
    t.reset()
    assert t.step_response(1) == "fold" and t.is_over() and t.payoffs() == (1.0, -1.0)


def test_standard_variant_unchanged():
    assert isinstance(make_table("pokerkit", seed=0), PokerkitTable)
    assert isinstance(make_table("pokerkit", seed=0, rules="standard"), PokerkitTable)
    with pytest.raises(ValueError):
        make_table("pokerkit", seed=0, action_order="sb_first")
    with pytest.raises(ValueError):
        make_table("rlcard", seed=0, rules="botzone")
    from alphanl.transformer.config import ExperimentConfig
    assert ExperimentConfig().env.rules == "standard"
    # PokerKit standard table still puts BB first postflop
    t = make_table("pokerkit", seed=1, starting_stack=400, small_blind=1, big_blind=2)
    t.reset(); t.step(1); t.step(1)
    assert t.street == 1 and t.current_player() == 0


def test_selfplay_env_botzone_runs_and_checkpoints():
    import numpy as np
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    class RandomActor:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def act_batch(self, batch, choice):
            legal = [i for i, f in enumerate(batch["legal_mask"][0].tolist()) if f > 0.5]
            return int(self.rng.choice(legal))

    tok = BettingHistoryTokenizer(max_seq_len=64)
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=1.0)
    env = SelfPlayEnv(tok, RandomActor(), pool, seed=5, backend="pokerkit", rules="botzone",
                      starting_stack=200, big_blind=2, small_blind=1)
    obs = env.reset()
    rng = np.random.default_rng(1)
    for _ in range(200):
        legal = [i for i, f in enumerate(obs["legal_mask"][0].tolist()) if f > 0.5]
        obs, reward, done, _ = env.step(int(rng.choice(legal)))
        if done:
            obs = env.reset()
    blob = env.state_dict()
    env2 = SelfPlayEnv(tok, RandomActor(), pool, seed=5, backend="pokerkit", rules="botzone",
                       starting_stack=200, big_blind=2, small_blind=1)
    env2.load_state_dict(blob)
    env3 = SelfPlayEnv(tok, RandomActor(), pool, seed=5, backend="pokerkit",
                       starting_stack=200, big_blind=2, small_blind=1)
    with pytest.raises(ValueError):
        env3.load_state_dict(blob)

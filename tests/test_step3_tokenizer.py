"""Tokenizer, pot-fraction buckets, and rlcard-history replay."""

import math

from alphanl.transformer.buckets import pot_fraction_bucket
from alphanl.transformer.history import NlEnvTracker, make_action_event, replay_events
from alphanl.transformer.schema import HandSnapshot, normalize_card
from alphanl.transformer.tokenizer import (
    CLS_ID,
    TYPE_ACTION,
    TYPE_CLS,
    TYPE_HOLE_CARD,
    BettingHistoryTokenizer,
)


def _snap(**overrides) -> HandSnapshot:
    data = dict(
        events=(),
        hole_cards=("SA", "HK"),
        board_cards=(),
        me_seat=0,
        sb_seat=0,
        street=0,
        pot=3.0,
        stacks=(99.0, 98.0),
        to_call=1.0,
        legal_action_ids=(0, 1, 2, 3, 4),
        big_blind=2.0,
        starting_stack=100.0,
    )
    data.update(overrides)
    return HandSnapshot(**data)


def test_card_formats_and_buckets():
    assert normalize_card("SA") == normalize_card("As") == "SA"
    assert normalize_card("hK") == normalize_card("Kh") == "HK"
    assert pot_fraction_bucket(0) == 0
    assert pot_fraction_bucket(0.5) == 2
    assert pot_fraction_bucket(1.0) == 4
    assert pot_fraction_bucket(24.5) == 8


def test_replay_matches_rlcard_chip_rules():
    history = [
        [[0, 1, [0, 1, 2, 3, 4]], [1, 1, [0, 1]]],
        [[0, 2, [0, 1, 2, 3, 4]]],
        [],
        [],
    ]
    events = replay_events(history, me_seat=0, sb_seat=0, starting_stack=100)
    call, check, bet = events
    assert call.action_type == "call"
    assert call.chips_added == 1
    assert call.pot_before == 3
    assert abs(call.pot_fraction - 1 / 3) < 1e-9
    assert call.actor_is_me and call.position == "SB"
    assert check.action_type == "check" and check.chips_added == 0
    assert bet.action_type == "bet"
    assert bet.pot_before == 4 and bet.chips_added == 2
    assert bet.pot_fraction == 0.5 and bet.size_bucket == 2
    assert bet.street == 1


def test_all_in_and_explicit_type_for_pokerkit():
    history = [[[0, 4, [0, 1, 4]]], [], [], []]
    event = replay_events(history, me_seat=1, sb_seat=0)[0]
    assert event.action_type == "all_in"
    assert event.chips_added == 99
    assert event.actor_is_me is False
    assert event.size_bucket == 8

    custom = make_action_event(
        street=1,
        actor_seat=1,
        me_seat=0,
        sb_seat=0,
        action_id=None,
        action_type="raise",
        pot_before=10,
        chips_added=7,
        to_call_before=2,
        stack_before=80,
    )
    assert custom.action_id is None
    assert custom.pot_fraction == 0.7
    assert custom.position == "BB"


def test_tracker_reads_wrapper_shaped_env():
    raw = {
        "hand": ["SA", "D2"],
        "public_cards": [],
        "all_chips": [1, 2],
        "stakes": [99, 98],
        "pot": 3,
        "stage": 0,
        "current_player": 0,
        "legal_actions": [0, 1, 2, 3, 4],
    }

    class FakeEnv:
        def __init__(self):
            self.history = [[], [], [], []]
            self.last_obs = ({"raw_obs": raw}, 0)

    env = FakeEnv()
    tracker = NlEnvTracker()
    tracker.start(env)
    snap = tracker.snapshot(env)
    assert tracker.sb_seat == 0
    assert snap.me_seat == 0
    assert snap.to_call == 1
    assert snap.hole_cards == ("SA", "D2")
    assert snap.legal_action_ids == (0, 1, 2, 3, 4)
    assert snap.events == ()


def test_encode_shapes_street_marker_and_legal_mask():
    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    call = make_action_event(
        street=0,
        actor_seat=0,
        me_seat=0,
        sb_seat=0,
        action_id=1,
        action_type=None,
        pot_before=3,
        chips_added=1,
        to_call_before=1,
        stack_before=99,
        legal_action_ids=(0, 1),
    )
    snap = _snap(
        events=(call,),
        hole_cards=("As", "Kh"),
        legal_action_ids=(0, 1),
        board_cards=(),
    )
    batch = tokenizer.encode(snap)
    assert batch["token_ids"].shape == (1, 32)
    assert batch["type_ids"].shape == (1, 32)
    assert batch["numeric"].shape == (1, 32, 4)
    assert batch["pad_mask"].shape == (1, 32)
    assert batch["legal_mask"].shape == (1, 5)
    assert batch["legal_mask"].tolist() == [[1, 1, 0, 0, 0]]
    assert int(batch["token_ids"][0, 0]) == CLS_ID
    assert int(batch["type_ids"][0, 0]) == TYPE_CLS
    assert tokenizer.describe(int(batch["token_ids"][0, 2])).startswith("CARD_S")
    names = [tokenizer.describe(int(token)) for token in batch["token_ids"][0] if int(token) != 0]
    assert "STREET_0" in names
    assert any(name.startswith("ACT_call_SB_me_") for name in names)
    # pot=3, big_blind=2 → log1p(chips / big_blind) / log1p(200)
    expected_pot = math.log1p(3.0 / 2.0) / math.log1p(200.0)
    assert abs(float(batch["numeric"][0, 0, 0]) - expected_pot) < 1e-5
    assert bool(batch["pad_mask"][0, 0]) is False
    assert bool(batch["pad_mask"][0, -1]) is True


def test_truncation_keeps_cls_hole_and_newest_actions():
    tokenizer = BettingHistoryTokenizer(max_seq_len=8)
    events = []
    for i in range(10):
        events.append(
            make_action_event(
                street=0,
                actor_seat=i % 2,
                me_seat=0,
                sb_seat=0,
                action_id=1,
                action_type="check" if i % 2 else "call",
                pot_before=3 + i,
                chips_added=0 if i % 2 else 1,
                to_call_before=0 if i % 2 else 1,
                stack_before=90,
            )
        )
    # Distinguish the newest action.
    events[-1] = make_action_event(
        street=0,
        actor_seat=1,
        me_seat=0,
        sb_seat=0,
        action_id=3,
        action_type="raise",
        pot_before=20,
        chips_added=20,
        to_call_before=2,
        stack_before=80,
    )
    batch = tokenizer.encode(_snap(events=tuple(events)))
    ids = [int(t) for t in batch["token_ids"][0]]
    assert ids[0] == CLS_ID
    assert int(batch["type_ids"][0, 2]) == TYPE_HOLE_CARD
    assert int(batch["type_ids"][0, 3]) == TYPE_HOLE_CARD
    described = [tokenizer.describe(t) for t in ids if t != 0]
    assert any(name.startswith("ACT_raise_") for name in described)
    assert described.count("STREET_0") == 1
    action_count = sum(1 for type_id in batch["type_ids"][0].tolist() if type_id == TYPE_ACTION)
    assert action_count == 3
    assert len(described) == 8


def test_log1p_chip_norm_one_bb_and_200bb():
    """Botzone big blind is 100. 1bb ≈ 0.131 and 200bb = 1 on every chip channel."""

    big_blind = 100.0
    one_bb = big_blind
    stack_200bb = 200.0 * big_blind
    event = make_action_event(
        street=0,
        actor_seat=0,
        me_seat=0,
        sb_seat=0,
        action_id=1,
        action_type="call",
        pot_before=one_bb,
        chips_added=one_bb,
        to_call_before=stack_200bb,
        stack_before=stack_200bb,
        legal_action_ids=(0, 1),
    )
    snap = _snap(
        events=(event,),
        pot=one_bb,
        stacks=(stack_200bb, one_bb),
        to_call=stack_200bb,
        legal_action_ids=(0, 1),
        big_blind=big_blind,
        starting_stack=stack_200bb,
    )
    batch = BettingHistoryTokenizer(max_seq_len=32).encode(snap)
    cls = batch["numeric"][0, 0]
    assert abs(float(cls[0]) - 0.131) < 1e-3
    assert abs(float(cls[1]) - 1.0) < 1e-6
    assert abs(float(cls[2]) - 0.131) < 1e-3
    assert abs(float(cls[3]) - 1.0) < 1e-6

    action_cols = [
        i for i, type_id in enumerate(batch["type_ids"][0].tolist()) if type_id == TYPE_ACTION
    ]
    assert len(action_cols) == 1
    action = batch["numeric"][0, action_cols[0]]
    assert abs(float(action[0]) - 0.1) < 1e-6
    assert abs(float(action[1]) - 0.131) < 1e-3
    assert abs(float(action[2]) - 1.0) < 1e-6
    assert abs(float(action[3]) - 1.0) < 1e-6

"""Rule-bot BC data: chip-to-id mapping, free-fold rewrite, metadata."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from alphanl.botzone.actions import (
    ALL_IN_ACTION,
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    RAISE_LARGE_ACTION,
    RAISE_MIN_ACTION,
    action_responses,
    legal_action_ids,
)
from alphanl.botzone.legal import ALL_IN, CHECK_CALL, FOLD, apply, new_hand, to_call
from alphanl.botzone.rule_bot import BotzoneRuleBot

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "bc_rulebot"


def _load():
    path = ROOT / "scripts" / "gen_rulebot_bc.py"
    spec = importlib.util.spec_from_file_location("gen_rulebot_bc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load()


def _opening_amounts(state):
    responses = action_responses(state)
    raise_min = responses[RAISE_MIN_ACTION]
    raise_large = responses[RAISE_LARGE_ACTION]
    assert raise_min is not None and raise_large is not None
    assert raise_large > raise_min
    stack = int(state.stacks[state.actor])
    assert stack > raise_large
    return int(raise_min), int(raise_large), stack


def test_mapping_rule_is_in_the_docstring_and_metadata_text(gen):
    def collapse(text: str) -> str:
        return " ".join(text.split())

    assert collapse(gen.ACTION_MAPPING_RULE) in collapse(gen.__doc__)
    assert "nearest" in gen.__doc__.lower()
    assert "free check" in gen.__doc__


def test_exact_packed_amounts_map_to_their_ids(gen):
    state = new_hand(0)
    raise_min, raise_large, stack = _opening_amounts(state)
    assert gen.map_response_to_action_id(state, raise_min, mask_free_fold=True) == RAISE_MIN_ACTION
    assert gen.map_response_to_action_id(state, raise_large, mask_free_fold=True) == RAISE_LARGE_ACTION
    assert gen.map_response_to_action_id(state, ALL_IN, mask_free_fold=True) == ALL_IN_ACTION
    # A positive extra equal to the remaining stack is the all-in chip amount.
    assert gen.map_response_to_action_id(state, stack, mask_free_fold=True) == ALL_IN_ACTION
    assert gen.map_response_to_action_id(state, CHECK_CALL, mask_free_fold=True) == CHECK_CALL_ACTION


def test_nearest_wager_and_tie_break(gen):
    state = new_hand(0)
    raise_min, raise_large, stack = _opening_amounts(state)
    closer_min = raise_min + 1
    closer_large = raise_large - 1
    assert abs(closer_min - raise_min) < abs(closer_min - raise_large)
    assert abs(closer_large - raise_large) < abs(closer_large - raise_min)
    assert gen.map_response_to_action_id(state, closer_min, mask_free_fold=True) == RAISE_MIN_ACTION
    assert gen.map_response_to_action_id(state, closer_large, mask_free_fold=True) == RAISE_LARGE_ACTION
    # One chip under the stack is closer to all-in than to raise_large.
    assert gen.map_response_to_action_id(state, stack - 1, mask_free_fold=True) == ALL_IN_ACTION

    gap = raise_large - raise_min
    if gap % 2 == 0:
        tie = raise_min + gap // 2
        assert abs(tie - raise_min) == abs(tie - raise_large)
        assert gen.map_response_to_action_id(state, tie, mask_free_fold=True) == RAISE_MIN_ACTION
    span = stack - raise_large
    if span % 2 == 0:
        tie = raise_large + span // 2
        assert abs(tie - raise_large) == abs(tie - stack)
        assert gen.map_response_to_action_id(state, tie, mask_free_fold=True) == RAISE_LARGE_ACTION


def test_all_in_only_wager_maps_every_positive_amount_to_all_in(gen):
    state = apply(new_hand(0), ALL_IN)
    assert RAISE_MIN_ACTION not in legal_action_ids(state)
    assert RAISE_LARGE_ACTION not in legal_action_ids(state)
    assert ALL_IN_ACTION in legal_action_ids(state)
    assert gen.map_response_to_action_id(state, ALL_IN, mask_free_fold=True) == ALL_IN_ACTION
    assert gen.map_response_to_action_id(state, 500, mask_free_fold=True) == ALL_IN_ACTION
    assert to_call(state) > 0
    assert gen.map_response_to_action_id(state, FOLD, mask_free_fold=True) == FOLD_ACTION


def test_free_check_fold_is_rewritten_to_check(gen):
    state = apply(new_hand(0), CHECK_CALL)
    assert to_call(state) == 0
    assert CHECK_CALL_ACTION in legal_action_ids(state)
    assert FOLD_ACTION in legal_action_ids(state)
    assert gen.is_free_check(state)
    assert gen.map_response_to_action_id(state, FOLD, mask_free_fold=True) == CHECK_CALL_ACTION
    assert gen.map_response_to_action_id(state, FOLD, mask_free_fold=False) == FOLD_ACTION
    assert gen.map_response_to_action_id(state, CHECK_CALL, mask_free_fold=True) == CHECK_CALL_ACTION


def test_fold_facing_a_bet_stays_fold(gen):
    state = new_hand(0)
    assert to_call(state) > 0
    assert gen.map_response_to_action_id(state, FOLD, mask_free_fold=True) == FOLD_ACTION
    assert gen.map_response_to_action_id(state, FOLD, mask_free_fold=False) == FOLD_ACTION


def test_unknown_response_is_rejected(gen):
    state = new_hand(0)
    with pytest.raises(ValueError):
        gen.map_response_to_action_id(state, -3, mask_free_fold=True)


def test_single_legal_mask_is_not_counted(gen):
    assert gen.decision_is_counted(np.array([0, 1, 0, 0, 1], dtype=np.float32))
    assert not gen.decision_is_counted(np.array([0, 1, 0, 0, 0], dtype=np.float32))
    assert not gen.decision_is_counted(np.array([0, 0, 0, 0, 0], dtype=np.float32))


def test_policy_observation_drops_free_fold_and_matches_tokenizer(gen):
    from alphanl.transformer.botzone_table import BotzoneEnvTable
    from alphanl.transformer.decision_stats import policy_action_mask
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    table = BotzoneEnvTable(seed=0)
    tokenizer = BettingHistoryTokenizer(max_seq_len=gen.default_max_seq_len())
    table.reset()
    assert int(table.current_player()) == int(table.sb_seat)
    table.step(CHECK_CALL_ACTION)
    seat = int(table.current_player())
    assert float(table.snapshot(seat).to_call) == 0.0
    encoded = gen.encode_policy_observation(table, seat, tokenizer, mask_free_fold=True)
    batch = tokenizer.encode(table.snapshot(seat))
    assert np.array_equal(encoded["token_ids"], batch["token_ids"][0].numpy())
    assert np.array_equal(encoded["type_ids"], batch["type_ids"][0].numpy())
    assert np.allclose(encoded["numeric"], batch["numeric"][0].numpy())
    assert np.array_equal(encoded["pad_mask"], batch["pad_mask"][0].numpy())
    expected = policy_action_mask(batch["legal_mask"], 0.0, True)[0].numpy()
    assert np.array_equal(encoded["legal_mask"], expected)
    assert encoded["env_legal_mask"][FOLD_ACTION] == 1.0
    assert encoded["legal_mask"][FOLD_ACTION] == 0.0
    assert encoded["legal_mask"][CHECK_CALL_ACTION] == 1.0
    assert gen.decision_is_counted(encoded["legal_mask"])


def test_live_rule_bot_round_trip_except_documented_rewrite(gen):
    """Packed responses map back to the same id. The bot does not fold a free check."""

    from alphanl.botzone.table import BotzoneTable
    import random

    bot = BotzoneRuleBot()
    table = BotzoneTable(rng=random.Random(0))
    seen = 0
    for _hand in range(12):
        table.start_hand()
        guard = 0
        while table.state is not None and not table.state.hand_over:
            guard += 1
            assert guard < 80
            raw, response, mapped = gen.rulebot_target(table, bot, mask_free_fold=True)
            assert mapped == raw
            assert response == action_responses(table.state)[raw]
            table.act(response)
            seen += 1
    assert seen > 10


def test_generate_metadata_mask_and_reproducibility(gen, tmp_path: Path):
    arrays, meta = gen.collect_decisions(seed=7, hands=6, epsilon=0.2, mask_free_fold=True)
    for key in gen.REQUIRED_METADATA_KEYS:
        assert key in meta
    assert meta["seed"] == 7
    assert meta["num_hands"] == 6
    assert meta["num_decisions"] == int(arrays["actions"].shape[0])
    assert meta["num_decisions"] >= 1
    assert meta["mask_free_fold"] is True
    assert meta["action_mapping"] == gen.ACTION_MAPPING_RULE
    assert sum(meta["action_id_distribution"].values()) == meta["num_decisions"]
    assert sum(meta["opponent_hand_counts"].values()) == meta["num_hands"]
    assert set(meta["opponent_mix"]["kinds"]) == {"rule", "random", "epsilon"}
    assert meta["env_version"]["backend"] == "botzone"
    assert meta["env_version"]["game"] == "TexasHoldem2p"
    assert meta["env_version"]["stacks"] == 20000
    assert meta["env_version"]["small_blind"] == 50
    assert meta["env_version"]["big_blind"] == 100
    legal = (ROOT / "src" / "alphanl" / "botzone" / "legal.py").read_bytes()
    import hashlib

    assert meta["env_version"]["legal_py_sha256"] == hashlib.sha256(legal).hexdigest()
    assert meta["env_version"]["git_commit"]
    assert meta["max_seq_len"] == gen.default_max_seq_len()
    assert meta["skipped_single_legal"] >= 0

    assert np.isfinite(arrays["numeric"]).all()
    assert np.isfinite(arrays["legal_mask"]).all()
    assert np.isfinite(arrays["probs"]).all()
    legal_counts = (arrays["legal_mask"] > 0.5).sum(axis=1)
    assert int(legal_counts.min()) >= 2
    chosen = arrays["legal_mask"][np.arange(len(arrays["actions"])), arrays["actions"]]
    assert np.all(chosen > 0.5)
    assert np.allclose(arrays["probs"].sum(axis=1), 1.0)
    assert np.array_equal(arrays["probs"].argmax(axis=1), arrays["actions"])

    free = arrays["to_call"] == 0
    if np.any(free):
        assert np.all(arrays["legal_mask"][free, FOLD_ACTION] == 0.0)
        assert np.all(arrays["actions"][free] != FOLD_ACTION)
        # Environment still offers fold; the policy mask is what dropped it.
        check_open = arrays["env_legal_mask"][free, CHECK_CALL_ACTION] > 0.5
        assert np.any(check_open)
        assert np.all(arrays["env_legal_mask"][free][check_open, FOLD_ACTION] == 1.0)

    game_ids = arrays["game_id"]
    assert int(game_ids.min()) >= 0
    assert int(game_ids.max()) < meta["num_hands"]
    assert list(game_ids) == sorted(game_ids.tolist())

    again, again_meta = gen.collect_decisions(seed=7, hands=6, epsilon=0.2, mask_free_fold=True)
    for key in ("actions", "legal_mask", "token_ids", "raw_response", "game_id", "opponent_id"):
        assert np.array_equal(arrays[key], again[key])
    assert again_meta["action_id_distribution"] == meta["action_id_distribution"]
    assert again_meta["opponent_hand_counts"] == meta["opponent_hand_counts"]

    out = tmp_path / "bc_rulebot"
    gen.write_dataset(out, arrays, meta, force=False)
    (out / "keep.txt").write_text("leave-me", encoding="utf-8")
    with pytest.raises(SystemExit):
        gen.write_dataset(out, arrays, meta, force=False)
    gen.write_dataset(out, arrays, meta, force=True)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "leave-me"
    loaded, loaded_meta = gen.load_dataset(out)
    assert loaded_meta["num_decisions"] == meta["num_decisions"]
    assert np.array_equal(loaded["actions"], arrays["actions"])
    assert np.array_equal(loaded["legal_mask"], arrays["legal_mask"])


def test_refuses_to_write_outside_bc_rulebot_under_runs(gen):
    with pytest.raises(SystemExit):
        gen.assert_output_dir(ROOT / "runs" / "transformer_botzone_laptop", force=True)
    with pytest.raises(SystemExit):
        gen.assert_output_dir(ROOT / "runs", force=True)
    # The allowed directory may be created later. Checking the path does not create it
    # and does not delete anything already under runs/.
    allowed = gen.assert_output_dir(ROOT / "runs" / "bc_rulebot", force=True)
    assert allowed.name == "bc_rulebot"


def test_committed_sample_matches_a_fresh_run(gen):
    if not (FIXTURE / "metadata.json").is_file():
        pytest.skip("committed sample is produced by the generator before this assertion is required")
    arrays, meta = gen.load_dataset(FIXTURE)
    for key in gen.REQUIRED_METADATA_KEYS:
        assert key in meta
    assert meta["seed"] == gen.FIXTURE_SEED
    assert meta["num_hands"] == gen.FIXTURE_HANDS
    assert meta["mask_free_fold"] is True
    fresh, fresh_meta = gen.collect_decisions(
        seed=gen.FIXTURE_SEED,
        hands=gen.FIXTURE_HANDS,
        epsilon=gen.DEFAULT_EPSILON,
        mask_free_fold=True,
    )
    assert np.array_equal(fresh["actions"], arrays["actions"])
    assert np.array_equal(fresh["legal_mask"], arrays["legal_mask"])
    assert np.array_equal(fresh["token_ids"], arrays["token_ids"])
    assert fresh_meta["action_id_distribution"] == meta["action_id_distribution"]
    assert fresh_meta["opponent_hand_counts"] == meta["opponent_hand_counts"]
    text = json.dumps({key: meta[key] for key in gen.REQUIRED_METADATA_KEYS})
    assert "mask_free_fold" in text

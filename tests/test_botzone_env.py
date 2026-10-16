"""M4: botzone backend on SelfPlayEnv, checkpoints, and --init-weights."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from alphanl.botzone.actions import response_for_action
from alphanl.botzone.table import card_text
from alphanl.transformer.botzone_table import botzone_id_to_card, card_to_botzone_id
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.schema import normalize_card
from alphanl.transformer.tables import make_table

HAS_RLCARD = importlib.util.find_spec("rlcard") is not None
HAS_POKERKIT = importlib.util.find_spec("pokerkit") is not None
ROOT = Path(__file__).resolve().parents[1]


class _Call:
    def act(self, snapshot, choice):
        legal = list(snapshot.legal_action_ids)
        return 1 if 1 in legal else int(legal[0])

    def prepare_pool(self, weights):
        return None


def _env(seed, backend="botzone", **kwargs):
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=seed)
    env = SelfPlayEnv(
        tokenizer,
        _Call(),
        pool,
        seed=seed,
        backend=backend,
        record_results=False,
        **kwargs,
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    return env


def _tiny_cfg(tmp_path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.0
    cfg.device = "cpu"
    cfg.rollout_device = "cpu"
    cfg.seed = 0
    cfg.env.backend = "botzone"
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 2
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 2
    cfg.train.num_updates = 1
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 1
    cfg.train.log_every = 1
    cfg.pool.snapshot_every = 0
    cfg.pool.max_size = 2
    cfg.train.out_dir = str(tmp_path)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def test_wiki_ids_become_suit_rank_strings():
    assert botzone_id_to_card(0) == "H2"
    assert botzone_id_to_card(1) == "D2"
    assert botzone_id_to_card(2) == "S2"
    assert botzone_id_to_card(3) == "C2"
    assert botzone_id_to_card(50) == "SA"
    for card_id in range(52):
        text = botzone_id_to_card(card_id)
        assert text == normalize_card(card_text(card_id))
        assert card_to_botzone_id(text) == card_id
        assert card_to_botzone_id(card_text(card_id)) == card_id


def test_opening_snapshot_uses_m3_and_keeps_free_fold():
    table = make_table("botzone", seed=1)
    table.reset()
    assert table.sb_seat == 1
    assert table.current_player() == 1
    state = table._table.state
    assert response_for_action(state, 2) == 150
    snap = table.snapshot(1)
    assert snap.pot == 150
    assert snap.stacks == (19900.0, 19950.0)
    assert snap.to_call == 50
    assert snap.big_blind == 100
    assert snap.starting_stack == 20000
    assert snap.legal_action_ids == table.legal_action_ids()
    assert snap.legal_action_ids[0] == 0
    assert set(snap.legal_action_ids) == {0, 1, 2, 3, 4}
    assert all(len(card) == 2 and card[0] in "CDHS" for card in snap.hole_cards)

    before = snap.stacks[1]
    table.step(2)
    event = table.events_for(1)[-1]
    assert event.action_id == 2
    assert event.chips_added == 150
    assert event.action_type == "raise"
    assert table.snapshot(0).stacks[1] == before - 150
    assert table.snapshot(0).pot == 300
    assert table.current_player() == 0

    # Call, then check the flop. Fold stays legal when a check is free.
    table.step(1)
    called = table.events_for(0)[-1]
    assert called.action_id == 1
    assert called.chips_added == 100
    assert table.street == 1
    assert table.current_player() == table.sb_seat
    flop = table.snapshot(table.sb_seat)
    assert flop.to_call == 0
    assert 0 in flop.legal_action_ids
    assert 1 in flop.legal_action_ids
    assert len(flop.board_cards) == 3
    assert all(card[0] in "CDHS" for card in flop.board_cards)

    fresh = make_table("botzone", seed=2)
    fresh.reset()
    fresh.step(0)
    assert fresh.is_over()
    folded = fresh.events_for(0)[-1]
    assert folded.action_id == 0
    assert folded.chips_added == 0
    assert sum(fresh.payoffs()) == 0

    all_in = make_table("botzone", seed=3)
    all_in.reset()
    actor = all_in.current_player()
    stack = all_in.snapshot(actor).me_stack
    all_in.step(4)
    shoved = all_in.events_for(actor)[-1]
    assert shoved.action_id == 4
    assert shoved.chips_added == stack
    assert shoved.action_type == "all_in"


def test_botzone_defaults_when_rlcard_limits_are_left_in_place():
    cfg = ExperimentConfig.from_dict({"env": {"backend": "botzone"}})
    assert cfg.env.small_blind == 50
    assert cfg.env.big_blind == 100
    assert cfg.env.starting_stack == 20000

    untouched = ExperimentConfig()
    assert untouched.env.backend == "rlcard"
    assert (untouched.env.small_blind, untouched.env.big_blind, untouched.env.starting_stack) == (
        1.0,
        2.0,
        100.0,
    )
    with pytest.raises(ValueError, match="botzone requires"):
        ExperimentConfig.from_dict(
            {"env": {"backend": "botzone", "small_blind": 1, "big_blind": 2, "starting_stack": 400}}
        )

    env = _env(0)
    assert env.backend == "botzone"
    assert env.small_blind == 50
    assert env.big_blind == 100.0
    assert env.starting_stack == 20000.0
    env.reset()
    assert env.table.snapshot(env.hero).pot == 150


def test_checkpoint_roundtrip_and_cross_backend():
    env = _env(3)
    obs = env.reset()
    blob = env.state_dict()
    assert blob["backend"] == "botzone"
    assert blob["small_blind"] == 50
    assert float(blob["big_blind"]) == 100.0
    assert float(blob["starting_stack"]) == 20000.0
    assert blob["table"]["blinds"] == (50, 100)
    assert blob["table"]["starting_stack"] == 20000

    other = _env(4)
    other.load_state_dict(blob)
    assert torch.equal(obs["token_ids"], other.observe()["token_ids"])
    assert torch.equal(obs["legal_mask"], other.observe()["legal_mask"])
    action = 1 if float(obs["legal_mask"][0, 1]) > 0.5 else int(torch.where(obs["legal_mask"][0] > 0.5)[0][0])
    _, reward_a, done_a, _ = env.step(action)
    _, reward_b, done_b, _ = other.step(action)
    assert reward_a == reward_b and done_a == done_b
    if not done_a:
        assert torch.equal(env.observe()["token_ids"], other.observe()["token_ids"])
        mid = env.state_dict()
        third = _env(5)
        third.load_state_dict(mid)
        assert torch.equal(env.observe()["token_ids"], third.observe()["token_ids"])
        assert any(event.chips_added > 0 or event.action_id == 1 for event in env.table.events_for(0))

    mismatched = _env(8)
    mismatched.big_blind = 200.0
    with pytest.raises(ValueError, match="blinds/stack"):
        mismatched.load_state_dict(blob)
    broken = dict(env.table.state_dict())
    broken["blinds"] = (50, 200)
    with pytest.raises(ValueError, match="blinds"):
        make_table("botzone", seed=6).load_state_dict(broken)

    if HAS_RLCARD:
        rlcard_env = _env(6, backend="rlcard")
        rlcard_env.reset()
        with pytest.raises(ValueError, match="backend"):
            rlcard_env.load_state_dict(blob)
        with pytest.raises(ValueError, match="backend"):
            other.load_state_dict(rlcard_env.state_dict())
    if HAS_POKERKIT:
        pokerkit_env = _env(7, backend="pokerkit")
        pokerkit_env.reset()
        with pytest.raises(ValueError, match="backend"):
            pokerkit_env.load_state_dict(blob)
        with pytest.raises(ValueError, match="backend"):
            other.load_state_dict(pokerkit_env.state_dict())


def test_two_step_rollout_and_one_ppo_update(tmp_path: Path):
    from alphanl.transformer.train import run_training

    cfg = _tiny_cfg(tmp_path)
    assert cfg.env.small_blind == 1.0
    run_training(cfg)
    assert cfg.env.backend == "botzone"
    assert cfg.env.small_blind == 50.0
    assert cfg.env.big_blind == 100.0
    assert cfg.env.starting_stack == 20000.0
    text = (tmp_path / "metrics.csv").read_text()
    assert "policy_loss" in text
    ckpt = tmp_path / "ckpt" / "update_0000000.pt"
    assert ckpt.is_file()
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt, map_location="cpu")
    assert blob["config"]["env"]["backend"] == "botzone"
    assert blob["config"]["env"]["small_blind"] == 50.0
    assert blob["config"]["env"]["big_blind"] == 100.0
    assert blob["config"]["env"]["starting_stack"] == 20000.0
    saved_env = blob["envs"][0]
    assert saved_env["backend"] == "botzone"
    assert saved_env["small_blind"] == 50
    assert saved_env["big_blind"] == 100.0
    assert saved_env["starting_stack"] == 20000.0


def test_init_weights_refused_for_botzone_unless_opt_in(tmp_path: Path):
    from alphanl.transformer.train import run_training

    cfg = _tiny_cfg(tmp_path)
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="refusing --init-weights"):
        run_training(cfg, init_weights=missing)
    with pytest.raises(Exception) as opted:
        run_training(cfg, init_weights=missing, allow_cross_backend_init=True)
    assert "refusing --init-weights" not in str(opted.value)

    rlcard_cfg = _tiny_cfg(tmp_path / "rl")
    rlcard_cfg.env.backend = "rlcard"
    with pytest.raises(Exception) as rlcard_exc:
        run_training(rlcard_cfg, init_weights=missing)
    assert "refusing --init-weights" not in str(rlcard_exc.value)

    config_path = tmp_path / "botzone.json"
    payload = ExperimentConfig.from_dict({"name": "botzone_m4", "env": {"backend": "botzone"}}).to_dict()
    payload["train"]["num_updates"] = 1
    payload["train"]["eval_every"] = 0
    payload["model"] = {
        "d_model": 32,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "max_seq_len": 32,
    }
    payload["ppo"]["num_envs"] = 1
    payload["ppo"]["rollout_steps"] = 2
    payload["ppo"]["minibatch_size"] = 2
    config_path.write_text(json.dumps(payload))
    refused = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_transformer.py"),
            "--config",
            str(config_path),
            "--init-weights",
            str(missing),
            "--updates",
            "1",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "refusing --init-weights" in (refused.stderr + refused.stdout)

    allowed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_transformer.py"),
            "--config",
            str(config_path),
            "--init-weights",
            str(missing),
            "--allow-cross-backend-init",
            "--updates",
            "1",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Dry-run still has to get past the refusal. It does not open the file.
    assert allowed.returncode == 0, allowed.stderr
    assert "refusing --init-weights" not in (allowed.stderr + allowed.stdout)

"""Rollout wiring on fake envs, and one real rlcard hand when that package is installed."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.pool import OpponentPool
from alphanl.transformer.schema import HandSnapshot
from alphanl.transformer.tokenizer import NUM_NUMERIC, NUM_TYPES, VOCAB_SIZE, BettingHistoryTokenizer
from alphanl.transformer.train import collect_rollout, flatten_rollout


def _model() -> PokerTransformer:
    return PokerTransformer(
        vocab_size=VOCAB_SIZE,
        num_types=NUM_TYPES,
        num_numeric=NUM_NUMERIC,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=16,
    )


def _snap() -> HandSnapshot:
    return HandSnapshot(
        events=(),
        hole_cards=("SA", "HK"),
        board_cards=(),
        me_seat=0,
        sb_seat=0,
        street=0,
        pot=3.0,
        stacks=(99.0, 98.0),
        to_call=1.0,
        legal_action_ids=(0, 1),
        big_blind=2.0,
        starting_stack=100.0,
    )


class _ToyEnv:
    def __init__(self, tokenizer: BettingHistoryTokenizer):
        self.tokenizer = tokenizer
        self.n = 0

    def reset(self):
        self.n = 0
        return self.tokenizer.encode(_snap())

    def step(self, action):
        assert action in (0, 1)
        self.n += 1
        done = self.n >= 2
        return self.tokenizer.encode(_snap()), (0.5 if done else 0.0), done, {}

    def drain_passive_rewards(self):
        return []


def test_collect_rollout_shapes_and_gae():
    tokenizer = BettingHistoryTokenizer(max_seq_len=16)
    envs = [_ToyEnv(tokenizer), _ToyEnv(tokenizer)]
    obs = [env.reset() for env in envs]
    storage, stats = collect_rollout(envs, obs, _model(), torch.device("cpu"), 4)
    assert storage["rewards"].shape == (2, 4)
    assert storage["legal_mask"].shape == (2, 4, 5)
    assert stats["episodes"] == 4
    assert abs(stats["mean_reward_bb"] - 0.5) < 1e-6
    flat = flatten_rollout(storage, gamma=1.0, gae_lambda=1.0)
    assert flat["actions"].shape == (8,)
    assert flat["advantages"].shape == (8,)
    assert torch.isfinite(flat["advantages"]).all()
    assert torch.isfinite(flat["returns"]).all()


@pytest.mark.skipif(__import__("importlib").util.find_spec("rlcard") is None, reason="rlcard is not installed")
def test_one_ppo_update_writes_logs(tmp_path: Path):
    from alphanl.transformer.config import ExperimentConfig
    from alphanl.transformer.train import run_training

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
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 2
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 2
    cfg.ppo.entropy_coef = 0.01
    cfg.train.num_updates = 1
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 1
    cfg.train.log_every = 1
    cfg.pool.snapshot_every = 1
    cfg.pool.max_size = 2
    cfg.train.out_dir = str(tmp_path)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    run_training(cfg)
    text = (tmp_path / "metrics.csv").read_text()
    assert "entropy_coef" in text.splitlines()[0]
    assert "policy_loss" in text
    assert (tmp_path / "ckpt" / "update_0000000.pt").is_file()


def test_dry_run_cli():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "train_transformer.py"), "--dry-run"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "parameters" in proc.stdout
    assert "entropy_coef=0.01" in proc.stdout


@pytest.mark.skipif(__import__("importlib").util.find_spec("rlcard") is None, reason="rlcard is not installed")
def test_replay_chip_deltas_match_live_rlcard():
    import rlcard

    from alphanl.transformer.history import infer_sb_seat, legal_action_ids, replay_events, stage_value

    rl_env = rlcard.make("no-limit-holdem", config={"seed": 0, "game_num_players": 2, "chips_for_each": 100})
    rng = __import__("numpy").random.default_rng(1)
    for _hand in range(12):
        state, _player = rl_env.reset()
        raw = state["raw_obs"]
        sb_seat = infer_sb_seat(raw["all_chips"])
        history = [[], [], [], []]
        steps = 0
        while not rl_env.game.is_over() and steps < 30:
            legal = legal_action_ids(raw["legal_actions"])
            action = int(rng.choice(legal))
            actor = int(raw["current_player"])
            stage = stage_value(raw["stage"])
            before = [int(c) for c in raw["all_chips"]]
            history[stage].append([actor, action, list(legal)])
            state, _player = rl_env.step(action)
            raw = state["raw_obs"]
            events = replay_events(history, me_seat=actor, sb_seat=sb_seat)
            actual = int(raw["all_chips"][actor]) - before[actor]
            assert events[-1].chips_added == actual
            assert events[-1].street == stage
            assert events[-1].action_id == action
            steps += 1


@pytest.mark.skipif(__import__("importlib").util.find_spec("rlcard") is None, reason="rlcard is not installed")
def test_rlcard_hand_masks_legal_actions_and_finishes():
    import numpy as np

    from alphanl.transformer.env import PolicyActor, SelfPlayEnv

    tokenizer = BettingHistoryTokenizer(max_seq_len=64)
    model = PokerTransformer(dropout=0.0)
    actor = PolicyActor(model, PokerTransformer(dropout=0.0), tokenizer, np.random.default_rng(0))
    pool = OpponentPool(max_size=2, selfplay_prob=0.0, random_prob=1.0, seed=0)
    pool.add(model.state_dict())
    env = SelfPlayEnv(
        tokenizer,
        actor,
        pool,
        seed=3,
        record_results=False,
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    obs = env.reset()
    assert obs["legal_mask"].shape[-1] == 5
    assert int(obs["legal_mask"].sum()) >= 1
    assert int(obs["token_ids"][0, 0]) == 1
    saw_terminal = False
    for _ in range(40):
        legal = torch.where(obs["legal_mask"][0] > 0.5)[0]
        action = int(legal[0])
        obs, reward, done, info = env.step(action)
        assert info["kind"] == "random"
        if done:
            assert torch.isfinite(torch.tensor(reward))
            saw_terminal = True
            break
        assert set(torch.where(obs["legal_mask"][0] > 0.5)[0].tolist()).issubset({0, 1, 2, 3, 4})
    assert saw_terminal

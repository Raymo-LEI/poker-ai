"""M6: botzone GAE reward is chip delta / big blind / 200; logs multiply back.

rlcard and pokerkit stay in big blinds. The known hand is an immediate
small-blind fold: the hero loses the small blind, and the other seat wins it.
"""

import importlib.util

import pytest
import torch

from alphanl.botzone.legal import BIG_BLIND, SMALL_BLIND
from alphanl.transformer.env import gae_reward_depth, logged_bb_from_reward
from alphanl.transformer.ppo import compute_gae
from alphanl.transformer.train import collect_rollout, flatten_rollout

HAS_RLCARD = importlib.util.find_spec("rlcard") is not None
HAS_POKERKIT = importlib.util.find_spec("pokerkit") is not None


class _Fold:
    def act(self, snapshot, choice):
        return 0

    def prepare_pool(self, weights):
        return None


class _AlwaysFold:
    """Hero action 0, value 0. Forces the known fold without touching the net."""

    def eval(self):
        return self

    def act(self, batch, deterministic=False):
        n = int(batch["legal_mask"].shape[0])
        zeros = torch.zeros(n)
        return torch.zeros(n, dtype=torch.long), zeros, zeros, zeros

    def __call__(self, batch, mask_illegal=True):
        n = int(batch["legal_mask"].shape[0])
        return torch.zeros(n, 5), torch.zeros(n)


class _ScriptedHero:
    """Pick the small blind or the big blind after the table has posted."""

    def __init__(self, env, want_small_blind):
        self.env = env
        self.want_small_blind = list(want_small_blind)

    def integers(self, low, high):
        want = self.want_small_blind.pop(0)
        sb = int(self.env.sb_seat)
        return sb if want else 1 - sb


def _env(seed, backend, seats):
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=seed)
    env = SelfPlayEnv(
        tokenizer,
        _Fold(),
        pool,
        seed=seed,
        backend=backend,
        record_results=False,
    )
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    env.rng = _ScriptedHero(env, seats)
    return env


def test_reward_depth_is_botzone_only():
    assert gae_reward_depth("botzone") == 200.0
    assert gae_reward_depth("rlcard") == 1.0
    assert gae_reward_depth("pokerkit") == 1.0
    assert logged_bb_from_reward(-0.5, "rlcard") == pytest.approx(-0.5)
    assert logged_bb_from_reward(-0.5, "pokerkit") == pytest.approx(-0.5)


def test_botzone_sb_fold_reward_and_logged_bb():
    """SB folds. Chip delta is -50. GAE reward is -50/100/200; log is x200."""

    env = _env(1, "botzone", [True])
    env.reset()
    assert env.hero == env.sb_seat
    assert 0 in env.table.legal_action_ids()
    _, reward, done, _info = env.step(0)
    assert done
    chip_delta = float(env.table.payoffs()[env.hero])
    assert chip_delta == -float(SMALL_BLIND)
    assert env.big_blind == float(BIG_BLIND)
    expected = chip_delta / env.big_blind / 200.0
    assert reward == pytest.approx(expected)
    assert reward == pytest.approx(-0.0025)
    logged = logged_bb_from_reward(reward, "botzone")
    assert logged == pytest.approx(reward * 200.0)
    assert logged == pytest.approx(chip_delta / env.big_blind)
    assert logged == pytest.approx(-0.5)


def test_botzone_passive_bb_fold_is_scaled_the_same_way():
    """BB never acts: SB folds, BB wins 50. The stored reward is still /200."""

    env = _env(2, "botzone", [False, True])
    env.reset()
    raw = env.drain_passive_rewards()
    assert len(raw) == 1
    expected = float(SMALL_BLIND) / float(BIG_BLIND) / 200.0
    assert raw[0] == pytest.approx(expected)
    assert logged_bb_from_reward(raw[0], "botzone") == pytest.approx(raw[0] * 200.0)
    assert logged_bb_from_reward(raw[0], "botzone") == pytest.approx(0.5)


def test_botzone_rollout_logs_bb_and_gae_sees_scaled_reward():
    """One hero fold enters GAE at /200. The rollout stat reports bb = reward x 200.

    The hand after that fold is a passive BB win, so the passive log is checked too.
    """

    env = _env(3, "botzone", [True, False, True])
    obs = [env.reset()]
    storage, stats = collect_rollout([env], obs, _AlwaysFold(), torch.device("cpu"), 1)
    gae_reward = float(storage["rewards"][0, 0])
    expected = -float(SMALL_BLIND) / float(BIG_BLIND) / 200.0
    assert float(storage["dones"][0, 0]) == 1.0
    assert gae_reward == pytest.approx(expected)
    assert stats["episodes"] == 1
    assert stats["mean_reward_bb"] == pytest.approx(gae_reward * 200.0)
    assert stats["mean_reward_bb"] == pytest.approx(-0.5)
    assert stats["passive_hands"] == 1
    assert stats["passive_mean_bb"] == pytest.approx(0.5)

    flat = flatten_rollout(storage, gamma=1.0, gae_lambda=1.0)
    assert float(flat["advantages"][0]) == pytest.approx(gae_reward)
    assert float(flat["returns"][0]) == pytest.approx(gae_reward)
    advantages, returns = compute_gae(
        storage["rewards"],
        storage["old_values"],
        storage["dones"],
        storage["bootstrap"],
        1.0,
        1.0,
    )
    assert float(advantages[0, 0]) == pytest.approx(gae_reward)
    assert float(returns[0, 0]) == pytest.approx(gae_reward)


@pytest.mark.skipif(not HAS_POKERKIT, reason="pokerkit is not installed")
def test_pokerkit_sb_fold_stays_in_big_blinds():
    env = _env(4, "pokerkit", [True])
    env.reset()
    assert env.hero == env.sb_seat
    assert 0 in env.table.legal_action_ids()
    _, reward, done, _info = env.step(0)
    assert done
    chip_delta = float(env.table.payoffs()[env.hero])
    assert chip_delta == -float(env.small_blind)
    assert reward == pytest.approx(chip_delta / env.big_blind)
    assert logged_bb_from_reward(reward, "pokerkit") == pytest.approx(reward)

    logged_env = _env(5, "pokerkit", [True, True])
    obs = [logged_env.reset()]
    storage, stats = collect_rollout([logged_env], obs, _AlwaysFold(), torch.device("cpu"), 1)
    gae_reward = float(storage["rewards"][0, 0])
    assert gae_reward == pytest.approx(chip_delta / logged_env.big_blind)
    assert stats["mean_reward_bb"] == pytest.approx(gae_reward)
    assert stats["passive_hands"] == 0


@pytest.mark.skipif(not HAS_RLCARD, reason="rlcard is not installed")
def test_rlcard_sb_fold_stays_in_big_blinds():
    env = _env(6, "rlcard", [True])
    env.reset()
    assert env.hero == env.sb_seat
    assert 0 in env.table.legal_action_ids()
    _, reward, done, _info = env.step(0)
    assert done
    chip_delta = float(env.table.payoffs()[env.hero])
    assert chip_delta == -float(env.small_blind)
    assert reward == pytest.approx(chip_delta / env.big_blind)
    assert logged_bb_from_reward(reward, "rlcard") == pytest.approx(reward)

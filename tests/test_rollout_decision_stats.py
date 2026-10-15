"""M11a: rollout free-check fold ratio and legal entropy logged on metrics.csv.

The hand-built case below fixes the masks, the chips owed, and the sampled
actions. Expected entropy is the Shannon formula in nats. The ratio counts
the stored action, including a fold that is not the greedy choice.

Bit-for-bit: the same tiny CPU run (3 updates, dropout 0, AMP off) is
executed twice in this process. The second run monkeypatches
``_log_decision_stats_enabled`` to false, which skips the to_call read and
the reduction. ``model.act`` is still called once per decision, in the same
order, and its sample is not redrawn. Loss columns and the final
``state_dict`` must match. A stored golden file is not used: a CPU
state_dict from another machine is not a stable oracle.
"""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from alphanl.transformer.decision_stats import (
    DECISION_STAT_COLUMNS,
    empty_decision_stats,
    is_free_check,
    summarize_rollout_decisions,
)
from alphanl.transformer.model import entropy_from_probs, masked_log_probs
from alphanl.transformer.schema import HandSnapshot
from alphanl.transformer.tokenizer import BettingHistoryTokenizer
from alphanl.transformer.train import _append_csv, _read_to_call

_OLD_METRIC_COLUMNS = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "mean_reward_bb",
    "episodes",
    "update",
)


def _hand_rows():
    """Six decisions. Row 0 is a sampled fold against a check-preferring policy.

    Row 3 is a forced fold while a check is free. It must not enter the
    ratio or the entropy mean. Counting it would make the ratio 3/4.
    """

    mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],  # free check; sampled fold, greedy check
            [1, 1, 1, 0, 0],  # free check; sampled check
            [1, 1, 0, 0, 0],  # owes chips; sampled fold
            [1, 0, 0, 0, 0],  # forced fold, free check, excluded
            [1, 1, 0, 0, 0],  # free check; sampled fold
            [0, 1, 1, 0, 0],  # nothing owed, fold illegal
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 1, 0, 0, 0, 1], dtype=torch.long)
    to_call = torch.tensor([0, 0, 5, 0, 0, 0], dtype=torch.float32)
    logits = torch.zeros(6, 5)
    logits[0, 1] = 5.0
    return mask, actions, to_call, logits


def _analytic_entropy(mask: torch.Tensor, logits: torch.Tensor) -> list:
    """Hand Shannon entropy of each masked row, in nats, from the logits."""

    expected = []
    for row in range(int(mask.shape[0])):
        legal = [index for index, flag in enumerate(mask[row].tolist()) if float(flag) > 0.5]
        scores = [float(logits[row, index]) for index in legal]
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        total = sum(weights)
        probs = [weight / total for weight in weights]
        expected.append(-sum(prob * math.log(prob) for prob in probs if prob > 0.0))
    return expected


def test_is_free_check_matches_the_collapse_definition():
    mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.float32)
    assert is_free_check(0.0, mask)
    assert not is_free_check(1.0, mask)
    assert not is_free_check(0.0, torch.tensor([0, 1, 1, 0, 0], dtype=torch.float32))


def test_hand_built_ratio_and_legal_entropy():
    mask, actions, to_call, logits = _hand_rows()
    _, log_probs, probs, legal = masked_log_probs(logits, mask)
    entropy = entropy_from_probs(probs, log_probs, legal)
    analytic = _analytic_entropy(mask, logits)
    for index, value in enumerate(analytic):
        assert float(entropy[index]) == pytest.approx(value, rel=1e-5, abs=1e-6)

    # Greedy on row 0 is check. The stored sample is fold, and that is what counts.
    assert int(torch.argmax(probs[0]).item()) == 1
    assert int(actions[0]) == 0

    stats = summarize_rollout_decisions(actions, mask, entropy, to_call)
    kept = (0, 1, 2, 4, 5)
    expected_entropy = sum(analytic[index] for index in kept) / len(kept)
    assert stats["free_check_fold_ratio"] == pytest.approx(2 / 3)
    assert stats["free_check_count"] == 3
    assert stats["legal_entropy"] == pytest.approx(expected_entropy, rel=1e-5, abs=1e-6)
    # Forced free-check fold (row 3) is outside the mean. Including it
    # changes the denominator from 5 to 6.
    with_forced = (sum(analytic[index] for index in kept) + analytic[3]) / 6
    assert stats["legal_entropy"] != pytest.approx(with_forced, abs=1e-4)


def test_missing_to_call_is_nan_and_does_not_drop_entropy():
    mask, actions, to_call, logits = _hand_rows()
    _, log_probs, probs, legal = masked_log_probs(logits, mask)
    entropy = entropy_from_probs(probs, log_probs, legal)
    analytic = _analytic_entropy(mask, logits)
    expected_entropy = sum(analytic[index] for index in (0, 1, 2, 4, 5)) / 5

    missing = summarize_rollout_decisions(actions, mask, entropy, None)
    assert math.isnan(missing["free_check_fold_ratio"])
    assert math.isnan(missing["free_check_count"])
    assert missing["legal_entropy"] == pytest.approx(expected_entropy, rel=1e-5, abs=1e-6)

    poisoned = to_call.clone()
    poisoned[0] = float("nan")
    mixed = summarize_rollout_decisions(actions, mask, entropy, poisoned)
    assert math.isnan(mixed["free_check_fold_ratio"])
    assert math.isnan(mixed["free_check_count"])
    assert mixed["legal_entropy"] == pytest.approx(expected_entropy, rel=1e-5, abs=1e-6)

    owed = torch.full((6,), 2.0)
    quiet = summarize_rollout_decisions(actions, mask, entropy, owed)
    assert math.isnan(quiet["free_check_fold_ratio"])
    assert quiet["free_check_count"] == 0
    assert quiet["legal_entropy"] == pytest.approx(expected_entropy, rel=1e-5, abs=1e-6)


def test_forced_decisions_alone_log_nan_entropy_and_zero_count():
    mask = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)
    actions = torch.tensor([0, 1])
    entropy = torch.zeros(2)
    stats = summarize_rollout_decisions(actions, mask, entropy, torch.zeros(2))
    assert math.isnan(stats["legal_entropy"])
    assert math.isnan(stats["free_check_fold_ratio"])
    assert stats["free_check_count"] == 0
    assert set(empty_decision_stats()) == set(DECISION_STAT_COLUMNS)


def test_read_to_call_uses_snapshot_or_returns_none():
    class _Snap:
        to_call = 7.0

    class _Table:
        def snapshot(self, seat):
            assert seat == 1
            return _Snap()

    class _Env:
        table = _Table()
        hero = 1

    assert _read_to_call(_Env()) == 7.0
    assert _read_to_call(object()) is None

    class _NoHero:
        table = _Table()

    assert _read_to_call(_NoHero()) is None


def test_botzone_rollout_has_a_numeric_free_check_count():
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool
    from alphanl.transformer.train import collect_rollout

    class _Fold:
        def act(self, snapshot, choice):
            return 0

        def prepare_pool(self, weights):
            return None

        def eval(self):
            return self

        def act_policy(self, batch, deterministic=False):
            n = int(batch["legal_mask"].shape[0])
            zeros = torch.zeros(n)
            return torch.zeros(n, dtype=torch.long), zeros, zeros, zeros

        def __call__(self, batch, mask_illegal=True):
            n = int(batch["legal_mask"].shape[0])
            return torch.zeros(n, 5), torch.zeros(n)

    # collect_rollout calls model.act, not the table actor.
    class _HeroNet(_Fold):
        def act(self, batch, deterministic=False):
            return self.act_policy(batch, deterministic)

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)
    env = SelfPlayEnv(tokenizer, _Fold(), pool, seed=0, backend="botzone", record_results=False)
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    obs = [env.reset()]
    assert _read_to_call(env) == pytest.approx(float(env.table.snapshot(env.hero).to_call))
    _storage, stats = collect_rollout([env], obs, _HeroNet(), torch.device("cpu"), 1)
    assert not math.isnan(float(stats["free_check_count"]))
    assert stats["free_check_count"] >= 0


@pytest.mark.skipif(importlib.util.find_spec("rlcard") is None, reason="rlcard is not installed")
def test_rlcard_to_call_comes_from_the_table():
    _assert_backend_to_call("rlcard")


@pytest.mark.skipif(importlib.util.find_spec("pokerkit") is None, reason="pokerkit is not installed")
def test_pokerkit_to_call_comes_from_the_table():
    _assert_backend_to_call("pokerkit")


def _assert_backend_to_call(backend: str) -> None:
    from alphanl.transformer.env import SelfPlayEnv
    from alphanl.transformer.pool import OpponentPool

    class _Fold:
        def act(self, snapshot, choice):
            return 0

        def prepare_pool(self, weights):
            return None

    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)
    env = SelfPlayEnv(tokenizer, _Fold(), pool, seed=0, backend=backend, record_results=False)
    env.force_opponent = {"kind": "random", "policy_id": None, "weights": None}
    env.reset()
    assert _read_to_call(env) == pytest.approx(float(env.table.snapshot(env.hero).to_call))


class _ToyEnv:
    """Tokenizer obs with a real to_call baked into tokens, and no table.

    The logger must not recover chips owed from the numeric channel.
    """

    def __init__(self):
        self.tokenizer = BettingHistoryTokenizer(max_seq_len=16)
        self.n = 0

    def reset(self):
        self.n = 0
        return self.tokenizer.encode(_snap())

    def step(self, action):
        self.n += 1
        return self.tokenizer.encode(_snap()), 0.0, True, {}

    def drain_passive_rewards(self):
        return []


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
        to_call=0.0,
        legal_action_ids=(0, 1),
        big_blind=2.0,
        starting_stack=100.0,
    )


def test_env_without_a_table_logs_nan_even_if_tokens_encode_zero_to_call():
    from alphanl.transformer.train import collect_rollout

    class _Net:
        def eval(self):
            return self

        def act(self, batch, deterministic=False):
            n = int(batch["legal_mask"].shape[0])
            zeros = torch.zeros(n)
            return torch.zeros(n, dtype=torch.long), zeros, zeros, torch.full((n,), 0.25)

        def __call__(self, batch, mask_illegal=True):
            n = int(batch["legal_mask"].shape[0])
            return torch.zeros(n, 5), torch.zeros(n)

    env = _ToyEnv()
    obs = [env.reset()]
    assert float(obs[0]["numeric"][0, 0, 3]) == 0.0
    _storage, stats = collect_rollout([env], obs, _Net(), torch.device("cpu"), 1)
    assert math.isnan(stats["free_check_fold_ratio"])
    assert math.isnan(stats["free_check_count"])
    assert stats["legal_entropy"] == pytest.approx(0.25)


def test_append_csv_keeps_an_old_header(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    old_header = "update,entropy,eval_random_bb"
    path.write_text(old_header + "\n0,1.5,\n")
    _append_csv(
        path,
        {
            "update": 1,
            "entropy": 1.25,
            "eval_random_bb": "",
            "free_check_fold_ratio": float("nan"),
            "legal_entropy": 0.5,
            "free_check_count": 0,
        },
    )
    lines = path.read_text().splitlines()
    assert lines[0] == old_header
    assert lines[1] == "0,1.5,"
    assert len(lines) == 3
    parsed = next(csv.DictReader([lines[0], lines[2]]))
    # DictReader keeps the original header. Extra trailing fields do not raise.
    assert parsed["update"] == "1"
    assert parsed["entropy"] == "1.25"
    assert "nan" in lines[2]


def _tiny_cfg(tmp_path: Path, updates: int):
    from alphanl.transformer.config import ExperimentConfig

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
    cfg.train.amp = False
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 4
    cfg.ppo.epochs = 1
    cfg.ppo.minibatch_size = 4
    cfg.train.num_updates = updates
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 0
    cfg.train.log_every = 1
    cfg.pool.snapshot_every = 0
    cfg.pool.max_size = 2
    cfg.train.out_dir = str(tmp_path)
    cfg.train.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rows(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def test_fresh_metrics_append_the_new_columns(tmp_path: Path):
    from alphanl.transformer.train import run_training

    run_training(_tiny_cfg(tmp_path, 1))
    rows, header = _rows(tmp_path / "metrics.csv")
    assert header[-3:] == list(DECISION_STAT_COLUMNS)
    assert "entropy" in header
    assert header.index("entropy") < header.index("legal_entropy")
    assert len(rows) == 1
    assert rows[0]["entropy"] != ""
    assert not math.isnan(float(rows[0]["free_check_count"]))


def test_resume_into_old_metrics_does_not_crash(tmp_path: Path):
    from alphanl.transformer.train import run_training

    path = tmp_path / "metrics.csv"
    old_header = (
        "update,timesteps,episodes,mean_reward_bb,passive_hands,pool_size,"
        "entropy_coef,seconds,loss,policy_loss,value_loss,entropy,approx_kl,"
        "clip_fraction,explained_variance,eval_random_bb"
    )
    path.write_text(old_header + "\n0,1,0,0,0,1,0.01,0.1,1,1,1,1,0,0,0,\n")
    run_training(_tiny_cfg(tmp_path, 1))
    lines = path.read_text().splitlines()
    assert lines[0] == old_header
    assert len(lines) == 3
    assert lines[2].count(",") > lines[0].count(",")


def _equal_tree(left, right, path: str) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert torch.equal(left.cpu(), right.cpu()), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _equal_tree(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (item_l, item_r) in enumerate(zip(left, right)):
            _equal_tree(item_l, item_r, f"{path}[{index}]")
        return
    try:
        import numpy as np

        if isinstance(left, np.ndarray):
            assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
            return
    except ImportError:
        pass
    assert left == right, path


def test_logging_matches_training_with_the_reduction_disabled(tmp_path: Path):
    """Enabled logging vs the monkeypatched pre-log path. Same seed, CPU, no AMP."""

    # Import the lazy backends and tensorboard once, before either seeded run,
    # so the second run does not pay a first-import that could touch RNG.
    import alphanl.transformer.botzone_table  # noqa: F401
    import alphanl.transformer.train as train_mod

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        pass

    original = train_mod._log_decision_stats_enabled

    def _run(out: Path, enabled: bool):
        train_mod._log_decision_stats_enabled = (lambda: True) if enabled else (lambda: False)
        train_mod.run_training(_tiny_cfg(out, 3))
        rows, header = _rows(out / "metrics.csv")
        blob = _load_checkpoint(out / "ckpt" / "update_0000002.pt")
        return rows, header, blob

    try:
        rows_on, header_on, blob_on = _run(tmp_path / "on", True)
        rows_off, header_off, blob_off = _run(tmp_path / "off", False)
    finally:
        train_mod._log_decision_stats_enabled = original

    assert header_on == header_off
    assert header_on[-3:] == list(DECISION_STAT_COLUMNS)
    assert len(rows_on) == 3 and len(rows_off) == 3
    skip = {"seconds", *DECISION_STAT_COLUMNS}
    for update, (left, right) in enumerate(zip(rows_on, rows_off)):
        for key in _OLD_METRIC_COLUMNS:
            assert left[key] == right[key], f"update {update} {key}: {left[key]!r} != {right[key]!r}"
        for key in left:
            if key not in skip:
                assert left[key] == right[key], f"update {update} {key}"
        assert math.isnan(float(right["free_check_fold_ratio"]))
        assert math.isnan(float(right["legal_entropy"]))
        assert math.isnan(float(right["free_check_count"]))
    assert any(not math.isnan(float(row["legal_entropy"])) for row in rows_on)
    assert all(not math.isnan(float(row["free_check_count"])) for row in rows_on)
    _equal_tree(blob_on["model"], blob_off["model"], "model")
    _equal_tree(blob_on["rng"], blob_off["rng"], "rng")

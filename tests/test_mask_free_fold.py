"""T-M13a: optional policy mask that drops fold on a free check.

Environment legality is unchanged. The switch defaults to false. With it
on, rollout sampling and the PPO re-forward use the same mask, and
``free_check_fold_ratio`` is 0 because the stored action is never fold.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch

from alphanl.botzone.actions import (
    CHECK_CALL_ACTION,
    FOLD_ACTION,
    NUM_ACTIONS,
    legal_action_ids,
    response_for_action,
)
from alphanl.botzone.legal import to_call
from alphanl.botzone.rule_bot import BotzoneRuleBot
from alphanl.botzone.rule_eval import uniform_legal_action
from alphanl.botzone.table import BotzoneTable
from alphanl.transformer.checkpoint import checkpoint_path, save_checkpoint
from alphanl.transformer.config import ExperimentConfig
from alphanl.transformer.decision_stats import policy_action_mask, summarize_rollout_decisions
from alphanl.transformer.env import SelfPlayEnv
from alphanl.transformer.pool import OpponentPool
from alphanl.transformer.tokenizer import BettingHistoryTokenizer

ROOT = Path(__file__).resolve().parents[1]
LAPTOP = ROOT / "configs" / "transformer" / "botzone_laptop_8gb.json"
MASKFOLD = ROOT / "configs" / "transformer" / "botzone_laptop_8gb_maskfold.json"


def _load_script(name: str):
    path = ROOT / "eval" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLAPSE = _load_script("eval_botzone_collapse.py")
CHECKPOINT = _load_script("eval_botzone_checkpoint.py")


def _ids_mask(ids) -> torch.Tensor:
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for action in ids:
        mask[int(action)] = 1.0
    return mask


def test_policy_action_mask_false_returns_the_same_tensor():
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    assert policy_action_mask(mask, 0.0, False) is mask
    assert policy_action_mask(mask, 5.0, False) is mask


def test_policy_action_mask_drops_only_free_fold_when_check_is_open():
    free = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],  # check closed: leave fold
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    owed = torch.tensor([0.0, 0.0, 0.0, 4.0])
    masked = policy_action_mask(free, owed, True)
    off = policy_action_mask(free, owed, False)
    assert torch.equal(off, free)
    assert masked[0, FOLD_ACTION] == 0.0
    assert masked[0, CHECK_CALL_ACTION] == 1.0
    assert torch.equal(masked[0, 1:], free[0, 1:])
    assert masked[1, FOLD_ACTION] == 0.0
    assert torch.equal(masked[2], free[2])
    assert torch.equal(masked[3], free[3])


def test_fold_facing_ratio_is_zero_free_folds_when_actions_never_fold_there():
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],  # free check, action check
            [1.0, 1.0, 1.0, 0.0, 0.0],  # facing, action fold
            [1.0, 1.0, 0.0, 0.0, 0.0],  # free check, action check
        ]
    )
    actions = torch.tensor([1, 0, 1])
    owed = torch.tensor([0.0, 8.0, 0.0])
    entropy = torch.zeros(3)
    stats = summarize_rollout_decisions(actions, mask, entropy, owed)
    assert stats["free_check_fold_ratio"] == 0.0
    assert stats["free_check_count"] == 2
    assert stats["fold_facing_ratio"] == pytest.approx(1.0)

    none_facing = summarize_rollout_decisions(
        torch.tensor([1, 1]),
        mask[:2].clone(),
        torch.zeros(2),
        torch.zeros(2),
    )
    # Second row of this slice still has to_call 0 after we pass zeros.
    assert math.isnan(none_facing["fold_facing_ratio"])


def test_random_hands_mask_fold_only_on_a_free_check():
    """1000 hands of uniform legal ids. Env fold stays legal on a free check."""

    table = BotzoneTable(rng=random.Random(0))
    action_rng = random.Random(1)
    n_free = 0
    n_facing = 0
    n_all_in = 0
    for _ in range(1000):
        table.start_hand()
        guard = 0
        while not table.hand_over:
            guard += 1
            assert guard < 40
            state = table.state
            assert state is not None
            ids = legal_action_ids(state)
            env_mask = _ids_mask(ids)
            owed = float(to_call(state))
            on = policy_action_mask(env_mask, owed, True)
            off = policy_action_mask(env_mask, owed, False)
            assert torch.equal(off, env_mask)
            if any(state.all_in):
                n_all_in += 1
                assert torch.equal(on, env_mask)
                assert CHECK_CALL_ACTION not in ids
                assert FOLD_ACTION in ids
                assert set(ids) <= {FOLD_ACTION, 4}
            elif owed == 0.0:
                n_free += 1
                assert FOLD_ACTION in ids
                assert CHECK_CALL_ACTION in ids
                assert -1 in table.legal_responses()
                assert float(on[FOLD_ACTION]) == 0.0
                assert float(on[CHECK_CALL_ACTION]) > 0.5
                assert torch.equal(on[1:], env_mask[1:])
            else:
                n_facing += 1
                assert torch.equal(on, off)
            action = uniform_legal_action(state, action_rng)
            response = response_for_action(state, action)
            assert response is not None
            table.act(int(response))
    assert n_free > 0
    assert n_facing > 0
    assert n_all_in > 0


def test_all_in_response_set_is_unchanged_by_the_policy_mask():
    table = BotzoneTable(rng=random.Random(0))
    table.start_hand(dealer=0)
    assert -2 in table.legal_responses()
    table.act(-2)
    assert table.legal_responses() == {-1, -2}
    state = table.state
    assert state is not None
    ids = legal_action_ids(state)
    assert set(ids) == {FOLD_ACTION, 4}
    owed = float(to_call(state))
    assert owed > 0.0
    env_mask = _ids_mask(ids)
    assert torch.equal(policy_action_mask(env_mask, owed, True), env_mask)
    assert torch.equal(policy_action_mask(env_mask, owed, False), env_mask)


def test_rule_bot_and_random_still_see_env_fold_on_a_free_check():
    table = BotzoneTable(rng=random.Random(0))
    table.start_hand(dealer=0)
    assert to_call(table.state) == 50
    table.act(0)
    state = table.state
    assert state is not None
    assert to_call(state) == 0
    ids = legal_action_ids(state)
    assert FOLD_ACTION in ids
    assert CHECK_CALL_ACTION in ids
    action = BotzoneRuleBot().action_id(table)
    assert action in ids
    drawn = {uniform_legal_action(state, random.Random(i)) for i in range(40)}
    assert FOLD_ACTION in drawn
    assert CHECK_CALL_ACTION in drawn


def test_random_opponent_batch_keeps_fold_and_policy_batch_drops_it():
    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)

    class _Call:
        def act(self, snapshot, choice):
            legal = list(snapshot.legal_action_ids)
            return 1 if 1 in legal else int(legal[0])

        def prepare_pool(self, weights, policy_id=None, live_ids=None):
            return None

    env = SelfPlayEnv(
        tokenizer,
        _Call(),
        pool,
        seed=1,
        backend="botzone",
        record_results=False,
        mask_free_fold=True,
    )
    env.force_opponent = {"kind": "self", "policy_id": None, "weights": None}
    env.reset()
    found = False
    for _ in range(40):
        if env.table.is_over():
            env.reset()
            continue
        seat = int(env.table.current_player())
        snap = env.table.snapshot(seat)
        if snap.to_call == 0.0 and FOLD_ACTION in snap.legal_action_ids and CHECK_CALL_ACTION in snap.legal_action_ids:
            batch = tokenizer.encode(snap)
            env.choice = {"kind": "random", "policy_id": None, "weights": None}
            raw = env._policy_batch(batch, seat)
            assert float(raw["legal_mask"][0, FOLD_ACTION]) > 0.5
            env.choice = {"kind": "self", "policy_id": None, "weights": None}
            masked = env._policy_batch(batch, seat)
            assert float(masked["legal_mask"][0, FOLD_ACTION]) == 0.0
            assert float(masked["legal_mask"][0, CHECK_CALL_ACTION]) > 0.5
            env.mask_free_fold = False
            assert env._policy_batch(batch, seat)["legal_mask"] is batch["legal_mask"]
            found = True
            break
        legal = env.table.legal_action_ids()
        env.table.step(1 if 1 in legal else int(legal[0]))
    assert found


class _FirstLegal:
    def eval(self):
        return self

    def act(self, batch, deterministic=False):
        actions = (batch["legal_mask"] > 0.5).to(torch.float32).argmax(dim=-1).to(torch.long)
        zeros = torch.zeros(actions.shape[0])
        return actions, zeros, zeros, zeros

    def __call__(self, batch, mask_illegal=True):
        n = int(batch["legal_mask"].shape[0])
        return torch.zeros(n, 5), torch.zeros(n)


class _CallActor:
    def act(self, snapshot, choice):
        legal = list(snapshot.legal_action_ids)
        return 1 if 1 in legal else int(legal[0])

    def prepare_pool(self, weights, policy_id=None, live_ids=None):
        return None


def _one_env(seed: int, mask: bool) -> SelfPlayEnv:
    tokenizer = BettingHistoryTokenizer(max_seq_len=32)
    pool = OpponentPool(max_size=1, selfplay_prob=0.0, random_prob=1.0, seed=0)
    env = SelfPlayEnv(
        tokenizer,
        _CallActor(),
        pool,
        seed=seed,
        backend="botzone",
        record_results=False,
        mask_free_fold=mask,
    )
    env.force_opponent = {"kind": "self", "policy_id": None, "weights": None}
    return env


def test_switch_on_logs_zero_free_check_folds_from_the_rollout():
    from alphanl.transformer.train import collect_rollout

    env = _one_env(2, True)
    obs = [env.reset()]
    _storage, stats = collect_rollout([env], obs, _FirstLegal(), torch.device("cpu"), 8, mask_free_fold=True)
    assert stats["free_check_count"] > 0
    assert stats["free_check_fold_ratio"] == 0.0

    plain = _one_env(2, False)
    obs = [plain.reset()]
    _storage, unmasked = collect_rollout(
        [plain], obs, _FirstLegal(), torch.device("cpu"), 8, mask_free_fold=False
    )
    assert unmasked["free_check_count"] > 0
    assert unmasked["free_check_fold_ratio"] == pytest.approx(1.0)


def test_rollout_mask_matches_the_ppo_reforward():
    from alphanl.transformer.model import PokerTransformer, masked_log_probs
    from alphanl.transformer.ppo import ppo_losses
    from alphanl.transformer.train import collect_rollout, flatten_rollout, set_seed

    set_seed(0)
    model = PokerTransformer(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=32,
    )
    model.eval()
    seen = []
    real_act = model.act

    def _wrapped(batch, deterministic=False):
        seen.append(batch["legal_mask"].detach().cpu().clone())
        return real_act(batch, deterministic=deterministic)

    model.act = _wrapped
    env = _one_env(3, True)
    obs = [env.reset()]
    storage, stats = collect_rollout([env], obs, model, torch.device("cpu"), 6, mask_free_fold=True)
    assert len(seen) == 6
    assert "to_call" in storage
    rollout_mask = torch.stack(seen, dim=1)
    update_mask = policy_action_mask(storage["legal_mask"], storage["to_call"], True)
    assert torch.equal(rollout_mask, update_mask)
    owed = storage["to_call"]
    env_mask = storage["legal_mask"]
    free = (owed == 0) & (env_mask[..., CHECK_CALL_ACTION] > 0.5)
    facing = owed > 0
    assert int(free.sum()) > 0
    assert torch.all(update_mask[..., FOLD_ACTION][free] == 0)
    assert torch.all(env_mask[..., FOLD_ACTION][free] > 0.5)
    assert torch.equal(update_mask[facing], env_mask[facing])
    assert stats["free_check_count"] > 0
    assert stats["free_check_fold_ratio"] == 0.0

    model.act = real_act
    flat = flatten_rollout(storage, 1.0, 0.95)
    flat_update = policy_action_mask(flat["legal_mask"], flat["to_call"], True)
    assert torch.equal(flat_update, rollout_mask.reshape(-1, NUM_ACTIONS))
    skip = {
        "actions",
        "old_logp",
        "old_values",
        "advantages",
        "returns",
        "rewards",
        "dones",
        "to_call",
    }
    batch = {key: value for key, value in flat.items() if key not in skip}
    # One row at a time matches the rollout's batch of 1. A joint forward of
    # every step can differ by one float32 ulp inside the encoder; that is
    # not a mask mismatch (the masks above are torch.equal).
    with torch.no_grad():
        recomputed = []
        for row in range(flat_update.shape[0]):
            one = {key: value[row : row + 1] for key, value in batch.items()}
            one["legal_mask"] = flat_update[row : row + 1]
            logits, _value = model(one, mask_illegal=True)
            _masked, logp, _probs, _legal = masked_log_probs(logits, one["legal_mask"])
            recomputed.append(logp.gather(1, flat["actions"][row : row + 1].long().unsqueeze(1)).squeeze(1))
        got = torch.cat(recomputed)
    assert torch.equal(got, flat["old_logp"])
    _loss, metrics = ppo_losses(
        model,
        batch,
        actions=flat["actions"],
        old_logp=flat["old_logp"],
        old_values=flat["old_values"],
        advantages=flat["advantages"],
        returns=flat["returns"],
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        mask_free_fold=True,
        to_call=flat["to_call"],
    )
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)


def test_maskfold_config_is_the_laptop_config_plus_the_switch_and_dirs():
    base = json.loads(LAPTOP.read_text())
    masked = json.loads(MASKFOLD.read_text())
    assert base["ppo"]["entropy_coef"] == 0.01
    assert masked["ppo"]["entropy_coef"] == 0.01
    assert "mask_free_fold" not in base["env"]
    assert masked["env"]["mask_free_fold"] is True
    assert masked["train"]["out_dir"] == "runs/transformer_botzone_laptop_maskfold"
    assert masked["train"]["checkpoint_dir"] == "runs/transformer_botzone_laptop_maskfold/ckpt"
    edited = json.loads(LAPTOP.read_text())
    edited["env"]["mask_free_fold"] = True
    edited["train"]["out_dir"] = masked["train"]["out_dir"]
    edited["train"]["checkpoint_dir"] = masked["train"]["checkpoint_dir"]
    assert masked == edited
    loaded = ExperimentConfig.from_json(MASKFOLD)
    assert loaded.env.mask_free_fold is True
    assert loaded.ppo.entropy_coef == 0.01
    assert ExperimentConfig.from_json(LAPTOP).env.mask_free_fold is False


def test_checkpoint_without_the_key_means_false_and_cli_overrides(tmp_path: Path):
    cfg = ExperimentConfig.from_json(LAPTOP)
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    raw = cfg.to_dict()
    assert raw["env"]["mask_free_fold"] is False
    del raw["env"]["mask_free_fold"]
    assert ExperimentConfig.from_dict(raw).env.mask_free_fold is False
    value, how = COLLAPSE.resolve_mask_free_fold(None, raw)
    assert value is False
    assert "default false" in how
    forced, forced_how = COLLAPSE.resolve_mask_free_fold(True, raw)
    assert forced is True
    assert "--mask-free-fold" in forced_how
    off, off_how = COLLAPSE.resolve_mask_free_fold(False, {"env": {"mask_free_fold": True}})
    assert off is False
    assert "--no-mask-free-fold" in off_how

    from alphanl.transformer.train import build_model

    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint_path(ckpt_dir, 0),
        model=model,
        optimizer=None,
        update=0,
        config=raw,
    )
    _model, _tok, loaded, _path, update = COLLAPSE.load_botzone_model(ckpt_dir, torch.device("cpu"))
    assert update == 0
    assert loaded.env.mask_free_fold is False
    assert "mask_free_fold" not in loaded.checkpoint_config["env"]
    recorded = cfg.to_dict()
    recorded["env"]["mask_free_fold"] = True
    save_checkpoint(
        checkpoint_path(ckpt_dir, 1),
        model=model,
        optimizer=None,
        update=1,
        config=recorded,
    )
    _model, _tok, loaded_on, _path, update_on = COLLAPSE.load_botzone_model(ckpt_dir, torch.device("cpu"))
    assert update_on == 1
    assert loaded_on.env.mask_free_fold is True
    assert loaded_on.checkpoint_config["env"]["mask_free_fold"] is True


def test_eval_flags_default_to_the_checkpoint_and_print_the_value(tmp_path: Path, capsys):
    parser = COLLAPSE.build_parser()
    assert parser.parse_args([]).mask_free_fold is None
    assert parser.parse_args(["--mask-free-fold"]).mask_free_fold is True
    assert parser.parse_args(["--no-mask-free-fold"]).mask_free_fold is False
    other = CHECKPOINT.build_parser()
    assert other.parse_args([]).mask_free_fold is None
    assert other.parse_args(["--mask-free-fold"]).mask_free_fold is True
    assert other.parse_args(["--no-mask-free-fold"]).mask_free_fold is False

    cfg = ExperimentConfig.from_json(LAPTOP)
    cfg.model.d_model = 32
    cfg.model.nhead = 4
    cfg.model.num_layers = 1
    cfg.model.dim_feedforward = 64
    cfg.model.max_seq_len = 32
    raw = cfg.to_dict()
    del raw["env"]["mask_free_fold"]
    from alphanl.transformer.train import build_model

    model = build_model(cfg)
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(checkpoint_path(ckpt_dir, 0), model=model, optimizer=None, update=0, config=raw)
    collapse_out = tmp_path / "collapse.json"
    COLLAPSE.main(
        [
            "--ckpt-dir",
            str(ckpt_dir),
            "--decisions",
            "4",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--mask-free-fold",
            "--out",
            str(collapse_out),
        ]
    )
    text = capsys.readouterr().out
    assert "mask_free_fold: true (--mask-free-fold)" in text
    saved = json.loads(collapse_out.read_text())
    assert saved["mask_free_fold"] is True
    checkpoint_out = tmp_path / "checkpoint.json"
    CHECKPOINT.main(
        [
            "--ckpt-dir",
            str(ckpt_dir),
            "--hands",
            "2",
            "--seed",
            "1",
            "--device",
            "cpu",
            "--mode",
            "argmax",
            "--no-mask-free-fold",
            "--out",
            str(checkpoint_out),
        ]
    )
    text = capsys.readouterr().out
    assert "mask_free_fold: false (--no-mask-free-fold)" in text
    saved = json.loads(checkpoint_out.read_text())
    assert saved["mask_free_fold"] is False


def test_old_metrics_header_keeps_m11a_columns_and_appends_fold_facing(tmp_path: Path):
    from alphanl.transformer.train import _append_csv

    path = tmp_path / "metrics.csv"
    old_header = "update,entropy,free_check_fold_ratio,legal_entropy,free_check_count"
    path.write_text(old_header + "\n0,1.0,0.5,0.4,3\n")
    _append_csv(
        path,
        {
            "update": 1,
            "entropy": 1.1,
            "free_check_fold_ratio": 0.0,
            "legal_entropy": 0.2,
            "free_check_count": 4,
            "fold_facing_ratio": 0.25,
        },
    )
    lines = path.read_text().splitlines()
    assert lines[0] == old_header
    assert lines[1] == "0,1.0,0.5,0.4,3"
    assert lines[2].endswith(",0.25")

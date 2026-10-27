"""Deep-stack action abstraction, replay compatibility and inference masks."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.evaluation import CFRActor
from alphanl.cfr.model import RegretNetwork, regret_matching
from alphanl.cfr.traversal import HunlState
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import HunlTable, HunlTokenizer, NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer


def flop():
    table = HunlTable(seed=42)
    table.reset()
    table.step(1)
    table.step(1)
    assert table.street == 1
    return table


@pytest.mark.parametrize("street", range(4))
@pytest.mark.parametrize("seat", (0, 1))
def test_deep_allin_blocked_on_every_street_and_seat(street, seat):
    snapshot = replace(flop().snapshot(0), street=street, me_seat=seat)
    guard = CFRActionGuard()
    guarded = guard.apply(snapshot)
    assert 4 in snapshot.legal_action_ids and 4 not in guarded.legal_action_ids
    assert guarded.legal_action_ids == tuple(a for a in snapshot.legal_action_ids if a != 4)
    # Every fallback, including all-zero and tiny positive regret noise,
    # must honor the identical mask used to enumerate traversal branches.
    mask = HunlTokenizer(64).encode(guarded)["legal_mask"]
    for raw in (torch.zeros(1, 5), torch.full((1, 5), -1.), torch.tensor([[0., 0., 0., 0., .001]])):
        for fallback in ("softmax", "uniform", "argmax"):
            probs = regret_matching(raw, mask, fallback=fallback)
            assert probs[0, 4] == 0 and probs.sum() == pytest.approx(1.)


def test_real_betting_line_builds_pot_then_unlocks_and_resets_each_street():
    table = flop()
    guard = CFRActionGuard()
    state = HunlState(table, HunlTokenizer(64), guard)
    assert state.observation()["legal_mask"][0, 4] == 0
    assert 4 not in state.legal_actions() and 4 in table.legal_action_ids()
    table.step(3)
    assert 4 not in guard.apply(table.snapshot(1)).legal_action_ids
    table.step(3)
    assert guard.street_raises(table.snapshot(0)) == 2
    assert 4 in guard.apply(table.snapshot(0)).legal_action_ids
    table.step(1)
    assert table.street == 2
    assert guard.street_raises(table.snapshot(0)) == 0
    assert 4 not in guard.apply(table.snapshot(0)).legal_action_ids


@pytest.mark.parametrize("ratio,allowed", [(2.5, True), (3., True), (3.001, False)])
def test_threshold_boundary(ratio, allowed):
    s = replace(flop().snapshot(0), stacks=(ratio * 100, ratio * 100), pot=100.)
    assert (4 in CFRActionGuard().apply(s).legal_action_ids) == allowed


def test_facing_bet_uses_opponent_remainder_plus_pending_call():
    table = flop()
    table.step_raise_to(1500)
    s = table.snapshot(1)
    assert s.to_call == 1500 and s.pot == 1700
    assert CFRActionGuard.all_in_pot_ratio(s) == pytest.approx(19900 / 1700)
    assert 4 not in CFRActionGuard().apply(s).legal_action_ids
    # An opponent shove leaves no chips behind, but the response must remain
    # fold/call (and must never be stripped by a raise guard).
    table.step(1)
    table.step(4)
    s = table.snapshot(1)
    assert 1 in CFRActionGuard().apply(s).legal_action_ids


def test_old_absolute_bb_exceptions_do_not_unlock_high_spr():
    s = replace(flop().snapshot(0), stacks=(1900., 1900.), pot=200.)
    assert 4 not in CFRActionGuard().apply(s).legal_action_ids
    s = replace(s, stacks=(18000., 18000.))
    assert 4 not in CFRActionGuard().apply(s).legal_action_ids
    assert CFRActionGuard(spr_enabled=False).apply(s) is s  # legacy postflop


def test_only_current_street_active_raises_count():
    table = flop()
    table.step(3)
    event = table.snapshot(1).events[-1]
    s = table.snapshot(1)
    for ignored in (replace(event, street=0), replace(event, action_type="call"),
                    replace(event, action_type="all_in", chips_added=event.to_call_before)):
        assert CFRActionGuard.street_raises(replace(s, events=(ignored, ignored))) == 0
    assert CFRActionGuard.street_raises(replace(s, events=(event, event))) == 2


def test_config_and_old_checkpoint_semantics():
    cfg = CFRConfig()
    assert cfg.spr_action_guard and cfg.guard_max_pot_ratio == 3
    old = cfg.to_dict()
    old.pop("spr_action_guard")
    old.pop("guard_max_pot_ratio")
    loaded = CFRConfig.from_checkpoint(old)
    assert not loaded.spr_action_guard and loaded.preflop_action_guard
    assert CFRConfig.from_checkpoint(cfg.to_dict()).spr_action_guard
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="guard_max_pot_ratio"):
            CFRConfig(guard_max_pot_ratio=value)
    with pytest.raises(ValueError, match="boolean"):
        CFRConfig(spr_action_guard=1)


def test_eval_intervention_masks_a_model_that_always_prefers_shoving(tmp_path):
    cfg = CFRConfig(spr_action_guard=False, model=ModelConfig(
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_seq_len=32, dropout=0))
    model = RegretNetwork(PokerTransformer(**asdict(cfg.model)))
    for param in model.parameters():
        param.data.zero_()
    # A constant head is independent of private cards and history.
    model.regret_head.bias.data[4] = 100.
    torch.save({"format": "sd_cfr_strategy_v2", "iteration": 1, "weight": 1,
                "rules_version": RULES_VERSION, "numeric_version": NUMERIC_VERSION,
                "config": cfg.to_dict(), "models": {"shared": model.state_dict()}},
               tmp_path / "iteration_0000001.pt")
    s = flop().snapshot(0)
    baseline = CFRActor(tmp_path, seed=1)
    baseline.begin_hand(0)
    assert baseline.act(s) == 4
    guarded = CFRActor(tmp_path, seed=1, spr_guard=True)
    guarded.begin_hand(0)
    assert guarded.act(s) != 4
    counts = guarded.audit.summary()["counts"]
    assert counts["spr_0_1_blocked"] == 1
    assert counts["spr_0_1_early_high_spr_shoves"] == 0
    assert counts["mask_leaks"] == 0
    assert not guarded.policy.config.spr_action_guard  # no archive mutation

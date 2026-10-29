"""Voluntary all-in mask, ladder replicas, and the native Stage 2 config."""
import json
from pathlib import Path

import pytest
import torch

from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.cfr.evaluation import RuleActor, play_hand
from alphanl.stage2.allin_mask import (
    SETTINGS_KEY, blocks_voluntary_allin, mask_config, mask_snapshot,
)
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.ladder_bots import (
    JitteredLadderReplicaActor, LadderReplicaActor, flop_pressure_hand, top_pair,
)
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import DEFENSE, MatchOpponentPool, NATIVE_WEIGHTS, PRO_WEIGHTS
from alphanl.stage2.train import load_checkpoint, save_checkpoint, warm_start_policy
from alphanl.transformer.actor_actions import PotFractionBet, apply_actor_decision
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.equity import preflop_key
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.model import ILLEGAL_LOGIT
from alphanl.transformer.schema import ActionEvent, HandSnapshot

ROOT = Path(__file__).resolve().parents[1]
ON = {"enabled": True, "spr_above": 2.5, "preflop_equity": 0.72, "postflop_equity": 0.80,
      "equity_samples": 8, "equity_seed": 17}


def snap(**overrides):
    base = dict(
        events=(), hole_cards=("S7", "D2"), board_cards=(), me_seat=1, sb_seat=1, street=0,
        pot=150.0, stacks=(19850.0, 19950.0), to_call=50.0,
        legal_action_ids=(0, 1, 2, 3, 4), big_blind=100.0, starting_stack=20000.0,
    )
    base.update(overrides)
    return HandSnapshot(**base)


def raise_event(actor=0):
    return ActionEvent(0, actor, "BB" if actor == 0 else "SB", actor == 1, 3, "raise",
                       400.0, 300.0, 0.75, 3, 100.0, 19000.0, (0, 1, 2, 3, 4))


@pytest.fixture
def tiny():
    torch.manual_seed(5)
    return CFRConfig(model=ModelConfig(d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                                       max_seq_len=24, dropout=0.0))


def test_strong_preflop_keeps_allin_and_weak_deep_spr_masks_it():
    enabled = mask_config(ON)
    strong = mask_snapshot(snap(hole_cards=("SA", "HA")), enabled)
    ako = mask_snapshot(snap(hole_cards=("SA", "HK")), enabled)
    weak = mask_snapshot(snap(), enabled)
    assert preflop_key(("SA", "HA")) == "AA" and preflop_key(("SA", "HK")) == "AKo"
    assert 4 in strong.legal_action_ids and 4 in ako.legal_action_ids
    assert 4 not in weak.legal_action_ids
    assert blocks_voluntary_allin(snap(), enabled)
    assert not blocks_voluntary_allin(snap(), mask_config({"enabled": False}))
    assert 4 in mask_snapshot(snap(), None).legal_action_ids


def test_spr_at_or_below_threshold_leaves_allin(monkeypatch):
    monkeypatch.setattr("alphanl.stage2.allin_mask.estimated_equity", lambda *args, **kwargs: 0.10)
    at = snap(pot=1000.0, stacks=(2500.0, 2500.0), to_call=0.0)
    above = snap(pot=1000.0, stacks=(2501.0, 2501.0), to_call=0.0)
    assert CFRActionGuard.all_in_pot_ratio(at) == pytest.approx(2.5)
    assert CFRActionGuard.all_in_pot_ratio(above) > 2.5
    assert 4 in mask_snapshot(at, ON).legal_action_ids
    assert 4 not in mask_snapshot(above, ON).legal_action_ids


def test_forced_call_allin_stays_legal(monkeypatch):
    monkeypatch.setattr("alphanl.stage2.allin_mask.estimated_equity", lambda *args, **kwargs: 0.10)
    covered = snap(pot=200.0, stacks=(20000.0, 1000.0), to_call=1000.0, legal_action_ids=(0, 1, 4))
    only = snap(pot=200.0, stacks=(20000.0, 1000.0), to_call=1000.0, legal_action_ids=(4,))
    assert CFRActionGuard.all_in_pot_ratio(covered) > 2.5
    assert 4 in mask_snapshot(covered, ON).legal_action_ids
    assert mask_snapshot(only, ON).legal_action_ids == (4,)


def test_postflop_true_two_pair_keeps_allin_and_equity_threshold(monkeypatch):
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("categorical premium must not consult equity")

    monkeypatch.setattr("alphanl.stage2.allin_mask.estimated_equity", boom)
    two = snap(hole_cards=("SA", "SK"), board_cards=("HA", "DK", "C2"), street=1, pot=400.0, to_call=0.0,
               stacks=(18000.0, 18000.0))
    assert 4 in mask_snapshot(two, ON).legal_action_ids and calls["n"] == 0

    def equity(snapshot, **kwargs):
        return equity.value

    air = snap(hole_cards=("S7", "D2"), board_cards=("HA", "DK", "C2"), street=1, pot=400.0, to_call=0.0,
               stacks=(18000.0, 18000.0))
    monkeypatch.setattr("alphanl.stage2.allin_mask.estimated_equity", equity)
    equity.value = 0.80
    assert 4 in mask_snapshot(air, ON).legal_action_ids
    equity.value = 0.799
    assert 4 not in mask_snapshot(air, ON).legal_action_ids


def test_forward_sets_masked_allin_logit_before_softmax(tiny):
    model = ResidualExploiterPolicy(PolicyNetwork(tiny), botzone_native_allin_mask=ON).eval()
    masked = mask_snapshot(snap(), model.settings[SETTINGS_KEY])
    batch = HunlTokenizer(tiny.model.max_seq_len, **tiny.tokenizer_kwargs()).encode(masked)
    batch["match_ctx"] = torch.zeros(1, len(model.context_fields))
    logits, _ = model(batch)
    logits = logits.detach()
    assert float(logits[0, 4]) < -1e8
    assert float(torch.softmax(logits, -1)[0, 4]) == 0.0
    assert logits[0, 4].item() == pytest.approx(ILLEGAL_LOGIT, rel=0, abs=1.0)
    kept = mask_snapshot(snap(hole_cards=("SA", "HA")), model.settings[SETTINGS_KEY])
    assert 4 in kept.legal_action_ids


def test_table_guard_unlock_still_masks_weak_allin_and_keeps_aces():
    pytest.importorskip("pokerkit")
    pytest.importorskip("phevaluator")
    from alphanl.transformer.tables import aligned_decks

    def faced(hole_sb):
        _, deck = aligned_decks(("HA", "DK"), hole_sb, ("CQ", "SJ", "D5", "C3", "H9"))
        table = HunlTable(seed=1)
        table.reset(deck)
        assert table.current_player() == 1
        table.step(3)
        table.step(3)
        assert table.current_player() == 1
        return table

    config = CFRConfig()
    guard = CFRActionGuard.from_config(config)
    weak = faced(("S7", "D2"))
    guarded = guard.apply(weak.snapshot(1))
    assert preflop_key(guarded.hole_cards) == "72o"
    assert 4 in guarded.legal_action_ids
    assert 4 not in mask_snapshot(guarded, ON).legal_action_ids
    env = BotzoneMatchEnv(config, RuleActor("call", seed=1, equity_samples=2), allin_mask=ON)
    env.table, env.hero, env.done = weak, 1, False
    env.ctx.begin_hand()
    assert env.observe()["legal_mask"][0, 4].item() == 0
    with pytest.raises(ValueError, match="Stage 1 abstraction"):
        env.step(4)
    strong = guard.apply(faced(("SA", "CA")).snapshot(1))
    assert preflop_key(strong.hole_cards) == "AA"
    assert 4 in mask_snapshot(strong, ON).legal_action_ids


def test_export_observation_matches_training_mask(tiny, tmp_path):
    from alphanl.stage2.export import export_agent
    import importlib.util
    model = ResidualExploiterPolicy(PolicyNetwork(tiny)).eval()
    path = tmp_path / "stage2.pt"
    save_checkpoint(path, model)
    exported = tmp_path / "agent.py"
    export_agent(path, exported)
    spec = importlib.util.spec_from_file_location("native_agent", exported)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.Agent()
    # SPR 2.6 is inside the new mask and outside the older SPR>3 guard.
    band = dict(pot=1000.0, stacks=(2600.0, 2600.0), to_call=0.0)
    weak, strong = snap(**band), snap(hole_cards=("SA", "HA"), **band)
    ctx = module.MatchCtx()
    plain = agent.observation(weak, ctx)
    assert plain["legal_mask"][0, 4] == 1
    agent.settings = dict(agent.settings)
    agent.settings[SETTINGS_KEY] = mask_config(ON)
    masked = agent.observation(weak, ctx)
    assert masked["legal_mask"][0, 4] == 0
    assert float(agent.logits(masked)[0, 4]) < -1e8
    assert agent.observation(strong, ctx)["legal_mask"][0, 4] == 1


def test_ladder_scripts_cover_shove_mix_offtree_size_and_flop_wars(monkeypatch):
    post = snap(street=1, board_cards=("HA", "D7", "C2"), pot=600.0, to_call=0.0,
                stacks=(15000.0, 15000.0), hole_cards=("SK", "DQ"))
    aggro = LadderReplicaActor("OffTree_Aggro_Boss", seed=1, equity_samples=2)
    aggro.params["postflop_pressure"] = 1.0
    bet = aggro.act(post)
    assert isinstance(bet, PotFractionBet) and 0.35 <= bet.fraction <= 1.15
    aggro.params["five_bet_shove"] = 1.0
    facing = snap(events=(raise_event(0), raise_event(1)))
    assert aggro.act(facing) == 4
    aggro.params["open_raise"] = 1.0
    assert aggro.act(snap()) == 3

    class Sequence(object):
        def __init__(self, values):
            self.values = list(values)

        def random(self):
            return self.values.pop(0)

    trap = LadderReplicaActor("Check_Shove_Trapper", seed=1, equity_samples=2)
    monkeypatch.setattr("alphanl.stage2.ladder_bots.is_value_hand", lambda *args, **kwargs: True)
    trap.rng = Sequence([0.0, 0.10])
    assert trap.act(snap(to_call=0.0)) == 4
    trap.rng = Sequence([0.0, 0.90])
    assert trap.act(snap(to_call=0.0)) == 1
    monkeypatch.setattr("alphanl.stage2.ladder_bots.is_value_hand", lambda *args, **kwargs: False)
    trap.rng = Sequence([0.0, 0.90])
    assert trap.act(snap(to_call=0.0)) == 4
    trap.rng = Sequence([0.50])
    assert trap.act(snap(to_call=0.0)) == 1

    flop = snap(street=1, board_cards=("HA", "D7", "C2"), hole_cards=("SA", "CK"), pot=600.0, to_call=0.0,
                stacks=(15000.0, 15000.0))
    assert top_pair(flop) and flop_pressure_hand(flop)
    war = LadderReplicaActor("Flop_Raise_War_Bot", seed=1, equity_samples=2)
    war.params["flop_raise"] = 1.0
    assert war.act(flop) in (2, 3, 4)
    air = snap(street=1, board_cards=("HA", "DK", "C2"), hole_cards=("S7", "D2"), pot=600.0, to_call=0.0,
               stacks=(15000.0, 18000.0))
    monkeypatch.setattr("alphanl.stage2.ladder_bots.texture", lambda snapshot: {"strong_draw": False, "strong": False, "dry": True})
    assert war.act(air) == 1
    left = JitteredLadderReplicaActor("OffTree_Aggro_Boss", seed=9, equity_samples=2)
    right = JitteredLadderReplicaActor("OffTree_Aggro_Boss", seed=9, equity_samples=2)
    assert left.params == right.params
    assert 0.35 <= left.params["size_low"] and left.params["size_high"] <= 1.15


def test_offtree_bet_and_short_matches_stay_legal():
    pytest.importorskip("pokerkit")
    pytest.importorskip("phevaluator")
    table = HunlTable(seed=2)
    table.reset()
    pot = int(table.state.total_pot_amount)
    apply_actor_decision(table, PotFractionBet(1.0))
    event = table.snapshot(table.current_player()).events[-1]
    assert event.chips_added == pot + 50  # call the extra blind, then one pot
    for name in ("OffTree_Aggro_Boss", "Check_Shove_Trapper", "Flop_Raise_War_Bot"):
        opponent = RuleActor(name, seed=4, equity_samples=2)
        for hand in range(2):
            play_hand(RuleActor("call", seed=hand + 1, equity_samples=2), opponent,
                      deck_seed=20 + hand, hero_seat=hand % 2)


def test_native_pool_keeps_defense_and_samples_replicas():
    pool = MatchOpponentPool("unused", weights=NATIVE_WEIGHTS, seed=3)
    assert pool.probabilities() == pytest.approx(NATIVE_WEIGHTS)
    for key, value in DEFENSE.items():
        assert pool.weights[key] == pytest.approx(value)
    assert set(PRO_WEIGHTS) < set(NATIVE_WEIGHTS)
    pool.template = object()
    pool.schedule = ["OffTree_Aggro_Boss", "Check_Shove_Trapper", "Flop_Raise_War_Bot"]
    seen = [pool.sample()[0] for _ in range(3)]
    assert set(seen) == {"OffTree_Aggro_Boss", "Check_Shove_Trapper", "Flop_Raise_War_Bot"}
    with pytest.raises(ValueError):
        MatchOpponentPool("unused", weights={**NATIVE_WEIGHTS, "reference": 0.2, "equity": NATIVE_WEIGHTS["equity"] - 0.1})


def test_native_config_budget_mask_and_warm_start_flag(tiny, tmp_path):
    cfg = json.loads((ROOT / "configs/stage2/botzone_native.json").read_text())
    assert cfg["updates"] == 250 and cfg["checkpoint_every"] == 50
    assert cfg["pool"]["self_snapshot_every"] == 50
    assert cfg["evaluation"]["every_updates"] == 50
    assert cfg["policy"]["botzone_native_allin_mask"] == cfg["botzone_native_allin_mask"]
    assert cfg["policy"]["botzone_native_allin_mask"]["enabled"] is True
    assert cfg["policy"]["alpha"] == 1.0 and cfg["policy"]["beta_kl"] == 0.05
    assert cfg["policy"]["late_beta_kl"] == 0.03
    assert cfg["ppo"] == {"epochs": 4, "minibatch_size": 256, "gamma": 1.0, "gae_lambda": 1.0}
    for name in ("equity", "reference", "OffTree_Aggro_Boss", "Check_Shove_Trapper", "Flop_Raise_War_Bot"):
        assert name in cfg["evaluation"]["opponents"]
    pool = MatchOpponentPool("unused", **cfg["pool"])
    assert pool.weights.keys() == NATIVE_WEIGHTS.keys()
    assert pool.probabilities() == pytest.approx(NATIVE_WEIGHTS)
    model = ResidualExploiterPolicy(PolicyNetwork(tiny), **cfg["policy"]).eval()
    assert model.settings == cfg["policy"]
    assert model.settings[SETTINGS_KEY]["enabled"] is True
    assert model.settings[SETTINGS_KEY]["spr_above"] == 2.5
    source = ResidualExploiterPolicy(model.reference, alpha=1.0, hidden=64, beta_kl=0.05, late_beta_kl=0.03,
                                     context_fields=tuple(cfg["policy"]["context_fields"]))
    # Rebuild destination from the same copied reference so the warm-start tensor check passes.
    destination = ResidualExploiterPolicy(source.reference, **cfg["policy"])
    checkpoint = tmp_path / "c200.pt"
    save_checkpoint(checkpoint, source)
    warm_start_policy(destination, checkpoint)
    assert destination.settings[SETTINGS_KEY]["enabled"] is True
    restored, blob = load_checkpoint(checkpoint)
    assert SETTINGS_KEY not in restored.settings and blob["settings"]["beta_kl"] == 0.05


def test_bundle_ships_relative_allin_mask(tmp_path):
    from alphanl.stage2.export import pack_botzone_zip
    import zipfile
    archive = tmp_path / "stage2.zip"
    pack_botzone_zip(archive)
    with zipfile.ZipFile(archive) as zipped:
        text = zipped.read("stage2_runtime/allin_mask.py").decode()
        runtime = zipped.read("stage2_runtime/runtime.py").decode()
    assert "from alphanl" not in text
    assert "import torch" not in text
    assert "def mask_snapshot" in text
    assert "mask_snapshot" in runtime

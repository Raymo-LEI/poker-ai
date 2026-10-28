"""Analytic weighting, fixed-root opponents, and genuine spawned HUNL coverage."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer, iteration_weighted_mse, iteration_weighted_huber
from alphanl.cfr.traversal import ExternalSamplingTraverser
from alphanl.cfr.weighting import fit_learning_rate
from alphanl.cfr.diagnostics import regression_probe
from test_cfr_trainer import config, TinyGame
from test_cfr_traversal import _Tree, _analytic_tree, _RecordingBuffer


@pytest.mark.parametrize("power", [1., 1.5, 2.])
def test_polynomial_loss_keeps_absolute_weights_and_mask(power):
    pred = torch.zeros(2, 5, requires_grad=True)
    target = torch.tensor([[1., 900, 900, 900, 900], [1., 900, 900, 900, 900]])
    mask = torch.tensor([[1., 0, 0, 0, 0]]).expand(2, -1)
    t = torch.tensor([1, 3])
    loss = iteration_weighted_mse(pred, target, mask, t, power=power)
    assert loss.item() == pytest.approx((1 + 3**power) / 2)
    loss.backward()
    assert pred.grad[1, 0] / pred.grad[0, 0] == pytest.approx(3**power)
    assert not pred.grad[:, 1:].any()
    assert iteration_weighted_huber(pred, target, mask, t, power=power).item() == pytest.approx(loss.item() / 2)
    assert iteration_weighted_mse(pred[:1], target[:1], mask[:1], torch.tensor([3]), power=power).item() == pytest.approx(3**power)
    assert iteration_weighted_mse(pred, target, mask, t, power=power, reference=3).item() == pytest.approx(loss.item() / 3**power)


def test_cosine_restarts_at_fit_barriers_and_reaches_requested_endpoints(tmp_path):
    cfg = replace(config(tmp_path), train_steps=20, learning_rate=2e-4,
                  learning_rate_schedule="warmup_cosine", warmup_steps=2)
    values = [fit_learning_rate(cfg, s) for s in range(20)]
    assert values[0] == pytest.approx(1e-4)
    assert values[1] == values[2] == pytest.approx(2e-4)
    assert values[-1] == pytest.approx(2e-5)
    assert all(a >= b for a, b in zip(values[2:], values[3:]))


def test_normalized_probe_weights_keep_true_weighted_mean():
    p = torch.zeros(2, 1)
    target = torch.tensor([[1.], [3.]])
    a = regression_probe(p, target, torch.ones_like(p), torch.tensor([1., 9.]))
    b = regression_probe(p, target, torch.ones_like(p), torch.tensor([.01, .09]))
    assert b["r2"].item() == pytest.approx(a["r2"].item())
    assert b["normalized_mse"].item() == pytest.approx(a["normalized_mse"].item())


class ScriptTree(_Tree):
    def child(self, action):
        if self.trace is not None:
            self.trace.append((self.node, action))
        return ScriptTree(self.nodes, self.nodes[self.node]["children"][action], self.trace)

    def scripted_action(self, actor):
        return 0  # Different from the network opponent's action 3.


@pytest.mark.parametrize("player", [0, 1])
def test_script_only_replaces_nontraverser_and_signed_regrets_remain_exact(player):
    original, policies = _analytic_tree(player)
    tree = ScriptTree(original.nodes, trace=[])
    sink = _RecordingBuffer()
    stats = ExternalSamplingTraverser(seed=2, scripted_opponent_prob=1,
        scripted_opponent_pool={"equity": 1}).collect([tree], traverser=player,
            iteration=8, policies=policies, buffer=sink)
    rows = {node: target[0] for node, target, _ in sink.records}
    # Root value = .25*1 + .75*(-100) = -74.75.
    assert rows[0].tolist() == pytest.approx([72.75, 75.75, 0, 0, -25.25])
    assert rows[2].tolist() == pytest.approx([0, 3, 0, -1, 0])
    assert stats.scripted_roots == stats.scripted_equity_roots == 1
    assert stats.selfplay_roots == 0 and stats.scripted_opponent_actions == 1
    assert stats.scripted_regret_samples == stats.regret_samples == 2
    assert stats.inference_rows == 2 and not policies[1 - player].batch_sizes
    assert (0, 0) in tree.trace  # Zero-probability traverser action still enumerated.


def test_mixed_roots_choose_once_and_rng_barrier_can_resume():
    _, policies = _analytic_tree()
    def run(traverser):
        roots = [ScriptTree(_analytic_tree()[0].nodes) for _ in range(300)]
        sink = _RecordingBuffer()
        return traverser.collect(roots, traverser=0, iteration=4, policies=policies, buffer=sink), sink.records
    a = ExternalSamplingTraverser(seed=7, scripted_opponent_prob=.15)
    state = a.rng.bit_generator.state
    first, rows = run(a)
    b = ExternalSamplingTraverser(seed=999, scripted_opponent_prob=.15,
                                  scripted_opponent_pool={"nit": .25, "equity": .75})
    b.rng.bit_generator.state = state
    second, copied = run(b)
    assert first == second and a.rng.bit_generator.state == b.rng.bit_generator.state
    assert 20 < first.scripted_roots < 75
    assert first.scripted_roots + first.selfplay_roots == 300
    assert first.scripted_equity_roots + first.scripted_nit_roots == first.scripted_roots
    assert all(torch.equal(x[1], y[1]) for x, y in zip(rows, copied))


def test_parallel_real_hunl_carries_features_and_script_stats(tmp_path):
    cfg = config(tmp_path, separate_player_models=False, equity_features=True,
        scripted_opponent_prob=1., scripted_opponent_pool={"nit": 1.},
        scripted_opponent_equity_samples=4, equity_samples=4,
        iteration_weight_power=2., normalize_iteration_weights=True,
        traversal_workers=2, worker_sample_batch=4,
        learning_rate_schedule="warmup_cosine", warmup_steps=1)
    with SDCFRTrainer(cfg) as trainer:
        row = trainer.run_iteration()
        assert row["p0_scripted_roots"] == row["p1_scripted_roots"] == 4
        assert row["p0_scripted_nit_roots"] == row["p1_scripted_nit_roots"] == 4
        assert row["p0_selfplay_roots"] == row["p1_selfplay_roots"] == 0
        assert row["p0_training_draws"] == row["p1_training_draws"]
        assert math.isfinite(row["shared_loss"])
        batch, _, _ = trainer.shared_reservoir.sample(8)
        assert (batch["numeric"][:, 2, 0] > 0).all()
        trainer.save_checkpoint(tmp_path / "train.pt")
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "restored"))) as trainer:
        trainer.load_checkpoint(tmp_path / "train.pt")
        assert trainer.run_iteration()["iteration"] == 2


def test_scripted_single_process_resume_keeps_next_fit_exact(tmp_path):
    cfg = config(tmp_path / "source", separate_player_models=False, equity_features=True,
        equity_samples=4, scripted_opponent_prob=.5, scripted_opponent_equity_samples=4,
        iteration_weight_power=2., normalize_iteration_weights=True,
        learning_rate_schedule="warmup_cosine", warmup_steps=1)
    with SDCFRTrainer(cfg) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(tmp_path / "source/train.pt")
        expected_row = trainer.run_iteration()
        expected = {key: value.clone() for key, value in trainer.shared_regret_net.state_dict().items()}
    with SDCFRTrainer(replace(cfg, out_dir=str(tmp_path / "resumed"))) as trainer:
        trainer.load_checkpoint(tmp_path / "source/train.pt")
        row = trainer.run_iteration()
        for name in ("p0_scripted_roots", "p1_scripted_roots", "shared_loss"):
            assert row[name] == expected_row[name]
        assert all(torch.equal(value, expected[key]) for key, value in trainer.shared_regret_net.state_dict().items())


def test_trace_partitions_returns_and_does_not_consume_actor_rng_for_oracle(tmp_path):
    from alphanl.cfr.evaluation import CFRActor, RuleActor, play_hand
    from scripts.diagnose_cfr_equity import trace_hand, summarize
    cfg = config(tmp_path, separate_player_models=False)
    with SDCFRTrainer(cfg) as trainer:
        trainer.run_iteration()
    def actors():
        return CFRActor(tmp_path / "strategies", seed=91), RuleActor("equity", seed=92)
    hero, opponent = actors()
    traced = [trace_hand(hero, opponent, deck_seed=seed, hero_seat=seat) for seed in range(3) for seat in (1, 0)]
    hero, opponent = actors()
    original = [play_hand(hero, opponent, deck_seed=seed, hero_seat=seat) for seed in range(3) for seat in (1, 0)]
    assert [h["payoff_chips"] for h in traced] == original
    result = summarize(traced)
    for prefix in ("category:", "terminal_street:", "position:"):
        assert sum(v["hands"] for key, v in result["buckets"].items() if key.startswith(prefix)) == 6
        assert sum(v["chips"] for key, v in result["buckets"].items() if key.startswith(prefix)) == sum(original)


def test_script_adapter_exposes_only_acting_seat_and_keeps_rule_bot_action_set():
    from alphanl.transformer.hunl import HunlTable, HunlTokenizer
    from alphanl.cfr.traversal import HunlState
    from alphanl.cfr.actions import CFRActionGuard
    table = HunlTable(seed=3)
    table.reset()
    state = HunlState(table, HunlTokenizer(), CFRActionGuard())
    class Actor:
        def act(self, snapshot):
            assert snapshot.hole_cards == table.hole_cards(1)
            assert not set(snapshot.hole_cards) & set(table.hole_cards(0))
            assert 4 in snapshot.legal_action_ids
            return 4
    assert 4 not in state.legal_actions()
    assert state.scripted_action(Actor()) == 4
    assert 4 in state.scripted_legal_actions()


def test_full_preset_uses_one_8m_pool_and_expected_model_and_input_schema():
    from pathlib import Path
    source = Path(__file__).resolve().parents[1]
    cfg = CFRConfig.from_json(source / "configs/cfr/stage1_149m_8m_equity_league_2400.json")
    with SDCFRTrainer(replace(cfg, device="cpu", traversal_workers=0)) as trainer:
        assert trainer.shared_reservoir.capacity == 8_000_000
        assert len(trainer.models) == len(trainer.buffers) == 1
        assert sum(p.numel() for p in trainer.shared_regret_net.parameters() if p.requires_grad) == 1_491_013
        assert trainer._hunl_tokenizer.numeric_version.startswith("hunl_random_equity_odds_v1:")
        assert cfg.iteration_weight_power == 2 and cfg.scripted_opponent_prob == .15
        assert cfg.train_steps == 2400 and cfg.learning_rate == 2e-4


def test_cpu_supervisor_runs_through_paired_completion_evaluation(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    source = Path(__file__).resolve().parents[1]
    cfg = replace(config(tmp_path / "run", separate_player_models=False, equity_features=True,
                         equity_samples=4, scripted_opponent_prob=.5, scripted_opponent_equity_samples=4),
                  traversals_per_player=2)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(cfg.to_dict()))
    subprocess.run([sys.executable, str(source / "scripts/run_cfr_fullupgrade.py"),
        "--config", str(path), "--poll-seconds", "1", "--evaluation-pairs", "2"],
        cwd=source, check=True, capture_output=True, text=True)
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["phase"] == "complete" and status["training_complete"]
    assert status["evaluation_complete"]
    assert (tmp_path / "run/launch_manifest.json").is_file()
    manifest = json.loads((tmp_path / "run/launch_manifest.json").read_text())
    assert manifest["objective"]["scripted_root_probability"] == cfg.scripted_opponent_prob
    assert manifest["retention"]["snapshot_every"] == cfg.snapshot_every
    assert (tmp_path / "run/resources.jsonl").is_file()
    for name, selected, power in (("full", [1, 2], 1.), ("burnin_quadratic", [2], 2.)):
        result = json.loads((tmp_path / f"run/evaluations/{name}/report.json").read_text())
        assert result["selected_iterations"] == selected
        assert result["sampling_power"] == power
        assert result["summary"]["pairs"] == 2

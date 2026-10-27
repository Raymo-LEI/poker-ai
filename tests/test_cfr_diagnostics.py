"""Probe scale invariance, masking, RNG isolation and actual monitor metrics."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from alphanl.cfr import SDCFRTrainer
from alphanl.cfr.diagnostics import regression_probe, probe_comparison
from test_cfr_trainer import TinyGame, config


def test_weighted_r2_and_ratio_ignore_weight_scale_and_illegal_targets():
    target = torch.tensor([[1., 900], [2., -900], [4., 100]])
    mask = torch.tensor([[1, 0], [1, 0], [1, 0]])
    weights = torch.tensor([1, 2, 3])
    start = regression_probe(torch.zeros_like(target), target, mask, weights)
    end = regression_probe(target + torch.tensor([.25, 10000]), target, mask, weights)
    scaled = regression_probe(target + torch.tensor([.25, -10000]), target, mask, weights * 100)
    assert float(end["unweighted_mse"]) == .0625
    assert float(end["r2"]) == pytest.approx(float(scaled["r2"]))
    assert float(end["normalized_mse"]) == pytest.approx(.0625)
    values = probe_comparison(start, end)
    assert values["probe_loss_ratio"] < .01
    assert values["probe_r2"] < 1
    assert values["explained_variance"] == 1
    perfect = regression_probe(target, target, mask, weights)
    assert float(perfect["r2"]) == 1
    assert float(perfect["weighted_mse"]) == 0


def test_constant_target_r2_and_zero_start_ratio_are_undefined():
    probe = regression_probe(torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 2), torch.ones(3))
    assert torch.isnan(probe["r2"])
    assert np.isnan(probe_comparison(probe, probe)["probe_loss_ratio"])


def test_probe_does_not_consume_training_rng(tmp_path):
    cfg = config(tmp_path, reinitialize=False)
    with SDCFRTrainer(cfg, root_factory=lambda seed: TinyGame()) as trainer:
        trainer.run_iteration()
        buffer = trainer.buffers[0]
        before = buffer.state_dict(include_data=False)["rng_state"]
        buffer.sample(8, rng=np.random.default_rng(100))
        assert before == buffer.state_dict(include_data=False)["rng_state"]
        trainer.train_player(0)
        assert trainer.last_fit_diagnostics["probe_samples"] == len(buffer)
        assert np.isfinite(trainer.last_fit_diagnostics["probe_loss_ratio"])


def test_offline_command_writes_measured_scores_and_separate_player_actions(tmp_path):
    cfg = config(tmp_path / "run")
    cfg.model.max_seq_len = 64

    class FullObservationGame(TinyGame):
        def observation(self):
            from alphanl.transformer.tokenizer import BettingHistoryTokenizer
            batch = BettingHistoryTokenizer(64).blank()
            batch["legal_mask"][0, 2] = 1
            return batch
        def child(self, action):
            return FullObservationGame(self.actions + (action,))

    with SDCFRTrainer(cfg, root_factory=lambda seed: FullObservationGame()) as trainer:
        trainer.run_iteration()
        trainer.save_checkpoint(Path(cfg.out_dir) / "train.pt")
    checkpoint = Path(cfg.out_dir) / "train.pt"
    before = checkpoint.stat().st_mtime_ns
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/collect_monitor_diagnostics.py"),
        "--run-dir", cfg.out_dir, "--iteration", "1", "--pairs", "2", "--health-hands", "4",
        "--probe-samples", "8", "--opponents", "random"], check=True, capture_output=True, text=True)
    assert checkpoint.stat().st_mtime_ns == before
    probe = json.loads((Path(cfg.out_dir) / "diagnostics.jsonl").read_text())
    assert "p0_probe_r2" in probe and "p0_probe_loss_ratio" not in probe
    assert "p0_probe_end_zero_ratio" in probe
    rows = [json.loads(line) for line in (Path(cfg.out_dir) / "eval.jsonl").read_text().splitlines()]
    assert rows[0]["hud/eval_hands"] == 4
    assert "hud/action_freq_p0_preflop/fold" in rows[0]
    assert "hud/action_freq_p1_preflop/all_in" in rows[0]
    assert rows[1]["eval/winrate_vs_random_mbb_pairs"] == 2


def test_hud_aggregation_uses_raw_denominators_and_keeps_no_opportunity_null():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("monitor_collector", root / "scripts/collect_monitor_diagnostics.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    audits = []
    for hands, vpip, regret_sum, regret_count in ((9, 2, 2, 2), (1, 1, 3, 1)):
        audits.append({"counts": {"hands_bb": hands, "vpip_hands_bb": vpip},
                       "positive_regret_actions_sum": regret_sum,
                       "positive_regret_decisions": regret_count})
    row = collector.health_metrics({"hands": 10, "hero": audits[0], "opponent": audits[1]}, 100)
    assert row["hud/preflop_vpip_bb"] == .3
    assert row["hud/preflop_vpip_bb_hands"] == 10
    assert row["hud/positive_regret_actions_mean"] == pytest.approx(5 / 3)
    assert row["hud/preflop_3bet_bb_opportunities"] == 0
    assert row["hud/preflop_3bet_bb"] is None
    json.dumps(row, allow_nan=False)

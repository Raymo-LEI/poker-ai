"""Trajectory-mixture semantics, archive isolation, fitting and portable export."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import (
    PolicyNetwork, collect_trajectories, cpu_latency, evaluate_student, fit_student,
    load_student, prepare_archive, save_student, soft_cross_entropy, verify_archive,
)
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.hunl import NUMERIC_VERSION, RULES_VERSION
from alphanl.transformer.model import PokerTransformer


@pytest.fixture
def archive(tmp_path):
    torch.set_num_threads(1)
    config = CFRConfig(model=ModelConfig(d_model=16, nhead=2, num_layers=1,
                                        dim_feedforward=32, max_seq_len=16, dropout=0))
    directory = tmp_path / "strategies"
    directory.mkdir()
    for t in (1, 2, 3):
        network = RegretNetwork(PokerTransformer(**config.to_dict()["model"]), fallback=config.fallback)
        with torch.no_grad():
            network.regret_head.bias.copy_(torch.tensor([t, 4-t, 1., 2., 3.]))
        torch.save({"format": "sd_cfr_strategy_v2", "iteration": t, "weight": t,
                    "rules_version": RULES_VERSION, "numeric_version": NUMERIC_VERSION,
                    "config": config.to_dict(), "models": {"shared": network.state_dict()}},
                   directory / f"iteration_{t:07d}.pt")
    (directory / "index.json").write_text(json.dumps({"format": "sd_cfr_archive_v1", "iterations": [1, 2, 3]}))
    return directory, config


def mutate_snapshot(directory, iteration, change):
    path = directory / f"iteration_{iteration:07d}.pt"
    blob = torch.load(path, weights_only=True)
    change(blob)
    torch.save(blob, path)


def test_archive_defaults_preserve_history_and_selection_is_explicit(archive):
    directory, _ = archive
    _, full = prepare_archive(directory)
    assert full["iterations"] == [1, 2, 3]
    assert [s["weight"] for s in full["snapshots"]] == [1, 2, 3]
    assert full["approximation"] is False
    _, tail = prepare_archive(directory, through=3, after=1, stride=2)
    assert tail["iterations"] == [2]
    assert tail["approximation"] is True
    with pytest.raises(ValueError, match="not present"):
        prepare_archive(directory, through=350)
    with pytest.raises(ValueError, match="empty"):
        prepare_archive(directory, after=100)
    for stride in (0, -1, True):
        with pytest.raises(ValueError):
            prepare_archive(directory, stride=stride)


def test_only_indexed_history_is_used(archive):
    directory, _ = archive
    (directory / "index.json").write_text(json.dumps({"format": "sd_cfr_archive_v1", "iterations": [1, 2]}))
    _, manifest = prepare_archive(directory)
    assert manifest["iterations"] == [1, 2]  # Future file still exists but is inactive.


@pytest.mark.parametrize("change,match", [
    (lambda b: b.update(numeric_version="wrong"), "incompatible strategy"),
    (lambda b: b.update(weight=99), "incompatible strategy"),
    (lambda b: b["config"].update(guard_max_pot_ratio=4.), "incompatible policy semantics"),
    (lambda b: b["config"].update(separate_player_models=True), "unified position"),
    (lambda b: b["models"]["shared"]["regret_head.bias"].fill_(float("nan")), "nonfinite"),
])
def test_prepare_checks_every_snapshot_not_just_randomly_selected_one(archive, change, match):
    directory, _ = archive
    mutate_snapshot(directory, 2, change)
    with pytest.raises(ValueError, match=match):
        prepare_archive(directory)


def test_snapshot_replacement_is_detected(archive):
    directory, _ = archive
    _, manifest = prepare_archive(directory)
    mutate_snapshot(directory, 2, lambda b: b["models"]["shared"]["regret_head.bias"].add_(1))
    with pytest.raises(ValueError, match="changed"):
        verify_archive(directory, manifest)


def test_cross_entropy_respects_masks_and_rejects_bad_targets():
    logits = torch.tensor([[.2, .8, float("nan"), 10., -3.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 0., 0., 0.]])
    target = torch.tensor([[.25, .75, 0., 0., 0.]])
    loss = soft_cross_entropy(logits, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(logits.grad[0, 2:], torch.zeros(3))
    for invalid in (target * 0, torch.tensor([[.25, .5, .25, 0., 0.]]), target * float("nan")):
        with pytest.raises(ValueError, match="invalid"):
            soft_cross_entropy(logits, invalid, mask)
    with pytest.raises(ValueError):
        soft_cross_entropy(logits, target, mask * 0)


def test_trajectory_labels_optimize_own_reach_average_not_naive_local_average():
    # Two historical policies with linear weights 1 and 3. Hero reaches a
    # later infoset with probabilities .1 and .9 respectively. Their local
    # action distributions there are [1,0] and [0,1]. The correct posterior
    # mixture is [1/28,27/28], NOT the naive t-only mean [1/4,3/4].
    targets = torch.tensor([[1., 0.]] + [[0., 1.]] * 27)
    correct = torch.tensor([1/28, 27/28]).log().requires_grad_()
    loss = soft_cross_entropy(correct.expand(28, -1), targets, torch.ones_like(targets))
    loss.backward()
    assert correct.grad.abs().max() < 1e-6
    naive = torch.tensor([.25, .75]).log()
    assert soft_cross_entropy(naive.expand(28, -1), targets, torch.ones_like(targets)) > loss


def test_real_hunl_collection_is_fixed_per_hand_and_keeps_guard_and_position(archive):
    directory, _ = archive
    config, manifest = prepare_archive(directory)
    data, report = collect_trajectories(directory, config, manifest, hands=24, seed=29)
    repeated, _ = collect_trajectories(directory, config, manifest, hands=24, seed=29)
    assert all(torch.equal(value, repeated[key]) for key, value in data.items())
    assert set(data["hero_seat"].tolist()) == {0, 1}
    assert report["hands"] == sum(report["snapshot_hand_counts"].values()) == 24
    for hand in data["hand_id"].unique():
        assert len(data["iteration"][data["hand_id"] == hand].unique()) == 1
    for seat in (0, 1):
        numeric = data["numeric"][data["hero_seat"] == seat, 1, :2]
        assert (numeric == torch.tensor([float(seat == 0), float(seat == 1)])).all()
    assert (data["target_probs"][data["legal_mask"] == 0] == 0).all()
    # At an initial deep-stack SB decision the checkpoint's SPR guard forbids shove.
    initial = [int((data["hand_id"] == hand).nonzero()[0]) for hand in range(1, 24, 2)]
    assert (data["legal_mask"][initial, 4] == 0).all()
    # Every row's label must come from its hand's selected network.
    for t in manifest["iterations"]:
        network = RegretNetwork(PokerTransformer(**config.to_dict()["model"]), fallback=config.fallback).eval()
        network.load_state_dict(torch.load(directory / f"iteration_{t:07d}.pt", weights_only=True)["models"]["shared"])
        selected = data["iteration"] == t
        if selected.any():
            with torch.no_grad():
                expected = network.strategy({key: value[selected] for key, value in data.items()})
            assert torch.allclose(data["target_probs"][selected], expected, atol=1e-6)
    validation, _ = collect_trajectories(directory, config, manifest, hands=10, seed=29, hand_offset=24)
    assert not set(data["hand_id"].tolist()) & set(validation["hand_id"].tolist())
    with pytest.raises(ValueError, match="max_decisions"):
        collect_trajectories(directory, config, manifest, hands=24, seed=29, max_decisions=1)


def test_fit_export_reload_and_single_cpu_forward(archive, tmp_path):
    directory, _ = archive
    config, manifest = prepare_archive(directory)
    data, _ = collect_trajectories(directory, config, manifest, hands=24, seed=19)
    torch.manual_seed(5)
    model = PolicyNetwork(config)
    before = evaluate_student(model, data)
    fit_student(model, data, steps=80, batch_size=32, seed=1, learning_rate=.003)
    after = evaluate_student(model, data)
    assert after["soft_cross_entropy_nats"] < before["soft_cross_entropy_nats"] - .02
    assert after["illegal_probability_mass"] == 0
    path = tmp_path / "policy.pt"
    save_student(path, model, manifest, after)
    restored, blob = load_student(path)
    assert blob["teacher"] == manifest
    assert not any("regret_head" in key or "value_head" in key for key in blob["model"])
    with torch.no_grad():
        assert torch.equal(restored.strategy(data), model.strategy(data))
    timings = cpu_latency(restored, data, decisions=3)
    assert timings["p99_ms"] > 0
    calls = []
    hook = restored.backbone.encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        restored.strategy({key: value[:1] for key, value in data.items()})
    hook.remove()
    assert len(calls) == 1
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["numeric_version"] = "legacy"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_student(path)
    with pytest.raises(ValueError, match="semantics disagree"):
        save_student(path, PolicyNetwork(replace(config, guard_max_pot_ratio=4)), manifest, {})


def test_cli_prepare_and_tiny_end_to_end(archive, tmp_path):
    directory, _ = archive
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "distill_cfr.py")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    base = [sys.executable, script, "--strategies", str(directory)]
    prepare = tmp_path / "prepared"
    result = subprocess.run(base + ["--out-dir", str(prepare), "--prepare-only"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (prepare / "teacher.json").exists() and not (prepare / "policy.pt").exists()
    run = tmp_path / "smoke"
    result = subprocess.run(base + ["--out-dir", str(run), "--train-hands", "8",
                            "--validation-hands", "8", "--steps", "2", "--batch-size", "8"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((run / "report.json").read_text())
    assert report["validation_after"]["illegal_probability_mass"] == 0
    assert report["cpu_latency"]["batch_size"] == 1
    load_student(run / "policy.pt")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    again = subprocess.run(base + ["--out-dir", str(run), "--prepare-only"],
                           env=env, capture_output=True, text=True)
    assert again.returncode != 0  # Never overwrite an existing experiment.


def test_quadratic_burn_in_teacher_uses_single_weighted_hand_draw(archive):
    directory, config = archive
    _, manifest = prepare_archive(directory, after=1, through=3, sampling_power=2.)
    assert manifest["iterations"] == [2, 3]
    assert manifest["sampling_power"] == 2.
    assert [r["sampling_weight"] for r in manifest["snapshots"]] == [4.,9.]
    from alphanl.cfr.strategy import SnapshotPolicy
    policy = SnapshotPolicy(directory, snapshot_iterations=manifest["iterations"], sampling_power=2., cache_size=2)
    draws = [policy.begin_hand(i % 2) for i in range(500)]
    assert set(draws) == {2,3}
    assert draws.count(3) / len(draws) == pytest.approx(9 / 13, abs=.06)
    dataset, report = collect_trajectories(directory, config, manifest, hands=10, seed=91)
    assert set(dataset["iteration"].tolist()) <= {2,3}
    for hand in dataset["hand_id"].unique():
        assert dataset["iteration"][dataset["hand_id"] == hand].unique().numel() == 1
    assert sum(report["snapshot_hand_counts"].values()) == 10
    with pytest.raises(ValueError):
        prepare_archive(directory, sampling_power=3.)

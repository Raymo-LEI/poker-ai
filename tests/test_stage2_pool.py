"""Match identity, defensive quotas, cached inference, PFSP and held-out metrics."""
from collections import Counter
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.cfr.evaluation import RuleActor
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.hunl import HunlTable, RULES_VERSION
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool, DEFAULT_WEIGHTS, DEFENSE
from alphanl.stage2.opponents import JitteredRuleActor, NeuralMatchActor, neural_actions
from alphanl.stage2.train import collect_match, collect_matches, train_matches, save_checkpoint, load_checkpoint
from alphanl.stage2.evaluation import MatchEvaluator, summarize_match_pairs, evaluation_metrics


def test_metrics_sanitizer_preserves_finite_values_and_strict_json():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/train_stage2.py"
    spec = importlib.util.spec_from_file_location("metrics_trainer", script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    metrics = {"explained_variance": float("nan"), "overflow": float("inf"),
               "underflow": float("-inf"), "loss": 1.25, "update": 104,
               "passed": True, "label": "nan", "missing": None,
               "nested": [{"ci95": (float("-inf"), 2.0)}]}
    sanitized = trainer.sanitize_metrics(metrics)
    decoded = json.loads(json.dumps(sanitized, allow_nan=False))
    assert decoded == {"explained_variance": None, "overflow": None, "underflow": None,
                       "loss": 1.25, "update": 104, "passed": True, "label": "nan",
                       "missing": None, "nested": [{"ci95": [None, 2.0]}]}
    assert np.isnan(metrics["explained_variance"])
    assert metrics["nested"][0]["ci95"][0] == float("-inf")


@pytest.fixture
def league(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = CFRConfig(equity_features=True, equity_samples=2, model=ModelConfig(
        d_model=32, nhead=2, num_layers=1, dim_feedforward=64, max_seq_len=24, dropout=0.))
    model = ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    directory = tmp_path / "strategies"
    directory.mkdir()
    network = RegretNetwork(PokerTransformer(**config.to_dict()["model"]), fallback=config.fallback)
    for t in (100,110,140,160,240,260,320,350):
        torch.save({"format":"sd_cfr_strategy_v2", "iteration":t, "weight":t,
                    "config":config.to_dict(), "rules_version":RULES_VERSION, "numeric_version":config.numeric_version,
                    "models":{"shared":network.state_dict()}}, directory / f"iteration_{t:07d}.pt")
    pool = MatchOpponentPool(directory, seed=5, equity_samples=2, history_cache_size=8)
    pool.initialize(model)
    return model, pool


def test_stage2_ppo_singleton_tail_metrics_are_strict_json(league):
    import warnings
    from alphanl.stage2.train import MatchRollout, train_match
    model, _ = league
    env = BotzoneMatchEnv(model.config, RuleActor("call", seed=3), seed=3)
    with torch.no_grad():
        batch = model.prepare(env.reset())
        batch = {k: torch.cat([v] * 3) for k, v in batch.items()}
        actions, logp, values, _ = model.act(batch)
    rollout = MatchRollout(batch, actions, logp, values,
                          torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 1.0]), {})
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        metrics = train_match(model, optimizer, rollout, epochs=1, minibatch_size=2, gamma=.9)
    assert all(np.isfinite(v) for v in metrics.values())
    assert json.loads(json.dumps(metrics, allow_nan=False)) == metrics


def test_quota_is_exact_despite_rule_weakness_and_history_cache_shared(league):
    model, pool = league
    pool.record("equity", -500)
    pool.record("nit", -20)
    seen = Counter()
    for _ in range(5):
        selected = [pool.sample() for _ in range(20)]
        counts = Counter(k for k, _ in selected)
        assert {k: counts[k] for k in DEFENSE} == {"reference": 2, "history": 3, "self": 2}
        for k, actor in selected:
            if k == "history":
                assert actor.model is pool.history_models[actor.snapshot_id]
            seen[k] += 1
    assert seen["reference"] + seen["history"] + seen["self"] == 35
    assert pool.disk_loads == 8
    assert [len(b) for b in pool.resident_buckets] == [2,2,2,2]
    pool.sample()  # Rotate the working set at a complete quota boundary.
    assert pool.disk_loads == 8
    assert sum(pool.probabilities()[k] for k in DEFENSE) == pytest.approx(.35)


def test_pfsp_threefold_priority_rolling_windows_and_ties():
    pool = MatchOpponentPool("unused", window_size=3)
    assert pool.probabilities() == pytest.approx(DEFAULT_WEIGHTS)
    for name in ("equity", "nit", "call", "maniac", "overbet", "push_fold"):
        pool.record(name, 30.)  # +600 mbb/hand, 100% wins.
    baseline = pool.probabilities()
    pool.record("equity", -200.)
    p = pool.probabilities()
    assert (p["equity"] / p["call"]) / (baseline["equity"] / baseline["call"]) == pytest.approx(3.)
    assert {k:p[k] for k in DEFENSE} == DEFENSE
    for _ in range(3):
        pool.record("equity", 25.)  # +500 exactly and >60% -> no boost.
    assert pool.probabilities() == pytest.approx(baseline)
    pool.record("equity", 0.)
    assert pool.rolling_metrics()["equity"]["match_win"] == pytest.approx(2/3)
    assert pool.rolling_metrics()["equity"]["mbb_per_hand"] < 500
    with pytest.raises(ValueError, match="retain"):
        MatchOpponentPool("unused", weights={**DEFAULT_WEIGHTS, "history": .20, "equity": .30})


def test_jitter_standard_baseline_exact_and_match_parameters_stable(monkeypatch):
    import alphanl.stage2.opponents as opponents
    import alphanl.cfr.evaluation as baselines
    table = HunlTable(seed=1)
    table.reset()
    snapshot = table.snapshot(1)
    for seed in range(12):
        actor = JitteredRuleActor("equity", variant="standard", seed=seed, equity_samples=4)
        baseline = RuleActor("equity", seed=seed, equity_samples=4)
        assert [actor.act(snapshot) for _ in range(6)] == [baseline.act(snapshot) for _ in range(6)]
    tight = JitteredRuleActor("equity", variant="tight", seed=4)
    aggressive = JitteredRuleActor("equity", variant="aggressive", seed=4)
    standard = JitteredRuleActor("equity", variant="standard", seed=4)
    assert .03 <= tight.params["margin"] <= .08
    facing = replace(snapshot, to_call=100, pot=100, legal_action_ids=(0,1,2,3,4))
    monkeypatch.setattr(opponents, "random_equity", lambda *args: .56)
    monkeypatch.setattr(baselines, "random_equity", lambda *args: .56)
    assert standard.act(facing) == 1 and tight.act(facing) == 0
    monkeypatch.setattr(opponents, "random_equity", lambda *args: .72)
    monkeypatch.setattr(baselines, "random_equity", lambda *args: .72)
    assert standard.act(facing) == 2 and aggressive.act(facing) == 3
    for name in ("nit", "call", "maniac", "overbet", "push_fold"):
        actors = [JitteredRuleActor(name, seed=seed) for seed in (1,2)]
        assert actors[0].params != actors[1].params
        parameters = dict(actors[0].params)
        for _ in range(50):
            actors[0].begin_hand(1)
            assert actors[0].act(snapshot) in snapshot.legal_action_ids
        assert actors[0].params == parameters


def test_reset_owns_one_opponent_and_never_encodes_identity(league):
    model, _ = league
    class Pool:
        calls = 0
        def sample(self):
            self.calls += 1
            return f"secret-{self.calls}", JitteredRuleActor("push_fold", seed=self.calls)
    pool = Pool()
    env = BotzoneMatchEnv(model.config, pool=pool)
    for match in range(2):
        rollout = collect_match(model, env)
        assert pool.calls == match + 1
        assert env.ctx.hands_played == 50
        assert set(rollout.batch) == {"token_ids", "type_ids", "numeric", "pad_mask", "legal_mask", "match_ctx", "poker_features", "ref_logits"}
        assert rollout.batch["match_ctx"].shape[-1] == 12
        assert torch.isfinite(rollout.batch["match_ctx"]).all()


def test_grouped_neural_forward_and_independent_hud(league):
    model, pool = league
    actors = [NeuralMatchActor(model.reference, kind="reference", snapshot_id="ref", seed=seed) for seed in (1,2)]
    table = HunlTable(seed=8)
    table.reset()
    for actor in actors:
        actor.begin_hand(0)
    calls = []
    hook = model.reference.register_forward_hook(lambda _, args, output: calls.append(len(output)))
    actions = neural_actions([(a, table.snapshot(0)) for a in actors])
    hook.remove()
    assert calls == [2]
    expected = [NeuralMatchActor(model.reference, kind="reference", snapshot_id="ref", seed=s).act(table.snapshot(0)) for s in (1,2)]
    assert actions == expected
    assert actors[0].ctx is not actors[1].ctx


def test_vector_rollout_conserves_rewards_done_and_frozen_reference(league):
    model, pool = league
    frozen = {k: v.clone() for k,v in model.reference.state_dict().items()}
    envs = [BotzoneMatchEnv(model.config, pool=pool, seed=700+i, defer_opponents=True) for i in range(20)]
    completed = {}
    def record(index, report):
        assert index not in completed
        assert envs[index].done and len(report['hands']) == 50
        assert report['opponent'] == envs[index].opponent_name
        assert torch.equal(model.delta_head.weight, torch.zeros_like(model.delta_head.weight))
        completed[index] = report
        pool.record(report['opponent'], report['hero_net_bb'])
    rollouts = collect_matches(model, envs, on_match_end=record)
    assert len(completed) == 20
    assert sum(pool.matches.values()) == 20
    assert Counter(e.opponent_name for e in envs)["history"] == 3
    for env, rollout in zip(envs, rollouts):
        assert rollout.dones.sum() == 1 and rollout.dones[-1] == 1
        assert rollout.report["hands_played"] == 50
        assert rollout.report == completed[envs.index(env)]
        assert float(rollout.rewards.sum()) == pytest.approx(env.ctx.hero_net_bb / 200, abs=2e-5)
        if isinstance(env.opponent, NeuralMatchActor):
            assert env.opponent.ctx.hands_played == 50
            assert env.opponent.ctx.hero_net_bb == -env.ctx.hero_net_bb
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    metrics = train_matches(model, optimizer, rollouts, epochs=1, minibatch_size=256)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(v, frozen[k]) for k,v in model.reference.state_dict().items())
    assert metrics['final_base_kl'] >= 0
    assert 0 <= metrics['final_policy_tv'] <= 1
    assert 0 <= metrics['final_argmax_changed'] <= 1


def test_pfsp_feedback_changes_without_crossing_priority_threshold():
    pool = MatchOpponentPool('unused')
    pool.record('equity', -200.)
    probabilities = pool.probabilities()
    pool.record('equity', -100.)
    assert pool.matches['equity'] == 2
    assert pool.rolling_metrics()['equity']['mbb_per_hand'] == -3000.
    assert pool.probabilities() == probabilities


def test_training_cli_writes_match_ledger_and_hand_traces_together(league, tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from alphanl.cfr.distillation import save_student
    model, pool = league
    root = Path(__file__).resolve().parents[1]
    reference = tmp_path / 'reference.pt'
    save_student(reference, model.reference, pool.manifest, {})
    settings = json.loads((root / 'configs/stage2/botzone_50hand.json').read_text())
    settings.update(reference=str(reference), strategies=str(pool.strategies),
                    pool=pool.contract, device='cpu')
    settings['ppo']['epochs'] = 1
    settings['traces'].update(enabled=True, train_matches=2)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(settings))
    output = tmp_path / 'run'
    subprocess.run([sys.executable, str(root / 'scripts/train_stage2.py'), '--config', str(config_path),
                    '--out-dir', str(output), '--updates', '1'],
                   env={**os.environ, 'PYTHONPATH': str(root / 'src')}, capture_output=True,
                   text=True, check=True)
    matches = [json.loads(line) for line in (output / 'matches.jsonl').read_text().splitlines()]
    metrics = json.loads((output / 'metrics.jsonl').read_text())
    hands = [json.loads(line) for line in
             (output / 'evaluations/stage2_train_u0000001/hands.jsonl').read_text().splitlines()]
    assert len(matches) == 20 and len({row['match'] for row in matches}) == 20
    assert len(hands) == 100
    assert metrics['stage2/opponent'] == 'mixed'
    assert metrics['stage2/match_wins'] == sum(row['hero_net_bb'] > 0 for row in matches)
    for row in matches:
        assert row['opponent'] in pool.weights and row['hands_played'] == 50
        assert metrics['stage2/feedback/' + row['opponent']] == sum(
            other['opponent'] == row['opponent'] for other in matches)
    for match in (1, 2):
        ledger = next(row for row in matches if row['match'] == match)
        traced = [row for row in hands if row['match'] == match]
        assert len(traced) == 50
        assert {row['opponent'] for row in traced} == {ledger['opponent']}
        assert sum(row['payoff_chips'] for row in traced) / 100 == ledger['hero_net_bb']


def test_resumable_schedule_self_weights_and_rng(league, tmp_path):
    model, pool = league
    pool.sample()
    pool.record("equity", -30)
    pool.add_self_snapshot(model, update=25)
    with torch.no_grad():
        model.delta_head.bias.add_(.5)
    assert not torch.equal(pool.self_weights[25]["delta_head.bias"], model.delta_head.bias)
    path = tmp_path / "train.pt"
    save_checkpoint(path, model, pool=pool, update=25)
    restored_model, blob = load_checkpoint(path)
    restored = MatchOpponentPool(pool.strategies, seed=99, equity_samples=2, history_cache_size=8)
    restored.initialize(restored_model)
    restored.load_state_dict(blob["pool"])
    def identity(choice):
        name, actor = choice
        return name, getattr(actor, "snapshot_id", None), getattr(actor, "variant", None), getattr(actor, "params", None)
    assert [identity(pool.sample()) for _ in range(40)] == [identity(restored.sample()) for _ in range(40)]
    assert restored.rolling_metrics() == pool.rolling_metrics()
    state = copy.deepcopy(blob["pool"])
    state["archive"]["snapshots"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="archive"):
        restored.load_state_dict(state)


def test_duplicate_units_gain_baseline_and_rng_isolation(league, monkeypatch):
    import alphanl.stage2.evaluation as evaluation
    model, _ = league
    monkeypatch.setattr(evaluation, "EVAL_OPPONENTS", ("equity", "reference"))
    evaluator = MatchEvaluator(model, match_pairs=2, equity_samples=2)
    original = torch.get_rng_state().clone()
    report = evaluator.evaluate(model, update=4)
    assert torch.equal(torch.get_rng_state(), original)
    for name in ("equity", "reference"):
        row = report["opponents"][name]
        assert row["gain_mbb_per_hand"] == 0  # Zero residual, identical common deals/RNG.
        assert row["candidate"]["matches"] == 4
        assert row["candidate"]["deck_pairs"] == 100
        assert row["candidate"]["hands"] == 200
    metrics = evaluation_metrics(report)
    assert metrics["stage2/eval/update"] == 4
    assert metrics["stage2/eval/equity/gain_mbb_per_hand"] == 0
    result = summarize_match_pairs([[10, -10], [0,20]])
    assert result["mbb_per_hand"] == 100
    assert result["match_win"] == .5
    assert result["ties"] == 1


def test_cache_misses_preserve_action_rng_and_self_snapshot_eviction(league):
    model, pool = league
    before = torch.get_rng_state().clone()
    for snapshot in list(pool.self_weights):
        pool._self_model(snapshot)
    assert torch.equal(before, torch.get_rng_state())
    pool.history_models.clear()
    pool._load_residents()
    assert torch.equal(before, torch.get_rng_state())
    for update in range(1, 20):
        pool.add_self_snapshot(model, update=update)
        pool._self_model(update)
    assert len(pool.self_weights) == pool.self_history_size
    assert set(pool.self_models) <= set(pool.self_weights)
    assert 0 not in pool.self_weights
    assert all(not p.requires_grad for m in pool.self_models.values() for p in m.parameters())
    assert model.delta_head.weight.requires_grad


def test_trainer_resume_replays_next_complete_quota_block(league, tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys
    from pathlib import Path
    from alphanl.cfr.distillation import save_student
    model, pool = league
    reference = tmp_path / "reference.pt"
    save_student(reference, model.reference, manifest=pool.manifest, report={})
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/stage2/botzone_50hand.json").read_text())
    settings.update(reference=str(reference), strategies=str(pool.strategies))
    settings["pool"].update(history_cache_size=8, equity_samples=2, self_snapshot_every=1)
    settings["ppo"].update(epochs=1, minibatch_size=256)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    script = Path(__file__).resolve().parents[1] / "scripts/train_stage2.py"
    spec = importlib.util.spec_from_file_location("matchpool_trainer", script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    train_matches = trainer.train_matches
    def nonfinite_metrics(*args, **kwargs):
        metrics = train_matches(*args, **kwargs)
        return {**metrics, "explained_variance": float("nan"),
                "positive_inf": float("inf"), "negative_inf": float("-inf")}
    monkeypatch.setattr(trainer, "train_matches", nonfinite_metrics)
    def run(name, updates, resume=None):
        args = [str(script), "--config", str(config), "--out-dir", str(tmp_path/name), "--updates", str(updates)]
        if resume:
            args += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", args)
        trainer.main()
        rows = [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
        stdout_rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert rows == stdout_rows and len(rows) == updates
        for row in rows:
            assert row["explained_variance"] is None
            assert row["positive_inf"] is None and row["negative_inf"] is None
            assert np.isfinite(row["loss"])
            json.dumps(row, allow_nan=False)
        return tmp_path / name / "train.pt"
    uninterrupted = run("uninterrupted", 2)
    first = run("first", 1)
    resumed = run("resumed", 1, first)
    whole, original = load_checkpoint(uninterrupted)
    continued, state = load_checkpoint(resumed)
    assert original["metadata"]["matches"] == state["metadata"]["matches"] == 40
    assert all(torch.equal(v, continued.state_dict()[k]) for k,v in whole.state_dict().items())
    assert original["pool"]["results"] == state["pool"]["results"]
    assert original["pool"]["rng"] == state["pool"]["rng"]
    assert torch.equal(original["torch_rng"], state["torch_rng"])


def test_trainer_writes_failed_retention_before_stopping(league, tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from alphanl.cfr.distillation import save_student
    model, pool = league
    reference = tmp_path / "reference.pt"
    save_student(reference, model.reference, manifest=pool.manifest, report={})
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/stage2/botzone_50hand.json").read_text())
    settings.update(reference=str(reference), strategies=str(pool.strategies))
    settings["pool"].update(history_cache_size=8, equity_samples=2)
    settings["ppo"].update(epochs=1, minibatch_size=256)
    settings["evaluation"].update(every_updates=1, match_pairs=2)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    script = Path(__file__).resolve().parents[1] / "scripts/train_stage2.py"
    spec = importlib.util.spec_from_file_location("retention_trainer", script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    def fail(self, *args, update, **kwargs):
        return {"update":update, "retention_pass":False, "equity_positive":False, "opponents":{},
                "diagnostics": {"ci95": [float("-inf"), float("inf")], "undefined": float("nan")}}
    monkeypatch.setattr(trainer.MatchEvaluator, "evaluate", fail)
    output = tmp_path / "failed"
    monkeypatch.setattr(sys, "argv", [str(script), "--config",str(config), "--out-dir",str(output), "--updates","2"])
    with pytest.raises(RuntimeError, match="retention"):
        trainer.main()
    _, state = load_checkpoint(output / "train.pt")
    assert state["update"] == 1 and state["metadata"]["retention_failed"]
    report = json.loads((output / "stage2_evaluations/update_0000001.json").read_text())
    assert report["diagnostics"] == {"ci95": [None, None], "undefined": None}
    assert not (output / "retained.pt").exists()
    metrics = json.loads((output/"metrics.jsonl").read_text())
    assert metrics["stage2/eval/retention_pass"] == 0

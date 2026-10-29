"""Real complete-match trajectories and passive JSONL recording contracts."""
import json
import pytest
import torch
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.distillation import PolicyNetwork
from alphanl.cfr.evaluation import RuleActor
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.train import collect_match, collect_matches
from alphanl.stage2.evaluation import MatchEvaluator
from alphanl.stage2.traces import HandTraceWriter, trace_settings
from alphanl.transformer.config import ModelConfig


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    config = CFRConfig(model=ModelConfig(d_model=32, nhead=2, num_layers=1,
        dim_feedforward=64, max_seq_len=24, dropout=0.))
    return ResidualExploiterPolicy(PolicyNetwork(config)).eval()


def load(path):
    return [json.loads(line) for line in (path / 'hands.jsonl').read_text().splitlines()]


def test_complete_match_trace_does_not_change_actions_rewards_or_rng(model, tmp_path):
    def collect(sink=None):
        env = BotzoneMatchEnv(model.config, RuleActor('call', seed=3), seed=7, hand_trace=sink)
        torch.manual_seed(91)
        rollout = collect_match(model, env)
        return rollout, torch.get_rng_state()
    plain, plain_rng = collect()
    path = tmp_path / 'trace'
    with HandTraceWriter(path, update=7, scope='training') as writer:
        traced, traced_rng = collect(writer.sink(match=1, opponent='call'))
    assert torch.equal(plain.actions, traced.actions)
    assert torch.equal(plain.rewards, traced.rewards)
    assert torch.equal(plain_rng, traced_rng)
    rows = load(path)
    assert len(rows) == 50
    assert sum(r['payoff_chips'] for r in rows) / 100 == traced.report['hero_net_bb']
    assert [r['hero_seat'] for r in rows] == [1,0]*25
    assert [r['hand'] for r in rows] == list(range(1,51))
    for row in rows:
        assert len(row['hero_hole']) == 2
        assert row['opponent'] == 'call' and row['scope'] == 'training'
        assert 'villain_hole' not in row and 'deck' not in row
        assert row['terminal_betting_street'] == row['events'][-1]['street']
        for event in row['events']:
            assert len(event['board']) == [0,3,4,5][event['street']]
            assert 0 <= event['chips_added'] <= event['stack_before']
            assert event['pot'] >= 150 and event['to_call'] >= 0
    report = json.loads((path/'report.json').read_text())
    assert report['recorded_hands'] == 50 and report['status'] == 'complete'


def test_batched_and_passive_hands_are_traced(model, tmp_path):
    path = tmp_path / 'batched'
    with HandTraceWriter(path, update=1, scope='training') as writer:
        envs = [BotzoneMatchEnv(model.config, RuleActor('nit', seed=i), seed=40+i,
                 first_hero_seat=i, defer_opponents=True, hand_trace=writer.sink(match=i)) for i in (0,1)]
        rollouts = collect_matches(model, envs)
    rows = load(path)
    assert len(rows) == 100
    for i, rollout in enumerate(rollouts):
        hands = [r for r in rows if r['match'] == i]
        assert len(hands) == 50
        assert sum(r['payoff_chips'] for r in hands)/100 == rollout.report['hero_net_bb']
    # Opponent can end a hand before any hero decision; complete-hand capture is still required.
    passive = []
    env = BotzoneMatchEnv(model.config, RuleActor('call', seed=8), hand_trace=passive.append)
    env.opponent.act = lambda snapshot: 0 if snapshot.to_call > 0 else 1
    env.reset()
    while not env.done:
        env.step(1)
    assert len(passive) == 50
    assert any(len(r['events']) == 1 and r['events'][0]['seat'] != r['hero_seat'] for r in passive)


def test_evaluation_duplicate_legs_keep_seed_and_payoff_identity(model, tmp_path):
    evaluator = MatchEvaluator(model, match_pairs=2, seed=101, equity_samples=1)
    plain = evaluator._totals(model, 'call')
    path = tmp_path / 'evaluation'
    with HandTraceWriter(path, update=100, scope='evaluation') as writer:
        traced = evaluator._totals(model, 'call', trace_writer=writer, trace_match_pairs=1)
    assert traced == plain
    rows = load(path)
    assert len(rows) == 100  # first pair only, both 50-hand seat-swapped legs
    assert {r['pair'] for r in rows} == {0}
    for leg, seat in enumerate((1,0)):
        assert sum(r['payoff_chips'] for r in rows if r['first_hero_seat'] == seat)/100 == plain[0][leg]
    assert all(r['policy'] == 'candidate' and r['scope'] == 'evaluation' for r in rows)


def test_writer_cap_failure_metadata_and_no_overwrite(tmp_path):
    path = tmp_path / 'capped'
    with pytest.raises(RuntimeError), HandTraceWriter(path, update=1, scope='training', max_bytes=1024) as writer:
        sink = writer.sink()
        for _ in range(20):
            sink({'events': [], 'payoff_chips': 0, 'padding': 'x'*200})
        raise RuntimeError('runner failed')
    rows = load(path)
    report = json.loads((path/'report.json').read_text())
    assert 0 < len(rows) < 20
    assert (path/'hands.jsonl').stat().st_size <= 1024
    assert report['dropped_hands'] == 20-len(rows)
    assert report['truncated'] and report['status'] == 'failed'
    with pytest.raises(FileExistsError):
        HandTraceWriter(path, update=1, scope='training')
    assert trace_settings({'traces': {'enabled':False}})['enabled'] is False
    with pytest.raises(ValueError):
        trace_settings({'traces': {'train_every_updates':0}})

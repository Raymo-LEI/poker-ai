"""Cross-process terminal reconciliation and bounded, iteration-persistent RNR."""
import copy
import importlib
import importlib.util
import json
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from alphanl.stage2.context import MatchCtx, PRO_CONTEXT_FIELDS
from alphanl.stage2 import deep_solver_v2 as solver
from alphanl.stage2.solver_prior import profile, response_prior
from alphanl.transformer.schema import ActionEvent


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    from alphanl.stage2.export import pack_botzone_zip
    folder = tmp_path_factory.mktemp('terminal_runtime')
    archive = folder / 'development.zip'
    pack_botzone_zip(archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
        assert 'stage2_runtime/solver_prior.py' in z.namelist()
    name = '_terminal_context_acceptance'
    spec = importlib.util.spec_from_file_location(name, folder / 'stage2_runtime/__init__.py',
        submodule_search_locations=[str(folder / 'stage2_runtime')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + '.runtime')


class FixedAgent:
    decision_identity = 'terminal_test'
    release_id = 'development_unreleased'
    deployment_guard = {'enabled': False}
    last_guard = {}
    search_audit = {'status': 'test'}

    def __init__(self, action=2):
        self.action = action

    def act(self, *args, **kwargs):
        return self.action


def request(hand=0, net=0, street=1, dealer=None, hero=1):
    dealer = hand % 2 if dealer is None else dealer
    sb = 1 - dealer
    history = []
    if street:
        history = [dict(round=0, player_id=sb, action=0), dict(round=0, player_id=dealer, action=0)]
        for s in range(1, street):
            history += [dict(round=s, player_id=sb, action=0), dict(round=s, player_id=dealer, action=0)]
        if hero == dealer:
            history.append(dict(round=street, player_id=sb, action=0))
    elif hero == dealer:
        history = [dict(round=0, player_id=sb, action=0)]
    return dict(num_players=2, dealer_id=dealer, my_id=hero,
        my_chips=19900 if street or hero == dealer else 19950, my_cards=[51, 23],
        public_cards=[0, 5, 10, 31, 44][:(0, 3, 4, 5)[street]], history=history,
        hand=hand, max_hand=50, total_win_chips=[net if i == hero else -net for i in (0, 1)],
        total_win_games=[0, 0])


def transition(runtime, previous, delta, action=2, next_hand=None):
    result = runtime.respond(FixedAgent(action), {'requests': [previous]})
    next_hand = previous['hand'] + 1 if next_hand is None else next_hand
    current = request(next_hand, previous['total_win_chips'][previous['my_id']] + delta,
                      street=0, dealer=(previous['dealer_id'] + next_hand - previous['hand']) % 2,
                      hero=previous['my_id'])
    ctx, data = runtime.context_from_payload({'requests': [current], 'data': json.dumps(result['data'])})
    return ctx, data, result, current


def test_ten_hidden_folds_exact_data_roundtrip_and_cumulative_parity(runtime):
    data = ''
    requests, responses = [], []
    for hand in range(10):
        req = request(hand, hand * 100)
        payload = {'requests': [req], 'data': json.dumps(data) if data else ''}
        result = runtime.respond(FixedAgent(), payload)
        repeated = runtime.respond(FixedAgent(), dict(payload, data=result['data']))
        assert repeated == result
        data = result['data']
        requests.append(req)
        responses.append(result['response'])
    final = request(10, 1000, street=0)
    ctx, saved = runtime.context_from_payload({'requests': [final], 'data': data})
    assert ctx.reconstructed_folds == 10
    assert ctx.successes['fold_to_postflop_bet'] == 10
    assert ctx.opportunities['fold_to_postflop_bet'] == 10
    assert ctx.opp_postflop_bet_folds == ctx.opp_postflop_bet_responses == 10
    assert ctx.opportunities['showdown_win'] == 0
    assert len(ctx.as_dict(PRO_CONTEXT_FIELDS)) == 14
    assert len(json.dumps(saved)) < 100 * 1024
    full, _ = runtime.context_from_payload({'requests': requests + [final], 'responses': responses})
    assert full.successes == ctx.successes
    assert full.opportunities == ctx.opportunities
    assert full.response_by_size == ctx.response_by_size


def test_hidden_preflop_fold_updates_open_and_threebet_opportunity(runtime):
    ctx, _, _, _ = transition(runtime, request(street=0), 100)
    assert ctx.reconstructed_folds == 1
    assert ctx.successes['fold_to_preflop_raise'] == 1
    assert ctx.opportunities['three_bet'] == 1
    assert ctx.successes['three_bet'] == 0


def test_official_judge_scores_exclude_new_hand_blinds(runtime):
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('terminal_judge_fixture', Path(__file__).parent / 'fixtures/botzone_judge.py')
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    game = judge.Holdem([20000, 20000], 0)
    game.set_deck_array(list(range(52)))
    game.deal_cards_and_blind()
    game.player_action(150)
    outcome = game.player_action(-1)
    final = game.get_player_final_chips(outcome)
    net = final[1] - 20000
    assert net == 100
    new_game = judge.Holdem([20000, 20000], 1)
    new_game.set_deck_array(list(range(52)))
    new_game.deal_cards_and_blind()
    # Opponent SB calls so the hero BB receives the next request.
    new_game.player_action(0)
    match = dict(hand=1, max_hand=50, total_win_chips=[-net, net], total_win_games=[0, 1])
    req = json.loads(judge.make_request_json(new_game, match))['content']['1']
    assert req['my_chips'] == 19900 and req['total_win_chips'][1] == 100
    assert runtime._handoff_delta(request(street=0), req) == (100, [])


@pytest.mark.parametrize('net', [20000, -20000, 0, 1])
def test_hidden_allin_call_closes_before_river(runtime, net):
    ctx, _, _, _ = transition(runtime, request(street=1), net, action=4)
    assert ctx.reconstructed_calls == 1 and ctx.opportunities['showdown_win'] == 1
    assert ctx.successes['showdown_win'] == (.5 if abs(net) <= 1 else float(net < 0))


def test_hero_river_call_needs_no_invented_opponent_call(runtime):
    req = request(street=3, dealer=1)
    req['history'][-1]['action'] = 100
    ctx, _, _, _ = transition(runtime, req, -200, action=1)
    assert ctx.opportunities['showdown_win'] == ctx.successes['showdown_win'] == 1
    assert ctx.reconstructed_calls == 0


def test_observed_terminal_fold_is_never_counted_twice(runtime):
    req = request()
    result = runtime.respond(FixedAgent(), {'requests': [req]})
    final = copy.deepcopy(req)
    final['history'] += [dict(round=1, player_id=1, action=100), dict(round=1, player_id=0, action=-1)]
    final['my_chips'] = 19800
    ctx, data = runtime.context_from_payload({'requests': [final], 'data': result['data']})
    ctx, data = runtime.context_from_payload({'requests': [request(1, 100, street=0)], 'data': data})
    assert ctx.reconstructed_folds == 0
    assert ctx.successes['fold_to_postflop_bet'] == ctx.opportunities['fold_to_postflop_bet'] == 1


@pytest.mark.parametrize('net,win', [(200, 0), (-200, 1), (0, .5), (1, .5), (-1, .5)])
def test_hidden_river_call_win_loss_split_and_odd_chip(runtime, net, win):
    ctx, _, _, _ = transition(runtime, request(street=3), net)
    assert ctx.reconstructed_calls == 1
    assert ctx.opportunities['fold_to_postflop_bet'] == 1
    assert ctx.successes['fold_to_postflop_bet'] == 0
    assert ctx.opportunities['showdown_win'] == 1
    assert ctx.successes['showdown_win'] == win
    assert ctx.last_terminal_status == ('split' if abs(net) <= 1 else 'call')


@pytest.mark.parametrize('net,win', [(100, 0), (-100, 1), (0, .5), (-1, .5)])
def test_river_ip_checkbehind_is_a_showdown_without_fake_call(runtime, net, win):
    ctx, _, _, _ = transition(runtime, request(street=3, dealer=1), net, action=1)
    assert ctx.opportunities['showdown_win'] == 1
    assert ctx.successes['showdown_win'] == win
    assert ctx.reconstructed_calls == ctx.reconstructed_folds == 0


def test_unknown_rolls_back_entire_hand_and_does_not_guess_early_street_call(runtime):
    for net in (777, -200, 0):
        ctx, _, _, _ = transition(runtime, request(street=1), net)
        assert ctx.last_terminal_status == 'unknown'
        assert ctx.unknown_hands == 1
        assert sum(ctx.hand_statistics_baseline['opportunities'].values()) == 0
        assert ctx.reconstructed_calls == ctx.reconstructed_folds == 0
        assert ctx.hands_played == 1
        assert ctx.hero_net_bb == net / 100.


def test_skip_bb_blind_profit_and_reject_ambiguous_multi_hand_jump(runtime):
    ctx, _, _, current = transition(runtime, request(street=1), 150, next_hand=2)
    assert ctx.reconstructed_folds == 1
    assert ctx.successes['fold_to_postflop_bet'] == 1
    assert ctx.opportunities['vpip'] == 2  # Seen hand plus skipped BB hand.
    assert ctx.hands_played == 2 and ctx.hero_net_bb == 1.5
    assert runtime._handoff_delta(request(), current) == (100, [1])
    ctx, _, _, _ = transition(runtime, request(), 250, next_hand=4)
    assert ctx.last_terminal_status == 'unknown'
    assert sum(ctx.opportunities.values()) == 0
    # The opposite parity skips an SB turn and cannot prove its outcome.
    ctx, _, _, _ = transition(runtime, request(dealer=1), 150, next_hand=2)
    assert ctx.last_terminal_status == 'unknown'


@pytest.mark.parametrize('lead', [3400, -3400])
def test_lock_freezes_profile_for_either_seat_and_new_match_resets(runtime, lead):
    first = request(10, lead)
    result = runtime.respond(FixedAgent(), {'requests': [first]})
    frozen = result['data']['context']
    assert frozen['frequency_frozen'] and frozen['frozen_at_hand'] == 10
    ctx, data = runtime.context_from_payload({'requests': [request(11, lead + 100)], 'data': result['data']})
    assert ctx.frequency_frozen
    assert ctx.successes == frozen['successes']
    assert ctx.opportunities == frozen['opportunities']
    assert ctx.postflop_size_count == 0 and ctx.response_by_size == {}
    reset, _ = runtime.context_from_payload({'requests': [request(0, 0, street=0)], 'data': data})
    assert not reset.frequency_frozen and reset.hands_played == 0


def event(street, kind, chips, pot=1000, owed=0, mine=False, position='SB'):
    return ActionEvent(street, 1, position, mine, None, kind, pot, chips, chips / pot, 0, owed, 20000, ())


def test_size_fingerprints_json_roundtrip_and_legacy_compatibility():
    ctx = MatchCtx()
    for size in (150, 250):
        ctx.begin_hand()
        ctx.observe_action(event(0, 'raise', size, owed=50))
        ctx.observe_action(event(1, 'bet', 400))
        ctx.observe_action(event(2, 'bet', 1001))
        ctx.finish_hand(0)
    restored = MatchCtx.from_state_dict(json.loads(json.dumps(ctx.state_dict())))
    assert restored.sb_open_count == 2 and restored.sb_open_mean_bb == 2.
    assert restored.sb_open_variance_bb == .25
    assert restored.postflop_size_count == 4
    assert restored.postflop_small_bets == restored.postflop_overbets == 2
    legacy = {k: v for k, v in ctx.state_dict().items() if k in ('format', 'hands_played', 'hero_net_bb', 'successes', 'opportunities')}
    assert MatchCtx.from_state_dict(legacy).sb_open_count == 0


def solver_spot(pot=200):
    return SimpleNamespace(hole_cards=('5s', '6h'), board_cards=('2s', '3h', '4d', '9c', 'Ks'),
        pot=pot, to_call=0, me_stack=2000, opp_stack=2000, big_blind=100., street=3,
        legal_action_ids=(1, 2, 3, 4), events=(), me_seat=0, postflop_first_seat=0)


def folder_profile(runtime):
    requests = [request(h, h * 100, street=3) for h in range(11)]
    responses = [100] * 10
    ctx, _ = runtime.context_from_payload({'requests': requests, 'responses': responses})
    return MatchCtx.from_state_dict(ctx.state_dict())


def test_true_rnr_every_iteration_and_legal_simplexes(runtime):
    ctx = folder_profile(runtime)
    s = solver_spot()
    sol = solver.PublicSolver(s, context=ctx, iterations=8)
    node = sol.tree.children[sol.tree.labels.index(2)]
    eta = node.match_eta
    assert 0 < eta < .25 and sol.match_profile['eta_sample'] == .25
    for step in range(1, 9):
        sol.walk(sol.tree, sol.rh, sol.ro, step)
        sol.update_strategies(sol.tree)
        positive = np.maximum(node.regret, 0.)
        den = positive.sum(axis=1, keepdims=True)
        nash = np.divide(positive, den, out=np.full_like(positive, 1. / len(node.children)), where=den > 0)
        np.testing.assert_allclose(node.strategy, (1 - eta) * nash + eta * node.match_prior)
        np.testing.assert_allclose(node.strategy.sum(1), 1.)
        assert np.min(node.strategy) >= 0
    # Final iteration is still constrained, not just the initialization.
    assert np.max(np.abs(node.strategy - nash)) > .01

    # A large hypothetical bet must not retain the small root pot's weight.
    s.me_stack = s.opp_stack = 5000
    large = solver.PublicSolver(s, context=ctx, iterations=1)
    jam = large.tree.children[large.tree.labels.index(4)]
    assert jam.match_eta == 0


def test_response_prior_conditions_on_street_and_size_bucket():
    ctx = MatchCtx(hands_played=14)
    ctx.opportunities['fold_to_postflop_bet'] = 20
    ctx.successes['fold_to_postflop_bet'] = 10
    ctx.response_by_size = {'3:small': [10, 10, 0], '3:overbet': [10, 0, 10], '1:small': [10, 0, 0]}
    strength = np.linspace(0, 1, 20)
    info = profile(ctx, 3)
    small = response_prior(['fold', 'call', 'raise_all_in'], strength, info, ctx, .4)
    big = response_prior(['fold', 'call', 'raise_all_in'], strength, info, ctx, 1.01)
    assert small[:, 0].mean() > big[:, 0].mean() + .4
    flop = response_prior(['fold', 'call'], strength, profile(ctx, 1), ctx, .4)
    assert small[:, 0].mean() > flop[:, 0].mean() + .4


def test_pot_risk_scaling_and_training_bypass(runtime):
    ctx = folder_profile(runtime)
    etas = [profile(ctx, 3, solver_spot(p))['eta'] for p in (200, 1000, 3999, 4000, 10000)]
    assert etas[0] > etas[1] > etas[2] > etas[3] == etas[4] == 0
    neutral = solver.solve_public_subgame(solver_spot(), context=None, iterations=4)
    training = solver.solve_public_subgame(solver_spot(), context=ctx, context_prior=False, iterations=4)
    np.testing.assert_array_equal(neutral.average, training.average)
    np.testing.assert_array_equal(neutral.ro, training.ro)
    big_a = solver.solve_public_subgame(solver_spot(4000), context=ctx, iterations=4)
    big_b = solver.solve_public_subgame(solver_spot(4000), iterations=4)
    np.testing.assert_array_equal(big_a.average, big_b.average)


def test_context_size_range_uses_public_support_and_retains_positive_mass(runtime):
    ctx = folder_profile(runtime)
    s = solver_spot()
    s.events = (event(3, 'bet', 500, pot=200),)
    uniform = np.ones(1081) / 1081
    sol = solver.PublicSolver(s, opponent_range=uniform, hero_range=uniform, context=ctx, iterations=2)
    assert sol.ro.sum() == pytest.approx(1.) and sol.ro.min() > 0
    assert np.max(np.abs(sol.ro - uniform)) > 1e-6
    assert np.dot(sol.ro, sol.match_strength) > np.dot(uniform, sol.match_strength)
    s.hole_cards = ('As', 'Ah')
    other = solver.PublicSolver(s, opponent_range=uniform, hero_range=uniform, context=ctx, iterations=2)
    np.testing.assert_array_equal(other.ro, sol.ro)


def test_ten_hidden_folds_shift_actual_solver_toward_stealing(runtime):
    ctx = folder_profile(runtime)
    s = solver_spot()
    a = solver.solve_public_subgame(s, iterations=24)
    b = solver.solve_public_subgame(s, context=ctx, iterations=24)
    # Aggregate weak holdings: increased betting is the measurable steal signal.
    weak = b.match_strength < .3
    aggressive = [i for i, action in enumerate(b.tree.labels) if action in (2, 3, 4)]
    gain = b.average[weak][:, aggressive].sum(1).mean() - a.average[weak][:, aggressive].sum(1).mean()
    assert gain > .01
    assert abs(b.trace['zero_sum_residual']) < 1e-10


def test_exported_64_token_14_context_online_path_and_archived_module_manifest(runtime, tmp_path, monkeypatch):
    from pathlib import Path
    import torch
    from alphanl.stage2 import export, cfv
    from alphanl.stage2.train import save_checkpoint
    from alphanl.stage2.policy import ResidualExploiterPolicy
    from alphanl.cfr.config import CFRConfig
    from alphanl.cfr.distillation import PolicyNetwork
    from alphanl.transformer.config import ModelConfig
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/sprint24'))
    from evening_d import Bundle
    from test_evening_v2 import _random_nets
    from overnight_data import TREE
    torch.set_num_threads(1)
    model = ResidualExploiterPolicy(PolicyNetwork(CFRConfig(model=ModelConfig(
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_seq_len=64)),
        ), context_fields=PRO_CONTEXT_FIELDS)
    checkpoint = tmp_path / 'tiny.pt'
    save_checkpoint(checkpoint, model)
    nets = tmp_path / 'nets.npz'
    cfv.publish(nets, _random_nets(), 2, 1, 'nets', dict(tree=TREE, river_samples=1, turn_samples=1))
    source, weights = tmp_path / 'development.zip', tmp_path / 'development.npz'
    export.pack_botzone_zip(source)
    export.export_botzone_weights(checkpoint, weights, cfv_nets=nets)
    manifest = export._sources
    monkeypatch.setattr(export, '_sources', lambda *a, **kw: dict(manifest(*a, **kw), future_module=''))
    bundle = Bundle(source, weights)
    ctx = folder_profile(runtime)
    req = request(10, 1000, street=3)
    snap, responses = bundle.runtime.request_snapshot(req)
    observation = bundle.agent.observation(snap, ctx)
    assert observation['token_ids'].shape == (1, 64)
    assert observation['match_ctx'].shape == (1, 14)
    amounts = {a: snap.me_stack if r == -2 else min(snap.to_call, snap.me_stack) if a == 1 else max(0, r)
               for a, r in responses.items()}
    out = bundle.agent.policy(snap, ctx, amounts=amounts, request=req)
    assert out.sum() == pytest.approx(1.)
    assert bundle.agent.last_evening['match_prior']['eta'] > 0
    # Large-river terminals use the existing match utility, without an RNR prior.
    s = solver_spot(4000)
    from alphanl.stage2.evening_strategy import search_policy
    _, trace = search_policy(s, ctx, np.array([0, .25, .25, .25, .25]), _random_nets(), online=True, iterations=2)
    assert trace['match_utility'] and trace['match_prior']['eta'] == 0

"""Feature boundaries, persisted CFR inputs and legacy compatibility."""
from dataclasses import replace
import json
from itertools import combinations

import numpy as np
import pytest
import torch

from alphanl.cfr import CFRConfig, SDCFRTrainer, SnapshotPolicy
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.distillation import prepare_archive, collect_trajectories, PolicyNetwork, save_student, load_student
from alphanl.cfr.evaluation import RuleActor, random_equity, Audit
from alphanl.transformer.config import ModelConfig
from alphanl.transformer.equity import estimated_equity, equity_features, preflop_key, preflop_table, _postflop
from alphanl.transformer.hunl import HunlTable, HunlTokenizer, NUMERIC_VERSION
from alphanl.transformer.model import PokerTransformer
from alphanl.transformer.tables import standard_deck


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def opening():
    table = HunlTable(seed=1)
    table.reset()
    return table, table.snapshot(1)


def tiny_config(tmp_path, **kwargs):
    return CFRConfig(iterations=2,traversals_per_player=1,traversal_batch_size=1,
        buffer_capacity=100, train_steps=1, minibatch_size=4,
        model=ModelConfig(d_model=16,nhead=2,num_layers=1,dim_feedforward=32,max_seq_len=16,dropout=0),
        out_dir=str(tmp_path), equity_features=True, **kwargs)


def test_all_1326_preflop_combinations_have_correct_lookup_class():
    _, snapshot = opening()
    counts = {}
    for hole in combinations(standard_deck(),2):
        key = preflop_key(hole)
        counts[key] = counts.get(key,0)+1
        equity = estimated_equity(replace(snapshot,hole_cards=hole))
        assert 0 <= equity <= 1
    assert len(counts) == 169
    assert sum(counts.values()) == 1326
    assert all(n == (4 if k.endswith('s') else 12 if k.endswith('o') else 6) for k,n in counts.items())
    assert .82 < preflop_table()['AA'] < .88
    assert .30 < preflop_table()['72o'] < .37


def test_equity_ties_nuts_cache_and_public_card_order():
    _, s = opening()
    # Royal flush on board means every legal opposing hand ties.
    tie = replace(s,hole_cards=('C2','D3'),board_cards=('SA','SK','SQ','SJ','ST'),street=3)
    assert estimated_equity(tie,samples=8) == .5
    nuts = replace(s,hole_cards=('SA','SK'),board_cards=('SQ','SJ','ST','D2','C3'),street=3)
    assert estimated_equity(nuts,samples=8) == 1.
    before = _postflop.cache_info().hits
    assert estimated_equity(replace(nuts,hole_cards=tuple(reversed(nuts.hole_cards)),
        board_cards=tuple(reversed(nuts.board_cards))),samples=8) == 1.
    assert _postflop.cache_info().hits > before
    with pytest.raises(ValueError,match='distinct'):
        estimated_equity(replace(s,board_cards=('SA','SA','D2')))


def test_odds_features_preserve_legacy_tensors_and_short_call_cost():
    _, s = opening()
    legacy = HunlTokenizer(8).encode(s)
    feature = HunlTokenizer(8,equity_features=True).encode(s)
    for key in legacy:
        if key != 'numeric':
            assert torch.equal(legacy[key],feature[key])
    expected = torch.tensor(equity_features(s))
    assert expected[1] == .25
    assert torch.equal(feature['numeric'][0,2,:3],expected)
    feature['numeric'][0,2] = 0.
    assert torch.equal(legacy['numeric'],feature['numeric'])
    assert equity_features(replace(s,to_call=0))[1] == 0
    assert equity_features(replace(s,stacks=(19900.,20.),to_call=100,pot=200))[1] == pytest.approx(20/220)
    assert HunlTokenizer().numeric_version == NUMERIC_VERSION


def test_observation_never_uses_hidden_deck_or_opponent_hole_cards():
    table, _ = opening()
    while table.street == 0 and not table.is_over():
        table.step(1)
    s = table.snapshot(table.current_player())
    tokenizer = HunlTokenizer(equity_features=True,equity_samples=8)
    rng_state = table._py_random.getstate()
    before = tokenizer.encode(s)
    # Replace all hidden underlying table cards and the future deck while
    # keeping the information set unchanged. Tokenizer only receives s.
    table.state.deck_cards.clear()
    table.state.hole_cards[1-s.me_seat].clear()
    after = tokenizer.encode(s)
    assert all(torch.equal(before[k],after[k]) for k in before)
    assert table._py_random.getstate() == rng_state


def test_rule_baseline_sampling_is_bit_identical_after_refactor():
    _,s = opening()
    # Historical algorithm pinned here to protect the established opponent.
    from phevaluator import evaluate_cards
    rng = np.random.default_rng(41)
    known = set(s.hole_cards+s.board_cards)
    remaining = [c[1]+c[0].lower() for c in standard_deck() if c not in known]
    hole = [c[1]+c[0].lower() for c in s.hole_cards]
    score = 0.
    for _ in range(16):
        draw = rng.choice(len(remaining),size=7,replace=False)
        board = [remaining[i] for i in draw[:5]]
        own = evaluate_cards(*(hole+board))
        opp = evaluate_cards(*([remaining[i] for i in draw[5:]]+board))
        score += float(own<opp)+.5*float(own==opp)
    assert random_equity(s,np.random.default_rng(41),16) == score/16


def test_feature_collection_buffer_checkpoint_distillation_and_actor(tmp_path):
    cfg = tiny_config(tmp_path/'train')
    with SDCFRTrainer(cfg) as trainer:
        trainer.run_iteration()
        batch,_,_ = trainer.shared_reservoir.sample(16)
        assert (batch['numeric'][:,2,0] > 0).all()
        trainer.save_checkpoint(tmp_path/'train.pt')
    with SDCFRTrainer(replace(cfg,out_dir=str(tmp_path/'resume'))) as restored:
        restored.load_checkpoint(tmp_path/'train.pt')
        assert restored.iteration == 1
        snapshot = restored._hunl_root(1)._snapshot()
    policy = SnapshotPolicy(tmp_path/'train'/'strategies')
    policy.begin_hand(1)
    assert policy.config.equity_features
    # Wrong schemas cannot load weights just because their tensor shapes match.
    with SDCFRTrainer(replace(cfg,equity_features=False,out_dir=str(tmp_path/'wrong'))) as wrong:
        with pytest.raises(ValueError,match='version mismatch'):
            wrong.load_checkpoint(tmp_path/'train.pt')
    config,manifest = prepare_archive(tmp_path/'train'/'strategies')
    data,_ = collect_trajectories(tmp_path/'train'/'strategies',config,manifest,hands=4,seed=5)
    assert (data['numeric'][:,2,0] > 0).all()
    student = PolicyNetwork(config).eval()
    save_student(tmp_path/'student.pt',student,manifest,{})
    loaded,_ = load_student(tmp_path/'student.pt')
    with torch.no_grad():
        assert torch.equal(student(data),loaded(data))
    from alphanl.cfr.evaluation import CFRActor
    actor = CFRActor(tmp_path/'train'/'strategies',seed=17)
    actor.begin_hand(1)
    assert actor.tokenizer.equity_features
    assert actor.act(snapshot) in snapshot.legal_action_ids
    # The existing residual collector must use the frozen archive's feature
    # schema as well, without introducing a new Stage 2 opponent/PPO objective.
    from alphanl.exploiter import FrozenBase, ResidualPolicy
    from alphanl.exploiter.train import collect_match
    rollout = collect_match(ResidualPolicy(FrozenBase(tmp_path/'train'/'strategies')), seed=9)
    equity = rollout.batch['numeric'][:,2,0]
    assert ((equity >= 0) & (equity <= 1)).all() and (equity > 0).any()
    assert torch.allclose(rollout.batch['numeric'][:,2,2], equity - rollout.batch['numeric'][:,2,1])


def test_scaled_regret_action_gap_and_sb_open_denominator():
    table,s = opening()
    guard = CFRActionGuard()
    s = guard.apply(s)
    assert s.legal_action_ids == (0,1,3)
    audit = Audit()
    audit.begin_hand(1)
    audit.observe(s,np.array([1.,0,0,0,0]),0,0.)
    # A later preflop decision increases decision-fold denominator only.
    table.step(1)
    table.step(3)
    audit.observe(guard.apply(table.snapshot(1)),np.array([0.,1,0,0,0]),1,0.)
    assert audit.summary()['sb_open_fold_ratio'] == 1
    assert audit.summary()['counts_by_seat']['1:preflop_decisions'] == 2


def test_mixed_feature_archive_rejected_even_when_parameter_shapes_match(tmp_path):
    cfg = tiny_config(tmp_path)
    with SDCFRTrainer(cfg) as trainer:
        for t in (1,2):
            blob = trainer._snapshot(t)
            if t == 2:
                legacy = replace(cfg,equity_features=False)
                blob.update(config=legacy.to_dict(),numeric_version=legacy.numeric_version)
            directory = tmp_path/'strategies'
            directory.mkdir(exist_ok=True)
            torch.save(blob,directory/f'iteration_{t:07d}.pt')
    p = SnapshotPolicy(directory,seed=1)
    p.rng = np.random.default_rng(3)  # selects iteration 1
    assert p.begin_hand(1) == 1
    p.rng = np.random.default_rng(0)  # selects iteration 2
    with pytest.raises(ValueError,match='different network configurations'):
        p.begin_hand(1)


def test_parallel_real_hunl_injects_feature_before_shared_memory_transport(tmp_path):
    cfg = tiny_config(tmp_path,traversal_workers=1,worker_sample_batch=8)
    with SDCFRTrainer(cfg) as trainer:
        trainer.run_iteration()
        batch,_,_ = trainer.shared_reservoir.sample(32)
        assert (batch['numeric'][:,2,0] > 0).all()
        assert (batch['numeric'][:,2,0] <= 1).all()


def test_paired_feature_comparison_uses_pair_means_and_matches_decks(tmp_path):
    from scripts.eval_equity_ablation import paired_difference, validate_controls
    base=[{'pair':0,'deck_seed':1,'chips_sb_bb':[100,-100]},
          {'pair':1,'deck_seed':2,'chips_sb_bb':[0,0]}]
    candidate=[{'pair':0,'deck_seed':1,'chips_sb_bb':[200,0]},
               {'pair':1,'deck_seed':2,'chips_sb_bb':[0,0]}]
    summary=paired_difference(base,candidate)
    assert summary['n']==2 and summary['mean']==500
    assert summary['ci95']==pytest.approx([-480,1480])
    candidate[1]['deck_seed']=3
    with pytest.raises(ValueError,match='identical decks'):
        paired_difference(base,candidate)
    cfg=tiny_config(tmp_path)
    validate_controls(replace(cfg,equity_features=False),cfg)
    with pytest.raises(ValueError,match='learning_rate'):
        validate_controls(replace(cfg,equity_features=False,learning_rate=.01),cfg)


def test_legacy_distilled_checkpoint_without_equity_fields_loads(tmp_path):
    from alphanl.cfr.distillation import FORMAT, SEMANTIC_FIELDS
    from alphanl.transformer.hunl import RULES_VERSION
    cfg=replace(tiny_config(tmp_path),equity_features=False)
    model=PolicyNetwork(cfg).eval()
    fields=('equity_features','equity_samples','equity_seed')
    config={k:v for k,v in cfg.to_dict().items() if k not in fields}
    semantics={k:v for k,v in config.items() if k in SEMANTIC_FIELDS}
    path=tmp_path/'old_student.pt'
    torch.save({'format':FORMAT,'rules_version':RULES_VERSION,'numeric_version':NUMERIC_VERSION,
                'config':config,'teacher':{'semantics':semantics},'model':model.state_dict()},path)
    restored,_=load_student(path)
    _,s=opening()
    batch=HunlTokenizer(cfg.model.max_seq_len).encode(s)
    with torch.no_grad():
        assert torch.equal(model(batch),restored(batch))

"""Production precision/budget separation, vector evaluator and pool contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
from alphanl.stage2 import deep_solver_v2 as v
from alphanl.stage2 import cfv

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/sprint24'))
from evening_production import plan_counts,ONLINE,OFFLINE,find_champion
from evening_parallel import jobs,worker_environment,default_workers


def test_vector_evaluator_exact_all_card_counts(monkeypatch):
    monkeypatch.setattr(v,'_PE',None)
    rng=np.random.RandomState(41)
    for count in (5,6,7):
        for j in range(20):
            board=rng.choice(v.DECK,count-2,replace=False).tolist()
            remaining=[c for c in v.DECK if c not in board]
            hands=[tuple(rng.choice(remaining,2,replace=False)) for i in range(128)]
            actual=v.pack_many(hands,board)
            expected=np.array([v._pack(list(h)+board) for h in hands])
            np.testing.assert_array_equal(actual,expected)


def test_policy_only_is_exact_same_average_without_extra_q_walk():
    from evening_pipeline import sample_spot
    s,rh,ro=sample_spot(901,3)
    full=v.solve_public_subgame(s,rh,ro,iterations=5)
    fast=v.solve_public_subgame(s,rh,ro,iterations=5,policy_only=True)
    np.testing.assert_array_equal(fast.average,full.average)
    np.testing.assert_array_equal(fast.state,full.state)
    assert abs(fast.trace['zero_sum_residual'])<1e-9


def test_gpu_runtime_precision_is_not_label_budget():
    assert ONLINE['iterations']=={'1':4,'2':4,'3':8}
    assert ONLINE['chance_samples']==1
    assert OFFLINE['river_iterations']>=64
    assert OFFLINE['turn_iterations']>=32
    assert OFFLINE['teacher_iterations']>=32
    p=plan_counts({'river':.003,'turn':.01,'rebel':.008},3.,600.,60.,24)
    assert p['predicted_seconds']<=3.5*3600
    assert 8192<=p['river_n']<=300000
    assert 4096<=p['turn_n']<=200000
    assert 4096<=p['rebel_n']<=200000
    with pytest.raises(RuntimeError):plan_counts({'river':10,'turn':20,'rebel':15},2.,5000.,500.,24)


def test_spawn_workers_enforce_single_thread_and_hide_cuda():
    assert 1<=default_workers()<=24
    rows=list(jobs(worker_environment,range(6),6,2,label='THREAD_TEST'))
    for row in rows:
        assert row['torch_threads']==row['interop_threads']==1
        assert row['environment']=={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}


def test_parallel_labels_match_serial(tmp_path):
    from evening_pipeline import make_dataset
    x,y=make_dataset(tmp_path/'serial.npz',3,4,1001,iterations=3,workers=1)
    xx,yy=make_dataset(tmp_path/'parallel.npz',3,4,1001,iterations=3,workers=2)
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
    _,m=cfv.load_artifact(tmp_path/'parallel.npz','dataset',4)
    assert m['workers']==2 and m['iterations']==3


def test_blueprint_prior_belongs_only_to_current_private_root_infoset():
    from evening_pipeline import sample_spot
    s,rh,ro=sample_spot(901,3)
    at=next(i for i,h in enumerate(v._public(tuple(sorted(cfv.card(c) for c in s.board_cards)))[0]) if frozenset(h)==frozenset(s.hole_cards))
    prior=np.array([.1,.2,.3,.25,.15])
    sol=v.PublicSolver(s,rh,ro,root_prior=prior,hero_at=at,iterations=4)
    projected=prior[sol.tree.labels];projected/=projected.sum()
    np.testing.assert_array_equal(sol.tree.strategy[at],projected)
    other=next(i for i in range(sol.width) if i!=at)
    np.testing.assert_allclose(sol.tree.strategy[other],1/len(sol.tree.labels))
    sol.update_strategies(sol.tree)
    np.testing.assert_array_equal(sol.tree.strategy[at],projected)
    sol.tree.regret[at]=0;sol.tree.regret[at,1]=3
    sol.update_strategies(sol.tree)
    assert sol.tree.strategy[at,1]==1


def test_teacher_is_pure_and_online_mixture_tracks_updated_blueprint():
    from test_evening_v2 import snap,_random_nets
    from alphanl.stage2.evening_strategy import search_policy
    from evening_cd import search_label
    s=snap();weights=_random_nets()
    first=np.array([.8,.2,0,0,0]);second=np.array([0,.2,0,0,.8])
    teacher1=search_label(s,None,first,weights,iterations=8)
    teacher2=search_label(s,None,second,weights,iterations=8)
    np.testing.assert_array_equal(teacher1,teacher2)
    out1,trace1=search_policy(s,None,first,weights,iterations=8,online=True)
    out2,trace2=search_policy(s,None,second,weights,iterations=8,online=True)
    np.testing.assert_allclose(out1,.3*first+.7*np.array(trace1['dcfr_average']),atol=1e-14)
    np.testing.assert_allclose(out2,.3*second+.7*np.array(trace2['dcfr_average']),atol=1e-14)
    assert .5*np.abs(out1-out2).sum()>.05
    assert trace1['initial_strategy']=='hero_root_blueprint'
    assert trace1['target']=='blueprint_dcfr_mixture'
    assert search_label.last_trace['target']=='dcfr_average'


def test_one_iteration_stores_blueprint_without_fake_pseudocounts():
    from evening_pipeline import sample_spot
    s,rh,ro=sample_spot(902,3)
    s.legal_action_ids=(1,2,3,4);s.to_call=0
    hands=v._public(tuple(sorted(cfv.card(c) for c in s.board_cards)))[0]
    at=next(i for i,h in enumerate(hands) if frozenset(h)==frozenset(s.hole_cards))
    prior=np.array([0.,.4,.3,.2,.1])
    sol=v.solve_public_subgame(s,rh,ro,iterations=1,root_prior=prior,hero_at=at,policy_only=True)
    np.testing.assert_allclose(sol.average[at],prior[sol.tree.labels],atol=1e-15)
    assert sol.tree.total[at].sum()==pytest.approx(1.)


def test_failed_package_allocation_never_reuses_visible_version(tmp_path):
    from pack_wizard import pack_evening
    import json
    # Missing nets intentionally fail after reserving a visible release.
    for expected in (22,23):
        with pytest.raises(FileNotFoundError):pack_evening(tmp_path/'missing.pt',tmp_path/'nets.npz',tmp_path)
        assert json.loads((tmp_path/'EVENING_RELEASE_COUNTER.json').read_text())['last']==expected
        assert any(p.name.startswith('UPLOAD_V%d_'%expected) for p in tmp_path.iterdir())


def test_offline_teacher_selects_full_street_budget_without_online_prior(monkeypatch):
    import evening_cd
    from test_evening_v2 import snap
    seen=[]
    def teacher(snapshot,context,prior,nets,amounts,iterations,samples):
        seen.append(iterations)
        return np.array([0.,1.,0.,0.,0.]),{'target':'dcfr_average'}
    monkeypatch.setattr(evening_cd,'search_policy',teacher)
    evening_cd.search_label(snap(),None,np.ones(5)/5,{},iterations={'1':32,'2':32,'3':64})
    assert seen==[64]
    assert evening_cd.search_label.last_trace['target']=='dcfr_average'

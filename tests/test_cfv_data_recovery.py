"""Remote regression suite. Existing tests and hard holdout boundary are untouched."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/sprint24'))
from alphanl.stage2 import cfv
import cfv_recovery as r
from evening_pipeline import sample_spot,_train
from overnight_data import holdout_gate,TREE
from server_arena import profiles,assert_qualified_manifest


def spot(seed=12,street=3):
    s,h,o=sample_spot(seed,street)
    first=0 if s.me_seat==getattr(s,'postflop_first_seat',0) else 1
    return cfv.spot_features(s.board_cards,h,o,s.pot,s.to_call,s.me_stack,100.,street,s.opp_stack,first).astype(np.float32)


def zero_weights(prefix='cfv_river.',bias=0.):
    w={'cfv_schema':np.frombuffer(cfv.SCHEMA.encode(),dtype=np.uint8)}
    for i,shape in enumerate(((256,136),(256,256),(256,256),(256,256),(32,256))):
        w[prefix+str(i)+'.weight']=np.zeros(shape,dtype=np.float32)
        w[prefix+str(i)+'.bias']=np.full(shape[0],bias if i==4 else 0.,dtype=np.float32)
    return w


@pytest.mark.parametrize('counts',[{'river':79999,'turn':50000},{'river':80000,'turn':49999},{}, {'river':8192,'turn':4352}])
def test_floor_rejects_starved_data(counts):
    with pytest.raises(r.RecoveryStop,match='INSUFFICIENT_CFV_SAMPLES'):r.require_floor(counts)


def test_floor_accepts_only_sufficient_both_streets():
    r.require_floor({'river':80000,'turn':50000});r.require_floor({'river':108800,'turn':72192})


@pytest.mark.parametrize('street',[2,3])
def test_public_state_reconstruction_preserves_v3(street):
    for seed in range(16):
        x=spot(seed,street);s,h,o=r.decode_state(x)
        first=0 if x[125]>.5 else 1
        rebuilt=cfv.spot_features(s.board_cards,h,o,s.pot,s.to_call,s.me_stack,100.,street,s.opp_stack,first)
        np.testing.assert_allclose(rebuilt,x,rtol=2e-6,atol=2e-6)
        assert s.first_player==(0 if s.to_call else first)


def test_board_exclusion_catches_old_train_even_with_different_range():
    x=spot()[None,:];v=x.copy();v[0,:32]=1/32
    with pytest.raises(ValueError,match='LEAKAGE'):r.assert_disjoint(x,v)
    fresh=spot(90)[None,:]
    assert r.assert_disjoint(x,fresh)['board_overlap']==0
    with pytest.raises(ValueError,match='LEAKAGE'):r.assert_disjoint(fresh,x,r.board_ids(x))


def test_board_exclusion_is_order_independent():
    x=spot()[None,:];assert r.board_ids(x).dtype==np.uint64
    assert int(r.board_ids(x)[0]).bit_count()==5


def test_historical_import_rejects_schema_units_sources_and_corruption(tmp_path):
    p=tmp_path/'data.npz';a=dict(x=spot()[None,:],y=np.zeros((1,32),np.float32))
    m=cfv.publish(p,a,1,0,'dataset');mp=Path(str(p)+'.manifest.json');sources=m['sources']
    assert len(r.old_artifact(p,sources,'dataset')[0]['x'])==1
    for key,value in (('schema','v2'),('units','chips'),('complete',False),('sources',{}),('arrays_sha256','bad')):
        bad=dict(m,**{key:value});cfv.atomic_json(mp,bad)
        with pytest.raises(ValueError):r.old_artifact(p,sources,'dataset')
    cfv.atomic_json(mp,m);p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):r.old_artifact(p,sources,'dataset')


def test_audit_proves_fixed_evaluator_and_relabels_old_tree(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[1];root=tmp_path/'old';(root/'b').mkdir(parents=True)
    (root/'FINAL_REPORT.json').write_text('{"status":"COMPLETE"}')
    sources=cfv.source_hashes()
    for name,street in (('river',3),('turn',2)):
        a=dict(x=spot(12,street)[None,:],y=np.zeros((1,32),np.float32),seeds=np.array([12]))
        for suffix in ('ds','holdout'):
            cfv.publish(root/'b'/(name+'_'+suffix+'.npz'),a,1,0,'dataset',
                dict(street=street,active_compute_s=1.,iterations=64,chance_samples=4))
    weights=zero_weights();weights.update(zero_weights('cfv_turn.'))
    cfv.publish(root/'b/cfv_nets.npz',weights,2,30,'nets',dict(river_samples=1,turn_samples=1))
    def git_source(command):return (repo/command[-1].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(r.subprocess,'check_output',git_source)
    audit=r.inspect_history(root,repo,tmp_path/'audit')
    assert audit['schema_sha256']==r.schema_hash() and audit['warm_start']['compatible']
    assert audit['evaluator']['version'].endswith('post-fix')
    for name in ('river','turn'):
        assert not audit['datasets'][name]['direct_labels_compatible']
        assert audit['datasets'][name]['action']=='reuse_public_states_and_relabel'
    # A new schema string cannot conceal a changed feature/evaluator source.
    bad=dict(sources);bad['cfv.py']='bad'
    with pytest.raises(ValueError):r.old_artifact(root/'b/river_ds.npz',bad,'dataset')


def test_warm_start_copies_weights_and_runs_new_epochs():
    x=np.stack([spot(),spot(14)]);y=np.full((2,32),.125,np.float32)
    initial=zero_weights(bias=.125)
    out,stats=_train(x,y,'cfv_river.',1,1,'cpu',(x,y),cosine=True,initial_weights=initial)
    assert stats['warm_started'] and stats['epochs_completed']==1
    for key in initial:np.testing.assert_array_equal(out[key],initial[key])
    bad=dict(initial);bad['cfv_river.0.weight']=np.zeros((256,135))
    with pytest.raises(ValueError,match='invalid CFV layer'):_train(x,y,'cfv_river.',1,initial_weights=bad)


def plan_inputs():
    audit=dict(datasets={k:dict(samples=n) for k,n in [('river',100608),('turn',68096)]})
    bench={k:dict(selected=dict(labels_per_second=100.,iterations=i,chance_samples=4)) for k,i in [('river',32),('turn',16)]}
    rates=dict(river=.05,turn=.05,rebel=.005,anchor=.001,cfv_epoch=.000001,policy_epoch=.000001)
    return audit,bench,rates


def test_recovery_plan_fits_all_data_and_online_is_unchanged():
    a,b,rates=plan_inputs();plan=r.choose_plan(a,b,rates,8.5,10,1.,1.,24)
    assert (plan['river_n'],plan['turn_n'])==(108800,72192)
    assert plan['net_epochs']==50 and plan['holdout_n']==2048 and len(plan['arms'])==4
    assert plan['offline']['river']['iterations']==32
    for profile in profiles().values():assert profile['iterations']=={'1':8,'2':8,'3':16}


def test_insufficient_budget_stops_instead_of_low_sample_training():
    a,b,rates=plan_inputs();rates.update(river=10.,turn=10.)
    with pytest.raises(r.RecoveryStop,match='INSUFFICIENT_LABEL_BUDGET'):r.choose_plan(a,b,rates,8.5,0,20,10,24)


def test_bad_floor_stops_before_warm_load_or_optimizer(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'_train',lambda *a,**kw:pytest.fail('must not train'))
    with pytest.raises(r.RecoveryStop,match='INSUFFICIENT_CFV_SAMPLES'):
        r.phase_b('absent',tmp_path,dict(net_epochs=50,holdout_n=2048,river_n=8192,turn_n=4352),24,'cpu',{})


def test_holdout_boundary_is_still_point_twelve(monkeypatch):
    x=np.zeros((2048,136));x[:,119]=1/200;y=np.ones((2048,32))
    # Constant is 1 BB worse; learned error is still >0.12 pot.
    monkeypatch.setattr(cfv,'predict_residual_cfv',lambda *args:y+.1201)
    gate=holdout_gate({},'cfv_river.',np.zeros((4,32)),(x,y))
    assert gate['pot_mae_ratio']>.12 and not gate['passed']


def test_persisted_gate_failure_cannot_retry_same_seed_fit(tmp_path,monkeypatch):
    a,_,_=plan_inputs();a.update(warm_start=dict(path='unused',sha256='unused'),sources={})
    plan=dict(net_epochs=50,holdout_n=2048,river_n=108800,turn_n=72192,
              reused_rows={'river':100608,'turn':68096},new_rows=r.NEW_SAMPLES,offline={'river':{},'turn':{}})
    (tmp_path/'river_REJECTED.json').write_text('{}')
    monkeypatch.setattr(cfv,'digest',lambda *args:'unused')
    monkeypatch.setattr(r,'_train',lambda *a,**kw:pytest.fail('identical fit retry'))
    # Warm loader validates both heads, so include them for this isolation test.
    w=zero_weights();w.update(zero_weights('cfv_turn.'))
    monkeypatch.setattr(r,'old_artifact',lambda *args:(w,{}))
    with pytest.raises(r.RecoveryStop,match='persisted rejection'):r.phase_b('unused',tmp_path,plan,24,'cpu',a)

"""Public-request regressions from V22, with no revealed opponent hole cards."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/sprint24'))
from alphanl.stage2 import cfv,deep_solver_v2 as solver
from alphanl.stage2.search_belief import history_combo_ranges
from alphanl.stage2.deployment_guard import apply_guard,apply_search_guard,guard_config,V102_R7_GUARD,good_hand_blind_side
from alphanl.transformer.actor_actions import PotFractionBet,apply_actor_decision
from alphanl.transformer.schema import HandSnapshot,ActionEvent
from ladder_actors import SlayerActor

CASES=json.loads((ROOT/'tests/fixtures/overnight_bot17_audit.json').read_text())


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    from evening_production import find_champion
    from overnight_production import ONLINE
    from overnight_data import TREE
    from evening_d import Bundle
    from test_evening_v2 import _random_nets
    folder=tmp_path_factory.mktemp('bot17_zip')
    nets=folder/'nets.npz'
    cfv.publish(nets,_random_nets(),2,1,'nets',dict(tree=TREE,river_samples=1,turn_samples=1))
    model=folder/'V999_MODEL.npz';source=folder/'V999_SOURCE.zip'
    export_botzone_weights(find_champion(),model,release_id='V999_BOT17_REGRESSION',cfv_nets=nets,
        online_profile=ONLINE,deployment_guard=V102_R7_GUARD)
    pack_botzone_zip(source,release_id='V999_BOT17_REGRESSION')
    return Bundle(source,model)


def case(bundle,hand,call,match='match2'):
    row=next(r for r in CASES if r['case']==f'{match}_h{hand}_s'+str(0 if hand==36 else 1 if hand in (7,22) else 3)+'_c'+str(call))
    snap,resp=bundle.runtime.request_snapshot(row['request'])
    amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in resp.items()}
    return row,snap,amounts


@pytest.mark.parametrize('hand,call,raw',[(7,101,[.853,.123,.015,.009,0]),(22,192,[.848,.107,.026,.020,0])])
def test_audited_high_fold_flops_survive_all_legacy_rules(bundle,hand,call,raw):
    _,snap,amounts=case(bundle,hand,call)
    raw=np.array(raw);raw/=raw.sum()
    # Fixed equity from the audit makes the old forced-call failure reproducible.
    equity=.267 if hand==7 else .185
    final,info=apply_guard(snap,raw,guard_config(V102_R7_GUARD),lambda *a,**kw:equity,amounts,search_active=True)
    np.testing.assert_allclose(final,raw,atol=1e-15)
    assert final[0]>.84 and final[1]<.13 and info['search_policy_preserved']


def test_hand36_large_open_keeps_original_blueprint_fold(bundle):
    _,snap,amounts=case(bundle,36,241)
    raw=np.array([.485,.461,.017,.036,0]);raw/=raw.sum()
    config=guard_config(V102_R7_GUARD)
    assert good_hand_blind_side(snap,config,lambda:.60) is None
    final,info=apply_guard(snap,raw,config,lambda *a,**kw:.60,amounts)
    np.testing.assert_allclose(final,raw,atol=1e-15)
    assert final[0]>.48 and final[1]<.47 and info['large_open_blueprint_preserved']


@pytest.mark.parametrize('call',[219,220,241])
def test_large_open_boundary_is_relative_to_big_blind(bundle,call):
    from dataclasses import replace
    _,snap,amounts=case(bundle,36,241)
    event=replace(snap.events[0],chips_added=call+50)
    snap=replace(snap,to_call=float(call),events=(event,),pot=float(200+call))
    side=good_hand_blind_side(snap,guard_config(V102_R7_GUARD),lambda:.7)
    assert side==('bb' if call<220 else None)


def test_hand17_full_combo_conditioning_reaches_direct_solver_and_folds(bundle):
    _,snap,_=case(bundle,17,16112,'match3')
    (rh,ro),audit=history_combo_ranges(snap,return_audit=True)
    assert rh.shape==ro.shape==(1081,) and ro.min()>0 and ro.max()>100*ro.min()
    assert audit['conditioned_actions'][1]>=3
    direct=solver.PublicSolver(snap,iterations=40,policy_only=True,check_raise=True).solve()
    at=next(i for i,h in enumerate(direct.hands) if frozenset(h)==frozenset(cfv.card(c) for c in snap.hole_cards))
    np.testing.assert_allclose(direct.ro,ro,atol=1e-15)
    assert direct.average[at,direct.tree.labels.index(1)]<.01
    assert direct.eq(direct.ro)[0][at]<snap.to_call/(snap.pot+snap.to_call)
    uniform=solver.PublicSolver(snap,opponent_range=np.ones(1081),iterations=40,policy_only=True).solve()
    assert uniform.average[at,uniform.tree.labels.index(1)]>.95
    assert abs(direct.trace['zero_sum_residual'])<1e-10
    final,info=apply_search_guard(snap,np.array([.032,.968,0,0,0]))
    np.testing.assert_array_equal(final,[1.,0.,0.,0.,0.])
    assert info['call_fuse']=='history_posterior_giant_allin'


def test_range_inputs_ignore_opponent_cards_and_actual_hero_hole(bundle):
    from dataclasses import replace
    _,snap,_=case(bundle,17,16112,'match3')
    r=history_combo_ranges(snap)
    changed=replace(snap,hole_cards=('HA','SK'))
    rr=history_combo_ranges(changed)
    for a,b in zip(r,rr):np.testing.assert_array_equal(a,b)
    nohistory=replace(snap,events=())
    plain=history_combo_ranges(nohistory)[1]
    np.testing.assert_allclose(plain,np.ones(1081)/1081)
    assert not np.allclose(r[1],plain)


def test_giant_safety_preserves_nuts_and_does_not_treat_allin_calls_as_raises(bundle):
    from dataclasses import replace
    _,snap,_=case(bundle,17,16112,'match3')
    nuts=replace(snap,hole_cards=('S2','H2'))
    p,info=apply_search_guard(nuts,np.array([.1,.9,0,0,0]))
    np.testing.assert_allclose(p,[.1,.9,0,0,0])
    assert info['posterior_equity']==pytest.approx(1.)
    polar=history_combo_ranges(snap,return_audit=True)[1]['opponent_entropy']
    events=list(snap.events);events[-1]=replace(events[-1],to_call_before=events[-1].chips_added)
    calling=history_combo_ranges(replace(snap,events=tuple(events)),return_audit=True)[1]['opponent_entropy']
    assert calling>polar
    stacks=list(snap.stacks);stacks[snap.me_seat]=100.
    short=replace(snap,stacks=tuple(stacks))
    _,short_info=apply_search_guard(short,np.array([.1,.9,0,0,0]))
    assert short_info['pot_odds']==pytest.approx(100./(snap.pot-(snap.to_call-100.)+100.))


def test_hand9_real_river_bluff_catch_and_runtime_guard_order(bundle,monkeypatch):
    row,snap,amounts=case(bundle,9,295)
    agent=bundle.agent
    raw=np.array([.630,.341,.017,.012,0.])
    monkeypatch.setattr(agent,'logits',lambda batch:np.log(np.maximum(raw,1e-30))[None,:])
    ctx,_=bundle.runtime.context_from_payload({'requests':[row['request']]})
    final=agent.policy(snap,ctx,amounts=amounts,request=row['request'])
    mixed=.3*raw+.7*np.asarray(agent.last_evening['dcfr_average'])
    np.testing.assert_allclose(final,mixed/mixed.sum(),atol=1e-12)
    assert final[0]<.30 and final[1]>.50
    assert agent.last_guard['search_policy_preserved']


def test_hand17_actual_zip_policy_cannot_revive_giant_call(bundle,monkeypatch):
    row,snap,amounts=case(bundle,17,16112,'match3')
    agent=bundle.agent
    monkeypatch.setattr(agent,'logits',lambda batch:np.log(np.maximum([.032,.968,0,0,0],1e-30))[None,:])
    ctx,_=bundle.runtime.context_from_payload({'requests':[row['request']]})
    p=agent.policy(snap,ctx,amounts=amounts,request=row['request'])
    np.testing.assert_allclose(p,[1,0,0,0,0],atol=1e-12)
    assert agent.last_guard['call_fuse']=='history_posterior_giant_allin'


@pytest.mark.parametrize('street',[1,2,3])
def test_all_streets_and_preflop_size_condition_current_combo_ranges(bundle,street):
    from dataclasses import replace
    _,snap,_=case(bundle,17,16112,'match3')
    first=ActionEvent(0,1,'BB',False,3,'raise',150,291,291/150.,0,50,19950,(0,1,2,3,4))
    small=replace(first,chips_added=160,pot_fraction=160/150.)
    snap=replace(snap,street=street,board_cards=snap.board_cards[:street+2],events=(first,))
    r=history_combo_ranges(snap)[1];less=history_combo_ranges(replace(snap,events=(small,)))[1]
    assert len(r)==(1176 if street==1 else 1128 if street==2 else 1081)
    assert not np.allclose(r,less)
    assert np.isfinite(r).all() and (r>0).all()


@pytest.mark.parametrize('equity,added',[(.37,0),(.40,50),(.43,210),(.48,233),(.50,235),(.58,285),(.586,291),(.82,291)])
def test_replica_sb_strength_sizes_are_actual_added_chips(monkeypatch,equity,added):
    snap=SimpleNamespace(street=0,events=(),me_seat=1,sb_seat=1,to_call=50.,pot=150.,me_stack=19950.,big_blind=100.,legal_action_ids=(0,1,2,3,4))
    actor=SlayerActor('Bot17_Probe3Bet',seed=7);monkeypatch.setattr(actor,'_equity',lambda s:equity)
    action=actor.act(snap)
    if added==0:assert action==0
    elif added==50:assert action==1
    else:assert isinstance(action,PotFractionBet) and round(snap.to_call+action.fraction*snap.pot)==added


@pytest.mark.parametrize('pot',[200.,400.,918.,1000.])
def test_replica_weak_postflop_probe_sizes(monkeypatch,pot):
    snap=SimpleNamespace(street=3,events=(),me_seat=0,sb_seat=1,to_call=0.,pot=pot,me_stack=19000.,big_blind=100.,
        legal_action_ids=(1,2,3,4),hole_cards=('HJ','DQ'),board_cards=('H6','SK','CT','D3','C7'))
    actor=SlayerActor('Bot17_Probe3Bet',seed=5);monkeypatch.setattr(actor,'_equity',lambda s:.2)
    action=actor.act(snap);assert isinstance(action,PotFractionBet)
    assert action.fraction==.5 if pot==200. else .25<=action.fraction<=.34


def test_replica_offtree_open_executes_in_both_engine_and_judge(bundle,monkeypatch):
    from alphanl.transformer.hunl import HunlTable
    from evening_d import actor_response
    table=HunlTable(seed=7);table.reset()
    actor=SlayerActor('Bot17_Probe3Bet',seed=7);monkeypatch.setattr(actor,'_equity',lambda s:.65)
    apply_actor_decision(table,actor.act(table.snapshot(1)))
    assert table.events_for(0)[-1].chips_added==291
    req=copy.deepcopy(CASES[0]['request']);req.update(history=[],public_cards=[],my_id=1,dealer_id=0,my_chips=19950)
    snap,options=bundle.runtime.request_snapshot(req)
    assert actor_response(bundle,snap,req,options,actor.act(snap))==291


def test_replica_late_made_hand_overbet_and_fullhouse_jam(monkeypatch):
    from dataclasses import replace
    snap=HandSnapshot(events=(),hole_cards=('H6','H2'),board_cards=('D2','S7','C5','C2'),
        me_seat=0,sb_seat=1,street=2,pot=800.,stacks=(19800.,19400.),to_call=400.,
        legal_action_ids=(0,1,2,3,4),big_blind=100.,starting_stack=20000.)
    actor=SlayerActor('Bot17_Probe3Bet',seed=7);monkeypatch.setattr(actor,'_equity',lambda s:.8)
    decision=actor.act(snap)
    assert isinstance(decision,PotFractionBet) and round(400+decision.fraction*800)==1744
    river=replace(snap,street=3,board_cards=snap.board_cards+('D6',),pot=5832.,to_call=1944.)
    assert actor.act(river)==4


def test_bot17_is_in_teacher_sampling_and_five_opponent_evaluation_pool(bundle):
    from evening_production_eval import TOURNAMENT
    from overnight_data import trajectory
    from evening_production import find_champion
    from alphanl.stage2.train import load_checkpoint
    from test_evening_v2 import _random_nets
    model,_=load_checkpoint(find_champion())
    items=trajectory(model,4,73117,teacher=True,nets=_random_nets(),iterations=2,chance_samples=1,opponent='Bot17_Probe3Bet')
    assert 'Bot17_Probe3Bet' in TOURNAMENT and len(TOURNAMENT)==5
    assert all(r['opponent']=='Bot17_Probe3Bet' and np.isfinite(r['value']) for r in items)


def test_history_observation_model_imports_without_repository_root(tmp_path):
    code="from alphanl.stage2.search_belief import history_combo_ranges; from types import SimpleNamespace; s=SimpleNamespace(board_cards=('2s','3h','4d','9c','Ks'),street=3,big_blind=100.,events=()); print(len(history_combo_ranges(s)[1]))"
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='1081'

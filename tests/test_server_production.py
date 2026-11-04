"""Server-profile properties, strict provenance, isolated resources and restart contracts."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from test_overnight_bot17_repair import bundle,case
from alphanl.stage2.deployment_guard import apply_surgical_guard,guard_config,V102_R7_GUARD
from alphanl.stage2.solver_prior import profile
from server_benchmark import gates,sample_states
from server_arena import profiles,assert_qualified_manifest,assert_production_cfv,match_jobs,compare_arms,rank_arms,interval
from server_resources import core_plan,cpu_list
from server_checkpoint import Checkpoints
from server_packaging import reserve
from overnight_data import TREE

CONFIG=guard_config(V102_R7_GUARD)


def equity(value):return lambda *a,**kw:value


@pytest.mark.parametrize('seed',[51,911,2007,20261105])
def test_surgical_random_simplex_and_state_properties(bundle,seed):
    """4096 random distributions/states per seed, covering all streets/prices."""
    rng=np.random.default_rng(seed)
    _,base,_=case(bundle,7,101)
    for index in range(4096):
        street=index%4
        board=() if street==0 else base.board_cards if street==1 else base.board_cards+('cJ',) if street==2 else base.board_cards+('cJ','d9')
        free=bool(rng.integers(2));owed=0. if free else float(rng.integers(1,16000))
        stack=float(rng.integers(max(2,int(owed)),20001));pot=float(rng.integers(max(200,int(owed)),40001))
        legal=tuple([0,1]+[a for a in (2,3,4) if rng.random()>.2])
        s=replace(base,street=street,board_cards=board,to_call=owed,pot=pot,stacks=tuple(stack if i==base.me_seat else base.stacks[i] for i in (0,1)),legal_action_ids=legal)
        raw=rng.dirichlet(np.full(5,.35));raw[[a for a in range(5) if a not in legal]]=0.;raw/=raw.sum()
        amounts={0:0.,1:min(stack,owed),2:min(stack,owed+pot*.5),3:min(stack,owed+pot),4:stack}
        amounts={a:amounts[a] for a in legal}
        p,info=apply_surgical_guard(s,raw,CONFIG,equity(float(rng.random())),amounts)
        assert np.isfinite(p).all() and (p>=0).all() and p.sum()==pytest.approx(1.)
        assert all(p[a]==0 for a in range(5) if a not in legal)
        if free:assert p[0]==0  # explicit free Fold -> Check exemption
        else:
            assert p[0]+1e-12>=raw[0]
            assert not (p[1]==1. and raw[1]!=1.)
        if street==3:
            expected=raw.copy()
            if free:expected[1]+=expected[0];expected[0]=0
            np.testing.assert_allclose(p,expected,atol=1e-14)
            assert info['river_solver_untouched']


def test_legal_mask_exemption_can_remove_fold_and_force_only_legal_call(bundle):
    _,s,_=case(bundle,7,101)
    s=replace(s,legal_action_ids=(1,))
    p,info=apply_surgical_guard(s,[.9,.1,0,0,0],CONFIG,equity(.1),{1:101})
    np.testing.assert_array_equal(p,[0,1,0,0,0])
    assert info['rules']['legal_mask_normalize']['mass_moved']>0


def test_caps_treat_call_and_jam_symmetrically_and_do_not_manufacture_pure_call(bundle):
    _,s,_=case(bundle,7,101)
    s=replace(s,pot=10000.,to_call=8000.,stacks=tuple(16000. if i==s.me_seat else s.stacks[i] for i in (0,1)),legal_action_ids=(0,1,2,3,4))
    amounts={0:0,1:8000,2:10000,3:12000,4:16000}
    p,info=apply_surgical_guard(s,[.1,.4,.1,.1,.3],CONFIG,equity(.6),amounts)
    np.testing.assert_allclose(p,[1,0,0,0,0])
    assert {'cap_call','cap_raise','cap_all_in'}<=set(info['rules'])
    s=replace(s,to_call=100.)
    amounts[1]=100
    p,info=apply_surgical_guard(s,[0,0,0,0,1.],CONFIG,equity(.6),amounts)
    np.testing.assert_allclose(p,[.05,.95,0,0,0])
    assert info['rules']['no_forced_call']['mass_moved']==pytest.approx(.05)
    p,_=apply_surgical_guard(s,[.1,.1,.1,.1,.6],CONFIG,equity(.2),amounts)
    assert p[0]>.7 and p[1]<=.1+1e-12


def test_free_caps_check_and_river_caps_are_disabled(bundle):
    _,s,_=case(bundle,7,101)
    s=replace(s,to_call=0.,pot=400.,legal_action_ids=(0,1,2,3,4))
    p,info=apply_surgical_guard(s,[.2,.1,0,0,.7],CONFIG,equity(.2),{0:0,1:0,2:200,3:400,4:19900})
    np.testing.assert_allclose(p,[0,1,0,0,0])
    assert 'free_fold_to_check' in info['rules'] and 'cap_all_in' in info['rules']
    _,river,amounts=case(bundle,17,16112,'match3')
    raw=np.array([.1,.8,0,0,.1]);p,info=apply_surgical_guard(river,raw,CONFIG,equity(.01),amounts)
    legal=set(river.legal_action_ids)&set(amounts);expected=raw.copy()
    expected[[a for a in range(5) if a not in legal]]=0;expected/=expected.sum()
    np.testing.assert_array_equal(p,expected)
    assert info['river_solver_untouched'] and not any(k.startswith('cap_') for k in info['rules'])


def test_lock_exemptions_through_real_export(bundle,monkeypatch):
    monkeypatch.setitem(bundle.agent.settings,'evening_online',dict(profiles()['C']))
    row,_,_=case(bundle,9,0);req=copy.deepcopy(row['request']);req['total_win_chips']=[50000,-50000]
    assert bundle.respond({'requests':[req]})['response']==-1
    assert bundle.agent.last_guard['rules']['lock_own']['fires']==1
    req.update(history=[],public_cards=[],my_id=0,dealer_id=1,my_chips=19950,hand=10,total_win_chips=[-10000,10000])
    s,resp=bundle.runtime.request_snapshot(req);ctx,_=bundle.runtime.context_from_payload({'requests':[req]})
    amounts={a:s.me_stack if r==-2 else min(s.to_call,s.me_stack) if a==1 else max(r,0) for a,r in resp.items()}
    raw=np.array([.8,.1,.05,.05,0]);monkeypatch.setattr(bundle.agent,'logits',lambda batch:np.log(np.maximum(raw,1e-30))[None,:])
    state=dict(eligible=True,rejected=False,confirmed=True,probe_used=False)
    p=bundle.agent.policy(s,ctx,amounts=amounts,request=req,lock_state=state)
    np.testing.assert_array_equal(p,[0,1,0,0,0])
    assert bundle.agent.last_guard['rules']['lock_opponent_confirmed']['fires']==1


@pytest.mark.parametrize('values,hard,soft',[
    ((64.9,119.9,299.9,.499),True,True),((65.,119.,250.,.4),True,False),
    ((90.,119.,250.,.4),True,False),((20.,120.,250.,.4),False,True),
    ((20.,119.,300.,.4),False,True),((20.,119.,250.,.5),False,True),
    ((float('nan'),119.,250.,.4),False,False)])
def test_server_latency_strict_hard_boundaries_and_soft_mean(values,hard,soft):
    result=gates(*values)
    assert result['hard_passed']==hard and result['soft_mean_passed']==soft


def test_real_state_sampling_is_balanced_and_never_invents_cards():
    states=[dict(id=str(i),payload={'requests':[{'public_cards':list(range(n))}]}) for i,n in enumerate((0,3,4,5))]
    corpus={'states':states};sample=sample_states(corpus,512,123)
    assert [sum(r['id']==str(i) for r in sample) for i in range(4)]==[128]*4
    assert all(row in states for row in sample)


def qualified_meta():
    return dict(epochs=50,fixture=False,smoke=False,production=True,offline_hud_prior=False,tree=TREE,
                river_samples=8192,turn_samples=4096,validation={k:dict(epochs_completed=50,lr_schedule='cosine',CFV_HOLDOUT_GATE=dict(passed=True,samples=2048)) for k in ('river','turn')})


@pytest.mark.parametrize('field,value',[('epochs',49),('epochs',51),('fixture',True),('smoke',True),('production',False),
                                        ('offline_hud_prior',True),('river_samples',8191),('turn_samples',4095)])
def test_final_arena_rejects_wrong_epochs_fixtures_and_insufficient_production(field,value):
    meta=qualified_meta();meta[field]=value
    with pytest.raises(ValueError,match='FINAL_ARENA'):assert_qualified_manifest(meta)


def test_qualification_requires_both_holdout_gates_and_actual_loaded_manifest(bundle):
    assert_qualified_manifest(qualified_meta())
    meta=qualified_meta();meta['validation']['turn']['CFV_HOLDOUT_GATE']['passed']=False
    with pytest.raises(ValueError,match='HOLDOUT'):assert_qualified_manifest(meta)
    # The existing actual one-epoch fixture must fail; no arena smoke escape.
    with pytest.raises(ValueError):assert_qualified_manifest(bundle.meta['cfv'])


def test_arm_identity_and_hud_eta_bound_and_risk_decay():
    arms=profiles();assert arms['A']['guard_mode']=='legacy_full' and arms['B']['guard_mode']=='slim'
    assert arms['C']['guard_mode']==arms['D']['guard_mode']=='surgical'
    assert not any(arms[n]['context_prior'] or arms[n]['rnr_eta'] for n in 'ABC')
    assert arms['D']['context_prior'] and arms['D']['context_eta_max']==.25 and not arms['D']['context_range_prior']
    ctx=SimpleNamespace(hands_played=49,hero_net_bb=10,opportunities={'fold_to_postflop_bet':10},successes={})
    values=[profile(ctx,3,SimpleNamespace(pot=p,to_call=0,big_blind=100),eta_max=.25)['eta'] for p in (200,1000,3999,4000)]
    assert .25>values[0]>values[1]>values[2]>values[3]==0
    for arm in arms:
        jobs=match_jobs(arm)
        assert len(jobs)==200 and all(j['hands']==50 for j in jobs)
        assert [{k:v for k,v in j.items() if k!='arm'} for j in jobs]==[{k:v for k,v in j.items() if k!='arm'} for j in match_jobs('A')]


def test_inference_core_and_siblings_excluded_from_workers(tmp_path):
    for cpu in range(6):
        folder=tmp_path/('cpu%d'%cpu);(folder/'topology').mkdir(parents=True)
        (folder/'topology/thread_siblings_list').write_text('1,5' if cpu in (1,5) else str(cpu))
        (folder/'cpufreq').mkdir();(folder/'cpufreq/scaling_governor').write_text('performance')
    r=core_plan(None,set(range(6)),tmp_path)
    assert r['inference_core']==1 and r['worker_cpus']==[0,2,3,4] and r['governor']=='performance'
    assert cpu_list('0-2,5,7-8')=={0,1,2,5,7,8}
    with pytest.raises(ValueError):core_plan(0,{0,1},tmp_path)


def test_resume_skips_completed_training_and_retries_only_incomplete_stage(tmp_path):
    identity={'commit':'locked','champion':'hash'};calls=[]
    def trained(folder):
        calls.append('train');(folder/'model.pt').write_bytes(b'trained50')
        return dict(path=str(folder/'model.pt'))
    state=Checkpoints(tmp_path,identity);result=state.run('training',trained)
    with pytest.raises(RuntimeError):state.run('arena_A',lambda p:(_ for _ in ()).throw(RuntimeError('crash')))
    resumed=Checkpoints(tmp_path,identity,True)
    assert resumed.run('training',trained)==result and calls==['train']
    assert resumed.run('arena_A',lambda p:dict(complete=True))['complete']
    Path(result['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact'):resumed.run('training',trained)
    with pytest.raises(ValueError,match='identity'):Checkpoints(tmp_path,{'commit':'different'},True)


def test_reservation_survives_crash_without_version_reallocation(tmp_path):
    root=tmp_path/'deliver';out=tmp_path/'run';out.mkdir()
    first=reserve(root,out,'locked-run',['A','D','C'])
    assert first['versions']=={'26':'A','27':'D','28':'C'}
    assert reserve(root,out,'locked-run',['A','D','C'])==first
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='consumed'):reserve(root,other,'other',['A','C','D'])


def test_paired_ranking_cis_and_no_promotion_legacy_first():
    from evening_production_eval import TOURNAMENT
    arms={}
    for name,score,win in (('A',5,.5),('B',4,.6),('C',6,.7),('D',7,.8)):
        arms[name]=dict(net_bb_pairs={o:[[score,score]]*20 for o in TOURNAMENT},match_win_rate=win,per_opponent={o:{} for o in TOURNAMENT})
    compared=compare_arms(arms)
    assert compared['D']['paired_ev_vs_A']['mean']==40 and compared['D']['promoted']
    assert rank_arms(compared,{})==['D','C','A']
    assert rank_arms(compared,{'D':'hard latency'})==['C','A','B']
    compared['C']['promoted']=False;compared['D']['promoted']=False
    assert rank_arms(compared,{})[0]=='A'


def test_budget_drops_b_first_and_prices_serial_inference(monkeypatch):
    import overnight_server as s
    seen=[]
    def planner(rates,hours,evaluation,elapsed,workers):
        seen.append(evaluation)
        if len(seen)==1:raise RuntimeError('does not fit')
        return dict(net_epochs=50)
    monkeypatch.setattr(s,'plan_counts',planner)
    result=s.choose_plan({},8.5,10.,20.,100.,24)
    assert result['arms']==['A','C','D'] and result['dropped_arms']==['B']
    assert result['estimated_serial_arena_seconds']==6000 and result['inference_parallelism']==1
    assert seen[0]>8000 and seen[1]>6000


def test_hard_latency_failure_excludes_one_arm_but_does_not_abort_the_others(tmp_path,monkeypatch):
    import overnight_server as s
    ran=[];updates=[]
    def measured(release,*args):
        return dict(cold_start_s=.3,gate=dict(hard_passed=release!='D'))
    def arena(arm,*args):
        ran.append(arm)
        return dict(runtime=dict(mean_ms=75.,p99_ms=110.,max_ms=200.))
    monkeypatch.setattr(s,'measure',measured);monkeypatch.setattr(s,'run_arm',arena)
    def stage(name,fn,clean=True):
        folder=tmp_path/name;folder.mkdir();return fn(folder)
    arms,exclusions,latencies=s.arena_phase({a:a for a in 'ABCD'},'nets','corpus',{},24,stage,lambda *r:updates.append(r))
    assert ran==['A','C','D','B'] and set(arms)==set('ABCD')
    assert set(exclusions)=={'D'} and len(updates)==4
    assert all(not arm['arena_latency_gate']['soft_mean_passed'] for arm in arms.values())
    assert all(arm['arena_latency_gate']['hard_passed'] for arm in arms.values())


def test_fixture_cfv_is_rejected_before_any_arena_or_packaging_work(bundle,tmp_path):
    from server_arena import run_arm
    from server_packaging import package_one
    nets=bundle.model.parent/'nets.npz'
    with pytest.raises(ValueError):run_arm('A',{},nets,tmp_path/'arena',{},1)
    with pytest.raises(ValueError):package_one('missing',nets,tmp_path/'deliver',tmp_path,{},26,'A',{},False,None)
    assert not (tmp_path/'arena').exists() and not (tmp_path/'deliver').exists()


def test_rpc_returns_exact_response_distributions_and_server_timing(monkeypatch):
    """Transport unit test only: a stub policy, never a final arena/model result."""
    import queue
    import threading
    import torch
    import server_arena as a
    class StubBundle:
        def __init__(self,*args):
            self.meta={'cfv':qualified_meta()};self.runtime=SimpleNamespace()
            self.agent=SimpleNamespace(last_evening=dict(search=True,support=1081,cfv_calls=0),
                last_guard={'rules':{'test_rule':dict(fires=1,mass_moved=.1)}},search_audit={'status':'active'},
                guard_before=np.array([.2,.5,.1,.1,.1]),guard_after=np.array([.3,.5,.1,.1,0]))
        def respond(self,payload):return dict(response=payload['response'],data={'marker':7})
        def debug(self):return 'py unit stub'
    monkeypatch.setattr(a,'Bundle',StubBundle);monkeypatch.setattr(a,'pin',lambda cpus:cpus)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda _:None)
    requests=queue.Queue();replies=[queue.Queue()];ready=queue.Queue()
    thread=threading.Thread(target=a._serve,args=({'source':'stub','model':'stub'},1,requests,replies,ready))
    thread.start()
    try:
        assert ready.get(timeout=5)['affinity']==[1]
        client=a.RPCBundle({'source':'stub','model':'stub'},requests,replies[0],0)
        client.reset(11,0)
        assert client.respond({'response':123})==dict(response=123,data={'marker':7})
        np.testing.assert_array_equal(client.agent.guard_before,[.2,.5,.1,.1,.1])
        np.testing.assert_array_equal(client.agent.guard_after,[.3,.5,.1,.1,0])
        assert client.agent.last_guard['rules']['test_rule']['fires']==1
        assert len(client.engine_s)==len(client.roundtrip_s)==1 and client.roundtrip_s[0]>=client.engine_s[0]
    finally:
        requests.put(None);thread.join(timeout=5)
    assert not thread.is_alive()


def _stub_package_io(monkeypatch):
    """Publication transaction fixtures; no inference or CFV qualification claim."""
    import server_packaging as p
    import zipfile
    def weights(checkpoint,destination,**kw):
        meta=dict(deployment_release=kw['release_id'],settings={'evening_online':kw['online_profile']})
        np.savez(destination,meta_json=np.frombuffer(json.dumps(meta).encode(),np.uint8))
    def source(destination,**kw):
        with zipfile.ZipFile(destination,'w') as z:z.writestr('__main__.py','print("unit")\n')
    monkeypatch.setattr(p,'assert_production_cfv',lambda _:None)
    monkeypatch.setattr(p,'export_botzone_weights',weights);monkeypatch.setattr(p,'pack_botzone_zip',source)
    return p


def test_publication_recovers_after_receipt_before_rename_without_retesting(tmp_path,monkeypatch):
    p=_stub_package_io(monkeypatch);root=tmp_path/'deliver';out=tmp_path/'run';out.mkdir()
    reservation=reserve(root,out,'locked',['A','C','D']);real_rename=p.os.rename;calls=[]
    def gate(*args):calls.append(1);return dict(gate={'hard_passed':True})
    def crash(*args):raise RuntimeError('injected crash before rename')
    monkeypatch.setattr(p.os,'rename',crash)
    with pytest.raises(RuntimeError,match='injected'):
        p.package_one('cp','nets',root,out,reservation,26,'A',profiles()['A'],False,gate)
    monkeypatch.setattr(p.os,'rename',real_rename)
    result=p.package_one('cp','nets',root,out,reservation,26,'A',profiles()['A'],False,gate)
    assert calls==[1] and result['status']=='not promoted'
    assert {f.name for f in Path(result['folder']).iterdir()}=={'V26_SOURCE.zip','V26_MODEL.npz','UPLOAD_README.zh.md'}
    assert p.package_one('cp','nets',root,out,reservation,26,'A',profiles()['A'],False,gate)==result


def test_hard_package_exclusion_persists_across_resume(tmp_path,monkeypatch):
    p=_stub_package_io(monkeypatch);root=tmp_path/'deliver';out=tmp_path/'run';out.mkdir()
    reservation=reserve(root,out,'locked',['A','C','D']);calls=[]
    def gate(*args):calls.append(1);return dict(gate={'hard_passed':False})
    first=p.package_one('cp','nets',root,out,reservation,26,'A',profiles()['A'],False,gate)
    again=p.package_one('cp','nets',root,out,reservation,26,'A',profiles()['A'],False,gate)
    assert first==again and calls==[1] and first['excluded']
    assert not list(root.glob('UPLOAD_V26*'))


def test_d_ranges_condition_once_and_subnode_hud_eta_is_bounded(bundle):
    from alphanl.stage2 import deep_solver_v2 as solver
    from alphanl.stage2.search_belief import history_combo_ranges
    from alphanl.stage2.context import MatchCtx
    _,s,_=case(bundle,9,295)
    ctx=MatchCtx(hands_played=20,hero_net_bb=1.)
    ctx.opportunities['fold_to_postflop_bet']=10;ctx.successes['fold_to_postflop_bet']=7
    rh,ro=history_combo_ranges(s,eta=.2)
    sol=solver.PublicSolver(s,rh,ro,iterations=2,context=ctx,context_eta_max=.25,context_range_prior=False)
    np.testing.assert_array_equal(sol.ro,ro)
    assert sol.trace['match_prior']['eta_max']==.25
    def check(node):
        if hasattr(node,'match_eta'):assert 0<=node.match_eta<=.25
        for child in node.children or ():check(child)
    check(sol.tree)


def test_benchmark_repeated_states_never_mutate_the_public_hud_corpus(monkeypatch):
    """Reconciliation may consume nested dictionaries: every call owns a copy."""
    import server_benchmark as b
    calls=[]
    class Stub:
        def __init__(self,*args):
            self.meta={'settings':{'evening_online':{}},'config':{},'deployment_release':'unit'}
            self.agent=SimpleNamespace(last_evening={'search':True,'support':1081},search_audit={'status':'active'})
        def respond(self,payload):
            calls.append(payload['data']['counter']);payload['data']['counter']+=1
            return {'response':0}
    states=[dict(id=str(i),payload={'requests':[{'public_cards':list(range(n))}],'data':{'counter':0}}) for i,n in enumerate((0,3,4,5))]
    corpus={'states':states,'by_board_size':{str(n):1 for n in (0,3,4,5)}};original=copy.deepcopy(corpus)
    monkeypatch.setattr(b,'Bundle',Stub);monkeypatch.setattr(b,'pin',lambda cpus:cpus)
    monkeypatch.setattr(b.cfv,'digest',lambda path:'unit')
    monkeypatch.setattr(b.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=json.dumps({'debug':'py unit ok model'})))
    b.benchmark('source','model',corpus,1,warmup=2,count=8,cold=1,diagnostic=True)
    assert corpus==original and calls==[0]*10

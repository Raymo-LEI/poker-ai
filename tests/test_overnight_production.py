"""Phase-two contracts: independent ledgers, dense DCFR, PBS and release gates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/sprint24'))
from alphanl.stage2 import cfv,deep_solver_v2 as v
from overnight_production import plan_counts,select_handoff,score_cells,group_stop,HARD_SECONDS
from overnight_data import posterior_ranges,bayes,mixed_dataset,trajectory,holdout_gate
from overnight_distill import expand,bind_champion,paired_improvement,gate,surgery_fn,fit,measures
from evening_production import find_champion
from evening_pipeline import sample_spot


@pytest.mark.parametrize('me,opp',[(9000,9000),(700,9000),(9000,700),(1500,900),(900,1500)])
def test_checkraise_all_leaves_match_independent_chip_ledger(me,opp):
    from alphanl.stage2.search_game import PublicState
    pot=1000
    tree=v.betting_tree(pot,0,me,opp,legal=(1,),check_raise=True)
    after_check=tree.children[0]
    for bet,node in zip(after_check.labels[1:],after_check.children[1:]):
        state=PublicState(pot,(me,opp)).step(1,0).step(3,int(bet))
        assert node.labels[:2]==['fold','call']
        for i,name in enumerate(node.labels[2:],2):
            raise_to=float(name.rsplit('_',1)[1]);response=node.children[i]
            assert raise_to>bet and (raise_to>=2*bet or raise_to==me)
            raised=state.step(4 if raise_to==me else 3,int(raise_to))
            for act,leaf in zip((0,1),response.children):
                final=raised.step(act,0 if act==0 else int(min(opp-bet,raise_to-bet)))
                kind,h,o,p=leaf.leaf
                for eq in (0.,.5,1.):
                    ledger=final.settlement(eq)
                    if kind=='show':value=eq*(p+h+o)-h-pot/2
                    else:value=pot+o-pot/2
                    assert ledger.sum()==pot+me+opp
                    assert ledger[0]-me-pot/2==pytest.approx(value)


def test_no_checkraise_after_opponent_jam_or_ip_closing_check():
    tree=v.betting_tree(1000,0,9000,500,legal=(1,),check_raise=True)
    assert all(n.labels==['fold','call'] for n in tree.children[0].children[1:])
    tree=v.betting_tree(1000,0,9000,9000,legal=(1,),check_closes=True,check_raise=True)
    assert tree.children[0].leaf==('show',0.,0.,1000.)
    legacy=v.betting_tree(1000,0,9000,9000,legal=(1,))
    assert all(n.labels==['fold','call'] for n in legacy.children[0].children[1:])


def test_checkraise_dcfr_matches_independent_dense_small_game():
    from alphanl.stage2.search_math import Node,Terminal,DCFR
    snap,rh,ro=sample_spot(902,3);snap.to_call=0;snap.me_seat=0
    snap.legal_action_ids=(1,2,3,4);snap.pot=1000;snap.me_stack=2000;snap.opp_stack=1600
    sol=v.PublicSolver(snap,iterations=50,check_raise=True)
    hi=np.array([0,100,500,900]);oi=np.array([35,250,550,1000])
    sol.rh[:]=0;sol.ro[:]=0
    sol.rh[hi]=[.1,.2,.3,.4];sol.ro[oi]=[.4,.3,.2,.1]
    sol.rootmass=(sol.mass(sol.ro),sol.mass(sol.rh))
    valid=np.array([[not set(sol.hands[i])&set(sol.hands[j]) for j in oi] for i in hi])
    eq=(sol.rank[hi,None]>sol.rank[None,oi])+.5*(sol.rank[hi,None]==sol.rank[None,oi])
    def convert(node,key=()):
        if node.player is None:
            kind,h,o,p=node.leaf
            if kind=='show':value=(eq*(p+h+o)-h-snap.pot/2)/100.
            else:value=np.full(eq.shape,(-h if kind=='fold0' else snap.pot+o)/100.-snap.pot/200.)
            return Terminal(value)
        return Node(node.player,{i:convert(n,key+(i,)) for i,n in enumerate(node.children)},key=key)
    root=convert(sol.tree)
    dense=DCFR(root,sol.rh[hi,None]*sol.ro[None,oi]*valid).solve(50)
    sol.solve()
    np.testing.assert_allclose(sol.average[hi],dense['strategy'][()],atol=1e-12)
    assert abs(sol.trace['zero_sum_residual'])<1e-12
    assert any('check_raise_to_' in str(n.labels) for n in sol.tree.children[0].children[1:])
    assert np.any(sol.tree.regret<0)


def test_public_bayes_depends_on_observed_policy_and_masks_board_only():
    hands=(('2s','3s'),('4h','5h'),('As','Ac'))
    weights=np.array([.2,.3,.5]);updated=bayes(weights,np.array([.1,.5,.9]))
    np.testing.assert_allclose(updated,weights*np.array([.1,.5,.9])/.62)
    a,b=posterior_ranges(hands,(updated,weights),('2s','6d','7c','8s','9h'))
    assert a.sum()==pytest.approx(1.) and b.sum()==pytest.approx(1.)
    assert not np.allclose(a,b)
    with pytest.raises(ValueError):bayes(weights,np.zeros(3))
    with pytest.raises(ValueError):bayes(weights,np.array([1.,2.,3.]))


def test_full_budget_includes_three_generations_50_epochs_holdout_and_watchdog():
    rates=dict(river=.002,turn=.004,rebel=.003,anchor=.001,cfv_epoch=.000002,policy_epoch=.000004)
    p=plan_counts(rates,8.5,600,100,24)
    assert p['net_epochs']==50 and p['holdout_n']==2048
    assert p['generations']==['g3','g4','g5']
    assert 8192<=p['river_n']<=500000 and 4096<=p['turn_n']<=350000
    assert p['predicted_seconds']<=8.5*3600<HARD_SECONDS
    with pytest.raises(RuntimeError):plan_counts(dict(rates,river=100,turn=100,rebel=100),8.5,600,100,24)
    with pytest.raises(ValueError):plan_counts(rates,9,600,100,24)
    with pytest.raises(ValueError):plan_counts(dict(rates,turn=float('nan')),8.5,600,100,24)


def test_paired_surgery_gate_rejects_ties_noise_and_regressions():
    x=np.ones(2048)
    assert paired_improvement(x,x*.99)['accepted']
    assert not paired_improvement(x,x)['accepted']
    assert not paired_improvement(x,x*1.01)['accepted']
    assert not paired_improvement([1.],[0.])['accepted']
    noisy=np.r_[np.full(1024,.5),np.full(1024,1.4999)]
    assert not paired_improvement(x,noisy)['accepted']
    gain=np.r_[np.full(1024,.1),np.full(1024,-.08)]
    assert paired_improvement(x,x-gain)['accepted']
    assert not paired_improvement(x,x-gain,groups=np.repeat([0,1],1024))['accepted']
    assert not paired_improvement(x,x*.99,groups=np.zeros(2048))['accepted']


def test_strict_cfv_holdout_gate_rejects_constant_and_large_pot_error(monkeypatch):
    import overnight_data as d
    x=np.zeros((2048,136));x[:,119]=.05
    y=np.tile(np.arange(32),(2048,1)).astype(float)
    train=np.zeros((10,32))
    monkeypatch.setattr(cfv,'predict_residual_cfv',lambda *a:y)
    assert holdout_gate({},'',train,(x,y))['passed']
    monkeypatch.setattr(cfv,'predict_residual_cfv',lambda *a:np.zeros_like(y))
    assert not holdout_gate({},'',train,(x,y))['passed']


@pytest.fixture(scope='module')
def champion_model():
    from alphanl.stage2.train import load_checkpoint
    return load_checkpoint(find_champion(),device='cpu')[0]


def test_expansion_preserves_champion_route_and_checkpoint_roundtrip(champion_model,tmp_path):
    import torch
    from alphanl.stage2.train import load_checkpoint,save_checkpoint
    original=copy.deepcopy(champion_model);model=expand(bind_champion(copy.deepcopy(original),original))
    items=trajectory(original,4,102003,preflop=True)
    before=measures(original,items)[2]
    np.testing.assert_array_equal(before,measures(model,items)[2])
    with torch.no_grad():
        for name,p in model.named_parameters():
            if not name.startswith('reference.'):p.add_(.01)
    np.testing.assert_array_equal(before,measures(model,items)[2])
    path=tmp_path/'expanded.pt';save_checkpoint(path,model)
    loaded,_=load_checkpoint(path)
    assert loaded.settings['hidden']==256
    np.testing.assert_array_equal(before,measures(loaded,items)[2])
    assert loaded._evening_preflop_anchor['delta_head.weight'].shape==(5,64)


def test_exported_expanded_route_and_full_slim_guards(champion_model,tmp_path):
    import torch
    from alphanl.stage2.train import save_checkpoint
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    from alphanl.stage2.deployment_guard import V10_GUARD
    from evening_d import Bundle
    from evening_latency import cases
    from overnight_production import ONLINE
    from overnight_data import TREE
    from test_evening_v2 import _random_nets
    nets=tmp_path/'nets.npz'
    cfv.publish(nets,_random_nets(),2,1,'nets',dict(river_samples=1,turn_samples=1,tree=TREE))
    source_model=copy.deepcopy(champion_model);expanded=expand(bind_champion(copy.deepcopy(source_model),source_model))
    with torch.no_grad():expanded.delta_head.bias.add_(.1)
    paths={};bundles={}
    for name,model,mode in (('original',source_model,'slim'),('full',expanded,'slim'),('slim',expanded,'slim')):
        folder=tmp_path/name;folder.mkdir();cp=folder/'checkpoint.pt';save_checkpoint(cp,model)
        release='V999_TEST_'+name.upper();zip_path=folder/'V999_SOURCE.zip';npz=folder/'V999_MODEL.npz'
        export_botzone_weights(cp,npz,release_id=release,cfv_nets=nets,online_profile=dict(ONLINE,guard_mode=mode),
            deployment_guard=dict(V10_GUARD,enabled=True,victory_blind_lock_enabled=True))
        pack_botzone_zip(zip_path,release_id=release);bundles[name]=Bundle(zip_path,npz)
    for payload in cases(8,1931000000):
        probabilities={}
        for name,b in bundles.items():
            snap,resp=b.runtime.request_snapshot(payload['requests'][0]);ctx,_=b.runtime.context_from_payload(payload)
            amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in resp.items()}
            probabilities[name]=b.agent.policy(snap,ctx,amounts=amounts,request=payload['requests'][0])
            assert probabilities[name].sum()==pytest.approx(1.)
            assert all(p==0 or a in amounts for a,p in enumerate(probabilities[name]))
        if snap.street==0:
            np.testing.assert_array_equal(probabilities['original'],probabilities['full'])
            np.testing.assert_array_equal(probabilities['original'],probabilities['slim'])
        else:
            assert bundles['slim'].agent.last_guard['postflop_guard_mode']=='slim'
            assert not bundles['slim'].agent.search_audit['guard_changed']


def test_real_onpolicy_mixture_parallel_serial_and_value_labels(champion_model,tmp_path):
    from alphanl.stage2.train import save_checkpoint
    from test_evening_v2 import _random_nets
    path=tmp_path/'model.pt';save_checkpoint(path,champion_model)
    x,y,m=mixed_dataset(tmp_path/'a.npz',path,3,4,43001,1,2,1)
    xx,yy,mm=mixed_dataset(tmp_path/'b.npz',path,3,4,43001,2,2,1)
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy)
    assert m['on_policy_n']==m['dirichlet_n']==2
    assert len(m['identities'])==4 and m['identities'][:2]!=m['identities'][2:]
    items=trajectory(champion_model,4,123011,teacher=True,nets=_random_nets(),iterations=2,chance_samples=1)
    assert all(np.isfinite(r['value']) and r['target'].sum()==pytest.approx(1.) for r in items)
    assert any(not np.allclose(r['rh'],np.full(32,1/32.)) for r in items)


def test_joint_value_training_and_actual_rank_one_surgery(champion_model):
    from test_evening_v2 import _random_nets
    import torch
    model=expand(bind_champion(copy.deepcopy(champion_model),champion_model))
    items=trajectory(champion_model,8,83110,teacher=True,nets=_random_nets(),iterations=2,chance_samples=1)
    before=model.value_head.weight.detach().clone()
    result=fit(model,items,2)
    assert result['kl']==.01 and not torch.equal(before,model.value_head.weight)
    edited,s=surgery_fn(model,items,items)  # invocation only; production splits are disjoint
    assert s['called'] and len(s['candidates'])==18 and s['rank']==1
    assert edited._evening_preflop_anchor.keys()==model._evening_preflop_anchor.keys()


def test_handoff_fresh_missing_and_incomplete_do_not_pick_unfinished(tmp_path):
    original=tmp_path/'champion';original.write_bytes(b'original')
    incomplete=tmp_path/'new';incomplete.mkdir()
    (incomplete/'FINAL_REPORT.json').write_text(json.dumps(dict(status='RUNNING')))
    assert select_handoff(tmp_path,original)[0]==original
    assert select_handoff(tmp_path,original,True)[1]['mode']=='fresh'


def test_handoff_selects_scored_checkpoint_and_rejects_tampering(tmp_path,champion_model):
    from alphanl.stage2.train import save_checkpoint
    from alphanl.stage2.export import export_botzone_weights
    from evening_production_eval import BENCHMARK,TOURNAMENT
    champion=tmp_path/'champion.pt';save_checkpoint(champion,champion_model)
    results={};releases={}
    for role,score in (('baseline',1),('g1',3),('g2',2)):
        # Handoff only needs a valid evaluated weights archive and source hash.
        model=tmp_path/(role+'.npz');export_botzone_weights(champion,model)
        with np.load(model) as z:payload={k:np.array(z[k]) for k in z.files}
        meta=json.loads(payload['meta_json'].tobytes());meta['deployment_release']='V22_TEST'
        payload['meta_json']=np.frombuffer(json.dumps(meta).encode(),np.uint8)
        np.savez(model,**payload)
        source=tmp_path/(role+'.zip');source.write_bytes(b'source')
        releases[role]=dict(model=str(model),source=str(source),model_sha256=cfv.digest(model),source_sha256=cfv.digest(source),release='V22_TEST')
        results[role]={phase:{name:dict(mbb_per_hand=score,hands=5000 if phase=='benchmark' else 1000) for name in names}
                      for phase,names in (('benchmark',BENCHMARK),('tournament',TOURNAMENT))}
    report=dict(status='COMPLETE',identity=dict(smoke=False,champion_sha256=cfv.digest(champion)),evaluation=dict(results=results),releases=releases)
    (tmp_path/'FINAL_REPORT.json').write_text(json.dumps(report))
    selected,info=select_handoff(tmp_path,champion)
    assert selected==champion and info['role']=='g1'
    Path(releases['g1']['model']).write_bytes(b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):select_handoff(tmp_path,champion)
    report['evaluation']['results']['g2']['benchmark']['equity']['hands']=2
    (tmp_path/'FINAL_REPORT.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='incomplete'):select_handoff(tmp_path,champion)


@pytest.mark.skipif(sys.platform=='win32',reason='POSIX process groups')
def test_watchdog_kills_descendant_after_leader_exits(tmp_path):
    import os
    pidfile=tmp_path/'pid'
    script="import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(100)']); open(sys.argv[1],'w').write(str(p.pid))"
    leader=subprocess.Popen([sys.executable,'-c',script,str(pidfile)],start_new_session=True)
    leader.wait(timeout=5)
    descendant=int(pidfile.read_text());group_stop(leader,grace=.05)
    # An orphan can briefly remain a zombie on Linux; it must not be running.
    state=subprocess.check_output(['ps','-o','stat=','-p',str(descendant)],text=True).strip() if subprocess.run(['kill','-0',str(descendant)],capture_output=True).returncode==0 else ''
    assert not state or state.startswith('Z')

"""Four architecture contracts: safety, full river, terminal utility, RNR."""
import copy
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from alphanl.stage2 import deep_solver_v2 as v
from alphanl.stage2.deployment_guard import apply_search_guard
from alphanl.stage2.search_belief import history_combo_ranges
from test_overnight_bot17_repair import bundle,case


def test_slim_only_legal_normalization_and_free_check():
    s=SimpleNamespace(street=3,to_call=0.,legal_action_ids=(0,1,2,3,4))
    p,info=apply_search_guard(s,np.array([.6,.1,.2,.1,-.01]),{0:0,1:0,2:100,3:200,4:2000})
    np.testing.assert_allclose(p,[0,.7,.2,.1,0])
    assert info['call_fuse'] is None
    s.to_call=100.;raw=np.array([.85,.1,.03,.02,0])
    np.testing.assert_array_equal(apply_search_guard(s,raw)[0],raw)


@pytest.mark.parametrize('eta',[0.,.25,.5])
def test_rnr_exact_formula_and_uniform_floor(bundle,eta):
    _,s,_=case(bundle,17,16112,'match3')
    # Recover the size component from the eta=.5 mixture, not from Hero cards.
    mixed,audit=history_combo_ranges(s,True,eta=.5)
    size=2*mixed[1]-np.ones(1081)/1081
    actual,info=history_combo_ranges(s,True,eta=eta)
    np.testing.assert_allclose(actual[1],(1-eta)/1081+eta*size,atol=1e-15)
    assert actual[1].min()>=(1-eta)/1081-1e-16
    assert info['eta']==eta
    with pytest.raises(ValueError):history_combo_ranges(s,eta=.51)


def test_match_cdf_risk_curvature_blind_boundaries_and_final_hand():
    p=dict(lead=800.,sigma=1200*math.sqrt(33),remaining=33,own_budget=2500.,opponent_budget=2450.)
    safe=v.match_probability(0,p);gamble=v.match_probability(np.array([-20000.,20000.]),p).mean()
    assert safe>gamble
    q=dict(p,lead=-800.)
    assert v.match_probability(0,q)<v.match_probability(np.array([-20000.,20000.]),q).mean()
    assert v.match_probability(1701,p)==1.
    assert v.match_probability(-3251,p)==0.
    np.testing.assert_array_equal(v.match_probability(np.array([-801.,-800.,-799.]),dict(p,remaining=0)),[0,.5,1])


def test_current_hand_sunk_chips_are_in_match_payoff(bundle):
    _,s,_=case(bundle,17,16112,'match3')
    ctx=SimpleNamespace(hero_net_bb=9.11,hands_played=16)
    sol=v.PublicSolver(s,utility=True,context=ctx,iterations=4,policy_only=True)
    assert sol.match['sunk']==3888
    assert sol.match['remaining']==33 and sol.match['opponent_budget']==2450
    fold=sol.tree.children[sol.tree.labels.index(0)]
    hv,ov=sol.terminal(fold.leaf,sol.rh,sol.ro)
    # Folding this late locks in -2977, so forcing Fold is not risk aversion.
    np.testing.assert_allclose(hv/sol.rootmass[0],-.5)
    assert abs(np.dot(sol.rh,hv)+np.dot(sol.ro,ov))<1e-12


def test_wtl_utility_is_dense_exact_and_not_utility_of_mean(bundle):
    _,s,_=case(bundle,9,295)
    ctx=SimpleNamespace(hero_net_bb=2.,hands_played=45)
    sol=v.PublicSolver(s,utility=True,context=ctx,iterations=2)
    leaf=sol.tree.children[sol.tree.labels.index(1)].leaf
    hv,ov=sol.terminal(leaf,sol.rh,sol.ro)
    kind,h,o,p=leaf;at=next(i for i,a in enumerate(sol.hands) if set(a)==set(v._ev(c) for c in s.hole_cards))
    compatible=np.array([not set(sol.hands[at])&set(pair) for pair in sol.hands])
    win=float(v.match_probability(p+o-sol.match['sunk'],sol.match))-.5
    lose=float(v.match_probability(-h-sol.match['sunk'],sol.match))-.5
    tie=float(v.match_probability(.5*(p+o-h)-sol.match['sunk'],sol.match))-.5
    payoff=np.where(sol.rank[at]>sol.rank,win,np.where(sol.rank[at]==sol.rank,tie,lose))
    assert hv[at]==pytest.approx(np.dot(sol.ro*compatible,payoff),abs=1e-13)
    assert abs(np.dot(sol.rh,hv)+np.dot(sol.ro,ov))<1e-12


@pytest.mark.parametrize('street',[1,2])
def test_allin_runout_and_nonterminal_cfv_share_consistent_utility_units(bundle,street):
    from test_evening_v2 import _random_nets
    from evening_pipeline import sample_spot
    s,rh,ro=sample_spot(991+street,street);s.to_call=0;s.legal_action_ids=(1,4)
    ctx=SimpleNamespace(hero_net_bb=4.,hands_played=20)
    sol=v.PublicSolver(s,rh,ro,weights=_random_nets(),utility=True,context=ctx,iterations=2,chance_samples=1,policy_only=True).solve()
    assert sol.trace['terminal_match_utility']
    assert sol.trace['cfv_continuation']=='first_order_root_tangent_not_U_of_mean'
    assert np.isfinite(sol.state).all() and abs(sol.trace['zero_sum_residual'])<1e-10


@pytest.mark.parametrize('call',[0,295])
def test_respond_always_runs_full_river_without_cfv_requirement(bundle,call,monkeypatch):
    from alphanl.stage2.evening_strategy import search_policy
    row,s,amounts=case(bundle,9,call)
    p,t=search_policy(s,None,np.array([.1,.7,.1,.1,0]),None,amounts,4,1,check_raise=True)
    assert t['search'] and t['support']==1081 and t['reason']=='exact_river_terminals'
    # No score lock: actual exported respond invokes the full solver.
    request=copy.deepcopy(row['request']);request['total_win_chips']=[0,0]
    body=bundle.respond({'requests':[request]})
    assert isinstance(body['response'],int)
    assert bundle.agent.search_audit['river_full_combo']
    assert bundle.agent.last_evening['support']==1081
    assert bundle.agent.search_audit['terminal_match_utility']


def test_legacy_guard_cannot_be_published_or_selected(bundle,tmp_path):
    from pack_wizard import pack_evening
    with pytest.raises(ValueError,match='require slim'):
        pack_evening('missing.pt','missing.npz',tmp_path,release_label='OVERNIGHT_TEST',online_profile={'guard_mode':'legacy_full'})


def test_slim_preflop_bypasses_heuristic_allin_mask_and_r5_rules(bundle,monkeypatch):
    row,s,amounts=case(bundle,36,241)
    ctx,_=bundle.runtime.context_from_payload({'requests':[row['request']]})
    raw=np.array([.2,.3,.05,.05,.4])
    monkeypatch.setattr(bundle.agent,'logits',lambda batch:np.log(raw)[None,:])
    obs=bundle.agent.observation(s,ctx)
    assert obs['legal_mask'][0,4]==1 and 4 in amounts
    np.testing.assert_allclose(bundle.agent.policy(s,ctx,amounts=amounts,request=row['request']),raw,atol=1e-12)


def test_own_lock_can_free_fold_but_unconfirmed_opponent_cannot_force_call(bundle,monkeypatch):
    row,s,amounts=case(bundle,9,0)
    request=copy.deepcopy(row['request']);request['total_win_chips']=[50000,-50000]
    assert bundle.respond({'requests':[request]})['response']==-1
    assert bundle.agent.search_audit['reason']=='blind_lock_own'
    request.update(history=[],public_cards=[],my_id=0,dealer_id=1,my_chips=19950,hand=10,total_win_chips=[-10000,10000])
    snap,resp=bundle.runtime.request_snapshot(request)
    ctx,_=bundle.runtime.context_from_payload({'requests':[request]})
    amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in resp.items()}
    raw=np.array([.7,.2,.05,.05,0])
    monkeypatch.setattr(bundle.agent,'logits',lambda batch:np.log(np.maximum(raw,1e-30))[None,:])
    state=dict(eligible=True,rejected=False,confirmed=False,probe_used=False)
    p=bundle.agent.policy(snap,ctx,amounts=amounts,request=request,lock_state=state)
    np.testing.assert_allclose(p,raw,atol=1e-12)
    state['confirmed']=True
    p=bundle.agent.policy(snap,ctx,amounts=amounts,request=request,lock_state=state)
    np.testing.assert_array_equal(p,[0,1,0,0,0])
    assert bundle.agent.search_audit['reason']=='blind_lock_opponent_confirmed'


def test_200_match_paired_comparison_counts_and_uncertainty():
    from evening_production_eval import compare_guards,TOURNAMENT
    results={}
    for mode,net in (('slim',10.),('legacy_full',-1.)):
        results['g_'+mode]={'tournament':{name:{'net_bb_pairs':[[net,net]]*20} for name in TOURNAMENT}}
    r=compare_guards({'results':results,'hands_per_match':50},['g'])['g']
    assert r['matches_per_mode']==200 and r['duplicate_pairs']==100
    assert r['paired_ev_gain_mbb_per_hand']['mean']==pytest.approx(220.)
    assert r['paired_win_rate_gain']['mean']==1 and r['improvement_demonstrated']


def test_rollout_uses_authoritative_legal_domain_without_native_shove_mask():
    from overnight_data import trajectory
    from evening_production import find_champion
    from alphanl.stage2.train import load_checkpoint
    model,_=load_checkpoint(find_champion())
    items=trajectory(model,8,117333,preflop=True)
    for item in items:
        expected=np.zeros(5);expected[list(item['snapshot'].legal_action_ids)]=1
        np.testing.assert_array_equal(item['row']['legal_mask'][0],expected)


def test_late_start_absolute_cutoff_never_extends_nine_hours():
    import datetime
    from overnight_production import deadline_budget
    utc=datetime.timezone.utc
    now=datetime.datetime(2026,11,4,16,45,tzinfo=utc)  # 01:45 JST
    target,hard=deadline_budget(8.5,'2026-11-05T10:00:00+09:00',now)
    assert hard==8.25*3600 and target==7.75
    assert deadline_budget(8.5,None,now)==(8.5,9*3600)
    assert deadline_budget(8.5,'2026-11-06T10:00:00+09:00',now)==(8.5,9*3600)
    with pytest.raises(ValueError):deadline_budget(8.5,'2026-11-05T10:00:00',now)
    with pytest.raises(ValueError):deadline_budget(8.5,'2026-11-05T01:00:00+09:00',now)

"""Latest explicit 8/8/16 + memory-aware platform budget; legacy tests remain intact."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from server_benchmark import full_power_gates
from server_arena import profiles,SERVER_ITERATIONS
from alphanl.stage2 import solver_prior as p
from alphanl.stage2 import deep_solver_v2 as v
from alphanl.stage2.context import MatchCtx
from evening_pipeline import sample_spot


@pytest.mark.parametrize('values,hard,soft',[
    ((159.9,599.9,1499.9,2.999,219.9),True,True),
    ((160.,599.,1499.,2.9,219.),True,False),
    ((200.,184.7,799.,.323,100.),True,False),
    ((59.2,600.,799.,.323,100.),False,True),
    ((59.2,599.,1500.,.323,100.),False,True),
    ((59.2,599.,799.,3.,100.),False,True),
    ((59.2,599.,799.,1.9,220.),False,True),
    ((59.2,599.,799.,1.9,float('inf')),False,True)])
def test_latest_hard_memory_and_latency_boundaries(values,hard,soft):
    r=full_power_gates(*values)
    assert r['hard_passed']==hard and r['soft_mean_passed']==soft


def test_all_production_arms_lock_full_power_budget_and_reject_lowering():
    for arm,info in profiles().items():assert info['iterations']=={'1':8,'2':8,'3':16}
    for bad in ({'1':3,'2':4,'3':8},{'1':4,'2':3,'3':8},{'1':4,'2':4,'3':6},{'1':2,'2':2,'3':4}):
        with pytest.raises(ValueError,match='locked'):profiles(bad)


def context():
    ctx=MatchCtx(hands_played=20,hero_net_bb=3.)
    ctx.opportunities['fold_to_postflop_bet']=12;ctx.successes['fold_to_postflop_bet']=8
    ctx.opportunities['river_aggression']=20;ctx.successes['river_aggression']=7
    ctx.response_by_size={'3:small':[12,9,1],'3:regular':[15,7,4],'3:overbet':[9,1,7]}
    ctx.sb_open_count=4;ctx.sb_open_mean_bb=2.7;ctx.sb_open_m2_bb=.4
    ctx.postflop_size_count=20;ctx.postflop_small_bets=12;ctx.postflop_overbets=3
    return ctx


def test_cached_all_sigma_schemas_match_reference_exactly():
    s,_,_=sample_spot(8123,3);ctx=context();strength=np.linspace(0,1,1081)
    cache=p.prepare_context(ctx,3,s,.25);sigma=p.SigmaPriors(cache,strength)
    for fraction in (None,.25,.4,.5,1.,1.5):
        for labels in (['fold','call'],['fold','call','raise_all_in'],['check',50.],['check',50.,100.]):
            expected=p.response_prior(labels,strength,cache.profile,ctx,fraction)
            got=sigma.at(labels,fraction)
            np.testing.assert_allclose(got,expected,atol=1e-15,rtol=1e-15)
            assert not got.flags.writeable
    base=np.full(1081,1/1081.)
    np.testing.assert_allclose(p.condition_cached_range(s,base,strength,cache),p.condition_range(s,ctx,base,strength,cache.profile),atol=1e-15)


def test_no_live_matchctx_dictionary_access_in_tree_init_or_iterations(monkeypatch):
    s,_,_=sample_spot(18123,3);s.pot=200.;s.to_call=0.;s.legal_action_ids=(1,2,3,4)
    ctx=context();cache=p.prepare_context(ctx,3,s,.25)
    class Forbidden(dict):
        def get(self,*args,**kwargs):raise AssertionError('live HUD dictionary read after entry')
    ctx.successes=Forbidden(ctx.successes);ctx.opportunities=Forbidden(ctx.opportunities)
    ctx.response_by_size=Forbidden(ctx.response_by_size)
    monkeypatch.setattr(p,'profile',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('profile re-parsed in node traversal')))
    monkeypatch.setattr(p,'response_prior',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('sigma rebuilt per node')))
    sol=v.PublicSolver(s,context=ctx,prepared_priors=cache,context_eta_max=.25,iterations=8,check_raise=True).solve()
    assert sol.width==1081 and sol.trace['iterations']==8 and sol.trace['prior_precomputations']==1
    assert np.isfinite(sol.average).all() and abs(sol.trace['zero_sum_residual'])<1e-10

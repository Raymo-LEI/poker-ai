"""Independent small-game, dense-reference and public-contract acceptance."""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from alphanl.stage2 import cfv
from alphanl.stage2 import deep_solver_v2 as v

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/sprint24'))
import evening_pipeline as pipeline


def snap(board=('2s','3h','4d','9c','Ks'),pot=1000.,call=400.,me=2000.,opp=1700.):
    return SimpleNamespace(hole_cards=('5s','6h'),board_cards=board,pot=pot,to_call=call,
        me_stack=me,opp_stack=opp,big_blind=100.,street=len(board)-2,
        legal_action_ids=(0,1,2,3,4) if call else (1,2,3,4),events=(),me_seat=0)


def test_fallback_categories_kickers_wheel_ties(monkeypatch):
    from phevaluator import evaluate_cards
    monkeypatch.setattr(v,'_PE',None)
    categories=['As Kd 9h 6c 3s','As Ad Kh 9c 3s','As Ad Kh Kc 3s',
        'As Ad Ah Kc 3s','2s 3d 4h 5c 6s','As Js 8s 4s 2s','As Ad Ah Kc Ks',
        'As Ad Ah Ac Ks','9s Ts Js Qs Ks']
    values=[v._pack(x.split()) for x in categories]
    assert all(a<b for a,b in zip(values,values[1:]))
    assert v._pack('As 2d 3h 4c 5s'.split())<v._pack('2s 3d 4h 5c 6s'.split())
    board='As Ks Qs Js Ts'.split()
    assert v._pack(board+['2d','3h'])==v._pack(board+['9d','8h'])
    assert v._pack('As Ad Ks Qh Jc'.split())>v._pack('As Ad Ks Qh Tc'.split())
    rng=np.random.RandomState(74)
    for n in (5,6,7):
        for i in range(1000):
            cards=rng.choice(v.DECK,n+2,replace=False).tolist()
            a,b=cards[:n],cards[:n-2]+cards[n:]
            assert np.sign(v._pack(a)-v._pack(b))==np.sign(evaluate_cards(*b)-evaluate_cards(*a))
    assert v._pack('5s 6h 2s 3h 4d 9c Ks'.split())>v._pack('As Qh 2s 3h 4d 9c Ks'.split())


def test_prefix_dense_random_sparse_ties_zero():
    c1,c2,rank,at,hands=v._select(snap().board_cards,snap().hole_cards)
    assert len(hands)==1081
    compatible=(c1[:,None]!=c1)&(c1[:,None]!=c2)&(c2[:,None]!=c1)&(c2[:,None]!=c2)
    rng=np.random.RandomState(19)
    rank=rng.randint(0,20,len(rank))
    eqmat=(rank[:,None]>rank)+.5*(rank[:,None]==rank)
    for weights in (rng.dirichlet(np.ones(len(rank))),np.eye(1,len(rank),at)[0],np.zeros(len(rank))):
        eq,z=v.equity_vs(rank,c1,c2,weights,*v._groups(rank))
        refz=compatible@weights
        ref=np.divide((compatible*eqmat)@weights,refz,out=np.zeros_like(refz),where=refz>1e-14)
        np.testing.assert_allclose(z,refz,atol=2e-14)
        np.testing.assert_allclose(eq,ref,atol=2e-14)
        assert np.all(eq[z==0]==0)
    eq,z=v.equity_vs(np.ones(len(rank)),c1,c2,np.ones(len(rank)),*v._groups(np.ones(len(rank))))
    np.testing.assert_allclose(eq[z>0],.5)


def _leaves(node):
    if node.player is None:yield node.leaf
    else:
        for child in node.children:yield from _leaves(child)


def test_call_terminal_ledger_refunds_and_action_amounts():
    assert v.call_value(.5,1000,400,2000,100)==3.
    assert v.call_value(.5,1000,400,200,100)==3.  # 200 uncalled chips refunded
    s=snap()
    sol=v.PublicSolver(s,iterations=2,amounts={2:777,3:1234,4:2000})
    assert sol.tree.children[sol.tree.labels.index(1)].player is None
    # Compare every showdown leaf with an independent two-stack ledger.
    for kind,h,o,p in _leaves(sol.tree):
        if kind!='show':continue
        assert h<=s.me_stack and o<=s.opp_stack
        assert abs(h-o-s.to_call)<1e-8 or o==0
        before=np.array([s.me_stack,s.opp_stack])
        for won in (True,False):
            after=before-np.array([h,o])
            after[0 if won else 1]+=p+h+o
            delta=after[0]-before[0]
            expected=p+o if won else -h
            assert delta==expected
    assert any(abs(h-777)<1e-9 for k,h,o,p in _leaves(sol.tree) if k=='show')
    sol.solve()
    assert sol.opponent_response.shape==(3,1081,3)
    assert abs(sol.trace['zero_sum_residual'])<1e-10
    # No terminal call acquires a responder subtree, including short calls.
    short=v.betting_tree(1000,400,200,1000,legal=(0,1))
    assert short.children[1].leaf==('show',200,0.,800.)


def test_kuhn_exact_equilibrium_and_counterfactual_reach():
    s=SimpleNamespace(pot=2.,to_call=0.,me_stack=1.,opp_stack=1.,big_blind=1.,
        legal_action_ids=(1,4),board_cards=('2s','3h','4d','9c','Ks'))
    sol=object.__new__(v.PublicSolver)
    sol.s=s;sol.board=s.board_cards;sol.hands=(('2s','3s'),('4s','5s'),('6s','7s'))
    sol.width=3;sol.rank=np.arange(3);sol.bins=np.arange(3)
    sol.c1=np.arange(3);sol.c2=np.arange(3)+3;sol.groups=v._groups(sol.rank)
    sol.rh=np.full(3,1/3.);sol.ro=sol.rh.copy()
    sol.rootmass=(sol.mass(sol.ro),sol.mass(sol.rh))
    sol.tree=v.betting_tree(2.,0.,1.,1.,(1,4));sol.tree.init(3)
    sol.iterations=6000;sol.utility=False;sol.context=None;sol.trace={}
    sol.solve()
    joint=np.dot(sol.rh,sol.rootmass[0])
    value=np.dot(sol.rh*sol.rootmass[0],sol.state)/joint
    assert value==pytest.approx(-1/18,abs=.003)
    def best(node,rh,ro,player):
        if node.player is None:return sol.terminal(node.leaf,rh,ro)[player]
        den=node.total.sum(axis=1,keepdims=True)
        strategy=np.divide(node.total,den,out=np.full_like(node.total,1./len(node.children)),where=den>0)
        children=[best(n,rh*strategy[:,i] if node.player==0 and player!=0 else rh,
                            ro*strategy[:,i] if node.player==1 and player!=1 else ro,player) for i,n in enumerate(node.children)]
        return np.max(children,axis=0) if node.player==player else np.sum(children,axis=0)
    br0=np.dot(sol.rh,best(sol.tree,sol.rh,sol.ro,0))/joint
    br1=np.dot(sol.ro,best(sol.tree,sol.rh,sol.ro,1))/joint
    assert br0+br1<.02
    assert np.any(sol.tree.regret<0)  # signed DCFR, not clipped CFR+


def test_ranges_change_labels_and_stable_public_buckets():
    s=snap();r=np.full(32,1/32.)
    a,b=np.full(32,.00001),np.full(32,.00001);a[0]=1;b[31]=1
    fa,ya,sa=pipeline.label_spot(s,r,a,iterations=5)
    fb,yb,sb=pipeline.label_spot(s,b,b,iterations=5)
    assert np.max(np.abs(ya-yb))>.1
    assert v.build_public_bucket_map(s.board_cards)==v.build_public_bucket_map(tuple(reversed(s.board_cards)))
    np.testing.assert_allclose(fa[32:64],a/a.sum(),atol=1e-7)
    np.testing.assert_array_equal(sa.bins,sb.bins)
    assert np.max(np.abs(sa.opponent_response-sb.opponent_response))>1e-4


def test_features_residual_forward_and_atomic_manifest(tmp_path):
    import torch
    x=[]
    for i in range(3):
        s,rh,ro=pipeline.sample_spot(200+i,3)
        x.append(cfv.spot_features(s.board_cards,rh,ro,s.pot,s.to_call,s.me_stack,100,3,s.opp_stack))
    x=np.array(x,dtype=np.float32);y=np.zeros((3,32),dtype=np.float32)
    weights,stats=pipeline._train(x,y,'cfv_river.',1)
    assert stats['forward_max_error']<1e-5
    assert len(cfv.TAIL_NAMES)==20
    assert not np.all(x[:,121:]==0)
    path=tmp_path/'weights.npz'
    cfv.publish(path,weights,3,1,'river_net')
    a,m=cfv.load_artifact(path,'river_net',3,1)
    assert m['sources']==cfv.source_hashes()
    assert not Path(str(path)+'.tmp').exists()
    with path.open('ab') as handle:handle.write(b'corrupt')
    with pytest.raises(ValueError):cfv.load_artifact(path,'river_net')
    with pytest.raises(ValueError):cfv.validate_weights({})


def _random_nets():
    rng=np.random.RandomState(17)
    out={'cfv_schema':np.frombuffer(cfv.SCHEMA.encode(),dtype=np.uint8)}
    for p in ('cfv_river.','cfv_turn.'):
        for i,shape in enumerate(((256,136),(256,256),(256,256),(256,256),(32,256))):
            out[p+'%d.weight'%i]=rng.normal(0,.01,shape).astype(np.float32)
            out[p+'%d.bias'%i]=np.zeros(shape[0],dtype=np.float32)
    return out


def test_cross_street_identity_chips_posteriors_and_real_network(monkeypatch):
    nets=_random_nets();calls=[]
    original=v.predict_residual_cfv
    def traced(w,p,f):
        assert w is nets and np.asarray(f).shape==(2,136)
        calls.append((p,np.array(f)))
        return original(w,p,f)
    monkeypatch.setattr(v,'predict_residual_cfv',traced)
    s=snap(('2s','3h','4d','9c'),pot=1200,call=300,me=1800,opp=2200)
    sol=v.solve_public_subgame(s,weights=nets,iterations=2,chance_samples=1)
    assert sol.trace['cfv_calls']>0 and calls
    for d in sol.trace['leaf_states']:
        assert d['pot']==s.pot+d['H']+d['O']
        assert d['hero_stack']==s.me_stack-d['H']
        assert d['opp_stack']==s.opp_stack-d['O']
    assert any(not np.allclose(f[0,:32],f[0,32:64]) for p,f in calls)
    for board,bins,idx in [c for k,c in sol._chance_cache.items() if isinstance(k,str)]:
        hands,_,_=v._public(board)
        for i,j in enumerate(idx):
            assert (j<0)==bool(set(board)-set(sol.board) & set(sol.hands[i]))
            if j>=0:assert frozenset(hands[j])==frozenset(sol.hands[i])
    assert abs(sol.trace['zero_sum_residual'])<1e-8
    from evening_cd import search_label
    out=search_label(s,None,np.ones(5)/5,nets,iterations=1,chance_samples=1)
    assert out.sum()==pytest.approx(1)
    assert search_label.last_trace['cfv_calls']>0
    assert search_label.last_trace['target']=='dcfr_average'
    with pytest.raises(ValueError):search_label(s,None,np.ones(5)/5,{},iterations=1)


def test_terminal_utility_expectation_and_symmetric_zero_sum():
    s=snap();s.legal_action_ids=(1,)
    ctx=SimpleNamespace(hero_net_bb=40,hands_played=49)
    sol=v.PublicSolver(s,utility=True,context=ctx,iterations=1)
    leaf=sol.tree.children[0].leaf
    hv,ov=sol.terminal(leaf,sol.rh,sol.ro)
    assert abs(np.dot(sol.rh,hv)+np.dot(sol.ro,ov))<1e-10
    x=np.array([-30.,10.])
    transformed=v._phi_blend(x,100,40,49)
    sigma=1200.;lead=4000.
    ref=.5*x+.5*(np.array([.5*(1+math.erf((lead+t*100)/(sigma*math.sqrt(2)))) for t in x])-.5*(1+math.erf(lead/(sigma*math.sqrt(2)))))*sigma/100
    np.testing.assert_allclose(transformed,ref)
    assert abs(transformed.mean()-v._phi_blend(x.mean(),100,40,49))>1e-4


def test_stats_budget_lock_and_rollback_gate():
    import tournament
    import evening_d
    from alphanl.stage2.evening_strategy import locked_action
    rows=[[100,-10],[100,-10]]
    assert evening_d.summarize(rows)['match_win']==.5
    assert tournament.summarize(rows)['win_rate']==.5
    assert tournament.remaining_posts(0,0)==3750
    assert tournament.remaining_posts(0,1)==3750
    assert tournament.remaining_posts(49,0)==50
    assert tournament.remaining_posts(49,1)==100
    s=snap();s.starting_stack=2000
    losing=SimpleNamespace(hero_net_bb=-1000,hands_played=49)
    assert locked_action(s,losing) is None
    winning=SimpleNamespace(hero_net_bb=1000,hands_played=49)
    assert locked_action(s,winning)==0
    s.to_call=0
    assert locked_action(s,winning) is None
    import evening_cd
    class Model:
        pass
    a,b=Model(),Model()
    monkey=pytest.MonkeyPatch()
    monkey.setattr(evening_cd,'_metrics',lambda m,r,t:{'ce':2 if m is a else 1,'max_tv':.02 if m is a else 0})
    monkey.setattr(evening_cd,'_probabilities',lambda m,r:np.array([[1.,0,0,0,0]]) if m is a else np.array([[0.,1,0,0,0]]))
    assert not evening_cd.validation_gate(a,b,([{}],np.zeros((1,5))),([{}],np.zeros((1,5))))['accepted']
    monkey.undo()


def test_frozen_preflop_route_survives_shared_residual_edits(tmp_path):
    import torch
    from alphanl.cfr.config import CFRConfig
    from alphanl.cfr.distillation import PolicyNetwork
    from alphanl.stage2.policy import ResidualExploiterPolicy
    from alphanl.transformer.config import ModelConfig
    from alphanl.stage2.env import BotzoneMatchEnv
    from alphanl.cfr.evaluation import RuleActor
    config=CFRConfig(equity_features=False,model=ModelConfig(d_model=32,nhead=2,num_layers=1,dim_feedforward=64,max_seq_len=24,dropout=0.))
    model=ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    model._evening_preflop_anchor={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('context_encoder.','fusion.','delta_head.'))}
    env=BotzoneMatchEnv(config,RuleActor('call',seed=1),seed=5)
    batch=env.reset()
    assert env._hero_decision_snapshot().street==0
    before=model(batch)[0].detach().clone()
    with torch.no_grad():model.delta_head.bias.add_(torch.tensor([10.,-10.,8.,-8.,4.]))
    after=model(batch)[0].detach()
    torch.testing.assert_close(after,before,rtol=0,atol=0)
    from alphanl.stage2.train import save_checkpoint,load_checkpoint
    path=tmp_path/'routed.pt'
    save_checkpoint(path,model)
    restored,_=load_checkpoint(path)
    torch.testing.assert_close(restored(batch)[0],before,rtol=0,atol=0)


def test_tree_settlement_against_pokerkit_engine():
    from alphanl.transformer.hunl import HunlTable
    # Actual river successors, legal root amounts, folds/calls/re-raises and jams.
    for seed,stacks,bet in ((5,(20000,20000),100),(17,(1000,1600),100),(31,(1600,1000),100),(42,(3000,800),1000)):
        table=HunlTable(seed=seed);table.starting_stacks=stacks;table.reset()
        while table.street<3:table.step(1)
        table.step_raise_to(bet)
        hero=table.current_player()
        s=table.snapshot(hero)
        amounts={a:table.action_chips(a) for a in s.legal_action_ids}
        tree=v.betting_tree(s.pot,s.to_call,s.me_stack,s.opp_stack,s.legal_action_ids,amounts)
        hero_rank=v._pack([cfv.card(c) for c in table.hole_cards(hero)+table.board_cards()])
        opp_rank=v._pack([cfv.card(c) for c in table.hole_cards(1-hero)+table.board_cards()])
        eq=float(hero_rank>opp_rank)+.5*float(hero_rank==opp_rank)
        before=s.me_stack
        def visit(node,t):
            if node.player is None:
                assert t.is_over()
                k,h,o,p=node.leaf
                expected=-h if k=='fold0' else s.pot+o if k=='fold1' else eq*(p+h+o)-h
                actual=float(t.state.stacks[hero])-before
                assert actual==pytest.approx(expected,abs=.5)
                return
            assert t.current_player()==(hero if node.player==0 else 1-hero)
            for label,child in zip(node.labels,node.children):
                a={'fold':0,'call':1,'raise_all_in':4}.get(label,label)
                branch=t.clone();branch.step(a);visit(child,branch)
        visit(tree,table)


def test_collection_executes_search_policy_not_blueprint(monkeypatch):
    import torch
    import evening_cd
    import alphanl.stage2.env as env_module
    executed=[]
    class Model:
        config=SimpleNamespace()
        context_fields=()
        settings={}
        def eval(self):return self
        def parameters(self):yield torch.zeros(1)
        def __call__(self,obs):return torch.tensor([[100.,0.,0.,0.,0.]]),None
    class Env:
        def __init__(self,*a,**kw):
            self.done=False;self.ctx=SimpleNamespace(hands_played=0);self.table=SimpleNamespace(action_chips=lambda a:0)
        def reset(self):pass
        def _hero_decision_snapshot(self):return SimpleNamespace(street=2,events=(),legal_action_ids=(0,1,2,3,4))
        def observe(self):return {'dummy':torch.zeros((1,1))}
        def step(self,a):executed.append(a);self.done=True
    monkeypatch.setattr(env_module,'BotzoneMatchEnv',Env)
    monkeypatch.setattr(evening_cd,'search_label',lambda *a,**kw:np.array([0.,1.,0.,0.,0.],dtype=np.float32))
    rows,targets=evening_cd.collect(Model(),1,123,nets={})
    assert executed==[1]
    np.testing.assert_array_equal(targets[0],[0,1,0,0,0])


def test_release_verifier_handles_native_replay_html():
    import json
    from verify_evening_release import parse_replay,extract
    debug='py3.6.5 np1.19.5 ok V20_MODEL.npz release=V20_EVENING_V2_20261104_SMOKE wiz=v2/'
    raw=[{'0':{'debug':debug}}]
    html='<script>var _rawLogJSON = '+json.dumps(json.dumps(raw))+';</script>'
    assert list(extract(parse_replay(html)))==[debug]
    with pytest.raises(ValueError):parse_replay('<script>var _rawLogJSON = "[Truncated]";</script>')

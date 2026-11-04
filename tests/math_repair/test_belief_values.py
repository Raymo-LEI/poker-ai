import numpy as np
import pytest
from repair_runtime.search_belief import ComboBelief,RangeError,normalized_policy,sample_ranges
from repair_runtime.search_math import combos,Chance,DCFR
from repair_runtime.search_values import (inputs,flatten_input,VALUE_CONTRACT,recognize_artifacts,
    checkdown_equity,solve_label,full_turn_tree,translate_leaf,encode_outputs,decode_outputs)
from repair_runtime.search_game import PublicState


def test_bayes_reveal_and_independent_buckets():
    hands=(('As','Ah'),('Ks','Kh'),('Qs','Qh'))
    h=ComboBelief(hands,[.2,.3,.5]);o=ComboBelief(hands,[.5,.4,.1])
    h.update([.5,.2,0.]);o.update([.1,.5,0.])
    assert np.allclose(h.weights,[.625,.375,0.])
    h.reveal(['As','2d','3c'])
    assert np.array_equal(h.weights,[0,1,0])
    assert np.array_equal(h.aggregate([2,0,1],3),[1,0,0])
    assert np.allclose(o.aggregate([0,2,1],3),[.2,0,.8])
    with pytest.raises(RangeError):
        h.update([1,0,1])
    p=normalized_policy(np.array([[1,2,3,4,5.]]),[0,1])
    assert np.allclose(p,[[1/3,2/3,0,0,0]])


def test_sampled_hands_are_pairs_and_candidates_may_overlap():
    hs=combos(['2s','3h','4d','9c','Ks'])
    ws=(np.ones(len(hs))/len(hs),)*2
    support,mass=sample_ranges(hs,ws,hs[0],count=30)
    assert all(len(h)==2 for side in support for h in side)
    assert len(set(c for h in support[0] for c in h))<2*len(support[0])
    assert mass[0][0]==pytest.approx(1/1081)
    assert mass[0][1]==pytest.approx(1080/(30*1081))


def test_value_schema_baselines_and_legacy_rejection():
    state=PublicState(1000,(9000,9000),street=3)
    hs=((('As','Ah'),),(('Ks','Kh'),))
    row=inputs(state,['2s','3h','4d','9c','Qs'],hs,([1.],[1.]))
    assert row['contract']==VALUE_CONTRACT and row['pot_bb']==10
    assert np.array_equal(decode_outputs(encode_outputs([3.,-3.])),[3.,-3.])
    assert len(flatten_input(row))==22
    leaf=state.step(3,1000).step(1,1000)
    # leaf is +2000 pot and -1000 each stack: centered value shift is zero.
    assert translate_leaf(7.,state,leaf,0)==7.
    state2=PublicState(1000,(9000,8600),400,street=3)
    leaf2=state2.step(1,400)
    assert translate_leaf(7.,state2,leaf2,0)==5.
    status=recognize_artifacts({'cfv_river.0.weight':np.zeros((32,200))},{'schema':'combo_value_v1'})
    assert status['rejected_legacy']==['cfv_river.0.weight']
    assert status['network'].startswith('unavailable')
    with pytest.raises(ValueError,match='street/board'):
        inputs(state,['2s','3h','4d','9c'],hs,([1.],[1.]))


def test_river_targets_depend_on_input_ranges_and_weighted_zero_sum():
    board=['2s','3h','4d','9c','Ks']
    hands=((('5s','6h'),('Ah','Ad')),(('As','Qh'),('9s','9h')))
    state=PublicState(1000,(100,0),100,street=3)
    a,ra=solve_label(state,board,hands,([.4,.6],[.9,.1]),iterations=50)
    b,rb=solve_label(state,board,hands,([.4,.6],[.1,.9]),iterations=50)
    assert not np.allclose(a['targets'][0],b['targets'][0])
    assert max(a['zero_sum_error'],b['zero_sum_error'])<1e-12
    assert a['continuation']=='river_terminal'
    assert a['contract']['units']=='BB'


def test_turn_chance_enumeration_and_river_betting():
    board=['2s','3h','4d','9c']
    hands=((('5s','6h'),('Ah','Ad')),(('As','Qh'),('9s','9h')))
    state=PublicState(1000,(100,100),street=2)
    tree=full_turn_tree(state,board,hands,raise_cap=1)
    # Check/check is an actual chance node; no river label is copied.
    chance=tree.children[1].children[1]
    assert isinstance(chance,Chance)
    assert len(chance.children)==48
    assert np.allclose(sum(p for p,_ in chance.children),1)
    assert any(hasattr(child,'actions') for _,child in chance.children)
    row,result=solve_label(state,board,hands,([.4,.6],[.9,.1]),iterations=20)
    assert row['continuation']=='enumerated_river_betting'
    assert row['zero_sum_error']<1e-11
    eq,valid=checkdown_equity(board,hands)
    assert (eq[valid]>=0).all() and (eq[valid]<=1).all()


def test_event_likelihood_uses_then_board_each_candidate_and_migrates_history():
    from types import SimpleNamespace as S
    from repair_runtime.search_belief import history_beliefs
    class Snap(S):
        @property
        def me_stack(self):return self.stacks[self.me_seat]
        @property
        def opp_stack(self):return self.stacks[1-self.me_seat]
    def event(street,seat,kind,pot,chips,owed,stack):
        return S(street=street,actor_seat=seat,actor_is_me=seat==1,action_type=kind,
                 pot_before=pot,chips_added=chips,to_call_before=owed,stack_before=stack)
    events=(event(0,1,'call',150,50,50,19950),event(0,0,'check',200,0,0,19900),
            event(1,1,'check',200,0,0,19900),event(1,0,'bet',200,100,0,19900),
            event(1,1,'call',300,100,100,19900),event(2,1,'check',400,0,0,19800),
            event(2,0,'check',400,0,0,19800))
    snap=Snap(events=events,hole_cards=('S5','H6'),board_cards=('S2','H3','D4','C9','SK'),
              me_seat=1,sb_seat=1,street=3,pot=400,stacks=(19800,19800),to_call=0,
              legal_action_ids=(1,2,3,4),big_blind=100,starting_stack=20000)
    class Policy:
        def __init__(self):self.seen=set()
        def raw_observation(self,s,c):
            self.seen.add((s.street,len(s.board_cards)))
            hand=tuple(map(lambda x:'23456789TJQKA'.index(x[1]),s.hole_cards))
            return dict(legal_mask=np.array([[a in s.legal_action_ids for a in range(5)]]),
                        hidden=np.array([[sum(hand)]]))
        def logits(self,b):
            logits=np.zeros((len(b['hidden']),5))
            logits[:,1]=b['hidden'][:,0]/13
            return np.where(b['legal_mask'],logits,-1e9)
    policy=Policy()
    hands,ranges=history_beliefs(policy,snap,S())
    assert policy.seen=={(0,0),(1,3),(2,4)}
    assert len(hands)==1081 and np.std(ranges[1])>0
    snap.hole_cards=('HA','DA')
    hands2,ranges2=history_beliefs(policy,snap,S())
    assert hands==hands2
    assert np.array_equal(ranges[0],ranges2[0]) and np.array_equal(ranges[1],ranges2[1])


def test_unreachable_label_rows_use_none_and_mask():
    state=PublicState(1000,(100,0),100,street=3)
    hs=((('5s','6h'),('As','Ah')),(('As','Qh'),))
    row,_=solve_label(state,['2s','3h','4d','9c','Ks'],hs,([1.,1.],[1.]),iterations=10)
    assert row['reachable'][0]==[True,False]
    assert row['targets'][0][1] is None


def test_sampled_runouts_are_reproducible_complementary_and_support_independent():
    hands=((('As','Ah'),('7s','2h')),(('Ks','Kh'),('Qs','Jh')))
    a,valid=checkdown_equity([],hands,samples=16)
    reverse,_=checkdown_equity([],hands[::-1],samples=16)
    assert np.array_equal(a[valid],(1-reverse.T)[valid])
    singleton,_=checkdown_equity([],((hands[0][0],),(hands[1][0],)),samples=16)
    assert singleton[0,0]==a[0,0]

import itertools
import numpy as np
import pytest
from repair_runtime.portable_evaluator import strength
from repair_runtime.search_math import rank,ShowdownKernel,DCFR,Terminal,Node,hand_masks


def test_all_nine_categories_kickers_wheel_and_board_ties():
    examples=['AsKd9c7h4d','AsAhKd9c7h','AsAhKdKc7h','AsAhAcKd7h',
              '2s3h4d5c6s','AsJs9s7s4s','AsAhAcKdKc','AsAhAcAdKc','9sTsJsQsKs']
    # Parse explicit two-character cards; the evaluator uses independent tuples.
    hands=[[s[i:i+2] for i in range(0,len(s),2)] for s in examples]
    assert [strength(h)[0] for h in hands] == list(range(9))
    assert [rank(h) for h in hands] == sorted(rank(h) for h in hands)
    assert rank(['As','Ah','Kd','9c','7h']) > rank(['As','Ah','Qd','9c','7h'])
    assert rank(['As','2h','3d','4c','5h']) < rank(['2s','3h','4d','5c','6h'])
    board=['Ts','Js','Qs','Ks','As']
    assert rank(['2h','3c']+board) == rank(['9d','8c']+board)


def dense_reference(kernel,w):
    # No packed ranks, prefixes or container internals in the oracle.
    keys=[strength(list(h)+list(kernel.board)) for h in kernel.hands]
    out=[];mass=[]
    for i,h in enumerate(kernel.hands):
        legal=[j for j,o in enumerate(kernel.hands) if not (set(h)&set(o))]
        z=sum(w[j] for j in legal)
        out.append(sum(w[j]*((keys[i]>keys[j])+.5*(keys[i]==keys[j])) for j in legal)/z if z else np.nan)
        mass.append(z)
    return np.array(out),np.array(mass)


@pytest.mark.parametrize('board,sparse', [(['2s','3h','4d','9c','Ks'],False),
    (['Ts','Js','Qs','Ks','As'],False),(['2s','2h','2d','9c','9s'],True)])
def test_full1081_kernel_against_independent_dense(board,sparse):
    kernel=ShowdownKernel(board)
    rng=np.random.RandomState(41)
    w=rng.random_sample(1081)
    if sparse:
        w[rng.random_sample(1081)<.995]=0.
    expected,z=dense_reference(kernel,w)
    actual,reachable=kernel.equity(w)
    assert len(kernel.hands)==1081
    assert np.array_equal(reachable,z>0)
    assert np.nanmax(abs(actual-expected))<2e-13
    h=kernel.hands[0]
    assert sum(not (set(h)&set(o)) for o in kernel.hands)==990


def test_zero_mass_is_unreachable_not_an_epsilon_prior():
    kernel=ShowdownKernel(['2s','3h','4d','9c','Ks'])
    w=np.zeros(1081);w[0]=.1
    value,reachable=kernel.equity(w)
    conflicts=[bool(set(h)&set(kernel.hands[0])) for h in kernel.hands]
    assert np.isnan(value[conflicts]).all()
    assert not reachable[conflicts].any()
    with pytest.raises(ValueError,match='zero compatible'):
        DCFR(Terminal(coefficients=(1.,0.)),kernel=kernel,ranges=(np.zeros(1081),w))


def test_signed_regret_and_iteration_discount():
    r=np.array([[-1.,2.]])
    assert np.allclose(DCFR.discount(r,1),[[-.5,1.]])
    assert np.allclose(DCFR.discount(r,2),[[-.5,2*2**1.5/(2**1.5+1)]])
    assert np.allclose(DCFR.matching(r),[[0.,1.]])
    assert np.allclose(DCFR.matching(np.array([[-1.,-2.]])),[[.5,.5]])


def test_multilayer_average_uses_own_reach_and_same_snapshot():
    leaf=lambda x:Terminal(np.array([[float(x)]]))
    later=Node(0,{'lose':leaf(-1),'win':leaf(1)},key='later')
    root=Node(0,{'enter':later,'stop':leaf(0)},key='root')
    solver=DCFR(root,np.ones((1,1)))
    solver.regret['root'][:]=[1.,3.]
    solver.regret['later'][:]=[3.,1.]
    solver.solve(2)
    expected=.25*np.array([.75,.25])+4/14*np.array([1/3,2/3])
    assert np.allclose(solver.sums['later'][0],expected)
    assert not np.allclose(expected/expected.sum(),(np.array([.75,.25])+4*np.array([1/3,2/3]))/5)


def test_kuhn_value_with_later_hero_infoset():
    show=np.sign(np.arange(3)[:,None]-np.arange(3)[None,:]).astype(float)
    terminal=lambda x:Terminal(np.full((3,3),x) if np.isscalar(x) else x)
    later=Node(0,{'fold':terminal(-1),'call':terminal(2*show)},key='p0_check_bet')
    check=Node(1,{'check':terminal(show),'bet':later},key='p1_after_check')
    bet=Node(1,{'fold':terminal(1),'call':terminal(2*show)},key='p1_after_bet')
    root=Node(0,{'check':check,'bet':bet},key='root')
    joint=(1-np.eye(3))/6
    result=DCFR(root,joint).solve(5000)
    assert abs(result['value']+1/18)<.002
    assert result['zero_sum_error']<1e-14


def test_full_solver_fast_and_dense_on_same_sparse_game():
    from repair_runtime.search_game import PublicState,build_tree
    k=ShowdownKernel(['2s','3h','4d','9c','Ks'])
    hi=np.array([0,100,500,900]);oi=np.array([35,250,550,1000])
    w0=np.zeros(1081);w1=np.zeros(1081)
    w0[hi]=[.1,.2,.3,.4];w1[oi]=[.4,.3,.2,.1]
    keys=[strength(list(h)+list(k.board)) for h in k.hands]
    valid=np.array([[not(set(k.hands[i])&set(k.hands[j])) for j in oi] for i in hi])
    eq=np.array([[(keys[i]>keys[j])+.5*(keys[i]==keys[j]) for j in oi] for i in hi])
    s=PublicState(1000,(200,200))
    fast_tree=build_tree(s,100,kernel=k)
    dense_tree=build_tree(s,100,equity=eq)
    fast=DCFR(fast_tree,kernel=k,ranges=(w0,w1)).solve(40)
    dense=DCFR(dense_tree,w0[hi,None]*w1[None,oi]*valid).solve(40)
    assert np.allclose(fast['values'][0][hi],dense['values'][0],atol=2e-12,equal_nan=True)
    assert np.allclose(fast['values'][1][oi],dense['values'][1],atol=2e-12,equal_nan=True)
    assert np.allclose(fast['strategy'][fast_tree.key][hi],dense['strategy'][dense_tree.key],atol=2e-12)


def test_kuhn_independent_pure_best_response_gap():
    # Independent normal-form enumeration checks both players, not just EV.
    show=np.sign(np.arange(3)[:,None]-np.arange(3)[None,:]).astype(float)
    term=lambda x:Terminal(np.full((3,3),x) if np.isscalar(x) else x)
    later=Node(0,{'fold':term(-1),'call':term(2*show)},key='later')
    check=Node(1,{'check':term(show),'bet':later},key='check')
    bet=Node(1,{'fold':term(1),'call':term(2*show)},key='bet')
    root=Node(0,{'check':check,'bet':bet},key='root')
    joint=(1-np.eye(3))/6
    result=DCFR(root,joint).solve(5000);s=result['strategy']
    def ev(p0,p1):
        open_bet=p0[:3,None];call=p0[3:,None]
        bet_checked=p1[None,:3];call_bet=p1[None,3:]
        u_bet=(1-call_bet)+call_bet*2*show
        u_check=(1-bet_checked)*show+bet_checked*((1-call)*(-1)+call*2*show)
        return float(np.sum(joint*(open_bet*u_bet+(1-open_bet)*u_check)))
    p0=np.r_[s['root'][:,1],s['later'][:,1]]
    p1=np.r_[s['check'][:,1],s['bet'][:,1]]
    pure=list(map(np.array,itertools.product((0.,1.),repeat=6)))
    gap=max(ev(x,p1) for x in pure)-min(ev(p0,x) for x in pure)
    assert gap<.01


def test_extremely_sparse_mass_uses_dense_cancellation_fallback():
    k=ShowdownKernel(['2s','3h','4d','9c','Ks'])
    i=0
    blocked=next(j for j,h in enumerate(k.hands) if j!=i and set(h)&set(k.hands[i]))
    legal=next(j for j,h in enumerate(k.hands) if not(set(h)&set(k.hands[i])))
    w=np.zeros(1081);w[blocked]=1.;w[legal]=1e-100
    z,won=k.contract(w)
    assert z[i]==pytest.approx(1e-100,rel=1e-15,abs=0.)
    expected=(k.ranks[i]>k.ranks[legal])+.5*(k.ranks[i]==k.ranks[legal])
    assert won[i]/z[i]==expected

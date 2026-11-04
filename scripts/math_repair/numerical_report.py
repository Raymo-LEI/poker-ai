"""Independent small-game and dense showdown numerical acceptance report."""
import itertools
import json
import sys
import types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
pkg=types.ModuleType('repair_runtime');pkg.__path__=[str(ROOT/'src/alphanl/stage2')]
sys.modules['repair_runtime']=pkg
from repair_runtime.search_math import ShowdownKernel,DCFR,Node,Terminal
from repair_runtime.portable_evaluator import strength


def main():
    cases=[];rng=np.random.RandomState(41)
    for board in (['2s','3h','4d','9c','Ks'],['Ts','Js','Qs','Ks','As'],['2s','2h','2d','9c','9s']):
        k=ShowdownKernel(board)
        keys=[strength(list(h)+board) for h in k.hands]
        ids={key:i for i,key in enumerate(sorted(set(keys)))}
        score=np.array([ids[key] for key in keys])
        valid=np.array([[not(set(h)&set(o)) for o in k.hands] for h in k.hands])
        dense=((score[:,None]>score[None,:])+.5*(score[:,None]==score[None,:]))*valid
        for sparse in (False,True):
            w=rng.random_sample(1081)
            if sparse:w[rng.random_sample(1081)<.995]=0.
            z=valid.dot(w)
            reference=np.full(1081,np.nan);np.divide(dense.dot(w),z,out=reference,where=z>0)
            actual,reachable=k.equity(w)
            cases.append(dict(board=board,sparse=sparse,max_absolute_error=float(np.nanmax(abs(reference-actual))),
                              reachable_mask_matches=bool(np.array_equal(reachable,z>0))))
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
    p0=np.r_[s['root'][:,1],s['later'][:,1]];p1=np.r_[s['check'][:,1],s['bet'][:,1]]
    pure=list(map(np.array,itertools.product((0.,1.),repeat=6)))
    upper=max(ev(x,p1) for x in pure);lower=min(ev(p0,x) for x in pure)
    report=dict(kernel_cases=cases,kernel_max_error=max(c['max_absolute_error'] for c in cases),
                kuhn=dict(iterations=5000,value=result['value'],analytic_value=-1/18,
                          pure_best_response_upper=upper,pure_best_response_lower=lower,
                          nash_conv_gap=upper-lower,zero_sum_error=result['zero_sum_error']),
                forced_terminal_10bb=DCFR(Terminal(np.full((2,2),10.)),np.ones((2,2))).solve(40)['value'])
    with Path(sys.argv[1]).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Combo-mass preflop reference generator; history-addressed policies, no 169 hack."""
import json
import numpy as np
from .search_math import combos,card_text,DCFR
from .search_game import PublicState,build_tree
from .search_values import checkdown_equity,VALUE_CONTRACT


def class_label(hand):
    a,b=map(card_text,hand)
    if a[0]==b[0]:
        return a[0]*2
    if '23456789TJQKA'.index(a[0])<'23456789TJQKA'.index(b[0]):
        a,b=b,a
    return a[0]+b[0]+('s' if a[1]==b[1] else 'o')


def class_groups():
    groups={}
    for h in combos():
        groups.setdefault(class_label(h),[]).append(h)
    return groups


def history_key(snapshot):
    """Actual event-time actor, call cost, and EXTRA amount, including all-in calls."""
    history=[]
    for e in snapshot.events:
        if e.street != 0:
            continue
        kind=('fold' if e.action_type=='fold' else 'check' if e.chips_added==0 else
              'call' if e.chips_added<=e.to_call_before else 'raise')
        history.append((int(e.actor_seat),kind,int(e.chips_added)))
    return tuple(history)


def solve_depth(stack_bb,hands=None,ranges=None,iterations=100,runouts=16,raise_cap=4):
    """Real chip history tree at THIS depth, with explicitly sampled check-down leaves.

    Full 1326-combo support is available but expensive. The smoke CLI supplies a
    tiny public support and never describes it as a production preflop chart.
    Caller conditioning and both players' reach are handled in the shared tree.
    """
    hs=(combos(),combos()) if hands is None else hands
    ws=(np.ones(len(hs[0])),np.ones(len(hs[1]))) if ranges is None else ranges
    total=int(stack_bb*100)
    state=PublicState(150,(total-50,total-100),50,street=0,street_put=(50,100),round_raise=100,pending=2)
    eq,valid=checkdown_equity([],hs,samples=runouts)
    joint=np.asarray(ws[0])[:,None]*np.asarray(ws[1])[None,:]*valid
    tree=build_tree(state,100.,equity=eq,raise_cap=raise_cap)
    solver=DCFR(tree,joint)
    result=solver.solve(iterations)
    rows=[]
    for key,node in solver.nodes.items():
        rows.append(dict(history=list(key),player=node.player,actions=list(node.actions),
                         strategy=result['strategy'][key].tolist()))
    return dict(schema='preflop_public_history_v1',value_contract=VALUE_CONTRACT,
                stack_bb=stack_bb,continuation='sampled_checkdown',runouts=runouts,
                support=[[list(h) for h in side] for side in hs],
                range_weights=[list(map(float,w)) for w in ws],rows=rows,
                iterations=iterations,zero_sum_error=result['zero_sum_error'],
                production_chart=False)

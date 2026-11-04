"""Shared search adapter: DCFR average policy, never a pairwise oracle or KL mix."""
import json
import numpy as np
from .search_math import CONTRACT, ShowdownKernel, DCFR, card_text
from .search_game import PublicState, build_tree
from .search_belief import history_beliefs, sample_ranges, RangeError
from .search_values import checkdown_equity, recognize_artifacts

# Kept only for old HUD clients; these priors do not enter the search likelihood.
HUD = (('vpip',(.45,6)),('pfr',(.25,6)))


def bind(evaluator):
    """Legacy hook retained for clients. Search always uses the validated comparator."""
    return None


def fuse(agent, snapshot, context, probs, amounts=None):
    state = PublicState.from_snapshot(snapshot)
    if amounts is None:
        amounts = {a:x for a,x in state.actions().items() if a in snapshot.legal_action_ids}
    elif set(amounts) != set(snapshot.legal_action_ids):
        raise ValueError('search requires a complete executable action mapping')
    hands, ranges = history_beliefs(agent,snapshot,context)
    options = getattr(agent,'settings',{}).get('search_v1',{})
    iterations = int(options.get('iterations',24))
    cap = int(options.get('raise_cap',2))
    if iterations < 1 or cap < 1:
        raise ValueError('invalid search budget')
    if snapshot.street == 3:
        kernel = ShowdownKernel(snapshot.board_cards)
        if hands != kernel.hands:
            raise ValueError('river support mismatch')
        tree=build_tree(state,snapshot.big_blind,kernel=kernel,root_amounts=amounts,raise_cap=cap)
        result=DCFR(tree,kernel=kernel,ranges=ranges).solve(iterations)
        at=next(i for i,h in enumerate(hands) if set(h)==set(map(card_text,snapshot.hole_cards)))
        continuation='river_full1081'
    else:
        hs,ws=sample_ranges(hands,ranges,snapshot.hole_cards,count=int(options.get('sample_hands',8)))
        eq,valid=checkdown_equity(snapshot.board_cards,hs,samples=int(options.get('runouts',8)))
        joint=ws[0][:,None]*ws[1][None,:]*valid
        tree=build_tree(state,snapshot.big_blind,equity=eq,root_amounts=amounts,raise_cap=cap)
        result=DCFR(tree,joint).solve(iterations)
        at=0
        continuation='sampled_support_checkdown'
    if not result['reachable'][0][at]:
        raise RangeError('actual hero has zero compatible posterior')
    out=np.zeros(5)
    for a,p in zip(tree.actions,result['strategy'][tree.key][at]):
        out[a]=p
    artifact=recognize_artifacts(agent.weights)
    agent.search_audit=dict(status='active',solver=CONTRACT,value=continuation,
                            output='dcfr_average',iterations=iterations,raise_cap=cap,
                            observation='blueprint_neutral_context_size_coarsened',
                            legacy_rejected=bool(artifact['rejected_legacy']),
                            zero_sum_error=result['zero_sum_error'])
    agent.last_wizard='search=active solver=%s value=%s schema=combo_value_v1 output=dcfr_average iter=%d legacy=%s' % (
        CONTRACT,continuation,iterations,'rejected' if artifact['rejected_legacy'] else 'absent')
    return out

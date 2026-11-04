"""Single strict policy path for evening distillation, evaluation and respond()."""
import numpy as np
from . import cfv
from . import deep_solver_v2 as solver


def public_ranges(snapshot):
    """Public history likelihood approximation, uniformly distributed within buckets.

    This is a declared belief model, not an exact reconstruction of hidden hands.
    The real hero cards are not used to define the public combo support.
    """
    hands,rank,bins=solver._public(tuple(sorted(cfv.card(c) for c in snapshot.board_cards)))
    order=np.argsort(rank,kind='mergesort')
    strength=np.empty(len(rank));strength[order]=(np.arange(len(rank))+.5)/len(rank)
    weights=[np.ones(len(rank)),np.ones(len(rank))]
    for e in getattr(snapshot,'events',()):
        if e.street<=0:continue
        player=0 if e.actor_is_me else 1
        if e.action_type in ('bet','raise','all_in'):
            likelihood=.12+.88*strength**(1+min(float(e.pot_fraction),3.))
        elif e.action_type=='call':
            likelihood=.3+.7*strength
        elif e.action_type=='check':
            likelihood=1.-.5*strength
        else:continue
        weights[player]*=likelihood
    return tuple(solver.compress_range(w,bins) for w in weights)


def locked_action(snapshot,context):
    """Only fold the current facing bet after a conservative guaranteed-win bound.

    Past current-hand investments can still be lost. Budget includes every
    remaining blind at the larger blind, so seat ambiguity cannot create a lock.
    No match is truncated and the opponent's future choices are unrestricted.
    """
    if context is None:return None
    lead=float(getattr(context,'hero_net_bb',0))*snapshot.big_blind
    remaining=max(0,50-int(getattr(context,'hands_played',0)))
    invested=max(0.,float(getattr(snapshot,'starting_stack',snapshot.me_stack))-snapshot.me_stack)
    bound=remaining*snapshot.big_blind+invested
    if lead>bound and snapshot.to_call>0 and 0 in snapshot.legal_action_ids:
        return 0
    return None


def search_policy(snapshot,context,prior,weights,amounts=None,iterations=None,chance_samples=2):
    cfv.validate_weights(weights)
    prior=np.asarray(prior,dtype=np.float64)
    mask=np.zeros(5)
    for a in snapshot.legal_action_ids:mask[a]=1
    prior=np.maximum(prior,0)*mask
    prior=prior/prior.sum() if prior.sum()>0 else mask/mask.sum()
    if int(snapshot.street)<=0:
        return prior,{'search':False,'reason':'preflop_blueprint','cfv_calls':0}
    lock=locked_action(snapshot,context)
    if lock is not None:
        out=np.zeros(5);out[lock]=1
        return out,{'search':False,'reason':'current_hand_guaranteed_lock','cfv_calls':0}
    rh,ro=public_ranges(snapshot)
    sol=solver.solve_public_subgame(snapshot,rh,ro,weights=weights,amounts=amounts,
                                    iterations=iterations,chance_samples=chance_samples,
                                    utility=False,context=context,seed=0)
    hero=frozenset(cfv.card(c) for c in snapshot.hole_cards)
    at=next(i for i,h in enumerate(sol.hands) if frozenset(h)==hero)
    out=np.zeros(5)
    for i,a in enumerate(sol.tree.labels):out[a]=sol.average[at,i]
    out*=mask
    if out.sum()<=0:raise ValueError('solver returned no legal strategy')
    out/=out.sum()
    trace=dict(sol.trace)
    trace.update(search=True,reason='exact_river_terminals' if len(snapshot.board_cards)==5 else 'range_conditioned_cfv_leaves',
                 target='dcfr_average',feature_dim=136)
    return out,trace


def fuse(agent,snapshot,context,probs,amounts):
    weights=cfv.validate_weights(agent.weights)
    iterations=agent.settings.get('evening_iterations')
    samples=agent.settings.get('evening_chance_samples',2)
    out,trace=search_policy(snapshot,context,probs,weights,amounts,iterations,samples)
    agent.search_audit=dict(status='active' if trace['search'] else 'skipped',
                            solver=cfv.SCHEMA,reason=trace['reason'],output='dcfr_average',
                            cfv_calls=trace['cfv_calls'],feature_dim=136)
    agent.last_evening=trace
    agent.last_wizard='wiz=v2/search=%d/cfv=%d/dim=136/reason=%s' % (trace['search'],trace['cfv_calls'],trace['reason'])
    return out

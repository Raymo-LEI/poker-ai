"""Single strict policy path for evening distillation, evaluation and respond()."""
import numpy as np
from . import cfv
from . import deep_solver_v2 as solver


def public_ranges(snapshot):
    """Full-combo public history posterior; never round it through 32 buckets."""
    from .search_belief import history_combo_ranges
    return history_combo_ranges(snapshot)


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


def search_policy(snapshot,context,prior,weights,amounts=None,iterations=None,chance_samples=2,online=False,blueprint_weight=None,check_raise=False,ranges=None,rnr_eta=.5,terminal_match_utility=True,context_prior=True,context_eta_max=.35,size_risk_decay=False,context_range_prior=True):
    if 0<int(snapshot.street)<3:cfv.validate_weights(weights)
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
    from .search_belief import history_combo_ranges
    range_audit=None
    if online and size_risk_decay:
        from .solver_prior import profile
        rnr_eta=min(rnr_eta,profile(context,int(snapshot.street),snapshot,context_eta_max)['eta'])
    if ranges is None:(rh,ro),range_audit=history_combo_ranges(snapshot,return_audit=True,eta=rnr_eta)
    else:rh,ro=ranges
    hero=frozenset(cfv.card(c) for c in snapshot.hole_cards)
    hands=solver._public(tuple(sorted(cfv.card(c) for c in snapshot.board_cards)))[0]
    at=next(i for i,h in enumerate(hands) if frozenset(h)==hero)
    utility=bool(online and terminal_match_utility and context is not None)
    sol=solver.solve_public_subgame(snapshot,rh,ro,weights=weights,amounts=amounts,
                                    iterations=iterations,chance_samples=chance_samples,
                                    utility=utility,context=context,context_prior=bool(online and context_prior),context_eta_max=context_eta_max,context_range_prior=context_range_prior,seed=0,policy_only=True,root_prior=prior if online else None,hero_at=at,check_raise=check_raise)

    out=np.zeros(5)
    for i,a in enumerate(sol.tree.labels):out[a]=sol.average[at,i]
    out*=mask
    if out.sum()<=0:raise ValueError('solver returned no legal strategy')
    out/=out.sum()
    average=out.copy()
    mix=(.3 if len(snapshot.board_cards)==5 else .5) if blueprint_weight is None else float(blueprint_weight)
    if not 0<=mix<=1:raise ValueError('invalid blueprint mixture weight')
    if online:
        out=(1-mix)*average+mix*prior
        out*=mask;out/=out.sum()
    trace=dict(sol.trace)
    trace.update(search=True,reason='exact_river_terminals' if len(snapshot.board_cards)==5 else 'range_conditioned_cfv_leaves',
                 target='blueprint_dcfr_mixture' if online else 'dcfr_average',feature_dim=136,
                 blueprint_weight=mix if online else 0.,initial_strategy='hero_root_blueprint' if online else 'uniform',
                 dcfr_average=average.tolist(),blueprint=prior.tolist() if online else None,
                 root_value_bb=None if utility else float(sol.state[at]),
                 root_match_equity=float(sol.state[at])+.5 if utility else None,range_audit=range_audit,match_utility=utility)

    return out,trace


def fuse(agent,snapshot,context,probs,amounts):
    weights=agent.weights
    online=agent.settings.get('evening_online',{})
    iterations=agent.settings.get('evening_iterations')
    if iterations is None:
        iterations=online.get('iterations',{'1':4,'2':4,'3':8}).get(str(int(snapshot.street)),4)
    samples=online.get('chance_samples',agent.settings.get('evening_chance_samples',1))
    out,trace=search_policy(snapshot,context,probs,weights,amounts,iterations,samples,online=True,
                            blueprint_weight=online.get('blueprint_weight',{'1':.5,'2':.5,'3':.3}).get(str(int(snapshot.street)),.5),
                            check_raise=online.get('check_raise',False),rnr_eta=online.get('rnr_eta',.5),
                            terminal_match_utility=online.get('terminal_match_utility',True),context_prior=online.get('context_prior',True),context_eta_max=online.get('context_eta_max',.35),
                            size_risk_decay=online.get('size_risk_decay',False),context_range_prior=online.get('context_range_prior',True))
    agent.search_audit=dict(status='active' if trace['search'] else 'skipped',
                            solver=cfv.SCHEMA,reason=trace['reason'],output=trace.get('target','blueprint'),
                            cfv_calls=trace['cfv_calls'],feature_dim=136,
                            river_full_combo=bool(snapshot.street==3 and trace['search']),
                            terminal_match_utility=trace.get('terminal_match_utility',False))

    if 'match_prior' in trace:
        agent.search_audit['match_prior']=trace['match_prior']
        agent.search_audit['match_utility']=trace['match_utility']
    agent.last_evening=trace
    agent.last_wizard='wiz=v2/search=%d/cfv=%d/dim=136/reason=%s' % (trace['search'],trace['cfv_calls'],trace['reason'])
    if trace.get('range_audit'):
        agent.search_audit['range_model']=trace['range_audit']['model']
        agent.search_audit['range_support']=trace['range_audit']['support']
        agent.last_wizard+=' range='+trace['range_audit']['model']
    return out

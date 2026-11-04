"""On-policy blueprint self-play, finite-particle public Bayes, mixed CFV labels.

The particle support is sampled independently of both dealt private hands.
No oracle ranges, heuristic strength likelihoods, epsilon mass or hidden-card
conditioning are used. This is a documented finite-support PBS approximation,
projected to 32 public buckets for the existing CFV input contract.
"""
import copy
import time
from pathlib import Path
import numpy as np
from alphanl.stage2 import cfv, deep_solver_v2 as v
from alphanl.stage2.evening_strategy import search_policy
from evening_cd import _batch
from evening_pipeline import sample_spot, _train
from evening_parallel import jobs

TREE='bilateral-checkraise-pot-jam-v1'
PARTICLES=64
_MODEL=None
_NET=None


def init_worker(source,nets=None):
    global _MODEL,_NET
    from alphanl.stage2.train import load_checkpoint
    _MODEL,_=load_checkpoint(source,device='cpu')
    _NET=None if nets is None else cfv.load_artifact(nets)[0]


def posterior_ranges(hands,weights,board):
    public,_,bins=v._public(tuple(sorted(cfv.card(c) for c in board)))
    lookup={h:int(b) for h,b in zip(public,bins)}
    valid=np.array([h in lookup for h in hands])
    ids=np.array([lookup.get(h,0) for h in hands])
    return tuple(cfv.normalize_range(np.bincount(ids,weights=w*valid,minlength=32)) for w in weights)


def bayes(weights,likelihood):
    p=np.asarray(likelihood,dtype=float)
    if p.shape!=weights.shape or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1+1e-12):
        raise ValueError('invalid public likelihood')
    out=weights*p
    if out.sum()<=0:raise ValueError('particle posterior exhausted')
    return out/out.sum()


def trajectory(model,count,seed,street=None,teacher=False,nets=None,iterations=None,chance_samples=2,preflop=False,opponent=None):
    """Collect unique current-policy states, ranges and optional pure DCFR targets."""
    import torch
    from alphanl.transformer.hunl import HunlTable,HunlTokenizer
    from alphanl.stage2.context import MatchCtx
    rng=np.random.default_rng(seed)
    device=next(model.parameters()).device
    tokenizer=HunlTokenizer(model.config.model.max_seq_len,**model.config.tokenizer_kwargs())
    table=HunlTable(seed=seed)
    if opponent is not None:
        from ladder_actors import SlayerActor
        replica=SlayerActor(opponent,seed=seed,equity_samples=16)
    contexts=[MatchCtx(),MatchCtx()]
    result=[];hands_seen=0;exhaustions=0
    def observe(snap,ctx):
        # Match the online legal-only domain, including voluntary all-ins.
        return snap,{**tokenizer.encode(snap),'match_ctx':ctx.tensor(model.context_fields)}
    while len(result)<count:
        hands_seen+=1
        if hands_seen>max(512,count*256):raise RuntimeError('on-policy collection cannot reach requested street')
        # No deal is reused between chunks/splits. Particle RNG never sees deck.
        hand_seed=seed+hands_seen
        table.reset();[ctx.begin_hand() for ctx in contexts]
        hero_seat=hands_seen%2
        if opponent is not None:replica.begin_hand(1-hero_seat)
        public_hands=list(__import__('itertools').combinations(v.DECK,2))
        selected=rng.choice(len(public_hands),PARTICLES,replace=False)
        particles=tuple(tuple(sorted(public_hands[i],key=cfv.CARD_ID.get)) for i in selected)
        mass=np.full((2,PARTICLES),1./PARTICLES)
        exhausted=False
        while not table.is_over() and len(result)<count:
            seat=table.current_player()
            if opponent is not None and seat!=hero_seat:
                from alphanl.transformer.actor_actions import apply_actor_decision
                apply_actor_decision(table,replica.act(table.snapshot(seat)))
                for p in (0,1):contexts[p].observe_action(table.events_for(p)[-1])
                continue
            snap,obs=observe(table.snapshot(seat),contexts[seat])
            board=set(cfv.card(c) for c in snap.board_cards)
            valid=np.array([not set(h)&board for h in particles])
            if opponent is None and np.any(mass[:,valid].sum(axis=1)<=0):
                exhausted=True;break
            if opponent is None:mass*=valid;mass/=mass.sum(axis=1,keepdims=True)
            with torch.inference_mode():
                prior=model({k:x.to(device) for k,x in obs.items()})[0][0].softmax(-1).cpu().double().numpy()
            wanted=(snap.street==0 if preflop else snap.street>0 and (street is None or snap.street==street))
            if wanted:
                item=dict(identity=(hand_seed,len(snap.events),seat),snapshot=snap,
                          rollout_seed=seed,opponent=opponent or 'current_blueprint',row={k:x.numpy().copy() for k,x in obs.items()},prior=prior.astype(np.float32))
                if snap.street>0:
                    if opponent is None:rh,ro=posterior_ranges(particles,(mass[seat],mass[1-seat]),snap.board_cards)
                    else:
                        from alphanl.stage2.search_belief import history_combo_ranges
                        rh,ro=history_combo_ranges(snap)
                    item.update(rh=rh,ro=ro)
                    if teacher:
                        it=iterations.get(str(snap.street)) if isinstance(iterations,dict) else iterations
                        target,trace=search_policy(snap,None,prior,nets,
                            amounts={a:table.action_chips(a) for a in snap.legal_action_ids},iterations=it,
                            chance_samples=chance_samples,check_raise=True,ranges=(rh,ro))
                        if trace.get('target')!='dcfr_average':raise RuntimeError('impure distillation teacher')
                        item.update(target=target.astype(np.float32),value=np.float32(trace['root_value_bb']))
                result.append(item)
                if len(result)==count:break
            action=int(rng.choice(5,p=prior/prior.sum()))
            if opponent is not None:
                table.step(action)
                for p in (0,1):contexts[p].observe_action(table.events_for(p)[-1])
                continue
            # Evaluate the actual observed action under every public particle.
            batches=[];indices=[]
            for i,h in enumerate(particles):
                if not valid[i]:continue
                view=copy.copy(snap);object.__setattr__(view,'hole_cards',tuple(c[1].upper()+c[0] for c in h))
                _,batch=observe(view,contexts[seat]);batches.append({k:x.numpy() for k,x in batch.items()});indices.append(i)
            with torch.inference_mode():
                likelihood=model(_batch(batches,device))[0].softmax(-1)[:,action].cpu().double().numpy()
            obs_likelihood=np.zeros(PARTICLES);obs_likelihood[indices]=likelihood
            try:mass[seat]=bayes(mass[seat],obs_likelihood)
            except ValueError:exhausted=True;break
            table.step(action)
            for p in (0,1):contexts[p].observe_action(table.events_for(p)[-1])
        if table.is_over() and not exhausted:
            for p in (0,1):
                contexts[p].finish_hand(float(table.payoffs()[p])/100.,showdown=not any(e.action_type=='fold' for e in table.events_for(p)))
        if exhausted:exhaustions+=1
        # The HUD resets at complete 50-hand boundaries, independently of splits.
        if hands_seen%50==0:contexts=[MatchCtx(),MatchCtx()]
    trajectory.last_stats=dict(hands=hands_seen,particle_exhaustions=exhaustions)
    return result


def label(snapshot,rh,ro,nets,it,chance,seed):
    sol=v.solve_public_subgame(snapshot,rh,ro,weights=nets,iterations=it,chance_samples=chance,
        seed=seed%(2**32),check_raise=True,first_player=getattr(snapshot,'first_player',0),policy_only=True)
    y=v.bucket_values(sol.state,sol.rh*sol.rootmass[0],sol.bins)
    x=cfv.spot_features(snapshot.board_cards,rh,ro,snapshot.pot,snapshot.to_call,snapshot.me_stack,
        snapshot.big_blind,snapshot.street,snapshot.opp_stack,
        0 if getattr(snapshot,'me_seat',0)==getattr(snapshot,'postflop_first_seat',0) else 1)
    if abs(sol.trace['zero_sum_residual'])>1e-7:raise ValueError('non-zero-sum label')
    return x.astype(np.float32),y.astype(np.float32)


def mixed_worker(payload):
    index,n,seed,street,it,chance=payload;start=time.monotonic()
    real=trajectory(_MODEL,n//2,seed,street=street)
    trajectory_stats=dict(trajectory.last_stats)
    rows=[];identities=[];clusters=[]
    for i,item in enumerate(real):
        rows.append(label(item['snapshot'],item['rh'],item['ro'],_NET,it,chance,seed+i))
        identities.append(item['identity'])
        clusters.append(seed)
    for i in range(n-n//2):
        # SeedSequence accepts the full namespace, without 32-bit wrapping.
        rng=np.random.default_rng(seed+2048+i);deck=rng.permutation(v.DECK)
        from evening_pipeline import _snap
        pot,stack,opp=rng.uniform(400,6000),rng.uniform(800,18000),rng.uniform(800,18000)
        call=0. if rng.random()<.5 else min(stack*.7,pot*.6)
        snap=_snap(deck[street+2:street+4],deck[:street+2],pot,call,stack,street,opp)
        snap.first_player=0 if call else int(rng.integers(2));snap.me_seat=int(rng.integers(2)) if call else snap.first_player
        rh,ro=rng.dirichlet(np.full(32,.7)),rng.dirichlet(np.full(32,.7))
        rows.append(label(snap,rh,ro,_NET,it,chance,seed+2048+i));identities.append((seed+2048+i,-1,-1))
        clusters.append(seed+2048+i)
    x,y=map(np.stack,zip(*rows))
    return index,x,y,identities,start,time.monotonic(),trajectory_stats,clusters


def teacher_worker(payload):
    index,n,seed,it,chance,preflop=payload;start=time.monotonic()
    opponent='Bot17_Probe3Bet' if not preflop and index%5==0 else None
    rows=trajectory(_MODEL,n,seed,teacher=not preflop,nets=_NET,iterations=it,chance_samples=chance,preflop=preflop,opponent=opponent)
    return index,rows,start,time.monotonic()


def collect_teacher(source,n,seed,nets,workers,it,chance,preflop=False):
    payloads=[(i,min(32,n-i*32),seed+i*262144,it,chance,preflop) for i in range((n+31)//32)]
    parts=sorted(jobs(teacher_worker,payloads,len(payloads),workers,init_worker,(str(source),str(nets)),timeout=600,label='ONPOLICY_TEACHER'))
    collect_teacher.last_seconds=max(p[3] for p in parts)-min(p[2] for p in parts)
    return [row for p in parts for row in p[1]]


def mixed_dataset(path,source,street,n,seed,workers,it,chance,nets=None):
    if n<2 or n%2:raise ValueError('50/50 mixture requires a positive even count')
    # Every even chunk contains exactly half trajectories and half exploration.
    payloads=[(i,min(32,n-i*32),seed+i*262144,street,it,chance) for i in range((n+31)//32)]
    parts=sorted(jobs(mixed_worker,payloads,len(payloads),workers,init_worker,(str(source),None if nets is None else str(nets)),timeout=600,label='MIXED_CFV'))
    x=np.concatenate([p[1] for p in parts]);y=np.concatenate([p[2] for p in parts])
    meta=dict(tree=TREE,particles=PARTICLES,pbs='current_blueprint_selfplay_bayes',on_policy_n=n//2,dirichlet_n=n//2,
              seed=seed,identities=[i for p in parts for i in p[3]],source_sha256=cfv.digest(source),
              iterations=it,chance_samples=chance,active_compute_s=max(p[5] for p in parts)-min(p[4] for p in parts),
              trajectory_hands=sum(p[6]['hands'] for p in parts),particle_exhaustions=sum(p[6]['particle_exhaustions'] for p in parts),
              teacher_sha256=None if nets is None else cfv.digest(nets))
    clusters=np.array([c for p in parts for c in p[7]],dtype=np.int64)
    cfv.publish(path,dict(x=x,y=y,cluster_ids=clusters),n,0,'dataset',meta)
    meta['cluster_ids']=clusters
    return x,y,meta


def paired_gain(gain,groups=None,minimum=0.):
    """Paired mean with cluster-robust uncertainty across independent rollouts."""
    gain=np.asarray(gain,dtype=float)
    if gain.ndim!=1 or len(gain)<2 or not np.isfinite(gain).all():
        return dict(accepted=False,reason='insufficient finite pairs')
    groups=np.arange(len(gain)) if groups is None else np.asarray(groups)
    if groups.shape!=gain.shape:raise ValueError('invalid paired cluster identities')
    unique,inverse=np.unique(groups,return_inverse=True);n=len(unique)
    if n<2:return dict(accepted=False,reason='fewer than two independent rollouts',n=len(gain),clusters=n)
    centered=np.bincount(inverse,weights=gain-gain.mean())
    se=float(np.sqrt(n/(n-1)*np.dot(centered,centered))/len(gain))
    lower=float(gain.mean()-1.96*se)
    return dict(accepted=bool(lower>minimum),mean_gain=float(gain.mean()),lower95=lower,n=len(gain),clusters=n,minimum=minimum)


def holdout_gate(weights,prefix,train_y,validation,clusters=None):
    x,y=validation
    pred=cfv.predict_residual_cfv(weights,prefix,x)
    error=np.abs(pred-y);constant=np.abs(train_y.mean(axis=0)-y)
    rows=error.mean(1);gain=(constant-error).mean(1)
    confidence=paired_gain(gain,clusters)
    ratio=float(np.mean(error/np.maximum(x[:,119:120]*200.,1.)))
    result=dict(mae=float(rows.mean()),constant_mae=float(constant.mean()),pot_mae_ratio=ratio,
                gain_lower95=confidence.get('lower95'),samples=len(x),independent_clusters=confidence.get('clusters',0))
    result['passed']=bool(confidence['accepted'] and np.isfinite(rows).all() and np.isfinite(ratio) and rows.mean()<constant.mean()*.90 and ratio<=.12)
    return result


def phase_b(source,out,plan,workers,device,smoke=False):
    out=Path(out);stats={};weights={}
    for street,name,prefix,seed in ((3,'river','cfv_river.',11000000000),(2,'turn','cfv_turn.',21000000000)):
        net_path=None if street==3 else out/'cfv_river.npz'
        it=2 if smoke else 64 if street==3 else 32;chance=1 if smoke else 4
        x,y,meta=mixed_dataset(out/(name+'_ds.npz'),source,street,plan[name+'_n'],seed,workers,it,chance,net_path)
        vx,vy,vm=mixed_dataset(out/(name+'_holdout.npz'),source,street,plan['holdout_n'],seed+5000000000,workers,it,chance,net_path)
        w,s=_train(x,y,prefix,plan['net_epochs'],street,device,(vx,vy),cosine=True)
        gate=holdout_gate(w,prefix,y,(vx,vy),vm['cluster_ids']);s['CFV_HOLDOUT_GATE']=gate
        if not smoke and not gate['passed']:raise RuntimeError('CFV_HOLDOUT_GATE_FAILED '+name+' '+str(gate))
        weights.update(w);stats[name]=s
        if street==3:
            cfv.publish(net_path if net_path else out/'cfv_river.npz',w,len(x),plan['net_epochs'],'river_net',dict(tree=TREE,validation=s))
    cfv.publish(out/'cfv_nets.npz',weights,plan['river_n']+plan['turn_n'],plan['net_epochs'],'nets',
        dict(tree=TREE,river_samples=plan['river_n'],turn_samples=plan['turn_n'],validation=stats,
             mixture='50% current-blueprint on-policy PBS / 50% Dirichlet',particles=PARTICLES))
    cfv.atomic_json(out/'validation.json',stats)
    return stats

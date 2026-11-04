"""Duplicate target-judge evaluation, CPU parallelism, exact packaged respond()."""
import json
import time
from pathlib import Path
import numpy as np
from evening_d import Bundle,play_judge,summarize
from evening_parallel import jobs
from alphanl.stage2 import cfv

BENCHMARK=('equity','TAG_Pro','Trapper_CR')
TOURNAMENT=('Bot17_Probe3Bet','TAG_Pro','Trapper_CR','equity','NearShove_19949')
_BUNDLES={}


def _init_eval(releases):
    global _BUNDLES
    _BUNDLES={name:Bundle(r['source'],r['model']) for name,r in releases.items()}


def _eval_job(payload):
    phase,name,opponent,pair,seat,seed,hands=payload
    result=play_judge(_BUNDLES[name],seed,seat,hands,opponent,keep_trace=False)
    if result['errors'] or (hands==50 and not result['complete']):raise RuntimeError('incomplete or invalid judge match')
    return phase,name,opponent,pair,seat,result


def evaluate(releases,out,workers=24,benchmark_pairs=50,tournament_pairs=10,hands=50,seed_base=1800000000):
    started=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    payloads=[]
    for phase,opponents,pairs in (('benchmark',BENCHMARK,benchmark_pairs),('tournament',TOURNAMENT,tournament_pairs)):
        for name in releases:
            for oi,opponent in enumerate(opponents):
                for pair in range(pairs):
                    for seat in (1,0):
                        # Independent of all label/collection/tuning seeds.
                        seed=seed_base+(0 if phase=='benchmark' else 100000)+oi*1000+pair
                        payloads.append((phase,name,opponent,pair,seat,seed,hands))
    grouped={};latency=[];count=0
    for phase,name,opponent,pair,seat,result in jobs(_eval_job,payloads,len(payloads),workers,_init_eval,(releases,),timeout=240,label='ZIP_EVAL'):
        key=(phase,name,opponent)
        grouped.setdefault(key,{})[(pair,seat)]=result['hero_net_bb']
        latency.extend(result.pop('turn_s'));count+=result['hero_decisions']
        cfv.atomic_json(out/('%s_%s_%s_p%03d_s%d.json'%(phase,name,opponent,pair,seat)),result)
    report={'entry':'packaged runtime.respond','seed_base':seed_base,'opponents':list(TOURNAMENT),'hands_per_match':hands,
            'hero_decisions':count,'elapsed_s':time.monotonic()-started,'results':{}}
    for (phase,name,opponent),scores in grouped.items():
        rows=[[scores[(pair,1)],scores[(pair,0)]] for pair in sorted({k[0] for k in scores})]
        stats=summarize(rows) if hands==50 else {'partial_protocol_only':True,'net_bb':rows}
        report['results'].setdefault(name,{}).setdefault(phase,{})[opponent]=stats
    report['runtime_ms']={'mean':float(np.mean(latency))*1000,'p99':float(np.percentile(latency,99))*1000,'n':len(latency)}
    report['total_played_hands']=len(payloads)*hands
    cfv.atomic_json(out/'summary.json',report)
    return report


def policy_evolution(releases,count=32):
    """Independent actual-runtime policy check, excluding action sampling noise."""
    from evening_latency import cases
    bundles={name:Bundle(r['source'],r['model']) for name,r in releases.items()}
    outputs={name:[] for name in bundles};pre={name:[] for name in bundles}
    for payload in cases(count,1912345678):
        request=payload['requests'][0]
        for name,b in bundles.items():
            snap,responses=b.runtime.request_snapshot(request)
            ctx,_=b.runtime.context_from_payload(payload)
            amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in responses.items()}
            b.agent.search_request=request
            probs=b.agent.policy(snap,ctx,safety=bool(b.meta.get('safety',False)),amounts=amounts,request=request)
            (outputs if snap.street>0 else pre)[name].append(probs)
    result={}
    for before,after in (('baseline','g1'),('g1','g2')):
        tv=.5*np.abs(np.array(outputs[before])-np.array(outputs[after])).sum(axis=1)
        pre_tv=.5*np.abs(np.array(pre[before])-np.array(pre[after])).sum(axis=1)
        result[after]=dict(postflop_mean_tv=float(tv.mean()),postflop_max_tv=float(tv.max()),
                           preflop_max_tv=float(pre_tv.max()),changed=bool(tv.max()>1e-8))
    return result

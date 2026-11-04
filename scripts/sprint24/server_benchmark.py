"""Real-trajectory, isolated-core server latency profile (independent of legacy gates)."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv
from evening_d import Bundle,play_judge
from server_resources import pin

WARMUP=32
TIMED=512


def gates(mean_ms,p99_ms,max_ms,cold_start_s):
    hard=bool(np.isfinite([mean_ms,p99_ms,max_ms,cold_start_s]).all() and
              p99_ms<120. and max_ms<300. and cold_start_s<.5)
    return dict(hard_passed=hard,soft_mean_passed=bool(mean_ms<65.),
                warning=None if mean_ms<65. else 'MEAN_LATENCY_SOFT_WARNING',
                hard_limits=dict(p99_ms=120.,max_ms=300.,cold_start_s=.5),soft_limit_mean_ms=65.)


def collect_corpus(source,model,path,seed=2026110500,max_matches=12):
    """Actual judge deals/actions and their full cumulative request/data payloads."""
    bundle=Bundle(source,model);by_street={0:[],3:[],4:[],5:[]};seen=set()
    def observe(payload):
        request=payload['requests'][-1]
        # Keep the genuine cumulative requests/responses and previous data.
        # No opponent hole cards or future cards enter the payload.
        snapshot,_=bundle.runtime.request_snapshot(request)
        ctx,_=bundle.runtime.context_from_payload(payload)
        state=bundle.modules['blind_lock'].state_from_payload(payload)
        mode,_=bundle.modules['blind_lock'].choice(request,snapshot,state)
        if mode=='own' or bundle.modules['evening_strategy'].locked_action(snapshot,ctx) is not None:return
        row=copy.deepcopy(payload)
        key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key);by_street[len(request['public_cards'])].append(dict(id=key,payload=row))
    matches=[]
    for index in range(max_matches):
        opponent='call' if index%2==0 else 'TAG_Pro'
        result=play_judge(bundle,seed+index//2,index%2,50,opponent,keep_trace=False,observer=observe)
        if not result['complete'] or result['errors']:raise RuntimeError('corpus judge match incomplete')
        matches.append(dict(seed=seed+index//2,seat=index%2,opponent=opponent,net_bb=result['hero_net_bb']))
        if min(map(len,by_street.values()))>=12:break
    if not all(by_street.values()):raise RuntimeError('real corpus lacks a street')
    result=dict(schema='real-judge-states-v1',seed=seed,matches=matches,states=[r for rows in by_street.values() for r in rows],
                by_board_size={str(k):len(v) for k,v in by_street.items()},
                note='Timed sampling is balanced by street with replacement; unique counts are reported.')
    cfv.atomic_json(path,result)
    return result


def sample_states(corpus,count,seed):
    rng=np.random.RandomState(seed);groups={n:[] for n in (0,3,4,5)}
    for row in corpus['states']:groups[len(row['payload']['requests'][-1]['public_cards'])].append(row)
    if not all(groups.values()):raise ValueError('real states required on all four streets')
    keys=(0,3,4,5)
    return [groups[keys[i%4]][rng.randint(len(groups[keys[i%4]]))] for i in range(count)]


def benchmark(source,model,corpus,core,warmup=32,count=512,cold=12,diagnostic=False):
    if not diagnostic and (warmup!=32 or count<500 or cold<1):
        raise ValueError('server benchmark requires 32 warmups, >=500 timed real requests and cold starts')
    affinity=pin([core]);bundle=Bundle(source,model)
    for row in sample_states(corpus,warmup,404):bundle.respond(row['payload'])
    elapsed=[];by={};identities=[];river=0;hud=0;errors=[]
    for row in sample_states(corpus,count,505):
        started=time.perf_counter();bundle.respond(row['payload']);seconds=time.perf_counter()-started
        elapsed.append(seconds*1000);n=len(row['payload']['requests'][-1]['public_cards'])
        by.setdefault(str(n),[]).append(seconds*1000);identities.append(row['id'])
        trace=getattr(bundle.agent,'last_evening',{})
        active=bundle.agent.search_audit.get('status')=='active'
        if n==5 and active and trace.get('search'):
            if trace.get('support')!=1081:raise AssertionError('river did not solve all 1081 public combos')
            river+=1
        if active and trace.get('match_prior',{}).get('eta',0)>0:hud+=1
        if bundle.agent.search_audit.get('status')=='error_blueprint':errors.append(bundle.agent.search_audit)
    cold_times=[];cold_debug=[]
    env=dict(os.environ,POKER_STAGE2_WEIGHTS=str(Path(model).resolve()))
    for row in sample_states(corpus,cold,606):
        started=time.perf_counter()
        p=subprocess.run([sys.executable,str(Path(source).resolve())],input=json.dumps(row['payload'])+'\n',
                         capture_output=True,text=True,env=env,timeout=5)
        cold_times.append(time.perf_counter()-started)
        if p.returncode:raise RuntimeError('cold entry failure: '+p.stderr[-2000:])
        body=json.loads(p.stdout);debug=body.get('debug','');cold_debug.append(debug)
        if not debug.startswith('py') or ' ok ' not in debug or 'error_blueprint' in debug or 'fallback' in debug:
            raise RuntimeError('cold load/search failed: '+debug)
    cm=max(cold_times) if cold_times else float('inf')
    mean=float(np.mean(elapsed));p99=float(np.percentile(elapsed,99));maximum=float(max(elapsed))
    result=dict(samples=count,warmup_requests=warmup,p50_ms=float(np.median(elapsed)),mean_ms=mean,p99_ms=p99,max_ms=maximum,
                cold_start_s=cm,cold_samples=cold,cold_debug=cold_debug,gate=gates(mean,p99,maximum,cm),
                by_board_size={k:dict(n=len(v),mean_ms=float(np.mean(v)),p99_ms=float(np.percentile(v,99)),max_ms=max(v)) for k,v in by.items()},
                timed_ms=elapsed,timed_state_ids=identities,cold_start_samples_s=cold_times,
                cpu_affinity=affinity,corpus_unique_states=len(corpus['states']),timed_unique_states=len(set(identities)),
                corpus_counts=corpus['by_board_size'],river_full_combo_calls=river,hud_active_calls=hud,
                source_sha256=cfv.digest(source),model_sha256=cfv.digest(model),profile=bundle.meta['settings']['evening_online'],
                diagnostic=diagnostic,python=sys.version,numpy=np.__version__,errors=errors)
    if errors or river==0:result['gate']['hard_passed']=False
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--model',required=True)
    p.add_argument('--corpus',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--core',required=True,type=int);p.add_argument('--samples',type=int,default=TIMED)
    p.add_argument('--warmup',type=int,default=WARMUP);p.add_argument('--cold',type=int,default=12)
    p.add_argument('--collect',action='store_true');p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args(argv);pin([a.core])
    if a.collect and not a.corpus.exists():collect_corpus(a.zip,a.model,a.corpus)
    corpus=json.loads(a.corpus.read_text())
    r=benchmark(a.zip,a.model,corpus,a.core,a.warmup,a.samples,a.cold,a.diagnostic)
    cfv.atomic_json(a.out,r)
    print(json.dumps({k:r[k] for k in ('mean_ms','p50_ms','p99_ms','max_ms','cold_start_s','gate','river_full_combo_calls','hud_active_calls')}),flush=True)
    return 0 if r['gate']['hard_passed'] else 2

if __name__=='__main__':raise SystemExit(main())

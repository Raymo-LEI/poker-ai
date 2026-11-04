"""Hard latency gate for the actual ZIP respond path and fresh process entry."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from evening_d import Bundle
from selftest_evening_v2 import requests as probe_requests
from alphanl.stage2 import cfv


def cases(count,seed=609113):
    rng=np.random.RandomState(seed)
    base=probe_requests()
    rows=[]
    for i in range(count):
        r=dict(base[i%4]);n=len(r['public_cards'])
        cards=rng.choice(52,n+2,replace=False).tolist()
        r['my_cards']=cards[:2];r['public_cards']=cards[2:]
        r['total_win_chips']=[0,0];r['hand']=i%45
        rows.append({'requests':[r]})
    return rows


def benchmark(source,model,count=256,cold_count=12,gate=True,pin=True):
    affinity=None
    if pin and hasattr(os,'sched_getaffinity'):
        affinity=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity})
    bundle=Bundle(source,model)
    # Exercise every street before measuring so schedutil can raise CPU frequency.
    # Fresh-process cold probes below remain fully timed, including startup.
    warmup_count=16
    for payload in cases(warmup_count,609114):
        bundle.respond(payload)
    warm=[];per={};audits=[]
    for payload in cases(count):
        t=time.perf_counter();body=bundle.respond(payload);elapsed=time.perf_counter()-t
        if not isinstance(body['response'],int):raise ValueError('non-integer response')
        trace=getattr(bundle.agent,'last_evening',{})
        n=len(payload['requests'][0]['public_cards'])
        if n and (not trace.get('search') or (n<5 and trace.get('cfv_calls',0)<=0)):
            raise RuntimeError('latency probe skipped actual CFV search')
        warm.append(elapsed);per.setdefault(str(n),[]).append(elapsed)
        if len(audits)<4:audits.append(bundle.debug())
    env=dict(os.environ,POKER_STAGE2_WEIGHTS=str(Path(model).resolve()))
    cold=[];cold_debug=[]
    for payload in cases(cold_count,818191):
        t=time.perf_counter()
        proc=subprocess.run([sys.executable,str(Path(source).resolve())],input=json.dumps(payload)+'\n',capture_output=True,text=True,env=env,timeout=5)
        elapsed=time.perf_counter()-t
        if proc.returncode:raise RuntimeError(proc.stderr)
        body=json.loads(proc.stdout);debug=body.get('debug','')
        if not debug.startswith('py') or ' ok ' not in debug or 'fallback' in debug or 'error_blueprint' in debug:
            raise RuntimeError('cold runtime failure: '+debug)
        cold.append(elapsed);cold_debug.append(debug)
    mean=float(np.mean(warm));p99=float(np.percentile(warm,99));cm=max(cold) if cold else None
    result=dict(mean_ms=mean*1000,p99_ms=p99*1000,max_ms=max(warm)*1000,
        cold_max_s=cm,samples=count,cold_samples=cold_count,warmup_samples=warmup_count,
        by_board_size={k:dict(mean_ms=float(np.mean(v))*1000,p99_ms=float(np.percentile(v,99))*1000,n=len(v)) for k,v in per.items()},
        settings=bundle.meta['settings'],arch=bundle.meta['config']['model'],release=bundle.meta['deployment_release'],
        source_sha256=cfv.digest(source),model_sha256=cfv.digest(model),python=sys.version.split()[0],numpy=np.__version__,
        cpu_affinity=affinity,threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        gate_passed=bool(mean<.065 and p99<.120 and cm is not None and cm<.5),debug=audits,cold_debug=cold_debug)
    if gate and not result['gate_passed']:raise RuntimeError('LATENCY_GATE_FAILED '+json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--samples',type=int,default=256);p.add_argument('--cold',type=int,default=12)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--observe',action='store_true')
    a=p.parse_args();r=benchmark(a.zip,a.model,a.samples,a.cold,False)
    cfv.atomic_json(a.out,r);print(json.dumps(r),flush=True)
    if not a.observe and not r['gate_passed']:raise SystemExit('LATENCY_GATE_FAILED')

if __name__=='__main__':main()

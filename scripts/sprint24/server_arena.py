"""Paired arms, one isolated inference server, resumable complete judge matches."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from types import SimpleNamespace
import numpy as np
from alphanl.stage2 import cfv
from evening_d import Bundle,play_judge
from evening_production_eval import TOURNAMENT
from overnight_production import ONLINE
from overnight_data import TREE
from server_resources import pin

ARM_ORDER=('A','C','D','B')  # B is the first optional arm, never the baseline.


def profiles(iterations=None):
    base=dict(ONLINE,contract='server-arena-v1',rnr_eta=0.,context_prior=False,
              context_eta_max=.25,context_range_prior=False,size_risk_decay=False,
              server_arena_release=True,comparison_only=False)
    if iterations is not None:base['iterations']=iterations
    return {name:dict(base,guard_mode=mode,context_prior=name=='D',rnr_eta=.25 if name=='D' else 0.,
                      size_risk_decay=name=='D') for name,mode in
            (('A','legacy_full'),('B','slim'),('C','surgical'),('D','surgical'))}


def assert_production_cfv(nets):
    weights,meta=cfv.load_nets(nets)
    assert_qualified_manifest(meta)
    return meta


def assert_qualified_manifest(meta):
    if (meta.get('epochs')!=50 or meta.get('fixture') is not False or meta.get('smoke') is not False or
        meta.get('production') is not True or meta.get('offline_hud_prior') is not False or
        meta.get('tree')!=TREE or meta.get('river_samples',0)<8192 or meta.get('turn_samples',0)<4096):
        raise ValueError('FINAL_ARENA_REQUIRES_PRODUCTION_50_EPOCH_CFV: fixture/unqualified CFVs forbidden')
    for street in ('river','turn'):
        training=meta.get('validation',{}).get(street,{})
        if training.get('epochs_completed')!=50 or training.get('lr_schedule')!='cosine':
            raise ValueError('FINAL_ARENA_REQUIRES_50_COSINE_EPOCHS '+street)
        gate=training.get('CFV_HOLDOUT_GATE',{})
        if not gate.get('passed') or gate.get('samples',0)<2048:
            raise ValueError('FINAL_ARENA_REQUIRES_CFV_HOLDOUT_GATE '+street)
    return meta


def match_jobs(arm,seed_base=2026110600):
    return [dict(arm=arm,opponent=opponent,pair=pair,seat=seat,seed=seed_base+oi*1000+pair,hands=50)
            for oi,opponent in enumerate(TOURNAMENT) for pair in range(20) for seat in (0,1)]


def _digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _serve(release,core,requests,replies,ready):
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        affinity=pin([core])
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        bundle=Bundle(release['source'],release['model'])
        assert_qualified_manifest(bundle.meta['cfv'])
        ready.put(dict(pid=os.getpid(),affinity=affinity))
        while True:
            message=requests.get()
            if message is None:break
            slot,payload,draw_seed=message
            try:
                random.seed(draw_seed)
                started=time.perf_counter();body=bundle.respond(payload);engine_s=time.perf_counter()-started
                agent=bundle.agent
                trace=getattr(agent,'last_evening',{})
                diagnostics=dict(guard=copy.deepcopy(agent.last_guard),before=agent.guard_before.tolist(),after=agent.guard_after.tolist(),
                    search_audit=copy.deepcopy(agent.search_audit),trace={k:trace.get(k) for k in ('search','support','cfv_calls','reason','match_prior')},
                    debug=bundle.debug(),engine_s=engine_s)
                replies[slot].put(dict(body=body,diagnostics=diagnostics))
            except BaseException as exc:
                replies[slot].put(dict(error=type(exc).__name__+': '+str(exc)))
    except BaseException as exc:
        ready.put(dict(error=type(exc).__name__+': '+str(exc)))
        raise


_RPC=None
class RPCBundle:
    def __init__(self,release,requests,reply,slot):
        local=Bundle(release['source'],release['model'])
        self.runtime=local.runtime;self.meta=local.meta
        self.agent=SimpleNamespace(last_evening={},last_guard={},search_audit={})
        self.requests,self.reply,self.slot=requests,reply,slot
        self.seed=0;self.counter=0;self.engine_s=[];self.roundtrip_s=[];self._debug=''
    def reset(self,seed,seat):
        self.seed=seed*1000000+seat*100000;self.counter=0;self.engine_s=[];self.roundtrip_s=[]
    def respond(self,payload):
        self.counter+=1;started=time.perf_counter()
        self.requests.put((self.slot,payload,self.seed+self.counter))
        result=self.reply.get(timeout=120)
        self.roundtrip_s.append(time.perf_counter()-started)
        if 'error' in result:raise RuntimeError(result['error'])
        d=result['diagnostics'];self.engine_s.append(d['engine_s'])
        self.agent.last_guard=d['guard'];self.agent.guard_before=np.array(d['before']);self.agent.guard_after=np.array(d['after'])
        self.agent.last_evening=d['trace'];self.agent.search_audit=d['search_audit'];self._debug=d['debug']
        return result['body']
    def debug(self):return self._debug


def _init_worker(release,requests,replies,slots,cpus):
    global _RPC
    os.environ['CUDA_VISIBLE_DEVICES']=''
    pin(cpus)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    slot=slots.get(timeout=120)
    _RPC=RPCBundle(release,requests,replies[slot],slot)


def _arena_job(job):
    _RPC.reset(job['seed'],job['seat'])
    result=play_judge(_RPC,job['seed'],job['seat'],50,job['opponent'],keep_trace=True)
    if result['errors'] or not result['complete'] or result['hands']!=50:
        raise RuntimeError('arena requires complete 50-hand judge matches')
    result['turn_s']=_RPC.engine_s
    result['request_roundtrip_s']=_RPC.roundtrip_s
    for row in result['losing_hands']:row.update({k:job[k] for k in ('arm','opponent','pair','seat','seed')})
    result['worker_affinity']=sorted(os.sched_getaffinity(0));result['worker_pid']=os.getpid()
    return job,result


def _read_match(path,identity):
    row=json.loads(Path(path).read_text())
    if row['identity']!=identity or row['result_sha256']!=_digest_json(row['result']):
        raise ValueError('arena match checkpoint identity/hash mismatch: '+str(path))
    return row['result']


def run_arm(arm,release,nets,out,resources,workers=24,seed_base=2026110600):
    """Never accepts a fixture bypass; a restart only schedules missing matches."""
    qualification=assert_production_cfv(nets)
    with np.load(release['model'],allow_pickle=False) as blob:
        embedded=json.loads(blob['meta_json'].tobytes())['cfv']
    if embedded!=qualification:raise ValueError('arena embedded CFVs differ from trained checkpoint')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    binding=dict(source_sha256=cfv.digest(release['source']),model_sha256=cfv.digest(release['model']),nets_sha256=cfv.digest(nets))
    records={};pending=[]
    for job in match_jobs(arm,seed_base):
        key='%s_p%02d_s%d'%(job['opponent'],job['pair'],job['seat']);identity=dict(job,**binding)
        path=out/(key+'.json')
        if path.exists():records[key]=dict(job=job,result=_read_match(path,identity))
        else:pending.append(job)
    service_info=None
    if pending:
        ctx=mp.get_context('spawn');requests=ctx.Queue();replies=[ctx.Queue() for _ in range(workers)]
        slots=ctx.Queue()
        for slot in range(workers):slots.put(slot)
        ready=ctx.Queue();service=ctx.Process(target=_serve,args=(release,resources['inference_core'],requests,replies,ready))
        service.start()
        try:
            service_info=ready.get(timeout=120)
            if 'error' in service_info:raise RuntimeError(service_info['error'])
            with ctx.Pool(workers,initializer=_init_worker,initargs=(release,requests,replies,slots,resources['worker_cpus'])) as pool:
                iterator=pool.imap_unordered(_arena_job,pending,chunksize=1)
                for index in range(len(pending)):
                    job,result=iterator.next(600)
                    key='%s_p%02d_s%d'%(job['opponent'],job['pair'],job['seat'])
                    cfv.atomic_json(out/(key+'.json'),dict(identity=dict(job,**binding),result=result,result_sha256=_digest_json(result)))
                    records[key]=dict(job=job,result=result)
                    if (index+1)%10==0:print('SERVER_ARENA',arm,len(records),'/ 200',flush=True)
        finally:
            requests.put(None);service.join(timeout=5)
            if service.is_alive():service.terminate();service.join(timeout=5)
            if service.is_alive():service.kill();service.join(timeout=5)
            for queue in [requests,slots,ready]+replies:queue.close()
    result=summarize_records(arm,list(records.values()))
    result.update(binding=binding,qualification=qualification,inference_service=service_info,
                  resources=resources,resumed_matches=200-len(pending))
    cfv.atomic_json(out/'ARM_COMPLETE.json',result)
    return result


def interval(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():raise ValueError('independent duplicate pairs required')
    mean=float(values.mean());se=float(values.std(ddof=1)/len(values)**.5)
    return dict(mean=mean,ci95=[mean-1.96*se,mean+1.96*se],n_duplicate_pairs=len(values))


def summarize_records(arm,records):
    if len(records)!=200:raise ValueError('each arena arm needs all 200 matches')
    scores={};guard={};losses=[];engine=[];roundtrip=[]
    for row in records:
        j,r=row['job'],row['result'];key=(j['opponent'],j['pair'],j['seat'])
        if key in scores:raise ValueError('duplicate arena match identity')
        scores[key]=float(r['hero_net_bb']);engine.extend(r['turn_s']);roundtrip.extend(r.get('request_roundtrip_s',[]))
        losses.extend(r.get('losing_hands',[]))
        for name,value in r.get('guard_stats',{}).items():
            dst=guard.setdefault(name,dict(fires=0,mass_moved=0.));dst['fires']+=value['fires'];dst['mass_moved']+=value['mass_moved']
    cells={name:np.array([[scores[(name,p,seat)] for seat in (0,1)] for p in range(20)]) for name in TOURNAMENT}
    all_scores=np.concatenate(list(cells.values()))
    runtime=dict(mean_ms=float(np.mean(engine))*1000,p99_ms=float(np.percentile(engine,99))*1000,
                 max_ms=float(max(engine))*1000,n=len(engine))
    return dict(arm=arm,matches=200,duplicate_pairs=100,hands=10000,net_bb_pairs={k:v.tolist() for k,v in cells.items()},
                match_win_rate=float((all_scores>0).mean()),match_tie_rate=float((all_scores==0).mean()),
                mbb_per_hand=float(all_scores.mean()/50*1000),absolute_ev_ci95=interval(all_scores.mean(1)/50*1000),
                per_opponent={k:dict(match_win_rate=float((v>0).mean()),mbb_per_hand=float(v.mean()/50*1000),
                                    ev_ci95=interval(v.mean(1)/50*1000)) for k,v in cells.items()},
                guard_rules=guard,largest_20_losing_hands=sorted(losses,key=lambda r:r['net_bb'])[:20],
                runtime=runtime,request_roundtrip_ms=dict(mean=float(np.mean(roundtrip))*1000,p99=float(np.percentile(roundtrip,99))*1000) if roundtrip else None)


def compare_arms(arms):
    if 'A' not in arms:raise ValueError('paired ranking requires the A baseline')
    result=copy.deepcopy(arms);base=arms['A']['net_bb_pairs']
    for name,row in result.items():
        paired=[];wins=[]
        for opponent in TOURNAMENT:
            a=np.array(base[opponent]);b=np.array(row['net_bb_pairs'][opponent])
            delta=(b-a).mean(1)/50*1000
            paired.extend(delta);wins.extend(((b>0).astype(float)-(a>0).astype(float)).mean(1))
            row['per_opponent'][opponent]['paired_ev_vs_A']=interval(delta)
        row['paired_ev_vs_A']=interval(paired);row['paired_win_rate_vs_A']=interval(wins)
        row['promoted']=bool(name!='A' and row['paired_ev_vs_A']['ci95'][0]>0.)
    return result


def rank_arms(arms,exclusions):
    candidates=[name for name in arms if name not in exclusions]
    ordered=sorted(candidates,key=lambda n:(arms[n]['paired_ev_vs_A']['ci95'][0],arms[n]['match_win_rate'],n),reverse=True)
    if not any(arms[n]['promoted'] for n in candidates) and 'A' in ordered:
        ordered.remove('A');ordered.insert(0,'A')
    return ordered[:3]

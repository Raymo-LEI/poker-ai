"""Audited historical state reuse; independent offline labels and frozen holdout.

Historical Y is never grandfathered into an expanded tree. Verified public X
can be relabelled, and verified same-topology weights are transfer initialization.
Neither operation qualifies historical weights for the final arena.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from alphanl.stage2 import cfv, deep_solver_v2 as solver
from evening_parallel import jobs
from evening_pipeline import _snap, _train
from overnight_data import TREE, label, mixed_dataset, holdout_gate

FLOORS={'river':80000,'turn':50000}
FULL_PRECISION_COUNTS={'river':108800,'turn':72192}
NEW_SAMPLES={'river':8192,'turn':4096}
OLD_COMMIT='5cbf1db67d6b5f34c855c56b7aeddfbc7b8bb001'
OLD_ROOT='/root/autodl-tmp/sprint24/evening_production/20261104T142449Z_5cbf1db6'
OLD_SOURCE='/root/autodl-tmp/poker-evening-anchor-fix-5cbf1db'
RECOVERY='cfv-state-relabel-transfer-v1'


class RecoveryStop(RuntimeError):
    pass


def schema_hash():
    contract=dict(schema=cfv.SCHEMA,features=cfv.FEATURE_DIM,buckets=cfv.BUCKETS,
                  tail=cfv.TAIL_NAMES,deck=cfv.DECK,units='zero_sum_root_bb',
                  pot='x[119] * 200 = pot / big_blind',range='public-static32-uniform-within-bucket')
    return hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()


def function_hash(source,name):
    node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    if name=='expand_range' and len(node.body)==4:
        # The current solver has a full-combo fast path. Historical rows
        # have 32 entries and >1,000 legal combos, so that branch is unreachable.
        # Ignore only the precisely identified guard, not arbitrary new code.
        first=node.body[0]
        guard=ast.parse('np.asarray(ranges).shape == (len(bins),)',mode='eval').body
        if isinstance(first,ast.If) and not first.orelse and ast.dump(first.test)==ast.dump(guard):
            node.body=node.body[1:]
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def old_artifact(path,sources,kind):
    """Explicit audited import; normal cfv.load_artifact remains fail-closed."""
    path=Path(path);meta=json.loads(Path(str(path)+'.manifest.json').read_text())
    if (meta.get('schema')!=cfv.SCHEMA or meta.get('units')!='zero_sum_root_bb' or
        meta.get('complete') is not True or meta.get('kind')!=kind or
        meta.get('sha256')!=cfv.digest(path) or meta.get('sources')!=sources):
        raise ValueError('HISTORICAL_ARTIFACT_INCOMPATIBLE '+str(path))
    with np.load(path,allow_pickle=False) as blob:arrays={k:blob[k].copy() for k in blob.files}
    if meta.get('arrays_sha256')!=cfv.array_digest(arrays):raise ValueError('historical array digest mismatch')
    if kind=='dataset':
        x,y=arrays['x'],arrays['y']
        if x.shape!=(meta['samples'],136) or y.shape!=(len(x),32) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('historical dataset shape/nonfinite values')
    return arrays,meta


def inspect_history(root,checkout,out):
    root,checkout,out=map(Path,(root,checkout,out));out.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'FINAL_REPORT.json').read_text())
    if report.get('status')!='COMPLETE':raise ValueError('historical run is not COMPLETE')
    paths={'cfv.py':'src/alphanl/stage2/cfv.py','deep_solver_v2.py':'src/alphanl/stage2/deep_solver_v2.py',
           'evening_pipeline.py':'scripts/sprint24/evening_pipeline.py'}
    sources={};text={}
    for name,path in paths.items():
        original=subprocess.check_output(['git','-C',str(checkout),'show',OLD_COMMIT+':'+path])
        if (checkout/path).read_bytes()!=original:raise ValueError('historical source checkout changed: '+path)
        sources[name]=hashlib.sha256(original).hexdigest();text[name]=original.decode()
    # Bind the normalization contract to exact bytes, and evaluator/public bucket
    # semantics to ASTs. A schema string alone cannot attest evaluator correctness.
    if sources['cfv.py']!=cfv.digest(Path(cfv.__file__)):
        raise ValueError('historical feature/pot normalization implementation differs')
    current=Path(solver.__file__).read_text()
    evaluators={}
    for name in ('pack_many','_public','expand_range','bucket_values'):
        a,b=function_hash(text['deep_solver_v2.py'],name),function_hash(current,name)
        if a!=b:raise ValueError('historical evaluator/bucket contract differs: '+name)
        evaluators[name]=a
    evaluator_path='src/alphanl/stage2/portable_evaluator.py'
    original=subprocess.check_output(['git','-C',str(checkout),'show',OLD_COMMIT+':'+evaluator_path])
    if original!=Path(solver.__file__).with_name('portable_evaluator.py').read_bytes():
        raise ValueError('historical portable evaluator differs from repaired evaluator')
    result=dict(recovery=RECOVERY,old_commit=OLD_COMMIT,root=str(root),sources=sources,schema=cfv.SCHEMA,
                completed_report_sha256=cfv.digest(root/'FINAL_REPORT.json'),
                schema_sha256=schema_hash(),evaluator=dict(version='exact-vector-pack-v1-post-fix',
                ast_sha256=evaluators,portable_sha256=hashlib.sha256(original).hexdigest(),
                range_contract='32 buckets only; unreachable full-combo input branch excluded from expand_range hash'),
                pot_normalization='identical cfv.py bytes; x[119]*200 = pot/BB; targets zero_sum_root_bb',datasets={})
    for name,street in (('river',3),('turn',2)):
        data,meta=old_artifact(root/'b'/(name+'_ds.npz'),sources,'dataset')
        if meta.get('street')!=street:raise ValueError('historical street mismatch')
        direct=(meta.get('tree')==TREE and name=='river')
        reasons=[]
        if meta.get('tree')!=TREE:reasons.append('old labels lack bilateral-checkraise-pot-jam-v1; old label_spot uses check_raise=False default')
        if name=='turn':reasons.append('turn labels depend on historical river teacher; relabel against this run river CFV')
        result['datasets'][name]=dict(path=str(root/'b'/(name+'_ds.npz')),samples=len(data['x']),sha256=meta['sha256'],
            direct_labels_compatible=direct,incompatibilities=reasons,action='reuse_labels' if direct else 'reuse_public_states_and_relabel',
            original_labels_per_second=meta['samples']/meta['active_compute_s'],original_iterations=meta['iterations'],
            original_chance_samples=meta['chance_samples'])
        # Old validation is never new validation and never training augmentation.
        _,valid_meta=old_artifact(root/'b'/(name+'_holdout.npz'),sources,'dataset')
        result['datasets'][name]['historical_holdout_sha256']=valid_meta['sha256']
    weights,meta=old_artifact(root/'b/cfv_nets.npz',sources,'nets');cfv.validate_weights(weights)
    if meta.get('epochs',0)<1:raise ValueError('historical CFV checkpoint is untrained')
    result['warm_start']=dict(path=str(root/'b/cfv_nets.npz'),sha256=meta['sha256'],epochs=meta['epochs'],
                             compatible=True,usage='transfer initialization only; 50 new cosine epochs and fresh 0.12 gate required')
    cfv.atomic_json(out/'COMPATIBILITY.json',result)
    return result


def historical_data(audit,name,holdout=False):
    p=Path(audit['datasets'][name]['path'])
    if holdout:p=p.with_name(name+'_holdout.npz')
    arrays,meta=old_artifact(p,audit['sources'],'dataset')
    key='historical_holdout_sha256' if holdout else 'sha256'
    if meta['sha256']!=audit['datasets'][name][key]:raise ValueError('historical CFV artifact changed after audit: '+str(p))
    return arrays


def board_ids(x):
    bits=np.asarray(x)[:,64:116]
    if not np.isin(bits,(0.,1.)).all():raise ValueError('invalid public board bits')
    return (bits.astype(np.uint64)*(np.uint64(1)<<np.arange(52,dtype=np.uint64))).sum(1,dtype=np.uint64)


def require_floor(counts):
    missing={k:dict(actual=int(counts.get(k,0)),minimum=v) for k,v in FLOORS.items() if counts.get(k,0)<v}
    if missing:raise RecoveryStop('INSUFFICIENT_CFV_SAMPLES '+json.dumps(missing,sort_keys=True))


def assert_disjoint(train,holdout,forbidden=()):
    forbidden=np.concatenate([board_ids(train),np.asarray(forbidden,dtype=np.uint64)])
    overlap=np.intersect1d(forbidden,board_ids(holdout))
    if len(overlap):raise ValueError('CFV_HOLDOUT_LEAKAGE: public boards overlap '+str(len(overlap)))
    return dict(board_overlap=0,train_rows=len(train),holdout_rows=len(holdout),
                method='canonical 52-bit public-board exclusion, all historical train/holdout + new train',
                train_sha256=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
                holdout_sha256=hashlib.sha256(np.ascontiguousarray(holdout).tobytes()).hexdigest())


def decode_state(row):
    """Invert public v3 X only; real private hands are not used in public solving."""
    row=np.asarray(row);street=int(round(float(row[121])*3))
    board=[cfv.DECK[i] for i in np.flatnonzero(row[64:116])]
    if street not in (2,3) or len(board)!=street+2:raise ValueError('bad reusable board/street')
    pool=[c for c in cfv.DECK if c not in board]
    pot=float(row[119])*20000;call=float(row[120])*pot
    snap=_snap(pool[:2],board,pot,call,float(row[116])*20000,street,float(row[117])*20000)
    first=0 if row[125]>.5 else 1
    snap.first_player=0 if call else first;snap.me_seat=first;snap.postflop_first_seat=0
    rh,ro=cfv.normalize_range(row[:32]),cfv.normalize_range(row[32:64])
    rebuilt=cfv.spot_features(board,rh,ro,pot,call,snap.me_stack,100.,street,snap.opp_stack,first)
    if not np.allclose(rebuilt,row,rtol=2e-6,atol=2e-6):raise ValueError('historical feature roundtrip mismatch')
    return snap,rh,ro


_LABEL_NET=None

def init_labels(path):
    global _LABEL_NET
    _LABEL_NET=None if path is None else cfv.load_artifact(path,'river_net',min_epochs=1)[0]


def relabel_worker(payload):
    index,x,seeds,it,chance=payload;started=time.process_time();wall=time.monotonic()
    rows=[label(*decode_state(row),_LABEL_NET,it,chance,int(seed)) for row,seed in zip(x,seeds)]
    xx,yy=map(np.stack,zip(*rows))
    return index,xx,yy,time.process_time()-started,wall,time.monotonic()


def relabel(x,seeds,it,chance,workers,nets=None):
    grain=min(64,max(1,int(math.ceil(len(x)/workers))))
    payloads=[(i,x[start:start+grain],seeds[start:start+grain],it,chance) for i,start in enumerate(range(0,len(x),grain))]
    started=time.monotonic()
    parts=sorted(jobs(relabel_worker,payloads,len(payloads),workers,init_labels,(None if nets is None else str(nets),),
                      timeout=600,label='REUSE_RELABEL'))
    xx=np.concatenate([p[1] for p in parts]);yy=np.concatenate([p[2] for p in parts])
    wall=time.monotonic()-started;cpu=sum(p[3] for p in parts)
    return xx,yy,dict(wall_s=wall,cpu_s=cpu,labels_per_second=len(xx)/wall,labels_per_cpu_second=len(xx)/max(cpu,1e-9))


def benchmark_labels(audit,out,workers,full_precision=False):
    """Choose offline-only depth using train states, never the final holdout.

    Reference 64/32 x4 remains the quality anchor. Online is untouched 8/8/16.
    Report both measured CPU work and wall throughput (pool startup included).
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True);result={}
    # Calibrate turn labels with a real trained river head, not the latency pilot.
    warm,wm=old_artifact(audit['warm_start']['path'],audit['sources'],'nets')
    reference_net=out/'label_reference_river.npz'
    cfv.publish(reference_net,warm,wm['river_samples'],wm['epochs'],'river_net',
                dict(tree='historical-transfer-only',production=False,provenance=audit['warm_start']))
    previous=Path('/root/autodl-tmp/sprint24/overnight_server/408a2ff_production/OVERNIGHT_FINAL_REPORT.json')
    previous_rates=json.loads(previous.read_text()).get('plan',{}).get('rates',{}) if previous.exists() else {}
    for name in ('river','turn'):
        data=historical_data(audit,name);n=min(max(64,workers*32),len(data['x']))
        # Fixed stratified indices: no final-holdout/model-selection leakage.
        ids=np.linspace(0,len(data['x'])-1,n,dtype=int);x=data['x'][ids];seeds=data['seeds'][ids]
        net=None if name=='river' else reference_net;reference=64 if name=='river' else 32
        _,ref,ref_rate=relabel(x,seeds,reference,4,workers,net)
        options=[dict(iterations=reference,chance_samples=4,quality_passed=True,pot_mae_vs_reference=0.,**ref_rate)]
        alternatives=() if full_precision else (16,32) if name=='river' else (8,16)
        for it in alternatives:
            _,y,rate=relabel(x,seeds,it,4,workers,net)
            errors=np.abs(y-ref)/np.maximum(x[:,119:120]*200.,1.)
            options.append(dict(iterations=it,chance_samples=4,quality_passed=bool(errors.mean()<=.02 and np.quantile(errors.mean(1),.99)<=.08),
                pot_mae_vs_reference=float(errors.mean()),p99_row_pot_mae=float(np.quantile(errors.mean(1),.99)),**rate))
        selected=min((r for r in options if r['quality_passed']),key=lambda r:r['cpu_s'])
        result[name]=dict(reference=options[0],candidates=options,selected=selected,benchmark_rows=n,
                          old_v23_labels_per_second=audit['datasets'][name]['original_labels_per_second'],
                          previous_collapsed_plan_conservative_labels_per_second=(1/previous_rates[name] if name in previous_rates else None),
                          previous_collapsed_rate_note='1 / saved plan seconds per sample; includes its 1.35 safety factor and PBS collection',
                          reference_teacher=str(reference_net) if name=='turn' else None,
                          comparison='same repaired expanded-tree public X; offline HUD-free; online depth unchanged')
        cfv.atomic_json(out/'LABEL_THROUGHPUT.json',result)
    return result


def choose_plan(audit,benchmark,rates,hours,elapsed,match_s,bench_s,workers,full_precision=False):
    """Price all labels/epochs/three generations/serial arena; never fit below floors."""
    if any(not math.isfinite(float(x)) or x<=0 for x in rates.values()):raise ValueError('invalid measured rates')
    counts={name:audit['datasets'][name]['samples']+NEW_SAMPLES[name] for name in FLOORS}
    require_floor(counts)
    if full_precision:
        if any(counts[name]<n for name,n in FULL_PRECISION_COUNTS.items()):
            raise ValueError('Full precision requires the compatible V23 public-state reservoir')
        for name,iterations in (('river',64),('turn',32)):
            selected=benchmark[name]['selected']
            if (selected['iterations'],selected['chance_samples'])!=(iterations,4):
                raise ValueError('Full precision requires offline 64/32 iterations with four chance samples')
    attempts=[]
    # Keep the complete recovered reservoir whenever possible. Drop optional B
    # before downsampling states. Fail closed if even the minimum cannot fit.
    targets=(FULL_PRECISION_COUNTS,) if full_precision else (counts,FLOORS)
    arm_options=(['A','C','D','B'],) if full_precision else (['A','C','D','B'],['A','C','D'])
    for target in targets:
        for arms in arm_options:
            reused={name:target[name]-NEW_SAMPLES[name] for name in FLOORS}
            # Board-disjoint validation is rejection-sampled. Price its pool
            # above 2048 (especially turn, with many historical boards).
            holdout_pool={name:holdout_pool_size(audit,name) for name in FLOORS}
            labels=sum(reused[name]/benchmark[name]['selected']['labels_per_second']+
                       (NEW_SAMPLES[name]+holdout_pool[name])*rates[name] for name in FLOORS)
            training=50*sum(target.values())*rates['cfv_epoch']
            # Same 3-generation pricing as the original controller, including
            # its disjoint tuning/acceptance sets and preflop anchors.
            generations=3*(4096*(rates['rebel']+12*rates['policy_epoch'])+2048*(5*rates['rebel']+rates['anchor']))
            arena=200*len(arms)*match_s;bench=(len(arms)+3+(6 if full_precision else 0))*bench_s
            reserve=1800+hours*3600*.08
            predicted=elapsed+1.25*(labels+training+generations+arena+bench)+reserve
            attempts.append(dict(counts=dict(target),arms=arms,predicted_seconds=predicted))
            if predicted>hours*3600 and not full_precision:continue
            return dict(river_n=target['river'],turn_n=target['turn'],reused_rows=reused,new_rows=NEW_SAMPLES,
                holdout_n=2048,net_epochs=50,policy_epochs=12,rebel_n=4096,workers=workers,hours=hours,
                arms=arms,dropped_arms=[] if 'B' in arms else ['B'],generations=['g3','g4','g5'],
                rates=rates,predicted_seconds=predicted,elapsed_at_plan=elapsed,fit_attempts=attempts,
                estimated_label_seconds=labels,estimated_train_seconds=training,
                estimated_serial_arena_seconds=arena,estimated_benchmark_seconds=bench,
                offline={name:{k:benchmark[name]['selected'][k] for k in ('iterations','chance_samples')} for name in FLOORS},
                recovery=RECOVERY,minimum_samples=FLOORS,estimated_holdout_pool=holdout_pool,
                full_precision=full_precision,forecast_exceeds_target=predicted>hours*3600)
    raise RecoveryStop('INSUFFICIENT_LABEL_BUDGET: cannot fit 80000 river / 50000 turn and required arena; '+json.dumps(attempts))


def holdout_pool_size(audit,name):
    # Conservative union bound on occupied public boards, with extra allowance
    # for rejecting a complete PBS rollout rather than individual decisions.
    from math import comb
    total=comb(52,5 if name=='river' else 4)
    occupancy=min(.8,(audit['datasets'][name]['samples']+NEW_SAMPLES[name]+2048)/total)
    return int(math.ceil(2048/(1-occupancy)**2/32))*32


def cached_mixed(path,source,street,n,seed,workers,config,nets):
    path=Path(path)
    identity=dict(source_sha256=cfv.digest(source),street=street,n=n,seed=seed,config=config,
                  teacher=None if nets is None else cfv.digest(nets))
    receipt=Path(str(path)+'.identity.json')
    if receipt.exists():
        if json.loads(receipt.read_text())!=identity:raise ValueError('mixed resume inputs changed')
        arrays,meta=cfv.load_artifact(path,'dataset',min_samples=n)
        meta['cluster_ids']=arrays['cluster_ids'];return arrays['x'],arrays['y'],meta
    result=mixed_dataset(path,source,street,n,seed,workers,config['iterations'],config['chance_samples'],nets)
    cfv.atomic_json(receipt,identity);return result


def recovered_dataset(path,audit,name,count,workers,config,nets):
    path=Path(path);original=historical_data(audit,name)
    if count>len(original['x']):raise ValueError('insufficient historical reservoir')
    indices=np.linspace(0,len(original['x'])-1,count,dtype=int)
    identity=dict(original_sha256=audit['datasets'][name]['sha256'],count=count,config=config,
                  teacher=None if nets is None else cfv.digest(nets),schema_sha256=schema_hash())
    receipt=Path(str(path)+'.identity.json');chunks=path.parent/(name+'_relabel_chunks');chunks.mkdir(exist_ok=True)
    if receipt.exists():
        if json.loads(receipt.read_text())!=identity:raise ValueError('relabel resume inputs changed')
        arrays,meta=cfv.load_artifact(path,'dataset',min_samples=count);return arrays,meta
    rows=[];timing=[]
    for start in range(0,count,4096):
        chunk=chunks/('%06d.npz'%start);ids=indices[start:start+4096]
        if chunk.exists():
            a,m=cfv.load_artifact(chunk,'dataset',min_samples=len(ids))
            if m.get('relabel_identity')!=identity or m.get('offset')!=start:raise ValueError('relabel chunk identity changed')
        else:
            if audit['datasets'][name]['direct_labels_compatible']:
                x,y=original['x'][ids],original['y'][ids];t=dict(wall_s=0.,cpu_s=0.)
            else:
                x,y,t=relabel(original['x'][ids],original['seeds'][ids],config['iterations'],config['chance_samples'],workers,nets)
            a=dict(x=x,y=y,cluster_ids=original['seeds'][ids].astype(np.int64))
            m=cfv.publish(chunk,a,len(ids),0,'dataset',dict(relabel_identity=identity,offset=start,timing=t,tree=TREE))
        rows.append(a);timing.append(m['timing'])
    arrays={k:np.concatenate([r[k] for r in rows]) for k in ('x','y','cluster_ids')}
    meta=cfv.publish(path,arrays,count,0,'dataset',dict(tree=TREE,relabel_identity=identity,
        wall_s=sum(t['wall_s'] for t in timing),cpu_s=sum(t['cpu_s'] for t in timing),provenance=audit['datasets'][name]))
    cfv.atomic_json(receipt,identity);return arrays,meta


def fresh_holdout(folder,source,street,name,n,workers,config,nets,train,forbidden):
    # Reject whole trajectories (clusters), not individual decisions, when ANY
    # board in that rollout overlaps old/new training or historical validation.
    banned=set(map(int,np.concatenate([board_ids(train),forbidden])));selected=[];groups=[];seen=set()
    seed=(9300000000000 if street==3 else 9400000000000)
    # Use smaller refill pools; checkpoint them individually for crash recovery.
    for attempt in range(32):
        remaining=n-len(selected)
        pool_n=int(math.ceil(max(256,remaining*1.25)/32))*32
        x,y,meta=cached_mixed(folder/(name+'_holdout_pool_%02d.npz'%attempt),source,street,pool_n,seed+attempt*100000000,
                              workers,config,nets)
        ids=board_ids(x);clusters=meta['cluster_ids'];bad=set(map(int,clusters[[int(b) in banned for b in ids]]))
        for cluster in np.unique(clusters):
            c=int(cluster)
            if c in bad or c in seen:continue
            seen.add(c);ix=np.flatnonzero(clusters==cluster)
            selected.extend((x[i],y[i]) for i in ix);groups.extend([c]*len(ix))
            if len(selected)>=n:
                xx,yy=map(np.stack,zip(*selected[:n]));gg=np.asarray(groups[:n],dtype=np.int64)
                check=assert_disjoint(train,xx,forbidden)
                cfv.publish(folder/(name+'_holdout.npz'),dict(x=xx,y=yy,cluster_ids=gg),n,0,'dataset',
                            dict(tree=TREE,leakage_check=check,seed_namespace=seed))
                return xx,yy,gg,check
    raise RecoveryStop('FRESH_DISJOINT_HOLDOUT_EXHAUSTED '+name)


def phase_b(source,out,plan,workers,device,audit,deadline=None):
    """Resume current-source chunks/heads; never delete a failed fit or lower .12."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if plan['net_epochs']!=50 or plan['holdout_n']!=2048:raise ValueError('recovery requires 50 epochs / 2048 holdout')
    require_floor({name:plan[name+'_n'] for name in FLOORS})
    # Check both reservoirs before starting ANY optimizer.
    require_floor({name:min(plan['reused_rows'][name],audit['datasets'][name]['samples'])+plan['new_rows'][name] for name in FLOORS})
    warm,_=old_artifact(audit['warm_start']['path'],audit['sources'],'nets');cfv.validate_weights(warm)
    if cfv.digest(audit['warm_start']['path'])!=audit['warm_start']['sha256']:raise ValueError('historical warm-start changed')
    stats={};weights={}
    for name,street,prefix in (('river',3,'cfv_river.'),('turn',2,'cfv_turn.')):
        if deadline is not None and time.monotonic()>=deadline:
            if plan.get('full_precision'):raise TimeoutError('target deadline reached before '+name)
            raise RecoveryStop('INSUFFICIENT_LABEL_BUDGET: target deadline reached before '+name)
        net=None if street==3 else out/'cfv_river.npz';config=plan['offline'][name]
        receipt=out/(name+'_TRAINED.json');dest=out/('cfv_river.npz' if street==3 else 'cfv_turn.npz')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['plan']!=plan or old['audit']!=audit:raise ValueError('CFV fit resume identity changed')
            w,_=cfv.load_artifact(dest,'river_net' if street==3 else 'turn_net',min_epochs=50)
            weights.update(w);stats[name]=old['stats'];continue
        rejected=out/(name+'_REJECTED.json')
        if rejected.exists():raise RecoveryStop('CFV_HOLDOUT_GATE_FAILED: persisted rejection; identical retry forbidden '+str(rejected))
        reused,rm=recovered_dataset(out/(name+'_reused.npz'),audit,name,plan['reused_rows'][name],workers,config,net)
        nx,ny,nm=cached_mixed(out/(name+'_new.npz'),source,street,plan['new_rows'][name],
                             9100000000000+street*1000000000,workers,config,net)
        x=np.concatenate([reused['x'],nx]);y=np.concatenate([reused['y'],ny])
        if len(x)<FLOORS[name]:raise RecoveryStop('INSUFFICIENT_CFV_SAMPLES '+name)
        original=historical_data(audit,name);old_valid=historical_data(audit,name,True)
        forbidden=np.concatenate([board_ids(original['x']),board_ids(old_valid['x'])])
        vx,vy,clusters,leak=fresh_holdout(out,source,street,name,2048,workers,config,net,x,forbidden)
        cfv.publish(out/(name+'_ds.npz'),dict(x=x,y=y),len(x),0,'dataset',dict(tree=TREE,
                    reused_states=len(reused['x']),new_on_policy=nm['on_policy_n'],new_dirichlet=nm['dirichlet_n'],
                    source_sha256=cfv.digest(source),leakage_check=leak))
        if deadline is not None and plan.get('full_precision') and time.monotonic()>=deadline:
            raise TimeoutError('target deadline reached before fitting '+name)
        if deadline is not None and not plan.get('full_precision') and time.monotonic()+50*len(x)*plan['rates']['cfv_epoch']>=deadline:
            raise RecoveryStop('INSUFFICIENT_LABEL_BUDGET: labels saved but 50 epochs cannot fit for '+name)
        w,s=_train(x,y,prefix,50,street,device,(vx,vy),cosine=True,initial_weights=warm)
        gate=holdout_gate(w,prefix,y,(vx,vy),clusters);s['CFV_HOLDOUT_GATE']=gate
        s.update(samples=len(x),warm_start=audit['warm_start'],leakage_check=leak,offline_label_config=config,
                 reuse_labels_per_second=len(reused['x'])/max(rm['wall_s'],1e-9))
        # Persist diagnostics and rejected weights BEFORE raising. A restart can
        # never turn a rejected fit into a qualified checkpoint.
        cfv.publish(dest if gate['passed'] else out/(name+'_REJECTED.npz'),w,len(x),50,
                    ('river_net' if street==3 else 'turn_net') if gate['passed'] else 'rejected_net',
                    dict(tree=TREE,validation=s,production=False))
        cfv.atomic_json(out/(name+'_VALIDATION.json'),s)
        if not gate['passed']:
            cfv.atomic_json(rejected,s);raise RecoveryStop('CFV_HOLDOUT_GATE_FAILED '+name+' '+str(gate))
        cfv.atomic_json(receipt,dict(plan=plan,audit=audit,stats=s));weights.update(w);stats[name]=s
    cfv.publish(out/'cfv_nets.npz',weights,plan['river_n']+plan['turn_n'],50,'nets',
        dict(tree=TREE,river_samples=plan['river_n'],turn_samples=plan['turn_n'],validation=stats,
             smoke=False,fixture=False,production=True,offline_hud_prior=False,recovery=RECOVERY,
             minimum_samples=FLOORS,compatibility_audit=audit,schema_sha256=schema_hash(),
             mixture='audited historical public states relabelled + new 50/50 PBS/Dirichlet; historical states are synthetic'))
    return stats

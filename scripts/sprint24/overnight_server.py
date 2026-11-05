"""Resumable Xeon/5090 production: baseline measurement -> training -> paired arena -> V26-28."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv
from evening_production import git,find_champion,EXPECTED_CHAMPION_SHA256
from overnight_production import select_handoff,plan_counts,group_stop,ONLINE,OFFLINE
from server_resources import core_plan,pin,SERVER_TARGET_HOURS,SERVER_WATCHDOG_SECONDS
from server_checkpoint import Checkpoints
from server_arena import profiles,ARM_ORDER,assert_production_cfv,run_arm,compare_arms,rank_arms,SERVER_ITERATIONS
from server_benchmark import gates,full_power_gates
from server_pr85 import PR85_SHA,build as build_pr85
from server_packaging import reserve,package_one



def source_lock(expected):
    if git('rev-parse','HEAD')!=expected or git('status','--porcelain'):
        raise RuntimeError('SOURCE_LOCK_FAILED: expected clean immutable commit '+expected)


def internal(checkpoint,nets,folder,profile,label):
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    from alphanl.stage2.deployment_guard import V102_R7_GUARD
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=folder/'V999_SOURCE.zip';model=folder/'V999_MODEL.npz';release='V999_SERVER_'+label.upper()
    if not source.exists() and not model.exists():
        export_botzone_weights(checkpoint,model,cfv_nets=nets,release_id=release,online_profile=profile,deployment_guard=V102_R7_GUARD)
        pack_botzone_zip(source,release_id=release)
    if not source.exists() or not model.exists():raise ValueError('partial internal package')
    return dict(source=str(source),model=str(model),release=release,source_sha256=cfv.digest(source),model_sha256=cfv.digest(model))


def measure(release,out,corpus,resources,smoke=False,collect=False):
    command=[sys.executable,str(REPO/'scripts/sprint24/server_benchmark.py'),'--zip',release['source'],'--model',release['model'],
             '--out',str(out),'--corpus',str(corpus),'--core',str(resources['inference_core'])]
    if collect:command+=['--collect']
    if smoke:command+=['--diagnostic','--samples','16','--cold','1']
    try:
        p=subprocess.run(command,timeout=1800)
        if p.returncode not in (0,2):raise RuntimeError('benchmark process exit '+str(p.returncode))
        result=json.loads(Path(out).read_text())
    except (Exception,subprocess.TimeoutExpired) as exc:
        result=dict(gate=dict(hard_passed=False,soft_mean_passed=False),error=type(exc).__name__+': '+str(exc))
        cfv.atomic_json(out,result)
    return result


def choose_plan(rates,hours,match_seconds,benchmark_seconds,elapsed,workers):
    """Inference is SERIAL on its reserved core, never divided by 24 workers."""
    attempts=[]
    for arms in (list(ARM_ORDER),['A','C','D']):
        arena=len(arms)*200*match_seconds
        # Full per-arm timing plus exact released-byte gates and package I/O.
        evaluation=arena+(len(arms)+3)*benchmark_seconds+180.
        try:plan=plan_counts(rates,hours,evaluation,elapsed,workers)
        except RuntimeError as exc:
            attempts.append(dict(arms=arms,reason=str(exc)));continue
        plan.update(arms=arms,dropped_arms=[] if 'B' in arms else ['B'],arena_matches=200*len(arms),
                    estimated_serial_arena_seconds=arena,estimated_benchmark_seconds=(len(arms)+3)*benchmark_seconds,
                    fit_attempts=attempts,inference_parallelism=1)
        return plan
    raise RuntimeError('production training/arena floors cannot fit even after dropping B: '+str(attempts))



def arena_phase(releases,nets,corpus,resources,workers,stage,update,gate_function=gates):
    arenas={};exclusions={};latencies={}
    for arm in ARM_ORDER:
        if arm not in releases:continue
        latency=stage('trained_latency_'+arm,lambda folder,arm=arm:measure(releases[arm],folder/'benchmark.json',corpus,resources))
        latencies[arm]=latency
        if not latency['gate']['hard_passed']:exclusions[arm]='hard trained-runtime latency gate'
        try:
            result=stage('arena_'+arm,lambda folder,arm=arm:run_arm(arm,releases[arm],nets,folder,resources,workers),clean=False)
        except Exception as exc:
            exclusions[arm]='arena error: '+type(exc).__name__+': '+str(exc)
            update(arenas,exclusions,latencies);continue
        arenas[arm]=result
        runtime=result['runtime']
        parameters=(runtime['mean_ms'],runtime['p99_ms'],runtime['max_ms'],latency.get('cold_start_s',float('inf')))
        combined=(gate_function(*parameters,max(runtime.get('rss_mb',float('inf')),latency.get('rss_mb',float('inf'))))
                  if gate_function is full_power_gates else gate_function(*parameters))
        if not combined['hard_passed']:exclusions[arm]='hard arena inference latency gate'
        result['arena_latency_gate']=combined
        update(arenas,exclusions,latencies)
    return arenas,exclusions,latencies

def _execute(a):
    import copy
    import torch
    from alphanl.stage2.train import load_checkpoint,save_checkpoint
    from overnight_data import phase_b,mixed_dataset,collect_teacher,TREE
    from overnight_distill import bind_champion,expand,fit,one_generation
    from evening_d import Bundle,play_judge
    from evening_pipeline import _train
    import numpy as np
    source_lock(a.expected_commit)
    out=a.out.resolve()
    if out.is_relative_to(REPO) and not a.smoke:raise ValueError('production output must be outside the locked checkout')
    out.mkdir(parents=True,exist_ok=True)
    execution=(out/'.execution.lock').open('a')
    fcntl.flock(execution.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    resource_path=out/'CPU_RESERVATION.json'
    if a.resume and resource_path.exists():resources=json.loads(resource_path.read_text())
    else:resources=core_plan(a.inference_core);cfv.atomic_json(resource_path,resources)
    if a.inference_core is not None and resources['inference_core']!=a.inference_core:
        raise ValueError('resume inference core differs from the reservation')
    pin(resources['worker_cpus'])
    print('CPU_RESERVATION',json.dumps({k:resources[k] for k in ('inference_core','inference_siblings','governor','cpu_quota')}),flush=True)
    workers=min(24,len(resources['worker_cpus'])) if a.workers is None else a.workers
    if not 1<=workers<=min(24,len(resources['worker_cpus'])):raise ValueError('workers must fit CPUs excluding inference core/siblings')
    champion=find_champion(a.champion)
    if cfv.digest(champion)!=EXPECTED_CHAMPION_SHA256:raise ValueError('champion SHA256 mismatch')
    previous_checkpoint=out/'CHECKPOINTS.json'
    if a.resume and previous_checkpoint.exists():
        locked=json.loads(previous_checkpoint.read_text())['identity']
        handoff=locked['handoff'];source=Path(handoff['checkpoint'])
        if cfv.digest(source)!=locked['handoff_sha256']:raise ValueError('locked warm-start checkpoint changed')
    else:source,handoff=select_handoff(a.evening_root,champion,a.fresh)
    identity=dict(commit=a.expected_commit,champion=str(champion),champion_sha256=cfv.digest(champion),
                  handoff_sha256=cfv.digest(source),handoff=handoff,workers=workers,resources=resources,
                  hours=a.hours,full_precision=a.full_precision,smoke=a.smoke,deliver=str(a.deliver.resolve()),server_profile='xeon-four-arm-v1',
                  runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__))
    if a.cfv_data_recovery:
        identity['cfv_recovery']=dict(root=str(a.reuse_evening_run.resolve()),source=str(a.reuse_source.resolve()))
    store=Checkpoints(out,identity,a.resume)
    summary=dict(status='RUNNING',identity=identity,gates={},arena={},packages={},exclusions={},checkpoint=str(store.path),
                 report=str(out/'OVERNIGHT_FINAL_REPORT.json'),output=str(out),started_utc=store.state['started_utc'])
    def publish():
        cfv.atomic_json(out/'OVERNIGHT_FINAL_REPORT.json',summary)
        cfv.atomic_json(out/'FINAL_REPORT.json',summary)
    def stage(name,fn,clean=True):
        source_lock(a.expected_commit)
        if store.age()>=a.hours*3600:raise TimeoutError('target budget exhausted before '+name)
        summary['stage']=name;publish();print('SERVER_STAGE',name,flush=True)
        result=store.run(name,fn,clean)
        summary['completed_stages']=list(store.state['stages']);publish()
        return result
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    try:
        if not a.smoke and not a.benchmark_only:
            if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):raise RuntimeError('RTX 5090 CUDA required')
            x=torch.randn(64,64,device='cuda',requires_grad=True);(x@x.T).sum().backward();torch.cuda.synchronize()
            if not torch.isfinite(x.grad).all():raise RuntimeError('RTX 5090 backward kernel probe failed')
            summary['gpu']=dict(name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda)
        initial,_=load_checkpoint(champion,device='cpu')
        if (initial.config.model.d_model,initial.config.model.num_layers,initial.config.model.nhead)!=(192,4,6):
            raise ValueError('actual 192/4/6 champion required')
        def baseline_stage(folder):
            model,_=load_checkpoint(source,device='cpu');bind_champion(model,initial)
            path=folder/'baseline.pt';save_checkpoint(path,model,metadata=dict(handoff=handoff))
            return dict(checkpoint=str(path))
        baseline=stage('baseline',baseline_stage)['checkpoint']
        def original_benchmark(folder):
            releases=build_pr85(REPO,source,folder/'frozen_pr85',resources['worker_cpus'])
            corpus=folder/'real_match_states.json'
            # Frozen PR85 is a historical 16-river-iteration control. Production
            # arms always use the locked 8/8/16 budget; no fallback is allowed.
            results={}
            for name in ('with_hud','as_built'):
                results[name]=measure(releases[name],folder/(name+'.json'),corpus,resources,a.smoke,collect=name=='with_hud')
            return dict(base_sha=PR85_SHA,results=results,corpus=str(corpus),releases=releases)
        old=stage('frozen_pr85_server_benchmark',original_benchmark)
        summary['gates']['frozen_pr85']=old['results'];corpus=old['corpus'];publish()
        device='cpu' if a.smoke or a.benchmark_only else 'cuda'
        def runtime_calibration(folder):
            phase_b(baseline,folder/'pilot',dict(river_n=8,turn_n=8,holdout_n=4,net_epochs=1),min(workers,2),device,True)
            model,_=load_checkpoint(baseline,device='cpu');wide=folder/'wide.pt';save_checkpoint(wide,expand(model))
            pilot=folder/'pilot/cfv_nets.npz';selected=dict(SERVER_ITERATIONS)
            all_profiles=profiles();arm_gates={}
            for arm in ARM_ORDER:
                release=internal(wide,pilot,folder/('arm_'+arm),all_profiles[arm],'pilot_'+arm)
                arm_gates[arm]=measure(release,folder/('arm_'+arm+'.json'),corpus,resources,a.smoke)
            return dict(iterations=selected,iteration_budget_locked=True,pilot_gates=arm_gates,nets=str(pilot),wide=str(wide),
                        profiles=all_profiles,note='Calibration fixture only; never eligible for the final arena or upload.')
        calibration=stage('runtime_calibration',runtime_calibration)
        summary['gates']['calibration']=calibration;publish()
        if a.benchmark_only:
            summary['status']='BENCHMARK_COMPLETE';return 0
        audit=None;offline_benchmark=None
        if a.cfv_data_recovery:
            if a.smoke:raise ValueError('production recovery cannot use smoke fixtures')
            from cfv_recovery import inspect_history,benchmark_labels
            audit=stage('historical_cfv_compatibility',lambda folder:inspect_history(a.reuse_evening_run,a.reuse_source,folder))
            summary['historical_cfv_compatibility']=audit;publish()
            offline_benchmark=stage('offline_label_quality_throughput',lambda folder:benchmark_labels(
                audit,folder,workers,full_precision=a.full_precision))
            summary['offline_label_benchmark']=offline_benchmark;publish()
        def compute_plan(folder):
            if a.smoke:
                return dict(river_n=8,turn_n=8,rebel_n=8,holdout_n=4,net_epochs=50,policy_epochs=1,arms=list(ARM_ORDER),
                            smoke=True,note='Development only; final arena is intentionally forbidden.')
            rates={};n=max(64,workers*32)
            for street,name in ((3,'river'),(2,'turn')):
                config=(offline_benchmark[name]['selected'] if a.cfv_data_recovery else
                        dict(iterations=OFFLINE[name+'_iterations'],chance_samples=4))
                teacher=(offline_benchmark['turn']['reference_teacher'] if a.cfv_data_recovery else
                         str(Path(calibration['nets']).with_name('cfv_river.npz')))
                x,y,meta=mixed_dataset(folder/(name+'.npz'),baseline,street,n,31000000000+street*10000000000,workers,
                    config['iterations'],config['chance_samples'],None if street==3 else teacher)
                rates[name]=meta['active_compute_s']/n*1.35
            teachers=collect_teacher(baseline,n,71000000000,calibration['nets'],workers,OFFLINE['teacher_iterations'],2)
            rates['rebel']=collect_teacher.last_seconds/n*1.35
            collect_teacher(baseline,max(32,workers*32),72000000000,calibration['nets'],workers,2,1,True)
            rates['anchor']=collect_teacher.last_seconds/max(32,workers*32)*1.35
            start=time.monotonic();_train(x,y,'cfv_turn.',2,17,device,cosine=True);torch.cuda.synchronize()
            rates['cfv_epoch']=(time.monotonic()-start)/len(x)/2*1.5
            model,_=load_checkpoint(calibration['wide'],device=device)
            start=time.monotonic();fit(model,teachers,2);torch.cuda.synchronize()
            rates['policy_epoch']=(time.monotonic()-start)/len(teachers)/2*1.5
            # Judge + opponent + inference all timed on one core: conservative
            # upper bound for a single-core RPC arena with parallel opponents.
            from server_benchmark import sample_states
            release=internal(baseline,calibration['nets'],folder/'match_timing',calibration['profiles']['D'],'match_timing')
            command=[sys.executable,str(REPO/'scripts/sprint24/server_match_probe.py'),'--zip',release['source'],'--model',release['model'],
                     '--core',str(resources['inference_core']),'--out',str(folder/'match_seconds.json')]
            subprocess.run(command,check=True,timeout=1800)
            probe=json.loads((folder/'match_seconds.json').read_text())
            bench=max((r.get('mean_ms',100.) for r in calibration['pilot_gates'].values()))/1000*512+12*.5+10.
            if a.cfv_data_recovery:
                from cfv_recovery import choose_plan as recovery_plan
                plan=recovery_plan(audit,offline_benchmark,rates,a.hours,store.age(),probe['seconds'],bench,workers,full_precision=a.full_precision)
                cfv.atomic_json(folder/'RECOVERY_PLAN.json',plan)
                return plan
            return choose_plan(rates,a.hours,probe['seconds'],bench,store.age(),workers)
        plan=stage('production_plan',compute_plan);summary['plan']=plan;publish()
        def cfv_training(folder):
            if a.cfv_data_recovery:
                from cfv_recovery import phase_b as recovery_phase_b
                stats=recovery_phase_b(baseline,folder,plan,workers,device,audit,
                    deadline=time.monotonic()+max(0.,a.hours*3600-store.age()))
            else:stats=phase_b(baseline,folder,plan,workers,device,a.smoke)
            nets=str(folder/'cfv_nets.npz')
            if not a.smoke:assert_production_cfv(nets)
            return dict(nets=nets,validation=stats)
        trained=stage('production_cfv_50_epochs',cfv_training,clean=not a.cfv_data_recovery);summary['training']=trained;nets=trained['nets'];publish()
        previous=baseline;checkpoints={'baseline':baseline};distillation={}
        for generation,seed in (('g3',100000000000),('g4',200000000000),('g5',300000000000)):
            def evolve(folder,previous=previous,seed=seed):
                dest=folder/'model.pt'
                stats=one_generation(previous,dest,champion,nets,seed,plan,workers,device,a.smoke)
                return dict(checkpoint=str(dest),validation=stats)
            generation_result=stage(generation,evolve)
            previous=generation_result['checkpoint'];checkpoints[generation]=previous;distillation[generation]=generation_result['validation']
        summary['distillation']=distillation;summary['checkpoints']=checkpoints
        if a.smoke:
            summary['status']='DEVELOPMENT_SMOKE_COMPLETE_NO_FINAL_ARENA';return 0
        assert_production_cfv(nets)
        arm_profiles=calibration['profiles']
        def variants_stage(folder):
            return {arm:internal(previous,nets,folder/arm,arm_profiles[arm],'arena_'+arm) for arm in plan['arms']}
        releases=stage('trained_arm_variants',variants_stage)
        def preflop_stage(folder):
            reference=internal(champion,nets,folder/'reference',arm_profiles['B'],'preflop_reference')
            bundles={'champion':Bundle(reference['source'],reference['model'])}
            for name,path in checkpoints.items():
                release=internal(path,nets,folder/name,arm_profiles['B'],'preflop_'+name)
                bundles[name]=Bundle(release['source'],release['model'])
            rows=json.loads(Path(corpus).read_text())['states'];drifts={name:0. for name in checkpoints}
            for row in rows:
                payload=row['payload'];request=payload['requests'][-1]
                if request['public_cards']:continue
                probabilities={}
                for name,bundle in bundles.items():
                    snapshot,_=bundle.runtime.request_snapshot(request);ctx,_=bundle.runtime.context_from_payload(copy.deepcopy(payload))
                    logits=bundle.agent.logits(bundle.agent.observation(snapshot,ctx))[0].astype(float)
                    p=np.exp(logits-logits.max());probabilities[name]=p/p.sum()
                for name in checkpoints:drifts[name]=max(drifts[name],float(.5*np.abs(probabilities[name]-probabilities['champion']).sum()))
            if max(drifts.values())>1e-12:raise RuntimeError('ORIGINAL_CHAMPION_PREFLOP_ZERO_DRIFT_FAILED')
            return drifts
        summary['preflop_max_tv']=stage('preflop_zero_drift',preflop_stage)
        def arm_update(arenas,exclusions,latencies):
            summary.update(arena=arenas,exclusions=exclusions)
            summary['gates']['trained_arms']=latencies;publish()
            if not a.full_precision or 'A' not in arenas or len(arenas)==len(plan['arms']):
                return
            compared=compare_arms(arenas)
            ranking=rank_arms(compared,exclusions)
            if not ranking:
                return
            snapshot=''.join(arenas)
            def preview(folder):
                root=folder/'candidates'
                run_id=a.expected_commit+':'+str(out)+':preview:'+snapshot
                reservation=reserve(root,folder,run_id,ranking)
                return {version:package_one(previous,nets,root,folder,reservation,int(version),
                    arm,arm_profiles[arm],compared[arm]['promoted'],
                    lambda release,path:measure(release,path,corpus,resources),preview=snapshot)
                    for version,arm in reservation['versions'].items()}
            summary['provisional_packages']=stage('arena_packages_'+snapshot,preview,clean=False)
            publish()
        arenas,exclusions,latencies=arena_phase(releases,nets,corpus,resources,workers,stage,arm_update,gate_function=full_power_gates)
        if 'A' not in arenas:
            summary['status']='FAILED_NO_PAIRED_BASELINE';summary['packaging_reason']='No valid paired A results; no unsupported promotion or unsafe fallback is published.'
            return 1
        compared=compare_arms(arenas);ranking=rank_arms(compared,exclusions)
        summary.update(arena=compared,ranking=ranking,exclusions=exclusions)
        reservation=reserve(a.deliver,out,a.expected_commit+':'+str(out),ranking)
        packages={}
        for version_text,arm in reservation['versions'].items():
            version=int(version_text)
            packed=package_one(previous,nets,a.deliver,out,reservation,version,arm,arm_profiles[arm],compared[arm]['promoted'],
                lambda release,path:measure(release,path,corpus,resources))
            if packed.get('excluded'):exclusions[arm]=packed['reason'];summary.setdefault('package_exclusions',[]).append(packed)
            else:packages[str(version)]=packed
            summary.update(packages=packages,exclusions=exclusions);publish()
        summary.pop('provisional_packages',None)
        source_lock(a.expected_commit)
        if cfv.digest(champion)!=identity['champion_sha256'] or cfv.digest(source)!=identity['handoff_sha256']:
            raise RuntimeError('source checkpoint changed')
        summary.update(status='COMPLETE' if len(packages)==3 else 'COMPLETE_WITH_EXCLUSIONS',
                       packaging_reason='Ranked by paired EV lower 95% bound vs A, then match win rate. No promoted arm => eligible Legacy first.',
                       elapsed_s=store.age(),promoted=[name for name,row in compared.items() if row['promoted'] and name not in exclusions])
        if len(packages)<3:summary['limitations']=['Fewer than three latency-qualified arms/packages were available; excluded arms are never uploaded.']
        return 0
    except BaseException as exc:
        status='FAILED'
        if 'CFV_HOLDOUT_GATE_FAILED' in str(exc):status='STOPPED_CFV_HOLDOUT_GATE'
        elif 'INSUFFICIENT_' in str(exc):status='STOPPED_CFV_DATA_BUDGET'
        summary.update(status=status,error=type(exc).__name__+': '+str(exc),elapsed_s=store.age())
        raise
    finally:
        summary['elapsed_s']=store.age();publish()
        execution.close();print('SERVER_FINAL_REPORT',out/'OVERNIGHT_FINAL_REPORT.json',summary['status'],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours',type=float,default=None);p.add_argument('--workers',type=int);p.add_argument('--inference-core',type=int)
    p.add_argument('--expected-commit',required=True);p.add_argument('--champion',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deliver',type=Path,default=Path('/root/autodl-tmp/sprint24/deliver'))
    p.add_argument('--evening-root',type=Path,default=Path('/root/autodl-tmp/sprint24/evening_production'))
    p.add_argument('--resume',action='store_true');p.add_argument('--fresh',action='store_true')
    p.add_argument('--benchmark-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--cfv-data-recovery',action='store_true',help='audit/relabel historical public states and transfer-initialize CFVs; enforce 80k/50k')
    p.add_argument('--reuse-evening-run',type=Path,default=Path('/root/autodl-tmp/sprint24/evening_production/20261104T142449Z_5cbf1db6'))
    p.add_argument('--reuse-source',type=Path,default=Path('/root/autodl-tmp/poker-evening-anchor-fix-5cbf1db'))
    p.add_argument('--full-precision',action='store_true',help='lock V23 recovery counts, offline 64/32 x4, and all four arms')
    p.add_argument('--_run',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.hours is None:a.hours=SERVER_TARGET_HOURS if a.cfv_data_recovery else 8.5
    if not 0<a.hours<=SERVER_TARGET_HOURS:p.error('hours must be >0 and <=14.5')
    if not a.cfv_data_recovery and a.hours>8.5 and not (a.smoke or a.benchmark_only):
        p.error('server hours above 8.5 require --cfv-data-recovery')
    if a.full_precision and (not a.cfv_data_recovery or a.smoke or a.benchmark_only):
        p.error('--full-precision requires a production --cfv-data-recovery run')
    source_lock(a.expected_commit)
    if a._run:
        try:return _execute(a)
        except BaseException as exc:
            report=a.out/'OVERNIGHT_FINAL_REPORT.json'
            if not report.exists():
                cfv.atomic_json(report,dict(status='FAILED_SETUP',error=type(exc).__name__+': '+str(exc),commit=a.expected_commit))
            raise
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'.watchdog.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=datetime.datetime.now(datetime.timezone.utc)
        checkpoint=a.out/'CHECKPOINTS.json'
        if checkpoint.exists():started=datetime.datetime.fromisoformat(json.loads(checkpoint.read_text())['started_utc'])
        remaining=SERVER_WATCHDOG_SECONDS-(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds()
        if remaining<=0:raise RuntimeError('the run has exhausted its cumulative 16h watchdog')
        arguments=list(sys.argv[1:] if argv is None else argv)
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*arguments,'--_run'],start_new_session=True)
        def stop(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        try:return child.wait(timeout=remaining)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            group_stop(child,grace=0.)
            path=a.out/'OVERNIGHT_FINAL_REPORT.json'
            report=json.loads(path.read_text()) if path.exists() else {}
            report.update(status='WATCHDOG_FAILED',reason='maximum cumulative 16h process-group watchdog or interruption',commit=a.expected_commit)
            cfv.atomic_json(path,report)
            return 1
        finally:
            if child.poll() is not None and child.returncode:group_stop(child,grace=0.)

if __name__=='__main__':raise SystemExit(main())

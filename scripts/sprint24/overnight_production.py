"""Calibrated phase-two production, 8.5h target and 9h process-group watchdog.

Run from an immutable, clean checkout. No server training is started locally.
All quality/performance evidence comes from the exact exported respond path.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import datetime
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from evening_production import EXPECTED_CHAMPION_SHA256,find_champion,git,latency_gate

HARD_SECONDS=9.*3600
ONLINE=dict(iterations={'1':4,'2':4,'3':16},chance_samples=1,blueprint_weight={'1':.5,'2':.5,'3':.3},
            check_raise=True,contract='slim-rnr-terminal-match-v1',guard_mode='slim',legal_only=True,
            rnr_eta=.5,terminal_match_utility=True,mathematical_locks=True)
OFFLINE=dict(river_iterations=64,turn_iterations=32,chance_samples=4,
             teacher_iterations={'1':32,'2':32,'3':64},teacher_chance_samples=2)
ROLES=('baseline','g3','g4','g5')
HISTORICAL_SEED=1800000000
HOLDOUT_SEED=1971000000
GUARD_TUNE_SEED=1951000000
GUARD_CONFIRM_SEED=1961000000


def score_cells(row,strict=True):
    """Equal phase weight; within a phase all opponents get equal weight."""
    import numpy as np
    from evening_production_eval import BENCHMARK,TOURNAMENT
    means=[]
    for phase,opponents in (('benchmark',BENCHMARK),('tournament',TOURNAMENT)):
        values=[]
        for name in opponents:
            cell=row[phase][name]
            if strict and (cell.get('hands',0)<(5000 if phase=='benchmark' else 1000) or cell.get('partial_protocol_only')):
                raise ValueError('incomplete production evaluation: '+phase+'/'+name)
            value=float(cell['mbb_per_hand'])
            if not math.isfinite(value):raise ValueError('nonfinite evaluation score')
            values.append(value)
        means.append(float(np.mean(values)))
    return float(np.mean(means))


def select_handoff(root,champion,fresh=False):
    """Latest completed report, then best scored *verified* checkpoint."""
    from alphanl.stage2 import cfv
    if fresh:return Path(champion),dict(mode='fresh',checkpoint=str(champion))
    reports=[]
    for path in Path(root).glob('**/FINAL_REPORT.json'):
        report=json.loads(path.read_text())
        if report.get('status')=='COMPLETE' and not report.get('identity',{}).get('smoke',True):reports.append((path,report))
    if not reports:return Path(champion),dict(mode='original_champion_no_completed_report',checkpoint=str(champion))
    path,report=max(reports,key=lambda p:(p[0].stat().st_mtime_ns,str(p[0])))
    if report['identity']['champion_sha256']!=cfv.digest(champion):raise ValueError('handoff belongs to a different champion')
    scored={name:score_cells(report['evaluation']['results'][name]) for name in ('baseline','g1','g2')}
    role=max(scored,key=lambda name:(scored[name],-('baseline','g1','g2').index(name)))
    release=report['releases'][role]
    for key in ('source','model'):
        if cfv.digest(release[key])!=release[key+'_sha256']:raise ValueError('handoff release hash mismatch: '+key)
    import numpy as np
    from alphanl.stage2.train import load_checkpoint
    with np.load(release['model'],allow_pickle=False) as blob:
        meta=json.loads(blob['meta_json'].tobytes())
        if meta.get('deployment_release')!=release['release']:raise ValueError('handoff release binding mismatch')
        source=Path(meta['source']['checkpoint'])
        model,_=load_checkpoint(source,device='cpu')
        if model.config.to_dict()!=meta['config'] or any(meta['settings'].get(k)!=val for k,val in model.settings.items()):
            raise ValueError('handoff checkpoint configuration differs from evaluated package')
        # Bind the evaluated package to the actual .pt, never trust a filename.
        for name,value in model.state_dict().items():
            key=name.removeprefix('reference.backbone.')
            if key not in blob or not np.array_equal(value.cpu().numpy(),blob[key]):
                raise ValueError('handoff checkpoint differs from evaluated package: '+key)
    return source,dict(mode='evening_best',report=str(path),report_sha256=cfv.digest(path),role=role,
                       scores=scored,checkpoint=str(source),checkpoint_sha256=cfv.digest(source),release=release['release'])


def plan_counts(rates,hours,eval_seconds,elapsed,workers,smoke=False):
    if not 0<hours<=8.5:raise ValueError('target hours must be >0 and <=8.5')
    if smoke:return dict(river_n=8,turn_n=8,rebel_n=8,holdout_n=4,net_epochs=1,policy_epochs=1,smoke=True)
    if any(not math.isfinite(float(v)) or v<=0 for v in rates.values()):raise ValueError('invalid measured rates')
    # All GPU epochs, holdouts and 3 generations are priced, not a flat guess.
    reserve=eval_seconds*1.5+1800+hours*3600*.12
    holdout=2048
    fixed=holdout*(rates['river']+rates['turn']+15*rates['rebel']+3*rates['anchor'])
    budget=hours*3600-elapsed-reserve-fixed
    if budget<=0:raise RuntimeError('measured evaluation/training/holdout floors cannot fit 8.5h')
    costs=dict(river_n=rates['river']+50*rates['cfv_epoch'],turn_n=rates['turn']+50*rates['cfv_epoch'],
               rebel_n=3*(rates['rebel']+12*rates['policy_epoch']))
    result={}
    for name,share,floor,cap in (('river_n',.25,8192,500000),('turn_n',.30,4096,350000),('rebel_n',.45,4096,200000)):
        count=int(budget*share/costs[name])//256*256
        if count<floor:raise RuntimeError('calibrated production floor cannot fit: '+name)
        result[name]=min(count,cap)
    predicted=elapsed+reserve+fixed+sum(result[name]*cost for name,cost in costs.items())
    if predicted>hours*3600 or predicted>=HARD_SECONDS:raise RuntimeError('plan exceeds target or hard watchdog')
    result.update(net_epochs=50,policy_epochs=12,holdout_n=holdout,workers=workers,predicted_seconds=predicted,
        reserve_seconds=reserve,fixed_holdout_seconds=fixed,rates=rates,generations=['g3','g4','g5'],hours=hours)
    return result


def group_stop(child,grace=5):
    """Kill the run's whole session, including descendants after leader exits."""
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline:
        child.poll()
        time.sleep(.05)
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()


def run(a):
    import copy
    import numpy as np
    import torch
    from alphanl.stage2 import cfv
    from alphanl.stage2.train import load_checkpoint,save_checkpoint
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    from alphanl.stage2.deployment_guard import V102_R7_GUARD
    from evening_parallel import default_workers
    from evening_d import Bundle,play_judge
    from evening_latency import cases
    from evening_production_eval import evaluate,compare_guards
    from overnight_data import mixed_dataset,phase_b,collect_teacher,TREE
    from overnight_distill import bind_champion,expand,fit,one_generation
    from pack_wizard import pack_evening
    start=time.monotonic();commit=git('rev-parse','HEAD')
    if a.allow_dirty and not a.smoke:raise ValueError('--allow-dirty is restricted to smoke')
    if a.expected_commit and commit!=a.expected_commit:raise RuntimeError('COMMIT_SHA mismatch')
    def source_lock():
        if not a.allow_dirty and (git('rev-parse','HEAD')!=commit or git('status','--porcelain')):
            raise RuntimeError('SOURCE_LOCK_FAILED: production requires git status --porcelain to be empty')
    source_lock()
    workers=default_workers() if a.workers is None else a.workers
    if not 1<=workers<=default_workers():raise ValueError('workers must be <= min(24,cpu_count)')
    champion=find_champion(a.champion);champion_sha=cfv.digest(champion)
    if champion_sha!=a.expected_champion_sha256:raise RuntimeError('champion SHA256 mismatch')
    initial,blob=load_checkpoint(champion,device='cpu');arch=initial.config.model
    if (arch.d_model,arch.num_layers,arch.nhead)!=(192,4,6):raise RuntimeError('expected actual 192/4/6 champion')
    source,handoff=select_handoff(a.evening_root,champion,a.fresh)
    handoff_sha=cfv.digest(source)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    gpu={}
    if not a.smoke:
        if not torch.cuda.is_available():raise RuntimeError('production requires RTX 5090 CUDA; CPU only allowed with --smoke')
        gpu=dict(name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda)
        if '5090' not in gpu['name']:raise RuntimeError('expected RTX 5090')
        x=torch.randn(64,64,device='cuda',requires_grad=True);(x@x.T).sum().backward();torch.cuda.synchronize()
        if not torch.isfinite(x.grad).all():raise RuntimeError('5090 backward kernel probe failed')
    device='cpu' if a.smoke else 'cuda'
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(a.out) if a.out else Path('/root/autodl-tmp/sprint24/overnight_production')/(timestamp+'_'+commit[:8])
    if not a.smoke and out.resolve().is_relative_to(REPO):raise ValueError('production output must be outside locked checkout')
    out.mkdir(parents=True,exist_ok=False)
    identity=dict(commit=commit,champion=str(champion),champion_sha256=champion_sha,handoff=handoff,
        workers=workers,gpu=gpu,arch=arch.__dict__,online=ONLINE,offline=OFFLINE,tree=TREE,smoke=a.smoke,hard_seconds=HARD_SECONDS)
    cfv.atomic_json(out/'run_identity.json',identity)
    def stage(name,**fields):
        source_lock()
        if name!='COMPLETE' and time.monotonic()-start>a.hours*3600:raise TimeoutError('8.5-hour target exhausted; refusing to start another stage')
        cfv.atomic_json(out/'progress.json',dict(stage=name,elapsed_s=time.monotonic()-start,**fields))
        print('OVERNIGHT_STAGE',name,json.dumps(fields),flush=True)
    full_guard=dict(V102_R7_GUARD,enabled=True,victory_blind_lock_enabled=True)
    def internal(cp,nets,role,profile):
        if cfv.load_nets(nets)[1].get('tree')!=TREE:raise ValueError('overnight CFVs must use the expanded check-raise tree')
        folder=out/'internal_packages'/role;folder.mkdir(parents=True,exist_ok=True)
        source_zip=folder/'V999_SOURCE.zip';model_npz=folder/'V999_MODEL.npz'
        release='V999_INTERNAL_OVERNIGHT_'+role.upper()
        export_botzone_weights(cp,model_npz,cfv_nets=nets,release_id=release,online_profile=profile,deployment_guard=full_guard)
        pack_botzone_zip(source_zip,release_id=release)
        return dict(source=str(source_zip.resolve()),model=str(model_npz.resolve()),release=release,
                    source_sha256=cfv.digest(source_zip),model_sha256=cfv.digest(model_npz),internal=True)
    try:
        stage('bind_original_preflop')
        base,_=load_checkpoint(source,device='cpu');bind_champion(base,initial)
        baseline=out/'baseline.pt';save_checkpoint(baseline,base,metadata=dict(handoff=handoff))
        stage('expanded_tree_pilot')
        pilot_plan=dict(river_n=8,turn_n=8,holdout_n=4,net_epochs=1)
        phase_b(baseline,out/'pilot',pilot_plan,min(workers,2),device,True)
        nets_path=out/'pilot/cfv_nets.npz'
        # Use the largest deployed residual for timing calibration.
        wide=expand(copy.deepcopy(base));wide_path=out/'pilot/wide.pt';save_checkpoint(wide_path,wide)
        pilot_latency={};profiles=[]
        for iterations in ({'1':4,'2':4,'3':16},{'1':4,'2':4,'3':12},{'1':4,'2':4,'3':8},
                           {'1':3,'2':3,'3':6},{'1':2,'2':2,'3':4}):
            profile=dict(ONLINE,iterations=iterations)
            fixture=internal(wide_path,nets_path,'pilot_'+str(len(profiles)),profile)
            try:
                passed=latency_gate(fixture,out/('pilot_latency_'+str(len(profiles))+'.json'),64 if a.smoke else 256,4 if a.smoke else 12)
            except subprocess.CalledProcessError:
                pilot_latency[str(iterations)]=dict(passed=False);profiles.append(profile);continue
            pilot_latency[str(iterations)]=passed;online=profile;timing=fixture;break
        else:raise RuntimeError('expanded check-raise runtime cannot fit hard respond latency gates')
        stage('actual_machine_calibration',online=online)
        n=8 if a.smoke else max(64,workers*32)
        rates={}
        for street,name in ((3,'river'),(2,'turn')):
            x,y,m=mixed_dataset(out/('calibration_'+name+'.npz'),baseline,street,n,31000000000+street*10000000000,workers,
                2 if a.smoke else OFFLINE[name+'_iterations'],1 if a.smoke else 4,None if street==3 else out/'pilot/cfv_river.npz')
            rates[name]=m['active_compute_s']/n*1.35
        teachers=collect_teacher(baseline,n,71000000000,nets_path,workers,2 if a.smoke else OFFLINE['teacher_iterations'],1 if a.smoke else 2)
        rates['rebel']=collect_teacher.last_seconds/n*1.35
        collect_teacher(baseline,8 if a.smoke else max(32,workers*2),72000000000,nets_path,workers,2,1,True)
        rates['anchor']=collect_teacher.last_seconds/(8 if a.smoke else max(32,workers*2))*1.35
        from evening_pipeline import _train
        t=time.monotonic();_train(x,y,'cfv_turn.',2,17,device,cosine=True)
        if device=='cuda':torch.cuda.synchronize()
        rates['cfv_epoch']=(time.monotonic()-t)/len(x)/2*1.5
        t=time.monotonic();fit(wide.to(device),teachers,2)
        if device=='cuda':torch.cuda.synchronize()
        rates['policy_epoch']=(time.monotonic()-t)/len(teachers)/2*1.5
        bundle=Bundle(timing['source'],timing['model']);t=time.monotonic()
        play_judge(bundle,1700000000,0,2 if a.smoke else 50,'TAG_Pro',keep_trace=False)
        match_s=time.monotonic()-t
        # Two full suites + 200 paired/seat-swapped matches per guard per role.
        matches=2*len(ROLES)*(3*50+5*10)*2+len(ROLES)*2*5*20*2
        eval_seconds=matches*match_s/workers
        plan=plan_counts(rates,a.hours,eval_seconds,time.monotonic()-start,workers,a.smoke)
        plan.update(online=online,offline=OFFLINE,eval_matches=matches,match_seconds=match_s,
                    estimated_evaluation_seconds=eval_seconds,pilot_latency=pilot_latency)
        cfv.atomic_json(out/'plan.json',plan)
        stage('mixed_cfv_50_epoch_training',plan=plan)
        stats=phase_b(baseline,out/'b',plan,workers,device,a.smoke);nets_path=out/'b/cfv_nets.npz'
        stage('g3_g4_g5_policy_value_and_astra')
        checkpoints={'baseline':baseline};distillation={};previous=baseline
        for generation,seed in (('g3',100000000000),('g4',200000000000),('g5',300000000000)):
            stage('generation_'+generation)
            dest=out/('c/u250_overnight_'+generation+'.pt')
            distillation[generation]=one_generation(previous,dest,champion,nets_path,seed,plan,workers,device,a.smoke)
            checkpoints[generation]=dest;previous=dest
        stage('independent_empirical_guard_ablation')
        variants={role+'_'+mode:internal(cp,nets_path,role+'_'+mode,
                  dict(online,guard_mode=mode,comparison_only=mode=='legacy_full'))
                  for role,cp in checkpoints.items() for mode in ('slim','legacy_full')}
        ablation={}
        for name,release in variants.items():
            latency_gate(release,out/(name+'_latency.json'),64 if a.smoke else 256,4 if a.smoke else 12)
        ablation=evaluate(variants,out/'guard_200_match_comparison',workers,0,1 if a.smoke else 20,2 if a.smoke else 50,seed_base=GUARD_TUNE_SEED)
        comparisons=compare_guards(ablation,ROLES)
        selected={role:variants[role+'_slim'] for role in checkpoints}
        # Architectural safety is mandatory; a score cannot re-enable old rules.
        guard_selection={role:dict(mode='slim',legacy_comparison=comparisons[role]) for role in checkpoints}
        cfv.atomic_json(out/'guard_selection.json',guard_selection)
        stage('actual_runtime_preflop_and_generation_gate')
        bundles={name:Bundle(variants[name+'_slim']['source'],variants[name+'_slim']['model']) for name in ROLES}
        original_package=internal(champion,nets_path,'original_preflop_reference',online)
        bundles['original_champion']=Bundle(original_package['source'],original_package['model'])
        evolution={};outputs={k:[] for k in bundles};pre={k:[] for k in bundles};blueprint={k:[] for k in bundles}
        for payload in cases(32 if a.smoke else 128,1931000000):
            for name,b in bundles.items():
                snap,resp=b.runtime.request_snapshot(payload['requests'][0]);ctx,_=b.runtime.context_from_payload(payload)
                amounts={a:snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0) for a,r in resp.items()}
                p=b.agent.policy(snap,ctx,amounts=amounts,request=payload['requests'][0])
                (outputs if snap.street>0 else pre)[name].append(p)
                if snap.street>0:
                    logits=b.agent.logits(b.agent.observation(snap,ctx))[0].astype(float)
                    bp=np.exp(logits-logits.max());blueprint[name].append(bp/bp.sum())
        for i,role in enumerate(ROLES[1:],1):
            drift=float((.5*np.abs(np.array(pre[role])-np.array(pre['original_champion']))).sum(1).max())
            tv=.5*np.abs(np.array(outputs[role])-np.array(outputs[ROLES[i-1]])).sum(1)
            if drift>1e-12:raise RuntimeError('ORIGINAL_CHAMPION_PREFLOP_ZERO_DRIFT_FAILED')
            evolution[role]=dict(preflop_max_tv=drift,postflop_mean_tv=float(tv.mean()),postflop_max_tv=float(tv.max()),
                                 blueprint_max_tv=float((.5*np.abs(np.array(blueprint[role])-np.array(blueprint[ROLES[i-1]]))).sum(1).max()),
                                 accepted=distillation[role]['accepted'])
        stage('allocate_final_packages_and_gate')
        deliver=Path(a.deliver) if a.deliver else Path('/root/autodl-tmp/sprint24/deliver')
        releases={};gates={}
        for role,cp in checkpoints.items():
            if a.smoke:releases[role]=selected[role];continue
            mode=guard_selection[role]['mode']
            releases[role]=pack_evening(cp,nets_path,deliver,online_profile=dict(online,guard_mode=mode),
                deployment_guard=full_guard,release_label='OVERNIGHT_'+role.upper())
            # Exact released bytes are checked once more, not just internal twins.
            gates[role]=latency_gate(releases[role],out/(role+'_formal_latency.json'),256,12)
        evaluation={}
        for label,seed in (('historical',HISTORICAL_SEED),('independent_holdout',HOLDOUT_SEED)):
            stage('full_zip_evaluation_'+label)
            report=evaluate(releases,out/('evaluation_'+label),workers,1 if a.smoke else 50,1 if a.smoke else 10,2 if a.smoke else 50,seed_base=seed)
            if report['runtime_ms']['mean']>=40 or report['runtime_ms']['p99']>=120:raise RuntimeError('evaluation latency gate failed')
            evaluation[label]=report
        stage('immutable_publication')
        if cfv.digest(champion)!=champion_sha or cfv.digest(source)!=handoff_sha:raise RuntimeError('source checkpoint changed')
        eligible=[role for role in ROLES if comparisons[role]['improvement_demonstrated']]
        chosen=None if a.smoke or not eligible else max(eligible,key=lambda role:score_cells(evaluation['independent_holdout']['results'][role]))
        result=dict(status='COMPLETE',identity=identity,handoff=handoff,plan=plan,cfv=stats,distillation=distillation,
            guard_ablation=ablation,guard_selection=guard_selection,policy_evolution=evolution,evaluation=evaluation,
            checkpoints={k:dict(path=str(v),sha256=cfv.digest(v)) for k,v in checkpoints.items()},releases=releases,
            final_latency=gates,recommended_role=chosen,guard_improvement_gate=dict(passed=bool(eligible),eligible_roles=eligible),
            elapsed_s=time.monotonic()-start,smoke=a.smoke)
        source_lock()
        cfv.atomic_json(out/'QUALITY_REPORT.json',result)
        if not a.smoke and not eligible:
            raise RuntimeError('GUARD_PAIRED_IMPROVEMENT_GATE_FAILED: no slim candidate beats legacy EV and win rate at paired 95% confidence; see QUALITY_REPORT.json')
        cfv.atomic_json(out/'OVERNIGHT_FINAL_REPORT.json',result)
        if not a.smoke:
            cfv.atomic_json(deliver/('OVERNIGHT_FINAL_REPORT_'+timestamp+'.json'),result)
            cfv.atomic_json(deliver/'OVERNIGHT_FINAL_REPORT.json',result)
        lines=['版本：'+', '.join('V%d'%r['version'] if 'version' in r else 'V999' for r in releases.values())+'（内部 R1），release='+', '.join(r['release'] for r in releases.values()),
            '', 'commit='+commit,'recommended_role='+str(chosen),'',json.dumps(releases,indent=2),
            '', 'Smoke uses partial protocol matches only.' if a.smoke else 'Both full seed suites and all exact released ZIP latency gates passed.']
        (out/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
        stage('COMPLETE',report=str(out/'OVERNIGHT_FINAL_REPORT.json'))
        return 0
    except BaseException as exc:
        cfv.atomic_json(out/'FAILED.json',dict(error=type(exc).__name__,message=str(exc),commit=commit,elapsed_s=time.monotonic()-start))
        print('OVERNIGHT_FAILED',type(exc).__name__,str(exc),flush=True)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours',type=float,default=8.5);p.add_argument('--workers',type=int)
    p.add_argument('--champion',type=Path);p.add_argument('--expected-champion-sha256',default=EXPECTED_CHAMPION_SHA256)
    p.add_argument('--expected-commit');p.add_argument('--out',type=Path);p.add_argument('--deliver',type=Path)
    p.add_argument('--evening-root',type=Path,default=Path('/root/autodl-tmp/sprint24/evening_production'))
    p.add_argument('--fresh',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--allow-dirty',action='store_true')
    p.add_argument('--_run',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 0<a.hours<=8.5:p.error('--hours must be >0 and <=8.5 (9h watchdog is fixed)')
    if a._run:return run(a)
    argv=list(sys.argv[1:])
    if a.out is None:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        a.out=Path('/root/autodl-tmp/sprint24/overnight_production')/(stamp+'_'+git('rev-parse','HEAD')[:8])
        argv+=['--out',str(a.out)]
    hard_deadline=time.monotonic()+HARD_SECONDS
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv,'--_run'],start_new_session=True)
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        code=child.wait(timeout=max(0,hard_deadline-time.monotonic()))
        if code:group_stop(child,grace=min(5,max(0,hard_deadline-time.monotonic())))
        return code
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        group_stop(child,grace=min(5,max(0,hard_deadline-time.monotonic())))
        from alphanl.stage2 import cfv
        cfv.atomic_json(a.out/'WATCHDOG_FAILED.json',dict(status='FAILED',reason='9.0-hour hard deadline or interruption',hard_seconds=HARD_SECONDS))
        raise SystemExit('OVERNIGHT_WATCHDOG_STOPPED: 9.0-hour hard deadline or interruption')

if __name__=='__main__':raise SystemExit(main())

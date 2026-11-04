"""One-command calibrated 24-CPU/RTX-5090 production run; never edits source.

Every child has one BLAS thread. Offline precision is fixed; counts are sized
from this machine's measured pool throughput. A process-group watchdog enforces
an overall wall deadline instead of silently extending the run for many hours.
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

ONLINE={'iterations':{'1':4,'2':4,'3':8},'chance_samples':1,'blueprint_weight':{'1':.5,'2':.5,'3':.3},'contract':'hero-root-prior-convex-v1'}
EXPECTED_CHAMPION_SHA256='f298ab7a44956f92f2fa02c914ec4a68749a1381be2e2952159f989cc53c48fe'
OFFLINE={'river_iterations':64,'turn_iterations':32,'chance_samples':4,'teacher_iterations':32,'teacher_river_iterations':64,'teacher_chance_samples':2}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def find_champion(override=None):
    candidates=([Path(override)] if override else [])+[
        Path('/root/autodl-tmp/sprint24/ladder/u250_ladder_slayer.pt'),
        Path('/root/autodl-tmp/poker-ai/scripts/sprint24/ladder/u250_ladder_slayer.pt'),
        REPO.parent/'UPLOAD_V12_R12_LADDER_20261104/u250_ladder_slayer.pt',
        *[parent/'UPLOAD_V12_R12_LADDER_20261104/u250_ladder_slayer.pt' for parent in list(REPO.parents)[:4]],
        *[hit for parent in list(REPO.parents)[:3] for hit in parent.glob('archive_milestones_*/live_whitelist/u250_ladder_slayer.pt')]]
    for p in candidates:
        if p.is_file():return p.resolve()
    for base in (Path('/root/autodl-tmp'),REPO.parent):
        hits=sorted(base.glob('**/u250_ladder_slayer.pt')) if base.exists() else []
        if len(hits)==1:return hits[0].resolve()
        if len(hits)>1:raise RuntimeError('ambiguous champion paths: use --champion')
    raise FileNotFoundError('u250_ladder_slayer.pt not found; pass --champion /absolute/path')


def plan_counts(rates,hours,eval_seconds,elapsed,workers,smoke=False):
    if smoke:return dict(river_n=8,turn_n=8,rebel_n=8,net_epochs=1,policy_epochs=1,holdout_n=4,predicted_seconds=None)
    # Reserve measured full evaluation, GPU training, startup/IO and 15% slack.
    reserved=eval_seconds*1.35+900+hours*3600*.15
    budget=hours*3600-elapsed-reserved
    if budget<=0:raise RuntimeError('deadline cannot accommodate required evaluation and training reserves')
    result={}
    for key,share,cost,floor,cap in (
        ('river_n',.25,rates['river'],8192,300000),
        ('turn_n',.35,rates['turn'],4096,200000),
        ('rebel_n',.40,2*rates['rebel'],4096,200000)):
        count=int(budget*share/cost)//256*256
        if count<floor:raise RuntimeError('measured throughput cannot fit minimum production '+key)
        result[key]=min(cap,count)
    result.update(net_epochs=30,policy_epochs=12,holdout_n=1024,
                  predicted_seconds=elapsed+reserved+result['river_n']*rates['river']+result['turn_n']*rates['turn']+2*result['rebel_n']*rates['rebel'],
                  rates_wall_seconds_per_row=rates,workers=workers,reserved_seconds=reserved)
    if result['predicted_seconds']>3.5*3600:raise RuntimeError('plan exceeds 3.5-hour wall budget')
    return result


def make_timing_package(checkpoint,nets,out):
    from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Internal benchmark fixture, never an allocated/uploaded Botzone release.
    source=out/'V999_SOURCE.zip';model=out/'V999_MODEL.npz'
    export_botzone_weights(checkpoint,model,cfv_nets=nets,release_id='V999_INTERNAL_PRODUCTION_GATE',online_profile=ONLINE)
    pack_botzone_zip(source,release_id='V999_INTERNAL_PRODUCTION_GATE')
    return dict(source=str(source.resolve()),model=str(model.resolve()))


def latency_gate(release,out,samples=256,cold=12,enforce=True):
    cmd=[sys.executable,str(REPO/'scripts/sprint24/evening_latency.py'),'--zip',release['source'],'--model',release['model'],
         '--out',str(out),'--samples',str(samples),'--cold',str(cold)]
    if not enforce:cmd.append('--observe')
    # Pinning happens only in this subprocess; the 24-worker parent keeps its affinity.
    subprocess.run(cmd,check=True,timeout=180,stdout=subprocess.DEVNULL)
    return json.loads(Path(out).read_text())


def run(a):
    import numpy as np
    import torch
    import evening_pipeline as b
    import evening_cd as c
    from evening_parallel import default_workers
    from evening_d import Bundle,play_judge
    from evening_production_eval import evaluate,policy_evolution,BENCHMARK,TOURNAMENT
    from pack_wizard import pack_evening
    from alphanl.stage2 import cfv
    from alphanl.stage2.train import load_checkpoint
    start=time.monotonic();commit=git('rev-parse','HEAD')
    if a.allow_dirty and not a.smoke:raise ValueError('--allow-dirty is restricted to local smoke validation')
    if not a.allow_dirty and git('status','--porcelain','--untracked-files=no'):
        raise RuntimeError('tracked source is dirty: production source must be the locked commit')
    workers=default_workers() if a.workers is None else a.workers
    if not 1<=workers<=default_workers():raise ValueError('workers must be <= min(24, cpu_count)')
    champion=find_champion(a.champion)
    before=cfv.digest(champion)
    if before!=a.expected_champion_sha256:
        raise RuntimeError('champion SHA256 mismatch: '+before+'; refusing an unverified or random blueprint')
    model,blob=load_checkpoint(champion,device='cpu')
    arch=model.config.model
    if (arch.d_model,arch.num_layers,arch.nhead)!=(192,4,6):raise RuntimeError('production requires the real 192-d, 4-layer, 6-head champion')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    gpu={}
    if not a.smoke:
        if not torch.cuda.is_available():raise RuntimeError('production requires CUDA; --smoke is the only CPU run mode')
        gpu={'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'cuda':torch.version.cuda}
        if '5090' not in gpu['name']:raise RuntimeError('expected RTX 5090, found '+gpu['name'])
        x=torch.randn(64,64,device='cuda',requires_grad=True);(x@x.T).sum().backward();torch.cuda.synchronize()
    device='cpu' if a.smoke else 'cuda'
    timestamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(a.out) if a.out else Path('/root/autodl-tmp/sprint24/evening_production')/(timestamp+'_'+commit[:8])
    out.mkdir(parents=True,exist_ok=False)
    identity=dict(commit=commit,champion=str(champion),champion_sha256=before,champion_update=blob.get('update'),
        arch=arch.__dict__,workers=workers,online=ONLINE,offline=OFFLINE,gpu=gpu,smoke=a.smoke)
    cfv.atomic_json(out/'run_identity.json',identity)
    def stage(name,**fields):
        if not a.allow_dirty and (git('rev-parse','HEAD')!=commit or git('status','--porcelain','--untracked-files=no')):
            raise RuntimeError('locked source changed during production run')
        cfv.atomic_json(out/'progress.json',dict(stage=name,elapsed_s=time.monotonic()-start,**fields))
        print('STAGE',name,json.dumps(fields),flush=True)
    try:
        stage('pilot_and_latency')
        b.phase_b(out/'pilot',4,4,1,iterations=2,chance_samples=1,device=device,heldout_n=2,workers=min(workers,2))
        timing=make_timing_package(champion,out/'pilot/cfv_nets.npz',out/'timing')
        pilot_latency=latency_gate(timing,out/'pilot_latency.json',64 if a.smoke else 256,4 if a.smoke else 12)
        nets,_=cfv.load_nets(out/'pilot/cfv_nets.npz')
        stage('throughput_calibration')
        n=max(4,workers*4) if not a.smoke else 4
        rates={}
        for label,street,it,weights,path in (
            ('river',3,OFFLINE['river_iterations'],None,None),
            ('turn',2,OFFLINE['turn_iterations'],nets,out/'pilot/cfv_river.npz')):
            t=time.monotonic()
            b.make_dataset(out/('calibration_'+label+'.npz'),street,n,61000000+street*10000,weights,it,OFFLINE['chance_samples'],workers,path)
            _,m=cfv.load_artifact(out/('calibration_'+label+'.npz'),'dataset',n)
            rates[label]=m['active_compute_s']/n*1.25
        t=time.monotonic();cn=max(4,workers*32) if not a.smoke else 4
        teacher_budget={'1':OFFLINE['teacher_iterations'],'2':OFFLINE['teacher_iterations'],'3':OFFLINE['teacher_river_iterations']}
        c.collect_parallel(champion,cn,70000000,out/'pilot/cfv_nets.npz',workers,False,teacher_budget,OFFLINE['teacher_chance_samples'])
        rates['rebel']=c.collect_parallel.last_compute_seconds/cn*1.25
        # Measure a full 50-hand packaged respond match, not per-solver microseconds.
        bundle=Bundle(timing['source'],timing['model']);t=time.monotonic()
        match=play_judge(bundle,1500000000,0,2 if a.smoke else 50,'TAG_Pro',keep_trace=False)
        match_seconds=time.monotonic()-t
        total_matches=3*(len(BENCHMARK)*50+len(TOURNAMENT)*10)*2
        eval_est=total_matches*match_seconds/workers
        plan=plan_counts(rates,a.hours,eval_est,time.monotonic()-start,workers,a.smoke)
        plan.update(calibration_match_seconds=match_seconds,estimated_evaluation_seconds=eval_est,offline=OFFLINE,online=ONLINE)
        cfv.atomic_json(out/'plan.json',plan)
        stage('cfv_training',plan=plan)
        stats=b.phase_b(out/'b',plan['river_n'],plan['turn_n'],plan['net_epochs'],
            chance_samples=1 if a.smoke else OFFLINE['chance_samples'],device=device,heldout_n=plan['holdout_n'],workers=workers,
            river_iters=2 if a.smoke else OFFLINE['river_iterations'],turn_iters=2 if a.smoke else OFFLINE['turn_iterations'])
        if not a.smoke:
            for name in ('river','turn'):
                metrics=stats[name]
                if (not math.isfinite(metrics['heldout_mae']) or
                    metrics['heldout_mae']>=metrics['heldout_constant_mae']*.95 or metrics['heldout_mae_pot_ratio']>.15):
                    raise RuntimeError('CFV_HOLDOUT_GATE_FAILED '+name+' '+json.dumps(metrics))
        stage('baseline_package_and_gate')
        deliver=Path(a.deliver) if a.deliver else Path('/root/autodl-tmp/sprint24/deliver')
        def package(checkpoint,role):
            if a.smoke:
                r=make_timing_package(checkpoint,out/'b/cfv_nets.npz',out/('internal_packages/'+role))
                return dict(r,version=999,release='V999_INTERNAL_PRODUCTION_GATE',smoke=True,
                            source_sha256=cfv.digest(r['source']),model_sha256=cfv.digest(r['model']))
            return pack_evening(checkpoint,out/'b/cfv_nets.npz',deliver,online_profile=ONLINE)
        releases={'baseline':package(champion,'baseline')}
        gates={'baseline':latency_gate(releases['baseline'],out/'baseline_latency.json',64 if a.smoke else 256,4 if a.smoke else 12)}
        stage('two_distillation_generations')
        cd=c.phase_c(champion,out/'c',out/'b/cfv_nets.npz',count=plan['rebel_n'],epochs=plan['policy_epochs'],device=device,
            iterations=2 if a.smoke else teacher_budget,chance_samples=1 if a.smoke else OFFLINE['teacher_chance_samples'],
            holdout_n=plan['holdout_n'],workers=workers)
        for generation in ('g1','g2'):
            releases[generation]=package(out/('c/u250_deep_rebel_'+generation+'.pt'),generation)
            gates[generation]=latency_gate(releases[generation],out/(generation+'_latency.json'),64 if a.smoke else 256,4 if a.smoke else 12)
        cfv.atomic_json(out/'releases.json',releases)
        evolution=policy_evolution(releases,16 if a.smoke else 64)
        cfv.atomic_json(out/'policy_evolution.json',evolution)
        if not a.smoke and any(not v['changed'] or v['preflop_max_tv']>1e-12 for v in evolution.values()):
            raise RuntimeError('POLICY_EVOLUTION_GATE_FAILED '+json.dumps(evolution))
        stage('parallel_zip_evaluation')
        report=evaluate(releases,out/'evaluation',workers,1 if a.smoke else 50,1 if a.smoke else 10,2 if a.smoke else 50)
        if report['runtime_ms']['mean']>=40 or report['runtime_ms']['p99']>=120:
            raise RuntimeError('full evaluation latency gate failed: '+json.dumps(report['runtime_ms']))
        if cfv.digest(champion)!=before:raise RuntimeError('source champion changed during run')
        result=dict(status='COMPLETE',identity=identity,plan=plan,cfv=stats,distillation=cd,releases=releases,latency=gates,
                    evaluation=report,policy_evolution=evolution,elapsed_s=time.monotonic()-start)
        cfv.atomic_json(out/'FINAL_REPORT.json',result)
        # Final reviewable delivery index; each upload directory stays exactly three files.
        lines=['版本：'+', '.join('V%d'%r['version'] for r in releases.values())+'（内部 R1），release='+', '.join(r['release'] for r in releases.values()),'',
               'commit='+commit,'Champion SHA256='+before,'',json.dumps(releases,indent=2),
               '', 'All packaged respond latency gates passed. No upload or deployment was performed.']
        (out/'DELIVERY.md').write_text('\n'.join(lines)+'\n')
        stage('COMPLETE',report=str(out/'FINAL_REPORT.json'))
        return 0
    except BaseException as exc:
        cfv.atomic_json(out/'FAILED.json',dict(stage_error=type(exc).__name__,message=str(exc),elapsed_s=time.monotonic()-start,commit=commit))
        stage('FAILED',error=str(exc));raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expected-champion-sha256',default=EXPECTED_CHAMPION_SHA256)
    p.add_argument('--champion',type=Path);p.add_argument('--out',type=Path);p.add_argument('--deliver',type=Path)
    p.add_argument('--workers',type=int);p.add_argument('--hours',type=float,default=3.)
    p.add_argument('--smoke',action='store_true');p.add_argument('--allow-dirty',action='store_true')
    p.add_argument('--_run',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 2<=a.hours<=3.5:p.error('--hours must be between 2 and 3.5')
    if a._run:return run(a)
    # Own process group contains only this run and its workers. Never touch other runs.
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--_run'],start_new_session=True)
    try:return child.wait(timeout=3.5*3600)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise SystemExit('PRODUCTION_WATCHDOG_STOPPED: 3.5-hour deadline or interruption')

if __name__=='__main__':raise SystemExit(main())

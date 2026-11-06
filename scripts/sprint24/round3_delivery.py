#!/usr/bin/env python3
"""Adopt the live Round 3 turn job and deliver evaluation packages before distillation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

HOUR = 3600
RESERVE_SECONDS = 3*HOUR
HEARTBEAT_SECONDS = 30*60
PACKAGE_SECONDS = 10*60
DISTILL_ESTIMATE_SECONDS = 20*60
SCRIPT = Path(__file__).resolve()


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as handle:
        json.dump(value,handle,indent=2,sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def budget(started, hours):
    end = started+hours*HOUR
    return end-RESERVE_SECONDS,end


def allow_second(now, cutoff, qualified):
    return qualified and now+DISTILL_ESTIMATE_SECONDS<=cutoff


def bootstrap(round3, prior):
    runtime = round3/'runtime_checkout'
    pre = round3.parent/'cfv_turnfix_8014688'
    paths = [str(p) for p in (runtime/'src',runtime/'scripts/sprint24',round3,prior,pre)]
    if not all(Path(p).is_dir() for p in paths):
        raise ValueError('missing Round 3 runtime or retained stage code')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PYTHONPATH']=os.pathsep.join(paths)
    sys.path[:0]=paths


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
    except FileNotFoundError:
        return None


def owned_pid(round3, name):
    marker = round3/(name+'.pid')
    if not marker.exists():
        return None
    pid = int(marker.read_text())
    if process_state(pid) in (None,'Z'):
        return None
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\x00')
    except FileNotFoundError:
        return None
    if os.fsencode(round3) not in cmd or not any(Path(os.fsdecode(a)).name==name+'.py' for a in cmd if a):
        raise ValueError(f'PID {pid} is no longer owned by {name}')
    return pid


def stop_owned(round3, name):
    pid = owned_pid(round3,name)
    if pid is None:
        return
    def descendants(parent):
        try:
            children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
        except FileNotFoundError:
            return []
        return [int(child) for child in children]+[d for child in children for d in descendants(int(child))]
    # Freeze the controller briefly so it cannot spawn another pool while its
    # existing workers are enumerated. It is resumed only to handle termination.
    try:
        os.kill(pid,signal.SIGSTOP)
    except ProcessLookupError:
        return
    for child in [pid]+descendants(pid):
        try:
            os.kill(child,signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid,signal.SIGCONT)
    except ProcessLookupError:
        pass


def latest_match_count(folder):
    return sum(1 for f in folder.glob('pair_*.json') if f.is_file())


class Status:
    def __init__(self, path, round3, cutoff, deadline):
        self.path,self.round3,self.cutoff,self.deadline = path,round3,cutoff,deadline
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.fields = {}
        self.arena = None
        self.thread = threading.Thread(target=self.run,daemon=True)

    def publish(self):
        with self.lock:
            now = time.time()
            labels = load(self.round3/'TURN_RELABEL_STATUS.json',{})
            done,target = labels.get('rows_done',0),labels.get('total_rows',0)
            elapsed = labels.get('session_elapsed_s',0)
            labels_eta = (target-done)*elapsed/done if done and target>=done else None
            old = load(self.round3/'POST_RIVER_STATUS.json',{})
            result = dict(self.fields,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          training_cutoff_epoch=self.cutoff,deadline_epoch=self.deadline,
                          training_cutoff_seconds_left=max(0,self.cutoff-now),
                          total_seconds_left=max(0,self.deadline-now),
                          relabel=dict(done=done,target=target,eta_seconds=labels_eta,status=labels.get('status')),
                          upstream_stage=old.get('status'))
            if self.fields.get('stage')=='WAITING_FOR_FIRST_CFVS':
                split = 'final' if old.get('status')=='LOCKED_TURN_CANDIDATE_FINAL_GATE' else 'selection'
                kept = self.round3/split/'kept/rows.npz.manifest.json'
                log = self.round3/'round3_fit.log'
                counts = {}
                if kept.exists():
                    counts = load(kept)['counts']
                elif log.exists():
                    for name,number in re.findall(
                            rf'^TURN_EVAL_ACCEPTED {split} (on_policy|dirichlet) (\d+)',
                            log.read_text(errors='replace'),re.M):
                        counts[name]=int(number)
                labeled=sum(counts.values())
                marker=self.round3/'POST_RIVER_STATUS.json'
                split_elapsed=max(0,now-marker.stat().st_mtime) if marker.exists() else 0
                split_eta=(2048-labeled)*split_elapsed/labeled if labeled else None
                result['evaluation_labels']=dict(split=split,done=labeled,target=2048,by_source=counts,
                                                 eta_seconds=split_eta)
                candidates=load(self.round3/'TURN_SELECTION.json',{}).get('candidates',{})
                if old.get('status')=='FITTING_TURN_CANDIDATES':
                    remaining=2-len(candidates)
                    known=sum(c['fit_or_restore_wall_s'] for c in candidates.values())
                    result['training_progress']=dict(done=len(candidates),target=2,
                        eta_seconds=remaining*known/len(candidates) if candidates else None)
            if self.arena:
                folder,start,target = self.arena
                matches = latest_match_count(folder)
                result['arena_progress']=dict(matches_done=matches,target=target,
                    eta_seconds=(target-matches)*(now-start)/matches if 0<matches<target else None)
            if self.fields.get('stage')=='RUNNING_OPTIONAL_GENERATION_2':
                generation=self.path.parent/'gen2_distillation'
                job=load(generation/'JOB.json')
                if job:
                    plan=job['plan']
                    target=plan['rebel_n']+5*plan['holdout_n']
                    done=sum(load(p)['samples'] for p in generation.glob('model.*.npz.manifest.json'))
                    elapsed=max(0,now-(generation/'JOB.json').stat().st_mtime)
                    result['generation_2_labels']=dict(done=done,target=target,
                        eta_seconds=(target-done)*elapsed/done if 0<done<=target else None)
            atomic(self.path,result)

    def set(self, stage, *, arena=None, **fields):
        with self.lock:
            self.fields=dict(stage=stage,**fields)
            self.arena=arena
        self.publish()

    def run(self):
        while not self.finished.wait(HEARTBEAT_SECONDS):
            self.publish()

    def close(self):
        self.finished.set()
        self.thread.join()


def subprocess_until(command, deadline, log):
    left = deadline-time.time()
    if left<=0:
        return dict(returncode=None,timed_out=True)
    with log.open('ab') as handle:
        proc = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return dict(returncode=proc.wait(timeout=left),timed_out=False)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                os.killpg(proc.pid,signal.SIGTERM)
                proc.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            if isinstance(error,KeyboardInterrupt):
                raise
            return dict(returncode=proc.returncode,timed_out=True)


def child_command(args, mode, job):
    return [sys.executable,'-u',str(SCRIPT),'--round3',str(args.round3),
            '--prior',str(args.prior),'--worker',mode,'--job',str(job)]


def run_worker(args):
    payload = load(args.job)
    if args.worker=='arena':
        from round3_delivery_arena import run
        run(args.round3,payload['prep'],args.job.parent,payload['workers'])
        return
    from alphanl.stage2 import cfv
    from overnight_distill import one_generation
    from server_arena import assert_production_cfv
    from server_resources import core_plan, pin
    import torch
    nets = Path(payload['nets'])
    assert_production_cfv(nets)
    if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('RTX 5090 unavailable for optional generation')
    pin(core_plan(1)['worker_cpus'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint = args.job.parent/'model.pt'
    result = one_generation(Path(payload['blueprint']),checkpoint,Path(payload['champion']),
                            nets,(2**32)*333401+100000000,payload['plan'],24,'cuda')
    atomic(args.job.parent/'DISTILL_RESULT.json',
           dict(binding=payload,accepted=result['accepted'],checkpoint=str(checkpoint),
                checkpoint_sha256=cfv.digest(checkpoint),validation=result))


def wait_for_cfvs(round3, work, cutoff, status):
    receipt = work/'INPUT_CUTOFF.json'
    if receipt.exists():
        return load(receipt)['reason']
    status.set('WAITING_FOR_FIRST_CFVS')
    while not (round3/'FINAL_TURN_GATE.json').exists():
        phase = load(round3/'POST_RIVER_STATUS.json',{}).get('status','')
        if time.time()>=cutoff:
            reason='budget cutoff before completed turn gate'
            break
        if phase.startswith(('FAILED','STOPPED')) or owned_pid(round3,'round3_fit') is None:
            reason='turn pipeline ended before completed gate'
            break
        time.sleep(min(20,max(.1,cutoff-time.time())))
    else:
        reason=None
    if reason:
        # A gate receipt is written immediately before a bounded single NumPy
        # calculation. Let that existing call finish if already in progress.
        if (round3/'FINAL_TURN_HOLDOUT_USED.json').exists():
            for _ in range(20):
                if (round3/'FINAL_TURN_GATE.json').exists() or owned_pid(round3,'round3_fit') is None:
                    break
                time.sleep(.25)
        if not (round3/'FINAL_TURN_GATE.json').exists():
            stop_owned(round3,'round3_fit')
            stop_owned(round3,'relabel')
            checkpoint=dict(reason=reason,relabel=load(round3/'TURN_RELABEL_STATUS.json'),
                stage=load(round3/'POST_RIVER_STATUS.json'),selection=load(round3/'TURN_SELECTION.json'),
                persisted_shard_manifests=sorted(str(p) for p in
                    (round3/'turn_relabel_shards').glob('*/*/rows.npz.manifest.json')))
            atomic(work/'BUDGET_CHECKPOINT.json',checkpoint)
        else:
            reason=None
    atomic(receipt,dict(reason=reason))
    return reason


def deliver(args, work, run_id, generation, checkpoint, nets, decision, deadline, status):
    from round3_delivery_arena import PAIRS, summary
    from round3_delivery_artifacts import ARMS, prepared, publish, reserve
    from server_arena import profiles
    folder = work/f'g{generation}'
    folder.mkdir(exist_ok=True)
    receipt = folder/'DELIVERED.json'
    if receipt.exists():
        from alphanl.stage2 import cfv
        result=load(receipt)
        for entry in result.values():
            for key in ('source','model'):
                if cfv.digest(entry[key])!=entry[key+'_sha256']:
                    raise ValueError('published package changed: '+entry[key])
        return result
    reservation = reserve(args.deliver,run_id,generation)
    assembled = {}
    status.set('PREPARING_EXACT_BYTE_ARCHIVES',generation=generation,versions=reservation['versions'])
    for arm in ARMS:
        arm_work=folder/arm
        arm_work.mkdir(exist_ok=True)
        assembled[arm]=prepared(args.deliver,arm_work,reservation,arm,checkpoint,nets,profiles()[arm])
    result = {}
    runtime = args.round3/'runtime_checkout'
    benchmark = runtime/'scripts/sprint24/server_benchmark.py'
    for index,arm in enumerate(ARMS):
        arm_work=folder/arm
        prep=assembled[arm]
        evaluation=arm_work/'ARENA_RESULT.json'
        latency_path=arm_work/'EXACT_BYTE_RUNTIME.json'
        if not evaluation.exists():
            status.set('RUNNING_PAIRED_V22_ARENA',generation=generation,arm=arm,
                       arena=(arm_work,time.time(),PAIRS*2))
            left=max(0,deadline-PACKAGE_SECONDS-time.time())
            arm_end=time.time()+left/(len(ARMS)-index)
            job=arm_work/'ARENA_JOB.json'
            atomic(job,dict(prep=prep,workers=args.arena_workers))
            # Save up to four minutes of this arm's slice for the exact-byte
            # runtime check. A failed or timed-out slice still publishes.
            run=subprocess_until(child_command(args,'arena',job),arm_end-240,arm_work/'arena.log')
            row=summary(arm_work,prep)
            atomic(evaluation,dict(row,worker_outcome=run))
            if decision['qualified'] and row['significant_win_rate_improvement'] and args.corpus.is_file():
                cmd=[sys.executable,str(benchmark),'--zip',prep['source'],'--model',prep['model'],
                     '--out',str(latency_path),'--corpus',str(args.corpus),'--core','1']
                subprocess_until(cmd,arm_end,arm_work/'runtime.log')
        arena=load(evaluation)
        latency=load(latency_path,dict(gate=dict(hard_passed=False),reason='runtime qualification unavailable'))
        status.set('PUBLISHING_ARENA_PACKAGE',generation=generation,arm=arm)
        result[arm]=publish(prep,arm_work,decision,arena,latency,
                           runtime/'scripts/sprint24/verify_evening_release.py')
    atomic(receipt,result)
    return result


def execute(args):
    from alphanl.stage2 import cfv
    from fit_river import source_for
    from round3_delivery_artifacts import BASELINE_RELEASE, select_pair

    work=args.round3/'delivery_first'
    work.mkdir(exist_ok=True)
    lock=(work/'.delivery.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=load(args.round3/'PLAN.json')
    hours=args.max_hours if args.max_hours is not None else plan['max_hours']
    if hours<=3:
        raise ValueError('MAX_HOURS must exceed the three-hour delivery reservation')
    # The PID receipt was written when the existing relabel began. A later
    # handoff or restart must not grant a fresh 24-hour budget.
    identity_path=work/'RUN_IDENTITY.json'
    existing=load(identity_path)
    started=existing['started_epoch'] if existing else (args.round3/'relabel.pid').stat().st_mtime
    identity=dict(started_epoch=started,max_hours=hours,round3=str(args.round3),prior=str(args.prior),
                  deliver=str(args.deliver),plan_sha256=cfv.digest(args.round3/'PLAN.json'),
                  code_commit=args.expected_commit)
    if existing is not None and existing!=identity:
        raise ValueError('delivery resume identity changed')
    if existing is None:
        atomic(identity_path,identity)
    cutoff,deadline=budget(started,hours)
    run_id='round3-delivery-first:'+str(args.round3)+':'+identity['plan_sha256']
    status=Status(work/'STATUS.json',args.round3,cutoff,deadline)
    status.thread.start()
    try:
        status.set('ADOPTING_ROUND3')
        from arena_v22 import reference_paths
        reference_paths(args.round3)
        archived_turn=load(args.prior/'FINAL_TURN_GATE.json')
        archived_river=load(args.prior/'FINAL_RIVER_GATE.json')
        if (cfv.digest(args.prior/'turn_candidate/weights.npz')!=archived_turn['weights_sha256']
                or cfv.digest(args.prior/'cfv_river.npz')!=archived_river['weights_sha256']
                or archived_turn['teacher_sha256']!=archived_river['weights_sha256']):
            raise ValueError('required non-fixture fallback bytes do not bind')
        stop_owned(args.round3,'round3_continue')
        stop_owned(args.round3,'arena_v22')
        reason=wait_for_cfvs(args.round3,work,cutoff,status)
        nets,decision=select_pair(args.round3,args.prior,work,reason)
        blueprint=source_for(args.prior)
        first=deliver(args,work,run_id,1,blueprint,nets,decision,deadline,status)
        second=None
        second_status='not started — insufficient time or CFV gate failed'
        dest=work/'gen2_distillation'
        previous=load(dest/'DISTILL_RESULT.json')
        completed_second=(work/'g2/DELIVERED.json').is_file()
        if previous or (not (dest/'WORKER_EXIT.json').is_file() and allow_second(time.time(),cutoff,decision['qualified'])):
            dest.mkdir(exist_ok=True)
            if previous is None:
                status.set('RUNNING_OPTIONAL_GENERATION_2')
                production=load(args.production_report)
                champion=Path(production['identity']['champion'])
                if cfv.digest(champion)!=production['identity']['champion_sha256']:
                    raise ValueError('archived champion checkpoint changed')
                job=dest/'JOB.json'
                payload=dict(nets=str(nets),blueprint=str(blueprint),champion=str(champion),
                             plan=production['plan'])
                atomic(job,payload)
                execution=subprocess_until(child_command(args,'distill',job),cutoff,dest/'distill.log')
                atomic(dest/'WORKER_EXIT.json',execution)
                previous=load(dest/'DISTILL_RESULT.json')
                second_status='stopped at cutoff or did not finish'
            if previous and previous['accepted'] and decision['qualified'] and (completed_second or time.time()<deadline-PACKAGE_SECONDS):
                checkpoint=Path(previous['checkpoint'])
                if cfv.digest(checkpoint)!=previous['checkpoint_sha256']:
                    raise ValueError('generation-2 checkpoint changed')
                second=deliver(args,work,run_id,2,checkpoint,nets,decision,deadline,status)
                second_status='delivered'
            elif previous:
                second_status=('not delivered — generation-2 validation failed' if not previous['accepted']
                               else 'not delivered — delivery deadline reached')
        promoted=[p for generation in (first,second or {}) for p in generation.values() if p['promoted']]
        recommended=(max(promoted,key=lambda p:p['arena']['candidate_minus_v22_win_rate']['mean'])['release']
                     if promoted else BASELINE_RELEASE)
        all_arms=[p for generation in (first,second or {}) for p in generation.values()]
        terminal=('COMPLETE' if all(p['arena']['matches']==p['arena']['target_matches'] for p in all_arms)
                  else 'COMPLETE_WITH_PARTIAL_ARENA')
        result=dict(status=terminal,cfv=decision,generation_1=first,generation_2=second,
                    generation_2_status=second_status,recommended_deployment=recommended)
        atomic(work/'FINAL_DELIVERY.json',result)
        status.set(terminal,recommended_deployment=recommended,result=str(work/'FINAL_DELIVERY.json'))
    except BaseException as exc:
        status.set('FAILED',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        status.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round3',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--deliver',type=Path)
    parser.add_argument('--max-hours',type=float)
    parser.add_argument('--expected-commit')
    parser.add_argument('--arena-workers',type=int,default=24,choices=range(1,25))
    parser.add_argument('--worker',choices=('arena','distill'),help=argparse.SUPPRESS)
    parser.add_argument('--job',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.round3=args.round3.resolve()
    root=args.round3.parent
    args.prior=(args.prior or root/'cfv_fix2_8014688').resolve()
    args.deliver=(args.deliver or root/'deliver').resolve()
    production=root/'overnight_server/cfv_full_8014688'
    args.production_report=production/'FINAL_REPORT.json'
    args.corpus=production/'stages/frozen_pr85_server_benchmark/real_match_states.json'
    if not args.worker:
        repo=SCRIPT.parents[2]
        actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        if actual!=args.expected_commit:
            raise ValueError('run this supervisor from the requested locked commit')
        subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD','--'],check=True)
    bootstrap(args.round3,args.prior)
    if args.worker:
        run_worker(args)
    else:
        execute(args)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Remote-only preflight and supervised recovery. Never invoke locally for tests.

Run under nohup on Xeon/5090. All expensive validation is intentionally here.
The locked checkout is immutable; stages and logs live outside it.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/sprint24')]
from alphanl.stage2 import cfv
from server_resources import core_plan,SERVER_TARGET_HOURS,SERVER_WATCHDOG_SECONDS
from cfv_recovery import OLD_ROOT,OLD_SOURCE


CUDA_PROBE = '''
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
name = torch.cuda.get_device_name(0)
print(name, torch.__version__, flush=True)
assert '5090' in name, 'RTX 5090 required'
x = torch.randn(64, 64, device='cuda', requires_grad=True)
(x @ x.T).sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all().item(), 'nonfinite CUDA gradients'
print('CUDA_FORWARD_BACKWARD_OK', flush=True)
'''


def kernel_wait(pid):
    proc = Path('/proc') / str(pid)
    try:
        state = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
        return (proc / 'wchan').read_text().strip() if state == 'D' else None
    except OSError:
        return None


def stop_child(child, isolated, grace=5):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if isolated:
                os.killpg(child.pid, sig)
            else:
                child.send_signal(sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace)
            return True
        except subprocess.TimeoutExpired:
            continue
    return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hours',type=float,default=SERVER_TARGET_HOURS)
    p.add_argument('--skip-pytest',action='store_true',help='use only the 10-second CUDA smoke check')
    p.add_argument('--full-precision',action='store_true',help='lock offline 64/32 x4, full V23 recovery counts and all four arms')
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--inference-core',type=int,default=1)
    p.add_argument('--resume',action='store_true',help='require an existing production checkpoint; identity must match')
    p.add_argument('--champion',type=Path,default=Path('/root/autodl-tmp/sprint24/ladder/u250_ladder_slayer.pt'))
    p.add_argument('--deliver',type=Path,default=Path('/root/autodl-tmp/sprint24/deliver'))
    p.add_argument('--reuse-evening-run',type=Path,default=Path(OLD_ROOT))
    p.add_argument('--reuse-source',type=Path,default=Path(OLD_SOURCE))
    a=p.parse_args(argv);out=a.out.resolve()
    if not 0<a.hours<=SERVER_TARGET_HOURS:p.error('hours must be >0 and <=14.5')
    def lock_source():
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
        if head!=a.expected_commit or dirty:raise RuntimeError('SOURCE_LOCK_FAILED')
    lock_source()
    if out.is_relative_to(REPO):raise ValueError('output must be outside checkout')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.launcher.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume and not (out/'CHECKPOINTS.json').exists():raise ValueError('--resume requires CHECKPOINTS.json; do not resume old 408a2ff run')
    resources=core_plan(a.inference_core);cfv.atomic_json(out/'LAUNCH_CPU_RESERVATION.json',resources)
    env=dict(os.environ,PYTHONPATH=str(REPO/'src')+os.pathsep+str(REPO/'scripts/sprint24'))
    validation=out/'REMOTE_PYTEST.json'
    # Log each test without the in-process timer that crashed while dumping
    # the stack of a long-running portable evaluator test on the server.
    command=[sys.executable,'-u','-m','pytest','-vv']
    launch=dict(commit=a.expected_commit,resources=resources,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                command=command,status='REMOTE_PREFLIGHT')
    def status(state,**extra):
        launch.update(status=state,**extra);cfv.atomic_json(out/'LAUNCH_STATUS.json',launch)
    def run(command,log,timeout,isolated=True):
        # Reserve inference core AND its SMT sibling during tests/training.
        command=(['taskset','-c',','.join(map(str,resources['worker_cpus']))]+command) if isolated else command
        with Path(log).open('a',buffering=1) as handle:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,
                                   start_new_session=isolated)
            started=time.monotonic()
            wait_channel=None
            wait_started=started
            try:
                while child.poll() is None:
                    now=time.monotonic()
                    channel=kernel_wait(child.pid) if isolated else None
                    if channel!=wait_channel:
                        wait_channel,wait_started=channel,now
                    stuck=channel is not None and now-wait_started>=120
                    status(launch['status'],child_pid=child.pid,kernel_wait=channel,
                           heartbeat_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    if now-started>=timeout or stuck:
                        reason=f'continuous kernel wait ({channel})' if stuck else 'wall-clock timeout'
                        stopped=stop_child(child,isolated,grace=5 if isolated else 20)
                        handle.write(f'\n{reason}: PID {child.pid}, exited={stopped}\n')
                        return 124
                    try:child.wait(timeout=min(10,max(0.01,timeout-(now-started))))
                    except subprocess.TimeoutExpired:pass
                return child.returncode
            except BaseException:
                if child.poll() is None:
                    stop_child(child,isolated,grace=5 if isolated else 20)
                raise
    def final_failure(state,reason):
        report=dict(status=state,error=reason,commit=a.expected_commit,remote_validation=str(validation),launch=str(out/'LAUNCH_STATUS.json'))
        for name in ('FINAL_REPORT.json','OVERNIGHT_FINAL_REPORT.json'):cfv.atomic_json(out/name,report)
        status(state,error=reason)
        return 1
    # Archive completed V23-V25 reports before doing any new work. Never overwrite
    # originals or copy NPZs into the locked checkout.
    archive=out/'previous_completed_reports';archive.mkdir(exist_ok=True)
    for path in a.reuse_evening_run.rglob('*.json'):
        target=archive/path.relative_to(a.reuse_evening_run);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and cfv.digest(target)!=cfv.digest(path):raise ValueError('archived report changed')
        if not target.exists():shutil.copy2(path,target)
    existing=Path('/root/autodl-tmp/sprint24/code_handoffs/V23_V25_completed_reports_20261105.zip')
    if existing.exists():shutil.copy2(existing,archive/existing.name)
    cfv.atomic_json(archive/'ARCHIVE_SHA256.json',{str(f.relative_to(archive)):cfv.digest(f) for f in archive.rglob('*') if f.is_file() and f.name!='ARCHIVE_SHA256.json'})
    probe=[sys.executable,'-u','-c',CUDA_PROBE]
    status('REMOTE_CUDA_PROBE',command=probe)
    probe_started=time.monotonic()
    rc=run(probe,out/'remote_cuda_probe.log',10 if a.skip_pytest else 45)
    if rc:return final_failure('STOPPED_REMOTE_CUDA',f'RTX 5090 compute probe exited {rc}; see remote_cuda_probe.log.')
    if a.skip_pytest:
        validation=out/'REMOTE_SELF_CHECK.json'
        cfv.atomic_json(validation,dict(commit=a.expected_commit,pytest_skipped=True,
            cuda_probe_returncode=rc,seconds=time.monotonic()-probe_started))
    else:
        status('REMOTE_PREFLIGHT',command=command)
        receipt=json.loads(validation.read_text()) if validation.exists() else None
        if receipt is not None and receipt.get('commit')!=a.expected_commit:
            return final_failure('STOPPED_REMOTE_TESTS','Prior pytest receipt belongs to a different commit.')
        if receipt is None or receipt.get('returncode')==124:
            start=time.monotonic();rc=run(command,out/'remote_pytest.log',7200)
            cfv.atomic_json(validation,dict(commit=a.expected_commit,returncode=rc,command=command,seconds=time.monotonic()-start))
            if rc:return final_failure('STOPPED_REMOTE_TESTS','Full pytest failed; see remote_pytest.log.')
        elif receipt.get('returncode')!=0:
            return final_failure('STOPPED_REMOTE_TESTS','Prior full pytest failed; see remote_pytest.log.')
    lock_source()
    command=[sys.executable,str(REPO/'scripts/sprint24/overnight_production.py'),'--server-profile','--cfv-data-recovery',
        '--expected-commit',a.expected_commit,'--out',str(out),'--hours',str(a.hours),'--workers',str(a.workers),
        '--inference-core',str(a.inference_core),'--champion',str(a.champion),'--deliver',str(a.deliver),
        '--reuse-evening-run',str(a.reuse_evening_run),'--reuse-source',str(a.reuse_source)]
    if a.full_precision:command.append('--full-precision')
    # Do NOT taskset the controller to worker CPUs: it must first discover and
    # reserve inference CPU1 before pinning its own training/judge descendants.
    for attempt in range(3):
        arguments=command+(['--resume'] if (out/'CHECKPOINTS.json').exists() else [])
        status('PRODUCTION_RUNNING',command=arguments)
        rc=run(arguments,out/'production.log',SERVER_WATCHDOG_SECONDS+60,isolated=False)
        report_path=out/'OVERNIGHT_FINAL_REPORT.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        report['remote_validation']=json.loads(validation.read_text());report['launcher_attempt']=attempt+1
        for name in ('FINAL_REPORT.json','OVERNIGHT_FINAL_REPORT.json'):cfv.atomic_json(out/name,report)
        state=report.get('status','FAILED_NO_REPORT');status(state,returncode=rc,attempt=attempt+1)
        if rc==0:return 0
        # Stable mathematical/data/source failures must not loop on identical
        # seeds or weaken their gates. Transient process failures may resume.
        if state.startswith('STOPPED_') or state in ('WATCHDOG_FAILED','FAILED_SETUP') or 'SOURCE_LOCK' in report.get('error',''):
            return rc or 1
        if attempt<2:time.sleep(30)
    return rc or 1


if __name__=='__main__':raise SystemExit(main())

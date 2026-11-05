"""Run-owned CPU isolation; never changes host governor or other jobs."""
import os
from pathlib import Path


SERVER_TARGET_HOURS = 14.5
SERVER_WATCHDOG_SECONDS = 16 * 3600

def cpu_list(text):
    result=set()
    for part in text.strip().split(','):
        if not part:continue
        lo,sep,hi=part.partition('-')
        result.update(range(int(lo),int(hi)+1) if sep else [int(lo)])
    return result


def core_plan(core=None, allowed=None, sys_root='/sys/devices/system/cpu'):
    allowed=set(allowed if allowed is not None else os.sched_getaffinity(0))
    candidates=sorted(allowed-{0})
    if not candidates:raise RuntimeError('a nonzero dedicated inference core is required')
    core=candidates[0] if core is None else int(core)
    if core==0 or core not in allowed:raise ValueError('inference core must be allowed and not CPU 0')
    folder=Path(sys_root)/('cpu%d'%core)
    siblings_file=folder/'topology/thread_siblings_list'
    siblings=cpu_list(siblings_file.read_text()) if siblings_file.exists() else {core}
    worker=sorted(allowed-siblings)
    if not worker:raise RuntimeError('no CPUs remain for training/judge workers')
    governors={}
    for cpu in sorted(allowed):
        path=Path(sys_root)/('cpu%d'%cpu)/'cpufreq/scaling_governor'
        governors[str(cpu)]=path.read_text().strip() if path.exists() else 'unavailable'
    quota=Path('/sys/fs/cgroup/cpu.max')
    return dict(inference_core=core,inference_siblings=sorted(siblings),worker_cpus=worker,
                allowed_cpus=sorted(allowed),governors=governors,
                governor=governors[str(core)],cpu_quota=quota.read_text().strip() if quota.exists() else 'unavailable',
                isolation='run-owned processes only; host-wide exclusive reservation is not asserted')


def pin(cpus):
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,set(cpus))
        actual=set(os.sched_getaffinity(0))
        if actual!=set(cpus):raise RuntimeError('CPU affinity did not take effect')
        return sorted(actual)
    raise RuntimeError('server profile requires Linux sched_setaffinity')

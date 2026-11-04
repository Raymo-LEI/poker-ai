"""Bounded spawn pools: one BLAS thread per worker, no inherited CUDA context."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import multiprocessing as mp
import time


def default_workers():
    return min(24,os.cpu_count() or 24)


def worker_init(initializer=None,args=(),ready=None):
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if initializer is not None:initializer(*args)
    if ready is not None:ready.put(os.getpid())


def jobs(function,payloads,count,workers=None,initializer=None,initargs=(),timeout=180,label='POOL'):
    workers=default_workers() if workers is None else int(workers)
    if workers<1:raise ValueError('workers must be positive')
    start=time.monotonic();last=start;done=0
    ctx=mp.get_context('spawn')
    ready=ctx.Queue()
    with ctx.Pool(workers,initializer=worker_init,initargs=(initializer,initargs,ready),maxtasksperchild=256) as pool:
        for _ in range(workers):ready.get(timeout=120)
        it=pool.imap_unordered(function,payloads,chunksize=1)
        while done<count:
            try:result=it.next(timeout)
            except mp.TimeoutError:
                pool.terminate();raise TimeoutError(label+' worker stalled for '+str(timeout)+'s')
            done+=1;now=time.monotonic()
            if now-last>=10 or done==count:
                elapsed=now-start
                print(label,done,'/',count,'elapsed_s',round(elapsed,1),'eta_s',round(elapsed/max(done,1)*(count-done),1),flush=True)
                last=now
            yield result


def worker_environment(_):
    import torch
    time.sleep(.05)
    return dict(pid=os.getpid(),torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')})

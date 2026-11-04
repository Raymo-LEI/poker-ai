"""Layered local timings; never a target-platform latency promise."""
import hashlib
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from alphanl.stage2.export import _sources,BOTZONE_ENTRY
from alphanl.stage2.search_math import ShowdownKernel,DCFR
from alphanl.stage2.search_game import PublicState,build_tree
from alphanl.stage2.portable_evaluator import strength


def timings(fn,n):
    times=[]
    for _ in range(n):
        t=time.perf_counter();fn();times.append((time.perf_counter()-t)*1000)
    return dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),repeats=n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    board=['2s','3h','4d','9c','Ks'];kernel=ShowdownKernel(board)
    w=np.random.RandomState(41).random_sample(1081);w/=w.sum()
    keys=[strength(list(h)+board) for h in kernel.hands]
    valid=np.array([[not(set(h)&set(o)) for o in kernel.hands] for h in kernel.hands])
    matrix=np.array([[(x>y)+.5*(x==y) for y in keys] for x in keys])*valid
    dense=matrix.dot(w)/valid.dot(w);fast,_=kernel.equity(w)
    state=PublicState(1000,(9000,9000))
    def solve():
        tree=build_tree(state,100,kernel=kernel)
        return DCFR(tree,kernel=kernel,ranges=(w,w)).solve(24)
    source_paths=list((ROOT/'src/alphanl/stage2').glob('search_*.py'))+[ROOT/'src/alphanl/stage2/portable_runtime.py',ROOT/'src/alphanl/stage2/wizard.py',ROOT/'src/alphanl/stage2/export.py',ROOT/'botzone_bots/stage2/botzone_entry.py']
    result=dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},weights_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                                 threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}),
                kernel=timings(lambda:kernel.equity(w),100),kernel_max_dense_error=float(np.max(abs(fast-dense))),
                solve_warm_ranks=timings(solve,3),
                solve_definition='full1081,24 iterations,raise_cap2,includes tree; excludes model/history/rank build')
    # Cold solve includes rank/group construction in addition to the full tree.
    def cold():
        k=ShowdownKernel(board)
        return DCFR(build_tree(state,100,kernel=k),kernel=k,ranges=(w,w)).solve(24)
    result['solve_cold_ranks']=timings(cold,3)
    with np.load(a.weights,allow_pickle=False) as archive:
        meta=json.loads(archive['meta_json'].tobytes().decode())
    with tempfile.TemporaryDirectory(prefix='poker_math_process_') as folder:
        tree=Path(folder)
        sources=_sources(meta['numeric_version'],botzone=True)
        sources['runtime']='WEIGHT_DATA=None\nCONFIG=None\nSETTINGS=None\nSAFETY=False\n'+(ROOT/'src/alphanl/stage2/portable_runtime.py').read_text()
        pkg=tree/'stage2_runtime';pkg.mkdir();(pkg/'__init__.py').write_text('')
        for name,source in sources.items():
            (pkg/(name+'.py')).write_text(source)
        for name in ('__main__.py','botzone_entry.py','dataclasses_compat.py'):
            (tree/name).write_bytes((BOTZONE_ENTRY/name).read_bytes())
        req=dict(num_players=2,dealer_id=0,my_id=1,my_chips=19900,my_cards=[51,23],public_cards=[0,5,10,31,44],
                 history=[dict(round=s,player_id=p,action=0) for s in range(3) for p in (1,0)],
                 hand=0,max_hand=50,total_win_chips=[0,0],total_win_games=[0,0])
        env=dict(os.environ,POKER_STAGE2_WEIGHTS=str(a.weights.resolve()))
        outputs=[]
        def process():
            run=subprocess.run([sys.executable,str(tree/'__main__.py')],input=json.dumps({'requests':[req],'responses':[]}),
                               text=True,capture_output=True,env=env,check=True,timeout=120)
            output=json.loads(run.stdout)
            if 'search=active' not in output.get('debug','') or output['data']['search_audit']['status']!='active':
                raise RuntimeError('full process failed search acceptance: '+run.stdout+run.stderr)
            outputs.append(output)
        result['process_model_history_decision']=timings(process,3)
        result['process_evidence']=dict(debug=outputs[-1]['debug'],search_audit=outputs[-1]['data']['search_audit'],
                                        response=outputs[-1]['response'],weights=str(a.weights),
                                        source='development_unreleased from this checkout; not the audited V15 ZIP')
    with a.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    main()

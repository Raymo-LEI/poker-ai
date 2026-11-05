"""Under-30-second local smoke; no full pytest, benchmark, dataset or training run."""
import sys
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/sprint24')]
from alphanl.stage2 import cfv
from evening_pipeline import sample_spot
from overnight_data import label,TREE
from cfv_recovery import require_floor,RecoveryStop,decode_state,assert_disjoint,old_artifact,choose_plan,function_hash,OLD_COMMIT


def main():
    started=time.monotonic()
    relative='src/alphanl/stage2/deep_solver_v2.py'
    before=subprocess.check_output(['git','show',OLD_COMMIT+':'+relative],cwd=ROOT,text=True)
    after=(ROOT/relative).read_text()
    for function in ('pack_many','_public','expand_range','bucket_values'):
        assert function_hash(before,function)==function_hash(after,function),function
    require_floor({'river':80000,'turn':50000})
    try:require_floor({'river':8192,'turn':4352})
    except RecoveryStop:pass
    else:raise AssertionError('starved data allowed')
    rows=[]
    for seed in (12,90):
        s,h,o=sample_spot(seed,3)
        first=0 if s.me_seat==getattr(s,'postflop_first_seat',0) else 1
        x=cfv.spot_features(s.board_cards,h,o,s.pot,s.to_call,s.me_stack,100.,3,s.opp_stack,first).astype(np.float32)
        state,rh,ro=decode_state(x)
        assert np.allclose(x[:32],rh) and state.street==3
        rows.append(x)
    assert assert_disjoint(np.array(rows[:1]),np.array(rows[1:]))['board_overlap']==0
    try:assert_disjoint(np.array(rows[:1]),np.array(rows[:1]))
    except ValueError:pass
    else:raise AssertionError('leakage allowed')
    # One real expanded-tree label, never the online timing benchmark.
    xx,y=label(*decode_state(rows[0]),None,1,1,12)
    assert xx.shape==(136,) and y.shape==(32,) and np.isfinite(y).all()
    with tempfile.TemporaryDirectory(prefix='cfv-recovery-smoke-') as folder:
        path=Path(folder)/'data.npz'
        meta=cfv.publish(path,dict(x=xx[None,:],y=y[None,:]),1,0,'dataset',dict(tree=TREE))
        assert old_artifact(path,meta['sources'],'dataset')[1]['samples']==1
        meta['schema']='old';cfv.atomic_json(str(path)+'.manifest.json',meta)
        try:old_artifact(path,meta['sources'],'dataset')
        except ValueError:pass
        else:raise AssertionError('incompatible schema allowed')
    a=dict(datasets={'river':{'samples':100608},'turn':{'samples':68096}})
    b={k:dict(selected=dict(labels_per_second=100.,iterations=32,chance_samples=4)) for k in ('river','turn')}
    rates=dict(river=.05,turn=.05,rebel=.005,anchor=.001,cfv_epoch=.000001,policy_epoch=.000001)
    p=choose_plan(a,b,rates,8.5,10,1,1,24)
    assert (p['river_n'],p['turn_n'])==(108800,72192)
    elapsed=time.monotonic()-started
    assert elapsed<25
    print('CFV_RECOVERY_SMOKE_PASS seconds=%.3f'%elapsed)

if __name__=='__main__':main()

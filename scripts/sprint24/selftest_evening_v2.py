"""Independent small end-to-end acceptance; never uses a live checkpoint/run.

Creates a fresh tiny policy, 4 river + 3 turn rows, two gated 4-row distill
updates, a new versioned ZIP, actual respond probes and held-out judge matches.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv


def requests():
    base={'num_players':2,'dealer_id':0,'my_id':0,'my_chips':19900,'my_cards':[51,23],
          'public_cards':[3,8,41],'hand':3,'max_hand':50,'total_win_chips':[1200,-1200],
          'total_win_games':[2,1],'history':[
              {'round':0,'player_id':1,'action':0,'action_type':'call'},
              {'round':0,'player_id':0,'action':0,'action_type':'check'},
              {'round':1,'player_id':1,'action':100,'action_type':'raise'}]}
    rows=[dict(base)]
    turn=dict(base,public_cards=base['public_cards']+[12],my_chips=19800)
    turn['history']=base['history']+[
        {'round':1,'player_id':0,'action':0,'action_type':'call'},
        {'round':2,'player_id':1,'action':100,'action_type':'raise'}]
    rows.append(turn)
    river=dict(turn,public_cards=turn['public_cards']+[25],my_chips=19700)
    river['history']=turn['history']+[
        {'round':2,'player_id':0,'action':0,'action_type':'call'},
        {'round':3,'player_id':1,'action':100,'action_type':'raise'}]
    rows.append(river)
    pre=dict(base,public_cards=[],my_id=1,my_chips=19950,history=[],hand=0,total_win_chips=[0,0])
    rows.append(pre)
    return rows


def subprocess_probes(packed,python=sys.executable):
    results=[]
    for request in requests():
        # Force the portable evaluator and block phevaluator import altogether.
        # Execute the real ZIP entry, including load_agent, release checks/debug.
        code='''import sys,runpy,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith('phevaluator'):raise ImportError('acceptance: no phevaluator')
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1],run_name='__main__')
'''
        env=dict(os.environ,POKER_STAGE2_WEIGHTS=packed['model'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        started=time.perf_counter()
        proc=subprocess.run([python,'-c',code,packed['source']],input=json.dumps({'requests':[request]})+'\n',
            text=True,capture_output=True,env=env,timeout=60)
        if proc.returncode:raise RuntimeError(proc.stderr)
        body=json.loads(proc.stdout)
        debug=body.get('debug','')
        assert debug.startswith('py') and ' ok ' in debug and 'fallback' not in debug and 'wiz=err' not in debug,debug
        assert packed['release'] in debug and Path(packed['model']).name in debug,debug
        if len(request['public_cards']) in (3,4):
            import re
            m=re.search('cfv=(\\d+)',debug)
            assert m and int(m.group(1))>0,debug
        results.append({'board_size':len(request['public_cards']),'response':body['response'],'debug':debug,
                        'elapsed_s':time.perf_counter()-started,'stderr':proc.stderr})
    return results


def main():
    import torch
    from alphanl.cfr.config import CFRConfig
    from alphanl.cfr.distillation import PolicyNetwork
    from alphanl.transformer.config import ModelConfig
    from alphanl.stage2.policy import ResidualExploiterPolicy
    from alphanl.stage2.train import save_checkpoint
    import evening_pipeline as b
    import evening_cd as c
    import evening_d as d
    from pack_wizard import pack_evening
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=REPO/'eval/evening_v2_repair/smoke')
    p.add_argument('--deliver',type=Path,default=REPO.parent);p.add_argument('--skip-units',action='store_true')
    p.add_argument('--hands',type=int,default=50)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20261104)
    report={}
    if not a.skip_units:
        result=subprocess.run([sys.executable,'-m','pytest','-q',str(REPO/'tests/test_evening_v2.py')],
            cwd=REPO,capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(REPO/'src')))
        (a.out/'unit.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        report['unit_tests']=result.stdout
    report['phase_b']=b.phase_b(a.out,4,3,1,iterations=2,chance_samples=1,heldout_n=2)
    print('SELFTEST_B_DONE',flush=True)
    config=CFRConfig(equity_features=True,equity_samples=2,
        model=ModelConfig(d_model=32,nhead=2,num_layers=1,dim_feedforward=64,max_seq_len=24,dropout=0.))
    model=ResidualExploiterPolicy(PolicyNetwork(config)).eval()
    source=a.out/'fresh_smoke_policy.pt';save_checkpoint(source,model)
    report['phase_c']=c.phase_c(source,a.out,a.out/'cfv_nets.npz',count=4,epochs=1,iterations=1,chance_samples=1,holdout_n=2)
    def bad_surgery(candidate):
        with torch.no_grad():
            candidate.delta_head.bias.copy_(torch.tensor([10.,-10.,8.,-8.,4.]))
        return candidate
    probe=c.one_generation(a.out/'u250_deep_rebel_g2.pt',a.out/'rollback_probe.pt',20281104,a.out/'cfv_nets.npz',
                           count=2,epochs=1,iterations=2,chance_samples=1,holdout_n=2,surgery_fn=bad_surgery)
    assert 'surgery_validation' in probe['rollback'],probe
    report['surgery_rollback_test']=probe
    print('SELFTEST_C_DONE',flush=True)
    packed=pack_evening(a.out/'u250_deep_rebel_g2.pt',a.out/'cfv_nets.npz',a.deliver,
                        iterations=1,chance_samples=1,smoke=True)
    report['release']=packed
    cfv.atomic_json(a.out/'release.json',packed)
    report['subprocess_respond']=subprocess_probes(packed)
    print('SELFTEST_RESPOND_DONE',flush=True)
    report['phase_d']=d.evaluate_bundle(packed['source'],packed['model'],a.out/'judge_report.json',1,a.hands)
    from evening_d import Bundle
    bundle=Bundle(packed['source'],packed['model'])
    # Independent final preflop parity: the search layer must leave the blueprint unchanged.
    r=requests()[-1];s,amounts=bundle.runtime.request_snapshot(r)
    ctx,_=bundle.runtime.context_from_payload({'requests':[r]})
    import numpy as np
    prior=bundle.agent.logits(bundle.agent.observation(s,ctx))[0]
    prior=np.exp(prior-prior.max());prior/=prior.sum()
    out,trace=bundle.modules['evening_strategy'].search_policy(s,ctx,prior,bundle.agent.weights,amounts,1,1)
    np.testing.assert_allclose(out,prior,atol=1e-12)
    with torch.no_grad():
        obs=bundle.agent.observation(s,ctx)
        expected=model({k:torch.as_tensor(v) for k,v in obs.items()})[0].numpy()[0]
        actual=bundle.agent.logits(obs)[0]
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
    report['preflop_source_logit_error']=float(np.max(np.abs(actual-expected)))
    report['preflop_search_parity']=float(np.max(np.abs(out-prior)))
    # Source syntax is compatible with 3.6 grammar. Native 3.6/NumPy1.19 and
    # Linux resource limits require an actual target interpreter, reported separately.
    import ast,zipfile
    with zipfile.ZipFile(packed['source']) as archive:
        for name in archive.namelist():
            if name.endswith('.py'):ast.parse(archive.read(name).decode(),feature_version=(3,6))
    report['python36_grammar']=True
    report['target_python36_numpy119_executed']=False
    report['linux_limits_enforced']=False
    report['status']='small_sample_pipeline_passed'
    cfv.atomic_json(a.out/'acceptance.json',report)
    print('SELFTEST_DONE',json.dumps({'release':packed,'report':str(a.out/'acceptance.json')}),flush=True)

if __name__=='__main__':main()

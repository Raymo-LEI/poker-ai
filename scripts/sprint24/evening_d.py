"""Evaluate versioned Botzone bundles through their actual respond() entry.

Native 50-hand match statistics retain both legs. Target judge runs use the
checked-in referee and cumulative requests/data, with no early lock truncation.
"""
import argparse
import importlib.util
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv


def summarize(rows):
    from alphanl.stage2.evaluation import summarize_match_pairs
    stats=summarize_match_pairs(rows)
    stats['win_rate']=stats['match_win']
    return stats


class Bundle:
    """Load the exact ZIP without resolving any module from another runtime."""
    def __init__(self,source,model):
        import types
        self.source=Path(source);self.model=Path(model)
        # Isolated package name permits comparing bundles without sys.path swaps.
        name='_evening_'+cfv.digest(source)[:12]
        package=types.ModuleType(name);package.__path__=[]
        sys.modules[name]=package
        with zipfile.ZipFile(source) as archive:
            from alphanl.stage2.export import _sources
            order=list(_sources(None,botzone=True))+['runtime']
            self.modules={}
            for short in order:
                filename='stage2_runtime/'+short+'.py'
                text=archive.read(filename).decode()
                module=types.ModuleType(name+'.'+short)
                module.__package__=name
                module.__file__=str(source)+'/'+filename
                import zipimport
                module.__loader__=zipimport.zipimporter(str(source))
                sys.modules[module.__name__]=module
                setattr(package,short,module)
                exec(compile(text,filename,'exec'),module.__dict__)
                self.modules[short]=module
        with np.load(model,allow_pickle=False) as blob:
            self.meta=json.loads(blob['meta_json'].tobytes())
            weights={k:np.array(blob[k]) for k in blob.files if k!='meta_json'}
        self.modules['cfv'].validate_embedded(weights,self.meta['cfv'])
        self.runtime=self.modules['runtime']
        self.agent=self.runtime.Agent(weights,self.meta['config'],self.meta['settings'],self.meta['numeric_version'],self.meta.get('deployment_guard'))
        self.agent.release_id=self.meta.get('deployment_release','development_unreleased')
        self.agent.decision_identity+=':'+self.agent.release_id
        self.agent.force_argmax=False

    def respond(self,payload):
        return self.runtime.respond(self.agent,payload,bool(self.meta.get('safety',False)))

    def debug(self):
        return 'py%s np%s ok %s release=%s %s'%(sys.version.split()[0],np.__version__,self.model.name,self.meta['deployment_release'],getattr(self.agent,'last_wizard',''))


def _fixture(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def play_judge(bundle,seed=0,hero_seat=0,max_hands=50,opponent='call',keep_trace=True):
    judge=_fixture('_evening_judge_'+str(seed),REPO/'tests/fixtures/botzone_judge.py')
    judge.random=random.Random(seed)
    requests={0:[],1:[]};responses={0:[],1:[]};data={0:'',1:''}
    log=[];initdata=None;times=[];traces=[];counts={};score=None;complete=False
    actor=None;opponent_hand=-1;cfv_calls=0
    if opponent not in ('call','fold'):
        from alphanl.cfr.evaluation import RuleActor
        from ladder_actors import SlayerActor
        actor=SlayerActor(opponent,seed=seed,equity_samples=16) if opponent in ('Bot17_Probe3Bet','NearShove_19949') else RuleActor(opponent,seed=seed,equity_samples=32)
    random.seed(seed*2+hero_seat)
    while True:
        query={'log':log}
        if initdata is not None:query['initdata']=initdata
        output=json.loads(judge.judge(json.dumps(query)))
        if output['display'].get('error'):raise RuntimeError('judge rejection: '+str(output['display']['error']))
        if output['command']=='finish':
            score=output['content'][str(hero_seat)];complete=True;break
        initdata=output.get('initdata',initdata)
        (seat,request),= [(int(k),v) for k,v in output['content'].items()]
        if request['hand']>=max_hands:
            score=request['total_win_chips'][hero_seat]/100.;break
        requests[seat].append(request)
        payload={'requests':requests[seat],'responses':responses[seat],'data':data[seat]}
        if seat==hero_seat:
            started=time.perf_counter();body=bundle.respond(payload);times.append(time.perf_counter()-started)
            response=body['response'];data[seat]=body.get('data','')
            trace=getattr(bundle.agent,'last_evening',{})
            street=len(request['public_cards'])
            counts[str(street)]=counts.get(str(street),0)+1
            cfv_calls+=trace.get('cfv_calls',0)
            if keep_trace and trace.get('search') and len(traces)<12:traces.append(dict(trace,debug=bundle.debug()))
        else:
            snap,options=bundle.runtime.request_snapshot(request)
            response=options[1] if 1 in options else -1
            if opponent=='fold' and snap.to_call>0:response=-1
            if actor is not None:
                if request['hand']!=opponent_hand:
                    actor.begin_hand(snap.me_seat);opponent_hand=request['hand']
                response=actor_response(bundle,snap,request,options,actor.act(snap))
        responses[seat].append(response)
        log.append({'output':output});log.append({str(seat):{'verdict':'OK','response':response}})
    return dict(hero_net_bb=score,complete=complete,hands=max_hands,hero_decisions=len(times),errors=0,
                max_turn_s=max(times) if times else 0,mean_turn_s=float(np.mean(times)) if times else 0,
                decisions_by_board_size=counts,traces=traces,cfv_calls=cfv_calls,turn_s=times)


def actor_response(bundle,snap,request,options,decision):
    from alphanl.transformer.actor_actions import PotFractionBet
    if isinstance(decision,PotFractionBet):
        desired=min(snap.me_stack,int(round(snap.to_call+decision.fraction*snap.pot)))
        view=bundle.runtime.judge.reconstruct(request)
        minimum=bundle.runtime.judge.min_raise_extra(view)
        if desired>=snap.me_stack:return -2
        if minimum is not None:
            desired=max(desired,minimum)
            if desired>snap.to_call and bundle.runtime.judge.judge_accepts(view,int(desired)):
                return int(desired)
        candidates={a:(snap.me_stack if r==-2 else min(snap.to_call,snap.me_stack) if a==1 else max(r,0)) for a,r in options.items()}
        action=min(candidates,key=lambda a:abs(candidates[a]-desired))
        return options[action]
    action=int(decision)
    if action not in options:raise ValueError('opponent selected unavailable judge button')
    return options[action]


def evaluate_bundle(source,model,out=None,match_pairs=1,max_hands=50,seed=880001):
    bundle=Bundle(source,model)
    rows=[];runs=[]
    for pair in range(match_pairs):
        legs=[]
        for seat in (1,0):
            r=play_judge(bundle,seed+pair,seat,max_hands)
            runs.append(r);legs.append(r['hero_net_bb'])
        rows.append(legs)
    stats=summarize(rows) if max_hands==50 else {'partial_protocol_only':True,'scores_bb':rows}
    report={'release':bundle.meta['deployment_release'],'source_sha256':cfv.digest(source),
            'model_sha256':cfv.digest(model),'stats':stats,'runs':runs,'entry':'ZIP runtime.respond',
            'platform_limits_enforced':False}
    if out is not None:cfv.atomic_json(out,report)
    return report


def tournament(source,model,**kwargs):
    return evaluate_bundle(source,model,**kwargs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--nets',type=Path)
    p.add_argument('--deliver',type=Path,default=REPO.parent);p.add_argument('--version',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--match-pairs',type=int,default=1)
    p.add_argument('--out',type=Path,default=REPO/'eval/evening_v2_repair/eval_report.json')
    a=p.parse_args()
    if a.checkpoint:
        from pack_wizard import pack_evening
        packed=pack_evening(a.checkpoint,a.nets,a.deliver,a.version,1 if a.smoke else None,1 if a.smoke else 2,a.smoke)
        a.zip,a.model=Path(packed['source']),Path(packed['model'])
        cfv.atomic_json(a.out.with_name('release.json'),packed)
    if a.zip is None or a.model is None:p.error('provide --zip/--model or --checkpoint/--nets')
    print(json.dumps(evaluate_bundle(a.zip,a.model,a.out,a.match_pairs,2 if a.smoke else 50)),flush=True)

if __name__=='__main__':main()

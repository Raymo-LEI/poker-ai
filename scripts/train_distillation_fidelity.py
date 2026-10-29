"""Opponent-augmented exact-mixture distillation with playing-strength selection."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from alphanl.cfr.distillation import PolicyNetwork,load_student,prepare_archive,save_student
from alphanl.cfr.fidelity import ExactTeacher,collect_pool_states,FidelityTrainer,fidelity_report
from alphanl.cfr.evaluation import RuleActor,duplicate_match
from eval_distilled_gauntlet import StudentActor


def evaluate_selection(model,*,pairs,temperature=1.,cutoff=0.):
    result={}
    for index,name in enumerate(('call','equity')):
        seed=711029+index*10000
        result[name]=duplicate_match(StudentActor(model,seed=seed+11,temperature=temperature,cutoff=cutoff),
            RuleActor(name,seed=seed+19,equity_samples=64),pairs=pairs,seed=seed,log=lambda _:None)
    return dict(score=sum(r['mbb_per_hand'] for r in result.values())/2,opponents=result,
                pairs_per_opponent=pairs,inference=dict(temperature=temperature,cutoff=cutoff),seed=711029)


def bake_temperature(model,temperature):
    """Constant logit temperature is absorbed into the existing linear head."""
    baked=copy.deepcopy(model).eval()
    with torch.no_grad():
        baked.backbone.policy_head.weight.div_(temperature)
        baked.backbone.policy_head.bias.div_(temperature)
    return baked


def acceptance_gate(result):
    """Never promote a development budget, incomplete report or hidden transform."""
    expected={'random','call','maniac','overbet','nit','equity'}
    if not result.get('complete') or set(result.get('opponents',{}))!=expected:
        return False
    if result.get('inference',dict(temperature=1.,cutoff=0.))!=dict(temperature=1.,cutoff=0.):
        return False
    scores={}
    for name,row in result['opponents'].items():
        student=row['student']
        if student['pairs']!=2500 or student['hands']!=5000 or not row.get('identical_deals_verified'):
            return False
        score=student['mbb_per_hand']
        if not math.isfinite(score):return False
        scores[name]=score
    mean=sum(scores.values())/6
    if not math.isclose(mean,result['student_uniform_mean_mbb_per_hand'],abs_tol=1e-6):return False
    return mean>2000 and scores['call']>1800 and scores['equity']>-3300


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategies',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--teacher-report',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--train-hands',type=int,default=20000)
    p.add_argument('--validation-hands',type=int,default=3000)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--check-every',type=int,default=500)
    p.add_argument('--selection-pairs',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--candidates',default='warm_pruned,warm_raw')
    p.add_argument('--dataset',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--stage2-template',type=Path)
    p.add_argument('--stage2-out',type=Path)
    p.add_argument('--stage2-updates',type=int,default=300)
    a=p.parse_args()
    if any(getattr(a,k)<1 for k in ('train_hands','validation_hands','steps','check_every','selection_pairs','batch_size','stage2_updates')):
        p.error('positive budgets required')
    if bool(a.stage2_template)!=bool(a.stage2_out):p.error('provide both Stage 2 paths or neither')
    a.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    config,manifest=prepare_archive(a.strategies,after=149,through=350)
    teacher_report=json.loads(a.teacher_report.read_text())
    if teacher_report['settings']['pairs']!=2500:p.error('acceptance requires the full 2500-pair teacher report')
    original,_=load_student(a.reference,device=a.device)
    report=dict(format='distillation_fidelity_experiment_v1',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                original_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),selection=[],postprocessing=[],
                gate_passed=False,stage2_started=False,teacher=manifest)
    def publish():
        temp=a.out_dir/'report.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(a.out_dir/'report.json')
    def progress(phase,done,total):
        if done%25==0 or done==total:print(json.dumps(dict(phase=phase,done=done,total=total)),flush=True)
    publish()
    if a.dataset:
        saved=torch.load(a.dataset,weights_only=True,map_location='cpu')
        if saved['teacher']!=manifest:raise ValueError('cached dataset teacher differs')
        train,validation=saved['train'],saved['validation']
        report['dataset_source']=str(a.dataset)
    else:
        ensemble=ExactTeacher(a.strategies,config,manifest,device=a.device,progress=progress)
        train,train_info=collect_pool_states(ensemble,hands=a.train_hands,seed=a.seed,student=original,progress=progress)
        validation,val_info=collect_pool_states(ensemble,hands=a.validation_hands,seed=a.seed,
            hand_offset=a.train_hands,student=original,progress=progress)
        report.update(train_collection=train_info,validation_collection=val_info)
        train=ensemble.label(train,batch_size=a.batch_size,progress=progress)
        validation=ensemble.label(validation,batch_size=a.batch_size,progress=progress)
        torch.save(dict(train=train,validation=validation,teacher=manifest),a.out_dir/'dataset.pt')
        del ensemble
        if a.device=='cuda':torch.cuda.empty_cache()
    if set(train['hand_id'].tolist())&set(validation['hand_id'].tolist()):raise ValueError('hand split overlap')
    report['original_validation']=fidelity_report(original,validation,batch_size=a.batch_size)
    publish()
    best_rank=(False,-float('inf'));best_path=a.out_dir/'policy_best.pt'
    def consider(model,candidate,steps,*,include_cutoff=False):
        nonlocal best_rank
        for temperature,cutoff in [(1.,0.),(.90,0.),(.85,0.)]+([ (1.,.05),(.90,.05)] if include_cutoff else []):
            selected=evaluate_selection(model,pairs=a.selection_pairs,temperature=temperature,cutoff=cutoff)
            row=dict(candidate=candidate,steps=steps,**selected)
            report['postprocessing' if include_cutoff else 'selection'].append(row)
            print(json.dumps(dict(phase='selection',candidate=candidate,steps=steps,temperature=temperature,
                cutoff=cutoff,score=selected['score'],call=selected['opponents']['call']['mbb_per_hand'],
                equity=selected['opponents']['equity']['mbb_per_hand'])),flush=True)
            # Hard cutoff is a diagnostic only. Temperature is baked into the
            # checkpoint so Stage 2 uses exactly the validated probabilities.
            eligible=(selected['opponents']['call']['mbb_per_hand']>1800
                      and selected['opponents']['equity']['mbb_per_hand']>-3300)
            rank=(eligible,selected['score'])
            if cutoff==0 and rank>best_rank:
                best_rank=rank
                baked=bake_temperature(model,temperature)
                metadata=dict(selection=row,source='playing_strength_selected',temperature_baked=temperature)
                save_student(best_path,baked,manifest,metadata)
                report['best']=metadata
            publish()
    consider(original,'original',0,include_cutoff=True)
    for name in a.candidates.split(','):
        if name not in ('warm_pruned','warm_raw','fresh_pruned'):raise ValueError('unknown candidate')
        torch.manual_seed(a.seed)
        model=copy.deepcopy(original) if name.startswith('warm') else PolicyNetwork(config).to(a.device)
        trainer=FidelityTrainer(model,train,learning_rate=a.learning_rate,seed=a.seed,
                                target_prune=0. if name=='warm_raw' else .03)
        while trainer.steps<a.steps:
            loss=trainer.fit(min(a.check_every,a.steps-trainer.steps),batch_size=a.batch_size)
            validation_report=fidelity_report(model,validation,batch_size=a.batch_size)
            report['last_fit']=dict(candidate=name,steps=trainer.steps,weighted_kl=loss,validation=validation_report)
            save_student(a.out_dir/f'{name}_latest.pt',model,manifest,report['last_fit'])
            consider(model,name,trainer.steps)
        del trainer,model
        if a.device=='cuda':torch.cuda.empty_cache()
    selected,blob=load_student(best_path,device=a.device)
    report['selected_validation']=fidelity_report(selected,validation,batch_size=a.batch_size)
    # A fresh path and a separate full-budget report prevent selection metrics
    # from being confused with the held-out six-opponent acceptance gate.
    benchmark=a.out_dir/'acceptance_2500'
    subprocess.run([sys.executable,str(Path(__file__).with_name('eval_distilled_gauntlet.py')),
        '--reference',str(best_path),'--teacher-report',str(a.teacher_report),'--out-dir',str(benchmark),
        '--device',a.device],check=True)
    result=json.loads((benchmark/'report.json').read_text())
    scores={name:row['student']['mbb_per_hand'] for name,row in result['opponents'].items()}
    mean=result['student_uniform_mean_mbb_per_hand']
    passed=acceptance_gate(result)
    report.update(gate_passed=passed,acceptance=dict(mean_mbb_per_hand=mean,scores=scores,
        report=str(benchmark/'report.json'),thresholds=dict(mean_gt=2000,call_gt=1800,equity_gt=-3300)))
    if passed:
        # Validate the actual frozen-reference wrapper before any Stage 2 start.
        from alphanl.stage2.policy import ResidualExploiterPolicy
        n=min(32,len(validation['legal_mask']))
        probe={k:validation[k][:n].to(a.device) for k in ('token_ids','type_ids','numeric','pad_mask','legal_mask')}
        probe['match_ctx']=torch.zeros(n,12,device=a.device)
        residual=ResidualExploiterPolicy(selected,alpha=1.,beta_kl=.05,late_beta_kl=.03).to(a.device).eval()
        with torch.no_grad():torch.testing.assert_close(residual(probe)[0].softmax(-1),selected.strategy(probe),atol=1e-6,rtol=1e-6)
        save_student(a.out_dir/'policy.pt',selected,manifest,dict(acceptance=report['acceptance'],selection=report['best']))
        report['stage2_reference_parity_passed']=True
        if a.stage2_template:
            running=subprocess.check_output(['ps','-eo','stat,args'],text=True).splitlines()
            if any('scripts/train_stage2.py' in line and not line.strip().startswith(('T','Z')) for line in running):
                raise RuntimeError('another Stage 2 trainer is active; accepted policy preserved, launch refused')
            stage2=json.loads(a.stage2_template.read_text());stage2.pop('experiment',None);stage2.pop('resume_source',None)
            stage2.update(reference=str((a.out_dir/'policy.pt').resolve()),updates=a.stage2_updates)
            stage2['policy'].update(alpha=1.,beta_kl=.05,late_beta_kl=.03)
            stage2['notes']=dict(reference_acceptance=report['acceptance'],fresh_start=True)
            path=a.out_dir/'stage2_accepted_config.json';path.write_text(json.dumps(stage2,indent=2)+'\n')
            cmd=[sys.executable,str(Path(__file__).with_name('train_stage2.py')),'--config',str(path),
                '--out-dir',str(a.stage2_out),'--device',a.device,'--updates',str(a.stage2_updates),
                '--experiment-kind','formal','--experiment-name','stage2_fidelity_accepted_fresh']
            with (a.out_dir/'stage2.log').open('xb') as stream:
                process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(stage2_started=True,stage2_pid=process.pid,stage2_command=cmd)
    publish()
    print(json.dumps(dict(gate_passed=passed,acceptance=report['acceptance'],stage2_started=report['stage2_started'])),flush=True)


if __name__=='__main__':main()

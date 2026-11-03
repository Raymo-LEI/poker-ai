"""Replay held-out student benchmark states and compare exact teacher distributions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from alphanl.cfr.distillation import load_student,prepare_archive
from alphanl.cfr.evaluation import RuleActor,play_hand
from alphanl.cfr.fidelity import ExactTeacher,OBS_KEYS,RULES,fidelity_report,postprocess
from alphanl.cfr.teacher_window import archive_kwargs,assert_window_matches,load_teacher_window
from eval_distilled_gauntlet import StudentActor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategies',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=200)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--window-manifest',type=Path,
                   default=Path(__file__).resolve().parents[1]/'eval/r03/teacher_window_manifest.json')
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    baseline=json.loads(a.baseline_report.read_text())
    if hashlib.sha256(a.reference.read_bytes()).hexdigest()!=baseline['reference_sha256']:
        raise ValueError('baseline student fingerprint changed')
    teacher_report=json.loads(Path(baseline['teacher_report']).read_text())
    window=load_teacher_window(a.window_manifest)
    config,manifest=prepare_archive(a.strategies,**archive_kwargs(window))
    assert_window_matches(manifest,window)
    model,_=load_student(a.reference,device=a.device)
    rows,meta=[],{k:[] for k in ('parent','action','hand_id','street','pot_bb','hero_seat','opponent_id','origin','spr')}
    next_hand=0
    for name in RULES:
        key=f"duplicate:{teacher_report['hero_iteration']}_vs_{name}"
        seed=teacher_report['settings']['seed']+int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
        class Recorder(StudentActor):
            def begin_hand(self,seat):
                super().begin_hand(seat)
                self.parent=-1;self.hand=next_hand+self.audit.counts['hands_sb']+self.audit.counts['hands_bb']-1
                self.seat=seat
            def act(self,snapshot):
                guarded=self.guard.apply(snapshot)
                obs=self.tokenizer.encode(guarded)
                action=super().act(snapshot)
                row=dict(parent=self.parent,action=action,hand_id=self.hand,street=snapshot.street,
                         pot_bb=snapshot.pot/100.,hero_seat=self.seat,opponent_id=RULES.index(name),origin=1,
                         spr=self.guard.all_in_pot_ratio(snapshot))
                for k,v in row.items():meta[k].append(v)
                self.parent=len(rows);rows.append(obs)
                return action
        hero=Recorder(model,seed=seed+11)
        opponent=RuleActor(name,seed=seed+19,equity_samples=teacher_report['settings']['equity_samples'])
        rng=np.random.default_rng(seed)
        original=[json.loads(x) for x in (a.baseline_report.parent/(name+'.jsonl')).read_text().splitlines()]
        for pair in range(a.pairs):
            deck_seed=int(rng.integers(0,2**63-1))
            chips=[play_hand(hero,opponent,deck_seed=deck_seed,hero_seat=seat) for seat in (1,0)]
            if deck_seed!=original[pair]['deck_seed'] or chips!=original[pair]['chips_sb_bb']:
                raise AssertionError('held-out trajectory replay differs')
        next_hand+=2*a.pairs
        print(json.dumps(dict(phase='replayed',opponent=name,states=len(rows))),flush=True)
    data={k:torch.cat([row[k] for row in rows]) for k in OBS_KEYS}
    data.update({k:torch.tensor(v,dtype=torch.float32 if k in ('pot_bb','spr') else torch.long) for k,v in meta.items()})
    def progress(phase,done,total):
        if done%25==0 or done==total:print(json.dumps(dict(phase=phase,done=done,total=total)),flush=True)
    ensemble=ExactTeacher(a.strategies,config,manifest,device=a.device,progress=progress)
    data=ensemble.label(data,progress=progress)
    torch.save(data,a.out_dir/'heldout_audit_states.pt')
    result=fidelity_report(model,data)
    result.update(format='distillation_exact_teacher_audit_v1',pairs_per_opponent=a.pairs,
                  identical_student_payoffs_verified=True,manifest=manifest)
    (a.out_dir/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}),flush=True)


if __name__=='__main__':main()

"""Evaluate a distilled student on the exact recorded Stage 1 duplicate deals."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.distillation import load_student
from alphanl.cfr.evaluation import Audit, RuleActor, duplicate_match, paired_summary
from alphanl.cfr.fidelity import postprocess
from alphanl.transformer.hunl import HunlTokenizer


class StudentActor:
    def __init__(self, model, *, seed, temperature=1., cutoff=0.):
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.audit = Audit()
        self.tokenizer = HunlTokenizer(model.config.model.max_seq_len, **model.config.tokenizer_kwargs())
        self.guard = CFRActionGuard.from_config(model.config)
        self.device = next(model.parameters()).device
        self.temperature,self.cutoff=temperature,cutoff

    def begin_hand(self, seat):
        self.audit.begin_hand(seat)

    @torch.inference_mode()
    def act(self, snapshot):
        began = time.perf_counter()
        original = snapshot
        snapshot = self.guard.apply(snapshot)
        batch = {key: value.to(self.device) for key, value in self.tokenizer.encode(snapshot).items()}
        probs = postprocess(self.model.strategy(batch),temperature=self.temperature,cutoff=self.cutoff)[0].cpu().double().numpy()
        if not np.isfinite(probs).all():
            raise FloatingPointError('nonfinite student probabilities')
        action = int(self.rng.choice(5, p=probs / probs.sum()))
        self.audit.observe(snapshot, probs, action, (time.perf_counter()-began)*1000)
        self.audit.observe_guard(original, self.guard, action)
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--teacher-report', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--temperature',type=float,default=1.)
    parser.add_argument('--cutoff',type=float,default=0.)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, _ = load_student(args.reference, device=args.device)
    teacher = json.loads(args.teacher_report.read_text())
    settings = teacher['settings']
    opponents = ('random','call','maniac','overbet','nit','equity')
    report = dict(format='distilled_student_same_deal_gauntlet_v1', reference=str(args.reference),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  teacher_report=str(args.teacher_report), settings=settings, opponents={},
                  inference=dict(temperature=args.temperature,cutoff=args.cutoff),
                  scope='Six unjittered standard bots; paired gains use recorded identical deck seeds.')
    for name in opponents:
        key = f"duplicate:{teacher['hero_iteration']}_vs_{name}"
        old = teacher['results'][key]
        path = args.teacher_report.parent / old['raw_log']
        if hashlib.sha256(path.read_bytes()).hexdigest() != old['raw_log_sha256']:
            raise ValueError('teacher raw log fingerprint mismatch')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != settings['pairs']:
            raise ValueError('teacher pair count mismatch')
        task_seed = settings['seed'] + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
        hero = StudentActor(model, seed=task_seed + 11,temperature=args.temperature,cutoff=args.cutoff)
        opponent = RuleActor(name, seed=task_seed + 19, equity_samples=settings['equity_samples'])
        chips = []
        with (args.out_dir / f'{name}.jsonl').open('x') as stream:
            def log(row):
                expected = records[row['pair']]
                if row['deck_seed'] != expected['deck_seed'] or row['pair'] != expected['pair']:
                    raise AssertionError('student and teacher decks do not match')
                chips.append(row['chips_sb_bb'])
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            result = duplicate_match(hero, opponent, pairs=settings['pairs'], seed=task_seed, log=log,
                progress=lambda done,total: print(json.dumps(dict(opponent=name,pairs=done,total=total)),flush=True))
        teacher_chips = np.asarray([row['chips_sb_bb'] for row in records])
        gain = paired_summary(np.asarray(chips)-teacher_chips)
        report['opponents'][name] = dict(student=result,
            teacher={key: old[key] for key in ('mbb_per_hand','ci95','pairs','hands')},
            gain_mbb_per_hand=gain['mbb_per_hand'], gain_ci95=gain['ci95'], identical_deals_verified=True)
        (args.out_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(completed=name,student=result['mbb_per_hand'],teacher=old['mbb_per_hand'],
                             gain=gain['mbb_per_hand'],gain_ci95=gain['ci95'])),flush=True)
    report['student_uniform_mean_mbb_per_hand'] = sum(r['student']['mbb_per_hand'] for r in report['opponents'].values())/6
    report['teacher_uniform_mean_mbb_per_hand'] = sum(r['teacher']['mbb_per_hand'] for r in report['opponents'].values())/6
    report['complete'] = True
    (args.out_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k.endswith('_mbb_per_hand') or k=='complete'}),flush=True)


if __name__ == '__main__':
    main()

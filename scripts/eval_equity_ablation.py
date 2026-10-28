"""Compare matched Stage-1 feature/control archives on identical deck pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from alphanl.cfr.distillation import prepare_archive, verify_archive
from alphanl.cfr.evaluation import CFRActor, RuleActor, OPPONENTS, duplicate_match, mean_interval
from alphanl.cfr.evaluation_io import sha256_file, atomic_json
from alphanl.transformer.hunl import BIG_BLIND


def paired_difference(control, equity):
    if len(control) != len(equity) or any(a['deck_seed'] != b['deck_seed'] or a['pair'] != b['pair']
                                           for a,b in zip(control,equity)):
        raise ValueError('paired logs do not use identical decks')
    delta = [(np.mean(b['chips_sb_bb'])-np.mean(a['chips_sb_bb']))/BIG_BLIND*1000
             for a,b in zip(control,equity)]
    return mean_interval(delta)


def validate_controls(control, equity):
    if control.equity_features or not equity.equity_features:
        raise ValueError('requires feature-disabled control and feature-enabled candidate')
    a,b=control.to_dict(),equity.to_dict()
    for key in a:
        if key not in ('equity_features','out_dir','buffer_directory') and a[key] != b[key]:
            raise ValueError(f'unmatched ablation configuration: {key}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--equity',type=Path,required=True)
    p.add_argument('--through',type=int,default=250)
    p.add_argument('--pairs',type=int,default=2500)
    p.add_argument('--seed',type=int,default=98341)
    p.add_argument('--opponents',default=','.join(OPPONENTS))
    p.add_argument('--equity-samples',type=int,default=64)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    names=args.opponents.split(',')
    if (min(args.through,args.equity_samples)<1 or args.pairs<2 or args.seed<0
            or set(names)-set(OPPONENTS) or len(set(names)) != len(names)):
        p.error('positive budgets, >=2 pairs and distinct supported opponents required')
    if args.out.exists():
        p.error('use a new output directory')
    torch.set_num_threads(1)
    controls,manifests={},{}
    for variant,directory in (('control',args.control),('equity',args.equity)):
        controls[variant],manifests[variant]=prepare_archive(directory,through=args.through)
        if manifests[variant]['iterations'] != list(range(1,args.through+1)):
            raise ValueError('paired full-mixture evaluation requires every snapshot 1..T')
    validate_controls(controls['control'],controls['equity'])
    args.out.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]
    report={'format':'equity_feature_paired_ablation_v1','quality_claim':False,
            'status':'running', 'pairs_per_opponent':args.pairs,'seed':args.seed,'through':args.through,
            'teacher_manifests':manifests,'equity_opponent_samples':args.equity_samples,
            'ci_unit':'independent_seat_swapped_deck_pair',
            'source_sha256':{str(path.relative_to(source)):sha256_file(path)
                for path in [Path(__file__).resolve(),*sorted((source/'src/alphanl').rglob('*.py'))]},
            'opponents':{}}
    for index,name in enumerate(names):
        rows,results={},{}
        task_seed=int(np.random.SeedSequence([args.seed,OPPONENTS.index(name)]).generate_state(1)[0])
        for variant,directory in (('control',args.control),('equity',args.equity)):
            records=[]
            hero=CFRActor(directory,seed=task_seed,through=args.through,cache_size=args.through)
            opponent=RuleActor(name,seed=task_seed+1,equity_samples=args.equity_samples)
            with (args.out/f'{variant}_{name}.jsonl').open('w') as handle:
                def log(row):
                    records.append(row)
                    handle.write(json.dumps(row)+'\n')
                results[variant]=duplicate_match(hero,opponent,pairs=args.pairs,seed=task_seed+2,log=log)
            rows[variant]=records
        report['opponents'][name]={'variants':results,
            'equity_minus_control_mbb_per_hand':paired_difference(rows['control'],rows['equity'])}
        atomic_json(args.out/'report.json',report)
        print(json.dumps({name:report['opponents'][name]}),flush=True)
    for variant,directory in (('control',args.control),('equity',args.equity)):
        verify_archive(directory,manifests[variant])

    report['status']='completed_archives_verified'
    atomic_json(args.out/'report.json',report)

if __name__=='__main__':
    main()

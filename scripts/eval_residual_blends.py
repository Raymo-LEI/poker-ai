"""Extend an existing Pro Gauntlet on its exact decks without replaying old cells."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from alphanl.cfr.gauntlet import Candidate, GAUNTLET_OPPONENTS, match_factories, rule_actor, sha256, select_candidate, paired_candidate_gain
from alphanl.stage2.evaluation import summarize_match_pairs
from eval_pro_gauntlet import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--blends", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    report = json.loads(args.baseline_report.read_text())
    if report.get("format") != "pro_gauntlet_v1" or not report.get("complete"):
        raise ValueError("requires a complete original Pro Gauntlet report")
    report["parent_report_sha256"] = sha256(args.baseline_report)
    report["complete"] = False
    report["pending_matchups"] = []
    args.out_dir.mkdir(parents=True, exist_ok=False)
    schedule = np.random.default_rng(report["seed"]).integers(0, 2**63-1, size=(report["pairs"]//50,50))
    if hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest() != report["deck_schedule_sha256"]:
        raise ValueError("baseline deck schedule mismatch")
    (args.out_dir/'deck_seeds.json').write_text(json.dumps(schedule.tolist())+'\n')
    originals = {}
    for name, spec in report["candidates"].items():
        if Path(spec["path"]).exists():
            candidate = Candidate(spec)
            if spec.get("sha256") and candidate.metadata["sha256"] != spec["sha256"]:
                raise ValueError("original candidate identity changed")
            originals[name] = candidate
    original_names = list(report["candidates"])
    blends = [Candidate(dict(name=path.stem,kind="blend",path=str(path.resolve()))) for path in args.blends]
    if any(c.name in report['candidates'] for c in blends):
        raise ValueError("blend name already present")
    index = 0
    def run(left, name, factory, neural=False):
        nonlocal index
        path = args.out_dir/f'blend_matchup_{index:04d}.jsonl'; index += 1
        print(f'Evaluating {left.name} vs {name}',flush=True)
        with path.open('x') as log:
            def record(row):
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            result = match_factories(left.actor,factory,schedule=schedule,seed=report['seed'],log=record)
        result.update(raw_log=path.name,raw_log_sha256=sha256(path))
        report['matrix'][left.name][name] = result
        if neural:
            values = -np.asarray(result['net_bb_pairs'])
            report['matrix'][name][left.name] = {**summarize_match_pairs(values),
                'net_bb_pairs':values.tolist(),'derived_by_zero_sum':True,
                'raw_log':path.name,'raw_log_sha256':result['raw_log_sha256']}
        publish(args.out_dir,report)
    for blend in blends:
        report['candidates'][blend.name]=blend.metadata
        report['matrix'][blend.name]={}
        for name in GAUNTLET_OPPONENTS:
            run(blend,name,lambda seed,name=name:rule_actor(name,seed,report['equity_samples']))
        for name in original_names:
            if name in originals:
                run(blend,name,originals[name].actor,True)
            else:
                report['pending_matchups'].append([blend.name,name])
    for i,left in enumerate(blends):
        for right in blends[i+1:]:run(left,right.name,right.actor,True)
    names=list(report['matrix'])
    report['common_deal_gains']={a:{b:{o:paired_candidate_gain(report['matrix'][a][o],report['matrix'][b][o])
        for o in GAUNTLET_OPPONENTS} for b in names if b!=a} for a in names}
    report['selection']=(select_candidate(report['matrix'],names,full_budget=report['pairs']>=1000,
        all_required_present=True) if not report['pending_matchups'] else
        dict(status='pending_teacher_matchups',recommendation=None,champion=None))
    report['complete']=not report['pending_matchups']
    report['notes'].append('Original cells reused by parent report fingerprint; only new blend cells played. New logs live in this directory; old log paths resolve against the parent report directory.')
    publish(args.out_dir,report)


if __name__=='__main__':main()

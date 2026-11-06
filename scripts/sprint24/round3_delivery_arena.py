"""Resumable exact-byte arena against V22, with a partial-result disposition."""
import math
from pathlib import Path

import numpy as np

from alphanl.stage2 import cfv
from arena_v22 import (PAIRS, RELEASE, SEED_BASE, init_pair_worker, play_match,
                      reference_paths)
from evening_parallel import jobs
from server_arena import _digest_json, _read_match, interval
from server_resources import core_plan, pin


def identities(prep):
    b = dict(release=prep['binding']['release'], source_sha256=prep['source_sha256'],
             model_sha256=prep['model_sha256'], opponent=RELEASE)
    return [dict(b,pair=p,seat=s,seed=SEED_BASE+p) for p in range(PAIRS) for s in (0,1)]


def path_for(work, identity):
    return work/f"pair_{identity['pair']:03d}_{identity['seat']}.json"


def run_match(identity):
    try:
        return play_match(identity)
    except Exception as exc:
        return identity, dict(error=type(exc).__name__+': '+str(exc))


def run(round3, prep, work, workers):
    ref = reference_paths(round3)
    runtime = round3/'runtime_checkout'
    pin(core_plan(1)['worker_cpus'])
    planned = identities(prep)
    for _ in range(2):
        pending = [i for i in planned if not path_for(work,i).exists()]
        if not pending:
            break
        for identity,result in jobs(run_match,pending,len(pending),min(workers,len(pending)),init_pair_worker,
                ((prep['source'],prep['model']),ref,str(runtime/'tests/fixtures/botzone_judge.py')),
                timeout=900,label='ROUND3_DELIVERY_ARENA'):
            path = path_for(work,identity)
            if 'error' in result:
                cfv.atomic_json(path.with_suffix('.error'),dict(identity=identity,**result))
            else:
                cfv.atomic_json(path,dict(identity=identity,result=result,result_sha256=_digest_json(result)))


def summary(work, prep):
    scores = np.full((PAIRS,2),np.nan)
    errors = 0
    for identity in identities(prep):
        path = path_for(work,identity)
        if path.exists():
            row = _read_match(path,identity)
            score = row['score_bb']
            if row['hands'] != 50 or not math.isfinite(score):
                raise ValueError('invalid saved 50-hand match')
            scores[identity['pair'],identity['seat']] = score
        elif path.with_suffix('.error').exists():
            errors += 1
    complete = scores[np.isfinite(scores).all(1)]
    flat = scores[np.isfinite(scores)]
    count = len(flat)
    diff = interval(np.sign(complete).mean(1)) if len(complete)>=2 else None
    return dict(matches=count, target_matches=2*PAIRS, duplicate_pairs=len(complete),
                hands=count*50, failed_matches=errors,
                candidate_win_rate=float((flat>0).mean()) if count else None,
                v22_win_rate=float((flat<0).mean()) if count else None,
                tie_rate=float((flat==0).mean()) if count else None,
                candidate_minus_v22_win_rate=diff,
                significant_win_rate_improvement=bool(count==2*PAIRS and diff['ci95'][0]>0))

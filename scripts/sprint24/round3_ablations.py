"""Resumable offline Round 3 ablations and a read-only distillation audit."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# Set before NumPy/Torch or worker import. No GPU inference or training.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OPPONENTS = ('Bot17_Probe3Bet', 'TAG_Pro', 'Trapper_CR', 'equity', 'NearShove_19949', 'V22')
QUEUE = ('E6','E7','E0','E1','E2','E3','E4','E5','V26','V27','V28',
         'L_no_pot_control','L_no_blind_protection','L_no_anti_punt_caps')
SEED_BASE = 2026110700
HANDS = 50
MIN_PAIRS = 34   # 6 opponents * 34 pairs * 2 seats = 408 matches per arm.
ROUND3_CFVS = 'ccdbd01104b0e17f82293de581fbede8742a2e60061fdfff1b1fe1b77770125e'
_CANDIDATE = _REFERENCE = _JUDGE = None


def read(path):
    return json.loads(Path(path).read_text())


def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pinned_inputs(round3):
    from alphanl.stage2 import cfv
    from round3_ablations_policy import V22_SHA
    from server_arena import assert_production_cfv

    work = round3/'delivery_first'
    receipt = read(work/'FINAL_DELIVERY.json')
    if receipt['status']!='COMPLETE' or not receipt['cfv']['qualified']:
        raise ValueError('expected the completed, gate-passing Round 3')
    v22 = [str(round3/'online_v22'/p) for p in ('V22_SOURCE.zip','data/V22_MODEL.npz')]
    if tuple(map(cfv.digest,v22)) != V22_SHA:
        raise ValueError('archived V22 bytes differ')
    nets = work/'cfv_nets.npz'
    nets_sha = cfv.digest(nets)
    if nets_sha != ROUND3_CFVS:
        raise ValueError('Round 3 CFV hash differs')
    assert_production_cfv(nets)
    packages = {}
    for arm, version in zip(('A','C','D'), (26,27,28)):
        version = f'V{version}'
        entry = receipt['generation_1'][arm]
        if entry['release'] != f'{version}_CFV_R3_G1_{arm}_20261107':
            raise ValueError('unexpected gen-1 release')
        paths = [entry['source'],entry['model']]
        if list(map(cfv.digest,paths)) != [entry['source_sha256'],entry['model_sha256']]:
            raise ValueError('gen-1 package hash differs')
        packages[version] = paths
    with np.load(v22[1],allow_pickle=False) as blob:
        names = tuple(k for k in blob.files if k not in ('meta_json','wizard_beta') and not k.startswith('vleaf.'))
        arrays = {k:blob[k] for k in names}
    if not arrays:
        raise ValueError('empty frozen U250 policy')
    for source,model in packages.values():
        with np.load(model,allow_pickle=False) as blob:
            if not all(k in blob.files and np.array_equal(blob[k],v) for k,v in arrays.items()):
                raise ValueError(f'gen-1 blueprint differs from V22 in {model}')
    distill = read(work/'gen2_distillation/model.pt.manifest.json')
    distill_job = read(work/'gen2_distillation/JOB.json')
    if (not distill['complete'] or not distill['accepted'] or
            distill_job['blueprint'] != distill['source_checkpoint'] or distill_job['nets'] != str(nets) or
            cfv.digest(distill['source_checkpoint']) != distill['source_sha256'] or
            cfv.digest(work/'gen2_distillation/model.pt') != distill['sha256']):
        raise ValueError('distillation artifact provenance changed')
    for arm in ('A','C','D'):
        prepared = read(work/f'g1/{arm}/PREPARED.json')
        if (prepared['binding']['checkpoint_sha256']!=distill['source_sha256'] or
                prepared['binding']['cfv_sha256']!=nets_sha):
            raise ValueError('distillation source was not the released gen-1 checkpoint')
    judge = round3/'runtime_checkout/tests/fixtures/botzone_judge.py'
    return dict(v22=v22, v22_sha256=list(V22_SHA), packages=packages, nets=str(nets),
                cfv_sha256=nets_sha, frozen_policy_arrays_equal=len(arrays), judge=str(judge),
                judge_sha256=cfv.digest(judge), delivery_receipt_sha256=cfv.digest(work/'FINAL_DELIVERY.json'),
                distill=distill)


class DeckRandom(random.Random):
    """The archived judge calls seed() again on entry. Keep that call fixed."""
    def __init__(self, seed):
        self.match_seed = seed
        super().__init__(seed)

    def seed(self, a=None, version=2):
        super().seed(self.match_seed if a is None else a, version=version)


def expected_deck(seed, hands):
    rng = random.Random(seed)
    dealer = rng.randint(0,1)
    decks = []
    for _ in range(hands):
        deck = list(range(52))
        rng.shuffle(deck)
        decks.append(deck)
    return json_digest(dict(dealer=dealer,decks=decks))


def init_match_worker(arm, assets):
    from alphanl.stage2 import cfv
    from evening_d import Bundle, _fixture
    from round3_ablations_policy import V22, legacy_variant

    global _CANDIDATE, _REFERENCE, _JUDGE
    _REFERENCE = V22(*assets['v22'])
    if arm.startswith('E'):
        nets = cfv.load_nets(assets['nets'])[0] if arm=='E1' else None
        _CANDIDATE = V22(*assets['v22'],arm=arm,nets=nets)
    else:
        version = 'V26' if arm.startswith('L_') else arm
        _CANDIDATE = Bundle(*assets['packages'][version])
        if arm.startswith('L_'):
            legacy_variant(_CANDIDATE,arm)
    _JUDGE = _fixture('_round3_ablation_judge',Path(assets['judge']))


def play(identity, hands=HANDS):
    from round3_distill_audit import HIST_SHAPE, cell

    seed, hero, opponent = (identity[k] for k in ('seed','seat','opponent'))
    actor = None
    if opponent!='V22':
        from alphanl.cfr.evaluation import RuleActor
        from ladder_actors import SlayerActor
        actor = (SlayerActor(opponent,seed=seed,equity_samples=16)
                 if opponent in ('Bot17_Probe3Bet','NearShove_19949')
                 else RuleActor(opponent,seed=seed,equity_samples=32))
    _JUDGE.random = DeckRandom(seed)
    requests, responses, data = [[],[]], [[],[]], ['','']
    log, initdata, deck_sha = [], dict(max_hand=hands), None
    population = np.zeros(HIST_SHAPE,dtype=np.int64)
    hand_ids, last_hand = set(), [None,None]
    draws, native_fallbacks = [0,0], [0,0]
    fires_before = getattr(_CANDIDATE,'fires',0)
    for _ in range(12000):
        result = json.loads(_JUDGE.judge(json.dumps(dict(log=log,initdata=initdata))))
        if result['display'].get('error'):
            raise ValueError('judge rejected arena response: '+str(result['display']['error']))
        if result['command']=='finish':
            score = float(result['content'][str(hero)])
            if hand_ids!=set(range(hands)) or score+float(result['content'][str(1-hero)])!=0:
                raise ValueError('arena requires a complete zero-sum match')
            return dict(score_bb=score,hands=hands,deck_sha256=deck_sha,
                        decisions=population.tolist(),adapter_calls=getattr(_CANDIDATE,'fires',0)-fires_before,
                        native_wizard_fallbacks=native_fallbacks)
        if not log:
            initdata = result['initdata']
            deck_sha = json_digest({k:initdata[k] for k in ('dealer','decks')})
            if deck_sha != expected_deck(seed,hands):
                raise ValueError('judge did not deal the planned duplicate sequence')
        (key, request), = result['content'].items()
        seat, hand = int(key), request['hand']
        hand_ids.add(hand)
        requests[seat].append(request)
        payload = dict(requests=requests[seat],responses=responses[seat],data=data[seat])
        if last_hand[seat]!=hand:
            last_hand[seat],draws[seat] = hand,0
            if seat!=hero and actor is not None:
                snapshot,_ = _REFERENCE.runtime.request_snapshot(request)
                actor.begin_hand(snapshot.me_seat)
        if seat==hero or actor is None:
            bundle = _CANDIDATE if seat==hero else _REFERENCE
            if seat==hero:
                snapshot,_ = bundle.runtime.request_snapshot(request)
                population[cell(snapshot.street,snapshot.pot,snapshot.me_stack,snapshot.opp_stack,snapshot.to_call)] += 1
            bundle.agent.last_wizard = ''
            random.seed(seed*10**7 + hand*10**4 + seat*10**3 + draws[seat])
            body = bundle.respond(payload)
            # Preserve the frozen archive's native response on a wizard
            # fallback, but expose the count for every arm and opponent.
            native_fallbacks[int(seat!=hero)] += int(bundle.agent.last_wizard=='wiz=err')
            response, data[seat] = body['response'], body.get('data','')
        else:
            from evening_d import actor_response
            snapshot,options = _REFERENCE.runtime.request_snapshot(request)
            response = actor_response(_REFERENCE,snapshot,request,options,actor.act(snapshot))
        draws[seat] += 1
        if type(response) is not int:
            raise ValueError('arena expected an integer judge response')
        responses[seat].append(response)
        log.extend((dict(output=result),{key:dict(verdict='OK',response=response)}))
    raise RuntimeError('arena judge failed to terminate')


def job(identity):
    return identity, play(identity)


def match_rows(arm, pairs, identity_sha):
    for oi, opponent in enumerate(OPPONENTS):
        for pair in range(pairs):
            for seat in (0,1):
                yield dict(arm=arm,opponent=opponent,pair=pair,seat=seat,
                           seed=SEED_BASE+oi*10000+pair,identity_sha256=identity_sha)


def match_path(folder, identity):
    return folder/f"{identity['opponent']}_{identity['pair']:04d}_{identity['seat']}.json"


def load_match(path, identity):
    row = read(path)
    result = row['result']
    if (row['identity'] != identity or row['result_sha256'] != json_digest(result) or
            result['hands'] != HANDS or not np.isfinite(result['score_bb']) or
            result['deck_sha256'] != expected_deck(identity['seed'],HANDS)):
        raise ValueError('invalid match checkpoint '+str(path))
    return result


def interval(values):
    values = np.asarray(values,dtype=float)
    mean = float(values.mean())
    error = 1.96 * float(values.std(ddof=1)) / len(values)**.5
    return dict(mean=mean,ci95=[mean-error,mean+error],n_duplicate_pairs=len(values))


def score_summary(scores, reference=None, legacy=None):
    win = (scores>0).astype(float)
    result = dict(matches=int(scores.size),duplicate_pairs=len(scores),hands=int(scores.size*HANDS),
                  mbb_per_hand=interval(scores.mean(axis=1)*1000/HANDS),
                  match_win_rate=float(win.mean()),tie_rate=float((scores==0).mean()))
    for name, control in (('vs_v22_baseline', reference),('vs_legacy_baseline', legacy)):
        if control is not None:
            result[name] = dict(mbb_per_hand=interval((scores-control).mean(axis=1)*1000/HANDS),
                                match_win_rate_difference=interval((win-(control>0)).mean(axis=1)))
    return result


def summarize(summary, v22=None, legacy=None):
    rows = summary['legs']
    for opponent, matches in rows.items():
        scores = np.array([[leg['score_bb'] for leg in pair] for pair in matches])
        controls = []
        for control in (v22, legacy):
            if control is None:
                controls.append(None)
                continue
            reference = control['legs'][opponent]
            if [[m['deck_sha256'] for m in pair] for pair in matches] != [
                    [m['deck_sha256'] for m in pair] for pair in reference]:
                raise ValueError('baseline and candidate decks differ')
            controls.append(np.array([[leg['score_bb'] for leg in pair] for pair in reference]))
        summary['per_opponent'][opponent] = score_summary(scores,*controls)
    summary['comparison_status'] = 'complete' if v22 else 'awaiting E0 for paired baseline deltas'
    return summary


def run_arm(out, arm, assets, pairs, workers, identity_sha, completed):
    from alphanl.stage2 import cfv
    from evening_parallel import jobs
    from round3_ablations_policy import DESCRIPTION
    from round3_distill_audit import HIST_SHAPE

    folder = out/'arms'/arm
    folder.mkdir(parents=True,exist_ok=True)
    identities = list(match_rows(arm,pairs,identity_sha))
    saved, pending = {}, []
    started = time.monotonic()
    for identity in identities:
        p = match_path(folder,identity)
        if p.is_file():
            saved[p] = load_match(p,identity)
        else:
            pending.append(identity)
    before = len(saved)
    if pending:
        cfv.atomic_json(out/'STATUS.json',dict(stage='ARENA',arm=arm,completed_arms=list(completed),
                        done=before,target=len(identities)))
        for identity,result in jobs(job,pending,len(pending),min(workers,len(pending)),
                init_match_worker,(arm,assets),timeout=900,label='ROUND3_'+arm):
            p = match_path(folder,identity)
            cfv.atomic_json(p,dict(identity=identity,result=result,result_sha256=json_digest(result)))
            saved[p] = result
            done = len(saved)
            if done%8==0 or done==len(identities):
                cfv.atomic_json(out/'STATUS.json',dict(stage='ARENA',arm=arm,completed_arms=list(completed),
                    done=done,target=len(identities),eta_seconds=(len(identities)-done)*(time.monotonic()-started)/(done-before)))
    prior_path = folder/'RESULT.json'
    prior = read(prior_path) if prior_path.is_file() else {}
    population = np.zeros(HIST_SHAPE,dtype=np.int64)
    legs = {opponent:[[None,None] for _ in range(pairs)] for opponent in OPPONENTS}
    adapter_calls = 0
    fallbacks = {opponent:[0,0] for opponent in OPPONENTS}
    for identity in identities:
        result = saved[match_path(folder,identity)]
        legs[identity['opponent']][identity['pair']][identity['seat']] = {
            k: result[k] for k in ('score_bb','deck_sha256')}
        population += result['decisions']
        adapter_calls += result['adapter_calls']
        for side, count in enumerate(result['native_wizard_fallbacks']):
            fallbacks[identity['opponent']][side] += count
    if arm.startswith('E') and arm!='E0' and not adapter_calls:
        raise ValueError(f'{arm} never exercised its intervention')
    return dict(arm=arm,description=DESCRIPTION[arm],legs=legs,per_opponent={},
                decision_counts=population.tolist(),adapter_calls=adapter_calls,
                native_wizard_fallbacks={opponent:dict(candidate=counts[0],opponent=counts[1])
                                         for opponent,counts in fallbacks.items()},
                elapsed_seconds=prior.get('elapsed_seconds',time.monotonic()-started),
                resumed_matches=prior.get('resumed_matches',before),identity_sha256=identity_sha)


def run(args, sha):
    from alphanl.stage2 import cfv
    from round3_ablations_policy import V22_RELEASE
    from round3_distill_audit import audit, update_arena_comparison

    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assets = pinned_inputs(args.round3)
        binding = dict(commit=sha,assets=assets,seed_base=SEED_BASE,opponents=list(OPPONENTS),
                       queue=list(QUEUE),pairs_per_opponent=args.pairs_per_opponent,hands_per_match=HANDS)
        identity = out/'RUN_IDENTITY.json'
        if identity.is_file() and read(identity)!=binding:
            raise ValueError('experiment inputs or plan changed; use a new output directory')
        cfv.atomic_json(identity,binding)
        identity_sha = json_digest(binding)
        audit_path = out/'DISTILLATION_AUDIT.json'
        cfv.atomic_json(out/'STATUS.json',dict(stage='DISTILLATION_AUDIT'))
        if audit_path.is_file():
            report = read(audit_path)
            if report['identity_sha256']!=identity_sha:
                raise ValueError('distillation audit belongs to another experiment')
        else:
            report = dict(audit(args.round3,assets),identity_sha256=identity_sha)
            cfv.atomic_json(audit_path,report)
        if args.audit_only:
            cfv.atomic_json(out/'STATUS.json',dict(stage='AUDIT_COMPLETE',result=str(audit_path)))
            return
        completed = {}
        for arm in QUEUE:
            completed[arm] = run_arm(out,arm,assets,args.pairs_per_opponent,args.workers,identity_sha,completed)
            publish = completed if arm=='E0' else {arm:completed[arm]}
            for name, summary in publish.items():
                summarize(summary,completed.get('E0'),completed.get('V26') if name.startswith('L_') else None)
                cfv.atomic_json(out/'arms'/name/'RESULT.json',summary)
            update_arena_comparison(report,arm,completed[arm]['decision_counts'])
            cfv.atomic_json(audit_path,report)
            cfv.atomic_json(out/'CHECKPOINTS.json',dict(identity_sha256=identity_sha,
                completed={name:str(out/'arms'/name/'RESULT.json') for name in completed},
                next_arm=next((name for name in QUEUE if name not in completed),None)))
        final = out/'FINAL_ABLATIONS.json'
        cfv.atomic_json(final,dict(status='COMPLETE',recommended_deployment=V22_RELEASE,
            identity_sha256=identity_sha,opponents=list(OPPONENTS),arms={
                name:{k:v for k,v in summary.items() if k!='legs'} for name,summary in completed.items()},
            distillation_audit=str(audit_path)))
        cfv.atomic_json(out/'STATUS.json',dict(stage='COMPLETE',result=str(final)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round3',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--workers',type=int,default=24,choices=range(1,25))
    p.add_argument('--pairs-per-opponent',type=int,default=MIN_PAIRS)
    p.add_argument('--audit-only',action='store_true')
    args = p.parse_args()
    args.round3, args.out = args.round3.resolve(), args.out.resolve()
    if args.pairs_per_opponent<MIN_PAIRS:
        p.error('at least 34 duplicate pairs per opponent required')
    if args.out==args.round3 or args.round3 in args.out.parents:
        p.error('choose a new experiment directory outside the archived Round 3 run')
    sha = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if sha != args.expected_commit:
        p.error('checkout does not match --expected-commit')
    subprocess.run(['git','-C',str(REPO),'diff','--quiet','HEAD','--'],check=True)
    runtime = args.round3/'runtime_checkout'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts/sprint24')]
    try:
        run(args,sha)
    except BaseException as exc:
        from alphanl.stage2 import cfv
        # Keep the prior progress visible. The matching output directory is
        # resumable from the same commit and command.
        if args.out.is_dir():
            lock_path = args.out/'.run.lock'
            if lock_path.exists():
                with lock_path.open('a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise exc
                    status = args.out/'STATUS.json'
                    previous = read(status) if status.is_file() else {}
                    cfv.atomic_json(status,dict(previous,stage='FAILED',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    main()

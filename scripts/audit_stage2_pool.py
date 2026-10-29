"""Bounded paired pool diagnostics and a disposable KL-coefficient experiment."""
import argparse
import copy
import json
from pathlib import Path
import torch
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool
from alphanl.stage2.train import collect_matches, load_checkpoint, train_matches


def environments(model, pool, seed, count):
    return [BotzoneMatchEnv(model.config, pool=pool, seed=seed+i,
                           first_hero_seat=i % 2, defer_opponents=True) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--kl-updates', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.blocks < 1 or args.kl_updates < 1:
        parser.error('use a new output and positive diagnostic budgets')
    torch.set_num_threads(1)
    settings = json.loads(args.config.read_text())
    candidate, blob = load_checkpoint(args.checkpoint, device=args.device)
    torch.manual_seed(120029)
    zero = ResidualExploiterPolicy(candidate.reference, **candidate.settings).to(args.device).eval()
    pool = MatchOpponentPool(settings['strategies'], seed=120029, **settings['pool'])
    pool.initialize(candidate)
    pool.load_state_dict(blob['pool'])
    initial_state = copy.deepcopy(pool.state_dict())
    report = dict(format='stage2_paired_pool_diagnostic_v1', checkpoint_update=blob['update'],
                  scope='Disposable diagnostic; original process and checkpoint stay untouched.', pool_matches=[])
    for block in range(args.blocks):
        before = copy.deepcopy(pool.state_dict())
        seed = 120029 + block * 20
        torch.manual_seed(seed)
        base = collect_matches(zero, environments(zero, pool, seed, 20))
        pool.load_state_dict(before)
        torch.manual_seed(seed)
        current = collect_matches(candidate, environments(candidate, pool, seed, 20))
        for a, b in zip(base, current):
            for key in ('opponent', 'opponent_snapshot', 'opponent_variant'):
                if a.report[key] != b.report[key]:
                    raise AssertionError('paired opponent identity differs')
            report['pool_matches'].append(dict(opponent=a.report['opponent'],
                zero_net_bb=a.report['hero_net_bb'], candidate_net_bb=b.report['hero_net_bb']))
            pool.record(b.report['opponent'], b.report['hero_net_bb'])
    report['by_opponent'] = {}
    for name in ['all'] + list(pool.weights):
        rows = [r for r in report['pool_matches'] if name == 'all' or r['opponent'] == name]
        if rows:
            stats = {'matches':len(rows), 'hands':len(rows)*50}
            for prefix in ('zero', 'candidate'):
                stats[prefix+'_mbb_per_hand'] = sum(r[prefix+'_net_bb'] for r in rows)/len(rows)/50*1000
                stats[prefix+'_wins'] = sum(r[prefix+'_net_bb'] > 0 for r in rows)
            report['by_opponent'][name] = stats
    print(json.dumps({'paired_pool':report['by_opponent']}), flush=True)
    report['kl_experiment'] = {}
    for label, beta, late in [('original', 1., .1), ('relaxed', .05, .03)]:
        model, trial_blob = load_checkpoint(args.checkpoint, device=args.device)
        model.settings.update(beta_kl=beta, late_beta_kl=late, alpha=1.)
        pool.load_state_dict(initial_state)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['learning_rate'])
        optimizer.load_state_dict(trial_blob['optimizer'])
        rows = []
        for update in range(args.kl_updates):
            seed = 130029 + update*20
            torch.manual_seed(seed)
            def completed(index, result):
                pool.record(result['opponent'], result['hero_net_bb'])
            rollouts = collect_matches(model, environments(model, pool, seed, 20), on_match_end=completed)
            metrics = train_matches(model, optimizer, rollouts, **settings['ppo'])
            metrics['sampled'] = {name:sum(r.report['opponent'] == name for r in rollouts) for name in pool.weights}
            metrics['pool_feedback'] = dict(pool.matches)
            rows.append(metrics)
            print(json.dumps({'kl_trial':label, 'diagnostic_update':update+1, **metrics}), flush=True)
        report['kl_experiment'][label] = dict(policy=model.settings, updates=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report':str(args.output)}), flush=True)


if __name__ == '__main__':
    main()

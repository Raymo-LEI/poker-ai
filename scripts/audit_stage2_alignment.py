"""Zero-update wrapper audit on common deals against the direct Stage 1 path."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.evaluation import RuleActor
from alphanl.cfr.traversal import HunlState
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.train import load_checkpoint
from alphanl.stage2.opponents import JitteredRuleActor
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.model import masked_log_probs
from alphanl.transformer.tables import standard_deck


@torch.inference_mode()
def wrapper_match(model, opponent, decks, first_seat, *, check=False):
    env = BotzoneMatchEnv(model.config, opponent, decks=decks, first_hero_seat=first_seat)
    obs = env.reset()
    device = next(model.parameters()).device
    checked, max_error, max_kl, seats, streets = 0, 0., 0., set(), set()
    actions = []
    while not env.done:
        if check:
            expected = HunlState(env.table, HunlTokenizer(model.config.model.max_seq_len,
                **model.config.tokenizer_kwargs()), CFRActionGuard.from_config(model.config)).observation()
            for key, value in expected.items():
                if not torch.equal(obs[key], value):
                    raise AssertionError(f"Stage 1 observation differs: {key}")
            seats.add(env.hero)
            streets.add(env.table.street)
        batch = {k: v.to(device) for k, v in obs.items()}
        if check:
            ref = model.reference(batch).softmax(-1)
            final = model(batch)[0].softmax(-1)
            max_error = max(max_error, float((ref - final).abs().max()))
            torch.testing.assert_close(final, ref, atol=1e-6, rtol=1e-6)
            max_kl = max(max_kl, float(model.kl_penalty(batch)[1].max()))
            checked += 1
        action, *_ = model.act(batch)
        actions.append(int(action))
        obs, _, _, _ = env.step(int(action))
    return dict(net_bb=env.ctx.hero_net_bb, hand_bb=[h['hero_net_bb'] for h in env.hands],
                actions=actions, checked_decisions=checked, max_probability_error=max_error,
                max_kl=max_kl, seats=sorted(seats), streets=sorted(streets))


@torch.inference_mode()
def direct_match(reference, opponent, decks, first_seat):
    """Direct table/tokenizer/student path; no Stage 2 env, HUD or residual."""
    config = reference.config
    table = HunlTable(seed=1)
    tokenizer = HunlTokenizer(config.model.max_seq_len, **config.tokenizer_kwargs())
    guard = CFRActionGuard.from_config(config)
    device = next(reference.parameters()).device
    actions, hand_bb = [], []
    for hand, deck in enumerate(decks):
        hero = first_seat ^ (hand % 2)
        table.reset(deck)
        opponent.begin_hand(1 - hero)
        while not table.is_over():
            if table.current_player() == hero:
                obs = HunlState(table, tokenizer, guard).observation()
                batch = {k: v.to(device) for k, v in obs.items()}
                probs = masked_log_probs(reference(batch), batch['legal_mask'])[2]
                action = int(torch.distributions.Categorical(probs=probs).sample())
                actions.append(action)
            else:
                action = int(opponent.act(table.snapshot(1 - hero)))
            table.step(action)
        hand_bb.append(float(table.payoffs()[hero]) / 100.)
    return dict(net_bb=sum(hand_bb), hand_bb=hand_bb, actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--seed', type=int, default=92017)
    parser.add_argument('--match-pairs', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.match_pairs < 2:
        parser.error('use a new output path and at least two match pairs')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = ResidualExploiterPolicy.from_reference(args.reference).to(args.device).eval()
    candidate, blob = load_checkpoint(args.checkpoint, device=args.device) if args.checkpoint else (None, None)
    if candidate is not None:
        for key, value in model.reference.state_dict().items():
            if not torch.equal(value, candidate.reference.state_dict()[key]):
                raise AssertionError('checkpoint reference differs from requested student')
    rng = np.random.default_rng(args.seed)
    decks = [[rng.permutation(standard_deck()).tolist() for _ in range(50)] for _ in range(args.match_pairs)]
    report = dict(format='stage2_zero_update_alignment_v1', seed=args.seed,
                  reference=str(args.reference), reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  checkpoint_update=blob['update'] if blob else None,
                  hands_per_opponent=100 * args.match_pairs, equity_samples=64, opponents={})
    for name in ('random', 'call', 'overbet', 'equity'):
        rows = []
        for pair, pack in enumerate(decks):
            for seat in (1, 0):
                seed = args.seed + pair * 2 + seat
                torch.manual_seed(seed)
                wrapped = wrapper_match(model, RuleActor(name, seed=seed), pack, seat, check=True)
                torch.manual_seed(seed)
                direct = direct_match(model.reference, RuleActor(name, seed=seed), pack, seat)
                if wrapped['actions'] != direct['actions'] or wrapped['hand_bb'] != direct['hand_bb']:
                    raise AssertionError(f'wrapper/direct trajectory differs: {name}, {pair}, {seat}')
                row = dict(pair=pair, first_hero_seat=seat, zero_net_bb=wrapped['net_bb'],
                           direct_net_bb=direct['net_bb'], checked_decisions=wrapped['checked_decisions'],
                           max_probability_error=wrapped['max_probability_error'], max_kl=wrapped['max_kl'],
                           seats=wrapped['seats'], streets=wrapped['streets'])
                if candidate is not None:
                    torch.manual_seed(seed)
                    row['candidate_net_bb'] = wrapper_match(candidate, RuleActor(name, seed=seed), pack, seat)['net_bb']
                if name != 'random':
                    torch.manual_seed(seed)
                    row['jitter_zero_net_bb'] = wrapper_match(model, JitteredRuleActor(name, seed=seed), pack, seat)['net_bb']
                rows.append(row)
        stats = dict(matches=len(rows), hands=50*len(rows), legs=rows,
                     zero_mbb_per_hand=sum(r['zero_net_bb'] for r in rows)/(50*len(rows))*1000,
                     zero_match_wins=sum(r['zero_net_bb'] > 0 for r in rows), exact_trajectory_parity=True)
        for prefix in ('candidate', 'jitter_zero'):
            if prefix+'_net_bb' in rows[0]:
                stats[prefix+'_mbb_per_hand'] = sum(r[prefix+'_net_bb'] for r in rows)/(50*len(rows))*1000
        report['opponents'][name] = stats
        print(json.dumps({name: {k:v for k,v in stats.items() if k != 'legs'}}), flush=True)
    report['all_parity_checks_passed'] = True
    report['scope'] = 'Small paired diagnostic; not a playing-strength acceptance or convergence claim.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report':str(args.output),'all_parity_checks_passed':True}), flush=True)


if __name__ == '__main__':
    main()

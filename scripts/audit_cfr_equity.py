"""Audit all 1,326 SB opening hands in frozen snapshots; no training state read."""
import argparse
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from alphanl.cfr.actions import CFRActionGuard
from alphanl.cfr.config import CFRConfig
from alphanl.cfr.model import RegretNetwork
from alphanl.transformer.model import PokerTransformer
from dataclasses import asdict
from alphanl.cfr.model import regret_matching
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.equity import preflop_key, preflop_table
from alphanl.transformer.tables import standard_deck
from alphanl.transformer.tokenizer import cat_batches


@torch.inference_mode()
def audit(directory, iteration, *, trusted_local_checkpoints=False):
    source = Path(directory) / f"iteration_{iteration:07d}.pt"
    blob = torch.load(source, map_location="cpu", weights_only=not trusted_local_checkpoints)
    cfg = CFRConfig.from_checkpoint(blob["config"])
    from alphanl.transformer.hunl import RULES_VERSION
    if (blob.get("numeric_version") != cfg.numeric_version or blob.get("rules_version") != RULES_VERSION
            or blob.get("iteration") != iteration or blob.get("format") not in ("sd_cfr_strategy_v1", "sd_cfr_strategy_v2")):
        raise ValueError("incompatible snapshot")
    model = RegretNetwork(PokerTransformer(**asdict(cfg.model)), fallback=cfg.fallback).eval()
    model.load_state_dict(blob["models"][1 if cfg.separate_player_models else "shared"], strict=True)
    table = HunlTable(seed=1)
    table.reset()
    opening = table.snapshot(1)
    guard = CFRActionGuard.from_config(cfg)
    tokenizer = HunlTokenizer(cfg.model.max_seq_len, **cfg.tokenizer_kwargs())
    rows, distributions, regrets, legal = [], [], [], []
    combos = list(combinations(standard_deck(), 2))
    for start in range(0, len(combos), 128):
        hands = combos[start:start+128]
        snapshots = [guard.apply(replace(opening, hole_cards=hole)) for hole in hands]
        batch = cat_batches([tokenizer.encode(s) for s in snapshots])
        raw = model(batch)
        probs = regret_matching(raw, batch['legal_mask'], fallback=cfg.fallback)
        regrets.extend(raw.tolist())
        distributions.extend(probs.tolist())
        legal.extend(batch['legal_mask'].tolist())
        for hole, p, regret in zip(hands, probs.tolist(), raw.tolist()):
            rows.append({'hole': list(hole), 'class': preflop_key(hole),
                         'random_equity': preflop_table()[preflop_key(hole)],
                         'fold_probability': p[0], 'call_probability': p[1],
                         'regrets_scaled': regret, 'probabilities': p})
    r, p, mask = np.array(regrets), np.array(distributions), np.array(legal) > .5
    fallback = ~(np.where(mask, r, 0) > 0).any(-1)
    classes = {}
    for key in preflop_table():
        selected = [row for row in rows if row['class'] == key]
        classes[key] = {'combos': len(selected), 'random_equity': preflop_table()[key],
                        'fold_probability': float(np.mean([row['fold_probability'] for row in selected]))}
    return {'iteration': iteration, 'snapshot_semantics': 'sigma_t_archived_before_fit',
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'hands': len(combos), 'numeric_version': cfg.numeric_version,
            'legal_actions': np.flatnonzero(mask[0]).tolist(),
            'pot_chips': opening.pot, 'to_call_chips': opening.to_call,
            'action_chips': {str(a): table.action_chips(a) for a in guard.apply(opening).legal_action_ids},
            'fold_terminal_bb': -.5, 'regret_target_scale': cfg.regret_target_scale,
            'mean_action_probabilities': p.mean(0).tolist(),
            'sb_open_fold_probability': float(p[:,0].mean()),
            'nonpositive_fallback_fraction': float(fallback.mean()),
            'fold_gt_call_fraction': float((r[:,0] > r[:,1]).mean()),
            'fold_minus_call_scaled_quantiles': np.quantile(r[:,0]-r[:,1], [0,.25,.5,.75,1]).tolist(),
            'regrets_scaled_mean': r.mean(0).tolist(),
            'class_summary': classes, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategies', type=Path, required=True)
    parser.add_argument('--iterations', default='100,238,250')
    parser.add_argument('--trusted-local-checkpoints', action='store_true',
                        help='allow protocol-4 pickle from your own trusted training run only')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('use a new output file')
    torch.set_num_threads(1)
    report = {'format': 'sb_open_equity_audit_v1', 'scope': 'uniform 1326 visible SB opening hands; no sampled decisions',
              'snapshots': [audit(args.strategies, int(t), trusted_local_checkpoints=args.trusted_local_checkpoints) for t in args.iterations.split(',')]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in report['snapshots']:
        print(json.dumps({k: v for k, v in row.items() if k not in ('class_summary','rows')}))

if __name__ == '__main__':
    main()

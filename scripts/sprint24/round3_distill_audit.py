"""Read-only comparison of the archived Round 3 student, teacher and blueprint."""
from collections import Counter

import numpy as np

from alphanl.stage2 import cfv
from alphanl.transformer.tokenizer import STREET_ID_BASE, TYPE_STREET

STREETS = ('preflop', 'flop', 'turn', 'river')
SPR_EDGES = (1., 2., 4., 8.)
SPR_NAMES = ('[0,1)', '[1,2)', '[2,4)', '[4,8)', '[8,+inf)')
HIST_SHAPE = (len(STREETS), len(SPR_NAMES), 2)
ROW_FIELDS = ('token_ids', 'type_ids', 'numeric', 'pad_mask', 'legal_mask', 'match_ctx')
TARGET_FILES = {
    'scripts/sprint24/overnight_data.py': 'bd1205c6fd932795600c511a63d7a89dcd711093554eb6dba2aef1dd2548e0d8',
    'scripts/sprint24/overnight_distill.py': 'd0cd9bc30d0149656b9265d9993ab8bd03b4700e7e94902497b75ca91adb2d68',
    'src/alphanl/stage2/evening_strategy.py': '4e56478b019d50c9cb1b46f0beffc43c0347764bda5f973a4178de1d3f864df5',
    'src/alphanl/stage2/deep_solver_v2.py': '7b2cd4b095ff38ad7f0203623dc87562b63f8a146a013ea4c6a5f246c8209381',
}


def cell(street, pot, me, opp, to_call):
    return street, int(np.searchsorted(SPR_EDGES, min(me, opp) / pot, side='right')), int(to_call > 0)


def decode_cells(rows):
    marker = np.where(rows['type_ids'] == TYPE_STREET, rows['token_ids'] - STREET_ID_BASE, -1)
    street = marker.max(axis=1)
    # The CLS numeric channels are pot, hero stack, opposing stack and call,
    # encoded in BB as log1p(chips/BB)/log1p(200). Judge chip amounts are integers.
    pot, me, opp, call = (np.rint(100*np.expm1(rows['numeric'][:, 0].astype(float)*np.log1p(200.))) / 100).T
    if (np.any(pot <= 0) or np.any((street < 0) | (street > 3)) or np.any(np.minimum(me, opp) < 0)):
        raise ValueError('archived observations do not decode to valid street/SPR')
    spr = np.searchsorted(SPR_EDGES, np.minimum(me, opp)/pot, side='right')
    return street, spr, (call > 0).astype(int)


def hist(cells):
    out = np.zeros(HIST_SHAPE, dtype=np.int64)
    np.add.at(out, cells, 1)
    return out


def distribution(counts):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0 or counts.shape != HIST_SHAPE:
        raise ValueError('empty or incompatible street/SPR distribution')
    return dict(decisions=total, counts=counts.tolist(),
                street_share=dict(zip(STREETS, (counts.sum(axis=(1,2))/total).tolist())),
                facing_bet_share=float(counts[:, :, 1].sum()/total))


def tv_summary(a, b):
    tv = .5*np.abs(a-b).sum(axis=1)
    if not len(tv):
        return dict(n=0, mean=None, p95=None, max=None)
    return dict(n=len(tv), mean=float(tv.mean()), p95=float(np.quantile(tv,.95)), max=float(tv.max()))


def tv_by_cell(rows, gen2, gen1):
    street, spr, _ = decode_cells(rows)
    out = {}
    for i, name in enumerate(STREETS):
        groups = [('all', street == i)]
        groups += [(label, (street == i) & (spr == j)) for j, label in enumerate(SPR_NAMES)]
        out[name] = {}
        for label, mask in groups:
            out[name][label] = dict(gen2_vs_gen1=tv_summary(gen2[mask],gen1[mask]),
                                    gen2_vs_archived_target=tv_summary(gen2[mask],rows['target'][mask]),
                                    gen1_vs_archived_target=tv_summary(gen1[mask],rows['target'][mask]))
    return out


def probs(model, rows):
    import torch
    result = []
    with torch.inference_mode():
        for i in range(0,len(rows['target']),256):
            batch = {key: torch.as_tensor(rows[key][i:i+256]) for key in ROW_FIELDS}
            logits, _ = model(batch)
            result.append(logits.softmax(-1).cpu().numpy())
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite holdout policy')
    return values


def audit(round3, assets):
    from alphanl.stage2.train import load_checkpoint
    runtime = round3/'runtime_checkout'
    src = {name:cfv.digest(runtime/name) for name in TARGET_FILES}
    if src != TARGET_FILES:
        raise ValueError('archived distillation target source differs from audited bytes')
    delivery = round3/'delivery_first'
    gen2 = delivery/'gen2_distillation'
    manifest = assets['distill']
    older, _ = load_checkpoint(manifest['source_checkpoint'])
    newer, newer_blob = load_checkpoint(gen2/'model.pt')
    if newer_blob['metadata'].get('teacher') != 'pure_signed_dcfr_policy_and_centered_root_bb':
        raise ValueError('unexpected gen-2 distillation provenance')
    if not all(np.array_equal(a.cpu().numpy(), b.cpu().numpy()) for a,b in
               zip(older.reference.state_dict().values(),newer.reference.state_dict().values())):
        raise ValueError('gen-2 reference backbone differs from frozen gen-1')
    populations = {}
    results = {}
    for split in ('train','final','anchor'):
        rows, record = cfv.load_artifact(gen2/f'model.{split}.npz','joint_distillation')
        if record['sha256'] != manifest['split_sha256'][split] or record['teacher_sha256'] != assets['cfv_sha256']:
            raise ValueError('archived distillation split receipt differs')
        if not np.allclose(rows['target'].sum(1),1,atol=1e-6):
            raise ValueError('archived targets are not distributions')
        populations[split] = dict(distribution(hist(decode_cells(rows))), sha256=record['sha256'],
                                  opponents=record['opponent_counts'])
        if split == 'train':
            continue
        g1, g2 = probs(older,rows), probs(newer,rows)
        street = decode_cells(rows)[0]
        if split == 'anchor' and (np.any(street != 0) or not np.array_equal(rows['prior'],rows['target'])):
            raise ValueError('anchor must hold preflop prior rows only')
        if split == 'final' and np.any(street == 0):
            raise ValueError('distillation final holdout must be postflop')
        results[split] = dict(by_street_spr=tv_by_cell(rows,g2,g1),
                              gen1_vs_saved_rollout_prior=tv_summary(g1,rows['prior']))
    pre = results['anchor']['by_street_spr']['preflop']['all']['gen2_vs_gen1']
    return dict(
        analysis='frozen policy distributions before online search and guard',
        source_hashes=src,
        target=dict(gen1='Frozen U250 source checkpoint. No Round 3 gen-1 student fit or teacher targets.',
            gen2='Pure signed DCFR average, offline, zero direct blueprint mixture, uniform solver initialization.',
            gen2_iterations=dict(flop=32,turn=32,river=64), chance_samples=2,
            warm_started_from_network=False, postflop_target_contains_network_policy=False,
            preflop='Original champion anchor. Archived target field holds the rollout prior for anchor comparison only.',
            indirect_network_dependency='Frozen U250 chooses rollout actions and drives self-play particle posteriors. '
                'Training also has a separate KL-to-rollout-prior penalty and a root-BB value loss. '
                'Neither term blends the blueprint into the recorded DCFR policy label.',
            regression_interpretation='No direct self-target or network warm start in the recorded labels. '
                'Compare holdout TV and state distribution before attributing the arena drop.',
            source_checkpoint=manifest['source_checkpoint'], source_sha256=manifest['source_sha256'],
            student_sha256=manifest['sha256'], cfv_sha256=assets['cfv_sha256'], rollback=manifest['rollback']),
        preflop=dict(pre,exact_zero=pre['max']==0.),
        splits=results, distillation_populations=populations,
        arena_population_comparison=dict(status='awaiting arena decision counts',by_arm={}))


def update_arena_comparison(report, arm, counts):
    counts = np.asarray(counts,dtype=np.int64)
    dist = distribution(counts)
    scores = {}
    for split in ('train','final'):
        reference = np.asarray(report['distillation_populations'][split]['counts'],dtype=np.int64)
        arena = counts[1:]
        reference = reference[1:]
        if arena.sum() <= 0:
            raise ValueError('no postflop decisions in arena population')
        scores[split] = dict(
            postflop_street_spr_facing_tv=float(.5*np.abs(arena/arena.sum()-reference/reference.sum()).sum()),
            postflop_arena_decisions=int(arena.sum()),postflop_distillation_decisions=int(reference.sum()))
    comparison = report['arena_population_comparison']
    comparison['status'] = 'available for completed arms'
    comparison['by_arm'][arm] = dict(arena=dist, vs_distillation=scores)

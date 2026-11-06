"""Verify real CFV inputs and publish new, immutable evaluation packages."""
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from alphanl.stage2 import cfv
from alphanl.stage2.deployment_guard import V102_R7_GUARD
from alphanl.stage2.export import export_botzone_weights, pack_botzone_zip
from cfv_recovery import old_artifact
from server_arena import assert_production_cfv

THRESHOLD = .12
ARMS = ('A', 'C', 'D')
BASELINE_RELEASE = 'V22_SEARCH_LOCK_20261104'


def read(path):
    return json.loads(Path(path).read_text())


def manifest_path(path):
    return Path(str(path) + '.manifest.json')


def gate_value(result):
    gate = result['gate']
    if (type(gate['passed']) is not bool or gate['samples'] != 2048
            or any(not math.isfinite(gate[k]) for k in
                   ('pot_mae_ratio', 'mae', 'constant_mae', 'gain_lower95'))):
        raise ValueError('invalid holdout receipt')
    return float(gate['pot_mae_ratio'])


def import_head(path, sources, kind, receipt, prefix, minimum):
    weights, meta = old_artifact(path, sources, kind)
    gate_value(receipt)
    if (meta['sha256'] != receipt['weights_sha256'] or meta['epochs'] != 50
            or meta['samples'] < minimum or meta.get('fixture') or meta.get('smoke')):
        raise ValueError('CFV head is not a real gate-evaluated candidate: ' + str(path))
    cfv.validate_weights(weights, (prefix,))
    if meta.get('target_normalization') != cfv.target_normalization(weights):
        raise ValueError('CFV target contract differs from its manifest')
    return weights, meta


def select_pair(round3, prior, work, cutoff_reason):
    """Use recorded scores only. Every alternative must match the same river teacher."""
    old_source = prior/'fit_checkout'
    sources = {p.name: cfv.digest(p) for p in (
        old_source/'src/alphanl/stage2/cfv.py',
        old_source/'src/alphanl/stage2/deep_solver_v2.py',
        old_source/'scripts/sprint24/evening_pipeline.py',
    )}
    river_receipt = read(prior/'FINAL_RIVER_GATE.json')
    river, rm = import_head(prior/'cfv_river.npz', sources, 'river_net', river_receipt, 'cfv_river.', 80000)
    if (rm['validation']['CFV_HOLDOUT_GATE'] != river_receipt['gate']
            or cfv.target_normalization(river) != cfv.POT_TARGET):
        raise ValueError('river gate and immutable weights do not bind')

    def turn_candidate(folder):
        g = read(folder/'FINAL_TURN_GATE.json')
        used = read(folder/'FINAL_TURN_HOLDOUT_USED.json')
        if (g['weights_sha256'] != used['weights_sha256']
                or g['holdout_sha256'] != used['holdout_sha256']
                or g['teacher_sha256'] != rm['sha256']
                or g['commit'] != river_receipt['commit']):
            raise ValueError('turn gate, one-shot receipt or river teacher differs')
        if folder == round3:
            selection_path = folder/'TURN_SELECTION.json'
            s = read(selection_path)
            if cfv.digest(selection_path) != g['selection_sha256'] or s['winner'] != g['winner']:
                raise ValueError('fresh turn gate was not bound to the locked selection')
            if g['winner'] not in ('pot_bb_l1', 'raw_bb_smooth_l1'):
                raise ValueError('unrecognized turn candidate')
            path = folder/'turn_candidates'/g['winner']/'weights.npz'
        else:
            path = folder/'turn_candidate/weights.npz'
        weights, meta = import_head(path, sources, 'turn_candidate', g, 'cfv_turn.', 50000)
        return folder, g, weights, meta

    old = turn_candidate(prior)
    current = turn_candidate(round3) if (round3/'FINAL_TURN_GATE.json').exists() else None
    latest_river = read(round3/'FINAL_RIVER_GATE.json')
    gate_value(latest_river)
    if latest_river['gate']['passed'] and latest_river != river_receipt:
        raise ValueError('Round 3 unexpectedly changed the accepted river')
    qualified = bool(current and current[1]['gate']['passed'] and latest_river['gate']['passed'])
    alternatives = [old] + ([current] if current else [])
    picked = current if qualified else min(alternatives, key=lambda x: gate_value(x[1]))
    folder, turn_receipt, turn, tm = picked
    failed = {}
    for street, g in (('river', latest_river), ('turn', current[1] if current else turn_receipt)):
        if not g['gate']['passed']:
            failed[street] = gate_value(g)
    if len(failed)==1:
        reason=f'CFV gate failed (value {next(iter(failed.values())):.5f} vs {THRESHOLD:.2f})'
    elif failed:
        reason='CFV gate failed ('+'; '.join(f'{k} value {v:.5f} vs {THRESHOLD:.2f}' for k,v in failed.items())+')'
    else:
        reason=None if qualified else 'current CFV gate pending at cutoff'
    decision = dict(qualified=qualified, reason=reason, fallback_trigger=cutoff_reason,
                    failed_streets=failed,river=river_receipt, turn=turn_receipt, turn_input_run=str(folder),
                    comparison=[dict(run=str(p),ratio=gate_value(g),iterations=g['iterations'],
                                     chance_samples=g['chance_samples'],passed=g['gate']['passed'])
                                for p,g,_,_ in alternatives])
    selection = work/'CFV_INPUT.json'
    if selection.exists() and read(selection) != decision:
        raise ValueError('delivery CFV selection is locked to different gate receipts')
    if not selection.exists():
        cfv.atomic_json(selection, decision)
    weights = dict(river, **turn)
    if cfv.target_normalization(turn) is None:
        weights[cfv.RAW_TURN_KEY] = np.ones(1, dtype=np.uint8)
    cfv.validate_weights(weights)
    nets = work/'cfv_nets.npz'
    if not nets.exists():
        cfv.publish(nets, weights, rm['samples']+tm['samples'], 50, 'nets',
                    dict(tree=rm['tree'], production=qualified, fixture=False, smoke=False,
                         offline_hud_prior=False, offline_evaluation_only=not qualified,
                         river_samples=rm['samples'], turn_samples=tm['samples'],
                         validation=dict(river=rm['validation'], turn=dict(
                             tm['training'], CFV_HOLDOUT_GATE=turn_receipt['gate'])),
                         delivery_gate_disposition=reason,
                         input_river_sha256=rm['sha256'], input_turn_sha256=tm['sha256'],
                         turn_label_depth=dict(iterations=turn_receipt['iterations'],
                                               chance_samples=turn_receipt['chance_samples'])))
    stored, _ = cfv.load_nets(nets, expected_target_normalization=cfv.POT_TARGET)
    if cfv.array_digest(stored) != cfv.array_digest(weights):
        raise ValueError('persisted delivery CFV differs')
    if qualified:
        assert_production_cfv(nets)
    return nets, decision


def reserve(root, run_id, generation):
    """Shared numbering lock with the original packaging tools."""
    root.mkdir(parents=True, exist_ok=True)
    ledger = root/'.round3_reservations'
    ledger.mkdir(exist_ok=True)
    receipt = ledger/(hashlib.sha256(run_id.encode()).hexdigest() + f'.g{generation}.json')
    with (root/'.evening_release.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = root/'EVENING_RELEASE_COUNTER.json'
        previous = read(counter) if counter.exists() else dict(last=25)
        occupied = [int(m.group(1)) for p in root.iterdir()
                    if (m := re.match(r'(?:UPLOAD_)?V(\d+)(?:_|$)', p.name))]
        for old in ledger.glob('*.json'):
            occupied.extend(read(old)['versions'].values())
        last = max([25, previous['last'], *occupied])
        if receipt.exists():
            reservation = read(receipt)
            if reservation['run_id'] != run_id or reservation['generation'] != generation:
                raise ValueError('package reservation mismatch')
        else:
            pending = previous.get('round3_reservation',{})
            if pending.get('run_id')==run_id and pending.get('generation')==generation:
                reservation=pending
            else:
                day = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d')
                reservation = dict(run_id=run_id, generation=generation, day=day,
                                   versions={arm: last+1+i for i,arm in enumerate(ARMS)})
                # Burn the range in the shared counter first. Other packaging
                # commands consult that counter even if this process crashes.
                previous=dict(last=max(reservation['versions'].values()),
                              round3_reservation=reservation)
                cfv.atomic_json(counter,previous)
            cfv.atomic_json(receipt, reservation)
        high = max(last, *reservation['versions'].values())
        if high > previous['last']:
            cfv.atomic_json(counter, dict(last=high, reserved_by=run_id, generation=generation))
        return reservation


def prepared(root, work, reservation, arm, checkpoint, nets, profile):
    generation = reservation['generation']
    version = reservation['versions'][arm]
    release = f'V{version}_CFV_R3_G{generation}_{arm}_{reservation["day"]}'
    target = root/f'UPLOAD_{release}'
    staging = root/'.round3_staging'/release
    source_name, model_name = f'V{version}_SOURCE.zip', f'V{version}_MODEL.npz'
    record = work/'PREPARED.json'
    binding = dict(run_id=reservation['run_id'], release=release, version=version, arm=arm,
                   checkpoint_sha256=cfv.digest(checkpoint), cfv_sha256=cfv.digest(nets), profile=profile)
    if record.exists():
        prior = read(record)
        if prior['binding'] != binding:
            raise ValueError('staged archive input binding changed')
    else:
        if target.exists():
            raise FileExistsError(target)
        staging.parent.mkdir(parents=True, exist_ok=True)
        if not staging.exists():
            tmp = Path(tempfile.mkdtemp(prefix='.'+release+'-',dir=staging.parent))
            try:
                export_botzone_weights(checkpoint,tmp/model_name,cfv_nets=nets,release_id=release,
                                       online_profile=profile,deployment_guard=V102_R7_GUARD)
                pack_botzone_zip(tmp/source_name,release_id=release)
                os.rename(tmp,staging)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
        with np.load(staging/model_name,allow_pickle=False) as p:
            meta = json.loads(p['meta_json'].tobytes())
        if meta['deployment_release']!=release or meta['settings']['evening_online']!=profile:
            raise ValueError('exported package metadata binding differs')
        prior = dict(binding=binding,source_sha256=cfv.digest(staging/source_name),
                     model_sha256=cfv.digest(staging/model_name))
        cfv.atomic_json(record, prior)
    folder = target if target.exists() else staging
    for kind,name in (('source',source_name),('model',model_name)):
        if cfv.digest(folder/name) != prior[kind+'_sha256']:
            raise ValueError('prepared package bytes changed: '+str(folder/name))
    return dict(prior,source=str(folder/source_name),model=str(folder/model_name),
                staging=str(staging),target=str(target))


def disposition(decision, arena, latency):
    if not decision['qualified']:
        return 'not promoted — '+decision['reason'], False
    if not arena.get('significant_win_rate_improvement'):
        return 'not promoted — paired V22 improvement not demonstrated', False
    if not latency.get('gate',{}).get('hard_passed'):
        return 'not promoted — exact-byte runtime gate not passed', False
    return 'promoted', True


def publish(prep, work, decision, arena, latency, verify):
    b = prep['binding']
    release, version = b['release'], b['version']
    target, staging = Path(prep['target']), Path(prep['staging'])
    status, promoted = disposition(decision, arena, latency)
    source_name, model_name = Path(prep['source']).name, Path(prep['model']).name
    result = dict(version=version, release=release, arm=b['arm'], status=status, promoted=promoted,
                  recommended_deployment=release if promoted else BASELINE_RELEASE,
                  folder=str(target), source=str(target/source_name),model=str(target/model_name),
                  source_sha256=prep['source_sha256'],model_sha256=prep['model_sha256'],
                  arena=arena, exact_byte_runtime=latency, cfv_input=decision)
    record = work/'PUBLISHED.json'
    if record.exists():
        if read(record) != result:
            raise ValueError('published disposition or result changed')
    else:
        if target.exists():
            raise FileExistsError(target)
        source_instruction = '评估文件。建议部署 '+BASELINE_RELEASE+'。' if not promoted else '上传以下两个文件。'
        note = (f'版本：V{version}（内部 R3），release={release}\n\n'
                f'状态：{status}\n建议 Bot 名称：V{version} (CFV Round 3 {b["arm"]})\n'
                f'{source_instruction}\n\n'
                f'{source_name} SHA256：{result["source_sha256"]}\n'
                f'{model_name} SHA256：{result["model_sha256"]}\n\n'
                f'上线 debug 应以 py 开头，并显示：ok {model_name} release={release} wiz=v2/\n'
                f'核验命令：python3 "{verify}" --url "比赛URL" --expected "{release}"\n')
        (staging/'UPLOAD_README.zh.md').write_text(note)
        # Record the complete publication before the atomic rename. Recovery
        # checks the hashes of the same archive and model, without re-export.
        cfv.atomic_json(record,result)
    if not target.exists():
        os.rename(staging,target)
    for kind in ('source','model'):
        if cfv.digest(result[kind])!=result[kind+'_sha256']:
            raise ValueError('published archive or model changed')
    return result

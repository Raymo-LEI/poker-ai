"""Crash-safe V26/V27/V28 reservation and publication for this authorized run."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
from alphanl.stage2 import cfv
from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
from alphanl.stage2.deployment_guard import V102_R7_GUARD
from server_arena import assert_production_cfv


def reserve(root,out,run_id,ranking):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=Path(out)/'PACKAGE_RESERVATION.json'
    with (root/'.evening_release.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if path.exists():
            receipt=json.loads(path.read_text())
            if receipt['run_id']!=run_id or receipt['ranking']!=ranking or receipt['root']!=str(root.resolve()):
                raise ValueError('packaging reservation identity differs')
            return receipt
        import re
        occupied=[int(m.group(1)) for p in root.iterdir() for m in [re.match(r'(?:UPLOAD_)?V(\d+)(?:_|$)',p.name)] if m]
        counter=root/'EVENING_RELEASE_COUNTER.json'
        last=json.loads(counter.read_text()).get('last',21) if counter.exists() else 21
        if max([last]+occupied)>=26:raise ValueError('V26-V28 are already consumed; refusing version reuse')
        day=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d')
        receipt=dict(run_id=run_id,root=str(root.resolve()),ranking=ranking,day=day,
                     versions={str(26+i):name for i,name in enumerate(ranking)})
        # The reservation itself persists before export, so a restart does not
        # attempt to allocate the already consumed versions a second time.
        cfv.atomic_json(path,receipt)
        cfv.atomic_json(counter,dict(last=28,reserved_by=run_id,versions=receipt['versions']))
        return receipt


def package_one(checkpoint,nets,root,out,reservation,version,arm,profile,promoted,gate):
    assert_production_cfv(nets)
    root=Path(root);out=Path(out)
    release='V%d_SERVER_ARENA_%s_%s'%(version,arm,reservation['day'])
    name='UPLOAD_V%d_SERVER_ARENA_%s_%s'%(version,arm,reservation['day'])
    target=root/name;receipt=out/('V%d_PUBLISHED.json'%version)
    excluded=out/('V%d_EXCLUDED.json'%version)
    if excluded.exists():
        saved=json.loads(excluded.read_text())
        if saved.get('run_id')!=reservation['run_id'] or saved.get('arm')!=arm:raise ValueError('foreign exclusion receipt')
        return saved
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('run_id')!=reservation['run_id'] or saved.get('arm')!=arm:raise ValueError('foreign publication receipt')
        for key in ('source','model'):
            if cfv.digest(saved[key])!=saved[key+'_sha256']:raise ValueError('published package was modified')
        return saved
    # If publication finished just before the process crashed, recover its
    # internal receipt rather than recreating or overwriting formal bytes.
    published=root/('.V%d_SERVER_RECEIPT.json'%version)
    if target.exists():
        if not published.exists():raise ValueError('occupied upload directory has no run-bound receipt')
        saved=json.loads(published.read_text())
        if saved.get('run_id')!=reservation['run_id']:raise ValueError('foreign package collision')
        for key in ('source','model'):
            if cfv.digest(saved[key])!=saved[key+'_sha256']:raise ValueError('published bytes differ')
        cfv.atomic_json(receipt,saved);return saved
    staging=out/('package_staging_V%d'%version)
    if published.exists():
        saved=json.loads(published.read_text())
        if saved.get('run_id')!=reservation['run_id'] or saved.get('arm')!=arm:raise ValueError('foreign staged receipt')
        for key in ('source','model'):
            staged=staging/Path(saved[key]).name
            if not staged.exists() or cfv.digest(staged)!=saved[key+'_sha256']:raise ValueError('lost/corrupt staged publication')
        os.rename(staging,target);cfv.atomic_json(receipt,saved);return saved
    if staging.exists():shutil.rmtree(staging)
    staging.mkdir()
    source=staging/('V%d_SOURCE.zip'%version);model=staging/('V%d_MODEL.npz'%version)
    export_botzone_weights(checkpoint,model,cfv_nets=nets,release_id=release,online_profile=profile,deployment_guard=V102_R7_GUARD)
    pack_botzone_zip(source,release_id=release)
    with zipfile.ZipFile(source) as z:
        for filename in z.namelist():
            if filename.endswith('.py'):compile(z.read(filename),filename,'exec')
    with np.load(model,allow_pickle=False) as blob:
        meta=json.loads(blob['meta_json'].tobytes())
        if meta['deployment_release']!=release or meta['settings']['evening_online']!=profile:
            raise ValueError('formal release/profile binding differs')
    latency=gate(dict(source=str(source),model=str(model)),out/('V%d_exact_bytes_latency.json'%version))
    if not latency['gate']['hard_passed']:
        saved=dict(run_id=reservation['run_id'],version=version,arm=arm,excluded=True,reason='hard latency gate on exact released bytes',latency=latency)
        cfv.atomic_json(excluded,saved)
        return saved
    label='promoted' if promoted else 'not promoted'
    source_sha=cfv.digest(source);model_sha=cfv.digest(model)
    verify=Path(__file__).resolve().with_name('verify_evening_release.py')
    note=('版本：V%d（内部 R1），release=%s\n\n状态：%s\n评测臂：%s\n建议 Bot 名称：V%d (%s)\n\n'
          '上传 V%d_SOURCE.zip 与 V%d_MODEL.npz。\n源码 SHA256：%s\n模型 SHA256：%s\n\n'
          '上线 debug 应以 py 开头，并包含：ok V%d_MODEL.npz release=%s wiz=v2/\n\n'
          '核验命令：python3 "%s" --url "比赛URL" --expected "%s"\n')%(version,release,label,arm,version,label,
          version,version,source_sha,model_sha,version,release,verify,release)
    (staging/'UPLOAD_README.zh.md').write_text(note)
    saved=dict(run_id=reservation['run_id'],version=version,arm=arm,release=release,status=label,promoted=promoted,
               folder=str(target),source=str(target/source.name),model=str(target/model.name),
               source_sha256=source_sha,model_sha256=model_sha,latency=latency)
    # Receipt before rename repairs the crash window between publish and local
    # checkpoint. Only these three upload files enter the visible directory.
    cfv.atomic_json(published,saved)
    os.rename(staging,target)
    cfv.atomic_json(receipt,saved)
    return saved

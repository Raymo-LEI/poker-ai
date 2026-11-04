"""Retired ZIP-transplant packer. Canonical exports own the complete runtime.

Use alphanl.stage2.export.pack_botzone_zip(path, release_id='Vn_...') for a
future versioned package. Allocate an unused increasing Vn, bind source and
model metadata, and satisfy AGENTS.md hashes/debug/replay verification. This
repair PR intentionally creates no upload package or production model.
"""

def prepare_tree(*args,**kwargs):
    raise RuntimeError('retired source transplant: use canonical stage2.export; allocate a fresh Vn and follow AGENTS.md')


def write_model(*args,**kwargs):
    raise RuntimeError('legacy Vleaf/CFV attachment is incompatible with combo_value_v1; generate new labels and train first')


def pack(*args,**kwargs):
    raise RuntimeError('historical V10 packer is retired; no implicit release, upload or deployment')


if __name__=='__main__':
    pack()


import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pack_evening(checkpoint, nets, destination_root, version=None, iterations=None, chance_samples=1, smoke=False, online_profile=None):
    """Build a fresh versioned release; source and weights share the same pipeline.

    A smoke-trained model is explicitly marked in metadata and the one-page
    upload note. Never overwrite or automatically select an existing release.
    """
    import sys
    import re
    from datetime import datetime, timezone, timedelta
    repo=Path(__file__).resolve().parents[2]
    if str(repo/'src') not in sys.path:sys.path.insert(0,str(repo/'src'))
    from alphanl.stage2.export import pack_botzone_zip,export_botzone_weights
    from alphanl.stage2 import cfv
    import fcntl
    root=Path(destination_root)
    root.mkdir(parents=True,exist_ok=True)
    day=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d')
    # One persistent counter across runs; a failed allocation still consumes Vn.
    with (root/'.evening_release.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        counter=root/'EVENING_RELEASE_COUNTER.json'
        previous=json.loads(counter.read_text()).get('last',21) if counter.exists() else 21
        versions=[int(m.group(1)) for p in root.iterdir() for m in [re.match(r'(?:UPLOAD_)?V(\d+)(?:_|$)',p.name)] if m]
        minimum=max([21,int(previous)]+versions)+1
        version=minimum if version is None else int(version)
        if version<minimum:raise ValueError('release version must be new and monotonically increasing')
        release='V%d_EVENING_V2_%s%s'%(version,day,'_SMOKE' if smoke else '')
        folder=root/('UPLOAD_V%d_EVENING_V2_%s'%(version,day))
        folder.mkdir()
        cfv.atomic_json(counter,dict(last=version,release=release))
    weights,_=cfv.load_nets(nets)
    cfv.validate_weights(weights)
    model=folder/('V%d_MODEL.npz'%version)
    source=folder/('V%d_SOURCE.zip'%version)
    export_botzone_weights(checkpoint,model,cfv_nets=nets,release_id=release,
                            evening_iterations=iterations,chance_samples=chance_samples,online_profile=online_profile)
    pack_botzone_zip(source,release_id=release)
    required={'stage2_runtime/deep_solver_v2.py','stage2_runtime/cfv.py','stage2_runtime/evening_strategy.py','stage2_runtime/wizard.py'}
    with zipfile.ZipFile(source) as archive:
        if not required<=set(archive.namelist()):raise ValueError('incomplete runtime bundle')
        for name in archive.namelist():
            if name.endswith('.py'):compile(archive.read(name),name,'exec')
    with np.load(model,allow_pickle=False) as blob:
        meta=json.loads(blob['meta_json'].tobytes())
        if meta['deployment_release']!=release:raise ValueError('release binding failed')
        cfv.validate_weights(blob)
    verify=repo/'scripts/sprint24/verify_evening_release.py'
    note='''版本：V{v}（内部 R1），release={release}

建议 Bot 名称：V{v} (Evening V2{tag})

上传 V{v}_SOURCE.zip 为源码，V{v}_MODEL.npz 为用户存储模型。
源码 SHA256：{sh}
模型 SHA256：{mh}

上线后 debug 必须以 py 开头，并包含：
ok V{v}_MODEL.npz release={release} wiz=v2/

比赛核验命令（替换 URL）：
python3 "{verify}" --url "比赛URL" --expected "{release}"

状态：{status}
特征契约：{schema}；CFV 标签单位：zero_sum_root_bb。
'''.format(v=version,release=release,tag=' SMOKE' if smoke else '',sh=_digest(source),mh=_digest(model),verify=verify,
           status='仅小样本闭环验收，不代表棋力或 Botzone 平台性能通过。' if smoke else '须结合独立评测与平台验收使用。',schema=cfv.SCHEMA)
    (folder/'UPLOAD_README.zh.md').write_text(note)
    return dict(version=version,release=release,folder=str(folder),source=str(source),model=str(model),
                source_sha256=_digest(source),model_sha256=_digest(model),smoke=smoke)

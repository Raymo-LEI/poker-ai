"""Explicit experiment identity shared by launchers and the standalone monitor."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import UUID, uuid4

FORMAT = 'overwatch_experiment_v1'
KINDS = ('formal', 'development', 'smoke', 'benchmark', 'evaluation')
LABELS = {'formal':'正式', 'development':'开发', 'smoke':'Smoke', 'benchmark':'Benchmark', 'evaluation':'评测', 'unclassified':'未标记'}


def read_experiment(directory):
    root = Path(directory).resolve()
    path = root / 'experiment.json'
    unknown = {'kind':'unclassified', 'is_formal':False, 'label':LABELS['unclassified']}
    try:
        if not path.is_file() or not path.resolve().is_relative_to(root) or path.stat().st_size > 16384:
            return unknown
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('format') != FORMAT or value.get('kind') not in KINDS:
            return unknown
        ident = str(UUID(value['experiment_id']))
        return {'format':FORMAT, 'experiment_id':ident, 'kind':value['kind'],
                'name':str(value.get('name') or root.name)[:200], 'stage':value.get('stage'),
                'registered_at':value.get('registered_at'), 'labeled_at':value.get('labeled_at'),
                'is_formal':value['kind']=='formal', 'label':LABELS[value['kind']]}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return unknown


def write_experiment(directory, *, kind=None, name=None, stage=None, relabel=False):
    root = Path(directory)
    existing = read_experiment(root)
    if existing.get('experiment_id'):
        if not relabel:
            if kind is not None and kind != existing['kind']:
                raise ValueError('experiment kind differs; use label_experiment.py to explicitly relabel')
            return existing
    elif (root/'experiment.json').exists():
        raise ValueError('invalid experiment.json; preserve and repair it explicitly')
    kind = kind or 'development'
    if kind not in KINDS:
        raise ValueError('invalid experiment kind')
    now = datetime.now(timezone.utc).isoformat()
    value = {'format':FORMAT, 'experiment_id':existing.get('experiment_id',str(uuid4())),
             'kind':kind, 'name':name or existing.get('name') or root.name,
             'stage':stage or existing.get('stage'),
             'registered_at':existing.get('registered_at') or now, 'labeled_at':now}
    root.mkdir(parents=True,exist_ok=True)
    temporary = root/'experiment.json.tmp'
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(root/'experiment.json')
    return read_experiment(root)


def experiment_tags(experiment):
    return {'experiment_id':experiment['experiment_id'], 'experiment_kind':experiment['kind']}

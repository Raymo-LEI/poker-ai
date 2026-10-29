"""Explicit registration, stable identity and monitor admission boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from experiment_metadata import read_experiment, write_experiment, experiment_tags
from monitor_training import Store


def test_unmarked_or_malformed_files_never_become_formal(tmp_path):
    run=tmp_path/'formal_cuda_production'
    run.mkdir()
    (run/'metrics.csv').write_text('iteration,p0_loss\n1,2\n')
    store=Store([tmp_path])
    assert not store.discover()['0/'+run.name]['experiment']['is_formal']
    (run/'experiment.json').write_text('{"kind":"formal","is_formal":true}')
    assert read_experiment(run)['kind']=='unclassified'
    with pytest.raises(ValueError):
        write_experiment(run,kind='formal')


def test_explicit_sidecar_updates_cache_and_preserves_identity(tmp_path):
    run=tmp_path/'candidate';run.mkdir()
    log=run/'metrics.csv';log.write_text('iteration,p0_loss\n1,2\n')
    before=log.read_bytes()
    store=Store([tmp_path]);old=store.load('0/candidate')
    first=write_experiment(run,kind='formal',name='正式 Stage 2',stage='stage2')
    new=store.load('0/candidate')
    assert new is not old and new['experiment']['is_formal']
    assert new['experiment']['name']=='正式 Stage 2'
    assert store.catalog()['experiments'][0]['experiment']['experiment_id']==first['experiment_id']
    assert write_experiment(run)['experiment_id']==first['experiment_id']
    assert experiment_tags(first)['experiment_kind']=='formal'
    with pytest.raises(ValueError):
        write_experiment(run,kind='smoke')
    second=write_experiment(run,kind='smoke',relabel=True)
    assert second['experiment_id']==first['experiment_id'] and not second['is_formal']
    assert before==log.read_bytes()


def test_default_development_and_symlink_marker_are_not_formal(tmp_path):
    real=tmp_path/'real'
    value=write_experiment(real)
    assert value['kind']=='development' and not value['is_formal']
    external=tmp_path/'other';write_experiment(external,kind='formal')
    (real/'experiment.json').unlink()
    (real/'experiment.json').symlink_to(external/'experiment.json')
    assert not read_experiment(real)['is_formal']


def test_cfr_cli_writes_formal_identity_into_real_csv(tmp_path):
    import csv
    import os
    import subprocess
    config=tmp_path/'tiny.json'
    config.write_text(json.dumps({'iterations':1,'traversals_per_player':1,'train_steps':1,
        'minibatch_size':4,'buffer_capacity':32,'device':'cpu',
        'model':{'d_model':32,'nhead':2,'num_layers':1,'dim_feedforward':64,'max_seq_len':24,'dropout':0.}}))
    run=tmp_path/'formal'
    result=subprocess.run([sys.executable,str(SCRIPTS/'train_cfr.py'),'--config',str(config),
        '--out-dir',str(run),'--experiment-kind','formal','--experiment-name','registered CPU fixture'],
        env={**os.environ,'PYTHONPATH':str(SCRIPTS.parent/'src'),'OMP_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    identity=read_experiment(run)
    assert identity['is_formal'] and identity['stage']=='stage1'
    with (run/'metrics.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert rows and all(row['experiment_id']==identity['experiment_id'] for row in rows)
    assert all(row['experiment_kind']=='formal' for row in rows)

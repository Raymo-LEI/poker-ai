"""Live resource logging must not invalidate an exact learning-state restore."""
import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import os
import json

import torch

from alphanl.cfr import SDCFRTrainer
from test_cfr_trainer import TinyGame,config


def test_resume_return_is_deterministic_while_all_live_telemetry_is_logged(tmp_path,monkeypatch):
    import alphanl.cfr.study as study
    readings=iter([
        dict(ram_current_bytes=100,vram_used_bytes=10,vram_peak_allocated_bytes=1),
        dict(ram_current_bytes=200,vram_used_bytes=20,vram_peak_allocated_bytes=2),
        dict(ram_current_bytes=999,vram_used_bytes=99,vram_peak_allocated_bytes=9)])
    monkeypatch.setattr(study,'memory_usage',lambda:next(readings))
    cfg=config(tmp_path/'original')
    with SDCFRTrainer(cfg,root_factory=lambda seed:TinyGame()) as trainer:
        trainer.run_iteration();trainer.save_checkpoint(tmp_path/'checkpoint.pt')
        expected=trainer.run_iteration()
        weights={p:{k:v.clone() for k,v in m.state_dict().items()} for p,m in trainer.models.items()}
        assert trainer.last_resource_usage['ram_current_bytes']==200
    with SDCFRTrainer(replace(cfg,out_dir=str(tmp_path/'resumed')),root_factory=lambda seed:TinyGame()) as restored:
        restored.load_checkpoint(tmp_path/'checkpoint.pt')
        actual=restored.run_iteration()
        assert actual==expected
        assert not set(('ram_current_bytes','vram_used_bytes','vram_peak_allocated_bytes')) & set(actual)
        assert restored.last_resource_usage['ram_current_bytes']==999
        assert restored.last_timing['vram_used_bytes']==99
        for p,model in restored.models.items():
            for key,value in weights[p].items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    for filename in ('metrics.csv','performance.csv'):
        with (tmp_path/'resumed'/filename).open() as f:rows=list(csv.DictReader(f))
        assert rows[-1]['ram_current_bytes']=='999' and rows[-1]['vram_peak_allocated_bytes']=='9'


def test_r06_cpu_entry_hides_inherited_cuda_without_changing_parent_environment():
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    for name in ('a5.json','a6.json'):
        p=subprocess.run([sys.executable,str(repo/'eval/r06/launch_pilot.py'),'--manifest',
                          str(repo/'eval/r06/manifests'/name),'--execute'],env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stderr+p.stdout
        result=json.loads(p.stdout)
        assert result['cpu_execute_ok'] and result['started_gpu'] is False and result['gpu_hours_used']==0
    assert env['CUDA_VISIBLE_DEVICES']=='0'

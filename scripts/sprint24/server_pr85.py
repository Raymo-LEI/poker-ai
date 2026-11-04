"""Build the frozen PR85 runtime for measurements, never for the final arena."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

PR85_SHA='257a7623dc399a5e63b272f4d7143b1f19160eea'

BUILD=r'''
import os,sys,json
from pathlib import Path
archive,checkpoint,folder,cpus=sys.argv[1:]
sys.path[:0]=[archive+'/src',archive+'/scripts/sprint24']
os.sched_setaffinity(0,set(map(int,cpus.split(','))))
import torch
torch.set_num_threads(1)
from overnight_data import phase_b
from overnight_distill import bind_champion,expand
from overnight_production import ONLINE
from alphanl.stage2.train import load_checkpoint,save_checkpoint
from alphanl.stage2.export import export_botzone_weights,pack_botzone_zip
from alphanl.stage2.deployment_guard import V102_R7_GUARD
folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
model,_=load_checkpoint(checkpoint,device='cpu')
wide=expand(bind_champion(model,model));save_checkpoint(folder/'wide.pt',wide)
phase_b(checkpoint,folder/'pilot',dict(river_n=8,turn_n=8,holdout_n=4,net_epochs=1),2,'cpu',True)
releases={}
for mode in ('as_built','with_hud'):
    d=folder/mode;d.mkdir()
    profile=dict(ONLINE)
    if mode=='with_hud':profile['context_prior']=True
    release='V999_PR85_'+mode.upper()
    source=d/'V999_SOURCE.zip';weights=d/'V999_MODEL.npz'
    export_botzone_weights(folder/'wide.pt',weights,cfv_nets=folder/'pilot/cfv_nets.npz',release_id=release,
        online_profile=profile,deployment_guard=V102_R7_GUARD)
    pack_botzone_zip(source,release_id=release)
    releases[mode]=dict(source=str(source),model=str(weights),profile=profile)
(folder/'releases.json').write_text(json.dumps(releases,indent=2))
'''


def build(repo,checkpoint,folder,worker_cpus):
    folder=Path(folder);archive=folder/'pr85_source';archive.mkdir(parents=True,exist_ok=True)
    content=subprocess.check_output(['git','-C',str(repo),'archive','--format=tar',PR85_SHA])
    with tarfile.open(fileobj=io.BytesIO(content)) as tar:tar.extractall(archive,filter='data')
    env=dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    subprocess.run([sys.executable,'-c',BUILD,str(archive),str(checkpoint),str(folder),','.join(map(str,worker_cpus))],
                   check=True,env=env)
    return json.loads((folder/'releases.json').read_text())

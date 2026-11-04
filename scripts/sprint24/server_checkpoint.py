"""Atomic stage receipts bound to immutable inputs and output content hashes."""
import datetime
import json
from pathlib import Path
import shutil
import time
from alphanl.stage2 import cfv


class Checkpoints:
    def __init__(self,out,identity,resume=False):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.path=self.out/'CHECKPOINTS.json'
        if self.path.exists():
            if not resume:raise FileExistsError('use --resume for an existing production run')
            self.state=json.loads(self.path.read_text())
            if self.state['identity']!=identity:raise ValueError('resume identity differs from locked run')
        else:
            if resume:raise FileNotFoundError('resume requested without CHECKPOINTS.json')
            self.state=dict(identity=identity,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages={})
            self.save()
    def save(self):cfv.atomic_json(self.path,self.state)
    def age(self):
        return (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(self.state['started_utc'])).total_seconds()
    def run(self,name,function,clean=True):
        stages=self.state['stages']
        if name in stages:
            receipt=stages[name]
            for path,digest in receipt['artifacts'].items():
                if not Path(path).is_file() or cfv.digest(path)!=digest:
                    raise ValueError('checkpoint artifact missing or changed: '+path)
            print('RESUME_STAGE',name,flush=True)
            return receipt['result']
        folder=self.out/'stages'/name
        if clean and folder.exists():shutil.rmtree(folder)
        folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();result=function(folder)
        # Published weights, manifests and each match row are checked on resume.
        artifacts={str(p):cfv.digest(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        stages[name]=dict(result=result,artifacts=artifacts,seconds=time.monotonic()-started)
        self.save()
        return result

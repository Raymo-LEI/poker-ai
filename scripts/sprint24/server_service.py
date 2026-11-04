"""Controller for the separate NumPy-only process; no forked Torch image."""
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import tempfile


class Service:
    def __init__(self,release,core,folder=None):
        self.temp=tempfile.TemporaryDirectory(prefix='s24-')
        self.address=str(Path(self.temp.name)/'engine.sock')
        command=[sys.executable,str(Path(__file__).with_name('server_numpy_runtime.py')),'--zip',release['source'],
                 '--model',release['model'],'--socket',self.address,'--core',str(core)]
        self.stderr=None if folder is None else (Path(folder)/'inference_server.stderr.log').open('a')
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=self.stderr,text=True)
        line=self.process.stdout.readline()
        if not line:
            self.close();raise RuntimeError('NumPy inference service failed to start')
        self.info=json.loads(line)
        if not self.info.get('ready') or self.info['affinity']!=[core] or not self.info.get('numpy_only'):
            self.close();raise RuntimeError('inference service isolation failed')
        self.connection=Client(self.address,family='AF_UNIX')
    def infer(self,payload,seed):
        self.connection.send(dict(command='infer',payload=payload,seed=seed))
        if not self.connection.poll(120):raise TimeoutError('inference service stalled')
        result=self.connection.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def meta(self):
        self.connection.send({'command':'info'});return self.connection.recv()
    def close(self):
        if hasattr(self,'connection'):
            try:self.connection.send({'command':'shutdown'})
            except (OSError,EOFError):pass
        if hasattr(self,'process'):
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.process.stdout:self.process.stdout.close()
        if hasattr(self,'connection'):self.connection.close()
        if self.stderr:self.stderr.close()
        self.temp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

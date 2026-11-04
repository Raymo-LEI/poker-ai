"""NumPy-only ZIP loader and isolated inference service. Never imports alphanl/Torch."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
from multiprocessing.connection import Listener,Client
from pathlib import Path
import queue
import random
import resource
import sys
import threading
import time
import types
import zipfile
import zipimport
import numpy as np


def peak_rss_mb():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value)/(1024.**2 if sys.platform=='darwin' else 1024.)


class NumpyBundle:
    def __init__(self,source,model):
        self.source=Path(source);self.model=Path(model)
        name='_server_numpy_'+hashlib.sha256(self.source.read_bytes()).hexdigest()[:12]
        package=types.ModuleType(name);package.__path__=[];sys.modules[name]=package
        self.modules={}
        with zipfile.ZipFile(source) as archive:
            # Canonical archives store their modules in dependency order.
            for filename in archive.namelist():
                if not filename.startswith('stage2_runtime/') or not filename.endswith('.py'):continue
                short=Path(filename).stem
                if short=='__init__':continue
                module=types.ModuleType(name+'.'+short);module.__package__=name
                module.__file__=str(source)+'/'+filename;module.__loader__=zipimport.zipimporter(str(source))
                sys.modules[module.__name__]=module;setattr(package,short,module)
                exec(compile(archive.read(filename),filename,'exec'),module.__dict__);self.modules[short]=module
        with np.load(model,allow_pickle=False) as blob:
            self.meta=json.loads(blob['meta_json'].tobytes())
            weights={k:np.array(blob[k]) for k in blob.files if k!='meta_json'}
        self.modules['cfv'].validate_embedded(weights,self.meta['cfv'])
        self.runtime=self.modules['runtime']
        self.agent=self.runtime.Agent(weights,self.meta['config'],self.meta['settings'],self.meta['numeric_version'],self.meta.get('deployment_guard'))
        self.agent.release_id=self.meta.get('deployment_release','development_unreleased')
        self.agent.decision_identity+=':'+self.agent.release_id;self.agent.force_argmax=False
    def respond(self,payload):return self.runtime.respond(self.agent,payload,bool(self.meta.get('safety',False)))
    def debug(self):
        return 'py%s np%s ok %s release=%s %s'%(sys.version.split()[0],np.__version__,self.model.name,self.meta['deployment_release'],getattr(self.agent,'last_wizard',''))


def inference(bundle,payload,seed):
    random.seed(seed)
    started=time.perf_counter();body=bundle.respond(payload);seconds=time.perf_counter()-started
    agent=bundle.agent;trace=getattr(agent,'last_evening',{})
    return dict(body=body,diagnostics=dict(guard=agent.last_guard,
        before=getattr(agent,'guard_before',np.zeros(5)).tolist(),after=getattr(agent,'guard_after',np.zeros(5)).tolist(),
        search_audit=agent.search_audit,trace={k:trace.get(k) for k in ('search','support','cfv_calls','reason','match_prior','iterations','prior_precomputations')},
        debug=bundle.debug(),engine_s=seconds,rss_mb=peak_rss_mb()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--model',required=True)
    p.add_argument('--socket',required=True);p.add_argument('--core',required=True,type=int);a=p.parse_args()
    os.sched_setaffinity(0,{a.core});bundle=NumpyBundle(a.zip,a.model)
    if 'torch' in sys.modules:raise RuntimeError('inference service must remain NumPy-only')
    listener=Listener(a.socket,family='AF_UNIX');os.chmod(a.socket,0o600)
    pending=queue.Queue();connections=[]
    def read(connection):
        try:
            while True:pending.put((connection,connection.recv()))
        except (EOFError,OSError):pass
    def accept():
        try:
            while True:
                connection=listener.accept();connections.append(connection)
                threading.Thread(target=read,args=(connection,),daemon=True).start()
        except (EOFError,OSError):pass
    threading.Thread(target=accept,daemon=True).start()
    print(json.dumps(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),rss_mb=peak_rss_mb(),numpy_only=True)),flush=True)
    try:
        while True:
            connection,message=pending.get()
            try:
                command=message.get('command','infer')
                if command=='shutdown':connection.send({'stopped':True});break
                if command=='info':connection.send(dict(meta=bundle.meta,rss_mb=peak_rss_mb()));continue
                connection.send(inference(bundle,message['payload'],message['seed']))
            except BaseException as exc:
                connection.send(dict(error=type(exc).__name__+': '+str(exc)))
    finally:
        listener.close()
        for connection in connections:connection.close()

if __name__=='__main__':main()

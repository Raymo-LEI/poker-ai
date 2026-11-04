"""NumPy-only ZIP loader and isolated inference service. Never imports alphanl/Torch."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import importlib
import importlib.machinery
import json
from multiprocessing.connection import Listener,Client
from pathlib import Path
import queue
import random
import resource
import sys
import threading
import time
import zipfile
import numpy as np


def peak_rss_mb():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value)/(1024.**2 if sys.platform=='darwin' else 1024.)


class _ZipSourceLoader:
    """Execute packaged stage2_runtime modules and serve their original bytes.

    ``validate_embedded`` hashes sources through ``__loader__.get_data(__file__)``.
    ``__file__`` stays ``<zip>/stage2_runtime/<name>.py`` so that check is unchanged.
    """
    def __init__(self,archive,files):
        self.archive=archive;self.files=files
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        member=module.__spec__.loader_state
        module.__file__=self.archive+'/'+member
        module.__loader__=self
        source=self.files.get(member)
        if source is None:
            if member.endswith('/__init__.py'):return
            raise ImportError(member)
        exec(compile(source,member,'exec'),module.__dict__)
    def get_data(self,pathname):
        path=str(pathname);prefix=self.archive+'/'
        if not path.startswith(prefix):raise OSError(pathname)
        try:return self.files[path[len(prefix):]]
        except KeyError:raise OSError(pathname) from None


class _ZipSourceFinder:
    """Resolve one hashed package from zip members already read into memory."""
    def __init__(self,package,loader):
        self.package=package;self.loader=loader
    def _spec(self,fullname,member,*,package=False):
        # has_location is what makes importlib copy origin onto __file__.
        spec=importlib.machinery.ModuleSpec(fullname,self.loader,origin=self.loader.archive+'/'+member,is_package=package)
        spec.has_location=True
        spec.loader_state=member
        if package:spec.submodule_search_locations=[self.loader.archive+'/stage2_runtime']
        return spec
    def find_spec(self,fullname,path,target=None):
        if fullname==self.package:return self._spec(fullname,'stage2_runtime/__init__.py',package=True)
        prefix=self.package+'.'
        if not fullname.startswith(prefix):return None
        short=fullname[len(prefix):]
        if not short.isidentifier():return None
        member='stage2_runtime/'+short+'.py'
        if member not in self.loader.files:return None
        return self._spec(fullname,member)


def _drop_loaded_package(name):
    sys.meta_path[:]=[finder for finder in sys.meta_path if getattr(finder,'package',None)!=name]
    prefix=name+'.'
    for key in list(sys.modules):
        if key==name or key.startswith(prefix):del sys.modules[key]


def load_modules(source):
    """Import every packaged module. Relative imports do not depend on zip order.

    A meta-path finder loads a dependency the first time a module asks for it,
    and each module is registered before its body runs. Alphabetical member
    order, notably ``allin_mask`` importing ``equity``, therefore works.
    ``__file__`` and ``get_data`` stay available for ``validate_embedded``.
    """
    source=Path(source);archive=str(source)
    name='_server_numpy_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    package=sys.modules.get(name)
    if package is not None and getattr(package,'_server_numpy_ready',False):
        return package._server_numpy_modules
    if package is not None:_drop_loaded_package(name)
    files={}
    with zipfile.ZipFile(source) as archive_file:
        for filename in archive_file.namelist():
            if not filename.startswith('stage2_runtime/') or not filename.endswith('.py'):continue
            if '/' in filename[len('stage2_runtime/'):-3]:
                raise RuntimeError('nested stage2_runtime module is not supported: '+filename)
            files[filename]=archive_file.read(filename)
    missing=[item for item in ('stage2_runtime/runtime.py','stage2_runtime/cfv.py') if item not in files]
    if missing:raise RuntimeError('source zip missing '+', '.join(missing))
    loader=_ZipSourceLoader(archive,files);finder=_ZipSourceFinder(name,loader)
    sys.meta_path.insert(0,finder)
    try:
        importlib.import_module(name)
        modules={}
        for filename in sorted(item for item in files if not item.endswith('/__init__.py')):
            short=filename.rsplit('/',1)[-1][:-3]
            modules[short]=importlib.import_module(name+'.'+short)
        package=sys.modules[name]
        package._server_numpy_modules=modules;package._server_numpy_ready=True
        return modules
    except BaseException:
        _drop_loaded_package(name);raise


class NumpyBundle:
    def __init__(self,source,model):
        self.source=Path(source);self.model=Path(model)
        # Zip namelist order is alphabetical, not a dependency order.
        self.modules=load_modules(self.source)
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

"""Read-only identity check and reproducer for immutable audited releases."""
import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit=args.workspace/'audit_readonly_20261104'
    manifest=json.loads((audit/'manifest.json').read_text())
    identities=[]
    for entry in manifest:
        digest=hashlib.sha256((args.workspace/entry['path']).read_bytes()).hexdigest()
        if digest!=entry['sha256']:
            raise ValueError('legacy identity mismatch: '+entry['path'])
        identities.append(dict(path=entry['path'],sha256=digest,match=True))
    pkg=types.ModuleType('legacy_audit');pkg.__path__=[str(audit/'V15/stage2_runtime')]
    sys.modules[pkg.__name__]=pkg
    ds=importlib.import_module('legacy_audit.deep_solver')
    board=['2s','3h','4d','9c','Ks']
    ranks=[int(ds._rank_with_strength(*hand,board)) for hand in (['5s','6h'],['As','Qh'])]
    pay=np.zeros((1,2,2,3));pay[:,:,:,1]=10.
    opp=np.zeros((1,2,3));opp[:,:,1]=1.
    value,_=ds._dcfr(pay,opp,np.ones((2,2)),np.ones(2)/2,np.ones(2)/2,[1.],40)
    result=dict(identities=identities,old_straight_high_card=ranks,old_forced_10bb=value[:,0].tolist())
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    main()

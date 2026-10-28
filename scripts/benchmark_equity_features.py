"""CPU observation cost for equity features, including cold misses and cache hits."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from alphanl.transformer.hunl import HunlTable, HunlTokenizer
from alphanl.transformer.equity import _postflop, preflop_table


def summary(times):
    return {'n':len(times),'mean_ms':float(np.mean(times)),
            'p50_ms':float(np.median(times)),'p99_ms':float(np.quantile(times,.99))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',type=int,default=128)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.states<2 or args.out.exists():
        p.error('requires >=2 states and a new output file')
    starts,flops=[],[]
    for seed in range(args.states):
        table=HunlTable(seed=seed)
        table.reset()
        starts.append(table.snapshot(1))
        table.step(1)
        table.step(1)
        flops.append(table.snapshot(0))
    preflop_table()  # exclude one-time JSON initialization from steady-state lookup
    legacy=HunlTokenizer()
    feature=HunlTokenizer(equity_features=True,equity_samples=32)
    report={'scope':'observation_encode_only_excludes_model_and_engine',
            'platform':platform.platform(),'numeric_version':feature.numeric_version,
            'equity_samples':32,'cache_maxsize':_postflop.cache_info().maxsize,'results':{}}
    for name,tokenizer,states in [('legacy_preflop',legacy,starts),('equity_preflop',feature,starts),
                                  ('legacy_flop',legacy,flops),('equity_flop_cold',feature,flops),
                                  ('equity_flop_warm',feature,flops)]:
        if name=='equity_flop_cold':
            _postflop.cache_clear()
        times=[]
        for s in states:
            start=time.perf_counter()
            tokenizer.encode(s)
            times.append((time.perf_counter()-start)*1000)
        report['results'][name]=summary(times)
    report['cache']=_postflop.cache_info()._asdict()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

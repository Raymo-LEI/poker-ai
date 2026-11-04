"""Small deterministic CPU acceptance data. Refuses to overwrite an artifact."""
import argparse
import json
import sys
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Portable runtime intentionally has no training-framework import dependency.
pkg=types.ModuleType('repair_runtime');pkg.__path__=[str(ROOT/'src/alphanl/stage2')]
sys.modules['repair_runtime']=pkg
from repair_runtime.search_game import PublicState
from repair_runtime.search_values import solve_label
from repair_runtime.search_preflop import solve_depth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--preflop',action='store_true')
    args=parser.parse_args()
    hs=((('5s','6h'),('Ah','Ad')),(('As','Qh'),('9s','9h')))
    rows=[]
    if args.preflop:
        # Both depths are solved independently, never copied into two slots.
        for depth in (100,200):
            rows.append(solve_depth(depth,hs,([.4,.6],[.7,.3]),iterations=args.iterations,runouts=8))
    else:
        for street,board in ((3,['2s','3h','4d','9c','Ks']),(2,['2s','3h','4d','9c'])):
            for weights in ([.9,.1],[.1,.9]):
                state=PublicState(1000,(100,100),street=street)
                row,_=solve_label(state,board,hs,([.4,.6],weights),iterations=args.iterations)
                rows.append(row)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(dict(generator='math_contract_v1',scope='tiny_numerical_acceptance',rows=rows),handle,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(args.output),rows=len(rows),max_zero_sum_error=max(r['zero_sum_error'] for r in rows))))


if __name__=='__main__':
    main()

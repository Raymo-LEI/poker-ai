"""Evaluate JSONL Botzone payloads through the exact production respond pipeline.

This is decision replay, not a match or a playing-strength measurement. The
source ZIP/entry must be created by the canonical export module. Never transplant
wizard.py into an older runtime. Does not write model or deployment artifacts.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-tree',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--payloads',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source_tree.resolve()))
    import botzone_entry
    from stage2_runtime import runtime
    import os
    os.environ['POKER_STAGE2_WEIGHTS']=str(args.weights.resolve())
    agent,safety,_=botzone_entry.load_agent(runtime)
    for line in args.payloads.read_text().splitlines():
        payload=json.loads(line)
        start=time.perf_counter()
        result=runtime.respond(agent,payload,safety)
        print(json.dumps(dict(response=result['response'],release=agent.release_id,
                              search=agent.search_audit,guard=agent.last_guard,
                              decision_ms=(time.perf_counter()-start)*1000),allow_nan=False))


if __name__=='__main__':
    main()

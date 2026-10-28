"""One-file export plus fresh-process CPU acceptance. No GPU imports in the agent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from alphanl.stage2.export import export_agent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety", action="store_true")
    parser.add_argument("--decisions", type=int, default=100)
    args = parser.parse_args()
    if args.decisions < 10:
        parser.error("at least ten decisions required")
    report = export_agent(args.checkpoint, args.output, safety=args.safety)
    # Import and inference in fresh process; thread environment is set before NumPy.
    code = '''import importlib.util, json, time, sys
started=time.perf_counter()
spec=importlib.util.spec_from_file_location("agent",sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
agent=module.Agent()
load_ms=(time.perf_counter()-started)*1000
# Unique visible-card states avoid measuring only an equity cache hit.
requests=[]
for k in range(int(sys.argv[2])):
    cards=[(k*7+j*9)%52 for j in range(7)]
    board=cards[2:] if k%3==0 else cards[2:5] if k%3==1 else []
    street={0:0,3:1,5:3}[len(board)]
    history=[] if street==0 else [{"player_id":1,"action":0,"round":0},{"player_id":0,"action":0,"round":0}]
    requests.append({"my_id":1,"dealer_id":0,"my_chips":19950 if street==0 else 19900,"my_cards":cards[:2],"public_cards":board,"history":history,"hand":0,"total_win_chips":[0,0]})
times=[]
for request in requests:
    start=time.perf_counter(); module.respond(agent,{"requests":[request]}); times.append((time.perf_counter()-start)*1000)
import numpy as np
print(json.dumps({"decisions":len(times),"p50_ms":float(np.median(times)),"p99_ms":float(np.quantile(times,.99)),"max_ms":max(times),"load_ms":load_ms,"scope":"warm_process_request_projection_context_equity_tokenize_forward_sample","under_15ms":max(times)<15.,"load_under_1s":load_ms<1000.,"torch_imported":"torch" in sys.modules}))
'''
    result = subprocess.run([sys.executable, "-c", code, str(args.output.resolve()), str(args.decisions)],
                            capture_output=True, text=True, check=True)
    report["acceptance"] = json.loads(result.stdout)
    args.output.with_suffix(".acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["acceptance"]["under_15ms"] or not report["acceptance"]["load_under_1s"]:
        raise SystemExit("Local latency gate failed; inspect acceptance report before deployment")

if __name__ == "__main__":
    main()

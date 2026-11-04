"""Conservative single-core complete-match arena cost calibration."""
import argparse
import time
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
if str(REPO/'src') not in sys.path:sys.path.insert(0,str(REPO/'src'))
from alphanl.stage2 import cfv
from evening_d import Bundle,play_judge
from server_resources import pin

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--model',required=True)
    p.add_argument('--core',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    pin([a.core]);bundle=Bundle(a.zip,a.model);times=[]
    for seat,opponent in ((0,'TAG_Pro'),(1,'call')):
        start=time.monotonic();result=play_judge(bundle,2026110550,seat,50,opponent,keep_trace=False)
        if not result['complete']:raise RuntimeError('incomplete calibration match')
        times.append(time.monotonic()-start)
    cfv.atomic_json(a.out,dict(seconds=max(times)*1.20,raw_seconds=times,hands=50,inference_parallelism=1))
if __name__=='__main__':main()

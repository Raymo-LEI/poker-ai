"""Label an existing run without loading models or rewriting its historical logs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from experiment_metadata import KINDS, write_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--kind',choices=KINDS,required=True)
    parser.add_argument('--name')
    parser.add_argument('--stage',choices=('stage1','stage2'))
    args = parser.parse_args()
    if not args.run.is_dir() or not any((args.run/name).is_file() for name in (
            'metrics.csv','metrics.jsonl','config.json','status.json','launch_manifest.json','report.json')):
        parser.error('expected an existing run directory containing logs or configuration')
    value = write_experiment(args.run,kind=args.kind,name=args.name,stage=args.stage,relabel=True)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

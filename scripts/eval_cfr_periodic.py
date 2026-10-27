#!/usr/bin/env python3
"""One idempotent milestone poll, intended for a timer/cron or manual invocation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphanl.cfr.evaluation_io import EvaluationBusy
from alphanl.cfr.evaluation_schedule import load_config, run_pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    try:
        status = run_pending(config, repository=ROOT)
    except EvaluationBusy as exc:
        print(str(exc), flush=True)
        return 0
    print(json.dumps({"state": status["state"], "available_iteration": status.get("available_iteration"),
                      "completed_jobs": sum(j["state"] == "complete" for j in status["jobs"].values())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

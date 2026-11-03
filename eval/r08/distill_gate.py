"""Print the distill command and exit 2. Do not run it.

The Teacher gate is not passed in this prep scaffold. ``--execute`` is also
refused, with the same command printed and no subprocess.
"""
from __future__ import annotations

import argparse
import json
import sys

DISTILL_COMMAND = (
    "python3 scripts/distill_cfr.py "
    "--strategies <PENDING_R07_TEACHER_OUT> "
    "--out-dir <NEW_DIR> "
    "--prepare-only"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--teacher-report", help="ignored; a local file cannot open the gate")
    args = parser.parse_args(argv)
    reason = "execute_forbidden" if args.execute else "teacher_gate_not_passed"
    print(DISTILL_COMMAND)
    print(json.dumps({
        "status": "NOT_RUN",
        "reason": reason,
        "teacher_gate": "NOT_PASSED",
        "distill_started": False,
        "command_executed": False,
    }))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

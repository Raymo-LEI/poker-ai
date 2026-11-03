"""Offline GTO Wizard fixture compare.

Writes KL and L1 for recorded action distributions. ``--live`` is off by
default. Passing it does not open a socket: a missing ``GTOWIZARD_API_KEY``
exits 2, and a present key still exits 2 because live play is not a gate for
this prep scaffold. No leaderboard submission and no training side effects.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.gto_wizard.adapter_hunl import AdapterError, prepare_spot  # noqa: E402
from eval.gto_wizard.client import live_refusal  # noqa: E402
from eval.gto_wizard.compare import CompareError, compare_spot  # noqa: E402
from eval.gto_wizard.schema import SchemaError, require_spot  # noqa: E402


def refuse_live() -> int:
    reason = live_refusal(os.environ.get("GTOWIZARD_API_KEY"))
    print(
        json.dumps({
            "status": "REFUSED",
            "reason": reason,
            "network_used": False,
            "leaderboard_submitted": False,
            "training_modified": False,
        }),
        file=sys.stderr,
    )
    return 2


def run_fixture(fixture: Path, out_dir: Path) -> dict:
    document = json.loads(fixture.read_text(encoding="utf-8"))
    spots = document.get("spots")
    if not isinstance(spots, list) or len(spots) < 2:
        raise ValueError("fixture needs at least two spots")
    rows = []
    for spot in spots:
        require_spot(spot)
        prepared = prepare_spot(spot)
        rows.append(compare_spot(spot, prepared))
    report = {
        "format": "gto_wizard_offline_fixture_v1",
        "live": False,
        "network_used": False,
        "leaderboard_submitted": False,
        "training_modified": False,
        "botzone_used": False,
        "hands": len(rows),
        "chips": None,
        "aivat_bb_per_100": None,
        "metrics_status": "OFFLINE_BUCKET_COMPARE",
        "kl": sum(row["kl"] for row in rows) / len(rows),
        "l1": sum(row["l1"] for row in rows) / len(rows),
        "same_bucket_count": sum(bool(row["same_bucket"]) for row in rows),
        "spots": rows,
        "notes": [
            "KL is natural log. L1 is the sum of absolute bucket differences.",
            "chips and AIVAT bb/100 are not measured. This is not a live result.",
            "Botzone is not used. Training loops are not started.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    temporary = out_dir / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out_dir / "report.json")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if args.live:
        return refuse_live()
    if args.fixture is None or args.out_dir is None:
        parser.error("--fixture and --out-dir are required")
    try:
        report = run_fixture(args.fixture, args.out_dir)
    except (AdapterError, CompareError, SchemaError, OSError, ValueError, json.JSONDecodeError):
        print(json.dumps({"status": "REFUSED", "network_used": False, "training_modified": False}), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "OK",
        "hands": report["hands"],
        "kl": report["kl"],
        "l1": report["l1"],
        "live": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

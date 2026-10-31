"""Compare one synthetic teacher distribution with a GTO Wizard fixture spot.

The distribution is the constant ``FAKE_SNAPSHOT`` below. This script does not
open strategy archives, AutoDL outputs, or ``SnapshotPolicy`` weight files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.gto_wizard.adapter_hunl import FakeSnapshot, prepare_spot, refuse_weight_source  # noqa: E402
from eval.gto_wizard.compare import compare_distributions  # noqa: E402
from eval.gto_wizard.schema import require_spot  # noqa: E402

FAKE_SNAPSHOT = {
    "label": "synthetic_not_a_checkpoint",
    "spot_id": "a6_flop_cbet",
    "schema_id": "botzone_a6_v2",
    "probs": [0.0, 0.2, 0.25, 0.2, 0.05, 0.3],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spot-id", default=FAKE_SNAPSHOT["spot_id"])
    args = parser.parse_args(argv)
    if args.spot_id != FAKE_SNAPSHOT["spot_id"]:
        parser.error("this prep script only compares its built-in fake snapshot")
    document = json.loads(args.fixture.read_text(encoding="utf-8"))
    spot = next((item for item in document["spots"] if item["spot_id"] == args.spot_id), None)
    if spot is None:
        print("fixture spot is missing", file=sys.stderr)
        return 2
    require_spot(spot)
    refuse_weight_source(FAKE_SNAPSHOT["probs"])
    prepared = prepare_spot(spot)
    fake = FakeSnapshot(FAKE_SNAPSHOT["schema_id"], FAKE_SNAPSHOT["probs"])
    probs = fake.probabilities(prepared["snapshot"])
    metrics = compare_distributions(
        probs,
        spot["reference_probs"],
        FAKE_SNAPSHOT["schema_id"],
        legal_action_ids=prepared["legal_action_ids"],
    )
    report = {
        "format": "gto_wizard_fake_snapshot_v1",
        "snapshot": FAKE_SNAPSHOT["label"],
        "weights_read": False,
        "autodl_outputs_read": False,
        "live": False,
        "chips": None,
        "aivat_bb_per_100": None,
        "kl": metrics["kl"],
        "l1": metrics["l1"],
        "same_bucket": metrics["same_bucket"],
        "spot_id": args.spot_id,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OK", "kl": report["kl"], "l1": report["l1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

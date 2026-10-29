"""Pack the Stage 2 NumPy forward for Botzone TexasHoldem2p.

Always writes the checkpoint-independent source zip (``__main__.py`` at
its root). With ``--checkpoint`` it also exports ``stage2_model.npz``,
which goes into Botzone user storage, not into the zip.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_ZIP = ROOT / "botzone_bots" / "stage2.zip"


def pack(dest=None):
    from alphanl.stage2.export import pack_botzone_zip
    return pack_botzone_zip(Path(dest) if dest is not None else DEFAULT_ZIP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--checkpoint", type=Path, help="Stage 2 train.pt to export")
    parser.add_argument("--weights-out", type=Path, default=Path("stage2_model.npz"))
    parser.add_argument("--safety", action="store_true",
                        help="move small high-SPR all-in mass to call (deployment intervention)")
    args = parser.parse_args()
    report = {"zip": pack(args.zip)}
    if args.checkpoint is not None:
        from alphanl.stage2.export import export_botzone_weights
        report["weights"] = export_botzone_weights(args.checkpoint, args.weights_out, safety=args.safety)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

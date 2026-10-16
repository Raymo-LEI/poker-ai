#!/usr/bin/env python3
"""Print ``fresh`` or ``resume`` for tools/train_autoresume.ps1.

``fresh`` means the checkpoint directory has no complete PPO ``update_*.pt``.
``resume`` means at least one file passes the same check as ``--resume auto``.
The PowerShell loop calls this on every attempt and must not pass
``--init-weights`` together with ``--resume``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alphanl.transformer.checkpoint import choose_launch_mode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="train.checkpoint_dir, the directory --resume auto scans.",
    )
    args = parser.parse_args()
    mode = choose_launch_mode(args.checkpoint_dir)
    if mode not in ("fresh", "resume"):
        raise SystemExit(f"unexpected launch mode {mode!r}")
    print(mode)


if __name__ == "__main__":
    main()

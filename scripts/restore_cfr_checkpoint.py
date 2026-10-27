#!/usr/bin/env python3
"""Verify and unpack a retained full checkpoint; never loads pickle objects."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alphanl.cfr.retention import restore_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(restore_archive(args.archive, args.out_dir))


if __name__ == "__main__":
    main()

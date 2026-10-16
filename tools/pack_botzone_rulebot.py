"""Pack ``botzone_bots/rulebot`` into a zip Botzone can run.

The zip root contains ``__main__.py``. Other modules stay beside it, which
is the directory layout ``python bot.zip`` and an unzipped folder both
import. Stdlib only; the rule-bot policy is the copy under ``rulebot/``,
not ``src/alphanl``.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "botzone_bots" / "rulebot"
DEFAULT_ZIP = ROOT / "botzone_bots" / "rulebot.zip"


def pack(dest=None):
    """Write the zip and return its path."""

    target = Path(dest) if dest is not None else DEFAULT_ZIP
    files = [
        path
        for path in sorted(SOURCE.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    if not any(path.name == "__main__.py" and path.parent == SOURCE for path in files):
        raise SystemExit("botzone_bots/rulebot/__main__.py is missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(SOURCE).as_posix())
    return target


def main():
    print(pack())


if __name__ == "__main__":
    main()

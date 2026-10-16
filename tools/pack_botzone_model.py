"""Pack ``botzone_bots/model`` into a zip Botzone can run.

The zip root contains ``__main__.py``. Weights stay out of the zip: the
wiki says to upload data files through account storage and read them from
the ``data`` path.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "botzone_bots" / "model"
DEFAULT_ZIP = ROOT / "botzone_bots" / "model.zip"


def pack(dest=None):
    """Write the zip and return its path."""

    target = Path(dest) if dest is not None else DEFAULT_ZIP
    files = [
        path
        for path in sorted(SOURCE.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    if not any(path.name == "__main__.py" and path.parent == SOURCE for path in files):
        raise SystemExit("botzone_bots/model/__main__.py is missing")
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "import torch" in text or "from torch" in text:
            raise SystemExit("%s imports torch" % path.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(SOURCE).as_posix())
    return target


def main():
    print(pack())


if __name__ == "__main__":
    main()

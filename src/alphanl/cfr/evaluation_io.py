"""Atomic reports and process locks shared by offline evaluation entrypoints."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path


class EvaluationBusy(RuntimeError):
    pass


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def output_lock(directory: Path):
    """Fail immediately if another writer owns this output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".eval.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationBusy(f"evaluation output is already locked: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

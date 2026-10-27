"""Compressed restart archives with a shared immutable strategy object store."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from uuid import uuid4

from .checkpoint import _references, sync_directory
from .study import publish, sha256


def archive_due(config, iteration):
    start = config.checkpoint_archive_from
    return start is not None and iteration >= start and (iteration - start) % config.checkpoint_archive_every == 0


def immutable_link_or_copy(source, target):
    temporary = target.with_name(target.name + f".{uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


class ReservedDiskWriter:
    """Check space throughout streaming compression, not only at the start."""
    def __init__(self, handle, directory, reserve):
        self.handle, self.directory, self.reserve = handle, directory, reserve
        self.checked_at = -8 * 1024**2

    def write(self, data):
        position = self.handle.tell()
        if position - self.checked_at >= 8 * 1024**2 or len(data) >= 8 * 1024**2:
            if shutil.disk_usage(self.directory).free < self.reserve + len(data) + 8 * 1024**2:
                raise OSError("checkpoint archive disk reserve reached; existing restart states retained")
            self.checked_at = position
        return self.handle.write(data)

    def __getattr__(self, name):
        return getattr(self.handle, name)


def archive_checkpoint(checkpoint, payload, *, iteration, remaining, min_free_bytes, growth_factor=1.):
    """Caller holds the checkpoint publisher lock, protecting referenced files."""
    checkpoint = Path(checkpoint)
    directory = checkpoint.parent / "checkpoint_archives"
    directory.mkdir(exist_ok=True)
    archive = directory / f"iteration_{iteration:07d}.tar.gz"
    metadata = archive.with_suffix(archive.suffix + ".json")
    if archive.exists() or metadata.exists():
        raise FileExistsError(f"immutable checkpoint archive already exists: {archive}")
    files = {"train.pt": checkpoint}
    strategy_names = set(payload["strategy_archive"].values())
    strategy_objects = {}
    objects = directory / "objects"
    objects.mkdir(exist_ok=True)
    new_objects = []
    for generation in sorted(_references(payload, checkpoint.parent)):
        for path in sorted(generation.rglob("*")):
            if path.is_file():
                name = path.relative_to(checkpoint.parent).as_posix()
                if name in strategy_names:
                    fingerprint = sha256(path)
                    strategy_objects[name] = {"object": fingerprint + ".pt", "sha256": fingerprint,
                                              "bytes": path.stat().st_size}
                else:
                    files[name] = path
    packed_bytes = sum(path.stat().st_size for path in files.values())
    # Leave enough room for the next uncompressed checkpoint publication.
    reserve = min_free_bytes + packed_bytes
    if shutil.disk_usage(directory).free < reserve:
        raise OSError("insufficient disk reserve for compressed checkpoint retention")
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        # Store each historical network once per run rather than repeating the
        # entire growing policy history in every 10-round restart archive.
        for name, entry in strategy_objects.items():
            target = objects / entry["object"]
            if target.exists():
                if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                    raise ValueError("retained strategy object fingerprint mismatch")
                continue
            if shutil.disk_usage(directory).free < reserve + entry["bytes"]:
                raise OSError("checkpoint archive strategy-object disk reserve reached")
            source = checkpoint.parent / name
            immutable_link_or_copy(source, target)
            new_objects.append(target)
        sync_directory(objects)
        with temporary.open("xb") as handle:
            writer = ReservedDiskWriter(handle, directory, reserve)
            with tarfile.open(fileobj=writer, mode="w:gz", compresslevel=1, dereference=True) as bundle:
                for name, path in files.items():
                    bundle.add(path, arcname=name, recursive=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(archive)
        sync_directory(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        for path in new_objects:
            path.unlink(missing_ok=True)
        raise
    compressed_bytes = archive.stat().st_size
    projected = math.ceil(compressed_bytes * max(1., growth_factor) * remaining * 1.5)
    report = {"format": "sd_cfr_restart_archive_v1", "iteration": iteration,
              "archive": archive.name, "archive_sha256": sha256(archive),
              "compressed_bytes": compressed_bytes,
              "uncompressed_bytes": packed_bytes + sum(e["bytes"] for e in strategy_objects.values()),
              "compressed_payload_uncompressed_bytes": packed_bytes,
              "files": {name: path.stat().st_size for name, path in files.items()},
              "strategy_objects": strategy_objects,
              "restore_requires": "archive + its JSON manifest + sibling objects/ directory",
              "remaining_archives": remaining, "projected_remaining_bytes": projected,
              "projection_note": "Measured compression, full-pool growth factor and 50% margin; estimate, not a guarantee.",
              "workspace_reserve_bytes": reserve, "free_bytes": shutil.disk_usage(directory).free,
              "full_restart_state": True}
    publish(metadata, report)
    with metadata.open("rb") as handle:
        os.fsync(handle.fileno())
    sync_directory(directory)
    with (checkpoint.parent / "checkpoint_archives.jsonl").open("a") as handle:
        handle.write(json.dumps(report) + "\n")
    if report["free_bytes"] < reserve + projected:
        raise OSError(f"projected retention exceeds free disk; archive preserved at {archive}; see {metadata}")
    return report


def restore_archive(archive, destination, *, min_free_bytes=2 * 1024**3):
    """Materialize a verified complete restart folder without pickle loading."""
    archive, destination = Path(archive), Path(destination)
    metadata = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
    if metadata.get("format") != "sd_cfr_restart_archive_v1" or sha256(archive) != metadata["archive_sha256"]:
        raise ValueError("checkpoint archive fingerprint mismatch")
    inventory = metadata["files"]
    strategies = metadata.get("strategy_objects", {})
    for name in (*inventory, *strategies):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not (name == "train.pt" or name.startswith("checkpoint_data/")):
            raise ValueError("unsafe checkpoint archive path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < sum(inventory.values()) + sum(e["bytes"] for e in strategies.values()) + min_free_bytes:
        raise OSError("insufficient disk for archive materialization and reserve")
    destination.mkdir(exist_ok=False)
    seen = set()
    try:
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                if (not member.isreg() or member.name not in inventory or member.name in seen
                        or member.size != inventory[member.name]):
                    raise ValueError("checkpoint archive inventory mismatch")
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024**2)
                seen.add(member.name)
        if seen != set(inventory):
            raise ValueError("checkpoint archive is incomplete")
        for name, entry in strategies.items():
            if entry["object"] != entry["sha256"] + ".pt" or len(entry["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in entry["sha256"]):
                raise ValueError("unsafe retained strategy object name")
            source = archive.parent / "objects" / entry["object"]
            if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
                raise ValueError("retained strategy object fingerprint mismatch")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            immutable_link_or_copy(source, target)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination / "train.pt"

"""One polling pass for milestone evaluations; no background training changes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from .evaluation import OPPONENTS
from .evaluation_io import atomic_json, output_lock, sha256_file


DEFAULT_MILESTONES = [50, 100, 200, 400, 500, 600, 800, 1000]
DEFAULT_EVALUATION = dict(pairs=2500, matches=500, health_hands=10000,
                          opponents=list(OPPONENTS), stages=["duplicate", "matches", "health"],
                          threads=1, cache_size=64, equity_samples=64, seed=1,
                          skip_ladder=False)


class ArchiveChanged(RuntimeError):
    """An active publisher replaced the prefix while it was being frozen."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def load_config(path: Path) -> dict:
    raw = json.loads(path.read_text())
    allowed = {"strategies", "out_dir", "milestones", "max_evaluations", "evaluation"}
    if not isinstance(raw, dict) or set(raw) - allowed or not {"strategies", "out_dir"} <= set(raw):
        raise ValueError(f"config requires strategies/out_dir and only these fields: {sorted(allowed)}")
    config = dict(raw)
    for key in ("strategies", "out_dir"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a path string")
        config[key] = (path.parent / config[key]).resolve()
    source, output = config["strategies"], config["out_dir"]
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("strategy input and evaluation output directories must not contain each other")
    milestones = config.get("milestones", DEFAULT_MILESTONES)
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("milestones must be a nonempty integer list")
    for milestone in milestones:
        positive_int(milestone, "milestone")
    if len(set(milestones)) != len(milestones):
        raise ValueError("milestones must be distinct")
    config["milestones"] = sorted(milestones)
    config["max_evaluations"] = positive_int(config.get("max_evaluations", 1), "max_evaluations")
    overrides = config.get("evaluation", {})
    if not isinstance(overrides, dict) or set(overrides) - set(DEFAULT_EVALUATION):
        raise ValueError("unknown evaluation options")
    options = {**DEFAULT_EVALUATION, **overrides}
    for key in ("pairs", "matches", "health_hands", "threads", "cache_size", "equity_samples", "seed"):
        positive_int(options[key], key, 0 if key in ("cache_size", "seed") else 2 if key == "pairs" else 1)
    for key, choices in (("opponents", set(OPPONENTS)), ("stages", {"duplicate", "matches", "health"})):
        values = options[key]
        if (not isinstance(values, list) or not all(isinstance(v, str) for v in values)
                or len(set(values)) != len(values) or set(values) - choices):
            raise ValueError(f"invalid {key}")
    if not options["stages"] or not isinstance(options["skip_ladder"], bool):
        raise ValueError("nonempty stages and boolean skip_ladder required")
    if "matches" in options["stages"] and not options["opponents"]:
        raise ValueError("matches require at least one opponent")
    config["evaluation"] = options
    return config


def read_index(source: Path):
    path = source / "index.json"
    if not path.exists():
        return None
    index = json.loads(path.read_text())
    if not isinstance(index, dict) or index.get("format") != "sd_cfr_archive_v1":
        raise ValueError("unsupported strategy index")
    completed, iterations = index.get("completed_iteration"), index.get("iterations")
    positive_int(completed, "completed_iteration", 0)
    if not isinstance(iterations, list):
        raise ValueError("invalid strategy iterations")
    for iteration in iterations:
        positive_int(iteration, "archived iteration")
    if iterations != sorted(set(iterations)) or (iterations and iterations[-1] > completed):
        raise ValueError("strategy index iterations must be sorted, unique and completed")
    return index


def prefix_fingerprints(source: Path, target: int):
    result = {}
    for iteration in range(1, target + 1):
        name = f"iteration_{iteration:07d}.pt"
        stat = (source / name).stat()
        result[name] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return result


def freeze_prefix(source: Path, destination: Path, target: int) -> Path:
    """Hardlink atomically published snapshots; copy if on another filesystem.

    The trainer replaces files atomically, so a hardlink retains its original
    inode after restore/retraining. Existing frozen prefixes are never updated.
    """
    index = read_index(source)
    expected = list(range(1, target + 1))
    if index is None or index["completed_iteration"] < target:
        raise ArchiveChanged("milestone is no longer complete")
    if [i for i in index["iterations"] if i <= target] != expected:
        raise ValueError("full mixture requires snapshot_every=1 and every snapshot 1..T")
    try:
        before = prefix_fingerprints(source, target)
    except FileNotFoundError as exc:
        raise ArchiveChanged("an indexed snapshot disappeared") from exc
    if destination.exists():
        metadata = json.loads((destination / "archive.json").read_text())
        if (metadata.get("format") != "sd_cfr_eval_frozen_v1"
                or metadata["source"] != str(source) or metadata["source_fingerprints"] != before):
            raise ValueError("training history changed; use a new evaluation out_dir")
        frozen_index = read_index(destination / "strategies")
        if frozen_index != {"format": "sd_cfr_archive_v1", "completed_iteration": target,
                            "iterations": expected}:
            raise ValueError("frozen archive index changed; use a new out_dir")
        for name in before:
            stat = (destination / "strategies" / name).stat()
            if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns] != metadata["frozen_fingerprints"][name]:
                raise ValueError("frozen archive changed; preserve it and use a new out_dir")
        return destination / "strategies"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-", dir=destination.parent))
    try:
        strategies = temporary / "strategies"
        strategies.mkdir()
        for name in before:
            try:
                os.link(source / name, strategies / name)
            except OSError:
                shutil.copyfile(source / name, strategies / name)
        after_index = read_index(source)
        if (after_index is None or after_index["completed_iteration"] < target
                or [i for i in after_index["iterations"] if i <= target] != expected
                or prefix_fingerprints(source, target) != before):
            raise ArchiveChanged("strategy prefix changed while freezing; retry on the next poll")
        atomic_json(strategies / "index.json", {"format": "sd_cfr_archive_v1",
                    "completed_iteration": target, "iterations": expected})
        atomic_json(temporary / "archive.json", {"format": "sd_cfr_eval_frozen_v1",
                    "source": str(source), "iteration": target,
                    "source_fingerprints": before,
                    "frozen_fingerprints": prefix_fingerprints(strategies, target)})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "strategies"


def completed_report(path: Path):
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if (report.get("format") != "sd_cfr_readiness_v1" or report.get("error")
            or not report.get("completed_utc") or not report.get("identity")):
        return None
    for result in report["results"].values():
        raw = path.parent / result["raw_log"]
        if not raw.is_file() or sha256_file(raw) != result.get("raw_log_sha256"):
            raise ValueError(f"completed evaluation has a missing or changed raw log: {raw}")
    return report


def specification(config, target, repository):
    files = [repository / "scripts/eval_cfr_readiness.py", repository / "scripts/eval_cfr_periodic.py",
             *sorted((repository / "src/alphanl").rglob("*.py"))]
    return {"evaluation": config["evaluation"],
            "milestones": [i for i in config["milestones"] if i <= target],
            "source_sha256": {str(p.relative_to(repository)): sha256_file(p) for p in files},
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        **{name: version(name) for name in ("torch", "numpy", "pokerkit", "phevaluator")}}}


def run_pending(config, *, repository: Path, execute=subprocess.run):
    """Run at most max_evaluations new/resumed milestones, under one lock."""
    root, source = config["out_dir"], config["strategies"]
    with output_lock(root):
        path = root / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {
            "format": "sd_cfr_periodic_v1", "strategies": str(source), "jobs": {}}
        if status.get("format") != "sd_cfr_periodic_v1" or status["strategies"] != str(source):
            raise ValueError("out_dir already belongs to a different training run")
        status.update(state="checking", checked_utc=utc_now())
        status.pop("error", None)
        atomic_json(path, status)
        ran = 0
        try:
            index = read_index(source)
            status["available_iteration"] = index["completed_iteration"] if index else None
            for target in config["milestones"]:
                if index is None or target > index["completed_iteration"]:
                    continue
                frozen = freeze_prefix(source, root / "archives" / f"iteration_{target:07d}", target)
                spec = specification(config, target, repository)
                key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
                job_id = f"{target}:{key}"
                output = root / "results" / f"iteration_{target:07d}" / key
                report_path = output / "report.json"
                report = completed_report(report_path)
                job = status["jobs"].setdefault(job_id, {"iteration": target, "report": str(report_path)})
                if report:
                    if job.get("identity", report["identity"]) != report["identity"]:
                        raise ValueError("completed report identity changed")
                    job.update(state="complete", identity=report["identity"],
                               verdict=report["launch_gate"]["verdict"])
                    continue
                command = [sys.executable, str(repository / "scripts/eval_cfr_readiness.py"),
                           "--strategies", str(frozen), "--out", str(output),
                           "--hero-iteration", str(target), "--milestones",
                           ",".join(map(str, spec["milestones"])), "--resume"]
                for name, value in config["evaluation"].items():
                    if name == "skip_ladder":
                        if value:
                            command.append("--skip-ladder")
                    else:
                        command.extend(["--" + name.replace("_", "-"),
                                        ",".join(value) if isinstance(value, list) else str(value)])
                output.mkdir(parents=True, exist_ok=True)
                job.update(state="running", started_utc=utc_now())
                status["state"] = "running"
                atomic_json(path, status)
                environment = {**os.environ, "OMP_NUM_THREADS": str(config["evaluation"]["threads"]),
                               "MKL_NUM_THREADS": str(config["evaluation"]["threads"])}
                with (output / "evaluation.log").open("a") as log:
                    result = execute(command, cwd=repository, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT)
                if result.returncode:
                    job["state"] = "failed"
                    raise RuntimeError(f"evaluation for iteration {target} exited {result.returncode}: {output / 'evaluation.log'}")
                report = completed_report(report_path)
                if report is None:
                    job["state"] = "failed"
                    raise RuntimeError("evaluation exited successfully without a completed report")
                job.update(state="complete", identity=report["identity"], completed_utc=report["completed_utc"],
                           verdict=report["launch_gate"]["verdict"])
                atomic_json(path, status)
                ran += 1
                if ran >= config["max_evaluations"]:
                    break
            status.update(state="idle", finished_utc=utc_now())
        except ArchiveChanged as exc:
            status.update(state="deferred", error=str(exc))
        except Exception as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(path, status)
            raise
        atomic_json(path, status)
        return status

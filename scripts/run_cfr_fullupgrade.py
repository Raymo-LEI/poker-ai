"""Launch one isolated trainer, record resource watermarks and fail visibly.

All strategy snapshots remain in the run. Never removes other runs or retries
failed training automatically. The latest full checkpoint is saved every ten
iterations; the provided preset additionally archives T100/T200/T300.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphanl.cfr.config import CFRConfig


def publish(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def resources():
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "disk_free_bytes": shutil.disk_usage(ROOT).free}
    for field, name in (("ram_current_bytes", "memory.current"), ("ram_limit_bytes", "memory.max")):
        p = Path("/sys/fs/cgroup") / name
        if p.exists() and p.read_text().strip() != "max":
            result[field] = int(p.read_text().strip())
    if shutil.which("nvidia-smi"):
        row = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()[0]
        used, total, utilization = [int(value.strip()) for value in row.split(",")]
        result.update(vram_used_bytes=used*1024**2, vram_total_bytes=total*1024**2, gpu_utilization_percent=utilization)
    return result


def latest_metrics(path):
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    # An active append can leave an incomplete final row; select a full row.
    return next((row for row in reversed(rows) if None not in row.values()), None)


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--poll-seconds", type=float, default=5.)
    p.add_argument("--ram-limit-bytes", type=int, default=90_000_000_000)
    p.add_argument("--vram-limit-bytes", type=int, default=24_000_000_000)
    p.add_argument("--evaluate-on-completion", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluation-pairs", type=int, default=2500)
    args = p.parse_args()
    if not 1 <= args.poll_seconds <= 60 or min(args.ram_limit_bytes, args.vram_limit_bytes) < 1 or args.evaluation_pairs < 2:
        p.error("invalid monitoring interval or memory limit")
    config = CFRConfig.from_json(args.config)
    if args.out_dir is not None:
        config.out_dir = str(args.out_dir)
    out = Path(config.out_dir).resolve()
    config.out_dir = str(out)
    if config.device == "cuda":
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError(f"GPU already has compute processes; new run was not started: {active}")
    out.mkdir(parents=True, exist_ok=False)
    publish(out/"config.json", config.to_dict())
    command = [sys.executable, str(ROOT/"scripts/train_cfr.py"), "--config", str(out/"config.json"), "--threads", "1"]
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    dry = subprocess.run([*command, "--dry-run"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    effective = json.loads(dry.stdout)
    source = {str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()
              for top in ("src", "scripts", "configs") for q in sorted((ROOT/top).rglob("*"))
              if q.is_file() and q.suffix in (".py", ".json")}
    manifest = dict(effective_config=effective, source_sha256=source,
        started_utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),
        ram_limit_bytes=args.ram_limit_bytes, vram_limit_bytes=args.vram_limit_bytes,
        evaluation_pairs=args.evaluation_pairs,
        objective=dict(selfplay_root_probability=1-config.scripted_opponent_prob,
            scripted_root_probability=config.scripted_opponent_prob,
            scripted_opponent_pool=config.scripted_opponent_pool,
            iteration_weight_power=config.iteration_weight_power,
            normalize_iteration_weights=config.normalize_iteration_weights),
        retention=dict(snapshot_every=config.snapshot_every, full_checkpoint_every=config.checkpoint_every,
            archive_from=config.checkpoint_archive_from, archive_every=config.checkpoint_archive_every))
    publish(out/"launch_manifest.json", manifest)
    child = None
    training_complete = False
    high = {"ram_current_bytes": 0, "vram_used_bytes": 0}
    try:
        with (out/"train.log").open("x") as log, (out/"resources.jsonl").open("x") as telemetry:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (out/"train.pid").write_text(str(child.pid)+"\n")
            while True:
                values = resources()
                for key in high:
                    high[key] = max(high[key], values.get(key, 0))
                telemetry.write(json.dumps(values)+"\n"); telemetry.flush()
                status = dict(phase="training", trainer_pid=child.pid, supervisor_pid=os.getpid(),
                    resources=values, high_watermarks=high, last_completed_fit=latest_metrics(out/"metrics.csv"))
                if values.get("ram_current_bytes", 0) >= args.ram_limit_bytes or values.get("vram_used_bytes", 0) >= args.vram_limit_bytes:
                    raise RuntimeError(f"resource limit exceeded: {values}")
                code = child.poll()
                if code is not None:
                    status.update(phase="complete" if code == 0 else "failed", exit_code=code)
                    publish(out/"status.json", status)
                    if code:
                        raise RuntimeError(f"trainer exited {code}; see {out/'train.log'}")
                    training_complete = True
                    break
                publish(out/"status.json", status)
                time.sleep(args.poll_seconds)
        if args.evaluate_on_completion:
            publish(out/"status.json", dict(phase="evaluating", training_complete=True,
                high_watermarks=high, completed_iterations=config.iterations))
            for selection in ("full", "burnin_quadratic"):
                with (out/f"evaluation_{selection}.log").open("x") as log:
                    subprocess.run([sys.executable, str(ROOT/"scripts/diagnose_cfr_equity.py"),
                        "--strategies", str(out/"strategies"), "--through", str(config.iterations),
                        "--selection", selection, "--pairs", str(args.evaluation_pairs), "--out", str(out/"evaluations"/selection)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            from eval_cfr_stage1 import paired_delta
            delta = paired_delta(out/"evaluations/full/pairs.jsonl", out/"evaluations/burnin_quadratic/pairs.jsonl")
            delta["positive_means"] = "burn-in quadratic improves on full-history linear"
            publish(out/"evaluations/paired_delta.json", delta)
            publish(out/"status.json", dict(phase="complete", training_complete=True,
                completed_iterations=config.iterations, evaluation_complete=True,
                high_watermarks=high, burnin_minus_full_mbb_per_hand=delta))
    except BaseException as error:
        if child is not None:
            stop_child(child)
        publish(out/"status.json", dict(phase="evaluation_failed" if training_complete else "failed",
            training_complete=training_complete, error=f"{type(error).__name__}: {error}",
            trainer_pid=child.pid if child else None, high_watermarks=high, time_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()

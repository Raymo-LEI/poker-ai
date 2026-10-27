"""Read-only, dependency-free training dashboard. Run with Python 3.10+."""

from __future__ import annotations

import argparse
import csv
import errno
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "tools" / "training_monitor"
LOGS = ("metrics.csv", "performance.csv", "metrics.jsonl", "eval.jsonl",
        "diagnostics.jsonl", "checkpoint_performance.jsonl", "benchmark.json", "report.json")
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 5000
MAX_HISTORY_REPORTS = 200


def history_reports(directory):
    """Only include the run's periodic full-budget evaluations, not interventions."""
    base = directory / "evaluations/readiness_periodic"
    if not base.is_dir() or not base.resolve().is_relative_to(directory.resolve()):
        return []
    reports = []
    for child in base.iterdir():
        if (child.name.startswith("full_budget_t") and child.name[13:].isdigit()
                and child.is_dir() and child.resolve().is_relative_to(directory.resolve())):
            path = safe_file(directory, f"evaluations/readiness_periodic/{child.name}/report.json")
            if path:
                reports.append(path)
    return sorted(reports, key=lambda p: (p.stat().st_mtime_ns, str(p)))[-MAX_HISTORY_REPORTS:]


def number(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def flatten(row, prefix=""):
    """Accept flat TensorBoard tags and nested metric dictionaries."""
    result = {}
    for key, value in row.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(value, "" if name == "metrics" else name))
        else:
            result[name] = value
    return result


def safe_file(directory, name):
    path = directory / name
    # Do not follow symlinks to files outside the selected run.
    if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
        return path
    return None


def read_rows(path, warnings):
    """Read a bounded tail; ignore an incomplete append at the end of a log."""
    with path.open("rb") as handle:
        header = handle.readline() if path.suffix == ".csv" else b""
        header_end = handle.tell()
        size = path.stat().st_size
        offset = max(header_end, size - MAX_BYTES)
        handle.seek(offset)
        if offset > header_end:
            handle.readline()  # possibly truncated first record
            warnings.append(f"{path.name} 较大，仅展示文件末尾及最近 {MAX_ROWS} 条记录。")
        data = handle.read(MAX_BYTES)
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    lines = data.decode("utf-8-sig", errors="replace").splitlines()[-MAX_ROWS:]
    rows = []
    if path.suffix == ".csv":
        fields = next(csv.reader([header.decode("utf-8-sig", errors="replace")]), [])
        for values in csv.reader(lines):
            if len(values) != len(fields):
                warnings.append(f"{path.name} 有不完整记录，已跳过。")
                continue
            rows.append(dict(zip(fields, values)))
    else:
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(flatten(row))
                else:
                    warnings.append(f"{path.name} 有非对象记录，已跳过。")
            except (ValueError, TypeError, RecursionError):
                warnings.append(f"{path.name} 有损坏记录，已跳过。")
    return rows


def read_object(path, warnings):
    if path is None:
        return {}
    if path.stat().st_size > MAX_BYTES:
        warnings.append(f"{path.name} 超过 8 MiB，未读取。")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError, RecursionError):
        warnings.append(f"{path.name} 暂时无法解析。")
        return {}


def normalize(row):
    row = flatten(row)
    converted = {k: number(v) for k, v in row.items() if number(v) is not None}
    aliases = {
        "p0_loss": "train/loss_p0", "p1_loss": "train/loss_p1",
        "policy_loss": "train/policy_loss", "value_loss": "train/value_loss",
        "entropy": "train/entropy", "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction", "explained_variance": "train/explained_variance",
        "legal_entropy": "hud/rollout_legal_entropy",
    }
    for old, new in aliases.items():
        if old in converted and new not in converted:
            converted[new] = converted[old]
    for player in (0, 1):
        for field in ("probe_loss_ratio", "probe_end_zero_ratio", "probe_r2", "probe_samples",
                      "probe_normalized_mse", "probe_unweighted_mse", "explained_variance"):
            old, new = f"p{player}_{field}", f"train/{field}_p{player}"
            if old in converted and new not in converted:
                converted[new] = converted[old]
    if "eval_random_bb" in converted and "eval/winrate_vs_random_mbb" not in converted:
        converted["eval/winrate_vs_random_mbb"] = 1000 * converted["eval_random_bb"]
    collect = sum(converted.get(f"p{p}_collect_s", 0) for p in (0, 1))
    if collect > 0 and all(f"p{p}_regret_samples" in converted for p in (0, 1)):
        converted["system/regret_samples_per_s"] = sum(converted[f"p{p}_regret_samples"] for p in (0, 1)) / collect
    calls = sum(converted.get(f"p{p}_inference_calls", 0) for p in (0, 1))
    if calls > 0:
        converted["system/average_inference_batch"] = sum(converted.get(f"p{p}_inference_rows", 0) for p in (0, 1)) / calls
    return converted


def readiness_rows(report):
    """Adapt the repository's frozen-policy evaluation report, if present."""
    if report.get("format") != "sd_cfr_readiness_v1":
        return []
    target = number(report.get("hero_iteration"))
    if target is None:
        return []
    row = {"iteration": target}
    for key, result in report.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        prefix = f"duplicate:{int(target)}_vs_"
        if key.startswith(prefix):
            raw_opponent = key[len(prefix):]
            opponent = raw_opponent
            opponent = {"call": "calling_station", "maniac": "aggressive", "1": "iter_1"}.get(opponent, opponent)
            if opponent == str(int(target) - 1):
                opponent = "prev_checkpoint"
            metric = f"eval/winrate_vs_{opponent}_mbb"
            row[metric] = result.get("mbb_per_hand")
            row[metric + "_pairs"] = result.get("pairs")
            ci = result.get("ci95")
            if isinstance(ci, list) and len(ci) == 2:
                row[metric + "_ci_low"], row[metric + "_ci_high"] = ci
            if raw_opponent == str(int(target) - 1) and opponent != "prev_checkpoint":
                for suffix in ("", "_ci_low", "_ci_high", "_pairs"):
                    if metric + suffix in row:
                        row["eval/winrate_vs_prev_checkpoint_mbb" + suffix] = row[metric + suffix]
            if opponent == "random":
                row["eval/positional_sb_mbb"] = result.get("sb_mbb_per_hand")
                row["eval/positional_bb_mbb"] = result.get("bb_mbb_per_hand")
        if key == f"matches:{int(target)}_vs_random":
            # These use standard HUNL, not the legacy Botzone simple referee.
            row["eval/hunl_50hand_match_win_rate_vs_random"] = result.get("win_fraction")
        if key == "health":
            hero = result.get("hero", {})
            opponent = result.get("opponent", {}) or {}
            if result.get("hands") is not None:
                row["hud/eval_hands"] = result.get("hands")
            row["hud/preflop_vpip_sb"] = hero.get("sb_vpip")
            row["hud/policy_entropy_mean"] = hero.get("mean_nonforced_entropy_nats")
            # Rates already present on hero audit; same names diagnostics writes to eval.jsonl.
            if hero.get("sb_open_shove") is not None:
                row["hud/sb_open_all_in"] = hero.get("sb_open_shove")
            if hero.get("bb_fold_to_open_raise") is not None:
                row["hud/bb_fold_to_open_raise"] = hero.get("bb_fold_to_open_raise")
            counts = hero.get("action_counts_by_street", {}) or {}
            for street_id, street in enumerate(("preflop", "flop", "turn", "river")):
                values = [number(counts.get(f"{street_id}:{a}", 0)) or 0 for a in range(5)]
                total = sum(values)
                if total:
                    for action, value in zip(("fold", "call", "half_pot", "pot", "all_in"), values):
                        row[f"hud/action_freq_{street}/{action}"] = value / total
            # Frontend street charts require per-seat keys hud/action_freq_p{0,1}_{street}/...
            # Prefer seat_street counts when present (unified readiness); else leave unset (fail-closed).
            seat_counts = hero.get("action_counts_by_seat_street") or {}
            if seat_counts:
                audits = [hero]
                if isinstance(opponent, dict) and (opponent.get("action_counts_by_seat_street") or opponent.get("counts")):
                    audits.append(opponent)
                for street_id, street in enumerate(("preflop", "flop", "turn", "river")):
                    combined = [0] * 5
                    for player in (0, 1):
                        values = [
                            sum((a.get("action_counts_by_seat_street") or {}).get(f"{player}:{street_id}:{action}", 0) or 0
                                for a in audits)
                            for action in range(5)
                        ]
                        total = sum(values)
                        if total:
                            row[f"hud/p{player}_{street}_decisions"] = total
                            for name, value in zip(("fold", "call", "half_pot", "pot", "all_in"), values):
                                row[f"hud/action_freq_p{player}_{street}/{name}"] = value / total
                        combined = [x + y for x, y in zip(combined, values)]
                    if sum(combined) and not any(f"hud/action_freq_{street}/{n}" in row for n in ("fold", "call", "half_pot", "pot", "all_in")):
                        for name, value in zip(("fold", "call", "half_pot", "pot", "all_in"), combined):
                            row[f"hud/action_freq_{street}/{name}"] = value / sum(combined)
    return [row]


class Store:
    def __init__(self, roots, source_label="本机日志"):
        self.roots = [Path(root).resolve() for root in roots]
        self.source_label = source_label
        self.cache = {}
        self.lock = threading.Lock()

    def discover(self):
        result = {}
        for index, root in enumerate(self.roots):
            if not root.is_dir():
                continue
            candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
            for directory in candidates:
                if not directory.resolve().is_relative_to(root):
                    continue
                paths = [safe_file(directory, name) for name in LOGS]
                paths = [path for path in paths if path is not None]
                if not paths:
                    continue
                ident = f"{index}/{directory.relative_to(root).as_posix()}"
                result[ident] = {"id": ident, "name": directory.name,
                                 "path": str(directory),
                                 "updated_at": max(p.stat().st_mtime for p in paths)}
        return result

    def load(self, ident):
        with self.lock:
            runs = self.discover()
            if ident not in runs:
                raise KeyError(ident)
            info = runs[ident]
            directory = Path(info["path"])
            names = LOGS + ("config.json", "ops/run_config.json", "ops/environment.json", "train.pt", "strategies/index.json")
            paths = {name: safe_file(directory, name) for name in names}
            history = history_reports(directory)
            paths.update({p.relative_to(directory).as_posix(): p for p in history})
            signature = tuple((name, p.stat().st_mtime_ns, p.stat().st_size)
                              for name, p in paths.items() if p is not None)
            cached = self.cache.get(ident)
            if cached and cached[0] == signature:
                return cached[1]
            warnings = []
            benchmark = read_object(paths["benchmark.json"], warnings)
            report = read_object(paths["report.json"], warnings)
            config = (benchmark.get("config") or read_object(paths["config.json"], warnings)
                      or read_object(paths["ops/run_config.json"], warnings) or report.get("settings", {}))
            environment = (benchmark.get("environment") or read_object(paths["ops/environment.json"], warnings)
                           or report.get("runtime", {}))
            if report.get("format") == "sd_cfr_readiness_v1":
                settings = report.get("settings", {})
                warnings.append(f"T={report.get('hero_iteration', '未记录')} 报告的评测预算：每个对手 {settings.get('pairs', '未记录')} 对发牌，"
                                f"{settings.get('matches', '未记录')} 场 50 手比赛，"
                                f"{settings.get('health_hands', '未记录')} 手 health 检查。预算不代表所有任务均已完成。")
            merged = {}
            step_key = "iteration"

            def merge(row, *, checkpoint_probe=False):
                nonlocal step_key
                row = flatten(row)
                key = next((k for k in ("iteration", "update", "step") if number(row.get(k)) is not None), None)
                if key is None:
                    warnings.append("有记录缺少 iteration / update / step，已跳过。")
                    return
                if key == "update":
                    step_key = "update"
                step = number(row[key])
                target = merged.setdefault(step, {"step": step})
                # Keep nulls, so NaN at the current step cannot silently show an old zero.
                for name, value in row.items():
                    if isinstance(value, (int, float)) or value is None or (isinstance(value, str) and value.strip()):
                        # Keep post-hoc checkpoint scores separate from the
                        # fixed batch used before/after this iteration's fit.
                        if checkpoint_probe and any(name.startswith(f"p{p}_probe_") for p in (0, 1)):
                            seat, suffix = name.split("_", 1)
                            target[f"{seat}_checkpoint_{suffix}"] = number(value)
                            if name in target:
                                continue
                        target[name] = number(value)

            for row in benchmark.get("iterations", [])[-MAX_ROWS:]:
                if isinstance(row, dict):
                    merge(row)
            for name in LOGS:
                if name in ("benchmark.json", "report.json") or paths[name] is None:
                    continue
                for row in read_rows(paths[name], warnings):
                    merge(row, checkpoint_probe=name == "diagnostics.jsonl")
            # The root report is just the latest published copy. Preserve older
            # periodic reports and make their mtimes part of cache invalidation.
            history_bytes = 0
            for path in reversed(history):
                history_bytes += path.stat().st_size
                if history_bytes > 4 * MAX_BYTES:
                    warnings.append("历史评测报告超过 32 MiB，仅读取最近的报告。")
                    break
                for row in readiness_rows(read_object(path, warnings)):
                    merge(row)
            for row in readiness_rows(report):
                merge(row)
            rows = []
            for step in sorted(merged)[-MAX_ROWS:]:
                raw = merged[step]
                rows.append({**{k: v for k, v in raw.items() if v is None}, **normalize(raw)})
            latest = rows[-1] if rows else {}
            eval_rows = [r for r in rows if any(k.startswith("eval/") and v is not None for k, v in r.items())]
            hud_rows = [r for r in rows if any(k.startswith("hud/") and k != "hud/rollout_legal_entropy"
                                             and v is not None for k, v in r.items())]
            eval_latest = eval_rows[-1] if eval_rows else {}
            hud_latest = hud_rows[-1] if hud_rows else {}
            diagnostic_rows = [r for r in rows if any(k.startswith("train/probe_r2_") for k in r)]
            all_in = hud_latest.get("hud/action_freq_preflop/all_in")
            if all_in is not None and all_in > .1:
                warnings.append(f"策略告警：翻前 all-in 比例为 {all_in:.1%}，超过 10%。")
            for key, value in latest.items():
                if "loss" in key and value is None:
                    warnings.append(f"{key} 为非有限值或缺失，请检查训练日志。")
            strategies = read_object(paths["strategies/index.json"], warnings)
            checkpoint = paths["train.pt"]
            data = {**info, "updated_at": max([info["updated_at"]] + [p.stat().st_mtime for p in history]),
                    "kind": "PPO" if step_key == "update" else "SD-CFR", "source": self.source_label,
                    "evaluation_only": bool(paths["report.json"] and not paths["metrics.csv"]
                                            and not paths["metrics.jsonl"] and not paths["benchmark.json"]),
                    "step_key": step_key, "rows": rows, "latest": latest,
                    "eval_latest": eval_latest, "hud_latest": hud_latest,
                    "diagnostic_latest": diagnostic_rows[-1] if diagnostic_rows else {},
                    "config": config, "environment": environment,
                    "sources": [{"name": name, "bytes": p.stat().st_size,
                                 "updated_at": p.stat().st_mtime} for name, p in paths.items() if p is not None],
                    "checkpoint": {"bytes": checkpoint.stat().st_size,
                                   "updated_at": checkpoint.stat().st_mtime} if checkpoint else None,
                    "strategies": strategies, "warnings": list(dict.fromkeys(warnings))[:30],
                    "evaluation_notes": report.get("notes", []),
                    "history_limit": MAX_ROWS}
            # Bound cache size when directories are rotated/deleted.
            self.cache = {k: v for k, v in self.cache.items() if k in runs}
            self.cache[ident] = (signature, data)
            return data


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type, *, etag=None):
            if etag and self.headers.get("If-None-Match") == etag:
                status, body = 304, b""
            accepts_gzip = False
            for item in self.headers.get("Accept-Encoding", "").split(","):
                parts = item.strip().lower().split(";")
                if parts[0] != "gzip":
                    continue
                try:
                    accepts_gzip = next((float(p.strip()[2:]) for p in parts[1:]
                                         if p.strip().startswith("q=")), 1) > 0
                except ValueError:
                    pass
            compressed = status != 304 and len(body) > 1024 and accepts_gzip
            if compressed:
                body = gzip.compress(body, compresslevel=1, mtime=0)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache" if etag else "no-store")
            self.send_header("Vary", "Accept-Encoding")
            if etag:
                self.send_header("ETag", etag)
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, value, *, conditional=False):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            etag = 'W/"' + hashlib.sha256(body).hexdigest() + '"' if conditional else None
            self.send(status, body, "application/json; charset=utf-8", etag=etag)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/runs":
                    runs = sorted(store.discover().values(), key=lambda r: r["updated_at"], reverse=True)
                    self.json(200, {"runs": runs, "server_time": time.time(), "source": store.source_label})
                elif url.path == "/api/run":
                    ident = parse_qs(url.query).get("id", [""])[0]
                    self.json(200, store.load(ident), conditional=True)
                elif url.path in ("/", "/index.html", "/app.js", "/styles.css"):
                    name = "index.html" if url.path == "/" else url.path[1:]
                    mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[1]]
                    self.send(200, (ASSETS / name).read_bytes(), mime + "; charset=utf-8")
                else:
                    self.json(404, {"error": "未找到页面"})
            except (BrokenPipeError, ConnectionResetError):
                return  # A cancelled browser request must not trigger another write.
            except KeyError:
                self.json(404, {"error": "实验不存在，请刷新实验列表"})
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                self.json(503, {"error": getattr(store, "unavailable_message", "日志正在变化或暂时无法读取，请稍后刷新"), "detail": type(exc).__name__})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", action="append", type=Path, help="run directory or its parent; repeat for multiple roots")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--open-browser", action="store_true", help="open the local dashboard after binding")
    parser.add_argument("--source-label", default="本机日志", help="visible label for the monitored host")
    args = parser.parse_args()
    store = Store(args.runs_dir or [REPO / "runs"], args.source_label)
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(1, f"Port {args.port} is already in use; stop the previous monitor or choose --port.\n")
        raise
    print(f"Overwatch: http://{args.host}:{server.server_port}", flush=True)
    print("Read-only logs: " + ", ".join(map(str, store.roots)), flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

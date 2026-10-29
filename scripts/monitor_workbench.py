"""Bounded, read-only adapters for Overwatch experiments and recorded hands.

No ML imports, subprocesses, checkpoint deserialization, or filesystem writes.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
from itertools import islice
from pathlib import Path
import re
import threading

MAX_JSON = 8 * 1024 * 1024
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_TRACE_ROWS = 25000
MAX_LINE = 1024 * 1024
SIDECARS = ("launch_manifest.json", "status.json", "study_status.json", "manifest.json",
            "evaluation_report.json", "arm1200.json", "arm2400.json", "ops/branch.json",
            "config.json", "ops/run_config.json", "ops/environment.json", "train.pt.refs.json",
            "evaluations/paired_delta.json", "resources.jsonl")
ALIASES = {"call": "calling_station", "maniac": "aggressive"}
PHASES = {"training": "训练", "evaluating": "评测", "complete": "完成", "failed": "失败",
          "evaluation_failed": "评测失败", "initializing": "初始化", "saving": "保存",
          "bootstrap": "初始化", "unknown": "未知"}


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_path(root, relative):
    path = root / relative
    if path.is_file() and path.resolve().is_relative_to(root.resolve()):
        return path
    return None


def object_file(path):
    if path is None or path.stat().st_size > MAX_JSON:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return clean(value) if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def timestamp(value):
    if numeric(value) is not None:
        return numeric(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def tail_json(path, max_bytes=512 * 1024, max_rows=1800):
    if path is None:
        return []
    with path.open("rb") as handle:
        size = path.stat().st_size
        offset = max(0, size - max_bytes)
        handle.seek(offset)
        if offset:
            handle.readline()
        body = handle.read(max_bytes)
    if body and not body.endswith(b"\n"):
        body = body[:body.rfind(b"\n") + 1]
    rows = []
    for line in body.splitlines()[-max_rows:]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(clean(value))
        except (ValueError, RecursionError):
            continue
    return rows


def bounded_children(root, relative):
    base = root / relative
    if not base.is_dir() or not base.resolve().is_relative_to(root.resolve()):
        return []
    return sorted((p for p in base.iterdir() if not p.name.startswith(".") and p.is_dir()
                  and p.resolve().is_relative_to(root.resolve())), key=lambda p: (p.stat().st_mtime_ns, p.name))[-200:]


def report_paths(root):
    paths = {"report.json": safe_path(root, "report.json")}
    # Fixed, shallow layouts only; never descend into checkpoint generations.
    for base in ("evaluations", "monitor_diagnostics", "monitor_hud"):
        for directory in bounded_children(root, base):
            candidates = [directory]
            if base == "evaluations" and directory.name == "readiness_periodic":
                candidates = bounded_children(root, "evaluations/readiness_periodic")
            for candidate in candidates:
                relative = (candidate / "report.json").relative_to(root).as_posix()
                path = safe_path(root, relative)
                if path:
                    paths[relative] = path
    base = root / 'stage2_evaluations'
    if base.is_dir() and base.resolve().is_relative_to(root.resolve()):
        for candidate in base.glob('update_*.json'):
            name = candidate.relative_to(root).as_posix()
            if (path := safe_path(root, name)) is not None:
                paths[name] = path
    return dict(sorted(((k,p) for k,p in paths.items() if p is not None),
                       key=lambda item: (item[1].stat().st_mtime_ns, item[0]))[-200:])


def extra_paths(root):
    reports = report_paths(root)
    traces = {}
    for path in reports.values():
        name = (path.parent / "hands.jsonl").relative_to(root).as_posix()
        candidate = safe_path(root, name)
        if candidate:
            traces[name] = candidate
    archives = {}
    for path in archive_paths(root):
        archives[path.relative_to(root).as_posix()] = path
        name = path.relative_to(root).as_posix() + ".json"
        if (metadata := safe_path(root, name)) is not None:
            archives[name] = metadata
    return {**{name: p for name in SIDECARS if (p := safe_path(root, name)) is not None},
            **reports, **traces, **archives}


def archive_paths(root):
    result = []
    for directory in [root] + [root / name for name in ("checkpoints", "archives", "checkpoint_archives")
                              if (root / name).is_dir() and (root / name).resolve().is_relative_to(root.resolve())]:
        for path in directory.glob("*.tar.gz"):
            if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                result.append(path)
                if len(result) >= 100:
                    return result
    return result


def category(name, config, has_metrics, has_report):
    lower = name.lower()
    if "smoke" in lower:
        return "smoke"
    if "benchmark" in lower:
        return "benchmark"
    if has_report and not has_metrics:
        return "evaluation"
    if "branch" in lower or "arm1200" in lower or "arm2400" in lower:
        return "branch"
    return "training" if has_metrics or config else "other"


def effective_config(root):
    manifest = object_file(safe_path(root, "launch_manifest.json"))
    declared = object_file(safe_path(root, "config.json")) or object_file(safe_path(root, "ops/run_config.json"))
    effective = manifest.get("effective_config") or {}
    # Old launchers may wrap the effective configuration in a config key.
    config = {**declared, **(effective.get("config") or {}), **{k: v for k, v in effective.items() if k != "config"}}
    return config, manifest


def config_summary(config, manifest):
    shared = config.get("separate_player_models") is False
    capacity = numeric(config.get("buffer_capacity"))
    effective_capacity = numeric(config.get("effective_buffer_capacity"))
    inferred = effective_capacity is None and capacity is not None and isinstance(config.get("separate_player_models"), bool)
    if inferred:
        effective_capacity = capacity * 2 if shared else capacity
    model = config.get("model") or {}
    fields = {key: config.get(key) for key in (
        "stage", "updates", "matches_per_update", "reference", "strategies", "policy", "ppo", "pool", "evaluation", "traces",
        "seed", "device", "effective_device", "iterations", "train_steps", "minibatch_size", "loss",
        "regret_target_scale", "learning_rate", "learning_rate_schedule", "iteration_weight_power",
        "normalize_iteration_weights", "equity_features", "position_conditioning", "separate_player_models",
        "scripted_opponent_prob", "scripted_opponent_pool", "preflop_action_guard", "spr_action_guard",
        "numeric_version", "rules_version", "amp_dtype", "effective_amp_dtype", "regret_trainable_parameters")}
    return {**fields, "model": model, "declared_capacity": capacity, "effective_capacity": effective_capacity,
            "capacity_basis": "配置语义换算" if inferred else "有效配置记录" if effective_capacity is not None else "未记录",
            "source": "launch_manifest.json" if manifest.get("effective_config") else "config.json / ops/run_config.json",
            "source_file_count": len(manifest.get("source_sha256") or {}),
            "objective": manifest.get("objective") or {}, "retention": manifest.get("retention") or {}}


def preview(root, info):
    config, manifest = effective_config(root)
    status_path = safe_path(root, "status.json") or safe_path(root, "study_status.json")
    status = object_file(status_path)
    latest = {}
    metrics = safe_path(root, "metrics.csv")
    if metrics:
        with metrics.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig").strip()
            offset = max(handle.tell(), metrics.stat().st_size - 16384)
            handle.seek(offset)
            if offset > len(header.encode()) + 1:
                handle.readline()
            body = handle.read(16384)
        if not body.endswith(b"\n"):
            body = body[:body.rfind(b"\n") + 1]
        fields = next(csv.reader([header]), [])
        for values in csv.reader(body.decode("utf-8", errors="replace").splitlines()):
            if len(values) == len(fields):
                latest = dict(zip(fields, values))
    json_metrics = safe_path(root, "metrics.jsonl")
    if json_metrics:
        latest = (tail_json(json_metrics, max_bytes=64 * 1024, max_rows=500) or [latest])[-1]
    report = object_file(safe_path(root, "report.json"))
    eval_steps = []
    if any(isinstance(v, dict) and (numeric(v.get("mbb_per_hand")) is not None or numeric(v.get("win_fraction")) is not None)
           for v in (report.get("results") or {}).values()):
        if numeric(report.get("hero_iteration")) is not None:
            eval_steps.append(numeric(report["hero_iteration"]))
    if numeric((report.get("summary") or {}).get("mbb_per_hand")) is not None:
        eval_steps.extend(numeric(v) for v in report.get("selected_iterations", []) if numeric(v) is not None)
    for row in tail_json(safe_path(root, "eval.jsonl"), max_bytes=64 * 1024, max_rows=500):
        values = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
        if numeric(values.get("stage2/eval/update")) is not None:
            eval_steps.append(numeric(values["stage2/eval/update"]))
        if any(str(k).startswith("eval/") and numeric(v) is not None for k, v in values.items()) or isinstance(values.get("eval"), dict):
            step = next((numeric(row[k]) for k in ("iteration", "update", "step") if numeric(row.get(k)) is not None), None)
            if step is not None:
                eval_steps.append(step)
    return {**info, "category": category(info["name"], config, bool(metrics or json_metrics), bool(report)),
            "stage": config.get("stage", "stage2" if latest.get("stage2") == 1 else "stage1"),
            "last_step": numeric(latest.get("iteration", latest.get("update"))),
            "eval_step": max(eval_steps) if eval_steps else None,
            "reported_phase": status.get("phase", "unknown"),
            "phase_recorded_at": status_path.stat().st_mtime if status_path else None,
            "effective": config_summary(config, manifest)}


def stage2_summary(root, data, config):
    enabled = config.get("stage") == "stage2" or data.get("stage") == "stage2"
    if not enabled:
        return {"enabled": False}
    # Keep textual opponent identity separately from the numeric chart adapter.
    records = tail_json(safe_path(root, "metrics.jsonl"), max_bytes=2 * 1024 * 1024, max_rows=5000)
    records = [r for r in records if numeric(r.get("update")) is not None and r.get("stage2") == 1]
    block = numeric(config.get('matches_per_update')) is not None or any(any(k.startswith('stage2/sampled/') for k in r) for r in records)
    returns = [(r, numeric(r.get('stage2/hero_net_bb'))) for r in records]
    returns = [(r,v) for r,v in returns if v is not None]
    latest = records[-1] if records else {}
    groups = defaultdict(list)
    for row,value in returns:
        if not block:
            groups[str(row.get('stage2/opponent','unknown'))].append(value)
    recent = returns[-100:]
    weighted = [(value, sum(numeric(v) or 0 for k,v in row.items() if k.startswith('stage2/sampled/')) if block else 1) for row,value in recent]
    count = sum(n for _,n in weighted)
    opponents = []
    names = sorted({key.split('/')[2] for key in latest if key.startswith('stage2/window/')})
    for name in names:
        prefix = f'stage2/window/{name}/'
        opponents.append({'name':name, 'matches':numeric(latest.get(prefix+'matches')),
                          'mbb_per_hand':numeric(latest.get(prefix+'mbb_per_hand')),
                          'match_win':numeric(latest.get(prefix+'match_win')),
                          'sampled':numeric(latest.get('stage2/sampled/'+name))})
    # Per-match records are authoritative for W/D/L. Older block logs still
    # expose exact wins via match_win * sampled_count; a block mean is never a win.
    ledger = tail_json(safe_path(root, 'matches.jsonl'), max_bytes=2 * 1024 * 1024, max_rows=5000)
    recent_updates = {r['update'] for r, _ in recent}
    ledger = [r for r in ledger if r.get('format') == 'stage2_training_match_v1'
              and r.get('update') in recent_updates and numeric(r.get('hero_net_bb')) is not None
              and isinstance(r.get('opponent'), str)]
    wins = sum(n for value, n in weighted if value > 0) if not block else None
    draws = sum(n for value, n in weighted if value == 0) if not block else None
    if block and recent and all(numeric(r.get('stage2/match_win')) is not None for r, _ in recent):
        wins = sum(round(r['stage2/match_win'] * n) for (r, _), (_, n) in zip(recent, weighted))
    if block and recent and all(numeric(r.get('stage2/match_draws')) is not None for r, _ in recent):
        draws = sum(r['stage2/match_draws'] for r, _ in recent)
    # Reject a partial ledger rather than presenting incomplete totals as exact.
    if block and len(ledger) == count and len({r.get('match') for r in ledger}) == count:
        wins = sum(r['hero_net_bb'] > 0 for r in ledger)
        draws = sum(r['hero_net_bb'] == 0 for r in ledger)
    sampled = {k.split('/')[-1]: v for k, v in latest.items()
               if k.startswith('stage2/sampled/') and numeric(v) is not None and v > 0}
    return {'enabled':True, 'block_updates':block,
            'latest_opponent':'mixed' if block else latest.get('stage2/opponent'),
            'latest_sampled':sampled,
            'window_updates':len(recent), 'window_matches':count,
            'window_wins':wins, 'window_draws':draws,
            'window_losses':count-wins-draws if wins is not None and draws is not None else None,
            'window_net_bb_mean':sum(value*n for value,n in weighted)/count if count else None,
            'opponents':opponents if block else [{'name':name,'matches':len(values),
                'mean_net_bb':sum(values)/len(values), 'wins':sum(v>0 for v in values),
                'draws':sum(v==0 for v in values)} for name,values in sorted(groups.items())],
            'scope':'metrics.jsonl 有界尾部的训练统计；对手与策略持续变化，不是独立评测。',
            'recorded_updates':len(returns),
            'context_scope':'最后一场结束时的对手 HUD，含 Beta 先验；未记录原始机会数。'}


def stage2_report_row(report):
    if report.get('format') != 'stage2_duplicate_matches_v1' or numeric(report.get('update')) is None:
        return {}
    row = {'step':numeric(report['update']), 'update':numeric(report['update']), 'stage2':1, 'stage2/eval/update':numeric(report['update'])}
    for flag in ('retention_pass','equity_positive'):
        if isinstance(report.get(flag),bool):
            row['stage2/eval/'+flag] = int(report[flag])
    for name,item in (report.get('opponents') or {}).items():
        result = item.get('candidate') or {}
        for field,value in result.items():
            if numeric(value) is not None:
                row[f'stage2/eval/{name}/{field}'] = numeric(value)
        row[f'stage2/eval/{name}/gain_mbb_per_hand'] = numeric(item.get('gain_mbb_per_hand'))
        key = 'eval/winrate_vs_'+ALIASES.get(name,name)+'_mbb'
        row[key] = numeric(result.get('mbb_per_hand'))
        ci = result.get('mbb_ci95')
        if isinstance(ci,list) and len(ci)==2:
            row[key+'_ci_low'],row[key+'_ci_high'] = map(numeric,ci)
        row[key+'_pairs'] = numeric(result.get('match_pairs'))
    return row


def result_entry(name, result, config):
    opponent = ALIASES.get(str(name), str(name))
    ci = result.get("ci95")
    return {"opponent": opponent, "mbb_per_hand": numeric(result.get("mbb_per_hand", result.get("mean"))),
            "ci95": ci if isinstance(ci, list) and len(ci) == 2 else None,
            "pairs": numeric(result.get("pairs")),
            "training_opponent": bool(numeric(config.get("scripted_opponent_prob")) and
                                      name in (config.get("scripted_opponent_pool") or {}))}


def adapt_report(report, source, readiness_adapter, config):
    fmt = report.get("format")
    rows = readiness_adapter(report)
    selected = report.get("selected_iterations") or []
    step = numeric(report.get("hero_iteration")) or (max(selected) if selected else None)
    selection = report.get("selection") or (report.get("settings") or {}).get("selection") or "full"
    results = []
    row = dict(rows[0]) if rows else {"iteration": step}
    if fmt == 'stage2_duplicate_matches_v1':
        row = stage2_report_row(report)
        step = row.get('step')
        selection = 'stage2_candidate'
        for name,item in (report.get('opponents') or {}).items():
            candidate = item.get('candidate') or {}
            results.append(result_entry(name, {'mbb_per_hand':candidate.get('mbb_per_hand'),
                'ci95':candidate.get('mbb_ci95'), 'pairs':candidate.get('match_pairs')}, config))
    elif fmt == "sd_cfr_readiness_v1" and step is not None:
        for key, value in (report.get("results") or {}).items():
            prefix = f"duplicate:{int(step)}_vs_"
            if key.startswith(prefix) and isinstance(value, dict):
                opponent = key[len(prefix):]
                opponent = "iter_1" if opponent == "1" else "prev_checkpoint" if opponent == str(int(step)-1) else opponent
                results.append(result_entry(opponent, value, config))
        row["diagnostic/eval_average_policy"] = 1
    elif fmt == "cfr_equity_loss_trace_v1":
        results.append(result_entry("equity", report.get("summary") or {}, config))
        for result in results:
            metric = "eval/winrate_vs_equity_mbb"
            row[metric] = result["mbb_per_hand"]
            row[metric + "_pairs"] = result["pairs"]
            if result["ci95"]:
                row[metric + "_ci_low"], row[metric + "_ci_high"] = result["ci95"]
        audit = report.get("hero_audit") or {}
        for key, field in (("sb_vpip", "preflop_vpip_sb"), ("sb_open_shove", "sb_open_all_in"),
                           ("bb_fold_to_open_raise", "bb_fold_to_open_raise"),
                           ("mean_nonforced_entropy_nats", "policy_entropy_mean")):
            if audit.get(key) is not None:
                row["hud/" + field] = audit[key]
    else:
        return None
    if step is None:
        return None
    row["step"] = step
    identity = {"format": fmt, "step": step, "selection": selection,
                "settings": report.get("settings"), "seed": report.get("seed"),
                "snapshots": report.get("snapshots", report.get("snapshot_sha256")),
                "source_sha256": report.get("source_sha256"), "selected_iterations": selected}
    ident = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    labels = {"stage2_candidate":"Stage 2 独立配对比赛", "full": "全历史线性平均", "latest": "单个策略快照", "burnin_quadratic": "后半程二次混合"}
    settings = report.get("settings") or {}
    override = settings.get("spr_action_guard")
    return {"id": ident, "source": source, "format": fmt, "step": step,
            "selection": selection, "label": labels.get(selection, selection),
            "seed": report.get("seed", settings.get("seed")), "pairs": settings.get("pairs"),
            "guard_override": override, "intervention": override is not None,
            "selected_range": [min(selected), max(selected)] if selected else [1, step],
            "sampling_power": report.get("sampling_power", 1),
            "results": results, "metrics": clean(row), "breakdown": report.get("opponents") if fmt == "stage2_duplicate_matches_v1" else report.get("breakdown") or {}}


def assets(root, index, checkpoint):
    advertised = index.get("iterations") or []
    advertised = sorted({int(numeric(v)) for v in advertised if numeric(v) is not None
                         and numeric(v).is_integer() and 0 < numeric(v) < 1000000})[:10000]
    strategy_dir = root / "strategies"
    found = set()
    if strategy_dir.is_dir() and strategy_dir.resolve().is_relative_to(root.resolve()):
        for p in strategy_dir.iterdir():
            match = re.fullmatch(r"iteration_(\d+)\.pt", p.name)
            if match and p.is_file() and p.resolve().is_relative_to(root.resolve()):
                found.add(int(match[1]))
    missing_files = [v for v in advertised if v not in found]
    gaps = list(islice((v for v in range(1, max(advertised, default=0)+1) if v not in found), 20))
    refs = object_file(safe_path(root, "train.pt.refs.json"))
    refs_valid = False
    generations = []
    cp = safe_path(root, "train.pt")
    if cp and isinstance(refs.get("fingerprint"), list):
        stat = cp.stat()
        refs_valid = refs["fingerprint"] == [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    for name in (refs.get("generations") or [])[:30]:
        if not isinstance(name, str) or not re.fullmatch(r"iteration_\d{7,}_[0-9a-f]{32}", name):
            continue
        p = root / "checkpoint_data" / name
        generations.append({"name": name, "directory_present": p.is_dir() and p.resolve().is_relative_to(root.resolve())})
    archives = []
    for p in archive_paths(root):
        # Only published manifests and object metadata; never read large weights.
        metadata = object_file(safe_path(root, p.relative_to(root).as_posix() + ".json"))
        objects = list((metadata.get("strategy_objects") or {}).values())[:2000]
        present = 0
        for entry in objects:
            if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-f]{64}\.pt", str(entry.get("object", ""))):
                continue
            candidate = safe_path(root, (p.parent / "objects" / entry["object"]).relative_to(root).as_posix())
            if candidate and candidate.stat().st_size == numeric(entry.get("bytes")):
                present += 1
        archives.append({"name": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
            "updated_at": p.stat().st_mtime, "manifest_present": bool(metadata),
            "objects_checked": len(objects), "objects_present": present,
            "restore_requires": metadata.get("restore_requires"), "verification": "仅清单、对象存在性与大小；未哈希或恢复"})
    return {"checkpoint": checkpoint, "verification": "仅文件与引用目录检查；未加载或恢复 checkpoint",
            "references_match_file": refs_valid, "generations": generations,
            "advertised_snapshots": len(advertised), "present_snapshots": len(found),
            "through": max(advertised, default=None), "missing_files": missing_files[:20], "history_gaps": gaps,
            "history_complete": bool(advertised) and not missing_files and not gaps,
            "archives": archives[:100]}


class Workbench:
    def __init__(self):
        self.trace_cache = OrderedDict()
        self.trace_lock = threading.Lock()

    def detail(self, root, data, readiness_adapter):
        config, manifest = effective_config(root)
        if not config:
            config = data.get("config") or {}
        status_path = safe_path(root, "status.json") or safe_path(root, "study_status.json")
        status = object_file(status_path)
        resources = tail_json(safe_path(root, "resources.jsonl"))
        for item in resources:
            item["step"] = timestamp(item.get("time_utc", item.get("time")))
        resources = [r for r in resources if r["step"] is not None]
        evaluations = {}
        trace_sources = []
        total_bytes = 0
        for source, path in report_paths(root).items():
            total_bytes += path.stat().st_size
            if total_bytes > 32 * 1024 * 1024:
                break
            report = object_file(path)
            evaluation = adapt_report(report, source, readiness_adapter, config)
            if evaluation:
                evaluations.setdefault(evaluation["id"], evaluation)
            trace = safe_path(root, (path.parent / "hands.jsonl").relative_to(root).as_posix())
            if trace:
                name = trace.relative_to(root).as_posix()
                trace_sources.append({"id": hashlib.sha256(name.encode()).hexdigest()[:16], "source": name,
                    "bytes": trace.stat().st_size, "updated_at": trace.stat().st_mtime,
                    "evaluation_id": evaluation["id"] if evaluation else None,
                    "scope": report.get("scope"), "status": report.get("status"),
                    "recorded_hands": report.get("recorded_hands"), "truncated": report.get("truncated"),
                    "dropped_hands": report.get("dropped_hands")})
        represented_steps = {e["step"] for e in evaluations.values() if e["selection"] in ("full", "stage2_candidate")}
        for row in data["rows"]:
            if row["step"] in represented_steps:
                continue
            results = []
            for key, value in row.items():
                if key.startswith("eval/winrate_vs_") and key.endswith("_mbb") and numeric(value) is not None:
                    name = key[len("eval/winrate_vs_"):-len("_mbb")]
                    ci = [row.get(key + "_ci_low"), row.get(key + "_ci_high")]
                    results.append(result_entry(name, {"mbb_per_hand": value,
                        "pairs": row.get(key + "_pairs", row.get("eval/pairs")),
                        "ci95": ci if all(numeric(v) is not None for v in ci) else None}, config))
            if results:
                ident = "logged-" + str(row["step"])
                evaluations[ident] = {"id": ident, "step": row["step"], "selection": "logged",
                    "label": "日志评测（策略类型以原报告为准）", "source": "eval.jsonl / metrics",
                    "seed": None, "results": results, "metrics": row, "intervention": False, "breakdown": {}}
        phase = status.get("phase", "unknown")
        latest_resource = resources[-1] if resources else status.get("resources") or {}
        source_iteration = numeric((status.get("last_completed_fit") or {}).get("iteration"))
        events = []
        def event(kind, label, when, detail):
            if when is not None:
                events.append({"kind": kind, "label": label, "time": when, "detail": detail})
        event("launch", "运行启动记录", timestamp(manifest.get("started_utc")), "launch_manifest.json")
        event("status", "阶段报告：" + PHASES.get(phase, str(phase)), status_path.stat().st_mtime if status_path else None, "状态文件记录，非实时进程探测")
        checkpoint = data.get("checkpoint")
        event("checkpoint", "最近恢复点文件", checkpoint.get("updated_at") if checkpoint else None, "仅发布文件元数据，不代表实际恢复验证通过")
        for item in evaluations.values():
            p = safe_path(root, item["source"])
            event("evaluation", f"T={item['step']:g} {item['label']}", p.stat().st_mtime if p else None, item["source"])
        deltas = object_file(safe_path(root, "evaluations/paired_delta.json"))
        study_report = object_file(safe_path(root, "evaluation_report.json"))
        branch = object_file(safe_path(root, "ops/branch.json"))
        training_rows = [r for r in data["rows"] if any(k in r for k in ("shared_loss", "train/loss_p0", "train/policy_loss"))]
        result = {"version": 2, "category": category(root.name, config, bool(safe_path(root, "metrics.csv") or safe_path(root, "metrics.jsonl")), data.get("evaluation_only", False)),
            "effective": config_summary(config, manifest),
            "stage2": stage2_summary(root, data, config),
            "runtime": {"reported_phase": phase, "phase_label": PHASES.get(phase, str(phase)),
                "recorded_at": status_path.stat().st_mtime if status_path else None,
                "source_iteration": source_iteration, "process_liveness": "unknown",
                "training_step": training_rows[-1]["step"] if training_rows else None,
                "training_complete": status.get("training_complete"), "evaluation_complete": status.get("evaluation_complete"),
                "exit_code": status.get("exit_code"), "error": status.get("error", status.get("traceback")),
                "resources": latest_resource, "resource_time": timestamp(latest_resource.get("time_utc")),
                "high_watermarks": status.get("high_watermarks") or {}, "resources_history": resources},
            "evaluations": sorted(evaluations.values(), key=lambda e: (e["step"], e["label"])),
            "paired_delta": deltas, "study_paired_deltas": study_report.get("paired_deltas") or {},
            "branch": branch, "trace_sources": trace_sources,
            "assets": assets(root, data.get("strategies") or {}, checkpoint),
            "events": sorted(events, key=lambda e: e["time"], reverse=True)[:40]}
        return clean(result)

    def trace_path(self, root, trace_id):
        for path in report_paths(root).values():
            candidate = safe_path(root, (path.parent / "hands.jsonl").relative_to(root).as_posix())
            if candidate and hashlib.sha256(candidate.relative_to(root).as_posix().encode()).hexdigest()[:16] == trace_id:
                return candidate
        raise KeyError("unknown trace")

    def trace_index(self, root, trace_id):
        path = self.trace_path(root, trace_id)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self.trace_lock:
            if key in self.trace_cache:
                self.trace_cache.move_to_end(key)
                return self.trace_cache[key]
        records = []
        skipped = 0
        with path.open("rb") as handle:
            while handle.tell() < MAX_TRACE_BYTES and len(records) < MAX_TRACE_ROWS:
                offset = handle.tell()
                line = handle.readline(min(MAX_LINE + 1, MAX_TRACE_BYTES - offset))
                if not line:
                    break
                if len(line) > MAX_LINE:
                    # Stop before a pathological record; never spend unbounded work skipping it.
                    skipped += 1
                    break
                if not line.endswith(b"\n"):
                    break
                try:
                    hand = json.loads(line)
                    if not isinstance(hand, dict) or not isinstance(hand.get("events"), list):
                        skipped += 1
                        continue
                    payoff = numeric(hand.get("payoff_chips"))
                    if payoff is None:
                        skipped += 1
                        continue
                    records.append({"record": len(records), "offset": offset, "payoff_chips": payoff,
                        "hero_seat": hand.get("hero_seat"), "category": hand.get("category", "unclassified"),
                        "street": hand.get("terminal_betting_street"), "snapshot": hand.get("snapshot"),
                        "pair": hand.get("pair"), "events": len(hand["events"]),
                        **{k: hand[k] for k in ("scope", "update", "match", "hand", "opponent") if k in hand}})
                except (ValueError, TypeError, RecursionError):
                    skipped += 1
            read_to = handle.tell()
        index = {"path": path, "records": records, "truncated": read_to < stat.st_size,
                 "skipped": skipped, "bytes_scanned": read_to, "file_bytes": stat.st_size}
        with self.trace_lock:
            self.trace_cache[key] = index
            self.trace_cache.move_to_end(key)
            while len(self.trace_cache) > 4:
                self.trace_cache.popitem(last=False)
        return index

    def hands(self, root, trace_id, *, offset=0, limit=30, seat="all", category_filter="all", order="loss"):
        index = self.trace_index(root, trace_id)
        rows = index["records"]
        buckets = defaultdict(lambda: {"hands": 0, "chips": 0.0})
        for row in rows:
            bucket = buckets[row["category"]]
            bucket["hands"] += 1
            bucket["chips"] += row["payoff_chips"]
        for bucket in buckets.values():
            bucket["contribution_mbb_per_all_hands"] = bucket["chips"] * 10 / len(rows)
            bucket["conditional_mbb_per_hand"] = bucket["chips"] * 10 / bucket["hands"]
        filtered = [r for r in rows if (seat == "all" or str(r["hero_seat"]) == seat)
                    and (category_filter == "all" or r["category"] == category_filter)]
        if order in ("loss", "win"):
            filtered = sorted(filtered, key=lambda r: r["payoff_chips"], reverse=order == "win")
        return {"indexed_hands": len(rows), "filtered_hands": len(filtered), "buckets": dict(buckets),
            "items": [{k: v for k, v in r.items() if k != "offset"} for r in filtered[max(0, offset):max(0, offset)+min(max(1, limit), 100)]],
            "offset": max(0, offset), "truncated": index["truncated"], "skipped": index["skipped"],
            "bytes_scanned": index["bytes_scanned"], "big_blind_chips": 100,
            "scope": "标准 HUNL 已记录轨迹；亏损分桶为描述性归因，非逐动作反事实 EV"}

    def hand(self, root, trace_id, record):
        index = self.trace_index(root, trace_id)
        if record < 0 or record >= len(index["records"]):
            raise KeyError("unknown hand")
        with index["path"].open("rb") as handle:
            handle.seek(index["records"][record]["offset"])
            hand = json.loads(handle.readline(MAX_LINE))
        # Return recorded fields only, with a bounded number of events.
        keep = ("hero_seat", "payoff_chips", "snapshot", "category", "terminal_betting_street", "pair",
                "deck_seed", "hole_cards", "hero_hole", "post_hand_oracle_audit",
                "terminal_board", "scope", "update", "match", "opponent", "opponent_snapshot", "first_hero_seat")
        return clean({**{k: hand[k] for k in keep if k in hand}, "record": record,
                      "events": hand["events"][:1000], "big_blind_chips": 100})

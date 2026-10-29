"""Frozen eight-script tournament plus candidate round-robin; no training imports execute training."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from alphanl.cfr.gauntlet import (Candidate, GAUNTLET_OPPONENTS, match_factories,
    paired_candidate_gain, rule_actor, select_candidate, sha256)
from alphanl.cfr.pro_bots import PRO_VERSION
from alphanl.stage2.evaluation import summarize_match_pairs
from alphanl.transformer.hunl import RULES_VERSION


def publish(out, report):
    temporary = out / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "report.json")
    columns = list(GAUNTLET_OPPONENTS) + list(report["matrix"])
    with (out / "matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["candidate", "opponent", "mbb_per_hand", "ci95_low", "ci95_high", "hands", "match_pairs"])
        for name, row in report["matrix"].items():
            for opponent, cell in row.items():
                writer.writerow([name, opponent, cell["mbb_per_hand"], *cell["mbb_ci95"], cell["hands"], cell["match_pairs"]])
    rows = []
    for name, row in report["matrix"].items():
        cells = []
        for opponent in columns:
            cell = row.get(opponent)
            value = (f'{cell["mbb_per_hand"]:+,.1f}<small>[{cell["mbb_ci95"][0]:+,.1f}, '
                     f'{cell["mbb_ci95"][1]:+,.1f}]</small>') if cell else ("—" if name == opponent else "Pending")
            cells.append(f"<td>{value}</td>")
        rows.append(f"<tr><th>{html.escape(name)}</th>{''.join(cells)}</tr>")
    selection = html.escape(json.dumps(report.get("selection", {}), indent=2))
    body = ("<!doctype html><meta charset=utf-8><title>Pro Gauntlet</title>"
            "<style>body{font:14px system-ui;margin:32px;color:#222}table{border-collapse:collapse}"
            "td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}small{display:block;color:#666}"
            "pre{white-space:pre-wrap}h1{font-size:24px}</style>"
            f"<h1>Pro Gauntlet · {'Complete' if report.get('complete') else 'In progress'}</h1>"
            "<p>Payoff: mbb/hand; brackets: approximate 95% CI clustered by duplicate 50-hand match pair. "
            "Rows are candidates; positive means the row wins. Script styles are not certified ladder players.</p>"
            f"<p>Seed: {report['seed']} · Deck pairs per matchup: {report['pairs']} · {html.escape(RULES_VERSION)}</p>"
            "<table><thead><tr><th>Candidate / opponent</th>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"
            + "".join(rows) + f"</tbody></table><h2>Selection evidence</h2><pre>{selection}</pre>"
            "<p>Source: report.json and per-matchup JSONL in this directory. Same deck schedule across all cells. "
            "Slumbot is independent online play and is reported separately.</p>")
    (out / "matrix.html").write_text(body)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--pairs", type=int, default=2500, help="Duplicate deals per cell; multiple of 50, >=100")
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--equity-samples", type=int, default=64)
    p.add_argument("--teacher-cache-size", type=int, default=16)
    p.add_argument("--eval-slumbot", action="store_true", help="After local tournament, separately evaluate all candidates online")
    p.add_argument("--slumbot-hands", type=int, default=1000)
    args = p.parse_args(argv)
    if args.pairs < 100 or args.pairs % 50 or args.equity_samples < 1:
        p.error("pairs must be >=100 and divisible by 50; equity samples must be positive")
    if args.eval_slumbot and (args.slumbot_hands < 100 or args.slumbot_hands % 50):
        p.error("Slumbot hands must be >=100 and divisible by 50")
    specs = json.loads(args.manifest.read_text())["candidates"]
    names = [s["name"] for s in specs]
    if not names or len(set(names)) != len(names) or any(n in GAUNTLET_OPPONENTS for n in names):
        p.error("candidate names must be nonempty, unique, and distinct from bot names")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(format="pro_gauntlet_v1", complete=False, rules_version=RULES_VERSION,
        pro_scripts_version=PRO_VERSION, seed=args.seed, pairs=args.pairs, equity_samples=args.equity_samples,
        ci_unit="duplicate_50_hand_match_pair", matrix={}, candidates={}, missing=[],
        source_sha256={str(f.relative_to(ROOT)): sha256(f) for f in
                       [Path(__file__), *sorted((ROOT / "src/alphanl").rglob("*.py"))]},
        notes=["Same full deck schedule across every candidate, opponent and round-robin cell.",
               "Opponent and policy RNG reset per leg; both sides receive their own public 50-hand HUD.",
               "Six original bots use unchanged parameters and full legal actions; no training-pool changes."])
    publish(args.out_dir, report)
    try:
        candidates = []
        for raw in specs:
            spec = dict(raw)
            path = Path(spec["path"])
            if not path.is_absolute():
                path = args.manifest.resolve().parent / path
            spec["path"] = str(path)
            if not path.exists():
                report["missing"].append(dict(spec, reason="path does not exist"))
                if spec.get("required", True):
                    raise FileNotFoundError(f"required candidate {spec['name']}: {path}")
                continue
            candidate = Candidate(spec, device=args.device, cache_size=args.teacher_cache_size)
            candidates.append(candidate)
            report["matrix"][candidate.name] = {}
            report["candidates"][candidate.name] = candidate.metadata
        if not candidates:
            raise ValueError("no candidates available")
        schedule = np.random.default_rng(args.seed).integers(0, 2**63 - 1, size=(args.pairs // 50, 50))
        report["deck_schedule_sha256"] = __import__("hashlib").sha256(schedule.astype("<i8").tobytes()).hexdigest()
        (args.out_dir / "deck_seeds.json").write_text(json.dumps(schedule.tolist()) + "\n")
        publish(args.out_dir, report)
        def run(a, opponent_name, factory, index):
            path = args.out_dir / f"matchup_{index:04d}.jsonl"
            print(f"Evaluating {a.name} vs {opponent_name}", flush=True)
            with path.open("x") as log:
                def record(row):
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                result = match_factories(a.actor, factory, schedule=schedule, seed=args.seed,
                    log=record, progress=lambda done, total: print(f"  match pairs {done}/{total}", flush=True))
            result.update(raw_log=path.name, raw_log_sha256=sha256(path))
            report["matrix"][a.name][opponent_name] = result
            publish(args.out_dir, report)
            return result
        index = 0
        for a in candidates:
            for name in GAUNTLET_OPPONENTS:
                run(a, name, lambda seed, name=name: rule_actor(name, seed, args.equity_samples), index)
                index += 1
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                result = run(a, b.name, b.actor, index)
                values = -np.asarray(result["net_bb_pairs"])
                report["matrix"][b.name][a.name] = {**summarize_match_pairs(values),
                    "net_bb_pairs": values.tolist(), "raw_log": result["raw_log"],
                    "raw_log_sha256": result["raw_log_sha256"], "derived_by_zero_sum": True}
                index += 1
        report["common_deal_gains"] = {a.name: {b.name: {o: paired_candidate_gain(
            report["matrix"][a.name][o], report["matrix"][b.name][o]) for o in GAUNTLET_OPPONENTS}
            for b in candidates if a.name != b.name} for a in candidates}
        report["selection"] = select_candidate(report["matrix"], [c.name for c in candidates],
            full_budget=args.pairs >= 1000, all_required_present=not any(s.get("required", True) for s in report["missing"]))
        report["complete"] = True
        publish(args.out_dir, report)
        if args.eval_slumbot:
            from eval_pro_slumbot import evaluate_slumbot
            report["slumbot"] = {}
            for i, candidate in enumerate(candidates):
                report["slumbot"][candidate.name] = evaluate_slumbot(candidate, args.out_dir / f"slumbot_{i:02d}",
                    hands=args.slumbot_hands, seed=args.seed)
                publish(args.out_dir, report)
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        publish(args.out_dir, report)
        raise
    print(json.dumps(report["selection"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

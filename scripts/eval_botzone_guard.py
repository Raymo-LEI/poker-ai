"""Frozen C200 baseline/v10 ablation on common duplicate 50-hand matches.

Uses the standard HUNL engine, not the Botzone judge. Protocol/runtime
acceptance is independently checked by tools/accept_botzone_stage2.py.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from alphanl.cfr.gauntlet import Candidate, ModelActor, match_factories, paired_candidate_gain, rule_actor, sha256
from alphanl.stage2.deployment_guard import apply_guard, guard_config, V10_GUARD
from alphanl.transformer.equity import estimated_equity
from alphanl.transformer.hunl import RULES_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--device", default="cpu")
    p.add_argument("--guard-config", type=Path)
    p.add_argument("--opponents", nargs="+", default=["equity", "TAG_Pro", "Trapper_CR"])
    args = p.parse_args()
    if args.pairs < 100 or args.pairs % 50:
        p.error("pairs must be >=100 and a multiple of 50")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = guard_config(json.loads(args.guard_config.read_text()) if args.guard_config else V10_GUARD)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    candidate = Candidate(dict(name="C200", kind="stage2", path=str(args.checkpoint)), device=args.device)
    schedule = np.random.default_rng(args.seed).integers(0, 2**63-1, size=(args.pairs//50, 50))
    report = dict(format="v10_guard_ablation_v1", complete=False, seed=args.seed, pairs=args.pairs,
                  rules_version=RULES_VERSION, engine="standard_hunl_not_botzone_judge", guard=config,
                  deck_schedule_sha256=hashlib.sha256(schedule.astype("<i8").tobytes()).hexdigest(),
                  checkpoint=candidate.metadata, matrix={}, paired_gain={},
                  source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in
                      [Path(__file__), *sorted((ROOT/"src/alphanl").rglob("*.py"))]})
    (args.out_dir/"deck_seeds.json").write_text(json.dumps(schedule.tolist()))
    def publish():
        (args.out_dir/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    publish()
    for mode in ("baseline", "v10"):
        report["matrix"][mode] = {}
        for opponent in args.opponents:
            def factory(seed):
                actor = ModelActor(candidate.model, seed=seed, residual=True)
                if mode == "v10":
                    def filt(snapshot, probabilities):
                        probs, info = apply_guard(snapshot, probabilities, config, estimated_equity)
                        actor.audit.counts["v10_decisions"] += 1
                        actor.audit.counts["v10_pruned_decisions"] += bool(info["pruned"])
                        actor.audit.counts["v10_active_cap"] += info["active_cap"]
                        if info["call_fuse"]:
                            actor.audit.counts["v10_"+info["call_fuse"]] += 1
                        return probs
                    actor.probability_filter = filt
                return actor
            path = args.out_dir/(mode+"_"+opponent+".jsonl")
            print(mode, opponent, flush=True)
            with path.open("x") as log:
                rows = []
                def record(row):
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
                result = match_factories(factory, lambda seed:rule_actor(opponent, seed, 64),
                    schedule=schedule, seed=args.seed, log=record,
                    progress=lambda n,total:print("  %d/%d"%(n,total),flush=True))
            nets = [h["net_bb"] for row in rows for h in row["hands"]]
            counts = Counter()
            for row in rows:
                counts.update(row["left_audit"]["counts"])
            result.update(raw_log=path.name, raw_log_sha256=sha256(path), audit_counts=dict(counts),
                          loss_100bb_hands=sum(n<=-100 for n in nets),
                          loss_190bb_hands=sum(n<=-190 for n in nets),
                          match_wins=sum(row["net_bb"]>0 for row in rows), matches=len(rows))
            report["matrix"][mode][opponent] = result
            if mode == "v10":
                report["paired_gain"][opponent] = paired_candidate_gain(result, report["matrix"]["baseline"][opponent])
            publish()
    report["complete"] = True
    publish()


if __name__ == "__main__":
    main()

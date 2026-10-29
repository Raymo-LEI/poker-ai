"""Explicit opt-in Stage 2 PPO; a 20-match quota block per update."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool
from alphanl.stage2.train import collect_matches, train_matches, save_checkpoint, load_checkpoint
from alphanl.stage2.evaluation import MatchEvaluator, evaluation_metrics


def sanitize_metrics(value):
    """Copy JSON metrics, replacing nonfinite floats with null at any depth."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metrics(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2/botzone_50hand.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    device = args.device or settings["device"]
    updates = args.updates if args.updates is not None else settings["updates"]
    count = settings.get("matches_per_update", 20)
    evaluation = settings.get("evaluation", {})
    every = evaluation.get("every_updates", 100)
    if (isinstance(count, bool) or not isinstance(count, int) or count < 20 or count % 20
            or updates < 1 or not isinstance(every, int) or every < 1):
        parser.error("updates/evaluation interval must be positive; matches_per_update must be a multiple of 20")
    settings.update(device=device, updates=updates)
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    pool = MatchOpponentPool(settings["strategies"], seed=settings["seed"], **settings["pool"])
    if args.resume:
        model, blob = load_checkpoint(args.resume, device=device)
        start = blob["update"]
        if settings["policy"] != model.settings:
            raise ValueError("resume policy settings differ from checkpoint")
        if blob["metadata"].get("evaluation_settings") != evaluation:
            raise ValueError("resume evaluation settings differ from checkpoint")
        timesteps = blob["metadata"].get("timesteps", 0)
        matches = blob["metadata"].get("matches", 0)
        if blob["metadata"].get("retention_failed"):
            raise ValueError("retention-failed checkpoint is diagnostic only; resume a retained checkpoint")
    else:
        model = ResidualExploiterPolicy.from_reference(settings["reference"], **settings["policy"]).to(device)
        start, timesteps, matches = 0, 0, 0
    pool.initialize(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"])
    evaluator = MatchEvaluator(model, match_pairs=evaluation.get("match_pairs", 50),
                               seed=evaluation.get("seed", 92017), equity_samples=evaluation.get("equity_samples", 64))
    if args.resume:
        optimizer.load_state_dict(blob["optimizer"])
        pool.load_state_dict(blob["pool"])
        evaluator.baseline_totals = blob["metadata"].get("evaluation_baseline", {})
        torch.set_rng_state(blob["torch_rng"])
        if device == "cuda" and "cuda_rng" in blob:
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        random.setstate(blob["python_rng"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for update in range(start + 1, start + updates + 1):
        began = time.perf_counter()
        envs = [BotzoneMatchEnv(model.config, pool=pool, seed=settings["seed"] + matches + i,
                               first_hero_seat=i % 2, defer_opponents=True) for i in range(count)]
        rollouts = collect_matches(model, envs)
        metrics = train_matches(model, optimizer, rollouts, **settings["ppo"])
        for env in envs:
            pool.record(env.opponent_name, env.ctx.hero_net_bb)
        if update % pool.self_snapshot_every == 0:
            pool.add_self_snapshot(model, update=update)
        timesteps += sum(len(r.actions) for r in rollouts)
        matches += count
        row = {"update": update, "stage2": 1, "timesteps": timesteps, **metrics,
               "stage2/matches": matches, "stage2/hands": matches * 50,
               "stage2/hero_net_bb": sum(env.ctx.hero_net_bb for env in envs) / count,
               "stage2/match_win": sum(env.ctx.hero_net_bb > 0 for env in envs) / count,
               "stage2/defense_fraction": sum(env.opponent_name in ("reference", "history", "self") for env in envs) / count,
               "stage2/history_resident": len(pool.history_models), "stage2/history_disk_loads": pool.disk_loads,
               "stage2/self_snapshots": len(pool.self_weights),
               **{f"stage2/context/{k}": v for k, v in envs[-1].ctx.as_dict().items()},
               **{f"stage2/pfsp/{k}": v for k, v in pool.probabilities().items()},
               **{f"stage2/window/{name}/{key}": value for name, stats in pool.rolling_metrics().items()
                  for key, value in stats.items()},
               **{f"stage2/sampled/{name}": sum(env.opponent_name == name for env in envs) for name in pool.weights}}
        report = None
        if update % every == 0:
            report = evaluator.evaluate(model, update=update,
                progress=lambda name: print(f"Stage 2 update {update}: evaluating {name}", flush=True))
            report_dir = args.out_dir / "stage2_evaluations"
            report_dir.mkdir(exist_ok=True)
            (report_dir / f"update_{update:07d}.json").write_text(
                json.dumps(sanitize_metrics(report), indent=2, allow_nan=False) + "\n")
            row.update(evaluation_metrics(report))
        failed = report is not None and not report["retention_pass"]
        row["seconds"] = time.perf_counter() - began
        row = sanitize_metrics(row)
        with (args.out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        metadata = {"reference": settings["reference"], "stage": "stage2", "timesteps": timesteps,
                    "matches": matches, "evaluation_settings": evaluation,
                    "evaluation_baseline": evaluator.baseline_totals, "retention_failed": failed,
                    "last_matches": [r.report for r in rollouts]}
        save_checkpoint(args.out_dir / "train.pt", model, optimizer=optimizer, pool=pool, update=update, metadata=metadata)
        if report is not None and not failed:
            save_checkpoint(args.out_dir / "retained.pt", model, optimizer=optimizer, pool=pool, update=update, metadata=metadata)
        print(json.dumps(row, allow_nan=False), flush=True)
        if failed and evaluation.get("stop_on_retention_failure", True):
            raise RuntimeError("GTO retention below -20 mbb/hand; diagnostic checkpoint saved, PPO stopped")


if __name__ == "__main__":
    main()

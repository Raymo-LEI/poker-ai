"""Explicit opt-in Stage 2 trainer. Defaults to two CPU matches, never auto-launches."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from alphanl.stage2.env import BotzoneMatchEnv
from alphanl.stage2.policy import ResidualExploiterPolicy
from alphanl.stage2.pool import MatchOpponentPool
from alphanl.stage2.train import collect_match, train_match, save_checkpoint, load_checkpoint

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
    if updates < 1:
        parser.error("updates must be positive")
    settings.update(device=device, updates=updates)
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])
    random.seed(settings["seed"])
    pool = MatchOpponentPool(settings["strategies"], seed=settings["seed"], **settings["pool"])
    # Validate every history snapshot before training, so the 15% branch cannot fail late.
    from alphanl.cfr.distillation import prepare_archive
    history_config, _ = prepare_archive(settings["strategies"])
    if args.resume:
        model, blob = load_checkpoint(args.resume, device=device)
        start = blob["update"]
        if settings["policy"] != model.settings:
            raise ValueError("resume policy settings differ from checkpoint")
        timesteps = blob["metadata"].get("timesteps", 0)
    else:
        model = ResidualExploiterPolicy.from_reference(settings["reference"], **settings["policy"]).to(device)
        start, timesteps = 0, 0
    from alphanl.cfr.distillation import SEMANTIC_FIELDS
    if any(getattr(history_config, k) != getattr(model.config, k) for k in SEMANTIC_FIELDS):
        raise ValueError("history opponents and reference have different Stage 1 contracts")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings["learning_rate"])
    if args.resume:
        optimizer.load_state_dict(blob["optimizer"])
        pool.load_state_dict(blob["pool"])
        torch.set_rng_state(blob["torch_rng"])
        if device == "cuda" and "cuda_rng" in blob:
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        random.setstate(blob["python_rng"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    for update in range(start + 1, start + updates + 1):
        began = time.perf_counter()
        name, opponent = pool.sample()
        env = BotzoneMatchEnv(model.config, opponent, seed=settings["seed"] + update)
        rollout = collect_match(model, env)
        metrics = train_match(model, optimizer, rollout, **settings["ppo"])
        pool.record(name, env.ctx.hero_net_bb)
        timesteps += len(rollout.actions)
        row = {"update": update, "stage2": 1, "timesteps": timesteps, "seconds": time.perf_counter() - began, **metrics,
               "stage2/matches": update, "stage2/hands": update * 50,
               "stage2/hero_net_bb": env.ctx.hero_net_bb, "stage2/match_win": float(env.ctx.hero_net_bb > 0),
               "stage2/opponent": name,
               **{f"stage2/context/{k}": v for k, v in env.ctx.as_dict().items()},
               **{f"stage2/pfsp/{k}": v for k, v in pool.probabilities().items()}}
        with (args.out_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        save_checkpoint(args.out_dir / "train.pt", model, optimizer=optimizer, pool=pool, update=update,
                        metadata={"reference": settings["reference"], "stage": "stage2", "last_match": rollout.report, "timesteps": timesteps})
        print(json.dumps(row, allow_nan=False), flush=True)

if __name__ == "__main__":
    main()

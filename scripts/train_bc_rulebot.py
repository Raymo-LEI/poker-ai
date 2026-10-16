#!/usr/bin/env python3
"""KL behavior cloning from ``scripts/gen_rulebot_bc.py`` into the botzone transformer.

The loss and the minibatch update are :func:`train_bc.train_kl_epoch` and
:func:`train_bc.kl_value`, which call ``bc_update`` / ``bc_losses``. This
file does not reimplement that KL. The network is the config's
``laptop_8gb`` transformer. Observations are the token fields already stored
in the npz (the same tokenizer PPO uses). ``legal_mask`` is the maskfold
policy mask.

Ten percent of **games** (``game_id``), not rows, are held out. The draw is
``numpy`` ``PCG64(data_seed)`` over the sorted game ids.
``n_val = clamp(round(0.1 * n_games), 1, n_games - 1)``.

The checkpoint is the PPO checkpoint format with ``update = -1``, so
``--resume`` starts PPO at update 0. ``bc_metadata`` on the same file records
``mask_free_fold``, the data's environment version, ``legal_py_sha256``, and
the BC data seed. Training refuses a dataset whose hash is not the current
``src/alphanl/botzone/legal.py``.

``--resume`` restores Adam, including the learning rate used for this BC run.
The continuation config's ``1e-4`` is what a fresh Adam gets from
``--init-weights``. See that config.

    PYTHONPATH=src python3 scripts/train_bc_rulebot.py \\
        --config configs/transformer/botzone_bc_rulebot.json \\
        --data runs/bc_rulebot \\
        --out runs/bc_rulebot_train/bc.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_bc  # noqa: E402
from alphanl.transformer.checkpoint import load_checkpoint  # noqa: E402
from alphanl.transformer.config import ExperimentConfig, resolve_device  # noqa: E402
from alphanl.transformer.tokenizer import slice_batch  # noqa: E402
from alphanl.transformer.train import build_model, build_optimizer  # noqa: E402

VAL_FRACTION = 0.1
_GEN_NAME = "gen_rulebot_bc"


def _gen():
    cached = sys.modules.get(_GEN_NAME)
    if cached is not None:
        return cached
    path = REPO_ROOT / "scripts" / "gen_rulebot_bc.py"
    spec = importlib.util.spec_from_file_location(_GEN_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GEN_NAME] = module
    spec.loader.exec_module(module)
    return module


def split_game_mask(
    game_ids: np.ndarray,
    seed: int,
    fraction: float = VAL_FRACTION,
) -> Tuple[np.ndarray, List[int]]:
    """Boolean val mask over rows, and the held-out game ids.

    Every row of a held-out game is in the mask. No game is in both splits.
    """

    games = np.array(sorted({int(game) for game in np.asarray(game_ids).tolist()}), dtype=np.int64)
    n_games = int(games.shape[0])
    if n_games < 2:
        raise ValueError(f"need at least 2 games to hold out {fraction:.0%}, got {n_games}")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"val fraction must be in (0, 1), got {fraction}")
    n_val = int(round(float(fraction) * n_games))
    n_val = min(max(n_val, 1), n_games - 1)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    picked = rng.choice(games, size=n_val, replace=False)
    val_games = {int(game) for game in picked.tolist()}
    val_mask = np.asarray([int(game) in val_games for game in np.asarray(game_ids).tolist()], dtype=bool)
    if int(val_mask.sum()) == 0 or int((~val_mask).sum()) == 0:
        raise RuntimeError("game holdout produced an empty split")
    return val_mask, sorted(val_games)


def _observation_tensors(arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    required = ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask", "probs", "actions", "game_id")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"rule-bot BC npz is missing {missing}")
    numeric = np.asarray(arrays["numeric"], dtype=np.float32)
    legal = np.asarray(arrays["legal_mask"], dtype=np.float32)
    probs = np.asarray(arrays["probs"], dtype=np.float32)
    if not np.isfinite(numeric).all() or not np.isfinite(legal).all() or not np.isfinite(probs).all():
        raise RuntimeError("refusing BC data with NaN or Inf in observations or teacher probs")
    return {
        "token_ids": torch.from_numpy(np.asarray(arrays["token_ids"], dtype=np.int64)),
        "type_ids": torch.from_numpy(np.asarray(arrays["type_ids"], dtype=np.int64)),
        "numeric": torch.from_numpy(numeric),
        "pad_mask": torch.from_numpy(np.asarray(arrays["pad_mask"]).astype(bool)),
        "legal_mask": torch.from_numpy(legal),
        "probs": torch.from_numpy(probs),
        "actions": torch.from_numpy(np.asarray(arrays["actions"], dtype=np.int64)),
        "game_id": torch.from_numpy(np.asarray(arrays["game_id"], dtype=np.int64)),
    }


def _take(tensors: Dict[str, torch.Tensor], mask: np.ndarray, device: torch.device):
    index = torch.from_numpy(np.flatnonzero(mask).astype(np.int64)).to(device)
    moved = {key: value.to(device) for key, value in tensors.items()}
    batch = slice_batch({key: moved[key] for key in ("token_ids", "type_ids", "numeric", "pad_mask", "legal_mask")}, index)
    teacher = moved["probs"].index_select(0, index)
    return batch, teacher


def assert_train_out(path: Path) -> Path:
    """Allow a new ``runs/bc_rulebot_train`` tree. Refuse every other ``runs/`` path."""

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    runs = (REPO_ROOT / "runs").resolve()
    allowed = (runs / "bc_rulebot_train").resolve()
    if path == runs or (runs in path.parents and allowed != path and allowed not in path.parents):
        raise SystemExit(
            f"refusing to write {path}. Under runs/, only runs/bc_rulebot_train/ is allowed."
        )
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _attach_metadata(path: Path, metadata: dict) -> None:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    blob["bc_metadata"] = metadata
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(blob, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def train_rulebot_bc(
    *,
    config: Path,
    data: Path,
    out: Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    device: str | None = None,
) -> dict:
    """Train and write the checkpoint. Returns the ``bc_metadata`` dict."""

    gen = _gen()
    arrays, meta = gen.load_dataset(data, check_legal=True)
    cfg = ExperimentConfig.from_json(config)
    if cfg.encoder != "transformer":
        raise ValueError("rule-bot BC uses the botzone transformer, not the conv encoder")
    if cfg.env.backend != "botzone":
        raise ValueError("rule-bot BC config env.backend must be botzone")
    if not cfg.env.mask_free_fold:
        raise ValueError("rule-bot BC config must set env.mask_free_fold true")
    if device is not None:
        cfg.device = device
    n_epochs = int(cfg.ppo.epochs if epochs is None else epochs)
    batch = int(cfg.ppo.minibatch_size if batch_size is None else batch_size)
    learning_rate = float(cfg.ppo.lr if lr is None else lr)
    if n_epochs < 1 or batch < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and lr must be positive")
    cfg.ppo.epochs = n_epochs
    cfg.ppo.minibatch_size = batch
    cfg.ppo.lr = learning_rate
    cfg.train.num_updates = n_epochs

    val_mask, val_games = split_game_mask(arrays["game_id"], int(meta["seed"]), VAL_FRACTION)
    tensors = _observation_tensors(arrays)
    torch.manual_seed(int(cfg.seed))
    device_t = torch.device(resolve_device(cfg.device))
    model = build_model(cfg).to(device_t)
    optimizer = build_optimizer(model, cfg)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    train_batch, train_teacher = _take(tensors, ~val_mask, device_t)
    val_batch, val_teacher = _take(tensors, val_mask, device_t)
    minibatch = min(batch, int(train_teacher.shape[0]))
    curve: List[dict] = []
    for epoch in range(n_epochs):
        train_kl = train_bc.train_kl_epoch(
            model,
            optimizer,
            train_batch,
            train_teacher,
            minibatch_size=minibatch,
            max_grad_norm=float(cfg.ppo.max_grad_norm),
        )
        val_kl = train_bc.kl_value(model, val_batch, val_teacher, minibatch_size=minibatch)
        row = {"epoch": epoch, "train_kl": train_kl, "val_kl": val_kl}
        curve.append(row)
        print(f"epoch {epoch} train_kl {train_kl:.6f} val_kl {val_kl:.6f}", flush=True)
    for param in model.parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError("non-finite parameter after BC")
    out = assert_train_out(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_bc.save_bc_checkpoint(out, model=model, optimizer=optimizer, config=cfg.to_dict())
    train_games = sorted({int(game) for game in arrays["game_id"].tolist()} - set(val_games))
    bc_metadata = {
        "mask_free_fold": True,
        "env_version": meta["env_version"],
        "legal_py_sha256": meta["env_version"]["legal_py_sha256"],
        "data_seed": int(meta["seed"]),
        "split_seed": int(meta["seed"]),
        "val_fraction": VAL_FRACTION,
        "val_game_ids": [int(game) for game in val_games],
        "train_game_ids": train_games,
        "n_train_rows": int((~val_mask).sum()),
        "n_val_rows": int(val_mask.sum()),
        "epochs": n_epochs,
        "batch_size": batch,
        "lr": learning_rate,
        "loss": "kl",
        "kl_curve": curve,
        "max_seq_len": int(cfg.model.max_seq_len),
    }
    _attach_metadata(out, bc_metadata)
    _write_json(out.with_name("bc_metadata.json"), bc_metadata)
    _write_json(out.with_name("kl_curve.json"), {"kl_curve": curve})
    # Reload once so a truncated write fails here, not in PPO.
    blob = load_checkpoint(out, model, optimizer=None, map_location="cpu", restore_rng=False)
    if int(blob["update"]) != -1:
        raise RuntimeError(f"BC checkpoint update must be -1, got {blob['update']}")
    if "bc_metadata" not in blob:
        raise RuntimeError("BC checkpoint is missing bc_metadata")
    ExperimentConfig.from_dict(blob["config"])
    print(f"wrote {out}")
    return bc_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "transformer" / "botzone_bc_rulebot.json")
    parser.add_argument("--data", type=Path, required=True, help="Directory with decisions.npz and metadata.json")
    parser.add_argument("--out", type=Path, default=None, help="Default: <config train.out_dir>/bc.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Default: config ppo.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: config ppo.minibatch_size")
    parser.add_argument("--lr", type=float, default=None, help="Default: config ppo.lr")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_json(args.config)
    out = args.out if args.out is not None else Path(cfg.train.out_dir) / "bc.pt"
    try:
        train_rulebot_bc(
            config=args.config,
            data=args.data,
            out=out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    except _gen().LegalPyMismatch as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

"""Export a transformer checkpoint to an fp32 ``.npz`` the Botzone entry loads.

One command, weights in and ``.npz`` out::

    python3 tools/export_numpy_weights.py --checkpoint path/to/update_0000999.pt --out model.npz

``--random`` builds a freshly initialised ``laptop_8gb`` botzone network
from ``configs/transformer/botzone_laptop_8gb_maskfold.json`` when that
checkpoint file is not on the machine. It does not run a training step.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphanl.transformer.checkpoint import save_checkpoint  # noqa: E402
from alphanl.transformer.config import ExperimentConfig  # noqa: E402
from alphanl.transformer.train import build_model, set_seed  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "transformer" / "botzone_laptop_8gb_maskfold.json"
LEGAL_PY = ROOT / "src" / "alphanl" / "botzone" / "legal.py"
ENV_VERSION = "botzone-simple-50-100-20000"


def export_model(model, config, dest, source):
    """Write ``dest`` and return the metadata dict."""

    model.eval()
    arch = _arch(model, config)
    env = config.env
    meta = {
        "format": 1,
        "arch": arch,
        "mask_free_fold": bool(env.mask_free_fold),
        "sample": False,
        "legal_py_sha256": hashlib.sha256(LEGAL_PY.read_bytes()).hexdigest(),
        "environment": {
            "version": ENV_VERSION,
            "backend": str(env.backend),
            "small_blind": float(env.small_blind),
            "big_blind": float(env.big_blind),
            "starting_stack": float(env.starting_stack),
            "observation": "HandSnapshot/BettingHistoryTokenizer",
        },
        "source": source,
        "export": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _torch_version(),
        },
    }
    arrays = {}
    count = 0
    for name, tensor in model.state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().float().numpy(), dtype=np.float32)
        arrays[name] = array
        count += int(array.size)
    meta["parameter_count"] = count
    meta["weight_bytes_fp32"] = count * 4
    _write(dest, arrays, meta)
    return meta


def export_checkpoint(checkpoint, dest, config_path=None):
    """Load a training ``update_*.pt`` and write the ``.npz``."""

    import torch

    path = Path(checkpoint)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict) or "model" not in blob:
        raise SystemExit("checkpoint has no model state_dict: %s" % path)
    config = _config_from_blob(blob, config_path)
    if config.encoder != "transformer":
        raise SystemExit("export only supports the transformer encoder")
    set_seed(int(config.seed))
    model = build_model(config)
    model.load_state_dict(blob["model"])
    source = {
        "weights_source": "checkpoint",
        "checkpoint": str(path),
        "update": blob.get("update"),
        "config_name": config.name,
        "random_init": False,
    }
    return export_model(model, config, dest, source)


def export_random(dest, config_path, seed, write_checkpoint):
    """Initialise the maskfold architecture and write the ``.npz``."""

    config = ExperimentConfig.from_json(config_path)
    if config.encoder != "transformer":
        raise SystemExit("export only supports the transformer encoder")
    set_seed(int(seed))
    model = build_model(config)
    ckpt = None
    if write_checkpoint:
        ckpt = Path(write_checkpoint)
        save_checkpoint(
            ckpt,
            model=model,
            optimizer=None,
            update=0,
            config=config.to_dict(),
            keep_last=0,
        )
    source = {
        "weights_source": "random_init",
        "checkpoint": None if ckpt is None else str(ckpt),
        "update": 0,
        "config_name": config.name,
        "config_path": str(Path(config_path)),
        "seed": int(seed),
        "random_init": True,
    }
    return export_model(model, config, dest, source)


def _config_from_blob(blob, config_path):
    payload = blob.get("config")
    if isinstance(payload, dict) and payload.get("model"):
        return ExperimentConfig.from_dict(payload)
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    return ExperimentConfig.from_json(path)


def _arch(model, config):
    layer = model.encoder.layers[0]
    return {
        "d_model": int(model.d_model),
        "nhead": int(layer.self_attn.num_heads),
        "num_layers": int(len(model.encoder.layers)),
        "dim_feedforward": int(layer.linear1.out_features),
        "dropout": float(config.model.dropout),
        "max_seq_len": int(model.max_seq_len),
        "vocab_size": int(model.token_emb.num_embeddings),
        "num_types": int(model.type_emb.num_embeddings),
        "num_numeric": int(model.numeric_proj.in_features),
        "num_actions": int(model.num_actions),
        "activation": "gelu",
        "norm_first": True,
        "layer_norm_eps": 1e-5,
        "illegal_logit": -1.0e9,
    }


def _write(dest, arrays, meta):
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = dict(arrays)
    payload["meta_json"] = np.frombuffer(raw, dtype=np.uint8).copy()
    np.savez(str(path), **payload)


def _torch_version():
    import torch

    return torch.__version__


def main():
    parser = argparse.ArgumentParser(description="Export fp32 NumPy weights for the Botzone model entry.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Training update_*.pt to read.")
    parser.add_argument("--out", type=str, required=True, help="Destination .npz path.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--random",
        action="store_true",
        help="Export a freshly initialised laptop_8gb botzone network. Does not train.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-checkpoint",
        type=str,
        default=None,
        help="With --random, also write a training-format .pt at this path.",
    )
    args = parser.parse_args()
    if args.random == bool(args.checkpoint):
        raise SystemExit("pass exactly one of --checkpoint and --random")
    if args.random:
        meta = export_random(args.out, args.config, args.seed, args.write_checkpoint)
    else:
        meta = export_checkpoint(args.checkpoint, args.out, args.config)
    size = Path(args.out).stat().st_size
    print(
        json.dumps(
            {
                "out": args.out,
                "bytes": size,
                "parameter_count": meta["parameter_count"],
                "mask_free_fold": meta["mask_free_fold"],
                "legal_py_sha256": meta["legal_py_sha256"],
                "environment_version": meta["environment"]["version"],
                "random_init": meta["source"]["random_init"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

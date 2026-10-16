"""Load the exported fp32 ``.npz``.

Botzone does not put data files in the source zip. The wiki tells a
Python bot to read them from the ``data`` path after they are uploaded
to the account storage page. This loader opens ``data/model.npz``.
``POKER_WEIGHTS`` is a local override for tests; Botzone does not set it.
"""

import json
import os

_CACHE = {}


def weights_path():
    """Path the entry will open. Local tests may set ``POKER_WEIGHTS``."""

    override = os.environ.get("POKER_WEIGHTS")
    if override:
        return override
    return os.path.join("data", "model.npz")


def load_weights(path=None):
    """Return ``(arrays, meta)``. Raises if the file is missing or short."""

    import numpy as np

    target = weights_path() if path is None else path
    cached = _CACHE.get(target)
    if cached is not None:
        return cached
    if not os.path.isfile(target):
        raise IOError("weights file is missing: %s" % target)
    archive = np.load(target)
    try:
        arrays = {}
        meta_raw = None
        for key in archive.files:
            if key == "meta_json":
                meta_raw = archive[key]
            else:
                arrays[key] = np.array(archive[key], dtype=np.float32)
    finally:
        archive.close()
    if meta_raw is None:
        raise ValueError("weights file has no meta_json")
    meta = json.loads(meta_raw.tobytes().decode("utf-8"))
    _require(arrays, "token_emb.weight")
    _require(arrays, "policy_head.weight")
    _require(arrays, "value_head.weight")
    arch = meta.get("arch") or {}
    layers = int(arch.get("num_layers", 0))
    if layers < 1:
        raise ValueError("weights meta has no layers")
    for index in range(layers):
        _require(arrays, "encoder.layers.%d.self_attn.in_proj_weight" % index)
    loaded = (arrays, meta)
    _CACHE[target] = loaded
    return loaded


def clear_cache():
    """Tests swap files under the same path."""

    _CACHE.clear()


def _require(arrays, key):
    if key not in arrays:
        raise ValueError("weights file is missing %s" % key)

"""Load a c_1048 behavior-cloning npz into the conv control-group batch.

The dump script runs in the Python 3.7 / TensorFlow 1.15 environment and
writes ``card_info``, ``action_info``, ``extra_info``, and ``legal_mask``.
The mainline Transformer does not consume this layout. Its token rebuild
stays in ``alphanl.transformer.bc``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def conv_tensors_from_npz(arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    required = ("card_info", "action_info", "extra_info", "legal_mask", "probs", "actions")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError("BC npz is missing %s" % missing)
    return {
        "card_info": torch.from_numpy(np.asarray(arrays["card_info"], dtype=np.float32)),
        "action_info": torch.from_numpy(np.asarray(arrays["action_info"], dtype=np.float32)),
        "extra_info": torch.from_numpy(np.asarray(arrays["extra_info"], dtype=np.float32)),
        "legal_mask": torch.from_numpy(np.asarray(arrays["legal_mask"], dtype=np.float32)),
        "probs": torch.from_numpy(np.asarray(arrays["probs"], dtype=np.float32)),
        "actions": torch.from_numpy(np.asarray(arrays["actions"], dtype=np.int64)),
    }

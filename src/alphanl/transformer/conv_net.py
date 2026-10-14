"""PyTorch port of AlphaNLHoldem's TF 1.15 ``NlHoldemNet``.

The topology, including the quirks, follows ``agi/nl_holdem_net.py``:

* card input ``(4, 13, 6)`` and action-history input ``(4, 5, 25)``, NHWC
* three stages ``[16, s1, 1 block], [32, s2, 2 blocks], [64, s2, 2 blocks]``
* a residual add only when the stage index is greater than 1
* stride applied on every block of a stage, not only the first block
* ``extra_fc`` is a 16-d ReLU on the flattened card vector, not on ``extra_info``
* then Dense 256 → 128 → 64, a 5-d policy (``conv_fuse``) and a scalar value

TensorFlow ``padding="same"`` with stride 2 is not ``nn.Conv2d(padding="same")``
(PyTorch rejects stride other than 1). ``tf_same_pad`` reproduces the TF pad,
including the asymmetric pad on even lengths.

Flatten is NHWC order so a loaded ``fc_1`` / ``extra_fc`` kernel lines up with
the TF variable. Observations stay NHWC, the same layout the env produces.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import masked_log_probs, policy_act
from .schema import NUM_ACTIONS
from .tokenizer import TensorBatch

CARD_FLAT = 64
HISTORY_FLAT = 64
STAGES = (
    (16, 3, 1, 1),
    (32, 3, 2, 2),
    (64, 3, 2, 2),
)


def tf_same_pad(x: torch.Tensor, kernel: Tuple[int, int], stride: Tuple[int, int]) -> torch.Tensor:
    """Pad NCHW ``x`` the way TF ``Conv2D(padding='same')`` pads before a valid conv."""

    _, _, height, width = x.shape
    kh, kw = kernel
    sh, sw = stride
    out_h = (height + sh - 1) // sh
    out_w = (width + sw - 1) // sw
    pad_h = max((out_h - 1) * sh + kh - height, 0)
    pad_w = max((out_w - 1) * sw + kw - width, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class TFConv2d(nn.Conv2d):
    """Conv2d whose padding matches TF ``same`` for any stride."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int) -> None:
        k = (int(kernel), int(kernel))
        s = (int(stride), int(stride))
        super().__init__(in_channels, out_channels, k, stride=s, padding=0, bias=True)
        self.tf_kernel = k
        self.tf_stride = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(tf_same_pad(x, self.tf_kernel, self.tf_stride))


def _flatten_nhwc(x: torch.Tensor) -> torch.Tensor:
    """NCHW → NHWC row-major flatten. This is what Keras ``Flatten`` does."""

    return x.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], -1)


class ConvTower(nn.Module):
    def __init__(self, in_channels: int, prefix: str) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        self.specs: List[Tuple[str, str, Optional[str], int]] = []
        channels = int(in_channels)
        for stage, (out_size, kernel, stride, blocks) in enumerate(STAGES):
            for block in range(blocks):
                name = "%s_conv%db%d" % (prefix, stage, block)
                name2 = "%s_conv%db%d_2" % (prefix, stage, block)
                self.layers[name] = TFConv2d(channels, out_size, kernel, stride)
                self.layers[name2] = TFConv2d(out_size, out_size, kernel, 1)
                shortcut = None
                if stride > 1 and stage > 1:
                    shortcut = "%s_conv%db%d_s" % (prefix, stage, block)
                    self.layers[shortcut] = TFConv2d(channels, out_size, 1, 2)
                self.specs.append((name, name2, shortcut, stage))
                channels = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name, name2, shortcut, stage in self.specs:
            y = F.relu(self.layers[name](x))
            y = self.layers[name2](y)
            if shortcut is not None:
                x = self.layers[shortcut](x)
            if stage > 1:
                x = x + y
            else:
                x = y
            x = F.relu(x)
        return x


def _as_nchw(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("conv input must be NHWC or NCHW, got shape %s" % (tuple(x.shape),))
    if x.shape[-1] == channels:
        return x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] == channels:
        return x
    raise ValueError("expected %d channels, got shape %s" % (channels, tuple(x.shape)))


class AlphaNLConvNet(nn.Module):
    """Control-group encoder: the published conv net, with the shared action mask.

    ``forward`` returns masked 5-d logits and a scalar value, same contract as
    ``PokerTransformer``. The internal tower is the original 256-128-64 stack,
    not the transformer's GELU trunk. Those widths are what ``c_1048`` stores.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_actions = NUM_ACTIONS
        self.card = ConvTower(6, "card")
        self.history = ConvTower(25, "history")
        self.extra_fc = nn.Linear(CARD_FLAT, 16)
        fused = CARD_FLAT + HISTORY_FLAT + 16
        self.fc_1 = nn.Linear(fused, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)
        self.conv_fuse = nn.Linear(64, NUM_ACTIONS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # TF used normc(0.01). Orthogonal with the same gain keeps the
                # fresh policy near-uniform; loaded c_1048 weights replace this.
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def encode(self, card_info: torch.Tensor, action_info: torch.Tensor) -> torch.Tensor:
        card = self.card(_as_nchw(card_info, 6))
        history = self.history(_as_nchw(action_info, 25))
        card_flat = _flatten_nhwc(card)
        history_flat = _flatten_nhwc(history)
        if card_flat.shape[-1] != CARD_FLAT or history_flat.shape[-1] != HISTORY_FLAT:
            raise RuntimeError(
                "flatten size card=%d history=%d, expected %d and %d. "
                "TF same-padding was not reproduced."
                % (card_flat.shape[-1], history_flat.shape[-1], CARD_FLAT, HISTORY_FLAT)
            )
        # Upstream bug, ported on purpose: extra_fc reads the card vector.
        extra = F.relu(self.extra_fc(card_flat))
        fused = torch.cat([card_flat, history_flat, extra], dim=-1)
        hidden = F.relu(self.fc_1(fused))
        hidden = F.relu(self.fc_2(hidden))
        return F.relu(self.fc_3(hidden))

    def forward(
        self,
        batch: TensorBatch,
        *,
        mask_illegal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "extra_info" not in batch:
            raise KeyError("alphanl_conv batch is missing extra_info")
        features = self.encode(batch["card_info"], batch["action_info"])
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            logits = self.conv_fuse(features)
            value = self.value_out(features).squeeze(-1)
        if mask_illegal:
            logits, _, _, _ = masked_log_probs(logits, batch["legal_mask"])
        return logits, value

    def act(
        self,
        batch: TensorBatch,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return policy_act(self, batch, deterministic=deterministic)


def _split_tf_key(key: object) -> Optional[Tuple[str, str]]:
    if isinstance(key, bytes):
        key = key.decode("utf-8", "replace")
    text = str(key).split(":")[0].strip("/")
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return None
    kind = parts[-1]
    if kind not in ("kernel", "bias"):
        return None
    return parts[-2], kind


def _index_params(model: AlphaNLConvNet) -> Dict[Tuple[str, str], torch.nn.Parameter]:
    found: Dict[Tuple[str, str], torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        leaf = name.split(".")[-1]
        layer = name.split(".")[-2]
        if leaf == "weight":
            found[(layer, "kernel")] = param
        elif leaf == "bias":
            found[(layer, "bias")] = param
    return found


def _to_torch_layout(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    if array.ndim == 4:
        array = np.transpose(array, (3, 2, 0, 1))
    elif array.ndim == 2:
        array = np.transpose(array, (1, 0))
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)


def export_tf_weights(model: AlphaNLConvNet) -> Dict[str, np.ndarray]:
    """Inverse of ``load_tf_weights``, for tests and for writing a pickle."""

    exported: Dict[str, np.ndarray] = {}
    for (layer, kind), param in _index_params(model).items():
        array = param.detach().cpu().numpy()
        if array.ndim == 4:
            array = np.transpose(array, (2, 3, 1, 0))
        elif array.ndim == 2:
            array = np.transpose(array, (1, 0))
        exported["oppo_policy/%s/%s" % (layer, kind)] = np.ascontiguousarray(array)
    return exported


def _coerce_weight_dict(blob: object) -> Dict[object, np.ndarray]:
    if isinstance(blob, dict) and any(_split_tf_key(key) for key in blob):
        return blob
    if isinstance(blob, dict):
        for value in blob.values():
            if isinstance(value, dict) and any(_split_tf_key(key) for key in value):
                return value
    raise ValueError("pickle does not look like an AlphaNLHoldem weight dict")


def load_pickle(path: Path | str) -> object:
    path = Path(path)
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def load_tf_weights(
    model: AlphaNLConvNet,
    source: Path | str | Dict[object, np.ndarray],
    *,
    strict: bool = True,
) -> List[str]:
    """Copy a ``c_1048``-style pickle into this module.

    TF conv kernels are HWIO and dense kernels are ``(in, out)``. Returns the
    PyTorch parameter keys that were not in the file. ``strict`` raises if any
    are missing. Optimizer slots and unrelated keys are ignored.
    """

    blob = source if isinstance(source, dict) else _coerce_weight_dict(load_pickle(source))
    incoming: Dict[Tuple[str, str], np.ndarray] = {}
    for key, value in blob.items():
        parsed = _split_tf_key(key)
        if parsed is None:
            continue
        array = np.asarray(value)
        if parsed in incoming and incoming[parsed].shape != array.shape:
            raise ValueError("conflicting TF tensors for %s: %s vs %s" % (parsed, incoming[parsed].shape, array.shape))
        if parsed not in incoming:
            incoming[parsed] = array
    missing: List[str] = []
    with torch.no_grad():
        for key, param in _index_params(model).items():
            if key not in incoming:
                missing.append("%s/%s" % key)
                continue
            tensor = _to_torch_layout(incoming[key])
            if tuple(tensor.shape) != tuple(param.shape):
                raise ValueError(
                    "%s/%s: TF %s -> torch %s, parameter is %s"
                    % (key[0], key[1], tuple(incoming[key].shape), tuple(tensor.shape), tuple(param.shape))
                )
            param.copy_(tensor)
    if strict and missing:
        raise KeyError("c_1048 pickle is missing layers: %s" % missing)
    return missing

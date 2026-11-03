"""NumPy forward for the laptop_8gb poker transformer.

Eval mode: dropout is off. The encoder is pre-norm (LayerNorm, then
attention, then the residual; the same order as ``norm_first=True``).
GELU is the erf form. Attention scales the queries by ``head_dim ** -0.5``
before the product, then masks padded keys with -inf.

Illegal policy logits are set to -1e9. When ``mask_free_fold`` is set,
fold is also illegal on a row whose ``to_call`` is 0 and whose check/call
id is legal. A row with no legal id is left unmasked so the softmax stays
finite.
"""

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ILLEGAL_LOGIT = np.float32(-1.0e9)
_LAYER_NORM_EPS = np.float32(1.0e-5)


def forward_logits(weights, meta, batch, to_call):
    """Masked policy logits, shape ``(batch, num_actions)``, float32."""

    raw, _value = raw_forward(weights, meta, batch)
    masked = mask_logits(
        raw,
        batch["legal_mask"],
        to_call,
        bool(meta.get("mask_free_fold")),
    )
    if not np.isfinite(masked).all():
        raise ValueError("non-finite logits")
    return masked


def raw_forward(weights, meta, batch):
    """Unmasked logits and value. Both float32."""

    arch = meta["arch"]
    nhead = int(arch["nhead"])
    layers = int(arch["num_layers"])
    token_ids = batch["token_ids"]
    type_ids = batch["type_ids"]
    numeric = batch["numeric"]
    pad_mask = np.asarray(batch["pad_mask"], dtype=np.bool_)
    hidden = weights["token_emb.weight"][token_ids]
    hidden = hidden + weights["type_emb.weight"][type_ids]
    positions = np.arange(token_ids.shape[1])
    hidden = hidden + weights["pos_emb.weight"][positions][None, :, :]
    hidden = _linear(numeric, weights["numeric_proj.weight"], weights["numeric_proj.bias"]) + hidden
    for index in range(layers):
        hidden = _encoder_layer(hidden, weights, index, nhead, pad_mask)
    cls = hidden[:, 0, :]
    features = _gelu(_linear(cls, weights["trunk.0.weight"], weights["trunk.0.bias"]))
    logits = _linear(features, weights["policy_head.weight"], weights["policy_head.bias"])
    value = _linear(features, weights["value_head.weight"], weights["value_head.bias"])
    value = value.reshape(-1)
    return logits.astype(np.float32, copy=False), value.astype(np.float32, copy=False)


def mask_logits(logits, legal_mask, to_call, mask_free_fold):
    """Environment mask, then the optional free-check fold mask."""

    legal = np.asarray(legal_mask, dtype=np.float32) > np.float32(0.5)
    if mask_free_fold:
        owed = np.asarray(to_call, dtype=np.float32).reshape(-1)
        if owed.shape[0] != legal.shape[0]:
            raise ValueError("to_call does not line up with the legal mask")
        free = np.isfinite(owed) & (owed == np.float32(0))
        drop = free & legal[:, 1]
        legal = legal.copy()
        legal[drop, 0] = False
    none_legal = ~legal.any(axis=-1, keepdims=True)
    legal = legal | none_legal
    return np.where(legal, logits, ILLEGAL_LOGIT).astype(np.float32, copy=False)


def select_action(logits, deterministic, rng):
    """Argmax, or one draw from the masked softmax."""

    row = np.asarray(logits, dtype=np.float32).reshape(-1)
    if deterministic:
        return int(np.argmax(row))
    shifted = row.astype(np.float64)
    shifted = shifted - np.max(shifted)
    probs = np.exp(shifted)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("softmax is empty")
    pick = float(rng.random()) * total
    acc = 0.0
    last = len(probs) - 1
    for index in range(len(probs)):
        acc += float(probs[index])
        if pick <= acc:
            return int(index)
    return int(last)


def _encoder_layer(hidden, weights, index, nhead, pad_mask):
    prefix = "encoder.layers.%d." % index
    normed = _layer_norm(hidden, weights[prefix + "norm1.weight"], weights[prefix + "norm1.bias"])
    attended = _attention(
        normed,
        weights[prefix + "self_attn.in_proj_weight"],
        weights[prefix + "self_attn.in_proj_bias"],
        weights[prefix + "self_attn.out_proj.weight"],
        weights[prefix + "self_attn.out_proj.bias"],
        nhead,
        pad_mask,
    )
    hidden = hidden + attended
    normed = _layer_norm(hidden, weights[prefix + "norm2.weight"], weights[prefix + "norm2.bias"])
    ff = _gelu(_linear(normed, weights[prefix + "linear1.weight"], weights[prefix + "linear1.bias"]))
    ff = _linear(ff, weights[prefix + "linear2.weight"], weights[prefix + "linear2.bias"])
    return hidden + ff


def _attention(hidden, in_weight, in_bias, out_weight, out_bias, nhead, pad_mask):
    batch, seq, width = hidden.shape
    if width % nhead != 0:
        raise ValueError("d_model is not divisible by nhead")
    head_dim = width // nhead
    projected = _linear(hidden, in_weight, in_bias)
    query, key, value = np.split(projected, 3, axis=-1)

    def _heads(tensor):
        tensor = tensor.reshape(batch, seq, nhead, head_dim)
        return np.transpose(tensor, (0, 2, 1, 3))

    query = _heads(query) * np.float32(head_dim ** -0.5)
    key = _heads(key)
    value = _heads(value)
    scores = _matmul(query, np.transpose(key, (0, 1, 3, 2)))
    scores = np.where(pad_mask[:, None, None, :], np.float32(-np.inf), scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    context = _matmul(weights, value)
    context = np.transpose(context, (0, 2, 1, 3)).reshape(batch, seq, width)
    return _linear(context, out_weight, out_bias)


def _matmul(left, right):
    """Product of (..., M, K) and (..., K, N) without a BLAS GEMM.

    NumPy 1.19.5's OpenBLAS segfaults inside ``np.matmul`` and ``np.dot``
    when the address space is capped at 256MB, even if the thread counts
    were set to 1 before the import. ``einsum(..., optimize=False)`` stays
    off that path. ``OPENBLAS_CORETYPE`` did not.
    """

    left = np.asarray(left, dtype=np.float32)
    right = np.ascontiguousarray(right, dtype=np.float32)
    return np.einsum("...mk,...kn->...mn", left, right, optimize=False)


def _linear(inputs, weight, bias):
    return _matmul(inputs, weight.T) + bias


def _layer_norm(inputs, weight, bias):
    mean = inputs.mean(axis=-1, keepdims=True)
    var = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)
    normed = (inputs - mean) / np.sqrt(var + _LAYER_NORM_EPS)
    return normed * weight + bias


def _gelu(inputs):
    """erf-GELU. The rational erf is Abramowitz and Stegun 7.1.26.

    ``math.erf`` matches libm, and this approximation stays inside 1.5e-7,
    which is enough for the 1e-4 logit budget. It stays inside NumPy so a
    batch does not call back into the interpreter once per element.
    """

    values = inputs.astype(np.float64) / math.sqrt(2.0)
    sign = np.sign(values)
    absolute = np.abs(values)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t) - 0.284496736) * t + 0.254829592
    poly = poly * t
    approx = sign * (1.0 - poly * np.exp(-absolute * absolute))
    return (0.5 * inputs.astype(np.float64) * (1.0 + approx)).astype(np.float32)

def features_forward(weights, meta, batch):
    """Unmasked logits and value. Both float32."""

    arch = meta["arch"]
    nhead = int(arch["nhead"])
    layers = int(arch["num_layers"])
    token_ids = batch["token_ids"]
    type_ids = batch["type_ids"]
    numeric = batch["numeric"]
    pad_mask = np.asarray(batch["pad_mask"], dtype=np.bool_)
    # Padded keys never contribute to CLS; omit trailing padding computation.
    length = int(np.max(np.sum(~pad_mask, axis=1)))
    token_ids, type_ids, numeric, pad_mask = (x[:, :length] for x in (token_ids, type_ids, numeric, pad_mask))
    hidden = weights["token_emb.weight"][token_ids]
    hidden = hidden + weights["type_emb.weight"][type_ids]
    positions = np.arange(token_ids.shape[1])
    hidden = hidden + weights["pos_emb.weight"][positions][None, :, :]
    hidden = _linear(numeric, weights["numeric_proj.weight"], weights["numeric_proj.bias"]) + hidden
    for index in range(layers):
        hidden = _encoder_layer(hidden, weights, index, nhead, pad_mask)
    cls = hidden[:, 0, :]
    features = _gelu(_linear(cls, weights["trunk.0.weight"], weights["trunk.0.bias"]))
    return features

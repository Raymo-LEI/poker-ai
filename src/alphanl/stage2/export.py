"""Emit a self-contained NumPy-only Python agent, including weights and sources."""
import ast
import base64
import io
import json
from pathlib import Path
import numpy as np
from .train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]

def _definition(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.get_source_segment(source, node)


def export_agent(checkpoint, destination, *, safety=False):
    model, blob = load_checkpoint(checkpoint)
    config = model.config
    weights = {}
    for key, value in model.state_dict().items():
        if key.startswith("reference.backbone."):
            key = key.removeprefix("reference.backbone.")
        weights[key] = value.numpy()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **weights)
    base = ROOT / "alphanl"
    transformer = base / "transformer"
    sources = {}
    sources["schema"] = (transformer / "schema.py").read_text()
    sources["buckets"] = (transformer / "buckets.py").read_text()
    sources["arraytorch"] = '''import numpy as np
Tensor = np.ndarray
long = np.int64
float32 = np.float32
bool = np.bool_
def zeros(*shape, dtype=np.float32):
    return np.zeros(shape[0] if len(shape)==1 and isinstance(shape[0], tuple) else shape, dtype=dtype)
def ones(*shape, dtype=np.float32):
    return np.ones(shape[0] if len(shape)==1 and isinstance(shape[0], tuple) else shape, dtype=dtype)
def tensor(value, dtype=None):
    return np.array(value, dtype=dtype)
def from_numpy(value):
    return value
def cat(values, dim=0):
    return np.concatenate(values, axis=dim)
'''
    sources["tokenizer"] = (transformer / "tokenizer.py").read_text().replace("import torch", "from . import arraytorch as torch")
    sources["tables"] = 'from .schema import SUITS, RANKS\ndef standard_deck():\n    return tuple(s+r for s in SUITS for r in RANKS)\n'
    sources["portable_evaluator"] = (base / "stage2/portable_evaluator.py").read_text()
    equity = (transformer / "equity.py").read_text()
    equity = equity.replace('from phevaluator import evaluate_cards', 'from .portable_evaluator import evaluate_cards')
    equity = equity.replace('TABLE_PATH = Path(__file__).with_name("data") / "preflop_equity_v1.json"',
        '_TABLE_DATA = ' + repr(json.loads((transformer / "data/preflop_equity_v1.json").read_text())))
    equity = equity.replace('"hunl_random_equity_odds_v1:" + hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()', repr(config.numeric_version))
    equity = equity.replace('json.loads(TABLE_PATH.read_text())', '_TABLE_DATA')
    sources["equity"] = equity
    hunl = (transformer / "hunl.py").read_text()
    sources["hunl"] = ('from __future__ import annotations\nimport numpy as np\nfrom . import arraytorch as torch\n'
                       'from .tokenizer import BettingHistoryTokenizer\nfrom .schema import ActionEvent\n'
                       'NUMERIC_VERSION="hunl_exact_fraction_v1"\n' + _definition(hunl, "HunlTokenizer"))
    sources["context"] = (base / "stage2/context.py").read_text().replace('import torch', 'from . import arraytorch as torch')
    sources["actions"] = (base / "cfr/actions.py").read_text()
    repo = ROOT.parent
    sources["reconstruct"] = (repo / "botzone_bots/model/reconstruct.py").read_text()
    forward = (repo / "botzone_bots/model/forward.py").read_text()
    feature_function = _definition(forward, "raw_forward").replace('def raw_forward(', 'def features_forward(')
    feature_function = feature_function[:feature_function.index('    logits = _linear(features,')] + '    return features\n'
    feature_function = feature_function.replace('    hidden = weights["token_emb.weight"][token_ids]', '''    # Padded keys never contribute to CLS; omit trailing padding computation.
    length = int(np.max(np.sum(~pad_mask, axis=1)))
    token_ids, type_ids, numeric, pad_mask = (x[:, :length] for x in (token_ids, type_ids, numeric, pad_mask))
    hidden = weights["token_emb.weight"][token_ids]''')
    # Use single-thread BLAS in this runtime. Old 256MB deployment limits are a separate gate.
    forward = forward.replace('return np.einsum("...mk,...kn->...mn", left, right, optimize=False)', 'return np.matmul(left, right)')
    sources["forward"] = forward + '\n' + feature_function
    sources["runtime"] = ('WEIGHT_DATA = ' + repr(base64.b64encode(buffer.getvalue()).decode()) + '\nCONFIG = '
                          + repr(config.to_dict()) + '\nSETTINGS = ' + repr(model.settings)
                          + '\nSAFETY = ' + repr(safety) + '\n' + (base / "stage2/portable_runtime.py").read_text())
    # Modules are isolated from real alphanl/torch. Importing an export is safe in parity tests.
    bootstrap = '''# Generated Stage 2 agent. Requires NumPy only. Weights and sources embedded.
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import sys, types
_package = types.ModuleType("_stage2_export")
_package.__path__ = []
sys.modules[_package.__name__] = _package
'''
    output = [bootstrap]
    for name, source in sources.items():
        output.append(f'_module = types.ModuleType("_stage2_export.{name}")\n_module.__package__ = "_stage2_export"\nsys.modules[_module.__name__] = _module\nsetattr(_package, {name!r}, _module)\nexec(compile({source!r}, "<stage2:{name}>", "exec"), _module.__dict__)\n')
    output.append('Agent = _package.runtime.Agent\nrespond = _package.runtime.respond\nrequest_snapshot = _package.runtime.request_snapshot\nMatchCtx = _package.context.MatchCtx\nif __name__ == "__main__":\n    _package.runtime.run()\n')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(output))
    return {"path": str(destination), "bytes": destination.stat().st_size,
            "numeric_version": config.numeric_version, "reference_width": config.model.d_model,
            "runtime": "numpy_only_embedded_weights", "safety_intervention": safety}

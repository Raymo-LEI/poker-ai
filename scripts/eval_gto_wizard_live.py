"""Play a frozen Stage 2 policy against the GTO Wizard researcher API.

Smoke cap is 1000 hands. Nothing is submitted to the leaderboard, and a
state-changing act is not retried. ``GTOWIZARD_API_KEY`` is read from the
environment only. A missing key exits 2 and does not open a socket.

The weights stay on disk. ``--policy`` must be a Stage 2 residual checkpoint
such as ``u250_retained.pt``. ``--config`` is the sibling training config.
``model_structure/*.json`` is not read and is not used to build a network.

Example, 1000 hands:

  export GTOWIZARD_API_KEY=...
  python3 scripts/eval_gto_wizard_live.py \\
    --policy "$HOME/Documents/ChatGPT/Poker/botzone_native_v2_20261030/weights/u250_retained.pt" \\
    --config "$HOME/Documents/ChatGPT/Poker/botzone_native_v2_20261030/weights/config.json" \\
    --hands 1000 \\
    --out-dir runs/gto_wizard_live_u250
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.gto_wizard.adapter_hunl import AdapterError  # noqa: E402
from eval.gto_wizard.client import ApiError, GtoWizardClient  # noqa: E402
from eval.gto_wizard.live import MAX_HANDS, run_session  # noqa: E402
from eval.gto_wizard.policy import PolicyLoadError, Stage2Decider, load_frozen_policy  # noqa: E402


def _refuse(reason: str, *, detail: str | None = None, secret: str | None = None) -> int:
    body = {
        "status": "REFUSED",
        "reason": reason,
        "network_used": False,
        "leaderboard_submitted": False,
        "training_modified": False,
        "botzone_used": False,
        "live": False,
    }
    if detail:
        text = detail.replace(secret, "[redacted]") if secret else detail
        body["detail"] = text
    rendered = json.dumps(body)
    if secret:
        rendered = rendered.replace(secret, "[redacted]")
    print(rendered, file=sys.stderr)
    return 2


def main(argv=None, *, transport_factory=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=Path, required=True, help="frozen Stage 2 checkpoint, for example u250_retained.pt")
    parser.add_argument("--config", type=Path, required=True, help="sibling training config.json; not a model-structure file")
    parser.add_argument("--hands", type=int, required=True, help=f"hands to play, from 1 to {MAX_HANDS}")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for report.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    key = os.environ.get("GTOWIZARD_API_KEY")
    if key is None or not str(key).strip():
        return _refuse("missing_api_key")
    secret = str(key).strip()
    if isinstance(args.hands, bool) or not isinstance(args.hands, int) or not 1 <= args.hands <= MAX_HANDS:
        return _refuse("invalid_hands", secret=secret)
    try:
        model, _blob = load_frozen_policy(args.policy, args.config, device=args.device)
    except PolicyLoadError as exc:
        return _refuse("policy_refused", detail=str(exc), secret=secret)
    factory = transport_factory
    if factory is None:
        from eval.gto_wizard.client import RequestsTransport
        factory = RequestsTransport
    try:
        transport = factory(secret)
    except ApiError as exc:
        return _refuse("transport_refused", detail=str(exc), secret=secret)
    client = GtoWizardClient(transport, secret)
    import torch
    torch.manual_seed(args.seed)
    try:
        report, code = run_session(
            client,
            Stage2Decider(model),
            hands=args.hands,
            out_dir=args.out_dir,
            policy_name=args.policy.name,
            config_name=args.config.name,
            redact=secret,
        )
    except AdapterError as exc:
        return _refuse("session_refused", detail=str(exc), secret=secret)
    summary = {
        "status": "OK" if code == 0 else "STOPPED",
        "hands": report.get("hands"),
        "live": True,
        "leaderboard_submitted": False,
        "training_modified": False,
        "botzone_used": False,
        "network_used": report.get("network_used"),
        "stop_reason": report.get("stop_reason"),
        "chips": report.get("chips"),
        "aivat_bb_per_100": report.get("aivat_bb_per_100"),
        "score_claimed": False,
    }
    rendered = json.dumps(summary)
    rendered = rendered.replace(secret, "[redacted]")
    print(rendered, file=sys.stdout if code == 0 else sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

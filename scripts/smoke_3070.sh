#!/usr/bin/env bash
# T-A: 300-update AMP smoke for the step-3 Transformer on the RTX 3070 Laptop (8GB).
# Works in Git Bash, WSL and Linux. Run from the repo root with the venv active:
#
#   bash scripts/smoke_3070.sh                 # 300 steps, laptop_8gb, --amp
#   STEPS=10 bash scripts/smoke_3070.sh        # quick check
#   bash scripts/smoke_3070.sh --minibatch 128 # extra args go to smoke_metrics.py
#
# PowerShell equivalent (repo root, venv active):
#   $env:PYTHONPATH="src"; python scripts\smoke_metrics.py --config configs\transformer\laptop_8gb.json --steps 300 --log-every 50 --amp --out-dir runs\smoke_3070
#
# Expected runtime: ~7-9 s per update on the laptop (mostly CPU rollout), so
# 300 updates take ~35-45 min. Logs every 50 updates: steps/s, rollout vs
# update seconds, loss, torch.cuda.max_memory_allocated (GB), GradScaler scale.
# Output (fresh, separate from real runs): runs/smoke_3070/{metrics.csv,
# smoke_steps.csv, smoke_summary.csv, smoke_summary.json, smoke_summary.md, ckpt/}.
# Exit code 0 = PASS (no NaN, no OOM, peak < 7GB), 1 = FAIL.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${STEPS:-300}"
LOG_EVERY="${LOG_EVERY:-50}"
CONFIG="${CONFIG:-configs/transformer/laptop_8gb.json}"
OUT_DIR="${OUT_DIR:-runs/smoke_3070}"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"
"$PY" scripts/smoke_metrics.py --config "$CONFIG" --steps "$STEPS" --log-every "$LOG_EVERY" \
  --amp --out-dir "$OUT_DIR" "$@" 2>&1 | tee "$OUT_DIR/smoke.log"
exit "${PIPESTATUS[0]}"

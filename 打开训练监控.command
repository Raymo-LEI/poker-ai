#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
exec python3 scripts/monitor_training.py --open-browser "$@"

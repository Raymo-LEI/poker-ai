from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHANL_ROOT = REPO_ROOT / "third_party" / "AlphaNLHoldem"
DEFAULT_CONFIG = ALPHANL_ROOT / "confs" / "nl_holdem.py"
DEFAULT_WEIGHTS = ALPHANL_ROOT / "weights" / "c_1048.pkl"
BIG_BLIND_CHIPS = 2  # rlcard no-limit-holdem default (50bb stacks = 100 chips)

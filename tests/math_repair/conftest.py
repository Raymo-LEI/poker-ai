"""Exercise the portable modules without importing the optional Torch trainer."""
import sys
import types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
pkg = types.ModuleType('repair_runtime')
pkg.__path__ = [str(ROOT / 'src/alphanl/stage2')]
sys.modules['repair_runtime'] = pkg

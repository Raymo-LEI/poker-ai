"""History-tree preflop smoke generator, with separate 100BB and 200BB solves.

No class-index override, playability equity rescaling, or copied depth slot.
This is a finite-support sampled-checkdown game, not a production HUNL chart.
"""
import runpy
import sys
from pathlib import Path
if __name__=='__main__':
    sys.argv.append('--preflop')
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'math_repair/generate_labels.py'),run_name='__main__')

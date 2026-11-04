"""Migration entry for range-conditioned v1 labels; production training is pending.

Old random-range/river-target labels and 72:200 placeholder features are invalid.
This entry generates the small CPU acceptance set, never trains or overwrites.
Use --output NEW_PATH [--iterations N].
"""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'math_repair/generate_labels.py'),run_name='__main__')

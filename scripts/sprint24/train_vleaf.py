"""The unversioned Vleaf label path is retired. Generate v1 acceptance labels.

There is no compatible production value network yet. Range-conditioned labels
and enumerated turn chance/river betting replace realized-return bucket targets.
This entry performs CPU label generation only; --output must be a new path.
"""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'math_repair/generate_labels.py'),run_name='__main__')

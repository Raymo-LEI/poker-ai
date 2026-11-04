"""Evaluate supplied Botzone decision payloads through canonical respond().

The historical standard-HUNL match adapter used different minimums and bypassed
production lock state; its scores are not production-search acceptance evidence.
This migration entry no longer starts matches, calibrates beta, or packs releases.
"""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'math_repair/replay_decisions.py'),run_name='__main__')

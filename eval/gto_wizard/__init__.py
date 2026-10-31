"""Offline GTO Wizard benchmark scaffold.

Fixture comparison only. This package does not train, does not read strategy
weights, and does not submit anything to a leaderboard.
"""

from .compare import compare_distributions
from .schema import SCHEMA_A5, SCHEMA_A6

__all__ = ["SCHEMA_A5", "SCHEMA_A6", "compare_distributions"]

"""GTO Wizard benchmark helpers.

Importing this package does not train, read weights, or submit a leaderboard
row. ``eval.gto_wizard.live`` is the opt-in researcher-API runner.
"""

from .compare import compare_distributions
from .schema import SCHEMA_A5, SCHEMA_A6

__all__ = ["SCHEMA_A5", "SCHEMA_A6", "compare_distributions"]

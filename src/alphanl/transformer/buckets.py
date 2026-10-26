"""Pot-fraction buckets shared by the rlcard replay and the pokerkit table.

Bucket 0 is a zero-size action (check or fold). The remaining buckets are
right-closed intervals on ``chips_added / pot_before``.
"""

from __future__ import annotations

# Right edges of buckets 1..7. Anything larger is bucket 8.
BUCKET_RIGHT_EDGES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
NUM_BUCKETS = 1 + len(BUCKET_RIGHT_EDGES) + 1  # 0, the edges, and overflow


def pot_fraction_bucket(pot_fraction: float) -> int:
    if pot_fraction <= 0.0:
        return 0
    for index, edge in enumerate(BUCKET_RIGHT_EDGES, start=1):
        if pot_fraction <= edge:
            return index
    return NUM_BUCKETS - 1


def nearest_pot_fraction_bucket(pot_fraction: float) -> int:
    """Nearest existing positive bucket representative for off-tree HUNL bets.

    Bucket ids and vocabulary width are unchanged. A tie uses the smaller
    representative; a zero-chip action remains bucket 0. Overflow uses 8x pot.
    The continuous feature records the actual fraction independently.
    """

    if pot_fraction <= 0:
        return 0
    representatives = (*BUCKET_RIGHT_EDGES, 8.0)
    return 1 + min(range(len(representatives)), key=lambda i: abs(representatives[i] - pot_fraction))

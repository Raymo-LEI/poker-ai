"""Dependency-free 5-7 card comparator for portable equity sampling.

Returns a lexicographic strength tuple (larger is stronger). Used only for
comparisons; numeric phevaluator rank IDs are deliberately not reproduced.
"""
from collections import Counter

def strength(cards):
    ranks = ["23456789TJQKA".index(c[0]) + 2 for c in cards]
    groups = Counter(ranks)
    suits = Counter(c[1] for c in cards)
    def straight(values):
        values = set(values)
        if 14 in values:
            values.add(1)
        return next((high for high in range(14, 4, -1) if all(v in values for v in range(high-4, high+1))), 0)
    flush = next((s for s,n in suits.items() if n >= 5), None)
    flushed = sorted([r for r,c in zip(ranks,cards) if c[1] == flush], reverse=True) if flush else []
    sf = straight(flushed)
    if sf:
        return (8, sf)
    quads = sorted([r for r,n in groups.items() if n == 4], reverse=True)
    if quads:
        return (7, quads[0], max(r for r in ranks if r != quads[0]))
    trips = sorted([r for r,n in groups.items() if n >= 3], reverse=True)
    pairs = sorted([r for r,n in groups.items() if n >= 2], reverse=True)
    if trips and any(r != trips[0] for r in pairs):
        return (6, trips[0], next(r for r in pairs if r != trips[0]))
    if flushed:
        return (5, *flushed[:5])
    run = straight(ranks)
    if run:
        return (4, run)
    if trips:
        return (3, trips[0], *sorted([r for r in groups if r != trips[0]], reverse=True)[:2])
    if len(pairs) >= 2:
        return (2, *pairs[:2], max(r for r in groups if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *sorted([r for r in groups if r != pairs[0]], reverse=True)[:3])
    return (0, *sorted(groups, reverse=True)[:5])

def evaluate_cards(*cards):
    # sample_equity uses smaller-is-stronger phevaluator comparisons.
    return tuple(-x for x in strength(cards))

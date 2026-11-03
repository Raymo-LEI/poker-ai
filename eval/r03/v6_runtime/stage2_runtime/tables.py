from .schema import SUITS, RANKS
def standard_deck():
    return tuple(s+r for s in SUITS for r in RANKS)

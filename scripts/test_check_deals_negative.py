"""Negative test for eval_t_c1c_duplicate.py --check-deals: the checker must FAIL on corrupted deals.

    PYTHONPATH=src python scripts/test_check_deals_negative.py [--n 200] [--tables 16]

Cases (each corrupts the OLD-path records of exactly one deck k, new path untouched):
  clean          no corruption                       -> PASS expected
  board          one community card replaced         -> FAIL, line "deal k ... board"
  hero_seat      hero seat flipped                   -> FAIL, line "deal k ... hero seat"
  opponent_pick  opponent kind/policy id changed     -> FAIL, line "deal k ... opponent pick"
  hole           one opponent hole card replaced     -> FAIL, "deal k ... hole"
PASS iff every case matches its expectation, and each FAIL names deck k and only deck k.
"""
from __future__ import annotations

import argparse, contextlib, io, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_t_c1c_duplicate as D  # noqa: E402

UNIVERSE: set = set()  # every card text seen in the records (all 52 after a few dozen deals)


def _spare(used):
    return sorted(UNIVERSE - set(used))[0]


def corrupt(rec, field):
    d, tab, holes, board, hero, kind, pid = rec
    used = set(board) | set(holes[0]) | set(holes[1])
    if field == "board":
        board = board[:2] + (_spare(used),) + board[3:]
    elif field == "hole":
        h = list(holes[1]); h[0] = _spare(used); holes = (holes[0], tuple(h) if isinstance(holes[1], tuple) else h)
    elif field == "hero_seat":
        hero = 1 - hero
    elif field == "opponent_pick":
        kind, pid = ("random", None) if kind != "random" else ("self", None)
    return (d, tab, holes, board, hero, kind, pid)


def run_case(field, k, a):
    orig = D._deal_records

    def patched(path, n, tables, master_seed):
        recs = orig(path, n, tables, master_seed)
        for r in recs:
            UNIVERSE.update(r[3]); UNIVERSE.update(r[2][0]); UNIVERSE.update(r[2][1])
        if path == "old" and field != "clean":
            recs[k] = corrupt(recs[k], field)
        return recs

    D._deal_records = patched
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = D.check_deals(a.n, a.tables, a.master_seed)
    finally:
        D._deal_records = orig
    out = buf.getvalue()
    lines = [l for l in out.splitlines() if not l.startswith("[check-deals]")]
    key = {"board": "board", "hole": "hole", "hero_seat": "hero seat", "opponent_pick": "opponent pick"}.get(field)
    if field == "clean":
        ok = code == 0 and "PASS" in out
    else:
        ok = (code == 1 and "FAIL" in out and lines
              and all(l.startswith(f"deal {k} ") for l in lines) and any(key in l for l in lines))
    first = lines[0] if lines else out.strip().splitlines()[-1]
    print(f"[neg] {field:13s} deck {k}: exit={code} -> {'OK' if ok else 'WRONG'} | {first}", flush=True)
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tables", type=int, default=16)
    p.add_argument("--master-seed", type=int, default=12345)
    p.add_argument("--deck", type=int, default=37)
    a = p.parse_args()
    ok = True
    for field in ("clean", "board", "hole", "hero_seat", "opponent_pick"):
        ok &= run_case(field, a.deck, a)
    print(f"[neg] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

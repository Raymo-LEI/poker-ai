"""Frozen, visible-card-only style probes, not certified ladder professionals.

Keep these out of training pools unless a separate experiment opts in.
The preflop percentile weights all 1,326 combinations, not 169 classes.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np

from alphanl.transformer.equity import estimated_equity, preflop_key, preflop_table
from alphanl.transformer.schema import RANKS

PRO_OPPONENTS = ("TAG_Pro", "Trapper_CR")
PRO_VERSION = "pro_scripts_v1"
PRO_DEFAULTS = dict(open_raise=.65, open_continue=.35, defend_base=.30, defend_size=.055,
                    reraise=.88, cbet_dry=.75, cbet_wet=.40, trap_strong=.50, trap_draw=.25,
                    semi_raise=.25, semi_bet=.45, river_fold=.65, river_bluff=.08, equity_margin=.10)


@lru_cache(maxsize=1)
def preflop_percentiles():
    table = preflop_table()
    weights = {k: 6 if len(k) == 2 else 4 if k.endswith("s") else 12 for k in table}
    total = sum(weights.values())
    result, below = {}, 0
    for key in sorted(table, key=lambda k: (table[k], k)):
        result[key] = (below + weights[key] / 2) / total
        below += weights[key]
    return result


def texture(snapshot):
    """Detect made value and draws using only own cards and exposed board."""
    from phevaluator import evaluate_cards
    hole, board = snapshot.hole_cards, snapshot.board_cards
    ranks = lambda cards: {RANKS.index(c[1]) + 2 for c in cards}
    hr, br = ranks(hole), ranks(board)
    counts = Counter(c[1] for c in board)
    # A set requires a pocket pair; board trips alone never qualify.
    is_set = hole[0][1] == hole[1][1] and counts[hole[0][1]] >= 1
    top_two = len(hr) == 2 and len(br) >= 2 and hr == set(sorted(br)[-2:])
    convert = lambda cards: [c[1] + c[0].lower() for c in cards]
    own_score = evaluate_cards(*convert(hole + board))
    board_score = evaluate_cards(*convert(board)) if len(board) == 5 else 7463
    # Straight or better, with private-card improvement over a river board.
    strong = (own_score <= 1609 and own_score < board_score) or is_set or top_two
    suits = Counter(c[0] for c in hole + board)
    flush_draw = len(board) < 5 and any(n == 4 and any(c[0] == s for c in hole)
                                           for s, n in suits.items())
    all_ranks = hr | br
    if 14 in all_ranks:
        all_ranks = all_ranks | {1}
    board_ranks = br | ({1} if 14 in br else set())
    straight_out_ranks = set()
    for start in range(1, 11):
        window = set(range(start, start + 5))
        missing = window - all_ranks
        if len(missing) == 1 and len(window & board_ranks) < 4:
            straight_out_ranks |= missing
    draw = len(board) < 5 and (flush_draw or len(straight_out_ranks) >= 2)
    dry = (len(set(c[0] for c in board)) == len(board)
           and len(br) == len(board) and all(b - a > 2 for a, b in combinations(sorted(br), 2)))
    return dict(strong=strong, strong_draw=draw, dry=dry)


def raises(events):
    return [e for e in events if e.action_type in ("bet", "raise", "all_in")
            and e.chips_added > e.to_call_before]


class ProScriptActor:
    def __init__(self, name, *, seed, equity_samples=64):
        if name not in PRO_OPPONENTS or equity_samples < 1:
            raise ValueError("invalid pro script or equity sample count")
        self.name, self.rng = name, np.random.default_rng(seed)
        self.equity_samples = equity_samples
        self.params = dict(PRO_DEFAULTS)
        self.counts = Counter()
        self.begin_hand(0)

    def begin_hand(self, seat):
        self.trap_streets = set()
        self.trap_decided = set()

    def act(self, s):
        legal = s.legal_action_ids
        def prefer(*choices):
            return next(a for a in choices if a in legal)
        if s.street == 0:
            strength = preflop_percentiles()[preflop_key(s.hole_cards)]
            history = raises(s.events)
            if not history and s.me_seat == s.sb_seat:
                if strength >= self.params["open_raise"]:
                    self.counts["sb_open_raise"] += 1
                    return prefer(3, 2, 1, 0, 4)
                return prefer(1, 0, 2, 3, 4) if strength >= self.params["open_continue"] else prefer(0, 1, 2, 3, 4)
            size_bb = s.to_call / s.big_blind
            if not history:
                return prefer(3, 2, 1, 0, 4) if strength >= .8 else prefer(1, 0, 2, 3, 4)
            # Defend wider against a small open, tighter against large/multiple raises.
            threshold = min(.93, self.params["defend_base"] + self.params["defend_size"] * size_bb + .10 * max(0, len(history) - 1))
            suited_ace = s.hole_cards[0][0] == s.hole_cards[1][0] and any(c[1] == "A" for c in s.hole_cards)
            if strength >= (self.params["reraise"] if len(history) == 1 else .97) or (suited_ace and size_bb < 5 and self.rng.random() < .08):
                self.counts["preflop_reraise"] += 1
                return prefer(3, 2, 1, 0, 4)
            return prefer(1, 0, 2, 3, 4) if strength >= threshold or not s.to_call else prefer(0, 1, 2, 3, 4)

        features = texture(s)
        equity = estimated_equity(s, samples=self.equity_samples)
        odds = min(s.to_call, s.me_stack) / max(1., s.pot + min(s.to_call, s.me_stack))
        street = [e for e in s.events if e.street == s.street]
        aggression = raises(street)
        can_raise = any(a in legal for a in (2, 3))
        if self.name == "Trapper_CR":
            if not s.to_call and not street and s.street not in self.trap_decided:
                self.trap_decided.add(s.street)
                if features["strong"] and self.rng.random() < self.params["trap_strong"]:
                    self.trap_streets.add(s.street)
                    self.counts["trap_checks"] += 1
                    return prefer(1, 0, 2, 3, 4)
                if features["strong_draw"] and self.rng.random() < self.params["trap_draw"]:
                    self.trap_streets.add(s.street)
                    self.counts["draw_trap_checks"] += 1
                    return prefer(1, 0, 2, 3, 4)
            checked = any(e.actor_seat == s.me_seat and e.action_type == "check" for e in street)
            if s.to_call and checked and can_raise and s.street in self.trap_streets:
                self.trap_streets.discard(s.street)
                self.counts["check_raises"] += 1
                self.counts["semi_bluff_check_raises"] += int(not features["strong"])
                return prefer(3, 2, 1, 0, 4)
        if s.to_call:
            if features["strong"] and can_raise and len(aggression) < 3:
                return prefer(3, 2, 1, 0, 4)
            if features["strong_draw"] and can_raise and len(aggression) < 2 and self.rng.random() < self.params["semi_raise"]:
                self.counts["semi_bluff_raises"] += 1
                return prefer(2, 3, 1, 0, 4)
            # Uniform-card equity overvalues bluff catchers versus a strong river range.
            bet_ratio = s.to_call / max(1., s.pot - s.to_call)
            if s.street == 3 and bet_ratio >= .75 and .40 <= equity < .75 and not features["strong"]:
                self.counts["river_bluffcatch_opportunities"] += 1
                if self.rng.random() < self.params["river_fold"]:
                    self.counts["river_bluffcatch_folds"] += 1
                    return prefer(0, 1, 2, 3, 4)
            return prefer(1, 0, 2, 3, 4) if equity >= odds + self.params["equity_margin"] else prefer(0, 1, 2, 3, 4)
        if features["strong"] or equity > .76:
            return prefer(3, 2, 1, 0, 4)
        preflop = raises([e for e in s.events if e.street == 0])
        if s.street == 1 and not aggression and preflop and preflop[-1].actor_seat == s.me_seat:
            if self.rng.random() < (self.params["cbet_dry"] if features["dry"] else self.params["cbet_wet"]):
                self.counts["cbets"] += 1
                return prefer(2, 3, 1, 0, 4)
        if features["strong_draw"] and self.rng.random() < self.params["semi_bet"]:
            self.counts["semi_bluff_bets"] += 1
            return prefer(2, 3, 1, 0, 4)
        if s.street == 3 and equity < .40 and self.rng.random() < self.params["river_bluff"]:
            self.counts["river_bluffs"] += 1
            return prefer(2, 3, 1, 0, 4)
        return prefer(1, 0, 2, 3, 4)


class JitteredProScriptActor(ProScriptActor):
    """Training-only parameters drawn once and held fixed for a full match."""
    def __init__(self, name, *, seed, equity_samples=64):
        super().__init__(name, seed=seed, equity_samples=equity_samples)
        self.variant = "jittered_pro_v1"
        bounds = dict(open_raise=(.60,.70), open_continue=(.30,.40), defend_base=(.26,.34),
            defend_size=(.045,.065), reraise=(.84,.92), cbet_dry=(.65,.85), cbet_wet=(.30,.50),
            trap_strong=(.40,.60), trap_draw=(.18,.32), semi_raise=(.20,.30), semi_bet=(.35,.55),
            river_fold=(.55,.75), river_bluff=(.05,.11), equity_margin=(.075,.125))
        self.params = {k: float(self.rng.uniform(*v)) for k,v in bounds.items()}

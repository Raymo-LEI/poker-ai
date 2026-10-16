"""Differential fuzz: our Botzone legality against the official judge.

Primary oracle: ``botzone_referee_adapter.referee_is_legal`` (real judge).
Secondary: ``spec_is_legal``, labelled spec-based, not the referee.

Run:
    python3 eval/botzone_referee_diff.py --hands 20000 --out eval/botzone_referee_diff_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from botzone_referee_adapter import (  # noqa: E402
    JUDGE,
    RefereeState,
    judge_min_positive,
    new_dealt_holdem,
    referee_is_legal,
    spec_is_legal,
    spec_label,
    state_from_holdem,
)

from alphanl.botzone.actions import action_responses, legal_action_ids, response_for_action
from alphanl.botzone.legal import (
    HandState,
    legal_response,
    lower_bound_a,
    lower_bound_b,
    new_hand,
    safe_extra,
    to_call,
)
from alphanl.botzone.rule_bot import BotzoneRuleBot
from alphanl.botzone.table import BotzoneTable

_REPO = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO / "tests" / "fixtures" / "botzone_real.jsonl.gz"
_EXAMPLE_CAP = 24


def _load_env_table():
    """Import the env table without going through a second copy of this package."""

    name = "alphanl.transformer.botzone_table"
    if name in sys.modules:
        return sys.modules[name].BotzoneEnvTable
    # The transformer package imports torch. Callers that need the random-init
    # policy have torch; the legality-only paths do not call this.
    from alphanl.transformer.botzone_table import BotzoneEnvTable

    return BotzoneEnvTable


def _load_policy_tools():
    from alphanl.transformer.model import PokerTransformer
    from alphanl.transformer.tokenizer import BettingHistoryTokenizer

    return PokerTransformer, BettingHistoryTokenizer


class RandomInitPolicy:
    """Untrained 5-way head. Samples with the production mask and ``act``."""

    def __init__(self, seed: int = 0) -> None:
        import torch

        PokerTransformer, BettingHistoryTokenizer = _load_policy_tools()
        torch.manual_seed(seed)
        # Same 5-way head, mask, and orthogonal 0.01 policy init as training.
        # Depth is smaller than the laptop net so a 20k-hand sample stays a
        # sample of an untrained masked policy rather than a benchmark of the
        # encoder. Dropout is off.
        self.model = PokerTransformer(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_seq_len=64,
        )
        self.model.eval()
        self.tokenizer = BettingHistoryTokenizer(max_seq_len=64)
        self.table_cls = _load_env_table()

    def choose(self, table) -> int:
        seat = int(table.current_player())
        batch = self.tokenizer.encode(table.snapshot(seat))
        action, _logp, _value, _ent = self.model.act(batch)
        return int(action.item())


def _chips_put(state: RefereeState, response: int) -> int:
    """Chips the actor commits. Mirrors the judge branches for bookkeeping."""

    if response == -1:
        return 0
    if response == -2:
        return int(state.actor_chips)
    if response == 0:
        return int(state.round_bet) - int(state.own_round_bet)
    return int(response)


def _our_hand(state: RefereeState, bound_b: int) -> HandState:
    """Our ``HandState`` at a judge node, using our bound tracker."""

    commits = []
    all_in = []
    folded = []
    for seat in (0, 1):
        bet = int(state.round_player_bet[seat])
        folded.append(bet == -1)
        all_in.append(bet == -2)
        commits.append(0 if bet < 0 else bet)
    return HandState(
        dealer=int(state.dealer),
        street=int(state.street),
        stacks=(int(state.chips[0]), int(state.chips[1])),
        street_commit=(commits[0], commits[1]),
        pot=int(state.pot),
        bound_b=int(bound_b),
        actor=int(state.actor),
        all_in=(all_in[0], all_in[1]),
        folded=(folded[0], folded[1]),
        needs_action=(True, True),
        hand_over=False,
    )


def _next_bound(bound_b: int, state: RefereeState, response: int) -> int:
    """``legal.apply``'s bound update, including sizes our gate would reject.

    A positive bet or an all-in updates the bound only when the new street
    commitment is strictly above the previous maximum. That is lines 215-219
    of ``legal.py``. The judge updates ``round_raise`` on every accepted
    positive bet, which is a wider condition; this function stays with ours.
    """

    if response == -1 or int(state.round_bet) < 0:
        return int(bound_b)
    chips = _chips_put(state, response)
    prev = []
    for bet in state.round_player_bet:
        prev.append(0 if int(bet) < 0 else int(bet))
    prev_max = max(prev) if prev else 0
    actor = int(state.actor)
    if prev[actor] + chips > prev_max:
        return max(int(bound_b), 2 * chips)
    return int(bound_b)


def _street_reset_bound(game) -> int:
    """Our postflop bound is 100. The judge resets ``round_raise`` to 50."""

    return 100 if int(game.round) > 0 else 200


def _repro(state: RefereeState, response, *, our_safe, judge_min, note: str) -> Dict[str, Any]:
    return {
        "note": note,
        "dealer": state.dealer,
        "street": state.street,
        "actor": state.actor,
        "chips": list(state.chips),
        "round_bet": state.round_bet,
        "round_raise": state.round_raise,
        "round_player_bet": list(state.round_player_bet),
        "pot": state.pot,
        "response": response,
        "our_safe": our_safe,
        "judge_min_positive": judge_min,
        "to_call": None
        if state.round_bet < 0 or state.own_round_bet < 0
        else int(state.round_bet) - int(state.own_round_bet),
    }


def _push(bucket: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    if len(bucket) < _EXAMPLE_CAP:
        bucket.append(item)


def compare_decision(
    state: RefereeState,
    our: HandState,
    *,
    illegal_examples: List[Dict[str, Any]],
    conservative_examples: List[Dict[str, Any]],
    spec_examples: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Compare our mask and the two mapped raises to the live judge."""

    counts = {
        "decisions": 1,
        "we_legal_judge_illegal": 0,
        "we_illegal_judge_legal_responses": 0,
        "conservative_decisions": 0,
        "mapped_illegal": 0,
        "spec_action_amount_disagree": 0,
        "spec_street_total_disagree": 0,
        "gap_sum": 0,
        "gap_max": 0,
        "gap_decisions": 0,
    }
    judge_min = judge_min_positive(state)
    our_safe = safe_extra(our)
    our_legal = legal_response(our)

    # legal_response's positive region is the interval [safe, chips). The
    # sentinels and the two endpoints are enough to know the whole interval.
    probes = {response for response in (-2, -1, 0) if response in our_legal}
    if our_safe is not None:
        probes.add(int(our_safe))
        top = int(our.stacks[our.actor]) - 1
        if top >= int(our_safe):
            probes.add(top)
    for response in sorted(probes):
        if not referee_is_legal(state, response):
            counts["we_legal_judge_illegal"] += 1
            _push(
                illegal_examples,
                _repro(state, response, our_safe=our_safe, judge_min=judge_min, note="we legal, judge illegal"),
            )

    mapped = action_responses(our)
    for name, response in (("raise_min", mapped[2]), ("raise_large", mapped[3])):
        if response is None:
            continue
        if response not in our_legal or not referee_is_legal(state, int(response)):
            counts["mapped_illegal"] += 1
            _push(
                illegal_examples,
                _repro(
                    state,
                    response,
                    our_safe=our_safe,
                    judge_min=judge_min,
                    note=f"mapped {name} rejected",
                ),
            )

    missing_count, smallest = _missing_span(state, our, our_legal, judge_min, our_safe)
    if missing_count:
        counts["conservative_decisions"] += 1
        counts["we_illegal_judge_legal_responses"] += missing_count
        if our_safe is not None and judge_min is not None:
            gap = int(our_safe) - int(judge_min)
        elif judge_min is not None:
            gap = int(our.stacks[our.actor]) - int(judge_min)
        else:
            gap = 0
        if gap > 0:
            counts["gap_decisions"] += 1
            counts["gap_sum"] += gap
            counts["gap_max"] = max(counts["gap_max"], gap)
        _push(
            conservative_examples,
            _repro(
                state,
                smallest,
                our_safe=our_safe,
                judge_min=judge_min,
                note=f"judge allows {missing_count} responses we reject; smallest {smallest}",
            ),
        )

    spec_probes = set(probes)
    if judge_min is not None:
        spec_probes.add(int(judge_min))
        if int(judge_min) > 1:
            spec_probes.add(int(judge_min) - 1)
    for mode, key in (
        ("action_amount", "spec_action_amount_disagree"),
        ("street_total", "spec_street_total_disagree"),
    ):
        for response in sorted(spec_probes):
            spec_ok = spec_is_legal(state, response, mode)
            judge_ok = referee_is_legal(state, response)
            if spec_ok != judge_ok:
                counts[key] += 1
                _push(
                    spec_examples,
                    _repro(
                        state,
                        response,
                        our_safe=our_safe,
                        judge_min=judge_min,
                        note=f"{spec_label()} mode={mode} spec={spec_ok} judge={judge_ok}",
                    ),
                )
                break
    return counts


def _missing_span(
    state: RefereeState,
    our: HandState,
    our_legal: set,
    judge_min: Optional[int],
    our_safe: Optional[int],
) -> Tuple[int, Optional[int]]:
    """How many judge-legal ints we reject, and the smallest such int.

    Positives are an interval. The count is the length of that interval.
    """

    count = 0
    smallest: Optional[int] = None
    for response in (-2, -1, 0):
        if response not in our_legal and referee_is_legal(state, response):
            count += 1
            smallest = response if smallest is None else smallest
    if judge_min is None:
        return count, smallest
    chips = int(our.stacks[our.actor])
    hi = chips if our_safe is None else min(int(our_safe), chips)
    if int(judge_min) < hi:
        count += hi - int(judge_min)
        if smallest is None:
            smallest = int(judge_min)
    return count, smallest


def _add_counts(total: Dict[str, int], part: Dict[str, int]) -> None:
    for key, value in part.items():
        if key == "gap_max":
            total[key] = max(int(total.get(key, 0)), int(value))
        else:
            total[key] = int(total.get(key, 0)) + int(value)


def _sync_or_record(
    our: HandState,
    game,
    desyncs: List[Dict[str, Any]],
) -> None:
    """Betting fields that must agree while the street is still numeric."""

    if our.hand_over:
        return
    problems = []
    if int(our.actor) != int(game.round_idx):
        problems.append(f"actor {our.actor} != {game.round_idx}")
    if int(our.street) != int(game.round):
        problems.append(f"street {our.street} != {game.round}")
    if int(our.pot) != int(game.pot):
        problems.append(f"pot {our.pot} != {game.pot}")
    if [int(x) for x in our.stacks] != [int(x) for x in game.player_chips]:
        problems.append(f"stacks {our.stacks} != {list(game.player_chips)}")
    if int(game.round_bet) >= 0:
        for seat in (0, 1):
            bet = int(game.round_player_bet[seat])
            if bet >= 0 and int(our.street_commit[seat]) != bet:
                problems.append(f"commit[{seat}] {our.street_commit[seat]} != {bet}")
        if int(our.bound_b) // 2 != int(game.round_raise):
            problems.append(f"bound_b/2 {our.bound_b // 2} != round_raise {game.round_raise}")
    if problems and len(desyncs) < _EXAMPLE_CAP:
        desyncs.append(
            {
                "problems": problems,
                "dealer": our.dealer,
                "street": our.street,
                "actor": our.actor,
                "commits": list(our.street_commit),
                "bound_b": our.bound_b,
                "judge_round_bet": int(game.round_bet),
                "judge_round_raise": int(game.round_raise),
                "judge_rpb": list(game.round_player_bet),
            }
        )


def _play_our_hand(
    *,
    opponent: str,
    rng: random.Random,
    table: BotzoneTable,
    env_table,
    rule_bot: BotzoneRuleBot,
    policy: Optional[RandomInitPolicy],
    dealer: int,
    illegal_examples,
    conservative_examples,
    spec_examples,
    desyncs,
) -> Dict[str, int]:
    if opponent == "policy":
        if env_table is None or policy is None:
            raise RuntimeError("policy opponent requires the env table and an untrained net")
        env_table.reset()
        pure = env_table._table
        state = pure.state
    else:
        pure = table
        state = table.start_hand(dealer=dealer)
    game = new_dealt_holdem(int(pure.dealer))
    totals = {"hands": 1, "desyncs": 0}
    guard = 0
    while state is not None and not state.hand_over and guard < 80:
        guard += 1
        _sync_or_record(state, game, desyncs)
        judge_state = state_from_holdem(game)
        part = compare_decision(
            judge_state,
            state,
            illegal_examples=illegal_examples,
            conservative_examples=conservative_examples,
            spec_examples=spec_examples,
        )
        _add_counts(totals, part)
        if opponent == "policy":
            action_id = policy.choose(env_table)
        elif opponent == "rule":
            action_id = rule_bot.action_id(pure)
        else:
            ids = legal_action_ids(state)
            action_id = ids[rng.randrange(len(ids))]
        response = response_for_action(state, action_id)
        if response is None or not referee_is_legal(judge_state, int(response)):
            totals["we_legal_judge_illegal"] = int(totals.get("we_legal_judge_illegal", 0)) + 1
            _push(
                illegal_examples,
                _repro(
                    judge_state,
                    response,
                    our_safe=safe_extra(state),
                    judge_min=judge_min_positive(judge_state),
                    note=f"sent action {action_id} from {opponent}",
                ),
            )
            break
        if opponent == "policy":
            env_table.step(action_id)
            state = env_table._table.state
        else:
            state = pure.act(int(response))
        try:
            game.player_action(int(response))
        except ValueError as exc:
            totals["we_legal_judge_illegal"] = int(totals.get("we_legal_judge_illegal", 0)) + 1
            _push(
                illegal_examples,
                _repro(judge_state, response, our_safe=None, judge_min=None, note=f"judge rejected sent bet {exc}"),
            )
            break
    return totals


def _play_judge_exploration(
    rng: random.Random,
    dealer: int,
    illegal_examples,
    conservative_examples,
    spec_examples,
    undercall_examples,
) -> Dict[str, int]:
    """Hands the judge can reach, including raises below our minimum."""

    game = new_dealt_holdem(dealer, deck=_shuffled(rng))
    bound_b = 200
    totals = {"hands": 1, "undercall_accepts": 0}
    guard = 0
    while guard < 80:
        guard += 1
        if int(game.round_action_left) <= 0 and _street_matched(game):
            break
        state = state_from_holdem(game)
        our = _our_hand(state, bound_b)
        if our.stacks[our.actor] <= 0:
            break
        part = compare_decision(
            state,
            our,
            illegal_examples=illegal_examples,
            conservative_examples=conservative_examples,
            spec_examples=spec_examples,
        )
        _add_counts(totals, part)
        response = _sample_judge_bet(rng, state, undercall_examples, totals)
        if response is None:
            break
        street_before = int(game.round)
        try:
            result = game.player_action(int(response))
        except ValueError:
            result = game.player_action(-1)
            response = -1
        if int(game.round) != street_before:
            bound_b = 100
        else:
            bound_b = _next_bound(bound_b, state, int(response))
        if result:
            break
    return totals


def _street_matched(game) -> bool:
    live = [int(bet) for bet in game.round_player_bet if int(bet) != -1]
    return bool(live) and all(bet == int(game.round_bet) for bet in live)


def _shuffled(rng: random.Random) -> Tuple[int, ...]:
    cards = list(range(52))
    rng.shuffle(cards)
    return tuple(cards)


def _sample_judge_bet(rng, state: RefereeState, undercall_examples, totals) -> Optional[int]:
    chips = int(state.actor_chips)
    if chips <= 0:
        return None
    candidates = []
    for response in (-1, -2, 0):
        if referee_is_legal(state, response):
            candidates.append(response)
    judge_min = judge_min_positive(state)
    if judge_min is not None and judge_min < chips:
        candidates.append(int(judge_min))
        if judge_min + 1 < chips:
            candidates.append(rng.randrange(int(judge_min), chips))
        # Explicit under-call probe: the min positive, even if it does not cover.
        own = int(state.own_round_bet)
        if state.round_bet >= 0 and int(judge_min) + own < int(state.round_bet):
            totals["undercall_accepts"] = int(totals.get("undercall_accepts", 0)) + 1
            _push(
                undercall_examples,
                _repro(
                    state,
                    int(judge_min),
                    our_safe=None,
                    judge_min=int(judge_min),
                    note="accepted raise whose total is below the current street bet",
                ),
            )
    if not candidates:
        return -1
    return int(rng.choice(candidates))


def run_fuzz(
    n_hands: int,
    seed: int = 0,
    *,
    explore_hands: Optional[int] = None,
    use_policy: bool = True,
) -> Dict[str, Any]:
    """Play ``n_hands`` on our action set, plus judge-exploration hands.

    Opponents rotate random-legal, the M7 rule bot, and an untrained policy.
    Exploration hands sample any judge-legal bet, including sizes we reject.
    """

    if n_hands < 1:
        raise ValueError(f"n_hands must be positive, got {n_hands}")
    explore = n_hands // 4 if explore_hands is None else int(explore_hands)
    rng = random.Random(seed)
    rule_bot = BotzoneRuleBot()
    policy = RandomInitPolicy(seed) if use_policy else None
    table = BotzoneTable(rng=random.Random(seed + 1), dealer=0)
    env_table = None
    if policy is not None:
        env_table = policy.table_cls(seed=seed + 2, dealer=0)
    illegal_examples: List[Dict[str, Any]] = []
    conservative_examples: List[Dict[str, Any]] = []
    spec_examples: List[Dict[str, Any]] = []
    undercall_examples: List[Dict[str, Any]] = []
    desyncs: List[Dict[str, Any]] = []
    totals: Dict[str, int] = {}
    names = ("random", "rule", "policy") if use_policy else ("random", "rule")
    for index in range(n_hands):
        opponent = names[index % len(names)]
        part = _play_our_hand(
            opponent=opponent,
            rng=rng,
            table=table,
            env_table=env_table if opponent == "policy" else None,
            rule_bot=rule_bot,
            policy=policy,
            dealer=index % 2,
            illegal_examples=illegal_examples,
            conservative_examples=conservative_examples,
            spec_examples=spec_examples,
            desyncs=desyncs,
        )
        _add_counts(totals, part)
    for index in range(explore):
        part = _play_judge_exploration(
            rng,
            index % 2,
            illegal_examples,
            conservative_examples,
            spec_examples,
            undercall_examples,
        )
        _add_counts(totals, part)
    gap_n = int(totals.get("gap_decisions", 0))
    summary = {
        "oracle": "tests/fixtures/botzone_judge.py Holdem.player_action",
        "spec_reference": spec_label(),
        "hands_our_action_set": n_hands,
        "hands_judge_exploration": explore,
        "opponents": list(names),
        "seed": seed,
        "decisions": int(totals.get("decisions", 0)),
        "we_legal_judge_illegal": int(totals.get("we_legal_judge_illegal", 0)),
        "we_illegal_judge_legal_responses": int(totals.get("we_illegal_judge_legal_responses", 0)),
        "conservative_decisions": int(totals.get("conservative_decisions", 0)),
        "mapped_raise_illegal": int(totals.get("mapped_illegal", 0)),
        "raise_min_and_raise_large_always_judge_legal": int(totals.get("mapped_illegal", 0)) == 0
        and int(totals.get("we_legal_judge_illegal", 0)) == 0,
        "gap_decisions": gap_n,
        "gap_mean": (int(totals.get("gap_sum", 0)) / gap_n) if gap_n else 0,
        "gap_max": int(totals.get("gap_max", 0)),
        "spec_action_amount_disagree": int(totals.get("spec_action_amount_disagree", 0)),
        "spec_street_total_disagree": int(totals.get("spec_street_total_disagree", 0)),
        "undercall_accepts": int(totals.get("undercall_accepts", 0)),
        "desyncs": len(desyncs),
        "examples_we_legal_judge_illegal": illegal_examples,
        "examples_we_illegal_judge_legal": conservative_examples,
        "examples_spec_disagreement": spec_examples,
        "examples_undercall": undercall_examples,
        "examples_desync": desyncs,
    }
    return summary


def scenario_tables() -> Dict[str, List[Dict[str, Any]]]:
    """Numeric mins for the three street shapes, taken from the live judge."""

    rows: Dict[str, List[Dict[str, Any]]] = {"preflop": [], "postflop": [], "reraise": []}

    def record(bucket: str, game, bound_b: int, label: str) -> None:
        state = state_from_holdem(game)
        our = _our_hand(state, bound_b)
        judge_min = judge_min_positive(state)
        our_safe = safe_extra(our)
        rows[bucket].append(
            {
                "label": label,
                "actor": state.actor,
                "own": state.own_round_bet,
                "round_raise": state.round_raise,
                "round_bet": state.round_bet,
                "judge_min": judge_min,
                "our_safe": our_safe,
                "our_bound_a": lower_bound_a(our),
                "our_bound_b": lower_bound_b(our),
                "gap": None if our_safe is None or judge_min is None else int(our_safe) - int(judge_min),
                "to_call": to_call(our),
            }
        )

    game = new_dealt_holdem(0)
    bound = 200
    record("preflop", game, bound, "SB first after blinds 50/100")
    game.player_action(0)
    record("preflop", game, bound, "BB after SB calls")
    opener = new_dealt_holdem(0)
    pre = state_from_holdem(opener)
    opener.player_action(150)
    bound = _next_bound(200, pre, 150)
    record("preflop", opener, bound, "BB after SB raises 150 (judge-legal, we reject 150)")

    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    bound = 100
    record("postflop", game, bound, "first postflop bet, round_raise reset")
    pre = state_from_holdem(game)
    game.player_action(100)
    bound = _next_bound(100, pre, 100)
    record("postflop", game, bound, "facing a 100 bet")
    pre = state_from_holdem(game)
    game.player_action(200)
    bound = _next_bound(bound, pre, 200)
    record("postflop", game, bound, "original bettor facing a 200 raise")

    game = new_dealt_holdem(0)
    game.player_action(0)
    game.player_action(0)
    bound = 100
    pre = state_from_holdem(game)
    game.player_action(100)
    bound = _next_bound(bound, pre, 100)
    pre = state_from_holdem(game)
    game.player_action(300)
    bound = _next_bound(bound, pre, 300)
    record("reraise", game, bound, "A bet 100, B raised by 300; A to act (owner example)")

    # Consecutive 1-chip raises that leave round_raise far below the street total.
    game = new_dealt_holdem(0)
    script = (0, 100, 101, 102)
    bound = 200
    for bet in script:
        pre = state_from_holdem(game)
        game.player_action(bet)
        if int(game.round) == int(pre.street):
            bound = _next_bound(bound, pre, bet)
        else:
            bound = 100
    record("reraise", game, bound, "after calls/raises 0,100,101,102; min positive is an under-call")
    return rows


def undercall_sequence() -> RefereeState:
    """Minimal judge state where a legal raise does not cover the street bet."""

    game = new_dealt_holdem(0)
    for bet in (0, 100, 101, 102):
        game.player_action(bet)
    return state_from_holdem(game)


def replay_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Replay the 41-match log. Order and judge acceptance must hold.

    A real bet our ``legal_response`` rejects is a conservative mismatch.
    It is counted, not treated as the judge rejecting us.
    """

    import gzip

    fixture = Path(path) if path is not None else _FIXTURE
    hands = []
    with gzip.open(fixture, "rt") as handle:
        for line in handle:
            hands.append(json.loads(line))
    order_bad = []
    judge_rejected = []
    our_rejected = []
    mapped_bad = []
    decisions = 0
    order_mismatches = 0
    judge_rejected_total = 0
    our_rejected_total = 0
    mapped_illegal = 0
    for hand in hands:
        dealer = int(hand["d"])
        game = new_dealt_holdem(dealer)
        bound_b = 200
        for index, (pid, response, street) in enumerate(hand["a"]):
            decisions += 1
            state = state_from_holdem(game)
            if int(game.round_idx) != int(pid) or int(game.round) != int(street):
                order_mismatches += 1
                if len(order_bad) < _EXAMPLE_CAP:
                    order_bad.append(
                        {
                            "match": hand["m"],
                            "hand": hand["h"],
                            "index": index,
                            "judge_actor": int(game.round_idx),
                            "judge_street": int(game.round),
                            "fixture_actor": int(pid),
                            "fixture_street": int(street),
                        }
                    )
                break
            our = _our_hand(state, bound_b)
            if int(our.actor) != int(pid) or int(our.street) != int(street):
                order_mismatches += 1
                if len(order_bad) < _EXAMPLE_CAP:
                    order_bad.append(
                        {
                            "match": hand["m"],
                            "hand": hand["h"],
                            "index": index,
                            "our_actor": int(our.actor),
                            "our_street": int(our.street),
                            "fixture_actor": int(pid),
                            "fixture_street": int(street),
                        }
                    )
                break
            if not referee_is_legal(state, int(response)):
                judge_rejected_total += 1
                if len(judge_rejected) < _EXAMPLE_CAP:
                    judge_rejected.append(
                        {"match": hand["m"], "hand": hand["h"], "index": index, "response": int(response)}
                    )
            if int(response) not in legal_response(our):
                our_rejected_total += 1
                if len(our_rejected) < _EXAMPLE_CAP:
                    our_rejected.append(
                        {
                            "match": hand["m"],
                            "hand": hand["h"],
                            "index": index,
                            "actor": int(pid),
                            "street": int(street),
                            "response": int(response),
                            "our_safe": safe_extra(our),
                            "judge_min": judge_min_positive(state),
                            "own": state.own_round_bet,
                            "round_raise": state.round_raise,
                            "round_bet": state.round_bet,
                            "to_call": to_call(our),
                        }
                    )
            for mapped in action_responses(our):
                if mapped is None:
                    continue
                if not referee_is_legal(state, int(mapped)):
                    mapped_illegal += 1
                    if len(mapped_bad) < _EXAMPLE_CAP:
                        mapped_bad.append(
                            {
                                "match": hand["m"],
                                "hand": hand["h"],
                                "index": index,
                                "response": int(mapped),
                            }
                        )
                    break
            pre = state
            street_before = int(game.round)
            game.player_action(int(response))
            if int(game.round) != street_before:
                bound_b = 100
            else:
                bound_b = _next_bound(bound_b, pre, int(response))
    try:
        fixture_name = str(fixture.resolve().relative_to(_REPO))
    except ValueError:
        fixture_name = str(fixture)
    return {
        "fixture": fixture_name,
        "hands": len(hands),
        "matches": len({hand["m"] for hand in hands}),
        "decisions": decisions,
        "order_mismatches": order_mismatches,
        "order_examples": order_bad,
        "judge_rejected_real_actions": judge_rejected_total,
        "our_rejected_real_actions": our_rejected_total,
        "our_rejected_examples": our_rejected,
        "mapped_illegal": mapped_illegal,
        "mapped_examples": mapped_bad,
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Botzone referee differential fuzz")
    parser.add_argument("--hands", type=int, default=20000)
    parser.add_argument("--explore-hands", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--skip-fixture", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args(argv)
    summary = run_fuzz(
        args.hands,
        seed=args.seed,
        explore_hands=args.explore_hands,
        use_policy=not args.skip_policy,
    )
    summary["scenarios"] = scenario_tables()
    summary["undercall_minimal"] = _repro(
        undercall_sequence(),
        3,
        our_safe=None,
        judge_min=3,
        note="scripted preflop 0,100,101,102 then bet 3",
    )
    if not args.skip_fixture:
        summary["fixture_replay"] = replay_fixture()
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


if __name__ == "__main__":
    main()

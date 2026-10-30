"""Step-level diff of the branchable kernel against the official referee.

The kernel's transitions are ``Holdem.player_action``. This harness still
drives a second, independent ``Holdem`` with the same decks and responses
and compares actor, street, pot, stacks, board, holes, remaining deck,
history, legality, showdown payoffs, chip conservation, and 50-hand scores.

The 33-match Mac tree is not on this machine. Whole-hand replay uses
``tests/fixtures/botzone_real.jsonl.gz`` (41 matches / 2050 hands), the
corpus already loaded by ``tests/test_botzone_referee_diff.py``.

Run:
    python3 eval/botzone_kernel_diff.py --out eval/r02/engine_diff_report.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "eval"))

from botzone_referee_adapter import JUDGE, judge_min_positive, state_from_holdem  # noqa: E402

from alphanl.botzone.kernel import BranchableKernel, LegacyFiveActionBridge  # noqa: E402
from alphanl.botzone.legal import BIG_BLIND, STARTING_STACK  # noqa: E402

_FIXTURE = _REPO / "tests" / "fixtures" / "botzone_real.jsonl.gz"
_JUDGE = _REPO / "tests" / "fixtures" / "botzone_judge.py"
_JUDGE_SHA = "b70702fdcd28333f91087f1a3dc3ede1ca59737404622c64dc6dbf7e9206e544"
_EXAMPLE_CAP = 8

_JUDGE_FIELDS = (
    "chips",
    "pot",
    "dealer",
    "street",
    "actor",
    "round_bet",
    "round_raise",
    "round_player_bet",
    "round_action_left",
    "public_cards",
    "holes",
    "deck",
    "history",
)


class KernelDiffError(RuntimeError):
    def __init__(self, example: Dict[str, Any]) -> None:
        self.example = example
        super().__init__(json.dumps(example, default=str)[:4000])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_counters() -> Dict[str, int]:
    keys = (
        "hands",
        "matches",
        "decisions",
        "showdowns",
        "folds",
        "step_mismatches",
        "legal_reject_mismatches",
        "legal_reject_checks",
        "payoff_mismatches",
        "next_actor_mismatches",
        "pot_mismatches",
        "stack_mismatches",
        "street_mismatches",
        "showdown_payoff_mismatches",
        "conservation_failures",
        "policy_view_leaks",
        "policy_view_checks",
        "legacy_bridge_mismatches",
        "legacy_bridge_checks",
        "legacy_id_differs_from_chip_response",
        "legacy_id_equals_chip_response",
        "match_score_mismatches",
        "clone_checks",
        "clone_isolation_failures",
        "min_positive_mismatches",
        "states",
        "probe_count",
    )
    return {key: 0 for key in keys}


def deal_shadow(deck: Sequence[int], dealer: int):
    """Official constructor plus the official deal. Global RNG is restored."""

    state = random.getstate()
    try:
        game = JUDGE.Holdem([STARTING_STACK, STARTING_STACK], dealer_idx=int(dealer))
        game.set_deck_array(list(deck))
        game.deal_cards_and_blind()
        return game
    finally:
        random.setstate(state)


def shadow_view(game) -> Dict[str, Any]:
    return {
        "chips": (int(game.player_chips[0]), int(game.player_chips[1])),
        "pot": int(game.pot),
        "dealer": int(game.dealer_idx),
        "street": int(game.round),
        "actor": int(game.round_idx),
        "round_bet": int(game.round_bet),
        "round_raise": int(game.round_raise),
        "round_player_bet": (int(game.round_player_bet[0]), int(game.round_player_bet[1])),
        "round_action_left": int(game.round_action_left),
        "public_cards": tuple(int(card.to_int()) for card in game.public_cards),
        "holes": (
            tuple(int(card.to_int()) for card in game.player_cards[0]),
            tuple(int(card.to_int()) for card in game.player_cards[1]),
        ),
        "deck": tuple(int(card.to_int()) for card in game.deck),
        "history": [dict(item) for item in game.history],
    }


def _independent_clone(game):
    """Field copy that does not call the kernel's cloner."""

    import copy

    return copy.deepcopy(game)


def shadow_accepts(game, response) -> bool:
    parsed = JUDGE.parse_bot_response(response)
    if parsed is None:
        return False
    probe = _independent_clone(game)
    try:
        probe.player_action(parsed)
    except ValueError:
        return False
    return True


def _note(totals: Dict[str, int], key: str, amount: int = 1) -> None:
    totals[key] = int(totals.get(key, 0)) + amount


def _push(bucket: List[Dict[str, Any]], item: Dict[str, Any]) -> None:
    if len(bucket) < _EXAMPLE_CAP:
        bucket.append(item)


def _fail_or_push(
    examples: List[Dict[str, Any]],
    example: Dict[str, Any],
    *,
    raise_now: bool,
) -> None:
    _push(examples, example)
    if raise_now:
        raise KernelDiffError(example)


def judge_field_diffs(audit: Dict[str, Any], shadow: Dict[str, Any]) -> List[str]:
    return [field for field in _JUDGE_FIELDS if audit[field] != shadow[field]]


def deck_from_partial(holes: Sequence[Sequence[int]], board: Sequence[int]) -> List[int]:
    """Judge pop-order deck whose dealt prefix is these holes and this board.

    Unknown burn-free tail cards are the unused ids, lowest first. Showdown
    only needs the recorded board; folded hands never deal the filler.
    """

    known = [int(card) for card in (*holes[0], *holes[1], *board)]
    if len(known) != len(set(known)):
        raise ValueError("holes and board overlap")
    unused = [card for card in range(52) if card not in set(known)]
    full_board = list(int(card) for card in board) + unused[: 5 - len(board)]
    rest = unused[5 - len(board) :]
    sequence = [
        int(holes[0][0]),
        int(holes[0][1]),
        int(holes[1][0]),
        int(holes[1][1]),
        *full_board,
    ]
    deck = rest + list(reversed(sequence))
    if len(deck) != 52 or len(set(deck)) != 52:
        raise RuntimeError("reconstructed deck is not 52 distinct cards")
    return deck


def _all_keys(value: Any, found: set) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            _all_keys(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _all_keys(item, found)


def _int_lists(value: Any, found: List[List[int]]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _int_lists(item, found)
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            found.append([int(item) for item in value])
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                _int_lists(item, found)


def policy_view_leak(kernel: BranchableKernel) -> Optional[str]:
    """Return a short reason if any seat's view shows a hidden card."""

    audit = kernel.referee_audit_state()
    for seat in (0, 1):
        view = kernel.policy_view(seat)
        hidden = set(audit["holes"][1 - seat]) | set(audit["deck"])
        if hidden.intersection(view["hole_cards"]) or hidden.intersection(view["public_cards"]):
            return "hidden card in hole_cards or public_cards"
        if list(view["hole_cards"]) != list(audit["holes"][seat]):
            return "hole_cards are not this seat's cards"
        if list(view["public_cards"]) != list(audit["public_cards"]):
            return "public_cards are not the dealt board"
        keys: set = set()
        _all_keys(view, keys)
        for banned in ("deck", "opponent_cards", "opponent_hole", "undealt", "remaining_deck"):
            if banned in keys:
                return f"view key {banned}"
        lists: List[List[int]] = []
        _int_lists(view, lists)
        opponent = list(audit["holes"][1 - seat])
        if opponent in lists or list(reversed(opponent)) in lists:
            return "opponent hole list is present"
        if len(view["feature_vector"]) != len(view["feature_names"]):
            return "feature vector length"
        if set(view["feature_names"]) != set(view["features"]):
            return "feature names"
    return None


def _apply_shadow(game, response):
    parsed = JUDGE.parse_bot_response(response)
    if parsed is None:
        raise ValueError(f"shadow rejected parse of {response!r}")
    result = game.player_action(parsed)
    return parsed, result


def _settle_totals(totals: List[float], games: List[int], game, winners) -> Tuple[float, float]:
    finals = game.get_player_final_chips(winners)
    settled = (float(finals[0]), float(finals[1]))
    mean = sum(settled) / float(len(settled))
    for seat, chips in enumerate(settled):
        totals[seat] += chips - mean
        if chips > mean:
            games[seat] += 1
    return settled


def _numbers_equal(left, right) -> bool:
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def compare_live(
    kernel: BranchableKernel,
    game,
    *,
    totals: Dict[str, int],
    examples: List[Dict[str, Any]],
    where: Dict[str, Any],
    raise_now: bool,
) -> None:
    diffs = judge_field_diffs(kernel.referee_audit_state(), shadow_view(game))
    if diffs:
        _note(totals, "step_mismatches")
        for field in diffs:
            if field == "actor":
                _note(totals, "next_actor_mismatches")
            elif field == "pot":
                _note(totals, "pot_mismatches")
            elif field == "chips":
                _note(totals, "stack_mismatches")
            elif field == "street":
                _note(totals, "street_mismatches")
        _fail_or_push(
            examples,
            {**where, "kind": "state", "fields": diffs},
            raise_now=raise_now,
        )
    if not kernel.conserved():
        _note(totals, "conservation_failures")
        _fail_or_push(examples, {**where, "kind": "conservation"}, raise_now=raise_now)


def check_view(
    kernel: BranchableKernel,
    *,
    totals: Dict[str, int],
    examples: List[Dict[str, Any]],
    where: Dict[str, Any],
    raise_now: bool,
) -> None:
    _note(totals, "policy_view_checks")
    leak = policy_view_leak(kernel)
    if leak:
        _note(totals, "policy_view_leaks")
        _fail_or_push(examples, {**where, "kind": "policy_view", "leak": leak}, raise_now=raise_now)


def check_legacy(
    kernel: BranchableKernel,
    game,
    *,
    totals: Dict[str, int],
    examples: List[Dict[str, Any]],
    where: Dict[str, Any],
    raise_now: bool,
) -> None:
    if kernel.hand_over:
        return
    _note(totals, "legacy_bridge_checks")
    bridge = LegacyFiveActionBridge(kernel)
    for action_id, response in enumerate(bridge.responses()):
        if response is None:
            continue
        if int(response) == int(action_id):
            _note(totals, "legacy_id_equals_chip_response")
        else:
            _note(totals, "legacy_id_differs_from_chip_response")
        if not shadow_accepts(game, int(response)):
            _note(totals, "legacy_bridge_mismatches")
            _fail_or_push(
                examples,
                {**where, "kind": "legacy", "action_id": action_id, "response": int(response)},
                raise_now=raise_now,
            )


def check_probes(
    kernel: BranchableKernel,
    game,
    probes: Sequence[Any],
    *,
    totals: Dict[str, int],
    examples: List[Dict[str, Any]],
    where: Dict[str, Any],
    raise_now: bool,
) -> None:
    _note(totals, "states")
    deduped = []
    used = []
    for probe in probes:
        key = (type(probe).__name__, repr(probe))
        if key in used:
            continue
        used.append(key)
        deduped.append(probe)
    for probe in deduped:
        _note(totals, "probe_count")
        _note(totals, "legal_reject_checks")
        ours = kernel.is_legal(probe)
        theirs = shadow_accepts(game, probe)
        if ours != theirs:
            _note(totals, "legal_reject_mismatches")
            _fail_or_push(
                examples,
                {**where, "kind": "legal", "probe": probe, "kernel": ours, "judge": theirs},
                raise_now=raise_now,
            )
    judge_min = judge_min_positive(state_from_holdem(game))
    ours_min = kernel.min_positive_extra()
    if ours_min != judge_min:
        _note(totals, "min_positive_mismatches")
        _note(totals, "legal_reject_mismatches")
        _fail_or_push(
            examples,
            {**where, "kind": "min_positive", "kernel": ours_min, "judge": judge_min},
            raise_now=raise_now,
        )


def boundary_probes(kernel: BranchableKernel) -> List[Any]:
    chips = int(kernel.stacks[kernel.actor])
    probes: List[Any] = [
        None,
        True,
        False,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        49,
        50,
        99,
        100,
        149,
        150,
        150.0,
        150.5,
        "0",
        chips - 1,
        chips,
        chips + 1,
    ]
    minimum = kernel.min_positive_extra()
    if minimum is not None:
        probes.extend([minimum - 1, minimum, minimum + 1, float(minimum)])
    return probes


def _choose_boundary_action(rng: random.Random, kernel: BranchableKernel) -> int:
    if kernel.actions_this_hand >= 24:
        return -1
    chips = int(kernel.stacks[kernel.actor])
    minimum = kernel.min_positive_extra()
    options = [response for response in (-1, -2, 0) if kernel.is_legal(response)]
    if minimum is not None and kernel.is_legal(minimum):
        options.append(int(minimum))
        if minimum < chips - 1:
            options.append(chips - 1)
            options.append(int(rng.randrange(int(minimum), chips)))
    if not options:
        return -1
    # Prefer the edges over an immediate fold so the walk reaches later streets.
    if len(options) > 1 and rng.random() < 0.85:
        options = [option for option in options if option != -1] or options
    return int(rng.choice(options))


def _choose_random_action(rng: random.Random, kernel: BranchableKernel) -> int:
    if kernel.actions_this_hand >= 24:
        return -1
    chips = int(kernel.stacks[kernel.actor])
    minimum = kernel.min_positive_extra()
    options = [response for response in (-1, -2, 0) if kernel.is_legal(response)]
    if minimum is not None and kernel.is_legal(minimum):
        options.append(int(minimum))
        pot_sized = max(int(minimum), int(kernel.pot))
        if pot_sized < chips and kernel.is_legal(pot_sized):
            options.append(int(pot_sized))
        if minimum < chips:
            options.append(int(rng.randrange(int(minimum), chips)))
    if not options:
        return -1
    return int(rng.choice(options))


def _play_pair(
    kernel: BranchableKernel,
    game,
    choose,
    *,
    totals: Dict[str, int],
    examples: List[Dict[str, Any]],
    where: Dict[str, Any],
    probes: bool,
    legacy: bool,
    raise_now: bool,
    clone_once: bool,
) -> Optional[Tuple[Tuple[int, ...], Tuple[float, float]]]:
    cloned = False
    finished: Optional[Tuple[Tuple[int, ...], Tuple[float, float]]] = None
    while not kernel.hand_over:
        spot = {**where, "decision": int(totals.get("decisions", 0)), "street": kernel.street}
        compare_live(kernel, game, totals=totals, examples=examples, where=spot, raise_now=raise_now)
        check_view(kernel, totals=totals, examples=examples, where=spot, raise_now=raise_now)
        if legacy:
            check_legacy(
                kernel, game, totals=totals, examples=examples, where=spot, raise_now=raise_now
            )
        if probes:
            check_probes(
                kernel,
                game,
                boundary_probes(kernel),
                totals=totals,
                examples=examples,
                where=spot,
                raise_now=raise_now,
            )
            illegal = next(
                (probe for probe in boundary_probes(kernel) if not kernel.is_legal(probe)),
                None,
            )
            if illegal is not None:
                before = kernel.referee_audit_state()
                try:
                    kernel.step(illegal)
                except ValueError:
                    pass
                else:
                    _note(totals, "legal_reject_mismatches")
                    _fail_or_push(
                        examples,
                        {**spot, "kind": "illegal_step_accepted", "probe": illegal},
                        raise_now=raise_now,
                    )
                if kernel.referee_audit_state() != before:
                    _note(totals, "legal_reject_mismatches")
                    _fail_or_push(
                        examples,
                        {**spot, "kind": "illegal_step_mutated", "probe": illegal},
                        raise_now=raise_now,
                    )
        if clone_once and not cloned:
            twin = kernel.clone()
            saved = twin.referee_audit_state()
            _note(totals, "clone_checks")
            cloned = True
        else:
            twin = None
            saved = None
        response = int(choose(kernel))
        _note(totals, "legal_reject_checks")
        if not shadow_accepts(game, response) or not kernel.is_legal(response):
            _note(totals, "legal_reject_mismatches")
            _fail_or_push(
                examples,
                {**spot, "kind": "chosen_action", "response": response},
                raise_now=raise_now,
            )
            response = -1
        _note(totals, "decisions")
        kernel.step(response)
        _parsed, result = _apply_shadow(game, response)
        del _parsed
        if twin is not None and twin.referee_audit_state() != saved:
            _note(totals, "clone_isolation_failures")
            _fail_or_push(examples, {**spot, "kind": "clone"}, raise_now=raise_now)
        if bool(result) != kernel.hand_over:
            _note(totals, "step_mismatches")
            _fail_or_push(
                examples,
                {**spot, "kind": "terminal", "judge": bool(result), "kernel": kernel.hand_over},
                raise_now=raise_now,
            )
        compare_live(kernel, game, totals=totals, examples=examples, where=spot, raise_now=raise_now)
        if kernel.hand_over:
            if result:
                winners = tuple(int(seat) for seat in result)
                finals = game.get_player_final_chips(result)
                settled = (float(finals[0]), float(finals[1]))
                if kernel.winners != winners or kernel.settled_chips != settled:
                    _note(totals, "showdown_payoff_mismatches")
                    _note(totals, "payoff_mismatches")
                    _fail_or_push(
                        examples,
                        {
                            **spot,
                            "kind": "settlement",
                            "kernel_winners": kernel.winners,
                            "judge_winners": winners,
                            "kernel_settled": kernel.settled_chips,
                            "judge_settled": settled,
                        },
                        raise_now=raise_now,
                    )
                finished = (winners, settled)
                if any(int(action.get("action", 0)) == -1 for action in game.history[-1:]):
                    _note(totals, "folds")
                else:
                    _note(totals, "showdowns")
            check_view(kernel, totals=totals, examples=examples, where=spot, raise_now=raise_now)
    return finished


def _shuffled(rng: random.Random) -> List[int]:
    cards = list(range(52))
    rng.shuffle(cards)
    return cards


def run_boundary(n_states: int = 1000, seed: int = 20261031, *, raise_now: bool = True) -> Dict[str, Any]:
    """Probe legal edges on scripted lines and random walks until ``n_states``."""

    if n_states < 1:
        raise ValueError(f"n_states must be positive, got {n_states}")
    rng = random.Random(seed)
    totals = _fresh_counters()
    examples: List[Dict[str, Any]] = []
    scripts: Tuple[Tuple[int, ...], ...] = (
        (-1,),
        (0, -1),
        (0, 0, -1),
        (150, -1),
        (150, 0),
        (0, 100, 101, 102, 3, -1),
        (-2, -2),
        (-2, -1),
        (0, -2, -1),
        (0, 0, 100, 200, -1),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, -2, -2),
    )
    hand_no = 0

    def play_deck(deck: Sequence[int], dealer: int, script: Optional[Sequence[int]]) -> None:
        nonlocal hand_no
        kernel = BranchableKernel.start_hand(deck, dealer=dealer)
        game = deal_shadow(deck, dealer)
        _note(totals, "hands")
        where = {"suite": "boundary", "hand": hand_no, "seed": seed}
        hand_no += 1
        cursor = {"index": 0}

        def choose(kernel_now: BranchableKernel) -> int:
            if script is not None and cursor["index"] < len(script):
                response = int(script[cursor["index"]])
                cursor["index"] += 1
                if kernel_now.is_legal(response):
                    return response
            return _choose_boundary_action(rng, kernel_now)

        _play_pair(
            kernel,
            game,
            choose,
            totals=totals,
            examples=examples,
            where=where,
            probes=True,
            legacy=True,
            raise_now=raise_now,
            clone_once=True,
        )

    for dealer in (0, 1):
        for script in scripts:
            play_deck(list(range(52)), dealer, script)
            if totals["states"] >= n_states:
                break
        if totals["states"] >= n_states:
            break
    while totals["states"] < n_states:
        play_deck(_shuffled(rng), hand_no % 2, None)
    return {
        "seed": seed,
        "requested_states": n_states,
        **totals,
        "examples": examples,
    }


def run_trajectories(
    n_hands: int = 10000,
    seed: int = 404,
    *,
    match_length: int = 50,
    raise_now: bool = True,
) -> Dict[str, Any]:
    """Random legal hands grouped into 50-hand matches. Scores are checked."""

    if n_hands < 1:
        raise ValueError(f"n_hands must be positive, got {n_hands}")
    if match_length < 1:
        raise ValueError(f"match_length must be positive, got {match_length}")
    rng = random.Random(seed)
    totals = _fresh_counters()
    examples: List[Dict[str, Any]] = []
    remaining = int(n_hands)
    match_index = 0
    while remaining > 0:
        length = min(match_length, remaining)
        dealer0 = match_index % 2
        decks = [_shuffled(rng) for _ in range(length)]
        kernel = BranchableKernel.start_match(decks, initial_dealer=dealer0, max_hands=length)
        score_chips = [0.0, 0.0]
        score_games = [0, 0]
        _note(totals, "matches")
        for hand_index in range(length):
            dealer = (dealer0 + hand_index) % 2
            game = deal_shadow(decks[hand_index], dealer)
            where = {"suite": "random", "seed": seed, "match": match_index, "hand": hand_index}
            _note(totals, "hands")

            def choose(kernel_now: BranchableKernel, hand_rng: random.Random = rng) -> int:
                return _choose_random_action(hand_rng, kernel_now)

            _play_pair(
                kernel,
                game,
                choose,
                totals=totals,
                examples=examples,
                where=where,
                probes=False,
                legacy=True,
                raise_now=raise_now,
                clone_once=hand_index == 0,
            )
            if not kernel.hand_over:
                _note(totals, "step_mismatches")
                _fail_or_push(examples, {**where, "kind": "hand_not_over"}, raise_now=raise_now)
            else:
                # Recompute the settlement the judge would report and compare.
                # The shadow game is already terminal; winners were applied inside
                # _play_pair. Recover them from the kernel and check chips.
                if kernel.settled_chips is None:
                    _note(totals, "showdown_payoff_mismatches")
                    _fail_or_push(examples, {**where, "kind": "missing_settlement"}, raise_now=raise_now)
                else:
                    # Independent settlement from the shadow's current pot and chips
                    # plus the kernel winners must match. Winners came from the judge
                    # return, already compared via the terminal flag. Re-derive the
                    # same formula from the shadow object.
                    winners = list(kernel.winners)
                    settled = _settle_totals(score_chips, score_games, game, winners)
                    if settled != kernel.settled_chips:
                        _note(totals, "showdown_payoff_mismatches")
                        _note(totals, "payoff_mismatches")
                        _fail_or_push(
                            examples,
                            {
                                **where,
                                "kind": "payoff",
                                "kernel": kernel.settled_chips,
                                "judge": settled,
                            },
                            raise_now=raise_now,
                        )
                    if sum(kernel.payoffs) != 0:
                        _note(totals, "conservation_failures")
                        _fail_or_push(examples, {**where, "kind": "payoff_sum"}, raise_now=raise_now)
            if hand_index + 1 < length:
                kernel.next_hand()
        expected_scores = (score_chips[0] / BIG_BLIND, score_chips[1] / BIG_BLIND)
        scores_match = _numbers_equal(kernel.scores[0], expected_scores[0]) and _numbers_equal(
            kernel.scores[1], expected_scores[1]
        )
        if not scores_match or kernel.referee_audit_state()["total_win_games"] != (
            score_games[0],
            score_games[1],
        ):
            _note(totals, "match_score_mismatches")
            _fail_or_push(
                examples,
                {
                    "suite": "random",
                    "seed": seed,
                    "match": match_index,
                    "kind": "match_score",
                    "kernel": kernel.scores,
                    "judge": expected_scores,
                },
                raise_now=raise_now,
            )
        if length == match_length and not kernel.match_over:
            _note(totals, "match_score_mismatches")
            _fail_or_push(
                examples,
                {"suite": "random", "match": match_index, "kind": "match_not_over"},
                raise_now=raise_now,
            )
        remaining -= length
        match_index += 1
        if match_index % 20 == 0:
            print(
                f"trajectories matches={match_index} hands={totals['hands']} "
                f"decisions={totals['decisions']}",
                flush=True,
            )
    return {"seed": seed, "match_length": match_length, **totals, "examples": examples}


def run_replay(path: Optional[Path] = None, *, raise_now: bool = True) -> Dict[str, Any]:
    """Replay the in-repo fixture. Payoffs are compared to the recorded ``r``."""

    fixture = Path(path) if path is not None else _FIXTURE
    grouped: Dict[str, List[dict]] = {}
    with gzip.open(fixture, "rt") as handle:
        for line in handle:
            hand = json.loads(line)
            grouped.setdefault(str(hand["m"]), []).append(hand)
    totals = _fresh_counters()
    examples: List[Dict[str, Any]] = []
    for match_id, hands in grouped.items():
        hands.sort(key=lambda row: int(row["h"]))
        _note(totals, "matches")
        decks = [deck_from_partial(hand["c"], hand["b"]) for hand in hands]
        dealer0 = int(hands[0]["d"])
        kernel = BranchableKernel.start_match(
            decks, initial_dealer=dealer0, max_hands=len(hands)
        )
        score_chips = [0.0, 0.0]
        recorded = [0.0, 0.0]
        for hand in hands:
            dealer = int(hand["d"])
            game = deal_shadow(decks[int(hand["h"])], dealer)
            where = {"suite": "replay", "match": match_id, "hand": int(hand["h"])}
            _note(totals, "hands")
            cursor = {"index": 0}
            actions = hand["a"]

            def choose(kernel_now: BranchableKernel, actions=actions, cursor=cursor) -> int:
                if cursor["index"] >= len(actions):
                    return -1
                pid, response, street = actions[cursor["index"]]
                cursor["index"] += 1
                if int(kernel_now.actor) != int(pid) or int(kernel_now.street) != int(street):
                    _note(totals, "next_actor_mismatches")
                    _note(totals, "street_mismatches")
                    _fail_or_push(
                        examples,
                        {
                            **where,
                            "kind": "fixture_order",
                            "kernel_actor": kernel_now.actor,
                            "kernel_street": kernel_now.street,
                            "fixture_actor": int(pid),
                            "fixture_street": int(street),
                        },
                        raise_now=raise_now,
                    )
                return int(response)

            _play_pair(
                kernel,
                game,
                choose,
                totals=totals,
                examples=examples,
                where=where,
                probes=False,
                legacy=True,
                raise_now=raise_now,
                clone_once=int(hand["h"]) == 0,
            )
            if cursor["index"] != len(actions) or not kernel.hand_over:
                _note(totals, "step_mismatches")
                _fail_or_push(examples, {**where, "kind": "truncated_replay"}, raise_now=raise_now)
            if tuple(kernel.public_cards) != tuple(int(card) for card in hand["b"]):
                _note(totals, "step_mismatches")
                _fail_or_push(examples, {**where, "kind": "board"}, raise_now=raise_now)
            payoffs = kernel.payoffs
            if payoffs is None or not (
                _numbers_equal(payoffs[0], hand["r"][0]) and _numbers_equal(payoffs[1], hand["r"][1])
            ):
                _note(totals, "payoff_mismatches")
                _note(totals, "showdown_payoff_mismatches")
                _fail_or_push(
                    examples,
                    {**where, "kind": "fixture_payoff", "kernel": payoffs, "fixture": hand["r"]},
                    raise_now=raise_now,
                )
            winners = list(kernel.winners)
            settled = _settle_totals(score_chips, [0, 0], game, winners)
            if payoffs is not None and (
                not _numbers_equal(settled[0] - STARTING_STACK, payoffs[0])
                or not _numbers_equal(settled[1] - STARTING_STACK, payoffs[1])
            ):
                _note(totals, "showdown_payoff_mismatches")
                _fail_or_push(examples, {**where, "kind": "shadow_payoff"}, raise_now=raise_now)
            recorded[0] += float(hand["r"][0])
            recorded[1] += float(hand["r"][1])
            if int(hand["h"]) + 1 < len(hands):
                kernel.next_hand()
        expected = (recorded[0] / BIG_BLIND, recorded[1] / BIG_BLIND)
        if not (
            _numbers_equal(kernel.scores[0], expected[0]) and _numbers_equal(kernel.scores[1], expected[1])
        ):
            _note(totals, "match_score_mismatches")
            _fail_or_push(
                examples,
                {"suite": "replay", "match": match_id, "kind": "match_score", "kernel": kernel.scores, "fixture": expected},
                raise_now=raise_now,
            )
    try:
        fixture_name = str(fixture.resolve().relative_to(_REPO))
    except ValueError:
        fixture_name = str(fixture)
    return {
        "path": fixture_name,
        "sha256": _sha256(fixture),
        "corpus": "in_repo_botzone_real_jsonl_gz",
        "mac_33_match_tree": "not_attached",
        "note": (
            "botzone_replay_analysis_20261030 (33 unique matches) was not on this machine. "
            "This replay is tests/fixtures/botzone_real.jsonl.gz, the corpus already used by "
            "eval/botzone_referee_diff.py and tests/test_botzone_referee_diff.py. "
            "It has 41 complete 50-hand matches, not the 33-match Mac tree."
        ),
        **totals,
        "examples": examples,
    }


def _mismatch_total(report: Dict[str, Any]) -> int:
    keys = (
        "step_mismatches",
        "legal_reject_mismatches",
        "payoff_mismatches",
        "next_actor_mismatches",
        "pot_mismatches",
        "stack_mismatches",
        "street_mismatches",
        "showdown_payoff_mismatches",
        "conservation_failures",
        "policy_view_leaks",
        "legacy_bridge_mismatches",
        "match_score_mismatches",
        "clone_isolation_failures",
        "min_positive_mismatches",
    )
    return sum(int(report.get(key, 0)) for key in keys)


def build_report(
    *,
    boundary_states: int = 1000,
    trajectory_hands: int = 10000,
    boundary_seed: int = 20261031,
    trajectory_seed: int = 404,
    raise_now: bool = True,
) -> Dict[str, Any]:
    judge_sha = _sha256(_JUDGE)
    replay = run_replay(raise_now=raise_now)
    print(
        f"replay matches={replay['matches']} hands={replay['hands']} decisions={replay['decisions']}",
        flush=True,
    )
    boundary = run_boundary(boundary_states, boundary_seed, raise_now=raise_now)
    print(
        f"boundary states={boundary['states']} hands={boundary['hands']} probes={boundary['probe_count']}",
        flush=True,
    )
    trajectories = run_trajectories(trajectory_hands, trajectory_seed, raise_now=raise_now)
    print(
        f"trajectories hands={trajectories['hands']} decisions={trajectories['decisions']}",
        flush=True,
    )
    residual = _mismatch_total(replay) + _mismatch_total(boundary) + _mismatch_total(trajectories)
    status = "PASS"
    if judge_sha != _JUDGE_SHA:
        status = "FAIL"
    if residual:
        status = "FAIL"
    if int(boundary["states"]) < 1000 or int(trajectories["hands"]) < 10000:
        status = "FAIL"
    if int(replay["hands"]) < 1 or int(replay["matches"]) < 1:
        status = "FAIL"
    return {
        "task_id": "R02",
        "status": status,
        "gpu_hours_used": 0,
        "kernel": "alphanl.botzone.kernel.BranchableKernel",
        "legacy_bridge": "alphanl.botzone.kernel.LegacyFiveActionBridge",
        "judge_fixture": "tests/fixtures/botzone_judge.py",
        "judge_class_method": "Holdem.player_action",
        "judge_sha256": judge_sha,
        "judge_sha256_expected": _JUDGE_SHA,
        "judge_sha256_unchanged": judge_sha == _JUDGE_SHA,
        "transition": (
            "Each step calls the unmodified judge. A second Holdem is dealt from the same "
            "deck and fed the same chip responses. Legality probes call player_action on a "
            "throwaway copy. legal.apply is not the transition."
        ),
        "seeds": {
            "boundary": boundary_seed,
            "random_legal_trajectories": trajectory_seed,
            "replay": None,
            "evaluation_contract_notes": {
                "boundary_uses": "evaluation_master_seed 20261031",
                "trajectories_use": "dev_eval_seed 404",
            },
        },
        "corpora": {
            "replay": replay,
            "boundary": boundary,
            "random_legal_trajectories": trajectories,
        },
        "requirements": {
            "replay_corpus": "in-repo fixture; 33-match Mac tree not attached",
            "boundary_states_at_least": 1000,
            "random_legal_trajectories_at_least": 10000,
        },
        "residual_mismatches": residual,
        "examples": (replay["examples"] + boundary["examples"] + trajectories["examples"])[:_EXAMPLE_CAP],
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Branchable kernel vs official Botzone referee")
    parser.add_argument("--boundary-states", type=int, default=1000)
    parser.add_argument("--trajectory-hands", type=int, default=10000)
    parser.add_argument("--boundary-seed", type=int, default=20261031)
    parser.add_argument("--trajectory-seed", type=int, default=404)
    parser.add_argument("--out", type=str, default="eval/r02/engine_diff_report.json")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--skip-trajectories", action="store_true")
    args = parser.parse_args(argv)
    if args.skip_replay or args.skip_boundary or args.skip_trajectories:
        report: Dict[str, Any] = {"task_id": "R02", "status": "INCOMPLETE", "gpu_hours_used": 0}
        if not args.skip_replay:
            report["replay"] = run_replay()
        if not args.skip_boundary:
            report["boundary"] = run_boundary(args.boundary_states, args.boundary_seed)
        if not args.skip_trajectories:
            report["trajectories"] = run_trajectories(args.trajectory_hands, args.trajectory_seed)
    else:
        report = build_report(
            boundary_states=args.boundary_states,
            trajectory_hands=args.trajectory_hands,
            boundary_seed=args.boundary_seed,
            trajectory_seed=args.trajectory_seed,
        )
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} status={report.get('status')} residual={report.get('residual_mismatches')}")
    if report.get("status") not in (None, "PASS"):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()

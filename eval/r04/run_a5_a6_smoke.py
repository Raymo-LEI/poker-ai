"""CPU smoke of A5 and A6 on the sealed branchable kernel.

No training. Each boundary state is a real judge decision. Mapped chip
responses are checked for legality and stack limits, then one of them is
stepped. Identical chip responses are one executable outcome.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from alphanl.botzone.action_spec import (  # noqa: E402
    A5,
    A6,
    chip_responses,
    hardcoded_width_audit,
    legacy_v1_document,
    mass_by_name,
    quarter_disposition,
    response_mass,
    schema_bundle,
    shared_action_names,
)
from alphanl.botzone.kernel import BranchableKernel  # noqa: E402
from alphanl.botzone.legal import to_call  # noqa: E402

_SEED = 20261031
_MIN_STATES = 1000


def _shuffled(rng: random.Random):
    cards = list(range(52))
    rng.shuffle(cards)
    return cards


def _choose(rng: random.Random, kernel: BranchableKernel) -> int:
    if kernel.actions_this_hand >= 12:
        return -1
    chips = int(kernel.stacks[kernel.actor])
    minimum = kernel.min_positive_extra()
    options = [response for response in (-1, -2, 0) if kernel.is_legal(response)]
    if minimum is not None and kernel.is_legal(minimum):
        options.append(int(minimum))
        if minimum < chips:
            options.append(int(rng.randrange(int(minimum), chips)))
    if len(options) > 1 and rng.random() < 0.8:
        options = [option for option in options if option != -1] or options
    return int(rng.choice(options)) if options else -1


def _chips_committed(state, response: int) -> int:
    if response == -1:
        return 0
    if response == -2:
        return int(state.stacks[state.actor])
    if response == 0:
        return int(to_call(state))
    return int(response)


def _check_state(kernel: BranchableKernel, stats: dict) -> None:
    state = kernel.as_hand_state()
    stack = int(state.stacks[state.actor])
    for schema in (A5, A6):
        responses = chip_responses(state, schema)
        executable = [response for response in responses if response is not None]
        if len(executable) != len(set(executable)):
            stats["duplicate_executable"] += 1
        for response in executable:
            stats["checked_responses"] += 1
            spent = _chips_committed(state, int(response))
            if spent > stack or (response > 0 and response >= stack) or not kernel.is_legal(int(response)):
                stats["illegal_or_overstack"] += 1
                stats["examples"].append(
                    {
                        "schema": schema.schema_id,
                        "response": int(response),
                        "stack": stack,
                        "street": int(state.street),
                    }
                )
        if executable:
            probe = kernel.clone()
            probe.step(int(executable[0]))
            if not probe.conserved():
                stats["conservation_failures"] += 1
    disposition = quarter_disposition(state)
    stats["states"] += 1
    if disposition == "new":
        stats["quarter_new"] += 1
    elif disposition == "collapsed":
        stats["quarter_collapsed"] += 1
    else:
        stats["quarter_masked"] += 1
    a5_mass = response_mass(state, A5)
    a6_mass = response_mass(state, A6)
    named5 = mass_by_name(state, A5)
    named6 = mass_by_name(state, A6)
    for name in shared_action_names():
        left = named5[name]
        right = named6[name]
        same_chips = (left is None and right is None) or (
            left is not None and right is not None and left["response"] == right["response"]
        )
        if not same_chips:
            stats["shared_response_mismatches"] += 1
        if left is not None and right is not None and left["response"] == right["response"]:
            response = int(left["response"])
            stats["shared_mass_pairs"].append((a5_mass[response], a6_mass[response]))


def collect(n_states: int = _MIN_STATES, seed: int = _SEED) -> dict:
    if n_states < 1:
        raise ValueError("n_states must be positive")
    rng = random.Random(seed)
    stats = {
        "states": 0,
        "checked_responses": 0,
        "illegal_or_overstack": 0,
        "duplicate_executable": 0,
        "conservation_failures": 0,
        "shared_response_mismatches": 0,
        "quarter_new": 0,
        "quarter_collapsed": 0,
        "quarter_masked": 0,
        "shared_mass_pairs": [],
        "examples": [],
    }
    hand = 0
    while stats["states"] < n_states:
        kernel = BranchableKernel.start_hand(_shuffled(rng), dealer=hand % 2)
        guard = 0
        while not kernel.hand_over and stats["states"] < n_states and guard < 24:
            _check_state(kernel, stats)
            kernel.step(_choose(rng, kernel))
            guard += 1
        hand += 1
    pairs = stats.pop("shared_mass_pairs")
    if pairs:
        abs_diff = [abs(left - right) for left, right in pairs]
        mean_abs = sum(abs_diff) / len(abs_diff)
    else:
        mean_abs = 0.0
    coverage = stats["quarter_new"] / stats["states"]
    return {
        "seed": seed,
        "hands_started": hand,
        "states": stats["states"],
        "checked_responses": stats["checked_responses"],
        "illegal_or_overstack": stats["illegal_or_overstack"],
        "duplicate_executable": stats["duplicate_executable"],
        "conservation_failures": stats["conservation_failures"],
        "shared_response_mismatches": stats["shared_response_mismatches"],
        "shared_mass_pairs": len(pairs),
        "shared_mass_mean_abs_diff": mean_abs,
        "quarter_new": stats["quarter_new"],
        "quarter_collapsed": stats["quarter_collapsed"],
        "quarter_masked": stats["quarter_masked"],
        "quarterpot_effective_new_coverage": coverage,
        "examples": stats["examples"][:8],
        "small_bet_skill_claimed": False,
    }


def write_reports(root: Path, n_states: int = _MIN_STATES, seed: int = _SEED) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    bundle = schema_bundle()
    (root / "action_schema_v2.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    smoke = collect(n_states, seed)
    (root / "a5_a6_smoke_report.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    equivalence = {
        "comparison": "chip_response_probability_mass",
        "not": "naive action-id alignment",
        "shared_names": list(shared_action_names()),
        "shared_response_mismatches": smoke["shared_response_mismatches"],
        "shared_mass_mean_abs_diff": smoke["shared_mass_mean_abs_diff"],
        "shared_mass_pairs": smoke["shared_mass_pairs"],
        "quarterpot_effective_new_coverage": smoke["quarterpot_effective_new_coverage"],
        "quarter_new": smoke["quarter_new"],
        "quarter_collapsed": smoke["quarter_collapsed"],
        "quarter_masked": smoke["quarter_masked"],
        "small_bet_skill_claimed": False,
        "note": (
            "Uniform mass over each schema's executable ids is summed by chip "
            "response. A6 quarter-pot counts as new coverage only when that "
            "chip amount is absent from A5."
        ),
    }
    (root / "response_equivalence.json").write_text(
        json.dumps(equivalence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "retained_v1_sha256": legacy_v1_document()["sha256"],
        "a5_sha256": bundle["a5"]["sha256"],
        "a6_sha256": bundle["a6"]["sha256"],
        "sites": hardcoded_width_audit(),
    }
    (root / "hardcoded_action_width_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"smoke": smoke, "bundle": bundle, "equivalence": equivalence}


def main() -> None:
    root = Path(__file__).resolve().parent
    payload = write_reports(root)
    smoke = payload["smoke"]
    print(json.dumps({
        "states": smoke["states"],
        "illegal_or_overstack": smoke["illegal_or_overstack"],
        "duplicate_executable": smoke["duplicate_executable"],
        "quarterpot_effective_new_coverage": smoke["quarterpot_effective_new_coverage"],
    }))
    if smoke["states"] < _MIN_STATES or smoke["illegal_or_overstack"] or smoke["duplicate_executable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

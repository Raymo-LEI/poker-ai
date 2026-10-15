"""1000 random legal hands on BotzoneTable.

Every response is drawn from ``BotzoneTable.legal_responses``, the same
set M1 ``legal_response`` returns. After each step, and again when the
hand ends, the two stacks plus the pot are 40000. The two seats' chip
deltas (hero and villain) sum to 0.

A chopped odd pot would leave a 1-chip remainder. That chip goes to the
smaller seat id. Heads-up showdowns put the same number of chips in from
each seat, so the pot is even and this run usually never sees a remainder.
The loop asserts the rule if one appears. If none does, the placeholder
at the end locks the same rule through ``split_pot``.
"""

import random

from alphanl.botzone.legal import TOTAL_CHIPS, apply, legal_response
from alphanl.botzone.table import BotzoneTable, split_pot

HANDS = 1000
SEED = 1000
_MAX_STEPS = 64


def _pick(rng: random.Random, legal: set) -> int:
    """One member of the set ``legal_response`` already returned."""

    options = sorted(legal)
    return options[rng.randrange(len(options))]


def test_thousand_random_legal_hands_conserve_chips_and_are_zero_sum():
    assert TOTAL_CHIPS == 40000
    rng = random.Random(SEED)
    table = BotzoneTable(rng=rng, dealer=0)
    odd_chops = 0
    steps = 0

    for hand_index in range(HANDS):
        table.start_hand()
        assert table.stacks[0] + table.stacks[1] + table.pot == 40000
        for _ in range(_MAX_STEPS):
            if table.hand_over:
                break
            state = table.state
            legal = table.legal_responses()
            assert legal == legal_response(state)
            assert legal
            response = _pick(rng, legal)
            assert response in legal_response(state)
            # Betting ends with the pot still sitting there. Showdown award
            # happens inside ``act``. Preview is that pre-award state.
            preview = apply(state, response)
            table.act(response)
            assert table.stacks[0] + table.stacks[1] + table.pot == 40000
            steps += 1
            if (
                preview.hand_over
                and not preview.folded[0]
                and not preview.folded[1]
                and table.winners == (0, 1)
            ):
                award0 = table.stacks[0] - preview.stacks[0]
                award1 = table.stacks[1] - preview.stacks[1]
                assert award0 + award1 == preview.pot
                if preview.pot % 2 == 1:
                    odd_chops += 1
                    assert award0 == preview.pot // 2 + 1, (
                        f"hand {hand_index}: odd pot {preview.pot} "
                        f"remainder did not go to seat 0 "
                        f"(awards {award0}, {award1})"
                    )
                    assert award1 == preview.pot // 2
                else:
                    assert award0 == award1 == preview.pot // 2
        assert table.hand_over, f"hand {hand_index} did not finish within {_MAX_STEPS} steps"
        assert table.pot == 0
        assert table.stacks[0] + table.stacks[1] + table.pot == 40000
        seat0_delta, seat1_delta = table.payoffs
        assert seat0_delta + seat1_delta == 0

    assert steps > HANDS
    if odd_chops == 0:
        # Placeholder: these 1000 hands never chopped an odd pot.
        # Remainder 1 still belongs to the smaller seat id.
        assert split_pot(1, (0, 1)) == (1, 0)
        assert split_pot(151, (0, 1)) == (76, 75)
        assert split_pot(151, (1, 0)) == (76, 75)

"""Slumbot observations must match the exact training engine and tokenizer."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_cfr_slumbot import api_action, decision_table
from alphanl.transformer.hunl import HunlTable, HunlTokenizer


@pytest.mark.parametrize("seed", (4, 11, 73))
def test_external_replay_matches_training_observation_with_offtree_bets(seed):
    table = HunlTable(seed=seed)
    table.reset()
    rng = np.random.default_rng(seed)
    history = ""
    decisions = 0
    while not table.is_over():
        s = table.snapshot(table.current_player())
        response = {"client_pos": s.me_seat, "action": history,
                    "hole_cards": list(s.hole_cards), "board": list(s.board_cards)}
        replay, parsed = decision_table(response)
        got = replay.snapshot(s.me_seat)
        assert got.pot == s.pot and got.stacks == s.stacks and got.to_call == s.to_call
        assert got.legal_action_ids == s.legal_action_ids
        original, observed = HunlTokenizer(64).encode(s), HunlTokenizer(64).encode(got)
        for key in original:
            torch.testing.assert_close(original[key], observed[key], rtol=0, atol=0)
        old_street = table.street
        if s.street == 1 and not any(e.street == 1 for e in s.events):
            # An off-tree 7.5x-pot bet must preserve exact chips/features.
            history += "b1500"
            table.step_raise_to(1500)
        else:
            choices = [a for a in s.legal_action_ids if a not in (0, 4)]
            action = int(rng.choice(choices)) if s.street == 0 else 1
            sent = api_action(replay, parsed, action)
            table.step(action)
            history += sent
        if not table.is_over() and table.street > old_street:
            history += "/"
        decisions += 1
        assert decisions < 100
    assert decisions >= 8


def test_replay_rejects_wrong_seat_and_future_board():
    with pytest.raises(ValueError, match="hero decision"):
        decision_table({"client_pos": 0, "action": "", "hole_cards": ["As", "Kh"], "board": []})
    with pytest.raises(ValueError, match="card count"):
        decision_table({"client_pos": 1, "action": "", "hole_cards": ["As", "Kh"],
                        "board": ["2c", "3c", "4c"]})

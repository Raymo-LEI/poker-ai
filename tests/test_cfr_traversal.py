"""Analytic external-sampling regrets and HUNL information-set boundaries."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from alphanl.cfr.traversal import ExternalSamplingTraverser, HunlState
from alphanl.transformer.tokenizer import BettingHistoryTokenizer


class _RecordingBuffer:
    def __init__(self):
        self.records = []

    def add(self, batch, regrets, iteration):
        node = int(batch["numeric"][0, 0, 0])
        self.records.append((node, regrets.detach().clone(), iteration))


@dataclass
class _Tree:
    nodes: dict
    node: int = 0
    trace: list | None = None

    def is_terminal(self):
        return "utility" in self.nodes[self.node]

    def current_player(self):
        return self.nodes[self.node]["player"]

    def legal_actions(self):
        return tuple(self.nodes[self.node]["children"])

    def observation(self):
        batch = BettingHistoryTokenizer(max_seq_len=8).blank()
        batch["numeric"][0, 0, 0] = self.node
        batch["legal_mask"].zero_()
        batch["legal_mask"][0, list(self.legal_actions())] = 1
        return batch

    def child(self, action):
        if self.trace is not None:
            self.trace.append((self.node, action))
        return _Tree(self.nodes, self.nodes[self.node]["children"][action], self.trace)

    def returns(self):
        value = self.nodes[self.node]["utility"]
        return (value, -value)


class _Policy(nn.Module):
    def __init__(self, strategies):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.strategies = strategies
        self.batch_sizes = []

    def strategy(self, batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.batch_sizes.append(batch["token_ids"].shape[0])
        return torch.tensor([
            self.strategies[int(node)] for node in batch["numeric"][:, 0, 0]
        ], device=self.anchor.device, dtype=torch.float32)


def _analytic_tree(traverser=0):
    sign = 1 if traverser == 0 else -1
    nodes = {
        0: {"player": traverser, "children": {0: 1, 1: 2, 4: 5}},
        1: {"utility": sign * -2.0},
        2: {"player": traverser, "children": {1: 3, 3: 4}},
        3: {"utility": sign * 4.0},
        4: {"utility": sign * 0.0},
        5: {"player": 1 - traverser, "children": {0: 6, 3: 7}},
        6: {"utility": sign * -100.0},
        7: {"utility": sign * 6.0},
    }
    strategies = {
        0: [0.0, 0.25, 0.0, 0.0, 0.75],
        2: [0.0, 0.25, 0.0, 0.75, 0.0],
        5: [0.0, 0.0, 0.0, 1.0, 0.0],
    }
    return _Tree(nodes, trace=[]), {0: _Policy(strategies), 1: _Policy(strategies)}


@pytest.mark.parametrize("player", [0, 1])
def test_exact_signed_regrets_enumerate_zero_policy_actions_and_ignore_own_reach(player):
    tree, policies = _analytic_tree(player)
    policies[1].eval()
    buffer = _RecordingBuffer()
    stats = ExternalSamplingTraverser(seed=13).collect(
        [tree], traverser=player, iteration=7, policies=policies, buffer=buffer,
    )
    samples = {node: regrets[0] for node, regrets, _ in buffer.records}
    # The second own node has value .25*4 + .75*0 = 1. Its regret
    # is not multiplied by the root's .25 probability of reaching it.
    assert torch.equal(samples[2], torch.tensor([0.0, 3.0, 0.0, -1.0, 0.0]))
    # Root value is 0*(-2) + .25*1 + .75*6 = 4.75.
    assert torch.equal(samples[0], torch.tensor([-6.75, -3.75, 0.0, 0.0, 1.25]))
    assert all(iteration == 7 for _, _, iteration in buffer.records)
    assert (0, 0) in tree.trace  # zero-probability traverser action is explored
    assert (5, 3) in tree.trace and (5, 0) not in tree.trace
    assert stats.nodes == 7 and stats.terminal_nodes == 4
    assert stats.regret_samples == 2
    assert stats.inference_rows == stats.inference_calls == 3
    assert policies[0].training and not policies[1].training


def test_vectorized_workers_reduce_forwards_without_changing_regrets():
    roots = [_analytic_tree()[0] for _ in range(8)]
    _, serial_policies = _analytic_tree()
    _, batched_policies = _analytic_tree()
    serial_buffer, batched_buffer = _RecordingBuffer(), _RecordingBuffer()
    serial = ExternalSamplingTraverser(batch_size=1).collect(
        roots, traverser=0, iteration=1, policies=serial_policies, buffer=serial_buffer,
    )
    batched = ExternalSamplingTraverser(batch_size=4).collect(
        roots, traverser=0, iteration=1, policies=batched_policies, buffer=batched_buffer,
    )
    assert serial.inference_rows == batched.inference_rows == 24
    assert serial.inference_calls == 24 and batched.inference_calls == 6
    assert max(batched_policies[0].batch_sizes) == 4
    for node in (0, 2):
        lhs = [r for n, r, _ in serial_buffer.records if n == node]
        rhs = [r for n, r, _ in batched_buffer.records if n == node]
        assert len(lhs) == len(rhs) == 8
        assert all(torch.equal(a, b) for a, b in zip(lhs, rhs))


def test_opponent_stochastic_sampling_is_seeded_and_visits_only_one_child():
    tree, policies = _analytic_tree()
    # Start at an opponent decision whose children are terminal.
    tree.node = 5
    for policy in policies.values():
        policy.strategies[5] = [0.3, 0.0, 0.0, 0.7, 0.0]
    roots_a = [_Tree(tree.nodes, 5, []) for _ in range(30)]
    roots_b = [_Tree(tree.nodes, 5, []) for _ in range(30)]
    buffers = [_RecordingBuffer(), _RecordingBuffer()]
    for roots, buffer in zip((roots_a, roots_b), buffers):
        stats = ExternalSamplingTraverser(seed=24, batch_size=6).collect(
            roots, traverser=0, iteration=1, policies=policies, buffer=buffer,
        )
        assert stats.terminal_nodes == 30 and stats.regret_samples == 0
        assert not buffer.records
    choices_a = [root.trace[0][1] for root in roots_a]
    choices_b = [root.trace[0][1] for root in roots_b]
    assert choices_a == choices_b
    assert set(choices_a) == {0, 3}
    assert all(len(root.trace) == 1 for root in roots_a)


@pytest.mark.parametrize("bad_probs,message", [
    ([0.2] * 5, "illegal action"),
    ([0.0, 0.1, 0.0, 0.0, 0.1], "invalid strategy"),
    ([0.0, float("nan"), 0.0, 0.0, 1.0], "invalid strategy"),
])
def test_invalid_policy_rejected_and_modes_restored(bad_probs, message):
    tree, policies = _analytic_tree()
    policies[0].strategies[0] = bad_probs
    policies[1].eval()
    with pytest.raises(ValueError, match=message):
        ExternalSamplingTraverser().collect(
            [tree], traverser=0, iteration=1, policies=policies, buffer=_RecordingBuffer(),
        )
    assert policies[0].training and not policies[1].training


def test_mismatched_environment_mask_is_rejected_before_inference():
    class BadMask(_Tree):
        def observation(self):
            batch = super().observation()
            batch["legal_mask"][0, 2] = 1
            return batch

    tree, policies = _analytic_tree()
    with pytest.raises(ValueError, match="match its observation mask"):
        ExternalSamplingTraverser().collect(
            [BadMask(tree.nodes)], traverser=0, iteration=1, policies=policies,
            buffer=_RecordingBuffer(),
        )
    assert policies[0].batch_sizes == policies[1].batch_sizes == []


def test_node_budget_fails_instead_of_inventing_a_truncated_value():
    tree, policies = _analytic_tree()
    with pytest.raises(RuntimeError, match="no truncated utility"):
        ExternalSamplingTraverser(max_nodes=1).collect(
            [tree], traverser=0, iteration=1, policies=policies, buffer=_RecordingBuffer(),
        )
    assert policies[0].training and policies[1].training


def test_hunl_infoset_hides_opponent_cards_and_future_board():
    from alphanl.transformer.hunl import HunlTable, HunlTokenizer
    from alphanl.transformer.tables import aligned_decks

    # Same acting SB hand, different BB hand and entire future board.
    _, deck_a = aligned_decks(("C2", "D3"), ("SA", "HK"), ("C4", "D5", "H6", "S7", "C8"))
    _, deck_b = aligned_decks(("C9", "DT"), ("SA", "HK"), ("CJ", "DQ", "H2", "S3", "C5"))
    tables = [HunlTable(seed=1), HunlTable(seed=2)]
    for table, deck in zip(tables, (deck_a, deck_b)):
        table.reset(deck)
    states = [HunlState(table, HunlTokenizer()) for table in tables]
    assert all(torch.equal(states[0].observation()[k], states[1].observation()[k])
               for k in states[0].observation())

    # On the flop BB acts. Its own cards and public flop are the same;
    # opponent holes and the unseen turn/river must not affect its input.
    _, deck_a = aligned_decks(("SA", "HK"), ("C2", "D3"), ("C4", "D5", "H6", "S7", "C8"))
    _, deck_b = aligned_decks(("SA", "HK"), ("C9", "DT"), ("C4", "D5", "H6", "SJ", "CQ"))
    for table, deck in zip(tables, (deck_a, deck_b)):
        table.reset(deck)
        table.step(1)
        table.step(1)
        assert table.street == 1 and table.current_player() == 0
    obs_a, obs_b = states[0].observation(), states[1].observation()
    assert all(torch.equal(obs_a[k], obs_b[k]) for k in obs_a)


def test_hunl_child_preserves_parent_chance_and_returns_big_blinds():
    from alphanl.transformer.hunl import HunlTable, HunlTokenizer

    table = HunlTable(seed=7)
    table.reset()
    state = HunlState(table, HunlTokenizer())
    before = state.observation()
    child = state.child(0)
    assert child.is_terminal() and not state.is_terminal()
    assert child.returns() == (0.5, -0.5)
    assert all(torch.equal(before[k], state.observation()[k]) for k in before)
    continuing = state.child(1)
    assert table.hole_cards(0) == continuing.table.hole_cards(0)
    assert table.hole_cards(1) == continuing.table.hole_cards(1)

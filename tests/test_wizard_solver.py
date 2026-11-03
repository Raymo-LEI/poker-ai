"""NumPy subgame solver: blueprint stays put unless a leaf head is present."""
import numpy as np

from alphanl.stage2.wizard import bind, fuse


class _Event:
    def __init__(self, street, kind, fraction):
        self.actor_is_me = False
        self.street = street
        self.action_type = kind
        self.pot_fraction = fraction


class _Snap:
    hole_cards = ("SA", "HA")
    board_cards = ("DK", "CQ", "HJ", "ST")
    events = (_Event(1, "check", 0.0), _Event(2, "raise", 1.1))
    street = 2
    pot = 1800.0
    stacks = (9000.0, 8200.0)
    to_call = 600.0
    legal_action_ids = (0, 1, 2, 3, 4)
    big_blind = 100.0
    me_stack = 9000.0
    opp_stack = 8200.0


class _Ctx:
    def __init__(self):
        from alphanl.stage2.wizard import HUD
        self.hands_played = 10
        self.hero_net_bb = 4.0
        self.successes = {name: 1.0 for name, _prior in HUD}
        self.opportunities = {name: 4 for name, _prior in HUD}
        self.trap_successes = {"check_raise_freq": 1.0}
        self.trap_opportunities = {"check_raise_freq": 3}


def _agent(with_leaf):
    agent = type("Agent", (), {})()
    agent.weights = {}
    if with_leaf:
        rng = np.random.default_rng(0)
        agent.weights = {
            "vleaf.0.weight": rng.normal(0, 0.02, (48, 32)).astype(np.float32),
            "vleaf.0.bias": np.zeros(48, np.float32),
            "vleaf.1.weight": rng.normal(0, 0.02, (16, 48)).astype(np.float32),
            "vleaf.1.bias": np.zeros(16, np.float32),
            "vleaf.2.weight": rng.normal(0, 0.02, (6, 16)).astype(np.float32),
            "vleaf.2.bias": np.zeros(6, np.float32),
        }
    return agent


def _bind():
    bind(lambda *cards: (sum(ord(card[0]) for card in cards),))


def test_solver_stays_off_without_a_leaf():
    _bind()
    probs = np.array([0.05, 0.4, 0.25, 0.2, 0.1])
    agent = _agent(False)
    out = fuse(agent, _Snap(), _Ctx(), probs, None)
    assert agent.last_wizard == "wiz=0"
    assert np.allclose(out, probs)


def test_solver_mixes_a_legal_distribution_when_facing_a_bet():
    _bind()
    probs = np.array([0.05, 0.4, 0.25, 0.2, 0.1])
    agent = _agent(True)
    out = fuse(agent, _Snap(), _Ctx(), probs, None)
    assert agent.last_wizard.startswith("wiz=1")
    assert abs(out.sum() - 1) < 1e-6
    assert (out >= 0).all()


def test_small_preflop_pots_use_the_blueprint():
    _bind()
    snap = _Snap()
    snap.street = 0
    snap.pot = 150.0
    snap.to_call = 0.0
    probs = np.array([0.0, 0.5, 0.2, 0.2, 0.1])
    agent = _agent(True)
    out = fuse(agent, snap, _Ctx(), probs, None)
    assert agent.last_wizard == "wiz=0"
    assert np.allclose(out, probs)

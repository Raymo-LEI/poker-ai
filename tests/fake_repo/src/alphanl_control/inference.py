"""Test double for AlphaNLHoldemAgent: picks a random legal abstract action (no TF needed)."""
import random
class AlphaNLHoldemAgent:
    def __init__(self, weights_path=None, **kw): print("fake model loaded")  # stdout noise on purpose
    def make_action(self, obs):
        assert obs["card_info"].shape == (4, 13, 6) and obs["action_info"].shape == (4, 5, 25)
        return random.choice([i for i, v in enumerate(obs["legal_moves"]) if v])

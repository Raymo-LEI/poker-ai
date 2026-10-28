"""One opponent identity per complete match; smoothed PFSP and nonzero floors."""
import math
import numpy as np
from alphanl.cfr.evaluation import RuleActor, CFRActor

DEFAULT_WEIGHTS = {"equity": .35, "nit": .15, "call": .15, "maniac": .10, "overbet": .10, "history": .15}

class MatchOpponentPool:
    def __init__(self, strategies, *, seed=1, weights=None, pfsp_power=2., uniform_mix=.2, equity_samples=64):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if (set(self.weights) != set(DEFAULT_WEIGHTS) or any(not math.isfinite(w) or w <= 0 for w in self.weights.values())
                or not math.isclose(sum(self.weights.values()), 1.) or not math.isfinite(pfsp_power) or pfsp_power < 0
                or not 0 < uniform_mix <= 1):
            raise ValueError("invalid PFSP weights/settings")
        self.strategies, self.rng = strategies, np.random.default_rng(seed)
        self.power, self.mix, self.equity_samples = pfsp_power, uniform_mix, equity_samples
        self.wins, self.matches = dict.fromkeys(self.weights, 0.), dict.fromkeys(self.weights, 0)

    def probabilities(self):
        difficulty = {k: (1 - (self.wins[k] + 1) / (self.matches[k] + 2)) ** self.power for k in self.weights}
        total = sum(self.weights[k] * difficulty[k] for k in self.weights)
        return {k: self.mix * w + (1 - self.mix) * w * difficulty[k] / total for k, w in self.weights.items()}

    def sample(self):
        probabilities = self.probabilities()
        name = str(self.rng.choice(list(probabilities), p=list(probabilities.values())))
        seed = int(self.rng.integers(0, 2**31))
        if name == "history":
            # Pin one historical iteration for all 50 hands, retain same identity.
            from alphanl.cfr.strategy import SnapshotPolicy
            archive = SnapshotPolicy(self.strategies, seed=seed)
            iteration = int(self.rng.choice(archive.iterations))
            actor = CFRActor(self.strategies, seed=seed, iterations=[iteration])
        else:
            actor = RuleActor(name, seed=seed, equity_samples=self.equity_samples)
        return name, actor

    def record(self, name, hero_net_bb):
        if name not in self.weights or not math.isfinite(hero_net_bb):
            raise ValueError("invalid match result")
        self.matches[name] += 1
        self.wins[name] += .5 if hero_net_bb == 0 else float(hero_net_bb > 0)

    def state_dict(self):
        return {"wins": self.wins, "matches": self.matches, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if set(state["wins"]) != set(self.weights) or set(state["matches"]) != set(self.weights):
            raise ValueError("incompatible PFSP state")
        self.wins, self.matches = dict(state["wins"]), dict(state["matches"])
        self.rng.bit_generator.state = state["rng"]

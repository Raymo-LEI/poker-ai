"""Three ladder-style opponents seen in Botzone replays."""
import numpy as np

from alphanl.cfr.pro_bots import raises
from alphanl.stage2.deployment_guard import strong_made_hand
from alphanl.transformer.actor_actions import PotFractionBet
from alphanl.transformer.equity import estimated_equity

SLAYER_NAMES = ("Bot17_Probe3Bet", "NearShove_19949", "CheckShove_Trapper")


class SlayerActor(object):
    def __init__(self, name, *, seed, equity_samples=16):
        if name not in SLAYER_NAMES:
            raise ValueError("unknown ladder slayer %s" % name)
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.equity_samples = int(equity_samples)
        self.variant = "ladder_slayer_v1"

    def begin_hand(self, seat):
        return None

    def act(self, snapshot):
        if self.name == "Bot17_Probe3Bet":
            return self._probe(snapshot)
        if self.name == "NearShove_19949":
            return self._near(snapshot)
        return self._trap(snapshot)

    def _prefer(self, snapshot, *choices):
        legal = snapshot.legal_action_ids
        return next(action for action in choices if action in legal)

    def _equity(self, snapshot):
        return estimated_equity(snapshot, samples=self.equity_samples, seed=17)

    def _leave_one(self, snapshot):
        pot = max(float(snapshot.pot), 1.0)
        owed = float(snapshot.to_call)
        extra = max(float(snapshot.me_stack) - 1.0 - owed, 1.0)
        return PotFractionBet(extra / pot)

    def _probe(self, snapshot):
        if snapshot.street == 0:
            seen = len(raises(snapshot.events))
            if seen >= 2 and self.rng.random() < 0.75:
                return self._prefer(snapshot, 4, 3, 2, 1, 0)
            if self.rng.random() < (0.80 if seen == 0 else 0.65):
                return self._prefer(snapshot, 3, 2, 4, 1, 0)
            return self._prefer(snapshot, 1, 0, 2, 3, 4)
        facing_big = snapshot.to_call > 0 and snapshot.to_call >= 0.6 * max(snapshot.pot, 1.0)
        if facing_big:
            if strong_made_hand(snapshot) or self._equity(snapshot) >= 0.75:
                return self._prefer(snapshot, 1, 4, 0)
            return self._prefer(snapshot, 0, 1)
        if snapshot.street >= 2 and snapshot.to_call <= 0 and self.rng.random() < 0.72:
            return PotFractionBet(0.27)
        return self._prefer(snapshot, 1, 0, 2, 3, 4)

    def _near(self, snapshot):
        if self.rng.random() < 0.70 and snapshot.me_stack > 2:
            return self._leave_one(snapshot)
        return self._prefer(snapshot, 1, 0, 4, 2, 3)

    def _trap(self, snapshot):
        value = strong_made_hand(snapshot) or self._equity(snapshot) >= 0.72
        if snapshot.street <= 1:
            if snapshot.to_call > 0:
                return self._prefer(snapshot, 1, 0)
            return self._prefer(snapshot, 1, 0, 2)
        if value or self.rng.random() < 0.18:
            if self.rng.random() < 0.55:
                return self._prefer(snapshot, 4, 3, 2, 1, 0)
            return PotFractionBet(1.5)
        if snapshot.to_call > 0:
            return self._prefer(snapshot, 0, 1)
        return self._prefer(snapshot, 1, 0, 2)

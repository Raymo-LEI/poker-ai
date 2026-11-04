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
        self.variant = "bot17_public_size_v2" if name=="Bot17_Probe3Bet" else "ladder_slayer_v1"

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
        """Size-aware approximation of the three audited public bot17 replays.

        Decisions use only this actor's cards and public state. Thresholds are
        observations from a small sample, not recovered proprietary source.
        PotFractionBet adds a fraction AFTER calling, so convert chip targets.
        """
        equity=self._equity(snapshot)
        blind=float(snapshot.big_blind)
        def wager(added):
            added=min(float(snapshot.me_stack),float(added))
            if added<=snapshot.to_call:return self._prefer(snapshot,1,0)
            if added>=snapshot.me_stack:return self._prefer(snapshot,4,1,0)
            return PotFractionBet((added-snapshot.to_call)/max(snapshot.pot,1.))
        if snapshot.street == 0:
            seen = len(raises(snapshot.events))
            if snapshot.me_seat==snapshot.sb_seat and seen==0:
                if equity<.38:return self._prefer(snapshot,0,1)
                if equity<.43:return self._prefer(snapshot,1,0)
                if equity>=.586:added=291.
                else:added=float(np.interp(equity,[.43,.48,.50,.58,.586],[210.,233.,235.,285.,291.]))
                return wager(round(added)*blind/100.)
            if snapshot.to_call==0 and seen==0:
                if equity<.50:return self._prefer(snapshot,1,0)
                return wager(round(float(np.interp(equity,[.50,.62],[220.,292.])))*blind/100.)
            if equity>=.68 and any(a in snapshot.legal_action_ids for a in (2,3,4)):
                return wager(max(3*snapshot.to_call,snapshot.pot))
            if equity>=max(.38,snapshot.to_call/(snapshot.pot+snapshot.to_call)+.12):
                return self._prefer(snapshot,1,0)
            return self._prefer(snapshot, 0, 1)
        from alphanl.stage2.portable_evaluator import strength
        from alphanl.stage2.cfv import card
        category=strength([card(c) for c in snapshot.hole_cards+snapshot.board_cards])[0]
        if snapshot.to_call>0:
            if snapshot.street>=2 and category>=3 and strong_made_hand(snapshot) and any(a in snapshot.legal_action_ids for a in (2,3,4)):
                if snapshot.street==3 and category>=6:return self._prefer(snapshot,4,3,1,0)
                return wager(max(2*snapshot.to_call,2.18*snapshot.pot))
            # Rare off-style bluffs remain possible; the audit includes one.
            if self.rng.random()<.02 and any(a in snapshot.legal_action_ids for a in (2,3,4)):
                return wager(2.09*snapshot.pot)
            if equity>=max(.42,snapshot.to_call/(snapshot.pot+snapshot.to_call)+.10):return self._prefer(snapshot,1,0)
            return self._prefer(snapshot,0,1)
        if category>=3 and strong_made_hand(snapshot) and equity>=.70:return wager(snapshot.pot*(1.5 if snapshot.street>=2 else 1.))
        fraction=.5 if snapshot.pot<=2*blind else float(self.rng.uniform(.25,.34))
        return wager(max(blind,fraction*snapshot.pot))

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

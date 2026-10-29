"""Four-tier match pool: exact defensive quotas and rolling weakness priority."""
from collections import deque, OrderedDict
import copy
import math
import numpy as np
import torch
from alphanl.cfr.strategy import SnapshotPolicy
from alphanl.cfr.distillation import prepare_archive, SEMANTIC_FIELDS
from .ladder_bots import LADDER_OPPONENTS, JitteredLadderReplicaActor
from .opponents import JitteredRuleActor, NeuralMatchActor
from alphanl.cfr.pro_bots import PRO_OPPONENTS, JitteredProScriptActor

DEFAULT_WEIGHTS = {"equity": .35, "nit": .12, "call": .06, "maniac": .035,
                   "overbet": .035, "push_fold": .05, "reference": .10, "history": .15, "self": .10}
DEFENSE = {"reference": .10, "history": .15, "self": .10}
PRO_WEIGHTS = {**{k: v if k in DEFENSE else v * (.35/.65) for k,v in DEFAULT_WEIGHTS.items()},
               "TAG_Pro": .15, "Trapper_CR": .15}
# Initial ladder probability is 20%. Previous exploit families keep the other
# 45%, scaled from the pro-hunter mix. PFSP may reweight that 65%. Defense
# quotas stay the fixed 10/15/10 block.
_LADDER_SHARE = {"OffTree_Aggro_Boss": .08, "Check_Shove_Trapper": .06, "Flop_Raise_War_Bot": .06}
_NATIVE_SCALE = (sum(v for k, v in PRO_WEIGHTS.items() if k not in DEFENSE) - sum(_LADDER_SHARE.values())) / sum(
    v for k, v in PRO_WEIGHTS.items() if k not in DEFENSE)
NATIVE_WEIGHTS = {k: (v if k in DEFENSE else v * _NATIVE_SCALE) for k, v in PRO_WEIGHTS.items()}
NATIVE_WEIGHTS.update(_LADDER_SHARE)
DEFAULT_BUCKETS = ((140, 160), (240, 260), (320, 350), (100, 350))
FORMAT = "stage2_matchpool_v2"
_ALLOWED_POOLS = (frozenset(DEFAULT_WEIGHTS), frozenset(PRO_WEIGHTS), frozenset(NATIVE_WEIGHTS))


class MatchOpponentPool:
    def __init__(self, strategies, *, seed=1, weights=None, equity_samples=64,
                 window_size=100, history_cache_size=16, history_refresh_matches=100,
                 history_buckets=None, self_snapshot_every=25, self_history_size=16):
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if (frozenset(self.weights) not in _ALLOWED_POOLS
                or any(not math.isfinite(w) or w <= 0 for w in self.weights.values())
                or not math.isclose(sum(self.weights.values()), 1.)
                or any(not math.isclose(self.weights[k], v) for k, v in DEFENSE.items())):
            raise ValueError("pool weights must retain 10% reference, 15% history, 10% self")
        for value in (equity_samples, window_size, history_refresh_matches, self_snapshot_every, self_history_size):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("pool counters must be positive integers")
        if not 8 <= history_cache_size <= 16 or history_refresh_matches % 20:
            raise ValueError("history cache must hold 8..16 models; refresh must align with 20 matches")
        self.strategies, self.rng = strategies, np.random.default_rng(seed)
        self.equity_samples, self.window_size = equity_samples, window_size
        self.history_cache_size, self.history_refresh_matches = history_cache_size, history_refresh_matches
        self.history_buckets = [list(b) for b in (history_buckets or DEFAULT_BUCKETS)]
        if (len(self.history_buckets) > history_cache_size or any(len(b) != 2 or not 100 <= b[0] <= b[1] <= 350
                                                               for b in self.history_buckets)):
            raise ValueError("history buckets must lie in [100,350]")
        self.self_snapshot_every, self.self_history_size = self_snapshot_every, self_history_size
        self.wins, self.matches = dict.fromkeys(self.weights, 0.), dict.fromkeys(self.weights, 0)
        self.results = {k: deque(maxlen=window_size) for k in self.weights}
        self.schedule, self.sampled = [], 0
        self.resident_ids, self.resident_buckets = [], []
        self.history_models, self.self_models = {}, OrderedDict()
        self.self_weights = OrderedDict()
        self.template = None
        self.disk_loads = 0
        self.contract = {"weights": self.weights, "window_size": window_size, "equity_samples": equity_samples,
                         "history_cache_size": history_cache_size, "history_refresh_matches": history_refresh_matches,
                         "history_buckets": self.history_buckets, "self_snapshot_every": self_snapshot_every,
                         "self_history_size": self_history_size}

    def initialize(self, learner):
        """Validate archive once; keep immutable resident models on rollout device."""
        config, self.manifest = prepare_archive(self.strategies, after=99, through=max(b[1] for b in self.history_buckets))
        if any(getattr(config, k) != getattr(learner.config, k) for k in SEMANTIC_FIELDS):
            raise ValueError("history opponents and reference have different Stage 1 contracts")
        available = set(self.manifest["iterations"])
        # The last broad bucket is disjoint from the three priority periods.
        used, self.buckets = set(), []
        for low, high in self.history_buckets:
            bucket = sorted(t for t in available if low <= t <= high and t not in used)
            if not bucket:
                raise ValueError(f"empty required history bucket [{low},{high}]")
            self.buckets.append(bucket)
            used.update(bucket)
        self.template = learner
        self.reference = learner.reference
        self.device = next(learner.parameters()).device
        if not self.self_weights:
            self.add_self_snapshot(learner, update=0)  # Frozen initial policy, including an explicit warm start.
        if self.resident_ids:
            self._load_residents()
        else:
            self._rotate_history()

    def rolling_metrics(self):
        return {k: {"matches": len(rows), "mbb_per_hand": sum(rows) / len(rows) / 50 * 1000 if rows else None,
                    "match_win": sum(float(x > 0) for x in rows) / len(rows) if rows else None}
                for k, rows in self.results.items()}

    def probabilities(self):
        metrics = self.rolling_metrics()
        priorities = {k: (3. if metrics[k]["matches"] and (metrics[k]["mbb_per_hand"] < 500
                         or metrics[k]["match_win"] < .60) else 1.) for k in self.weights if k not in DEFENSE}
        total = sum(self.weights[k] * p for k, p in priorities.items())
        return {k: DEFENSE[k] if k in DEFENSE else .65 * self.weights[k] * priorities[k] / total
                for k in self.weights}

    def _rotate_history(self):
        selected = [[] for _ in self.buckets]
        for _ in range(self.history_cache_size):
            candidates = [i for i, b in enumerate(self.buckets) if len(selected[i]) < len(b)]
            if not candidates:
                break
            i = min(candidates, key=lambda i: len(selected[i]))
            choices = sorted(set(self.buckets[i]) - set(selected[i]))
            selected[i].append(int(self.rng.choice(choices)))
        self.resident_buckets = selected
        self.resident_ids = [t for bucket in selected for t in bucket]
        self._load_residents()

    def _load_residents(self):
        previous, self.history_models = self.history_models, {}
        for t in self.resident_ids:
            if t in previous:
                self.history_models[t] = previous[t]
                continue
            # Cache misses must not change the learner's action RNG stream.
            with torch.random.fork_rng(devices=[]):
                archive = SnapshotPolicy(self.strategies, snapshot_iterations=[t], device=self.device)
                archive.begin_hand(0)
            model = archive.model.requires_grad_(False).eval()
            model.config = archive.config
            self.history_models[t] = model
            self.disk_loads += archive.disk_loads

    def add_self_snapshot(self, learner, *, update):
        if update < 0 or update in self.self_weights:
            raise ValueError("self snapshot id must be new and nonnegative")
        self.self_weights[update] = {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()
                                     if not k.startswith("reference.")}
        while len(self.self_weights) > self.self_history_size:
            expired, _ = self.self_weights.popitem(last=False)
            self.self_models.pop(expired, None)

    def _self_model(self, update):
        if update not in self.self_models:
            from .policy import ResidualExploiterPolicy
            with torch.random.fork_rng(devices=[]):
                model = ResidualExploiterPolicy(self.reference, **self.template.settings).to(self.device)
            # Share the permanently frozen backbone across residual snapshots.
            model.reference = self.reference
            state = model.state_dict()
            state.update(self.self_weights[update])
            model.load_state_dict(state, strict=True)
            self.self_models[update] = model.requires_grad_(False).eval()
        return self.self_models[update]

    def sample(self):
        """Called exactly once by env.reset; shuffled 20-slot quota blocks."""
        if self.template is None:
            raise RuntimeError("initialize the pool with the frozen reference before sampling")
        if not self.schedule:
            if self.sampled and self.sampled % self.history_refresh_matches == 0:
                self._rotate_history()
            self.schedule = ["reference"] * 2 + ["history"] * 3 + ["self"] * 2 + ["exploit"] * 13
            self.rng.shuffle(self.schedule)
        name = self.schedule.pop()
        if name == "exploit":
            probabilities = {k: v for k, v in self.probabilities().items() if k not in DEFENSE}
            name = str(self.rng.choice(list(probabilities), p=np.array(list(probabilities.values())) / .65))
        seed = int(self.rng.integers(0, 2**31))
        self.sampled += 1
        if name == "reference":
            actor = NeuralMatchActor(self.reference, kind="reference", snapshot_id="reference", seed=seed)
        elif name == "history":
            bucket = self.resident_buckets[int(self.rng.integers(len(self.resident_buckets)))]
            t = int(self.rng.choice(bucket))
            actor = NeuralMatchActor(self.history_models[t], kind="cfr", snapshot_id=t, seed=seed)
        elif name == "self":
            update = int(self.rng.choice(list(self.self_weights)))
            actor = NeuralMatchActor(self._self_model(update), kind="residual", snapshot_id=update, seed=seed)
        else:
            if name in LADDER_OPPONENTS:
                actor = JitteredLadderReplicaActor(name, seed=seed, equity_samples=self.equity_samples)
            elif name in PRO_OPPONENTS:
                actor = JitteredProScriptActor(name, seed=seed, equity_samples=self.equity_samples)
            else:
                actor = JitteredRuleActor(name, seed=seed, equity_samples=self.equity_samples)
        return name, actor

    def record(self, name, hero_net_bb):
        if name not in self.weights or not math.isfinite(hero_net_bb):
            raise ValueError("invalid match result")
        self.matches[name] += 1
        self.wins[name] += .5 if hero_net_bb == 0 else float(hero_net_bb > 0)
        self.results[name].append(float(hero_net_bb))

    def state_dict(self):
        return {"format": FORMAT, "contract": copy.deepcopy(self.contract), "wins": dict(self.wins),
                "matches": dict(self.matches), "results": {k: list(v) for k, v in self.results.items()},
                "rng": copy.deepcopy(self.rng.bit_generator.state), "schedule": list(self.schedule),
                "sampled": self.sampled, "resident_ids": list(self.resident_ids),
                "resident_buckets": copy.deepcopy(self.resident_buckets), "self_weights": self.self_weights,
                "archive": getattr(self, "manifest", None)}

    def load_state_dict(self, state):
        if state.get("format") != FORMAT or state["contract"] != self.contract:
            raise ValueError("incompatible MatchPool state/settings; start a fresh Stage 2 run")
        if self.template is not None and state["archive"] != self.manifest:
            raise ValueError("resume history archive differs from checkpoint")
        self.wins, self.matches = dict(state["wins"]), dict(state["matches"])
        self.results = {k: deque(v, maxlen=self.window_size) for k, v in state["results"].items()}
        self.rng.bit_generator.state = state["rng"]
        self.schedule, self.sampled = list(state["schedule"]), state["sampled"]
        self.resident_ids, self.resident_buckets = list(state["resident_ids"]), copy.deepcopy(state["resident_buckets"])
        self.self_weights = OrderedDict(state["self_weights"])
        self.self_models = {}
        if self.template is not None:
            self._load_residents()

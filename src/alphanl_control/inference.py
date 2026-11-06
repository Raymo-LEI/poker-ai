"""Load AlphaNLHoldem released weights for greedy action selection (unchanged upstream net)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import ALPHANL_ROOT, DEFAULT_CONFIG, DEFAULT_WEIGHTS


def _ensure_alphanl_on_path() -> None:
    root = str(ALPHANL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_policy_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG
    return eval(path.read_text().strip())  # noqa: S307 — upstream stores config as Python literal


class AlphaNLHoldemAgent:
    """Greedy policy from a released checkpoint (default c_1048.pkl shipped in upstream)."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        variable_scope: str = "oppo_policy",
    ) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        _ensure_alphanl_on_path()

        from ray.rllib.models import ModelCatalog
        from agi.evaluation_tools import NNAgent
        from agi.nl_holdem_env import NlHoldemEnvWrapper
        from agi.nl_holdem_net import NlHoldemNet

        ModelCatalog.register_custom_model("NlHoldemNet", NlHoldemNet)

        self._config = load_policy_config(config_path)
        self._env = NlHoldemEnvWrapper(self._config)
        weights = Path(weights_path or DEFAULT_WEIGHTS)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Missing weights at {weights}. Submodule init should include weights/c_1048.pkl."
            )
        self._agent = NNAgent(
            self._env.observation_space,
            self._env.action_space,
            self._config,
            str(weights),
            variable_scope,
        )
        self._inner_env = self._env

    @property
    def env(self):
        """NlHoldemEnvWrapper instance (rlcard-backed 50bb HU env)."""
        return self._inner_env

    def make_action(self, obs: Dict[str, Any]) -> int:
        return int(self._agent.make_action(obs))

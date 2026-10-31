"""Thin request builder for the public GTO Wizard researcher API.

The fixture command does not construct ``RequestsTransport``. There is no
retry: a state-changing ``/act`` must not be repeated after an ambiguous
failure. The API key stays on the header and is not copied into errors.
Only ``POST /hands`` and ``POST /hands/{id}/act`` are allowed. Leaderboard,
results, and winnings routes are refused here.
"""

from __future__ import annotations

import re
from typing import Mapping

from .schema import API_URL, GAME_NAME

_HAND_PATH = re.compile(r"^/hands(?:/[0-9]+/act)?$")


class ApiError(RuntimeError):
    """The researcher API was not called, or the call failed closed."""


def live_refusal(api_key: str | None) -> str:
    """Reason string for ``--live``. Never includes the key."""

    if api_key is None or not str(api_key).strip():
        return "missing_api_key"
    return "live_not_a_gate"


class GtoWizardClient:
    def __init__(self, transport, api_key: str, *, game_name: str = GAME_NAME) -> None:
        if not str(api_key).strip():
            raise ApiError("missing api key")
        if game_name != GAME_NAME:
            raise ApiError("only HUNL 200BB is supported")
        self.transport = transport
        self._api_key = api_key
        self.game_name = game_name

    def __repr__(self) -> str:
        return f"GtoWizardClient(game_name={self.game_name!r})"

    def _post(self, path: str, payload: Mapping) -> dict:
        if _HAND_PATH.fullmatch(path) is None:
            raise ApiError("refusing endpoint outside /hands and /act")
        return self.transport.post(path, dict(payload), headers={"X-API-Key": self._api_key})

    def new_hand(self) -> dict:
        return self._post("/hands", {"game_name": self.game_name})

    def act(self, hand_id: int, action: str, amount: int | None = None) -> dict:
        if action not in ("f", "k", "c", "b"):
            raise ApiError("action must be f, k, c, or b")
        body: dict = {"action": action}
        if action == "b":
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise ApiError("bet amount is required")
            body["amount"] = amount
        elif amount is not None:
            raise ApiError("amount is only valid for a bet")
        return self._post(f"/hands/{int(hand_id)}/act", body)


class RequestsTransport:
    """HTTP transport. Not used by the fixture CLI."""

    def __init__(self, api_key: str, *, base_url: str = API_URL, timeout=(10, 30)) -> None:
        if not str(api_key).strip():
            raise ApiError("missing api key")
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key

    def __repr__(self) -> str:
        return f"RequestsTransport(base_url={self.base_url!r})"

    def post(self, path: str, payload: Mapping, headers: Mapping | None = None) -> dict:
        import requests

        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                json=dict(payload),
                headers=dict(headers or {}),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError("researcher API request failed: " + type(exc).__name__) from None
        if response.status_code >= 400:
            raise ApiError(f"researcher API HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise ApiError("researcher API returned non-JSON") from None
        if not isinstance(body, dict):
            raise ApiError("researcher API returned a non-object")
        return body

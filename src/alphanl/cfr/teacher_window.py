"""Teacher iteration window loaded from a manifest, not a hardcoded T range.

The historical distillation window is iterations 150 through 350
(``after_exclusive=149``, ``through=350``). Callers pass that manifest into
``prepare_archive``. A missing snapshot directory is ``MISSING``; a zero
own-reach infoset is ``UNREACHABLE/OOD``. Neither case is filled with an
unweighted average of whatever profiles happen to be nearby.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


FORMAT = "route1_r03_teacher_window_v1"
REQUIRED = ("after_exclusive", "through", "stride", "sampling_power")


def load_teacher_window(path: Path | str) -> dict:
    """Read and check a teacher-window manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"teacher window format must be {FORMAT}, got {payload.get('format')!r}")
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"teacher window missing {missing}")
    after = payload["after_exclusive"]
    through = payload["through"]
    stride = payload["stride"]
    power = payload["sampling_power"]
    if isinstance(after, bool) or not isinstance(after, int) or after < 0:
        raise ValueError("after_exclusive must be a nonnegative integer")
    if isinstance(through, bool) or not isinstance(through, int) or through <= after:
        raise ValueError("through must be an integer greater than after_exclusive")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if power not in (1, 1.0, 2, 2.0):
        raise ValueError("sampling_power must be 1 or 2")
    expected = list(range(after + 1, through + 1, stride))
    if not expected:
        raise ValueError("teacher window selects no iterations")
    window = dict(payload)
    window["inclusive_iterations"] = expected
    window["unweighted_fill"] = False
    return window


def archive_kwargs(window: Mapping) -> dict:
    """Keyword arguments for ``prepare_archive`` taken only from the manifest."""

    return {
        "after": int(window["after_exclusive"]),
        "through": int(window["through"]),
        "stride": int(window["stride"]),
        "sampling_power": float(window["sampling_power"]),
    }


def assert_window_matches(manifest: Mapping, window: Mapping) -> None:
    """Refuse a shorter or shifted archive instead of relabeling it."""

    selection = manifest["selection"]
    expected = archive_kwargs(window)
    actual = {
        "after": int(selection["after_exclusive"]),
        "through": int(selection["through"]),
        "stride": int(selection["stride"]),
        "sampling_power": float(manifest["sampling_power"]),
    }
    if actual != expected or list(manifest["iterations"]) != list(window["inclusive_iterations"]):
        raise ValueError(
            "teacher archive does not match the window manifest; "
            "refusing to fill missing iterations with an unweighted mixture. "
            "status=MISSING"
        )


def missing_teacher_record(window: Mapping, reason: str) -> dict:
    """Explicit missing-teacher label. No fabricated probabilities."""

    return {
        "status": "MISSING",
        "reason": reason,
        "target_probs": None,
        "own_reach_mass": None,
        "unweighted_fill": False,
        "window": {
            "after_exclusive": int(window["after_exclusive"]),
            "through": int(window["through"]),
            "stride": int(window["stride"]),
            "inclusive_iterations": list(window["inclusive_iterations"]),
        },
    }

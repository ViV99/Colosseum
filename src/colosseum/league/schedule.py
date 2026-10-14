"""Share and weight schedules (spec block 5).

A schedule value is a number or a piecewise-linear function of the run's global env steps given
by points ``{step: value}``: linear between consecutive points, constant before the first and
after the last. Keys are non-negative integers; strings such as ``"1e6"`` or ``"2_000_000"`` are
accepted (YAML keeps ``1e6`` as a string). Values are finite and >= 0 (shares and anchor weights).
"""

from __future__ import annotations

import bisect
import math
from typing import Any

ScheduleValue = float | dict[int, float]


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what}: {value!r} is not finite")
    if number < 0:
        raise ValueError(f"{what}: {value!r} must be >= 0")
    return number


def _step(key: Any, what: str) -> int:
    if isinstance(key, bool):
        raise ValueError(f"{what}: schedule step {key!r} is not a number")
    if isinstance(key, str):
        try:
            number = float(key.strip().replace("_", ""))
        except ValueError:
            raise ValueError(f"{what}: schedule step {key!r} is not a number") from None
    elif isinstance(key, (int, float)):
        number = float(key)
    else:
        raise ValueError(f"{what}: schedule step {key!r} is not a number")
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{what}: schedule step {key!r} must be an integer number of env steps")
    if number < 0:
        raise ValueError(f"{what}: schedule step {key!r} must be non-negative")
    return int(number)


def parse_schedule(value: Any, what: str = "schedule") -> ScheduleValue:
    """Normalize a raw number or ``{step: value}`` mapping (points sorted by step); ValueError naming ``what``."""
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{what}: an empty schedule; give a number or {{step: value}} points")
        points: dict[int, float] = {}
        for key, item in value.items():
            step = _step(key, what)
            if step in points:
                raise ValueError(f"{what}: schedule step {step} is given twice")
            points[step] = _number(item, f"{what} at step {step}")
        return dict(sorted(points.items()))
    return _number(value, what)


def schedule_value(value: ScheduleValue, env_steps: int) -> float:
    """The value at ``env_steps`` (piecewise linear, constant outside the points)."""
    if not isinstance(value, dict):
        return float(value)
    steps = list(value)                      # sorted by parse_schedule
    if env_steps <= steps[0]:
        return float(value[steps[0]])
    if env_steps >= steps[-1]:
        return float(value[steps[-1]])
    i = bisect.bisect_right(steps, env_steps)
    a, b = steps[i - 1], steps[i]
    return float(value[a] + (value[b] - value[a]) * (env_steps - a) / (b - a))


def schedule_points(value: ScheduleValue) -> list[int]:
    """The schedule's steps (``[0]`` for a number)."""
    return sorted(value) if isinstance(value, dict) else [0]

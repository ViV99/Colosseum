"""Per-interval aggregation of match results and system throughput."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from colosseum.core.types import LATEST_NETWORK_ID

OPPONENT_TYPES = ("latest", "past", "arena")


def opponent_type(result, seat) -> str | None:
    """'arena' (another agent present), 'past' (own checkpoint present), 'latest', or None (solo)."""
    others = [s for s in result.seats if s is not seat]
    if not others:
        return None
    if any(s.agent_id != seat.agent_id for s in others):
        return "arena"
    if any(s.network_id != LATEST_NETWORK_ID for s in others):
        return "past"
    return "latest"


class EpisodeAggregator:
    """Per agent, over the seats that play the agent's latest weights:
    mean return, mean length, W/D/L by opponent type, seat counts."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._lengths: dict[str, list[int]] = defaultdict(list)
        self._wdl: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {t: [0, 0, 0] for t in OPPONENT_TYPES})
        self._seats: dict[str, list[int]] = defaultdict(list)

    def add(self, result) -> None:
        for seat in result.seats:
            if seat.network_id != LATEST_NETWORK_ID:
                continue
            agent_id = seat.agent_id
            self._returns[agent_id].append(float(seat.reward))
            self._lengths[agent_id].append(int(result.episode_length))
            counts = self._seats[agent_id]
            while len(counts) <= seat.seat:
                counts.append(0)
            counts[seat.seat] += 1
            kind = opponent_type(result, seat)
            if kind is None:
                continue
            best_other = max(s.outcome for s in result.seats if s is not seat)
            idx = 0 if seat.outcome > best_other else (1 if seat.outcome == best_other else 2)
            self._wdl[agent_id][kind][idx] += 1

    def flush(self) -> dict[str, dict[str, Any]]:
        """Stats since the previous flush, per agent; resets the accumulators."""
        out: dict[str, dict[str, Any]] = {}
        for agent_id, returns in self._returns.items():
            out[agent_id] = {
                "episodes": len(returns),
                "return_mean": float(np.mean(returns)),
                "length_mean": float(np.mean(self._lengths[agent_id])),
                "wdl": {t: list(v) for t, v in self._wdl[agent_id].items()},
                "seat_counts": list(self._seats[agent_id]),
            }
        self._reset()
        return out


class SystemStats:
    """Throughput and queue health between two snapshots."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_t = clock()
        self._last_env_steps = 0
        self._train_steps: dict[str, int] = {}
        self._last_train_steps: dict[str, int] = {}
        self._workers: dict[int, dict] = {}

    def on_train_step(self, agent_id: str, train_step: int) -> None:
        self._train_steps[agent_id] = max(int(train_step), self._train_steps.get(agent_id, 0))

    def on_worker_stats(self, stats: dict) -> None:
        self._workers[int(stats["worker_id"])] = dict(stats)

    def snapshot(self, env_steps: int, queue_depths: dict[str, int]) -> dict[str, Any]:
        now = self._clock()
        dt = now - self._last_t

        def rate(delta: float) -> float:
            return delta / dt if dt > 1e-3 else 0.0

        snap = {
            "env_steps": int(env_steps),
            "env_steps_per_sec": rate(env_steps - self._last_env_steps),
            "train_steps_per_sec": {
                a: rate(s - self._last_train_steps.get(a, 0)) for a, s in self._train_steps.items()
            },
            "queue_depths": dict(queue_depths),
            "parked_buffers": int(sum(int(w.get("parked_buffers", 0)) for w in self._workers.values())),
            "workers_reporting": len(self._workers),
        }
        self._last_t = now
        self._last_env_steps = int(env_steps)
        self._last_train_steps = dict(self._train_steps)
        return snap

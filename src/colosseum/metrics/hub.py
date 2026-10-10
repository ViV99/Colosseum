"""MetricsHub: the main process's single sink for train metrics, results and system stats."""

from __future__ import annotations

import numbers
import time
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.jsonl import MetricsWriter, write_json_atomic


def _is_number(value: Any) -> bool:
    """Real numbers, python or numpy scalars; bools are not numbers here."""
    return (isinstance(value, (numbers.Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if _is_number(v)}


def flatten(prefix: str, value: Any) -> dict[str, float]:
    """Nested dicts/lists of numbers -> ``{"prefix/a/b": float}``; non-numbers are dropped."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(f"{prefix}/{k}", v))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten(f"{prefix}/{i}", v))
    elif _is_number(value):
        out[prefix] = float(value)
    return out


def _episodes_row(ep: dict[str, Any]) -> dict[str, Any]:
    """An ``episodes`` record without its W/D/L counts (top level and per role), for WandB.

    W/D/L counts stay in ``metrics.jsonl`` only, as in SP1.
    """
    row = {k: v for k, v in ep.items() if k not in ("wdl", "by_layout")}
    row["by_layout"] = {layout: {role: {k: v for k, v in cell.items() if k != "wdl"} for role, cell in roles.items()}
                        for layout, roles in ep.get("by_layout", {}).items()}
    return row


def wr_vs_past_over_layouts(layouts: Mapping[str, Mapping[str, Any]], agent_id: str) -> float | None:
    """``wr_vs_past`` of ``agent_id`` over all layouts, weighted by each layout's ``past_games``
    (counted latest-vs-checkpoint member pairs, see ``RatingBook``)."""
    total = weight = 0.0
    for table in layouts.values():
        value = table.get("wr_vs_past", {}).get(agent_id)
        games = table.get("past_games", {}).get(agent_id, 0)
        if value is not None and games:
            total += value * games
            weight += games
    return total / weight if weight else None


def _ratings_row(ratings: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ratings for WandB without the PFSP tables: one column per (owner, snapshot) would grow without bound;
    they stay in metrics.jsonl and ratings.json."""
    return {layout: {k: v for k, v in table.items() if k != "pfsp"} for layout, table in ratings.items()}


def wr_arena_over_layouts(layouts: Mapping[str, Mapping[str, Any]], agent_id: str,
                          opponents: Collection[str] | None = None) -> float | None:
    """Win rate of ``agent_id`` against other agents over all layouts, weighted by ``games``
    (counted member pairs per opponent, not matches; see ``RatingBook``). ``opponents`` limits the
    opponents counted (the hub passes the trainable agents: anchors stay out of the arena win rate)."""
    total = games = 0.0
    for table in layouts.values():
        rates = table.get("win_rates", {}).get(agent_id, {})
        for other, n in table.get("games", {}).get(agent_id, {}).items():
            if n and (opponents is None or other in opponents):
                total += rates[other] * n
                games += n
    return total / games if games else None


class MetricsHub:
    """Writes ``train`` records as they arrive (every ``log_interval`` train steps per agent).

    Every ``console_interval_sec`` it also writes ``episodes``, ``system`` and ``ratings``
    records (``ratings``: ``{"env_steps", "layouts"}`` with one table set per layout), rewrites
    ``ratings.json`` with the same content and prints one console line per agent (ratings
    aggregated over layouts). ``initial_env_steps`` / ``initial_train_steps`` are the counters
    a resumed run continues from (rate baselines, see ``SystemStats``). ``agent_ids`` are the trainable
    agents: one console line each, and the only opponents of the console's arena win rate (ruling P11).
    """

    def __init__(self, *, writer: MetricsWriter, ratings_path: str | Path, agent_ids: list[str],
                 total_timesteps: int, log_interval: int, console_interval_sec: float,
                 wandb_logger: Any = None, clock: Callable[[], float] = time.monotonic,
                 initial_env_steps: int = 0, initial_train_steps: dict[str, int] | None = None) -> None:
        self._writer = writer
        self._ratings_path = Path(ratings_path)
        self._agent_ids = list(agent_ids)
        self._log_interval = max(1, int(log_interval))
        self._interval = float(console_interval_sec)
        self._wandb = wandb_logger
        self._clock = clock
        self._episodes = EpisodeAggregator()
        self._system = SystemStats(clock=clock, initial_env_steps=initial_env_steps,
                                   initial_train_steps=initial_train_steps)
        self._console = ConsoleReporter(total_timesteps)
        self._last_tick = clock()
        self._last_train: dict[str, dict[str, float]] = {}
        self._last_logged_step: dict[str, int] = {}
        self._last_returns: dict[str, float] = {}

    def on_train_metrics(self, metrics: dict) -> None:
        agent_id = str(metrics.get("agent_id", "agent_0"))
        step = int(metrics.get("train_step", 0))
        values = {k: v for k, v in _numeric(metrics).items() if k != "train_step"}
        self._system.on_train_step(agent_id, step)
        self._last_train[agent_id] = {"train_step": float(step), **values}
        last = self._last_logged_step.get(agent_id)
        if last is None or step - last >= self._log_interval:
            self._writer.write("train", agent=agent_id, train_step=step, **values)
            if self._wandb is not None:
                self._wandb.log_train(agent_id, values, step)
            self._last_logged_step[agent_id] = step

    def on_worker_stats(self, stats: dict) -> None:
        self._system.on_worker_stats(stats)

    def on_match_result(self, result) -> None:
        self._episodes.add(result)

    def maybe_tick(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int],
                   force: bool = False) -> bool:
        """``ratings`` is ``RatingBook.snapshot()``: ``{layout: {elo, win_rates, ...}}``."""
        now = self._clock()
        if not force and now - self._last_tick < self._interval:
            return False
        self._last_tick = now
        episodes = self._episodes.flush()
        for agent_id, ep in episodes.items():
            self._writer.write("episodes", agent=agent_id, env_steps=int(env_steps), **ep)
            self._last_returns[agent_id] = ep["return_mean"]
        system = self._system.snapshot(env_steps, queue_depths)
        self._writer.write("system", **system)
        self._writer.write("ratings", env_steps=int(env_steps), layouts=ratings)
        write_json_atomic(self._ratings_path, {"env_steps": int(env_steps), "layouts": ratings})
        if self._wandb is not None:
            row = flatten("system", {k: v for k, v in system.items() if k != "env_steps"})
            row.update(flatten("ratings", _ratings_row(ratings)))
            for agent_id, ep in episodes.items():
                row.update(flatten(f"episodes/{agent_id}", _episodes_row(ep)))
            self._wandb.log_global(row, int(env_steps))
        self._report_console(int(env_steps), system, ratings)
        return True

    def _report_console(self, env_steps: int, system: dict, ratings: dict) -> None:
        lines = []
        for agent_id in self._agent_ids:
            train = self._last_train.get(agent_id, {})
            lines.append(self._console.format_line(
                agent_id,
                train_step=int(train.get("train_step", 0)),
                env_steps=env_steps,
                fps=float(system["env_steps_per_sec"]),
                loss=train.get("total_loss"),
                entropy=train.get("entropy"),
                return_mean=self._last_returns.get(agent_id),
                wr_vs_past=wr_vs_past_over_layouts(ratings, agent_id),
                wr_arena=wr_arena_over_layouts(ratings, agent_id, opponents=self._agent_ids),
            ))
        self._console.emit(lines)

    def close(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int]) -> None:
        """Final records and ratings.json, then close the file (also if writing them fails)."""
        try:
            self.maybe_tick(env_steps=env_steps, ratings=ratings, queue_depths=queue_depths, force=True)
        finally:
            self._writer.close()

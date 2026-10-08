"""MetricsHub: the main process's single sink for train metrics, results and system stats."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.jsonl import MetricsWriter, write_json_atomic


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def flatten(prefix: str, value: Any) -> dict[str, float]:
    """Nested dicts/lists of numbers -> ``{"prefix/a/b": float}``; non-numbers are dropped."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(f"{prefix}/{k}", v))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten(f"{prefix}/{i}", v))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


class MetricsHub:
    """Writes ``train`` records as they arrive (every ``log_interval`` train steps per agent).

    Every ``console_interval_sec`` it also writes ``episodes``, ``system`` and ``ratings``
    records, rewrites ``ratings.json`` and prints one console line per agent.
    """

    def __init__(self, *, writer: MetricsWriter, ratings_path: str | Path, agent_ids: list[str],
                 total_timesteps: int, log_interval: int, console_interval_sec: float,
                 wandb_logger: Any = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._writer = writer
        self._ratings_path = Path(ratings_path)
        self._agent_ids = list(agent_ids)
        self._log_interval = max(1, int(log_interval))
        self._interval = float(console_interval_sec)
        self._wandb = wandb_logger
        self._clock = clock
        self._episodes = EpisodeAggregator()
        self._system = SystemStats(clock=clock)
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
        self._writer.write("ratings", env_steps=int(env_steps), **ratings)
        write_json_atomic(self._ratings_path, {"env_steps": int(env_steps), **ratings})
        if self._wandb is not None:
            row = flatten("system", {k: v for k, v in system.items() if k != "env_steps"})
            row.update(flatten("ratings", ratings))
            for agent_id, ep in episodes.items():
                row.update(flatten(f"episodes/{agent_id}", {k: v for k, v in ep.items() if k != "wdl"}))
            self._wandb.log_global(row, int(env_steps))
        self._report_console(int(env_steps), system, ratings)
        return True

    def _report_console(self, env_steps: int, system: dict, ratings: dict) -> None:
        wr_vs_past = ratings.get("wr_vs_past", {})
        win_rates = ratings.get("win_rates", {})
        games = ratings.get("games", {})
        lines = []
        for agent_id in self._agent_ids:
            train = self._last_train.get(agent_id, {})
            n_games = sum(games.get(agent_id, {}).values())
            wr_arena = None
            if n_games:
                wr_arena = sum(win_rates[agent_id][b] * g for b, g in games[agent_id].items()) / n_games
            lines.append(self._console.format_line(
                agent_id,
                train_step=int(train.get("train_step", 0)),
                env_steps=env_steps,
                fps=float(system["env_steps_per_sec"]),
                loss=train.get("total_loss"),
                entropy=train.get("entropy"),
                return_mean=self._last_returns.get(agent_id),
                wr_vs_past=wr_vs_past.get(agent_id),
                wr_arena=wr_arena,
            ))
        self._console.emit(lines)

    def close(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int]) -> None:
        """Final records and ratings.json, then close the file."""
        self.maybe_tick(env_steps=env_steps, ratings=ratings, queue_depths=queue_depths, force=True)
        self._writer.close()

"""Shared checks for the demo games (T8.1, T8.2): example configs and random matches through
the real ``VectorEnv`` + ``MatchRunner`` (every contract check of ``EpisodeTracker`` included)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.registry import validate_config
from colosseum.core.types import Lineup, MatchResult, SeatAssignment
from colosseum.envs.game import GameSpec, MultiAgentEnv
from colosseum.envs.vector import VectorEnv
from colosseum.worker.match_runner import MatchRunner
from game_helpers import RandomPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]


def example_config(name: str) -> Path:
    return REPO_ROOT / "configs" / "examples" / f"{name}.yaml"


def load_example(name: str, overrides: dict[str, Any] | None = None) -> ColosseumConfig:
    return load_config(example_config(name), overrides)


def check_example_config(name: str) -> None:
    """``colosseum validate`` in-process: spec, matchmaking, a few random steps of every layout,
    ``step`` and ``unroll`` of every agent's model. Raises on any problem."""
    validate_config(load_example(name))


class _Pool:
    def __init__(self, models: dict) -> None:
        self._models = models

    def get(self, agent_id: str, network_id: str):
        return self._models.get(agent_id)


class MatchLog:
    """A ``MatchObserver`` that keeps every ``MatchResult`` and counts acts and eliminations."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []
        self.acts = 0
        self.terminated = 0

    def on_act(self, env: int, seat: int, record) -> None:
        self.acts += 1

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        pass

    def on_terminated(self, env: int, seats: list[int]) -> None:
        self.terminated += len(seats)

    def on_episode_end(self, env: int, end) -> None:
        self.results.append(end.result)

    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None:
        pass


def random_lineup(spec: GameSpec, layout: str) -> Lineup:
    """Every seat played by the random player of its role (model key ``random_<role>``)."""
    return Lineup(layout, [SeatAssignment(f"random_{seat.role}") for seat in spec.layouts[layout]])


def random_matches(env_fn: Callable[[], MultiAgentEnv], *, min_episodes: int, num_envs: int = 4, seed: int = 0,
                   max_env_steps: int = 50_000) -> tuple[list[MatchResult], MatchLog]:
    """Uniformly random legal play in every layout (env ``e`` plays layout ``e % L`` of the sorted
    layouts) until ``min_episodes`` episodes have ended."""
    vec = VectorEnv(env_fn, num_envs)
    spec = vec.spec
    layouts = sorted(spec.layouts)
    models = {f"random_{name}": RandomPolicy(role) for name, role in spec.roles.items()}
    log = MatchLog()
    runner = MatchRunner(vec_env=vec, lineups=[random_lineup(spec, layouts[e % len(layouts)]) for e in range(num_envs)],
                         models=_Pool(models), observer=log, seed=seed, context="demo, ")
    steps = 0
    try:
        while len(log.results) < min_episodes:
            steps += runner.step()
            assert steps < max_env_steps, f"only {len(log.results)} episodes after {steps} env steps"
    finally:
        runner.close()
    return log.results, log

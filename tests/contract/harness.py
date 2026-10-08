"""Helpers to drive the real ``RolloutLoop`` in-process (no worker processes)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import Tensor

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk, WeightPayload
from colosseum.worker.rollout_loop import RolloutLoop
from dataflow_helpers import Collected
from dataflow_helpers import make_loop as build_loop
from helpers import CountingEnv, make_simple_model

OBS_DIM = 4
NUM_ACTIONS = 3


def simple_factory(core: str = "none") -> Any:
    """Model factory used by the contract tests: SimpleEncoder -> core -> heads."""
    return make_simple_model(obs_dim=OBS_DIM, hidden_dim=16, num_actions=NUM_ACTIONS, core=core)


def counting_env_fn(num_players: int = 2, episode_length: int = 5) -> Callable[[], CountingEnv]:
    return partial(CountingEnv, num_players=num_players, episode_length=episode_length,
                   num_actions=NUM_ACTIONS, obs_dim=OBS_DIM)


def weights_payload(agent_id: str, model: Any, version: int) -> WeightPayload:
    """Numpy weight payload carrying ``model``'s current parameters."""
    return WeightPayload.from_model(agent_id, version, model)


def make_loop(
    *,
    agent_ids: list[str] | None = None,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    env_fn: Callable[[], Any] | None = None,
    num_envs: int = 2,
    chunk_length: int = 4,
    seed: int = 123,
    initial_weights: dict[str, WeightPayload] | None = None,
    weight_sync_interval: float = 5.0,
    **loop_kwargs: Any,
) -> tuple[RolloutLoop, Collected]:
    """``dataflow_helpers.make_loop`` with the contract tests' defaults.

    ``initial_weights`` are served to the initial sync in ``RolloutLoop.__init__``.
    """
    agent_ids = agent_ids or ["agent_0"]
    col = Collected(weights={aid: [p] for aid, p in (initial_weights or {}).items()})
    return build_loop(
        env_fn or counting_env_fn(), model_factories or simple_factory,
        agent_ids=agent_ids, num_envs=num_envs, chunk_length=chunk_length, collected=col,
        seed=seed, weight_sync_interval=weight_sync_interval, **loop_kwargs,
    )


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, rec: Collected, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step until at least ``n_chunks`` chunks were sent; return the first ``n_chunks``."""
    for _ in range(max_steps):
        if len(rec.chunks) >= n_chunks:
            return rec.chunks[:n_chunks]
        loop.step()
    raise AssertionError(f"only {len(rec.chunks)} chunks after {max_steps} steps")


def step_index(chunk: TrajectoryChunk, episode_length: int = 5) -> list[int]:
    """In-episode step index of every transition (decoded from CountingEnv obs)."""
    return [round(float(x) * episode_length) for x in chunk.observations[:, 0]]


def player_index(chunk: TrajectoryChunk) -> set[int]:
    """Set of player indices whose observations appear in ``chunk``."""
    return {round(float(x)) for x in chunk.observations[:, 1]}


def learner_eval(model: Any, chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Re-evaluate ``chunks`` with a real APPO around ``model``.

    Returns ``(learner_log_probs, learner_values, worker_log_probs, worker_values)``,
    all ``[T*B]`` in time-major order (index ``t*B + b`` for chunk ``b``).
    """
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    lp, v = algo.evaluate_chunks(chunks)
    worker_lp = torch.stack([c.action_log_probs for c in chunks], dim=1).reshape(-1)
    worker_v = torch.stack([c.values for c in chunks], dim=1).reshape(-1)
    return lp, v, worker_lp, worker_v

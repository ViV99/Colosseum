"""Helpers to drive the real ``RolloutLoop`` in-process (no worker processes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
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
    """Weight payload carrying ``model``'s current parameters."""
    return WeightPayload(
        agent_id=agent_id,
        policy_version=version,
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
    )


@dataclass
class LoopRecorder:
    """In-memory endpoints for ``LoopIO``: collects outputs, serves queued inputs."""

    chunks: list[TrajectoryChunk] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    pending_weights: dict[str, WeightPayload] = field(default_factory=dict)
    pending_commands: list[WorkerCommand] = field(default_factory=list)

    def poll_weights(self, agent_id: str) -> WeightPayload | None:
        return self.pending_weights.pop(agent_id, None)

    def poll_command(self) -> WorkerCommand | None:
        return self.pending_commands.pop(0) if self.pending_commands else None

    def io(self) -> LoopIO:
        return LoopIO(
            send_chunk=self.chunks.append,
            poll_weights=self.poll_weights,
            report_result=self.results.append,
            poll_command=self.poll_command,
        )


def make_loop(
    *,
    agent_ids: list[str] | None = None,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    env_fn: Callable[[], Any] | None = None,
    num_envs: int = 2,
    chunk_length: int = 4,
    seed: int = 123,
    initial_weights: dict[str, WeightPayload] | None = None,
    **loop_kwargs: Any,
) -> tuple[RolloutLoop, LoopRecorder]:
    """Build a ``RolloutLoop`` wired to a fresh ``LoopRecorder``."""
    agent_ids = agent_ids or ["agent_0"]
    if model_factories is None:
        model_factories = {aid: simple_factory for aid in agent_ids}
    rec = LoopRecorder()
    if initial_weights:
        rec.pending_weights.update(initial_weights)
    loop = RolloutLoop(
        worker_id=0,
        env_fn=env_fn or counting_env_fn(),
        num_envs=num_envs,
        chunk_length=chunk_length,
        agent_ids=agent_ids,
        model_factories=model_factories,
        io=rec.io(),
        seed=seed,
        **loop_kwargs,
    )
    return loop, rec


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, rec: LoopRecorder, n_chunks: int,
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

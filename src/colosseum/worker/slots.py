"""Per-slot rollout bookkeeping: agent-owned buffers with parking (spec blocks 2-3).

A :class:`RolloutBuffer` holds up to ``chunk_length`` consecutive transitions of
ONE agent. Each collecting (env, seat) slot holds one buffer exclusively, so two
seats of the same agent in one env write to different buffers.

When a slot stops collecting for an agent at an episode boundary (match
re-assignment), its partial buffer is *parked* in the worker's
:class:`BufferPool` for that agent; the next slot that starts collecting for the
agent takes the parked buffer first. Parked buffers always end with a
transition whose ``done`` flag is set, and the model state is reset on
``done``, so a chunk may mix episodes from different envs without any padding
and without losing data (R2-12).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from colosseum.core.types import TrajectoryChunk
from colosseum.networks.state import State, tree_map


class RolloutBuffer:
    """Pre-allocated numpy storage for one chunk of one agent's transitions.

    A transition is *opened* when the slot acts (:meth:`open`), receives rewards
    while it is open (:meth:`add_reward`) and may be marked as the last one of
    its episode (:meth:`mark_done`).
    """

    def __init__(
        self,
        chunk_length: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: Any,
        mask_size: int = 0,
    ) -> None:
        T = chunk_length
        self.chunk_length = chunk_length
        self._obs = np.zeros((T, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((T, *action_shape), dtype=action_dtype)
        self._log_probs = np.zeros(T, dtype=np.float32)
        self._rewards = np.zeros(T, dtype=np.float32)
        self._dones = np.zeros(T, dtype=np.bool_)
        self._values = np.zeros(T, dtype=np.float32)
        self._masks: np.ndarray | None = (
            np.ones((T, mask_size), dtype=np.bool_) if mask_size > 0 else None
        )
        self._has_masks = False
        self._cursor = 0
        self.initial_state: State = None
        self.first_version: int = 0

    # -- state ---------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._cursor

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.chunk_length

    @property
    def last_done(self) -> bool:
        """True if the buffer is empty or its last transition ends an episode."""
        return self._cursor == 0 or bool(self._dones[self._cursor - 1])

    # -- writing -------------------------------------------------------
    def begin_chunk(self, initial_state: State, policy_version: int) -> None:
        """Record the model state before the chunk's first transition and the
        policy version that produced it. Call only when ``steps == 0``.

        The state leaves are cloned: a slot's state is a row view into its
        inference group's batched state, and a view would carry (and serialize)
        the whole batch storage.
        """
        if self._cursor != 0:
            raise RuntimeError("begin_chunk() on a non-empty buffer")
        self.initial_state = tree_map(lambda t: t.detach().clone(), initial_state)
        self.first_version = int(policy_version)

    def open(
        self,
        obs: np.ndarray,
        action: Any,
        log_prob: float,
        value: float,
        mask: np.ndarray | None,
        reward: float = 0.0,
    ) -> None:
        """Append a new (open) transition with an initial reward."""
        if self.is_full:
            raise RuntimeError("open() on a full buffer; seal it first")
        i = self._cursor
        self._obs[i] = obs
        self._actions[i] = action
        self._log_probs[i] = log_prob
        self._values[i] = value
        self._rewards[i] = reward
        self._dones[i] = False
        if self._masks is not None:
            if mask is not None:
                self._masks[i] = mask
                self._has_masks = True
            else:
                self._masks[i] = True
        self._cursor += 1

    def add_reward(self, reward: float) -> None:
        """Add ``reward`` to the most recent transition."""
        self._require_transition("add_reward")
        self._rewards[self._cursor - 1] += reward

    def mark_done(self) -> None:
        """Mark the most recent transition as the last one of its episode."""
        self._require_transition("mark_done")
        self._dones[self._cursor - 1] = True

    def _require_transition(self, what: str) -> None:
        if self._cursor == 0:
            raise RuntimeError(f"{what}() on an empty buffer: there is no transition to update")

    # -- sealing -------------------------------------------------------
    def build_chunk(self, agent_id: str, bootstrap_value: float) -> TrajectoryChunk:
        """Copy the (full) buffer into a TrajectoryChunk. Does not reset."""
        if not self.is_full:
            raise RuntimeError(f"build_chunk() on a partial buffer ({self._cursor}/{self.chunk_length})")
        return TrajectoryChunk(
            agent_id=agent_id,
            observations=torch.from_numpy(self._obs.copy()),
            actions=torch.from_numpy(self._actions.copy()),
            action_log_probs=torch.from_numpy(self._log_probs.copy()),
            rewards=torch.from_numpy(self._rewards.copy()),
            dones=torch.from_numpy(self._dones.copy()),
            values=torch.from_numpy(self._values.copy()),
            bootstrap_value=torch.tensor(float(bootstrap_value), dtype=torch.float32),
            behavior_policy_version=self.first_version,
            initial_state=self.initial_state,
            action_masks=(
                torch.from_numpy(self._masks.copy())
                if self._masks is not None and self._has_masks else None
            ),
        )

    def reset(self) -> None:
        self._cursor = 0
        self._has_masks = False
        self.initial_state = None
        self.first_version = 0


class BufferPool:
    """Per-worker pool of RolloutBuffers keyed by agent.

    ``acquire(agent)`` returns a parked partial buffer of that agent if there is
    one, else an empty recycled buffer, else a new one. ``park(agent, buf)``
    stores a buffer released by a slot at an episode boundary.
    """

    def __init__(
        self,
        chunk_length: int,
        obs_shape: tuple,
        action_shape: tuple,
        action_dtype: Any,
        mask_size: int = 0,
    ) -> None:
        self._spec = (chunk_length, tuple(obs_shape), tuple(action_shape), action_dtype, mask_size)
        self._parked: dict[str, list[RolloutBuffer]] = defaultdict(list)
        self._free: list[RolloutBuffer] = []

    def acquire(self, agent_id: str) -> RolloutBuffer:
        parked = self._parked.get(agent_id)
        if parked:
            return parked.pop(0)
        if self._free:
            return self._free.pop()
        return RolloutBuffer(*self._spec)

    def park(self, agent_id: str, buf: RolloutBuffer) -> None:
        if buf.steps == 0:
            buf.reset()
            self._free.append(buf)
            return
        if not buf.last_done:
            raise RuntimeError(
                f"cannot park a buffer of {agent_id!r} whose last transition is not done"
            )
        if buf.is_full:
            raise RuntimeError(f"cannot park a full buffer of {agent_id!r}; seal it first")
        self._parked[agent_id].append(buf)

    def parked_count(self) -> int:
        return sum(len(v) for v in self._parked.values())

    def parked_transitions(self, agent_id: str) -> int:
        return sum(b.steps for b in self._parked.get(agent_id, []))


@dataclass
class SlotTrack:
    """Rollout state of one (env, seat) slot."""

    buffer: RolloutBuffer | None
    has_open: bool = False
    pending_reward: float = 0.0
    state: State = None

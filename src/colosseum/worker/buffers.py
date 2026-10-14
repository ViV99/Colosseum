"""Agent-owned rollout buffers with act / boot / pad slots and parking (spec block 4).

A :class:`RolloutBuffer` holds up to ``num_slots`` consecutive slots of ONE agent; each
collecting (env, seat) holds one buffer exclusively. The slot write rules (which slot to
write when a seat acts, is eliminated or its episode ends) are applied by ``RolloutLoop``;
this module enforces the invariants they rely on:

- an ACT never takes the last slot, so an open ACT always has room for its BOOT;
- BOOT follows an open ACT; a BOOT without ``reset_after`` only closes a chunk (it takes
  the last slot); PAD follows a slot that ends an episode (never an open ACT);
- the first ACT of a buffer needs ``begin()`` (model state and policy version) since the
  last reset;
- BOOT and PAD slots carry ``ActionSpec.boot_mask()`` and zero actions; a PAD copies the
  previous slot's observation and ``global_state`` (zeros could produce NaN in user encoders);
- a teacher label (``has_teacher``) only on ACT slots; BOOT and PAD carry zero labels;
- a buffer is parked only at an episode boundary (empty, or ending with a terminal ACT or
  a BOOT with ``reset_after``), never full (full buffers are sealed at once).

:class:`BufferPool` keeps parked and free buffers per agent (agents may have different
observation/action spaces, so buffers are never shared between agents).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import Tree, tree_assign, tree_index, tree_map
from colosseum.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD, TrajectoryChunk
from colosseum.networks.state import State
from colosseum.networks.state import tree_map as state_tree_map


@dataclass(frozen=True)
class BufferSpec:
    """Shapes of one agent's slots: its role's observation, action and global-state specs.

    ``teacher``: the agent has a scripted kickstart teacher; its chunks carry ``teacher_action`` /
    ``has_teacher``.
    """

    obs: ObsSpec
    action: ActionSpec
    global_state: ObsSpec | None
    teacher: bool = False


def put_row(dst: Tree, idx: Any, src: Tree) -> None:
    """In place ``dst[idx] = src`` for a bare array or for every leaf of a dict tree."""
    if isinstance(dst, dict):
        tree_assign(dst, idx, src)
    else:
        dst[idx] = src


def _to_torch(tree: Tree) -> Tree:
    return tree_map(lambda a: torch.from_numpy(a.copy()), tree)


class RolloutBuffer:
    """Pre-allocated numpy storage for one chunk of one agent's slots."""

    def __init__(self, num_slots: int, spec: BufferSpec) -> None:
        if num_slots < 2:
            raise ValueError(f"a rollout buffer needs at least 2 slots, got {num_slots}")
        S = num_slots
        self.num_slots = S
        self.spec = spec
        self._obs = spec.obs.allocate((S,))
        self._gs = None if spec.global_state is None else spec.global_state.allocate((S,))
        self._actions = spec.action.allocate_actions((S,))
        self._zero_action = spec.action.allocate_actions(())
        self._masks = spec.action.full_mask((S,)) if spec.action.has_masks else None
        self._full_mask = spec.action.full_mask(()) if spec.action.has_masks else None
        self._boot_mask = spec.action.boot_mask() if spec.action.has_masks else None
        self._num_deciders = spec.action.num_deciders
        self._kind = np.zeros(S, dtype=np.int8)
        self._reward = np.zeros(S, dtype=np.float32)
        self._terminal = np.zeros(S, dtype=np.bool_)
        self._reset_after = np.zeros(S, dtype=np.bool_)
        self._logp = np.zeros(S, dtype=np.float32)
        self._unit_logp = (
            np.zeros((S, self._num_deciders), dtype=np.float32) if self._num_deciders > 1 else None
        )
        self._teacher_actions = spec.action.allocate_actions((S,)) if spec.teacher else None
        self._has_teacher = np.zeros(S, dtype=np.bool_) if spec.teacher else None
        self._cursor = 0
        self._num_acts = 0
        self._begun = False
        self.initial_state: State = None
        self.policy_version = 0

    # -- state -----------------------------------------------------------
    @property
    def slots_used(self) -> int:
        return self._cursor

    @property
    def free_slots(self) -> int:
        return self.num_slots - self._cursor

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.num_slots

    @property
    def has_open(self) -> bool:
        """The last slot is a non-terminal ACT (a transition still collecting rewards)."""
        i = self._cursor - 1
        return bool(i >= 0 and self._kind[i] == SLOT_ACT and not self._terminal[i])

    @property
    def ends_episode(self) -> bool:
        """Empty, or the last non-PAD slot is a terminal ACT or a BOOT with ``reset_after``."""
        for i in range(self._cursor - 1, -1, -1):
            kind = self._kind[i]
            if kind == SLOT_PAD:
                continue
            if kind == SLOT_ACT:
                return bool(self._terminal[i])
            return bool(self._reset_after[i])
        return True

    @property
    def num_acts(self) -> int:
        return self._num_acts

    @property
    def reward_sum(self) -> float:
        """Sum of the rewards recorded in the used slots (only ACT slots carry rewards)."""
        return float(self._reward[: self._cursor].sum(dtype=np.float64))

    # -- writing ---------------------------------------------------------
    def begin(self, initial_state: State, policy_version: int) -> None:
        """Record the model state before slot 0 and the policy version that acts in it.

        Only on an empty buffer. State leaves are cloned: a seat's state is a row view of
        its inference group's batched state and would otherwise carry the whole batch.
        """
        if self._cursor != 0:
            raise RuntimeError("begin() on a non-empty buffer")
        self.initial_state = state_tree_map(lambda t: t.detach().clone(), initial_state)
        self.policy_version = int(policy_version)
        self._begun = True

    def write_act(
        self,
        obs: Tree,
        global_state: Tree | None,
        mask: Tree | None,
        action: Tree,
        log_prob: float,
        unit_log_probs: np.ndarray | None,
        reward: float,
        teacher_action: Tree | None = None,
    ) -> None:
        """Append an open ACT carrying ``reward`` (the seat's pending reward). ``teacher_action``: the scripted
        kickstart teacher's action for this decision, if it was asked."""
        if teacher_action is not None and self._teacher_actions is None:
            raise ValueError("teacher_action given, but this agent's buffers carry no teacher labels "
                             "(BufferSpec.teacher)")
        if self.free_slots < 2:
            raise RuntimeError(
                f"write_act() with {self.free_slots} free slot(s): an ACT never takes the last slot"
            )
        if self._cursor == 0 and not self._begun:
            raise RuntimeError("write_act() on an empty buffer needs begin() first (initial state and version)")
        i = self._cursor
        self._write_obs(i, obs, global_state)
        if self._masks is not None:
            put_row(self._masks, i, self._full_mask if mask is None else mask)
        put_row(self._actions, i, action)
        self._kind[i] = SLOT_ACT
        self._reward[i] = reward
        self._terminal[i] = False
        self._reset_after[i] = False
        self._logp[i] = log_prob
        if self._unit_logp is not None:
            if unit_log_probs is None:
                raise ValueError(f"unit_log_probs is required: the action has {self._num_deciders} deciders")
            self._unit_logp[i] = unit_log_probs
        if self._teacher_actions is not None:
            put_row(self._teacher_actions, i, self._zero_action if teacher_action is None else teacher_action)
            self._has_teacher[i] = teacher_action is not None
        self._cursor += 1
        self._num_acts += 1

    def add_reward(self, reward: float) -> None:
        """Add ``reward`` to the open ACT."""
        self._require_open("add_reward")
        self._reward[self._cursor - 1] += reward

    def mark_terminal(self) -> None:
        """Close the open ACT as the seat's last decision of the episode (bootstrap 0)."""
        self._require_open("mark_terminal")
        self._terminal[self._cursor - 1] = True
        self._reset_after[self._cursor - 1] = True

    def write_boot(self, obs: Tree, global_state: Tree | None, reset_after: bool) -> None:
        """Append a BOOT after the open ACT: the observation the learner bootstraps from.

        ``reset_after=False`` (the episode goes on in the next chunk) only closes a chunk, so
        it needs exactly one free slot; ``reset_after=True`` follows a truncation.
        """
        self._require_open("write_boot")
        if not reset_after and self.free_slots != 1:
            raise RuntimeError(
                f"write_boot(reset_after=False) with {self.free_slots} free slots: a BOOT without "
                f"reset_after only closes a chunk and needs exactly one free slot"
            )
        i = self._cursor
        self._write_obs(i, obs, global_state)
        self._write_non_act(i, SLOT_BOOT, reset_after)

    def write_pad(self) -> None:
        """Append a PAD after a slot that ends an episode; copies that slot's observations."""
        if self._cursor == 0:
            raise RuntimeError("write_pad() on an empty buffer")
        if self.has_open:
            raise RuntimeError("write_pad() after an open ACT; write a BOOT instead")
        if not self.ends_episode:
            raise RuntimeError("write_pad() after a slot that does not end an episode")
        if self.is_full:
            raise RuntimeError("write_pad() on a full buffer")
        i = self._cursor
        put_row(self._obs, i, tree_index(self._obs, i - 1))
        if self._gs is not None:
            put_row(self._gs, i, tree_index(self._gs, i - 1))
        self._write_non_act(i, SLOT_PAD, True)

    def _write_obs(self, i: int, obs: Tree, global_state: Tree | None) -> None:
        put_row(self._obs, i, obs)
        if self._gs is not None:
            if global_state is None:
                raise ValueError("global_state is required: the agent's role declares global_state_space")
            put_row(self._gs, i, global_state)

    def _write_non_act(self, i: int, kind: int, reset_after: bool) -> None:
        if self._masks is not None:
            put_row(self._masks, i, self._boot_mask)
        put_row(self._actions, i, self._zero_action)
        self._kind[i] = kind
        self._reward[i] = 0.0
        self._terminal[i] = False
        self._reset_after[i] = reset_after
        self._logp[i] = 0.0
        if self._unit_logp is not None:
            self._unit_logp[i] = 0.0
        if self._teacher_actions is not None:
            put_row(self._teacher_actions, i, self._zero_action)
            self._has_teacher[i] = False
        self._cursor += 1

    def _require_open(self, what: str) -> None:
        if not self.has_open:
            raise RuntimeError(f"{what}() needs an open ACT as the last slot")

    # -- sealing ---------------------------------------------------------
    def build_chunk(self, agent_id: str) -> TrajectoryChunk:
        """Copy the full buffer into a TrajectoryChunk (torch, CPU). Does not reset."""
        if not self.is_full:
            raise RuntimeError(f"build_chunk() on a partial buffer ({self._cursor}/{self.num_slots})")
        return TrajectoryChunk(
            agent_id=agent_id,
            policy_version=self.policy_version,
            initial_state=self.initial_state,
            obs=_to_torch(self._obs),
            global_state=None if self._gs is None else _to_torch(self._gs),
            actions=_to_torch(self._actions),
            action_masks=None if self._masks is None else _to_torch(self._masks),
            kind=torch.from_numpy(self._kind.copy()),
            reward=torch.from_numpy(self._reward.copy()),
            terminal=torch.from_numpy(self._terminal.copy()),
            reset_after=torch.from_numpy(self._reset_after.copy()),
            behavior_logp=torch.from_numpy(self._logp.copy()),
            behavior_unit_logp=None if self._unit_logp is None else torch.from_numpy(self._unit_logp.copy()),
            teacher_action=None if self._teacher_actions is None else _to_torch(self._teacher_actions),
            has_teacher=None if self._has_teacher is None else torch.from_numpy(self._has_teacher.copy()),
        )

    def reset(self) -> None:
        self._cursor = 0
        self._num_acts = 0
        self._begun = False
        self.initial_state = None
        self.policy_version = 0


class BufferPool:
    """Per-worker pool of RolloutBuffers keyed by agent.

    ``acquire(agent)`` returns a parked buffer of that agent if there is one, else a free
    (empty) buffer of that agent, else a new one. ``park(agent, buf)`` takes a buffer
    released by a seat at an episode boundary; it must be one of this pool's buffers of
    that agent (built from the agent's spec).
    """

    def __init__(self, num_slots: int, specs: Mapping[str, BufferSpec]) -> None:
        self._num_slots = num_slots
        self._specs = dict(specs)
        self._parked: dict[str, list[RolloutBuffer]] = defaultdict(list)
        self._free: dict[str, list[RolloutBuffer]] = defaultdict(list)

    def _spec_of(self, agent_id: str) -> BufferSpec:
        try:
            return self._specs[agent_id]
        except KeyError:
            raise KeyError(f"no buffer spec for agent {agent_id!r}") from None

    def acquire(self, agent_id: str) -> RolloutBuffer:
        spec = self._spec_of(agent_id)
        if self._parked[agent_id]:
            return self._parked[agent_id].pop(0)
        if self._free[agent_id]:
            return self._free[agent_id].pop()
        return RolloutBuffer(self._num_slots, spec)

    def park(self, agent_id: str, buf: RolloutBuffer) -> None:
        if buf.spec is not self._spec_of(agent_id):
            raise ValueError(f"cannot park a buffer under {agent_id!r}: it was not built from that agent's spec")
        if buf.slots_used == 0:
            buf.reset()
            self._free[agent_id].append(buf)
            return
        if not buf.ends_episode:
            raise RuntimeError(f"cannot park a buffer of {agent_id!r} in the middle of an episode")
        if buf.is_full:
            raise RuntimeError(f"cannot park a full buffer of {agent_id!r}; seal it first")
        self._parked[agent_id].append(buf)

    def parked_count(self) -> int:
        return sum(len(v) for v in self._parked.values())

    def parked_acts(self, agent_id: str) -> int:
        return sum(b.num_acts for b in self._parked.get(agent_id, []))

    def parked_reward(self, agent_id: str) -> float:
        """Rewards recorded in ``agent_id``'s parked buffers (not yet sent)."""
        return sum(b.reward_sum for b in self._parked.get(agent_id, []))

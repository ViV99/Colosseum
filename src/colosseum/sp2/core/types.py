"""Core data types of the SP2 pipeline: chunk v2, lineups, match results, commands, weights.

Everything that crosses a process boundary has a numpy form: ``TrajectoryChunk.to_payload``,
``WeightPayload`` (numpy ``state_dict``), ``WorkerCommand`` (numpy checkpoints) and the plain
dataclasses ``Lineup`` / ``MatchResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from colosseum.core.ipc import numpy_to_tensor, tensor_to_numpy
from colosseum.networks.state import State, state_from_numpy, state_to_numpy
from colosseum.networks.state import tree_map as state_tree_map
from colosseum.sp2.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch

# Network id of an agent's current weights, as opposed to a checkpoint id ("ckpt_v<N>").
LATEST_NETWORK_ID = "latest"

# Values of TrajectoryChunk.kind (int8).
SLOT_ACT, SLOT_BOOT, SLOT_PAD = 0, 1, 2


def state_dict_to_numpy(state_dict: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detached CPU numpy copies of a torch ``state_dict`` (bfloat16 -> float32)."""
    return {key: tensor_to_numpy(value) for key, value in state_dict.items()}


def state_dict_from_numpy(state_dict: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """CPU torch tensors from a numpy ``state_dict``; inverse of :func:`state_dict_to_numpy`."""
    return {key: numpy_to_tensor(value) for key, value in state_dict.items()}


def _map_optional(fn: Callable[[Any], Any], tree: Tree | None) -> Tree | None:
    return None if tree is None else tree_map(fn, tree)


@dataclass
class TrajectoryChunk:
    """``S`` consecutive slots of ONE agent (spec block 4).

    Each slot is one of:
    - ``SLOT_ACT``: a decision of a seat (observation, optional ``global_state``, mask,
      action, behavior log-probs, the reward accumulated while it was open, ``terminal``);
    - ``SLOT_BOOT``: only an observation (and ``global_state``) for the learner's bootstrap
      value; no action, no loss;
    - ``SLOT_PAD``: a filler that affects nothing.

    ``reset_after[s]`` resets the model state after slot ``s`` (terminal ACTs, truncation
    BOOTs and PADs). An ACT never sits in the last slot. Trees (``obs``, ``global_state``,
    ``actions``, ``action_masks``) have leaves ``[S, ...]`` with their native dtypes.
    ``behavior_unit_logp`` (``[S, K]``) exists only when the role's action has ``K > 1``
    deciders. ``initial_state`` is the model state before slot 0 (leaves ``[1, ...]``).
    """

    agent_id: str
    policy_version: int
    initial_state: State
    obs: Tree
    global_state: Tree | None
    actions: Tree
    action_masks: Tree | None
    kind: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    reset_after: torch.Tensor
    behavior_logp: torch.Tensor
    behavior_unit_logp: torch.Tensor | None

    @property
    def num_slots(self) -> int:
        return int(self.kind.shape[0])

    @property
    def num_acts(self) -> int:
        return int((self.kind == SLOT_ACT).sum())

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> TrajectoryChunk:
        return TrajectoryChunk(
            agent_id=self.agent_id,
            policy_version=self.policy_version,
            initial_state=state_tree_map(fn, self.initial_state),
            obs=tree_map(fn, self.obs),
            global_state=_map_optional(fn, self.global_state),
            actions=tree_map(fn, self.actions),
            action_masks=_map_optional(fn, self.action_masks),
            kind=fn(self.kind),
            reward=fn(self.reward),
            terminal=fn(self.terminal),
            reset_after=fn(self.reset_after),
            behavior_logp=fn(self.behavior_logp),
            behavior_unit_logp=None if self.behavior_unit_logp is None else fn(self.behavior_unit_logp),
        )

    def to(self, device: str | torch.device) -> TrajectoryChunk:
        """Copy with every tensor (state leaves included) on ``device``."""
        return self._apply(lambda t: t.to(device))

    def pin_memory(self) -> TrajectoryChunk:
        """Copy with every tensor in page-locked memory."""
        return self._apply(lambda t: t.pin_memory())

    def to_payload(self) -> dict[str, Any]:
        """Numpy trees + primitives only, for crossing a process boundary."""
        return {
            "agent_id": str(self.agent_id),
            "policy_version": int(self.policy_version),
            "initial_state": state_to_numpy(self.initial_state),
            "obs": tree_to_numpy(self.obs),
            "global_state": None if self.global_state is None else tree_to_numpy(self.global_state),
            "actions": tree_to_numpy(self.actions),
            "action_masks": None if self.action_masks is None else tree_to_numpy(self.action_masks),
            "kind": tensor_to_numpy(self.kind),
            "reward": tensor_to_numpy(self.reward),
            "terminal": tensor_to_numpy(self.terminal),
            "reset_after": tensor_to_numpy(self.reset_after),
            "behavior_logp": tensor_to_numpy(self.behavior_logp),
            "behavior_unit_logp": (
                None if self.behavior_unit_logp is None else tensor_to_numpy(self.behavior_unit_logp)
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrajectoryChunk:
        """Rebuild a chunk (CPU tensors) from :meth:`to_payload` output."""
        gs, masks, unit_logp = payload["global_state"], payload["action_masks"], payload["behavior_unit_logp"]
        return cls(
            agent_id=str(payload["agent_id"]),
            policy_version=int(payload["policy_version"]),
            initial_state=state_from_numpy(payload["initial_state"]),
            obs=tree_to_torch(payload["obs"]),
            global_state=None if gs is None else tree_to_torch(gs),
            actions=tree_to_torch(payload["actions"]),
            action_masks=None if masks is None else tree_to_torch(masks),
            kind=numpy_to_tensor(payload["kind"]),
            reward=numpy_to_tensor(payload["reward"]),
            terminal=numpy_to_tensor(payload["terminal"]),
            reset_after=numpy_to_tensor(payload["reset_after"]),
            behavior_logp=numpy_to_tensor(payload["behavior_logp"]),
            behavior_unit_logp=None if unit_logp is None else numpy_to_tensor(unit_logp),
        )


_SLOT_RULES = (
    "an ACT in the last slot",
    "an open ACT followed by a PAD",
    "a BOOT that does not follow an open ACT",
    "a PAD that does not follow the end of an episode",
    "a BOOT without reset_after before the last slot",
)


def _spell_slots(kind: np.ndarray, terminal: np.ndarray, reset_after: np.ndarray) -> str:
    """Slot letters as in the SP2 plan: A open ACT, T terminal ACT, B / R BOOT, P PAD, ? unknown."""
    letters = []
    for k, term, reset in zip(kind.tolist(), terminal.tolist(), reset_after.tolist(), strict=True):
        if k == SLOT_ACT:
            letters.append("T" if term else "A")
        elif k == SLOT_BOOT:
            letters.append("R" if reset else "B")
        else:
            letters.append("P" if k == SLOT_PAD else "?")
    return "".join(letters)


def validate_slot_structure(chunk: TrajectoryChunk) -> None:
    """Raise ``ValueError`` (naming the agent) if ``chunk``'s slots break the chunk v2 rules.

    The rules the worker's ``RolloutBuffer`` guarantees and the learner's V-trace relies on:
    an ACT is never the last slot; an open (non-terminal) ACT is followed by an ACT or a
    BOOT; a BOOT follows an open ACT; a PAD follows a slot that ends an episode (a terminal
    ACT or a BOOT with ``reset_after``, possibly after other PADs); a BOOT without
    ``reset_after`` takes only the last slot. The earliest broken slot is reported. A cheap
    numpy check over ``kind`` / ``terminal`` / ``reset_after``.
    """
    kind = chunk.kind.cpu().numpy()
    terminal = chunk.terminal.cpu().numpy().astype(bool)
    reset_after = chunk.reset_after.cpu().numpy().astype(bool)
    where = f"agent {chunk.agent_id!r}: malformed chunk v2 (policy_version {chunk.policy_version}"
    if not (kind.ndim == terminal.ndim == reset_after.ndim == 1
            and kind.shape == terminal.shape == reset_after.shape):
        raise ValueError(
            f"{where}): kind, terminal and reset_after must be 1-D arrays of one length, got shapes "
            f"{kind.shape}, {terminal.shape}, {reset_after.shape}"
        )
    if kind.size == 0:
        raise ValueError(f"{where}): the chunk has no slots")
    where = f"{where}, slots {_spell_slots(kind, terminal, reset_after)!r})"
    unknown = np.flatnonzero(~np.isin(kind, (SLOT_ACT, SLOT_BOOT, SLOT_PAD)))
    if unknown.size:
        raise ValueError(f"{where}: slot {unknown[0]}: unknown slot kind {kind[unknown[0]]}")

    act, boot, pad = kind == SLOT_ACT, kind == SLOT_BOOT, kind == SLOT_PAD
    open_act = act & ~terminal
    ends_episode = (act & terminal) | (boot & reset_after)
    prev_open = np.concatenate([[False], open_act[:-1]])
    prev_pad = np.concatenate([[False], pad[:-1]])
    prev_ends = np.concatenate([[False], ends_episode[:-1]])
    next_pad = np.concatenate([pad[1:], [False]])
    not_last = np.arange(kind.size) < kind.size - 1
    broken = (
        act & ~not_last,                        # an ACT in the last slot
        open_act & next_pad,                    # an open ACT followed by a PAD
        boot & ~prev_open,                      # a BOOT that does not follow an open ACT
        pad & ~prev_pad & ~prev_ends,           # a PAD (first of a run) not after an episode end
        boot & ~reset_after & not_last,         # a BOOT without reset_after before the last slot
    )
    # Earliest broken slot; ties go to the first rule in _SLOT_RULES.
    found = [(int(np.argmax(mask)), r) for r, mask in enumerate(broken) if mask.any()]
    if found:
        slot, rule = min(found)
        raise ValueError(f"{where}: slot {slot}: {_SLOT_RULES[rule]}")


@dataclass
class SeatAssignment:
    """Who plays one seat: an agent's latest weights or a checkpoint, and whether it collects."""

    agent_id: str
    network_id: str = LATEST_NETWORK_ID
    collect: bool = True


@dataclass
class Lineup:
    """One match composition: a layout of the game and one assignment per seat of it."""

    layout: str
    seats: list[SeatAssignment]


@dataclass
class SeatResult:
    """One seat of a finished match. ``reward`` is the undiscounted episode return."""

    seat: int
    role: str
    team: int
    agent_id: str
    network_id: str
    reward: float
    eliminated_step: int | None = None


@dataclass
class TeamResult:
    """One team of a finished match: rank (1 = best, ties share) and score."""

    team: int
    rank: float
    score: float


@dataclass
class MatchResult:
    """A finished match: one SeatResult per occupied seat, one TeamResult per team."""

    match_id: str
    layout: str
    outcome_kind: Literal["score", "wdl", "rank"]
    seats: list[SeatResult]
    teams: list[TeamResult]
    episode_length: int


@dataclass
class WorkerCommand:
    """Runtime update from the coordinator to one worker.

    ``lineups[e]`` replaces env ``e``'s lineup at its next episode end (``None`` = keep).
    ``new_checkpoints`` (``{agent_id: {checkpoint_id: numpy state_dict}}``) carries only the
    checkpoints the worker does not have yet; they are loaded at once.
    """

    lineups: list[Lineup | None]
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)


@dataclass
class WeightPayload:
    """Model weights flowing from a learner to workers (numpy, never torch)."""

    agent_id: str
    policy_version: int
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_model(cls, agent_id: str, policy_version: int, model: nn.Module) -> WeightPayload:
        """Snapshot ``model``'s current weights as a numpy payload."""
        return cls(
            agent_id=agent_id,
            policy_version=int(policy_version),
            state_dict=state_dict_to_numpy(model.state_dict()),
        )

    def to_torch_state_dict(self) -> dict[str, torch.Tensor]:
        """CPU torch ``state_dict`` for ``model.load_state_dict``."""
        return state_dict_from_numpy(self.state_dict)

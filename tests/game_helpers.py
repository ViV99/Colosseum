"""Shared SP2 test kit: toy ``MultiAgentEnv`` games (T1.4), drivers (T1.5), tiny models (T2.3).

Every game is small, pure numpy and deterministic given the reset seed. Games are
top-level classes, so ``functools.partial(Game, ...)`` or the class itself is a picklable
``env_fn`` for spawned processes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary

from colosseum.networks.cores import Core, GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree, tree_get, tree_leaves, tree_map
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import Distribution, make_distribution
from colosseum.sp2.networks.heads import UnitsHead
from colosseum.sp2.networks.model import PolicyModel, PolicyStep, UnrollOutput


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class SoloCounterGame(MultiAgentEnv):
    """Solo, ``length`` steps. Obs ``[t / length, 1]``; ``Discrete(2)``; reward 1 for action 1.

    With ``truncate_at`` (< length) the episode is cut after that many steps (``truncated``,
    ``final_obs``).
    """

    def __init__(self, length: int = 8, truncate_at: int | None = None) -> None:
        self.length, self.truncate_at = length, truncate_at
        self.spec = GameSpec.solo(Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _obs(self) -> np.ndarray:
        return _vec(self.t / self.length, 1.0)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        rewards = {0: float(int(actions[0]) == 1)}
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        if self.truncate_at is not None and self.t >= self.truncate_at:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True, truncated=True,
                              final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()}, rewards=rewards)


class TurnTakingGame(MultiAgentEnv):
    """Two seats alternate (seat 0 first). Obs ``[t / length, seat, 1]``; ``Discrete(3)`` where action 2
    is always masked. The acting seat's action ``a`` pays ``a`` to the WAITING seat (so seat 1 gets a
    reward before its first move). ``length`` moves in total; outcome from returns (wdl)."""

    def __init__(self, length: int = 6) -> None:
        self.length = length
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (3,), dtype=np.float32), Discrete(3))
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        seat = self.t % 2
        return StepResult(acting={seat}, obs={seat: _vec(self.t / self.length, seat, 1.0)},
                          action_masks={seat: np.array([True, True, False])}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        (seat, action), = actions.items()
        rewards = {1 - seat: float(int(action))}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class SimultaneousGame(MultiAgentEnv):
    """Matching pennies: both seats act every step for ``length`` steps. Obs ``[t / length, seat]``;
    ``Discrete(2)``; seat 0 gets +1 if the actions match, else -1; seat 1 the opposite."""

    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p) for p in (0, 1)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return StepResult(acting={0, 1}, obs=self._obs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        r = 1.0 if int(actions[0]) == int(actions[1]) else -1.0
        rewards = {0: r, 1: -r}
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return StepResult(acting={0, 1}, obs=self._obs(), rewards=rewards)


class EliminationFFA(MultiAgentEnv):
    """FFA with layouts ``"2p"..f"{max_players}p"``; every live seat acts every step.

    Obs ``[t / length, seat, live seats]``; ``Discrete(2)``. Each acting seat gets 0.1 per step.
    ``eliminate_at`` maps seat -> the step in which it is eliminated (an extra -1 in that step);
    default: in an n-seat layout seat s (s >= 1) is eliminated at step n - s. The episode ends when
    at most one seat is live or after ``length`` steps. Outcome: ``team_rank`` by elimination order
    (survivors share rank 1).
    """

    def __init__(self, max_players: int = 4, eliminate_at: Mapping[int, int] | None = None,
                 length: int = 10) -> None:
        self.max_players, self.length = max_players, length
        self.eliminate_at = dict(eliminate_at) if eliminate_at is not None else None
        self.spec = GameSpec.symmetric(range(2, max_players + 1), Box(-np.inf, np.inf, (3,), dtype=np.float32),
                                       Discrete(2))
        self.n, self.t = 0, 0
        self.live: set[int] = set()
        self.out_step: dict[int, int] = {}
        self.schedule: dict[int, int] = {}

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p, len(self.live)) for p in sorted(self.live)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.n = self.spec.layout_size(layout)
        self.t = 0
        self.live = set(range(self.n))
        self.out_step = {}
        if self.eliminate_at is None:
            self.schedule = {s: self.n - s for s in range(1, self.n)}
        else:
            self.schedule = {s: k for s, k in self.eliminate_at.items() if s < self.n}
        return StepResult(acting=set(self.live), obs=self._obs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        rewards = {p: 0.1 for p in actions}
        out = {p for p in self.live if self.schedule.get(p) == self.t}
        for p in out:
            rewards[p] = rewards.get(p, 0.0) - 1.0
            self.out_step[p] = self.t
        self.live -= out
        if len(self.live) <= 1 or self.t >= self.length:
            last = {p: self.out_step.get(p, self.t + 1) for p in range(self.n)}
            rank = {p: float(1 + sum(1 for q in range(self.n) if last[q] > last[p])) for p in range(self.n)}
            return StepResult(acting=set(), obs={}, rewards=rewards, terminated=out, episode_over=True,
                              outcome=Outcome(team_rank=rank))
        return StepResult(acting=set(self.live), obs=self._obs(), rewards=rewards, terminated=out)


class TeamDeadTeammateGame(MultiAgentEnv):
    """``"2v2"`` (seats 0, 1 = team 0; 2, 3 = team 1). Seat 1 stops acting after step ``dead_at`` but is
    not terminated: every step each seat of a team, the dead one included, gets the mean action of the
    team's acting seats. Obs ``[t / length, seat]``; ``Discrete(2)``; ``length`` steps."""

    def __init__(self, length: int = 6, dead_at: int = 2) -> None:
        self.length, self.dead_at = length, dead_at
        self.spec = GameSpec.teams_of([2, 2], Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _acting(self) -> set[int]:
        return {0, 2, 3} if self.t >= self.dead_at else {0, 1, 2, 3}

    def _result(self, rewards: dict[int, float]) -> StepResult:
        acting = self._acting()
        return StepResult(acting=acting, obs={p: _vec(self.t / self.length, p) for p in acting}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        rewards: dict[int, float] = {}
        for team in ((0, 1), (2, 3)):
            moves = [int(actions[p]) for p in team if p in actions]
            for p in team:
                rewards[p] = float(np.mean(moves))
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class UnitsGame(MultiAgentEnv):
    """Solo bot with units. Obs ``Dict(grid uint8 [4, 4] (float32 if not uint8_grid), entities [U, 3],
    entity_mask [U])``. Action ``Dict(base: Discrete(3), units: Units(U, Dict(move: Discrete(4),
    target: Discrete(U)), only_if target <- move == 3))``.

    Units are born and die: at step t, units ``0 .. (t % U)`` exist. Absent units have ``unit=False``
    and empty action rows; a unit may target only existing units. Reward: 0.5 for base action 1 plus
    0.25 per existing unit choosing move 0. ``length`` steps.
    """

    def __init__(self, max_units: int = 4, uint8_grid: bool = True, length: int = 6) -> None:
        self.U, self.uint8_grid, self.length = max_units, uint8_grid, length
        grid = Box(0, 255, (4, 4), dtype=np.uint8) if uint8_grid else Box(0.0, 255.0, (4, 4), dtype=np.float32)
        obs = Dict([("grid", grid), ("entities", Box(-1.0, 1.0, (max_units, 3), dtype=np.float32)),
                    ("entity_mask", MultiBinary(max_units))])
        act = Dict([("base", Discrete(3)),
                    ("units", Units(max_units, Dict([("move", Discrete(4)), ("target", Discrete(max_units))]),
                                    only_if={"target": ("move", {3})}))])
        self.spec = GameSpec.solo(obs, act)
        self.t = 0

    def _alive(self) -> np.ndarray:
        return np.arange(self.U) <= (self.t % self.U)

    def _result(self, rewards: dict[int, float]) -> StepResult:
        alive = self._alive()
        grid = np.full((4, 4), self.t, dtype=np.uint8 if self.uint8_grid else np.float32)
        entities = np.zeros((self.U, 3), dtype=np.float32)
        entities[alive] = [self.t / self.length, 1.0, 0.0]
        obs = {"grid": grid, "entities": entities, "entity_mask": alive.astype(np.int8)}
        action = np.zeros((self.U, 4 + self.U), dtype=bool)
        action[alive, :4] = True
        action[np.ix_(alive, 4 + np.flatnonzero(alive))] = True
        mask = {"units": {"unit": alive.copy(), "action": action}}
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        a = actions[0]
        alive = self._alive()
        reward = 0.5 * float(int(a["base"]) == 1) + 0.25 * float(np.sum((np.asarray(a["units"]["move"]) == 0) & alive))
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return self._result({0: reward})


class AsymmetricGame(MultiAgentEnv):
    """Layout ``"1v2"``: seat 0 = role ``hunter`` (obs ``[4]``, ``Discrete(5)``), seats 1, 2 = role ``prey``
    (obs ``[3]``, ``Discrete(3)``), all acting each step for ``length`` steps. The hunter gets 1 per
    prey whose action equals ``hunter_action % 3``; that prey gets -1, the others +0.5."""

    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.spec = GameSpec(
            roles={"hunter": RoleSpec(Box(-np.inf, np.inf, (4,), dtype=np.float32), Discrete(5)),
                   "prey": RoleSpec(Box(-np.inf, np.inf, (3,), dtype=np.float32), Discrete(3))},
            layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))},
        )
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        f = self.t / self.length
        return StepResult(acting={0, 1, 2}, obs={0: _vec(f, 0, 0, 1), 1: _vec(f, 1, 0), 2: _vec(f, 2, 0)},
                          rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        target = int(actions[0]) % 3
        rewards = {0: 0.0}
        for p in (1, 2):
            caught = int(actions[p]) == target
            rewards[0] += float(caught)
            rewards[p] = -1.0 if caught else 0.5
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class CoopGame(MultiAgentEnv):
    """One team of ``size`` seats (layout ``"coop<size>"``), all acting for ``length`` steps.
    Obs ``[t / length, seat]``; ``Discrete(2)``; every seat gets 1 when all actions are 1."""

    def __init__(self, size: int = 2, length: int = 5) -> None:
        self.size, self.length = size, length
        self.spec = GameSpec.teams_of([size], Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        seats = set(range(self.size))
        return StepResult(acting=seats, obs={p: _vec(self.t / self.length, p) for p in seats}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        r = float(all(int(a) == 1 for a in actions.values()))
        rewards = {p: r for p in range(self.size)}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class GlobalStateGame(MultiAgentEnv):
    """``"2p"``, both seats act; obs ``[t / length, seat]``, ``Discrete(2)``, ``global_state`` ``[4]`` =
    ``[seat, t / length, last action of seat 0, last action of seat 1]`` for every acting seat (and,
    with ``truncate_at``, for every live seat with ``final_obs``). Reward: +1 to a seat choosing 1."""

    def __init__(self, length: int = 5, truncate_at: int | None = None) -> None:
        self.length, self.truncate_at = length, truncate_at
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2),
                                       global_state=Box(-np.inf, np.inf, (4,), dtype=np.float32))
        self.t = 0
        self.last = [0, 0]

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p) for p in (0, 1)}

    def _gs(self) -> dict[int, np.ndarray]:
        return {p: _vec(p, self.t / self.length, *self.last) for p in (0, 1)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t, self.last = 0, [0, 0]
        return StepResult(acting={0, 1}, obs=self._obs(), global_state=self._gs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.last = [int(actions[0]), int(actions[1])]
        rewards = {p: float(self.last[p]) for p in (0, 1)}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        if self.truncate_at is not None and self.t >= self.truncate_at:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True, truncated=True,
                              final_obs=self._obs(), global_state=self._gs())
        return StepResult(acting={0, 1}, obs=self._obs(), rewards=rewards, global_state=self._gs())


class ScriptedGame(MultiAgentEnv):
    """Replays prepared results: ``reset`` returns ``script[0]``, the k-th ``step`` returns ``script[k]``.
    The actions it received are kept in ``received``. For contract-violation tests."""

    def __init__(self, spec: GameSpec, script: Sequence[StepResult]) -> None:
        self.spec = spec
        self.script = list(script)
        self.k = 0
        self.received: list[dict[int, Any]] = []
        self.reset_calls: list[tuple[int | None, str]] = []

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.k = 0
        self.reset_calls.append((seed, layout))
        return self.script[0]

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.received.append(dict(actions))
        self.k += 1
        return self.script[self.k]


TOY_GAMES: dict[str, Callable[[], MultiAgentEnv]] = {
    "solo": SoloCounterGame,
    "turns": TurnTakingGame,
    "simultaneous": SimultaneousGame,
    "ffa": EliminationFFA,
    "dead_teammate": TeamDeadTeammateGame,
    "units": UnitsGame,
    "asymmetric": AsymmetricGame,
    "coop": CoopGame,
    "global_state": GlobalStateGame,
}


# ---------------------------------------------------------------------------
# Drivers (T1.5)
# ---------------------------------------------------------------------------


def _legal(row: np.ndarray | None, n: int) -> np.ndarray:
    legal = np.arange(n) if row is None else np.flatnonzero(row)
    return legal if legal.size else np.zeros(1, dtype=np.int64)


def sample_legal_action(spec: ActionSpec, mask: Tree | None, rng: np.random.Generator) -> Tree:
    """A uniformly random legal action (numpy, env format) for a normalized ``mask`` (or None).

    Discrete parts pick among legal values (0 when a units row is empty or the unit absent);
    box parts are uniform in [-1, 1].
    """
    values: dict[tuple[str, ...], Any] = {}
    for group in spec.groups:
        row = spec.group_mask(mask, group)
        if group.kind == "discrete":
            values[group.path] = np.int64(rng.choice(_legal(row, group.nvec[0])))
        elif group.kind == "multi_discrete":
            out, offset = [], 0
            for n in group.nvec:
                out.append(rng.choice(_legal(None if row is None else row[offset:offset + n], n)))
                offset += n
            values[group.path] = np.array(out, dtype=np.int64)
        elif group.kind == "box":
            values[group.path] = rng.uniform(-1.0, 1.0, group.box_dim).astype(np.float32)
        else:
            units = group.units
            U = units.max_units
            unit = np.ones(U, dtype=bool) if row is None else row["unit"]
            comps: dict[str, np.ndarray] = {}
            offset = 0
            for c in units.components:
                if c.kind == "discrete":
                    arr = np.zeros(U, dtype=np.int64)
                    for u in np.flatnonzero(unit):
                        sub = None if row is None else row["action"][u, offset:offset + c.size]
                        arr[u] = rng.choice(_legal(sub, c.size))
                    offset += c.size
                else:
                    arr = rng.uniform(-1.0, 1.0, (U, c.size)).astype(np.float32)
                    arr[~unit] = 0.0
                comps[c.name] = arr
            if units.per_unit_kind == "dict":
                values[group.path] = comps
            elif units.per_unit_kind == "multi_discrete":
                values[group.path] = np.stack([comps[c.name] for c in units.components], axis=1)
            else:
                values[group.path] = comps["0"]
    if not spec.is_dict:
        return values[spec.groups[0].path]
    out_tree: dict = {}
    for path, value in values.items():
        node = out_tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return out_tree


def play_episode(env: MultiAgentEnv, layout: str, *, seed: int | None = None,
                 rng: np.random.Generator | None = None, tracker: EpisodeTracker | None = None,
                 max_steps: int = 10_000) -> tuple[EpisodeTracker, list[StepResult]]:
    """Play one episode with random legal actions, every result checked by an ``EpisodeTracker``.

    Returns the tracker (phases, returns, team result) and every result (reset first).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    tracker = tracker if tracker is not None else EpisodeTracker(env.spec)
    specs = {role: ActionSpec.from_space(r.action_space) for role, r in env.spec.roles.items()}
    result = env.reset(seed, layout)
    masks = tracker.on_reset(layout, result)
    results = [result]
    for _ in range(max_steps):
        if tracker.episode_over:
            return tracker, results
        actions = {seat: sample_legal_action(specs[env.spec.role_of(layout, seat)], masks[seat], rng)
                   for seat in sorted(tracker.acting())}
        result = env.step(actions)
        masks = tracker.on_step(actions, result)
        results.append(result)
    raise RuntimeError(f"play_episode: no episode_over within {max_steps} steps")


# ---------------------------------------------------------------------------
# Tiny models (T2.3)
# ---------------------------------------------------------------------------

CORE_KINDS = ("none", "lstm", "gru", "attention")


def make_core(kind: str, input_dim: int, hidden: int = 16) -> Core:
    """A core of ``kind`` (one of ``CORE_KINDS``) sized for tests."""
    if kind == "none":
        return NoCore(input_dim)
    if kind == "lstm":
        return LSTMCore(input_dim, hidden_size=hidden)
    if kind == "gru":
        return GRUCore(input_dim, hidden_size=hidden)
    if kind == "attention":
        return WindowAttentionCore(input_dim, d_model=hidden, window=4, num_heads=2)
    raise ValueError(f"unknown core kind {kind!r}; use one of {CORE_KINDS}")


def flatten_tree(spec: ObsSpec, tree: Tree) -> torch.Tensor:
    """Every leaf ``[B, ...]`` cast to float and flattened, concatenated in leaf order -> ``[B, F]``."""
    parts = []
    for leaf in spec.leaves:
        x = tree_get(tree, leaf.path) if spec.is_dict else tree
        parts.append(x.reshape(x.shape[0], -1).float())
    return torch.cat(parts, dim=-1)


def params_tree(spec: ActionSpec, per_group: dict[tuple[str, ...], Any]) -> Tree:
    """Per-group distribution parameters (keyed by group path) as the tree ``make_distribution`` takes."""
    if not spec.is_dict:
        return per_group[spec.groups[0].path]
    tree: dict = {}
    for path, value in per_group.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


def _flat_size(spec: ObsSpec) -> int:
    return sum(int(np.prod(leaf.shape)) if leaf.shape else 1 for leaf in spec.leaves)


class GenericEncoder(BaseEncoder):
    """Any observation space: flatten every leaf (cast to float) -> Linear -> ReLU -> ``[B, hidden]``.

    Records the dtypes of the leaves it was last called with in ``seen_dtypes``.
    """

    def __init__(self, observation_space: gymnasium.Space, hidden: int = 16) -> None:
        super().__init__()
        self.obs_spec = ObsSpec.from_space(observation_space)
        self.fc = nn.Linear(_flat_size(self.obs_spec), hidden)
        self._latent_dim = hidden
        self.seen_dtypes: list[torch.dtype] = []

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: Tree) -> EncoderOutput:
        self.seen_dtypes = [leaf.dtype for leaf in tree_leaves(obs)]
        return EncoderOutput(torch.relu(self.fc(flatten_tree(self.obs_spec, obs))), {})


class GenericCriticEncoder(BaseCriticEncoder):
    """Any global-state space: flatten -> Linear -> ReLU -> ``[B, hidden]``."""

    def __init__(self, global_state_space: gymnasium.Space, hidden: int = 16) -> None:
        super().__init__()
        self.gs_spec = ObsSpec.from_space(global_state_space)
        self.fc = nn.Linear(_flat_size(self.gs_spec), hidden)
        self._out = hidden

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, global_state: Tree) -> torch.Tensor:
        return torch.relu(self.fc(flatten_tree(self.gs_spec, global_state)))


class TreePolicyHead(BasePolicy):
    """Any action space: a linear head per group; a units group gets the features concatenated with a
    learned per-slot embedding of size ``hidden``, fed to a ``UnitsHead``."""

    def __init__(self, in_dim: int, action_spec: ActionSpec, hidden: int = 16) -> None:
        super().__init__()
        self.spec = action_spec
        self.heads = nn.ModuleDict()
        self.log_std = nn.ParameterDict()
        self.slots = nn.ParameterDict()
        for i, group in enumerate(action_spec.groups):
            key = str(i)
            if group.kind == "units":
                self.slots[key] = nn.Parameter(torch.randn(group.units.max_units, hidden) * 0.5)
                self.heads[key] = UnitsHead(group, in_dim + hidden)
            elif group.kind == "box":
                self.heads[key] = nn.Linear(in_dim, group.box_dim)
                self.log_std[key] = nn.Parameter(torch.zeros(group.box_dim))
            else:
                self.heads[key] = nn.Linear(in_dim, group.mask_size)

    def forward(self, features: torch.Tensor, aux: dict[str, torch.Tensor]) -> Distribution:
        params: dict[tuple[str, ...], Any] = {}
        for i, group in enumerate(self.spec.groups):
            key = str(i)
            if group.kind == "units":
                slots = self.slots[key].unsqueeze(0).expand(features.shape[0], -1, -1)
                per_unit = features.unsqueeze(1).expand(-1, slots.shape[1], -1)
                params[group.path] = self.heads[key](torch.cat([per_unit, slots], dim=-1))
            elif group.kind == "box":
                params[group.path] = {"mean": self.heads[key](features), "log_std": self.log_std[key]}
            else:
                params[group.path] = self.heads[key](features)
        return make_distribution(self.spec, params_tree(self.spec, params))


class GenericValue(BaseValue):
    """``[B, in_dim]`` -> Linear -> ReLU -> Linear -> ``[B]``."""

    def __init__(self, in_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_test_model(role: RoleSpec, core: str = "none", hidden: int = 16) -> ComposedModel:
    """A small ``ComposedModel`` for any role; a role with a global state gets a critic encoder."""
    encoder = GenericEncoder(role.observation_space, hidden)
    trunk = make_core(core, encoder.latent_dim, hidden)
    critic = GenericCriticEncoder(role.global_state_space, hidden) if role.global_state_space is not None else None
    value_in = trunk.output_dim + (critic.output_dim if critic is not None else 0)
    return ComposedModel(encoder, trunk, TreePolicyHead(trunk.output_dim, ActionSpec.from_space(role.action_space),
                                                        hidden), GenericValue(value_in, hidden), critic)


class RandomPolicy(PolicyModel):
    """Stateless: uniform over legal discrete actions (zero logits + the mask), N(0, 1) for box parts.
    ``unroll`` returns zero values (or None with ``with_value=False``)."""

    def __init__(self, role: RoleSpec) -> None:
        super().__init__()
        self.spec = ActionSpec.from_space(role.action_space)

    def _dist(self, batch: int, device: torch.device) -> Distribution:
        params: dict[tuple[str, ...], Any] = {}
        for group in self.spec.groups:
            if group.kind == "units":
                U = group.units.max_units
                params[group.path] = {
                    c.name: (torch.zeros(batch, U, c.size, device=device) if c.kind == "discrete"
                             else {"mean": torch.zeros(batch, U, c.size, device=device),
                                   "log_std": torch.zeros(c.size, device=device)})
                    for c in group.units.components}
            elif group.kind == "box":
                params[group.path] = {"mean": torch.zeros(batch, group.box_dim, device=device),
                                      "log_std": torch.zeros(group.box_dim, device=device)}
            else:
                params[group.path] = torch.zeros(batch, group.mask_size, device=device)
        return make_distribution(self.spec, params_tree(self.spec, params))

    def step(self, obs: Tree, state: Any, action_mask: Tree | None = None) -> PolicyStep:
        first = tree_leaves(obs)[0]
        dist = self._dist(int(first.shape[0]), first.device)
        return PolicyStep(dist=dist if action_mask is None else dist.apply_mask(action_mask), state=None)

    def unroll(self, obs: Tree, state0: Any, reset_after: torch.Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        first = tree_leaves(obs)[0]
        S, B = int(first.shape[0]), int(first.shape[1])
        dist = self._dist(S * B, first.device)
        if action_mask is not None:
            dist = dist.apply_mask(tree_map(lambda m: m.reshape(S * B, *m.shape[2:]), action_mask))
        return UnrollOutput(dist=dist, value=torch.zeros(S * B, device=first.device) if with_value else None)


# ---------------------------------------------------------------------------
# Part B (T3.3): scripted game, dict model pool, recording observer
# ---------------------------------------------------------------------------


SCRIPT_OBS_SPACE = gymnasium.spaces.Box(-1e6, 1e6, (5,), np.float32)
SCRIPT_GS_SPACE = gymnasium.spaces.Box(-1e6, 1e6, (2,), np.float32)


@dataclass
class Tick:
    """What a TickGame returns at one episode step (tick 0 = the reset result)."""

    acting: Collection[int] = ()
    rewards: Mapping[int, float] = field(default_factory=dict)
    terminated: Collection[int] = ()
    over: bool = False
    truncated: bool = False
    outcome: Outcome | None = None


class TickGame(MultiAgentEnv):
    """Replays a fixed script of Ticks every episode (or cycles through several scripts).

    Observation of seat ``s`` at step ``t`` of episode ``k`` (``k`` counts this env's
    resets from 0): ``[tag, k, t, s, 0]``; its truncation ``final_obs`` is ``[tag, k, t, s, 1]``;
    ``global_state`` (if enabled) is ``[k, t + 0.5 * s]``. ``obs_dtype`` sets the observation
    Box dtype (e.g. ``np.uint8``). ``mask_fn(k, t, seat)`` gives the
    acting seats' masks (``None`` = no masks). ``log`` records ``(k, t, seed, actions)``
    for every reset (``actions`` None) and step.
    """

    def __init__(
        self,
        script: Sequence[Tick] | Sequence[Sequence[Tick]],
        num_seats: int = 1,
        *,
        action_space: gymnasium.Space | None = None,
        global_state: bool = False,
        mask_fn: Callable[[int, int, int], Any] | None = None,
        tag: int = 0,
        obs_dtype: Any = np.float32,
    ) -> None:
        self.scripts = [list(script)] if isinstance(script[0], Tick) else [list(s) for s in script]
        act = action_space if action_space is not None else gymnasium.spaces.Discrete(3)
        gs = SCRIPT_GS_SPACE if global_state else None
        self.obs_dtype = np.dtype(obs_dtype)
        obs = SCRIPT_OBS_SPACE if self.obs_dtype == np.float32 else gymnasium.spaces.Box(
            0, 255, (5,), self.obs_dtype)
        if num_seats == 1:
            self.spec = GameSpec.solo(obs, act, gs)
        else:
            self.spec = GameSpec.symmetric(num_seats, obs, act, gs)
        self.layout = next(iter(self.spec.layouts))
        self.num_seats = num_seats
        self.global_state_enabled = global_state
        self.mask_fn = mask_fn
        self.tag = tag
        self.k = -1
        self.t = 0
        self.eliminated: set[int] = set()
        self.log: list[tuple[int, int, int | None, dict | None]] = []

    def _obs(self, seat: int, final: bool = False) -> np.ndarray:
        return np.array([self.tag, self.k, self.t, seat, 1.0 if final else 0.0], self.obs_dtype)

    def _gs(self, seat: int) -> np.ndarray:
        return np.array([self.k, self.t + 0.5 * seat], np.float32)

    def _result(self, tick: Tick) -> StepResult:
        acting = set(tick.acting)
        res = StepResult(acting=acting, obs={s: self._obs(s) for s in acting})
        if self.mask_fn is not None:
            res.action_masks = {s: self.mask_fn(self.k, self.t, s) for s in acting}
        if self.global_state_enabled:
            res.global_state = {s: self._gs(s) for s in acting}
        return res

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.k += 1
        self.t = 0
        self.eliminated = set()
        self.log.append((self.k, 0, seed, None))
        return self._result(self.scripts[self.k % len(self.scripts)][0])

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        self.log.append((self.k, self.t, None, dict(actions)))
        tick = self.scripts[self.k % len(self.scripts)][self.t]
        res = self._result(tick)
        res.rewards = dict(tick.rewards)
        res.terminated = set(tick.terminated)
        res.episode_over = tick.over
        res.truncated = tick.truncated
        res.outcome = tick.outcome
        self.eliminated |= res.terminated
        if tick.truncated:
            live = [s for s in range(self.num_seats) if s not in self.eliminated]
            res.final_obs = {s: self._obs(s, final=True) for s in live}
            if self.global_state_enabled:
                res.global_state = {s: self._gs(s) for s in live}
        return res


class DictModelPool:
    """ModelPool over a plain ``{(agent_id, network_id): model}`` dict."""

    def __init__(self, models: Mapping[tuple[str, str], Any]) -> None:
        self.models = dict(models)

    def get(self, agent_id: str, network_id: str):
        return self.models.get((agent_id, network_id))


class RecordingObserver:
    """MatchObserver that appends every event to ``events`` as a tuple."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_act(self, env, seat, record):
        self.events.append(("act", env, seat, record))

    def on_rewards(self, env, rewards):
        self.events.append(("rewards", env, dict(rewards)))

    def on_terminated(self, env, seats):
        self.events.append(("terminated", env, list(seats)))

    def on_episode_end(self, env, end):
        self.events.append(("end", env, end))

    def on_lineup_applied(self, env, old, new):
        self.events.append(("lineup", env, old, new))

    def kinds(self, env: int | None = None) -> list[str]:
        return [e[0] for e in self.events if env is None or e[1] == env]

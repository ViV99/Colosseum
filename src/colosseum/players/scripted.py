"""Scripted bots (SP3 spec block 2).

A bot is a ``ScriptedBot`` subclass built from ``agents.<id>.class`` and ``kwargs``. The framework sets
``game_spec`` right after construction, calls ``reset`` at the start of every episode in which the bot
sits at a seat (``rng`` from :func:`bot_rng`), and ``act`` on every turn of that seat with numpy trees:
the observation (cast to the role's dtypes), the normalized action mask (or None) and
``StepResult.infos.get(seat)`` of the latest step (or None). A bot instance belongs to one
(agent, env, seat) and lives across episodes; per-episode memory goes into ``self``.

Every action passes :func:`check_bot_action`, the same legality gate as recorded actions
(``ActionSpec.first_illegal_action``, which applies ``units_component_valid`` to ``Units``).
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from colosseum.core.errors import PlayerError
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import Tree, tree_map, tree_to_torch
from colosseum.core.validation import random_legal_action
from colosseum.envs.game import GameSpec, RoleSpec

__all__ = ["RandomBot", "ScriptedBot", "bot_rng", "check_bot_action"]


class ScriptedBot(ABC):
    """Base class of rule-based bots (module docstring)."""

    game_spec: GameSpec  # set by the framework right after construction, before the first reset

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        """Start of an episode in which this bot sits at ``seat`` (role ``role``) of ``layout``."""
        return None

    @abstractmethod
    def act(self, obs: Tree, mask: Tree | None, info: Any) -> Tree:
        """An action tree in the role's action space for the current observation."""


class RandomBot(ScriptedBot):
    """A uniformly random legal action (``core.validation.random_legal_action``) from the episode RNG."""

    def __init__(self) -> None:
        self._role: RoleSpec | None = None
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._role = self.game_spec.roles[role]
        self._rng = rng

    def act(self, obs: Tree, mask: Tree | None, info: Any) -> Tree:
        if self._role is None:
            raise RuntimeError("RandomBot.act before reset")
        return random_legal_action(self._role, mask, self._rng)


def bot_rng(episode_seed: int | None, seat: int, agent_id: str) -> np.random.Generator:
    """The RNG a bot gets at ``reset``: ``SeedSequence([episode_seed, seat, crc32(agent_id)])``, so bots are
    reproducible under ``training.seed``; fresh entropy when ``episode_seed`` is None."""
    if episode_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence(
        [int(episode_seed) % 2**64, int(seat), zlib.crc32(agent_id.encode("utf-8"))]))


def check_bot_action(role: RoleSpec, action: Tree, mask: Tree | None, where: str, *,
                     action_spec: ActionSpec | None = None) -> Tree:
    """The bot's ``action`` as numpy leaves with the role's dtypes, or ``PlayerError(f"{where}: ...")``.

    Checks, in order: the tree structure of the role's action space (dict key order does not matter),
    each component's kind (integer components take integers) and shape, ``action_space.contains``
    (ranges and bounds), and under a normalized ``mask`` ``ActionSpec.first_illegal_action``.
    """
    spec = action_spec if action_spec is not None else ActionSpec.from_space(role.action_space)

    def cast(zero: np.ndarray, leaf: Any) -> np.ndarray:
        if leaf is None:
            raise PlayerError(f"{where}: the action has None where the role's action space expects a value")
        try:
            arr = np.asarray(leaf)
        except ValueError as e:  # e.g. a ragged nested list
            raise PlayerError(f"{where}: an action component is not an array of one shape ({e})") from e
        if np.issubdtype(zero.dtype, np.integer) and arr.dtype.kind not in "iub":
            raise PlayerError(f"{where}: an integer action component got {arr.dtype} values ({arr!r})")
        if np.issubdtype(zero.dtype, np.floating) and arr.dtype.kind not in "iuf":
            raise PlayerError(f"{where}: a float action component got {arr.dtype} values ({arr!r})")
        if arr.shape != zero.shape:
            raise PlayerError(f"{where}: an action component has shape {arr.shape}, the role's action space "
                              f"expects {zero.shape}")
        return arr.astype(zero.dtype, copy=True)

    try:
        out = tree_map(cast, spec.allocate_actions(()), action)
    except ValueError as e:  # tree_map: dict keys or leaf/dict structure differ
        raise PlayerError(f"{where}: the action does not have the structure of the role's action space "
                          f"{role.action_space} ({e})") from e
    if not role.action_space.contains(out):
        raise PlayerError(f"{where}: action {out!r} is not in the role's action space {role.action_space} "
                          f"(out of range or bounds)")
    if mask is not None:
        found = spec.first_illegal_action(
            tree_to_torch(tree_map(lambda leaf: leaf[None], out)),
            tree_to_torch(tree_map(lambda m: None if m is None else np.asarray(m)[None], mask)))
        if found is not None:
            raise PlayerError(f"{where}: illegal action: {found[1]}")
    return out

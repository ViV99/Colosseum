"""Scripted and frozen agents of a config as picklable specs, their roles, and their construction
(SP3 spec blocks 1 and 3).

Nothing here holds torch tensors or bot instances: ``BotSpec`` (class path + kwargs) and ``FrozenSpec``
(numpy weights + the networks section) cross process boundaries; every process builds its own bots
(``make_bot``) and frozen models (``build_frozen_model``). ``load_frozen`` is the one loader of fixed
weights (frozen agents, ``eval -a``, and later kickstart teachers and ``init``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from colosseum.coordinator.checkpoint_manager import (
    check_model_state,
    load_checkpoint_dir,
    read_checkpoint_meta,
    read_weights_file,
)
from colosseum.core.config import ColosseumConfig, FrozenAgent, NetworkConfig, deep_merge
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_network, import_class
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.core.types import state_dict_from_numpy
from colosseum.envs.game import GameSpec
from colosseum.networks.model import PolicyModel
from colosseum.players.scripted import ScriptedBot

__all__ = ["BotSpec", "FixedPlayers", "FrozenSpec", "build_frozen_model", "load_fixed_players", "load_frozen",
           "make_bot", "resolve_player_roles"]


@dataclass(frozen=True)
class BotSpec:
    """A scripted bot as data: its class path and constructor kwargs (picklable)."""

    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenSpec:
    """Fixed weights with everything needed to build their model in any process.

    ``networks`` is a ``NetworkConfig`` dump (by alias); ``source`` the path the weights came from;
    ``networks_source`` says where ``networks`` came from (prefix of architecture errors).
    """

    agent_id: str
    roles: tuple[str, ...]
    networks: dict[str, Any]
    model_state: dict[str, np.ndarray]
    source: str
    networks_source: str = ""


@dataclass(frozen=True)
class FixedPlayers:
    """Every scripted (``bots``) and frozen (``frozen``) agent of a config, and their roles."""

    bots: dict[str, BotSpec]
    frozen: dict[str, FrozenSpec]
    roles: dict[str, tuple[str, ...]]


def make_bot(bot: BotSpec, spec: GameSpec) -> ScriptedBot:
    """Import and construct a bot, then set its ``game_spec``; ConfigError for a bad class or kwargs."""
    try:
        cls = import_class(bot.class_path)
    except Exception as e:  # noqa: BLE001 - any import failure is a config problem
        raise ConfigError(f"scripted bot class {bot.class_path!r} cannot be imported ({type(e).__name__}: {e}); "
                          f"use a dotted path 'package.module.Class' importable from the working directory") from e
    if not issubclass(cls, ScriptedBot):
        raise ConfigError(f"scripted bot class {bot.class_path!r} must subclass colosseum.players.ScriptedBot, "
                          f"got {cls.__name__}")
    try:
        instance = cls(**bot.kwargs)
    except Exception as e:  # noqa: BLE001 - a constructor failure from config kwargs
        raise ConfigError(f"scripted bot {bot.class_path!r}: constructing it with kwargs {bot.kwargs} failed "
                          f"({type(e).__name__}: {e})") from e
    instance.game_spec = spec
    return instance


def _check_role_names(spec: GameSpec, agent_id: str, roles: list[str]) -> None:
    unknown = [r for r in roles if r not in spec.roles]
    if unknown:
        raise ConfigError(f"agents.{agent_id}.roles: unknown roles {unknown}; the game has roles {list(spec.roles)}")


def _bad_path(agent_id: str, path: str) -> str:
    return (f"agents.{agent_id}.path={path!r}: expected a checkpoint dir (with model.pt and meta.json) or a "
            f".pt state_dict file")


def _check_dir_entry(agent_id: str, entry: FrozenAgent | None, path: Path) -> None:
    """A frozen agent with a checkpoint dir takes roles and networks from its ``meta.json`` only."""
    if entry is not None and (entry.networks is not None or entry.roles is not None):
        raise ConfigError(f"agents.{agent_id}: {path} is a checkpoint dir, whose meta.json gives the roles and "
                          f"networks; remove agents.{agent_id}.networks / roles (they are for .pt files)")


def _pt_roles(config: ColosseumConfig, spec: GameSpec, name: str, path: Path, explicit: list[str] | None) -> list[str]:
    """Roles of a ``.pt`` player: a trainable agent's roles; else ``explicit`` (one set of spaces); else every
    role of the game, which then must share one set of spaces."""
    if name in config.get_trainable_agent_ids():
        return resolve_agent_roles(config, spec)[name]
    if explicit is not None:
        _check_role_names(spec, name, explicit)
        if len({role_signature(spec.roles[r]) for r in explicit}) > 1:
            raise ConfigError(f"agents.{name}.roles: roles {explicit} have different spaces; one network serves all "
                              f"roles of an agent, so list roles with the same spaces")
        return list(explicit)
    roles = list(spec.roles)
    if len({role_signature(spec.roles[r]) for r in roles}) > 1:
        raise ConfigError(
            f"{path}: agent {name!r}: a .pt file plays every role of the game unless agents.{name}.roles says "
            f"otherwise, but the roles {roles} have different spaces; set agents.{name}.roles (kind: frozen), "
            f"name the agent after a configured agent, or use a checkpoint dir"
        )
    return roles


def _checkpoint_roles(spec: GameSpec, path: Path, meta: dict) -> list[str]:
    roles, signature = meta.get("roles"), meta.get("role_signature")
    if roles is None or signature is None:
        raise ConfigError(f"Checkpoint {path}: meta.json has no roles/role_signature (not an SP2 checkpoint)")
    unknown = sorted(set(roles) - set(spec.roles))
    if unknown:
        raise ConfigError(f"Checkpoint {path}: roles {unknown} are not roles of the game {sorted(spec.roles)}")
    for role in roles:  # every role: one network serves them all
        expected = role_signature(spec.roles[role])
        if signature != expected:
            raise ConfigError(
                f"Checkpoint {path}: role signature {signature!r} does not match the game's spaces for role "
                f"{role!r} ({expected!r}); it was trained on a different game or game version"
            )
    return list(roles)


def resolve_player_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]:
    """``{agent_id: roles}`` for EVERY agent in config order (``ColosseumConfig.agent_ids``).

    Trainable: ``core.roles.resolve_agent_roles``. Scripted: ``roles`` or every role of the game (their
    spaces may differ). Frozen: a checkpoint dir's ``meta.json`` roles (checked against the game; the entry
    must not set ``roles`` / ``networks``), or for a ``.pt`` the ``roles`` field / every role with one set
    of spaces. ConfigError with a hint otherwise.
    """
    trainable = resolve_agent_roles(config, spec)
    out: dict[str, list[str]] = {}
    for agent_id in config.agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind == "trainable":
            out[agent_id] = list(trainable[agent_id])
        elif entry.kind == "scripted":
            if entry.roles is not None:
                _check_role_names(spec, agent_id, entry.roles)
            out[agent_id] = list(entry.roles) if entry.roles is not None else list(spec.roles)
        else:
            path = Path(entry.path)
            if path.is_dir():
                _check_dir_entry(agent_id, entry, path)
                out[agent_id] = _checkpoint_roles(spec, path, read_checkpoint_meta(path))
            elif path.is_file() and path.suffix == ".pt":
                out[agent_id] = _pt_roles(config, spec, agent_id, path, entry.roles)
            else:
                raise ConfigError(_bad_path(agent_id, entry.path))
    return out


def load_frozen(config: ColosseumConfig, agent_id: str, path: str | Path, spec: GameSpec) -> FrozenSpec:
    """Read fixed weights for the player ``agent_id`` from a checkpoint dir or a ``.pt`` (read-only).

    - Checkpoint dir (``load_checkpoint_dir``): roles and role signature from ``meta.json`` (required; checked
      against the game), networks from ``meta.json`` (else the agent's / global networks). A frozen agent
      with a checkpoint dir must not set ``networks`` or ``roles``.
    - ``.pt``: roles by ``_pt_roles``; networks: a trainable agent's effective networks, a frozen agent's
      ``networks`` override deep-merged onto the global networks, else the global networks.

    ConfigError naming the path; a path that is neither raises FileNotFoundError (callers check first).
    """
    p = Path(path)
    entry = config.agent_entry(agent_id) if agent_id in config.agent_ids() else None
    if entry is not None and entry.kind == "scripted":
        raise ConfigError(f"agent {agent_id!r} is a scripted agent: it has no weights to load from {p}")
    frozen_entry = entry if isinstance(entry, FrozenAgent) else None
    base = config.get_agent_config(agent_id).networks if entry is not None and entry.kind == "trainable" \
        else config.networks
    if p.is_dir():
        _check_dir_entry(agent_id, frozen_entry, p)
        loaded = load_checkpoint_dir(p)
        roles = _checkpoint_roles(spec, p, loaded["meta"])
        raw = loaded["meta"].get("networks")
        if raw is not None:
            try:
                networks = NetworkConfig.model_validate(raw)
            except ValidationError as e:
                raise ConfigError(f"Checkpoint {p}: invalid meta.json networks:\n{e}") from e
            networks_source = f"Checkpoint {p}: meta.json networks"
        else:
            networks, networks_source = base, f"Checkpoint {p}: networks of the config"
        model_state = loaded["model_state"]
    elif p.is_file() and p.suffix == ".pt":
        roles = _pt_roles(config, spec, agent_id, p, frozen_entry.roles if frozen_entry is not None else None)
        networks = base
        if frozen_entry is not None and frozen_entry.networks:
            try:
                networks = NetworkConfig.model_validate(
                    deep_merge(config.networks.model_dump(by_alias=True), frozen_entry.networks))
            except ValidationError as e:
                raise ConfigError(f"agents.{agent_id}.networks: invalid override:\n{e}") from e
        networks_source = f"{p}: networks"
        try:
            model_state = read_weights_file(p)
        except ValueError as e:
            raise ConfigError(f"{p}: {e}") from e
    else:
        raise FileNotFoundError(f"{p}: expected a checkpoint directory or a .pt file")
    return FrozenSpec(agent_id=agent_id, roles=tuple(roles), networks=networks.model_dump(mode="json", by_alias=True),
                      model_state=model_state, source=str(p), networks_source=networks_source)


def build_frozen_model(config: ColosseumConfig | None, frozen: FrozenSpec, spec: GameSpec) -> PolicyModel:
    """The frozen player's model, weights loaded, in eval mode. ``config`` is not needed (``frozen`` holds the
    networks) and may be None, e.g. in a worker process. ConfigError naming ``frozen.source`` when the model
    cannot be built or the weights do not fit it."""
    role = agent_role_spec(spec, list(frozen.roles))
    try:
        model = build_network(NetworkConfig.model_validate(frozen.networks), role)
    except ConfigError as e:
        raise ConfigError(f"{frozen.source}: {e}") from e
    except Exception as e:  # noqa: BLE001 - a networks section is user data: any constructor failure
        raise ConfigError(f"{frozen.source}: cannot build the agent's model ({type(e).__name__}: {e})") from e
    check_model_state(model, frozen.model_state, frozen.source)
    model.load_state_dict(state_dict_from_numpy(frozen.model_state))
    model.eval()
    return model


def load_fixed_players(config: ColosseumConfig, spec: GameSpec) -> FixedPlayers:
    """Every scripted and frozen agent of ``config`` (main process; before any child starts).

    Scripted bots are imported and constructed once (a bad class or kwargs fail here); frozen weights are
    read with ``load_frozen`` (roles and signatures checked). Models are built where they run.
    """
    roles = resolve_player_roles(config, spec)
    bots: dict[str, BotSpec] = {}
    frozen: dict[str, FrozenSpec] = {}
    for agent_id in config.fixed_agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind == "scripted":
            bot = BotSpec(entry.class_path, dict(entry.kwargs))
            make_bot(bot, spec)
            bots[agent_id] = bot
        else:
            frozen[agent_id] = load_frozen(config, agent_id, entry.path, spec)
    return FixedPlayers(bots=bots, frozen=frozen, roles={a: tuple(roles[a]) for a in config.fixed_agent_ids()})

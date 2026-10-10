"""Dynamic import, env construction and model building for config v2 (SP2 T2.4).

``build_model`` injects the role's spaces into the classes that ask for them by name:
``observation_space``, ``action_space``, ``global_state_space`` and ``action_spec`` are
passed to a constructor only if its signature names that parameter explicitly (the same
probing rule as SP1's ``in_dim``), and never when ``networks.kwargs`` already sets it.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from colosseum.core.errors import ConfigError

if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig, NetworkConfig
    from colosseum.envs.game import GameSpec, MultiAgentEnv, RoleSpec
    from colosseum.networks.model import PolicyModel

_INJECTABLE = ("observation_space", "action_space", "global_state_space", "action_spec")


def import_class(dotted_path: str) -> type:
    """Import and return a class from ``"package.module.ClassName"``.

    Raises ValueError (no dot), ModuleNotFoundError, AttributeError, or TypeError (not a class).
    """
    if "." not in dotted_path:
        raise ValueError(f"dotted_path must be in the form 'module.ClassName', got: {dotted_path!r}")
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"{dotted_path!r} resolved to {type(cls).__name__}, not a class")
    return cls


def _params(cls: type) -> set[str]:
    try:
        return set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return set()


def _inject(cls: type, role: RoleSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The role values ``cls.__init__`` names explicitly and ``kwargs`` does not already set."""
    from colosseum.core.specs import ActionSpec

    available = {
        "observation_space": role.observation_space,
        "action_space": role.action_space,
        "global_state_space": role.global_state_space,
        "action_spec": ActionSpec.from_space(role.action_space),
    }
    accepted = _params(cls)
    return {name: available[name] for name in _INJECTABLE if name in accepted and name not in kwargs}


def _construct(cls: type, role: RoleSpec, kwargs: dict[str, Any]) -> Any:
    return cls(**_inject(cls, role, kwargs), **kwargs)


def _with_in_dim(cls: type, kwargs: dict[str, Any], in_dim: int) -> dict[str, Any]:
    """``kwargs`` plus ``in_dim`` when ``cls.__init__`` names it (the framework's value wins)."""
    return {**kwargs, "in_dim": in_dim} if "in_dim" in _params(cls) else dict(kwargs)


def make_env(config: ColosseumConfig) -> MultiAgentEnv:
    """Instantiate ``config.env``; any failure becomes a ConfigError."""
    from colosseum.envs.game import MultiAgentEnv

    try:
        env = import_class(config.env.env_class)(**config.env.kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to create env {config.env.env_class!r}: {type(e).__name__}: {e}") from e
    if not isinstance(env, MultiAgentEnv):
        raise ConfigError(f"env.env_class {config.env.env_class!r} must subclass "
                          f"colosseum.envs.game.MultiAgentEnv, got {type(env).__name__}")
    return env


def env_spec(config: ColosseumConfig) -> GameSpec:
    """The env's ``GameSpec``: built once from a fresh env, validated (EnvContractError), env closed."""
    from colosseum.envs.game import GameSpec

    env = make_env(config)
    try:
        spec = getattr(env, "spec", None)
        if not isinstance(spec, GameSpec):
            raise ConfigError(f"{config.env.env_class}.spec must be a colosseum.envs.game.GameSpec, "
                              f"got {type(spec).__name__}")
        spec.validate()
        return spec
    finally:
        env.close()


def build_model(agent_config: ColosseumConfig, role: RoleSpec) -> PolicyModel:
    """Build the agent's ``PolicyModel`` for ``role``'s spaces from ``agent_config.networks`` (``build_network``)."""
    return build_network(agent_config.networks, role)


def build_network(net: NetworkConfig, role: RoleSpec) -> PolicyModel:
    """Build a ``PolicyModel`` for ``role``'s spaces from a ``networks`` section.

    - ``model_class``: ``cls(**inject, **networks.kwargs)``; it must be a ``PolicyModel``.
    - otherwise a ``ComposedModel``: ``encoder(**inject, **kwargs)``;
      ``core(input_dim=encoder.latent_dim, **core.kwargs)`` (``NoCore`` when ``core`` is null);
      ``critic_encoder(**inject, **kwargs)`` if set (its role must declare a global state);
      ``policy(in_dim=core.output_dim, **inject, **kwargs)``;
      ``value(in_dim=core.output_dim + critic.output_dim, **kwargs)``. ``in_dim`` is passed only to
      heads that name it.
    """
    from colosseum.networks.base import BaseCriticEncoder
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.cores import Core, NoCore
    from colosseum.networks.model import PolicyModel

    kwargs = dict(net.kwargs)
    if net.model_class:
        model_cls = import_class(net.model_class)
        if not issubclass(model_cls, PolicyModel):
            raise ConfigError(f"networks.model_class {net.model_class!r} must subclass "
                              f"colosseum.networks.model.PolicyModel, got {model_cls.__name__}")
        return _construct(model_cls, role, kwargs)

    encoder = _construct(import_class(net.encoder_class), role, kwargs)
    if net.core is None:
        core = NoCore(encoder.latent_dim)
    else:
        core_cls = import_class(net.core.class_path)
        if not issubclass(core_cls, Core):
            raise ConfigError(f"networks.core.class {net.core.class_path!r} must subclass "
                              f"colosseum.networks.cores.Core, got {core_cls.__name__}")
        core = core_cls(input_dim=encoder.latent_dim, **net.core.kwargs)
    critic = None
    critic_dim = 0
    if net.critic_encoder_class:
        if role.global_state_space is None:
            raise ConfigError(f"networks.critic_encoder_class is set ({net.critic_encoder_class!r}) but the agent's "
                              f"role declares no global_state_space; remove it or add a global state to the role")
        critic_cls = import_class(net.critic_encoder_class)
        if not issubclass(critic_cls, BaseCriticEncoder):
            raise ConfigError(f"networks.critic_encoder_class {net.critic_encoder_class!r} must subclass "
                              f"colosseum.networks.base.BaseCriticEncoder, got {critic_cls.__name__}")
        critic = _construct(critic_cls, role, kwargs)
        critic_dim = int(critic.output_dim)
    policy_cls = import_class(net.policy_class)
    policy = _construct(policy_cls, role, _with_in_dim(policy_cls, kwargs, core.output_dim))
    value_cls = import_class(net.value_class)
    value = value_cls(**_with_in_dim(value_cls, kwargs, core.output_dim + critic_dim))
    return ComposedModel(encoder, core, policy, value, critic)


def validate_config(config: ColosseumConfig) -> None:
    """Every check that can run before a run starts (spec block 9); see ``colosseum.core.validation``.

    Callers use this name (``registry.validate_config``), so tests can monkeypatch it here.
    """
    from colosseum.core.validation import validate_config as _validate_config

    _validate_config(config)

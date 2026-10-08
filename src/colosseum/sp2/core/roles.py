"""Agents and roles (SP2 spec block 5, "Роли и агенты").

An agent plays one or more roles; all its roles must have the same observation, action
and global-state spaces (one network serves them). ``agents.<id>.roles`` may be omitted
when every role of the game has the same spaces: the agent then plays every role.
"""

from __future__ import annotations

from collections.abc import Sequence

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.envs.game import GameSpec, RoleSpec


def role_signature(role: RoleSpec) -> str:
    """Identity of a role's spaces: observation, action and global-state signatures."""
    gs = "none" if role.global_state_space is None else ObsSpec.from_space(role.global_state_space).signature()
    return (f"obs={ObsSpec.from_space(role.observation_space).signature()}"
            f"|act={ActionSpec.from_space(role.action_space).signature()}|gs={gs}")


def resolve_agent_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]:
    """``{agent_id: roles}`` for every trainable agent; ConfigError with a fix hint on any mismatch."""
    signatures = {name: role_signature(role) for name, role in spec.roles.items()}
    out: dict[str, list[str]] = {}
    for agent_id in config.get_trainable_agent_ids():
        roles = config.agent_roles(agent_id)
        if roles is None:
            roles = list(spec.roles)
            if len({signatures[r] for r in roles}) > 1:
                raise ConfigError(
                    f"agent {agent_id!r} plays every role by default, but the roles {roles} have different "
                    f"spaces: set agents.{agent_id}.roles; roles with different spaces need separate agents"
                )
        else:
            unknown = [r for r in roles if r not in spec.roles]
            if unknown:
                raise ConfigError(f"agents.{agent_id}.roles: unknown roles {unknown}; the game has roles "
                                  f"{list(spec.roles)}")
            first = roles[0]
            for r in roles[1:]:
                if signatures[r] != signatures[first]:
                    raise ConfigError(
                        f"agents.{agent_id}.roles: roles {first!r} and {r!r} have different spaces "
                        f"({signatures[first]} vs {signatures[r]}); roles with different spaces need separate agents"
                    )
        out[agent_id] = list(roles)
    return out


def agent_role_spec(spec: GameSpec, roles: Sequence[str]) -> RoleSpec:
    """The spaces an agent's network is built for (all its roles share them)."""
    return spec.roles[roles[0]]

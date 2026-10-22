"""What a run needs before the hub starts anything (SP5 T2.5; controller ruling R27): ``RunSetup`` / ``setup_run``
(the game's spec, every agent's roles, effective configs and role specs, the fixed players), ``validate_run_config``,
``warn_static_ownership_skew`` and ``_record_run_state``. Moved unchanged from the SP4 launcher (``launcher.py``,
removed in T2.5); ``colosseum.hub.hub`` re-exports them (the contract names ``colosseum.hub.hub.setup_run`` etc.)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from colosseum.core.config import ColosseumConfig
from colosseum.core.run_dir import RunDir
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.players.registry import FixedPlayers

logger = logging.getLogger(__name__)

__all__ = ["RunSetup", "setup_run", "validate_run_config", "warn_static_ownership_skew"]


def warn_static_ownership_skew(config: ColosseumConfig) -> None:
    """Warn when static matchmaking splits envs unevenly between trainable agents.

    Env ``g`` is owned by ``agents[(g + refresh_round) % n]``. With
    ``match_refresh_interval_sec == 0`` the round never advances, so if the env count
    is not a multiple of the agent count (or smaller than it) some agents get
    permanently more training data than others, or none at all.
    """
    n = len(config.get_trainable_agent_ids())
    total_envs = config.rollout.num_workers * config.rollout.envs_per_worker
    if config.rollout.match_refresh_interval_sec == 0 and n > 1 and total_envs % n != 0:
        logger.warning(
            f"rollout.match_refresh_interval_sec=0 with {total_envs} envs "
            f"(num_workers * envs_per_worker) and {n} trainable agents: env ownership never "
            f"rotates, so data per agent is skewed"
            + (" and some agents get no data" if total_envs < n else "")
            + ". Use a multiple of the agent count or enable match refresh."
        )


@dataclass(frozen=True)
class RunSetup:
    """What a run needs before any process starts: the game's spec, every agent's roles
    (``player_roles``), the trainable agents' roles, effective configs (overrides merged) and role
    specs (the spaces their models are built for), and the fixed players (loaded once in the main
    process)."""

    spec: GameSpec
    agent_roles: dict[str, list[str]]
    agent_configs: dict[str, ColosseumConfig]
    role_specs: dict[str, RoleSpec]
    player_roles: dict[str, list[str]]
    fixed: FixedPlayers


def validate_run_config(config: ColosseumConfig) -> None:
    """Fail fast (ConfigError / EnvContractError) before any process starts: ``registry.validate_config``
    (looked up at call time, so tests can count or replace it)."""
    from colosseum.core import registry

    registry.validate_config(config)


def setup_run(config: ColosseumConfig, validate: bool = True) -> RunSetup:
    """``RunSetup`` of ``config``; ``validate`` first runs ``validate_run_config``."""
    from colosseum.core.registry import env_spec
    from colosseum.core.roles import agent_role_spec
    from colosseum.players.registry import load_fixed_players, resolve_player_roles

    if validate:
        validate_run_config(config)
    spec = env_spec(config)
    player_roles = resolve_player_roles(config, spec)
    agent_ids = config.get_trainable_agent_ids()
    return RunSetup(
        spec=spec,
        agent_roles={aid: list(player_roles[aid]) for aid in agent_ids},
        agent_configs={aid: config.get_agent_config(aid) for aid in agent_ids},
        role_specs={aid: agent_role_spec(spec, player_roles[aid]) for aid in agent_ids},
        player_roles=player_roles,
        fixed=load_fixed_players(config, spec),
    )


def _record_run_state(run_dir: RunDir, status: str, *, exit_code: int | None = None) -> None:
    """run_state.json for report / dashboard (spec block 5); a failure to write it never stops a run."""
    from colosseum.core.run_state import write_run_state

    try:
        write_run_state(run_dir.root, status, os.getpid(), exit_code=exit_code)
    except OSError as e:
        logger.warning(f"cannot write {run_dir.run_state_path}: {e}")

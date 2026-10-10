"""Matchmaker interface (spec block 5): ``BaseMatchmaker`` and its read-only ``MatchmakerContext``.

A matchmaker builds the ``Lineup`` of one env whose training data belongs to ``owner``
(``lineup_for``); the coordinator checks every lineup (``check_lineup``) and passes every
finished match to ``on_result``. The context gives the game, the agents with their kinds and
roles, each trainable agent's effective ``matchmaking`` config, the stored snapshots, the PFSP
statistics (read only), the coordinator's env-step count and the run's RNG.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from colosseum.core.config import AgentKind, ColosseumConfig, MatchmakingConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.league.pfsp import PlayerKey
from colosseum.league.schedule import schedule_value


@dataclass(frozen=True)
class AgentView:
    """An agent as a matchmaker sees it."""

    agent_id: str
    kind: AgentKind
    roles: frozenset[str]


def _no_snapshots(agent_id: str) -> list[str]:
    return []


def _zero_env_steps() -> int:
    return 0


@dataclass
class MatchmakerContext:
    """What a matchmaker may read (spec block 5). ``agents`` and ``trainable`` are in config order;
    ``matchmaking`` holds every trainable agent's effective config (per-agent overrides merged)."""

    spec: GameSpec
    agents: dict[str, AgentView]
    trainable: list[str]
    matchmaking: dict[str, MatchmakingConfig]
    rng: random.Random
    snapshots_fn: Callable[[str], Sequence[str]] = _no_snapshots
    pfsp_fn: Callable[[str, str, PlayerKey], float] | None = None
    env_steps_fn: Callable[[], int] = _zero_env_steps

    @classmethod
    def from_config(cls, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]], *,
                    rng: random.Random | None = None,
                    snapshots_fn: Callable[[str], Sequence[str]] = _no_snapshots,
                    pfsp_fn: Callable[[str, str, PlayerKey], float] | None = None,
                    env_steps_fn: Callable[[], int] = _zero_env_steps) -> MatchmakerContext:
        """The context of a run: every agent of ``player_roles`` (``resolve_player_roles``) with its kind."""
        trainable = config.get_trainable_agent_ids()
        missing = [a for a in [*trainable, *config.fixed_agent_ids()] if a not in player_roles]
        if missing:
            raise ConfigError(f"matchmaking: no roles resolved for agents {missing}")
        agents = {aid: AgentView(aid, config.agent_kind(aid), frozenset(roles)) for aid, roles in player_roles.items()}
        return cls(
            spec=spec, agents=agents, trainable=list(trainable),
            matchmaking={aid: config.get_agent_config(aid).matchmaking for aid in trainable},
            rng=rng if rng is not None else random.Random(config.training.seed),
            snapshots_fn=snapshots_fn, pfsp_fn=pfsp_fn, env_steps_fn=env_steps_fn,
        )

    @property
    def fixed(self) -> list[str]:
        """Scripted and frozen agents, config order."""
        return [a for a, view in self.agents.items() if view.kind != "trainable"]

    def snapshots(self, agent_id: str) -> list[str]:
        """Stored snapshot ids of a trainable agent, ascending version ([] for other agents)."""
        view = self.agents.get(agent_id)
        if view is None or view.kind != "trainable":
            return []
        return list(self.snapshots_fn(agent_id))

    def pfsp_score(self, layout: str, owner: str, player: PlayerKey) -> float:
        """EMA score of ``owner``@latest against ``player`` in ``layout`` (0.5 before any game)."""
        return 0.5 if self.pfsp_fn is None else float(self.pfsp_fn(layout, owner, player))

    def env_steps(self) -> int:
        return int(self.env_steps_fn())

    def anchors(self, owner: str) -> dict[str, float]:
        """``owner``'s effective anchors -> weight at ``env_steps()``."""
        return self.anchors_at(owner, self.env_steps())

    def anchors_at(self, owner: str, env_steps: int) -> dict[str, float]:
        """``owner``'s anchors -> weight at ``env_steps``: ``null`` = every scripted and frozen agent
        (weight 1), a list = weight 1 each, a mapping = its weights or schedules."""
        configured = self.matchmaking[owner].anchors
        if configured is None:
            return {a: 1.0 for a in self.fixed}
        if isinstance(configured, list):
            return {a: 1.0 for a in configured}
        return {a: schedule_value(weight, env_steps) for a, weight in configured.items()}


class BaseMatchmaker(ABC):
    """Base of every matchmaker (``matchmaking.matchmaker_class``). The built-in one is
    ``colosseum.league.mixture.MixtureMatchmaker``."""

    def __init__(self, context: MatchmakerContext) -> None:
        self.context = context

    @abstractmethod
    def lineup_for(self, owner: str) -> Lineup:
        """The lineup of one env whose training data belongs to ``owner`` (a trainable agent)."""

    def on_result(self, result: MatchResult) -> None:
        """Called with every finished match (default: nothing)."""
        return None


def build_matchmaker(cls: type[BaseMatchmaker], context: MatchmakerContext, path: str | None) -> BaseMatchmaker:
    """Construct the run's matchmaker (the coordinator and ``validate`` share it): a ``ConfigError`` the class
    raises passes through; any other exception from its ``__init__`` becomes a ConfigError naming
    ``matchmaking.matchmaker_class`` (``path``; the class name for the built-in one)."""
    try:
        return cls(context)
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001 - user code
        name = path if path is not None else cls.__name__
        raise ConfigError(f"matchmaking.matchmaker_class {name!r}: constructing it failed: "
                          f"{type(e).__name__}: {e}") from e


def load_matchmaker_class(path: str) -> type[BaseMatchmaker]:
    """Import ``matchmaking.matchmaker_class``; ConfigError unless it is a ``BaseMatchmaker`` subclass."""
    from colosseum.core.registry import import_class

    try:
        cls = import_class(path)
    except Exception as e:  # noqa: BLE001 - any import failure is a config problem
        raise ConfigError(f"matchmaking.matchmaker_class {path!r} cannot be imported: {type(e).__name__}: {e}") from e
    if not issubclass(cls, BaseMatchmaker):
        raise ConfigError(f"matchmaking.matchmaker_class {path!r} must subclass colosseum.league.BaseMatchmaker")
    return cls

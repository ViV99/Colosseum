"""Construction of a trainable agent's learner side (spec block 6): algorithm, kickstart teacher,
warm start.

The launcher and ``distributed.run_distributed_learner`` both build algorithms here. Teachers (and,
from T4.2, ``init`` weights) are resolved in the main process (``resolve_teacher``: ConfigError
before any process starts) into numpy dataclasses that cross into the learner process as
``mp.Process`` arguments; the learner turns them into torch objects (``build_algorithm``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.players.registry import BotSpec, FrozenSpec

if TYPE_CHECKING:
    from colosseum.algorithms.base import BaseAlgorithm
    from colosseum.bc.kickstart import KickstartLoss
    from colosseum.networks.model import PolicyModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherSpec:
    """A resolved kickstart teacher (numpy only; picklable)."""

    kind: Literal["neural", "scripted"]
    source: str
    lambda_: float
    decay_steps: int
    kl: Literal["forward", "reverse"]
    frozen: FrozenSpec | None = None      # neural
    bot: BotSpec | None = None            # scripted


def agent_ids(config: ColosseumConfig) -> list[str]:
    """Every agent id of the config (the implicit ``agent_0`` included), config order
    (``ColosseumConfig.agent_ids``)."""
    return config.agent_ids()


def _student_pt(config: ColosseumConfig, agent_id: str, path: str, spec: GameSpec, where: str) -> FrozenSpec:
    """A ``.pt`` teacher (SP2 semantics): the student's networks and roles with the file's weights."""
    from colosseum.coordinator.checkpoint_manager import read_weights_file
    from colosseum.players.registry import resolve_player_roles

    try:
        state = read_weights_file(path)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
    return FrozenSpec(
        agent_id=agent_id, roles=tuple(resolve_player_roles(config, spec)[agent_id]),
        networks=config.get_agent_config(agent_id).networks.model_dump(mode="json", by_alias=True),
        model_state=state, source=str(path),
    )


def _load_frozen(config: ColosseumConfig, agent_id: str, path: str, spec: GameSpec, where: str) -> FrozenSpec:
    from colosseum.players.registry import load_frozen

    try:
        return load_frozen(config, agent_id, path, spec)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from e


def resolve_teacher(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> TeacherSpec | None:
    """The kickstart teacher of trainable ``agent_id`` (effective ``kickstart`` section), or None.

    ``teacher`` is the name of a frozen agent (its own architecture), a ``.pt`` path (the student's
    architecture, SP2) or a checkpoint dir (architecture and roles from its ``meta.json``). The name
    of a trainable agent is a ConfigError with a hint. Main process only (it reads weight files).
    """
    ks = config.get_agent_config(agent_id).kickstart
    if ks.teacher is None:
        return None
    ref = ks.teacher
    where = f"agent {agent_id!r}: kickstart.teacher={ref!r}"
    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "trainable":
            raise ConfigError(
                f"{where} names a trainable agent; a teacher is fixed: to learn from a snapshot, declare it as a "
                f"frozen agent with path (agents.<name>: {{kind: frozen, path: <checkpoint dir>}}) and name that agent"
            )
        if kind == "scripted":
            raise ConfigError(f"{where} names a scripted agent; scripted kickstart teachers are not supported yet")
        entry = config.agent_entry(ref)
        frozen = _load_frozen(config, ref, entry.path, spec, where)
        source = f"frozen agent {ref!r} ({entry.path})"
    else:
        path = Path(ref)
        if path.is_file() and path.suffix == ".pt":
            frozen = _student_pt(config, agent_id, ref, spec, where)
        elif path.is_dir():
            frozen = _load_frozen(config, agent_id, ref, spec, where)
        else:
            raise ConfigError(f"{where} is neither an agent of the config ({agent_ids(config)}) nor an existing .pt "
                              f"file or checkpoint dir")
        source = ref
    return TeacherSpec(kind="neural", source=source, lambda_=float(ks.lambda_), decay_steps=int(ks.decay_steps),
                       kl=ks.kl, frozen=frozen)


def build_teacher_model(agent_config: ColosseumConfig, teacher: TeacherSpec, spec: GameSpec) -> PolicyModel:
    """The neural teacher's model (its own architecture, weights loaded, eval mode); ConfigError when the
    weights do not fit."""
    from colosseum.players.registry import build_frozen_model

    try:
        return build_frozen_model(agent_config, teacher.frozen, spec)
    except ConfigError as e:
        raise ConfigError(f"kickstart teacher {teacher.source}: {e}") from e


def build_kickstart(agent_config: ColosseumConfig, teacher: TeacherSpec | None, spec: GameSpec | None,
                    device: str) -> KickstartLoss | None:
    """The ``KickstartLoss`` of a resolved teacher (None without one)."""
    if teacher is None:
        return None
    from colosseum.bc.kickstart import KickstartLoss

    if teacher.kind != "neural":
        raise ConfigError(f"kickstart teacher {teacher.source}: scripted kickstart teachers are not supported yet")
    model = build_teacher_model(agent_config, teacher, spec)
    model.to(device)
    return KickstartLoss(model, initial_lambda=teacher.lambda_, decay_steps=teacher.decay_steps,
                         direction=teacher.kl)


def build_algorithm(agent_config: ColosseumConfig, role_spec: RoleSpec, spec: GameSpec | None, *, device: str,
                    teacher: TeacherSpec | None) -> BaseAlgorithm:
    """The agent's algorithm (``algorithm.algorithm_class``) around a fresh model for ``role_spec``, with the
    kickstart of ``teacher``. ``spec`` is needed only to build a neural teacher."""
    from colosseum.core.registry import build_model, import_class
    from colosseum.core.specs import ActionSpec

    path = agent_config.algorithm.algorithm_class
    if not path:
        raise ValueError("algorithm.algorithm_class is not set. "
                         "Provide a dotted import path (e.g. 'colosseum.algorithms.appo.APPO').")
    algo_cls = import_class(path)
    model = build_model(agent_config, role_spec)
    kwargs: dict = {"device": device, "pin_memory": agent_config.learner.pin_memory}
    kickstart = build_kickstart(agent_config, teacher, spec, device)
    if kickstart is not None:
        kwargs["kickstart"] = kickstart
        logger.info(f"Kickstart from {teacher.source} (lambda {teacher.lambda_}, decay over {teacher.decay_steps} "
                    f"train steps, {teacher.kind} teacher)")
    return algo_cls(model, agent_config.algorithm, ActionSpec.from_space(role_spec.action_space), **kwargs)

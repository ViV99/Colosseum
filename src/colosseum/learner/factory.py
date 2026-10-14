"""Construction of a trainable agent's learner side (spec block 6): algorithm, kickstart teacher,
warm start.

The launcher and ``distributed.run_distributed_learner`` both build algorithms here. Teachers and
``init`` weights are resolved in the main process (``resolve_teacher``, ``resolve_init``: ConfigError
before any process starts) into numpy dataclasses (``TeacherSpec``, ``InitState``) that cross into the
learner process as ``mp.Process`` arguments; the learner turns them into torch objects
(``build_algorithm``, ``apply_init``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

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


def _check_teacher_roles(config: ColosseumConfig, spec: GameSpec, agent_id: str, teacher_roles: Sequence[str],
                         where: str) -> None:
    from colosseum.players.registry import resolve_player_roles

    student_roles = resolve_player_roles(config, spec)[agent_id]
    missing = [r for r in student_roles if r not in teacher_roles]
    if missing:
        raise ConfigError(f"{where}: the teacher plays roles {list(teacher_roles)}; a teacher must play every role "
                          f"of its student ({list(student_roles)}), missing {missing}")


def resolve_teacher(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> TeacherSpec | None:
    """The kickstart teacher of trainable ``agent_id`` (effective ``kickstart`` section), or None.

    ``teacher`` is the name of a frozen agent (neural, its own architecture) or of a scripted agent
    (DAgger: ``BotSpec``), a ``.pt`` path (the student's architecture, SP2) or a checkpoint dir
    (architecture and roles from its ``meta.json``). The teacher must play every role of its student;
    the name of a trainable agent is a ConfigError with a hint. Main process only (it reads weight files).
    """
    from colosseum.players.registry import resolve_player_roles

    ks = config.get_agent_config(agent_id).kickstart
    if ks.teacher is None:
        return None
    ref = ks.teacher
    where = f"agent {agent_id!r}: kickstart.teacher={ref!r}"
    settings = {"lambda_": float(ks.lambda_), "decay_steps": int(ks.decay_steps), "kl": ks.kl}
    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "trainable":
            raise ConfigError(
                f"{where} names a trainable agent; a teacher is fixed: to learn from a snapshot, declare it as a "
                f"frozen agent with path (agents.<name>: {{kind: frozen, path: <checkpoint dir>}}) and name that agent"
            )
        entry = config.agent_entry(ref)
        if kind == "scripted":
            _check_teacher_roles(config, spec, agent_id, resolve_player_roles(config, spec)[ref], where)
            return TeacherSpec(kind="scripted", source=f"scripted agent {ref!r}",
                               bot=BotSpec(entry.class_path, dict(entry.kwargs)), **settings)
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
    _check_teacher_roles(config, spec, agent_id, frozen.roles, where)
    return TeacherSpec(kind="neural", source=source, frozen=frozen, **settings)


def build_teacher_model(agent_config: ColosseumConfig, teacher: TeacherSpec, spec: GameSpec) -> PolicyModel:
    """The neural teacher's model (its own architecture, weights loaded, eval mode); ConfigError when the
    weights do not fit."""
    from colosseum.players.registry import build_frozen_model

    try:
        return build_frozen_model(agent_config, teacher.frozen, spec)
    except ConfigError as e:
        raise ConfigError(f"kickstart teacher {teacher.source}: {e}") from e


def check_teacher_compat(student: PolicyModel, teacher: PolicyModel, where: str) -> None:
    """ConfigError unless ``teacher`` can be unrolled on ``student``'s chunks (``check_teacher_state_layout``)."""
    from colosseum.bc.kickstart import check_teacher_state_layout

    try:
        check_teacher_state_layout(student, teacher)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e


def build_kickstart(agent_config: ColosseumConfig, teacher: TeacherSpec | None, spec: GameSpec | None,
                    device: str) -> KickstartLoss | None:
    """The ``KickstartLoss`` of a resolved teacher (None without one)."""
    if teacher is None:
        return None
    from colosseum.bc.kickstart import KickstartLoss

    if teacher.kind == "scripted":                 # DAgger: the workers write the labels into the chunks
        return KickstartLoss(None, initial_lambda=teacher.lambda_, decay_steps=teacher.decay_steps)
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
    warmup = agent_config.init.critic_warmup_steps
    if warmup > 0:
        kwargs["critic_warmup_steps"] = warmup
    return algo_cls(model, agent_config.algorithm, ActionSpec.from_space(role_spec.action_space), **kwargs)


# ---------------------------------------------------------------------------
# init (spec block 6): weights only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InitState:
    """Resolved ``init`` weights of a trainable agent (numpy only; picklable). With ``strict=False``
    ``model_state`` holds only the tensors that match the agent's model by name and shape."""

    model_state: dict[str, np.ndarray]
    strict: bool
    source: str
    report: list[str]


_CKPT_DIR_RE = re.compile(r"ckpt_v(\d+)")


def _latest_checkpoint_dir(agent_dir: Path) -> Path | None:
    """The highest-version ``ckpt_v<N>`` dir with a ``model.pt`` (read-only; ``.tmp-*`` leftovers ignored)."""
    if not agent_dir.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in agent_dir.iterdir()
             if p.is_dir() and (m := _CKPT_DIR_RE.fullmatch(p.name)) and (p / "model.pt").is_file()]
    return max(found)[1] if found else None


def _check_init_signature(found: str, signature: str, what: str, where: str) -> None:
    """The ``init`` source's role signature must be the student's (same answer by path and by frozen name)."""
    if found != signature:
        raise ConfigError(f"{where}: {what} has role signature {found!r}, but the agent's roles have "
                          f"{signature!r}: the observation/action/global-state spaces differ")


def _checkpoint_weights(ckpt_dir: Path, signature: str, where: str) -> tuple[dict[str, np.ndarray], str]:
    from colosseum.coordinator.checkpoint_manager import load_checkpoint_dir

    state = load_checkpoint_dir(ckpt_dir)          # read-only; ConfigError naming the dir
    found = state["role_signature"]
    if found is None:
        raise ConfigError(f"{where}: checkpoint {ckpt_dir} has no role_signature in its meta.json (written before "
                          f"SP2); init from its model.pt file instead")
    _check_init_signature(found, signature, f"checkpoint {ckpt_dir}", where)
    return state["model_state"], str(ckpt_dir)


def _init_weights(config: ColosseumConfig, agent_id: str, ref: str, signature: str, spec: GameSpec,
                  where: str) -> tuple[dict[str, np.ndarray], str]:
    """``(numpy state_dict, source)`` of an ``init.from`` reference."""
    from colosseum.coordinator.checkpoint_manager import MODEL_FILE, read_weights_file
    from colosseum.core.roles import agent_role_spec, role_signature

    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "frozen":
            entry = config.agent_entry(ref)
            frozen = _load_frozen(config, ref, entry.path, spec, where)
            source = f"frozen agent {ref!r} ({entry.path})"
            found = role_signature(agent_role_spec(spec, list(frozen.roles)))
            _check_init_signature(found, signature, source, where)
            return dict(frozen.model_state), source
        if kind == "scripted":
            raise ConfigError(f"{where}: {ref!r} is a scripted agent, which has no weights; record its games "
                              f"(colosseum record), train a network on them (colosseum bc) and init from that .pt")
        raise ConfigError(f"{where}: {ref!r} is a trainable agent of this run, which has no weights yet; init from a "
                          f"checkpoint dir or run dir path, or declare that snapshot as a frozen agent with path")
    path = Path(ref)
    if path.is_dir() and (path / "checkpoints").is_dir():
        ckpt = _latest_checkpoint_dir(path / "checkpoints" / agent_id)
        if ckpt is None:
            raise ConfigError(f"{where}: the run dir has no checkpoints of agent {agent_id!r}")
        return _checkpoint_weights(ckpt, signature, where)
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return _checkpoint_weights(path, signature, where)
    if path.is_file() and path.suffix == ".pt":
        try:
            return read_weights_file(path), str(path)
        except ValueError as e:
            raise ConfigError(f"{where}: {e}") from e
    raise ConfigError(f"{where} is neither a frozen agent of the config nor a .pt file, a checkpoint dir (with "
                      f"{MODEL_FILE}) or a run dir (with checkpoints/)")


def resume_source_of(config: ColosseumConfig, agent_id: str) -> str | None:
    """Where ``training.resume_from`` will restore trainable ``agent_id`` from, decided from the file system
    only (nothing loaded): a ``.pt`` or checkpoint-dir source restores every agent; a run dir restores the
    agents with a ``ckpt_v<N>`` (with ``model.pt``) there. None without ``resume_from``, for an agent the
    run dir does not restore, or for a source that is none of these (the launcher's resume reports it)."""
    from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source

    resume_from = config.training.resume_from
    if not resume_from:
        return None
    try:
        kind = classify_resume_source(resume_from)
    except ConfigError:
        return None
    if kind == RESUME_RUN_DIR:
        ckpt = _latest_checkpoint_dir(Path(resume_from) / "checkpoints" / agent_id)
        return str(ckpt) if ckpt is not None else None
    return str(resume_from)


def resolve_init(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> InitState | None:
    """The ``init`` weights of trainable ``agent_id`` (effective ``init`` section), checked against the
    agent's model; None without ``init.from``. Main process only (it reads weight files).

    Strict: every tensor must match by name and shape (ConfigError with the mismatches otherwise).
    Partial: the matching tensors are kept and the rest reported; no match at all is a ConfigError.
    """
    from colosseum.coordinator.checkpoint_manager import check_model_state
    from colosseum.core.registry import build_model
    from colosseum.core.roles import agent_role_spec, role_signature
    from colosseum.players.registry import resolve_player_roles

    agent_config = config.get_agent_config(agent_id)
    init = agent_config.init
    if init.from_ is None:
        return None
    where = f"agent {agent_id!r}: init.from={init.from_!r}"
    role = agent_role_spec(spec, resolve_player_roles(config, spec)[agent_id])
    state, source = _init_weights(config, agent_id, init.from_, role_signature(role), spec, where)
    model = build_model(agent_config, role)
    if init.strict:
        try:
            check_model_state(model, state, source)
        except ConfigError as e:
            raise ConfigError(f"{where}: {e}\nSet init.strict: false to load only the matching tensors") from e
        return InitState(model_state=dict(state), strict=True, source=source,
                         report=[f"agent {agent_id!r}: init from {source} (strict): {len(state)} tensors"])
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(np.asarray(v).shape) for k, v in state.items()}
    matched = [k for k in expected if got.get(k) == expected[k]]
    if not matched:
        raise ConfigError(f"{where}: no tensor of {source} matches the agent's model by name and shape (probably "
                          f"the wrong file)")
    missing = [k for k in expected if k not in got]
    mismatched = [k for k in expected if k in got and got[k] != expected[k]]
    unexpected = sorted(set(got) - set(expected))
    report = [f"agent {agent_id!r}: init from {source} (partial): loaded {len(matched)} of {len(expected)} tensors"]
    if missing:
        report.append(f"  not in the source (kept as initialized): {missing}")
    if mismatched:
        report.append("  shape mismatch (kept as initialized): "
                      + ", ".join(f"{k} {got[k]} vs model {expected[k]}" for k in mismatched))
    if unexpected:
        report.append(f"  not in the model (ignored): {unexpected}")
    return InitState(model_state={k: state[k] for k in matched}, strict=False, source=source, report=report)


def apply_init(model: PolicyModel, init: InitState) -> None:
    """Load ``init`` into a freshly built model (the learner, right after ``build_algorithm``)."""
    from colosseum.core.types import state_dict_from_numpy

    model.load_state_dict(state_dict_from_numpy(init.model_state), strict=init.strict)
    logger.info(f"Init weights loaded from {init.source} ({'strict' if init.strict else 'partial'})")

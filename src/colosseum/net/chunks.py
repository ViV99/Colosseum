"""Chunk, weights and checkpoint frames, and the learner's chunk checks (SP5 spec block 4.9).

- **Chunks:** ``encode_chunk`` frames ``TrajectoryChunk.to_payload()`` (kind ``chunk``; agent, policy version,
  learner generation, sender node and worker in the header meta, so a node routes a frame by ``peek_meta`` without
  decoding it). ``decode_chunk`` checks the frame (``colosseum.net.frame``) and the payload's fields and builds the
  chunk; :class:`ChunkValidator` then checks it against the agent's role on the learner: the slot rules
  (``validate_slot_structure``), every leaf of ``obs`` / ``global_state`` / ``actions`` / ``action_masks`` /
  ``teacher_action`` with shape ``[S, ...]`` and the dtype the worker's buffers use, the chunk length, finite
  ``behavior_logp`` (and ``behavior_unit_logp``) on ACT slots, finite rewards and zero rewards on BOOT / PAD, discrete
  actions (and teacher actions) inside the action space and finite box values, the recorded action legal under its
  mask, ``teacher_action`` / ``has_teacher`` both or neither (only for an agent with a scripted teacher), the agent
  id, and (with a template) the structure of ``initial_state``. Every failure is a :class:`ChunkError` with reason
  ``invalid`` and a ``detail``; the learner drops the chunk and counts it (spec 4.7), it never stops.
- **Weights:** ``encode_weights`` / ``decode_weights``: a numpy ``state_dict`` with :class:`WeightsMeta` (agent,
  kind, learner generation, policy version, ``teacher_active``, optional checkpoint id, architecture and roles).
  Never compressed (float32 does not compress).
- **Checkpoints:** ``encode_checkpoint`` / ``decode_checkpoint``: ``learner.make_checkpoint_payload()`` with its
  ``torch.save`` trainer-state bytes as one ``uint8`` leaf; the hub stores them as bytes and never unpickles them
  (a learner reads its own trainer state with ``torch.load(..., weights_only=True)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from colosseum.core.config import check_agent_id
from colosseum.core.errors import ConfigError
from colosseum.core.specs import ActionSpec, ObsSpec, split_unit_actions
from colosseum.core.types import SLOT_ACT, TrajectoryChunk, check_slot_arrays
from colosseum.envs.game import RoleSpec
from colosseum.net.frame import FrameError, decode_frame, encode_frame, peek_meta
from colosseum.networks.state import State

WEIGHTS_KINDS = ("latest", "snapshot", "fixed", "teacher", "init", "recovery")
CHECKPOINT_KINDS = ("periodic", "final", "recovery")
_FLAT = ("kind", "reward", "terminal", "reset_after", "behavior_logp")
_FLAT_DTYPES = {"kind": np.int8, "reward": np.float32, "terminal": np.bool_, "reset_after": np.bool_,
                "behavior_logp": np.float32}
_PAYLOAD_KEYS = {"initial_state", "obs", "global_state", "actions", "action_masks", *_FLAT, "behavior_unit_logp",
                 "teacher_action", "has_teacher"}


class ChunkError(ValueError):
    """A chunk the learner drops (spec 4.7): ``reason`` is the ``dropped_chunks`` key, ``detail`` says what."""

    def __init__(self, reason: Literal["invalid", "stale", "stale_generation"], detail: str = "") -> None:
        super().__init__(f"{reason} chunk: {detail}" if detail else f"{reason} chunk")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ChunkMeta:
    agent_id: str
    policy_version: int
    generation: int          # learner generation of the weights with policy_version (0 = unknown / SP4 path)
    node: int                # sender node number (-1 in local tests)
    worker: int


def encode_chunk(chunk: TrajectoryChunk, *, generation: int, node: int, worker: int,
                 compress: bool = False) -> bytes:
    """One ``chunk`` frame of ``chunk`` (its numpy payload; agent and version travel in the header meta)."""
    payload = chunk.to_payload()
    meta = {"agent_id": payload.pop("agent_id"), "policy_version": payload.pop("policy_version"),
            "generation": int(generation), "node": int(node), "worker": int(worker)}
    return encode_frame(payload, kind="chunk", meta=meta, compress=compress)


def _int_field(meta: Mapping[str, Any], key: str, low: int) -> int:
    value = meta.get(key)
    if type(value) is not int or value < low:
        raise ChunkError("invalid", f"chunk meta {key!r} must be an integer >= {low}, got {value!r}")
    return value


def chunk_meta(meta: Mapping[str, Any]) -> ChunkMeta:
    """The :class:`ChunkMeta` of a ``chunk`` frame's header meta (``peek_meta`` result); ChunkError if malformed."""
    if set(meta) != {"agent_id", "policy_version", "generation", "node", "worker"}:
        raise ChunkError("invalid", f"chunk meta keys {sorted(meta)}")
    agent_id = meta["agent_id"]
    try:
        check_agent_id(agent_id if isinstance(agent_id, str) else "")
    except ConfigError:
        raise ChunkError("invalid", f"chunk meta agent_id {agent_id!r}") from None
    return ChunkMeta(agent_id, _int_field(meta, "policy_version", 0), _int_field(meta, "generation", 0),
                     _int_field(meta, "node", -1), _int_field(meta, "worker", 0))


def _check_tree(tree: Any, name: str, num_slots: int) -> None:
    """A numpy array or a dict tree of them (str keys) with the slot dim first."""
    if isinstance(tree, np.ndarray):
        if tree.ndim == 0 or tree.shape[0] != num_slots:
            raise ChunkError("invalid", f"field {name!r} has a leaf of shape {tree.shape}, expected S={num_slots} "
                                        f"first")
        return
    if isinstance(tree, dict) and tree and all(isinstance(k, str) for k in tree):
        for key, value in tree.items():
            _check_tree(value, f"{name}/{key}", num_slots)
        return
    raise ChunkError("invalid", f"field {name!r} must be a numpy array or a dict tree of them, "
                                f"got {type(tree).__name__}")


def _check_state_nodes(state: Any, depth: int = 0) -> None:
    if state is None or isinstance(state, np.ndarray):
        return
    if depth < 64 and isinstance(state, tuple | list):
        for item in state:
            _check_state_nodes(item, depth + 1)
        return
    if depth < 64 and isinstance(state, dict):
        for item in state.values():
            _check_state_nodes(item, depth + 1)
        return
    raise ChunkError("invalid", f"initial_state holds a {type(state).__name__}")


def check_payload_fields(payload: Any) -> int:
    """Field checks of a chunk payload without ``agent_id`` / ``policy_version`` (presence, types, one slot dim);
    returns ``S``. ChunkError("invalid") otherwise."""
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        keys = sorted(payload, key=str) if isinstance(payload, dict) else type(payload).__name__
        raise ChunkError("invalid", f"payload fields {keys}")
    for name in _FLAT:
        value = payload[name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ChunkError("invalid", f"field {name!r} must be a 1-D array [S]")
    num_slots = payload["kind"].shape[0]
    if num_slots < 2:
        raise ChunkError("invalid", f"a chunk needs at least 2 slots, got {num_slots}")
    for name in _FLAT:
        if payload[name].shape[0] != num_slots:
            raise ChunkError("invalid", f"field {name!r} has shape {payload[name].shape}, expected [S={num_slots}]")
    for name in ("obs", "actions"):
        _check_tree(payload[name], name, num_slots)
    for name in ("global_state", "action_masks", "teacher_action"):
        if payload[name] is not None:
            _check_tree(payload[name], name, num_slots)
    for name in ("behavior_unit_logp", "has_teacher"):
        value = payload[name]
        if value is not None and (not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != num_slots):
            raise ChunkError("invalid", f"field {name!r} must be None or an array with S={num_slots} first")
    if (payload["teacher_action"] is None) != (payload["has_teacher"] is None):
        raise ChunkError("invalid", "fields 'teacher_action' and 'has_teacher' come together")
    _check_state_nodes(payload["initial_state"])
    return num_slots


def decode_chunk(buf: bytes | bytearray | memoryview, *, cap: int,
                 validator: ChunkValidator | None = None) -> tuple[TrajectoryChunk, ChunkMeta]:
    """The chunk of a ``chunk`` frame and its meta (CPU tensors). Checks the frame, the meta and the payload's fields
    (presence, types, one slot dim ``S``); with ``validator`` also every role check of spec 4.9 (the learner's
    intake path: frame -> decode -> check -> batch). Every problem is a ChunkError("invalid"); ``detail`` starts
    with ``frame:`` for frame errors."""
    try:
        payload, meta = decode_frame(buf, cap=cap, kind="chunk", copy=True)
    except FrameError as e:
        raise ChunkError("invalid", f"frame:{e.reason}: {e}") from None
    info = chunk_meta(meta)
    check_payload_fields(payload)
    if validator is not None:
        validator.check(payload, info.agent_id, info.policy_version)
    payload["agent_id"], payload["policy_version"] = info.agent_id, info.policy_version
    try:
        chunk = TrajectoryChunk.from_payload(payload)
    except (KeyError, TypeError, ValueError) as e:
        raise ChunkError("invalid", f"malformed payload: {type(e).__name__}: {e}") from None
    return chunk, info


# ---------------------------------------------------------------------------
# The learner's checks
# ---------------------------------------------------------------------------


def _template(tree: Any, path: str = "") -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """``{path: (shape without the leading dim, dtype)}`` of a numpy template allocated with leading ``(1,)`` (a
    bare array has the path ``""``)."""
    if isinstance(tree, np.ndarray):
        return {path: (tuple(tree.shape[1:]), tree.dtype)}
    out: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for key, value in tree.items():
        out.update(_template(value, f"{path}/{key}" if path else key))
    return out


def _leaves(tree: Any, path: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, np.ndarray):
        return {path: tree}
    out: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        out.update(_leaves(value, f"{path}/{key}" if path else key))
    return out


def _state_template(state: Any) -> Any:
    """Structure of a numpy state (batch 1): ('nd', shape, dtype) / ('seq', type, items) / ('dict', keys, items) /
    ('none',)."""
    if state is None:
        return ("none",)
    if isinstance(state, np.ndarray):
        return ("nd", tuple(state.shape), state.dtype)
    if isinstance(state, tuple | list):
        return ("seq", type(state), tuple(_state_template(s) for s in state))
    return ("dict", tuple(state), tuple(_state_template(v) for v in state.values()))


class ChunkValidator:
    """Spec 4.9 checks of a chunk on its agent's learner (see the module docstring), on the numpy payload.

    ``role_spec``: the agent's spaces (``agent_role_spec``); ``teacher``: the agent has a scripted kickstart teacher
    (its chunks carry DAgger labels); ``chunk_length``: the expected ``S`` (``rollout.chunk_length``; None = any);
    ``state_template``: the model's ``initial_state(1)`` as numpy (``state_to_numpy``), whose structure, shapes and
    dtypes every chunk's ``initial_state`` must have (None = not checked)."""

    def __init__(self, role_spec: RoleSpec, *, agent_id: str, teacher: bool, chunk_length: int | None = None,
                 state_template: State | None = None) -> None:
        self._agent_id = agent_id
        self._teacher = bool(teacher)
        self._chunk_length = chunk_length
        self._obs = _template(ObsSpec.from_space(role_spec.observation_space).allocate((1,)))
        gs = role_spec.global_state_space
        self._gs = None if gs is None else _template(ObsSpec.from_space(gs).allocate((1,)))
        self._action_spec = ActionSpec.from_space(role_spec.action_space)
        self._actions = _template(self._action_spec.allocate_actions((1,)))
        self._masks = _template(self._action_spec.full_mask((1,))) if self._action_spec.has_masks else None
        self._deciders = self._action_spec.num_deciders
        self._state = None if state_template is None else _state_template(state_template)
        self._flat = {name: np.dtype(dtype) for name, dtype in _FLAT_DTYPES.items()}

    def _fail(self, detail: str) -> ChunkError:
        return ChunkError("invalid", f"agent {self._agent_id!r}: {detail}")

    def _check_leaves(self, tree: Any, template: dict, name: str, num_slots: int) -> None:
        if isinstance(tree, np.ndarray) != ("" in template):
            raise self._fail(f"{name}: expected {'an array' if '' in template else 'a dict tree'}")
        leaves = _leaves(tree)
        if leaves.keys() != template.keys():
            raise self._fail(f"{name}: leaves {sorted(leaves)}, expected {sorted(template)}")
        for path, leaf in leaves.items():
            shape, dtype = template[path]
            if leaf.shape != (num_slots, *shape) or leaf.dtype != dtype:
                raise self._fail(f"{name}/{path or '<root>'}: {leaf.dtype} {leaf.shape}, expected {dtype} "
                                 f"{(num_slots, *shape)}")

    def _check_action_values(self, actions: Any, name: str) -> None:
        """Discrete values inside the space, box values finite (every slot: BOOT / PAD carry zeros)."""
        for group in self._action_spec.groups:
            value = actions if not self._action_spec.is_dict else ActionSpec._get(actions, group.path)
            where = f"{name}/{'/'.join(group.path) or '<root>'}"
            if group.kind == "discrete":
                bad = (value < 0) | (value >= group.nvec[0])
            elif group.kind == "multi_discrete":
                bad = (value < 0) | (value >= np.asarray(group.nvec))
            elif group.kind == "box":
                bad = ~np.isfinite(value)
            else:
                parts = split_unit_actions(group.units, value)
                for comp in group.units.components:
                    part = parts[comp.name]
                    bad = (part < 0) | (part >= comp.size) if comp.kind == "discrete" else ~np.isfinite(part)
                    if bad.any():
                        raise self._fail(f"{where} component {comp.name!r}: a value outside the action space")
                continue
            if bad.any():
                raise self._fail(f"{where}: a value outside the action space")

    def check(self, payload: Mapping[str, Any], agent_id: str, policy_version: int | None = None) -> None:
        """Every check (the slot rules too) on a numpy payload that passed ``check_payload_fields``."""
        if agent_id != self._agent_id:
            raise self._fail(f"a chunk of agent {agent_id!r}")
        num_slots = payload["kind"].shape[0]
        if self._chunk_length is not None and num_slots != self._chunk_length:
            raise self._fail(f"{num_slots} slots, expected rollout.chunk_length {self._chunk_length}")
        for name, dtype in self._flat.items():
            if payload[name].dtype != dtype:
                raise self._fail(f"{name}: {payload[name].dtype}, expected {dtype}")
        self._check_leaves(payload["obs"], self._obs, "obs", num_slots)
        if (payload["global_state"] is None) != (self._gs is None):
            raise self._fail("global_state present for a role without one" if self._gs is None
                             else "global_state missing")
        if self._gs is not None:
            self._check_leaves(payload["global_state"], self._gs, "global_state", num_slots)
        self._check_leaves(payload["actions"], self._actions, "actions", num_slots)
        if (payload["action_masks"] is None) != (self._masks is None):
            raise self._fail("action_masks present for an action space without masks" if self._masks is None
                             else "action_masks missing")
        if self._masks is not None:
            self._check_leaves(payload["action_masks"], self._masks, "action_masks", num_slots)
        unit = payload["behavior_unit_logp"]
        if (unit is None) != (self._deciders <= 1):
            raise self._fail(f"behavior_unit_logp {'missing' if unit is None else 'present'} for "
                             f"{self._deciders} decider(s)")
        if unit is not None and (unit.shape != (num_slots, self._deciders) or unit.dtype != np.float32):
            raise self._fail(f"behavior_unit_logp: {unit.dtype} {unit.shape}, expected float32 "
                             f"({num_slots}, {self._deciders})")
        if (payload["teacher_action"] is not None) != self._teacher:
            raise self._fail("DAgger labels on a chunk of an agent without a scripted teacher" if not self._teacher
                             else "DAgger labels missing")
        act = payload["kind"] == SLOT_ACT
        if not np.isfinite(payload["behavior_logp"][act]).all():
            raise self._fail("a non-finite behavior_logp on an ACT slot")
        if unit is not None and not np.isfinite(unit[act]).all():
            raise self._fail("a non-finite behavior_unit_logp on an ACT slot")
        reward = payload["reward"]
        if not np.isfinite(reward).all():
            raise self._fail("a non-finite reward")
        if (reward[~act] != 0).any():
            raise self._fail("a non-zero reward on a BOOT or PAD slot")
        self._check_action_values(payload["actions"], "actions")
        if self._teacher:
            self._check_leaves(payload["teacher_action"], self._actions, "teacher_action", num_slots)
            has = payload["has_teacher"]
            if has.dtype != np.bool_ or has.shape != (num_slots,):
                raise self._fail(f"has_teacher: {has.dtype} {has.shape}, expected bool ({num_slots},)")
            self._check_action_values(payload["teacher_action"], "teacher_action")
        if self._state is not None and _state_template(payload["initial_state"]) != self._state:
            raise self._fail("initial_state does not match the model's state structure, shapes and dtypes")
        try:
            check_slot_arrays(payload["kind"], payload["terminal"], payload["reset_after"], payload["has_teacher"],
                              agent_id=agent_id, policy_version=-1 if policy_version is None else policy_version)
        except ValueError as e:
            raise ChunkError("invalid", str(e)) from None

    def validate(self, chunk: TrajectoryChunk) -> None:
        """Every check on a chunk that is already built (e.g. from an SP4 payload on a queue)."""
        payload = chunk.to_payload()
        agent_id = payload.pop("agent_id")
        version = payload.pop("policy_version")
        check_payload_fields(payload)
        self.check(payload, agent_id, version)


# ---------------------------------------------------------------------------
# Weights and checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightsMeta:
    agent_id: str
    kind: Literal["latest", "snapshot", "fixed", "teacher", "init", "recovery"]
    generation: int
    policy_version: int
    teacher_active: bool = False
    checkpoint_id: str | None = None
    architecture: dict | None = None     # meta.json "networks" (or a frozen entry's) for fixed / teacher / init
    roles: tuple[str, ...] | None = None


def encode_weights(state_dict: Mapping[str, np.ndarray], meta: WeightsMeta) -> bytes:
    """One ``weights`` frame (never compressed). ``state_dict`` is numpy (``state_dict_to_numpy``)."""
    bad = [k for k, v in state_dict.items() if not isinstance(k, str) or not isinstance(v, np.ndarray)]
    if bad:
        raise TypeError(f"weights must be a dict of str -> numpy array, got {bad[0]!r}")
    if meta.kind not in WEIGHTS_KINDS:
        raise ValueError(f"weights kind must be one of {WEIGHTS_KINDS}, got {meta.kind!r}")
    data = asdict(meta)
    data["roles"] = None if meta.roles is None else list(meta.roles)
    return encode_frame(dict(state_dict), kind="weights", meta=data)


def _weights_meta(meta: Mapping[str, Any]) -> WeightsMeta:
    fields = {"agent_id", "kind", "generation", "policy_version", "teacher_active", "checkpoint_id", "architecture",
              "roles"}
    ok = (set(meta) == fields and isinstance(meta["agent_id"], str) and meta["kind"] in WEIGHTS_KINDS
          and type(meta["generation"]) is int and meta["generation"] >= 0
          and type(meta["policy_version"]) is int and meta["policy_version"] >= 0
          and isinstance(meta["teacher_active"], bool)
          and (meta["checkpoint_id"] is None or isinstance(meta["checkpoint_id"], str))
          and (meta["architecture"] is None or isinstance(meta["architecture"], dict))
          and (meta["roles"] is None or (isinstance(meta["roles"], list)
                                         and all(isinstance(r, str) for r in meta["roles"]))))
    if not ok:
        raise FrameError("weights_meta", f"malformed weights meta {dict(meta)!r}"[:300])
    roles = None if meta["roles"] is None else tuple(meta["roles"])
    return WeightsMeta(meta["agent_id"], meta["kind"], meta["generation"], meta["policy_version"],
                       meta["teacher_active"], meta["checkpoint_id"], meta["architecture"], roles)


def decode_weights(buf: bytes | bytearray | memoryview, *, cap: int) -> tuple[dict[str, np.ndarray], WeightsMeta]:
    """``(state_dict, meta)`` of a ``weights`` frame (views of ``buf``; FrameError if malformed)."""
    kind, _ = peek_meta(buf)
    if kind != "weights":
        raise FrameError("kind", f"expected a weights frame, got {kind!r}")
    tree, meta = decode_frame(buf, cap=cap)
    if not isinstance(tree, dict) or not all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in tree.items()):
        raise FrameError("weights", "a weights frame holds a dict of str -> array")
    return tree, _weights_meta(meta)


def encode_checkpoint(payload: Mapping[str, Any], *, kind: Literal["periodic", "final", "recovery"],
                      generation: int) -> bytes:
    """One ``checkpoint`` frame of ``learner.make_checkpoint_payload()`` (trainer-state bytes as a uint8 leaf)."""
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"checkpoint kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    trainer = payload.get("trainer_state_bytes")
    tree = {"model_state": dict(payload["model_state"]),
            "trainer_state": None if trainer is None else np.frombuffer(trainer, dtype=np.uint8)}
    meta = {"agent_id": str(payload["agent_id"]), "policy_version": int(payload["policy_version"]),
            "final": bool(payload.get("final", False)), "kind": kind, "generation": int(generation)}
    return encode_frame(tree, kind="checkpoint", meta=meta)


def decode_checkpoint(buf: bytes | bytearray | memoryview, *, cap: int) -> tuple[dict, dict]:
    """``(payload as make_checkpoint_payload, meta)`` of a ``checkpoint`` frame; FrameError if malformed."""
    kind, _ = peek_meta(buf)
    if kind != "checkpoint":
        raise FrameError("kind", f"expected a checkpoint frame, got {kind!r}")
    tree, meta = decode_frame(buf, cap=cap)
    ok = (isinstance(tree, dict) and set(tree) == {"model_state", "trainer_state"}
          and isinstance(tree["model_state"], dict)
          and all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in tree["model_state"].items())
          and (tree["trainer_state"] is None or (isinstance(tree["trainer_state"], np.ndarray)
                                                 and tree["trainer_state"].dtype == np.uint8
                                                 and tree["trainer_state"].ndim == 1))
          and set(meta) == {"agent_id", "policy_version", "final", "kind", "generation"}
          and isinstance(meta["agent_id"], str) and type(meta["policy_version"]) is int
          and meta["policy_version"] >= 0 and isinstance(meta["final"], bool) and meta["kind"] in CHECKPOINT_KINDS
          and type(meta["generation"]) is int and meta["generation"] >= 0)
    if not ok:
        raise FrameError("checkpoint", "malformed checkpoint frame")
    trainer = tree["trainer_state"]
    payload = {"agent_id": meta["agent_id"], "policy_version": meta["policy_version"],
               "model_state": tree["model_state"],
               "trainer_state_bytes": None if trainer is None else trainer.tobytes(), "final": meta["final"]}
    return payload, dict(meta)

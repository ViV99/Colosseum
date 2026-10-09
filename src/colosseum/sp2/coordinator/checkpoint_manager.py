"""Checkpoint storage: atomic per-agent checkpoints, FIFO pool, resume resolution.

Layout (``base_dir`` is ``<run_dir>/checkpoints``)::

    base_dir/<agent_id>/ckpt_v<policy_version>/
        model.pt            torch.save of the model state_dict (CPU tensors)
        trainer_state.pt    torch.save of BaseAlgorithm.state_dict() (optional)
        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras
                            (SP2: roles and role_signature of the agent; required by resume and eval)

A checkpoint's path is always ``base_dir/agent_id/checkpoint_id``; both ids must be
safe path components (``core.config.check_path_component``), and agent ids may not
contain ``.`` (``core.config.check_agent_id``). A ``path`` stored in
``meta.json`` (older layouts, copied runs) is never used (R6-06).

Writes go to ``.tmp-<id>-<rand>/`` and are moved into place with ``os.replace``.
Replacing an existing id first moves it aside to ``.tmp-old-<id>-<rand>/``. Eviction
first renames the victim to ``.tmp-evict-<id>-<rand>/`` and then deletes it, so a crash
or a partial delete never leaves a half-deleted ``ckpt_v*`` dir (which would make a
strict run-dir resume fail). On scan, a ``.tmp-old-*`` dir is restored when its id is
missing (a crash between the two moves) and deleted otherwise; any other ``.tmp-*`` dir
(write or eviction leftover) is deleted only when older than ``STALE_TMP_AGE_SEC``.
(A writer of another process caught exactly between its two moves would see its save
fail; the previous checkpoint stays intact.)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import check_agent_id, check_path_component
from colosseum.sp2.core.types import state_dict_from_numpy, state_dict_to_numpy

logger = logging.getLogger(__name__)

# The single numpy <-> torch state_dict conversion lives in colosseum.core.types.
numpy_state_to_torch = state_dict_from_numpy
torch_state_to_numpy = state_dict_to_numpy

MODEL_FILE = "model.pt"
TRAINER_FILE = "trainer_state.pt"
META_FILE = "meta.json"
_CKPT_RE = re.compile(r"ckpt_v(\d+)")
_TMP_PREFIX = ".tmp-"
_OLD_TMP_RE = re.compile(r"\.tmp-old-(ckpt_v\d+)-[0-9a-f]+")
# A ``.tmp-<id>-*`` write dir older than this is a leftover of a crashed writer and is
# deleted on scan. Age, not the writer's pid, decides: in distributed mode learners on
# other machines may share the checkpoint dir, and their pids mean nothing here. A
# real save takes seconds, so a younger tmp dir may belong to a live writer.
STALE_TMP_AGE_SEC = 3600.0


@dataclass
class CheckpointInfo:
    """One complete checkpoint on disk."""

    checkpoint_id: str
    agent_id: str
    policy_version: int
    path: Path
    timestamp: float
    meta: dict = field(default_factory=dict)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_meta(ckpt_dir: Path) -> dict:
    """Read and validate ``ckpt_dir/meta.json``; raise ValueError naming the problem.

    Required: a JSON object with an integer ``policy_version`` (equal to the dir's
    ``ckpt_v<N>`` when the dir is named so). Optional: a numeric ``timestamp`` and an
    integer or null ``env_steps``.
    """
    try:
        meta = json.loads((ckpt_dir / META_FILE).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"unreadable {META_FILE} ({type(e).__name__}: {e})") from e
    if not isinstance(meta, dict):
        raise ValueError(f"{META_FILE} is not a JSON object")
    version = meta.get("policy_version")
    if not _is_int(version):
        raise ValueError(f"{META_FILE}: policy_version is missing or not an integer ({version!r})")
    match = _CKPT_RE.fullmatch(ckpt_dir.name)
    if match is not None and int(match.group(1)) != version:
        raise ValueError(f"{META_FILE}: policy_version {version} does not match {ckpt_dir.name}")
    if not _is_number(meta.get("timestamp", 0.0)):
        raise ValueError(f"{META_FILE}: timestamp is not a number ({meta.get('timestamp')!r})")
    env_steps = meta.get("env_steps")
    if env_steps is not None and not _is_int(env_steps):
        raise ValueError(f"{META_FILE}: env_steps is not an integer or null ({env_steps!r})")
    roles = meta.get("roles")
    if roles is not None and not (isinstance(roles, list) and roles and all(isinstance(r, str) for r in roles)):
        raise ValueError(f"{META_FILE}: roles is not a non-empty list of strings ({roles!r})")
    signature = meta.get("role_signature")
    if signature is not None and not isinstance(signature, str):
        raise ValueError(f"{META_FILE}: role_signature is not a string ({signature!r})")
    return meta


def _read_agent_dir(agent_dir: Path, strict: bool = False) -> list[CheckpointInfo]:
    """Checkpoints of one agent dir (``ckpt_v<N>/`` with model.pt + valid meta.json), by version.

    A malformed checkpoint (missing file, unreadable or invalid meta.json) is skipped
    with a warning, or raises ConfigError naming its path when ``strict`` (explicit
    resume: never silently fall back to an older checkpoint or to a fresh start).
    """
    infos: list[CheckpointInfo] = []
    if not agent_dir.is_dir():
        return infos
    for d in agent_dir.iterdir():
        match = _CKPT_RE.fullmatch(d.name)
        if match is None or not d.is_dir():
            continue
        try:
            missing = [f for f in (MODEL_FILE, META_FILE) if not (d / f).is_file()]
            if missing:
                raise ValueError(f"missing {', '.join(missing)}")
            meta = _parse_meta(d)
        except ValueError as e:
            if strict:
                raise ConfigError(f"Malformed checkpoint {d}: {e}") from e
            logger.warning(f"Skipping malformed checkpoint {d}: {e}")
            continue
        infos.append(CheckpointInfo(
            checkpoint_id=d.name,
            agent_id=agent_dir.name,
            policy_version=int(match.group(1)),
            path=d,
            timestamp=float(meta.get("timestamp", 0.0)),
            meta=meta,
        ))
    infos.sort(key=lambda c: c.policy_version)
    return infos


def _clean_tmp_dirs(agent_dir: Path) -> None:
    """Recover or delete leftovers of interrupted saves (see the module docstring)."""
    now = time.time()
    for d in agent_dir.glob(f"{_TMP_PREFIX}*"):
        old = _OLD_TMP_RE.fullmatch(d.name)
        if old is not None:
            target = agent_dir / old.group(1)
            if target.exists():
                shutil.rmtree(d, ignore_errors=True)
                continue
            try:
                os.replace(d, target)
                logger.warning(f"Restored checkpoint {target} from an interrupted replace")
            except OSError as e:
                logger.warning(f"Could not restore {d} to {target}: {e}")
            continue
        try:
            age = now - d.stat().st_mtime
        except FileNotFoundError:
            continue
        if age > STALE_TMP_AGE_SEC:
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Removed stale checkpoint write dir {d}")


class CheckpointManager:
    """Saves checkpoints atomically and keeps a FIFO pool of ``pool_size`` per agent."""

    def __init__(self, base_dir: str | Path, pool_size: int = 20) -> None:
        self._base_dir = Path(base_dir)
        self._pool_size = pool_size
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def agents(self) -> list[str]:
        return list(self._index.keys())

    def _agent_dir(self, agent_id: str) -> Path:
        return self._base_dir / check_agent_id(agent_id)

    def _ckpt_dir(self, agent_id: str, checkpoint_id: str) -> Path:
        return self._agent_dir(agent_id) / check_path_component(checkpoint_id, "checkpoint id")

    def _scan(self) -> None:
        for agent_dir in sorted(self._base_dir.iterdir()):
            if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                continue
            _clean_tmp_dirs(agent_dir)
            infos = _read_agent_dir(agent_dir)
            if infos:
                self._index[agent_dir.name] = infos

    def save(
        self,
        agent_id: str,
        policy_version: int,
        model_state: dict[str, np.ndarray],
        trainer_state: dict | bytes | None = None,
        meta_extra: dict | None = None,
    ) -> str:
        """Write ``ckpt_v<policy_version>`` atomically and evict the oldest beyond ``pool_size``.

        The files are written into ``.tmp-<id>-<rand>/`` and the directory is moved into
        place with ``os.replace``. An existing checkpoint with the same id is replaced.
        """
        checkpoint_id = f"ckpt_v{int(policy_version)}"
        agent_dir = self._agent_dir(agent_id)
        agent_dir.mkdir(parents=True, exist_ok=True)
        final_dir = agent_dir / checkpoint_id
        tmp_dir = agent_dir / f"{_TMP_PREFIX}{checkpoint_id}-{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir()
        timestamp = time.time()
        meta = {
            "agent_id": agent_id,
            "checkpoint_id": checkpoint_id,
            "policy_version": int(policy_version),
            "timestamp": timestamp,
            **(meta_extra or {}),
        }
        try:
            torch.save(numpy_state_to_torch(model_state), tmp_dir / MODEL_FILE)
            if isinstance(trainer_state, (bytes, bytearray)):
                (tmp_dir / TRAINER_FILE).write_bytes(bytes(trainer_state))
            elif trainer_state is not None:
                torch.save(trainer_state, tmp_dir / TRAINER_FILE)
            (tmp_dir / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
            if final_dir.exists():
                old_dir = agent_dir / f"{_TMP_PREFIX}old-{checkpoint_id}-{uuid.uuid4().hex[:8]}"
                os.replace(final_dir, old_dir)
                os.replace(tmp_dir, final_dir)
                shutil.rmtree(old_dir, ignore_errors=True)
            else:
                os.replace(tmp_dir, final_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        entries = [c for c in self._index.get(agent_id, []) if c.checkpoint_id != checkpoint_id]
        entries.append(CheckpointInfo(checkpoint_id, agent_id, int(policy_version), final_dir, timestamp, meta))
        entries.sort(key=lambda c: c.policy_version)
        while len(entries) > self._pool_size:
            victim = next(c for c in entries if c.checkpoint_id != checkpoint_id)
            entries.remove(victim)
            self._evict(agent_id, victim.checkpoint_id)
            logger.debug(f"Evicted checkpoint {victim.checkpoint_id} of {agent_id}")
        self._index[agent_id] = entries
        logger.info(f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(entries)}/{self._pool_size})")
        return checkpoint_id

    def _evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Rename the checkpoint out of the ``ckpt_v*`` namespace, then delete it."""
        victim = self._ckpt_dir(agent_id, checkpoint_id)
        doomed = self._agent_dir(agent_id) / f"{_TMP_PREFIX}evict-{checkpoint_id}-{uuid.uuid4().hex[:8]}"
        try:
            os.replace(victim, doomed)
        except FileNotFoundError:
            return  # already gone (e.g. removed by hand)
        except OSError as e:  # like the plain rmtree before: an eviction failure never fails the save
            logger.warning(f"Could not evict {victim}: {e}")
            return
        shutil.rmtree(doomed, ignore_errors=True)

    def load_model(self, agent_id: str, checkpoint_id: str) -> dict[str, np.ndarray]:
        path = self._ckpt_dir(agent_id, checkpoint_id) / MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch_state_to_numpy(torch.load(path, map_location="cpu", weights_only=True))

    def load_trainer_state(self, agent_id: str, checkpoint_id: str) -> dict | None:
        path = self._ckpt_dir(agent_id, checkpoint_id) / TRAINER_FILE
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def list_checkpoints(self, agent_id: str) -> list[CheckpointInfo]:
        return list(self._index.get(agent_id, []))

    def latest(self, agent_id: str) -> CheckpointInfo | None:
        entries = self._index.get(agent_id, [])
        return entries[-1] if entries else None


def read_weights_file(path: str | Path) -> dict[str, np.ndarray]:
    """Numpy state_dict from a ``torch.save``-d dict of tensors (``model.pt`` or a ``.pt`` file).

    Any failure (missing or unreadable file, unpickling error, not a dict of tensors)
    raises ValueError; callers wrap it in a ConfigError that names their own context.
    """
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - any read/unpickling failure means unusable weights
        raise ValueError(f"cannot load weights from {path.name} ({type(e).__name__}: {e})") from e
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path.name} is not a state_dict of tensors")
    return torch_state_to_numpy(state)


def _read_checkpoint(ckpt_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """``(meta, numpy model state_dict)`` of one checkpoint dir; any problem raises (no
    ConfigError wrapping here: callers name their own context)."""
    return _parse_meta(ckpt_dir), read_weights_file(ckpt_dir / MODEL_FILE)


def load_checkpoint_dir(ckpt_dir: str | Path) -> dict[str, Any]:
    """Read one checkpoint dir (e.g. ``<run>/checkpoints/<agent>/ckpt_v<N>``) without touching it.

    Returns ``{"model_state": numpy state_dict, "meta": validated meta.json, "roles": list | None,
    "role_signature": str | None}`` (the last two from ``meta.json``; None when absent). Unlike
    ``CheckpointManager`` (whose scan cleans up and restores ``.tmp-*`` dirs), nothing is
    written, so it is safe on a live run. A missing dir or file, an invalid ``meta.json``
    or an unreadable ``model.pt`` raises ConfigError naming the dir.
    """
    path = Path(ckpt_dir)
    try:
        if not path.is_dir():
            raise ValueError("not a directory")
        meta, model_state = _read_checkpoint(path)
    except ValueError as e:
        raise ConfigError(f"Malformed checkpoint {path}: {e}") from e
    return {"model_state": model_state, "meta": meta,
            "roles": meta.get("roles"), "role_signature": meta.get("role_signature")}


def _load_checkpoint_dir(ckpt_dir: Path, resume_from: str) -> dict[str, Any]:
    """Resume state from one checkpoint dir; any unreadable part (also a missing
    ``meta.json``, the only source of the version) raises ConfigError."""
    trainer_path = ckpt_dir / TRAINER_FILE
    try:
        meta, model_state = _read_checkpoint(ckpt_dir)
        version = int(meta["policy_version"])
        env_steps = int(meta.get("env_steps") or 0)  # null: unknown (distributed learners)
        trainer_state = trainer_path.read_bytes() if trainer_path.is_file() else None
    except Exception as e:  # noqa: BLE001 - any read/parse/unpickling failure means a bad resume source
        raise ConfigError(
            f"training.resume_from={resume_from!r}: cannot read checkpoint {ckpt_dir} "
            f"({type(e).__name__}: {e})"
        ) from e
    return {
        "model_state": model_state,
        "trainer_state": trainer_state,
        "policy_version": version,
        "env_steps": env_steps,
        "source": str(ckpt_dir),
        "roles": meta.get("roles"),
        "role_signature": meta.get("role_signature"),
    }


def check_role_signature(state: dict, expected: str, context: str) -> None:
    """ConfigError naming ``state["source"]`` unless the checkpoint's role signature is ``expected``.

    ``state`` is a ``resolve_resume`` result (or anything with ``source`` and ``role_signature``).
    A checkpoint without a signature (written before SP2) is rejected too.
    """
    signature = state.get("role_signature")
    if signature is None:
        raise ConfigError(
            f"{context}: checkpoint {state['source']} has no role_signature in its meta.json "
            f"(written before SP2); SP1 checkpoints cannot be resumed: drop training.resume_from to start fresh"
        )
    if signature != expected:
        raise ConfigError(
            f"{context}: checkpoint {state['source']} has role signature {signature!r}, but the agent's "
            f"roles in this game have {expected!r}: the observation/action/global-state spaces differ; "
            f"resume from a checkpoint of an agent with the same roles, or drop training.resume_from"
        )


RESUME_RUN_DIR = "run_dir"
RESUME_CHECKPOINT_DIR = "checkpoint_dir"
RESUME_PT_FILE = "pt_file"


def classify_resume_source(resume_from: str | Path) -> str:
    """Kind of a ``training.resume_from`` source, from the file system only (nothing is loaded).

    Returns ``RESUME_RUN_DIR`` (a dir containing ``checkpoints/``), ``RESUME_CHECKPOINT_DIR``
    (a dir containing ``model.pt``) or ``RESUME_PT_FILE`` (a ``.pt`` file); anything else
    raises ConfigError.
    """
    path = Path(resume_from)
    if path.is_dir() and (path / "checkpoints").is_dir():
        return RESUME_RUN_DIR
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return RESUME_CHECKPOINT_DIR
    if path.is_file() and path.suffix == ".pt":
        return RESUME_PT_FILE
    raise ConfigError(
        f"training.resume_from={str(resume_from)!r}: expected a checkpoint dir (containing {MODEL_FILE}), "
        f"a run dir (containing checkpoints/), or a .pt file"
    )


def resolve_resume(resume_from: str, agent_id: str, expected_signature: str | None = None) -> dict | None:
    """Resolve ``training.resume_from`` for one agent.

    Accepted forms (see ``classify_resume_source``):
    - a checkpoint dir (contains ``model.pt``; ``meta.json`` is required): its weights,
      trainer state and version;
    - a previous run dir (contains ``checkpoints/``): the agent's latest checkpoint
      there, or ``None`` (with a warning) if the agent has none;
    - a ``.pt`` file (e.g. the output of ``colosseum bc``): weights only, version 0.

    With ``expected_signature`` (``core.roles.role_signature`` of the agent's roles), a checkpoint
    source must carry exactly that ``role_signature`` in its ``meta.json`` (``check_role_signature``);
    a ``.pt`` file has no signature and is checked only by ``check_model_state``.

    The result holds only numpy arrays, bytes and primitives (plus ``roles`` and
    ``role_signature``, None for a ``.pt``), so it can be passed to a learner process.
    """
    check_agent_id(agent_id)
    path = Path(resume_from)
    kind = classify_resume_source(path)
    if kind == RESUME_RUN_DIR:
        try:
            infos = _read_agent_dir(path / "checkpoints" / agent_id, strict=True)
        except ConfigError as e:
            raise ConfigError(f"training.resume_from={resume_from!r}: {e}") from e
        if not infos:
            logger.warning(f"resume_from={resume_from}: no checkpoints for agent '{agent_id}'; starting fresh")
            return None
        state = _load_checkpoint_dir(infos[-1].path, resume_from)
    elif kind == RESUME_CHECKPOINT_DIR:
        state = _load_checkpoint_dir(path, resume_from)
    else:
        try:
            model_state = read_weights_file(path)
        except ValueError as e:
            raise ConfigError(f"training.resume_from={resume_from!r}: {e}") from e
        return {"model_state": model_state, "trainer_state": None, "policy_version": 0, "env_steps": 0,
                "source": str(path), "roles": None, "role_signature": None}
    if expected_signature is not None:
        check_role_signature(state, expected_signature, f"training.resume_from={resume_from!r} [{agent_id}]")
    return state


def check_model_state(model: torch.nn.Module, model_state: dict[str, np.ndarray], source: str) -> None:
    """Raise ConfigError if ``model_state`` cannot be loaded into ``model``."""
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(np.asarray(v).shape) for k, v in model_state.items()}
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    if not (missing or unexpected or mismatched):
        return
    lines = [f"Weights from {source} do not match the agent's architecture:"]
    if missing:
        lines.append(f"  missing keys: {missing[:10]}")
    if unexpected:
        lines.append(f"  unexpected keys: {unexpected[:10]}")
    for k in mismatched[:10]:
        lines.append(f"  shape mismatch {k}: checkpoint {got[k]} vs model {expected[k]}")
    raise ConfigError("\n".join(lines))

"""Checkpoint storage: atomic per-agent checkpoints, FIFO pool, resume resolution.

Layout (``base_dir`` is ``<run_dir>/checkpoints``)::

    base_dir/<agent_id>/ckpt_v<policy_version>/
        model.pt            torch.save of the model state_dict (CPU tensors)
        trainer_state.pt    torch.save of BaseAlgorithm.state_dict() (optional)
        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras

A checkpoint's path is always ``base_dir/agent_id/checkpoint_id``; both ids must be
safe path components (``core.config.check_path_component``). A ``path`` stored in
``meta.json`` (older layouts, copied runs) is never used (R6-06).

Writes go to ``.tmp-<id>-<rand>/`` and are moved into place with ``os.replace``.
Replacing an existing id first moves it aside to ``.tmp-old-<id>-<rand>/``. On scan,
a ``.tmp-old-*`` dir is restored when its id is missing (a crash between the two
moves) and deleted otherwise; a ``.tmp-<id>-*`` dir is deleted only when older than
``STALE_TMP_AGE_SEC``. (A writer of another process caught exactly between its two
moves would see its save fail; the previous checkpoint stays intact.)
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

from colosseum.core.config import check_path_component
from colosseum.core.errors import ConfigError
from colosseum.core.types import state_dict_from_numpy, state_dict_to_numpy

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


def _read_agent_dir(agent_dir: Path) -> list[CheckpointInfo]:
    """Complete checkpoints of one agent dir (model.pt + meta.json), sorted by version."""
    infos: list[CheckpointInfo] = []
    if not agent_dir.is_dir():
        return infos
    for d in agent_dir.iterdir():
        match = _CKPT_RE.fullmatch(d.name)
        if match is None or not d.is_dir():
            continue
        meta_path = d / META_FILE
        if not (d / MODEL_FILE).is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping checkpoint {d}: unreadable {META_FILE} ({e})")
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
        return self._base_dir / check_path_component(agent_id, "agent id")

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
            shutil.rmtree(self._ckpt_dir(agent_id, victim.checkpoint_id), ignore_errors=True)
            logger.debug(f"Evicted checkpoint {victim.checkpoint_id} of {agent_id}")
        self._index[agent_id] = entries
        logger.info(f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(entries)}/{self._pool_size})")
        return checkpoint_id

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


def _load_checkpoint_dir(ckpt_dir: Path, resume_from: str) -> dict[str, Any]:
    """Resume state from one checkpoint dir; any unreadable part raises ConfigError."""
    meta_path = ckpt_dir / META_FILE
    trainer_path = ckpt_dir / TRAINER_FILE
    match = _CKPT_RE.fullmatch(ckpt_dir.name)
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        if not isinstance(meta, dict):
            raise ValueError(f"{META_FILE} is not a JSON object")
        version = int(meta.get("policy_version", match.group(1) if match else 0))
        env_steps = int(meta.get("env_steps") or 0)  # null: unknown (distributed learners)
        state = torch.load(ckpt_dir / MODEL_FILE, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError(f"{MODEL_FILE} is not a state_dict of tensors")
        trainer_state = trainer_path.read_bytes() if trainer_path.is_file() else None
    except Exception as e:  # noqa: BLE001 - any read/parse/unpickling failure means a bad resume source
        raise ConfigError(
            f"training.resume_from={resume_from!r}: cannot read checkpoint {ckpt_dir} "
            f"({type(e).__name__}: {e})"
        ) from e
    return {
        "model_state": torch_state_to_numpy(state),
        "trainer_state": trainer_state,
        "policy_version": version,
        "env_steps": env_steps,
        "source": str(ckpt_dir),
    }


def resolve_resume(resume_from: str, agent_id: str) -> dict | None:
    """Resolve ``training.resume_from`` for one agent.

    Accepted forms:
    - a checkpoint dir (contains ``model.pt``): its weights, trainer state and version;
    - a previous run dir (contains ``checkpoints/``): the agent's latest checkpoint
      there, or ``None`` (with a warning) if the agent has none;
    - a ``.pt`` file (e.g. the output of ``colosseum bc``): weights only, version 0.

    The result holds only numpy arrays, bytes and primitives, so it can be passed
    to a learner process.
    """
    check_path_component(agent_id, "agent id")
    path = Path(resume_from)
    if path.is_dir() and (path / "checkpoints").is_dir():
        infos = _read_agent_dir(path / "checkpoints" / agent_id)
        if not infos:
            logger.warning(f"resume_from={resume_from}: no checkpoints for agent '{agent_id}'; starting fresh")
            return None
        return _load_checkpoint_dir(infos[-1].path, resume_from)
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return _load_checkpoint_dir(path, resume_from)
    if path.is_file() and path.suffix == ".pt":
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 - any unpickling failure means a bad resume source
            raise ConfigError(f"training.resume_from={resume_from!r}: cannot load weights ({e})") from e
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ConfigError(f"training.resume_from={resume_from!r}: expected a state_dict of tensors")
        return {"model_state": torch_state_to_numpy(state), "trainer_state": None,
                "policy_version": 0, "env_steps": 0, "source": str(path)}
    raise ConfigError(
        f"training.resume_from={resume_from!r}: expected a checkpoint dir (containing {MODEL_FILE}), "
        f"a run dir (containing checkpoints/), or a .pt file"
    )


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

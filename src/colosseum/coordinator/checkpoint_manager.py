"""Checkpoint storage: atomic per-agent snapshots, retention (``keep_last`` / ``keep_every`` / final / pinned /
top-k / ``max_snapshots``), the run-dir pool import, resume resolution.

Layout (``base_dir`` is ``<run_dir>/checkpoints``)::

    base_dir/<agent_id>/ckpt_v<policy_version>/
        model.pt            torch.save of the model state_dict (CPU tensors)
        trainer_state.pt    torch.save of BaseAlgorithm.state_dict() (optional)
        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras
                            (SP2: roles and role_signature of the agent; required by resume and eval;
                            SP4: pinned, and run_id when the manager has one)

A checkpoint's path is always ``base_dir/agent_id/checkpoint_id``; both ids must be
safe path components (``core.config.check_path_component``), and agent ids may not
contain ``.`` (``core.config.check_agent_id``). A ``path`` stored in
``meta.json`` (older layouts, copied runs) is never used (R6-06).

Retention (SP4 spec block 4; the decision is ``colosseum.arena.retention.retain``), after every save, import
and ``reapply_retention``: the newest ``keep_last``, the final snapshot of the current run (``run_id``; with no
run id every final, as in SP3), pinned snapshots (``meta.json`` ``pinned``, re-read from disk before every
decision, so ``colosseum pin`` works while a run trains), the snapshot just saved, every ``keep_every``-th
version, and with ``keep_top_k > 0`` the top-k by the scores provider's lower bound plus new snapshots with
fewer than ``top_k_min_games`` rated games; then the ``max_snapshots`` cap. Evicted ids leave the index first,
then ``on_evict(agent_id, checkpoint_id)`` is called for each (a raising callback never leaves deleted dirs in
the index). Only the newest ``keep_last`` keep ``trainer_state.pt``. Every new ``meta.json`` gets ``pinned``
(false) and, with a run id, ``run_id``.

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.arena.retention import RetentionPolicy, SnapshotInfo, SnapshotScore, retain
from colosseum.core.config import check_agent_id, check_path_component
from colosseum.core.errors import ConfigError
from colosseum.core.types import state_dict_from_numpy, state_dict_to_numpy
from colosseum.utils.fs import write_text_atomic

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


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``; copy when links are impossible (another file system, no support)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


_link_or_copy = link_or_copy   # the SP3 name


def read_snapshots(agent_dir: str | Path) -> list[CheckpointInfo]:
    """Valid snapshots of one agent dir, by version, read without touching it (no tmp cleanup, malformed ones
    skipped with a warning). For read-only callers on a live run (``colosseum pin``, the tournament)."""
    return _read_agent_dir(Path(agent_dir))


def set_pinned(ckpt_dir: str | Path, pinned: bool) -> dict:
    """Set ``pinned`` in ``ckpt_dir/meta.json`` by an atomic rewrite (temp file + ``os.replace``: a hard-linked
    copy of the file in another run dir keeps its old content); returns the new meta. ConfigError for a
    missing dir or an invalid ``meta.json``."""
    path = Path(ckpt_dir)
    meta = read_checkpoint_meta(path)
    meta["pinned"] = bool(pinned)
    write_text_atomic(path / META_FILE, json.dumps(meta, indent=2, sort_keys=True))
    return meta


SNAPSHOT_REF_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_-]*)@v(\d+)")


def parse_snapshot_ref(text: str) -> tuple[str, int]:
    """``"main@v1200"`` -> ``("main", 1200)``: a snapshot reference of the CLI (spec §8: ``agent@v<N>`` in the CLI
    only; configs name agents). ConfigError for anything else."""
    match = SNAPSHOT_REF_RE.fullmatch(text)
    if match is None:
        raise ConfigError(f"{text!r} is not a snapshot reference; write agent@v<version>, e.g. main@v1200")
    return match.group(1), int(match.group(2))


def _run_checkpoints(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    ckpts = root / "checkpoints"
    if not ckpts.is_dir():
        raise ConfigError(f"{root} is not a run dir (no checkpoints/ inside); pass runs/<name>")
    return ckpts


def list_snapshots(run_dir: str | Path) -> dict[str, list[CheckpointInfo]]:
    """Stored snapshots of every agent of a run dir, agents in name order, snapshots by version. Read-only: no
    ``CheckpointManager`` scan (it would clean the tmp dirs of a live writer); malformed snapshots are skipped
    with a warning (``read_snapshots``); agents without valid snapshots are left out."""
    ckpts = _run_checkpoints(run_dir)
    out: dict[str, list[CheckpointInfo]] = {}
    for agent_dir in sorted(p for p in ckpts.iterdir() if p.is_dir() and not p.name.startswith(".")):
        infos = read_snapshots(agent_dir)
        if infos:
            out[agent_dir.name] = infos
    return out


def pin_snapshot(run_dir: str | Path, agent_id: str, version: int, *, pinned: bool = True) -> Path:
    """Pin (unpin) snapshot ``ckpt_v<version>`` of ``agent_id`` in ``run_dir`` by ``set_pinned``; idempotent and
    safe while the run trains (its retention re-reads pins before every decision). Returns the snapshot dir.
    ConfigError (not a run dir, an unknown agent or snapshot) naming the stored agents or snapshots."""
    snapshots = list_snapshots(run_dir)
    if agent_id not in snapshots:
        raise ConfigError(f"{run_dir}: no snapshots of agent {agent_id!r}; agents with snapshots: {sorted(snapshots)}")
    by_id = {c.checkpoint_id: c for c in snapshots[agent_id]}
    checkpoint_id = f"ckpt_v{int(version)}"
    if checkpoint_id not in by_id:
        raise ConfigError(f"{run_dir}: agent {agent_id!r} has no stored snapshot {checkpoint_id} (newest stored: "
                          f"{list(by_id)[-10:]}); see colosseum pin {run_dir} --list")
    set_pinned(by_id[checkpoint_id].path, pinned)
    return by_id[checkpoint_id].path


class CheckpointManager:
    """Saves snapshots atomically and applies the retention rules per agent (module docstring)."""

    def __init__(self, base_dir: str | Path, keep_last: int = 20, keep_every: int = 0, interval: int = 1,
                 on_evict: Callable[[str, str], None] | None = None, *, run_id: str | None = None,
                 keep_top_k: int = 0, top_k_min_games: int = 200, max_snapshots: int | None = None,
                 scores_fn: Callable[[str], Mapping[str, SnapshotScore] | None] | None = None) -> None:
        """``run_id``: the current run (only its final snapshot is protected; None keeps the SP3 rule: every
        final); ``scores_fn(agent_id)``: ``{checkpoint_id: SnapshotScore}`` of the agent's rated snapshots, or
        None while there are no ratings yet (``arena.retention``: unrated snapshots are new, the cap waits), read
        only with ``keep_top_k > 0``."""
        if keep_last < 1 or keep_every < 0 or interval < 1 or keep_top_k < 0 or top_k_min_games < 0 \
                or (max_snapshots is not None and max_snapshots < keep_last):
            raise ValueError(f"CheckpointManager: need keep_last >= 1, keep_every >= 0, interval >= 1, "
                             f"keep_top_k >= 0, top_k_min_games >= 0 and max_snapshots >= keep_last; got "
                             f"{keep_last}, {keep_every}, {interval}, {keep_top_k}, {top_k_min_games}, "
                             f"{max_snapshots}")
        self._base_dir = Path(base_dir)
        self._keep_last = int(keep_last)
        self._keep_every = int(keep_every)
        self._interval = int(interval)
        self._on_evict = on_evict
        self._run_id = run_id
        self._keep_top_k = int(keep_top_k)
        self._top_k_min_games = int(top_k_min_games)
        self._max_snapshots = max_snapshots
        self._scores_fn = scores_fn
        # (agent, decided without ratings) whose cap overflow was logged: once per agent and kind, so a WARNING
        # against the raised pre-refit cap never hides a later one against max_snapshots itself
        self._warned: set[tuple[str, bool]] = set()
        self._noted: set[str] = set()        # agents whose raised pre-refit cap was announced (INFO once)
        self._scores_failed = False          # the first failing scores_fn call is logged with its traceback
        self._reasons: dict[str, dict[str, str]] = {}   # per agent: the last decision's reasons (retain)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def run_id(self) -> str | None:
        """The current run's id (``meta.json`` ``run_id`` of new snapshots), or None (the SP3 final rule)."""
        return self._run_id

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
        """Write ``ckpt_v<policy_version>`` atomically, then apply the retention rules (this snapshot is always kept).

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
        previous = next((c for c in self._index.get(agent_id, []) if c.checkpoint_id == checkpoint_id), None)
        if previous is not None:
            self._refresh_pins([previous])     # a replaced snapshot keeps its pin
        meta.setdefault("pinned", bool(previous is not None and previous.meta.get("pinned") is True))
        if self._run_id is not None:
            meta.setdefault("run_id", self._run_id)
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
        self._index[agent_id] = entries          # indexed before retention: the dir is in place whatever happens
        self._index[agent_id], evicted = self._retain(agent_id, entries, protect=checkpoint_id)
        logger.info(f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(self._index[agent_id])})")
        self._notify_evicted(agent_id, evicted)
        return checkpoint_id

    def _policy(self) -> RetentionPolicy:
        return RetentionPolicy(keep_last=self._keep_last, keep_every_period=self._interval * self._keep_every,
                               keep_top_k=self._keep_top_k, top_k_min_games=self._top_k_min_games,
                               max_snapshots=self._max_snapshots, current_run_id=self._run_id)

    @staticmethod
    def _refresh_pins(entries: list[CheckpointInfo]) -> None:
        """Re-read ``pinned`` from every entry's ``meta.json`` (another process may have pinned it); an unreadable
        file keeps the indexed value."""
        for c in entries:
            try:
                disk = json.loads((c.path / META_FILE).read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(disk, dict):
                c.meta["pinned"] = disk.get("pinned") is True

    def _scores(self, agent_id: str) -> Mapping[str, SnapshotScore] | None:
        """The provider's scores (only with ``keep_top_k > 0``; ``{}`` without a provider); None = no ratings yet
        (``arena.retention``). A raising provider is logged (ERROR, once per manager) and read as "no ratings yet":
        retention never fails a save over the ratings."""
        if self._scores_fn is None or self._keep_top_k == 0:
            return {}
        try:
            return self._scores_fn(agent_id)
        except Exception:  # noqa: BLE001 - the snapshot is already stored; keep it and wait for good ratings
            logger.error(f"Snapshot scores of {agent_id} failed; retention treats them as not rated yet",
                         exc_info=not self._scores_failed)
            self._scores_failed = True
            return None

    def _retain(self, agent_id: str, entries: list[CheckpointInfo],
                protect: str | None = None) -> tuple[list[CheckpointInfo], list[str]]:
        """Apply ``arena.retention.retain`` to ``entries`` (sorted by version); returns the kept ones and the ids
        evicted (the caller updates the index first, then calls ``_notify_evicted``)."""
        if not entries:
            return [], []
        self._refresh_pins(entries)
        scores = self._scores(agent_id)
        infos = [SnapshotInfo(c.checkpoint_id, c.policy_version, c.meta.get("final") is True, c.meta.get("run_id"),
                              c.meta.get("pinned") is True) for c in entries]
        decision = retain(infos, scores, self._policy(), protect=protect)
        self._reasons[agent_id] = dict(decision.reasons)
        if decision.warnings and (agent_id, decision.waiting) not in self._warned:   # once (ruling D4)
            self._warned.add((agent_id, decision.waiting))
            for warning in decision.warnings:
                logger.warning(f"Snapshots of {agent_id}: {warning}")
        if decision.notes and agent_id not in self._noted:
            self._noted.add(agent_id)
            for note in decision.notes:
                logger.info(f"Snapshots of {agent_id}: {note}")
        keep = set(decision.keep)
        newest = {c.checkpoint_id for c in entries[-self._keep_last:]}
        kept: list[CheckpointInfo] = []
        evicted: list[str] = []
        for c in entries:
            if c.checkpoint_id not in keep:
                self._evict(agent_id, c.checkpoint_id)
                logger.debug(f"Evicted checkpoint {c.checkpoint_id} of {agent_id}: "
                             f"{decision.reasons[c.checkpoint_id]}")
                evicted.append(c.checkpoint_id)
                continue
            kept.append(c)
            if c.checkpoint_id not in newest:
                (c.path / TRAINER_FILE).unlink(missing_ok=True)
        return kept, evicted

    def reapply_retention(self, agent_id: str) -> list[str]:
        """Rerun retention for ``agent_id`` without a save (new ratings, new pins); returns the evicted ids, after
        ``on_evict`` ran for each. Main loop only (spec block 4)."""
        entries = list(self._index.get(agent_id, []))
        if not entries:
            return []
        self._index[agent_id], evicted = self._retain(agent_id, entries)
        if evicted:
            logger.info(f"Retention of {agent_id}: evicted {evicted} (pool {len(self._index[agent_id])})")
        self._notify_evicted(agent_id, evicted)
        return evicted

    def pin(self, agent_id: str, checkpoint_id: str, pinned: bool = True) -> None:
        """Pin (or unpin) a stored snapshot: ``meta.json`` is rewritten atomically. A pinned snapshot is never
        evicted; an unpinned one becomes evictable at the next retention pass. ConfigError for an unknown id."""
        entry = next((c for c in self._index.get(agent_id, []) if c.checkpoint_id == checkpoint_id), None)
        if entry is None:
            stored = [c.checkpoint_id for c in self._index.get(agent_id, [])]
            raise ConfigError(f"agent {agent_id!r} has no stored snapshot {checkpoint_id!r} (stored: {stored[-10:]})")
        entry.meta.update(set_pinned(entry.path, pinned))

    def retention_reasons(self, agent_id: str) -> dict[str, str]:
        """``{checkpoint_id: "kept: ..." | "evicted: ..."}`` of the agent's last retention decision (``{}`` before
        one); diagnostics and tests."""
        return dict(self._reasons.get(agent_id, {}))

    def pinned(self, agent_id: str) -> list[str]:
        """Ids of the pinned snapshots of ``agent_id`` (as on disk now), by version."""
        entries = self._index.get(agent_id, [])
        self._refresh_pins(entries)
        return [c.checkpoint_id for c in entries if c.meta.get("pinned") is True]

    def _notify_evicted(self, agent_id: str, evicted: list[str]) -> None:
        """Call ``on_evict`` for every evicted id (the index is already updated); every callback runs, the
        first callback error is raised after the last one, and every later error is logged."""
        if self._on_evict is None:
            return
        first_error: Exception | None = None
        for checkpoint_id in evicted:
            try:
                self._on_evict(agent_id, checkpoint_id)
            except Exception as e:
                if first_error is None:
                    first_error = e
                else:
                    logger.exception(f"on_evict({agent_id!r}, {checkpoint_id!r}) failed as well (the first "
                                     f"eviction callback error is raised)")
        if first_error is not None:
            raise first_error

    def import_snapshots(self, src_checkpoints_dir: str | Path, agent_id: str,
                         expected_signature: str | None = None) -> list[str]:
        """Carry a previous run's snapshots of ``agent_id`` (``<src>/<agent_id>/ckpt_v*``) into this store.

        ``model.pt`` and ``meta.json`` are hard-linked (copied where a link fails), ``trainer_state.pt`` never;
        then the retention rules apply. The source is only read (no tmp cleanup), a malformed snapshot there
        is a ConfigError (as a strict resume). Ids already in this store are skipped. With
        ``expected_signature`` every source snapshot must carry that ``role_signature``. Returns the ids of the
        agent's snapshots after retention.
        """
        src_agent = Path(src_checkpoints_dir) / check_agent_id(agent_id)
        infos = _read_agent_dir(src_agent, strict=True)
        if expected_signature is not None:
            for info in infos:
                signature = info.meta.get("role_signature")
                if signature != expected_signature:
                    raise ConfigError(
                        f"Snapshot {info.path}: role signature {signature!r} differs from the agent's "
                        f"{expected_signature!r}; the snapshot pool of the resumed run cannot be carried over "
                        f"(resume from a checkpoint dir to start with an empty pool)"
                    )
        entries = list(self._index.get(agent_id, []))
        present = {c.checkpoint_id for c in entries}
        agent_dir = self._agent_dir(agent_id)
        imported = 0
        for info in infos:
            if info.checkpoint_id in present:
                continue
            agent_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = agent_dir / f"{_TMP_PREFIX}{info.checkpoint_id}-{uuid.uuid4().hex[:8]}"
            tmp_dir.mkdir()
            try:
                for name in (MODEL_FILE, META_FILE):
                    link_or_copy(info.path / name, tmp_dir / name)
                os.replace(tmp_dir, agent_dir / info.checkpoint_id)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
            entries.append(CheckpointInfo(info.checkpoint_id, agent_id, info.policy_version,
                                          agent_dir / info.checkpoint_id, info.timestamp, dict(info.meta)))
            imported += 1
        entries.sort(key=lambda c: c.policy_version)
        self._index[agent_id] = entries          # indexed before retention: the dirs are in place
        self._index[agent_id], evicted = self._retain(agent_id, entries)
        kept = [c.checkpoint_id for c in self._index[agent_id]]
        if infos:
            skipped = len(infos) - imported
            logger.info(f"Imported {imported} snapshots of {agent_id} from {src_agent}"
                        f"{f' ({skipped} already present)' if skipped else ''}; pool now {kept}")
        self._notify_evicted(agent_id, evicted)
        return kept

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


def read_checkpoint_meta(ckpt_dir: str | Path) -> dict:
    """The validated ``meta.json`` of one checkpoint dir, read without touching the dir.

    A missing dir or an invalid ``meta.json`` raises ConfigError naming the dir.
    """
    path = Path(ckpt_dir)
    try:
        if not path.is_dir():
            raise ValueError("not a directory")
        return _parse_meta(path)
    except ValueError as e:
        raise ConfigError(f"Malformed checkpoint {path}: {e}") from e


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

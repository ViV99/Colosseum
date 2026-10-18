"""The per-machine shared-memory weight cache (SP5 spec blocks 4.6 and 7.8; ruling R10).

Workers and evaluators read weights only from here; the writer of an agent's latest weights is its learner when the
learner runs on this machine, else the node (versions fetched from another machine); snapshots and hub blobs (frozen,
teacher, ``init`` and evaluation weights; immutable, by id) are written by the node. One writer per file, any number
of readers, no locks between readers and writers (spec 4.7 principle; only taking over a node lock and removing a
dead node's files hold the cache directory's ``flock``, for a few file operations).

Files in :func:`shm_dir` (``$COLOSSEUM_SHM_DIR``, else ``/dev/shm`` on Linux, else ``<tempdir>/colosseum-shm``), all
named with the prefix ``P = colosseum-<ca8>-<run8>-n<node>`` (:func:`cache_prefix`), so two hubs, two runs and two
nodes on one machine never share a file:

=============================  ===========================================================  ====================
``{P}-{agent}-v{N}``           weights frame (``colosseum.net.chunks.encode_weights``)       learner or node
``{P}-{agent}.latest``         JSON ``{"segment", "version", "generation", "size",          same single writer
                               "digest", "teacher_active"}``, replaced atomically
``{P}-{agent}-{checkpoint}``   weights frame of a snapshot (``ckpt_v<N>``)                  node
``{P}-blob-{kind}-{blob_id}``  weights frame of a hub blob, ``kind`` in ``fixed`` /         node
                               ``teacher`` / ``init`` / ``eval`` (:func:`blob_key`, A18)
``{P}.lock``                   JSON ``{"pid", "proc_start"}`` of the node process            node
=============================  ===========================================================  ====================

Segments are immutable plain (never lz4) frames: a writer writes ``.{name}.tmp-<pid>-<thread>`` (mode ``0600``,
created exclusively) and renames it into place, then replaces the pointer; every ``latest`` publish rescans the
agent's segments on disk and keeps the newest 2 versions (so a handover between the node and a learner as the
writer leaks nothing). A reader opens the segment with ``os.open`` + ``mmap`` (read-only) and decodes the frame
into ``np.frombuffer`` views (no copy; the mapping lives while any view does; ``load_state_dict`` copies). Deleting
a file a reader has mapped is safe on POSIX; a reader that finds the pointer's segment gone re-reads the pointer
(``READ_RETRIES``). Not ``multiprocessing.shared_memory`` (on Python 3.12 another process unlinks a segment it did
not create) and not a seqlock (readers starve): ``research/sp5_data_plane.md`` §2.6.

One blob segment may serve several agents: the file name carries only ``{kind}-{blob_id}``, and blob ids are
content-addressed (or embed the agent). :meth:`WeightCacheWriter.drop` of a blob is therefore safe only for such ids
(content-addressed, or agent-embedding): never for an id another agent's different weights could reuse. Agent ids
``blob`` and ``blob-*`` are refused (their files would collide with blob names).

A missing (or empty) segment reads as ``None`` silently; a corrupt or mismatched segment, a corrupt or empty
pointer, and any other I/O error read as ``None`` with one WARNING per file (per file and errno for I/O errors), so
a lineup held for missing weights always has a log line. Before every write the free space must be
``size x 2 + 64 MiB``, else :class:`CacheFullError` (the node logs it as an ERROR with the hint and takes no work).
:func:`cleanup_dead` removes only the files of prefixes whose lock names a dead process, and stray files of prefixes
without a lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import mmap
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from colosseum.core.run_state import process_start_ticks
from colosseum.net.chunks import WeightsMeta, decode_weights
from colosseum.net.frame import FLAG_LZ4, FRAME_MAGIC, FrameError

logger = logging.getLogger(__name__)

SHM_DIR_ENV = "COLOSSEUM_SHM_DIR"
KEEP_LATEST = 2                   # versions of an agent's latest weights kept (the pointer's and the previous one)
READ_RETRIES = 5
SPACE_MARGIN_BYTES = 64 * 2**20
BLOB_KINDS = ("fixed", "teacher", "init", "eval")
RESERVED_AGENT_PREFIX = "blob"     # agent ids "blob" / "blob-*" would collide with blob segment names
_FRAME_PREFIX = struct.Struct("<4sHH")   # magic, version, flags (colosseum.net.frame)
_PREFIX_RE = re.compile(r"\.?(colosseum-[0-9a-f]{8}-[0-9a-f]{8}-n[0-9]+)[-.]")
_HEX_ID_RE = re.compile(r"[0-9a-f]{8,}")
_AGENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")
_CKPT_RE = re.compile(r"ckpt_v[0-9]+")
_BLOB_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}")
_BLOB_RE = re.compile(r"(fixed|teacher|init|eval)-[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}")   # A18 (always fullmatch)
FULL_HINT = ("the shared-memory weight cache is full: free space in {root} (Docker: `docker run --shm-size=1g`, "
             "K8s: an `emptyDir: {{medium: Memory}}` volume at /dev/shm), or set COLOSSEUM_SHM_DIR to a larger "
             "tmpfs")


class CacheFullError(RuntimeError):
    """Not enough space for a segment (message: the hint)."""


class CacheLockedError(RuntimeError):
    """Another live process owns this cache prefix (a second node of the same identity and run on this machine)."""


def shm_dir() -> Path:
    """Where the cache lives: ``$COLOSSEUM_SHM_DIR``, else ``/dev/shm`` (Linux), else ``<tempdir>/colosseum-shm``."""
    configured = os.environ.get(SHM_DIR_ENV)
    if configured:
        return Path(configured)
    if sys.platform.startswith("linux") and Path("/dev/shm").is_dir():
        return Path("/dev/shm")
    return Path(tempfile.gettempdir()) / "colosseum-shm"


def cache_prefix(ca_fingerprint: str, run_id: str, node: int) -> str:
    """``colosseum-<ca8>-<run8>-n<node>`` (hex ids: the CA fingerprint and the uuid4-hex run id)."""
    if not _HEX_ID_RE.fullmatch(ca_fingerprint) or not _HEX_ID_RE.fullmatch(run_id):
        raise ValueError(f"cache_prefix needs hex ids, got {ca_fingerprint!r} / {run_id!r}")
    if int(node) < 0:
        raise ValueError(f"node number must be >= 0, got {node}")
    return f"colosseum-{ca_fingerprint[:8]}-{run_id[:8]}-n{int(node)}"


@dataclass(frozen=True)
class CacheKey:
    kind: Literal["latest", "snapshot", "blob"]
    agent_id: str = ""
    ident: str = ""                     # checkpoint id (snapshot), f"{kind}-{blob_id}" (blob, A18), "" for latest


@dataclass(frozen=True)
class CachedWeights:
    key: CacheKey
    state_dict: dict[str, np.ndarray]    # read-only views over the mapping (load_state_dict copies)
    meta: WeightsMeta
    nbytes: int


def blob_key(kind: str, agent_id: str, blob_id: str) -> CacheKey:
    """The cache key of a hub blob (A18): ``CacheKey("blob", agent_id, f"{kind}-{blob_id}")``, ``kind`` in
    :data:`BLOB_KINDS`; ValueError for another kind or a malformed ``blob_id``."""
    if kind not in BLOB_KINDS:
        raise ValueError(f"blob kind must be one of {BLOB_KINDS}, got {kind!r}")
    if not isinstance(blob_id, str) or not _BLOB_ID_RE.fullmatch(blob_id):
        raise ValueError(f"invalid blob id {blob_id!r}")
    return CacheKey("blob", agent_id, f"{kind}-{blob_id}")


def split_blob_key(key: CacheKey) -> tuple[str, str]:
    """``(kind, blob_id)`` of a :func:`blob_key` (what a ``BlobRequest`` for it carries)."""
    if key.kind != "blob" or not _BLOB_RE.fullmatch(key.ident):
        raise ValueError(f"not a blob key: {key!r}")
    kind, blob_id = key.ident.split("-", 1)
    return kind, blob_id


def _check_agent(agent_id: str) -> str:
    if not isinstance(agent_id, str) or not _AGENT_RE.fullmatch(agent_id):
        raise ValueError(f"invalid agent id {agent_id!r}")
    if agent_id == RESERVED_AGENT_PREFIX or agent_id.startswith(RESERVED_AGENT_PREFIX + "-"):
        raise ValueError(f"agent id {agent_id!r} is reserved: the weight cache names blob segments "
                         f"'<prefix>-{RESERVED_AGENT_PREFIX}-...'")
    return agent_id


def segment_name(prefix: str, key: CacheKey, version: int | None = None) -> str:
    """The file name of ``key`` (``latest`` needs ``version``); ValueError for a malformed key."""
    if key.kind == "latest":
        if version is None or int(version) < 0:
            raise ValueError("a latest segment needs its version")
        return f"{prefix}-{_check_agent(key.agent_id)}-v{int(version)}"
    if key.kind == "snapshot":
        if not _CKPT_RE.fullmatch(key.ident):
            raise ValueError(f"invalid checkpoint id {key.ident!r}")
        return f"{prefix}-{_check_agent(key.agent_id)}-{key.ident}"
    if key.kind == "blob":
        if not _BLOB_RE.fullmatch(key.ident):
            raise ValueError(f"invalid blob ident {key.ident!r} (A18: '<fixed|teacher|init|eval>-<blob id>')")
        return f"{prefix}-blob-{key.ident}"
    raise ValueError(f"unknown cache key kind {key.kind!r}")


def pointer_name(prefix: str, agent_id: str) -> str:
    return f"{prefix}-{_check_agent(agent_id)}.latest"


def frame_digest(frame: bytes | memoryview) -> str:
    return hashlib.blake2b(frame, digest_size=16).hexdigest()


def _temp_path(path: Path) -> Path:
    """A temp name next to ``path``, unique per process and thread (P-PF1-6; still matched by the prefix rule)."""
    return path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _write_file(path: Path, data: bytes | memoryview) -> None:
    """Write ``data`` to ``path`` through a fresh ``0600`` temp file and a rename (readers never see half a file).
    A stale temp of the same name (a crashed thread whose pid and thread id came back) is removed first; the temp
    file is created exclusively and never through a symlink."""
    tmp = _temp_path(path)
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _check_plain_frame(frame: bytes | memoryview) -> None:
    """The cache holds plain frames only: readers map them zero-copy (criterion 4); an lz4 frame would be
    decompressed into private memory by every reader."""
    if len(frame) >= _FRAME_PREFIX.size:
        magic, _version, flags = _FRAME_PREFIX.unpack_from(frame)
        if magic == FRAME_MAGIC and flags & FLAG_LZ4:
            raise ValueError("the weight cache takes plain frames only (got an lz4-compressed frame)")


class WeightCacheWriter:
    """Writes segments and pointers of one prefix (one process writes a given agent's latest at a time; when that
    writer changes, e.g. the node while the learner is remote and the learner once it runs here, the next writer
    prunes what the previous one left, because every publish rescans the agent's segments on disk)."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = Path(root)
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)

    def _check_space(self, size: int) -> None:
        free = shutil.disk_usage(self.root).free
        if free < 2 * size + SPACE_MARGIN_BYTES:
            raise CacheFullError(f"{size / 2**20:.1f} MiB segment, {free / 2**20:.1f} MiB free: "
                                 + FULL_HINT.format(root=self.root))

    def _prune(self, agent_id: str, keep: str) -> None:
        """Delete ``agent_id``'s latest segments on disk except ``keep`` and the newest ``KEEP_LATEST - 1`` others."""
        pattern = re.compile(re.escape(f"{self.prefix}-{agent_id}-v") + r"([0-9]+)")
        others = sorted(((int(m.group(1)), name) for name in os.listdir(self.root)
                         if name != keep and (m := pattern.fullmatch(name))), reverse=True)
        for _, name in others[KEEP_LATEST - 1:]:
            (self.root / name).unlink(missing_ok=True)

    def publish_latest(self, agent_id: str, frame: bytes, *, version: int, generation: int,
                       teacher_active: bool) -> str:
        """Write version ``version`` of ``agent_id``'s latest weights, point the pointer at it, keep the newest
        ``KEEP_LATEST`` versions; returns the segment name. If the pointer cannot be written, a segment this call
        created is removed again and the error is raised."""
        name = segment_name(self.prefix, CacheKey("latest", agent_id), version)
        _check_plain_frame(frame)
        self._check_space(len(frame))
        segment = self.root / name
        existed = segment.exists()
        _write_file(segment, frame)
        pointer = {"segment": name, "version": int(version), "generation": int(generation), "size": len(frame),
                   "digest": frame_digest(frame), "teacher_active": bool(teacher_active)}
        try:
            _write_file(self.root / pointer_name(self.prefix, agent_id), json.dumps(pointer).encode("utf-8"))
        except BaseException:
            if not existed:
                segment.unlink(missing_ok=True)
            raise
        self._prune(agent_id, name)
        return name

    def put(self, key: CacheKey, frame: bytes) -> str:
        """Write an immutable snapshot / blob segment (kept as is if it exists already); returns its name."""
        if key.kind == "latest":
            raise ValueError("latest weights go through publish_latest")
        name = segment_name(self.prefix, key)
        _check_plain_frame(frame)
        if not (self.root / name).exists():
            self._check_space(len(frame))
            _write_file(self.root / name, frame)
        return name

    def drop(self, key: CacheKey) -> None:
        """Remove a snapshot / blob segment (readers that mapped it keep their views)."""
        if key.kind == "latest":
            raise ValueError("latest segments are pruned by publish_latest")
        (self.root / segment_name(self.prefix, key)).unlink(missing_ok=True)


def _open_segment(path: Path) -> mmap.mmap:
    """A read-only mapping of ``path`` (FileNotFoundError when it is gone; ValueError when empty)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)


_GONE = object()


class WeightCacheReader:
    """Reads segments of one prefix. Never writes; never blocks on a writer."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = Path(root)
        self.prefix = prefix
        self._warned: set[tuple[str, int | None]] = set()

    def _warn_once(self, name: str, code: int | None, message: str) -> None:
        if (name, code) not in self._warned:
            self._warned.add((name, code))
            logger.warning(message)

    def _io_error(self, name: str, error: OSError) -> None:
        self._warn_once(name, error.errno, f"weight cache: cannot read {self.root / name} ({error}); "
                                           f"treated as not cached")

    def _pointer(self, agent_id: str) -> dict | None:
        name = pointer_name(self.prefix, agent_id)
        try:
            data = json.loads((self.root / name).read_bytes())
        except FileNotFoundError:
            return None
        except ValueError as e:           # empty or corrupt (renames are atomic: never half a pointer)
            self._warn_once(name, None, f"weight cache pointer {name} is not JSON ({e}); ignored")
            return None
        except OSError as e:
            self._io_error(name, e)
            return None
        ok = (isinstance(data, dict) and type(data.get("version")) is int and data["version"] >= 0
              and type(data.get("generation")) is int and data["generation"] >= 0
              and data.get("segment") == segment_name(self.prefix, CacheKey("latest", agent_id), data["version"]))
        if not ok:
            self._warn_once(name, None, f"weight cache pointer {name} is malformed ({str(data)[:200]}); ignored")
            return None
        return data

    def _map(self, name: str) -> mmap.mmap | object | None:
        """The mapping of ``name``; ``_GONE`` when the file does not exist; None when it cannot be read (empty: a
        silent None; another I/O error: one WARNING per errno)."""
        try:
            return _open_segment(self.root / name)
        except FileNotFoundError:
            return _GONE
        except ValueError:
            return None                   # an empty file
        except OSError as e:
            self._io_error(name, e)
            return None

    def latest_version(self, agent_id: str) -> tuple[int, int] | None:
        """``(version, generation)`` the pointer names (no segment is opened), or None."""
        pointer = self._pointer(agent_id)
        return None if pointer is None else (pointer["version"], pointer["generation"])

    def _decode(self, key: CacheKey, mapping: mmap.mmap, name: str) -> CachedWeights | None:
        try:
            state_dict, meta = decode_weights(memoryview(mapping), cap=len(mapping))
        except FrameError as e:
            self._warn_once(name, None, f"weight cache segment {name} is not a weights frame ({e}); ignored")
            return None
        return CachedWeights(key, state_dict, meta, len(mapping))

    def read_latest(self, agent_id: str, *, newer_than: int = -1) -> CachedWeights | None:
        """The pointer's version if it is newer than ``newer_than``, else None (also when nothing is published)."""
        for _ in range(READ_RETRIES):
            pointer = self._pointer(agent_id)
            if pointer is None or pointer["version"] <= newer_than:
                return None
            name = pointer["segment"]
            mapping = self._map(name)
            if mapping is _GONE:
                continue                  # replaced and pruned since the pointer was read: read it again
            if mapping is None:
                return None
            cached = self._decode(CacheKey("latest", agent_id), mapping, name)
            if cached is not None and (cached.meta.agent_id != agent_id
                                       or cached.meta.policy_version != pointer["version"]):
                self._warn_once(name, None, f"weight cache segment {name} holds {cached.meta.agent_id} "
                                            f"v{cached.meta.policy_version}, the pointer says {agent_id} "
                                            f"v{pointer['version']}; ignored")
                return None
            return cached
        return None

    def get(self, key: CacheKey) -> CachedWeights | None:
        """The weights of ``key`` (``latest``: the pointer's version), or None when absent or unreadable."""
        if key.kind == "latest":
            return self.read_latest(key.agent_id)
        name = segment_name(self.prefix, key)
        mapping = self._map(name)
        if mapping is _GONE or mapping is None:
            return None
        return self._decode(key, mapping, name)

    def has(self, key: CacheKey) -> bool:
        if key.kind == "latest":
            return (self.root / pointer_name(self.prefix, key.agent_id)).exists()
        return (self.root / segment_name(self.prefix, key)).exists()


# ---------------------------------------------------------------------------
# Ownership and cleanup
# ---------------------------------------------------------------------------


def _read_lock(lock: Path) -> tuple[int, int | None] | None:
    """``(pid, proc_start)`` of a lock file; None when it is missing or broken (a broken lock owns nothing)."""
    try:
        data = json.loads(lock.read_text())
        pid, start = data["pid"], data.get("proc_start")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if type(pid) is not int or (start is not None and type(start) is not int):
        return None
    return pid, start


def _owner_alive(lock: Path) -> bool:
    """Whether the process a lock file names still runs (pid, and its start time where the OS tells it).
    The same rule as ``colosseum.core.run_state.process_alive`` (T4.4 makes that portable; then use it here)."""
    owner = _read_lock(lock)
    if owner is None or owner[0] <= 0:
        return False
    pid, start = owner
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                              # exists, owned by another user
    now = process_start_ticks(pid)
    return not (start is not None and now is not None and start != now)


def _lock_body() -> bytes:
    return json.dumps({"pid": os.getpid(), "proc_start": process_start_ticks(os.getpid())}).encode("utf-8")


@contextmanager
def _dir_lock(root: Path) -> Iterator[None]:
    """An exclusive ``flock`` on the cache directory itself, held for a few file operations: taking a lock over,
    and removing a dead owner's lock and files, are then exclusive between processes (the directory is never
    removed, unlike a lock file, so every process locks the same inode)."""
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)                      # closing the descriptor releases the flock


def acquire_node_lock(root: Path, prefix: str) -> Path:
    """Write ``{P}.lock`` with this process's pid and start time (before any segment of the prefix);
    CacheLockedError if a live process owns the prefix already. The check and the write run under the directory's
    ``flock``, so of two processes taking over a missing or dead owner's lock exactly one wins."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f"{prefix}.lock"
    with _dir_lock(root):
        owner = _read_lock(lock)
        if owner is not None and owner[0] != os.getpid() and _owner_alive(lock):
            raise CacheLockedError(f"weight cache {lock} belongs to the running process {owner[0]} (another node of "
                                   f"the same identity and run on this machine?)")
        _write_file(lock, _lock_body())
    return lock


def _unlink(root: Path, name: str) -> bool:
    try:
        (root / name).unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as e:                  # e.g. another user's file in a sticky /dev/shm
        logger.debug(f"weight cache {root}: cannot remove {name}: {e}")
        return False


def _prefix_files(prefix: str, names: list[str]) -> list[str]:
    """The names of ``prefix``'s files other than its lock (temp files included)."""
    lock = f"{prefix}.lock"
    return [name for name in names
            if name != lock and (m := _PREFIX_RE.match(name)) is not None and m.group(1) == prefix]


def release_node_lock(root: Path, prefix: str) -> None:
    """Remove every file of ``prefix`` and its lock (a node that stops cleanly). A lock of another live process is
    left alone with a WARNING (its owner took the prefix over)."""
    root = Path(root)
    lock = root / f"{prefix}.lock"
    if not root.is_dir():
        return
    with _dir_lock(root):
        owner = _read_lock(lock)
        if owner is not None and owner[0] != os.getpid() and _owner_alive(lock):
            logger.warning(f"weight cache {lock} belongs to the running process {owner[0]}; its files are kept")
            return
        names = os.listdir(root)
        removed = [name for name in _prefix_files(prefix, names) if _unlink(root, name)]
        if f"{prefix}.lock" in names and _unlink(root, f"{prefix}.lock"):
            removed.append(f"{prefix}.lock")
    logger.debug(f"weight cache {prefix}: removed {len(removed)} files")


def cleanup_dead(root: Path) -> list[str]:
    """Remove the files of every prefix whose lock names a dead process, and stray files of prefixes without a lock
    (a lock is always written before a prefix's first segment). Live owners' files are never touched; other files
    in ``root`` neither. Each prefix is re-checked and removed under the directory's ``flock`` (an owner that took
    the prefix over since the listing keeps everything). Returns the removed names."""
    root = Path(root)
    try:
        names = os.listdir(root)
    except (FileNotFoundError, NotADirectoryError):
        return []
    prefixes = {m.group(1) for name in names if (m := _PREFIX_RE.match(name))}
    removed: list[str] = []
    for prefix in sorted(prefixes):
        lock_name = f"{prefix}.lock"
        lock = root / lock_name
        if lock_name in names and _owner_alive(lock):
            continue                      # cheap pre-check without the directory lock
        with _dir_lock(root):
            if lock.exists() and (_owner_alive(lock) or lock_name not in names):
                continue                  # alive now, or an owner that started after the listing
            removed += [name for name in _prefix_files(prefix, names) if _unlink(root, name)]
            if lock_name in names and _unlink(root, lock_name):
                removed.append(lock_name)
    if removed:
        logger.info(f"weight cache {root}: removed {len(removed)} files of dead nodes")
    return removed

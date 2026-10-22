"""The code bundle of a hub (SP5 spec 2.3): one deterministic ``tar.gz`` of the code every node runs.

Archive layout (POSIX member names, sorted, regular files only, zeroed metadata)::

    bundle.json        {"format": 1, "python_path": [...], "files": N}
    project/...        the hub's project directory (its working directory) minus the exclusions
    pkg/<i>/...        import roots that are not part of ``project/``: the ``colosseum`` package when it is
                       installed outside the project, editable packages installed from paths outside it

``python_path`` lists the ``PYTHONPATH`` entries of node processes, relative to the unpacked directory and in the
hub's order: ``project`` first (the hub puts its cwd first on ``sys.path``), then import roots inside the project
(``project/src`` for a source checkout), then ``pkg/<i>``. The code hash is the SHA-256 of the archive bytes.

The module imports only the standard library at module level and nothing from ``colosseum``: the installer
script embeds it next to the loader (T6.2), which unpacks bundles before any colosseum code is on the machine.
"""

from __future__ import annotations

import ast
import contextlib
import fnmatch
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shutil
import stat
import tarfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
MANIFEST = "bundle.json"
ARCHIVE_SUFFIX = ".tar.gz"
COMPLETE_MARKER = ".colosseum-bundle.json"     # written into the unpacked dir before it is renamed into place
PROJECT = "project"
PKG = "pkg"
MIB = 1 << 20
MAX_ARCHIVE_BYTES = 2 << 30                    # a bundle larger than this is refused (both sides)
MAX_UNPACKED_BYTES = 8 << 30
MAX_MEMBERS = 200_000
# A hub started from the wrong directory (a home dir) must not walk or ship it.
MAX_PROJECT_FILES = 50_000
MAX_PROJECT_BYTES = 1 << 30
# Never bundled from the project (spec 2.3; plus WandB's ./wandb run files): top-level directories, directories at
# any depth, file patterns.
TOP_EXCLUDES = ("runs", ".superpowers", "wandb")
ANY_DIR_EXCLUDES = (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")
FILE_EXCLUDES = ("*.pyc", "*.pyo")
_HASH_RE = re.compile(r"[0-9a-f]{64}")


class BundleError(Exception):
    """A bundle that cannot be built, verified or unpacked (the message says which file and why)."""


@dataclass(frozen=True)
class ImportRoot:
    """A directory a package is imported from, and what of it goes into the bundle."""

    path: Path                         # the sys.path entry (absolute)
    packages: tuple[str, ...] = ()     # top-level package dirs / module files under it; () = the whole dir
    name: str = ""                     # for messages (distribution name)


@dataclass(frozen=True)
class Bundle:
    path: Path                         # <out_dir>/<code_hash>.tar.gz
    code_hash: str                     # sha256 hex of the archive bytes
    files: tuple[str, ...]             # member names (without bundle.json), sorted
    size: int                          # archive bytes
    python_path: tuple[str, ...] = ()  # PYTHONPATH entries relative to the unpacked dir
    skipped: tuple[str, ...] = ()      # project files left out by the size limit (relative paths)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# import roots of the hub's environment
# ---------------------------------------------------------------------------------------------------------------

def _package_root(name: str) -> ImportRoot | None:
    """The sys.path entry a top-level package or module is imported from (no import of the package itself)."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        pkg_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        return ImportRoot(pkg_dir.parent, (pkg_dir.name,), name)
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        origin = Path(spec.origin).resolve()
        return ImportRoot(origin.parent, (origin.name,), name)
    return None


def _editable_roots(dist: importlib.metadata.Distribution) -> list[ImportRoot] | None:
    """Import roots of an editable distribution: path lines of its ``.pth`` files (hatchling, flit, pdm,
    setuptools "compat"; a relative line is relative to the ``.pth`` file's directory, as in ``site``), or the
    ``MAPPING`` of a setuptools ``__editable___*_finder.py``. None: not found."""
    roots: list[ImportRoot] = []
    name = dist.metadata["Name"] or "?"
    for file in dist.files or ():
        located = Path(str(dist.locate_file(file)))
        if file.suffix == ".pth":
            try:
                lines = located.read_text().splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith(("#", "import ", "import\t")):
                    continue
                entry = Path(line) if Path(line).is_absolute() else located.parent / line
                if entry.is_dir():
                    roots.append(ImportRoot(entry.resolve(), (), name))
        elif file.name.startswith("__editable___") and file.name.endswith("_finder.py"):
            try:
                tree = ast.parse(located.read_text())
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAPPING"
                                                         for t in node.targets)):
                    try:
                        mapping = ast.literal_eval(node.value)
                    except ValueError:
                        continue
                    for target in mapping.values():
                        p = Path(target).resolve()
                        roots.append(ImportRoot(p.parent, (p.name,), name))
    return roots or None


def import_roots(*, distributions: Iterable[importlib.metadata.Distribution] | None = None) -> list[ImportRoot]:
    """The ``colosseum`` package's root first, then the roots of every other editable distribution of the
    hub's environment (one WARNING per editable distribution whose sources cannot be located)."""
    roots: list[ImportRoot] = []
    own = _package_root("colosseum")
    if own is not None:
        roots.append(own)
    for dist in distributions if distributions is not None else importlib.metadata.distributions():
        raw = dist.read_text("direct_url.json")
        if not raw:
            continue
        try:
            info = json.loads(raw)
        except ValueError:
            continue
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if not info.get("dir_info", {}).get("editable") or name == "colosseum":
            continue
        found = _editable_roots(dist)
        if found is None:
            logger.warning("editable package %s: its sources were not found, so it is not in the code bundle "
                           "(add it with distributed.bundle.include or install it on the nodes with "
                           "distributed.requirements)", dist.metadata["Name"])
            continue
        roots.extend(found)
    return roots


# ---------------------------------------------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------------------------------------------

def _hit(rel: str, patterns: Sequence[str]) -> bool:
    """``rel`` (a POSIX path relative to the project) or one of its parent directories matches a pattern."""
    parts = PurePosixPath(rel).parts
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    return any(fnmatch.fnmatchcase(prefix, pattern.rstrip("/")) for pattern in patterns for prefix in prefixes)


def _may_contain(rel_dir: str, patterns: Sequence[str]) -> bool:
    """An include pattern names something inside the directory ``rel_dir``: its literal prefix (up to the first
    wildcard) is that directory, lies under it or above it. A pattern without a literal prefix (``*.pt``) never
    opens a default-excluded directory."""
    for pattern in patterns:
        literal = pattern.rstrip("/")
        for i, ch in enumerate(literal):
            if ch in "*?[":
                literal = literal[:i]
                break
        literal = literal.rstrip("/")
        if literal and (literal == rel_dir or literal.startswith(rel_dir + "/") or rel_dir.startswith(literal + "/")):
            return True
    return False


def _default_excluded(rel: str, *, is_dir: bool, top_excludes: Sequence[str] = TOP_EXCLUDES) -> bool:
    parts = PurePosixPath(rel).parts
    if parts and parts[0] in top_excludes:
        return True
    dir_parts = parts if is_dir else parts[:-1]
    if any(part in ANY_DIR_EXCLUDES for part in dir_parts):
        return True
    return (not is_dir) and any(fnmatch.fnmatchcase(parts[-1], p) for p in FILE_EXCLUDES)


def _walk(root: Path, *, include: Sequence[str], exclude: Sequence[str], max_bytes: int, skip_dirs: Sequence[Path],
          top_excludes: Sequence[str] = TOP_EXCLUDES) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """(rel path, source) of every file of ``root`` that goes into the bundle; skipped big files; skipped
    symlinked directories. ``include`` beats the default exclusions and the size limit; ``exclude`` beats both.
    ``top_excludes``: the top-level directory names left out (the project's ``runs/`` ...; none for import roots)."""
    files: list[tuple[str, Path]] = []
    big: list[str] = []
    links: list[str] = []
    total = 0
    skip = {p.resolve() for p in skip_dirs}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        rel_here = here.relative_to(root).as_posix()
        rel_here = "" if rel_here == "." else rel_here
        kept = []
        for d in sorted(dirnames):
            rel = f"{rel_here}/{d}" if rel_here else d
            full = here / d
            if full.is_symlink():
                links.append(rel)
                continue
            if full.resolve() in skip or _hit(rel, exclude):
                continue
            if _default_excluded(rel, is_dir=True, top_excludes=top_excludes) and not _may_contain(rel, include):
                continue
            kept.append(d)
        dirnames[:] = kept
        for name in sorted(filenames):
            rel = f"{rel_here}/{name}" if rel_here else name
            full = here / name
            if _hit(rel, exclude):
                continue
            included = _hit(rel, include)
            if _default_excluded(rel, is_dir=False, top_excludes=top_excludes) and not included:
                continue
            try:
                st = full.stat()                       # follows a symlinked file
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if st.st_size > max_bytes and not included:
                big.append(rel)
                continue
            files.append((rel, full))
            total += st.st_size
            if len(files) > MAX_PROJECT_FILES or total > MAX_PROJECT_BYTES:
                raise BundleError(f"{root} holds more than {MAX_PROJECT_FILES} files or {MAX_PROJECT_BYTES >> 20} "
                                  "MiB to bundle: start the hub from the project directory, or leave paths out with "
                                  "distributed.bundle.exclude")
    return files, big, links


def _inside(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def build_bundle(project_dir: Path, *, include: Sequence[str], exclude: Sequence[str], max_file_mb: float,
                 out_dir: Path, roots: Sequence[ImportRoot] | None = None) -> Bundle:
    """Archive the project dir and the import roots (``import_roots()`` by default) into
    ``out_dir/<code hash>.tar.gz`` (atomic; an existing archive of the same hash is reused)."""
    project = Path(project_dir).resolve()
    if project == Path.home().resolve() or project.parent == project:
        raise BundleError(f"the hub's working directory {project} is a home or root directory: start the hub "
                          "from the project directory (it is what the nodes receive)")
    out = Path(out_dir).resolve()
    roots = import_roots() if roots is None else list(roots)
    max_bytes = int(max_file_mb * MIB)
    files, big, links = _walk(project, include=include, exclude=exclude, max_bytes=max_bytes, skip_dirs=[out])
    members: dict[str, Path] = {f"{PROJECT}/{rel}": src for rel, src in files}
    python_path: list[str] = [PROJECT]
    outside = 0
    # An import root outside the project may contain it (an editable install of a parent directory): its copy
    # leaves out the project's run files as the project walk does.
    root_skip = [out, *(project / d for d in TOP_EXCLUDES)]
    for root in roots:
        rel_root = _inside(root.path, project)
        if rel_root not in (None, ".") and not _hit(rel_root, exclude) \
                and not _default_excluded(rel_root, is_dir=True):
            entry = f"{PROJECT}/{rel_root}"           # already in project/: only a PYTHONPATH entry
            if entry not in python_path:
                python_path.append(entry)
            continue
        if rel_root == ".":
            continue                                   # the project dir itself is already entry "project"
        prefix = f"{PKG}/{outside}"
        outside += 1
        python_path.append(prefix)
        if root.packages:                              # package dirs / module files under the root
            for top in root.packages:
                src = root.path / top
                if src.is_dir():
                    sub, sub_big, _ = _walk(src, include=(), exclude=(), max_bytes=max_bytes, skip_dirs=root_skip,
                                            top_excludes=())
                    big.extend(f"{root.name}:{top}/{r}" for r in sub_big)
                    members.update({f"{prefix}/{top}/{rel}": path for rel, path in sub})
                elif src.is_file():
                    members[f"{prefix}/{top}"] = src
        else:                                          # a whole sys.path directory (an editable .pth entry)
            sub, sub_big, _ = _walk(root.path, include=(), exclude=(), max_bytes=max_bytes, skip_dirs=root_skip,
                                    top_excludes=())
            big.extend(f"{root.name}:{r}" for r in sub_big)
            members.update({f"{prefix}/{rel}": path for rel, path in sub})
    if big:
        logger.warning("code bundle: %d file(s) larger than distributed.bundle.max_file_mb (%s MB) left out: %s%s "
                       "(add them with distributed.bundle.include)", len(big), max_file_mb, ", ".join(big[:5]),
                       " ..." if len(big) > 5 else "")
    if links:
        logger.warning("code bundle: %d symlinked director%s not followed: %s", len(links),
                       "y" if len(links) == 1 else "ies", ", ".join(links[:5]))
    manifest = json.dumps({"format": BUNDLE_FORMAT, "python_path": python_path, "files": len(members)},
                          sort_keys=True).encode()
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".tmp-{uuid.uuid4().hex}{ARCHIVE_SUFFIX}"
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    tar.addfile(_info(MANIFEST, len(manifest), 0o644), _BytesReader(manifest))
                    for name in sorted(members):
                        _add_file(tar, name, members[name])
        archive_bytes = tmp.stat().st_size
        if archive_bytes > MAX_ARCHIVE_BYTES:
            raise BundleError(f"code bundle of {project} is {archive_bytes >> 20} MiB, above the "
                              f"{MAX_ARCHIVE_BYTES >> 20} MiB nodes accept: leave large files out with "
                              "distributed.bundle.exclude or a smaller distributed.bundle.max_file_mb (weights and "
                              "data the nodes need are fetched from the hub, not shipped in the bundle)")
        code_hash = sha256_file(tmp)
        final = out / f"{code_hash}{ARCHIVE_SUFFIX}"
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return Bundle(final, code_hash, tuple(sorted(members)), final.stat().st_size, tuple(python_path), tuple(big))


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data, self._pos = data, 0

    def read(self, n: int = -1) -> bytes:
        end = len(self._data) if n < 0 else self._pos + n
        chunk, self._pos = self._data[self._pos:end], min(end, len(self._data))
        return chunk


def _info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.type = size, 0, tarfile.REGTYPE
    info.mode = 0o755 if mode & 0o111 else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _add_file(tar: tarfile.TarFile, name: str, src: Path) -> None:
    try:
        with open(src, "rb") as f:
            st = os.fstat(f.fileno())
            tar.addfile(_info(name, st.st_size, st.st_mode), f)
    except OSError as e:
        raise BundleError(f"code bundle: cannot read {src} ({e}); was it changed while the bundle was built?") from e


# ---------------------------------------------------------------------------------------------------------------
# unpacking (node side)
# ---------------------------------------------------------------------------------------------------------------

def bundles_dir(cache_root: Path) -> Path:
    return Path(cache_root) / "colosseum" / "bundles"


def _safe_name(name: str) -> str:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or "\\" in name or any(part in ("..", ".", "") for part in name.split("/")):
        raise BundleError(f"code bundle: unsafe member name {name!r}")
    return p.as_posix()


def _check_manifest(tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
    """``bundle.json`` (read from the archive before anything is written): format 1, safe relative entries."""
    src = tar.extractfile(member)
    assert src is not None
    with src:
        manifest = json.loads(src.read())
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        found = manifest.get("format") if isinstance(manifest, dict) else manifest
        raise BundleError(f"code bundle format {found} is not supported "
                          f"(this loader reads format {BUNDLE_FORMAT}; update colosseum on this machine)")
    entries = manifest.get("python_path")
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise BundleError("code bundle: bundle.json has no python_path list")
    for entry in entries:
        _safe_name(entry)


def unpack_bundle(archive: Path, code_hash: str, *, cache_root: Path) -> Path:
    """Verify ``archive`` against ``code_hash`` and unpack it into ``<cache_root>/colosseum/bundles/<hash>/``
    (atomic rename; an existing unpacked bundle of that hash is reused). Only regular files with safe relative
    names are accepted; sizes are checked before anything is written."""
    if not _HASH_RE.fullmatch(code_hash):
        raise BundleError(f"code bundle: {code_hash!r} is not a code hash (64 lowercase hex digits)")
    target = bundles_dir(cache_root) / code_hash
    if (target / COMPLETE_MARKER).is_file():
        return target
    archive = Path(archive)
    try:
        archive_bytes = archive.stat().st_size
    except OSError as e:
        raise BundleError(f"code bundle {archive} cannot be read: {e}") from e
    if archive_bytes > MAX_ARCHIVE_BYTES:
        raise BundleError(f"code bundle {archive} is larger than {MAX_ARCHIVE_BYTES >> 20} MiB")
    actual = sha256_file(archive)
    if actual != code_hash:
        raise BundleError(f"code bundle hash mismatch: expected {code_hash}, got {actual}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp-{code_hash[:16]}-{uuid.uuid4().hex[:8]}"
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_MEMBERS:
                raise BundleError(f"code bundle has {len(members)} members (limit {MAX_MEMBERS})")
            total, names = 0, set()
            for m in members:
                name = _safe_name(m.name)
                if not m.isreg():
                    raise BundleError(f"code bundle: member {m.name!r} is not a regular file")
                if name in names:
                    raise BundleError(f"code bundle: duplicate member {m.name!r}")
                names.add(name)
                total += m.size
            if total > MAX_UNPACKED_BYTES:
                raise BundleError(f"code bundle unpacks to {total} bytes (limit {MAX_UNPACKED_BYTES})")
            if MANIFEST not in names:
                raise BundleError("code bundle has no bundle.json")
            _check_manifest(tar, next(m for m in members if m.name == MANIFEST))
            for m in members:
                dest = tmp / _safe_name(m.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.chmod(dest, 0o755 if m.mode & 0o111 else 0o644)
        (tmp / COMPLETE_MARKER).write_text(json.dumps({"code_hash": code_hash, "files": len(names)}))
        _place(tmp, target)
    except (tarfile.TarError, OSError, ValueError, EOFError) as e:
        raise BundleError(f"code bundle {archive} cannot be unpacked: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def cache_lock_path(directory: Path) -> Path:
    """The lock file of a node cache directory (``bundles`` / ``envs``): next to it, so its listing stays clean."""
    directory = Path(directory)
    return directory.with_name(directory.name + ".lock")


@contextlib.contextmanager
def cache_lock(directory: Path):
    """An exclusive ``flock`` on ``cache_lock_path(directory)``. Held while an entry is placed (``_place``), while
    the loader marks a complete entry in use and while it prunes (T6.2), so a marker check and the rename or
    removal that follows it never interleave with another loader's placement, use or pruning."""
    import fcntl  # POSIX only: nodes run on Linux / macOS

    path = cache_lock_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _place(tmp: Path, target: Path) -> None:
    """Rename the complete ``tmp`` dir to ``target`` under the cache lock. A complete ``target`` (another loader won
    the race) is kept; a ``target`` without the completion marker is a stale partial dir (placements hold the lock,
    so no loader is writing it): it is moved away and removed first."""
    with cache_lock(target.parent):
        if (target / COMPLETE_MARKER).is_file():
            return
        if target.is_dir():
            stale = target.parent / f".stale-{target.name[:16]}-{uuid.uuid4().hex[:8]}"
            logger.warning("code bundle: removing the incomplete directory %s (no %s)", target, COMPLETE_MARKER)
            os.rename(target, stale)
            shutil.rmtree(stale, ignore_errors=True)
        os.rename(tmp, target)


def bundle_python_path(unpacked: Path) -> list[Path]:
    """Absolute ``PYTHONPATH`` entries of an unpacked bundle, in order."""
    manifest = json.loads((Path(unpacked) / MANIFEST).read_text())
    return [Path(unpacked) / entry for entry in manifest["python_path"]]


def project_dir(unpacked: Path) -> Path:
    """The working directory of node processes (the bundle's project dir)."""
    return Path(unpacked) / PROJECT

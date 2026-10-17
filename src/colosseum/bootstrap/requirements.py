"""Python requirements of node environments (SP5 spec 2.4): the hub environment's exact versions.

Default list: every installed distribution as ``name==version`` (local version labels such as ``+cu130``
dropped), without the packages whose build depends on the node's hardware (``torch``, ``triton``,
``nvidia-*``, ``cuda-*``, ``pytorch-triton*``), without ``colosseum`` and without distributions installed from
local paths (editable or ``file://``: their sources travel in the code bundle). ``distributed.requirements``
replaces the list with a file of the user's. The loader adds ``torch==<the hub's version>`` itself, with the
build chosen for the node (``torch_select``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import re
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDED = ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_COMMENT_RE = re.compile(r"(^|\s)#.*$")                 # pip: a comment starts at "#" after whitespace


def canonical_name(name: str) -> str:
    """PEP 503 normalized name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def excluded(name: str) -> bool:
    canon = canonical_name(name)
    return any(fnmatchcase(canon, pattern) for pattern in EXCLUDED)


def _from_local_path(dist: importlib.metadata.Distribution) -> bool:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return False
    try:
        return str(json.loads(raw).get("url", "")).startswith("file:")
    except ValueError:
        return False


def hub_freeze(distributions: Iterable[importlib.metadata.Distribution] | None = None) -> list[str]:
    """``name==version`` of the hub environment's distributions without the exclusions, sorted by name; the first
    distribution of a name wins (``importlib.metadata`` order = ``sys.path`` order)."""
    seen: dict[str, str] = {}
    for dist in distributions if distributions is not None else importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        canon = canonical_name(name)
        if canon in seen or excluded(canon) or _from_local_path(dist):
            seen.setdefault(canon, "")
            continue
        seen[canon] = f"{canon}=={dist.version.split('+', 1)[0]}"
    return sorted(line for line in seen.values() if line)


def read_requirements_file(path: Path) -> list[str]:
    """The lines of a user requirements file: comments and blanks dropped; pip options are refused
    (``ValueError``); torch and the other hardware-dependent packages are dropped with one WARNING."""
    lines: list[str] = []
    dropped: list[str] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = _COMMENT_RE.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError(f"{path}:{number}: pip options are not supported in distributed.requirements ({line!r})")
        match = _NAME_RE.match(line)
        if match is None:
            raise ValueError(f"{path}:{number}: not a requirement: {line!r}")
        if excluded(match.group(1)):
            dropped.append(line)
            continue
        lines.append(line)
    if dropped:
        logger.warning("distributed.requirements: %s dropped (torch is pinned to the hub's version and its build "
                       "is chosen per node)", ", ".join(dropped))
    return lines


def requirements_hash(requirements: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest()


def node_requirements(*, override: Path | None,
                      distributions: Iterable[importlib.metadata.Distribution] | None = None
                      ) -> tuple[list[str], str]:
    """(requirement lines, their hash): ``override``'s lines, else ``hub_freeze()``."""
    reqs = read_requirements_file(override) if override is not None else hub_freeze(distributions)
    return reqs, requirements_hash(reqs)

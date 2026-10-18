"""Python requirements of node environments (SP5 spec 2.4): the hub environment's exact versions.

Default list: every installed distribution as ``name==version`` (local version labels such as ``+cu130``
dropped), without the packages whose build depends on the node's hardware (``torch``, ``triton``,
``nvidia-*``, ``cuda-*``, ``pytorch-triton*``), without ``colosseum`` and without distributions installed from
local paths (``file:`` URLs in ``direct_url.json``). Only editable local-path installs travel in the code
bundle; a non-editable one (``pip install ./mylib``, a local wheel) is not shipped either (a compiled extension
may not fit the node's platform): one WARNING names it. ``distributed.requirements``
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


def _local_install(dist: importlib.metadata.Distribution) -> str | None:
    """``"editable"`` / ``"copy"`` for a distribution installed from a local path (``file:`` URL), else None."""
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(info, dict) or not str(info.get("url", "")).startswith("file:"):
        return None
    dir_info = info.get("dir_info")
    return "editable" if isinstance(dir_info, dict) and dir_info.get("editable") else "copy"


def hub_freeze(distributions: Iterable[importlib.metadata.Distribution] | None = None) -> list[str]:
    """``name==version`` of the hub environment's distributions without the exclusions, sorted by name; the first
    distribution of a name wins (``importlib.metadata`` order = ``sys.path`` order)."""
    seen: dict[str, str] = {}
    for dist in distributions if distributions is not None else importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        canon = canonical_name(name)
        if canon in seen or excluded(canon):
            seen.setdefault(canon, "")
            continue
        local = _local_install(dist)
        if local is not None:
            seen[canon] = ""
            if local == "copy":
                logger.warning("%s %s was installed from a local path without -e: it is neither pinned for the "
                               "nodes nor in the code bundle, so nodes will miss it unless it is installed editable "
                               "(pip install -e) or listed in distributed.requirements", name, dist.version)
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

"""Rating entities (SP4 spec block 2).

An entity is what a Bradley-Terry rating is fitted for:

- a trainable agent's seat counts for ``<agent>@v<B>``, where ``B`` is the seat's rating version: ``N`` for a
  snapshot ``ckpt_v<N>``, and for the latest weights the version bucket
  ``floor(version_start / checkpoint.interval) * checkpoint.interval`` (the snapshot saved at the start of
  that stretch of training; bucket ``v0`` has no snapshot: it is a rating entity, not a snapshot);
- a scripted or frozen agent (network ``fixed``) is one entity named by its agent id;
- a seat whose rating version is unknown (``latest`` without ``version_start``: eval, tournament players whose
  agent id already is the entity name) is the entity named by its agent id.

The lineage of an entity is its name without the ``@v<N>`` suffix (``main@v1200`` -> ``main``,
``b/main@v3`` -> ``b/main``); averages over a field are taken per lineage, not per snapshot.
"""

from __future__ import annotations

import re

from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID

EntityKey = str

_SNAPSHOT_RE = re.compile(r"ckpt_v(\d+)")
_VERSION_SUFFIX_RE = re.compile(r"@v(\d+)$")


def snapshot_version(network_id: str) -> int | None:
    """``"ckpt_v1200"`` -> 1200; any other network id (``latest``, ``fixed``) -> None."""
    match = _SNAPSHOT_RE.fullmatch(network_id)
    return None if match is None else int(match.group(1))


def rating_version(network_id: str, version_start: int | None, interval: int) -> int | None:
    """The version a seat's match counts for (module docstring): ``ckpt_v<N>`` -> N; ``latest`` -> the
    bucket of ``version_start`` (None without it); ``fixed`` -> None. ``interval`` >= 1."""
    if network_id == FIXED_NETWORK_ID:
        return None
    version = snapshot_version(network_id)
    if version is not None:
        return version
    if network_id != LATEST_NETWORK_ID:
        raise ValueError(f"unknown network id {network_id!r} (expected 'latest', 'fixed' or 'ckpt_v<N>')")
    if version_start is None:
        return None
    if interval < 1:
        raise ValueError(f"checkpoint interval must be >= 1, got {interval}")
    return (int(version_start) // int(interval)) * int(interval)


def entity_key(agent_id: str, version: int | None) -> EntityKey:
    """``("main", 1200)`` -> ``"main@v1200"``; ``(agent, None)`` -> ``agent``."""
    return agent_id if version is None else f"{agent_id}@v{int(version)}"


def lineage(entity: EntityKey) -> str:
    """The entity without its ``@v<N>`` suffix (module docstring)."""
    return _VERSION_SUFFIX_RE.sub("", entity)


def entity_version(entity: EntityKey) -> int | None:
    """``"main@v1200"`` -> 1200; an entity without a version suffix -> None."""
    match = _VERSION_SUFFIX_RE.search(entity)
    return None if match is None else int(match.group(1))

"""Inter-process helpers: numpy <-> torch conversion and payload checks.

Rule (spec block 2): nothing that crosses a process boundary may contain a
``torch.Tensor``. Tensors put on an ``mp.Queue`` (or passed as ``Process``
arguments under spawn) are shared through file descriptors served by the
sending process, so the receiver crashes when the sender has already exited
(R6-02). Everything goes through numpy payloads.

This module is the single place that converts between torch and numpy:
``TrajectoryChunk.to_payload``, ``WeightPayload``, ``state_dict_to_numpy``,
``colosseum.networks.state.state_to_numpy`` and the learner / launcher all use
the helpers below.

Conversion rules:
- torch -> numpy always copies (the payload never aliases live tensors such as
  model parameters, which the optimizer keeps updating while an ``mp.Queue``
  feeder thread may still be pickling the payload). ``bfloat16`` (and other
  reduced-precision float dtypes numpy lacks) is upcast to ``float32``, so a
  round trip returns ``float32``, not the original dtype.
- numpy -> torch shares memory with a writable, C-contiguous array (zero copy:
  received payload arrays are owned by the receiver) and copies otherwise, so
  read-only arrays (e.g. from ``np.frombuffer``) never trigger torch's
  non-writable-array warning.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

# torch float dtypes without a numpy equivalent; upcast to float32 on export.
_UPCAST_DTYPES = {torch.bfloat16}
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if hasattr(torch, _name):
        _UPCAST_DTYPES.add(getattr(torch, _name))


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detached CPU numpy copy of ``t`` (bfloat16/float8 -> float32)."""
    t = t.detach().cpu()
    if t.dtype in _UPCAST_DTYPES:
        t = t.float()
    return t.numpy().copy()


def numpy_to_tensor(a: np.ndarray) -> torch.Tensor:
    """CPU tensor from ``a``: shares memory if writable and C-contiguous, else copies."""
    a = np.asarray(a)
    if not a.flags.writeable or not a.flags.c_contiguous:
        a = np.array(a, copy=True, order="C")
    return torch.from_numpy(a)


def to_numpy_tree(obj: Any) -> Any:
    """Copy of ``obj`` with every ``torch.Tensor`` replaced by a numpy array.

    Recurses into dict / list / tuple (namedtuples keep their type); other
    leaves are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return tensor_to_numpy(obj)
    if isinstance(obj, dict):
        return {k: to_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_numpy_tree(v) for v in obj]
    if _is_namedtuple(obj):
        return type(obj)(*(to_numpy_tree(v) for v in obj))
    if isinstance(obj, tuple):
        return tuple(to_numpy_tree(v) for v in obj)
    return obj


def from_numpy_tree(obj: Any) -> Any:
    """Inverse of :func:`to_numpy_tree`: every ``np.ndarray`` becomes a CPU tensor."""
    if isinstance(obj, np.ndarray):
        return numpy_to_tensor(obj)
    if isinstance(obj, dict):
        return {k: from_numpy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_numpy_tree(v) for v in obj]
    if _is_namedtuple(obj):
        return type(obj)(*(from_numpy_tree(v) for v in obj))
    if isinstance(obj, tuple):
        return tuple(from_numpy_tree(v) for v in obj)
    return obj


def find_tensor(obj: Any, path: str = "item") -> str | None:
    """Return the path of the first ``torch.Tensor`` inside ``obj``, or None.

    Recurses into dict / list / tuple / set and dataclass instances.
    """
    if isinstance(obj, torch.Tensor):
        return path
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = find_tensor(v, f"{path}[{k!r}]")
            if found is not None:
                return found
        return None
    if isinstance(obj, list | tuple | set | frozenset):
        for i, v in enumerate(obj):
            found = find_tensor(v, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            found = find_tensor(getattr(obj, f.name), f"{path}.{f.name}")
            if found is not None:
                return found
    return None


def assert_no_tensors(obj: Any, what: str = "item") -> None:
    """Raise TypeError if ``obj`` contains a ``torch.Tensor`` anywhere."""
    found = find_tensor(obj, what)
    if found is not None:
        raise TypeError(
            f"torch.Tensor at {found}: inter-process data must be numpy/primitives "
            f"(use to_payload() / WeightPayload.from_model() / to_numpy_tree())"
        )

"""Opaque model-state pytrees.

A ``State`` is ``None``, a ``torch.Tensor``, or a (possibly nested) ``tuple``,
``list`` or ``dict`` of those. Every tensor leaf has the batch dimension first
(``dim 0``). Models decide what their state contains; the framework only
slices, concatenates, resets, moves and serializes it with the helpers below.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

State = Any  # None | Tensor | tuple | list | dict[str, State]; tensor leaves are batch-first


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def tree_map(fn: Callable[[Tensor], Tensor], state: State) -> State:
    """Apply ``fn`` to every tensor leaf; keep the container structure."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return fn(state)
    if _is_namedtuple(state):
        return type(state)(*(tree_map(fn, s) for s in state))
    if isinstance(state, tuple):
        return tuple(tree_map(fn, s) for s in state)
    if isinstance(state, list):
        return [tree_map(fn, s) for s in state]
    if isinstance(state, dict):
        return {k: tree_map(fn, v) for k, v in state.items()}
    raise TypeError(f"Unsupported state node type: {type(state).__name__}")


def _map2(fn: Callable[[Tensor, Tensor], Tensor], a: State, b: State) -> State:
    """``tree_map`` over two states with identical structure."""
    if a is None and b is None:
        return None
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return fn(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        mapped = [_map2(fn, x, y) for x, y in zip(a, b)]
        return type(a)(*mapped) if _is_namedtuple(a) else tuple(mapped)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_map2(fn, x, y) for x, y in zip(a, b)]
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return {k: _map2(fn, a[k], b[k]) for k in a}
    raise ValueError(f"State structures differ: {type(a).__name__} vs {type(b).__name__}")


def tree_leaves(state: State) -> list[Tensor]:
    """All tensor leaves in deterministic (insertion) order."""
    leaves: list[Tensor] = []
    tree_map(lambda t: leaves.append(t) or t, state)
    return leaves


def batch_size_of(state: State) -> int | None:
    """Common batch size of all leaves, or None if the state has no leaves."""
    leaves = tree_leaves(state)
    if not leaves:
        return None
    sizes = set()
    for leaf in leaves:
        if leaf.dim() == 0:
            raise ValueError("State leaves must have a batch dimension (got a 0-dim tensor)")
        sizes.add(int(leaf.shape[0]))
    if len(sizes) != 1:
        raise ValueError(f"State leaves disagree on batch size: {sorted(sizes)}")
    return sizes.pop()


def slice_batch(state: State, idx: int | Sequence[int] | Tensor) -> State:
    """Select batch rows.

    A single integer index (any ``numbers.Integral``, e.g. ``np.int64``, or a
    0-dim integer tensor) keeps the batch dim: the result has batch 1. A
    sequence of integers, an integer index tensor or a 1-D bool mask tensor
    selects several rows. A single ``bool`` (Python, ``np.bool_`` or 0-dim
    tensor) is rejected (it would silently mean index 0 or 1), and so are
    non-integer indices (floats, float/complex tensors): they are never
    truncated to integers.
    """
    if isinstance(idx, np.ndarray):
        idx = torch.as_tensor(idx)
    if isinstance(idx, bool | np.bool_) or (isinstance(idx, Tensor) and idx.dim() == 0
                                            and idx.dtype == torch.bool):
        raise TypeError("slice_batch: a single bool is not a valid batch index")
    if isinstance(idx, Tensor):
        if idx.dtype.is_floating_point or idx.dtype.is_complex:
            raise TypeError(f"slice_batch: index tensor must have an integer or bool dtype, got {idx.dtype}")
        if idx.dim() == 0:
            idx = int(idx)
    if isinstance(idx, numbers.Integral):
        i = int(idx)
        return tree_map(lambda t: t[i].unsqueeze(0), state)
    if isinstance(idx, Tensor):
        index = idx if idx.dtype == torch.bool else idx.long()
    elif isinstance(idx, Sequence) and not isinstance(idx, str | bytes):
        bad = [i for i in idx if isinstance(i, bool | np.bool_) or not isinstance(i, numbers.Integral)]
        if bad:
            raise TypeError(f"slice_batch: index sequence must contain integers only, got {bad!r}")
        index = torch.as_tensor([int(i) for i in idx], dtype=torch.long)
    else:
        raise TypeError(f"slice_batch: batch index must be an integer, an integer sequence or an "
                        f"index tensor, got {type(idx).__name__}")
    return tree_map(lambda t: t[index.to(t.device)], state)


def cat_batch(states: Sequence[State]) -> State:
    """Concatenate states along the batch dim. All-``None`` gives ``None``."""
    states = list(states)
    if not states:
        raise ValueError("cat_batch needs at least one state")
    first = states[0]
    if first is None:
        if any(s is not None for s in states):
            raise ValueError("Cannot concatenate None with non-None states")
        return None
    if isinstance(first, Tensor):
        if not all(isinstance(s, Tensor) for s in states):
            raise ValueError("State structures differ: expected tensors")
        return torch.cat(states, dim=0)
    if isinstance(first, tuple | list | dict):
        if not all(type(s) is type(first) for s in states):
            names = sorted({type(s).__name__ for s in states})
            raise ValueError(f"State structures differ: node types {names}")
    if isinstance(first, tuple):
        if not all(len(s) == len(first) for s in states):
            raise ValueError("State structures differ: tuple length mismatch")
        parts = [cat_batch([s[i] for s in states]) for i in range(len(first))]
        return type(first)(*parts) if _is_namedtuple(first) else tuple(parts)
    if isinstance(first, list):
        if not all(len(s) == len(first) for s in states):
            raise ValueError("State structures differ: list length mismatch")
        return [cat_batch([s[i] for s in states]) for i in range(len(first))]
    if isinstance(first, dict):
        if not all(s.keys() == first.keys() for s in states):
            raise ValueError("State structures differ: dict keys mismatch")
        return {k: cat_batch([s[k] for s in states]) for k in first}
    raise TypeError(f"Unsupported state node type: {type(first).__name__}")


def where_done(done: Tensor, reset: State, state: State) -> State:
    """Row-wise select: rows where ``done`` is True come from ``reset``, others from ``state``.

    ``done`` is a ``[B]`` bool tensor; ``reset`` and ``state`` share structure and batch size.
    """
    batch = batch_size_of(state)
    if batch is not None and tuple(done.shape) != (batch,):
        raise ValueError(f"where_done: done must have shape ({batch},), got {tuple(done.shape)}")
    def _select(r: Tensor, s: Tensor) -> Tensor:
        mask = done.to(device=s.device, dtype=torch.bool).view(-1, *([1] * (s.dim() - 1)))
        return torch.where(mask, r.to(dtype=s.dtype, device=s.device), s)

    return _map2(_select, reset, state)


def state_to_numpy(state: State) -> Any:
    """Same structure with ``np.ndarray`` leaves (detached CPU copies)."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach().cpu().numpy().copy()
    if _is_namedtuple(state):
        return type(state)(*(state_to_numpy(s) for s in state))
    if isinstance(state, tuple):
        return tuple(state_to_numpy(s) for s in state)
    if isinstance(state, list):
        return [state_to_numpy(s) for s in state]
    if isinstance(state, dict):
        return {k: state_to_numpy(v) for k, v in state.items()}
    raise TypeError(f"Unsupported state node type: {type(state).__name__}")


def state_from_numpy(obj: Any, device: str | torch.device = "cpu") -> State:
    """Inverse of :func:`state_to_numpy`."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj)).to(device)
    if _is_namedtuple(obj):
        return type(obj)(*(state_from_numpy(o, device) for o in obj))
    if isinstance(obj, tuple):
        return tuple(state_from_numpy(o, device) for o in obj)
    if isinstance(obj, list):
        return [state_from_numpy(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: state_from_numpy(v, device) for k, v in obj.items()}
    raise TypeError(f"Unsupported state payload node type: {type(obj).__name__}")


def state_to(state: State, device: str | torch.device) -> State:
    """Move every leaf to ``device``."""
    return tree_map(lambda t: t.to(device), state)

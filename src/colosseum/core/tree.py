"""Tree utilities for observations, actions, masks and global state (SP2 spec block 2).

A ``Tree`` is a leaf (``np.ndarray``, ``torch.Tensor``, a number or ``None``) or a
``dict[str, Tree]``. Dict insertion order is significant: it is the natural order of
the gymnasium space the tree comes from (``Dict.spaces`` order), and every function
here walks dicts in that order. Model states keep their own helpers in
:mod:`colosseum.networks.state`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from colosseum.core.ipc import numpy_to_tensor, tensor_to_numpy

Tree = Any  # np.ndarray | torch.Tensor | number | None | dict[str, Tree]


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _map(fn: Callable[..., Any], path: tuple[str, ...], tree: Tree, rest: tuple[Tree, ...]) -> Tree:
    if isinstance(tree, dict):
        for other in rest:
            if not isinstance(other, dict) or other.keys() != tree.keys():
                got = sorted(other) if isinstance(other, dict) else type(other).__name__
                raise ValueError(f"tree structures differ at {_fmt(path)}: keys {list(tree)} vs {got}")
        return {k: _map(fn, (*path, k), tree[k], tuple(o[k] for o in rest)) for k in tree}
    for other in rest:
        if isinstance(other, dict):
            raise ValueError(f"tree structures differ at {_fmt(path)}: a leaf vs a dict with keys {list(other)}")
    return fn(tree, *rest)


def tree_map(fn: Callable[..., Any], tree: Tree, *rest: Tree) -> Tree:
    """Apply ``fn(leaf, *rest_leaves)`` to aligned leaves; the result has the structure of ``tree``.

    ``rest`` trees must have the same dict keys at every node (``ValueError`` naming the
    path otherwise); their key order does not matter.
    """
    return _map(fn, (), tree, rest)


def tree_leaves(tree: Tree) -> list[Any]:
    """All leaves, depth-first in dict insertion order."""
    out: list[Any] = []
    tree_map(lambda leaf: out.append(leaf), tree)
    return out


def tree_paths(tree: Tree) -> list[tuple[str, ...]]:
    """Path of every leaf, in :func:`tree_leaves` order; ``()`` for a bare leaf."""
    out: list[tuple[str, ...]] = []

    def walk(node: Tree, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, (*path, k))
        else:
            out.append(path)

    walk(tree, ())
    return out


def tree_get(tree: Tree, path: tuple[str, ...]) -> Any:
    """The subtree at ``path`` (``KeyError`` naming the path if it does not exist)."""
    node = tree
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"no node {_fmt(path[: i + 1])} in the tree")
        node = node[key]
    return node


def _stack_leaves(leaves: Sequence[Any], axis: int) -> Any:
    if all(isinstance(x, torch.Tensor) for x in leaves):
        return torch.stack(list(leaves), dim=axis)
    if any(isinstance(x, torch.Tensor) for x in leaves):
        raise TypeError("tree_stack: cannot stack torch tensors with numpy arrays")
    return np.stack([np.asarray(x) for x in leaves], axis=axis)


def tree_stack(trees: Sequence[Tree], axis: int = 0) -> Tree:
    """Stack aligned leaves of ``trees`` along a new ``axis`` (``torch.stack`` for tensors,
    ``np.stack`` otherwise; numpy dtypes are preserved)."""
    trees = list(trees)
    if not trees:
        raise ValueError("tree_stack needs at least one tree")
    return tree_map(lambda *leaves: _stack_leaves(leaves, axis), trees[0], *trees[1:])


def tree_index(tree: Tree, idx: Any) -> Tree:
    """``leaf[idx]`` for every leaf."""
    return tree_map(lambda leaf: leaf[idx], tree)


def tree_assign(dst: Tree, idx: Any, src: Tree) -> None:
    """In place: ``dst_leaf[idx] = src_leaf`` for every aligned leaf (numpy casts to dst's dtype)."""
    if not isinstance(dst, dict):
        raise TypeError("tree_assign: dst must be a dict of arrays (a bare array cannot be assigned "
                        "through a function argument); index it directly")

    def assign(d: Any, s: Any) -> None:
        d[idx] = s

    tree_map(assign, dst, src)


def tree_to_torch(tree: Tree, device: str | torch.device = "cpu") -> Tree:
    """numpy leaves -> tensors on ``device`` (zero copy on CPU when possible); dtypes preserved.

    Tensor leaves are moved to ``device``; ``None`` leaves stay ``None``.
    """
    def convert(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device)
        return numpy_to_tensor(np.asarray(leaf)).to(device)

    return tree_map(convert, tree)


def tree_to_numpy(tree: Tree) -> Tree:
    """Tensor leaves -> detached CPU numpy copies; numpy leaves are kept; ``None`` stays ``None``."""
    def convert(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return tensor_to_numpy(leaf)
        return np.asarray(leaf)

    return tree_map(convert, tree)


def tree_same_structure(a: Tree, b: Tree) -> bool:
    """True if ``a`` and ``b`` have the same dict keys in the same order at every node."""
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if not isinstance(a, dict):
        return True
    if list(a) != list(b):
        return False
    return all(tree_same_structure(a[k], b[k]) for k in a)

"""``TreeDist``: one distribution per action group of an ``ActionSpec``, and ``make_distribution``.

Deciders: decider 0 is all non-units groups together (their per-decider values are summed),
then each units group in spec order contributes its ``max_units`` deciders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup, ActionSpec
from colosseum.sp2.core.tree import Tree, tree_get
from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist
from colosseum.sp2.networks.dist.units import UnitsDist


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _check_part(group: ActionGroup, part: Distribution) -> None:
    where = f"action group {_fmt(group.path)}"
    if not isinstance(part, Distribution):
        raise TypeError(f"{where}: expected a Distribution, got {type(part).__name__}")
    if group.kind == "units":
        if part.num_deciders != group.units.max_units:
            raise ValueError(f"{where}: Units({group.units.max_units}) needs a distribution with "
                             f"{group.units.max_units} deciders, got {part.num_deciders}")
        return
    if part.num_deciders != 1:
        raise ValueError(f"{where}: a {group.kind} group needs a one-decider distribution, got {part.num_deciders}")
    if group.kind == "discrete" and isinstance(part, CategoricalDist) and part.num_categories != group.nvec[0]:
        raise ValueError(f"{where}: Discrete({group.nvec[0]}) needs {group.nvec[0]} logits, got {part.num_categories}")
    if group.kind == "multi_discrete" and isinstance(part, MultiCategoricalDist) and part.nvec != group.nvec:
        raise ValueError(f"{where}: MultiDiscrete{list(group.nvec)} got a MultiCategoricalDist with nvec "
                         f"{list(part.nvec)}")
    if group.kind == "box" and isinstance(part, DiagGaussianDist) and part.action_dim != group.box_dim:
        raise ValueError(f"{where}: Box({group.box_dim},) got a DiagGaussianDist of dimension {part.action_dim}")
    expected = {"discrete": CategoricalDist, "multi_discrete": MultiCategoricalDist, "box": DiagGaussianDist}
    builtin = (CategoricalDist, MultiCategoricalDist, DiagGaussianDist)
    if isinstance(part, builtin) and not isinstance(part, expected[group.kind]):
        raise ValueError(f"{where}: a {group.kind} group needs a {expected[group.kind].__name__}, "
                         f"got {type(part).__name__}")


class TreeDist(Distribution):
    """Composition of per-group distributions along an ``ActionSpec``."""

    def __init__(self, spec: ActionSpec, parts: Mapping[tuple[str, ...], Distribution]) -> None:
        paths = [g.path for g in spec.groups]
        if set(parts) != set(paths):
            raise ValueError(f"TreeDist: parts for groups {sorted(map(_fmt, parts))}, the action space has "
                             f"{[_fmt(p) for p in paths]}")
        self.spec = spec
        self.parts: dict[tuple[str, ...], Distribution] = {p: parts[p] for p in paths}
        for group in spec.groups:
            _check_part(group, self.parts[group.path])
        sizes = {part.batch_size for part in self.parts.values()}
        if len(sizes) != 1:
            raise ValueError(f"TreeDist: parts disagree on the batch size: {sorted(sizes)}")
        self._flat = [g for g in spec.groups if g.kind != "units"]
        self._units = [g for g in spec.groups if g.kind == "units"]

    @property
    def batch_size(self) -> int:
        return next(iter(self.parts.values())).batch_size

    @property
    def num_deciders(self) -> int:
        return self.spec.num_deciders

    def _action(self, actions: Tree, group: ActionGroup) -> Any:
        return actions if not self.spec.is_dict else tree_get(actions, group.path)

    def _assemble(self, values: dict[tuple[str, ...], Any]) -> Tree:
        if not self.spec.is_dict:
            return values[self.spec.groups[0].path]
        out: dict = {}
        for path, value in values.items():
            node = out
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value
        return out

    def sample(self) -> Tree:
        return self._assemble({p: part.sample() for p, part in self.parts.items()})

    def mode(self) -> Tree:
        return self._assemble({p: part.mode() for p, part in self.parts.items()})

    def _per_decider(self, fn: Any, actions: Tree) -> Tensor:
        columns: list[Tensor] = []
        if self._flat:
            columns.append(sum(fn(self.parts[g.path], self._action(actions, g)) for g in self._flat))
        columns.extend(fn(self.parts[g.path], self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def unit_log_prob(self, actions: Tree) -> Tensor:
        return self._per_decider(lambda part, a: part.unit_log_prob(a), actions)

    def unit_entropy(self, actions: Tree) -> Tensor:
        return self._per_decider(lambda part, a: part.unit_entropy(a), actions)

    def unit_valid(self, actions: Tree) -> Tensor:
        columns: list[Tensor] = []
        if self._flat:
            device = self.parts[self._flat[0].path].unit_valid(self._action(actions, self._flat[0])).device
            columns.append(torch.ones(self.batch_size, 1, dtype=torch.bool, device=device))
        columns.extend(self.parts[g.path].unit_valid(self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def unit_kl(self, other: Distribution, actions: Tree) -> Tensor:
        if not isinstance(other, TreeDist) or other.spec != self.spec:
            raise TypeError(f"KL between TreeDist and {type(other).__name__} over a different action space")
        columns: list[Tensor] = []
        if self._flat:
            columns.append(sum(self.parts[g.path].unit_kl(other.parts[g.path], self._action(actions, g))
                               for g in self._flat))
        columns.extend(self.parts[g.path].unit_kl(other.parts[g.path], self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def apply_mask(self, mask: Tree | None) -> TreeDist:
        if mask is None:
            return self
        parts = {}
        for group in self.spec.groups:
            part = self.parts[group.path]
            group_mask = self.spec.group_mask(mask, group)
            parts[group.path] = part if group_mask is None else part.apply_mask(group_mask)
        return TreeDist(self.spec, parts)

    @classmethod
    def cat(cls, dists: Sequence[TreeDist]) -> TreeDist:
        spec = dists[0].spec
        return TreeDist(spec, {p: type(part).cat([d.parts[p] for d in dists]) for p, part in dists[0].parts.items()})


def _group_params(params: Tree, spec: ActionSpec, group: ActionGroup) -> Any:
    if not spec.is_dict:
        return params
    try:
        return tree_get(params, group.path)
    except KeyError:
        raise ValueError(f"make_distribution: no parameters for action group {_fmt(group.path)}") from None


def make_distribution(spec: ActionSpec, params: Tree) -> TreeDist:
    """Built-in distribution for ``spec`` from a parameter tree that mirrors the action tree.

    discrete -> logits ``[B, n]``; multi_discrete -> logits ``[B, sum(nvec)]``; box ->
    ``{"mean": [B, d], "log_std": [B, d] | [d]}``; units -> ``UnitsDist`` parameters
    (one entry per component: logits ``[B, U, n]`` or ``{"mean", "log_std"}``).
    A parameter or shape mismatch raises ``ValueError`` naming the action group.
    """
    parts: dict[tuple[str, ...], Distribution] = {}
    for group in spec.groups:
        p = _group_params(params, spec, group)
        try:
            parts[group.path] = _make_part(group, p)
        except (ValueError, TypeError) as e:
            raise ValueError(f"action group {_fmt(group.path)}: {e}") from e
    return TreeDist(spec, parts)


def _make_part(group: ActionGroup, p: Any) -> Distribution:
    if group.kind == "discrete":
        return CategoricalDist(p)
    if group.kind == "multi_discrete":
        return MultiCategoricalDist(p, group.nvec)
    if group.kind == "box":
        if not isinstance(p, Mapping) or set(p) != {"mean", "log_std"}:
            got = sorted(p) if isinstance(p, Mapping) else type(p).__name__
            raise ValueError(f"a box group needs params {{'mean': [B, d], 'log_std': [B, d] | [d]}}, got {got}")
        return DiagGaussianDist(p["mean"], p["log_std"])
    return UnitsDist(group, p)

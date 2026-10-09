"""Observation and action specs, and the one set of mask rules (SP2 spec block 2).

Successor of SP1's ``core/action_spec.py`` (flat float32 codec, removed) and
``core/seat_info.py`` (per-seat mask rules). ``EpisodeTracker`` (worker, eval) and
``validate`` use these rules; nothing else re-implements them.

Observations (``ObsSpec``): ``Box``, ``Discrete``, ``MultiBinary``, ``MultiDiscrete`` or a
nested ``Dict`` of them; leaf dtypes are preserved end to end.

Actions (``ActionSpec``): ``Discrete``, ``MultiDiscrete`` (1-D), ``Box`` (1-D), ``Units``
or a nested ``Dict`` of them. Every non-Dict node is one *group*. Action values: discrete
``int64[]``, multi_discrete ``int64[C]``, box ``float32[d]``, units as documented in
:mod:`colosseum.envs.spaces`; a Dict action is a dict of those.

Mask trees mirror the discrete parts of the action tree; a missing leaf means "all allowed":
discrete ``bool[n]``, multi_discrete ``bool[sum(nvec)]``, box no leaf, units
``{"unit": bool[U], "action": bool[U, sum of discrete component sizes]}``.

Deciders: decider 0 is all non-units groups together (if any), then each units group in
spec order contributes ``max_units`` deciders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium
import numpy as np
import torch

from colosseum.core.errors import EnvContractError
from colosseum.core.tree import Tree
from colosseum.envs.spaces import Units


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _set_path(tree: dict, path: tuple[str, ...], value: Any) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _shape_str(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(s) for s in shape) + "]"


# ---------------------------------------------------------------------------
# Observations (and global state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafSpec:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype


class ObsSpec:
    """Leaves (path, shape, dtype) of an observation or global-state space."""

    def __init__(self, leaves: tuple[LeafSpec, ...], is_dict: bool) -> None:
        self.leaves = leaves
        self.is_dict = is_dict  # False: the value is one bare array (path ())

    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ObsSpec:
        leaves: list[LeafSpec] = []
        cls._collect(space, (), leaves)
        return cls(tuple(leaves), isinstance(space, gymnasium.spaces.Dict))

    @classmethod
    def _collect(cls, space: gymnasium.Space, path: tuple[str, ...], out: list[LeafSpec]) -> None:
        if isinstance(space, gymnasium.spaces.Dict):
            if not space.spaces:
                raise ValueError(f"observation space at {_fmt(path)}: an empty Dict has no leaves")
            for key, sub in space.spaces.items():
                cls._collect(sub, (*path, key), out)
        elif isinstance(space, gymnasium.spaces.Box | gymnasium.spaces.MultiBinary):
            out.append(LeafSpec(path, tuple(int(s) for s in space.shape), np.dtype(space.dtype)))
        elif isinstance(space, gymnasium.spaces.Discrete):
            out.append(LeafSpec(path, (), np.dtype(space.dtype)))
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            out.append(LeafSpec(path, tuple(int(s) for s in space.nvec.shape), np.dtype(space.dtype)))
        else:
            raise TypeError(f"unsupported observation space {type(space).__name__} at {_fmt(path)}: use Box, "
                            f"Discrete, MultiBinary, MultiDiscrete or a Dict of them")

    def allocate(self, leading: tuple[int, ...]) -> Tree:
        """Zeros with shape ``leading + leaf.shape`` per leaf (numpy, dtypes preserved)."""
        leading = tuple(leading)
        if not self.is_dict:
            leaf = self.leaves[0]
            return np.zeros(leading + leaf.shape, dtype=leaf.dtype)
        out: dict = {}
        for leaf in self.leaves:
            _set_path(out, leaf.path, np.zeros(leading + leaf.shape, dtype=leaf.dtype))
        return out

    def check(self, value: Tree, where: str) -> None:
        """Cheap structure and shape check of one value (no leading dims); EnvContractError."""
        if not self.is_dict:
            if isinstance(value, dict):
                raise EnvContractError(f"{where}: expected an array of shape {self.leaves[0].shape}, got a dict "
                                       f"with keys {list(value)}")
            self._check_leaf(self.leaves[0], value, where)
            return
        self._check_keys(value, (), where)
        for leaf in self.leaves:
            node = value
            for key in leaf.path:
                node = node[key]
            self._check_leaf(leaf, node, where)

    def _check_keys(self, value: Any, path: tuple[str, ...], where: str) -> None:
        expected: list[str] = []
        for leaf in self.leaves:
            if leaf.path[: len(path)] == path and len(leaf.path) > len(path):
                key = leaf.path[len(path)]
                if key not in expected:
                    expected.append(key)
        if not expected:
            return  # a leaf
        if not isinstance(value, dict):
            raise EnvContractError(f"{where}: expected a dict with keys {expected} at {_fmt(path)}, "
                                   f"got {type(value).__name__}")
        if set(value) != set(expected):
            missing = [k for k in expected if k not in value]
            extra = [k for k in value if k not in expected]
            raise EnvContractError(f"{where}: keys at {_fmt(path)} do not match the space "
                                   f"(missing {missing}, unexpected {extra})")
        for key in expected:
            self._check_keys(value[key], (*path, key), where)

    @staticmethod
    def _check_leaf(leaf: LeafSpec, value: Any, where: str) -> None:
        shape = np.shape(value)
        if tuple(shape) != leaf.shape:
            raise EnvContractError(f"{where}: leaf {_fmt(leaf.path)} has shape {tuple(shape)}, "
                                   f"expected {leaf.shape}")

    def signature(self) -> str:
        return ";".join(f"{_fmt(leaf.path)}:{leaf.dtype.name}{_shape_str(leaf.shape)}" for leaf in self.leaves)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObsSpec) and self.signature() == other.signature() and self.is_dict == other.is_dict

    def __hash__(self) -> int:
        return hash(self.signature())

    def __repr__(self) -> str:
        return f"ObsSpec({self.signature()})"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionGroup:
    path: tuple[str, ...]                    # () when the action space is not a Dict
    kind: Literal["discrete", "multi_discrete", "box", "units"]
    nvec: tuple[int, ...]                    # discrete: (n,); multi_discrete: nvec; else ()
    box_dim: int                             # box: d; else 0
    units: Units | None                      # units only
    mask_size: int                           # width of the (per-unit) action mask row


def _units_mask_size(units: Units) -> int:
    return sum(c.size for c in units.components if c.kind == "discrete")


def split_unit_actions(units: Units, actions: Any) -> dict[str, Any]:
    """Component name -> values ``[..., U]`` (discrete) / ``[..., U, d]`` (box) of a units action
    with any leading dims (the env format of :mod:`colosseum.envs.spaces`)."""
    if units.per_unit_kind == "dict":
        return {c.name: actions[c.name] for c in units.components}
    if units.per_unit_kind == "multi_discrete":
        return {c.name: actions[..., i] for i, c in enumerate(units.components)}
    return {"0": actions}


def units_component_valid(units: Units, base: dict[str, torch.Tensor],
                          values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Validity ``[..., U]`` per component: ``base[c]`` (the unit is present and, for a discrete
    component, its mask row is non-empty) gated by ``only_if`` in dependency order. The gate of
    ``child <- (parent, values)`` holds iff the parent is valid and the recorded parent value
    (``values[parent]``) is in ``values``. The one implementation of the gate rule (``UnitsDist``,
    ``ActionSpec.first_illegal_action``)."""
    valid: dict[str, torch.Tensor] = {}
    pending = list(units.components)
    while pending:
        for c in list(pending):
            rule = units.only_if.get(c.name)
            if rule is not None and rule[0] not in valid:
                continue  # parent first
            ok = base[c.name]
            if rule is not None:
                parent, allowed = rule
                n = next(p.size for p in units.components if p.name == parent)
                table = torch.zeros(n, dtype=torch.bool, device=ok.device)
                table[sorted(allowed)] = True
                ok = ok & valid[parent] & table[values[parent].long().clamp(0, n - 1)]
            valid[c.name] = ok
            pending.remove(c)
    return valid


class ActionSpec:
    """Groups of an action space in natural order, with mask and decider rules."""

    def __init__(self, groups: tuple[ActionGroup, ...], is_dict: bool) -> None:
        self.groups = groups
        self.is_dict = is_dict
        self.has_units = any(g.kind == "units" for g in groups)
        self.has_masks = any(g.mask_size > 0 or g.kind == "units" for g in groups)
        has_flat = any(g.kind != "units" for g in groups)
        self.num_deciders = sum(g.units.max_units for g in groups if g.kind == "units") + (1 if has_flat else 0)

    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ActionSpec:
        groups: list[ActionGroup] = []
        cls._collect(space, (), groups)
        return cls(tuple(groups), isinstance(space, gymnasium.spaces.Dict))

    @classmethod
    def _collect(cls, space: gymnasium.Space, path: tuple[str, ...], out: list[ActionGroup]) -> None:
        where = f"action space at {_fmt(path)}"
        if isinstance(space, gymnasium.spaces.Dict):
            if not space.spaces:
                raise ValueError(f"{where}: an empty Dict has no actions")
            for key, sub in space.spaces.items():
                cls._collect(sub, (*path, key), out)
        elif isinstance(space, Units):
            out.append(ActionGroup(path, "units", (), 0, space, _units_mask_size(space)))
        elif isinstance(space, gymnasium.spaces.Discrete):
            if int(space.start) != 0:
                raise ValueError(f"{where}: Discrete(start={int(space.start)}) is not supported; use start=0")
            n = int(space.n)
            out.append(ActionGroup(path, "discrete", (n,), 0, None, n))
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            if space.nvec.ndim != 1 or np.any(space.start != 0):
                raise ValueError(f"{where}: MultiDiscrete must be 1-D with start 0")
            nvec = tuple(int(n) for n in space.nvec)
            out.append(ActionGroup(path, "multi_discrete", nvec, 0, None, sum(nvec)))
        elif isinstance(space, gymnasium.spaces.Box):
            if len(space.shape) != 1 or not np.issubdtype(space.dtype, np.floating):
                raise ValueError(f"{where}: Box actions must be 1-D float, got shape {space.shape} {space.dtype}")
            out.append(ActionGroup(path, "box", (), int(space.shape[0]), None, 0))
        else:
            raise TypeError(f"unsupported {where}: {type(space).__name__} (use Discrete, MultiDiscrete, Box, "
                            f"Units or a Dict of them)")

    # ---- allocation ---------------------------------------------------------

    def _assemble(self, values: dict[tuple[str, ...], Any]) -> Tree:
        if not self.is_dict:
            return values[self.groups[0].path]
        out: dict = {}
        for path, value in values.items():
            _set_path(out, path, value)
        return out

    @staticmethod
    def _zeros_for(group: ActionGroup, leading: tuple[int, ...]) -> Any:
        if group.kind == "discrete":
            return np.zeros(leading, dtype=np.int64)
        if group.kind == "multi_discrete":
            return np.zeros(leading + (len(group.nvec),), dtype=np.int64)
        if group.kind == "box":
            return np.zeros(leading + (group.box_dim,), dtype=np.float32)
        units = group.units
        U = units.max_units
        if units.per_unit_kind == "discrete":
            return np.zeros(leading + (U,), dtype=np.int64)
        if units.per_unit_kind == "multi_discrete":
            return np.zeros(leading + (U, len(units.components)), dtype=np.int64)
        if units.per_unit_kind == "box":
            return np.zeros(leading + (U, units.components[0].size), dtype=np.float32)
        return {c.name: (np.zeros(leading + (U,), dtype=np.int64) if c.kind == "discrete"
                         else np.zeros(leading + (U, c.size), dtype=np.float32)) for c in units.components}

    def allocate_actions(self, leading: tuple[int, ...]) -> Tree:
        """Zero actions with shape ``leading + ...`` (int64 discrete, float32 box)."""
        leading = tuple(leading)
        return self._assemble({g.path: self._zeros_for(g, leading) for g in self.groups})

    def _mask_tree(self, leading: tuple[int, ...], unit_value: bool) -> Tree | None:
        if not self.has_masks:
            return None
        values: dict[tuple[str, ...], Any] = {}
        for g in self.groups:
            if g.kind == "units":
                U = g.units.max_units
                values[g.path] = {"unit": np.full(leading + (U,), unit_value, dtype=bool),
                                  "action": np.ones(leading + (U, g.mask_size), dtype=bool)}
            elif g.mask_size > 0:
                values[g.path] = np.ones(leading + (g.mask_size,), dtype=bool)
        return self._assemble(values)

    def full_mask(self, leading: tuple[int, ...] = ()) -> Tree | None:
        """Everything allowed (units: ``unit=True``); None if the action space has no masks."""
        return self._mask_tree(tuple(leading), True)

    def boot_mask(self) -> Tree | None:
        """The mask of ``boot`` / ``pad`` slots: everything allowed, units ``unit=False``."""
        return self._mask_tree((), False)

    # ---- mask rules -----------------------------------------------------------

    def _raw_group_mask(self, raw: Any, group: ActionGroup, where: str) -> Any:
        if not self.is_dict:
            return raw
        node = raw
        for i, key in enumerate(group.path):
            if not isinstance(node, dict):
                raise EnvContractError(f"{where}: action mask at {_fmt(group.path[:i])} must be a dict, "
                                       f"got {type(node).__name__}")
            if key not in node:
                return None
            node = node[key]
        return node

    def _check_unknown_keys(self, raw: Any, where: str) -> None:
        masked = [g.path for g in self.groups if g.mask_size > 0 or g.kind == "units"]

        def walk(node: Any, path: tuple[str, ...]) -> None:
            if path in masked:
                return
            if not isinstance(node, dict):
                raise EnvContractError(f"{where}: action mask at {_fmt(path)} must be a dict, "
                                       f"got {type(node).__name__}")
            for key, sub in node.items():
                child = (*path, key)
                if not any(m[: len(child)] == child for m in masked):
                    raise EnvContractError(f"{where}: action mask has an unexpected key {_fmt(child)} "
                                           f"(masked groups: {[_fmt(m) for m in masked]})")
                walk(sub, child)

        walk(raw, ())

    @staticmethod
    def _bool_leaf(value: Any, shape: tuple[int, ...], what: str, where: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.bool_:
            raise EnvContractError(f"{where}: {what} must be a bool array, got dtype {arr.dtype}")
        if arr.shape != shape:
            raise EnvContractError(f"{where}: {what} has shape {arr.shape}, expected {shape}")
        return arr

    def normalize_mask(self, raw: Tree | None, where: str) -> Tree | None:
        """A fresh, complete mask tree from an env mask (missing leaves -> all allowed).

        Checks structure, shapes and the bool dtype; EnvContractError otherwise.
        """
        if not self.has_masks:
            if raw is not None:
                raise EnvContractError(f"{where}: the action space has no discrete parts, so it takes no "
                                       f"action mask")
            return None
        out = self.full_mask()
        if raw is None:
            return out
        if self.is_dict:
            self._check_unknown_keys(raw, where)
        for g in self.groups:
            if g.mask_size == 0 and g.kind != "units":
                continue
            part = self._raw_group_mask(raw, g, where)
            if part is None:
                continue
            what = f"action mask {_fmt(g.path)}"
            dst = self.group_mask(out, g)
            if g.kind == "units":
                if not isinstance(part, dict) or not set(part) <= {"unit", "action"}:
                    got = list(part) if isinstance(part, dict) else type(part).__name__
                    raise EnvContractError(f"{where}: {what} of a Units group must be a dict with keys "
                                           f"'unit' and/or 'action', got {got}")
                U = g.units.max_units
                if part.get("unit") is not None:
                    dst["unit"][...] = self._bool_leaf(part["unit"], (U,), f"{what}/unit", where)
                if part.get("action") is not None:
                    dst["action"][...] = self._bool_leaf(part["action"], (U, g.mask_size), f"{what}/action", where)
            else:
                dst[...] = self._bool_leaf(part, (g.mask_size,), what, where)
        return out

    @staticmethod
    def _get(tree: Tree, path: tuple[str, ...]) -> Any:
        node = tree
        for key in path:
            node = node[key]
        return node

    def group_mask(self, mask: Tree | None, group: ActionGroup) -> Any:
        """The part of a normalized mask tree that belongs to ``group`` (None for box / no mask)."""
        if mask is None or (group.mask_size == 0 and group.kind != "units"):
            return None
        return mask if not self.is_dict else self._get(mask, group.path)

    def check_acting_mask(self, mask: Tree | None, where: str) -> None:
        """Empty-row rule for an acting seat: a non-units discrete group (or one sub-action of a
        multi_discrete group) without a legal action is an EnvContractError. Units groups may
        have empty rows (those deciders are invalid)."""
        if mask is None:
            return
        for g in self.groups:
            if g.kind not in ("discrete", "multi_discrete"):
                continue
            row = self.group_mask(mask, g)
            offset = 0
            for i, n in enumerate(g.nvec):
                if not row[offset:offset + n].any():
                    part = "" if g.kind == "discrete" else f" (sub-action {i})"
                    raise EnvContractError(f"{where}: action mask {_fmt(g.path)}{part} has no legal action "
                                           f"for an acting seat")
                offset += n

    def first_illegal_action(self, actions: Tree, mask: Tree | None) -> tuple[int, str] | None:
        """The first decision whose recorded action a normalized mask forbids, as ``(index, reason)``,
        or None. ``actions`` and ``mask`` are torch trees with a leading ``[N]``; discrete values
        must be in range.

        - discrete / each multi_discrete sub-action: the row must be non-empty (the acting-seat
          rule of :meth:`check_acting_mask`) and allow the recorded value;
        - units: every valid discrete component of a present unit (non-empty row, ``only_if`` gate
          holding for the recorded parent value) must be allowed by its row; invalid ones are free.
        """
        if mask is None:
            return None
        found: list[tuple[int, str]] = []

        def first(bad: torch.Tensor) -> list[int] | None:
            """Index of the first True of ``bad`` ([N] or [N, U]; row-major: smallest decision first)."""
            return [int(x) for x in bad.nonzero()[0]] if bad.any() else None

        for g in self.groups:
            if g.kind == "box":
                continue
            row = self.group_mask(mask, g)
            a = actions if not self.is_dict else self._get(actions, g.path)
            w = _fmt(g.path)
            if g.kind in ("discrete", "multi_discrete"):
                offset = 0
                for k, n in enumerate(g.nvec):
                    seg = row[..., offset:offset + n]
                    value = (a if g.kind == "discrete" else a[..., k]).long()
                    part = "" if g.kind == "discrete" else f" (sub-action {k})"
                    empty = ~seg.any(dim=-1)
                    legal = seg.gather(-1, value.unsqueeze(-1)).squeeze(-1)
                    if (at := first(empty)) is not None:
                        found.append((at[0], f"action mask {w}{part} has no legal action"))
                    if (at := first(~legal & ~empty)) is not None:
                        found.append((at[0], f"action {int(value[at[0]])} at {w}{part} is illegal under its "
                                             f"action mask"))
                    offset += n
                continue
            units = g.units
            comps = split_unit_actions(units, a)
            rows: dict[str, torch.Tensor] = {}
            base: dict[str, torch.Tensor] = {}
            offset = 0
            for c in units.components:
                base[c.name] = row["unit"]
                if c.kind == "discrete":
                    rows[c.name] = row["action"][..., offset:offset + c.size]
                    base[c.name] = row["unit"] & rows[c.name].any(dim=-1)
                    offset += c.size
            valid = units_component_valid(units, base, comps)
            for name, r in rows.items():
                value = comps[name].long()
                legal = r.gather(-1, value.unsqueeze(-1)).squeeze(-1)
                if (at := first(valid[name] & ~legal)) is not None:
                    i, u = at
                    found.append((i, f"unit {u} component {name!r}: action {int(value[i, u])} at {w} is illegal "
                                     f"under its action mask"))
        return min(found, key=lambda item: item[0]) if found else None

    # ---- identity -------------------------------------------------------------

    def signature(self) -> str:
        parts = []
        for g in self.groups:
            if g.kind == "discrete":
                body = f"discrete({g.nvec[0]})"
            elif g.kind == "multi_discrete":
                body = f"multi_discrete({','.join(map(str, g.nvec))})"
            elif g.kind == "box":
                body = f"box({g.box_dim})"
            else:
                comps = ",".join(f"{c.name}:{c.kind}({c.size})" for c in g.units.components)
                rules = ",".join(f"{k}<-{p}{sorted(v)}" for k, (p, v) in sorted(g.units.only_if.items()))
                body = f"units({g.units.max_units},{g.units.per_unit_kind},[{comps}],[{rules}])"
            parts.append(f"{_fmt(g.path)}:{body}")
        return ";".join(parts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActionSpec) and self.signature() == other.signature() and self.is_dict == other.is_dict

    def __hash__(self) -> int:
        return hash(self.signature())

    def __repr__(self) -> str:
        return f"ActionSpec({self.signature()})"

"""``Units``: an action space of up to ``max_units`` units with the same per-unit action (SP2 block 2).

The action of a ``Units`` group is an array (or a dict of arrays) with a leading unit
dimension ``[U]``:

- ``per_unit = Discrete(A)``: ``int64[U]``;
- ``per_unit = MultiDiscrete([A1..Ac])``: ``int64[U, C]``;
- ``per_unit = Box(d,)``: ``float32[U, d]``;
- ``per_unit = Dict(...)`` of ``Discrete`` / 1-D ``Box``: ``{name: int64[U] | float32[U, d]}``.

The per-unit action is a list of named components in natural order: ``"0"`` for
``Discrete`` / ``Box``, ``"0".."C-1"`` for ``MultiDiscrete``, and ``Dict.spaces`` order for
``Dict``. Note that ``gymnasium.spaces.Dict`` built from a plain ``dict`` sorts its keys;
pass a list of ``(key, space)`` pairs to keep your own order. The order fixes the layout of
the ``action`` mask: ``{"unit": bool[U], "action": bool[U, sum of discrete sizes]}``.

``only_if={child: (parent, values)}``: the child component counts (log-prob, entropy, KL,
loss) only where the chosen value of ``parent`` (a discrete component of the same unit) is
in ``values``. Unit slot indices are assigned by the env; the framework does not link slots
across steps.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium
import numpy as np


@dataclass(frozen=True)
class UnitComponent:
    """One named component of the per-unit action."""

    name: str
    kind: Literal["discrete", "box"]
    size: int  # categories (discrete) or dimension (box)


def _discrete_size(space: gymnasium.spaces.Discrete, what: str) -> int:
    if int(space.start) != 0:
        raise ValueError(f"{what}: Discrete(start={int(space.start)}) is not supported; use start=0")
    return int(space.n)


def _box_size(space: gymnasium.spaces.Box, what: str) -> int:
    if len(space.shape) != 1:
        raise ValueError(f"{what}: Box must be 1-D (shape (d,)), got shape {space.shape}")
    if not np.issubdtype(space.dtype, np.floating):
        raise ValueError(f"{what}: Box must have a float dtype, got {space.dtype}")
    return int(space.shape[0])


class Units(gymnasium.spaces.Space):
    """Per-unit actions for up to ``max_units`` units (see the module docstring)."""

    def __init__(
        self,
        max_units: int,
        per_unit: gymnasium.spaces.Discrete | gymnasium.spaces.MultiDiscrete
        | gymnasium.spaces.Box | gymnasium.spaces.Dict,
        only_if: Mapping[str, tuple[str, Collection[int]]] | None = None,
        seed: int | None = None,
    ) -> None:
        if int(max_units) < 1:
            raise ValueError(f"Units: max_units must be >= 1, got {max_units}")
        self.max_units = int(max_units)
        self.per_unit = per_unit
        components: list[UnitComponent] = []
        self._boxes: dict[str, gymnasium.spaces.Box] = {}
        if isinstance(per_unit, gymnasium.spaces.Discrete):
            self.per_unit_kind: Literal["discrete", "multi_discrete", "box", "dict"] = "discrete"
            components.append(UnitComponent("0", "discrete", _discrete_size(per_unit, "Units per_unit")))
        elif isinstance(per_unit, gymnasium.spaces.MultiDiscrete):
            if per_unit.nvec.ndim != 1:
                raise ValueError(f"Units per_unit: MultiDiscrete must be 1-D, got nvec shape {per_unit.nvec.shape}")
            if np.any(per_unit.start != 0):
                raise ValueError("Units per_unit: MultiDiscrete with a non-zero start is not supported")
            self.per_unit_kind = "multi_discrete"
            components.extend(UnitComponent(str(i), "discrete", int(n)) for i, n in enumerate(per_unit.nvec))
        elif isinstance(per_unit, gymnasium.spaces.Box):
            self.per_unit_kind = "box"
            components.append(UnitComponent("0", "box", _box_size(per_unit, "Units per_unit")))
            self._boxes["0"] = per_unit
        elif isinstance(per_unit, gymnasium.spaces.Dict):
            if not per_unit.spaces:
                raise ValueError("Units per_unit: an empty Dict has no components")
            self.per_unit_kind = "dict"
            for name, sub in per_unit.spaces.items():
                what = f"Units per_unit[{name!r}]"
                if isinstance(sub, gymnasium.spaces.Discrete):
                    components.append(UnitComponent(name, "discrete", _discrete_size(sub, what)))
                elif isinstance(sub, gymnasium.spaces.Box):
                    components.append(UnitComponent(name, "box", _box_size(sub, what)))
                    self._boxes[name] = sub
                else:
                    raise ValueError(f"{what}: Dict values may only be Discrete or 1-D Box, "
                                     f"got {type(sub).__name__}")
        else:
            raise ValueError(f"Units per_unit must be Discrete, MultiDiscrete, Box or Dict, "
                             f"got {type(per_unit).__name__}")
        self.components: tuple[UnitComponent, ...] = tuple(components)
        self.only_if: dict[str, tuple[str, frozenset[int]]] = self._check_only_if(only_if or {})
        super().__init__(shape=None, dtype=None, seed=seed)

    # ------------------------------------------------------------------
    # Construction checks
    # ------------------------------------------------------------------

    def _check_only_if(self, only_if: Mapping[str, tuple[str, Collection[int]]]) -> dict:
        by_name = {c.name: c for c in self.components}
        out: dict[str, tuple[str, frozenset[int]]] = {}
        for child, rule in only_if.items():
            if child not in by_name:
                raise ValueError(f"Units only_if: unknown component {child!r} (components: {list(by_name)})")
            try:
                parent, values = rule
            except (TypeError, ValueError):
                raise ValueError(f"Units only_if[{child!r}] must be (parent, values), got {rule!r}") from None
            if parent not in by_name:
                raise ValueError(f"Units only_if[{child!r}]: unknown parent {parent!r} (components: {list(by_name)})")
            if parent == child:
                raise ValueError(f"Units only_if[{child!r}]: a component cannot depend on itself")
            if by_name[parent].kind != "discrete":
                raise ValueError(f"Units only_if[{child!r}]: parent {parent!r} must be a discrete component")
            allowed = frozenset(int(v) for v in values)
            if not allowed:
                raise ValueError(f"Units only_if[{child!r}]: the set of parent values is empty")
            n = by_name[parent].size
            bad = sorted(v for v in allowed if not 0 <= v < n)
            if bad:
                raise ValueError(f"Units only_if[{child!r}]: values {bad} are outside parent {parent!r} range [0, {n})")
            out[child] = (parent, allowed)
        for start in out:  # every chain child -> parent -> ... must end: no cycles
            seen = {start}
            node = start
            while node in out:
                node = out[node][0]
                if node in seen:
                    raise ValueError(f"Units only_if: cycle through component {node!r}")
                seen.add(node)
        return out

    # ------------------------------------------------------------------
    # gymnasium.Space API
    # ------------------------------------------------------------------

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def _component_values(self, x: Any) -> dict[str, np.ndarray] | None:
        """``x`` split into component arrays ``[U]`` / ``[U, d]``, or None if the structure is wrong."""
        U = self.max_units
        if self.per_unit_kind == "dict":
            if not isinstance(x, Mapping) or set(x) != {c.name for c in self.components}:
                return None
            return {c.name: np.asarray(x[c.name]) for c in self.components}
        arr = np.asarray(x)
        if self.per_unit_kind == "discrete":
            return {"0": arr} if arr.shape == (U,) else None
        if self.per_unit_kind == "multi_discrete":
            if arr.shape != (U, len(self.components)):
                return None
            return {c.name: arr[:, i] for i, c in enumerate(self.components)}
        return {"0": arr}  # box

    def contains(self, x: Any) -> bool:
        values = self._component_values(x)
        if values is None:
            return False
        U = self.max_units
        for c in self.components:
            v = values[c.name]
            if c.kind == "discrete":
                if v.shape != (U,) or not np.issubdtype(v.dtype, np.integer):
                    return False
                if np.any(v < 0) or np.any(v >= c.size):
                    return False
            else:
                box = self._boxes[c.name]
                if v.shape != (U, c.size) or not np.issubdtype(v.dtype, np.floating):
                    return False
                if np.any(v < box.low) or np.any(v > box.high):
                    return False
        return True

    def sample(self, mask: Any = None, probability: Any = None) -> Any:
        """A random action. ``mask`` (``{"unit", "action"}``, keys optional) restricts the discrete
        components to legal values; units with ``unit=False`` and empty rows give 0."""
        if probability is not None:
            raise NotImplementedError("Units.sample does not support `probability`")
        U = self.max_units
        rng = self.np_random
        unit = np.ones(U, dtype=bool)
        action = None
        if mask is not None:
            if mask.get("unit") is not None:
                unit = np.asarray(mask["unit"], dtype=bool)
            if mask.get("action") is not None:
                action = np.asarray(mask["action"], dtype=bool)
        values: dict[str, np.ndarray] = {}
        offset = 0
        for c in self.components:
            if c.kind == "discrete":
                out = np.zeros(U, dtype=np.int64)
                for u in range(U):
                    if not unit[u]:
                        continue
                    legal = np.arange(c.size) if action is None else np.flatnonzero(action[u, offset:offset + c.size])
                    if legal.size:
                        out[u] = int(rng.choice(legal))
                values[c.name] = out
                offset += c.size
            else:
                box = self._boxes[c.name]
                low = np.broadcast_to(box.low, (c.size,)).astype(np.float64)
                high = np.broadcast_to(box.high, (c.size,)).astype(np.float64)
                bounded = np.isfinite(low) & np.isfinite(high)
                draw = np.where(bounded, rng.uniform(np.where(bounded, low, 0.0), np.where(bounded, high, 1.0),
                                                     size=(U, c.size)),
                                rng.normal(size=(U, c.size)))
                draw = np.clip(draw, low, high).astype(np.float32)
                draw[~unit] = 0.0
                values[c.name] = draw
        if self.per_unit_kind == "dict":
            return values
        if self.per_unit_kind == "multi_discrete":
            return np.stack([values[c.name] for c in self.components], axis=1)
        return values["0"]

    def __eq__(self, other: object) -> bool:
        # ``components`` pins the order: gymnasium's Dict equality ignores key order,
        # but the order fixes the mask layout.
        return (isinstance(other, Units) and self.max_units == other.max_units
                and self.components == other.components
                and self.per_unit == other.per_unit and self.only_if == other.only_if)

    def __hash__(self) -> int:
        return hash((self.max_units, self.components))

    def __repr__(self) -> str:
        rule = f", only_if={ {k: (p, sorted(v)) for k, (p, v) in self.only_if.items()} }" if self.only_if else ""
        return f"Units({self.max_units}, {self.per_unit!r}{rule})"

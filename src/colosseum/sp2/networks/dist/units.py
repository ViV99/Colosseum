"""``UnitsDist``: per-unit components of one ``Units`` group, with masks and ``only_if`` (SP2 block 2).

K = ``max_units`` deciders (one per unit). A unit is a valid decider iff its ``unit`` mask is
True and at least one of its components is valid:
- a discrete component is valid iff its mask row is non-empty and its ``only_if`` gate holds;
- a box component is valid iff its ``only_if`` gate holds;
- the gate of ``child <- (parent, values)`` holds iff the parent component is valid and the
  given action's parent value is in ``values``.
Log-prob, entropy and KL sum the valid components of a unit; invalid positions are 0 via
``torch.where`` (no NaN, no gradient).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup, split_unit_actions, units_component_valid
from colosseum.sp2.envs.spaces import UnitComponent
from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import (
    categorical_entropy,
    categorical_kl,
    categorical_log_prob,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_prob,
    masked_log_softmax,
    sample_categorical,
)


class UnitsDist(Distribution):
    """``params``: discrete component -> logits ``[B, U, n]``; box component -> ``{"mean": [B, U, d],
    "log_std": [B, U, d] | [d]}``. ``mask``: ``{"unit": [B, U], "action": [B, U, sum discrete sizes]}``
    (either key optional). Actions: the env format of the group with a leading ``B``."""

    def __init__(self, group: ActionGroup, params: Mapping[str, Any], mask: Mapping[str, Tensor] | None = None) -> None:
        if group.kind != "units":
            raise ValueError(f"UnitsDist needs a units action group, got {group.kind!r}")
        self.group = group
        self.units = group.units
        self.U = self.units.max_units
        names = [c.name for c in self.units.components]
        if not isinstance(params, Mapping):
            raise ValueError(f"UnitsDist: params must be a dict {self._params_form()}, got {type(params).__name__}")
        if set(params) != set(names):
            raise ValueError(f"UnitsDist: params for components {sorted(map(str, params))}, the units have {names}")
        self.params: dict[str, Any] = {}
        first = params[names[0]]
        ref = first.get("mean") if isinstance(first, Mapping) else first
        if not isinstance(ref, Tensor) or ref.dim() != 3:
            kind = "mean" if isinstance(first, Mapping) else "logits"
            got = tuple(ref.shape) if isinstance(ref, Tensor) else type(ref).__name__
            raise ValueError(f"UnitsDist: component {names[0]!r} needs {kind} [B, {self.U}, "
                             f"{self.units.components[0].size}], got {got}")
        self.B = int(ref.shape[0])
        device = ref.device
        for c in self.units.components:
            p = params[c.name]
            if c.kind == "discrete":
                if not isinstance(p, Tensor) or tuple(p.shape) != (self.B, self.U, c.size):
                    got = tuple(p.shape) if isinstance(p, Tensor) else type(p).__name__
                    raise ValueError(f"UnitsDist: component {c.name!r} needs logits [B, {self.U}, {c.size}], got {got}")
                self.params[c.name] = p
            else:
                if not isinstance(p, Mapping) or set(p) != {"mean", "log_std"}:
                    raise ValueError(f"UnitsDist: box component {c.name!r} needs {{'mean', 'log_std'}}")
                if not isinstance(p["mean"], Tensor) or tuple(p["mean"].shape) != (self.B, self.U, c.size):
                    got = tuple(p["mean"].shape) if isinstance(p["mean"], Tensor) else type(p["mean"]).__name__
                    raise ValueError(f"UnitsDist: component {c.name!r} needs mean [B, {self.U}, {c.size}], got {got}")
                if not isinstance(p["log_std"], Tensor) or tuple(p["log_std"].shape) not in (
                        (self.B, self.U, c.size), (c.size,)):
                    got = tuple(p["log_std"].shape) if isinstance(p["log_std"], Tensor) else type(p["log_std"]).__name__
                    raise ValueError(f"UnitsDist: component {c.name!r} needs log_std [B, {self.U}, {c.size}] or "
                                     f"[{c.size}], got {got}")
                self.params[c.name] = {"mean": p["mean"], "log_std": torch.broadcast_to(p["log_std"], p["mean"].shape)}
        unit, action = self._check_mask(mask)
        self.unit_mask = (torch.ones(self.B, self.U, dtype=torch.bool, device=device) if unit is None
                          else unit.to(device=device, dtype=torch.bool))
        self.action_mask = None if action is None else action.to(device=device, dtype=torch.bool)
        self._row: dict[str, Tensor | None] = {}   # component mask rows [B, U, n]
        self._log_p: dict[str, Tensor] = {}
        offset = 0
        for c in self.units.components:
            if c.kind == "discrete":
                row = None if self.action_mask is None else self.action_mask[..., offset:offset + c.size]
                self._row[c.name] = row
                self._log_p[c.name] = masked_log_softmax(self.params[c.name], row)
                offset += c.size

    def _params_form(self) -> str:
        parts = []
        for c in self.units.components:
            if c.kind == "discrete":
                parts.append(f"{c.name!r}: logits [B, {self.U}, {c.size}]")
            else:
                parts.append(f"{c.name!r}: {{'mean': [B, {self.U}, {c.size}], "
                             f"'log_std': [B, {self.U}, {c.size}] | [{c.size}]}}")
        return "{" + ", ".join(parts) + "}"

    def _check_mask(self, mask: Mapping[str, Tensor] | None) -> tuple[Tensor | None, Tensor | None]:
        """``(unit, action)`` of a mask dict after checking its keys and shapes."""
        mask = mask or {}
        unknown = sorted(set(mask) - {"unit", "action"})
        if unknown:
            raise ValueError(f"UnitsDist: unknown mask keys {unknown} (expected 'unit' and/or 'action')")
        unit, action = mask.get("unit"), mask.get("action")
        if unit is not None and tuple(unit.shape) != (self.B, self.U):
            raise ValueError(f"UnitsDist: unit mask must be [B, {self.U}], got {tuple(unit.shape)}")
        if action is not None and tuple(action.shape) != (self.B, self.U, self.group.mask_size):
            raise ValueError(f"UnitsDist: action mask must be [B, {self.U}, {self.group.mask_size}], "
                             f"got {tuple(action.shape)}")
        return unit, action

    # ---- layout --------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.B

    @property
    def num_deciders(self) -> int:
        return self.U

    def _split(self, actions: Any) -> dict[str, Tensor]:
        return split_unit_actions(self.units, actions)

    def _join(self, values: dict[str, Tensor]) -> Any:
        kind = self.units.per_unit_kind
        if kind == "dict":
            return values
        if kind == "multi_discrete":
            return torch.stack([values[c.name] for c in self.units.components], dim=-1)
        return values["0"]

    def _row_ok(self, c: UnitComponent) -> Tensor:
        row = self._row.get(c.name)
        if row is None:
            return self.unit_mask
        return self.unit_mask & row.any(dim=-1)

    def _component_valid(self, actions: Any) -> dict[str, Tensor]:
        """``[B, U]`` validity per component for the given actions (``units_component_valid``)."""
        base = {c.name: self._row_ok(c) if c.kind == "discrete" else self.unit_mask for c in self.units.components}
        return units_component_valid(self.units, base, self._split(actions))

    # ---- protocol ----------------------------------------------------------------------

    def _draw(self, greedy: bool) -> Any:
        values: dict[str, Tensor] = {}
        for c in self.units.components:
            if c.kind == "discrete":
                log_p = self._log_p[c.name]
                x = log_p.argmax(dim=-1) if greedy else sample_categorical(log_p)
                values[c.name] = torch.where(self._row_ok(c), x, torch.zeros_like(x))
            else:
                p = self.params[c.name]
                x = p["mean"] if greedy else p["mean"] + torch.exp(p["log_std"]) * torch.randn_like(p["mean"])
                values[c.name] = torch.where(self.unit_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self._join(values)

    def sample(self) -> Any:
        with torch.no_grad():
            return self._draw(greedy=False)

    def mode(self) -> Any:
        return self._draw(greedy=True)

    def _sum_valid(self, terms: dict[str, Tensor], valid: dict[str, Tensor]) -> Tensor:
        total = torch.zeros(self.B, self.U, dtype=next(iter(terms.values())).dtype,
                            device=self.unit_mask.device)
        for name, value in terms.items():
            total = total + torch.where(valid[name], value, torch.zeros_like(value))
        return total

    def unit_log_prob(self, actions: Any) -> Tensor:
        a = self._split(actions)
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_log_prob(self._log_p[c.name], a[c.name].long().clamp(0, c.size - 1))
            else:
                p = self.params[c.name]
                terms[c.name] = gaussian_log_prob(p["mean"], p["log_std"], a[c.name].to(p["mean"].dtype))
        return self._sum_valid(terms, self._component_valid(actions))

    def unit_entropy(self, actions: Any) -> Tensor:
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_entropy(self._log_p[c.name], self._row[c.name])
            else:
                terms[c.name] = gaussian_entropy(self.params[c.name]["log_std"])
        return self._sum_valid(terms, self._component_valid(actions))

    def unit_valid(self, actions: Any) -> Tensor:
        valid = self._component_valid(actions)
        any_valid = torch.zeros_like(self.unit_mask)
        for v in valid.values():
            any_valid = any_valid | v
        return self.unit_mask & any_valid

    def unit_kl(self, other: Distribution, actions: Any) -> Tensor:
        if not isinstance(other, UnitsDist) or other.group != self.group:
            raise TypeError(f"KL between UnitsDist and {type(other).__name__} over a different group")
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_kl(self.params[c.name], self._row[c.name],
                                               other.params[c.name], other._row[c.name])
            else:
                p, q = self.params[c.name], other.params[c.name]
                terms[c.name] = gaussian_kl(p["mean"].float(), p["log_std"].float(),
                                            q["mean"].float(), q["log_std"].float())
        return self._sum_valid(terms, self._component_valid(actions))

    def apply_mask(self, mask: Mapping[str, Tensor] | None) -> UnitsDist:
        if mask is None:
            return self
        unit, action = self._check_mask(mask)
        new_unit = self.unit_mask if unit is None else self.unit_mask & unit.to(self.unit_mask.device, torch.bool)
        if action is None:
            new_action = self.action_mask
        else:
            action = action.to(self.unit_mask.device, torch.bool)
            new_action = action if self.action_mask is None else self.action_mask & action
        combined = {"unit": new_unit}
        if new_action is not None:
            combined["action"] = new_action
        return UnitsDist(self.group, self.params, combined)

    @classmethod
    def cat(cls, dists: Sequence[UnitsDist]) -> UnitsDist:
        first = dists[0]
        params: dict[str, Any] = {}
        for c in first.units.components:
            if c.kind == "discrete":
                params[c.name] = torch.cat([d.params[c.name] for d in dists], dim=0)
            else:
                params[c.name] = {k: torch.cat([d.params[c.name][k] for d in dists], dim=0)
                                  for k in ("mean", "log_std")}
        mask: dict[str, Tensor] = {"unit": torch.cat([d.unit_mask for d in dists], dim=0)}
        if any(d.action_mask is not None for d in dists):
            mask["action"] = torch.cat([
                d.action_mask if d.action_mask is not None
                else torch.ones(d.B, d.U, d.group.mask_size, dtype=torch.bool, device=d.unit_mask.device)
                for d in dists], dim=0)
        return UnitsDist(first.group, params, mask)

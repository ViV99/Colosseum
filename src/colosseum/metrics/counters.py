"""Run-level sums of cumulative counters reported by restartable sources (SP5 spec block 8; plan amendment A2).

A *source* is anything that counts from zero when it starts and reports its cumulative values: a node session
(``"<node_id>/<session_id>"``), a learner incarnation (``"learner/<agent>/<generation>"``), the hub process itself
(``"hub/<hub incarnation>"``). The book keeps the highest value per (source, name) and adds every increase to the
run total, so the totals never decrease while sources join, restart and leave, and a resumed hub that hears the
same sources report again (their cumulative values continue) counts nothing twice: ``state()`` saves the totals
together with the per-source values they were computed from (the ``BudgetCounter`` rule of spec 4.4, for any
number of names).

A value below the source's previous one means the source restarted under the same key (never by construction:
session ids and generations are new per start): the new value is counted from zero. Names are flat strings;
dotted names (``"dropped_chunks.disconnect"``) are grouped by ``grouped()`` for the ``system`` record and
``hub_state.json``. ``by_source_prefix`` gives per-node views (``f"{node_id}/"``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["CounterBook"]


class CounterBook:
    """Monotonic run totals of cumulative per-source counters (module docstring). Not thread-safe: one owner
    (the hub main loop, or ``MetricsHub``)."""

    def __init__(self, totals: Mapping[str, int] | None = None,
                 sources: Mapping[str, Mapping[str, int]] | None = None) -> None:
        self._totals: dict[str, int] = {str(k): int(v) for k, v in (totals or {}).items()}
        self._sources: dict[str, dict[str, int]] = {str(s): {str(k): int(v) for k, v in vals.items()}
                                                    for s, vals in (sources or {}).items()}

    def report(self, source: str, values: Mapping[str, int]) -> dict[str, int]:
        """Take ``source``'s cumulative ``values``; return the increments applied (names with a zero increment
        are left out). Negative or non-integer values are ignored (a malformed report never lowers a total)."""
        last = self._sources.setdefault(str(source), {})
        applied: dict[str, int] = {}
        for name, raw in values.items():
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                continue
            name = str(name)
            prev = last.get(name, 0)
            delta = raw - prev if raw >= prev else raw      # a lower value: the source started over
            last[name] = raw
            if delta:
                self._totals[name] = self._totals.get(name, 0) + delta
                applied[name] = delta
        return applied

    def add(self, name: str, delta: int) -> None:
        """Count ``delta`` (> 0; anything else is ignored) directly (an event of the book's owner, e.g. a snapshot
        hold of the hub); it belongs to no source."""
        if isinstance(delta, int) and not isinstance(delta, bool) and delta > 0:
            self._totals[str(name)] = self._totals.get(str(name), 0) + delta

    def get(self, name: str) -> int:
        return self._totals.get(str(name), 0)

    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def by_source_prefix(self, prefix: str) -> dict[str, int]:
        """Per name, the sum of the last cumulative values of every source whose key starts with ``prefix``
        (a node's view: ``f"{node_id}/"``; direct ``add`` counts belong to no source)."""
        out: dict[str, int] = {}
        for source, values in self._sources.items():
            if source.startswith(prefix):
                for name, value in values.items():
                    out[name] = out.get(name, 0) + value
        return out

    def grouped(self, *, groups: Mapping[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
        """Totals with dotted names nested one level: ``{"relay_bytes": 5, "dropped_chunks": {"disconnect": 2}}``.
        ``groups`` lists the members every group must show (zeros included), e.g. the six chunk-drop reasons."""
        out: dict[str, Any] = {}
        for group, members in (groups or {}).items():
            out[group] = {m: 0 for m in members}
        for name, value in sorted(self._totals.items()):
            head, dot, tail = name.partition(".")
            if dot:
                out.setdefault(head, {})
                out[head][tail] = value
            else:
                out[name] = value
        return out

    def state(self) -> dict[str, Any]:
        return {"totals": dict(self._totals), "sources": {s: dict(v) for s, v in self._sources.items()}}

    @classmethod
    def from_state(cls, data: Mapping[str, Any] | None) -> CounterBook:
        data = data or {}
        return cls(data.get("totals") or {}, data.get("sources") or {})

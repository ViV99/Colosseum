"""Live snapshot ratings of a training run (SP4 spec block 2, "Живой рейтинг").

``LiveRatings`` is a thread of the training main process. Every ``interval_sec`` (monotonic; at once at the start
when the table already holds matches, as on a resume with the parent's journal) it takes a copy of the
in-memory ``ComparisonTable`` that the journal writer feeds (``table.copy()``; nothing else is shared), fits every
layout of the run's sources (``RUN_SOURCES``: train and pulse; ``colosseum.arena.bt.fit`` warm-started from the
previous fit), computes Laplace uncertainty (``LAPLACE_SAMPLES`` draws) and the G2 test, summarizes one-team
layouts, and publishes an immutable ``RatingsSnapshot`` (``latest()``). Above ``bt.DENSE_LIMIT`` parameters the
fit's own ``bt.ITERATIVE_SAMPLES`` perturb-and-solve draws replace the Laplace samples (``bt.laplace``). A refit is
skipped while the table has not changed. It never touches the coordinator, the storage or the metrics hub; the main
loop reads ``latest()`` at its metrics tick. A failed refit is an ERROR (with traceback, once) and the thread goes
on. Anchor notes (a configured or default anchor that did not play in a cell) are logged once each (INFO).

``bt_section(snapshot)`` is the ``bt`` key of ``ratings.json``: ``{layout: {group: [entity rows]}}`` sorted by
rating; keys starting with ``@`` (impossible in role names) carry per-layout extras: ``@bias`` (team index -> Elo),
``@anchors`` (``"<group>/<component>"`` -> anchor, members and rule), ``@transitivity``
(``TransitivityResult.to_dict()``; ``p_value`` null when untestable), ``@fit`` (convergence, iterations,
``approximate``) and, for one-team layouts, ``@score`` (entities and compositions; intervals null below two
matches). Missing or non-finite numbers are null, never ``inf``. ``bt_summary(snapshot, agents)`` is the compact
form for the ``ratings`` record of ``metrics.jsonl``: per agent and layout its latest version bucket and its best
entity.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from colosseum.arena.bt import (
    DEFAULT_OPTIONS,
    BTFit,
    BTOptions,
    ScoreSummary,
    TransitivityResult,
    Uncertainty,
    fit,
    laplace,
    score_summary,
    transitivity_test,
)
from colosseum.arena.comparisons import ComparisonTable, coefficients
from colosseum.arena.entities import entity_version, lineage
from colosseum.arena.matchlog import RUN_SOURCES

logger = logging.getLogger(__name__)

LAPLACE_SAMPLES = 1000
# The final refit at the end of a run (``LiveRatings.finish``) runs only when the last refit took at most this long
# (seconds); before any refit finished, only when every layout's fit stays on the dense path (``dense_limit``).
FINAL_REFIT_MAX_SEC = 2.0


@dataclass(frozen=True)
class RatingsSnapshot:
    version: int                                   # increments per published refit
    env_steps: int                                 # the global env-step counter when the refit started
    fits: Mapping[str, BTFit]                      # per layout with team pairs
    uncertainty: Mapping[str, Uncertainty]
    scores: Mapping[str, ScoreSummary]             # per one-team layout
    transitivity: Mapping[str, TransitivityResult] = field(default_factory=dict)
    refit_sec: float = 0.0                         # how long the refit took (monotonic)


def rating_inputs(config) -> tuple[BTOptions, list[str], str | None]:
    """A config's fit inputs, the one place that turns a config into them (live ratings, report, tournament
    ``top:N``, scripts; contract amendment A13): the options (module constants, ``ratings.anchor``, the config's
    agent order for the "first trainable agent" anchor rule), the scripted / frozen agents and the first
    ``RandomBot`` agent."""
    options = BTOptions(anchor=config.ratings.anchor, agent_order=tuple(config.agent_ids()))
    return options, list(config.fixed_agent_ids()), random_bot_agent(config)


def random_bot_agent(config) -> str | None:
    """The config's first scripted agent whose class is ``colosseum.players.RandomBot`` (config order), or None."""
    from colosseum.core.registry import import_class
    from colosseum.players import RandomBot

    for agent_id in config.fixed_agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind != "scripted":
            continue
        try:
            if import_class(entry.class_path) is RandomBot:
                return agent_id
        except (ImportError, AttributeError, ValueError, TypeError):
            continue
    return None


def compute_snapshot(table: ComparisonTable, *, version: int, env_steps: int, options: BTOptions = DEFAULT_OPTIONS,
                     fixed_agents: Collection[str] = (), random_bot: str | None = None,
                     agent_order: Sequence[str] = (), top_k: int = 0, quantile: float = 0.1,
                     rng: np.random.Generator, previous: RatingsSnapshot | None = None,
                     samples: int = LAPLACE_SAMPLES, sources: Collection[str] = RUN_SOURCES) -> RatingsSnapshot:
    """One refit of every layout of ``table`` (module docstring)."""
    started = time.monotonic()
    fits: dict[str, BTFit] = {}
    uncertainty: dict[str, Uncertainty] = {}
    scores: dict[str, ScoreSummary] = {}
    transitivity: dict[str, TransitivityResult] = {}
    for layout in table.layouts():
        if table.pairs(layout, sources):
            warm = previous.fits.get(layout) if previous is not None else None
            fitted = fit(table, layout, sources=sources, options=options, fixed_agents=fixed_agents,
                         random_bot=random_bot, agent_order=agent_order, warm=warm)
            fits[layout] = fitted
            uncertainty[layout] = laplace(fitted, samples=samples, rng=rng, top_k=top_k, quantile=quantile)
            transitivity[layout] = transitivity_test(table, fitted, sources=sources)
        if table.scores(layout, sources):
            scores[layout] = score_summary(table, layout, sources=sources)
    return RatingsSnapshot(version=version, env_steps=int(env_steps), fits=fits, uncertainty=uncertainty,
                           scores=scores, transitivity=transitivity, refit_sec=time.monotonic() - started)


class LiveRatings:
    """The refit thread (module docstring)."""

    def __init__(self, table: ComparisonTable, *, interval_sec: float, options: BTOptions = DEFAULT_OPTIONS,
                 fixed_agents: Collection[str] = (), random_bot: str | None = None, agent_order: Sequence[str] = (),
                 top_k: int = 0, quantile: float = 0.1, env_steps_fn: Callable[[], int],
                 samples: int = LAPLACE_SAMPLES, seed: int = 0) -> None:
        self._table = table
        self._interval = float(interval_sec)
        self._kwargs = dict(options=options, fixed_agents=tuple(fixed_agents), random_bot=random_bot,
                            agent_order=tuple(agent_order), top_k=int(top_k), quantile=float(quantile),
                            samples=int(samples))
        self._env_steps_fn = env_steps_fn
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()          # guards _latest
        self._refit_lock = threading.Lock()    # one refit at a time (the thread, or refit_now from the main loop)
        self._latest: RatingsSnapshot | None = None
        self._table_version = -1
        self._notes: set[str] = set()
        self._failed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def latest(self) -> RatingsSnapshot | None:
        """The newest published snapshot (immutable), or None before the first refit."""
        with self._lock:
            return self._latest

    @property
    def last_refit_sec(self) -> float | None:
        snapshot = self.latest()
        return None if snapshot is None else snapshot.refit_sec

    def refit_now(self) -> RatingsSnapshot | None:
        """Refit now unless the table is unchanged since the last refit; returns the latest snapshot."""
        with self._refit_lock:
            previous = self.latest()
            if previous is not None and self._table.version == self._table_version:
                return previous
            table = self._table.copy()          # the writer goes on adding to the shared table meanwhile
            version = table.version
            snapshot = compute_snapshot(table, version=1 if previous is None else previous.version + 1,
                                        env_steps=int(self._env_steps_fn()), rng=self._rng, previous=previous,
                                        **self._kwargs)
            for fitted in snapshot.fits.values():
                for group in fitted.groups:
                    key = f"{fitted.layout}/{group.group}/{group.component}: {group.note}"
                    if group.note and key not in self._notes:
                        self._notes.add(key)
                        logger.info(f"Ratings, layout {fitted.layout}, group {group.group}: {group.note}; the anchor "
                                    f"is {group.anchor} ({group.rule})")
            with self._lock:
                self._latest = snapshot
                self._table_version = version
            return snapshot

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-ratings", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # A table that already holds matches when the thread starts (a resume: the parent's journal) is fitted at
        # once, so snapshot retention gets ratings without waiting a whole interval (T4.1); an empty one (a fresh
        # run) waits for the first interval as before.
        if self._table.layouts() and not self._stop.is_set():
            self._refit_logged()
        while not self._stop.wait(self._interval):
            self._refit_logged()

    def _refit_logged(self) -> None:
        try:
            self.refit_now()
        except Exception as e:  # noqa: BLE001 - ratings must never stop training
            logger.error(f"Live ratings: a refit failed ({type(e).__name__}: {e}); training goes on",
                         exc_info=not self._failed)
            self._failed = True

    @property
    def alive(self) -> bool:
        """The refit thread is running (started and not yet ended)."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float) -> None:
        """Stop the thread (a refit in progress finishes first, up to ``timeout`` seconds)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def finish(self, timeout: float) -> RatingsSnapshot | None:
        """End of a run (the journal writer has flushed): stop the thread, then refit once more on the final data
        unless that would be slow; returns the newest snapshot (None if there is none). The final refit is skipped,
        with one INFO line naming the reason and the snapshot kept, when the thread is still refitting after
        ``timeout`` seconds, when the last refit took longer than ``FINAL_REFIT_MAX_SEC``, or, before any refit
        finished, when a layout's fit would take the iterative path (more parameters than ``dense_limit``; seconds
        per refit). A failed final refit is an ERROR; the last snapshot is kept."""
        self.stop(timeout)
        reason = None
        last = self.last_refit_sec
        if self.alive:
            reason = f"a refit is still running {timeout:g} s after the stop"
        elif last is not None and last > FINAL_REFIT_MAX_SEC:
            reason = f"the last refit took {last:.1f} s (over {FINAL_REFIT_MAX_SEC:g} s)"
        elif last is None:
            size = self._largest_fit()
            if size is not None and size[1] > self._kwargs["options"].dense_limit:
                reason = (f"no refit finished yet and layout {size[0]!r} has {size[1]} parameters (over "
                          f"{self._kwargs['options'].dense_limit}: the iterative path)")
        if reason is None:
            try:
                return self.refit_now()
            except Exception:  # noqa: BLE001 - the final records are still written
                logger.exception("Live ratings: the final refit failed; ratings.json keeps the last snapshot")
                return self.latest()
        kept = self.latest()
        what = ("has no Bradley-Terry section" if kept is None
                else f"keeps the snapshot of env step {kept.env_steps} and may miss the games since")
        logger.info(f"Live ratings: {reason}; no final refit, so ratings.json {what}")
        return kept

    def _largest_fit(self) -> tuple[str, int] | None:
        """``(layout, parameters)`` of the layout with the most fit parameters in the table (entities with a
        non-zero coefficient plus the free side-bias parameters), None without team pairs."""
        largest = None
        table = self._table
        for layout in table.layouts():
            pairs = table.pairs(layout, RUN_SOURCES)
            if not pairs:
                continue
            entities = set()
            for a, b, _ in pairs:
                entities.update(coefficients(a, b))
            size = len(entities) + max(b.team for _, b, _ in pairs)       # canonical order: a.team < b.team
            if largest is None or size > largest[1]:
                largest = (layout, size)
        return largest


# ---------------------------------------------------------------------------
# ratings.json and metrics.jsonl forms
# ---------------------------------------------------------------------------


def _finite(value: float | None) -> float | None:
    """A JSON-ready number: None for a missing or non-finite value (never ``inf`` / ``nan`` in JSON)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _entity_rows(fitted: BTFit, unc: Uncertainty) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for group in fitted.groups:
        for entity in group.entities:
            lo, hi = unc.ci95.get(entity, (fitted.rating[entity], fitted.rating[entity]))
            row = {"entity": entity, "lineage": lineage(entity), "version": entity_version(entity),
                   "rating": fitted.rating[entity], "se": _finite(fitted.se[entity]), "ci_low": _finite(lo),
                   "ci_high": _finite(hi), "lcb": _finite(unc.lcb.get(entity)),
                   "games": int(fitted.games.get(entity, 0)), "p_best": unc.p_best_group.get(entity),
                   "p_best_lineage": unc.p_best_lineage.get(entity), "component": group.component,
                   "anchor": entity in group.anchor_members}
            if unc.p_top_k:
                row["p_top_k"] = unc.p_top_k.get(entity)
            rows.setdefault(group.group, []).append(row)
    for group_rows in rows.values():
        group_rows.sort(key=lambda r: (-r["rating"], r["entity"]))
    return rows


def bt_section(snapshot: RatingsSnapshot | None) -> dict:
    """The ``bt`` key of ``ratings.json`` (module docstring); ``{}`` without a snapshot."""
    if snapshot is None:
        return {}
    out: dict[str, dict] = {}
    for layout, fitted in snapshot.fits.items():
        section: dict = dict(_entity_rows(fitted, snapshot.uncertainty[layout]))
        section["@bias"] = {str(t): _finite(b) for t, b in fitted.bias.items()}
        section["@anchors"] = {f"{g.group}/{g.component}": {"anchor": g.anchor, "members": g.anchor_members,
                                                            "rule": g.rule} for g in fitted.groups}
        test = snapshot.transitivity.get(layout)
        if test is not None:
            section["@transitivity"] = test.to_dict()
        section["@fit"] = {"converged": fitted.converged, "iterations": fitted.iterations,
                           "approximate": fitted.approximate}
        out[layout] = section
    for layout, summary in snapshot.scores.items():
        out.setdefault(layout, {})["@score"] = {
            "entities": [{"entity": e, "n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high}
                         for e, c in summary.entities.items()],
            "compositions": [{"composition": k, "n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high,
                              "homogeneous": summary.homogeneous[k]} for k, c in summary.compositions.items()],
        }
    return out


def _cell(entity: str, fitted: BTFit, unc: Uncertainty) -> dict:
    lo, hi = unc.ci95.get(entity, (fitted.rating[entity], fitted.rating[entity]))
    return {"entity": entity, "rating": fitted.rating[entity], "ci_low": _finite(lo), "ci_high": _finite(hi),
            "lcb": _finite(unc.lcb.get(entity)), "games": int(fitted.games.get(entity, 0))}


def bt_summary(snapshot: RatingsSnapshot | None, agents: Sequence[str]) -> dict:
    """``{agent: {layout: {"latest": cell, "best": cell}}}`` for the ``ratings`` record: ``latest`` is the agent's
    newest version bucket, ``best`` its entity with the highest rating (Bradley-Terry layouts), or the same from the
    mean scores of one-team layouts (cells with ``mean``, ``ci_low``, ``ci_high``, ``n``)."""
    if snapshot is None:
        return {}
    out: dict[str, dict] = {}
    for agent in agents:
        per_layout: dict[str, dict] = {}
        for layout, fitted in snapshot.fits.items():
            mine = [e for e in fitted.rating if lineage(e) == agent and entity_version(e) is not None]
            if not mine:
                continue
            unc = snapshot.uncertainty[layout]
            newest = max(mine, key=lambda e: entity_version(e))
            best = max(mine, key=lambda e: (fitted.rating[e], entity_version(e)))
            per_layout[layout] = {"latest": _cell(newest, fitted, unc), "best": _cell(best, fitted, unc)}
        for layout, summary in snapshot.scores.items():
            mine = [e for e in summary.entities if lineage(e) == agent and entity_version(e) is not None]
            if not mine:
                continue
            newest = max(mine, key=lambda e: entity_version(e))
            best = max(mine, key=lambda e: (summary.entities[e].mean, entity_version(e)))
            per_layout[layout] = {
                key: {"entity": e, "mean": summary.entities[e].mean, "ci_low": summary.entities[e].ci_low,
                      "ci_high": summary.entities[e].ci_high, "n": summary.entities[e].n}
                for key, e in (("latest", newest), ("best", best))}
        if per_layout:
            out[agent] = per_layout
    return out


def console_bt(summary: Mapping[str, Mapping]) -> tuple[float, float] | None:
    """``(rating, half 95 % CI)`` of an agent's latest bucket for the console (``bt_summary`` of one agent): the
    Bradley-Terry layout where that bucket has the most games; None without one."""
    best = None
    for cells in summary.values():
        latest = cells.get("latest", {})
        if "rating" not in latest or latest.get("ci_low") is None or latest.get("ci_high") is None:
            continue
        if best is None or latest["games"] > best["games"]:
            best = latest
    if best is None:
        return None
    return best["rating"], (best["ci_high"] - best["ci_low"]) / 2.0

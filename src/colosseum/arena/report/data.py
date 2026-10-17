"""Report data (SP4 spec block 5): one JSON-ready dict per page section, built from a run dir.

Sections (``SECTIONS``): summary, ratings, matrix, anchors, opponents, training, system, matches,
tournaments. The same builders serve ``colosseum report`` (``build_report_data``) and the dashboard
(``ReportBuilder.section`` behind ``/api/<section>``).

Rules:
- read-only: the match log is opened with ``MatchLog.open_readonly``; nothing in the run dir is written;
- order by counters (``train_step``, ``env_steps``, versions, journal row order), never by ``ts``
  (WSL2 wall clocks jump; ``ts`` is informational);
- bounded (``ReportLimits``): series thinned to 2 000 points (first and last kept), 2 000 entity rows
  per rating group, the latest 1 000 matches, at most 100 snapshots in the per-agent matrix;
- robust to a running run (a partial last line of ``metrics.jsonl`` is left for the next read) and to
  SP3 run dirs (no ``matches.db`` / ``run_state.json``): a section without its source carries a note;
- ratings: while the run is running (``run_state.json`` + live pid) the BT tables come from the
  ``bt`` section of ``ratings.json`` (live refit, Laplace intervals; the ``@``-keyed extras give the
  bias, anchors and transitivity, contract amendment A14); otherwise they are fitted from the
  journal (sources ``train`` + ``pulse``) with ``ratings.bootstrap_samples`` bootstrap replicas and cached
  until the journal files change; the fit, its bootstrap and the G2 test use one read of the journal
  (``bt.bootstrap_data`` / ``BootstrapData.to_table``), so a live writer cannot make them disagree;
- bounded memory: ``ratings.json`` and the config are parsed once per file change, ``MetricsTail`` keeps
  only the record kinds the page draws (``REPORT_KINDS``), each (kind, agent) series thinned on ingest to
  about ``2 * series_points`` records whatever the run's length, and reads in fixed-size chunks; score-layout
  compositions are capped like the entity rows (``compositions_truncated``);
- no ``inf`` / ``nan`` reaches the data: missing intervals (a score with fewer than two games, an infinite
  standard error) are ``null``;
- anchors with pulse: a stored snapshot is "unevaluated" when the journal's ``pulse`` source has no game of
  its version. The journal (not ``pulse_state.json``, which knows only this run's offers) is the source, so
  snapshots imported from a parent run count as evaluated when the parent's pulse played them (a run-dir
  resume copies the parent's pulse matches into the journal) and as unevaluated otherwise;
- tournaments: each ``tournaments/<name>/report.json`` through ``modes.report_view`` (contract amendment A26).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from colosseum.arena.entities import entity_version
from colosseum.arena.matchlog import RUN_SOURCES
from colosseum.core.run_state import effective_status, read_run_state

logger = logging.getLogger(__name__)

REPORT_FORMAT = 1
SECTIONS = ("summary", "ratings", "matrix", "anchors", "opponents", "training", "system", "matches",
            "tournaments")
MIN_CELL_GAMES = 20           # heat-map cells with fewer comparisons are drawn hatched ("few games")
PFSP_TOP = 20                 # rows of each owner's PFSP table in "opponents"
FILTERS = {"ratings": ("agent", "layout"), "matrix": ("agent", "layout"),
           "matches": ("agent", "layout", "source", "outcome")}
TRAIN_DEFAULT = ("total_loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                 "explained_variance", "rho_trunc_mass", "ess", "lr", "kickstart_loss", "kickstart_lambda")
# runs before SP4 T3.1 recorded the clipped fractions instead of the truncated masses (e.g. the SP3 fixture)
TRAIN_LEGACY = {"rho_trunc_mass": "rho_clip_frac", "c_trunc_mass": "c_clip_frac"}
SYSTEM_DEFAULT = ("env_steps_per_sec", "dropped_match_results", "dropped_reward_episodes")
_RECORD_META = ("ts", "kind", "agent", "train_step", "env_steps")
_Z95 = 1.959963984540054


@dataclass(frozen=True)
class ReportLimits:
    series_points: int = 2000
    entity_rows: int = 2000
    matches: int = 1000
    snapshot_matrix: int = 100


def thin(n: int, limit: int) -> list[int]:
    """Indices of at most ``limit`` evenly spaced items out of ``n``; the first and last are kept."""
    if n <= limit:
        return list(range(n))
    if limit <= 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})


def wilson(p: float, n: float) -> list[float]:
    """95% Wilson interval of a (fractional) score ``p`` over ``n`` comparisons; n <= 0 -> [0, 1]."""
    if n <= 0:
        return [0.0, 1.0]
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(max(0.0, p * (1 - p)) / n + z2 / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _flat(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Numbers (bools as 0/1) of nested dicts as ``{"a.b": float}``."""
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        out[prefix] = float(value)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    if is_dataclass(obj):
        return asdict(obj)
    return dict(obj)


def _file_sig(*paths: Path) -> tuple:
    """Change detector of files (size, mtime_ns); missing files count as (-1, -1). Not an ordering key."""
    sig = []
    for p in paths:
        try:
            s = p.stat()
            sig.append((s.st_size, s.st_mtime_ns))
        except OSError:
            sig.append((-1, -1))
    return tuple(sig)


REPORT_KINDS = ("train", "system", "episodes")    # the metrics.jsonl record kinds the report draws
READ_CHUNK_BYTES = 1 << 20                         # MetricsTail reads metrics.jsonl in chunks of this size
SERIES_RECORDS = 4000                              # MetricsTail's default per-series cap (2 x series_points)
_COUNTER = {"train": "train_step"}                 # the counter of a record kind; every other kind: env_steps


class _Series:
    """One (kind, agent) series of ``MetricsTail``, thinned on ingest by its counter (stride doubling): one record
    per counter bucket of width ``width`` (the one with the smallest counter) is kept; whenever more than ``cap``
    buckets are kept, the width doubles and neighbouring buckets merge. The record with the largest counter (the
    newest) is always kept as well. Records without a finite counter: the newest ``cap`` of them. At most
    ``2 * cap + 1`` records, whatever the number ingested."""

    __slots__ = ("counter", "cap", "width", "buckets", "newest", "uncounted")

    def __init__(self, counter: str, cap: int) -> None:
        self.counter, self.cap = counter, max(1, int(cap))
        self.width = 1
        self.buckets: dict[int, tuple[float, dict]] = {}
        self.newest: tuple[float, dict] | None = None
        self.uncounted: deque[dict] = deque(maxlen=self.cap)

    def add(self, record: dict) -> None:
        c = record.get(self.counter)
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
            self.uncounted.append(record)
            return
        if self.newest is None or c >= self.newest[0]:
            self.newest = (c, record)
        b = math.floor(c / self.width)
        held = self.buckets.get(b)
        if held is None or c < held[0]:
            self.buckets[b] = (c, record)
        while len(self.buckets) > self.cap:
            self.width *= 2
            merged: dict[int, tuple[float, dict]] = {}
            for b, item in self.buckets.items():
                nb = b // 2                    # floor(floor(c / w) / 2) == floor(c / 2w)
                if nb not in merged or item[0] < merged[nb][0]:
                    merged[nb] = item
            self.buckets = merged

    def records(self) -> list[dict]:
        out = [r for _, r in sorted(self.buckets.values(), key=lambda item: item[0])]
        if self.newest is not None and (not out or out[-1] is not self.newest[1]):
            out.append(self.newest[1])
        return out + list(self.uncounted)


class MetricsTail:
    """Incremental reader of ``metrics.jsonl``: only complete lines are parsed; a partial last line (a
    running run is writing it) waits for the next ``poll``. A malformed complete line is counted, not raised.
    Only records of ``kinds`` are kept (default ``REPORT_KINDS``; e.g. the large ``ratings`` records are not);
    None keeps every kind.

    Bounded memory over a run of any length: each (kind, agent) series is thinned on ingest by its counter
    (``train_step`` for ``train``, else ``env_steps``) with stride doubling (``_Series``), so it holds at most
    ``2 * max_per_series + 1`` records (in practice ``max_per_series + 1``: records without a counter are rare);
    the first and the newest record stay. The report passes ``2 * ReportLimits.series_points``, so its own
    thinning (``thin``) still has its point count after a halving. A poll reads the file in chunks of
    ``READ_CHUNK_BYTES`` (plus at most one line longer than a chunk), up to the size seen at its start."""

    def __init__(self, path: Path, kinds: Sequence[str] | None = REPORT_KINDS, *,
                 max_per_series: int = SERIES_RECORDS) -> None:
        self.path = Path(path)
        self.kinds = None if kinds is None else frozenset(kinds)
        self.max_per_series = int(max_per_series)
        self.offset = 0
        self._series: dict[tuple[str, str | None], _Series] = {}
        self.bad_lines = 0

    def poll(self) -> list[dict]:
        """Read what was appended since the last poll; returns every kept record (series by series)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return self._all()
        if size < self.offset:                 # rewritten (another run with the same dir): start over
            self.offset, self.bad_lines, self._series = 0, 0, {}
        left = size - self.offset
        if left <= 0:
            return self._all()
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            rest = b""
            while left > 0:
                chunk = fh.read(min(READ_CHUNK_BYTES, left))
                if not chunk:
                    break
                left -= len(chunk)
                data = rest + chunk if rest else chunk
                end = data.rfind(b"\n")
                if end < 0:                    # a line longer than a chunk, or the partial last line
                    rest = data
                    continue
                self._ingest(data[:end])
                self.offset += end + 1
                rest = data[end + 1:]
        return self._all()

    def _ingest(self, block: bytes) -> None:
        for line in block.split(b"\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                self.bad_lines += 1
                continue
            if not isinstance(record, dict):
                continue
            kind = str(record.get("kind"))
            if self.kinds is not None and kind not in self.kinds:
                continue
            agent = record.get("agent")
            key = (kind, None if agent is None else str(agent))
            series = self._series.get(key)
            if series is None:
                series = self._series[key] = _Series(_COUNTER.get(kind, "env_steps"), self.max_per_series)
            series.add(record)

    def _all(self) -> list[dict]:
        return [r for series in self._series.values() for r in series.records()]

    def of(self, kind: str) -> list[dict]:
        return [r for (k, _), series in self._series.items() if k == kind for r in series.records()]


def failed_section(error: str) -> dict:
    """The section drawn in place of one whose build failed (the static report and the dashboard's page)."""
    return {"notes": [f"This section failed: {error}"]}


def _sorted_by(records: list[dict], key: str) -> list[dict]:
    """Stable sort by a counter (records without it keep file order at the end)."""
    return sorted(records, key=lambda r: (r.get(key) is None, r.get(key) or 0))


def _team_entity(team) -> str | None:
    entities = {e for e, _ in team.members}
    return entities.pop() if len(entities) == 1 else None


def _team_lineage(team) -> str | None:
    from colosseum.arena.entities import lineage

    names = {lineage(e) for e, _ in team.members}
    return names.pop() if len(names) == 1 else None


class ReportBuilder:
    """Builds report sections of one run dir, caching what is expensive (journal tables and fits) until the
    journal files change. Not thread-safe: the dashboard serializes calls with a lock."""

    def __init__(self, run_dir: str | Path, *, limits: ReportLimits | None = None,
                 include_tournaments: bool = True, seed: int = 0, table_min_age_sec: float = 0.0) -> None:
        self.run_dir = Path(run_dir)
        self.table_min_age_sec = float(table_min_age_sec)
        self.limits = limits or ReportLimits()
        self.include_tournaments = include_tournaments
        self.seed = seed
        self._metrics = MetricsTail(self.run_dir / "metrics.jsonl",
                                    max_per_series=2 * self.limits.series_points)
        self._config_cache: tuple[tuple, Any] | None = None
        self._config_error: str | None = None
        self._ratings_cache: tuple[tuple, Any] | None = None
        self._reader = None
        self._reader_failed: str | None = None
        self._cache: dict[tuple, Any] = {}
        self.failed_sections: dict[str, str] = {}      # of the last build(): section -> error

    # ----- sources -----
    @property
    def journal_path(self) -> Path:
        from colosseum.arena.matchlog import MATCH_LOG_FILE

        return self.run_dir / MATCH_LOG_FILE

    def journal_sig(self) -> tuple:
        p = self.journal_path
        return _file_sig(p, p.with_name(p.name + "-wal"))

    def config(self):
        """The run's resolved config (None when missing or unreadable)."""
        from colosseum.core.run_dir import RESOLVED_CONFIG_FILE

        path = self.run_dir / RESOLVED_CONFIG_FILE
        sig = _file_sig(path)
        if self._config_cache is None or self._config_cache[0] != sig:
            cfg, error = None, None
            if path.exists():
                try:
                    from colosseum.core.config import load_config

                    cfg = load_config(path)
                except Exception as e:  # noqa: BLE001 - a report never fails on a config it cannot read
                    logger.warning(f"report: cannot load {path}: {e}")
                    error = f"{RESOLVED_CONFIG_FILE} cannot be read ({type(e).__name__}: {e})"
            else:
                error = f"No {RESOLVED_CONFIG_FILE} in the run dir"
            self._config_cache = (sig, cfg)
            self._config_error = None if error is None else (
                f"{error}: agent kinds, the budget and the rating inputs (anchor, fixed agents) are unknown; the "
                f"ratings use the module defaults.")
        return self._config_cache[1]

    def config_note(self) -> str | None:
        """Why the config is missing (None when it was read)."""
        self.config()
        return self._config_error

    def ratings_json(self) -> dict | None:
        """``ratings.json`` parsed, re-read only when the file changes (None when missing or malformed)."""
        path = self.run_dir / "ratings.json"
        sig = _file_sig(path)
        if self._ratings_cache is None or self._ratings_cache[0] != sig:
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = None
            self._ratings_cache = (sig, data if isinstance(data, dict) else None)
        return self._ratings_cache[1]

    def reader(self):
        """The read-only journal, or None (no ``matches.db``: an SP3 run, ``match_log.enabled: false``)."""
        if self._reader is not None:
            return self._reader
        if not self.journal_path.exists():
            return None
        try:
            from colosseum.arena.matchlog import MatchLog

            self._reader = MatchLog.open_readonly(self.journal_path)
            self._reader_failed = None
        except Exception as e:  # noqa: BLE001 - a broken journal degrades the sections, not the report
            self._reader_failed = f"the match log cannot be opened: {e}"
            return None
        return self._reader

    def _journal_note(self) -> str:
        if self._reader_failed:
            return self._reader_failed
        return "No match log in this run (an SP3 run dir, or match_log.enabled: false)."

    def table(self, sources: Sequence[str] = RUN_SOURCES):
        """The journal's comparison table for ``sources``; reloaded when the journal files changed, but not
        more often than every ``table_min_age_sec`` (the dashboard of a running run: the journal changes
        every second, a reload reads it whole)."""
        entry = self._table_entry(sources)
        return None if entry is None else entry[1]

    def _table_entry(self, sources: Sequence[str] = RUN_SOURCES) -> tuple[tuple, Any] | None:
        """``(journal signature the table was read at, table)``; the signature keys the fits of that table."""
        reader = self.reader()
        if reader is None:
            return None
        key = ("table", tuple(sources))
        sig, now = self.journal_sig(), time.monotonic()
        hit = self._cache.get(key)
        if hit is not None and (hit[0] == sig or now - hit[1] < self.table_min_age_sec):
            return hit[0], hit[2]
        table = reader.comparison_table(sources=tuple(sources))
        self._cache[key] = (sig, now, table)
        return sig, table

    def records(self, kind: str) -> list[dict]:
        self._metrics.poll()
        return self._metrics.of(kind)

    def status(self) -> tuple[str, str, Any]:
        state = read_run_state(self.run_dir)
        status, detail = effective_status(state)
        return status, detail, state

    def layout_kinds(self) -> dict[str, str]:
        """layout -> outcome kind, from ratings.json, else from the journal's first match of the layout."""
        kinds: dict[str, str] = {}
        for layout, table in ((self.ratings_json() or {}).get("layouts") or {}).items():
            kinds[layout] = str(table.get("outcome_kind", "wdl"))
        table = self.table()
        reader = self.reader()
        if table is not None:
            for layout in table.layouts():
                if layout not in kinds:
                    rows = reader.matches(layout=layout, limit=1)
                    kinds[layout] = str(rows[0].to_dict()["outcome_kind"]) if rows else "wdl"
        return dict(sorted(kinds.items()))

    def agent_kinds(self) -> dict[str, str]:
        cfg = self.config()
        return {aid: str(cfg.agent_kind(aid)) for aid in cfg.agent_ids()} if cfg is not None else {}

    def _kind_of(self, entity: str) -> str:
        from colosseum.arena.entities import lineage

        return self.agent_kinds().get(lineage(entity), "other")

    # ----- public -----
    def build(self) -> dict:
        """Every section. Sections are isolated as in the dashboard: a section whose build raises is logged with
        its traceback and drawn as a note (``failed_section``); its name and error are in ``failed_sections``."""
        self.failed_sections = {}
        sections = {}
        for name in SECTIONS:
            if name == "tournaments" and not self.include_tournaments:
                continue
            try:
                sections[name] = self.section(name)
            except Exception as e:  # noqa: BLE001 - one failing section never costs the whole report
                logger.exception(f"report: building section {name} failed")
                self.failed_sections[name] = f"{type(e).__name__}: {e}"
                sections[name] = failed_section(self.failed_sections[name])
        return {"format": REPORT_FORMAT, "title": self.title(), "sections": sections}

    def title(self) -> str:
        """The page title: the config's ``run.name``, else the run dir's name."""
        cfg = self.config()
        return str(cfg.run.name if cfg is not None and cfg.run.name else self.run_dir.name)

    def section(self, name: str, **filters: Any) -> dict:
        if name not in SECTIONS:
            raise KeyError(name)
        allowed = FILTERS.get(name, ())
        clean = {k: v for k, v in filters.items() if k in allowed and v not in (None, "")}
        return getattr(self, f"_{name}")(**clean)

    def close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self._reader = None

    # ----- summary -----
    def _summary(self) -> dict:
        from colosseum.metrics.hub import wr_vs_past_over_layouts

        notes: list[str] = []
        status, detail, state = self.status()
        if state is None:
            notes.append("No run_state.json (an SP3 run dir or a distributed role): the state is unknown.")
        cfg = self.config()
        if self.config_note():
            notes.append(self.config_note())
        system = _sorted_by(self.records("system"), "env_steps")
        last = system[-1] if system else {}
        env_steps = last.get("env_steps")
        total = cfg.training.total_timesteps if cfg is not None else None
        rj = self.ratings_json() or {}
        layouts = rj.get("layouts") or {}
        ratings = self._ratings()
        agents = []
        for aid, kind in self.agent_kinds().items():
            elo = None
            for table in layouts.values():
                if sum((table.get("games") or {}).get(aid, {}).values()):
                    elo = (table.get("elo") or {}).get(aid)
                    break
            agents.append({"agent": aid, "kind": kind, "bt": self._newest_bt(ratings, aid), "elo": elo,
                           "wr_vs_past": wr_vs_past_over_layouts(layouts, aid) if layouts else None})
        parents: list[str] = []
        reader = self.reader()
        if reader is not None:
            own = self.run_dir.name
            parents = [r.name for r in reader.runs() if r.name != own]
        retention = None
        if cfg is not None:
            from colosseum.arena.retention import describe_retention

            retention = describe_retention(cfg.checkpoint)
        return {"run": self.run_dir.name, "path": str(self.run_dir), "status": status, "status_detail": detail,
                "retention": retention,
                "pid": getattr(state, "pid", None), "exit_code": getattr(state, "exit_code", None),
                "env_steps": env_steps, "total_timesteps": total,
                "budget_frac": (env_steps / total) if env_steps is not None and total else None,
                "env_steps_per_sec": _finite(last.get("env_steps_per_sec")),
                "agents": agents, "parents": parents, "notes": notes}

    @staticmethod
    def _newest_bt(ratings: dict, agent: str) -> dict | None:
        for layout, lay in ratings["layouts"].items():
            for group in lay.get("groups", []):
                rows = [r for r in group["entities"] if r["lineage"] == agent]
                if rows:
                    row = max(rows, key=lambda r: (r["version"] is not None, r["version"] or 0))
                    return {"layout": layout, "entity": row["entity"], "rating": row["rating"], "ci": row["ci"]}
        return None

    # ----- ratings -----
    def _ratings(self, agent: str | None = None, layout: str | None = None) -> dict:
        status, _, _ = self.status()
        rj = self.ratings_json() or {}
        live = status == "running" and isinstance(rj.get("bt"), Mapping)
        notes: list[str] = []
        out_layouts: dict[str, dict] = {}
        table = self.table()
        if table is None and not live:
            notes.append(self._journal_note())
        elif not live and self.config_note():
            notes.append(self.config_note())
        if live and not rj["bt"]:
            notes.append("The run has no live rating refit yet (ratings.json has an empty 'bt' section).")
        for name, kind in self.layout_kinds().items():
            if layout and name != layout:
                continue
            if kind == "score":
                entry = self._score_layout(name)
                if entry is None and live:
                    entry = self._live_score_layout(rj["bt"].get(name) or {})
            elif live:
                entry = self._live_layout(name, rj["bt"].get(name) or {})
            else:
                entry = self._fitted_layout(name, notes)
            if entry is None:
                continue
            entry["outcome_kind"] = kind
            if agent:
                entry = self._filter_agent(entry, agent)
            out_layouts[name] = entry
        if live:
            notes.append("Live ratings (ratings.json, refit every ratings.refit_interval_sec); Laplace intervals.")
        uncertainty = "laplace" if live or not self._bootstrap_samples() else "bootstrap"
        return {"source": "live" if live else ("journal" if table is not None else "none"),
                "uncertainty": uncertainty, "env_steps": rj.get("env_steps"), "layouts": out_layouts,
                "notes": notes}

    @staticmethod
    def _filter_agent(entry: dict, agent: str) -> dict:
        entry = dict(entry)
        if "groups" in entry:
            entry["groups"] = [dict(g, entities=[e for e in g["entities"] if e["lineage"] == agent],
                                    series=[s for s in g["series"] if s["lineage"] == agent]) for g in entry["groups"]]
        if "entities" in entry:
            entry["entities"] = [e for e in entry["entities"] if e["lineage"] == agent]
        if "compositions" in entry:          # "a+a+b": sorted member entities (no '+' in entity names)
            from colosseum.arena.entities import lineage

            entry["compositions"] = [c for c in entry["compositions"]
                                     if any(lineage(m) == agent for m in c["composition"].split("+"))]
        return entry

    def _bootstrap_samples(self) -> int:
        cfg = self.config()
        return int(cfg.ratings.bootstrap_samples) if cfg is not None else 200

    def _group_payload(self, label: str, component: int, anchor: str, rows: list[dict]) -> dict:
        rows = sorted(rows, key=lambda r: -(r["rating"] if r["rating"] is not None else -1e18))
        keep = rows[: self.limits.entity_rows]
        by_lineage: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["version"] is not None:
                by_lineage[r["lineage"]].append(r)
        series = []
        for name in sorted(by_lineage):
            pts = sorted(by_lineage[name], key=lambda r: r["version"])
            idx = thin(len(pts), self.limits.series_points)
            series.append({"lineage": name, "points": [[pts[i]["version"], pts[i]["rating"], pts[i]["ci"][0],
                                                        pts[i]["ci"][1], pts[i]["games"]] for i in idx]})
        anchors = [{"entity": r["entity"], "rating": r["rating"], "ci": r["ci"]} for r in rows if r["version"] is None]
        return {"label": label, "component": component, "anchor": anchor, "entities": keep,
                "truncated": len(rows) - len(keep), "series": series, "anchors": anchors}

    def _row(self, entity: str, rating: float, ci: Sequence[float], games: int, *, p_lin=None, p_grp=None,
             lcb=None) -> dict:
        from colosseum.arena.entities import lineage

        return {"entity": entity, "lineage": lineage(entity), "version": entity_version(entity),
                "kind": self._kind_of(entity), "rating": _finite(rating), "ci": [_finite(ci[0]), _finite(ci[1])],
                "games": int(games or 0), "p_best_lineage": _finite(p_lin), "p_best_group": _finite(p_grp),
                "lcb": _finite(lcb)}

    def _live_layout(self, layout: str, payload: Mapping) -> dict:
        """ratings.json ``bt[layout]`` (T2.2's ``bt_section``, contract amendment A14): ``{group label: [rows]}``,
        rows ``{"entity", "rating", "ci_low", "ci_high", "games", "lcb", "p_best" (within the cell),
        "p_best_lineage", "component", ...}``, plus extras under keys starting with ``@`` (``@bias``, ``@anchors``
        keyed ``"<label>/<component>"``, ``@transitivity``). One payload group per (label, component)."""
        anchors = payload.get("@anchors") or {}
        groups = []
        for label, entries in payload.items():
            if str(label).startswith("@") or not isinstance(entries, list):
                continue
            by_component: dict[int, list[dict]] = defaultdict(list)
            for d in entries:
                ci = d.get("ci") or [d.get("ci_low"), d.get("ci_high")]
                by_component[int(d.get("component", 0))].append(self._row(
                    str(d["entity"]), d.get("rating"), ci, d.get("games", 0), p_lin=d.get("p_best_lineage"),
                    p_grp=d.get("p_best_group", d.get("p_best")), lcb=d.get("lcb")))
            for component, rows in sorted(by_component.items()):
                anchor = (anchors.get(f"{label}/{component}") or {}).get("anchor", "")
                groups.append(self._group_payload(str(label), component, str(anchor), rows))
        trans = payload.get("@transitivity")
        return {"kind": "bt", "groups": groups, "bias": dict(payload.get("@bias") or {}),
                "transitivity": dict(trans) if isinstance(trans, Mapping) else None}

    def _live_score_layout(self, payload: Mapping) -> dict | None:
        """A one-team layout from ratings.json ``bt[layout]["@score"]`` (a running run without a readable journal)."""
        from colosseum.arena.entities import lineage

        score = payload.get("@score")
        if not isinstance(score, Mapping):
            return None
        entities = [{"entity": str(d["entity"]), "lineage": lineage(str(d["entity"])),
                     "version": entity_version(str(d["entity"])), "kind": self._kind_of(str(d["entity"])),
                     "n": int(d.get("n") or 0), "mean": _finite(d.get("mean")),
                     "ci": [_finite(d.get("ci_low")), _finite(d.get("ci_high"))]} for d in score.get("entities") or []]
        comps = [{"composition": str(d["composition"]), "n": int(d.get("n") or 0), "mean": _finite(d.get("mean")),
                  "ci": [_finite(d.get("ci_low")), _finite(d.get("ci_high"))],
                  "homogeneous": bool(d.get("homogeneous"))} for d in score.get("compositions") or []]
        return self._score_payload(entities, comps)

    def _score_payload(self, entities: list[dict], comps: list[dict]) -> dict:
        """A score layout's entry: entities and compositions by mean score (descending), each list capped at
        ``entity_rows`` with the number left out."""
        def by_mean(r: dict) -> float:
            return -(r["mean"] if r["mean"] is not None else -1e18)

        entities, comps = sorted(entities, key=by_mean), sorted(comps, key=by_mean)
        cap = self.limits.entity_rows
        return {"kind": "score", "entities": entities[:cap], "truncated": max(0, len(entities) - cap),
                "compositions": comps[:cap], "compositions_truncated": max(0, len(comps) - cap)}

    def _fit(self, layout: str):
        """(fit, uncertainty, transitivity dict | None) of ``layout``. The fit, its bootstrap and the transitivity
        test all use ONE read of the journal (``bt.bootstrap_data``; the fitted table is ``BootstrapData.to_table``,
        as in ``--snapshots top:N``): a writer that appends between reads cannot make them disagree. Cached until
        the journal files change, but not refitted more often than every ``table_min_age_sec``."""
        from colosseum.arena import bt
        from colosseum.arena.live import rating_inputs

        key = ("fit", layout)
        sig, now = self.journal_sig(), time.monotonic()      # before the read: a later write refits next time
        hit = self._cache.get(key)
        if hit is not None and (hit[0] == sig or now - hit[1] < self.table_min_age_sec):
            return hit[2]
        cfg = self.config()
        if cfg is not None:
            options, fixed, random_bot = rating_inputs(cfg)
        else:                                  # no readable config (a note says so): module defaults
            options, fixed, random_bot = bt.DEFAULT_OPTIONS, [], None
        data = bt.bootstrap_data(self.reader(), layout, sources=RUN_SOURCES)
        table = data.to_table(layout)
        sources = (bt.DATA_SOURCE,)
        fit_ = bt.fit(table, layout, sources=sources, options=options, fixed_agents=fixed, random_bot=random_bot)
        rng = np.random.default_rng(self.seed)
        samples = self._bootstrap_samples()
        unc = (bt.bootstrap(fit_, data, samples=samples, rng=rng) if samples > 0 else bt.laplace(fit_, rng=rng))
        trans = bt.transitivity_test(table, fit_, sources=sources).to_dict()
        self._cache[key] = (sig, now, (fit_, unc, trans))
        return fit_, unc, trans

    def _fitted_layout(self, layout: str, notes: list[str]) -> dict | None:
        table = self.table()
        if table is None or layout not in table.layouts():
            return None
        fit_, unc, trans = self._fit(layout)
        groups = []
        for g in fit_.groups:
            rows = []
            for e in g.entities:
                r = fit_.rating[e]
                ci = unc.ci95.get(e) or (r - _Z95 * fit_.se[e], r + _Z95 * fit_.se[e])
                rows.append(self._row(e, r, ci, fit_.games.get(e, 0), p_lin=unc.p_best_lineage.get(e),
                                      p_grp=unc.p_best_group.get(e), lcb=unc.lcb.get(e)))
            groups.append(self._group_payload(g.group, g.component, str(g.anchor), rows))
        if not fit_.converged:
            notes.append(f"{layout}: the BT fit did not converge in {fit_.iterations} iterations.")
        return {"kind": "bt", "groups": groups, "bias": {str(k): v for k, v in fit_.bias.items()},
                "transitivity": trans}

    def _score_layout(self, layout: str) -> dict | None:
        """One-team layouts: mean score per entity and per team composition (``bt.score_summary``)."""
        table = self.table()
        if table is None:
            return None
        from colosseum.arena import bt
        from colosseum.arena.entities import lineage

        summary = bt.score_summary(table, layout, sources=RUN_SOURCES).to_dict()
        entities = [{"entity": e, "lineage": lineage(e), "version": entity_version(e), "kind": self._kind_of(e),
                     "n": int(d["n"]), "mean": _finite(d["mean"]), "ci": [_finite(d["ci_low"]), _finite(d["ci_high"])]}
                    for e, d in summary["entities"].items()]
        comps = [{"composition": c, "n": int(d["n"]), "mean": _finite(d["mean"]),
                  "ci": [_finite(d["ci_low"]), _finite(d["ci_high"])], "homogeneous": bool(d.get("homogeneous"))}
                 for c, d in summary["compositions"].items()]
        return self._score_payload(entities, comps)

    # ----- matrix -----
    def _matrix(self, agent: str | None = None, layout: str | None = None) -> dict:
        notes: list[str] = []
        table = self.table()
        if table is None:
            return {"min_games": MIN_CELL_GAMES, "layouts": {}, "notes": [self._journal_note()]}
        ratings = self._ratings()
        out: dict[str, dict] = {}
        for name, kind in self.layout_kinds().items():
            if (layout and name != layout) or name not in table.layouts():
                continue
            if kind == "score":
                entry = ratings["layouts"].get(name, {})
                out[name] = {"kind": "score", "cross_play": entry.get("compositions", []),
                             "truncated": int(entry.get("compositions_truncated", 0))}
                continue
            order = self._lineage_order(ratings["layouts"].get(name, {}))
            skipped = 0
            acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
            entity_acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
            for a, b, stat in table.pairs(name, RUN_SOURCES):
                la, lb = _team_lineage(a), _team_lineage(b)
                if la is None or lb is None:
                    skipped += 1
                    continue
                if la != lb:
                    _add(acc, la, lb, stat)
                ea, eb = _team_entity(a), _team_entity(b)
                if ea is not None and eb is not None and ea != eb:
                    _add(entity_acc, ea, eb, stat)
            axis = sorted({x for pair in acc for x in pair}, key=lambda x: order.get(x, (1, 0.0, x)))
            entry = {"kind": "bt", "lineages": {"axis": axis, "cells": _cells(axis, acc)}}
            trainable = [x for x in axis if self.agent_kinds().get(x) == "trainable"]
            target = agent if agent else (trainable[0] if trainable else None)
            entry["snapshots"] = self._snapshot_matrix(target, entity_acc, axis, order) if target else None
            if skipped:
                notes.append(f"{name}: {skipped} team pairs with members of several agents are not in the matrix.")
            out[name] = entry
        return {"min_games": MIN_CELL_GAMES, "layouts": out, "notes": notes}

    @staticmethod
    def _lineage_order(layout_ratings: Mapping) -> dict[str, tuple]:
        """Sort key per lineage: rating of its newest entity (descending), unrated last."""
        best: dict[str, tuple[int, float]] = {}
        for group in layout_ratings.get("groups", []):
            for r in group["entities"]:
                v = r["version"] if r["version"] is not None else math.inf
                if r["rating"] is not None and (r["lineage"] not in best or v > best[r["lineage"]][0]):
                    best[r["lineage"]] = (v, r["rating"])
        return {name: (0, -rating, name) for name, (_, rating) in best.items()}

    def _snapshot_matrix(self, agent: str, entity_acc, lineage_axis: list[str], order) -> dict:
        from colosseum.arena.entities import lineage

        versions = sorted({e for pair in entity_acc for e in pair if lineage(e) == agent},
                          key=lambda e: entity_version(e) or 0)
        others = [x for x in lineage_axis if x != agent]
        room = max(1, self.limits.snapshot_matrix - len(others))
        chosen = [versions[i] for i in thin(len(versions), room)]
        chosen_set = set(chosen)
        acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for (ea, eb), (n, sw, swy) in entity_acc.items():
            la, lb = lineage(ea), lineage(eb)
            xa = ea if la == agent else la
            xb = eb if lb == agent else lb
            if (la == agent and ea not in chosen_set) or (lb == agent and eb not in chosen_set) or xa == xb:
                continue
            cell = acc[(xa, xb)]
            cell[0] += n
            cell[1] += sw
            cell[2] += swy
        axis = chosen + others
        return {"agent": agent, "axis": axis, "cells": _cells(axis, acc), "thinned_from": len(versions) + len(others)}

    # ----- anchors -----
    def _anchors(self) -> dict:
        """Score of every snapshot against each anchor, by version. Pairwise layouts: the share of the pair
        weight won against the anchor's team (``metric: "pairwise"``, Wilson intervals); one-team layouts: the
        mean team score of the snapshot's team with each anchor as teammate, ``"(self)"`` for a team of the
        snapshot alone (``metric: "score"``, normal intervals, null below two games). With pulse, only pulse
        games, and ``unevaluated`` (per layout entry, and per agent of the pulse plan at the top) lists the stored
        snapshots without a pulse game."""
        cfg = self.config()
        notes: list[str] = []
        pulse = bool(cfg is not None and cfg.evaluation.enabled)
        source = "pulse" if pulse else "train"
        table = self.table()                   # the cached RUN_SOURCES table, read per source below
        if table is None:
            return {"source": "none", "biased": False, "layouts": {}, "unevaluated": {},
                    "notes": [self._journal_note()]}
        fixed = set(cfg.fixed_agent_ids()) if cfg is not None else set()
        field = set(cfg.evaluation.field if cfg.evaluation.field is not None else fixed) if pulse else fixed
        kinds = self.layout_kinds()
        layouts: dict[str, dict] = {}
        evaluated: dict[str, set[int]] = defaultdict(set)
        for name in table.layouts():
            if kinds.get(name) == "score":
                per_agent = self._score_anchors(table, name, source, field)
            else:
                per_agent = self._pairwise_anchors(table, name, source, field)
            for agent, d in per_agent.items():
                evaluated[agent].update(p[0] for series in d["anchors"].values() for p in series)
                d["anchors"] = {k: [v[i] for i in thin(len(v), self.limits.series_points)]
                                for k, v in sorted(d["anchors"].items())}
            if per_agent:
                layouts[name] = per_agent
        unevaluated: dict[str, list[int]] = {}
        if pulse:
            ev_agents = cfg.evaluation.agents            # as pulse_plan: null = every trainable agent, [] = none
            plan = list(ev_agents) if ev_agents is not None else list(cfg.get_trainable_agent_ids())
            for agent in plan:
                unevaluated[agent] = sorted(set(self._stored_versions(agent)) - evaluated[agent])
            for layout_entry in layouts.values():
                for agent, d in layout_entry.items():
                    d["unevaluated"] = sorted(set(self._stored_versions(agent)) - evaluated[agent])
            if not layouts:
                notes.append("Pulse evaluation is enabled but has no results yet.")
        elif not fixed:
            notes.append("The config has no scripted or frozen agents (anchors).")
        return {"source": source, "biased": not pulse, "layouts": layouts, "unevaluated": unevaluated,
                "notes": notes}

    def _pairwise_anchors(self, table, name: str, source: str, field: set[str]) -> dict[str, dict]:
        from colosseum.arena.entities import lineage

        acc: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for a, b, stat in table.pairs(name, (source,)):
            for x, y, flip in ((a, b, False), (b, a, True)):
                ex, ly = _team_entity(x), _team_lineage(y)
                vx = entity_version(ex) if ex is not None else None
                if vx is None or ly not in field:
                    continue
                cell = acc[(lineage(ex), vx, ly)]
                cell[0] += stat.n
                cell[1] += stat.sw
                cell[2] += (stat.sw - stat.swy) if flip else stat.swy
        per_agent: dict[str, dict] = {}
        for (agent, version, anchor), (n, sw, swy) in sorted(acc.items()):
            if sw <= 0:
                continue
            p = swy / sw
            entry = per_agent.setdefault(agent, {"metric": "pairwise", "anchors": defaultdict(list),
                                                 "unevaluated": []})
            entry["anchors"][anchor].append([version, p, *wilson(p, n), n])
        return per_agent

    def _score_anchors(self, table, name: str, source: str, field: set[str]) -> dict[str, dict]:
        """One-team layouts: a team of one versioned entity plus field members (or alone)."""
        from colosseum.arena.entities import lineage

        acc: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        for key, stat in table.scores(name, (source,)):
            members = key.entities()
            versioned = {e for e in members if entity_version(e) is not None}
            if len(versioned) != 1:
                continue
            (snapshot,) = versioned
            mates = sorted({lineage(e) for e in members if e != snapshot})
            if any(m not in field for m in mates):
                continue
            cell = acc[(lineage(snapshot), entity_version(snapshot), "+".join(mates) or "(self)")]
            cell[0] += stat.n
            cell[1] += stat.total
            cell[2] += stat.total_sq
        per_agent: dict[str, dict] = {}
        for (agent, version, anchor), (n, total, total_sq) in sorted(acc.items()):
            if n <= 0:
                continue
            mean = total / n
            lo = hi = None
            if n >= 2:
                half = _Z95 * math.sqrt(max((total_sq - n * mean * mean) / (n - 1), 0.0) / n)
                lo, hi = _finite(mean - half), _finite(mean + half)
            entry = per_agent.setdefault(agent, {"metric": "score", "anchors": defaultdict(list), "unevaluated": []})
            entry["anchors"][anchor].append([version, _finite(mean), lo, hi, int(n)])
        return per_agent

    def _stored_versions(self, agent: str) -> list[int]:
        root = self.run_dir / "checkpoints" / agent
        if not root.is_dir():
            return []
        out = []
        for d in root.iterdir():
            m = re.fullmatch(r"ckpt_v(\d+)", d.name)
            if m and d.is_dir():
                out.append(int(m.group(1)))
        return sorted(out)

    # ----- opponents -----
    def _opponents(self) -> dict:
        agents: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in _sorted_by(self.records("episodes"), "env_steps"):
            for layout, cell in (r.get("opponents") or {}).items():
                d = agents[str(r.get("agent"))].setdefault(layout, {"env_steps": [], "categories": defaultdict(list),
                                                                     "anchors": defaultdict(list)})
                i = len(d["env_steps"])
                d["env_steps"].append(r.get("env_steps"))
                for group in ("categories", "anchors"):
                    values = cell.get(group) or {}
                    for k in set(values) | set(d[group]):
                        series = d[group][k]
                        series.extend([0.0] * (i - len(series)))
                        series.append(_finite(values.get(k)) or 0.0)
        for layouts in agents.values():
            for d in layouts.values():
                idx = thin(len(d["env_steps"]), self.limits.series_points)
                d["env_steps"] = [d["env_steps"][i] for i in idx]
                for group in ("categories", "anchors"):
                    d[group] = {k: [v[i] for i in idx] for k, v in sorted(d[group].items())}
        pfsp: dict[str, dict] = {}
        for layout, table in ((self.ratings_json() or {}).get("layouts") or {}).items():
            for owner, players in (table.get("pfsp") or {}).items():
                rows = [{"player": p, "score": _finite(v.get("score")), "games": _finite(v.get("games"))}
                        for p, v in players.items()]
                rows.sort(key=lambda r: -(r["games"] or 0))
                pfsp.setdefault(layout, {})[owner] = rows[:PFSP_TOP]
        notes = [] if agents else ["No 'episodes' records with opponent shares yet."]
        return {"agents": dict(agents), "pfsp": pfsp, "notes": notes}

    # ----- training -----
    def _training(self) -> dict:
        per_agent: dict[str, list[dict]] = defaultdict(list)
        for r in self.records("train"):
            per_agent[str(r.get("agent"))].append(r)
        agents = {}
        for agent, recs in sorted(per_agent.items()):
            recs = _sorted_by(recs, "train_step")
            idx = thin(len(recs), self.limits.series_points)
            recs = [recs[i] for i in idx]
            names = sorted({k for r in recs for k, v in r.items() if k not in _RECORD_META and _finite(v) is not None
                            and not isinstance(v, bool)})
            agents[agent] = {"train_step": [r.get("train_step") for r in recs],
                             "metrics": {k: [_finite(r.get(k)) for r in recs] for k in names}}
        notes = []
        if self._metrics.bad_lines:
            notes.append(f"{self._metrics.bad_lines} malformed lines of metrics.jsonl were skipped.")
        present = {k for a in agents.values() for k in a["metrics"]}
        default = list(TRAIN_DEFAULT)
        for new, old in TRAIN_LEGACY.items():
            if new not in present and old in present:
                default.insert(default.index(new) + 1 if new in default else len(default), old)
        return {"agents": agents, "default": default, "notes": notes}

    # ----- system -----
    def _system(self) -> dict:
        recs = _sorted_by(self.records("system"), "env_steps")
        idx = thin(len(recs), self.limits.series_points)
        flat = []
        for i in idx:
            row: dict[str, float] = {}
            _flat("", {k: v for k, v in recs[i].items() if k not in _RECORD_META}, row)
            flat.append(row)
        names = sorted({k for row in flat for k in row})
        latest: dict[str, float] = {}
        if recs:
            _flat("", {k: v for k, v in recs[-1].items() if k not in _RECORD_META}, latest)
        try:
            free_mb = shutil.disk_usage(self.run_dir).free / 2**20
        except OSError:
            free_mb = None
        journal = None
        if self.journal_path.exists():
            wal = self.journal_path.with_name(self.journal_path.name + "-wal")
            journal = {"file_mb": self.journal_path.stat().st_size / 2**20,
                       "wal_mb": wal.stat().st_size / 2**20 if wal.exists() else 0.0}
            reader = self.reader()
            if reader is not None:
                sizes = _as_dict(reader.sizes())
                for k in ("counts_rows", "detail_bytes", "counts_bytes"):
                    if k in sizes:
                        journal[k] = sizes[k]
            flags = [k for k in latest if k.endswith("counts_only") or k.endswith("degraded")]
            journal["counts_only"] = any(latest[k] for k in flags)
        return {"env_steps": [recs[i].get("env_steps") for i in idx], "series": {k: [r.get(k) for r in flat]
                                                                                   for k in names},
                "default": list(SYSTEM_DEFAULT), "latest": latest, "free_disk_mb": free_mb, "journal": journal,
                "notes": [] if recs else ["No 'system' records yet."]}

    # ----- matches -----
    def _matches(self, agent: str | None = None, layout: str | None = None, source: str | None = None,
                 outcome: str | None = None) -> dict:
        filters = {"agent": agent, "layout": layout, "source": source, "outcome": outcome}
        reader = self.reader()
        if reader is None:
            return {"rows": [], "limit": self.limits.matches, "filters": filters, "notes": [self._journal_note()]}
        notes = []
        if outcome and not agent:
            notes.append("The outcome filter needs an agent; ignored.")
            outcome = None
        rows = reader.matches(agent=agent, layout=layout, source=source, outcome=outcome, limit=self.limits.matches)
        return {"rows": [r.to_dict() for r in rows], "limit": self.limits.matches, "filters": filters, "notes": notes}

    # ----- tournaments -----
    def _tournaments(self) -> dict:
        items, notes = [], []
        root = self.run_dir / "tournaments"
        if self.run_dir.joinpath("tournament.yaml").exists() and self.run_dir.joinpath("report.json").exists():
            dirs = [self.run_dir]                      # a tournament dir itself (colosseum report <dir>)
        else:
            dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        for d in dirs:
            path = d / "report.json"
            if not path.exists():
                notes.append(f"{d.name}: no report.json yet (running or interrupted).")
                continue
            try:
                from colosseum.arena.tournament.modes import report_view  # Part 2's schema -> per layout / group

                data = json.loads(path.read_text())
                if not isinstance(data, Mapping):
                    raise ValueError("not a JSON object")
                items.append({"name": d.name, "path": str(d), "report": report_view(data)})
            except (OSError, ValueError, KeyError, TypeError) as e:
                notes.append(f"{d.name}: report.json cannot be read ({type(e).__name__}: {e}).")
        return {"items": items, "notes": notes}


def _add(acc, a: str, b: str, stat) -> None:
    cell = acc[(a, b)]
    cell[0] += stat.n
    cell[1] += stat.sw
    cell[2] += stat.swy
    rev = acc[(b, a)]
    rev[0] += stat.n
    rev[1] += stat.sw
    rev[2] += stat.sw - stat.swy


def _cells(axis: Sequence[str], acc: Mapping[tuple[str, str], Sequence[float]]) -> list[list[dict | None]]:
    out = []
    for a in axis:
        row = []
        for b in axis:
            n, sw, swy = acc.get((a, b), (0, 0.0, 0.0))
            if sw <= 0:
                row.append(None)
            else:
                p = swy / sw
                row.append({"score": p, "ci": wilson(p, n), "n": int(n)})
        out.append(row)
    return out


def build_report_data(run_dir: str | Path, *, include_tournaments: bool = True,
                      limits: ReportLimits | None = None) -> dict:
    """Every section of the run's report (``ReportBuilder.build``: a failing section is a note in its place)."""
    builder = ReportBuilder(run_dir, limits=limits, include_tournaments=include_tournaments)
    try:
        return builder.build()
    finally:
        builder.close()


def section(run_dir: str | Path, name: str, **filters: Any) -> dict:
    """One section (``/api/<name>`` of the dashboard); unknown names raise KeyError."""
    builder = ReportBuilder(run_dir)
    try:
        return builder.section(name, **filters)
    finally:
        builder.close()

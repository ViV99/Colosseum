"""The match log: a bounded SQLite journal of finished matches (SP4 spec block 1).

``runs/<name>/matches.db`` holds every training, tournament and pulse match of a run. Tables:

- ``meta`` (schema version; the running size estimate and count of the detailed training matches), ``entities``
  (numeric ids of agents, networks, roles, sources, layouts, rating entities, opponent categories and
  tournaments), ``runs`` (this run and the runs it was resumed from);
- ``matches`` / ``seats`` / ``teams``: detailed matches (``uid`` unique; per seat the weights versions at the
  episode's start and end and the rating version, see ``colosseum.arena.entities``);
- ``counts``: compacted matches as team-pair sufficient statistics (``n``, ``sum w``, ``sum w*y``) and, for
  one-team layouts, team-score sums (``n``, ``sum``, ``sum of squares``) per ``(layout, source, team keys)``.
  ``comparison_table`` reads detailed matches and counts into one ``ComparisonTable``, the same before and after
  a compaction (up to float rounding of the sums).

Bounded by construction. Tournament and pulse matches stay detailed. Detailed training matches are kept within
``window_bytes``: every match row records its estimated size (``estimated_size``), ``meta`` keeps the running sum,
and ``compact`` folds the oldest into ``counts`` in batches of at most ``max_batch`` matches (one transaction each),
then runs ``PRAGMA incremental_vacuum`` and ``wal_checkpoint(TRUNCATE)``. Training matches of evicted entities are
folded first. When the journal's disk has less than ``min_free_disk_mb`` free, new training matches go straight to
``counts`` (``degraded``; a WARNING on the switch, an INFO when detail recording resumes); a tournament writes as
usual.

SQLite rules: WAL; ``auto_vacuum = INCREMENTAL`` (set before the first table); ``synchronous = NORMAL``;
``journal_size_limit``; ``busy_timeout``; one connection per thread or process; a writer's transaction is one
batch, so two writers (training and a tournament) can share the file. ``MatchLogReader`` opens read-only and every
query is its own transaction; a read of several queries that spans a compaction (``meta.compact_generation``
changed, bumped by every fold transaction) is retried. A file that is not an SQLite database is renamed
(``matches.db.corrupt-<k>``) and a new journal starts (ERROR); any other error while opening (locked, unreadable)
propagates and the file stays; a journal of another schema version is a ``MatchLogError``. A failed transaction
surfaces its own error (SQLite may already have rolled it back) and clears the id cache. ``import_parent`` runs
in one transaction and skips a parent already imported.

Order is by row id (insertion order), never by the informational ``ts``.

``MatchLogWriter`` (training main process, SP4 T1.3) is a thread with its own ``MatchLog`` connection: the main
loop submits results to its bounded queue (a full queue drops and counts them), the thread appends them in batches,
feeds the in-memory ``ComparisonTable`` of the live ratings with what it journaled, compacts every
``compact_interval_sec``, and flushes its queue on ``stop``. A write error is an ERROR in the log and a count,
never an exception in the training process.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from colosseum.arena.comparisons import (
    ComparisonTable,
    PairStat,
    ScoreStat,
    TeamKey,
    pairs_of,
    score_of,
    team_keys,
)
from colosseum.arena.entities import EntityKey, entity_key, rating_version
from colosseum.core.types import MATCH_SOURCES, MatchResult, SeatResult, TeamResult

logger = logging.getLogger(__name__)

MATCH_LOG_FILE = "matches.db"
SCHEMA_VERSION = 1
# The sources of a run's own ratings (live ratings, top-k, the report's ratings): tournaments rate only their own
# matches (spec block 2). Defined here once (contract amendment A18).
RUN_SOURCES = ("train", "pulse")
BUSY_TIMEOUT_MS = 10_000
JOURNAL_SIZE_LIMIT = 64 * 1024 * 1024
READ_BATCH = 10_000
MAX_BATCH = 10_000
# Bound parameters per statement: SQLite before 3.32 allows at most 999.
BIND_CHUNK = 500
# A reader retries a table read this often when a compaction committed in between (``compact_generation``),
# sleeping ``CONSISTENT_READ_BACKOFF_SEC * attempt`` before each retry so that a running compaction can finish.
CONSISTENT_READ_ATTEMPTS = 5
CONSISTENT_READ_BACKOFF_SEC = 0.05
# Estimated bytes of one detailed match: the rows' payload (the uid twice: its UNIQUE index repeats it) times an
# overhead factor for B-tree pages and the indexes. Calibrated on 1v1, 2v2 and 4-team FFA matches: the estimate
# is within 7 % of the pages SQLite really uses (T1.2).
MATCH_ROW_BYTES = 48
SEAT_ROW_BYTES = 40
TEAM_ROW_BYTES = 20
INDEX_OVERHEAD = 1.2
_PAIR, _SCORE = 0, 1
_WORKER_RE = re.compile(r"^w(\d+)_e")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, "
    "UNIQUE (kind, name))",
    "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
    "parent_run_id TEXT, game TEXT NOT NULL, config_hash TEXT NOT NULL, checkpoint_interval INTEGER NOT NULL, "
    "created REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY, uid TEXT NOT NULL UNIQUE, run INTEGER, "
    "source INTEGER NOT NULL, layout INTEGER NOT NULL, outcome_kind TEXT NOT NULL, length INTEGER NOT NULL, "
    "seed INTEGER, env_steps INTEGER, block TEXT, tournament INTEGER, worker INTEGER, ts REAL NOT NULL, "
    "replay TEXT, size INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS matches_by_source ON matches (source, id)",
    "CREATE INDEX IF NOT EXISTS matches_by_tournament ON matches (tournament, block) WHERE tournament IS NOT NULL",
    "CREATE TABLE IF NOT EXISTS seats (match INTEGER NOT NULL, seat INTEGER NOT NULL, team INTEGER NOT NULL, "
    "role INTEGER NOT NULL, agent INTEGER NOT NULL, network INTEGER NOT NULL, entity INTEGER NOT NULL, "
    "version_start INTEGER, version_end INTEGER, rating_version INTEGER, reward REAL NOT NULL, "
    "eliminated_step INTEGER, category INTEGER NOT NULL, PRIMARY KEY (match, seat)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS seats_by_agent ON seats (agent, rating_version, match)",
    "CREATE INDEX IF NOT EXISTS seats_by_entity ON seats (entity, match)",
    "CREATE TABLE IF NOT EXISTS teams (match INTEGER NOT NULL, team INTEGER NOT NULL, rank REAL NOT NULL, "
    "score REAL NOT NULL, PRIMARY KEY (match, team)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS counts (id INTEGER PRIMARY KEY, kind INTEGER NOT NULL, layout INTEGER NOT NULL, "
    "source INTEGER NOT NULL, key_a TEXT NOT NULL, key_b TEXT NOT NULL, n INTEGER NOT NULL, s1 REAL NOT NULL, "
    "s2 REAL NOT NULL, UNIQUE (kind, layout, source, key_a, key_b))",
)
_UPSERT_COUNT = (
    "INSERT INTO counts (kind, layout, source, key_a, key_b, n, s1, s2) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (kind, layout, source, key_a, key_b) DO UPDATE SET n = n + excluded.n, s1 = s1 + excluded.s1, "
    "s2 = s2 + excluded.s2"
)
_MATCH_COLUMNS = ("uid, run, source, layout, outcome_kind, length, seed, env_steps, block, tournament, worker, "
                  "ts, replay, size")
_SEAT_COLUMNS = ("seat, team, role, agent, network, entity, version_start, version_end, rating_version, reward, "
                 "eliminated_step, category")


class MatchLogError(Exception):
    """A journal this code cannot use (another schema version, not a match log)."""


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    name: str
    parent_run_id: str | None
    game: str
    config_hash: str
    checkpoint_interval: int


@dataclass
class AppendStats:
    detailed: int = 0
    counted: int = 0            # folded straight into counts (disk guard)
    duplicates: int = 0         # uid already in the journal: ignored
    accepted: list[bool] = field(default_factory=list)   # per result: journaled (detailed or counted)
    degraded: bool = False


@dataclass
class CompactStats:
    folded: int = 0             # detailed training matches folded into counts
    evicted_folded: int = 0     # of which because they had an evicted entity
    batches: int = 0
    interrupted: bool = False


@dataclass
class ImportStats:
    counts_rows: int = 0
    folded: int = 0             # the parent's detailed training matches, folded into counts
    copied: int = 0             # the parent's tournament and pulse matches, copied in detail
    skipped: bool = False       # the parent was imported before (nothing done)


@dataclass(frozen=True)
class LogSizes:
    file_bytes: int
    wal_bytes: int
    detail_bytes: int           # estimate of the detailed training matches
    detail_matches: int         # detailed training matches
    other_matches: int          # detailed tournament and pulse matches
    counts_rows: int
    free_disk_bytes: int
    degraded: bool


@dataclass(frozen=True)
class SeatRow:
    seat: int
    team: int
    role: str
    agent_id: str
    network_id: str
    entity: EntityKey
    version_start: int | None
    version_end: int | None
    rating_version: int | None
    reward: float
    eliminated_step: int | None
    category: str


@dataclass(frozen=True)
class MatchRow:
    id: int
    uid: str
    run_id: str | None
    source: str
    layout: str
    outcome_kind: str
    length: int
    seed: int | None
    env_steps: int | None
    block: str | None
    tournament: str | None
    worker: int | None
    ts: float
    seats: tuple[SeatRow, ...]
    teams: tuple[tuple[int, float, float], ...]     # (team, rank, score)
    replay: str | None = None                        # a replay link, when the env gave one (spec block 5)

    def to_dict(self) -> dict:
        """JSON-ready form (report, dashboard; contract amendment A7): the match's columns (no ``ts``: order is by
        row id), its seats and its teams."""
        return {"uid": self.uid, "run_id": self.run_id, "source": self.source, "layout": self.layout,
                "outcome_kind": self.outcome_kind, "episode_length": self.length, "seed": self.seed,
                "env_steps": self.env_steps, "block": self.block, "tournament": self.tournament,
                "worker": self.worker, "replay": self.replay,
                "seats": [{"seat": s.seat, "team": s.team, "role": s.role, "agent_id": s.agent_id,
                           "network_id": s.network_id, "entity": s.entity, "category": s.category,
                           "version_start": s.version_start, "version_end": s.version_end,
                           "rating_version": s.rating_version, "reward": s.reward,
                           "eliminated_step": s.eliminated_step} for s in self.seats],
                "teams": [{"team": t, "rank": r, "score": sc} for t, r, sc in self.teams]}

    def to_result(self) -> tuple[MatchResult, list[int | None]]:
        """The ``MatchResult`` back (``match_id`` = ``uid``; a seat's ``source`` = its opponent category) and the
        seats' rating versions in seat order (``ComparisonTable.add_result`` / ``match_pairs`` take both)."""
        seats = [SeatResult(seat=s.seat, role=s.role, team=s.team, agent_id=s.agent_id, network_id=s.network_id,
                            reward=s.reward, eliminated_step=s.eliminated_step, source=s.category,
                            version_start=s.version_start, version_end=s.version_end) for s in self.seats]
        result = MatchResult(match_id=self.uid, layout=self.layout, outcome_kind=self.outcome_kind, seats=seats,
                             teams=[TeamResult(team=t, rank=r, score=sc) for t, r, sc in self.teams],
                             episode_length=self.length, uid=self.uid, source=self.source, seed=self.seed,
                             env_steps=self.env_steps, block=self.block)
        return result, [s.rating_version for s in self.seats]


def default_disk_free(directory: Path) -> int:
    """Free bytes on the disk of ``directory``."""
    return int(shutil.disk_usage(directory).free)


def estimated_size(result: MatchResult) -> int:
    """Estimated bytes of a detailed match (module docstring)."""
    payload = MATCH_ROW_BYTES + 2 * len(result.uid) + len(result.block or "")
    payload += SEAT_ROW_BYTES * len(result.seats) + TEAM_ROW_BYTES * len(result.teams)
    return int(payload * INDEX_OVERHEAD)


def _connect(path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, isolation_level=None,
                               timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def _is_corrupt(error: sqlite3.DatabaseError) -> bool:
    """A file that is not (or no longer) an SQLite database; other errors (locked, I/O, permissions) are not."""
    text = str(error).lower()
    return "not a database" in text or "malformed" in text


def _rollback(conn: sqlite3.Connection) -> None:
    """Roll back the open transaction, if any: SQLite rolls back by itself on some errors (SQLITE_FULL, IOERR,
    NOMEM, some BUSY), and an explicit ROLLBACK then fails and would mask the original error."""
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error as e:
            logger.debug(f"Match log: ROLLBACK failed after an error ({e})")


def _chunks(ids: Sequence[int], size: int = BIND_CHUNK) -> Iterator[Sequence[int]]:
    for k in range(0, len(ids), size):
        yield ids[k:k + size]


def _worker_of(match_id: str) -> int | None:
    match = _WORKER_RE.match(match_id or "")
    return None if match is None else int(match.group(1))


def _add_detail(table: ComparisonTable, layout: str, source: str, keys: Mapping[int, TeamKey],
                ranks: Mapping[int, float], scores: Mapping[int, float]) -> None:
    pairs = pairs_of(keys, ranks)
    if pairs:
        for a, b, w, y in pairs:
            table.add(layout, source, a, b, PairStat(1, w, w * y))
        return
    score = score_of(keys, scores)
    if score is not None:
        key, value = score
        table.add_score(layout, source, key, ScoreStat(1, value, value * value))


def _fold_into(acc: dict, layout: str, source: str, keys: Mapping[int, TeamKey], ranks: Mapping[int, float],
               scores: Mapping[int, float]) -> None:
    """Add one match to ``acc`` = ``{(kind, layout, source, TeamKey a, TeamKey b or None): [n, s1, s2]}``."""
    pairs = pairs_of(keys, ranks)
    if pairs:
        for a, b, w, y in pairs:
            cell = acc.setdefault((_PAIR, layout, source, a, b), [0, 0.0, 0.0])
            cell[0] += 1
            cell[1] += w
            cell[2] += w * y
        return
    score = score_of(keys, scores)
    if score is not None:
        key, value = score
        cell = acc.setdefault((_SCORE, layout, source, key, None), [0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += value
        cell[2] += value * value


class _Store:
    """Reading code shared by the writer and the reader (one connection, an id <-> name cache)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ids: dict[tuple[str, str], int] = {}
        self._names: dict[int, str] = {}

    def _lookup(self, kind: str, name: str) -> int | None:
        key = (kind, name)
        found = self._ids.get(key)
        if found is not None:
            return found
        row = self._conn.execute("SELECT id FROM entities WHERE kind = ? AND name = ?", key).fetchone()
        if row is None:
            return None
        self._ids[key] = int(row[0])
        self._names[int(row[0])] = name
        return int(row[0])

    def _name(self, entity_id: int | None) -> str | None:
        if entity_id is None:
            return None
        name = self._names.get(entity_id)
        if name is None:
            row = self._conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                raise MatchLogError(f"entity id {entity_id} is not in the entities table")
            name = self._names[entity_id] = str(row[0])
        return name

    def _meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def _decode_key(self, text: str) -> TeamKey:
        """A ``counts`` key ``"<team>|<entity id>:<role id>,..."`` back to a ``TeamKey``."""
        team, _, members = text.partition("|")
        pairs = []
        for item in members.split(","):
            entity, _, role = item.partition(":")
            pairs.append((self._name(int(entity)), self._name(int(role))))
        return TeamKey.of(int(team), pairs)

    def _match_filter(self, sources: Collection[str], tournament: str | None) -> tuple[str, list]:
        ids = [i for s in sources if (i := self._lookup("source", s)) is not None]
        if not ids:
            return "0", []
        clause = f"source IN ({','.join('?' * len(ids))})"
        args: list = list(ids)
        if tournament is None:
            clause += " AND tournament IS NULL"
        else:
            tid = self._lookup("tournament", tournament)
            if tid is None:
                return "0", []
            clause += " AND tournament = ?"
            args.append(tid)
        return clause, args

    def _iter_detail(self, where: str, args: Sequence) -> Iterator[tuple[int, str, str, dict[int, TeamKey],
                                                                         dict[int, float], dict[int, float]]]:
        """``(match id, layout, source, team keys, ranks, scores)`` of the detailed matches that ``where`` (on
        ``matches``) selects, in id order, read in batches of ``READ_BATCH`` (each query is one transaction)."""
        last = 0
        while True:
            heads = self._conn.execute(
                f"SELECT id, layout, source FROM matches WHERE id > ? AND ({where}) ORDER BY id LIMIT ?",
                (last, *args, READ_BATCH)).fetchall()
            if not heads:
                return
            lo, hi = heads[0][0], heads[-1][0]
            members: dict[int, dict[int, list[tuple[str, str]]]] = {}
            for match, team, entity, role in self._conn.execute(
                    "SELECT match, team, entity, role FROM seats WHERE match BETWEEN ? AND ? ORDER BY match, seat",
                    (lo, hi)):
                members.setdefault(match, {}).setdefault(team, []).append((self._name(entity), self._name(role)))
            ranks: dict[int, dict[int, float]] = {}
            scores: dict[int, dict[int, float]] = {}
            for match, team, rank, score in self._conn.execute(
                    "SELECT match, team, rank, score FROM teams WHERE match BETWEEN ? AND ?", (lo, hi)):
                ranks.setdefault(match, {})[team] = float(rank)
                scores.setdefault(match, {})[team] = float(score)
            for match, layout, source in heads:
                keys = {t: TeamKey.of(t, m) for t, m in sorted(members.get(match, {}).items())}
                yield match, self._name(layout), self._name(source), keys, ranks.get(match, {}), scores.get(match, {})
            last = hi

    def _generation(self) -> int:
        return int(self._meta("compact_generation", "0"))

    def _consistent(self, read: Callable[[], object], what: str):
        """Run ``read`` (several queries, each its own transaction) until no compaction committed during it
        (``compact_generation`` unchanged): a compaction moves matches from detail to counts, so a read across one
        can miss them. At most ``CONSISTENT_READ_ATTEMPTS`` attempts with a growing back-off; then the last read
        is returned with a WARNING (it may undercount)."""
        overlapped = 0
        for attempt in range(CONSISTENT_READ_ATTEMPTS):
            if attempt:
                time.sleep(CONSISTENT_READ_BACKOFF_SEC * attempt)
            before = self._generation()
            out = read()
            after = self._generation()
            if after == before:
                return out
            overlapped += after - before
        logger.warning(f"Match log: the {what} read overlapped {overlapped} compaction batches in "
                       f"{CONSISTENT_READ_ATTEMPTS} attempts; the result may undercount until the next read")
        return out

    def _table(self, sources: Collection[str], tournament: str | None) -> ComparisonTable:
        return self._consistent(lambda: self._read_table(sources, tournament), "comparison table")

    def _read_table(self, sources: Collection[str], tournament: str | None) -> ComparisonTable:
        table = ComparisonTable()
        src_ids = [i for s in sources if (i := self._lookup("source", s)) is not None]
        if tournament is None and src_ids:
            for kind, layout, source, key_a, key_b, n, s1, s2 in self._conn.execute(
                    f"SELECT kind, layout, source, key_a, key_b, n, s1, s2 FROM counts "
                    f"WHERE source IN ({','.join('?' * len(src_ids))}) ORDER BY id", src_ids):
                if kind == _PAIR:
                    table.add(self._name(layout), self._name(source), self._decode_key(key_a),
                              self._decode_key(key_b), PairStat(int(n), float(s1), float(s2)))
                else:
                    table.add_score(self._name(layout), self._name(source), self._decode_key(key_a),
                                    ScoreStat(int(n), float(s1), float(s2)))
        where, args = self._match_filter(sources, tournament)
        for _, layout, source, keys, ranks, scores in self._iter_detail(where, args):
            _add_detail(table, layout, source, keys, ranks, scores)
        return table

    def bootstrap_units(self, layout: str, sources: Collection[str]) -> tuple[
            list[tuple[TeamKey, TeamKey, PairStat]],
            list[tuple[str | None, list[tuple[TeamKey, TeamKey, float, float]]]]]:
        """The resampling units of ``layout`` outside tournaments (T2.1's ``bt.bootstrap_data``, spec block 2): the
        ``counts`` rows of ``sources`` and, per detailed match of ``sources`` in id order, its block id (None in
        training) and its team-pair contributions ``(a, b, weight, outcome of a)``."""
        return self._consistent(lambda: self._read_bootstrap(layout, sources), "bootstrap units")

    def _read_bootstrap(self, layout: str, sources: Collection[str]) -> tuple[list, list]:
        lid = self._lookup("layout", layout)
        src_ids = [i for s in sources if (i := self._lookup("source", s)) is not None]
        if lid is None or not src_ids:
            return [], []
        marks = ",".join("?" * len(src_ids))
        counts = [(self._decode_key(key_a), self._decode_key(key_b), PairStat(int(n), float(s1), float(s2)))
                  for key_a, key_b, n, s1, s2 in self._conn.execute(
                      f"SELECT key_a, key_b, n, s1, s2 FROM counts WHERE kind = ? AND layout = ? "
                      f"AND source IN ({marks}) ORDER BY id", (_PAIR, lid, *src_ids))]
        where = f"layout = ? AND source IN ({marks}) AND tournament IS NULL"
        args = (lid, *src_ids)
        blocks = {int(i): str(b) for i, b in self._conn.execute(
            f"SELECT id, block FROM matches WHERE {where} AND block IS NOT NULL", args)}
        details = []
        for match, _, _, keys, ranks, _ in self._iter_detail(where, args):
            pairs = pairs_of(keys, ranks)
            if pairs:
                details.append((blocks.get(match), pairs))
        return counts, details

    def _sizes(self, path: Path, free: int, degraded: bool) -> LogSizes:
        wal = Path(f"{path}-wal")
        total = int(self._conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        detail_matches = int(self._meta("detail_train_matches", "0"))
        return LogSizes(
            file_bytes=path.stat().st_size if path.exists() else 0,
            wal_bytes=wal.stat().st_size if wal.exists() else 0,
            detail_bytes=int(self._meta("detail_train_bytes", "0")),
            detail_matches=detail_matches,
            other_matches=total - detail_matches,
            counts_rows=int(self._conn.execute("SELECT COUNT(*) FROM counts").fetchone()[0]),
            free_disk_bytes=int(free),
            degraded=degraded,
        )


class MatchLog(_Store):
    """A writer connection to a run's journal (module docstring).

    ``run``: the training run this connection writes for (registered in ``runs``; its ``checkpoint_interval`` is
    the width of the version buckets); None for a writer that is not a training run (a tournament): its matches
    have no run, and their rating versions come from the network ids only. ``disk_free_fn(directory) -> bytes``
    is the disk guard's probe (tests pass their own).
    """

    def __init__(self, path: str | Path, run: RunInfo | None, *,
                 disk_free_fn: Callable[[Path], int] = default_disk_free, min_free_disk_mb: float = 2048.0) -> None:
        self.path = Path(path)
        self.run = run
        self._disk_free_fn = disk_free_fn
        self._min_free = int(float(min_free_disk_mb) * 1024 * 1024)
        self._degraded = False
        self._interval = int(run.checkpoint_interval) if run is not None else 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(self._open())
        try:
            self._run_row = self._register_run(run) if run is not None else None
        except BaseException:
            self._conn.close()
            raise

    # ---- opening ---------------------------------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        try:
            return self._open_once()
        except sqlite3.DatabaseError as e:
            if not _is_corrupt(e):
                raise
            broken = self._quarantine()
            logger.error(f"Match log {self.path} is not a readable SQLite database ({e}); moved it to {broken} and "
                         f"started a new journal")
            return self._open_once()

    def _open_once(self) -> sqlite3.Connection:
        new = not self.path.exists() or self.path.stat().st_size == 0
        conn = _connect(self.path, readonly=False)
        try:
            if new:
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL")    # only takes effect before the first table
            conn.execute("PRAGMA journal_mode = WAL").fetchall()
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}").fetchall()
            self._create_schema(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _quarantine(self) -> Path:
        k = 1
        while Path(f"{self.path}.corrupt-{k}").exists():
            k += 1
        target = Path(f"{self.path}.corrupt-{k}")
        os.replace(self.path, target)
        for suffix in ("-wal", "-shm"):
            side = Path(f"{self.path}{suffix}")
            if side.exists():
                os.replace(side, Path(f"{target}{suffix}"))
        return target

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                                 [("schema_version", str(SCHEMA_VERSION)), ("detail_train_bytes", "0"),
                                  ("detail_train_matches", "0"), ("compact_generation", "0")])
            elif int(row[0]) != SCHEMA_VERSION:
                raise MatchLogError(f"match log {self.path} has schema version {row[0]}; this Colosseum reads "
                                    f"version {SCHEMA_VERSION} (move the file away to start a new journal)")
            conn.execute("COMMIT")
        except BaseException:
            _rollback(conn)
            raise

    def _register_run(self, run: RunInfo) -> int:
        def register() -> int:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                "created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run.run_id, run.name, run.parent_run_id, run.game, run.config_hash, int(run.checkpoint_interval),
                 time.time()))
            return int(self._conn.execute("SELECT id FROM runs WHERE run_id = ?", (run.run_id,)).fetchone()[0])
        return self._transaction(register)

    @classmethod
    def open_readonly(cls, path: str | Path) -> MatchLogReader:
        return MatchLogReader(path)

    # ---- writing helpers ---------------------------------------------------------------------------------------

    def _transaction(self, body: Callable[[], object]):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            out = body()
            self._conn.execute("COMMIT")
        except BaseException:
            self._ids.clear()          # ids inserted by the rolled-back transaction are gone
            self._names.clear()
            _rollback(self._conn)      # the original error propagates, never a failed ROLLBACK
            raise
        return out

    def _id(self, kind: str, name: str) -> int:
        found = self._lookup(kind, name)
        if found is not None:
            return found
        self._conn.execute("INSERT OR IGNORE INTO entities (kind, name) VALUES (?, ?)", (kind, name))
        found = self._lookup(kind, name)
        assert found is not None
        return found

    def _encode_key(self, key: TeamKey) -> str:
        return f"{key.team}|" + ",".join(f"{self._id('entity', e)}:{self._id('role', r)}" for e, r in key.members)

    def _write_counts(self, acc: Mapping) -> None:
        """Upsert ``{(kind, layout, source, TeamKey a, TeamKey b or None): [n, s1, s2]}`` into ``counts``."""
        self._conn.executemany(_UPSERT_COUNT, [
            (kind, self._id("layout", layout), self._id("source", source), self._encode_key(a),
             "" if b is None else self._encode_key(b), n, s1, s2)
            for (kind, layout, source, a, b), (n, s1, s2) in acc.items()])

    def _bump_detail(self, nbytes: int, nmatches: int) -> None:
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key = 'detail_train_bytes'",
                           (int(nbytes),))
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key = 'detail_train_matches'",
                           (int(nmatches),))

    def rating_versions(self, result: MatchResult) -> list[int | None]:
        """The rating version of every seat of ``result`` (``entities.rating_version`` with this run's
        ``checkpoint.interval``)."""
        return [rating_version(s.network_id, s.version_start, self._interval) for s in result.seats]

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _check_disk(self) -> bool:
        free = self._disk_free_fn(self.path.parent)
        low = free < self._min_free
        if low and not self._degraded:
            logger.warning(f"Match log: {free / 2**20:.0f} MB free on the journal's disk, below "
                           f"match_log.min_free_disk_mb ({self._min_free / 2**20:.0f}); new training matches go to "
                           f"the compacted counts only until space is back")
        elif not low and self._degraded:
            logger.info(f"Match log: {free / 2**20:.0f} MB free again; detailed recording of training matches "
                        f"resumes")
        self._degraded = low
        return low

    # ---- appending ---------------------------------------------------------------------------------------------

    def append(self, results: Sequence[MatchResult], *, tournament: str | None = None) -> AppendStats:
        """Journal ``results`` in one transaction (module docstring); ``tournament``: the id of the tournament
        they belong to. A uid already present is a duplicate (ignored, counted). ValueError (nothing written)
        for an unknown source or an empty uid."""
        stats = AppendStats()
        if not results:
            return stats
        for result in results:
            if result.source not in MATCH_SOURCES:
                raise ValueError(f"match {result.uid or result.match_id!r}: source {result.source!r} is not one of "
                                 f"{MATCH_SOURCES}")
            if not result.uid:
                raise ValueError(f"match {result.match_id!r} has no uid")
        degraded = tournament is None and self._check_disk()
        stats.degraded = self._degraded

        def body() -> None:
            tid = self._id("tournament", tournament) if tournament is not None else None
            train_bytes = train_matches = 0
            for result in results:
                versions = self.rating_versions(result)
                is_train = result.source == "train" and tournament is None
                if degraded and is_train:
                    acc: dict = {}
                    keys = team_keys(result, versions)
                    _fold_into(acc, result.layout, result.source, keys, {t.team: t.rank for t in result.teams},
                               {t.team: t.score for t in result.teams})
                    self._write_counts(acc)
                    stats.counted += 1
                    stats.accepted.append(True)
                    continue
                size = estimated_size(result)
                cur = self._conn.execute(
                    f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    f"NULL, ?)",
                    (result.uid, self._run_row, self._id("source", result.source), self._id("layout", result.layout),
                     result.outcome_kind, int(result.episode_length), result.seed, result.env_steps, result.block,
                     tid, _worker_of(result.match_id), time.time(), size))
                if cur.rowcount == 0:
                    stats.duplicates += 1
                    stats.accepted.append(False)
                    continue
                match = int(cur.lastrowid)
                self._conn.executemany(
                    f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(match, s.seat, s.team, self._id("role", s.role), self._id("agent", s.agent_id),
                      self._id("network", s.network_id), self._id("entity", entity_key(s.agent_id, v)),
                      s.version_start, s.version_end, v, float(s.reward), s.eliminated_step,
                      self._id("category", s.source)) for s, v in zip(result.seats, versions, strict=True)])
                self._conn.executemany("INSERT INTO teams (match, team, rank, score) VALUES (?, ?, ?, ?)",
                                       [(match, t.team, float(t.rank), float(t.score)) for t in result.teams])
                if is_train:
                    train_bytes += size
                    train_matches += 1
                stats.detailed += 1
                stats.accepted.append(True)
            if train_matches:
                self._bump_detail(train_bytes, train_matches)

        self._transaction(body)
        return stats

    # ---- compaction --------------------------------------------------------------------------------------------

    def _fold_matches(self, ids: Sequence[int]) -> None:
        """Fold the detailed matches ``ids`` into counts and delete them, in one transaction (parameters bound in
        chunks of ``BIND_CHUNK``); bumps ``compact_generation`` so that readers retry a read across it."""
        def body() -> None:
            acc: dict = {}
            for chunk in _chunks(ids):
                marks = ",".join("?" * len(chunk))
                for _, layout, source, keys, ranks, scores in self._iter_detail(f"id IN ({marks})", chunk):
                    _fold_into(acc, layout, source, keys, ranks, scores)
            self._write_counts(acc)
            train = self._lookup("source", "train")
            nbytes = nmatches = 0
            for chunk in _chunks(ids):
                marks = ",".join("?" * len(chunk))
                size, count = self._conn.execute(
                    f"SELECT COALESCE(SUM(size), 0), COUNT(*) FROM matches WHERE id IN ({marks}) AND source = ? "
                    f"AND tournament IS NULL", (*chunk, train)).fetchone()
                nbytes, nmatches = nbytes + int(size), nmatches + int(count)
                for table, column in (("seats", "match"), ("teams", "match"), ("matches", "id")):
                    self._conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", chunk)
            self._bump_detail(-nbytes, -nmatches)
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('compact_generation', '1') ON CONFLICT (key) DO UPDATE SET "
                "value = CAST(value AS INTEGER) + 1")

        self._transaction(body)

    def compact(self, *, window_bytes: int, max_batch: int = MAX_BATCH, evicted: Collection[EntityKey] = (),
                should_stop: Callable[[], bool] = lambda: False) -> CompactStats:
        """Fold the training matches of ``evicted`` entities, then the oldest detailed training matches beyond
        ``window_bytes``, into counts, in batches of at most ``max_batch`` (<= 10 000) matches; then give the
        space back. Stops between batches once ``should_stop()`` (``CompactStats.interrupted``)."""
        stats = CompactStats()
        train = self._lookup("source", "train")
        if train is None:
            return stats
        max_batch = max(1, min(int(max_batch), MAX_BATCH))
        evicted_ids = [i for e in sorted(set(evicted)) if (i := self._lookup("entity", e)) is not None]
        for k in range(0, len(evicted_ids), 500):
            chunk = evicted_ids[k:k + 500]
            marks = ",".join("?" * len(chunk))
            while not stats.interrupted:
                if should_stop():
                    stats.interrupted = True
                    break
                ids = [int(r[0]) for r in self._conn.execute(
                    f"SELECT DISTINCT s.match FROM seats s JOIN matches m ON m.id = s.match "
                    f"WHERE s.entity IN ({marks}) AND m.source = ? AND m.tournament IS NULL ORDER BY s.match LIMIT ?",
                    (*chunk, train, max_batch))]
                if not ids:
                    break
                self._fold_matches(ids)
                stats.folded += len(ids)
                stats.evicted_folded += len(ids)
                stats.batches += 1
        while not stats.interrupted:
            nbytes = int(self._meta("detail_train_bytes", "0"))
            nmatches = int(self._meta("detail_train_matches", "0"))
            if nbytes <= window_bytes or nmatches <= 0:
                break
            if should_stop():
                stats.interrupted = True
                break
            needed = max(1, math.ceil((nbytes - window_bytes) / (nbytes / nmatches)))
            ids = [int(r[0]) for r in self._conn.execute(
                "SELECT id FROM matches WHERE source = ? AND tournament IS NULL ORDER BY id LIMIT ?",
                (train, min(max_batch, needed)))]
            if not ids:
                break
            self._fold_matches(ids)
            stats.folded += len(ids)
            stats.batches += 1
        if stats.folded:
            self._conn.execute("PRAGMA incremental_vacuum").fetchall()    # one step per fetched row
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        return stats

    # ---- resume ------------------------------------------------------------------------------------------------

    def import_parent(self, parent_path: str | Path) -> ImportStats:
        """Run-dir resume (spec block 1): copy the parent journal's ``runs`` rows, counts, tournament and pulse
        matches; fold its detailed training matches into this journal's counts (their detail stays in the
        parent's journal). One transaction: a failure leaves this journal unchanged. The transaction holds the
        journal's write lock for the whole import, so it must run at start-up, before any other writer opens the
        journal. Idempotent: a parent already
        imported (marker ``imported:<parent run id>`` in ``meta``) is skipped with an INFO line
        (``ImportStats.skipped``)."""
        stats = ImportStats()
        with MatchLogReader(parent_path) as parent:
            runs = parent._conn.execute("SELECT run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                                        "created FROM runs ORDER BY id").fetchall()
            identity = runs[0][0] if runs else f"path:{Path(parent_path).resolve()}"
            marker = f"imported:{identity}"

            def body() -> bool:
                if self._conn.execute("SELECT 1 FROM meta WHERE key = ?", (marker,)).fetchone() is not None:
                    return False
                self._conn.executemany(
                    "INSERT OR IGNORE INTO runs (run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                    "created) VALUES (?, ?, ?, ?, ?, ?, ?)", runs)
                stats.counts_rows = self._copy_counts(parent)
                train = parent._lookup("source", "train")
                if train is not None:
                    acc: dict = {}
                    for _, layout, source, keys, ranks, scores in parent._iter_detail(
                            "source = ? AND tournament IS NULL", (train,)):
                        _fold_into(acc, layout, source, keys, ranks, scores)
                        stats.folded += 1
                    self._write_counts(acc)
                stats.copied = self._copy_detail(parent, train)
                self._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (marker, str(time.time())))
                return True

            if not self._transaction(body):
                logger.info(f"Match log: the journal of run {identity} ({parent_path}) is already imported; skipped")
                return ImportStats(skipped=True)
        return stats

    def _copy_counts(self, parent: MatchLogReader) -> int:
        """Add the parent's ``counts`` rows (re-encoded with this journal's ids); returns how many were read."""
        rows_read = last = 0
        while True:
            rows = parent._conn.execute(
                "SELECT id, kind, layout, source, key_a, key_b, n, s1, s2 FROM counts WHERE id > ? ORDER BY id "
                "LIMIT ?", (last, READ_BATCH)).fetchall()
            if not rows:
                return rows_read
            self._write_counts({(kind, parent._name(layout), parent._name(source), parent._decode_key(key_a),
                                 parent._decode_key(key_b) if key_b else None): (n, s1, s2)
                                for _, kind, layout, source, key_a, key_b, n, s1, s2 in rows})
            rows_read += len(rows)
            last = rows[-1][0]

    def _copy_detail(self, parent: MatchLogReader, train: int | None) -> int:
        """Copy the parent's detailed tournament and pulse matches (inside the caller's transaction); returns how
        many were new here."""
        where = "1" if train is None else "(source != ? OR tournament IS NOT NULL)"
        args: tuple = () if train is None else (train,)
        run_ids = {r[0]: r[1] for r in parent._conn.execute("SELECT id, run_id FROM runs")}
        copied = last = 0
        while True:
            heads = parent._conn.execute(
                f"SELECT id, {_MATCH_COLUMNS} FROM matches WHERE id > ? AND {where} ORDER BY id LIMIT ?",
                (last, *args, READ_BATCH)).fetchall()
            if not heads:
                return copied
            last = heads[-1][0]
            for (pid, uid, run, source, layout, kind, length, seed, env_steps, block, tour, worker, ts, replay,
                 size) in heads:
                run_row = None
                if run in run_ids:
                    row = self._conn.execute("SELECT id FROM runs WHERE run_id = ?", (run_ids[run],)).fetchone()
                    run_row = None if row is None else int(row[0])
                tid = None if tour is None else self._id("tournament", parent._name(tour))
                cur = self._conn.execute(
                    f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (uid, run_row, self._id("source", parent._name(source)),
                     self._id("layout", parent._name(layout)), kind, length, seed, env_steps, block, tid, worker,
                     ts, replay, size))
                if cur.rowcount == 0:
                    continue
                match = int(cur.lastrowid)
                seats = parent._conn.execute(f"SELECT {_SEAT_COLUMNS} FROM seats WHERE match = ?", (pid,)).fetchall()
                self._conn.executemany(
                    f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(match, seat, team, self._id("role", parent._name(role)),
                      self._id("agent", parent._name(agent)), self._id("network", parent._name(network)),
                      self._id("entity", parent._name(entity)), vs, ve, rv, reward, elim,
                      self._id("category", parent._name(category)))
                     for seat, team, role, agent, network, entity, vs, ve, rv, reward, elim, category in seats])
                self._conn.executemany(
                    "INSERT INTO teams (match, team, rank, score) VALUES (?, ?, ?, ?)",
                    [(match, *row) for row in parent._conn.execute(
                        "SELECT team, rank, score FROM teams WHERE match = ?", (pid,)).fetchall()])
                copied += 1

    # ---- reading -----------------------------------------------------------------------------------------------

    def comparison_table(self, sources: Collection[str] = RUN_SOURCES,
                         tournament: str | None = None) -> ComparisonTable:
        """Detailed matches and counts of ``sources`` as one table; ``tournament``: only that tournament's
        detailed matches (counts never hold tournament matches); None: only matches outside tournaments."""
        return self._table(sources, tournament)

    def sizes(self) -> LogSizes:
        return self._sizes(self.path, self._disk_free_fn(self.path.parent), self._degraded)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error as e:
            logger.warning(f"Match log {self.path}: closing the connection failed ({e})")


class MatchLogReader(_Store):
    """Read-only access (report, dashboard, tournament, SP5's hub). Every query is its own transaction; a missing
    file reads as an empty journal. The connection may be used from another thread than the one that opened it
    (one thread at a time)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            for statement in _SCHEMA:
                conn.execute(statement)
            super().__init__(conn)
            return
        super().__init__(_connect(self.path, readonly=True))
        try:
            version = self._meta("schema_version")
        except sqlite3.DatabaseError as e:
            self._conn.close()
            raise MatchLogError(f"{self.path} is not a match log ({e})") from e
        if version is None or int(version) != SCHEMA_VERSION:
            self._conn.close()
            raise MatchLogError(f"{self.path} is a match log of schema version {version}, not {SCHEMA_VERSION}")

    def __enter__(self) -> MatchLogReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def comparison_table(self, sources: Collection[str] = RUN_SOURCES,
                         tournament: str | None = None) -> ComparisonTable:
        """As ``MatchLog.comparison_table``."""
        return self._table(sources, tournament)

    def match_results(self, *, source: str, tournament: str | None = None) -> list[MatchResult]:
        """Every detailed match of ``source`` (and ``tournament``, when given) as a ``MatchResult``, oldest first
        (row ids): tournament resume (T5.4) and pulse tests (T6.1). Tournament and pulse matches are never
        compacted, so they are complete."""
        rows = self.matches(source=source, tournament=tournament, limit=None)
        return [row.to_result()[0] for row in reversed(rows)]

    def blocks(self, tournament: str) -> set[str]:
        """Block ids of ``tournament`` in the journal (blocks are written whole, one transaction each)."""
        tid = self._lookup("tournament", tournament)
        if tid is None:
            return set()
        return {str(r[0]) for r in self._conn.execute(
            "SELECT DISTINCT block FROM matches WHERE tournament = ? AND block IS NOT NULL", (tid,))}

    def runs(self) -> list[RunInfo]:
        """The ``runs`` rows in insertion order (this run first, then the imported parents)."""
        return [RunInfo(run_id=r[0], name=r[1], parent_run_id=r[2], game=r[3], config_hash=r[4],
                        checkpoint_interval=int(r[5]))
                for r in self._conn.execute("SELECT run_id, name, parent_run_id, game, config_hash, "
                                            "checkpoint_interval FROM runs ORDER BY id")]

    def sizes(self) -> LogSizes:
        free = default_disk_free(self.path.parent) if self.path.parent.exists() else 0
        return self._sizes(self.path, free, False)

    def matches(self, *, agent: str | None = None, version: int | None = None, layout: str | None = None,
                source: str | None = None, tournament: str | None = None, outcome: str | None = None,
                env_steps: tuple[int, int] | None = None, limit: int | None = 1000) -> list[MatchRow]:
        """Detailed matches, newest first. ``agent`` (with ``version``: a rating version) selects matches with such
        a seat; ``outcome`` (``win`` / ``draw`` / ``loss``; needs ``agent``) is that agent's team result against the
        best other team; ``env_steps`` is an inclusive range; ``limit=None`` returns every matching row."""
        if outcome is not None and agent is None:
            raise ValueError("an outcome filter needs an agent")
        if version is not None and agent is None:
            raise ValueError("a version filter needs an agent")
        clauses: list[str] = []
        args: list = []
        for kind, name in (("layout", layout), ("source", source), ("tournament", tournament)):
            if name is not None:
                found = self._lookup(kind, name)
                if found is None:
                    return []
                clauses.append(f"m.{kind} = ?")
                args.append(found)
        if env_steps is not None:
            clauses.append("m.env_steps BETWEEN ? AND ?")
            args.extend(int(x) for x in env_steps)
        if agent is not None:
            agent_id = self._lookup("agent", agent)
            if agent_id is None:
                return []
            sub = "SELECT match FROM seats WHERE agent = ?"
            args.append(agent_id)
            if version is not None:
                sub += " AND rating_version = ?"
                args.append(int(version))
            clauses.append(f"m.id IN ({sub})")
        where = " AND ".join(clauses) or "1"
        out: list[MatchRow] = []
        last: int | None = None
        page_size = READ_BATCH if limit is None else max(limit, 100)
        while limit is None or len(out) < limit:
            page = self._conn.execute(
                f"SELECT m.id, m.uid, r.run_id, m.source, m.layout, m.outcome_kind, m.length, m.seed, m.env_steps, "
                f"m.block, m.tournament, m.worker, m.ts, m.replay FROM matches m LEFT JOIN runs r ON r.id = m.run "
                f"WHERE {where}{'' if last is None else ' AND m.id < ?'} ORDER BY m.id DESC LIMIT ?",
                (*args, *(() if last is None else (last,)), page_size)).fetchall()
            if not page:
                break
            last = page[-1][0]
            for row in page:
                match = self._row(row)
                if outcome is not None and _outcome_for(match, agent) != outcome:
                    continue
                out.append(match)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def _row(self, row) -> MatchRow:
        mid = row[0]
        seats = tuple(SeatRow(
            seat=s[0], team=s[1], role=self._name(s[2]), agent_id=self._name(s[3]), network_id=self._name(s[4]),
            entity=self._name(s[5]), version_start=s[6], version_end=s[7], rating_version=s[8], reward=float(s[9]),
            eliminated_step=s[10], category=self._name(s[11]))
            for s in self._conn.execute(f"SELECT {_SEAT_COLUMNS} FROM seats WHERE match = ? ORDER BY seat", (mid,)))
        teams = tuple((int(t[0]), float(t[1]), float(t[2])) for t in self._conn.execute(
            "SELECT team, rank, score FROM teams WHERE match = ? ORDER BY team", (mid,)))
        return MatchRow(id=mid, uid=row[1], run_id=row[2], source=self._name(row[3]), layout=self._name(row[4]),
                        outcome_kind=row[5], length=int(row[6]), seed=row[7], env_steps=row[8], block=row[9],
                        tournament=self._name(row[10]), worker=row[11], ts=float(row[12]), seats=seats, teams=teams,
                        replay=row[13])

    def close(self) -> None:
        self._conn.close()


def _outcome_for(match: MatchRow, agent: str) -> str | None:
    ranks = {team: rank for team, rank, _ in match.teams}
    mine = sorted({s.team for s in match.seats if s.agent_id == agent})
    if not mine or len(ranks) < 2:
        return None
    own = ranks[mine[0]]
    best_other = min(r for t, r in ranks.items() if t != mine[0])
    return "win" if own < best_other else ("draw" if own == best_other else "loss")


# ---------------------------------------------------------------------------------------------------------------
# The writer thread of the training main process (T1.3)
# ---------------------------------------------------------------------------------------------------------------

WRITER_QUEUE_SIZE = 100_000
WRITE_BATCH = 2_000
# The parts of the file compaction does not touch (counts, tournament and pulse matches) may grow past the window:
# one WARNING when they exceed max(window, this many bytes).
UNTOUCHED_WARNING_FLOOR = 64 * 1024 * 1024
# ``stop(timeout)`` ends the flush this long before ``timeout``, so that closing the journal fits in it too.
_STOP_MARGIN_SEC = 0.5
# The ``LogSizes`` fields of ``WriterStats.as_dict()`` (null until the first sizes are read).
_SIZE_KEYS = ("file_bytes", "wal_bytes", "detail_bytes", "detail_matches", "other_matches", "counts_rows",
              "free_disk_bytes")


@dataclass(frozen=True)
class WriterStats:
    """What the writer did so far (an immutable snapshot; the main loop reads it for the ``system`` record).
    Every submitted result ends up in exactly one of ``written``, ``duplicates``, ``dropped`` and ``pending``."""

    written: int = 0            # journaled results (detailed or counted)
    dropped: int = 0            # lost: writer queue full, write errors, not written at stop, journal unusable
    duplicates: int = 0         # uids already in the journal
    write_errors: int = 0       # failed appends, compaction passes and table feeds; a failed open or a dead thread
    compactions: int = 0
    folded: int = 0             # training matches folded into counts by compaction
    evicted_folded: int = 0     # of which because they had an evicted snapshot's entity
    degraded: bool = False      # the disk guard is on (counts only)
    sizes: LogSizes | None = None   # after the last compaction pass (or the start / the stop)
    pending: int = 0            # results waiting in the queue or being written

    def as_dict(self) -> dict:
        """JSON-ready form for the ``system`` record (``match_log``); the ``LogSizes`` keys are null until the
        sizes were read, so the record keeps one shape."""
        out = {"written": self.written, "dropped": self.dropped, "duplicates": self.duplicates,
               "write_errors": self.write_errors, "compactions": self.compactions, "folded": self.folded,
               "evicted_folded": self.evicted_folded, "degraded": self.degraded, "pending": self.pending}
        out.update({key: None if self.sizes is None else getattr(self.sizes, key) for key in _SIZE_KEYS})
        return out


class MatchLogWriter:
    """Journal writer thread (module docstring). ``log_factory()`` opens the thread's own ``MatchLog`` (called in
    the thread: a ``MatchLog`` connection belongs to the thread that opened it); ``table`` (optional) gets every
    journaled result; ``clock`` measures the compaction cadence and the stop deadline (monotonic).

    Thread rules: the thread touches nothing of the caller but its queue, its stats, its in-flight count and the
    evicted entities (all under its lock) and ``table`` (internally locked). ``submit``, ``note_evicted`` and
    ``stop`` are called from one thread (the main loop).

    Accounting: a batch the thread took from the queue is "in flight" (``pending``) until its append returns.
    When ``stop`` gives up waiting for the thread, the in-flight and queued results are counted as dropped and the
    writer is abandoned: whatever the thread does afterwards no longer changes the stats. When the journal cannot
    be opened or the thread dies, the writer is dead: queued and later results are counted as dropped."""

    def __init__(self, log_factory: Callable[[], MatchLog], *, queue_size: int = WRITER_QUEUE_SIZE,
                 compact_interval_sec: float, window_bytes: int, table: ComparisonTable | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory = log_factory
        self._queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self._interval = float(compact_interval_sec)
        self._window = int(window_bytes)
        self._untouched_limit = max(int(window_bytes), UNTOUCHED_WARNING_FLOOR)
        self._table = table
        self._clock = clock
        self._lock = threading.Lock()
        self._stats = WriterStats()
        self._in_flight = 0
        self._abandoned = False      # stop() gave up waiting: the thread's late updates are ignored
        self._dead = False           # the thread cannot write (the journal is unusable or the thread died)
        self._evicted: set[EntityKey] = set()
        self._stop = threading.Event()
        self._deadline = math.inf
        self._thread: threading.Thread | None = None
        self._warned_drop = self._warned_error = self._warned_untouched = False

    @property
    def stats(self) -> WriterStats:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> WriterStats:
        return replace(self._stats, pending=self._queue.qsize() + self._in_flight)

    def _bump_locked(self, **delta: int) -> None:
        self._stats = replace(self._stats, **{k: getattr(self._stats, k) + v for k, v in delta.items()})

    def _drain_locked(self) -> int:
        """Remove everything queued; returns how many results that was."""
        n = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return n
            n += 1

    def _thread_update(self, *, values: Mapping | None = None, **delta: int) -> bool:
        """A stats update by the thread; ignored (False) once the writer is abandoned."""
        with self._lock:
            if self._abandoned:
                return False
            self._bump_locked(**delta)
            if values:
                self._stats = replace(self._stats, **values)
            return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="match-log-writer", daemon=True)
        self._thread.start()

    def submit(self, results: Sequence[MatchResult]) -> int:
        """Queue ``results`` without blocking; returns how many were dropped (queue full, the writer stopped or
        dead; counted, the first time a WARNING)."""
        with self._lock:
            if self._stop.is_set() or self._dead:
                reason = "is stopped" if self._stop.is_set() else "is not running (the journal is unusable)"
                dropped = len(results)
            else:
                reason = f"queue is full ({self._queue.maxsize} results)"
                dropped = 0
                for result in results:
                    try:
                        self._queue.put_nowait(result)
                    except queue.Full:
                        dropped += 1
            if dropped:
                self._bump_locked(dropped=dropped)
        if dropped and not self._warned_drop:
            self._warned_drop = True
            logger.warning(f"Match log writer {reason}; dropped {dropped} match results (counted in "
                           f"dropped_match_results; further drops are only counted)")
        return dropped

    def note_evicted(self, entities: Collection[EntityKey]) -> None:
        """Entities of evicted snapshots: their training matches are folded at the next compaction pass."""
        with self._lock:
            self._evicted.update(entities)

    def stop(self, timeout: float) -> WriterStats:
        """Stop compacting, write what is queued, close the journal; within ``timeout`` seconds. What could not be
        written in time (queued, or in flight when the thread does not stop in time) is counted as dropped (one
        WARNING); the returned stats are final. Idempotent; harmless without ``start``."""
        timeout = max(0.0, float(timeout))
        self._deadline = self._clock() + max(0.0, timeout - _STOP_MARGIN_SEC)
        self._stop.set()
        alive = False
        if self._thread is not None:
            self._thread.join(timeout)
            alive = self._thread.is_alive()
        with self._lock:
            in_flight = 0
            if alive:
                self._abandoned = True
                in_flight, self._in_flight = self._in_flight, 0
            left = self._drain_locked()
            if in_flight + left:
                self._bump_locked(dropped=in_flight + left)
            final = self._snapshot_locked()
        if alive:
            logger.warning(f"Match log writer did not stop within {timeout:.1f} s; the {in_flight} results it was "
                           f"writing are counted as dropped (the journal stays open until the process exits)")
        if left:
            logger.warning(f"Match log writer stopped with {left} results not written (counted in "
                           f"dropped_match_results)")
        return final

    # ---- the thread --------------------------------------------------------------------------------------------

    def _take(self, timeout: float) -> list[MatchResult]:
        """Up to ``WRITE_BATCH`` queued results, now in flight; waits up to ``timeout`` for the first."""
        batch: list[MatchResult] = []
        if timeout > 0:
            try:
                batch.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                return []
        with self._lock:
            if self._abandoned:
                if batch:
                    self._bump_locked(dropped=len(batch))
                return []
            while len(batch) < WRITE_BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._in_flight = len(batch)
        return batch

    def _error(self, what: str, error: Exception) -> None:
        """One ERROR per failure; the traceback only with the first."""
        logger.error(f"Match log: {what} ({type(error).__name__}: {error}); training goes on",
                     exc_info=not self._warned_error)
        self._warned_error = True

    def _write(self, log: MatchLog, batch: list[MatchResult]) -> None:
        try:
            stats = log.append(batch)
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            with self._lock:
                if self._abandoned:
                    return
                self._in_flight = 0
                self._bump_locked(dropped=len(batch), write_errors=1)
            self._error(f"writing {len(batch)} results failed; they are counted in dropped_match_results", e)
            return
        with self._lock:
            if self._abandoned:                     # counted as dropped by stop()
                return
            self._in_flight = 0
            self._bump_locked(written=stats.detailed + stats.counted, duplicates=stats.duplicates)
            self._stats = replace(self._stats, degraded=stats.degraded)
        if self._table is None:
            return
        try:
            for result, accepted in zip(batch, stats.accepted, strict=True):
                if accepted:
                    self._table.add_result(result, log.rating_versions(result))
        except Exception as e:  # noqa: BLE001 - journaled already; only the live ratings miss them
            if self._thread_update(write_errors=1):
                self._error(f"adding {len(batch)} journaled results to the live comparison table failed", e)

    def _compact(self, log: MatchLog) -> None:
        with self._lock:
            evicted, self._evicted = self._evicted, set()
        try:
            result = log.compact(window_bytes=self._window, evicted=evicted, should_stop=self._stop.is_set)
            sizes = log.sizes()
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            with self._lock:
                self._evicted |= evicted
            if self._thread_update(write_errors=1):
                self._error("compaction failed", e)
            return
        if result.interrupted:
            with self._lock:
                self._evicted |= evicted          # the evicted entities not folded yet stay pending
        untouched = sizes.file_bytes - sizes.detail_bytes
        if untouched > self._untouched_limit and not self._warned_untouched:
            self._warned_untouched = True
            logger.warning(f"Match log: the parts compaction does not touch (counts, tournament and pulse matches) "
                           f"take {untouched / 2**20:.0f} MB, more than the detail window; the journal grows past "
                           f"match_log.detail_window_mb")
        self._thread_update(compactions=1, folded=result.folded, evicted_folded=result.evicted_folded,
                            values={"sizes": sizes})

    def _refresh_sizes(self, log: MatchLog) -> None:
        try:
            sizes = log.sizes()
        except Exception as e:  # noqa: BLE001 - only the reported sizes are stale
            self._error("reading the journal sizes failed", e)
            return
        self._thread_update(values={"sizes": sizes})

    def _die(self) -> None:
        """The thread cannot write any more: the in-flight and queued results are dropped, and so are later ones
        (``submit``)."""
        with self._lock:
            self._dead = True
            if self._abandoned:
                return
            lost = self._in_flight + self._drain_locked()
            self._in_flight = 0
            self._bump_locked(dropped=lost, write_errors=1)

    def _loop(self, log: MatchLog) -> None:
        self._refresh_sizes(log)
        last_compact = self._clock()
        while not self._stop.is_set():
            batch = self._take(0.2)
            if batch:
                self._write(log, batch)
            if not self._stop.is_set() and self._clock() - last_compact >= self._interval:
                self._compact(log)
                last_compact = self._clock()
        while self._clock() < self._deadline:          # stop(): flush the queue, no more compaction
            batch = self._take(0.0)
            if not batch:
                break
            self._write(log, batch)
        self._refresh_sizes(log)

    def _run(self) -> None:
        try:
            log = self._factory()
        except Exception as e:  # noqa: BLE001 - reported; results are only counted from now on
            logger.error(f"Match log: opening the journal failed ({type(e).__name__}: {e}); match results are "
                         f"counted as dropped, training goes on", exc_info=True)
            self._die()
            return
        try:
            self._loop(log)
        except Exception as e:  # noqa: BLE001 - reported; results are only counted from now on
            logger.error(f"Match log: the match-log writer thread died ({type(e).__name__}: {e}); match results "
                         f"are counted as dropped from now on, training goes on", exc_info=True)
            self._die()
        finally:
            log.close()

"""``colosseum tournament`` (SP4 spec block 3): the tournament dir, the settings check and resume, snapshot
links, the journal, the reports, ``--pin``.

Where things go:
- with ``--run runs/x`` (several: the first one): matches into ``runs/x/matches.db`` (the run's journal in WAL
  mode: the run may be training meanwhile), the tournament dir ``runs/x/tournaments/<name>/``;
- without ``--run``: ``--output DIR`` laid out like a run dir: ``DIR/matches.db``, ``DIR/tournaments/<name>/``
  (ruling C5).

The tournament dir holds ``tournament.yaml`` (format, name, id, the request, the resolved players, layouts,
layout weights and the seed domains), ``snapshots/`` (hard links, else copies, of every run snapshot the
tournament plays, made before the first match: a snapshot the training evicts meanwhile still plays; a new
tournament empties it first, so a failed start's links never play under a retry's names),
``logs/`` (``tournament.log`` and the workers' logs) and the reports (``report.json``, ``report.txt``,
``report.html``).

Paths: every path of the request and of the stored players is absolute (resolved at the start), so a tournament
continues from any working directory.

Resume: a start with an existing ``--name`` compares its request (mode, sizes, seed, layouts, the players as
given on the command line, the ``--set`` overrides and the env section of the config; not ``--num-workers`` /
``--num-envs`` / ``--pin``) with ``tournament.yaml``: a difference is a ConfigError listing the fields. The stored
players are reused (not re-resolved: ``top:N`` of a training run moves), and every block already in the journal
(written whole, one transaction per block) is taken from it instead of being played again. Match uids are
deterministic (``schedule``), so the journal never holds a match twice.

``--pin`` pins only a confirmed ``select`` winner that is a run snapshot; any other winner gets a WARNING telling
the user to pin by hand with ``colosseum pin`` (ruling C6). ``confirmed: false`` is a result, not an error.
"""

from __future__ import annotations

import dataclasses
import filecmp
import functools
import json
import logging
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from colosseum.arena.tournament import modes
from colosseum.arena.tournament.engine import play_blocks
from colosseum.arena.tournament.players import (
    PlayerRequest,
    PlayerSource,
    RunSelection,
    TournamentPlayer,
    load_run_config,
    player_from_dict,
    player_to_dict,
    resolve_players,
    run_agent,
)
from colosseum.arena.tournament.schedule import STAGES, Block, stage_seed
from colosseum.coordinator.checkpoint_manager import CheckpointInfo, link_or_copy, pin_snapshot
from colosseum.core.config import ColosseumConfig, check_path_component, config_hash, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RunDir
from colosseum.core.types import MatchResult
from colosseum.utils.fs import write_text_atomic

logger = logging.getLogger(__name__)

TOURNAMENT_FILE = "tournament.yaml"
TOURNAMENT_FORMAT = 1
REPORT_JSON, REPORT_TXT, REPORT_HTML = "report.json", "report.txt", "report.html"
_SNAPSHOT_FILES = ("model.pt", "meta.json")


@dataclass(frozen=True)
class TournamentOptions:
    """The command line, parsed (``cli.tournament_cmd``)."""

    mode: str = "gauntlet"
    candidates: tuple[tuple[str, str | None], ...] = ()   # parsed -a specs
    runs: tuple[RunSelection, ...] = ()
    field: tuple[str, ...] | None = None
    layouts: tuple[str, ...] = ()
    blocks: int = modes.DEFAULT_BLOCKS
    halving_blocks: int = modes.DEFAULT_HALVING_BLOCKS
    confirm_blocks: int = modes.DEFAULT_CONFIRM_BLOCKS
    finalists: int = modes.DEFAULT_FINALISTS
    seed: int = 0
    deterministic: bool = False
    num_workers: int = 1
    num_envs: int = 8
    name: str | None = None
    output: str | None = None
    pin: bool = False
    config_path: str | None = None
    overrides: dict | None = None                          # parsed --set


@dataclass
class TournamentOutcome:
    report: modes.TournamentReport
    directory: Path
    played_blocks: int
    reused_blocks: int
    pinned: list[str]
    written: list[Path]
    pin_errors: list[str] = dataclasses.field(default_factory=list)   # --pin failures (the CLI exits 1)


def diff_fields(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two JSON-like trees (lists compared whole)."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            out.extend(diff_fields(a.get(key), b.get(key), f"{prefix}{key}."))
        return out
    return [] if a == b else [prefix.rstrip(".") or "<root>"]


class TournamentDir:
    """``runs/x/tournaments/<name>/`` (module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def yaml_path(self) -> Path:
        return self.path / TOURNAMENT_FILE

    @property
    def snapshots(self) -> Path:
        return self.path / "snapshots"

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    @property
    def report_paths(self) -> tuple[Path, Path, Path]:
        """``report.json``, ``report.txt``, ``report.html``."""
        return self.path / REPORT_JSON, self.path / REPORT_TXT, self.path / REPORT_HTML

    def load(self) -> dict | None:
        if not self.yaml_path.is_file():
            return None
        data = yaml.safe_load(self.yaml_path.read_text()) or {}
        if data.get("format") != TOURNAMENT_FORMAT:
            raise ConfigError(f"{self.yaml_path}: format {data.get('format')!r}, expected {TOURNAMENT_FORMAT}")
        return data

    def write(self, data: Mapping[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.yaml_path, yaml.safe_dump(dict(data), sort_keys=False))

    def check(self, stored: Mapping[str, Any], request: Mapping[str, Any]) -> None:
        """ConfigError naming every request field that differs from ``tournament.yaml`` (spec block 3)."""
        fields = diff_fields(stored.get("request", {}), dict(request))
        if fields:
            raise ConfigError(f"tournament {stored.get('name')!r} exists ({self.yaml_path}) with other settings: "
                              f"{', '.join(fields)} differ (the check covers every size flag, also those the mode "
                              f"does not use); rerun with the same settings to continue it, or choose another "
                              f"--name")

    def link(self, run_index: int, selection: RunSelection, info: CheckpointInfo) -> Path:
        """The snapshot's hard-linked (else copied) ``model.pt`` + ``meta.json`` in ``snapshots/``. An existing link
        dir is reused only when it holds the snapshot's own files (the same file, or equal bytes for a copy), never
        by its path alone; anything else there is replaced."""
        dst = self.snapshots / f"{run_index}-{Path(selection.path).name}" / info.agent_id / info.checkpoint_id
        if all((dst / name).is_file() for name in _SNAPSHOT_FILES) and \
                all(_same_file(dst / name, info.path / name) for name in _SNAPSHOT_FILES):
            return dst
        if dst.exists():          # another snapshot's files, or an incomplete link dir
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{info.checkpoint_id}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            for name in _SNAPSHOT_FILES:
                link_or_copy(info.path / name, tmp / name)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return dst

    def write_reports(self, report: modes.TournamentReport) -> list[Path]:
        """``report.json``, ``report.txt`` and ``report.html`` (the run report's page with the tournaments
        section only): the single writer of the reports (contract amendment A27)."""
        from colosseum.arena.report.render import write_tournament_html

        json_path, txt_path, _html_path = self.report_paths
        written = [write_text_atomic(json_path, json.dumps(report.to_dict(), indent=2, sort_keys=False) + "\n"),
                   write_text_atomic(txt_path, report.text() + "\n")]
        written.append(write_tournament_html(self.path))   # report.html: the same page, tournaments section only
        return written


class BlockRunner:
    """The ``play`` of ``modes.run``: blocks already in the journal come from it; the others are played by
    ``engine.play_blocks`` and every finished block is appended to the journal in one transaction (``_store``)."""

    def __init__(self, *, config: ColosseumConfig, spec, players: Mapping[str, TournamentPlayer], log,
                 done: Mapping[str, list[MatchResult]], num_workers: int, num_envs: int, deterministic: bool,
                 log_dir: Path) -> None:
        self._config, self._spec, self._players, self._log = config, spec, players, log
        self._done = {k: list(v) for k, v in done.items()}
        self._num_workers, self._num_envs, self._deterministic = num_workers, num_envs, deterministic
        self._log_dir = log_dir
        self.played = 0
        self.reused = 0

    def _store(self, block_id: str, results: list[MatchResult]) -> None:
        # one transaction: a block is in the journal whole or not at all; the tournament id is the block-id prefix
        self._log.append(results, tournament=block_id.split("/", 1)[0])
        self._done[block_id] = list(results)
        self.played += 1

    def __call__(self, blocks: Sequence[Block]) -> dict[str, list[MatchResult]]:
        out: dict[str, list[MatchResult]] = {}
        todo: list[Block] = []
        for block in blocks:
            got = self._done.get(block.id)
            if got is None:
                todo.append(block)
                continue
            if sorted(r.uid for r in got) != sorted(lu.uid for lu in block.lineups):
                raise ConfigError(f"journal block {block.id} has matches {sorted(r.uid for r in got)}, the schedule "
                                  f"{sorted(lu.uid for lu in block.lineups)}: the tournament's settings changed; "
                                  f"choose another --name")
            out[block.id] = got
            self.reused += 1
        if todo:
            play_blocks(self._config, self._spec, self._players, todo, num_workers=self._num_workers,
                        num_envs=self._num_envs, deterministic=self._deterministic, on_block=self._store,
                        log_dir=self._log_dir, source="tournament")
            out.update({b.id: self._done[b.id] for b in todo})
        return out


def _same_file(a: Path, b: Path) -> bool:
    return os.path.samefile(a, b) or filecmp.cmp(a, b, shallow=False)


def _absolute(path: str | Path) -> str:
    return str(Path(path).resolve())


def _absolute_options(options: TournamentOptions) -> TournamentOptions:
    """Every path of the command line resolved (tournament.yaml and resume work from any cwd)."""
    return dataclasses.replace(
        options,
        candidates=tuple((n, None if p is None else _absolute(p)) for n, p in options.candidates),
        runs=tuple(dataclasses.replace(r, path=Path(_absolute(r.path))) for r in options.runs),
        output=None if options.output is None else _absolute(options.output))


def _absolute_player(player: TournamentPlayer) -> TournamentPlayer:
    """``player`` with absolute weights / origin (neural players) and run dir (a config's frozen agent may name a
    relative path)."""
    source = player.source
    if source.kind == "model":
        source = PlayerSource("model", origin=_absolute(source.origin), weights=_absolute(source.weights),
                              networks=source.networks)
    return dataclasses.replace(player, source=source, run=None if player.run is None else _absolute(player.run))


def _request(options: TournamentOptions, config: ColosseumConfig) -> dict[str, Any]:
    return {
        "mode": options.mode,
        "candidates": [f"{n}={p}" if p is not None else n for n, p in options.candidates],
        "runs": [{"path": str(r.path), "alias": r.alias, "agent": r.agent, "snapshots": r.snapshots}
                 for r in options.runs],
        "field": None if options.field is None else list(options.field),
        "layouts": list(options.layouts),
        "blocks": options.blocks, "halving_blocks": options.halving_blocks,
        "confirm_blocks": options.confirm_blocks, "finalists": options.finalists,
        "seed": options.seed, "deterministic": options.deterministic,
        "overrides": dict(sorted((options.overrides or {}).items())),
        "env": config.env.model_dump(mode="json", by_alias=True),
    }


def _config(options: TournamentOptions, run_configs: Sequence[ColosseumConfig]) -> ColosseumConfig:
    if options.config_path is not None:
        config = load_config(options.config_path, options.overrides or None)
    elif options.runs:
        config = load_config(Path(options.runs[0].path) / "config.resolved.yaml", options.overrides or None)
    else:
        raise ConfigError("give -c CONFIG or --run DIR (the run's config.resolved.yaml is used then)")
    own = config.env.model_dump(mode="json", by_alias=True)
    for sel, run_config in zip(options.runs, run_configs):
        theirs = run_config.env.model_dump(mode="json", by_alias=True)
        if theirs != own:
            raise ConfigError(f"--run {sel.path}: its env section differs from the tournament config's "
                              f"({', '.join(diff_fields(theirs, own))}); snapshots play only in the game they were "
                              f"trained on: drop -c (the run's config is used) or fix the config's env")
    return config


def _layouts(options: TournamentOptions, config: ColosseumConfig, spec,
             first_config: ColosseumConfig | None) -> tuple[list[str], dict[str, float]]:
    """The tournament's layouts and weights: the first ``--run`` agent's effective ``matchmaking.layouts`` (else
    the config's); equal weights with an INFO when a chosen layout has none (ruling D13)."""
    from colosseum.league.lineups import enabled_layouts

    if options.runs:
        first = Path(options.runs[0].path)
        matchmaking = first_config.get_agent_config(run_agent(first, first_config, options.runs[0].agent)).matchmaking
    else:
        matchmaking = config.matchmaking
    weights = enabled_layouts(spec, matchmaking)
    chosen = list(dict.fromkeys(options.layouts)) if options.layouts else [n for n in weights if n in spec.layouts]
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    if all(name in weights for name in chosen):
        return chosen, {name: float(weights[name]) for name in chosen}
    logger.info("layouts %s: some have no weight in matchmaking.layouts; weighting them equally", chosen)
    return chosen, dict.fromkeys(chosen, 1.0)


def _journal_run(options: TournamentOptions, config: ColosseumConfig, db: Path, tournament_id: str,
                 first_config: ColosseumConfig | None):
    """The ``RunInfo`` the tournament writes its matches under: the first ``--run``'s (its run id, created for an
    SP3 run dir: P16), else one of the tournament itself (``--output``)."""
    from colosseum.arena.matchlog import MatchLog, RunInfo

    if not options.runs:
        return RunInfo(run_id=tournament_id, name=f"tournament-{options.name}", parent_run_id=None,
                       game=config.env.env_class, config_hash=config_hash(config),
                       checkpoint_interval=config.checkpoint.interval)
    first = Path(options.runs[0].path)
    run_id = RunDir.open(first).ensure_run_id()
    if db.is_file():
        reader = MatchLog.open_readonly(db)
        try:
            known = {r.run_id: r for r in reader.runs()}
        finally:
            reader.close()
        if run_id in known:
            return known[run_id]
    return RunInfo(run_id=run_id, name=first.name, parent_run_id=None, game=first_config.env.env_class,
                   config_hash=config_hash(first_config), checkpoint_interval=first_config.checkpoint.interval)


def _file_logging(path: Path) -> tuple[logging.Handler, int]:
    """``tournament.log`` for the ``colosseum`` loggers of this command; returns the handler and the logger's old
    level (the caller removes the one and restores the other)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] tournament %(name)s: %(message)s"))
    root = logging.getLogger("colosseum")
    old_level = root.level
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler, old_level


def _journal_blocks(db: Path, name: str) -> int:
    """How many blocks of tournament ``name`` the journal at ``db`` holds (0 without a journal)."""
    from colosseum.arena.matchlog import MatchLog

    if not db.is_file():
        return 0
    reader = MatchLog.open_readonly(db)
    try:
        return len(reader.blocks(name))
    finally:
        reader.close()


def _new_tournament(options: TournamentOptions, config: ColosseumConfig, spec, tdir: TournamentDir,
                    request: dict[str, Any], *, db: Path, first_config: ColosseumConfig | None) -> dict[str, Any]:
    """Validate, resolve the players (linking their snapshots into ``tdir``) and write ``tournament.yaml``.

    A name whose blocks are already in the journal (its ``tournament.yaml`` was removed) is refused: those blocks
    were played with settings nobody can check any more, and a new tournament would silently reuse them."""
    from colosseum.core.registry import validate_config

    if tdir.path.exists() and any(p.name not in ("logs", "snapshots") for p in tdir.path.iterdir()):
        raise ConfigError(f"{tdir.path} exists without {TOURNAMENT_FILE}; choose another --name")
    held = _journal_blocks(db, options.name)
    if held:
        raise ConfigError(f"the match log {db} already holds {held} blocks of a tournament named {options.name!r} "
                          f"whose {TOURNAMENT_FILE} is gone ({tdir.yaml_path}); its settings cannot be checked, so "
                          f"its blocks cannot be continued: choose another --name")
    if tdir.snapshots.exists():       # links of a start that failed before tournament.yaml: nothing uses them
        shutil.rmtree(tdir.snapshots)
    bare = {name for name, path in options.candidates if path is None}
    fixed_names = [n for n in config.fixed_agent_ids()
                   if options.field is None or n in options.field or n in bare]
    report_v = validate_config(config, scope="play", players=fixed_names)
    layouts, weights = _layouts(options, config, spec, first_config)
    request_players = PlayerRequest(user=tuple(options.candidates), runs=tuple(options.runs), field=options.field)
    # an empty field: round-robin plays the pairs; one-team layouts play the homogeneous team (B-I1)
    cand, fld = resolve_players(config, spec, request_players, fixed=report_v.fixed, link=tdir.link,
                                rng=np.random.default_rng(options.seed),
                                allow_empty_field=options.mode == "round-robin"
                                or all(spec.num_teams(L) == 1 for L in layouts))
    players = {p.name: _absolute_player(p) for p in [*cand, *fld]}
    tdir.write({
        "format": TOURNAMENT_FORMAT, "name": options.name, "id": uuid.uuid4().hex,
        "created": datetime.now().isoformat(timespec="seconds"),   # informational
        "request": request,
        "players": [player_to_dict(p) for p in players.values()],
        "candidates": [p.name for p in cand], "field": [p.name for p in fld],
        "layouts": layouts, "layout_weights": weights,
        "seeds": {"seed": options.seed,
                  "domains": {stage: [stage_seed(options.seed, stage, k) for k in range(3)] for stage in STAGES}},
    })
    logger.info("Tournament %s (%s): candidates %s, field %s, layouts %s", options.name, options.mode,
                [p.name for p in cand], [p.name for p in fld], layouts)
    stored = tdir.load()
    if stored is None:
        raise ConfigError(f"{tdir.yaml_path} vanished right after it was written; is another process removing "
                          f"{tdir.path}?")
    return stored


def run_tournament(options: TournamentOptions) -> TournamentOutcome:
    """Run (or continue) one tournament (module docstring)."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
    from colosseum.core.registry import env_spec

    options = _absolute_options(options)
    run_configs = [load_run_config(Path(sel.path)) for sel in options.runs]   # each read once per start
    first_config = run_configs[0] if run_configs else None
    config = _config(options, run_configs)
    spec = env_spec(config)
    name = options.name or f"{options.mode}-{datetime.now():%Y%m%d-%H%M%S}"
    check_path_component(name, "tournament name")
    options = dataclasses.replace(options, name=name)
    if options.runs:
        root = Path(options.runs[0].path)
    elif options.output is not None:
        root = Path(options.output)
    else:
        raise ConfigError("--output DIR is required without --run")
    tdir = TournamentDir(RunDir.open(root).tournaments_dir / name)
    db = root / MATCH_LOG_FILE
    handler, old_level = _file_logging(tdir.logs / "tournament.log")
    try:
        request = _request(options, config)
        stored = tdir.load()
        if stored is not None:
            tdir.check(stored, request)
            missing = [d["name"] for d in stored["players"]
                       if d["source"]["kind"] == "model" and not Path(d["source"]["weights"]).is_file()]
            if missing:
                raise ConfigError(f"tournament {name!r}: the weights of {missing} are gone (snapshots/ of "
                                  f"{tdir.path} or the -a paths); they cannot be played again")
            logger.info("Continuing tournament %s (%s)", name, tdir.path)
        else:
            stored = _new_tournament(options, config, spec, tdir, request, db=db, first_config=first_config)
        players = {d["name"]: player_from_dict(d) for d in stored["players"]}
        candidates, field_names = list(stored["candidates"]), list(stored["field"])
        layouts, weights = list(stored["layouts"]), {k: float(v) for k, v in stored["layout_weights"].items()}
        log = MatchLog(db, _journal_run(options, config, db, stored["id"], first_config),
                       min_free_disk_mb=config.match_log.min_free_disk_mb)
        try:
            reader = MatchLog.open_readonly(db)
            try:
                whole = reader.blocks(name)
                done: dict[str, list[MatchResult]] = {}
                for result in reader.match_results(source="tournament", tournament=name):
                    if result.block in whole:
                        done.setdefault(result.block, []).append(result)
            finally:
                reader.close()
            runner = BlockRunner(config=config, spec=spec, players=players, log=log, done=done,
                                 num_workers=options.num_workers, num_envs=options.num_envs,
                                 deterministic=options.deterministic, log_dir=tdir.logs)
            settings = modes.TournamentSettings(
                name=name, layouts=layouts, layout_weights=weights, blocks=options.blocks,
                halving_blocks=options.halving_blocks, confirm_blocks=options.confirm_blocks,
                finalists=options.finalists, seed=options.seed, deterministic=options.deterministic)
            bt_fn = functools.partial(modes.bt_section, players=players, anchor=config.ratings.anchor,
                                      samples=config.ratings.bootstrap_samples, seed=options.seed)
            report = modes.run(options.mode, spec=spec, players=players, candidates=candidates, field=field_names,
                               settings=settings, play=runner, bt_fn=bt_fn)
        finally:
            log.close()
        written = tdir.write_reports(report)
        pinned, pin_errors = _pin(report, players) if options.pin else ([], [])
        logger.info("Tournament %s: %d blocks played, %d from the journal", name, runner.played, runner.reused)
        return TournamentOutcome(report, tdir.path, runner.played, runner.reused, pinned, written, pin_errors)
    finally:
        colosseum_logger = logging.getLogger("colosseum")
        colosseum_logger.removeHandler(handler)
        colosseum_logger.setLevel(old_level)
        handler.close()


def _pin(report: modes.TournamentReport,
         players: Mapping[str, TournamentPlayer]) -> tuple[list[str], list[str]]:
    """Pin every confirmed ``select`` winner that is a run snapshot (ruling C6); the others get a WARNING.
    Returns ``(pinned names, errors)``: a winner the run evicted while the tournament played cannot be pinned
    (an error after the reports are written; its copy stays in the tournament's ``snapshots/``)."""
    pinned: list[str] = []
    errors: list[str] = []
    for group, sel in (report.to_dict().get("selection") or {}).items():
        winner = players[sel["winner"]]
        if not sel["confirmed"]:
            by_hand = (f"colosseum pin {winner.run} {winner.run_agent}@v{winner.version}"
                       if winner.kind == "snapshot" else "colosseum pin <run dir> <agent>@v<N>")
            logger.warning("--pin: group %s: the winner %s is not confirmed, so nothing was pinned; if you accept "
                           "it, pin it by hand: %s", group, winner.name, by_hand)
            continue
        if winner.kind != "snapshot":
            logger.warning("--pin: group %s: the winner %s is not a run snapshot, so nothing was pinned (colosseum "
                           "pin pins run snapshots only)", group, winner.name)
            continue
        stored_dir = Path(winner.run) / "checkpoints" / winner.run_agent / f"ckpt_v{winner.version}"
        try:
            if not stored_dir.is_dir():
                raise ConfigError(f"{stored_dir} is gone")
            pin_snapshot(winner.run, winner.run_agent, winner.version)
        except ConfigError as e:
            errors.append(f"--pin: the winner {winner.name} was evicted by the run {winner.run} while the "
                          f"tournament played ({e}), so nothing was pinned; the tournament's copy is "
                          f"{winner.source.origin}")
            logger.error(errors[-1])
            continue
        logger.info("--pin: pinned %s in %s", winner.name, winner.run)
        pinned.append(winner.name)
    return pinned, errors

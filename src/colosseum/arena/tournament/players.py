"""Tournament players (SP4 spec blocks 2 and 3): who can play, under which entity name, built how.

Kinds and names (the name is the entity key the journal computes from the seat, spec block 2):
- ``snapshot``: ``ckpt_v<N>`` of a run (``--run [alias=]runs/x --agent A``): name ``A@v<N>`` for the first
  ``--run``, ``<alias>/A@v<N>`` for the others (alias: the given one, else the run dir's name); seat
  ``(agent_id="[alias/]A", network_id="ckpt_v<N>")``, lineage ``[alias/]A``; architecture and roles from its
  ``meta.json`` (``networks`` absent: the run config's agent); every distinct architecture is checked once with
  that snapshot's weights (``eval``'s model check);
- ``checkpoint`` / ``weights``: ``-a name=path`` (a checkpoint dir / a ``.pt``): name ``name#<hash8>`` (the first
  8 hex digits of sha256 of the resolved path), display ``name``; seat ``(name#hash8, "fixed")``; loaded like
  ``eval -a name=path`` (``eval.load_frozen_player``, ``use_entry=False``);
- ``scripted`` / ``frozen``: an agent of the config (``-a name``, a ``--field`` name, the default field): name
  ``name``, seat ``(name, "fixed")``.

Candidates and field (``resolve_players``; Part 2 rulings D14, D15, D20):
- the field is ``--field`` (names in order), else every scripted / frozen agent of the config that is not an
  ``-a name`` candidate; a ``--field`` name is, in this order, an ``-a`` player's name, a scripted / frozen agent
  of the config, or a snapshot reference ``[alias/]agent@v<N>`` of a ``--run`` (any stored snapshot, not only
  the selected ones);
- candidates are every ``-a`` player and every selected run snapshot that is not in the field;
- an empty candidate list, an empty field (unless ``allow_empty_field``: round-robin, or only one-team layouts),
  a duplicate name, an unknown ``--field`` name, two runs with one alias, an alias that is not a path
  component: ConfigError.

Snapshot selection (``--snapshots``): ``all``; ``last:N`` (newest N); ``every:K`` (every K-th stored snapshot,
counting from the newest, which is always in); ``top:N`` (the N best by the lower bound of the BT rating from
the run's journal, ``source`` train and pulse, bootstrap; spec block 3). ``top:N`` needs ``runs/x/matches.db``
(an SP3 run dir has none: ConfigError).

``TournamentPlayer`` is picklable data (no models, no bot instances); ``player_to_dict`` / ``player_from_dict``
store it in ``tournament.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from colosseum.coordinator.checkpoint_manager import (
    MODEL_FILE,
    CheckpointInfo,
    read_checkpoint_meta,
    read_snapshots,
    read_weights_file,
)
from colosseum.core.config import ColosseumConfig, NetworkConfig, check_agent_id, check_path_component, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec
from colosseum.core.types import FIXED_NETWORK_ID
from colosseum.envs.game import GameSpec
from colosseum.players.registry import BotSpec, FixedPlayers, FrozenSpec, build_frozen_model, checkpoint_roles

PlayerKind = Literal["snapshot", "checkpoint", "weights", "scripted", "frozen"]
ENTITY_HASH_LEN = 8
logger = logging.getLogger(__name__)

_SNAPSHOT_RE = re.compile(r"(?:(?P<alias>[^/@]+)/)?(?P<agent>[A-Za-z0-9_][A-Za-z0-9_-]*)@v(?P<version>\d+)")
_SELECTION_RE = re.compile(r"(last|top|every):([1-9]\d*)")


@dataclass(frozen=True)
class PlayerSource:
    """How a worker builds the player: ``model`` (``weights`` file + ``networks`` dump) or ``bot`` (``BotSpec``).
    ``origin`` names it in messages and reports."""

    kind: Literal["model", "bot"]
    origin: str
    weights: str | None = None
    networks: dict[str, Any] | None = None
    bot: BotSpec | None = None


@dataclass(frozen=True)
class TournamentPlayer:
    name: str
    display: str
    kind: PlayerKind
    agent_id: str
    network_id: str
    roles: tuple[str, ...]
    lineage: str
    source: PlayerSource
    version: int | None = None
    run: str | None = None          # the --run dir of a snapshot (pin target)
    run_agent: str | None = None    # its agent id inside that run


@dataclass(frozen=True)
class RunSelection:
    path: Path
    alias: str | None = None
    agent: str | None = None
    snapshots: str = "all"


@dataclass(frozen=True)
class PlayerRequest:
    user: tuple[tuple[str, str | None], ...]     # parsed -a specs, in command-line order
    runs: tuple[RunSelection, ...]
    field: tuple[str, ...] | None                # --field names; None = the default field


def entity_hash(path: str | Path) -> str:
    """First ``ENTITY_HASH_LEN`` hex digits of sha256 of the resolved path (the ``#<hash>`` of ``-a name=path``)."""
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:ENTITY_HASH_LEN]


def _weights_file(path: Path) -> Path:
    return path / MODEL_FILE if path.is_dir() else path


def user_player(config: ColosseumConfig, spec: GameSpec, name: str, path: str, *,
                validated: set[str]) -> TournamentPlayer:
    """``-a name=path`` (module docstring); the weights are loaded and the architecture checked here once."""
    from colosseum.eval import load_frozen_player

    check_agent_id(name)
    p = Path(path)
    if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
        raise ConfigError(f"-a {name}={path}: expected a checkpoint dir or a .pt file")
    _model, frozen = load_frozen_player(config, name, p, spec=spec, validated=validated, use_entry=False)
    key = f"{name}#{entity_hash(p)}"
    return TournamentPlayer(
        name=key, display=name, kind="checkpoint" if p.is_dir() else "weights", agent_id=key,
        network_id=FIXED_NETWORK_ID, roles=tuple(frozen.roles), lineage=key,
        source=PlayerSource("model", origin=str(p), weights=str(_weights_file(p)), networks=dict(frozen.networks)))


def config_player(config: ColosseumConfig, spec: GameSpec, name: str, fixed: FixedPlayers) -> TournamentPlayer:
    """A scripted or frozen agent of the config (``fixed``: what ``validate_config`` loaded)."""
    entry = config.agent_entry(name)                  # ConfigError for an unknown name
    if entry.kind == "trainable":
        raise ConfigError(f"'{name}' is a trainable agent of the config, whose weights are not in the config; "
                          f"name a checkpoint with -a {name}=<dir or .pt> or a snapshot with --run ... / "
                          f"--field {name}@v<N>")
    if name in fixed.bots:
        return TournamentPlayer(name=name, display=name, kind="scripted", agent_id=name, network_id=FIXED_NETWORK_ID,
                                roles=tuple(fixed.roles[name]), lineage=name,
                                source=PlayerSource("bot", origin=f"agents.{name}", bot=fixed.bots[name]))
    if name not in fixed.frozen:
        raise ConfigError(f"agent {name!r} was not loaded (validate_config scope 'play' with players={name!r})")
    frozen = fixed.frozen[name]
    return TournamentPlayer(name=name, display=name, kind="frozen", agent_id=name, network_id=FIXED_NETWORK_ID,
                            roles=tuple(frozen.roles), lineage=name,
                            source=PlayerSource("model", origin=frozen.source,
                                                weights=str(_weights_file(Path(frozen.source))),
                                                networks=dict(frozen.networks)))


def snapshot_player(ckpt_dir: str | Path, *, agent: str, prefix: str, spec: GameSpec, run_config: ColosseumConfig,
                    run: str | Path, validated: set[str]) -> TournamentPlayer:
    """One run snapshot (module docstring). ``prefix`` is ``""`` or ``"<alias>/"``; ``ckpt_dir`` may be a linked
    copy (T5.4) whose name is ``ckpt_v<N>``."""
    path = Path(ckpt_dir)
    meta = read_checkpoint_meta(path)
    version = int(meta["policy_version"])
    roles = checkpoint_roles(spec, path, meta)
    raw = meta.get("networks")
    try:
        networks = NetworkConfig.model_validate(raw) if raw is not None else \
            run_config.get_agent_config(agent).networks
    except ValidationError as e:
        raise ConfigError(f"Checkpoint {path}: invalid meta.json networks:\n{e}") from e
    dump = networks.model_dump(mode="json", by_alias=True)
    key = json_key(dump, roles)
    if key not in validated:          # one model check per architecture, with this snapshot's weights
        from colosseum.core.validation import check_model

        frozen = FrozenSpec(agent_id=agent, roles=tuple(roles), networks=dump,
                            model_state=read_weights_file(path / MODEL_FILE), source=str(path),
                            networks_source=f"Checkpoint {path}: networks")
        model = build_frozen_model(None, frozen, spec)      # also checks the weights against the architecture
        check_model(model, agent_role_spec(spec, roles), None, f"snapshot {path}")
        validated.add(key)
    agent_id = f"{prefix}{agent}"
    return TournamentPlayer(
        name=f"{agent_id}@v{version}", display=f"{agent_id}@v{version}", kind="snapshot", agent_id=agent_id,
        network_id=f"ckpt_v{version}", roles=tuple(roles), lineage=agent_id,
        source=PlayerSource("model", origin=str(path), weights=str(path / MODEL_FILE), networks=dump),
        version=version, run=str(run), run_agent=agent)


def json_key(networks: Mapping[str, Any], roles: Sequence[str]) -> str:
    """One architecture as a string: (networks dump, roles), the key of ``validated`` (as ``eval``'s)."""
    return json.dumps([networks, list(roles)], sort_keys=True)


def parse_snapshots(text: str) -> tuple[str, int]:
    """``all`` -> ("all", 0); ``last:N`` / ``top:N`` / ``every:K`` -> (kind, n). ConfigError otherwise."""
    if text == "all":
        return "all", 0
    match = _SELECTION_RE.fullmatch(text)
    if match is None:
        raise ConfigError(f"--snapshots {text!r}: expected all, last:N, top:N or every:K (N, K >= 1)")
    return match.group(1), int(match.group(2))


def select_snapshots(infos: Sequence[CheckpointInfo], kind: str, n: int, *,
                     lcb: Mapping[str, float] | None = None) -> list[CheckpointInfo]:
    """The selected snapshots, by version (module docstring). ``lcb``: checkpoint id -> lower bound (``top``)."""
    ordered = sorted(infos, key=lambda c: c.policy_version)
    if kind == "all":
        return ordered
    if kind == "last":
        return ordered[-n:]
    if kind == "every":
        picked = {c.checkpoint_id for c in ordered[::-1][::n]}
        return [c for c in ordered if c.checkpoint_id in picked]
    rated = [c for c in ordered if lcb is not None and c.checkpoint_id in lcb]
    rated.sort(key=lambda c: (-lcb[c.checkpoint_id], -c.policy_version))
    best = {c.checkpoint_id for c in rated[:n]}
    return [c for c in ordered if c.checkpoint_id in best]


def top_snapshot_lcb(run: Path, agent: str, run_config: ColosseumConfig, spec: GameSpec, *,
                     rng: np.random.Generator) -> dict[str, float]:
    """``{checkpoint id: lower bound}`` of ``agent``'s rated snapshots from the run's journal: BT per competitive
    layout (``RUN_SOURCES``; ``bt.bootstrap`` over ``bt.bootstrap_data``: matches and pulse blocks; the fit inputs of
    ``live.rating_inputs``), combined with the agent's layout weights
    (``retention.combine_scores``, quantile ``checkpoint.top_k_quantile``); ``score``-only games use the mean
    score. ConfigError without a journal.

    The journal may be live (a training run writes to it). ``bt.bootstrap`` needs data that decomposes the fitted
    table exactly, so each layout's fit is built from the very ``bootstrap_data`` it is bootstrapped with (one
    read; ``BootstrapData.to_table``) instead of a separate ``comparison_table`` read that a write could land
    between."""
    from colosseum.arena import bt
    from colosseum.arena.live import rating_inputs
    from colosseum.arena.matchlog import MATCH_LOG_FILE, RUN_SOURCES, MatchLog, MatchLogError
    from colosseum.arena.retention import combine_scores
    from colosseum.league.lineups import enabled_layouts

    db = run / MATCH_LOG_FILE
    if not db.is_file():
        raise ConfigError(f"--snapshots top:N needs the run's match log {db}; this run has none (an SP3 run dir, or "
                          f"match_log.enabled was false): use all, last:N or every:K")
    sources = RUN_SOURCES
    options, fixed_agents, random_bot = rating_inputs(run_config)
    fits, uncertainty, scores = {}, {}, {}
    try:
        with MatchLog.open_readonly(db) as reader:
            score_table = None
            for layout in spec.layouts:
                if spec.num_teams(layout) == 1:
                    if score_table is None:
                        score_table = reader.comparison_table(sources=sources)
                    scores[layout] = bt.score_summary(score_table, layout, sources=sources)
                    continue
                # spec block 2: bootstrap by matches (training) and by blocks (pulse), from the same reader
                data = bt.bootstrap_data(reader, layout, sources=sources)
                fit = bt.fit(data.to_table(layout), layout, sources=(bt.DATA_SOURCE,), options=options,
                             fixed_agents=fixed_agents, random_bot=random_bot)
                if not fit.rating:
                    continue
                fits[layout] = fit
                uncertainty[layout] = bt.bootstrap(fit, data, samples=run_config.ratings.bootstrap_samples, rng=rng,
                                                   quantile=run_config.checkpoint.top_k_quantile)
    except MatchLogError as e:
        raise ConfigError(f"--snapshots top:N: cannot read the run's match log: {e}") from e
    weights = enabled_layouts(spec, run_config.get_agent_config(agent).matchmaking)
    combined = combine_scores(agent, fits=fits, uncertainty=uncertainty, scores=scores, layout_weights=weights,
                              quantile=run_config.checkpoint.top_k_quantile)
    return {cid: s.lcb for cid, s in combined.items() if s.lcb is not None}


def load_run_config(run: Path) -> ColosseumConfig:
    path = run / "config.resolved.yaml"
    if not path.is_file():
        raise ConfigError(f"--run {run}: no config.resolved.yaml; pass a run dir written by colosseum train")
    return load_config(str(path))


def run_agent(run: Path, run_config: ColosseumConfig, agent: str | None) -> str:
    """``--agent`` of a run, default its only trainable agent; ConfigError naming the choices."""
    trainable = run_config.get_trainable_agent_ids()
    if agent is None:
        if len(trainable) != 1:
            raise ConfigError(f"--run {run} has trainable agents {trainable}; choose one with --agent")
        return trainable[0]
    if agent not in trainable:
        raise ConfigError(f"--run {run} --agent {agent!r}: not a trainable agent of that run ({trainable})")
    return agent


LinkFn = Callable[[int, RunSelection, CheckpointInfo], Path]


def resolve_players(config: ColosseumConfig, spec: GameSpec, request: PlayerRequest, *, fixed: FixedPlayers,
                    link: LinkFn | None = None, rng: np.random.Generator | None = None,
                    allow_empty_field: bool = False) -> tuple[list[TournamentPlayer], list[TournamentPlayer]]:
    """``(candidates, field)`` by the rules of the module docstring. ``link(run_index, selection, info)`` returns
    the dir a snapshot is played from (T5.4: a hard link in the tournament dir); default the run's own dir."""
    rng = rng if rng is not None else np.random.default_rng(0)
    validated: set[str] = set()
    link = link if link is not None else (lambda _i, _sel, info: info.path)
    given = [name for name, _path in request.user]
    repeated = sorted({name for name in given if given.count(name) > 1})
    if repeated:
        raise ConfigError(f"-a {repeated[0]!r} is given more than once ({repeated}); give every -a player its own name")
    users: dict[str, TournamentPlayer] = {}
    for name, path in request.user:
        users[name] = config_player(config, spec, name, fixed) if path is None else \
            user_player(config, spec, name, path, validated=validated)
    runs = []        # (index, selection, prefix, alias, agent, run_config, stored infos)
    aliases: set[str] = set()
    for i, sel in enumerate(request.runs):
        run = Path(sel.path)
        run_config = load_run_config(run)
        agent = run_agent(run, run_config, sel.agent)
        alias = sel.alias if sel.alias is not None else run.resolve().name    # "--run ." too
        check_path_component(alias, "--run alias")
        if alias in aliases:
            raise ConfigError(f"--run alias {alias!r} is used twice; give the runs different aliases (alias=runs/x)")
        aliases.add(alias)
        infos = read_snapshots(run / "checkpoints" / agent)
        if not infos:
            raise ConfigError(f"--run {run} --agent {agent}: no stored snapshots in {run / 'checkpoints' / agent}")
        runs.append((i, sel, "" if i == 0 else f"{alias}/", alias, agent, run_config, infos))

    def snapshot(i, sel, prefix, agent, run_config, info) -> TournamentPlayer:
        return snapshot_player(link(i, sel, info), agent=agent, prefix=prefix, spec=spec, run_config=run_config,
                               run=sel.path, validated=validated)

    selected: list[TournamentPlayer] = []
    for i, sel, prefix, _alias, agent, run_config, infos in runs:
        kind, n = parse_snapshots(sel.snapshots)
        lcb = top_snapshot_lcb(Path(sel.path), agent, run_config, spec, rng=rng) if kind == "top" else None
        chosen = select_snapshots(infos, kind, n, lcb=lcb)
        if kind == "top" and len(chosen) < n:
            logger.info("--run %s --snapshots top:%d: only %d snapshots are rated", sel.path, n, len(chosen))
        selected.extend(snapshot(i, sel, prefix, agent, run_config, info) for info in chosen)

    field_players: list[TournamentPlayer] = []
    if request.field is None:
        names = [n for n in config.fixed_agent_ids() if not (n in users and users[n].kind in ("scripted", "frozen"))]
        field_players = [config_player(config, spec, n, fixed) for n in names]
    else:
        by_alias = {("" if i == 0 else alias): (i, sel, prefix, agent, rc, infos)
                    for i, sel, prefix, alias, agent, rc, infos in runs}
        for name in request.field:
            if name in users:
                field_players.append(users[name])
                continue
            if name in config.agent_ids():
                field_players.append(config_player(config, spec, name, fixed))
                continue
            match = _SNAPSHOT_RE.fullmatch(name)
            target = by_alias.get(match.group("alias") or "") if match is not None else None
            if target is None or target[3] != match.group("agent"):
                raise ConfigError(
                    f"--field {name!r} is not an -a player {sorted(users)}, an agent of the config "
                    f"{config.agent_ids()} or a snapshot [alias/]agent@v<N> of a --run "
                    f"({[f'{a}/{t[3]}@v<N>' if a else f'{t[3]}@v<N>' for a, t in by_alias.items()]})")
            i, sel, prefix, agent, rc, infos = target
            info = next((c for c in infos if c.policy_version == int(match.group("version"))), None)
            if info is None:
                raise ConfigError(f"--field {name!r}: {sel.path} has no stored snapshot ckpt_v{match.group('version')} "
                                  f"of {agent} (newest: {[c.checkpoint_id for c in infos][-5:]})")
            field_players.append(snapshot(i, sel, prefix, agent, rc, info))
    in_field = {p.name for p in field_players}     # an -a player in --field is that same player: names suffice
    candidates = [p for p in [*users.values(), *selected] if p.name not in in_field]
    names = [p.name for p in [*candidates, *field_players]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"players {duplicates} appear more than once (as candidates and in --field?)")
    if not candidates:
        raise ConfigError("no candidates: give -a players or --run snapshots that are not in --field")
    if not field_players and not allow_empty_field:
        raise ConfigError("the field is empty: name opponents with --field (agents of the config, -a players or "
                          "snapshots), or add scripted / frozen agents to the config")
    everyone = [*candidates, *field_players]
    return _unique_displays(candidates, everyone), _unique_displays(field_players, everyone)


def _unique_displays(players: list[TournamentPlayer], everyone: Sequence[TournamentPlayer]) -> list[TournamentPlayer]:
    """``players`` with every display that another player of ``everyone`` shares replaced by the full name (``-a
    bc=other.pt`` next to the config's ``bc``: ``bc#1a2b3c4d`` and ``bc``), so reports that show displays stay
    unambiguous. Names are unique already."""
    displays = [p.display for p in everyone]
    return [replace(p, display=p.name) if p.display != p.name and displays.count(p.display) > 1 else p
            for p in players]


def player_to_dict(player: TournamentPlayer) -> dict[str, Any]:
    data = asdict(player)
    data["roles"] = list(player.roles)
    return data


def player_from_dict(data: Mapping[str, Any]) -> TournamentPlayer:
    src = dict(data["source"])
    bot = src.get("bot")
    source = PlayerSource(kind=src["kind"], origin=src["origin"], weights=src.get("weights"),
                          networks=src.get("networks"),
                          bot=BotSpec(bot["class_path"], dict(bot.get("kwargs") or {})) if bot else None)
    return TournamentPlayer(name=data["name"], display=data["display"], kind=data["kind"], agent_id=data["agent_id"],
                            network_id=data["network_id"], roles=tuple(data["roles"]), lineage=data["lineage"],
                            source=source, version=data.get("version"), run=data.get("run"),
                            run_agent=data.get("run_agent"))

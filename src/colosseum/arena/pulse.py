"""Pulse: evaluating new snapshots against a fixed field during training (SP4 spec block 6).

Main process (``PulseFeeder``): after every snapshot save the launcher offers the snapshot (numpy state dict, as
for workers). Agents of ``evaluation.agents`` (null: every trainable agent); every ``evaluation.every``-th saved
snapshot of each, and the final one always. Requests wait in a backlog of ``max_backlog``; when it is full the
oldest waiting request is dropped (INFO, ``skipped``: newer snapshots matter more). ``pump`` hands the oldest
waiting request to the one-slot request queue whenever the pulse process has taken the previous one (the feeder
then keeps only its ids, not its weights); ``done`` records an evaluated snapshot. ``state()``
(``runs/<name>/pulse_state.json``) lists per agent the versions offered, evaluated, skipped and unevaluated
(waiting or in flight when the run stopped); ``unevaluated_final()`` the final snapshots never evaluated (waiting,
in flight or skipped).

Child process (``pulse_target`` -> ``pulse_main``; started like a worker: ``run_child`` writes ``logs/pulse.log``,
SIGINT is ignored, it dies with the parent, a crash stops the run): one request at a time, the snapshot's model
built with the agent's networks and roles, the gauntlet blocks of spec block 3 (``schedule.gauntlet_blocks`` /
``score_blocks``, stage ``P``: the same seeds for every snapshot; seed ``training.seed`` or 0) against
``evaluation.field`` in ``evaluation.layouts`` (null: the enabled layouts with a seat of the agent's roles),
``evaluation.blocks`` blocks per composition, ``num_envs`` envs, greedy with ``deterministic``. Its seats never
collect. Every result has ``source = pulse`` and the snapshot's ``env_steps``. One snapshot's results go to the
main process in one ``PulseResults`` message (``engine.put_until_stop``: a snapshot finished just after the stop
still goes out when the queue has room). The stop is checked before every env step, so the process exits within
the shutdown grace period; the snapshot being played then stays unevaluated. Compositions the roles cannot seat
are logged once per agent (not once per snapshot). Block ids:
``pulse-<run id[:12]>/P/<layout>/<agent>@v<N>/<composition>/<k>``.

The main process gives pulse results to the journal writer only (the live BT ratings read the journal's table);
SP2/SP3 online ELO, win rates, PFSP and the matchmaker stay training-only.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import queue
from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum.arena.tournament.engine import check_blocks, make_tasks, put_until_stop
from colosseum.arena.tournament.players import PlayerSource, TournamentPlayer, config_player
from colosseum.arena.tournament.schedule import Block, DroppedComposition, gauntlet_blocks, score_blocks
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import FIXED_NETWORK_ID, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.players.registry import FixedPlayers, FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import run_child
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)
_SCHEDULE_LOGGER = "colosseum.arena.tournament.schedule"

PULSE_STATE_FILE = "pulse_state.json"


@dataclass
class PulseRequest:
    """Main -> pulse (numpy only)."""

    agent_id: str
    checkpoint_id: str
    version: int
    env_steps: int
    state_dict: dict[str, np.ndarray]
    final: bool = False


@dataclass
class PulseResults:
    """Pulse -> main: every match of one snapshot (``complete``: all its blocks were played)."""

    agent_id: str
    checkpoint_id: str
    version: int
    results: list[MatchResult]
    complete: bool


@dataclass(frozen=True)
class PulseAgent:
    roles: tuple[str, ...]
    networks: dict[str, Any]
    layouts: tuple[str, ...]


@dataclass(frozen=True)
class PulsePlan:
    agents: dict[str, PulseAgent]
    field: list[str]


@dataclass(frozen=True)
class _Seat:     # what a schedule needs of a player (validation only)
    agent_id: str
    network_id: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class _Ticket:   # a request handed to the pulse process, without its weights
    agent_id: str
    checkpoint_id: str
    version: int
    final: bool


def _plan_seats(agent_id: str, roles: Sequence[str], field: Sequence[str],
                player_roles: Mapping[str, Sequence[str]]) -> dict[str, _Seat]:
    """Schedule stand-ins of a placeholder snapshot ``<agent>@v0`` and the field (validation and the printout)."""
    return {f"{agent_id}@v0": _Seat(agent_id, "ckpt_v0", tuple(roles)),
            **{f: _Seat(f, FIXED_NETWORK_ID, tuple(player_roles[f])) for f in field}}


def pulse_blocks(spec: GameSpec, layouts: Sequence[str], candidate: str, field: Sequence[str],
                 players: Mapping[str, Any], *, blocks: int, seed: int, prefix: str,
                 dropped: list[DroppedComposition] | None = None) -> list[Block]:
    """The pulse gauntlet of one snapshot: stage P blocks of every layout (spec blocks 3 and 6); compositions the
    roles cannot seat are appended to ``dropped``."""
    out: list[Block] = []
    for layout in layouts:
        build = score_blocks if spec.num_teams(layout) == 1 else gauntlet_blocks
        out.extend(build(spec, layout, [candidate], list(field), players, blocks=blocks, seed=seed, stage="P",
                         prefix=prefix, dropped=dropped))
    return out


def pulse_plan(config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]]) -> PulsePlan:
    """The evaluated agents (roles, networks, layouts) and the field, checked (spec block 6: the field names exist
    and are scripted / frozen agents of the config, the agents are trainable, the roles cover the layouts, the
    field is not empty unless every planned layout has one team: those play the homogeneous team). ConfigError
    otherwise."""
    from colosseum.league.lineups import playable_layouts

    ev = config.evaluation
    trainable = config.get_trainable_agent_ids()
    agents = list(ev.agents) if ev.agents is not None else trainable
    for name in agents:
        if name not in trainable:
            raise ConfigError(f"evaluation.agents: {name!r} is not a trainable agent of the config ({trainable})")
    fixed = config.fixed_agent_ids()
    field = list(ev.field) if ev.field is not None else fixed
    for name in field:
        if name not in fixed:
            raise ConfigError(f"evaluation.field: {name!r} is not a scripted or frozen agent of the config ({fixed}); "
                              f"the config names agents only (snapshots are named on the command line: SP3 ruling 9)")
    if ev.layouts is not None:
        unknown = sorted(set(ev.layouts) - set(spec.layouts))
        if unknown:
            raise ConfigError(f"evaluation.layouts {unknown}: not layouts of the game {sorted(spec.layouts)}")
    out: dict[str, PulseAgent] = {}
    for name in agents:
        acfg = config.get_agent_config(name)
        roles = tuple(player_roles[name])
        if ev.layouts is not None:
            layouts = [n for n in ev.layouts if any(seat.role in roles for seat in spec.layouts[n])]
        else:
            layouts = playable_layouts(spec, acfg.matchmaking, roles)
        if not layouts:
            raise ConfigError(f"evaluation: agent {name!r} (roles {list(roles)}) has a seat in none of the layouts "
                              f"{ev.layouts or sorted(spec.layouts)}")
        if not field and any(spec.num_teams(L) >= 2 for L in layouts):   # one-team layouts need no field (B-I1)
            raise ConfigError("evaluation.field is empty: the config has no scripted / frozen agents; add one "
                              "(e.g. kind: scripted, class: colosseum.players.RandomBot) or disable evaluation")
        try:
            pulse_blocks(spec, layouts, f"{name}@v0", field, _plan_seats(name, roles, field, player_roles), blocks=1,
                         seed=0, prefix="pulse")
        except ConfigError as e:
            raise ConfigError(f"evaluation: agent {name!r}: {e}") from e
        out[name] = PulseAgent(roles=roles, networks=acfg.networks.model_dump(mode="json", by_alias=True),
                               layouts=tuple(layouts))
    return PulsePlan(agents=out, field=field)


def describe_pulse(plan: PulsePlan, config: ColosseumConfig, spec: GameSpec, player_roles) -> str:
    """One ``validate`` line with the effective pulse settings and the matches per snapshot."""
    ev = config.evaluation
    matches = {}
    for name, agent in plan.agents.items():
        seats = _plan_seats(name, agent.roles, plan.field, player_roles)
        blocks = pulse_blocks(spec, agent.layouts, f"{name}@v0", plan.field, seats, blocks=ev.blocks, seed=0,
                              prefix="pulse")
        matches[name] = sum(len(b.lineups) for b in blocks)
    per = ", ".join(f"{matches[n]} matches per snapshot of '{n}'" for n in plan.agents)
    return (f"pulse: agents {list(plan.agents)}, every {ev.every} saved snapshot(s) (the final one always); field "
            f"{plan.field}; layouts { {n: list(a.layouts) for n, a in plan.agents.items()} }; {ev.blocks} blocks per "
            f"composition: {per}; {ev.num_envs} envs, {'greedy' if ev.deterministic else 'stochastic'}; backlog "
            f"{ev.max_backlog}")


class PulseFeeder:
    """Main-process side (module docstring). Main loop only."""

    def __init__(self, agents: Collection[str], every: int, max_backlog: int) -> None:
        self._agents = set(agents)
        self._every = int(every)
        self._max_backlog = int(max_backlog)
        self._saved: dict[str, int] = defaultdict(int)
        self._waiting: deque[PulseRequest] = deque()
        self._in_flight: dict[tuple[str, str], _Ticket] = {}
        self._offered: dict[str, list[int]] = defaultdict(list)
        self._evaluated: dict[str, list[int]] = defaultdict(list)
        self._skipped: dict[str, list[int]] = defaultdict(list)
        self._final: dict[str, str] = {}
        self._skipped_final: set[tuple[str, str]] = set()
        self.skipped = 0
        self.evaluated = 0
        self.changed = False

    def offer(self, agent_id: str, checkpoint_id: str, version: int, env_steps: int,
              state_dict: Mapping[str, np.ndarray], *, final: bool = False) -> bool:
        """A saved snapshot; True when it was queued for evaluation."""
        if agent_id not in self._agents:
            return False
        self._saved[agent_id] += 1
        if not final and self._saved[agent_id] % self._every != 0:
            return False
        if final:
            self._final[agent_id] = checkpoint_id
        if len(self._waiting) >= self._max_backlog:
            old = self._waiting.popleft()
            self._skipped[old.agent_id].append(old.version)
            self.skipped += 1
            if old.final:
                self._skipped_final.add((old.agent_id, old.checkpoint_id))
            what = "the final snapshot" if old.final else "snapshot"
            logger.info(f"pulse: {what} {old.checkpoint_id} of {old.agent_id} skipped (backlog of "
                        f"{self._max_backlog} full; newer snapshots go first)")
        self._waiting.append(PulseRequest(agent_id, checkpoint_id, int(version), int(env_steps), dict(state_dict),
                                          final=bool(final)))
        self._offered[agent_id].append(int(version))
        self.changed = True
        return True

    def pump(self, requests: Any) -> int:
        """Move waiting requests into ``requests`` while it takes them (non-blocking); returns how many moved."""
        moved = 0
        while self._waiting:
            try:
                requests.put_nowait(self._waiting[0])
            except queue.Full:
                break
            request = self._waiting.popleft()      # the queue holds the weights now; keep only the ids
            self._in_flight[(request.agent_id, request.checkpoint_id)] = _Ticket(
                request.agent_id, request.checkpoint_id, request.version, request.final)
            moved += 1
        return moved

    def done(self, message: PulseResults) -> None:
        self._in_flight.pop((message.agent_id, message.checkpoint_id), None)
        self._evaluated[message.agent_id].append(int(message.version))
        self.evaluated += 1
        self.changed = True

    @property
    def pending(self) -> int:
        return len(self._waiting) + len(self._in_flight)

    def _unevaluated(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for request in [*self._in_flight.values(), *self._waiting]:
            out[request.agent_id].append(request.version)
        return {a: sorted(v) for a, v in out.items()}

    def unevaluated_final(self) -> list[tuple[str, str]]:
        """``(agent, checkpoint id)`` of final snapshots never evaluated: still waiting, in flight, or skipped."""
        left = {(r.agent_id, r.checkpoint_id) for r in [*self._in_flight.values(), *self._waiting]}
        return [(a, c) for a, c in self._final.items() if (a, c) in left or (a, c) in self._skipped_final]

    def state(self) -> dict[str, Any]:
        unevaluated = self._unevaluated()
        agents = sorted(self._agents)
        return {"offered": {a: list(self._offered.get(a, [])) for a in agents if a in self._offered},
                "evaluated": {a: sorted(self._evaluated.get(a, [])) for a in agents if a in self._offered},
                "skipped": {a: sorted(self._skipped.get(a, [])) for a in agents if a in self._offered},
                "unevaluated": {a: unevaluated.get(a, []) for a in agents if a in self._offered}}


class _PulsePool:
    """``PlayerPool``: the current snapshot under ``(agent, ckpt_v<N>)``, the field under ``(name, "fixed")``."""

    def __init__(self, spec: GameSpec, field_specs: FixedPlayers) -> None:
        self._fixed: dict[tuple[str, str], Any] = {}
        for name, bot in field_specs.bots.items():
            self._fixed[(name, FIXED_NETWORK_ID)] = ScriptedPlayer(functools.partial(make_bot, bot, spec))
        for name, frozen in field_specs.frozen.items():
            self._fixed[(name, FIXED_NETWORK_ID)] = build_frozen_model(None, frozen, spec)
        self._candidate: tuple[tuple[str, str], Any] | None = None

    def set_candidate(self, key: tuple[str, str] | None, model: Any = None) -> None:
        """The snapshot being evaluated (``None``: none; its model is released)."""
        self._candidate = None if key is None else (key, model)

    def get(self, agent_id: str, network_id: str) -> Any:
        if self._candidate is not None and self._candidate[0] == (agent_id, network_id):
            return self._candidate[1]
        return self._fixed.get((agent_id, network_id))


@contextlib.contextmanager
def _quiet(name: str):
    """The logger ``name`` passes WARNING and above only, inside the block (the pulse process is single-threaded)."""
    target = logging.getLogger(name)
    level = target.level
    target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(level)


def pulse_target(*, log_dir: str | None = None, **kwargs: Any) -> None:
    """Process entry point (``mp.Process`` target): logging and the child signal policy first."""
    run_child("pulse", log_dir, pulse_main, **kwargs)


def pulse_main(config: ColosseumConfig, *, requests: Any, results: Any, stop_event: Any, field_specs: FixedPlayers,
               field: Sequence[str], agents: Mapping[str, PulseAgent], run_id: str) -> None:
    """The pulse process body (module docstring); returns on ``None`` or at the stop."""
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.eval import play_lineups

    configure_torch_threads(1)
    spec = env_spec(config)
    ev = config.evaluation
    seed = config.training.seed if config.training.seed is not None else 0
    prefix = f"pulse-{run_id[:12]}"
    pool = _PulsePool(spec, field_specs)
    field_players = {name: config_player(config, spec, name, field_specs) for name in field}
    reported: set[str] = set()       # agents whose unscheduled compositions were logged
    logger.info(f"pulse: agents {list(agents)}, field {list(field)}, {ev.blocks} blocks per composition")

    def evaluate(request: PulseRequest) -> PulseResults | None:
        """One snapshot's matches; None when the stop came first (the snapshot stays unevaluated)."""
        agent = agents[request.agent_id]
        name = f"{request.agent_id}@v{request.version}"
        origin = f"pulse snapshot {request.checkpoint_id} of {request.agent_id}"
        pool.set_candidate((request.agent_id, request.checkpoint_id), build_frozen_model(
            None, FrozenSpec(agent_id=request.agent_id, roles=agent.roles, networks=dict(agent.networks),
                             model_state=request.state_dict, source=origin, networks_source=origin), spec))
        candidate = TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=request.agent_id,
                                     network_id=request.checkpoint_id, roles=agent.roles, lineage=request.agent_id,
                                     source=PlayerSource("model", origin=origin, networks=dict(agent.networks)),
                                     version=request.version)
        dropped: list[DroppedComposition] = []
        with _quiet(_SCHEDULE_LOGGER):     # the same drops for every snapshot: logged once per agent below
            blocks = pulse_blocks(spec, agent.layouts, name, field, {name: candidate, **field_players},
                                  blocks=ev.blocks, seed=seed, prefix=prefix, dropped=dropped)
        if request.agent_id not in reported:
            reported.add(request.agent_id)
            for d in dropped:
                logger.info(f"pulse: agent {request.agent_id!r}, layout {d.layout!r}: composition "
                            f"{'+'.join(d.composition) or '(homogeneous team)'} is not played: {d.reason}")
        check_blocks(blocks, players=[candidate, *field_players.values()])   # one lineup per episode (T1.1)
        played: list[MatchResult] = []
        for task in make_tasks(blocks, ev.num_envs):
            lineups = [lu for b in task for lu in b.lineups]
            got = play_lineups(env_fn=lambda: make_env(config), pool=pool, lineups=lineups, num_envs=ev.num_envs,
                               seed=task[0].seed, deterministic=ev.deterministic,
                               max_idle_steps=config.env.max_idle_steps, source="pulse",
                               context=f"pulse, {name}, ", match_id_prefix="pulse", should_stop=stop_event.is_set)
            if len(got) < len(lineups):
                return None                                  # stopped mid-task
            for result in got:
                result.env_steps = request.env_steps
            played.extend(got)
        played.sort(key=lambda r: r.uid)
        logger.info(f"pulse: {name} evaluated ({len(played)} matches)")
        return PulseResults(request.agent_id, request.checkpoint_id, request.version, played, True)

    while not stop_event.is_set():
        try:
            request = requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if request is None:
            return
        try:
            message = evaluate(request)
        finally:
            pool.set_candidate(None)                         # release the model before the next request
            del request
        if message is None or not put_until_stop(results, message, stop_event):
            return

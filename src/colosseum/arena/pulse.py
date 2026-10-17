"""Pulse: evaluating new snapshots against a fixed field during training (SP4 spec block 6).

Main process (``PulseFeeder``): after every snapshot save the launcher offers the snapshot (numpy state dict, as
for workers). Agents of ``evaluation.agents`` (null: every trainable agent); every ``evaluation.every``-th saved
snapshot of each, and the final one always. Requests wait in a backlog of ``max_backlog``; when it is full the
oldest waiting request is dropped (INFO, ``skipped``: newer snapshots matter more). ``pump`` hands the oldest
waiting request to the one-slot request queue whenever the pulse process has taken the previous one; ``done``
records an evaluated snapshot. ``state()`` (``runs/<name>/pulse_state.json``) lists per agent the versions
offered, evaluated, skipped and unevaluated (waiting or in flight when the run stopped).

Child process (``pulse_target`` -> ``pulse_main``; started like a worker: ``run_child`` writes ``logs/pulse.log``,
SIGINT is ignored, it dies with the parent, a crash stops the run): one request at a time, the snapshot's model
built with the agent's networks and roles, the gauntlet blocks of spec block 3 (``schedule.gauntlet_blocks`` /
``score_blocks``, stage ``P``: the same seeds for every snapshot; seed ``training.seed`` or 0) against
``evaluation.field`` in ``evaluation.layouts`` (null: the enabled layouts with a seat of the agent's roles),
``evaluation.blocks`` blocks per composition, ``num_envs`` envs, greedy with ``deterministic``. Its seats never
collect. Every result has ``source = pulse`` and the snapshot's ``env_steps``. One snapshot's results go to the
main process in one ``PulseResults`` message with a blocking put that gives up only at the stop. Block ids:
``pulse-<run id[:12]>/P/<layout>/<agent>@v<N>/<composition>/<k>``.

The main process gives pulse results to the journal writer only (the live BT ratings read the journal's table);
SP2/SP3 online ELO, win rates, PFSP and the matchmaker stay training-only.
"""

from __future__ import annotations

import functools
import logging
import queue
from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum.arena.tournament.engine import _put, check_blocks, make_tasks
from colosseum.arena.tournament.players import PlayerSource, TournamentPlayer, config_player
from colosseum.arena.tournament.schedule import Block, gauntlet_blocks, score_blocks
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import FIXED_NETWORK_ID, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.players.registry import FixedPlayers, FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import run_child
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)

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


def pulse_blocks(spec: GameSpec, layouts: Sequence[str], candidate: str, field: Sequence[str],
                 players: Mapping[str, Any], *, blocks: int, seed: int, prefix: str) -> list[Block]:
    """The pulse gauntlet of one snapshot: stage P blocks of every layout (spec blocks 3 and 6)."""
    out: list[Block] = []
    for layout in layouts:
        build = score_blocks if spec.num_teams(layout) == 1 else gauntlet_blocks
        out.extend(build(spec, layout, [candidate], list(field), players, blocks=blocks, seed=seed, stage="P",
                         prefix=prefix))
    return out


def pulse_plan(config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]]) -> PulsePlan:
    """The evaluated agents (roles, networks, layouts) and the field, checked (spec block 6: the field names exist
    and are scripted / frozen agents of the config, the agents are trainable, the roles cover the layouts).
    ConfigError otherwise."""
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
    if not field:
        raise ConfigError("evaluation.field is empty: the config has no scripted / frozen agents; add one "
                          "(e.g. kind: scripted, class: colosseum.players.RandomBot) or disable evaluation")
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
        seats = {f"{name}@v0": _Seat(name, "ckpt_v0", roles),
                 **{f: _Seat(f, FIXED_NETWORK_ID, tuple(player_roles[f])) for f in field}}
        try:
            pulse_blocks(spec, layouts, f"{name}@v0", field, seats, blocks=1, seed=0, prefix="pulse")
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
        seats = {f"{name}@v0": _Seat(name, "ckpt_v0", agent.roles),
                 **{f: _Seat(f, FIXED_NETWORK_ID, tuple(player_roles[f])) for f in plan.field}}
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
        self._in_flight: dict[tuple[str, str], PulseRequest] = {}
        self._offered: dict[str, list[int]] = defaultdict(list)
        self._evaluated: dict[str, list[int]] = defaultdict(list)
        self._skipped: dict[str, list[int]] = defaultdict(list)
        self._final: dict[str, str] = {}
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
            logger.info(f"pulse: snapshot {old.checkpoint_id} of {old.agent_id} skipped (backlog of "
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
            request = self._waiting.popleft()
            self._in_flight[(request.agent_id, request.checkpoint_id)] = request
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
        """``(agent, checkpoint id)`` of final snapshots still waiting or in flight."""
        left = {(r.agent_id, r.checkpoint_id) for r in [*self._in_flight.values(), *self._waiting]}
        return [(a, c) for a, c in self._final.items() if (a, c) in left]

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

    def set_candidate(self, key: tuple[str, str], model: Any) -> None:
        self._candidate = (key, model)

    def get(self, agent_id: str, network_id: str) -> Any:
        if self._candidate is not None and self._candidate[0] == (agent_id, network_id):
            return self._candidate[1]
        return self._fixed.get((agent_id, network_id))


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
    logger.info(f"pulse: agents {list(agents)}, field {list(field)}, {ev.blocks} blocks per composition")
    while not stop_event.is_set():
        try:
            request = requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if request is None:
            return
        agent = agents[request.agent_id]
        name = f"{request.agent_id}@v{request.version}"
        origin = f"pulse snapshot {request.checkpoint_id} of {request.agent_id}"
        model = build_frozen_model(None, FrozenSpec(agent_id=request.agent_id, roles=agent.roles,
                                                    networks=dict(agent.networks), model_state=request.state_dict,
                                                    source=origin, networks_source=origin), spec)
        pool.set_candidate((request.agent_id, request.checkpoint_id), model)
        candidate = TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=request.agent_id,
                                     network_id=request.checkpoint_id, roles=agent.roles, lineage=request.agent_id,
                                     source=PlayerSource("model", origin=origin, networks=dict(agent.networks)),
                                     version=request.version)
        blocks = pulse_blocks(spec, agent.layouts, name, field, {name: candidate, **field_players}, blocks=ev.blocks,
                              seed=seed, prefix=prefix)
        check_blocks(blocks, players=[candidate, *field_players.values()])   # one lineup per episode (T1.1)
        played: list[MatchResult] = []
        for task in make_tasks(blocks, ev.num_envs):
            if stop_event.is_set():
                return                                   # stopping: this snapshot stays unevaluated
            got = play_lineups(env_fn=lambda: make_env(config), pool=pool,
                               lineups=[lu for b in task for lu in b.lineups], num_envs=ev.num_envs,
                               seed=task[0].seed, deterministic=ev.deterministic,
                               max_idle_steps=config.env.max_idle_steps, source="pulse",
                               context=f"pulse, {name}, ", match_id_prefix="pulse")
            for result in got:
                result.env_steps = request.env_steps
            played.extend(got)
        played.sort(key=lambda r: r.uid)
        logger.info(f"pulse: {name} evaluated ({len(played)} matches)")
        if not _put(results, PulseResults(request.agent_id, request.checkpoint_id, request.version, played, True),
                    stop_event):
            return

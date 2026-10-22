"""The learners a node runs (SP5 spec 4.4, 4.5, 4.7; A28; controller ruling R24): their lifecycle and their
messages, owned by the node agent's main loop (``colosseum.node.agent``).

A learner assignment (``Welcome.learners`` or ``Command(start_learner)``) downloads its start point first (A28:
``weights_blob`` ``start:<ident>`` is one checkpoint frame of ``Download(kind="start")``; ``init`` and ``""`` need no
download), then waits until every ``Welcome.fixed`` blob is cached
(teacher and ``init`` weights come from there) and is spawned from the main-loop thread (PDEATHSIG). At its first
``VersionPublished`` (after its intake runs) the workers are routed to its listener; every ``VersionPublished``
becomes a ``VersionAnnounce`` (the latest one kept while the hub is away), ``LearnerMetrics`` / ``IntakeReport``
feed ``MetricsBatch``, the global env steps are forwarded (``LearnerEnvSteps``; ruling R28: the hub's last announced
count plus this node's own env steps since its receipt, a lower bound of the global count that never decreases; the
run's ``Welcome`` is the first such count, FX3).
``stop_learner`` sends ``Stop`` (the final snapshot is uploaded by the agent); a learner that ends without being
stopped is ``learner_failed`` (until T4.2: the hub stops the run).

``NodeLearners`` talks to the agent only through ``LearnerHost`` (the agent's ``_LearnerHost`` adapter): the
supervisor, posting into the inbox, sending to a child, setting a worker route, events, a connected-only send, the
transfer pool, downloads, whether the fixed blobs are cached, the node number and the CA prefix. Snapshot uploads
stay with the agent (T4.2's ``node.uploads``), downloads too (T3.1's ``node.fetch``)."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Protocol

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.procs import ChildSpec, ChildSupervisor

logger = logging.getLogger(__name__)

__all__ = ["TRAIN_METRICS_KEPT", "LearnerBlobsDone", "LearnerEntry", "LearnerHost", "NodeLearners"]

TRAIN_METRICS_KEPT = 1000       # learner train records kept while the hub is away (oldest dropped)


@dataclass
class LearnerBlobsDone:
    agent_id: str
    generation: int
    resume_state: dict | None
    error: str | None
    serial: int


@dataclass
class LearnerEntry:
    """A learner this node runs for the run (``state``: ``fetching`` its start blobs, ``ready`` to spawn (once every
    ``Welcome.fixed`` blob is cached), ``starting`` (spawned, no hello yet), ``running``, ``stopping`` (``Stop``
    sent))."""

    agent_id: str
    generation: int
    assignment: Any                    # pb.LearnerAssignment
    name: str
    state: str = "fetching"
    resume_state: dict | None = None
    listener_name: str = ""
    version: int = -1
    stop_reason: str = ""
    failure: str = ""                  # why the node gave up on it (its learner_failed message)
    routed: bool = False               # the workers were routed to its listener


class LearnerHost(Protocol):
    """What ``NodeLearners`` uses of the node agent."""

    @property
    def supervisor(self) -> ChildSupervisor: ...
    @property
    def number(self) -> int: ...
    @property
    def ca8(self) -> str: ...
    def current_run(self) -> Any: ...
    def post(self, item: object) -> None: ...
    def send_child(self, name: str, message: Any) -> bool: ...
    def set_route(self, route: Any) -> None: ...
    def event(self, kind: str, **fields: Any) -> None: ...
    def send_if_connected(self, message: pb.NodeMessage) -> bool: ...
    def send(self, message: pb.NodeMessage) -> bool: ...
    def submit(self, fn: Callable, *args: Any) -> Future: ...
    def download_bytes(self, run: Any, kind: str, agent_id: str, ident: str) -> bytes: ...
    def fixed_cached(self, run: Any) -> bool: ...
    def rpc_text(self, error: BaseException) -> str: ...


class NodeLearners:
    """The node's learners (module docstring). ``train_metrics`` / ``intake_stats`` are what the agent puts into its
    next ``MetricsBatch`` (it clears them after a successful send)."""

    def __init__(self, host: LearnerHost) -> None:
        self._host = host
        self._entries: dict[str, LearnerEntry] = {}          # agent -> its learner on this node
        self._pending_versions: dict[str, pb.NodeMessage] = {}   # agent -> VersionAnnounce not sent yet (hub away)
        self.train_metrics: deque = deque(maxlen=TRAIN_METRICS_KEPT)
        self.intake_stats: dict[str, pb.IntakeStats] = {}
        self.hub_env_steps = 0                               # the estimate forwarded to the learners (R28)
        self._announced = 0                                  # the hub's last Announce.env_steps
        self._own_at_announce = 0                            # this node's own env steps when it arrived
        self._based = False                                  # a count arrived this run (the Welcome's at the latest)

    # ---- books --------------------------------------------------------------------------------------------------

    def of(self, name: str) -> LearnerEntry | None:
        return next((e for e in self._entries.values() if e.name == name), None)

    def note_failure(self, name: str, text: str) -> None:
        """The node closed ``name``'s connection for ``text``: its ``learner_failed`` says so."""
        entry = self.of(name)
        if entry is not None and not entry.failure:
            entry.failure = text

    def agents(self) -> list[str]:
        return list(self._entries)

    def routed(self) -> dict[str, str]:
        """Agents whose learner runs here and takes chunks (``running``, workers routed to it) -> its listener name:
        the node opens its own chunk connection to it for frames from other machines (T3.2)."""
        return {e.agent_id: e.listener_name for e in self._entries.values() if e.state == "running" and e.routed}

    def hello_entries(self) -> tuple:
        """``(agent, generation, version)`` of the running learners (``Hello.learners``)."""
        return tuple((e.agent_id, e.generation, max(e.version, 0)) for e in self._entries.values()
                     if e.state in ("running", "stopping"))

    def clear(self) -> None:
        self._entries.clear()
        self._pending_versions.clear()
        self.intake_stats.clear()
        self.hub_env_steps = self._announced = self._own_at_announce = 0   # the next run's counter starts anew
        self._based = False

    def resend_pending(self) -> None:
        """After ``SessionUp``: the version announcements kept while the hub was away."""
        for agent, message in list(self._pending_versions.items()):
            if self._host.send(message):
                del self._pending_versions[agent]

    def forward_env_steps(self, env_steps: int, own: int = 0) -> None:
        """``Welcome.env_steps`` / ``Announce.env_steps`` (the hub's global counter) received while this node's
        workers had taken ``own`` env steps in total: it becomes the base of the estimate (R28) when it is the run's
        first count (the Welcome's, 0 for a new run: own steps count from the run's start, FX3) or it grew; then
        ``advance(own)``."""
        if not self._based or int(env_steps) > self._announced:
            self._announced, self._own_at_announce, self._based = int(env_steps), int(own), True
        self.advance(own)

    def advance(self, own: int) -> None:
        """Ruling R28: ``LearnerEnvSteps`` to every learner when the estimate grew. The estimate is the last announced
        count plus this node's own env steps since that announce arrived (``own``: the workers' cumulative total), and
        never decreases: own steps after the receipt are not in the announced value, so it is never ahead of the
        global count; the hub's next ``Announce`` catches up with the other nodes' steps."""
        from colosseum.net.local import LearnerEnvSteps

        estimate = self._announced + max(0, int(own) - self._own_at_announce) if self._based else 0
        if estimate > self.hub_env_steps:
            self.hub_env_steps = estimate
            for entry in self._entries.values():
                self._host.send_child(entry.name, LearnerEnvSteps(self.hub_env_steps))

    # ---- lifecycle ----------------------------------------------------------------------------------------------

    def assign(self, run: Any, la: pb.LearnerAssignment) -> None:
        """A learner assignment (``Welcome.learners`` or ``start_learner``): its start blobs are downloaded first;
        an agent whose learner runs here already keeps it (generations change only from T4.1 on)."""
        agent = la.agent_id
        if agent not in run.agent_ids:
            logger.warning(f"node: refused a learner assignment for {agent!r}: no trainable agent of this run")
            return
        current = self._entries.get(agent)
        if current is not None:
            if current.generation != int(la.generation):
                logger.warning(f"node: learner {agent} generation {la.generation} assigned while generation "
                               f"{current.generation} runs here; ignored")
            return
        assignment = pb.LearnerAssignment()
        assignment.CopyFrom(la)
        entry = self._entries[agent] = LearnerEntry(agent, int(la.generation), assignment, f"learner-{agent}")
        if la.weights_blob in ("", "init"):          # factory weights / the init blob of Welcome.fixed
            entry.state = "ready"
            return
        serial, generation = run.serial, entry.generation
        future = self._host.submit(self._fetch_blobs, run, assignment)

        def done(f: Future) -> None:
            error = ("cancelled" if f.cancelled()
                     else (None if f.exception() is None else self._host.rpc_text(f.exception())))
            self._host.post(LearnerBlobsDone(agent, generation, None if error else f.result(), error, serial))

        future.add_done_callback(done)

    def _fetch_blobs(self, run: Any, la: pb.LearnerAssignment) -> dict:
        """A transfer-pool thread: the start state of an assignment (A28: ``start:<ident>`` names one checkpoint
        frame the hub serves as ``Download(kind="start", ident=<ident>)``)."""
        from colosseum.net.frame import payload_byte_cap
        from colosseum.node.learner_main import resume_state_from_frame

        kind, _, ident = la.weights_blob.partition(":")
        if kind != "start" or not ident:
            raise ValueError(f"unknown learner start point {la.weights_blob!r}")
        frame = self._host.download_bytes(run, "start", la.agent_id, ident)
        cap = payload_byte_cap(run.config.distributed.max_message_mb)
        return resume_state_from_frame(frame, agent_id=la.agent_id, cap=cap, env_steps=int(la.env_steps),
                                       source=f"{ident} of the hub")

    def on_blobs(self, done: LearnerBlobsDone) -> None:
        run = self._host.current_run()
        entry = self._entries.get(done.agent_id)
        if (run is None or run.serial != done.serial or entry is None or entry.state != "fetching"
                or entry.generation != done.generation):
            return
        if done.error is not None:
            del self._entries[done.agent_id]
            logger.error(f"node: the start state of learner {done.agent_id} could not be fetched: {done.error}")
            self._host.event("learner_failed", proc=entry.name, message=f"start state not fetched: {done.error}",
                             generation=entry.generation)
            return
        entry.resume_state = done.resume_state
        entry.state = "ready"

    def start_pending(self, run: Any) -> None:
        ready = [e for e in self._entries.values() if e.state == "ready"]
        if not ready or not self._host.fixed_cached(run):
            return                                   # teacher / init weights come from the fixed blobs
        for entry in ready:
            try:
                self._start(run, entry)
            except Exception as e:  # noqa: BLE001 - this learner cannot start: the hub decides (until T4.2: stop)
                del self._entries[entry.agent_id]
                logger.exception(f"node: learner {entry.agent_id} cannot start")
                self._host.event("learner_failed", proc=entry.name, message=f"{type(e).__name__}: {e}",
                                 generation=entry.generation)

    def _start(self, run: Any, entry: LearnerEntry) -> None:
        """Spawn ``entry``'s learner (from the main-loop thread only: PDEATHSIG)."""
        from colosseum.net.frame import payload_byte_cap
        from colosseum.net.local import address, learner_proc_name
        from colosseum.node.learner_main import init_from_cached, learner_target, teacher_for
        from colosseum.utils.seeding import learner_seed

        supervisor = self._host.supervisor
        agent, la = entry.agent_id, entry.assignment
        incarnation = supervisor.incarnation(entry.name) + 1
        entry.listener_name = address(self._host.ca8, run.run_id[:8], run.number,
                                      learner_proc_name(agent, incarnation))
        config = run.agent_configs[agent]
        if la.device:
            config = config.model_copy(update={"learner": config.learner.model_copy(update={"device": la.device})})

        def cached(kind: str) -> Any:
            key = next((k for k, k_kind in run.fixed_keys.items() if k_kind == kind and k.agent_id == agent), None)
            if key is None:
                return None
            weights = run.reader.get(key)
            if weights is None:
                raise RuntimeError(f"the {kind} blob of {agent!r} is not in the weight cache")
            return weights

        teacher = teacher_for(run.config, agent, cached=cached("teacher"), bot=run.teachers.get(agent))
        init = cached("init")
        kwargs = dict(name=entry.name, log_dir=str(run.log_dir), node=run.number, agent_id=agent,
                      generation=entry.generation, listener_name=entry.listener_name,
                      node_address=run.node_address, authkey=run.authkey, config=config,
                      role_spec=run.role_specs[agent], spec=run.spec,
                      seed=learner_seed(run.config.training.seed, run.agent_ids.index(agent)), teacher=teacher,
                      init_state=None if init is None else init_from_cached(run.config, agent, init),
                      resume_state=entry.resume_state, version_floor=int(la.version_floor),
                      env_steps=max(int(la.env_steps), self.hub_env_steps), num_learners=len(self._entries),
                      cache_root=str(run.root), cache_prefix=run.prefix,
                      checkpoint_interval=int(run.config.checkpoint.interval),
                      total_timesteps=int(run.config.training.total_timesteps),
                      cap_bytes=payload_byte_cap(run.config.distributed.max_message_mb))
        supervisor.start(ChildSpec(entry.name, "learner", learner_target, kwargs))
        entry.state, entry.resume_state = "starting", None
        logger.info(f"node: started learner {agent} (generation {entry.generation}, version floor "
                    f"{la.version_floor})")

    def on_connected(self, name: str) -> None:
        """The learner's control hello. Workers are routed to its listener only at its first ``VersionPublished``
        (``learner_process`` publishes before it trains, after its intake runs): a learner whose start failed opens
        the control connection only to report the error, and its listener never existed."""
        from colosseum.net.local import LearnerEnvSteps, Stop

        entry = self.of(name)
        if entry is None:
            return
        if entry.state == "stopping":
            self._host.send_child(name, Stop(entry.stop_reason or "stop_learner"))
            return
        entry.state = "running"
        if self.hub_env_steps:
            self._host.send_child(name, LearnerEnvSteps(self.hub_env_steps))

    def stop(self, agent: str, reason: str) -> None:
        """``stop_learner``: ``Stop`` to the learner (it sends its final snapshot, which is uploaded); workers no
        longer send it chunks."""
        from colosseum.net.local import LearnerRoute, Stop

        entry = self._entries.get(agent)
        if entry is None:
            return
        if entry.state in ("fetching", "ready"):
            del self._entries[agent]                 # never started (a fetch still running is ignored on arrival)
            return
        entry.state, entry.stop_reason = "stopping", reason
        self._host.send_child(entry.name, Stop(reason))
        self._host.set_route(LearnerRoute(agent, None, False, 0))
        logger.info(f"node: stopping learner {agent} ({reason})")

    def on_exit(self, name: str, code: int | None, log_path: str, routes: dict) -> None:
        """A learner process ended (not during the node's stop); ``routes``: the agent's current worker routes."""
        from colosseum.net.local import LearnerRoute

        entry = self.of(name)
        if entry is None:
            return
        del self._entries[entry.agent_id]
        route = routes.get(entry.agent_id)
        if route is not None and route.address is not None:
            self._host.set_route(LearnerRoute(entry.agent_id, None, False, 0))
        if entry.state == "stopping" and code == 0 and not entry.failure:
            logger.info(f"node: learner {entry.agent_id} stopped")
            return
        why = entry.failure or "without being stopped"
        logger.error(f"node: learner {entry.agent_id} ended (exit {code}): {why}; see {log_path}")
        self._host.event("learner_failed", proc=name, exit_code=int(code or 0), log_path=log_path,
                         generation=entry.generation, message=entry.failure)

    # ---- messages -----------------------------------------------------------------------------------------------

    def on_message(self, name: str, message: Any) -> None:
        """``VersionPublished`` -> ``VersionAnnounce``; ``LearnerMetrics`` / ``IntakeReport`` -> ``MetricsBatch``."""
        from colosseum.net.local import IntakeReport, LearnerMetrics, VersionPublished
        from colosseum.net.messages import VersionInfo, version_to_proto

        entry = self.of(name)
        if entry is None:
            logger.warning(f"node: unexpected {type(message).__name__} from {name}")
        elif isinstance(message, VersionPublished):
            from colosseum.net.local import LearnerRoute

            entry.version = max(entry.version, int(message.version))
            # The route goes to the workers before the VersionAnnounce below goes to the hub, and the hub grants
            # collecting seats for an agent only once it accepted a version: on a first start no lineup collects for
            # this learner before every worker of this node has its route (T3.2, carried item of T2.4 (b)).
            if not entry.routed and entry.state == "running":
                entry.routed = True
                self._host.set_route(LearnerRoute(entry.agent_id, entry.listener_name, False, entry.generation))
            info = VersionInfo(agent_id=entry.agent_id, generation=int(message.generation),
                               version=int(message.version), size=int(message.size), digest=message.digest,
                               node=self._host.number, teacher_active=bool(message.teacher_active))
            announce = pb.NodeMessage(version=pb.VersionAnnounce(info=version_to_proto(info)))
            if self._host.send_if_connected(announce):
                self._pending_versions.pop(entry.agent_id, None)
            else:
                self._pending_versions[entry.agent_id] = announce        # only the latest matters
        elif isinstance(message, LearnerMetrics):
            values = {k: float(v) for k, v in message.metrics.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)}
            self.train_metrics.append(pb.TrainMetrics(
                agent_id=entry.agent_id, generation=entry.generation,
                train_step=int(message.metrics.get("train_step", 0)), values=values))
        elif isinstance(message, IntakeReport):
            self.intake_stats[entry.agent_id] = pb.IntakeStats(
                agent_id=entry.agent_id, generation=entry.generation, depth=int(message.depth),
                dropped={k: int(v) for k, v in message.dropped.items()})
        else:
            logger.warning(f"node: unexpected {type(message).__name__} from {name}")

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
``stop_learner`` sends ``Stop`` (the final snapshot is uploaded by the agent; ``superseded`` sends none); a learner
that ends without being stopped is reported (T4.2, spec 7.1): ``proc_died`` within ``procs.LEARNER_RESTARTS`` (the hub
assigns it again with a new generation from its recovery point), else ``learner_failed``; both name its generation,
exit code and reason (a ``ChildError`` it sent, ``note_failure``) and carry its log tail.

Generations (SP5 T4.1, spec 7.2): at most one learner per agent; the same assignment twice (``Welcome.learners`` and
the hub's deferred ``start_learner``) is one start; a newer generation replaces the running one (``superseded``: it
exits without a final, then the new one starts as a new incarnation with a new intake address); an older one is
ignored; ``stop_learner`` stops only the generation it names. ``Welcome.learners`` is the node's desired learner set
(``NodeAgent._on_welcome`` stops the others as superseded). A learner starts at ``max(assignment floor, the
machine's cached latest + 1, every version a learner of the agent published here + 1)`` (R9: no version names two
weights) with the seed ``learner_seed(training.seed, agent index, generation)``. When an agent's learner here ends
for good, the node removes the agent's latest weights from the cache (``WeightCacheWriter.release_latest``): it
becomes their writer again (fetched from the agent's next learner) and is never held back by the dead learner's pointer;
the workers take the next learner's weights although their versions may lie below the dead one's, because they accept
a pointer of another generation at any version (controller ruling R40, ``WeightCacheReader.read_latest``). A learner
that does not exit within ``distributed.shutdown_grace_sec`` after its ``Stop`` is terminated; a deliberately stopped
learner is never ``learner_failed``. A start point that cannot be fetched is fetched again every
``START_RETRY_SEC`` (one WARNING); the learner never starts without it.

``NodeLearners`` talks to the agent only through ``LearnerHost`` (the agent's ``_LearnerHost`` adapter): the
supervisor, posting into the inbox, sending to a child, setting a worker route, events, a connected-only send, the
transfer pool, downloads, whether the fixed blobs are cached, the node number and the CA prefix. Snapshot uploads
stay with the agent (T4.2's ``node.uploads``), downloads too (T3.1's ``node.fetch``)."""

from __future__ import annotations

import logging
import time
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
START_RETRY_SEC = 5.0           # a start point that could not be fetched is fetched again after this (T4.1)
KILL_AFTER_TERMINATE_SEC = 2.0  # a stopped learner still alive this long after SIGTERM is killed


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
    successor: Any = None              # a newer pb.LearnerAssignment that starts once this one exited (T4.1)
    stop_deadline: float | None = None  # monotonic: a stopping learner still alive then is terminated (T4.1)
    terminated_at: float | None = None
    retry_at: float | None = None      # fetching: its start point is fetched again then (None: a fetch in flight)
    fetch_warned: bool = False


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
        self._published: dict[str, int] = {}                 # agent -> highest version a learner published here

    # ---- books --------------------------------------------------------------------------------------------------

    def of(self, name: str) -> LearnerEntry | None:
        return next((e for e in self._entries.values() if e.name == name), None)

    def note_failure(self, name: str, text: str) -> None:
        """Why ``name`` is about to end (a ``ChildError`` it sent, or the node closed its connection for ``text``):
        its ``proc_died`` / ``learner_failed`` says so (the first reason wins)."""
        entry = self.of(name)
        if entry is not None and not entry.failure:
            entry.failure = text

    def agents(self) -> list[str]:
        return list(self._entries)

    def routed(self) -> dict[str, str]:
        """Agents whose learner runs here and takes chunks (``running``, workers routed to it) -> its listener name:
        the node opens its own chunk connection to it for frames from other machines (T3.2)."""
        return {e.agent_id: e.listener_name for e in self._entries.values() if e.state == "running" and e.routed}

    def generations(self) -> dict[str, int]:
        """Agent -> generation of every learner this node has (fetching, starting, running or stopping)."""
        return {a: e.generation for a, e in self._entries.items()}

    def queued(self) -> dict[str, int]:
        """Agent -> generation of a successor queued behind a learner that is stopping."""
        return {a: int(e.successor.generation) for a, e in self._entries.items() if e.successor is not None}

    def hello_entries(self) -> tuple:
        """``(agent, generation, version)`` of the running learners (``Hello.learners``)."""
        return tuple((e.agent_id, e.generation, max(e.version, 0)) for e in self._entries.values()
                     if e.state in ("running", "stopping"))

    def clear(self) -> None:
        self._entries.clear()
        self._pending_versions.clear()
        self._published.clear()
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
        """A learner assignment (``Welcome.learners`` or ``start_learner``; module docstring): the same generation
        again is one start, an older one is ignored, a newer one supersedes the agent's learner here (a learner not
        spawned yet is replaced at once; a spawned one is stopped and the new one starts when it exited). The start
        point is downloaded first (A28)."""
        agent = la.agent_id
        if agent not in run.agent_ids:
            logger.warning(f"node: refused a learner assignment for {agent!r}: no trainable agent of this run")
            return
        generation = int(la.generation)
        assignment = pb.LearnerAssignment()
        assignment.CopyFrom(la)
        current = self._entries.get(agent)
        if current is not None:
            queued = 0 if current.successor is None else int(current.successor.generation)
            if generation in (current.generation, queued):
                return                               # the same assignment again (Welcome + start_learner): one start
            if generation < max(current.generation, queued):
                logger.info(f"node: learner {agent} generation {generation} is older than generation "
                            f"{max(current.generation, queued)} here; ignored")
                return
            if current.state in ("fetching", "ready"):
                del self._entries[agent]             # never spawned: replaced at once (a late fetch is ignored)
            else:
                current.successor = assignment
                logger.info(f"node: learner {agent} generation {generation} supersedes generation "
                            f"{current.generation}; it starts once that one exited")
                if current.state != "stopping":
                    self.stop(agent, "superseded", current.generation)
                return
        entry = self._entries[agent] = LearnerEntry(agent, int(la.generation), assignment, f"learner-{agent}")
        if la.weights_blob in ("", "init"):          # factory weights / the init blob of Welcome.fixed
            entry.state = "ready"
            return
        self._submit_fetch(run, entry)

    def _submit_fetch(self, run: Any, entry: LearnerEntry) -> None:
        """Download ``entry``'s start point in the transfer pool; the result comes back as ``LearnerBlobsDone``."""
        agent, assignment = entry.agent_id, entry.assignment
        serial, generation = run.serial, entry.generation
        entry.retry_at = None
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
        if done.error is not None:                   # retried at a later pass; never started without its start point
            entry.retry_at = time.monotonic() + START_RETRY_SEC
            if not entry.fetch_warned:
                entry.fetch_warned = True
                logger.warning(f"node: the start point of learner {done.agent_id} (generation {entry.generation}, "
                               f"{entry.assignment.weights_blob}) could not be fetched: {done.error}; retried every "
                               f"{START_RETRY_SEC:g} s")
            return
        entry.resume_state = done.resume_state
        entry.state = "ready"

    def start_pending(self, run: Any, *, spawn: bool = True) -> None:
        """Main loop, every pass: fetch start points again that are due, spawn the learners that are ready (not while
        ``spawn`` is False: the node's cache is full or its processes are stopped while the hub is away, T4.4; they
        start once it is True again), and terminate a stopping learner past its grace (``stop``)."""
        now = time.monotonic()
        for entry in list(self._entries.values()):
            if entry.state == "fetching" and entry.retry_at is not None and now >= entry.retry_at:
                self._submit_fetch(run, entry)
            elif entry.state == "stopping":
                self._enforce_stop(entry, now)
        ready = [e for e in self._entries.values() if e.state == "ready"]
        if not spawn or not ready or not self._host.fixed_cached(run):
            return                                   # teacher / init weights come from the fixed blobs
        for entry in ready:
            try:
                self._start(run, entry)
            except Exception as e:  # noqa: BLE001 - this learner cannot start: the hub decides (until T4.2: stop)
                del self._entries[entry.agent_id]
                logger.exception(f"node: learner {entry.agent_id} cannot start")
                self._host.event("learner_failed", proc=entry.name, message=f"{type(e).__name__}: {e}",
                                 generation=entry.generation)

    def floor(self, run: Any, agent: str, assigned: int) -> int:
        """The version floor of a learner of ``agent`` starting here (R9, W04-7): the assignment's, above the
        machine's cached latest pointer (also one a dead incarnation published without announcing it) and above every
        version a learner of the agent published here."""
        cached = run.reader.latest_version(agent)
        return max(int(assigned), -1 if cached is None else int(cached[0]) + 1, self._published.get(agent, -1) + 1)

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
        floor = self.floor(run, agent, int(la.version_floor))
        start = la.weights_blob or "factory"
        kwargs = dict(name=entry.name, log_dir=str(run.log_dir), node=run.number, agent_id=agent,
                      generation=entry.generation, listener_name=entry.listener_name,
                      node_address=run.node_address, authkey=run.authkey, config=config,
                      role_spec=run.role_specs[agent], spec=run.spec,
                      seed=learner_seed(run.config.training.seed, run.agent_ids.index(agent), entry.generation),
                      teacher=teacher, init_state=None if init is None else init_from_cached(run.config, agent, init),
                      resume_state=entry.resume_state, version_floor=floor, start_point=start,
                      env_steps=max(int(la.env_steps), self.hub_env_steps), num_learners=len(self._entries),
                      cache_root=str(run.root), cache_prefix=run.prefix,
                      checkpoint_interval=int(run.config.checkpoint.interval),
                      total_timesteps=int(run.config.training.total_timesteps),
                      cap_bytes=payload_byte_cap(run.config.distributed.max_message_mb))
        supervisor.start(ChildSpec(entry.name, "learner", learner_target, kwargs))
        entry.state, entry.resume_state = "starting", None
        logger.info(f"node: started learner {agent} (generation {entry.generation}, device {la.device or 'config'}, "
                    f"versions from {floor}, start {start}, listener {entry.listener_name})")

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

    def stop(self, agent: str, reason: str, generation: int | None = None) -> None:
        """``stop_learner``: ``Stop`` to the learner of ``generation`` (None: whichever runs; another generation is
        ignored, a queued successor of that generation is dropped); it sends its final snapshot (none for
        ``superseded``); workers no longer send it chunks."""
        from colosseum.net.local import LearnerRoute, Stop

        entry = self._entries.get(agent)
        if entry is None:
            return
        if generation is not None and int(generation) != entry.generation:
            if entry.successor is not None and int(entry.successor.generation) == int(generation):
                entry.successor = None
                logger.info(f"node: the queued learner {agent} generation {generation} is cancelled ({reason})")
            else:
                logger.debug(f"node: stop_learner {agent} generation {generation} ignored (generation "
                             f"{entry.generation} here)")
            return
        if entry.state == "stopping":
            return
        if entry.state in ("fetching", "ready"):
            del self._entries[agent]                 # never started (a fetch still running is ignored on arrival)
            return
        entry.state, entry.stop_reason = "stopping", reason
        run = self._host.current_run()
        grace = 0.0 if run is None else float(run.config.distributed.shutdown_grace_sec)
        entry.stop_deadline = time.monotonic() + grace
        self._host.send_child(entry.name, Stop(reason))
        self._host.set_route(LearnerRoute(agent, None, False, 0))
        logger.info(f"node: stopping learner {agent} ({reason})")

    def stop_every(self, reason: str) -> None:
        """Stop every learner of this node with ``reason`` and drop the successors queued behind them (the hub away
        past ``distributed.hub_lost_timeout_sec``, T4.4: ``hub_lost`` sends no final snapshot, P-PF3-19; the hub's next
        ``Welcome`` names the learners to start again)."""
        for entry in list(self._entries.values()):
            entry.successor = None
            self.stop(entry.agent_id, reason, entry.generation)

    def _enforce_stop(self, entry: LearnerEntry, now: float) -> None:
        """A stopping learner still alive after ``distributed.shutdown_grace_sec`` is terminated (spec 7.2 / Step
        7.2), and killed ``KILL_AFTER_TERMINATE_SEC`` later; its exit then reaches ``on_exit``."""
        if entry.stop_deadline is None or now < entry.stop_deadline:
            return
        proc = self._host.supervisor.get(entry.name)
        if proc is None or proc.exitcode is not None:
            return
        if entry.terminated_at is None:
            entry.terminated_at = now
            logger.warning(f"node: learner {entry.agent_id} generation {entry.generation} did not stop within the "
                           f"grace ({entry.stop_reason}); terminated")
            proc.terminate()
        elif now - entry.terminated_at >= KILL_AFTER_TERMINATE_SEC:
            entry.terminated_at = float("inf")       # killed once
            proc.kill()

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
        run = self._host.current_run()
        stopped = entry.state == "stopping"         # a deliberate stop: never learner_failed, whatever the exit
        if stopped and (code != 0 or entry.failure):
            logger.warning(f"node: learner {entry.agent_id} generation {entry.generation} ended after its stop "
                           f"({entry.stop_reason}) with exit {code}{': ' + entry.failure if entry.failure else ''}; "
                           f"see {log_path}")
        if entry.successor is not None and run is not None:
            logger.info(f"node: learner {entry.agent_id} generation {entry.generation} ended (exit {code}); "
                        f"generation {entry.successor.generation} starts")
            self.assign(run, entry.successor)        # the next writer of the agent's latest: its floor is above
            return
        if run is not None:
            self._release_latest(run, entry.agent_id)
        if stopped:
            logger.info(f"node: learner {entry.agent_id} stopped")
            return
        self._report_death(entry, code, log_path)

    def _report_death(self, entry: LearnerEntry, code: int | None, log_path: str) -> None:
        """A learner that ended without being stopped (spec 7.1; T4.2): the node never restarts it itself (the hub
        assigns a new generation with a start point only it holds, W04-6): within ``LEARNER_RESTARTS`` (counted per
        name on this node, across generations) it is ``proc_died``, after the limit ``learner_failed``; both carry its
        generation, exit code, the reason it gave (a ``ChildError``, an oversized checkpoint) and the tail of its
        log."""
        from pathlib import Path

        from colosseum.node import procs
        from colosseum.node.logs import log_tail

        name = entry.name
        why = entry.failure or "without being stopped"
        tail = log_tail(Path(log_path)) if log_path else b""
        fields = dict(proc=name, exit_code=int(code or 0), log_path=log_path, log_tail=tail,
                      generation=entry.generation)
        if self._host.supervisor.may_restart(name):
            logger.error(f"node: learner {entry.agent_id} (generation {entry.generation}) ended (exit {code}): {why}; "
                         f"the hub restarts it; see {log_path}")
            self._host.event("proc_died", message=entry.failure, **fields)
            return
        limit, window = procs.LEARNER_RESTARTS
        reason = f"{limit} restarts within {window:g} s on this node" + (f"; last: {entry.failure}"
                                                                          if entry.failure else "")
        logger.error(f"node: learner {entry.agent_id} (generation {entry.generation}) ended (exit {code}): {why}; "
                     f"{limit} restarts within {window:g} s: it is not restarted; see {log_path}")
        self._host.event("learner_failed", message=reason, **fields)

    def _release_latest(self, run: Any, agent: str) -> None:
        """The agent's learner here ended for good: remember its versions for a later floor, then the node gives the
        agent's latest weights up (``WeightCacheWriter.release_latest``; module docstring)."""
        cached = run.reader.latest_version(agent)
        if cached is not None:
            self._published[agent] = max(self._published.get(agent, -1), int(cached[0]))
        try:
            run.writer.release_latest(agent)
        except OSError as e:
            logger.warning(f"node: the cached latest weights of {agent} could not be removed ({e})")

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
            self._published[entry.agent_id] = max(self._published.get(entry.agent_id, -1), entry.version)
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

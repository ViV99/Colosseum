"""The node's routing section (SP5 spec 4.7, 4.8; T3.2): chunks between machines, owned by the node agent's main loop
(``colosseum.node.agent``, item 12).

Each run has a ``node.router.ChunkRouter`` (``start``), created before the data port (a new data port takes the
router's ``LocalDelivery`` at creation, the process's existing one through ``DataServer.set_delivery``; the router is
per run, the port per process). Every pass at the fetch cadence (``update_routes``): every trainable agent whose newest
known version comes from another node (not parked: node -1 is no route, pre-flight ruling P-PF3-13) and whose learner
does not run here is a ``RemoteRoute`` (its node, generation, data addresses); the router starts a ``TargetSender``
per target node at once, which probes the path (``direct`` within ``distributed.connect_timeout_sec``, else the hub
relay (T3.2 (b)), else ``down``) and reports it (``PathChanged`` into the inbox; ``Heartbeat.paths``), so the hub
grants collecting seats only where the learner is reachable (P-PF3-1). Workers get ``LearnerRoute(agent, None,
via_node=True)`` for such agents while the path is not ``down`` (``via_node=False`` otherwise: the worker drops and
counts ``no_learner``): their frames come on the workers' ``route`` connections (``add_worker``: one ``RouteReader``
each; nothing is decoded) and leave in one gRPC stream per target. A path change to ``relay`` / ``down`` is one WARNING
per target per node process and one ``Event(warning)`` to the hub.

Frames from other machines arrive at the data port's ``Chunks`` (T3.2 (b): also through the hub relay) and go to the
local learner over the node's own ``chunks`` connection to that learner incarnation (``sync_delivery``: opened by a
short thread off the loop, ``LearnerConnOpened``; reopened at most every ``DELIVERY_RETRY_SEC`` after a failure).

The router's counters (``chunks_*``, ``dropped_chunks.*``) are cumulative per node process: a stopped router's values
are carried (``counters``). ``stop`` (before the node kills its children, at a release and at the end) stops every
router thread; frames still queued are counted ``disconnect``.

``NodeRouting`` talks to the agent only through ``RoutingHost`` (the agent's ``_RoutingHost`` adapter)."""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from colosseum.net.proto import colosseum_v1_pb2 as pb

logger = logging.getLogger(__name__)

__all__ = ["DELIVERY_RETRY_SEC", "LEARNER_CONNECT_SEC", "LearnerConnOpened", "NodeRouting", "PathChanged",
           "RoutingHost"]

DELIVERY_RETRY_SEC = 1.0        # the node's chunk connection to a local learner is (re)opened at most this often
LEARNER_CONNECT_SEC = 5.0


@dataclass
class PathChanged:
    """A chunk sender's path to ``target`` changed (``node.router.PATHS``; posted by its thread)."""

    serial: int                        # the run whose router reported it
    target: int
    path: str


@dataclass
class LearnerConnOpened:
    """The node's chunk connection to a local learner opened (``conn``) or not (``error``)."""

    serial: int
    agent_id: str
    listener: str
    conn: Any
    error: str


@dataclass
class _DeliveryLink:
    listener: str
    opening: bool = False
    next_try: float = 0.0


class RoutingHost(Protocol):
    """What ``NodeRouting`` uses of the node agent."""

    @property
    def number(self) -> int: ...
    @property
    def stopping(self) -> bool: ...
    def current_run(self) -> Any: ...
    def post(self, item: object) -> None: ...
    def planner(self) -> Any: ...                     # node.fetch.FetchPlanner | None
    def data_link(self) -> Any: ...                   # node.fetch._DataLink | None
    def local_learners(self) -> list[str]: ...        # agents whose learner runs here
    def routed_learners(self) -> dict[str, str]: ...  # agent -> listener of a running, routed local learner
    def worker_route(self, agent_id: str) -> Any: ... # the LearnerRoute the workers have (None: none sent)
    def set_route(self, route: Any) -> None: ...
    def event(self, kind: str, **fields: Any) -> None: ...
    def set_delivery(self, delivery: Any) -> None: ...   # the data port's (when it exists)


class NodeRouting:
    """The node's chunk routing (module docstring). Every method runs on the main loop."""

    def __init__(self, host: RoutingHost) -> None:
        self._host = host
        self.router: Any = None                     # node.router.ChunkRouter of the run
        self._carry: Counter = Counter()            # counters of stopped routers (cumulative per node process)
        self._remote: dict[str, Any] = {}           # agent -> RemoteRoute given to the router
        self._paths: dict[int, str] = {}            # target node -> path (from PathChanged)
        self._warned: set[tuple[int, str]] = set()
        self._links: dict[str, _DeliveryLink] = {}  # agent -> the node's chunk connection to its local learner

    @property
    def delivery(self) -> Any:
        return None if self.router is None else self.router.delivery

    # ---- lifecycle ----------------------------------------------------------------------------------------------

    def start(self, run: Any) -> None:
        """The run's router: direct channels through the run's data-port link (a TLS channel names its target node),
        no relay until T3.2 (b)."""
        from functools import partial

        from colosseum.net.frame import payload_byte_cap
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
        from colosseum.node.router import ChunkRouter

        link = self._host.data_link()
        make_channel = None if link is None else link.make_channel
        metadata = () if link is None else tuple(link.metadata)
        serial, post = run.serial, self._host.post
        dist = run.config.distributed
        self.router = ChunkRouter(
            make_frame=lambda agent, gen, frame, target: pb.ChunkFrame(agent_id=agent, target_node=int(target),
                                                                       generation=int(gen), frame=frame),
            make_channel=make_channel,
            direct_stub=lambda channel: partial(pb_grpc.NodeDataStub(channel).Chunks, metadata=metadata),
            relay_call=None,                             # the hub relay: T3.2 (b)
            connect_timeout_sec=float(dist.connect_timeout_sec),
            on_path=lambda target, path: post(PathChanged(serial, int(target), str(path))),
            max_frame_bytes=payload_byte_cap(int(dist.max_message_mb)))
        self._remote, self._paths = {}, {}

    def stop(self) -> None:
        """Stop the run's router (idempotent); its counters are carried."""
        router, self.router = self.router, None
        if router is None:
            return
        router.stop()
        self._carry.update(router.counters())
        self._host.set_delivery(None)
        self._links.clear()
        self._remote, self._paths = {}, {}

    def add_worker(self, conn: Any, worker: int) -> bool:
        """A worker's ``route`` connection; False (the caller closes it) when no router runs."""
        if self.router is None:
            return False
        self.router.add_worker(conn, worker=worker)
        return True

    # ---- views --------------------------------------------------------------------------------------------------

    def paths(self) -> dict[int, str]:
        return {} if self.router is None else self.router.paths()

    def heartbeat_paths(self) -> list[pb.PathState]:
        return [pb.PathState(node=int(t), path=p) for t, p in sorted(self.paths().items())]

    def counters(self) -> dict[str, int]:
        """The router counters of this node process (A2 names)."""
        out = Counter(self._carry)
        if self.router is not None:
            out.update(self.router.counters())
        return {k: int(v) for k, v in out.items()}

    # ---- routes -------------------------------------------------------------------------------------------------

    def update_routes(self, run: Any) -> None:
        """The router's remote routes and the workers' ``LearnerRoute`` of every agent whose learner runs on another
        node (from the newest version this node knows of)."""
        from colosseum.net.local import LearnerRoute
        from colosseum.node.router import RemoteRoute

        planner = self._host.planner()
        if self.router is None or planner is None or self._host.stopping:
            return
        local = set(self._host.local_learners())
        remote = {}
        for agent in run.agent_ids:
            info = planner.known(agent)
            if agent in local or info is None or int(info.node) in (self._host.number, -1):
                continue
            remote[agent] = RemoteRoute(int(info.node), int(info.generation), tuple(info.addresses),
                                        bool(info.relay_only))
        if remote != self._remote:
            self.router.set_routes(remote)
            self._remote = remote
        for agent in run.agent_ids:
            if agent in local:
                continue
            route = remote.get(agent)
            if route is None:
                want = LearnerRoute(agent, None, False, 0)
            else:                                    # via the node unless the path is known to be down
                want = LearnerRoute(agent, None, self._paths.get(route.node) != "down", route.generation)
            if (self._host.worker_route(agent) or LearnerRoute(agent, None, False, 0)) != want:
                self._host.set_route(want)

    def on_path_changed(self, ev: PathChanged) -> None:
        """A sender's path changed: the workers' routes follow; ``relay`` / ``down`` is one WARNING per target per node
        process and an ``Event(warning)`` to the hub (spec 4.8)."""
        run = self._host.current_run()
        if run is None or run.serial != ev.serial or self.router is None:
            return
        self._paths[ev.target] = ev.path
        route = self.router.routes.targets().get(ev.target)
        agents = sorted(a for a, r in self._remote.items() if r.node == ev.target)
        if ev.path in ("relay", "down") and (ev.target, ev.path) not in self._warned:
            self._warned.add((ev.target, ev.path))
            where = ", ".join(route.addresses) if route is not None and route.addresses else "no announced address"
            timeout = float(run.config.distributed.connect_timeout_sec)
            text = f"the data port of node {ev.target} is unreachable ({where}) within {timeout:g} s; "
            text += ("chunks go through the hub" if ev.path == "relay"
                     else f"the relay is off: this node stops collecting for {agents}")
            logger.warning(f"node: {text}")
            self._host.event("warning", proc="node", message=text)
        elif ev.path == "direct":
            logger.info(f"node: chunks for {agents} go directly to the data port of node {ev.target}")
        self.update_routes(run)

    # ---- delivery to local learners -----------------------------------------------------------------------------

    def sync_delivery(self, run: Any, now: float) -> None:
        """The node's chunk connection to every local learner that takes chunks (frames from other machines); a broken
        or missing one is (re)opened off the loop at most every ``DELIVERY_RETRY_SEC``."""
        router = self.router
        if router is None or self._host.stopping:
            return
        want = self._host.routed_learners()
        for agent in [a for a, link in self._links.items() if want.get(a) != link.listener]:
            del self._links[agent]
            router.delivery.set_learner(agent, None)
        for agent, listener in want.items():
            link = self._links.setdefault(agent, _DeliveryLink(listener))
            if link.opening or now < link.next_try or router.delivery.connected(agent):
                continue
            link.opening, link.next_try = True, now + DELIVERY_RETRY_SEC
            threading.Thread(target=self._open_learner_conn, args=(run.serial, agent, listener, run.authkey),
                             name=f"node-chunks-{agent}", daemon=True).start()

    def _open_learner_conn(self, serial: int, agent: str, listener: str, authkey: bytes) -> None:
        """A short thread: the node's ``chunks`` connection to a local learner's listener."""
        from colosseum.net.local import ChildHello, connect

        try:
            conn = connect(listener, authkey, timeout=LEARNER_CONNECT_SEC)
            conn.send(ChildHello("node", "node", 0, os.getpid(), "chunks"))
        except Exception as e:  # noqa: BLE001 - retried by sync_delivery
            self._host.post(LearnerConnOpened(serial, agent, listener, None, f"{type(e).__name__}: {e}"))
            return
        self._host.post(LearnerConnOpened(serial, agent, listener, conn, ""))

    def on_learner_conn(self, ev: LearnerConnOpened) -> None:
        run, link = self._host.current_run(), self._links.get(ev.agent_id)
        current = (run is not None and run.serial == ev.serial and self.router is not None
                   and not self._host.stopping and link is not None and link.listener == ev.listener)
        if current:
            link.opening = False
        if ev.conn is None:
            if current:
                logger.debug(f"node: no chunk connection to learner {ev.agent_id} yet ({ev.error})")
            return
        if not current:
            ev.conn.close()
            return
        self.router.delivery.set_learner(ev.agent_id, ev.conn)

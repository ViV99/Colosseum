"""The node's routing section (SP5 spec 4.7, 4.8; T3.2): chunks between machines, owned by the node agent's main loop
(``colosseum.node.agent``, item 12).

Each run has a ``node.router.ChunkRouter`` (``start``), created before the data port (a new data port takes the router's
``LocalDelivery`` at creation, the process's existing one through ``DataServer.set_delivery``; the router is per run,
the port per process). Every pass at the fetch cadence (``update_routes``): every trainable agent whose newest known
version comes from another node (not parked: node -1 is no route, pre-flight ruling P-PF3-13) and whose learner does not
run here is a ``RemoteRoute`` (its node, generation, data addresses); the router starts a ``TargetSender`` per target
node at once, which probes the path (``direct`` within ``distributed.connect_timeout_sec``, else the hub relay
(``Hub.RelaySend``), else ``down`` with ``distributed.relay: false``) and reports it (``PathChanged`` into the inbox;
``Heartbeat.paths``), so the hub grants collecting seats only where the learner is reachable (P-PF3-1). Workers get
``LearnerRoute(agent, None, via_node=True)`` for such agents while the path is not ``down`` (``via_node=False``
otherwise: the worker drops and counts ``no_learner``): their frames come on the workers' ``route`` connections
(``add_worker``: one ``RouteReader`` each; nothing is decoded) and leave in one gRPC stream per target. A path change to
``relay`` / ``down`` is one WARNING per target per node process and one ``Event(warning)`` to the hub.

Frames from other machines arrive at the data port's ``Chunks`` or through the hub relay and go to the local learner
over the node's own ``chunks`` connection to that learner incarnation (``sync_delivery``: opened by a short thread off
the loop, ``LearnerConnOpened``; reopened at most every ``DELIVERY_RETRY_SEC`` after a failure).

The hub relay (spec 4.8): while this node runs a learner and ``distributed.relay`` is on, a ``router.RelayReceiver``
keeps the node's ``Hub.RelayInbox`` stream open (``sync_relay_inbox``): relayed frames go to the same delivery, and a
weights pull (the hub needs the newest weights of a learner here for nodes that cannot reach this data port) becomes
``RelayPull`` on the main loop, which uploads the cache's newest frame of that agent with ``Hub.Upload(kind=
"relay_latest")`` in the transfer pool (``on_relay_pull``; at most one upload in flight per agent: a pull meanwhile is
ignored, the running upload sends the newest anyway).

The hub away (spec 4.5, SP5 T4.4, pre-flight ruling P18): past ``distributed.hub_lost_collect_sec`` without the hub,
``cut_relay`` takes every agent whose learner this node reaches only through the hub relay out of the router's routes
and gives the workers ``LearnerRoute(agent, None, via_node=False)`` for it (their chunks for it are dropped and counted
``no_learner``; the matches are still played); ``restore_relay`` (the next ``Welcome``) brings the routes back.

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

__all__ = ["DELIVERY_RETRY_SEC", "LEARNER_CONNECT_SEC", "RELAY_UPLOAD_TIMEOUT_SEC", "LearnerConnOpened", "NodeRouting",
           "PathChanged", "RelayPull", "RelayUploadDone", "RoutingHost"]

DELIVERY_RETRY_SEC = 1.0        # the node's chunk connection to a local learner is (re)opened at most this often
LEARNER_CONNECT_SEC = 5.0
RELAY_UPLOAD_TIMEOUT_SEC = 300.0


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
class RelayPull:
    """The hub asked (``RelayItem.pull``) for the newest weights of a learner here (posted by the relay inbox)."""

    serial: int
    request: Any                       # pb.LatestRequest


@dataclass
class RelayUploadDone:
    serial: int
    agent_id: str
    reply: Any                         # pb.UploadReply, None: nothing to send or a transfer error
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
    @property
    def node_id(self) -> str: ...
    def hub_channel(self) -> Any: ...                 # grpc.Channel to the hub (None before the session)
    def hub_metadata(self) -> tuple: ...              # the node's R1 credentials
    def submit(self, fn: Any, *args: Any) -> Any: ... # the transfer pool
    @property
    def remote_nodes(self) -> bool: ...               # other machines' nodes may exist (not train w/o --listen)


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
        self._receiver: Any = None                  # router.RelayReceiver (relay inbox) of the run
        self._relay = False                         # distributed.relay of the run
        self._uploading: set[str] = set()           # agents with a relay upload in flight
        self._cut: set[str] = set()                 # agents cut off by the hub-lost collect rule (cut_relay)

    @property
    def delivery(self) -> Any:
        return None if self.router is None else self.router.delivery

    # ---- lifecycle ----------------------------------------------------------------------------------------------

    def start(self, run: Any) -> None:
        """The run's router: direct channels through the run's data-port link (a TLS channel names its target node),
        the hub relay as the fallback unless ``distributed.relay: false``."""
        from functools import partial

        from colosseum.net.frame import payload_byte_cap
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
        from colosseum.node.router import ChunkRouter

        link = self._host.data_link()
        make_channel = None if link is None else link.make_channel
        metadata = () if link is None else tuple(link.metadata)
        serial, post = run.serial, self._host.post
        dist = run.config.distributed
        self._relay = bool(dist.relay)
        host, hub_md = self._host, tuple(self._host.hub_metadata())
        relay_call = None
        if self._relay and host.hub_channel() is not None:
            def relay_call() -> Any:                     # the node's current hub channel (replaced after an outage)
                return partial(pb_grpc.HubStub(host.hub_channel()).RelaySend, metadata=hub_md)
        self.router = ChunkRouter(
            make_frame=lambda agent, gen, frame, target: pb.ChunkFrame(agent_id=agent, target_node=int(target),
                                                                       generation=int(gen), frame=frame),
            make_channel=make_channel,
            direct_stub=lambda channel: partial(pb_grpc.NodeDataStub(channel).Chunks, metadata=metadata),
            relay_call=relay_call,                       # None: distributed.relay is false
            connect_timeout_sec=float(dist.connect_timeout_sec),
            on_path=lambda target, path: post(PathChanged(serial, int(target), str(path))),
            max_frame_bytes=payload_byte_cap(int(dist.max_message_mb)))
        self._remote, self._paths = {}, {}

    def stop(self) -> None:
        """Stop the run's router (idempotent); its counters are carried."""
        router, self.router = self.router, None
        if router is None:
            return
        self._stop_receiver()
        router.stop()
        self._carry.update(router.counters())
        self._host.set_delivery(None)
        self._links.clear()
        self._remote, self._paths = {}, {}
        self._cut.clear()

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
        remote, cut = {}, {}
        for agent in run.agent_ids:
            info = planner.known(agent)
            if agent in local or info is None or int(info.node) in (self._host.number, -1):
                continue
            route = RemoteRoute(int(info.node), int(info.generation), tuple(info.addresses), bool(info.relay_only))
            if agent in self._cut:
                cut[agent] = route                   # the hub-lost collect rule: no route, chunks not collected
            else:
                remote[agent] = route
        if remote != self._remote:
            self.router.set_routes(remote)
            self._remote = remote
        for agent in run.agent_ids:
            if agent in local:
                continue
            route = remote.get(agent)
            if agent in cut:
                want = LearnerRoute(agent, None, False, cut[agent].generation)
            elif route is None:
                want = LearnerRoute(agent, None, False, 0)
            else:                                    # via the node unless the path is known to be down
                want = LearnerRoute(agent, None, self._paths.get(route.node) != "down", route.generation)
            if (self._host.worker_route(agent) or LearnerRoute(agent, None, False, 0)) != want:
                self._host.set_route(want)

    def through_relay(self) -> list[str]:
        """The agents whose current route goes through the hub relay: their target's path is ``relay``, or it is not
        known to be ``direct`` and the learner announced no reachable data address (``relay_only`` / none)."""
        out = []
        for agent, route in self._remote.items():
            path = self._paths.get(route.node)
            if path == "relay" or (path != "direct" and (route.relay_only or not route.addresses)):
                out.append(agent)
        return sorted(out)

    def cut_relay(self, run: Any) -> list[str]:
        """``distributed.hub_lost_collect_sec`` passed without the hub (module docstring): the agents reachable only
        through the relay are cut off; returns the newly cut ones."""
        agents = [a for a in self.through_relay() if a not in self._cut]
        if agents:
            self._cut.update(agents)
            self.update_routes(run)
        return agents

    def restore_relay(self, run: Any) -> None:
        """The hub is back (``Welcome``): the agents cut off by ``cut_relay`` get their routes again."""
        if self._cut:
            self._cut.clear()
            self.update_routes(run)

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
                     else f"the hub relay is off (distributed.relay: false): this node plays {agents} only from "
                          f"their stored snapshots and sends their learners no data. Open node {ev.target}'s data "
                          f"port to this machine (the direct path) or set distributed.relay: true")
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

    # ---- the hub relay: inbox and weight pulls ---------------------------------------------------------------------

    def sync_relay_inbox(self, run: Any) -> None:
        """The relay inbox runs while this node runs a learner, the relay is on and the hub accepts nodes of other
        machines (W03-7; not for ``train`` without ``--listen``, whose local node is the only node)."""
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
        from colosseum.node.router import RelayReceiver

        want = (self.router is not None and self._relay and self._host.remote_nodes and not self._host.stopping
                and bool(self._host.local_learners()) and self._host.hub_channel() is not None)
        if not want:
            self._stop_receiver()
            return
        if self._receiver is not None:
            return
        host, md, serial, post = self._host, tuple(self._host.hub_metadata()), run.serial, self._host.post
        self._receiver = RelayReceiver(
            node_id=self._host.node_id,
            open_inbox=lambda requests: pb_grpc.HubStub(host.hub_channel()).RelayInbox(requests, metadata=md),
            make_credit=lambda n, node_id: pb.RelayCredit(frames=int(n), node_id=node_id),
            delivery=self.router.delivery,
            chunk_of=lambda item: ((item.chunk.agent_id, item.chunk.frame) if item.WhichOneof("i") == "chunk"
                                   else None),
            on_pull=lambda item: post(RelayPull(serial, item.pull)),
            counters=self.router.counters_obj)
        self._receiver.start()

    def _stop_receiver(self) -> None:
        receiver, self._receiver = self._receiver, None
        if receiver is not None:
            receiver.stop()

    def relay_connected(self) -> bool:
        return self._receiver is not None and self._receiver.connected.is_set()

    def on_relay_pull(self, ev: RelayPull) -> None:
        """The hub wants the newest weights of a learner here: one upload in the transfer pool (at most one in flight
        per agent)."""
        run = self._host.current_run()
        agent = str(ev.request.agent_id)
        if (run is None or run.serial != ev.serial or self._host.stopping or agent in self._uploading
                or agent not in self._host.local_learners()):
            return
        self._uploading.add(agent)
        future = self._host.submit(self._upload_latest, run, agent, str(ev.request.request_id))
        post, serial = self._host.post, run.serial

        def done(f: Any) -> None:
            if f.cancelled():
                post(RelayUploadDone(serial, agent, None, "cancelled"))
            elif f.exception() is not None:
                e = f.exception()
                post(RelayUploadDone(serial, agent, None, f"{type(e).__name__}: {e}"))
            else:
                post(RelayUploadDone(serial, agent, f.result(), ""))

        future.add_done_callback(done)

    def _upload_latest(self, run: Any, agent: str, request_id: str) -> Any:
        """A transfer-pool thread: the cache's newest frame of ``agent`` as ``Upload(kind="relay_latest")`` (parts
        straight from the shared-memory segment). None when the cache holds none."""
        import mmap

        from colosseum.net.grpc_util import send_parts
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

        opened = run.reader.open_latest(agent)
        if opened is None:
            return None
        f, pointer = opened
        with f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            header = pb.UploadHeader(kind="relay_latest", agent_id=agent, generation=int(pointer["generation"]),
                                     version=int(pointer["version"]), total_size=len(mapping),
                                     digest=str(pointer["digest"]), request_id=request_id)

            def parts() -> Any:
                yield pb.UploadPart(header=header)
                for part in send_parts(mapping, digest=str(pointer["digest"])):
                    if part.data:
                        yield pb.UploadPart(data=part.data)

            return pb_grpc.HubStub(self._host.hub_channel()).Upload(
                parts(), metadata=tuple(self._host.hub_metadata()), timeout=RELAY_UPLOAD_TIMEOUT_SEC)

    def on_relay_upload_done(self, ev: RelayUploadDone) -> None:
        self._uploading.discard(ev.agent_id)
        if ev.error:
            logger.info(f"node: the relayed weights of {ev.agent_id} were not uploaded ({ev.error}); the hub asks "
                        f"again")
        elif ev.reply is not None and not ev.reply.accepted:
            level = logging.DEBUG if ev.reply.reason == "not_newer" else logging.INFO
            logger.log(level, f"node: the hub did not take the relayed weights of {ev.agent_id} ({ev.reply.reason})")

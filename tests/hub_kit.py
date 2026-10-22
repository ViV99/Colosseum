"""Test support for the hub's gRPC server (SP5 T2.3, ruling R13: hub logic tested with fake nodes against the real
server).

- ``serve(tmp_path, ...)``: a real ``hub.server`` on ``127.0.0.1:0`` (TLS unless ``insecure``) with a real
  ``HubIdentity`` under ``tmp_path`` and a ``MiniHubLoop`` as its main loop; ``ServedHub.stop()`` in a ``finally``.
- ``FakeNode``: an in-process client of the real server (register, session streams, uploads, downloads).
- ``MiniHubLoop``: a minimal main loop over ``colosseum.hub.state`` (it follows pre-flight ruling P-PF2-1: streams
  are told apart by their ``SessionHandle`` serial). Test support only: T2.5's ``Hub`` implements the same handlers
  on the real coordinator.
- ``make_result``: a 1v1 training ``MatchResult`` of the toy game ``turns`` as a node sends it.
- T2.5: ``lineup_hub`` (a real ``Hub`` after its setup steps), ``open_node`` / ``RecordingHandle``, ``accept_version``,
  ``request_cells``, ``result_batch``, ``hub_results``, ``node_message``: the real hub's handlers driven in-process.
"""
from __future__ import annotations

import itertools
import json
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from colosseum.core.run_dir import RunDir
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.server import (
    AdminIn,
    BlobStore,
    NodeMessageIn,
    RegisterIn,
    SessionClosed,
    SessionHandle,
    SessionOpened,
    UploadIn,
    build_server,
    hello_from_proto,
)
from colosseum.hub.state import CellKey, CellTable, EpochAllocator, NodeRegistry, ResultIntake, VersionRegistry
from colosseum.net.grpc_util import channel_options, node_metadata, receive_parts, send_parts, token_metadata
from colosseum.net.messages import PROTOCOL_VERSION, lineup_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
from colosseum.net.security import HUB_SERVER_NAME, HubIdentity, Token, hub_channel_credentials

MAX_MB = 4                     # the served hub's max_message_mb (upload cap = payload_byte_cap(4) = 32 MiB)
CLIENT_MAX_MB = 64
_END = object()


def poll_until(predicate: Callable[[], Any], timeout: float = 10.0, interval: float = 0.02) -> Any:
    """Poll ``predicate`` until it returns a truthy value (returned) or ``timeout`` passes (AssertionError)."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(interval)


def make_result(node: int = 1, worker: int = 0, env: int = 0, epoch: int = 1, k: int = 0, *, run_id: str = "R",
                agents: Sequence[str] = ("a", "random")):
    """A 1v1 ``MatchResult`` of ``turns`` (layout ``2p``, role ``player``): ``agents[0]``'s latest weights (generation
    1, versions 4..5) against ``agents[1]`` fixed, uid ``<run_id>/train/n<node>/w<W>_e<E>_g<epoch>_ep<K>``."""
    from game_helpers import make_match_result

    match_id = f"w{worker}_e{env}_g{epoch}_ep{k}"
    seats = [(0, agents[0], "latest", "player", 4, 5), (1, agents[1], "fixed", "player")]
    result = make_match_result(seats, uid=f"{run_id}/train/n{node}/{match_id}", match_id=match_id)
    result.node = node
    for seat in result.seats:
        seat.generation = 1 if seat.network_id == "latest" else None
    return result


# ---------------------------------------------------------------------------------------------------------------
# FakeNode
# ---------------------------------------------------------------------------------------------------------------


class FakeNode:
    """In-process client of the real hub server. ``token``: the hub's ``Token`` (for ``register``); ``ca_pem``: the
    pinned hub CA (TLS); ``insecure``: a plain channel. After ``register`` (or with ``node_id`` / ``node_secret`` set
    by the test) every other call carries the node's credentials."""

    def __init__(self, address: str, *, token: Token, ca_pem: bytes | None = None, insecure: bool = False) -> None:
        self.address, self.token, self.ca_pem, self.insecure = address, token, ca_pem, insecure
        self.node_id: str | None = None
        self.node_secret: str | None = None
        self.number: int | None = None
        self._key = ec.generate_private_key(ec.SECP256R1())
        self._channel: grpc.Channel | None = None
        self._requests: queue.Queue | None = None
        self._call = None
        self._reader: threading.Thread | None = None
        self.received: list[pb.HubMessage] = []
        self.ended = threading.Event()
        self.code: grpc.StatusCode | None = None
        self.details: str = ""
        self.env_steps = 0                       # the cumulative env steps its heartbeats report
        self._beating = threading.Event()        # heartbeats are sent while set (T4.1: pause / resume)
        self._beat_stop: threading.Event | None = None

    # ---- channel ------------------------------------------------------------------------------------------------

    def channel(self) -> grpc.Channel:
        if self._channel is None:
            if self.insecure:
                self._channel = grpc.insecure_channel(self.address, options=channel_options(CLIENT_MAX_MB))
            else:
                self._channel = grpc.secure_channel(self.address, hub_channel_credentials(self.ca_pem),
                                                    options=channel_options(CLIENT_MAX_MB,
                                                                            server_name=HUB_SERVER_NAME))
        return self._channel

    def stub(self) -> pb_grpc.HubStub:
        return pb_grpc.HubStub(self.channel())

    def _node_md(self) -> tuple:
        return node_metadata(self.node_id or "", self.node_secret or "")

    def public_key_pem(self) -> str:
        return self._key.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo).decode("ascii")

    # ---- calls --------------------------------------------------------------------------------------------------

    def register(self, machine_id: str = "m", name: str = "fake", *, token: Token | None | bool = True
                 ) -> pb.RegisterReply:
        """``Register`` with the token (``token=None``: without one; another ``Token``: that one); keeps the
        identity from the reply."""
        md = () if token is None else token_metadata(self.token if token is True else token)
        reply = self.stub().Register(pb.RegisterRequest(public_key_pem=self.public_key_pem(), machine_id=machine_id,
                                                        name=name, node_id=self.node_id or ""),
                                     metadata=md, timeout=15)
        self.node_id, self.node_secret, self.number = reply.node_id, reply.node_secret, reply.number
        return reply

    def hello(self, session_id: str, *, live_cells: Sequence[tuple[int, int, int, int]] = (),
              last_acked: int = 0, fields: dict | None = None) -> pb.Hello:
        """``fields``: more ``pb.Hello`` fields (T4.1: ``slots``, ``hardware``, ``learners``, ``data_addresses``,
        ``name``, ``tags``)."""
        kw: dict[str, Any] = dict(node_id=self.node_id, session_id=session_id, protocol_major=PROTOCOL_VERSION[0],
                                  protocol_minor=PROTOCOL_VERSION[1], env_steps=self.env_steps,
                                  last_acked_seq=last_acked,
                                  live_cells=[pb.LiveCell(worker=w, env=e, epoch=ep, incarnation=i)
                                              for w, e, ep, i in live_cells])
        kw.update(fields or {})
        return pb.Hello(**kw)

    def open(self, session_id: str, *, live_cells: Sequence[tuple[int, int, int, int]] = (), last_acked: int = 0,
             first: pb.NodeMessage | None = None, read: bool = True, hello_fields: dict | None = None,
             heartbeat_sec: float | None = None) -> FakeNode:
        """Open a session stream whose requests come from a queue: ``Hello`` first (``first``: another first
        message instead; ``hello_fields``: more ``Hello`` fields); responses are collected into ``received`` by a
        thread (``read=False``: never read). ``heartbeat_sec``: a thread sends a ``Heartbeat`` (``env_steps``) at that
        cadence until the stream ends (``pause_heartbeats`` / ``resume_heartbeats``)."""
        if self._beat_stop is not None:
            self._beat_stop.set()
        self._requests = requests = queue.Queue()
        requests.put(first if first is not None else pb.NodeMessage(
            hello=self.hello(session_id, live_cells=live_cells, last_acked=last_acked, fields=hello_fields)))
        self.received, self.ended, self.code, self.details = [], threading.Event(), None, ""
        if heartbeat_sec is not None:
            stop = self._beat_stop = threading.Event()
            self._beating.set()

            def beat() -> None:
                while not stop.wait(heartbeat_sec):
                    if self._beating.is_set():
                        requests.put(pb.NodeMessage(heartbeat=pb.Heartbeat(env_steps=self.env_steps)))

            threading.Thread(target=beat, name="fake-node-heartbeat", daemon=True).start()

        def gen():
            while True:
                item = requests.get()
                if item is _END:
                    return
                yield item

        call = self.stub().Session(gen(), metadata=self._node_md())
        self._call = call
        if read:
            ended = self.ended

            def reader():
                try:
                    for msg in call:
                        self.received.append(msg)
                    self.code = grpc.StatusCode.OK
                except grpc.RpcError as e:
                    self.code = e.code() if hasattr(e, "code") else None
                    self.details = e.details() if hasattr(e, "details") else str(e)
                finally:
                    requests.put(_END)
                    ended.set()

            self._reader = threading.Thread(target=reader, name="fake-node-reader", daemon=True)
            self._reader.start()
        return self

    def send(self, **oneof) -> None:
        """Send one ``NodeMessage`` (``send(heartbeat=pb.Heartbeat(env_steps=10))``)."""
        self._requests.put(pb.NodeMessage(**oneof))

    def pause_heartbeats(self) -> None:
        self._beating.clear()

    def resume_heartbeats(self) -> None:
        self._beating.set()

    @staticmethod
    def wait_until(predicate: Callable[[], Any], timeout: float = 10.0) -> Any:
        """``poll_until(predicate, timeout)`` (A4's name)."""
        return poll_until(predicate, timeout)

    def cut(self) -> None:
        """Close the channel abruptly (the hub sees a cut stream)."""
        if self._beat_stop is not None:
            self._beat_stop.set()
        if self._requests is not None:
            self._requests.put(_END)
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def close(self) -> None:
        """End the request stream (half-close), wait briefly for the hub to end it, close the channel."""
        if self._beat_stop is not None:
            self._beat_stop.set()
        if self._requests is not None:
            self._requests.put(_END)
        self.ended.wait(5)
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def messages(self, kind: str) -> list:
        return [getattr(m, kind) for m in list(self.received) if m.WhichOneof("m") == kind]

    def wait_for(self, kind: str, n: int = 1, timeout: float = 10.0) -> list:
        """The first ``n`` received ``HubMessage`` payloads of oneof ``kind`` (waits up to ``timeout``)."""
        return poll_until(lambda: (lambda ms: ms[:n] if len(ms) >= n else None)(self.messages(kind)), timeout)

    def wait_ended(self, timeout: float = 10.0) -> grpc.StatusCode | None:
        if not self.ended.wait(timeout):
            raise AssertionError(f"the session stream did not end within {timeout} s")
        return self.code

    def download(self, kind: str, agent: str, ident: str, *, cap: int = 1 << 30,
                 since_version: int = 0) -> tuple[bytes | None, str]:
        parts = self.stub().Download(pb.BlobRequest(kind=kind, agent_id=agent, ident=ident,
                                                    since_version=since_version),
                                     metadata=self._node_md(), timeout=30)
        return receive_parts(parts, cap=cap)

    def upload(self, header: pb.UploadHeader | None, data: bytes) -> pb.UploadReply:
        """``Upload`` of ``data`` in parts after ``header`` (None: data parts only)."""
        def parts():
            if header is not None:
                yield pb.UploadPart(header=header)
            for part in send_parts(data):
                if part.data:
                    yield pb.UploadPart(data=part.data)

        return self.stub().Upload(parts(), metadata=self._node_md(), timeout=30)


# ---------------------------------------------------------------------------------------------------------------
# MiniHubLoop
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Call:
    fn: Callable[[], Any]
    reply: Future = field(default_factory=Future)


AGENTS = ("a", "random")
GRANT_LINEUP = Lineup(layout="2p", seats=[SeatAssignment("a", "latest", True), SeatAssignment("random", "fixed",
                                                                                                  False)])


def _toy_spec():
    from game_helpers import TurnTakingGame

    return TurnTakingGame().spec


SPEC = _toy_spec()


class MiniHubLoop:
    """A thread over the server inbox with ``hub.state`` objects (module docstring). ``received``: every inbox event
    in arrival order; ``connects``: ``NodeRegistry.connect`` kinds; ``accounted``: result uids accounted (at intake,
    no journal) after ``ResultIntake.check`` with the node's number and uid prefix; ``dropped``: ``(uid, reason)`` of
    results that failed it (an undecodable batch is acked and nothing of it accounted); ``acked``: seqs acked in
    order; ``uploads``: ``(header, bytes, path)`` of handled uploads.
    ``upload_reply(header, data) -> pb.UploadReply`` decides uploads (default: accepted, ``ckpt_v<version>``;
    ``answer_uploads=False``: uploads are recorded in ``received`` and never answered)."""

    def __init__(self, server_inbox: queue.SimpleQueue, *, identity: HubIdentity, node_lost_after_sec: float = 10.0,
                 hold_acks: bool = False, run_id: str = "R", ttl_sec: float = 60.0,
                 upload_reply: Callable[[pb.UploadHeader, bytes], pb.UploadReply] | None = None,
                 answer_uploads: bool = True) -> None:
        self.inbox = server_inbox
        self.identity = identity
        self.node_lost_after_sec = node_lost_after_sec
        self.hold_acks = hold_acks
        self.run_id = run_id
        self.ttl_sec = ttl_sec
        self.upload_reply = upload_reply or (lambda h, d: pb.UploadReply(accepted=True,
                                                                         checkpoint_id=f"ckpt_v{h.version}"))
        self.answer_uploads = answer_uploads
        self.registry = NodeRegistry()
        self.cells = CellTable(EpochAllocator(0, lambda until: None))
        self.intake = ResultIntake(journal=False)
        self.versions = VersionRegistry()
        self.versions.new_generation("a")                          # every learner is generation 1 (T2.3 - T2.5)
        self.sessions: dict[str, SessionHandle] = {}
        self.received: list[object] = []
        self.connects: list[str] = []
        self.accounted: list[str] = []
        self.dropped: list[tuple[str, str]] = []
        self.acked: list[int] = []
        self.uploads: list[tuple[pb.UploadHeader, bytes, Path]] = []
        self.errors: list[str] = []
        self._held: list[tuple[str, str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mini-hub-loop", daemon=True)
        self._thread.start()

    # ---- test API (any thread) ----------------------------------------------------------------------------------

    def call(self, fn: Callable[[], Any], timeout: float = 10.0) -> Any:
        """Run ``fn`` in the loop thread and return its result."""
        c = _Call(fn)
        self.inbox.put(c)
        return c.reply.result(timeout)

    def events(self, cls: type) -> list:
        return [e for e in list(self.received) if isinstance(e, cls)]

    def messages(self, kind: str) -> list[NodeMessageIn]:
        return [e for e in self.events(NodeMessageIn) if e.message.WhichOneof("m") == kind]

    def wait_events(self, cls: type, n: int = 1, timeout: float = 10.0) -> list:
        return poll_until(lambda: (lambda es: es[:n] if len(es) >= n else None)(self.events(cls)), timeout)

    def handle(self, node_id: str) -> SessionHandle | None:
        return self.call(lambda: self.sessions.get(node_id))

    def close_session(self, node_id: str, reason: str) -> None:
        def close():
            handle = self.sessions.get(node_id)
            if handle is not None:
                handle.close(reason)
        self.call(close)

    def flood(self, handle: SessionHandle, n: int, size: int = 1 << 20) -> None:
        """``n`` messages of about ``size`` bytes on ``handle`` (P-PF2-16: large messages, so HTTP/2 flow control
        blocks the server's writer and the outbound queue fills)."""
        for _ in range(n):
            handle.send(pb.HubMessage(welcome=pb.Welcome(config_json="x" * size)))

    def release_acks(self) -> None:
        def release():
            held, self._held = self._held, []
            for node_id, session_id, seq in held:
                self._ack(node_id, session_id, seq)
        self.call(release)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(5)

    # ---- the loop -----------------------------------------------------------------------------------------------

    def _run(self) -> None:
        next_check = time.monotonic()
        while not self._stop.is_set():
            try:
                ev = self.inbox.get(timeout=0.05)
            except queue.Empty:
                ev = None
            if ev is not None:
                if isinstance(ev, _Call):
                    try:
                        ev.reply.set_result(ev.fn())
                    except Exception as e:  # noqa: BLE001 - handed to the caller
                        ev.reply.set_exception(e)
                else:
                    self.received.append(ev)
                    try:
                        self._dispatch(ev)
                    except Exception as e:  # noqa: BLE001 - recorded for the test
                        self.errors.append(f"{type(e).__name__}: {e}")
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 0.1
                for entry in self.registry.lost(now, self.node_lost_after_sec):
                    self.cells.drop_node(entry.number)
                    handle = self.sessions.pop(entry.node_id, None)
                    if handle is not None:
                        handle.close("lost")

    def _dispatch(self, ev: object) -> None:
        now = time.monotonic()
        if isinstance(ev, RegisterIn):
            self._on_register(ev)
        elif isinstance(ev, SessionOpened):
            self._on_opened(ev, now)
        elif isinstance(ev, SessionClosed):
            if self.sessions.get(ev.node_id) is ev.handle:        # P-PF2-1: only the current stream's close
                self.registry.disconnect(ev.node_id, ev.handle.serial)
                del self.sessions[ev.node_id]
        elif isinstance(ev, NodeMessageIn):
            handle = self.sessions.get(ev.node_id)
            if handle is None or handle.serial != ev.serial:      # a superseded stream's message
                return
            self._on_message(ev, handle, now)
        elif isinstance(ev, UploadIn):
            if not self.answer_uploads:                           # the servicer's reply times out
                return
            data = ev.path.read_bytes()
            self.uploads.append((ev.header, data, ev.path))
            ev.path.unlink(missing_ok=True)
            ev.reply.set_result(self.upload_reply(ev.header, data))
        elif isinstance(ev, AdminIn):
            reply = pb.AdminReply(code=0, text=f"ok {ev.command}", data_json=json.dumps(ev.args))
            if isinstance(ev.reply, Future):
                ev.reply.set_result(reply)
            else:
                ev.reply.put(pb.AdminReply(code=0, text="progress"))
                reply.done = True
                ev.reply.put(reply)

    def _on_register(self, ev: RegisterIn) -> None:
        req = ev.request
        try:
            record, secret = self.identity.nodes.register(machine_id=req.machine_id, name=req.name,
                                                          tags=list(req.tags), node_id=req.node_id or None)
            cert = self.identity.issue_node_certificate(record.node_id, req.public_key_pem.encode("ascii"))
        except Exception as e:  # noqa: BLE001 - the servicer maps it to a status
            ev.reply.set_exception(e)
            return
        ev.reply.set_result(pb.RegisterReply(node_id=record.node_id, number=record.number, name=record.name,
                                             certificate_pem=cert.decode("ascii"), node_secret=secret,
                                             ca_pem=self.identity.ca_pem.decode("ascii")))

    def _on_opened(self, ev: SessionOpened, now: float) -> None:
        handle, hello = ev.handle, ev.hello
        record = self.identity.nodes.get(handle.node_id)
        if record is None:
            handle.close("unknown node")
            return
        old = self.sessions.get(handle.node_id)
        if old is not None and old is not handle:
            old.close("superseded")                               # P-PF2-1 (c)
        self.sessions[handle.node_id] = handle
        kind = self.registry.connect(handle.node_id, record.number, record.name, hello.session_id,
                                     hello_from_proto(hello), now, serial=handle.serial)
        self.connects.append(kind)
        if kind == "returned":
            self.cells.drop_node(record.number)
        self.intake.open(handle.node_id, hello.session_id, int(hello.last_acked_seq))
        handle.send(pb.HubMessage(welcome=pb.Welcome(node_number=record.number, run_id=self.run_id,
                                                     config_json="{}")))

    def _on_message(self, ev: NodeMessageIn, handle: SessionHandle, now: float) -> None:
        kind = ev.message.WhichOneof("m")
        entry = self.registry.by_id(ev.node_id)
        if kind == "heartbeat":
            self.registry.heartbeat(ev.node_id, ev.session_id, now)
        elif kind == "lineup_request":
            cells = [(CellKey(entry.number, c.worker, c.env), int(c.epoch), int(c.incarnation))
                     for c in ev.message.lineup_request.cells]
            grants = []
            for cell in self.cells.request(entry.number, cells, now):
                self.cells.grant(cell.key, "a", self.ttl_sec, now)
                lineup = Lineup(layout=GRANT_LINEUP.layout, seats=list(GRANT_LINEUP.seats), epoch=cell.epoch)
                grants.append(pb.Grant(worker=cell.key.worker, env=cell.key.env, epoch=cell.epoch,
                                       ttl_sec=self.ttl_sec, lineup=lineup_to_proto(lineup)))
            handle.send(pb.HubMessage(grants=pb.LineupGrants(grants=grants)))
        elif kind == "results":
            seq = int(ev.message.results.seq)
            if not self.intake.begin(ev.node_id, ev.session_id, seq):
                if self.intake.is_acked(ev.node_id, ev.session_id, seq):
                    self._send_ack(ev.node_id, seq)
                return
            for result in ev.results or ():                        # an undecodable batch: acked, nothing accounted
                reason = self.intake.check(result, spec=SPEC, agents=AGENTS, versions=self.versions,
                                           node=entry.number, uid_prefix=f"{self.run_id}/train/n{entry.number}/")
                if reason is None:
                    self.accounted.append(result.uid)
                else:
                    self.dropped.append((result.uid, reason))
            if self.hold_acks:
                self._held.append((ev.node_id, ev.session_id, seq))
            else:
                self._ack(ev.node_id, ev.session_id, seq)

    def _ack(self, node_id: str, session_id: str, seq: int) -> None:
        self.intake.committed(node_id, session_id, seq)
        self.acked.append(seq)
        self._send_ack(node_id, seq)

    def _send_ack(self, node_id: str, seq: int) -> None:
        handle = self.sessions.get(node_id)
        if handle is not None:
            handle.send(pb.HubMessage(ack=pb.Ack(seqs=[seq])))


# ---------------------------------------------------------------------------------------------------------------
# A served hub
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class ServedHub:
    server: grpc.Server
    port: int
    address: str
    identity: HubIdentity
    blobs: BlobStore
    run_dir: RunDir
    inbox: queue.SimpleQueue
    loop: MiniHubLoop
    insecure: bool

    def node(self) -> FakeNode:
        return FakeNode(self.address, token=self.identity.token, ca_pem=self.identity.ca_pem, insecure=self.insecure)

    def registered_node(self, **kw) -> FakeNode:
        node = self.node()
        node.register(**kw)
        return node

    def stop(self) -> None:
        self.server.stop(None)
        self.loop.stop()


def serve(tmp_path: Path, *, insecure: bool = False, max_message_mb: int = MAX_MB, identity: HubIdentity | None = None,
          relay: Any = None, latest: Any = None, **loop_kwargs) -> ServedHub:
    """A started hub server on ``127.0.0.1:0`` with a ``MiniHubLoop`` (``loop_kwargs`` go to it); the run dir is
    ``tmp_path / "run"`` and the hub identity ``tmp_path / "hub-identity"``; ``relay`` / ``latest``: the hub relay's
    ``RelayHub`` / ``LatestRelay`` (T3.2; the test publishes the online nodes and versions itself)."""
    identity = identity or HubIdentity.load_or_create(Path(tmp_path) / "hub-identity")
    run_dir = RunDir(Path(tmp_path) / "run")
    (run_dir.root / "checkpoints").mkdir(parents=True, exist_ok=True)
    blobs = BlobStore(run_dir)
    inbox: queue.SimpleQueue = queue.SimpleQueue()
    server, ports = build_server(inbox=inbox, identity=identity, insecure=insecure, listen=["127.0.0.1:0"],
                                 loopback_plain=False, blobs=blobs, max_message_mb=max_message_mb, relay=relay,
                                 latest=latest)
    server.start()
    loop = MiniHubLoop(inbox, identity=identity, **loop_kwargs)
    port = ports["127.0.0.1:0"]
    return ServedHub(server=server, port=port, address=f"127.0.0.1:{port}", identity=identity, blobs=blobs,
                     run_dir=run_dir, inbox=inbox, loop=loop, insecure=insecure)


# ---------------------------------------------------------------------------------------------------------------
# The real Hub's handlers, driven in-process (T2.5)
# ---------------------------------------------------------------------------------------------------------------


_handle_serials = itertools.count(1_000_000)


class RecordingHandle:
    """A ``SessionHandle`` stand-in that records what the hub sends (``sent``) and whether it was closed."""

    def __init__(self, node_id: str, session_id: str) -> None:
        self.node_id, self.session_id = node_id, session_id
        self.serial = next(_handle_serials)
        self.sent: list[pb.HubMessage] = []
        self.closed_reason: str | None = None

    def send(self, msg: pb.HubMessage) -> None:
        self.sent.append(msg)

    def close(self, reason: str) -> None:
        self.closed_reason = reason

    def of(self, kind: str) -> list:
        """The payloads of the sent messages of oneof ``kind``, in order."""
        return [getattr(m, kind) for m in self.sent if m.WhichOneof("m") == kind]

    def grants(self) -> list[pb.Grant]:
        return [g for batch in self.of("grants") for g in batch.grants]

    def acks(self) -> list[int]:
        return [s for ack in self.of("ack") for s in ack.seqs]


def make_node_entry(**kw: Any):
    """A ``hub.state.NodeEntry`` with neutral defaults (an online node 1 without data addresses); ``kw`` override."""
    from colosseum.hub.state import NodeEntry

    fields: dict[str, Any] = dict(node_id="n" * 32, number=1, name="n", session_id="s", state="online", hardware={},
                                  slots={}, tags=(), data_addresses=(), data_path="relay", last_heartbeat=0.0)
    fields.update(kw)
    return NodeEntry(**fields)


def lineup_hub(config, tmp_path: Path, *, mode: str = "hub", sinks: bool = False, name: str = "hub-run"):
    """A ``Hub`` (``hub`` mode, no local node) after ``launch``'s setup steps (``_prepare``: the coordinator, the
    identity under ``tmp_path``, the pure hub state); ``sinks``: the metrics hub, the journal writer and the live
    ratings too (``_open_sinks``; the caller ends them with ``hub._finish_metrics()``). Nothing listens."""
    from colosseum.hub.hub import Hub, HubOptions
    from game_helpers import make_test_run_dir

    run_dir = make_test_run_dir(config, tmp_path, name=name)
    hub = Hub(config, run_dir, options=HubOptions(mode=mode, local_node=None,
                                                  identity_dir=Path(tmp_path) / "hub-identity"))
    hub._prepare()
    if sinks:
        hub._open_sinks()
    return hub


def open_node(hub, *, name: str = "n", workers: int = 2, envs: int | None = None, cpu_learners: int = 8,
              no_learners: bool = False, session_id: str | None = None, last_acked: int = 0,
              node_id: str | None = None, data_addresses: Sequence[str] = (), gpus: Sequence[int] = (),
              gpu_kind: str = "cuda", tags: Sequence[str] = (),
              learners: Sequence[tuple[str, int, int]] = ()) -> RecordingHandle:
    """Register a node in the hub's node book (unless ``node_id`` names one) and open a session on the real
    ``Hub._on_session_opened``; returns its recording handle. Its ``Hello`` offers ``cpu_learners`` CPU learner slots
    (default 8: a node that takes every learner of a small test config, as before T4.1's placement) and the GPU slots
    ``gpus`` (hardware GPUs of kind ``gpu_kind``), and names its running ``learners`` (``(agent, generation,
    version)``)."""
    import uuid

    from colosseum.hub.server import SessionOpened

    if node_id is None:
        record, _secret = hub.identity.nodes.register(machine_id=uuid.uuid4().hex, name=name, tags=list(tags),
                                                      node_id=None)
        node_id = record.node_id
    handle = RecordingHandle(node_id, session_id or uuid.uuid4().hex)
    envs = hub._config.rollout.envs_per_worker if envs is None else envs
    hello = pb.Hello(node_id=node_id, session_id=handle.session_id, protocol_major=PROTOCOL_VERSION[0],
                     protocol_minor=PROTOCOL_VERSION[1], last_acked_seq=last_acked,
                     slots=pb.Slots(workers=workers, cpu_learners=cpu_learners, no_learners=no_learners,
                                    envs_per_worker=envs, gpus=list(gpus)), data_addresses=list(data_addresses),
                     hardware=pb.Hardware(gpus=[pb.Gpu(index=int(i), kind=gpu_kind) for i in gpus]),
                     learners=[pb.RunningLearner(agent_id=a, generation=g, version=v) for a, g, v in learners],
                     name=name, tags=list(tags))
    hub._on_session_opened(SessionOpened(handle, hello))
    return handle


def accept_version(hub, agent: str, version: int = 0) -> str:
    """An accepted ``VersionAnnounce`` of ``agent``'s learner (from its node): the agent can collect."""
    from colosseum.net.messages import VersionInfo

    number = hub._learner_node_of(agent)
    info = VersionInfo(agent_id=agent, generation=hub._versions.generation(agent), version=version, size=10,
                       digest="d" * 32, node=number)
    return hub._versions.announce(info)


def node_message(handle: RecordingHandle, **oneof) -> NodeMessageIn:
    """A ``NodeMessageIn`` of ``handle``'s stream; a ``results`` batch carries its converted results."""
    from colosseum.net.messages import results_from_batch

    message = pb.NodeMessage(**oneof)
    results = results_from_batch(message.results) if message.WhichOneof("m") == "results" else None
    return NodeMessageIn(handle.node_id, handle.session_id, message, results=results, serial=handle.serial)


def request_cells(hub, handle: RecordingHandle, cells: Sequence[tuple[int, int, int, int]]) -> list[pb.Grant]:
    """A ``LineupRequest`` of ``(worker, env, epoch, incarnation)`` cells; returns the grants of this request."""
    before = len(handle.grants())
    hub._on_message(node_message(handle, lineup_request=pb.LineupRequest(cells=[
        pb.CellRequest(worker=w, env=e, epoch=ep, incarnation=i) for w, e, ep, i in cells])))
    return handle.grants()[before:]


def result_batch(hub, handle: RecordingHandle, results: Sequence, *, seq: int) -> None:
    """A ``ResultBatch`` of ``results`` (``MatchResult``) through ``Hub._on_message``."""
    from colosseum.net.messages import results_to_batch

    hub._on_message(node_message(handle, results=results_to_batch(seq, results)))


def hub_results(hub, number: int, n: int, *, first: int = 0, agents: Sequence[str] = ("a", "random")) -> list:
    """``n`` results of node ``number`` with uids under the hub's run id (``make_result``)."""
    return [make_result(node=number, worker=0, env=k % 4, epoch=1, k=k, run_id=hub.run_id, agents=agents)
            for k in range(first, first + n)]


# ---------------------------------------------------------------------------------------------------------------
# Ports of the SP2-SP4 launcher tests (T2.5 b)
# ---------------------------------------------------------------------------------------------------------------


def make_hub(config, run_dir: RunDir, **options: Any):
    """A ``train``-mode ``Hub`` of ``config`` on ``run_dir`` with its local node options
    (``LocalNodeOptions.from_config``; nothing started). ``options``: ``HubOptions`` fields to override, plus
    ``validated``."""
    from colosseum.hub.hub import Hub, HubOptions, LocalNodeOptions

    validated = bool(options.pop("validated", False))
    fields: dict[str, Any] = {"mode": "train", "local_node": LocalNodeOptions.from_config(config, run_dir)}
    fields.update(options)
    return Hub(config, run_dir, options=HubOptions(**fields), validated=validated)


def hub_with_coordinator(config, tmp_path: Path, *, name: str = "hub-run", metrics: bool = False):
    """A ``train``-mode ``Hub`` whose ``_setup``, ``_coordinator`` and ``_budget`` (a resume's start point included) are
    set as ``launch`` sets them (``Hub._prepare``), without a port, a child or a thread; ``metrics``: the metrics hub
    ``_hub`` too (``_open_metrics``; the caller closes it with ``hub._finish_metrics()``)."""
    from game_helpers import make_test_run_dir

    hub = make_hub(config, make_test_run_dir(config, tmp_path, name=name))
    hub._prepare()
    if metrics:
        hub._open_metrics(hub._resume_states)
    return hub


def _session_of(hub, node_id: str | None):
    if node_id is None:
        node_id = next(iter(hub._sessions))
    return hub._sessions[node_id]


def result_event(hub, results: Sequence, *, seq: int, node_id: str | None = None) -> NodeMessageIn:
    """A ``NodeMessageIn`` with a ``ResultBatch`` of ``results`` (``MatchResult``) on the current session of ``node_id``
    (default: the hub's first session), as the servicer puts it on the inbox (for ``Hub._on_message`` /
    ``_on_result_batch``)."""
    from colosseum.net.messages import results_to_batch

    return node_message(_session_of(hub, node_id), results=results_to_batch(seq, results))


def upload_event(hub, payload: dict, kind: str = "checkpoint", *, node_id: str | None = None) -> UploadIn:
    """A spooled ``UploadIn`` of a learner's snapshot (``learner.make_checkpoint_payload``) as the servicer puts it on
    the inbox: ``kind`` ``checkpoint`` (periodic) or ``final``, the agent's current generation, from the node of the
    agent's learner (default) or ``node_id``. ``reply`` is the Future ``Hub._on_upload`` resolves."""
    import uuid

    from colosseum.net.chunks import encode_checkpoint

    agent = str(payload["agent_id"])
    generation = int(hub._versions.generation(agent))
    if node_id is None:
        number = hub._learner_node_of(agent)
        node_id = next(e.node_id for e in hub._registry.all() if e.number == number)
    data = encode_checkpoint(payload, kind="final" if kind == "final" else "periodic", generation=generation)
    hub._blobs.incoming.mkdir(parents=True, exist_ok=True)
    path = hub._blobs.incoming / f"{uuid.uuid4().hex}.part"
    path.write_bytes(data)
    header = pb.UploadHeader(kind=kind, agent_id=agent, generation=generation,
                             version=int(payload["policy_version"]), total_size=len(data))
    return UploadIn(header, path, node_id, Future())


class _NodeThreadProcess:
    """A stand-in for the local node's ``Process`` (target ``node.agent.local_node_target``): ``start`` runs the node
    the hub configured (``kwargs["options"]``) in a thread of this process (``node_kit.NodeThread``)."""

    def __init__(self, target: Callable, kwargs: dict, name: str | None = None, daemon: bool | None = None) -> None:
        from colosseum.node.agent import local_node_target

        assert target is local_node_target, target
        self.kwargs, self.name, self.node = kwargs, name, None

    def start(self) -> None:
        from node_kit import NodeThread

        self.node = NodeThread(self.kwargs["options"])

    def is_alive(self) -> bool:
        return self.node is not None and self.node.thread.is_alive()

    @property
    def exitcode(self) -> int | None:
        return None if self.node is None or self.is_alive() else self.node.exit_code

    def terminate(self) -> None:
        if self.node is not None:
            self.node.agent.request_stop()

    kill = terminate

    def join(self, timeout: float | None = None) -> None:
        if self.node is not None:
            self.node.thread.join(timeout)


class _ThreadNodeContext:
    """The hub's spawn context with the local node run in a thread (``_NodeThreadProcess``); the rest is real."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.processes: list[_NodeThreadProcess] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def Process(self, target: Callable, kwargs: dict, name: str | None = None, daemon: bool | None = None):
        proc = _NodeThreadProcess(target, kwargs, name, daemon)
        self.processes.append(proc)
        return proc


def local_node_spawns(config, tmp_path: Path, *, name: str = "spawns", timeout: float = 60.0):
    """What ``train`` spawns, without spawning: a ``train`` hub (``make_hub``, launched in a thread) whose local node
    (``Hub._start_local_node``: its own ``NodeOptions``) runs in a thread of this process instead of a child, with
    ``ChildSupervisor.start`` recording each ``ChildSpec`` instead of starting a process. Returns ``(specs, run_dir)``
    (specs in start order) once a learner per trainable agent and every worker was started; the hub is stopped first
    (its exit code 0 and the node's end asserted)."""
    from unittest import mock

    from colosseum.node.procs import ChildSupervisor
    from game_helpers import make_test_run_dir

    run_dir = make_test_run_dir(config, tmp_path, name=name)
    hub = make_hub(config, run_dir, identity_dir=Path(tmp_path) / "hub-identity")
    ctx = hub._mp = _ThreadNodeContext(hub._mp)
    specs: list = []

    def recording_start(self, spec):
        incarnation = self._incarnations.get(spec.name, 0) + 1
        self._incarnations[spec.name] = incarnation
        specs.append(spec)
        return incarnation

    learners = len(config.get_trainable_agent_ids())
    workers = int(config.rollout.num_workers)
    codes: list[int] = []
    with mock.patch.object(ChildSupervisor, "start", recording_start):
        thread = threading.Thread(target=lambda: codes.append(hub.launch()), name=f"hub-{name}", daemon=True)
        thread.start()
        try:
            poll_until(lambda: (sum(s.role == "learner" for s in specs) >= learners
                                and sum(s.role == "worker" for s in specs) >= workers) or not thread.is_alive(),
                       timeout)
        finally:
            hub.request_stop()
            thread.join(timeout)
            for proc in ctx.processes:
                proc.terminate()
                proc.join(timeout)
    assert not thread.is_alive() and codes == [0], codes
    assert [p.name for p in ctx.processes] == ["node"] and not ctx.processes[0].is_alive()
    return list(specs), run_dir

"""Test support for the node agent (SP5 T2.4, ruling R13: node logic tested with a fake hub against the real gRPC
server).

- ``FakeHub(tmp_path, config)``: the real ``hub.server`` (insecure, ``127.0.0.1:0``) with a real ``HubIdentity`` under
  ``tmp_path`` and a scripted main loop: it registers nodes, answers every session with a ``Welcome`` (the config,
  the run id, ``welcome_fields``), grants lineups through ``lineup_for(worker, env)`` with epochs from a ``CellTable``,
  accounts results at intake with a ``ResultIntake`` (dedup by (node, session, seq)) and acks them (or holds the acks),
  and records every inbox event. Downloads are served by the real servicer from ``GatedBlobs`` (gates, NOT_FOUND).
  ``pause()`` / ``resume()`` (P-PF2-16): while paused every new session is closed before its ``Welcome`` and live
  ones are closed, and the server keeps its port.
- ``toy_node_options(tmp_path, hub, **kw)``: ``NodeOptions`` of a registered, insecure one-worker node.
- ``run_node_in_thread(options)`` / ``stop_node(...)``: a ``NodeAgent`` in a thread of the test process.
- ``wait_for(pred, timeout)``: poll a predicate.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colosseum.core.run_dir import RunDir
from colosseum.core.types import Lineup
from colosseum.hub.server import (
    BlobStore,
    NodeMessageIn,
    RegisterIn,
    SessionClosed,
    SessionHandle,
    SessionOpened,
    UploadIn,
    blob_file_name,
    blob_ident,
    build_server,
)
from colosseum.hub.state import CellKey, CellTable, EpochAllocator, ResultIntake
from colosseum.net.messages import lineup_to_proto, results_from_batch
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import HubIdentity
from colosseum.node.agent import NodeAgent, NodeOptions


def wait_for(predicate: Callable[[], Any], timeout: float = 30.0, interval: float = 0.02) -> Any:
    """Poll ``predicate`` until it returns a truthy value (returned) or ``timeout`` passes (AssertionError)."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout} s")
        time.sleep(interval)


class GatedBlobs(BlobStore):
    """A ``BlobStore`` whose answers the test controls: ``gates[(kind, agent, ident)]`` (an Event) holds a download
    until set, ``missing`` answers NOT_FOUND; ``timeline`` (shared with the hub) records every request."""

    def __init__(self, run_dir: RunDir, timeline: list) -> None:
        super().__init__(run_dir)
        self.gates: dict[tuple[str, str, str], threading.Event] = {}
        self.missing: set[tuple[str, str, str]] = set()
        self.timeline = timeline

    def frame_path(self, kind: str, agent_id: str, ident: str) -> Path | None:
        triple = (kind, agent_id, ident)
        self.timeline.append(("download", triple))
        if triple in self.missing:
            return None
        gate = self.gates.get(triple)
        if gate is not None:
            gate.wait(60)
        return super().frame_path(kind, agent_id, ident)

    def requests(self) -> list[tuple[str, str, str]]:
        return [t for k, t in list(self.timeline) if k == "download"]


@dataclass
class _Call:
    fn: Callable[[], Any]
    reply: Future = field(default_factory=Future)


class FakeHub:
    """Module docstring. ``lineup_for(worker, env) -> Lineup | None`` (None: no grant for that cell); ``first_epoch``:
    the first epoch the hub hands out; ``welcome_fields``: extra ``pb.Welcome`` fields (``fixed``, ``learners``)."""

    def __init__(self, tmp_path: Path, config: Any, *, ttl_sec: float = 1.0, first_epoch: int = 5) -> None:
        self.config = config
        self.ttl_sec = ttl_sec
        self.identity = HubIdentity.load_or_create(Path(tmp_path) / "hub-id")
        self.token = self.identity.token
        self.run_dir = RunDir(Path(tmp_path) / "run")
        (self.run_dir.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.timeline: list = []
        self.blobs = GatedBlobs(self.run_dir, self.timeline)
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.server, ports = build_server(inbox=self.inbox, identity=self.identity, insecure=True,
                                          listen=["127.0.0.1:0"], loopback_plain=False, blobs=self.blobs,
                                          max_message_mb=64)
        self.server.start()
        self.port = ports["127.0.0.1:0"]
        self.address = f"127.0.0.1:{self.port}"
        self.run_id = uuid.uuid4().hex
        self.welcome_fields: dict = {}
        self.lineup_for: Callable[[int, int], Lineup | None] = lambda worker, env: None
        self.hold_acks = False
        self.refuse_once: set[int] = set()     # result seqs refused once (as a full journal writer does: no ack)
        self.paused = False
        self.cells = CellTable(EpochAllocator(first_epoch - 1, lambda until: None))
        self.intake = ResultIntake(journal=False)
        self.sessions: dict[str, SessionHandle] = {}
        self.received: list = []
        self.registrations: list[pb.RegisterRequest] = []
        self.accounted: list[str] = []
        self.acked: list[int] = []
        self.errors: list[str] = []
        self._held: list[tuple[str, str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="fake-hub", daemon=True)
        self._thread.start()

    # ---- test API (any thread) ----------------------------------------------------------------------------------

    def call(self, fn: Callable[[], Any], timeout: float = 10.0) -> Any:
        c = _Call(fn)
        self.inbox.put(c)
        return c.reply.result(timeout)

    def sync(self) -> None:
        """Return once every inbox event queued so far is handled (after a node stopped: its last messages)."""
        self.call(lambda: None)

    def events(self, cls: type) -> list:
        return [e for e in list(self.received) if isinstance(e, cls)]

    def node_messages(self, kind: str) -> list[NodeMessageIn]:
        return [e for e in self.events(NodeMessageIn) if e.message.WhichOneof("m") == kind]

    def messages(self, kind: str) -> list:
        """The payloads of every received ``NodeMessage`` of oneof ``kind``, in arrival order."""
        return [getattr(e.message, kind) for e in self.node_messages(kind)]

    def node_events(self, kind: str) -> list[pb.Event]:
        return [e for e in self.messages("event") if e.kind == kind]

    def results(self) -> list:
        """Every ``MatchResult`` received (resends included), in arrival order."""
        return [r for e in self.node_messages("results") for r in results_from_batch(e.message.results)]

    def welcome(self, **fields: Any) -> None:
        """Extra ``pb.Welcome`` fields of every later session (``fixed=[...]``, ``learners=[...]``)."""
        self.welcome_fields.update(fields)

    def grant(self, lineup_for: Callable[[int, int], Lineup | None]) -> None:
        """How cells are granted from now on: ``lineup_for(worker, env)`` (None: no grant)."""
        self.lineup_for = lineup_for

    def send(self, **oneof: Any) -> None:
        """One ``HubMessage`` to every live session (``send(command=pb.Command(kind="stop_run"))``)."""
        def send():
            for handle in self.sessions.values():
                handle.send(pb.HubMessage(**oneof))
        self.call(send)

    def send_command(self, kind: str, **fields: Any) -> None:
        self.send(command=pb.Command(kind=kind, **fields))

    def announce(self, **fields: Any) -> None:
        self.send(announce=pb.Announce(**fields))

    def close_session(self, reason: str = "closed by the test") -> None:
        def close():
            for handle in list(self.sessions.values()):
                handle.close(reason)
        self.call(close)

    def pause(self) -> None:
        def pause():
            self.paused = True
            for handle in list(self.sessions.values()):
                handle.close("paused")
        self.call(pause)

    def resume(self) -> None:
        def resume():
            self.paused = False
        self.call(resume)

    stop_server = pause            # A4's names: the hub goes away and comes back on the same port (P-PF2-16)
    restart_server = resume

    def release_acks(self) -> None:
        def release():
            held, self._held = self._held, []
            for node_id, session_id, seq in held:
                self._ack(node_id, session_id, seq)
        self.call(release)

    def serve_blob(self, kind: str, agent: str, ident: str, data: bytes, *,
                   gate: threading.Event | None = None) -> None:
        """Serve ``data`` for ``Download(kind, agent, ident)`` (a snapshot frame, or a registered fixed / teacher /
        init blob); ``gate``: held until it is set."""
        if kind == "snapshot":
            self.blobs.dir.mkdir(parents=True, exist_ok=True)
            (self.blobs.dir / blob_file_name("snapshot", agent, ident)).write_bytes(data)
        else:
            self.blobs.register_fixed(ident, data, kind=kind, agent_id=agent)
        if gate is not None:
            self.blobs.gates[(kind, agent, ident)] = gate

    def not_found(self, kind: str, agent: str, ident: str) -> None:
        self.blobs.missing.add((kind, agent, ident))

    def serve_frozen(self, agent: str, path: Path) -> str:
        """Register frozen agent ``agent``'s weights (``agents.<agent>.path``) as a fixed blob listed in the Welcome;
        returns its blob id."""
        from colosseum.core.registry import env_spec
        from colosseum.net.chunks import WeightsMeta, encode_weights
        from colosseum.players.registry import load_frozen

        frozen = load_frozen(self.config, agent, path, env_spec(self.config))
        frame = encode_weights(frozen.model_state, WeightsMeta(
            agent_id=agent, kind="fixed", generation=0, policy_version=0, architecture=dict(frozen.networks),
            roles=tuple(frozen.roles)))
        blob_id = blob_ident("fixed", agent, frame)
        self.serve_blob("fixed", agent, blob_id, frame)
        self.welcome_fields.setdefault("fixed", []).append(pb.FixedBlob(agent_id=agent, blob_id=blob_id,
                                                                        kind="fixed"))
        return blob_id

    def heartbeats(self) -> list[pb.Heartbeat]:
        return self.messages("heartbeat")

    def stop(self) -> None:
        self.server.stop(None)
        self._stop.set()
        self._thread.join(5)

    # ---- the loop -----------------------------------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue
            if isinstance(ev, _Call):
                try:
                    ev.reply.set_result(ev.fn())
                except Exception as e:  # noqa: BLE001 - handed to the caller
                    ev.reply.set_exception(e)
                continue
            self.received.append(ev)
            try:
                self._dispatch(ev)
            except Exception as e:  # noqa: BLE001 - recorded for the test
                self.errors.append(f"{type(e).__name__}: {e}")

    def _dispatch(self, ev: object) -> None:
        if isinstance(ev, RegisterIn):
            self.registrations.append(ev.request)
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
        elif isinstance(ev, SessionOpened):
            handle = ev.handle
            if self.paused:
                handle.close("paused")
                return
            record = self.identity.nodes.get(handle.node_id)
            if record is None:
                handle.close("unknown node")
                return
            old = self.sessions.get(handle.node_id)
            if old is not None and old is not handle:
                old.close("superseded")
            self.sessions[handle.node_id] = handle
            self.intake.open(handle.node_id, ev.hello.session_id, int(ev.hello.last_acked_seq))
            handle.send(pb.HubMessage(welcome=pb.Welcome(
                node_number=record.number, run_id=self.run_id,
                config_json=self.config.model_dump_json(by_alias=True), **self.welcome_fields)))
        elif isinstance(ev, SessionClosed):
            if self.sessions.get(ev.node_id) is ev.handle:
                del self.sessions[ev.node_id]
        elif isinstance(ev, NodeMessageIn):
            handle = self.sessions.get(ev.node_id)
            if handle is None or handle.serial != ev.serial:
                return
            self._on_message(ev, handle)
        elif isinstance(ev, UploadIn):
            ev.path.unlink(missing_ok=True)
            ev.reply.set_result(pb.UploadReply(accepted=True, checkpoint_id=f"ckpt_v{ev.header.version}"))

    def _on_message(self, ev: NodeMessageIn, handle: SessionHandle) -> None:
        kind = ev.message.WhichOneof("m")
        if kind == "lineup_request":
            self.timeline.append(("lineup_request", None))
            number = self.identity.nodes.get(ev.node_id).number
            now = time.monotonic()
            cells = [(CellKey(number, c.worker, c.env), int(c.epoch), int(c.incarnation))
                     for c in ev.message.lineup_request.cells]
            grants = []
            for cell in self.cells.request(number, cells, now):
                lineup = self.lineup_for(cell.key.worker, cell.key.env)
                if lineup is None:
                    continue
                self.cells.grant(cell.key, "a", self.ttl_sec, now)
                lineup = Lineup(layout=lineup.layout, seats=list(lineup.seats), epoch=cell.epoch)
                grants.append(pb.Grant(worker=cell.key.worker, env=cell.key.env, epoch=cell.epoch,
                                       ttl_sec=self.ttl_sec, lineup=lineup_to_proto(lineup)))
            if grants:
                handle.send(pb.HubMessage(grants=pb.LineupGrants(grants=grants)))
        elif kind == "results":
            seq = int(ev.message.results.seq)
            if seq in self.refuse_once:
                self.refuse_once.discard(seq)
                return
            if not self.intake.begin(ev.node_id, ev.session_id, seq):
                if self.intake.is_acked(ev.node_id, ev.session_id, seq):
                    handle.send(pb.HubMessage(ack=pb.Ack(seqs=[seq])))
                return
            self.accounted.extend(r.uid for r in ev.results or ())
            if self.hold_acks:
                self._held.append((ev.node_id, ev.session_id, seq))
            else:
                self._ack(ev.node_id, ev.session_id, seq)

    def _ack(self, node_id: str, session_id: str, seq: int) -> None:
        self.intake.committed(node_id, session_id, seq)
        self.acked.append(seq)
        handle = self.sessions.get(node_id)
        if handle is not None:
            handle.send(pb.HubMessage(ack=pb.Ack(seqs=[seq])))


def toy_node_options(tmp_path: Path, hub: FakeHub, **kw: Any) -> NodeOptions:
    """A registered, insecure node with one worker, its identity and logs under ``tmp_path``."""
    fields = dict(hub_address=hub.address, token=hub.token, insecure=True, workers=1,
                  identity_root=Path(tmp_path) / "node-id", log_dir=Path(tmp_path) / "node-logs")
    fields.update(kw)
    return NodeOptions(**fields)


class NodeThread:
    """A ``NodeAgent`` running in a thread of the test process; ``exit_code`` once ``run`` returned."""

    def __init__(self, options: NodeOptions) -> None:
        self.agent = NodeAgent(options)
        self.exit_code: int | None = None
        self.thread = threading.Thread(target=self._run, name="node-agent", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.exit_code = self.agent.run()

    def stop(self, timeout: float = 30.0) -> int | None:
        self.agent.request_stop()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the node did not stop"
        return self.exit_code


def run_node_in_thread(options: NodeOptions) -> NodeThread:
    return NodeThread(options)

"""Test support for the node agent (SP5 T2.4, ruling R13: node logic tested with a fake hub against the real gRPC
server).

- ``FakeHub(tmp_path, config)``: the real ``hub.server`` (insecure, ``127.0.0.1:0``) with a real ``HubIdentity`` under
  ``tmp_path`` and a scripted main loop: it registers nodes, answers every session with a ``Welcome`` (the config,
  the run id, ``welcome_fields``), grants lineups through ``lineup_for(worker, env)`` with epochs from a ``CellTable``,
  accounts results at intake with a ``ResultIntake`` (dedup by (node, session, seq)) and acks them (or holds the acks),
  and records every inbox event. Downloads are served by the real servicer from ``GatedBlobs`` (gates, NOT_FOUND).
  ``pause()`` / ``resume()`` (P-PF2-16): while paused every new session is closed before its ``Welcome`` and live
  ones are closed, and the server keeps its port. ``relay=True`` (T4.4): the server has a real ``RelayHub`` and
  serves ``Hub.RelaySend`` / ``RelayInbox`` for the nodes whose session is open (frames for other nodes are dropped
  and counted by the relay, as the hub does for an offline target); ``blobs.write_parked(agent, frame)`` serves an
  agent's latest weights through ``Download(kind="latest")``.
- ``toy_node_options(tmp_path, hub, **kw)``: ``NodeOptions`` of a registered, insecure one-worker node.
- ``run_node_in_thread(options)`` / ``stop_node(...)``: a ``NodeAgent`` in a thread of the test process.
- ``run_node_main(options_json)`` (T4.4): a node agent as a script (its own process, SIGINT / SIGTERM handlers,
  logging to stderr) for the signal and exit-code tests; ``node_options_json`` / ``start_node_main`` build and start
  one (``python -c``, its own session).
- ``wait_for(pred, timeout)``: poll a predicate.
- Evaluation (T5.1): ``FakeHub.lease(lease_id, tasks)`` sends a ``BlockLease``; every ``BlockResult`` is acked
  (``BlockAck``) unless ``block_acks`` is False (``ack_blocks()`` acks the ones received so far); eval blobs are
  served by the real ``BlobStore.register_weights`` (``eval_kit.gauntlet_tasks(..., blobs=hub.blobs)``);
  ``refuse_downloads()`` answers every eval download NOT_FOUND; ``eval_downloads()``, ``block_results()``,
  ``lease_requests()`` read what the hub saw.
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
from colosseum.hub.relay import RelayHub
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
        self.refuse_eval = False                          # every Download(kind="eval") NOT_FOUND (T5.1)
        self.timeline = timeline

    def frame_path(self, kind: str, agent_id: str, ident: str) -> Path | None:
        triple = (kind, agent_id, ident)
        self.timeline.append(("download", triple))
        if triple in self.missing or (kind == "eval" and self.refuse_eval):
            return None
        gate = self.gates.get(triple)
        if gate is not None:
            gate.wait(60)
        return super().frame_path(kind, agent_id, ident)

    def start_path(self, agent_id: str, ident: str) -> Path | None:
        """``Download(kind="start")`` (T4.1, A28) under the same controls as ``frame_path``."""
        triple = ("start", agent_id, ident)
        self.timeline.append(("download", triple))
        if triple in self.missing:
            return None
        gate = self.gates.get(triple)
        if gate is not None:
            gate.wait(60)
        return super().start_path(agent_id, ident)

    def requests(self) -> list[tuple[str, str, str]]:
        return [t for k, t in list(self.timeline) if k == "download"]


@dataclass
class _Call:
    fn: Callable[[], Any]
    reply: Future = field(default_factory=Future)


class FakeHub:
    """Module docstring. ``lineup_for(worker, env) -> Lineup | None`` (None: no grant for that cell); ``first_epoch``:
    the first epoch the hub hands out; ``welcome_fields``: extra ``pb.Welcome`` fields (``fixed``, ``learners``)."""

    def __init__(self, tmp_path: Path, config: Any, *, ttl_sec: float = 1.0, first_epoch: int = 5,
                 relay: bool = False) -> None:
        self.config = config
        self.ttl_sec = ttl_sec
        self.identity = HubIdentity.load_or_create(Path(tmp_path) / "hub-id")
        self.token = self.identity.token
        self.run_dir = RunDir(Path(tmp_path) / "run")
        (self.run_dir.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.timeline: list = []
        self.blobs = GatedBlobs(self.run_dir, self.timeline)
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.relay = RelayHub() if relay else None
        self.server, ports = build_server(inbox=self.inbox, identity=self.identity, insecure=True,
                                          listen=["127.0.0.1:0"], loopback_plain=False, blobs=self.blobs,
                                          max_message_mb=64, relay=self.relay)
        self.server.start()
        self.port = ports["127.0.0.1:0"]
        self.address = f"127.0.0.1:{self.port}"
        self.run_id = uuid.uuid4().hex
        self.welcome_fields: dict = {}
        self.lineup_for: Callable[[int, int], Lineup | None] = lambda worker, env: None
        self.grants_sent: list[tuple[int, int, int]] = []     # (worker, env, epoch) of every grant sent
        self.hold_acks = False
        self.block_acks = True                 # T5.1: every BlockResult is acked at once
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
        self.uploads: list[tuple[pb.UploadHeader, bytes]] = []     # every upload handled, in order
        self.upload_reply: Callable[[pb.UploadHeader, bytes], pb.UploadReply] = (
            lambda header, data: pb.UploadReply(accepted=True, checkpoint_id=f"ckpt_v{header.version}"))
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
        """Serve ``data`` for ``Download(kind, agent, ident)`` (a snapshot frame, a learner start point of kind
        ``start`` (A28), or a registered fixed / teacher / init blob); ``gate``: held until it is set."""
        if kind == "snapshot":
            self.blobs.dir.mkdir(parents=True, exist_ok=True)
            (self.blobs.dir / blob_file_name("snapshot", agent, ident)).write_bytes(data)
        elif kind == "start":
            self.blobs.write_start(agent, ident, data)
        else:
            self.blobs.register_fixed(ident, data, kind=kind, agent_id=agent)
        if gate is not None:
            self.blobs.gates[(kind, agent, ident)] = gate

    def not_found(self, kind: str, agent: str, ident: str) -> None:
        self.blobs.missing.add((kind, agent, ident))

    def serve_frozen(self, agent: str, path: Path, *, gate: threading.Event | None = None) -> str:
        """Register frozen agent ``agent``'s weights (``agents.<agent>.path``) as a fixed blob listed in the Welcome
        (``gate``: its download is held until set); returns its blob id."""
        from colosseum.core.registry import env_spec
        from colosseum.net.chunks import WeightsMeta, encode_weights
        from colosseum.players.registry import load_frozen

        frozen = load_frozen(self.config, agent, path, env_spec(self.config))
        frame = encode_weights(frozen.model_state, WeightsMeta(
            agent_id=agent, kind="fixed", generation=0, policy_version=0, architecture=dict(frozen.networks),
            roles=tuple(frozen.roles)))
        blob_id = blob_ident("fixed", agent, frame)
        self.serve_blob("fixed", agent, blob_id, frame, gate=gate)
        self.welcome_fields.setdefault("fixed", []).append(pb.FixedBlob(agent_id=agent, blob_id=blob_id,
                                                                        kind="fixed"))
        return blob_id

    # ---- evaluation (T5.1) -------------------------------------------------------------------------------------

    def lease(self, lease_id: str, tasks: list, lease_sec: float = 600.0) -> None:
        """One ``BlockLease`` of ``tasks`` (``BlockTask``s) to every live session."""
        from colosseum.net.messages import lease_to_proto

        self.send(lease=lease_to_proto(lease_id, lease_sec, tasks))

    def ack_blocks(self) -> None:
        """Ack every ``BlockResult`` received so far (``block_acks`` False holds them)."""
        def ack():
            for ev in self.node_messages("block_result"):
                handle = self.sessions.get(ev.node_id)
                if handle is not None:
                    r = ev.message.block_result
                    handle.send(pb.HubMessage(block_ack=pb.BlockAck(lease_id=r.lease_id, block_ids=[r.block_id])))
        self.call(ack)

    def refuse_downloads(self) -> None:
        self.blobs.refuse_eval = True

    def eval_downloads(self, ident: str | None = None) -> list[tuple[str, str, str]]:
        return [t for t in self.blobs.requests() if t[0] == "eval" and (ident is None or t[2] == ident)]

    def block_results(self, lease_id: str | None = None) -> list[pb.BlockResult]:
        return [r for r in self.messages("block_result") if lease_id is None or r.lease_id == lease_id]

    def lease_requests(self) -> list[int]:
        return [int(m.free_slots) for m in self.messages("lease_request")]

    def sessions_opened(self) -> list[pb.Hello]:
        return [e.hello for e in self.events(SessionOpened)]

    def versions(self, agent: str | None = None) -> list[pb.VersionInfo]:
        """Every ``VersionAnnounce``'s info received (of ``agent``), in arrival order."""
        return [m.info for m in self.messages("version") if agent is None or m.info.agent_id == agent]

    def uploads_of(self, kind: str) -> list[pb.UploadHeader]:
        return [h for h, _ in list(self.uploads) if h.kind == kind]

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
            self._relay_online()
            self.intake.open(handle.node_id, ev.hello.session_id, int(ev.hello.last_acked_seq))
            handle.send(pb.HubMessage(welcome=pb.Welcome(
                node_number=record.number, run_id=self.run_id,
                config_json=self.config.model_dump_json(by_alias=True), **self.welcome_fields)))
        elif isinstance(ev, SessionClosed):
            if self.sessions.get(ev.node_id) is ev.handle:
                del self.sessions[ev.node_id]
                self._relay_online()
        elif isinstance(ev, NodeMessageIn):
            handle = self.sessions.get(ev.node_id)
            if handle is None or handle.serial != ev.serial:
                return
            self._on_message(ev, handle)
        elif isinstance(ev, UploadIn):
            data = ev.path.read_bytes()
            ev.path.unlink(missing_ok=True)
            self.uploads.append((ev.header, data))
            ev.reply.set_result(self.upload_reply(ev.header, data))

    def _relay_online(self) -> None:
        if self.relay is not None:
            self.relay.set_online({node_id: self.identity.nodes.get(node_id).number for node_id in self.sessions})

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
                self.grants_sent.append((cell.key.worker, cell.key.env, cell.epoch))
                grants.append(pb.Grant(worker=cell.key.worker, env=cell.key.env, epoch=cell.epoch,
                                       ttl_sec=self.ttl_sec, lineup=lineup_to_proto(lineup)))
            if grants:
                handle.send(pb.HubMessage(grants=pb.LineupGrants(grants=grants)))
        elif kind == "block_result":
            if self.block_acks:
                r = ev.message.block_result
                handle.send(pb.HubMessage(block_ack=pb.BlockAck(lease_id=r.lease_id, block_ids=[r.block_id])))
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


_PATH_FIELDS = ("identity_root", "log_dir")


def node_options_json(options: NodeOptions) -> str:
    """``options`` as the JSON ``run_node_main`` takes (the token as its text; no in-memory data credentials)."""
    import dataclasses
    import json

    data: dict[str, Any] = {}
    for f in dataclasses.fields(options):
        value = getattr(options, f.name)
        if f.name == "data_credentials":
            continue
        if f.name == "token":
            value = str(value)
        elif isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        data[f.name] = value
    return json.dumps(data)


def run_node_main(options_json: str) -> int:
    """The node agent as a script (T4.4): ``NodeOptions`` from ``options_json`` (``node_options_json``), process
    logging to stderr, SIGINT / SIGTERM stop it (``node.agent.install_stop_signals``); returns ``NodeAgent.run``'s
    exit code."""
    import json
    import logging

    from colosseum.net.security import parse_token
    from colosseum.node.agent import install_stop_signals
    from colosseum.utils.logging import setup_process_logging

    data = json.loads(options_json)
    data["token"] = parse_token(data["token"])
    for name in _PATH_FIELDS:
        if data.get(name) is not None:
            data[name] = Path(data[name])
    for name, value in list(data.items()):
        if isinstance(value, list):
            data[name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
    setup_process_logging(None, "node", console_level=logging.INFO)
    agent = NodeAgent(NodeOptions(**data))
    install_stop_signals(agent)
    return agent.run()


def start_node_main(options: NodeOptions, stderr: Path) -> Any:
    """``run_node_main`` in its own process and session (``python -c``; ``tests/`` on its path); its stderr goes to
    ``stderr``. Returns the ``Popen``; the caller signals / waits / kills it (process group). Its cwd is the repo root,
    as for every CLI child (``cli_runner``): ``python -c`` puts the cwd on ``sys.path``, so configs may name
    ``examples.*`` (the test's own cwd is its ``tmp_path``, ``conftest._isolate_filesystem``)."""
    import subprocess
    import sys

    from cli_runner import REPO_ROOT, child_env

    code = "import sys, node_kit; sys.exit(node_kit.run_node_main(sys.argv[1]))"
    with open(stderr, "wb") as err:
        return subprocess.Popen([sys.executable, "-c", code, node_options_json(options)], cwd=REPO_ROOT,
                                env=child_env(), stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)


def start_node_process(options: NodeOptions, log_dir: Path) -> Any:
    """The node as a spawned process (``node.agent.local_node_target``: the child signal policy and logging of a
    hub's local node); returns the started ``Process``."""
    import multiprocessing as mp

    from colosseum.node.agent import local_node_target

    proc = mp.get_context("spawn").Process(target=local_node_target, kwargs={"options": options,
                                                                            "log_dir": str(log_dir)}, daemon=False)
    proc.start()
    return proc


def spawned_children(pid: int) -> list[int]:
    """Linux: the pids of ``pid``'s live children started by multiprocessing's spawn (``/proc``; zombies excluded)."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        fields = stat.rsplit(")", 1)[1].split()
        if int(fields[1]) == pid and fields[0] != "Z" and b"spawn_main" in cmdline:
            out.append(int(entry.name))
    return out


def pid_alive(pid: int) -> bool:
    """The process exists and is not a zombie."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


# ---------------------------------------------------------------------------------------------------------------
# The node's process bodies in this process (ports of the SP2-SP4 launcher tests, T2.5 b)
# ---------------------------------------------------------------------------------------------------------------


class _NullConn:
    """A connection / listener / channel stand-in: accepts sends, closes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sent: list = []
        self.stop_flag = threading.Event()
        self.env_steps = 0

    def send(self, item: Any) -> None:
        self.sent.append(item)

    def close(self) -> None:
        pass


def run_learner_main(monkeypatch: Any, config: Any, agent_id: str = "agent_0", *, intake_stub: Any = None,
                     **kwargs: Any) -> None:
    """``node.learner_main._learner_main`` of ``config``'s ``agent_id`` in this process, with the node side stubbed (no
    listener, no control connection, no weight cache: ``net.local`` / ``node.weights`` stand-ins). The body seeds, sets
    the torch threads, builds the algorithm (``teacher`` resolved like the hub does, ``init_state``) and calls
    ``learner.learner.learner_process``, which the caller monkeypatches to see what it gets. ``kwargs`` override the
    spawn kwargs (``seed``, ``num_learners``, ``init_state``, ...); ``intake_stub``: the ``ConnectionIntake`` stand-in
    (default ``_NullConn``)."""
    import colosseum.net.local as local
    import colosseum.node.weights as weights
    from colosseum.core.registry import env_spec
    from colosseum.core.roles import agent_role_spec
    from colosseum.learner.factory import resolve_teacher
    from colosseum.node.learner_main import _learner_main
    from colosseum.players.registry import resolve_player_roles

    for name in ("connect", "listener", "ControlChannel", "ConnectionIntake", "NodeOutputs"):
        monkeypatch.setattr(local, name, _NullConn)
    if intake_stub is not None:
        monkeypatch.setattr(local, "ConnectionIntake", intake_stub)
    monkeypatch.setattr(weights, "WeightCacheWriter", _NullConn)
    spec = env_spec(config)
    role = agent_role_spec(spec, resolve_player_roles(config, spec)[agent_id])
    fields: dict[str, Any] = dict(
        name=f"learner-{agent_id}", incarnation=1, node=0, agent_id=agent_id, generation=1, listener_name="unused",
        node_address="unused", authkey=b"unused", config=config.get_agent_config(agent_id), role_spec=role, spec=spec,
        seed=None, teacher=resolve_teacher(config, agent_id, spec), init_state=None, resume_state=None,
        version_floor=0, env_steps=0, num_learners=1, cache_root="", cache_prefix="",
        checkpoint_interval=int(config.checkpoint.interval), total_timesteps=int(config.training.total_timesteps),
        cap_bytes=1 << 26)
    fields.update(kwargs)
    _learner_main(**fields)


class PlayIO:
    """The ``WorkerIO`` side a node worker's body (``node.worker_main._play``) reads before it builds its loop: the
    first lineup of every env (``lineups``), never stopped; ``ready`` is the first weights' flag (``WeightsReady``)."""

    def __init__(self, lineups: list[Lineup]) -> None:
        from colosseum.net.local import CellLineup

        self._first = {e: CellLineup(e, lineup, {}) for e, lineup in enumerate(lineups)}
        self.weights_ready = False
        self.stats_fn = None

    def stopped(self) -> bool:
        return False

    def ready(self) -> bool:
        return self.weights_ready

    def take_lineups(self) -> dict:
        out, self._first = self._first, {}
        return out

    def note_snapshot(self, agent_id: str, checkpoint_id: str) -> None:
        pass

    def loop_io(self) -> Any:
        return self

    def idle(self) -> None:
        pass

    def close(self) -> None:
        pass

    def counters(self) -> dict:
        return {}


def run_worker_play(monkeypatch: Any, config: Any, lineups: list[Lineup], *, worker: int = 0,
                    run_id: str = "RUN") -> tuple[dict, PlayIO]:
    """``node.worker_main._play`` of ``config`` (its trainable agents, scripted bots and teachers as the node passes
    them; no frozen agents) with the first ``lineups``, ``RolloutLoop`` replaced by a recorder: returns the
    ``RolloutLoop`` keyword arguments and the io."""
    import colosseum.worker.rollout_loop as rollout_loop
    from colosseum.hub.setup import setup_run
    from colosseum.node.agent import _scripted_teachers
    from colosseum.node.worker_main import _play

    recorded: dict = {}

    class RecordingLoop:
        stats: dict = {}

        def __init__(self, **kwargs: Any) -> None:
            recorded.update(kwargs)

        def run(self, should_stop: Callable[[], bool]) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(rollout_loop, "RolloutLoop", RecordingLoop)
    setup = setup_run(config, validate=False)
    teachers = _scripted_teachers(config)
    io = PlayIO(lineups)
    _play(io, cache=None, node=0, worker=worker, incarnation=1, config=config,
          agent_ids=config.get_trainable_agent_ids(), agent_roles=setup.agent_roles,
          agent_configs=setup.agent_configs, role_specs=setup.role_specs, bots=dict(setup.fixed.bots),
          frozen_keys={}, fixed_roles=dict(setup.fixed.roles), teachers=teachers, run_id=run_id)
    return recorded, io


def train_on_node(tmp_path: Path, config: Any, *, learners: list[str], lineup: Callable[[int, int], Lineup | None],
                  frozen: dict[str, Path] | None = None, **node_options: Any) -> tuple[FakeHub, NodeThread]:
    """A ``FakeHub`` of ``config`` that places the learners of ``learners`` (generation 1, CPU) on a node run in a
    thread of this process (``toy_node_options``; ``node_options`` override, e.g. ``workers``) and grants
    ``lineup(worker, env)`` once each of them announced its first version; ``frozen``: frozen agents' weights served as
    fixed blobs (``FakeHub.serve_frozen``). Returns both; the caller stops them (``node.stop()``, ``hub.stop()``)."""
    hub = FakeHub(tmp_path, config)
    for agent, path in (frozen or {}).items():
        hub.serve_frozen(agent, path)
    hub.welcome(learners=[pb.LearnerAssignment(agent_id=a, generation=1, version_floor=0, device="cpu", env_steps=0)
                          for a in learners])
    hub.grant(lambda worker, env: lineup(worker, env) if all(hub.versions(a) for a in learners) else None)
    return hub, run_node_in_thread(toy_node_options(tmp_path, hub, **node_options))


def train_steps(hub: FakeHub, agent: str) -> int:
    """The highest train step ``agent``'s learner reported (0: none)."""
    return max((t.train_step for m in hub.messages("metrics") for t in m.train if t.agent_id == agent), default=0)


def intake_reports(hub: FakeHub, agent: str) -> list:
    """Every ``IntakeStats`` of ``agent``'s learner received so far, in arrival order (cumulative drops)."""
    return [i for m in hub.messages("metrics") for i in m.intake if i.agent_id == agent]


def intake_drops(hub: FakeHub, agent: str, *, after: int = 0) -> dict[str, int]:
    """The chunk drops ``agent``'s learner intake reported last (``IntakeStats.dropped``, by reason), from a report
    received after the first ``after`` ones: waits for one (so a report sent before the chunks cannot answer)."""
    reports = wait_for(lambda: intake_reports(hub, agent)[after:], timeout=30)
    return dict(reports[-1].dropped)


def worker_counters(hub: FakeHub, worker: int = 0) -> dict[str, int]:
    """The cumulative counters worker ``worker`` reported last (``WorkerStats.counters``)."""
    stats = [w for m in hub.messages("metrics") for w in m.workers if w.worker == worker]
    return dict(stats[-1].counters) if stats else {}

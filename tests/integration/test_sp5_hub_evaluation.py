"""The lease protocol and the eval blobs over the real hub server (SP5 T5.1, spec 5.3; T2.3's ``build_server`` on
loopback with a ``BlobStore``; ``hub_kit.FakeNode`` clients): a small evaluation loop in a thread stands in for the
hub's main loop (``EvalLoop``: registration, sessions -> ``HubEvaluation.node_joined`` + ``Welcome.evaluators``,
``BlockLeaseRequest`` / ``BlockResult`` -> the service). No evaluator process."""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.registry import env_spec
from colosseum.core.run_dir import RunDir
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.evaluation import HubEvaluation
from colosseum.hub.server import (
    BlobStore,
    NodeMessageIn,
    RegisterIn,
    SessionClosed,
    SessionOpened,
    blob_file_name,
    build_server,
)
from colosseum.net.chunks import decode_weights
from colosseum.net.messages import BlockTask, PlayerRef, block_task_from_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import HubIdentity
from colosseum.players.registry import BotSpec
from game_helpers import make_match_result, make_test_config, write_ckpt_dir
from hub_kit import FakeNode, poll_until

CFG = make_test_config("turns")
SPEC = env_spec(CFG)
BOT = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))
OWNER = "tournament:t"


class EvalLoop:
    """The hub's main loop for these tests: a thread over the server inbox driving one ``HubEvaluation``."""

    def __init__(self, tmp_path: Path) -> None:
        self.identity = HubIdentity.load_or_create(Path(tmp_path) / "hub-id")
        self.run_dir = RunDir(Path(tmp_path) / "run")
        self.run_dir.root.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.run_dir)
        self.inbox: queue.SimpleQueue = queue.SimpleQueue()
        self.server, ports = build_server(inbox=self.inbox, identity=self.identity, insecure=True,
                                          listen=["127.0.0.1:0"], loopback_plain=False, blobs=self.blobs,
                                          max_message_mb=64)
        self.server.start()
        self.address = f"127.0.0.1:{ports['127.0.0.1:0']}"
        self.handles: dict[int, Any] = {}
        self.sessions: dict[int, str] = {}
        self.sink: list[tuple[str, str, Any]] = []
        self.ev = HubEvaluation(lease_sec=600, send=self._send, spec=SPEC)
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="eval-loop", daemon=True)
        self._thread.start()

    def node(self, *, slots: int = 1) -> FakeNode:
        node = FakeNode(self.address, token=self.identity.token, insecure=True)
        node.register()
        node.slots = slots
        return node

    def open(self, node: FakeNode, session_id: str = "s1", *, leases=()) -> FakeNode:
        return node.open(session_id, hello_fields={"slots": pb.Slots(evaluators=node.slots), "leases": list(leases)})

    def call(self, fn: Callable[[], Any], timeout: float = 10.0) -> Any:
        box = (fn, Future())
        self.inbox.put(box)
        return box[1].result(timeout)

    def stop(self) -> None:
        self.server.stop(None)
        self._stop.set()
        self._thread.join(5)

    def _send(self, number: int, msg: pb.HubMessage) -> None:
        handle = self.handles.get(number)
        if handle is not None:
            handle.send(msg)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self.inbox.get(timeout=0.05)
            except queue.Empty:
                self.ev.tick()
                continue
            try:
                self._dispatch(ev)
            except Exception as e:  # noqa: BLE001 - recorded for the test
                self.errors.append(f"{type(e).__name__}: {e}")

    def _dispatch(self, ev: object) -> None:
        if isinstance(ev, tuple):
            fn, reply = ev
            try:
                reply.set_result(fn())
            except Exception as e:  # noqa: BLE001 - handed to the caller
                reply.set_exception(e)
        elif isinstance(ev, RegisterIn):
            req = ev.request
            record, secret = self.identity.nodes.register(machine_id=req.machine_id, name=req.name, tags=[],
                                                          node_id=req.node_id or None)
            cert = self.identity.issue_node_certificate(record.node_id, req.public_key_pem.encode("ascii"))
            ev.reply.set_result(pb.RegisterReply(node_id=record.node_id, number=record.number, name=record.name,
                                                 certificate_pem=cert.decode("ascii"), node_secret=secret,
                                                 ca_pem=self.identity.ca_pem.decode("ascii")))
        elif isinstance(ev, SessionOpened):
            number = self.identity.nodes.get(ev.handle.node_id).number
            fresh = self.sessions.get(number) != ev.hello.session_id
            self.sessions[number] = ev.hello.session_id
            self.handles[number] = ev.handle
            join = self.ev.node_joined(number, slots=ev.hello.slots.evaluators, held_leases=list(ev.hello.leases),
                                       fresh=fresh)
            ev.handle.send(pb.HubMessage(welcome=pb.Welcome(node_number=number, run_id="R", config_json="{}",
                                                            evaluators=join.evaluators)))
            for msg in join.after_welcome:
                ev.handle.send(msg)
        elif isinstance(ev, SessionClosed):
            number = next((n for n, h in self.handles.items() if h is ev.handle), None)
            if number is not None:
                del self.handles[number]
                self.ev.session_closed(number)
        elif isinstance(ev, NodeMessageIn):
            number = self.identity.nodes.get(ev.node_id).number
            kind = ev.message.WhichOneof("m")
            if kind == "lease_request":
                self.ev.on_lease_request(number, ev.message.lease_request.free_slots)
            elif kind == "block_result":
                self.ev.on_block_result(number, ev.message.block_result)


@pytest.fixture
def loop(tmp_path):
    lp = EvalLoop(tmp_path)
    try:
        yield lp
    finally:
        lp.stop()
        assert not lp.errors


def _tasks(n, owner=OWNER):
    out = []
    for k in range(n):
        block = f"t/A/2p/a/rnd/{k}"
        lineups = tuple(Lineup("2p", [SeatAssignment("a", "ckpt_v1", False), SeatAssignment("rnd", "fixed", False)],
                               seed=100 + k, block=block, uid=f"{block}/r{r}") for r in range(2))
        out.append(BlockTask(block, "tournament", owner, lineups, (BOT,), 0, False, 1))
    return out


def _result(lease_id: str, spec: pb.BlockSpec) -> pb.BlockResult:
    task = block_task_from_proto(spec, num_envs=1)
    results = [make_match_result([(0, "a", "ckpt_v1", "player", 1), (1, "rnd", "fixed")], uid=lu.uid,
                                 block=task.block_id, seed=lu.seed, source="tournament") for lu in task.lineups]
    return pb.BlockResult(lease_id=lease_id, block_id=task.block_id, results=[result_to_proto(r) for r in results])


def _with_owner(loop: EvalLoop, tasks) -> None:
    def submit():
        loop.ev.add_owner(OWNER, lambda *a: loop.sink.append(a))
        loop.ev.submit(tasks)
    loop.call(submit)


def test_leases_and_results_travel_over_the_session(loop):
    node = loop.open(loop.node())
    try:
        (welcome,) = node.wait_for("welcome")
        assert welcome.evaluators == 0                                    # no evaluation work can exist yet
        _with_owner(loop, _tasks(2))
        (command,) = node.wait_for("command")
        assert (command.kind, command.evaluators) == ("set_evaluators", 1)
        node.send(lease_request=pb.BlockLeaseRequest(free_slots=1))
        (lease,) = node.wait_for("lease")
        assert len(lease.blocks) == 2 and lease.num_envs == 1
        for spec in lease.blocks:
            node.send(block_result=_result(lease.lease_id, spec))
        acks = node.wait_for("block_ack", 2)
        assert sorted(b for a in acks for b in a.block_ids) == sorted(s.block_id for s in lease.blocks)
        poll_until(lambda: len(loop.sink) == 2)
        assert {r.node for _k, _b, results in loop.sink for r in results} == {node.number}
    finally:
        node.close()


def test_eval_blobs_are_served_by_download_and_removed_on_unregister(loop, tmp_path, caplog):
    ckpt = write_ckpt_dir(tmp_path / "snaps", CFG, "agent_0", policy_version=3, seed=1)
    networks = CFG.get_agent_config("agent_0").networks.model_dump(mode="json", by_alias=True)
    loop.blobs.register_weights("t-ab-main", ckpt / "model.pt", agent_id="a/main", networks=networks,
                                roles=("player",))                # a tournament agent id is never a path (P-PF4-3)
    node = loop.node()
    data, _meta = node.download("eval", "a/main", "t-ab-main")
    state, meta = decode_weights(data, cap=len(data))
    want = read_weights_file(ckpt / "model.pt")
    assert set(state) == set(want)
    for key, value in want.items():
        np.testing.assert_array_equal(state[key], value)
    assert (meta.agent_id, meta.kind, meta.architecture, tuple(meta.roles)) == ("a/main", "fixed", networks,
                                                                              ("player",))
    frame = loop.blobs.dir / blob_file_name("eval", "a/main", "t-ab-main")
    assert frame.is_file() and "a/main" not in frame.name
    for kind, agent, ident in (("eval", "other", "t-ab-main"), ("eval", "a/main", "nope"),
                               ("eval", "a/main", "../x")):
        with pytest.raises(grpc.RpcError) as info:
            node.download(kind, agent, ident)
        assert info.value.code() == grpc.StatusCode.NOT_FOUND
    loop.blobs.unregister("t-ab-main")
    assert not frame.exists()
    with pytest.raises(grpc.RpcError) as info:
        node.download("eval", "a/main", "t-ab-main")
    assert info.value.code() == grpc.StatusCode.NOT_FOUND
    loop.blobs.register_weights("t-gone", tmp_path / "missing.pt", agent_id="a", networks=networks, roles=("player",))
    with caplog.at_level(logging.ERROR, logger="colosseum.hub.server"):
        for _ in range(2):
            with pytest.raises(grpc.RpcError):
                node.download("eval", "a", "t-gone")
    assert len([r for r in caplog.records if "t-gone" in r.getMessage()]) == 1
    loop.blobs.register_weights("t-again", ckpt / "model.pt", agent_id="a", networks=networks, roles=("player",))
    node.download("eval", "a", "t-again")
    assert (loop.blobs.dir / blob_file_name("eval", "a", "t-again")).is_file()
    BlobStore(loop.run_dir)                                               # a new hub process: stale eval frames go
    assert not (loop.blobs.dir / blob_file_name("eval", "a", "t-again")).exists()
    with pytest.raises(ValueError, match="blob id"):
        loop.blobs.register_weights("../x", ckpt / "model.pt", agent_id="a", networks={}, roles=())


def test_node_loss_re_leases_the_blocks_to_another_node(loop):
    a, b = loop.open(loop.node()), loop.open(loop.node())
    try:
        a.wait_for("welcome")
        b.wait_for("welcome")
        _with_owner(loop, _tasks(2))
        a.send(lease_request=pb.BlockLeaseRequest(free_slots=1))
        (lease,) = a.wait_for("lease")
        b.send(lease_request=pb.BlockLeaseRequest(free_slots=1))
        poll_until(lambda: loop.call(lambda: loop.ev.queue.waiting()) == {b.number: 1})
        loop.call(lambda: loop.ev.node_lost(a.number))                  # T2.5's loss check stands in here
        (again,) = b.wait_for("lease")
        assert [s.block_id for s in again.blocks] == [s.block_id for s in lease.blocks]
    finally:
        a.close()
        b.close()


def test_a_resumed_session_with_hello_leases_keeps_them(loop):
    a = loop.open(loop.node(slots=2))
    try:
        a.wait_for("welcome")
        _with_owner(loop, _tasks(4))
        a.send(lease_request=pb.BlockLeaseRequest(free_slots=1))
        (lease,) = a.wait_for("lease")
        a.cut()
        poll_until(lambda: loop.call(lambda: a.number not in loop.handles))
        loop.open(a, "s1", leases=[lease.lease_id])
        a.wait_for("welcome")
        poll_until(lambda: loop.call(lambda: a.number in loop.handles))
        assert loop.call(lambda: loop.ev.queue.leases_of(a.number)) == [lease.lease_id]
        assert [c for c in a.messages("command") if c.kind == "cancel_leases"] == []
        a.cut()
        poll_until(lambda: loop.call(lambda: a.number not in loop.handles))
        loop.open(a, "s1", leases=["bogus"])
        a.wait_for("welcome")
        cancel = poll_until(lambda: [c for c in a.messages("command") if c.kind == "cancel_leases"])
        assert list(cancel[0].lease_ids) == ["bogus"]
        kinds = [m.WhichOneof("m") for m in a.received]
        assert kinds.index("welcome") < kinds.index("command")              # right after the Welcome
        assert loop.call(lambda: (loop.ev.queue.leases_of(a.number), loop.ev.queue.queued())) == ([], 4)
    finally:
        a.close()

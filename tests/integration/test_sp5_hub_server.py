"""The hub's gRPC server (SP5 T2.3; spec 4.1, 3.2-3.4; R1, R13): a real server on loopback port 0 with a real
``HubIdentity`` under ``tmp_path``, ``hub_kit.MiniHubLoop`` as the main loop and ``hub_kit.FakeNode`` clients in the
test process (TLS unless a test says otherwise). Registration, the session stream (Hello / Welcome, refusals, inbox
order, converted results, cut and resume, loss, outbound overflow), blobs in parts (snapshots converted once,
registered blobs, path rules of P-PF2-9), uploads spooled and checked, admin calls, the relay placeholders."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import grpc
import numpy as np
import pytest
import torch
from cryptography import x509
from cryptography.x509.oid import NameOID

from colosseum.hub import server as hub_server
from colosseum.hub.server import (
    AdminIn,
    NodeMessageIn,
    RegisterIn,
    SessionClosed,
    SessionOpened,
    UploadIn,
    blob_file_name,
    blob_ident,
    hello_from_proto,
)
from colosseum.hub.state import CellKey, NodeHello
from colosseum.net.chunks import WeightsMeta, decode_weights, encode_weights
from colosseum.net.frame import payload_byte_cap
from colosseum.net.grpc_util import blob_digest, token_metadata
from colosseum.net.messages import PROTOCOL_VERSION, result_to_proto, results_from_batch
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
from hub_kit import MAX_MB, FakeNode, make_result, poll_until, serve


@pytest.fixture
def hub(tmp_path):
    served = serve(tmp_path)
    try:
        yield served
    finally:
        served.stop()


def _session(hub, session_id="s1", **kw) -> FakeNode:
    node = hub.registered_node()
    node.open(session_id, **kw)
    node.wait_for("welcome")
    return node


def _code(excinfo) -> grpc.StatusCode:
    return excinfo.value.code()


def _blob_name(kind: str, agent: str, ident: str, suffix: str = ".frame") -> str:
    """The documented blob file name: ``<kind>-<32 hex of blake2b(agent id)>-<ident><suffix>``."""
    return f"{kind}-{hashlib.blake2b(agent.encode(), digest_size=16).hexdigest()}-{ident}{suffix}"


# ---- registration ---------------------------------------------------------------------------------------------


def test_registration_issues_an_identity_and_refuses_a_wrong_or_missing_token(hub):
    node = hub.node()
    reply = node.register(machine_id="m1", name="laptop")
    assert len(reply.node_id) == 32 and int(reply.node_id, 16) >= 0 and reply.number >= 1
    cert = x509.load_pem_x509_certificate(reply.certificate_pem.encode())
    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == reply.node_id
    assert reply.node_secret and reply.ca_pem.encode() == hub.identity.ca_pem
    assert hub.identity.nodes.verify(reply.node_id, reply.node_secret).number == reply.number

    other = hub.node()
    wrong = type(hub.identity.token)(secret="A" * 43, ca_fingerprint=hub.identity.token.ca_fingerprint)
    with pytest.raises(grpc.RpcError) as e:
        other.register(token=wrong)
    assert _code(e) == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.RpcError) as e:
        other.register(token=None)
    assert _code(e) == grpc.StatusCode.UNAUTHENTICATED

    hub.identity.nodes.ban(reply.node_id)                            # bans its machine id "m1" too
    with pytest.raises(grpc.RpcError) as e:
        hub.node().register(machine_id="m1", name="again")
    assert _code(e) == grpc.StatusCode.PERMISSION_DENIED


# ---- the session stream ---------------------------------------------------------------------------------------


def test_a_session_starts_with_hello_and_gets_a_welcome(hub):
    node = hub.registered_node()
    node.open("s1", live_cells=[(0, 1, 7, 2)], last_acked=3)
    welcome = node.wait_for("welcome")[0]
    assert node.received[0].WhichOneof("m") == "welcome"
    assert welcome.node_number == node.number and welcome.run_id == "R" and welcome.config_json == "{}"
    opened = hub.loop.wait_events(SessionOpened)[0]
    assert isinstance(opened.hello, pb.Hello) and opened.hello.session_id == "s1"
    assert opened.handle.node_id == node.node_id and opened.handle.session_id == "s1"
    parsed = hello_from_proto(opened.hello)
    assert isinstance(parsed, NodeHello)
    assert parsed.live_cells == ((0, 1, 7, 2),) and parsed.last_acked_seq == 3


def test_hello_from_proto_keeps_hardware_and_slots_as_plain_values():
    hello = pb.Hello(node_id="n", session_id="s", tags=["gpu", "lab"], data_addresses=["10.0.0.2:7000"],
                     hardware=pb.Hardware(logical_cores=8, memory_bytes=17_000_000_000,
                                          gpus=[pb.Gpu(index=0, name="RTX", memory_bytes=12 << 30, kind="cuda")]),
                     slots=pb.Slots(workers=3, gpus=[0]), env_steps=5)
    parsed = hello_from_proto(hello)
    assert parsed.hardware["memory_bytes"] == 17_000_000_000 and parsed.hardware["logical_cores"] == 8
    assert parsed.hardware["gpus"] == [{"index": 0, "name": "RTX", "memory_bytes": 12 << 30, "kind": "cuda"}]
    assert parsed.slots["workers"] == 3 and parsed.slots["gpus"] == [0] and parsed.slots["no_learners"] is False
    assert parsed.tags == ("gpu", "lab") and parsed.data_addresses == ("10.0.0.2:7000",) and parsed.env_steps == 5


def test_a_session_without_hello_or_with_another_nodes_id_is_refused(hub):
    node = hub.registered_node()
    node.open("s1", first=pb.NodeMessage(heartbeat=pb.Heartbeat(env_steps=1)))
    assert node.wait_ended() == grpc.StatusCode.INVALID_ARGUMENT

    other_id = pb.Hello(node_id="0" * 32, session_id="s1", protocol_major=PROTOCOL_VERSION[0])
    node.open("s1", first=pb.NodeMessage(hello=other_id))
    assert node.wait_ended() == grpc.StatusCode.PERMISSION_DENIED

    old = node.hello("s1")
    old.protocol_major = PROTOCOL_VERSION[0] + 1
    node.open("s1", first=pb.NodeMessage(hello=old))
    assert node.wait_ended() == grpc.StatusCode.FAILED_PRECONDITION and "protocol" in node.details

    no_session = node.hello("")
    node.open("", first=pb.NodeMessage(hello=no_session))
    assert node.wait_ended() == grpc.StatusCode.INVALID_ARGUMENT
    assert hub.loop.call(lambda: [e for e in hub.loop.received if not isinstance(e, RegisterIn)]) == []


def test_a_session_with_a_wrong_node_secret_is_refused(hub):
    node = hub.registered_node()
    node.node_secret = "x" * 43
    node.open("s1")
    assert node.wait_ended() == grpc.StatusCode.UNAUTHENTICATED
    assert hub.loop.events(SessionOpened) == []


def test_node_messages_reach_the_inbox_in_order_with_converted_results(hub):
    node = _session(hub)
    results = [make_result(node=node.number, k=k) for k in range(3)]
    node.send(heartbeat=pb.Heartbeat(env_steps=10, counters={"lineup_fetch_failed": 2}))
    node.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=0)]))
    node.send(results=pb.ResultBatch(seq=1, results=[result_to_proto(r) for r in results]))
    node.send(metrics=pb.MetricsBatch(intake=[pb.IntakeStats(agent_id="a", generation=1, depth=3)]))
    node.send(version=pb.VersionAnnounce(info=pb.VersionInfo(agent_id="a", generation=1, version=4)))
    node.send(event=pb.Event(kind="warning", message="hi"))
    poll_until(lambda: len(hub.loop.events(NodeMessageIn)) >= 6)
    events = hub.loop.events(NodeMessageIn)
    assert [e.message.WhichOneof("m") for e in events] == ["heartbeat", "lineup_request", "results", "metrics",
                                                           "version", "event"]
    assert all(e.node_id == node.node_id and e.session_id == "s1" for e in events)
    assert len({e.serial for e in events}) == 1 and events[0].serial == hub.loop.events(SessionOpened)[0].handle.serial
    assert events[0].message.heartbeat.counters["lineup_fetch_failed"] == 2
    batch = events[2]
    assert batch.results == results and batch.error is None
    assert all(e.results is None and e.error is None for e in events if e is not batch)
    assert node.wait_for("ack")[0].seqs == [1]
    assert hub.loop.accounted == [r.uid for r in results] and hub.loop.dropped == []


def test_a_result_with_another_nodes_uid_is_dropped_by_the_intake_check(hub):
    node = _session(hub)
    stray = make_result(node=node.number + 1, k=1)                 # uid and node of another node
    node.send(results=pb.ResultBatch(seq=1, results=[result_to_proto(make_result(node=node.number)),
                                                     result_to_proto(stray)]))
    node.wait_for("ack")
    assert hub.loop.dropped == [(stray.uid, "invalid")] and len(hub.loop.accounted) == 1


def test_a_malformed_result_batch_reaches_the_inbox_as_an_error(hub, monkeypatch):
    def broken(batch):
        if batch.seq == 1:
            raise ValueError("nonsense outcome kind")
        return results_from_batch(batch)

    monkeypatch.setattr(hub_server, "results_from_batch", broken)
    node = _session(hub)
    node.send(results=pb.ResultBatch(seq=1, results=[result_to_proto(make_result(node=node.number))]))
    node.send(heartbeat=pb.Heartbeat(env_steps=1))
    poll_until(lambda: hub.loop.messages("heartbeat"))
    bad = hub.loop.messages("results")[0]
    assert bad.results is None and "ValueError" in bad.error and "nonsense outcome kind" in bad.error
    assert not node.ended.is_set()


def test_a_cut_stream_resumes_and_resent_batches_are_accounted_once(tmp_path):
    hub = serve(tmp_path, hold_acks=True, node_lost_after_sec=60.0)   # never lost here: the test is a resume
    try:
        node = _session(hub)
        n = node.number
        node.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=e) for e in (0, 1)]))
        grants = node.wait_for("grants")[0].grants
        epochs = {(g.worker, g.env): g.epoch for g in grants}
        batches = {seq: [make_result(node=n, env=seq % 2, epoch=epochs[(0, seq % 2)], k=10 * seq + i)
                         for i in range(2)] for seq in (1, 2, 3, 4)}

        def send(seq):
            node.send(results=pb.ResultBatch(seq=seq, results=[result_to_proto(r) for r in batches[seq]]))

        for seq in (1, 2, 3):
            send(seq)
        poll_until(lambda: len(hub.loop.messages("results")) >= 3)
        node.cut()
        closed = hub.loop.wait_events(SessionClosed)[0]
        first = hub.loop.events(SessionOpened)[0].handle
        assert closed.node_id == node.node_id and closed.handle is first and closed.serial == first.serial

        live = [(g.worker, g.env, g.epoch, 0) for g in grants]
        node.open("s1", live_cells=live, last_acked=0)
        node.wait_for("welcome")
        assert hub.loop.connects == ["new", "resumed"]
        node.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=w, env=e, epoch=ep, incarnation=i)
                                                         for w, e, ep, i in live]))
        regrants = node.wait_for("grants")[0].grants
        assert {(g.worker, g.env): g.epoch for g in regrants} == epochs           # the cells kept their epochs
        for seq in (1, 2, 3, 4):
            send(seq)
        poll_until(lambda: len(hub.loop.messages("results")) >= 7)
        hub.loop.release_acks()
        acks = node.wait_for("ack", 4)
        assert [s for a in acks for s in a.seqs] == [1, 2, 3, 4] and hub.loop.acked == [1, 2, 3, 4]
        uids = [r.uid for seq in (1, 2, 3, 4) for r in batches[seq]]
        assert sorted(hub.loop.accounted) == sorted(uids) and len(set(uids)) == 8
    finally:
        hub.stop()


def test_a_late_close_of_a_superseded_stream_changes_nothing(hub):
    node = _session(hub)
    first = hub.loop.wait_events(SessionOpened)[0].handle
    second_node = FakeNode(hub.address, token=hub.identity.token, ca_pem=hub.identity.ca_pem)
    second_node.node_id, second_node.node_secret = node.node_id, node.node_secret
    second_node.open("s1")                                       # a second stream of the same node and session
    second_node.wait_for("welcome")
    second = hub.loop.wait_events(SessionOpened, 2)[1].handle
    assert second.serial != first.serial and hub.loop.connects == ["new", "resumed"]
    assert node.wait_ended() == grpc.StatusCode.OK               # the older stream was closed as superseded
    hub.loop.wait_events(SessionClosed)
    assert hub.loop.handle(node.node_id) is second
    entry = hub.loop.call(lambda: hub.loop.registry.by_id(node.node_id))
    assert entry.connected and entry.serial == second.serial
    second_node.send(heartbeat=pb.Heartbeat(env_steps=5))
    poll_until(lambda: hub.loop.messages("heartbeat"))


def test_a_silent_node_is_lost_and_comes_back_with_new_epochs(tmp_path):
    hub = serve(tmp_path, node_lost_after_sec=0.5)
    try:
        node = _session(hub)
        node.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=0, env=e) for e in (0, 1)]))
        grants = node.wait_for("grants")[0].grants
        old = {g.epoch for g in grants}
        poll_until(lambda: hub.loop.call(lambda: hub.loop.registry.by_id(node.node_id).state) == "lost")

        def no_more_losses():                                    # the returned node must not be lost again
            hub.loop.node_lost_after_sec = 60.0
        hub.loop.call(no_more_losses)
        assert hub.loop.call(lambda: [c for c in hub.loop.cells.live() if c.key.node == node.number]) == []
        node.wait_ended()                                        # the hub closed the lost node's stream
        node.open("s2", live_cells=[(g.worker, g.env, g.epoch, 0) for g in grants])
        node.wait_for("welcome")
        assert hub.loop.connects == ["new", "returned"]
        node.send(lineup_request=pb.LineupRequest(cells=[pb.CellRequest(worker=g.worker, env=g.env, epoch=g.epoch)
                                                         for g in grants]))
        new = {g.epoch for g in node.wait_for("grants")[0].grants}
        assert min(new) > max(old)
        assert hub.loop.call(lambda: sorted(c.key for c in hub.loop.cells.live())) == [
            CellKey(node.number, 0, 0), CellKey(node.number, 0, 1)]
    finally:
        hub.stop()


def test_an_outbound_overflow_closes_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_server, "OUTBOUND_QUEUE_SIZE", 4)
    hub = serve(tmp_path)
    try:
        node = hub.registered_node()
        node.open("s1", read=False)                              # never reads its responses
        handle = hub.loop.wait_events(SessionOpened)[0].handle
        hub.loop.flood(handle, 50)
        assert handle.closed_reason == "outbound overflow"
        closed = hub.loop.wait_events(SessionClosed)[0]
        assert closed.reason == "outbound overflow" and closed.handle is handle
        node.cut()
        fresh = hub.registered_node(machine_id="m2", name="fresh")
        fresh.open("s9")
        fresh.wait_for("welcome")
    finally:
        hub.stop()


# ---- blobs ----------------------------------------------------------------------------------------------------


def test_snapshots_are_converted_once_and_served_in_parts(hub, monkeypatch):
    ckpt = hub.run_dir.checkpoints / "a" / "ckpt_v5"
    ckpt.mkdir(parents=True)
    state = {"w": torch.randn(768, 1024), "b": torch.arange(10, dtype=torch.int64)}   # 3 MiB of float32
    torch.save(state, ckpt / "model.pt")
    calls = []
    real = hub_server.read_weights_file

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(hub_server, "read_weights_file", counting)
    node = hub.registered_node()
    out = [None, None]

    def fetch(i):
        out[i] = node.download("snapshot", "a", "ckpt_v5")

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    (d0, _), (d1, _) = out
    assert d0 == d1 and len(d0) > 3 * 2**20
    tree, meta = decode_weights(d0, cap=1 << 30)
    assert meta == WeightsMeta(agent_id="a", kind="snapshot", generation=0, policy_version=5, checkpoint_id="ckpt_v5")
    assert np.array_equal(tree["w"], state["w"].numpy()) and np.array_equal(tree["b"], state["b"].numpy())
    assert sorted(p.name for p in (hub.run_dir.root / "blobs").iterdir()) == [_blob_name("snapshot", "a", "ckpt_v5")]
    assert len(calls) == 1
    with pytest.raises(grpc.RpcError) as e:
        node.download("snapshot", "a", "ckpt_v6")
    assert _code(e) == grpc.StatusCode.NOT_FOUND


def test_registered_fixed_blobs_are_served_by_ident(hub):
    frame = encode_weights({"w": np.arange(12, dtype=np.float32)}, WeightsMeta("bot", "fixed", 0, 0))
    ident = blob_ident("fixed", "bot", frame)
    assert ident != blob_ident("fixed", "bot2", frame) and ident != blob_ident("teacher", "bot", frame)
    hub.blobs.register_fixed(ident, frame, kind="fixed", agent_id="bot")
    node = hub.registered_node()
    assert node.download("fixed", "bot", ident)[0] == frame
    for kind, agent, bad in (("fixed", "bot", "0" * 16), ("teacher", "bot", ident), ("fixed", "other", ident),
                             ("latest", "bot", ident)):
        with pytest.raises(grpc.RpcError) as e:
            node.download(kind, agent, bad)
        assert _code(e) == grpc.StatusCode.NOT_FOUND


def test_blob_paths_are_validated_before_any_path_is_formed(hub):
    root = hub.run_dir.root
    (root / "secret.frame").write_bytes(b"s")
    blobs = hub.blobs
    for kind, agent, ident in (("snapshot", "../a", "ckpt_v1"), ("snapshot", "a", "../../x"),
                               ("snapshot", "a", "ckpt_v1/../ckpt_v2"), ("fixed", "..", "x"),
                               ("fixed", "bot", "../secret"), ("fixed", "b/ot", "id"), ("snapshot", "", "ckpt_v1"),
                               ("fixed", "bot", ""), ("nonsense", "bot", "id")):
        assert blobs.frame_path(kind, agent, ident) is None
    with pytest.raises(ValueError):
        blobs.register_fixed("../x", b"f", kind="fixed", agent_id="bot")
    with pytest.raises(ValueError):
        blobs.register_fixed("x", b"f", kind="fixed", agent_id="../bot")
    with pytest.raises(ValueError):
        blobs.register_fixed("x", b"f", kind="snapshot", agent_id="bot")
    assert not (root / "blobs").exists() or list((root / "blobs").iterdir()) == []
    with pytest.raises(ValueError):                    # A28: no trainer_state / recovery blobs any more (T4.1)
        blobs.register_fixed("state1", b"raw", kind="trainer_state", agent_id="a")
    for agent, ident in (("../a", "resume"), ("a", "../resume"), ("a", "ckpt_v1/../x"), ("a", "latest"),
                         ("", "resume")):
        assert blobs.start_path(agent, ident) is None
    assert blobs.start_path("a", "ckpt_v3") == root / "blobs" / _blob_name("start", "a", "ckpt_v3")
    assert blobs.parked_path("a") == root / "blobs" / _blob_name("latest", "a", "parked")
    assert blobs.parked_path("../a") is None


def test_blob_file_name_is_the_public_rule_every_blob_path_uses(hub):
    """``blob_file_name`` is the documented rule (``BlobStore`` names its files through it); a kind with ``-``
    is refused (it would break injectivity)."""
    assert blob_file_name("fixed", "a-b", "c") == _blob_name("fixed", "a-b", "c")
    assert blob_file_name("start", "a", "resume") == _blob_name("start", "a", "resume")
    assert blob_file_name("snapshot", "agent", "ckpt_v7") == _blob_name("snapshot", "agent", "ckpt_v7")
    with pytest.raises(ValueError, match="'-'"):
        blob_file_name("bad-kind", "a", "c")
    hub.blobs.register_fixed("x1", b"frame", kind="init", agent_id="a")
    assert hub.blobs.frame_path("init", "a", "x1") == hub.run_dir.root / "blobs" / blob_file_name("init", "a", "x1")


def test_agent_ids_follow_the_config_rule_and_never_name_blob_files(hub):
    """Agent ids are checked with ``check_agent_id`` (no length limit of the hub's own: a 200-char config id is
    served), blob file names stay bounded and never contain the agent id, and triples that would join to the same
    ``kind-agent-ident`` string get different files."""
    long_id = "L" * 200
    ckpt = hub.run_dir.checkpoints / long_id / "ckpt_v3"
    ckpt.mkdir(parents=True)
    torch.save({"w": torch.arange(4, dtype=torch.float32)}, ckpt / "model.pt")
    node = hub.registered_node()
    data, _ = node.download("snapshot", long_id, "ckpt_v3")
    tree, meta = decode_weights(data, cap=1 << 30)
    assert meta.agent_id == long_id and meta.policy_version == 3 and np.array_equal(tree["w"], np.arange(4.0))

    hub.blobs.register_fixed("b-c", b"one", kind="fixed", agent_id="a")
    hub.blobs.register_fixed("c", b"two", kind="fixed", agent_id="a-b")
    first, second = hub.blobs.frame_path("fixed", "a", "b-c"), hub.blobs.frame_path("fixed", "a-b", "c")
    assert first != second and first.read_bytes() == b"one" and second.read_bytes() == b"two"
    assert node.download("fixed", "a", "b-c")[0] == b"one" and node.download("fixed", "a-b", "c")[0] == b"two"

    hub.blobs.register_fixed("x", b"long", kind="teacher", agent_id=long_id)
    names = [p.name for p in (hub.run_dir.root / "blobs").iterdir()]
    assert all(long_id not in n and "-a-" not in n and len(n) <= 13 + 1 + 32 + 1 + 160 + 6 for n in names)
    assert sorted(names) == sorted([_blob_name("snapshot", long_id, "ckpt_v3"), _blob_name("fixed", "a", "b-c"),
                                    _blob_name("fixed", "a-b", "c"), _blob_name("teacher", long_id, "x")])

    for agent in ("ratings", "a.b", "x" * 300):                  # refused by the config rule / beyond NAME_MAX
        assert hub.blobs.frame_path("snapshot", agent, "ckpt_v3") is None
    with pytest.raises(ValueError):
        hub.blobs.register_fixed("x", b"f", kind="fixed", agent_id="a.b")
    with pytest.raises(grpc.RpcError) as e:
        node.download("snapshot", "x" * 300, "ckpt_v3")
    assert _code(e) == grpc.StatusCode.NOT_FOUND


def test_missing_snapshots_leave_no_conversion_lock(hub):
    for v in range(50):
        assert hub.blobs.frame_path("snapshot", "a", f"ckpt_v{v}") is None
    assert hub.blobs._convert_locks == {}


# ---- uploads --------------------------------------------------------------------------------------------------


def _header(data: bytes, **kw) -> pb.UploadHeader:
    fields = dict(kind="checkpoint", agent_id="a", generation=1, version=20, total_size=len(data),
                  digest=blob_digest(data), request_id="r1")
    fields.update(kw)
    return pb.UploadHeader(**fields)


def test_uploads_are_spooled_checked_and_handed_to_the_main_loop(hub):
    node = hub.registered_node()
    data = os.urandom(5 * 2**19)                                  # 2.5 MiB
    reply = node.upload(_header(data, request_id="../../../escape"), data)
    assert reply.accepted and reply.checkpoint_id == "ckpt_v20"
    header, got, path = hub.loop.uploads[0]
    assert got == data and path.parent == hub.run_dir.root / ".incoming" and "escape" not in path.name
    assert header.request_id == "../../../escape" and header.version == 20
    assert (os.stat(path.parent).st_mode & 0o777) == 0o700
    assert len(hub.loop.events(UploadIn)) == 1

    cap = payload_byte_cap(MAX_MB)
    for header, payload, reason in ((_header(data, digest="0" * 32), data, "digest"),
                                    (_header(data, total_size=cap + 1), data, "too_large"),
                                    (None, data, "header"),
                                    (_header(data, total_size=len(data) + 5), data, "size"),
                                    (_header(data, kind="nonsense"), data, "kind")):
        reply = node.upload(header, payload)
        assert not reply.accepted and reply.reason == reason
    assert len(hub.loop.events(UploadIn)) == 1
    assert list((hub.run_dir.root / ".incoming").iterdir()) == []


def test_an_upload_the_main_loop_does_not_answer_times_out_and_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_server, "UPLOAD_REPLY_TIMEOUT_SEC", 0.3)
    hub = serve(tmp_path, answer_uploads=False)                  # the loop never resolves the reply
    try:
        node = hub.registered_node()
        data = b"payload"
        reply = node.upload(_header(data), data)
        assert not reply.accepted and reply.reason == "timeout"
        assert len(hub.loop.events(UploadIn)) == 1
        assert list((hub.run_dir.root / ".incoming").iterdir()) == []
    finally:
        hub.stop()


def test_an_upload_the_main_loop_claimed_is_answered_after_the_timeout(tmp_path, monkeypatch):
    """SP5 T4.2 (the claim race of T2.5 (a2)): a main loop that took the spool file (claimed) but answers after the
    servicer's timeout is waited for, so the node learns the real outcome instead of ``timeout`` (and does not upload
    a frame again that the hub stored)."""
    monkeypatch.setattr(hub_server, "UPLOAD_REPLY_TIMEOUT_SEC", 0.3)

    def slow(header, data):                                      # the loop removed the spool before this
        time.sleep(0.8)
        return pb.UploadReply(accepted=True, checkpoint_id=f"ckpt_v{header.version}")

    hub = serve(tmp_path, upload_reply=slow)
    try:
        node = hub.registered_node()
        data = b"payload"
        reply = node.upload(_header(data), data)
        assert reply.accepted and reply.checkpoint_id == "ckpt_v20", reply
        assert len(hub.loop.uploads) == 1
        assert list((hub.run_dir.root / ".incoming").iterdir()) == []
    finally:
        hub.stop()


@pytest.mark.parametrize("method", ["Register", "Upload", "Admin"])
def test_server_stop_ends_a_handler_that_waits_for_the_main_loop(tmp_path, monkeypatch, method):
    """A Register / Upload / Admin call the main loop never answers (the loop stopped draining the inbox) ends
    within about a poll interval of ``server.stop``, not after its 30-60 s reply timeout; an upload's spool file is
    removed."""
    ended: dict[str, float] = {}
    real = getattr(hub_server.HubServicer, method)

    def watched(self, request, context):
        try:
            return real(self, request, context)
        finally:
            ended[method] = time.monotonic()

    monkeypatch.setattr(hub_server.HubServicer, method, watched)  # before serve(): the server binds the method
    hub = serve(tmp_path)
    node = hub.registered_node() if method != "Register" else hub.node()
    hub.loop.stop()                                              # nothing answers from now on
    errors: list = []

    def call():
        try:
            if method == "Register":
                node.register(machine_id="m2")
            elif method == "Upload":
                node.upload(_header(b"payload"), b"payload")
            else:
                node.stub().Admin(pb.AdminRequest(command="hub.status"), metadata=token_metadata(hub.identity.token),
                                  timeout=30)
        except grpc.RpcError as e:
            errors.append(e.code())

    client = threading.Thread(target=call, daemon=True)
    try:
        client.start()
        poll_until(lambda: hub.inbox.qsize() >= 1)               # the handler posted its event and waits
        time.sleep(0.2)
        assert method not in ended
    finally:
        start = time.monotonic()
        hub.stop()
    poll_until(lambda: method in ended, timeout=10.0)
    assert ended[method] - start <= 3.0
    client.join(10)
    assert not client.is_alive()
    incoming = hub.run_dir.root / ".incoming"
    assert not incoming.exists() or list(incoming.iterdir()) == []


class _LoopTimeout(TimeoutError):
    """A ``TimeoutError`` subclass the main loop may raise."""


@pytest.mark.parametrize("method", ["Register", "Upload", "Admin"])
def test_a_timeout_error_set_by_the_main_loop_maps_like_any_exception_at_once(tmp_path, method):
    """``concurrent.futures.TimeoutError`` is the builtin ``TimeoutError`` (3.11+): a reply the main loop fails
    with ``TimeoutError`` (or a subclass) ends the call promptly with the handler's ordinary exception mapping
    (Register / Admin: INTERNAL, Upload: ``error: <type>``), not with a busy wait until the reply timeout."""
    hub = serve(tmp_path)
    node = hub.registered_node() if method != "Register" else hub.node()
    hub.loop.stop()                                              # this test answers the inbox itself
    out: dict = {}

    def call():
        try:
            if method == "Register":
                node.register(machine_id="m2")
            elif method == "Upload":
                out["reply"] = node.upload(_header(b"payload"), b"payload")
            else:
                node.stub().Admin(pb.AdminRequest(command="hub.status"), metadata=token_metadata(hub.identity.token),
                                  timeout=30)
        except grpc.RpcError as e:
            out["code"], out["details"] = e.code(), e.details()

    client = threading.Thread(target=call, daemon=True)
    try:
        start = time.monotonic()
        client.start()
        expected = {"Register": RegisterIn, "Upload": UploadIn, "Admin": AdminIn}[method]
        while True:
            ev = hub.inbox.get(timeout=10)
            if isinstance(ev, expected):
                break
        ev.reply.set_exception(_LoopTimeout("main loop") if method == "Upload" else TimeoutError("main loop"))
        client.join(10)
        assert not client.is_alive() and time.monotonic() - start <= 5.0
    finally:
        hub.stop()
    if method == "Upload":
        assert not out["reply"].accepted and out["reply"].reason == "error: _LoopTimeout"
        incoming = hub.run_dir.root / ".incoming"
        assert not incoming.exists() or list(incoming.iterdir()) == []
    else:
        assert out["code"] == grpc.StatusCode.INTERNAL and "TimeoutError" in out["details"]


# ---- admin, relay ---------------------------------------------------------------------------------------------


def test_admin_calls_reach_the_main_loop_with_the_token(hub):
    stub = pb_grpc.HubStub(hub.node().channel())
    reply = stub.Admin(pb.AdminRequest(command="hub.status", args_json='{"x": 1}'),
                       metadata=token_metadata(hub.identity.token), timeout=10)
    assert reply.code == 0 and reply.text == "ok hub.status" and json.loads(reply.data_json) == {"x": 1}
    replies = list(stub.AdminStream(pb.AdminRequest(command="t.watch"), metadata=token_metadata(hub.identity.token),
                                    timeout=10))
    assert [r.done for r in replies] == [False, True]
    assert [e.command for e in hub.loop.events(AdminIn)] == ["hub.status", "t.watch"]
    with pytest.raises(grpc.RpcError) as e:
        stub.Admin(pb.AdminRequest(command="hub.status", args_json="[not json"),
                   metadata=token_metadata(hub.identity.token), timeout=10)
    assert _code(e) == grpc.StatusCode.INVALID_ARGUMENT
    with pytest.raises(grpc.RpcError) as e:
        stub.Admin(pb.AdminRequest(command="hub.status"), timeout=10)
    assert _code(e) == grpc.StatusCode.UNAUTHENTICATED


def test_relay_methods_are_not_implemented_yet(hub):
    node = hub.registered_node()
    stub = node.stub()
    with pytest.raises(grpc.RpcError) as e:
        stub.RelaySend(iter([pb.ChunkFrame(agent_id="a")]), metadata=node._node_md(), timeout=10)
    assert _code(e) == grpc.StatusCode.UNIMPLEMENTED
    with pytest.raises(grpc.RpcError) as e:
        list(stub.RelayInbox(iter([pb.RelayCredit(frames=1)]), metadata=node._node_md(), timeout=10))
    assert _code(e) == grpc.StatusCode.UNIMPLEMENTED


def test_build_server_reports_a_taken_port(tmp_path):
    hub = serve(tmp_path)
    try:
        with pytest.raises(OSError, match=f"127.0.0.1:{hub.port}"):
            hub_server.build_server(inbox=hub.inbox, identity=hub.identity, insecure=False,
                                    listen=[f"127.0.0.1:{hub.port}"], loopback_plain=False, blobs=hub.blobs,
                                    max_message_mb=4)
        server, ports = hub_server.build_server(inbox=hub.inbox, identity=hub.identity, insecure=True, listen=[],
                                                loopback_plain=True, blobs=hub.blobs, max_message_mb=4)
        assert set(ports) == {hub_server.LOOPBACK_KEY} and ports[hub_server.LOOPBACK_KEY] > 0
        server.stop(None)
    finally:
        hub.stop()

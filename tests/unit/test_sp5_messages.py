"""Protocol v1 (SP5 T1.2, spec blocks 4.1 and 4.9): the generated code matches the .proto, every SP4 match-record and
lineup variant round-trips through protobuf bytes (None stays unset, int64 versions and counters), version
announcements round-trip, the services and messages of the contract exist, every addition of amendment A1 has its
field number and round-trips, and the protocol version rule."""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.net import messages
from colosseum.net.messages import (
    EVENT_KINDS,
    PROTOCOL_VERSION,
    VersionInfo,
    lineup_from_proto,
    lineup_to_proto,
    protocol_compatible,
    result_from_proto,
    result_to_proto,
    results_from_batch,
    results_to_batch,
    seat_result_from_proto,
    seat_result_to_proto,
    version_from_proto,
    version_to_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

PROTO = Path(messages.__file__).resolve().parent / "proto" / "colosseum_v1.proto"
BIG = 2**62 + 5                       # an int64 that int32 cannot hold
INT64, INT32, STRING, BOOL, MESSAGE = 3, 5, 9, 8, 11   # FieldDescriptor.TYPE_*


def _seat(seat: int, **kw) -> SeatResult:
    return SeatResult(seat, "player", seat, kw.pop("agent", "main"), kw.pop("network", "latest"),
                      kw.pop("reward", 1.0 - 2 * seat), **kw)


RESULTS = [
    # SP3 shape: only the positional fields
    MatchResult("m0", "2p", "wdl", [_seat(0), _seat(1)], [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 9),
    # SP4 training record: uid, versions, env_steps stamped, one eliminated seat, a snapshot opponent
    MatchResult("w0_e3_ep7", "2p", "wdl",
                [_seat(0, version_start=40, version_end=41, source="owner"),
                 _seat(1, network="ckpt_v20", version_start=20, version_end=20, eliminated_step=5,
                       source="snapshots")],
                [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 12, uid="RUN/train/w0_e3_ep7", seed=123456,
                env_steps=BIG),
    # SP5 training record: node, generations, epoch in the match id
    MatchResult("w1_e0_g4096_ep2", "2p", "wdl",
                [_seat(0, version_start=BIG, version_end=BIG + 1, generation=3), _seat(1, network="fixed",
                                                                                       agent="random")],
                [TeamResult(0, 1.5, 0.0), TeamResult(1, 1.5, 0.0)], 30, uid="RUN/train/n2/w1_e0_g4096_ep2",
                seed=0, env_steps=0, node=2),
    # tournament block, FFA rank outcome with ties, a draw-like score
    MatchResult("t_A_0_7_r1", "ffa4", "rank", [_seat(i, network="fixed", agent=f"b{i}", reward=0.25) for i in range(4)],
                [TeamResult(i, [1.0, 2.5, 2.5, 4.0][i], float(i)) for i in range(4)], 50, uid="T1/A/0/7/r1",
                source="tournament", seed=2**31, block="A/0/7"),
    # pulse: a snapshot's env_steps, a solo score layout with one team
    MatchResult("p0", "solo", "score", [_seat(0, network="ckpt_v100", version_start=100, version_end=100)],
                [TeamResult(0, 1.0, 17.5)], 100, uid="RUN/pulse/x", source="pulse", env_steps=4000, block="P/3"),
    # train's local node is node 0 (amendment A9); generation 0 is a parked version's: both stay 0, never None
    MatchResult("w0_e0_g1_ep0", "2p", "wdl", [_seat(0, version_start=0, version_end=0, generation=0), _seat(1)],
                [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 4, uid="RUN/train/n0/w0_e0_g1_ep0", node=0),
]


def test_the_new_record_fields_default_to_none_and_come_last():
    assert Lineup("2p", [SeatAssignment("a")]).epoch is None
    assert SeatResult(0, "player", 0, "a", "latest", 1.0).generation is None
    assert MatchResult("m", "2p", "wdl", [], [], 1).node is None
    assert [f.name for f in dataclasses.fields(Lineup)][-1] == "epoch"
    assert [f.name for f in dataclasses.fields(SeatResult)][-1] == "generation"
    assert [f.name for f in dataclasses.fields(MatchResult)][-1] == "node"


@pytest.mark.parametrize("result", RESULTS, ids=lambda r: r.match_id)
def test_every_match_record_variant_round_trips_through_bytes(result):
    wire = result_to_proto(result).SerializeToString()
    back = result_from_proto(pb.MatchResultProto.FromString(wire))
    assert back == result
    assert dataclasses.asdict(back) == dataclasses.asdict(result)


def test_a_seat_result_round_trips_on_its_own():
    seat = RESULTS[2].seats[0]
    assert seat_result_from_proto(pb.SeatResultProto.FromString(seat_result_to_proto(seat).SerializeToString())) == seat


def test_unset_optional_fields_stay_none_and_zero_stays_zero():
    proto = result_to_proto(RESULTS[0])
    for field in ("seed", "env_steps", "block", "node"):
        assert not proto.HasField(field)
    for field in ("eliminated_step", "version_start", "version_end", "generation"):
        assert not proto.seats[0].HasField(field)
    zero = result_to_proto(RESULTS[2])
    assert zero.HasField("seed") and zero.seed == 0 and zero.HasField("env_steps") and zero.env_steps == 0
    local = pb.MatchResultProto.FromString(result_to_proto(RESULTS[5]).SerializeToString())
    assert local.HasField("node") and local.node == 0
    assert local.seats[0].HasField("generation") and local.seats[0].generation == 0
    back = result_from_proto(local)
    assert back.node == 0 and back.seats[0].generation == 0 and back.seats[1].generation is None


def test_a_batch_of_256_results_round_trips_in_order():
    results = [dataclasses.replace(RESULTS[1], match_id=f"w0_e0_ep{i}", uid=f"RUN/train/w0_e0_ep{i}")
               for i in range(256)]
    wire = results_to_batch(7, results).SerializeToString()
    batch = pb.ResultBatch.FromString(wire)
    assert batch.seq == 7 and results_from_batch(batch) == results


LINEUPS = [
    Lineup("2p", [SeatAssignment("main"), SeatAssignment("main", "ckpt_v20", False, "snapshots")]),
    Lineup("2p", [SeatAssignment("a", source="owner"), SeatAssignment("random", "fixed", False, "anchors")],
           seed=99, block="A/1/2", uid="T/A/1/2/r0"),
    Lineup("2p", [SeatAssignment("a", source="owner"), SeatAssignment("b", source="rivals")], epoch=BIG),
    Lineup("2p", [SeatAssignment("a")], seed=0, epoch=0),
]


@pytest.mark.parametrize("lineup", LINEUPS)
def test_every_lineup_variant_round_trips_through_bytes(lineup):
    back = lineup_from_proto(pb.LineupProto.FromString(lineup_to_proto(lineup).SerializeToString()))
    assert back == lineup


def test_a_grant_carries_a_lineup_with_its_epoch_and_ttl():
    grant = pb.Grant(worker=1, env=3, epoch=1025, ttl_sec=30.0, lineup=lineup_to_proto(LINEUPS[2]))
    back = pb.HubMessage.FromString(pb.HubMessage(grants=pb.LineupGrants(grants=[grant])).SerializeToString())
    assert back.WhichOneof("m") == "grants"
    assert lineup_from_proto(back.grants.grants[0].lineup).epoch == BIG and back.grants.grants[0].ttl_sec == 30.0


INFO = VersionInfo("main", 3, BIG, 50 * 2**20, "ab" * 16, 2, ("10.0.0.5:47102", "[::1]:47102"), False, True)


def test_version_announcements_round_trip():
    back = version_from_proto(pb.VersionInfo.FromString(version_to_proto(INFO).SerializeToString()))
    assert back == INFO and isinstance(back.addresses, tuple)


def test_version_info_round_trips_through_json_dicts():
    d = INFO.to_dict()
    assert d["addresses"] == ["10.0.0.5:47102", "[::1]:47102"]
    assert VersionInfo.from_dict(json.loads(json.dumps(d))) == INFO
    minimal = {"agent_id": "a", "generation": 1, "version": 0, "size": 10, "digest": "d" * 32, "node": -1}
    assert VersionInfo.from_dict(minimal) == VersionInfo("a", 1, 0, 10, "d" * 32, -1)


def test_values_a_field_cannot_hold_are_refused():
    with pytest.raises(ValueError):
        result_to_proto(dataclasses.replace(RESULTS[0], seed=2**64))


def test_the_protocol_version_rule():
    assert PROTOCOL_VERSION == (1, 0)
    assert protocol_compatible(1, 0) and protocol_compatible(1, 7)
    assert not protocol_compatible(2, 0) and not protocol_compatible(0, 9)


def test_the_event_kinds_are_the_documented_ones():
    assert EVENT_KINDS == ("proc_died", "child_error", "slot_disabled", "learner_failed", "warning", "run_stopped",
                           "node_stopped", "node_leaving", "code_updated", "code_update_failed")


def test_the_contract_services_and_messages_exist():
    services = {s.name: [m.name for m in s.methods] for s in pb.DESCRIPTOR.services_by_name.values()}
    assert services == {
        "Bootstrap": ["Info", "Bundle"],
        "Hub": ["Register", "Session", "Upload", "Download", "RelaySend", "RelayInbox", "Admin", "AdminStream"],
        "NodeData": ["Chunks", "GetLatest"],
    }
    assert pb.DESCRIPTOR.package == "colosseum.v1"
    oneofs = {name: [f.name for f in pb.DESCRIPTOR.message_types_by_name[name].oneofs_by_name[o].fields]
              for name, o in (("NodeMessage", "m"), ("HubMessage", "m"))}
    assert oneofs == {
        "NodeMessage": ["hello", "heartbeat", "lineup_request", "results", "metrics", "version", "event",
                        "lease_request", "block_result", "addresses"],
        "HubMessage": ["welcome", "grants", "ack", "command", "announce", "lease", "block_ack"],
    }
    int64_fields = [("VersionInfo", "version"), ("Hello", "env_steps"), ("Heartbeat", "env_steps"),
                    ("SeatResultProto", "version_start"), ("MatchResultProto", "env_steps"), ("Grant", "epoch"),
                    ("ResultBatch", "seq"), ("LearnerAssignment", "version_floor"), ("BlobPart", "total_size")]
    for message, field in int64_fields:
        assert pb.DESCRIPTOR.message_types_by_name[message].fields_by_name[field].type == INT64, (message, field)
    assert hasattr(pb_grpc, "HubStub") and hasattr(pb_grpc, "add_NodeDataServicer_to_server")
    assert hasattr(pb_grpc, "BootstrapStub") and hasattr(pb_grpc, "NodeDataStub")
    assert hasattr(pb_grpc, "add_HubServicer_to_server") and hasattr(pb_grpc, "add_BootstrapServicer_to_server")


def test_the_bootstrap_messages_are_numbered_in_declaration_order():
    """The self-contained loader (T6.1) has its own wire codec for these three messages (amendment A1)."""
    for name in ("BootstrapInfo", "BundleRequest", "BlobPart"):
        fields = pb.DESCRIPTOR.message_types_by_name[name].fields
        assert [f.number for f in fields] == list(range(1, len(fields) + 1)), name


# Amendment A1: (message, field, number, type, repeated/map, a value for the round trip)
A1_ROWS = [
    ("Heartbeat", "counters", 6, MESSAGE, {"lineup_fetch_failed": 3, "results_sent": BIG}),
    ("Heartbeat", "paths", 7, MESSAGE, [pb.PathState(node=2, path="relay"), pb.PathState(node=3, path="direct")]),
    ("PathState", "node", 1, INT32, 4),
    ("PathState", "path", 2, STRING, "direct"),
    ("DataAddresses", "data_addresses", 1, STRING, ["10.0.0.5:47102", "[::1]:47102"]),
    ("Welcome", "banned_node_ids", 12, STRING, ["n-dead", "n-bad"]),
    ("Welcome", "no_workers", 13, BOOL, True),
    ("Welcome", "agents_version", 14, INT64, BIG),
    ("Announce", "banned_node_ids", 6, STRING, ["n-bad"]),
    ("Announce", "agents_version", 7, INT64, BIG),
    ("Announce", "config_json", 8, STRING, '{"agents": {}}'),
    ("Announce", "fixed", 9, MESSAGE, [pb.FixedBlob(agent_id="greedy", blob_id="b1", kind="fixed")]),
    ("Event", "generation", 9, INT64, BIG),
    ("Hello", "leases", 16, STRING, ["L1", "L2"]),
    ("Hello", "ssh_id", 17, STRING, "ssh-7"),
    ("BlockLease", "num_envs", 4, INT32, 8),
    ("BlockResult", "error_kind", 6, STRING, "missing_weights"),
    ("BlockAck", "lease_id", 1, STRING, "L1"),
    ("BlockAck", "block_ids", 2, STRING, ["A/0/1", "A/0/2"]),
    ("Command", "lease_ids", 6, STRING, ["L1"]),
    ("LearnerAssignment", "start_env_steps", 8, INT64, BIG),
]


@pytest.mark.parametrize("message, field, number, ftype, value", A1_ROWS, ids=lambda v: v if isinstance(v, str) else "")
def test_every_amendment_a1_field_has_its_number_and_round_trips(message, field, number, ftype, value):
    descriptor = pb.DESCRIPTOR.message_types_by_name[message].fields_by_name[field]
    assert (descriptor.number, descriptor.type) == (number, ftype)
    cls = getattr(pb, message)
    msg = cls()
    if isinstance(value, dict):
        getattr(msg, field).update(value)
    elif isinstance(value, list):
        getattr(msg, field).extend(value)
    else:
        setattr(msg, field, value)
    back = cls.FromString(msg.SerializeToString())
    got = getattr(back, field)
    if isinstance(value, dict):
        assert dict(got) == value
    elif isinstance(value, list):
        assert list(got) == value
    else:
        assert got == value


def test_the_a1_oneof_members_round_trip_in_their_envelopes():
    node = pb.NodeMessage(addresses=pb.DataAddresses(data_addresses=["10.0.0.5:47102"]))
    back = pb.NodeMessage.FromString(node.SerializeToString())
    assert back.WhichOneof("m") == "addresses" and list(back.addresses.data_addresses) == ["10.0.0.5:47102"]
    assert pb.NodeMessage.DESCRIPTOR.fields_by_name["addresses"].number == 10
    hub = pb.HubMessage(block_ack=pb.BlockAck(lease_id="L1", block_ids=["A/0/1"]))
    back = pb.HubMessage.FromString(hub.SerializeToString())
    assert back.WhichOneof("m") == "block_ack" and back.block_ack.lease_id == "L1"
    assert pb.HubMessage.DESCRIPTOR.fields_by_name["block_ack"].number == 7


def _without_generator_version(text: str) -> str:
    return re.sub(r"^GRPC_GENERATED_VERSION = .*$", "GRPC_GENERATED_VERSION = <any>", text, flags=re.M)


def test_the_committed_generated_code_matches_the_proto(tmp_path):
    """scripts/gen_proto.sh was run after the last .proto edit: compiling the .proto gives the committed descriptor and
    the committed _pb2_grpc.py (up to the grpcio-tools version line)."""
    pytest.importorskip("grpc_tools", reason="grpcio-tools is in the dev extra")
    from google.protobuf import descriptor_pb2

    src = PROTO.parents[3]                                                  # .../src
    out = tmp_path / "set.pb"
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{src}", f"--descriptor_set_out={out}",
                    f"--grpc_python_out={tmp_path}", str(PROTO.relative_to(src))], check=True, cwd=src)
    fresh_grpc = tmp_path / PROTO.relative_to(src).with_name("colosseum_v1_pb2_grpc.py")
    committed_grpc = Path(pb_grpc.__file__)
    assert _without_generator_version(fresh_grpc.read_text()) == _without_generator_version(committed_grpc.read_text())
    compiled = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes()).file[0]
    committed = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)

    def strip_json_names(messages) -> None:                               # protoc adds them to descriptor sets only
        for message in messages:
            for field in message.field:
                field.ClearField("json_name")
            strip_json_names(message.nested_type)

    strip_json_names(compiled.message_type)
    assert compiled == committed

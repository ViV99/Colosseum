"""Bad data from a node (SP5 T4.4, spec 3.4 and 7.1): dropped, counted by reason, one WARNING per node and reason; the
hub keeps working. A real ``Hub`` after its setup steps (``hub_kit.lineup_hub``) with two nodes driven in-process:
``n1`` runs the learner of ``a``, ``n2`` the learner of ``b``; each test sends the bad item three times from ``n1``
and once from ``n2``, then a good one; a Session stream without its ``Hello`` is cancelled by the real server."""
from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from concurrent.futures import Future

import grpc
import numpy as np
import pytest

from colosseum.hub import server as hub_server
from colosseum.hub.server import UploadIn
from colosseum.hub.state import HUB_STATE_FILE
from colosseum.metrics.counters import CounterBook
from colosseum.net.chunks import encode_checkpoint
from colosseum.net.messages import VersionInfo, results_to_batch, version_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, scripted_agent
from hub_kit import accept_version, lineup_hub, make_result, node_message, open_node, serve


class _Sink:
    """``Hub._hub`` stand-in: records what the metrics hub would get."""

    def __init__(self) -> None:
        self.train: list = []
        self.workers: list = []
        self.intake: list = []
        self.results: list = []
        self.intake_book = CounterBook()

    def on_train_metrics(self, record: dict) -> None:
        self.train.append(record)

    def on_worker_stats(self, record: dict) -> None:
        self.workers.append(record)

    def on_intake_stats(self, *args) -> None:
        self.intake.append(args)

    def on_match_result(self, result) -> None:
        self.results.append(result)

    def set_system_value(self, *args) -> None:
        pass

    def system_state(self, local_nodes=()) -> dict:
        """``MetricsHub.system_state`` (T4.3: ``hub_state.json`` ``counters.system_stats``): no worker totals here."""
        return {}


@pytest.fixture
def setup(tmp_path):
    config = make_test_config("turns", agents={"a": {}, "b": {}, "random": scripted_agent()})
    hub = lineup_hub(config, tmp_path)
    hub._hub = _Sink()
    n1 = open_node(hub, name="n1", cpu_learners=1)
    n2 = open_node(hub, name="n2", cpu_learners=1)
    assert (hub._learner_node_of("a"), hub._learner_node_of("b")) == (_number(hub, n1), _number(hub, n2))
    return hub, n1, n2


def _number(hub, handle) -> int:
    return hub._number_of(handle.node_id)


def _counter(hub, name: str) -> int:
    hub._save_hub_state(fsync=False)
    state = json.loads((hub._run_dir.root / HUB_STATE_FILE).read_text())
    return int(state["counters"]["book"]["totals"].get(name, 0))


def _warnings(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and needle in r.getMessage()]


def _send_results(hub, handle, results, seq):
    hub._on_message(node_message(handle, results=results_to_batch(seq, results)))


def _result(hub, handle, k, *, agent="random", **seat0):
    result = make_result(node=_number(hub, handle), k=k, run_id=hub.run_id, agents=(agent, "random"))
    for key, value in seat0.items():
        setattr(result.seats[0], key, value)
    return result


def test_invalid_results(setup, caplog):
    hub, n1, n2 = setup
    accept_version(hub, "a", 5)

    def unknown_layout(handle, k):
        r = _result(hub, handle, k, agent="a")
        r.layout = "nope"
        return r

    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        _send_results(hub, n1, [unknown_layout(n1, 0)], 1)
        _send_results(hub, n1, [_result(hub, n1, 1, agent="ghost")], 2)
        _send_results(hub, n1, [_result(hub, n1, 2, agent="a", reward=math.nan)], 3)
        _send_results(hub, n2, [unknown_layout(n2, 3)], 1)
        _send_results(hub, n1, [_result(hub, n1, 4, agent="a")], 4)
    assert _counter(hub, "dropped_match_results_by_reason.invalid") == 4
    assert len(_warnings(caplog, "(dropped_match_results_by_reason.invalid)")) == 2
    assert [r.uid.rsplit("/", 1)[1] for r in hub._hub.results] == ["w0_e0_g1_ep4"]
    assert 4 in [s for m in n1.of("ack") for s in m.seqs]


def test_stale_generation_results(setup, caplog):
    hub, n1, n2 = setup
    accept_version(hub, "a", 3)
    hub._versions.new_generation("a")                          # generation 2: generation 1 is capped at v3
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        for seq in (1, 2, 3):
            _send_results(hub, n1, [_result(hub, n1, seq, agent="a", generation=1, version_end=9)], seq)
        _send_results(hub, n2, [_result(hub, n2, 9, agent="a", generation=1, version_end=9)], 1)
        _send_results(hub, n1, [_result(hub, n1, 5, agent="a", generation=1, version_end=3)], 4)   # under the cap
    assert _counter(hub, "dropped_match_results_by_reason.stale_generation") == 4
    assert len(_warnings(caplog, "(dropped_match_results_by_reason.stale_generation)")) == 2
    assert len(hub._hub.results) == 1


def _announce(hub, handle, *, agent="a", version=1, generation=1, node=None):
    info = VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d" * 32,
                       node=_number(hub, handle) if node is None else node)
    hub._on_message(node_message(handle, version=pb.VersionAnnounce(info=version_to_proto(info))))


def test_invalid_version_announcements(setup, caplog):
    hub, n1, n2 = setup
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        _announce(hub, n1, agent="ghost")                            # an unknown agent
        _announce(hub, n1, version=-1)                               # a negative version
        _announce(hub, n1, generation=9)                             # a generation the hub never assigned
        _announce(hub, n2, agent="a")                                # not the node of a's learner
        _announce(hub, n1, version=4)
    assert _counter(hub, "dropped_announcements.invalid") == 4
    assert len(_warnings(caplog, "(dropped_announcements.invalid)")) == 2
    assert hub._versions.accepted("a").version == 4


def _upload(hub, handle, data: bytes, *, agent: str, version: int = 20) -> pb.UploadReply:
    hub._blobs.incoming.mkdir(parents=True, exist_ok=True)
    path = hub._blobs.incoming / f"{uuid.uuid4().hex}.part"
    path.write_bytes(data)
    reply: Future = Future()
    header = pb.UploadHeader(kind="checkpoint", agent_id=agent, generation=1, version=version, total_size=len(data))
    hub._on_upload(UploadIn(header, path, handle.node_id, reply))
    assert not path.exists() and not list(hub._blobs.incoming.iterdir())     # the spool is always removed
    return reply.result(1)


def _frame(agent: str, version: int = 20) -> bytes:
    payload = {"agent_id": agent, "policy_version": version, "model_state": {"w": np.arange(4, dtype=np.float32)},
               "trainer_state_bytes": None, "final": False}
    return encode_checkpoint(payload, kind="periodic", generation=1)


def test_invalid_uploads(setup, caplog):
    hub, n1, n2 = setup
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        replies = [_upload(hub, n1, b"not a frame", agent="a"),
                   _upload(hub, n1, _frame("b"), agent="a"),                   # the frame names another agent
                   _upload(hub, n1, b"\x00" * 64, agent="a"),
                   _upload(hub, n2, b"not a frame either", agent="b")]
        good = _upload(hub, n1, _frame("a"), agent="a")
    assert all(not r.accepted and r.reason.startswith("invalid") for r in replies)
    assert _counter(hub, "dropped_uploads.invalid") == 4
    assert len(_warnings(caplog, "(dropped_uploads.invalid)")) == 2
    assert good.accepted and good.checkpoint_id == "ckpt_v20"


def test_invalid_metrics(setup, caplog):
    hub, n1, n2 = setup
    unknown = pb.MetricsBatch(train=[pb.TrainMetrics(agent_id="ghost", generation=1, train_step=1)])
    negative = pb.MetricsBatch(workers=[pb.WorkerStats(worker=0, incarnation=1, counters={"episodes": -3})])
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        for batch in (unknown, unknown, negative):
            hub._on_message(node_message(n1, metrics=batch))
        hub._on_message(node_message(n2, metrics=negative))
        hub._on_message(node_message(n1, metrics=pb.MetricsBatch(
            workers=[pb.WorkerStats(worker=0, incarnation=1, counters={"episodes": 3})])))
    assert _counter(hub, "dropped_metrics.invalid") == 4
    assert len(_warnings(caplog, "(dropped_metrics.invalid)")) == 2
    assert hub._hub.train == [] and [w["episodes"] for w in hub._hub.workers] == [3]


def test_invalid_events(setup, caplog):
    hub, n1, n2 = setup
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        for _ in range(3):
            hub._on_message(node_message(n1, event=pb.Event(kind="exploded", proc="worker-0")))
        hub._on_message(node_message(n2, event=pb.Event(kind="proc_died", proc="../../x", log_tail=b"x")))
        big = b"0123456789abcdef" * (2**20 // 8)                      # 2 MiB: a valid event keeps its last 64 KiB
        hub._on_message(node_message(n2, event=pb.Event(kind="proc_died", proc="worker-1", exit_code=-9,
                                                        log_tail=big, log_path="/n/worker-1.log")))
    assert _counter(hub, "dropped_events.invalid") == 4
    assert len(_warnings(caplog, "(dropped_events.invalid)")) == 2
    tail = hub._run_dir.logs / "nodes" / str(_number(hub, n2)) / "worker-1.log.tail"
    assert tail.read_bytes() == big[-64 * 1024:]
    assert len(_warnings(caplog, "worker-1 of node")) == 1


def test_a_handler_error_is_counted_and_the_hub_goes_on(setup, caplog, monkeypatch):
    hub, n1, _n2 = setup

    def broken(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(hub, "_on_heartbeat", broken)
    with caplog.at_level(logging.ERROR, logger="colosseum.hub"):
        for _ in range(2):
            hub._on_message(node_message(n1, heartbeat=pb.Heartbeat(env_steps=1)))
    _announce(hub, n1, version=2)                                      # the next message is handled
    assert _counter(hub, "dropped_events.handler_error") == 2
    errors = [r for r in caplog.records if "handling its heartbeat message failed" in r.getMessage()]
    assert len(errors) == 2 and errors[0].exc_info and not errors[1].exc_info
    assert hub._versions.accepted("a").version == 2


def test_a_session_without_its_hello_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_server, "HELLO_TIMEOUT_SEC", 0.3)
    served = serve(tmp_path, insecure=True)
    try:
        node = served.registered_node()
        release = threading.Event()

        def silent():
            release.wait(10)
            return
            yield                                                       # a generator that sends nothing

        call = node.stub().Session(silent(), metadata=node._node_md())
        with pytest.raises(grpc.RpcError) as error:
            list(call)
        release.set()
        assert error.value.code() == grpc.StatusCode.CANCELLED
    finally:
        served.stop()

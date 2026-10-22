"""Log tails of remote processes (SP5 T4.4, spec 7.7): a remote node's death / error event carries the last 64 KiB of
the process's log; the hub keeps them in ``runs/<name>/logs/nodes/<node>/<proc>.log.tail`` and names both paths. A
real ``Hub`` after its setup steps (``hub_kit.lineup_hub``) with nodes driven in-process."""
from __future__ import annotations

import logging
import uuid

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.logs import LOG_TAIL_BYTES
from game_helpers import make_test_config, scripted_agent
from hub_kit import lineup_hub, node_message, open_node

NODE_LOG = "/home/x/.cache/colosseum/logs/ab12cd34/run/worker-0.log"


def _hub(tmp_path):
    hub = lineup_hub(make_test_config("turns", agents={"a": {}, "random": scripted_agent()}), tmp_path)
    open_node(hub, name="n1")
    node = open_node(hub, name="n2")
    assert hub._number_of(node.node_id) == 2
    return hub, node


def _died(hub, node, tail: bytes, *, proc: str = "worker-0", log_path: str = NODE_LOG) -> None:
    hub._on_message(node_message(node, event=pb.Event(kind="proc_died", proc=proc, exit_code=-9, log_tail=tail,
                                                      log_path=log_path)))


def test_a_remote_proc_died_event_writes_its_tail(tmp_path, caplog):
    hub, node = _hub(tmp_path)
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        _died(hub, node, b"line 1\nline 2\n")
    path = hub._run_dir.logs / "nodes" / "2" / "worker-0.log.tail"
    assert path.read_bytes() == b"line 1\nline 2\n"
    (said,) = [r.getMessage() for r in caplog.records if "worker-0 of node 2" in r.getMessage()]
    assert f"see {path} (node log {NODE_LOG})" in said


def test_a_tail_is_capped_and_overwritten(tmp_path):
    hub, node = _hub(tmp_path)
    big = bytes(range(256)) * 800                                       # 200 KB
    _died(hub, node, big)
    path = hub._run_dir.logs / "nodes" / "2" / "worker-0.log.tail"
    assert path.read_bytes() == big[-LOG_TAIL_BYTES:] and LOG_TAIL_BYTES == 64 * 1024
    _died(hub, node, b"again\n")
    assert path.read_bytes() == b"again\n"


def test_the_local_node_writes_no_tail_file(tmp_path, caplog):
    hub = lineup_hub(make_test_config("turns", agents={"a": {}, "random": scripted_agent()}), tmp_path)
    record, _secret = hub.identity.nodes.add_ephemeral("local", uuid.uuid4().hex)
    node = open_node(hub, name="local", node_id=record.node_id)
    assert hub._registry.by_id(node.node_id).data_path == "local"
    local_log = str(hub._run_dir.logs / "worker-0.log")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        _died(hub, node, b"local tail\n", log_path=local_log)
    assert not (hub._run_dir.logs / "nodes").exists()
    (said,) = [r.getMessage() for r in caplog.records if "worker-0 of node 0" in r.getMessage()]
    assert f"see {local_log}" in said and "logs/worker-0.log" in said


def test_a_hostile_proc_name_cannot_escape_the_logs_dir(tmp_path, caplog):
    hub, node = _hub(tmp_path)
    with caplog.at_level(logging.WARNING, logger="colosseum.hub"):
        _died(hub, node, b"evil\n", proc="../../x")
    assert not list(tmp_path.rglob("x.log.tail")) and not list(tmp_path.rglob("*.log.tail"))
    assert hub._book.totals().get("dropped_events.invalid") == 1

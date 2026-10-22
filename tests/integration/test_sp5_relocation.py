"""Network cuts and relocation on localhost (SP5 T4.2, criterion 11 shape): a short cut changes nothing; a long one
moves the learner to the other node with a new generation and stops the old one when the machine returns.

An in-process hub (``cluster_kit.start_hub``) on tic-tac-toe with one trainable agent ``main`` whose CPU learner
prefers node ``laptop`` (``learner.placement.node``; recovery points every 0.5 s); two ``NodeAgent``s in threads with
their own identity roots: ``laptop`` (no workers, one CPU learner slot, no data port: its chunks and weights go
through the hub relay) reaches the hub through a ``fault_proxy.TcpFaultProxy``, ``pc`` (one worker, one CPU learner
slot) directly. Loaded processes: ``pc``'s worker and the learner (2; while the learner moves, the old one is cut off
and exits once told). ``laptop`` starts first and has its learner before ``pc`` connects (P-PF3-20); the node's
reconnect backoff is shrunk (P-PF3-8)."""
from __future__ import annotations

import re

import pytest

from cluster_kit import (
    cluster_config,
    hub_version,
    journal_rows,
    read_hub_state,
    start_hub,
    start_node,
    wait_until,
)
from colosseum.node import session as node_session
from fault_proxy import TcpFaultProxy

EPOCH = re.compile(r"/w(\d+)_e(\d+)_g(\d+)_ep\d+$")


@pytest.fixture(autouse=True)
def _fast_reconnect(monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)


def _learner(hub) -> dict:
    return ((read_hub_state(hub).get("learners") or {}).get("agents") or {}).get("main") or {}


def _pc_epochs(hub, pc) -> dict[tuple[int, int], set[int]]:
    """Per (worker, env) of ``pc``: the epochs of its journaled training matches."""
    out: dict[tuple[int, int], set[int]] = {}
    for row in journal_rows(hub):
        match = EPOCH.search(row.uid)
        if f"/n{pc.agent.number}/" in row.uid and match:
            w, e, g = map(int, match.groups())
            out.setdefault((w, e), set()).add(g)
    return out


def _cluster(tmp_path, lost_sec: float, relocate_sec: float):
    config = cluster_config(tmp_path, {
        "agents.main": {"learner": {"placement": {"node": "laptop"}, "recovery_interval_sec": 0.5}},
        "training.total_timesteps": 10**9, "distributed.heartbeat_sec": 0.1,
        "distributed.node_lost_after_sec": lost_sec, "distributed.learner_relocate_after_sec": relocate_sec})
    hub = start_hub(tmp_path, config)
    host, _, port = hub.address.rpartition(":")
    proxy = TcpFaultProxy((host, int(port)))
    laptop = start_node(tmp_path, hub, "laptop", cpu_learners=1, workers=0, data_port=None,
                        hub_address=proxy.address)
    assert wait_until(lambda: _learner(hub).get("node") == laptop.agent.number
                      and hub_version(hub, "main") is not None, 120), _learner(hub)
    pc = start_node(tmp_path, hub, "pc", cpu_learners=1, workers=1)
    return hub, proxy, laptop, pc


def _stop(hub, proxy, *nodes):
    try:
        hub.stop()
    finally:
        for node in nodes:
            if node.thread.is_alive():
                node.stop()
        proxy.close()


def test_a_cut_shorter_than_node_lost_changes_nothing(tmp_path):
    hub, proxy, laptop, pc = _cluster(tmp_path, lost_sec=2.0, relocate_sec=4.0)
    try:
        assert wait_until(lambda: len(_pc_epochs(hub, pc)) == 8, 120)        # every cell of pc journaled a match
        before = _pc_epochs(hub, pc)
        number, version = laptop.agent.number, hub_version(hub, "main")
        proxy.cut(0.5)
        assert wait_until(lambda: (hub_version(hub, "main") or 0) > version, 60)   # it trains on after the cut
        rows = len(journal_rows(hub))
        assert wait_until(lambda: len(journal_rows(hub)) > rows + 8, 60)
        state = read_hub_state(hub)
        assert state.get("nodes_lost", 0) == 0
        assert (_learner(hub)["node"], _learner(hub)["generation"], _learner(hub)["status"]) == (number, 1, "running")
        assert laptop.agent.number == number
        after = _pc_epochs(hub, pc)
        assert all(after[cell] == before[cell] for cell in before)          # no new epoch for pc's cells
    finally:
        _stop(hub, proxy, laptop, pc)


@pytest.mark.slow
def test_a_long_cut_relocates_the_learner_and_stops_the_old_one(tmp_path):
    hub, proxy, laptop, pc = _cluster(tmp_path, lost_sec=1.0, relocate_sec=2.0)
    try:
        assert wait_until(lambda: (hub.root / "recovery" / "main" / "recovery.frame").exists(), 60)
        assert wait_until(lambda: len(_pc_epochs(hub, pc)) > 0, 120)
        old = hub.hub._versions.accepted("main").version
        proxy.cut(5.0)
        assert wait_until(lambda: _learner(hub).get("node") == pc.agent.number and _learner(hub).get("generation") == 2,
                          30), _learner(hub)
        entry = wait_until(lambda: pc.agent.learners.of("learner-main"), 30) and pc.agent.learners.of("learner-main")
        assert entry.assignment.weights_blob == "start:recovery"         # from the newest recovery point
        assert wait_until(lambda: (info := hub.hub._versions.accepted("main")) is not None and info.generation == 2,
                          120)
        cap = hub.hub._versions.state()["main"]["caps"]["1"]
        assert hub.hub._versions.accepted("main").version > max(cap, old)
        # the cut ends: laptop reconnects with its generation-1 learner, is told to stop it, and it exits
        assert wait_until(lambda: not laptop.agent.learners.agents(), 60)
        proc = laptop.agent.supervisor.get("learner-main")
        assert proc is not None and wait_until(lambda: proc.exitcode is not None, 30)
        assert _learner(hub)["node"] == pc.agent.number                  # a learner never moves back
        assert hub.hub._failure is None and not hub.hub._handler_errors
    finally:
        _stop(hub, proxy, laptop, pc)

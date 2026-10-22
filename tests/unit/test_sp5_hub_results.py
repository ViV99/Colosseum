"""The hub's result accounting (SP5 T2.5; spec 4.3, 4.4, 5.1; ruling R8; pre-flight rulings P-PF2-13, P-PF2-14): the
real ``Hub._on_result_batch`` / ``_drain_commits`` with a real journal writer under ``tmp_path`` (or the journal
off), heartbeats driving the budget, and the journal watermark in ``league_state.json``."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

import pytest

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader, MatchLogWriter
from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, scripted_agent
from hub_kit import hub_results, lineup_hub, node_message, open_node, result_batch

AGENTS = {"a": {}, "random": scripted_agent()}


def _wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _config(**sections):
    return make_test_config("turns", agents=dict(AGENTS), **sections)


@pytest.fixture
def results_hub(tmp_path):
    """A hub with its sinks open (journal on) and one node (number 1) running ``a``'s learner (generation 1)."""
    hubs = []

    def make(config=None, path=None, **kw):
        hub = lineup_hub(config or _config(), path or tmp_path, sinks=True, **kw)
        hubs.append(hub)
        accounted: list[str] = []
        original = hub._coordinator.report_match_result
        hub._coordinator.report_match_result = lambda r: (accounted.append(r.uid), original(r))[1]
        hub.accounted = accounted
        return hub

    yield make
    for hub in hubs:
        hub._finish_metrics()


def _commit(hub, n_accounted, timeout=10.0):
    """Drain the writer's commits until ``n_accounted`` results are accounted."""
    assert _wait(lambda: (hub._drain_commits() or True) and len(hub.accounted) >= n_accounted, timeout), \
        hub.accounted


def _book(hub, reason):
    return hub._book.get(f"dropped_match_results_by_reason.{reason}")


def test_results_are_accounted_once_after_their_commit_and_acked_then(results_hub):
    hub = results_hub()
    node = open_node(hub)
    results = hub_results(hub, 1, 3)
    result_batch(hub, node, results, seq=1)
    assert hub.accounted == [] and node.acks() == []                     # not before the commit
    _commit(hub, 3)
    assert hub.accounted == [r.uid for r in results] and node.acks() == [1]
    hub._save_league_state()
    assert json.loads(hub._run_dir.league_state_path.read_text())["journal_row"] == 3


def test_a_resent_batch_is_skipped_while_in_flight_and_reacked_after(results_hub):
    hub = results_hub()
    node = open_node(hub)
    results = hub_results(hub, 1, 3)
    submits = []
    original = hub._match_log_writer.submit
    hub._match_log_writer.submit = lambda rs, **kw: (submits.append(len(rs)), original(rs, **kw))[1]
    result_batch(hub, node, results, seq=1)
    result_batch(hub, node, results, seq=1)                               # in flight: skipped
    assert submits == [3]
    _commit(hub, 3)
    result_batch(hub, node, results, seq=1)                               # acked already: acked again
    assert submits == [3] and node.acks() == [1, 1] and len(hub.accounted) == 3


def test_a_resend_after_a_hub_restart_is_accounted_once(results_hub, tmp_path):
    from colosseum.hub.hub import Hub, HubOptions

    config = _config()
    first = results_hub(config)
    node = open_node(first)
    results = hub_results(first, 1, 3)
    result_batch(first, node, results, seq=1)
    _commit(first, 3)
    first._stop_match_log()
    # A new hub process on the same run dir and journal (a fresh ResultIntake: it knows no acked seq).
    second = Hub(config, first._run_dir, options=HubOptions(mode="hub", identity_dir=tmp_path / "hub-identity"))
    second._prepare()
    second._open_sinks()
    accounted: list[str] = []
    second._coordinator.report_match_result = lambda r: accounted.append(r.uid)
    try:
        again = open_node(second, node_id=node.node_id)                  # the node reconnects, seq 1 again
        commits = []
        take = second._match_log_writer.take_commits
        second._match_log_writer.take_commits = lambda: (lambda cs: (commits.extend(cs), cs)[1])(take())
        result_batch(second, again, results, seq=1)
        assert _wait(lambda: (second._drain_commits() or True) and commits)
        (commit,) = commits
        assert (commit.accepted, commit.duplicates, commit.failed) == ([], 3, False)
        assert accounted == [] and again.acks() == [1]
    finally:
        second._finish_metrics()


def test_a_batch_the_writer_cannot_take_is_neither_accounted_nor_acked(results_hub):
    hub = results_hub()
    node = open_node(hub)
    hub._match_log_writer.stop(5.0)
    writer = MatchLogWriter(hub._match_log_writer._factory, queue_size=2, compact_interval_sec=3600.0,
                            window_bytes=10**9)
    hub._match_log_writer = writer                                        # not started yet: its queue fills
    assert writer.submit(hub_results(hub, 1, 1, first=100), token="other") == 0    # R23: the queue is not empty
    result_batch(hub, node, hub_results(hub, 1, 3), seq=1)
    assert hub.accounted == [] and node.acks() == []
    assert not hub._intake.is_acked(node.node_id, node.session_id, 1)
    writer.start()
    assert _wait(lambda: writer.stats.pending == 0)
    writer.take_commits()                                                 # the other group's commit
    result_batch(hub, node, hub_results(hub, 1, 3), seq=1)                # the node resends it
    _commit(hub, 3)
    assert node.acks() == [1] and len(hub.accounted) == 3


def test_a_dead_writer_switches_to_accounting_at_intake(results_hub, caplog):
    hub = results_hub()
    node = open_node(hub)
    hub._match_log_writer.stop(5.0)

    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    writer = MatchLogWriter(broken, compact_interval_sec=3600.0, window_bytes=1)
    hub._match_log_writer = writer
    results = hub_results(hub, 1, 3)
    with caplog.at_level(logging.ERROR):
        result_batch(hub, node, results, seq=1)                           # queued before the thread runs
        writer.start()
        assert _wait(lambda: writer.dead)
        assert _wait(lambda: (hub._drain_commits() or True) and not hub._intake._sessions[
            (node.node_id, node.session_id)].in_flight)                   # the failed commit: not acked
        assert hub.accounted == [] and node.acks() == []
        result_batch(hub, node, results, seq=1)                           # the resend: accounted at intake
    assert hub.accounted == [r.uid for r in results] and node.acks() == [1]
    assert _book(hub, "writer") == 3
    assert sum("accounted without being journaled" in r.getMessage() for r in caplog.records
               if r.levelno == logging.ERROR) == 1


def test_without_the_journal_results_are_accounted_at_intake(results_hub):
    hub = results_hub(_config(match_log={"enabled": False}))
    node = open_node(hub)
    results = hub_results(hub, 1, 3)
    result_batch(hub, node, results, seq=1)
    assert hub.accounted == [r.uid for r in results] and node.acks() == [1]
    result_batch(hub, node, results, seq=1)                               # dedup by (node, session, seq) only
    assert len(hub.accounted) == 3 and node.acks() == [1, 1]
    hub._save_league_state()
    assert json.loads(hub._run_dir.league_state_path.read_text())["journal_row"] is None


def test_invalid_results_are_dropped_counted_and_warned_once_per_node(results_hub, caplog):
    hub = results_hub()
    node = open_node(hub)
    good = hub_results(hub, 1, 2)
    unknown_layout = hub_results(hub, 1, 1, first=50)[0]
    unknown_layout.layout = "9p"
    foreign = hub_results(hub, 7, 1, first=60)[0]                         # another node's uid and number
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.hub"):
        result_batch(hub, node, [good[0], unknown_layout, foreign], seq=1)
        bad = hub_results(hub, 1, 1, first=70)[0]
        bad.layout = "9p"
        result_batch(hub, node, [good[1], bad], seq=2)
    _commit(hub, 2)
    assert hub.accounted == [good[0].uid, good[1].uid]
    assert _book(hub, "invalid") == 3 and sorted(node.acks()) == [1, 2]
    assert caplog.text.count("dropped_match_results_by_reason.invalid") == 1


def test_an_undecodable_batch_is_counted_invalid_and_acked(results_hub):
    hub = results_hub()
    node = open_node(hub)
    ev = node_message(node, results=pb.ResultBatch(seq=1, results=[pb.MatchResultProto(), pb.MatchResultProto()]))
    ev.results, ev.error = None, "ValueError: broken"
    hub._on_message(ev)
    assert _book(hub, "invalid") == 2 and node.acks() == [1] and hub.accounted == []


def test_results_are_stamped_with_the_hubs_env_steps_at_intake(results_hub):
    hub = results_hub()
    node = open_node(hub)
    hub._budget.report(node.node_id, node.session_id, 1234)
    results = hub_results(hub, 1, 3)
    results[0].env_steps = 99                                             # a node never sets it: overwritten
    result_batch(hub, node, results, seq=1)
    _commit(hub, 3)
    hub._stop_match_log()
    with MatchLogReader(hub._run_dir.root / MATCH_LOG_FILE) as reader:
        rows = reader.matches(source="train", limit=None)
    assert sorted(r.env_steps for r in rows) == [1234, 1234, 1234]


def test_heartbeats_drive_the_budget_and_the_stop(results_hub, caplog):
    hub = results_hub(_config(training={"total_timesteps": 1000}))
    one, two = open_node(hub, name="one"), open_node(hub, name="two", cpu_learners=0)
    hub._on_message(node_message(one, heartbeat=pb.Heartbeat(env_steps=600, counters={"lineup_fetch_failed": 2})))
    hub._on_message(node_message(two, heartbeat=pb.Heartbeat(env_steps=500)))
    hub._on_message(node_message(one, heartbeat=pb.Heartbeat(env_steps=550)))     # not larger: ignored
    assert hub.env_steps_done == 1100 and hub._book.get("lineup_fetch_failed") == 2
    with caplog.at_level(logging.INFO, logger="colosseum.hub.hub"):
        assert hub._main_loop() == 0
    assert "Training budget reached: 1100 env steps (budget 1000)" in caplog.text
    announces = one.of("announce")
    assert announces and announces[-1].env_steps == 1100                    # the learners' LR progress


def test_the_league_state_records_the_last_accounted_journal_row(results_hub):
    hub = results_hub()
    node = open_node(hub)
    result_batch(hub, node, hub_results(hub, 1, 3), seq=1)
    _commit(hub, 3)
    result_batch(hub, node, hub_results(hub, 1, 2, first=3), seq=2)
    _commit(hub, 5)
    hub._save_league_state()
    assert json.loads(hub._run_dir.league_state_path.read_text())["journal_row"] == 5


def test_commits_made_while_the_writer_stops_are_accounted(results_hub):
    hub = results_hub()
    node = open_node(hub)
    result_batch(hub, node, hub_results(hub, 1, 4), seq=1)
    hub._stop_match_log()                                                 # P-PF2-14: no _drain_commits before
    assert len(hub.accounted) == 4 and node.acks() == [1]

"""The node's checkpoint uploads (SP5 T4.2, spec 5.2 and 7.4): finals are never dropped and go first, recovery
points and periodic snapshots wait one deep (a replaced periodic snapshot is counted), one upload per agent at a
time, failed uploads wait again, refused ones are not retried, the stop deadline drops what is left."""
from __future__ import annotations

import pytest

from colosseum.node.uploads import UploadQueue


def _kinds(uploads):
    return [(u.agent_id, u.kind, u.version) for u in uploads]


def test_finals_first_then_recovery_then_periodic_one_per_agent():
    q = UploadQueue(concurrency=4)
    q.offer("a", "periodic", 1, 10, b"p10")
    q.offer("a", "recovery", 1, 12, b"r12")
    q.offer("b", "periodic", 1, 5, b"bp5")
    q.offer("a", "final", 1, 13, b"f13")
    started = q.take()
    assert _kinds(started) == [("a", "final", 13), ("b", "periodic", 5)]
    assert q.take() == []                                     # a is busy, b has nothing else
    for upload in started:
        assert q.done(upload.ticket, True) is upload
    assert _kinds(q.take()) == [("a", "recovery", 12)]
    assert q.done(999, True) is None


def test_a_newer_periodic_snapshot_replaces_the_waiting_one_and_is_counted():
    q = UploadQueue()
    for version in (10, 20, 30):
        q.offer("a", "periodic", 1, version, b"x")
    assert q.replaced == 2 and q.waiting("periodic") == 1
    assert _kinds(q.take()) == [("a", "periodic", 30)]
    q.offer("a", "recovery", 1, 31, b"r")
    q.offer("a", "recovery", 1, 32, b"r")
    assert q.replaced == 2                                    # recovery points are replaced silently


def test_finals_are_never_replaced_and_retried_after_a_failure():
    q = UploadQueue(concurrency=1)
    t1 = q.offer("a", "final", 1, 13, b"f1")
    q.offer("a", "final", 2, 40, b"f2")                       # another generation's final: both kept
    assert _kinds(q.take()) == [("a", "final", 13)]
    q.done(t1, False)                                          # the hub is away
    assert _kinds(q.take()) == [("a", "final", 13)]           # first again
    q.done(t1, True)
    assert _kinds(q.take()) == [("a", "final", 40)]


def test_a_failed_periodic_upload_waits_again_unless_a_newer_one_came():
    q = UploadQueue()
    t = q.offer("a", "periodic", 1, 10, b"x")
    q.take()
    q.done(t, False)
    assert q.waiting("periodic") == 1 and q.replaced == 0
    t = q.take()[0].ticket
    q.offer("a", "periodic", 1, 20, b"y")
    q.done(t, False)
    assert q.replaced == 1 and _kinds(q.take()) == [("a", "periodic", 20)]


def test_a_refused_upload_is_not_retried_and_the_concurrency_bound_holds():
    q = UploadQueue(concurrency=2)
    for agent in "abc":
        q.offer(agent, "periodic", 1, 1, b"x")
    started = q.take()
    assert len(started) == 2
    q.done(started[0].ticket, False, retry=False)              # refused: stale_generation
    assert q.waiting() == 2 and len(q.take()) == 1


def test_drop_waiting_at_the_stop_deadline():
    q = UploadQueue(concurrency=1)
    q.offer("a", "final", 1, 1, b"f")
    q.offer("b", "final", 1, 1, b"g")
    q.offer("b", "periodic", 1, 1, b"p")
    q.take()
    dropped = q.drop_waiting()
    assert sorted((u.agent_id, u.kind) for u in dropped) == [("b", "final"), ("b", "periodic")]
    assert q.waiting() == 1                                    # the one in flight
    with pytest.raises(ValueError):
        q.offer("a", "weird", 1, 1, b"")

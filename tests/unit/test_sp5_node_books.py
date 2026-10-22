"""The node's result buffer and cell tracker (SP5 T2.4; spec 4.2, 4.3, R11): batches by size and age, seqs, kept
until acked, overflow drops the new batch with one WARNING; cells requested once until answered, again after their TTL
or a failed fetch, grants wait for their weights, whole-run lineups never expire, dead incarnations lose their cells."""
from __future__ import annotations

import logging

from colosseum.node.agent import CellTracker, ResultBuffer


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_results_are_batched_by_size_and_age_numbered_and_kept_until_acked():
    clock = Clock()
    buf = ResultBuffer(batch=3, flush_sec=0.25, limit_bytes=1000, make=lambda seq, b: (seq, b),
                       size_of=lambda m: 10 * len(m[1]), clock=clock)
    assert buf.add([1, 2]) == [] and buf.due() == []
    assert buf.add([3, 4]) == [(1, [1, 2, 3])]
    clock.t = 0.3
    assert buf.due() == [(2, [4])]
    assert buf.unacked() == [(1, [1, 2, 3]), (2, [4])] and buf.kept_bytes == 40 and buf.last_acked == 0
    assert buf.ack([1, 99]) == 1 and buf.unacked() == [(2, [4])] and buf.last_acked == 1
    buf.ack([2])
    assert buf.unacked() == [] and buf.last_acked == 2 and buf.kept_bytes == 0


def test_a_full_buffer_drops_the_new_batch_and_warns_once(caplog):
    clock = Clock()
    buf = ResultBuffer(batch=2, flush_sec=1.0, limit_bytes=50, make=lambda seq, b: (seq, b),
                       size_of=lambda m: 20 * len(m[1]), clock=clock)
    with caplog.at_level(logging.WARNING):
        assert [s for s, _ in buf.add(range(6))] == [1]            # 40 kept, then two batches of 40 do not fit
        assert buf.add([7]) == [] and buf.flush() == []
    assert buf.dropped == 5 and buf.unacked() == [(1, [0, 1])]
    assert len([r for r in caplog.records if "result buffer full" in r.getMessage()]) == 1
    buf.ack([1])
    assert buf.add([8, 9]) == [(2, [8, 9])]                         # room again; numbering goes on


def test_cells_are_requested_once_until_answered_and_again_after_their_ttl():
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 2, incarnation=1)
    assert cells.due() == [(0, 0, 0, 1), (0, 1, 0, 1)] and cells.due() == []
    assert cells.granted(0, 0, epoch=7, ttl_sec=30.0, lineup="L0", keys={0: "k-latest"})
    assert cells.waiting() == [(0, 0, "L0", {0: "k-latest"})] and cells.referenced() == {"k-latest"}
    assert cells.deliver(0, 0) == ("L0", {0: "k-latest"}) and cells.waiting() == []
    clock.t = 1.5
    assert cells.due() == [(0, 1, 0, 1)]                            # no answer within retry_sec: asked again
    clock.t = 31.0
    assert (0, 0, 7, 1) in cells.due()                              # ttl expired: asked with its epoch
    assert cells.live() == [(0, 0, 7, 1)]


def test_a_ttl_shorter_than_the_retry_wait_renews_at_the_ttl():
    """``retry_sec`` holds back only a request without an answer: an answered cell is due again at its TTL (FX6: a
    0.1 s ``match_refresh_interval_sec`` renewed only every ``LINEUP_RETRY_SEC`` = 1 s)."""
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 1, incarnation=1)
    assert cells.due() == [(0, 0, 0, 1)]
    clock.t = 0.01
    assert cells.granted(0, 0, epoch=4, ttl_sec=0.1, lineup="L0", keys={})
    cells.deliver(0, 0)
    clock.t = 0.1
    assert cells.due() == []                                        # the TTL runs from the grant
    clock.t = 0.12
    assert cells.due() == [(0, 0, 4, 1)]                            # not held until the retry wait (1.0)
    assert cells.due() == []                                        # requested: once until answered
    clock.t = 1.2
    assert cells.due() == [(0, 0, 4, 1)]                            # no answer within retry_sec: asked again


def test_a_failed_fetch_keeps_the_old_lineup_and_retries_later():
    clock = Clock()
    cells = CellTracker(retry_sec=2.0, clock=clock)
    cells.add_worker(0, 1, incarnation=1)
    cells.due()
    cells.granted(0, 0, epoch=3, ttl_sec=30.0, lineup="old", keys={0: "a"})
    cells.deliver(0, 0)
    clock.t = 31.0
    cells.due()
    cells.granted(0, 0, epoch=3, ttl_sec=30.0, lineup="new", keys={0: "snap"})
    assert cells.referenced() == {"a", "snap"}
    cells.failed(0, 0)
    assert cells.referenced() == {"a"} and cells.due() == []
    clock.t = 33.5
    assert cells.due() == [(0, 0, 3, 1)]


def test_whole_run_lineups_never_expire_and_dead_incarnations_lose_their_cells():
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(1, 1, incarnation=1)
    cells.due()
    cells.granted(1, 0, epoch=4, ttl_sec=0.0, lineup="L", keys={})
    cells.deliver(1, 0)
    clock.t = 10**6
    assert cells.due() == []
    cells.drop_worker(1)
    assert not cells.granted(1, 0, epoch=5, ttl_sec=0.0, lineup="L", keys={})
    cells.add_worker(1, 1, incarnation=2)
    assert cells.due() == [(1, 0, 0, 2)]


def test_unacked_batches_are_resent_after_their_resend_interval_and_given_up_at_the_stop(caplog):
    clock = Clock()                                                 # P-PF2-4 (a, b)
    buf = ResultBuffer(batch=2, flush_sec=1.0, limit_bytes=1000, make=lambda seq, b: (seq, b),
                       size_of=lambda m: 10 * len(m[1]), clock=clock)
    assert buf.add([1, 2, 3, 4]) == [(1, [1, 2]), (2, [3, 4])]
    clock.t = 5.0
    assert buf.add([5]) == [] and buf.overdue(10.0) == []
    clock.t = 10.5
    assert buf.overdue(10.0) == [(1, [1, 2]), (2, [3, 4])] and buf.overdue(10.0) == []   # once per interval
    buf.ack([1])
    clock.t = 21.0
    assert buf.overdue(10.0) == [(2, [3, 4])]
    assert buf.resend_all() == [(2, [3, 4])] and buf.overdue(10.0) == []                  # a resend restarts it
    with caplog.at_level(logging.WARNING):
        assert buf.give_up() == 3                                   # two unacked and one pending result
    assert buf.dropped == 3 and buf.unacked() == [] and buf.kept_bytes == 0 and buf.flush() == []
    assert len([r for r in caplog.records if "gave up" in r.getMessage()]) == 1
    assert buf.give_up() == 0


def test_only_requested_cells_take_grants_and_a_restarted_worker_waits_for_its_old_answer():
    clock = Clock()
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 1, incarnation=1)
    assert not cells.granted(0, 0, epoch=3, ttl_sec=5.0, lineup="L", keys={})    # nothing was asked yet
    assert cells.due() == [(0, 0, 0, 1)]
    cells.drop_worker(0)                                            # restarted with its request unanswered
    cells.add_worker(0, 1, incarnation=2)
    assert cells.due() == [] and not cells.granted(0, 0, epoch=3, ttl_sec=5.0, lineup="old", keys={})
    clock.t = 1.0
    assert cells.due() == [(0, 0, 0, 2)]
    assert cells.granted(0, 0, epoch=4, ttl_sec=5.0, lineup="new", keys={})
    cells.deliver(0, 0)
    assert cells.granted(0, 0, epoch=4, ttl_sec=5.0, lineup="again", keys={})    # a late answer of its epoch
    assert not cells.granted(0, 0, epoch=9, ttl_sec=5.0, lineup="other", keys={})


def test_a_grant_waiting_too_long_for_its_weights_is_given_up():
    clock = Clock()                                                 # P-PF2-12
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 2, incarnation=1)
    cells.due()
    cells.granted(0, 0, epoch=1, ttl_sec=5.0, lineup="L", keys={0: "k"})
    cells.granted(0, 1, epoch=2, ttl_sec=60.0, lineup="L", keys={0: "k"})
    clock.t = 29.0
    assert cells.stale_waiting(30.0) == []
    clock.t = 30.0
    assert cells.stale_waiting(30.0) == [(0, 0, {0: "k"})]          # max(ttl, wait): 30 s for a 5 s ttl
    clock.t = 60.0
    assert [(w, e) for w, e, _ in cells.stale_waiting(30.0)] == [(0, 0), (0, 1)]


def test_a_new_incarnation_never_takes_an_epoch_of_a_dead_one_however_late_its_answer():
    clock = Clock()                                                 # review round 1: uids / seeds never repeat
    cells = CellTracker(retry_sec=1.0, clock=clock)
    cells.add_worker(0, 1, incarnation=1)
    cells.due()
    cells.granted(0, 0, epoch=7, ttl_sec=5.0, lineup="L7", keys={})
    cells.deliver(0, 0)
    clock.t = 6.0
    assert cells.due() == [(0, 0, 7, 1)]                            # incarnation 1 asks to renew epoch 7 ...
    cells.drop_worker(0)                                            # ... and dies before the answer
    clock.t = 7.5
    cells.add_worker(0, 1, incarnation=2)
    assert cells.due() == [(0, 0, 0, 2)]
    clock.t = 9.0                                                   # the old answer arrives > retry_sec later
    assert not cells.granted(0, 0, epoch=7, ttl_sec=5.0, lineup="L7", keys={})
    assert cells.granted(0, 0, epoch=8, ttl_sec=5.0, lineup="L8", keys={})   # the hub's answer for incarnation 2
    assert cells.live() == [(0, 0, 8, 2)]
    cells.drop_worker(0)                                            # floors only rise over restarts
    cells.add_worker(0, 1, incarnation=3)
    clock.t = 20.0
    cells.due()
    assert not cells.granted(0, 0, epoch=8, ttl_sec=5.0, lineup="L8", keys={})
    assert not cells.granted(0, 0, epoch=5, ttl_sec=5.0, lineup="L5", keys={})


def test_the_books_live_in_their_own_module_and_the_agent_re_exports_them():
    from colosseum.node import agent, books

    assert agent.ResultBuffer is books.ResultBuffer and agent.CellTracker is books.CellTracker

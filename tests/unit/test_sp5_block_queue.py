"""The hub's block queue (SP5 T5.1, spec 5.3): priority and FIFO, leases of one owner with a match target, round
robin over nodes, a block's result whole and once (first complete wins, late / cancelled / unknown counted),
expiry, attempts, node loss, session resume, cancel. Pure: the clock is a number."""
from __future__ import annotations

import pytest

from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.evaluation import MAX_BLOCK_ATTEMPTS, BlockQueue, check_block_results
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.players.registry import BotSpec
from game_helpers import make_match_result, make_test_config

BOT = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))
SPEC = env_spec(make_test_config("turns"))       # layout "2p": two "player" seats, teams 0 and 1


def _task(block_id, owner="tournament:t", priority=0, matches=2, num_envs=1):
    lineups = tuple(Lineup("2p", [SeatAssignment("a", "ckpt_v1", False), SeatAssignment("rnd", "fixed", False)],
                           seed=7, block=block_id, uid=f"{block_id}/r{r}") for r in range(matches))
    source = "pulse" if owner.startswith("pulse:") else "tournament"
    return BlockTask(block_id, source, owner, lineups, (BOT,), priority, False, num_envs)


def _ids():
    counter = iter(range(10_000))
    return lambda: f"L{next(counter)}"


def _results(task):
    return [make_match_result([(0, lu.seats[0].agent_id, lu.seats[0].network_id, "player", 1),
                               (1, lu.seats[1].agent_id, lu.seats[1].network_id)],
                              uid=lu.uid, block=task.block_id, seed=lu.seed, source=task.source)
            for lu in task.lineups]


def test_priority_fifo_grouping_and_round_robin():
    q = BlockQueue(task_matches=2, new_id=_ids())          # target 2 x num_envs (1) = 2 matches per lease
    q.submit([_task("p0", "pulse:a:c1", 1), _task("p1", "pulse:a:c1", 1)])
    q.submit([_task("t0"), _task("t1"), _task("t2")])
    (lease, tasks), = q.lease(1, 1, 0.0, 10.0)
    assert [t.block_id for t in tasks] == ["t0"] and lease.deadline == 10.0 and lease.node == 1
    q.want(1, 2)
    q.want(2, 2)
    got = q.dispatch(0.0, 10.0)
    assert [(n, [t.block_id for t in ts]) for n, _l, ts in got] == [(1, ["t1"]), (2, ["t2"]), (1, ["p0"]), (2, ["p1"])]
    assert q.waiting() == {}
    big = BlockQueue(task_matches=4, new_id=_ids())
    big.submit([_task(f"b{i}", num_envs=2) for i in range(5)])        # target 8 matches: 4 blocks of 2
    (_l1, first), (_l2, rest) = big.lease(3, 2, 0.0, 1.0)
    assert [t.block_id for t in first] == ["b0", "b1", "b2", "b3"] and [t.block_id for t in rest] == ["b4"]


def test_a_lease_never_mixes_owners():
    q = BlockQueue(task_matches=10, new_id=_ids())
    q.submit([_task("x0", "tournament:x"), _task("y0", "tournament:y"), _task("x1", "tournament:x")])
    assert [[t.block_id for t in ts] for _l, ts in q.lease(1, 3, 0.0, 1.0)] == [["x0"], ["y0"], ["x1"]]


def test_dispatch_keeps_the_wanted_slots_when_the_queue_runs_dry():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.want(4, 3)
    assert q.dispatch(0.0, 1.0) == [] and q.waiting() == {4: 3}
    q.submit([_task("t0")])
    (node, _lease, tasks), = q.dispatch(0.0, 1.0)
    assert node == 4 and [t.block_id for t in tasks] == ["t0"] and q.waiting() == {4: 2}
    q.unwant(4)
    assert q.waiting() == {}


def test_an_expired_lease_requeues_and_its_late_result_still_wins():
    q = BlockQueue(task_matches=1, new_id=_ids())
    t = _task("t0")
    q.submit([t])
    (lease, _), = q.lease(1, 1, 0.0, 5.0)
    assert q.expire(4.9) == [] and q.expire(5.0) == ["t0"]                 # back in the queue, one attempt used
    (lease2, _), = q.lease(2, 1, 6.0, 5.0)
    assert q.complete(lease.lease_id, "t0", _results(t), 7.0) == "accepted"   # the expired lease came back first
    assert q.complete(lease2.lease_id, "t0", _results(t), 8.0) == "late"
    assert q.complete("nope", "t0", [], 8.0) == "unknown"                   # never issued for this block
    assert q.complete("nope", "zz", [], 8.0) == "unknown"
    assert q.leases_of(2) == [] and q.task("t0") is None
    q.forget("tournament:t")
    assert q.complete(lease2.lease_id, "t0", [], 9.0) == "late"             # remembered by id after forget


def test_attempts_node_loss_and_failure():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0")])
    for _ in range(MAX_BLOCK_ATTEMPTS - 1):
        (lease, _), = q.lease(1, 1, 0.0, 1.0)
        assert q.release(lease.lease_id, "t0", count=True, reason="crash") == "requeued"
    (lease, _), = q.lease(1, 1, 0.0, 1.0)
    assert q.drop_node(1) == ["t0"]                                          # node loss: no attempt used
    (lease, _), = q.lease(1, 1, 0.0, 1.0)
    assert q.release(lease.lease_id, "t0", reason="crash") == "failed"
    assert q.take_failed() == [("t0", "tournament:t", "crash")] and q.take_failed() == []
    assert q.task("t0") is None and q.complete(lease.lease_id, "t0", [], 0.0) == "late"
    assert q.release("other", "t0") == "ignored"


def test_keep_only_cancel_and_resubmit():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0"), _task("t1"), _task("t2"), _task("p0", "pulse:a:c", 1)])
    (a, _), (_b, _) = q.lease(1, 2, 0.0, 9.0)
    (c, _), = q.lease(2, 1, 0.0, 9.0)
    assert q.keep_only(1, [a.lease_id]) == ["t1"] and q.leases_of(1) == [a.lease_id]
    blocks, leases = q.cancel("tournament:t")
    assert blocks == ["t0", "t1", "t2"] and leases == {1: [a.lease_id], 2: [c.lease_id]}
    assert q.complete(a.lease_id, "t0", [], 0.0) == "cancelled"
    assert q.queued() == 1 and q.queued("pulse:") == 1 and q.queued("tournament:") == 0 and q.live("pulse:a:c") == 1
    q.submit([_task("t0")])                                                 # a cancelled block may be queued again
    with pytest.raises(ValueError, match="already"):
        q.submit([_task("t0")])
    with pytest.raises(ValueError, match="no lineups"):
        q.submit([BlockTask("e", "tournament", "tournament:t", (), (BOT,), 0, False, 1)])


def test_stats_count_every_outcome():
    q = BlockQueue(task_matches=1, new_id=_ids())
    q.submit([_task("t0"), _task("t1")])
    (lease, _), (_lease1, _) = q.lease(1, 2, 0.0, 1.0)
    q.complete(lease.lease_id, "t0", [], 0.0)
    q.complete(lease.lease_id, "t0", [], 0.0)
    q.expire(5.0)
    stats = q.stats()            # current states (queued, leased_now, leases_now) and cumulative counters
    assert (stats["submitted"], stats["leases"], stats["leased"], stats["completed"], stats["late"],
            stats["expired"], stats["requeued"], stats["queued"], stats["leased_now"],
            stats["leases_now"]) == (2, 2, 2, 1, 1, 1, 1, 1, 0, 0)
    assert set(stats) >= {"cancelled", "unknown", "failed", "cancelled_blocks"}      # every key always present


def test_check_block_results_accepts_exactly_the_block():
    t = _task("t0")

    def check(results):
        return check_block_results(t, results, spec=SPEC)

    assert check(_results(t)) is None
    assert "results for" in check(_results(t)[:1])
    repeated = _results(t)
    repeated[1].uid = repeated[0].uid
    assert "repeats" in check(repeated)
    seed = _results(t)
    seed[0].seed = 8
    assert "seed" in check(seed)
    source = _results(t)
    source[0].source = "train"
    assert "source" in check(source)
    seat = _results(t)
    seat[0].seats[1].agent_id = "x"
    assert "seat 1" in check(seat)
    seats = _results(t)
    seats[0].seats.pop()
    assert "seats" in check(seats)
    nan = _results(t)
    nan[0].teams[0].score = float("nan")
    assert "non-finite" in check(nan)
    kind = _results(t)
    kind[0].outcome_kind = "chess"
    assert "outcome kind" in check(kind)


def test_check_block_results_checks_roles_teams_and_versions():
    """P-PF4-14: a seat's role and team are the layout's; ``ckpt_v<N>`` seats carry version N, fixed ones none."""
    t = _task("t0")

    def check(results):
        return check_block_results(t, results, spec=SPEC)

    role = _results(t)
    role[0].seats[0].role = "other"
    assert "role" in check(role)
    team = _results(t)
    team[0].seats[0].team, team[0].seats[1].team = 1, 0
    assert "team" in check(team)
    teams = _results(t)
    teams[0].teams.pop()
    assert "teams" in check(teams)
    version = _results(t)
    version[0].seats[0].version_end = 2
    assert "version" in check(version)
    unversioned = _results(t)
    unversioned[0].seats[0].version_start = unversioned[0].seats[0].version_end = None
    assert "version" in check(unversioned)
    fixed = _results(t)
    fixed[0].seats[1].version_start = 3
    assert "version" in check(fixed)
    layout = _results(t)
    for r in layout:
        r.layout = "9p"
    assert "layout" in check(layout)
    junk = _results(t)
    junk[0].seats[0].seat = "zero"
    assert check(junk) is not None                     # never raises on node data

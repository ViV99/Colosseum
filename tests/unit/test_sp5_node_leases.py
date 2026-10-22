"""The node's evaluation bookkeeping (SP5 T5.1): slot requests as increments, leases waiting for their weights,
assignments to idle evaluators, outcomes kept until acked and resent after a reconnect, dead and stopped
evaluators, failed fetches, cancels, the evaluation blobs kept in the machine cache."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.net.messages import BlockTask, PlayerRef
from colosseum.node.leases import EVAL_BLOBS_KEPT, NodeLease, NodeLeases
from colosseum.node.weights import blob_key

KEY = blob_key("eval", "main", "pulse-main-ckpt_v3")


def _lease(lease_id, key=KEY, blocks=("b0",)):
    ref = PlayerRef(key.ident, key.agent_id, "ckpt_v3", ("player",), "neural", blob=key, networks={})
    tasks = [BlockTask(b, "pulse", "", (Lineup("2p", [SeatAssignment("main", "ckpt_v3")], seed=1, block=b,
                                               uid=f"{b}/r0"),), (ref,), 0, False, 2) for b in blocks]
    return NodeLease(lease_id, "pulse", tasks, 2, False)


def _two_idle():
    nl = NodeLeases()
    nl.evaluator_up(0)
    nl.evaluator_up(1)
    return nl


def test_request_asks_for_idle_evaluators_once():
    nl = _two_idle()
    assert nl.request(connected=False) == 0
    assert nl.request(connected=True) == 2 and nl.request(connected=True) == 0
    nl.on_lease(_lease("L1"), have=lambda k: True)
    assert nl.request(connected=True) == 0                     # 2 idle - 1 waiting lease - 1 still asked


def test_a_lease_waits_for_its_weights_then_an_idle_evaluator():
    nl = _two_idle()
    assert nl.on_lease(_lease("L1"), have=lambda k: False) == [KEY]
    assert nl.assignments() == []
    assert nl.on_fetched(KEY, True) == []
    ((index, lease),) = nl.assignments()
    assert (index, lease.lease_id, lease.state) == (0, "L1", "playing") and nl.idle() == [1]


def test_outcomes_stay_unacked_until_the_hub_acks_them():
    nl = _two_idle()
    nl.on_lease(_lease("L1", blocks=("b0", "b1")), have=lambda k: True)
    nl.assignments()
    outcomes = nl.on_done(0, "L1", {"b0": ["r0"], "b1": ["r1"]}, cancelled=False)
    assert [(o.block_id, o.results, o.failed) for o in outcomes] == [("b0", ["r0"], False), ("b1", ["r1"], False)]
    held, resend = nl.on_reconnect()
    assert held == ["L1"] and len(resend) == 2
    nl.on_ack("L1", ["b0", "b1"])
    assert nl.on_reconnect() == ([], [])


def test_done_without_a_block_reports_a_crash_for_it():
    nl = _two_idle()
    nl.on_lease(_lease("L1", blocks=("b0", "b1")), have=lambda k: True)
    nl.assignments()
    outcomes = nl.on_done(0, "L1", {"b0": ["r0"]}, cancelled=False)
    assert [(o.block_id, o.error_kind) for o in outcomes] == [("b0", ""), ("b1", "crash")]


def test_a_dead_or_stopped_evaluator_releases_its_lease():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: True)
    nl.on_lease(_lease("L2"), have=lambda k: True)
    nl.assignments()
    assert [(o.block_id, o.error_kind) for o in nl.evaluator_down(0)] == [("b0", "crash")]
    assert [(o.lease_id, o.error_kind) for o in nl.evaluator_down(1, error_kind="abandoned")] == [("L2", "abandoned")]
    assert nl.up() == [] and nl.evaluator_down(1) == []


def test_a_failed_fetch_reports_missing_weights():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: False)
    ((outcome,),) = [nl.on_fetched(KEY, False)]
    assert outcome.failed and outcome.error_kind == "missing_weights" and KEY.ident in outcome.error
    assert nl.assignments() == []


def test_cancel_frees_the_evaluator_only_when_it_answers():
    nl = _two_idle()
    nl.on_lease(_lease("L1"), have=lambda k: True)
    ((index, _lease_),) = nl.assignments()
    assert nl.on_cancel(["L1", "unknown"]) == [(index, "L1")] and nl.idle() == [1]
    assert nl.on_done(index, "L1", {}, cancelled=True) == [] and nl.idle() == [0, 1]
    assert nl.on_reconnect() == ([], [])


def test_unused_evaluation_blobs_beyond_the_newest_eight_are_dropped():
    nl = _two_idle()
    keys = [blob_key("eval", "p", f"b{n}") for n in range(11)]
    for n, key in enumerate(keys):
        nl.on_lease(_lease(f"X{n}", key=key), have=lambda k: True)
        nl.on_cancel([f"X{n}"])
    nl.on_lease(_lease("live", key=keys[0]), have=lambda k: True)      # b0 is in use again (a live lease)
    drop = nl.drop_candidates()                                          # 10 unused: the oldest 10 - 8 go
    assert drop == keys[1:1 + 10 - EVAL_BLOBS_KEPT] and keys[0] not in drop
    assert nl.drop_candidates() == []

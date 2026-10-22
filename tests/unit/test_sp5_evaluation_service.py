"""The hub's evaluation service (SP5 T5.1, spec 5.3) with owner sinks (the pulse comes in (b)): demand and
set_evaluators, joins with held leases, leases to asking nodes, acks, delivery once with the node stamped,
invalid results, typed errors, crashes, attempts, node loss, the waiting INFO, stop. No gRPC: ``send`` records."""
from __future__ import annotations

import logging

from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.evaluation import MAX_BLOCK_ATTEMPTS, HubEvaluation
from colosseum.net.messages import BlockTask, PlayerRef, block_task_from_proto, result_to_proto
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.players.registry import BotSpec
from game_helpers import make_match_result, make_test_config

BOT = PlayerRef("rnd", "rnd", "fixed", ("player",), "scripted", bot=BotSpec("colosseum.players.RandomBot"))
OWNER = "tournament:t"
SPEC = env_spec(make_test_config("turns"))


def _tasks(n, owner=OWNER):
    out = []
    for k in range(n):
        block = f"t/A/2p/a/rnd/{k}"
        lineups = tuple(Lineup("2p", [SeatAssignment("a", "ckpt_v1", False), SeatAssignment("rnd", "fixed", False)],
                               seed=100 + k, block=block, uid=f"{block}/r{r}") for r in range(2))
        out.append(BlockTask(block, "tournament", owner, lineups, (BOT,), 0, False, 1))
    return out


class Recorder:
    def __init__(self):
        self.sent: list[tuple[int, pb.HubMessage]] = []
        self.sink: list[tuple[str, str, object]] = []

    def send(self, node, msg):
        self.sent.append((node, msg))

    def of(self, kind):
        return [(n, getattr(m, kind)) for n, m in self.sent if m.WhichOneof("m") == kind]


def _service(rec, **kw):
    return HubEvaluation(lease_sec=600, send=rec.send, spec=SPEC, clock=kw.pop("clock", lambda: 0.0), **kw)


def _result_msg(lease_id, spec, *, drop=0):
    task = block_task_from_proto(spec, num_envs=1)
    results = [make_match_result([(0, "a", "ckpt_v1", "player", 1), (1, "rnd", "fixed")], uid=lu.uid,
                                 block=task.block_id, seed=lu.seed, source="tournament") for lu in task.lineups]
    return pb.BlockResult(lease_id=lease_id, block_id=task.block_id,
                          results=[result_to_proto(r) for r in results[:len(results) - drop]])


def test_join_returns_the_evaluator_count_from_the_demand_and_cancels_unknown_leases():
    rec = Recorder()
    ev = _service(rec)
    assert ev.node_joined(1, slots=2, held_leases=[], fresh=True).evaluators == 0      # no work can exist yet
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    assert [c.evaluators for _n, c in rec.of("command")] == [2]                        # set_evaluators 2
    join = ev.node_joined(2, slots=1, held_leases=["old-1", "old-2"], fresh=False)
    assert join.evaluators == 1
    (msg,) = join.after_welcome
    assert msg.command.kind == "cancel_leases" and list(msg.command.lease_ids) == ["old-1", "old-2"]
    ev.remove_owner(OWNER)
    assert sorted((n, c.evaluators) for n, c in rec.of("command") if c.kind == "set_evaluators") == \
        [(1, 0), (1, 2), (2, 0)]


def test_leases_are_granted_to_asking_nodes_and_results_reach_the_owner_once_with_the_node():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(3))
    assert rec.of("lease") == []                                           # nobody asked
    ev.on_lease_request(1, 1)
    ((node, lease),) = rec.of("lease")
    assert node == 1 and lease.num_envs == 1 and len(lease.blocks) == 2      # TASK_MATCHES x 1 env = 4 matches
    for spec in lease.blocks:
        ev.on_block_result(1, _result_msg(lease.lease_id, spec))
    ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[0]))     # a duplicate (a resend after a cut)
    assert [(k, key) for k, key, _v in rec.sink] == [("block", lease.blocks[0].block_id),
                                                     ("block", lease.blocks[1].block_id)]
    assert {r.node for _k, _b, results in rec.sink for r in results} == {1}
    assert len(rec.of("block_ack")) == 3 and ev.queue.stats()["late"] == 1


def test_invalid_results_are_replayed_and_warned_once_per_node(caplog):
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(2))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    with caplog.at_level(logging.WARNING, logger="colosseum.hub.evaluation"):
        ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[0], drop=1))
        ev.on_block_result(1, _result_msg(lease.lease_id, lease.blocks[1], drop=1))
    assert ev.invalid_results == 2 and ev.queue.queued() == 2 and rec.sink == []
    assert len([r for r in caplog.records if "invalid results" in r.getMessage()]) == 1
    assert len(rec.of("block_ack")) == 2                                     # acked whatever the outcome


def test_a_typed_block_error_fails_its_owner_and_a_crash_is_replayed():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(4))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    first, second = (s.block_id for s in lease.blocks)
    ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=first, failed=True, error_kind="crash"))
    assert ev.queue.task(first) is not None and ev.queue.queued() == 3          # back in the queue
    ev.on_block_result(1, pb.BlockResult(lease_id=lease.lease_id, block_id=second, failed=True,
                                         error_kind="player", error="bot raised on purpose"))
    ((kind, key, text),) = rec.sink
    assert (kind, key) == ("failed", "player") and "bot raised on purpose" in text and "node 1" in text
    assert ev.queue.queued() == 0 and ev.queue.live(OWNER) == 0                # the owner's blocks are gone


def test_a_block_that_fails_its_attempts_fails_its_owner():
    rec = Recorder()
    now = [0.0]
    ev = _service(rec, clock=lambda: now[0])
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(1))
    for _attempt in range(MAX_BLOCK_ATTEMPTS):
        ev.on_lease_request(1, 1)
        now[0] += 601.0                                                    # the lease expires (600 s)
        ev.tick()
    ((kind, key, text),) = rec.sink
    assert (kind, key) == ("failed", "attempts") and f"failed {MAX_BLOCK_ATTEMPTS} times" in text
    assert "lease expired" in text and ev.queue.stats()["expired"] == MAX_BLOCK_ATTEMPTS


def test_node_loss_and_a_fresh_session_release_the_leases_without_an_attempt():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.node_joined(2, slots=1, held_leases=[], fresh=True)
    ev.submit(_tasks(2))
    ev.on_lease_request(1, 1)
    ((_n, lease),) = rec.of("lease")
    ev.on_lease_request(2, 1)                                              # node 2 waits: nothing queued
    ev.node_lost(1)                                                        # node 1's blocks go to node 2 at once
    assert [n for n, _l in rec.of("lease")] == [1, 2] and ev.queue.stats()["requeued"] == 2
    ev.node_joined(2, slots=1, held_leases=[], fresh=True)                 # node 2's process restarted
    assert ev.queue.queued() == 2 and ev.queue.stats()["failed"] == 0


def test_a_resumed_session_keeps_its_held_leases_only():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=2, held_leases=[], fresh=True)
    ev.submit(_tasks(4))
    ev.on_lease_request(1, 2)
    (_n, a), (_n2, _b) = rec.of("lease")
    ev.session_closed(1)
    join = ev.node_joined(1, slots=2, held_leases=[a.lease_id], fresh=False)
    assert join.after_welcome == [] and ev.queue.leases_of(1) == [a.lease_id] and ev.queue.queued() == 2


def test_blocks_wait_with_one_info_while_no_node_runs_an_evaluator(caplog):
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.submit(_tasks(2))
    with caplog.at_level(logging.INFO, logger="colosseum.hub.evaluation"):
        ev.tick()
        ev.tick()
    assert len([r for r in caplog.records if "no connected node runs an evaluator" in r.getMessage()]) == 1
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.tick()
    ev.node_lost(1)
    with caplog.at_level(logging.INFO, logger="colosseum.hub.evaluation"):
        ev.tick()
    assert len([r for r in caplog.records if "no connected node runs an evaluator" in r.getMessage()]) == 2


def test_stop_ends_the_dispatch():
    rec = Recorder()
    ev = _service(rec)
    ev.add_owner(OWNER, lambda *a: rec.sink.append(a))
    ev.node_joined(1, slots=1, held_leases=[], fresh=True)
    ev.stop()
    ev.submit(_tasks(1))
    ev.on_lease_request(1, 1)
    ev.tick()
    assert rec.of("lease") == [] and ev.stats()["queued"] == 1


def test_stats_have_every_key():
    ev = _service(Recorder())
    assert set(ev.stats()) == {"queued", "leased_now", "leases_now", "submitted", "leases", "leased", "completed",
                               "late", "cancelled", "unknown", "expired", "requeued", "failed", "cancelled_blocks",
                               "invalid_results", "evaluators", "pulse_results"}

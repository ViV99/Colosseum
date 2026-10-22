"""The hub's collect rule across machines (SP5 T3.2, spec 4.7, 4.8): an agent collects on a node when its learner
runs, on that node or anywhere with the relay on, or on a node with a direct path to it with the relay off; and the
learner node's loopback data addresses reach only nodes on its own machine."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import VersionInfo, collect_allowed, strip_collect, visible_version


def test_collect_needs_a_running_learner_and_a_way_to_it():
    assert collect_allowed(1, None, relay=True, path=None) is False          # no learner (pending, failed)
    assert collect_allowed(1, 1, relay=False, path=None) is True             # the learner is on this node
    assert collect_allowed(1, 2, relay=True, path=None) is True              # direct or relayed
    assert collect_allowed(1, 2, relay=False, path="direct") is True
    assert collect_allowed(1, 2, relay=False, path="down") is False
    assert collect_allowed(1, 2, relay=False, path=None) is False            # not reported yet: not granted


def test_strip_collect_keeps_the_seats_and_turns_off_only_disallowed_collection():
    lineup = Lineup("2p", [SeatAssignment("a", "latest", True, "owner"), SeatAssignment("b", "latest", True, "rivals"),
                           SeatAssignment("b", "ckpt_v4", False, "snapshots")], seed=None, epoch=7)
    out = strip_collect(lineup, {"a"})
    assert [(s.agent_id, s.network_id, s.collect, s.source) for s in out.seats] == [
        ("a", "latest", True, "owner"), ("b", "latest", False, "rivals"), ("b", "ckpt_v4", False, "snapshots")]
    assert out.epoch == 7 and lineup.seats[1].collect is True                # a copy; the input is unchanged


def test_loopback_data_addresses_reach_only_the_learner_nodes_machine():
    """Carried decision 3a of T3.1: a node joined through a loopback hub address announces 127.0.0.1 first; nodes on
    other machines never get it (a loopback-only node is relay-only for them)."""
    info = VersionInfo(agent_id="main", generation=1, version=3, size=4, digest="d", node=2,
                       addresses=("127.0.0.1:47102", "10.0.0.5:47102", "[::1]:47102", "localhost:47102"))
    assert visible_version(info, same_machine=True) is info
    other = visible_version(info, same_machine=False)
    assert other.addresses == ("10.0.0.5:47102",) and not other.relay_only
    only_loopback = VersionInfo(agent_id="main", generation=1, version=3, size=4, digest="d", node=2,
                                addresses=("127.0.0.1:47102",))
    seen = visible_version(only_loopback, same_machine=False)
    assert seen.addresses == () and seen.relay_only
    parked = VersionInfo(agent_id="main", generation=0, version=3, size=4, digest="d", node=-1, relay_only=True)
    assert visible_version(parked, same_machine=False) is parked


def test_seats_of_unreachable_learners_play_their_newest_snapshot_or_the_lineup_is_impossible():
    """Controller ruling R37: with the relay off a node that cannot reach A's learner plays A only from stored
    snapshots (no collecting); without one the lineup cannot be played there."""
    from colosseum.hub.state import seat_snapshots

    lineup = Lineup("2p", [SeatAssignment("b", "latest", True, "owner"), SeatAssignment("a", "latest", True, "rivals")],
                       seed=None, epoch=3)
    out = seat_snapshots(lineup, {"a": "ckpt_v40"})
    assert [(s.agent_id, s.network_id, s.collect, s.source) for s in out.seats] == [
        ("b", "latest", True, "owner"), ("a", "ckpt_v40", False, "rivals")]
    assert out.epoch == 3 and lineup.seats[1].network_id == "latest"
    assert seat_snapshots(lineup, {"a": None}) is None
    assert seat_snapshots(lineup, {"c": None}) is lineup                    # no seat of the agent: unchanged
    snapshot_seat = Lineup("2p", [SeatAssignment("b"), SeatAssignment("a", "ckpt_v2", False, "snapshots")], seed=None)
    assert seat_snapshots(snapshot_seat, {"a": None}) is snapshot_seat      # a stored snapshot already

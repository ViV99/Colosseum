"""SnapshotHolds (SP5 T3.1, spec 4.2): an evicted snapshot stays downloadable while a granted lineup names it, and at
most EVICTED_HOLD_FACTOR x match_refresh_interval_sec after its eviction. A cell's references are those of its last
two grants (pre-flight ruling P-PF3-16: a cell that re-requests while its previous grant's download is in flight
keeps that snapshot held)."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import EVICTED_HOLD_FACTOR, CellKey, SnapshotHolds, snapshot_refs

A, B = CellKey(1, 0, 0), CellKey(1, 0, 1)


def _lineup(*networks):
    return Lineup("2p", [SeatAssignment("main", n, collect=n == "latest") for n in networks])


def test_snapshot_refs_lists_only_snapshot_seats():
    assert snapshot_refs(_lineup("latest", "ckpt_v40")) == {("main", "ckpt_v40")}
    assert snapshot_refs(Lineup("2p", [SeatAssignment("random", "fixed", collect=False)])) == frozenset()


def test_references_follow_the_last_two_grants_of_each_cell():
    h = SnapshotHolds(30.0)
    h.grant(A, {("main", "ckpt_v40")})
    h.grant(B, {("main", "ckpt_v40")})
    assert h.referenced("main", "ckpt_v40")
    h.grant(A, {("main", "ckpt_v60")})                            # A's previous grant still counts (P-PF3-16)
    assert h.referenced("main", "ckpt_v40") and h.referenced("main", "ckpt_v60")
    h.drop_cell(B)
    assert h.referenced("main", "ckpt_v40")                       # A's previous grant
    h.grant(A, {("main", "ckpt_v60")})                            # two grants later A no longer names v40
    assert not h.referenced("main", "ckpt_v40") and h.referenced("main", "ckpt_v60")
    h.drop_node(1)
    assert not h.referenced("main", "ckpt_v60")


def test_a_hold_ends_when_no_lineup_names_the_snapshot_or_at_its_deadline():
    h = SnapshotHolds(10.0)
    assert h.max_hold_sec == EVICTED_HOLD_FACTOR * 10.0
    h.grant(A, {("main", "ckpt_v40")})
    h.grant(B, {("main", "ckpt_v50")})
    h.hold("main", "ckpt_v40", now=100.0)
    h.hold("main", "ckpt_v50", now=100.0)
    assert h.due(now=101.0) == []
    h.grant(A, ())
    assert h.due(now=101.0) == []                                 # still named by A's previous grant
    h.grant(A, ())                                                # A's last two grants no longer name v40
    assert h.due(now=101.0) == [("main", "ckpt_v40")]
    assert h.is_held("main", "ckpt_v50") and not h.is_held("main", "ckpt_v40")
    assert h.due(now=119.9) == []
    assert h.due(now=120.0) == [("main", "ckpt_v50")]             # still named, but the deadline passed
    assert h.held() == []


def test_without_a_match_refresh_nothing_is_held():
    h = SnapshotHolds(0.0)                                        # match_refresh_interval_sec: 0
    h.grant(A, {("main", "ckpt_v40")})
    assert h.max_hold_sec == 0.0 and not h.referenced("main", "ckpt_v40")

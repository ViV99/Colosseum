"""Pure hub state (SP5 T2.1; spec 4.1-4.5, 4.7, 7.2, 7.6; rulings R7, R8, R16): node registry, epochs reserved in
blocks, cells, owner rotation by deficit (it reproduces SP4's rotation on one node), the budget per (node, session),
versions and generations with the cap rule, result checks and batch bookkeeping, hub_state.json."""
from __future__ import annotations

import dataclasses
import json
import math
import random

import pytest

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import (
    EPOCH_BLOCK,
    HUB_STATE_FILE,
    HUB_STATE_VERSION,
    BudgetCounter,
    CellKey,
    CellTable,
    EpochAllocator,
    HubStateData,
    NodeHello,
    NodeRegistry,
    OwnerRotation,
    ResultIntake,
    VersionInfo,
    VersionRegistry,
    load_hub_state,
    restrict_collecting,
    save_hub_state,
)
from colosseum.metrics.counters import CounterBook
from colosseum.net import messages
from game_helpers import TurnTakingGame, make_match_result, owner_sequence

HELLO = NodeHello(hardware={"logical_cores": 4}, slots={"workers": 3}, tags=("gpu",), data_addresses=())


# ---- nodes ----------------------------------------------------------------------------------------------------


def test_a_node_is_new_then_resumed_with_its_session_then_returned_after_a_restart():
    reg = NodeRegistry()
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0, serial=1) == "new"
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=1.0, serial=2) == "resumed"   # the stream was cut and came back
    assert reg.connect("n1", 1, "pc", "s2", HELLO, now=2.0, serial=3) == "returned"  # the node process restarted
    entry = reg.by_id("n1")
    assert (entry.number, entry.name, entry.session_id, entry.state, entry.tags) == (1, "pc", "s2", "online", ("gpu",))
    assert (entry.serial, entry.connected, entry.data_path) == (3, True, "relay")
    assert reg.get(1) is entry and reg.get(2) is None and reg.online() == [entry] and reg.all() == [entry]


def test_a_closed_stream_is_not_a_loss_only_silence_is():
    """P-PF2-1 (b): a close keeps the session id; the same session id comes back as ``resumed``."""
    reg = NodeRegistry()
    reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0, serial=1)
    reg.disconnect("n1", 1)
    entry = reg.by_id("n1")
    assert (entry.state, entry.session_id, entry.connected) == ("online", "s1", False)
    assert reg.lost(now=5.0, after_sec=10.0) == []
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=6.0, serial=2) == "resumed"   # back within node_lost_after_sec
    assert reg.by_id("n1").connected and reg.heartbeat("n1", "s1", now=7.0)
    assert [e.node_id for e in reg.lost(now=17.5, after_sec=10.0)] == ["n1"]
    assert not reg.by_id("n1").connected
    assert reg.lost(now=30.0, after_sec=10.0) == []                          # each loss once
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=31.0, serial=3) == "returned"  # a lost node: returned


def test_a_late_close_of_the_old_stream_after_the_new_open_changes_nothing():
    """P-PF2-1 (b): the hub learns of a cut late; the old stream's close arrives after the new stream's open."""
    reg = NodeRegistry()
    reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0, serial=1)
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=0.5, serial=2) == "resumed"
    reg.disconnect("n1", 1)                                                  # the superseded stream's close
    entry = reg.by_id("n1")
    assert (entry.serial, entry.session_id, entry.connected) == (2, "s1", True)
    assert reg.heartbeat("n1", "s1", now=1.0) and entry.last_heartbeat == 1.0
    reg.disconnect("nx", 2)                                                  # an unknown node: nothing
    reg.disconnect("n1", 2)
    assert not entry.connected and entry.state == "online"


def test_heartbeats_of_another_session_or_an_unknown_node_are_ignored():
    reg = NodeRegistry()
    reg.connect("n1", 1, "pc", "s1", HELLO, now=0.0)
    assert not reg.heartbeat("n1", "old", now=9.0) and not reg.heartbeat("nx", "s1", now=9.0)
    assert reg.by_id("n1").last_heartbeat == 0.0
    assert reg.remove("n1") is reg.by_id("n1") and reg.remove("nx") is None
    assert reg.by_id("n1").state == "removed" and reg.online() == [] and not reg.heartbeat("n1", "s1", now=10.0)
    assert reg.connect("n1", 1, "pc", "s1", HELLO, now=11.0) == "returned"   # a removed node never resumes


def test_the_data_path_and_the_hello_are_kept():
    reg = NodeRegistry()
    reg.connect("n0", 0, "local", "s", NodeHello(data_addresses=("10.0.0.1:7",)), now=0.0, data_path="local")
    entry = reg.by_id("n0")
    assert (entry.data_path, entry.data_addresses, entry.hardware, entry.serial) == ("local", ("10.0.0.1:7",), {}, None)


# ---- epochs and cells -------------------------------------------------------------------------------------------


def test_epochs_are_reserved_in_blocks_before_they_are_handed_out():
    reserved: list[int] = []
    epochs = EpochAllocator(0, reserved.append)
    assert epochs.next() == 1 and reserved == [EPOCH_BLOCK]
    handed = [epochs.next() for _ in range(EPOCH_BLOCK - 1)]
    assert handed[-1] == EPOCH_BLOCK and reserved == [EPOCH_BLOCK]
    assert epochs.next() == EPOCH_BLOCK + 1 and reserved == [EPOCH_BLOCK, 2 * EPOCH_BLOCK]


def test_a_resumed_allocator_starts_after_the_saved_reservation():
    reserved: list[int] = []
    epochs = EpochAllocator(EPOCH_BLOCK, reserved.append)
    assert epochs.next() == EPOCH_BLOCK + 1 and reserved == [2 * EPOCH_BLOCK]


def test_a_failing_reservation_hands_out_no_epoch():
    calls = []

    def on_reserve(until):
        calls.append(until)
        if len(calls) == 1:
            raise OSError("disk full")

    epochs = EpochAllocator(0, on_reserve)
    with pytest.raises(OSError):
        epochs.next()
    assert epochs.reserved_until == 0
    assert epochs.next() == 1 and calls == [EPOCH_BLOCK, EPOCH_BLOCK]


def _table(reserved_until: int = 0) -> CellTable:
    return CellTable(EpochAllocator(reserved_until, lambda until: None))


def test_a_cell_keeps_its_epoch_within_an_incarnation_and_gets_a_new_one_otherwise():
    table, key = _table(), CellKey(1, 0, 2)
    (first,) = table.request(1, [(key, 0, 1)], now=0.0)
    (again,) = table.request(1, [(key, first.epoch, 1)], now=1.0)
    assert again.epoch == first.epoch
    (restarted,) = table.request(1, [(key, first.epoch, 2)], now=2.0)       # a new worker incarnation
    assert restarted.epoch > first.epoch and restarted.incarnation == 2
    (unknown,) = table.request(1, [(key, 999, 2)], now=3.0)                 # an epoch this table never granted
    assert unknown.epoch > restarted.epoch


def test_a_new_hub_process_never_keeps_an_epoch_of_the_old_one():
    (entry,) = _table(EPOCH_BLOCK).request(1, [(CellKey(1, 0, 0), 5, 1)], now=0.0)
    assert entry.epoch == EPOCH_BLOCK + 1


def test_cells_are_per_node_and_a_lost_node_takes_only_its_own():
    table = _table()
    table.request(1, [(CellKey(1, 0, e), 0, 1) for e in range(2)], now=0.0)
    table.request(2, [(CellKey(2, 0, 0), 0, 1)], now=0.0)
    with pytest.raises(ValueError):
        table.request(1, [(CellKey(2, 0, 1), 0, 1)], now=0.0)
    with pytest.raises(ValueError):                                         # one foreign key refuses the request
        table.request(1, [(CellKey(1, 1, 0), 0, 1), (CellKey(2, 0, 1), 0, 1)], now=0.0)
    assert table.get(CellKey(1, 1, 0)) is None
    (fresh,) = table.request(1, [(CellKey(1, 1, 0), 0, 1)], now=0.0)
    assert fresh.epoch == 4                                                 # no epoch was burnt by the refusal
    table.drop_cells([CellKey(1, 1, 0)])
    assert [e.key for e in table.drop_node(1)] == [CellKey(1, 0, 0), CellKey(1, 0, 1)]
    assert [e.key for e in table.live()] == [CellKey(2, 0, 0)]


def test_a_grant_moves_the_owner_to_previous_and_a_new_incarnation_keeps_it_as_previous():
    table, key = _table(), CellKey(1, 0, 0)
    table.request(1, [(key, 0, 1)], now=0.0)
    table.grant(key, "a", 30.0, now=0.0)
    entry = table.grant(key, "b", 30.0, now=30.0)
    assert (entry.owner, entry.previous, entry.ttl_sec, entry.granted_at) == ("b", "a", 30.0, 30.0)
    assert table.owners() == {key: "b"} and table.owners(exclude=[key]) == {}
    (fresh,) = table.request(1, [(key, entry.epoch, 2)], now=31.0)
    assert (fresh.owner, fresh.previous) == (None, "b")


# ---- owner rotation ---------------------------------------------------------------------------------------------


def _agent_ids(n: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(n)]                       # owner_sequence's names


@pytest.mark.parametrize("n_agents", [1, 2, 3, 4])
@pytest.mark.parametrize("workers,envs", [(1, 1), (1, 3), (2, 4), (3, 8), (2, 5)])
def test_one_node_with_simultaneous_expiry_reproduces_the_sp4_rotation(n_agents, workers, envs):
    agents = _agent_ids(n_agents)
    rotation, rounds = OwnerRotation(agents), 6
    expected = owner_sequence(workers, envs, n_agents, rounds)          # SP4's trainable[(g + round) % n]
    cells = [CellKey(1, w, e) for w in range(workers) for e in range(envs)]
    previous: dict[CellKey, str] = {}
    for rnd in range(rounds):
        owners = rotation.assign(cells, collecting=agents, owners={}, previous=previous)
        assert owners == {c: expected[rnd][c.worker][c.env] for c in cells}, rnd
        previous = owners


@pytest.mark.parametrize("n_agents", [2, 3])
def test_workers_that_request_one_after_another_keep_the_shares_balanced(n_agents):
    """Staggered expiry: the start reproduces SP4 (workers start one after another); later rounds keep the shares
    within one cell of each other (the deficit rule prefers balance over SP4's exact sequence)."""
    agents = _agent_ids(n_agents)
    rotation, workers, envs = OwnerRotation(agents), 3, 4
    start = owner_sequence(workers, envs, n_agents, 1)[0]
    current: dict[CellKey, str] = {}
    for rnd in range(5):
        for w in range(workers):                                   # each worker's cells expire on their own
            mine = [CellKey(1, w, e) for e in range(envs)]
            others = {k: v for k, v in current.items() if k.worker != w}
            got = rotation.assign(mine, collecting=agents, owners=others, previous=current)
            if rnd == 0:
                assert got == {c: start[w][c.env] for c in mine}, w
            current.update(got)
            counts = [list(current.values()).count(a) for a in agents]
            if rnd > 0:
                assert max(counts) - min(counts) <= 1, (rnd, w, counts)


def test_shares_stay_balanced_while_nodes_join_and_leave():
    agents = ["a", "b", "c"]
    rotation, rng = OwnerRotation(agents), random.Random(0)
    current: dict[CellKey, str] = {}
    nodes = {1: 5}
    for step in range(40):
        if step % 7 == 3:
            nodes[len(nodes) + 1] = rng.randint(1, 9)             # a node joins
        if step % 11 == 10 and len(nodes) > 1:
            gone = rng.choice(sorted(nodes))
            del nodes[gone]                                        # a node is lost: its cells vanish
            current = {k: v for k, v in current.items() if k.node != gone}
        cells = [CellKey(n, 0, e) for n in sorted(nodes) for e in range(nodes[n])]
        current = rotation.assign(cells, collecting=agents, owners={}, previous=current)   # a full refresh
        counts = [list(current.values()).count(a) for a in agents]
        assert max(counts) - min(counts) <= 1, (step, counts)


def test_new_cells_go_to_the_agents_that_own_the_fewest():
    rotation = OwnerRotation(["a", "b", "c"])
    others = {CellKey(1, 0, e): o for e, o in enumerate("aaabbc")}
    got = rotation.assign([CellKey(2, 0, 0), CellKey(2, 0, 1), CellKey(2, 0, 2)], collecting=["a", "b", "c"],
                          owners=others, previous={})
    assert list(got.values()) == ["c", "b", "c"]


def test_only_collecting_agents_own_cells_and_without_any_there_is_no_owner():
    rotation = OwnerRotation(["a", "b", "c"])
    cells = [CellKey(1, 0, e) for e in range(4)]
    assert rotation.assign(cells, collecting=[], owners={}, previous={}) == {}
    got = rotation.assign(cells, collecting=["c", "a"], owners={CellKey(2, 0, 0): "b"}, previous={})
    assert list(got.values()) == ["a", "c", "a", "c"]                     # config order, b ignored


def test_restrict_collecting_turns_off_seats_of_agents_without_a_learner():
    lineup = Lineup("2p", [SeatAssignment("a", collect=True, source="owner"),
                           SeatAssignment("b", collect=True, source="rivals")], seed=None)
    assert restrict_collecting(lineup, ["a", "b"]) is lineup
    out = restrict_collecting(lineup, ["a"])
    assert [s.collect for s in out.seats] == [True, False] and [s.collect for s in lineup.seats] == [True, True]
    assert out.seats[1].source == "rivals" and out.layout == "2p"


# ---- budget -------------------------------------------------------------------------------------------------


def test_the_budget_counts_increments_per_session_and_never_twice():
    budget = BudgetCounter()
    assert budget.report("n1", "s1", 100) == 100 and budget.report("n1", "s1", 150) == 50
    assert budget.report("n1", "s1", 120) == 0                     # out of order / repeated: nothing
    assert budget.report("n1", "s2", 30) == 30                    # the node restarted: a new session from 0
    assert budget.report("n2", "s1", 7) == 7
    assert budget.value == 187
    assert budget.state() == {"env_steps": 187, "sessions": {"n1/s1": 150, "n1/s2": 30, "n2/s1": 7}}


def test_a_resumed_budget_continues_from_the_saved_sessions():
    saved = BudgetCounter()
    saved.report("n1", "s1", 150)
    budget = BudgetCounter(saved.state()["env_steps"], saved.state()["sessions"])
    assert budget.report("n1", "s1", 170) == 20 and budget.report("n1", "s9", 5) == 5 and budget.value == 175
    budget.add(25)
    assert budget.value == 200
    with pytest.raises(ValueError):
        BudgetCounter(-1)


# ---- versions and generations -------------------------------------------------------------------------------


def _info(version: int, generation: int, agent: str = "a") -> VersionInfo:
    return VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d" * 32, node=1)


def test_versions_grow_and_come_only_from_the_current_generation():
    reg = VersionRegistry()
    assert reg.generation("a") == 0 and reg.version_floor("a") == 0 and reg.latest("a") is None
    assert reg.announce(_info(0, 1)) == "invalid"                  # no generation assigned yet
    assert reg.new_generation("a") == 1
    assert reg.announce(_info(0, 1)) == "accepted" and reg.announce(_info(0, 1)) == "not_newer"
    assert reg.announce(_info(5, 1)) == "accepted" and reg.announce(_info(6, 2)) == "invalid"
    assert reg.new_generation("a") == 2
    assert reg.announce(_info(6, 1)) == "stale_generation"         # the old learner after a network split
    assert reg.announce(_info(3, 2)) == "not_newer"                # versions never go back (R9)
    assert reg.version_floor("a") == 6
    assert reg.announce(_info(6, 2)) == "accepted" and reg.latest("a") == _info(6, 2)


def test_a_superseded_generation_is_stale_only_above_its_cap():
    reg = VersionRegistry()
    reg.new_generation("a")
    for v in (0, 3, 5):
        reg.announce(_info(v, 1))
    reg.new_generation("a")
    assert not reg.is_stale("a", 1, 5) and not reg.is_stale("a", 1, 0)
    assert reg.is_stale("a", 1, 6) and reg.is_stale("a", 1, None)
    assert not reg.is_stale("a", 2, 99) and not reg.is_stale("a", None, 99) and not reg.is_stale("a", 0, 99)
    assert reg.is_stale("a", 3, 1) and reg.is_stale("other", 1, 0)
    reg.new_generation("a")                                        # generation 2 announced nothing: cap -1
    assert reg.is_stale("a", 2, 0)


def test_invalid_announcements_are_refused():
    reg = VersionRegistry()
    reg.new_generation("a")
    for bad in (dataclasses.replace(_info(1, 1), version=-1), dataclasses.replace(_info(1, 1), digest=""),
                dataclasses.replace(_info(1, 1), size=-5), dataclasses.replace(_info(1, 1), generation=0)):
        assert reg.announce(bad) == "invalid"


def test_the_version_registry_round_trips_through_its_state():
    reg = VersionRegistry()
    reg.new_generation("a")
    reg.announce(_info(4, 1))
    reg.new_generation("a")
    reg.announce(_info(7, 2))
    back = VersionRegistry.from_state(json.loads(json.dumps(reg.state())))
    assert back.state() == reg.state() and back.latest("a") == _info(7, 2)
    assert back.is_stale("a", 1, 5) and not back.is_stale("a", 1, 4) and back.version_floor("a") == 8


def test_version_info_is_the_net_messages_dataclass():
    assert VersionInfo is messages.VersionInfo                     # A7: re-exported, never redefined
    info = dataclasses.replace(_info(3, 1), addresses=("h:1",), relay_only=True)
    assert VersionInfo.from_dict(json.loads(json.dumps(info.to_dict()))) == info


# ---- results ------------------------------------------------------------------------------------------------

SPEC = TurnTakingGame().spec
AGENTS = ("a", "random")
PREFIX = "R/train/n3/"


def _result(**changes):
    seats = [(0, "a", "latest", "player", 4, 5), (1, "random", "fixed", "player", None, None)]
    result = make_match_result(seats, uid=f"{PREFIX}w0_e1_g7_ep2", match_id="w0_e1_g7_ep2")
    result.node = changes.pop("node", 3)
    for seat in result.seats:
        seat.generation = 1 if seat.network_id == "latest" else None
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _versions() -> VersionRegistry:
    reg = VersionRegistry()
    reg.new_generation("a")
    return reg


def _check(result, versions=None):
    return ResultIntake(journal=True).check(result, spec=SPEC, agents=AGENTS, versions=versions or _versions(),
                                            node=3, uid_prefix=PREFIX)


def test_a_valid_result_passes_the_checks():
    assert _check(_result()) is None


def _seat(i, **changes):
    def mutate(result):
        for key, value in changes.items():
            setattr(result.seats[i], key, value)
        return result
    return mutate


@pytest.mark.parametrize("mutate", [
    lambda r: setattr(r, "layout", "9p") or r,
    lambda r: setattr(r, "outcome_kind", "rank") or r,
    lambda r: setattr(r, "seats", r.seats[:1]) or r,
    lambda r: setattr(r, "seats", None) or r,                       # raises inside: still only "invalid"
    lambda r: setattr(r, "source", "pulse") or r,
    lambda r: setattr(r, "node", 4) or r,
    lambda r: setattr(r, "uid", "R/train/n4/w0_e1_g7_ep2") or r,
    lambda r: setattr(r, "uid", PREFIX) or r,
    lambda r: setattr(r, "episode_length", -1) or r,
    lambda r: setattr(r.teams[0], "rank", math.nan) or r,
    lambda r: setattr(r.teams[1], "team", 5) or r,
    _seat(0, seat=1), _seat(0, role="hunter"), _seat(1, team=0), _seat(0, agent_id="ghost"),
    _seat(0, network_id="ckpt_vx"), _seat(0, reward=math.inf), _seat(0, version_end=-1),
    _seat(0, generation=-1), _seat(1, generation=-3), _seat(0, generation="1"),
])
def test_every_malformed_result_is_invalid_and_nothing_raises(mutate):
    assert _check(mutate(_result())) == "invalid"


def test_a_latest_seat_played_by_a_superseded_generation_above_its_cap_is_stale():
    versions = _versions()
    versions.announce(_info(5, 1))
    versions.new_generation("a")
    assert _check(_result(), versions) is None                     # version_end 5 <= cap(1) = 5
    stale = _result()
    stale.seats[0].version_end = 6
    assert _check(stale, versions) == "stale_generation"


def test_batches_are_new_once_and_acked_only_after_their_commit():
    intake = ResultIntake(journal=True)
    assert not intake.accounts_at_intake and ResultIntake(journal=False).accounts_at_intake
    assert intake.begin("n", "s", 1) and not intake.begin("n", "s", 1)      # in flight: a resend is skipped
    assert not intake.is_acked("n", "s", 1)
    intake.committed("n", "s", 1)
    assert intake.is_acked("n", "s", 1) and not intake.begin("n", "s", 1)
    assert intake.begin("n", "s", 2)
    intake.failed("n", "s", 2)                                    # the writer could not take it
    assert not intake.is_acked("n", "s", 2) and intake.begin("n", "s", 2)  # the node's resend is new again
    assert intake.begin("n", "other-session", 1)


def test_the_ack_watermark_advances_over_out_of_order_commits():
    intake = ResultIntake(journal=True)
    for seq in (1, 2, 3):
        intake.begin("n", "s", seq)
    intake.committed("n", "s", 3)
    intake.committed("n", "s", 1)
    assert intake.is_acked("n", "s", 3) and not intake.is_acked("n", "s", 2)
    intake.committed("n", "s", 2)
    seqs = intake._sessions[("n", "s")]
    assert (seqs.low, seqs.acked, seqs.in_flight) == (3, set(), set())


def test_open_takes_the_nodes_last_acked_seq():
    intake = ResultIntake(journal=True)
    intake.open("n", "s", 5)
    assert intake.is_acked("n", "s", 3) and not intake.begin("n", "s", 5) and intake.begin("n", "s", 6)


def test_drops_are_counted_by_reason_and_warned_once_per_node_and_reason():
    intake = ResultIntake(journal=True)
    assert intake.count_drop(3, "invalid", 2) and not intake.count_drop(3, "invalid")
    assert intake.count_drop(4, "invalid") and intake.count_drop(3, "stale_generation")
    assert intake.dropped == {"invalid": 4, "stale_generation": 1}
    with pytest.raises(ValueError, match="unknown reason"):
        intake.count_drop(3, "writer")
    assert intake.dropped == {"invalid": 4, "stale_generation": 1}


# ---- hub_state.json -----------------------------------------------------------------------------------------


def _counters() -> dict:
    book, intake_book = CounterBook(), CounterBook()
    book.report("n1/s1", {"lineup_fetch_failed": 2})
    book.add("dropped_match_results_by_reason.invalid", 1)
    intake_book.report("learner/a/1", {"dropped_chunks.invalid": 3})
    return {"book": book.state(), "intake_book": intake_book.state()}       # A3


def _data(**changes) -> HubStateData:
    base = dict(version=HUB_STATE_VERSION, run_id="R", env_steps=187, sessions={"n1/s1": 150},
                epoch_reserved_until=EPOCH_BLOCK, versions={}, counters=_counters(), nodes_lost=1)
    return HubStateData(**{**base, **changes})


@pytest.mark.parametrize("fsync", [False, True])
def test_hub_state_round_trips_atomically(tmp_path, fsync):
    run_dir = tmp_path / "run"                                     # tmp_path also holds the isolated XDG dirs
    run_dir.mkdir()
    path = run_dir / HUB_STATE_FILE
    save_hub_state(path, _data(), fsync=fsync)
    assert load_hub_state(path) == _data()
    save_hub_state(path, _data(env_steps=200), fsync=fsync)       # an overwrite replaces the file whole
    assert load_hub_state(path) == _data(env_steps=200)
    assert sorted(p.name for p in run_dir.iterdir()) == [HUB_STATE_FILE]


def test_hub_state_counters_are_saved_and_loaded_whole(tmp_path):
    """A3 / D16: both books come back with CounterBook.from_state; unknown keys survive a resume."""
    path = tmp_path / HUB_STATE_FILE
    versions = VersionRegistry()
    versions.new_generation("a")
    versions.announce(_info(4, 1))
    counters = {**_counters(), "system_stats": {"future": [1, 2]}}
    save_hub_state(path, _data(counters=counters, versions=versions.state()), fsync=False)
    back = load_hub_state(path)
    assert back.counters == counters
    assert CounterBook.from_state(back.counters["book"]).totals() == {
        "lineup_fetch_failed": 2, "dropped_match_results_by_reason.invalid": 1}
    assert CounterBook.from_state(back.counters["intake_book"]).get("dropped_chunks.invalid") == 3
    assert VersionRegistry.from_state(back.versions).latest("a") == _info(4, 1)
    on_disk = json.loads(path.read_text())
    assert sorted(on_disk) == ["counters", "env_steps", "epoch_reserved_until", "nodes_lost", "run_id", "sessions",
                               "version", "versions"]


def test_a_missing_hub_state_is_none_and_a_malformed_one_a_value_error(tmp_path):
    path = tmp_path / HUB_STATE_FILE
    assert load_hub_state(path) is None
    path.write_text("{not json")
    with pytest.raises(ValueError, match="malformed"):
        load_hub_state(path)
    path.write_text(json.dumps({**dataclasses.asdict(_data()), "version": 99}))
    with pytest.raises(ValueError, match="version 99"):
        load_hub_state(path)


def test_an_epoch_reservation_survives_a_crash_of_the_hub(tmp_path):
    path = tmp_path / HUB_STATE_FILE
    reg = VersionRegistry()

    def reserve(until: int) -> None:
        save_hub_state(path, _data(epoch_reserved_until=until, versions=reg.state()), fsync=True)

    first = EpochAllocator(0, reserve)
    handed = [first.next() for _ in range(3)]                      # then the hub is killed: nothing else saved
    resumed = EpochAllocator(load_hub_state(path).epoch_reserved_until, reserve)
    assert resumed.next() > max(handed)

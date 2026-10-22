"""Learner placement and the learner table (SP5 T4.1; spec blocks 1, 2.6, 7.1, 7.2): device kinds, GPU / CPU / MPS
slots, the local node's unlimited slots (R18), preferences and ``strict``, stability, config-order priority,
pending reasons; the table's statuses, relocation by the node's last heartbeat, reconciliation of a reconnecting
node, restarts and failures, and its hub_state form."""
from __future__ import annotations

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.hub.placement import (
    LearnerSlot,
    LearnerTable,
    NodeCapacity,
    capacity_from_hello,
    parse_device,
    place,
    requested_slot,
)


def _cfg(device: str = "auto", *, node: str | None = None, tags: tuple[str, ...] = (), strict: bool = False):
    return LearnerConfig(device=device, placement={"node": node, "tags": list(tags), "strict": strict})


def _node(number: int, name: str | None = None, *, gpus=(), cpu: int = 0, tags=(), unlimited=False, mps=False):
    return NodeCapacity(number, name or f"n{number}", tuple(tags), tuple(gpus), cpu, unlimited=unlimited, mps=mps)


# ---- devices ------------------------------------------------------------------------------------------------


def test_parse_device_kinds():
    assert parse_device("cpu") == ("cpu", None)
    assert parse_device("cuda") == ("cuda", None)
    assert parse_device("cuda:3") == ("cuda", 3)
    assert parse_device("mps") == ("mps", None)
    with pytest.raises(ValueError, match="expected auto, cpu, cuda, cuda:N or mps"):
        parse_device("tpu")


def test_auto_takes_cuda_when_any_node_offers_cuda_else_cpu():
    assert requested_slot("auto", [_node(1, cpu=1), _node(2, gpus=(0,))]) == ("cuda", None)
    assert requested_slot("auto", [_node(1, cpu=1)]) == ("cpu", None)
    assert requested_slot("auto", [_node(1, gpus=(0,), mps=True)]) == ("cpu", None)   # MPS only by name


def test_auto_waits_for_a_busy_gpu_instead_of_taking_cpu():
    nodes = [_node(1, gpus=(0,), cpu=1)]
    plan = place({"a": _cfg("auto"), "b": _cfg("auto")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0")}
    assert plan.pending == ["b"]
    assert "no free CUDA GPU slot" in plan.reasons["b"]


# ---- slots --------------------------------------------------------------------------------------------------


def test_one_learner_per_gpu_and_cpu_slots_count():
    nodes = [_node(1, gpus=(0, 1), cpu=1)]
    agents = {"a": _cfg("cuda"), "b": _cfg("cuda"), "c": _cfg("cuda"), "d": _cfg("cpu"), "e": _cfg("cpu")}
    plan = place(agents, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0"), "b": LearnerSlot(1, "cuda:1"),
                             "d": LearnerSlot(1, "cpu")}
    assert plan.pending == ["c", "e"]
    assert "CPU learner slot" in plan.reasons["e"]


def test_cuda_index_is_honoured_across_nodes():
    nodes = [_node(1, gpus=(0,)), _node(2, gpus=(0, 1))]
    plan = place({"a": _cfg("cuda:1")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cuda:1")}
    plan = place({"a": _cfg("cuda:3")}, nodes, current={})
    assert plan.pending == ["a"] and "slot on GPU 3" in plan.reasons["a"]


def test_mps_slots_only_for_mps_and_never_for_cuda():
    nodes = [_node(1, gpus=(0,), mps=True)]
    assert place({"a": _cfg("mps")}, nodes, current={}).assigned == {"a": LearnerSlot(1, "mps")}
    plan = place({"a": _cfg("cuda")}, nodes, current={})
    assert plan.pending == ["a"]


def test_local_node_takes_every_learner_on_one_gpu_like_before():
    """R18: train's local node is unlimited; with one GPU every GPU learner shares cuda:0 (as before SP5)."""
    nodes = [_node(1, gpus=(0,), cpu=0, unlimited=True)]
    agents = {"a": _cfg("auto"), "b": _cfg("auto"), "c": _cfg("cpu")}
    plan = place(agents, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cuda:0"), "b": LearnerSlot(1, "cuda:0"),
                             "c": LearnerSlot(1, "cpu")}
    assert plan.pending == []


def test_local_node_without_gpu_runs_auto_on_cpu():
    plan = place({"a": _cfg("auto"), "b": _cfg("auto")}, [_node(1, unlimited=True)], current={})
    assert plan.assigned == {"a": LearnerSlot(1, "cpu"), "b": LearnerSlot(1, "cpu")}


def test_local_node_spreads_over_its_gpus_least_used_first():
    nodes = [_node(1, gpus=(0, 1), unlimited=True)]
    plan = place({"a": _cfg("cuda"), "b": _cfg("cuda"), "c": _cfg("cuda")}, nodes, current={})
    assert [plan.assigned[a].device for a in "abc"] == ["cuda:0", "cuda:1", "cuda:0"]


def test_no_learner_nodes_offer_nothing():
    nodes = [capacity_from_hello(1, "laptop", (), {"gpus": [0], "cpu_learners": 1, "no_learners": True},
                                 {0: "cuda"}, unlimited=False)]
    plan = place({"a": _cfg("cpu")}, nodes, current={})
    assert plan.pending == ["a"] and plan.reasons["a"] == "no online node offers learner slots"


def test_capacity_from_hello_reads_gpu_kinds():
    cap = capacity_from_hello(3, "mac", ("m1",), {"gpus": [0], "cpu_learners": 1}, {0: "mps"}, unlimited=False)
    assert cap == NodeCapacity(3, "mac", ("m1",), (0,), 1, unlimited=False, mps=True)


# ---- preferences --------------------------------------------------------------------------------------------


def test_preferred_node_first_then_any_other():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    plan = place({"a": _cfg("cpu", node="laptop")}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cpu")}
    plan = place({"b": _cfg("cpu"), "a": _cfg("cpu", node="pc")}, nodes, current={})
    # b (first in config order) takes pc's slot; a's preferred node is full, so it goes elsewhere
    assert plan.assigned == {"b": LearnerSlot(1, "cpu"), "a": LearnerSlot(2, "cpu")}


def test_strict_never_leaves_the_preferred_nodes():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    plan = place({"b": _cfg("cpu", node="pc"), "a": _cfg("cpu", node="pc", strict=True)}, nodes, current={})
    assert plan.assigned == {"b": LearnerSlot(1, "cpu")}
    assert plan.pending == ["a"] and "strict" in plan.reasons["a"]
    plan = place({"a": _cfg("cpu", node="cloud", strict=True)}, nodes, current={})
    assert plan.reasons["a"].startswith("no online node matches learner.placement (node 'cloud', strict)")


def test_tags_must_all_match():
    nodes = [_node(1, cpu=1, tags=("fast",)), _node(2, cpu=1, tags=("fast", "big"))]
    plan = place({"a": _cfg("cpu", tags=("fast", "big"))}, nodes, current={})
    assert plan.assigned == {"a": LearnerSlot(2, "cpu")}


# ---- stability ----------------------------------------------------------------------------------------------


def test_current_slots_are_kept_and_count_against_capacity():
    nodes = [_node(1, gpus=(0,), cpu=1), _node(2, gpus=(0,))]
    current = {"b": LearnerSlot(1, "cuda:0")}
    plan = place({"a": _cfg("cuda"), "b": _cfg("cuda")}, nodes, current=current)
    assert plan.assigned == {"a": LearnerSlot(2, "cuda:0"), "b": LearnerSlot(1, "cuda:0")}
    assert plan.new(current) == {"a": LearnerSlot(2, "cuda:0")}


def test_a_learner_never_moves_to_its_preference_while_its_node_lives():
    nodes = [_node(1, "pc", cpu=1), _node(2, "laptop", cpu=1)]
    current = {"a": LearnerSlot(1, "cpu")}
    plan = place({"a": _cfg("cpu", node="laptop")}, nodes, current=current)
    assert plan.assigned == current and plan.new(current) == {}


def test_relocate_places_again_on_the_online_nodes():
    """Criterion 11: a cpu agent preferring the laptop moves to the PC's CPU slot when the laptop is gone."""
    nodes = [_node(1, "pc", gpus=(0,), cpu=1)]                     # the laptop (node 2) is not online
    current = {"main": LearnerSlot(1, "cuda:0"), "cpu_agent": LearnerSlot(2, "cpu")}
    agents = {"main": _cfg("auto"), "cpu_agent": _cfg("cpu", node="laptop")}
    plan = place(agents, nodes, current=current, relocate={"cpu_agent"})
    assert plan.assigned == {"main": LearnerSlot(1, "cuda:0"), "cpu_agent": LearnerSlot(1, "cpu")}


def test_placement_is_deterministic():
    nodes = [_node(3, cpu=1), _node(1, cpu=1), _node(2, gpus=(1, 0))]
    agents = {"a": _cfg("cpu"), "b": _cfg("cpu"), "c": _cfg("cuda")}
    first = place(agents, nodes, current={})
    assert first == place(agents, list(reversed(nodes)), current={})
    assert first.assigned == {"a": LearnerSlot(1, "cpu"), "b": LearnerSlot(3, "cpu"), "c": LearnerSlot(2, "cuda:0")}


# ---- the table ----------------------------------------------------------------------------------------------


def test_table_statuses_from_pending_to_running():
    table = LearnerTable(["a", "b"])
    assert table.to_place() == ["a", "b"] and table.collecting() == []
    table.assign("a", LearnerSlot(1, "cpu"), generation=1)
    assert table["a"].status == "starting" and table.can_collect("a")    # R11 holds its lineups for its weights
    assert table.announced("a", 2) is False                       # another generation: no change
    assert table.announced("a", 1) is True and table["a"].status == "running"
    assert table.collecting() == ["a"] and table.to_place() == ["b"] and not table.can_collect("b")
    with pytest.raises(ValueError, match="not above"):
        table.assign("a", LearnerSlot(1, "cpu"), generation=1)


def test_node_loss_relocates_after_the_wait_measured_from_the_last_heartbeat():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    table.announced("a", 1)
    assert table.node_lost(2, last_heartbeat=100.0) == ["a"]
    assert table["a"].status == "relocating" and not table.can_collect("a")
    assert table.current() == {"a": LearnerSlot(2, "cpu")}        # still awaited: its slot is kept
    assert table.relocation_due(219.0, after_sec=120.0) == []
    assert table.relocation_due(220.0, after_sec=120.0) == ["a"]
    assert table["a"].status == "pending" and table["a"].slot is None


def test_reconcile_adopts_the_current_generation_and_stops_the_rest():
    table = LearnerTable(["a", "b", "c"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=3)
    table.announced("a", 3)
    table.node_lost(2, last_heartbeat=10.0)
    table.assign("b", LearnerSlot(1, "cuda:0"), generation=5)       # b lives on node 1
    out = table.reconcile(2, {"a": 3, "b": 4, "zz": 1}, announced={"a": 3})
    assert out.adopted == ["a"] and table["a"].status == "running"
    assert sorted(out.stop) == [("b", 4), ("zz", 1)]
    assert out.gone == []


def test_reconcile_after_relocation_stops_the_old_learner():
    """Spec 7.2: a learner that kept running on a cut-off machine is told to stop when the machine returns."""
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    table.announced("a", 1)
    table.node_lost(2, last_heartbeat=0.0)
    table.relocation_due(200.0, after_sec=120.0)
    table.assign("a", LearnerSlot(1, "cpu"), generation=2)
    out = table.reconcile(2, {"a": 1}, announced={"a": 1})
    assert out.stop == [("a", 1)] and out.adopted == []
    assert table["a"].slot == LearnerSlot(1, "cpu") and table["a"].generation == 2


def test_reconcile_marks_learners_the_node_no_longer_runs_as_pending():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=1)
    out = table.reconcile(2, {}, announced={})
    assert out.gone == ["a"] and table["a"].status == "pending" and table.to_place() == ["a"]


def test_reconcile_keeps_a_starting_learner_starting():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=7)
    out = table.reconcile(2, {"a": 7}, announced={"a": 6})
    assert out.adopted == ["a"] and table["a"].status == "starting"


def test_died_restarts_only_the_current_generation_and_fail_is_final():
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(1, "cuda:0"), generation=2)
    table.announced("a", 2)
    assert table.died("a", 1) is None                              # an old incarnation: ignored
    assert table.died("a", 2) == LearnerSlot(1, "cuda:0")
    table.assign("a", LearnerSlot(1, "cuda:0"), generation=3)
    table.fail("a", "3 restarts in 1800 s")
    assert table["a"].status == "failed" and table.failed() == ["a"] and table.to_place() == []
    assert table.current() == {}
    with pytest.raises(ValueError, match="failed"):
        table.assign("a", LearnerSlot(1, "cpu"), generation=4)
    table.set_pending("a", "x")
    assert table["a"].status == "failed"


def test_table_state_round_trip_on_resume_waits_for_remote_nodes_only():
    table = LearnerTable(["a", "b", "c", "e"])
    table.assign("a", LearnerSlot(2, "cpu"), generation=4)
    table.announced("a", 4)
    table.assign("b", LearnerSlot(1, "cuda:0"), generation=2)
    table.fail("b", "limit")
    table.assign("e", LearnerSlot(1, "cpu"), generation=7)          # on the local node (dies with the hub)
    state = table.state(local_nodes={1})
    assert state["agents"]["a"] == {"status": "running", "node": 2, "device": "cpu", "generation": 4, "reason": "",
                                    "local": False, "lost_node": None, "lost_device": None}      # lost_*: T4.3, R42
    assert state["agents"]["e"]["local"] is True
    resumed = LearnerTable(["a", "b", "c", "d", "e"])
    assert resumed.load_resumed(state, now=50.0) == ["a"]
    assert resumed["a"].status == "relocating" and resumed["a"].lost_since == 50.0
    assert resumed["a"].generation == 4 and resumed["b"].status == "pending" and resumed["b"].generation == 2
    assert resumed["e"].status == "pending" and resumed["e"].generation == 7
    assert resumed.to_place() == ["b", "c", "d", "e"]
    out = resumed.reconcile(2, {"a": 4}, announced={"a": 4})
    assert out.adopted == ["a"] and resumed["a"].status == "running"


def test_a_pending_agent_s_node_returning_with_its_learner_gets_it_back():
    """Controller ruling R42: a relocation that found no slot leaves the agent pending; its old node returning with the
    same generation's learner is adopted; returning without it (or after a newer generation) is not."""
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(1, "cpu"), 1)
    table.node_lost(1, 0.0)
    assert table.relocation_due(10.0, 5.0) == ["a"] and table["a"].status == "pending"
    out = table.reconcile(1, {"a": 1}, announced={"a": 1})
    assert (out.adopted, out.stop) == (["a"], []) and table["a"].status == "running"
    assert table["a"].slot == LearnerSlot(1, "cpu") and table["a"].lost_slot is None

    table.node_lost(1, 20.0)
    table.relocation_due(30.0, 5.0)
    assert table.reconcile(1, {}, announced={}).adopted == [] and table["a"].lost_slot is None
    assert table.reconcile(1, {"a": 1}, announced={}).stop == [("a", 1)]    # it came back without it before

    table.assign("a", LearnerSlot(2, "cpu"), 2)
    table.node_lost(2, 40.0)
    table.relocation_due(50.0, 5.0)
    table.assign("a", LearnerSlot(3, "cpu"), 3)                             # a newer generation was assigned
    assert table.reconcile(2, {"a": 2}, announced={}).stop == [("a", 2)]

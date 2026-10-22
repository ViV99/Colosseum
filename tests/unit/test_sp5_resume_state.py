"""What an in-place resume carries over besides the journal (SP5 T4.3): a pending agent's lost slot (ruling R42) in the
learner table's ``hub_state.json`` state, and the run-level worker / journal drop totals of ``SystemStats``
(P-PF3-5), so the counters of the ``system`` record stay monotonic across the resume."""
from __future__ import annotations

from colosseum.hub.placement import LearnerSlot, LearnerTable
from colosseum.metrics.aggregator import SystemStats


def _pending_with_lost_slot(node: int) -> LearnerTable:
    table = LearnerTable(["a"])
    table.assign("a", LearnerSlot(node, "cpu"), 1)
    table.node_lost(node, 0.0)
    assert table.relocation_due(100.0, 10.0) == ["a"]            # no free slot afterwards: pending, lost slot kept
    assert table["a"].status == "pending" and table["a"].lost_slot == LearnerSlot(node, "cpu")
    return table


def test_a_pending_agents_lost_slot_survives_a_resume_and_its_node_brings_the_learner_back():
    state = _pending_with_lost_slot(2).state(local_nodes={0})
    resumed = LearnerTable(["a"])
    assert resumed.load_resumed(state, now=5.0) == []
    assert resumed["a"].status == "pending" and resumed["a"].lost_slot == LearnerSlot(2, "cpu")
    assert resumed["a"].generation == 1
    out = resumed.reconcile(2, {"a": 1}, announced={})
    assert out.adopted == ["a"] and resumed["a"].slot == LearnerSlot(2, "cpu")


def test_a_lost_slot_of_a_local_node_is_not_kept():
    state = _pending_with_lost_slot(0).state(local_nodes={0})
    assert state["agents"]["a"]["lost_node"] is None
    resumed = LearnerTable(["a"])
    resumed.load_resumed(state, now=5.0)
    assert resumed["a"].lost_slot is None


def test_system_drop_totals_continue_after_a_resume():
    clock = iter(float(t) for t in range(100))
    before = SystemStats(clock=lambda: next(clock))
    before.on_worker_stats({"worker_id": 0, "node": 1, "incarnation": 1, "dropped_results": 4,
                            "dropped_reward_episodes": 2})
    last = before.snapshot(100, {}, match_log={"dropped": 3})
    assert (last["dropped_match_results"], last["dropped_reward_episodes"]) == (7, 2)
    after = SystemStats(clock=lambda: next(clock))
    after.restore(before.state())
    first = after.snapshot(100, {}, match_log={"dropped": 0})
    assert (first["dropped_match_results"], first["dropped_reward_episodes"]) == (7, 2)
    after.on_worker_stats({"worker_id": 0, "node": 0, "incarnation": 1, "dropped_results": 1})
    later = after.snapshot(200, {}, match_log={"dropped": 2})
    assert later["dropped_match_results"] == 10 and after.cumulative("dropped_results") == 5
    after.restore(None)                                            # an SP4 run dir: nothing to continue
    assert after.cumulative("dropped_results") == 1


def test_prepare_restores_hub_state_and_relays_restored_versions_until_their_node_confirms(tmp_path):
    """A3 / D16 / P-PF3-5 and carried item 2 of T3.1(a): the counters, epochs, versions, generations and learners of
    ``hub_state.json`` come back; a restored latest version keeps no data address (relay only) until its learner
    node's session confirms one; ``counters`` keys this version does not know are saved again."""
    import json

    from colosseum.hub.state import HUB_STATE_FILE, HubStateData, VersionRegistry, save_hub_state
    from colosseum.metrics.counters import CounterBook
    from colosseum.net.messages import VersionInfo
    from game_helpers import make_test_config, make_test_run_dir
    from hub_kit import make_hub

    config = make_test_config("turns")
    run_dir = make_test_run_dir(config, tmp_path)
    run_id = run_dir.ensure_run_id()
    registry = VersionRegistry()
    registry.new_generation("agent_0")
    registry.new_generation("agent_0")
    assert registry.announce(VersionInfo("agent_0", 2, 7, 1, "d", 3, addresses=("10.0.0.5:5000",))) == "accepted"
    counters = {"book": CounterBook({"lineup_redraw_failed": 3}).state(), "intake_book": {},
                "system_stats": {"cumulative": {"dropped_results": 4}, "match_log_dropped": 1}, "future": {"k": 1}}
    learners = {"version": 1, "agents": {"agent_0": {"status": "running", "node": 3, "device": "cpu",
                                                     "generation": 2, "reason": "", "local": False}}}
    save_hub_state(run_dir.root / HUB_STATE_FILE, HubStateData(
        version=1, run_id=run_id, env_steps=500, sessions={"n/s": 500}, epoch_reserved_until=2048,
        versions=registry.state(), counters=counters, nodes_lost=2, learners=learners), fsync=False)
    hub = make_hub(config, run_dir, resume=True, validated=True)
    hub._prepare()
    assert hub._budget.value == 500 and hub._epochs.reserved_until == 2048 and hub._nodes_lost == 2
    assert hub._book.get("lineup_redraw_failed") == 3
    latest = hub._versions.accepted("agent_0")
    assert (latest.version, latest.generation, latest.addresses, latest.relay_only) == (7, 2, (), True)
    assert hub._versions.version_floor("agent_0") == 8 and hub._versions.generation("agent_0") == 2
    entry = hub._learners["agent_0"]
    assert (entry.status, entry.generation, entry.slot.node) == ("relocating", 2, 3)
    hub._save_hub_state(fsync=False)
    saved = json.loads((run_dir.root / HUB_STATE_FILE).read_text())
    assert saved["counters"]["future"] == {"k": 1} and saved["counters"]["system_stats"] == counters["system_stats"]
    assert saved["epoch_reserved_until"] == 2048 and saved["env_steps"] == 500

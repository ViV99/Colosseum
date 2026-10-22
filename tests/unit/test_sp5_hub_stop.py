"""The hub's side of a node's stop and of the run's exit code (SP5 T4.4, spec 7.4, 7.5; W04-24; SP4 R3-08): a node
that leaves is lost once its stop is done and its learner moves at once (also one already waiting for a lost node);
a node counts once in ``nodes_lost``; a failure keeps exit code 1 after a signal. A real ``Hub`` after its setup steps
(``hub_kit.lineup_hub``) with nodes driven in-process."""
from __future__ import annotations

from colosseum.net.proto import colosseum_v1_pb2 as pb
from game_helpers import make_test_config, scripted_agent
from hub_kit import lineup_hub, node_message, open_node


def _hub(tmp_path):
    config = make_test_config("turns", agents={"a": {}, "random": scripted_agent()},
                              distributed={"node_lost_after_sec": 30.0, "learner_relocate_after_sec": 600.0})
    hub = lineup_hub(config, tmp_path)
    first = open_node(hub, name="first", cpu_learners=1)
    second = open_node(hub, name="second", cpu_learners=1)
    assert hub._learner_node_of("a") == hub._number_of(first.node_id)
    return hub, first, second


def _learner(hub) -> tuple[int | None, int, str]:
    entry = hub._learners["a"]
    return (None if entry.slot is None else entry.slot.node), entry.generation, entry.status


def test_a_leaving_node_is_lost_once_its_stop_is_done_and_its_learner_moves_at_once(tmp_path):
    hub, first, second = _hub(tmp_path)
    hub._on_message(node_message(first, event=pb.Event(kind="node_leaving")))
    assert hub._registry.by_id(first.node_id).state == "online"          # its finals and results may still come
    assert _learner(hub)[0] == hub._number_of(first.node_id)
    hub._on_message(node_message(first, event=pb.Event(kind="run_stopped")))
    assert hub._registry.by_id(first.node_id).state == "lost" and first.closed_reason == "left"
    assert _learner(hub) == (hub._number_of(second.node_id), 2, "starting")   # no learner_relocate_after_sec wait
    assert hub._nodes_lost == 1


def test_a_node_lost_and_then_removed_relocates_its_waiting_learner_and_counts_once(tmp_path):
    hub, first, second = _hub(tmp_path)
    entry = hub._registry.by_id(first.node_id)
    entry.state = "lost"                                                 # as NodeRegistry.lost marks it
    hub._on_node_lost(entry)
    assert _learner(hub)[2] == "relocating"                              # waits learner_relocate_after_sec (600 s)
    hub._on_node_lost(entry, reason="removed", relocate_now=True)        # e.g. nodes remove (T6.3)
    assert _learner(hub) == (hub._number_of(second.node_id), 2, "starting")
    assert hub._nodes_lost == 1


def test_a_failure_keeps_exit_code_1_after_a_signal(tmp_path, monkeypatch):
    """SP4 R3-08 (carried item of T2.5 (b)): e.g. a final snapshot that could not be saved during the stop after
    SIGINT fails the run (exit 1, run_state ``failed``); without a failure the signal's code stays."""
    hub, _first, _second = _hub(tmp_path)
    assert hub._exit_code_after_shutdown(130, []) == 130
    hub._fail_run("the final snapshot v3 of 'a' could not be saved (OSError: no space)")
    assert hub._exit_code_after_shutdown(130, []) == 1
    assert hub._exit_code_after_shutdown(143, []) == 1

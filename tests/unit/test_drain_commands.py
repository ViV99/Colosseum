"""Worker command draining keeps checkpoint deltas of coalesced commands (T2.1, B8)."""
import queue

from colosseum.core.types import WorkerCommand
from colosseum.worker.rollout_worker import _drain_commands


def test_drain_commands_returns_newest_with_merged_checkpoints():
    q = queue.Queue()
    first = WorkerCommand(slot_agent_map=[["a"]], new_checkpoints={"a": {"a_v1": {"w": 1}}})
    second = WorkerCommand(slot_agent_map=[["b"]],
                           new_checkpoints={"a": {"a_v2": {"w": 2}}, "b": {"b_v1": {"w": 3}}})
    q.put(first)
    q.put(second)

    cmd = _drain_commands(q)

    assert cmd.slot_agent_map == [["b"]]
    assert cmd.new_checkpoints == {"a": {"a_v1": {"w": 1}, "a_v2": {"w": 2}}, "b": {"b_v1": {"w": 3}}}
    assert q.empty()


def test_drain_commands_on_empty_queue_is_none():
    assert _drain_commands(queue.Queue()) is None

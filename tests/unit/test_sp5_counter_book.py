"""CounterBook (SP5 T2.1, moved from T3.1 by A2; spec block 8): run totals of cumulative per-source counters stay
monotonic across sources that join, restart and leave, a resumed book counts nothing twice, and per-node views sum
the sources of one node (``by_source_prefix``)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from colosseum.metrics.counters import CounterBook


def test_totals_add_the_increase_of_every_source():
    book = CounterBook()
    assert book.report("n1/s1", {"lineup_fetch_failed": 2, "weight_fetches": 5}) == {
        "lineup_fetch_failed": 2, "weight_fetches": 5}
    assert book.report("n1/s1", {"lineup_fetch_failed": 2, "weight_fetches": 7}) == {"weight_fetches": 2}
    book.report("n2/s1", {"weight_fetches": 1})
    assert book.totals() == {"lineup_fetch_failed": 2, "weight_fetches": 8}


def test_a_new_session_of_a_node_adds_to_the_totals_and_never_lowers_them():
    book = CounterBook()
    book.report("n1/s1", {"dropped_chunks.disconnect": 4})
    book.report("n1/s2", {"dropped_chunks.disconnect": 1})      # the node process restarted: a new session
    assert book.get("dropped_chunks.disconnect") == 5
    book.report("n1/s1", {"dropped_chunks.disconnect": 3})      # a late, lower report of the old session
    assert book.get("dropped_chunks.disconnect") == 8           # counted as a restart under the same key
    assert book.get("missing") == 0


def test_malformed_values_are_ignored():
    book = CounterBook()
    book.report("n1/s1", {"a": -3, "b": 1.5, "c": True, "d": "7", "e": 2})
    assert book.totals() == {"e": 2}
    book.add("f", -4)
    book.add("g", 0)
    assert book.totals() == {"e": 2}


def test_add_takes_any_integer_and_refuses_everything_else():
    book = CounterBook()
    book.add("a", np.int64(3))
    book.add("a", 2)
    assert book.totals() == {"a": 5} and type(book.get("a")) is int
    for bad in (True, 1.5, "2", np.float64(1.0), None):
        with pytest.raises(TypeError):
            book.add("a", bad)
    assert book.totals() == {"a": 5}


def test_a_resumed_book_continues_without_double_counting():
    book = CounterBook()
    book.report("n1/s1", {"relay_bytes": 100})
    book.add("dropped_chunks.overflow", 3)
    saved = json.loads(json.dumps(book.state()))                # hub_state.json round trip
    book.report("n1/s1", {"relay_bytes": 150})                  # counted after the save, lost with the crash
    resumed = CounterBook.from_state(saved)
    assert resumed.totals() == {"relay_bytes": 100, "dropped_chunks.overflow": 3}
    resumed.report("n1/s1", {"relay_bytes": 180})               # the same session reports again after --resume
    assert resumed.get("relay_bytes") == 180
    assert CounterBook.from_state(None).totals() == {}


def test_grouped_nests_dotted_names_and_lists_every_member_of_a_group():
    book = CounterBook()
    book.add("relay_bytes", 10)
    book.add("dropped_chunks.stale", 2)
    reasons = ("disconnect", "no_learner", "stale", "stale_generation", "invalid", "overflow")
    assert book.grouped(groups={"dropped_chunks": reasons}) == {
        "relay_bytes": 10,
        "dropped_chunks": {"disconnect": 0, "no_learner": 0, "stale": 2, "stale_generation": 0, "invalid": 0,
                           "overflow": 0},
    }


def test_by_source_prefix_sums_the_last_values_of_one_nodes_sessions():
    """A2: per name, the sum of the last cumulative values of every source whose key starts with the prefix."""
    book = CounterBook()
    book.report("n1/s1", {"results_sent": 10, "worker_restarts": 1})
    book.report("n1/s1", {"results_sent": 12, "worker_restarts": 1})
    book.report("n1/s2", {"results_sent": 5})                   # the node restarted: a second session
    book.report("n10/s1", {"results_sent": 100})                # another node whose id shares the digit prefix
    book.report("hub/abc", {"relay_bytes": 7})
    book.add("results_journaled.1", 4)                          # direct adds belong to no source
    assert book.by_source_prefix("n1/") == {"results_sent": 17, "worker_restarts": 1}
    assert book.by_source_prefix("n10/") == {"results_sent": 100}
    assert book.by_source_prefix("hub/") == {"relay_bytes": 7}
    assert book.by_source_prefix("nx/") == {}
    resumed = CounterBook.from_state(json.loads(json.dumps(book.state())))
    assert resumed.by_source_prefix("n1/") == {"results_sent": 17, "worker_restarts": 1}

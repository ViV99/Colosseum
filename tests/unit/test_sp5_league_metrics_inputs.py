"""The coordinator and metrics inputs the hub uses (SP5 T2.5, Step 4): ``Coordinator.lineup_for`` (one checked lineup
per owner), ``save_league_state(..., journal_row=)``, worker stats keyed by (node, worker, incarnation) and the
learners' intake reports in ``MetricsHub`` (amendment A2)."""
from __future__ import annotations

import json

import pytest

from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.league.base import BaseMatchmaker
from colosseum.metrics.aggregator import SystemStats
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from game_helpers import make_coordinator, make_match_result, make_test_config


class OwnerEverywhereMatchmaker(BaseMatchmaker):
    """Every seat plays the owner's latest weights; counts its calls."""

    calls: list[str] = []

    def lineup_for(self, owner):
        type(self).calls.append(owner)
        layout = next(iter(self.context.spec.layouts))
        return Lineup(layout, [SeatAssignment(owner, LATEST_NETWORK_ID, True, source="owner")]
                      * self.context.spec.layout_size(layout))


class SnapshotCollectsEverywhere(OwnerEverywhereMatchmaker):
    """Breaks the rules: a snapshot seat that collects."""

    def lineup_for(self, owner):
        lineup = super().lineup_for(owner)
        lineup.seats[1] = SeatAssignment(owner, "ckpt_v1", True)
        return lineup


def _coordinator(tmp_path, matchmaker, agents=("a", "b")):
    config = make_test_config("turns", agents={a: {} for a in agents},
                              matchmaking={"matchmaker_class": f"test_sp5_league_metrics_inputs.{matchmaker}"})
    return make_coordinator(config, tmp_path / "ckpt")


def test_lineup_for_gives_the_matchmakers_lineup_of_that_owner(tmp_path):
    OwnerEverywhereMatchmaker.calls = []
    coordinator = _coordinator(tmp_path, "OwnerEverywhereMatchmaker")
    lineup = coordinator.lineup_for("b")
    assert [(s.agent_id, s.collect) for s in lineup.seats] == [("b", True), ("b", True)]
    assert OwnerEverywhereMatchmaker.calls == ["b"]
    lineups = coordinator.generate_lineups(3, env_offset=1)          # SP4's rotation, one lineup_for per env
    assert [lu.seats[0].agent_id for lu in lineups] == ["b", "a", "b"]
    assert OwnerEverywhereMatchmaker.calls == ["b", "b", "a", "b"]


def test_lineup_for_checks_the_lineup(tmp_path):
    coordinator = _coordinator(tmp_path, "SnapshotCollectsEverywhere")
    with pytest.raises(ValueError, match="SnapshotCollectsEverywhere"):
        coordinator.lineup_for("a")


def test_the_league_state_carries_the_journal_row(tmp_path):
    coordinator = make_coordinator(make_test_config("turns"), tmp_path / "ckpt")
    coordinator.report_match_result(make_match_result([(0, "agent_0", "latest"), (1, "agent_0", "latest")],
                                                      [1.0, 2.0], uid="m0"))
    path = tmp_path / LEAGUE_STATE_FILE
    coordinator.save_league_state(path, env_steps=10)
    assert json.loads(path.read_text())["journal_row"] is None
    coordinator.save_league_state(path, env_steps=20, journal_row=57)
    saved = json.loads(path.read_text())
    assert (saved["env_steps"], saved["journal_row"]) == (20, 57)
    fresh = make_coordinator(make_test_config("turns"), tmp_path / "ckpt2")
    assert fresh.load_league_state(path) is True                       # the watermark is ignored on load
    assert fresh.ratings_snapshot() == coordinator.ratings_snapshot()


def test_worker_stats_are_keyed_by_node_worker_and_incarnation():
    stats = SystemStats(clock=lambda: 0.0)
    stats.on_worker_stats({"worker_id": 0, "dropped_results": 4, "parked_buffers": 1})            # SP4 path
    stats.on_worker_stats({"node": 1, "worker_id": 0, "incarnation": 0, "dropped_results": 3, "parked_buffers": 2})
    stats.on_worker_stats({"node": 2, "worker_id": 0, "incarnation": 0, "dropped_results": 5})
    stats.on_worker_stats({"node": 1, "worker_id": 0, "incarnation": 1, "dropped_results": 2})   # a restart
    stats.on_worker_stats({"node": 1, "worker_id": 0, "incarnation": 0, "dropped_results": 1})   # never lowers
    stats.on_worker_stats({"worker_id": 0, "incarnation": 0, "dropped_results": 6})               # (None, 0, 0)
    stats.on_worker_stats({"node": None, "worker_id": 0, "incarnation": None, "dropped_results": 5})  # likewise
    assert stats.cumulative("dropped_results") == 6 + 3 + 5 + 2
    snap = stats.snapshot(0, {})
    assert snap["workers_reporting"] == 4 and snap["dropped_match_results"] == 16
    assert snap["parked_buffers"] == 0                                    # the last report of each key counts


def test_intake_reports_keep_the_depth_and_count_drops_per_learner_incarnation(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["main"], total_timesteps=100, log_interval=1, console_interval_sec=10.0,
                     clock=lambda: 0.0)
    hub.on_intake_stats("main", 1, 3, {"invalid": 2})
    hub.on_intake_stats("main", 1, 5, {"invalid": 4, "disconnect": 1})   # cumulative per incarnation
    hub.on_intake_stats("main", 2, 0, {"invalid": 1})                     # a new generation counts from zero
    hub.on_intake_stats("other", 1, 7, {})
    assert hub.intake_queue_depths() == {"main": 0, "other": 7}
    assert hub.intake_book.totals() == {"dropped_chunks.invalid": 5, "dropped_chunks.disconnect": 1}
    assert hub.intake_book.by_source_prefix("learner/main/") == {"dropped_chunks.invalid": 5,
                                                                  "dropped_chunks.disconnect": 1}
    hub.close(env_steps=0, ratings={}, queue_depths={})

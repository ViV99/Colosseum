"""Played shares of opponent categories and anchors per agent and layout (spec block 5, T3.3)."""
from __future__ import annotations

import json

import pytest

from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.metrics.aggregator import EpisodeAggregator, opponent_draws
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter


def match(*teams, layout: str = "2p") -> MatchResult:
    """``teams``: one ``(source, [(agent, network), ...])`` per team, seats numbered in order; team t ranks t + 1."""
    seats, s = [], 0
    for team, (source, members) in enumerate(teams):
        for agent, network in members:
            seats.append(SeatResult(s, "player", team, agent, network, 0.0, source=source))
            s += 1
    kind = "wdl" if len(teams) == 2 else "rank"
    return MatchResult(match_id=f"m{s}", layout=layout, outcome_kind=kind, seats=seats,
                       teams=[TeamResult(t, float(t + 1), 0.0) for t in range(len(teams))], episode_length=3)


OWNER_A = ("owner", [("a", "latest")])


def test_opponent_draws_names_the_owner_and_each_opposing_teams_category():
    assert opponent_draws(match(OWNER_A, ("anchors", [("bot", "fixed")]))) == ("a", [("anchors", "bot")])
    assert opponent_draws(match(("latest", [("a", "latest")]), OWNER_A)) == ("a", [("latest", None)])
    ffa = match(OWNER_A, ("snapshots", [("a", "ckpt_v2")]), ("rivals", [("b", "latest")]), layout="3p")
    assert opponent_draws(ffa) == ("a", [("snapshots", None), ("rivals", None)])


@pytest.mark.parametrize("result", [
    match(("", [("a", "latest")]), ("", [("b", "latest")])),                       # eval / tests: no sources
    match(("owner", [("a", "latest"), ("b", "latest")]), ("latest", [("a", "latest"), ("a", "latest")])),
])
def test_results_without_a_single_owner_are_not_counted(result):
    assert opponent_draws(result) is None


def test_anchor_of_a_mixed_team_is_its_most_frequent_fixed_agent():
    result = match(("owner", [("a", "latest"), ("a", "latest")]),
                   ("anchors", [("bot", "fixed"), ("bot", "fixed"), ("a", "latest")]))
    assert opponent_draws(result) == ("a", [("anchors", "bot")])


def test_episode_aggregator_reports_played_shares_per_layout():
    agg = EpisodeAggregator()
    for result in (match(OWNER_A, ("latest", [("a", "latest")])),
                   match(OWNER_A, ("anchors", [("bot", "fixed")])),
                   match(OWNER_A, ("anchors", [("bot", "fixed")])),
                   match(OWNER_A, ("snapshots", [("a", "ckpt_v1")])),
                   match(OWNER_A, ("snapshots", [("a", "ckpt_v1")]), ("fallback", [("a", "latest")]), layout="3p")):
        agg.add(result)
    out = agg.flush()
    two = out["a"]["opponents"]["2p"]
    assert two["teams"] == 4
    assert two["categories"] == pytest.approx({"latest": 0.25, "snapshots": 0.25, "rivals": 0.0, "anchors": 0.5,
                                               "fallback": 0.0})
    assert two["anchors"] == {"bot": 0.5}
    three = out["a"]["opponents"]["3p"]
    assert three["teams"] == 2 and three["categories"]["fallback"] == 0.5 and three["anchors"] == {}
    assert agg.flush() == {}                              # flushed


def test_hub_writes_the_shares_to_metrics_jsonl_and_wandb(tmp_path):
    rows = []

    class FakeWandB:
        def log_train(self, agent_id, metrics, train_step):
            pass

        def log_global(self, metrics, env_steps):
            rows.append(dict(metrics))

    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    hub = MetricsHub(writer=writer, ratings_path=tmp_path / "ratings.json", agent_ids=["a"], total_timesteps=100,
                     log_interval=1, console_interval_sec=0.0, wandb_logger=FakeWandB())
    hub.on_match_result(match(OWNER_A, ("anchors", [("bot", "fixed")])))
    hub.close(env_steps=10, ratings={}, queue_depths={})
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    (episodes,) = [r for r in records if r["kind"] == "episodes"]
    assert episodes["opponents"]["2p"]["anchors"] == {"bot": 1.0}
    (row,) = rows
    assert row["episodes/a/opponents/2p/anchors/bot"] == 1.0
    assert row["episodes/a/opponents/2p/categories/anchors"] == 1.0


def test_a_layout_without_opposing_teams_has_no_opponent_cell():
    agg = EpisodeAggregator()
    agg.add(match(("owner", [("a", "latest"), ("a", "latest")]), layout="coop"))   # cooperative: one team
    assert opponent_draws(match(("owner", [("a", "latest")]), layout="coop")) == ("a", [])
    assert agg.flush()["a"]["opponents"] == {}

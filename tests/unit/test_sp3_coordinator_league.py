"""Coordinator with the league (spec block 5): built-in and custom matchmakers, lineup checks,
on_result, share schedules at the coordinator's env steps (T3.3)."""
from __future__ import annotations

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.launcher import Launcher, setup_run
from colosseum.league.base import BaseMatchmaker
from colosseum.league.mixture import MixtureMatchmaker
from colosseum.players.registry import resolve_player_roles
from game_helpers import StepCounter, make_coordinator, make_test_config, make_test_run_dir

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
SCHEDULED = {"opponents": {"latest": {0: 1.0, 1000: 0.0}, "snapshots": 0.0, "rivals": 0.0,
                           "anchors": {0: 0.0, 1000: 1.0}}}


class MirrorMatchmaker(BaseMatchmaker):
    """Every seat plays the owner's latest weights; remembers the results it was given."""

    def __init__(self, context):
        super().__init__(context)
        self.results: list[MatchResult] = []

    def lineup_for(self, owner):
        layout = next(iter(self.context.spec.layouts))
        size = self.context.spec.layout_size(layout)
        return Lineup(layout, [SeatAssignment(owner, LATEST_NETWORK_ID, True, source="owner")] * size)

    def on_result(self, result):
        self.results.append(result)


class SnapshotCollectsMatchmaker(MirrorMatchmaker):
    """Breaks the rules: a snapshot seat that collects."""

    def lineup_for(self, owner):
        lineup = super().lineup_for(owner)
        lineup.seats[1] = SeatAssignment(owner, "ckpt_v1", True)
        return lineup


class NotAMatchmaker:
    pass


def coordinator(tmp_path, steps=None, **sections) -> Coordinator:
    cfg = make_test_config("turns", **sections)
    spec = env_spec(cfg)
    return Coordinator(cfg, spec, resolve_player_roles(cfg, spec), tmp_path / "ckpt",
                       env_steps=steps if steps is not None else StepCounter(0))


def test_the_built_in_mixture_is_the_default(tmp_path):
    coord = make_coordinator(make_test_config("turns"), tmp_path / "ckpt")
    assert type(coord.matchmaker) is MixtureMatchmaker
    assert all(lineup.seats[0].source for lineup in coord.generate_lineups(8, 0))


def test_a_custom_matchmaker_class_builds_the_lineups_and_gets_every_result(tmp_path):
    coord = coordinator(tmp_path, matchmaking={"matchmaker_class": "test_sp3_coordinator_league.MirrorMatchmaker"})
    assert isinstance(coord.matchmaker, MirrorMatchmaker)
    lineups = coord.generate_lineups(4, 0)
    assert all([(s.agent_id, s.collect) for s in lu.seats] == [("agent_0", True)] * 2 for lu in lineups)
    layout = lineups[0].layout
    result = MatchResult(match_id="m", layout=layout, outcome_kind="wdl",
                         seats=[SeatResult(0, "player", 0, "agent_0", "latest", 1.0, source="owner"),
                                SeatResult(1, "player", 1, "agent_0", "latest", -1.0, source="owner")],
                         teams=[TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], episode_length=4)
    coord.report_match_result(result)
    assert coord.matchmaker.results == [result]


def test_lineups_of_a_custom_matchmaker_are_checked(tmp_path):
    coord = coordinator(tmp_path, matchmaking={
        "matchmaker_class": "test_sp3_coordinator_league.SnapshotCollectsMatchmaker"})
    with pytest.raises(ValueError, match="SnapshotCollectsMatchmaker"):
        coord.generate_lineups(1, 0)


@pytest.mark.parametrize("path, message", [
    ("test_sp3_coordinator_league.NotAMatchmaker", "BaseMatchmaker"),
    ("test_sp3_coordinator_league.Missing", "cannot be imported"),
])
def test_a_bad_matchmaker_class_is_a_config_error(tmp_path, path, message):
    with pytest.raises(ConfigError, match=message):
        coordinator(tmp_path, matchmaking={"matchmaker_class": path})


def test_share_schedules_follow_the_coordinators_env_steps(tmp_path):
    steps = StepCounter(0)
    coord = coordinator(tmp_path, steps, agents={"agent_0": {}, "bot": BOT}, matchmaking=SCHEDULED)

    def opposing() -> set[tuple[str, str]]:
        return {(s.source, s.agent_id) for lu in coord.generate_lineups(16, 0) for s in lu.seats if s.source != "owner"}

    assert opposing() == {("latest", "agent_0")}
    steps.value = 1000
    assert opposing() == {("anchors", "bot")}


def test_the_launchers_coordinator_reads_the_global_env_step_counter(tmp_path):
    cfg = make_test_config("turns", agents={"agent_0": {}, "bot": BOT}, matchmaking=SCHEDULED)
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    coord = launcher._build_coordinator(setup_run(cfg, validate=False))
    assert coord.context.env_steps() == 0
    launcher._env_step_counter.add(5000)
    assert coord.context.env_steps() == 5000
    assert {s.agent_id for s in coord.generate_lineups(1, 0)[0].seats if s.source != "owner"} == {"bot"}

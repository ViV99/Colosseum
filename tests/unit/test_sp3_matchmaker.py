"""MixtureMatchmaker (spec block 5): drawn category shares, redistribution of empty categories,
schedules, per-agent override, asymmetric games, the runtime fallback, PFSP, teammates, seat
balance and seat permutations (T3.2; ports the guarantees of SP2's test_sp2_matchmaker.py)."""
from __future__ import annotations

import logging
import random
from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, SOURCE_OWNER, SeatAssignment
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.lineups import check_lineup, permute_seats
from colosseum.league.mixture import MixtureMatchmaker, effective_mix
from game_helpers import StepCounter, make_matchmaker_context

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))
DRAWS = 4000
TOL = 0.03          # 4000 draws: three standard deviations of any share are below 0.024
ONLY = {"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}


def hunt_spec() -> GameSpec:
    """One hunter (team 0) against three prey (team 1)."""
    return GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )


def only(**shares) -> dict:
    return {"opponents": {**ONLY, **shares}}


def core_of(lineup) -> SeatAssignment:
    """The single opposing seat of a 2-seat lineup."""
    (core,) = [s for s in lineup.seats if s.source != SOURCE_OWNER]
    return core


def test_drawn_category_shares_match_the_configured_mix():
    spec = GameSpec.symmetric(2, OBS, ACT)
    mix = {"latest": 0.4, "snapshots": 0.3, "rivals": 0.2, "anchors": 0.1}
    ctx = make_matchmaker_context(
        spec, {"a": ["player"], "b": ["player"], "bot": ["player"]}, kinds={"bot": "scripted"},
        matchmaking={"opponents": mix}, snapshots={"a": ["ckpt_v1", "ckpt_v2"], "b": ["ckpt_v3"]})
    assert effective_mix(ctx, "a", "2p", 1, 0) == pytest.approx(mix)
    mm = MixtureMatchmaker(ctx)
    sources, players = Counter(), Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("a")
        check_lineup(ctx, lineup, "test")
        owner = [s for s in lineup.seats if s.source == SOURCE_OWNER]
        assert [(s.agent_id, s.network_id, s.collect) for s in owner] == [("a", LATEST_NETWORK_ID, True)]
        core = core_of(lineup)
        sources[core.source] += 1
        players[(core.agent_id, core.network_id, core.collect)] += 1
    for category, share in mix.items():
        assert abs(sources[category] / DRAWS - share) < TOL, sources
    assert players[("a", LATEST_NETWORK_ID, True)] == sources["latest"]       # self-play: the owner's latest
    assert players[("b", LATEST_NETWORK_ID, True)] == sources["rivals"]       # rivals collect too
    assert players[("bot", FIXED_NETWORK_ID, False)] == sources["anchors"]
    snaps = {k: v for k, v in players.items() if k[1].startswith("ckpt_")}
    assert set(snaps) == {("a", "ckpt_v1", False), ("a", "ckpt_v2", False), ("b", "ckpt_v3", False)}
    for count in snaps.values():                                              # equal PFSP scores: uniform
        assert abs(count / sources["snapshots"] - 1 / 3) < 0.05, snaps


def test_shares_of_empty_categories_are_spread_proportionally():
    spec = GameSpec.symmetric(2, OBS, ACT)
    ctx = make_matchmaker_context(spec, {"a": ["player"], "b": ["player"]},
                                  matchmaking={"opponents": {"latest": 0.4, "snapshots": 0.3, "rivals": 0.2,
                                                             "anchors": 0.1}})
    mm = MixtureMatchmaker(ctx)
    sources = Counter(core_of(mm.lineup_for("a")).source for _ in range(DRAWS))   # no snapshots, no anchors
    assert set(sources) == {"latest", "rivals"}
    assert abs(sources["latest"] / DRAWS - 2 / 3) < TOL and abs(sources["rivals"] / DRAWS - 1 / 3) < TOL
    # structurally snapshots count as available (they appear with the first checkpoint); anchors do not exist
    assert effective_mix(ctx, "a", "2p", 1, 0) == pytest.approx(
        {"latest": 0.4 / 0.9, "snapshots": 0.3 / 0.9, "rivals": 0.2 / 0.9})


def test_shares_and_anchor_weights_follow_the_env_step_schedules():
    spec = GameSpec.symmetric(2, OBS, ACT)
    steps = StepCounter(0)
    ctx = make_matchmaker_context(
        spec, {"a": ["player"], "r1": ["player"], "r2": ["player"]}, kinds={"r1": "scripted", "r2": "frozen"},
        matchmaking={"opponents": {**ONLY, "latest": {0: 1.0, 1000: 0.0}, "anchors": {0: 0.0, 1000: 1.0}},
                     "anchors": {"r1": {0: 1.0, 1000: 0.0}, "r2": {0: 0.0, 1000: 1.0}}},
        env_steps=steps)
    mm = MixtureMatchmaker(ctx)

    def cores(n: int) -> Counter:
        return Counter((c.source, c.agent_id) for c in (core_of(mm.lineup_for("a")) for _ in range(n)))

    assert cores(200) == Counter({("latest", "a"): 200})
    steps.value = 500
    half = cores(DRAWS)
    assert abs(half[("latest", "a")] / DRAWS - 0.5) < TOL
    assert abs(half[("anchors", "r1")] / DRAWS - 0.25) < TOL and abs(half[("anchors", "r2")] / DRAWS - 0.25) < TOL
    steps.value = 5000
    assert cores(200) == Counter({("anchors", "r2"): 200})        # r1's weight is 0 now


def test_a_per_agent_override_changes_only_that_owners_mix():
    spec = GameSpec.symmetric(2, OBS, ACT)
    ctx = make_matchmaker_context(spec, {"a": ["player"], "b": ["player"]}, matchmaking=only(latest=1.0),
                                  per_agent={"b": {"opponents": {"latest": 0.0, "rivals": 1.0}}})
    mm = MixtureMatchmaker(ctx)
    a_cores = Counter((c.source, c.agent_id) for c in (core_of(mm.lineup_for("a")) for _ in range(300)))
    b_cores = Counter((c.source, c.agent_id) for c in (core_of(mm.lineup_for("b")) for _ in range(300)))
    assert a_cores == Counter({("latest", "a"): 300})
    assert b_cores == Counter({("rivals", "a"): 300})


def test_asymmetric_games_draw_the_other_agents_snapshots():
    spec = hunt_spec()
    ctx = make_matchmaker_context(spec, {"h": ["hunter"], "p": ["prey"]}, matchmaking=only(latest=0.5, snapshots=0.5),
                                  snapshots={"h": ["ckpt_v9"], "p": ["ckpt_v4"]})
    mm = MixtureMatchmaker(ctx)
    prey_teams = Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("h")
        check_lineup(ctx, lineup, "test")
        hunter, *prey = lineup.seats
        assert (hunter.agent_id, hunter.network_id, hunter.source) == ("h", LATEST_NETWORK_ID, SOURCE_OWNER)
        assert len({(s.agent_id, s.network_id, s.source) for s in prey}) == 1          # teammates: self
        prey_teams[(prey[0].source, prey[0].agent_id, prey[0].network_id)] += 1
    assert set(prey_teams) == {("latest", "p", LATEST_NETWORK_ID), ("snapshots", "p", "ckpt_v4")}
    assert abs(prey_teams[("snapshots", "p", "ckpt_v4")] / DRAWS - 0.5) < TOL
    assert {mm.lineup_for("p").seats[0].network_id for _ in range(400)} == {LATEST_NETWORK_ID, "ckpt_v9"}


def test_rivals_count_as_latest_for_a_team_the_owner_cannot_play():
    spec = hunt_spec()
    ctx = make_matchmaker_context(spec, {"h1": ["hunter"], "h2": ["hunter"], "p": ["prey"]},
                                  matchmaking=only(rivals=1.0))
    mm = MixtureMatchmaker(ctx)
    assert effective_mix(ctx, "h1", "1v3", 1, 0) == {"latest": 1.0}
    for _ in range(100):                     # owner h1: the prey team is p@latest, drawn as "latest"
        _hunter, *prey = mm.lineup_for("h1").seats
        assert {(s.source, s.agent_id, s.network_id, s.collect) for s in prey} == {
            ("latest", "p", LATEST_NETWORK_ID, True)}
    hunters = Counter()
    for _ in range(DRAWS):                   # owner p: the latest of h1 or h2 by PFSP (equal scores)
        hunter = mm.lineup_for("p").seats[0]
        assert (hunter.source, hunter.network_id, hunter.collect) == ("latest", LATEST_NETWORK_ID, True)
        hunters[hunter.agent_id] += 1
    assert abs(hunters["h1"] / DRAWS - 0.5) < TOL


def test_runtime_fallback_takes_latest_and_warns_once_per_agent_and_layout(caplog):
    spec = GameSpec.symmetric([2, 3], OBS, ACT)
    ctx = make_matchmaker_context(spec, {"a": ["player"]}, matchmaking=only(snapshots=1.0))
    mm = MixtureMatchmaker(ctx)
    with caplog.at_level(logging.WARNING, logger="colosseum.league.mixture"):
        lineups = [mm.lineup_for("a") for _ in range(200)]
    for lineup in lineups:
        opposing = [s for s in lineup.seats if s.source != SOURCE_OWNER]
        assert opposing and all((s.source, s.agent_id, s.network_id) == ("fallback", "a", LATEST_NETWORK_ID)
                                for s in opposing)
    warnings = [r.getMessage() for r in caplog.records if "fallback" in r.getMessage()]
    assert len(warnings) == 2                                          # one per (agent, layout)
    assert sorted("'2p'" in m for m in warnings) == [False, True] and any("'3p'" in m for m in warnings)


def test_runtime_fallback_takes_anchors_when_no_trainable_agent_plays_the_team():
    spec = hunt_spec()
    ctx = make_matchmaker_context(spec, {"h": ["hunter"], "bot": ["prey"]}, kinds={"bot": "scripted"},
                                  matchmaking=only(snapshots=1.0))
    lineup = MixtureMatchmaker(ctx).lineup_for("h")
    assert {(s.source, s.agent_id, s.network_id, s.collect) for s in lineup.seats[1:]} == {
        ("fallback", "bot", FIXED_NETWORK_ID, False)}


@pytest.mark.parametrize("weighting, exponent, share_of_c", [
    ("hard", 1.0, 0.9), ("hard", 2.0, 0.81 / 0.82), ("balanced", 2.0, 0.5), ("uniform", 2.0, 0.5),
])
def test_pfsp_weighting_of_candidates(weighting, exponent, share_of_c):
    spec = GameSpec.symmetric(2, OBS, ACT)
    ctx = make_matchmaker_context(
        spec, {"a": ["player"], "b": ["player"], "c": ["player"]},
        matchmaking={**only(rivals=1.0), "pfsp": {"weighting": weighting, "exponent": exponent}},
        scores={("a", ("b", LATEST_NETWORK_ID)): 0.9, ("a", ("c", LATEST_NETWORK_ID)): 0.1})
    mm = MixtureMatchmaker(ctx)
    c = sum(core_of(mm.lineup_for("a")).agent_id == "c" for _ in range(DRAWS))
    assert abs(c / DRAWS - share_of_c) < TOL


def test_teammates_mixed_draw_other_latest_core_snapshots_and_anchors():
    spec = GameSpec.teams_of([2], OBS, ACT)
    ctx = make_matchmaker_context(
        spec, {"a": ["player"], "b": ["player"], "bot": ["player"]}, kinds={"bot": "scripted"},
        matchmaking={"teammates": "mixed", "teammate_self_prob": 0.5, "shuffle_seats": False},
        snapshots={"a": ["ckpt_v3"]})
    mm = MixtureMatchmaker(ctx)
    mates = Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("a")
        check_lineup(ctx, lineup, "test")
        assert all(s.source == SOURCE_OWNER for s in lineup.seats)
        core = next(i for i, s in enumerate(lineup.seats) if (s.agent_id, s.network_id) == ("a", LATEST_NETWORK_ID))
        other = lineup.seats[1 - core]
        mates[(other.agent_id, other.network_id, other.collect)] += 1
    assert abs(mates[("a", LATEST_NETWORK_ID, True)] / DRAWS - 0.5) < TOL
    for key in (("b", LATEST_NETWORK_ID, True), ("a", "ckpt_v3", False), ("bot", FIXED_NETWORK_ID, False)):
        assert abs(mates[key] / DRAWS - 1 / 6) < TOL, mates


@pytest.mark.parametrize("sizes", [[2, 2], [2, 1, 1], [3]])
def test_teammates_self_fills_every_team_with_its_core(sizes):
    """SP2's team guarantees: with ``teammates: self`` every team is one player with one source, the
    owner's team is drawn among all teams, and a cooperative layout is the owner's latest only."""
    spec = GameSpec.teams_of(sizes, OBS, ACT)
    ctx = make_matchmaker_context(spec, {"a": ["player"], "b": ["player"], "c": ["player"]},
                                  matchmaking=only(latest=0.5, rivals=0.5), snapshots={"a": ["ckpt_v1"]})
    mm = MixtureMatchmaker(ctx)
    owner_team_sizes, cores = Counter(), Counter()
    for _ in range(300):
        lineup = mm.lineup_for("a")
        check_lineup(ctx, lineup, "test")
        teams = [[lineup.seats[s] for s in members] for members in spec.teams(lineup.layout)]
        for members in teams:
            assert len({(s.agent_id, s.network_id, s.collect, s.source) for s in members}) == 1, members
        (owner_team,) = [members for members in teams if members[0].source == SOURCE_OWNER]
        assert all((s.agent_id, s.network_id, s.collect) == ("a", LATEST_NETWORK_ID, True) for s in owner_team)
        owner_team_sizes[len(owner_team)] += 1
        cores.update(members[0].agent_id for members in teams if members is not owner_team)
    assert set(owner_team_sizes) == set(sizes)
    if len(sizes) > 1:
        assert set(cores) == {"a", "b", "c"}        # self-play and rivals both happen


def test_a_seat_of_a_role_the_core_does_not_play_goes_to_a_latest_player_or_an_anchor():
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"pairs": (SeatSpec("hunter", 0), SeatSpec("prey", 0), SeatSpec("hunter", 1), SeatSpec("prey", 1))},
    )
    ctx = make_matchmaker_context(spec, {"h": ["hunter"], "h2": ["hunter"], "q": ["prey"]},
                                  matchmaking={**only(rivals=1.0), "shuffle_seats": False})
    mm = MixtureMatchmaker(ctx)
    hunters = Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("h")
        check_lineup(ctx, lineup, "test")
        assert lineup.seats[1].agent_id == lineup.seats[3].agent_id == "q"     # the only prey player
        hunters[tuple(sorted((lineup.seats[0].agent_id, lineup.seats[2].agent_id)))] += 1
    # opposing core h2 or q (rivals, equal PFSP); with core q its hunter seat is h or h2 uniformly
    assert set(hunters) == {("h", "h"), ("h", "h2")}, hunters
    assert abs(hunters[("h", "h")] / DRAWS - 0.25) < TOL, hunters
    with_bot = make_matchmaker_context(spec, {"h": ["hunter"], "bot": ["prey"]}, kinds={"bot": "frozen"},
                                       matchmaking={"shuffle_seats": False})
    mm = MixtureMatchmaker(with_bot)
    for _ in range(100):                     # no trainable agent plays prey: an anchor of the owner does
        lineup = mm.lineup_for("h")
        check_lineup(with_bot, lineup, "test")
        assert all((s.agent_id, s.network_id, s.collect) == ("bot", FIXED_NETWORK_ID, False)
                   for s in (lineup.seats[1], lineup.seats[3]))


def test_a_per_agent_layout_override_restricts_only_that_owner():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    mm = MixtureMatchmaker(make_matchmaker_context(spec, {"a": ["player"], "b": ["player"]},
                                                   per_agent={"b": {"layouts": {"4p": 1.0}}}))
    assert {mm.lineup_for("b").layout for _ in range(100)} == {"4p"}
    assert {mm.lineup_for("a").layout for _ in range(100)} == {"2p", "4p"}


def test_layouts_follow_their_weights_and_need_a_seat_of_the_owners_role():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    mm = MixtureMatchmaker(make_matchmaker_context(spec, {"a": ["player"]},
                                                   matchmaking={"layouts": {"2p": 0.25, "4p": 0.75}}))
    counts = Counter(mm.lineup_for("a").layout for _ in range(DRAWS))
    assert abs(counts["2p"] / DRAWS - 0.25) < TOL, counts
    mixed = GameSpec(
        roles={"player": RoleSpec(OBS, ACT), "hunter": HUNTER, "prey": PREY},
        layouts={"duel": (SeatSpec("player", 0), SeatSpec("player", 1)),
                 "hunt": (SeatSpec("hunter", 0), SeatSpec("prey", 1))},
    )
    mm = MixtureMatchmaker(make_matchmaker_context(mixed, {"p": ["player"], "h": ["hunter"], "q": ["prey"]}))
    assert {mm.lineup_for("h").layout for _ in range(50)} == {"hunt"}
    assert {mm.lineup_for("p").layout for _ in range(50)} == {"duel"}


@pytest.mark.parametrize(("players", "agents", "shares"), [
    (2, ["a", "b", "c"], {"latest": 0.15, "snapshots": 0.15, "rivals": 0.7}),
    (2, ["a", "b"], {"latest": 0.5, "snapshots": 0.5}),
    (2, ["a"], {"snapshots": 1.0}),
    (4, ["a", "b", "c"], {"rivals": 1.0}),
])
def test_seats_are_balanced_within_5_percent(players, agents, shares):
    """SP1's seat-balance guarantee: with shuffled seats every agent's collecting seats, and the
    snapshot seats, spread evenly over the seats."""
    spec = GameSpec.symmetric(players, OBS, ACT)
    ctx = make_matchmaker_context(spec, {a: ["player"] for a in agents}, matchmaking=only(**shares),
                                  snapshots={a: ["ckpt_v1", "ckpt_v2"] for a in agents}, seed=1)
    mm = MixtureMatchmaker(ctx)
    collecting: dict[str, Counter] = {a: Counter() for a in agents}
    snapshot_seats = Counter()
    for _ in range(9000 // len(agents)):
        for owner in agents:
            for index, seat in enumerate(mm.lineup_for(owner).seats):
                if seat.collect:
                    collecting[seat.agent_id][index] += 1
                else:
                    snapshot_seats[index] += 1
    for counts in [*collecting.values(), snapshot_seats]:
        total = sum(counts.values())
        if counts is snapshot_seats and "snapshots" not in shares:
            assert total == 0
            continue
        assert total > 1000, counts
        for index in range(players):
            assert abs(counts[index] / total - 1 / players) <= 0.05, counts


def test_lineup_for_rejects_an_owner_that_is_not_trainable():
    spec = GameSpec.symmetric(2, OBS, ACT)
    mm = MixtureMatchmaker(make_matchmaker_context(spec, {"a": ["player"], "bot": ["player"]},
                                                   kinds={"bot": "scripted"}))
    for owner in ("ghost", "bot"):
        with pytest.raises(KeyError, match="trainable"):
            mm.lineup_for(owner)


def test_permute_seats_keeps_team_and_role_structure_and_moves_sources_with_seats():
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"mix": (
            SeatSpec("hunter", 0), SeatSpec("prey", 0), SeatSpec("prey", 0),
            SeatSpec("hunter", 1), SeatSpec("prey", 1), SeatSpec("prey", 1),
            SeatSpec("hunter", 2),
        )},
    )
    seat_specs = spec.layouts["mix"]
    team_source = {0: SOURCE_OWNER, 1: "snapshots", 2: "anchors"}
    labels = [SeatAssignment(agent_id=f"s{i}", source=team_source[seat_specs[i].team]) for i in range(7)]
    rng = random.Random(0)
    team_swaps = within_team_swaps = 0
    for _ in range(500):
        out = permute_seats(spec, "mix", labels, rng)
        assert sorted(s.agent_id for s in out) == sorted(s.agent_id for s in labels)
        origin = {s.agent_id: i for i, s in enumerate(labels)}
        for target, assignment in enumerate(out):
            assert seat_specs[origin[assignment.agent_id]].role == seat_specs[target].role
            assert assignment.source == labels[origin[assignment.agent_id]].source
        for members in spec.teams("mix"):
            assert len({seat_specs[origin[out[s].agent_id]].team for s in members}) == 1   # teams move whole
            assert len({out[s].source for s in members}) == 1
        assert out[6].agent_id == "s6"            # the only team of its composition never moves
        team_swaps += out[0].agent_id == "s3"
        within_team_swaps += out[1].agent_id in ("s2", "s5")
    assert team_swaps > 0 and within_team_swaps > 0


def test_permute_seats_rejects_a_wrong_seat_count():
    spec = GameSpec.symmetric(2, OBS, ACT)
    with pytest.raises(ValueError, match="2 seats"):
        permute_seats(spec, "2p", [SeatAssignment("a")], random.Random(0))


def test_coordinator_draws_from_the_mixture_with_per_agent_halflives_and_checks_lineups(tmp_path):
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.registry import env_spec
    from colosseum.core.types import Lineup
    from colosseum.league.base import BaseMatchmaker
    from colosseum.players.registry import resolve_player_roles
    from game_helpers import make_test_config, scripted_agent

    config = make_test_config(
        "turns", agents={"a": {"matchmaking": {"pfsp": {"halflife_games": 50}}}, "b": {}, "bot": scripted_agent()},
        matchmaking={"opponents": {"latest": {0: 1.0, 1000: 0.0}, "snapshots": 0.0, "rivals": 0.0,
                                   "anchors": {0: 0.0, 1000: 1.0}}})
    spec = env_spec(config)
    steps = StepCounter(0)
    coordinator = Coordinator(config, spec, resolve_player_roles(config, spec), tmp_path / "ckpt", env_steps=steps)
    assert coordinator.pfsp._halflife == {"a": 50.0, "b": 200.0}
    assert isinstance(coordinator.matchmaker, MixtureMatchmaker)
    assert coordinator.context.trainable == ["a", "b"] and coordinator.context.fixed == ["bot"]
    lineups = coordinator.generate_lineups(4, 0)
    assert [next(s.agent_id for s in lu.seats if s.source == SOURCE_OWNER) for lu in lineups] == ["a", "b", "a", "b"]
    assert all(s.network_id == LATEST_NETWORK_ID for lu in lineups for s in lu.seats)
    steps.value = 1000                                  # the coordinator's env steps drive the schedules
    for lineup in coordinator.generate_lineups(4, 0):
        assert [(s.agent_id, s.network_id, s.collect, s.source) for s in lineup.seats if s.source != SOURCE_OWNER] \
            == [("bot", FIXED_NETWORK_ID, False, "anchors")]

    class Broken(BaseMatchmaker):
        def lineup_for(self, owner: str) -> Lineup:
            return Lineup(lineup_layout, [SeatAssignment(owner), SeatAssignment("bot")])

    lineup_layout = lineups[0].layout
    coordinator._matchmaker = Broken(coordinator.context)
    with pytest.raises(ValueError, match="Broken: invalid lineup .*'bot' must play network 'fixed'"):
        coordinator.generate_lineups(1, 0)

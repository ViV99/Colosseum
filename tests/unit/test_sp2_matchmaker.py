"""LineupMatchmaker: spec block 7 algorithm on FFA, teams, coop and asymmetric layouts (T5.1)."""
from __future__ import annotations

import random
from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator.matchmaker import LineupMatchmaker, permute_seats, validate_matchmaking
from colosseum.sp2.core.config import MatchmakingConfig
from colosseum.sp2.core.types import LATEST_NETWORK_ID, SeatAssignment
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))
DRAWS = 2000


def hunt_spec() -> GameSpec:
    """One hunter (team 0) against three prey (team 1)."""
    return GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )


def make_mm(spec, agent_roles, *, seed=0, ckpts=None, win_rate=None, **config) -> LineupMatchmaker:
    return LineupMatchmaker(
        spec=spec, agent_roles=agent_roles, config=MatchmakingConfig(**config),
        checkpoints=lambda agent_id: list((ckpts or {}).get(agent_id, [])),
        win_rate=win_rate or (lambda layout, a, b: 0.5),
        rng=random.Random(seed),
    )


def team_members(spec, lineup) -> list[list[SeatAssignment]]:
    return [[lineup.seats[s] for s in members] for members in spec.teams(lineup.layout)]


def check_lineup(spec, agent_roles, lineup) -> None:
    """Every seat is filled by an agent that plays its role; exactly latest seats collect."""
    seat_specs = spec.layouts[lineup.layout]
    assert len(lineup.seats) == len(seat_specs)
    for seat, seat_spec in zip(lineup.seats, seat_specs, strict=True):
        assert seat_spec.role in agent_roles[seat.agent_id], (seat, seat_spec)
        assert seat.collect == (seat.network_id == LATEST_NETWORK_ID), seat


def owner_collects(lineup, owner) -> bool:
    return any(s.agent_id == owner and s.network_id == LATEST_NETWORK_ID and s.collect for s in lineup.seats)


def test_ffa_15_players_self_play_mixes_latest_and_checkpoints():
    spec = GameSpec.symmetric(15, OBS, ACT)
    roles = {"a": ["player"]}
    mm = make_mm(spec, roles, ckpts={"a": ["ckpt_v1", "ckpt_v2"]}, mode="self_play", latest_prob=0.5)
    lineups = [mm.lineup_for("a") for _ in range(400)]
    networks = Counter()
    for lineup in lineups:
        assert lineup.layout == "15p"
        check_lineup(spec, roles, lineup)
        assert owner_collects(lineup, "a")
        networks.update(s.network_id for s in lineup.seats)
    # 1 owner seat (latest) + 14 opponents, each latest with probability 0.5
    total = sum(networks.values())
    assert abs(networks[LATEST_NETWORK_ID] / total - 8 / 15) < 0.03, networks
    assert networks["ckpt_v1"] > 0 and networks["ckpt_v2"] > 0


def test_ffa_arena_draws_opponents_by_pfsp_among_other_agents():
    spec = GameSpec.symmetric(15, OBS, ACT)
    roles = {"a": ["player"], "b": ["player"], "c": ["player"]}
    rates = {("a", "b"): 0.9, ("a", "c"): 0.1}
    mm = make_mm(spec, roles, mode="league", self_play_ratio=0.0,
                 win_rate=lambda layout, x, y: rates.get((x, y), 0.5))
    opponents = Counter()
    for _ in range(200):
        lineup = mm.lineup_for("a")
        check_lineup(spec, roles, lineup)
        agents = [s.agent_id for s in lineup.seats]
        assert agents.count("a") == 1  # one seat per team: the owner's team only
        assert all(s.collect for s in lineup.seats)
        opponents.update(a for a in agents if a != "a")
    # PFSP weights (1 - 0.9) vs (1 - 0.1): c is drawn 90% of the time
    assert abs(opponents["c"] / sum(opponents.values()) - 0.9) < 0.03, opponents


def test_2v2_teams_are_homogeneous_with_teammates_self():
    spec = GameSpec.teams_of([2, 2], OBS, ACT)
    roles = {"a": ["player"], "b": ["player"]}
    mm = make_mm(spec, roles, ckpts={"a": ["ckpt_v3"]}, mode="league", self_play_ratio=0.5)
    kinds = Counter()
    for _ in range(300):
        lineup = mm.lineup_for("a")
        assert lineup.layout == "2v2"
        check_lineup(spec, roles, lineup)
        assert owner_collects(lineup, "a")
        teams = team_members(spec, lineup)
        for members in teams:
            assert len({(s.agent_id, s.network_id) for s in members}) == 1, members
        kinds[tuple(sorted(members[0].agent_id for members in teams))] += 1
    assert kinds[("a", "b")] > 0 and kinds[("a", "a")] > 0  # arena and self-play both happen


def test_2v1v1_arena_fills_each_team_with_one_core():
    spec = GameSpec.teams_of([2, 1, 1], OBS, ACT)
    roles = {"a": ["player"], "b": ["player"], "c": ["player"]}
    mm = make_mm(spec, roles, mode="league", self_play_ratio=0.0)
    owner_team_sizes = Counter()
    for _ in range(300):
        lineup = mm.lineup_for("a")
        assert lineup.layout == "2v1v1"
        check_lineup(spec, roles, lineup)
        teams = team_members(spec, lineup)
        with_owner = [members for members in teams if any(s.agent_id == "a" for s in members)]
        assert len(with_owner) == 1  # the owner is the core of exactly one team
        assert all(s.agent_id == "a" for s in with_owner[0])
        owner_team_sizes[len(with_owner[0])] += 1
        for members in teams:
            assert len({s.agent_id for s in members}) == 1
    assert set(owner_team_sizes) == {1, 2}  # the owner's team is drawn among all teams


def test_one_hunter_vs_three_prey_under_self_play_mode():
    """Asymmetric roles work under mode: self_play: the team whose roles the owner does not
    play is filled by another agent (PFSP fallback), so every lineup is complete."""
    spec = hunt_spec()
    roles = {"hunter_agent": ["hunter"], "prey_agent": ["prey"]}
    mm = make_mm(spec, roles, ckpts={"hunter_agent": ["ckpt_v1"], "prey_agent": ["ckpt_v1"]},
                 mode="self_play", latest_prob=0.0)
    for owner in ("hunter_agent", "prey_agent"):
        for _ in range(100):
            lineup = mm.lineup_for(owner)
            check_lineup(spec, roles, lineup)
            assert owner_collects(lineup, owner)
            hunter, *prey = lineup.seats
            assert hunter.agent_id == "hunter_agent" and all(p.agent_id == "prey_agent" for p in prey)
            assert all(s.network_id == LATEST_NETWORK_ID for s in lineup.seats)  # PFSP picks latest


def test_two_prey_agents_mix_in_the_prey_team_with_teammates_mixed():
    spec = hunt_spec()
    roles = {"h": ["hunter"], "p1": ["prey"], "p2": ["prey"]}
    mm = make_mm(spec, roles, mode="league", self_play_ratio=0.0, teammates="mixed", teammate_self_prob=0.5)
    prey_teams = Counter()
    for _ in range(300):
        lineup = mm.lineup_for("h")
        check_lineup(spec, roles, lineup)
        prey_teams[len({s.agent_id for s in lineup.seats[1:]})] += 1
    assert prey_teams[1] > 0 and prey_teams[2] > 0


def test_coop_with_teammates_self_is_all_owner_latest():
    spec = GameSpec.teams_of([3], OBS, ACT)
    roles = {"a": ["player"], "b": ["player"]}
    mm = make_mm(spec, roles, ckpts={"a": ["ckpt_v1"]}, mode="league", self_play_ratio=0.0)
    for owner in ("a", "b"):
        lineup = mm.lineup_for(owner)
        assert lineup.layout == "coop3"
        assert [(s.agent_id, s.network_id, s.collect) for s in lineup.seats] == [(owner, "latest", True)] * 3


def test_coop_with_teammates_mixed_draws_teammates_by_the_rule():
    spec = GameSpec.teams_of([2], OBS, ACT)
    roles = {"a": ["player"], "b": ["player"]}
    mm = make_mm(spec, roles, ckpts={"a": ["ckpt_v3"]}, teammates="mixed", teammate_self_prob=0.5,
                 shuffle_seats=False)
    mates = Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("a")
        check_lineup(spec, roles, lineup)
        assert owner_collects(lineup, "a")
        core_seat = next(i for i, s in enumerate(lineup.seats) if (s.agent_id, s.network_id) == ("a", "latest"))
        other = lineup.seats[1 - core_seat]
        mates[(other.agent_id, other.network_id)] += 1
    # core with 0.5; else uniform over [b latest, a ckpt_v3]
    assert abs(mates[("a", "latest")] / DRAWS - 0.5) < 0.05, mates
    assert abs(mates[("b", "latest")] / DRAWS - 0.25) < 0.05, mates
    assert abs(mates[("a", "ckpt_v3")] / DRAWS - 0.25) < 0.05, mates


def test_layout_weights_are_followed_within_5_percent():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    mm = make_mm(spec, {"a": ["player"]}, layouts={"2p": 0.25, "4p": 0.75})
    counts = Counter(mm.lineup_for("a").layout for _ in range(4000))
    assert abs(counts["2p"] / 4000 - 0.25) < 0.05 and abs(counts["4p"] / 4000 - 0.75) < 0.05, counts
    assert mm.playable_layouts("a") == ["2p", "4p"]


def test_layouts_are_restricted_to_those_with_a_seat_of_the_owners_roles():
    spec = GameSpec(
        roles={"player": RoleSpec(OBS, ACT), "hunter": HUNTER, "prey": PREY},
        layouts={"duel": (SeatSpec("player", 0), SeatSpec("player", 1)),
                 "hunt": (SeatSpec("hunter", 0), SeatSpec("prey", 1))},
    )
    roles = {"p": ["player"], "h": ["hunter"], "q": ["prey"]}
    mm = make_mm(spec, roles)
    assert mm.playable_layouts("h") == ["hunt"] and mm.playable_layouts("p") == ["duel"]
    assert {mm.lineup_for("h").layout for _ in range(50)} == {"hunt"}
    assert {mm.lineup_for("p").layout for _ in range(50)} == {"duel"}


def test_mode_self_play_ignores_self_play_ratio():
    spec = GameSpec.symmetric(2, OBS, ACT)
    mm = make_mm(spec, {"a": ["player"], "b": ["player"]}, mode="self_play", self_play_ratio=0.0)
    assert mm.self_play_ratio == 1.0
    assert all({s.agent_id for s in mm.lineup_for("a").seats} == {"a"} for _ in range(100))


def test_seat_of_a_role_the_core_does_not_play_goes_to_any_latest_player_of_it():
    """Teams of mixed roles: a seat whose role the core does not play gets the latest weights of a
    uniformly drawn agent with that role, the owner included (spec block 7, item 4)."""
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"pairs": (SeatSpec("hunter", 0), SeatSpec("prey", 0), SeatSpec("hunter", 1), SeatSpec("prey", 1))},
    )
    roles = {"h": ["hunter"], "h2": ["hunter"], "q": ["prey"]}
    mm = make_mm(spec, roles, ckpts={"h": ["ckpt_v1"]}, mode="league", self_play_ratio=0.0, shuffle_seats=False)
    hunters = Counter()
    for _ in range(DRAWS):
        lineup = mm.lineup_for("h")
        check_lineup(spec, roles, lineup)
        assert all(s.collect for s in lineup.seats)
        assert lineup.seats[1].agent_id == lineup.seats[3].agent_id == "q"  # the only prey player
        hunters[tuple(sorted((lineup.seats[0].agent_id, lineup.seats[2].agent_id)))] += 1
    # opposing core h2 or q (PFSP, equal weights); with core q its hunter seat is h or h2 uniformly
    assert set(hunters) == {("h", "h"), ("h", "h2")}, hunters
    assert abs(hunters[("h", "h")] / DRAWS - 0.25) < 0.05, hunters


def test_permute_seats_keeps_team_and_role_structure():
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"mix": (
            SeatSpec("hunter", 0), SeatSpec("prey", 0), SeatSpec("prey", 0),
            SeatSpec("hunter", 1), SeatSpec("prey", 1), SeatSpec("prey", 1),
            SeatSpec("hunter", 2),
        )},
    )
    seat_specs = spec.layouts["mix"]
    labels = [SeatAssignment(agent_id=f"s{i}") for i in range(7)]
    rng = random.Random(0)
    team_swaps = within_team_swaps = 0
    for _ in range(500):
        out = permute_seats(spec, "mix", labels, rng)
        assert sorted(s.agent_id for s in out) == sorted(s.agent_id for s in labels)
        origin = {s.agent_id: i for i, s in enumerate(labels)}
        for target, assignment in enumerate(out):
            source = origin[assignment.agent_id]
            assert seat_specs[source].role == seat_specs[target].role
        for members in spec.teams("mix"):
            assert len({seat_specs[origin[out[s].agent_id]].team for s in members}) == 1  # teams move whole
        assert out[6].agent_id == "s6"  # the only team of its composition never moves
        team_swaps += out[0].agent_id == "s3"
        within_team_swaps += out[1].agent_id in ("s2", "s5")
    assert team_swaps > 0 and within_team_swaps > 0


def test_permute_seats_rejects_a_wrong_seat_count():
    spec = GameSpec.symmetric(2, OBS, ACT)
    with pytest.raises(ValueError, match="2 seats"):
        permute_seats(spec, "2p", [SeatAssignment("a")], random.Random(0))


def test_validate_matchmaking_errors():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    with pytest.raises(ConfigError, match="unknown layouts"):
        validate_matchmaking(spec, {"a": ["player"]}, MatchmakingConfig(layouts={"3p": 1.0}))
    with pytest.raises(ConfigError, match="> 0"):
        validate_matchmaking(spec, {"a": ["player"]}, MatchmakingConfig.model_construct(layouts={"2p": 0.0}))
    hunt = hunt_spec()
    with pytest.raises(ConfigError, match="prey"):
        validate_matchmaking(hunt, {"h": ["hunter"]}, MatchmakingConfig())
    mixed = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": hunt.layouts["1v3"], "prey_only": (SeatSpec("prey", 0), SeatSpec("prey", 1))},
    )
    with pytest.raises(ConfigError, match="agent 'h'"):
        validate_matchmaking(mixed, {"h": ["hunter"], "q": ["prey"]}, MatchmakingConfig(layouts={"prey_only": 1.0}))
    with pytest.raises(ConfigError):  # the constructor runs the same checks
        make_mm(hunt, {"h": ["hunter"]})

"""Tournament schedules (SP4 T5.2, spec block 3): blocks, admissible rotations, FFA compositions, score layouts,
seed domains, role groups."""
from __future__ import annotations

import collections

import pytest

from colosseum.arena.tournament.engine import check_blocks
from colosseum.arena.tournament.schedule import (
    check_role_groups,
    compositions,
    gauntlet_blocks,
    round_robin_blocks,
    score_blocks,
    stage_seed,
)
from colosseum.core.errors import ConfigError
from game_helpers import (
    AsymmetricGame,
    CoopGame,
    EliminationFFA,
    TeamDeadTeammateGame,
    TurnTakingGame,
    synthetic_players,
)

NAMES = ["main@v10", "main@v20", "greedy", "random", "old"]
ROLES = {"h@v1": ("hunter",), "y@v1": ("prey",), "bot": ("hunter", "prey"), "preybot": ("prey",)}
PLAYERS = synthetic_players([*NAMES, *ROLES], ROLES)


def who(lineup) -> list[tuple[str, str]]:
    return [(s.agent_id, s.network_id) for s in lineup.seats]


def test_1v1_blocks_hold_both_sides_and_every_candidate_gets_the_same_seeds():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "main@v20"], ["greedy", "random"], PLAYERS, blocks=3, seed=7,
                             stage="A", prefix="t1")
    assert len(blocks) == 2 * 2 * 3
    b = blocks[0]
    assert (b.candidate, b.composition, b.index, b.rotations) == ("main@v10", ("greedy",), 0, (0, 1))
    assert who(b.lineups[0]) == [("main", "ckpt_v10"), ("greedy", "fixed")]
    assert who(b.lineups[1]) == [("greedy", "fixed"), ("main", "ckpt_v10")]
    assert all(not s.collect for lu in b.lineups for s in lu.seats)
    assert b.id == "t1/A/2p/main@v10/greedy/0"
    assert [lu.uid for lu in b.lineups] == [f"{b.id}/r0", f"{b.id}/r1"]
    assert {lu.seed for lu in b.lineups} == {b.seed} and all(lu.block == b.id for lu in b.lineups)
    seeds = collections.defaultdict(set)
    for blk in blocks:
        seeds[(blk.composition, blk.index)].add((blk.candidate, blk.seed))
    for cell in seeds.values():                       # one seed per (composition, index) for both candidates
        assert len({s for _c, s in cell}) == 1 and {c for c, _s in cell} == {"main@v10", "main@v20"}
    assert len({blk.seed for blk in blocks}) == 3      # one seed per index, shared by the compositions too


def test_seed_domains_never_collide_and_never_repeat_within_a_domain():
    n = 50_000
    domains = {stage: {stage_seed(3, stage, k) for k in range(n)} for stage in ("A", "B", "P")}
    assert all(len(seeds) == n for seeds in domains.values())
    assert not (domains["A"] & domains["B"]) and not (domains["A"] & domains["P"]) and not (domains["B"] & domains["P"])
    assert all(0 <= s < 2**32 for seeds in domains.values() for s in seeds)
    assert stage_seed(3, "A", 5) == stage_seed(3, "A", 5) and stage_seed(3, "A", 5) != stage_seed(4, "A", 5)
    with pytest.raises(ValueError):
        stage_seed(0, "X", 0)
    with pytest.raises(ValueError):
        stage_seed(0, "A", 2**30)


def test_ffa_compositions_meet_every_field_member_equally_often():
    assert compositions(["a", "b", "c"], 2) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert compositions(["a", "b"], 3) == [("a", "a", "a"), ("a", "a", "b"), ("a", "b", "b"), ("b", "b", "b")]
    spec = EliminationFFA(4).spec
    for field in (["greedy", "random", "old"], ["greedy", "random"]):
        blocks = gauntlet_blocks(spec, "4p", ["main@v10"], field, PLAYERS, blocks=2, seed=0, stage="A")
        meetings = collections.Counter()
        seats = collections.Counter()
        for b in blocks:
            assert len(b.lineups) == 4 and b.rotations == (0, 1, 2, 3)
            for lu in b.lineups:
                meetings.update(s.agent_id for s in lu.seats if s.agent_id != "main")
                mine = [i for i, s in enumerate(lu.seats) if s.agent_id == "main"]
                assert len(mine) == 1
                seats[mine[0]] += 1
        assert len(set(meetings.values())) == 1, meetings
        assert len(set(seats.values())) == 1, seats      # the candidate sits on every seat equally often


def test_asymmetric_roles_keep_only_admissible_rotations():
    spec = AsymmetricGame().spec          # 1v2: team 0 = hunter, team 1 = two prey
    hunters = gauntlet_blocks(spec, "1v2", ["h@v1"], ["preybot", "bot"], PLAYERS, blocks=1, seed=0, stage="A")
    assert [(b.composition, b.rotations) for b in hunters] == [(("preybot",), (0,)), (("bot",), (0,))]
    prey = gauntlet_blocks(spec, "1v2", ["y@v1"], ["bot"], PLAYERS, blocks=1, seed=0, stage="A")
    assert [(b.composition, b.rotations) for b in prey] == [(("bot",), (1,))]
    assert who(prey[0].lineups[0]) == [("bot", "fixed"), ("y", "ckpt_v1"), ("y", "ckpt_v1")]
    with pytest.raises(ConfigError, match="cannot fill its teams"):
        gauntlet_blocks(spec, "1v2", ["y@v1"], ["preybot"], PLAYERS, blocks=1, seed=0, stage="A")
    assert check_role_groups(spec, "1v2", ["h@v1", "y@v1"], PLAYERS) == {"hunter": ["h@v1"], "prey": ["y@v1"]}
    with pytest.raises(ConfigError, match="overlapping role groups"):
        check_role_groups(spec, "1v2", ["h@v1", "bot"], PLAYERS)


def test_teams_of_two_seat_a_whole_team_per_entity():
    spec = TeamDeadTeammateGame().spec    # 2v2
    (b,) = gauntlet_blocks(spec, "2v2", ["main@v10"], ["greedy"], PLAYERS, blocks=1, seed=0, stage="A")
    assert who(b.lineups[0]) == [("main", "ckpt_v10")] * 2 + [("greedy", "fixed")] * 2
    assert who(b.lineups[1]) == [("greedy", "fixed")] * 2 + [("main", "ckpt_v10")] * 2


def test_score_layouts_homogeneous_team_and_cross_play_with_a_scripted_teammate():
    spec = CoopGame(size=2).spec
    blocks = score_blocks(spec, "coop2", ["main@v10"], ["greedy"], PLAYERS, blocks=2, seed=0, stage="A")
    assert [(b.composition, b.rotations, b.index) for b in blocks] == [((), (0,), 0), ((), (0,), 1),
                                                                       (("greedy",), (0, 1), 0),
                                                                       (("greedy",), (0, 1), 1)]
    assert who(blocks[2].lineups[0]) == [("greedy", "fixed"), ("main", "ckpt_v10")]
    assert who(blocks[2].lineups[1]) == [("main", "ckpt_v10"), ("greedy", "fixed")]
    assert blocks[0].id.endswith("/main@v10/-/0") and blocks[0].stratum == "-"


def test_round_robin_pairs_and_its_layout_rule():
    spec = TurnTakingGame().spec
    blocks = round_robin_blocks(spec, "2p", ["main@v10", "main@v20", "greedy"], PLAYERS, blocks=2, seed=0)
    assert collections.Counter((b.candidate, b.composition) for b in blocks) == {
        ("main@v10", ("main@v20",)): 2, ("main@v10", ("greedy",)): 2, ("main@v20", ("greedy",)): 2}
    with pytest.raises(ConfigError, match="two-team"):
        round_robin_blocks(EliminationFFA(4).spec, "3p", ["main@v10", "main@v20"], PLAYERS, blocks=1, seed=0)


def test_first_index_continues_the_seed_sequence_and_bad_arguments():
    spec = TurnTakingGame().spec
    first = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=2, seed=1, stage="A")
    more = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=2, seed=1, stage="A", first_index=2)
    assert [b.index for b in more] == [2, 3] and not {b.seed for b in first} & {b.seed for b in more}
    confirm = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=4, seed=1, stage="B")
    assert not {b.seed for b in first + more} & {b.seed for b in confirm}
    with pytest.raises(ConfigError, match="field is empty"):
        gauntlet_blocks(spec, "2p", ["main@v10"], [], PLAYERS, blocks=1, seed=0, stage="A")
    with pytest.raises(ConfigError, match="not a layout"):
        gauntlet_blocks(spec, "9p", ["main@v10"], ["greedy"], PLAYERS, blocks=1, seed=0, stage="A")
    with pytest.raises(ValueError, match="prefix"):
        gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=1, seed=0, stage="A", prefix="a/b")


def test_every_episode_gets_its_own_lineup_and_the_engine_accepts_every_builder():
    """T1.1's obligation: one lineup per episode (own uid, the block's seed and id), no shared mutable seats."""
    ffa, coop, turns = EliminationFFA(4).spec, CoopGame(size=2).spec, TurnTakingGame().spec
    schedules = [
        gauntlet_blocks(ffa, "4p", ["main@v10", "main@v20"], ["greedy", "random"], PLAYERS, blocks=3, seed=0,
                        stage="A"),
        score_blocks(coop, "coop2", ["main@v10", "main@v20"], ["greedy", "random"], PLAYERS, blocks=3, seed=0,
                     stage="B"),
        round_robin_blocks(turns, "2p", ["main@v10", "main@v20", "greedy"], PLAYERS, blocks=3, seed=0),
    ]
    for blocks in schedules:
        check_blocks(blocks)
        lineups = [lu for b in blocks for lu in b.lineups]
        assert len({lu.uid for lu in lineups}) == len(lineups) == len({id(lu) for lu in lineups})
        assert len({id(s) for lu in lineups for s in lu.seats}) == sum(len(lu.seats) for lu in lineups)
        assert len({id(lu.seats) for lu in lineups}) == len(lineups)
    check_blocks([b for blocks in schedules for b in blocks])


def test_repeated_or_unknown_names_are_refused():
    spec = TurnTakingGame().spec
    with pytest.raises(ValueError, match="more than once"):
        gauntlet_blocks(spec, "2p", ["main@v10", "main@v10"], ["greedy"], PLAYERS, blocks=1, seed=0, stage="A")
    with pytest.raises(ValueError, match="more than once"):
        gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy", "greedy"], PLAYERS, blocks=1, seed=0, stage="A")
    with pytest.raises(ValueError, match="more than once"):
        round_robin_blocks(spec, "2p", ["greedy", "greedy"], PLAYERS, blocks=1, seed=0)
    with pytest.raises(ValueError, match="no player"):
        score_blocks(CoopGame(size=2).spec, "coop2", ["nobody"], [], PLAYERS, blocks=1, seed=0, stage="A")

# SP2 Plan — Part C: League (matchmaking, ratings, coordinator, launcher), Eval, Validate, BC, Distributed, and the Switch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part turns the SP2 core (Parts A and B: env contract, specs, config v2, distributions, model protocol, roles, chunk v2, `MatchRunner`, `RolloutLoop`, V-trace, APPO v2, learner) into a usable system, then switches the repository to it.
- **Spec block 7 (league):** `LineupMatchmaker` (layouts, match type, owner's team, team cores, teammates, permutations), ratings per layout (team-pair ELO, win rates, `wr_vs_past`, `ScoreTracker`, cross-play), the coordinator with owner rotation, roles and role signatures in checkpoints, metrics per layout and role.
- **Launcher and CLI `train`** on the `sp2` types, with integration runs on toy games.
- **Spec block 8 (eval):** in-process API on `MatchRunner`, CLI rules, reports per outcome kind.
- **Spec block 9:** `validate_config` and `colosseum validate`; `meta.json` roles and signature (T5.3).
- **BC** on trees (`bc --agent`), **distributed mode** on chunk v2 (spec block 10 bullet).
- **The switch (T7.1–T7.3):** SP1 integration guarantees ported to the `sp2` CLI, the new tic-tac-toe, and the overlay that moves `colosseum.sp2` onto `colosseum` and deletes the legacy.

**Read `00-overview.md` first** (global constraints, shadow package strategy, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Execution order inside this part:** T5.1 → T5.2 → T5.3 → T5.4 → **T6.2 → T6.1** → T6.3 → T6.4 → **T7.2 → T7.1** → T7.3. The task sections below follow the IDs; execute them in this order.
- T6.2 runs first among the T6 tasks: the `eval`, `bc` and distributed entry points call `validate_config` (T6.2), as SP1's entry points do.
- T7.1 runs after T7.2: its ports use the new tic-tac-toe example and configs that T7.2 creates (Contract notes).
- T6.1–T6.4 each insert one block into `src/colosseum/sp2/cli.py`, always directly above the final `if __name__ == "__main__":` line.

**Conventions used in every task.**
- Run every command from the repository root with `.venv/bin/python` / `.venv/bin/ruff`.
- "Full fast suite" means `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `.venv/bin/ruff check .`.
- New test files have basenames that start with `test_sp2_` (unique across `tests/`); new support code goes into the shared kits (`tests/game_helpers.py`, `tests/learning/game_learning_envs.py`).
- **Copy steps** are Python scripts. They copy an SP1 module and apply exact edits: text edits assert that the old text occurs exactly once; function and method replacements locate the definition by name (`ast` line ranges), so they do not depend on unrelated changes T0.1 made to the same file. Each script is followed by `ruff check --fix` on its outputs (it only sorts imports and drops imports the edits made unused) and a plain `ruff check`. If an assertion of a script fails because Parts A/B or T0.1 changed the anchor text, apply the shown replacement to the equivalent statement (overview, "Cross-part execution notes") and say so in the commit body.
- Commit messages use conventional prefixes, no attribution lines; push the branch after every task (`git push origin sp2-game-model`).

**Shared helpers this part adds to `tests/game_helpers.py`** (Part A created the file with the toy games, `make_test_model`, `RandomPolicy`, `CORE_KINDS`): `GameTestModel`, `TEST_GAME_AGENTS`, `make_test_config`, `write_test_config`, `make_coordinator`, `agent_role_of` (T5.3); `make_test_run_dir` (T5.4); `CrashingAPPO` (T7.1). It reuses Part A's `TOY_GAMES` (game names), `make_test_model` and Part B's `chunk_v2_payload` / `learner_role` (T4.4 kit) instead of near-duplicates. `tests/learning/game_learning_envs.py` is new (T6.3, extended in T7.2).

---

### Task T5.1: `LineupMatchmaker`, `validate_matchmaking`, `permute_seats`

Spec block 7, items 1–6 and the config checks. One algorithm builds the lineup of every game structure: FFA, teams, 2v1v1, 1 vs N with asymmetric roles, cooperative layouts; `mode: self_play` is `self_play_ratio = 1`, and asymmetric games still get complete lineups under it (the team whose roles the owner does not play gets another agent through the PFSP branch).

**Files:**
- Create: `src/colosseum/sp2/coordinator/__init__.py` (empty; skip if it exists)
- Create: `src/colosseum/sp2/coordinator/matchmaker.py`
- Test: `tests/unit/test_sp2_matchmaker.py`

**Interfaces:**
- Consumes:
  - `GameSpec`, `RoleSpec`, `SeatSpec` (`colosseum.sp2.envs.game`, T1.4): `spec.layouts`, `spec.teams(layout)`, `spec.layout_size(layout)`; `GameSpec.symmetric(n | [n...], obs, act)` (layouts `"<n>p"`), `GameSpec.teams_of(sizes, obs, act)` (`"2v2"`, `"2v1v1"`, `"coop<n>"`).
  - `MatchmakingConfig` (`colosseum.sp2.core.config`, T1.7): `mode`, `layouts`, `self_play_ratio`, `pfsp_exponent`, `latest_prob`, `teammates`, `teammate_self_prob`, `shuffle_seats`.
  - `LATEST_NETWORK_ID`, `SeatAssignment`, `Lineup` (`colosseum.sp2.core.types`, T3.1).
  - `ConfigError` (`colosseum.core.errors`).
- Produces (contract, plus additions marked *):
  - `LineupMatchmaker(*, spec, agent_roles, config, checkpoints, win_rate, rng)` with `lineup_for(owner) -> Lineup`, `playable_layouts(agent_id) -> list[str]`, *`pfsp_weight(layout, owner, candidate) -> float`, *`self_play_ratio` (property). The constructor runs `validate_matchmaking`.
  - `validate_matchmaking(spec, agent_roles, config) -> None` (ConfigError).
  - `permute_seats(spec, layout, seats, rng) -> list[SeatAssignment]`.
  - *`enabled_layouts(spec, config) -> dict[str, float]` (used by T5.3 ratings consumers, T6.2 validate, T6.4 distributed).
  - *`PFSP_MIN_WEIGHT = 1e-6`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp2_matchmaker.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_matchmaker.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.coordinator.matchmaker'`.

- [ ] **Step 3: Write the implementation**

Create `src/colosseum/sp2/coordinator/__init__.py` (empty) if it is missing, then `src/colosseum/sp2/coordinator/matchmaker.py`:

```python
"""Lineup matchmaking (spec block 7): one ``Lineup`` per env, built for an owner agent.

A lineup is a layout (teams of seats with roles) plus one ``SeatAssignment`` per seat. One
algorithm covers solo, FFA, teams, cooperative and asymmetric games:

1. Layout: drawn by the ``matchmaking.layouts`` weights (every layout, equal weights, when
   empty), only among layouts with a seat of one of the owner's roles.
2. Match type, once per match: self-play with probability ``self_play_ratio`` (always under
   ``mode: self_play``), else arena.
3. Owner's team: uniform among the teams with a seat of one of the owner's roles.
4. A core per team. The owner's team: the owner's latest weights. Every other team, drawn
   independently: under self-play, when the owner plays a role of the team, the owner's latest
   weights (probability ``latest_prob``) or one of its checkpoints; otherwise (arena, or the
   owner plays none of the team's roles) an agent drawn by PFSP among the OTHER trainable agents
   that play a role of the team, ``f(wr) = (1 - wr) ** pfsp_exponent`` with the win rate of the
   layout; without such agents, the self-play choice.
5. The core takes one seat of a role it plays. The team's other seats of roles the core plays
   follow ``teammates``: ``self`` gives them to the core; ``mixed`` gives each to the core with
   probability ``teammate_self_prob``, else uniformly to the latest weights of another trainable
   agent with that role or to a checkpoint of the core's agent. A seat of a role the core does
   not play gets the latest weights of a uniformly drawn trainable agent that plays it (the
   owner included).
6. Seats with latest weights collect; checkpoint seats do not.
7. ``shuffle_seats``: ``permute_seats`` permutes teams of equal role composition and, inside a
   team, seats of the same role.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import MatchmakingConfig
from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.sp2.envs.game import GameSpec

logger = logging.getLogger(__name__)

PFSP_MIN_WEIGHT = 1e-6


def enabled_layouts(spec: GameSpec, config: MatchmakingConfig) -> dict[str, float]:
    """``{layout: weight}`` matchmaking draws from: ``config.layouts``, or every layout with weight 1."""
    if not config.layouts:
        return {name: 1.0 for name in spec.layouts}
    return {name: float(weight) for name, weight in config.layouts.items()}


def validate_matchmaking(spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                         config: MatchmakingConfig) -> None:
    """Spec block 7 checks; ConfigError with a fix hint.

    - ``matchmaking.layouts`` names only layouts of the game, with weights > 0;
    - every role of every enabled layout is played by at least one agent;
    - every agent has a seat in at least one enabled layout.
    """
    unknown = sorted(set(config.layouts) - set(spec.layouts))
    if unknown:
        raise ConfigError(
            f"matchmaking.layouts names unknown layouts {unknown}; the game's layouts are {sorted(spec.layouts)}"
        )
    not_positive = sorted(name for name, weight in config.layouts.items() if not weight > 0)
    if not_positive:
        raise ConfigError(f"matchmaking.layouts weights must be > 0; fix {not_positive}")
    layouts = enabled_layouts(spec, config)
    played = {role for roles in agent_roles.values() for role in roles}
    for name in layouts:
        missing = sorted({seat.role for seat in spec.layouts[name]} - played)
        if missing:
            raise ConfigError(
                f"layout {name!r}: no trainable agent plays role(s) {missing}; add them to some "
                f"agents.<id>.roles or leave {name!r} out of matchmaking.layouts"
            )
    for agent_id, roles in agent_roles.items():
        if not any(seat.role in roles for name in layouts for seat in spec.layouts[name]):
            raise ConfigError(
                f"agent {agent_id!r} (roles {list(roles)}) has no seat in the enabled layouts "
                f"{sorted(layouts)}; check agents.{agent_id}.roles and matchmaking.layouts"
            )


def permute_seats(spec: GameSpec, layout: str, seats: Sequence[SeatAssignment],
                  rng: random.Random) -> list[SeatAssignment]:
    """Randomly permute ``seats`` while keeping the layout's structure.

    Whole teams move only onto teams with the same multiset of roles; inside a team, an
    assignment moves only onto a seat of the same role.
    """
    seat_specs = spec.layouts[layout]
    if len(seats) != len(seat_specs):
        raise ValueError(f"permute_seats: layout {layout!r} has {len(seat_specs)} seats, got {len(seats)}")
    teams = spec.teams(layout)
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for team, members in enumerate(teams):
        groups[tuple(sorted(seat_specs[s].role for s in members))].append(team)
    out: list[SeatAssignment | None] = [None] * len(seats)
    for team_ids in groups.values():
        targets = list(team_ids)
        rng.shuffle(targets)
        for source, target in zip(team_ids, targets, strict=True):
            target_by_role: dict[str, list[int]] = defaultdict(list)
            for s in teams[target]:
                target_by_role[seat_specs[s].role].append(s)
            for role_seats in target_by_role.values():
                rng.shuffle(role_seats)
            for s in teams[source]:
                out[target_by_role[seat_specs[s].role].pop()] = seats[s]
    return out  # type: ignore[return-value]


def _seat(agent_id: str, network_id: str) -> SeatAssignment:
    """Latest weights collect, checkpoints never do."""
    return SeatAssignment(agent_id=agent_id, network_id=network_id, collect=network_id == LATEST_NETWORK_ID)


class LineupMatchmaker:
    """Builds the lineup of one match whose training data belongs to ``owner`` (module docstring)."""

    def __init__(
        self,
        *,
        spec: GameSpec,
        agent_roles: Mapping[str, Sequence[str]],
        config: MatchmakingConfig,
        checkpoints: Callable[[str], list[str]],
        win_rate: Callable[[str, str, str], float],
        rng: random.Random,
    ) -> None:
        validate_matchmaking(spec, agent_roles, config)
        self._spec = spec
        self._roles = {agent_id: frozenset(roles) for agent_id, roles in agent_roles.items()}
        self._agents = list(agent_roles)
        self._config = config
        self._checkpoints = checkpoints
        self._win_rate = win_rate
        self._rng = rng
        self._layouts = enabled_layouts(spec, config)
        self._self_play_ratio = 1.0 if config.mode == "self_play" else float(config.self_play_ratio)

    @property
    def self_play_ratio(self) -> float:
        return self._self_play_ratio

    def playable_layouts(self, agent_id: str) -> list[str]:
        """Enabled layouts with a seat of one of ``agent_id``'s roles, in config order."""
        roles = self._roles[agent_id]
        return [name for name in self._layouts if any(seat.role in roles for seat in self._spec.layouts[name])]

    def pfsp_weight(self, layout: str, owner: str, candidate: str) -> float:
        p = self._config.pfsp_exponent
        return max(PFSP_MIN_WEIGHT, (1.0 - self._win_rate(layout, owner, candidate)) ** p)

    def lineup_for(self, owner: str) -> Lineup:
        if owner not in self._roles:
            raise KeyError(f"unknown agent {owner!r}; known agents: {self._agents}")
        layouts = self.playable_layouts(owner)
        layout = self._rng.choices(layouts, weights=[self._layouts[name] for name in layouts], k=1)[0]
        seat_specs = self._spec.layouts[layout]
        teams = self._spec.teams(layout)
        self_play = self._rng.random() < self._self_play_ratio
        owner_teams = [t for t, members in enumerate(teams)
                       if any(seat_specs[s].role in self._roles[owner] for s in members)]
        owner_team = self._rng.choice(owner_teams)
        seats: list[SeatAssignment | None] = [None] * len(seat_specs)
        for team, members in enumerate(teams):
            if team == owner_team:
                core = (owner, LATEST_NETWORK_ID)
            else:
                core = self._opponent_core(owner, layout, members, self_play)
            self._fill_team(core, layout, members, seats)
        lineup_seats: list[SeatAssignment] = seats  # type: ignore[assignment]
        if self._config.shuffle_seats:
            lineup_seats = permute_seats(self._spec, layout, lineup_seats, self._rng)
        return Lineup(layout=layout, seats=lineup_seats)

    def _opponent_core(self, owner: str, layout: str, members: Sequence[int], self_play: bool) -> tuple[str, str]:
        team_roles = {self._spec.layouts[layout][s].role for s in members}
        owner_plays = bool(team_roles & self._roles[owner])
        if not (self_play and owner_plays):
            candidates = [a for a in self._agents if a != owner and team_roles & self._roles[a]]
            if candidates:
                weights = [self.pfsp_weight(layout, owner, c) for c in candidates]
                return self._rng.choices(candidates, weights=weights, k=1)[0], LATEST_NETWORK_ID
        return self._self_play_core(owner)

    def _self_play_core(self, owner: str) -> tuple[str, str]:
        checkpoints = self._checkpoints(owner)
        if not checkpoints or self._rng.random() < self._config.latest_prob:
            return owner, LATEST_NETWORK_ID
        return owner, self._rng.choice(checkpoints)

    def _fill_team(self, core: tuple[str, str], layout: str, members: Sequence[int],
                   seats: list[SeatAssignment | None]) -> None:
        agent_id, network_id = core
        seat_specs = self._spec.layouts[layout]
        own = [s for s in members if seat_specs[s].role in self._roles[agent_id]]
        core_seat = self._rng.choice(own) if own else None
        for s in members:
            role = seat_specs[s].role
            if role not in self._roles[agent_id]:
                players = [a for a in self._agents if role in self._roles[a]]
                seats[s] = _seat(self._rng.choice(players), LATEST_NETWORK_ID)
            elif s == core_seat or self._config.teammates == "self":
                seats[s] = _seat(agent_id, network_id)
            else:
                seats[s] = self._mixed_teammate(agent_id, network_id, role)

    def _mixed_teammate(self, agent_id: str, network_id: str, role: str) -> SeatAssignment:
        if self._rng.random() < self._config.teammate_self_prob:
            return _seat(agent_id, network_id)
        candidates = [(a, LATEST_NETWORK_ID) for a in self._agents if a != agent_id and role in self._roles[a]]
        candidates += [(agent_id, c) for c in self._checkpoints(agent_id)]
        if not candidates:
            return _seat(agent_id, network_id)
        return _seat(*self._rng.choice(candidates))
```

Notes for the reviewer:
- The match type is drawn once per match (before the owner's team), so an arena match is an arena for every opposing team.
- The core of a team takes exactly one seat of a role it plays (`core_seat`); only the team's *other* seats follow `teammates`. So the owner always collects in its own team, also with `teammates: mixed`.
- Seats with a role the core does not play get the latest weights of a uniformly drawn agent that plays it, owner included (spec block 7, item 4); `validate_matchmaking` guarantees such an agent exists.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_matchmaker.py -q`
Expected: 14 passed.

- [ ] **Step 5: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/coordinator/__init__.py src/colosseum/sp2/coordinator/matchmaker.py tests/unit/test_sp2_matchmaker.py
git commit -m "feat: lineup matchmaker for layouts, teams and roles (SP2 T5.1)"
git push origin sp2-game-model
```

---

### Task T5.2: Ratings per layout

Spec block 7, "Рейтинги" with the plan's rating-entity rule (overview, "Deviations"): ELO and win rates keyed by the base `agent_id`; latest vs own checkpoint feeds `wr_vs_past`; same-agent latest/latest and checkpoint/checkpoint pairs carry no signal and leave the divisor. Each pair of teams is one comparison by rank whose weight `1/(T-1)` is split equally between the counted member pairs. One-team layouts feed `ScoreTracker` and `CrossPlayTable`.

**Files:**
- Create: `src/colosseum/sp2/coordinator/ratings.py`
- Test: `tests/unit/test_sp2_ratings.py`

**Interfaces:**
- Consumes: `MatchResult`, `SeatResult`, `TeamResult`, `LATEST_NETWORK_ID` (T3.1); `pairwise_rank_score` (`colosseum.sp2.core.outcomes`, T1.4); `GameSpec.outcome_kind(layout)` (T1.4).
- Produces (contract, plus additions marked *):
  - `MemberPair(kind, a, b, score_a, weight, *role_a="", *role_b="")` and `member_pairs(result) -> list[MemberPair]`.
  - `EloRating` (SP1 methods `get`, `register`, `expected`, `update_pairs`, `all_ratings`, plus `update_weighted(pairs)`), `WinRateTracker` (`record_pair(a, b, score, weight=1.0)`, `games`, `get_win_rate`, `get_overall_win_rate`, `get_win_rate_matrix`, `get_games_matrix`), `PastWinRate` (`record(agent, score, weight=1.0)`, `get`, `games`).
  - `ScoreTracker.update(agent_id, score)`, `.summary() -> {agent: {"n", "mean", "ema", "ci_low", "ci_high"}}`; `CrossPlayTable.update(composition, score)`, `.summary() -> {"a+b": {"n", "mean"}}`; *`composition_key(agent_ids) -> str` (also used by T6.1).
  - `RatingBook(spec, agent_ids, k_factor=32.0, initial_rating=1200.0, past_window=500)` with `update(result)`, `win_rate(layout, a, b)`, `elo(layout, agent_id)`, `snapshot()`. Snapshot per layout: `{"outcome_kind"*, "elo", "win_rates", "games", "wr_vs_past", "past_games", "scores", "cross_play", "role_win_rates"*}` (`role_win_rates[role][a][b]`: score rate of `a` playing `role` against `b`, spec "Асимметрия: win-rate в разрезе ролей").

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp2_ratings.py`:

```python
"""Per-layout ratings: team-pair weights, ELO, win rates, past, scores, cross-play (T5.2)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from colosseum.sp2.coordinator.ratings import (
    CrossPlayTable,
    EloRating,
    MemberPair,
    PastWinRate,
    RatingBook,
    ScoreTracker,
    WinRateTracker,
    member_pairs,
)
from colosseum.sp2.core.types import MatchResult, SeatResult, TeamResult
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)


def seat(i, team, agent, network="latest", role="player", reward=0.0) -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=reward)


def result(layout, seats, ranks, scores=None, kind=None, length=5) -> MatchResult:
    kind = kind or ("score" if len(ranks) == 1 else "wdl" if len(ranks) == 2 else "rank")
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def test_one_vs_one_is_one_pair_of_weight_one():
    pairs = member_pairs(result("2p", [seat(0, 0, "a"), seat(1, 1, "b")], {0: 1, 1: 2}))
    assert pairs == [MemberPair("cross", "a", "b", 1.0, 1.0, "player", "player")]


def test_two_v_two_splits_the_team_pair_weight_between_member_pairs():
    r = result("2v2", [seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "b"), seat(3, 1, "b")], {0: 2, 1: 1})
    pairs = member_pairs(r)
    assert len(pairs) == 4 and all(p.score_a == 0.0 and p.weight == pytest.approx(0.25) for p in pairs)
    assert sum(p.weight for p in pairs) == pytest.approx(1.0)  # the same total weight as a 1v1


def test_ffa_team_pairs_get_one_over_t_minus_one():
    r = result("4p", [seat(i, i, agent) for i, agent in enumerate("abcd")], {0: 1, 1: 2, 2: 3, 3: 4})
    pairs = member_pairs(r)
    assert len(pairs) == 6 and all(p.weight == pytest.approx(1 / 3) for p in pairs)
    assert {(p.a, p.b): p.score_a for p in pairs}[("a", "d")] == 1.0


def test_teammates_are_not_compared_and_skipped_pairs_leave_the_divisor():
    r = result("2v1v1", [seat(0, 0, "a"), seat(1, 0, "b"), seat(2, 1, "c"), seat(3, 2, "c")],
               {0: 1, 1: 2, 2: 2})
    pairs = member_pairs(r)
    assert ("a", "b") not in {(p.a, p.b) for p in pairs}
    by_teams = [p for p in pairs if p.a in ("a", "b") and p.b == "c"]
    assert len(by_teams) == 4 and all(p.weight == pytest.approx(0.5 / 2) for p in by_teams)
    # team 1 vs team 2: c vs c, both latest -> skipped, no pair at all
    assert not [p for p in pairs if p.a == p.b == "c"]

    same = result("2v2", [seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "a"), seat(3, 1, "b")], {0: 1, 1: 2})
    pairs = member_pairs(same)  # a-vs-a latest pairs are skipped: 2 counted pairs share the weight
    assert [(p.a, p.b, p.weight) for p in pairs] == [("a", "b", 0.5), ("a", "b", 0.5)]


def test_latest_vs_own_checkpoint_is_a_past_pair_from_the_latest_side():
    r = result("2p", [seat(0, 0, "a", "ckpt_v3"), seat(1, 1, "a")], {0: 1, 1: 2})
    assert member_pairs(r) == [MemberPair("past", "a", "a", 0.0, 1.0, "player", "player")]
    two_checkpoints = result("2p", [seat(0, 0, "a", "ckpt_v1"), seat(1, 1, "a", "ckpt_v2")], {0: 1, 1: 2})
    assert member_pairs(two_checkpoints) == []
    assert member_pairs(result("solo", [seat(0, 0, "a")], {0: 1})) == []


def test_elo_weighted_updates_match_sp1_and_ignore_pair_order():
    sp1, v2 = EloRating(), EloRating()
    sp1.update_pairs([("a", "b", 1.0), ("a", "c", 0.5)], k_scale=0.5)
    v2.update_weighted([("a", "c", 0.5, 0.5), ("a", "b", 1.0, 0.5)])
    assert sp1.all_ratings == pytest.approx(v2.all_ratings)
    assert v2.get("a") == pytest.approx(1200 + 32 * 0.5 * 0.5)
    assert v2.get("b") + v2.get("c") + v2.get("a") == pytest.approx(3600)


def test_win_rates_are_weighted_and_draws_count_half():
    w = WinRateTracker()
    w.record_pair("a", "b", 1.0, weight=0.25)
    w.record_pair("a", "b", 0.5, weight=0.75)
    assert w.get_win_rate("a", "b") == pytest.approx(0.25 + 0.375)
    assert w.get_win_rate("b", "a") == pytest.approx(1 - 0.625)
    assert w.games("a", "b") == 2 and w.get_win_rate("a", "z") == 0.5
    with pytest.raises(ValueError):
        w.record_pair("a", "b", 1.5)


def test_past_win_rate_window_and_weights():
    past = PastWinRate(window=2)
    assert past.get("a") is None
    past.record("a", 1.0, 3.0)
    past.record("a", 0.0, 1.0)
    assert past.get("a") == pytest.approx(0.75) and past.games("a") == 2
    past.record("a", 0.0, 1.0)  # the oldest entry falls out of the window
    assert past.get("a") == 0.0


def test_score_tracker_mean_ema_and_ci():
    tracker = ScoreTracker(ema_alpha=0.5)
    for score in (1.0, 3.0):
        tracker.update("a", score)
    summary = tracker.summary()["a"]
    assert summary["n"] == 2 and summary["mean"] == 2.0 and summary["ema"] == 2.0
    half = 1.959963984540054 * np.std([1.0, 3.0], ddof=1) / np.sqrt(2)
    assert summary["ci_low"] == pytest.approx(2.0 - half) and summary["ci_high"] == pytest.approx(2.0 + half)
    tracker.update("b", 5.0)
    assert tracker.summary()["b"]["ci_low"] == tracker.summary()["b"]["ci_high"] == 5.0


def test_cross_play_keys_are_sorted_multisets():
    table = CrossPlayTable()
    table.update(["b", "a"], 2.0)
    table.update(["a", "b"], 4.0)
    table.update(["a", "a"], 1.0)
    assert table.summary() == {"a+a": {"n": 1, "mean": 1.0}, "a+b": {"n": 2, "mean": 3.0}}


def test_rating_book_keeps_layouts_apart_and_routes_score_layouts():
    spec = GameSpec(
        roles={"player": RoleSpec(OBS, ACT)},
        layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                 "4p": tuple(SeatSpec("player", i) for i in range(4)),
                 "coop2": (SeatSpec("player", 0), SeatSpec("player", 0))},
    )
    book = RatingBook(spec, ["a", "b"])
    book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "b")], {0: 1, 1: 2}))
    book.update(result("2p", [seat(0, 0, "a"), seat(1, 1, "a", "ckpt_v1")], {0: 1, 1: 1}))
    book.update(result("coop2", [seat(0, 0, "a"), seat(1, 0, "b")], {0: 1}, scores={0: 6.0}))
    snap = book.snapshot()
    assert set(snap) == {"2p", "4p", "coop2"}
    assert set(snap["2p"]) == {"outcome_kind", "elo", "win_rates", "games", "wr_vs_past", "past_games",
                               "scores", "cross_play", "role_win_rates"}
    assert snap["2p"]["elo"]["a"] > 1200 > snap["2p"]["elo"]["b"]
    assert snap["4p"]["elo"] == {"a": 1200.0, "b": 1200.0} and snap["4p"]["games"]["a"]["b"] == 0
    assert snap["2p"]["wr_vs_past"] == {"a": 0.5, "b": None} and snap["2p"]["past_games"]["a"] == 1
    assert snap["coop2"]["scores"]["a"]["mean"] == 6.0 and snap["coop2"]["scores"]["b"]["n"] == 1
    assert snap["coop2"]["cross_play"] == {"a+b": {"n": 1, "mean": 6.0}}
    assert book.win_rate("2p", "a", "b") == 1.0 and book.win_rate("4p", "a", "b") == 0.5
    assert book.win_rate("unknown", "a", "b") == 0.5
    assert book.elo("2p", "b") < 1200 and book.elo("unknown", "a") == 1200.0


def test_rating_book_records_win_rates_by_role():
    spec = GameSpec(
        roles={"hunter": RoleSpec(OBS, ACT), "prey": RoleSpec(OBS, gymnasium.spaces.Discrete(4))},
        layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )
    book = RatingBook(spec, ["h", "p"])
    book.update(result("1v2", [seat(0, 0, "h", role="hunter"), seat(1, 1, "p", role="prey"),
                               seat(2, 1, "p", role="prey")], {0: 1, 1: 2}))
    roles = book.snapshot()["1v2"]["role_win_rates"]
    assert roles == {"hunter": {"h": {"p": 1.0}}, "prey": {"p": {"h": 0.0}}}
    assert book.snapshot()["1v2"]["games"]["h"]["p"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_ratings.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.coordinator.ratings'`.

- [ ] **Step 3: Write the implementation**

Create `src/colosseum/sp2/coordinator/ratings.py`:

```python
"""Ratings per layout (spec block 7).

Every layout has its own tables. Rating entities follow SP1's code (overview, "Deviations"):
ELO and the win-rate matrix are keyed by the base ``agent_id``, so a checkpoint of X playing Y
counts as X vs Y; "latest of X vs a checkpoint of X" feeds ``wr_vs_past``; two seats of the same
agent that are both latest (or both checkpoints) carry no signal and are skipped.

Layouts with two or more teams (``wdl`` / ``rank``): every pair of teams (A, B) is one comparison
by team rank (``pairwise_rank_score``). Its weight ``1 / (T - 1)`` (times ``k_factor`` for ELO) is
split equally between the counted member pairs ``(a in A, b in B)`` (``member_pairs``);
teammates are never compared, and skipped member pairs are not in the divisor. ELO deltas of one
match are computed from the ratings before the match. Win rates and ``wr_vs_past`` are weighted
means of the same member-pair scores (fractional: a draw counts 0.5).

One-team layouts (solo, cooperative, ``score``): ``ScoreTracker`` keeps every agent's team score
(mean, EMA, 95% CI) and ``CrossPlayTable`` the mean score per team composition.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from colosseum.sp2.core.outcomes import pairwise_rank_score
from colosseum.sp2.core.types import LATEST_NETWORK_ID, MatchResult, SeatResult
from colosseum.sp2.envs.game import GameSpec

Z_95 = 1.959963984540054
SCORE_EMA_ALPHA = 0.05


@dataclass(frozen=True)
class MemberPair:
    """One counted comparison between seats of two different teams.

    ``cross``: different agents ``a`` and ``b``. ``past``: ``a`` is an agent's latest weights and
    ``b`` (== ``a``) one of its checkpoints. ``score_a`` is from ``a``'s side.
    """

    kind: Literal["cross", "past"]
    a: str
    b: str
    score_a: float
    weight: float
    role_a: str = ""
    role_b: str = ""


def _classify(sa: SeatResult, sb: SeatResult, score: float) -> tuple | None:
    if sa.agent_id != sb.agent_id:
        return "cross", sa.agent_id, sb.agent_id, score, sa.role, sb.role
    a_latest = sa.network_id == LATEST_NETWORK_ID
    b_latest = sb.network_id == LATEST_NETWORK_ID
    if a_latest and not b_latest:
        return "past", sa.agent_id, sa.agent_id, score, sa.role, sb.role
    if b_latest and not a_latest:
        return "past", sb.agent_id, sb.agent_id, 1.0 - score, sb.role, sa.role
    return None


def member_pairs(result: MatchResult) -> list[MemberPair]:
    """Counted member pairs of ``result`` with their weights (module docstring)."""
    by_team: dict[int, list[SeatResult]] = {}
    for seat in result.seats:
        by_team.setdefault(seat.team, []).append(seat)
    ranks = {team.team: team.rank for team in result.teams}
    teams = sorted(by_team)
    if len(teams) < 2:
        return []
    share = 1.0 / (len(teams) - 1)
    pairs: list[MemberPair] = []
    for i, team_a in enumerate(teams):
        for team_b in teams[i + 1:]:
            score = pairwise_rank_score(ranks[team_a], ranks[team_b])
            counted = [c for sa in by_team[team_a] for sb in by_team[team_b]
                       if (c := _classify(sa, sb, score)) is not None]
            for kind, a, b, s, role_a, role_b in counted:
                pairs.append(MemberPair(kind, a, b, s, share / len(counted), role_a, role_b))
    return pairs


class EloRating:
    """Pairwise ELO; all deltas of one update are computed from the ratings before it."""

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0) -> None:
        self.k_factor = float(k_factor)
        self.initial_rating = float(initial_rating)
        self._ratings: dict[str, float] = {}

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial_rating)

    def register(self, agent_id: str) -> None:
        self._ratings.setdefault(agent_id, self.initial_rating)

    def expected(self, a: str, b: str) -> float:
        """Expected score of ``a`` against ``b``."""
        return 1.0 / (1.0 + math.pow(10.0, (self.get(b) - self.get(a)) / 400.0))

    def update_weighted(self, pairs: Iterable[tuple[str, str, float, float]]) -> None:
        """Apply ``(a, b, score_a, weight)`` results together; each moves by ``k_factor * weight``."""
        deltas: dict[str, float] = {}
        for a, b, score_a, weight in pairs:
            k = self.k_factor * float(weight)
            e_a = self.expected(a, b)
            deltas[a] = deltas.get(a, 0.0) + k * (score_a - e_a)
            deltas[b] = deltas.get(b, 0.0) + k * ((1.0 - score_a) - (1.0 - e_a))
        for agent_id, delta in deltas.items():
            self._ratings[agent_id] = self.get(agent_id) + delta

    def update_pairs(self, pairs: Iterable[tuple[str, str, float]], k_scale: float = 1.0) -> None:
        """SP1 form: every ``(a, b, score_a)`` with the same weight ``k_scale``."""
        self.update_weighted((a, b, s, k_scale) for a, b, s in pairs)

    @property
    def all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


class WinRateTracker:
    """Pairwise score rates as weighted means (a draw is 0.5); ``games`` counts member pairs."""

    def __init__(self) -> None:
        self._scores: dict[str, dict[str, float]] = {}
        self._weights: dict[str, dict[str, float]] = {}
        self._games: dict[str, dict[str, int]] = {}

    def record_pair(self, a: str, b: str, score: float, weight: float = 1.0) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        if not weight > 0.0:
            raise ValueError(f"weight must be > 0, got {weight}")
        for x, y, s in ((a, b, score), (b, a, 1.0 - score)):
            self._scores.setdefault(x, {}).setdefault(y, 0.0)
            self._weights.setdefault(x, {}).setdefault(y, 0.0)
            self._games.setdefault(x, {}).setdefault(y, 0)
            self._scores[x][y] += weight * s
            self._weights[x][y] += weight
            self._games[x][y] += 1

    def games(self, a: str, b: str) -> int:
        return self._games.get(a, {}).get(b, 0)

    def get_win_rate(self, a: str, b: str) -> float:
        """a's score rate against b; 0.5 if they never met."""
        weight = self._weights.get(a, {}).get(b, 0.0)
        return self._scores[a][b] / weight if weight > 0.0 else 0.5

    def get_overall_win_rate(self, agent_id: str) -> float:
        weight = sum(self._weights.get(agent_id, {}).values())
        return sum(self._scores.get(agent_id, {}).values()) / weight if weight > 0.0 else 0.5

    def get_win_rate_matrix(self, agent_ids: Sequence[str]) -> dict[str, dict[str, float]]:
        return {a: {b: self.get_win_rate(a, b) for b in agent_ids if b != a} for a in agent_ids}

    def get_games_matrix(self, agent_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        return {a: {b: self.games(a, b) for b in agent_ids if b != a} for a in agent_ids}


class PastWinRate:
    """Weighted score rate of an agent's latest weights against its own checkpoints, over the
    last ``window`` member pairs (the self-play progress signal)."""

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._scores: dict[str, deque[tuple[float, float]]] = {}

    def record(self, agent_id: str, score: float, weight: float = 1.0) -> None:
        self._scores.setdefault(agent_id, deque(maxlen=self._window)).append((float(score), float(weight)))

    def get(self, agent_id: str) -> float | None:
        entries = self._scores.get(agent_id)
        if not entries:
            return None
        total = sum(w for _, w in entries)
        return sum(s * w for s, w in entries) / total if total > 0.0 else None

    def games(self, agent_id: str) -> int:
        return len(self._scores.get(agent_id, ()))


class ScoreTracker:
    """Per agent: count, mean, EMA and a 95% normal CI of the team score (one-team layouts)."""

    def __init__(self, ema_alpha: float = SCORE_EMA_ALPHA) -> None:
        self._alpha = float(ema_alpha)
        self._stats: dict[str, tuple[int, float, float, float]] = {}  # n, mean, M2, ema

    def update(self, agent_id: str, score: float) -> None:
        score = float(score)
        n, mean, m2, ema = self._stats.get(agent_id, (0, 0.0, 0.0, score))
        n += 1
        delta = score - mean
        mean += delta / n
        m2 += delta * (score - mean)
        ema = score if n == 1 else (1.0 - self._alpha) * ema + self._alpha * score
        self._stats[agent_id] = (n, mean, m2, ema)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for agent_id, (n, mean, m2, ema) in self._stats.items():
            half = Z_95 * math.sqrt(m2 / (n - 1)) / math.sqrt(n) if n >= 2 else 0.0
            out[agent_id] = {"n": n, "mean": mean, "ema": ema, "ci_low": mean - half, "ci_high": mean + half}
        return out


def composition_key(agent_ids: Iterable[str]) -> str:
    """Team composition as a multiset: sorted agent ids joined by ``+`` (``"a+a+b"``)."""
    return "+".join(sorted(agent_ids))


class CrossPlayTable:
    """Mean team score per team composition (one-team layouts)."""

    def __init__(self) -> None:
        self._cells: dict[str, list[float]] = {}  # key -> [n, sum]

    def update(self, composition: Sequence[str], score: float) -> None:
        cell = self._cells.setdefault(composition_key(composition), [0, 0.0])
        cell[0] += 1
        cell[1] += float(score)

    def summary(self) -> dict[str, dict[str, float]]:
        return {key: {"n": n, "mean": total / n} for key, (n, total) in sorted(self._cells.items())}


class _LayoutRatings:
    def __init__(self, outcome_kind: str, k_factor: float, initial_rating: float, past_window: int) -> None:
        self.outcome_kind = outcome_kind
        self.elo = EloRating(k_factor, initial_rating)
        self.win_rates = WinRateTracker()
        self.past = PastWinRate(past_window)
        self.scores = ScoreTracker()
        self.cross_play = CrossPlayTable()
        self.role_scores: dict[str, dict[str, dict[str, list[float]]]] = {}  # role -> a -> b -> [sum, weight]

    def record_role(self, pair: MemberPair) -> None:
        for role, x, y, s in ((pair.role_a, pair.a, pair.b, pair.score_a),
                              (pair.role_b, pair.b, pair.a, 1.0 - pair.score_a)):
            cell = self.role_scores.setdefault(role, {}).setdefault(x, {}).setdefault(y, [0.0, 0.0])
            cell[0] += pair.weight * s
            cell[1] += pair.weight

    def role_win_rates(self) -> dict[str, dict[str, dict[str, float]]]:
        return {role: {a: {b: s / w for b, (s, w) in row.items()} for a, row in table.items()}
                for role, table in sorted(self.role_scores.items())}


class RatingBook:
    """Rating tables of every layout of a game (module docstring)."""

    def __init__(self, spec: GameSpec, agent_ids: Sequence[str], k_factor: float = 32.0,
                 initial_rating: float = 1200.0, past_window: int = 500) -> None:
        self._agent_ids = list(agent_ids)
        self._args = (k_factor, initial_rating, past_window)
        self._books = {name: _LayoutRatings(spec.outcome_kind(name), *self._args) for name in spec.layouts}

    def _book(self, layout: str, outcome_kind: str) -> _LayoutRatings:
        if layout not in self._books:
            self._books[layout] = _LayoutRatings(outcome_kind, *self._args)
        return self._books[layout]

    def update(self, result: MatchResult) -> None:
        book = self._book(result.layout, result.outcome_kind)
        if result.outcome_kind == "score":
            score = float(result.teams[0].score)
            for agent_id in sorted({seat.agent_id for seat in result.seats}):
                book.scores.update(agent_id, score)
            if len(result.seats) >= 2:
                book.cross_play.update([seat.agent_id for seat in result.seats], score)
            return
        pairs = member_pairs(result)
        cross = [p for p in pairs if p.kind == "cross"]
        book.elo.update_weighted((p.a, p.b, p.score_a, p.weight) for p in cross)
        for p in cross:
            book.win_rates.record_pair(p.a, p.b, p.score_a, p.weight)
            book.record_role(p)
        for p in pairs:
            if p.kind == "past":
                book.past.record(p.a, p.score_a, p.weight)

    def win_rate(self, layout: str, a: str, b: str) -> float:
        """a's score rate against b in ``layout``; 0.5 for pairs that never met (SP1 prior)."""
        book = self._books.get(layout)
        return 0.5 if book is None else book.win_rates.get_win_rate(a, b)

    def elo(self, layout: str, agent_id: str) -> float:
        book = self._books.get(layout)
        return self._args[1] if book is None else book.elo.get(agent_id)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        ids = self._agent_ids
        return {
            layout: {
                "outcome_kind": book.outcome_kind,
                "elo": {a: book.elo.get(a) for a in ids},
                "win_rates": book.win_rates.get_win_rate_matrix(ids),
                "games": book.win_rates.get_games_matrix(ids),
                "wr_vs_past": {a: book.past.get(a) for a in ids},
                "past_games": {a: book.past.games(a) for a in ids},
                "scores": book.scores.summary(),
                "cross_play": book.cross_play.summary(),
                "role_win_rates": book.role_win_rates(),
            }
            for layout, book in self._books.items()
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_ratings.py -q`
Expected: 12 passed.

- [ ] **Step 5: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/coordinator/ratings.py tests/unit/test_sp2_ratings.py
git commit -m "feat: per-layout ratings with team-pair ELO, scores and cross-play (SP2 T5.2)"
git push origin sp2-game-model
```

---

### Task T5.3: Coordinator, checkpoint manager with roles, metrics per layout and role

The coordinator keeps SP1's owner rotation (env `g` belongs to `agents[(g + round) % n]`) and builds one `Lineup` per env with `LineupMatchmaker`. Checkpoints record the agent's roles and their role signature in `meta.json` (spec block 9); resume compares the signature. The metrics aggregator and hub keep every SP1 record and add the breakdown by layout and role (spec block 5, "Метрики эпизодов"): W/D/L by opponent type now come from team ranks, and teammates are never opponents.

**Files:**
- Create: `src/colosseum/sp2/coordinator/checkpoint_manager.py` (copy of `src/colosseum/coordinator/checkpoint_manager.py` + edits)
- Create: `src/colosseum/sp2/coordinator/coordinator.py`
- Create: `src/colosseum/sp2/metrics/__init__.py` (empty), `src/colosseum/sp2/metrics/jsonl.py` (copy + edits), `src/colosseum/sp2/metrics/aggregator.py`, `src/colosseum/sp2/metrics/hub.py`
- Modify: `tests/game_helpers.py` (append the T5.3 block)
- Modify: `pyproject.toml` (`[tool.ruff.lint.isort] known-first-party`)
- Test: `tests/unit/test_sp2_checkpoints.py` (copy of `tests/unit/test_checkpoint_store.py` + edits), `tests/unit/test_sp2_coordinator.py`, `tests/unit/test_sp2_metrics.py`

**Interfaces:**
- Consumes: `LineupMatchmaker`, `validate_matchmaking` (T5.1); `RatingBook` (T5.2); `role_signature`, `agent_role_spec`, `resolve_agent_roles` (`colosseum.sp2.core.roles`, T2.4); `env_spec`, `build_model` (`colosseum.sp2.core.registry`, T2.4); `ColosseumConfig`, `deep_merge`, `check_agent_id`, `check_path_component`, `config_hash` (T1.7); `state_dict_from_numpy`, `state_dict_to_numpy`, `MatchResult` (T3.1); `make_test_model(role, core, hidden)` and the toy games of `tests/game_helpers.py` (T1.4, T2.3); `make_checkpoint_payload`, `send_checkpoint`, `apply_resume_state` (`colosseum.sp2.learner.learner`, T4.4); `colosseum.coordinator.agent_pool.AgentPool` and `colosseum.metrics.console.ConsoleReporter` (unchanged SP1 modules).
- Produces:
  - `Coordinator(config, spec, agent_roles, checkpoint_dir)` with `agent_pool`, `checkpoint_manager`, `ratings`, `refresh_round`, `next_round()`, `generate_lineups(num_envs, env_offset)`, `report_match_result(result)`, `match_results`, `ratings_snapshot()`, `save_checkpoint_payload(payload, meta_extra=None)` (contract), plus *`matchmaker`, *`spec`, *`agent_roles`, *`role_signature(agent_id)`.
  - `CheckpointManager`, `load_checkpoint_dir`, `resolve_resume`, `check_model_state`, `read_weights_file`, `classify_resume_source` with SP1 behavior; `load_checkpoint_dir` and `resolve_resume` results gain `"roles"` and `"role_signature"`; *`resolve_resume(resume_from, agent_id, expected_signature=None)`; *`check_role_signature(state, expected, context)`.
  - `EpisodeAggregator`, `SystemStats`, `opponent_type`, `OPPONENT_TYPES`, `WORKER_STATS_INTERVAL_SEC` (`colosseum.sp2.metrics.aggregator`); `MetricsHub`, `flatten`, *`wr_vs_past_over_layouts`, *`wr_arena_over_layouts` (`colosseum.sp2.metrics.hub`); `MetricsWriter`, `write_json_atomic`, `METRIC_KINDS`, `GLOBAL_KINDS`, `REQUIRED_KEYS` (`colosseum.sp2.metrics.jsonl`, a copy: see Contract notes).
  - Record shapes: `episodes` records add `by_layout: {layout: {role: {"episodes", "return_mean", "team_score_mean", "eliminated_frac"}}}`; `ratings` records and `ratings.json` are `{"env_steps", "layouts": RatingBook.snapshot()}`; WandB rows `ratings/<layout>/...` and `episodes/<agent>/by_layout/<layout>/<role>/...`.
  - `tests/game_helpers.py`: `GameTestModel`, `TEST_GAME_AGENTS`, `make_test_config(game, **sections)` (`game` is a `TOY_GAMES` name), `write_test_config(path, game, **sections)`, `make_coordinator(config, checkpoint_dir)`, `agent_role_of(config, agent_id) -> (roles, RoleSpec)`.

- [ ] **Step 1: Add the shared test helpers**

Append to `tests/game_helpers.py` (keep everything Parts A and B put there). The block uses `TOY_GAMES` and `make_test_model` (Part A, same module), `RoleSpec` (`from colosseum.sp2.envs.game import RoleSpec`) and `PolicyModel` (`from colosseum.sp2.networks.model import PolicyModel`); add the two imports to the import block at the top of the file if they are not there yet.

```python
# ---------------------------------------------------------------------------
# Part C (T5.3): a config-driven test model and tiny configs for the toy games
# ---------------------------------------------------------------------------


class GameTestModel(PolicyModel):
    """``make_test_model`` behind ``networks.model_class``: ``build_model`` injects the role's spaces."""

    def __init__(self, observation_space, action_space, global_state_space=None, core: str = "none",
                 hidden: int = 16) -> None:
        super().__init__()
        role = RoleSpec(observation_space, action_space, global_state_space)
        self.inner = make_test_model(role, core=core, hidden=hidden)

    def initial_state(self, batch_size, device="cpu"):
        return self.inner.initial_state(batch_size, device)

    def step(self, obs, state, action_mask=None):
        return self.inner.step(obs, state, action_mask)

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        return self.inner.unroll(obs, state0, reset_after, action_mask, global_state=global_state,
                                 with_value=with_value)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)

    def update_normalizers(self, obs, global_state=None):
        self.inner.update_normalizers(obs, global_state)

    @property
    def is_stateful(self) -> bool:
        return self.inner.is_stateful


# Agents sections of the toy games (``TOY_GAMES`` names) that need more than the default agent.
TEST_GAME_AGENTS: dict[str, dict] = {
    "asymmetric": {"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]}},
}


def make_test_config(game: str, **sections):
    """A tiny run ``ColosseumConfig`` for a toy game of this module (a ``TOY_GAMES`` name, default kwargs).

    Each keyword is a top-level config section deep-merged onto the defaults below, except
    ``agents``, which replaces the game's agents section. ``env.env_class`` names this module,
    so spawned children need ``tests/`` on their path (the test process has it; CLI children get
    it through ``cli_runner.child_env``).
    """
    from colosseum.sp2.core.config import ColosseumConfig, deep_merge

    agents = TEST_GAME_AGENTS.get(game)
    data = {
        "env": {"env_class": f"game_helpers.{TOY_GAMES[game].__name__}", "kwargs": {}},
        "networks": {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 16}},
        "algorithm": {"learning_rate": 1.0e-3, "lr_schedule": "constant"},
        "rollout": {"num_workers": 1, "envs_per_worker": 4, "chunk_length": 8,
                    "weight_sync_interval_sec": 0.5, "match_refresh_interval_sec": 1.0},
        "learner": {"device": "cpu", "batch_chunks": 2, "queue_size": 16},
        "training": {"total_timesteps": 2000, "seed": 0},
        "checkpoint": {"interval": 20, "pool_size": 5},
        "metrics": {"use_wandb": False, "log_interval": 1, "console_interval_sec": 1.0},
    }
    if agents is not None:
        data["agents"] = {aid: dict(override) for aid, override in agents.items()}
    for section, values in sections.items():
        if section == "agents" or not isinstance(data.get(section), dict):
            data[section] = values
        else:
            data[section] = deep_merge(data[section], values)
    return ColosseumConfig.model_validate(data)


def write_test_config(path, game: str, **sections):
    """``make_test_config`` written as YAML to ``path`` (for CLI runs); returns the path."""
    from pathlib import Path

    import yaml

    path = Path(path)
    config = make_test_config(game, **sections)
    path.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
    return path


def make_coordinator(config, checkpoint_dir):
    """``Coordinator`` for ``config`` with the env's spec and the resolved agent roles."""
    from colosseum.sp2.coordinator.coordinator import Coordinator
    from colosseum.sp2.core.registry import env_spec
    from colosseum.sp2.core.roles import resolve_agent_roles

    spec = env_spec(config)
    return Coordinator(config, spec, resolve_agent_roles(config, spec), checkpoint_dir)


def agent_role_of(config, agent_id: str):
    """``(roles, RoleSpec)`` of a trainable agent of ``config``."""
    from colosseum.sp2.core.registry import env_spec
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    spec = env_spec(config)
    roles = resolve_agent_roles(config, spec)[agent_id]
    return roles, agent_role_spec(spec, roles)
```

In `pyproject.toml`, make the isort section know the SP2 support modules (Parts A/B added `game_helpers` and `game_harness`; add `game_learning_envs`, used from T6.3 on; keep the list sorted):

```toml
[tool.ruff.lint.isort]
known-first-party = ["cli_runner", "colosseum", "dataflow_helpers", "examples", "game_harness", "game_helpers", "game_learning_envs", "harness", "helpers", "learning_envs", "ttt_eval"]
```

- [ ] **Step 2: Write the failing tests**

(a) Create `tests/unit/test_sp2_coordinator.py`:

```python
"""Coordinator v2: owner rotation, lineups, per-layout ratings, roles in checkpoint meta (T5.3)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator.coordinator import Coordinator
from colosseum.sp2.core.registry import env_spec
from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.sp2.core.types import LATEST_NETWORK_ID, MatchResult, SeatResult, TeamResult
from game_helpers import make_coordinator, make_test_config


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def owners(coord: Coordinator, envs: int, offset: int = 0) -> list[str]:
    """The agent whose latest weights collect in each lineup (the owner always does)."""
    out = []
    for lineup in coord.generate_lineups(envs, env_offset=offset):
        collecting = {s.agent_id for s in lineup.seats if s.collect}
        assert len(collecting) == 1, lineup  # self-play: only the owner's agent collects
        out.append(collecting.pop())
    return out


def test_owner_rotates_with_global_env_index_and_round(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}, "c": {}},
                           matchmaking={"mode": "self_play", "latest_prob": 1.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert owners(coord, 4) == ["a", "b", "c", "a"]
    assert owners(coord, 2, offset=1) == ["b", "c"]
    coord.next_round()
    assert coord.refresh_round == 1
    assert owners(coord, 3) == ["b", "c", "a"]


def test_asymmetric_agents_both_own_envs_and_fill_each_others_teams(tmp_path):
    cfg = make_test_config("asymmetric", matchmaking={"mode": "self_play"})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    lineups = coord.generate_lineups(8, env_offset=0)
    for lineup in lineups:
        roles = coord.spec.layouts[lineup.layout]
        for seat, seat_spec in zip(lineup.seats, roles, strict=True):
            assert seat_spec.role in coord.agent_roles[seat.agent_id]
    assert {s.agent_id for lu in lineups for s in lu.seats} == {"hunter", "prey"}


def test_self_play_draws_saved_checkpoints_as_opponents(tmp_path):
    cfg = make_test_config("turns", matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert all(s.network_id == LATEST_NETWORK_ID for lu in coord.generate_lineups(4, 0) for s in lu.seats)
    coord.checkpoint_manager.save("agent_0", 7, sd(7))
    networks = Counter(s.network_id for lu in coord.generate_lineups(20, 0) for s in lu.seats)
    assert networks["ckpt_v7"] == 20 and networks[LATEST_NETWORK_ID] == 20  # one owner seat per 2-seat match


def test_report_match_result_updates_the_layout_ratings(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    layout = next(iter(coord.spec.layouts))
    role = coord.spec.role_of(layout, 0)
    result = MatchResult(
        match_id="m", layout=layout, outcome_kind="wdl",
        seats=[SeatResult(0, role, 0, "a", "latest", 1.0), SeatResult(1, role, 1, "b", "latest", -1.0)],
        teams=[TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], episode_length=4,
    )
    coord.report_match_result(result)
    snap = coord.ratings_snapshot()
    assert snap[layout]["elo"]["a"] > snap[layout]["elo"]["b"]
    assert coord.ratings.win_rate(layout, "a", "b") == 1.0
    assert coord.match_results == [result]


def test_checkpoint_meta_records_roles_and_signature(tmp_path):
    cfg = make_test_config("asymmetric")
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    payload = {"agent_id": "hunter", "policy_version": 3, "final": True, "model_state": sd(3),
               "trainer_state_bytes": b"opt"}
    assert coord.save_checkpoint_payload(payload, meta_extra={"env_steps": 12}) == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "hunter" / "ckpt_v3" / "meta.json").read_text())
    spec = env_spec(cfg)
    expected = role_signature(agent_role_spec(spec, resolve_agent_roles(cfg, spec)["hunter"]))
    assert meta["roles"] == ["hunter"] and meta["role_signature"] == expected == coord.role_signature("hunter")
    assert meta["final"] is True and meta["env_steps"] == 12
    assert coord.role_signature("prey") != coord.role_signature("hunter")
    assert (tmp_path / "ckpt" / "hunter" / "ckpt_v3" / "trainer_state.pt").read_bytes() == b"opt"


def test_trainer_state_is_dropped_without_save_optimizer(tmp_path):
    cfg = make_test_config("solo", checkpoint={"save_optimizer": False})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": 1, "model_state": sd(1),
                                   "trainer_state_bytes": b"opt"})
    assert not (tmp_path / "ckpt" / "agent_0" / "ckpt_v1" / "trainer_state.pt").exists()


def test_missing_roles_for_an_agent_is_a_config_error(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}})
    spec = env_spec(cfg)
    with pytest.raises(ConfigError, match="no roles"):
        Coordinator(cfg, spec, {"a": list(spec.roles)}, tmp_path / "ckpt")
```

(b) Create `tests/unit/test_sp2_metrics.py` (SP1's `test_metrics_jsonl.py` guarantees on `MatchResult` v2; the worker-stats test stays with the worker, Part B, and the launcher test moves to T7.1):

```python
"""metrics.jsonl schema, episode aggregation by layout/role, system stats, hub cadence (T5.3)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from colosseum.metrics.console import ConsoleReporter
from colosseum.sp2.core.types import MatchResult, SeatResult, TeamResult
from colosseum.sp2.metrics.aggregator import EpisodeAggregator, SystemStats, opponent_type
from colosseum.sp2.metrics.hub import MetricsHub, flatten, wr_arena_over_layouts, wr_vs_past_over_layouts
from colosseum.sp2.metrics.jsonl import METRIC_KINDS, REQUIRED_KEYS, MetricsWriter, write_json_atomic


def seat(i, team, agent, network="latest", reward=0.0, role="player", eliminated=None) -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=reward,
                      eliminated_step=eliminated)


def result(layout, seats, ranks, scores=None, length=5) -> MatchResult:
    kind = "score" if len(ranks) == 1 else "wdl" if len(ranks) == 2 else "rank"
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def duel(a, b, rank_a, rank_b, net_b="latest", ra=0.0, rb=0.0, length=5):
    return result("2p", [seat(0, 0, a, reward=ra), seat(1, 1, b, net_b, reward=rb)], {0: rank_a, 1: rank_b},
                  length=length)


RATINGS = {
    "2p": {"elo": {"a": 1216.0, "b": 1184.0}, "win_rates": {"a": {"b": 1.0}, "b": {"a": 0.0}},
           "games": {"a": {"b": 1}, "b": {"a": 1}}, "wr_vs_past": {"a": None, "b": 0.75},
           "past_games": {"a": 0, "b": 4}, "scores": {}, "cross_play": {}, "role_win_rates": {}},
    "4p": {"elo": {"a": 1200.0, "b": 1200.0}, "win_rates": {"a": {"b": 0.0}, "b": {"a": 1.0}},
           "games": {"a": {"b": 3}, "b": {"a": 3}}, "wr_vs_past": {"a": 0.5, "b": 0.25},
           "past_games": {"a": 2, "b": 4}, "scores": {}, "cross_play": {}, "role_win_rates": {}},
}


def test_writer_one_json_line_per_record_with_numpy_and_nan(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write("train", agent="a", train_step=np.int64(3), loss=np.float32(0.5), ev=float("nan"))
    with pytest.raises(ValueError):
        writer.write("bogus")
    writer.close()
    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert record["kind"] == "train" and record["train_step"] == 3 and record["ev"] is None


def test_write_json_atomic(tmp_path):
    write_json_atomic(tmp_path / "ratings.json", {"layouts": {"2p": {"elo": {"a": 1200.0}}}})
    assert json.loads((tmp_path / "ratings.json").read_text()) == {"layouts": {"2p": {"elo": {"a": 1200.0}}}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ratings.json"]


def test_opponent_type_ignores_teammates():
    r = duel("a", "a", 1, 2, net_b="ckpt_v5")
    assert opponent_type(r, r.seats[0]) == "past"
    r = duel("a", "b", 1, 2)
    assert opponent_type(r, r.seats[0]) == "arena"
    r = duel("a", "a", 1, 1)
    assert opponent_type(r, r.seats[1]) == "latest"
    coop = result("coop2", [seat(0, 0, "a"), seat(1, 0, "b")], {0: 1})
    assert opponent_type(coop, coop.seats[0]) is None  # a teammate of another agent is no opponent
    team = result("2v2", [seat(0, 0, "a"), seat(1, 0, "b"), seat(2, 1, "a"), seat(3, 1, "a")], {0: 1, 1: 2})
    assert opponent_type(team, team.seats[0]) == "latest" and opponent_type(team, team.seats[2]) == "arena"


def test_episode_aggregator_wdl_from_team_ranks_and_layout_breakdown():
    agg = EpisodeAggregator()
    agg.add(duel("a", "a", 1, 2, net_b="ckpt_v5", ra=1.0, rb=-1.0, length=7))
    agg.add(duel("a", "a", 1, 1, length=9))
    agg.add(duel("b", "a", 2, 1, ra=-1.0, rb=1.0, length=5))
    agg.add(result("4p", [seat(i, i, "a", eliminated=(3 if i == 3 else None)) for i in range(4)],
                   {0: 1, 1: 2, 2: 2, 3: 4}, length=10))
    agg.add(result("solo", [seat(0, 0, "s", reward=3.0)], {0: 1}, scores={0: 3.0}, length=10))
    out = agg.flush()
    assert out["a"]["wdl"] == {"latest": [1, 2, 3], "past": [1, 0, 0], "arena": [1, 0, 0]}
    assert out["a"]["episodes"] == 8 and out["a"]["seat_counts"] == [3, 3, 1, 1]
    assert out["b"]["wdl"]["arena"] == [0, 0, 1]
    assert out["s"]["return_mean"] == 3.0 and out["s"]["wdl"]["latest"] == [0, 0, 0]
    four = out["a"]["by_layout"]["4p"]["player"]
    assert four["episodes"] == 4 and four["eliminated_frac"] == 0.25
    assert out["s"]["by_layout"] == {"solo": {"player": {"episodes": 1, "return_mean": 3.0,
                                                         "team_score_mean": 3.0, "eliminated_frac": 0.0}}}
    assert set(out["a"]["by_layout"]) == {"2p", "4p"}
    assert agg.flush() == {}


def test_episode_aggregator_splits_roles():
    agg = EpisodeAggregator()
    agg.add(result("1v2", [seat(0, 0, "h", role="hunter", reward=2.0), seat(1, 1, "p", role="prey", reward=-1.0),
                           seat(2, 1, "p", role="prey", reward=-3.0)], {0: 1, 1: 2}))
    out = agg.flush()
    assert out["h"]["by_layout"]["1v2"]["hunter"]["return_mean"] == 2.0
    assert out["p"]["by_layout"]["1v2"]["prey"]["return_mean"] == -2.0
    assert out["p"]["by_layout"]["1v2"]["prey"]["episodes"] == 2


def test_system_stats_rates_and_resume_baselines():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0], initial_env_steps=1000, initial_train_steps={"a": 10})
    stats.on_train_step("a", 14)
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 2})
    clock[0] = 2.0
    snap = stats.snapshot(2000, {"a": 3})
    assert snap["env_steps_per_sec"] == 500.0 and snap["train_steps_per_sec"] == {"a": 2.0}
    assert snap["parked_buffers"] == 2 and snap["workers_reporting"] == 1
    clock[0] = 20.0
    assert stats.snapshot(2000, {})["workers_reporting"] == 0  # silent workers are forgotten


def test_console_line_and_layout_aggregates():
    line = ConsoleReporter(10_000).format_line(
        "a", train_step=5, env_steps=2500, fps=1234.5, loss=0.123, entropy=None,
        return_mean=0.5, wr_vs_past=0.61, wr_arena=None)
    assert line.startswith("[a] step 5 |") and "wr_vs_past 0.61" in line
    assert wr_vs_past_over_layouts(RATINGS, "b") == pytest.approx((0.75 * 4 + 0.25 * 4) / 8)
    assert wr_vs_past_over_layouts(RATINGS, "a") == 0.5  # 2p has no past games for a
    assert wr_arena_over_layouts(RATINGS, "a") == pytest.approx((1.0 * 1 + 0.0 * 3) / 4)
    assert wr_arena_over_layouts({}, "a") is None


def test_flatten():
    assert flatten("r", {"2p": {"elo": {"a": 1.0}}, "x": None, "l": [1, 2], "ok": True}) == {
        "r/2p/elo/a": 1.0, "r/l/0": 1.0, "r/l/1": 2.0}


def test_hub_cadence_schema_and_ratings_file(tmp_path, caplog):
    clock = [0.0]
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["a", "b"], total_timesteps=1000, log_interval=2, console_interval_sec=10.0,
                     clock=lambda: clock[0])
    for step in range(1, 6):
        hub.on_train_metrics({"agent_id": "a", "train_step": step, "total_loss": 0.1 * step, "note": "x"})
    hub.on_match_result(duel("a", "b", 1, 2, ra=1.0, rb=-1.0, length=7))
    hub.on_worker_stats({"kind": "worker_stats", "worker_id": 0, "env_steps": 100, "parked_buffers": 1})
    assert not hub.maybe_tick(env_steps=100, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    clock[0] = 11.0
    with caplog.at_level(logging.INFO):
        assert hub.maybe_tick(env_steps=300, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    assert "[a] step 5 |" in caplog.text and "[b] step 0 |" in caplog.text and "wr_arena 0.25" in caplog.text
    clock[0] = 12.0
    hub.close(env_steps=400, ratings=RATINGS, queue_depths={"a": 0, "b": 0})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_step"] for r in records if r["kind"] == "train"] == [1, 3, 5]
    assert {r["kind"] for r in records} == set(METRIC_KINDS)
    for record in records:
        assert not (REQUIRED_KEYS[record["kind"]] - set(record)), record["kind"]
    ratings_records = [r for r in records if r["kind"] == "ratings"]
    assert ratings_records[-1]["layouts"]["2p"]["elo"] == {"a": 1216.0, "b": 1184.0}
    episodes = [r for r in records if r["kind"] == "episodes"]
    assert {r["agent"] for r in episodes} == {"a", "b"} and "2p" in episodes[0]["by_layout"]
    ratings = json.loads((tmp_path / "ratings.json").read_text())
    assert ratings["env_steps"] == 400 and ratings["layouts"]["4p"]["wr_vs_past"] == {"a": 0.5, "b": 0.25}


def test_hub_forwards_layout_namespaces_to_wandb(tmp_path):
    calls = []

    class FakeWandB:
        def log_train(self, agent_id, metrics, train_step):
            calls.append(("train", agent_id, train_step, dict(metrics)))

        def log_global(self, metrics, env_steps):
            calls.append(("global", env_steps, dict(metrics)))

    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10, log_interval=1, console_interval_sec=0.0,
                     wandb_logger=FakeWandB())
    hub.on_train_metrics({"agent_id": "a", "train_step": 1, "total_loss": 0.5})
    hub.on_match_result(duel("a", "a", 1, 2))
    hub.maybe_tick(env_steps=5, ratings=RATINGS, queue_depths={"a": 0})
    assert calls[0] == ("train", "a", 1, {"total_loss": 0.5})
    row = calls[1][2]
    assert calls[1][0] == "global" and calls[1][1] == 5
    assert row["ratings/2p/elo/a"] == 1216.0 and row["ratings/4p/games/a/b"] == 3.0
    assert row["episodes/a/by_layout/2p/player/episodes"] == 2.0


def test_hub_keeps_numpy_scalar_metrics_and_closes_on_failure(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json",
                     agent_ids=["a"], total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    hub.on_train_metrics({"agent_id": "a", "train_step": np.int64(1), "loss": np.float32(0.5),
                          "flag": np.bool_(True), "ok": True})
    hub.close(env_steps=0, ratings=RATINGS, queue_depths={})
    train = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert train["loss"] == 0.5 and "flag" not in train and "ok" not in train

    writer = MetricsWriter(tmp_path / "m2.jsonl")
    failing = MetricsHub(writer=writer, ratings_path=tmp_path / "missing-dir" / "r.json", agent_ids=["a"],
                         total_timesteps=10, log_interval=1, console_interval_sec=10.0)
    with pytest.raises(OSError):
        failing.close(env_steps=0, ratings=RATINGS, queue_depths={})
    assert writer.closed
```

(c) Create `tests/unit/test_sp2_checkpoints.py` as a copy of SP1's checkpoint-store tests: the CheckpointManager, resume and coordinator-save tests stay; everything from `test_missing_checkpoint_falls_back_to_latest_and_collects` to the end moves elsewhere (launcher tests: T5.4 `test_sp2_launcher_checkpoints.py`; learner resume and final-checkpoint tests: Part B's T4.4 `tests/unit/test_learner_v2.py`; the worker fallback is Part B's T3.3 contract; `test_latest_network_id_lives_in_core_types` checks an SP1 module layout and is dropped); new tests cover roles and signatures. Save this script as `/tmp/make_sp2_checkpoint_tests.py` and run it:

```python
"""T5.3: tests/unit/test_sp2_checkpoints.py from SP1's tests/unit/test_checkpoint_store.py.

Keeps the CheckpointManager / resume / coordinator-save tests; the launcher tests move to
tests/unit/test_sp2_launcher_checkpoints.py (T5.4); the learner tests are covered by Part B's
tests/unit/test_learner_v2.py (T4.4) and the worker fallback test by Part B's T3.3.
"""
from pathlib import Path

s = Path("tests/unit/test_checkpoint_store.py").read_text()
s = s[: s.index("def test_missing_checkpoint_falls_back_to_latest_and_collects")].rstrip() + "\n"
start, end = s.index('"""Checkpoint store'), s.index("def sd(value: float = 0.0)")
s = s[:start] + '''"""Checkpoint store (SP1 guarantees) plus roles and role signatures in meta.json (T5.3)."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator import checkpoint_manager as cm_module
from colosseum.sp2.coordinator.checkpoint_manager import (
    CheckpointManager,
    check_model_state,
    check_role_signature,
    load_checkpoint_dir,
    resolve_resume,
)
from colosseum.sp2.core.config import ColosseumConfig, config_hash
from colosseum.sp2.learner.learner import apply_resume_state, make_checkpoint_payload, send_checkpoint
from game_helpers import make_coordinator, make_test_config


def make_config(**training) -> ColosseumConfig:
    return make_test_config("solo", training=training, checkpoint={"pool_size": 2},
                            matchmaking={"latest_prob": 0.0, "shuffle_seats": False})


''' + s[end:]
old = '''    cfg = make_config()
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    meta_extra = {"networks": cfg.networks.model_dump(mode="json", by_alias=True),
                  "config_hash": config_hash(cfg), "env_steps": 321}
    ckpt_id = coord.save_checkpoint_payload(received, meta_extra=meta_extra)
    assert ckpt_id == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "agent_0" / "ckpt_v3" / "meta.json").read_text())
    assert meta["final"] is True and meta["env_steps"] == 321
    assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder")
    assert len(meta["config_hash"]) == 16
'''
assert s.count(old) == 1
s = s.replace(old, '''    cfg = make_config()
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    meta_extra = {"networks": cfg.networks.model_dump(mode="json", by_alias=True),
                  "config_hash": config_hash(cfg), "env_steps": 321}
    ckpt_id = coord.save_checkpoint_payload(received, meta_extra=meta_extra)
    assert ckpt_id == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "agent_0" / "ckpt_v3" / "meta.json").read_text())
    assert meta["final"] is True and meta["env_steps"] == 321
    assert meta["networks"]["model_class"] == "game_helpers.GameTestModel"
    assert len(meta["config_hash"]) == 16
    assert meta["roles"] == coord.agent_roles["agent_0"]
    assert meta["role_signature"] == coord.role_signature("agent_0")
''')
s += '''

# ---------------------------------------------------------------------------
# SP2: roles and role signatures (spec block 9)
# ---------------------------------------------------------------------------


def _signed(mgr: CheckpointManager, agent: str, version: int, signature: str = "sig-A") -> None:
    mgr.save(agent, version, sd(version), meta_extra={"roles": ["player"], "role_signature": signature})


def test_meta_roles_and_signature_are_read_back(tmp_path):
    mgr = CheckpointManager(tmp_path / "run" / "checkpoints")
    _signed(mgr, "a", 3)
    loaded = load_checkpoint_dir(tmp_path / "run" / "checkpoints" / "a" / "ckpt_v3")
    assert loaded["roles"] == ["player"] and loaded["role_signature"] == "sig-A"
    state = resolve_resume(str(tmp_path / "run"), "a", expected_signature="sig-A")
    assert state["roles"] == ["player"] and state["role_signature"] == "sig-A" and state["policy_version"] == 3


@pytest.mark.parametrize("source", ["run", "ckpt"])
def test_resume_rejects_a_different_role_signature_naming_the_path(tmp_path, source):
    mgr = CheckpointManager(tmp_path / "run" / "checkpoints")
    _signed(mgr, "a", 3, signature="sig-A")
    path = tmp_path / "run" if source == "run" else tmp_path / "run" / "checkpoints" / "a" / "ckpt_v3"
    with pytest.raises(ConfigError, match="role signature") as exc:
        resolve_resume(str(path), "a", expected_signature="sig-B")
    assert "ckpt_v3" in str(exc.value) and "resume_from" in str(exc.value)


def test_resume_rejects_a_checkpoint_without_signature(tmp_path):
    CheckpointManager(tmp_path / "run" / "checkpoints").save("a", 1, sd(1))
    with pytest.raises(ConfigError, match="no role_signature"):
        resolve_resume(str(tmp_path / "run"), "a", expected_signature="sig-A")
    assert resolve_resume(str(tmp_path / "run"), "a")["role_signature"] is None  # unchecked without expectation


def test_pt_resume_has_no_signature_to_check(tmp_path):
    pt = tmp_path / "bc.pt"
    torch.save({k: torch.tensor(v) for k, v in sd(7).items()}, pt)
    state = resolve_resume(str(pt), "a", expected_signature="sig-A")
    assert state["roles"] is None and state["role_signature"] is None


@pytest.mark.parametrize("meta", [{"roles": "player"}, {"roles": []}, {"roles": [1]}, {"role_signature": 5}],
                         ids=["str_roles", "empty_roles", "int_role", "int_signature"])
def test_malformed_roles_or_signature_make_the_meta_invalid(tmp_path, meta):
    mgr = CheckpointManager(tmp_path)
    mgr.save("a", 2, sd(2), meta_extra=meta)
    with pytest.raises(ConfigError, match="Malformed checkpoint"):
        load_checkpoint_dir(tmp_path / "a" / "ckpt_v2")


def test_check_role_signature_messages():
    check_role_signature({"source": "x", "role_signature": "s"}, "s", "ctx")
    with pytest.raises(ConfigError, match="ctx: checkpoint x has role signature 's'"):
        check_role_signature({"source": "x", "role_signature": "s"}, "t", "ctx")
'''
Path("tests/unit/test_sp2_checkpoints.py").write_text(s)
print("wrote tests/unit/test_sp2_checkpoints.py")
```

```bash
.venv/bin/python /tmp/make_sp2_checkpoint_tests.py
.venv/bin/ruff check --fix tests/unit/test_sp2_checkpoints.py && .venv/bin/ruff check tests/unit/test_sp2_checkpoints.py
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_checkpoints.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.coordinator.coordinator'`, `... 'colosseum.sp2.metrics'`, `... 'colosseum.sp2.coordinator.checkpoint_manager'`.

- [ ] **Step 4: Copy the checkpoint manager and `metrics/jsonl.py`**

Save as `/tmp/make_sp2_checkpoint_manager.py` and run it:

```python
"""T5.3: src/colosseum/sp2/coordinator/checkpoint_manager.py from the SP1 module (run from the repo root)."""
from pathlib import Path

src = Path("src/colosseum/coordinator/checkpoint_manager.py").read_text()
dst = Path("src/colosseum/sp2/coordinator/checkpoint_manager.py")


def rep(old: str, new: str) -> None:
    global src
    assert src.count(old) == 1, f"expected exactly one occurrence of: {old[:80]!r}"
    src = src.replace(old, new)


rep('''        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras
''', '''        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras
                            (SP2: roles and role_signature of the agent; required by resume and eval)
''')
rep('''from colosseum.core.config import check_agent_id, check_path_component
from colosseum.core.errors import ConfigError
from colosseum.core.types import state_dict_from_numpy, state_dict_to_numpy''', '''from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import check_agent_id, check_path_component
from colosseum.sp2.core.types import state_dict_from_numpy, state_dict_to_numpy''')
rep('''    env_steps = meta.get("env_steps")
    if env_steps is not None and not _is_int(env_steps):
        raise ValueError(f"{META_FILE}: env_steps is not an integer or null ({env_steps!r})")
    return meta''', '''    env_steps = meta.get("env_steps")
    if env_steps is not None and not _is_int(env_steps):
        raise ValueError(f"{META_FILE}: env_steps is not an integer or null ({env_steps!r})")
    roles = meta.get("roles")
    if roles is not None and not (isinstance(roles, list) and roles and all(isinstance(r, str) for r in roles)):
        raise ValueError(f"{META_FILE}: roles is not a non-empty list of strings ({roles!r})")
    signature = meta.get("role_signature")
    if signature is not None and not isinstance(signature, str):
        raise ValueError(f"{META_FILE}: role_signature is not a string ({signature!r})")
    return meta''')
rep('''def load_checkpoint_dir(ckpt_dir: str | Path) -> dict[str, Any]:
    """Read one checkpoint dir (e.g. ``<run>/checkpoints/<agent>/ckpt_v<N>``) without touching it.

    Returns ``{"model_state": numpy state_dict, "meta": validated meta.json}``. Unlike''', '''def load_checkpoint_dir(ckpt_dir: str | Path) -> dict[str, Any]:
    """Read one checkpoint dir (e.g. ``<run>/checkpoints/<agent>/ckpt_v<N>``) without touching it.

    Returns ``{"model_state": numpy state_dict, "meta": validated meta.json, "roles": list | None,
    "role_signature": str | None}`` (the last two from ``meta.json``; None when absent). Unlike''')
rep('''        raise ConfigError(f"Malformed checkpoint {path}: {e}") from e
    return {"model_state": model_state, "meta": meta}''', '''        raise ConfigError(f"Malformed checkpoint {path}: {e}") from e
    return {"model_state": model_state, "meta": meta,
            "roles": meta.get("roles"), "role_signature": meta.get("role_signature")}''')
rep('''    return {
        "model_state": model_state,
        "trainer_state": trainer_state,
        "policy_version": version,
        "env_steps": env_steps,
        "source": str(ckpt_dir),
    }''', '''    return {
        "model_state": model_state,
        "trainer_state": trainer_state,
        "policy_version": version,
        "env_steps": env_steps,
        "source": str(ckpt_dir),
        "roles": meta.get("roles"),
        "role_signature": meta.get("role_signature"),
    }


def check_role_signature(state: dict, expected: str, context: str) -> None:
    """ConfigError naming ``state["source"]`` unless the checkpoint's role signature is ``expected``.

    ``state`` is a ``resolve_resume`` result (or anything with ``source`` and ``role_signature``).
    A checkpoint without a signature (written before SP2) is rejected too.
    """
    signature = state.get("role_signature")
    if signature is None:
        raise ConfigError(
            f"{context}: checkpoint {state['source']} has no role_signature in its meta.json "
            f"(written before SP2); SP1 checkpoints cannot be resumed"
        )
    if signature != expected:
        raise ConfigError(
            f"{context}: checkpoint {state['source']} has role signature {signature!r}, but the agent's "
            f"roles in this game have {expected!r}: the observation/action/global-state spaces differ"
        )''')
rep('''def resolve_resume(resume_from: str, agent_id: str) -> dict | None:''',
    '''def resolve_resume(resume_from: str, agent_id: str, expected_signature: str | None = None) -> dict | None:''')
rep('''    The result holds only numpy arrays, bytes and primitives, so it can be passed
    to a learner process.
    """''', '''    With ``expected_signature`` (``core.roles.role_signature`` of the agent's roles), a checkpoint
    source must carry exactly that ``role_signature`` in its ``meta.json`` (``check_role_signature``);
    a ``.pt`` file has no signature and is checked only by ``check_model_state``.

    The result holds only numpy arrays, bytes and primitives (plus ``roles`` and
    ``role_signature``, None for a ``.pt``), so it can be passed to a learner process.
    """''')
rep('''        if not infos:
            logger.warning(f"resume_from={resume_from}: no checkpoints for agent '{agent_id}'; starting fresh")
            return None
        return _load_checkpoint_dir(infos[-1].path, resume_from)
    if kind == RESUME_CHECKPOINT_DIR:
        return _load_checkpoint_dir(path, resume_from)
    try:
        model_state = read_weights_file(path)
    except ValueError as e:
        raise ConfigError(f"training.resume_from={resume_from!r}: {e}") from e
    return {"model_state": model_state, "trainer_state": None,
            "policy_version": 0, "env_steps": 0, "source": str(path)}''', '''        if not infos:
            logger.warning(f"resume_from={resume_from}: no checkpoints for agent '{agent_id}'; starting fresh")
            return None
        state = _load_checkpoint_dir(infos[-1].path, resume_from)
    elif kind == RESUME_CHECKPOINT_DIR:
        state = _load_checkpoint_dir(path, resume_from)
    else:
        try:
            model_state = read_weights_file(path)
        except ValueError as e:
            raise ConfigError(f"training.resume_from={resume_from!r}: {e}") from e
        return {"model_state": model_state, "trainer_state": None, "policy_version": 0, "env_steps": 0,
                "source": str(path), "roles": None, "role_signature": None}
    if expected_signature is not None:
        check_role_signature(state, expected_signature, f"training.resume_from={resume_from!r} [{agent_id}]")
    return state''')
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src)
print(f"wrote {dst}")
```

Save as `/tmp/make_sp2_jsonl.py` and run it (only the record shapes change; `GLOBAL_KINDS` / `METRIC_KINDS` and the writer stay as in SP1, so the unchanged `colosseum.metrics.wandb_logger` keeps working):

```python
"""T5.3: src/colosseum/sp2/metrics/jsonl.py from the SP1 module (run from the repo root)."""
from pathlib import Path

s = Path("src/colosseum/metrics/jsonl.py").read_text()
for old, new in [
    ('"""metrics.jsonl: the source of truth for training metrics (WandB is only a viewer)."""',
     '"""metrics.jsonl: the source of truth for training metrics (WandB is only a viewer).\n\n'
     'SP2 record shapes: ``episodes`` records carry ``by_layout`` (per layout and role) and ``ratings``\n'
     'records carry ``layouts`` (``RatingBook.snapshot()``: one table set per layout).\n"""'),
    ('''    "episodes": {"ts", "kind", "agent", "env_steps", "episodes", "return_mean", "length_mean", "wdl",
                 "seat_counts"},
    "ratings": {"ts", "kind", "env_steps", "elo", "win_rates", "games", "wr_vs_past"},''',
     '''    "episodes": {"ts", "kind", "agent", "env_steps", "episodes", "return_mean", "length_mean", "wdl",
                 "seat_counts", "by_layout"},
    "ratings": {"ts", "kind", "env_steps", "layouts"},'''),
]:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
Path("src/colosseum/sp2/metrics/jsonl.py").write_text(s)
print("wrote src/colosseum/sp2/metrics/jsonl.py")
```

```bash
mkdir -p src/colosseum/sp2/metrics && touch src/colosseum/sp2/metrics/__init__.py
.venv/bin/python /tmp/make_sp2_checkpoint_manager.py
.venv/bin/python /tmp/make_sp2_jsonl.py
.venv/bin/ruff check src/colosseum/sp2/coordinator/checkpoint_manager.py src/colosseum/sp2/metrics/jsonl.py
```

- [ ] **Step 5: Write the coordinator, aggregator and hub**

Create `src/colosseum/sp2/coordinator/coordinator.py`:

```python
"""Coordinator: matchmaking, checkpoint storage and ratings of a training run.

Manages:
- the agent pool (trainable agents) and their roles;
- owner rotation over the trainable agents and one ``Lineup`` per env (``LineupMatchmaker``);
- checkpoint storage (learner checkpoint payloads -> ``CheckpointManager``; ``meta.json`` gets the
  agent's roles and their role signature);
- match results and per-layout ratings (``RatingBook``).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager
from colosseum.sp2.coordinator.matchmaker import LineupMatchmaker
from colosseum.sp2.coordinator.ratings import RatingBook
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.roles import agent_role_spec, role_signature
from colosseum.sp2.core.types import Lineup, MatchResult
from colosseum.sp2.envs.game import GameSpec

logger = logging.getLogger(__name__)


class Coordinator:
    """Central coordinator of a single-machine training run."""

    def __init__(self, config: ColosseumConfig, spec: GameSpec, agent_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path) -> None:
        self._config = config
        self._spec = spec
        trainable = config.get_trainable_agent_ids()
        missing = [a for a in trainable if a not in agent_roles]
        if missing:
            raise ConfigError(f"Coordinator: no roles resolved for agents {missing}")
        self._agent_roles = {a: list(agent_roles[a]) for a in trainable}
        # One RNG for matchmaking and seat permutations: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
        self._agent_pool = AgentPool()
        for agent_id in trainable:
            self._agent_pool.register_trainable(agent_id)
        self._checkpoint_manager = CheckpointManager(base_dir=checkpoint_dir, pool_size=config.checkpoint.pool_size)
        self._ratings = RatingBook(spec, trainable)
        self._role_signatures = {a: role_signature(agent_role_spec(spec, roles))
                                 for a, roles in self._agent_roles.items()}
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._refresh_round = 0
        self._matchmaker = LineupMatchmaker(
            spec=spec, agent_roles=self._agent_roles, config=config.matchmaking,
            checkpoints=self._checkpoint_ids, win_rate=self._ratings.win_rate, rng=self._rng,
        )

    @property
    def agent_pool(self) -> AgentPool:
        return self._agent_pool

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def ratings(self) -> RatingBook:
        return self._ratings

    @property
    def matchmaker(self) -> LineupMatchmaker:
        return self._matchmaker

    @property
    def spec(self) -> GameSpec:
        return self._spec

    @property
    def agent_roles(self) -> dict[str, list[str]]:
        return {a: list(r) for a, r in self._agent_roles.items()}

    @property
    def refresh_round(self) -> int:
        return self._refresh_round

    def role_signature(self, agent_id: str) -> str:
        return self._role_signatures[agent_id]

    def _checkpoint_ids(self, agent_id: str) -> list[str]:
        return [c.checkpoint_id for c in self._checkpoint_manager.list_checkpoints(agent_id)]

    def next_round(self) -> None:
        """Advance the owner rotation. The launcher calls this once per match refresh."""
        self._refresh_round += 1

    def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]:
        """One lineup per env. Env ``e`` of this batch has global index ``g = env_offset + e``;
        its owner is ``agents[(g + refresh_round) % n_trainable]`` (SP1 rotation), so every
        trainable agent owns envs, and ownership rotates between refreshes."""
        agents = [a.agent_id for a in self._agent_pool.list_trainable()]
        if not agents:
            raise ValueError("Coordinator has no trainable agents")
        return [self._matchmaker.lineup_for(agents[(env_offset + e + self._refresh_round) % len(agents)])
                for e in range(num_envs)]

    def report_match_result(self, result: MatchResult) -> None:
        """Keep the result and update the ratings of its layout."""
        self._match_results.append(result)
        self._ratings.update(result)

    @property
    def match_results(self) -> list[MatchResult]:
        return list(self._match_results)

    def ratings_snapshot(self) -> dict:
        """``RatingBook.snapshot()``: JSON-serializable tables per layout."""
        return self._ratings.snapshot()

    def save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str:
        """Persist a learner checkpoint payload (see ``learner.make_checkpoint_payload``).

        ``meta.json`` gets ``final``, ``meta_extra``, and the agent's ``roles`` and
        ``role_signature``. The trainer state is kept only when ``checkpoint.save_optimizer``.
        """
        agent_id = payload["agent_id"]
        trainer_state = payload.get("trainer_state_bytes") if self._config.checkpoint.save_optimizer else None
        meta = {"final": bool(payload.get("final", False)), **(meta_extra or {}),
                "roles": list(self._agent_roles[agent_id]), "role_signature": self._role_signatures[agent_id]}
        return self._checkpoint_manager.save(
            agent_id=agent_id,
            policy_version=int(payload["policy_version"]),
            model_state=payload["model_state"],
            trainer_state=trainer_state,
            meta_extra=meta,
        )
```

Create `src/colosseum/sp2/metrics/aggregator.py` (`SystemStats` is SP1's, unchanged):

```python
"""Per-interval aggregation of match results (by layout and role) and system throughput."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from colosseum.sp2.core.types import LATEST_NETWORK_ID, MatchResult, SeatResult

OPPONENT_TYPES = ("latest", "past", "arena")
# Default cadence of worker stats (``rollout_worker_process(stats_interval_sec=...)``).
WORKER_STATS_INTERVAL_SEC = 2.0


def opponent_type(result: MatchResult, seat: SeatResult) -> str | None:
    """Kind of opposition a seat met; teammates are not opponents.

    'arena' (a seat of another team plays another agent), 'past' (one plays a checkpoint of the
    seat's agent), 'latest' (all play the agent's latest weights), or None (no other team: solo
    or cooperative layouts).
    """
    opponents = [s for s in result.seats if s.team != seat.team]
    if not opponents:
        return None
    if any(s.agent_id != seat.agent_id for s in opponents):
        return "arena"
    if any(s.network_id != LATEST_NETWORK_ID for s in opponents):
        return "past"
    return "latest"


class EpisodeAggregator:
    """Per agent, over the seats that play the agent's latest weights:

    - mean return and episode length, W/D/L by opponent type (win: the seat's team ranks strictly
      better than every other team; draw: ties the best one), seat counts;
    - ``by_layout[layout][role]``: episodes (seats), mean return, mean team score and the share
      of seats eliminated before the episode end.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._lengths: dict[str, list[int]] = defaultdict(list)
        self._wdl: dict[str, dict[str, list[int]]] = defaultdict(lambda: {t: [0, 0, 0] for t in OPPONENT_TYPES})
        self._seats: dict[str, list[int]] = defaultdict(list)
        # agent -> layout -> role -> [seats, return sum, team score sum, eliminated]
        self._by_layout: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(dict))

    def add(self, result: MatchResult) -> None:
        ranks = {team.team: team.rank for team in result.teams}
        scores = {team.team: team.score for team in result.teams}
        for seat in result.seats:
            if seat.network_id != LATEST_NETWORK_ID:
                continue
            agent_id = seat.agent_id
            self._returns[agent_id].append(float(seat.reward))
            self._lengths[agent_id].append(int(result.episode_length))
            counts = self._seats[agent_id]
            while len(counts) <= seat.seat:
                counts.append(0)
            counts[seat.seat] += 1
            cell = self._by_layout[agent_id][result.layout].setdefault(seat.role, [0, 0.0, 0.0, 0])
            cell[0] += 1
            cell[1] += float(seat.reward)
            cell[2] += float(scores.get(seat.team, 0.0))
            cell[3] += int(seat.eliminated_step is not None)
            kind = opponent_type(result, seat)
            if kind is None:
                continue
            own = ranks[seat.team]
            best_other = min(rank for team, rank in ranks.items() if team != seat.team)
            idx = 0 if own < best_other else (1 if own == best_other else 2)
            self._wdl[agent_id][kind][idx] += 1

    def flush(self) -> dict[str, dict[str, Any]]:
        """Stats since the previous flush, per agent; resets the accumulators."""
        out: dict[str, dict[str, Any]] = {}
        for agent_id, returns in self._returns.items():
            by_layout = {
                layout: {
                    role: {"episodes": int(n), "return_mean": ret / n, "team_score_mean": score / n,
                           "eliminated_frac": elim / n}
                    for role, (n, ret, score, elim) in sorted(roles.items())
                }
                for layout, roles in sorted(self._by_layout[agent_id].items())
            }
            out[agent_id] = {
                "episodes": len(returns),
                "return_mean": float(np.mean(returns)),
                "length_mean": float(np.mean(self._lengths[agent_id])),
                "wdl": {t: list(v) for t, v in self._wdl[agent_id].items()},
                "seat_counts": list(self._seats[agent_id]),
                "by_layout": by_layout,
            }
        self._reset()
        return out


class SystemStats:
    """Throughput and queue health between two snapshots (SP1 behavior, unchanged).

    Rate baselines start at ``initial_env_steps`` and ``initial_train_steps[agent]`` (0 for
    agents not listed): the counters a resumed run continues from, so the first snapshot
    of a resumed run measures only progress made since the start, not the resumed totals.
    Per-worker stats older than ``worker_timeout_sec`` (a dead or stuck worker) are forgotten.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, *, initial_env_steps: int = 0,
                 initial_train_steps: dict[str, int] | None = None,
                 worker_timeout_sec: float = 3 * WORKER_STATS_INTERVAL_SEC) -> None:
        self._clock = clock
        self._last_t = clock()
        self._last_env_steps = int(initial_env_steps)
        self._last_train_steps: dict[str, int] = {a: int(s) for a, s in (initial_train_steps or {}).items()}
        self._train_steps: dict[str, int] = dict(self._last_train_steps)
        self._worker_timeout = float(worker_timeout_sec)
        self._workers: dict[int, tuple[float, dict]] = {}

    def on_train_step(self, agent_id: str, train_step: int) -> None:
        self._train_steps[agent_id] = max(int(train_step), self._train_steps.get(agent_id, 0))

    def on_worker_stats(self, stats: dict) -> None:
        self._workers[int(stats["worker_id"])] = (self._clock(), dict(stats))

    def snapshot(self, env_steps: int, queue_depths: dict[str, int]) -> dict[str, Any]:
        now = self._clock()
        dt = now - self._last_t
        self._workers = {w: v for w, v in self._workers.items() if now - v[0] <= self._worker_timeout}

        def rate(delta: float) -> float:
            return delta / dt if dt > 1e-3 else 0.0

        snap = {
            "env_steps": int(env_steps),
            "env_steps_per_sec": rate(env_steps - self._last_env_steps),
            "train_steps_per_sec": {
                a: rate(s - self._last_train_steps.get(a, 0)) for a, s in self._train_steps.items()
            },
            "queue_depths": dict(queue_depths),
            "parked_buffers": int(sum(int(w.get("parked_buffers", 0)) for _, w in self._workers.values())),
            "workers_reporting": len(self._workers),
        }
        self._last_t = now
        self._last_env_steps = int(env_steps)
        self._last_train_steps = dict(self._train_steps)
        return snap
```

Create `src/colosseum/sp2/metrics/hub.py`:

```python
"""MetricsHub: the main process's single sink for train metrics, results and system stats."""

from __future__ import annotations

import numbers
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from colosseum.metrics.console import ConsoleReporter
from colosseum.sp2.metrics.aggregator import EpisodeAggregator, SystemStats
from colosseum.sp2.metrics.jsonl import MetricsWriter, write_json_atomic


def _is_number(value: Any) -> bool:
    """Real numbers, python or numpy scalars; bools are not numbers here."""
    return (isinstance(value, (numbers.Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)))


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if _is_number(v)}


def flatten(prefix: str, value: Any) -> dict[str, float]:
    """Nested dicts/lists of numbers -> ``{"prefix/a/b": float}``; non-numbers are dropped."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(f"{prefix}/{k}", v))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten(f"{prefix}/{i}", v))
    elif _is_number(value):
        out[prefix] = float(value)
    return out


def wr_vs_past_over_layouts(layouts: Mapping[str, Mapping[str, Any]], agent_id: str) -> float | None:
    """``wr_vs_past`` of ``agent_id`` over all layouts, weighted by each layout's ``past_games``."""
    total = weight = 0.0
    for table in layouts.values():
        value = table.get("wr_vs_past", {}).get(agent_id)
        games = table.get("past_games", {}).get(agent_id, 0)
        if value is not None and games:
            total += value * games
            weight += games
    return total / weight if weight else None


def wr_arena_over_layouts(layouts: Mapping[str, Mapping[str, Any]], agent_id: str) -> float | None:
    """Win rate of ``agent_id`` against other agents over all layouts, weighted by games."""
    total = games = 0.0
    for table in layouts.values():
        rates = table.get("win_rates", {}).get(agent_id, {})
        for other, n in table.get("games", {}).get(agent_id, {}).items():
            if n:
                total += rates[other] * n
                games += n
    return total / games if games else None


class MetricsHub:
    """Writes ``train`` records as they arrive (every ``log_interval`` train steps per agent).

    Every ``console_interval_sec`` it also writes ``episodes``, ``system`` and ``ratings``
    records (``ratings``: ``{"env_steps", "layouts"}`` with one table set per layout), rewrites
    ``ratings.json`` with the same content and prints one console line per agent (ratings
    aggregated over layouts). ``initial_env_steps`` / ``initial_train_steps`` are the counters
    a resumed run continues from (rate baselines, see ``SystemStats``).
    """

    def __init__(self, *, writer: MetricsWriter, ratings_path: str | Path, agent_ids: list[str],
                 total_timesteps: int, log_interval: int, console_interval_sec: float,
                 wandb_logger: Any = None, clock: Callable[[], float] = time.monotonic,
                 initial_env_steps: int = 0, initial_train_steps: dict[str, int] | None = None) -> None:
        self._writer = writer
        self._ratings_path = Path(ratings_path)
        self._agent_ids = list(agent_ids)
        self._log_interval = max(1, int(log_interval))
        self._interval = float(console_interval_sec)
        self._wandb = wandb_logger
        self._clock = clock
        self._episodes = EpisodeAggregator()
        self._system = SystemStats(clock=clock, initial_env_steps=initial_env_steps,
                                   initial_train_steps=initial_train_steps)
        self._console = ConsoleReporter(total_timesteps)
        self._last_tick = clock()
        self._last_train: dict[str, dict[str, float]] = {}
        self._last_logged_step: dict[str, int] = {}
        self._last_returns: dict[str, float] = {}

    def on_train_metrics(self, metrics: dict) -> None:
        agent_id = str(metrics.get("agent_id", "agent_0"))
        step = int(metrics.get("train_step", 0))
        values = {k: v for k, v in _numeric(metrics).items() if k != "train_step"}
        self._system.on_train_step(agent_id, step)
        self._last_train[agent_id] = {"train_step": float(step), **values}
        last = self._last_logged_step.get(agent_id)
        if last is None or step - last >= self._log_interval:
            self._writer.write("train", agent=agent_id, train_step=step, **values)
            if self._wandb is not None:
                self._wandb.log_train(agent_id, values, step)
            self._last_logged_step[agent_id] = step

    def on_worker_stats(self, stats: dict) -> None:
        self._system.on_worker_stats(stats)

    def on_match_result(self, result) -> None:
        self._episodes.add(result)

    def maybe_tick(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int],
                   force: bool = False) -> bool:
        """``ratings`` is ``RatingBook.snapshot()``: ``{layout: {elo, win_rates, ...}}``."""
        now = self._clock()
        if not force and now - self._last_tick < self._interval:
            return False
        self._last_tick = now
        episodes = self._episodes.flush()
        for agent_id, ep in episodes.items():
            self._writer.write("episodes", agent=agent_id, env_steps=int(env_steps), **ep)
            self._last_returns[agent_id] = ep["return_mean"]
        system = self._system.snapshot(env_steps, queue_depths)
        self._writer.write("system", **system)
        self._writer.write("ratings", env_steps=int(env_steps), layouts=ratings)
        write_json_atomic(self._ratings_path, {"env_steps": int(env_steps), "layouts": ratings})
        if self._wandb is not None:
            row = flatten("system", {k: v for k, v in system.items() if k != "env_steps"})
            row.update(flatten("ratings", ratings))
            for agent_id, ep in episodes.items():
                row.update(flatten(f"episodes/{agent_id}", {k: v for k, v in ep.items() if k != "wdl"}))
            self._wandb.log_global(row, int(env_steps))
        self._report_console(int(env_steps), system, ratings)
        return True

    def _report_console(self, env_steps: int, system: dict, ratings: dict) -> None:
        lines = []
        for agent_id in self._agent_ids:
            train = self._last_train.get(agent_id, {})
            lines.append(self._console.format_line(
                agent_id,
                train_step=int(train.get("train_step", 0)),
                env_steps=env_steps,
                fps=float(system["env_steps_per_sec"]),
                loss=train.get("total_loss"),
                entropy=train.get("entropy"),
                return_mean=self._last_returns.get(agent_id),
                wr_vs_past=wr_vs_past_over_layouts(ratings, agent_id),
                wr_arena=wr_arena_over_layouts(ratings, agent_id),
            ))
        self._console.emit(lines)

    def close(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int]) -> None:
        """Final records and ratings.json, then close the file (also if writing them fails)."""
        try:
            self.maybe_tick(env_steps=env_steps, ratings=ratings, queue_depths=queue_depths, force=True)
        finally:
            self._writer.close()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_checkpoints.py -q`
Expected: all passed (7 + 11 + the copied SP1 checkpoint tests + 10 new role/signature cases).

- [ ] **Step 7: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/sp2/coordinator src/colosseum/sp2/metrics tests/game_helpers.py pyproject.toml \
        tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_checkpoints.py
git commit -m "feat: SP2 coordinator, checkpoints with role signatures, per-layout metrics (SP2 T5.3)"
git push origin sp2-game-model
```

---

### Task T5.4: Launcher, run dir, CLI `train`; integration runs on toy games

The `sp2` launcher is SP1's launcher (after T0.1) on the `sp2` types: same processes, queues, signals, exit codes 0/1/130/143, child-death detection, final checkpoints within `SHUTDOWN_GRACE_SEC`, run dir layout. New: the game's `GameSpec` and the agents' roles drive everything (`RunSetup`), learners build models for their role spec, workers get initial `Lineup`s and `WorkerCommand(lineups=...)` refreshes, resume checks role signatures.

**Files:**
- Create: `src/colosseum/sp2/core/run_dir.py` (copy of `src/colosseum/core/run_dir.py` + one import edit)
- Create: `src/colosseum/sp2/launcher.py` (copy of `src/colosseum/launcher.py` + edits by script)
- Create: `src/colosseum/sp2/cli.py`, `src/colosseum/sp2/__main__.py` (overwrite if Part A created a placeholder)
- Modify: `tests/cli_runner.py` (`module` parameter, `TINY_SP2`, `tests/` on the children's `PYTHONPATH`)
- Modify: `tests/game_helpers.py` (append `make_test_run_dir`)
- Test: `tests/unit/test_sp2_launcher_checkpoints.py`, `tests/unit/test_sp2_run_dir.py` (copy of `tests/unit/test_run_dir_logging.py` + edits), `tests/integration/test_sp2_game_runs.py`

**Interfaces:**
- Consumes: `Coordinator` and the checkpoint functions (T5.3); `validate_matchmaking` (T5.1); `env_spec`, `build_model(agent_config, role)`, `import_class` (T2.4); `resolve_agent_roles`, `agent_role_spec`, `role_signature` (T2.4); `ActionSpec.from_space` (T1.3); `rollout_worker_process(..., agent_roles, lineups, max_idle_steps, ...)` (T3.4); `learner_process`, `resolve_device` (T4.4); `APPO(model, config, action_spec, device, pin_memory, kickstart)` and `KickstartLoss` (T4.2); `WorkerCommand(lineups, new_checkpoints)`, `Lineup`, `SeatAssignment` (T3.1); `MetricsHub`, `MetricsWriter` (T5.3); unchanged SP1 modules `colosseum.core.ipc`, `colosseum.utils.*`, `colosseum.metrics.wandb_logger`.
- Produces:
  - `Launcher(config, run_dir, validated=False)`, `Launcher.launch() -> int`, `run_training(config_path, overrides=None) -> int`, `RunDir` (contract).
  - *`RunSetup(spec, agent_roles, agent_configs, role_specs)`, *`setup_run(config, validate=True) -> RunSetup`, *`validate_run_config(config)` (T6.2 replaces its body with `registry.validate_config`).
  - *`_resolve_lineups(lineups, coordinator, agent_ids, already_sent=None) -> (new_checkpoints, lineups)` (replaces `_derive_worker_configs`); `Launcher._resolve_resume(agent_configs, role_specs)`; `_learner_main(..., role_spec, ...)`; `_worker_main(..., agent_roles, agent_configs, role_specs, ..., lineups, ...)`; `_create_env`, `_WEIGHT_QUEUE_SIZE`, `warn_static_ownership_skew` unchanged.
  - `python -m colosseum.sp2 train -c cfg.yaml [--set k=v ...]` (`colosseum.sp2.cli.main`).
  - `tests/cli_runner.py`: `train_cmd/run_train/start_train/training_process(..., module="colosseum")`, `TINY_SP2`, `TESTS_DIR`; `tests/game_helpers.py`: `make_test_run_dir(config, tmp_path, name="test-run")`.

- [ ] **Step 1: Test support: CLI runner and run-dir helper**

Replace `tests/cli_runner.py` with (SP1 behavior for `module="colosseum"` is unchanged; `module="colosseum.sp2"` uses the SP2 config schema; `tests/` goes on the children's `PYTHONPATH` so configs can name `game_helpers.*` classes):

```python
"""Run ``colosseum train`` (or ``python -m colosseum.sp2 train``) in a subprocess for integration
and learning tests."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
TTT_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"

# Small, fast settings for tic-tac-toe runs (about 10 s with 1 worker on CPU).
TINY: dict[str, str] = {
    "training.total_timesteps": "3000",
    "rollout.num_workers": "1",
    "rollout.envs_per_worker": "8",
    "rollout.chunk_length": "16",
    "rollout.weight_sync_interval_sec": "0.5",
    "rollout.match_refresh_interval_sec": "1.0",
    "learner.batch_chunks": "2",
    "learner.queue_size": "16",
    "self_play.checkpoint_interval": "20",
    "self_play.pool_size": "5",
    "metrics.log_interval": "1",
    "metrics.console_interval_sec": "1.0",
}
# The same for the SP2 config schema (``python -m colosseum.sp2``): checkpoint.* replaces self_play.*.
TINY_SP2: dict[str, str] = {
    **{k: v for k, v in TINY.items() if not k.startswith("self_play.")},
    "checkpoint.interval": "20",
    "checkpoint.pool_size": "5",
}


def child_env() -> dict[str, str]:
    """The test process's environment for a CLI child; ``tests/`` goes on PYTHONPATH, so configs
    may name support modules (``game_helpers.*``)."""
    env = dict(os.environ)
    env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")]))
    return env


def train_cmd(config: Path, run_parent: Path, name: str, overrides: dict[str, str] | None = None,
              module: str = "colosseum") -> list[str]:
    """``python -m <module> train`` with the tiny settings of that module's config schema."""
    tiny = TINY_SP2 if module == "colosseum.sp2" else TINY
    sets = {**tiny, **(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", module, "train", "-c", str(config)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


@dataclass
class TrainRun:
    returncode: int
    stdout: str
    stderr: str
    root: Path

    def records(self, kind: str | None = None) -> list[dict]:
        path = self.root / "metrics.jsonl"
        if not path.exists():
            return []
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in records if kind is None or r["kind"] == kind]

    def log(self, process_name: str) -> str:
        return (self.root / "logs" / f"{process_name}.log").read_text()

    def ratings(self) -> dict:
        return json.loads((self.root / "ratings.json").read_text())


def run_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
              timeout: float = 240.0, env: dict[str, str] | None = None, module: str = "colosseum") -> TrainRun:
    """``env`` entries are added to ``child_env()``."""
    run_parent = tmp_path / "runs"
    proc = run_in_session(train_cmd(config, run_parent, name, overrides, module), timeout, env)
    return TrainRun(proc.returncode, proc.stdout, proc.stderr, run_parent / name)


def run_in_session(cmd: list[str], timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` from the repo root in its own session (``env`` added to ``child_env()``);
    whatever is left of the session afterwards (also on a timeout) is SIGKILLed."""
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env={**child_env(), **(env or {})}, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    finally:
        _kill_group(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the session started for ``proc`` (whatever is left of it) and reap ``proc``."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
    except ProcessLookupError:
        pass
    proc.wait()


def start_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                env: dict[str, str] | None = None, module: str = "colosseum") -> tuple[subprocess.Popen, Path]:
    """Start training in its own session; stdout/stderr go to files next to the run dir."""
    run_parent = tmp_path / "runs"
    # The child keeps its own copies of the descriptors; the parent's handles close here.
    with open(tmp_path / f"{name}.stdout", "w") as out, open(tmp_path / f"{name}.stderr", "w") as err:
        proc = subprocess.Popen(train_cmd(config, run_parent, name, overrides, module), cwd=REPO_ROOT,
                                env={**child_env(), **(env or {})},
                                stdout=out, stderr=err, text=True, start_new_session=True)
    return proc, run_parent / name


@contextmanager
def training_process(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                     env: dict[str, str] | None = None,
                     module: str = "colosseum") -> Iterator[tuple[subprocess.Popen, Path]]:
    """``start_train`` whose whole process group (main, children, env grandchildren) is
    SIGKILLed and reaped on exit, so a failing test leaves no orphans behind."""
    proc, root = start_train(config, tmp_path, name, overrides, env, module)
    try:
        yield proc, root
    finally:
        _kill_group(proc)


def wait_for(predicate: Callable[[], bool], timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
```

Append to `tests/game_helpers.py`:

```python
def make_test_run_dir(config, tmp_path, name: str = "test-run"):
    """Point ``config.run`` at ``tmp_path / "runs"`` and create the run dir (tests never write to cwd)."""
    from colosseum.sp2.core.run_dir import RunDir

    config.run.dir = str(tmp_path / "runs")
    config.run.name = name
    return RunDir.create(config)
```

- [ ] **Step 2: Write the failing tests**

(a) Create `tests/unit/test_sp2_launcher_checkpoints.py` (SP1's launcher tests of `test_checkpoint_store.py` on lineups and roles):

```python
"""SP2 launcher: lineup resolution, refresh deltas, resume (with role signatures), shutdown saves (T5.4).

Ported from SP1's tests/unit/test_checkpoint_store.py (launcher part) onto lineups and roles.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager
from colosseum.sp2.core.config import config_hash
from colosseum.sp2.core.registry import build_model
from colosseum.sp2.core.roles import role_signature
from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.sp2.launcher import Launcher, _resolve_lineups, setup_run
from colosseum.sp2.learner.learner import make_checkpoint_payload, send_checkpoint
from game_helpers import agent_role_of, make_coordinator, make_test_config, make_test_run_dir


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


class FakeAlgorithm:
    """The BaseAlgorithm members used by checkpointing."""

    def __init__(self, version: int = 0):
        self._model = nn.Linear(3, 2)
        self._opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._version = version

    @property
    def model(self):
        return self._model

    @property
    def policy_version(self) -> int:
        return self._version

    def train_once(self):
        self._opt.zero_grad()
        self._model(torch.ones(4, 3)).sum().backward()
        self._opt.step()
        self._version += 1

    def state_dict(self):
        return {"optimizer": self._opt.state_dict(), "policy_version": self._version, "consumed_samples": 0}

    def load_state_dict(self, state):
        self._opt.load_state_dict(state["optimizer"])
        self._version = int(state["policy_version"])


def model_state(config, agent_id: str = "agent_0") -> dict[str, np.ndarray]:
    _roles, role = agent_role_of(config, agent_id)
    model = build_model(config.get_agent_config(agent_id), role)
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def signed_meta(config, agent_id: str = "agent_0", **extra) -> dict:
    roles, role = agent_role_of(config, agent_id)
    return {"roles": roles, "role_signature": role_signature(role), **extra}


def test_missing_checkpoint_falls_back_to_latest_and_collects(tmp_path, caplog):
    cfg = make_test_config("turns")
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    layout = next(iter(coord.spec.layouts))
    lineup = Lineup(layout, [SeatAssignment("agent_0", "ckpt_v10", False),
                             SeatAssignment("agent_0", "ckpt_v999", False)])
    new_ckpts, (resolved,) = _resolve_lineups([lineup], coord, ["agent_0"])
    assert [(s.network_id, s.collect) for s in resolved.seats] == [("ckpt_v10", False), (LATEST_NETWORK_ID, True)]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"] and "ckpt_v999" in caplog.text
    again, (resolved2,) = _resolve_lineups([lineup], coord, ["agent_0"], already_sent={"agent_0": {"ckpt_v10"}})
    assert again["agent_0"] == {}  # already on the worker: not reloaded or resent
    assert resolved2.seats[0].network_id == "ckpt_v10"


def test_resolve_lineups_two_agents(tmp_path):
    cfg = make_test_config("turns", agents={"alpha": {}, "beta": {}})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("beta", 4, sd(4))
    layout = next(iter(coord.spec.layouts))
    lineups = [
        Lineup(layout, [SeatAssignment("alpha"), SeatAssignment("beta")]),
        Lineup(layout, [SeatAssignment("alpha"), SeatAssignment("beta", "ckpt_v4", False)]),
    ]
    new_ckpts, resolved = _resolve_lineups(lineups, coord, ["alpha", "beta"])
    assert [[(s.agent_id, s.network_id, s.collect) for s in lu.seats] for lu in resolved] == [
        [("alpha", "latest", True), ("beta", "latest", True)],
        [("alpha", "latest", True), ("beta", "ckpt_v4", False)],
    ]
    assert new_ckpts["alpha"] == {} and float(new_ckpts["beta"]["ckpt_v4"]["w"][0, 0]) == 4.0


def test_refresh_marks_checkpoints_sent_only_after_successful_put(tmp_path):
    cfg = make_test_config("turns", matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    launcher = Launcher.__new__(Launcher)  # only _config is needed by _refresh_worker_matches
    launcher._config = cfg
    q = mp.get_context("spawn").Queue(maxsize=1)
    q.put("occupied")
    sent = [{"agent_0": set()}]
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert sent[0]["agent_0"] == set()  # put failed (queue full): nothing marked
    assert q.get(timeout=5) == "occupied"
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    cmd = q.get(timeout=5)
    assert "ckpt_v10" in cmd.new_checkpoints["agent_0"]
    assert len(cmd.lineups) == cfg.rollout.envs_per_worker
    assert sent[0]["agent_0"] == {"ckpt_v10"}
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert q.get(timeout=5).new_checkpoints == {}  # delta only


def _old_run(tmp_path: Path, cfg, version: int, env_steps: int, **meta) -> Path:
    run = tmp_path / "old_run"
    CheckpointManager(run / "checkpoints").save(
        "agent_0", version, model_state(cfg), meta_extra={"env_steps": env_steps, **signed_meta(cfg), **meta},
    )
    return run


def _resume(cfg, tmp_path, name="resumed"):
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path, name=name))
    setup = setup_run(cfg, validate=False)
    return launcher, launcher._resolve_resume(setup.agent_configs, setup.role_specs)


def test_resume_seeds_global_env_step_counter(tmp_path):
    run = _old_run(tmp_path, make_test_config("solo"), version=12, env_steps=1200)
    launcher, states = _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert states["agent_0"]["policy_version"] == 12
    assert launcher.env_steps_done == 1200


def test_resume_from_a_run_dir_written_by_the_launcher(tmp_path):
    cfg = make_test_config("solo")
    first = make_test_run_dir(cfg, tmp_path, name="first")
    launcher = Launcher(cfg, first)
    launcher._coordinator = make_coordinator(cfg, first.checkpoints)  # as in launch()
    launcher._env_step_counter.add(321)
    launcher._save_checkpoint({"agent_id": "agent_0", "policy_version": 7, "final": True,
                               "model_state": model_state(cfg), "trainer_state_bytes": None})
    assert (first.checkpoints / "agent_0" / "ckpt_v7" / "model.pt").is_file()
    resumed, states = _resume(make_test_config("solo", training={"resume_from": str(first.root)}), tmp_path)
    assert states["agent_0"]["policy_version"] == 7 and resumed.env_steps_done == 321


def test_resume_rejects_architecture_mismatch_before_spawning(tmp_path):
    run = tmp_path / "old_run"
    cfg = make_test_config("solo")
    CheckpointManager(run / "checkpoints").save("agent_0", 3, sd(3), meta_extra=signed_meta(cfg))
    with pytest.raises(ConfigError, match="do not match"):
        _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)


def test_resume_rejects_another_role_signature_naming_the_checkpoint(tmp_path):
    run = _old_run(tmp_path, make_test_config("solo"), version=5, env_steps=50, role_signature="other-game")
    with pytest.raises(ConfigError, match="role signature") as exc:
        _resume(make_test_config("solo", training={"resume_from": str(run)}), tmp_path)
    assert "ckpt_v5" in str(exc.value)


def _learner_like_checkpoint_sender(cq, stop_event, periodic: dict | None, final: dict,
                                    final_delay: float = 0.0) -> None:
    """Child process standing in for a learner: an optional periodic snapshot, then the final
    one ``final_delay`` seconds after stop."""
    if periodic is not None:
        send_checkpoint(cq, periodic, block=False)
    stop_event.wait(timeout=60)
    time.sleep(final_delay)
    send_checkpoint(cq, final, block=True)
    cq.close()
    cq.join_thread()


def _launcher_with_queue(tmp_path, cfg, agent_ids):
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = make_coordinator(cfg, tmp_path / "ckpt")
    launcher._agent_ids = list(agent_ids)
    ctx = mp.get_context("spawn")
    queues = {aid: ctx.Queue(maxsize=4) for aid in agent_ids}
    launcher._checkpoint_queues = queues
    launcher._all_queues = list(queues.values())
    return launcher, queues


def _start_sender(launcher, name, *args):
    proc = mp.get_context("spawn").Process(target=_learner_like_checkpoint_sender, args=args, daemon=True)
    proc.start()
    launcher._supervisor.add(name, proc)
    return proc


def _wait_not_empty(q) -> None:
    deadline = time.monotonic() + 60
    while q.empty() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not q.empty()


def test_shutdown_saves_checkpoints_still_queued(tmp_path):
    cfg = make_test_config("solo")
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["agent_0"])
    launcher._checkpoint_meta = {"agent_0": {"config_hash": config_hash(cfg)}}
    launcher._env_step_counter.add(777)
    algo = FakeAlgorithm()
    algo.train_once()
    periodic = make_checkpoint_payload("agent_0", algo)
    algo.train_once()
    final = make_checkpoint_payload("agent_0", algo, final=True)
    proc = _start_sender(launcher, "learner-agent_0", queues["agent_0"], launcher._stop_event, periodic, final)
    _wait_not_empty(queues["agent_0"])

    launcher._shutdown()

    assert proc.exitcode == 0
    infos = launcher._coordinator.checkpoint_manager.list_checkpoints("agent_0")
    assert [c.checkpoint_id for c in infos] == ["ckpt_v1", "ckpt_v2"]
    assert infos[-1].meta["final"] is True and infos[0].meta["final"] is False
    assert infos[-1].meta["env_steps"] == 777 and infos[-1].meta["config_hash"] == config_hash(cfg)
    assert infos[-1].meta["role_signature"] == launcher._coordinator.role_signature("agent_0")


def test_drain_saves_remaining_payloads_after_a_failed_save(tmp_path, caplog):
    cfg = make_test_config("solo")
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    launcher._coordinator = make_coordinator(cfg, tmp_path / "ckpt")
    q = queue.Queue()
    launcher._checkpoint_queues = {"agent_0": q}
    algo = FakeAlgorithm()
    for _ in range(2):
        algo.train_once()
        q.put(make_checkpoint_payload("agent_0", algo))
    real_save = launcher._save_checkpoint

    def save(payload):
        if payload["policy_version"] == 1:
            raise OSError("disk full")
        real_save(payload)

    launcher._save_checkpoint = save
    with pytest.raises(OSError, match="disk full"):
        launcher._drain_all_checkpoints()
    assert [c.checkpoint_id for c in launcher._coordinator.checkpoint_manager.list_checkpoints("agent_0")] == [
        "ckpt_v2"]
    assert "Failed to save checkpoint v1" in caplog.text and "Traceback" in caplog.text


def test_shutdown_tears_children_down_even_if_a_save_fails(tmp_path):
    cfg = make_test_config("solo")
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["agent_0"])
    calls: list[int] = []

    def failing_save(payload):
        calls.append(payload["policy_version"])
        raise OSError("disk full")

    launcher._save_checkpoint = failing_save
    algo = FakeAlgorithm()
    algo.train_once()
    proc = _start_sender(launcher, "learner-agent_0", queues["agent_0"], launcher._stop_event,
                         make_checkpoint_payload("agent_0", algo), make_checkpoint_payload("agent_0", algo, final=True))
    _wait_not_empty(queues["agent_0"])
    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert calls and not proc.is_alive() and proc.exitcode is not None


def test_shutdown_keeps_draining_after_a_failed_save(tmp_path):
    """A save failure does not end the grace window: a second learner's final snapshot,
    delivered later, is still saved; the error propagates after teardown."""
    cfg = make_test_config("solo", agents={"a0": {}, "a1": {}})
    launcher, queues = _launcher_with_queue(tmp_path, cfg, ["a0", "a1"])
    real_save = launcher._save_checkpoint

    def save(payload):
        if payload["agent_id"] == "a0" and not payload["final"]:
            raise OSError("disk full")
        real_save(payload)

    launcher._save_checkpoint = save
    algo = FakeAlgorithm()
    algo.train_once()
    procs = [
        _start_sender(launcher, "learner-a0", queues["a0"], launcher._stop_event,
                      make_checkpoint_payload("a0", algo), make_checkpoint_payload("a0", algo, final=True)),
        _start_sender(launcher, "learner-a1", queues["a1"], launcher._stop_event,
                      None, make_checkpoint_payload("a1", algo, final=True), 1.5),
    ]
    _wait_not_empty(queues["a0"])
    start = time.monotonic()
    with pytest.raises(OSError, match="disk full"):
        launcher._shutdown()
    assert time.monotonic() - start < 10.0
    assert all(not p.is_alive() and p.exitcode == 0 for p in procs)
    mgr = launcher._coordinator.checkpoint_manager
    assert [c.meta["final"] for c in mgr.list_checkpoints("a1")] == [True]
    assert [c.meta["final"] for c in mgr.list_checkpoints("a0")] == [True]
```

(b) Create `tests/unit/test_sp2_run_dir.py` from SP1's run-dir tests (the two distributed-role tests move to T6.4). Save as `/tmp/make_sp2_run_dir_test.py` and run it:

```python
"""T5.4: tests/unit/test_sp2_run_dir.py from SP1's tests/unit/test_run_dir_logging.py."""
import ast
from pathlib import Path

src = Path("tests/unit/test_run_dir_logging.py").read_text()
src = src.replace('"""RunDir layout and per-process logging (T6.2)."""',
                  '"""SP2 RunDir layout and per-process logging (SP1 guarantees on config v2, T5.4)."""')
for old, new in [
    ("from colosseum.core.config import ColosseumConfig, load_config\n",
     "from colosseum.sp2.core.config import ColosseumConfig, load_config\n"),
    ("from colosseum.core.run_dir import RunDir\n", "from colosseum.sp2.core.run_dir import RunDir\n"),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
src = src.replace("    import colosseum.core.run_dir as run_dir_module\n", "    import colosseum.sp2.core.run_dir as run_dir_module\n")
old_cfg = src[src.index('TTT = "examples.tic_tac_toe"'):src.index("    })\n", src.index("def make_config")) + len("    })\n")]
new_cfg = '''def make_config(tmp_path, name=None) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": "game_helpers.TurnTakingGame"},
        "networks": {"model_class": "game_helpers.GameTestModel"},
        "agents": {"alpha": {"algorithm": {"learning_rate": 1e-4}}, "beta": None},
        "run": {"dir": str(tmp_path / "runs"), "name": name},
    })
'''
src = src.replace(old_cfg, new_cfg)
# The distributed-role tests move to tests/unit/test_sp2_distributed_roles.py (T6.4).
lines = src.splitlines(keepends=True)
tree = ast.parse(src)
drop = {"test_distributed_workers_role_includes_the_sanitized_hostname",
        "test_distributed_workers_entry_point_records_the_base_run_name"}
for node in sorted((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in drop),
                   key=lambda n: n.lineno, reverse=True):
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    del lines[start:node.end_lineno]
Path("tests/unit/test_sp2_run_dir.py").write_text("".join(lines))
print("wrote tests/unit/test_sp2_run_dir.py")
```

```bash
.venv/bin/python /tmp/make_sp2_run_dir_test.py
.venv/bin/ruff check --fix tests/unit/test_sp2_run_dir.py && .venv/bin/ruff check tests/unit/test_sp2_run_dir.py
```

(c) Create `tests/integration/test_sp2_game_runs.py` (spec section 6, "Интеграция": asymmetric game with two agents, FFA with layouts 2p/4p, cooperative with `teammates: mixed`, resume continues versions and checks the role signature):

```python
"""`python -m colosseum.sp2 train` on toy games: self-play, asymmetric league, FFA layouts,
cooperative mixed teams, resume with role signatures (T5.4)."""
from __future__ import annotations

import json
from pathlib import Path

from cli_runner import TrainRun, run_train
from game_helpers import write_test_config

SP2 = "colosseum.sp2"


def train(tmp_path: Path, game: str, name: str, overrides: dict[str, str] | None = None, **sections) -> TrainRun:
    config = write_test_config(tmp_path / f"{name}.yaml", game, **sections)
    return run_train(config, tmp_path, name=name, overrides=overrides, module=SP2)


def metas(run: TrainRun, agent_id: str) -> list[dict]:
    agent_dir = run.root / "checkpoints" / agent_id
    found = [json.loads((d / "meta.json").read_text()) for d in agent_dir.glob("ckpt_v*")]
    return sorted(found, key=lambda m: m["policy_version"])


def train_steps(run: TrainRun, agent_id: str) -> list[int]:
    return [r["train_step"] for r in run.records("train") if r["agent"] == agent_id]


def test_self_play_reaches_the_budget_and_checkpoints_carry_roles(tmp_path):
    run = train(tmp_path, "turns", "selfplay")
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert max(train_steps(run, "agent_0")) >= 1
    final = metas(run, "agent_0")[-1]
    assert final["final"] is True and final["roles"] and isinstance(final["role_signature"], str)
    ratings = run.ratings()
    assert set(ratings) == {"env_steps", "layouts"} and ratings["layouts"]
    assert all("layouts" in r for r in run.records("ratings"))
    assert sum(r["episodes"] for r in run.records("episodes")) > 0


def test_asymmetric_two_agent_league_trains_both_roles(tmp_path):
    run = train(tmp_path, "asymmetric", "asymmetric", matchmaking={"mode": "league", "self_play_ratio": 0.5})
    assert run.returncode == 0, run.stderr[-3000:]
    for agent_id in ("hunter", "prey"):
        assert train_steps(run, agent_id), f"{agent_id} never trained"
        assert metas(run, agent_id)[-1]["roles"] == [agent_id]
    assert metas(run, "hunter")[-1]["role_signature"] != metas(run, "prey")[-1]["role_signature"]
    layouts = run.ratings()["layouts"]
    assert any(table["games"]["hunter"]["prey"] > 0 for table in layouts.values()), layouts


def test_ffa_with_two_and_four_player_layouts(tmp_path):
    run = train(tmp_path, "ffa", "ffa", overrides={"training.total_timesteps": "4000"},
                agents={"alpha": {}, "beta": {}},
                matchmaking={"mode": "league", "self_play_ratio": 0.5, "layouts": {"2p": 0.5, "4p": 0.5}})
    assert run.returncode == 0, run.stderr[-3000:]
    played = {layout for r in run.records("episodes") for layout in r["by_layout"]}
    assert {"2p", "4p"} <= played, played
    layouts = run.ratings()["layouts"]
    assert layouts["2p"]["outcome_kind"] == "wdl" and layouts["4p"]["outcome_kind"] == "rank"
    assert layouts["2p"]["games"]["alpha"]["beta"] + layouts["4p"]["games"]["alpha"]["beta"] > 0


def test_cooperative_game_with_mixed_teammates_fills_the_cross_play_table(tmp_path):
    run = train(tmp_path, "coop", "coop", agents={"a": {}, "b": {}},
                matchmaking={"teammates": "mixed", "teammate_self_prob": 0.5})
    assert run.returncode == 0, run.stderr[-3000:]
    (table,) = [t for t in run.ratings()["layouts"].values() if t["outcome_kind"] == "score"]
    assert "a+b" in table["cross_play"], table["cross_play"]
    assert set(table["scores"]) == {"a", "b"}


def test_resume_continues_versions_and_checks_the_role_signature(tmp_path):
    config = write_test_config(tmp_path / "turns.yaml", "turns")
    first = run_train(config, tmp_path, name="first", module=SP2,
                      overrides={"training.total_timesteps": "1500", "checkpoint.interval": "5"})
    assert first.returncode == 0, first.stderr[-3000:]
    final = metas(first, "agent_0")[-1]
    version = final["policy_version"]

    second = run_train(config, tmp_path, name="second", module=SP2,
                       overrides={"training.total_timesteps": "3000", "training.resume_from": str(first.root)})
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"(policy_version {version})" in second.log("main")
    assert train_steps(second, "agent_0")[0] == version + 1
    assert min(m["policy_version"] for m in metas(second, "agent_0")) > version

    ckpt = first.root / "checkpoints" / "agent_0" / f"ckpt_v{version}"
    (ckpt / "meta.json").write_text(json.dumps({**final, "role_signature": "another-game"}))
    third = run_train(config, tmp_path, name="third", module=SP2,
                      overrides={"training.total_timesteps": "3000", "training.resume_from": str(first.root)})
    assert third.returncode == 1
    assert "Config error" in third.stderr and "role signature" in third.stderr and str(ckpt) in third.stderr
    assert "Traceback" not in third.stderr
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_launcher_checkpoints.py tests/unit/test_sp2_run_dir.py tests/integration/test_sp2_game_runs.py -q`
Expected: collection errors `No module named 'colosseum.sp2.launcher'` / `'colosseum.sp2.core.run_dir'`; the integration tests fail with `No module named colosseum.sp2.__main__` in the captured stderr (or `colosseum.sp2.cli`).

- [ ] **Step 4: Copy the run dir**

```bash
cp src/colosseum/core/run_dir.py src/colosseum/sp2/core/run_dir.py
sed -i 's/^from colosseum.core.config import ColosseumConfig, check_path_component$/from colosseum.sp2.core.config import ColosseumConfig, check_path_component/' src/colosseum/sp2/core/run_dir.py
grep -n "^from colosseum.sp2.core.config import ColosseumConfig, check_path_component$" src/colosseum/sp2/core/run_dir.py
.venv/bin/ruff check --fix src/colosseum/sp2/core/run_dir.py && .venv/bin/ruff check src/colosseum/sp2/core/run_dir.py
```

Expected: the `grep` prints one line (the import was rewritten; `ruff --fix` moves it below `colosseum.core.errors`); nothing else in the file changes.

- [ ] **Step 5: Create the `sp2` launcher**

Save as `/tmp/make_sp2_launcher.py` and run it. Every replaced function is shown in full; everything not listed (queue readers and release helpers from T0.1, `_open_metrics`, `_monitor_loop`, `_shutdown`, `_drain_*`, `_finish_metrics`, `_save_checkpoint`, exit-code helpers) is copied unchanged.

```python
"""T5.4: create src/colosseum/sp2/launcher.py from the SP1 launcher (after T0.1).

Run from the repo root: .venv/bin/python <this script>. Functions and methods are replaced by
name (ast line ranges), so the edits do not depend on unrelated T0.1 changes in the file.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path("src/colosseum/launcher.py")
DST = Path("src/colosseum/sp2/launcher.py")

lines = SRC.read_text().splitlines(keepends=True)


def _find(tree: ast.Module, name: str, cls: str | None) -> ast.AST:
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def replace(name: str, new_code: str, cls: str | None = None) -> None:
    """Replace function/method ``name`` (with its decorators) by ``new_code``; '' deletes it."""
    global lines
    tree = ast.parse("".join(lines))
    node = _find(tree, name, cls)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    end = node.end_lineno
    indent = "    " if cls else ""
    code = textwrap.indent(textwrap.dedent(new_code).strip("\n") + "\n", indent) if new_code else ""
    lines = "".join(lines[:start] + [code] + lines[end:]).splitlines(keepends=True)


def replace_text(old: str, new: str) -> None:
    global lines
    text = "".join(lines)
    assert text.count(old) == 1, f"expected exactly one occurrence of: {old!r}"
    lines = text.replace(old, new).splitlines(keepends=True)


# --- module docstring and imports -------------------------------------------------------------
replace_text('"""Launcher: reads config, instantiates all components, starts processes.',
             '"""Launcher (SP2): reads config, instantiates all components, starts processes.')
replace_text("from colosseum.coordinator.coordinator import Coordinator\n",
             "from colosseum.sp2.coordinator.coordinator import Coordinator\n")
replace_text("from colosseum.core.config import ColosseumConfig, load_config\n",
             "from colosseum.sp2.core.config import ColosseumConfig, load_config\n")
replace_text("from colosseum.core.run_dir import RunDir\n", "from colosseum.sp2.core.run_dir import RunDir\n")
replace_text("from colosseum.core.types import LATEST_NETWORK_ID, MatchConfig\n",
             "from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment\n"
             "from colosseum.sp2.envs.game import GameSpec, RoleSpec\n")
# New stdlib imports; ``ruff check --fix`` (next plan step) sorts them and drops unused ones.
replace_text("from __future__ import annotations\n",
             "from __future__ import annotations\n\nfrom collections.abc import Mapping\nfrom dataclasses import dataclass\n")

replace_text("        from colosseum.metrics.hub import MetricsHub\n",
             "        from colosseum.sp2.metrics.hub import MetricsHub\n")
replace_text("        from colosseum.metrics.jsonl import MetricsWriter\n",
             "        from colosseum.sp2.metrics.jsonl import MetricsWriter\n")

# --- top-level helpers ------------------------------------------------------------------------
replace("_create_env", '''
def _create_env(env_class_path: str, kwargs: dict):
    """Create env instance inside a worker process."""
    from colosseum.sp2.core.registry import import_class
    cls = import_class(env_class_path)
    return cls(**kwargs)
''')

replace("trainable_agent_configs", '''
@dataclass(frozen=True)
class RunSetup:
    """What a run needs before any process starts: the game's spec, every trainable agent's
    roles, effective config (overrides merged) and role spec (the spaces its model is built for)."""

    spec: GameSpec
    agent_roles: dict[str, list[str]]
    agent_configs: dict[str, ColosseumConfig]
    role_specs: dict[str, RoleSpec]


def validate_run_config(config: ColosseumConfig) -> None:
    """Fail fast (ConfigError / EnvContractError) before any process starts.

    The GameSpec, the agents' roles, the matchmaking checks and one model per agent
    (T6.2 replaces this body with ``registry.validate_config``).
    """
    from colosseum.sp2.coordinator.matchmaker import validate_matchmaking
    from colosseum.sp2.core import registry
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    spec = registry.env_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    validate_matchmaking(spec, agent_roles, config.matchmaking)
    for aid, roles in agent_roles.items():
        registry.build_model(config.get_agent_config(aid), agent_role_spec(spec, roles))


def setup_run(config: ColosseumConfig, validate: bool = True) -> RunSetup:
    """``RunSetup`` of ``config``; ``validate`` first runs ``validate_run_config``."""
    from colosseum.sp2.core.registry import env_spec
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    if validate:
        validate_run_config(config)
    spec = env_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    agent_ids = config.get_trainable_agent_ids()
    return RunSetup(
        spec=spec,
        agent_roles={aid: list(agent_roles[aid]) for aid in agent_ids},
        agent_configs={aid: config.get_agent_config(aid) for aid in agent_ids},
        role_specs={aid: agent_role_spec(spec, agent_roles[aid]) for aid in agent_ids},
    )
''')
replace("validate_agent_configs", "")
replace("_create_model", "")

replace("_worker_main", '''
def _worker_main(
    *,
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_roles: dict[str, list[str]],
    agent_configs: dict[str, ColosseumConfig],
    role_specs: dict[str, RoleSpec],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    lineups: list[Lineup],
    env_step_counter: SharedCounter | None = None,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
    results_queue: mp.Queue | None = None,
    command_queue: mp.Queue | None = None,
    metrics_queue: mp.Queue | None = None,
) -> None:
    """Worker process body (see ``_worker_target``).

    Env and model factories are built INSIDE the process (``functools.partial`` over top-level
    functions is picklable under spawn, which the subprocess vector env needs to ship ``env_fn``
    to its children). Models are built for each agent's role spec.
    """
    from functools import partial

    from colosseum.sp2.core.registry import build_model
    from colosseum.sp2.worker.rollout_worker import rollout_worker_process

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        agent_roles=agent_roles,
        model_factories={aid: partial(build_model, agent_configs[aid], role_specs[aid]) for aid in agent_ids},
        trajectory_queues=trajectory_queues,
        weight_queues=weight_queues,
        stop_event=stop_event,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        env_step_counter=env_step_counter,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        lineups=lineups,
        results_queue=results_queue,
        command_queue=command_queue,
        stats_queue=metrics_queue,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
        max_idle_steps=config.env.max_idle_steps,
    )
''')

replace("_learner_main", '''
def _learner_main(
    *,
    agent_id: str,
    config: ColosseumConfig,
    role_spec: RoleSpec,
    trajectory_queue: mp.Queue,
    weight_queues: list[mp.Queue],
    stop_event: mp.Event,
    metrics_queue: mp.Queue,
    checkpoint_queue: mp.Queue | None = None,
    checkpoint_interval: int = 0,
    resume_state: dict | None = None,
    progress_counter: SharedCounter | None = None,
    total_timesteps: int = 0,
    num_learners: int = 1,
    seed: int | None = None,
) -> None:
    """Learner process body (see ``_learner_target``).

    ``num_learners`` (learner processes on this machine) feeds the automatic torch thread count
    when ``learner.torch_threads`` is unset. ``seed`` (from ``utils.seeding.learner_seed``) seeds
    this process before the model is built. The model (and a kickstart teacher) is built for
    ``role_spec``, the spaces of the agent's roles.
    """
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.sp2.core.registry import build_model, import_class
    from colosseum.sp2.core.specs import ActionSpec
    from colosseum.sp2.learner.learner import learner_process, resolve_device
    from colosseum.utils.seeding import apply_global_seed

    apply_global_seed(seed)

    device = resolve_device(config.learner.device)
    configure_torch_threads(resolve_learner_threads(
        config.learner.torch_threads, device, config.rollout.num_workers,
        config.rollout.torch_threads, num_learners,
    ))

    algo_class_path = config.algorithm.algorithm_class
    if not algo_class_path:
        raise ValueError(
            "algorithm.algorithm_class is not set. "
            "Provide a dotted import path (e.g. 'colosseum.sp2.algorithms.appo.APPO')."
        )
    algo_cls = import_class(algo_class_path)
    action_spec = ActionSpec.from_space(role_spec.action_space)
    teacher_path = config.training.kickstart_teacher

    def algorithm_factory():
        model = build_model(config, role_spec)
        kickstart = None
        if teacher_path:
            from colosseum.sp2.bc.kickstart import KickstartLoss
            teacher = build_model(config, role_spec)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True, map_location=device))
            teacher.to(device)
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=config.training.kickstart_lambda,
                decay_steps=config.training.kickstart_decay_steps,
                direction=config.training.kickstart_kl,
            )
            logger.info(f"Kickstart enabled for {agent_id} from {teacher_path}")
        kwargs = {"device": device, "pin_memory": config.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(model, config.algorithm, action_spec, **kwargs)

    learner_process(
        agent_id=agent_id,
        algorithm_factory=algorithm_factory,
        trajectory_queue=trajectory_queue,
        weight_queues=weight_queues,
        config=config.learner,
        stop_event=stop_event,
        metrics_queue=metrics_queue,
        checkpoint_queue=checkpoint_queue,
        checkpoint_interval=checkpoint_interval,
        resume_state=resume_state,
        progress_counter=progress_counter,
        total_timesteps=total_timesteps,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
    )
''')

replace("_derive_worker_configs", '''
def _resolve_lineups(
    lineups: list[Lineup],
    coordinator: Coordinator,
    agent_ids: list[str],
    already_sent: dict[str, set[str]] | None = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[Lineup]]:
    """Load the checkpoints the lineups need; returns ``(new checkpoints by agent, lineups)``.

    Checkpoint weights cross the process boundary (worker Process arguments and
    ``WorkerCommand.new_checkpoints``), so they are numpy, never torch. Checkpoints listed in
    ``already_sent[agent]`` are referenced but not reloaded. A checkpoint that cannot be loaded
    (evicted or missing) is replaced by the latest weights with ``collect=True`` and a warning
    (SP1 rule, R4-19).
    """
    already_sent = already_sent or {}
    new_ckpts: dict[str, dict[str, dict[str, np.ndarray]]] = {aid: {} for aid in agent_ids}
    missing: set[tuple[str, str]] = set()
    resolved: list[Lineup] = []
    for lineup in lineups:
        seats: list[SeatAssignment] = []
        for seat in lineup.seats:
            if seat.network_id == LATEST_NETWORK_ID:
                seats.append(SeatAssignment(seat.agent_id, LATEST_NETWORK_ID, seat.collect))
                continue
            agent_new = new_ckpts.setdefault(seat.agent_id, {})
            ckpt_id = seat.network_id
            available = ckpt_id in already_sent.get(seat.agent_id, set()) or ckpt_id in agent_new
            if not available and (seat.agent_id, ckpt_id) not in missing:
                try:
                    agent_new[ckpt_id] = coordinator.checkpoint_manager.load_model(seat.agent_id, ckpt_id)
                    available = True
                except FileNotFoundError:
                    missing.add((seat.agent_id, ckpt_id))
                    logger.warning(f"Checkpoint {ckpt_id} of {seat.agent_id} is missing; that seat plays "
                                   f"the latest weights and collects trajectories")
            if available:
                seats.append(SeatAssignment(seat.agent_id, ckpt_id, seat.collect))
            else:
                seats.append(SeatAssignment(seat.agent_id, LATEST_NETWORK_ID, True))
        resolved.append(Lineup(layout=lineup.layout, seats=seats))
    return new_ckpts, resolved
''')

# --- Launcher methods ---------------------------------------------------------------------------
replace_text("        self._coordinator: Coordinator | None = None\n",
             "        self._coordinator: Coordinator | None = None\n"
             "        self._setup: RunSetup | None = None\n")

replace("launch", '''
def launch(self) -> int:
    """Run the full training pipeline; returns the process exit code.

    0: the env-step budget was reached and every child stopped cleanly; 1: a child
    died (or exited non-zero on its own); 128 + signum after SIGINT / SIGTERM.
    """
    cfg = self._config
    trainable_agents = cfg.get_trainable_agent_ids()

    logger.info("Colosseum: Starting training pipeline")
    logger.info(f"  Agents: {trainable_agents}")
    logger.info(f"  Algorithm: {cfg.algorithm.name}")
    logger.info(f"  Workers: {cfg.rollout.num_workers}")
    logger.info(f"  Envs per worker: {cfg.rollout.envs_per_worker}")
    logger.info(f"  Chunk length: {cfg.rollout.chunk_length}")

    # Spec, roles and per-agent effective configs, validated before any process starts
    # (run_training validates before the run dir exists and passes validated=True).
    setup = setup_run(cfg, validate=not self._validated)
    self._setup = setup
    logger.info(f"  Roles: {setup.agent_roles}; layouts: {sorted(setup.spec.layouts)}")
    warn_static_ownership_skew(cfg)

    coordinator = Coordinator(cfg, setup.spec, setup.agent_roles, checkpoint_dir=self._run_dir.checkpoints)

    # Resume (before any process starts: a bad resume source fails fast).
    resume_states = self._resolve_resume(setup.agent_configs, setup.role_specs)

    from colosseum.sp2.core.config import config_hash
    cfg_hash = config_hash(cfg)
    self._checkpoint_meta = {
        aid: {"networks": setup.agent_configs[aid].networks.model_dump(mode="json", by_alias=True),
              "config_hash": cfg_hash}
        for aid in trainable_agents
    }

    trajectory_queues: dict[str, mp.Queue] = {}
    checkpoint_queues: dict[str, mp.Queue] = {}
    weight_queues_per_agent: dict[str, list[mp.Queue]] = {}
    for aid in trainable_agents:
        acfg = setup.agent_configs[aid]
        trajectory_queues[aid] = mp.Queue(maxsize=acfg.learner.queue_size)
        checkpoint_queues[aid] = mp.Queue(maxsize=_CHECKPOINT_QUEUE_SIZE)
        weight_queues_per_agent[aid] = [
            mp.Queue(maxsize=_WEIGHT_QUEUE_SIZE) for _ in range(cfg.rollout.num_workers)
        ]
    metrics_queue = mp.Queue(maxsize=_METRICS_QUEUE_SIZE)
    results_queue = mp.Queue(maxsize=_RESULTS_QUEUE_SIZE)
    # Per-worker command queues (runtime lineup re-assignment) and the set of checkpoint ids
    # already shipped to each worker (to send deltas).
    command_queues: list[mp.Queue] = [
        mp.Queue(maxsize=_COMMAND_QUEUE_SIZE) for _ in range(cfg.rollout.num_workers)
    ]
    worker_sent_ckpts: list[dict[str, set]] = [
        {aid: set() for aid in trainable_agents} for _ in range(cfg.rollout.num_workers)
    ]
    # Track every queue so shutdown can cancel feeder threads and avoid the interpreter
    # blocking on unflushed data when consumers have exited.
    self._all_queues = [metrics_queue, results_queue, *command_queues]
    for aid in trainable_agents:
        self._all_queues.append(trajectory_queues[aid])
        self._all_queues.append(checkpoint_queues[aid])
        self._all_queues.extend(weight_queues_per_agent[aid])

    self._coordinator = coordinator
    self._agent_ids = list(trainable_agents)
    self._checkpoint_queues = checkpoint_queues
    self._trajectory_queues = trajectory_queues
    self._command_queues = command_queues
    self._results_queue = results_queue
    self._metrics_queue = metrics_queue

    code = 1
    try:
        # SIGINT / SIGTERM only set stop_event from here on: children ignore SIGINT and the
        # shutdown below saves their final checkpoints.
        self._supervisor.install_signal_handlers()
        # The metrics hub (and its file) exists before any child starts: an open failure here
        # cannot orphan children. _finish_metrics closes whatever was opened.
        self._open_metrics(resume_states)
        started = False
        try:
            self._start_children(
                setup.agent_configs, resume_states, trajectory_queues, checkpoint_queues,
                weight_queues_per_agent, metrics_queue, results_queue, command_queues,
                worker_sent_ckpts,
            )
            started = True
            logger.info("Training started. Press Ctrl+C to stop.")
            code = self._monitor_loop(coordinator, trainable_agents, command_queues, worker_sent_ckpts)
        except BaseException as error:
            # Also after a failed start or a monitor error the started children are stopped
            # (final checkpoints saved); then the original error propagates.
            if not started:
                logger.error(f"Starting the child processes failed; stopping the "
                             f"{len(self._supervisor.names)} already started")
            self._shutdown_after_error(error)
            raise
        killed = self._shutdown()
        code = self._exit_code_after_shutdown(code, killed)
    finally:
        try:
            self._finish_metrics()
        finally:
            try:
                self._supervisor.restore_signal_handlers()
            finally:
                self._release_queues()
    return code
''', cls="Launcher")

replace("_start_children", '''
def _start_children(
    self,
    agent_configs: dict[str, ColosseumConfig],
    resume_states: dict[str, dict | None],
    trajectory_queues: dict[str, mp.Queue],
    checkpoint_queues: dict[str, mp.Queue],
    weight_queues_per_agent: dict[str, list[mp.Queue]],
    metrics_queue: mp.Queue,
    results_queue: mp.Queue,
    command_queues: list[mp.Queue],
    worker_sent_ckpts: list[dict[str, set]],
) -> None:
    """Start one learner per trainable agent, then the workers shared by all agents.

    Every started child is registered with the supervisor right away, so a failure part-way
    through tears down exactly the children that exist.
    """
    from colosseum.utils.seeding import learner_seed

    cfg = self._config
    setup = self._setup
    trainable_agents = self._agent_ids
    coordinator = self._coordinator
    for agent_index, aid in enumerate(trainable_agents):
        # numpy + bytes only: Process arguments cross the process boundary.
        learner_proc = mp.Process(
            target=_learner_target,
            name=f"learner-{aid}",
            kwargs=dict(
                agent_id=aid,
                log_dir=str(self._run_dir.logs),
                config=agent_configs[aid],
                role_spec=setup.role_specs[aid],
                trajectory_queue=trajectory_queues[aid],
                weight_queues=weight_queues_per_agent[aid],
                stop_event=self._stop_event,
                metrics_queue=metrics_queue,
                checkpoint_queue=checkpoint_queues[aid],
                checkpoint_interval=cfg.checkpoint.interval,
                resume_state=resume_states[aid],
                progress_counter=self._env_step_counter,
                total_timesteps=cfg.training.total_timesteps,
                num_learners=len(trainable_agents),
                seed=learner_seed(cfg.training.seed, agent_index),
            ),
            daemon=True,
        )
        # SIGINT stays blocked in the child from its first instruction until
        # init_child_process (run_child) ignores it and unblocks it.
        start_process(learner_proc)
        self._supervisor.add(f"learner-{aid}", learner_proc)
        logger.info(f"Learner started for agent {aid}")

    for worker_id in range(cfg.rollout.num_workers):
        lineups = coordinator.generate_lineups(
            cfg.rollout.envs_per_worker, env_offset=worker_id * cfg.rollout.envs_per_worker,
        )
        ckpt_dicts_by_agent, lineups = _resolve_lineups(lineups, coordinator, trainable_agents)
        # Initial checkpoints travel as process arguments: delivered by construction.
        for aid, ckpts in ckpt_dicts_by_agent.items():
            worker_sent_ckpts[worker_id].setdefault(aid, set()).update(ckpts)
        worker_weight_queues = {aid: weight_queues_per_agent[aid][worker_id] for aid in trainable_agents}
        # A subprocess vector env makes the worker spawn its own children, which daemonic
        # processes may not do, so such workers run non-daemon (the shutdown joins them).
        worker_daemon = cfg.rollout.vec_env != "subprocess"
        worker_proc = mp.Process(
            target=_worker_target,
            name=f"worker-{worker_id}",
            kwargs=dict(
                worker_id=worker_id,
                log_dir=str(self._run_dir.logs),
                config=cfg,
                agent_ids=trainable_agents,
                agent_roles=setup.agent_roles,
                agent_configs=agent_configs,
                role_specs=setup.role_specs,
                trajectory_queues=trajectory_queues,
                weight_queues=worker_weight_queues,
                stop_event=self._stop_event,
                lineups=lineups,
                env_step_counter=self._env_step_counter,
                checkpoint_state_dicts_by_agent=ckpt_dicts_by_agent,
                results_queue=results_queue,
                command_queue=command_queues[worker_id],
                metrics_queue=metrics_queue,
            ),
            daemon=worker_daemon,
        )
        start_process(worker_proc)
        self._supervisor.add(f"worker-{worker_id}", worker_proc)
''', cls="Launcher")

replace("_resolve_resume", '''
def _resolve_resume(self, agent_configs: Mapping[str, ColosseumConfig],
                    role_specs: Mapping[str, RoleSpec]) -> dict[str, dict | None]:
    """Resolve ``training.resume_from`` for every trainable agent (see ``resolve_resume``).

    A checkpoint source must carry the role signature of the agent's roles (ConfigError naming
    the checkpoint otherwise); every resumed agent's weights are checked against its
    architecture (ConfigError before any process starts); the global env-step counter
    continues from the largest resumed ``env_steps``, so the budget and LR progress continue.
    """
    from colosseum.sp2.coordinator.checkpoint_manager import check_model_state, resolve_resume
    from colosseum.sp2.core.registry import build_model
    from colosseum.sp2.core.roles import role_signature

    resume_from = self._config.training.resume_from
    resume_states: dict[str, dict | None] = {aid: None for aid in agent_configs}
    if not resume_from:
        return resume_states
    for aid, acfg in agent_configs.items():
        state = resolve_resume(resume_from, aid, expected_signature=role_signature(role_specs[aid]))
        if state is not None:
            check_model_state(build_model(acfg, role_specs[aid]), state["model_state"], state["source"])
            logger.info(f"Resume [{aid}]: {state['source']} (policy_version {state['policy_version']})")
        resume_states[aid] = state
    start_env_steps = max((s["env_steps"] for s in resume_states.values() if s), default=0)
    if start_env_steps > 0:
        self._env_step_counter.add(start_env_steps)
        logger.info(f"Resume: env-step counter continues from {start_env_steps}")
    return resume_states
''', cls="Launcher")

replace("_refresh_worker_matches", '''
def _refresh_worker_matches(
    self,
    coordinator: Coordinator,
    agent_ids: list[str],
    command_queues: list[mp.Queue],
    worker_sent_ckpts: list[dict[str, set]],
) -> None:
    """Advance the owner rotation and send every worker fresh lineups.

    Lineups for worker ``w`` are generated at global env offset ``w * envs_per_worker``. Each
    command carries only checkpoints that the worker does not have yet. A checkpoint counts as
    delivered only after its command was put successfully; a full command queue means the
    worker skips this round and gets the deltas with the next refresh.
    """
    from colosseum.sp2.core.types import WorkerCommand

    num_envs = self._config.rollout.envs_per_worker
    coordinator.next_round()
    for worker_id, cq in enumerate(command_queues):
        sent = worker_sent_ckpts[worker_id]
        lineups = coordinator.generate_lineups(num_envs, env_offset=worker_id * num_envs)
        new_ckpts, lineups = _resolve_lineups(lineups, coordinator, agent_ids, already_sent=sent)
        cmd = WorkerCommand(lineups=list(lineups), new_checkpoints={aid: c for aid, c in new_ckpts.items() if c})
        try:
            cq.put_nowait(cmd)
        except queue.Full:
            logger.debug(f"worker-{worker_id} has not consumed its previous command; skipping this refresh")
            continue
        for aid, ckpts in new_ckpts.items():
            sent.setdefault(aid, set()).update(ckpts)
''', cls="Launcher")

replace("run_training", '''
def run_training(config_path: str, overrides: dict | None = None) -> int:
    """Entry point of ``python -m colosseum.sp2 train``. Returns the process exit code.

    The run directory is printed (and logged) once it exists.
    """
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    # Before the run dir exists: an invalid config leaves nothing behind, so a corrected retry
    # with the same run.name works. Launcher.launch does not validate again.
    validate_run_config(config)
    if config.training.resume_from:
        from colosseum.sp2.coordinator.checkpoint_manager import classify_resume_source
        classify_resume_source(config.training.resume_from)  # exists and has a known layout (nothing loaded)
    apply_global_seed(config.training.seed)  # after overrides: --set training.seed works
    run_dir = RunDir.create(config, config_path)
    config = run_dir.with_run_name(config)  # the resolved config and checkpoint hashes agree
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    code = Launcher(config, run_dir, validated=True).launch()
    if code == 0:
        logger.info(f"Training finished; outputs in {run_dir.root}")
    return code
''')

DST.write_text("".join(lines))
print(f"wrote {DST}")
```

```bash
.venv/bin/python /tmp/make_sp2_launcher.py
.venv/bin/ruff check --fix src/colosseum/sp2/launcher.py && .venv/bin/ruff check src/colosseum/sp2/launcher.py
grep -n "colosseum\.\(core\.\(config\|types\|registry\|run_dir\)\|coordinator\|metrics\.\(hub\|jsonl\)\|worker\|learner\|algorithms\|bc\)" src/colosseum/sp2/launcher.py | grep -v "colosseum\.sp2\." || echo "no SP1 imports of replaced modules"
```

Expected: `no SP1 imports of replaced modules`.

- [ ] **Step 6: Create the CLI**

Create `src/colosseum/sp2/cli.py` (SP1's group, interrupt handling and config-error handling; `train` only — T6.1–T6.4 add `eval`, `validate`, `bc` and the distributed commands above the final `if __name__` line):

```python
"""CLI entry point (``python -m colosseum.sp2``; the ``colosseum`` console script after the switch)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import click

# Test hook: when set, the CLI touches this file when ``_interrupts`` becomes active, i.e. from
# then on a Ctrl-C is handled by it (lifecycle tests send SIGINT during startup only after the
# file exists).
_STARTUP_MARKER_ENV = "COLOSSEUM_TEST_STARTUP_MARKER"


@contextmanager
def _interrupts() -> Iterator[None]:
    """Ctrl-C before the run installs its own signal handling (imports, config loading,
    validation): one line on stderr and exit code 130, as after a handled SIGINT (click would
    turn it into "Aborted!" with exit code 1). Wraps every command (``_InterruptibleGroup``)."""
    try:
        marker = os.environ.get(_STARTUP_MARKER_ENV)
        if marker:
            open(marker, "a").close()
        yield
    except KeyboardInterrupt:
        click.echo("Interrupted", err=True)
        sys.exit(130)


class _InterruptibleGroup(click.Group):
    """Runs the group callback and the chosen command inside ``_interrupts``."""

    def invoke(self, ctx: click.Context):
        with _interrupts():
            return super().invoke(ctx)


@click.group(cls=_InterruptibleGroup)
def main() -> None:
    """Colosseum — Distributed RL Training Framework."""
    # User code (e.g. ``examples.*`` or ``my_game.*``) is imported relative to the cwd.
    # Spawned children inherit sys.path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


@contextmanager
def _config_errors() -> Iterator[None]:
    """A config (or env contract, or input data) problem found at startup: one line on stderr,
    exit code 1, no traceback."""
    from colosseum.core.errors import ConfigError, EnvContractError

    try:
        yield
    except (ConfigError, EnvContractError) as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)


_SET_HELP_YAML = (
    " Values are YAML scalars/lists (null, true, 1e-4, [1, 2]); quote a value to force a string, "
    "e.g. --set run.name='\"123\"'."
)


def _parse_overrides(overrides: tuple[str, ...]) -> dict:
    """Parse ``--set key=value`` pairs; values use YAML semantics (null, numbers, lists)."""
    from colosseum.sp2.core.config import parse_override_value

    result = {}
    for ov in overrides:
        if "=" not in ov:
            raise click.BadParameter(f"Override must be key=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        key = key.strip()
        result[key] = parse_override_value(value, key=key)
    return result


def _parse_agent_spec(spec: str) -> tuple[str, str]:
    """``name=path`` -> (name, path)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise click.BadParameter(f"expected name=path, got {spec!r}", param_hint="'--agent'")
    return name, path


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def train(config: str, overrides: tuple[str, ...]) -> None:
    """Train the config's agents (single machine).

    Exit code: 0 budget reached, 1 config error or a child process died, 130 SIGINT, 143 SIGTERM.
    """
    with _config_errors():
        from colosseum.sp2.launcher import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


if __name__ == "__main__":
    main()
```

Create (or overwrite) `src/colosseum/sp2/__main__.py`:

```python
"""Enables ``python -m colosseum.sp2 ...``."""

from colosseum.sp2.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_launcher_checkpoints.py tests/unit/test_sp2_run_dir.py tests/integration/test_sp2_game_runs.py -q`
Expected: all passed (the integration file takes about a minute: five short CLI runs plus one failing resume).

- [ ] **Step 8: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed (SP1 integration tests unaffected: `module` defaults to `"colosseum"`), no warnings; `All checks passed!`.

- [ ] **Step 9: Commit**

```bash
git add src/colosseum/sp2/core/run_dir.py src/colosseum/sp2/launcher.py src/colosseum/sp2/cli.py src/colosseum/sp2/__main__.py \
        tests/cli_runner.py tests/game_helpers.py tests/unit/test_sp2_launcher_checkpoints.py tests/unit/test_sp2_run_dir.py \
        tests/integration/test_sp2_game_runs.py
git commit -m "feat: SP2 launcher, run dir and CLI train on lineups and roles (SP2 T5.4)"
git push origin sp2-game-model
```

---

### Task T6.1: Eval engine on `MatchRunner`, reports, CLI `eval`

Spec block 8. The in-process API (`play_lineups`) takes explicit `PolicyModel` instances and explicit lineups and plays them on the training core (`MatchRunner`), so the learning tests' "random legal player" and SP4's tournament need no scripted players. The CLI schedules lineups by the spec's rules (pairs rotate teams; asymmetric pairs do not rotate; one agent plays every team; one-team layouts get homogeneous teams and cross-play) and reports per layout by outcome kind. Checkpoints are read with `load_checkpoint_dir` (read-only, SP1 ruling) and must carry a matching role signature.

**Files:**
- Create: `src/colosseum/sp2/eval.py`
- Modify: `src/colosseum/sp2/cli.py` (add the `eval` command)
- Test: `tests/unit/test_sp2_eval_schedule.py`, `tests/unit/test_sp2_eval_report.py`, `tests/contract/test_sp2_eval_engine.py`, `tests/integration/test_sp2_eval_cli.py`

**Interfaces:**
- Consumes: `MatchRunner(*, vec_env, lineups, models, observer, seed, max_idle_steps, deterministic, context, match_id_prefix)`, `MatchRunner.step()/set_next_lineup()/close()`, `EpisodeEnd` (T3.3); `VectorEnv(env_fn, num_envs)` (T1.6); `GameSpec.teams/num_teams/layout_size/outcome_kind` (T1.4); `pairwise_rank_score` (T1.4); `composition_key` (T5.2); `load_checkpoint_dir`, `read_weights_file`, `check_model_state` (T5.3); `env_spec`, `make_env`, `build_model` (T2.4); `validate_config` (T6.2, executed before this task); `role_signature`, `agent_role_spec`, `resolve_agent_roles` (T2.4); `make_test_model` (T2.3).
- Produces (contract, plus additions marked *):
  - `schedule_lineups(spec, layout, players, num_matches) -> list[Lineup]`, `play_lineups(*, env_fn, models, lineups, num_envs=8, seed=None, deterministic=False, max_idle_steps=1000) -> list[MatchResult]`, `summarize(spec, results, *, *agents=(), *num_matches=0, *deterministic=False) -> EvalReport`, `EvalReport.to_dict()`, `EvalReport.text()`, *`EvalReport.write_json(path)`, `evaluate(config, agents, *, layouts, num_matches, seed=None, deterministic=False, *num_envs=8)`, `load_eval_model(config, name, path, *, *spec=None) -> (model, roles)`, `wilson_interval`, `normal_interval`, *`default_layouts(spec, players)`, *`PairStats`.
  - Report JSON: `{"agents", "num_matches", "deterministic", "ci_level", "layouts": {layout: {"outcome_kind", "n", "by_role", ...}}}` with `pairs`/`solo`/`unattributed`/`score_ci_method` (wdl), `agents`/`higher` (rank), `compositions` (score).
  - CLI: `python -m colosseum.sp2 eval -c cfg -a NAME=PATH [-a ...] [--layout L ...] [-n N] [--num-envs E] [--deterministic] [--seed S] [-o out.json]`.

- [ ] **Step 1: Write the failing unit tests (schedules and reports)**

Create `tests/unit/test_sp2_eval_schedule.py`:

```python
"""Eval schedules by the CLI rules of spec block 8 (T6.1)."""
from __future__ import annotations

from collections import Counter

import gymnasium
import numpy as np
import pytest

from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.sp2.eval import default_layouts, schedule_lineups

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))


def agents(lineup) -> list[str]:
    return [seat.agent_id for seat in lineup.seats]


def test_pairs_rotate_teams_in_a_two_player_game():
    spec = GameSpec.symmetric(2, OBS, ACT)
    lineups = schedule_lineups(spec, "2p", {"a": ["player"], "b": ["player"]}, 4)
    assert [agents(lu) for lu in lineups] == [["a", "b"], ["b", "a"]] * 2
    assert all(not s.collect and s.network_id == "latest" for lu in lineups for s in lu.seats)


def test_pairs_alternate_over_ffa_teams_and_cover_every_pair():
    spec = GameSpec.symmetric(4, OBS, ACT)
    lineups = schedule_lineups(spec, "4p", {"a": ["player"], "b": ["player"], "c": ["player"]}, 2)
    assert len(lineups) == 3 * 2
    assert agents(lineups[0]) == ["a", "b", "a", "b"] and agents(lineups[1]) == ["b", "a", "b", "a"]
    assert {tuple(sorted(set(agents(lu)))) for lu in lineups} == {("a", "b"), ("a", "c"), ("b", "c")}


def test_teams_are_filled_homogeneously():
    spec = GameSpec.teams_of([2, 2], OBS, ACT)
    lineups = schedule_lineups(spec, "2v2", {"a": ["player"], "b": ["player"]}, 2)
    assert [agents(lu) for lu in lineups] == [["a", "a", "b", "b"], ["b", "b", "a", "a"]]


def test_asymmetric_pair_is_not_rotated():
    spec = GameSpec(roles={"hunter": HUNTER, "prey": PREY},
                    layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
    lineups = schedule_lineups(spec, "1v2", {"h": ["hunter"], "p": ["prey"]}, 3)
    assert [agents(lu) for lu in lineups] == [["h", "p", "p"]] * 3
    assert schedule_lineups(spec, "1v2", {"h": ["hunter"], "h2": ["hunter"]}, 2) == []  # nobody plays prey


def test_one_agent_in_a_competitive_layout_plays_every_team():
    spec = GameSpec.symmetric(3, OBS, ACT)
    lineups = schedule_lineups(spec, "3p", {"a": ["player"]}, 5)
    assert len(lineups) == 5 and all(agents(lu) == ["a"] * 3 for lu in lineups)


def test_one_team_gives_homogeneous_teams_then_cross_play():
    spec = GameSpec.teams_of([2], OBS, ACT)
    lineups = schedule_lineups(spec, "coop2", {"a": ["player"], "b": ["player"]}, 4)
    counts = Counter(tuple(agents(lu)) for lu in lineups)
    assert counts == {("a", "a"): 4, ("b", "b"): 4, ("a", "b"): 2, ("b", "a"): 2}
    solo = GameSpec.solo(OBS, ACT)
    assert [agents(lu) for lu in schedule_lineups(solo, "solo", {"a": ["player"], "b": ["player"]}, 2)] == [
        ["a"], ["a"], ["b"], ["b"]]


def test_cross_play_respects_roles():
    spec = GameSpec(roles={"hunter": HUNTER, "prey": PREY},
                    layouts={"duo": (SeatSpec("hunter", 0), SeatSpec("prey", 0))})
    lineups = schedule_lineups(spec, "duo", {"h": ["hunter"], "p": ["prey"]}, 3)
    assert [agents(lu) for lu in lineups] == [["h", "p"]] * 3  # no homogeneous team is possible


def test_default_layouts_and_errors():
    spec = GameSpec(roles={"player": RoleSpec(OBS, ACT), "hunter": HUNTER, "prey": PREY},
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                             "1v1": (SeatSpec("hunter", 0), SeatSpec("prey", 1))})
    assert default_layouts(spec, {"a": ["player"], "b": ["player"]}) == ["2p"]
    assert default_layouts(spec, {"h": ["hunter"], "p": ["prey"]}) == ["1v1"]
    with pytest.raises(ValueError, match="unknown layout"):
        schedule_lineups(spec, "9p", {"a": ["player"]}, 1)
    with pytest.raises(ValueError, match="num_matches"):
        schedule_lineups(spec, "2p", {"a": ["player"]}, 0)
```

Create `tests/unit/test_sp2_eval_report.py`:

```python
"""Eval reports per layout and outcome kind, with SP1's interval math (T6.1)."""
from __future__ import annotations

import json
import math

import gymnasium
import numpy as np
import pytest

from colosseum.sp2.core.types import MatchResult, SeatResult, TeamResult
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.sp2.eval import normal_interval, summarize, wilson_interval

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
SPEC = GameSpec(
    roles={"player": RoleSpec(OBS, ACT), "hunter": RoleSpec(OBS, gymnasium.spaces.Discrete(5))},
    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
             "4p": tuple(SeatSpec("player", i) for i in range(4)),
             "coop2": (SeatSpec("player", 0), SeatSpec("player", 0)),
             "hunt": (SeatSpec("hunter", 0), SeatSpec("player", 1))},
)


def match(layout, names, ranks, rewards=None, scores=None, roles=None, length=6) -> MatchResult:
    teams_of = [s.team for s in SPEC.layouts[layout]]
    seats = [SeatResult(seat=i, role=(roles or [s.role for s in SPEC.layouts[layout]])[i], team=teams_of[i],
                        agent_id=name, network_id="latest", reward=float((rewards or [0.0] * len(names))[i]))
             for i, name in enumerate(names)]
    teams = [TeamResult(team=t, rank=float(r), score=float((scores or {}).get(t, 0.0))) for t, r in ranks.items()]
    kind = SPEC.outcome_kind(layout)
    return MatchResult(match_id="m", layout=layout, outcome_kind=kind, seats=seats, teams=teams,
                       episode_length=length)


def test_wilson_and_normal_intervals_match_sp1():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high and 0.0 <= low and high <= 1.0
    assert wilson_interval(10, 10)[1] == 1.0
    mean, low, high = normal_interval([1.0, 3.0])
    assert mean == 2.0 and high - mean == pytest.approx(1.959963984540054 * math.sqrt(2.0) / math.sqrt(2))
    assert normal_interval([5.0]) == (5.0, 5.0, 5.0) and normal_interval([]) == (0.0, 0.0, 0.0)


def test_wdl_pairs_with_sides_and_reversed_rows():
    results = [
        match("2p", ["a", "b"], {0: 1, 1: 2}, rewards=[1, -1]),
        match("2p", ["b", "a"], {0: 1, 1: 1}),
        match("2p", ["b", "a"], {0: 2, 1: 1}, rewards=[-1, 1]),
    ]
    report = summarize(SPEC, results, agents=["a", "b"], num_matches=3).to_dict()
    section = report["layouts"]["2p"]
    assert section["outcome_kind"] == "wdl" and section["n"] == 3 and section["unattributed"] == 0
    a_row, b_row = section["pairs"]
    assert (a_row["agent_a"], a_row["wins"], a_row["draws"], a_row["losses"]) == ("a", 2, 1, 0)
    assert (b_row["agent_a"], b_row["wins"], b_row["losses"]) == ("b", 0, 2)
    assert a_row["score"] == pytest.approx(2.5 / 3) and a_row["score_ci"] == list(wilson_interval(2.5, 3))
    assert a_row["per_side"] == {"0": {"n": 1, "wins": 1, "draws": 0, "losses": 0, "score": 1.0},
                                 "1": {"n": 2, "wins": 1, "draws": 1, "losses": 0, "score": 0.75}}
    assert b_row["per_side"]["0"]["n"] == 2
    assert a_row["mean_return_a"] == pytest.approx(2 / 3)
    assert section["by_role"]["player"]["a"]["n"] == 3


def test_one_agent_in_a_wdl_layout_reports_per_seat():
    results = [match("2p", ["a", "a"], {0: 1, 1: 2}, rewards=[1, -1]), match("2p", ["a", "a"], {0: 2, 1: 1})]
    section = summarize(SPEC, results).to_dict()["layouts"]["2p"]
    assert section["pairs"] == []
    (row,) = section["solo"]
    assert row["agent"] == "a" and row["n"] == 4
    assert row["per_seat"]["0"]["wins"] == 1 and row["per_seat"]["0"]["losses"] == 1


def test_rank_layout_reports_mean_rank_first_places_and_higher_table():
    results = [
        match("4p", ["a", "b", "a", "b"], {0: 1, 1: 2, 2: 3, 3: 4}),
        match("4p", ["b", "a", "b", "a"], {0: 1, 1: 1, 2: 3, 3: 4}),
    ]
    section = summarize(SPEC, results).to_dict()["layouts"]["4p"]
    assert section["outcome_kind"] == "rank"
    a = section["agents"]["a"]
    assert a["n"] == 4 and a["mean_rank"] == pytest.approx((1 + 3 + 1 + 4) / 4) and a["first_place_rate"] == 0.5
    assert section["higher"]["a"]["b"]["n"] == 8
    assert section["higher"]["a"]["b"]["rate"] + section["higher"]["b"]["a"]["rate"] == pytest.approx(1.0)


def test_score_layout_reports_compositions_and_cross_play():
    results = [
        match("coop2", ["a", "a"], {0: 1}, scores={0: 4.0}),
        match("coop2", ["a", "a"], {0: 1}, scores={0: 6.0}),
        match("coop2", ["b", "a"], {0: 1}, scores={0: 1.0}),
    ]
    section = summarize(SPEC, results).to_dict()["layouts"]["coop2"]
    assert section["compositions"]["a+a"]["mean_score"] == 5.0 and section["compositions"]["a+a"]["homogeneous"]
    assert section["compositions"]["a+b"] == {"n": 1, "mean_score": 1.0, "score_ci": [1.0, 1.0],
                                              "homogeneous": False}


def test_roles_are_reported_separately_and_text_and_json_work(tmp_path):
    results = [match("hunt", ["h", "p"], {0: 1, 1: 2}, rewards=[2.0, -2.0])]
    report = summarize(SPEC, results, agents=["h", "p"], num_matches=1)
    section = report.to_dict()["layouts"]["hunt"]
    assert section["by_role"] == {"hunter": {"h": {"n": 1, "mean_return": 2.0}},
                                  "player": {"p": {"n": 1, "mean_return": -2.0}}}
    assert section["pairs"][0]["agent_a"] == "h" and section["pairs"][0]["wins"] == 1
    text = report.text()
    assert "layout hunt" in text and "role hunter" in text
    report.write_json(tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["layouts"]["hunt"]["n"] == 1


def test_mixed_teams_are_unattributed():
    spec = GameSpec.teams_of([2, 2], OBS, ACT)
    result = MatchResult(
        match_id="m", layout="2v2", outcome_kind="wdl",
        seats=[SeatResult(i, "player", i // 2, name, "latest", 0.0) for i, name in enumerate("abba")],
        teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    section = summarize(spec, [result]).to_dict()["layouts"]["2v2"]
    assert section["unattributed"] == 1 and section["pairs"] == []
```

- [ ] **Step 2: Write the failing engine and CLI tests**

Create `tests/contract/test_sp2_eval_engine.py` (spec section 6: "отчёт eval для каждого типа исхода" on the real `MatchRunner`):

```python
"""In-process eval API on the real MatchRunner: explicit models and lineups (T6.1)."""
from __future__ import annotations

from collections import Counter

import pytest
import torch

from colosseum.sp2.eval import play_lineups, schedule_lineups, summarize
from game_helpers import EliminationFFA, TurnTakingGame, make_test_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def models_for(spec, names, core="none", seed=0):
    torch.manual_seed(seed)
    role = spec.roles[next(iter(spec.roles))]
    return {name: make_test_model(role, core=core) for name in names}


def seat_agents(result) -> tuple[str, ...]:
    return tuple(s.agent_id for s in sorted(result.seats, key=lambda s: s.seat))


def turn_layout():
    spec = TurnTakingGame().spec
    return spec, next(iter(spec.layouts))


def test_every_lineup_is_played_exactly_once():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    lineups = schedule_lineups(spec, layout, players, 6)
    results = play_lineups(env_fn=TurnTakingGame, models=models_for(spec, players), lineups=lineups,
                           num_envs=4, seed=0)
    assert len(results) == len(lineups)
    assert Counter(seat_agents(r) for r in results) == Counter(tuple(s.agent_id for s in lu.seats) for lu in lineups)
    assert all(r.layout == layout and r.outcome_kind == "wdl" for r in results)
    report = summarize(spec, results, agents=["a", "b"], num_matches=6).to_dict()
    assert report["layouts"][layout]["pairs"][0]["n"] == 6


def test_more_envs_than_lineups_and_reproducible_with_a_seed():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles)}
    lineups = schedule_lineups(spec, layout, players, 3)
    models = models_for(spec, players)
    before = torch.get_rng_state()
    first = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=8, seed=5)
    assert torch.equal(torch.get_rng_state(), before)  # the caller's RNG is untouched
    second = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=8, seed=5)
    assert len(first) == 3
    key = sorted((r.match_id, tuple(s.reward for s in r.seats), r.episode_length) for r in first)
    assert key == sorted((r.match_id, tuple(s.reward for s in r.seats), r.episode_length) for r in second)


def test_ffa_layouts_with_eliminations_report_ranks():
    env = EliminationFFA(max_players=4)
    spec = env.spec
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    lineups = schedule_lineups(spec, "4p", players, 2) + schedule_lineups(spec, "2p", players, 2)
    results = play_lineups(env_fn=lambda: EliminationFFA(max_players=4), models=models_for(spec, players),
                           lineups=lineups, num_envs=2, seed=0)
    assert Counter(r.layout for r in results) == {"4p": 2, "2p": 2}
    assert {r.outcome_kind for r in results if r.layout == "4p"} == {"rank"}
    report = summarize(spec, results).to_dict()["layouts"]
    assert report["4p"]["agents"]["a"]["n"] == 4 and report["2p"]["outcome_kind"] == "wdl"


def test_stateful_models_play_and_keep_their_train_flags():
    spec, layout = turn_layout()
    players = {"a": list(spec.roles), "b": list(spec.roles)}
    models = models_for(spec, players, core="lstm")
    models["a"].train()
    models["b"].eval()
    results = play_lineups(env_fn=TurnTakingGame, models=models, lineups=schedule_lineups(spec, layout, players, 2),
                           num_envs=2, seed=1)
    assert len(results) == 2
    assert models["a"].training and not models["b"].training


def test_bad_arguments():
    spec, layout = turn_layout()
    lineups = schedule_lineups(spec, layout, {"a": list(spec.roles)}, 1)
    with pytest.raises(ValueError, match="unknown agents"):
        play_lineups(env_fn=TurnTakingGame, models={}, lineups=lineups)
    with pytest.raises(ValueError, match="num_envs"):
        play_lineups(env_fn=TurnTakingGame, models=models_for(spec, ["a"]), lineups=lineups, num_envs=0)
    assert play_lineups(env_fn=TurnTakingGame, models={}, lineups=[]) == []
```

Create `tests/integration/test_sp2_eval_cli.py` (SP1's eval CLI guarantees plus roles, signatures, layouts):

```python
"""`python -m colosseum.sp2 eval`: .pt and checkpoint-dir agents, roles and signatures, layouts, JSON (T6.1)."""
from __future__ import annotations

import json
import os
import time

import torch
from click.testing import CliRunner

from colosseum.sp2.cli import main
from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager
from colosseum.sp2.core.registry import build_model, env_spec
from colosseum.sp2.core.roles import role_signature
from colosseum.sp2.eval import load_eval_model
from game_helpers import agent_role_of, make_test_config, write_test_config

WIDE = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}}


def numpy_state(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _setup(tmp_path, game="turns", agent="agent_0", ckpt_agent="agent_b"):
    """Config file, config, a .pt of ``agent`` (config networks) and a checkpoint dir of a wider model."""
    cfg_path = write_test_config(tmp_path / "cfg.yaml", game)
    cfg = make_test_config(game)
    roles, role = agent_role_of(cfg, agent)
    torch.manual_seed(0)
    pt_path = tmp_path / "a.pt"
    torch.save(build_model(cfg.get_agent_config(agent), role).state_dict(), pt_path)
    wide_cfg = make_test_config(game, networks=WIDE)
    wide = build_model(wide_cfg.get_agent_config(agent), role)
    ckpt_id = CheckpointManager(tmp_path / "checkpoints").save(
        ckpt_agent, 3, numpy_state(wide),
        meta_extra={"networks": WIDE, "roles": roles, "role_signature": role_signature(role)},
    )
    return cfg_path, cfg, pt_path, tmp_path / "checkpoints" / ckpt_agent / ckpt_id


def invoke(*args):
    return CliRunner().invoke(main, ["eval", *map(str, args)])


def only_layout(cfg) -> str:
    return next(iter(env_spec(cfg).layouts))


def test_load_eval_model_builds_from_meta_networks_and_roles(tmp_path):
    _cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    model_pt, roles_pt = load_eval_model(cfg, "agent_0", pt_path)
    model_ckpt, roles_ckpt = load_eval_model(cfg, "b", ckpt_dir)
    assert roles_pt == roles_ckpt
    assert sum(p.numel() for p in model_ckpt.parameters()) > sum(p.numel() for p in model_pt.parameters())


def test_eval_cli_pairs_pt_and_checkpoint(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 4, "--num-envs", 2,
                    "--seed", 0, "--output", out)
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    section = data["layouts"][only_layout(cfg)]
    assert data["num_matches"] == 4 and section["outcome_kind"] == "wdl"
    assert [(r["agent_a"], r["agent_b"]) for r in section["pairs"]] == [("a", "b"), ("b", "a")]
    assert section["pairs"][0]["n"] == 4 and set(section["pairs"][0]["per_side"]) == {"0", "1"}
    assert "win_rate" in result.output


def test_eval_cli_rejects_bad_agent_specs(tmp_path):
    cfg_path, _cfg, pt_path, _ckpt = _setup(tmp_path)
    assert invoke("-c", cfg_path, "-a", "no-equals-sign").exit_code == 2
    assert invoke("-c", cfg_path, "-a", f"a={tmp_path / 'missing.pt'}").exit_code == 2
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"a={pt_path}")
    assert result.exit_code == 2 and "duplicate" in result.output


def test_eval_cli_rounds_an_odd_pairwise_count_up_and_keeps_it_for_one_agent(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path)
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 3, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    assert "using 4" in result.stderr and json.loads(out.read_text())["num_matches"] == 4
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-n", 3, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    section = json.loads(out.read_text())["layouts"][only_layout(cfg)]
    assert section["pairs"] == [] and section["solo"][0]["agent"] == "a" and section["n"] == 3


def test_malformed_checkpoint_and_role_signature_problems_are_config_errors(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path)
    meta = json.loads((ckpt_dir / "meta.json").read_text())

    def check(new_meta: dict, message: str) -> None:
        (ckpt_dir / "meta.json").write_text(json.dumps(new_meta))
        result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "-n", 2)
        assert result.exit_code == 1, result.output
        assert result.stderr.startswith("Config error:") and message in result.stderr, result.stderr
        assert "Traceback" not in result.output

    check({**meta, "policy_version": 7}, "policy_version")
    check({**meta, "role_signature": "another-game"}, "role signature")
    check({k: v for k, v in meta.items() if k not in ("roles", "role_signature")}, "not an SP2 checkpoint")
    check({**meta, "roles": ["nope"]}, "not roles of the game")
    check({**meta, "networks": {"bogus_key": 1}}, "networks")


def test_weights_of_another_architecture_and_unusable_pt_are_config_errors(tmp_path):
    cfg_path, _cfg, _pt, ckpt_dir = _setup(tmp_path)
    wide_pt = tmp_path / "wide.pt"
    torch.save(torch.load(ckpt_dir / "model.pt", weights_only=True), wide_pt)
    result = invoke("-c", cfg_path, "-a", f"a={wide_pt}")
    assert result.exit_code == 1 and "do not match" in result.stderr
    garbage = tmp_path / "bad.pt"
    garbage.write_bytes(b"not a torch file")
    result = invoke("-c", cfg_path, "-a", f"a={garbage}")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:") and str(garbage) in result.stderr


def test_load_eval_model_never_modifies_the_run(tmp_path):
    _cfg_path, cfg, _pt, ckpt_dir = _setup(tmp_path)
    agent_dir = ckpt_dir.parent
    stale = agent_dir / ".tmp-ckpt_v9-0123abcd"
    stale.mkdir()
    old = time.time() - 10 * 3600
    os.utime(stale, (old, old))
    interrupted = agent_dir / ".tmp-old-ckpt_v5-0123abcd"
    interrupted.mkdir()
    before = sorted(p.name for p in agent_dir.iterdir())
    load_eval_model(cfg, "b", ckpt_dir)
    assert sorted(p.name for p in agent_dir.iterdir()) == before


def test_layout_selection(tmp_path):
    cfg_path, _cfg, pt_path, ckpt_dir = _setup(tmp_path, game="ffa")
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-a", f"b={ckpt_dir}", "--layout", "2p", "-n", 2,
                    "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    assert list(json.loads(out.read_text())["layouts"]) == ["2p"]
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "--layout", "9p", "-n", 2)
    assert result.exit_code == 1 and "9p" in result.stderr


def test_asymmetric_agents_are_evaluated_in_their_roles(tmp_path):
    cfg_path, cfg, pt_path, ckpt_dir = _setup(tmp_path, game="asymmetric", agent="hunter", ckpt_agent="prey_ckpt")
    _roles, prey_role = agent_role_of(cfg, "prey")
    torch.manual_seed(1)
    prey_dir = tmp_path / "prey_ckpt"
    CheckpointManager(prey_dir).save(
        "prey", 1, numpy_state(build_model(cfg.get_agent_config("prey"), prey_role)),
        meta_extra={"roles": ["prey"], "role_signature": role_signature(prey_role)},
    )
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", f"hunter={pt_path}", "-a", f"prey={prey_dir / 'prey' / 'ckpt_v1'}",
                    "-n", 2, "--num-envs", 2, "-o", out)
    assert result.exit_code == 0, result.output
    sections = json.loads(out.read_text())["layouts"]
    assert sections and all(set(s["by_role"]) == {"hunter", "prey"} for s in sections.values())


def test_output_dir_must_exist_before_loading(tmp_path, monkeypatch):
    import colosseum.sp2.eval as eval_module

    cfg_path, _cfg, pt_path, _ckpt = _setup(tmp_path)

    def _never(*_args, **_kwargs):
        raise AssertionError("a model was loaded before --output was checked")

    monkeypatch.setattr(eval_module, "load_eval_model", _never)
    result = invoke("-c", cfg_path, "-a", f"a={pt_path}", "-o", tmp_path / "no_such_dir" / "r.json")
    assert result.exit_code == 2 and "no_such_dir" in result.output


def test_pt_that_is_not_a_state_dict_is_a_config_error(tmp_path):
    cfg_path, _cfg, _pt, _ckpt = _setup(tmp_path)
    path = tmp_path / "list.pt"
    torch.save([torch.zeros(2)], path)
    result = invoke("-c", cfg_path, "-a", f"a={path}")
    assert result.exit_code == 1 and str(path) in result.stderr
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_eval_schedule.py tests/unit/test_sp2_eval_report.py tests/contract/test_sp2_eval_engine.py tests/integration/test_sp2_eval_cli.py -q`
Expected: collection errors `No module named 'colosseum.sp2.eval'`.

- [ ] **Step 4: Write the eval module**

Create `src/colosseum/sp2/eval.py`:

```python
"""Evaluation: inference-only matches on ``MatchRunner`` (spec block 8).

Engine
------
``play_lineups`` plays explicit ``Lineup``s with explicit ``PolicyModel`` instances on the same
``MatchRunner`` core as training: seat lifecycle, masks, per-seat model states, results by team.
``Lineup.seats[*].agent_id`` is a key of ``models``; ``network_id`` is ignored (every seat uses
``models[agent_id]``). Every lineup is played once, to completion. Envs left without a scheduled
lineup keep playing their last one until the rest finish; those extra episodes are discarded,
so short episodes are not favoured. The in-process API is what the learning tests and the
future ``colosseum tournament`` (SP4) use.

Schedules (CLI, ``schedule_lineups``)
-------------------------------------
- Two or more teams, two or more agents: for every pair (a, b) and match m, team i gets the core
  ``(a, b)[(i + m) % 2]``; a seat whose role the core does not play goes to the other agent of the
  pair, so an asymmetric pair (hunter and prey) is never rotated. Pairs that cannot fill every
  seat, or that leave one of the two out, are not scheduled for the layout.
- Two or more teams, one agent: every team is that agent.
- One team (solo, cooperative): each agent alone (homogeneous team), then, with two or more
  agents and a team of two or more seats, every mixed composition (cross-play), rotating its
  seat orders over the matches.

Statistics (``summarize``), per layout, by the layout's outcome kind
--------------------------------------------------------------------
- ``wdl``: per pair, W/D/L from team ranks, win rate and score (draw = half) with 95% Wilson
  intervals (SP1), mean returns and length, and a per-side breakdown keyed by the team index of
  agent a. The reversed row is derived from the same counts. One agent: per-seat W/D/L and mean
  return. Lineups with a team mixing both agents are counted as ``unattributed``.
- ``rank``: per agent, mean team rank with a 95% normal interval and the share of first places;
  a pairwise "who ranked higher" table (``higher[a][b]``: score of a over b, draw = half).
- ``score``: per team composition (sorted agent ids joined by ``+``), mean team score with a 95%
  normal interval; homogeneous compositions are the per-agent results, the rest is cross-play.
- every kind: mean seat return per role and agent (``by_role``).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from colosseum.core.errors import ConfigError
from colosseum.sp2.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir, read_weights_file
from colosseum.sp2.coordinator.ratings import composition_key
from colosseum.sp2.core.config import ColosseumConfig, NetworkConfig
from colosseum.sp2.core.outcomes import pairwise_rank_score
from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, MatchResult, SeatAssignment, state_dict_from_numpy
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv
from colosseum.sp2.envs.vector import VectorEnv
from colosseum.sp2.networks.model import PolicyModel
from colosseum.sp2.worker.match_runner import EpisodeEnd, MatchRunner

logger = logging.getLogger(__name__)

Z_95 = 1.959963984540054
SCORE_CI_METHOD = "wilson-on-score (draw = half win; conservative approximation)"


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def _eval_seat(agent_id: str) -> SeatAssignment:
    return SeatAssignment(agent_id=agent_id, network_id=LATEST_NETWORK_ID, collect=False)


def _pair_lineup(spec: GameSpec, layout: str, pair: tuple[str, str], m: int,
                 players: Mapping[str, Sequence[str]]) -> Lineup | None:
    seat_specs = spec.layouts[layout]
    seats: list[SeatAssignment | None] = [None] * len(seat_specs)
    for team, members in enumerate(spec.teams(layout)):
        core = pair[(team + m) % 2]
        other = pair[1 - (team + m) % 2]
        for s in members:
            role = seat_specs[s].role
            if role in players[core]:
                seats[s] = _eval_seat(core)
            elif role in players[other]:
                seats[s] = _eval_seat(other)
            else:
                return None
    names = {seat.agent_id for seat in seats}  # type: ignore[union-attr]
    if names != set(pair):
        return None
    return Lineup(layout=layout, seats=seats)  # type: ignore[arg-type]


def _homogeneous(spec: GameSpec, layout: str, name: str, roles: Sequence[str]) -> Lineup | None:
    if any(seat.role not in roles for seat in spec.layouts[layout]):
        return None
    return Lineup(layout=layout, seats=[_eval_seat(name) for _ in spec.layouts[layout]])


def _cross_play_orders(spec: GameSpec, layout: str, composition: tuple[str, ...],
                       players: Mapping[str, Sequence[str]]) -> list[tuple[str, ...]]:
    roles = [seat.role for seat in spec.layouts[layout]]
    orders = sorted({order for order in itertools.permutations(composition)
                     if all(role in players[name] for name, role in zip(order, roles, strict=True))})
    return orders


def schedule_lineups(spec: GameSpec, layout: str, players: Mapping[str, Sequence[str]],
                     num_matches: int) -> list[Lineup]:
    """Lineups of one layout for ``players`` (name -> roles) by the CLI rules (module docstring).

    ``num_matches`` is per pair (two or more teams), per agent (one agent, or one team) and per
    mixed composition (cross-play). Returns ``[]`` when the players cannot fill the layout.
    """
    if layout not in spec.layouts:
        raise ValueError(f"schedule_lineups: unknown layout {layout!r}")
    if num_matches < 1:
        raise ValueError(f"schedule_lineups: num_matches must be >= 1, got {num_matches}")
    names = list(players)
    if not names:
        raise ValueError("schedule_lineups: no players")
    lineups: list[Lineup] = []
    if spec.num_teams(layout) >= 2 and len(names) >= 2:
        for pair in itertools.combinations(names, 2):
            pair_lineups = [_pair_lineup(spec, layout, pair, m, players) for m in range(num_matches)]
            if all(lineup is not None for lineup in pair_lineups):
                lineups.extend(pair_lineups)  # type: ignore[arg-type]
        return lineups
    for name in names:
        lineup = _homogeneous(spec, layout, name, players[name])
        if lineup is not None:
            lineups.extend(Lineup(layout, list(lineup.seats)) for _ in range(num_matches))
    size = spec.layout_size(layout)
    if spec.num_teams(layout) == 1 and len(names) >= 2 and size >= 2:
        for composition in itertools.combinations_with_replacement(names, size):
            if len(set(composition)) < 2:
                continue
            orders = _cross_play_orders(spec, layout, composition, players)
            for m in range(num_matches if orders else 0):
                lineups.append(Lineup(layout, [_eval_seat(name) for name in orders[m % len(orders)]]))
    return lineups


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class _FixedModels:
    """``ModelPool`` that serves ``models[agent_id]`` for any network id."""

    def __init__(self, models: Mapping[str, PolicyModel]) -> None:
        self._models = models

    def get(self, agent_id: str, network_id: str) -> PolicyModel | None:
        return self._models.get(agent_id)


class _Collector:
    """``MatchObserver`` that keeps the result of every scheduled lineup and feeds the next one."""

    def __init__(self, pending: deque[Lineup], scheduled: list[bool]) -> None:
        self.runner: MatchRunner | None = None
        self.results: list[MatchResult] = []
        self._pending = pending
        self._scheduled = scheduled

    def on_act(self, env, seat, record) -> None:
        pass

    def on_rewards(self, env, rewards) -> None:
        pass

    def on_terminated(self, env, seats) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        pass

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        if self._scheduled[env]:
            self.results.append(end.result)
        if self._pending:
            self.runner.set_next_lineup(env, self._pending.popleft())  # applied at this episode end
            self._scheduled[env] = True
        else:
            self._scheduled[env] = False


def play_lineups(
    *,
    env_fn: Callable[[], MultiAgentEnv],
    models: Mapping[str, PolicyModel],
    lineups: Sequence[Lineup],
    num_envs: int = 8,
    seed: int | None = None,
    deterministic: bool = False,
    max_idle_steps: int = 1000,
) -> list[MatchResult]:
    """Play every lineup once, to completion; one ``MatchResult`` per lineup, in completion order.

    ``seed`` seeds the episode resets and a forked torch RNG, so the caller's global RNG is
    untouched. Models run in eval mode; their train/eval flags are restored on return.
    """
    lineups = list(lineups)
    if not lineups:
        return []
    unknown = sorted({seat.agent_id for lineup in lineups for seat in lineup.seats} - set(models))
    if unknown:
        raise ValueError(f"play_lineups: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_lineups: num_envs must be >= 1, got {num_envs}")
    rng = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()
    was_training = [(m, m.training) for model in models.values() for m in model.modules()]
    try:
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            for model in models.values():
                model.eval()
            n = min(num_envs, len(lineups))
            pending = deque(lineups[n:])
            collector = _Collector(pending, [True] * n)
            runner = MatchRunner(vec_env=VectorEnv(env_fn, n), lineups=lineups[:n], models=_FixedModels(models),
                                 observer=collector, seed=seed, max_idle_steps=max_idle_steps,
                                 deterministic=deterministic, context="eval, ", match_id_prefix="eval")
            collector.runner = runner
            try:
                while len(collector.results) < len(lineups):
                    runner.step()
            finally:
                runner.close()
            return collector.results
    finally:
        for module, training in was_training:
            module.training = training


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for ``successes / n`` (fractional successes allowed); n=0 -> (0, 1)."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1.0 - p)) / n + z * z / (4.0 * n * n)) / denom
    return min(max(0.0, center - half), p), max(min(1.0, center + half), p)


def normal_interval(values: Sequence[float], z: float = Z_95) -> tuple[float, float, float]:
    """``(mean, low, high)`` with ``mean +- z * sd / sqrt(n)`` (ddof=1); n < 2 -> zero width."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean, mean
    half = z * float(x.std(ddof=1)) / math.sqrt(x.size)
    return mean, mean - half, mean + half


def _team_agents(result: MatchResult) -> dict[int, set[str]]:
    teams: dict[int, set[str]] = defaultdict(set)
    for seat in result.seats:
        teams[seat.team].add(seat.agent_id)
    return teams


def _team_return(result: MatchResult, team: int) -> float:
    return float(np.mean([seat.reward for seat in result.seats if seat.team == team]))


@dataclass
class PairStats:
    """WDL counts of one unordered pair, from ``agent_a``'s side; ``per_side`` is keyed by a's team."""

    agent_a: str
    agent_b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    return_a: float = 0.0
    return_b: float = 0.0
    total_length: int = 0
    per_side: dict[int, list[int]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def add(self, result: MatchResult, team_a: int, team_b: int) -> None:
        ranks = {team.team: team.rank for team in result.teams}
        score = pairwise_rank_score(ranks[team_a], ranks[team_b])
        cell = self.per_side.setdefault(team_a, [0, 0, 0])
        idx = 0 if score == 1.0 else (1 if score == 0.5 else 2)
        cell[idx] += 1
        if idx == 0:
            self.wins += 1
        elif idx == 1:
            self.draws += 1
        else:
            self.losses += 1
        self.return_a += _team_return(result, team_a)
        self.return_b += _team_return(result, team_b)
        self.total_length += result.episode_length

    def reversed(self) -> PairStats:
        return PairStats(
            agent_a=self.agent_b, agent_b=self.agent_a, wins=self.losses, draws=self.draws, losses=self.wins,
            return_a=self.return_b, return_b=self.return_a, total_length=self.total_length,
            per_side={1 - side: [v[2], v[1], v[0]] for side, v in self.per_side.items()},
        )

    def to_row(self) -> dict[str, Any]:
        n = self.n
        points = self.wins + 0.5 * self.draws
        per_side = {}
        for side, (w, d, losses) in sorted(self.per_side.items()):
            m = w + d + losses
            per_side[str(side)] = {"n": m, "wins": w, "draws": d, "losses": losses,
                                   "score": (w + 0.5 * d) / m if m else 0.0}
        return {
            "agent_a": self.agent_a, "agent_b": self.agent_b, "n": n,
            "wins": self.wins, "draws": self.draws, "losses": self.losses,
            "win_rate": self.wins / n if n else 0.0, "win_rate_ci": list(wilson_interval(self.wins, n)),
            "score": points / n if n else 0.0, "score_ci": list(wilson_interval(points, n)),
            "mean_return_a": self.return_a / n if n else 0.0, "mean_return_b": self.return_b / n if n else 0.0,
            "mean_episode_length": self.total_length / n if n else 0.0,
            "per_side": per_side,
        }


def _by_role(results: Sequence[MatchResult]) -> dict[str, dict[str, dict[str, float]]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for seat in result.seats:
            acc[seat.role][seat.agent_id].append(float(seat.reward))
    return {role: {agent: {"n": len(v), "mean_return": float(np.mean(v))} for agent, v in sorted(agents.items())}
            for role, agents in sorted(acc.items())}


def _summarize_wdl(results: Sequence[MatchResult]) -> dict[str, Any]:
    pairs: dict[tuple[str, str], PairStats] = {}
    solo: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    solo_wdl: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    unattributed = 0
    for result in results:
        teams = _team_agents(result)
        names = sorted(set().union(*teams.values()))
        if len(names) == 1:
            ranks = {team.team: team.rank for team in result.teams}
            for seat in result.seats:
                solo[names[0]][seat.seat].append(float(seat.reward))
                other = min(rank for team, rank in ranks.items() if team != seat.team)
                own = ranks[seat.team]
                solo_wdl[names[0]][seat.seat][0 if own < other else (1 if own == other else 2)] += 1
            continue
        if len(names) != 2 or any(len(agents) != 1 for agents in teams.values()):
            unattributed += 1
            continue
        a, b = names
        team_a = next(t for t, agents in teams.items() if agents == {a})
        team_b = next(t for t, agents in teams.items() if agents == {b})
        pairs.setdefault((a, b), PairStats(a, b)).add(result, team_a, team_b)
    rows = []
    for pair in pairs.values():
        rows.append(pair.to_row())
        rows.append(pair.reversed().to_row())
    solo_rows = []
    for name, seats in solo.items():
        all_returns = [r for v in seats.values() for r in v]
        mean, low, high = normal_interval(all_returns)
        solo_rows.append({
            "agent": name, "n": len(all_returns), "mean_return": mean, "return_ci": [low, high],
            "per_seat": {str(s): {"n": len(v), "mean_return": float(np.mean(v)),
                                  "wins": solo_wdl[name][s][0], "draws": solo_wdl[name][s][1],
                                  "losses": solo_wdl[name][s][2]}
                         for s, v in sorted(seats.items())},
        })
    return {"pairs": rows, "solo": solo_rows, "unattributed": unattributed, "score_ci_method": SCORE_CI_METHOD}


def _summarize_rank(results: Sequence[MatchResult]) -> dict[str, Any]:
    ranks: dict[str, list[float]] = defaultdict(list)
    higher: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        teams = _team_agents(result)
        team_rank = {team.team: team.rank for team in result.teams}
        homogeneous = {t: next(iter(agents)) for t, agents in teams.items() if len(agents) == 1}
        for team, agent in homogeneous.items():
            ranks[agent].append(team_rank[team])
        for ta, tb in itertools.combinations(sorted(homogeneous), 2):
            a, b = homogeneous[ta], homogeneous[tb]
            if a == b:
                continue
            score = pairwise_rank_score(team_rank[ta], team_rank[tb])
            higher[a][b].append(score)
            higher[b][a].append(1.0 - score)
    agents = {}
    for agent, values in sorted(ranks.items()):
        mean, low, high = normal_interval(values)
        agents[agent] = {"n": len(values), "mean_rank": mean, "rank_ci": [low, high],
                         "first_place_rate": float(np.mean([v == 1.0 for v in values]))}
    table = {a: {b: {"n": len(v), "rate": float(np.mean(v))} for b, v in sorted(row.items())}
             for a, row in sorted(higher.items())}
    return {"agents": agents, "higher": table}


def _summarize_score(results: Sequence[MatchResult]) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        scores[composition_key(seat.agent_id for seat in result.seats)].append(float(result.teams[0].score))
    compositions = {}
    for key, values in sorted(scores.items()):
        mean, low, high = normal_interval(values)
        compositions[key] = {"n": len(values), "mean_score": mean, "score_ci": [low, high],
                             "homogeneous": len(set(key.split("+"))) == 1}
    return {"compositions": compositions}


@dataclass
class EvalReport:
    """Result of an evaluation; ``to_dict()`` is the JSON written by ``--output``."""

    agents: list[str]
    num_matches: int
    deterministic: bool
    layouts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"agents": list(self.agents), "num_matches": self.num_matches, "deterministic": self.deterministic,
                "ci_level": 0.95, "layouts": self.layouts}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def text(self) -> str:
        lines: list[str] = []
        for layout, report in self.layouts.items():
            lines.append(f"== layout {layout} ({report['outcome_kind']}, {report['n']} matches)")
            if report["outcome_kind"] == "wdl":
                for r in report["pairs"]:
                    lines.append(
                        f"{r['agent_a']:<16} vs {r['agent_b']:<16} n={r['n']:<5} W={r['wins']:<5} "
                        f"D={r['draws']:<5} L={r['losses']:<5} "
                        f"win_rate {r['win_rate']:.3f} [{r['win_rate_ci'][0]:.3f}, {r['win_rate_ci'][1]:.3f}]  "
                        f"score {r['score']:.3f} [{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]")
                    for side, cell in r["per_side"].items():
                        lines.append(f"{'':<20}as team {side}: n={cell['n']} W={cell['wins']} "
                                     f"D={cell['draws']} L={cell['losses']} score={cell['score']:.3f}")
                for r in report["solo"]:
                    lines.append(f"{r['agent']:<16} alone: n={r['n']} mean_return {r['mean_return']:.3f} "
                                 f"[{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}]")
                if report["unattributed"]:
                    lines.append(f"({report['unattributed']} matches with mixed teams not attributed to a pair)")
            elif report["outcome_kind"] == "rank":
                for agent, r in report["agents"].items():
                    lines.append(f"{agent:<16} n={r['n']:<5} mean_rank {r['mean_rank']:.3f} "
                                 f"[{r['rank_ci'][0]:.3f}, {r['rank_ci'][1]:.3f}]  "
                                 f"first places {r['first_place_rate']:.3f}")
                for a, row in report["higher"].items():
                    for b, cell in row.items():
                        lines.append(f"{'':<4}{a} above {b}: {cell['rate']:.3f} (n={cell['n']})")
            else:
                for key, r in report["compositions"].items():
                    lines.append(f"{key:<24} n={r['n']:<5} mean_score {r['mean_score']:.3f} "
                                 f"[{r['score_ci'][0]:.3f}, {r['score_ci'][1]:.3f}]")
            for role, agents in report["by_role"].items():
                for agent, cell in agents.items():
                    lines.append(f"{'':<4}role {role}: {agent} mean_return {cell['mean_return']:.3f} (n={cell['n']})")
        return "\n".join(lines)


def summarize(spec: GameSpec, results: Sequence[MatchResult], *, agents: Sequence[str] = (),
              num_matches: int = 0, deterministic: bool = False) -> EvalReport:
    """Aggregate match results into an :class:`EvalReport`, one section per layout."""
    by_layout: dict[str, list[MatchResult]] = defaultdict(list)
    for result in results:
        by_layout[result.layout].append(result)
    names = list(agents) or sorted({seat.agent_id for r in results for seat in r.seats})
    report = EvalReport(agents=names, num_matches=num_matches, deterministic=deterministic)
    for layout, layout_results in by_layout.items():
        kind = spec.outcome_kind(layout)
        section: dict[str, Any] = {"outcome_kind": kind, "n": len(layout_results)}
        if kind == "wdl":
            section.update(_summarize_wdl(layout_results))
        elif kind == "rank":
            section.update(_summarize_rank(layout_results))
        else:
            section.update(_summarize_score(layout_results))
        section["by_role"] = _by_role(layout_results)
        report.layouts[layout] = section
    return report


# ---------------------------------------------------------------------------
# Loading agents and the CLI-level evaluation
# ---------------------------------------------------------------------------


def _agent_model_config(config: ColosseumConfig, name: str) -> ColosseumConfig:
    """``config.get_agent_config(name)`` for a configured agent, else the global sections."""
    if name in config.get_trainable_agent_ids():
        return config.get_agent_config(name)
    return config.model_copy(update={"agents": {}})


def _pt_roles(config: ColosseumConfig, spec: GameSpec, name: str) -> list[str]:
    if name in config.get_trainable_agent_ids():
        return resolve_agent_roles(config, spec)[name]
    roles = list(spec.roles)
    signatures = {role_signature(spec.roles[r]) for r in roles}
    if len(signatures) > 1:
        raise ConfigError(
            f"agent {name!r}: a .pt file without a matching agents.{name} entry plays every role of "
            f"the game, but the roles {roles} have different spaces; name the agent after a configured "
            f"agent (agents.<id>.roles) or use a checkpoint dir"
        )
    return roles


def load_eval_model(config: ColosseumConfig, name: str, path: str | Path, *,
                    spec: GameSpec | None = None) -> tuple[PolicyModel, list[str]]:
    """Build and load one evaluation agent; returns ``(model in eval mode, roles)``.

    - A checkpoint dir (read with ``load_checkpoint_dir``: strict, never modified): roles and
      ``role_signature`` from its ``meta.json`` (required; the signature must match the game's
      spaces for those roles), architecture from its ``networks`` (else the agent's config).
    - A ``.pt`` state_dict: architecture and roles of ``agents.<name>`` when configured, else the
      global ``networks`` with every role of the game (which must share one signature).

    Problems raise ConfigError naming the path; a path that is neither a dir nor a ``.pt`` file
    raises FileNotFoundError.
    """
    from colosseum.sp2.core.registry import build_model, env_spec

    spec = spec if spec is not None else env_spec(config)
    p = Path(path)
    model_config = _agent_model_config(config, name)
    if p.is_dir():
        loaded = load_checkpoint_dir(p)
        roles, signature = loaded["roles"], loaded["role_signature"]
        if roles is None or signature is None:
            raise ConfigError(f"Checkpoint {p}: meta.json has no roles/role_signature (not an SP2 checkpoint)")
        unknown = sorted(set(roles) - set(spec.roles))
        if unknown:
            raise ConfigError(f"Checkpoint {p}: roles {unknown} are not roles of the game {sorted(spec.roles)}")
        expected = role_signature(agent_role_spec(spec, roles))
        if signature != expected:
            raise ConfigError(
                f"Checkpoint {p}: role signature {signature!r} does not match the game's spaces for roles "
                f"{roles} ({expected!r}); it was trained on a different game or game version"
            )
        networks = loaded["meta"].get("networks")
        if networks is not None:
            try:
                model_config = model_config.model_copy(update={"networks": NetworkConfig.model_validate(networks)})
            except ValidationError as e:
                raise ConfigError(f"Checkpoint {p}: invalid meta.json networks:\n{e}") from e
        model_state = loaded["model_state"]
    elif p.is_file() and p.suffix == ".pt":
        roles = _pt_roles(config, spec, name)
        try:
            model_state = read_weights_file(p)
        except ValueError as e:
            raise ConfigError(f"{p}: {e}") from e
    else:
        raise FileNotFoundError(f"{p}: expected a checkpoint directory or a .pt file")
    try:
        model = build_model(model_config, agent_role_spec(spec, roles))
    except ConfigError as e:
        raise ConfigError(f"{p}: {e}") from e
    check_model_state(model, model_state, str(p))
    model.load_state_dict(state_dict_from_numpy(model_state))
    model.eval()
    return model, list(roles)


def default_layouts(spec: GameSpec, players: Mapping[str, Sequence[str]]) -> list[str]:
    """Every layout of the game for which ``schedule_lineups`` has at least one lineup."""
    return [name for name in spec.layouts if schedule_lineups(spec, name, players, 1)]


def evaluate(config: ColosseumConfig, agents: Mapping[str, str], *, layouts: Sequence[str] | None,
             num_matches: int, seed: int | None = None, deterministic: bool = False,
             num_envs: int = 8) -> EvalReport:
    """Load ``agents`` (name -> checkpoint dir or ``.pt``), schedule, play and summarize."""
    from colosseum.sp2.core.registry import env_spec, make_env

    spec = env_spec(config)
    models: dict[str, PolicyModel] = {}
    players: dict[str, list[str]] = {}
    for name, path in agents.items():
        models[name], players[name] = load_eval_model(config, name, path, spec=spec)
    chosen = list(layouts) if layouts else default_layouts(spec, players)
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    lineups: list[Lineup] = []
    for layout in chosen:
        layout_lineups = schedule_lineups(spec, layout, players, num_matches)
        if not layout_lineups:
            raise ConfigError(f"layout {layout!r}: the agents {sorted(players)} cannot fill its seats")
        lineups.extend(layout_lineups)
    if not lineups:
        raise ConfigError(f"no layout of the game can be filled by the agents {sorted(players)}")
    results = play_lineups(env_fn=lambda: make_env(config), models=models, lineups=lineups, num_envs=num_envs,
                           seed=seed, deterministic=deterministic, max_idle_steps=config.env.max_idle_steps)
    return summarize(spec, results, agents=list(agents), num_matches=num_matches, deterministic=deterministic)
```

How `play_lineups` keeps "every lineup once": `_Collector.on_episode_end` counts the result of an env whose lineup was scheduled and, inside the same callback, gives the env its next lineup with `set_next_lineup`; by the `MatchRunner` step order (overview, T3.3: `on_episode_end` → apply the next lineup) it is applied at that same episode end. Envs without a next lineup keep playing their last one; those episodes are not counted.

- [ ] **Step 5: Add the CLI command**

Insert into `src/colosseum/sp2/cli.py`, directly above the final `if __name__ == "__main__":` line:

```python
@main.command("eval")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; agents.<name> / networks build .pt agents")
@click.option("--agent", "-a", "agents", required=True, multiple=True,
              help="name=path. path is a checkpoint dir (architecture and roles from its meta.json) or a "
                   ".pt state_dict (architecture and roles of agents.<name>, else the global networks "
                   "playing every role). Repeatable.")
@click.option("--layout", "layouts", multiple=True,
              help="Layout to evaluate (repeatable). Default: every layout the agents can fill.")
@click.option("--num-matches", "-n", default=100, type=click.IntRange(min=1), show_default=True,
              help="Matches per agent pair and layout (one agent or one team: per agent; cross-play: per "
                   "composition). With several agents and a layout of two or more teams an odd count is "
                   "rounded up, so every agent plays every side equally often.")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Parallel environments")
@click.option("--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode)")
@click.option("--seed", default=None, type=int, help="Seed for env resets and sampling")
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False),
              help="Write the machine-readable result as JSON")
def eval_cmd(
    config: str,
    agents: tuple[str, ...],
    layouts: tuple[str, ...],
    num_matches: int,
    num_envs: int,
    deterministic: bool,
    seed: int | None,
    output: str | None,
) -> None:
    """Evaluate agents/checkpoints against each other (no training).

    Exit code: 0 done, 1 config error (also a malformed checkpoint, a role signature or weights
    that do not fit), 2 bad command-line arguments, 130 SIGINT (Ctrl+C), 143 SIGTERM.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if output is not None and not Path(output).parent.is_dir():
        raise click.BadParameter(f"directory {str(Path(output).parent)!r} does not exist", param_hint="'--output'")
    with _config_errors():
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import env_spec, validate_config
        from colosseum.sp2.eval import evaluate

        cfg = load_config(config)
        validate_config(cfg)
        specs = [_parse_agent_spec(spec) for spec in agents]
        names = [name for name, _ in specs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise click.BadParameter(f"duplicate agent names {duplicates}", param_hint="'--agent'")
        for _name, path in specs:
            p = Path(path)
            if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
                raise click.BadParameter(f"{p}: expected a checkpoint directory or a .pt file",
                                         param_hint="'--agent'")
        spec = env_spec(cfg)
        chosen = list(layouts) or list(spec.layouts)
        competitive = any(spec.num_teams(name) >= 2 for name in chosen if name in spec.layouts)
        if len(specs) > 1 and competitive and num_matches % 2:
            click.echo(f"Note: --num-matches {num_matches} is odd; using {num_matches + 1} per pair so every "
                       f"agent plays every side equally often.", err=True)
            num_matches += 1
        report = evaluate(cfg, dict(specs), layouts=list(layouts) or None, num_matches=num_matches, seed=seed,
                          deterministic=deterministic, num_envs=num_envs)
    click.echo("\n" + report.text())
    if output:
        report.write_json(output)
        click.echo(f"Result written to {output}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_eval_schedule.py tests/unit/test_sp2_eval_report.py tests/contract/test_sp2_eval_engine.py tests/integration/test_sp2_eval_cli.py -q`
Expected: all passed.

- [ ] **Step 7: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/sp2/eval.py src/colosseum/sp2/cli.py \
        tests/unit/test_sp2_eval_schedule.py tests/unit/test_sp2_eval_report.py \
        tests/contract/test_sp2_eval_engine.py tests/integration/test_sp2_eval_cli.py
git commit -m "feat: SP2 eval on MatchRunner with per-layout reports and CLI eval (SP2 T6.1)"
git push origin sp2-game-model
```

---

### Task T6.2: `validate_config` and CLI `validate`

Spec block 9, "validate": the `GameSpec` checks (block 1), roles and matchmaking (blocks 5, 7), the kickstart teacher signature (block 6), `critic_encoder_class` only with a `global_state_space` (block 3), a `reset` of every enabled layout plus a few random legal steps under the contract checks with full `space.contains`, and every agent's model: `step` on its role's observations and `unroll` on a synthetic chunk with `boot`/`pad` slots, a reset and `global_state`. SP1's model checks (state batch dimension with two batch sizes, state payload round trip, value shape, finite log-probs) carry over. `train`, `eval`, `bc` and the distributed roles call it before they build anything.

**Files:**
- Create: `src/colosseum/sp2/core/validation.py`
- Modify: `src/colosseum/sp2/core/registry.py` (add `validate_config`, a thin re-export)
- Modify: `src/colosseum/sp2/launcher.py` (`validate_run_config` calls it)
- Modify: `src/colosseum/sp2/cli.py` (add the `validate` command)
- Test: `tests/unit/test_sp2_validate.py`

**Interfaces:**
- Consumes: `env_spec`, `make_env`, `build_model` (T2.4); `resolve_agent_roles`, `agent_role_spec`, `role_signature` (T2.4); `validate_matchmaking`, `enabled_layouts` (T5.1); `EpisodeTracker(spec, max_idle_steps, context)` with `on_reset` / `on_step` returning the acting seats' normalized masks (T1.5); `ActionSpec.from_space` (`groups`, `has_masks`, `allocate_actions`, `full_mask`, `boot_mask`), `ObsSpec.from_space(...).allocate` (T1.3); `ActionGroup.kind/path/nvec/mask_size/units` and `Units.sample(mask=...)` (T1.2, T1.3); tree utilities (T1.1); `Distribution` (T2.1); `UnrollOutput` (T2.3); `check_model_state`, `read_weights_file` (T5.3); `pack_payload` / `unpack_payload` (`colosseum.transport.serialization`, unchanged until T6.4 copies it); `colosseum.networks.state` (unchanged).
- Produces:
  - `colosseum.sp2.core.registry.validate_config(config) -> None` (contract; ConfigError, or EnvContractError for an env that breaks the contract).
  - *`colosseum.sp2.core.validation`: `validate_config`, `random_legal_action(role, mask, rng)`, `VALIDATE_STEPS = 8`.
  - `python -m colosseum.sp2 validate -c cfg [--set k=v ...]` prints `  OK: agent '<id>'` per agent and `Config is valid.`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp2_validate.py`:

```python
"""validate_config (spec block 9) and `python -m colosseum.sp2 validate` (T6.2)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from colosseum.core.errors import ConfigError, EnvContractError
from colosseum.sp2.cli import main
from colosseum.sp2.core.registry import validate_config
from colosseum.sp2.core.validation import random_legal_action
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, RoleSpec, StepResult
from colosseum.sp2.networks.model import UnrollOutput
from game_helpers import GameTestModel, make_test_config

BOX = gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32)
TWO = gymnasium.spaces.Discrete(2)


class BoundedSolo(MultiAgentEnv):
    """Solo, 3 steps, Box(0, 1) observations; ``bad`` breaks one rule at step 2."""

    spec = GameSpec.solo(BOX, TWO)

    def __init__(self, bad: str = "") -> None:
        self.bad = bad
        self.t = 0

    def _obs(self) -> np.ndarray:
        obs = np.full(3, 0.5, np.float32)
        if self.bad == "out_of_bounds" and self.t == 2:
            obs[0] = 5.0
        return obs

    def reset(self, seed, layout):
        if self.bad == "reset":
            raise RuntimeError("cannot reset")
        self.t = 0
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: np.array([True, True])})

    def step(self, actions):
        self.t += 1
        if self.t >= 3:
            return StepResult(acting=set(), obs={}, rewards={0: 1.0}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()}, rewards={0: 1.0},
                          action_masks={0: np.array([True, True])})


class ValueAsMatrixModel(GameTestModel):
    """``unroll`` returns values shaped [S, B] instead of [S*B]."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state, with_value)
        if out.value is None:
            return out
        return UnrollOutput(out.dist, out.value.reshape(reset_after.shape))


class AlwaysValueModel(GameTestModel):
    """``unroll(with_value=False)`` still returns values."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        return super().unroll(obs, state0, reset_after, action_mask, global_state, True)


class LayerFirstStateModel(GameTestModel):
    """State laid out [layers, B, H]: the batch dimension is not first."""

    def initial_state(self, batch_size, device="cpu"):
        return torch.zeros(2, batch_size, 4, device=device)


def solo_config(**sections):
    data = {"env": {"env_class": "test_sp2_validate.BoundedSolo"},
            "networks": {"model_class": "game_helpers.GameTestModel"}}
    for key, value in sections.items():
        data[key] = {**data.get(key, {}), **value}
    from colosseum.sp2.core.config import ColosseumConfig

    return ColosseumConfig.model_validate(data)


@pytest.mark.parametrize("game", ["solo", "turns", "simultaneous", "ffa", "dead_teammate", "units", "asymmetric",
                                  "coop", "global_state"])
def test_toy_game_configs_are_valid(game):
    validate_config(make_test_config(game))


def test_a_bounded_env_passes():
    validate_config(solo_config())


def test_observation_outside_the_space_is_an_env_contract_error():
    with pytest.raises(EnvContractError, match="observation is not in the role's space"):
        validate_config(solo_config(env={"kwargs": {"bad": "out_of_bounds"}}))


def test_reset_failure_is_a_config_error():
    with pytest.raises(ConfigError, match="env.reset"):
        validate_config(solo_config(env={"kwargs": {"bad": "reset"}}))


def test_matchmaking_and_roles_are_checked():
    with pytest.raises(ConfigError, match="unknown layouts"):
        validate_config(make_test_config("turns", matchmaking={"layouts": {"9p": 1.0}}))
    with pytest.raises(ConfigError, match="prey"):
        validate_config(make_test_config("asymmetric", agents={"hunter": {"roles": ["hunter"]}}))


def test_critic_encoder_needs_a_global_state():
    cfg = solo_config(networks={"model_class": None, "encoder_class": "game_helpers.Nope",
                                "policy_class": "game_helpers.Nope", "value_class": "game_helpers.Nope",
                                "critic_encoder_class": "game_helpers.Nope"})
    with pytest.raises(ConfigError, match="critic_encoder_class"):
        validate_config(cfg)


@pytest.mark.parametrize(("model", "message"), [
    ("ValueAsMatrixModel", "time-major values"),
    ("AlwaysValueModel", "with_value=False"),
    ("LayerFirstStateModel", "batch dimension first"),
])
def test_model_protocol_violations(model, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(solo_config(networks={"model_class": f"test_sp2_validate.{model}"}))


def test_kickstart_teacher_must_fit_every_agent(tmp_path):
    cfg = solo_config()
    teacher = tmp_path / "teacher.pt"
    torch.save(GameTestModel(BOX, TWO).state_dict(), teacher)
    validate_config(solo_config(training={"kickstart_teacher": str(teacher)}))
    wrong = tmp_path / "wrong.pt"
    torch.save(torch.nn.Linear(2, 2).state_dict(), wrong)
    with pytest.raises(ConfigError, match="do not match"):
        validate_config(solo_config(training={"kickstart_teacher": str(wrong)}))
    assert cfg.training.kickstart_teacher is None


def test_one_global_teacher_cannot_serve_roles_with_different_spaces(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"never read")
    with pytest.raises(ConfigError, match="kickstart_teacher"):
        validate_config(make_test_config("asymmetric", training={"kickstart_teacher": str(teacher)}))


def test_random_legal_action_respects_masks():
    rng = np.random.default_rng(0)
    role = RoleSpec(BOX, gymnasium.spaces.Discrete(4))
    assert {int(random_legal_action(role, np.array([False, True, False, False]), rng)) for _ in range(20)} == {1}
    multi = RoleSpec(BOX, gymnasium.spaces.MultiDiscrete([2, 3]))
    mask = np.array([False, True, True, False, False])
    assert {tuple(random_legal_action(multi, mask, rng).tolist()) for _ in range(20)} == {(1, 0)}
    mixed = RoleSpec(BOX, gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(3),
                                                 "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)}))
    action = random_legal_action(mixed, {"move": np.array([False, False, True])}, rng)
    assert int(action["move"]) == 2 and action["aim"].shape == (2,) and mixed.action_space.contains(action)


def test_cli_validate_reports_ok_and_one_line_errors(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(make_test_config("asymmetric").model_dump(mode="json", by_alias=True)))
    result = CliRunner().invoke(main, ["validate", "-c", str(good)])
    assert result.exit_code == 0, result.output
    assert "OK: agent 'hunter'" in result.output and "Config is valid." in result.output
    result = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "rollout.num_worker=3"])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:") and "num_worker" in result.stderr
    result = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "matchmaking.layouts={9p: 1}"])
    assert result.exit_code == 1 and "unknown layouts" in result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_validate.py -q`
Expected: collection error `ImportError: cannot import name 'validate_config' from 'colosseum.sp2.core.registry'` (or `No module named 'colosseum.sp2.core.validation'`).

- [ ] **Step 3: Write the implementation**

Create `src/colosseum/sp2/core/validation.py`:

```python
"""``validate_config``: everything that can be checked before a run starts (spec block 9).

Re-exported as ``colosseum.sp2.core.registry.validate_config`` (callers use that name, so a test
can monkeypatch it there).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from colosseum.core.errors import ConfigError, EnvContractError
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.registry import build_model, env_spec, make_env
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import (
    tree_get,
    tree_index,
    tree_leaves,
    tree_map,
    tree_same_structure,
    tree_stack,
    tree_to_numpy,
    tree_to_torch,
)
from colosseum.sp2.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD
from colosseum.sp2.envs.game import GameSpec, RoleSpec
from colosseum.sp2.networks.dist import Distribution

VALIDATE_STEPS = 8  # random legal steps per enabled layout


def _subspace(space: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        space = space[key]
    return space


def _set_path(tree: Any, path: tuple[str, ...], value: Any) -> Any:
    if not path:
        return value
    node = tree
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return tree


def random_legal_action(role: RoleSpec, mask: Any, rng: Any) -> Any:
    """A uniformly random legal action of ``role`` under its normalized ``mask`` (numpy tree)."""

    spec = ActionSpec.from_space(role.action_space)
    action = spec.allocate_actions(())
    for group in spec.groups:
        group_mask = None
        if mask is not None and (group.mask_size > 0 or group.kind == "units"):
            group_mask = tree_get(mask, group.path)
        if group.kind == "discrete":
            legal = np.flatnonzero(group_mask) if group_mask is not None else np.arange(group.nvec[0])
            value: Any = np.asarray(rng.choice(legal), dtype=np.int64)
        elif group.kind == "multi_discrete":
            picks, offset = [], 0
            for n in group.nvec:
                row = group_mask[offset:offset + n] if group_mask is not None else np.ones(n, dtype=bool)
                picks.append(rng.choice(np.flatnonzero(row)))
                offset += n
            value = np.asarray(picks, dtype=np.int64)
        elif group.kind == "box":
            value = np.asarray(_subspace(role.action_space, group.path).sample(), dtype=np.float32)
        else:
            value = group.units.sample(mask=group_mask)
        action = _set_path(action, group.path, value)
    return action


def _contains(space: Any, value: Any, what: str, where: str) -> None:
    if space is not None and not space.contains(value):
        raise EnvContractError(
            f"{where}: {what} is not in the role's space {space!r}; check its dtype, shape and bounds"
        )


def _check_spaces(spec: GameSpec, layout: str, result: Any, where: str) -> None:
    """Full ``space.contains`` checks of the acting seats' observations, global states and final obs."""
    for seat in sorted(result.acting):
        role = spec.roles[spec.role_of(layout, seat)]
        _contains(role.observation_space, result.obs[seat], "observation", f"{where}, seat {seat}")
        if role.global_state_space is not None and result.global_state is not None and seat in result.global_state:
            _contains(role.global_state_space, result.global_state[seat], "global_state", f"{where}, seat {seat}")
    if result.truncated and result.final_obs is not None:
        for seat, obs in result.final_obs.items():
            role = spec.roles[spec.role_of(layout, seat)]
            _contains(role.observation_space, obs, "final_obs", f"{where}, seat {seat}")


def _exercise_env(config: ColosseumConfig, spec: GameSpec, layouts: Sequence[str]) -> dict[str, tuple]:
    """Reset every enabled layout and take up to ``VALIDATE_STEPS`` random legal steps under the
    contract checks (``EpisodeTracker``) and full ``space.contains`` checks.

    Returns one ``(obs, mask, global_state)`` sample per role that acted.
    """

    from colosseum.sp2.envs.contract import EpisodeTracker

    rng = np.random.default_rng(0)
    samples: dict[str, tuple] = {}
    env = make_env(config)
    try:
        for layout in layouts:
            where = f"validate, layout {layout}"
            tracker = EpisodeTracker(spec, max_idle_steps=config.env.max_idle_steps, context="validate")
            try:
                result = env.reset(seed=0, layout=layout)
            except EnvContractError:
                raise
            except Exception as e:
                raise ConfigError(f"env.reset(seed=0, layout={layout!r}) of {config.env.env_class!r} failed: "
                                  f"{type(e).__name__}: {e}") from e
            masks = tracker.on_reset(layout, result)
            for step in range(VALIDATE_STEPS + 1):
                _check_spaces(spec, layout, result, f"{where}, step {step}")
                for seat in sorted(result.acting):
                    gs = None if result.global_state is None else result.global_state.get(seat)
                    samples.setdefault(spec.role_of(layout, seat), (result.obs[seat], masks.get(seat), gs))
                if result.episode_over or step == VALIDATE_STEPS:
                    break
                actions = {seat: random_legal_action(spec.roles[spec.role_of(layout, seat)], masks.get(seat), rng)
                           for seat in sorted(result.acting)}
                try:
                    result = env.step(actions)
                except EnvContractError:
                    raise
                except Exception as e:
                    raise ConfigError(f"env.step of {config.env.env_class!r} failed in layout {layout!r}: "
                                      f"{type(e).__name__}: {e}") from e
                masks = tracker.on_step(actions, result)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return samples


def _check_state_batch_dim(state_a: Any, state_b: Any, batches: tuple[int, int], where: str) -> None:
    """The same state built for two batch sizes must differ only in dim 0 of every leaf (SP1)."""
    from colosseum.networks.state import tree_leaves as state_leaves

    leaves_a, leaves_b = state_leaves(state_a), state_leaves(state_b)
    if len(leaves_a) != len(leaves_b):
        raise ConfigError(f"{where}: the state structure depends on the batch size "
                          f"({len(leaves_a)} tensors for B={batches[0]}, {len(leaves_b)} for B={batches[1]})")
    for i, (a, b) in enumerate(zip(leaves_a, leaves_b, strict=True)):
        if a.dim() == 0 or b.dim() == 0 or a.shape[0] != batches[0] or b.shape[0] != batches[1] \
                or a.shape[1:] != b.shape[1:]:
            raise ConfigError(
                f"{where}: every state tensor must have the batch dimension first; state tensor #{i} "
                f"has shape {tuple(a.shape)} for B={batches[0]} and {tuple(b.shape)} for B={batches[1]}. "
                f"Store RNN states as [B, num_layers, H], not nn.LSTM/nn.GRU's native [num_layers, B, H]."
            )


def _check_state_payload(state: Any, where: str) -> None:
    """The model state must survive the inter-process payload round trip (SP1)."""
    from colosseum.networks.state import slice_batch, state_from_numpy, state_to_numpy
    from colosseum.transport.serialization import pack_payload, unpack_payload

    if state is None:
        return
    try:
        state_from_numpy(unpack_payload(*pack_payload(state_to_numpy(slice_batch(state, 0)))))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: initial_state cannot be sent between processes: {e}") from e


def _batch(tree: Any, n: int) -> Any:

    return None if tree is None else tree_to_torch(tree_stack([tree] * n))


def _check_model(model: Any, role: RoleSpec, sample: tuple | None, where: str) -> None:
    """``step`` on observations of the role and ``unroll`` on a synthetic chunk with BOOT/PAD
    slots, a reset and ``global_state`` (spec block 9)."""

    obs_spec = ObsSpec.from_space(role.observation_space)
    action_spec = ActionSpec.from_space(role.action_space)
    gs_spec = None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space)
    obs1 = sample[0] if sample is not None else obs_spec.allocate(())
    mask1 = None
    if action_spec.has_masks:
        mask1 = sample[1] if sample is not None and sample[1] is not None else action_spec.full_mask(())
    gs1 = None
    if gs_spec is not None:
        gs1 = sample[2] if sample is not None and sample[2] is not None else gs_spec.allocate(())
    b, b_alt, s = 2, 3, 4
    model.eval()
    with torch.no_grad():
        try:
            state0, state_alt = model.initial_state(b), model.initial_state(b_alt)
        except Exception as e:
            raise ConfigError(f"{where}: model.initial_state(B) failed: {type(e).__name__}: {e}") from e
        _check_state_batch_dim(state0, state_alt, (b, b_alt), f"{where}: initial_state")
        _check_state_payload(state0, where)
        try:
            out = model.step(_batch(obs1, b), state0, _batch(mask1, b))
            out_alt = model.step(_batch(obs1, b_alt), state_alt, _batch(mask1, b_alt))
        except Exception as e:
            raise ConfigError(f"{where}: model.step failed on a batch of the role's observations: "
                              f"{type(e).__name__}: {e}") from e
        if not isinstance(out.dist, Distribution):
            raise ConfigError(f"{where}: the policy must return a colosseum Distribution "
                              f"(colosseum.sp2.networks.dist), got {type(out.dist).__name__}")
        _check_state_batch_dim(out.state, out_alt.state, (b, b_alt), f"{where}: step() state")
        try:
            actions = out.dist.sample()
            log_probs = out.dist.log_prob(actions)
        except Exception as e:
            raise ConfigError(f"{where}: sampling from the policy distribution {type(out.dist).__name__} "
                              f"failed: {type(e).__name__}: {e}") from e
        expected = tree_to_torch(action_spec.allocate_actions((b,)))
        if not tree_same_structure(actions, expected) or any(
                tuple(x.shape) != tuple(y.shape) for x, y in zip(tree_leaves(actions), tree_leaves(expected),
                                                                 strict=True)):
            raise ConfigError(f"{where}: the policy samples actions that do not match the action space "
                              f"{role.action_space!r}; check the policy head / distribution")
        if tuple(log_probs.shape) != (b,) or not torch.isfinite(log_probs).all():
            raise ConfigError(f"{where}: log_prob of sampled actions must be finite with shape [B]=({b},), "
                              f"got {tuple(log_probs.shape)}")

        # Synthetic chunk, time-major [S=4, B=2]: column 0 = ACT, ACT(terminal), PAD, PAD;
        # column 1 = ACT, ACT, BOOT(truncation, reset after), PAD.
        kinds = torch.tensor([[SLOT_ACT, SLOT_ACT], [SLOT_ACT, SLOT_ACT], [SLOT_PAD, SLOT_BOOT], [SLOT_PAD, SLOT_PAD]])
        act_slot = kinds == SLOT_ACT
        reset_after = torch.tensor([[False, False], [True, False], [False, True], [False, False]])
        one_action = tree_to_numpy(tree_index(actions, 0))
        zero_action = action_spec.allocate_actions(())
        boot_mask = action_spec.boot_mask()

        def seq(act_value: Any, other_value: Any) -> Any:
            if act_value is None:
                return None
            rows = [tree_stack([act_value if act_slot[t, j] else other_value for j in range(b)]) for t in range(s)]
            return tree_to_torch(tree_stack(rows))

        obs_seq = seq(obs1, obs1)
        mask_seq = seq(mask1, boot_mask)
        gs_seq = seq(gs1, gs1)
        actions_seq = seq(one_action, zero_action)
        flat_actions = _flatten_time(actions_seq, s * b)
        try:
            unrolled = model.unroll(obs_seq, model.initial_state(b), reset_after, mask_seq,
                                    global_state=gs_seq, with_value=True)
            unroll_lp = unrolled.dist.log_prob(flat_actions)
            policy_only = model.unroll(obs_seq, model.initial_state(b), reset_after, mask_seq, with_value=False)
        except Exception as e:
            raise ConfigError(f"{where}: model.unroll failed on a synthetic [S={s}, B={b}] chunk with BOOT/PAD "
                              f"slots: {type(e).__name__}: {e}") from e
        acts = act_slot.reshape(-1)
        if unrolled.value is None or tuple(unrolled.value.shape) != (s * b,):
            got = None if unrolled.value is None else tuple(unrolled.value.shape)
            raise ConfigError(f"{where}: unroll(with_value=True) must return time-major values [S*B]=({s * b},), "
                              f"got {got}. Squeeze the last dim in the value head.")
        if tuple(unroll_lp.shape) != (s * b,) or not torch.isfinite(unroll_lp[acts]).all():
            raise ConfigError(f"{where}: unroll gives non-finite or mis-shaped log-probs on ACT slots for actions "
                              f"sampled from step (same masks); step and unroll must agree on the policy")
        if not torch.isfinite(unrolled.value[(kinds != SLOT_PAD).reshape(-1)]).all():
            raise ConfigError(f"{where}: unroll gives non-finite values on ACT or BOOT slots")
        if policy_only.value is not None:
            raise ConfigError(f"{where}: unroll(with_value=False) must return value=None")


def _flatten_time(tree: Any, rows: int) -> Any:

    return tree_map(lambda x: x.reshape(rows, *x.shape[2:]), tree)


def _check_kickstart_teacher(config: ColosseumConfig, agent_configs: dict, role_specs: dict) -> None:
    from colosseum.sp2.coordinator.checkpoint_manager import check_model_state, read_weights_file
    from colosseum.sp2.core.roles import role_signature

    path = config.training.kickstart_teacher
    if not path:
        return
    signatures = {aid: role_signature(role) for aid, role in role_specs.items()}
    if len(set(signatures.values())) > 1:
        raise ConfigError(
            f"training.kickstart_teacher is one global teacher, but the agents {sorted(signatures)} play roles "
            f"with different spaces; train such agents without kickstart (per-agent teachers come in SP3)"
        )
    try:
        teacher_state = read_weights_file(path)
    except ValueError as e:
        raise ConfigError(f"training.kickstart_teacher={path!r}: {e}") from e
    for aid, acfg in agent_configs.items():
        check_model_state(build_model(acfg, role_specs[aid]), teacher_state,
                          f"training.kickstart_teacher={path!r} (agent {aid!r})")


def validate_config(config: ColosseumConfig) -> None:
    """Check a whole config before any process starts (spec block 9).

    - the env's ``GameSpec`` (``env_spec``), the agents' roles (``resolve_agent_roles``) and the
      matchmaking checks (``validate_matchmaking``);
    - ``networks.critic_encoder_class`` only for agents whose roles declare a ``global_state_space``;
    - the kickstart teacher: one signature for all agents' roles, weights that fit every agent;
    - ``reset`` of every enabled layout and a few random legal steps under the contract checks,
      with full ``space.contains`` checks of observations, global states and final observations;
    - every agent's model: ``step`` on its role's observations and ``unroll`` on a synthetic chunk
      with BOOT/PAD slots, a reset and ``global_state``.

    Raises ConfigError (or EnvContractError for an env that breaks the contract).
    """
    from colosseum.sp2.coordinator.matchmaker import enabled_layouts, validate_matchmaking
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    spec = env_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    validate_matchmaking(spec, agent_roles, config.matchmaking)
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_roles}
    role_specs = {aid: agent_role_spec(spec, roles) for aid, roles in agent_roles.items()}
    for aid, acfg in agent_configs.items():
        if acfg.networks.critic_encoder_class and role_specs[aid].global_state_space is None:
            raise ConfigError(
                f"agent {aid!r}: networks.critic_encoder_class is set, but its roles {agent_roles[aid]} declare "
                f"no global_state_space; remove critic_encoder_class or give the roles a global_state_space"
            )
    samples = _exercise_env(config, spec, list(enabled_layouts(spec, config.matchmaking)))
    for aid, acfg in agent_configs.items():
        where = f"agent {aid!r}"
        try:
            model = build_model(acfg, role_specs[aid])
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"{where}: failed to build the model from networks: {type(e).__name__}: {e}") from e
        sample = next((samples[r] for r in agent_roles[aid] if r in samples), None)
        _check_model(model, role_specs[aid], sample, where)
    _check_kickstart_teacher(config, agent_configs, role_specs)
```

Add to `src/colosseum/sp2/core/registry.py` (at the end; replace a placeholder `validate_config` if Part A left one):

```python
def validate_config(config: ColosseumConfig) -> None:
    """Every check that can run before a run starts (spec block 9); see ``colosseum.sp2.core.validation``.

    Callers use this name (``registry.validate_config``), so tests can monkeypatch it here.
    """
    from colosseum.sp2.core.validation import validate_config as _validate_config

    _validate_config(config)
```

(`ColosseumConfig` is the annotation registry.py already uses for `build_model`; with `from __future__ import annotations` a `TYPE_CHECKING` import is enough.)

In `src/colosseum/sp2/launcher.py`, replace the body of `validate_run_config` (old → new):

```python
def validate_run_config(config: ColosseumConfig) -> None:
    """Fail fast (ConfigError / EnvContractError) before any process starts.

    The GameSpec, the agents' roles, the matchmaking checks and one model per agent
    (T6.2 replaces this body with ``registry.validate_config``).
    """
    from colosseum.sp2.coordinator.matchmaker import validate_matchmaking
    from colosseum.sp2.core import registry
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    spec = registry.env_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    validate_matchmaking(spec, agent_roles, config.matchmaking)
    for aid, roles in agent_roles.items():
        registry.build_model(config.get_agent_config(aid), agent_role_spec(spec, roles))
```

```python
def validate_run_config(config: ColosseumConfig) -> None:
    """Fail fast (ConfigError / EnvContractError) before any process starts: ``registry.validate_config``
    (looked up at call time, so tests can count or replace it)."""
    from colosseum.sp2.core import registry

    registry.validate_config(config)
```

Insert into `src/colosseum/sp2/cli.py`, directly above the final `if __name__ == "__main__":` line:

```python
@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set env.max_idle_steps=200)." + _SET_HELP_YAML)
def validate_cmd(config: str, overrides: tuple[str, ...]) -> None:
    """Validate a config: GameSpec, roles, matchmaking, env steps under the contract, every agent's model."""
    with _config_errors():
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import validate_config

        cfg = load_config(config, _parse_overrides(overrides) or None)
        validate_config(cfg)
        for aid in cfg.get_trainable_agent_ids():
            click.echo(f"  OK: agent '{aid}'")
    click.echo("Config is valid.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_validate.py tests/unit/test_sp2_launcher_checkpoints.py tests/integration/test_sp2_game_runs.py -q`
Expected: all passed (the launcher now runs the full validation; the toy-game runs still start).

- [ ] **Step 5: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/sp2/core/validation.py src/colosseum/sp2/core/registry.py src/colosseum/sp2/launcher.py \
        src/colosseum/sp2/cli.py tests/unit/test_sp2_validate.py
git commit -m "feat: SP2 validate_config with env steps, model step/unroll checks and CLI validate (SP2 T6.2)"
git push origin sp2-game-model
```

---

### Task T6.3: Offline BC on trees, CLI `bc --agent`

Spec block 6, BC paragraph: `colosseum bc` gets `--agent A` (networks and roles from that agent's config; default: the only trainable agent); data are trees (`observations`, `actions`, optional `action_masks`, `dones`) in the agent's spaces; the loss is the joint `-log_prob` for K = 1 and the mean over valid deciders for K > 1; the model runs without its value path (`step`, or `unroll(with_value=False)` for stateful models). SP1's behavior stays: windows for stateful models, normalizers updated once per sample, data errors as one `Config error:` line, and the SP1 residual (spec block 0, item 2): an unreadable file (`PermissionError` / `OSError` from `torch.load`) is a `DataError`, not a traceback.

**Files:**
- Create: `src/colosseum/sp2/bc/__init__.py` (empty; skip if Part B created it for `kickstart.py`), `src/colosseum/sp2/bc/offline_bc.py`
- Modify: `src/colosseum/sp2/cli.py` (add the `bc` command)
- Create: `tests/learning/game_learning_envs.py` (shared learning kit: MLP model and a masked expert game)
- Test: `tests/unit/test_sp2_bc_trainer.py`, `tests/integration/test_sp2_bc_cli.py`, `tests/learning/test_sp2_bc_learns.py`

**Interfaces:**
- Consumes: `ActionSpec` (`num_deciders`, `has_masks`, `has_units`, `groups`, `allocate_actions`, `full_mask`), `ObsSpec.allocate` (T1.3); tree utilities `tree_get`, `tree_index`, `tree_leaves`, `tree_map`, `tree_paths`, `tree_stack` (T1.1); `Distribution.log_prob/unit_log_prob/unit_valid/mode` (T2.1); `PolicyModel.step/unroll(with_value=False)/initial_state/update_normalizers/is_stateful`, `act` (T2.3); `BaseEncoder`, `BasePolicy`, `BaseValue`, `ComposedModel`, `CategoricalDist` (T2.1, T2.3); `GameSpec.solo`, `StepResult`, `MultiAgentEnv` (T1.4); `DataError` (`colosseum.core.errors`); `validate_config` (T6.2); `make_test_model` (T2.3).
- Produces:
  - *`OfflineBCTrainer(model, action_spec, obs_spec, lr=1e-3, device="cpu", seq_len=64)` with `add_data(observations, actions, action_masks=None, dones=None)`, `load_data(path) -> int`, `train(num_epochs=10, batch_size=256, log_interval=1) -> {"bc_loss", "bc_loss_first_epoch", "num_epochs", "num_samples", "accuracy"?}`, `num_samples`, `model`; *`per_sample_nll(dist, actions, num_deciders) -> (nll [B], valid [B])`; `_window_index` (SP1).
  - CLI: `python -m colosseum.sp2 bc -c cfg -d data -o out.pt [--agent A] [--epochs] [--batch-size] [--lr] [--seq-len]`.
  - `tests/learning/game_learning_envs.py`: `MLPEncoder`, `MLPPolicy`, `MLPValue`, `make_mlp_model(obs_dim, num_actions, hidden=32)`, `MaskedChoiceGame` (extended in T7.2).

- [ ] **Step 1: Create the learning kit**

Create `tests/learning/game_learning_envs.py`:

```python
"""Tiny solo games that the SP2 pipeline must solve in seconds, and a small MLP model for them."""
from __future__ import annotations

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.networks.cores import NoCore
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.sp2.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import CategoricalDist


class MLPEncoder(BaseEncoder):
    """[B, obs_dim] -> Linear -> relu -> [B, hidden]."""

    def __init__(self, obs_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self._latent = hidden
        self.fc = nn.Linear(obs_dim, hidden)

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(obs.float()))


class MLPPolicy(BasePolicy):
    def __init__(self, in_dim: int, num_actions: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_actions)

    def forward(self, features: torch.Tensor, aux: dict) -> CategoricalDist:
        return CategoricalDist(self.fc(features))


class MLPValue(BaseValue):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


def make_mlp_model(obs_dim: int, num_actions: int, hidden: int = 32) -> ComposedModel:
    """SP1's ``make_simple_model`` shape: Linear+relu encoder, no core, linear heads."""
    return ComposedModel(MLPEncoder(obs_dim, hidden), NoCore(hidden), MLPPolicy(hidden, num_actions), MLPValue(hidden))


class MaskedChoiceGame(MultiAgentEnv):
    """Solo, 8 decisions per episode. Observation: one-hot context c in 0..3. Legal actions:
    {c, (c + 1) % 4}; the expert plays c (reward 1), the other legal action gives 0."""

    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(4))
    LENGTH = 8

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0
        self._t = 0

    def expert_action(self) -> int:
        return self._context

    def _turn(self, rewards: dict) -> StepResult:
        self._context = int(self._rng.integers(4))
        obs = np.zeros(4, np.float32)
        obs[self._context] = 1.0
        mask = np.zeros(4, dtype=bool)
        mask[[self._context, (self._context + 1) % 4]] = True
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask}, rewards=rewards)

    def reset(self, seed, layout) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._turn({})

    def step(self, actions) -> StepResult:
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return self._turn({0: reward})
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp2_bc_trainer.py`:

```python
"""Offline BC on observation/action trees (T6.3)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import DataError
from colosseum.sp2.bc import offline_bc as bc_module
from colosseum.sp2.bc.offline_bc import OfflineBCTrainer, _window_index, per_sample_nll
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_stack
from colosseum.sp2.envs.game import RoleSpec
from game_helpers import make_test_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")

BOX = gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32)
DICT_OBS = gymnasium.spaces.Dict({"grid": gymnasium.spaces.Box(0, 255, (2, 2), np.uint8),
                                  "vec": gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)})
FOUR = gymnasium.spaces.Discrete(4)


def trainer_for(role: RoleSpec, core: str = "none", seq_len: int = 4, lr: float = 1e-2) -> OfflineBCTrainer:
    torch.manual_seed(0)
    return OfflineBCTrainer(make_test_model(role, core=core), ActionSpec.from_space(role.action_space),
                            ObsSpec.from_space(role.observation_space), lr=lr, seq_len=seq_len)


def samples(space, n: int, seed: int = 0):
    space.seed(seed)
    return tree_stack([space.sample() for _ in range(n)])


def test_window_index_tiles_with_offset_and_padding():
    assert _window_index(5, 2).tolist() == [[0, 1], [2, 3], [4, -1]]
    assert _window_index(5, 2, offset=1).tolist() == [[0, -1], [1, 2], [3, 4]]
    assert _window_index(0, 3).shape == (0, 3)


class _FixedDist:
    def __init__(self, unit_lp, unit_valid):
        self._lp, self._valid = unit_lp, unit_valid

    def log_prob(self, actions):
        return torch.where(self._valid, self._lp, torch.zeros_like(self._lp)).sum(-1)

    def unit_log_prob(self, actions):
        return self._lp

    def unit_valid(self, actions):
        return self._valid


def test_nll_is_joint_for_one_decider_and_a_mean_over_valid_deciders_otherwise():
    lp = torch.tensor([[-1.0, -3.0, -5.0], [-2.0, -4.0, -6.0]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    nll, ok = per_sample_nll(_FixedDist(lp, valid), None, num_deciders=3)
    assert nll[0].item() == pytest.approx(2.0) and ok.tolist() == [True, False]
    joint, ok1 = per_sample_nll(_FixedDist(lp[:, :1], torch.ones(2, 1, dtype=torch.bool)), None, num_deciders=1)
    assert joint.tolist() == [1.0, 2.0] and ok1.all()


def test_stateless_training_on_dict_observations_keeps_dtypes():
    role = RoleSpec(DICT_OBS, FOUR)
    trainer = trainer_for(role)
    obs = samples(DICT_OBS, 64)
    actions = np.full(64, 2, dtype=np.int64)
    trainer.add_data(obs, actions, dones=np.arange(64) % 8 == 7)
    assert trainer._observations[0]["grid"].dtype == torch.uint8  # kept until the model casts
    metrics = trainer.train(num_epochs=20, batch_size=16)
    assert metrics["bc_loss"] < metrics["bc_loss_first_epoch"] and metrics["accuracy"] > 0.9


def test_stateful_training_uses_windows_with_resets():
    role = RoleSpec(BOX, FOUR)
    trainer = trainer_for(role, core="lstm", seq_len=4)
    obs = samples(BOX, 40)
    trainer.add_data(obs, np.arange(40) % 4, np.ones((40, 4), dtype=bool), np.arange(40) % 5 == 4)
    metrics = trainer.train(num_epochs=3, batch_size=8)
    assert np.isfinite(metrics["bc_loss"]) and "accuracy" in metrics


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"actions": np.zeros(8, np.float32)}, "floating point"),
    ({"actions": np.zeros((8, 2), np.int64)}, "actions leaf"),
    ({"observations": {"x": np.zeros((8, 3), np.float32)}}, "observations has leaves"),
    ({"observations": np.zeros((8, 4), np.float32)}, "observations leaf"),
    ({"action_masks": np.ones((8, 3), dtype=bool)}, "action_masks leaf"),
    ({"dones": np.zeros(7, dtype=bool)}, "dones"),
], ids=["float-actions", "action-shape", "obs-structure", "obs-shape", "mask-shape", "dones"])
def test_data_that_does_not_fit_the_agent_is_a_data_error(kwargs, message):
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    data = {"observations": samples(BOX, 8), "actions": np.zeros(8, np.int64),
            "action_masks": np.ones((8, 4), dtype=bool), "dones": None, **kwargs}
    with pytest.raises(DataError, match=message):
        trainer.add_data(data["observations"], data["actions"], data["action_masks"], data["dones"])


def test_masks_must_be_given_for_every_file_or_none():
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    trainer.add_data(samples(BOX, 4), np.zeros(4, np.int64), np.ones((4, 4), dtype=bool))
    with pytest.raises(DataError, match="every batch/file"):
        trainer.add_data(samples(BOX, 4), np.zeros(4, np.int64))


def test_masks_on_a_box_action_space_are_rejected():
    box_actions = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)
    trainer = trainer_for(RoleSpec(BOX, box_actions))
    with pytest.raises(DataError, match="nothing to mask"):
        trainer.add_data(samples(BOX, 4), np.zeros((4, 2), np.float32), np.ones((4, 2), dtype=bool))


def test_illegal_expert_actions_under_their_masks_fail_loudly():
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    masks = np.zeros((8, 4), dtype=bool)
    masks[:, 1] = True
    trainer.add_data(samples(BOX, 8), np.zeros(8, np.int64), masks)
    with pytest.raises(ValueError, match="illegal under their action_masks"):
        trainer.train(num_epochs=1, batch_size=8)


def test_unreadable_files_are_data_errors(tmp_path, monkeypatch):
    trainer = trainer_for(RoleSpec(BOX, FOUR))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a torch file")
    with pytest.raises(DataError, match="garbage.pt"):
        trainer.load_data(garbage)

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(bc_module.torch, "load", denied)
    with pytest.raises(DataError, match="PermissionError"):
        trainer.load_data(garbage)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataError, match="no .pt files"):
        trainer.load_data(empty)


def test_files_load_as_trees(tmp_path):
    role = RoleSpec(DICT_OBS, FOUR)
    path = tmp_path / "data.pt"
    obs = {k: torch.as_tensor(v) for k, v in samples(DICT_OBS, 16).items()}
    torch.save({"observations": obs, "actions": torch.zeros(16, dtype=torch.long), "extra": torch.zeros(1)}, path)
    trainer = trainer_for(role)
    assert trainer.load_data(path) == 16 and trainer.num_samples == 16
```

Create `tests/integration/test_sp2_bc_cli.py`:

```python
"""`python -m colosseum.sp2 bc --agent A` (T6.3)."""
from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

from colosseum.sp2.cli import main
from colosseum.sp2.core.registry import build_model
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_map, tree_stack
from game_helpers import agent_role_of, make_test_config, write_test_config


def write_data(path, config, agent_id: str = "agent_0", n: int = 64) -> None:
    """Random decisions in the agent's spaces, all-legal masks, an episode end every 5."""
    _roles, role = agent_role_of(config, agent_id)
    role.observation_space.seed(0)
    role.action_space.seed(0)
    obs = tree_stack([role.observation_space.sample() for _ in range(n)])
    actions = tree_stack([role.action_space.sample() for _ in range(n)])
    spec = ActionSpec.from_space(role.action_space)
    data = {"observations": tree_map(torch.as_tensor, obs), "actions": tree_map(torch.as_tensor, actions),
            "dones": torch.arange(n) % 5 == 4}
    if spec.has_masks:
        data["action_masks"] = tree_map(torch.as_tensor, spec.full_mask((n,)))
    torch.save(data, path)


def bc(*args):
    return CliRunner().invoke(main, ["bc", *map(str, args)])


def test_bc_trains_and_saves_weights_that_load_into_the_agent_model(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    cfg = make_test_config("turns")
    write_data(tmp_path / "data.pt", cfg)
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", out, "--epochs", 2, "--batch-size", 16)
    assert result.exit_code == 0, result.output
    assert "final-epoch NLL" in result.output and "(agent_0)" in result.output
    _roles, role = agent_role_of(cfg, "agent_0")
    build_model(cfg.get_agent_config("agent_0"), role).load_state_dict(torch.load(out, weights_only=True))


def test_agent_selects_networks_and_roles(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "asymmetric")
    cfg = make_test_config("asymmetric")
    write_data(tmp_path / "prey.pt", cfg, agent_id="prey")
    out = tmp_path / "prey_bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out, "--agent", "prey", "--epochs", 1)
    assert result.exit_code == 0, result.output
    _roles, prey_role = agent_role_of(cfg, "prey")
    build_model(cfg.get_agent_config("prey"), prey_role).load_state_dict(torch.load(out, weights_only=True))

    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out)
    assert result.exit_code == 1 and "--agent" in result.stderr
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", out, "--agent", "nobody")
    assert result.exit_code == 1 and "nobody" in result.stderr
    result = bc("-c", cfg_path, "-d", tmp_path / "prey.pt", "-o", tmp_path / "x.pt", "--agent", "hunter")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")  # prey data for the hunter


def test_bc_rejects_non_positive_seq_len(tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    write_data(tmp_path / "data.pt", make_test_config("turns"))
    result = bc("-c", cfg_path, "-d", tmp_path / "data.pt", "-o", tmp_path / "x.pt", "--seq-len", 0)
    assert result.exit_code == 2 and "--seq-len" in result.output


def _garbage(tmp_path):
    path = tmp_path / "data.pt"
    path.write_bytes(b"not a torch file")
    return path


def _missing_actions(tmp_path):
    path = tmp_path / "data.pt"
    torch.save({"observations": torch.rand(8, 3)}, path)
    return path


def _empty_dir(tmp_path):
    path = tmp_path / "no_data"
    path.mkdir()
    return path


def _float_actions(tmp_path):
    path = tmp_path / "data.pt"
    write_data(path, make_test_config("turns"))
    data = torch.load(path, weights_only=True)
    data["actions"] = tree_map(lambda x: x.float() + 0.5, data["actions"])
    torch.save(data, path)
    return path


@pytest.mark.parametrize(("make_data", "message"), [
    (_garbage, "data.pt"),
    (_missing_actions, "missing BC data keys ['actions']"),
    (_float_actions, "floating point"),
    (_empty_dir, "no .pt files"),
], ids=["unreadable", "missing-key", "float-actions", "empty-dir"])
def test_bad_data_is_a_one_line_config_error(make_data, message, tmp_path):
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", make_data(tmp_path), "-o", out)
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:") and message in result.stderr, result.stderr
    assert len(result.stderr.strip().splitlines()) == 1 and "Traceback" not in result.output
    assert not out.exists()


def test_a_data_file_that_cannot_be_opened_is_a_one_line_config_error(tmp_path, monkeypatch):
    """SP1 residual: PermissionError / OSError from torch.load used to print a traceback."""
    import colosseum.sp2.bc.offline_bc as bc_module

    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns")
    data = tmp_path / "data.pt"
    write_data(data, make_test_config("turns"))

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(data))

    monkeypatch.setattr(bc_module.torch, "load", denied)
    result = bc("-c", cfg_path, "-d", data, "-o", tmp_path / "bc.pt")
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    assert "PermissionError" in result.stderr and len(result.stderr.strip().splitlines()) == 1
```

Create `tests/learning/test_sp2_bc_learns.py` (SP1's `test_bc_learns.py` guarantee on the new pipeline):

```python
"""BC on a masked scripted expert reaches high accuracy and plays well (SP1 guarantee on SP2, T6.3)."""
from __future__ import annotations

import numpy as np
import torch

from colosseum.sp2.bc.offline_bc import OfflineBCTrainer
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.networks.model import act
from game_learning_envs import MaskedChoiceGame, make_mlp_model


def record_expert(num_decisions: int, seed: int) -> dict:
    env = MaskedChoiceGame()
    result = env.reset(seed, "solo")
    data = {"observations": [], "actions": [], "action_masks": [], "dones": []}
    for _ in range(num_decisions):
        action = env.expert_action()
        data["observations"].append(result.obs[0])
        data["actions"].append(action)
        data["action_masks"].append(result.action_masks[0])
        result = env.step({0: action})
        data["dones"].append(result.episode_over)
        if result.episode_over:
            result = env.reset(None, "solo")
    return {
        "observations": np.stack(data["observations"]),
        "actions": np.asarray(data["actions"], dtype=np.int64),
        "action_masks": np.stack(data["action_masks"]),
        "dones": np.asarray(data["dones"]),
    }


def test_bc_imitates_a_masked_scripted_expert(restore_global_rng):
    torch.manual_seed(0)
    spec = MaskedChoiceGame.spec.roles["player"]
    model = make_mlp_model(obs_dim=4, num_actions=4)
    trainer = OfflineBCTrainer(model, ActionSpec.from_space(spec.action_space),
                               ObsSpec.from_space(spec.observation_space), lr=1e-2)
    trainer.add_data(**record_expert(2000, seed=0))
    metrics = trainer.train(num_epochs=30, batch_size=256)
    assert metrics["accuracy"] >= 0.95

    env = MaskedChoiceGame()
    result = env.reset(123, "solo")
    rewards = []
    model.eval()
    for _ in range(500):
        obs = torch.as_tensor(result.obs[0])[None]
        mask = torch.as_tensor(result.action_masks[0])[None]
        out = act(model, obs, model.initial_state(1), mask, deterministic=True)
        result = env.step({0: int(out.actions.reshape(-1)[0])})
        rewards.append(result.rewards.get(0, 0.0))
        if result.episode_over:
            result = env.reset(None, "solo")
    assert np.mean(rewards) >= 0.9
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_bc_trainer.py tests/integration/test_sp2_bc_cli.py tests/learning/test_sp2_bc_learns.py -q`
Expected: collection errors `No module named 'colosseum.sp2.bc.offline_bc'`.

- [ ] **Step 4: Write the trainer**

Create `src/colosseum/sp2/bc/offline_bc.py`:

```python
"""Offline behavioral cloning (BC) on observation and action trees.

Trains a :class:`~colosseum.sp2.networks.model.PolicyModel` by maximum likelihood on recorded
expert decisions of one agent (``colosseum bc --agent A``): the model and the spaces come from
that agent's config and roles.

Data format: one or more ``.pt`` files (``torch.save`` of a dict) with
- ``observations``: the role's observation tree (a tensor, or a dict of tensors for Dict
  spaces), every leaf ``[N, *leaf_shape]``; dtypes are kept (the model casts);
- ``actions``: the role's action tree with a leading ``[N]`` (``ActionSpec.allocate_actions``
  layout: int64 for discrete parts, float32 for boxes, ``Units`` groups as ``[N, U, ...]``);
- ``action_masks`` (optional): the mask tree (``ActionSpec.full_mask`` layout) with a leading
  ``[N]``, bool, True = legal;
- ``dones`` (optional): ``[N]`` bool, True when decision t ends its episode.

Loss = minus log-prob of the expert action: the joint log-prob when the action has one decider
(K = 1); with K > 1 (``Units``) the mean of ``unit_log_prob`` over the valid deciders, so a
decision weighs the same whatever its number of units. Decisions without a valid decider are
skipped. The model runs without its value path (``step``, or ``unroll(with_value=False)``).

Stateless models train on shuffled decisions. Stateful models (``model.is_stateful``) train
with ``unroll`` over contiguous windows of ``seq_len`` decisions, from ``initial_state`` and
with the state reset after every ``done``; a random window offset per epoch moves the cut
points. The last decision of every file (``add_data`` call) ends an episode. Observation
normalizers are updated once per sample, before the first epoch of ``train``.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from colosseum.core.errors import DataError
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_get, tree_index, tree_leaves, tree_map, tree_paths
from colosseum.sp2.networks.dist import Distribution
from colosseum.sp2.networks.model import PolicyModel

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"observations", "actions"}
_KNOWN_KEYS = _REQUIRED_KEYS | {"action_masks", "dones"}
_EVAL_BATCH = 4096


def _window_index(n: int, seq_len: int, offset: int = 0) -> torch.Tensor:
    """``[W, seq_len]`` dataset indices of consecutive windows; -1 marks padding.

    The windows tile ``range(n)`` in order: ``[0, offset)`` when ``offset > 0``, then
    ``[offset, offset + seq_len)``, and so on; the last window may be shorter.
    """
    if n <= 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    starts = list(range(offset, n, seq_len)) if offset > 0 else list(range(0, n, seq_len))
    if offset > 0:
        starts = [0] + starts
    ends = starts[1:] + [n]
    index = torch.full((len(starts), seq_len), -1, dtype=torch.long)
    for w, (start, end) in enumerate(zip(starts, ends)):
        index[w, : end - start] = torch.arange(start, end)
    return index


def per_sample_nll(dist: Distribution, actions: Any, num_deciders: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(nll [B], valid [B] bool)``: joint NLL for K = 1, mean NLL over valid deciders for K > 1."""
    if num_deciders == 1:
        nll = -dist.log_prob(actions)
        return nll, torch.ones_like(nll, dtype=torch.bool)
    unit_lp = dist.unit_log_prob(actions)
    unit_valid = dist.unit_valid(actions)
    count = unit_valid.sum(dim=-1)
    total = torch.where(unit_valid, unit_lp, torch.zeros_like(unit_lp)).sum(dim=-1)
    return -total / count.clamp(min=1).to(total.dtype), count > 0


def _as_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _as_tree(v) for k, v in value.items()}
    return torch.as_tensor(value)


def _leaf_shapes(tree: Any) -> dict[tuple[str, ...], tuple[int, ...]]:
    return {path: tuple(leaf.shape) for path, leaf in zip(tree_paths(tree), tree_leaves(tree), strict=True)}


def _check_tree(name: str, data: Any, expected: Any, n: int) -> None:
    """``data`` must have the paths of ``expected`` (allocated with a leading 1), a leading ``n``
    and the per-leaf trailing shapes of ``expected``."""
    got, want = _leaf_shapes(data), _leaf_shapes(expected)
    if set(got) != set(want):
        raise DataError(f"BC {name} has leaves {sorted(got)}, but the agent's space needs {sorted(want)}")
    for path, shape in want.items():
        if got[path][:1] != (n,) or got[path][1:] != shape[1:]:
            where = "/".join(path) or "<root>"
            raise DataError(f"BC {name} leaf {where} has shape {got[path]}, expected ({n}, *{shape[1:]})")


def _discrete_hits(spec: ActionSpec, mode: Any, actions: Any) -> torch.Tensor | None:
    """1.0 where the mode equals the expert on every discrete non-units group; None if there is none."""
    if spec.has_units:
        return None
    hits = None
    for group in spec.groups:
        if group.kind not in ("discrete", "multi_discrete"):
            continue
        same = tree_get(mode, group.path) == tree_get(actions, group.path)
        if same.dim() > 1:
            same = same.all(dim=-1)
        hits = same if hits is None else hits & same
    return None if hits is None else hits.float()


class OfflineBCTrainer:
    """Supervised behavioral cloning from offline data (module docstring).

    Usage::

        trainer = OfflineBCTrainer(model, action_spec, obs_spec, lr=1e-3, seq_len=64)
        trainer.load_data("expert_games/")      # or add_data(obs_tree, action_tree, mask_tree, dones)
        metrics = trainer.train(num_epochs=10, batch_size=256)
    """

    def __init__(self, model: PolicyModel, action_spec: ActionSpec, obs_spec: ObsSpec, lr: float = 1e-3,
                 device: str | torch.device = "cpu", seq_len: int = 64) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._action_spec = action_spec
        self._obs_spec = obs_spec
        self._num_deciders = int(action_spec.num_deciders)
        self._seq_len = int(seq_len)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._observations: list[Any] = []
        self._actions: list[Any] = []
        self._masks: list[Any] = []
        self._dones: list[torch.Tensor] = []
        self._normalized_upto = 0

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def num_samples(self) -> int:
        return sum(int(tree_leaves(o)[0].shape[0]) for o in self._observations)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def add_data(self, observations: Any, actions: Any, action_masks: Any = None, dones: Any = None) -> None:
        """Add ``N`` expert decisions (trees with a leading ``N``; module docstring)."""
        obs = _as_tree(observations)
        acts = _as_tree(actions)
        n = int(tree_leaves(obs)[0].shape[0]) if tree_leaves(obs) else 0
        if n == 0:
            raise DataError("BC data is empty")
        _check_tree("observations", obs, self._obs_spec.allocate((1,)), n)
        expected_actions = self._action_spec.allocate_actions((1,))
        _check_tree("actions", acts, expected_actions, n)
        for path in tree_paths(expected_actions):
            want, got = tree_get(expected_actions, path), tree_get(acts, path)
            if want.dtype.kind in "iu" and got.is_floating_point():
                where = "/".join(path) or "<root>"
                raise DataError(
                    f"BC actions leaf {where} is floating point but that part of the action space is "
                    f"discrete; store discrete actions as integers"
                )
        acts = tree_map(lambda x: x.long() if not x.is_floating_point() else x.float(), acts)
        masks = None
        if action_masks is not None:
            if not self._action_spec.has_masks:
                raise DataError("BC data has action_masks but the agent's action space has nothing to mask; "
                                "drop 'action_masks' from the data")
            masks = tree_map(lambda x: x.bool(), _as_tree(action_masks))
            _check_tree("action_masks", masks, self._action_spec.full_mask((1,)), n)
        if self._masks and (masks is None) != (self._masks[0] is None):
            raise DataError("BC data: either every batch/file has 'action_masks' or none does")
        if dones is None:
            if self._model.is_stateful:
                logger.warning("BC data has no 'dones': treating these %d decisions as one episode "
                               "(this matters only for stateful models, which this one is)", n)
            done_t = torch.zeros(n, dtype=torch.bool)
        else:
            done_t = torch.as_tensor(dones).bool().reshape(-1).clone()
            if done_t.shape[0] != n:
                raise DataError(f"BC data: {n} observations but {done_t.shape[0]} dones")
        done_t[-1] = True  # never carry state across add_data calls
        self._observations.append(obs)
        self._actions.append(acts)
        self._masks.append(masks)
        self._dones.append(done_t)

    def load_data(self, path: str | Path) -> int:
        """Load one ``.pt`` file or every ``*.pt`` in a directory; returns decisions loaded."""
        path = Path(path)
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]
        if not files:
            raise DataError(f"no .pt files in {path}")
        total = 0
        for f in files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=True)
            except (EOFError, RuntimeError, OSError, pickle.UnpicklingError) as e:
                detail = (str(e).strip().splitlines() or [""])[0]
                raise DataError(f"{f}: not a readable torch.save file ({type(e).__name__}: {detail})") from e
            if not isinstance(data, dict):
                raise DataError(f"{f}: expected a dict with keys {sorted(_KNOWN_KEYS)}")
            missing = _REQUIRED_KEYS - set(data)
            if missing:
                raise DataError(f"{f}: missing BC data keys {sorted(missing)} (need observations, actions)")
            unknown = set(data) - _KNOWN_KEYS
            if unknown:
                logger.warning("%s: ignoring unknown BC data keys %s", f, sorted(unknown))
            before = self.num_samples
            self.add_data(data["observations"], data["actions"], data.get("action_masks"), data.get("dones"))
            total += self.num_samples - before
            logger.info("Loaded %d BC decisions from %s", self.num_samples - before, f)
        logger.info("BC dataset: %d decisions", self.num_samples)
        return total

    def _dataset(self) -> tuple[Any, Any, Any, torch.Tensor]:
        def cat(trees: list[Any]) -> Any:
            return tree_map(lambda *xs: torch.cat(xs), *trees)

        masks = None if self._masks[0] is None else cat(self._masks)
        return cat(self._observations), cat(self._actions), masks, torch.cat(self._dones)

    def _to_device(self, tree: Any) -> Any:
        return None if tree is None else tree_map(lambda x: x.to(self._device), tree)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 10, batch_size: int = 256, log_interval: int = 1) -> dict[str, float]:
        """Run BC; returns the final-epoch NLL ``bc_loss`` and, for discrete actions, ``accuracy``."""
        if not self._observations:
            raise ValueError("No training data. Call add_data() or load_data() first.")
        if num_epochs < 1 or batch_size < 1:
            raise ValueError("num_epochs and batch_size must be >= 1")
        obs, actions, masks, dones = self._dataset()
        n = int(dones.shape[0])
        self._update_normalizers(obs, n)
        stateful = self._model.is_stateful
        logger.info("BC training: %d decisions, %d epochs, batch_size=%d, %s", n, num_epochs, batch_size,
                    f"stateful (seq_len={self._seq_len})" if stateful else "stateless")
        self._model.train()
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            if stateful:
                loss = self._sequence_epoch(obs, actions, masks, dones, batch_size)
            else:
                loss = self._flat_epoch(obs, actions, masks, n, batch_size)
            epoch_losses.append(loss)
            if (epoch + 1) % log_interval == 0:
                logger.info("BC epoch %d/%d: nll=%.4f", epoch + 1, num_epochs, loss)
        metrics = {"bc_loss": epoch_losses[-1], "bc_loss_first_epoch": epoch_losses[0],
                   "num_epochs": float(num_epochs), "num_samples": float(n)}
        accuracy = self._accuracy(obs, actions, masks, dones, n)
        if accuracy is not None:
            metrics["accuracy"] = accuracy
        return metrics

    def _update_normalizers(self, obs: Any, n: int) -> None:
        """Feed every decision added since the last call to ``model.update_normalizers`` once."""
        for start in range(self._normalized_upto, n, _EVAL_BATCH):
            idx = torch.arange(start, min(start + _EVAL_BATCH, n))
            self._model.update_normalizers(self._to_device(tree_index(obs, idx)))
        self._normalized_upto = n

    def _weighted_nll(self, dist: Distribution, actions: Any, weights: torch.Tensor,
                      masked: bool) -> tuple[torch.Tensor, torch.Tensor]:
        nll, valid = per_sample_nll(dist, actions, self._num_deciders)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        active = weights > 0
        nan = torch.isnan(nll) & active
        if nan.any():
            raise RuntimeError(
                f"BC NLL is NaN for {int(nan.sum())} samples: training diverged or the data contains "
                f"NaN/inf (check observations/actions, lower the learning rate)"
            )
        inf = torch.isinf(nll) & active
        if inf.any():
            if masked:
                raise ValueError(f"{int(inf.sum())} BC expert actions have zero probability under the policy: "
                                 f"they are illegal under their action_masks (fix the data)")
            raise ValueError(f"{int(inf.sum())} BC expert actions have zero probability under the policy "
                             f"(no action_masks are given: an out-of-support action or an underflowing std)")
        nll = torch.where(active, nll, torch.zeros_like(nll))
        return (nll * weights).sum(), weights.sum()

    def _step_dist(self, obs: Any, masks: Any, batch: int) -> Distribution:
        state0 = self._model.initial_state(batch, self._device)
        return self._model.step(self._to_device(obs), state0, self._to_device(masks)).dist

    def _flat_epoch(self, obs: Any, actions: Any, masks: Any, n: int, batch_size: int) -> float:
        total, count = 0.0, 0.0
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            dist = self._step_dist(tree_index(obs, idx), None if masks is None else tree_index(masks, idx), len(idx))
            weights = torch.ones(len(idx), device=self._device)
            loss_sum, weight = self._weighted_nll(dist, self._to_device(tree_index(actions, idx)), weights,
                                                  masks is not None)
            if float(weight) > 0:
                self._optimizer.zero_grad()
                (loss_sum / weight).backward()
                self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    def _sequence_forward(self, index: torch.Tensor, obs: Any, masks: Any,
                          dones: torch.Tensor) -> tuple[Distribution, torch.Tensor]:
        """Unroll windows ``index [L, b]``; returns (dist over L*b time-major rows, weights)."""
        valid = index >= 0
        safe = index.clamp(min=0)
        o = self._to_device(tree_index(obs, safe))                          # leaves [L, b, ...]
        reset_after = (dones[safe] | ~valid).to(self._device)               # [L, b]
        m = None if masks is None else self._to_device(tree_index(masks, safe))
        state0 = self._model.initial_state(index.shape[1], self._device)
        dist = self._model.unroll(o, state0, reset_after, m, with_value=False).dist
        return dist, valid.reshape(-1).to(self._device).float()

    def _window_actions(self, actions: Any, index: torch.Tensor) -> Any:
        flat = index.clamp(min=0).reshape(-1)
        return self._to_device(tree_index(actions, flat))

    def _sequence_epoch(self, obs: Any, actions: Any, masks: Any, dones: torch.Tensor, batch_size: int) -> float:
        seq_len = self._seq_len
        offset = int(torch.randint(0, seq_len, (1,)).item())
        windows = _window_index(int(dones.shape[0]), seq_len, offset)
        per_batch = max(1, batch_size // seq_len)
        order = torch.randperm(len(windows))
        total, count = 0.0, 0.0
        for start in range(0, len(windows), per_batch):
            index = windows[order[start:start + per_batch]].t()          # [L, b]
            dist, weights = self._sequence_forward(index, obs, masks, dones)
            loss_sum, weight = self._weighted_nll(dist, self._window_actions(actions, index), weights,
                                                  masks is not None)
            if float(weight) > 0:
                self._optimizer.zero_grad()
                (loss_sum / weight).backward()
                self._optimizer.step()
            total += float(loss_sum.detach())
            count += float(weight)
        return total / max(count, 1.0)

    @torch.no_grad()
    def _accuracy(self, obs: Any, actions: Any, masks: Any, dones: torch.Tensor, n: int) -> float | None:
        self._model.eval()
        try:
            correct, total = 0.0, 0.0
            if self._model.is_stateful:
                windows = _window_index(n, self._seq_len, 0)
                per_batch = max(1, _EVAL_BATCH // self._seq_len)
                for start in range(0, len(windows), per_batch):
                    index = windows[start:start + per_batch].t()
                    dist, weights = self._sequence_forward(index, obs, masks, dones)
                    hits = _discrete_hits(self._action_spec, dist.mode(), self._window_actions(actions, index))
                    if hits is None:
                        return None
                    correct += float((hits * weights).sum())
                    total += float(weights.sum())
            else:
                for start in range(0, n, _EVAL_BATCH):
                    idx = torch.arange(start, min(start + _EVAL_BATCH, n))
                    dist = self._step_dist(tree_index(obs, idx), None if masks is None else tree_index(masks, idx),
                                           len(idx))
                    hits = _discrete_hits(self._action_spec, dist.mode(), self._to_device(tree_index(actions, idx)))
                    if hits is None:
                        return None
                    correct += float(hits.sum())
                    total += float(len(idx))
            return correct / max(total, 1.0)
        finally:
            self._model.train()
```

- [ ] **Step 5: Add the CLI command**

Insert into `src/colosseum/sp2/cli.py`, directly above the final `if __name__ == "__main__":` line:

```python
@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="BC data: a .pt file or a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones; trees in the agent's spaces)")
@click.option("--output", "-o", required=True, type=click.Path(), help="Where to save the trained state_dict (.pt)")
@click.option("--agent", "-a", "agent", default=None,
              help="Agent whose networks and roles are trained (default: the config's only trainable agent)")
@click.option("--epochs", default=10, type=int, show_default=True, help="Number of BC epochs")
@click.option("--batch-size", default=256, type=int, show_default=True, help="Decisions per gradient step")
@click.option("--lr", default=1e-3, type=float, show_default=True, help="Adam learning rate")
@click.option("--seq-len", default=None, type=click.IntRange(min=1),
              help="Window length for stateful models (default: bc.seq_len from the config, 64)")
def bc(
    config: str,
    data: str,
    output: str,
    agent: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int | None,
) -> None:
    """Train one agent's policy by offline behavioral cloning (loss = -log pi(a|s), masks applied)."""
    import logging

    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with _config_errors():
        from colosseum.core.errors import ConfigError
        from colosseum.sp2.bc.offline_bc import OfflineBCTrainer
        from colosseum.sp2.core.config import load_config
        from colosseum.sp2.core.registry import build_model, env_spec, validate_config
        from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles
        from colosseum.sp2.core.specs import ActionSpec, ObsSpec

        cfg = load_config(config)
        validate_config(cfg)
        agent_ids = cfg.get_trainable_agent_ids()
        if agent is None:
            if len(agent_ids) != 1:
                raise ConfigError(f"the config has {len(agent_ids)} trainable agents {agent_ids}; "
                                  f"choose one with --agent")
            agent = agent_ids[0]
        elif agent not in agent_ids:
            raise ConfigError(f"--agent {agent!r} is not a trainable agent of the config ({agent_ids})")
        spec = env_spec(cfg)
        role = agent_role_spec(spec, resolve_agent_roles(cfg, spec)[agent])
        agent_cfg = cfg.get_agent_config(agent)
        model = build_model(agent_cfg, role)
    device = agent_cfg.learner.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = OfflineBCTrainer(
        model, ActionSpec.from_space(role.action_space), ObsSpec.from_space(role.observation_space),
        lr=lr, device=device, seq_len=seq_len if seq_len is not None else agent_cfg.bc.seq_len,
    )
    with _config_errors():  # unreadable or malformed data, actions that do not fit the policy (DataError)
        trainer.load_data(data)
        metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    message = f"BC training complete ({agent}): final-epoch NLL={metrics['bc_loss']:.4f}"
    if "accuracy" in metrics:
        message += f", accuracy={metrics['accuracy']:.3f}"
    click.echo(message)
    click.echo(f"Weights saved to {output}")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_bc_trainer.py tests/integration/test_sp2_bc_cli.py tests/learning/test_sp2_bc_learns.py -q`
Expected: all passed (the learning test takes a few seconds).

- [ ] **Step 7: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/sp2/bc src/colosseum/sp2/cli.py tests/learning/game_learning_envs.py \
        tests/unit/test_sp2_bc_trainer.py tests/integration/test_sp2_bc_cli.py tests/learning/test_sp2_bc_learns.py
git commit -m "feat: SP2 offline BC on observation/action trees with bc --agent (SP2 T6.3)"
git push origin sp2-game-model
```

---

### Task T6.4: Distributed mode on chunk v2

Spec block 10, distributed bullet, and criterion 8: `serve-weight-store`, `run-learner` and `run-workers` work on chunk v2 and `Lineup`s in SP1's scope (self-play on the latest weights, no coordinator). Supported: games where every agent of the run plays every role of the enabled layouts; each worker env gets a fixed lineup — a layout drawn by `matchmaking.layouts`, every seat the latest weights of one agent (agents in rotation). Anything else is a `ConfigError` pointing to SP5. Chunks still travel as one byte blob (`.proto` unchanged: its `behavior_policy_version` field carries `policy_version`). Distributed checkpoints carry roles and the role signature.

**Files:**
- Create (copies by script): `src/colosseum/sp2/distributed.py`, `src/colosseum/sp2/transport/__init__.py`, `src/colosseum/sp2/transport/serialization.py`, `src/colosseum/sp2/transport/grpc_transport.py`, `src/colosseum/sp2/weight_store/__init__.py`, `src/colosseum/sp2/weight_store/grpc_store.py`
- Modify: `src/colosseum/sp2/cli.py` (add `run-learner`, `run-workers`, `serve-weight-store`)
- Test: `tests/unit/test_sp2_serialization.py` (copy of `tests/unit/test_serialization.py` + edits), `tests/unit/test_sp2_distributed_roles.py`, `tests/integration/test_sp2_grpc.py`, `tests/integration/test_sp2_distributed_e2e.py`

**Interfaces:**
- Consumes: `TrajectoryChunk` (all fields; `to_payload()` keys equal the field names, see Contract notes), `SLOT_ACT`, `SLOT_BOOT`, `WeightPayload`, `Lineup`, `SeatAssignment` (T3.1); `rollout_worker_process(..., agent_roles, lineups, max_env_steps, max_idle_steps)` (T3.4); `learner_process` (T4.4); `APPO`, `KickstartLoss` (T4.2); `validate_config` (T6.2); `enabled_layouts` (T5.1); `CheckpointManager` (T5.3); `RunDir` (T5.4); `_create_env` (`colosseum.sp2.launcher`, T5.4); `chunk_v2_payload(S=4, version=0, agent_id="a", *, pattern=None, num_actions=3)` (Part B's T4.4 test kit in `tests/game_helpers.py`); unchanged `colosseum.transport.colosseum_pb2*`, `colosseum.transport.base`, `colosseum.weight_store.base`, `colosseum.weight_store.shared_memory`.
- Produces:
  - `run_distributed_learner(config_path, agent_id, traj_port, weight_store_address, overrides=None) -> int`, `run_distributed_workers(config_path, weight_store_address, learner_addresses, overrides=None) -> int`, `workers_role()`, `GRPCTrajectorySink`, `GRPCWeightSource`, `GRPCWeightSink` (SP1 names), *`DistributedSetup`, *`distributed_setup(config, agent_ids)`, *`distributed_lineups(setup, agent_ids, num_envs, rng)`.
  - `colosseum.sp2.transport.serialization`: SP1's `pack_payload`, `unpack_payload`, `payload_byte_cap`, `serialize_state_dict`, `deserialize_state_dict`, `validate_state_dict_payload`, `serialize_chunk_payload`, `deserialize_chunk_payload`, `serialize_chunk`; v2 `validate_chunk_payload(payload)`, `deserialize_chunk(agent_id, policy_version, data, compressed)`.
  - `colosseum.sp2.transport.grpc_transport` (`serve_trajectory_receiver`, `TrajectoryServicer`, `GRPCTransport`) and `colosseum.sp2.weight_store.grpc_store` (`serve_weight_store`, `GRPCWeightStore`) with SP1 behavior on the `sp2` types.

- [ ] **Step 1: Write the failing tests**

(a) Create `tests/unit/test_sp2_serialization.py` from SP1's serialization tests (the hostile-input tests stay as they are; the chunk tests use chunk v2; new `validate_chunk_payload` cases). Save as `/tmp/make_sp2_serialization_test.py` and run it:

```python
"""T6.4: tests/unit/test_sp2_serialization.py from SP1's tests/unit/test_serialization.py."""
import ast
import textwrap
from pathlib import Path

src = Path("tests/unit/test_serialization.py").read_text()
for old, new in [
    ('"""gRPC serialization of numpy payloads (T2.2; wire format changes in SP5)."""',
     '"""gRPC serialization of numpy payloads with chunk v2 (SP1 guarantees, T6.4; wire format changes in SP5)."""'),
    ("from colosseum.core.types import TrajectoryChunk\n", "from colosseum.sp2.core.types import TrajectoryChunk\n"),
    ("from colosseum.transport.serialization import (\n", "from colosseum.sp2.transport.serialization import (\n"),
    ("from dataflow_helpers import chunk_payload\n", "from game_helpers import chunk_v2_payload\n"),
    ("    from colosseum.transport.serialization import validate_state_dict_payload\n",
     "    from colosseum.sp2.transport.serialization import validate_state_dict_payload\n"),
    ('''    assert b'"test_serialization"' in data
    tampered = data.replace(b'"test_serialization"', b'"no_such_mod_xyz123"')  # same length''',
     '''    assert b'"test_sp2_serialization"' in data
    tampered = data.replace(b'"test_sp2_serialization"', b'"no_such_module_xyz1234"')  # same length'''),
    ('    with pytest.raises(ValueError, match="no_such_mod_xyz123"):',
     '    with pytest.raises(ValueError, match="no_such_module_xyz1234"):'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)

new_tests = {
    "test_chunk_payload_bytes_roundtrip_with_state": '''
def test_chunk_payload_bytes_roundtrip_with_state():
    payload = chunk_v2_payload(version=5)
    payload["initial_state"] = {"mem": np.ones((1, 2, 3), np.float32), "len": np.array([2])}
    data, compressed = serialize_chunk_payload(payload)
    back = deserialize_chunk_payload(data, compressed)
    validate_chunk_payload(back)
    chunk = TrajectoryChunk.from_payload(back)
    assert chunk.policy_version == 5
    assert torch.equal(chunk.initial_state["len"], torch.tensor([2]))
    assert chunk.action_masks.dtype == torch.bool and chunk.kind.dtype == torch.int8
''',
    "test_serialize_chunk_convenience_roundtrip": '''
def test_serialize_chunk_convenience_roundtrip():
    chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=1))
    data, compressed = serialize_chunk(chunk)
    back = deserialize_chunk("a", 9, data, compressed)
    assert torch.equal(back.obs, chunk.obs) and torch.equal(back.kind, chunk.kind)
    assert back.policy_version == 9 and back.agent_id == "a"
''',
    "test_namedtuple_state_survives_the_wire": '''
def test_namedtuple_state_survives_the_wire():
    """Every State node type round-trips, namedtuples included."""
    state = {"core": HC(torch.randn(1, 1, 8), torch.randn(1, 1, 8)),
             "extra": [torch.ones(1, 2), (torch.zeros(1),)]}
    payload = chunk_v2_payload()
    payload["initial_state"] = state_to_numpy(state)
    back = TrajectoryChunk.from_payload(deserialize_chunk_payload(*serialize_chunk_payload(payload))).initial_state
    assert type(back["core"]) is HC
    assert torch.equal(back["core"].h, state["core"].h) and torch.equal(back["core"].c, state["core"].c)
    assert isinstance(back["extra"], list) and isinstance(back["extra"][1], tuple)
    assert torch.equal(back["extra"][0], state["extra"][0])
''',
}
tree = ast.parse(src)
lines = src.splitlines(keepends=True)
for node in sorted((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in new_tests),
                   key=lambda n: n.lineno, reverse=True):
    lines[node.lineno - 1:node.end_lineno] = [textwrap.dedent(new_tests[node.name]).lstrip("\n")]
src = "".join(lines)
src = src.replace("    unpack_payload,\n)\n", "    unpack_payload,\n    validate_chunk_payload,\n)\n", 1)
src += '''

# ---------------------------------------------------------------------------
# Chunk v2 payload validation (the servicer's guard)
# ---------------------------------------------------------------------------


def test_validate_chunk_payload_accepts_trees_and_rejects_bad_fields():
    good = chunk_v2_payload()  # S = 4
    good["obs"] = {"grid": np.zeros((4, 2, 2), np.uint8), "vec": np.zeros((4, 3), np.float32)}
    validate_chunk_payload(good)
    for field, value, message in [
        ("kind", None, "kind"),
        ("reward", np.zeros(3, np.float32), "reward"),
        ("obs", {"grid": np.zeros((3, 2), np.uint8)}, "obs"),
        ("obs", [np.zeros(4)], "obs"),
        ("policy_version", "1", "policy_version"),
        ("behavior_unit_logp", np.zeros(4, np.float32), "behavior_unit_logp"),
    ]:
        bad = {**chunk_v2_payload(), field: value}
        with pytest.raises(ValueError, match=message):
            validate_chunk_payload(bad)
'''
Path("tests/unit/test_sp2_serialization.py").write_text(src)
print("wrote tests/unit/test_sp2_serialization.py")
```

```bash
.venv/bin/python /tmp/make_sp2_serialization_test.py
.venv/bin/ruff check --fix tests/unit/test_sp2_serialization.py && .venv/bin/ruff check tests/unit/test_sp2_serialization.py
```

(b) Create `tests/unit/test_sp2_distributed_roles.py` (the distributed-role tests of SP1's `test_run_dir_logging.py` and `test_launcher_lifecycle.py`, plus the scope check and the fixed lineups):

```python
"""Distributed roles on SP2: scope check, fixed lineups, run dirs, exit codes (T6.4)."""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import signal
from collections import Counter

import pytest

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import load_config
from game_helpers import make_test_config, write_test_config


class _ExitedProcess:
    """mp.Process stand-in that has already exited with ``exitcode``."""

    exitcode = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


@pytest.mark.parametrize(("host", "role"), [
    ("node-1", "workers-node-1"),
    ("my host.local/x", "workers-my-host.local-x"),
    ("-.odd", "workers-odd"),
    ("", "workers-host"),
])
def test_workers_role_includes_the_sanitized_hostname(monkeypatch, host, role):
    import colosseum.sp2.distributed as distributed

    monkeypatch.setattr(distributed.socket, "gethostname", lambda: host)
    assert distributed.workers_role() == role


def test_distributed_lineups_follow_layout_weights_and_rotate_agents():
    from colosseum.sp2.distributed import distributed_lineups, distributed_setup

    cfg = make_test_config("ffa", agents={"a": {}, "b": {}}, matchmaking={"layouts": {"2p": 0.25, "4p": 0.75}})
    setup = distributed_setup(cfg, ["a", "b"])
    lineups = distributed_lineups(setup, ["a", "b"], 4000, random.Random(0))
    counts = Counter(lu.layout for lu in lineups)
    assert abs(counts["2p"] / 4000 - 0.25) < 0.05
    for e, lineup in enumerate(lineups[:6]):
        expected = "ab"[e % 2]
        assert all(s.agent_id == expected and s.network_id == "latest" and s.collect for s in lineup.seats)
        assert len(lineup.seats) == len(setup.spec.layouts[lineup.layout])


def test_asymmetric_agents_are_refused_with_a_pointer_to_sp5(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.sp2.distributed as distributed

    path = write_test_config(tmp_path / "asym.yaml", "asymmetric")
    overrides = {"run.dir": str(tmp_path / "runs")}
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_learner(str(path), "hunter", 0, "localhost:1", overrides=overrides)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_workers(str(path), "localhost:1", {"hunter": "localhost:2"}, overrides)
    assert not (tmp_path / "runs").exists()


def test_workers_entry_point_records_the_base_run_name(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.sp2.distributed as distributed

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    monkeypatch.setattr(distributed.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(distributed.socket, "gethostname", lambda: "node 7")
    path = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}) == 0
    (root,) = (tmp_path / "runs").iterdir()
    resolved = load_config(root / "config.resolved.yaml")
    assert re.fullmatch(r"cfg-\d{8}-\d{6}", resolved.run.name)
    assert root.name == f"{resolved.run.name}-workers-node-7"


@pytest.mark.parametrize(("worker_exit", "expected"), [(0, 0), (3, 1)])
def test_run_workers_returns_an_exit_code(worker_exit, expected, tmp_path, monkeypatch, restore_root_logging,
                                          capsys):
    import colosseum.sp2.distributed as distributed

    class Exited(_ExitedProcess):
        exitcode = worker_exit

    monkeypatch.setattr(mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.mp, "Process", Exited)
    path = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})
    assert distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}) == expected
    assert ("worker-0 died (exit 3)" in capsys.readouterr().err) == (worker_exit == 3)


@pytest.fixture
def fake_learner_role(tmp_path, monkeypatch):
    """run_distributed_learner without gRPC: returns the config path."""
    import colosseum.sp2.transport.grpc_transport as grpc_transport
    import colosseum.sp2.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    return write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs")})


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_run_learner_returns_128_plus_signum(sig, fake_learner_role, monkeypatch, restore_root_logging):
    import colosseum.sp2.distributed as distributed
    import colosseum.sp2.learner.learner as learner_module

    def fake_learner_process(*, stop_event, **kwargs):
        os.kill(os.getpid(), sig)
        assert stop_event.wait(10)

    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    assert distributed.run_distributed_learner(str(fake_learner_role), "agent_0", 0, "localhost:1") == 128 + sig
    assert {s: signal.getsignal(s) for s in before} == before


def test_run_learner_setup_failure_leaves_signal_handlers_untouched(fake_learner_role, monkeypatch,
                                                                     restore_root_logging, restore_global_rng):
    import colosseum.sp2.distributed as distributed

    def failing_seed(seed):
        raise RuntimeError("seeding failed")

    monkeypatch.setattr(distributed, "apply_global_seed", failing_seed)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="seeding failed"):
        distributed.run_distributed_learner(str(fake_learner_role), "agent_0", 0, "localhost:1")
    assert {s: signal.getsignal(s) for s in before} == before
```

(c) Create `tests/integration/test_sp2_grpc.py` (SP1's `test_grpc.py` and `test_distributed.py` adapters on chunk v2):

```python
"""gRPC weight store, trajectory transport with chunk v2, and the distributed adapters (T6.4)."""
from __future__ import annotations

import queue
import socket
import time

import numpy as np
import pytest
import torch

from colosseum.sp2.core.types import TrajectoryChunk, WeightPayload
from game_helpers import chunk_v2_payload

grpc = pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _state_dict() -> dict[str, np.ndarray]:
    return {"w": np.random.randn(4, 4).astype(np.float32), "b": np.random.randn(4).astype(np.float32)}


def test_weight_store_roundtrip_and_adapters():
    from colosseum.sp2.distributed import GRPCWeightSink, GRPCWeightSource
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    time.sleep(0.3)
    store = GRPCWeightStore(f"localhost:{port}")
    try:
        assert store.get("agent_0") is None and store.get_version("agent_0") == -1
        sink, source = GRPCWeightSink(store, "agent_0"), GRPCWeightSource(store, "agent_0")
        with pytest.raises(queue.Empty):
            source.get_nowait()
        sd1 = _state_dict()
        sink.put_nowait(WeightPayload("agent_0", 1, sd1))
        payload = source.get_nowait()
        assert payload.policy_version == 1 and np.allclose(payload.state_dict["w"], sd1["w"])
        with pytest.raises(queue.Empty):  # no newer version: do not re-pull
            source.get_nowait()
        sink.put_nowait(WeightPayload("agent_0", 2, _state_dict()))
        assert source.get_nowait().policy_version == 2 and store.get_version("agent_0") == 2
    finally:
        store.close()
        server.stop(0)


def test_trajectory_transport_carries_chunk_v2_trees():
    from colosseum.sp2.distributed import GRPCTrajectorySink
    from colosseum.sp2.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue: queue.Queue = queue.Queue(maxsize=16)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    time.sleep(0.3)
    transport = GRPCTransport(f"localhost:{port}")
    try:
        chunk = TrajectoryChunk.from_payload(chunk_v2_payload(version=7, agent_id="agent_0"))
        chunk.obs = {"grid": torch.randint(0, 255, (4, 2, 2), dtype=torch.uint8), "vec": chunk.obs}
        GRPCTrajectorySink(transport, "agent_0").put(chunk, timeout=1.0)
        received = TrajectoryChunk.from_payload(chunk_queue.get(timeout=2.0))
        assert received.policy_version == 7 and received.agent_id == "agent_0"
        assert received.obs["grid"].dtype == torch.uint8 and torch.equal(received.obs["grid"], chunk.obs["grid"])
        assert torch.equal(received.kind, chunk.kind) and torch.equal(received.action_masks, chunk.action_masks)
        assert transport.send_chunks_batch("agent_0", [chunk_v2_payload(), chunk_v2_payload(version=1)]) == 2
    finally:
        transport.close()
        server.stop(0)


def test_servicer_rejects_a_chunk_without_obs():
    from colosseum.sp2.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver

    port = _free_port()
    chunk_queue: queue.Queue = queue.Queue(maxsize=8)
    server = serve_trajectory_receiver(chunk_queue, port=port)
    transport = GRPCTransport(f"localhost:{port}")
    try:
        bad = chunk_v2_payload()
        del bad["obs"]
        with pytest.raises(grpc.RpcError) as err:
            transport.send_chunk("agent_0", bad)
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "obs" in err.value.details()
        assert chunk_queue.empty()
        transport.send_chunk("agent_0", chunk_v2_payload())  # a good chunk still goes through
        assert TrajectoryChunk.from_payload(chunk_queue.get(timeout=2.0)).num_slots == 4
    finally:
        transport.close()
        server.stop(0)


def test_trajectory_sink_tolerates_a_dead_learner():
    from colosseum.sp2.distributed import GRPCTrajectorySink
    from colosseum.sp2.transport.grpc_transport import GRPCTransport

    transport = GRPCTransport(f"localhost:{_free_port()}")  # nothing listening
    GRPCTrajectorySink(transport, "agent_0").put(chunk_v2_payload(), timeout=0.5)
    transport.close()


def test_weight_store_rejects_non_array_weights():
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    port = _free_port()
    server = serve_weight_store(port=port)
    client = GRPCWeightStore(f"localhost:{port}")
    try:
        with pytest.raises(grpc.RpcError) as err:
            client.put("agent_0", WeightPayload("agent_0", 1, {"w": np.zeros((2, 2), np.float32), "b": [1.0]}))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT and "'b'" in err.value.details()
        assert client.get("agent_0") is None
    finally:
        client.close()
        server.stop(0)
```

(d) Create `tests/integration/test_sp2_distributed_e2e.py`:

```python
"""End-to-end distributed (gRPC) self-play on localhost: weight store + learner + workers (T6.4)."""
from __future__ import annotations

import json
import multiprocessing as mp
import socket
import time

import pytest
import torch

from game_helpers import write_test_config

pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for(condition, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(0.1)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("localhost", port)) == 0


@pytest.mark.timeout(600)
def test_distributed_grpc_pipeline(tmp_path, restore_root_logging):
    """The learner trains on chunk v2 from gRPC workers, publishes weights and saves signed checkpoints."""
    from colosseum.sp2.distributed import run_distributed_learner, run_distributed_workers, workers_role
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    ws_port, traj_port = _free_port(), _free_port()
    ws_addr, learner_addr = f"localhost:{ws_port}", f"localhost:{traj_port}"
    agent = "agent_0"
    cfg_path = str(write_test_config(tmp_path / "turns.yaml", "turns"))
    overrides = {
        "training.total_timesteps": 1200, "rollout.num_workers": 2, "rollout.envs_per_worker": 4,
        "rollout.chunk_length": 8, "rollout.weight_sync_interval_sec": 0.5, "learner.batch_chunks": 2,
        "learner.queue_size": 32, "checkpoint.interval": 1, "run.dir": str(tmp_path / "runs"), "run.name": "e2e",
    }
    learner_run = tmp_path / "runs" / f"e2e-learner-{agent}"
    workers_run = tmp_path / "runs" / f"e2e-{workers_role()}"
    ckpt_root = learner_run / "checkpoints" / agent
    ws_server = serve_weight_store(port=ws_port)
    learner = mp.Process(target=run_distributed_learner, args=(cfg_path, agent, traj_port, ws_addr, overrides),
                         daemon=False)
    learner.start()
    client = GRPCWeightStore(ws_addr)
    try:
        _wait_for(lambda: _port_open(traj_port), 60, "the learner's TrajectoryService to bind")
        assert run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides) == 0
        _wait_for(lambda: client.get_version(agent) > 0, 60, "a trained weight version in the store")
        _wait_for(lambda: any(ckpt_root.glob("ckpt_v*/meta.json")), 60, "a checkpoint saved by the learner")
        version = client.get_version(agent)
    finally:
        learner.terminate()
        learner.join(timeout=10)
        if learner.is_alive():
            learner.kill()
            learner.join()
        client.close()
        ws_server.stop(0)
    assert version > 0
    newest = max((m.parent for m in ckpt_root.glob("ckpt_v*/meta.json")), key=lambda d: int(d.name[len("ckpt_v"):]))
    assert all(isinstance(v, torch.Tensor) for v in torch.load(newest / "model.pt", weights_only=True).values())
    meta = json.loads((newest / "meta.json").read_text())
    assert meta["roles"] and isinstance(meta["role_signature"], str) and meta["env_steps"] is None
    for run in (learner_run, workers_run):
        assert (run / "config.resolved.yaml").is_file()
    for worker_id in range(2):
        log = (workers_run / "logs" / f"worker-{worker_id}.log").read_text()
        assert f"worker-{worker_id} started (pid" in log and f"worker-{worker_id} finished" in log
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_serialization.py tests/unit/test_sp2_distributed_roles.py tests/integration/test_sp2_grpc.py tests/integration/test_sp2_distributed_e2e.py -q`
Expected: collection errors `No module named 'colosseum.sp2.transport'` / `'colosseum.sp2.distributed'` (the gRPC tests are skipped instead if `grpc` is not installed; it is in the dev setup).

- [ ] **Step 3: Create the copies**

Save as `/tmp/make_sp2_distributed.py` and run it. The worker and learner roles are shown in full; the adapters and `workers_role` are copied unchanged.

```python
"""T6.4: create the sp2 copies of distributed.py, transport/serialization.py, transport/grpc_transport.py
and weight_store/grpc_store.py. Run from the repo root: .venv/bin/python <this script>, then
``.venv/bin/ruff check --fix`` on the four outputs (sorts the imports this script adds).
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path


class Copy:
    """Text of one copied module with ast-based function replacement."""

    def __init__(self, src: str, dst: str) -> None:
        self.dst = Path(dst)
        self.lines = Path(src).read_text().splitlines(keepends=True)

    def replace(self, name: str, new_code: str) -> None:
        tree = ast.parse("".join(self.lines))
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        code = textwrap.dedent(new_code).strip("\n") + "\n" if new_code else ""
        self.lines = "".join(self.lines[:start] + [code] + self.lines[node.end_lineno:]).splitlines(keepends=True)

    def text(self, old: str, new: str) -> None:
        text = "".join(self.lines)
        assert text.count(old) == 1, f"expected exactly one occurrence of: {old!r}"
        self.lines = text.replace(old, new).splitlines(keepends=True)

    def write(self) -> None:
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.dst.write_text("".join(self.lines))
        print(f"wrote {self.dst}")


for package in ("src/colosseum/sp2/transport", "src/colosseum/sp2/weight_store"):
    init = Path(package) / "__init__.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    init.touch()

# --- transport/serialization.py: chunk v2 payloads -------------------------------------------
ser = Copy("src/colosseum/transport/serialization.py", "src/colosseum/sp2/transport/serialization.py")
ser.text("from colosseum.core.types import TrajectoryChunk\n", "from colosseum.sp2.core.types import TrajectoryChunk\n")
ser.text('SP1 wire format (replaced in SP5):', 'Wire format (SP1, unchanged in SP2; replaced in SP5):')
ser.replace("validate_chunk_payload", '''
_CHUNK_FLAT = ("kind", "reward", "terminal", "reset_after", "behavior_logp")
_CHUNK_TREES = ("obs", "actions")
_CHUNK_OPTIONAL_TREES = ("global_state", "action_masks")


def _tree_arrays(tree: Any, name: str) -> list[np.ndarray]:
    """Leaves of a numpy payload tree (dicts with str keys); ValueError on anything else."""
    if isinstance(tree, np.ndarray):
        return [tree]
    if isinstance(tree, dict) and tree and all(isinstance(k, str) for k in tree):
        return [leaf for value in tree.values() for leaf in _tree_arrays(value, name)]
    raise ValueError(f"chunk payload field {name!r} must be a numpy array or a dict tree of them, "
                     f"got {type(tree).__name__}")


def validate_chunk_payload(payload: Any) -> None:
    """Raise ValueError unless ``payload`` is a well-formed chunk v2 payload.

    Checks the fields, their types and the common slot dimension ``S`` of every flat array and
    tree leaf, then builds the chunk once (which checks ``initial_state`` node types).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"chunk payload must be a dict, got {type(payload).__name__}")
    if not isinstance(payload.get("agent_id"), str) or type(payload.get("policy_version")) is not int:
        raise ValueError("chunk payload needs a str agent_id and an int policy_version")
    for name in _CHUNK_FLAT:
        value = payload.get(name)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(f"chunk payload field {name!r} must be a 1-D numpy array [S]")
    num_slots = payload["kind"].shape[0]
    for name in _CHUNK_FLAT:
        if payload[name].shape[0] != num_slots:
            raise ValueError(f"chunk payload field {name!r} has shape {payload[name].shape}, expected [S={num_slots}]")
    trees = [(name, payload.get(name)) for name in _CHUNK_TREES]
    trees += [(name, payload.get(name)) for name in _CHUNK_OPTIONAL_TREES if payload.get(name) is not None]
    unit_logp = payload.get("behavior_unit_logp")
    if unit_logp is not None:
        if not isinstance(unit_logp, np.ndarray) or unit_logp.ndim != 2:
            raise ValueError("chunk payload field 'behavior_unit_logp' must be None or a numpy array [S, K]")
        trees.append(("behavior_unit_logp", unit_logp))
    for name, tree in trees:
        if tree is None:
            raise ValueError(f"chunk payload field {name!r} is missing")
        for leaf in _tree_arrays(tree, name):
            if leaf.ndim == 0 or leaf.shape[0] != num_slots:
                raise ValueError(f"chunk payload field {name!r} has a leaf of shape {leaf.shape}, "
                                 f"expected S={num_slots} first")
    try:
        TrajectoryChunk.from_payload(payload)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed chunk payload: {type(e).__name__}: {e}") from e
''')
ser.replace("deserialize_chunk", '''
def deserialize_chunk(
    agent_id: str,
    policy_version: int,
    data: bytes,
    compressed: bool,
) -> TrajectoryChunk:
    """Deserialize bytes from :func:`serialize_chunk` into a TrajectoryChunk."""
    payload = deserialize_chunk_payload(data, compressed)
    payload["agent_id"] = agent_id
    payload["policy_version"] = int(policy_version)
    return TrajectoryChunk.from_payload(payload)
''')
ser.write()

# --- transport/grpc_transport.py ---------------------------------------------------------------
grpc_t = Copy("src/colosseum/transport/grpc_transport.py", "src/colosseum/sp2/transport/grpc_transport.py")
grpc_t.text("from colosseum.core.types import TrajectoryChunk\n", "from colosseum.sp2.core.types import TrajectoryChunk\n")
grpc_t.text("from colosseum.transport.serialization import (\n", "from colosseum.sp2.transport.serialization import (\n")
grpc_t.text('                payload["behavior_policy_version"] = int(proto_chunk.behavior_policy_version)\n',
            '                payload["policy_version"] = int(proto_chunk.behavior_policy_version)  # proto name kept\n')
grpc_t.text('            behavior_policy_version=int(payload["behavior_policy_version"]),\n',
            '            behavior_policy_version=int(payload["policy_version"]),\n')
grpc_t.write()

# --- weight_store/grpc_store.py ----------------------------------------------------------------
store = Copy("src/colosseum/weight_store/grpc_store.py", "src/colosseum/sp2/weight_store/grpc_store.py")
store.text("from colosseum.core.types import WeightPayload\n", "from colosseum.sp2.core.types import WeightPayload\n")
store.text("from colosseum.transport.serialization import (\n", "from colosseum.sp2.transport.serialization import (\n")
store.write()

# --- distributed.py ----------------------------------------------------------------------------
dist = Copy("src/colosseum/distributed.py", "src/colosseum/sp2/distributed.py")
dist.text('''Scope: distributed mode currently runs self-play with the latest policy of each
agent (opponents = latest weights pulled from the store). The dynamic
coordinator-driven matchmaking (PFSP / historical-checkpoint opponents, C1) is a
single-machine feature; closing that loop across machines would require running
the coordinator as its own service and is left as future work.''', '''Scope (SP2, spec block 10): self-play on the latest weights, without a coordinator, for games
where every agent of the run plays every role of the enabled layouts. Each worker env gets a
fixed lineup: a layout drawn by ``matchmaking.layouts`` and every seat the latest weights of one
agent (agents in rotation over the envs). Anything else (asymmetric agents, leagues across
machines) is a ConfigError pointing to SP5.''')
dist.text("from colosseum.core.config import ColosseumConfig, config_hash, load_config\n",
          "from colosseum.sp2.core.config import ColosseumConfig, config_hash, load_config\n")
dist.text("from colosseum.core.run_dir import RunDir, safe_path_component\n",
          "from colosseum.sp2.core.run_dir import RunDir, safe_path_component\n")
dist.text("from colosseum.core.types import WeightPayload\n",
          "from colosseum.sp2.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment, WeightPayload\n"
          "import random\n")
dist.text("from functools import partial\n", "from functools import partial\nfrom dataclasses import dataclass\n"
          "from colosseum.core.errors import ConfigError\nfrom colosseum.sp2.envs.game import GameSpec, RoleSpec\n")

dist.replace("run_distributed_learner", '''
@dataclass(frozen=True)
class DistributedSetup:
    """Spec, layouts and per-agent roles / role specs of a distributed role."""

    spec: GameSpec
    layouts: dict[str, float]
    agent_roles: dict[str, list[str]]
    role_specs: dict[str, RoleSpec]


def distributed_setup(config: ColosseumConfig, agent_ids: list[str]) -> DistributedSetup:
    """Validate the config and check the distributed scope (module docstring); ConfigError otherwise."""
    from colosseum.sp2.coordinator.matchmaker import enabled_layouts
    from colosseum.sp2.core.registry import env_spec, validate_config
    from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles

    validate_config(config)
    spec = env_spec(config)
    all_roles = resolve_agent_roles(config, spec)
    unknown = sorted(set(agent_ids) - set(all_roles))
    if unknown:
        raise ConfigError(f"agents {unknown} are not trainable agents of the config ({sorted(all_roles)})")
    layouts = enabled_layouts(spec, config.matchmaking)
    needed = {seat.role for name in layouts for seat in spec.layouts[name]}
    for aid in agent_ids:
        missing = sorted(needed - set(all_roles[aid]))
        if missing:
            raise ConfigError(
                f"distributed mode supports only games where every agent plays every role of the enabled "
                f"layouts; agent {aid!r} does not play {missing}. Asymmetric agents and leagues across "
                f"machines come with SP5 (train such configs with 'train' on one machine)"
            )
    return DistributedSetup(
        spec=spec, layouts=layouts,
        agent_roles={aid: list(all_roles[aid]) for aid in agent_ids},
        role_specs={aid: agent_role_spec(spec, all_roles[aid]) for aid in agent_ids},
    )


def distributed_lineups(setup: DistributedSetup, agent_ids: list[str], num_envs: int,
                        rng: random.Random) -> list[Lineup]:
    """Fixed lineups of one worker: env ``e`` plays a layout drawn by the layout weights, every
    seat the latest weights of ``agent_ids[e % n]`` (collecting)."""
    names = list(setup.layouts)
    weights = [setup.layouts[name] for name in names]
    lineups = []
    for e in range(num_envs):
        layout = rng.choices(names, weights=weights, k=1)[0]
        agent_id = agent_ids[e % len(agent_ids)]
        seats = [SeatAssignment(agent_id, LATEST_NETWORK_ID, True) for _ in setup.spec.layouts[layout]]
        lineups.append(Lineup(layout=layout, seats=seats))
    return lineups


def run_distributed_learner(
    config_path: str,
    agent_id: str,
    traj_port: int,
    weight_store_address: str,
    overrides: dict | None = None,
) -> int:
    """Run one trainable agent's learner as a standalone gRPC service; returns the exit code
    (0 when it stopped at its budget, 128 + signum after SIGINT / SIGTERM).

    Starts a TrajectoryService on ``traj_port`` (workers send chunks here), trains with the
    configured algorithm, and pushes weights to the WeightStore at ``weight_store_address``.
    """
    from colosseum.core.threads import configure_torch_threads, resolve_learner_threads
    from colosseum.sp2.core.registry import build_model, import_class
    from colosseum.sp2.core.roles import role_signature
    from colosseum.sp2.core.specs import ActionSpec
    from colosseum.sp2.learner.learner import learner_process, resolve_device
    from colosseum.sp2.transport.grpc_transport import serve_trajectory_receiver
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore

    setup_process_logging(None, f"learner-{agent_id}", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    setup = distributed_setup(config, [agent_id])  # before the run dir exists: a bad config leaves nothing
    acfg = config.get_agent_config(agent_id)
    role_spec = setup.role_specs[agent_id]
    run_dir = RunDir.create(config, config_path, role=f"learner-{agent_id}")
    config = run_dir.with_run_name(config)
    setup_process_logging(run_dir.logs, f"learner-{agent_id}", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    max_mb = config.transport.grpc_max_message_mb

    device = resolve_device(acfg.learner.device)
    # The learner role does not know which workers share its machine, so with
    # learner.torch_threads unset it assumes none (num_workers=0).
    configure_torch_threads(resolve_learner_threads(
        acfg.learner.torch_threads, device, num_workers=0,
        worker_threads=acfg.rollout.torch_threads, num_learners=1,
    ))

    # Trajectory inbox filled by the gRPC server, drained by learner_process.
    chunk_queue: queue.Queue = queue.Queue(maxsize=acfg.learner.queue_size)
    traj_server = serve_trajectory_receiver(chunk_queue, port=traj_port, max_message_mb=max_mb)

    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    weight_sink = [GRPCWeightSink(store, agent_id)]

    # Checkpoint persistence (drained off-thread so training never blocks). Always on:
    # the learner's final snapshot is saved even without periodic checkpoints.
    from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager
    checkpoint_queue: queue.Queue = queue.Queue(maxsize=16)
    coordinator_ckpt = CheckpointManager(base_dir=run_dir.checkpoints, pool_size=config.checkpoint.pool_size)

    algo_cls = import_class(acfg.algorithm.algorithm_class)
    action_spec = ActionSpec.from_space(role_spec.action_space)
    teacher_path = acfg.training.kickstart_teacher

    def algorithm_factory():
        model = build_model(acfg, role_spec)
        kickstart = None
        if teacher_path:
            from colosseum.sp2.bc.kickstart import KickstartLoss
            teacher = build_model(acfg, role_spec)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True, map_location=device))
            teacher.to(device)
            kickstart = KickstartLoss(
                teacher,
                initial_lambda=acfg.training.kickstart_lambda,
                decay_steps=acfg.training.kickstart_decay_steps,
                direction=acfg.training.kickstart_kl,
            )
        kwargs = {"device": device, "pin_memory": acfg.learner.pin_memory}
        if kickstart is not None:
            kwargs["kickstart"] = kickstart
        return algo_cls(model, acfg.algorithm, action_spec, **kwargs)

    stop_event = threading.Event()
    supervisor = ProcessSupervisor(stop_event)

    # Seed right before learner_process builds the model (validation above also draws from the
    # RNGs). Same per-agent stream as a local-mode learner.
    agent_index = config.get_trainable_agent_ids().index(agent_id)
    apply_global_seed(learner_seed(config.training.seed, agent_index))

    cfg_hash = config_hash(config)
    roles_meta = {"roles": setup.agent_roles[agent_id], "role_signature": role_signature(role_spec)}

    def _save(data: dict) -> None:
        """Persist one payload; a failure is logged and never kills the caller."""
        trainer_state = data.get("trainer_state_bytes") if config.checkpoint.save_optimizer else None
        try:
            coordinator_ckpt.save(
                agent_id=agent_id,
                policy_version=int(data["policy_version"]),
                model_state=data["model_state"],
                trainer_state=trainer_state,
                # env_steps is null: a distributed learner has no global env-step count
                # (consumed_samples counts ACT slots of its own seats, a different quantity),
                # so a resume from it does not seed the env-step budget.
                meta_extra={"final": bool(data.get("final", False)),
                            "networks": acfg.networks.model_dump(mode="json", by_alias=True),
                            "config_hash": cfg_hash, "env_steps": None, **roles_meta},
            )
        except Exception:  # noqa: BLE001 - one failed save must not stop checkpointing
            logger.exception(f"Distributed learner [{agent_id}]: failed to save checkpoint "
                             f"v{data.get('policy_version')}")

    def _drain_checkpoints():
        while not stop_event.is_set():
            try:
                data = checkpoint_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            _save(data)

    drainer = threading.Thread(target=_drain_checkpoints, daemon=True)
    drainer.start()

    logger.info(f"Distributed learner [{agent_id}] serving trajectories on :{traj_port}, "
                f"weights -> {weight_store_address}")
    try:
        # SIGINT / SIGTERM set stop_event (and are remembered for the exit code); installed
        # inside the try so the finally always restores them.
        supervisor.install_signal_handlers()
        learner_process(
            agent_id=agent_id,
            algorithm_factory=algorithm_factory,
            trajectory_queue=chunk_queue,
            weight_queues=weight_sink,
            config=acfg.learner,
            stop_event=stop_event,
            metrics_queue=None,
            progress_counter=None,
            total_timesteps=config.training.total_timesteps,
            checkpoint_queue=checkpoint_queue,
            checkpoint_interval=config.checkpoint.interval,
            weight_sync_interval=acfg.rollout.weight_sync_interval_sec,
        )
    finally:
        stop_event.set()
        drainer.join()  # it finishes the save in progress, then sees stop_event
        while True:  # the final snapshot arrives after stop_event is set
            try:
                _save(checkpoint_queue.get_nowait())
            except queue.Empty:
                break
        traj_server.stop(0)
        store.close()
        supervisor.restore_signal_handlers()
        logger.info(f"Distributed learner [{agent_id}] stopped.")
    if supervisor.received_signal is not None:
        signum = int(supervisor.received_signal)
        logger.warning(f"Distributed learner [{agent_id}] stopped by {signal.Signals(signum).name}")
        return 128 + signum
    return 0
''')

dist.replace("_dist_worker_main", '''
def _dist_worker_main(
    *,
    worker_id: int,
    config: ColosseumConfig,
    agent_ids: list[str],
    agent_roles: dict[str, list[str]],
    agent_configs: dict[str, ColosseumConfig],
    role_specs: dict[str, RoleSpec],
    weight_store_address: str,
    learner_addresses: dict[str, str],
    stop_event,
    total_timesteps: int,
    lineups: list[Lineup],
) -> None:
    """Worker process body: gRPC clients in, rollout_worker_process unchanged."""
    from colosseum.sp2.core.registry import build_model
    from colosseum.sp2.launcher import _create_env
    from colosseum.sp2.transport.grpc_transport import GRPCTransport
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore
    from colosseum.sp2.worker.rollout_worker import rollout_worker_process

    max_mb = config.transport.grpc_max_message_mb
    store = GRPCWeightStore(weight_store_address, max_message_mb=max_mb)
    transports = {aid: GRPCTransport(learner_addresses[aid], max_message_mb=max_mb) for aid in agent_ids}

    worker_seed = None
    if config.training.seed is not None:
        worker_seed = config.training.seed + worker_id * 1000

    rollout_worker_process(
        worker_id=worker_id,
        env_fn=partial(_create_env, config.env.env_class, config.env.kwargs),
        num_envs=config.rollout.envs_per_worker,
        chunk_length=config.rollout.chunk_length,
        agent_ids=agent_ids,
        agent_roles=agent_roles,
        model_factories={aid: partial(build_model, agent_configs[aid], role_specs[aid]) for aid in agent_ids},
        trajectory_queues={aid: GRPCTrajectorySink(transports[aid], aid) for aid in agent_ids},
        weight_queues={aid: GRPCWeightSource(store, aid) for aid in agent_ids},
        stop_event=stop_event,
        weight_sync_interval=config.rollout.weight_sync_interval_sec,
        torch_threads=config.rollout.torch_threads,
        max_env_steps=total_timesteps,
        lineups=lineups,
        seed=worker_seed,
        vec_env_kind=config.rollout.vec_env,
        subproc_workers=config.rollout.subproc_workers,
        max_idle_steps=config.env.max_idle_steps,
    )
''')

dist.replace("run_distributed_workers", '''
def run_distributed_workers(
    config_path: str,
    weight_store_address: str,
    learner_addresses: dict[str, str],
    overrides: dict | None = None,
) -> int:
    """Launch rollout workers that feed remote learners over gRPC; returns the exit code.

    Workers stop by themselves after their share of the budget (exit 0). A worker exiting
    non-zero stops the others (exit code 1); SIGINT / SIGTERM stop all of them (128 + signum).

    Args:
        config_path: path to the YAML config.
        weight_store_address: ``host:port`` of the WeightStore service.
        learner_addresses: ``{agent_id: host:port}`` of each agent's TrajectoryService.
    """
    setup_process_logging(None, "workers-main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    agent_ids = list(learner_addresses.keys()) or config.get_trainable_agent_ids()
    setup = distributed_setup(config, agent_ids)  # before the run dir exists
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    # One dir per machine: worker hosts sharing a run.name on a shared filesystem never collide.
    run_dir = RunDir.create(config, config_path, role=workers_role())
    config = run_dir.with_run_name(config)
    setup_process_logging(run_dir.logs, "workers-main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    mp.set_start_method("spawn", force=True)

    stop_event = mp.Event()
    supervisor = ProcessSupervisor(stop_event, log_dir=run_dir.logs)
    supervisor.install_signal_handlers()
    worker_daemon = config.rollout.vec_env != "subprocess"
    per_worker_steps = config.training.total_timesteps // config.rollout.num_workers

    code = 0
    try:
        for worker_id in range(config.rollout.num_workers):
            seed = None if config.training.seed is None else config.training.seed + worker_id
            lineups = distributed_lineups(setup, agent_ids, config.rollout.envs_per_worker, random.Random(seed))
            proc = mp.Process(
                target=_dist_worker_target,
                name=f"worker-{worker_id}",
                kwargs=dict(
                    worker_id=worker_id,
                    log_dir=str(run_dir.logs),
                    config=config,
                    agent_ids=agent_ids,
                    agent_roles=setup.agent_roles,
                    agent_configs=agent_configs,
                    role_specs=setup.role_specs,
                    weight_store_address=weight_store_address,
                    learner_addresses=learner_addresses,
                    stop_event=stop_event,
                    total_timesteps=per_worker_steps,
                    lineups=lineups,
                ),
                daemon=worker_daemon,
            )
            start_process(proc)
            supervisor.add(f"worker-{worker_id}", proc)

        logger.info(f"Started {config.rollout.num_workers} distributed workers -> learners "
                    f"{learner_addresses}, weights <- {weight_store_address}")
        while not stop_event.is_set() and supervisor.alive():
            failure = supervisor.first_failure(nonzero_only=True)
            if failure is not None:
                logger.error(failure.message())
                code = 1
                break
            time.sleep(0.5)
    finally:
        stop_event.set()
        supervisor.wait_all(SHUTDOWN_GRACE_SEC)
        killed = supervisor.kill_remaining()
        supervisor.restore_signal_handlers()
        logger.info("Distributed workers stopped.")
    if supervisor.received_signal is not None:
        signum = int(supervisor.received_signal)
        logger.warning(f"Received {signal.Signals(signum).name}; distributed workers stopped")
        return 128 + signum
    if code == 0:
        # Exits after the loop ended (e.g. one worker finished, another crashed meanwhile).
        failures = supervisor.failures(exclude=set(killed))
        for failure in failures:
            logger.error(failure.message())
        code = 1 if failures else 0
    return code
''')
dist.write()
```

```bash
.venv/bin/python /tmp/make_sp2_distributed.py
.venv/bin/ruff check --fix src/colosseum/sp2/distributed.py src/colosseum/sp2/transport src/colosseum/sp2/weight_store
.venv/bin/ruff check src/colosseum/sp2/distributed.py src/colosseum/sp2/transport src/colosseum/sp2/weight_store
grep -n "slot_agent_map\|num_players\|self_play\.\|gamma=" src/colosseum/sp2/distributed.py || echo "no SP1 leftovers"
```

Expected: `no SP1 leftovers`.

- [ ] **Step 4: Add the CLI commands**

Insert into `src/colosseum/sp2/cli.py`, directly above the final `if __name__ == "__main__":` line:

```python
@main.command("run-learner")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--agent", "-a", default="agent_0", help="Trainable agent id this learner owns")
@click.option("--traj-port", default=50052, type=int, help="Port for this learner's TrajectoryService")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--set", "overrides", multiple=True,
              help="Override config values (e.g., --set rollout.num_workers=8)." + _SET_HELP_YAML)
def run_learner_cmd(config: str, agent: str, traj_port: int, weight_store: str, overrides: tuple[str, ...]) -> None:
    """Run one agent's learner as a gRPC service (distributed mode)."""
    with _config_errors():
        from colosseum.sp2.distributed import run_distributed_learner

        code = run_distributed_learner(config, agent, traj_port, weight_store,
                                       overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command("run-workers")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--weight-store", required=True, help="WeightStore address host:port")
@click.option("--learner", "-l", "learners", required=True, multiple=True,
              help="Learner address per agent: agent_id=host:port (repeatable)")
@click.option("--set", "overrides", multiple=True, help="Override config values." + _SET_HELP_YAML)
def run_workers_cmd(config: str, weight_store: str, learners: tuple[str, ...], overrides: tuple[str, ...]) -> None:
    """Run rollout workers feeding remote learners over gRPC (distributed mode)."""
    learner_addresses: dict[str, str] = {}
    for spec in learners:
        if "=" not in spec:
            raise click.BadParameter(f"--learner must be agent_id=host:port, got {spec!r}")
        aid, addr = spec.split("=", 1)
        learner_addresses[aid] = addr

    with _config_errors():
        from colosseum.sp2.distributed import run_distributed_workers

        code = run_distributed_workers(config, weight_store, learner_addresses,
                                       overrides=_parse_overrides(overrides) or None)
    sys.exit(code)


@main.command("serve-weight-store")
@click.option("--port", default=50051, type=int, help="gRPC port")
@click.option("--max-message-mb", default=64, type=int, help="Max gRPC message size in MiB")
def serve_weight_store_cmd(port: int, max_message_mb: int) -> None:
    """Start a gRPC weight store server."""
    import logging

    from colosseum.sp2.weight_store.grpc_store import serve_weight_store

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = serve_weight_store(port=port, max_message_mb=max_message_mb)
    click.echo(f"Weight store serving on port {port}. Press Ctrl+C to stop.")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_serialization.py tests/unit/test_sp2_distributed_roles.py tests/integration/test_sp2_grpc.py tests/integration/test_sp2_distributed_e2e.py -q`
Expected: all passed (the e2e test takes up to about a minute).

- [ ] **Step 6: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/sp2/distributed.py src/colosseum/sp2/transport src/colosseum/sp2/weight_store src/colosseum/sp2/cli.py \
        tests/unit/test_sp2_serialization.py tests/unit/test_sp2_distributed_roles.py \
        tests/integration/test_sp2_grpc.py tests/integration/test_sp2_distributed_e2e.py
git commit -m "feat: SP2 distributed self-play on chunk v2 with fixed lineups (SP2 T6.4)"
git push origin sp2-game-model
```

---

### Task T7.1: Port the SP1 integration guarantees to the `sp2` CLI

Every guarantee of SP1's `tests/integration/*` (and the launcher-level unit tests that back them) still relevant after SP2 gets an `sp2` copy with a new basename, so T7.3 can delete the SP1 files without losing coverage: lifecycle (exit codes 0/1/130/143, child death with a pointer to its log, SIGTERM and Ctrl-C with final checkpoints, every descendant gone within 10 s, Ctrl-C during startup, one-line config errors), the budget stop, run dir outputs, metrics outputs (all four kinds, `ratings.json`, console, WandB axes), league runs (two-agent self-play with periodic and final checkpoints, three-agent league where all pairs meet with balanced seats, resume continuing versions, env steps and the linear LR), eval after train, the full pipelines (spawned workers, routing, checkpoint pool, LSTM, subprocess vector env), worker threads, learner seeding and threads, entry-point validation, newest-wins mailboxes and numpy payloads.

Already ported elsewhere (not repeated here): BC CLI (T6.3), eval CLI (T6.1), gRPC and distributed e2e (T6.4), run dir unit tests (T5.4), launcher checkpoint/resume/shutdown tests (T5.4), checkpoint store (T5.3), metrics unit tests (T5.3); learner exit with large weight payloads, learner resume counters and the final checkpoint on stop (Part B T4.4: `tests/integration/test_learner_v2_exit.py`, `tests/unit/test_learner_v2.py`); subprocess vector env parity and clean close (Part A T1.6: `tests/integration/test_game_subproc_vector_env.py`); chunk and weight payload round trips (Part B T3.1: `tests/unit/test_chunk_v2_types.py`).

**Files:**
- Modify: `tests/game_helpers.py` (add `CrashingAPPO`)
- Test (all new): `tests/integration/test_sp2_lifecycle.py`, `test_sp2_budget_stop.py`, `test_sp2_run_dir_outputs.py`, `test_sp2_metrics_outputs.py`, `test_sp2_league_runs.py`, `test_sp2_pipelines.py` (copies by script), `test_sp2_eval_after_train.py`, `test_sp2_worker_threads.py`; `tests/unit/test_sp2_launcher_lifecycle.py`, `test_sp2_ipc_latest.py` (copies by script), `test_sp2_learner_entry.py`, `test_sp2_entry_points.py`, `test_sp2_payloads.py`

**Interfaces:**
- Consumes: everything the CLI exposes (T5.4, T6.1–T6.4); `Launcher`, `setup_run`, `_learner_main(..., role_spec, ...)`, `_worker_target(..., agent_roles, agent_configs, role_specs, lineups, ...)`, `warn_static_ownership_skew`, `run_training` (T5.4); `_push_weights` (T4.4: SP1 name kept by the copy); `APPO` (T4.2); `SubprocessVectorEnv` (T1.6); `rollout_worker_process` (T3.4); `TicTacToeGame`, the `configs/sp2/*.yaml` and `cli_runner.TTT_SP2_*` (T7.2); `GameTestModel`, `make_test_config`, `write_test_config`, `make_test_run_dir`, `make_coordinator`, `agent_role_of` (T5.3–T5.4); `learner_role`, `make_test_model` (Parts A/B kit); `tests/fake_wandb.py` (unchanged).
- Produces: `tests/game_helpers.CrashingAPPO` (an `APPO` whose third `train_step` raises).

- [ ] **Step 1: Add `CrashingAPPO`**

Add `from colosseum.sp2.algorithms.appo import APPO` to the import block at the top of `tests/game_helpers.py`, and append:

```python
class CrashingAPPO(APPO):
    """APPO whose third ``train_step`` raises: a learner crash in the middle of training."""

    crash_at_step = 3

    def train_step(self, chunks):
        self._steps_seen = getattr(self, "_steps_seen", 0) + 1
        if self._steps_seen >= self.crash_at_step:
            raise RuntimeError("injected train_step failure")
        return super().train_step(chunks)
```

- [ ] **Step 2: Copy SP1's integration tests**

Save as `/tmp/make_t71_integration_ports.py` and run it (each edit asserts its anchor, so a stale copy fails loudly):

```python
"""T7.1: SP2 copies of SP1's integration tests (run from the repo root, then ``ruff check --fix``).

Every copy gets a new unique basename; the SP1 files stay until T7.3 deletes them.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

IT = Path("tests/integration")


class Port:
    def __init__(self, src: str, dst: str) -> None:
        self.src, self.dst = Path(src), Path(dst)
        self.s = self.src.read_text()

    def sub(self, old: str, new: str, count: int = 1) -> None:
        found = self.s.count(old)
        assert found == count, f"{self.src}: expected {count} occurrence(s) of {old!r}, found {found}"
        self.s = self.s.replace(old, new)

    def regex(self, pattern: str, repl: str, expect: int) -> None:
        self.s, n = re.subn(pattern, repl, self.s)
        assert n == expect, f"{self.src}: {pattern!r} matched {n} times, expected {expect}"

    def replace_function(self, name: str, new_code: str) -> None:
        tree = ast.parse(self.s)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = self.s.splitlines(keepends=True)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        lines[start:node.end_lineno] = [textwrap.dedent(new_code).strip("\n") + "\n"]
        self.s = "".join(lines)

    def write(self) -> None:
        self.dst.write_text(self.s)
        print(f"wrote {self.dst}")


# --- lifecycle --------------------------------------------------------------------------------
p = Port(IT / "test_lifecycle.py", IT / "test_sp2_lifecycle.py")
p.sub('"""Exit codes, signals, child death, README quickstart (T6.5). Linux (/proc) only."""',
      '"""SP2 CLI: exit codes, signals, child death, quickstart (SP1 guarantees, T7.1). Linux (/proc) only."""')
p.sub("from cli_runner import TTT_CONFIG, run_in_session, run_train, training_process, wait_for\n",
      "from cli_runner import TTT_SP2_CONFIG, run_in_session, run_train, training_process, wait_for\n")
p.sub('FOREVER = {"training.total_timesteps": "100000000"}\n',
      'FOREVER = {"training.total_timesteps": "100000000"}\nSP2 = "colosseum.sp2"\n')
p.regex(r"(run_train|training_process)\(TTT_CONFIG, tmp_path, ", r"\1(TTT_SP2_CONFIG, tmp_path, module=SP2, ", 7)
p.sub('"algorithm.algorithm_class": "helpers.CrashingAPPO"}', '"algorithm.algorithm_class": "game_helpers.CrashingAPPO"}')
p.sub("    data = yaml.safe_load(TTT_CONFIG.read_text())\n", "    data = yaml.safe_load(TTT_SP2_CONFIG.read_text())\n")
p.sub('    proc = run_in_session([sys.executable, "-m", "colosseum", "train", "-c", str(bad)], timeout=120)\n',
      '    proc = run_in_session([sys.executable, "-m", SP2, "train", "-c", str(bad)], timeout=120)\n')
p.replace_function("test_readme_quickstart_from_repo_root", '''
def test_quickstart_from_repo_root(tmp_path):
    """The example config trains from the repo root with only a budget and a run dir set."""
    cmd = [sys.executable, "-m", SP2, "train", "-c", "configs/sp2/tic_tac_toe.yaml",
           "--set", "training.total_timesteps=2000",
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=quickstart"]
    proc = run_in_session(cmd, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "runs" / "quickstart" / "config.resolved.yaml").exists()
''')
p.write()

# --- budget stop (in-process Launcher) --------------------------------------------------------
p = Port(IT / "test_budget_stop.py", IT / "test_sp2_budget_stop.py")
p.sub('"""The launcher stops every process once the global env-step budget is reached (T2.5)."""',
      '"""The SP2 launcher stops every process once the global env-step budget is reached (T7.1)."""')
p.sub("from colosseum.core.config import ColosseumConfig, load_config\nfrom colosseum.launcher import Launcher\n"
      "from helpers import make_test_run_dir\n",
      "from colosseum.sp2.core.config import ColosseumConfig, load_config\nfrom colosseum.sp2.launcher import Launcher\n"
      "from game_helpers import make_test_run_dir\n")
p.sub('load_config(REPO / "configs/examples/tic_tac_toe.yaml")', 'load_config(REPO / "configs/sp2/tic_tac_toe.yaml")')
p.sub('load_config(REPO / "configs/examples/tic_tac_toe_multi.yaml")',
      'load_config(REPO / "configs/sp2/tic_tac_toe_multi.yaml")')
p.sub('algorithm_class="colosseum.algorithms.appo.NoSuchAlgorithm"',
      'algorithm_class="colosseum.sp2.algorithms.appo.NoSuchAlgorithm"')
p.sub('with caplog.at_level("ERROR", logger="colosseum.launcher"):',
      'with caplog.at_level("ERROR", logger="colosseum.sp2.launcher"):')
p.write()

# --- run dir outputs ----------------------------------------------------------------------------
p = Port(IT / "test_run_dir_outputs.py", IT / "test_sp2_run_dir_outputs.py")
p.sub('"""A short `colosseum train` writes the run dir, the resolved config and per-process logs (T6.2)."""',
      '"""A short SP2 `train` writes the run dir, the resolved config and per-process logs (T7.1)."""')
p.sub("from cli_runner import TTT_CONFIG, run_train\nfrom colosseum.coordinator.checkpoint_manager import resolve_resume\n"
      "from colosseum.core.config import load_config\n",
      "from cli_runner import TTT_SP2_CONFIG, run_train\n"
      "from colosseum.sp2.coordinator.checkpoint_manager import resolve_resume\n"
      "from colosseum.sp2.core.config import load_config\n")
p.sub('    run = run_train(TTT_CONFIG, tmp_path, name="logs-run")\n',
      '    run = run_train(TTT_SP2_CONFIG, tmp_path, name="logs-run", module="colosseum.sp2")\n')
p.write()

# --- metrics outputs ----------------------------------------------------------------------------
p = Port(IT / "test_metrics_outputs.py", IT / "test_sp2_metrics_outputs.py")
p.sub('"""A short run writes metrics.jsonl with all four kinds, ratings.json and console progress (T6.3)."""',
      '"""A short SP2 run writes metrics.jsonl with all four kinds, per-layout ratings.json, console progress (T7.1)."""')
p.sub("from cli_runner import TTT_CONFIG, run_train\nfrom colosseum.metrics.jsonl import REQUIRED_KEYS\n",
      "from cli_runner import TTT_SP2_CONFIG, run_train\nfrom colosseum.sp2.metrics.jsonl import REQUIRED_KEYS\n")
p.sub('    run = run_train(TTT_CONFIG, tmp_path, name="metrics-run",\n'
      '                    overrides={"training.total_timesteps": str(BUDGET)})\n',
      '    run = run_train(TTT_SP2_CONFIG, tmp_path, name="metrics-run", module="colosseum.sp2",\n'
      '                    overrides={"training.total_timesteps": str(BUDGET)})\n')
p.sub('    assert set(episodes[0]["wdl"]) == {"latest", "past", "arena"}\n',
      '    assert set(episodes[0]["wdl"]) == {"latest", "past", "arena"} and "2p" in episodes[0]["by_layout"]\n')
p.sub('    assert set(ratings) >= {"env_steps", "elo", "win_rates", "games", "wr_vs_past", "past_games"}\n',
      '    assert set(ratings) == {"env_steps", "layouts"}\n'
      '    assert set(ratings["layouts"]["2p"]) >= {"elo", "win_rates", "games", "wr_vs_past", "past_games"}\n')
p.sub("    from colosseum.core.config import ColosseumConfig, load_config\n    from colosseum.launcher import Launcher\n"
      "    from colosseum.metrics.jsonl import GLOBAL_KINDS\n    from helpers import example_config, make_test_run_dir\n",
      "    from cli_runner import TTT_SP2_CONFIG\n    from colosseum.sp2.core.config import ColosseumConfig, load_config\n"
      "    from colosseum.sp2.launcher import Launcher\n    from colosseum.sp2.metrics.jsonl import GLOBAL_KINDS\n"
      "    from game_helpers import make_test_run_dir\n")
p.sub('    data = load_config(example_config("tic_tac_toe.yaml")).model_dump()\n',
      '    data = load_config(TTT_SP2_CONFIG).model_dump()\n')
p.write()

# --- league runs ---------------------------------------------------------------------------------
p = Port(IT / "test_league_runs.py", IT / "test_sp2_league_runs.py")
p.sub('"""End-to-end league behaviour through `colosseum train` (T5.4)."""',
      '"""End-to-end league behaviour through the SP2 `train` (SP1 guarantees on lineups, T7.1)."""')
p.sub("from cli_runner import TINY, TTT_CONFIG, TTT_MULTI_CONFIG, TrainRun, run_train, training_process, wait_for\n"
      "from colosseum.core.config import load_config\n",
      "from cli_runner import TINY_SP2, TTT_SP2_CONFIG, TTT_SP2_MULTI_CONFIG, TrainRun, run_train, training_process, "
      "wait_for\nfrom colosseum.sp2.core.config import load_config\n\nSP2 = \"colosseum.sp2\"\n")
p.sub('    run = run_train(TTT_MULTI_CONFIG, tmp_path, name="sp2", overrides={\n        "training.phase": "self_play",\n',
      '    run = run_train(TTT_SP2_MULTI_CONFIG, tmp_path, name="sp2", module=SP2, overrides={\n'
      '        "matchmaking.mode": "self_play",\n')
p.sub('        assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder")\n',
      '        assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder") and meta["roles"] == ["player"]\n')
p.sub('    run = run_train(TTT_MULTI_CONFIG, tmp_path, name="league3", overrides={\n        "training.phase": "league",\n'
      '        "self_play.self_play_ratio": "0.0",\n',
      '    run = run_train(TTT_SP2_MULTI_CONFIG, tmp_path, name="league3", module=SP2, overrides={\n'
      '        "matchmaking.mode": "league",\n        "matchmaking.self_play_ratio": "0.0",\n')
p.sub('    games = run.ratings()["games"]\n', '    games = run.ratings()["layouts"]["2p"]["games"]\n')
p.sub('    common = {"algorithm.lr_schedule": "linear", "self_play.checkpoint_interval": "10",\n',
      '    common = {"algorithm.lr_schedule": "linear", "checkpoint.interval": "10",\n')
p.sub('    with training_process(TTT_CONFIG, tmp_path, name="first", overrides=common) as (proc, root):\n',
      '    with training_process(TTT_SP2_CONFIG, tmp_path, name="first", overrides=common, module=SP2) as (proc, root):\n')
p.sub('    second = run_train(TTT_CONFIG, tmp_path, name="second", overrides={\n',
      '    second = run_train(TTT_SP2_CONFIG, tmp_path, name="second", module=SP2, overrides={\n')
p.sub('    interval = float(TINY["metrics.console_interval_sec"])\n',
      '    interval = float(TINY_SP2["metrics.console_interval_sec"])\n')
p.write()

# --- pipelines (spawned workers and full Launcher runs) -------------------------------------------
p = Port(IT / "test_pipelines.py", IT / "test_sp2_pipelines.py")
p.sub('"""End-to-end runs through real worker / learner processes (spawn start method)."""',
      '"""SP2 end-to-end runs through real worker / learner processes (spawn start method, T7.1)."""')
p.sub("from colosseum.coordinator.checkpoint_manager import CheckpointManager\n"
      "from colosseum.core.config import ColosseumConfig, load_config\nfrom colosseum.core.types import TrajectoryChunk\n"
      "from helpers import example_config, make_test_run_dir\n",
      "from colosseum.sp2.coordinator.checkpoint_manager import CheckpointManager\n"
      "from colosseum.sp2.core.config import ColosseumConfig, load_config\n"
      "from colosseum.sp2.core.types import Lineup, SeatAssignment, TrajectoryChunk\n"
      "from colosseum.sp2.launcher import setup_run\nfrom game_helpers import make_test_run_dir\n"
      "from cli_runner import REPO_ROOT\n\n\ndef example_config(name: str):\n"
      "    return REPO_ROOT / \"configs\" / \"sp2\" / name\n")
p.regex(r"    from colosseum\.launcher import (Launcher|_worker_target)\n", r"    from colosseum.sp2.launcher import \1\n", 8)
p.sub('''    proc = mp.Process(
        target=_worker_target,
        kwargs=dict(
            worker_id=0, config=config, agent_ids=[agent_id], agent_configs={agent_id: config},
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event,
        ),
        daemon=True,
    )''', '''    setup = setup_run(config, validate=False)
    lineups = [Lineup("2p", [SeatAssignment(agent_id), SeatAssignment(agent_id)]) for _ in range(2)]
    proc = mp.Process(
        target=_worker_target,
        kwargs=dict(
            worker_id=0, config=config, agent_ids=[agent_id], agent_roles=setup.agent_roles,
            agent_configs=setup.agent_configs, role_specs=setup.role_specs,
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event, lineups=lineups,
        ),
        daemon=True,
    )''')
p.sub("        assert chunk.observations.shape[0] == 8\n", "        assert chunk.num_slots == 8\n")
p.sub('''    agent_ids = config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    trajectory_queues = {aid: mp.Queue(maxsize=16) for aid in agent_ids}
    weight_queues = {aid: mp.Queue(maxsize=2) for aid in agent_ids}
    stop_event = mp.Event()
    slot_agent_map = [[agent_ids[0], agent_ids[1]], [agent_ids[0], agent_ids[1]]]
    slot_network_map = [["latest", "latest"], ["latest", "latest"]]
    collect_mask = [[True, True], [True, True]]
    proc = mp.Process(
        target=_worker_target,
        kwargs=dict(
            worker_id=0, config=config, agent_ids=agent_ids, agent_configs=agent_configs,
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event,
            slot_network_map=slot_network_map, collect_mask=collect_mask, slot_agent_map=slot_agent_map,
        ),
        daemon=True,
    )''', '''    agent_ids = config.get_trainable_agent_ids()
    setup = setup_run(config, validate=False)
    trajectory_queues = {aid: mp.Queue(maxsize=16) for aid in agent_ids}
    weight_queues = {aid: mp.Queue(maxsize=2) for aid in agent_ids}
    stop_event = mp.Event()
    lineups = [Lineup("2p", [SeatAssignment(agent_ids[0]), SeatAssignment(agent_ids[1])]) for _ in range(2)]
    proc = mp.Process(
        target=_worker_target,
        kwargs=dict(
            worker_id=0, config=config, agent_ids=agent_ids, agent_roles=setup.agent_roles,
            agent_configs=setup.agent_configs, role_specs=setup.role_specs,
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event, lineups=lineups,
        ),
        daemon=True,
    )''')
p.sub('''        self_play={"checkpoint_interval": 40, "pool_size": 5},''',
      '''        checkpoint={"interval": 40, "pool_size": 5},''')
p.sub('''        self_play={"checkpoint_interval": 50},''', '''        checkpoint={"interval": 50},''')
p.write()
```

```bash
.venv/bin/python /tmp/make_t71_integration_ports.py
.venv/bin/ruff check --fix tests/integration/test_sp2_*.py && .venv/bin/ruff check tests/integration/test_sp2_*.py
grep -n "from helpers\|from dataflow_helpers\|\bTTT_CONFIG\b\|\bTTT_MULTI_CONFIG\b\|self_play\.\|training\.phase" \
  tests/integration/test_sp2_*.py || echo "no SP1 test support left"
```

Expected: `no SP1 test support left`.

- [ ] **Step 3: Copy SP1's launcher lifecycle unit tests**

Save as `/tmp/make_t71_unit_ports.py` and run it (all module paths of replaced modules are rewritten, also in strings such as logger names and monkeypatch targets; the queue-reader tests no longer pin the logger name because T0.1 moved the reader to `core/ipc.py`; two metrics-teardown tests from SP1's `test_metrics_jsonl.py` and `test_wandb_logger.py` are appended):

```python
"""T7.1: SP2 copy of SP1's tests/unit/test_launcher_lifecycle.py (run from the repo root, then ruff --fix)."""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

src = Path("tests/unit/test_launcher_lifecycle.py")
dst = Path("tests/unit/test_sp2_launcher_lifecycle.py")
s = src.read_text()


def sub(old: str, new: str, count: int = 1) -> None:
    global s
    found = s.count(old)
    assert found == count, f"expected {count} occurrence(s) of {old!r}, found {found}"
    s = s.replace(old, new)


def replace_function(name: str, new_code: str) -> None:
    global s
    tree = ast.parse(s)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = s.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    lines[start:node.end_lineno] = [textwrap.dedent(new_code).strip("\n") + "\n"]
    s = "".join(lines)


sub('"""Launcher lifecycle in-process (T6.5): validation once, partial-start teardown, exit codes\n'
    'from child exits, drains that never block on a dead producer, CLI config errors."""',
    '"""SP2 launcher lifecycle in-process (SP1 guarantees, T7.1): validation once, partial-start teardown,\n'
    'exit codes from child exits, drains that never block on a dead producer, CLI config errors."""')
# Module paths of everything SP2 replaced (also inside strings: logger names, monkeypatch targets).
for old, new in [
    ("colosseum.launcher", "colosseum.sp2.launcher"),
    ("colosseum.cli", "colosseum.sp2.cli"),
    ("colosseum.coordinator.coordinator", "colosseum.sp2.coordinator.coordinator"),
    ("colosseum.core.config", "colosseum.sp2.core.config"),
    ("colosseum.core.registry", "colosseum.sp2.core.registry"),
    ("colosseum.distributed", "colosseum.sp2.distributed"),
    ("colosseum.transport.grpc_transport", "colosseum.sp2.transport.grpc_transport"),
    ("colosseum.weight_store.grpc_store", "colosseum.sp2.weight_store.grpc_store"),
    ("colosseum.learner.learner", "colosseum.sp2.learner.learner"),
]:
    s = re.sub(rf"\b{re.escape(old)}\b", new, s)
sub("from helpers import example_config, make_test_run_dir\n",
    "from game_helpers import make_coordinator, make_test_config, make_test_run_dir\n")
replace_function("ttt_data", '''
def ttt_data(**training) -> dict:
    """A small two-seat toy-game config as raw data (one worker, two envs, CPU learner)."""
    data = make_test_config("turns").model_dump(mode="json", by_alias=True)
    data["rollout"].update(num_workers=1, envs_per_worker=2)
    data["training"] = {**data["training"], "seed": None, **training}
    data["metrics"] = {**data["metrics"], "console_interval_sec": 10.0}
    return data
''')
sub("    launcher._coordinator = Coordinator(cfg, checkpoint_dir=tmp_path / \"ckpt\")\n",
    "    launcher._coordinator = make_coordinator(cfg, tmp_path / \"ckpt\")\n")
sub('    cfg = load_config(example_config("tic_tac_toe.yaml"))\n',
    "    cfg = make_test_config(\"turns\")\n", count=3)
sub('    data = yaml.safe_load(example_config("tic_tac_toe.yaml").read_text())\n',
    "    data = ttt_data()\n")
# T0.1 moved the queue reader out of the launcher: its log lines may come from another module.
sub('    with caplog.at_level(logging.ERROR, logger="colosseum.sp2.launcher"):\n'
    '        threading.Thread(target=drain_and_shutdown, daemon=True).start()\n',
    '    with caplog.at_level(logging.ERROR):\n'
    '        threading.Thread(target=drain_and_shutdown, daemon=True).start()\n')
sub('    with caplog.at_level(logging.DEBUG, logger="colosseum.sp2.launcher"):\n        launcher._release_queues()\n',
    '    with caplog.at_level(logging.DEBUG):\n        launcher._release_queues()\n')
sub('    with caplog.at_level(logging.ERROR, logger="colosseum.sp2.launcher"):\n'
    '        assert launcher._drain(cq, "checkpoint-agent_0", lambda: ["learner-agent_0"]) == []',
    '    with caplog.at_level(logging.ERROR):\n'
    '        assert launcher._drain(cq, "checkpoint-agent_0", lambda: ["learner-agent_0"]) == []')
sub('    assert ("[ERROR] workers-main colosseum.sp2.distributed: worker-0 died (exit 3)" in capsys.readouterr().err) == (\n'
    '        worker_exit == 3)\n',
    '    assert ("worker-0 died (exit 3)" in capsys.readouterr().err) == (worker_exit == 3)\n')
s += '''

# ---------------------------------------------------------------------------
# Metrics teardown (from SP1's test_metrics_jsonl.py and test_wandb_logger.py)
# ---------------------------------------------------------------------------


def test_launcher_closes_metrics_file_when_final_drain_fails(tmp_path):
    import queue

    from colosseum.sp2.metrics.hub import MetricsHub
    from colosseum.sp2.metrics.jsonl import MetricsWriter

    config = make_test_config("turns")
    run = make_test_run_dir(config, tmp_path)
    launcher = Launcher(config, run)
    writer = MetricsWriter(run.metrics_path)
    launcher._metrics_writer = writer
    launcher._hub = MetricsHub(writer=writer, ratings_path=run.ratings_path, agent_ids=["agent_0"],
                               total_timesteps=10, log_interval=1, console_interval_sec=10.0)

    class FailingCoordinator:
        def report_match_result(self, result):
            raise RuntimeError("boom")

        def ratings_snapshot(self):
            return {}

    results = queue.Queue()
    results.put("a result")
    launcher._coordinator = FailingCoordinator()
    launcher._results_queue, launcher._metrics_queue = results, queue.Queue()
    with pytest.raises(RuntimeError, match="boom"):
        launcher._finish_metrics()
    assert writer.closed


def test_launcher_finishes_wandb_when_final_metrics_fail(monkeypatch, tmp_path):
    import queue

    from colosseum.metrics.wandb_logger import WandBLogger
    from colosseum.sp2.core.config import MetricsConfig
    from colosseum.sp2.metrics.hub import MetricsHub
    from colosseum.sp2.metrics.jsonl import MetricsWriter

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = make_test_config("turns")
    run = make_test_run_dir(config, tmp_path)
    launcher = Launcher(config, run)
    launcher._wandb = WandBLogger(MetricsConfig(use_wandb=True))
    launcher._metrics_writer = MetricsWriter(run.metrics_path)
    launcher._hub = MetricsHub(writer=launcher._metrics_writer, ratings_path=run.ratings_path,
                               agent_ids=["agent_0"], total_timesteps=10, log_interval=1,
                               console_interval_sec=10.0, wandb_logger=launcher._wandb)

    class FailingCoordinator:
        def ratings_snapshot(self):
            raise RuntimeError("boom")

    launcher._coordinator = FailingCoordinator()
    launcher._results_queue, launcher._metrics_queue = queue.Queue(), queue.Queue()
    with pytest.raises(RuntimeError, match="boom"):
        launcher._finish_metrics()
    assert launcher._metrics_writer.closed and fake.finished
'''
dst.write_text(s)
print(f"wrote {dst}")
```

```bash
.venv/bin/python /tmp/make_t71_unit_ports.py
.venv/bin/ruff check --fix tests/unit/test_sp2_launcher_lifecycle.py && .venv/bin/ruff check tests/unit/test_sp2_launcher_lifecycle.py
```

- [ ] **Step 4: Write the remaining ports**

Create `tests/integration/test_sp2_eval_after_train.py`:

```python
"""Smoke: `python -m colosseum.sp2 eval` loads checkpoints written by a real SP2 `train` run (T7.1)."""
from __future__ import annotations

import json
import re
import sys

from cli_runner import TTT_SP2_CONFIG, run_in_session, run_train


def test_eval_loads_launcher_checkpoints(tmp_path):
    run = run_train(TTT_SP2_CONFIG, tmp_path, name="eval-smoke", module="colosseum.sp2")
    assert run.returncode == 0, run.stderr[-2000:]
    agent_dir = run.root / "checkpoints" / "agent_0"
    ckpts = sorted((d for d in agent_dir.iterdir() if re.fullmatch(r"ckpt_v\d+", d.name)),
                   key=lambda d: int(d.name[len("ckpt_v"):]))
    assert ckpts, f"no checkpoints in {agent_dir}"
    meta = json.loads((ckpts[-1] / "meta.json").read_text())
    assert "networks" in meta and meta["roles"] == ["player"]

    out = tmp_path / "out.json"
    proc = run_in_session([
        sys.executable, "-m", "colosseum.sp2", "eval", "-c", str(TTT_SP2_CONFIG),
        "-a", f"x={ckpts[0]}", "-a", f"y={ckpts[-1]}", "-n", "4", "--num-envs", "2", "--seed", "0",
        "-o", str(out),
    ], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    section = json.loads(out.read_text())["layouts"]["2p"]
    assert [(r["agent_a"], r["agent_b"]) for r in section["pairs"]] == [("x", "y"), ("y", "x")]
    assert all(r["n"] == 4 and r["wins"] + r["draws"] + r["losses"] == 4 for r in section["pairs"])
```

Create `tests/integration/test_sp2_worker_threads.py`:

```python
"""Spawned SP2 workers and SubprocessVectorEnv children run with the configured torch threads (T7.1).

``conftest.py`` exports ``OMP_NUM_THREADS=1`` to every child, so each test sets ``OMP_NUM_THREADS=3``
before spawning: only the explicit thread calls in the worker / env-child code can give the
asserted values.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.sp2.core.types import Lineup, SeatAssignment
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.sp2.envs.vector import SubprocessVectorEnv
from colosseum.sp2.worker.rollout_worker import rollout_worker_process
from game_helpers import GameTestModel

_PARENT_OMP = "3"
OBS = gymnasium.spaces.Box(0.0, 64.0, (3,), np.float32)
ACT = gymnasium.spaces.Discrete(2)


class ThreadProbeGame(MultiAgentEnv):
    """Solo; the observation reports [torch intra-op threads, OMP_NUM_THREADS, torch inter-op threads]."""

    spec = GameSpec.solo(OBS, ACT)

    def _obs(self) -> np.ndarray:
        omp = float(os.environ.get("OMP_NUM_THREADS", "0"))
        return np.array([torch.get_num_threads(), omp, torch.get_num_interop_threads()], np.float32)

    def reset(self, seed, layout) -> StepResult:
        self._t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions) -> StepResult:
        self._t += 1
        if self._t >= 3:
            return StepResult(acting=set(), obs={}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()})


def probe_model() -> GameTestModel:
    return GameTestModel(OBS, ACT)


@pytest.fixture(autouse=True)
def _omp_three(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", _PARENT_OMP)
    yield
    assert os.environ.get("OMP_NUM_THREADS") == _PARENT_OMP


def _run_probe_worker(torch_threads: int, vec_env_kind: str) -> np.ndarray:
    ctx = mp.get_context("spawn")
    tq, wq, stop = ctx.Queue(), ctx.Queue(maxsize=1), ctx.Event()
    proc = ctx.Process(
        target=rollout_worker_process,
        kwargs=dict(
            worker_id=0, env_fn=ThreadProbeGame, num_envs=1, chunk_length=4, agent_ids=["a"],
            agent_roles={"a": ["player"]}, model_factories={"a": probe_model},
            trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=stop,
            lineups=[Lineup("solo", [SeatAssignment("a")])], torch_threads=torch_threads,
            vec_env_kind=vec_env_kind, subproc_workers=1,
        ),
        daemon=False,  # a subprocess vector env spawns its own children
    )
    proc.start()
    try:
        item = tq.get(timeout=120)
    finally:
        stop.set()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    return np.asarray(item["obs"])


def test_spawned_worker_uses_rollout_torch_threads():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="sync")
    assert set(obs[:, 0].tolist()) == {2.0} and set(obs[:, 2].tolist()) == {1.0}


def test_subprocess_env_children_use_one_thread():
    obs = _run_probe_worker(torch_threads=2, vec_env_kind="subprocess")
    assert set(obs[:, 0].tolist()) == {1.0} and set(obs[:, 1].tolist()) == {1.0}


def test_subprocess_vector_env_children_threads_and_parent_env_restored():
    before = os.environ.get("OMP_NUM_THREADS")
    vec = SubprocessVectorEnv(ThreadProbeGame, num_envs=2, num_workers=2)
    try:
        results = vec.reset({0: (None, "solo"), 1: (None, "solo")})
        assert [results[e].obs[0][0] for e in (0, 1)] == [1.0, 1.0]
        assert [results[e].obs[0][1] for e in (0, 1)] == [1.0, 1.0]
    finally:
        vec.close()
    assert os.environ.get("OMP_NUM_THREADS") == before
```

Create `tests/unit/test_sp2_learner_entry.py` (SP1's `test_seeding.py` and the `_learner_main` tests of `test_threads.py`):

```python
"""SP2 learner entry points: per-agent seed streams and torch threads (SP1 guarantees, T7.1)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from colosseum.sp2.core.config import ColosseumConfig, load_config
from colosseum.utils.seeding import learner_seed
from game_helpers import agent_role_of, make_test_config, make_test_run_dir, write_test_config


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


@pytest.fixture
def capture_learner_model(monkeypatch):
    """Replace learner_process with one that builds the algorithm and records its initial weights."""
    import colosseum.core.threads as threads_module
    import colosseum.sp2.learner.learner as learner_module

    captured: list[np.ndarray] = []

    def fake_learner_process(*, algorithm_factory, **kwargs):
        captured.append(_flat(algorithm_factory().model.state_dict()))

    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    monkeypatch.setattr(threads_module, "configure_torch_threads", lambda *a, **k: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return captured


def _learner_main(config: ColosseumConfig, agent_id: str = "agent_0", **kwargs) -> None:
    from colosseum.sp2.launcher import _learner_main as main

    _roles, role = agent_role_of(config, agent_id)
    main(agent_id=agent_id, config=config.get_agent_config(agent_id), role_spec=role, trajectory_queue=None,
         weight_queues=[], stop_event=None, metrics_queue=None, **kwargs)


def test_local_learner_weights_follow_training_seed(capture_learner_model, restore_global_rng):
    cfg = make_test_config("turns", training={"seed": 3})

    def run(seed):
        torch.manual_seed(12345 + len(capture_learner_model))  # a different global state each time
        _learner_main(cfg, seed=seed)
        return capture_learner_model[-1]

    a, b, c = run(learner_seed(3, 0)), run(learner_seed(3, 0)), run(learner_seed(4, 0))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_launcher_gives_each_learner_its_own_seed_stream(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.sp2.launcher as launcher_module

    learner_kwargs: list[dict] = []

    class _Stop(Exception):
        pass

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, kwargs, name=None, daemon=None):
            self.target, self.kwargs = target, kwargs

        def start(self):
            if self.target is launcher_module._learner_target:
                learner_kwargs.append(self.kwargs)
            else:
                raise _Stop  # first worker: every learner has been started

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    cfg = make_test_config("asymmetric", training={"seed": 11})
    run = make_test_run_dir(cfg, tmp_path, name="seeded")
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, run).launch()
    assert [k["agent_id"] for k in learner_kwargs] == ["hunter", "prey"]
    assert [k["seed"] for k in learner_kwargs] == [learner_seed(11, 0), learner_seed(11, 1)]
    assert [k["role_spec"] for k in learner_kwargs] == [agent_role_of(cfg, a)[1] for a in ("hunter", "prey")]
    assert learner_kwargs[0]["checkpoint_interval"] == cfg.checkpoint.interval


def test_run_learner_seeds_before_building_the_model(tmp_path, monkeypatch, capture_learner_model,
                                                      restore_global_rng, restore_root_logging):
    import colosseum.sp2.distributed as distributed
    import colosseum.sp2.transport.grpc_transport as grpc_transport
    import colosseum.sp2.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(distributed.ProcessSupervisor, "install_signal_handlers", lambda self: None)
    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"alpha": {}, "beta": {}})

    def run(agent_id, seed):
        torch.manual_seed(999 + len(capture_learner_model))
        distributed.run_distributed_learner(str(path), agent_id, 0, "localhost:1",
                                            overrides={"training.seed": seed, "run.dir": str(tmp_path / "runs")})
        return capture_learner_model[-1]

    a, b = run("alpha", 5), run("alpha", 5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run("alpha", 6)) and not np.array_equal(a, run("beta", 5))
    for role_dir in sorted((tmp_path / "runs").iterdir()):
        resolved = load_config(role_dir / "config.resolved.yaml")
        assert role_dir.name in (f"{resolved.run.name}-learner-alpha", f"{resolved.run.name}-learner-beta")
    cfg = make_test_config("turns", agents={"alpha": {}, "beta": {}}, training={"seed": 5})
    _learner_main(cfg, agent_id="alpha", seed=learner_seed(5, 0))  # same stream as a local-mode learner
    assert np.array_equal(a, capture_learner_model[-1])


def _run_learner_target(monkeypatch, config, num_learners, seen=None):
    import colosseum.sp2.learner.learner as learner_mod

    seen = {} if seen is None else seen

    def fake_learner_process(**kwargs):
        seen["kwargs"] = kwargs
        seen["threads"] = torch.get_num_threads()
        seen["algorithm"] = kwargs["algorithm_factory"]()

    monkeypatch.setattr(learner_mod, "learner_process", fake_learner_process)
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = torch.get_num_threads()
    try:
        _learner_main(config, num_learners=num_learners)
    finally:
        torch.set_num_threads(before)
    return seen["threads"], seen["algorithm"]


def test_learner_target_sets_explicit_torch_threads(monkeypatch):
    config = make_test_config("turns", learner={"device": "cpu", "torch_threads": 3})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=1)
    assert threads == 3 and next(algorithm.model.parameters()).device.type == "cpu"


def test_learner_target_auto_threads_and_auto_device_resolve_to_cpu(monkeypatch):
    import colosseum.core.threads as threads_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(threads_mod.os, "cpu_count", lambda: 8)
    config = make_test_config("turns", learner={"device": "auto", "torch_threads": None},
                              rollout={"num_workers": 2, "torch_threads": 1})
    threads, algorithm = _run_learner_target(monkeypatch, config, num_learners=2)
    assert threads == 3  # (8 - 2 workers * 1 thread) // 2 learners
    assert next(algorithm.model.parameters()).device.type == "cpu"


def test_learner_target_passes_weight_sync_interval(monkeypatch):
    config = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1},
                              rollout={"weight_sync_interval_sec": 90.0})
    seen = {}
    _run_learner_target(monkeypatch, config, num_learners=1, seen=seen)
    assert seen["kwargs"]["weight_sync_interval"] == 90.0
```

Create `tests/unit/test_sp2_entry_points.py` (SP1's `test_entry_point_validation.py` and the CLI/launcher tests of `test_config_overrides.py`; the SP1 "misordered heads" config becomes a model whose `unroll` breaks the protocol):

```python
"""SP2 entry points validate before they build anything; CLI config errors are one line (T7.1)."""
from __future__ import annotations

import logging
import multiprocessing as mp

import pytest
from click.testing import CliRunner

from colosseum.core.errors import ConfigError
from colosseum.sp2.cli import main
from colosseum.sp2.networks.model import UnrollOutput
from game_helpers import GameTestModel, make_test_config, write_test_config


def _bad_policy_config(tmp_path):
    """A config whose model fails validation (values shaped [S, B])."""
    return write_test_config(tmp_path / "bad.yaml", "turns",
                             networks={"model_class": "test_sp2_entry_points.MatrixValueModel"})


def _fail(name):
    def _boom(*_args, **_kwargs):
        raise AssertionError(f"{name} ran before validate_config rejected the config")
    return _boom


class MatrixValueModel(GameTestModel):
    """``unroll`` returns values shaped [S, B] instead of [S*B] (validate_config must reject it)."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state, with_value)
        return out if out.value is None else UnrollOutput(out.dist, out.value.reshape(reset_after.shape))


def test_bc_eval_and_validate_reject_the_config_first(tmp_path):
    cfg = _bad_policy_config(tmp_path)
    data = tmp_path / "data.pt"
    data.write_bytes(b"")  # never read: validation fails first
    for args in (["bc", "-d", str(data), "-o", str(tmp_path / "out.pt")],
                 ["eval", "-a", f"x={tmp_path / 'never.pt'}"],
                 ["validate"]):
        result = CliRunner().invoke(main, [args[0], "-c", str(cfg), *args[1:]])
        assert result.exit_code == 1, (args, result.output)
        assert result.stderr.startswith("Config error:") and "time-major values" in result.stderr
        assert "Traceback" not in result.output
    assert not (tmp_path / "out.pt").exists()


def test_distributed_roles_reject_the_config_before_serving_or_spawning(tmp_path, monkeypatch,
                                                                        restore_root_logging):
    from colosseum.sp2 import distributed
    from colosseum.sp2.transport import grpc_transport

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", _fail("serve_trajectory_receiver"))
    with pytest.raises(ConfigError, match="time-major values"):
        distributed.run_distributed_learner(str(_bad_policy_config(tmp_path)), "agent_0", 0, "localhost:1")
    monkeypatch.setattr(mp, "set_start_method", _fail("mp.set_start_method"))
    monkeypatch.setattr(distributed.mp, "Process", _fail("mp.Process"))
    with pytest.raises(ConfigError, match="time-major values"):
        distributed.run_distributed_workers(str(_bad_policy_config(tmp_path)), "localhost:1",
                                            {"agent_0": "localhost:2"})
    assert not (tmp_path / "runs").exists()


def test_train_rejects_an_invalid_config_without_creating_a_run_dir(tmp_path, monkeypatch, restore_root_logging,
                                                                     restore_global_rng):
    import colosseum.sp2.launcher as launcher_module

    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)
    launched = []

    class FakeLauncher:
        def __init__(self, config, run_dir, validated=False):
            assert validated
            launched.append(run_dir)

        def launch(self):
            return 0

    monkeypatch.setattr(launcher_module, "Launcher", FakeLauncher)
    runs = tmp_path / "runs"
    overrides = {"run.dir": str(runs), "run.name": "retry"}
    with pytest.raises(ConfigError, match="time-major values"):
        launcher_module.run_training(str(_bad_policy_config(tmp_path)), overrides)
    assert not runs.exists() and not launched
    fixed = {**overrides, "networks.model_class": "game_helpers.GameTestModel"}
    assert launcher_module.run_training(str(_bad_policy_config(tmp_path)), fixed) == 0
    [run_dir] = launched
    assert run_dir.root == runs / "retry" and run_dir.resolved_config_path.is_file()


def test_train_rejects_a_missing_resume_source_without_creating_a_run_dir(tmp_path, monkeypatch,
                                                                         restore_root_logging):
    import colosseum.sp2.launcher as launcher_module

    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher_module, "Launcher", _fail("Launcher"))
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns")
    overrides = {"run.dir": str(tmp_path / "runs"), "run.name": "resumed",
                 "training.resume_from": str(tmp_path / "no_such_run")}
    with pytest.raises(ConfigError, match="resume_from"):
        launcher_module.run_training(str(cfg), overrides)
    assert not (tmp_path / "runs").exists()


def test_seed_is_applied_after_overrides(tmp_path, monkeypatch, restore_root_logging, restore_global_rng):
    import random

    import torch

    import colosseum.sp2.launcher as launcher_module

    # run_training forces the spawn start method; keep the test free of global side effects.
    monkeypatch.setattr(launcher_module.mp, "set_start_method", lambda *args, **kwargs: None)

    class FakeLauncher:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self):
            return 0

    monkeypatch.setattr(launcher_module, "Launcher", FakeLauncher)
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", training={"seed": 1})
    launcher_module.run_training(str(cfg), {"training.seed": 123, "run.dir": str(tmp_path / "runs")})
    assert torch.initial_seed() == 123
    assert random.random() == random.Random(123).random()


def test_cli_validate_malformed_set_value_exits_1(tmp_path):
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns")
    result = CliRunner().invoke(main, ["validate", "-c", str(cfg), "--set", "learner.batch_chunks=[1,"])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    assert "learner.batch_chunks" in result.stderr and "Traceback" not in result.output


@pytest.mark.parametrize(("workers", "envs", "agents", "refresh", "warns"), [
    (1, 3, 2, 0.0, True),    # 3 envs over 2 agents: agent 0 owns 2, agent 1 owns 1 forever
    (1, 1, 2, 0.0, True),    # fewer envs than agents: agent 1 never owns an env
    (2, 2, 2, 0.0, False),   # 4 envs over 2 agents: even split
    (1, 3, 2, 30.0, False),  # rotation advances, so ownership evens out over time
    (1, 3, 1, 0.0, False),   # single agent owns everything
])
def test_static_ownership_skew_warning(workers, envs, agents, refresh, warns, caplog):
    from colosseum.sp2.launcher import warn_static_ownership_skew

    cfg = make_test_config("turns", agents={f"a{i}": {} for i in range(agents)},
                           rollout={"num_workers": workers, "envs_per_worker": envs,
                                    "match_refresh_interval_sec": refresh})
    with caplog.at_level(logging.WARNING, logger="colosseum.sp2.launcher"):
        warn_static_ownership_skew(cfg)
    assert any("match_refresh_interval_sec" in r.getMessage() for r in caplog.records) == warns
```

Create `tests/unit/test_sp2_ipc_latest.py` from SP1's mailbox tests. Save as `/tmp/make_sp2_ipc_latest_test.py` and run it:

```python
"""T7.1: tests/unit/test_sp2_ipc_latest.py from SP1's tests/unit/test_ipc_latest.py."""
from pathlib import Path

s = Path("tests/unit/test_ipc_latest.py").read_text()
for old, new in [
    ('"""Newest-wins weight mailboxes (T2.3, regression for R2-05 / R3-10)."""',
     '"""Newest-wins weight mailboxes between the SP2 learner and workers (SP1 guarantees, T7.1)."""'),
    ("from colosseum.learner.learner import _push_weights\nfrom dataflow_helpers import TinyModel, publish_versions\n",
     "from colosseum.sp2.core.types import WeightPayload\nfrom colosseum.sp2.learner.learner import _push_weights\n"
     "from game_helpers import learner_role, make_test_model\n\n\n"
     "def publish_versions(q, n: int, done) -> None:\n"
     '    """Spawn target: publish WeightPayload v1..vn into a size-1 mailbox as fast as possible."""\n'
     "    for version in range(1, n + 1):\n"
     '        assert put_latest(q, WeightPayload("a", version, {"w": np.full(64, version, dtype=np.float32)}))\n'
     "    done.set()\n"),
    ("            self.model = TinyModel()\n", "            self.model = make_test_model(learner_role())\n"),
]:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
Path("tests/unit/test_sp2_ipc_latest.py").write_text(s)
print("wrote tests/unit/test_sp2_ipc_latest.py")
```

```bash
.venv/bin/python /tmp/make_sp2_ipc_latest_test.py
.venv/bin/ruff check --fix tests/unit/test_sp2_ipc_latest.py && .venv/bin/ruff check tests/unit/test_sp2_ipc_latest.py
```

Create `tests/unit/test_sp2_payloads.py`:

```python
"""Numpy payload forms of SP2 commands, optimizer states and the in-memory weight store (SP1 guarantees, T7.1;
chunk and weight payload round trips are Part B's tests/unit/test_chunk_v2_types.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors, from_numpy_tree, to_numpy_tree
from colosseum.sp2.core.types import (
    Lineup,
    SeatAssignment,
    WeightPayload,
    WorkerCommand,
    state_dict_to_numpy,
)
from colosseum.weight_store.shared_memory import InMemoryWeightStore


def tiny() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))






def test_worker_command_with_lineups_and_numpy_checkpoints_has_no_tensors():
    cmd = WorkerCommand(lineups=[Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v1", False)]), None],
                        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(tiny().state_dict())}})
    assert_no_tensors(cmd)


def test_optimizer_state_numpy_tree_roundtrip():
    model = tiny()
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    tree = to_numpy_tree(opt.state_dict())
    assert_no_tensors(tree)
    restored = torch.optim.Adam(tiny().parameters())
    restored.load_state_dict(from_numpy_tree(tree))
    key = next(iter(opt.state_dict()["state"]))
    assert torch.equal(restored.state_dict()["state"][key]["exp_avg"], opt.state_dict()["state"][key]["exp_avg"])


def test_assert_no_tensors_reports_the_path():
    with pytest.raises(TypeError, match=r"item\['x'\]\[1\]"):
        assert_no_tensors({"x": [1, torch.zeros(1)]})


def test_in_memory_weight_store_keeps_numpy():
    store = InMemoryWeightStore()
    store.put("a", WeightPayload.from_model("a", 2, tiny()))
    got = store.get("a")
    assert got.policy_version == 2 and store.get_version("a") == 2
    assert all(isinstance(v, np.ndarray) for v in got.state_dict.values())
```

- [ ] **Step 5: Run the ported tests**

Run: `.venv/bin/python -m pytest tests/integration/test_sp2_*.py tests/unit/test_sp2_launcher_lifecycle.py tests/unit/test_sp2_learner_entry.py tests/unit/test_sp2_entry_points.py tests/unit/test_sp2_ipc_latest.py tests/unit/test_sp2_payloads.py -q`
Expected: all passed. These are ports of passing SP1 tests: a failure is a regression of an SP1 guarantee in the SP2 code (fix the code, do not weaken the test; overview "Cross-part execution notes").

- [ ] **Step 6: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`. The suite now runs every lifecycle guarantee twice (SP1 and SP2 CLIs) until T7.3.

- [ ] **Step 7: Commit**

```bash
git add tests/game_helpers.py tests/integration/test_sp2_*.py tests/unit/test_sp2_launcher_lifecycle.py \
        tests/unit/test_sp2_learner_entry.py tests/unit/test_sp2_entry_points.py tests/unit/test_sp2_ipc_latest.py \
        tests/unit/test_sp2_payloads.py
git commit -m "test: port SP1 integration and lifecycle guarantees to the SP2 CLI (SP2 T7.1)"
git push origin sp2-game-model
```

---

### Task T7.2: Tic-tac-toe on the SP2 contract; SP1's fast learning tests on the new pipeline

The new tic-tac-toe is a turn-based two-seat `MultiAgentEnv` (layout `"2p"`, role `"player"`): only the player to move acts and gets the mask of empty cells; the last step gives both seats their reward and the team ranks. The configs keep SP1's workload and hyperparameters (LR 1e-3 constant, 600k env steps, `latest_prob` 0.8; SP1 ruling on the learning rate), so the slow ≥80% test that T8.3 ports stays reachable. The fast learning tests of SP1 (contextual bandit, combination lock, and the `gamma=0` negative control) run on `RolloutLoop` (on `MatchRunner`) → chunk v2 → APPO v2 with slot V-trace. The lock's 20-step limit is now a real truncation (`truncated` with `final_obs`), so the learner bootstraps it from its own value (spec block 4).

**Files:**
- Create: `examples/tic_tac_toe/game.py`, `examples/tic_tac_toe/models.py`
- Create: `configs/sp2/tic_tac_toe.yaml`, `configs/sp2/tic_tac_toe_multi.yaml`, `configs/sp2/tic_tac_toe_attention.yaml`
- Modify: `tests/learning/game_learning_envs.py` (append `ContextualBanditGame`, `ShortChainGame`)
- Modify: `tests/cli_runner.py` (add `TTT_SP2_CONFIG`, `TTT_SP2_MULTI_CONFIG`)
- Test: `tests/unit/test_sp2_ttt_game.py`, `tests/integration/test_sp2_ttt_example.py`, `tests/learning/test_sp2_fast_learning.py`

**Interfaces:**
- Consumes: `GameSpec.symmetric`, `MultiAgentEnv`, `StepResult`, `Outcome` (T1.4); `EpisodeTracker` (T1.5); `BaseEncoder`, `BasePolicy`, `BaseValue` (T2.3); `make_distribution`, `CategoricalDist`, `Distribution` (T2.1); `ActionSpec` (T1.3; `build_model` injects `action_spec` into a policy constructor that names it, T2.4); `RolloutLoop(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, agent_roles, model_factories, io, lineups, weight_sync_interval, seed)`, `LoopIO` (T3.4); `APPO(model, config, action_spec, device)` (T4.2); `act` (T2.3); `TrajectoryChunk`, `WeightPayload`, `Lineup`, `SeatAssignment` (T3.1); `validate_config` (T6.2); `make_mlp_model` (T6.3).
- Produces: `examples.tic_tac_toe.game.TicTacToeGame` (class attribute `spec`), `examples.tic_tac_toe.models.{TicTacToeEncoder, TicTacToePolicy, TicTacToeValue}`; configs under `configs/sp2/` (moved to `configs/examples/` by T7.3); `tests/learning/game_learning_envs.{ContextualBanditGame, ShortChainGame}`; `cli_runner.TTT_SP2_CONFIG`, `cli_runner.TTT_SP2_MULTI_CONFIG`.

- [ ] **Step 1: Test support**

In `tests/cli_runner.py`, add two constants after the line `TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"` so the block reads:

```python
TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"
# The SP2 tic-tac-toe configs (moved to configs/examples/ by T7.3).
TTT_SP2_CONFIG = REPO_ROOT / "configs" / "sp2" / "tic_tac_toe.yaml"
TTT_SP2_MULTI_CONFIG = REPO_ROOT / "configs" / "sp2" / "tic_tac_toe_multi.yaml"
```

Append to `tests/learning/game_learning_envs.py`:

```python
class ContextualBanditGame(MultiAgentEnv):
    """Solo, one decision per episode. Observation: one-hot context c in {0, 1}; reward 1 iff action == c."""

    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32), gymnasium.spaces.Discrete(2))

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0

    def reset(self, seed, layout) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._context = int(self._rng.integers(2))
        obs = np.zeros(2, np.float32)
        obs[self._context] = 1.0
        return StepResult(acting={0}, obs={0: obs})

    def step(self, actions) -> StepResult:
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)


class ShortChainGame(MultiAgentEnv):
    """Combination lock: states 0..4, start at 0, actions 0/1. The correct action at state i is
    ``KEY[i]`` and advances to i+1; a wrong action steps back to max(0, i-1). Reaching 4 gives +1
    and ends the episode by the rules; 20 decisions truncate it (``truncated`` with ``final_obs``).

    The key alternates, so no state-independent bias solves it, and only the last transition is
    rewarded: solving it needs credit assignment over several decisions (gamma > 0)."""

    KEY = (1, 0, 1, 0)
    LENGTH = len(KEY) + 1
    MAX_STEPS = 20
    spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (LENGTH,), np.float32), gymnasium.spaces.Discrete(2))

    def __init__(self) -> None:
        self._pos = 0
        self._t = 0

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.LENGTH, np.float32)
        obs[self._pos] = 1.0
        return obs

    def reset(self, seed, layout) -> StepResult:
        self._pos, self._t = 0, 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions) -> StepResult:
        self._t += 1
        if int(actions[0]) == self.KEY[self._pos]:
            self._pos += 1
        else:
            self._pos = max(0, self._pos - 1)
        if self._pos == self.LENGTH - 1:
            return StepResult(acting=set(), obs={}, rewards={0: 1.0}, episode_over=True)
        if self._t >= self.MAX_STEPS:
            return StepResult(acting=set(), obs={}, episode_over=True, truncated=True, final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()})
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp2_ttt_game.py`:

```python
"""Tic-tac-toe on the SP2 contract: turns, masks, outcomes, contract checks (T7.2)."""
from __future__ import annotations

import numpy as np
import pytest

from cli_runner import REPO_ROOT
from colosseum.sp2.core.config import load_config
from colosseum.sp2.core.registry import validate_config
from colosseum.sp2.envs.contract import EpisodeTracker
from examples.tic_tac_toe.game import TicTacToeGame

TTT_SP2 = REPO_ROOT / "configs" / "sp2"


def play(game: TicTacToeGame, cells: list[int]):
    """Play ``cells`` in turn under the contract tracker; returns the last StepResult."""
    tracker = EpisodeTracker(game.spec, context="test")
    result = game.reset(0, "2p")
    tracker.on_reset("2p", result)
    for cell in cells:
        (seat,) = result.acting
        actions = {seat: np.int64(cell)}
        result = game.step(actions)
        tracker.on_step(actions, result)
    return result


def test_spec_turns_and_masks():
    game = TicTacToeGame()
    assert list(game.spec.layouts) == ["2p"] and game.spec.outcome_kind("2p") == "wdl"
    result = game.reset(0, "2p")
    assert result.acting == {0} and result.action_masks[0].all() and result.obs[0][2].all()
    result = game.step({0: 4})
    assert result.acting == {1} and not result.action_masks[1][4]
    assert result.obs[1][1].reshape(-1)[4] == 1.0  # the opponent's mark from the mover's side
    with pytest.raises(ValueError, match="layout"):
        game.reset(0, "4p")


def test_win_gives_both_seats_their_reward_and_ranks():
    result = play(TicTacToeGame(), [0, 3, 1, 4, 2])
    assert result.episode_over and not result.acting
    assert result.rewards == {0: 1.0, 1: -1.0} and result.outcome.team_rank == {0: 1.0, 1: 2.0}


def test_draw():
    result = play(TicTacToeGame(), [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert result.episode_over and result.rewards == {0: 0.0, 1: 0.0}
    assert result.outcome.team_rank == {0: 1.0, 1: 1.0}


def test_illegal_move_loses():
    game = TicTacToeGame()
    game.reset(0, "2p")
    game.step({0: 4})
    result = game.step({1: 4})
    assert result.episode_over and result.rewards == {1: -1.0, 0: 1.0}


@pytest.mark.parametrize("name", ["tic_tac_toe.yaml", "tic_tac_toe_multi.yaml", "tic_tac_toe_attention.yaml"])
def test_example_configs_are_valid(name):
    validate_config(load_config(TTT_SP2 / name))
```

Create `tests/integration/test_sp2_ttt_example.py`:

```python
"""The tic-tac-toe example trains end to end on the SP2 CLI (smoke; learning is checked by T8.3)."""
from __future__ import annotations

from cli_runner import TTT_SP2_CONFIG, run_train


def test_tic_tac_toe_trains_to_the_budget(tmp_path):
    run = run_train(TTT_SP2_CONFIG, tmp_path, name="ttt", module="colosseum.sp2")
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert max(r["train_step"] for r in run.records("train") if r["agent"] == "agent_0") >= 1
    assert run.ratings()["layouts"]["2p"]["past_games"]["agent_0"] >= 0
```

Create `tests/learning/test_sp2_fast_learning.py`:

```python
"""Bandit and combination lock are solved in seconds by the SP2 pipeline: RolloutLoop on
MatchRunner -> chunk v2 -> APPO with slot V-trace (SP1's fast learning tests, T7.2)."""
from __future__ import annotations

import functools
import time

import pytest
import torch

from colosseum.sp2.algorithms.appo import APPO
from colosseum.sp2.core.config import AlgorithmConfig
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.types import Lineup, SeatAssignment, TrajectoryChunk, WeightPayload
from colosseum.sp2.networks.model import act
from colosseum.sp2.worker.rollout_loop import LoopIO, RolloutLoop
from game_learning_envs import ContextualBanditGame, ShortChainGame, make_mlp_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def train_until_solved(game_cls, obs_dim, solved, *, gamma=0.9, lr=3e-3, num_envs=8, chunk_length=16,
                       batch_chunks=4, max_updates=400, seed=0) -> int:
    """Collect and train in-process. Returns the number of updates used, or -1."""
    torch.manual_seed(seed)
    model_fn = functools.partial(make_mlp_model, obs_dim=obs_dim, num_actions=2, hidden=32)
    action_spec = ActionSpec.from_space(game_cls.spec.roles["player"].action_space)
    algo = APPO(model_fn(), AlgorithmConfig(learning_rate=lr, lr_schedule="constant", gamma=gamma,
                                            entropy_coeff=0.01), action_spec, device="cpu")
    chunks: list[TrajectoryChunk] = []
    latest = {"payload": WeightPayload.from_model("agent_0", algo.policy_version, algo.model)}
    io = LoopIO(
        send_chunk=lambda c: chunks.append(TrajectoryChunk.from_payload(c.to_payload())),
        poll_weights=lambda agent_id: latest["payload"],
    )
    lineups = [Lineup("solo", [SeatAssignment("agent_0")]) for _ in range(num_envs)]
    loop = RolloutLoop(worker_id=0, env_fn=game_cls, num_envs=num_envs, chunk_length=chunk_length,
                       agent_ids=["agent_0"], agent_roles={"agent_0": ["player"]},
                       model_factories={"agent_0": model_fn}, io=io, lineups=lineups,
                       weight_sync_interval=0.0, seed=seed)
    try:
        for update in range(1, max_updates + 1):
            while len(chunks) < batch_chunks:
                loop.step()
            batch = chunks[:batch_chunks]
            del chunks[:batch_chunks]
            algo.set_progress(update / max_updates)
            algo.train_step(batch)
            latest["payload"] = WeightPayload.from_model("agent_0", algo.policy_version, algo.model)
            loop.sync_weights()
            if update % 10 == 0 and solved(algo.model):
                return update
    finally:
        loop.close()
    return -1


@torch.no_grad()
def greedy_and_confident(model, obs: torch.Tensor, best: torch.Tensor, min_prob: float) -> bool:
    state = model.initial_state(obs.shape[0])
    out = act(model, obs, state, deterministic=True)
    if not torch.equal(out.actions.reshape(-1).long(), best):
        return False
    probs = model.step(obs, state).dist.log_prob(best).exp()
    return bool((probs >= min_prob).all())


def test_contextual_bandit_is_solved_in_seconds():
    start = time.monotonic()
    updates = train_until_solved(
        ContextualBanditGame, 2,
        lambda m: greedy_and_confident(m, torch.eye(2), torch.tensor([0, 1]), min_prob=0.9))
    assert updates != -1, "bandit not solved within 400 updates"
    assert time.monotonic() - start < 60


def _lock_solved(model) -> bool:
    """Greedy action == KEY[i] with p >= 0.8 in every non-terminal state of the lock."""
    states = torch.eye(ShortChainGame.LENGTH)[: ShortChainGame.LENGTH - 1]
    return greedy_and_confident(model, states, torch.tensor(ShortChainGame.KEY), 0.8)


def test_short_chain_is_solved_in_seconds():
    start = time.monotonic()
    updates = train_until_solved(ShortChainGame, ShortChainGame.LENGTH, _lock_solved)
    assert updates != -1, "chain not solved within 400 updates"
    assert time.monotonic() - start < 60


def test_short_chain_needs_discounting():
    """Negative control: with gamma=0 only the last move of the lock is ever credited, so the same
    setup must not solve it within the same 400-update cap."""
    assert train_until_solved(ShortChainGame, ShortChainGame.LENGTH, _lock_solved, gamma=0.0) == -1
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_ttt_game.py tests/integration/test_sp2_ttt_example.py tests/learning/test_sp2_fast_learning.py -q`
Expected: `test_sp2_ttt_game.py` and the example test fail with `No module named 'examples.tic_tac_toe.game'` (collection error / non-zero CLI exit); the learning tests pass already if Parts A/B are complete (they exercise the pipeline, not new code) — that is fine, they are ports.

- [ ] **Step 4: Write the game, the models and the configs**

Create `examples/tic_tac_toe/game.py`:

```python
"""Tic-tac-toe on the SP2 contract: a turn-based two-player ``MultiAgentEnv`` with action masks."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


class TicTacToeGame(MultiAgentEnv):
    """Two seats (layout ``"2p"``, role ``"player"``; team = seat), seat 0 moves first.

    - Only the player to move acts: ``acting`` is ``{current}``; it gets the observation and the
      mask of the empty cells.
    - Observation ``float32[3, 3, 3]`` from the mover's side: own marks, opponent marks, empty.
    - Action ``Discrete(9)``: the cell.
    - The last step gives +1 / -1 to the winner / loser (both seats, so the waiting loser's last
      move is credited too) or 0 / 0 for a draw, with ``Outcome.team_rank``. An illegal move
      (impossible with the mask) loses.
    """

    OBSERVATION_SPACE = gymnasium.spaces.Box(0.0, 1.0, (3, 3, 3), np.float32)
    ACTION_SPACE = gymnasium.spaces.Discrete(9)
    spec = GameSpec.symmetric(2, OBSERVATION_SPACE, ACTION_SPACE)

    def __init__(self) -> None:
        self._board = np.zeros(9, dtype=np.int8)  # 0 empty, 1 seat 0, 2 seat 1
        self._current = 0

    def reset(self, seed: int | None, layout: str) -> StepResult:
        if layout != "2p":
            raise ValueError(f"TicTacToeGame has the single layout '2p', got {layout!r}")
        self._board[:] = 0
        self._current = 0
        return self._turn()

    def step(self, actions: dict[int, Any]) -> StepResult:
        mover, other = self._current, 1 - self._current
        cell = int(actions[mover])
        if not (0 <= cell < 9) or self._board[cell] != 0:
            return self._end(winner=other)
        self._board[cell] = mover + 1
        if any(all(self._board[i] == mover + 1 for i in line) for line in WINNING_LINES):
            return self._end(winner=mover)
        if not (self._board == 0).any():
            return self._end(winner=None)
        self._current = other
        return self._turn()

    def _turn(self) -> StepResult:
        seat = self._current
        return StepResult(acting={seat}, obs={seat: self._obs(seat)}, action_masks={seat: self._board == 0})

    def _end(self, winner: int | None) -> StepResult:
        if winner is None:
            rewards, ranks = {0: 0.0, 1: 0.0}, {0: 1.0, 1: 1.0}
        else:
            rewards = {winner: 1.0, 1 - winner: -1.0}
            ranks = {winner: 1.0, 1 - winner: 2.0}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_rank=ranks))

    def _obs(self, seat: int) -> np.ndarray:
        board = self._board.reshape(3, 3)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[0] = board == seat + 1
        obs[1] = board == 2 - seat
        obs[2] = board == 0
        return obs
```

Create `examples/tic_tac_toe/models.py`:

```python
"""Networks for the tic-tac-toe example on the SP2 model protocol (parts of a ComposedModel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.sp2.networks.dist import CategoricalDist, Distribution, make_distribution


class TicTacToeEncoder(BaseEncoder):
    """MLP over the 3x3x3 board (own marks, opponent marks, empty)."""

    def __init__(self, hidden: int = 128, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(27, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


class TicTacToePolicy(BasePolicy):
    """Logits over the 9 cells. ``in_dim`` is the core's output size and ``action_spec`` the
    role's action spec (both injected by ``build_model``)."""

    def __init__(self, in_dim: int = 64, action_spec: ActionSpec | None = None, **kwargs) -> None:
        super().__init__()
        self._spec = action_spec
        self.net = nn.Linear(in_dim, 9)

    def forward(self, features: torch.Tensor, aux: dict[str, torch.Tensor]) -> Distribution:
        logits = self.net(features)
        return make_distribution(self._spec, logits) if self._spec is not None else CategoricalDist(logits)


class TicTacToeValue(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

Create `configs/sp2/tic_tac_toe.yaml`:

```yaml
# Tic-tac-toe self-play (SP2 contract: examples/tic_tac_toe/game.py). Same workload and
# hyperparameters as SP1's example (LR 1e-3, constant), which beat a random legal-move player in
# >= 80% of games after 600k env steps (tests/learning, slow test ported in T8.3).
run:
  name: null              # default: tic_tac_toe-<YYYYmmdd-HHMMSS>
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.game.TicTacToeGame"
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.models.TicTacToeEncoder"
  core: null              # stateless (MLP); see tic_tac_toe_attention.yaml for a core
  policy_class: "examples.tic_tac_toe.models.TicTacToePolicy"
  value_class: "examples.tic_tac_toe.models.TicTacToeValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 1.0
  eps_clip: 0.2
  value_loss_coeff: 0.5
  entropy_coeff: 0.01
  max_grad_norm: 0.5
  num_epochs: 1
  minibatch_chunks: 0
  vtrace_rho_bar: 1.0
  vtrace_c_bar: 1.0
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 2.0
  torch_threads: 1
  match_refresh_interval_sec: 30.0

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 600000   # global env steps across all workers
  seed: null

matchmaking:
  mode: self_play
  latest_prob: 0.8
  shuffle_seats: true

checkpoint:
  interval: 100            # train steps
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  wandb_project: "colosseum"
  log_interval: 10
  console_interval_sec: 10.0
```

Create `configs/sp2/tic_tac_toe_multi.yaml`:

```yaml
# Two trainable agents in a league: every match is an arena match (self_play_ratio 0.0).
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.game.TicTacToeGame"
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.models.TicTacToeEncoder"
  core: null
  policy_class: "examples.tic_tac_toe.models.TicTacToePolicy"
  value_class: "examples.tic_tac_toe.models.TicTacToeValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 1.0
  entropy_coeff: 0.01
  learning_rate: 3.0e-4
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 8
  weight_sync_interval_sec: 2.0
  torch_threads: 1
  match_refresh_interval_sec: 10.0

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 300000

matchmaking:
  mode: league
  self_play_ratio: 0.0
  latest_prob: 0.8
  pfsp_exponent: 1.0
  shuffle_seats: true

checkpoint:
  interval: 100
  pool_size: 5
  save_optimizer: false

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0

# Agent overrides are partial and deep-merged onto the sections above.
agents:
  agent_alpha: {}
  agent_beta:
    algorithm:
      learning_rate: 1.0e-4
```

Create `configs/sp2/tic_tac_toe_attention.yaml`:

```yaml
# Tic-tac-toe with a stateful core: causal attention over the last 8 latents of the episode.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.game.TicTacToeGame"
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.models.TicTacToeEncoder"
  core:
    class: "colosseum.networks.cores.WindowAttentionCore"
    kwargs: {d_model: 64, window: 8, num_heads: 4, num_layers: 1}
  policy_class: "examples.tic_tac_toe.models.TicTacToePolicy"   # gets in_dim = core.output_dim
  value_class: "examples.tic_tac_toe.models.TicTacToeValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 1.0
  entropy_coeff: 0.01
  learning_rate: 3.0e-4
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 2.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 600000

matchmaking:
  mode: self_play
  latest_prob: 0.8

checkpoint:
  interval: 100
  pool_size: 10

metrics:
  use_wandb: false
  log_interval: 10
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_sp2_ttt_game.py tests/integration/test_sp2_ttt_example.py tests/learning/test_sp2_fast_learning.py -q`
Expected: all passed; each learning test well under its 60 s bound, the negative control returns -1.

If a fast learning test does not reach its threshold within 400 updates, do not change the threshold or the cap: compare the V-trace targets of a lock episode with a hand-computed GAE(λ=1) on the ACT slots (T4.1's reference) and the boot slot at truncation; a failing control (solved with `gamma=0`) means the chain no longer needs credit assignment and the game is wrong.

- [ ] **Step 6: Learnability check of the example (manual, not a test)**

Run `.venv/bin/python -m colosseum.sp2 train -c configs/sp2/tic_tac_toe.yaml --set training.seed=0 --set run.dir=/tmp/ttt-check --set run.name=check` (about a minute on 8 cores), then evaluate the latest checkpoint against itself as a smoke: `.venv/bin/python -m colosseum.sp2 eval -c configs/sp2/tic_tac_toe.yaml -a a=/tmp/ttt-check/check/checkpoints/agent_0/ckpt_v<latest> -n 50`. Record the final console line (`return`, `wr_vs_past`) in the commit body; T8.3 adds the ≥80% test against a random legal player. Delete `/tmp/ttt-check` afterwards.

- [ ] **Step 7: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add examples/tic_tac_toe/game.py examples/tic_tac_toe/models.py configs/sp2 tests/cli_runner.py \
        tests/learning/game_learning_envs.py tests/unit/test_sp2_ttt_game.py tests/integration/test_sp2_ttt_example.py \
        tests/learning/test_sp2_fast_learning.py
git commit -m "feat: tic-tac-toe on the SP2 contract and SP1 fast learning tests on the new pipeline (SP2 T7.2)"
git push origin sp2-game-model
```

---

### Task T7.3: Overlay — delete the legacy, move `colosseum.sp2` onto `colosseum`, rename

The switch (overview, "Shadow package strategy", item 8). After this task `colosseum.sp2` no longer exists: the SP2 modules live at their final paths, the console script `colosseum` runs the SP2 CLI, the SP2 tic-tac-toe configs are the examples, every SP1-only module, test, example file and config is gone, and the suite is green. Part `04-…` is written against the final paths.

The procedure is ordered: deletions first (so moves never collide with legacy files), then the move, then one rename pass, then the edits that need final names, then verification. Do it in one session on a clean tree; commit once at the end.

**Files:**
- Delete: the legacy modules, tests, test support modules, example files and configs listed in Steps 2–4
- Move: every tracked file under `src/colosseum/sp2/` onto `src/colosseum/` (overwriting); `configs/sp2/*.yaml` → `configs/examples/`
- Modify (rename pass): every tracked file in `src/`, `tests/`, `configs/`, `examples/`, `scripts/`, `deployment/`, `docs/` (except `docs/superpowers/`), `README.md`, `CLAUDE.md`, `pyproject.toml` that mentions `colosseum.sp2`, `colosseum/sp2` or `configs/sp2`. Never `review/`, `research/`, `docs/superpowers/` (frozen: specs, plans, reports).
- Modify: `tests/cli_runner.py` (final form), `tests/unit/test_state.py`, `tests/unit/test_infrastructure.py`, `tests/unit/test_wandb_logger.py`, `tests/unit/test_threads.py`, `tests/integration/test_sp2_lifecycle.py`, `scripts/bench_throughput.py` (`_make_config` in config v2), `tests/unit/test_bench_throughput.py`, `pyproject.toml` (isort list)

**Interfaces:**
- Consumes: everything Parts A–C built under `colosseum.sp2`.
- Produces: the same names at `colosseum.*` (e.g. `colosseum.envs.game.GameSpec`, `colosseum.launcher.Launcher`, `colosseum.eval.play_lineups`); `colosseum` console script = `colosseum.cli:main` (SP2 CLI); `configs/examples/tic_tac_toe{,_multi,_attention}.yaml`; `tests/cli_runner.TTT_CONFIG/TTT_MULTI_CONFIG/TINY` (SP2 schema, `module` defaults to `"colosseum"`).

- [ ] **Step 1: Preconditions**

```bash
git status --porcelain            # must print nothing
.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .
```

Coverage gate. Every SP1 test file deleted below has successors; check each row before deleting (the successor files exist and pass). Rows marked "Part A"/"Part B" are covered by those parts' tests of the named task; if a row's successor is missing, stop and port it first (a new `test_sp2_*` file, same pattern as T7.1).

| SP1 file (deleted) | Successor tests |
|---|---|
| unit `test_action_masking`, `test_action_order`, `test_composite_actions`, `test_distributions` | Part A T1.3, T2.1, T2.2 |
| unit `test_model`, `test_composed`, `test_obs_normalization`, `test_registry`, `test_config`, `test_config_overrides`, `test_multi_agent` | Part A T1.7, T2.3, T2.4; T6.2 `test_sp2_validate`; T7.1 `test_sp2_entry_points` |
| unit `test_vec_env`, `test_vec_env_reset_info` | Part A T1.6 (`test_game_vector_env`, `test_game_subproc_vector_env`) |
| unit `test_appo`, `test_appo_metrics`, `test_appo_unroll`, `test_algorithm_state`, `test_kickstart_kl`, `test_vtrace`, `test_vtrace_lambda`, `test_lr_progress`, `test_performance` | Part B T4.1, T4.2, T4.3 |
| unit `test_budget`, `test_buffer_pool`, `test_collect_batch`, `test_drain_commands`, `test_policy_lag` | Part B T3.2, T3.4, T4.4 (`test_slot_buffers`, `test_rollout_worker_v2`, `test_learner_v2`) |
| unit `test_bc_trainer` | T6.3 `test_sp2_bc_trainer` |
| unit `test_checkpoint_store` | T5.3 `test_sp2_checkpoints`, T5.4 `test_sp2_launcher_checkpoints`; learner resume and final checkpoint: Part B T4.4 `test_learner_v2`; worker fallback: Part B T3.3 `test_match_runner` |
| unit `test_entry_point_validation`, `test_seeding` | T7.1 `test_sp2_entry_points`, `test_sp2_learner_entry` |
| unit `test_eval_engine`, `test_eval_report`; contract `test_eval_stateful` | T6.1 `test_sp2_eval_schedule`, `test_sp2_eval_report`, `test_sp2_eval_engine` |
| unit `test_examples` | T7.2 `test_sp2_ttt_game`; T8.5 (space_miners, chase) |
| unit `test_ipc_latest`, `test_payloads` | T7.1 `test_sp2_ipc_latest`, `test_sp2_payloads`; Part B T3.1 `test_chunk_v2_types` |
| unit `test_launcher_lifecycle`, `test_run_dir_logging` | T7.1 `test_sp2_launcher_lifecycle`; T5.4 `test_sp2_run_dir`; T6.4 `test_sp2_distributed_roles` |
| unit `test_league_matchmaking`, `test_league_ratings`, `test_review_fixes` | T5.1, T5.2, T5.3 (`test_sp2_coordinator`); outcomes: Part A T1.4 |
| unit `test_metrics_jsonl` | T5.3 `test_sp2_metrics`; T7.1 (launcher teardown) |
| unit `test_serialization` | T6.4 `test_sp2_serialization` |
| contract `harness`, `test_action_order_rollout`, `test_no_tensors_in_queues`, `test_open_transitions`, `test_parking`, `test_rollout_loop_characterization`, `test_rollout_state`, `test_seat_results`, `test_truncation`, `test_turn_based`, `test_worker_learner_consistency` | Part B T3.3, T3.4, T4.3 |
| integration `test_lifecycle`, `test_budget_stop`, `test_run_dir_outputs`, `test_metrics_outputs`, `test_league_runs`, `test_eval_after_train`, `test_pipelines`, `test_worker_threads` | T7.1 `test_sp2_*` of the same topic |
| integration `test_subproc_vec_env`, `test_learner_process_exit` | Part A T1.6 `test_game_subproc_vector_env`; Part B T4.4 `test_learner_v2_exit` |
| integration `test_bc_cli`, `test_eval_cli`, `test_grpc`, `test_distributed`, `test_distributed_e2e` | T6.3 `test_sp2_bc_cli`, T6.1 `test_sp2_eval_cli`, T6.4 `test_sp2_grpc`, `test_sp2_distributed_e2e` |
| learning `test_fast_learning`, `test_bc_learns` | T7.2 `test_sp2_fast_learning`, T6.3 `test_sp2_bc_learns` |
| learning `test_ttt_slow` (+ `ttt_eval`) | T8.3 (slow tic-tac-toe test on the new contract) |

- [ ] **Step 2: Delete the legacy modules**

```bash
git rm -q src/colosseum/envs/base_env.py src/colosseum/envs/vec_env.py src/colosseum/envs/subproc_vec_env.py \
          src/colosseum/core/action_spec.py src/colosseum/core/seat_info.py src/colosseum/networks/distributions.py \
          src/colosseum/worker/slots.py
```

Check that every remaining SP1 module is either replaced by an SP2 module of the same path or one of the unchanged modules the overview lists. Save as `/tmp/check_legacy.py` and run it:

```python
from pathlib import Path

KEEP = {
    "core/errors.py", "core/ipc.py", "core/threads.py", "coordinator/agent_pool.py",
    "networks/cores.py", "networks/state.py", "metrics/console.py", "metrics/wandb_logger.py",
    "transport/base.py", "transport/colosseum_pb2.py", "transport/colosseum_pb2_grpc.py", "transport/local.py",
    "weight_store/base.py", "weight_store/shared_memory.py",
    "utils/__init__.py", "utils/fs.py", "utils/logging.py", "utils/process.py", "utils/seeding.py",
}
src, sp2 = Path("src/colosseum"), Path("src/colosseum/sp2")
left = [f.relative_to(src).as_posix() for f in sorted(src.rglob("*.py"))
        if sp2 not in f.parents and f.relative_to(src).as_posix() not in KEEP
        and not (sp2 / f.relative_to(src)).exists()]
assert not left, f"SP1 modules without an SP2 replacement that are not kept: {left}"
print("every SP1 module is replaced or kept")
```

Expected: `every SP1 module is replaced or kept`. (An entry in `left` is a legacy module the list in this step missed: `git rm` it if nothing in `src/colosseum/sp2` imports it, otherwise it belongs in `KEEP`; record the decision in the commit body.)

- [ ] **Step 3: Delete the SP1 tests and test support modules**

```bash
cd tests
git rm -q unit/test_action_masking.py unit/test_action_order.py unit/test_algorithm_state.py unit/test_appo.py \
  unit/test_appo_metrics.py unit/test_appo_unroll.py unit/test_bc_trainer.py unit/test_budget.py unit/test_buffer_pool.py \
  unit/test_checkpoint_store.py unit/test_collect_batch.py unit/test_composed.py unit/test_composite_actions.py \
  unit/test_config.py unit/test_config_overrides.py unit/test_distributions.py unit/test_drain_commands.py \
  unit/test_entry_point_validation.py unit/test_eval_engine.py unit/test_eval_report.py unit/test_examples.py \
  unit/test_ipc_latest.py unit/test_kickstart_kl.py unit/test_launcher_lifecycle.py unit/test_league_matchmaking.py \
  unit/test_league_ratings.py unit/test_lr_progress.py unit/test_metrics_jsonl.py unit/test_model.py \
  unit/test_multi_agent.py unit/test_obs_normalization.py unit/test_payloads.py unit/test_performance.py \
  unit/test_policy_lag.py unit/test_registry.py unit/test_review_fixes.py unit/test_run_dir_logging.py \
  unit/test_seeding.py unit/test_serialization.py unit/test_vec_env.py unit/test_vec_env_reset_info.py \
  unit/test_vtrace.py unit/test_vtrace_lambda.py
git rm -q contract/harness.py contract/test_action_order_rollout.py contract/test_eval_stateful.py \
  contract/test_no_tensors_in_queues.py contract/test_open_transitions.py contract/test_parking.py \
  contract/test_rollout_loop_characterization.py contract/test_rollout_state.py contract/test_seat_results.py \
  contract/test_truncation.py contract/test_turn_based.py contract/test_worker_learner_consistency.py
git rm -q integration/test_bc_cli.py integration/test_budget_stop.py integration/test_distributed.py \
  integration/test_distributed_e2e.py integration/test_eval_after_train.py integration/test_eval_cli.py \
  integration/test_grpc.py integration/test_league_runs.py integration/test_learner_process_exit.py \
  integration/test_lifecycle.py integration/test_metrics_outputs.py integration/test_pipelines.py \
  integration/test_run_dir_outputs.py integration/test_subproc_vec_env.py integration/test_worker_threads.py
git rm -q learning/learning_envs.py learning/test_bc_learns.py learning/test_fast_learning.py \
  learning/test_ttt_slow.py learning/ttt_eval.py
git rm -q helpers.py dataflow_helpers.py
cd ..
```

The surviving SP1 tests are `tests/unit/test_bench_throughput.py`, `test_cores.py`, `test_infrastructure.py`, `test_process_lifecycle.py`, `test_state.py`, `test_threads.py`, `test_wandb_logger.py` (edited in Step 6), plus `tests/conftest.py`, `tests/cli_runner.py`, `tests/fake_wandb.py`; T0.1's `tests/unit/test_ipc_queue_helpers.py` stays too (after the move it checks the SP2 launcher, which imports the same `core.ipc` helpers). Every Part A/B test file stays.

- [ ] **Step 4: Delete the SP1 example files and configs**

```bash
git rm -q examples/tic_tac_toe/env.py examples/tic_tac_toe/networks.py \
          examples/composite_action/env.py examples/composite_action/networks.py \
          examples/space_miners/env.py examples/space_miners/networks.py
git rm -q configs/examples/tic_tac_toe.yaml configs/examples/tic_tac_toe_multi.yaml \
          configs/examples/tic_tac_toe_attention.yaml configs/examples/chase.yaml configs/examples/space_miners.yaml
```

`examples/space_miners/game_engine.py` and the packages' `__init__.py` stay (T8.5 builds the new `space_miners` and `composite_action` games on them).

- [ ] **Step 5: Move the SP2 package and configs into place**

```bash
git ls-files src/colosseum/sp2 | while read -r f; do
  dst="src/colosseum/${f#src/colosseum/sp2/}"
  mkdir -p "$(dirname "$dst")"
  git mv -f "$f" "$dst"
done
rm -rf src/colosseum/sp2
git mv configs/sp2/tic_tac_toe.yaml configs/sp2/tic_tac_toe_multi.yaml configs/sp2/tic_tac_toe_attention.yaml configs/examples/
rmdir configs/sp2
test ! -e src/colosseum/sp2 && test ! -e configs/sp2 && echo "moved"
```

Expected: `moved`. (`src/colosseum/sp2/__init__.py` lands on `src/colosseum/__init__.py`, `sp2/__main__.py` on `colosseum/__main__.py`, `sp2/envs/__init__.py` on `envs/__init__.py`, which drops SP1's re-exports of `BaseEnv` and `VectorEnv`.)

- [ ] **Step 6: Rename pass, then the edits that need final names**

One rename over every tracked file except the frozen material:

```bash
git ls-files -z -- src tests configs examples scripts deployment docs README.md CLAUDE.md pyproject.toml \
  | grep -zv '^docs/superpowers/' \
  | xargs -0 grep -lZ -E 'colosseum[./]sp2|configs/sp2|"configs" / "sp2"' \
  | xargs -0 -r sed -i -E \
      -e 's#colosseum\.sp2\.#colosseum.#g' -e 's#colosseum/sp2/#colosseum/#g' \
      -e 's#colosseum\.sp2\b#colosseum#g' -e 's#colosseum/sp2\b#colosseum#g' \
      -e 's#configs/sp2/#configs/examples/#g' -e 's#"configs" / "sp2"#"configs" / "examples"#g'
```

Test-side names of the transition (the SP2 constants become the only ones; `module=` arguments become the default and are dropped):

```bash
git ls-files -z -- tests | xargs -0 sed -i -E \
  -e 's/\bTTT_SP2_MULTI_CONFIG\b/TTT_MULTI_CONFIG/g' -e 's/\bTTT_SP2_CONFIG\b/TTT_CONFIG/g' -e 's/\bTINY_SP2\b/TINY/g' \
  -e 's/, module=(SP2|"colosseum")//g' -e 's/module=(SP2|"colosseum"), //g' \
  -e 's/"-m", SP2,/"-m", "colosseum",/g' -e '/^SP2 = "colosseum"$/d'
grep -rnE 'module=SP2|"-m", SP2|^SP2 = ' tests || echo "no transition names left"
```

Expected: `no transition names left`.

Replace `tests/cli_runner.py` with its final form (one config schema; `module` defaults to `"colosseum"`):

```python
"""Run ``colosseum train`` in a subprocess for integration and learning tests."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
TTT_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
TTT_MULTI_CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe_multi.yaml"

# Small, fast settings for tic-tac-toe runs (about 10 s with 1 worker on CPU).
TINY: dict[str, str] = {
    "training.total_timesteps": "3000",
    "rollout.num_workers": "1",
    "rollout.envs_per_worker": "8",
    "rollout.chunk_length": "16",
    "rollout.weight_sync_interval_sec": "0.5",
    "rollout.match_refresh_interval_sec": "1.0",
    "learner.batch_chunks": "2",
    "learner.queue_size": "16",
    "checkpoint.interval": "20",
    "checkpoint.pool_size": "5",
    "metrics.log_interval": "1",
    "metrics.console_interval_sec": "1.0",
}


def child_env() -> dict[str, str]:
    """The test process's environment for a CLI child; ``tests/`` goes on PYTHONPATH, so configs
    may name support modules (``game_helpers.*``)."""
    env = dict(os.environ)
    env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")]))
    return env


def train_cmd(config: Path, run_parent: Path, name: str, overrides: dict[str, str] | None = None,
              module: str = "colosseum") -> list[str]:
    """``python -m <module> train -c <config>`` with the ``TINY`` settings, then ``overrides``."""
    sets = {**TINY, **(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", module, "train", "-c", str(config)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


@dataclass
class TrainRun:
    returncode: int
    stdout: str
    stderr: str
    root: Path

    def records(self, kind: str | None = None) -> list[dict]:
        path = self.root / "metrics.jsonl"
        if not path.exists():
            return []
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in records if kind is None or r["kind"] == kind]

    def log(self, process_name: str) -> str:
        return (self.root / "logs" / f"{process_name}.log").read_text()

    def ratings(self) -> dict:
        return json.loads((self.root / "ratings.json").read_text())


def run_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
              timeout: float = 240.0, env: dict[str, str] | None = None, module: str = "colosseum") -> TrainRun:
    """``env`` entries are added to ``child_env()``."""
    run_parent = tmp_path / "runs"
    proc = run_in_session(train_cmd(config, run_parent, name, overrides, module), timeout, env)
    return TrainRun(proc.returncode, proc.stdout, proc.stderr, run_parent / name)


def run_in_session(cmd: list[str], timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` from the repo root in its own session (``env`` added to ``child_env()``);
    whatever is left of the session afterwards (also on a timeout) is SIGKILLed."""
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env={**child_env(), **(env or {})}, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    finally:
        _kill_group(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the session started for ``proc`` (whatever is left of it) and reap ``proc``."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
    except ProcessLookupError:
        pass
    proc.wait()


def start_train(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                env: dict[str, str] | None = None, module: str = "colosseum") -> tuple[subprocess.Popen, Path]:
    """Start training in its own session; stdout/stderr go to files next to the run dir."""
    run_parent = tmp_path / "runs"
    # The child keeps its own copies of the descriptors; the parent's handles close here.
    with open(tmp_path / f"{name}.stdout", "w") as out, open(tmp_path / f"{name}.stderr", "w") as err:
        proc = subprocess.Popen(train_cmd(config, run_parent, name, overrides, module), cwd=REPO_ROOT,
                                env={**child_env(), **(env or {})},
                                stdout=out, stderr=err, text=True, start_new_session=True)
    return proc, run_parent / name


@contextmanager
def training_process(config: Path, tmp_path: Path, name: str = "run", overrides: dict[str, str] | None = None,
                     env: dict[str, str] | None = None,
                     module: str = "colosseum") -> Iterator[tuple[subprocess.Popen, Path]]:
    """``start_train`` whose whole process group (main, children, env grandchildren) is
    SIGKILLed and reaped on exit, so a failing test leaves no orphans behind."""
    proc, root = start_train(config, tmp_path, name, overrides, env, module)
    try:
        yield proc, root
    finally:
        _kill_group(proc)


def wait_for(predicate: Callable[[], bool], timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
```

Apply the remaining edits. Save as `/tmp/t73_finish.py` and run it (it uses the final module names):

```python
"""T7.3 step 6: edits after the move and the rename (run from the repo root, then ``ruff check --fix``).

- surviving SP1 tests lose their ``helpers`` dependency;
- tests/cli_runner.py gets its final form (one config schema, ``colosseum`` only);
- the quickstart test runs the console script again;
- scripts/bench_throughput.py pins the workload in the config v2 schema.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def edit(path: str, pairs: list[tuple[str, str]]) -> None:
    p = Path(path)
    s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, f"{path}: expected exactly one occurrence of {old[:70]!r}"
        s = s.replace(old, new)
    p.write_text(s)


def replace_functions(path: str, functions: dict[str, str]) -> None:
    """Replace (or, with '', delete) top-level functions by name."""
    p = Path(path)
    s = p.read_text()
    tree = ast.parse(s)
    lines = s.splitlines(keepends=True)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert {n.name for n in nodes} == set(functions), f"{path}: missing {set(functions) - {n.name for n in nodes}}"
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        new = functions[node.name]
        lines[start:node.end_lineno] = [textwrap.dedent(new).strip("\n") + "\n"] if new else []
    p.write_text("".join(lines))


# --- tests/unit/test_state.py: the GPU test on the SP2 model protocol --------------------------
edit("tests/unit/test_state.py", [(
    "from helpers import CORE_KINDS, make_simple_model\n",
    "import gymnasium\n\nfrom colosseum.envs.game import RoleSpec\nfrom game_helpers import CORE_KINDS, make_test_model\n",
)])
replace_functions("tests/unit/test_state.py", {"test_model_and_state_move_between_devices": '''
@pytest.mark.gpu
@pytest.mark.parametrize("core", CORE_KINDS)
def test_model_and_state_move_between_devices(core):
    """Model + State pytree: CPU -> CUDA -> CPU keeps devices consistent and log-probs equal."""
    torch.manual_seed(0)
    role = RoleSpec(gymnasium.spaces.Box(-5.0, 5.0, (8,), np.float32), gymnasium.spaces.Discrete(4))
    model = make_test_model(role, core=core)
    obs = torch.randn(3, 8)
    actions = torch.tensor([0, 1, 2])
    state = model.initial_state(3)
    out_cpu = model.step(obs, state)

    model.to("cuda")
    state_cuda = state_to(state, "cuda")
    assert all(t.device.type == "cuda" for t in tree_leaves(state_cuda))
    out_cuda = model.step(obs.to("cuda"), state_cuda)
    assert all(t.device.type == "cuda" for t in tree_leaves(out_cuda.state))
    torch.testing.assert_close(out_cuda.dist.log_prob(actions.to("cuda")).cpu(), out_cpu.dist.log_prob(actions),
                               rtol=1e-4, atol=1e-5)

    model.to("cpu")
    back = state_to(out_cuda.state, "cpu")
    assert all(t.device.type == "cpu" for t in tree_leaves(back))
    for x, y in zip(tree_leaves(back), tree_leaves(out_cpu.state), strict=True):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
    model.step(obs, back)  # the CPU model accepts the moved-back state
'''})

# --- tests/unit/test_infrastructure.py ----------------------------------------------------------
edit("tests/unit/test_infrastructure.py", [(
    "from helpers import REPO_ROOT, example_config\n",
    'REPO_ROOT = Path(__file__).resolve().parents[2]\n\n\ndef example_config(name: str) -> Path:\n'
    '    return REPO_ROOT / "configs" / "examples" / name\n',
)])

# --- tests/unit/test_wandb_logger.py: the launcher test moved to test_sp2_launcher_lifecycle.py --
edit("tests/unit/test_wandb_logger.py", [("from helpers import example_config, make_test_run_dir\n", "")])
replace_functions("tests/unit/test_wandb_logger.py", {"test_launcher_finishes_wandb_when_final_metrics_fail": ""})

# --- tests/unit/test_threads.py: the learner-entry tests moved to test_sp2_learner_entry.py -----
replace_functions("tests/unit/test_threads.py", {
    "_ttt_config": "", "_run_learner_target": "", "test_learner_target_sets_explicit_torch_threads": "",
    "test_learner_target_auto_threads_and_auto_device_resolve_to_cpu": "",
    "test_learner_target_passes_weight_sync_interval": "",
})

# --- tests/integration/test_sp2_lifecycle.py: the quickstart uses the console script again ------
replace_functions("tests/integration/test_sp2_lifecycle.py", {"test_quickstart_from_repo_root": '''
def test_readme_quickstart_from_repo_root(tmp_path):
    exe = Path(sys.executable).parent / "colosseum"
    assert exe.exists(), "console script missing; run scripts/setup-dev.sh"
    cmd = [str(exe), "train", "-c", "configs/examples/tic_tac_toe.yaml",
           "--set", "training.total_timesteps=2000",
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=quickstart"]
    proc = run_in_session(cmd, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "runs" / "quickstart" / "config.resolved.yaml").exists()
'''})

# --- scripts/bench_throughput.py: the pinned workload in the config v2 schema --------------------
edit("scripts/bench_throughput.py", [(
    '''- ``env_steps/s``        env steps per second from the global env-step counter
  (``env_steps`` of the ``system`` records), so envs that mark inactive seats
  (``info["active"]``) are counted correctly;''',
    '''- ``env_steps/s``        env steps per second from the global env-step counter
  (``env_steps`` of the ``system`` records), so turn-based envs (only the acting seat
  decides) are counted correctly;''',
)])
replace_functions("scripts/bench_throughput.py", {"_make_config": '''
def _make_config(num_workers: int, run_parent: str):
    """The pinned benchmark workload. Every workload-relevant value is explicit.

    The run (logs, resolved config, checkpoints) goes to ``<run_parent>/bench-w<num_workers>``.
    """
    from colosseum.core.config import ColosseumConfig

    return ColosseumConfig(
        env={
            "env_class": "examples.tic_tac_toe.game.TicTacToeGame",
            "kwargs": {},
            "max_idle_steps": 1000,
        },
        networks={
            "encoder_class": "examples.tic_tac_toe.models.TicTacToeEncoder",
            "policy_class": "examples.tic_tac_toe.models.TicTacToePolicy",
            "value_class": "examples.tic_tac_toe.models.TicTacToeValue",
            "kwargs": {},
            "core": None,
        },
        algorithm={
            "algorithm_class": "colosseum.algorithms.appo.APPO",
            "gamma": 0.99,
            "vtrace_lambda": 1.0,
            "eps_clip": 0.2,
            "value_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "max_grad_norm": 0.5,
            "num_epochs": 1,
            "minibatch_chunks": 0,
            "vtrace_rho_bar": 1.0,
            "vtrace_c_bar": 1.0,
            "learning_rate": 3.0e-4,
            "lr_schedule": "constant",
            "use_torch_compile": False,
            "normalize_advantages": True,
            "use_amp": False,
            "ratio_mode": "auto",
            "unit_trace": "auto",
            "entropy_reduction": "auto",
        },
        rollout={
            "chunk_length": CHUNK_LENGTH,
            "num_workers": num_workers,
            "envs_per_worker": ENVS_PER_WORKER,
            "weight_sync_interval_sec": 2.0,
            "vec_env": "sync",
            "match_refresh_interval_sec": 0.0,
            "torch_threads": 1,
        },
        learner={
            "device": "cpu",
            "queue_size": QUEUE_SIZE,
            "batch_chunks": BATCH_CHUNKS,
            "weight_push_interval": 5,
            "pin_memory": False,
            "torch_threads": None,  # auto: (cpu_count - num_workers * 1) // 1
        },
        training={
            "total_timesteps": 10**12,  # stopped by the timer, not the budget
            "seed": 0,
        },
        matchmaking={
            "mode": "self_play",
            "layouts": {"2p": 1.0},
            "latest_prob": 0.5,
            "shuffle_seats": True,  # one agent, every seat latest+collect: shuffling is a no-op
        },
        checkpoint={
            "interval": 10**9,  # no checkpoints: every seat collects
            "pool_size": 10,
            "save_optimizer": True,
        },
        run={"dir": run_parent, "name": f"bench-w{num_workers}"},
        metrics={"use_wandb": False, "log_interval": 1, "console_interval_sec": SYSTEM_RECORD_INTERVAL_S},
        transport={"mode": "local"},
    )
'''})
edit("tests/unit/test_bench_throughput.py", [(
    "    assert cfg.networks.core is None\n",
    '    assert cfg.networks.core is None and cfg.matchmaking.layouts == {"2p": 1.0}\n'
    '    assert cfg.env.env_class == "examples.tic_tac_toe.game.TicTacToeGame"\n',
)])
print("T7.3 edits applied")
```

In `pyproject.toml`, drop the deleted support modules from the isort list:

```toml
[tool.ruff.lint.isort]
known-first-party = ["cli_runner", "colosseum", "examples", "game_harness", "game_helpers", "game_learning_envs"]
```

```bash
.venv/bin/python /tmp/t73_finish.py
.venv/bin/ruff check --fix . && .venv/bin/ruff check .
```

Expected: `T7.3 edits applied`, then `All checks passed!`.

- [ ] **Step 7: Verify the switch**

```bash
grep -rn "colosseum.sp2\|colosseum/sp2" src tests configs examples scripts || echo "no colosseum.sp2 left"
for m in envs/base_env.py core/seat_info.py core/action_spec.py networks/distributions.py envs/vec_env.py \
         envs/subproc_vec_env.py worker/slots.py; do
  test ! -e "src/colosseum/$m" || echo "LEFT: $m"
done
grep -rln "from helpers import\|from dataflow_helpers import\|from harness import\|from learning_envs import\|from ttt_eval import" tests \
  || echo "no SP1 test support imported"
.venv/bin/python - <<'EOF'
import importlib
import pkgutil

import colosseum

failed = []
for module in pkgutil.walk_packages(colosseum.__path__, "colosseum."):
    try:
        importlib.import_module(module.name)
    except ModuleNotFoundError as e:
        if e.name not in ("grpc", "wandb", "Box2D"):  # optional extras
            failed.append((module.name, repr(e)))
    except Exception as e:  # noqa: BLE001
        failed.append((module.name, repr(e)))
assert not failed, failed
print("every colosseum module imports")
EOF
.venv/bin/colosseum --help
```

Expected: `no colosseum.sp2 left`, no `LEFT:` line, `no SP1 test support imported`, `every colosseum module imports`, and the help lists `bc`, `eval`, `run-learner`, `run-workers`, `serve-weight-store`, `train`, `validate`.

- [ ] **Step 8: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw && .venv/bin/ruff check .`
Expected: all passed (every `test_sp2_*` file now exercises `colosseum.*`), no warnings; `All checks passed!`.

- [ ] **Step 9: Smoke run of the switched CLI**

```bash
rm -rf /tmp/sp2-smoke
.venv/bin/colosseum validate -c configs/examples/tic_tac_toe.yaml
timeout 300 .venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml \
  --set training.total_timesteps=20000 --set rollout.num_workers=1 \
  --set run.dir=/tmp/sp2-smoke --set run.name=smoke; echo "train exit $?"
latest=$(ls -d /tmp/sp2-smoke/smoke/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
.venv/bin/colosseum eval -c configs/examples/tic_tac_toe.yaml -a "a=$latest" -n 10 --num-envs 2; echo "eval exit $?"
grep -c '"kind": "ratings"' /tmp/sp2-smoke/smoke/metrics.jsonl
rm -rf /tmp/sp2-smoke
```

Expected: `Config is valid.`; the training run ends in about 20 s with `Training budget reached` in its log and `train exit 0`; `eval exit 0` with a `layout 2p` section; a non-zero count of `ratings` records.

- [ ] **Step 10: Commit**

```bash
git add -A src tests configs examples scripts deployment pyproject.toml
git add -u docs README.md CLAUDE.md          # renamed mentions only; never new files under docs/superpowers
git status --porcelain | grep -v '^[MADR] ' || echo "only staged changes"
git commit -m "refactor: switch to the SP2 game model and delete the SP1 legacy (SP2 T7.3)

colosseum.sp2 moves onto colosseum; SP1-only modules, tests, examples and configs are deleted
(successors listed in the plan's coverage gate); the tic-tac-toe example and the benchmark
workload use config v2."
git push origin sp2-game-model
```

`README.md` and the status in `CLAUDE.md` are rewritten in T8.6 (`docs/ENV_GUIDE.md` too); this task only renames `colosseum.sp2` where those files mention it.

---

## Contract notes

Additions and deviations of this part against the overview's interface contract (paths as `colosseum.sp2.*` until T7.3). The reconciliation pass appends the accepted ones to the overview.

**Execution order.** T6.2 runs before T6.1 (the `eval`, `bc` and distributed entry points call `validate_config`, as in SP1), and T7.2 before T7.1 (the ports use the new tic-tac-toe example and configs). The overview's dependency table allows both; this part's header fixes the order.

**Matchmaking (T5.1).**
- Additions: `LineupMatchmaker.pfsp_weight(layout, owner, candidate)`, `LineupMatchmaker.self_play_ratio`; module-level `enabled_layouts(spec, config)` and `PFSP_MIN_WEIGHT`.
- Interpretation of spec block 7, item 4 ("Остальные места команды"): the team's core occupies one seat of a role it plays; only the other seats follow `teammates`. So the owner always has a collecting seat in its own team, also under `teammates: mixed`.
- The match type is drawn before the owner's team and is shared by all opposing teams.

**Ratings (T5.2).**
- `MemberPair` gains `role_a`, `role_b` (defaults `""`) for the per-role win rates.
- `RatingBook.snapshot()` per layout adds `"outcome_kind"` and `"role_win_rates"` (`{role: {a: {b: rate}}}`) to the contract keys.
- Addition: `composition_key(agent_ids)` (shared with eval). `EloRating` is a plain class with the SP1 attribute and method names (constructor `EloRating(k_factor=32.0, initial_rating=1200.0)`). `WinRateTracker.games` counts member pairs (integers) while the rate is weight-weighted.

**Checkpoints, coordinator, metrics (T5.3).**
- `resolve_resume(resume_from, agent_id, expected_signature=None)` and `check_role_signature(state, expected, context)` are additions; `.pt` resume states carry `roles=None`, `role_signature=None`. A checkpoint without `role_signature` is rejected by resume (with an expected signature) and by eval.
- `Coordinator` adds `matchmaker`, `spec`, `agent_roles`, `role_signature(agent_id)`; it writes `roles` and `role_signature` into every `meta.json` itself (the launcher's `meta_extra` stays SP1's: `networks`, `config_hash`, `env_steps`).
- **Deviation:** `colosseum.metrics.jsonl` is listed as "reused unchanged", but its `REQUIRED_KEYS` describe SP1 record shapes; T5.3 creates `colosseum.sp2.metrics.jsonl` (same writer, same kinds, new `REQUIRED_KEYS`). `colosseum.metrics.wandb_logger` and `colosseum.metrics.console` stay reused unchanged.
- **Deviation (record format):** the spec sketches `ratings.json` as `{<layout>: {...}}`; this part writes `{"env_steps": N, "layouts": {<layout>: {...}}}` to `ratings.json` and to `ratings` records, so `env_steps` stays at the top and a layout can never collide with it. WandB keys are `ratings/<layout>/...` as in the spec; episode breakdowns are `episodes/<agent>/by_layout/<layout>/<role>/...` (the overview wrote `episodes/<layout>/...`; the agent level is kept so agents of one layout stay apart).

**Launcher and CLI (T5.4).**
- Additions: `RunSetup`, `setup_run(config, validate=True)`, `validate_run_config(config)`, `_resolve_lineups(lineups, coordinator, agent_ids, already_sent=None)` (replaces `_derive_worker_configs`).
- Signatures that differ from SP1: `_learner_main(..., role_spec, ...)`, `_worker_main(..., agent_roles, agent_configs, role_specs, ..., lineups, ...)`, `Launcher._resolve_resume(agent_configs, role_specs)`.
- **Requirement on Part A:** `RoleSpec` (including `Units` spaces) must be picklable: the launcher passes role specs to spawned learners and workers (`partial(build_model, agent_config, role_spec)`).

**Eval (T6.1).**
- Additions: `summarize(spec, results, *, agents=(), num_matches=0, deterministic=False)`, `evaluate(..., num_envs=8)`, `load_eval_model(config, name, path, *, spec=None)`, `EvalReport.write_json(path)`, `default_layouts(spec, players)`, `PairStats`.
- **Requirement on Part B (T3.3):** `MatchRunner.set_next_lineup(env, lineup)` called from inside `MatchObserver.on_episode_end` is applied at that same episode end (the overview's step order: `on_episode_end` → apply the next lineup). `play_lineups` relies on it to play every lineup exactly once.
- `play_lineups` ignores `network_id`; eval seats use `collect=False`.

**Validate (T6.2).** The implementation lives in `colosseum.sp2.core.validation`; `colosseum.sp2.core.registry.validate_config` is a thin re-export that every caller uses (tests monkeypatch it there). Additions: `random_legal_action(role, mask, rng)`, `VALIDATE_STEPS`. **Requirements on Part A:** `Units.sample(mask=...)` accepts the group's normalized mask tree (`{"unit", "action"}`); `EpisodeTracker.on_reset/on_step` return masks for the acting seats keyed by seat.

**BC (T6.3).** `OfflineBCTrainer(model, action_spec, obs_spec, lr=1e-3, device="cpu", seq_len=64)` (SP1's trainer took only the model); `per_sample_nll(dist, actions, num_deciders)` is new.

**Distributed (T6.4).**
- Additions: `DistributedSetup`, `distributed_setup(config, agent_ids)`, `distributed_lineups(setup, agent_ids, num_envs, rng)`.
- The `.proto` is unchanged: its `behavior_policy_version` field carries the chunk's `policy_version`.
- **Requirement on Part B (T3.1):** `TrajectoryChunk.to_payload()` uses the dataclass field names as keys (`agent_id`, `policy_version`, `initial_state`, `obs`, `global_state`, `actions`, `action_masks`, `kind`, `reward`, `terminal`, `reset_after`, `behavior_logp`, `behavior_unit_logp`); `validate_chunk_payload` and the worker-threads test read them.
- A distributed worker env keeps its layout for the worker's lifetime (no coordinator to redraw it).

**Test support.** `make_test_config(game, **sections)` takes a `TOY_GAMES` name (Part A) and top-level sections as dicts deep-merged onto tiny defaults (`agents` replaces); also `GameTestModel`, `TEST_GAME_AGENTS`, `write_test_config`, `make_coordinator`, `agent_role_of` (T5.3), `make_test_run_dir` (T5.4), `CrashingAPPO` (T7.1) in `tests/game_helpers.py`; `tests/learning/game_learning_envs.py` (T6.3, T7.2). Chunk payloads in this part's tests come from Part B's `chunk_v2_payload` / `learner_role` (T4.4 kit). `tests/cli_runner.py` gains `module=`, `TINY_SP2`, `TESTS_DIR` on the children's `PYTHONPATH` (T5.4) and `TTT_SP2_*` (T7.2); T7.3 folds them back into `TINY` / `TTT_*`. New test files keep their `test_sp2_` basenames after the switch.

**Requirements on Parts A/B used by this part's tests** (checked against the Part A/B plans; listed so a drift is found early):
- `TOY_GAMES` names `solo`, `turns`, `simultaneous`, `ffa`, `dead_teammate`, `units`, `asymmetric`, `coop`, `global_state`; `TurnTakingGame` has the single layout `"2p"`; `EliminationFFA(max_players=4)` has `"2p"`, `"3p"`, `"4p"`; `CoopGame` has `"coop2"`; `AsymmetricGame` has `"1v2"` with roles `hunter` / `prey` of different spaces; every toy game sets `self.spec` in `__init__`.
- `make_test_model(role, core, hidden)` builds a working model for every toy game's role (Box and Dict observations, Discrete and `Units` actions, `global_state`) and for `RoleSpec(Box, Discrete)`.
- `ComposedModel` applies the action mask to the policy's distribution and calls `policy(features, aux)`; `build_model` injects `action_spec` into a policy constructor that names the parameter.
- The learner copy keeps SP1's private helpers `_push_weights` and `_weight_flush_timeout`; `SubprocessVectorEnv` keeps `num_workers`, `_procs` and the env-child log names `worker-<i>-env<k>`; `rollout_worker_process` keeps SP1's process-level behavior (thread limits, logs, stats).

## Open risks

- **Parts A/B drift.** This part was written against the contract only. The copy scripts assert their anchors and fail loudly; the requirements above are the likeliest mismatch points (toy-game layouts, `to_payload` keys, `set_next_lineup` timing, picklable role specs).
- **Learning speed of the ported fast tests (T7.2).** Slot V-trace with BOOT/PAD slots and real truncation changes the targets slightly; the 400-update cap and the `gamma=0` control are SP1's. If a test misses, investigate the targets (Step 5 of T7.2) before touching thresholds; a threshold change is a ruling with measurements.
- **Tic-tac-toe learnability.** Same workload and hyperparameters as SP1; reaching ≥80% is verified only by T8.3's slow test. T7.2 records a manual run.
- **Suite duration.** Until T7.3 the lifecycle and pipeline guarantees run on both CLIs (roughly two extra minutes); T7.3 removes the SP1 half.
- **T7.3 rename.** The `sed` pass relies on `colosseum.sp2` never appearing in a context that must keep it; the frozen directories are excluded. The verification greps and the import sweep catch leftovers.

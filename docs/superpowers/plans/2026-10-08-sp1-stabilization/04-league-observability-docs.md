# SP1 Plan — Part D: League (block 5), Observability/config/lifecycle (block 6), Docs/examples/acceptance (block 8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part finishes SP1 on top of Parts A–C.
- **Block 5 (league):**
  - every trainable agent owns envs in rotation, with N-player arenas and shuffled seats;
  - pairwise ratings from per-seat results, plus `wr_vs_past`;
  - a checkpoint store that cannot delete foreign data, writes atomically and supports resume;
  - a final checkpoint per agent on every stop.
- **Block 6 (observability, config, lifecycle):**
  - a strict config with deep-merged agent overrides and one `--set` parser;
  - a run directory with per-process logs, `metrics.jsonl`, `ratings.json` and console progress;
  - WandB per-agent step axes;
  - exit codes, signal handling and child-death detection.
- **Block 8 (examples, learning tests, docs, acceptance).**

**Read `00-overview.md` first.** It holds the global constraints, the file map and the binding interface contract used below.

**Execution order inside this part** (from the overview):
- T5.1 → T5.2 → T5.3, then T6.1 → T6.2 → T6.3 → T6.4 → T6.5, then T5.4, then T8.1 → T8.2 → T8.3 → T8.4.
- `launcher.py` is edited by T5.1, T5.3, T6.2, T6.3 and T6.5, always in that order.
- Each edit is given against named functions and anchor statements. If an anchor was renamed by Parts A–C, apply the edit to the equivalent statement; the intent is stated next to every edit.

**Conventions used in every task.**
- Run commands from the repository root with `.venv/bin/python`.
- "Full fast suite" means `.venv/bin/python -m pytest -m "not gpu and not slow" -q`.
- New test files use distinctive names, so they never collide with files that T0.2 moved into `tests/unit|contract|integration|learning`.
- When a step says "find old-API uses", it gives the grep to run and a rule per hit. The file names listed in that step are the pre-T0.2 names; T0.2 may have moved them to `tests/<layer>/` with the same basename.

---

### Task T5.1: Owner rotation, N-player arena, seat shuffle

Fixes R4-01, R4-02, R5-15, R6-03, ET-15, and ET-16 (partly). Matches are generated per env for an owner taken in rotation over all trainable agents. League arenas fill all N seats. Seats are shuffled.

**Files:**
- Modify: `src/colosseum/core/config.py` (`SelfPlayConfig`: add `shuffle_seats`)
- Replace: `src/colosseum/coordinator/matchmaker.py` (whole file)
- Modify: `src/colosseum/coordinator/coordinator.py`:
  - `__init__`
  - delete `setup_matchmaker`
  - add `_build_matchmaker`, `next_round` and `refresh_round`
  - replace `generate_match_configs` and `maybe_save_checkpoint`
- Modify: `src/colosseum/launcher.py`:
  - the coordinator construction in `Launcher.launch`
  - the worker-start loop in `Launcher.launch`
  - `_refresh_worker_matches`
- Test: `tests/unit/test_league_matchmaking.py` (new)
- Update old tests: the hits of the grep in Step 8

**Interfaces:**
- Consumes:
  - `ColosseumConfig.get_trainable_agent_ids() -> list[str]`
  - `MatchConfig`, `PlayerSlot` (unchanged)
  - `CheckpointManager.list_checkpoints(agent_id)`, of which only `.checkpoint_id` of each entry is read
  - `WinRateTracker.get_win_rate(a, b) -> float` (unchanged)
- Produces:
  - `SelfPlayConfig.shuffle_seats: bool = True`
  - `Coordinator.__init__(self, config: ColosseumConfig, checkpoint_dir: str | Path | None = None)`. `None` falls back to `config.checkpoint.dir` until T6.2 makes the argument required. The coordinator registers every id from `config.get_trainable_agent_ids()` in its `AgentPool`.
  - `Coordinator.generate_match_configs(self, num_envs: int, env_offset: int) -> list[MatchConfig]`. The owner of global env `g = env_offset + e` is `agents[(g + refresh_round) % n]`.
  - `Coordinator.next_round(self) -> None`
  - `Coordinator.refresh_round -> int` (property)
  - `BaseMatchmaker.match_for(self, owner: str, num_players: int) -> MatchConfig`. Slot 0 is the owner's latest weights, collecting.
  - `SelfPlayMatchmaker(checkpoint_manager, latest_prob=0.5, rng=None)` with `.self_play_slots(owner, num_players) -> list[PlayerSlot]`
  - `PFSPMatchmaker(agent_pool, checkpoint_manager, win_rate_tracker, self_play_ratio=0.5, pfsp_exponent=1.0, latest_prob=0.5, rng=None)` with `.pfsp_weights(owner, candidates) -> list[float]` and `.select_opponents(owner, candidates, k) -> list[str]`
  - Removed: `SimpleSelfPlayMatchmaker`, `BaseMatchmaker.generate_matches`, `Coordinator.setup_matchmaker`, `PFSPMatchmaker._select_opponent`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_league_matchmaking.py`:

```python
"""Owner rotation over all trainable agents, N-player arenas, seat shuffling (T5.1)."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig

TTT = "examples.tic_tac_toe"


def make_config(agents, *, phase="self_play", num_players=2, self_play_ratio=0.5,
                latest_prob=0.5, shuffle_seats=True, seed=0) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": num_players},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": phase, "seed": seed},
        "self_play": {"self_play_ratio": self_play_ratio, "latest_prob": latest_prob,
                      "shuffle_seats": shuffle_seats},
        "agents": {a: {} for a in agents},
    })


def make_coordinator(tmp_path, agents, **kw) -> Coordinator:
    return Coordinator(make_config(agents, **kw), checkpoint_dir=tmp_path / "ckpt")


def run_rounds(coord, *, workers, envs_per_worker, rounds):
    """Generate matches the way the launcher does: per worker, then advance the round."""
    matches = []
    for _ in range(rounds):
        for w in range(workers):
            matches += coord.generate_match_configs(envs_per_worker, env_offset=w * envs_per_worker)
        coord.next_round()
    return matches


def fake_checkpoints(monkeypatch, coord, ids=("ckpt_v10",)):
    monkeypatch.setattr(
        coord.checkpoint_manager, "list_checkpoints",
        lambda agent_id: [SimpleNamespace(checkpoint_id=c, agent_id=agent_id) for c in ids],
    )


def test_self_play_two_agents_both_get_collecting_envs(tmp_path):
    coord = make_coordinator(tmp_path, ["alpha", "beta"])
    matches = run_rounds(coord, workers=2, envs_per_worker=2, rounds=1)
    agents_per_match = [{s.agent_id for s in m.player_slots} for m in matches]
    assert all(len(a) == 1 for a in agents_per_match), "self-play must not mix agents"
    assert Counter(next(iter(a)) for a in agents_per_match) == {"alpha": 2, "beta": 2}
    collecting = Counter(s.agent_id for m in matches for s in m.player_slots if s.collect_trajectories)
    assert collecting["alpha"] > 0 and collecting["beta"] > 0


def test_owner_rotates_with_global_env_index_and_round(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], shuffle_seats=False)

    def owner(g):
        return coord.generate_match_configs(1, env_offset=g)[0].player_slots[0].agent_id

    assert [owner(g) for g in range(4)] == ["a", "b", "c", "a"]
    coord.next_round()
    assert coord.refresh_round == 1
    assert [owner(g) for g in range(3)] == ["b", "c", "a"]


def test_league_three_agents_all_pairs_meet(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], phase="league", self_play_ratio=0.0)
    matches = run_rounds(coord, workers=2, envs_per_worker=4, rounds=20)
    pairs = Counter(tuple(sorted({s.agent_id for s in m.player_slots})) for m in matches)
    assert {("a", "b"), ("a", "c"), ("b", "c")} <= set(pairs), pairs
    for m in matches:
        assert all(s.collect_trajectories and s.checkpoint_id is None for s in m.player_slots)


@pytest.mark.parametrize("phase,agents,ratio", [
    ("league", ["a", "b", "c"], 0.3),
    ("self_play", ["a", "b"], 0.5),
])
def test_seat_distribution_balanced_within_5_percent(tmp_path, monkeypatch, phase, agents, ratio):
    coord = make_coordinator(tmp_path, agents, phase=phase, self_play_ratio=ratio, seed=1)
    fake_checkpoints(monkeypatch, coord)
    matches = run_rounds(coord, workers=2, envs_per_worker=8, rounds=400)
    for agent in agents:
        seats = Counter(i for m in matches for i, s in enumerate(m.player_slots)
                        if s.agent_id == agent and s.collect_trajectories)
        total = sum(seats.values())
        assert total > 1000
        assert abs(seats[0] / total - 0.5) <= 0.05, (agent, seats)


def test_checkpoint_opponents_spread_over_seats(tmp_path, monkeypatch):
    coord = make_coordinator(tmp_path, ["a"], latest_prob=0.0, seed=2)
    fake_checkpoints(monkeypatch, coord)
    matches = run_rounds(coord, workers=1, envs_per_worker=8, rounds=200)
    ckpt_seats = Counter(i for m in matches for i, s in enumerate(m.player_slots) if s.checkpoint_id is not None)
    assert sum(ckpt_seats.values()) == len(matches)  # latest_prob=0: one checkpoint seat per match
    assert abs(ckpt_seats[0] / len(matches) - 0.5) <= 0.05
    for m in matches:
        assert all(not s.collect_trajectories for s in m.player_slots if s.checkpoint_id is not None)


def test_four_player_arena_has_four_slots(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], phase="league", num_players=4, self_play_ratio=0.0)
    matches = run_rounds(coord, workers=1, envs_per_worker=6, rounds=5)
    for m in matches:
        assert len(m.player_slots) == 4
        assert all(s.collect_trajectories and s.checkpoint_id is None for s in m.player_slots)
    assert any(len({s.agent_id for s in m.player_slots}) == 3 for m in matches)


def test_shuffle_disabled_keeps_owner_in_seat_zero(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], phase="league", self_play_ratio=0.0, shuffle_seats=False)
    matches = coord.generate_match_configs(4, env_offset=0)
    assert [m.player_slots[0].agent_id for m in matches] == ["a", "b", "a", "b"]


def test_pfsp_prefers_hard_opponents(tmp_path, monkeypatch):
    coord = make_coordinator(tmp_path, ["a0", "easy", "hard"], phase="league",
                             self_play_ratio=0.0, shuffle_seats=False)
    wr = {"easy": 0.9, "hard": 0.1}
    monkeypatch.setattr(coord.win_rates, "get_win_rate",
                        lambda a, b: wr.get(b, 0.5) if a == "a0" else 0.5)
    picks = Counter()
    for _ in range(600):
        match = coord.generate_match_configs(1, env_offset=0)[0]  # env 0, round 0 -> owner a0
        picks[match.player_slots[1].agent_id] += 1
    assert picks["hard"] > 3 * picks["easy"], picks


def test_single_player_env_gets_solo_matches(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], phase="league", num_players=1, self_play_ratio=0.0)
    matches = coord.generate_match_configs(4, env_offset=0)
    assert all(len(m.player_slots) == 1 and m.player_slots[0].collect_trajectories for m in matches)
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_league_matchmaking.py -v`

Expected: every test fails, with errors such as:
- `TypeError: Coordinator.__init__() got an unexpected keyword argument 'checkpoint_dir'`
- `pydantic ... shuffle_seats ... Extra inputs` (or the field is silently ignored before T6.1)

- [ ] **Step 3: Add the config field**

In `src/colosseum/core/config.py`, class `SelfPlayConfig`, add after the `pfsp_exponent` field:

```python
    shuffle_seats: bool = Field(
        default=True,
        description="Shuffle the seat order of every generated match, so each agent (and each "
                    "checkpoint opponent) plays every seat equally often.",
    )
```

- [ ] **Step 4: Replace the matchmaker module**

Replace the whole content of `src/colosseum/coordinator/matchmaker.py` with:

```python
"""Matchmaking: build the match for one owner agent.

- SelfPlayMatchmaker: the owner's latest weights against its latest weights and
  its own checkpoints.
- PFSPMatchmaker (league): with probability ``self_play_ratio`` a self-play
  match, otherwise an arena of the owner plus N-1 opponents drawn by PFSP
  (with replacement) from the other trainable agents.

The coordinator decides which agent owns each env and shuffles the seats.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

from colosseum.coordinator.agent_pool import AgentPool
from colosseum.coordinator.ratings import WinRateTracker
from colosseum.core.types import MatchConfig, PlayerSlot

logger = logging.getLogger(__name__)


def _new_match(slots: list[PlayerSlot], rng: random.Random) -> MatchConfig:
    return MatchConfig(match_id=f"{rng.getrandbits(48):012x}", env_config={}, player_slots=slots)


class BaseMatchmaker(ABC):
    """Builds one match whose training data belongs to ``owner``."""

    @abstractmethod
    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        """Return a match with ``num_players`` slots.

        Slot 0 is always the owner's latest weights with ``collect_trajectories=True``.
        The coordinator shuffles the seats afterwards.
        """


class SelfPlayMatchmaker(BaseMatchmaker):
    """Self-play against the owner's own history.

    Slot 0: the owner's latest weights (collect). Every other slot: the latest weights
    (collect) with probability ``latest_prob``, otherwise a uniformly random checkpoint
    of the owner (no collect). Without checkpoints every slot is the latest weights.
    """

    def __init__(self, checkpoint_manager, latest_prob: float = 0.5,
                 rng: Optional[random.Random] = None) -> None:
        self._ckpt_mgr = checkpoint_manager
        self._latest_prob = latest_prob
        self._rng = rng or random.Random()

    def self_play_slots(self, owner: str, num_players: int) -> list[PlayerSlot]:
        checkpoints = self._ckpt_mgr.list_checkpoints(owner)
        slots = [PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True)]
        for _ in range(1, num_players):
            if not checkpoints or self._rng.random() < self._latest_prob:
                slots.append(PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True))
            else:
                ckpt = self._rng.choice(checkpoints)
                slots.append(PlayerSlot(agent_id=owner, checkpoint_id=ckpt.checkpoint_id,
                                        collect_trajectories=False))
        return slots

    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        return _new_match(self.self_play_slots(owner, num_players), self._rng)


class PFSPMatchmaker(BaseMatchmaker):
    """League matchmaker: self-play, or an N-player arena chosen by PFSP.

    PFSP weight of candidate ``c`` for owner ``o``: ``max(1e-6, (1 - wr(o, c)) ** p)``.
    Arena opponents are drawn with replacement. Every arena slot plays the latest
    weights and collects trajectories.
    """

    def __init__(self, agent_pool: AgentPool, checkpoint_manager, win_rate_tracker: WinRateTracker,
                 self_play_ratio: float = 0.5, pfsp_exponent: float = 1.0,
                 latest_prob: float = 0.5, rng: Optional[random.Random] = None) -> None:
        self._pool = agent_pool
        self._win_rates = win_rate_tracker
        self._self_play_ratio = self_play_ratio
        self._pfsp_exponent = pfsp_exponent
        self._rng = rng or random.Random()
        self._self_play = SelfPlayMatchmaker(checkpoint_manager, latest_prob, self._rng)

    def pfsp_weights(self, owner: str, candidates: list[str]) -> list[float]:
        return [
            max(1e-6, (1.0 - self._win_rates.get_win_rate(owner, c)) ** self._pfsp_exponent)
            for c in candidates
        ]

    def select_opponents(self, owner: str, candidates: list[str], k: int) -> list[str]:
        return self._rng.choices(candidates, weights=self.pfsp_weights(owner, candidates), k=k)

    def match_for(self, owner: str, num_players: int) -> MatchConfig:
        others = [a.agent_id for a in self._pool.list_trainable() if a.agent_id != owner]
        if num_players < 2 or not others or self._rng.random() < self._self_play_ratio:
            return _new_match(self._self_play.self_play_slots(owner, num_players), self._rng)
        opponents = self.select_opponents(owner, others, num_players - 1)
        slots = [PlayerSlot(agent_id=owner, checkpoint_id=None, collect_trajectories=True)]
        slots += [PlayerSlot(agent_id=o, checkpoint_id=None, collect_trajectories=True) for o in opponents]
        return _new_match(slots, self._rng)
```

- [ ] **Step 5: Update the coordinator**

In `src/colosseum/coordinator/coordinator.py`:

1. Update the imports:
   - add `import random` and `from pathlib import Path` next to the other stdlib imports;
   - replace the `from colosseum.coordinator.matchmaker import (...)` block with:

   ```python
   from colosseum.coordinator.matchmaker import BaseMatchmaker, PFSPMatchmaker, SelfPlayMatchmaker
   ```

2. Replace `__init__` with:

```python
    def __init__(self, config: ColosseumConfig, checkpoint_dir: str | Path | None = None) -> None:
        self._config = config
        # One RNG for matchmaking and seat shuffling: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
        self._agent_pool = AgentPool()
        for agent_id in config.get_trainable_agent_ids():
            self._agent_pool.register_trainable(agent_id)
        self._checkpoint_manager = CheckpointManager(
            base_dir=str(checkpoint_dir if checkpoint_dir is not None else config.checkpoint.dir),
            pool_size=config.self_play.pool_size,
            save_optimizer=config.checkpoint.save_optimizer,
        )
        self._match_results: deque[MatchResult] = deque(maxlen=10000)
        self._elo = EloRating()
        self._win_rates = WinRateTracker()
        self._refresh_round = 0
        self._matchmaker: BaseMatchmaker = self._build_matchmaker()
```

3. Delete the method `setup_matchmaker` entirely. Replace `generate_match_configs` with the three members below, then add `_build_matchmaker`:

```python
    @property
    def refresh_round(self) -> int:
        return self._refresh_round

    def next_round(self) -> None:
        """Advance the owner rotation. The launcher calls this once per match refresh."""
        self._refresh_round += 1

    def generate_match_configs(self, num_envs: int, env_offset: int) -> list[MatchConfig]:
        """One match per env. Env ``e`` of this batch has global index ``g = env_offset + e``.

        Its owner is ``agents[(g + refresh_round) % n_trainable]``, so every trainable
        agent owns envs in every phase, and ownership rotates between refreshes.
        """
        agents = [a.agent_id for a in self._agent_pool.list_trainable()]
        if not agents:
            raise ValueError("Coordinator has no trainable agents")
        num_players = self._config.env.num_players
        shuffle = self._config.self_play.shuffle_seats
        configs: list[MatchConfig] = []
        for e in range(num_envs):
            owner = agents[(env_offset + e + self._refresh_round) % len(agents)]
            match = self._matchmaker.match_for(owner, num_players)
            if shuffle:
                self._rng.shuffle(match.player_slots)
            configs.append(match)
        return configs

    def _build_matchmaker(self) -> BaseMatchmaker:
        sp = self._config.self_play
        if self._config.training.phase == TrainingPhase.LEAGUE:
            return PFSPMatchmaker(
                agent_pool=self._agent_pool,
                checkpoint_manager=self._checkpoint_manager,
                win_rate_tracker=self._win_rates,
                self_play_ratio=sp.self_play_ratio,
                pfsp_exponent=sp.pfsp_exponent,
                latest_prob=sp.latest_prob,
                rng=self._rng,
            )
        return SelfPlayMatchmaker(
            checkpoint_manager=self._checkpoint_manager,
            latest_prob=sp.latest_prob,
            rng=self._rng,
        )
```

4. In `maybe_save_checkpoint`, delete these two lines:

```python
            # Update matchmaker to use new checkpoints
            self.setup_matchmaker(agent_id)
```

   The matchmakers read `list_checkpoints` on every call, so they do not need rebuilding.

- [ ] **Step 6: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_league_matchmaking.py -v`

Expected: all 9 test functions (10 test cases with parametrization) PASS.

- [ ] **Step 7: Update the launcher call sites**

In `src/colosseum/launcher.py`:

1. In `Launcher.launch`, the coordinator now registers its own agents. Replace

```python
        coordinator = Coordinator(cfg)
        for aid in trainable_agents:
            coordinator.agent_pool.register_trainable(aid)
```

with

```python
        coordinator = Coordinator(cfg)
```

2. In the worker-start loop of `Launcher.launch` (`for worker_id in range(cfg.rollout.num_workers):`), replace the call

```python
            match_configs = coordinator.generate_match_configs(
                trainable_agents[0], cfg.rollout.envs_per_worker,
            )
```

with

```python
            match_configs = coordinator.generate_match_configs(
                cfg.rollout.envs_per_worker,
                env_offset=worker_id * cfg.rollout.envs_per_worker,
            )
```

3. In `_refresh_worker_matches`:
   - delete the line `primary = agent_ids[0]`;
   - insert `coordinator.next_round()` right before the `for worker_id, cq in enumerate(command_queues):` loop;
   - inside the loop, replace `coordinator.generate_match_configs(primary, num_envs)` with `coordinator.generate_match_configs(num_envs, env_offset=worker_id * num_envs)`.

- [ ] **Step 8: Update old tests that use removed APIs**

Run:

```bash
grep -rnE "SimpleSelfPlayMatchmaker|generate_matches\(|setup_matchmaker|_select_opponent|generate_match_configs\(\"|generate_match_configs\(['a-z_]+, *num_envs" tests/
```

Apply this rule to each hit:
- **Delete the whole test function.** Its behaviour is now covered by `tests/unit/test_league_matchmaking.py`. At the time of writing, the hits are:
  - `test_milestone2.py::test_matchmaker`
  - `test_milestone2.py::test_coordinator`
  - `test_ratings.py::test_pfsp_all_arena`
  - `test_ratings.py::test_pfsp_all_solo`
  - `test_ratings.py::test_pfsp_prioritizes_hard_opponents`
  - `test_review_fixes.py::test_pfsp_uses_win_rates`
  - `test_review_fixes.py::test_refresh_pushes_new_checkpoint_to_worker`. T5.3 re-adds a refresh test against the new checkpoint API.
- **Any other call** of the form `coordinator.generate_match_configs(<agent>, <n>)`: rewrite it to `coordinator.generate_match_configs(<n>, env_offset=0)`.
- **Tests that call `coord.agent_pool.register_trainable(...)`:** leave them as they are. Registering an extra id is harmless.

Afterwards, remove imports left unused (`ruff check tests/` reports them).

- [ ] **Step 9: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS. If an integration test still passes `agent_id` to `generate_match_configs`, fix it as described in Step 8.

- [ ] **Step 10: Commit**

```bash
git add src/colosseum/core/config.py src/colosseum/coordinator/matchmaker.py src/colosseum/coordinator/coordinator.py src/colosseum/launcher.py tests/
git commit -m "fix: rotate match owners over all agents, N-player arenas, shuffled seats"
```

---
### Task T5.2: Pairwise ratings from seats, `wr_vs_past`, `ratings_snapshot`

Fixes R4-04, ET-03 and G2. Also covers R4-07 (partly: ratings exist in a snapshot that T6.3 persists) and R4-08 (partly: order-independent, N-scaled ELO).

Each pair of seats is compared by `outcome`:
- the higher outcome scores 1, an equal outcome 0.5, the lower 0;
- pairs of different base agents feed the win-rate matrix and ELO, with K/(N−1) and all deltas computed from the pre-match ratings;
- latest-vs-own-checkpoint pairs feed `wr_vs_past`.

**Files:**
- Replace: `src/colosseum/coordinator/ratings.py` (whole file)
- Modify: `src/colosseum/coordinator/coordinator.py`:
  - imports and `__init__` (add `_past`);
  - replace `report_match_result`;
  - delete `_base_agent`;
  - replace `get_ratings_summary` with `ratings_snapshot`;
  - add the property `past_win_rate`.
- Test: `tests/unit/test_league_ratings.py` (new)
- Update old tests: the hits of the grep in Step 6

**Interfaces:**
- Consumes:
  - `MatchResult(match_id, seats: list[SeatResult], episode_length)` and `SeatResult(seat, agent_id, network_id, outcome, reward, rank=None)` from T3.4
  - `network_id` is `"latest"` or `"ckpt_v<N>"`
- Produces:
  - `pairwise_score(outcome_a: float, outcome_b: float) -> float` in `colosseum.coordinator.ratings`
  - `WinRateTracker.record_pair(self, a: str, b: str, score_a: float) -> None`
  - `WinRateTracker.games(self, a, b) -> int`
  - `WinRateTracker.get_games_matrix(self, agent_ids) -> dict[str, dict[str, int]]`
  - `WinRateTracker.get_win_rate`, `get_overall_win_rate` and `get_win_rate_matrix` keep their signatures. The matrix leaves out the diagonal.
  - `EloRating.update_pair(self, a, b, score_a, k_scale=1.0) -> None`
  - `EloRating.update_pairs(self, pairs: Iterable[tuple[str, str, float]], k_scale=1.0) -> None`: simultaneous, order-independent
  - `EloRating.expected(self, a, b) -> float`
  - `PastWinRate(window: int = 500)` with `.record(agent_id, score_latest)`, `.get(agent_id) -> float | None` and `.games(agent_id) -> int`
  - `Coordinator.report_match_result(self, result: MatchResult) -> None`
  - `Coordinator.ratings_snapshot(self) -> dict`, with keys:
    - `"elo"`: `{a: float}`
    - `"win_rates"`: `{a: {b: float}}`
    - `"games"`: `{a: {b: int}}`
    - `"wr_vs_past"`: `{a: float | None}`
    - `"past_games"`: `{a: int}`

    This is a superset of the contract's keys; see Contract notes.
  - `Coordinator.past_win_rate -> PastWinRate`
  - Removed: `EloRating.update`, `WinRateTracker.record`, `Coordinator.get_ratings_summary`, `Coordinator._base_agent`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_league_ratings.py`:

```python
"""Pairwise ratings from per-seat match results (T5.2)."""
from __future__ import annotations

import json

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.coordinator.ratings import EloRating, PastWinRate, WinRateTracker, pairwise_score
from colosseum.core.config import ColosseumConfig
from colosseum.core.types import MatchResult, SeatResult

TTT = "examples.tic_tac_toe"


def make_coordinator(tmp_path, agents, num_players=2) -> Coordinator:
    cfg = ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": num_players},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": "league"},
        "agents": {a: {} for a in agents},
    })
    return Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")


def seat(i, agent, outcome, network="latest", reward=0.0):
    return SeatResult(seat=i, agent_id=agent, network_id=network, outcome=outcome, reward=reward)


def result(*seats, length=5):
    return MatchResult(match_id="m", seats=list(seats), episode_length=length)


def test_pairwise_score():
    assert pairwise_score(1.0, 0.0) == 1.0
    assert pairwise_score(0.0, 1.0) == 0.0
    assert pairwise_score(0.5, 0.5) == 0.5


def test_win_rate_tracker_stores_fractional_sums():
    wr = WinRateTracker()
    wr.record_pair("a", "b", 0.5)
    wr.record_pair("a", "b", 1.0)
    wr.record_pair("a", "b", 0.5)
    assert wr.get_win_rate("a", "b") == pytest.approx(2.0 / 3.0)
    assert wr.get_win_rate("b", "a") == pytest.approx(1.0 / 3.0)
    assert wr.games("a", "b") == 3
    assert wr.get_win_rate("a", "c") == 0.5  # never met
    with pytest.raises(ValueError):
        wr.record_pair("a", "b", 1.5)


def test_elo_update_pairs_is_order_independent():
    pairs = [("x", "y", 1.0), ("x", "z", 1.0), ("y", "z", 0.5)]
    e1, e2 = EloRating(), EloRating()
    e1.update_pairs(pairs, k_scale=0.5)
    e2.update_pairs(list(reversed(pairs)), k_scale=0.5)
    assert e1.all_ratings == pytest.approx(e2.all_ratings)
    assert e1.get("y") == pytest.approx(e1.get("z"))


def test_elo_update_pair_two_player_win_moves_16_points():
    elo = EloRating(k_factor=32.0)
    elo.update_pair("a", "b", 1.0)
    assert elo.get("a") == pytest.approx(1216.0)
    assert elo.get("b") == pytest.approx(1184.0)


def test_ffa_ranking_gives_correct_pairwise_results(tmp_path):
    coord = make_coordinator(tmp_path, ["x", "y", "z"], num_players=3)
    for _ in range(10):
        coord.report_match_result(result(seat(0, "x", 1.0), seat(1, "y", 0.5), seat(2, "z", 0.0)))
    wr = coord.win_rates
    assert wr.get_win_rate("x", "y") == 1.0
    assert wr.get_win_rate("x", "z") == 1.0
    assert wr.get_win_rate("y", "z") == 1.0
    assert wr.get_win_rate("z", "x") == 0.0
    assert coord.elo.get("x") > coord.elo.get("y") > coord.elo.get("z")


def test_eight_player_ffa_winner_moves_like_a_two_player_win(tmp_path):
    agents = [f"p{i}" for i in range(8)]
    coord = make_coordinator(tmp_path, agents, num_players=8)
    coord.report_match_result(result(*[seat(i, a, 1.0 if i == 0 else 0.0) for i, a in enumerate(agents)]))
    assert coord.elo.get("p0") == pytest.approx(1216.0)  # K/(N-1) scaling (R4-08)


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 1, 0], [1, 2, 0]])
def test_ties_are_order_independent(tmp_path, order):
    coord = make_coordinator(tmp_path, ["w", "y", "z"], num_players=3)
    seats = [seat(0, "w", 1.0), seat(1, "y", 0.0), seat(2, "z", 0.0)]
    for _ in range(5):
        coord.report_match_result(result(*[seats[i] for i in order]))
    assert coord.win_rates.get_win_rate("y", "z") == 0.5
    assert coord.win_rates.get_win_rate("z", "y") == 0.5
    assert coord.elo.get("y") == pytest.approx(coord.elo.get("z"))
    reference = make_coordinator(tmp_path / "ref", ["w", "y", "z"], num_players=3)
    for _ in range(5):
        reference.report_match_result(result(*seats))
    assert coord.elo.all_ratings == pytest.approx(reference.elo.all_ratings)


def test_agent_holding_two_seats_counts_every_cross_pair(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], num_players=4)
    coord.report_match_result(result(
        seat(0, "a", 1.0), seat(1, "b", 0.5), seat(2, "a", 0.0), seat(3, "b", 0.5)))
    # cross pairs: a0>b1, a0>b3, a2<b1, a2<b3 -> 2 wins out of 4
    assert coord.win_rates.games("a", "b") == 4
    assert coord.win_rates.get_win_rate("a", "b") == 0.5
    assert coord.elo.get("a") == pytest.approx(coord.elo.get("b"))


def test_same_agent_pairs_do_not_touch_elo_but_update_wr_vs_past(tmp_path):
    coord = make_coordinator(tmp_path, ["a"])
    coord.report_match_result(result(seat(0, "a", 0.0, network="ckpt_v10"), seat(1, "a", 1.0)))
    coord.report_match_result(result(seat(0, "a", 0.5), seat(1, "a", 0.5, network="ckpt_v10")))
    coord.report_match_result(result(seat(0, "a", 1.0), seat(1, "a", 0.0)))  # latest vs latest: no signal
    assert coord.elo.all_ratings == {}
    assert coord.win_rates.games("a", "a") == 0
    assert coord.past_win_rate.get("a") == pytest.approx(0.75)
    assert coord.past_win_rate.games("a") == 2


def test_past_win_rate_window():
    past = PastWinRate(window=4)
    for s in [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]:
        past.record("a", s)
    assert past.get("a") == 1.0
    assert past.get("b") is None


def test_ratings_snapshot_is_json_serializable(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"])
    coord.report_match_result(result(seat(0, "b", 0.0), seat(1, "a", 1.0)))
    snap = coord.ratings_snapshot()
    assert set(snap) == {"elo", "win_rates", "games", "wr_vs_past", "past_games"}
    assert snap["win_rates"]["a"]["b"] == 1.0
    assert snap["games"]["b"]["a"] == 1
    assert snap["wr_vs_past"] == {"a": None, "b": None}
    json.dumps(snap)
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_league_ratings.py -v`

Expected: collection fails with `ImportError: cannot import name 'PastWinRate' from 'colosseum.coordinator.ratings'`.

- [ ] **Step 3: Replace the ratings module**

Replace the whole content of `src/colosseum/coordinator/ratings.py` with:

```python
"""Rating systems: pairwise ELO, pairwise win rates, latest-vs-past win rate.

Every tracker consumes *pairwise* scores (1.0 win, 0.5 draw, 0.0 loss). The
coordinator extracts these scores from the per-seat outcomes of each match.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def pairwise_score(outcome_a: float, outcome_b: float) -> float:
    """1.0 if a's outcome is higher, 0.0 if lower, 0.5 if equal."""
    if outcome_a > outcome_b:
        return 1.0
    if outcome_a < outcome_b:
        return 0.0
    return 0.5


@dataclass
class EloRating:
    """Pairwise ELO.

    ``update_pairs`` computes every delta from the ratings *before* the match and
    applies them together, so the result does not depend on the order of the pairs.
    """

    k_factor: float = 32.0
    initial_rating: float = 1200.0
    _ratings: dict[str, float] = field(default_factory=dict)

    def get(self, agent_id: str) -> float:
        return self._ratings.get(agent_id, self.initial_rating)

    def register(self, agent_id: str) -> None:
        self._ratings.setdefault(agent_id, self.initial_rating)

    def expected(self, a: str, b: str) -> float:
        """Expected score of ``a`` against ``b``."""
        return 1.0 / (1.0 + math.pow(10.0, (self.get(b) - self.get(a)) / 400.0))

    def update_pairs(self, pairs: Iterable[tuple[str, str, float]], k_scale: float = 1.0) -> None:
        """Apply several pairwise results ``(a, b, score_a)`` simultaneously."""
        k = self.k_factor * k_scale
        deltas: dict[str, float] = {}
        for a, b, score_a in pairs:
            e_a = self.expected(a, b)
            deltas[a] = deltas.get(a, 0.0) + k * (score_a - e_a)
            deltas[b] = deltas.get(b, 0.0) + k * ((1.0 - score_a) - (1.0 - e_a))
        for agent_id, delta in deltas.items():
            self._ratings[agent_id] = self.get(agent_id) + delta

    def update_pair(self, a: str, b: str, score_a: float, k_scale: float = 1.0) -> None:
        """Update both ratings from one pairwise result; ``score_a`` in {0, 0.5, 1}."""
        self.update_pairs([(a, b, score_a)], k_scale=k_scale)

    @property
    def all_ratings(self) -> dict[str, float]:
        return dict(self._ratings)


@dataclass
class WinRateTracker:
    """Pairwise win rates stored as float score sums (a draw adds 0.5 to both sides)."""

    _scores: dict[str, dict[str, float]] = field(default_factory=dict)
    _games: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_pair(self, a: str, b: str, score_a: float) -> None:
        """Record one pairwise result: 1.0 a won, 0.5 draw, 0.0 b won."""
        if not 0.0 <= score_a <= 1.0:
            raise ValueError(f"score_a must be in [0, 1], got {score_a}")
        self._scores.setdefault(a, {}).setdefault(b, 0.0)
        self._scores.setdefault(b, {}).setdefault(a, 0.0)
        self._games.setdefault(a, {}).setdefault(b, 0)
        self._games.setdefault(b, {}).setdefault(a, 0)
        self._scores[a][b] += score_a
        self._scores[b][a] += 1.0 - score_a
        self._games[a][b] += 1
        self._games[b][a] += 1

    def games(self, a: str, b: str) -> int:
        return self._games.get(a, {}).get(b, 0)

    def get_win_rate(self, a: str, b: str) -> float:
        """a's score rate against b; 0.5 if they never met."""
        n = self.games(a, b)
        if n == 0:
            return 0.5
        return self._scores[a][b] / n

    def get_overall_win_rate(self, agent_id: str) -> float:
        n = sum(self._games.get(agent_id, {}).values())
        if n == 0:
            return 0.5
        return sum(self._scores.get(agent_id, {}).values()) / n

    def get_win_rate_matrix(self, agent_ids: list[str]) -> dict[str, dict[str, float]]:
        return {a: {b: self.get_win_rate(a, b) for b in agent_ids if b != a} for a in agent_ids}

    def get_games_matrix(self, agent_ids: list[str]) -> dict[str, dict[str, int]]:
        return {a: {b: self.games(a, b) for b in agent_ids if b != a} for a in agent_ids}


class PastWinRate:
    """Score rate of an agent's latest weights against its own checkpoints.

    Only the last ``window`` pairwise scores per agent are kept, so the value tracks
    current progress (the self-play progress signal, R4-07) rather than an all-time mean.
    """

    def __init__(self, window: int = 500) -> None:
        self._window = window
        self._scores: dict[str, deque[float]] = {}

    def record(self, agent_id: str, score_latest: float) -> None:
        self._scores.setdefault(agent_id, deque(maxlen=self._window)).append(float(score_latest))

    def get(self, agent_id: str) -> Optional[float]:
        scores = self._scores.get(agent_id)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def games(self, agent_id: str) -> int:
        return len(self._scores.get(agent_id, ()))
```

- [ ] **Step 4: Update the coordinator**

In `src/colosseum/coordinator/coordinator.py`:

1. Change the ratings import to

```python
from colosseum.coordinator.ratings import EloRating, PastWinRate, WinRateTracker, pairwise_score
```

   Below `logger = logging.getLogger(__name__)`, add:

```python
LATEST_NETWORK_ID = "latest"  # SeatResult.network_id of a seat playing an agent's current weights
```

2. In `__init__`, add after `self._win_rates = WinRateTracker()`:

```python
        self._past = PastWinRate()
```

3. Add this property next to the `win_rates` property:

```python
    @property
    def past_win_rate(self) -> PastWinRate:
        return self._past
```

4. Delete the static method `_base_agent`. Replace `report_match_result` (whatever T3.4 left) and `get_ratings_summary` with:

```python
    def report_match_result(self, result: MatchResult) -> None:
        """Update ratings from one finished match.

        Every pair of seats is compared by ``outcome`` (higher 1, equal 0.5, lower 0).
        - Different base agents: update the win-rate matrix and ELO. ELO deltas are
          computed from the pre-match ratings with K scaled by 1/(N-1).
        - Same agent, one seat ``latest`` and the other a checkpoint: update
          ``wr_vs_past`` from the latest seat's point of view.
        - Two ``latest`` seats of one agent carry no signal and are skipped.
        """
        self._match_results.append(result)
        seats = result.seats
        n = len(seats)
        if n < 2:
            return
        cross: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = seats[i], seats[j]
                score = pairwise_score(a.outcome, b.outcome)
                if a.agent_id != b.agent_id:
                    cross.append((a.agent_id, b.agent_id, score))
                elif a.network_id == LATEST_NETWORK_ID and b.network_id != LATEST_NETWORK_ID:
                    self._past.record(a.agent_id, score)
                elif b.network_id == LATEST_NETWORK_ID and a.network_id != LATEST_NETWORK_ID:
                    self._past.record(b.agent_id, 1.0 - score)
        for a_id, b_id, score in cross:
            self._win_rates.record_pair(a_id, b_id, score)
        if cross:
            self._elo.update_pairs(cross, k_scale=1.0 / (n - 1))

    def ratings_snapshot(self) -> dict:
        """JSON-serializable ratings of all trainable agents (persisted by the metrics hub)."""
        ids = [a.agent_id for a in self._agent_pool.list_trainable()]
        return {
            "elo": {a: self._elo.get(a) for a in ids},
            "win_rates": self._win_rates.get_win_rate_matrix(ids),
            "games": self._win_rates.get_games_matrix(ids),
            "wr_vs_past": {a: self._past.get(a) for a in ids},
            "past_games": {a: self._past.games(a) for a in ids},
        }
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_league_ratings.py tests/unit/test_league_matchmaking.py -v`

Expected: all PASS.

- [ ] **Step 6: Update old tests that use removed APIs**

Run:

```bash
grep -rnE "\.update\(\"|elo\.update\(|\.record\(\"|win_rates\.record\(|get_ratings_summary|_base_agent|player_outcomes=" tests/ src/
```

Apply this rule to each hit:
- **Delete the whole test function.** These tests check the removed single-pair `update`/`record` APIs or key-string aggregation; the new tests cover the same behaviour on the seat API. At the time of writing, the hits are:
  - `test_ratings.py`: `test_elo_win`, `test_elo_draw`, `test_elo_repeated_wins`, `test_win_rate_basic`, `test_win_rate_draw`, `test_win_rate_no_matches`, `test_win_rate_matrix`, `test_coordinator_match_reporting`
  - `test_review_fixes.py`: `test_coordinator_aggregates_composite_keys`, `test_coordinator_skips_same_agent_pairs`. T3.4 may have rewritten these two for seats; delete them anyway, because they duplicate `test_league_ratings.py`.
- **A hit in `src/`:** `grep` must find nothing in `src/`. The only consumers of ratings are the coordinator and the PFSP matchmaker, and both use `get_win_rate`.

If a file ends up with no tests, delete the file.

- [ ] **Step 7: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/coordinator/ratings.py src/colosseum/coordinator/coordinator.py tests/
git commit -m "fix: pairwise seat ratings with fractional win rates, order-independent ELO, wr_vs_past"
```

---
### Task T5.3: Checkpoint store rewrite, final checkpoint, missing-checkpoint fallback, `resolve_resume`

Fixes R4-06, R6-05, R6-06, R3-06, R3-07 (final snapshot), R4-19, R2-06, and R4-10 #3 (no reload of checkpoints already sent).

**Note on scope.** The integration test "final checkpoint exists after a short run" and the resume integration test need the run-dir layout from T6.2. They live in T5.4 (see the overview: T5.4 runs after T6.5). This task covers the same behaviour with unit tests that use real queues and real files.

**Files:**
- Replace: `src/colosseum/coordinator/checkpoint_manager.py` (whole file)
- Modify: `src/colosseum/core/config.py` (add `config_hash`)
- Modify: `src/colosseum/coordinator/coordinator.py`:
  - `__init__`: the `CheckpointManager` arguments;
  - replace `maybe_save_checkpoint` with `save_checkpoint_payload`.
- Modify: `src/colosseum/learner/learner.py`:
  - add `make_checkpoint_payload`, `send_checkpoint`, `apply_resume_state` and `FINAL_CHECKPOINT_TIMEOUT_SEC`;
  - inside `learner_process`: the resume block, the periodic checkpoint block, and the end-of-loop final snapshot.
- Modify: `src/colosseum/launcher.py`:
  - `_COMMAND_QUEUE_SIZE`;
  - replace `_derive_worker_configs`;
  - delete `_resolve_resume_state`;
  - `Launcher.launch` (resume, checkpoint metadata, sent-set bookkeeping, attributes stored on `self`);
  - replace `_refresh_worker_matches` and `_shutdown`;
  - add `_drain_queue`, `_save_checkpoint` and `_drain_all_checkpoints`;
  - the checkpoint block of `_monitor_loop`.
- Modify: `src/colosseum/distributed.py` (`run_distributed_learner`: the checkpoint manager and the drain)
- Test: `tests/unit/test_checkpoint_store.py` (new)
- Update old tests: the hits of the grep in Step 11

**Interfaces:**
- Consumes:
  - `BaseAlgorithm.model`, `.policy_version`, `.state_dict()` and `.load_state_dict()` from T4.6. `state_dict()` contains the key `"policy_version"`, as the contract comment says.
  - `build_model(config)` (T1.4)
  - `ConfigError` (T1.4)
  - `SharedCounter.value` and `.add(n)` (T2.5)
  - `learner_process(..., checkpoint_queue=None, checkpoint_interval=0, resume_state=None, ...)` (contract)
  - `LATEST_NETWORK_ID` exported by `colosseum.worker.rollout_worker`, as today
  - `WorkerCommand.new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]]` (T2.2)
- Produces (in `colosseum.coordinator.checkpoint_manager`):
  - `CheckpointInfo(checkpoint_id: str, agent_id: str, policy_version: int, path: Path, timestamp: float, meta: dict)`
  - `CheckpointManager(base_dir: str | Path, pool_size: int = 20)` with:
    - `.save(agent_id, policy_version, model_state, trainer_state=None, meta_extra=None) -> str`. `trainer_state` may be a `dict` (passed to `torch.save`) or `bytes` (an already-serialized `torch.save` blob, written verbatim).
    - `.load_model(agent_id, checkpoint_id) -> dict[str, np.ndarray]`
    - `.load_trainer_state(agent_id, checkpoint_id) -> dict | None`
    - `.list_checkpoints(agent_id)`, `.latest(agent_id)`, `.base_dir` and `.agents`
  - `resolve_resume(resume_from: str, agent_id: str) -> dict | None`. It returns `{"model_state": dict[str, np.ndarray], "trainer_state": bytes | None, "policy_version": int, "env_steps": int, "source": str}`. `trainer_state` holds the raw bytes of `trainer_state.pt`, so the dict crosses process boundaries as numpy plus bytes. On failure it raises `ConfigError`.
  - `check_model_state(model: nn.Module, model_state: dict[str, np.ndarray], source: str) -> None`, which raises `ConfigError`
  - `numpy_state_to_torch(state) -> dict[str, Tensor]` and `torch_state_to_numpy(state) -> dict[str, np.ndarray]`
  - `MODEL_FILE = "model.pt"`, `TRAINER_FILE = "trainer_state.pt"`, `META_FILE = "meta.json"`
- Produces (in `colosseum.learner.learner`):
  - `FINAL_CHECKPOINT_TIMEOUT_SEC = 30.0`
  - `make_checkpoint_payload(agent_id: str, algorithm: BaseAlgorithm, final: bool = False) -> dict`. Keys:
    - `agent_id`
    - `policy_version`
    - `model_state` (numpy)
    - `trainer_state_bytes` (bytes)
    - `final`
  - `send_checkpoint(q, payload: dict, block: bool, timeout: float = FINAL_CHECKPOINT_TIMEOUT_SEC) -> bool`
  - `apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict) -> None`
- Produces (elsewhere):
  - `colosseum.core.config.config_hash(config: ColosseumConfig) -> str`
  - `Coordinator.save_checkpoint_payload(self, payload: dict, meta_extra: dict | None = None) -> str`
  - `meta.json` keys: `agent_id`, `checkpoint_id`, `policy_version`, `timestamp`, `final`, `networks` (the agent's `NetworkConfig` dumped with `mode="json", by_alias=True`), `config_hash`, `env_steps`.
  - `launcher._derive_worker_configs(match_configs, coordinator, agent_ids, already_sent: dict[str, set[str]] | None = None)`. It returns `(new_checkpoints_by_agent, slot_network_map, collect_mask, slot_agent_map)`.
  - `Launcher._env_counter: SharedCounter` (the T2.5 counter, now stored under this name)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_checkpoint_store.py`:

```python
"""Checkpoint store: confinement, atomic writes, resume, final snapshot, missing-ckpt fallback (T5.3)."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.coordinator import checkpoint_manager as cm_module
from colosseum.coordinator.checkpoint_manager import (
    CheckpointManager, check_model_state, resolve_resume,
)
from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, config_hash
from colosseum.core.errors import ConfigError
from colosseum.core.types import MatchConfig, PlayerSlot
from colosseum.learner.learner import apply_resume_state, make_checkpoint_payload, send_checkpoint

TTT = "examples.tic_tac_toe"


def make_config(**training) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 2},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": "self_play", **training},
        "self_play": {"pool_size": 2, "latest_prob": 0.0, "shuffle_seats": False},
    })


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32), "b": np.zeros(3, np.float32)}


class FakeAlgorithm:
    """Implements the BaseAlgorithm members used by checkpointing (contract T4.6)."""

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


def contains_tensor(obj) -> bool:
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(contains_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_tensor(v) for v in obj)
    return False


def test_eviction_never_touches_dirs_outside_base_dir(tmp_path):
    """Regression for R6-06: meta.json 'path' pointing at another run must be ignored."""
    outside = tmp_path / "other_run" / "agent_0" / "ckpt_v1"
    outside.mkdir(parents=True)
    (outside / "model.pt").write_bytes(b"precious")
    base = tmp_path / "run" / "checkpoints"
    copied = base / "agent_0" / "ckpt_v1"
    copied.mkdir(parents=True)
    torch.save({k: torch.tensor(v) for k, v in sd().items()}, copied / "model.pt")
    (copied / "meta.json").write_text(json.dumps({"path": str(outside), "policy_version": 1}))

    mgr = CheckpointManager(base, pool_size=2)
    mgr.save("agent_0", 2, sd(2))
    mgr.save("agent_0", 3, sd(3))

    assert (outside / "model.pt").read_bytes() == b"precious"
    assert sorted(p.name for p in (base / "agent_0").iterdir()) == ["ckpt_v2", "ckpt_v3"]
    assert [c.checkpoint_id for c in mgr.list_checkpoints("agent_0")] == ["ckpt_v2", "ckpt_v3"]


def test_scan_reads_only_own_dir_and_derives_path(tmp_path):
    base = tmp_path / "ckpts"
    CheckpointManager(base, pool_size=5).save("a", 7, sd(7))
    mgr = CheckpointManager(base, pool_size=5)
    info = mgr.latest("a")
    assert info.checkpoint_id == "ckpt_v7"
    assert info.path == base / "a" / "ckpt_v7"
    assert mgr.agents == ["a"]


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    mgr = CheckpointManager(tmp_path, pool_size=5)
    mgr.save("a", 1, sd(1))

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cm_module.json, "dumps", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        mgr.save("a", 2, sd(2))
    monkeypatch.undo()
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["ckpt_v1"]
    assert [c.checkpoint_id for c in mgr.list_checkpoints("a")] == ["ckpt_v1"]


def test_stale_tmp_dirs_are_cleaned_on_scan(tmp_path):
    (tmp_path / "a" / ".tmp-ckpt_v9-dead").mkdir(parents=True)
    CheckpointManager(tmp_path)
    assert not list((tmp_path / "a").glob(".tmp-*"))


def test_duplicate_id_is_replaced_not_duplicated(tmp_path):
    mgr = CheckpointManager(tmp_path, pool_size=5)
    mgr.save("a", 3, sd(1.0))
    mgr.save("a", 3, sd(30.0), trainer_state={"policy_version": 3})
    assert [c.checkpoint_id for c in mgr.list_checkpoints("a")] == ["ckpt_v3"]
    assert float(mgr.load_model("a", "ckpt_v3")["w"][0, 0]) == 30.0
    assert mgr.load_trainer_state("a", "ckpt_v3") == {"policy_version": 3}


def test_trainer_state_bytes_written_verbatim(tmp_path):
    buf = io.BytesIO()
    torch.save({"policy_version": 4, "x": torch.ones(2)}, buf)
    mgr = CheckpointManager(tmp_path)
    mgr.save("a", 4, sd(), trainer_state=buf.getvalue(), meta_extra={"env_steps": 99})
    assert mgr.load_trainer_state("a", "ckpt_v4")["policy_version"] == 4
    meta = json.loads((tmp_path / "a" / "ckpt_v4" / "meta.json").read_text())
    assert meta["env_steps"] == 99 and meta["agent_id"] == "a" and meta["policy_version"] == 4


def test_resolve_resume_checkpoint_dir_run_dir_and_pt(tmp_path):
    base = tmp_path / "old_run" / "checkpoints"
    mgr = CheckpointManager(base, pool_size=5)
    buf = io.BytesIO()
    torch.save({"policy_version": 12}, buf)
    mgr.save("agent_0", 5, sd(5), meta_extra={"env_steps": 500})
    mgr.save("agent_0", 12, sd(12), trainer_state=buf.getvalue(), meta_extra={"env_steps": 1200})

    from_run = resolve_resume(str(tmp_path / "old_run"), "agent_0")
    assert from_run["policy_version"] == 12 and from_run["env_steps"] == 1200
    assert isinstance(from_run["trainer_state"], bytes)
    assert float(from_run["model_state"]["w"][0, 0]) == 12.0

    from_dir = resolve_resume(str(base / "agent_0" / "ckpt_v5"), "agent_0")
    assert from_dir["policy_version"] == 5 and from_dir["trainer_state"] is None

    pt = tmp_path / "bc.pt"
    torch.save({k: torch.tensor(v) for k, v in sd(7).items()}, pt)
    from_pt = resolve_resume(str(pt), "agent_0")
    assert from_pt["policy_version"] == 0 and from_pt["trainer_state"] is None
    assert float(from_pt["model_state"]["w"][0, 0]) == 7.0

    assert resolve_resume(str(tmp_path / "old_run"), "unknown_agent") is None
    with pytest.raises(ConfigError, match="resume_from"):
        resolve_resume(str(tmp_path / "missing"), "agent_0")
    for result in (from_run, from_dir, from_pt):
        assert not contains_tensor(result)


def test_check_model_state_reports_architecture_mismatch():
    model = nn.Linear(2, 3)
    good = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    check_model_state(model, good, "ok.pt")
    with pytest.raises(ConfigError, match="shape mismatch weight"):
        check_model_state(model, {"weight": np.zeros((3, 4), np.float32), "bias": good["bias"]}, "bc.pt")
    with pytest.raises(ConfigError, match="missing keys"):
        check_model_state(model, {"weight": good["weight"]}, "bc.pt")


def test_final_checkpoint_roundtrip_and_version_continuation(tmp_path):
    algo = FakeAlgorithm()
    for _ in range(3):
        algo.train_once()
    payload = make_checkpoint_payload("agent_0", algo, final=True)
    assert not contains_tensor(payload)

    q = mp.get_context("spawn").Queue(maxsize=2)
    assert send_checkpoint(q, payload, block=True, timeout=5.0)
    received = q.get(timeout=5.0)

    cfg = make_config()
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
    meta_extra = {"networks": cfg.networks.model_dump(mode="json", by_alias=True),
                  "config_hash": config_hash(cfg), "env_steps": 321}
    ckpt_id = coord.save_checkpoint_payload(received, meta_extra=meta_extra)
    assert ckpt_id == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "agent_0" / "ckpt_v3" / "meta.json").read_text())
    assert meta["final"] is True and meta["env_steps"] == 321
    assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder")
    assert len(meta["config_hash"]) == 16

    resumed = FakeAlgorithm()
    apply_resume_state(resumed, resolve_resume(str(tmp_path / "ckpt" / "agent_0" / "ckpt_v3"), "agent_0"))
    assert resumed.policy_version == 3
    for k, v in algo.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v)
    resumed.train_once()
    assert resumed.policy_version == 4  # versions continue after resume


def test_apply_resume_without_trainer_state_keeps_version(tmp_path):
    mgr = CheckpointManager(tmp_path)
    model = FakeAlgorithm().model
    mgr.save("a", 40, {k: v.detach().numpy() for k, v in model.state_dict().items()})
    algo = FakeAlgorithm()
    apply_resume_state(algo, resolve_resume(str(tmp_path / "a" / "ckpt_v40"), "a"))
    assert algo.policy_version == 40


def test_send_checkpoint_nonblocking_reports_full_queue():
    q = mp.get_context("spawn").Queue(maxsize=1)
    assert send_checkpoint(q, {"policy_version": 1}, block=False)
    assert not send_checkpoint(q, {"policy_version": 2}, block=False)


def test_missing_checkpoint_falls_back_to_latest_and_collects(tmp_path, caplog):
    from colosseum.launcher import _derive_worker_configs

    coord = Coordinator(make_config(), checkpoint_dir=tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    match = MatchConfig(match_id="m", player_slots=[
        PlayerSlot("agent_0", None, True),
        PlayerSlot("agent_0", "ckpt_v10", False),
        PlayerSlot("agent_0", "ckpt_v999", False),
    ])
    new_ckpts, nets, collect, agents = _derive_worker_configs([match], coord, ["agent_0"])
    assert nets == [["latest", "ckpt_v10", "latest"]]
    assert collect == [[True, False, True]]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"]
    assert "ckpt_v999" in caplog.text

    again, nets2, _, _ = _derive_worker_configs([match], coord, ["agent_0"],
                                               already_sent={"agent_0": {"ckpt_v10"}})
    assert again["agent_0"] == {}  # already on the worker: not reloaded or resent
    assert nets2 == [["latest", "ckpt_v10", "latest"]]


def test_refresh_marks_checkpoints_sent_only_after_successful_put(tmp_path):
    from colosseum.launcher import Launcher

    cfg = make_config()
    coord = Coordinator(cfg, checkpoint_dir=tmp_path / "ckpt")
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
    assert sent[0]["agent_0"] == {"ckpt_v10"}
    launcher._refresh_worker_matches(coord, ["agent_0"], [q], sent)
    assert q.get(timeout=5).new_checkpoints == {}  # delta only
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_checkpoint_store.py -v`

Expected: collection fails with `ImportError: cannot import name 'check_model_state'` (or `config_hash`).

- [ ] **Step 3: Replace the checkpoint manager**

Replace the whole content of `src/colosseum/coordinator/checkpoint_manager.py` with:

```python
"""Checkpoint storage: atomic per-agent checkpoints, FIFO pool, resume resolution.

Layout (``base_dir`` is ``<run_dir>/checkpoints``)::

    base_dir/<agent_id>/ckpt_v<policy_version>/
        model.pt            torch.save of the model state_dict (CPU tensors)
        trainer_state.pt    torch.save of BaseAlgorithm.state_dict() (optional)
        meta.json           agent_id, checkpoint_id, policy_version, timestamp, + extras

A checkpoint's path is always ``base_dir/agent_id/checkpoint_id``. A ``path`` stored
in ``meta.json`` (older layouts, copied runs) is never used (R6-06).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from colosseum.core.errors import ConfigError

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
TRAINER_FILE = "trainer_state.pt"
META_FILE = "meta.json"
_CKPT_RE = re.compile(r"ckpt_v(\d+)")
_TMP_PREFIX = ".tmp-"


@dataclass
class CheckpointInfo:
    """One complete checkpoint on disk."""

    checkpoint_id: str
    agent_id: str
    policy_version: int
    path: Path
    timestamp: float
    meta: dict = field(default_factory=dict)


def numpy_state_to_torch(state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(np.asarray(v)) for k, v in state.items()}


def torch_state_to_numpy(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def _read_agent_dir(agent_dir: Path) -> list[CheckpointInfo]:
    """Complete checkpoints of one agent dir (model.pt + meta.json), sorted by version."""
    infos: list[CheckpointInfo] = []
    if not agent_dir.is_dir():
        return infos
    for d in agent_dir.iterdir():
        match = _CKPT_RE.fullmatch(d.name)
        if match is None or not d.is_dir():
            continue
        meta_path = d / META_FILE
        if not (d / MODEL_FILE).is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping checkpoint {d}: unreadable {META_FILE} ({e})")
            continue
        infos.append(CheckpointInfo(
            checkpoint_id=d.name,
            agent_id=agent_dir.name,
            policy_version=int(match.group(1)),
            path=d,
            timestamp=float(meta.get("timestamp", 0.0)),
            meta=meta,
        ))
    infos.sort(key=lambda c: c.policy_version)
    return infos


class CheckpointManager:
    """Saves checkpoints atomically and keeps a FIFO pool of ``pool_size`` per agent."""

    def __init__(self, base_dir: str | Path, pool_size: int = 20) -> None:
        self._base_dir = Path(base_dir)
        self._pool_size = pool_size
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def agents(self) -> list[str]:
        return list(self._index.keys())

    def _ckpt_dir(self, agent_id: str, checkpoint_id: str) -> Path:
        return self._base_dir / agent_id / checkpoint_id

    def _scan(self) -> None:
        for agent_dir in sorted(self._base_dir.iterdir()):
            if not agent_dir.is_dir() or agent_dir.name.startswith("."):
                continue
            for stale in agent_dir.glob(f"{_TMP_PREFIX}*"):
                shutil.rmtree(stale, ignore_errors=True)
            infos = _read_agent_dir(agent_dir)
            if infos:
                self._index[agent_dir.name] = infos

    def save(
        self,
        agent_id: str,
        policy_version: int,
        model_state: dict[str, np.ndarray],
        trainer_state: dict | bytes | None = None,
        meta_extra: Optional[dict] = None,
    ) -> str:
        """Write ``ckpt_v<policy_version>`` atomically and evict the oldest beyond ``pool_size``.

        The files are written into ``.tmp-<id>-<rand>/`` and the directory is moved into
        place with ``os.replace``. An existing checkpoint with the same id is replaced.
        """
        checkpoint_id = f"ckpt_v{int(policy_version)}"
        agent_dir = self._base_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        final_dir = agent_dir / checkpoint_id
        tmp_dir = agent_dir / f"{_TMP_PREFIX}{checkpoint_id}-{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir()
        timestamp = time.time()
        meta = {
            "agent_id": agent_id,
            "checkpoint_id": checkpoint_id,
            "policy_version": int(policy_version),
            "timestamp": timestamp,
            **(meta_extra or {}),
        }
        try:
            torch.save(numpy_state_to_torch(model_state), tmp_dir / MODEL_FILE)
            if isinstance(trainer_state, (bytes, bytearray)):
                (tmp_dir / TRAINER_FILE).write_bytes(bytes(trainer_state))
            elif trainer_state is not None:
                torch.save(trainer_state, tmp_dir / TRAINER_FILE)
            (tmp_dir / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
            if final_dir.exists():
                old_dir = agent_dir / f"{_TMP_PREFIX}old-{checkpoint_id}-{uuid.uuid4().hex[:8]}"
                os.replace(final_dir, old_dir)
                os.replace(tmp_dir, final_dir)
                shutil.rmtree(old_dir, ignore_errors=True)
            else:
                os.replace(tmp_dir, final_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        entries = [c for c in self._index.get(agent_id, []) if c.checkpoint_id != checkpoint_id]
        entries.append(CheckpointInfo(checkpoint_id, agent_id, int(policy_version), final_dir, timestamp, meta))
        entries.sort(key=lambda c: c.policy_version)
        while len(entries) > self._pool_size:
            victim = next(c for c in entries if c.checkpoint_id != checkpoint_id)
            entries.remove(victim)
            shutil.rmtree(self._ckpt_dir(agent_id, victim.checkpoint_id), ignore_errors=True)
            logger.debug(f"Evicted checkpoint {victim.checkpoint_id} of {agent_id}")
        self._index[agent_id] = entries
        logger.info(f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(entries)}/{self._pool_size})")
        return checkpoint_id

    def load_model(self, agent_id: str, checkpoint_id: str) -> dict[str, np.ndarray]:
        path = self._ckpt_dir(agent_id, checkpoint_id) / MODEL_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch_state_to_numpy(torch.load(path, map_location="cpu", weights_only=True))

    def load_trainer_state(self, agent_id: str, checkpoint_id: str) -> Optional[dict]:
        path = self._ckpt_dir(agent_id, checkpoint_id) / TRAINER_FILE
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def list_checkpoints(self, agent_id: str) -> list[CheckpointInfo]:
        return list(self._index.get(agent_id, []))

    def latest(self, agent_id: str) -> Optional[CheckpointInfo]:
        entries = self._index.get(agent_id, [])
        return entries[-1] if entries else None


def _load_checkpoint_dir(ckpt_dir: Path) -> dict[str, Any]:
    meta_path = ckpt_dir / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    match = _CKPT_RE.fullmatch(ckpt_dir.name)
    version = int(meta.get("policy_version", match.group(1) if match else 0))
    trainer_path = ckpt_dir / TRAINER_FILE
    return {
        "model_state": torch_state_to_numpy(
            torch.load(ckpt_dir / MODEL_FILE, map_location="cpu", weights_only=True)),
        "trainer_state": trainer_path.read_bytes() if trainer_path.is_file() else None,
        "policy_version": version,
        "env_steps": int(meta.get("env_steps", 0)),
        "source": str(ckpt_dir),
    }


def resolve_resume(resume_from: str, agent_id: str) -> Optional[dict]:
    """Resolve ``training.resume_from`` for one agent.

    Accepted forms:
    - a checkpoint dir (contains ``model.pt``): its weights, trainer state and version;
    - a previous run dir (contains ``checkpoints/``): the agent's latest checkpoint
      there, or ``None`` (with a warning) if the agent has none;
    - a ``.pt`` file (e.g. the output of ``colosseum bc``): weights only, version 0.

    The result holds only numpy arrays, bytes and primitives, so it can be passed
    to a learner process.
    """
    path = Path(resume_from)
    if path.is_dir() and (path / "checkpoints").is_dir():
        infos = _read_agent_dir(path / "checkpoints" / agent_id)
        if not infos:
            logger.warning(f"resume_from={resume_from}: no checkpoints for agent '{agent_id}'; starting fresh")
            return None
        return _load_checkpoint_dir(infos[-1].path)
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return _load_checkpoint_dir(path)
    if path.is_file() and path.suffix == ".pt":
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001 - any unpickling failure means a bad resume source
            raise ConfigError(f"training.resume_from={resume_from!r}: cannot load weights ({e})") from e
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ConfigError(f"training.resume_from={resume_from!r}: expected a state_dict of tensors")
        return {"model_state": torch_state_to_numpy(state), "trainer_state": None,
                "policy_version": 0, "env_steps": 0, "source": str(path)}
    raise ConfigError(
        f"training.resume_from={resume_from!r}: expected a checkpoint dir (containing {MODEL_FILE}), "
        f"a run dir (containing checkpoints/), or a .pt file"
    )


def check_model_state(model: torch.nn.Module, model_state: dict[str, np.ndarray], source: str) -> None:
    """Raise ConfigError if ``model_state`` cannot be loaded into ``model``."""
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(np.asarray(v).shape) for k, v in model_state.items()}
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    if not (missing or unexpected or mismatched):
        return
    lines = [f"Weights from {source} do not match the agent's architecture:"]
    if missing:
        lines.append(f"  missing keys: {missing[:10]}")
    if unexpected:
        lines.append(f"  unexpected keys: {unexpected[:10]}")
    for k in mismatched[:10]:
        lines.append(f"  shape mismatch {k}: checkpoint {got[k]} vs model {expected[k]}")
    raise ConfigError("\n".join(lines))
```

- [ ] **Step 4: Add `config_hash`**

In `src/colosseum/core/config.py`:
- add `import hashlib` and `import json` to the stdlib imports;
- append this function after `load_config`:

```python
def config_hash(config: ColosseumConfig) -> str:
    """Short stable hash of a resolved config (stored in every checkpoint's meta.json)."""
    payload = json.dumps(config.model_dump(mode="json", by_alias=True), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 5: Update the coordinator**

In `src/colosseum/coordinator/coordinator.py`:

1. Add `import io` and `import torch` to the imports.

2. In `__init__`, replace the `CheckpointManager(...)` construction with:

```python
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir if checkpoint_dir is not None else config.checkpoint.dir,
            pool_size=config.self_play.pool_size,
        )
```

3. Replace the method `maybe_save_checkpoint` with:

```python
    def save_checkpoint_payload(self, payload: dict, meta_extra: Optional[dict] = None) -> str:
        """Persist a learner checkpoint payload (see ``learner.make_checkpoint_payload``).

        The trainer state is kept only when ``checkpoint.save_optimizer`` is true.
        """
        trainer_state = None
        if self._config.checkpoint.save_optimizer:
            trainer_state = payload.get("trainer_state_bytes")
        meta = {"final": bool(payload.get("final", False)), **(meta_extra or {})}
        return self._checkpoint_manager.save(
            agent_id=payload["agent_id"],
            policy_version=int(payload["policy_version"]),
            model_state=payload["model_state"],
            trainer_state=trainer_state,
            meta_extra=meta,
        )
```

   Remove `import io` and `import torch` again if ruff reports them unused; the bytes are written verbatim.

- [ ] **Step 6: Learner helpers and the final snapshot**

In `src/colosseum/learner/learner.py`:

1. Add `import io` and `import queue as queue_lib` to the imports, and `import numpy as np` if it is missing. Add these module-level definitions after `logger = ...`:

```python
FINAL_CHECKPOINT_TIMEOUT_SEC = 30.0


def make_checkpoint_payload(agent_id: str, algorithm: BaseAlgorithm, final: bool = False) -> dict:
    """Checkpoint snapshot that may cross a process boundary: numpy weights + bytes.

    The trainer state (optimizer, scheduler, scaler, counters, policy_version) is
    serialized with ``torch.save`` into bytes, because it contains tensors (R6-02).
    """
    buf = io.BytesIO()
    torch.save(algorithm.state_dict(), buf)
    return {
        "agent_id": agent_id,
        "policy_version": int(algorithm.policy_version),
        "model_state": {k: v.detach().cpu().numpy().copy() for k, v in algorithm.model.state_dict().items()},
        "trainer_state_bytes": buf.getvalue(),
        "final": bool(final),
    }


def send_checkpoint(q, payload: dict, block: bool, timeout: float = FINAL_CHECKPOINT_TIMEOUT_SEC) -> bool:
    """Put a checkpoint payload on ``q``. Returns False (and logs) if the queue stays full."""
    try:
        if block:
            q.put(payload, timeout=timeout)
        else:
            q.put_nowait(payload)
        return True
    except queue_lib.Full:
        level = logging.ERROR if block else logging.WARNING
        logger.log(level, f"Checkpoint queue full; dropped snapshot v{payload.get('policy_version')}")
        return False


def apply_resume_state(algorithm: BaseAlgorithm, resume_state: dict) -> None:
    """Load weights and trainer state produced by ``resolve_resume`` into ``algorithm``.

    Without a trainer state, only ``policy_version`` is restored (from meta.json), so
    checkpoint ids keep increasing after a resume.
    """
    model = algorithm.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in resume_state["model_state"].items()})
    blob = resume_state.get("trainer_state")
    if blob is not None:
        algorithm.load_state_dict(torch.load(io.BytesIO(blob), map_location=device, weights_only=True))
    else:
        state = algorithm.state_dict()
        state["policy_version"] = int(resume_state.get("policy_version", 0))
        algorithm.load_state_dict(state)
    logger.info(f"Resumed from {resume_state.get('source')} at policy_version {algorithm.policy_version}")
```

2. In `learner_process`, replace the whole `if resume_state is not None:` block, whatever T4.6 left there, with:

```python
    if resume_state is not None:
        apply_resume_state(algorithm, resume_state)
```

3. Before the main `while` loop, add `last_ckpt_version = -1`. Replace the periodic checkpoint block (the statement that starts with `if (checkpoint_queue is not None` and puts a dict with `"policy_version"` on `checkpoint_queue`) with:

```python
        if checkpoint_queue is not None and checkpoint_interval > 0:
            pv = algorithm.policy_version
            if pv > 0 and pv % checkpoint_interval == 0 and pv != last_ckpt_version:
                if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm), block=False):
                    last_ckpt_version = pv
```

4. After the main `while` loop ends, before the final `logger.info(... finished ...)`, add:

```python
    if checkpoint_queue is not None:
        # Final snapshot on every stop. The main process saves it before tearing children down (R3-07).
        if send_checkpoint(checkpoint_queue, make_checkpoint_payload(agent_id, algorithm, final=True),
                           block=True, timeout=FINAL_CHECKPOINT_TIMEOUT_SEC):
            logger.info(f"Learner [{agent_id}]: sent final checkpoint v{algorithm.policy_version}")
        if hasattr(checkpoint_queue, "join_thread"):
            # Wait until the snapshot is flushed into the pipe; never cancel_join_thread here.
            checkpoint_queue.close()
            checkpoint_queue.join_thread()
```

   If T2.x added a `cancel_join_thread()` call for `checkpoint_queue` anywhere in `learner_process`, delete that call.

- [ ] **Step 7: Launcher: worker configs, resume, checkpoint saving, refresh, shutdown**

In `src/colosseum/launcher.py`:

1. Set `_COMMAND_QUEUE_SIZE = 1`. With one slot, a command that is put has been taken by the worker before the next one can be put, so no command (and none of its `new_checkpoints`) is ever replaced unseen.

2. Replace `_derive_worker_configs` with:

```python
def _derive_worker_configs(
    match_configs: list[MatchConfig],
    coordinator: Coordinator,
    agent_ids: list[str],
    already_sent: Optional[dict[str, set[str]]] = None,
) -> tuple[
    dict[str, dict[str, dict]],  # new checkpoints by agent: {agent_id: {ckpt_id: numpy state_dict}}
    list[list[str]],             # slot_network_map
    list[list[bool]],            # collect_mask
    list[list[str]],             # slot_agent_map
]:
    """Turn match configs into worker slot maps.

    Checkpoints listed in ``already_sent[agent]`` are referenced but not reloaded.
    A checkpoint that cannot be loaded (evicted or missing) is replaced by the
    latest weights with ``collect=True`` and a warning (R4-19).
    """
    from colosseum.worker.rollout_worker import LATEST_NETWORK_ID

    already_sent = already_sent or {}
    new_ckpts: dict[str, dict[str, dict]] = {aid: {} for aid in agent_ids}
    missing: set[tuple[str, str]] = set()
    collect_mask: list[list[bool]] = []
    slot_network_map: list[list[str]] = []
    slot_agent_map: list[list[str]] = []

    for mc in match_configs:
        env_collect: list[bool] = []
        env_nets: list[str] = []
        env_agents: list[str] = []
        for slot in mc.player_slots:
            net_id = LATEST_NETWORK_ID
            collect = slot.collect_trajectories
            ckpt_id = slot.checkpoint_id
            if ckpt_id is not None:
                agent_new = new_ckpts.setdefault(slot.agent_id, {})
                available = ckpt_id in already_sent.get(slot.agent_id, set()) or ckpt_id in agent_new
                if not available and (slot.agent_id, ckpt_id) not in missing:
                    try:
                        agent_new[ckpt_id] = coordinator.checkpoint_manager.load_model(slot.agent_id, ckpt_id)
                        available = True
                    except FileNotFoundError:
                        missing.add((slot.agent_id, ckpt_id))
                        logger.warning(
                            f"Checkpoint {ckpt_id} of {slot.agent_id} is missing; that slot plays "
                            f"the latest weights and collects trajectories"
                        )
                if available:
                    net_id = ckpt_id
                else:
                    collect = True
            env_collect.append(collect)
            env_nets.append(net_id)
            env_agents.append(slot.agent_id)
        collect_mask.append(env_collect)
        slot_network_map.append(env_nets)
        slot_agent_map.append(env_agents)

    return new_ckpts, slot_network_map, collect_mask, slot_agent_map
```

3. Delete the function `_resolve_resume_state`.

4. In `Launcher.launch`:

   a. T2.5 created a `SharedCounter` for the global env-step budget in this method. Make sure it is stored as `self._env_counter`: rename the attribute, or assign `self._env_counter = <that counter>` right after it is created. All later code uses `self._env_counter`.

   b. Right after `coordinator = Coordinator(cfg)`, add the resume resolution and the checkpoint metadata:

```python
        from colosseum.coordinator.checkpoint_manager import check_model_state, resolve_resume
        from colosseum.core.config import config_hash
        from colosseum.core.registry import build_model

        resume_states: dict[str, Optional[dict]] = {aid: None for aid in trainable_agents}
        if cfg.training.resume_from:
            for aid in trainable_agents:
                state = resolve_resume(cfg.training.resume_from, aid)
                if state is not None:
                    check_model_state(build_model(agent_configs[aid]), state["model_state"], state["source"])
                    logger.info(f"Resume [{aid}]: {state['source']} (policy_version {state['policy_version']})")
                resume_states[aid] = state
        start_env_steps = max((s["env_steps"] for s in resume_states.values() if s), default=0)
        if start_env_steps > 0:
            self._env_counter.add(start_env_steps)
            logger.info(f"Resume: env-step counter continues from {start_env_steps}")

        cfg_hash = config_hash(cfg)
        self._checkpoint_meta = {
            aid: {"networks": agent_configs[aid].networks.model_dump(mode="json", by_alias=True),
                  "config_hash": cfg_hash}
            for aid in trainable_agents
        }
```

   If the counter is created later in `launch()` than this point, move the counter creation above this block.

   c. In the learner-start loop, replace `resume_state = _resolve_resume_state(cfg, aid, coordinator)` with `resume_state = resume_states[aid]`. The variable is still passed to `_learner_target` as before.

   d. Rename `worker_broadcast_ckpts` to `worker_sent_ckpts` in `launch()` and in the call to `_monitor_loop`. In the worker-start loop, replace the block that unpacks `_derive_worker_configs(...)` and updates the sent sets with:

```python
            (
                ckpt_dicts_by_agent,
                slot_network_map,
                collect_mask,
                slot_agent_map,
            ) = _derive_worker_configs(match_configs, coordinator, trainable_agents)
            # Initial checkpoints travel as process arguments: delivered by construction.
            for aid, ckpts in ckpt_dicts_by_agent.items():
                worker_sent_ckpts[worker_id].setdefault(aid, set()).update(ckpts)
```

   e. After the queues are created (after `self._all_queues = ...`), store what the monitor and shutdown need:

```python
        self._coordinator = coordinator
        self._agent_ids = list(trainable_agents)
        self._checkpoint_queues = checkpoint_queues
```

5. Add these helpers at module level, after `_derive_worker_configs`:

```python
def _drain_queue(q) -> list:
    """Everything currently in ``q``. A broken item (dead producer) ends the drain with a warning."""
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items
        except (EOFError, OSError) as e:
            logger.warning(f"Dropped an unreadable queue item: {e!r}")
            return items
```

   Add these methods to `Launcher`:

```python
    def _save_checkpoint(self, payload: dict) -> None:
        aid = payload["agent_id"]
        meta = {**self._checkpoint_meta.get(aid, {}), "env_steps": int(self._env_counter.value)}
        self._coordinator.save_checkpoint_payload(payload, meta_extra=meta)

    def _drain_all_checkpoints(self) -> None:
        for cq in self._checkpoint_queues.values():
            for payload in _drain_queue(cq):
                self._save_checkpoint(payload)
```

6. In `_monitor_loop`, replace the block under `# Process per-agent checkpoint saves` (the loop over `checkpoint_queues[aid]` that called `maybe_save_checkpoint`) with:

```python
            # Process per-agent checkpoint saves
            self._drain_all_checkpoints()
```

7. Replace `_refresh_worker_matches` with:

```python
    def _refresh_worker_matches(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
    ) -> None:
        """Advance the owner rotation and send every worker fresh slot maps.

        Each command carries only checkpoints that the worker does not have yet.
        A checkpoint counts as delivered only after its command was put successfully.
        """
        from colosseum.core.types import WorkerCommand

        num_envs = self._config.rollout.envs_per_worker
        coordinator.next_round()
        for worker_id, cq in enumerate(command_queues):
            sent = worker_sent_ckpts[worker_id]
            match_configs = coordinator.generate_match_configs(num_envs, env_offset=worker_id * num_envs)
            new_ckpts, slot_network_map, collect_mask, slot_agent_map = _derive_worker_configs(
                match_configs, coordinator, agent_ids, already_sent=sent,
            )
            cmd = WorkerCommand(
                slot_agent_map=slot_agent_map,
                slot_network_map=slot_network_map,
                collect_mask=collect_mask,
                new_checkpoints={aid: c for aid, c in new_ckpts.items() if c},
            )
            try:
                cq.put_nowait(cmd)
            except queue.Full:
                logger.debug(f"worker-{worker_id} has not consumed its previous command; skipping this refresh")
                continue
            for aid, ckpts in new_ckpts.items():
                sent.setdefault(aid, set()).update(ckpts)
```

8. Replace `_shutdown` with this version. It drains checkpoints while the learners deliver their final snapshot (T6.5 replaces it again with the signal-aware supervisor):

```python
    def _shutdown(self) -> None:
        """Stop children; save every learner's final checkpoint before tearing them down."""
        self._stop_event.set()
        learner_procs = self._processes[: len(getattr(self, "_agent_ids", []))]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(p.is_alive() for p in learner_procs):
            if hasattr(self, "_checkpoint_queues"):
                self._drain_all_checkpoints()
            time.sleep(0.1)
        if hasattr(self, "_checkpoint_queues"):
            self._drain_all_checkpoints()

        for q in self._all_queues:
            try:
                q.cancel_join_thread()
            except (AttributeError, OSError):
                pass
        for proc in self._processes:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        logger.info("All processes stopped")
```

   T6.5 removes the `hasattr` guards together with this whole version.

9. In `_monitor_loop`, the branch that `break`s when all learner processes have exited must not skip checkpoint saving. `_shutdown` now drains after the loop, so nothing else changes.

- [ ] **Step 8: Distributed learner uses the new store**

In `src/colosseum/distributed.py`, `run_distributed_learner`:

1. Replace the `CheckpointManager(...)` construction with:

```python
        coordinator_ckpt = CheckpointManager(
            base_dir=config.checkpoint.dir,
            pool_size=config.self_play.pool_size,
        )
```

2. Replace the body of the nested `_drain_checkpoints` and add a final synchronous drain:

```python
    def _save(data: dict) -> None:
        if coordinator_ckpt is None:
            return
        trainer_state = data.get("trainer_state_bytes") if config.checkpoint.save_optimizer else None
        coordinator_ckpt.save(
            agent_id=agent_id,
            policy_version=int(data["policy_version"]),
            model_state=data["model_state"],
            trainer_state=trainer_state,
            meta_extra={"final": bool(data.get("final", False)),
                        "networks": acfg.networks.model_dump(mode="json", by_alias=True)},
        )

    def _drain_checkpoints():
        while not stop_event.is_set():
            try:
                data = checkpoint_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            _save(data)
```

   In the `finally:` block after `learner_process(...)`, add before `traj_server.stop(0)`:

```python
        while True:  # the final snapshot arrives after stop_event is set
            try:
                _save(checkpoint_queue.get_nowait())
            except queue.Empty:
                break
```

- [ ] **Step 9: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_checkpoint_store.py -v`

Expected: all 13 tests PASS.

- [ ] **Step 10: Grep for remaining old-API uses in `src/`**

Run:

```bash
grep -rnE "maybe_save_checkpoint|_resolve_resume_state|load_optimizer|get_latest|get_random|save_optimizer=|checkpoint_manager\.load\(|\"state_dict\": state_dict_cpu|optimizer_state_dict" src/
```

Expected: no hits, except `save_optimizer` as a config *field* (`config.checkpoint.save_optimizer`). Fix every other hit according to Steps 5–8.

- [ ] **Step 11: Update old tests that use removed APIs**

Run:

```bash
grep -rnE "CheckpointManager\(|maybe_save_checkpoint|_resolve_resume_state|load_optimizer|get_latest|checkpoint_manager\.(save|load)\(|_derive_worker_configs|test_monitor_loop_per_agent_checkpoint_queues" tests/ --include=*.py | grep -v test_checkpoint_store.py
```

Apply this rule to each hit:
- **Delete the whole test function.** Its behaviour is covered by `test_checkpoint_store.py`. At the time of writing, the hits are:
  - `test_milestone2.py::test_checkpoint_manager`
  - `test_milestone2.py::test_derive_worker_configs`
  - `test_multi_agent.py::test_derive_worker_configs`
  - `test_multi_agent.py::test_monitor_loop_per_agent_checkpoint_queues` (it never exercised the monitor loop)
  - `test_review_fixes.py::test_resume_from_path`
  - `test_review_fixes.py::test_resume_from_none_returns_none`
- **`CheckpointManager(base_dir=..., pool_size=..., save_optimizer=...)` elsewhere:** drop `save_optimizer=`.
- **`.save(agent, version, <torch state_dict>)` elsewhere:** pass `{k: v.detach().cpu().numpy() for k, v in sd.items()}` as `model_state`.
- **`.load(agent, ckpt)` elsewhere:** change to `.load_model(agent, ckpt)`, which returns numpy arrays.

- [ ] **Step 12: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add src/colosseum/coordinator/checkpoint_manager.py src/colosseum/coordinator/coordinator.py src/colosseum/core/config.py src/colosseum/learner/learner.py src/colosseum/launcher.py src/colosseum/distributed.py tests/
git commit -m "fix: confined atomic checkpoint store, final snapshot on stop, resume from ckpt/run/pt, safe checkpoint deltas"
```

---
### Task T5.4: Integration: 2-agent self-play to budget, 3-agent league (all pairs, seat balance), resume continuation

**Order:** run this task after T6.5, as the overview requires. It uses `metrics.jsonl`, `ratings.json`, the run-dir layout and the exit codes.

These are the spec §3 contract items "self-play with two agents" and "league on three agents where all pairs meet". They also cover the integration items "final checkpoint" and "resume continues versions and LR" (spec §6), which were moved here from T5.3. Each test is one or two short `colosseum train` subprocess runs with at most 2 workers.

**Files:**
- Test: `tests/integration/test_league_runs.py` (new)

**Interfaces:**
- Consumes:
  - `tests/integration/cli_runner.py`: `run_train`, `TTT_CONFIG`, `TTT_MULTI_CONFIG` and `TrainRun` (T6.2/T6.3)
  - `metrics.jsonl` schema (T6.3)
  - `ratings.json` keys `games` and `wr_vs_past` (T5.2/T6.3)
  - checkpoint layout and `meta.json` keys (T5.3)
  - APPO's learning-rate metric is named `lr` (T4.6). The test also accepts the older `learning_rate`.
  - The linear LR schedule decreases with `progress = env_steps / total_timesteps` (T2.5).
- Produces: nothing new (tests only).

- [ ] **Step 1: Write the tests**

Create `tests/integration/test_league_runs.py`:

```python
"""End-to-end league behaviour through `colosseum train` (T5.4)."""
from __future__ import annotations

import json
from collections import Counter

from tests.integration.cli_runner import TTT_CONFIG, TTT_MULTI_CONFIG, TrainRun, run_train


def checkpoint_versions(run: TrainRun, agent_id: str) -> list[int]:
    agent_dir = run.root / "checkpoints" / agent_id
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def final_meta(run: TrainRun, agent_id: str) -> dict:
    version = checkpoint_versions(run, agent_id)[-1]
    return json.loads((run.root / "checkpoints" / agent_id / f"ckpt_v{version}" / "meta.json").read_text())


def lr_of(record: dict) -> float:
    value = record.get("lr", record.get("learning_rate"))
    assert value is not None, f"no learning-rate metric in {sorted(record)}"
    return float(value)


def test_two_agent_self_play_reaches_budget_and_both_learners_train(tmp_path):
    run = run_train(TTT_MULTI_CONFIG, tmp_path, name="sp2", overrides={
        "training.phase": "self_play",
        "training.total_timesteps": "6000",
        "rollout.num_workers": "2",
    })
    assert run.returncode == 0, run.stderr[-3000:]
    assert "Training budget reached" in run.log("main")
    assert run.records("system")[-1]["env_steps"] >= 6000
    for agent_id in ("agent_alpha", "agent_beta"):
        steps = [r["train_step"] for r in run.records("train") if r["agent"] == agent_id]
        assert steps and max(steps) >= 1, f"{agent_id} never trained"
        assert sum(r["episodes"] for r in run.records("episodes") if r["agent"] == agent_id) > 0
        meta = final_meta(run, agent_id)
        assert meta["final"] is True and meta["agent_id"] == agent_id
        assert meta["networks"]["encoder_class"].endswith("TicTacToeEncoder")


def test_three_agent_league_all_pairs_meet_and_seats_balanced(tmp_path):
    run = run_train(TTT_MULTI_CONFIG, tmp_path, name="league3", overrides={
        "training.phase": "league",
        "self_play.self_play_ratio": "0.0",
        "agents.agent_gamma": "{}",
        "training.total_timesteps": "12000",
    })
    assert run.returncode == 0, run.stderr[-3000:]
    agents = ["agent_alpha", "agent_beta", "agent_gamma"]
    games = run.ratings()["games"]
    for a in agents:
        for b in agents:
            if a != b:
                assert games[a][b] > 0, f"{a} never met {b}: {games}"
    seats: dict[str, Counter] = {a: Counter() for a in agents}
    for record in run.records("episodes"):
        for seat_index, count in enumerate(record["seat_counts"]):
            seats[record["agent"]][seat_index] += count
    for agent_id, counts in seats.items():
        total = sum(counts.values())
        assert total >= 200, (agent_id, counts)
        # ±5% is asserted on 10^4 samples in tests/unit/test_league_matchmaking.py; a short run has
        # ~10^3 seats per agent, so the end-to-end check allows ±8% (≈5 standard deviations).
        assert abs(counts[0] / total - 0.5) <= 0.08, (agent_id, counts)


def test_resume_continues_versions_env_steps_and_lr(tmp_path):
    common = {"algorithm.lr_schedule": "linear", "self_play.checkpoint_interval": "10"}
    first = run_train(TTT_CONFIG, tmp_path, name="first",
                      overrides={**common, "training.total_timesteps": "3000"})
    assert first.returncode == 0, first.stderr[-3000:]
    first_final = checkpoint_versions(first, "agent_0")[-1]
    first_meta = final_meta(first, "agent_0")
    assert first_meta["final"] is True and first_meta["env_steps"] >= 3000
    first_lr = lr_of(first.records("train")[0])

    second = run_train(TTT_CONFIG, tmp_path, name="second", overrides={
        **common, "training.total_timesteps": "6000", "training.resume_from": str(first.root)})
    assert second.returncode == 0, second.stderr[-3000:]
    versions = checkpoint_versions(second, "agent_0")
    assert versions and min(versions) > first_final, (first_final, versions)
    assert final_meta(second, "agent_0")["env_steps"] >= 6000
    second_lr = lr_of(second.records("train")[0])
    # progress starts at first_meta["env_steps"] / 6000 >= 0.5, so the linear LR has already decayed
    assert second_lr < 0.75 * first_lr, (first_lr, second_lr)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/integration/test_league_runs.py -v`

Expected: all 3 tests PASS, each within about 60 s.

If a test fails, the failing assertion message names the agent or pair. Read `runs/<name>/logs/main.log` in the test's `tmp_path`: pytest prints the path in the failure output. The likely causes:
- an agent never trained: T5.1's owner rotation is not wired in `launch()` or `_refresh_worker_matches`;
- a pair never met: `rollout.match_refresh_interval_sec` is not applied, or `next_round` is not called;
- `env_steps` did not continue: T5.3 Step 7 item 4b (`resolve_resume` and `self._env_counter.add`) is missing in `launch()`.

- [ ] **Step 3: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_league_runs.py
git commit -m "test: end-to-end self-play with two agents, three-agent league, resume continuation"
```

---
### Task T6.1: Strict config, partial agent overrides, one `--set` parser, `run` section, seeds after overrides, `validate` checks `num_players`

Fixes R5-07, R5-08, R4-16, R5-10, R3-27, and R5-16 (the `num_players` part).

**Deviation (see Contract notes).** This task adds `RunConfig` and `ColosseumConfig.run`. It does not remove `checkpoint.dir`: that field disappears in T6.2, together with the run directory that replaces it. This way there is no interim checkpoint location.

**Files:**
- Modify: `src/colosseum/core/config.py`:
  - add `StrictModel`, `RunConfig`, `AgentOverride`, `deep_merge`, `parse_override_value`, `apply_overrides`;
  - make every model inherit from `StrictModel`;
  - replace `AgentConfig`, `ColosseumConfig.agents`, `get_agent_config` and `load_config`.
- Create: `src/colosseum/utils/__init__.py` (empty) and `src/colosseum/utils/seeding.py`
- Modify: `src/colosseum/core/registry.py` (add `check_env_num_players`; call it first in `validate_config`)
- Modify: `src/colosseum/launcher.py` (`run_training`)
- Modify: `src/colosseum/distributed.py`:
  - delete `_load`;
  - use `load_config(path, overrides)`;
  - seed `run-learner`.
- Modify: `src/colosseum/cli.py`:
  - `_parse_overrides`;
  - `validate` gets `--set` and clean errors.
- Modify: `configs/examples/*.yaml` (only if a key is now rejected)
- Test: `tests/unit/test_config_overrides.py` (new)
- Update old tests: the hits of the grep in Step 10

**Interfaces:**
- Consumes:
  - `ConfigError` (T1.4)
  - `NetworkConfig` / `CoreConfig` with alias `class` (T1.4)
  - `import_class` and `validate_config` (T1.4)
- Produces (all in `colosseum.core.config` unless noted):
  - `StrictModel(BaseModel)` with `model_config = ConfigDict(extra="forbid")`
  - `RunConfig(name: str | None = None, dir: str = "runs")` and `ColosseumConfig.run: RunConfig`
  - `AgentOverride(networks: dict | None, algorithm: dict | None, learner: dict | None)`, replacing `AgentConfig`
  - `deep_merge(base: dict, override: dict) -> dict`
  - `parse_override_value(raw: str) -> Any`
  - `apply_overrides(data: dict, overrides: dict[str, Any]) -> dict`. An unknown path raises `ConfigError`.
  - `load_config(path, overrides=None) -> ColosseumConfig`. Pydantic errors become `ConfigError`.
  - `ColosseumConfig.get_agent_config(agent_id) -> ColosseumConfig`. It deep-merges the override; an unknown agent raises `ConfigError`.
  - `colosseum.utils.seeding.apply_global_seed(seed: int | None) -> None`
  - `colosseum.core.registry.check_env_num_players(config) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config_overrides.py`:

```python
"""Strict config, deep-merged agent overrides, --set parsing (T6.1)."""
from __future__ import annotations

import datetime
import inspect
import random
from pathlib import Path

import pytest
import torch
import yaml
from click.testing import CliRunner
from pydantic import BaseModel

import colosseum.core.config as config_module
from colosseum.core.config import (
    ColosseumConfig, apply_overrides, deep_merge, load_config, parse_override_value,
)
from colosseum.core.errors import ConfigError

TTT = "examples.tic_tac_toe"
REPO_ROOT = Path(__file__).resolve().parents[2]


def base_data(**extra) -> dict:
    data = {
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 2},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "algorithm": {"lr_schedule": "constant", "learning_rate": 1e-3},
        "learner": {"batch_chunks": 2, "queue_size": 16},
    }
    data.update(extra)
    return data


@pytest.fixture
def restore_root_logging():
    """run_training reconfigures the root logger; put pytest's handlers back afterwards."""
    import logging

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def write_yaml(tmp_path, data, name="cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_every_config_model_forbids_extra_keys():
    models = [obj for _, obj in inspect.getmembers(config_module, inspect.isclass)
              if issubclass(obj, BaseModel) and obj.__module__ == config_module.__name__]
    assert len(models) >= 12
    for model in models:
        assert model.model_config.get("extra") == "forbid", model.__name__


@pytest.mark.parametrize("data", [
    base_data(rollot={}),
    base_data(rollout={"num_worker": 3}),
    base_data(training={"kickstart_teachr": "x.pt"}),
    base_data(agents={"x": {"algoritm": {}}}),
    base_data(agents={"x": {"algorithm": {"learnin_rate": 1.0}}}),
    base_data(agents={"x": {"training": {}}}),
])
def test_typos_are_rejected(tmp_path, data):
    with pytest.raises(ConfigError):
        load_config(write_yaml(tmp_path, data))


def test_agent_override_deep_merges_onto_global_section(tmp_path):
    data = base_data(agents={
        "alpha": None,
        "beta": {"algorithm": {"learning_rate": 1e-4}, "learner": {"batch_chunks": 8}},
    })
    cfg = load_config(write_yaml(tmp_path, data))
    beta = cfg.get_agent_config("beta")
    assert beta.algorithm.learning_rate == 1e-4
    assert beta.algorithm.lr_schedule == cfg.algorithm.lr_schedule  # kept the global "constant"
    assert beta.learner.batch_chunks == 8 and beta.learner.queue_size == 16
    alpha = cfg.get_agent_config("alpha")
    assert alpha.algorithm.learning_rate == 1e-3
    assert beta.agents == {}


def test_partial_networks_override_merges_core_kwargs(tmp_path):
    data = base_data()
    data["networks"]["core"] = {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 16}}
    data["agents"] = {"big": {"networks": {"core": {"kwargs": {"hidden_size": 64}}}}, "flat": {"networks": {"core": None}}}
    cfg = load_config(write_yaml(tmp_path, data))
    big = cfg.get_agent_config("big").networks
    assert big.core.class_path == "colosseum.networks.cores.LSTMCore"
    assert big.core.kwargs == {"hidden_size": 64}
    assert cfg.get_agent_config("flat").networks.core is None


def test_unknown_agent_id_raises():
    cfg = ColosseumConfig.model_validate(base_data(agents={"alpha": {}}))
    with pytest.raises(ConfigError, match="Unknown agent"):
        cfg.get_agent_config("alpah")
    single = ColosseumConfig.model_validate(base_data())
    assert single.get_agent_config("agent_0").learner.queue_size == 16
    with pytest.raises(ConfigError):
        single.get_agent_config("other")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": [1]}}
    out = deep_merge(base, {"a": {"b": 2}})
    assert out == {"a": {"b": 2, "c": [1]}} and base == {"a": {"b": 1, "c": [1]}}


@pytest.mark.parametrize("raw,expected", [
    ("null", None), ("~", None), ("", None),
    ("3", 3), ("-2", -2), ("3.0", 3.0), ("1e-4", 1e-4), ("1.5e3", 1500.0),
    ("true", True), ("false", False),
    ("[1, 2]", [1, 2]), ("{a: 1}", {"a": 1}), ("{}", {}),
    ("abc", "abc"), ("runs/x", "runs/x"), ("2026-10-08", "2026-10-08"),
])
def test_parse_override_value(raw, expected):
    value = parse_override_value(raw)
    assert value == expected and type(value) is type(expected)


def test_apply_overrides_sets_nested_values_and_creates_agent_sections():
    data = apply_overrides(base_data(), {
        "rollout.num_workers": 8,
        "training.resume_from": None,
        "agents.alpha.algorithm.learning_rate": 1e-4,
        "env.kwargs.size": 5,
        "run.name": "exp",
    })
    cfg = ColosseumConfig.model_validate(data)
    assert cfg.rollout.num_workers == 8 and cfg.training.resume_from is None
    assert cfg.get_agent_config("alpha").algorithm.learning_rate == 1e-4
    assert cfg.env.kwargs == {"size": 5} and cfg.run.name == "exp"


@pytest.mark.parametrize("key", [
    "rollout.num_worker", "rollot.x", "rollout.num_workers.x", "agents.a.trainin.x", "env.env_class.x", "a..b",
])
def test_apply_overrides_unknown_path_raises(key):
    with pytest.raises(ConfigError):
        apply_overrides(base_data(), {key: 1})


def test_load_config_applies_overrides_before_validation(tmp_path):
    path = write_yaml(tmp_path, base_data())
    cfg = load_config(path, {"learner.batch_chunks": 4, "agents.alpha.learner.queue_size": 8})
    assert cfg.learner.batch_chunks == 4
    assert cfg.get_agent_config("alpha").learner.queue_size == 8
    assert cfg.get_agent_config("alpha").learner.batch_chunks == 4


def test_seed_is_applied_after_overrides(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.launcher as launcher_module

    class FakeLauncher:
        def __init__(self, *args, **kwargs):
            pass

        def launch(self):
            return 0

    monkeypatch.setattr(launcher_module, "Launcher", FakeLauncher)
    path = write_yaml(tmp_path, base_data(training={"seed": 1}))
    launcher_module.run_training(str(path), {"training.seed": 123, "run.dir": str(tmp_path / "runs")})
    assert torch.initial_seed() == 123
    assert random.random() == random.Random(123).random()


def test_validate_rejects_num_players_mismatch():
    from colosseum.core.registry import validate_config

    cfg = ColosseumConfig.model_validate(base_data(env={"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 3}))
    with pytest.raises(ConfigError, match="num_players"):
        validate_config(cfg)


def test_cli_validate_reports_typo_with_exit_1(tmp_path, monkeypatch):
    from colosseum.cli import main

    monkeypatch.chdir(REPO_ROOT)
    bad = write_yaml(tmp_path, base_data(rollout={"num_worker": 3}))
    result = CliRunner().invoke(main, ["validate", "-c", str(bad)])
    assert result.exit_code == 1
    assert "num_worker" in result.output
    good = write_yaml(tmp_path, base_data(), name="good.yaml")
    ok = CliRunner().invoke(main, ["validate", "-c", str(good), "--set", "rollout.num_workers=2"])
    assert ok.exit_code == 0, ok.output
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_config_overrides.py -v`

Expected: collection fails with `ImportError: cannot import name 'apply_overrides'`.

- [ ] **Step 3: Make every config model strict**

In `src/colosseum/core/config.py`:

1. Change the pydantic import to

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
```

   Also add `import copy`, `import datetime`, `import re`, `import types` and `import typing` to the stdlib imports, and add `from colosseum.core.errors import ConfigError`.

2. Insert this class above the first section config:

```python
class StrictModel(BaseModel):
    """Base for every config model: unknown keys are errors, not silently ignored (R5-07)."""

    model_config = ConfigDict(extra="forbid")
```

3. Change the base class of **every** model in this module from `BaseModel` to `StrictModel`, including any section that Parts A–C added (for example a BC section). Where a model already sets `model_config = ConfigDict(...)` (for example `CoreConfig` with `populate_by_name=True`), keep it and make sure it contains `extra="forbid"`.

   Check: `grep -n "(BaseModel)" src/colosseum/core/config.py` must print only the line of `class StrictModel(BaseModel):`.

4. Add the run section after `MetricsConfig`:

```python
class RunConfig(StrictModel):
    """Where a training run writes its outputs: ``<dir>/<name>/``."""

    name: Optional[str] = Field(
        default=None,
        description="Run name; default '<config_stem>-<YYYYmmdd-HHMMSS>'. An existing non-empty run dir "
                    "with an explicit name is an error.",
    )
    dir: str = Field(default="runs", description="Parent directory of all runs.")
```

- [ ] **Step 4: Agent overrides, deep merge, `get_agent_config`**

In `src/colosseum/core/config.py`:

1. Replace the class `AgentConfig` with:

```python
class AgentOverride(StrictModel):
    """Per-agent overrides as partial dicts.

    They are deep-merged onto the global section before validation (R5-08), so an
    override that sets only ``learning_rate`` keeps every other global algorithm value.
    """

    networks: Optional[dict[str, Any]] = None
    algorithm: Optional[dict[str, Any]] = None
    learner: Optional[dict[str, Any]] = None


_AGENT_SECTIONS = ("networks", "algorithm", "learner")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Non-dict values replace."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
```

2. In `ColosseumConfig`:
   - add the field `run: RunConfig = Field(default_factory=RunConfig)` after `transport`;
   - change the `agents` field annotation to `dict[str, AgentOverride]`, keeping its default and description;
   - replace `get_agent_config` with the code below and add the two validators:

```python
    @field_validator("agents", mode="before")
    @classmethod
    def _null_agent_means_no_override(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ({} if v is None else v) for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _validate_agent_overrides(self) -> "ColosseumConfig":
        for agent_id in self.agents:
            try:
                self.get_agent_config(agent_id)
            except ValidationError as e:
                raise ValueError(f"agents.{agent_id}: invalid override:\n{e}") from None
        return self

    def get_agent_config(self, agent_id: str) -> ColosseumConfig:
        """Effective config of one agent: global sections deep-merged with its override."""
        if self.agents and agent_id not in self.agents:
            raise ConfigError(f"Unknown agent '{agent_id}'. Known agents: {sorted(self.agents)}")
        if not self.agents and agent_id != "agent_0":
            raise ConfigError(f"Unknown agent '{agent_id}': without an 'agents' section the only agent is 'agent_0'")
        data = self.model_dump(by_alias=True)
        override = self.agents.get(agent_id)
        if override is not None:
            for section in _AGENT_SECTIONS:
                part = getattr(override, section)
                if part:
                    data[section] = deep_merge(data[section], part)
        data["agents"] = {}
        return ColosseumConfig.model_validate(data)
```

- [ ] **Step 5: `--set` parsing, `apply_overrides`, `load_config`**

Replace `load_config` in `src/colosseum/core/config.py` with the block below. Keep `config_hash` (added in T5.3) after it.

```python
_NUMBER_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def parse_override_value(raw: str) -> Any:
    """Parse a ``--set key=value`` value with YAML semantics.

    ``null`` gives None, lists and dicts are YAML, and numbers include ``1e-4`` (which
    plain YAML 1.1 would keep as a string). A date-like value stays a string.
    """
    if not raw.strip():
        return None
    value = yaml.safe_load(raw)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return raw
    if isinstance(value, str) and _NUMBER_RE.fullmatch(value.strip()):
        number = float(value)
        is_int_literal = "." not in value and "e" not in value.lower()
        return int(number) if is_int_literal else number
    return value


def _unwrap_optional(tp: Any) -> Any:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _check_override_path(parts: list[str]) -> None:
    """Walk the schema of ColosseumConfig along ``parts``; raise ConfigError on an unknown key."""
    tp: Any = ColosseumConfig
    for i, part in enumerate(parts):
        tp = _unwrap_optional(tp)
        where = ".".join(parts[: i + 1])
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            fields = tp.model_fields
            name = next((n for n, f in fields.items() if part in (n, f.alias)), None)
            if name is None:
                raise ConfigError(f"Unknown config key '{where}' (valid keys here: {sorted(fields)})")
            tp = fields[name].annotation
        elif typing.get_origin(tp) is dict:
            tp = typing.get_args(tp)[1]
        elif tp is Any:
            return  # free-form dict (env.kwargs, agent override bodies): checked at validation
        else:
            raise ConfigError(f"Cannot set '{'.'.join(parts)}': '{'.'.join(parts[:i])}' is not a section")


def apply_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of raw config ``data`` with dotted-path ``overrides`` applied.

    Missing intermediate sections are created (e.g. ``agents.alpha.algorithm``). An
    unknown path raises ConfigError. Values are validated later by ``model_validate``.
    """
    out = copy.deepcopy(data)
    for key, value in overrides.items():
        parts = key.split(".")
        if not all(parts):
            raise ConfigError(f"Malformed override key '{key}'")
        _check_override_path(parts)
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise ConfigError(f"Cannot set '{key}': '{part}' holds a {type(child).__name__}, not a section")
            node = child
        node[parts[-1]] = value
    return out


def load_config(path: str | Path, overrides: Optional[dict[str, Any]] = None) -> ColosseumConfig:
    """Read YAML, apply ``--set`` overrides, validate. Any problem raises ConfigError."""
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"Cannot read config {path}: {e}") from e
    if overrides:
        raw = apply_overrides(raw, overrides)
    try:
        return ColosseumConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config {path}:\n{e}") from e
```

- [ ] **Step 6: Seeding helper and `num_players` check**

Create `src/colosseum/utils/__init__.py` with the content `"""Process-level utilities: logging, seeding, lifecycle."""`.

Create `src/colosseum/utils/seeding.py`:

```python
"""Global seeding of python, numpy and torch."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def apply_global_seed(seed: Optional[int]) -> None:
    """Seed python, numpy and torch in this process. ``None`` leaves the RNGs alone."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Global seed set to {seed}")
```

In `src/colosseum/core/registry.py`, add this function. Import `ConfigError` from `colosseum.core.errors` if the module does not already.

```python
def check_env_num_players(config: ColosseumConfig) -> None:
    """``env.num_players`` in the config must equal the env's own ``num_players`` (R5-16)."""
    env = import_class(config.env.env_class)(**config.env.kwargs)
    try:
        actual = int(env.num_players)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    if actual != config.env.num_players:
        raise ConfigError(
            f"env.num_players={config.env.num_players} but {config.env.env_class}.num_players={actual}"
        )
```

Make `check_env_num_players(config)` the first statement of `validate_config`.

- [ ] **Step 7: One override path for `train`, `run-learner`, `run-workers`, `validate`**

1. In `src/colosseum/launcher.py`, replace `run_training` with:

```python
def run_training(config_path: str, overrides: dict | None = None) -> None:
    """Entry point: load config (overrides applied), seed, launch training."""
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(config_path, overrides)
    apply_global_seed(config.training.seed)  # after overrides: --set training.seed works (R3-27)
    launcher = Launcher(config)
    launcher.launch()
```

   Remove the now-unused `ColosseumConfig(**config_dict)` code and the inline seeding.

2. In `src/colosseum/distributed.py`:
   - delete the helper `_load`;
   - replace both `config = _load(config_path, overrides)` with `config = load_config(config_path, overrides)`;
   - in `run_distributed_learner`, right after that line, add:

```python
    from colosseum.utils.seeding import apply_global_seed
    apply_global_seed(config.training.seed)
```

   Remove `ColosseumConfig` from the import if it is now unused (`ruff check` reports it).

3. In `src/colosseum/cli.py`, replace `_parse_overrides` with:

```python
def _parse_overrides(overrides: tuple[str, ...]) -> dict:
    """Parse ``--set key=value`` pairs; values use YAML semantics (null, numbers, lists)."""
    from colosseum.core.config import parse_override_value

    result = {}
    for ov in overrides:
        if "=" not in ov:
            raise click.BadParameter(f"Override must be key=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        result[key.strip()] = parse_override_value(value)
    return result
```

   Replace the `validate` command with:

```python
@main.command("validate")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--set", "overrides", multiple=True, help="Override config values (e.g., --set env.num_players=2)")
def validate_cmd(config: str, overrides: tuple[str, ...]) -> None:
    """Validate a config: schema, env num_players, and a dummy forward of every agent's model."""
    import sys

    from colosseum.core.config import load_config
    from colosseum.core.errors import ConfigError
    from colosseum.core.registry import validate_config

    try:
        cfg = load_config(config, _parse_overrides(overrides) or None)
        for aid in cfg.get_trainable_agent_ids():
            validate_config(cfg.get_agent_config(aid))
            click.echo(f"  OK: agent '{aid}'")
    except ConfigError as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)
    click.echo("Config is valid.")
```

- [ ] **Step 8: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_config_overrides.py -v`

Expected: all PASS.

- [ ] **Step 9: Example configs still load**

Run:

```bash
for f in configs/examples/*.yaml; do .venv/bin/python -c "from colosseum.core.config import load_config; load_config('$f')" || echo "FAIL $f"; done
```

Expected: no `FAIL` line. For any failure, delete the rejected key from that YAML: the error names it. With the configs after Parts A–C, no key should be rejected.

- [ ] **Step 10: Update old tests**

Run:

```bash
grep -rnE "AgentConfig\b|_parse_overrides|_load\(|load_config\([^)]*\)\s*$|test_get_agent_config_with_override|test_agent_config_defaults" tests/ --include=*.py | grep -v test_config_overrides.py
```

Apply this rule to each hit:
- **`AgentConfig` imports and uses:** rename to `AgentOverride`. A test that asserts `AgentConfig().networks is None` becomes `assert AgentOverride().networks is None`.
- **`test_multi_agent.py::test_get_agent_config_with_override`:** delete it. Overrides are now partial dicts, and `test_agent_override_deep_merges_onto_global_section` covers the new semantics.
- **Tests that build an `agents` section with `AgentConfig(networks=NetworkConfig(...))`:** pass a dict instead, e.g. `{"networks": {...}}`.
- **Tests that write unknown keys into a config dict** (anything `extra="forbid"` now rejects): remove the key.

- [ ] **Step 11: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add src/colosseum/core/config.py src/colosseum/core/registry.py src/colosseum/utils/__init__.py src/colosseum/utils/seeding.py src/colosseum/launcher.py src/colosseum/distributed.py src/colosseum/cli.py configs/examples tests/
git commit -m "feat: strict config, deep-merged agent overrides, yaml --set parsing, run section, seeds after overrides"
```

---
### Task T6.2: Run directory, per-process logging, resolved config

Fixes R5-01, R3-16, R5-12 (b, d), R5-13, R6-08 (logging part).

Every run writes to `runs/<name>/`:
- `config.resolved.yaml`;
- `logs/main.log`, `logs/learner-<agent>.log`, `logs/worker-<i>.log`, plus `logs/worker-<i>-env<k>.log` for subprocess env children;
- `checkpoints/`.

Every process target calls `setup_process_logging` first. The `checkpoint.dir` config field is removed.

**Files:**
- Create: `src/colosseum/core/run_dir.py`, `src/colosseum/utils/logging.py`, `src/colosseum/utils/process.py`
- Modify: `src/colosseum/core/config.py` (remove `CheckpointConfig.dir`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`checkpoint_dir` becomes required)
- Modify: `src/colosseum/launcher.py`:
  - process targets;
  - `Launcher.__init__` and `Launcher.launch`;
  - `run_training`.
- Modify: `src/colosseum/distributed.py`:
  - `run_distributed_learner`;
  - `_dist_worker_target`;
  - `run_distributed_workers`.
- Modify: `src/colosseum/envs/subproc_vec_env.py` (`_worker_loop`)
- Modify: `configs/examples/*.yaml` (delete `checkpoint.dir`)
- Modify: `tests/helpers.py` (add `make_test_run_dir`)
- Create: `tests/integration/__init__.py` if T0.2 did not create it, and `tests/integration/cli_runner.py`
- Test: `tests/unit/test_run_dir_logging.py` and `tests/integration/test_run_dir_outputs.py` (new)

**Interfaces:**
- Consumes:
  - `load_config`, `ColosseumConfig.run` (T6.1)
  - `ConfigError`
  - `CheckpointManager(base_dir, pool_size)` (T5.3)
  - `Coordinator(config, checkpoint_dir=...)` (T5.1)
- Produces:
  - `colosseum.core.run_dir.RunDir`, a frozen dataclass with:
    - field `root: Path`;
    - properties `logs`, `checkpoints`, `metrics_path`, `ratings_path`, `resolved_config_path`;
    - `RunDir.create(config, config_path=None, role: str | None = None) -> RunDir`;
    - `RunDir.open(root) -> RunDir`;
    - `write_resolved_config(config) -> Path`.
  - `colosseum.core.run_dir.RESOLVED_CONFIG_FILE = "config.resolved.yaml"`
  - `colosseum.utils.logging.setup_process_logging(log_dir: str | Path | None, process_name: str, console_level: int = logging.WARNING) -> None`. When `log_dir` is set, it logs `"<process_name> started (pid N)"` at INFO.
  - `colosseum.utils.logging.ENV_LOG_DIR = "COLOSSEUM_LOG_DIR"` and `ENV_PROCESS_NAME = "COLOSSEUM_PROCESS_NAME"`
  - `colosseum.utils.process.run_child(name: str, log_dir: str | None, fn, *args, **kwargs) -> None`. It sets up logging, exports the two env vars for grandchildren, logs `"<name> crashed"` with the traceback or `"<name> finished"`, and re-raises.
  - `Launcher.__init__(self, config: ColosseumConfig, run_dir: RunDir)`
  - `_learner_target(agent_id, *args, log_dir=None, **kwargs)` and `_worker_target(worker_id, *args, log_dir=None, **kwargs)`, thin wrappers over the renamed bodies `_learner_main` and `_worker_main`
  - `run_training(config_path, overrides=None) -> RunDir`. It prints `Run directory: <path>` to stdout.
  - `Coordinator.__init__(self, config, checkpoint_dir: str | Path)`, now required
  - `tests/helpers.make_test_run_dir(config, tmp_path, name="test-run") -> RunDir`
  - `tests/integration/cli_runner.py` with:
    - `REPO_ROOT`, `TTT_CONFIG`, `TTT_MULTI_CONFIG`, `TINY`;
    - `child_env()`, `train_cmd(...)`, `run_train(config, tmp_path, name="run", overrides=None, timeout=240.0) -> TrainRun`, `start_train(...)`, `wait_for(...)`;
    - `TrainRun` with `.returncode`, `.stdout`, `.stderr`, `.root`, `.records(kind=None)`, `.log(process_name)` and `.ratings()`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_run_dir_logging.py`:

```python
"""RunDir layout and per-process logging (T6.2)."""
from __future__ import annotations

import logging
import re

import pytest
import yaml

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RunDir
from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging
from colosseum.utils.process import run_child

TTT = "examples.tic_tac_toe"


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


def make_config(tmp_path, name=None) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv"},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "agents": {"alpha": {"algorithm": {"learning_rate": 1e-4}}, "beta": None},
        "run": {"dir": str(tmp_path / "runs"), "name": name},
    })


def test_default_name_uses_config_stem_and_timestamp(tmp_path):
    cfg = make_config(tmp_path)
    run = RunDir.create(cfg, config_path="configs/examples/tic_tac_toe.yaml")
    assert re.fullmatch(r"tic_tac_toe-\d{8}-\d{6}", run.root.name)
    assert run.root.parent == tmp_path / "runs"
    assert run.logs.is_dir() and run.checkpoints.is_dir()
    assert run.metrics_path == run.root / "metrics.jsonl"
    assert run.ratings_path == run.root / "ratings.json"
    again = RunDir.create(cfg, config_path="configs/examples/tic_tac_toe.yaml")
    assert again.root != run.root  # auto names never collide


def test_explicit_name_refuses_existing_run(tmp_path):
    cfg = make_config(tmp_path, name="exp1")
    run = RunDir.create(cfg)
    run.write_resolved_config(cfg)
    with pytest.raises(ConfigError, match="exp1"):
        RunDir.create(cfg)
    assert RunDir.create(cfg, role="learner-alpha").root.name == "exp1-learner-alpha"


def test_resolved_config_roundtrips(tmp_path):
    cfg = make_config(tmp_path, name="rt")
    run = RunDir.create(cfg)
    path = run.write_resolved_config(cfg)
    assert path == run.root / "config.resolved.yaml"
    assert load_config(path) == cfg
    assert yaml.safe_load(path.read_text())["agents"]["alpha"]["algorithm"] == {"learning_rate": 1e-4}


def test_setup_process_logging_file_info_console_warning(tmp_path, capsys, restore_root_logging):
    setup_process_logging(tmp_path / "logs", "worker-3")
    log = logging.getLogger("colosseum.worker.test")
    log.info("info-line")
    log.warning("warn-line")
    text = (tmp_path / "logs" / "worker-3.log").read_text()
    assert "worker-3 started (pid" in text and "info-line" in text and "warn-line" in text
    err = capsys.readouterr().err
    assert "warn-line" in err and "info-line" not in err


def test_setup_process_logging_main_console_info(tmp_path, capsys, restore_root_logging):
    setup_process_logging(None, "main", console_level=logging.INFO)
    logging.getLogger("colosseum.launcher").info("progress-line")
    assert "progress-line" in capsys.readouterr().err


def test_run_child_logs_finish_and_crash(tmp_path, monkeypatch, restore_root_logging):
    monkeypatch.delenv(ENV_LOG_DIR, raising=False)
    monkeypatch.delenv(ENV_PROCESS_NAME, raising=False)
    logs = tmp_path / "logs"
    run_child("learner-a", str(logs), lambda: logging.getLogger("colosseum.learner").info("trained"))
    text = (logs / "learner-a.log").read_text()
    assert "trained" in text and "learner-a finished" in text

    def boom():
        raise RuntimeError("bad env")

    with pytest.raises(RuntimeError):
        run_child("worker-1", str(logs), boom)
    text = (logs / "worker-1.log").read_text()
    assert "worker-1 crashed" in text and "RuntimeError: bad env" in text
    import os
    assert os.environ[ENV_LOG_DIR] == str(logs) and os.environ[ENV_PROCESS_NAME] == "worker-1"
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_run_dir_logging.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'colosseum.core.run_dir'`.

- [ ] **Step 3: Create `RunDir`**

Create `src/colosseum/core/run_dir.py`:

```python
"""Run directory: where one training run writes its outputs.

    runs/<name>/
      config.resolved.yaml     after --set overrides and agent merging
      logs/<process>.log       main, learner-<agent>, worker-<i>, worker-<i>-env<k>
      metrics.jsonl            one JSON record per line (T6.3)
      ratings.json             latest ratings snapshot (T6.3)
      checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError

RESOLVED_CONFIG_FILE = "config.resolved.yaml"


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def ratings_path(self) -> Path:
        return self.root / "ratings.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.root / RESOLVED_CONFIG_FILE

    @classmethod
    def open(cls, root: str | Path) -> "RunDir":
        """An existing run dir (no files are created)."""
        return cls(Path(root))

    @classmethod
    def create(cls, config: ColosseumConfig, config_path: str | Path | None = None,
               role: Optional[str] = None) -> "RunDir":
        """Create ``<run.dir>/<name>[-<role>]`` with ``logs/`` and ``checkpoints/``.

        The default name is ``<config_stem>-<YYYYmmdd-HHMMSS>`` and gets a ``-2``, ``-3``, ...
        suffix if taken. An explicit ``run.name`` that already holds files is an error,
        so a run never mixes into another run's checkpoints (R4-06, R5-13).
        """
        stem = Path(config_path).stem if config_path is not None else "run"
        explicit = config.run.name is not None
        base = config.run.name if explicit else f"{stem}-{datetime.now():%Y%m%d-%H%M%S}"
        if role:
            base = f"{base}-{role}"
        parent = Path(config.run.dir)
        root = parent / base
        if explicit:
            if root.exists() and any(root.iterdir()):
                raise ConfigError(
                    f"Run directory {root} already exists and is not empty; choose another run.name "
                    f"(--set run.name=...) or delete it"
                )
        else:
            suffix = 2
            while root.exists():
                root = parent / f"{base}-{suffix}"
                suffix += 1
        run = cls(root)
        run.logs.mkdir(parents=True, exist_ok=True)
        run.checkpoints.mkdir(parents=True, exist_ok=True)
        return run

    def write_resolved_config(self, config: ColosseumConfig) -> Path:
        path = self.resolved_config_path
        path.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
        return path
```

- [ ] **Step 4: Create the logging and process helpers**

Create `src/colosseum/utils/logging.py`:

```python
"""Per-process logging: every process writes its own file under ``<run>/logs/``."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

ENV_LOG_DIR = "COLOSSEUM_LOG_DIR"            # inherited by grandchildren (subprocess env workers)
ENV_PROCESS_NAME = "COLOSSEUM_PROCESS_NAME"


def setup_process_logging(log_dir: str | Path | None, process_name: str,
                          console_level: int = logging.WARNING) -> None:
    """Configure the root logger of this process. Call it first in every process target.

    - ``<log_dir>/<process_name>.log`` receives INFO and above (when ``log_dir`` is set);
    - stderr receives ``console_level`` and above: WARNING for children, INFO for main.

    Calling it again replaces the previous handlers.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [%(levelname)s] {process_name} %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / f"{process_name}.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info(f"{process_name} started (pid {os.getpid()})")
    logging.captureWarnings(True)


def inherited_log_dir() -> Optional[str]:
    """Log dir exported by the parent process (see ``utils.process.run_child``)."""
    return os.environ.get(ENV_LOG_DIR)
```

Create `src/colosseum/utils/process.py`:

```python
"""Child-process helpers shared by the launcher and the distributed roles."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging

logger = logging.getLogger(__name__)


def run_child(name: str, log_dir: Optional[str], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Body of every child-process target.

    Sets up logging first, exports the log dir and process name for grandchildren,
    and records a crash traceback in the process's own log before re-raising.
    """
    setup_process_logging(log_dir, name)
    if log_dir is not None:
        os.environ[ENV_LOG_DIR] = str(log_dir)
    os.environ[ENV_PROCESS_NAME] = name
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception(f"{name} crashed")
        raise
    logger.info(f"{name} finished")
```

- [ ] **Step 5: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/unit/test_run_dir_logging.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 6: Remove `checkpoint.dir`; checkpoint dir becomes required**

1. In `src/colosseum/core/config.py`, delete the `dir` field from `CheckpointConfig`. Keep `save_optimizer`.
2. In every `configs/examples/*.yaml`, delete the line `  dir: "./checkpoints"` under `checkpoint:`.
3. In `src/colosseum/coordinator/coordinator.py`, change the signature to `def __init__(self, config: ColosseumConfig, checkpoint_dir: str | Path) -> None:`, and change the manager construction argument to `base_dir=checkpoint_dir`.

- [ ] **Step 7: Launcher: run dir, named processes, logging in targets**

In `src/colosseum/launcher.py`:

1. Add the imports:

```python
from colosseum.core.run_dir import RunDir
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import run_child
```

2. Rename the function `_learner_target` to `_learner_main`, and `_worker_target` to `_worker_main`; their bodies stay unchanged. Add the new thin targets directly above them:

```python
def _learner_target(agent_id: str, *args, log_dir: Optional[str] = None, **kwargs) -> None:
    """Learner process entry point: process logging first, then the learner body."""
    run_child(f"learner-{agent_id}", log_dir, _learner_main, agent_id, *args, **kwargs)


def _worker_target(worker_id: int, *args, log_dir: Optional[str] = None, **kwargs) -> None:
    """Worker process entry point: process logging first, then the worker body."""
    run_child(f"worker-{worker_id}", log_dir, _worker_main, worker_id, *args, **kwargs)
```

3. Replace `Launcher.__init__` with:

```python
    def __init__(self, config: ColosseumConfig, run_dir: RunDir) -> None:
        self._config = config
        self._run_dir = run_dir
        self._processes: list[mp.Process] = []
        self._stop_event = mp.Event()
        self._all_queues: list[mp.Queue] = []
```

   Keep any other attribute that Parts A–C added to `__init__`.

4. In `Launcher.launch`:
   - replace `coordinator = Coordinator(cfg)` with `coordinator = Coordinator(cfg, checkpoint_dir=self._run_dir.checkpoints)`;
   - in the learner `mp.Process(...)` call, add `name=f"learner-{aid}",` and `kwargs={"log_dir": str(self._run_dir.logs)},`;
   - in the worker `mp.Process(...)` call, add `name=f"worker-{worker_id}",` and `kwargs={"log_dir": str(self._run_dir.logs)},`.

   The positional `args=(...)` tuples stay as they are.

5. Replace `run_training` with:

```python
def run_training(config_path: str, overrides: dict | None = None) -> RunDir:
    """Entry point: load config, create the run dir, log to it, launch training."""
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    apply_global_seed(config.training.seed)  # after overrides (R3-27)
    run_dir = RunDir.create(config, config_path)
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    Launcher(config, run_dir).launch()
    return run_dir
```

- [ ] **Step 8: Distributed roles get run dirs and process logging**

In `src/colosseum/distributed.py`:

1. Add the imports:

```python
from colosseum.core.run_dir import RunDir
from colosseum.utils.logging import setup_process_logging
from colosseum.utils.process import run_child
```

2. In `run_distributed_learner`, replace the `logging.basicConfig(...)` call and the `config = load_config(...)` line with the block below. The seeding line from T6.1 stays right after it.

```python
    setup_process_logging(None, f"learner-{agent_id}", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    run_dir = RunDir.create(config, config_path, role=f"learner-{agent_id}")
    setup_process_logging(run_dir.logs, f"learner-{agent_id}", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
```

   Then replace `base_dir=config.checkpoint.dir,` with `base_dir=run_dir.checkpoints,`.

3. Rename `_dist_worker_target` to `_dist_worker_main` and add above it:

```python
def _dist_worker_target(worker_id: int, *args, log_dir: Optional[str] = None, **kwargs) -> None:
    """Distributed worker process entry point: process logging first."""
    run_child(f"worker-{worker_id}", log_dir, _dist_worker_main, worker_id, *args, **kwargs)
```

4. In `run_distributed_workers`, replace the `logging.basicConfig(...)` call and the `config = load_config(...)` line with:

```python
    setup_process_logging(None, "workers-main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    run_dir = RunDir.create(config, config_path, role="workers")
    setup_process_logging(run_dir.logs, "workers-main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
```

   In its `mp.Process(...)` call, add `name=f"worker-{worker_id}",` and `kwargs={"log_dir": str(run_dir.logs)},`.

- [ ] **Step 9: Subprocess env children log too**

In `src/colosseum/envs/subproc_vec_env.py`, make these the first statements of `_worker_loop`, before `vec_env = VectorEnv(...)`:

```python
    from colosseum.utils.logging import ENV_PROCESS_NAME, inherited_log_dir, setup_process_logging

    setup_process_logging(inherited_log_dir(), f"{os.environ.get(ENV_PROCESS_NAME, 'envproc')}-env{global_offset}")
```

Keep any thread-limit lines that T2.1 added; they may come before or after these two lines.

- [ ] **Step 10: Shared test helpers and the CLI runner**

Append to `tests/helpers.py`:

```python
def make_test_run_dir(config, tmp_path, name: str = "test-run"):
    """Point ``config.run`` at ``tmp_path`` and create the run dir (tests never write to cwd)."""
    from colosseum.core.run_dir import RunDir

    config.run.dir = str(tmp_path / "runs")
    config.run.name = name
    return RunDir.create(config)
```

Create `tests/integration/cli_runner.py` (and an empty `tests/integration/__init__.py` if it does not exist):

```python
"""Run ``colosseum train`` in a subprocess for integration tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
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
}


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"WANDB_MODE": "disabled", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    return env


def train_cmd(config: Path, run_parent: Path, name: str, overrides: Optional[dict[str, str]] = None) -> list[str]:
    sets = {**TINY, **(overrides or {}), "run.dir": str(run_parent), "run.name": name}
    cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(config)]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


@dataclass
class TrainRun:
    returncode: int
    stdout: str
    stderr: str
    root: Path

    def records(self, kind: Optional[str] = None) -> list[dict]:
        path = self.root / "metrics.jsonl"
        if not path.exists():
            return []
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in records if kind is None or r["kind"] == kind]

    def log(self, process_name: str) -> str:
        return (self.root / "logs" / f"{process_name}.log").read_text()

    def ratings(self) -> dict:
        return json.loads((self.root / "ratings.json").read_text())


def run_train(config: Path, tmp_path: Path, name: str = "run", overrides: Optional[dict[str, str]] = None,
              timeout: float = 240.0) -> TrainRun:
    run_parent = tmp_path / "runs"
    proc = subprocess.run(train_cmd(config, run_parent, name, overrides), cwd=REPO_ROOT, env=child_env(),
                          capture_output=True, text=True, timeout=timeout)
    return TrainRun(proc.returncode, proc.stdout, proc.stderr, run_parent / name)


def start_train(config: Path, tmp_path: Path, name: str = "run",
                overrides: Optional[dict[str, str]] = None) -> tuple[subprocess.Popen, Path]:
    """Start training in its own session; stdout/stderr go to files next to the run dir."""
    run_parent = tmp_path / "runs"
    out = open(tmp_path / f"{name}.stdout", "w")
    err = open(tmp_path / f"{name}.stderr", "w")
    proc = subprocess.Popen(train_cmd(config, run_parent, name, overrides), cwd=REPO_ROOT, env=child_env(),
                            stdout=out, stderr=err, text=True, start_new_session=True)
    return proc, run_parent / name


def wait_for(predicate: Callable[[], bool], timeout: float, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
```

- [ ] **Step 11: Write the integration test**

Create `tests/integration/test_run_dir_outputs.py`:

```python
"""A short `colosseum train` writes the run dir, the resolved config and per-process logs (T6.2)."""
from __future__ import annotations

from colosseum.core.config import load_config
from tests.integration.cli_runner import TTT_CONFIG, run_train


def test_run_dir_has_resolved_config_and_process_logs(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="logs-run")
    assert run.returncode == 0, run.stderr[-3000:]
    assert f"Run directory: {run.root}" in run.stdout
    cfg = load_config(run.root / "config.resolved.yaml")
    assert cfg.run.name == "logs-run" and cfg.training.total_timesteps == 3000
    assert (run.root / "checkpoints").is_dir()
    assert "main started (pid" in run.log("main")
    learner = run.log("learner-agent_0")
    assert "learner-agent_0 started (pid" in learner and "learner-agent_0 finished" in learner
    worker = run.log("worker-0")
    assert "worker-0 started (pid" in worker and "worker-0 finished" in worker
```

- [ ] **Step 12: Run the integration test**

Run: `.venv/bin/python -m pytest tests/integration/test_run_dir_outputs.py -v`

Expected: PASS in under 60 s.

If it fails because `import tests.integration...` cannot be resolved, check that `tests/__init__.py` and `tests/integration/__init__.py` exist (T0.2 layout). Create empty ones if they are missing.

- [ ] **Step 13: Update old tests**

Run:

```bash
grep -rnE "checkpoint\"\]\[\"dir\"\]|CheckpointConfig\(dir|checkpoint\.dir|Launcher\(|Coordinator\(" tests/ --include=*.py
```

Apply this rule to each hit:
- **`cd["checkpoint"]["dir"] = X` / `CheckpointConfig(dir=X)`:** delete it. Then route the run to `tmp_path`: `cd["run"] = {"dir": str(tmp_path / "runs"), "name": "<unique-name>"}`, or `run=RunConfig(dir=..., name=...)`.
- **`Launcher(config)`:** change to `Launcher(config, make_test_run_dir(config, tmp_path))` and import `make_test_run_dir` from `tests.helpers`. Where a test still uses `tempfile.TemporaryDirectory()`, pass `Path(tmpdir)` instead of `tmp_path`.
- **`Coordinator(config)` without `checkpoint_dir`:** add `checkpoint_dir=tmp_path / "ckpt"`. If the test has no `tmp_path`, add the fixture to its signature.

- [ ] **Step 14: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS. Afterwards `git status --porcelain` shows no `runs/` or `checkpoints/` in the repo root.

- [ ] **Step 15: Commit**

```bash
git add src/colosseum/core/run_dir.py src/colosseum/utils/logging.py src/colosseum/utils/process.py src/colosseum/core/config.py src/colosseum/coordinator/coordinator.py src/colosseum/launcher.py src/colosseum/distributed.py src/colosseum/envs/subproc_vec_env.py configs/examples tests/
git commit -m "feat: run directory with resolved config and per-process log files"
```

---
### Task T6.3: `metrics.jsonl`, episode and system aggregation, console progress, `ratings.json`

Fixes R5-03, R5-06, R6-08, ET-12, and R4-07 (persisted ratings).

The main process is the only writer. It reads `metrics_queue` (learner train metrics and worker stats) and `results_queue` (per-seat match results).

**Files:**
- Create:
  - `src/colosseum/metrics/jsonl.py`
  - `src/colosseum/metrics/aggregator.py`
  - `src/colosseum/metrics/console.py`
  - `src/colosseum/metrics/hub.py`
- Modify: `src/colosseum/core/config.py` (`MetricsConfig.console_interval_sec`)
- Modify: `src/colosseum/worker/rollout_worker.py`:
  - add `report_worker_stats`;
  - `rollout_worker_process` gets `stats_queue` and `stats_interval_sec`.
- Modify: `src/colosseum/launcher.py`:
  - `_worker_main` passes `stats_queue`;
  - `Launcher.launch` creates the hub;
  - `_monitor_loop` (the results block and the metrics block);
  - add `_queue_depths`.
- Modify: `tests/integration/cli_runner.py` (add `metrics.console_interval_sec` to `TINY`)
- Test: `tests/unit/test_metrics_jsonl.py` and `tests/integration/test_metrics_outputs.py` (new)

**Interfaces:**
- Consumes:
  - `MatchResult` / `SeatResult` (T3.4)
  - `Coordinator.ratings_snapshot()` (T5.2)
  - `RunDir.metrics_path` and `RunDir.ratings_path` (T6.2)
  - `RolloutLoop.stats -> dict[str, int]` (contract; includes `parked_buffers`)
  - `Launcher._env_counter` (T5.3)
  - `_drain_queue` (T5.3)
  - Learner metric dicts on `metrics_queue` carry `agent_id` and `train_step`, as today.
- Produces:
  - `colosseum.metrics.jsonl`:
    - `METRIC_KINDS = ("train", "episodes", "ratings", "system")`
    - `REQUIRED_KEYS: dict[str, set[str]]`
    - `MetricsWriter(path)` with `.write(kind, **fields)`, `.close()` and `.path`
    - `write_json_atomic(path, data)`
  - `colosseum.metrics.aggregator`:
    - `OPPONENT_TYPES = ("latest", "past", "arena")`
    - `opponent_type(result, seat) -> str | None`
    - `EpisodeAggregator()` with `.add(result)` and `.flush() -> dict[str, dict]`
    - `SystemStats(clock=time.monotonic)` with `.on_train_step(agent_id, train_step)`, `.on_worker_stats(stats)` and `.snapshot(env_steps, queue_depths) -> dict`
  - `colosseum.metrics.console.ConsoleReporter(total_timesteps, log=None)` with `.format_line(agent_id, *, train_step, env_steps, fps, loss, entropy, return_mean, wr_vs_past, wr_arena) -> str` and `.emit(lines)`
  - `colosseum.metrics.hub`:
    - `MetricsHub(*, writer, ratings_path, agent_ids, total_timesteps, log_interval, console_interval_sec, wandb_logger=None, clock=time.monotonic)` with `.on_train_metrics(metrics)`, `.on_worker_stats(stats)`, `.on_match_result(result)`, `.maybe_tick(*, env_steps, ratings, queue_depths, force=False) -> bool` and `.close(*, env_steps, ratings, queue_depths)`
    - `flatten(prefix, value) -> dict[str, float]`
    - The hub calls `wandb_logger.log_train(agent_id, metrics, train_step)` and `wandb_logger.log_global(metrics, env_steps)` when a logger is given (T6.4).
  - `MetricsConfig.console_interval_sec: float = 10.0`
  - `colosseum.worker.rollout_worker.report_worker_stats(q, worker_id: int, stats: dict) -> None`, which puts `{"kind": "worker_stats", "worker_id", **stats}`
  - `rollout_worker_process(..., stats_queue=None, stats_interval_sec: float = 2.0)`
  - `launcher._queue_depths(queues: dict[str, Any]) -> dict[str, int]`
  - `Launcher._hub` and `Launcher._trajectory_queues`

**Record schema** (`metrics.jsonl`, one JSON object per line; `ts` is Unix time):

| kind | required keys | notes |
|---|---|---|
| `train` | `ts`, `kind`, `agent`, `train_step` | plus every numeric APPO metric; written when `train_step` advanced by ≥ `metrics.log_interval` since the last written record (the first step is always written) |
| `episodes` | `ts`, `kind`, `agent`, `env_steps`, `episodes`, `return_mean`, `length_mean`, `wdl`, `seat_counts` | `wdl = {"latest": [W, D, L], "past": [...], "arena": [...]}` for the agent's latest-weights seats since the previous record; `seat_counts[i]` = number of those seats at seat `i` |
| `ratings` | `ts`, `kind`, `env_steps`, `elo`, `win_rates`, `games`, `wr_vs_past` | the content of `Coordinator.ratings_snapshot()` |
| `system` | `ts`, `kind`, `env_steps`, `env_steps_per_sec`, `train_steps_per_sec`, `queue_depths`, `parked_buffers`, `workers_reporting` | rates are over the interval since the previous system record |

W/D/L per episode for a latest-weights seat: W if its outcome is above every other seat's outcome, D if it ties the best other outcome, L otherwise.

The opponent type is:
- `arena` if another base agent is in the match;
- `past` if a checkpoint of the same agent is;
- `latest` otherwise.

Single-player matches have no opponent type; they count toward returns and lengths only.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_metrics_jsonl.py`:

```python
"""metrics.jsonl schema, episode/system aggregation, console line, hub cadence (T6.3)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from colosseum.core.types import MatchResult, SeatResult
from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats, opponent_type
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.hub import MetricsHub, flatten
from colosseum.metrics.jsonl import METRIC_KINDS, REQUIRED_KEYS, MetricsWriter, write_json_atomic


def seat(i, agent, outcome, network="latest", reward=0.0):
    return SeatResult(seat=i, agent_id=agent, network_id=network, outcome=outcome, reward=reward)


def result(*seats, length=5):
    return MatchResult(match_id="m", seats=list(seats), episode_length=length)


RATINGS = {
    "elo": {"a": 1216.0, "b": 1184.0},
    "win_rates": {"a": {"b": 1.0}, "b": {"a": 0.0}},
    "games": {"a": {"b": 1}, "b": {"a": 1}},
    "wr_vs_past": {"a": None, "b": 0.75},
    "past_games": {"a": 0, "b": 4},
}


def test_writer_one_json_line_per_record_with_numpy_and_nan(tmp_path):
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    writer.write("train", agent="a", train_step=np.int64(3), loss=np.float32(0.5), ev=float("nan"))
    with pytest.raises(ValueError):
        writer.write("bogus")
    writer.close()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "train" and record["train_step"] == 3 and record["ev"] is None
    assert isinstance(record["ts"], float)


def test_write_json_atomic(tmp_path):
    write_json_atomic(tmp_path / "ratings.json", {"elo": {"a": 1200.0}})
    assert json.loads((tmp_path / "ratings.json").read_text()) == {"elo": {"a": 1200.0}}
    assert not list(tmp_path.glob(".*.tmp"))


def test_opponent_type():
    r = result(seat(0, "a", 1.0), seat(1, "a", 0.0, network="ckpt_v5"))
    assert opponent_type(r, r.seats[0]) == "past"
    r = result(seat(0, "a", 1.0), seat(1, "b", 0.0))
    assert opponent_type(r, r.seats[0]) == "arena"
    r = result(seat(0, "a", 1.0), seat(1, "a", 0.0))
    assert opponent_type(r, r.seats[1]) == "latest"
    r = result(seat(0, "a", 1.0))
    assert opponent_type(r, r.seats[0]) is None


def test_episode_aggregator_splits_wdl_by_opponent_type():
    agg = EpisodeAggregator()
    agg.add(result(seat(0, "a", 1.0, reward=1.0), seat(1, "a", 0.0, network="ckpt_v5", reward=-1.0), length=7))
    agg.add(result(seat(0, "a", 0.5), seat(1, "a", 0.5), length=9))
    agg.add(result(seat(0, "b", 0.0, reward=-1.0), seat(1, "a", 1.0, reward=1.0), length=5))
    agg.add(result(seat(0, "s", 0.0, reward=3.0), length=10))
    out = agg.flush()
    assert out["a"]["wdl"] == {"latest": [0, 2, 0], "past": [1, 0, 0], "arena": [1, 0, 0]}
    assert out["a"]["episodes"] == 4 and out["a"]["seat_counts"] == [2, 2]
    assert out["a"]["length_mean"] == pytest.approx((7 + 9 + 9 + 5) / 4)
    assert out["b"]["wdl"]["arena"] == [0, 0, 1]
    assert out["s"]["return_mean"] == 3.0 and out["s"]["wdl"]["latest"] == [0, 0, 0]
    assert agg.flush() == {}


def test_system_stats_rates():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_train_step("a", 10)
    stats.on_worker_stats({"worker_id": 0, "parked_buffers": 2})
    stats.on_worker_stats({"worker_id": 1, "parked_buffers": 1})
    clock[0] = 2.0
    first = stats.snapshot(1000, {"a": 3})
    assert first["env_steps_per_sec"] == 500.0 and first["train_steps_per_sec"] == {"a": 5.0}
    assert first["parked_buffers"] == 3 and first["workers_reporting"] == 2
    stats.on_train_step("a", 14)
    clock[0] = 4.0
    second = stats.snapshot(3000, {"a": 0})
    assert second["env_steps_per_sec"] == 1000.0 and second["train_steps_per_sec"] == {"a": 2.0}
    assert stats.snapshot(3000, {})["env_steps_per_sec"] == 0.0  # zero interval -> zero rate


def test_console_line():
    line = ConsoleReporter(10_000).format_line(
        "a", train_step=5, env_steps=2500, fps=1234.5, loss=0.123, entropy=None,
        return_mean=0.5, wr_vs_past=0.61, wr_arena=None)
    assert line.startswith("[a] step 5 |")
    assert "25.0% budget" in line and "loss 0.1230" in line and "entropy -" in line
    assert "wr_vs_past 0.61" in line and "wr_arena" not in line


def test_flatten():
    assert flatten("r", {"elo": {"a": 1.0}, "x": None, "l": [1, 2], "ok": True}) == {
        "r/elo/a": 1.0, "r/l/0": 1.0, "r/l/1": 2.0}


def test_hub_cadence_schema_and_ratings_file(tmp_path, caplog):
    clock = [0.0]
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["a", "b"], total_timesteps=1000, log_interval=2, console_interval_sec=10.0,
                     clock=lambda: clock[0])
    for step in range(1, 6):
        hub.on_train_metrics({"agent_id": "a", "train_step": step, "total_loss": 0.1 * step,
                              "entropy": 2.0, "note": "ignored"})
    hub.on_match_result(result(seat(0, "a", 1.0, reward=1.0), seat(1, "b", 0.0, reward=-1.0), length=7))
    hub.on_worker_stats({"kind": "worker_stats", "worker_id": 0, "env_steps": 100, "parked_buffers": 1})
    assert not hub.maybe_tick(env_steps=100, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    clock[0] = 11.0
    with caplog.at_level(logging.INFO):
        assert hub.maybe_tick(env_steps=300, ratings=RATINGS, queue_depths={"a": 1, "b": 0})
    assert "[a] step 5 |" in caplog.text and "[b] step 0 |" in caplog.text
    clock[0] = 12.0
    hub.close(env_steps=400, ratings=RATINGS, queue_depths={"a": 0, "b": 0})

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_step"] for r in records if r["kind"] == "train"] == [1, 3, 5]
    assert {r["kind"] for r in records} == set(METRIC_KINDS)
    for record in records:
        missing = REQUIRED_KEYS[record["kind"]] - set(record)
        assert not missing, (record["kind"], missing)
    assert "note" not in [r for r in records if r["kind"] == "train"][0]
    ratings = json.loads((tmp_path / "ratings.json").read_text())
    assert ratings["env_steps"] == 400 and ratings["wr_vs_past"] == {"a": None, "b": 0.75}


def test_hub_forwards_to_wandb_logger(tmp_path):
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
    hub.maybe_tick(env_steps=5, ratings={"elo": {"a": 1200.0}}, queue_depths={"a": 0})
    assert calls[0] == ("train", "a", 1, {"total_loss": 0.5})
    assert calls[1][0] == "global" and calls[1][1] == 5 and calls[1][2]["ratings/elo/a"] == 1200.0
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_metrics_jsonl.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'colosseum.metrics.aggregator'`.

- [ ] **Step 3: Create `metrics/jsonl.py`**

```python
"""metrics.jsonl: the source of truth for training metrics (WandB is only a viewer)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

METRIC_KINDS = ("train", "episodes", "ratings", "system")

REQUIRED_KEYS: dict[str, set[str]] = {
    "train": {"ts", "kind", "agent", "train_step"},
    "episodes": {"ts", "kind", "agent", "env_steps", "episodes", "return_mean", "length_mean", "wdl",
                 "seat_counts"},
    "ratings": {"ts", "kind", "env_steps", "elo", "win_rates", "games", "wr_vs_past"},
    "system": {"ts", "kind", "env_steps", "env_steps_per_sec", "train_steps_per_sec", "queue_depths",
               "parked_buffers", "workers_reporting"},
}


def _sanitize(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays to python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


class MetricsWriter:
    """Appends one JSON object per line: ``{"ts", "kind", **fields}``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, kind: str, **fields: Any) -> None:
        if kind not in METRIC_KINDS:
            raise ValueError(f"Unknown metrics kind {kind!r}; expected one of {METRIC_KINDS}")
        record = _sanitize({"ts": time.time(), "kind": kind, **fields})
        self._fh.write(json.dumps(record, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON to ``path`` via a temp file and ``os.replace`` (readers never see half a file)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_sanitize(data), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
```

- [ ] **Step 4: Create `metrics/aggregator.py`**

```python
"""Per-interval aggregation of match results and system throughput."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np

LATEST_NETWORK_ID = "latest"
OPPONENT_TYPES = ("latest", "past", "arena")


def opponent_type(result, seat) -> Optional[str]:
    """'arena' (another agent present), 'past' (own checkpoint present), 'latest', or None (solo)."""
    others = [s for s in result.seats if s is not seat]
    if not others:
        return None
    if any(s.agent_id != seat.agent_id for s in others):
        return "arena"
    if any(s.network_id != LATEST_NETWORK_ID for s in others):
        return "past"
    return "latest"


class EpisodeAggregator:
    """Per agent, over the seats that play the agent's latest weights:
    mean return, mean length, W/D/L by opponent type, seat counts."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._lengths: dict[str, list[int]] = defaultdict(list)
        self._wdl: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {t: [0, 0, 0] for t in OPPONENT_TYPES})
        self._seats: dict[str, list[int]] = defaultdict(list)

    def add(self, result) -> None:
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
            kind = opponent_type(result, seat)
            if kind is None:
                continue
            best_other = max(s.outcome for s in result.seats if s is not seat)
            idx = 0 if seat.outcome > best_other else (1 if seat.outcome == best_other else 2)
            self._wdl[agent_id][kind][idx] += 1

    def flush(self) -> dict[str, dict[str, Any]]:
        """Stats since the previous flush, per agent; resets the accumulators."""
        out: dict[str, dict[str, Any]] = {}
        for agent_id, returns in self._returns.items():
            out[agent_id] = {
                "episodes": len(returns),
                "return_mean": float(np.mean(returns)),
                "length_mean": float(np.mean(self._lengths[agent_id])),
                "wdl": {t: list(v) for t, v in self._wdl[agent_id].items()},
                "seat_counts": list(self._seats[agent_id]),
            }
        self._reset()
        return out


class SystemStats:
    """Throughput and queue health between two snapshots."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_t = clock()
        self._last_env_steps = 0
        self._train_steps: dict[str, int] = {}
        self._last_train_steps: dict[str, int] = {}
        self._workers: dict[int, dict] = {}

    def on_train_step(self, agent_id: str, train_step: int) -> None:
        self._train_steps[agent_id] = max(int(train_step), self._train_steps.get(agent_id, 0))

    def on_worker_stats(self, stats: dict) -> None:
        self._workers[int(stats["worker_id"])] = dict(stats)

    def snapshot(self, env_steps: int, queue_depths: dict[str, int]) -> dict[str, Any]:
        now = self._clock()
        dt = now - self._last_t

        def rate(delta: float) -> float:
            return delta / dt if dt > 1e-3 else 0.0

        snap = {
            "env_steps": int(env_steps),
            "env_steps_per_sec": rate(env_steps - self._last_env_steps),
            "train_steps_per_sec": {
                a: rate(s - self._last_train_steps.get(a, 0)) for a, s in self._train_steps.items()
            },
            "queue_depths": dict(queue_depths),
            "parked_buffers": int(sum(int(w.get("parked_buffers", 0)) for w in self._workers.values())),
            "workers_reporting": len(self._workers),
        }
        self._last_t = now
        self._last_env_steps = int(env_steps)
        self._last_train_steps = dict(self._train_steps)
        return snap
```

- [ ] **Step 5: Create `metrics/console.py`**

```python
"""One progress line per agent, printed by the main process."""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger("colosseum.progress")


def _fmt(value: Optional[float], spec: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return format(value, spec)


class ConsoleReporter:
    def __init__(self, total_timesteps: int, log: Optional[logging.Logger] = None) -> None:
        self._total = max(1, int(total_timesteps))
        self._log = log or logger

    def format_line(self, agent_id: str, *, train_step: int, env_steps: int, fps: float,
                    loss: Optional[float], entropy: Optional[float], return_mean: Optional[float],
                    wr_vs_past: Optional[float], wr_arena: Optional[float]) -> str:
        parts = [
            f"[{agent_id}] step {train_step}",
            f"{100.0 * min(1.0, env_steps / self._total):5.1f}% budget",
            f"{fps:,.0f} env-steps/s",
            f"loss {_fmt(loss, '.4f')}",
            f"entropy {_fmt(entropy, '.3f')}",
            f"return {_fmt(return_mean, '.3f')}",
        ]
        if wr_vs_past is not None:
            parts.append(f"wr_vs_past {wr_vs_past:.2f}")
        if wr_arena is not None:
            parts.append(f"wr_arena {wr_arena:.2f}")
        return " | ".join(parts)

    def emit(self, lines: list[str]) -> None:
        for line in lines:
            self._log.info(line)
```

- [ ] **Step 6: Create `metrics/hub.py`**

```python
"""MetricsHub: the main process's single sink for train metrics, results and system stats."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from colosseum.metrics.aggregator import EpisodeAggregator, SystemStats
from colosseum.metrics.console import ConsoleReporter
from colosseum.metrics.jsonl import MetricsWriter, write_json_atomic


def _numeric(d: dict) -> dict[str, float]:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def flatten(prefix: str, value: Any) -> dict[str, float]:
    """Nested dicts/lists of numbers -> ``{"prefix/a/b": float}``; non-numbers are dropped."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(f"{prefix}/{k}", v))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten(f"{prefix}/{i}", v))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


class MetricsHub:
    """Writes ``train`` records as they arrive (every ``log_interval`` train steps per agent).

    Every ``console_interval_sec`` it also writes ``episodes``, ``system`` and ``ratings``
    records, rewrites ``ratings.json`` and prints one console line per agent.
    """

    def __init__(self, *, writer: MetricsWriter, ratings_path: str | Path, agent_ids: list[str],
                 total_timesteps: int, log_interval: int, console_interval_sec: float,
                 wandb_logger: Any = None, clock: Callable[[], float] = time.monotonic) -> None:
        self._writer = writer
        self._ratings_path = Path(ratings_path)
        self._agent_ids = list(agent_ids)
        self._log_interval = max(1, int(log_interval))
        self._interval = float(console_interval_sec)
        self._wandb = wandb_logger
        self._clock = clock
        self._episodes = EpisodeAggregator()
        self._system = SystemStats(clock=clock)
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
        self._writer.write("ratings", env_steps=int(env_steps), **ratings)
        write_json_atomic(self._ratings_path, {"env_steps": int(env_steps), **ratings})
        if self._wandb is not None:
            row = flatten("system", {k: v for k, v in system.items() if k != "env_steps"})
            row.update(flatten("ratings", ratings))
            for agent_id, ep in episodes.items():
                row.update(flatten(f"episodes/{agent_id}", {k: v for k, v in ep.items() if k != "wdl"}))
            self._wandb.log_global(row, int(env_steps))
        self._report_console(int(env_steps), system, ratings)
        return True

    def _report_console(self, env_steps: int, system: dict, ratings: dict) -> None:
        wr_vs_past = ratings.get("wr_vs_past", {})
        win_rates = ratings.get("win_rates", {})
        games = ratings.get("games", {})
        lines = []
        for agent_id in self._agent_ids:
            train = self._last_train.get(agent_id, {})
            n_games = sum(games.get(agent_id, {}).values())
            wr_arena = None
            if n_games:
                wr_arena = sum(win_rates[agent_id][b] * g for b, g in games[agent_id].items()) / n_games
            lines.append(self._console.format_line(
                agent_id,
                train_step=int(train.get("train_step", 0)),
                env_steps=env_steps,
                fps=float(system["env_steps_per_sec"]),
                loss=train.get("total_loss"),
                entropy=train.get("entropy"),
                return_mean=self._last_returns.get(agent_id),
                wr_vs_past=wr_vs_past.get(agent_id),
                wr_arena=wr_arena,
            ))
        self._console.emit(lines)

    def close(self, *, env_steps: int, ratings: dict, queue_depths: dict[str, int]) -> None:
        """Final records and ratings.json, then close the file."""
        self.maybe_tick(env_steps=env_steps, ratings=ratings, queue_depths=queue_depths, force=True)
        self._writer.close()
```

- [ ] **Step 7: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/unit/test_metrics_jsonl.py -v`

Expected: all 9 tests PASS.

- [ ] **Step 8: Config field and worker stats**

1. In `src/colosseum/core/config.py`, class `MetricsConfig`, add:

```python
    console_interval_sec: float = Field(
        default=10.0, ge=0.0,
        description="Seconds between console progress lines and episodes/system/ratings records.",
    )
```

2. In `src/colosseum/worker/rollout_worker.py`:

   a. Add this module-level function (import `queue` at the top if it is missing):

```python
def report_worker_stats(q, worker_id: int, stats: dict) -> None:
    """Best-effort worker stats for the main process (system metrics). Never blocks."""
    try:
        q.put_nowait({"kind": "worker_stats", "worker_id": int(worker_id),
                      **{k: int(v) for k, v in stats.items()}})
    except queue.Full:
        pass
```

   b. Add two keyword parameters to `rollout_worker_process`: `stats_queue=None` and `stats_interval_sec: float = 2.0`.

   c. `rollout_worker_process` runs the loop as `loop.run(should_stop=stop_event.is_set, ...)`. This is the T0.5/T2.5 thin wrapper; `RolloutLoop.run` calls `should_stop()` once per iteration. Insert right before that `loop.run(` call:

```python
    last_stats = [time.monotonic()]

    def should_stop() -> bool:
        now = time.monotonic()
        if stats_queue is not None and now - last_stats[0] >= stats_interval_sec:
            last_stats[0] = now
            report_worker_stats(stats_queue, worker_id, loop.stats)
        return stop_event.is_set()
```

   Then change the call to `loop.run(should_stop=should_stop, ...)`, keeping its other arguments. If the wrapper passes a different stop predicate than `stop_event.is_set`, return that predicate's value from `should_stop()` instead. Import `time` if it is missing.

3. In `src/colosseum/launcher.py`:

   a. Give `_worker_main` a keyword parameter `metrics_queue=None`, and pass `stats_queue=metrics_queue` in its `rollout_worker_process(...)` call.

   b. In `Launcher.launch`, add `"metrics_queue": metrics_queue` to the worker process `kwargs={...}` (next to `"log_dir"`).

- [ ] **Step 9: Hub in the launcher**

In `src/colosseum/launcher.py`:

1. Add this module-level helper next to `_drain_queue`:

```python
def _queue_depths(queues: dict[str, Any]) -> dict[str, int]:
    """Approximate items waiting per queue (-1 where the platform cannot tell)."""
    depths = {}
    for name, q in queues.items():
        try:
            depths[name] = int(q.qsize())
        except (NotImplementedError, OSError):
            depths[name] = -1
    return depths
```

   Add `Any` to the `typing` import.

2. In `Launcher.launch`, store `self._trajectory_queues = trajectory_queues` next to the other attributes set in T5.3. Replace the WandB initialization block (`# Initialize WandB` ... `wandb_logger.log_config(...)`) with:

```python
        from colosseum.metrics.hub import MetricsHub
        from colosseum.metrics.jsonl import MetricsWriter

        self._hub = MetricsHub(
            writer=MetricsWriter(self._run_dir.metrics_path),
            ratings_path=self._run_dir.ratings_path,
            agent_ids=trainable_agents,
            total_timesteps=cfg.training.total_timesteps,
            log_interval=cfg.metrics.log_interval,
            console_interval_sec=cfg.metrics.console_interval_sec,
        )
```

   WandB is wired into the hub again in T6.4. Until then, `metrics.use_wandb` has no effect.

3. Remove `wandb_logger` and `log_interval` from the `_monitor_loop` signature and its call site. In the `finally:` that follows the monitor call, replace `wandb_logger.finish()` with:

```python
            self._hub.close(env_steps=int(self._env_counter.value),
                            ratings=coordinator.ratings_snapshot(),
                            queue_depths=_queue_depths(self._trajectory_queues))
```

   Remove the now-unused `WandBLogger` import.

4. In `_monitor_loop`:

   a. In the results block (`# Process episode results from workers`), right after `coordinator.report_match_result(result)`, add `self._hub.on_match_result(result)`.

   b. Replace the whole metrics block (`# Drain metrics queue` and the `while True` loop below it) with:

```python
            # Drain metrics queue: learner train metrics and worker stats
            for item in _drain_queue(metrics_queue):
                if item.get("kind") == "worker_stats":
                    self._hub.on_worker_stats(item)
                else:
                    self._hub.on_train_metrics(item)
            self._hub.maybe_tick(env_steps=int(self._env_counter.value),
                                 ratings=coordinator.ratings_snapshot(),
                                 queue_depths=_queue_depths(self._trajectory_queues))
```

5. In `tests/integration/cli_runner.py`, add `"metrics.console_interval_sec": "1.0",` to `TINY`.

- [ ] **Step 10: Write the integration test**

Create `tests/integration/test_metrics_outputs.py`:

```python
"""A short run writes metrics.jsonl with all four kinds, ratings.json and console progress (T6.3)."""
from __future__ import annotations

from colosseum.metrics.jsonl import REQUIRED_KEYS
from tests.integration.cli_runner import TTT_CONFIG, run_train


def test_metrics_jsonl_ratings_json_and_console(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="metrics-run")
    assert run.returncode == 0, run.stderr[-3000:]
    records = run.records()
    kinds = {r["kind"] for r in records}
    assert kinds == {"train", "episodes", "ratings", "system"}, kinds
    for record in records:
        assert not (REQUIRED_KEYS[record["kind"]] - set(record)), record
    assert max(r["train_step"] for r in run.records("train") if r["agent"] == "agent_0") >= 1
    episodes = run.records("episodes")
    assert sum(r["episodes"] for r in episodes) > 0
    assert set(episodes[0]["wdl"]) == {"latest", "past", "arena"}
    systems = run.records("system")
    assert systems[-1]["env_steps"] >= 3000
    assert any(r["workers_reporting"] >= 1 for r in systems)
    ratings = run.ratings()
    assert set(ratings) >= {"env_steps", "elo", "win_rates", "games", "wr_vs_past"}
    assert "[agent_0] step" in run.stderr  # console progress (main logs INFO to stderr)
```

- [ ] **Step 11: Run the integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_metrics_outputs.py tests/integration/test_run_dir_outputs.py -v`

Expected: PASS.

- [ ] **Step 12: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS. Any remaining test that asserts WandB calls from the launcher is updated in T6.4. If one fails here, delete its launcher-specific assertion; T6.4 adds dedicated WandB tests.

- [ ] **Step 13: Commit**

```bash
git add src/colosseum/metrics/ src/colosseum/core/config.py src/colosseum/worker/rollout_worker.py src/colosseum/launcher.py tests/
git commit -m "feat: metrics.jsonl, episode/system aggregation, ratings.json and console progress"
```

---
### Task T6.4: WandB with per-agent step axes, optional import

Fixes R5-02 and R5-06 (WandB is optional; `metrics.jsonl` is the source of truth).

The run is a single WandB run. Each agent gets its own x-axis via `define_metric(f"{agent}/*", step_metric=f"{agent}/train_step")`. Ratings, system and episode metrics use `env_steps`. No `wandb.log` call passes `step=`, so a lagging agent's points are no longer dropped.

**Files:**
- Replace: `src/colosseum/metrics/wandb_logger.py` (whole file)
- Modify: `src/colosseum/launcher.py` (`Launcher.launch`: create the logger, pass it to the hub, finish it)
- Test: `tests/unit/test_wandb_logger.py` (new)

**Interfaces:**
- Consumes:
  - `MetricsConfig.use_wandb`, `.wandb_project` and `.wandb_entity`
  - `MetricsHub(..., wandb_logger=...)`, which calls `log_train` and `log_global` (T6.3)
- Produces: `colosseum.metrics.wandb_logger.WandBLogger(config: MetricsConfig, run_name: str | None = None, run_config: dict | None = None)` with:
  - `.enabled -> bool`
  - `.log_train(agent_id: str, metrics: dict[str, float], train_step: int) -> None`
  - `.log_global(metrics: dict[str, float], env_steps: int) -> None`
  - `.finish() -> None`
  - Removed: `log_config`, `log_metrics`, `log_train_step`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_wandb_logger.py`:

```python
"""WandB logger: per-agent step axes, no step= argument, optional import (T6.4). No network."""
from __future__ import annotations

import logging
import sys
import types

from colosseum.core.config import MetricsConfig
from colosseum.metrics.wandb_logger import WandBLogger


class FakeWandb(types.ModuleType):
    """Records calls; mirrors the wandb functions the logger uses."""

    def __init__(self, fail_init: bool = False):
        super().__init__("wandb")
        self.fail_init = fail_init
        self.init_kwargs = None
        self.defined: list[tuple[str, object]] = []
        self.logged: list[dict] = []
        self.finished = False

    def init(self, **kwargs):
        if self.fail_init:
            raise RuntimeError("no network")
        self.init_kwargs = kwargs
        return types.SimpleNamespace(finish=self._finish)

    def _finish(self):
        self.finished = True

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, row, **kwargs):
        assert "step" not in kwargs, "wandb.log must not receive step= (R5-02)"
        self.logged.append(dict(row))


def test_disabled_does_not_import_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing would raise
    logger = WandBLogger(MetricsConfig(use_wandb=False))
    assert not logger.enabled
    logger.log_train("a", {"loss": 1.0}, 1)
    logger.log_global({"system/x": 1.0}, 10)
    logger.finish()


def test_missing_wandb_package_disables_with_warning(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with caplog.at_level(logging.WARNING):
        logger = WandBLogger(MetricsConfig(use_wandb=True))
    assert not logger.enabled
    assert "wandb is not installed" in caplog.text


def test_init_failure_disables(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail_init=True))
    with caplog.at_level(logging.WARNING):
        assert not WandBLogger(MetricsConfig(use_wandb=True)).enabled
    assert "no network" in caplog.text


def test_per_agent_step_axes_keep_lagging_agents(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    logger = WandBLogger(MetricsConfig(use_wandb=True, wandb_project="p"), run_name="r1", run_config={"a": 1})
    assert logger.enabled
    assert fake.init_kwargs["name"] == "r1" and fake.init_kwargs["config"] == {"a": 1}
    logger.log_train("alpha", {"loss": 1.0}, 100)
    logger.log_train("beta", {"loss": 2.0}, 95)   # lags behind alpha
    logger.log_train("alpha", {"loss": 0.5}, 110)
    logger.log_global({"ratings/elo/alpha": 1210.0}, env_steps=5000)
    assert ("alpha/*", "alpha/train_step") in fake.defined
    assert ("beta/*", "beta/train_step") in fake.defined
    assert ("ratings/*", "env_steps") in fake.defined and ("system/*", "env_steps") in fake.defined
    assert sum(1 for name, _ in fake.defined if name == "alpha/*") == 1  # defined once
    assert {"beta/loss": 2.0, "beta/train_step": 95} in fake.logged
    assert {"ratings/elo/alpha": 1210.0, "env_steps": 5000} in fake.logged
    logger.finish()
    assert fake.finished
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_wandb_logger.py -v`

Expected: FAIL. `WandBLogger` has no `log_train` (`AttributeError`), and the old `log_train_step` passes `step=`.

- [ ] **Step 3: Replace the WandB logger**

Replace the whole content of `src/colosseum/metrics/wandb_logger.py` with:

```python
"""Optional WandB viewer for the metrics written to metrics.jsonl.

One run. Every agent gets its own x-axis (``<agent>/train_step``) through
``define_metric``, so agents that train at different speeds are all kept (R5-02).
Ratings, system and episode metrics use ``env_steps``. ``wandb`` is imported only
when ``metrics.use_wandb`` is true (extra ``wandb``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from colosseum.core.config import MetricsConfig

logger = logging.getLogger(__name__)

_GLOBAL_PREFIXES = ("ratings/*", "system/*", "episodes/*")


class WandBLogger:
    def __init__(self, config: MetricsConfig, run_name: Optional[str] = None,
                 run_config: Optional[dict[str, Any]] = None) -> None:
        self._wandb: Any = None
        self._run: Any = None
        self._defined_agents: set[str] = set()
        if not config.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            logger.warning("metrics.use_wandb is true but wandb is not installed "
                           "(pip install -e '.[wandb]'); WandB logging is disabled")
            return
        try:
            self._run = wandb.init(project=config.wandb_project, entity=config.wandb_entity,
                                   name=run_name, config=run_config or {})
        except Exception as e:  # noqa: BLE001 - wandb.init raises many unrelated error types
            logger.warning(f"Failed to initialize WandB ({e}); WandB logging is disabled")
            return
        self._wandb = wandb
        wandb.define_metric("env_steps")
        for pattern in _GLOBAL_PREFIXES:
            wandb.define_metric(pattern, step_metric="env_steps")

    @property
    def enabled(self) -> bool:
        return self._wandb is not None

    def log_train(self, agent_id: str, metrics: dict[str, float], train_step: int) -> None:
        """Per-agent training metrics on the axis ``<agent>/train_step``."""
        if self._wandb is None:
            return
        if agent_id not in self._defined_agents:
            self._wandb.define_metric(f"{agent_id}/train_step")
            self._wandb.define_metric(f"{agent_id}/*", step_metric=f"{agent_id}/train_step")
            self._defined_agents.add(agent_id)
        row = {f"{agent_id}/{k}": float(v) for k, v in metrics.items()}
        row[f"{agent_id}/train_step"] = int(train_step)
        self._wandb.log(row)

    def log_global(self, metrics: dict[str, float], env_steps: int) -> None:
        """Ratings / system / episode metrics (already prefixed) on the ``env_steps`` axis."""
        if self._wandb is None:
            return
        self._wandb.log({**metrics, "env_steps": int(env_steps)})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_wandb_logger.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Wire the logger into the hub**

In `src/colosseum/launcher.py`, `Launcher.launch`:

1. Add the import `from colosseum.metrics.wandb_logger import WandBLogger` (module level).

2. Right before `self._hub = MetricsHub(` (added in T6.3), insert:

```python
        self._wandb = WandBLogger(
            cfg.metrics,
            run_name=self._run_dir.root.name,
            run_config=cfg.model_dump(mode="json", by_alias=True),
        )
```

   Add `wandb_logger=self._wandb,` as the last argument of `MetricsHub(...)`.

3. In the `finally:` after the monitor call, after `self._hub.close(...)`, add `self._wandb.finish()`.

Run `grep -rn "log_train_step\|log_config\|log_metrics" src/ tests/`. Expected: no hits. Delete any test that still calls the removed methods; `test_wandb_logger.py` replaces it.

- [ ] **Step 6: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/metrics/wandb_logger.py src/colosseum/launcher.py tests/
git commit -m "fix: wandb per-agent step axes via define_metric, optional import"
```

---
### Task T6.5: CLI and process lifecycle: `sys.path`, exit codes, signals, child-death detection, normal completion

Fixes R5-19, R5-17, R3-08 (detection and exit code), R3-09, R3-17, R6-07, R6-11, and R3-24 (`serve-trajectory` removed).

**Behaviour after this task:**
- `colosseum` puts the current directory first on `sys.path`. Spawned children inherit it.
- Exit codes:
  - 0: the budget was reached and shutdown was clean;
  - 1: a child process died, or the config is invalid;
  - 130: SIGINT;
  - 143: SIGTERM.
- Children:
  - ignore SIGINT;
  - get `PR_SET_PDEATHSIG=SIGTERM` (Linux), so they die if the main process is killed;
  - stop through `stop_event`.
  - Subprocess vec-env grandchildren do the same.
- The main process handles SIGINT and SIGTERM:
  - it sets `stop_event`;
  - it waits up to `SHUTDOWN_GRACE_SEC = 7` seconds while saving final checkpoints and draining results;
  - it then terminates and finally kills stragglers.

  All children are gone within 10 s of the signal.
- Any child exit before the budget is reached is a failure: `"<process> died (exit N), see runs/<name>/logs/<process>.log"`, exit code 1.
- Reaching the budget logs `Training budget reached: ...`, never "unexpected".

**Files:**
- Modify: `src/colosseum/utils/process.py`:
  - add `SHUTDOWN_GRACE_SEC`, `set_parent_death_signal`, `init_child_process`, `ChildFailure` and `ProcessSupervisor`;
  - `run_child` calls `init_child_process`.
- Modify: `src/colosseum/envs/subproc_vec_env.py` (`_worker_loop` calls `init_child_process`)
- Modify: `src/colosseum/launcher.py`:
  - `Launcher.launch` (supervisor, process names, stored queues, tail);
  - replace `_monitor_loop` and `_shutdown`;
  - add `_drain_results` and `_drain_metrics`;
  - `run_training` returns an exit code.
- Modify: `src/colosseum/distributed.py` (`run_distributed_workers` uses the supervisor and returns an exit code)
- Modify: `src/colosseum/cli.py`:
  - `main` inserts the cwd into `sys.path`;
  - exit codes for `train`, `run-learner`, `run-workers`;
  - delete `serve-trajectory`.
- Test: `tests/unit/test_process_lifecycle.py` and `tests/integration/test_lifecycle.py` (new)

**Interfaces:**
- Consumes:
  - `run_child`, `setup_process_logging` and `RunDir` (T6.2)
  - `MetricsHub` (T6.3) and `WandBLogger` (T6.4)
  - `Launcher._env_counter`, `_drain_queue`, `_drain_all_checkpoints`, `_refresh_worker_matches` and `_save_checkpoint` (T5.3)
  - `_queue_depths` (T6.3)
- Produces (in `colosseum.utils.process`):
  - `SHUTDOWN_GRACE_SEC = 7.0`
  - `set_parent_death_signal(sig: int = signal.SIGTERM) -> bool`
  - `init_child_process() -> None`
  - `ChildFailure(name, exitcode, log_path)` with `.message() -> str`
  - `ProcessSupervisor(stop_event, log_dir=None)` with:
    - `.add(name, proc)` and `.names`;
    - `.install_signal_handlers()`, `.restore_signal_handlers()` and `.received_signal`;
    - `.first_failure(nonzero_only=False) -> ChildFailure | None`;
    - `.alive() -> list[str]`;
    - `.wait_all(timeout, poll=None) -> bool`;
    - `.kill_remaining() -> list[str]`.
- Produces (elsewhere):
  - `Launcher.launch() -> int` (the exit code)
  - `run_training(config_path, overrides=None) -> int`
  - `run_distributed_workers(...) -> int`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_process_lifecycle.py`:

```python
"""ProcessSupervisor, child-process init and failure messages (T6.5)."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

import colosseum.utils.process as process_module
from colosseum.utils.process import ChildFailure, ProcessSupervisor, init_child_process


def spawn(target, *args, name="child"):
    proc = mp.get_context("spawn").Process(target=target, args=args, name=name, daemon=True)
    proc.start()
    return proc


def test_child_failure_message_points_to_log(tmp_path):
    failure = ChildFailure("worker-0", -9, tmp_path / "logs" / "worker-0.log")
    assert failure.message() == f"worker-0 died (exit -9), see {tmp_path / 'logs' / 'worker-0.log'}"


def test_supervisor_reports_dead_child(tmp_path):
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop, log_dir=tmp_path / "logs")
    sup.add("worker-1", spawn(os._exit, 3, name="worker-1"))
    sup.add("learner-a", spawn(time.sleep, 30, name="learner-a"))
    deadline = time.monotonic() + 20
    while sup.first_failure() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    failure = sup.first_failure()
    assert failure is not None and failure.name == "worker-1" and failure.exitcode == 3
    assert failure.log_path == tmp_path / "logs" / "worker-1.log"
    assert sup.alive() == ["learner-a"]
    assert sup.kill_remaining() == ["learner-a"]
    assert sup.alive() == []


def test_first_failure_nonzero_only_ignores_clean_exit():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    proc = spawn(os._exit, 0, name="worker-0")
    sup.add("worker-0", proc)
    proc.join(20)
    assert sup.first_failure(nonzero_only=True) is None
    assert sup.first_failure().exitcode == 0


def test_wait_all_polls_until_children_exit():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    sup.add("sleeper", spawn(time.sleep, 0.5, name="sleeper"))
    polls = []
    assert sup.wait_all(20.0, poll=lambda: polls.append(1))
    assert polls


def test_signal_handlers_set_stop_event_and_record_signal():
    stop = mp.get_context("spawn").Event()
    sup = ProcessSupervisor(stop)
    before = signal.getsignal(signal.SIGTERM)
    sup.install_signal_handlers()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 5
        while not stop.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stop.is_set() and sup.received_signal == signal.SIGTERM
    finally:
        sup.restore_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) == before


def test_init_child_process_ignores_sigint_and_sets_death_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(process_module, "set_parent_death_signal", lambda sig=signal.SIGTERM: calls.append(sig) or True)
    old = signal.getsignal(signal.SIGINT)
    try:
        init_child_process()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert calls == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGINT, old)
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_process_lifecycle.py -v`

Expected: collection fails with `ImportError: cannot import name 'ChildFailure' from 'colosseum.utils.process'`.

- [ ] **Step 3: Add the lifecycle helpers**

Replace the whole content of `src/colosseum/utils/process.py` with:

```python
"""Child-process helpers: entry wrapper, signal policy, supervision and shutdown."""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from colosseum.utils.logging import ENV_LOG_DIR, ENV_PROCESS_NAME, setup_process_logging

logger = logging.getLogger(__name__)

# Grace period between setting stop_event and terminating stragglers. Below the 10 s
# promised for SIGTERM (spec §3.5), so terminate/kill still fit into that window.
SHUTDOWN_GRACE_SEC = 7.0
_PR_SET_PDEATHSIG = 1


def set_parent_death_signal(sig: int = signal.SIGTERM) -> bool:
    """Linux: deliver ``sig`` to this process when its parent dies. False elsewhere."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def init_child_process() -> None:
    """Signal policy for every child: ignore Ctrl-C (the main process coordinates the stop)
    and die if the main process disappears (R3-17, R6-07)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    set_parent_death_signal(signal.SIGTERM)


def run_child(name: str, log_dir: Optional[str], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Body of every child-process target.

    Sets up logging first, applies the child signal policy, exports the log dir and
    process name for grandchildren, and records a crash traceback in the process's
    own log before re-raising.
    """
    setup_process_logging(log_dir, name)
    init_child_process()
    if log_dir is not None:
        os.environ[ENV_LOG_DIR] = str(log_dir)
    os.environ[ENV_PROCESS_NAME] = name
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.exception(f"{name} crashed")
        raise
    logger.info(f"{name} finished")


@dataclass
class ChildFailure:
    name: str
    exitcode: Optional[int]
    log_path: Optional[Path]

    def message(self) -> str:
        where = f", see {self.log_path}" if self.log_path is not None else ""
        return f"{self.name} died (exit {self.exitcode}){where}"


class ProcessSupervisor:
    """Named child processes plus the main-process signal handling."""

    def __init__(self, stop_event, log_dir: str | Path | None = None) -> None:
        self._stop_event = stop_event
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._procs: dict[str, Any] = {}
        self._old_handlers: dict[int, Any] = {}
        self.received_signal: Optional[int] = None

    def add(self, name: str, proc) -> None:
        self._procs[name] = proc

    @property
    def names(self) -> list[str]:
        return list(self._procs)

    def install_signal_handlers(self) -> None:
        """SIGINT/SIGTERM set ``stop_event``; the first signal received is remembered."""
        def _handler(signum, _frame):
            if self.received_signal is None:
                self.received_signal = signum
                logger.warning(f"Received {signal.Signals(signum).name}; stopping")
            self._stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[sig] = signal.signal(sig, _handler)

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        self._old_handlers.clear()

    def first_failure(self, nonzero_only: bool = False) -> Optional[ChildFailure]:
        """The first child that has exited (with a non-zero code if ``nonzero_only``)."""
        for name, proc in self._procs.items():
            code = proc.exitcode
            if code is None or (nonzero_only and code == 0):
                continue
            log_path = self._log_dir / f"{name}.log" if self._log_dir is not None else None
            return ChildFailure(name, code, log_path)
        return None

    def alive(self) -> list[str]:
        return [name for name, proc in self._procs.items() if proc.is_alive()]

    def wait_all(self, timeout: float, poll: Optional[Callable[[], None]] = None) -> bool:
        """Wait up to ``timeout`` seconds for every child to exit, calling ``poll`` meanwhile."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if poll is not None:
                poll()
            if not self.alive():
                return True
            time.sleep(0.05)
        return not self.alive()

    def kill_remaining(self) -> list[str]:
        """terminate(), then kill() whatever is still alive. Returns the names that had to be stopped."""
        remaining = self.alive()
        for name in remaining:
            self._procs[name].terminate()
        for name in remaining:
            self._procs[name].join(1.0)
        for name in self.alive():
            self._procs[name].kill()
            self._procs[name].join(1.0)
        return remaining
```

In `src/colosseum/envs/subproc_vec_env.py`, `_worker_loop`, add right after the `setup_process_logging(...)` line from T6.2:

```python
    from colosseum.utils.process import init_child_process

    init_child_process()
```

`run_child` now changes the signal disposition of the calling process. The T6.2 unit test calls it inside the pytest process, so neutralize the policy there: in `tests/unit/test_run_dir_logging.py`, make this the first statement of `test_run_child_logs_finish_and_crash`:

```python
    monkeypatch.setattr("colosseum.utils.process.init_child_process", lambda: None)
```

- [ ] **Step 4: Run the unit tests**

Run: `.venv/bin/python -m pytest tests/unit/test_process_lifecycle.py tests/unit/test_run_dir_logging.py -v`

Expected: all PASS.

- [ ] **Step 5: Launcher: supervisor, monitor loop, shutdown, exit code**

In `src/colosseum/launcher.py`:

1. Add the import:

```python
from colosseum.utils.process import SHUTDOWN_GRACE_SEC, ProcessSupervisor
```

2. In `Launcher.launch`, right before the learner-start loop (`for aid in trainable_agents:` that creates `learner_proc`), insert:

```python
        self._supervisor = ProcessSupervisor(self._stop_event, log_dir=self._run_dir.logs)
        self._supervisor.install_signal_handlers()
```

   After `learner_proc.start()`, add `self._supervisor.add(f"learner-{aid}", learner_proc)`. After `worker_proc.start()`, add `self._supervisor.add(f"worker-{worker_id}", worker_proc)`.

3. Next to the attributes stored in T5.3 and T6.3, add:

```python
        self._results_queue = results_queue
        self._metrics_queue = metrics_queue
```

4. Replace everything from `logger.info("Training started. Press Ctrl+C to stop.")` to the end of `launch()` with the block below. It replaces the old `try/except KeyboardInterrupt/finally`. Change the signature of `launch` to `def launch(self) -> int:`.

```python
        logger.info("Training started. Press Ctrl+C to stop.")
        code = 1
        try:
            code = self._monitor_loop(coordinator, trainable_agents, command_queues, worker_sent_ckpts)
        finally:
            self._shutdown()
            self._hub.close(env_steps=int(self._env_counter.value),
                            ratings=coordinator.ratings_snapshot(),
                            queue_depths=_queue_depths(self._trajectory_queues))
            self._wandb.finish()
            self._supervisor.restore_signal_handlers()
        return code
```

5. Replace `_monitor_loop` and `_shutdown` with the methods below, and add `_drain_results` and `_drain_metrics`:

```python
    def _drain_results(self) -> None:
        for result in _drain_queue(self._results_queue):
            self._coordinator.report_match_result(result)
            self._hub.on_match_result(result)

    def _drain_metrics(self) -> None:
        for item in _drain_queue(self._metrics_queue):
            if item.get("kind") == "worker_stats":
                self._hub.on_worker_stats(item)
            else:
                self._hub.on_train_metrics(item)

    def _monitor_loop(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
    ) -> int:
        """Run until the budget is reached, a signal arrives, or a child dies.

        Returns the exit code: 0 budget reached, 1 child failure, 128+signum on SIGINT/SIGTERM.
        """
        total = self._config.training.total_timesteps
        refresh_interval = self._config.rollout.match_refresh_interval_sec
        last_refresh = time.monotonic()
        while True:
            self._drain_all_checkpoints()
            self._drain_results()
            self._drain_metrics()
            env_steps = int(self._env_counter.value)
            self._hub.maybe_tick(env_steps=env_steps, ratings=coordinator.ratings_snapshot(),
                                 queue_depths=_queue_depths(self._trajectory_queues))
            if self._supervisor.received_signal is not None:
                return 128 + int(self._supervisor.received_signal)
            if env_steps >= total:
                logger.info(f"Training budget reached: {env_steps} env steps (budget {total})")
                return 0
            failure = self._supervisor.first_failure()
            if failure is not None:
                logger.error(failure.message())
                return 1
            if refresh_interval > 0 and time.monotonic() - last_refresh >= refresh_interval:
                self._refresh_worker_matches(coordinator, agent_ids, command_queues, worker_sent_ckpts)
                last_refresh = time.monotonic()
            time.sleep(0.2)

    def _shutdown(self) -> None:
        """Stop all children within SHUTDOWN_GRACE_SEC.

        Final checkpoints and results are saved while the children wind down; stragglers
        are then terminated and killed.
        """
        self._stop_event.set()

        def poll() -> None:
            self._drain_all_checkpoints()
            self._drain_results()
            self._drain_metrics()

        clean = self._supervisor.wait_all(SHUTDOWN_GRACE_SEC, poll=poll)
        poll()
        killed = self._supervisor.kill_remaining()
        if killed:
            logger.warning(f"Terminated processes that did not stop within {SHUTDOWN_GRACE_SEC:.0f}s: "
                           f"{', '.join(killed)}")
            poll()
        for q in self._all_queues:
            try:
                q.cancel_join_thread()
            except (AttributeError, OSError):
                pass
        logger.info("All processes stopped" if clean else "All processes stopped (some were terminated)")
```

   Delete any leftover code of the old loop: the "All learner processes exited" and "All worker processes exited unexpectedly" branches, and the separate budget check from T2.5. The new loop covers all of them.

6. Replace `run_training` with:

```python
def run_training(config_path: str, overrides: dict | None = None) -> int:
    """Entry point of ``colosseum train``. Returns the process exit code."""
    from colosseum.utils.seeding import apply_global_seed

    mp.set_start_method("spawn", force=True)
    setup_process_logging(None, "main", console_level=logging.INFO)
    config = load_config(config_path, overrides)
    apply_global_seed(config.training.seed)  # after overrides (R3-27)
    run_dir = RunDir.create(config, config_path)
    setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)
    run_dir.write_resolved_config(config)
    print(f"Run directory: {run_dir.root}", flush=True)
    code = Launcher(config, run_dir).launch()
    if code == 0:
        logger.info(f"Training finished; outputs in {run_dir.root}")
    return code
```

- [ ] **Step 6: Distributed workers: same supervision**

In `src/colosseum/distributed.py`, `run_distributed_workers`:
- change the return annotation to `-> int`;
- import `from colosseum.utils.process import SHUTDOWN_GRACE_SEC, ProcessSupervisor`;
- replace everything from `stop_event = mp.Event()` to the end of the function with:

```python
    stop_event = mp.Event()
    supervisor = ProcessSupervisor(stop_event, log_dir=run_dir.logs)
    supervisor.install_signal_handlers()
    worker_daemon = config.rollout.vec_env != "subprocess"
    per_worker_steps = config.training.total_timesteps // config.rollout.num_workers

    for worker_id in range(config.rollout.num_workers):
        proc = mp.Process(
            target=_dist_worker_target,
            args=(
                worker_id, config, agent_ids, agent_configs,
                weight_store_address, learner_addresses, stop_event,
                per_worker_steps, slot_agent_map,
            ),
            kwargs={"log_dir": str(run_dir.logs)},
            name=f"worker-{worker_id}",
            daemon=worker_daemon,
        )
        proc.start()
        supervisor.add(f"worker-{worker_id}", proc)

    logger.info(f"Started {config.rollout.num_workers} distributed workers -> learners {learner_addresses}, "
                f"weights <- {weight_store_address}")
    code = 0
    try:
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
        supervisor.kill_remaining()
        supervisor.restore_signal_handlers()
        logger.info("Distributed workers stopped.")
    if supervisor.received_signal is not None:
        code = 128 + int(supervisor.received_signal)
    return code
```

   Keep `args=(...)` identical to what T2.5 left: if T2.5 changed the worker arguments, for example by dropping `per_worker_steps`, keep its version of the tuple and its budget variable.

- [ ] **Step 7: CLI: `sys.path`, exit codes, remove `serve-trajectory`**

In `src/colosseum/cli.py`:

1. Add `import os` and `import sys` at the top. Replace the group function with:

```python
@click.group()
def main() -> None:
    """Colosseum — Distributed RL Training Framework."""
    # User code (e.g. ``examples.*`` or ``my_game.*``) is imported relative to the cwd.
    # Spawned children inherit sys.path (R5-19).
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
```

2. Replace the body of `train` with:

```python
    from colosseum.core.errors import ConfigError
    from colosseum.launcher import run_training

    try:
        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    except ConfigError as e:
        click.echo(f"Config error: {e}", err=True)
        sys.exit(1)
    sys.exit(code)
```

3. In `run_learner_cmd` and `run_workers_cmd`, wrap the call in the same `try/except ConfigError` block. For `run-workers`, end with `sys.exit(code)`, where `code = run_distributed_workers(...)`.

4. Delete the whole `serve-trajectory` command (`@main.command("serve-trajectory")` and `serve_trajectory_cmd`). `serve_trajectory_receiver` stays in `transport/grpc_transport.py`, because `run-learner` uses it.

- [ ] **Step 8: Write the integration tests**

Create `tests/integration/test_lifecycle.py`:

```python
"""Exit codes, signals, child death, README quickstart (T6.5). Linux (/proc) only."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tests.integration.cli_runner import REPO_ROOT, TTT_CONFIG, child_env, run_train, start_train, wait_for

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

FOREVER = {"training.total_timesteps": "100000000"}


def _proc_table() -> dict[int, tuple[int, str]]:
    table = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
        except OSError:
            continue
        rest = stat[stat.rfind(")") + 2:].split()
        table[int(d.name)] = (int(rest[1]), rest[0])  # (ppid, state)
    return table


def descendants(pid: int) -> set[int]:
    table = _proc_table()
    found, frontier = set(), [pid]
    while frontier:
        parent = frontier.pop()
        for child, (ppid, _state) in table.items():
            if ppid == parent and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


def logged_pid(root: Path, process_name: str) -> int | None:
    path = root / "logs" / f"{process_name}.log"
    if not path.exists():
        return None
    match = re.search(rf"{re.escape(process_name)} started \(pid (\d+)\)", path.read_text())
    return int(match.group(1)) if match else None


def training_started(root: Path) -> bool:
    path = root / "metrics.jsonl"
    return path.exists() and '"kind": "train"' in path.read_text()


def test_normal_finish_exit_0_and_final_checkpoint(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="normal")
    assert run.returncode == 0, run.stderr[-3000:]
    main_log = run.log("main")
    assert "Training budget reached" in main_log
    assert "unexpected" not in main_log.lower() and "died" not in main_log
    ckpts = sorted((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
    assert ckpts, "no checkpoint saved"
    metas = [json.loads((c / "meta.json").read_text()) for c in ckpts]
    assert any(m["final"] for m in metas)


def test_killed_worker_gives_exit_1_and_points_to_its_log(tmp_path):
    proc, root = start_train(TTT_CONFIG, tmp_path, name="killw", overrides=FOREVER)
    try:
        assert wait_for(lambda: logged_pid(root, "worker-0") is not None and training_started(root), 120)
        kids = descendants(proc.pid)
        os.kill(logged_pid(root, "worker-0"), signal.SIGKILL)
        assert proc.wait(60) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
    stderr = (tmp_path / "killw.stderr").read_text()
    assert f"worker-0 died (exit -9), see {root / 'logs' / 'worker-0.log'}" in stderr
    assert wait_for(lambda: not any(pid_alive(k) for k in kids), 10)


def test_sigterm_stops_all_descendants_within_10s(tmp_path):
    overrides = {**FOREVER, "rollout.vec_env": "subprocess", "rollout.subproc_workers": "2"}
    proc, root = start_train(TTT_CONFIG, tmp_path, name="term", overrides=overrides)
    try:
        # learner + worker + 2 env processes (+ resource tracker)
        assert wait_for(lambda: len(descendants(proc.pid)) >= 4 and training_started(root), 120)
        kids = descendants(proc.pid)
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(15) == 143
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), max(0.0, 10 - (time.monotonic() - t0)))
    finally:
        if proc.poll() is None:
            proc.kill()
    assert list((root / "checkpoints" / "agent_0").glob("ckpt_v*")), "final checkpoint not saved on SIGTERM"


def test_sigint_to_process_group_exits_130_without_child_tracebacks(tmp_path):
    proc, root = start_train(TTT_CONFIG, tmp_path, name="int", overrides=FOREVER)
    try:
        assert wait_for(lambda: training_started(root), 120)
        kids = descendants(proc.pid)
        os.killpg(proc.pid, signal.SIGINT)  # like Ctrl-C in a terminal
        assert proc.wait(15) == 130
        assert wait_for(lambda: not any(pid_alive(k) for k in kids), 10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "KeyboardInterrupt" not in (tmp_path / "int.stderr").read_text()


def test_config_error_exit_1_without_traceback(tmp_path):
    bad = tmp_path / "bad.yaml"
    data = yaml.safe_load(TTT_CONFIG.read_text())
    data["rollout"]["num_worker"] = 3
    bad.write_text(yaml.safe_dump(data))
    proc = subprocess.run([sys.executable, "-m", "colosseum", "train", "-c", str(bad)], cwd=REPO_ROOT,
                          env=child_env(), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "Config error" in proc.stderr and "num_worker" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_readme_quickstart_from_repo_root(tmp_path):
    exe = Path(sys.executable).parent / "colosseum"
    assert exe.exists(), "console script missing; run scripts/setup-dev.sh"
    cmd = [str(exe), "train", "-c", "configs/examples/tic_tac_toe.yaml",
           "--set", "training.total_timesteps=2000",
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=quickstart"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=child_env(), capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert (tmp_path / "runs" / "quickstart" / "config.resolved.yaml").exists()
```

- [ ] **Step 9: Run the integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_lifecycle.py -v`

Expected: all 6 tests PASS. Each takes 5–25 s.

If `test_sigterm_stops_all_descendants_within_10s` fails because a worker blocks while putting into a full trajectory queue after the learner exited, the bug is in the worker's chunk send. It must wait in pieces of at most 0.5 s and check `stop_event` (T2.x). Fix it there; do not raise the grace period.

- [ ] **Step 10: Remove leftovers and run the full fast suite**

Run:

```bash
grep -rn "serve-trajectory\|serve_trajectory_cmd\|KeyboardInterrupt" src/colosseum/launcher.py src/colosseum/cli.py
```

Expected: no hits in `launcher.py`. In `cli.py`, a hit is acceptable only inside `serve-weight-store`.

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/colosseum/utils/process.py src/colosseum/envs/subproc_vec_env.py src/colosseum/launcher.py src/colosseum/distributed.py src/colosseum/cli.py tests/
git commit -m "fix: exit codes, signal-safe shutdown with final checkpoints, child-death detection, cwd on sys.path"
```

---
### Task T8.1: Examples: tic-tac-toe `active` + mask, chase on numpy 2.x, configs on the new schema, attention example

Fixes R6-09, R6-12 (chase), R5-31 (example constructors without `**kwargs`), and R4-17 / R5 (configs match the code).

**Files:**
- Replace: `examples/tic_tac_toe/env.py` and `examples/tic_tac_toe/networks.py`
- Modify: `examples/composite_action/env.py` (`ChaseEnv.step`) and `examples/composite_action/networks.py` (constructors)
- Replace: `configs/examples/tic_tac_toe.yaml`, `tic_tac_toe_multi.yaml`, `chase.yaml` and `space_miners.yaml`
- Create: `configs/examples/tic_tac_toe_attention.yaml`
- Modify: `deployment/docker-compose.yaml` and `deployment/k8s/learner.yaml` (keep outputs on the mounted volume)
- Test: `tests/unit/test_examples.py` (new)

**Interfaces:**
- Consumes:
  - `build_model(config) -> PolicyModel`. It passes `in_dim=core.output_dim` to heads whose constructor has an `in_dim` parameter (T1.4).
  - `PolicyModel.is_stateful` (T1.2)
  - `WindowAttentionCore(input_dim, d_model=64, window=16, num_heads=4, num_layers=1)` (T1.3); its `output_dim` is `d_model`
  - `colosseum validate` (T6.1)
  - `info["active"]` and `info["action_mask"]` semantics of the worker (T3.2)
- Produces:
  - `TicTacToeEnv` infos: `{"current_player": int, "active": bool, "action_mask": np.ndarray[bool, (9,)]}` per player, in `reset` and in `step`
  - `TicTacToeEncoder(hidden: int = 128, latent: int = 64, **kwargs)` with `latent_dim`
  - `TicTacToePolicy(in_dim: int = 64, **kwargs)` and `TicTacToeValue(in_dim: int = 64, **kwargs)`
  - `ChaseEncoder(**kwargs)`, `ChasePolicy(in_dim: int = 32, **kwargs)` and `ChaseValue(in_dim: int = 32, **kwargs)`
  - The example configs listed above, all valid.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_examples.py`:

```python
"""Example envs and configs (T8.1)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import build_model
from examples.composite_action.env import ChaseEnv
from examples.tic_tac_toe.env import TicTacToeEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO_ROOT / "configs" / "examples").glob("*.yaml"))


def test_tic_tac_toe_reports_active_player_and_legal_moves():
    env = TicTacToeEnv()
    obs, infos = env.reset()
    assert infos[0]["active"] is True and infos[1]["active"] is False
    assert infos[0]["action_mask"].dtype == bool and infos[0]["action_mask"].all()
    _, _, term, _, infos = env.step({0: 4, 1: 0})  # player 1's action is ignored
    assert infos[0]["active"] is False and infos[1]["active"] is True
    expected = np.ones(9, bool)
    expected[4] = False
    assert np.array_equal(infos[1]["action_mask"], expected)
    assert np.array_equal(infos[0]["action_mask"], expected)
    assert not term[0]


def test_tic_tac_toe_full_board_mask_is_all_true():
    env = TicTacToeEnv()
    env.reset()
    # X O X / X O O / O X X  -> draw after 9 legal moves
    for cell in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
        current = env._current_player
        _, rewards, term, _, infos = env.step({current: cell, 1 - current: 0})
    assert term[0] and term[1] and rewards == {0: 0.0, 1: 0.0}
    assert infos[0]["action_mask"].all() and infos[1]["action_mask"].all()


def test_chase_accepts_its_own_sampled_actions_and_scalars():
    env = ChaseEnv()
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(3):
        env.step({0: env.action_space.sample(), 1: env.action_space.sample()})  # speed: shape (1,) array
    env.step({0: {"direction": 1, "speed": 0.5}, 1: {"direction": np.int64(2), "speed": np.float32(1.0)}})


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_example_config_validates(path, monkeypatch):
    if path.name == "space_miners.yaml" and importlib.util.find_spec("Box2D") is None:
        pytest.skip("space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output


def test_attention_example_builds_a_stateful_model():
    cfg = load_config(REPO_ROOT / "configs" / "examples" / "tic_tac_toe_attention.yaml")
    assert cfg.networks.core.class_path == "colosseum.networks.cores.WindowAttentionCore"
    model = build_model(cfg.get_agent_config("agent_0"))
    assert model.is_stateful


def test_example_configs_use_run_section_not_checkpoint_dir():
    for path in CONFIGS:
        cfg = load_config(path)
        assert cfg.run.dir == "runs" and cfg.run.name is None, path.name
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_examples.py -v`

Expected failures:
- `KeyError: 'active'` in the tic-tac-toe tests;
- `TypeError: only length-1 arrays can be converted` or `TypeError: only 0-dimensional arrays can be converted` in the chase test (numpy ≥ 2.x);
- `tic_tac_toe_attention.yaml` not found (the attention test errors).

- [ ] **Step 3: Tic-tac-toe env**

Replace `examples/tic_tac_toe/env.py` with:

```python
"""Tic-Tac-Toe environment for Colosseum (turn-based, 2 players)."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np

from colosseum.envs.base_env import BaseEnv


class TicTacToeEnv(BaseEnv):
    """2-player Tic-Tac-Toe.

    Observation: 3x3x3 from the player's own perspective (own marks, opponent marks, empty).
    Action: 0-8 (board cell).

    Turn-based: every info dict carries
    - ``active``: True for the player whose move it is (the worker runs inference and
      records transitions only for active players);
    - ``action_mask``: bool[9], the empty cells. A full board gives all True, because no one
      acts on it and the worker requires a legal action in an acting player's mask.

    Rewards: win +1 / loss -1 / draw 0. An illegal move (impossible with the mask) loses: -1, +1 to the opponent.
    """

    WINNING_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]

    def __init__(self) -> None:
        self._board = np.zeros(9, dtype=np.int8)  # 0 empty, 1 player 0, 2 player 1
        self._current_player = 0
        self._done = False
        self._step_count = 0

    @property
    def num_players(self) -> int:
        return 2

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(3, 3, 3), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(9)

    def reset(self, seed: Optional[int] = None) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        if seed is not None:
            np.random.seed(seed)
        self._board = np.zeros(9, dtype=np.int8)
        self._current_player = 0
        self._done = False
        self._step_count = 0
        return self._all_obs(), self._infos()

    def step(self, actions: dict[int, Any]) -> tuple[
        dict[int, np.ndarray], dict[int, float], dict[int, bool], dict[int, bool], dict[int, dict],
    ]:
        if self._done:
            return (self._all_obs(), {0: 0.0, 1: 0.0}, {0: True, 1: True}, {0: False, 1: False}, self._infos())

        action = int(actions[self._current_player])
        other = 1 - self._current_player
        rewards = {0: 0.0, 1: 0.0}
        terminated = {0: False, 1: False}
        truncated = {0: False, 1: False}

        if action < 0 or action >= 9 or self._board[action] != 0:
            rewards[self._current_player] = -1.0
            rewards[other] = 1.0
            terminated = {0: True, 1: True}
            self._done = True
        else:
            self._board[action] = self._current_player + 1
            self._step_count += 1
            if self._check_win(self._current_player + 1):
                rewards[self._current_player] = 1.0
                rewards[other] = -1.0
                terminated = {0: True, 1: True}
                self._done = True
            elif self._step_count >= 9:
                terminated = {0: True, 1: True}
                self._done = True
            else:
                self._current_player = other

        return self._all_obs(), rewards, terminated, truncated, self._infos()

    def _infos(self) -> dict[int, dict]:
        legal = self._board == 0
        if not legal.any():
            legal = np.ones(9, dtype=bool)
        return {
            p: {"current_player": self._current_player,
                "active": p == self._current_player,
                "action_mask": legal.copy()}
            for p in range(2)
        }

    def _all_obs(self) -> dict[int, np.ndarray]:
        return {p: self._get_obs(p) for p in range(2)}

    def _get_obs(self, player: int) -> np.ndarray:
        my_mark = player + 1
        opp_mark = 2 - player
        board = self._board.reshape(3, 3)
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[0] = board == my_mark
        obs[1] = board == opp_mark
        obs[2] = board == 0
        return obs

    def _check_win(self, mark: int) -> bool:
        return any(all(self._board[i] == mark for i in line) for line in self.WINNING_LINES)
```

- [ ] **Step 4: Tic-tac-toe networks accept `in_dim`**

Replace `examples/tic_tac_toe/networks.py` with:

```python
"""Networks for the Tic-Tac-Toe example (parts of a ComposedModel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist


class TicTacToeEncoder(BaseEncoder):
    """MLP over the 3x3x3 board."""

    def __init__(self, hidden: int = 128, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TicTacToePolicy(BasePolicy):
    """Categorical over 9 cells. ``in_dim`` is the core's output size (set by build_model)."""

    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 9)

    def forward(self, features: torch.Tensor) -> CategoricalDist:
        return CategoricalDist(logits=self.net(features))


class TicTacToeValue(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

- [ ] **Step 5: Chase on numpy 2.x**

In `examples/composite_action/env.py`, `ChaseEnv.step`, replace the two lines

```python
            d = int(a["direction"])
            s = float(np.clip(a["speed"], 0.0, 1.0)) if not isinstance(a["speed"], (int, float)) else float(np.clip(a["speed"], 0.0, 1.0))
```

with

```python
            d = int(np.asarray(a["direction"]).reshape(-1)[0])
            # speed may be a scalar (ActionSpec.decode) or a shape-(1,) array (action_space.sample());
            # float() of a 1-element array is an error on numpy 2.x (R6-12).
            s = float(np.clip(np.asarray(a["speed"], dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
```

In `examples/composite_action/networks.py`:
- change `ChaseEncoder.__init__(self)` to `__init__(self, **kwargs)`;
- change `ChasePolicy.__init__(self)` to `__init__(self, in_dim: int = _LATENT, **kwargs)` and use `in_dim` in place of `_LATENT` in its three layer definitions;
- change `ChaseValue.__init__(self)` to `__init__(self, in_dim: int = _LATENT, **kwargs)` and use `in_dim` in its first `nn.Linear`.

- [ ] **Step 6: Example configs on the new schema**

Replace `configs/examples/tic_tac_toe.yaml` with:

```yaml
# Tic-tac-toe self-play. README quickstart: about 3 minutes on 8 CPU cores, then the latest
# checkpoint beats a random legal-move player in >= 80% of games (tests/learning/test_ttt_slow.py).
run:
  name: null              # default: tic_tac_toe-<YYYYmmdd-HHMMSS>
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.env.TicTacToeEnv"
  num_players: 2
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.networks.TicTacToeEncoder"
  core: null              # stateless (MLP); see tic_tac_toe_attention.yaml for a core
  policy_class: "examples.tic_tac_toe.networks.TicTacToePolicy"
  value_class: "examples.tic_tac_toe.networks.TicTacToeValue"
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
  learning_rate: 3.0e-4
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
  phase: "self_play"
  total_timesteps: 600000   # global env steps across all workers
  seed: null

self_play:
  checkpoint_interval: 100  # train steps
  pool_size: 10
  latest_prob: 0.8
  shuffle_seats: true

checkpoint:
  save_optimizer: true

metrics:
  use_wandb: false
  wandb_project: "colosseum"
  log_interval: 10
  console_interval_sec: 10.0
```

Replace `configs/examples/tic_tac_toe_multi.yaml` with:

```yaml
# Two trainable agents in a league: every match is an arena match (self_play_ratio 0.0).
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.env.TicTacToeEnv"
  num_players: 2
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.networks.TicTacToeEncoder"
  core: null
  policy_class: "examples.tic_tac_toe.networks.TicTacToePolicy"
  value_class: "examples.tic_tac_toe.networks.TicTacToeValue"
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
  phase: "league"
  total_timesteps: 300000

self_play:
  checkpoint_interval: 100
  pool_size: 5
  latest_prob: 0.8
  self_play_ratio: 0.0
  pfsp_exponent: 1.0
  shuffle_seats: true

checkpoint:
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

Create `configs/examples/tic_tac_toe_attention.yaml`:

```yaml
# Tic-tac-toe with a stateful core: causal attention over the last 8 latents of the episode.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.tic_tac_toe.env.TicTacToeEnv"
  num_players: 2
  kwargs: {}

networks:
  encoder_class: "examples.tic_tac_toe.networks.TicTacToeEncoder"
  core:
    class: "colosseum.networks.cores.WindowAttentionCore"
    kwargs: {d_model: 64, window: 8, num_heads: 4, num_layers: 1}
  policy_class: "examples.tic_tac_toe.networks.TicTacToePolicy"   # gets in_dim = core.output_dim
  value_class: "examples.tic_tac_toe.networks.TicTacToeValue"
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
  phase: "self_play"
  total_timesteps: 600000

self_play:
  checkpoint_interval: 100
  pool_size: 10
  latest_prob: 0.8

metrics:
  use_wandb: false
  log_interval: 10
```

Replace `configs/examples/chase.yaml` with:

```yaml
# Composite Dict action space (direction: Discrete(4), speed: Box(1)).
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.composite_action.env.ChaseEnv"
  num_players: 2
  kwargs: {}

networks:
  encoder_class: "examples.composite_action.networks.ChaseEncoder"
  core: null
  policy_class: "examples.composite_action.networks.ChasePolicy"
  value_class: "examples.composite_action.networks.ChaseValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 1.0
  entropy_coeff: 0.01
  learning_rate: 3.0e-4
  lr_schedule: "constant"

rollout:
  chunk_length: 20
  num_workers: 2
  envs_per_worker: 4
  weight_sync_interval_sec: 2.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  phase: "self_play"
  total_timesteps: 50000

self_play:
  checkpoint_interval: 10
  pool_size: 5
  latest_prob: 0.5

metrics:
  use_wandb: false
  log_interval: 10
```

Replace `configs/examples/space_miners.yaml` with:

```yaml
# Requires Box2D: pip install -e ".[examples]"
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.space_miners.env.SpaceMinersEnv"
  num_players: 2
  kwargs:
    preset: "Round 1"
    max_ticks: 1000

networks:
  encoder_class: "examples.space_miners.networks.SpaceMinersEncoder"
  core: null
  policy_class: "examples.space_miners.networks.SpaceMinersPolicy"
  value_class: "examples.space_miners.networks.SpaceMinersValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 1.0
  entropy_coeff: 0.01
  learning_rate: 3.0e-4
  lr_schedule: "linear"

rollout:
  chunk_length: 256
  num_workers: 2
  envs_per_worker: 4
  weight_sync_interval_sec: 5.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 64
  batch_chunks: 16

training:
  phase: "self_play"
  total_timesteps: 500000

self_play:
  checkpoint_interval: 100
  pool_size: 10
  latest_prob: 0.5

metrics:
  use_wandb: false
  log_interval: 50
```

- [ ] **Step 7: Deployment keeps outputs on the volume**

The learner's outputs moved from `./checkpoints` to `runs/<name>/`. Keep them on the mounted volume, so restarts do not lose them:
- In `deployment/docker-compose.yaml`, after each line `"--config", "configs/examples/tic_tac_toe.yaml",`, add the line `"--set", "run.dir=/app/checkpoints",`.
- In `deployment/k8s/learner.yaml`, after its `"--config", "configs/examples/tic_tac_toe.yaml",` line, add the same line with the same indentation.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_examples.py -v`

Expected: PASS. The `space_miners.yaml` case is skipped only if Box2D is not installed.

If a config fails `validate` with a key error, the key does not exist after Parts A–C. Remove it from the YAML; do not add config fields here.

- [ ] **Step 9: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS. The tic-tac-toe integration runs from T5.4–T6.5 now use the turn-based env. Their assertions do not depend on episode lengths.

- [ ] **Step 10: Commit**

```bash
git add examples/ configs/examples/ deployment/docker-compose.yaml deployment/k8s/learner.yaml tests/unit/test_examples.py
git commit -m "feat: turn-based tic-tac-toe with masks, numpy-2 chase fix, configs on the new schema, attention example"
```

---
### Task T8.2: Learning tests: bandit and chain (fast), tic-tac-toe ≥ 80% vs random (slow)

Spec §3.3.
- **Fast tests** run on every suite run and finish in seconds. They drive the real `RolloutLoop` and the real `APPO` in-process, round-tripping every chunk through `to_payload`/`from_payload`.
- **Slow test** runs the real `colosseum train` on `configs/examples/tic_tac_toe.yaml`. It then evaluates the latest checkpoint greedily, with the action mask, against a uniformly random legal-move player.

**Files:**
- Create:
  - `tests/learning/__init__.py` (empty, if T0.2 did not create it)
  - `tests/learning/learning_envs.py`
  - `tests/learning/ttt_eval.py`
- Test:
  - `tests/learning/test_fast_learning.py`
  - `tests/learning/test_ttt_slow.py`
- Possibly modify: `configs/examples/tic_tac_toe.yaml` (calibration, Step 6)

**Interfaces:**
- Consumes:
  - `RolloutLoop(...)`, `LoopIO(send_chunk, poll_weights, ...)` and `RolloutLoop.step/sync_weights/close` (contract)
  - `APPO(model, config, device="cpu")`, `algo.train_step`, `algo.set_progress`, `algo.model` and `algo.policy_version`
  - `AlgorithmConfig(learning_rate, lr_schedule, gamma, entropy_coeff, ...)`
  - `WeightPayload.from_model(agent_id, version, model)`
  - `TrajectoryChunk.to_payload` and `from_payload`
  - `ComposedModel(encoder, core, policy_head, value_head)` and `NoCore(input_dim)`
  - `act(model, obs, state, action_mask=None, deterministic=False) -> ActOutput`
  - `PolicyModel.step(obs, state, action_mask) -> StepOutput` with `.dist.log_prob(actions)`
  - `build_model`, `load_config` and `CheckpointManager.latest/load_model` (T5.3), `numpy_state_to_torch` (T5.3)
  - `tests/integration/cli_runner.child_env` and `REPO_ROOT` (T6.2)
  - The `slow` marker and pytest-timeout (T0.2)
- Produces:
  - `tests.learning.learning_envs`: `ContextualBandit`, `ShortChain`, `make_mlp_model(obs_dim, num_actions, hidden=32) -> ComposedModel`
  - `tests.learning.ttt_eval`: `load_latest_model(run_root, agent_id="agent_0")` and `play_vs_random(model, num_games=400, seed=0) -> dict`

- [ ] **Step 1: Toy envs and model factory**

Create `tests/learning/learning_envs.py`:

```python
"""Tiny single-player envs that APPO must solve in seconds (spec §3.3)."""
from __future__ import annotations

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.distributions import CategoricalDist


class ContextualBandit(BaseEnv):
    """One step per episode. Observation: one-hot context c in {0, 1}. Reward 1 iff action == c."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context = 0

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(2)

    def _obs(self):
        obs = np.zeros(2, np.float32)
        obs[self._context] = 1.0
        return {0: obs}

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._context = int(self._rng.integers(2))
        return self._obs(), {0: {}}

    def step(self, actions):
        reward = 1.0 if int(actions[0]) == self._context else 0.0
        return self._obs(), {0: reward}, {0: True}, {0: False}, {0: {}}


class ShortChain(BaseEnv):
    """States 0..4, start at 0, actions 0=left 1=right. Reaching 4 gives +1 and ends the
    episode; 20 steps truncate. The optimal policy always goes right."""

    LENGTH = 5
    MAX_STEPS = 20

    def __init__(self) -> None:
        self._pos = 0
        self._t = 0

    @property
    def num_players(self) -> int:
        return 1

    @property
    def observation_space(self):
        return gymnasium.spaces.Box(0.0, 1.0, (self.LENGTH,), np.float32)

    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(2)

    def _obs(self):
        obs = np.zeros(self.LENGTH, np.float32)
        obs[self._pos] = 1.0
        return {0: obs}

    def reset(self, seed=None):
        self._pos, self._t = 0, 0
        return self._obs(), {0: {}}

    def step(self, actions):
        self._t += 1
        self._pos = min(self.LENGTH - 1, self._pos + 1) if int(actions[0]) == 1 else max(0, self._pos - 1)
        reached = self._pos == self.LENGTH - 1
        reward = 1.0 if reached else 0.0
        truncated = (not reached) and self._t >= self.MAX_STEPS
        return self._obs(), {0: reward}, {0: reached}, {0: truncated}, {0: {}}


class _Encoder(BaseEncoder):
    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self._hidden = hidden
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _Policy(BasePolicy):
    def __init__(self, in_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, num_actions)

    def forward(self, features: torch.Tensor) -> CategoricalDist:
        return CategoricalDist(logits=self.net(features))


class _Value(BaseValue):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_mlp_model(obs_dim: int, num_actions: int, hidden: int = 32) -> ComposedModel:
    return ComposedModel(_Encoder(obs_dim, hidden), NoCore(hidden), _Policy(hidden, num_actions), _Value(hidden))
```

- [ ] **Step 2: Write the fast learning tests**

Create `tests/learning/test_fast_learning.py`:

```python
"""Bandit and short chain are solved in seconds by the real RolloutLoop + APPO (spec §3.3)."""
from __future__ import annotations

import functools
import time

import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk, WeightPayload
from colosseum.networks.model import act
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from tests.learning.learning_envs import ContextualBandit, ShortChain, make_mlp_model


def train_until_solved(env_cls, obs_dim, solved, *, gamma=0.9, lr=3e-3, num_envs=8, chunk_length=16,
                       batch_chunks=4, max_updates=400, seed=0) -> int:
    """Run collection and training in-process. Returns the number of updates used, or -1."""
    torch.manual_seed(seed)
    model_fn = functools.partial(make_mlp_model, obs_dim, 2)
    algo = APPO(model_fn(), AlgorithmConfig(learning_rate=lr, lr_schedule="constant", gamma=gamma,
                                            entropy_coeff=0.01), device="cpu")
    chunks: list[TrajectoryChunk] = []
    latest = {"payload": WeightPayload.from_model("agent_0", algo.policy_version, algo.model)}
    io = LoopIO(
        send_chunk=lambda c: chunks.append(TrajectoryChunk.from_payload(c.to_payload())),
        poll_weights=lambda agent_id: latest["payload"],
    )
    loop = RolloutLoop(worker_id=0, env_fn=env_cls, num_envs=num_envs, chunk_length=chunk_length,
                       agent_ids=["agent_0"], model_factories={"agent_0": model_fn}, io=io,
                       gamma=gamma, weight_sync_interval=0.0, seed=seed)
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
        ContextualBandit, 2,
        lambda m: greedy_and_confident(m, torch.eye(2), torch.tensor([0, 1]), min_prob=0.9))
    assert updates != -1, "bandit not solved within 400 updates"
    assert time.monotonic() - start < 60


def test_short_chain_is_solved_in_seconds():
    start = time.monotonic()
    states = torch.eye(ShortChain.LENGTH)[: ShortChain.LENGTH - 1]  # every non-terminal state
    updates = train_until_solved(
        ShortChain, ShortChain.LENGTH,
        lambda m: greedy_and_confident(m, states, torch.ones(ShortChain.LENGTH - 1, dtype=torch.long), 0.8))
    assert updates != -1, "chain not solved within 400 updates"
    assert time.monotonic() - start < 60
```

- [ ] **Step 3: Run the fast learning tests**

Run: `.venv/bin/python -m pytest tests/learning/test_fast_learning.py -v`

Expected: both PASS, each in under 20 s.

If one fails with "not solved", first print `updates` and the policy probabilities at `max_updates`. A policy stuck near 0.5 on the bandit means rewards do not reach the transitions: a regression in T3.1/T3.2, so fix the producer, not the test. Do not raise `max_updates` above 400 or lower the thresholds.

- [ ] **Step 4: Tic-tac-toe evaluation helper**

Create `tests/learning/ttt_eval.py`:

```python
"""Evaluate a trained tic-tac-toe model against a uniformly random legal-move player."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from colosseum.coordinator.checkpoint_manager import CheckpointManager, numpy_state_to_torch
from colosseum.core.config import load_config
from colosseum.core.registry import build_model
from colosseum.networks.model import act
from examples.tic_tac_toe.env import TicTacToeEnv


def load_latest_model(run_root: Path, agent_id: str = "agent_0"):
    cfg = load_config(Path(run_root) / "config.resolved.yaml").get_agent_config(agent_id)
    model = build_model(cfg)
    manager = CheckpointManager(Path(run_root) / "checkpoints")
    latest = manager.latest(agent_id)
    assert latest is not None, f"no checkpoints for {agent_id} in {run_root}"
    model.load_state_dict(numpy_state_to_torch(manager.load_model(agent_id, latest.checkpoint_id)))
    model.eval()
    return model


@torch.no_grad()
def play_vs_random(model, num_games: int = 400, seed: int = 0) -> dict:
    """The agent alternates seats (game % 2), plays greedily with the action mask, and keeps
    its own model state (advanced only on its own moves, as in the worker)."""
    rng = np.random.default_rng(seed)
    env = TicTacToeEnv()
    per_seat = {0: [0, 0, 0], 1: [0, 0, 0]}  # W, D, L
    for game in range(num_games):
        agent_seat = game % 2
        obs, infos = env.reset(seed=seed + game)
        state = model.initial_state(1)
        while True:
            current = next(p for p in (0, 1) if infos[p]["active"])
            if current == agent_seat:
                out = act(model, torch.as_tensor(obs[current][None], dtype=torch.float32), state,
                          action_mask=torch.as_tensor(infos[current]["action_mask"][None]),
                          deterministic=True)
                state = out.state
                action = int(out.actions.reshape(-1)[0])
            else:
                action = int(rng.choice(np.flatnonzero(infos[current]["action_mask"])))
            obs, rewards, terminated, truncated, infos = env.step({current: action, 1 - current: 0})
            if terminated[0] or truncated[0]:
                r = rewards[agent_seat]
                per_seat[agent_seat][0 if r > 0 else (1 if r == 0 else 2)] += 1
                break
    wins = per_seat[0][0] + per_seat[1][0]
    return {"win_rate": wins / num_games, "per_seat": per_seat, "games": num_games}
```

- [ ] **Step 5: Write the slow tic-tac-toe test**

Create `tests/learning/test_ttt_slow.py`:

```python
"""Tic-tac-toe with mask and `active` reaches >= 80% greedy wins vs random in ~3 min on 8 cores (spec §3.3)."""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from tests.integration.cli_runner import REPO_ROOT, child_env
from tests.learning.ttt_eval import load_latest_model, play_vs_random


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_tic_tac_toe_beats_random_80_percent(tmp_path):
    cmd = [sys.executable, "-m", "colosseum", "train", "-c", "configs/examples/tic_tac_toe.yaml",
           "--set", f"run.dir={tmp_path / 'runs'}", "--set", "run.name=ttt", "--set", "training.seed=0"]
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=child_env(), capture_output=True, text=True, timeout=1100)
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = play_vs_random(load_latest_model(tmp_path / "runs" / "ttt"), num_games=400, seed=0)
    print(f"training took {elapsed:.0f}s; vs random: {result}")
    assert result["win_rate"] >= 0.80, result
```

- [ ] **Step 6: Run the slow test and calibrate the example config if needed**

Run: `.venv/bin/python -m pytest tests/learning/test_ttt_slow.py -m slow -v -s`

Expected: PASS, with `training took ~180s` (the 8-core dev box) and `win_rate >= 0.80`.

If it fails, change only `configs/examples/tic_tac_toe.yaml`, in this order, rerunning after each change:
1. `algorithm.learning_rate: 1.0e-3`;
2. `training.total_timesteps: 1000000`.

Never lower the 0.80 threshold. Record the final wall time; T8.3 puts it into the README quickstart.

- [ ] **Step 7: Run the full fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: PASS. The slow test is deselected.

- [ ] **Step 8: Commit**

```bash
git add tests/learning/ configs/examples/tic_tac_toe.yaml
git commit -m "test: learning checks (bandit, chain fast; tic-tac-toe >=80% vs random slow)"
```

---
### Task T8.3: Docs: README, CLAUDE.md status and roadmap, `docs/GPU_CHECKS.md`, benchmarks "after"

Fixes R4-17, the R5 documentation findings, R3-01 (distributed docs) and spec §3.7–3.8.

Each README command must work verbatim from the repo root after `scripts/setup-dev.sh`. T8.4 runs them.

**Files:**
- Replace: `README.md`
- Modify: `CLAUDE.md`:
  - replace the section from `## Implementation Status` up to `## Tech Stack` (not included);
  - fix the eval command in `### Eval Mode`;
  - fix the KL line in `## AGREED Architecture Decisions` → `**Behavioral Cloning:**`.
- Create: `docs/GPU_CHECKS.md` (Russian)
- Modify: `docs/benchmarks.md` (add the «После SP1» section, Russian)
- Modify (or create): `.gitignore` (add `runs/` and `eval.json`)

**Interfaces:**
- Consumes:
  - CLI commands and flags, as they exist after T4.4, T6.5 and T7.2: `train`, `validate`, `eval -a name=path --num-matches --output`, `bc --data --output --epochs --seq-len`, `run-learner`, `run-workers`, `serve-weight-store`. Step 1 re-checks them with `--help`.
  - `scripts/setup-dev.sh [--gpu]` (T0.1)
  - `scripts/bench_throughput.py` and the "до" section with its exact command in `docs/benchmarks.md` (T0.4)
  - The tic-tac-toe wall time and win rate measured in T8.2 Step 6.
- Produces: documentation only.

- [ ] **Step 1: Check the real CLI surface**

Run:

```bash
for c in train validate eval bc run-learner run-workers serve-weight-store; do .venv/bin/colosseum $c --help > /tmp/help-$c.txt || echo "MISSING $c"; done
grep -l "serve-trajectory" /tmp/help-*.txt; .venv/bin/colosseum --help
```

Expected:
- no `MISSING` line;
- `serve-trajectory` is not listed;
- `eval --help` shows `-a/--agents` with the `name=path` format, `--num-matches` described per pair, and `--output`.

If any flag name differs from the README text below, change the README text to the real flag. Do not change the CLI in this task.

- [ ] **Step 2: Replace `README.md`**

Replace `README.md` with the text below. Replace `~3 minutes` and `>= 80%` in the Quick start with the values measured in T8.2 Step 6, if they differ.

````markdown
# Colosseum

A reusable training framework for competitive bot-programming competitions (Lux AI, Neural MMO, CodeCraft, ...):
behavioural cloning → RL → self-play → league, built on an IMPALA-style asynchronous actor–learner (APPO with V-trace)
in PyTorch, without Ray.

> **Status (SP1 of 6).** Single-machine training is the supported mode: correct, observable and robust for
> 1v1 games with simultaneous or turn-based moves, solo games, multi-agent self-play and a league where all agents meet.
> Read [Status and limitations](#status-and-limitations) before relying on anything else.

## Quick start

```bash
git clone <repo-url> Colosseum && cd Colosseum
scripts/setup-dev.sh            # uv + .venv (Python 3.12) + CPU torch + colosseum[grpc,dev,examples]
source .venv/bin/activate

colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
```

`scripts/setup-dev.sh --gpu` installs the CUDA build of torch instead of the CPU one.

Training prints its run directory and one progress line per agent every 10 seconds:

```
Run directory: runs/ttt-quickstart
[agent_0] step 120 |  20.5% budget | 3,410 env-steps/s | loss 0.0412 | entropy 1.873 | return 0.312 | wr_vs_past 0.64
```

On an 8-core CPU the run takes ~3 minutes. Afterwards the latest checkpoint wins >= 80% of games against a random
legal-move player (checked by `tests/learning/test_ttt_slow.py`). Re-running with the same `run.name` is refused;
pick another name or delete `runs/ttt-quickstart`.

Evaluate the newest checkpoint against the oldest one still kept:

```bash
NEW=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
OLD=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | head -1)
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json
```

Continue training from that run (policy versions, optimizer state and the env-step counter continue):

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-continued \
  --set training.resume_from=runs/ttt-quickstart --set training.total_timesteps=1200000
```

A two-agent league:

```bash
colosseum train -c configs/examples/tic_tac_toe_multi.yaml --set run.name=ttt-league
```

## What a run writes

```
runs/<name>/
  config.resolved.yaml      the config after --set overrides and agent merging
  logs/main.log             main process (also printed to the console)
  logs/learner-<agent>.log  one file per learner
  logs/worker-<i>.log       one file per rollout worker (worker-<i>-env<k>.log for subprocess envs)
  metrics.jsonl             one JSON record per line (below)
  ratings.json              latest ELO / win-rate matrix / wr_vs_past, rewritten periodically and at the end
  checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
```

`metrics.jsonl` records (`"kind"` field):

| kind | content |
|---|---|
| `train` | APPO metrics of one agent (`agent`, `train_step`, losses, entropy, `approx_kl`, `explained_variance`, `grad_norm`, `rho_mean`, `rho_clip_frac`, `policy_lag_mean/max`, `lr`, ...) every `metrics.log_interval` train steps |
| `episodes` | per agent since the previous record: `episodes`, `return_mean`, `length_mean`, `wdl` = W/D/L against `latest` (self), `past` (own checkpoints) and `arena` (other agents), `seat_counts` |
| `ratings` | `elo`, `win_rates`, `games`, `wr_vs_past` |
| `system` | `env_steps`, `env_steps_per_sec`, `train_steps_per_sec` per agent, learner `queue_depths`, `parked_buffers`, `workers_reporting` |

`meta.json` of a checkpoint holds the agent id, policy version, time, `config_hash`, `env_steps`, `final`, and the
agent's `networks` section, so `colosseum eval` can rebuild any checkpoint's architecture.

WandB is optional (`pip install -e ".[wandb]"`, `metrics.use_wandb: true`). It is a viewer: one run, each agent on its own
`<agent>/train_step` axis, ratings and system metrics on `env_steps`. `metrics.jsonl` stays the source of truth.

## Process lifecycle

- `training.total_timesteps` is the global number of env steps over all workers. When it is reached, every learner
  sends a final checkpoint, the main process saves it, all processes stop, and the exit code is 0.
- If any worker or learner dies, training stops with exit code 1 and a message like
  `worker-0 died (exit -9), see runs/<name>/logs/worker-0.log`.
- Ctrl-C (SIGINT) and SIGTERM stop all child processes within 10 seconds (final checkpoints are saved);
  the exit codes are 130 and 143. An invalid config exits with code 1 and a one-line `Config error: ...`.

## Writing your own game

### 1. Environment

```python
# my_game/env.py
import gymnasium, numpy as np
from colosseum.envs.base_env import BaseEnv

class MyGameEnv(BaseEnv):
    @property
    def num_players(self) -> int: return 2
    @property
    def observation_space(self): return gymnasium.spaces.Box(0, 1, (16,), np.float32)
    @property
    def action_space(self): return gymnasium.spaces.Discrete(4)   # Dict / Tuple / MultiDiscrete also work

    def reset(self, seed=None):
        obs = {p: np.zeros(16, np.float32) for p in range(2)}
        infos = {p: {"active": p == 0, "action_mask": np.ones(4, bool)} for p in range(2)}
        return obs, infos

    def step(self, actions):            # actions: {player: action}
        ...
        return obs, rewards, terminated, truncated, infos   # every value is a dict keyed by player
```

Per-player `info` keys the framework understands:
- `active` (bool): whose move it is. Only active players run inference and record transitions.
  Rewards that arrive while a player waits are credited to its last move. Without `active`, every player acts every step.
- `action_mask` (bool array): legal actions, applied in training, evaluation and BC. An active player whose mask has
  no legal action is an error.
- `outcome` (float in [0, 1]) or `rank` (int, 1 = best) in the terminal info: the authoritative match result used by
  ratings and eval. Without them, the outcome is derived from episode rewards.

`terminated` ends an episode; `truncated` (time limit) bootstraps with the value of the final observation.

### 2. Model

The default model is `encoder → core → policy head / value head`:

```python
# my_game/networks.py
import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

class Encoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
    @property
    def latent_dim(self): return 64
    def forward(self, obs): return self.net(obs)

class Policy(BasePolicy):
    def __init__(self, in_dim: int = 64, **kwargs):     # in_dim = the core's output size
        super().__init__(); self.net = nn.Linear(in_dim, 4)
    def forward(self, features): return CategoricalDist(logits=self.net(features))

class Value(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs):
        super().__init__(); self.net = nn.Linear(in_dim, 1)
    def forward(self, features): return self.net(features).squeeze(-1)
```

Cores (`colosseum.networks.cores`):
- `null`: no memory;
- `LSTMCore` / `GRUCore` (`hidden_size`, `num_layers`);
- `WindowAttentionCore` (`d_model`, `window`, `num_heads`, `num_layers`): causal attention over the last `window`
  latents of the episode.

For anything else, subclass `colosseum.networks.model.PolicyModel` (`initial_state`, `step`, optionally `unroll`) and
set `networks.model_class`.

### 3. Config

```yaml
# my_game/config.yaml
run: {name: null, dir: runs}            # outputs go to runs/<name>/ (default name: <config stem>-<timestamp>)
env: {env_class: my_game.env.MyGameEnv, num_players: 2}
networks:
  encoder_class: my_game.networks.Encoder
  core: {class: colosseum.networks.cores.LSTMCore, kwargs: {hidden_size: 128}}   # or null
  policy_class: my_game.networks.Policy
  value_class: my_game.networks.Value
training: {phase: self_play, total_timesteps: 2000000}
```

```bash
colosseum validate -c my_game/config.yaml     # schema, env num_players, a dummy step/unroll of the model
colosseum train -c my_game/config.yaml
```

The current directory is put on `sys.path`, so run the commands from the directory that contains `my_game/`.

## Configuration reference

Unknown keys are errors at every level. `--set key=value` accepts YAML values (`null`, numbers such as `1e-4`, lists,
`{}`) and works for `train`, `validate`, `run-learner` and `run-workers`:

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=lr-test \
  --set algorithm.learning_rate=1e-3 --set rollout.num_workers=4 --set training.resume_from=null
```

| section | keys (defaults) |
|---|---|
| `run` | `name` (null → `<config stem>-<YYYYmmdd-HHMMSS>`), `dir` (`runs`) |
| `env` | `env_class`, `num_players` (2; must equal the env's), `kwargs` |
| `networks` | `model_class` (null) or `encoder_class` + `core` (`{class, kwargs}` or null) + `policy_class` + `value_class`; `kwargs` (passed to every constructor) |
| `algorithm` | `algorithm_class` (APPO), `gamma` 0.99, `vtrace_lambda` 1.0, `vtrace_rho_bar` 1.0, `vtrace_c_bar` 1.0, `eps_clip` 0.2, `value_loss_coeff` 0.5, `entropy_coeff` 0.01, `max_grad_norm` 0.5, `num_epochs` 1, `minibatch_chunks` 0, `learning_rate` 3e-4, `lr_schedule` (`linear`; follows the share of `total_timesteps` done), `normalize_advantages`, `use_amp`, `amp_dtype` |
| `rollout` | `chunk_length` 256, `num_workers` 4, `envs_per_worker` 8, `torch_threads` 1, `weight_sync_interval_sec` 5, `vec_env` (`sync`/`subprocess`), `subproc_workers`, `match_refresh_interval_sec` 30 |
| `learner` | `device` (`auto`), `batch_chunks` 16 (every update uses exactly this many chunks), `queue_size` 64, `weight_push_interval` 5, `torch_threads` (auto), `pin_memory` |
| `training` | `phase` (`self_play` / `league`), `total_timesteps` (global env steps), `seed`, `resume_from`, `kickstart_teacher`, `kickstart_lambda`, `kickstart_decay_steps`, `kickstart_kl` (`forward` = KL(teacher‖student)) |
| `self_play` | `checkpoint_interval` (train steps), `pool_size` (FIFO per agent), `latest_prob`, `self_play_ratio` (league: share of self-play matches), `pfsp_exponent`, `shuffle_seats` (true) |
| `checkpoint` | `save_optimizer` (true: also save `trainer_state.pt`) |
| `metrics` | `log_interval` (train steps between `train` records), `console_interval_sec` 10, `use_wandb`, `wandb_project`, `wandb_entity` |
| `agents` | `{agent_id: {networks: {...}, algorithm: {...}, learner: {...}}}`: partial overrides, deep-merged onto the global sections |

**Agents.** Without an `agents` section there is one agent, `agent_0`. With it, every key is a trainable agent with its
own learner; an override changes only the keys it names:

```yaml
agents:
  alpha: {}
  beta:
    algorithm: {learning_rate: 1.0e-4}                       # every other algorithm key stays global
    networks: {core: {class: colosseum.networks.cores.GRUCore, kwargs: {hidden_size: 64}}}
```

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index, rotating at every match refresh.
- `self_play`: the owner's latest weights in every seat, except that each non-owner seat plays a random own checkpoint
  (not collecting data) with probability `1 - latest_prob`.
- `league`: with probability `self_play_ratio` the match is a self-play match. Otherwise it is an arena match: the owner
  plus N−1 opponents drawn by PFSP, `(1 - win_rate)^pfsp_exponent`, from the other agents, with replacement; every
  arena seat collects data.

Seats are shuffled. Ratings: every pair of seats of different agents scores 1 / 0.5 / 0 by outcome. The win-rate matrix
and ELO (K scaled by 1/(N−1)) use those pairs. `wr_vs_past` is the latest weights' score against the agent's own
checkpoints over the last 500 pairs.

**Resume.** `training.resume_from` accepts:
- a checkpoint dir (weights + optimizer + versions);
- a previous run dir (each agent takes its latest checkpoint there);
- a `.pt` state dict (weights only, e.g. the output of `colosseum bc`). It is applied to every agent and must match
  each agent's architecture.

## Behavioural cloning and kickstarting

```bash
colosseum bc -c my_game/config.yaml --data path/to/demos/ --output bc.pt --epochs 20
colosseum train -c my_game/config.yaml --set training.resume_from=bc.pt
```

BC data: `.pt` files with `observations`, `actions` and optional `action_masks` and `dones`. The loss is `-log_prob` of
the recorded action for every distribution type. Masks are applied. Stateful models train on sequences of `--seq-len`
steps (default 64) that reset at `dones`.

Kickstarting adds `lambda * KL(teacher‖student)` to the RL loss, decaying linearly over `kickstart_decay_steps`:
`--set training.kickstart_teacher=bc.pt`. The teacher uses the student's architecture.

## Distributed mode (limited)

The roles run as separate processes, possibly on separate machines:

```bash
colosseum serve-weight-store --port 50051                                              # machine A
colosseum run-learner -c cfg.yaml --agent agent_0 --traj-port 50052 --weight-store A:50051   # machine B
colosseum run-workers -c cfg.yaml --weight-store A:50051 -l agent_0=B:50052                  # machines C, D, ...
```

`run-learner` writes `runs/<name>-learner-<agent>/` (logs, checkpoints). `run-workers` writes `runs/<name>-workers/`.

Limitations in SP1 (to be fixed in SP5):
- there is no coordinator: workers play the latest weights of every agent in a fixed round-robin, with no historical
  opponents, no league, no ratings and no `metrics.jsonl` episode data;
- every `run-workers` host collects `total_timesteps / num_workers` per worker on its own, and the learner's progress is
  `consumed_samples / total_timesteps`;
- no fault tolerance or authentication;
- `deployment/` (Docker, Kubernetes) is untested.

## Status and limitations

SP1 ("foundation and stabilization") is the first of six sub-projects that follow the full review in
[`review/README.md`](review/README.md):

| | Sub-project | Content |
|---|---|---|
| ✔ | SP1 Foundation and stabilization | correct single-machine training, stateful model protocol, run dir, metrics, lifecycle |
| | SP2 Game model | `GameSpec`/multi-agent env API, player elimination, teams, roles, per-unit actions, Dict observations |
| | SP3 Players, league, warm start | scripted, frozen and external players; PFSP over snapshots; per-agent `init`/kickstart/critic warm-up; top-k snapshot storage |
| | SP4 Selection and observability | match log, OpenSkill / Bradley–Terry ratings, `colosseum tournament`, dashboard, snapshot ratings |
| | SP5 Distributed | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| | SP6 Speed and extensions | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

Known limitations today:
- **Game model:** symmetric players with identical observation and action spaces; a match ends for everyone at once
  (no elimination); no teams or roles.
- **League:** only trainable agents. PFSP picks among agents' latest weights, not snapshots. Checkpoints are a FIFO pool.
  Online ELO is a progress indicator, not a selection-grade rating.
- **Speed:** recurrent and attention cores unroll step by step on the learner. Worker inference is CPU only.
- **Distributed:** see above.

## Tests

```bash
.venv/bin/python -m pytest -m "not gpu and not slow" -q     # full fast suite (CI)
.venv/bin/python -m pytest -m slow -v                       # tic-tac-toe learning test (~3-5 min)
.venv/bin/python -m pytest -m gpu -v                        # CUDA machine only, see docs/GPU_CHECKS.md
```

Layout:
- `tests/unit`: pure functions and classes;
- `tests/contract`: the real worker loop feeding the real APPO, in-process;
- `tests/integration`: `colosseum train` subprocess runs;
- `tests/learning`: does it learn.

Tests write only under pytest's `tmp_path`. Throughput measurements are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Project structure

```
src/colosseum/
  cli.py launcher.py distributed.py eval.py
  core/         config, types, registry, action_spec, outcomes, errors, ipc, run_dir
  networks/     model (PolicyModel, act), composed, cores, state, distributions, normalization, base
  algorithms/   appo, vtrace, base
  worker/       rollout_loop (RolloutLoop), slots, rollout_worker (process wrapper)
  learner/      learner process, checkpoint payloads
  coordinator/  coordinator, matchmaker, ratings, checkpoint_manager, agent_pool
  metrics/      jsonl, aggregator, console, hub, wandb_logger
  envs/         base_env, vec_env, subproc_vec_env
  bc/ transport/ weight_store/ utils/
examples/       tic_tac_toe, composite_action (chase), space_miners (needs Box2D)
configs/examples/
```
````

- [ ] **Step 3: Update `CLAUDE.md`**

1. In `### Eval Mode`, replace the line

```
- Command: `colosseum eval --agents A_ckpt_100 B_ckpt_200 --num_matches 1000 --env my_game`
```

with

```
- Command: `colosseum eval -c config.yaml -a A=runs/x/checkpoints/agent_0/ckpt_v100 -a B=runs/x/checkpoints/agent_0/ckpt_v200 --num-matches 1000 --output result.json` (`--num-matches` is per pair; the architecture comes from each checkpoint's `meta.json`)
```

2. Under `**Behavioral Cloning:**`, replace the line

```
- Online BC (kickstarting): `loss = RL_loss + λ * KL(policy || BC_policy)`, λ decays over training
```

with

```
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_policy || policy)` (forward KL by default, `training.kickstart_kl`), λ decays over training
```

3. Get the test counts:

```bash
.venv/bin/python -m pytest -m "not gpu and not slow" --collect-only -q | tail -1
.venv/bin/python -m pytest -m slow --collect-only -q | tail -1
.venv/bin/python -m pytest -m gpu --collect-only -q | tail -1
```

4. Replace everything from the line `## Implementation Status` up to the line before `## Tech Stack` with the text below. Write the three numbers printed in sub-step 3 into the first bullet, in place of the bold markers `FAST`, `SLOW` and `GPU`.

```markdown
## Implementation Status

State after SP1 (foundation and stabilization, branch `sp1-stabilization`). Test suite: **FAST** fast tests, **SLOW** slow, **GPU** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite). The full review that motivated SP1 is `review/README.md`; the SP1 spec is `docs/superpowers/specs/2026-10-08-sp1-stabilization-design.md`.

### Works (single machine)
- **Training loop:** IMPALA-style. Workers (CPU inference, `RolloutLoop`) → per-agent learners (APPO + V-trace with `vtrace_lambda`) → newest-wins weight queues.
  - Only numpy payloads cross process boundaries.
  - Every learner update uses exactly `learner.batch_chunks` chunks.
  - `training.total_timesteps` is a global env-step budget; the LR follows its progress.
- **Models:** `PolicyModel` protocol with an opaque `State` pytree.
  - `ComposedModel(encoder, core, policy, value)` with cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`.
  - Monolithic models are possible via `networks.model_class`.
  - The learner reproduces the worker's log-probs and values for all four cores (contract tests).
- **Transitions:**
  - Agent-owned buffers parked across match changes.
  - Transitions stay open until the slot acts again; there is no extra bootstrap forward.
  - Turn-based games via `info["active"]`.
  - Truncation adds `γ·V(final_obs)`.
  - Action masks everywhere.
  - `ActionSpec` keeps natural component order.
- **League:**
  - Every trainable agent owns envs in rotation.
  - Self-play against own checkpoints; league with N-player PFSP arenas; shuffled seats.
  - Pairwise per-seat ratings (ELO with K/(N−1), fractional win-rate matrix, `wr_vs_past`).
  - FIFO checkpoint pool, atomic and confined to the run dir.
  - Final checkpoint on every stop.
  - Resume from a checkpoint dir, a run dir or a `.pt`.
- **Observability:** `runs/<name>/` with `config.resolved.yaml`, per-process logs, `metrics.jsonl` (train / episodes / ratings / system), `ratings.json`, checkpoints; console progress per agent; optional WandB with per-agent step axes.
- **Config:**
  - Pydantic with `extra="forbid"`.
  - Partial agent overrides deep-merged onto the global sections.
  - `--set` with YAML values for `train` / `validate` / `run-learner` / `run-workers`.
  - `validate` checks `env.num_players` and the model protocol.
- **Lifecycle:**
  - Exit codes 0 / 1 / 130 / 143.
  - Children ignore SIGINT and die with the parent.
  - A dead child stops the run with a pointer to its log.
  - SIGTERM leaves no process alive after 10 s.
- **Eval:** `PolicyModel`-based with per-seat state and seat rotation; W/D/L with Wilson CIs, per-seat breakdown, solo mode, JSON output; architecture taken from checkpoint `meta.json`.
- **BC / kickstart:** distribution-aware `-log_prob` loss with masks, sequence training for stateful models, forward-KL kickstarting.

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) works for latest-weights self-play only: no coordinator, league, ratings or worker metrics; per-worker budgets.
- **`deployment/`** (Docker, K8s) is not tested.

### Not implemented (see Roadmap)
- Scripted, frozen and external players.
- Asymmetric or team games, player elimination, Dict observations.
- Snapshot-level PFSP, OpenSkill / Bradley–Terry ratings, a tournament command.
- Off-policy algorithms (R2D2/DQN), SAC, AlphaZero/MuZero.
- GPU inference on workers.

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, variable unit counts, per-unit actions, Dict observations, bootstrap moved to the learner |
| **SP3 Players, league, warm start** | scripted / frozen / external players, PFSP over snapshots, per-agent warm start (`init`, kickstart, critic warm-up), top-k snapshot storage |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

```

- [ ] **Step 4: Write `docs/GPU_CHECKS.md`**

1. List the GPU tests:

```bash
.venv/bin/python -m pytest -m gpu --collect-only -q | grep "::"
```

2. Create `docs/GPU_CHECKS.md` with the text below. Fill the table with one row per node id printed by the command above: the node id in the first column, and in the second column which of the listed areas the test checks, in one line taken from its docstring.

````markdown
# Проверки на GPU

На машине разработки нет GPU, поэтому всё, что требует CUDA, помечено маркером `gpu`. Без CUDA такие тесты пропускаются (`tests/conftest.py`). Обычный прогон `pytest -m "not gpu and not slow"` их не запускает.

## Как запустить

На машине с NVIDIA GPU и драйвером CUDA:

```bash
scripts/setup-dev.sh --gpu          # ставит сборку torch с CUDA вместо CPU-сборки
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # должно напечатать True
.venv/bin/python -m pytest -m gpu -v
```

Все тесты из таблицы должны пройти (`passed`, не `skipped`).

## Что проверяется

- AMP: обучение APPO в `float16` (с GradScaler) и `bfloat16`, без NaN и с уменьшением лосса;
- `learner.pin_memory`: батч закрепляется в памяти и переносится на устройство с `non_blocking=True`;
- перенос модели и состояния `State` (LSTM, GRU, окно внимания) между CPU и CUDA: `state_to`, `initial_state(device=...)`;
- kickstart на CUDA: учитель и студент на одном устройстве, маски на устройстве;
- BC на CUDA, включая обучение по последовательностям для модели с состоянием.

## Список тестов

| Тест | Что проверяет |
|---|---|
````

- [ ] **Step 5: Add the "after" benchmark**

1. Open `docs/benchmarks.md` and find the exact benchmark command recorded in the «до» section (T0.4). Run that same command on the same machine, with the machine otherwise idle: check with `uptime` that the load average is below 1.
2. Append this section to `docs/benchmarks.md`. Fill the table with the numbers the script prints, and the date with the day of the run:

```markdown
## После SP1

Та же машина и та же команда, что в разделе «до». Код — ветка `sp1-stabilization` после T8.2.

| Воркеры | Апдейты/с | Шаги сред/с |
|---|---|---|
| 1 | | |
| 2 | | |
| 4 | | |

Критерий спеки §3.6: throughput растёт монотонно от 1 к 2 и к 4 воркерам. Основные причины роста по сравнению с «до»:
- ограничение потоков torch (`rollout.torch_threads=1` и авто-значение для лёрнера);
- отсутствие лишнего bootstrap-forward;
- веса newest-wins;
- обучение ровно на `batch_chunks` чанках.
```

   If 1 → 2 → 4 is not monotonic, do not hide it: write the numbers as measured and stop at this step. Report to the user before T8.4, because spec §3.6 is then not met.

- [ ] **Step 6: Check README commands and links**

Run:

```bash
.venv/bin/python - <<'EOF'
import re, pathlib
text = pathlib.Path("README.md").read_text()
for link in re.findall(r"\]\(([^)#]+)\)", text):
    assert pathlib.Path(link).exists(), link
print("links ok")
EOF
.venv/bin/python -m pytest tests/integration/test_lifecycle.py::test_readme_quickstart_from_repo_root -v
```

Expected: `links ok` and PASS. T8.4 runs every other README command for real.

- [ ] **Step 7: Ignore run outputs**

The README quickstart writes `runs/` and `eval.json` into the repo root. Make sure neither can be committed:

```bash
touch .gitignore
grep -qxF "runs/" .gitignore || echo "runs/" >> .gitignore
grep -qxF "eval.json" .gitignore || echo "eval.json" >> .gitignore
```

- [ ] **Step 8: Commit**

```bash
git add README.md CLAUDE.md docs/GPU_CHECKS.md docs/benchmarks.md .gitignore
git commit -m "docs: README and CLAUDE.md match SP1 reality; roadmap, GPU checks, benchmarks after"
```

---
### Task T8.4: Acceptance against spec §3, then merge

Each step checks one criterion of spec §3 and gives the command and the expected observable result. Record the outcome of each step (pass/fail, numbers, paths) in the final message to the user.
- If any step fails, stop and report. Do not merge.
- The merge (Step 10) happens only after the user explicitly accepts the report.

**Files:** none are changed, except `docs/benchmarks.md` or the docs if a step reveals a documentation mismatch: fix it, commit with `docs:`, and rerun that step.

**Interfaces:**
- Consumes: everything in Parts A–D.
- Produces: the acceptance report; after acceptance, `main` containing SP1.

- [ ] **Step 1: §3.1 Tests: full fast suite, one process, no hangs, nothing written outside `tmp_path`**

```bash
git status --porcelain                       # must be empty before starting
timeout 2400 .venv/bin/python -m pytest -m "not gpu and not slow" -q -p no:cacheprovider; echo "exit=$?"
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache" || echo "clean"
```

Expected:
- `exit=0`, with all tests passed. A timeout would print `exit=124`.
- The status check prints `clean`. In particular no `runs/`, `checkpoints/` or `*.jsonl` appear in the repo.

Also run `.venv/bin/ruff check .` and expect no findings: CI runs it (T0.3).

- [ ] **Step 2: §3.2 Contract tests**

```bash
.venv/bin/python -m pytest tests/contract tests/unit/test_league_matchmaking.py tests/integration/test_league_runs.py -v -rA | tee /tmp/sp1-contract.txt | tail -40
```

Expected: everything passes. In `/tmp/sp1-contract.txt`, identify the test that covers each bullet of §3.2 and list its node id in the report:

| §3.2 bullet | Produced by |
|---|---|
| learner reproduces worker log-prob and value for no-memory, LSTM, GRU, window-attention cores | T1.6 contract test (`tests/contract`) |
| self-play with two agents | `tests/integration/test_league_runs.py::test_two_agent_self_play_reaches_budget_and_both_learners_train` |
| league on three agents, all pairs meet | `tests/integration/test_league_runs.py::test_three_agent_league_all_pairs_meet_and_seats_balanced` |
| turn-based: the loser gets −1 on its last transition, done flags on all collecting seats | T3.2 contract test (`tests/contract`) |
| action order with 12 `MultiDiscrete`/`Tuple` components | T4.5 test |
| seat balance ±5% | `tests/unit/test_league_matchmaking.py::test_seat_distribution_balanced_within_5_percent` |
| eval of a stateful model | T7.1 test |

A bullet without a passing test fails this criterion.

- [ ] **Step 3: §3.3 Learning**

```bash
.venv/bin/python -m pytest tests/learning -m "not slow" -v --durations=0
nproc; time .venv/bin/python -m pytest tests/learning -m slow -v -s
```

Expected:
- the bandit and chain tests pass, each in under 20 s (`--durations`);
- `nproc` prints 8 on the reference machine;
- the slow test prints `vs random: {'win_rate': ...}` with `win_rate >= 0.80`, and the `real` time is about 3–5 minutes.

- [ ] **Step 4: §3.4 Visibility without WandB**

```bash
rm -rf /tmp/sp1-acceptance
.venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml --set run.dir=/tmp/sp1-acceptance \
  --set run.name=vis --set training.total_timesteps=150000 2>&1 | tee /tmp/sp1-vis.txt | grep -E "Run directory|\[agent_0\] step" | head
ls /tmp/sp1-acceptance/vis /tmp/sp1-acceptance/vis/logs /tmp/sp1-acceptance/vis/checkpoints/agent_0
.venv/bin/python -c "import json,collections; print(collections.Counter(json.loads(l)['kind'] for l in open('/tmp/sp1-acceptance/vis/metrics.jsonl')))"
cat /tmp/sp1-acceptance/vis/ratings.json | head -20
```

Expected:
- `Run directory: /tmp/sp1-acceptance/vis`, followed by `[agent_0] step ...` lines about every 10 s;
- the run dir lists `config.resolved.yaml`, `logs`, `metrics.jsonl`, `ratings.json` and `checkpoints`;
- `logs` lists `main.log`, `learner-agent_0.log`, `worker-0.log` and `worker-1.log`;
- `checkpoints/agent_0` holds `ckpt_v*` dirs;
- the Counter shows all four kinds: `train`, `episodes`, `ratings`, `system`.

- [ ] **Step 5: §3.5 Lifecycle**

```bash
.venv/bin/python -m pytest tests/integration/test_lifecycle.py tests/integration/test_league_runs.py::test_resume_continues_versions_env_steps_and_lr -v
```

Expected: all pass. These cover:
- the budget is respected and a final checkpoint is written, exit 0;
- a killed worker gives exit 1 and a message with the log path;
- SIGTERM leaves no child alive after ≤ 10 s, including subprocess-env grandchildren;
- SIGINT gives 130;
- a config error gives exit 1.

Manual check of a dying **learner**; the test covers a worker:

```bash
rm -rf /tmp/sp1-acceptance/kill
.venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml --set run.dir=/tmp/sp1-acceptance --set run.name=kill \
  --set training.total_timesteps=100000000 > /tmp/sp1-kill.txt 2>&1 &
sleep 30; kill -9 $(grep -o "learner-agent_0 started (pid [0-9]*" /tmp/sp1-acceptance/kill/logs/learner-agent_0.log | grep -o "[0-9]*$")
wait $!; echo "exit=$?"; grep "died" /tmp/sp1-kill.txt
pgrep -f "multiprocessing.spawn" || echo "no leftover children"
```

Expected:
- `exit=1`;
- `learner-agent_0 died (exit -9), see /tmp/sp1-acceptance/kill/logs/learner-agent_0.log`;
- `no leftover children` (unless unrelated Python multiprocessing jobs run on the machine; check their parents with `ps -o ppid= -p <pid>`).

- [ ] **Step 6: §3.6 Speed**

```bash
sed -n '/## После SP1/,$p' docs/benchmarks.md
```

Expected:
- the table from T8.3 Step 5 shows updates/s and env steps/s strictly increasing from 1 to 2 to 4 workers;
- the «до» section exists with the same command.

- [ ] **Step 7: §3.7 Documentation works verbatim**

Run the README Quick start block exactly as written, from the repo root, in a fresh shell:

```bash
source .venv/bin/activate
colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
NEW=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
OLD=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | head -1)
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-continued \
  --set training.resume_from=runs/ttt-quickstart --set training.total_timesteps=1200000
colosseum train -c configs/examples/tic_tac_toe_multi.yaml --set run.name=ttt-league
```

Expected:
- every command exits 0;
- `eval.json` exists and the printed table has W/D/L, win rate with CI, and a per-seat breakdown for the pair;
- the continued run's first checkpoint version is above the quickstart's last.

Smoke-test the distributed commands on localhost, with the README's `cfg.yaml`, `A` and `B` substituted:

```bash
colosseum serve-weight-store --port 50051 & WS=$!
colosseum run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0 --traj-port 50052 --weight-store localhost:50051 \
  --set run.dir=/tmp/sp1-acceptance --set training.total_timesteps=20000 & LR=$!
sleep 10
colosseum run-workers -c configs/examples/tic_tac_toe.yaml --weight-store localhost:50051 -l agent_0=localhost:50052 \
  --set run.dir=/tmp/sp1-acceptance --set training.total_timesteps=20000; echo "workers exit=$?"
kill $LR $WS; wait
ls /tmp/sp1-acceptance | grep -E "learner-agent_0|workers"
```

Expected:
- `workers exit=0`;
- two run dirs, `tic_tac_toe-<ts>-learner-agent_0` and `tic_tac_toe-<ts>-workers`, each with `logs/`;
- the learner's run dir has checkpoints.

For the README sections with user placeholders (`my_game/...`, `path/to/demos/`), compare every flag with `colosseum <command> --help`. Check that `README.md` and `CLAUDE.md` (Implementation Status, Roadmap) describe what Steps 1–6 observed.

Clean up:

```bash
rm -rf runs eval.json /tmp/sp1-acceptance /tmp/sp1-*.txt
```

- [ ] **Step 8: §3.8 GPU**

```bash
.venv/bin/python -m pytest -m gpu --collect-only -q | grep "::" | sort > /tmp/gpu-tests.txt
grep -oE 'tests/[^ |`]+::[^ |`]+' docs/GPU_CHECKS.md | sort | diff - /tmp/gpu-tests.txt && echo "GPU list matches"
.venv/bin/python -m pytest -m gpu -q
```

Expected:
- `GPU list matches`;
- without CUDA, the last command reports every GPU test as `skipped`, none failed.

Tell the user that these tests still have to be run on a CUDA machine, with the command from `docs/GPU_CHECKS.md`.

- [ ] **Step 9: Report and ask for acceptance**

Send the user:
- the per-criterion results of Steps 1–8, with numbers: test counts, the learning win rate and time, the benchmark table;
- any deviations;
- the question "Accept SP1 and merge `sp1-stabilization` into `main`?"

Wait for an explicit yes.

- [ ] **Step 10: Merge and push (only after the user accepts)**

Use `--no-ff`. It keeps SP1 visible as one unit in `main`'s history and lets the whole sub-project be reverted with a single `git revert -m 1 <merge>`.

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff sp1-stabilization -m "Merge branch 'sp1-stabilization': SP1 foundation and stabilization"
.venv/bin/python -m pytest -m "not gpu and not slow" -q
git push origin main
```

Expected:
- the merge completes without conflicts. If `main` moved and conflicts appear, stop and ask the user;
- the suite passes on the merge commit;
- the push succeeds.

---
## Contract notes

The tasks above are written against the changes proposed here. Each change is small and keeps existing contract names. The plan owner should fold them into the contract section of `00-overview.md` before execution. Otherwise, the task that introduces a change updates that section in its own commit and says so in the commit message, as the overview requires.

1. **`CheckpointConfig.dir` is removed in T6.2, not T6.1.** The contract annotates it as "T5.3/T6.1". Removing it in T6.1 would need an interim checkpoint location before `RunDir` exists (T6.2). T6.1 adds `RunConfig` / `ColosseumConfig.run`; T6.2 removes `checkpoint.dir` together with introducing the run dir.

2. **`CheckpointManager.save(..., trainer_state: dict | bytes | None, ...)`.** The contract types it as `dict | None`.
   - The learner serializes `BaseAlgorithm.state_dict()` with `torch.save` into bytes, because that state contains tensors (optimizer moments), and the global constraint forbids tensors on `mp.Queue`. AMP bf16 tensors could not go through numpy either.
   - `save` writes such bytes verbatim; a dict is still accepted.
   - The checkpoint payload format is pinned in T5.3 as `make_checkpoint_payload` → `{agent_id, policy_version, model_state (numpy), trainer_state_bytes, final}`. The contract left it unspecified ("checkpoints … only in payload form", T2.2).

3. **`resolve_resume` returns** `{"model_state": dict[str, np.ndarray], "trainer_state": bytes | None, "policy_version": int, "env_steps": int, "source": str}`.
   - `trainer_state` is the raw content of `trainer_state.pt`, so the dict can be passed to the learner process without tensors.
   - `env_steps` lets the main process continue the global `SharedCounter`, so the LR schedule (driven by `progress`) and the budget continue after a resume. Spec §6: "resume continues versions and LR".
   - `source` is used in error messages.
   - The learner applies the dict with the new `apply_resume_state(algorithm, resume_state)`. When there is no trainer state, it restores `policy_version` through `algorithm.state_dict()` / `load_state_dict()`, which relies on the key `"policy_version"` named in the contract comment.

4. **`Coordinator.ratings_snapshot()`** returns the contract keys `elo`, `win_rates`, `wr_vs_past`, plus `games` (pairwise pair counts) and `past_games`. T5.4 needs `games` to prove "all pairs meet"; the console uses it for the arena win rate.

5. **`EloRating.update_pairs(pairs, k_scale)`** is added next to the contract's `update_pair`. The coordinator applies all pairs of one match simultaneously from the pre-match ratings, so ties in FFA are order-independent (T5.2 test). `update_pair` remains and delegates to it.

6. **Coordinator additions:**
   - `Coordinator.__init__(config, checkpoint_dir)`: required from T6.2; optional with a fallback to `config.checkpoint.dir` in T5.1–T6.1;
   - `next_round()` and the `refresh_round` property;
   - `save_checkpoint_payload(payload, meta_extra)`, which replaces `maybe_save_checkpoint`; the learner decides when to snapshot;
   - the `past_win_rate` property;
   - the coordinator registers every trainable agent from the config itself.

7. **`RunDir.create(config, config_path=None, role=None)`.** The extra `role` gives the distributed roles their own dirs (`<name>-learner-<agent>`, `<name>-workers`). `RunDir` also gets `resolved_config_path` and `open(root)`. The contract attributes are provided as properties of a frozen dataclass.

8. **Process entry helpers:**
   - `colosseum.utils.process.run_child(name, log_dir, fn, *args, **kwargs)` (T6.2): logging first, the child signal policy (T6.5), crash tracebacks into the process log;
   - the env vars `COLOSSEUM_LOG_DIR` / `COLOSSEUM_PROCESS_NAME`, so subprocess-env grandchildren log to the run dir;
   - `ProcessSupervisor` and `SHUTDOWN_GRACE_SEC = 7.0`. The grace is below 10 s so terminate/kill fit into the "≤ 10 s after SIGTERM" criterion.

9. **Worker commands are never "newest-wins".**
   - `_COMMAND_QUEUE_SIZE = 1` and the main process uses `put_nowait`; a full queue means "skip this worker this round".
   - A replaced command would silently lose its `new_checkpoints` while the main process already counts them as delivered (spec: "marked as sent only after a successful put").
   - If T2.3 switched `WorkerCommand` delivery to `put_latest`, T5.3's `_refresh_worker_matches` reverts that for commands only. Weights stay newest-wins.

10. **`Launcher.launch() -> int` and `run_training(...) -> int`** return the process exit code (T6.5). `Launcher.__init__(config, run_dir)` (T6.2). The T2.5 `SharedCounter` is stored as `Launcher._env_counter`.

11. **The eval engine API is not pinned in the contract.** T8.2's slow test therefore evaluates tic-tac-toe vs a random legal-move player with the contract's `act()` (per-seat model state, action mask, greedy) in `tests/learning/ttt_eval.py`, instead of a T7 function. T8.4 Step 7 exercises the T7.2 `colosseum eval` CLI end to end.

12. **`rollout_worker_process` gets `stats_queue=None, stats_interval_sec=2.0`** (T6.3). Worker stats (`parked_buffers`, env steps, chunks sent) reach the main process for the `system` record. This assumes the T0.5/T2.5 wrapper runs `loop.run(should_stop=stop_event.is_set, ...)`, as stated in T6.3 Step 8.

### Assumptions about Parts A–C

Beyond the contract, the tasks above assume the following:

- **T0.x:**
  - `tests/conftest.py` sets `spawn`, registers the `gpu`/`slow` markers, and pytest-timeout is installed;
  - the repo root is importable, so `tests.helpers` and `tests.integration.cli_runner` resolve;
  - `scripts/setup-dev.sh` exists; the `examples` extra provides Box2D;
  - `docs/benchmarks.md` has a «до» section with its exact command (T0.4).
- **T1.4:**
  - `build_model` passes `in_dim=core.output_dim` to heads whose constructor has an `in_dim` parameter;
  - `CoreConfig` uses the alias `class` with `populate_by_name=True`;
  - `validate_config` raises `ConfigError`.
- **T2.x:**
  - learners and workers run until `stop_event`; the main process stops on the global budget;
  - learner metric dicts on `metrics_queue` carry `agent_id` and `train_step`;
  - worker chunk sends wait in short pieces and check `stop_event`, so they never block shutdown;
  - `LATEST_NETWORK_ID` is still exported by `colosseum.worker.rollout_worker`.
- **T3.4:** `results_queue` carries `MatchResult` objects with `seats`.
- **T4.6:**
  - `BaseAlgorithm.state_dict()` includes `"policy_version"`, and `load_state_dict(state_dict())` round-trips;
  - APPO reports the learning rate as `lr` (T5.4 also accepts `learning_rate`);
  - `learner_process` still contains an `if resume_state is not None:` block and a periodic checkpoint block, which T5.3 replaces.
- **T7.2:** `colosseum eval` accepts `-a name=path` (checkpoint dir), `--num-matches` (per pair) and `--output`.

# SP2 Plan — Part D: Demo environments, learning tests, units experiment, reference examples, docs, acceptance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** Spec block 10 and the acceptance of section 3:
- six demo games on the new contract, each with models, an example config and fast contract-level tests (T8.1, T8.2);
- fast learning tests (a `Units` bandit, a cooperative bandit) and the slow learning tests of criterion 3 (T8.3);
- the units experiment of criterion 4, throughput "after" (criterion 5) and the `global_state` size measurement (T8.4);
- `space_miners` and `chase` rewritten as reference examples (criterion 6, T8.5);
- documentation: `docs/ENV_GUIDE.md`, README, CLAUDE.md, `docs/GPU_CHECKS.md` (criteria 7, 9; T8.6);
- the acceptance run and report, then a stop for the owner's approval before any merge (T8.7).

**Read `00-overview.md` first** (global constraints, interface contract, contract amendments). This part runs after T7.3 (overlay), so it uses the **final module paths**: every `colosseum.sp2.X` of the contract is `colosseum.X` here, and the CLI is `colosseum ...` / `python -m colosseum ...`. Deviations and additions are listed in `## Contract notes` at the end; the contract amendments appended to the overview win over them.

**Execution order inside this part:** T8.1 → T8.2 → T8.3 → T8.4; T8.5 may run any time after T7.3; T8.6 after T8.4 and T8.5; T8.7 last. T8.2 reuses `tests/demo_checks.py` from T8.1 (see Contract notes).

**Conventions used in every task.**
- Run commands from the repository root with `.venv/bin/python`; "full fast suite" means `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with zero warnings), plus `.venv/bin/ruff check .`.
- Every task ends with a commit on `sp2-game-model` and `git push origin sp2-game-model`. Commit messages use conventional prefixes and carry no attribution lines.
- New support modules under `tests/` are imported by bare name (`from demo_checks import ...`). Their names are added to `[tool.ruff.lint.isort] known-first-party` in `pyproject.toml` in the task that creates them, so `ruff` sorts them with the first-party block.
- Demo games are pure numpy, deterministic per seed, and small: a few thousand env steps per second in one process. Their reference policies (`greedy_action`, `scripted_action`, `safe_action`, `chase_action`, `flee_action`) exist only to sanity-check that the game rewards what it claims to; training never uses them.
- Code in this part was prototyped outside the repository against a stub of the contract (`GameSpec`, `StepResult`, `Units` with the contract's fields): every env-level test below passed there. Calls into framework code (`MatchRunner`, `RolloutLoop`, `APPO`, `play_lineups`, `load_eval_model`, `RandomPolicy`, `UnitsHead`, `make_distribution`) are written against the contract only. If a contract name differs in the repository, follow "Plan code vs. current code" in the overview: keep the intent and the tests, adapt the call, say so in the commit body.

**Machine.** The reference machine is the SP1 one: WSL2, 8 cores, about 11 GB RAM, no GPU. Learning budgets and the units-experiment budget are sized for it.

---

### Task T8.1: Demo games `coin_grid`, `unit_harvest`, `tron` (+ models, configs, smoke)

Spec block 10 (demo envs), criterion 3 (the games behind the solo, units and FFA rows). Each game is written in full and exercises the contract features named in its docstring:
- `coin_grid`: `GameSpec.solo`, Dict observation with a `uint8` grid leaf and a `float32` vector leaf, action masks, an artificial step limit → `truncated=True` with `final_obs`;
- `unit_harvest`: two sides, simultaneous moves, `Dict` action = base `Discrete(2)` + workers `Units(max_units, Discrete(5))` (order kept with a list of pairs), units born (base builds) and killed (collisions), entity lists with masks, `max_units` as a constructor parameter for K=8 / K=128, a rule-based end at a fixed step with `Outcome.team_score`;
- `tron`: light cycles, layouts `2p`/`3p`/`4p` via `GameSpec.symmetric([2, 3, 4], ...)`, crash = elimination via `terminated` with the reward in the same step, ranks by elimination order with shared places, simultaneous moves, a `uint8` Box observation (no Dict, no masks).

**Files:**
- Create:
  - `examples/coin_grid/__init__.py`, `examples/coin_grid/game.py`, `examples/coin_grid/models.py`, `configs/examples/coin_grid.yaml`
  - `examples/unit_harvest/__init__.py`, `examples/unit_harvest/game.py`, `examples/unit_harvest/models.py`, `configs/examples/unit_harvest.yaml`
  - `examples/tron/__init__.py`, `examples/tron/game.py`, `examples/tron/models.py`, `configs/examples/tron.yaml`
  - `tests/demo_checks.py` (shared support for T8.1, T8.2, T8.5)
- Test:
  - `tests/contract/test_demo_coin_grid.py`
  - `tests/contract/test_demo_unit_harvest.py`
  - `tests/contract/test_demo_tron.py`
  - `tests/integration/test_demo_train_smoke.py`
- Modify: `pyproject.toml` (`known-first-party` += `demo_checks`; also `game_helpers`, `game_harness` if T1.4/T3.4 did not add them)

**Interfaces:**
- Consumes:
  - `colosseum.envs.game`: `GameSpec` (`solo`, `symmetric`, `teams`, `outcome_kind`, `layouts`, `roles`), `MultiAgentEnv`, `StepResult`, `Outcome`; `colosseum.envs.spaces.Units`
  - `colosseum.core.specs.ActionSpec` (`from_space`, `groups`, `has_units`, `num_deciders`)
  - `colosseum.networks.base`: `BaseEncoder`, `BasePolicy`, `BaseValue`, `EncoderOutput`; `colosseum.networks.heads`: `UnitsHead(group, in_dim, hidden)`, `make_distribution(spec, params)`
  - `colosseum.envs.vector.VectorEnv`, `colosseum.worker.match_runner.MatchRunner` (+ the `MatchObserver` methods), `colosseum.core.types`: `Lineup`, `SeatAssignment`, `MatchResult`
  - `colosseum.core.config.load_config(path, overrides)`, `colosseum.core.registry.validate_config(config)`
  - `game_helpers.RandomPolicy(role)` (T2.3), `cli_runner.run_train` (TINY overrides on the new schema, T7.1/T7.3)
  - `build_model` injection rule (T2.4): constructors get `observation_space`, `action_space`, `global_state_space`, `action_spec` by name; heads get `in_dim`
- Produces:
  - `examples.coin_grid.game`: `CoinGridGame(size=7, num_coins=5, max_steps=50)`, `MOVES`, `greedy_action(obs) -> int`
  - `examples.coin_grid.models`: `CoinGridEncoder`, `CoinGridPolicy`, `CoinGridValue`
  - `examples.unit_harvest.game`: `UnitHarvestGame(max_units=8, size=8, num_resources=6, initial_workers=2, build_cost=1, max_steps=50, deposit_reward=0.1)`, `UNIT_FEATURES`, `scripted_action(obs) -> dict`
  - `examples.unit_harvest.models`: `HarvestEncoder`, `HarvestPolicy`, `HarvestValue`, `masked_mean`
  - `examples.tron.game`: `TronGame(size=10, view_radius=3, players=(2, 3, 4))`, `safe_action(obs, rng) -> int`
  - `examples.tron.models`: `TronEncoder`, `TronPolicy`, `TronValue`
  - `tests/demo_checks.py`: `REPO_ROOT`, `example_config(name) -> Path`, `load_example(name, overrides=None)`, `check_example_config(name)`, `MatchLog`, `random_lineup(spec, layout)`, `random_matches(env_fn, *, min_episodes, num_envs=4, seed=0, max_env_steps=50_000) -> (list[MatchResult], MatchLog)`

- [ ] **Step 1: Shared demo checks**

Create `tests/demo_checks.py`:

```python
"""Shared checks for the demo games (T8.1, T8.2): example configs and random matches through
the real ``VectorEnv`` + ``MatchRunner`` (every contract check of ``EpisodeTracker`` included)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.registry import validate_config
from colosseum.core.types import Lineup, MatchResult, SeatAssignment
from colosseum.envs.game import GameSpec, MultiAgentEnv
from colosseum.envs.vector import VectorEnv
from colosseum.worker.match_runner import MatchRunner
from game_helpers import RandomPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]


def example_config(name: str) -> Path:
    return REPO_ROOT / "configs" / "examples" / f"{name}.yaml"


def load_example(name: str, overrides: dict[str, Any] | None = None) -> ColosseumConfig:
    return load_config(example_config(name), overrides)


def check_example_config(name: str) -> None:
    """``colosseum validate`` in-process: spec, matchmaking, a few random steps of every layout,
    ``step`` and ``unroll`` of every agent's model. Raises on any problem."""
    validate_config(load_example(name))


class _Pool:
    def __init__(self, models: dict) -> None:
        self._models = models

    def get(self, agent_id: str, network_id: str):
        return self._models.get(agent_id)


class MatchLog:
    """A ``MatchObserver`` that keeps every ``MatchResult`` and counts acts and eliminations."""

    def __init__(self) -> None:
        self.results: list[MatchResult] = []
        self.acts = 0
        self.terminated = 0

    def on_act(self, env: int, seat: int, record) -> None:
        self.acts += 1

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        pass

    def on_terminated(self, env: int, seats: list[int]) -> None:
        self.terminated += len(seats)

    def on_episode_end(self, env: int, end) -> None:
        self.results.append(end.result)

    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None:
        pass


def random_lineup(spec: GameSpec, layout: str) -> Lineup:
    """Every seat played by the random player of its role (model key ``random_<role>``)."""
    return Lineup(layout, [SeatAssignment(f"random_{seat.role}") for seat in spec.layouts[layout]])


def random_matches(env_fn: Callable[[], MultiAgentEnv], *, min_episodes: int, num_envs: int = 4, seed: int = 0,
                   max_env_steps: int = 50_000) -> tuple[list[MatchResult], MatchLog]:
    """Uniformly random legal play in every layout (env ``e`` plays layout ``e % L`` of the sorted
    layouts) until ``min_episodes`` episodes have ended."""
    vec = VectorEnv(env_fn, num_envs)
    spec = vec.spec
    layouts = sorted(spec.layouts)
    models = {f"random_{name}": RandomPolicy(role) for name, role in spec.roles.items()}
    log = MatchLog()
    runner = MatchRunner(vec_env=vec, lineups=[random_lineup(spec, layouts[e % len(layouts)]) for e in range(num_envs)],
                         models=_Pool(models), observer=log, seed=seed, context="demo")
    steps = 0
    try:
        while len(log.results) < min_episodes:
            steps += runner.step()
            assert steps < max_env_steps, f"only {len(log.results)} episodes after {steps} env steps"
    finally:
        runner.close()
    return log.results, log
```

Add `"demo_checks"` to `known-first-party` in `pyproject.toml` (keep the list sorted; also add `"game_harness"` and `"game_helpers"` if they are missing):

```bash
grep -n "known-first-party" pyproject.toml
```

Expected after the edit: the list contains `"demo_checks"`, `"game_harness"`, `"game_helpers"` next to the SP1 names.

- [ ] **Step 2: Write the failing tests for the three games**

Create `tests/contract/test_demo_coin_grid.py`:

```python
"""coin_grid demo game (T8.1): solo, Dict observation with a uint8 leaf, masks, truncation."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.coin_grid.game import CoinGridGame, greedy_action


def test_reset_gives_a_uint8_grid_and_a_float_vector():
    env = CoinGridGame(size=5, num_coins=3, max_steps=10)
    res = env.reset(0, "solo")
    obs = res.obs[0]
    assert res.acting == {0}
    assert list(obs) == ["grid", "vec"]
    assert obs["grid"].dtype == np.uint8 and obs["grid"].shape == (2, 5, 5)
    assert obs["grid"][0].sum() == 1 and obs["grid"][1].sum() == 3
    assert obs["vec"].dtype == np.float32 and obs["vec"][0] == 1.0
    assert env.spec.roles["player"].observation_space.contains(obs)


def test_moves_off_the_grid_are_masked():
    env = CoinGridGame(size=5)
    env.reset(0, "solo")
    env._pos = np.array([0, 0])
    assert env._mask().tolist() == [True, False, True, False, True]
    env._pos = np.array([4, 4])
    assert env._mask().tolist() == [True, True, False, True, False]


def test_a_collected_coin_pays_one_and_reappears_elsewhere():
    env = CoinGridGame(size=5, num_coins=1)
    env.reset(0, "solo")
    env._coins[:] = False
    env._pos = np.array([2, 2])
    env._coins[2, 3] = True
    res = env.step({0: 4})                                   # right
    assert res.rewards == {0: 1.0}
    assert env._coins.sum() == 1 and not env._coins[2, 3]


def test_the_step_limit_truncates_with_final_obs():
    env = CoinGridGame(size=5, num_coins=2, max_steps=4)
    res = env.reset(1, "solo")
    for _ in range(4):
        assert not res.episode_over
        res = env.step({0: 0})
    assert res.episode_over and res.truncated and res.acting == set()
    assert set(res.final_obs) == {0}
    assert res.final_obs[0]["grid"].dtype == np.uint8 and res.final_obs[0]["vec"][0] == 0.0


def test_the_same_seed_gives_the_same_episode():
    a, b = CoinGridGame().reset(7, "solo"), CoinGridGame().reset(7, "solo")
    assert np.array_equal(a.obs[0]["grid"], b.obs[0]["grid"])


def _mean_score(policy, episodes: int = 100) -> float:
    env, rng, total = CoinGridGame(), np.random.default_rng(0), 0.0
    for ep in range(episodes):
        res = env.reset(ep, "solo")
        while not res.episode_over:
            res = env.step({0: policy(res.obs[0], res.action_masks[0], rng)})
            total += res.rewards.get(0, 0.0)
    return total / episodes


def test_the_scripted_reference_collects_far_more_than_random_play():
    greedy = _mean_score(lambda obs, mask, rng: greedy_action(obs))
    random = _mean_score(lambda obs, mask, rng: int(rng.choice(np.flatnonzero(mask))))
    assert greedy >= 5 * random > 0


def test_the_example_config_validates():
    check_example_config("coin_grid")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(CoinGridGame, min_episodes=12)
    assert {r.layout for r in results} == {"solo"}
    for r in results:
        assert r.outcome_kind == "score" and len(r.teams) == 1 and r.episode_length == 50
        assert r.teams[0].score == pytest.approx(r.seats[0].reward)   # default: mean of the team's returns
```

Create `tests/contract/test_demo_unit_harvest.py`:

```python
"""unit_harvest demo game (T8.1): a base plus Units workers, births and deaths, entity lists."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.core.specs import ActionSpec
from demo_checks import check_example_config, random_matches
from examples.unit_harvest.game import UnitHarvestGame, scripted_action

K128 = {"max_units": 128, "size": 16, "num_resources": 16, "initial_workers": 32, "max_steps": 80}


def _stay(env: UnitHarvestGame, base: int = 0) -> dict:
    return {"base": base, "workers": np.zeros(env.max_units, np.int64)}


def test_action_space_keeps_its_order_and_counts_deciders():
    env = UnitHarvestGame(max_units=8)
    space = env.spec.roles["player"].action_space
    assert list(space.spaces) == ["base", "workers"]
    spec = ActionSpec.from_space(space)
    assert spec.has_units and spec.num_deciders == 1 + 8


def test_reset_gives_entity_lists_and_masks():
    env = UnitHarvestGame()
    res = env.reset(0, "2p")
    assert res.acting == {0, 1}
    obs, mask = res.obs[0], res.action_masks[0]
    assert obs["units"].shape == (8, 10) and obs["unit_mask"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert obs["resources"].shape == (6, 2) and obs["resource_mask"].all()
    assert mask["base"].tolist() == [True, False]                       # no stock to build yet
    assert mask["workers"]["unit"].tolist() == [True, True] + [False] * 6
    assert mask["workers"]["action"].shape == (8, 5) and mask["workers"]["action"][:, 0].all()
    assert env.spec.roles["player"].observation_space.contains(obs)


def test_both_sides_see_the_same_mirrored_board():
    for seed in range(20):
        res = UnitHarvestGame().reset(seed, "2p")
        a, b = res.obs[0], res.obs[1]
        np.testing.assert_allclose(np.sort(a["resources"], axis=0), np.sort(b["resources"], axis=0))
        np.testing.assert_allclose(a["units"][:, :3], b["units"][:, :3])


def test_seat_one_actions_are_mirrored_back():
    env = UnitHarvestGame()
    env.reset(0, "2p")
    env._pos[1, 0] = [5, 3]
    act = _stay(env)
    act["workers"][0] = 4                                   # "right" as seat 1 sees the board
    env.step({0: _stay(env), 1: act})
    assert env._pos[1, 0].tolist() == [4, 3]


def test_the_base_builds_a_worker_into_the_lowest_free_slot():
    env = UnitHarvestGame(build_cost=1)
    env.reset(0, "2p")
    env._stock[0] = 1
    res = env.step({0: _stay(env, base=1), 1: _stay(env)})
    assert res.obs[0]["unit_mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert res.action_masks[0]["workers"]["unit"][2]
    assert env._stock[0] == 0 and env._pos[0, 2].tolist() == env._bases[0].tolist()


def test_workers_of_both_sides_in_one_cell_die():
    env = UnitHarvestGame()
    env.reset(0, "2p")
    env._pos[0, 0], env._pos[1, 0] = [3, 3], [4, 3]
    act = _stay(env)
    act["workers"][0] = 4                                   # side 0 steps right onto the enemy
    res = env.step({0: act, 1: _stay(env)})
    assert not env._alive[0, 0] and not env._alive[1, 0]
    assert res.obs[0]["unit_mask"][0] == 0 and res.obs[1]["unit_mask"][0] == 0
    assert not res.action_masks[0]["workers"]["unit"][0]


def test_a_carrier_on_its_base_deposits():
    env = UnitHarvestGame(deposit_reward=0.1)
    env.reset(0, "2p")
    env._carry[0, 0] = True                                  # slot 0 stands on its base
    res = env.step({0: _stay(env), 1: _stay(env)})
    assert env._score.tolist() == [1, 0] and env._stock.tolist() == [1, 0]
    assert res.rewards == {0: pytest.approx(0.1), 1: 0.0}
    assert res.action_masks[0]["base"].tolist() == [True, True]


def test_the_match_ends_by_rule_with_team_scores():
    env = UnitHarvestGame(max_steps=3)
    env.reset(0, "2p")
    env._score[:] = [2, 1]
    for _ in range(3):
        res = env.step({0: _stay(env), 1: _stay(env)})
    assert res.episode_over and not res.truncated and res.acting == set()
    assert res.outcome.team_score == {0: 2.0, 1: 1.0}
    assert res.rewards == {0: pytest.approx(1.0), 1: pytest.approx(-1.0)}


def test_k128_configuration():
    env = UnitHarvestGame(**K128)
    res = env.reset(0, "2p")
    assert res.obs[0]["units"].shape == (128, 10) and res.obs[0]["unit_mask"].sum() == 32
    assert ActionSpec.from_space(env.spec.roles["player"].action_space).num_deciders == 129


def test_the_scripted_reference_beats_idle_play():
    env = UnitHarvestGame()
    res = env.reset(0, "2p")
    while not res.episode_over:
        res = env.step({0: scripted_action(res.obs[0]), 1: _stay(env)})
    assert res.outcome.team_score[0] > 10 and res.outcome.team_score[1] == 0


def test_the_example_config_validates():
    check_example_config("unit_harvest")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(UnitHarvestGame, min_episodes=8)
    for r in results:
        assert r.layout == "2p" and r.outcome_kind == "wdl" and len(r.teams) == 2
        rank = {t.team: t.rank for t in r.teams}
        score = {t.team: t.score for t in r.teams}
        assert (rank[0] < rank[1]) == (score[0] > score[1])
```

Create `tests/contract/test_demo_tron.py`:

```python
"""tron demo game (T8.1): FFA with elimination, 2p/3p/4p layouts, ranks by elimination order."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.tron.game import TronGame, safe_action


def _place(env: TronGame, layout: str, heads: list[list[int]], headings: list[int]) -> None:
    env.reset(0, layout)
    env._walls[:] = False
    for seat, (head, heading) in enumerate(zip(heads, headings)):
        env._head[seat], env._heading[seat] = head, heading
        env._walls[head[0], head[1]] = True


def test_layouts_and_teams():
    spec = TronGame().spec
    assert sorted(spec.layouts) == ["2p", "3p", "4p"]
    assert spec.teams("3p") == [[0], [1], [2]]
    assert spec.outcome_kind("2p") == "wdl" and spec.outcome_kind("4p") == "rank"


def test_every_layout_starts_on_distinct_cells():
    env = TronGame()
    for layout, n in (("2p", 2), ("3p", 3), ("4p", 4)):
        for seed in range(10):
            res = env.reset(seed, layout)
            assert res.acting == set(range(n))
            assert len({tuple(h) for h in env._head[:n]}) == n
            assert all(o.dtype == np.uint8 and o.shape == (2, 7, 7) for o in res.obs.values())


def test_the_window_is_rotated_to_the_heading():
    env = TronGame(size=10, view_radius=3)
    _place(env, "2p", [[5, 5], [0, 0]], [1, 0])           # seat 0 heads right
    env._walls[5, 6] = True                                # the cell straight ahead
    assert env._obs(0)[0, 2, 3] == 1                       # straight ahead = one row up from the centre
    env._walls[5, 6] = False
    env._walls[4, 5] = True                                # above the head = on its left
    assert env._obs(0)[0, 3, 2] == 1


def test_crashes_eliminate_and_ranks_follow_the_elimination_order():
    env = TronGame(size=10)
    _place(env, "3p", [[0, 5], [5, 5], [7, 5]], [0, 1, 1])
    res = env.step({0: 0, 1: 0, 2: 0})                     # seat 0 drives off the top edge
    assert res.terminated == {0} and res.rewards == {0: -1.0}
    assert res.acting == {1, 2} and not res.episode_over and set(res.obs) == {1, 2}
    env._head[1], env._head[2] = [5, 9], [7, 9]           # both at the right edge, heading right
    res = env.step({1: 0, 2: 0})
    assert res.terminated == {1, 2} and res.episode_over and res.acting == set()
    assert res.outcome.team_rank == {0: 3.0, 1: 1.5, 2: 1.5}
    assert res.rewards == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}


def test_a_head_on_crash_in_2p_is_a_draw():
    env = TronGame(size=10)
    _place(env, "2p", [[5, 3], [5, 5]], [1, 3])            # both move into (5, 4)
    res = env.step({0: 0, 1: 0})
    assert res.terminated == {0, 1} and res.episode_over
    assert res.outcome.team_rank == {0: 1.5, 1: 1.5} and res.rewards == {0: 0.0, 1: 0.0}


def test_the_last_cycle_wins_2p():
    env = TronGame(size=10)
    _place(env, "2p", [[0, 5], [5, 5]], [0, 1])
    res = env.step({0: 0, 1: 0})
    assert res.terminated == {0} and res.episode_over
    assert res.outcome.team_rank == {0: 2.0, 1: 1.0} and res.rewards == {0: -1.0, 1: 1.0}


def test_the_safe_reference_outlives_random_cycles():
    env, rng, first = TronGame(), np.random.default_rng(0), 0
    for ep in range(200):
        res = env.reset(ep, "4p")
        while not res.episode_over:
            res = env.step({s: safe_action(res.obs[s], rng) if s == 0 else int(rng.integers(3)) for s in res.acting})
        first += res.outcome.team_rank[0] == 1.0
    assert first / 200 >= 0.6


def test_the_example_config_validates():
    check_example_config("tron")


def test_random_matches_run_through_the_match_runner():
    results, seen = random_matches(TronGame, min_episodes=24)
    assert {r.layout for r in results} == {"2p", "3p", "4p"}
    for r in results:
        n = int(r.layout[0])
        assert r.outcome_kind == ("wdl" if n == 2 else "rank") and len(r.teams) == n
        assert sorted(t.team for t in r.teams) == list(range(n))
        assert sum(t.rank for t in r.teams) == pytest.approx(n * (n + 1) / 2)   # places are shared, never lost
    assert seen.terminated > 0
```

Create `tests/integration/test_demo_train_smoke.py`:

```python
"""Short `colosseum train` runs of the demo games (T8.1): the whole pipeline on a Units game."""
from __future__ import annotations

from cli_runner import run_train
from demo_checks import example_config


def test_unit_harvest_trains_briefly_and_logs_unit_diagnostics(tmp_path):
    run = run_train(example_config("unit_harvest"), tmp_path, "uh", overrides={"training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]
    train = run.records("train")
    assert train, "no train records"
    for key in ("clip_fraction", "clip_fraction_joint", "ess", "log_rho_abs_p95", "deciders_valid_mean"):
        assert key in train[-1], key
    assert train[-1]["deciders_valid_mean"] > 1.0             # several workers decide per act slot
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/contract/test_demo_coin_grid.py tests/contract/test_demo_unit_harvest.py tests/contract/test_demo_tron.py tests/integration/test_demo_train_smoke.py -q`

Expected: collection errors `ModuleNotFoundError: No module named 'examples.coin_grid'` (and `examples.unit_harvest`, `examples.tron`).

- [ ] **Step 4: `coin_grid` game**

Create an empty `examples/coin_grid/__init__.py` and `examples/coin_grid/game.py`:

```python
"""Coin grid: a solo game (spec block 10).

One player walks on a ``size x size`` grid and collects coins; a collected coin reappears on a
random free cell. There is no rule-based end: the episode is cut by an artificial step limit, so
every episode ends with ``truncated=True`` and a ``final_obs`` (the learner bootstraps from it).

Exercises: ``GameSpec.solo``, a Dict observation with a ``uint8`` grid leaf and a float vector
leaf, action masks (moves off the grid are illegal), truncation with ``final_obs``.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult

# stay, up, down, left, right as (drow, dcol)
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


class CoinGridGame(MultiAgentEnv):
    def __init__(self, size: int = 7, num_coins: int = 5, max_steps: int = 50) -> None:
        if size < 3 or not 1 <= num_coins < size * size:
            raise ValueError(f"need size >= 3 and 1 <= num_coins < size*size, got {size}, {num_coins}")
        self.size, self.num_coins, self.max_steps = size, num_coins, max_steps
        obs_space = gymnasium.spaces.Dict(
            [("grid", gymnasium.spaces.Box(0, 1, (2, size, size), np.uint8)),   # agent, coins
             ("vec", gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32))])          # steps left / max_steps
        self.spec = GameSpec.solo(obs_space, gymnasium.spaces.Discrete(5))
        self._rng = np.random.default_rng()
        self._pos = np.zeros(2, np.int64)
        self._coins = np.zeros((size, size), bool)
        self._t = 0

    def _free_cell(self) -> np.ndarray:
        free = ~self._coins
        free[self._pos[0], self._pos[1]] = False
        cells = np.argwhere(free)
        return cells[self._rng.integers(len(cells))]

    def _obs(self) -> dict:
        grid = np.zeros((2, self.size, self.size), np.uint8)
        grid[0, self._pos[0], self._pos[1]] = 1
        grid[1] = self._coins
        vec = np.array([(self.max_steps - self._t) / self.max_steps], np.float32)
        return {"grid": grid, "vec": vec}

    def _mask(self) -> np.ndarray:
        nxt = self._pos + MOVES
        return ((nxt >= 0) & (nxt < self.size)).all(axis=1)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._coins[:] = False
        self._pos = self._rng.integers(self.size, size=2)
        for _ in range(self.num_coins):
            r, c = self._free_cell()
            self._coins[r, c] = True
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: self._mask()})

    def step(self, actions: dict) -> StepResult:
        move = int(actions[0])
        self._pos = np.clip(self._pos + MOVES[move], 0, self.size - 1)
        self._t += 1
        reward = 0.0
        if self._coins[self._pos[0], self._pos[1]]:
            reward = 1.0
            self._coins[self._pos[0], self._pos[1]] = False
            r, c = self._free_cell()
            self._coins[r, c] = True
        if self._t >= self.max_steps:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True, truncated=True,
                              final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()}, action_masks={0: self._mask()}, rewards={0: reward})


def greedy_action(obs: dict) -> int:
    """Scripted reference: one step towards the nearest coin (used to sanity-check the env)."""
    agent = np.argwhere(obs["grid"][0])[0]
    coins = np.argwhere(obs["grid"][1])
    target = coins[np.abs(coins - agent).sum(axis=1).argmin()]
    d = target - agent
    if d[0] != 0:
        return 1 if d[0] < 0 else 2
    if d[1] != 0:
        return 3 if d[1] < 0 else 4
    return 0
```

- [ ] **Step 5: `unit_harvest` game**

Create an empty `examples/unit_harvest/__init__.py` and `examples/unit_harvest/game.py`:

```python
"""Unit harvest: one bot controls a base and a variable number of workers (spec block 10).

Two sides on a ``size x size`` grid move simultaneously. Each side has a base cell (seat 0 on the
left edge, seat 1 on the right edge) and up to ``max_units`` workers.
- A worker that stands on a resource cell with empty hands picks up one resource; a worker that
  carries a resource and stands on its own base deposits it (+1 score, +1 stock).
- The base may build a worker for ``build_cost`` stock; it appears on the base in the lowest free
  unit slot (units are born).
- After the moves, every cell that holds workers of both sides loses all of them (units die).
- The match lasts exactly ``max_steps`` steps (a rule, not a cut: ``truncated=False``). The side
  with the higher score wins (``Outcome.team_score``; equal scores are a draw).

Rewards: ``deposit_reward`` per deposit, plus +1 / -1 / 0 for a win / loss / draw at the end.

Observation (``Dict``, per seat, mirrored for seat 1 so both sides "start on the left"):
- ``units``: ``float32[max_units, 10]`` per own worker: x, y, carrying, direction to the nearest
  resource (dx, dy), to the own base (dx, dy), to the nearest enemy worker (dx, dy), enemy adjacent;
- ``unit_mask``: ``MultiBinary(max_units)``, 1 = the slot holds a live worker;
- ``enemies``: ``float32[max_units, 2]`` enemy worker positions + ``enemy_mask``;
- ``resources``: ``float32[num_resources, 2]`` resource positions + ``resource_mask`` (all ones:
  resources never run out; the mask follows the entity-list convention);
- ``base``: ``float32[4]``: stock / 10, own score / 10, opponent score / 10, steps left fraction.

Action (``Dict`` in this order): ``base``: ``Discrete(2)`` (0 idle, 1 build; build is masked when
the stock is short or every slot is used) and ``workers``: ``Units(max_units, Discrete(5))``
(stay, up, down, left, right; moves off the grid are masked). Deciders: 1 + ``max_units``.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult
from colosseum.envs.spaces import Units

# stay, up, down, left, right as (dx, dy); x grows to the right, y grows down
MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
MIRROR_ACTION = np.array([0, 1, 2, 4, 3], dtype=np.int64)   # left <-> right for seat 1
UNIT_FEATURES = 10


class UnitHarvestGame(MultiAgentEnv):
    def __init__(self, max_units: int = 8, size: int = 8, num_resources: int = 6, initial_workers: int = 2,
                 build_cost: int = 1, max_steps: int = 50, deposit_reward: float = 0.1) -> None:
        if not 1 <= initial_workers <= max_units:
            raise ValueError(f"initial_workers must be in [1, max_units], got {initial_workers}")
        if num_resources % 2 or num_resources < 2 or size < 4:
            raise ValueError("num_resources must be even and >= 2, size >= 4")
        self.max_units, self.size, self.num_resources = max_units, size, num_resources
        self.initial_workers, self.build_cost, self.max_steps = initial_workers, build_cost, max_steps
        self.deposit_reward = deposit_reward
        U, R = max_units, num_resources
        box = gymnasium.spaces.Box
        obs_space = gymnasium.spaces.Dict([
            ("units", box(-1.0, 1.0, (U, UNIT_FEATURES), np.float32)),
            ("unit_mask", gymnasium.spaces.MultiBinary(U)),
            ("enemies", box(0.0, 1.0, (U, 2), np.float32)),
            ("enemy_mask", gymnasium.spaces.MultiBinary(U)),
            ("resources", box(0.0, 1.0, (R, 2), np.float32)),
            ("resource_mask", gymnasium.spaces.MultiBinary(R)),
            ("base", box(0.0, np.inf, (4,), np.float32)),
        ])
        act_space = gymnasium.spaces.Dict([
            ("base", gymnasium.spaces.Discrete(2)),
            ("workers", Units(U, gymnasium.spaces.Discrete(5))),
        ])
        self.spec = GameSpec.symmetric(2, obs_space, act_space)
        self._rng = np.random.default_rng()
        self._bases = np.array([[0, size // 2], [size - 1, size // 2]], dtype=np.int64)
        self._pos = np.zeros((2, U, 2), np.int64)       # [side, slot, (x, y)]
        self._alive = np.zeros((2, U), bool)
        self._carry = np.zeros((2, U), bool)
        self._resources = np.zeros((R, 2), np.int64)
        self._stock = np.zeros(2, np.int64)
        self._score = np.zeros(2, np.int64)
        self._t = 0

    # ----- helpers -------------------------------------------------------------------------
    def _place_resources(self) -> None:
        half = self.num_resources // 2
        cells = [(x, y) for x in range(1, self.size // 2) for y in range(self.size)]
        cells = [c for c in cells if (c[0], c[1]) != tuple(self._bases[0])]
        idx = self._rng.choice(len(cells), size=half, replace=False)
        left = np.array([cells[i] for i in idx], dtype=np.int64)
        right = left.copy()
        right[:, 0] = self.size - 1 - left[:, 0]
        self._resources = np.concatenate([left, right])

    def _spawn(self, side: int) -> None:
        slot = int(np.flatnonzero(~self._alive[side])[0])
        self._alive[side, slot] = True
        self._carry[side, slot] = False
        self._pos[side, slot] = self._bases[side]

    def _view(self, side: int, xy: np.ndarray) -> np.ndarray:
        """Grid coordinates as seen by ``side`` (seat 1 sees the board mirrored left-right)."""
        out = xy.copy()
        if side == 1:
            out[..., 0] = self.size - 1 - out[..., 0]
        return out

    @staticmethod
    def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Per source cell: (dx, dy) to its nearest destination cell (Manhattan); zeros if none."""
        if len(dst) == 0:
            return np.zeros_like(src)
        diff = dst[None, :, :] - src[:, None, :]
        best = np.abs(diff).sum(axis=2).argmin(axis=1)
        return diff[np.arange(len(src)), best]

    def _obs(self, side: int) -> dict:
        U, scale = self.max_units, float(self.size - 1)
        own = self._view(side, self._pos[side])
        enemy_alive = self._alive[1 - side]
        enemy = self._view(side, self._pos[1 - side])[enemy_alive]
        res = self._view(side, self._resources)
        base = self._view(side, self._bases[side])
        units = np.zeros((U, UNIT_FEATURES), np.float32)
        units[:, 0:2] = own / scale
        units[:, 2] = self._carry[side]
        units[:, 3:5] = np.sign(self._nearest(own, res))
        units[:, 5:7] = np.sign(base[None, :] - own)
        to_enemy = self._nearest(own, enemy)
        units[:, 7:9] = np.sign(to_enemy)
        units[:, 9] = (np.abs(to_enemy).sum(axis=1) == 1) if len(enemy) else 0.0
        units[~self._alive[side]] = 0.0
        enemies = np.zeros((U, 2), np.float32)
        enemies[enemy_alive] = enemy / scale
        return {
            "units": units,
            "unit_mask": self._alive[side].astype(np.int8),
            "enemies": enemies,
            "enemy_mask": enemy_alive.astype(np.int8),
            "resources": (res / scale).astype(np.float32),
            "resource_mask": np.ones(self.num_resources, np.int8),
            "base": np.array([self._stock[side] / 10.0, self._score[side] / 10.0, self._score[1 - side] / 10.0,
                              (self.max_steps - self._t) / self.max_steps], np.float32),
        }

    def _mask(self, side: int) -> dict:
        can_build = self._stock[side] >= self.build_cost and not self._alive[side].all()
        own = self._view(side, self._pos[side])
        nxt = own[:, None, :] + MOVES[None, :, :]
        moves = ((nxt >= 0) & (nxt < self.size)).all(axis=2)
        moves[~self._alive[side]] = True
        return {"base": np.array([True, can_build]),
                "workers": {"unit": self._alive[side].copy(), "action": moves}}

    def _result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)},
                          action_masks={s: self._mask(s) for s in (0, 1)}, rewards=rewards)

    # ----- MultiAgentEnv -------------------------------------------------------------------
    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._alive[:] = False
        self._carry[:] = False
        self._stock[:] = 0
        self._score[:] = 0
        self._place_resources()
        for side in (0, 1):
            for _ in range(self.initial_workers):
                self._spawn(side)
        return self._result({})

    def step(self, actions: dict) -> StepResult:
        for side in (0, 1):
            act = actions[side]
            moves = np.asarray(act["workers"], np.int64).reshape(self.max_units)
            if side == 1:
                moves = MIRROR_ACTION[moves]
            alive = self._alive[side]
            self._pos[side, alive] = np.clip(self._pos[side, alive] + MOVES[moves[alive]], 0, self.size - 1)
            if int(act["base"]) == 1 and self._stock[side] >= self.build_cost and not alive.all():
                self._stock[side] -= self.build_cost
                self._spawn(side)
        # fights: every cell that holds workers of both sides loses all of them
        cell = self._pos[..., 0] * self.size + self._pos[..., 1]                  # [2, U]
        contested = np.intersect1d(cell[0, self._alive[0]], cell[1, self._alive[1]])
        if len(contested):
            hit = self._alive & np.isin(cell, contested)
            self._alive[hit] = False
            self._carry[hit] = False
        # harvest and deposit (vectorized over sides and slots)
        res_cell = self._resources[:, 0] * self.size + self._resources[:, 1]
        base_cell = self._bases[:, 0] * self.size + self._bases[:, 1]               # [2]
        at_base = self._alive & self._carry & (cell == base_cell[:, None])
        on_res = self._alive & ~self._carry & np.isin(cell, res_cell)
        deposits = at_base.sum(axis=1)
        self._carry[at_base] = False
        self._carry[on_res] = True
        self._score += deposits
        self._stock += deposits
        self._t += 1
        rewards = {s: self.deposit_reward * float(deposits[s]) for s in (0, 1)}
        if self._t < self.max_steps:
            return self._result(rewards)
        diff = int(self._score[0] - self._score[1])
        final = {0: float(np.sign(diff)), 1: float(-np.sign(diff))}
        rewards = {s: rewards[s] + final[s] for s in (0, 1)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(self._score[0]), 1: float(self._score[1])}))


def scripted_action(obs: dict) -> dict:
    """Reference policy for sanity checks: carriers walk home, the others walk to the nearest
    resource; the base builds whenever it can (build legality is checked by the env)."""
    units = obs["units"]
    moves = np.zeros(len(units), np.int64)
    for i, f in enumerate(units):
        dx, dy = (f[5], f[6]) if f[2] > 0 else (f[3], f[4])
        if dx != 0:
            moves[i] = 4 if dx > 0 else 3
        elif dy != 0:
            moves[i] = 2 if dy > 0 else 1
    return {"base": 1 if obs["base"][0] * 10 >= 1 else 0, "workers": moves}
```

Notes for the reviewer:
- The per-unit features are egocentric hints (direction to the nearest resource, base, enemy). They keep the demo learnable in minutes; the entity lists (`enemies`, `resources`) still exercise masked pooling.
- Seat 1 sees the board mirrored left-right and its moves are mirrored back (`MIRROR_ACTION`), so one network plays both sides. `test_both_sides_see_the_same_mirrored_board` pins this over 20 seeds.
- Speed (prototype, one process, random actions): about 2,700 env steps/s at K=8 and 1,100 at K=128.

- [ ] **Step 6: `tron` game**

Create an empty `examples/tron/__init__.py` and `examples/tron/game.py`:

```python
"""Tron light cycles: free-for-all with elimination, 2 to 4 players (spec block 10).

Every live cycle moves one cell per step, all at once, and leaves a wall behind. A cycle crashes
(``terminated``) when it moves off the board, into any wall (its own included, current heads
included) or into the same cell as another cycle in the same step. The match ends when at most
one cycle is left or after ``size * size`` steps.

Ranks follow the elimination order: cycles that crash in the same step share the average of their
places (two cycles fighting for places 3 and 4 both get 3.5; a 2p head-on crash is 1.5 / 1.5, a
draw). Survivors at the end share the top places the same way. ``Outcome.team_rank`` carries the
ranks (team = seat). Each cycle gets one reward when its place is known (at its crash or at the
end): ``1 - 2 * (rank - 1) / (n - 1)``, so +1 for first, -1 for last.

Layouts ``2p``, ``3p``, ``4p`` (``GameSpec.symmetric([2, 3, 4], ...)``), one role.
Observation: ``uint8[2, 2r+1, 2r+1]``, a window around the head rotated so that the heading points
up (row 0): channel 0 = blocked (walls, trails, off-board), channel 1 = other cycles' heads.
Action: ``Discrete(3)``: 0 straight, 1 turn left, 2 turn right. No masks.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

# headings: 0 up, 1 right, 2 down, 3 left, as (drow, dcol)
HEADINGS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
TURN = np.array([0, -1, 1], dtype=np.int64)   # straight, left, right


class TronGame(MultiAgentEnv):
    def __init__(self, size: int = 10, view_radius: int = 3, players: tuple[int, ...] = (2, 3, 4)) -> None:
        if size < 6 or not set(players) <= {2, 3, 4}:
            raise ValueError("size must be >= 6 and players a subset of {2, 3, 4}")
        self.size, self.radius = size, view_radius
        side = 2 * view_radius + 1
        obs_space = gymnasium.spaces.Box(0, 1, (2, side, side), np.uint8)
        self.spec = GameSpec.symmetric(list(players), obs_space, gymnasium.spaces.Discrete(3))
        self._rng = np.random.default_rng()
        self._n = 0
        self._walls = np.zeros((size, size), bool)
        self._head = np.zeros((4, 2), np.int64)
        self._heading = np.zeros(4, np.int64)
        self._alive = np.zeros(4, bool)
        self._ranks: dict[int, float] = {}
        self._t = 0

    def _starts(self) -> tuple[np.ndarray, np.ndarray]:
        """Start cells on the four sides of the board, heading inwards, shuffled per episode."""
        s, q = self.size, self.size // 4
        j = self._rng.integers(-1, 2, size=4)
        cells = np.array([[s // 2 + j[0], q - 1], [q - 1, s // 2 + j[1]],
                          [s // 2 + j[2], s - q], [s - q, s // 2 + j[3]]], dtype=np.int64)
        headings = np.array([1, 2, 3, 0], dtype=np.int64)          # each faces the centre
        order = self._rng.permutation(4)[: self._n]
        return cells[order], headings[order]

    def _obs(self, seat: int) -> np.ndarray:
        r, side = self.radius, 2 * self.radius + 1
        padded = np.ones((self.size + 2 * r, self.size + 2 * r), bool)
        padded[r:-r, r:-r] = self._walls
        heads = np.zeros_like(padded)
        for other in np.flatnonzero(self._alive[: self._n]):
            if other != seat:
                heads[self._head[other, 0] + r, self._head[other, 1] + r] = True
        row, col = self._head[seat]
        win = np.stack([padded[row:row + side, col:col + side], heads[row:row + side, col:col + side]])
        return np.ascontiguousarray(np.rot90(win, k=int(self._heading[seat]), axes=(1, 2))).astype(np.uint8)

    def _live(self) -> list[int]:
        return [int(s) for s in np.flatnonzero(self._alive[: self._n])]

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._n = len(self.spec.layouts[layout])
        self._t = 0
        self._ranks: dict[int, float] = {}
        self._walls[:] = False
        self._alive[:] = False
        self._alive[: self._n] = True
        cells, headings = self._starts()
        self._head[: self._n], self._heading[: self._n] = cells, headings
        for seat in range(self._n):
            self._walls[cells[seat, 0], cells[seat, 1]] = True
        live = self._live()
        return StepResult(acting=set(live), obs={s: self._obs(s) for s in live})

    def _reward(self, rank: float) -> float:
        return 1.0 - 2.0 * (rank - 1.0) / (self._n - 1)

    def step(self, actions: dict) -> StepResult:
        live = self._live()
        targets = {}
        for seat in live:
            self._heading[seat] = (self._heading[seat] + TURN[int(actions[seat])]) % 4
            targets[seat] = self._head[seat] + HEADINGS[self._heading[seat]]
        crashed = set()
        for seat, (row, col) in targets.items():
            off = not (0 <= row < self.size and 0 <= col < self.size)
            if off or self._walls[row, col]:
                crashed.add(seat)
        cells = [tuple(t) for t in targets.values()]
        for seat, cell in zip(targets, cells):
            if cells.count(cell) > 1:
                crashed.add(seat)
        for seat in live:
            if seat not in crashed:
                self._head[seat] = targets[seat]
                self._walls[targets[seat][0], targets[seat][1]] = True
        self._t += 1
        survivors = [s for s in live if s not in crashed]
        rewards: dict[int, float] = {}
        if crashed:   # places len(survivors)+1 .. len(live), shared
            shared = len(survivors) + (len(crashed) + 1) / 2.0
            for seat in crashed:
                self._ranks[seat] = shared
                rewards[seat] = self._reward(shared)
                self._alive[seat] = False
        over = len(survivors) <= 1 or self._t >= self.size * self.size
        if not over:
            return StepResult(acting=set(survivors), obs={s: self._obs(s) for s in survivors},
                              rewards=rewards, terminated=crashed)
        if survivors:
            shared = (len(survivors) + 1) / 2.0
            for seat in survivors:
                self._ranks[seat] = shared
                rewards[seat] = self._reward(shared)
        return StepResult(acting=set(), obs={}, rewards=rewards, terminated=crashed, episode_over=True,
                          outcome=Outcome(team_rank=dict(self._ranks)))


def safe_action(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Reference policy for sanity checks: a random action among those whose next cell is free."""
    r = obs.shape[1] // 2
    free = [a for a, (dr, dc) in enumerate(((-1, 0), (0, -1), (0, 1))) if not obs[0, r + dr, r + dc]]
    return int(rng.choice(free)) if free else 0
```

Prototype measurements (numpy only, 400 games each) that the thresholds of criterion 3 rest on: random cycles last 5–7 steps; the one-step `safe_action` reference takes first place in 91.5% of 2p and 83% of 4p games against random cycles, so 80% (2p) and 50% (4p first places) are reachable by a policy that has learned "do not crash".

- [ ] **Step 7: Models**

Create `examples/coin_grid/models.py`:

```python
"""Model parts for coin_grid: a small CNN over the uint8 grid plus the vector leaf."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class CoinGridEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space["grid"].shape
        v = observation_space["vec"].shape[0]
        self._latent = latent
        self.conv = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(channels * h * w + v, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> torch.Tensor:
        grid = obs["grid"].float()           # the env sends uint8; the model casts
        return self.mlp(torch.cat([self.conv(grid), obs["vec"]], dim=-1))


class CoinGridPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class CoinGridValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

Create `examples/unit_harvest/models.py`:

```python
"""Model parts for unit_harvest: an entity encoder with per-unit embeddings in ``aux`` and a
policy with a ``Discrete`` base head and a ``UnitsHead`` for the workers."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.heads import UnitsHead, make_distribution


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x[B, N, F]`` over the entities where ``mask[B, N]`` is set (0 when there are none)."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class HarvestEncoder(BaseEncoder):
    def __init__(self, observation_space, unit_dim: int = 64, latent: int = 128, **kwargs) -> None:
        super().__init__()
        f = observation_space["units"].shape[1]
        g = observation_space["base"].shape[0]
        self._latent = latent
        self.unit_net = nn.Sequential(nn.Linear(f, unit_dim), nn.ReLU(), nn.Linear(unit_dim, unit_dim), nn.ReLU())
        self.enemy_net = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.resource_net = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(unit_dim + 32 + 32 + g, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> EncoderOutput:
        units = self.unit_net(obs["units"])                                        # [B, U, E]
        own = masked_mean(units, obs["unit_mask"])
        enemies = masked_mean(self.enemy_net(obs["enemies"]), obs["enemy_mask"])
        resources = masked_mean(self.resource_net(obs["resources"]), obs["resource_mask"])
        latent = self.mlp(torch.cat([own, enemies, resources, obs["base"]], dim=-1))
        return EncoderOutput(latent=latent, aux={"units": units})


class HarvestPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, unit_dim: int = 64, unit_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        groups = {g.path: g for g in action_spec.groups}
        self.base_head = nn.Linear(in_dim, groups[("base",)].nvec[0])
        self.workers_head = UnitsHead(groups[("workers",)], unit_dim + in_dim, unit_hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        units = aux["units"]
        context = features.unsqueeze(1).expand(-1, units.shape[1], -1)
        params = {"base": self.base_head(features),
                  "workers": self.workers_head(torch.cat([units, context], dim=-1))}
        return make_distribution(self.action_spec, params)


class HarvestValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

Create `examples/tron/models.py`:

```python
"""Model parts for tron: a CNN over the rotated uint8 window around the head."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class TronEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space.shape
        self._latent = latent
        self.net = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(channels * h * w, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


class TronPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class TronValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

- [ ] **Step 8: Example configs**

Create `configs/examples/coin_grid.yaml`:

```yaml
# Coin grid: a solo game (Dict observation with a uint8 grid, masks, step-limit truncation).
# tests/learning/test_demo_learning_slow.py: the greedy policy scores >= 2x random play.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.coin_grid.game.CoinGridGame"
  kwargs: {size: 7, num_coins: 5, max_steps: 50}

networks:
  encoder_class: "examples.coin_grid.models.CoinGridEncoder"
  core: null
  policy_class: "examples.coin_grid.models.CoinGridPolicy"
  value_class: "examples.coin_grid.models.CoinGridValue"
  kwargs: {}

algorithm:
  gamma: 0.97
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 300000   # global env steps
  seed: null

matchmaking:
  mode: self_play

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

Create `configs/examples/unit_harvest.yaml`:

```yaml
# Unit harvest: one bot = a base (Discrete) + up to max_units workers (Units), simultaneous moves.
# K=128 (units experiment, docs/benchmarks.md): --set env.kwargs='{max_units: 128, size: 16,
#   num_resources: 16, initial_workers: 32, max_steps: 80}'
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.unit_harvest.game.UnitHarvestGame"
  kwargs: {max_units: 8, size: 8, num_resources: 6, initial_workers: 2, max_steps: 50}

networks:
  encoder_class: "examples.unit_harvest.models.HarvestEncoder"
  core: null
  policy_class: "examples.unit_harvest.models.HarvestPolicy"
  value_class: "examples.unit_harvest.models.HarvestValue"
  kwargs: {}

algorithm:
  gamma: 0.98
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"
  ratio_mode: "auto"        # per_unit (the action has Units)
  unit_trace: "auto"        # see docs/benchmarks.md, units experiment
  entropy_reduction: "auto" # mean over valid deciders with per_unit

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 400000
  seed: null

matchmaking:
  mode: self_play
  latest_prob: 0.8

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

Create `configs/examples/tron.yaml`:

```yaml
# Tron light cycles: FFA with elimination; 2, 3 and 4 players in one run (one network).
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.tron.game.TronGame"
  kwargs: {size: 10, view_radius: 3}

networks:
  encoder_class: "examples.tron.models.TronEncoder"
  core: null
  policy_class: "examples.tron.models.TronPolicy"
  value_class: "examples.tron.models.TronValue"
  kwargs: {}

algorithm:
  gamma: 0.97
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 400000
  seed: null

matchmaking:
  mode: self_play
  layouts: {2p: 0.4, 3p: 0.2, 4p: 0.4}   # weights of the layouts the matchmaker picks
  latest_prob: 0.8

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

The budgets (`training.total_timesteps`) are first guesses; T8.3 calibrates them against the slow learning tests.

- [ ] **Step 9: Run the new tests**

Run: `.venv/bin/python -m pytest tests/contract/test_demo_coin_grid.py tests/contract/test_demo_unit_harvest.py tests/contract/test_demo_tron.py tests/integration/test_demo_train_smoke.py -q -rw --durations=5`

Expected: all pass, no warnings; the smoke train takes 10–25 s, every other test under 5 s.

If `test_the_example_config_validates` fails, read the `ConfigError` / `EnvContractError` text: it names the rule. Fix the game or the model, never the check. If `random_matches` fails with an `EnvContractError`, the message carries "env E, seat P, episode step K, layout L": reproduce it with a direct `env.reset(seed, layout)` / `env.step(...)` loop and fix the game.

- [ ] **Step 10: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: everything passes with zero warnings; ruff prints `All checks passed!`.

- [ ] **Step 11: Commit and push**

```bash
git add examples/coin_grid examples/unit_harvest examples/tron configs/examples/coin_grid.yaml \
  configs/examples/unit_harvest.yaml configs/examples/tron.yaml tests/demo_checks.py \
  tests/contract/test_demo_coin_grid.py tests/contract/test_demo_unit_harvest.py tests/contract/test_demo_tron.py \
  tests/integration/test_demo_train_smoke.py pyproject.toml
git commit -m "feat: demo games coin_grid, unit_harvest and tron on the SP2 contract"
git push origin sp2-game-model
```

---

### Task T8.2: Demo games `team_tag`, `predator_prey`, `coop_buttons` (+ models, configs, smoke)

Spec block 10, criterion 3 (team vs team, asymmetric 1 vs N, cooperative rows). Each game is written in full:
- `team_tag`: `GameSpec.teams_of([2, 2], ...)`, each bot sees a local `uint8` window, `global_state` = the full map from the seat's perspective (switchable with `with_global_state` for the T8.4 measurement), frozen bots are removed from `acting` but not terminated and keep receiving team rewards (the contract's "dead teammate" rule), termination by rule;
- `predator_prey`: one hunter vs two prey, roles with **different** observation (`9` vs `7` floats) and action (`Discrete(5)` vs `Discrete(9)`) spaces, a hand-written `GameSpec`, caught prey are eliminated with `terminated`, two agents with `agents.<id>.roles`;
- `coop_buttons`: one team of two seats (`coop2`), both must press their buttons in the same step, `Outcome.team_score`, two agents with `teammates: mixed`.

**Files:**
- Create:
  - `examples/team_tag/__init__.py`, `examples/team_tag/game.py`, `examples/team_tag/models.py`, `configs/examples/team_tag.yaml`
  - `examples/predator_prey/__init__.py`, `examples/predator_prey/game.py`, `examples/predator_prey/models.py`, `configs/examples/predator_prey.yaml`
  - `examples/coop_buttons/__init__.py`, `examples/coop_buttons/game.py`, `examples/coop_buttons/models.py`, `configs/examples/coop_buttons.yaml`
- Test:
  - `tests/contract/test_demo_team_tag.py`
  - `tests/contract/test_demo_predator_prey.py`
  - `tests/contract/test_demo_coop_buttons.py`
  - `tests/integration/test_demo_roles_smoke.py`

**Interfaces:**
- Consumes: everything T8.1 consumes; `tests/demo_checks.py` (T8.1); `GameSpec(roles=..., layouts=...)`, `RoleSpec`, `SeatSpec`, `GameSpec.teams_of`; `BaseCriticEncoder` (`output_dim`); `networks.critic_encoder_class` (T1.7, T2.4: the value head gets `in_dim = core.output_dim + critic.output_dim`); `agents.<id>.roles`, `matchmaking.teammates: mixed` (T1.7, T5.1); `TrainRun.ratings()` of `cli_runner`; `ratings.json` = `{"env_steps": N, <layout>: {"elo", "win_rates", "games", "wr_vs_past", "past_games", "scores", "cross_play"}}` (T5.2, T5.3).
- Produces:
  - `examples.team_tag.game`: `TeamTagGame(size=7, view_radius=2, max_steps=30, tag_reward=0.2, with_global_state=True)`, `TAG`, `TEAM`, `MOVES`, `MIRROR_ACTION`, `chase_action(obs, mask) -> int`
  - `examples.team_tag.models`: `TagEncoder`, `TagCriticEncoder`, `TagPolicy`, `TagValue`
  - `examples.predator_prey.game`: `PredatorPreyGame(size=7, max_steps=30, catch_radius=1)`, `HUNTER_MOVES`, `PREY_MOVES`, `chase_action(obs) -> int`, `flee_action(obs, mask, size=7) -> int`
  - `examples.predator_prey.models`: `PPEncoder`, `PPPolicy`, `PPValue` (one set for both roles; sizes come from the injected spaces)
  - `examples.coop_buttons.game`: `CoopButtonsGame(size=5, max_steps=40, button_reward=0.05)`, `PRESS`, `scripted_action(obs, size=5) -> int`, `measure_baselines(episodes=300, seed=0, **env_kwargs) -> {"oracle": float, "random": float}`
  - `examples.coop_buttons.models`: `CoopEncoder`, `CoopPolicy`, `CoopValue`

- [ ] **Step 1: Write the failing tests**

Create `tests/contract/test_demo_team_tag.py`:

```python
"""team_tag demo game (T8.2): 2v2, local window, global_state, frozen teammates stay live."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.team_tag.game import TAG, TEAM, TeamTagGame, chase_action

STAY = {0: 0, 1: 0, 2: 0, 3: 0}


def _place(env: TeamTagGame, positions: list[list[int]]) -> None:
    env.reset(0, "2v2")
    env._pos = np.array(positions, dtype=np.int64)


def test_layout_teams_and_global_state_space():
    env = TeamTagGame(size=7)
    assert list(env.spec.layouts) == ["2v2"] and env.spec.teams("2v2") == [[0, 1], [2, 3]]
    role = env.spec.roles["player"]
    assert role.global_state_space.shape == (4, 7, 7) and role.global_state_space.dtype == np.uint8
    assert TeamTagGame(with_global_state=False).spec.roles["player"].global_state_space is None


def test_reset_gives_windows_masks_and_global_state_to_every_acting_seat():
    env = TeamTagGame(size=7, view_radius=2)
    res = env.reset(0, "2v2")
    assert res.acting == {0, 1, 2, 3}
    assert set(res.global_state) == {0, 1, 2, 3}
    for seat in range(4):
        assert res.obs[seat]["window"].shape == (4, 5, 5) and res.obs[seat]["window"].dtype == np.uint8
        assert res.global_state[seat].shape == (4, 7, 7) and res.global_state[seat][3].sum() == 1
        assert not res.action_masks[seat][TAG]                  # nobody within reach at the start
    assert TeamTagGame(with_global_state=False).reset(0, "2v2").global_state is None


def test_team_one_sees_the_board_mirrored():
    env = TeamTagGame(size=7)
    _place(env, [[0, 1], [0, 5], [6, 1], [6, 5]])
    assert env._obs(0)["vec"][0] == 0.0 and env._obs(2)["vec"][0] == 0.0   # both "start on the left"
    env.step({0: 0, 1: 0, 2: 4, 3: 0})                         # seat 2 steps "right" in its own view
    assert env._pos[2].tolist() == [5, 1]


def test_a_tag_freezes_the_enemy_who_keeps_getting_team_rewards():
    env = TeamTagGame(size=7, tag_reward=0.2, max_steps=10)
    _place(env, [[3, 3], [0, 0], [4, 3], [6, 6]])
    assert env._mask(0)[TAG] and env._mask(2)[TAG]
    res = env.step({0: TAG, 1: 0, 2: 0, 3: 0})
    assert res.acting == {0, 1, 3} and not res.terminated       # frozen, not terminated
    assert res.rewards == {0: pytest.approx(0.2), 1: pytest.approx(0.2), 2: pytest.approx(-0.2),
                           3: pytest.approx(-0.2)}
    res = env.step({0: 0, 1: 0, 3: 0})
    assert 2 in res.rewards and 2 not in res.acting               # the frozen seat is still live


def test_mutual_tags_freeze_both():
    env = TeamTagGame(size=7)
    _place(env, [[3, 3], [0, 0], [4, 4], [6, 6]])
    res = env.step({0: TAG, 1: 0, 2: TAG, 3: 0})
    assert res.acting == {1, 3}


def test_freezing_a_whole_team_ends_the_match_by_rule():
    env = TeamTagGame(size=7)
    _place(env, [[3, 3], [3, 4], [4, 3], [4, 4]])
    res = env.step({0: TAG, 1: 0, 2: 0, 3: 0})                   # seat 0 reaches both enemies
    assert res.episode_over and not res.truncated and res.acting == set() and not res.terminated
    assert res.outcome.team_score == {0: 2.0, 1: 0.0}
    assert res.rewards[0] == pytest.approx(1.0 + 0.4) and res.rewards[3] == pytest.approx(-1.0 - 0.4)


def test_time_limit_is_a_draw_on_equal_numbers():
    env = TeamTagGame(size=7, max_steps=2)
    _place(env, [[0, 0], [0, 6], [6, 0], [6, 6]])
    env.step(STAY)
    res = env.step(STAY)
    assert res.episode_over and not res.truncated
    assert res.outcome.team_score == {0: 2.0, 1: 2.0} and res.rewards == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}


def test_the_chasing_reference_beats_random_teams():
    env, rng, wins = TeamTagGame(), np.random.default_rng(0), 0
    for ep in range(100):
        res = env.reset(ep, "2v2")
        while not res.episode_over:
            res = env.step({s: chase_action(res.obs[s], res.action_masks[s]) if TEAM[s] == 0
                            else int(rng.choice(np.flatnonzero(res.action_masks[s]))) for s in res.acting})
        wins += res.outcome.team_score[0] > res.outcome.team_score[1]
    assert wins >= 75


def test_the_example_config_validates():
    check_example_config("team_tag")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(TeamTagGame, min_episodes=8)
    for r in results:
        assert r.layout == "2v2" and r.outcome_kind == "wdl"
        assert sorted((s.seat, s.team) for s in r.seats) == [(0, 0), (1, 0), (2, 1), (3, 1)]
        assert all(s.eliminated_step is None for s in r.seats)       # frozen bots are never eliminated
```

Create `tests/contract/test_demo_predator_prey.py`:

```python
"""predator_prey demo game (T8.2): 1 vs 2, roles with different spaces, prey elimination."""
from __future__ import annotations

import numpy as np

from demo_checks import check_example_config, random_matches
from examples.predator_prey.game import PredatorPreyGame, chase_action, flee_action


def _place(env: PredatorPreyGame, positions: list[list[int]]) -> None:
    env.reset(0, "1v2")
    env._pos = np.array(positions, dtype=np.int64)


def test_roles_have_different_spaces():
    spec = PredatorPreyGame().spec
    hunter, prey = spec.roles["hunter"], spec.roles["prey"]
    assert hunter.observation_space.shape == (9,) and prey.observation_space.shape == (7,)
    assert hunter.action_space.n == 5 and prey.action_space.n == 9
    assert [(s.role, s.team) for s in spec.layouts["1v2"]] == [("hunter", 0), ("prey", 1), ("prey", 1)]
    assert spec.outcome_kind("1v2") == "wdl"


def test_reset_keeps_the_prey_out_of_reach():
    env = PredatorPreyGame()
    for seed in range(30):
        res = env.reset(seed, "1v2")
        assert res.acting == {0, 1, 2}
        assert (np.abs(env._pos[1:] - env._pos[0]).max(axis=1) > env.catch_radius).all()
        assert res.obs[0].shape == (9,) and res.obs[1].shape == (7,)
        assert res.action_masks[0].shape == (5,) and res.action_masks[1].shape == (9,)


def test_a_caught_prey_is_eliminated_with_its_penalty():
    env = PredatorPreyGame(size=7)
    _place(env, [[3, 3], [5, 3], [0, 0]])
    res = env.step({0: 4, 1: 0, 2: 0})                       # hunter steps right, prey 1 is adjacent
    assert res.terminated == {1} and res.rewards == {1: -1.0, 0: 0.5}
    assert res.acting == {0, 2} and not res.episode_over


def test_catching_both_prey_ends_the_match_for_the_hunter():
    env = PredatorPreyGame(size=7)
    _place(env, [[3, 3], [5, 3], [5, 4]])
    res = env.step({0: 4, 1: 0, 2: 0})
    assert res.terminated == {1, 2} and res.episode_over and not res.truncated
    assert res.outcome.team_rank == {0: 1.0, 1: 2.0}
    assert res.rewards == {1: -1.0, 2: -1.0, 0: 1.0}


def test_a_surviving_prey_wins_at_the_time_limit():
    env = PredatorPreyGame(size=7, max_steps=1)
    _place(env, [[0, 0], [6, 6], [6, 0]])
    res = env.step({0: 0, 1: 0, 2: 0})
    assert res.episode_over and not res.truncated and not res.terminated
    assert res.outcome.team_rank == {0: 2.0, 1: 1.0}
    assert res.rewards == {0: -1.0, 1: 1.0, 2: 1.0}


def _hunter_win_rate(hunter, prey, episodes: int = 200) -> float:
    env, rng, wins = PredatorPreyGame(), np.random.default_rng(0), 0
    for ep in range(episodes):
        res = env.reset(ep, "1v2")
        while not res.episode_over:
            res = env.step({s: (hunter if s == 0 else prey)(res.obs[s], res.action_masks[s], rng)
                            for s in res.acting})
        wins += res.outcome.team_rank[0] == 1.0
    return wins / episodes


def _random(obs, mask, rng):
    return int(rng.choice(np.flatnonzero(mask)))


def test_random_play_is_balanced_and_the_references_win_their_roles():
    assert 0.35 <= _hunter_win_rate(_random, _random) <= 0.75
    assert _hunter_win_rate(lambda o, m, r: chase_action(o), _random) >= 0.9
    assert _hunter_win_rate(_random, lambda o, m, r: flee_action(o, m)) <= 0.15


def test_the_example_config_validates():
    check_example_config("predator_prey")


def test_random_matches_run_through_the_match_runner():
    results, seen = random_matches(PredatorPreyGame, min_episodes=12)
    for r in results:
        assert r.layout == "1v2" and r.outcome_kind == "wdl"
        assert [s.role for s in sorted(r.seats, key=lambda s: s.seat)] == ["hunter", "prey", "prey"]
    assert seen.terminated > 0
```

Create `tests/contract/test_demo_coop_buttons.py`:

```python
"""coop_buttons demo game (T8.2): one team, joint presses, score outcome."""
from __future__ import annotations

import numpy as np
import pytest

from demo_checks import check_example_config, random_matches
from examples.coop_buttons.game import PRESS, CoopButtonsGame, measure_baselines


def _place(env: CoopButtonsGame, pos: list[list[int]], buttons: list[list[int]]) -> None:
    env.reset(0, "coop2")
    env._pos = np.array(pos, dtype=np.int64)
    env._button = np.array(buttons, dtype=np.int64)


def test_one_team_layout():
    spec = CoopButtonsGame().spec
    assert list(spec.layouts) == ["coop2"] and spec.teams("coop2") == [[0, 1]]
    assert spec.outcome_kind("coop2") == "score"


def test_a_joint_press_scores_and_moves_the_buttons():
    env = CoopButtonsGame(button_reward=0.05)
    _place(env, [[1, 1], [3, 3]], [[1, 1], [3, 3]])
    res = env.step({0: PRESS, 1: PRESS})
    assert env._score == 1 and res.acting == {0, 1}
    assert res.rewards[0] >= 1.0 and res.rewards[1] >= 1.0
    assert not (env._button == np.array([[1, 1], [3, 3]])).all()


def test_a_lonely_press_scores_nothing():
    env = CoopButtonsGame(button_reward=0.05)
    _place(env, [[1, 1], [3, 2]], [[1, 1], [3, 3]])
    res = env.step({0: PRESS, 1: PRESS})
    assert env._score == 0
    assert res.rewards == {0: pytest.approx(0.05), 1: 0.0}            # only the button bonus


def test_the_match_ends_by_rule_with_the_team_score():
    env = CoopButtonsGame(max_steps=2)
    _place(env, [[1, 1], [3, 3]], [[1, 1], [3, 3]])
    env.step({0: PRESS, 1: PRESS})
    res = env.step({0: 0, 1: 0})
    assert res.episode_over and not res.truncated and res.outcome.team_score == {0: 1.0}


def test_the_oracle_scores_and_random_play_does_not():
    base = measure_baselines(episodes=100)
    assert base["oracle"] >= 5.0 and base["random"] <= 0.1


def test_the_example_config_validates():
    check_example_config("coop_buttons")


def test_random_matches_run_through_the_match_runner():
    results, _ = random_matches(CoopButtonsGame, min_episodes=8)
    for r in results:
        assert r.layout == "coop2" and r.outcome_kind == "score" and len(r.teams) == 1
```

Create `tests/integration/test_demo_roles_smoke.py`:

```python
"""Short `colosseum train` runs of the demo games with several agents (T8.2): roles and teammates."""
from __future__ import annotations

from cli_runner import run_train
from demo_checks import example_config


def test_predator_prey_trains_one_agent_per_role(tmp_path):
    run = run_train(example_config("predator_prey"), tmp_path, "pp", overrides={"training.total_timesteps": "3000"})
    assert run.returncode == 0, run.stderr[-3000:]
    agents = {r["agent"] for r in run.records("train")}
    assert agents == {"hunter", "prey"}
    for agent in ("hunter", "prey"):
        assert any((run.root / "checkpoints" / agent).glob("ckpt_v*")), agent
    assert "1v2" in run.ratings()


def test_coop_buttons_with_mixed_teammates_forms_a_cross_play_table(tmp_path):
    run = run_train(example_config("coop_buttons"), tmp_path, "coop", overrides={"training.total_timesteps": "4000"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert {r["agent"] for r in run.records("train")} == {"coop_a", "coop_b"}
    cross = run.ratings()["coop2"]["cross_play"]
    assert cross, "no cross-play entries after training with teammates: mixed"
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest tests/contract/test_demo_team_tag.py tests/contract/test_demo_predator_prey.py tests/contract/test_demo_coop_buttons.py tests/integration/test_demo_roles_smoke.py -q`

Expected: collection errors `ModuleNotFoundError: No module named 'examples.team_tag'` (and the other two).

- [ ] **Step 3: `team_tag` game**

Create an empty `examples/team_tag/__init__.py` and `examples/team_tag/game.py`:

```python
"""Team tag: two teams of two, each seat is a separate bot (spec block 10).

Four bots on a ``size x size`` grid act simultaneously. A bot may tag (freeze) the enemies within
one cell of it (Chebyshev distance <= 1, its own cell included); all tags of a step resolve at
once, so two bots can freeze each other. A frozen bot stays on the board, stops acting and is
*not* terminated: it still receives its team's rewards and gets ``terminal`` at the end of the
episode (the "dead teammate" rule of the contract).

The match ends by rule when a team has no active bot or after ``max_steps`` steps
(``truncated=False``). ``Outcome.team_score`` = active bots per team; more wins, equal is a draw.
Rewards (shared by every member of a team, frozen or not): ``tag_reward`` for each enemy the team
froze this step, ``-tag_reward`` for each own bot frozen, and +1 / -1 / 0 at the end.

Each bot sees only a local window; the centralized critic gets the full map as ``global_state``.
Team 1 sees the board mirrored left-right (both teams "start on the left"); its actions are mirrored
back.
- Observation (``Dict``): ``window``: ``uint8[4, 2r+1, 2r+1]`` around the bot: allies (active),
  enemies (active), frozen bots, off-board; ``vec``: ``float32[4]``: x, y, steps left fraction,
  active teammates other than itself.
- ``global_state`` (if ``with_global_state``): ``uint8[4, size, size]``: own team active, enemies
  active, frozen bots, the bot itself.
- Action: ``Discrete(6)``: stay, up, down, left, right, tag. Moves off the board are masked; tag is
  masked unless an active enemy is within reach.
Layout ``2v2`` (``GameSpec.teams_of([2, 2], ...)``).
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

# stay, up, down, left, right as (dx, dy); x grows to the right, y grows down
MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
MIRROR_ACTION = np.array([0, 1, 2, 4, 3, 5], dtype=np.int64)   # left <-> right for team 1
TAG = 5
TEAM = np.array([0, 0, 1, 1])


class TeamTagGame(MultiAgentEnv):
    def __init__(self, size: int = 7, view_radius: int = 2, max_steps: int = 30, tag_reward: float = 0.2,
                 with_global_state: bool = True) -> None:
        if size < 5:
            raise ValueError(f"size must be >= 5, got {size}")
        self.size, self.radius, self.max_steps, self.tag_reward = size, view_radius, max_steps, tag_reward
        w = 2 * view_radius + 1
        obs_space = gymnasium.spaces.Dict([
            ("window", gymnasium.spaces.Box(0, 1, (4, w, w), np.uint8)),
            ("vec", gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32)),
        ])
        gs_space = gymnasium.spaces.Box(0, 1, (4, size, size), np.uint8) if with_global_state else None
        self.with_global_state = with_global_state
        self.spec = GameSpec.teams_of([2, 2], obs_space, gymnasium.spaces.Discrete(6), gs_space)
        self._rng = np.random.default_rng()
        self._pos = np.zeros((4, 2), np.int64)
        self._active = np.ones(4, bool)
        self._t = 0

    def _view(self, seat: int, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, copy=True)
        if TEAM[seat] == 1:
            out[..., 0] = self.size - 1 - out[..., 0]
        return out

    def _maps(self, seat: int) -> np.ndarray:
        """``bool[4, size, size]`` from ``seat``'s perspective: allies, enemies, frozen, self."""
        maps = np.zeros((4, self.size, self.size), bool)
        pos = self._view(seat, self._pos)
        for other in range(4):
            x, y = pos[other]
            if not self._active[other]:
                maps[2, y, x] = True
            elif TEAM[other] == TEAM[seat]:
                maps[0, y, x] = True
            else:
                maps[1, y, x] = True
        maps[3, pos[seat, 1], pos[seat, 0]] = True
        return maps

    def _obs(self, seat: int) -> dict:
        r, w = self.radius, 2 * self.radius + 1
        maps = self._maps(seat)
        padded = np.zeros((4, self.size + 2 * r, self.size + 2 * r), bool)
        padded[:3, r:-r, r:-r] = maps[:3]
        padded[3] = True
        padded[3, r:-r, r:-r] = False
        x, y = self._view(seat, self._pos[seat])
        mates = [s for s in range(4) if TEAM[s] == TEAM[seat] and s != seat and self._active[s]]
        vec = np.array([x / (self.size - 1), y / (self.size - 1), (self.max_steps - self._t) / self.max_steps,
                        float(len(mates))], np.float32)
        return {"window": padded[:, y:y + w, x:x + w].astype(np.uint8), "vec": vec}

    def _enemy_in_reach(self, seat: int) -> bool:
        enemies = (TEAM != TEAM[seat]) & self._active
        dist = np.abs(self._pos - self._pos[seat]).max(axis=1)
        return bool((enemies & (dist <= 1)).any())

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._view(seat, self._pos[seat]) + MOVES
        mask = np.ones(6, bool)
        mask[:5] = ((nxt >= 0) & (nxt < self.size)).all(axis=1)
        mask[TAG] = self._enemy_in_reach(seat)
        return mask

    def _acting_result(self, rewards: dict[int, float]) -> StepResult:
        acting = {int(s) for s in np.flatnonzero(self._active)}
        gs = {s: self._maps(s).astype(np.uint8) for s in acting} if self.with_global_state else None
        return StepResult(acting=acting, obs={s: self._obs(s) for s in acting},
                          action_masks={s: self._mask(s) for s in acting}, rewards=rewards, global_state=gs)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._active[:] = True
        rows = self._rng.choice(self.size, size=4, replace=False)
        self._pos = np.array([[0, rows[0]], [0, rows[1]], [self.size - 1, rows[2]], [self.size - 1, rows[3]]],
                             dtype=np.int64)
        return self._acting_result({})

    def step(self, actions: dict) -> StepResult:
        taggers = []
        for seat, action in actions.items():
            a = int(action)
            if TEAM[seat] == 1:
                a = int(MIRROR_ACTION[a])
            if a == TAG:
                taggers.append(seat)
        frozen_now = set()
        for seat in taggers:   # resolved against the positions before the moves, all at once
            dist = np.abs(self._pos - self._pos[seat]).max(axis=1)
            for other in np.flatnonzero((TEAM != TEAM[seat]) & self._active & (dist <= 1)):
                frozen_now.add(int(other))
        for seat, action in actions.items():
            a = int(action)
            if TEAM[seat] == 1:
                a = int(MIRROR_ACTION[a])
            if a != TAG and seat not in frozen_now:
                self._pos[seat] = np.clip(self._pos[seat] + MOVES[a], 0, self.size - 1)
        for seat in frozen_now:
            self._active[seat] = False
        self._t += 1
        lost = np.array([sum(1 for s in frozen_now if TEAM[s] == t) for t in (0, 1)])
        team_reward = {t: self.tag_reward * float(lost[1 - t] - lost[t]) for t in (0, 1)}
        active = np.array([int(self._active[TEAM == t].sum()) for t in (0, 1)])
        over = active.min() == 0 or self._t >= self.max_steps
        if not over:
            return self._acting_result({s: team_reward[TEAM[s]] for s in range(4)})
        final = {0: float(np.sign(active[0] - active[1])), 1: float(np.sign(active[1] - active[0]))}
        rewards = {s: team_reward[TEAM[s]] + final[TEAM[s]] for s in range(4)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(active[0]), 1: float(active[1])}))


def chase_action(obs: dict, mask: np.ndarray) -> int:
    """Reference policy for sanity checks: tag when possible, else step towards the nearest
    visible enemy, else walk right (towards the enemy side)."""
    if mask[TAG]:
        return TAG
    enemies = np.argwhere(obs["window"][1])
    if len(enemies):
        r = obs["window"].shape[1] // 2
        dy, dx = (enemies[np.abs(enemies - r).sum(axis=1).argmin()] - r)
        if dx != 0 and mask[4 if dx > 0 else 3]:
            return 4 if dx > 0 else 3
        if dy != 0 and mask[2 if dy > 0 else 1]:
            return 2 if dy > 0 else 1
        return 0
    return 4 if mask[4] else 0
```

Prototype measurements (400 games): random vs random — team 0 wins 35%, draws 37%; the `chase_action` reference team beats a random team in 89–91% of games from either side, so 80% is reachable.

- [ ] **Step 4: `predator_prey` game**

Create an empty `examples/predator_prey/__init__.py` and `examples/predator_prey/game.py`:

```python
"""Predator and prey: one hunter against two prey, roles with different spaces (spec block 10).

On a ``size x size`` grid the hunter (team 0) and two prey (team 1) move simultaneously. After the
moves, every prey within ``catch_radius`` of the hunter (Chebyshev distance) is caught: it is
eliminated (``terminated``) with reward -1, and the hunter gets +0.5 per catch. The match ends by
rule when both prey are caught (the hunter wins) or after ``max_steps`` steps (the prey team wins
if at least one prey is still free; a surviving prey gets +1, the hunter -1).
``Outcome.team_rank`` = {winner: 1, loser: 2}.

Roles (layout ``1v2``: seat 0 hunter / team 0, seats 1-2 prey / team 1):
- ``hunter``: observation ``float32[9]``: own x, y; for both prey (nearest first) dx, dy and a free
  flag; steps left fraction. Action ``Discrete(5)``: stay, up, down, left, right.
- ``prey``: observation ``float32[7]``: own x, y; dx, dy to the hunter; dx, dy to the other prey and
  its free flag. Action ``Discrete(9)``: stay and the eight king moves.
Moves off the board are masked. The roles have different spaces, so they need separate agents
(``agents.<id>.roles``).
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult

HUNTER_MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
PREY_MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]],
                      dtype=np.int64)
HUNTER, PREY = 0, (1, 2)


class PredatorPreyGame(MultiAgentEnv):
    def __init__(self, size: int = 7, max_steps: int = 30, catch_radius: int = 1) -> None:
        if size < 4:
            raise ValueError(f"size must be >= 4, got {size}")
        self.size, self.max_steps, self.catch_radius = size, max_steps, catch_radius
        box = gymnasium.spaces.Box
        hunter = RoleSpec(box(-1.0, 1.0, (9,), np.float32), gymnasium.spaces.Discrete(5))
        prey = RoleSpec(box(-1.0, 1.0, (7,), np.float32), gymnasium.spaces.Discrete(9))
        self.spec = GameSpec(roles={"hunter": hunter, "prey": prey},
                             layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
        self._rng = np.random.default_rng()
        self._pos = np.zeros((3, 2), np.int64)
        self._free = np.ones(3, bool)
        self._t = 0

    def _moves(self, seat: int) -> np.ndarray:
        return HUNTER_MOVES if seat == HUNTER else PREY_MOVES

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._pos[seat] + self._moves(seat)
        return ((nxt >= 0) & (nxt < self.size)).all(axis=1)

    def _obs(self, seat: int) -> np.ndarray:
        scale = float(self.size - 1)
        own = self._pos[seat] / scale
        if seat == HUNTER:
            prey = sorted(PREY, key=lambda p: (not self._free[p], np.abs(self._pos[p] - self._pos[0]).sum()))
            parts = [own]
            for p in prey:
                d = (self._pos[p] - self._pos[0]) / scale if self._free[p] else np.zeros(2)
                parts += [d, [float(self._free[p])]]
            parts.append([(self.max_steps - self._t) / self.max_steps])
        else:
            other = PREY[1] if seat == PREY[0] else PREY[0]
            d_other = (self._pos[other] - self._pos[seat]) / scale if self._free[other] else np.zeros(2)
            parts = [own, (self._pos[HUNTER] - self._pos[seat]) / scale, d_other, [float(self._free[other])]]
        return np.concatenate(parts).astype(np.float32)

    def _acting(self) -> set[int]:
        return {HUNTER} | {p for p in PREY if self._free[p]}

    def _result(self, rewards: dict[int, float], terminated: set[int]) -> StepResult:
        acting = self._acting()
        return StepResult(acting=acting, obs={s: self._obs(s) for s in acting},
                          action_masks={s: self._mask(s) for s in acting}, rewards=rewards, terminated=terminated)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._free[:] = True
        while True:
            self._pos = self._rng.integers(self.size, size=(3, 2))
            gap = np.abs(self._pos[1:] - self._pos[0]).max(axis=1)
            if (gap > self.catch_radius + 1).all():
                break
        return self._result({}, set())

    def step(self, actions: dict) -> StepResult:
        for seat, action in actions.items():
            self._pos[seat] = np.clip(self._pos[seat] + self._moves(seat)[int(action)], 0, self.size - 1)
        self._t += 1
        caught = {p for p in PREY if self._free[p]
                  and np.abs(self._pos[p] - self._pos[HUNTER]).max() <= self.catch_radius}
        rewards = {p: -1.0 for p in caught}
        rewards[HUNTER] = 0.5 * len(caught)
        for p in caught:
            self._free[p] = False
        if not self._free[1:].any():
            return StepResult(acting=set(), obs={}, rewards=rewards, terminated=caught, episode_over=True,
                              outcome=Outcome(team_rank={0: 1.0, 1: 2.0}))
        if self._t < self.max_steps:
            return self._result(rewards, caught)
        rewards[HUNTER] -= 1.0
        for p in PREY:
            if self._free[p]:
                rewards[p] = rewards.get(p, 0.0) + 1.0
        return StepResult(acting=set(), obs={}, rewards=rewards, terminated=caught, episode_over=True,
                          outcome=Outcome(team_rank={0: 2.0, 1: 1.0}))


def chase_action(obs: np.ndarray) -> int:
    """Reference hunter: step towards the nearest free prey."""
    dx, dy = obs[2], obs[3]
    if abs(dx) >= abs(dy) and dx != 0:
        return 4 if dx > 0 else 3
    if dy != 0:
        return 2 if dy > 0 else 1
    return 0


def flee_action(obs: np.ndarray, mask: np.ndarray, size: int = 7) -> int:
    """Reference prey: the legal move that maximizes the Chebyshev distance to the hunter."""
    to_hunter = np.rint(obs[2:4] * (size - 1))
    dist = [np.abs(to_hunter - PREY_MOVES[a]).max() if mask[a] else -1.0 for a in range(9)]
    return int(np.argmax(dist))
```

The defaults (`size=7`, `max_steps=30`) were chosen so that both thresholds of criterion 3 mean something: a random hunter catches both random prey in 53.5% of games (so a random prey team wins 46.5%, below the 70% a trained prey needs); the chasing reference hunter wins 100% against random prey; the fleeing reference prey team wins 93% against a random hunter (400 games each). Smaller boards made random hunters win too often (91% at 5×5), larger ones made random prey safe (67% prey wins at 9×9).

- [ ] **Step 5: `coop_buttons` game**

Create an empty `examples/coop_buttons/__init__.py` and `examples/coop_buttons/game.py`:

```python
"""Coop buttons: one team of two seats and no opponent (spec block 10).

Two bots on a ``size x size`` grid each have their own button. The team scores a point only when
both bots press while standing on their buttons in the same step; then both buttons move to new
random cells. A press anywhere else does nothing. The match lasts ``max_steps`` steps (a rule:
``truncated=False``); the team score is the number of points (``Outcome.team_score``).

Rewards: every point gives both seats +1. A seat that ends a step on its own button also gets
``button_reward`` (default 0.05): without it, uniformly random play almost never scores and the
team would get no learning signal. The score (and so every threshold) counts points only.

Layout ``coop2`` (``GameSpec.teams_of([2], ...)``), outcome kind ``score``.
Observation ``float32[9]``: own x, y; own button dx, dy; partner dx, dy; partner's button dx, dy
(relative to the partner); steps left fraction. Action ``Discrete(6)``: stay, up, down, left, right,
press. Moves off the board are masked.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

MOVES = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
PRESS = 5


class CoopButtonsGame(MultiAgentEnv):
    def __init__(self, size: int = 5, max_steps: int = 40, button_reward: float = 0.05) -> None:
        if size < 3:
            raise ValueError(f"size must be >= 3, got {size}")
        self.size, self.max_steps, self.button_reward = size, max_steps, button_reward
        obs_space = gymnasium.spaces.Box(-1.0, 1.0, (9,), np.float32)
        self.spec = GameSpec.teams_of([2], obs_space, gymnasium.spaces.Discrete(6))
        self._rng = np.random.default_rng()
        self._pos = np.zeros((2, 2), np.int64)
        self._button = np.zeros((2, 2), np.int64)
        self._score = 0
        self._t = 0

    def _new_buttons(self) -> None:
        cells = self._rng.choice(self.size * self.size, size=2, replace=False)
        self._button = np.stack([cells // self.size, cells % self.size], axis=1).astype(np.int64)

    def _obs(self, seat: int) -> np.ndarray:
        scale, other = float(self.size - 1), 1 - seat
        return np.concatenate([
            self._pos[seat] / scale,
            (self._button[seat] - self._pos[seat]) / scale,
            (self._pos[other] - self._pos[seat]) / scale,
            (self._button[other] - self._pos[other]) / scale,
            [(self.max_steps - self._t) / self.max_steps],
        ]).astype(np.float32)

    def _mask(self, seat: int) -> np.ndarray:
        nxt = self._pos[seat] + MOVES
        return np.append(((nxt >= 0) & (nxt < self.size)).all(axis=1), True)

    def _result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)},
                          action_masks={s: self._mask(s) for s in (0, 1)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t, self._score = 0, 0
        self._pos = self._rng.integers(self.size, size=(2, 2))
        self._new_buttons()
        return self._result({})

    def step(self, actions: dict) -> StepResult:
        acts = [int(actions[0]), int(actions[1])]
        on_button = [bool((self._pos[s] == self._button[s]).all()) for s in (0, 1)]
        point = all(acts[s] == PRESS and on_button[s] for s in (0, 1))
        for s in (0, 1):
            if acts[s] != PRESS:
                self._pos[s] = np.clip(self._pos[s] + MOVES[acts[s]], 0, self.size - 1)
        if point:
            self._score += 1
            self._new_buttons()
        self._t += 1
        on_button = [bool((self._pos[s] == self._button[s]).all()) for s in (0, 1)]
        rewards = {s: (1.0 if point else 0.0) + (self.button_reward if on_button[s] else 0.0) for s in (0, 1)}
        if self._t < self.max_steps:
            return self._result(rewards)
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_score={0: float(self._score)}))


def scripted_action(obs: np.ndarray, size: int = 5) -> int:
    """Oracle used to set the learning threshold: walk to the own button, wait there, press when
    the partner stands on its button too (both see the same facts, so both press together)."""
    to_button = np.rint(obs[2:4] * (size - 1))
    partner_on = not np.rint(obs[6:8] * (size - 1)).any()
    if not to_button.any():
        return PRESS if partner_on else 0
    dx, dy = to_button
    if dx != 0:
        return 4 if dx > 0 else 3
    return 2 if dy > 0 else 1


def measure_baselines(episodes: int = 300, seed: int = 0, **env_kwargs) -> dict[str, float]:
    """Mean team score of the oracle and of uniformly random legal play (numpy only)."""
    env = CoopButtonsGame(**env_kwargs)
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("oracle", "random"):
        scores = []
        for ep in range(episodes):
            res = env.reset(seed + ep, "coop2")
            while not res.episode_over:
                if name == "oracle":
                    acts = {s: scripted_action(res.obs[s], env.size) for s in res.acting}
                else:
                    acts = {s: int(rng.choice(np.flatnonzero(res.action_masks[s]))) for s in res.acting}
                res = env.step(acts)
            scores.append(res.outcome.team_score[0])
        out[name] = float(np.mean(scores))
    return out
```

Measured with `measure_baselines()` (300 episodes): oracle 7.34 points per match, random play 0.0. Without `button_reward` random play would almost never produce a learning signal (0.01 points per match even on a 4×4 board).

- [ ] **Step 6: Models**

Create `examples/team_tag/models.py`:

```python
"""Model parts for team_tag: the policy sees the local window, the value additionally sees the
whole map through a critic encoder over ``global_state`` (centralized critic)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class TagEncoder(BaseEncoder):
    def __init__(self, observation_space, channels: int = 16, latent: int = 128, **kwargs) -> None:
        super().__init__()
        c, h, w = observation_space["window"].shape
        v = observation_space["vec"].shape[0]
        self._latent = latent
        self.conv = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(channels * h * w + v, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> torch.Tensor:
        return self.mlp(torch.cat([self.conv(obs["window"].float()), obs["vec"]], dim=-1))


class TagCriticEncoder(BaseCriticEncoder):
    def __init__(self, global_state_space, channels: int = 16, critic_dim: int = 64, **kwargs) -> None:
        super().__init__()
        c, h, w = global_state_space.shape
        self._dim = critic_dim
        self.net = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(channels * h * w, critic_dim), nn.ReLU())

    @property
    def output_dim(self) -> int:
        return self._dim

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state.float())


class TagPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class TagValue(BaseValue):
    """``in_dim`` = core features + critic features (``build_model`` adds them up)."""

    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

Create `examples/predator_prey/models.py`:

```python
"""Model parts for predator_prey: an MLP over a flat float observation and a categorical policy.
The same classes serve every role: sizes come from the role's spaces (injected by ``build_model``)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class PPEncoder(BaseEncoder):
    def __init__(self, observation_space, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], latent), nn.Tanh(),
                                 nn.Linear(latent, latent), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class PPPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class PPValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.Tanh(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

Create `examples/coop_buttons/models.py`:

```python
"""Model parts for coop_buttons: an MLP over a flat float observation and a categorical policy.
The same classes serve every role: sizes come from the role's spaces (injected by ``build_model``)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class CoopEncoder(BaseEncoder):
    def __init__(self, observation_space, latent: int = 64, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], latent), nn.Tanh(),
                                 nn.Linear(latent, latent), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CoopPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.logits = nn.Linear(in_dim, action_spec.groups[0].nvec[0])

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.logits(features))


class CoopValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.Tanh(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

- [ ] **Step 7: Example configs**

Create `configs/examples/team_tag.yaml`:

```yaml
# Team tag: 2v2, each seat is its own bot with a local window; the value sees the whole map
# (global_state) through the critic encoder. Frozen bots stay live and share the team reward.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.team_tag.game.TeamTagGame"
  kwargs: {size: 7, view_radius: 2, max_steps: 30, tag_reward: 0.2, with_global_state: true}

networks:
  encoder_class: "examples.team_tag.models.TagEncoder"
  critic_encoder_class: "examples.team_tag.models.TagCriticEncoder"   # needs with_global_state: true
  core: null
  policy_class: "examples.team_tag.models.TagPolicy"
  value_class: "examples.team_tag.models.TagValue"
  kwargs: {}

algorithm:
  gamma: 0.97
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 500000
  seed: null

matchmaking:
  mode: self_play
  teammates: self           # homogeneous teams
  latest_prob: 0.8

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

Create `configs/examples/predator_prey.yaml`:

```yaml
# Predator and prey: 1 hunter vs 2 prey. The roles have different observation and action spaces,
# so each role has its own agent (agents.<id>.roles). With mode self_play the owner's opponents
# come from the other agent, because the owner cannot play the other role.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.predator_prey.game.PredatorPreyGame"
  kwargs: {size: 7, max_steps: 30, catch_radius: 1}

networks:
  encoder_class: "examples.predator_prey.models.PPEncoder"
  core: null
  policy_class: "examples.predator_prey.models.PPPolicy"
  value_class: "examples.predator_prey.models.PPValue"
  kwargs: {}

algorithm:
  gamma: 0.97
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 400000
  seed: null

matchmaking:
  mode: self_play
  latest_prob: 0.8

agents:
  hunter: {roles: [hunter]}
  prey: {roles: [prey]}

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

Create `configs/examples/coop_buttons.yaml`:

```yaml
# Coop buttons: one team of two seats, no opponent (outcome kind "score"). Two agents with
# teammates: mixed, so teams are sometimes mixed; ratings.json gets a cross-play table.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.coop_buttons.game.CoopButtonsGame"
  kwargs: {size: 5, max_steps: 40, button_reward: 0.05}

networks:
  encoder_class: "examples.coop_buttons.models.CoopEncoder"
  core: null
  policy_class: "examples.coop_buttons.models.CoopPolicy"
  value_class: "examples.coop_buttons.models.CoopValue"
  kwargs: {}

algorithm:
  gamma: 0.97
  vtrace_lambda: 0.95
  entropy_coeff: 0.01
  learning_rate: 1.0e-3
  lr_schedule: "constant"

rollout:
  chunk_length: 32
  num_workers: 2
  envs_per_worker: 16
  weight_sync_interval_sec: 1.0
  torch_threads: 1

learner:
  device: "auto"
  queue_size: 32
  batch_chunks: 8

training:
  total_timesteps: 400000
  seed: null

matchmaking:
  mode: self_play
  teammates: mixed
  teammate_self_prob: 0.5
  latest_prob: 0.8

agents:
  coop_a: {}
  coop_b: {}

checkpoint:
  interval: 100           # train steps between snapshots
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

- [ ] **Step 8: Run the new tests**

Run: `.venv/bin/python -m pytest tests/contract/test_demo_team_tag.py tests/contract/test_demo_predator_prey.py tests/contract/test_demo_coop_buttons.py tests/integration/test_demo_roles_smoke.py -q -rw --durations=5`

Expected: all pass, no warnings; each smoke run takes 10–25 s.

- If `validate` rejects `predator_prey` with "roles with different spaces need separate agents", the `agents` section did not reach `resolve_agent_roles`: check `load_example("predator_prey").agent_roles("hunter") == ["hunter"]`.
- If `test_coop_buttons_with_mixed_teammates_forms_a_cross_play_table` finds no `coop2` key, print `run.ratings()` and compare with the T5.3 `ratings.json` layout; fix the test's key path only if the contract amendments changed the layout, otherwise fix the producer.

- [ ] **Step 9: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, zero warnings, `All checks passed!`.

- [ ] **Step 10: Commit and push**

```bash
git add examples/team_tag examples/predator_prey examples/coop_buttons configs/examples/team_tag.yaml \
  configs/examples/predator_prey.yaml configs/examples/coop_buttons.yaml tests/contract/test_demo_team_tag.py \
  tests/contract/test_demo_predator_prey.py tests/contract/test_demo_coop_buttons.py tests/integration/test_demo_roles_smoke.py
git commit -m "feat: demo games team_tag, predator_prey and coop_buttons (teams, roles, cooperation)"
git push origin sp2-game-model
```

---

### Task T8.3: Learning tests: fast (units bandit, coop bandit) and slow (criterion 3)

Spec section 3, criterion 3 and section 6 ("Учится").
- **Fast tests** run on every suite run and finish in seconds. They drive the real `RolloutLoop` and one real `APPO` per agent in-process, every chunk through `to_payload`/`from_payload`, and evaluate greedily through `play_lineups`.
  - Units bandit: up to K=8 units, each must pick the arm named in its own features from one shared reward (per-unit credit with `ratio_mode: per_unit`).
  - Coop bandit: two seats of one team, two independent agents; the team scores only when both answer correctly in the same step.
- **Slow tests** train every demo game with the real `colosseum train` on its example config (2 workers, about 3 minutes or less each on 8 cores), load the newest checkpoint read-only (`load_eval_model`), and play it greedily against uniformly random legal players with `play_lineups`. Lineups and thresholds are exactly criterion 3's:

| Game | Lineups (eval) | Threshold |
|---|---|---|
| `coin_grid` | 200 solo episodes of the trained agent, 200 of the random player | mean score ≥ 2 × random mean |
| `tic_tac_toe` | 400 games, trained agent in seat `m % 2`, random player in the other | win rate ≥ 0.80 |
| `unit_harvest` | 200 games, seats alternate | win rate ≥ 0.80 |
| `team_tag` | 200 games, trained agent in every seat of team `m % 2`, random players in the other team | team win rate ≥ 0.80 |
| `tron` | 2p: 200 games, seats alternate; 4p: 400 games, trained agent in seat `m % 4`, three random cycles | 2p win rate ≥ 0.80; 4p first-place share ≥ 0.50 |
| `predator_prey` | 200 games hunter agent vs two random prey; 200 games random hunter vs the prey agent in both prey seats | each ≥ 0.70 |
| `coop_buttons` | 100 episodes of each agent's homogeneous team (A+A, B+B); 100 random+random; the oracle | each ≥ max(5 × random mean, 0.6 × oracle mean); `ratings.json` has a `coop_a+coop_b` cross-play entry |

A "win" is a strictly better team rank (`won()` in `demo_learning.py`): draws and shared first places count as non-wins.

**Threshold for `coop_buttons`** (the spec leaves it to the plan): `threshold = max(5 × random_mean, 0.6 × oracle_mean)`, both measured inside the test on the trained run's env kwargs. `random_mean` is the mean team score of two `RandomPolicy` seats through `play_lineups` (100 episodes); `oracle_mean` is `measure_baselines(episodes=300)["oracle"]` (the scripted coordinating pair, numpy only). With the default env both are measured at random 0.0 and oracle 7.34, so the bar is about 4.4 points per match: the trained team must coordinate joint presses repeatedly, not once by luck.

**Greedy vs random in one match.** `play_lineups(deterministic=True)` would also make `RandomPolicy` deterministic (the mode of a uniform distribution is its first legal action). The tests therefore keep `deterministic=False` and wrap the trained model in `GreedyPolicy`, which plays the model's mode as a point-mass distribution.

**Files:**
- Create:
  - `tests/learning/sp2_bandits.py` (fast-test games and the units-bandit model)
  - `tests/learning/demo_learning.py` (in-process trainer, greedy and scripted wrappers, CLI training, checkpoint loading, outcome helpers; also used by `scripts/units_experiment.py` in T8.4)
  - `docs/superpowers/reports/<YYYY-MM-DD>-sp2-acceptance.md` (draft: measurements and rulings of T8.3/T8.4; `<YYYY-MM-DD>` = `date +%F` on the day this step runs; T8.7 completes it)
- Test:
  - `tests/learning/test_sp2_fast_learning.py`
  - `tests/learning/test_demo_learning_slow.py`
- Delete (if still present after T7.2/T7.3): `tests/learning/test_ttt_slow.py`, `tests/learning/ttt_eval.py` (SP1's slow tic-tac-toe test; its port is `test_tic_tac_toe_beats_random_80_percent`)
- Modify: `pyproject.toml` (`known-first-party` += `demo_learning`, `sp2_bandits`); possibly `configs/examples/{coin_grid,tic_tac_toe,unit_harvest,team_tag,tron,predator_prey,coop_buttons}.yaml` (calibration, Step 9)

**Interfaces:**
- Consumes:
  - `RolloutLoop(worker_id, env_fn, num_envs, chunk_length, agent_ids, agent_roles, model_factories, io, lineups, weight_sync_interval, seed)`, `RolloutLoop.step/sync_weights/close`, `LoopIO(send_chunk, poll_weights)` (T3.4)
  - `APPO(model, config, action_spec, device)`, `.train_step`, `.set_progress`, `.model`, `.policy_version` (T4.2); `AlgorithmConfig(..., ratio_mode=...)` (T1.7)
  - `TrajectoryChunk.to_payload/from_payload`, `WeightPayload.from_model(agent_id, version, model)`, `Lineup`, `SeatAssignment`, `MatchResult`/`TeamResult`/`SeatResult` fields (T3.1)
  - `play_lineups(env_fn=, models=, lineups=, num_envs=, seed=, deterministic=False)`, `summarize(spec, results).text()`, `load_eval_model(config, name, path) -> (model, roles)` (T6.1)
  - `make_distribution`, `UnitsHead`, `PolicyModel`, `PolicyStep`, `EncoderOutput`, `ComposedModel`, `NoCore` (T2.3); `ActionSpec`/`ActionGroup`/`Units.components`/`per_unit_kind` (T1.2, T1.3); tree utilities (T1.1)
  - `env_spec`, `make_env` (T2.4); `load_config`, `RESOLVED_CONFIG_FILE`; `cli_runner.run_in_session`, `REPO_ROOT`
  - `game_helpers.RandomPolicy(role)`, `game_helpers.make_test_model(role, core, hidden)` (T2.3)
  - the `slow` marker, pytest-timeout, the `restore_global_rng` fixture (SP1)
  - the demo games and configs of T8.1/T8.2, `examples.tic_tac_toe` + `configs/examples/tic_tac_toe.yaml` (T7.2/T7.3)
- Produces:
  - `sp2_bandits`: `UnitsBandit(max_units=8, arms=4)`, `CoopBandit(arms=4)`, `make_units_bandit_model(max_units=8, arms=4, hidden=32)`
  - `demo_learning`: `LEARNING_SEED` (env `COLOSSEUM_LEARNING_SEED`, default 0), `EVAL_SEED`, `random_model(role)`, `point_params(spec, actions)`, `GreedyPolicy(inner, action_spec)`, `ScriptedPolicy(fn, action_spec)`, `AgentSetup(roles, model_fn, config, action_spec)`, `train_in_process(...) -> int`, `TrainedRun(root, config, elapsed)` with `open(root)`, `env_fn()`, `records(kind)`, `env_steps_per_sec()`, `train_cmd(config, run_parent, name, sets)`, `train_example(name, tmp_path, sets=None, timeout=900.0)`, `newest_checkpoint(root, agent_id)`, `load_agent(run, agent_id)`, `greedy_agent(run, agent_id)`, `play(env_fn, models, lineups)`, `rotating_lineups(layout, num_seats, agent, others, num_matches)`, `team_lineups(team_seats, layout, agent, others, num_matches)`, `team_of`, `won`, `win_rate`, `mean_team_score`

- [ ] **Step 1: Check that the new basenames are free**

```bash
ls tests/learning
find tests -name "sp2_bandits.py" -o -name "demo_learning.py" -o -name "test_sp2_fast_learning.py" -o -name "test_demo_learning_slow.py"
```

Expected: the `find` prints nothing. If `tests/learning/test_ttt_slow.py` or `tests/learning/ttt_eval.py` still exist, check that nothing else imports them (`grep -rn "ttt_eval" tests scripts`) and delete them with `git rm` in this task: they test the SP1 env API, and the port follows in Step 5.

- [ ] **Step 2: Write the fast learning tests**

Create `tests/learning/test_sp2_fast_learning.py`:

```python
"""Fast SP2 learning checks (spec section 3, criterion 3, and section 6): a Units bandit that needs
per-unit credit and a cooperative bandit that needs both seats right at once."""
from __future__ import annotations

import time

import pytest

from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.types import Lineup, SeatAssignment
from demo_learning import AgentSetup, GreedyPolicy, mean_team_score, play, train_in_process
from game_helpers import make_test_model
from sp2_bandits import CoopBandit, UnitsBandit, make_units_bandit_model

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def _solo(agent: str, n: int) -> list[Lineup]:
    return [Lineup("solo", [SeatAssignment(agent)]) for _ in range(n)]


def test_units_bandit_is_solved_with_per_unit_ratios():
    role = UnitsBandit().spec.roles["player"]
    action_spec = ActionSpec.from_space(role.action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003, ratio_mode="per_unit")

    def solved(models) -> bool:
        greedy = GreedyPolicy(models["agent_0"], action_spec)
        return mean_team_score(play(UnitsBandit, {"g": greedy}, _solo("g", 64)), "g") >= 0.95

    start = time.monotonic()
    updates = train_in_process(
        env_fn=UnitsBandit,
        agents={"agent_0": AgentSetup(["player"], make_units_bandit_model, config, action_spec)},
        lineups=_solo("agent_0", 16), max_updates=300, solved=solved)
    assert updates != -1, "units bandit not solved within 300 updates (random play scores 0.25)"
    assert time.monotonic() - start < 60


def test_coop_bandit_two_agents_learn_their_joint_answer():
    role = CoopBandit().spec.roles["player"]
    action_spec = ActionSpec.from_space(role.action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003)
    setup = AgentSetup(["player"], lambda: make_test_model(role, "none", hidden=32), config, action_spec)
    lineups = [Lineup("coop2", [SeatAssignment("a"), SeatAssignment("b")]) for _ in range(16)]

    def solved(models) -> bool:
        greedy = {name: GreedyPolicy(models[name], action_spec) for name in ("a", "b")}
        results = play(CoopBandit, greedy, [Lineup("coop2", [SeatAssignment("a"), SeatAssignment("b")])
                                            for _ in range(64)])
        return mean_team_score(results, "a") >= 0.95

    start = time.monotonic()
    updates = train_in_process(env_fn=CoopBandit, agents={"a": setup, "b": setup}, lineups=lineups,
                               max_updates=300, solved=solved)
    assert updates != -1, "coop bandit not solved within 300 updates (random play scores 1/16)"
    assert time.monotonic() - start < 60
```

Run: `.venv/bin/python -m pytest tests/learning/test_sp2_fast_learning.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'demo_learning'`.

- [ ] **Step 3: Bandit games**

Create `tests/learning/sp2_bandits.py`:

```python
"""One-step games for the fast SP2 learning tests (spec section 6, "Учится": fast tests)."""
from __future__ import annotations

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.specs import ActionSpec
from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.envs.spaces import Units
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.heads import UnitsHead, make_distribution


class UnitsBandit(MultiAgentEnv):
    """Solo, one step. Between 1 and ``max_units`` units exist (random slots); unit ``u`` sees a
    one-hot context ``c_u`` in ``{0..arms-1}`` and should pick arm ``c_u``. The reward is the share
    of existing units that picked their own arm, so every unit has to get its own decision right
    from one shared scalar (per-unit credit). Random play scores ``1 / arms``."""

    def __init__(self, max_units: int = 8, arms: int = 4) -> None:
        self.max_units, self.arms = max_units, arms
        obs_space = gymnasium.spaces.Dict([
            ("units", gymnasium.spaces.Box(0.0, 1.0, (max_units, arms), np.float32)),
            ("unit_mask", gymnasium.spaces.MultiBinary(max_units)),
        ])
        self.spec = GameSpec.solo(obs_space, Units(max_units, gymnasium.spaces.Discrete(arms)))
        self._rng = np.random.default_rng()
        self._alive = np.zeros(max_units, bool)
        self._ctx = np.zeros(max_units, np.int64)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        count = int(self._rng.integers(1, self.max_units + 1))
        self._alive = self._rng.permutation(self.max_units) < count
        self._ctx = self._rng.integers(self.arms, size=self.max_units)
        units = np.zeros((self.max_units, self.arms), np.float32)
        units[self._alive, self._ctx[self._alive]] = 1.0
        obs = {"units": units, "unit_mask": self._alive.astype(np.int8)}
        mask = {"unit": self._alive.copy(), "action": np.ones((self.max_units, self.arms), bool)}
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask})

    def step(self, actions: dict) -> StepResult:
        picked = np.asarray(actions[0], np.int64).reshape(self.max_units)
        reward = float((picked[self._alive] == self._ctx[self._alive]).mean())
        return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)


class CoopBandit(MultiAgentEnv):
    """One team of two seats, one step. Both seats see the same one-hot context ``c`` in
    ``{0..arms-1}`` plus their own seat id. The team scores 1 only if seat 0 picks ``c`` and seat 1
    picks ``arms - 1 - c`` in the same step (both seats get that reward), so neither seat can be
    rewarded without the other. Random play scores ``1 / arms**2``."""

    def __init__(self, arms: int = 4) -> None:
        self.arms = arms
        obs_space = gymnasium.spaces.Box(0.0, 1.0, (arms + 2,), np.float32)
        self.spec = GameSpec.teams_of([2], obs_space, gymnasium.spaces.Discrete(arms))
        self._rng = np.random.default_rng()
        self._ctx = 0

    def _obs(self, seat: int) -> np.ndarray:
        obs = np.zeros(self.arms + 2, np.float32)
        obs[self._ctx] = 1.0
        obs[self.arms + seat] = 1.0
        return obs

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._ctx = int(self._rng.integers(self.arms))
        return StepResult(acting={0, 1}, obs={s: self._obs(s) for s in (0, 1)})

    def step(self, actions: dict) -> StepResult:
        hit = int(actions[0]) == self._ctx and int(actions[1]) == self.arms - 1 - self._ctx
        reward = 1.0 if hit else 0.0
        return StepResult(acting=set(), obs={}, rewards={0: reward, 1: reward}, episode_over=True)


class _UnitsEncoder(BaseEncoder):
    """Per-unit MLP (the unit's own context reaches its own head through ``aux``)."""

    def __init__(self, arms: int, hidden: int) -> None:
        super().__init__()
        self._hidden = hidden
        self.unit_net = nn.Sequential(nn.Linear(arms, hidden), nn.Tanh())

    @property
    def latent_dim(self) -> int:
        return self._hidden

    def forward(self, obs: dict) -> EncoderOutput:
        units = self.unit_net(obs["units"])
        m = obs["unit_mask"].to(units.dtype).unsqueeze(-1)
        pooled = (units * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return EncoderOutput(latent=pooled, aux={"units": units})


class _UnitsPolicy(BasePolicy):
    def __init__(self, action_spec: ActionSpec, hidden: int) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.head = UnitsHead(action_spec.groups[0], hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        return make_distribution(self.action_spec, self.head(aux["units"]))


class _Value(BaseValue):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_units_bandit_model(max_units: int = 8, arms: int = 4, hidden: int = 32) -> ComposedModel:
    spec = UnitsBandit(max_units, arms).spec
    action_spec = ActionSpec.from_space(spec.roles["player"].action_space)
    return ComposedModel(_UnitsEncoder(arms, hidden), NoCore(hidden), _UnitsPolicy(action_spec, hidden),
                         _Value(hidden))
```

Measured with numpy (prototype): `UnitsBandit` random play scores 0.23 (expected 0.25), the oracle 1.0; `CoopBandit` random play 0.067 (expected 1/16).

- [ ] **Step 4: Learning support module**

Create `tests/learning/demo_learning.py`:

```python
"""Support for the SP2 learning tests (T8.3) and ``scripts/units_experiment.py`` (T8.4).

- ``train_in_process``: the real ``RolloutLoop`` and ``APPO`` in one process, every chunk through
  ``to_payload``/``from_payload`` (fast tests);
- ``GreedyPolicy`` / ``ScriptedPolicy``: wrap a model's mode, or a numpy per-observation function,
  as a point-mass distribution. A greedy agent and the stochastic ``RandomPolicy`` can then meet in
  one ``play_lineups(..., deterministic=False)`` (``deterministic=True`` would make the random
  player always pick its first legal action);
- ``train_example`` / ``load_agent``: ``colosseum train`` on an example config, then the newest
  checkpoint of an agent, read-only through ``load_eval_model``;
- ``won`` / ``win_rate`` / ``mean_team_score``: outcome helpers over ``MatchResult``.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cli_runner import REPO_ROOT, run_in_session
from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, ColosseumConfig, load_config
from colosseum.core.registry import env_spec, make_env
from colosseum.core.run_dir import RESOLVED_CONFIG_FILE
from colosseum.core.specs import ActionGroup, ActionSpec
from colosseum.core.tree import tree_get, tree_index, tree_leaves, tree_map, tree_stack, tree_to_numpy, tree_to_torch
from colosseum.core.types import Lineup, MatchResult, SeatAssignment, TrajectoryChunk, WeightPayload
from colosseum.envs.game import MultiAgentEnv, RoleSpec
from colosseum.eval import load_eval_model, play_lineups
from colosseum.networks.heads import make_distribution
from colosseum.networks.model import PolicyModel, PolicyStep
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from game_helpers import RandomPolicy

# Seed of the slow tests; the calibration in T8.3 reruns them with 1 and 2.
LEARNING_SEED = int(os.environ.get("COLOSSEUM_LEARNING_SEED", "0"))
EVAL_SEED = 12345
POINT_LOGIT = 1.0e4          # logit gap of a point-mass categorical
POINT_LOG_STD = -20.0        # log std of a point-mass Gaussian
_CKPT_RE = re.compile(r"ckpt_v(\d+)")


# ----- models ---------------------------------------------------------------------------------
def random_model(role: RoleSpec) -> PolicyModel:
    """The uniformly random legal player of the SP2 tests."""
    return RandomPolicy(role)


def _point_logits(action: torch.Tensor, n: int) -> torch.Tensor:
    return (F.one_hot(action.long(), n).to(torch.float32) - 1.0) * POINT_LOGIT    # 0 at the action, -1e4 elsewhere


def _point_box(action: torch.Tensor, dim: int) -> dict[str, torch.Tensor]:
    return {"mean": action.to(torch.float32), "log_std": torch.full((dim,), POINT_LOG_STD, device=action.device)}


def _point_units(group: ActionGroup, action: Any) -> dict[str, Any]:
    units, params = group.units, {}
    for i, comp in enumerate(units.components):
        if units.per_unit_kind == "dict":
            a = action[comp.name]
        elif units.per_unit_kind == "multi_discrete":
            a = action[..., i]
        else:
            a = action
        params[comp.name] = _point_logits(a, comp.size) if comp.kind == "discrete" else _point_box(a, comp.size)
    return params


def point_params(spec: ActionSpec, actions: Any) -> Any:
    """``make_distribution`` params that put (numerically) all mass on ``actions`` (a torch tree, batch first)."""
    out: dict[str, Any] = {}
    for g in spec.groups:
        a = tree_get(actions, g.path) if g.path else actions
        if g.kind == "discrete":
            p = _point_logits(a, g.nvec[0])
        elif g.kind == "multi_discrete":
            p = torch.cat([_point_logits(a[:, i], n) for i, n in enumerate(g.nvec)], dim=-1)
        elif g.kind == "box":
            p = _point_box(a, g.box_dim)
        else:
            p = _point_units(g, a)
        if not spec.is_dict:
            return p
        node = out
        for key in g.path[:-1]:
            node = node.setdefault(key, {})
        node[g.path[-1]] = p
    return out


def _point_dist(spec: ActionSpec, actions: Any, action_mask: Any):
    dist = make_distribution(spec, point_params(spec, actions))
    return dist if action_mask is None else dist.apply_mask(action_mask)


class GreedyPolicy(PolicyModel):
    """Plays the mode of ``inner`` (with ``inner``'s state) as a point-mass distribution."""

    def __init__(self, inner: PolicyModel, action_spec: ActionSpec) -> None:
        super().__init__()
        self.inner = inner
        self.action_spec = action_spec

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu"):
        return self.inner.initial_state(batch_size, device)

    def reset_state(self, state, done):
        return self.inner.reset_state(state, done)

    @property
    def is_stateful(self) -> bool:
        return self.inner.is_stateful

    def step(self, obs, state, action_mask=None) -> PolicyStep:
        out = self.inner.step(obs, state, action_mask)
        return PolicyStep(_point_dist(self.action_spec, out.dist.mode(), action_mask), out.state)

    def unroll(self, *args, **kwargs):
        raise NotImplementedError("GreedyPolicy is for evaluation only")


class ScriptedPolicy(PolicyModel):
    """A numpy function ``fn(obs) -> action`` (one observation, env action format) as a stateless model."""

    def __init__(self, fn: Callable[[Any], Any], action_spec: ActionSpec) -> None:
        super().__init__()
        self.fn = fn
        self.action_spec = action_spec

    def step(self, obs, state, action_mask=None) -> PolicyStep:
        np_obs = tree_to_numpy(obs)
        batch = tree_leaves(np_obs)[0].shape[0]
        actions = [tree_map(np.asarray, self.fn(tree_index(np_obs, b))) for b in range(batch)]
        return PolicyStep(_point_dist(self.action_spec, tree_to_torch(tree_stack(actions)), action_mask), state)

    def unroll(self, *args, **kwargs):
        raise NotImplementedError("ScriptedPolicy is for evaluation only")


# ----- in-process training (fast tests) -------------------------------------------------------
@dataclass
class AgentSetup:
    roles: list[str]
    model_fn: Callable[[], PolicyModel]
    config: AlgorithmConfig
    action_spec: ActionSpec


def train_in_process(*, env_fn: Callable[[], MultiAgentEnv], agents: Mapping[str, AgentSetup],
                     lineups: Sequence[Lineup], chunk_length: int = 16, batch_chunks: int = 4,
                     max_updates: int = 300, solved: Callable[[dict[str, PolicyModel]], bool],
                     check_every: int = 10, seed: int = 0) -> int:
    """Collect with one ``RolloutLoop`` (one env per lineup) and train one ``APPO`` per agent.
    Every update trains every agent on exactly ``batch_chunks`` of its chunks. Returns the number
    of updates after which ``solved(models)`` first held (checked every ``check_every``), or -1."""
    torch.manual_seed(seed)
    algos = {a: APPO(s.model_fn(), s.config, s.action_spec, device="cpu") for a, s in agents.items()}
    pending: dict[str, list[TrajectoryChunk]] = {a: [] for a in agents}
    latest = {a: WeightPayload.from_model(a, algo.policy_version, algo.model) for a, algo in algos.items()}
    io = LoopIO(send_chunk=lambda c: pending[c.agent_id].append(TrajectoryChunk.from_payload(c.to_payload())),
                poll_weights=lambda agent_id: latest[agent_id])
    loop = RolloutLoop(worker_id=0, env_fn=env_fn, num_envs=len(lineups), chunk_length=chunk_length,
                       agent_ids=list(agents), agent_roles={a: s.roles for a, s in agents.items()},
                       model_factories={a: s.model_fn for a, s in agents.items()}, io=io, lineups=list(lineups),
                       weight_sync_interval=0.0, seed=seed)
    try:
        for update in range(1, max_updates + 1):
            while any(len(chunks) < batch_chunks for chunks in pending.values()):
                loop.step()
            for agent_id, algo in algos.items():
                batch = pending[agent_id][:batch_chunks]
                del pending[agent_id][:batch_chunks]
                algo.set_progress(update / max_updates)
                algo.train_step(batch)
                latest[agent_id] = WeightPayload.from_model(agent_id, algo.policy_version, algo.model)
            loop.sync_weights()
            if update % check_every == 0 and solved({a: algo.model for a, algo in algos.items()}):
                return update
    finally:
        loop.close()
    return -1


# ----- CLI training and checkpoints (slow tests, units experiment) ---------------------------
@dataclass
class TrainedRun:
    root: Path
    config: ColosseumConfig
    elapsed: float

    @classmethod
    def open(cls, root: Path) -> TrainedRun:
        return cls(Path(root), load_config(Path(root) / RESOLVED_CONFIG_FILE), 0.0)

    def env_fn(self) -> Callable[[], MultiAgentEnv]:
        return functools.partial(make_env, self.config)

    def records(self, kind: str) -> list[dict]:
        lines = (self.root / "metrics.jsonl").read_text().splitlines()
        return [r for r in map(json.loads, filter(str.strip, lines)) if r["kind"] == kind]

    def env_steps_per_sec(self) -> float:
        system = self.records("system")
        if len(system) < 2 or system[-1]["ts"] <= system[0]["ts"]:
            return float("nan")
        return (system[-1]["env_steps"] - system[0]["env_steps"]) / (system[-1]["ts"] - system[0]["ts"])


def train_cmd(config: Path, run_parent: Path, name: str, sets: Mapping[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(config),
           "--set", f"run.dir={run_parent}", "--set", f"run.name={name}"]
    for key, value in sets.items():
        cmd += ["--set", f"{key}={value}"]
    return cmd


def train_example(name: str, tmp_path: Path, sets: Mapping[str, Any] | None = None,
                  timeout: float = 900.0) -> TrainedRun:
    """``colosseum train -c configs/examples/<name>.yaml`` with ``training.seed=LEARNING_SEED``."""
    config = REPO_ROOT / "configs" / "examples" / f"{name}.yaml"
    run_parent = tmp_path / "runs"
    cmd = train_cmd(config, run_parent, name, {"training.seed": LEARNING_SEED, **(sets or {})})
    start = time.monotonic()
    proc = run_in_session(cmd, timeout)
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr[-3000:]
    run = TrainedRun.open(run_parent / name)
    run.elapsed = elapsed
    return run


def newest_checkpoint(root: Path, agent_id: str) -> Path:
    agent_dir = Path(root) / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints for {agent_id} in {root}"
    return versions[max(versions)]


def load_agent(run: TrainedRun, agent_id: str) -> tuple[PolicyModel, RoleSpec, ActionSpec]:
    """The newest checkpoint of ``agent_id`` (architecture and roles from its ``meta.json``)."""
    model, roles = load_eval_model(run.config, agent_id, newest_checkpoint(run.root, agent_id))
    model.eval()
    role = env_spec(run.config).roles[roles[0]]
    return model, role, ActionSpec.from_space(role.action_space)


def greedy_agent(run: TrainedRun, agent_id: str) -> tuple[GreedyPolicy, RoleSpec]:
    model, role, action_spec = load_agent(run, agent_id)
    return GreedyPolicy(model, action_spec), role


# ----- evaluation -----------------------------------------------------------------------------
def play(env_fn: Callable[[], MultiAgentEnv], models: Mapping[str, PolicyModel],
         lineups: Sequence[Lineup]) -> list[MatchResult]:
    return play_lineups(env_fn=env_fn, models=models, lineups=lineups, num_envs=8, seed=EVAL_SEED)


def rotating_lineups(layout: str, num_seats: int, agent: str, others: str, num_matches: int) -> list[Lineup]:
    """``agent`` in seat ``m % num_seats`` of match ``m``, ``others`` in every other seat (FFA, 1v1)."""
    return [Lineup(layout, [SeatAssignment(agent if s == m % num_seats else others) for s in range(num_seats)])
            for m in range(num_matches)]


def team_lineups(team_seats: Sequence[Sequence[int]], layout: str, agent: str, others: str,
                 num_matches: int) -> list[Lineup]:
    """``agent`` in every seat of team ``m % T`` of match ``m``, ``others`` in the other teams."""
    num_seats = sum(len(t) for t in team_seats)
    lineups = []
    for m in range(num_matches):
        mine = set(team_seats[m % len(team_seats)])
        lineups.append(Lineup(layout, [SeatAssignment(agent if s in mine else others) for s in range(num_seats)]))
    return lineups


def team_of(result: MatchResult, agent: str) -> int:
    teams = {s.team for s in result.seats if s.agent_id == agent}
    assert len(teams) == 1, f"{agent} plays teams {teams} in {result.match_id}"
    return teams.pop()


def won(result: MatchResult, agent: str) -> bool:
    """``agent``'s team is strictly first (a win in ``wdl``, a sole first place in ``rank``)."""
    ranks = {t.team: t.rank for t in result.teams}
    mine = team_of(result, agent)
    return all(ranks[mine] < r for team, r in ranks.items() if team != mine)


def win_rate(results: Sequence[MatchResult], agent: str) -> float:
    return sum(won(r, agent) for r in results) / len(results)


def mean_team_score(results: Sequence[MatchResult], agent: str) -> float:
    scores = [next(t.score for t in r.teams if t.team == team_of(r, agent)) for r in results]
    return float(np.mean(scores))
```

Add `"demo_learning"` and `"sp2_bandits"` to `known-first-party` in `pyproject.toml`.

- [ ] **Step 5: Run the fast learning tests**

Run: `.venv/bin/python -m pytest tests/learning/test_sp2_fast_learning.py -v --durations=0`

Expected: both PASS, each in under 20 s.

If one fails with "not solved":
- print the greedy score every 10 updates inside `solved` (temporarily) and the train metrics of the last update (`algo.train_step` returns them): `deciders_valid_mean` must be about 1 + the number of live units on the units bandit; `clip_fraction` must be above 0 at some point; `value_loss` must fall;
- a score stuck at 0.25 (units) or 1/16 (coop) means rewards do not reach the act slots or the per-unit heads do not get the per-unit features: a producer bug (T3.x, T4.2) or in `make_units_bandit_model`, not a test to relax;
- do not raise `max_updates` above 300, lower 0.95, or change the bandits without a ruling (Step 9 format).

- [ ] **Step 6: Write the slow tests**

Create `tests/learning/test_demo_learning_slow.py`:

```python
"""Slow learning checks of SP2 (spec section 3, criterion 3): each demo game is trained with
``colosseum train`` on its example config (about 3 minutes or less on 8 cores, 2 workers), then
the newest checkpoint plays greedily against uniformly random legal players through the in-process
eval API (``play_lineups``). Thresholds are the spec's; changing one needs a ruling with
measurements (T8.3 Step 9), never a silent edit."""
from __future__ import annotations

import json

import pytest

from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import summarize
from demo_learning import (
    greedy_agent,
    mean_team_score,
    play,
    random_model,
    rotating_lineups,
    team_lineups,
    train_example,
    win_rate,
)
from examples.coop_buttons.game import measure_baselines

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1200), pytest.mark.usefixtures("restore_global_rng")]
TWO_WORKERS = {"rollout.num_workers": 2}


def _report(name: str, run, **numbers) -> None:
    text = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in numbers.items())
    print(f"[{name}] training {run.elapsed:.0f}s, {run.env_steps_per_sec():.0f} env steps/s; {text}")


def test_coin_grid_scores_at_least_twice_random(tmp_path):
    run = train_example("coin_grid", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    models = {"trained": trained, "random": random_model(role)}
    solo = [[Lineup("solo", [SeatAssignment(key)]) for _ in range(200)] for key in ("trained", "random")]
    score = mean_team_score(play(run.env_fn(), models, solo[0]), "trained")
    baseline = mean_team_score(play(run.env_fn(), models, solo[1]), "random")
    _report("coin_grid", run, score=score, random=baseline, ratio=score / baseline)
    assert baseline > 0 and score >= 2.0 * baseline


def test_tic_tac_toe_beats_random_80_percent(tmp_path):
    run = train_example("tic_tac_toe", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    (layout,) = env_spec(run.config).layouts
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   rotating_lineups(layout, 2, "trained", "random", 400))
    rate = win_rate(results, "trained")
    _report("tic_tac_toe", run, win_rate=rate)
    assert rate >= 0.80


def test_unit_harvest_beats_random_80_percent(tmp_path):
    run = train_example("unit_harvest", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   rotating_lineups("2p", 2, "trained", "random", 200))
    rate = win_rate(results, "trained")
    _report("unit_harvest", run, win_rate=rate)
    assert rate >= 0.80


def test_team_tag_team_beats_random_team_80_percent(tmp_path):
    run = train_example("team_tag", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    spec = env_spec(run.config)
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   team_lineups(spec.teams("2v2"), "2v2", "trained", "random", 200))
    rate = win_rate(results, "trained")
    _report("team_tag", run, win_rate=rate)
    assert rate >= 0.80


def test_tron_wins_2p_and_takes_first_place_in_4p(tmp_path):
    run = train_example("tron", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    models = {"trained": trained, "random": random_model(role)}
    rate_2p = win_rate(play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "random", 200)), "trained")
    first_4p = win_rate(play(run.env_fn(), models, rotating_lineups("4p", 4, "trained", "random", 400)), "trained")
    _report("tron", run, win_rate_2p=rate_2p, first_place_4p=first_4p)
    assert rate_2p >= 0.80
    assert first_4p >= 0.50


def test_predator_prey_each_role_beats_random_70_percent(tmp_path):
    run = train_example("predator_prey", tmp_path, TWO_WORKERS)
    hunter, hunter_role = greedy_agent(run, "hunter")
    prey, prey_role = greedy_agent(run, "prey")
    models = {"hunter": hunter, "prey": prey,
              "random_hunter": random_model(hunter_role), "random_prey": random_model(prey_role)}

    def lineups(h: str, p: str) -> list[Lineup]:
        return [Lineup("1v2", [SeatAssignment(h), SeatAssignment(p), SeatAssignment(p)]) for _ in range(200)]

    hunter_rate = win_rate(play(run.env_fn(), models, lineups("hunter", "random_prey")), "hunter")
    prey_rate = win_rate(play(run.env_fn(), models, lineups("random_hunter", "prey")), "prey")
    random_rate = win_rate(play(run.env_fn(), models, lineups("random_hunter", "random_prey")), "random_hunter")
    _report("predator_prey", run, hunter_vs_random=hunter_rate, prey_vs_random=prey_rate,
            random_hunter_vs_random_prey=random_rate)
    assert hunter_rate >= 0.70
    assert prey_rate >= 0.70


def test_coop_buttons_homogeneous_teams_beat_the_measured_threshold(tmp_path):
    run = train_example("coop_buttons", tmp_path, TWO_WORKERS)
    a, role = greedy_agent(run, "coop_a")
    b, _ = greedy_agent(run, "coop_b")
    models = {"coop_a": a, "coop_b": b, "random": random_model(role)}

    def team(x: str, y: str, n: int = 100) -> list[Lineup]:
        return [Lineup("coop2", [SeatAssignment(x), SeatAssignment(y)]) for _ in range(n)]

    random_mean = mean_team_score(play(run.env_fn(), models, team("random", "random")), "random")
    oracle_mean = measure_baselines(episodes=300, **run.config.env.kwargs)["oracle"]
    threshold = max(5.0 * random_mean, 0.6 * oracle_mean)
    score_a = mean_team_score(play(run.env_fn(), models, team("coop_a", "coop_a")), "coop_a")
    score_b = mean_team_score(play(run.env_fn(), models, team("coop_b", "coop_b")), "coop_b")
    mixed = play(run.env_fn(), models, team("coop_a", "coop_b"))
    print(summarize(env_spec(run.config), mixed).text())
    cross = json.loads((run.root / "ratings.json").read_text())["coop2"]["cross_play"]
    _report("coop_buttons", run, coop_a=score_a, coop_b=score_b, threshold=threshold, random=random_mean,
            oracle=oracle_mean, cross_play=json.dumps(cross))
    assert score_a >= threshold and score_b >= threshold
    assert cross.get("coop_a+coop_b", {}).get("n", 0) > 0          # the cross-play table was formed in training
```

Check the collection: `.venv/bin/python -m pytest tests/learning -m slow --collect-only -q`

Expected: the seven `test_demo_learning_slow.py` tests (and SP1's other slow tests, e.g. torch.compile ones, if they still exist); no `test_ttt_slow.py`.

- [ ] **Step 7: Run each slow test once and record the numbers**

```bash
nproc; uptime
.venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -v -s 2>&1 | tee /tmp/sp2-slow-seed0.txt
grep -E "^\[|PASSED|FAILED" /tmp/sp2-slow-seed0.txt
```

Expected: `nproc` prints 8, the load average is below 1 before the run; one line per test like `[tron] training 160s, 3100 env steps/s; win_rate_2p=0.930, first_place_4p=0.640`; every test PASSED. The whole file takes about 25 minutes.

- [ ] **Step 8: Check robustness on two more seeds**

```bash
for s in 1 2; do
  COLOSSEUM_LEARNING_SEED=$s .venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -v -s 2>&1 \
    | tee /tmp/sp2-slow-seed$s.txt | grep -E "^\[|PASSED|FAILED"
done
```

Expected: every test PASSED on seeds 1 and 2 too. A threshold that passes on one seed only is not met: go to Step 9 for that game.

- [ ] **Step 9: Calibration and rulings (only for a test that failed in Step 7 or 8)**

Work on one game at a time. Change only that game's `configs/examples/<game>.yaml` (env `kwargs` and training knobs; never the game's class defaults, which the T8.1/T8.2 tests pin), rerun its test on seeds 0, 1, 2 after each change, and stop at the first change that passes all three:

1. **Budget within the time cap.** The cap is 170 s of training: `cap = floor(170 × env_steps_per_s / 10000) × 10000`, with `env_steps_per_s` from the test's printed line. If `training.total_timesteps` is below `cap`, set it to `cap`.
2. **Optimizer.** If the printed metric rose and flattened early, `algorithm.learning_rate: 3.0e-4`; if it was still rising at the end, `algorithm.learning_rate: 2.0e-3`. Then `algorithm.entropy_coeff: 0.003`.
3. **Game size** (env `kwargs` in the config; one change at a time, in this order):
   - `coin_grid`: `size: 6`;
   - `unit_harvest`: `max_steps: 40`, then `size: 7`;
   - `team_tag`: `view_radius: 3`, then `size: 6`;
   - `tron`: `size: 8`;
   - `predator_prey`: `max_steps: 40`; then rerun `test_random_play_is_balanced_and_the_references_win_their_roles` with the new kwargs by hand — random vs random must stay within 0.35–0.75 hunter wins, otherwise revert;
   - `coop_buttons`: `size: 4` (the threshold follows the oracle automatically);
   - `tic_tac_toe`: no size knob; only steps 1–2.

If none passes, write a ruling instead of changing the test. Never edit a threshold in the test without a ruling, and never silently. Ruling options, cheapest first:
- (a) **budget above 3 minutes** (up to 6 minutes of training): the measured time goes into the ruling;
- (b) **a smaller game** than the steps above;
- (c) **a threshold change**: only together with the numbers that justify it (best value on each seed, random baseline, reference-policy value from the T8.1/T8.2 measurements) and flagged to the owner in the report section «Отклонения».

Format (one line in the report draft, section «Решения (rulings) T8.3–T8.4»): `Ruling: <what> — <why, with the measured numbers> — <cost if wrong>`. Example: `Ruling: team_tag budget 600k env steps (215 s, above the 3-minute target) — 500k gives 0.74/0.81/0.77 on seeds 0-2, 600k gives 0.86/0.88/0.84 — if wrong: the slow suite is 35 s longer`.

- [ ] **Step 10: Start the acceptance report draft**

```bash
REPORT=docs/superpowers/reports/$(date +%F)-sp2-acceptance.md; echo $REPORT
```

Create `$REPORT` with this content, filled from `/tmp/sp2-slow-seed{0,1,2}.txt` and Step 9 (Russian; T8.4 adds its sections, T8.7 completes the report):

```markdown
# SP2: приёмка по спеке §3

Черновик. T8.3 записал замеры «учится» и решения калибровки, T8.4 — эксперимент по юнитам и скорость, T8.7 — итог.

## Замеры «учится» (T8.3)

Машина: <вывод `nproc`> ядер, <ОЗУ из `free -g`> ГБ, без GPU. Каждый тест: `colosseum train` на конфиге примера, 2 воркера, затем greedy против случайных легальных игроков через `play_lineups`.

| Тест | Порог | seed 0 | seed 1 | seed 2 | Обучение, с (seed 0) | Шаги сред/с | `total_timesteps` |
|---|---|---|---|---|---|---|---|
| coin_grid | счёт ≥ 2 × случайный | | | | | | |
| tic_tac_toe | ≥ 0.80 побед | | | | | | |
| unit_harvest | ≥ 0.80 побед | | | | | | |
| team_tag | ≥ 0.80 побед команды | | | | | | |
| tron 2p / 4p | ≥ 0.80 / ≥ 0.50 первых мест | | | | | | |
| predator_prey охотник / жертвы | ≥ 0.70 / ≥ 0.70 | | | | | | |
| coop_buttons A / B | ≥ max(5 × случайный, 0.6 × оракул) | | | | | | |

Быстрые тесты: `test_sp2_fast_learning.py` — <длительности из Step 5>.

## Решения (rulings) T8.3–T8.4

<по одной строке на решение из Step 9; если решений не было — «Нет: все пороги выполнены на конфигах примеров без изменений.»>
```

Every cell holds the printed value (for `coop_buttons` both scores and the threshold, e.g. `5.9 / 6.1 (≥ 4.4)`).

- [ ] **Step 11: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, zero warnings; the slow tests are deselected.

- [ ] **Step 12: Commit and push**

```bash
git add tests/learning pyproject.toml configs/examples docs/superpowers/reports/*-sp2-acceptance.md
git commit -m "test: SP2 learning checks (units and coop bandits fast; every demo game vs random, slow)"
git push origin sp2-game-model
```

If Step 1 deleted the SP1 files, `git add tests/learning` stages the deletions too (`git status` shows them as `deleted:`).

---

### Task T8.4: Units experiment, throughput "after", `global_state` size

Spec section 3, criteria 4 and 5; section 8, risks "Дефолт `unit_trace`", "Деревья и dict на место в Python замедлят воркер", "Размер `global_state`"; block 10 "Замеры".

**Files:**
- Create:
  - `scripts/units_experiment.py`
  - `scripts/measure_global_state.py`
  - `docs/benchmarks/after-sp2.json`, `docs/benchmarks/units-experiment.json`, `docs/benchmarks/global-state-team-tag.json` (raw data written by the scripts)
- Modify:
  - `docs/benchmarks.md` (Russian; new sections «После SP2», «Эксперимент по юнитам (SP2, критерий 4)», «Размер `global_state` (team_tag)»)
  - `docs/superpowers/reports/<date>-sp2-acceptance.md` (sections «Эксперимент по юнитам», «Скорость», the `unit_trace` ruling)
  - only if the ruling changes the default: `src/colosseum/algorithms/appo.py` (resolution of `unit_trace: auto` with Units) and `configs/examples/unit_harvest.yaml` comments
- Test (only if no test pins the `auto` resolution yet, or the default changes): `tests/unit/test_unit_trace_default.py`

**Interfaces:**
- Consumes:
  - `demo_learning`: `TrainedRun`, `train_cmd`, `greedy_agent`, `ScriptedPolicy`, `random_model`, `play`, `rotating_lineups`, `team_of`, `win_rate` (T8.3)
  - `examples.unit_harvest.game.scripted_action`, `configs/examples/unit_harvest.yaml` (T8.1); `configs/examples/team_tag.yaml` with `with_global_state` and `critic_encoder_class` (T8.2)
  - `RolloutLoop`, `LoopIO`, `APPO`, `TrajectoryChunk`, `Lineup`, `SeatAssignment`; `colosseum.core.roles.resolve_agent_roles`, `agent_role_spec`; `colosseum.core.registry.build_model`, `env_spec`, `make_env`; `ColosseumConfig.get_agent_config`; `colosseum.core.tree.tree_leaves`
  - train metrics of APPO v2 in `metrics.jsonl` `train` records: `clip_fraction`, `clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `log_rho_joint_abs_mean`, `c_clip_frac`, `deciders_valid_mean` (T4.2, T5.3)
  - `scripts/bench_throughput.py` on the new tic-tac-toe (T7.3) and the SP1 numbers in `docs/benchmarks.md` («После SP1»: 4641 / 9444 / 10664 env steps/s at 1 / 2 / 4 workers)
  - `game_helpers.UnitsGame`, `make_test_model`; `ObsSpec.allocate`, `ActionSpec.allocate_actions/full_mask/num_deciders`, `SLOT_ACT`, `SLOT_BOOT`, `APPO.compute_loss(...)["value_loss"]`
- Produces: `scripts/units_experiment.py` (CLI below; functions `train`, `diagnostics`, `evaluate`, `recommend`), `scripts/measure_global_state.py` (CLI below; function `measure(config, num_chunks) -> dict`), the benchmark sections, the `unit_trace` ruling.

- [ ] **Step 1: The experiment script**

Create `scripts/units_experiment.py`:

```python
"""Units experiment (SP2 spec, section 3, criterion 4).

``unit_harvest`` at K=8 and K=128 units x ``algorithm.ratio_mode`` in {joint, per_unit} x
``algorithm.unit_trace`` in {joint, geo_mean, none}, every run on the same env-step budget.
Each run is a real ``colosseum train`` (subprocess) on ``configs/examples/unit_harvest.yaml``.
Afterwards the newest checkpoint of every run plays greedily (in-process ``play_lineups``):
- against the uniformly random legal player: win rate and mean own score;
- against the scripted reference bot (``examples.unit_harvest.game.scripted_action``): win rate and
  score share own / (own + opponent), which does not saturate when every mode beats random.
Diagnostics are the means over the last quarter of the run's ``train`` records in
``metrics.jsonl``: ``clip_fraction`` (per decider), ``clip_fraction_joint``, ``ess``,
``log_rho_abs_p95``, ``log_rho_joint_abs_mean``, ``c_clip_frac``, ``deciders_valid_mean``.

The script prints a Markdown table and the ``unit_trace`` recommendation of the decision rule
(T8.4): keep ``geo_mean`` unless another trace, with ``ratio_mode=per_unit``, has a K=128 score
share vs scripted at least 0.05 higher and a K=8 share at most 0.05 lower.

Usage (the evaluation imports the test kit, so it puts ``tests/`` and ``tests/learning`` on
``sys.path``)::

    .venv/bin/python scripts/units_experiment.py --steps 400000 --seeds 0 --parallel 2 \
        --json docs/benchmarks/units-experiment.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = REPO_ROOT / "configs" / "examples" / "unit_harvest.yaml"
K_KWARGS = {
    8: {"max_units": 8, "size": 8, "num_resources": 6, "initial_workers": 2, "max_steps": 50},
    128: {"max_units": 128, "size": 16, "num_resources": 16, "initial_workers": 32, "max_steps": 80},
}
RATIO_MODES = ("joint", "per_unit")
UNIT_TRACES = ("joint", "geo_mean", "none")
DIAG_KEYS = ("clip_fraction", "clip_fraction_joint", "ess", "log_rho_abs_p95", "log_rho_joint_abs_mean",
             "c_clip_frac", "deciders_valid_mean")
DEFAULT_TRACE = "geo_mean"
MARGIN = 0.05


def run_name(k: int, ratio: str, trace: str, seed: int) -> str:
    return f"k{k}-{ratio}-{trace}-s{seed}"


def train(k: int, ratio: str, trace: str, seed: int, *, steps: int, workers: int, run_dir: Path) -> dict:
    from demo_learning import train_cmd

    name = run_name(k, ratio, trace, seed)
    sets = {"training.total_timesteps": steps, "training.seed": seed, "rollout.num_workers": workers,
            "algorithm.ratio_mode": ratio, "algorithm.unit_trace": trace, "env.kwargs": json.dumps(K_KWARGS[k])}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()
    with open(run_dir / f"{name}.out", "w") as out:
        proc = subprocess.run(train_cmd(CONFIG, run_dir, name, sets), cwd=REPO_ROOT, env=env, stdout=out,
                              stderr=subprocess.STDOUT, timeout=4 * 3600)
    return {"k": k, "ratio_mode": ratio, "unit_trace": trace, "seed": seed, "name": name,
            "returncode": proc.returncode, "wall_s": time.monotonic() - start}


def diagnostics(run) -> dict:
    train_records = run.records("train")
    tail = train_records[-max(1, len(train_records) // 4):]
    out = {}
    for key in DIAG_KEYS:
        values = [r[key] for r in tail if r.get(key) is not None]
        out[key] = sum(values) / len(values) if values else float("nan")
    out["env_steps_per_s"] = run.env_steps_per_sec()
    out["train_steps"] = train_records[-1]["train_step"] if train_records else 0
    return out


def evaluate(run, matches: int) -> dict:
    from colosseum.core.types import MatchResult
    from demo_learning import ScriptedPolicy, greedy_agent, play, random_model, rotating_lineups, team_of, win_rate
    from examples.unit_harvest.game import scripted_action

    trained, role = greedy_agent(run, "agent_0")
    action_spec = trained.action_spec
    models = {"trained": trained, "random": random_model(role),
              "scripted": ScriptedPolicy(scripted_action, action_spec)}

    def scores(r: MatchResult) -> tuple[float, float]:
        mine = team_of(r, "trained")
        own = next(t.score for t in r.teams if t.team == mine)
        opp = next(t.score for t in r.teams if t.team != mine)
        return own, opp

    vs_random = play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "random", matches))
    vs_scripted = play(run.env_fn(), models, rotating_lineups("2p", 2, "trained", "scripted", matches))
    shares = [own / (own + opp) if own + opp > 0 else 0.5 for own, opp in map(scores, vs_scripted)]
    return {"win_vs_random": win_rate(vs_random, "trained"),
            "score_vs_random": sum(scores(r)[0] for r in vs_random) / len(vs_random),
            "win_vs_scripted": win_rate(vs_scripted, "trained"),
            "share_vs_scripted": sum(shares) / len(shares)}


def recommend(rows: list[dict]) -> tuple[str, str]:
    """The T8.4 decision rule for the ``unit_trace: auto`` default (with Units)."""
    def share(k: int, trace: str) -> float:
        vals = [r["share_vs_scripted"] for r in rows if "share_vs_scripted" in r
                and r["k"] == k and r["ratio_mode"] == "per_unit" and r["unit_trace"] == trace]
        return sum(vals) / len(vals) if vals else float("nan")

    base128, base8 = share(128, DEFAULT_TRACE), share(8, DEFAULT_TRACE)
    better = [t for t in UNIT_TRACES if t != DEFAULT_TRACE
              and share(128, t) >= base128 + MARGIN and share(8, t) >= base8 - MARGIN]
    choice = max(better, key=lambda t: share(128, t)) if better else DEFAULT_TRACE
    detail = "; ".join(f"{t}: K=128 {share(128, t):.3f}, K=8 {share(8, t):.3f}" for t in UNIT_TRACES)
    return choice, detail


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 128], choices=sorted(K_KWARGS))
    parser.add_argument("--ratio-modes", nargs="+", default=list(RATIO_MODES), choices=RATIO_MODES)
    parser.add_argument("--unit-traces", nargs="+", default=list(UNIT_TRACES), choices=UNIT_TRACES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=400_000, help="env-step budget of every run")
    parser.add_argument("--workers", type=int, default=2, help="rollout workers per run")
    parser.add_argument("--parallel", type=int, default=2, help="runs at the same time")
    parser.add_argument("--eval-matches", type=int, default=100, help="matches per opponent")
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/colosseum-units-experiment"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from demo_learning import TrainedRun

    args.run_dir.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(args.k, args.ratio_modes, args.unit_traces, args.seeds))
    print(f"{len(grid)} runs, {args.steps} env steps each, {args.parallel} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        rows = list(pool.map(lambda g: train(*g, steps=args.steps, workers=args.workers, run_dir=args.run_dir), grid))
    for row in rows:
        if row["returncode"] != 0:
            log = args.run_dir / f"{row['name']}.out"
            print(f"ERROR {row['name']}: exit {row['returncode']}, see {log}", file=sys.stderr)
            continue
        run = TrainedRun.open(args.run_dir / row["name"])
        row.update(diagnostics(run))
        row.update(evaluate(run, args.eval_matches))
        print(f"{row['name']}: {json.dumps({k: row[k] for k in row if k not in ('name',)}, default=str)}", flush=True)

    cols = ["k", "ratio_mode", "unit_trace", "seed", "win_vs_random", "score_vs_random", "win_vs_scripted",
            "share_vs_scripted", *DIAG_KEYS, "env_steps_per_s", "wall_s"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(c, "-")) for c in cols) + " |")
    choice, detail = recommend(rows)
    print(f"\nunit_trace recommendation (per_unit, share vs scripted): {choice} ({detail})")
    if args.json:
        args.json.write_text(json.dumps({"steps": args.steps, "workers": args.workers, "k_kwargs": K_KWARGS,
                                         "rows": rows, "recommendation": choice}, indent=2, default=str) + "\n")
    return 1 if any(r["returncode"] != 0 for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Check it starts and parses: `.venv/bin/python scripts/units_experiment.py --help`

Expected: the usage text with `--k`, `--ratio-modes`, `--unit-traces`, `--seeds`, `--steps`, `--workers`, `--parallel`, `--eval-matches`, `--run-dir`, `--json`.

- [ ] **Step 2: Pilot run and the budget**

The budget must fit the machine (8 cores, about 11 GB RAM, no GPU): 12 runs, two at a time (each run = 2 workers + 1 learner + the main process), about one hour of wall time in total.

```bash
rm -rf /tmp/colosseum-units-pilot
.venv/bin/python -m colosseum train -c configs/examples/unit_harvest.yaml --set run.dir=/tmp/colosseum-units-pilot \
  --set run.name=k128 --set training.total_timesteps=60000 \
  --set 'env.kwargs={"max_units": 128, "size": 16, "num_resources": 16, "initial_workers": 32, "max_steps": 80}'
.venv/bin/python - <<'EOF'
import sys; sys.path[:0] = ["tests", "tests/learning"]
from demo_learning import TrainedRun
run = TrainedRun.open("/tmp/colosseum-units-pilot/k128")
rate = run.env_steps_per_sec()
print(f"K=128 env steps/s: {rate:.0f}; budget: {max(200_000, min(1_000_000, int(rate * 480) // 10_000 * 10_000))}")
EOF
free -g
```

Expected: exit 0; a printed rate (prototype estimate: several hundred env steps/s at K=128) and the budget `STEPS` = the env steps one K=128 run makes in about 8 minutes alone, clamped to [200k, 1M]. `free -g` shows at least 6 GB available while nothing runs.

Use one seed (`--seeds 0`): 12 runs × 2 seeds would not fit an hour on this machine. The benchmark section says so. If the decision in Step 4 is within 0.02 of a rule boundary, rerun only the six `per_unit` rows with `--seeds 1` to confirm and report both seeds.

- [ ] **Step 3: Run the experiment**

```bash
STEPS=<budget from Step 2>
nohup .venv/bin/python scripts/units_experiment.py --steps $STEPS --seeds 0 --parallel 2 \
  --json docs/benchmarks/units-experiment.json > /tmp/units-experiment.txt 2>&1 &
```

Watch `/tmp/units-experiment.txt` until it prints the Markdown table and the line `unit_trace recommendation (per_unit, share vs scripted): ...`. Expected: exit 0 (no `ERROR` lines), 12 rows. Typical sanity values: `deciders_valid_mean` about 3–10 at K=8 and 30–130 at K=128; `ess` close to 1 for `geo_mean`/`none` and lower for `joint` at K=128; `win_vs_random` high for every row (the random side rarely deposits).

If a run fails, read `/tmp/colosseum-units-experiment/<name>.out` and `<name>/logs/`. Out of memory (a killed learner) → rerun with `--parallel 1`.

- [ ] **Step 4: The `unit_trace` ruling**

The decision rule (implemented in `recommend()`):
- metric: `share_vs_scripted` (own score / (own + scripted) against the scripted bot, 100 games), with `ratio_mode=per_unit` (the `auto` policy mode with Units);
- keep `geo_mean` (the spec's pre-experiment default) unless another trace has a K=128 share at least **0.05** higher **and** a K=8 share at most 0.05 lower; among those, take the highest K=128 share.

Record the ruling in the report draft, section «Решения (rulings) T8.3–T8.4», with the numbers, e.g.:
`Ruling: unit_trace auto (with Units) = geo_mean — K=128 share vs scripted geo_mean 0.41, joint 0.33, none 0.43 (< +0.05); K=8 0.55 / 0.54 / 0.55 — if wrong: value targets at K=128 slightly worse than the best mode; one-line change in appo`.

Also record what the experiment says about `ratio_mode` (joint vs per_unit at K=128). The spec fixes `ratio_mode: auto = per_unit` with Units; if `joint` is better by more than 0.05 at K=128, do not change the default: add it to the report as an open item for the owner.

- [ ] **Step 5: Apply the ruling in code (only if it is not `geo_mean`) and pin it with a test**

Check whether a test already pins the resolution of `auto`:

```bash
grep -rn "unit_trace" tests/unit | grep -n "auto"
grep -n "geo_mean" src/colosseum/algorithms/appo.py
```

- If the ruling keeps `geo_mean` **and** a test from T4.2 already asserts that `auto` resolves to `geo_mean` with Units: no code change; skip to Step 6.
- Otherwise create `tests/unit/test_unit_trace_default.py` with `CHOSEN` set to the ruling's value:

```python
"""``algorithm.unit_trace: auto`` with Units resolves to the default chosen by the units experiment
(T8.4 ruling; docs/benchmarks.md). Behaviour-level check: the value loss of ``auto`` equals the
chosen mode's and differs from the other modes' on the same chunks and weights."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import tree_map, tree_to_torch
from colosseum.core.types import SLOT_ACT, SLOT_BOOT, TrajectoryChunk
from game_helpers import UnitsGame, make_test_model

CHOSEN = "geo_mean"   # the T8.4 ruling; keep in sync with docs/benchmarks.md
SLOTS = 8


def _chunk(role, seed: int) -> TrajectoryChunk:
    """ACT slots then a BOOT slot; behaviour log-probs 0, so every ratio is the current probability
    (< 1, unclipped) and the three traces give different value targets."""
    rng = np.random.default_rng(seed)

    def fill(a: np.ndarray) -> np.ndarray:
        return (rng.random(a.shape) if a.dtype.kind == "f" else rng.integers(0, 2, a.shape)).astype(a.dtype)

    obs = tree_map(fill, ObsSpec.from_space(role.observation_space).allocate((SLOTS,)))
    spec = ActionSpec.from_space(role.action_space)
    kind = np.full(SLOTS, SLOT_ACT, np.int8)
    kind[-1] = SLOT_BOOT
    reset_after = np.zeros(SLOTS, bool)
    reset_after[-1] = True
    mask = spec.full_mask((SLOTS,))
    k = spec.num_deciders
    return TrajectoryChunk(
        agent_id="agent_0", policy_version=0, initial_state=None, obs=tree_to_torch(obs), global_state=None,
        actions=tree_to_torch(spec.allocate_actions((SLOTS,))),
        action_masks=None if mask is None else tree_to_torch(mask),
        kind=torch.as_tensor(kind), reward=torch.as_tensor(rng.normal(size=SLOTS).astype(np.float32)),
        terminal=torch.zeros(SLOTS, dtype=torch.bool), reset_after=torch.as_tensor(reset_after),
        behavior_logp=torch.zeros(SLOTS), behavior_unit_logp=torch.zeros(SLOTS, k) if k > 1 else None)


def _value_loss(role, unit_trace: str) -> float:
    torch.manual_seed(0)
    model = make_test_model(role, "none", hidden=16)
    config = AlgorithmConfig(unit_trace=unit_trace, ratio_mode="per_unit")
    algo = APPO(model, config, ActionSpec.from_space(role.action_space), device="cpu")
    return float(algo.compute_loss([_chunk(role, 0), _chunk(role, 1)])["value_loss"])


def test_auto_unit_trace_with_units_is_the_experiment_default():
    role = next(iter(UnitsGame(max_units=4).spec.roles.values()))
    assert role.global_state_space is None and ActionSpec.from_space(role.action_space).has_units
    auto = _value_loss(role, "auto")
    assert auto == pytest.approx(_value_loss(role, CHOSEN), rel=1e-6)
    for other in sorted({"joint", "geo_mean", "none"} - {CHOSEN}):
        assert auto != pytest.approx(_value_loss(role, other), rel=1e-6), other
```

Run: `.venv/bin/python -m pytest tests/unit/test_unit_trace_default.py -v`

Expected: PASS when `CHOSEN` equals the current resolution; FAIL when the ruling changed the default and the code is not updated yet. In that case change the one place in `src/colosseum/algorithms/appo.py` that maps `unit_trace == "auto"` with Units to `"geo_mean"` (found by the grep above) so it maps to the chosen value, update its docstring/comment, the `unit_trace` comment in `configs/examples/unit_harvest.yaml`, and rerun until PASS.

- [ ] **Step 6: Throughput "after" (criterion 5)**

```bash
uptime                                   # load average below 1 before measuring
.venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15 --json docs/benchmarks/after-sp2.json
```

Expected: three rows and `env steps/s increases monotonically with workers: yes`. Criterion 5 also needs every env-steps/s value at least 90% of SP1's on the same machine: ≥ 4177 (1 worker), ≥ 8500 (2), ≥ 9598 (4).

If a value is below 90% of SP1's:
1. rerun once on an idle machine (another process may have interfered);
2. re-measure SP1 on the same day to rule out machine drift:
   ```bash
   git worktree add /tmp/colosseum-sp1 main
   PYTHONPATH=/tmp/colosseum-sp1/src:/tmp/colosseum-sp1 .venv/bin/python -c "import colosseum; print(colosseum.__file__)"
   PYTHONPATH=/tmp/colosseum-sp1/src:/tmp/colosseum-sp1 .venv/bin/python /tmp/colosseum-sp1/scripts/bench_throughput.py \
     --workers 1 2 4 --duration 60 --warmup 15 --json /tmp/sp1-sameday.json
   git worktree remove /tmp/colosseum-sp1
   ```
   (the first command must print a path under `/tmp/colosseum-sp1/src`); compare against those numbers instead and say so in the doc;
3. if SP2 is still more than 10% slower, profile the worker path with the command below and **stop**: report the profile to the controller. The spec's mitigation (hot paths on preallocated per-role arrays) is a separate task the controller creates; it is not done inside T8.4.

```bash
.venv/bin/python - <<'EOF'
import cProfile, functools, pstats, time
from colosseum.core.config import load_config
from colosseum.core.registry import build_model, env_spec, make_env
from colosseum.core.roles import agent_role_spec, resolve_agent_roles
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
cfg = load_config("configs/examples/tic_tac_toe.yaml")
spec = env_spec(cfg); roles = resolve_agent_roles(cfg, spec); layout = next(iter(spec.layouts))
factory = functools.partial(build_model, cfg.get_agent_config("agent_0"), agent_role_spec(spec, roles["agent_0"]))
loop = RolloutLoop(worker_id=0, env_fn=functools.partial(make_env, cfg), num_envs=8, chunk_length=32,
                   agent_ids=["agent_0"], agent_roles=roles, model_factories={"agent_0": factory},
                   io=LoopIO(send_chunk=lambda c: c.to_payload(), poll_weights=lambda a: None),
                   lineups=[Lineup(layout, [SeatAssignment("agent_0"), SeatAssignment("agent_0")]) for _ in range(8)],
                   weight_sync_interval=1e9, seed=0)
prof = cProfile.Profile(); prof.enable(); end = time.time() + 20
while time.time() < end:
    loop.step()
prof.disable(); loop.close()
pstats.Stats(prof).sort_stats("cumulative").print_stats(30)
EOF
```

- [ ] **Step 7: `global_state` size on team_tag**

Create `scripts/measure_global_state.py`:

```python
"""Cost of ``global_state`` on team_tag (SP2 spec, risk "Размер global_state"; docs/benchmarks.md).

For ``configs/examples/team_tag.yaml`` as is (global_state on, critic encoder on) and with
``with_global_state: false`` and no critic encoder, the script:
1. collects chunks in-process with the real ``RolloutLoop`` (8 envs, the config's chunk length);
2. reports the numpy payload bytes per chunk: observation leaves, ``global_state`` leaves, total;
3. times ``APPO.train_step`` on ``learner.batch_chunks`` of those chunks (median of 20 steps after
   3 warm-up steps, 1 torch thread).

Usage::

    .venv/bin/python scripts/measure_global_state.py [--chunks 64] [--json out.json]
"""
from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
AGENT = "agent_0"


def _nbytes(tree) -> int:
    from colosseum.core.tree import tree_leaves

    return 0 if tree is None else sum(int(getattr(leaf, "nbytes", 0)) for leaf in tree_leaves(tree))


def measure(config, num_chunks: int) -> dict:
    import torch

    from colosseum.algorithms.appo import APPO
    from colosseum.core.registry import build_model, env_spec, make_env
    from colosseum.core.roles import agent_role_spec, resolve_agent_roles
    from colosseum.core.specs import ActionSpec
    from colosseum.core.types import Lineup, SeatAssignment, TrajectoryChunk
    from colosseum.worker.rollout_loop import LoopIO, RolloutLoop

    torch.set_num_threads(1)
    spec = env_spec(config)
    roles = resolve_agent_roles(config, spec)
    role = agent_role_spec(spec, roles[AGENT])
    agent_config = config.get_agent_config(AGENT)
    factory = functools.partial(build_model, agent_config, role)
    payloads: list[dict] = []
    io = LoopIO(send_chunk=lambda c: payloads.append(c.to_payload()), poll_weights=lambda agent_id: None)
    lineups = [Lineup("2v2", [SeatAssignment(AGENT) for _ in range(4)]) for _ in range(8)]
    loop = RolloutLoop(worker_id=0, env_fn=functools.partial(make_env, config), num_envs=8,
                       chunk_length=config.rollout.chunk_length, agent_ids=[AGENT], agent_roles=roles,
                       model_factories={AGENT: factory}, io=io, lineups=lineups, weight_sync_interval=1e9, seed=0)
    try:
        while len(payloads) < num_chunks:
            loop.step()
    finally:
        loop.close()
    obs_bytes = statistics.mean(_nbytes(p["obs"]) for p in payloads)
    gs_bytes = statistics.mean(_nbytes(p["global_state"]) for p in payloads)
    total_bytes = statistics.mean(_nbytes(p) for p in payloads)
    algo = APPO(factory(), agent_config.algorithm, ActionSpec.from_space(role.action_space), device="cpu")
    batch_size = config.learner.batch_chunks
    times = []
    for i in range(23):
        batch = [TrajectoryChunk.from_payload(p) for p in payloads[(i * batch_size) % len(payloads):][:batch_size]]
        if len(batch) < batch_size:
            batch = [TrajectoryChunk.from_payload(p) for p in payloads[:batch_size]]
        start = time.perf_counter()
        algo.train_step(batch)
        if i >= 3:
            times.append(time.perf_counter() - start)
    return {"chunk_length": config.rollout.chunk_length, "obs_bytes_per_chunk": obs_bytes,
            "global_state_bytes_per_chunk": gs_bytes, "payload_bytes_per_chunk": total_bytes,
            "batch_chunks": batch_size, "train_step_ms_median": 1000 * statistics.median(times)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from colosseum.core.config import load_config

    base = load_config(CONFIG)
    kwargs = {**base.env.kwargs, "with_global_state": False}
    without = load_config(CONFIG, {"env.kwargs": kwargs, "networks.critic_encoder_class": None})
    results = {"with_global_state": measure(base, args.chunks), "without_global_state": measure(without, args.chunks)}
    for name, r in results.items():
        print(f"{name}: {json.dumps(r)}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `.venv/bin/python scripts/measure_global_state.py --json docs/benchmarks/global-state-team-tag.json`

Expected: two lines. With the default team_tag (`size 7`, window radius 2, `chunk_length 32`): observation leaves `32 × (4·5·5 + 4·4) = 3,712` bytes per chunk, `global_state` `32 × 4·7·7 = 6,272` bytes per chunk (both `uint8` except the 4-float vector), so `global_state` more than doubles the observation payload of a chunk; `without_global_state` shows 0 for it. The train-step times show the cost of the critic encoder.

- [ ] **Step 8: Write the benchmark sections**

Append to `docs/benchmarks.md` (Russian; fill every cell from the outputs of Steps 3, 6, 7 and the date of the run):

```markdown
## После SP2

Та же машина и та же команда, что в разделах «до» и «После SP1». Код — ветка `sp2-game-model` после T8.3, коммит `<git rev-parse --short HEAD>`. Сырые данные: [`benchmarks/after-sp2.json`](benchmarks/after-sp2.json).

Нагрузка та же по смыслу: крестики-нолики, self-play, 8 сред на воркер, `chunk_length=32`, `batch_chunks=8`, ходит одно место за шаг среды. Изменилось внутри SP2: наблюдения и действия — деревья, чанк v2 со слотами `act`/`boot`/`pad`, value (и bootstrap) считает лёрнер, воркер считает только политику.

| Воркеры | Апдейты/с | Шаги сред/с | Доля от SP1 | глубина очереди | узкое место |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |

Критерий 5 спеки SP2: не хуже SP1 более чем на 10% (≥ 90% от 4641 / 9444 / 10664) — <да/нет>; рост 1 → 2 → 4 монотонный — <да/нет>.

## Эксперимент по юнитам (SP2, критерий 4)

`scripts/units_experiment.py`: `unit_harvest`, K=8 (поле 8×8, 2 стартовых рабочих, 50 шагов) и K=128 (поле 16×16, 32 стартовых рабочих, 80 шагов), `ratio_mode` ∈ {joint, per_unit} × `unit_trace` ∈ {joint, geo_mean, none}, бюджет <STEPS> шагов сред на прогон, 2 воркера, по два прогона одновременно, seed 0 (один сид: 12 прогонов × 2 сида не помещаются в час на этой машине). Оценка — greedy-политика последнего чекпоинта, 100 партий против случайного игрока и 100 против скриптового бота (`scripted_action`). Диагностики — среднее по последней четверти записей `train`. Сырые данные: [`benchmarks/units-experiment.json`](benchmarks/units-experiment.json).

<таблица, которую напечатал скрипт>

Выводы:
- `unit_trace` (влияет только на value-таргеты): <что показали share_vs_scripted, ess, log_rho_abs_p95 при K=8 и K=128>;
- `ratio_mode`: <joint против per_unit при K=128: clip_fraction_joint, share>;
- решение: `unit_trace: auto` с `Units` = **<значение>** (ruling в отчёте приёмки SP2).

## Размер `global_state` (team_tag)

`scripts/measure_global_state.py`: `configs/examples/team_tag.yaml` как есть и с `with_global_state: false` без critic encoder. Сырые данные: [`benchmarks/global-state-team-tag.json`](benchmarks/global-state-team-tag.json).

| Вариант | Байт наблюдений на чанк | Байт `global_state` на чанк | Весь payload чанка | `train_step`, мс (медиана) |
|---|---|---|---|---|
| с `global_state` | | | | |
| без | | | | |

Вывод: <во сколько раз `global_state` увеличивает чанк и шаг лёрнера; когда стоит включать>.
```

Add the same numbers (shorter: the criteria verdicts and the ruling) to the report draft as sections «Эксперимент по юнитам (T8.4)» and «Скорость (T8.4)».

- [ ] **Step 9: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, zero warnings.

- [ ] **Step 10: Commit and push**

```bash
git add scripts/units_experiment.py scripts/measure_global_state.py docs/benchmarks.md docs/benchmarks/*.json \
  docs/superpowers/reports/*-sp2-acceptance.md
git add tests/unit/test_unit_trace_default.py src/colosseum/algorithms/appo.py configs/examples/unit_harvest.yaml 2>/dev/null || true
git commit -m "docs: SP2 units experiment, throughput after SP2, global_state size; unit_trace default ruling"
git push origin sp2-game-model
```

If the ruling changed the default, commit the code change separately first: `git commit -m "feat: unit_trace auto with Units = <value> (units experiment ruling)"` with `appo.py`, the test and the config comment.

---

### Task T8.5: Reference examples `space_miners` and `chase` on the new contract

Spec criterion 6 and block 10 ("Эталонные примеры"): rewritten for the new contract; `validate` and a short smoke `train` pass; no learning thresholds.
- `space_miners` becomes a "normal competitive game" on the new features: ships as `Units(3, Dict(accel=Box(2), push=Discrete(2)))` (a continuous and a discrete component per unit), asteroids as an entity list with a mask, a `Dict` observation, masks inside the units (`push` only within reach), `Outcome` with scores and the engine's tie-break ranks. The Box2D engine `examples/space_miners/game_engine.py` is kept unchanged; Box2D stays optional (the `examples` extra), so its tests use `pytest.importorskip("Box2D")`.
- `chase` (`examples/composite_action`) uses a tree action `Dict(direction=Discrete(4), speed=Box(1))` with `make_distribution` params mirroring the tree.

**Files:**
- Create:
  - `examples/space_miners/game.py`, `examples/space_miners/models.py`
  - `examples/composite_action/game.py`, `examples/composite_action/models.py`
- Replace (write in full; T7.3 may already have deleted them): `configs/examples/space_miners.yaml`, `configs/examples/chase.yaml`
- Delete (if T7.3 has not): `examples/space_miners/env.py`, `examples/space_miners/networks.py`, `examples/composite_action/env.py`, `examples/composite_action/networks.py`
- Keep unchanged: `examples/space_miners/game_engine.py`, `examples/*/__init__.py`
- Test:
  - `tests/contract/test_reference_examples.py`
  - `tests/integration/test_reference_examples_smoke.py`

**Interfaces:**
- Consumes: as T8.1 (contract types, `Units` with a `Dict` per-unit space and a `Box` component, `UnitsHead` box params `{"mean": [B, U, d], "log_std": ...}`, `make_distribution` with a `Dict` action tree); `tests/demo_checks.py` (T8.1); `cli_runner.run_train`; `examples/space_miners/game_engine.py` (`SpaceMinersGameState`, constants).
- Produces:
  - `examples.space_miners.game`: `SpaceMinersGame(preset="Round 1", max_ticks=1000, max_asteroids=24)`, `MAX_SHIPS`, `SHIP_FEATURES`, `ASTEROID_FEATURES`
  - `examples.space_miners.models`: `MinersEncoder`, `MinersPolicy`, `MinersValue`, `masked_mean`
  - `examples.composite_action.game`: `ChaseGame()`, `DIRS`, `GRID`, `MAX_STEPS`
  - `examples.composite_action.models`: `ChaseEncoder`, `ChasePolicy`, `ChaseValue`

- [ ] **Step 1: Find leftovers of the old examples**

```bash
ls examples/space_miners examples/composite_action configs/examples
grep -rn "space_miners.env\|space_miners.networks\|composite_action.env\|composite_action.networks" \
  --include="*.py" --include="*.yaml" --include="*.md" . | grep -v "^./review/\|^./docs/superpowers/\|^./research/"
```

Expected: the `grep` prints nothing (T7.3 removed the users of the old files). If it prints a test or a config, that file is legacy that T7.3 missed: delete it in this task if it only tests removed behaviour, otherwise port it to the new files. Never edit `review/`, `research/` or old SP specs/plans.

- [ ] **Step 2: Write the failing tests**

Create `tests/contract/test_reference_examples.py`:

```python
"""Reference examples on the SP2 contract (T8.5): chase (tree action) and space_miners (Units with a
Box component, entity list). No learning thresholds: env semantics, validate, random matches."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.core.specs import ActionSpec
from demo_checks import check_example_config, random_matches
from examples.composite_action.game import DIRS, GRID, ChaseGame


def test_chase_action_is_a_tree_in_natural_order():
    space = ChaseGame().spec.roles["player"].action_space
    assert list(space.spaces) == ["direction", "speed"]
    spec = ActionSpec.from_space(space)
    assert [g.kind for g in spec.groups] == ["discrete", "box"] and not spec.has_units


def test_chase_accepts_scalar_and_array_components():
    env = ChaseGame()
    env.reset(0, "2p")
    before = env._pos.copy()
    env.step({0: {"direction": np.int64(1), "speed": np.array([0.5], np.float32)},
              1: {"direction": 2, "speed": np.float32(1.0)}})
    expected = np.clip(before + np.stack([DIRS[1] * 0.5, DIRS[2] * 1.0]), 0.0, GRID)
    np.testing.assert_allclose(env._pos, expected, atol=1e-6)


def test_chase_ends_by_rule_after_twenty_steps():
    env = ChaseGame()
    res = env.reset(0, "2p")
    steps = 0
    while not res.episode_over:
        res = env.step({p: {"direction": 0, "speed": np.zeros(1, np.float32)} for p in res.acting})
        steps += 1
    assert steps == 20 and not res.truncated and set(res.rewards) == {0, 1}
    assert res.rewards[0] == -res.rewards[1]


def test_chase_config_validates():
    check_example_config("chase")


def test_chase_random_matches():
    results, _ = random_matches(ChaseGame, min_episodes=8)
    assert all(r.layout == "2p" and r.outcome_kind == "wdl" for r in results)


def test_space_miners_spaces_and_masks():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import MAX_SHIPS, SpaceMinersGame

    env = SpaceMinersGame(max_ticks=20)
    role = env.spec.roles["player"]
    assert list(role.action_space.per_unit.spaces) == ["accel", "push"]
    spec = ActionSpec.from_space(role.action_space)
    assert spec.has_units and spec.num_deciders == MAX_SHIPS
    res = env.reset(0, "2p")
    obs, mask = res.obs[0], res.action_masks[0]
    assert obs["asteroids"].shape == (24, 7) and obs["asteroid_mask"].sum() >= 3
    assert mask["unit"].tolist() == [True] * MAX_SHIPS and mask["action"].shape == (MAX_SHIPS, 2)
    assert mask["action"][:, 0].all()                          # "no push" is always legal
    assert role.observation_space.contains(obs)


def test_space_miners_players_see_mirrored_starts():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import SpaceMinersGame

    res = SpaceMinersGame(max_ticks=20).reset(3, "2p")
    np.testing.assert_allclose(res.obs[0]["ships"], res.obs[1]["ships"], atol=1e-6)


def test_space_miners_match_ends_by_rule_with_scores():
    pytest.importorskip("Box2D")
    from examples.space_miners.game import MAX_SHIPS, SpaceMinersGame

    env = SpaceMinersGame(max_ticks=5)
    res = env.reset(0, "2p")
    idle = {"accel": np.zeros((MAX_SHIPS, 2), np.float32), "push": np.zeros(MAX_SHIPS, np.int64)}
    for _ in range(5):
        res = env.step({0: idle, 1: idle})
    assert res.episode_over and not res.truncated
    assert set(res.outcome.team_score) == {0, 1} and set(res.outcome.team_rank) == {0, 1}


def test_space_miners_config_validates():
    pytest.importorskip("Box2D")
    check_example_config("space_miners")


def test_space_miners_random_matches():
    pytest.importorskip("Box2D")
    import functools

    from examples.space_miners.game import SpaceMinersGame

    results, _ = random_matches(functools.partial(SpaceMinersGame, max_ticks=30), min_episodes=4)
    assert all(r.layout == "2p" and r.episode_length == 30 for r in results)
```

Create `tests/integration/test_reference_examples_smoke.py`:

```python
"""Short `colosseum train` runs of the reference examples (T8.5; spec criterion 6: validate and a
smoke train, no learning threshold)."""
from __future__ import annotations

import pytest

from cli_runner import run_train
from demo_checks import example_config


def test_chase_trains_briefly(tmp_path):
    run = run_train(example_config("chase"), tmp_path, "chase", overrides={"training.total_timesteps": "2000"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.records("train")
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))


def test_space_miners_trains_briefly(tmp_path):
    pytest.importorskip("Box2D")
    run = run_train(example_config("space_miners"), tmp_path, "miners",
                    overrides={"training.total_timesteps": "1500", "env.kwargs": "{max_ticks: 60}"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert run.records("train")
    assert "deciders_valid_mean" in run.records("train")[-1]
    assert any((run.root / "checkpoints" / "agent_0").glob("ckpt_v*"))
```

Run: `.venv/bin/python -m pytest tests/contract/test_reference_examples.py tests/integration/test_reference_examples_smoke.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'examples.composite_action.game'`.

- [ ] **Step 3: `chase` game and models**

Create `examples/composite_action/game.py`:

```python
"""Chase: a 1v1 game with a tree action (reference example).

Each player moves on a continuous ``10 x 10`` field towards its own fixed target. An action is a
``Dict`` (built from a list, so the order is ``direction``, ``speed``): ``direction``:
``Discrete(4)`` (up, right, down, left) and ``speed``: ``Box(0, 1, (1,))``. After 20 steps the
player closer to its target wins (+1 / -1, equal distance is a draw). The end is a rule
(``truncated=False``); without an ``Outcome`` the framework derives the result from the returns.

Observation ``float32[5]``: own x, y, target x, y (all / 10), steps left fraction.
Layout ``2p`` (``GameSpec.symmetric(2, ...)``), simultaneous moves, no masks.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult

DIRS = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.float32)   # up, right, down, left
GRID = 10.0
MAX_STEPS = 20


def _scalar(value, name: str, player: int) -> float:
    """The one element of a scalar or size-1 array action component."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"player {player}: action component {name!r} must have one element, got shape {arr.shape}")
    return arr.reshape(-1)[0]


class ChaseGame(MultiAgentEnv):
    def __init__(self) -> None:
        obs_space = gymnasium.spaces.Box(0.0, 1.0, (5,), np.float32)
        act_space = gymnasium.spaces.Dict([
            ("direction", gymnasium.spaces.Discrete(4)),
            ("speed", gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32)),
        ])
        self.spec = GameSpec.symmetric(2, obs_space, act_space)
        self._rng = np.random.default_rng()
        self._pos = np.zeros((2, 2), np.float32)
        self._target = np.zeros((2, 2), np.float32)
        self._t = 0

    def _obs(self, p: int) -> np.ndarray:
        return np.array([*(self._pos[p] / GRID), *(self._target[p] / GRID), 1.0 - self._t / MAX_STEPS], np.float32)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._pos = (self._rng.random((2, 2)) * GRID).astype(np.float32)
        self._target = (self._rng.random((2, 2)) * GRID).astype(np.float32)
        self._t = 0
        return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)})

    def step(self, actions: dict) -> StepResult:
        for p in (0, 1):
            d = int(_scalar(actions[p]["direction"], "direction", p))
            s = float(np.clip(_scalar(actions[p]["speed"], "speed", p), 0.0, 1.0))
            self._pos[p] = np.clip(self._pos[p] + DIRS[d] * s, 0.0, GRID)
        self._t += 1
        if self._t < MAX_STEPS:
            return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)})
        dist = np.linalg.norm(self._pos - self._target, axis=1)
        sign = float(np.sign(dist[1] - dist[0]))          # +1 when player 0 is closer
        return StepResult(acting=set(), obs={}, rewards={0: sign, 1: -sign}, episode_over=True)
```

Create `examples/composite_action/models.py`:

```python
"""Model parts for chase: a tree action, ``direction`` (categorical) and ``speed`` (Gaussian)."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution


class ChaseEncoder(BaseEncoder):
    def __init__(self, observation_space, latent: int = 32, **kwargs) -> None:
        super().__init__()
        self._latent = latent
        self.net = nn.Sequential(nn.Linear(observation_space.shape[0], latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ChasePolicy(BasePolicy):
    """``make_distribution`` params mirror the action tree: logits for ``direction``,
    ``{"mean", "log_std"}`` for ``speed``."""

    def __init__(self, in_dim: int, action_spec, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.direction = nn.Linear(in_dim, 4)
        self.speed_mean = nn.Linear(in_dim, 1)
        self.speed_log_std = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, aux: dict):
        params = {"direction": self.direction(features),
                  "speed": {"mean": self.speed_mean(features), "log_std": self.speed_log_std}}
        return make_distribution(self.action_spec, params)


class ChaseValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 16, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

- [ ] **Step 4: `space_miners` game and models**

Create `examples/space_miners/game.py`:

```python
"""Space Miners on the SP2 contract: a 1v1 competitive game with units (reference example).

Two players, three ships each, push asteroids into their own base to score (Box2D physics in
``game_engine.py``). Presets: ``Round 1`` (no energy or upgrades), ``Round 2`` (energy),
``Final Round`` (energy + upgrades; upgrades are not controlled by this wrapper).

Requires Box2D (``pip install -e ".[examples]"``).

Contract features shown here:
- ``GameSpec.symmetric(2, ...)``, simultaneous moves, a rule-based end at ``max_ticks``
  (``truncated=False``) with ``Outcome.team_score`` = game scores (ties broken by the engine's
  "who scored first" rule through ``Outcome.team_rank``);
- ships as ``Units(3, Dict(accel=Box(2), push=Discrete(2)))``: a continuous component and a
  discrete one per unit (the ``Dict`` is built from a list, so the order is ``accel``, ``push``);
- asteroids as an entity list with a mask (``asteroids`` + ``asteroid_mask``) in a ``Dict``
  observation;
- masks inside ``Units``: ``push`` is legal only when an asteroid is within push range (otherwise
  the push would do nothing; with energy it would also waste energy).

Observation from the player's perspective (``Dict``, positions in [-1, 1], player 1 mirrored so
that both players' bases are on the left):
- ``ships``: ``float32[3, 9]`` own ships: x, y, vx, vy, energy, 4 upgrade levels;
- ``enemy_ships``: ``float32[3, 9]``;
- ``asteroids``: ``float32[max_asteroids, 7]``: x, y, vx, vy, size one-hot; ``asteroid_mask``;
- ``global``: ``float32[3]``: own score / 100, opponent score / 100, time fraction.
Rewards: (own score gain - opponent score gain) / 20 per step, +1 / -1 at the end for win / loss.
"""
from __future__ import annotations

import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult
from colosseum.envs.spaces import Units
from examples.space_miners.game_engine import (
    ASTEROID_RADIUS_UNITS,
    GAME_HEIGHT,
    GAME_WIDTH,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    PPM,
    PUSH_RADIUS_UNITS,
    SHIP_RADIUS_UNITS,
    SpaceMinersGameState,
)

MAX_SHIPS = 3
SHIP_FEATURES = 9
ASTEROID_FEATURES = 7
DEFAULT_MAX_ASTEROIDS = 24     # the engine keeps at most 21 (20 + one pair overshoot)
SIZE_NAMES = ("small", "medium", "large")


class SpaceMinersGame(MultiAgentEnv):
    def __init__(self, preset: str = "Round 1", max_ticks: int = 1000,
                 max_asteroids: int = DEFAULT_MAX_ASTEROIDS) -> None:
        self._preset, self._max_ticks, self._max_asteroids = preset, max_ticks, max_asteroids
        box = gymnasium.spaces.Box
        obs_space = gymnasium.spaces.Dict([
            ("ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
            ("enemy_ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
            ("asteroids", box(-1.0, 1.0, (max_asteroids, ASTEROID_FEATURES), np.float32)),
            ("asteroid_mask", gymnasium.spaces.MultiBinary(max_asteroids)),
            ("global", box(-1.0, 1.0, (3,), np.float32)),
        ])
        per_ship = gymnasium.spaces.Dict([("accel", box(-1.0, 1.0, (2,), np.float32)),
                                          ("push", gymnasium.spaces.Discrete(2))])
        self.spec = GameSpec.symmetric(2, obs_space, Units(MAX_SHIPS, per_ship))
        self._game: SpaceMinersGameState | None = None

    # ----- observation ---------------------------------------------------------------------
    @staticmethod
    def _xy(pos: tuple[float, float], player: int) -> tuple[float, float]:
        x = pos[0] / (GAME_WIDTH / 2) - 1.0
        return (-x if player == 1 else x), pos[1] / (GAME_HEIGHT / 2) - 1.0

    @staticmethod
    def _v(vel: tuple[float, float], player: int) -> tuple[float, float]:
        vx = float(np.clip(vel[0] / MAX_VELOCITY, -1.0, 1.0))
        return (-vx if player == 1 else vx), float(np.clip(vel[1] / MAX_VELOCITY, -1.0, 1.0))

    def _ships(self, owner: int, viewer: int) -> np.ndarray:
        game = self._game
        out = np.zeros((MAX_SHIPS, SHIP_FEATURES), np.float32)
        for i, ship in enumerate(game.players[owner].ships):
            out[i, 0:2] = self._xy(ship.pos_game, viewer)
            out[i, 2:4] = self._v(ship.vel_game, viewer)
            if game.energy_enabled:
                out[i, 4] = ship.energy / 50.0 - 1.0
            if game.upgrades_enabled:
                out[i, 5:9] = [ship.upgrades[k] / 5.0 for k in
                               ("max_speed", "max_accel", "push_force", "energy_efficiency")]
        return out

    def _obs(self, player: int) -> dict:
        game = self._game
        asteroids = np.zeros((self._max_asteroids, ASTEROID_FEATURES), np.float32)
        mask = np.zeros(self._max_asteroids, np.int8)
        for i, a in enumerate(game.asteroids[: self._max_asteroids]):
            asteroids[i, 0:2] = self._xy(a.pos_game, player)
            asteroids[i, 2:4] = self._v(a.vel_game, player)
            asteroids[i, 4 + SIZE_NAMES.index(a.size)] = 1.0
            mask[i] = 1
        me, opp = game.players[player], game.players[1 - player]
        glob = np.array([np.clip(me.score / 100.0, -1, 1), np.clip(opp.score / 100.0, -1, 1),
                         game.tick / game.max_ticks * 2.0 - 1.0], np.float32)
        return {"ships": self._ships(player, player), "enemy_ships": self._ships(1 - player, player),
                "asteroids": asteroids, "asteroid_mask": mask, "global": glob}

    def _mask(self, player: int) -> dict:
        reach = np.zeros(MAX_SHIPS, bool)
        for i, ship in enumerate(self._game.players[player].ships):
            sx, sy = ship.body.position
            for a in self._game.asteroids:
                ax, ay = a.body.position
                gap = np.hypot(ax - sx, ay - sy) - (SHIP_RADIUS_UNITS + ASTEROID_RADIUS_UNITS[a.size]) / PPM
                if gap <= PUSH_RADIUS_UNITS / PPM:
                    reach[i] = True
                    break
        action = np.ones((MAX_SHIPS, 2), bool)
        action[:, 1] = reach                       # push=1 only within reach; push=0 always legal
        return {"unit": np.ones(MAX_SHIPS, bool), "action": action}

    def _acting_result(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={p: self._obs(p) for p in (0, 1)},
                          action_masks={p: self._mask(p) for p in (0, 1)}, rewards=rewards)

    # ----- MultiAgentEnv -------------------------------------------------------------------
    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._game = SpaceMinersGameState(preset=self._preset, max_ticks=self._max_ticks, seed=seed)
        return self._acting_result({})

    def _commands(self, player: int, action: dict) -> dict:
        accel = np.asarray(action["accel"], np.float32).reshape(MAX_SHIPS, 2)
        push = np.asarray(action["push"], np.int64).reshape(MAX_SHIPS)
        sign = -1.0 if player == 1 else 1.0         # undo the mirror of player 1
        return {"commands": [{"ship_id": i,
                              "acceleration": {"x": sign * float(np.clip(accel[i, 0], -1, 1)) * MAX_ACCELERATION,
                                               "y": float(np.clip(accel[i, 1], -1, 1)) * MAX_ACCELERATION},
                              "push": bool(push[i])} for i in range(MAX_SHIPS)]}

    def step(self, actions: dict) -> StepResult:
        game = self._game
        before = [p.score for p in game.players]
        game.update([self._commands(0, actions[0]), self._commands(1, actions[1])])
        gain = [game.players[i].score - before[i] for i in (0, 1)]
        rewards = {i: (gain[i] - gain[1 - i]) / 20.0 for i in (0, 1)}
        if not game.is_game_over():
            return self._acting_result(rewards)
        winner = game.get_winner_index()
        if winner >= 0:
            rewards[winner] += 1.0
            rewards[1 - winner] -= 1.0
        ranks = {0: 1.5, 1: 1.5} if winner < 0 else {winner: 1.0, 1 - winner: 2.0}
        scores = {i: float(game.players[i].score) for i in (0, 1)}
        return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True,
                          outcome=Outcome(team_rank=ranks, team_score=scores))

    def close(self) -> None:
        self._game = None
```

Prototype check with Box2D (one process, random legal actions, `max_ticks=200`): about 950 env steps/s; outcomes with scores and tie ranks as expected; both players' ships appear at mirrored positions.

Create `examples/space_miners/models.py`:

```python
"""Model parts for space_miners: ship embeddings go to a ``UnitsHead`` (``accel`` Gaussian and
``push`` categorical per ship); asteroids are pooled with their mask."""
from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.networks.heads import UnitsHead, make_distribution


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MinersEncoder(BaseEncoder):
    def __init__(self, observation_space, ship_dim: int = 64, latent: int = 128, **kwargs) -> None:
        super().__init__()
        ship_f = observation_space["ships"].shape[1]
        rock_f = observation_space["asteroids"].shape[1]
        glob = observation_space["global"].shape[0]
        self._latent = latent
        self.ship_net = nn.Sequential(nn.Linear(ship_f, ship_dim), nn.ReLU(), nn.Linear(ship_dim, ship_dim), nn.ReLU())
        self.enemy_net = nn.Sequential(nn.Linear(ship_f, 32), nn.ReLU())
        self.rock_net = nn.Sequential(nn.Linear(rock_f, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(ship_dim + 32 + 64 + glob, latent), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return self._latent

    def forward(self, obs: dict) -> EncoderOutput:
        ships = self.ship_net(obs["ships"])                                    # [B, 3, E]
        enemies = self.enemy_net(obs["enemy_ships"]).mean(dim=1)
        rocks = masked_mean(self.rock_net(obs["asteroids"]), obs["asteroid_mask"])
        latent = self.mlp(torch.cat([ships.mean(dim=1), enemies, rocks, obs["global"]], dim=-1))
        return EncoderOutput(latent=latent, aux={"ships": ships})


class MinersPolicy(BasePolicy):
    def __init__(self, in_dim: int, action_spec, ship_dim: int = 64, unit_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.action_spec = action_spec
        self.ships_head = UnitsHead(action_spec.groups[0], ship_dim + in_dim, unit_hidden)

    def forward(self, features: torch.Tensor, aux: dict):
        ships = aux["ships"]
        context = features.unsqueeze(1).expand(-1, ships.shape[1], -1)
        return make_distribution(self.action_spec, self.ships_head(torch.cat([ships, context], dim=-1)))


class MinersValue(BaseValue):
    def __init__(self, in_dim: int, value_hidden: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, value_hidden), nn.ReLU(), nn.Linear(value_hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
```

- [ ] **Step 5: Configs**

Write `configs/examples/chase.yaml`:

```yaml
# Chase: 1v1 with a tree action, Dict(direction: Discrete(4), speed: Box(1)). A reference example
# without a learning threshold.
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.composite_action.game.ChaseGame"
  kwargs: {}

networks:
  encoder_class: "examples.composite_action.models.ChaseEncoder"
  core: null
  policy_class: "examples.composite_action.models.ChasePolicy"
  value_class: "examples.composite_action.models.ChaseValue"
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
  total_timesteps: 50000
  seed: null

matchmaking:
  mode: self_play
  latest_prob: 0.5

checkpoint:
  interval: 10
  pool_size: 5
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0
```

Write `configs/examples/space_miners.yaml`:

```yaml
# Space Miners: 1v1 with ships as Units (Box acceleration + discrete push) and asteroids as an
# entity list. A reference example without a learning threshold.
# Requires Box2D: pip install -e ".[examples]"
run:
  name: null
  dir: "runs"

env:
  env_class: "examples.space_miners.game.SpaceMinersGame"
  kwargs: {preset: "Round 1", max_ticks: 1000}

networks:
  encoder_class: "examples.space_miners.models.MinersEncoder"
  core: null
  policy_class: "examples.space_miners.models.MinersPolicy"
  value_class: "examples.space_miners.models.MinersValue"
  kwargs: {}

algorithm:
  gamma: 0.99
  vtrace_lambda: 0.95
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
  total_timesteps: 500000
  seed: null

matchmaking:
  mode: self_play
  latest_prob: 0.5

checkpoint:
  interval: 100
  pool_size: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 50
  console_interval_sec: 10.0
```

- [ ] **Step 6: Delete the old files**

```bash
git rm --ignore-unmatch examples/space_miners/env.py examples/space_miners/networks.py \
  examples/composite_action/env.py examples/composite_action/networks.py
```

- [ ] **Step 7: Run the tests and the CLI checks**

```bash
.venv/bin/python -m pytest tests/contract/test_reference_examples.py tests/integration/test_reference_examples_smoke.py -v -rs
.venv/bin/colosseum validate -c configs/examples/chase.yaml
.venv/bin/colosseum validate -c configs/examples/space_miners.yaml
```

Expected: all tests pass (with Box2D installed by `scripts/setup-dev.sh`, nothing is skipped; without it, the four `space_miners` tests are reported as skipped with `could not import 'Box2D'`); both `validate` commands exit 0.

- [ ] **Step 8: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, zero warnings.

- [ ] **Step 9: Commit and push**

```bash
git add examples/space_miners examples/composite_action configs/examples/chase.yaml configs/examples/space_miners.yaml \
  tests/contract/test_reference_examples.py tests/integration/test_reference_examples_smoke.py
git commit -m "feat: space_miners (units with Box components, entity list) and chase (tree action) on the SP2 contract"
git push origin sp2-game-model
```

---

### Task T8.6: Docs: `ENV_GUIDE.md`, README, CLAUDE.md, `GPU_CHECKS.md`

Spec criterion 7 (documentation describes the real state and how to write an env for every game type, with examples) and criterion 9 (new CUDA paths are marked `gpu` and listed in `docs/GPU_CHECKS.md`). Language: `docs/ENV_GUIDE.md` and `docs/GPU_CHECKS.md` in Russian; `README.md` and `CLAUDE.md` in English.

**Files:**
- Create: `docs/ENV_GUIDE.md`
- Replace: `README.md`
- Modify: `CLAUDE.md` (sections listed in Step 5), `docs/GPU_CHECKS.md`

**Interfaces:**
- Consumes: the real CLI surface (`colosseum <command> --help`), the demo games and configs (T8.1, T8.2, T8.5), the report draft and `docs/benchmarks.md` (T8.3, T8.4: learning numbers, the `unit_trace` ruling, throughput), `pytest --collect-only` counts.
- Produces: documentation only.

- [ ] **Step 1: Check the real CLI surface**

```bash
for c in train validate eval bc run-learner run-workers serve-weight-store; do
  .venv/bin/colosseum $c --help > /tmp/help-$c.txt || echo "MISSING $c"
done
grep -E -- "--agent|--layout" /tmp/help-bc.txt /tmp/help-eval.txt
ls configs/examples
```

Expected: no `MISSING` line; `bc --help` lists `--agent`, `eval --help` lists `--layout` (repeatable), `-a/--agents` (`name=path`), `--num-matches`, `--output`, `--deterministic`, `--seed`; `configs/examples` lists `chase`, `coin_grid`, `coop_buttons`, `predator_prey`, `space_miners`, `team_tag`, `tic_tac_toe`, `tron`, `unit_harvest` (and any tic-tac-toe variants T7.3 kept). If a flag name differs from the text below, change the text to the real flag; do not change the CLI in this task.

- [ ] **Step 2: Write `docs/ENV_GUIDE.md`**

Create `docs/ENV_GUIDE.md`:

````markdown
# Как написать среду для Colosseum

Этот документ объясняет контракт среды SP2 (`GameSpec` + `MultiAgentEnv` + `StepResult`) на примерах
для каждого типа игры. Все примеры — рабочие демо-среды из `examples/`, их конфиги лежат в
`configs/examples/`. Полное описание контракта — спека
`docs/superpowers/specs/2026-10-08-sp2-game-model-design.md` (блоки 1–3).

| Тип игры | Демо-среда | Что в ней показано |
|---|---|---|
| Соло | `examples/coin_grid` | `GameSpec.solo`, Dict-наблюдение с `uint8`, маски, обрыв по лимиту шагов (`truncated`, `final_obs`) |
| 1v1 пошаговая | `examples/tic_tac_toe` | ходит одно место из двух (`acting`), маски, награды ждущему месту |
| Бот с юнитами | `examples/unit_harvest` | `Units`, рождение и гибель юнитов, списки сущностей с маской, `Dict`-действие |
| Команда на команду | `examples/team_tag` | `GameSpec.teams_of`, локальное окно, `global_state` для критика, выбывший сокомандник |
| FFA с выбыванием | `examples/tron` | `GameSpec.symmetric([2, 3, 4])`, `terminated`, ранги по порядку выбывания |
| 1 vs N, асимметрия | `examples/predator_prey` | две роли с разными пространствами, два агента (`agents.<id>.roles`) |
| Кооператив | `examples/coop_buttons` | одна команда, исход `score`, `teammates: mixed`, cross-play |
| Соревновательная игра «как на соревновании» | `examples/space_miners` | `Units` с `Box`-компонентом, маски внутри юнитов, сущности, Box2D |
| Дерево действий | `examples/composite_action` | `Dict(direction=Discrete, speed=Box)` |

## 1. Минимальная среда

Среда — подкласс `colosseum.envs.game.MultiAgentEnv` с атрибутом `spec` и двумя методами:

```python
import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult


class Counter(MultiAgentEnv):
    def __init__(self, length: int = 10) -> None:
        self.length = length
        self.spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32), gymnasium.spaces.Discrete(2))
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.array([self._t / self.length], np.float32)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions: dict) -> StepResult:
        self._t += 1
        reward = float(actions[0])                 # +1 за действие 1
        if self._t == self.length:                 # конец по правилам
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()}, rewards={0: reward})
```

Конфиг: `env.env_class: my_game.env.Counter`, `env.kwargs: {length: 10}`. Перед обучением всегда
запускайте `colosseum validate -c my_game.yaml`: он проверяет спеку, делает `reset` каждого варианта
партии и несколько шагов случайными легальными действиями с полной проверкой контракта
(`space.contains`), а затем прогоняет модель каждого агента.

## 2. `GameSpec`: роли, места, команды, варианты партии

```python
RoleSpec(observation_space, action_space, global_state_space=None)   # тип места
SeatSpec(role="player", team=0)                                      # одно место варианта
GameSpec(roles={"player": RoleSpec(...)}, layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1))})
```

- **Вариант партии** (`layout`) — кортеж мест; место `i` варианта — это ключ `i` во всех словарях
  `StepResult`. Номера команд варианта — ровно `0..T-1`.
- **Тип исхода выводится из числа команд**: одна команда — `score` (соло, кооператив), две — `wdl`,
  три и больше — `rank`. Одна игра может иметь варианты разных типов.
- **Разное число игроков в одной игре** — разные варианты: `GameSpec.symmetric([2, 3, 4], obs, act)`
  даёт `2p`, `3p`, `4p`. Места `n..P_max-1` в варианте `np` — пустые: им нельзя давать наблюдения,
  награды или `terminated`. Доли вариантов при обучении задаёт `matchmaking.layouts`
  (`configs/examples/tron.yaml`).

Хелперы (роль по умолчанию называется `player`):

| Хелпер | Варианты | Пример |
|---|---|---|
| `GameSpec.solo(obs, act, global_state=None)` | `solo` | `coin_grid` |
| `GameSpec.symmetric(n или [n1, n2, ...], obs, act, ...)` | `2p`, `3p`, ... (команда = место) | `tic_tac_toe`, `tron` |
| `GameSpec.teams_of([2, 2], obs, act, ...)` | `2v2`; `[[2, 2], [3, 3]]` — несколько вариантов; `[2, 1, 1]` — `2v1v1` | `team_tag` |
| `GameSpec.teams_of([2], obs, act, ...)` | `coop2` (одна команда) | `coop_buttons` |

**Асимметричные роли** — `GameSpec` целиком руками (`examples/predator_prey/game.py`):

```python
hunter = RoleSpec(Box(-1, 1, (9,), np.float32), Discrete(5))
prey = RoleSpec(Box(-1, 1, (7,), np.float32), Discrete(9))
spec = GameSpec(roles={"hunter": hunter, "prey": prey},
                layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
```

Один агент играет только роли с одинаковыми пространствами (наблюдение, действие и `global_state`).
Роли с разными пространствами — разные агенты:

```yaml
agents:
  hunter: {roles: [hunter]}
  prey: {roles: [prey]}
```

Если `roles` не указаны, агент играет все роли игры, и тогда у всех ролей должны совпадать
пространства (иначе `ConfigError` с подсказкой). Роли с общим пространством, которые различаются
по смыслу (например, «нападающий» и «защитник» одной сети), кодируйте во входе наблюдения.

## 3. `StepResult`: кто ходит, награды, конец эпизода

```python
StepResult(
    acting={...},            # места, которые ходят на СЛЕДУЮЩЕМ шаге
    obs={seat: obs},         # обязательно для каждого ходящего места
    action_masks={seat: m},  # только для ходящих; нет ключа — всё разрешено
    rewards={seat: r},       # любому живому месту; нет ключа — 0
    terminated={...},        # места, выбывшие на этом шаге
    episode_over=False,
    truncated=False,         # оборван искусственным лимитом, а не по правилам
    final_obs=None,          # при truncated: каждому живому месту
    global_state=None,       # для ролей с global_state_space (раздел 6)
    outcome=None,            # при episode_over (раздел 5)
    infos={},
)
```

- `reset(seed, layout)` возвращает `StepResult` с `acting`, `obs`, `action_masks`, `global_state`;
  наград и `terminated` нет. `seed` нужно применять: одинаковый сид — одинаковый эпизод.
- `step(actions)` получает словарь ровно по местам из предыдущего `acting`.
- **Пошаговые игры**: в `acting` одно место (`tic_tac_toe`). **Одновременные ходы**: все живые места
  (`tron`, `unit_harvest`).
- **Шаг без решений** (`acting` пуст, эпизод не закончен) допустим: воркер вызовет `step({})`. Больше
  `env.max_idle_steps` (по умолчанию 1000) таких шагов подряд — ошибка контракта.
- Наблюдения не ходящим местам можно не отдавать: фреймворк их игнорирует, модель места обновляет
  состояние только на своих ходах. **Память между ходами** (туман войны, история ходов соперника)
  должна быть в наблюдении: среда сама кладёт туда то, что место должно помнить.
- **Награды ждущему месту** разрешены (в `tic_tac_toe` проигравший получает −1, когда ходит победитель):
  они копятся в последнем переходе места; награды до первого хода места уходят в его первый переход.

### Жизненный цикл места

| Состояние | Что можно | Что нельзя |
|---|---|---|
| пустое (нет в варианте) | ничего | наблюдение, награда, `terminated`, `acting` |
| живое | ходить, ждать, получать награды | — |
| выбывшее (`terminated` раньше) | ничего | ходить, награды, повторный `terminated` |

- **Выбывание** (`terminated={seat}`) значит «return этого места окончателен». Награда в том же
  `StepResult` разрешена (типичный штраф за смерть) и применяется первой (`tron`, пойманная жертва в
  `predator_prey`).
- **Мёртвый юнит или игрок команды, которому ещё положена командная награда в конце,** *не*
  терминируется: среда просто перестаёт включать его в `acting` и продолжает начислять ему награды.
  `terminal` он получит в конце эпизода. Пример — заморозка в `team_tag`: замороженный бот стоит на
  поле, не ходит и получает общую награду команды. Выбирайте `terminated`, только если места больше
  ничего не ждёт.

### Конец эпизода: правила против обрыва

- **По правилам** (победа, выбывание всех, фиксированная длина матча вроде 505 шагов Lux S3):
  `episode_over=True`, `acting` пуст, `truncated=False`. Bootstrap = 0.
- **Искусственный обрыв** (лимит шагов, которого нет в правилах игры): `episode_over=True`,
  `truncated=True` и `final_obs` для **каждого живого места** (и финальный `global_state`, если роль
  его объявила). Лёрнер посчитает V по `final_obs` своей свежей сетью. `coin_grid` — пример: у игры
  нет конца, лимит в 50 шагов — обрыв.

Если сомневаетесь: известна ли длина матча игроку и является ли она частью игры? Да — это правило
(подайте оставшееся время в наблюдение). Нет — это обрыв.

## 4. Наблюдения: деревья, `uint8`, списки сущностей

`observation_space` роли: `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete` или вложенный `Dict` из
них. Наблюдение — дерево numpy-массивов той же структуры.

- **Тип данных сохраняется** всюду: среда → буфер → чанк → батч лёрнера. Карта в `uint8` остаётся
  `uint8` (в 4 раза меньше трафика, чем `float32`); к `float` приводит энкодер:
  `obs["grid"].float()` (`examples/coin_grid/models.py`).
- **Список сущностей** — соглашение `Dict(entities=Box(N_max, F), entity_mask=MultiBinary(N_max))`:
  строки без сущности — нули, маска — 0. Энкодер усредняет или считает attention только по маске
  (`masked_mean` в `examples/unit_harvest/models.py`).
- Предупреждение для attention-энкодеров: в служебных слотах чанка (`pad`) фреймворк копирует
  предыдущее наблюдение, но если среда может отдать наблюдение с пустой маской сущностей, энкодер не
  должен давать NaN (softmax по пустому множеству). Усреднение с `clamp(min=1)` безопасно.
- **Перспектива места — забота среды.** В симметричных играх удобно отражать доску так, чтобы каждое
  место «начинало слева» (`unit_harvest`, `team_tag`, `space_miners`), и отражать действия обратно.

### Порядок ключей `gymnasium.spaces.Dict`

`gymnasium.spaces.Dict({...})` из обычного `dict` **сортирует ключи по алфавиту**. От порядка
компонентов зависят раскладка маски юнитов и порядок голов. Чтобы порядок был таким, как написано,
передавайте список пар (или `sort_keys=False`):

```python
gymnasium.spaces.Dict([("base", Discrete(2)), ("workers", Units(8, Discrete(5)))])   # порядок как написано
gymnasium.spaces.Dict({"workers": ..., "base": ...})                                  # станет base, workers
```

## 5. Исход матча (`Outcome`)

`outcome` задаётся при `episode_over` и описывает **команды** варианта (в FFA и соло команда = место):

```python
Outcome(team_rank={0: 1.0, 1: 2.0})                # 1 — лучший; ничья — равные ранги; дробные допустимы
Outcome(team_score={0: 7.0, 1: 3.0})               # игровой счёт; ранги выводятся (больше — лучше)
```

- Без `outcome` счёт команды = **среднее** по её местам их наград за эпизод (среднее, а не сумма,
  чтобы команды разного размера сравнивались честно), ранги выводятся из счёта.
- Ключи — ровно команды варианта.
- Общие места в FFA: `tron` делит места поровну между выбывшими в одном шаге (два места за 3-е и
  4-е — оба 3.5).

## 6. `global_state` и централизованный критик

Если у роли есть `global_state_space`, среда отдаёт `global_state[seat]` каждому ходящему месту этой
роли на каждом шаге (и при `reset`), а при обрыве — каждому живому месту. Он доходит только до
value-пути модели: `networks.critic_encoder_class` (подкласс `BaseCriticEncoder`) превращает его в
вектор, который конкатенируется с признаками ядра перед value-головой. Политика его не видит, поэтому
на соревновании модель работает без него.

```yaml
networks:
  encoder_class: examples.team_tag.models.TagEncoder          # локальное окно
  critic_encoder_class: examples.team_tag.models.TagCriticEncoder   # вся карта
```

`global_state` хранится в каждом слоте чанка, поэтому он стоит памяти и трафика: замер на `team_tag`
— в `docs/benchmarks.md`. Используйте `uint8` и компактные представления.

## 7. Действия: деревья, `Units`, маски

`action_space` роли: `Discrete`, `MultiDiscrete`, `Box` (1-D), `Units` или вложенный `Dict` из них.
Действие — дерево: `int64` для дискретных частей, `float32` для `Box`. `Box`-действия фреймворк не
обрезает: обрезайте в среде (`np.clip`).

### `Units`: один бот, много юнитов

```python
from colosseum.envs.spaces import Units

Units(max_units, per_unit, only_if=None)
# per_unit: Discrete(A) | MultiDiscrete([A1, ...]) | Box(d,) | Dict из Discrete и Box
```

- Действие группы — массив с ведущим измерением `[U]`: `int64[U]` для `Discrete`, `int64[U, C]` для
  `MultiDiscrete`, `float32[U, d]` для `Box`, словарь таких массивов для `Dict`.
- Индекс слота назначает среда; фреймворк не связывает слоты между шагами. Новый юнит — в свободный
  слот, погибший освобождает слот (`unit_harvest`).
- Разные типы юнитов — `Dict` из нескольких `Units`:
  `Dict([("factories", Units(10, Discrete(4))), ("robots", Units(200, MultiDiscrete([5, 64]))), ("global", Discrete(3))])`.
- **GridNet** — частный случай `Units(H * W, ...)`: юнит = клетка. Хелпер
  `colosseum.networks.heads.gridnet_to_units` перекладывает `[B, C, H, W]` в `[B, H*W, C]`.
- **Цель-указатель** — компонент `Discrete(N_targets)` с маской по существующим целям.
- **`only_if={компонент: (родитель, {значения})}`**: компонент учитывается (log-prob, энтропия, KL,
  лосс), только если выбранное значение родителя (дискретный компонент того же юнита) входит в
  множество. Пример: цель `sap` учитывается, только если тип действия = sap. Боту без юнитов с
  условной головой подойдёт `Units(1, ...)`.
- Решения одного места раскладываются на K «решающих»: каждый юнит — один, все не юнитовые части
  вместе — ещё один. Режимы `algorithm.ratio_mode` / `unit_trace` / `entropy_reduction` и итоги
  эксперимента по K=8 и K=128 — в `docs/benchmarks.md`.

Модель юнитов: энкодер отдаёт `EncoderOutput(latent, aux={"units": [B, U, E]})`, политика собирает
параметры через `UnitsHead` (`examples/unit_harvest/models.py`):

```python
params = {"base": self.base_head(features),
          "workers": self.workers_head(torch.cat([units, context], dim=-1))}
return make_distribution(self.action_spec, params)
```

### Маски

Маска — дерево, повторяющее дискретные части действия; нет листа — всё разрешено.

| Пространство | Маска |
|---|---|
| `Discrete(n)` | `bool[n]` |
| `MultiDiscrete([A1, ...])` | `bool[A1 + ...]` |
| `Box` | нет |
| `Units` | `{"unit": bool[U], "action": bool[U, сумма размеров дискретных компонентов]}` |

- У ходящего места дискретный лист без единого разрешённого действия — ошибка контракта. Держите
  хотя бы одно «безопасное» действие (стоять, пас).
- В `Units` строка `unit=False` — юнита нет; пустая строка компонента у существующего юнита делает
  этот компонент невалидным без ошибки. Пример маски внутри юнитов — `push` в `space_miners`
  разрешён, только если астероид рядом.

## 8. Модель

`networks` в конфиге: либо `model_class` (монолитная модель, реализует `step` и `unroll` протокола
`PolicyModel`), либо части `ComposedModel`:

```yaml
networks:
  encoder_class: ...         # obs-дерево -> Tensor [B, D] или EncoderOutput(latent, aux)
  core: null                 # или {class: colosseum.networks.cores.LSTMCore, kwargs: {...}}
  policy_class: ...          # (features, aux) -> Distribution (make_distribution)
  value_class: ...           # features [B, D (+ G)] -> [B]
  critic_encoder_class: null # global_state -> [B, G]
  kwargs: {}                 # передаются каждому конструктору
```

Конструкторы получают по имени то, что объявляют: `observation_space`, `action_space`,
`global_state_space`, `action_spec`; головы — `in_dim`. Поэтому одни и те же классы работают для
ролей с разными размерами (`examples/predator_prey/models.py`).

## 9. Проверка

```bash
colosseum validate -c configs/examples/<game>.yaml
colosseum train -c configs/examples/<game>.yaml --set training.total_timesteps=20000 --set run.name=smoke
```

Ошибки контракта приходят как `EnvContractError` с контекстом «воркер, среда, место, шаг эпизода,
вариант» — по нему видно, какое правило нарушено.
````

- [ ] **Step 3: Check that the guide's names and paths are real**

```bash
.venv/bin/python - <<'EOF'
import importlib, pathlib, re
text = pathlib.Path("docs/ENV_GUIDE.md").read_text()
for p in sorted(set(re.findall(r"`((?:examples|configs/examples|docs)/[\w./-]+)`", text))):
    assert pathlib.Path(p).exists() or pathlib.Path(p + ".py").exists(), p
for mod, name in [("colosseum.envs.game", "GameSpec"), ("colosseum.envs.game", "MultiAgentEnv"),
                  ("colosseum.envs.game", "StepResult"), ("colosseum.envs.game", "Outcome"),
                  ("colosseum.envs.game", "RoleSpec"), ("colosseum.envs.game", "SeatSpec"),
                  ("colosseum.envs.spaces", "Units"), ("colosseum.networks.heads", "UnitsHead"),
                  ("colosseum.networks.heads", "gridnet_to_units"), ("colosseum.networks.heads", "make_distribution"),
                  ("colosseum.networks.base", "BaseCriticEncoder"), ("colosseum.networks.base", "EncoderOutput"),
                  ("colosseum.core.errors", "EnvContractError")]:
    assert hasattr(importlib.import_module(mod), name), (mod, name)
print("guide ok")
EOF
```

Expected: `guide ok`. A failing path means the guide names a file that does not exist: fix the guide.

Then run the guide's minimal example (section 1) through `validate` to prove it is correct. The model is any flat-Box MLP with a categorical head; the coop_buttons classes fit a `Box(1)` observation and a `Discrete(2)` action:

```bash
REPO=$(pwd)
mkdir -p /tmp/envguide/my_game
sed -n '/^## 1\. Минимальная среда/,/^Конфиг:/p' docs/ENV_GUIDE.md | sed -n '/^```python$/,/^```$/p' | sed '1d;$d' \
  > /tmp/envguide/my_game/env.py
touch /tmp/envguide/my_game/__init__.py
cat > /tmp/envguide/my_game/config.yaml <<'EOF'
env: {env_class: my_game.env.Counter, kwargs: {length: 10}}
networks:
  encoder_class: examples.coop_buttons.models.CoopEncoder
  core: null
  policy_class: examples.coop_buttons.models.CoopPolicy
  value_class: examples.coop_buttons.models.CoopValue
EOF
(cd /tmp/envguide && PYTHONPATH=$REPO $REPO/.venv/bin/colosseum validate -c my_game/config.yaml); echo "exit=$?"
rm -rf /tmp/envguide
```

Expected: `exit=0`. A failure means the guide's example is wrong: fix the guide's example (and rerun), not the framework.

- [ ] **Step 4: Replace `README.md`**

Replace `README.md` with the text below, then fill the `@...@` tokens:
- `@TTT_TIME@`: the tic-tac-toe training time from the report draft (T8.3), as "about N seconds (M–K s over three seeds)";
- `@TTT_WINS@`: the range of the tic-tac-toe win rates over the three seeds, e.g. `86–94%`;
- `@UNIT_TRACE_DEFAULT@`: the T8.4 ruling's value in backticks, e.g. `` `geo_mean` ``;
- `@FAST_COUNT@`: the number printed by `.venv/bin/python -m pytest -m "not gpu and not slow" --collect-only -q | tail -1`;
- `@SLOW_TIME@`: the wall time of T8.3 Step 7, e.g. `about 25 min`.

````markdown
# Colosseum

A reusable training framework for competitive bot-programming competitions (Lux AI, Neural MMO, CodeCraft, ...):
behavioural cloning → RL → self-play → league, built on an IMPALA-style asynchronous actor–learner (APPO with V-trace)
in PyTorch, without Ray.

> **Status (SP2 of 6).** Single-machine training is the supported mode for every game structure: solo, 1v1 with
> turn-based or simultaneous moves, one bot controlling many units, team vs team, free-for-all with elimination and
> 2–4 players in one run, asymmetric roles, and cooperative games. Each structure has a demo game that is trained
> end-to-end in the test suite. Read [Status and limitations](#status-and-limitations) before relying on anything else.

## Quick start

```bash
git clone <repo-url> Colosseum && cd Colosseum
scripts/setup-dev.sh            # uv + .venv (Python 3.12) + CPU torch + colosseum[grpc,dev,examples]
source .venv/bin/activate

colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
```

`scripts/setup-dev.sh --gpu` installs the CUDA build of torch instead of the CPU one.

Training prints its run directory and one progress line per agent every 10 seconds (log lines go to stderr; the
same lines are in `runs/ttt-quickstart/logs/main.log`). On an 8-core CPU the run takes @TTT_TIME@. Afterwards the
latest checkpoint, played greedily, wins @TTT_WINS@ of games against a random legal-move player
(`tests/learning/test_demo_learning_slow.py` requires >= 80%). Re-running with the same `run.name` is refused; pick
another name or delete `runs/ttt-quickstart`.

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

Other game structures (each config trains in about three minutes or less on 8 cores):

```bash
colosseum train -c configs/examples/tron.yaml --set run.name=tron                    # FFA, 2/3/4 players, elimination
colosseum train -c configs/examples/predator_prey.yaml --set run.name=predator-prey  # asymmetric roles, two agents
colosseum train -c configs/examples/coop_buttons.yaml --set run.name=coop            # cooperative, mixed teammates
```

`runs/` and `eval.json` are git-ignored.

## Demo games

| Game | Config | Structure | Contract features |
|---|---|---|---|
| `coin_grid` | `coin_grid.yaml` | solo | Dict observation with a `uint8` grid, masks, step-limit truncation |
| `tic_tac_toe` | `tic_tac_toe.yaml` | 1v1 turn-based | one acting seat, masks, rewards to the waiting seat |
| `unit_harvest` | `unit_harvest.yaml` | one bot, many units, simultaneous | `Units`, units born and killed, entity lists with masks |
| `team_tag` | `team_tag.yaml` | 2v2 | local window per bot, `global_state` for a centralized critic, frozen teammates |
| `tron` | `tron.yaml` | FFA 2p/3p/4p | elimination (`terminated`), ranks by elimination order, several layouts in one run |
| `predator_prey` | `predator_prey.yaml` | 1 vs 2 | roles with different spaces, one agent per role |
| `coop_buttons` | `coop_buttons.yaml` | cooperative | one team, `score` outcome, mixed teammates, cross-play table |
| `space_miners` | `space_miners.yaml` | 1v1 (Box2D) | units with a continuous and a discrete component, entity list; reference only |
| `composite_action` | `chase.yaml` | 1v1 | tree action `Dict(direction=Discrete, speed=Box)`; reference only |

The slow learning tests train each of the first seven and check it against random players: coin_grid scores at least
twice the random score; tic-tac-toe, unit_harvest and team_tag win at least 80%; tron wins 80% of 2p games and takes
first place in at least half of 4p games against three random cycles; each predator_prey role beats a random
opponent at least 70%; both coop_buttons agents' homogeneous teams beat a threshold set from a measured random and
scripted baseline.

## Writing your own game

Read [`docs/ENV_GUIDE.md`](docs/ENV_GUIDE.md) (in Russian): how to write an env for every game type, with the demo
games as examples. In short:

```python
# my_game/env.py
import gymnasium, numpy as np
from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

class MyGame(MultiAgentEnv):
    def __init__(self):
        obs = gymnasium.spaces.Box(0, 1, (16,), np.float32)
        self.spec = GameSpec.symmetric(2, obs, gymnasium.spaces.Discrete(4))   # layout "2p"

    def reset(self, seed, layout):
        return StepResult(acting={0}, obs={0: np.zeros(16, np.float32)}, action_masks={0: np.ones(4, bool)})

    def step(self, actions):                       # actions: {seat: action} for exactly the acting seats
        ...
        return StepResult(acting={1}, obs={1: ...}, action_masks={1: ...}, rewards={0: 0.0, 1: 0.0})
        # at the end: StepResult(acting=set(), obs={}, rewards=..., episode_over=True,
        #                        outcome=Outcome(team_rank={0: 1.0, 1: 2.0}))
```

- `GameSpec.solo`, `GameSpec.symmetric(n or [2, 3, 4])`, `GameSpec.teams_of([2, 2])` or a hand-written
  `GameSpec(roles=..., layouts=...)` describe roles (observation, action and optional `global_state` spaces), seats
  and teams. The outcome kind follows the number of teams: one → `score`, two → `wdl`, three or more → `rank`.
- `StepResult.acting` says who moves next; `terminated` eliminates seats; `truncated=True` with `final_obs` marks an
  artificial cut (the learner bootstraps from `final_obs`); `outcome` gives team ranks or scores.
- Observations and actions are trees (`Dict`); `uint8` stays `uint8` up to the model. Many units per bot:
  `colosseum.envs.spaces.Units(max_units, per_unit, only_if=None)`.
- Contract violations raise `EnvContractError` with "worker, env, seat, episode step, layout"; run
  `colosseum validate -c my_game.yaml` first.

The default model is `encoder → core → policy head / value head` (+ an optional critic encoder over `global_state`):

```python
# my_game/networks.py
import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution

class Encoder(BaseEncoder):
    def __init__(self, observation_space, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(observation_space.shape[0], 64), nn.ReLU())
    @property
    def latent_dim(self): return 64
    def forward(self, obs): return self.net(obs)

class Policy(BasePolicy):
    def __init__(self, in_dim, action_spec, **kwargs):      # in_dim = the core's output size
        super().__init__(); self.action_spec = action_spec; self.net = nn.Linear(in_dim, 4)
    def forward(self, features, aux): return make_distribution(self.action_spec, self.net(features))

class Value(BaseValue):
    def __init__(self, in_dim, **kwargs):
        super().__init__(); self.net = nn.Linear(in_dim, 1)
    def forward(self, features): return self.net(features).squeeze(-1)
```

Constructors receive by name what they declare: `observation_space`, `action_space`, `global_state_space`,
`action_spec`, and `in_dim` for heads. Cores (`colosseum.networks.cores`): `null` (no memory), `LSTMCore` /
`GRUCore` (`hidden_size`, `num_layers`), `WindowAttentionCore` (`d_model`, `window`, `num_heads`, `num_layers`). For
anything else subclass `colosseum.networks.model.PolicyModel` (`step` and `unroll`) and set `networks.model_class`.

```yaml
# my_game/config.yaml
run: {name: null, dir: runs}
env: {env_class: my_game.env.MyGame, kwargs: {}}
networks:
  encoder_class: my_game.networks.Encoder
  core: null
  policy_class: my_game.networks.Policy
  value_class: my_game.networks.Value
training: {total_timesteps: 2000000}
```

```bash
colosseum validate -c my_game/config.yaml
colosseum train -c my_game/config.yaml
```

The current directory is put on `sys.path` (also for the spawned worker and learner processes), so run the commands
from the directory that contains `my_game/`.

## What a run writes

```
runs/<name>/
  config.resolved.yaml      the config after --set overrides and agent merging
  logs/main.log             main process (also printed to the console)
  logs/learner-<agent>.log  one file per learner
  logs/worker-<i>.log       one file per rollout worker (worker-<i>-env<k>.log for subprocess envs)
  metrics.jsonl             one JSON record per line (below)
  ratings.json              per layout: ELO, win rates, wr_vs_past, scores, cross-play; rewritten periodically
  checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
```

`metrics.jsonl` records (`"kind"` field):

| kind | content |
|---|---|
| `train` | APPO metrics of one agent every `metrics.log_interval` train steps: losses, entropy, `approx_kl`, `clip_fraction` (per decider) and `clip_fraction_joint`, `ess`, `log_rho_abs_mean/p95`, `rho_clip_frac`, `c_clip_frac`, `deciders_valid_mean/max`, `boot_frac`, `pad_frac`, `explained_variance`, `grad_norm`, `lr`, ... |
| `episodes` | per agent, layout and role: episodes, returns, lengths, team scores, elimination share, W/D/L by opponent type |
| `ratings` | per layout: `elo`, `win_rates`, `games`, `wr_vs_past`, `scores`, `cross_play` |
| `system` | `env_steps`, `env_steps_per_sec`, `train_steps_per_sec`, learner `queue_depths`, `parked_buffers`, `workers_reporting` |

`meta.json` of a checkpoint holds the agent's `networks` section, its roles and the signature of their spaces, so
`colosseum eval` can rebuild any checkpoint and resume/eval refuse a checkpoint whose spaces do not match.

### WandB

WandB is optional and only a viewer; `metrics.jsonl` stays the source of truth. Install the extra with
`uv pip install --python .venv/bin/python -e ".[wandb]"` and set `metrics.use_wandb: true`. Per-agent training
metrics are `<agent>/<metric>` on the axis `<agent>/train_step`; `ratings/<layout>/*`, `system/*` and
`episodes/*` are on the axis `env_steps`. A WandB failure logs one warning and disables WandB; it never stops
training.

## Process lifecycle

- `training.total_timesteps` is the global number of env steps over all workers. When it is reached, every learner
  sends a final checkpoint, the main process saves it, all processes stop, and the exit code is 0.
- If any worker or learner dies, training stops with exit code 1 and a message like
  `worker-0 died (exit -9), see runs/<name>/logs/worker-0.log`.
- Ctrl-C (SIGINT) and SIGTERM stop the run: final checkpoints are saved within a shutdown grace of 7 seconds; no
  process is left after 10 seconds. The exit codes are 130 (SIGINT) and 143 (SIGTERM).
- An invalid config exits with code 1 and a `Config error: ...` message without a traceback.
- `colosseum eval` uses the same codes (0 / 1 / 130 / 143), plus 2 for bad command-line arguments.

## Configuration reference

Unknown keys are errors at every level. `--set key=value` accepts YAML values (`null`, numbers such as `1e-4`, lists,
mappings; quote a value to keep it a string) and works for `train`, `validate`, `run-learner` and `run-workers`:

```bash
colosseum train -c configs/examples/tron.yaml --set run.name=tron-4p --set 'matchmaking.layouts={4p: 1.0}'
```

Inside a list, YAML 1.1 rules apply: `--set x=[1e-4,2]` keeps `1e-4` as a string; write `1.0e-4` there.

| section | keys (defaults) |
|---|---|
| `run` | `name` (null → `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is an error), `dir` (`runs`) |
| `env` | `env_class`, `kwargs`, `max_idle_steps` (1000 consecutive steps without an acting seat is an error) |
| `networks` | `model_class` (null) or `encoder_class` + `core` (`{class, kwargs}` or null) + `policy_class` + `value_class` + `critic_encoder_class` (null; needs a role with `global_state_space`); `kwargs` (passed to every constructor) |
| `algorithm` | `algorithm_class` (APPO), `gamma` 0.99, `vtrace_lambda` 1.0, `vtrace_rho_bar` 1.0, `vtrace_c_bar` 1.0, `eps_clip` 0.2, `value_loss_coeff` 0.5, `entropy_coeff` 0.01, `max_grad_norm` 0.5, `num_epochs` 1, `minibatch_chunks` 0, `learning_rate` 3e-4, `lr_schedule` (`linear`; also `constant`, `cosine`), `normalize_advantages` (true), `use_amp` (false), `amp_dtype`, `use_torch_compile` (false), `ratio_mode` (`auto` = `per_unit` with `Units`, else `joint`), `unit_trace` (`auto` = `joint` without `Units`, @UNIT_TRACE_DEFAULT@ with `Units`; also `none`), `entropy_reduction` (`auto` = `sum` for `joint`, `mean_valid` for `per_unit`) |
| `rollout` | `chunk_length` 256 (>= 2), `num_workers` 4, `envs_per_worker` 8, `torch_threads` 1, `weight_sync_interval_sec` 5, `vec_env` (`sync`/`subprocess`), `subproc_workers`, `match_refresh_interval_sec` 30 |
| `learner` | `device` (`auto`), `batch_chunks` 16 (every update uses exactly this many chunks), `queue_size` 64, `weight_push_interval` 5, `torch_threads` (auto), `pin_memory` (false) |
| `training` | `total_timesteps` (global env steps), `seed`, `resume_from`, `kickstart_teacher`, `kickstart_lambda` 1.0, `kickstart_decay_steps` 50000, `kickstart_kl` (`forward` / `reverse`) |
| `matchmaking` | `mode` (`self_play` / `league`), `layouts` (`{layout: weight}`; empty = every layout equally), `self_play_ratio` 0.5, `pfsp_exponent` 1.0, `latest_prob` 0.5, `teammates` (`self` / `mixed`), `teammate_self_prob` 0.5, `shuffle_seats` (true) |
| `checkpoint` | `interval` 1000 (train steps), `pool_size` 20 (FIFO per agent), `save_optimizer` (true) |
| `metrics` | `log_interval`, `console_interval_sec` 10, `use_wandb`, `wandb_project`, `wandb_entity` |
| `bc` | `seq_len` 64 (sequence length for stateful models in `colosseum bc`) |
| `transport` | `grpc_max_message_mb` 64 (distributed mode) |
| `agents` | `{agent_id: {roles: [...], networks: {...}, algorithm: {...}, learner: {...}}}`: roles and partial overrides, deep-merged onto the global sections |

**Agents and roles.** Without an `agents` section there is one agent, `agent_0`, playing every role (then all roles
must have the same spaces). With it, every key is a trainable agent with its own learner. `roles` lists the roles an
agent plays; the roles of one agent must have the same observation, action and `global_state` spaces. An agent id may
contain letters, digits, `_` and `-` (not starting with `-`), no `.`, and cannot be `ratings`, `system`, `episodes` or
`train`.

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index. For each match the matchmaker
picks a layout (`matchmaking.layouts`, only layouts with a seat for the owner's roles), a match type once per match
(self-play with probability `self_play_ratio`, else arena; `mode: self_play` means always self-play), and the owner's
team. Every other team gets a core: in self-play the owner's latest weights (probability `latest_prob`) or one of
its checkpoints, if the owner plays a role of that team; otherwise (arena, or roles the owner does not play) another
trainable agent picked by PFSP, `(1 - win_rate)^pfsp_exponent`, on that layout. `teammates: self` fills a team with
its core; `mixed` gives each other seat to the core with probability `teammate_self_prob`, else to another trainable
agent's latest or a checkpoint of the core. All seats with latest weights collect data; checkpoint seats do not.
`shuffle_seats` permutes teams with the same role composition and same-role seats within a team.

**Ratings** are kept per layout. Two or more teams: ELO over team pairs (each pair is one comparison by rank; its
weight `K / (T - 1)` is split among the counted member pairs of different agents), a fractional win-rate matrix and
`wr_vs_past` (the latest weights against the agent's own checkpoints, last 500 pairs). One team: mean score with EMA
and a 95% interval per agent, and with `teammates: mixed` a cross-play table "team composition → mean score".

**Resume.** `training.resume_from` accepts a checkpoint dir, a previous run dir (each agent takes its latest
checkpoint there) or a `.pt` state dict (weights only, e.g. from `colosseum bc`). An explicit resume is strict: a
missing or malformed `meta.json`, or a role signature that does not match the agent's roles, is a config error naming
the path.

## Behavioural cloning and kickstarting

```bash
colosseum bc -c my_game/config.yaml --agent agent_0 --data path/to/demos/ --output bc.pt --epochs 20
colosseum train -c my_game/config.yaml --set training.resume_from=bc.pt
```

BC data: `.pt` files with the trees `observations`, `actions` and optional `action_masks` and `dones`. The network
and roles come from the `--agent` (default: the only agent). The loss is `-log_prob` of the recorded action (the mean
over valid deciders for actions with units); masks are applied; stateful models train on sequences of `--seq-len`
steps that reset at `dones`.

Kickstarting adds `lambda * KL(teacher‖student)` per decider to the RL loss, decaying linearly over
`kickstart_decay_steps`: `--set training.kickstart_teacher=bc.pt`. The teacher's spaces must match every agent's roles.

## Evaluation

```bash
colosseum eval -c cfg.yaml -a A=<ckpt-dir|.pt> [-a B=...] [--layout L ...] --num-matches N [--output r.json] [--deterministic] [--seed S]
```

Matches run on the same engine as training (`MatchRunner`): per-seat model state, acting seats and masks. An agent's
architecture and roles come from its checkpoint's `meta.json`; a `.pt` is built from the config agent of the same
name, or from the global `networks` with every role. By default every layout with seats for the agents is played.
- Two or more teams, two or more agents: for every pair (a, b) match m gives team i to `(a, b)[(i + m) % 2]` where
  the roles allow; teams are filled homogeneously; `--num-matches` is per pair and layout.
- Two or more teams, one agent: all teams are that agent.
- One team: each agent in a homogeneous team, plus mixed compositions for cross-play.

Reports per layout: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side; `rank` — mean rank with a
95% interval, first-place share, a pairwise "who placed higher" matrix; `score` — mean score with a 95% interval and
the cross-play table. Everything is also broken down by role. `--output` writes JSON.

The in-process API `colosseum.eval.play_lineups(env_fn=..., models=..., lineups=...)` plays any `PolicyModel`
instances in explicit lineups (the learning tests use it with a random legal-move player).

## Distributed mode (limited)

```bash
colosseum serve-weight-store --port 50051                                                     # machine A
colosseum run-learner -c cfg.yaml --agent agent_0 --traj-port 50052 --weight-store A:50051    # machine B
colosseum run-workers -c cfg.yaml --weight-store A:50051 -l agent_0=B:50052                   # machines C, D, ...
```

Trajectory chunks travel as numpy trees over gRPC. Limitations until SP5: games where one agent plays every role only
(the layout is picked by `matchmaking.layouts`, every seat plays the latest weights); no coordinator, league, ratings,
`metrics.jsonl` or WandB; every `run-workers` host collects `total_timesteps / num_workers` per worker, and the
learner's progress is `consumed_samples / total_timesteps` (decisions, not env steps); `run-learner` cannot resume; no
fault tolerance or authentication; `deployment/` (Docker, Kubernetes) is untested.

## Status and limitations

| | Sub-project | Content |
|---|---|---|
| ✔ | SP1 Foundation and stabilization | correct single-machine training, stateful model protocol, run dir, metrics, lifecycle |
| ✔ | SP2 Game model | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic |
| | SP3 Players, league, warm start | scripted, frozen and external players; PFSP over snapshots; per-agent `init`/kickstart/critic warm-up; top-k snapshot storage |
| | SP4 Selection and observability | match log, OpenSkill / Bradley–Terry ratings, `colosseum tournament`, dashboard, snapshot ratings |
| | SP5 Distributed | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| | SP6 Speed and extensions | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

Known limitations today:
- **Players:** only trainable agents; no scripted, frozen or external players in training (SP3). PFSP picks among
  agents' latest weights, not snapshots. Online ELO is a progress indicator, not a selection-grade rating.
- **Game model:** one agent cannot play roles with different spaces; no per-unit rewards or values; no PettingZoo
  adapter.
- **Speed:** the worker handles observation trees in Python per seat (hundreds of seats per env, as in Neural MMO,
  need SP6's vectorized path); recurrent and attention cores unroll step by step on the learner; worker inference is
  CPU only.
- **Distributed:** see above.
- **GPU:** not run on CUDA yet. All CUDA paths are covered only by `gpu`-marked tests; see
  [`docs/GPU_CHECKS.md`](docs/GPU_CHECKS.md).

## Tests

```bash
.venv/bin/python -m pytest -m "not gpu and not slow" -q     # full fast suite (CI), @FAST_COUNT@ tests
.venv/bin/python -m pytest -m slow -v                       # learning tests of every demo game (@SLOW_TIME@)
.venv/bin/python -m pytest -m gpu -v                        # CUDA machine only, see docs/GPU_CHECKS.md
```

Layout:
- `tests/unit`: pure functions and classes;
- `tests/contract`: the real `MatchRunner` / `RolloutLoop` feeding the real APPO in-process, the demo games;
- `tests/integration`: CLI and multi-process runs;
- `tests/learning`: does it learn (fast bandits; the slow demo-game runs).

Tests write only under pytest's `tmp_path`. Throughput and the units experiment are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Project structure

```
src/colosseum/
  cli.py launcher.py distributed.py eval.py
  core/         config, types (chunk v2, lineups, results), tree, specs (ObsSpec, ActionSpec, masks), roles,
                registry, outcomes, errors, ipc, run_dir, threads
  envs/         game (GameSpec, MultiAgentEnv, StepResult), spaces (Units), contract (EpisodeTracker), vector
  networks/     model (PolicyModel, act), composed, base, heads (UnitsHead), dist/, cores, state, normalization
  algorithms/   appo, vtrace, base
  worker/       match_runner (MatchRunner), rollout_loop (RolloutLoop), buffers, rollout_worker
  learner/      learner process, checkpoint payloads
  coordinator/  coordinator, matchmaker (lineups), ratings (per layout), checkpoint_manager, agent_pool
  metrics/      jsonl, aggregator, console, hub, wandb_logger
  transport/ weight_store/ bc/ utils/
examples/       coin_grid, tic_tac_toe, unit_harvest, team_tag, tron, predator_prey, coop_buttons,
                space_miners (Box2D), composite_action (chase)
configs/examples/
scripts/        setup-dev.sh, bench_throughput.py, units_experiment.py, measure_global_state.py
docs/           ENV_GUIDE.md, benchmarks.md, GPU_CHECKS.md
```
````

Check: `grep -n "@[A-Z_]*@" README.md` prints nothing.

Every statement in the README must match the code. Check these against the repository and correct the README text where they differ (the README follows the code; spec deviations are reported, not hidden):
- the `metrics.jsonl` keys in the table (`grep -o '"[a-z_]*"' src/colosseum/algorithms/appo.py | sort -u` for train metrics; `src/colosseum/metrics/jsonl.py::REQUIRED_KEYS`);
- the `ratings.json` layout (open one from a T8.3 run);
- the `Project structure` tree (`ls src/colosseum src/colosseum/*/`);
- the eval report contents (`colosseum eval` on a `tron` checkpoint with two `-a`, `--layout 4p`).

- [ ] **Step 5: Update `CLAUDE.md`**

1. Write the new status block to `/tmp/claude_status.md` (the text below), filling the tokens:
   - `@FAST_COUNT@`, `@SLOW_COUNT@`, `@GPU_COUNT@`: from `pytest --collect-only -q` with `-m "not gpu and not slow"`, `-m slow`, `-m gpu`;
   - `@REPORT_FILE@`: the basename of `docs/superpowers/reports/*-sp2-acceptance.md`;
   - `@UNIT_TRACE_DEFAULT@`: the T8.4 ruling's value;
   - `@OPEN_ITEMS@`: the open-items block below, where `@SP2_OPEN_ITEMS@` becomes one `  - ...` line per item parked during SP2 (controller rulings that say "parked", the T8.4 `ratio_mode` note if any); `  - none` if there are none.

````markdown
## Implementation Status

State after SP2 (game model; developed on branch `sp2-game-model`, merged into `main` only after the owner accepts the acceptance report). Test suite: **@FAST_COUNT@** fast tests, **@SLOW_COUNT@** slow, **@GPU_COUNT@** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite). The SP2 spec is `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md`, the plan `docs/superpowers/plans/2026-10-08-sp2-game-model/`, the acceptance report `docs/superpowers/reports/@REPORT_FILE@`. SP1: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the pre-SP1 review `review/README.md` is a frozen snapshot (never edit it).

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: `GameSpec` and config checks, `reset` of every layout and random legal steps with full contract checks, `step` and `unroll` of every agent's model.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a ...] [--layout L ...] --num-matches N --output result.json [--deterministic] [--seed S]`.
- `colosseum bc -c cfg.yaml --agent A --data <file-or-dir> --output bc.pt [--epochs E] [--seq-len L]`.
- Distributed (limited): `colosseum serve-weight-store --port P`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.
- Measurements: `scripts/bench_throughput.py`, `scripts/units_experiment.py`, `scripts/measure_global_state.py` (results in `docs/benchmarks.md`).

### Works (single machine)
- **Game model (SP2):**
  - env contract `GameSpec` (roles, layouts = lists of seats with roles and teams) + `MultiAgentEnv` + `StepResult` (`acting`, `terminated`, `truncated` + `final_obs`, `global_state`, `Outcome`); `EpisodeTracker` checks every rule and raises `EnvContractError` with "worker, env, seat, episode step, layout";
  - outcome kind from the number of teams (1 → `score`, 2 → `wdl`, ≥3 → `rank`); default outcome = mean of a team's seat returns;
  - seat lifecycle: empty / live / eliminated; a dead teammate is removed from `acting` but keeps team rewards; steps without decisions bounded by `env.max_idle_steps`;
  - observation, action, mask and `global_state` trees with preserved dtypes (`uint8` reaches the model); entity lists with masks; `Units(max_units, per_unit, only_if)` with `Discrete`/`MultiDiscrete`/`Box`/`Dict` components; deciders (each unit + the non-unit part).
- **Training loop:** IMPALA-style; one `MatchRunner` core for training (`RolloutLoop`) and eval. Workers infer only the policy; chunk v2 with `act`/`boot`/`pad` slots; the learner computes every value (bootstrap from `boot` slots and `final_obs`) with its current network. Only numpy payloads cross process boundaries. Every learner update uses exactly `learner.batch_chunks` chunks; `training.total_timesteps` is a global env-step budget.
- **Algorithm:** APPO with V-trace over slots; `ratio_mode` (`joint` / `per_unit`), `unit_trace` (`joint` / `geo_mean` / `none`; `auto` = @UNIT_TRACE_DEFAULT@ with `Units`, decided by the units experiment), `entropy_reduction`; every loss and metric reduces over `act` slots; per-unit diagnostics (`clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `deciders_valid_mean`, ...). Centralized critic: `networks.critic_encoder_class` over `global_state`, value path only.
- **Models:** `PolicyModel` protocol (`step` for the policy path, `unroll` for the learner, `with_value=False` for BC and the kickstart teacher); `ComposedModel(encoder, core, policy, value, critic_encoder)` with `EncoderOutput(latent, aux)` for per-unit features; cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`; `UnitsHead`, `gridnet_to_units`; `NormalizeObs` on a leaf path. The learner reproduces the worker's joint and per-unit log-probs for all four cores (contract tests).
- **League:** lineups (layout + seat assignments); owner rotation; layout weights; self-play / arena with PFSP per layout; teams with a core and `teammates: self | mixed`; roles per agent (`agents.<id>.roles`), asymmetric games with one agent per role; seat permutation within equal role compositions. Ratings per layout: team-pair ELO, win-rate matrix, `wr_vs_past`, `ScoreTracker` and cross-play for one-team layouts. FIFO checkpoint pool with roles and role signature in `meta.json`; strict resume.
- **Eval:** `MatchRunner`-based; CLI lineups per spec block 8; reports for `wdl` (Wilson intervals, per side), `rank` (mean rank, first places, pairwise matrix), `score` (mean, cross-play), by role; in-process `play_lineups` with any `PolicyModel`.
- **Demo games** (`examples/`, configs in `configs/examples/`): `coin_grid` (solo), `tic_tac_toe` (1v1 turn-based), `unit_harvest` (units), `team_tag` (2v2, `global_state`), `tron` (FFA 2–4, elimination), `predator_prey` (asymmetric roles), `coop_buttons` (cooperative); reference examples `space_miners` (Box2D) and `composite_action` (chase). Each demo game has a slow learning test against random players (`tests/learning/test_demo_learning_slow.py`); fast learning tests cover a Units bandit and a cooperative bandit. `docs/ENV_GUIDE.md` (Russian) explains how to write an env for every game type.
- **Observability, config, lifecycle:** as in SP1 (run dir, `metrics.jsonl`, `ratings.json`, console, optional WandB, strict pydantic config with `--set`, exit codes 0/1/130/143, `SHUTDOWN_GRACE_SEC = 7`, no child alive after 10 s), with per-layout and per-role breakdowns.

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) runs on chunk v2 for games where one agent plays every role, latest weights only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`.
- **`deployment/`** (Docker, K8s) is not tested.
- **Unused, kept for SP5:** `transport/local.py::LocalTransport`, `weight_store/shared_memory.py::SharedMemoryWeightStore`, `transport.mode` / `transport.grpc_port`.
- **GPU:** never run on CUDA. All CUDA paths (learner device, AMP, `pin_memory`, kickstart/BC, tree batches and `Units` distributions on CUDA) are covered only by `gpu`-marked tests; see `docs/GPU_CHECKS.md`.

### Not implemented (see Roadmap)
- Scripted, frozen and external players; PFSP over snapshots; per-agent warm start (SP3).
- One agent on roles with different spaces; per-unit rewards, values or recurrent state; built-in autoregression between action components (a custom `Distribution` can do it); a PettingZoo adapter.
- OpenSkill / Bradley–Terry ratings, a tournament command, match log, dashboard (SP4).
- Off-policy algorithms (R2D2/DQN, replay buffer), SAC, AlphaZero/MuZero.
- GPU inference on workers; a vectorized worker path for envs with hundreds of seats (SP6).
- Shared-memory weight store and trajectory ring buffer; adding or removing agents and machines during a run; gRPC control plane (SP5).

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** (done) | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic, demo games |
| **SP3 Players, league, warm start** | scripted / frozen / external players, PFSP over snapshots (including opponent checkpoints in asymmetric games), per-agent warm start (`init`, kickstart, critic warm-up), top-k snapshot storage |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, distributed league, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | vectorized fast path for array envs and hundreds of seats, cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

### Open items parked during SP1 and SP2
@OPEN_ITEMS@

### Next step
SP3 (players, league, warm start). Like SP1 and SP2, it starts with a brainstorm and a written spec before any plan or code (see Development Workflow). The owner still has to run `docs/GPU_CHECKS.md` on a CUDA machine.
````

Open-items block (replaces `@OPEN_ITEMS@`):

````markdown
- **SP4:**
  - paired, rating-based checkpoint selection (Bradley–Terry with bootstrap) instead of per-pair Wilson intervals.
- **SP5:**
  - distributed learner resume (`run-learner` never applies `training.resume_from`);
  - metrics hub / `metrics.jsonl` / WandB for the distributed roles (today: per-process logs only);
  - resource leak when `run-learner` setup fails after the trajectory server, weight-store client or drainer started;
  - one budget semantics for distributed mode (today `total_timesteps / num_workers` per worker; learner progress = `consumed_samples / total_timesteps`, which counts decisions, not env steps);
  - newest-wins weight-queue eviction unpickles the stale payload (CPU ∝ model size × workers) → per-machine weight cache;
  - the distributed learner's final checkpoint can be lost when the checkpoint drainer stops first and the 16-slot queue is full;
  - distributed league and games where one agent does not play every role.
- **SP6:**
  - normalizer statistics are updated before the loss forward, so the ratio is not exactly 1 at zero lag on early steps;
  - `WindowAttentionCore` rebuilds its masks every step (cache with the fast path);
  - BC windows are not episode-aligned;
  - observation trees are handled per seat in Python on the worker (vectorized per-role arrays for envs with hundreds of seats).
- **Tier 2 algorithms:** replay buffer for R2D2/DQN (the replay path would also need its own normalizer-stat and warm-up handling).
- **Minor, unscheduled:**
  - `--set` lists follow YAML 1.1, so `--set x=[1e-4,2]` keeps `1e-4` as a string (write `1.0e-4` in lists);
  - the main process builds a ratings snapshot on every monitor pass (negligible cost);
  - stale parked rollout buffers have no age limit (bounded by agents × envs × seats);
  - SharedMemory zero-copy weight store (raw `shared_memory` instead of queue payloads);
  - dynamic add/remove of agents and machines during a run;
  - config inheritance / profiles (`extends: base.yaml`).
- **Parked during SP2** (from the SP2 acceptance report, «Открытые пункты»):
@SP2_OPEN_ITEMS@
````

2. Apply the edits with this script (it asserts every anchor, so a changed CLAUDE.md fails loudly instead of being half-edited):

```python
"""T8.6: bring CLAUDE.md to the SP2 state (run once from the repo root; asserts every anchor)."""
from pathlib import Path

path = Path("CLAUDE.md")
text = path.read_text()
status = Path("/tmp/claude_status.md").read_text()      # Step 4 writes it

REPLACEMENTS = [
    ("- User writes: `MyGameEnv(BaseEnv)` — Gymnasium-compatible, supports 1-N players",
     "- User writes: `MyGame(MultiAgentEnv)` with a `GameSpec` (roles, layouts, teams) — see `docs/ENV_GUIDE.md`"),
    ("- If arena match has fewer agents than player slots → duplicate agents to fill",
     "- Every team gets a core agent (the owner, its checkpoints, or another trainable agent by PFSP); "
     "`matchmaking.teammates: self | mixed` fills the team's other seats; a role the core does not play goes to "
     "an agent that plays it"),
    ("- Same semantics as training rollouts: one model `State` per (env, seat), `info[\"active\"]` and action masks",
     "- Same engine as training rollouts (`MatchRunner`): one model `State` per (env, seat), `StepResult.acting` and "
     "action masks"),
    ("- Seat rotation via `schedule_lineups`: for a pair (a, b), match m gives seat s to `(a, b)[(s + m) % 2]`; "
     "`--num-matches` is per pair (rounded up to even), so every agent plays every seat equally often; in an N-player "
     "game a pairwise match fills all N seats with the two agents alternately",
     "- Lineups via `schedule_lineups` (per layout): for a pair (a, b), match m gives team i to the core "
     "`(a, b)[(i + m) % 2]` where the roles allow, teams filled homogeneously; one agent fills every team; one-team "
     "layouts play each agent's homogeneous team plus mixed compositions (cross-play); `--num-matches` is per pair "
     "(or composition) and layout"),
    ("- Pairwise report: W/D/L, win rate and score (draw = half) with draw-aware 95% Wilson intervals, per-seat "
     "breakdown, mean returns and episode length",
     "- Reports per layout and role: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side; `rank` — "
     "mean rank with CI, first-place share, pairwise matrix; `score` — mean score with CI, cross-play table"),
    ("- Solo mode (one agent, or a 1-player env): mean return and outcome with 95% normal intervals\n", ""),
    ("- Player slots mapped to agents via `slot_agent_map[env_idx][player_idx] -> agent_id`",
     "- Seats mapped to agents per env by a `Lineup` (layout + one `SeatAssignment(agent_id, network_id, collect)` "
     "per seat); an agent's roles come from `agents.<id>.roles`"),
    ("- Episode boundaries handled within chunks (V-trace traces are cut at done, new episode starts in same chunk)",
     "- Chunk slots are `act` (a decision), `boot` (an observation for the learner's bootstrap value) or `pad`; "
     "episode boundaries are handled within chunks (traces stop at `terminal` and `boot`), and the learner computes "
     "every value with its current network"),
    ("  - Action masking for constrained action spaces\n",
     "  - Action masking for constrained action spaces\n"
     "  - Many units per seat (`Units`): `ratio_mode` (`joint` / `per_unit`), `unit_trace` (scalar ρ for V-trace), "
     "`entropy_reduction`; per-decider diagnostics\n"
     "  - Centralized critic: `global_state` reaches only the value path (`networks.critic_encoder_class`)\n"),
    ("class MyEnv(BaseEnv): ...           # Gymnasium-like interface, 1-N players",
     "class MyGame(MultiAgentEnv): ...    # GameSpec + reset/step -> StepResult (docs/ENV_GUIDE.md)"),
]
for old, new in REPLACEMENTS:
    assert text.count(old) == 1, f"anchor not found exactly once: {old[:70]!r}"
    text = text.replace(old, new)

start, end = text.index("## Implementation Status"), text.index("## Development Workflow")
text = text[:start] + status + text[end:]
path.write_text(text)
print("CLAUDE.md updated")
```

Save it as `/tmp/update_claude_md.py` and run: `.venv/bin/python /tmp/update_claude_md.py`

Expected: `CLAUDE.md updated`. Then `grep -n "BaseEnv\|slot_agent_map\|info\[\"active\"\]\|@[A-Z_]*@" CLAUDE.md` prints nothing.

If an anchor assertion fires, T7.x already changed that line: read the current line and apply the intent of the replacement by hand.

- [ ] **Step 6: Update `docs/GPU_CHECKS.md`**

```bash
.venv/bin/python -m pytest -m gpu --collect-only -q | grep "::" | sort > /tmp/gpu-tests.txt
grep -oE 'tests/[^ |`]+::[^ |`]+' docs/GPU_CHECKS.md | sort > /tmp/gpu-listed.txt
comm -23 /tmp/gpu-tests.txt /tmp/gpu-listed.txt     # collected but not listed: add rows
comm -13 /tmp/gpu-tests.txt /tmp/gpu-listed.txt     # listed but gone: remove rows
```

1. Remove the table rows of node ids that no longer exist (SP1 tests deleted or renamed by T7.3).
2. Add one row per new node id: the node id; "Что проверяет" in one Russian line taken from the test's docstring; the command `.venv/bin/python -m pytest -v <file> -k <test name without parameters>`.
3. In «Что проверяется», add these bullets (keep the SP1 ones that still have tests):
   - деревья наблюдений, действий и масок на CUDA: батч лёрнера из чанков v2 (`uint8`-листья доходят до модели без приведения), `act`/`boot`/`pad`-слоты;
   - распределения `Units` (`UnitsDist`, `TreeDist`) на CUDA и под AMP: `log_prob` = сумма валидных `unit_log_prob`, без NaN при пустых строках масок;
   - режимы `ratio_mode` / `unit_trace` / `entropy_reduction` на CUDA;
   - централизованный критик (`critic_encoder` и `global_state`) на CUDA.
   Drop a bullet only if no `gpu` test checks it.
4. Rerun the two `grep`/`comm` commands: both `comm` outputs must be empty.

- [ ] **Step 7: Check README and guide links**

```bash
.venv/bin/python - <<'EOF'
import pathlib, re
for doc in ("README.md", "docs/ENV_GUIDE.md", "docs/benchmarks.md", "docs/GPU_CHECKS.md"):
    base = pathlib.Path(doc).parent
    for link in re.findall(r"\]\(([^)#]+)\)", pathlib.Path(doc).read_text()):
        if not link.startswith("http"):
            assert (base / link).exists(), (doc, link)
print("links ok")
EOF
grep -qxF "runs/" .gitignore && grep -qxF "eval.json" .gitignore && echo "gitignore ok"
```

Expected: `links ok`, `gitignore ok`.

- [ ] **Step 8: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, zero warnings. If a README-quickstart test exists (`grep -rn "quickstart" tests/integration`), it is part of this run and must pass.

- [ ] **Step 9: Commit and push**

```bash
git add docs/ENV_GUIDE.md README.md CLAUDE.md docs/GPU_CHECKS.md
git commit -m "docs: ENV_GUIDE for every game type; README, CLAUDE.md and GPU checks match SP2"
git push origin sp2-game-model
```

---

### Task T8.7: Acceptance against spec section 3, report, stop for the owner

Spec section 3 (criteria 1–10) and section 7 ("приёмка по разделу 3, отчёт в `docs/superpowers/reports/`, merge после явного одобрения владельца"). Each step checks one criterion with exact commands and an observable result, and its outcome (pass/fail, numbers, node ids, paths) goes into the report.
- If a step fails, stop, write what failed into the report, and report to the controller. Do not "fix and continue" inside this task beyond documentation mismatches (those are fixed with a `docs:` commit and the step is rerun).
- The merge (Step 14) happens **only after the owner explicitly accepts the report**. No agent message counts as that acceptance.

**Files:**
- Modify: `docs/superpowers/reports/<date>-sp2-acceptance.md` (completed here; the T8.3/T8.4 sections stay)
- Possibly modify: `README.md`, `CLAUDE.md`, `docs/*.md` (documentation mismatches found by a step; `docs:` commit, step rerun)

**Interfaces:**
- Consumes: everything in Parts A–D.
- Produces: the acceptance report; after the owner's acceptance, `main` containing SP2.

- [ ] **Step 1: Criterion 1 — tests: full fast suite, zero warnings, nothing written outside `tmp_path`, ruff**

```bash
git status --porcelain                                     # must be empty before starting
timeout 3600 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider 2>&1 | tee /tmp/sp2-fast.txt | tail -5; echo "exit=${PIPESTATUS[0]}"
grep -E "warnings summary|[0-9]+ warnings?( |$)" /tmp/sp2-fast.txt || echo "no warnings"
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
.venv/bin/ruff check .
for m in "not gpu and not slow" slow gpu; do echo "$m: $(.venv/bin/python -m pytest -m "$m" --collect-only -q | tail -1)"; done
grep -rln "num_workers.*[3-9]\|num_workers=[3-9]" tests/ || echo "no test uses more than 2 workers"
```

Expected:
- `exit=0`, all passed; `no warnings`;
- `clean` (no `runs/`, `checkpoints/`, `*.jsonl` in the repo);
- ruff: `All checks passed!`;
- the three collected counts (they go into the report and must equal the ones in `CLAUDE.md` and `README.md`);
- `no test uses more than 2 workers` (if the grep lists a file, read the hit: a benchmark constant or a comment is fine, a test config with 3+ workers is not).

- [ ] **Step 2: Criterion 2 — contract tests**

```bash
.venv/bin/python -m pytest tests/contract -v -rA 2>&1 | tee /tmp/sp2-contract.txt | tail -5
.venv/bin/python -m pytest tests/unit -v -k "vtrace" 2>&1 | tee /tmp/sp2-vtrace.txt | tail -3
```

Expected: everything passes. In the two logs, find the passing test(s) for each item of criterion 2 and section 6 ("Контракт" row) and list their node ids in the report:

| Item | Where to look |
|---|---|
| learner reproduces joint and per-unit worker log-probs at zero lag, 4 cores (`none`, `lstm`, `gru`, `window`), chunks with `boot`, `pad`, truncation and elimination | T4.3 tests in `tests/contract` |
| V-trace with `boot` slots equals the reference (zero lag = GAE(λ) on `act` slots; `boot` cuts the trace; `terminal` gives 0; NaN on `pad` never reaches targets) | T4.1 tests (`/tmp/sp2-vtrace.txt`) |
| no reward is lost, except for seats that never acted in an episode, which are counted (`dropped_reward_episodes`) | T3.4 contract tests |
| bootstrap from `final_obs` is computed by the learner's network | T4.3 |
| `uint8` reaches the model; `global_state` reaches only the value path | T3.4 / T4.3 |
| every env contract error of spec block 1, including `max_idle_steps` | T1.5 / T3.4 |
| seat lifecycle (rewards to waiting seats, reward in the elimination step, dead teammate rewarded at the end, empty seats, never-acting seat) | T3.4 |
| chunk boundary in mid-episode with a parked buffer continues in the seat's reset buffer (LSTM) | T3.4 |
| matchmaking invariants (roles, homogeneous `self` teams, layout shares ±5%, permutations, every agent owns envs, asymmetry under `mode: self_play`) | T5.1 contract tests |
| `MatchResult` by teams; eval report for every outcome kind | T3.3 / T6.1 |
| the demo games run through the real `MatchRunner` without contract errors | `tests/contract/test_demo_*.py`, `test_reference_examples.py` |

An item without a passing test fails this criterion.

- [ ] **Step 3: Criterion 3 — learning**

```bash
.venv/bin/python -m pytest tests/learning -m "not slow" -v --durations=0 2>&1 | tail -15
nproc; uptime
time .venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -v -s 2>&1 | tee /tmp/sp2-slow-accept.txt | grep -E "^\[|PASSED|FAILED"
```

Expected: the fast tests pass, each under 20 s; `nproc` = 8; the seven slow tests PASS with the printed values at or above the thresholds of the T8.3 table; each training at about 3 minutes or less (or the time accepted by a T8.3 ruling). Compare with the T8.3 numbers (seed 0): a value much lower than in T8.3 means something changed since; investigate before going on.

Also confirm the cross-play report for `coop_buttons`: the `[coop_buttons]` line prints a `cross_play` JSON with a `coop_a+coop_b` entry, and the test printed an eval report (`summarize(...).text()`) above it.

- [ ] **Step 4: Criterion 4 — units experiment**

```bash
sed -n '/## Эксперимент по юнитам/,/## Размер/p' docs/benchmarks.md | head -60
.venv/bin/python -c "import json; d = json.load(open('docs/benchmarks/units-experiment.json')); print(len(d['rows']), d['recommendation'])"
grep -n "Ruling: unit_trace" docs/superpowers/reports/*-sp2-acceptance.md
```

Expected: the section has the 12-row table (K ∈ {8, 128} × `ratio_mode` × `unit_trace`) with win rate, `clip_fraction`, `clip_fraction_joint`, `ess`; the JSON has 12 rows; the `unit_trace` ruling is in the report; the code default matches it (`tests/unit/test_unit_trace_default.py` or the T4.2 test passed in Step 1).

- [ ] **Step 5: Criterion 5 — speed**

```bash
git log --oneline -1 -- docs/benchmarks/after-sp2.json
git diff --stat $(git log --format=%h -1 -- docs/benchmarks/after-sp2.json)..HEAD -- src/colosseum/worker src/colosseum/learner src/colosseum/algorithms src/colosseum/envs src/colosseum/networks src/colosseum/core
sed -n '/## После SP2/,/## Эксперимент по юнитам/p' docs/benchmarks.md
```

Expected: the «После SP2» table shows env steps/s at least 90% of SP1's (4641 / 9444 / 10664) and strictly increasing over 1 → 2 → 4 workers. If the `git diff --stat` shows changes to those source directories since the measurement, rerun the T8.4 Step 6 command, update the table and the JSON (`docs:` commit), and use the new numbers.

- [ ] **Step 6: Criterion 6 — reference examples**

```bash
.venv/bin/colosseum validate -c configs/examples/space_miners.yaml; echo "exit=$?"
.venv/bin/colosseum validate -c configs/examples/chase.yaml; echo "exit=$?"
.venv/bin/python -m pytest tests/integration/test_reference_examples_smoke.py tests/contract/test_reference_examples.py -v -rs
```

Expected: both `exit=0`; all tests pass, none skipped (Box2D is installed by `scripts/setup-dev.sh`).

- [ ] **Step 7: Criterion 7 — documentation works verbatim**

In a fresh shell (`env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc`), from the repo root, run the README Quick start block exactly as written, then the three "Other game structures" commands with a smaller budget:

```bash
source .venv/bin/activate
colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
NEW=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
OLD=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | head -1)
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-continued \
  --set training.resume_from=runs/ttt-quickstart --set training.total_timesteps=1200000
for g in tron predator_prey coop_buttons; do
  colosseum train -c configs/examples/$g.yaml --set run.name=$g-doc --set training.total_timesteps=60000 || echo "FAILED $g"
done
colosseum eval -c configs/examples/tron.yaml -a a=$(ls -d runs/tron-doc/checkpoints/agent_0/ckpt_v* | sort -V | tail -1) \
  -a b=$(ls -d runs/tron-doc/checkpoints/agent_0/ckpt_v* | sort -V | head -1) --layout 4p --num-matches 40
```

Expected:
- every command exits 0, no `FAILED` line;
- the quickstart training time and the printed win rates are consistent with the README's `@TTT_TIME@`/`@TTT_WINS@` values (filled in T8.6);
- `eval.json` exists; the tic-tac-toe report has W/D/L, win rate with Wilson CI and a per-side breakdown; the tron `4p` report has mean rank with CI, first-place share and the pairwise matrix;
- the continued run's first checkpoint version is above the quickstart's last.

Then:
- compare every flag in README sections with placeholders (`my_game/...`, `path/to/demos/`, `cfg.yaml`) with `colosseum <command> --help`;
- check that `docs/ENV_GUIDE.md` covers every game type of the spec (solo, 1v1 turn-based and simultaneous, units, team vs team, FFA with elimination and variable player counts, asymmetric roles, cooperative) with a demo-game reference each: its first table lists them;
- check that README "Status and limitations" and CLAUDE.md "Implementation Status" / "Roadmap" describe what Steps 1–6 observed (test counts, the `unit_trace` default, the distributed limits).

Clean up: `rm -rf runs eval.json`.

- [ ] **Step 8: Criterion 8 — distributed mode on the new format**

```bash
.venv/bin/python -m pytest tests/integration -v -k "distributed or grpc" 2>&1 | tail -15
rm -rf /tmp/sp2-acceptance
.venv/bin/colosseum serve-weight-store --port 50051 & WS=$!
.venv/bin/colosseum run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0 --traj-port 50052 \
  --weight-store localhost:50051 --set run.dir=/tmp/sp2-acceptance --set training.total_timesteps=20000 & LR=$!
sleep 10
.venv/bin/colosseum run-workers -c configs/examples/tic_tac_toe.yaml --weight-store localhost:50051 -l agent_0=localhost:50052 \
  --set run.dir=/tmp/sp2-acceptance --set training.total_timesteps=20000; echo "workers exit=$?"
kill $LR $WS; wait
ls /tmp/sp2-acceptance
.venv/bin/colosseum run-workers -c configs/examples/predator_prey.yaml --weight-store localhost:50051 -l hunter=localhost:50052 \
  --set run.dir=/tmp/sp2-acceptance; echo "asymmetric exit=$?"
```

The weight store is already stopped for the last command on purpose: the role check happens before any connection.

Expected: the distributed tests pass; `workers exit=0`; a learner run dir with checkpoints and a workers run dir with logs; the last command (a game where one agent does not play every role) fails fast with `exit=1` and a `Config error:` line naming SP5 (spec block 10: such games are out of the distributed scope). Clean up: `rm -rf /tmp/sp2-acceptance`.

- [ ] **Step 9: Criterion 9 — GPU**

```bash
.venv/bin/python -m pytest -m gpu --collect-only -q | grep "::" | sort > /tmp/gpu-tests.txt
grep -oE 'tests/[^ |`]+::[^ |`]+' docs/GPU_CHECKS.md | sort | diff - /tmp/gpu-tests.txt && echo "GPU list matches"
.venv/bin/python -m pytest -m gpu -q 2>&1 | tail -3
```

Expected: `GPU list matches`; without CUDA every GPU test is `skipped`, none failed. The report says that the owner still has to run them on a CUDA machine (`docs/GPU_CHECKS.md`).

- [ ] **Step 10: Criterion 10 — SP1 residuals closed first**

```bash
git log --oneline --reverse main..sp2-game-model | head -8
grep -rn "cancel_join_thread\|PermissionError\|_QueueReader" tests/unit tests/integration | head
grep -n "class _QueueReader\|def " src/colosseum/core/ipc.py | head -20
```

Expected: the first commits of the branch (after the spec and plan commits) are T0.1's residual fixes; tests exist for the second Ctrl+C during `_release_command_queues`, for `colosseum bc` with an unreadable data file (one `Config error:` line), and `_QueueReader` lives in `core/ipc.py`. List the commits and node ids in the report.

- [ ] **Step 11: Hygiene**

```bash
git status --porcelain
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
ps -eo pid,args | grep -E "colosseum|multiprocessing" | grep -v grep || echo "no leftover processes"
```

Expected: an empty status (except the report being written), `clean`, `no leftover processes`.

- [ ] **Step 12: Write the report**

Complete `docs/superpowers/reports/<date>-sp2-acceptance.md` (Russian) with the structure of the SP1 report (`docs/superpowers/reports/2026-10-08-sp1-acceptance.md`):

```markdown
# SP2: приёмка по спеке §3 (T8.7)

- Ветка `sp2-game-model`, HEAD `<git rev-parse --short HEAD>`.
- Дата: <date>.
- Машина: <nproc> ядер, <ОЗУ> ГБ, без GPU. Python <версия>, torch <версия>.
- Шаг 14 (merge) не выполнялся: ждёт явного одобрения владельца.

## Итог

<ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ / список невыполненных>

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | <PASS/FAIL: N passed, M deselected, время, предупреждений 0; ruff; дерево чистое> |
| 3.2 | Контрактные тесты | <...> |
| 3.3 | «Учится» | <быстрые: длительности; медленные: значения против порогов> |
| 3.4 | Эксперимент по юнитам | <таблица в docs/benchmarks.md; дефолт unit_trace = ...> |
| 3.5 | Скорость | <шаги сред/с 1/2/4 воркера, доля от SP1, монотонность> |
| 3.6 | Эталонные примеры | <validate, smoke train> |
| 3.7 | Документация | <ENV_GUIDE, README дословно, CLAUDE.md> |
| 3.8 | Распределённый режим | <тесты, localhost smoke, отказ для асимметрии> |
| 3.9 | GPU | <список совпадает, N skipped; ждёт прогона на CUDA> |
| 3.10 | Остатки SP1 | <коммиты, тесты> |

## §3.1 ... §3.10
<по разделу на критерий: команда, вывод, node ids, числа — как в отчёте SP1>

## Замеры «учится» (T8.3)
<раздел из черновика, без изменений, плюс прогон шага 3>

## Эксперимент по юнитам и скорость (T8.4)
<раздел из черновика>

## Отклонения и замечания
<каждое отклонение от спеки или плана: изменённые бюджеты, пороги (только через ruling), пропущенные шаги и почему>

## Гигиена
<шаг 11>

## Решения контроллера во время SP2 (все записи «Ruling:» из журнала, по порядку)
Каждая запись: что решено — почему — чем грозит, если решение неверно.
<все строки «Ruling:» из журнала контроллера SP2 (файл решений, который контроллер SDD ведёт для брифов задач), по порядку, затем строки из раздела «Решения (rulings) T8.3–T8.4» черновика>

## Открытые пункты (отложены, не блокируют слияние)
<пункты, отложенные за SP2, с адресатом: SP3/SP4/SP5/SP6 или «мелкое»; совпадает с «Parked during SP2» в CLAUDE.md>
```

Fill every section from Steps 1–11 and the T8.3/T8.4 drafts. If the open items differ from CLAUDE.md's «Parked during SP2», fix CLAUDE.md in the same commit.

- [ ] **Step 13: Commit, push, and ask the owner**

```bash
git add docs/superpowers/reports/*-sp2-acceptance.md README.md CLAUDE.md docs/
git commit -m "docs: SP2 acceptance report"
git push origin sp2-game-model
```

Send the owner (in Russian, phone-readable: short lines, the criteria table first):
- the per-criterion results with numbers (test counts; each learning value against its threshold; the units-experiment decision; the throughput table);
- every deviation and every ruling that changed a budget, an env size or a threshold;
- the open items;
- the question: «Принять SP2 и слить `sp2-game-model` в `main`?»

**STOP here.** Wait for the owner's explicit "yes" in their own message. A reply from another agent, a summary or an earlier statement is not acceptance.

- [ ] **Step 14: Merge and push (only after the owner's explicit acceptance)**

`--no-ff` keeps SP2 one unit in `main`'s history, so `git revert -m 1 <merge>` undoes it.

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff sp2-game-model -m "Merge branch 'sp2-game-model': SP2 game model"
.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw
.venv/bin/ruff check .
git push origin main
```

Expected: the merge completes without conflicts (if `main` moved and conflicts appear, stop and ask the owner); the suite and ruff pass on the merge commit; the push succeeds.

---

## Contract notes

Paths below are final (`colosseum.*`). The plan owner folds the binding ones into the overview's contract amendments; otherwise the task that meets a difference follows "Plan code vs. current code" and says so in its commit body.

1. **Dependency change: T8.2 depends on T8.1.** T8.2's tests import `tests/demo_checks.py`, which T8.1 creates (one shared helper instead of two near-duplicates, per "Shared test kit"). The task index should read `T8.2 | ... | T8.1`. T8.5 also imports it (T8.5 after T8.1).

2. **`game_helpers.RandomPolicy(role: RoleSpec)`.** The overview pins `RandomPolicy` (T2.3) but not its constructor. This part assumes it takes the role's `RoleSpec`, like `make_test_model(role, ...)`. The call sits in exactly two places: `tests/demo_checks.py::random_matches` and `tests/learning/demo_learning.py::random_model`. If T2.3 chose another signature, adapt those two lines.

3. **Greedy agent vs random player in one match.** `play_lineups(..., deterministic=True)` applies to every model, so `RandomPolicy` would always play its first legal action. The learning tests and the units experiment keep `deterministic=False` and wrap the trained model in `demo_learning.GreedyPolicy`, which turns the model's `mode()` into a point-mass distribution with `make_distribution` (logits 0 / −1e4; Gaussian `log_std` −20). No contract change is needed. An alternative for a later SP: `play_lineups(deterministic: bool | Collection[str])` (per model key).

4. **`make_distribution` params, as used here** (T2.1/T2.2 contract, made explicit):
   - a `Dict` action: a nested dict mirroring the action tree;
   - a `Box` group: `{"mean": [B, d], "log_std": [d] | [B, d]}`;
   - a `Units` group: the dict returned by `UnitsHead` (`{component: logits [B, U, n]}` for discrete components, `{component: {"mean": [B, U, d], "log_std": ...}}` for box components), used as is;
   - a non-`Dict` action: the single group's params, not wrapped.

5. **`ComposedModel.step` applies `action_mask` to the policy's distribution** (SP1 behaviour); demo policies never apply masks themselves. `GreedyPolicy`/`ScriptedPolicy` apply the mask again to their point-mass distribution (harmless; needed by `ScriptedPolicy`, whose script may pick an illegal "build").

6. **`Outcome` with both `team_rank` and `team_score`** (`space_miners`: scores plus the engine's tie-break ranks). The contract allows both fields; this part assumes `resolve_outcome` then keeps both as given (ranks are not re-derived from scores). If T1.4 re-derives ranks when scores are present, `space_miners` should send only `team_rank` plus scores through `infos`; its test checks only that both team keys exist.

7. **`ratings.json` layout.** The coop tests read `ratings.json` as `{"env_steps": N, <layout>: {"elo", "win_rates", "games", "wr_vs_past", "past_games", "scores", "cross_play"}}` with `cross_play` keys `"A+B"` (sorted, `+`-joined) and values `{"n", "mean"}` (`RatingBook.snapshot`, `CrossPlayTable.summary`, SP1's `{"env_steps", **ratings}` file format).

8. **Train metric names.** The per-decider clip fraction keeps SP1's name `clip_fraction` (the spec's `clip_frac`); the others are the overview's: `clip_fraction_joint`, `c_clip_frac`, `log_rho_abs_mean`, `log_rho_abs_p95`, `log_rho_joint_abs_mean`, `ess`, `deciders_valid_mean`, `deciders_valid_max`, `boot_frac`, `pad_frac`. The T8.1 smoke test and the units experiment read them from `metrics.jsonl` `train` records.

9. **`MatchRunner.close()` closes its vector env**, and an in-process `VectorEnv` accepts a `functools.partial` env factory. `MatchObserver` is a protocol: `demo_checks.MatchLog` implements all five methods.

10. **`load_eval_model(config, name, path)`** is called with the run's resolved config (`config.resolved.yaml`) and the agent id as `name`, on a checkpoint dir; architecture and roles come from `meta.json`, roles[0] selects the `RoleSpec`.

11. **`cli_runner` after T7.x:** `TINY` holds new-schema keys (`checkpoint.interval`, not `self_play.*`), `run_train(...)` merges `TINY` with the overrides, `TrainRun.records(kind)` and `TrainRun.ratings()` exist (SP1). The slow tests do not use `TINY`: they build their command with `demo_learning.train_cmd`.

12. **`--set` with a mapping value** (`--set 'env.kwargs={"max_units": 128, ...}'`) replaces the whole `env.kwargs` dict (SP1's `parse_override_value` parses YAML/JSON flow mappings). `scripts/units_experiment.py` and the T8.5 smoke test rely on it; `load_config(path, {"env.kwargs": {...}, "networks.critic_encoder_class": None})` does the same in-process.

13. **New names produced by this part** (beyond the overview's file map): `tests/demo_checks.py`; `tests/learning/sp2_bandits.py`, `tests/learning/demo_learning.py` (support modules; added to `known-first-party` with `demo_checks`); `scripts/units_experiment.py`, `scripts/measure_global_state.py`; `docs/benchmarks/{after-sp2,units-experiment,global-state-team-tag}.json`; the demo games' reference policies (`greedy_action`, `scripted_action`, `safe_action`, `chase_action`, `flee_action`, `measure_baselines`). `tests/unit/test_unit_trace_default.py` is created only if no T4.2 test pins the `auto` resolution or the default changes.

14. **Removed by this part if T7.x left them:** `tests/learning/test_ttt_slow.py`, `tests/learning/ttt_eval.py` (ported to `test_demo_learning_slow.py::test_tic_tac_toe_beats_random_80_percent`), `examples/space_miners/{env,networks}.py`, `examples/composite_action/{env,networks}.py`.

15. **Scripts import the test kit.** `scripts/units_experiment.py` puts `tests/` and `tests/learning/` on `sys.path` to reuse `demo_learning` and `game_helpers.RandomPolicy` instead of duplicating them in `src/`. It is a development script, not part of the package.

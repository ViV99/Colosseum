# SP3 Plan — Part 02: Opponent Selection (League) and Warm Start

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.**
- **Spec block 5 (opponent selection), tasks T3.1–T3.3.** Matchmaking config v3 (`opponents` shares, schedules, `anchors`, `pfsp`, per-agent override), the translation of the SP2 knobs, the package `colosseum.league` (`BaseMatchmaker`, `MatchmakerContext`, `MixtureMatchmaker`, `check_lineup`, `validate_matchmaking`, `effective_mix`, `describe_mix`), the deletion of `coordinator/matchmaker.py`, the coordinator and launcher integration (schedules on the coordinator's env steps, custom matchmaker classes), played-share metrics and the mix printout of `validate`.
- **Spec block 6 (warm start), tasks T4.1–T4.5.** The `init` and `kickstart` sections (global default + per-agent override, `training.kickstart_*` translated), `learner/factory.py` (used by the launcher AND the distributed learner), `init` sources (strict / partial, resume precedence, report), critic warm-up, a neural kickstart teacher per agent of any architecture (state-layout rule changed), a scripted kickstart teacher (DAgger labels in the chunk, worker-side teacher instances, the label loss in APPO).

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; everything this part adds or pins down beyond it is listed in `## Contract notes` at the end (proposed amendments A1–A16).

**Execution order inside this part:** T3.1 → T3.2 → T3.3 → T4.1 → T4.2 → T4.3 → T4.4 → T4.5 (the overview's table order). T4.4 needs only T4.1, but runs after T4.3 so T4.5 finds both.

**Cross-task notes.**
- **What T1.x / T2.x left behind.** This part relies only on the overview contract. Where a task edits a function that an earlier part also changed (`Coordinator.__init__`, `Launcher.launch`, `_start_children`, `RolloutLoop.__init__`, `validate_config`), the step names the statement to replace or the place to insert, and shows the new code. If the earlier part named a private attribute differently (e.g. the coordinator's `PfspStats` instance, assumed `self._pfsp`), use its name and say so in the commit body.
- **One sub-step per process boundary.** Teachers and `init` weights are resolved in the main process (`resolve_teacher`, `resolve_init`) into numpy dataclasses (`TeacherSpec`, `InitState`) and passed to learner processes as `mp.Process` kwargs; bots cross as `BotSpec`, never as instances (overview, "Inter-process data").
- **Old knobs.** `matchmaking.mode/self_play_ratio/latest_prob/pfsp_exponent` (T3.1) and `training.kickstart_*` (T4.1) are translated by `model_validator(mode="before")` hooks of `ColosseumConfig` on the raw input, logged with ONE `logging` warning each, and never stored. `--set` accepts their paths through `LEGACY_OVERRIDE_KEYS` (amendment A5), so the SP2 integration tests that pass `--set matchmaking.mode=...` keep working.
- **`coordinator/matchmaker.py`** is used by the coordinator until T3.2. T3.1 removes the SP2 fields from `MatchmakingConfig`, so it gives `LineupMatchmaker` a one-task shim (SP2 numbers read back from `opponents` at env step 0); T3.2 deletes the module and switches the coordinator to `MixtureMatchmaker` (amendment A12).
- **Conventions.** Run everything from the repo root with `.venv/bin/python` / `.venv/bin/ruff`. "Full fast suite" = `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings) followed by `.venv/bin/ruff check .`. New test files start with `test_sp3_`. Every commit message has a conventional prefix and no attribution lines; push after every task (`git push origin sp3-league`).
- **Test-kit additions to `tests/game_helpers.py` in this part:** `write_ckpt_dir` (T4.2), `GameTestModel.value_parameters` (T4.3), `LabelBot` (T4.5). Test-local bots, matchmakers and models live in the test modules themselves (importable by bare module name, like `test_sp2_validate.ValueAsMatrixModel`).

---

### Task T3.1: Matchmaking config v3, schedules, SP2 knob translation, per-agent override

Spec block 5 «Конфиг», «Расписания», «Override на агента», «Ручки SP2». The config gets `opponents` (shares, each a number or a piecewise-linear schedule over the run's global env steps), `anchors`, `pfsp` and `matchmaker_class`; the SP2 knobs disappear from the model and are translated from the raw input; `agents.<id>.matchmaking` overrides the global section.

**Files:**
- Create: `src/colosseum/league/schedule.py`
- Modify: `src/colosseum/league/__init__.py` (created empty or with eager exports by T2.3; becomes lazy, see Step 3)
- Modify: `src/colosseum/core/config.py`
- Modify: `src/colosseum/coordinator/matchmaker.py` (one-task shim; the module is deleted in T3.2)
- Modify: `scripts/bench_throughput.py` (`_make_config`)
- Modify (tests whose SP2 assumptions change): `tests/unit/test_config_v2.py`, `tests/unit/test_sp2_matchmaker.py` (helper `make_mm` only)
- Test: `tests/unit/test_sp3_matchmaking_config.py`

**Interfaces:**
- Consumes: `StrictModel` (with `populate_by_name=True`, T1.1), `TrainableAgent.matchmaking: dict | None` (T1.1, free dict), `ColosseumConfig.get_agent_config` / `_AGENT_SECTIONS` / `deep_merge` (T1.1 form), `ConfigError`.
- Produces (contract): `ScheduleValue`, `schedule_value`, `schedule_points` (`colosseum.league.schedule`); `OpponentShares`, `PfspConfig`, `MatchmakingConfig` (v3), `AGENT_MATCHMAKING_KEYS`, `SP2_MATCHMAKING_KNOBS`, `translate_sp2_matchmaking` (`colosseum.core.config`).
- Produces (additions, Contract notes A5/A6): `parse_schedule(value, what="schedule") -> ScheduleValue`; `ScheduleField` (annotated pydantic type); `merge_matchmaking(base: dict, override: dict) -> dict`; the four matchmaking knob paths in T2.1's `LEGACY_OVERRIDE_KEYS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_matchmaking_config.py`:

```python
"""Matchmaking config v3 (spec block 5): shares and schedules, anchors, pfsp, per-agent override,
translation of the SP2 knobs (T3.1)."""
from __future__ import annotations

import logging

import pytest
import yaml
from pydantic import ValidationError

from colosseum.core.config import (
    AGENT_MATCHMAKING_KEYS,
    SP2_MATCHMAKING_KNOBS,
    ColosseumConfig,
    MatchmakingConfig,
    load_config,
    merge_matchmaking,
    translate_sp2_matchmaking,
)
from colosseum.core.errors import ConfigError
from colosseum.league.schedule import parse_schedule, schedule_points, schedule_value

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
CONFIG_LOGGER = "colosseum.core.config"


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _knob_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.name == CONFIG_LOGGER and r.levelno == logging.WARNING and "SP2 matchmaking knobs" in r.getMessage()]


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_a_number_is_a_constant_schedule():
    assert parse_schedule(0.25) == 0.25 and parse_schedule(1) == 1.0
    assert schedule_value(0.25, 10**9) == 0.25
    assert schedule_points(0.25) == [0]


def test_a_schedule_is_piecewise_linear_and_constant_outside_its_points():
    s = parse_schedule({1000: 0.5, 3000: 0.1})
    assert schedule_points(s) == [1000, 3000]
    assert schedule_value(s, 0) == 0.5 and schedule_value(s, 1000) == 0.5
    assert schedule_value(s, 2000) == pytest.approx(0.3)
    assert schedule_value(s, 3000) == pytest.approx(0.1) and schedule_value(s, 10**7) == pytest.approx(0.1)


def test_schedule_keys_accept_strings_like_1e6_and_underscores():
    s = parse_schedule({"0": 1.0, "1e6": 0.0, "2_000_000": 0.5})
    assert schedule_points(s) == [0, 1_000_000, 2_000_000]
    assert schedule_value(s, 500_000) == pytest.approx(0.5)
    assert list(s) == [0, 1_000_000, 2_000_000]  # stored sorted


@pytest.mark.parametrize("raw, message", [
    ({}, "empty"),
    ({-1: 0.5}, "non-negative"),
    ({"1.5": 0.5}, "integer"),
    ({"abc": 0.5}, "not a number"),
    ({0: -0.1}, ">= 0"),
    (-0.5, ">= 0"),
    (float("nan"), "finite"),
    ({0: "x"}, "number"),
    ("0.5", "number"),
    (True, "number"),
    ({0: 0.1, "0": 0.2}, "twice"),
])
def test_schedule_errors_name_the_problem(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_schedule(raw, "matchmaking.opponents.latest")


# ---------------------------------------------------------------------------
# The v3 model
# ---------------------------------------------------------------------------


def test_v3_defaults_and_no_sp2_fields():
    m = ColosseumConfig.model_validate(BASE).matchmaking
    o = m.opponents
    assert (o.latest, o.snapshots, o.rivals, o.anchors) == (0.7, 0.2, 0.0, 0.1)
    assert m.anchors is None
    assert (m.pfsp.weighting, m.pfsp.exponent, m.pfsp.halflife_games) == ("hard", 2.0, 200.0)
    assert (m.layouts, m.teammates, m.teammate_self_prob, m.shuffle_seats, m.matchmaker_class) == \
        ({}, "self", 0.5, True, None)
    assert not set(SP2_MATCHMAKING_KNOBS) & set(MatchmakingConfig.model_fields)


def test_shares_take_schedules_and_anchors_take_lists_or_weights():
    m = MatchmakingConfig.model_validate({
        "opponents": {"latest": {0: 0.9, "1e6": 0.5}, "anchors": {0: 0.1, 1_000_000: 0.5}},
        "anchors": {"greedy": 2.0, "random": {0: 1.0, 500: 0.0}},
    })
    assert m.opponents.latest == {0: 0.9, 1_000_000: 0.5}
    assert m.anchors == {"greedy": 2.0, "random": {0: 1.0, 500: 0.0}}
    assert MatchmakingConfig(anchors=["greedy", "random"]).anchors == ["greedy", "random"]
    assert MatchmakingConfig(anchors=[]).anchors == []


@pytest.mark.parametrize("data", [
    {"opponents": {"latest": -0.1}},
    {"opponents": {"latest": {}}},
    {"opponents": {"bogus": 0.1}},
    {"anchors": ["a", "a"]},
    {"anchors": {"a": -1.0}},
    {"anchors": [""]},
    {"pfsp": {"weighting": "soft"}},
    {"pfsp": {"exponent": -1.0}},
    {"pfsp": {"halflife_games": 0}},
    {"layouts": {"2p": 0.0}},
    {"teammates": "random"},
    {"teammate_self_prob": 1.5},
    {"matchmaker_class": ""},
])
def test_v3_bounds(data):
    with pytest.raises(ValidationError):
        MatchmakingConfig.model_validate(data)


@pytest.mark.parametrize("knob", SP2_MATCHMAKING_KNOBS)
def test_the_model_itself_rejects_sp2_knobs_and_names_the_replacement(knob):
    with pytest.raises(ValidationError, match="opponents"):
        MatchmakingConfig.model_validate({knob: 0.5})


# ---------------------------------------------------------------------------
# SP2 knob translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("knobs, latest_snapshots_rivals, exponent", [
    ({"mode": "self_play", "latest_prob": 0.8}, (0.8, 0.2, 0.0), 1.0),
    ({"mode": "self_play"}, (0.5, 0.5, 0.0), 1.0),
    ({"latest_prob": 0.3}, (0.3, 0.7, 0.0), 1.0),            # mode defaults to self_play
    ({"mode": "league"}, (0.25, 0.25, 0.5), 1.0),            # self_play_ratio and latest_prob default to 0.5
    ({"mode": "league", "self_play_ratio": 0.0, "latest_prob": 0.8, "pfsp_exponent": 3.0}, (0.0, 0.0, 1.0), 3.0),
    ({"self_play_ratio": 0.2}, (0.5, 0.5, 0.0), 1.0),        # ignored under the default mode self_play
    ({"pfsp_exponent": 0.0}, (0.5, 0.5, 0.0), 0.0),
])
def test_translate_sp2_knobs(knobs, latest_snapshots_rivals, exponent):
    out = translate_sp2_matchmaking({**knobs, "layouts": {"2p": 1.0}, "shuffle_seats": False})
    assert not set(SP2_MATCHMAKING_KNOBS) & set(out)
    assert out["layouts"] == {"2p": 1.0} and out["shuffle_seats"] is False
    got = out["opponents"]
    assert (got["latest"], got["snapshots"], got["rivals"]) == pytest.approx(latest_snapshots_rivals)
    assert got["anchors"] == 0.0
    assert out["pfsp"] == {"weighting": "hard", "exponent": exponent}


def test_translate_without_knobs_returns_a_deep_copy():
    raw = {"opponents": {"latest": 1.0}, "layouts": {"2p": 1.0}}
    out = translate_sp2_matchmaking(raw)
    assert out == raw and out is not raw and out["layouts"] is not raw["layouts"]


@pytest.mark.parametrize("raw, message", [
    ({"mode": "league", "opponents": {"latest": 1.0}}, "opponents"),
    ({"latest_prob": 0.5, "pfsp": {"exponent": 1.0}}, "pfsp"),
    ({"mode": "arena"}, "mode"),
    ({"latest_prob": 1.5}, "latest_prob"),
    ({"self_play_ratio": "x"}, "self_play_ratio"),
    ({"pfsp_exponent": -1}, "pfsp_exponent"),
])
def test_translate_errors(raw, message):
    with pytest.raises(ConfigError, match=message):
        translate_sp2_matchmaking(raw)


def test_loading_sp2_knobs_translates_once_and_never_stores_them(tmp_path, caplog):
    path = _write(tmp_path, {**BASE, "matchmaking": {"mode": "league", "self_play_ratio": 0.4, "latest_prob": 0.75,
                                                     "teammates": "mixed"}})
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = load_config(path)
    warnings = _knob_warnings(caplog)
    assert len(warnings) == 1 and "opponents" in warnings[0].getMessage()
    o = cfg.matchmaking.opponents
    assert (o.latest, o.snapshots, o.rivals, o.anchors) == pytest.approx((0.3, 0.1, 0.6, 0.0))
    assert cfg.matchmaking.teammates == "mixed" and cfg.matchmaking.pfsp.exponent == 1.0
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert not set(SP2_MATCHMAKING_KNOBS) & set(dumped["matchmaking"])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        again = load_config(_write(tmp_path, dumped, "resolved.yaml"))
        cfg.get_agent_config("agent_0")                    # derived configs do not warn either
    assert not _knob_warnings(caplog)
    assert again.matchmaking == cfg.matchmaking


def test_sp2_knobs_together_with_new_keys_are_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="opponents"):
        load_config(_write(tmp_path, {**BASE, "matchmaking": {"mode": "league", "opponents": {"latest": 1.0}}}))


def test_set_accepts_the_sp2_knob_paths(tmp_path):
    cfg = load_config(_write(tmp_path, BASE), {"matchmaking.mode": "league", "matchmaking.self_play_ratio": 0.0})
    assert cfg.matchmaking.opponents.rivals == 1.0
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(_write(tmp_path, BASE), {"matchmaking.latest_probb": 0.5})


def test_resolved_schedules_round_trip_through_yaml(tmp_path):
    data = {**BASE, "matchmaking": {"opponents": {"latest": {0: 0.9, "2e6": 0.5}},
                                    "anchors": {"bot": {0: 1.0, 1000: 0.2}}}}
    cfg = load_config(_write(tmp_path, data))
    again = load_config(_write(tmp_path, cfg.model_dump(mode="json", by_alias=True), "resolved.yaml"))
    assert again.matchmaking.opponents.latest == {0: 0.9, 2_000_000: 0.5}
    assert again.matchmaking.anchors == {"bot": {0: 1.0, 1000: 0.2}}


# ---------------------------------------------------------------------------
# Per-agent override
# ---------------------------------------------------------------------------


def _two_agents(**agent_a) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        **BASE,
        "matchmaking": {"opponents": {"latest": {0: 0.6, 1000: 0.2}, "snapshots": 0.3}, "anchors": ["bot"],
                        "layouts": {"2p": 1.0, "4p": 1.0}},
        "agents": {"a": {"matchmaking": agent_a}, "b": {}},
    })


def test_agent_matchmaking_keys():
    assert AGENT_MATCHMAKING_KEYS == {"opponents", "anchors", "pfsp", "layouts", "teammates", "teammate_self_prob"}


def test_per_agent_override_merges_opponents_and_pfsp_per_key_and_replaces_the_rest():
    cfg = _two_agents(opponents={"latest": 1.0}, anchors={"other": 0.5}, layouts={"4p": 1.0},
                      pfsp={"weighting": "uniform"}, teammates="mixed")
    a, b = cfg.get_agent_config("a").matchmaking, cfg.get_agent_config("b").matchmaking
    assert a.opponents.latest == 1.0 and a.opponents.snapshots == 0.3 and a.opponents.anchors == 0.1
    assert a.anchors == {"other": 0.5} and a.layouts == {"4p": 1.0}
    assert (a.pfsp.weighting, a.pfsp.exponent) == ("uniform", 2.0) and a.teammates == "mixed"
    assert b.opponents.latest == {0: 0.6, 1000: 0.2} and b.anchors == ["bot"] and b.layouts == {"2p": 1.0, "4p": 1.0}


def test_a_per_agent_schedule_replaces_the_global_schedule_whole():
    cfg = _two_agents(opponents={"latest": {500: 0.9}})
    assert cfg.get_agent_config("a").matchmaking.opponents.latest == {500: 0.9}


def test_merge_matchmaking_does_not_mutate_its_inputs():
    base = {"opponents": {"latest": {0: 0.5}}, "anchors": ["x"]}
    override = {"opponents": {"rivals": 0.1}, "anchors": ["y"]}
    out = merge_matchmaking(base, override)
    assert out == {"opponents": {"latest": {0: 0.5}, "rivals": 0.1}, "anchors": ["y"]}
    assert base == {"opponents": {"latest": {0: 0.5}}, "anchors": ["x"]} and override["anchors"] == ["y"]
    out["opponents"]["latest"][0] = 9.0
    assert base["opponents"]["latest"][0] == 0.5


@pytest.mark.parametrize("override, message", [
    ({"shuffle_seats": False}, "global only"),
    ({"matchmaker_class": "x.Y"}, "global only"),
    ({"mode": "league"}, "global matchmaking section"),
    ({"oponents": {}}, "unknown keys"),
    ({"opponents": {"latest": -1.0}}, "matchmaking"),
])
def test_per_agent_override_errors(tmp_path, override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, {**BASE, "agents": {"a": {"matchmaking": override}}}))


def test_set_reaches_per_agent_matchmaking(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "agents": {"a": {}}}), {"agents.a.matchmaking.opponents.rivals": 0.5})
    assert cfg.get_agent_config("a").matchmaking.opponents.rivals == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_matchmaking_config.py -q`
Expected: collection error `ImportError: cannot import name 'AGENT_MATCHMAKING_KEYS' from 'colosseum.core.config'` (and `ModuleNotFoundError: colosseum.league.schedule`).

- [ ] **Step 3: Write the implementation**

3a. Create `src/colosseum/league/schedule.py`:

```python
"""Share and weight schedules (spec block 5).

A schedule value is a number or a piecewise-linear function of the run's global env steps given
by points ``{step: value}``: linear between consecutive points, constant before the first and
after the last. Keys are non-negative integers; strings such as ``"1e6"`` or ``"2_000_000"`` are
accepted (YAML keeps ``1e6`` as a string). Values are finite and >= 0 (shares and anchor weights).
"""

from __future__ import annotations

import bisect
import math
from typing import Any

ScheduleValue = float | dict[int, float]


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: expected a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what}: {value!r} is not finite")
    if number < 0:
        raise ValueError(f"{what}: {value!r} must be >= 0")
    return number


def _step(key: Any, what: str) -> int:
    if isinstance(key, bool):
        raise ValueError(f"{what}: schedule step {key!r} is not a number")
    if isinstance(key, str):
        try:
            number = float(key.strip().replace("_", ""))
        except ValueError:
            raise ValueError(f"{what}: schedule step {key!r} is not a number") from None
    elif isinstance(key, (int, float)):
        number = float(key)
    else:
        raise ValueError(f"{what}: schedule step {key!r} is not a number")
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{what}: schedule step {key!r} must be an integer number of env steps")
    if number < 0:
        raise ValueError(f"{what}: schedule step {key!r} must be non-negative")
    return int(number)


def parse_schedule(value: Any, what: str = "schedule") -> ScheduleValue:
    """Normalize a raw number or ``{step: value}`` mapping (points sorted by step); ValueError naming ``what``."""
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{what}: an empty schedule; give a number or {{step: value}} points")
        points: dict[int, float] = {}
        for key, item in value.items():
            step = _step(key, what)
            if step in points:
                raise ValueError(f"{what}: schedule step {step} is given twice")
            points[step] = _number(item, f"{what} at step {step}")
        return dict(sorted(points.items()))
    return _number(value, what)


def schedule_value(value: ScheduleValue, env_steps: int) -> float:
    """The value at ``env_steps`` (piecewise linear, constant outside the points)."""
    if not isinstance(value, dict):
        return float(value)
    steps = list(value)                      # sorted by parse_schedule
    if env_steps <= steps[0]:
        return float(value[steps[0]])
    if env_steps >= steps[-1]:
        return float(value[steps[-1]])
    i = bisect.bisect_right(steps, env_steps)
    a, b = steps[i - 1], steps[i]
    return float(value[a] + (value[b] - value[a]) * (env_steps - a) / (b - a))


def schedule_points(value: ScheduleValue) -> list[int]:
    """The schedule's steps (``[0]`` for a number)."""
    return sorted(value) if isinstance(value, dict) else [0]
```

3b. `src/colosseum/league/__init__.py`: `core.config` imports `colosseum.league.schedule`, which runs this package's `__init__`; T3.2 adds modules that import `core.config`, so the package exports lazily (PEP 562) to keep the import graph acyclic and `colosseum.core.config` torch-free. Replace the file's body with the following, keeping any name T2.3 exported as one more `_EXPORTS` entry (e.g. `"PfspStats": "colosseum.league.pfsp"`):

```python
"""Opponent selection (spec block 5): matchmaker interface, built-in mixture matchmaker, PFSP
statistics and share schedules.

Exports are resolved lazily (PEP 562): ``colosseum.core.config`` imports
``colosseum.league.schedule``, and the matchmaker modules import ``colosseum.core.config``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "PfspStats": "colosseum.league.pfsp",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'colosseum.league' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
```

3c. `src/colosseum/core/config.py`.

Imports: add `import logging`, `from typing import Annotated` (next to `Any, Literal`), `BeforeValidator` to the pydantic import, and `from colosseum.league.schedule import ScheduleValue, parse_schedule`; add `logger = logging.getLogger(__name__)` below the imports if T2.1 did not add it.

Module constants, directly above `class AlgorithmConfig`:

```python
# Matchmaking keys an agent may override (spec block 5); shuffle_seats and matchmaker_class stay global.
AGENT_MATCHMAKING_KEYS = frozenset({"opponents", "anchors", "pfsp", "layouts", "teammates", "teammate_self_prob"})
# SP2 matchmaking knobs: accepted only in the raw global matchmaking section, translated, never stored.
SP2_MATCHMAKING_KNOBS = ("mode", "self_play_ratio", "latest_prob", "pfsp_exponent")
_SP2_MATCHMAKING_DEFAULTS = {"mode": "self_play", "self_play_ratio": 0.5, "latest_prob": 0.5, "pfsp_exponent": 1.0}

# A share or an anchor weight: a number or {step: value} points over the run's global env steps.
ScheduleField = Annotated[ScheduleValue, BeforeValidator(parse_schedule)]
```

Replace the whole `class MatchmakingConfig(StrictModel)` with:

```python
class OpponentShares(StrictModel):
    """Shares of the opponent categories, drawn independently for every opposing team (spec block 5).

    Each is a number or a schedule ``{env_step: value}``. Empty categories pass their share to the
    others in proportion; ``rivals`` counts as ``latest`` for a team the owner cannot play.
    """

    latest: ScheduleField = Field(default=0.7, description="The owner's latest weights (self-play); for a team "
                                                            "the owner cannot play, another agent's latest by PFSP.")
    snapshots: ScheduleField = Field(default=0.2, description="Stored snapshots of the agents that play the team "
                                                               "(own, the opponent's in asymmetric games, others'), "
                                                               "by PFSP.")
    rivals: ScheduleField = Field(default=0.0, description="Latest weights of the other trainable agents that play "
                                                            "the team (arena; they collect too), by PFSP.")
    anchors: ScheduleField = Field(default=0.1, description="The owner's anchors (scripted / frozen agents) that "
                                                             "play the team, by their weights.")


class PfspConfig(StrictModel):
    """Prioritized fictitious self-play over candidates of a category (spec block 5)."""

    weighting: Literal["hard", "balanced", "uniform"] = Field(
        default="hard", description="hard: (1 - x)^exponent; balanced: x(1 - x); uniform: 1 (x = the owner's EMA "
                                    "score against the candidate; floor 1e-6).")
    exponent: float = Field(default=2.0, ge=0.0, description="Exponent of the hard weighting.")
    halflife_games: float = Field(default=200.0, gt=0.0, description="EMA half-life of the PFSP score, in games.")


class MatchmakingConfig(StrictModel):
    """How the coordinator builds lineups (spec block 5).

    ``opponents``, ``anchors``, ``pfsp``, ``layouts``, ``teammates`` and ``teammate_self_prob`` may be
    overridden per agent (``agents.<id>.matchmaking``); ``shuffle_seats`` and ``matchmaker_class``
    are global only. The SP2 knobs (``mode``, ``self_play_ratio``, ``latest_prob``,
    ``pfsp_exponent``) are translated from the raw global section (``translate_sp2_matchmaking``).
    """

    opponents: OpponentShares = Field(default_factory=OpponentShares)
    anchors: list[str] | dict[str, ScheduleField] | None = Field(
        default=None,
        description="Scripted / frozen agents the owner meets as anchors: null = every scripted and frozen agent "
                    "(weight 1 each); [] = none; a list of names (weight 1 each) or {name: weight or schedule}.",
    )
    pfsp: PfspConfig = Field(default_factory=PfspConfig)
    layouts: dict[str, float] = Field(
        default_factory=dict,
        description="Layout weights, e.g. {2p: 0.5, 4p: 0.5}. Empty = every layout with equal weight. Only "
                    "layouts with a seat for the data owner's role are drawn.",
    )
    teammates: Literal["self", "mixed"] = Field(
        default="self",
        description="'self' = the team core takes every seat of the team it can play; 'mixed' = each such "
                    "seat goes to the core with probability teammate_self_prob, else to another candidate.",
    )
    teammate_self_prob: float = Field(
        default=0.5, ge=0.0, le=1.0, description="With teammates: mixed, probability that a seat goes to the core.",
    )
    shuffle_seats: bool = Field(
        default=True,
        description="Permute teams with equal role composition and seats of the same role within a team (global).",
    )
    matchmaker_class: str | None = Field(
        default=None,
        description="Dotted path to a colosseum.league.BaseMatchmaker subclass that replaces the built-in "
                    "mixture (global).",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_sp2_knobs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            knobs = [k for k in SP2_MATCHMAKING_KNOBS if k in data]
            if knobs:
                raise ValueError(
                    f"{knobs} are SP2 matchmaking knobs: they are accepted only in the global matchmaking section "
                    f"of the input config (translated into opponents and pfsp); write opponents/pfsp here"
                )
        return data

    @field_validator("layouts")
    @classmethod
    def _check_layout_weights(cls, layouts: dict[str, float]) -> dict[str, float]:
        bad = {name: weight for name, weight in layouts.items() if not weight > 0}
        if bad:
            raise ValueError(f"matchmaking.layouts weights must be > 0, got {bad}")
        return layouts

    @field_validator("anchors")
    @classmethod
    def _check_anchor_names(cls, anchors: Any) -> Any:
        names = list(anchors) if anchors is not None else []
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"matchmaking.anchors: names must be non-empty strings, got {names}")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"matchmaking.anchors lists {duplicates} more than once")
        return anchors

    @field_validator("matchmaker_class")
    @classmethod
    def _check_matchmaker_class(cls, path: str | None) -> str | None:
        if path is not None and "." not in path:
            raise ValueError(f"matchmaking.matchmaker_class must be a dotted path 'module.Class', got {path!r}")
        return path


def translate_sp2_matchmaking(raw: dict) -> dict:
    """Pure translation of SP2 matchmaking knobs in a raw global section (spec block 5).

    Without knobs: a deep copy. With at least one: missing knobs take SP2's defaults
    (``mode: self_play``, ``self_play_ratio: 0.5``, ``latest_prob: 0.5``, ``pfsp_exponent: 1.0``);
    ``spr = 1`` under ``self_play``, else ``self_play_ratio``; ``latest = spr * latest_prob``,
    ``snapshots = spr * (1 - latest_prob)``, ``rivals = 1 - spr``, ``anchors = 0``;
    ``pfsp = {weighting: hard, exponent: pfsp_exponent}``. Knobs together with ``opponents`` or
    ``pfsp``, or a knob out of range, raise ConfigError.
    """
    present = [k for k in SP2_MATCHMAKING_KNOBS if k in raw]
    if not present:
        return copy.deepcopy(raw)
    clash = [k for k in ("opponents", "pfsp") if k in raw]
    if clash:
        raise ConfigError(
            f"matchmaking: the SP2 knobs {present} cannot be combined with {clash}; drop the knobs and set "
            f"matchmaking.opponents / matchmaking.pfsp only"
        )
    knobs = {**_SP2_MATCHMAKING_DEFAULTS, **{k: raw[k] for k in present}}

    def number(name: str, low: float, high: float | None) -> float:
        value = knobs[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
                low <= value and (high is None or value <= high)):
            bound = f"[{low}, {high}]" if high is not None else f">= {low}"
            raise ConfigError(f"matchmaking.{name}={value!r} (SP2 knob) must be a number {bound}")
        return float(value)

    if knobs["mode"] not in ("self_play", "league"):
        raise ConfigError(f"matchmaking.mode={knobs['mode']!r} (SP2 knob) must be 'self_play' or 'league'")
    self_play_ratio = number("self_play_ratio", 0.0, 1.0)        # checked even where mode makes it unused
    spr = 1.0 if knobs["mode"] == "self_play" else self_play_ratio
    latest_prob = number("latest_prob", 0.0, 1.0)
    exponent = number("pfsp_exponent", 0.0, None)
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k not in SP2_MATCHMAKING_KNOBS}
    out["opponents"] = {"latest": spr * latest_prob, "snapshots": spr * (1.0 - latest_prob),
                        "rivals": 1.0 - spr, "anchors": 0.0}
    out["pfsp"] = {"weighting": "hard", "exponent": exponent}
    return out


def merge_matchmaking(base: dict, override: dict) -> dict:
    """A per-agent matchmaking override on the global section (spec block 5).

    ``opponents`` and ``pfsp`` merge key by key (one share's schedule is replaced whole, never merged
    point by point); every other key (``anchors``, ``layouts``, ...) replaces the global value.
    Inputs are not mutated.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in ("opponents", "pfsp") and isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **copy.deepcopy(value)}
        else:
            out[key] = copy.deepcopy(value)
    return out
```

`TrainableAgent` (T1.1): add a validator for its free `matchmaking` dict (inside the class body):

```python
    @field_validator("matchmaking")
    @classmethod
    def _check_matchmaking_override(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        knobs = sorted(set(value) & set(SP2_MATCHMAKING_KNOBS))
        if knobs:
            raise ValueError(f"matchmaking: {knobs} are SP2 knobs, accepted only in the global matchmaking "
                             f"section; set opponents / pfsp in agents.<id>.matchmaking")
        global_only = sorted(set(value) & {"shuffle_seats", "matchmaker_class"})
        if global_only:
            raise ValueError(f"matchmaking: {global_only} are global only; set them in the top-level "
                             f"matchmaking section")
        unknown = sorted(set(value) - AGENT_MATCHMAKING_KEYS)
        if unknown:
            raise ValueError(f"matchmaking: unknown keys {unknown}; an agent may override "
                             f"{sorted(AGENT_MATCHMAKING_KEYS)}")
        return value
```

`_AGENT_SECTIONS`: add `"matchmaking"` (T4.1 adds `"init"`, `"kickstart"`):

```python
_AGENT_SECTIONS = ("networks", "algorithm", "learner", "matchmaking")
```

`ColosseumConfig.get_agent_config`: in its merge loop, the matchmaking section uses `merge_matchmaking`; replace the line `data[section] = deep_merge(data[section], part)` with:

```python
                    merge = merge_matchmaking if section == "matchmaking" else deep_merge
                    data[section] = merge(data[section], part)
```

`ColosseumConfig`: add the translation hook (a second `mode="before"` validator next to the ones T1.1/T2.1 added):

```python
    @model_validator(mode="before")
    @classmethod
    def _translate_sp2_matchmaking_knobs(cls, data: Any) -> Any:
        """SP2 knobs in the raw global matchmaking section -> opponents + pfsp, with one warning."""
        if not isinstance(data, dict):
            return data
        raw = data.get("matchmaking")
        if not isinstance(raw, dict) or not any(k in raw for k in SP2_MATCHMAKING_KNOBS):
            return data
        translated = translate_sp2_matchmaking(raw)
        logger.warning(
            f"SP2 matchmaking knobs {[k for k in SP2_MATCHMAKING_KNOBS if k in raw]} translated to "
            f"opponents={translated['opponents']}, pfsp={translated['pfsp']}; write these in the config "
            f"(the old knobs are not kept in config.resolved.yaml)"
        )
        return {**data, "matchmaking": translated}
```

`--set` of the old knob paths (applied to the raw input, so the translation sees them): T2.1's `LEGACY_OVERRIDE_KEYS` (the dotted keys `apply_overrides` accepts without the schema walk) gets the four paths, right below its definition:

```python
LEGACY_OVERRIDE_KEYS.update(f"matchmaking.{knob}" for knob in SP2_MATCHMAKING_KNOBS)
```

3d. `src/colosseum/coordinator/matchmaker.py` (shim for one task; T3.2 deletes the module). Add `from colosseum.league.schedule import schedule_value`. In `LineupMatchmaker.__init__` replace the line `self._self_play_ratio = 1.0 if config.mode == "self_play" else float(config.self_play_ratio)` with:

```python
        # SP3 T3.1 shim: SP2's numbers read back from the v3 shares at env step 0 (exact for translated
        # SP2 configs; anchors are ignored). The built-in MixtureMatchmaker replaces this class in T3.2.
        latest = schedule_value(config.opponents.latest, 0)
        snapshots = schedule_value(config.opponents.snapshots, 0)
        self._self_play_ratio = 1.0 - schedule_value(config.opponents.rivals, 0)
        self._latest_prob = latest / (latest + snapshots) if latest + snapshots > 0 else 1.0
        self._pfsp_exponent = float(config.pfsp.exponent)
```

and replace `self._config.pfsp_exponent` (in `pfsp_weight`) with `self._pfsp_exponent`, and `self._config.latest_prob` (in `_self_play_core`) with `self._latest_prob`.

3e. `tests/unit/test_sp2_matchmaker.py`: the SP2 tests keep their SP2 knob dicts; only the helper translates them. Change the import to `from colosseum.core.config import MatchmakingConfig, translate_sp2_matchmaking` and replace `make_mm` with:

```python
def make_mm(spec, agent_roles, *, seed=0, ckpts=None, win_rate=None, **config) -> LineupMatchmaker:
    return LineupMatchmaker(
        spec=spec, agent_roles=agent_roles,
        config=MatchmakingConfig.model_validate(translate_sp2_matchmaking(config)),
        checkpoints=lambda agent_id: list((ckpts or {}).get(agent_id, [])),
        win_rate=win_rate or (lambda layout, a, b: 0.5),
        rng=random.Random(seed),
    )
```

3f. `tests/unit/test_config_v2.py` (SP2 field assertions → v3):
- in `test_defaults` replace the line `assert (m.mode, m.layouts, m.self_play_ratio, m.pfsp_exponent, m.latest_prob) == ("self_play", {}, 0.5, 1.0, 0.5)` with
  `assert (m.opponents.latest, m.opponents.snapshots, m.layouts, m.pfsp.exponent, m.anchors) == (0.7, 0.2, {}, 2.0, None)`;
- in `test_matchmaking_bounds` replace the parameter list with
  `{"opponents": {"latest": -1.0}}, {"layouts": {"2p": 0.0}}, {"layouts": {"2p": -1.0}}, {"pfsp": {"exponent": -1.0}}, {"teammates": "random"}, {"teammate_self_prob": 2.0}, {"mode": "league"}, {"layouts": {"2p": float("nan")}},`
  and the body `MatchmakingConfig(**kwargs)` with `MatchmakingConfig.model_validate(kwargs)`;
- `test_matchmaking_layout_weights` becomes:
  ```python
  def test_matchmaking_layout_weights():
      m = MatchmakingConfig(layouts={"2p": 0.25, "4p": 0.75}, teammates="mixed")
      assert m.layouts == {"2p": 0.25, "4p": 0.75} and m.teammates == "mixed"
  ```
- in `test_overrides_reach_the_new_keys` keep `"matchmaking.mode": "league"` (a legacy `--set` path) and replace `cfg.matchmaking.mode == "league"` with `cfg.matchmaking.opponents.rivals == 0.5`.

3g. `scripts/bench_throughput.py::_make_config`: replace the `matchmaking={...}` argument with the explicit v3 form (same workload: one agent, no checkpoints, no anchors):

```python
        matchmaking={
            "opponents": {"latest": 0.5, "snapshots": 0.5, "rivals": 0.0, "anchors": 0.0},
            "anchors": [],
            "pfsp": {"weighting": "hard", "exponent": 1.0, "halflife_games": 200.0},
            "layouts": {LAYOUT: 1.0},
            "teammates": "self",
            "teammate_self_prob": 0.5,
            "shuffle_seats": True,  # one agent, every seat latest+collect: shuffling is a no-op
            "matchmaker_class": None,
        },
```

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_matchmaking_config.py tests/unit/test_config_v2.py tests/unit/test_sp2_matchmaker.py tests/unit/test_sp2_config_overrides.py tests/unit/test_bench_throughput.py tests/unit/test_sp2_coordinator.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean. (Per-agent `matchmaking` overrides are parsed but not used by the SP2 matchmaker until T3.2.)

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/league/schedule.py src/colosseum/league/__init__.py src/colosseum/core/config.py \
    src/colosseum/coordinator/matchmaker.py scripts/bench_throughput.py tests/unit/test_sp3_matchmaking_config.py \
    tests/unit/test_config_v2.py tests/unit/test_sp2_matchmaker.py
git commit -m "feat: matchmaking config v3 with share schedules, anchors, pfsp and SP2 knob translation"
git push origin sp3-league
```

---

### Task T3.2: `colosseum.league`: `BaseMatchmaker`, `MixtureMatchmaker`, lineup checks

Spec block 5 «Алгоритм встроенного матчмейкера», «Проверки конфига», «Свой класс матчмейкера» (interface and lineup checks). The coordinator switches to the new matchmaker in this task, because `coordinator/matchmaker.py` is deleted here (amendment A12); T3.3 adds custom classes, `on_result`, metrics and the `validate` printout.

**Files:**
- Create: `src/colosseum/league/lineups.py`, `src/colosseum/league/base.py`, `src/colosseum/league/mixture.py`
- Modify: `src/colosseum/league/__init__.py` (exports), `src/colosseum/coordinator/coordinator.py`, `src/colosseum/core/validation.py`, `src/colosseum/distributed.py` (import of `enabled_layouts`)
- Delete: `src/colosseum/coordinator/matchmaker.py`, `tests/unit/test_sp2_matchmaker.py` (its guarantees are ported to the new tests: layout restriction and weights, teammates, roles the core does not play, seat balance within 5 %, `permute_seats`, matchmaking validation)
- Modify (test kit): `tests/game_helpers.py` — add `StepCounter`, `make_matchmaker_context`
- Test: `tests/unit/test_sp3_matchmaker.py`, `tests/unit/test_sp3_lineup_checks.py`

**Interfaces:**
- Consumes: `MatchmakingConfig` v3, `merge_matchmaking`, `ScheduleValue`/`schedule_value`/`schedule_points` (T3.1); `PlayerKey`, `pfsp_weight`, `PfspStats` (T2.3); `FIXED_NETWORK_ID`, `SOURCE_OWNER`, `OPPONENT_CATEGORIES`, `SeatAssignment.source` (T1.3); `ColosseumConfig.agent_kind/get_trainable_agent_ids/fixed_agent_ids/get_agent_config` (T1.1); `resolve_player_roles` (T1.2); `Coordinator(config, spec, player_roles, checkpoint_dir, env_steps)` (T1.4).
- Produces (contract): `enabled_layouts`, `permute_seats`, `check_lineup` (`league/lineups.py`); `AgentView`, `MatchmakerContext`, `BaseMatchmaker` (`league/base.py`); `MixtureMatchmaker`, `validate_matchmaking`, `effective_mix`, `describe_mix` (`league/mixture.py`).
- Produces (additions, amendment A4): `MatchmakerContext` dataclass fields `snapshots_fn`, `pfsp_fn`, `env_steps_fn`; `MatchmakerContext.from_config(config, spec, player_roles, *, rng=None, snapshots_fn=..., pfsp_fn=None, env_steps_fn=...)`; `MatchmakerContext.fixed` (scripted + frozen ids) and `anchors_at(owner, env_steps)`; `playable_layouts(spec, config, roles) -> list[str]`; `load_matchmaker_class(path) -> type[BaseMatchmaker]` (used by T3.3); `CATEGORIES`, `FALLBACK`, `schedule_points_of(config) -> list[int]` (mixture); `Coordinator.context` property.

- [ ] **Step 1: Add the test kit**

Append to `tests/game_helpers.py`:

```python
# ---------------------------------------------------------------------------
# SP3 (T3.2): matchmaker contexts over plain dicts
# ---------------------------------------------------------------------------


class StepCounter:
    """A settable env-step source (``MatchmakerContext.env_steps_fn``, ``Coordinator(env_steps=...)``)."""

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def __call__(self) -> int:
        return self.value


def make_matchmaker_context(spec, roles, *, kinds=None, matchmaking=None, per_agent=None, snapshots=None,
                            scores=None, env_steps=None, seed: int = 0):
    """A ``MatchmakerContext`` over plain dicts.

    ``roles``: ``{agent: [roles]}`` in config order; ``kinds``: ``{agent: kind}`` (default trainable);
    ``matchmaking``: the raw global section; ``per_agent``: raw per-agent overrides;
    ``snapshots``: ``{agent: [checkpoint ids]}``; ``scores``: ``{(owner, (agent, network)): PFSP score}``
    (default 0.5); ``env_steps``: a callable (default 0).
    """
    import random as _random

    from colosseum.core.config import MatchmakingConfig, merge_matchmaking
    from colosseum.league.base import AgentView, MatchmakerContext

    kinds = kinds or {}
    agents = {a: AgentView(a, kinds.get(a, "trainable"), frozenset(r)) for a, r in roles.items()}
    trainable = [a for a in roles if agents[a].kind == "trainable"]
    base = MatchmakingConfig.model_validate(matchmaking or {}).model_dump(by_alias=True)
    configs = {a: MatchmakingConfig.model_validate(merge_matchmaking(base, (per_agent or {}).get(a, {})))
               for a in trainable}
    return MatchmakerContext(
        spec=spec, agents=agents, trainable=trainable, matchmaking=configs, rng=_random.Random(seed),
        snapshots_fn=lambda agent_id: list((snapshots or {}).get(agent_id, [])),
        pfsp_fn=lambda layout, owner, player: (scores or {}).get((owner, player), 0.5),
        env_steps_fn=env_steps if env_steps is not None else StepCounter(0),
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp3_matchmaker.py`:

```python
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
```

Create `tests/unit/test_sp3_lineup_checks.py`:

```python
"""check_lineup, validate_matchmaking, effective_mix and describe_mix (spec block 5, T3.2)."""
from __future__ import annotations

import re

import gymnasium
import numpy as np
import pytest

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.lineups import check_lineup
from colosseum.league.mixture import describe_mix, effective_mix, validate_matchmaking
from game_helpers import make_matchmaker_context, make_test_config

OBS = gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32)
ACT = gymnasium.spaces.Discrete(3)
HUNTER = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (4,), np.float32), gymnasium.spaces.Discrete(5))
PREY = RoleSpec(gymnasium.spaces.Box(0.0, 1.0, (3,), np.float32), gymnasium.spaces.Discrete(4))
BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def hunt_spec() -> GameSpec:
    return GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1), SeatSpec("prey", 1))},
    )


def cfg(**sections) -> ColosseumConfig:
    return ColosseumConfig.model_validate({**BASE, **sections})


def duel_context():
    return make_matchmaker_context(GameSpec.symmetric(2, OBS, ACT), {"a": ["player"], "b": ["player"],
                                                                     "bot": ["player"]},
                                   kinds={"bot": "scripted"}, snapshots={"a": ["ckpt_v1"]})


def test_valid_lineups_pass():
    ctx = duel_context()
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a", source="owner"),
                                    SeatAssignment("a", "ckpt_v1", False, source="snapshots")]), "M")
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, False)]), "M")
    check_lineup(ctx, Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", LATEST_NETWORK_ID, False)]), "M")


@pytest.mark.parametrize("seats, problem", [
    ([SeatAssignment("a")], "1 seats"),
    ([SeatAssignment("a"), SeatAssignment("ghost")], "unknown agent 'ghost'"),
    ([SeatAssignment("a"), SeatAssignment("a", "ckpt_v7", False)], "no snapshot 'ckpt_v7'"),
    ([SeatAssignment("a"), SeatAssignment("a", "ckpt_v1", True)], "collect"),
    ([SeatAssignment("a"), SeatAssignment("bot", LATEST_NETWORK_ID, False)], "must play network 'fixed'"),
    ([SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, True)], "collect"),
    ([SeatAssignment("a"), SeatAssignment("a", source="bogus")], "unknown source 'bogus'"),
])
def test_invalid_lineups_name_the_matchmaker_and_the_problem(seats, problem):
    with pytest.raises(ValueError, match=re.escape(problem)) as info:
        check_lineup(duel_context(), Lineup("2p", seats), "MyMatchmaker")
    assert "MyMatchmaker" in str(info.value) and "2p" in str(info.value)


def test_unknown_layouts_roles_and_non_lineups_are_rejected():
    ctx = duel_context()
    with pytest.raises(ValueError, match="unknown layout '3p'"):
        check_lineup(ctx, Lineup("3p", [SeatAssignment("a")] * 3), "M")
    hunt = make_matchmaker_context(hunt_spec(), {"h": ["hunter"], "p": ["prey"]})
    with pytest.raises(ValueError, match="does not play role 'hunter'"):
        check_lineup(hunt, Lineup("1v3", [SeatAssignment("p")] * 4), "M")
    with pytest.raises(ValueError, match="must return a Lineup"):
        check_lineup(ctx, ["not", "a", "lineup"], "M")


def test_validate_matchmaking_accepts_the_defaults():
    validate_matchmaking(GameSpec.symmetric(2, OBS, ACT), {"agent_0": ["player"]}, cfg())


def test_layout_and_anchor_names_are_checked():
    spec = GameSpec.symmetric([2, 4], OBS, ACT)
    with pytest.raises(ConfigError, match="unknown layouts"):
        validate_matchmaking(spec, {"agent_0": ["player"]}, cfg(matchmaking={"layouts": {"3p": 1.0}}))
    roles = {"a": ["player"], "b": ["player"], "bot": ["player"]}
    agents = {"a": {}, "b": {}, "bot": BOT}
    with pytest.raises(ConfigError, match="'ghost'"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["ghost"]}))
    with pytest.raises(ConfigError, match="trainable agent 'b'"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["b"]}))
    with pytest.raises(ConfigError, match="agent 'a'"):        # per-agent anchors are checked per agent
        validate_matchmaking(spec, roles, cfg(agents={**agents, "a": {"matchmaking": {"anchors": ["ghost"]}}}))
    validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": ["bot"]}))


def test_every_role_needs_a_trainable_agent_or_an_anchor_of_the_owner():
    spec = hunt_spec()
    with pytest.raises(ConfigError, match="prey"):
        validate_matchmaking(spec, {"h": ["hunter"]}, cfg(agents={"h": {"roles": ["hunter"]}}))
    agents = {"h": {"roles": ["hunter"]}, "bot": {**BOT, "roles": ["prey"]}}
    roles = {"h": ["hunter"], "bot": ["prey"]}
    validate_matchmaking(spec, roles, cfg(agents=agents))                    # anchors: null = every fixed agent
    with pytest.raises(ConfigError, match="prey"):
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": []}))
    with pytest.raises(ConfigError, match="prey"):                           # weight 0 at every point
        validate_matchmaking(spec, roles, cfg(agents=agents, matchmaking={"anchors": {"bot": 0.0}}))


def test_every_agent_needs_a_playable_layout():
    spec = GameSpec(
        roles={"hunter": HUNTER, "prey": PREY},
        layouts={"1v3": hunt_spec().layouts["1v3"], "prey_only": (SeatSpec("prey", 0), SeatSpec("prey", 1))},
    )
    config = cfg(agents={"h": {"roles": ["hunter"]}, "q": {"roles": ["prey"]}},
                 matchmaking={"layouts": {"prey_only": 1.0}})
    with pytest.raises(ConfigError, match="agent 'h'"):
        validate_matchmaking(spec, {"h": ["hunter"], "q": ["prey"]}, config)


@pytest.mark.parametrize("opponents, anchors, error", [
    ({"latest": {0: 1.0, 1000: 0.0}, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}, None, "env step 1000"),
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 1.0, "anchors": 0.0}, None, "env step 0"),
    ({"latest": 0.0, "snapshots": 1.0, "rivals": 0.0, "anchors": 0.0}, None, None),   # snapshots count
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 1.0}, [], "env step 0"),
    ({"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 1.0}, ["bot"], None),
])
def test_every_schedule_point_needs_a_fillable_category(opponents, anchors, error):
    spec = GameSpec.symmetric(2, OBS, ACT)
    config = cfg(agents={"agent_0": {}, "bot": BOT}, matchmaking={"opponents": opponents, "anchors": anchors})
    roles = {"agent_0": ["player"], "bot": ["player"]}
    if error is None:
        validate_matchmaking(spec, roles, config)
    else:
        with pytest.raises(ConfigError, match=error):
            validate_matchmaking(spec, roles, config)


def test_effective_mix_without_anchors_spreads_their_share():
    ctx = make_matchmaker_context(GameSpec.symmetric(2, OBS, ACT), {"a": ["player"]})
    assert effective_mix(ctx, "a", "2p", 0, 0) == pytest.approx({"latest": 0.7 / 0.9, "snapshots": 0.2 / 0.9})


def test_describe_mix_prints_shares_at_the_start_and_end_of_the_schedules_and_the_anchors():
    config = make_test_config(
        "turns", agents={"agent_0": {}, "bot": BOT},
        matchmaking={"opponents": {"latest": {0: 0.6, 1000: 0.4}, "snapshots": 0.2, "rivals": 0.0,
                                   "anchors": {0: 0.2, 1000: 0.4}}})
    lines = describe_mix(config, env_spec(config))
    text = "\n".join(lines)
    assert lines[0].startswith("agent 'agent_0': opponents by pfsp hard")
    assert ("step 0: latest 0.60, snapshots 0.20, rivals 0.00, anchors 0.20; "
            "step 1000: latest 0.40, snapshots 0.20, rivals 0.00, anchors 0.40") in text
    assert "  anchors: bot 1" in lines
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_matchmaker.py tests/unit/test_sp3_lineup_checks.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.league.base'` (from `game_helpers.make_matchmaker_context` and the imports of `colosseum.league.lineups` / `mixture`).

- [ ] **Step 4: Write `src/colosseum/league/lineups.py`**

```python
"""Lineup helpers of the league (spec block 5): enabled layouts, seat permutation, lineup checks.

``enabled_layouts`` and ``permute_seats`` moved here from SP2's ``coordinator/matchmaker.py``.
``check_lineup`` is the framework's check of every lineup a matchmaker (built-in or custom)
returns.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from colosseum.core.config import MatchmakingConfig
from colosseum.core.types import (
    FIXED_NETWORK_ID,
    LATEST_NETWORK_ID,
    OPPONENT_CATEGORIES,
    SOURCE_OWNER,
    Lineup,
    SeatAssignment,
)
from colosseum.envs.game import GameSpec

if TYPE_CHECKING:
    from colosseum.league.base import MatchmakerContext

_SOURCES = ("", SOURCE_OWNER, *OPPONENT_CATEGORIES)


def enabled_layouts(spec: GameSpec, config: MatchmakingConfig) -> dict[str, float]:
    """``{layout: weight}`` matchmaking draws from: ``config.layouts``, or every layout with weight 1."""
    if not config.layouts:
        return {name: 1.0 for name in spec.layouts}
    return {name: float(weight) for name, weight in config.layouts.items()}


def playable_layouts(spec: GameSpec, config: MatchmakingConfig, roles: Iterable[str]) -> list[str]:
    """Enabled layouts (of the game) with a seat of one of ``roles``, in config order."""
    roles = set(roles)
    return [name for name in enabled_layouts(spec, config)
            if name in spec.layouts and any(seat.role in roles for seat in spec.layouts[name])]


def permute_seats(spec: GameSpec, layout: str, seats: Sequence[SeatAssignment],
                  rng: random.Random) -> list[SeatAssignment]:
    """Randomly permute ``seats`` while keeping the layout's structure.

    Whole teams move only onto teams with the same multiset of roles; inside a team, an
    assignment moves only onto a seat of the same role. Assignments move as objects, so each
    keeps its ``source``.
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


def check_lineup(context: MatchmakerContext, lineup: Lineup, who: str) -> None:
    """ValueError naming ``who`` and the lineup unless ``lineup`` is valid (spec block 5).

    The layout exists; the seat count is the layout's; every seat's agent exists and plays the
    seat's role; a trainable agent plays ``latest`` or one of its stored snapshots, a scripted or
    frozen agent plays ``fixed``; only the latest weights of trainable agents collect; ``source``
    is empty, ``owner`` or an opponent category.
    """
    if not isinstance(lineup, Lineup):
        raise ValueError(f"{who}: lineup_for must return a Lineup, got {type(lineup).__name__}")

    def fail(problem: str) -> ValueError:
        return ValueError(f"{who}: invalid lineup {lineup}: {problem}")

    spec = context.spec
    if lineup.layout not in spec.layouts:
        raise fail(f"unknown layout {lineup.layout!r} (the game has {sorted(spec.layouts)})")
    seat_specs = spec.layouts[lineup.layout]
    if len(lineup.seats) != len(seat_specs):
        raise fail(f"{len(lineup.seats)} seats for layout {lineup.layout!r}, which has {len(seat_specs)}")
    snapshots: dict[str, set[str]] = {}
    for seat, (assignment, seat_spec) in enumerate(zip(lineup.seats, seat_specs, strict=True)):
        agent_id, network_id = assignment.agent_id, assignment.network_id
        view = context.agents.get(agent_id)
        if view is None:
            raise fail(f"seat {seat}: unknown agent {agent_id!r} (agents: {list(context.agents)})")
        if seat_spec.role not in view.roles:
            raise fail(f"seat {seat}: agent {agent_id!r} does not play role {seat_spec.role!r} "
                       f"(it plays {sorted(view.roles)})")
        trainable = view.kind == "trainable"
        if trainable and network_id != LATEST_NETWORK_ID:
            stored = snapshots.setdefault(agent_id, set(context.snapshots(agent_id)))
            if network_id not in stored:
                raise fail(f"seat {seat}: agent {agent_id!r} has no snapshot {network_id!r} (stored: {sorted(stored)})")
        if not trainable and network_id != FIXED_NETWORK_ID:
            raise fail(f"seat {seat}: {view.kind} agent {agent_id!r} must play network {FIXED_NETWORK_ID!r}, "
                       f"not {network_id!r}")
        if assignment.collect and not (trainable and network_id == LATEST_NETWORK_ID):
            raise fail(f"seat {seat}: only the latest weights of trainable agents collect; "
                       f"{agent_id!r}@{network_id} has collect=True")
        if assignment.source not in _SOURCES:
            raise fail(f"seat {seat}: unknown source {assignment.source!r} (expected one of {list(_SOURCES)})")
```

- [ ] **Step 5: Write `src/colosseum/league/base.py`**

```python
"""Matchmaker interface (spec block 5): ``BaseMatchmaker`` and its read-only ``MatchmakerContext``.

A matchmaker builds the ``Lineup`` of one env whose training data belongs to ``owner``
(``lineup_for``); the coordinator checks every lineup (``check_lineup``) and passes every
finished match to ``on_result``. The context gives the game, the agents with their kinds and
roles, each trainable agent's effective ``matchmaking`` config, the stored snapshots, the PFSP
statistics (read only), the coordinator's env-step count and the run's RNG.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from colosseum.core.config import AgentKind, ColosseumConfig, MatchmakingConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import Lineup, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.league.pfsp import PlayerKey
from colosseum.league.schedule import schedule_value


@dataclass(frozen=True)
class AgentView:
    """An agent as a matchmaker sees it."""

    agent_id: str
    kind: AgentKind
    roles: frozenset[str]


def _no_snapshots(agent_id: str) -> list[str]:
    return []


def _zero_env_steps() -> int:
    return 0


@dataclass
class MatchmakerContext:
    """What a matchmaker may read (spec block 5). ``agents`` and ``trainable`` are in config order;
    ``matchmaking`` holds every trainable agent's effective config (per-agent overrides merged)."""

    spec: GameSpec
    agents: dict[str, AgentView]
    trainable: list[str]
    matchmaking: dict[str, MatchmakingConfig]
    rng: random.Random
    snapshots_fn: Callable[[str], Sequence[str]] = _no_snapshots
    pfsp_fn: Callable[[str, str, PlayerKey], float] | None = None
    env_steps_fn: Callable[[], int] = _zero_env_steps

    @classmethod
    def from_config(cls, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]], *,
                    rng: random.Random | None = None,
                    snapshots_fn: Callable[[str], Sequence[str]] = _no_snapshots,
                    pfsp_fn: Callable[[str, str, PlayerKey], float] | None = None,
                    env_steps_fn: Callable[[], int] = _zero_env_steps) -> MatchmakerContext:
        """The context of a run: every agent of ``player_roles`` (``resolve_player_roles``) with its kind."""
        trainable = config.get_trainable_agent_ids()
        missing = [a for a in [*trainable, *config.fixed_agent_ids()] if a not in player_roles]
        if missing:
            raise ConfigError(f"matchmaking: no roles resolved for agents {missing}")
        agents = {aid: AgentView(aid, config.agent_kind(aid), frozenset(roles)) for aid, roles in player_roles.items()}
        return cls(
            spec=spec, agents=agents, trainable=list(trainable),
            matchmaking={aid: config.get_agent_config(aid).matchmaking for aid in trainable},
            rng=rng if rng is not None else random.Random(config.training.seed),
            snapshots_fn=snapshots_fn, pfsp_fn=pfsp_fn, env_steps_fn=env_steps_fn,
        )

    @property
    def fixed(self) -> list[str]:
        """Scripted and frozen agents, config order."""
        return [a for a, view in self.agents.items() if view.kind != "trainable"]

    def snapshots(self, agent_id: str) -> list[str]:
        """Stored snapshot ids of a trainable agent, ascending version ([] for other agents)."""
        view = self.agents.get(agent_id)
        if view is None or view.kind != "trainable":
            return []
        return list(self.snapshots_fn(agent_id))

    def pfsp_score(self, layout: str, owner: str, player: PlayerKey) -> float:
        """EMA score of ``owner``@latest against ``player`` in ``layout`` (0.5 before any game)."""
        return 0.5 if self.pfsp_fn is None else float(self.pfsp_fn(layout, owner, player))

    def env_steps(self) -> int:
        return int(self.env_steps_fn())

    def anchors(self, owner: str) -> dict[str, float]:
        """``owner``'s effective anchors -> weight at ``env_steps()``."""
        return self.anchors_at(owner, self.env_steps())

    def anchors_at(self, owner: str, env_steps: int) -> dict[str, float]:
        """``owner``'s anchors -> weight at ``env_steps``: ``null`` = every scripted and frozen agent
        (weight 1), a list = weight 1 each, a mapping = its weights or schedules."""
        configured = self.matchmaking[owner].anchors
        if configured is None:
            return {a: 1.0 for a in self.fixed}
        if isinstance(configured, list):
            return {a: 1.0 for a in configured}
        return {a: schedule_value(weight, env_steps) for a, weight in configured.items()}


class BaseMatchmaker(ABC):
    """Base of every matchmaker (``matchmaking.matchmaker_class``). The built-in one is
    ``colosseum.league.mixture.MixtureMatchmaker``."""

    def __init__(self, context: MatchmakerContext) -> None:
        self.context = context

    @abstractmethod
    def lineup_for(self, owner: str) -> Lineup:
        """The lineup of one env whose training data belongs to ``owner`` (a trainable agent)."""

    def on_result(self, result: MatchResult) -> None:
        """Called with every finished match (default: nothing)."""
        return None


def load_matchmaker_class(path: str) -> type[BaseMatchmaker]:
    """Import ``matchmaking.matchmaker_class``; ConfigError unless it is a ``BaseMatchmaker`` subclass."""
    from colosseum.core.registry import import_class

    try:
        cls = import_class(path)
    except Exception as e:  # noqa: BLE001 - any import failure is a config problem
        raise ConfigError(f"matchmaking.matchmaker_class {path!r} cannot be imported: {type(e).__name__}: {e}") from e
    if not issubclass(cls, BaseMatchmaker):
        raise ConfigError(f"matchmaking.matchmaker_class {path!r} must subclass colosseum.league.BaseMatchmaker")
    return cls
```

- [ ] **Step 6: Write `src/colosseum/league/mixture.py`**

```python
"""The built-in matchmaker (spec block 5): opponent categories mixed by shares.

For an env whose training data belongs to owner O (the coordinator rotates owners):

1. **Layout**: by O's ``layouts`` weights among the enabled layouts with a seat of O's roles.
2. **O's team**: uniform among the teams with a seat of O's roles; O@latest takes a random seat
   of its role there (source ``owner``).
3. **The core of every other team T**, drawn independently. E_T = the trainable agents that play
   a role of T. Categories:
   - ``latest``: O@latest if O is in E_T, else the latest weights of an agent of E_T by PFSP;
   - ``snapshots``: every stored snapshot of the agents of E_T by PFSP (own, the opponent's in an
     asymmetric game, other agents');
   - ``rivals``: if O is in E_T, the latest weights of the other agents of E_T by PFSP (these
     seats collect too); otherwise empty, and its share is added to ``latest``;
   - ``anchors``: O's anchors (scripted / frozen agents) that play a role of T, by weight.
   The category is drawn by O's ``opponents`` shares at the coordinator's env step among the
   non-empty categories (the shares of empty ones are spread proportionally); its name is the
   source of every seat of T. If every category with a positive share is empty (e.g.
   ``snapshots: 1`` before the first snapshot), the core comes from ``latest``, else from
   ``anchors``, with source ``fallback`` and one warning per (agent, layout). The core takes a
   random seat of its role in T.
4. **The other seats of every team**: a seat of a role the core plays follows ``teammates``
   (``self``: the core; ``mixed``: the core with probability ``teammate_self_prob``, else uniformly
   another trainable agent's latest weights with that role, a snapshot of the core's agent or one
   of O's anchors with that role); a seat of a role the core does not play gets the latest
   weights of a uniformly drawn trainable agent with that role, else one of O's anchors with it
   (by weight).
5. Only seats with the latest weights of trainable agents collect.
6. ``shuffle_seats``: ``permute_seats`` (teams of equal role composition, seats of equal role).

PFSP weight of a candidate X: ``pfsp_weight(score of O@latest against X, weighting, exponent)``
with O's ``pfsp`` config.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence

from colosseum.core.config import ColosseumConfig, MatchmakingConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.envs.game import GameSpec
from colosseum.league.base import BaseMatchmaker, MatchmakerContext
from colosseum.league.lineups import enabled_layouts, permute_seats, playable_layouts
from colosseum.league.pfsp import PlayerKey, pfsp_weight
from colosseum.league.schedule import schedule_points, schedule_value

logger = logging.getLogger(__name__)

CATEGORIES = ("latest", "snapshots", "rivals", "anchors")
FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Shares
# ---------------------------------------------------------------------------


def category_shares(config: MatchmakingConfig, env_steps: int) -> dict[str, float]:
    """The ``opponents`` shares at ``env_steps``."""
    return {c: schedule_value(getattr(config.opponents, c), env_steps) for c in CATEGORIES}


def fold_rivals(shares: Mapping[str, float], owner_in_team: bool) -> dict[str, float]:
    """For a team the owner cannot play, ``rivals`` counts as ``latest`` (latest of other agents)."""
    out = dict(shares)
    if not owner_in_team:
        out["latest"] += out["rivals"]
        out["rivals"] = 0.0
    return out


def spread(shares: Mapping[str, float], available: set[str]) -> dict[str, float]:
    """Positive shares of the available categories, normalized ({} if none)."""
    positive = {c: s for c, s in shares.items() if c in available and s > 0}
    total = sum(positive.values())
    return {c: s / total for c, s in positive.items()} if total > 0 else {}


def schedule_points_of(config: MatchmakingConfig) -> list[int]:
    """0 and every breakpoint of the share and anchor-weight schedules, ascending."""
    points = {0}
    for category in CATEGORIES:
        points.update(schedule_points(getattr(config.opponents, category)))
    if isinstance(config.anchors, dict):
        for weight in config.anchors.values():
            points.update(schedule_points(weight))
    return sorted(points)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def team_roles(spec: GameSpec, layout: str, team: int) -> frozenset[str]:
    seats = spec.layouts[layout]
    return frozenset(seats[s].role for s in spec.teams(layout)[team])


def owner_teams(spec: GameSpec, layout: str, roles: frozenset[str]) -> list[int]:
    """Teams with a seat of one of ``roles``."""
    seats = spec.layouts[layout]
    return [t for t, members in enumerate(spec.teams(layout)) if any(seats[s].role in roles for s in members)]


def opposing_teams(spec: GameSpec, layout: str, roles: frozenset[str]) -> list[int]:
    """Teams that oppose the owner in at least one draw of its team."""
    own = owner_teams(spec, layout, roles)
    return [t for t in range(spec.num_teams(layout)) if any(o != t for o in own)]


def _team_view(context: MatchmakerContext, owner: str, layout: str, team: int,
               env_steps: int) -> tuple[list[str], bool, dict[str, float]]:
    """``(E_T in config order, owner in E_T, owner's anchors that play a role of T -> weight)``."""
    roles = team_roles(context.spec, layout, team)
    players = [a for a in context.trainable if context.agents[a].roles & roles]
    anchors = {a: w for a, w in context.anchors_at(owner, env_steps).items()
               if a in context.agents and context.agents[a].roles & roles}
    return players, owner in players, anchors


def _structural(context: MatchmakerContext, owner: str, layout: str, team: int,
                env_steps: int) -> tuple[dict[str, float], set[str]]:
    """Folded shares and the categories that can ever be filled (snapshots count as available)."""
    players, owner_in, anchors = _team_view(context, owner, layout, team, env_steps)
    available: set[str] = set()
    if players:
        available.update(("latest", "snapshots"))
    if owner_in and len(players) > 1:
        available.add("rivals")
    if any(w > 0 for w in anchors.values()):
        available.add("anchors")
    return fold_rivals(category_shares(context.matchmaking[owner], env_steps), owner_in), available


def effective_mix(context: MatchmakerContext, owner: str, layout: str, team: int, env_steps: int) -> dict[str, float]:
    """The category shares for opposing team ``team`` of ``owner`` in ``layout`` at ``env_steps``:
    ``rivals`` folded into ``latest`` when the owner cannot play the team, shares of categories that
    can never be filled spread proportionally. Snapshots count as available (their temporary
    absence is the runtime fallback's job). ``{}``: nothing with a positive share can be filled."""
    shares, available = _structural(context, owner, layout, team, env_steps)
    return spread(shares, available)


# ---------------------------------------------------------------------------
# The matchmaker
# ---------------------------------------------------------------------------


class MixtureMatchmaker(BaseMatchmaker):
    """The built-in matchmaker (module docstring)."""

    def __init__(self, context: MatchmakerContext) -> None:
        super().__init__(context)
        self._rng: random.Random = context.rng
        self._warned: set[tuple[str, str]] = set()

    def lineup_for(self, owner: str) -> Lineup:
        ctx = self.context
        if owner not in ctx.matchmaking:
            raise KeyError(f"{owner!r} is not a trainable agent; trainable agents: {ctx.trainable}")
        config = ctx.matchmaking[owner]
        spec = ctx.spec
        env_steps = ctx.env_steps()
        anchors = ctx.anchors_at(owner, env_steps)
        roles = ctx.agents[owner].roles
        weights = enabled_layouts(spec, config)
        layouts = playable_layouts(spec, config, roles)
        if not layouts:
            raise KeyError(f"agent {owner!r} has no playable layout (validate_matchmaking should have caught it)")
        layout = self._rng.choices(layouts, weights=[weights[name] for name in layouts], k=1)[0]
        own_team = self._rng.choice(owner_teams(spec, layout, roles))
        seats: list[SeatAssignment | None] = [None] * spec.layout_size(layout)
        for team, members in enumerate(spec.teams(layout)):
            if team == own_team:
                core, source = (owner, LATEST_NETWORK_ID), SOURCE_OWNER
            else:
                core, source = self._opponent_core(owner, config, layout, team, env_steps)
            self._fill_team(config, anchors, core, source, layout, members, seats)
        lineup_seats: list[SeatAssignment] = seats  # type: ignore[assignment]
        if config.shuffle_seats:
            lineup_seats = permute_seats(spec, layout, lineup_seats, self._rng)
        return Lineup(layout=layout, seats=lineup_seats)

    # -- cores ---------------------------------------------------------

    def _opponent_core(self, owner: str, config: MatchmakingConfig, layout: str, team: int,
                       env_steps: int) -> tuple[PlayerKey, str]:
        ctx = self.context
        players, owner_in, anchors = _team_view(ctx, owner, layout, team, env_steps)
        candidates: dict[str, list[PlayerKey]] = {
            "latest": [(owner, LATEST_NETWORK_ID)] if owner_in else [(a, LATEST_NETWORK_ID) for a in players],
            "snapshots": [(a, c) for a in players for c in ctx.snapshots(a)],
            "rivals": [(a, LATEST_NETWORK_ID) for a in players if a != owner] if owner_in else [],
            "anchors": [(a, FIXED_NETWORK_ID) for a, w in anchors.items() if w > 0],
        }
        filled = {c for c, items in candidates.items() if items}
        mix = spread(fold_rivals(category_shares(config, env_steps), owner_in), filled)
        if mix:
            names = list(mix)
            category = self._rng.choices(names, weights=[mix[c] for c in names], k=1)[0]
            source = category
        else:
            category = "latest" if candidates["latest"] else "anchors"
            if not candidates["anchors"]:
                candidates["anchors"] = [(a, FIXED_NETWORK_ID) for a in anchors]
            if not candidates[category]:
                raise RuntimeError(f"agent {owner!r}, layout {layout!r}, team {team}: nobody can play this team "
                                   f"(validate_matchmaking should have caught it)")
            source = FALLBACK
            if (owner, layout) not in self._warned:
                self._warned.add((owner, layout))
                logger.warning(
                    f"agent {owner!r}, layout {layout!r}: every opponent category with a positive share is empty "
                    f"now (e.g. no snapshot yet); the core comes from {category!r} (source 'fallback'). Logged once "
                    f"per agent and layout."
                )
        return self._pick(owner, config, layout, category, candidates[category], anchors), source

    def _pick(self, owner: str, config: MatchmakingConfig, layout: str, category: str,
              candidates: Sequence[PlayerKey], anchors: Mapping[str, float]) -> PlayerKey:
        if len(candidates) == 1:
            return candidates[0]
        if category == "anchors":
            return self._weighted(candidates, [anchors.get(a, 0.0) for a, _ in candidates])
        weights = [pfsp_weight(self.context.pfsp_score(layout, owner, c), config.pfsp.weighting,
                               config.pfsp.exponent) for c in candidates]
        return self._rng.choices(list(candidates), weights=weights, k=1)[0]

    def _weighted(self, items: Sequence[PlayerKey], weights: Sequence[float]) -> PlayerKey:
        if sum(weights) <= 0:
            return self._rng.choice(list(items))
        return self._rng.choices(list(items), weights=list(weights), k=1)[0]

    # -- teams ---------------------------------------------------------

    def _fill_team(self, config: MatchmakingConfig, anchors: Mapping[str, float], core: PlayerKey, source: str,
                   layout: str, members: Sequence[int], seats: list[SeatAssignment | None]) -> None:
        ctx = self.context
        seat_specs = ctx.spec.layouts[layout]
        core_roles = ctx.agents[core[0]].roles
        core_seat = self._rng.choice([s for s in members if seat_specs[s].role in core_roles])
        for s in members:
            role = seat_specs[s].role
            if s == core_seat or (role in core_roles and config.teammates == "self"):
                player = core
            elif role in core_roles:
                player = self._mixed_teammate(config, anchors, core, role)
            else:
                player = self._role_player(anchors, role)
            seats[s] = self._seat(player, source)

    def _mixed_teammate(self, config: MatchmakingConfig, anchors: Mapping[str, float], core: PlayerKey,
                        role: str) -> PlayerKey:
        if self._rng.random() < config.teammate_self_prob:
            return core
        ctx = self.context
        agent = core[0]
        candidates: list[PlayerKey] = [(a, LATEST_NETWORK_ID) for a in ctx.trainable
                                       if a != agent and role in ctx.agents[a].roles]
        candidates += [(agent, c) for c in ctx.snapshots(agent)]
        candidates += [(a, FIXED_NETWORK_ID) for a, w in anchors.items()
                       if w > 0 and a != agent and a in ctx.agents and role in ctx.agents[a].roles]
        return self._rng.choice(candidates) if candidates else core

    def _role_player(self, anchors: Mapping[str, float], role: str) -> PlayerKey:
        ctx = self.context
        players = [a for a in ctx.trainable if role in ctx.agents[a].roles]
        if players:
            return self._rng.choice(players), LATEST_NETWORK_ID
        options = [a for a in anchors if a in ctx.agents and role in ctx.agents[a].roles]
        if not options:
            raise RuntimeError(f"nobody plays role {role!r} (validate_matchmaking should have caught it)")
        return self._weighted([(a, FIXED_NETWORK_ID) for a in options], [anchors[a] for a in options])

    def _seat(self, player: PlayerKey, source: str) -> SeatAssignment:
        agent, network = player
        collect = network == LATEST_NETWORK_ID and self.context.agents[agent].kind == "trainable"
        return SeatAssignment(agent_id=agent, network_id=network, collect=collect, source=source)


# ---------------------------------------------------------------------------
# Config checks and the validate printout
# ---------------------------------------------------------------------------


def _fmt_shares(shares: Mapping[str, float]) -> str:
    return ", ".join(f"{c} {shares.get(c, 0.0):.2f}" for c in CATEGORIES)


def validate_matchmaking(spec: GameSpec, player_roles: Mapping[str, Sequence[str]], config: ColosseumConfig) -> None:
    """Spec block 5 «Проверки конфига»; ConfigError with a fix hint. Per trainable agent O (its
    effective config):

    - ``layouts`` names layouts of the game; ``anchors`` names scripted or frozen agents;
    - O has at least one playable layout;
    - every role of every playable layout is played by a trainable agent or by an anchor of O
      (with a positive weight at some schedule point);
    - for every playable layout, opposing team and schedule point some category with a positive
      share can be filled (snapshots count as available).
    """
    context = MatchmakerContext.from_config(config, spec, player_roles, rng=random.Random(0))
    fixed = set(context.fixed)
    trainable_roles = {role for a in context.trainable for role in context.agents[a].roles}
    for owner in context.trainable:
        m = context.matchmaking[owner]
        where = f"agent {owner!r}"
        unknown = sorted(set(m.layouts) - set(spec.layouts))
        if unknown:
            raise ConfigError(f"{where}: matchmaking.layouts names unknown layouts {unknown}; the game's layouts are "
                              f"{sorted(spec.layouts)}")
        for name in list(m.anchors) if m.anchors is not None else []:
            if name not in context.agents:
                raise ConfigError(f"{where}: matchmaking.anchors names {name!r}, which is not an agent of the config; "
                                  f"anchors are scripted or frozen agents ({sorted(fixed)})")
            if name not in fixed:
                raise ConfigError(f"{where}: matchmaking.anchors names trainable agent {name!r}; anchors are scripted "
                                  f"or frozen agents: use opponents.rivals to meet other trainable agents, or declare "
                                  f"a snapshot as a frozen agent with path")
        roles = context.agents[owner].roles
        layouts = playable_layouts(spec, m, roles)
        if not layouts:
            raise ConfigError(f"{where} (roles {sorted(roles)}) has no seat in its enabled layouts "
                              f"{sorted(enabled_layouts(spec, m))}; check agents.{owner}.roles and matchmaking.layouts")
        points = schedule_points_of(m)
        live_anchors = {a for p in points for a, w in context.anchors_at(owner, p).items() if w > 0}
        covered = trainable_roles | {role for a in live_anchors for role in context.agents[a].roles}
        for layout in layouts:
            missing = sorted({seat.role for seat in spec.layouts[layout]} - covered)
            if missing:
                raise ConfigError(
                    f"{where}, layout {layout!r}: no trainable agent and no anchor of {owner!r} plays role(s) "
                    f"{missing}; add an agent or a scripted/frozen anchor with these roles, or leave {layout!r} out "
                    f"of matchmaking.layouts"
                )
            for team in opposing_teams(spec, layout, roles):
                for point in points:
                    shares, available = _structural(context, owner, layout, team, point)
                    if not spread(shares, available):
                        raise ConfigError(
                            f"{where}, layout {layout!r}, opposing team {team}: at env step {point} no opponent "
                            f"category with a positive share can be filled (shares: {_fmt_shares(shares)}; "
                            f"fillable: {sorted(available) or 'none'}); give a positive share to one of "
                            f"{sorted(available) or ['anchors (add an anchor)']}"
                        )


def describe_mix(config: ColosseumConfig, spec: GameSpec) -> list[str]:
    """``validate`` lines: every trainable agent's effective mix per playable layout and opposing
    team at env step 0 and at the last schedule point, and its anchors with weights."""
    from colosseum.players.registry import resolve_player_roles

    context = MatchmakerContext.from_config(config, spec, resolve_player_roles(config, spec), rng=random.Random(0))
    lines: list[str] = []
    for owner in context.trainable:
        m = context.matchmaking[owner]
        last = schedule_points_of(m)[-1]
        steps = [0, last] if last > 0 else [0]
        lines.append(f"agent {owner!r}: opponents by pfsp {m.pfsp.weighting} (exponent {m.pfsp.exponent:g}, "
                     f"half-life {m.pfsp.halflife_games:g} games), teammates {m.teammates}")
        roles = context.agents[owner].roles
        for layout in playable_layouts(spec, m, roles):
            teams = opposing_teams(spec, layout, roles)
            if not teams:
                lines.append(f"  {layout}: no opposing team")
            for team in teams:
                parts = [f"step {p}: {_fmt_shares(effective_mix(context, owner, layout, team, p))}" for p in steps]
                lines.append(f"  {layout}, team {team}: " + "; ".join(parts))
        first, final = context.anchors_at(owner, steps[0]), context.anchors_at(owner, steps[-1])
        if not first:
            lines.append("  anchors: none")
        else:
            lines.append("  anchors: " + ", ".join(
                f"{a} {first[a]:g}" if first[a] == final[a] else f"{a} {first[a]:g} -> {final[a]:g}" for a in first))
    return lines
```

- [ ] **Step 7: Exports, coordinator, validation, distributed; delete the SP2 matchmaker**

7a. `src/colosseum/league/__init__.py`: add to `_EXPORTS`:

```python
    "BaseMatchmaker": "colosseum.league.base",
    "MatchmakerContext": "colosseum.league.base",
    "MixtureMatchmaker": "colosseum.league.mixture",
```

7b. `src/colosseum/coordinator/coordinator.py`:
- imports: drop `from colosseum.coordinator.matchmaker import LineupMatchmaker`; add
  ```python
  from colosseum.league.base import BaseMatchmaker, MatchmakerContext
  from colosseum.league.lineups import check_lineup
  from colosseum.league.mixture import MixtureMatchmaker, validate_matchmaking
  ```
- `__init__`: the `PfspStats` that T2.3 constructs gets the configured half-lives (T2.3 ran before `pfsp` existed):
  ```python
          self._pfsp = PfspStats({aid: config.get_agent_config(aid).matchmaking.pfsp.halflife_games
                                  for aid in config.get_trainable_agent_ids()})
  ```
  and the statement `self._matchmaker = LineupMatchmaker(...)` becomes (after the checkpoint manager and `self._pfsp` exist; `player_roles` and `env_steps` are the constructor parameters of T1.4):
  ```python
          validate_matchmaking(spec, player_roles, config)
          self._context = MatchmakerContext.from_config(
              config, spec, player_roles, rng=self._rng, snapshots_fn=self._checkpoint_ids,
              pfsp_fn=self._pfsp.score, env_steps_fn=env_steps,
          )
          self._matchmaker: BaseMatchmaker = MixtureMatchmaker(self._context)
  ```
- the `matchmaker` property returns `BaseMatchmaker`; add
  ```python
      @property
      def context(self) -> MatchmakerContext:
          return self._context
  ```
- `generate_lineups` becomes:
  ```python
      def generate_lineups(self, num_envs: int, env_offset: int) -> list[Lineup]:
          """One lineup per env. Env ``e`` of this batch has global index ``g = env_offset + e``;
          its owner is ``trainable[(g + refresh_round) % n_trainable]`` (SP1 rotation over the trainable
          agents in config order), so every trainable agent owns envs and ownership rotates between
          refreshes. Every lineup passes ``check_lineup`` (ValueError naming the matchmaker class)."""
          agents = self._context.trainable
          if not agents:
              raise ValueError("Coordinator has no trainable agents")
          who = type(self._matchmaker).__name__
          lineups = []
          for e in range(num_envs):
              lineup = self._matchmaker.lineup_for(agents[(env_offset + e + self._refresh_round) % len(agents)])
              check_lineup(self._context, lineup, who)
              lineups.append(lineup)
          return lineups
  ```
  Update the module docstring line about `LineupMatchmaker` to "the matchmaker (``colosseum.league``)".

7c. `src/colosseum/core/validation.py`, in `validate_config`: replace the import `from colosseum.coordinator.matchmaker import enabled_layouts, validate_matchmaking` with `from colosseum.league.mixture import validate_matchmaking`; replace the call `validate_matchmaking(spec, agent_roles, config.matchmaking)` with `validate_matchmaking(spec, player_roles, config)` (`player_roles = resolve_player_roles(config, spec)`, from `colosseum.players.registry`; reuse T1.5's variable if it computes one), and `samples = _exercise_env(config, spec, list(enabled_layouts(spec, config.matchmaking)))` with `samples = _exercise_env(config, spec, _all_enabled_layouts(config, spec))`, adding:

```python
def _all_enabled_layouts(config: ColosseumConfig, spec: GameSpec) -> list[str]:
    """Layouts any trainable agent may play (per-agent ``matchmaking.layouts`` included), first-seen order."""
    from colosseum.league.lineups import enabled_layouts

    names: dict[str, None] = {}
    for aid in config.get_trainable_agent_ids():
        names.update(dict.fromkeys(enabled_layouts(spec, config.get_agent_config(aid).matchmaking)))
    return list(names)
```

Update the `validate_config` docstring bullet "the matchmaking checks (``validate_matchmaking``)" to "(``colosseum.league.mixture.validate_matchmaking``, per agent)".

7d. `src/colosseum/distributed.py::distributed_setup`: `from colosseum.coordinator.matchmaker import enabled_layouts` → `from colosseum.league.lineups import enabled_layouts`.

7e. Delete the SP2 module and its test file (ported above):

```bash
git rm src/colosseum/coordinator/matchmaker.py tests/unit/test_sp2_matchmaker.py
```

Check nothing else imports it: `grep -rn "coordinator.matchmaker\|LineupMatchmaker" src tests scripts` prints nothing.

- [ ] **Step 8: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_matchmaker.py tests/unit/test_sp3_lineup_checks.py tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_validate.py tests/unit/test_sp2_distributed_roles.py tests/unit/test_sp2_launcher_checkpoints.py -q`
Expected: all pass. (The SP2 coordinator tests pass unchanged: `latest_prob: 1.0` gives only self-play; `latest_prob: 0.0` before the first snapshot takes the runtime fallback to latest, then the only snapshot.)

- [ ] **Step 9: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings (the runtime-fallback warning goes through `logging`); ruff clean.

- [ ] **Step 10: Commit**

```bash
git add src/colosseum/league src/colosseum/coordinator/coordinator.py src/colosseum/core/validation.py \
    src/colosseum/distributed.py tests/game_helpers.py tests/unit/test_sp3_matchmaker.py tests/unit/test_sp3_lineup_checks.py
git commit -m "feat: colosseum.league with BaseMatchmaker, MixtureMatchmaker and lineup checks

The coordinator draws lineups from the mixture of opponent categories; coordinator/matchmaker.py
and its SP2 test file are removed (their guarantees are ported to test_sp3_matchmaker.py)."
git push origin sp3-league
```

---

### Task T3.3: Coordinator/launcher integration, share metrics, `validate` mix printout

Spec block 5: schedules evaluated at the coordinator's env steps; `matchmaker_class` (a custom `BaseMatchmaker`, its lineups checked, `on_result` fed); played shares of categories and anchors per agent and layout in `metrics.jsonl` / WandB; `validate` prints the effective mix.

**Files:**
- Modify: `src/colosseum/coordinator/coordinator.py` (custom class, `on_result`)
- Modify: `src/colosseum/launcher.py` (`Launcher._build_coordinator`; `source` survives `_resolve_lineups`)
- Modify: `src/colosseum/metrics/aggregator.py` (`opponent_draws`, opponent cells in `EpisodeAggregator`)
- Modify: `src/colosseum/core/validation.py` (mix lines, custom matchmaker check), `src/colosseum/cli.py` (only if T1.5 does not print `ValidationReport.lines` yet)
- Test: `tests/unit/test_sp3_coordinator_league.py`, `tests/unit/test_sp3_share_metrics.py`, `tests/unit/test_sp3_validate_mix.py`, `tests/integration/test_sp3_anchor_runs.py`

**Interfaces:**
- Consumes: `MatchmakerContext`, `MixtureMatchmaker`, `check_lineup`, `describe_mix`, `load_matchmaker_class`, `validate_matchmaking` (T3.2); `ValidationReport` (T1.5); `SeatResult.source` (T1.3); `Coordinator(..., env_steps=...)` (T1.4); `resolve_player_roles` (T1.2).
- Produces (contract): `Coordinator.report_match_result` calls `matchmaker.on_result`; `generate_lineups` checks every lineup; `validate_config(...).lines` holds the mix.
- Produces (additions, amendment A11): `colosseum.metrics.aggregator.opponent_draws(result) -> tuple[str, list[tuple[str, str | None]]] | None`; the per-agent `episodes` record field `opponents: {layout: {"teams": n, "categories": {category: share}, "anchors": {anchor: share}}}` (inside the agent's own record, so no new reserved agent ids); `Launcher._build_coordinator(setup) -> Coordinator`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_coordinator_league.py`:

```python
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
```

Create `tests/unit/test_sp3_share_metrics.py`:

```python
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
```

Create `tests/unit/test_sp3_validate_mix.py`:

```python
"""`validate` prints the effective opponent mix and checks a custom matchmaker class (spec block 5, T3.3)."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.core.types import LATEST_NETWORK_ID, Lineup, SeatAssignment
from colosseum.league.base import BaseMatchmaker
from game_helpers import make_test_config, write_test_config

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


class SelfOnly(BaseMatchmaker):
    def lineup_for(self, owner):
        return Lineup("2p", [SeatAssignment(owner, LATEST_NETWORK_ID, True)] * 2)


class WrongLayout(BaseMatchmaker):
    def lineup_for(self, owner):
        return Lineup("9p", [SeatAssignment(owner)] * 9)


class Exploding(BaseMatchmaker):
    def lineup_for(self, owner):
        raise RuntimeError("boom")


def test_validate_report_holds_every_agents_mix():
    report = validate_config(make_test_config("turns", agents={"agent_0": {}, "bot": BOT}))
    assert any(line.startswith("agent 'agent_0': opponents by pfsp hard") for line in report.lines)
    assert "  anchors: bot 1" in report.lines


def test_cli_validate_prints_the_mix(tmp_path):
    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"agent_0": {}, "bot": BOT})
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output
    assert "opponents by pfsp hard" in result.output and "anchors: bot 1" in result.output


def test_a_valid_custom_matchmaker_passes():
    validate_config(make_test_config("turns", matchmaking={"matchmaker_class": "test_sp3_validate_mix.SelfOnly"}))


@pytest.mark.parametrize("name, message", [
    ("WrongLayout", "WrongLayout.*unknown layout '9p'"),
    ("Exploding", "lineup_for\\('agent_0'\\) raised RuntimeError: boom"),
])
def test_a_broken_custom_matchmaker_fails_validate(name, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(make_test_config("turns", matchmaking={"matchmaker_class": f"test_sp3_validate_mix.{name}"}))
```

Create `tests/integration/test_sp3_anchor_runs.py`:

```python
"""`colosseum train` with a scripted anchor: anchors are drawn by their share, never collect, are
rating entities, and the played shares reach metrics.jsonl (spec block 5, T3.3)."""
from __future__ import annotations

from pathlib import Path

from cli_runner import run_train
from game_helpers import write_test_config


def test_a_scripted_anchor_plays_its_share_and_shows_in_metrics_and_ratings(tmp_path: Path):
    config = write_test_config(
        tmp_path / "anchor.yaml", "turns",
        agents={"agent_0": {}, "bot": {"kind": "scripted", "class": "colosseum.players.RandomBot"}},
        matchmaking={"opponents": {"latest": 0.5, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.5}},
    )
    run = run_train(config, tmp_path, name="anchor")
    assert run.returncode == 0, run.stderr[-3000:]
    teams = anchors = 0.0
    for record in run.records("episodes"):
        assert record["agent"] == "agent_0"                    # the bot never owns data
        cell = record["opponents"].get("2p")
        if cell:
            teams += cell["teams"]
            anchors += cell["teams"] * cell["categories"]["anchors"]
            assert set(cell["anchors"]) <= {"bot"}
    assert teams >= 100, teams
    assert 0.3 <= anchors / teams <= 0.7, anchors / teams
    assert "bot" in run.ratings()["layouts"]["2p"]["elo"]
    assert not (run.root / "checkpoints" / "bot").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_coordinator_league.py tests/unit/test_sp3_share_metrics.py tests/unit/test_sp3_validate_mix.py -q`
Expected: failures: `AttributeError: 'Launcher' object has no attribute '_build_coordinator'`; `ImportError: cannot import name 'opponent_draws'`; the custom class is not used (`MixtureMatchmaker` built instead, so `isinstance(..., MirrorMatchmaker)` fails); the `validate` lines are missing.

- [ ] **Step 3: Coordinator: custom matchmaker class and `on_result`**

In `Coordinator.__init__` replace `self._matchmaker: BaseMatchmaker = MixtureMatchmaker(self._context)` (T3.2) with:

```python
        path = config.matchmaking.matchmaker_class
        matchmaker_cls = MixtureMatchmaker if path is None else load_matchmaker_class(path)
        try:
            self._matchmaker: BaseMatchmaker = matchmaker_cls(self._context)
        except ConfigError:
            raise
        except Exception as e:  # noqa: BLE001 - a user class failing in __init__ is a config problem
            raise ConfigError(f"matchmaking.matchmaker_class {path!r}: constructing it failed: "
                              f"{type(e).__name__}: {e}") from e
```

(import `load_matchmaker_class` from `colosseum.league.base`). In `report_match_result`, after the ratings and PFSP updates, add `self._matchmaker.on_result(result)` and extend its docstring: "then ``matchmaker.on_result``".

- [ ] **Step 4: Launcher: coordinator with the env-step counter; `source` through `_resolve_lineups`**

Add to `Launcher`:

```python
    def _build_coordinator(self, setup: RunSetup) -> Coordinator:
        """The run's coordinator; its matchmaker reads the global env-step counter (share schedules)."""
        return Coordinator(self._config, setup.spec, setup.player_roles, checkpoint_dir=self._run_dir.checkpoints,
                           env_steps=lambda: int(self._env_step_counter.value))
```

(`setup.player_roles` = the every-agent roles T1.4 passes to the coordinator; use T1.4's field name.) In `launch()` replace the `coordinator = Coordinator(...)` statement with `coordinator = self._build_coordinator(setup)`.

`_resolve_lineups` rebuilds seat assignments; every rebuilt seat must keep its `source` (the played-share metric depends on it). Where it constructs `SeatAssignment(seat.agent_id, <network>, <collect>)`, use `dataclasses.replace(seat, network_id=<network>, collect=<collect>)` instead (T1.4 may already do so; the integration test below fails if a seat loses its source). `MatchRunner._resolve` (T1.3) copies `source` the same way by contract.

- [ ] **Step 5: Played shares in `EpisodeAggregator`**

In `src/colosseum/metrics/aggregator.py`: import `Counter` from `collections` and `FIXED_NETWORK_ID, OPPONENT_CATEGORIES, SOURCE_OWNER` from `colosseum.core.types`; add below `opponent_type`:

```python
def opponent_draws(result: MatchResult) -> tuple[str, list[tuple[str, str | None]]] | None:
    """The matchmaker's draws behind a result: ``(owner, [(category, anchor or None) per opposing team])``.

    The owner's team is the team whose seats carry ``SOURCE_OWNER``; the owner is the only agent with
    latest weights on it. None for results without matchmaker sources (eval, tests) and for owner
    teams with the latest weights of several agents (``teammates: mixed``): their owner is ambiguous,
    and leaving them out does not bias the shares (teammates are drawn independently of opponents).
    The anchor of a team drawn from ``anchors`` is its most frequent agent on ``fixed`` seats.
    """
    by_team: dict[int, list[SeatResult]] = defaultdict(list)
    for seat in result.seats:
        by_team[seat.team].append(seat)
    owner_teams = [team for team, seats in by_team.items() if seats[0].source == SOURCE_OWNER]
    if len(owner_teams) != 1:
        return None
    owner_team = owner_teams[0]
    latest = {s.agent_id for s in by_team[owner_team] if s.network_id == LATEST_NETWORK_ID}
    if len(latest) != 1:
        return None
    draws: list[tuple[str, str | None]] = []
    for team in sorted(by_team):
        if team == owner_team:
            continue
        category = by_team[team][0].source
        if category not in OPPONENT_CATEGORIES:
            return None
        anchor = None
        if category == "anchors":
            fixed = Counter(s.agent_id for s in by_team[team] if s.network_id == FIXED_NETWORK_ID)
            anchor = fixed.most_common(1)[0][0] if fixed else None
        draws.append((category, anchor))
    return latest.pop(), draws


@dataclass
class _OpponentCell:
    """Opposing teams of one (owner, layout) and the categories / anchors they were drawn from."""

    teams: int = 0
    categories: Counter = field(default_factory=Counter)
    anchors: Counter = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        n = self.teams
        return {"teams": n, "categories": {c: self.categories[c] / n for c in OPPONENT_CATEGORIES},
                "anchors": {a: k / n for a, k in sorted(self.anchors.items())}}
```

`EpisodeAggregator`:
- docstring: add the bullet "``opponents[layout]``: opposing teams drawn for the agent as data owner, the played share of every opponent category and of every anchor (``opponent_draws``; played, not drawn: a staged lineup can be replaced before it is played)";
- `_reset`: add `self._opponents: dict[str, dict[str, _OpponentCell]] = defaultdict(lambda: defaultdict(_OpponentCell))`;
- `add`: append at the end of the method
  ```python
          draws = opponent_draws(result)
          if draws is not None:
              owner, teams = draws
              cell = self._opponents[owner][result.layout]
              for category, anchor in teams:
                  cell.teams += 1
                  cell.categories[category] += 1
                  if anchor is not None:
                      cell.anchors[anchor] += 1
  ```
- `flush`: add to every agent's dict
  ```python
                  "opponents": {layout: cell.summary()
                                for layout, cell in sorted(self._opponents.get(agent_id, {}).items())},
  ```

The hub writes the agent's dict as its `episodes` record (`**ep`) and `_episodes_row` passes `opponents` to WandB as `episodes/<agent>/opponents/<layout>/{teams, categories/<c>, anchors/<id>}`; no hub change is needed. `REQUIRED_KEYS` stays (the field is present in every new record, but records of older runs lack it).

- [ ] **Step 6: `validate` prints the mix and checks a custom matchmaker**

In `src/colosseum/core/validation.py::validate_config`, right after the `validate_matchmaking(...)` call, add:

```python
    _check_matchmaker_class(config, spec, player_roles)
    report.lines.extend(describe_mix(config, spec))
```

(`report` is T1.5's `ValidationReport` that `validate_config` returns; import `describe_mix` from `colosseum.league.mixture`), and the helper:

```python
VALIDATE_LINEUPS = 16  # lineups drawn per trainable agent from a custom matchmaker


def _check_matchmaker_class(config: ColosseumConfig, spec: GameSpec, player_roles: dict[str, list[str]]) -> None:
    """A custom ``matchmaking.matchmaker_class``: imports, constructs, and its lineups pass ``check_lineup``."""
    import random

    from colosseum.league.base import MatchmakerContext, load_matchmaker_class
    from colosseum.league.lineups import check_lineup

    path = config.matchmaking.matchmaker_class
    if path is None:
        return
    cls = load_matchmaker_class(path)
    context = MatchmakerContext.from_config(config, spec, player_roles, rng=random.Random(0))
    try:
        matchmaker = cls(context)
    except Exception as e:  # noqa: BLE001 - user code
        raise ConfigError(f"matchmaking.matchmaker_class {path!r}: constructing it failed: "
                          f"{type(e).__name__}: {e}") from e
    for owner in context.trainable:
        for _ in range(VALIDATE_LINEUPS):
            try:
                lineup = matchmaker.lineup_for(owner)
            except Exception as e:  # noqa: BLE001 - user code
                raise ConfigError(f"matchmaking.matchmaker_class {path!r}: lineup_for({owner!r}) raised "
                                  f"{type(e).__name__}: {e}") from e
            try:
                check_lineup(context, lineup, cls.__name__)
            except ValueError as e:
                raise ConfigError(f"matchmaking.matchmaker_class {path!r}: {e}") from e
```

Update the `validate_config` docstring: "- the matchmaking checks per agent, a custom matchmaker's lineups, and the effective opponent mix of every agent (``ValidationReport.lines``)".

`src/colosseum/cli.py::validate_cmd`: if T1.5 does not print the report yet, print it before the `OK` lines:

```python
        report = validate_config(cfg)
        for line in report.lines:
            click.echo(line)
```

- [ ] **Step 7: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_coordinator_league.py tests/unit/test_sp3_share_metrics.py tests/unit/test_sp3_validate_mix.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_coordinator.py tests/integration/test_sp3_anchor_runs.py -q`
Expected: all pass (the integration run takes about 10 s).

- [ ] **Step 8: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/colosseum/coordinator/coordinator.py src/colosseum/launcher.py src/colosseum/metrics/aggregator.py \
    src/colosseum/core/validation.py src/colosseum/cli.py tests/unit/test_sp3_coordinator_league.py \
    tests/unit/test_sp3_share_metrics.py tests/unit/test_sp3_validate_mix.py tests/integration/test_sp3_anchor_runs.py
git commit -m "feat: league in the coordinator and launcher: custom matchmakers, share schedules, played-share metrics, validate mix"
git push origin sp3-league
```

---

### Task T4.1: `init`/`kickstart` config sections, `learner/factory.py`

Spec block 6 «`init` и `kickstart` — секции агента», «Глобальные дефолты и совместимость». The top-level `init:` and `kickstart:` are defaults for every trainable agent, `agents.<id>.init/kickstart` deep-merge onto them; `training.kickstart_*` are translated into the top-level `kickstart:`. The learner builds its algorithm in `learner/factory.py`, used by the launcher and by `run-learner`; the teacher is resolved in the main process into a numpy `TeacherSpec`. This task keeps SP2's teacher semantics working (a `.pt` with the student's architecture) and already resolves frozen agents and checkpoint dirs; T4.4 adds the state-layout rule and the checks for teachers of another architecture, T4.5 the scripted teacher.

**Files:**
- Modify: `src/colosseum/core/config.py` (`InitConfig`, `KickstartConfig`, sections, `training.kickstart_*` translation, `TrainingConfig` loses the four fields)
- Create: `src/colosseum/learner/factory.py`
- Modify: `src/colosseum/launcher.py` (`_learner_main`, teacher resolution in `launch`, learner kwargs)
- Modify: `src/colosseum/distributed.py` (`run_distributed_learner` builds through the factory)
- Modify: `src/colosseum/core/validation.py` (per-agent teacher check replaces `_check_kickstart_teacher`)
- Modify: `src/colosseum/bc/kickstart.py` (docstring only: `kickstart.kl`)
- Modify: `scripts/bench_throughput.py` (`_make_config`: explicit `init` / `kickstart`)
- Modify (test kit): `tests/game_helpers.py` — add `write_ckpt_dir`
- Modify (tests whose SP2 assumptions change): `tests/unit/test_sp2_learner_entry.py` (helper `_learner_main` passes `spec` and `teacher`), `tests/unit/test_sp2_validate.py` (`match="kickstart_teacher"` → `match="kickstart.teacher"` in `test_one_global_teacher_cannot_serve_roles_with_different_spaces`)
- Test: `tests/unit/test_sp3_warmstart_config.py`, `tests/unit/test_sp3_learner_factory.py`

**Interfaces:**
- Consumes: `StrictModel` (`populate_by_name`), `TrainableAgent.init/kickstart` free dicts (T1.1); `LEGACY_OVERRIDE_KEYS` (T2.1); `FrozenSpec`, `load_frozen`, `build_frozen_model`, `resolve_player_roles` (T1.2); `ColosseumConfig.agent_kind/agent_entry` (T1.1).
- Produces (contract): `InitConfig`, `KickstartConfig`, `ColosseumConfig.init/kickstart`, `_AGENT_SECTIONS` with `"init"`, `"kickstart"`; `TeacherSpec`, `resolve_teacher`, `build_algorithm` (`colosseum.learner.factory`).
- Produces (additions, amendments A3/A10): `SP2_KICKSTART_KNOBS: dict[str, str]`, `translate_sp2_kickstart(raw: dict) -> dict` (`core.config`); `build_teacher_model(agent_config, teacher, spec) -> PolicyModel`, `build_kickstart(agent_config, teacher, spec, device) -> KickstartLoss | None` (factory); `_learner_main(..., spec: GameSpec | None = None, teacher: TeacherSpec | None = None)`; `Launcher._teachers: dict[str, TeacherSpec | None]`.

- [ ] **Step 1: Add the test kit**

Append to `tests/game_helpers.py`:

```python
# ---------------------------------------------------------------------------
# SP3 (T4.1): checkpoint dirs for warm-start tests
# ---------------------------------------------------------------------------


def write_ckpt_dir(root, config, agent_id: str = "agent_0", *, model=None, policy_version: int = 5,
                   networks: dict | None = None, seed: int = 0):
    """``root/ckpt_v<policy_version>`` with ``model.pt`` and ``meta.json`` as the coordinator writes them, for
    trainable ``agent_id`` of ``config``: its networks (or ``networks``), roles and role signature. ``model``
    defaults to a fresh model of those networks (torch seed ``seed``). Returns the dir."""
    import json
    from pathlib import Path

    from colosseum.core.config import NetworkConfig
    from colosseum.core.registry import build_model
    from colosseum.core.roles import role_signature

    roles, role = agent_role_of(config, agent_id)
    agent_config = config.get_agent_config(agent_id)
    net = NetworkConfig.model_validate(networks) if networks is not None else agent_config.networks
    if model is None:
        torch.manual_seed(seed)
        model = build_model(agent_config.model_copy(update={"networks": net}), role)
    path = Path(root) / f"ckpt_v{policy_version}"
    path.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / "model.pt")
    meta = {"agent_id": agent_id, "checkpoint_id": path.name, "policy_version": policy_version, "timestamp": 0.0,
            "final": True, "env_steps": 100, "roles": list(roles), "role_signature": role_signature(role),
            "networks": net.model_dump(mode="json", by_alias=True)}
    (path / "meta.json").write_text(json.dumps(meta))
    return path
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp3_warmstart_config.py`:

```python
"""init / kickstart config sections (spec block 6): defaults, aliases, per-agent overrides, and
the translation of SP2's training.kickstart_* (T4.1)."""
from __future__ import annotations

import logging

import pytest
import yaml

from colosseum.core.config import (
    SP2_KICKSTART_KNOBS,
    ColosseumConfig,
    TrainingConfig,
    load_config,
    translate_sp2_kickstart,
)
from colosseum.core.errors import ConfigError

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}
CONFIG_LOGGER = "colosseum.core.config"
SP2_FIELDS = {"kickstart_teacher", "kickstart_lambda", "kickstart_decay_steps", "kickstart_kl"}


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def _kickstart_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == CONFIG_LOGGER and r.levelno == logging.WARNING
            and "training.kickstart_" in r.getMessage()]


def test_defaults_and_no_training_fields():
    cfg = ColosseumConfig.model_validate(BASE)
    assert (cfg.init.from_, cfg.init.strict, cfg.init.critic_warmup_steps) == (None, True, 0)
    k = cfg.kickstart
    assert (k.teacher, k.lambda_, k.decay_steps, k.kl) == (None, 1.0, 50_000, "forward")
    assert not SP2_FIELDS & set(TrainingConfig.model_fields)
    assert set(SP2_KICKSTART_KNOBS) == SP2_FIELDS


def test_yaml_aliases_load_and_dump(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "init": {"from": "bc.pt", "critic_warmup_steps": 5},
                                        "kickstart": {"teacher": "bot", "lambda": 0.3}}))
    assert (cfg.init.from_, cfg.init.critic_warmup_steps) == ("bc.pt", 5) and cfg.kickstart.lambda_ == 0.3
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert dumped["init"]["from"] == "bc.pt" and dumped["kickstart"]["lambda"] == 0.3
    assert load_config(_write(tmp_path, dumped, "resolved.yaml")).init == cfg.init


def test_per_agent_sections_deep_merge_onto_the_global_ones():
    cfg = ColosseumConfig.model_validate({
        **BASE, "init": {"from": "global.pt"}, "kickstart": {"teacher": "bot", "lambda": 0.5},
        "agents": {"a": {"init": {"strict": False}, "kickstart": {"lambda": 0.1}},
                   "b": {"kickstart": {"teacher": None}}},
    })
    a, b = cfg.get_agent_config("a"), cfg.get_agent_config("b")
    assert (a.init.from_, a.init.strict) == ("global.pt", False)
    assert (a.kickstart.teacher, a.kickstart.lambda_) == ("bot", 0.1)
    assert b.kickstart.teacher is None and b.init.from_ == "global.pt"


def test_set_reaches_the_new_sections(tmp_path):
    cfg = load_config(_write(tmp_path, {**BASE, "agents": {"a": {}}}),
                      {"init.from": "x.pt", "agents.a.kickstart.lambda": 0.2})
    assert cfg.init.from_ == "x.pt" and cfg.get_agent_config("a").kickstart.lambda_ == 0.2


@pytest.mark.parametrize("data", [
    {"init": {"critic_warmup_steps": -1}},
    {"init": {"frm": "x"}},
    {"kickstart": {"lambda": -0.1}},
    {"kickstart": {"decay_steps": 0}},
    {"kickstart": {"kl": "sideways"}},
    {"agents": {"a": {"kickstart": {"teachr": "x"}}}},
    {"agents": {"a": {"init": {"strict": "maybe"}}}},
])
def test_bad_values_are_config_errors(tmp_path, data):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {**BASE, **data}))


def test_training_kickstart_knobs_are_translated_once_and_never_stored(tmp_path, caplog):
    data = {**BASE, "training": {"total_timesteps": 100, "kickstart_teacher": "bc.pt", "kickstart_lambda": 0.4,
                                 "kickstart_decay_steps": 10, "kickstart_kl": "reverse"}}
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        cfg = load_config(_write(tmp_path, data))
    assert len(_kickstart_warnings(caplog)) == 1
    k = cfg.kickstart
    assert (k.teacher, k.lambda_, k.decay_steps, k.kl) == ("bc.pt", 0.4, 10, "reverse")
    assert cfg.training.total_timesteps == 100
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert not SP2_FIELDS & set(dumped["training"])
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        again = load_config(_write(tmp_path, dumped, "resolved.yaml"))
        cfg.get_agent_config("agent_0")
    assert not _kickstart_warnings(caplog) and again.kickstart == cfg.kickstart


def test_partially_set_knobs_keep_the_other_defaults():
    cfg = ColosseumConfig.model_validate({**BASE, "training": {"kickstart_lambda": 0.2}})
    assert (cfg.kickstart.teacher, cfg.kickstart.lambda_, cfg.kickstart.decay_steps) == (None, 0.2, 50_000)
    sp2_resolved = {**BASE, "training": {"kickstart_teacher": None, "kickstart_lambda": 1.0,
                                         "kickstart_decay_steps": 50000, "kickstart_kl": "forward"}}
    assert ColosseumConfig.model_validate(sp2_resolved).kickstart == ColosseumConfig.model_validate(BASE).kickstart


def test_knobs_together_with_a_kickstart_section_are_a_config_error(tmp_path):
    data = {**BASE, "training": {"kickstart_teacher": "a.pt"}, "kickstart": {"teacher": "b.pt"}}
    with pytest.raises(ConfigError, match="kickstart"):
        load_config(_write(tmp_path, data))


def test_set_accepts_the_training_kickstart_paths(tmp_path):
    cfg = load_config(_write(tmp_path, BASE),
                      {"training.kickstart_teacher": "x.pt", "training.kickstart_kl": "reverse"})
    assert (cfg.kickstart.teacher, cfg.kickstart.kl) == ("x.pt", "reverse")


def test_translate_sp2_kickstart_is_pure():
    raw = {"training": {"kickstart_lambda": 0.5, "seed": 1}}
    out = translate_sp2_kickstart(raw)
    assert out == {"training": {"seed": 1}, "kickstart": {"lambda": 0.5}}
    assert raw == {"training": {"kickstart_lambda": 0.5, "seed": 1}}
    assert translate_sp2_kickstart(BASE) is BASE
```

Create `tests/unit/test_sp3_learner_factory.py`:

```python
"""learner/factory.py (spec block 6): teachers are resolved in the main process into numpy specs and
built in the learner; the launcher and the distributed learner build their algorithms there (T4.1)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from colosseum.core.errors import ConfigError
from colosseum.core.ipc import assert_no_tensors
from colosseum.core.registry import build_model, env_spec
from colosseum.learner.factory import TeacherSpec, build_algorithm, resolve_teacher
from game_helpers import agent_role_of, make_test_config, make_test_run_dir, write_ckpt_dir, write_test_config

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


def _teacher_pt(tmp_path, config, agent_id: str = "agent_0"):
    """A teacher .pt with the agent's architecture, different from any fresh model."""
    _roles, role = agent_role_of(config, agent_id)
    torch.manual_seed(7)
    source = build_model(config.get_agent_config(agent_id), role)
    with torch.no_grad():
        for p in source.parameters():
            p.add_(0.5)
    path = tmp_path / f"{agent_id}_teacher.pt"
    torch.save(source.state_dict(), path)
    return path, source


def _build(config, agent_id, teacher):
    spec = env_spec(config)
    _roles, role = agent_role_of(config, agent_id)
    return build_algorithm(config.get_agent_config(agent_id), role, spec, device="cpu", teacher=teacher)


def test_no_teacher_builds_plain_appo():
    cfg = make_test_config("turns")
    assert resolve_teacher(cfg, "agent_0", env_spec(cfg)) is None
    algo = _build(cfg, "agent_0", None)
    assert algo._kickstart is None and algo.policy_version == 0


def test_a_pt_teacher_has_the_students_architecture_and_the_kickstart_settings(tmp_path):
    base = make_test_config("turns")
    path, source = _teacher_pt(tmp_path, base)
    cfg = make_test_config("turns", kickstart={"teacher": str(path), "lambda": 0.5, "decay_steps": 10, "kl": "reverse"})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert isinstance(teacher, TeacherSpec) and teacher.kind == "neural" and teacher.bot is None
    assert (teacher.lambda_, teacher.decay_steps, teacher.kl, teacher.source) == (0.5, 10, "reverse", str(path))
    assert teacher.frozen.roles == ("player",)
    assert teacher.frozen.networks["model_class"] == "game_helpers.GameTestModel"
    assert_no_tensors(teacher, "teacher spec")                  # crosses into the learner process
    algo = _build(cfg, "agent_0", teacher)
    kick = algo._kickstart
    assert kick.direction == "reverse" and kick.current_lambda == 0.5
    assert kick.teacher is not algo.model and not any(p.requires_grad for p in kick.teacher.parameters())
    assert np.array_equal(_flat(kick.teacher.state_dict()), _flat(source.state_dict()))


def test_a_frozen_agent_is_a_teacher_by_name(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=3)
    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "prev"}},
                                            "prev": {"kind": "frozen", "path": str(ckpt)}},
                           matchmaking={"anchors": []})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert teacher.kind == "neural" and "prev" in teacher.source
    torch.manual_seed(3)
    expected = build_model(base.get_agent_config("agent_0"), agent_role_of(base, "agent_0")[1])
    assert np.array_equal(_flat(_build(cfg, "agent_0", teacher)._kickstart.teacher.state_dict()),
                          _flat(expected.state_dict()))


def test_a_checkpoint_dir_path_is_a_teacher(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=4)
    cfg = make_test_config("turns", kickstart={"teacher": str(ckpt)})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert teacher.kind == "neural" and teacher.frozen.roles == ("player",)


@pytest.mark.parametrize("teacher, message", [
    ("agent_0", "trainable agent"),
    ("missing_file.pt", "neither an agent"),
])
def test_bad_teacher_references_are_config_errors(teacher, message):
    cfg = make_test_config("turns", kickstart={"teacher": teacher})
    with pytest.raises(ConfigError, match=message):
        resolve_teacher(cfg, "agent_0", env_spec(cfg))


def test_a_scripted_teacher_is_not_supported_yet():
    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "bot"}}, "bot": BOT})
    with pytest.raises(ConfigError, match="scripted"):
        resolve_teacher(cfg, "agent_0", env_spec(cfg))


def test_teachers_are_per_agent(tmp_path):
    base = make_test_config("turns", agents={"a": {}, "b": {}})
    path, _ = _teacher_pt(tmp_path, base, "a")
    cfg = make_test_config("turns", agents={"a": {"kickstart": {"teacher": str(path), "lambda": 0.3}}, "b": {}})
    spec = env_spec(cfg)
    assert resolve_teacher(cfg, "a", spec).lambda_ == 0.3 and resolve_teacher(cfg, "b", spec) is None


def test_launcher_passes_numpy_teacher_specs_and_the_spec_to_learners(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module

    base = make_test_config("turns")
    path, _ = _teacher_pt(tmp_path, base)
    cfg = make_test_config("turns", kickstart={"teacher": str(path)})
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
                raise _Stop

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    (kwargs,) = learner_kwargs
    assert isinstance(kwargs["teacher"], TeacherSpec) and kwargs["teacher"].source == str(path)
    assert kwargs["spec"] == env_spec(cfg)
    assert_no_tensors(kwargs, "learner kwargs")


def test_the_distributed_learner_builds_through_the_factory(tmp_path, monkeypatch, restore_global_rng,
                                                             restore_root_logging):
    import colosseum.distributed as distributed
    import colosseum.learner.learner as learner_module
    import colosseum.transport.grpc_transport as grpc_transport
    import colosseum.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    built = []
    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(distributed.ProcessSupervisor, "install_signal_handlers", lambda self: None)
    monkeypatch.setattr(learner_module, "learner_process", lambda *, algorithm_factory, **kw: built.append(
        algorithm_factory()))
    monkeypatch.setattr(sys, "path", list(sys.path))
    base = make_test_config("turns")
    teacher, _ = _teacher_pt(tmp_path, base)
    config = write_test_config(tmp_path / "cfg.yaml", "turns", kickstart={"teacher": str(teacher), "kl": "reverse"})
    assert distributed.run_distributed_learner(str(config), "agent_0", 0, "localhost:1",
                                               overrides={"run.dir": str(tmp_path / "runs")}) == 0
    (algo,) = built
    assert algo._kickstart is not None and algo._kickstart.direction == "reverse"
```

(`restore_global_rng` and `restore_root_logging` are the fixtures the SP2 learner-entry tests already use, from `tests/conftest.py`.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_warmstart_config.py tests/unit/test_sp3_learner_factory.py -q`
Expected: collection errors: `ImportError: cannot import name 'SP2_KICKSTART_KNOBS'` and `ModuleNotFoundError: No module named 'colosseum.learner.factory'`.

- [ ] **Step 4: Config sections and the translation**

In `src/colosseum/core/config.py`:

Remove `kickstart_teacher`, `kickstart_lambda`, `kickstart_decay_steps`, `kickstart_kl` from `TrainingConfig`. Add after `CheckpointConfig`:

```python
class InitConfig(StrictModel):
    """Warm start of a trainable agent's weights (spec block 6). The top-level section is the default of
    every trainable agent; ``agents.<id>.init`` deep-merges onto it."""

    from_: str | None = Field(
        default=None, alias="from",
        description=".pt state_dict | checkpoint dir (model.pt, role signature checked) | run dir (the agent's "
                    "latest checkpoint there) | name of a frozen agent. Weights only: policy version 0, a fresh "
                    "optimizer, counters from zero. Ignored for agents restored by training.resume_from.",
    )
    strict: bool = Field(
        default=True,
        description="false: load only the tensors whose name and shape match; the rest are listed by validate "
                    "and in the log (no matching tensor at all is an error).",
    )
    critic_warmup_steps: int = Field(
        default=0, ge=0,
        description="The first N learner train steps update only the value path (PolicyModel.value_parameters()): "
                    "policy, entropy and kickstart losses off, normalizer statistics frozen, kickstart decay "
                    "starts afterwards.",
    )


class KickstartConfig(StrictModel):
    """A decaying pull of the student towards a teacher (spec block 6). The top-level section is the default
    of every trainable agent; ``agents.<id>.kickstart`` deep-merges onto it."""

    teacher: str | None = Field(
        default=None,
        description="Name of a frozen or scripted agent, or a path: a .pt (the student's architecture) or a "
                    "checkpoint dir (its own architecture from meta.json). None = no kickstart.",
    )
    lambda_: float = Field(
        default=1.0, ge=0.0, alias="lambda",
        description="Initial weight of the kickstart term; decays linearly to 0 over decay_steps train steps "
                    "(after the critic warm-up).",
    )
    decay_steps: int = Field(default=50_000, ge=1, description="Train steps over which lambda decays to 0.")
    kl: Literal["forward", "reverse"] = Field(
        default="forward",
        description="Neural teachers: 'forward' = KL(teacher || student), 'reverse' = KL(student || teacher). "
                    "Scripted teachers use the label loss -log pi(a_teacher) instead.",
    )
```

Module constants (next to `SP2_MATCHMAKING_KNOBS`):

```python
# SP2 kickstart knobs of the training section -> keys of the top-level kickstart section.
SP2_KICKSTART_KNOBS = {"kickstart_teacher": "teacher", "kickstart_lambda": "lambda",
                       "kickstart_decay_steps": "decay_steps", "kickstart_kl": "kl"}
```

and the pure translation (next to `translate_sp2_matchmaking`):

```python
def translate_sp2_kickstart(raw: dict) -> dict:
    """Move SP2's ``training.kickstart_*`` of a raw config into a top-level ``kickstart`` section.

    Returns ``raw`` itself when there is nothing to translate, else a new dict (inputs are not
    mutated). Old knobs together with a top-level ``kickstart`` raise ConfigError.
    """
    training = raw.get("training")
    if not isinstance(training, dict):
        return raw
    present = [k for k in SP2_KICKSTART_KNOBS if k in training]
    if not present:
        return raw
    if raw.get("kickstart") is not None:
        raise ConfigError(
            f"training.{', training.'.join(present)} (SP2 knobs) cannot be combined with a top-level kickstart "
            f"section; move them into kickstart: {{teacher, lambda, decay_steps, kl}}"
        )
    out = dict(raw)
    out["training"] = {k: v for k, v in training.items() if k not in SP2_KICKSTART_KNOBS}
    out["kickstart"] = {SP2_KICKSTART_KNOBS[k]: copy.deepcopy(training[k]) for k in present}
    return out
```

`_AGENT_SECTIONS = ("networks", "algorithm", "learner", "matchmaking", "init", "kickstart")`.

`ColosseumConfig`: add the fields after `checkpoint`:

```python
    init: InitConfig = Field(default_factory=InitConfig)
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)
```

and the hook (next to `_translate_sp2_matchmaking_knobs`):

```python
    @model_validator(mode="before")
    @classmethod
    def _translate_sp2_kickstart_knobs(cls, data: Any) -> Any:
        """SP2's training.kickstart_* -> the top-level kickstart section, with one warning."""
        if not isinstance(data, dict):
            return data
        translated = translate_sp2_kickstart(data)
        if translated is not data:
            logger.warning(
                f"training.kickstart_* are SP2 knobs; translated to kickstart: {translated['kickstart']} (write that "
                f"section instead; the old keys are not kept in config.resolved.yaml)"
            )
        return translated
```

`LEGACY_OVERRIDE_KEYS.update(f"training.{knob}" for knob in SP2_KICKSTART_KNOBS)` below the existing update.

Update the module docstring: "SP3: ``init`` and ``kickstart`` sections (global + per agent); ``training.kickstart_*`` are translated into ``kickstart``."

- [ ] **Step 5: Create `src/colosseum/learner/factory.py`**

```python
"""Construction of a trainable agent's learner side (spec block 6): algorithm, kickstart teacher,
warm start.

The launcher and ``distributed.run_distributed_learner`` both build algorithms here. Teachers (and,
from T4.2, ``init`` weights) are resolved in the main process (``resolve_teacher``: ConfigError
before any process starts) into numpy dataclasses that cross into the learner process as
``mp.Process`` arguments; the learner turns them into torch objects (``build_algorithm``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.players.registry import BotSpec, FrozenSpec

if TYPE_CHECKING:
    from colosseum.algorithms.base import BaseAlgorithm
    from colosseum.bc.kickstart import KickstartLoss
    from colosseum.networks.model import PolicyModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherSpec:
    """A resolved kickstart teacher (numpy only; picklable)."""

    kind: Literal["neural", "scripted"]
    source: str
    lambda_: float
    decay_steps: int
    kl: Literal["forward", "reverse"]
    frozen: FrozenSpec | None = None      # neural
    bot: BotSpec | None = None            # scripted


def agent_ids(config: ColosseumConfig) -> list[str]:
    """Every agent id of the config (the implicit ``agent_0`` included), config order."""
    return list(dict.fromkeys([*config.get_trainable_agent_ids(), *config.agents]))


def _student_pt(config: ColosseumConfig, agent_id: str, path: str, spec: GameSpec, where: str) -> FrozenSpec:
    """A ``.pt`` teacher (SP2 semantics): the student's networks and roles with the file's weights."""
    from colosseum.coordinator.checkpoint_manager import read_weights_file
    from colosseum.players.registry import resolve_player_roles

    try:
        state = read_weights_file(path)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
    return FrozenSpec(
        agent_id=agent_id, roles=tuple(resolve_player_roles(config, spec)[agent_id]),
        networks=config.get_agent_config(agent_id).networks.model_dump(mode="json", by_alias=True),
        model_state=state, source=str(path),
    )


def _load_frozen(config: ColosseumConfig, agent_id: str, path: str, spec: GameSpec, where: str) -> FrozenSpec:
    from colosseum.players.registry import load_frozen

    try:
        return load_frozen(config, agent_id, path, spec)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from e


def resolve_teacher(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> TeacherSpec | None:
    """The kickstart teacher of trainable ``agent_id`` (effective ``kickstart`` section), or None.

    ``teacher`` is the name of a frozen agent (its own architecture), a ``.pt`` path (the student's
    architecture, SP2) or a checkpoint dir (architecture and roles from its ``meta.json``). The name
    of a trainable agent is a ConfigError with a hint. Main process only (it reads weight files).
    """
    ks = config.get_agent_config(agent_id).kickstart
    if ks.teacher is None:
        return None
    ref = ks.teacher
    where = f"agent {agent_id!r}: kickstart.teacher={ref!r}"
    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "trainable":
            raise ConfigError(
                f"{where} names a trainable agent; a teacher is fixed: to learn from a snapshot, declare it as a "
                f"frozen agent with path (agents.<name>: {{kind: frozen, path: <checkpoint dir>}}) and name that agent"
            )
        if kind == "scripted":
            raise ConfigError(f"{where} names a scripted agent; scripted kickstart teachers are not supported yet")
        entry = config.agent_entry(ref)
        frozen = _load_frozen(config, ref, entry.path, spec, where)
        source = f"frozen agent {ref!r} ({entry.path})"
    else:
        path = Path(ref)
        if path.is_file() and path.suffix == ".pt":
            frozen = _student_pt(config, agent_id, ref, spec, where)
        elif path.is_dir():
            frozen = _load_frozen(config, agent_id, ref, spec, where)
        else:
            raise ConfigError(f"{where} is neither an agent of the config ({agent_ids(config)}) nor an existing .pt "
                              f"file or checkpoint dir")
        source = ref
    return TeacherSpec(kind="neural", source=source, lambda_=float(ks.lambda_), decay_steps=int(ks.decay_steps),
                       kl=ks.kl, frozen=frozen)


def build_teacher_model(agent_config: ColosseumConfig, teacher: TeacherSpec, spec: GameSpec) -> PolicyModel:
    """The neural teacher's model (its own architecture, weights loaded, eval mode); ConfigError when the
    weights do not fit."""
    from colosseum.players.registry import build_frozen_model

    try:
        return build_frozen_model(agent_config, teacher.frozen, spec)
    except ConfigError as e:
        raise ConfigError(f"kickstart teacher {teacher.source}: {e}") from e


def build_kickstart(agent_config: ColosseumConfig, teacher: TeacherSpec | None, spec: GameSpec | None,
                    device: str) -> KickstartLoss | None:
    """The ``KickstartLoss`` of a resolved teacher (None without one)."""
    if teacher is None:
        return None
    from colosseum.bc.kickstart import KickstartLoss

    if teacher.kind != "neural":
        raise ConfigError(f"kickstart teacher {teacher.source}: scripted kickstart teachers are not supported yet")
    model = build_teacher_model(agent_config, teacher, spec)
    model.to(device)
    return KickstartLoss(model, initial_lambda=teacher.lambda_, decay_steps=teacher.decay_steps,
                         direction=teacher.kl)


def build_algorithm(agent_config: ColosseumConfig, role_spec: RoleSpec, spec: GameSpec | None, *, device: str,
                    teacher: TeacherSpec | None) -> BaseAlgorithm:
    """The agent's algorithm (``algorithm.algorithm_class``) around a fresh model for ``role_spec``, with the
    kickstart of ``teacher``. ``spec`` is needed only to build a neural teacher."""
    from colosseum.core.registry import build_model, import_class
    from colosseum.core.specs import ActionSpec

    path = agent_config.algorithm.algorithm_class
    if not path:
        raise ValueError("algorithm.algorithm_class is not set. "
                         "Provide a dotted import path (e.g. 'colosseum.algorithms.appo.APPO').")
    algo_cls = import_class(path)
    model = build_model(agent_config, role_spec)
    kwargs: dict = {"device": device, "pin_memory": agent_config.learner.pin_memory}
    kickstart = build_kickstart(agent_config, teacher, spec, device)
    if kickstart is not None:
        kwargs["kickstart"] = kickstart
        logger.info(f"Kickstart from {teacher.source} (lambda {teacher.lambda_}, decay over {teacher.decay_steps} "
                    f"train steps, {teacher.kind} teacher)")
    return algo_cls(model, agent_config.algorithm, ActionSpec.from_space(role_spec.action_space), **kwargs)
```

- [ ] **Step 6: Launcher, distributed learner, validation, benchmark**

6a. `src/colosseum/launcher.py`:
- `_learner_main` gets two keyword parameters, `spec: GameSpec | None = None` and `teacher: TeacherSpec | None = None` (type-only import `from colosseum.learner.factory import TeacherSpec` under `TYPE_CHECKING`), and its body after the thread setup becomes:
  ```python
      from colosseum.learner.factory import build_algorithm

      def algorithm_factory():
          return build_algorithm(config, role_spec, spec, device=device, teacher=teacher)
  ```
  (the SP2 block from `algo_class_path = ...` to the old `algorithm_factory` is removed; drop the now unused imports `build_model`, `import_class`, `ActionSpec`). Docstring: "The algorithm (and a kickstart teacher resolved by the main process) is built by ``learner.factory.build_algorithm``."
- `Launcher.__init__`: `self._teachers: dict = {}`.
- `launch()`, right after `resume_states = self._resolve_resume(...)`:
  ```python
          from colosseum.learner.factory import resolve_teacher
          # Kickstart teachers are resolved here (ConfigError before any process starts) into numpy specs.
          self._teachers = {aid: resolve_teacher(cfg, aid, setup.spec) for aid in trainable_agents}
  ```
- `_start_children`, learner `kwargs=dict(...)`: add `spec=setup.spec, teacher=self._teachers.get(aid),`.

6b. `src/colosseum/distributed.py::run_distributed_learner`: replace the block from `algo_cls = import_class(acfg.algorithm.algorithm_class)` to the end of the nested `algorithm_factory` with:

```python
    from colosseum.learner.factory import build_algorithm, resolve_teacher

    teacher = resolve_teacher(config, agent_id, setup.spec)

    def algorithm_factory():
        return build_algorithm(acfg, role_spec, setup.spec, device=device, teacher=teacher)
```

and drop the imports ruff then reports as unused (`build_model`, `import_class`, `ActionSpec`, and the module-level `torch` if nothing else uses it).

6c. `src/colosseum/core/validation.py`: replace `_check_kickstart_teacher` with

```python
def _check_kickstart_teachers(config: ColosseumConfig, spec: GameSpec,
                              agent_configs: dict[str, ColosseumConfig]) -> None:
    """Every trainable agent's kickstart teacher resolves, and a neural one builds with its weights."""
    from colosseum.learner.factory import build_teacher_model, resolve_teacher

    for aid, acfg in agent_configs.items():
        teacher = resolve_teacher(config, aid, spec)
        if teacher is not None and teacher.kind == "neural":
            build_teacher_model(acfg, teacher, spec)
```

and the last statement of `validate_config` `_check_kickstart_teacher(config, agent_configs, role_specs)` with `_check_kickstart_teachers(config, spec, agent_configs)`. Docstring bullet: "- every agent's kickstart teacher (``learner.factory.resolve_teacher``) and its weights".

6c'. `src/colosseum/bc/kickstart.py` module docstring: "Direction (``training.kickstart_kl``)" → "Direction (``kickstart.kl``)".

6d. `scripts/bench_throughput.py::_make_config`: add (after `checkpoint=...`):

```python
        init={"from": None, "strict": True, "critic_warmup_steps": 0},
        kickstart={"teacher": None, "lambda": 1.0, "decay_steps": 50_000, "kl": "forward"},
```

6e. `tests/unit/test_sp2_learner_entry.py`: the helper `_learner_main` passes what the launcher passes now:

```python
def _learner_main(config: ColosseumConfig, agent_id: str = "agent_0", **kwargs) -> None:
    from colosseum.core.registry import env_spec
    from colosseum.launcher import _learner_main as main
    from colosseum.learner.factory import resolve_teacher

    spec = env_spec(config)
    _roles, role = agent_role_of(config, agent_id)
    main(agent_id=agent_id, config=config.get_agent_config(agent_id), role_spec=role, spec=spec,
         teacher=resolve_teacher(config, agent_id, spec), trajectory_queue=None, weight_queues=[], stop_event=None,
         metrics_queue=None, **kwargs)
```

`tests/unit/test_sp2_validate.py::test_one_global_teacher_cannot_serve_roles_with_different_spaces`: `match="kickstart_teacher"` → `match="kickstart.teacher"` (the global teacher now applies per agent; its file is unreadable either way).

- [ ] **Step 7: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_warmstart_config.py tests/unit/test_sp3_learner_factory.py tests/unit/test_sp2_learner_entry.py tests/unit/test_sp2_validate.py tests/unit/test_kickstart_v2.py tests/unit/test_sp2_config_overrides.py tests/unit/test_bench_throughput.py -q`
Expected: all pass.

- [ ] **Step 8: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/colosseum/core/config.py src/colosseum/learner/factory.py src/colosseum/launcher.py \
    src/colosseum/distributed.py src/colosseum/core/validation.py src/colosseum/bc/kickstart.py \
    scripts/bench_throughput.py tests/game_helpers.py \
    tests/unit/test_sp3_warmstart_config.py tests/unit/test_sp3_learner_factory.py tests/unit/test_sp2_learner_entry.py \
    tests/unit/test_sp2_validate.py
git commit -m "feat: per-agent init/kickstart sections and learner/factory.py for local and distributed learners"
git push origin sp3-league
```

---

### Task T4.2: `init`: sources, strict/partial, resume precedence, report

Spec block 6 «`init`»: the source is a `.pt`, a checkpoint dir (role signature checked), a run dir (the agent's latest checkpoint there) or a frozen agent's name; only weights are loaded (policy version 0, fresh optimizer, counters from zero); `strict: false` loads the tensors with matching name and shape and reports the rest (no match at all is a ConfigError); agents restored by `training.resume_from` ignore `init` with a log line; `validate` prints the report.

**Files:**
- Modify: `src/colosseum/learner/factory.py` (`InitState`, `resolve_init`, `apply_init`)
- Modify: `src/colosseum/launcher.py` (`Launcher._resolve_init`, learner kwarg `init_state`, `_learner_main` applies it)
- Modify: `src/colosseum/core/validation.py` (report lines)
- Test: `tests/unit/test_sp3_init.py`

**Interfaces:**
- Consumes: `InitConfig` (T4.1); `load_frozen`, `resolve_player_roles` (T1.2); `load_checkpoint_dir`, `read_weights_file`, `check_model_state`, `MODEL_FILE` (`coordinator.checkpoint_manager`); `ValidationReport` (T1.5); `write_ckpt_dir` (T4.1 test kit); the SP2 checkpoint fixture of T0.1 (`SP2_CHECKPOINT`, the `ckpt_v3` dir of `agent_0`, and `SP2_TTT_TINY`, the SP2-format config it was generated with, both in `tests/game_helpers.py`).
- Produces (contract): `InitState(model_state, strict, source, report)`, `resolve_init(config, agent_id, spec) -> InitState | None`.
- Produces (additions, amendment A10): `apply_init(model, init: InitState) -> None`; `Launcher._resolve_init(spec, resume_states) -> dict[str, InitState | None]`; `_learner_main(..., init_state: InitState | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_init.py`:

```python
"""init (spec block 6): .pt, checkpoint dir, run dir and frozen-agent sources; strict and partial
loading with a report; resume takes precedence; the learner applies it (T4.2)."""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pytest
import torch

from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, validate_config
from colosseum.learner.factory import InitState, apply_init, resolve_init
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    agent_role_of,
    make_test_config,
    make_test_run_dir,
    write_ckpt_dir,
)

BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).ravel()
                           for _, v in sorted(state.items())])


def _model(core: str = "none", seed: int = 5):
    cfg = make_test_config("turns", networks={"kwargs": {"core": core, "hidden": 16}})
    torch.manual_seed(seed)
    return build_model(cfg.get_agent_config("agent_0"), agent_role_of(cfg, "agent_0")[1])


def _pt(tmp_path, model, name: str = "init.pt"):
    path = tmp_path / name
    torch.save(model.state_dict(), path)
    return path


def _resolve(config, agent_id: str = "agent_0"):
    return resolve_init(config, agent_id, env_spec(config))


def _fresh(config, agent_id: str = "agent_0"):
    torch.manual_seed(123)
    return build_model(config.get_agent_config(agent_id), agent_role_of(config, agent_id)[1])


def test_without_init_from_there_is_nothing_to_load():
    assert _resolve(make_test_config("turns")) is None
    assert _resolve(make_test_config("turns", init={"critic_warmup_steps": 3})) is None


def test_strict_pt_loads_every_tensor(tmp_path):
    source = _model()
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, source))})
    init = _resolve(cfg)
    assert isinstance(init, InitState) and init.strict and init.source == str(tmp_path / "init.pt")
    assert len(init.report) == 1 and "(strict)" in init.report[0] and "agent 'agent_0'" in init.report[0]
    model = _fresh(cfg)
    apply_init(model, init)
    assert np.array_equal(_flat(model.state_dict()), _flat(source.state_dict()))


def test_strict_mismatch_is_a_config_error_with_a_hint(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model(core="lstm")))})
    with pytest.raises(ConfigError, match="strict: false"):
        _resolve(cfg)


def test_partial_loads_the_matching_tensors_and_reports_the_rest(tmp_path):
    source = _model(core="lstm")
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, source)), "strict": False})
    init = _resolve(cfg)
    assert not init.strict and "(partial)" in init.report[0]
    assert any(line.startswith("  not in the model (ignored)") and "core" in line for line in init.report)
    model = _fresh(cfg)
    apply_init(model, init)
    for key, value in model.state_dict().items():
        assert torch.equal(value, source.state_dict()[key]), key            # every target tensor matched
    target_lstm = make_test_config("turns", networks={"kwargs": {"core": "lstm", "hidden": 16}},
                                   init={"from": str(_pt(tmp_path, _model(core="none"), "plain.pt")),
                                         "strict": False})
    report = _resolve(target_lstm).report
    assert any(line.startswith("  not in the source (kept as initialized)") and "core" in line for line in report)


def test_partial_without_any_match_is_a_config_error(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, torch.nn.Linear(2, 2))), "strict": False})
    with pytest.raises(ConfigError, match="no tensor"):
        _resolve(cfg)


def test_checkpoint_dir_with_its_role_signature(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "old", base, seed=6)
    init = _resolve(make_test_config("turns", init={"from": str(ckpt)}))
    assert init.source == str(ckpt)
    torch.manual_seed(6)
    expected = build_model(base.get_agent_config("agent_0"), agent_role_of(base, "agent_0")[1])
    assert np.array_equal(_flat(init.model_state), _flat(expected.state_dict()))
    with pytest.raises(ConfigError, match="role signature"):          # turns has 3-d observations, simultaneous 2-d
        _resolve(make_test_config("simultaneous", init={"from": str(ckpt)}))
    meta = json.loads((ckpt / "meta.json").read_text())
    del meta["role_signature"]
    (ckpt / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ConfigError, match="no role_signature"):
        _resolve(make_test_config("turns", init={"from": str(ckpt)}))


def test_run_dir_takes_the_agents_latest_checkpoint(tmp_path):
    base = make_test_config("turns")
    agent_dir = tmp_path / "old_run" / "checkpoints" / "agent_0"
    write_ckpt_dir(agent_dir, base, policy_version=3, seed=1)
    latest = write_ckpt_dir(agent_dir, base, policy_version=8, seed=2)
    (agent_dir / ".tmp-ckpt_v9-abc").mkdir()                       # leftovers are ignored
    init = _resolve(make_test_config("turns", init={"from": str(tmp_path / "old_run")}))
    assert init.source == str(latest)
    other = make_test_config("turns", agents={"other": {}}, init={"from": str(tmp_path / "old_run")})
    with pytest.raises(ConfigError, match="no checkpoints of agent 'other'"):
        _resolve(other, "other")


def test_frozen_agent_by_name(tmp_path):
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "prev", base, seed=9)
    cfg = make_test_config("turns", agents={"agent_0": {"init": {"from": "prev"}},
                                            "prev": {"kind": "frozen", "path": str(ckpt)}},
                           matchmaking={"anchors": []})
    init = _resolve(cfg)
    assert "frozen agent 'prev'" in init.source and init.strict


@pytest.mark.parametrize("agents, source, message", [
    ({"agent_0": {}, "bot": BOT}, "bot", "scripted agent"),
    ({"agent_0": {}, "b": {}}, "b", "trainable agent"),
    ({"agent_0": {}}, "nowhere.pt", "neither a frozen agent"),
])
def test_bad_sources_are_config_errors(agents, source, message):
    cfg = make_test_config("turns", agents=agents, init={"from": source})
    with pytest.raises(ConfigError, match=message):
        _resolve(cfg)


def test_init_is_per_agent(tmp_path):
    path = _pt(tmp_path, _model())
    cfg = make_test_config("turns", agents={"a": {}, "b": {"init": {"from": None}}}, init={"from": str(path)})
    assert _resolve(cfg, "a") is not None and _resolve(cfg, "b") is None


def test_validate_prints_the_init_report(tmp_path):
    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model(core="lstm"))), "strict": False})
    lines = validate_config(cfg).lines
    assert any("init from" in line and "(partial)" in line for line in lines)
    assert any(line.startswith("  not in the model (ignored)") for line in lines)


def test_the_sp2_checkpoint_fixture_works_as_init():
    cfg = load_config(SP2_TTT_TINY, {"init.from": str(SP2_CHECKPOINT)})    # SP2-format config, translated
    init = _resolve(cfg)
    assert init.strict and init.source == str(SP2_CHECKPOINT)


def test_resume_takes_precedence_over_init(tmp_path, caplog):
    from colosseum.launcher import Launcher

    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, init={"from": str(_pt(tmp_path, _model()))})
    launcher = Launcher(cfg, make_test_run_dir(cfg, tmp_path))
    with caplog.at_level(logging.INFO, logger="colosseum.launcher"):
        states = launcher._resolve_init(env_spec(cfg), {"a": {"source": "old/checkpoints/a/ckpt_v3"}, "b": None})
    assert states["a"] is None and isinstance(states["b"], InitState)
    assert any("'a'" in r.getMessage() and "init.from" in r.getMessage() and "ignored" in r.getMessage()
               for r in caplog.records)


def test_the_learner_starts_from_the_init_weights_at_version_zero(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.learner.learner as learner_module
    from colosseum.launcher import _learner_main

    source = _model(seed=11)
    cfg = make_test_config("turns", learner={"device": "cpu", "torch_threads": 1},
                           init={"from": str(_pt(tmp_path, source))})
    built = []
    monkeypatch.setattr(learner_module, "learner_process", lambda *, algorithm_factory, **kw: built.append(
        algorithm_factory()))
    monkeypatch.setattr(sys, "path", list(sys.path))
    _roles, role = agent_role_of(cfg, "agent_0")
    before = torch.get_num_threads()
    try:
        _learner_main(agent_id="agent_0", config=cfg.get_agent_config("agent_0"), role_spec=role,
                      spec=env_spec(cfg), init_state=_resolve(cfg), trajectory_queue=None, weight_queues=[],
                      stop_event=None, metrics_queue=None)
    finally:
        torch.set_num_threads(before)
    (algo,) = built
    assert np.array_equal(_flat(algo.model.state_dict()), _flat(source.state_dict()))
    assert algo.policy_version == 0 and algo.state_dict()["optimizer"]["state"] == {}


def test_the_launcher_passes_init_states_to_learners(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module

    cfg = make_test_config("turns", init={"from": str(_pt(tmp_path, _model()))})
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
                raise _Stop

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    (kwargs,) = learner_kwargs
    assert isinstance(kwargs["init_state"], InitState) and kwargs["init_state"].strict
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_init.py -q`
Expected: collection error `ImportError: cannot import name 'InitState' from 'colosseum.learner.factory'`.

- [ ] **Step 3: `InitState`, `resolve_init`, `apply_init` in `src/colosseum/learner/factory.py`**

Add `import re` and `import numpy as np`; then:

```python
@dataclass(frozen=True)
class InitState:
    """Resolved ``init`` weights of a trainable agent (numpy only; picklable). With ``strict=False``
    ``model_state`` holds only the tensors that match the agent's model by name and shape."""

    model_state: dict[str, np.ndarray]
    strict: bool
    source: str
    report: list[str]


_CKPT_DIR_RE = re.compile(r"ckpt_v(\d+)")


def _latest_checkpoint_dir(agent_dir: Path) -> Path | None:
    """The highest-version ``ckpt_v<N>`` dir with a ``model.pt`` (read-only; ``.tmp-*`` leftovers ignored)."""
    if not agent_dir.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in agent_dir.iterdir()
             if p.is_dir() and (m := _CKPT_DIR_RE.fullmatch(p.name)) and (p / "model.pt").is_file()]
    return max(found)[1] if found else None


def _checkpoint_weights(ckpt_dir: Path, signature: str, where: str) -> tuple[dict[str, np.ndarray], str]:
    from colosseum.coordinator.checkpoint_manager import load_checkpoint_dir

    state = load_checkpoint_dir(ckpt_dir)          # read-only; ConfigError naming the dir
    found = state["role_signature"]
    if found is None:
        raise ConfigError(f"{where}: checkpoint {ckpt_dir} has no role_signature in its meta.json (written before "
                          f"SP2); init from its model.pt file instead")
    if found != signature:
        raise ConfigError(f"{where}: checkpoint {ckpt_dir} has role signature {found!r}, but the agent's roles "
                          f"have {signature!r}: the observation/action/global-state spaces differ")
    return state["model_state"], str(ckpt_dir)


def _init_weights(config: ColosseumConfig, agent_id: str, ref: str, signature: str, spec: GameSpec,
                  where: str) -> tuple[dict[str, np.ndarray], str]:
    """``(numpy state_dict, source)`` of an ``init.from`` reference."""
    from colosseum.coordinator.checkpoint_manager import MODEL_FILE, read_weights_file

    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "frozen":
            entry = config.agent_entry(ref)
            frozen = _load_frozen(config, ref, entry.path, spec, where)
            return dict(frozen.model_state), f"frozen agent {ref!r} ({entry.path})"
        if kind == "scripted":
            raise ConfigError(f"{where}: {ref!r} is a scripted agent, which has no weights; record its games "
                              f"(colosseum record), train a network on them (colosseum bc) and init from that .pt")
        raise ConfigError(f"{where}: {ref!r} is a trainable agent of this run, which has no weights yet; init from a "
                          f"checkpoint dir or run dir path, or declare that snapshot as a frozen agent with path")
    path = Path(ref)
    if path.is_dir() and (path / "checkpoints").is_dir():
        ckpt = _latest_checkpoint_dir(path / "checkpoints" / agent_id)
        if ckpt is None:
            raise ConfigError(f"{where}: the run dir has no checkpoints of agent {agent_id!r}")
        return _checkpoint_weights(ckpt, signature, where)
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return _checkpoint_weights(path, signature, where)
    if path.is_file() and path.suffix == ".pt":
        try:
            return read_weights_file(path), str(path)
        except ValueError as e:
            raise ConfigError(f"{where}: {e}") from e
    raise ConfigError(f"{where} is neither a frozen agent of the config nor a .pt file, a checkpoint dir (with "
                      f"{MODEL_FILE}) or a run dir (with checkpoints/)")


def resolve_init(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> InitState | None:
    """The ``init`` weights of trainable ``agent_id`` (effective ``init`` section), checked against the
    agent's model; None without ``init.from``. Main process only (it reads weight files).

    Strict: every tensor must match by name and shape (ConfigError with the mismatches otherwise).
    Partial: the matching tensors are kept and the rest reported; no match at all is a ConfigError.
    """
    from colosseum.coordinator.checkpoint_manager import check_model_state
    from colosseum.core.registry import build_model
    from colosseum.core.roles import agent_role_spec, role_signature
    from colosseum.players.registry import resolve_player_roles

    agent_config = config.get_agent_config(agent_id)
    init = agent_config.init
    if init.from_ is None:
        return None
    where = f"agent {agent_id!r}: init.from={init.from_!r}"
    role = agent_role_spec(spec, resolve_player_roles(config, spec)[agent_id])
    state, source = _init_weights(config, agent_id, init.from_, role_signature(role), spec, where)
    model = build_model(agent_config, role)
    if init.strict:
        try:
            check_model_state(model, state, source)
        except ConfigError as e:
            raise ConfigError(f"{where}: {e}\nSet init.strict: false to load only the matching tensors") from e
        return InitState(model_state=dict(state), strict=True, source=source,
                         report=[f"agent {agent_id!r}: init from {source} (strict): {len(state)} tensors"])
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(np.asarray(v).shape) for k, v in state.items()}
    matched = [k for k in expected if got.get(k) == expected[k]]
    if not matched:
        raise ConfigError(f"{where}: no tensor of {source} matches the agent's model by name and shape (probably "
                          f"the wrong file)")
    missing = [k for k in expected if k not in got]
    mismatched = [k for k in expected if k in got and got[k] != expected[k]]
    unexpected = sorted(set(got) - set(expected))
    report = [f"agent {agent_id!r}: init from {source} (partial): loaded {len(matched)} of {len(expected)} tensors"]
    if missing:
        report.append(f"  not in the source (kept as initialized): {missing}")
    if mismatched:
        report.append("  shape mismatch (kept as initialized): "
                      + ", ".join(f"{k} {got[k]} vs model {expected[k]}" for k in mismatched))
    if unexpected:
        report.append(f"  not in the model (ignored): {unexpected}")
    return InitState(model_state={k: state[k] for k in matched}, strict=False, source=source, report=report)


def apply_init(model: PolicyModel, init: InitState) -> None:
    """Load ``init`` into a freshly built model (the learner, right after ``build_algorithm``)."""
    from colosseum.core.types import state_dict_from_numpy

    model.load_state_dict(state_dict_from_numpy(init.model_state), strict=init.strict)
    logger.info(f"Init weights loaded from {init.source} ({'strict' if init.strict else 'partial'})")
```

Update the module docstring: "Teachers and ``init`` weights are resolved in the main process (``resolve_teacher``, ``resolve_init``) ...".

- [ ] **Step 4: Launcher and validation**

4a. `src/colosseum/launcher.py`:
- `_learner_main` gets `init_state: InitState | None = None` (type-only import), and its `algorithm_factory` becomes:
  ```python
      from colosseum.learner.factory import apply_init, build_algorithm

      def algorithm_factory():
          algorithm = build_algorithm(config, role_spec, spec, device=device, teacher=teacher)
          if init_state is not None:
              apply_init(algorithm.model, init_state)   # weights only: version 0, fresh optimizer
          return algorithm
  ```
- `Launcher.__init__`: `self._init_states: dict = {}`.
- new method:
  ```python
      def _resolve_init(self, spec: GameSpec, resume_states: Mapping[str, dict | None]) -> dict[str, InitState | None]:
          """``init`` weights per trainable agent (spec block 6). An agent restored by ``training.resume_from``
          ignores ``init`` (logged), so a run continues with the same config."""
          from colosseum.learner.factory import resolve_init

          states: dict[str, InitState | None] = {}
          for aid in self._config.get_trainable_agent_ids():
              resumed = resume_states.get(aid)
              if resumed is not None:
                  if self._config.get_agent_config(aid).init.from_ is not None:
                      logger.info(f"Agent '{aid}' resumes from {resumed['source']}; its init.from is ignored")
                  states[aid] = None
                  continue
              state = resolve_init(self._config, aid, spec)
              if state is not None:
                  for line in state.report:
                      logger.info(line)
              states[aid] = state
          return states
  ```
- `launch()`, after the teachers (T4.1): `self._init_states = self._resolve_init(setup.spec, resume_states)`.
- `_start_children`, learner `kwargs`: add `init_state=self._init_states.get(aid),`.

4b. `src/colosseum/core/validation.py::validate_config`, after `_check_kickstart_teachers(...)`:

```python
    from colosseum.learner.factory import resolve_init

    for aid in agent_configs:
        init = resolve_init(config, aid, spec)
        if init is not None:
            report.lines.extend(init.report)
    if config.training.resume_from:
        report.lines.append("training.resume_from is set: agents restored from it ignore init")
```

Docstring bullet: "- every agent's ``init`` source (strict / partial report in ``ValidationReport.lines``)".

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_init.py tests/unit/test_sp3_learner_factory.py tests/unit/test_sp2_launcher_checkpoints.py tests/unit/test_sp2_learner_entry.py -q`
Expected: all pass.

- [ ] **Step 6: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/learner/factory.py src/colosseum/launcher.py src/colosseum/core/validation.py \
    tests/unit/test_sp3_init.py
git commit -m "feat: per-agent init from .pt, checkpoint, run dir or frozen agent, strict or partial, resume first"
git push origin sp3-league
```

---

### Task T4.3: Critic warm-up

Spec block 6 «Прогрев критика»: for the first `init.critic_warmup_steps` learner train steps (one step = one policy version, all epochs and minibatches inside) only `PolicyModel.value_parameters()` are updated; policy, entropy and kickstart losses are off; every other parameter gets no gradient (`grad = None`, so Adam touches neither it nor its state); `NormalizeObs` statistics are not updated, so the workers' policy stays bit-identical; the policy version grows as usual; the LR schedule runs as usual; the kickstart decay starts after the warm-up; the counter is trainer state and a resume continues it.

**Files:**
- Modify: `src/colosseum/networks/model.py` (`PolicyModel.value_parameters`), `src/colosseum/networks/composed.py` (`ComposedModel.value_parameters`)
- Modify: `src/colosseum/algorithms/appo.py` (`critic_warmup_steps`)
- Modify: `src/colosseum/learner/factory.py` (`build_algorithm` passes it)
- Modify: `src/colosseum/core/validation.py` (`_check_critic_warmup`)
- Modify (test kit): `tests/game_helpers.py` — `GameTestModel.value_parameters`
- Modify (tests whose SP2 assumptions change): `tests/unit/test_appo_v2.py` (`STATE_KEYS` gains `"critic_warmup_done"`)
- Test: `tests/unit/test_sp3_critic_warmup.py`

**Interfaces:**
- Consumes: `InitConfig.critic_warmup_steps` (T4.1), `build_algorithm` (T4.1).
- Produces (contract): `PolicyModel.value_parameters() -> list[nn.Parameter]` (default `NotImplementedError`; `ComposedModel`: value head + critic encoder); `APPO(..., critic_warmup_steps: int = 0)`.
- Produces (additions, amendment A13): `APPO.critic_warming_up: bool` (property); trainer-state key `critic_warmup_done` (missing in an older trainer state -> 0, i.e. the warm-up restarts with the fresh optimizer); train metric `critic_warmup` (1.0 during the warm-up).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_critic_warmup.py`:

```python
"""Critic warm-up (spec block 6): only the value path learns for the first N train steps, and the
policy the workers receive stays bit-identical, normalizer statistics included (T4.3)."""
from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, validate_config
from colosseum.core.specs import ActionSpec
from colosseum.core.types import WeightPayload
from colosseum.envs.game import RoleSpec
from colosseum.learner.factory import build_algorithm
from colosseum.networks.base import BaseCriticEncoder, BaseEncoder, EncoderOutput
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import NoCore
from colosseum.networks.model import PolicyModel
from colosseum.networks.normalization import NormalizeObs
from game_helpers import (
    GameTestModel,
    GenericValue,
    TreePolicyHead,
    agent_role_of,
    make_test_config,
    make_test_model,
    synthetic_chunk,
)

ROLE = RoleSpec(gymnasium.spaces.Box(-5.0, 5.0, (4,), np.float32), gymnasium.spaces.Discrete(3),
                gymnasium.spaces.Box(-5.0, 5.0, (2,), np.float32))
SPEC = ActionSpec.from_space(ROLE.action_space)


class NormEncoder(BaseEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.norm = NormalizeObs((4,))
        self.fc = nn.Linear(4, 16)

    @property
    def latent_dim(self) -> int:
        return 16

    def forward(self, obs):
        return EncoderOutput(torch.relu(self.fc(self.norm(obs))), {})


class NormCritic(BaseCriticEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.norm = NormalizeObs((2,), source="global_state")
        self.fc = nn.Linear(2, 8)

    @property
    def output_dim(self) -> int:
        return 8

    def forward(self, global_state):
        return torch.relu(self.fc(self.norm(global_state)))


class NoValueParams(GameTestModel):
    """A model without value_parameters()."""

    def value_parameters(self):
        return PolicyModel.value_parameters(self)


class OldAPPO(APPO):
    """An algorithm class without the critic_warmup_steps argument."""

    def __init__(self, model, config, action_spec, device="cpu", pin_memory=False, kickstart=None):
        super().__init__(model, config, action_spec, device=device, pin_memory=pin_memory, kickstart=kickstart)


def norm_model(seed: int = 0) -> ComposedModel:
    torch.manual_seed(seed)
    return ComposedModel(NormEncoder(), NoCore(16), TreePolicyHead(16, SPEC), GenericValue(24), NormCritic())


def chunks(model) -> list:
    return [synthetic_chunk(model, ROLE, "AAAAAAAB", seed=s) for s in range(4)]


def value_keys(model) -> set[str]:
    ids = {id(p) for p in model.value_parameters()}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def test_value_parameters_of_the_models():
    model = norm_model()
    assert value_keys(model) == {name for name, _ in model.named_parameters()
                                 if name.startswith(("value.", "critic_encoder."))}
    plain = make_test_model(RoleSpec(ROLE.observation_space, ROLE.action_space))
    assert {id(p) for p in plain.value_parameters()} == {id(p) for p in plain.value.parameters()}
    wrapped = GameTestModel(ROLE.observation_space, ROLE.action_space)
    assert {id(p) for p in wrapped.value_parameters()} == {id(p) for p in wrapped.inner.value.parameters()}
    with pytest.raises(NotImplementedError, match="value_parameters"):
        NoValueParams(ROLE.observation_space, ROLE.action_space).value_parameters()


def test_warmup_trains_only_the_value_path_and_keeps_the_workers_policy_bit_identical():
    model = norm_model()
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.3)
    kick = KickstartLoss(teacher, initial_lambda=0.5, decay_steps=10)
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2), SPEC, kickstart=kick, critic_warmup_steps=3)
    batch = chunks(model)
    keys = value_keys(model)
    before = WeightPayload.from_model("a", 0, model).state_dict           # what the workers receive
    for _ in range(3):
        assert algo.critic_warming_up
        metrics = algo.train_step(batch)
        assert metrics["critic_warmup"] == 1.0 and metrics["kickstart_loss"] == 0.0
        assert "policy_loss" in metrics and "entropy" in metrics           # diagnostics stay
    after = WeightPayload.from_model("a", algo.policy_version, model).state_dict
    assert algo.policy_version == 3 and not algo.critic_warming_up
    for key, value in before.items():
        if key not in keys:                                                # policy weights AND normalizer buffers
            assert np.array_equal(value, after[key]), key
    assert any(not np.array_equal(before[k], after[k]) for k in keys)
    value_ids = {id(p) for p in model.value_parameters()}
    adam = {id(p) for p in algo._optimizer.state}
    assert adam and adam <= value_ids                                     # no Adam state for frozen parameters
    assert kick.step_count == 0 and kick.current_lambda == 0.5            # the decay starts after the warm-up
    metrics = algo.train_step(batch)
    final = WeightPayload.from_model("a", algo.policy_version, model).state_dict
    assert metrics["critic_warmup"] == 0.0 and metrics["kickstart_loss"] > 0.0 and kick.step_count == 1
    assert not np.array_equal(after["encoder.fc.weight"], final["encoder.fc.weight"])
    assert not np.array_equal(after["encoder.norm.rms.count"], final["encoder.norm.rms.count"])


def test_the_lr_schedule_runs_as_usual_during_the_warmup():
    model = norm_model()
    warm = APPO(model, AlgorithmConfig(learning_rate=1e-2, lr_schedule="linear"), SPEC, critic_warmup_steps=5)
    warm.set_progress(0.25)
    assert warm._optimizer.param_groups[0]["lr"] == pytest.approx(0.75e-2)


def test_the_warmup_counter_is_trainer_state():
    batch = chunks(norm_model())
    first = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    first.train_step(batch)
    state = first.state_dict()
    assert state["critic_warmup_done"] == 1
    resumed = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    resumed.load_state_dict(state)
    resumed.train_step(batch)
    assert resumed.critic_warming_up
    resumed.train_step(batch)
    assert not resumed.critic_warming_up
    del state["critic_warmup_done"]                                       # a trainer state written before SP3
    older = APPO(norm_model(), AlgorithmConfig(), SPEC, critic_warmup_steps=3)
    older.load_state_dict(state)
    assert older.critic_warming_up and older.state_dict()["critic_warmup_done"] == 0


def test_appo_rejects_a_warmup_it_cannot_do():
    with pytest.raises(ValueError, match="value_parameters"):
        APPO(NoValueParams(ROLE.observation_space, ROLE.action_space), AlgorithmConfig(), SPEC, critic_warmup_steps=1)
    with pytest.raises(ValueError, match="value_loss_coeff"):
        APPO(norm_model(), AlgorithmConfig(value_loss_coeff=0.0), SPEC, critic_warmup_steps=1)
    assert not APPO(norm_model(), AlgorithmConfig(), SPEC).critic_warming_up


def test_build_algorithm_passes_the_agents_warmup():
    cfg = make_test_config("turns", agents={"a": {"init": {"critic_warmup_steps": 2}}, "b": {}})
    spec = env_spec(cfg)
    for agent_id, warming in (("a", True), ("b", False)):
        algo = build_algorithm(cfg.get_agent_config(agent_id), agent_role_of(cfg, agent_id)[1], spec, device="cpu",
                               teacher=None)
        assert algo.critic_warming_up is warming


@pytest.mark.parametrize("sections, message", [
    ({"networks": {"model_class": "test_sp3_critic_warmup.NoValueParams"}}, "value_parameters"),
    ({"algorithm": {"algorithm_class": "test_sp3_critic_warmup.OldAPPO"}}, "takes no critic_warmup_steps"),
    ({"algorithm": {"value_loss_coeff": 0.0}}, "value_loss_coeff"),
])
def test_validate_rejects_a_warmup_the_agent_cannot_do(sections, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(make_test_config("turns", init={"critic_warmup_steps": 3}, **sections))
    validate_config(make_test_config("turns", **sections))                # fine without a warm-up
```

In `tests/unit/test_appo_v2.py` set `STATE_KEYS = {"optimizer", "progress", "scaler", "kickstart", "policy_version", "consumed_samples", "critic_warmup_done"}`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_critic_warmup.py tests/unit/test_appo_v2.py -q`
Expected: failures: `AttributeError: 'ComposedModel' object has no attribute 'value_parameters'`, `TypeError: APPO.__init__() got an unexpected keyword argument 'critic_warmup_steps'`, the `STATE_KEYS` assertions of `test_appo_v2.py`.

- [ ] **Step 3: `value_parameters`**

`src/colosseum/networks/model.py`, in `PolicyModel` (after `reset_state`):

```python
    def value_parameters(self) -> list[nn.Parameter]:
        """Parameters only the value path uses (critic warm-up, ``init.critic_warmup_steps``): the value head and
        e.g. a critic encoder, never a parameter the policy uses. Models that support the warm-up override it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement value_parameters(); return the value head's parameters "
            f"(and a critic encoder's) to use init.critic_warmup_steps"
        )
```

`src/colosseum/networks/composed.py`, in `ComposedModel`:

```python
    def value_parameters(self) -> list[torch.nn.Parameter]:
        """The value head and the critic encoder (the encoder and the core feed the policy too)."""
        params = list(self.value.parameters())
        if self.critic_encoder is not None:
            params += list(self.critic_encoder.parameters())
        return params
```

`tests/game_helpers.py`, in `GameTestModel`:

```python
    def value_parameters(self):
        return self.inner.value_parameters()
```

- [ ] **Step 4: APPO**

In `src/colosseum/algorithms/appo.py`:

Module docstring, new bullet: "- Critic warm-up (``critic_warmup_steps``, spec block 6 of SP3): the first N train steps optimize only ``value_loss_coeff * value_loss`` over ``PolicyModel.value_parameters()``; every other parameter has ``requires_grad`` off for the step (no gradient, no Adam update or state), normalizer statistics are not updated, kickstart is off and its decay waits; policy loss and entropy are still reported."

`APPO.__init__`: add the parameter `critic_warmup_steps: int = 0` after `kickstart`, and after the optimizer is created:

```python
        # Critic warm-up (SP3 spec block 6): the first N train steps update only the value path.
        if critic_warmup_steps < 0:
            raise ValueError(f"critic_warmup_steps must be >= 0, got {critic_warmup_steps}")
        self._critic_warmup_steps = int(critic_warmup_steps)
        self._critic_warmup_done = 0
        self._frozen_in_warmup: list[torch.nn.Parameter] = []
        if self._critic_warmup_steps > 0:
            if not config.value_loss_coeff > 0:
                raise ValueError("critic_warmup_steps > 0 needs algorithm.value_loss_coeff > 0 "
                                 "(the warm-up trains only the value loss)")
            try:
                value_params = list(self._model.value_parameters())
            except NotImplementedError as e:
                raise ValueError(f"critic_warmup_steps={critic_warmup_steps} needs "
                                 f"PolicyModel.value_parameters(): {e}") from e
            if not value_params:
                raise ValueError(f"critic_warmup_steps={critic_warmup_steps}: "
                                 f"{type(self._model).__name__}.value_parameters() returned no parameters")
            value_ids = {id(p) for p in value_params}
            self._frozen_in_warmup = [p for p in self._model.parameters() if id(p) not in value_ids]
```

New property (after `modes`):

```python
    @property
    def critic_warming_up(self) -> bool:
        """True during the first ``critic_warmup_steps`` train steps (value path only)."""
        return self._critic_warmup_done < self._critic_warmup_steps
```

`compute_loss`: `return self._loss_from_batch(self._prepare_batch(chunks), warmup=self.critic_warming_up)`.

`_loss_from_batch(self, batch: dict, warmup: bool = False)`: replace

```python
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy

        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
```

with

```python
        if warmup:                                   # critic warm-up: the value path only
            total_loss = cfg.value_loss_coeff * value_loss
        else:
            total_loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy

        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0 and not warmup:
```

and add `warmup` to its docstring ("``warmup``: the critic warm-up's loss (value only, no kickstart)").

`train_step`:
- first statement after `cfg = self._config`: `warmup = self.critic_warming_up`;
- `self._update_normalizers(full_batch)` becomes
  ```python
          if not warmup:                   # frozen during the warm-up: the workers' policy stays bit-identical
              self._update_normalizers(full_batch)
          frozen = [p for p in self._frozen_in_warmup if p.requires_grad] if warmup else []
          for p in frozen:                 # no gradient: Adam updates neither them nor their state
              p.requires_grad_(False)
  ```
- wrap the `for _epoch in range(cfg.num_epochs):` loop in `try:` / `finally:` that restores `p.requires_grad_(True)` for every `p in frozen`, and call `self._loss_from_batch(_select_chunks(...), warmup=warmup)` inside it;
- replace `if self._kickstart is not None: self._kickstart.step()` with
  ```python
          if self._kickstart is not None and not warmup:
              self._kickstart.step()           # the decay starts after the warm-up
          if warmup:
              self._critic_warmup_done += 1
  ```
- before `return metrics`: `metrics["critic_warmup"] = 1.0 if warmup else 0.0`.

`state_dict`: add `"critic_warmup_done": int(self._critic_warmup_done),` (docstring keys list too). `load_state_dict`: add `self._critic_warmup_done = int(state.get("critic_warmup_done", 0))` with the comment `# absent in trainer states written before SP3: the warm-up restarts with the fresh optimizer`.

- [ ] **Step 5: Factory and validation**

`src/colosseum/learner/factory.py::build_algorithm`, before the `return`:

```python
    warmup = agent_config.init.critic_warmup_steps
    if warmup > 0:
        kwargs["critic_warmup_steps"] = warmup
```

`src/colosseum/core/validation.py`: add

```python
def _check_critic_warmup(where: str, agent_config: ColosseumConfig, model: Any) -> None:
    """``init.critic_warmup_steps > 0`` needs a value loss, ``PolicyModel.value_parameters()`` and an algorithm
    class that takes ``critic_warmup_steps``."""
    import inspect

    from colosseum.core.registry import import_class

    steps = agent_config.init.critic_warmup_steps
    if steps <= 0:
        return
    if not agent_config.algorithm.value_loss_coeff > 0:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps} trains only the value loss; "
                          f"algorithm.value_loss_coeff must be > 0")
    try:
        params = list(model.value_parameters())
    except NotImplementedError as e:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}: {e}") from e
    if not params:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}: value_parameters() returned no parameters")
    path = agent_config.algorithm.algorithm_class
    if "critic_warmup_steps" not in inspect.signature(import_class(path)).parameters:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}, but algorithm.algorithm_class {path!r} takes "
                          f"no critic_warmup_steps argument; use colosseum.algorithms.appo.APPO (or a subclass that "
                          f"accepts it) or set critic_warmup_steps: 0")
```

and call it in `validate_config`'s per-agent model loop right after `_check_model(model, role_specs[aid], sample, where)`: `_check_critic_warmup(where, acfg, model)`. Docstring bullet: "- the critic warm-up's requirements (``init.critic_warmup_steps``)".

- [ ] **Step 6: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_critic_warmup.py tests/unit/test_appo_v2.py tests/unit/test_kickstart_v2.py tests/contract/test_learner_reproduces_worker.py tests/unit/test_policy_model_v2.py -q`
Expected: all pass (the contract tests still reproduce the worker's log-probs: the warm-up is off by default).

- [ ] **Step 7: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/colosseum/networks/model.py src/colosseum/networks/composed.py src/colosseum/algorithms/appo.py \
    src/colosseum/learner/factory.py src/colosseum/core/validation.py tests/game_helpers.py \
    tests/unit/test_sp3_critic_warmup.py tests/unit/test_appo_v2.py
git commit -m "feat: critic warm-up that trains only the value path and keeps the workers' policy bit-identical"
git push origin sp3-league
```

---

### Task T4.4: Neural teacher per agent (any architecture, state-layout rule)

Spec block 6 «Нейросетевой учитель»: the teacher of each agent is a frozen agent or a path, built with its own architecture (checkpoint dir: from `meta.json`; `.pt`: the student's); the per-decider KL on ACT slots stays as in SP2; the teacher must play every role of its student; `validate` computes the teacher/student KL on a synthetic batch (incompatible distributions → ConfigError); a stateless teacher works with any student (`state0 = None`), a recurrent teacher needs the student's state layout (`state0` from the chunk, an approximation), a recurrent teacher with another layout is a ConfigError. The rule `_check_teacher_state_layout` in `appo.py`, which rejects a stateless teacher of a recurrent student today, changes accordingly (it moves to `bc/kickstart.py` as `check_teacher_state_layout`).

**Files:**
- Modify: `src/colosseum/bc/kickstart.py` (stateless teacher gets `state0=None`; `check_teacher_state_layout`)
- Modify: `src/colosseum/algorithms/appo.py` (uses `check_teacher_state_layout`; `_check_teacher_state_layout` removed)
- Modify: `src/colosseum/learner/factory.py` (roles check in `resolve_teacher`; `check_teacher_compat`)
- Modify: `src/colosseum/core/validation.py` (`_check_kickstart_teachers` adds the layout rule and the KL check)
- Modify (tests whose SP2 assumptions change): `tests/unit/test_kickstart_v2.py::test_appo_adds_kickstart_with_the_entropy_reduction_and_checks_the_state_layout` (a stateless teacher of an LSTM student is allowed now; the rejected case becomes a GRU teacher)
- Test: `tests/unit/test_sp3_neural_teacher.py`

**Interfaces:**
- Consumes: `resolve_teacher`, `build_teacher_model`, `build_algorithm` (T4.1); `write_ckpt_dir` (T4.1 kit); `resolve_player_roles` (T1.2).
- Produces (contract): `KickstartLoss` (unchanged signature); the state-layout rule.
- Produces (additions, amendment A10): `colosseum.bc.kickstart.check_teacher_state_layout(student, teacher) -> None` (ValueError); `colosseum.learner.factory.check_teacher_compat(student, teacher, where) -> None` (ConfigError).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_neural_teacher.py`:

```python
"""Neural kickstart teacher per agent, of any architecture (spec block 6, T4.4)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, validate_config
from colosseum.core.specs import ActionSpec
from colosseum.envs.game import RoleSpec
from colosseum.learner.factory import build_algorithm, check_teacher_compat, resolve_teacher
from colosseum.networks.dist import make_distribution
from colosseum.networks.model import UnrollOutput
from game_helpers import (
    GameTestModel,
    agent_role_of,
    make_test_config,
    make_test_model,
    synthetic_chunk,
    write_ckpt_dir,
)

ROLE = RoleSpec(gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32), gymnasium.spaces.Discrete(3))
SPEC = ActionSpec.from_space(ROLE.action_space)


class WrongDistModel(GameTestModel):
    """``unroll`` returns a distribution over 5 actions, whatever the role's action space."""

    def unroll(self, obs, state0, reset_after, action_mask=None, global_state=None, with_value=True):
        out = super().unroll(obs, state0, reset_after, action_mask, global_state=global_state, with_value=with_value)
        five = ActionSpec.from_space(gymnasium.spaces.Discrete(5))
        return UnrollOutput(dist=make_distribution(five, torch.zeros(reset_after.numel(), 5)), value=out.value)


def frozen_teacher_config(tmp_path, *, teacher_networks: dict, student_networks: dict | None = None):
    """turns config whose agent_0 learns from frozen agent 'teacher' (a checkpoint of ``teacher_networks``)."""
    base = make_test_config("turns")
    ckpt = write_ckpt_dir(tmp_path / "teacher", base, networks=teacher_networks, seed=2)
    student = {"kickstart": {"teacher": "teacher"}}
    if student_networks is not None:
        student["networks"] = student_networks
    return make_test_config("turns", agents={"agent_0": student, "teacher": {"kind": "frozen", "path": str(ckpt)}},
                            matchmaking={"anchors": []})


def test_a_stateless_teacher_works_with_a_recurrent_student():
    torch.manual_seed(0)
    student, teacher = make_test_model(ROLE, core="lstm"), make_test_model(ROLE, core="none")
    seen = []
    original = teacher.unroll

    def spy(obs, state0, *args, **kwargs):
        seen.append(state0)
        return original(obs, state0, *args, **kwargs)

    teacher.unroll = spy
    algo = APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher, initial_lambda=0.5))
    metrics = algo.train_step([synthetic_chunk(student, ROLE, "AATP", seed=s) for s in range(2)])
    assert metrics["kickstart_loss"] > 0 and seen and all(state0 is None for state0 in seen)


def test_a_recurrent_teacher_with_the_students_layout_gets_the_chunk_state():
    torch.manual_seed(0)
    student, teacher = make_test_model(ROLE, core="gru"), make_test_model(ROLE, core="gru")
    algo = APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher))
    assert algo.train_step([synthetic_chunk(student, ROLE, "AATP", seed=1)])["kickstart_loss"] > 0


@pytest.mark.parametrize("student_core, teacher_core", [("lstm", "gru"), ("none", "lstm"), ("gru", "lstm")])
def test_a_recurrent_teacher_needs_the_students_state_layout(student_core, teacher_core):
    student, teacher = make_test_model(ROLE, core=student_core), make_test_model(ROLE, core=teacher_core)
    with pytest.raises(ValueError, match="state layout"):
        APPO(student, AlgorithmConfig(), SPEC, kickstart=KickstartLoss(teacher))
    with pytest.raises(ConfigError, match="agent 'a'.*state layout"):
        check_teacher_compat(student, teacher, "agent 'a'")


def test_a_teacher_of_another_architecture_is_built_from_its_meta(tmp_path):
    cfg = frozen_teacher_config(tmp_path, teacher_networks={"model_class": "game_helpers.GameTestModel",
                                                            "kwargs": {"core": "none", "hidden": 32}})
    validate_config(cfg)
    spec = env_spec(cfg)
    teacher = resolve_teacher(cfg, "agent_0", spec)
    _roles, role = agent_role_of(cfg, "agent_0")
    algo = build_algorithm(cfg.get_agent_config("agent_0"), role, spec, device="cpu", teacher=teacher)
    assert algo._kickstart.teacher.inner.encoder.fc.out_features == 32
    assert algo.model.inner.encoder.fc.out_features == 16
    chunks = [synthetic_chunk(algo.model, role, "AAAB", seed=s, agent_id="agent_0") for s in range(2)]
    assert algo.train_step(chunks)["kickstart_loss"] > 0


def test_teachers_are_per_agent_and_of_their_own_architecture(tmp_path):
    base = make_test_config("turns")
    big = write_ckpt_dir(tmp_path / "big", base, networks={"model_class": "game_helpers.GameTestModel",
                                                           "kwargs": {"core": "none", "hidden": 32}})
    small = write_ckpt_dir(tmp_path / "small", base)
    cfg = make_test_config("turns", agents={"a": {"kickstart": {"teacher": "big"}},
                                            "b": {"kickstart": {"teacher": str(small)}},
                                            "big": {"kind": "frozen", "path": str(big)}},
                           matchmaking={"anchors": []})
    spec = env_spec(cfg)
    hidden = {}
    for agent_id in ("a", "b"):
        algo = build_algorithm(cfg.get_agent_config(agent_id), agent_role_of(cfg, agent_id)[1], spec, device="cpu",
                               teacher=resolve_teacher(cfg, agent_id, spec))
        hidden[agent_id] = algo._kickstart.teacher.inner.encoder.fc.out_features
    assert hidden == {"a": 32, "b": 16}


def test_a_teacher_must_play_every_role_of_its_student(tmp_path):
    base = make_test_config("asymmetric")
    ckpt = write_ckpt_dir(tmp_path / "old_hunter", base, agent_id="hunter")
    cfg = make_test_config("asymmetric", agents={
        "hunter": {"roles": ["hunter"]},
        "prey": {"roles": ["prey"], "kickstart": {"teacher": "old_hunter"}},
        "old_hunter": {"kind": "frozen", "path": str(ckpt)},
    })
    with pytest.raises(ConfigError, match="every role"):
        resolve_teacher(cfg, "prey", env_spec(cfg))


@pytest.mark.parametrize("teacher_networks, student_networks, message", [
    ({"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "gru", "hidden": 16}},
     {"kwargs": {"core": "lstm", "hidden": 16}}, "state layout"),
    ({"model_class": "test_sp3_neural_teacher.WrongDistModel", "kwargs": {"core": "none", "hidden": 16}},
     None, "KL"),
])
def test_validate_rejects_teachers_the_student_cannot_learn_from(tmp_path, teacher_networks, student_networks,
                                                                  message):
    cfg = frozen_teacher_config(tmp_path, teacher_networks=teacher_networks, student_networks=student_networks)
    with pytest.raises(ConfigError, match=message):
        validate_config(cfg)
```

In `tests/unit/test_kickstart_v2.py::test_appo_adds_kickstart_with_the_entropy_reduction_and_checks_the_state_layout`, replace the final `with pytest.raises(...)` block with:

```python
    with pytest.raises(ValueError, match="state layout"):          # a recurrent teacher needs the student's layout
        APPO(make_test_model(UNITS, core="lstm"), AlgorithmConfig(), spec,
             kickstart=KickstartLoss(make_test_model(UNITS, core="gru")))
    APPO(make_test_model(UNITS, core="lstm"), AlgorithmConfig(), spec,   # a stateless teacher: allowed (SP3)
         kickstart=KickstartLoss(make_test_model(UNITS, core="none")))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_neural_teacher.py tests/unit/test_kickstart_v2.py -q`
Expected: `ImportError: cannot import name 'check_teacher_compat'`; with that stubbed, the stateless-teacher cases fail with `ValueError: kickstart teacher must share the student's state layout`.

- [ ] **Step 3: `bc/kickstart.py`**

Add `from colosseum.networks.state import tree_leaves` and the rule:

```python
def check_teacher_state_layout(student: PolicyModel, teacher: PolicyModel) -> None:
    """ValueError unless ``teacher`` can be unrolled on ``student``'s chunks.

    A stateless teacher works with any student (it is unrolled with ``state0=None``). A recurrent teacher
    reuses the chunks' initial states, which the student's behavior policy recorded, so it needs the
    student's exact state layout (exact while teacher == student, an approximation afterwards).
    """
    teacher_shapes = [tuple(t.shape) for t in tree_leaves(teacher.initial_state(1))]
    if not teacher_shapes:
        return
    student_shapes = [tuple(t.shape) for t in tree_leaves(student.initial_state(1))]
    if student_shapes != teacher_shapes:
        raise ValueError(
            "a recurrent kickstart teacher must share the student's state layout (student state leaves "
            f"{student_shapes}, teacher {teacher_shapes}); use a stateless teacher or one with the student's core"
        )
```

In `KickstartLoss.__init__` (after freezing the teacher): `self._teacher_stateful = bool(teacher.is_stateful)`. In `compute`, the teacher unroll becomes:

```python
        with torch.no_grad():
            teacher_state0 = state0 if self._teacher_stateful else None   # a stateless teacher needs no state
            teacher_dist = self._teacher.unroll(obs, teacher_state0, reset_after.bool(), action_mask,
                                                with_value=False).dist
```

Module docstring: replace the last paragraph by "Teachers are per agent (SP3) and may have any architecture. A stateless teacher is unrolled with ``state0=None``; a recurrent teacher must share the student's state layout (``check_teacher_state_layout``) and gets the chunk's ``initial_state`` (recorded by the student's behavior policy): exact when teacher == student, an approximation after." and the `compute` argument doc of `state0`: "the student's chunk initial state (leaves ``[B, ...]``); passed to a recurrent teacher only".

- [ ] **Step 4: APPO uses the rule**

In `src/colosseum/algorithms/appo.py`: delete `_check_teacher_state_layout`; import `check_teacher_state_layout` from `colosseum.bc.kickstart` (next to `KickstartLoss`); in `__init__` replace `_check_teacher_state_layout(self._model, kickstart.teacher)` with `check_teacher_state_layout(self._model, kickstart.teacher)`. Remove `tree_leaves` from the `colosseum.networks.state` import if ruff reports it unused.

- [ ] **Step 5: Factory: roles check and `check_teacher_compat`**

In `resolve_teacher`, right before the final `return TeacherSpec(...)`:

```python
    from colosseum.players.registry import resolve_player_roles

    student_roles = resolve_player_roles(config, spec)[agent_id]
    missing = [r for r in student_roles if r not in frozen.roles]
    if missing:
        raise ConfigError(f"{where}: the teacher plays roles {list(frozen.roles)}; a teacher must play every role of "
                          f"its student ({list(student_roles)}), missing {missing}")
```

and add:

```python
def check_teacher_compat(student: PolicyModel, teacher: PolicyModel, where: str) -> None:
    """ConfigError unless ``teacher`` can be unrolled on ``student``'s chunks (``check_teacher_state_layout``)."""
    from colosseum.bc.kickstart import check_teacher_state_layout

    try:
        check_teacher_state_layout(student, teacher)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
```

- [ ] **Step 6: Validation: layout rule and KL on a synthetic batch**

In `src/colosseum/core/validation.py`, replace `_check_kickstart_teachers` (T4.1) with:

```python
def _check_kickstart_teachers(config: ColosseumConfig, spec: GameSpec, agent_configs: dict[str, ColosseumConfig],
                              role_specs: dict[str, RoleSpec], samples: dict[str, tuple | None]) -> None:
    """Every trainable agent's kickstart teacher resolves (roles included); a neural one builds with its
    weights, can be unrolled on the student's chunks (state-layout rule), and the KL between teacher and
    student is finite on a synthetic batch of the role's observations."""
    from colosseum.learner.factory import build_teacher_model, check_teacher_compat, resolve_teacher

    for aid, acfg in agent_configs.items():
        teacher = resolve_teacher(config, aid, spec)
        if teacher is None or teacher.kind != "neural":
            continue
        where = f"agent {aid!r}: kickstart teacher {teacher.source}"
        teacher_model = build_teacher_model(acfg, teacher, spec)
        student = build_model(acfg, role_specs[aid])
        check_teacher_compat(student, teacher_model, where)
        _check_teacher_kl(student, teacher_model, role_specs[aid], samples.get(aid), where, teacher.kl)


def _check_teacher_kl(student: Any, teacher: Any, role: RoleSpec, sample: tuple | None, where: str,
                      direction: str) -> None:
    """The kickstart KL of ``teacher`` vs ``student`` on a synthetic [S=2, B=2] batch of the role's observations."""
    from colosseum.bc.kickstart import KickstartLoss

    obs_spec = ObsSpec.from_space(role.observation_space)
    action_spec = ActionSpec.from_space(role.action_space)
    obs1 = _as_spec(obs_spec, sample[0]) if sample is not None else obs_spec.allocate(())
    mask1 = None
    if action_spec.has_masks:
        mask1 = sample[1] if sample is not None and sample[1] is not None else action_spec.full_mask(())
    s, b = 2, 2

    def seq(value: Any) -> Any:
        return None if value is None else tree_to_torch(tree_stack([tree_stack([value] * b)] * s))

    obs_seq, mask_seq = seq(obs1), seq(mask1)
    reset_after = torch.zeros(s, b, dtype=torch.bool)
    student.eval()
    try:
        with torch.no_grad():
            state0 = student.initial_state(b)
            student_dist = student.unroll(obs_seq, state0, reset_after, mask_seq, with_value=False).dist
            loss = KickstartLoss(teacher, initial_lambda=1.0, direction=direction).compute(
                student_dist=student_dist, obs=obs_seq, reset_after=reset_after, state0=state0, action_mask=mask_seq,
                actions=student_dist.sample(), is_act=torch.ones(s, b, dtype=torch.bool), reduction="sum")
    except Exception as e:  # noqa: BLE001 - any failure means an incompatible teacher
        raise ConfigError(f"{where}: the kickstart KL between teacher and student failed ({type(e).__name__}: {e}); "
                          f"the teacher's action distribution must match the student's") from e
    if not torch.isfinite(loss):
        raise ConfigError(f"{where}: the kickstart KL between teacher and student is not finite on a synthetic batch")
```

and the call in `validate_config` becomes `_check_kickstart_teachers(config, spec, agent_configs, role_specs, agent_samples)`, where `agent_samples` collects the per-agent `sample` the model loop already computes (in that loop add `agent_samples[aid] = sample` with `agent_samples: dict[str, tuple | None] = {}` before it).

- [ ] **Step 7: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_neural_teacher.py tests/unit/test_kickstart_v2.py tests/unit/test_sp3_learner_factory.py tests/unit/test_sp2_learner_entry.py tests/unit/test_sp2_validate.py tests/contract/test_learner_reproduces_worker.py -q`
Expected: all pass.

- [ ] **Step 8: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/colosseum/bc/kickstart.py src/colosseum/algorithms/appo.py src/colosseum/learner/factory.py \
    src/colosseum/core/validation.py tests/unit/test_sp3_neural_teacher.py tests/unit/test_kickstart_v2.py
git commit -m "feat: per-agent neural kickstart teachers of any architecture; stateless teachers for recurrent students"
git push origin sp3-league
```

---

### Task T4.5: Scripted teacher (DAgger): chunk labels, worker teachers, loss

Spec block 6 «Скриптовый учитель (DAgger)»: the teacher plays every role of its student (`validate`); teacher instances are separate from the bots that sit at seats, one per (student agent, env, seat), reset at the start of every episode in which the student collects at that seat; on every decision of a collecting seat of the student the worker asks that seat's teacher with the same `obs`, `mask`, `info`; the teacher's action goes into the ACT slot (`teacher_action` in the action-tree layout + `has_teacher` per slot, never on BOOT/PAD); the learner adds `lambda * mean(-log pi(a_teacher))` over ACT slots with `has_teacher` (one decider: the joint log-prob; `K > 1`: the mean `unit_log_prob` over valid deciders, as in BC); chunks without labels add 0 and stay out of the denominator; an illegal teacher action is a `PlayerError` with the block-2 context; the flag "teacher active" reaches the worker with the weights, and at lambda 0 the worker stops asking.

**Files:**
- Modify: `src/colosseum/core/types.py` (`TrajectoryChunk.teacher_action/has_teacher`, payload, `_apply`, `validate_slot_structure`; `WeightPayload.teacher_active`, `from_model(..., teacher_active=False)`)
- Modify: `src/colosseum/transport/serialization.py` (`validate_chunk_payload` accepts and checks the two fields)
- Modify: `src/colosseum/worker/buffers.py` (`BufferSpec.teacher`, `write_act(..., teacher_action=None)`)
- Modify: `src/colosseum/worker/rollout_loop.py` (`teachers`, per-env episode bookkeeping, `on_episode_start`, labels, flag)
- Modify: `src/colosseum/worker/rollout_worker.py` (`teachers` parameter)
- Modify: `src/colosseum/worker/match_runner.py` (only if T1.3 left `ActRecord.info` unset for neural seats; amendment A1)
- Modify: `src/colosseum/bc/kickstart.py` (`teacher=None`, `compute_labels`)
- Modify: `src/colosseum/algorithms/base.py` (`teacher_active`), `src/colosseum/algorithms/appo.py` (labels in the batch and the loss, `teacher_active`)
- Modify: `src/colosseum/learner/learner.py` (`_push_weights` sends the flag)
- Modify: `src/colosseum/learner/factory.py` (scripted `TeacherSpec`; labels-only `KickstartLoss`)
- Modify: `src/colosseum/launcher.py` (`_worker_main(..., teachers=None)`, worker kwargs)
- Modify (test kit): `tests/game_helpers.py` — add `LabelBot`
- Modify (tests of T4.1 whose assumption changes): `tests/unit/test_sp3_learner_factory.py::test_a_scripted_teacher_is_not_supported_yet` is replaced (Step 1)
- Test: `tests/unit/test_sp3_dagger_chunks.py`, `tests/contract/test_sp3_dagger_worker.py`, `tests/unit/test_sp3_dagger_loss.py`, `tests/integration/test_sp3_dagger_run.py`

**Interfaces:**
- Consumes: `ScriptedBot`, `bot_rng`, `check_bot_action` (`colosseum.players.scripted`), `BotSpec`, `make_bot`, `resolve_player_roles` (`colosseum.players.registry`), `PlayerError` (T1.2); `ActRecord.info`, the optional observer hook `on_episode_start(env, layout, episode_seed)` (T1.3); `RolloutLoop(..., fixed_players=...)` (T1.4); `resolve_teacher`, `build_kickstart`, `TeacherSpec` (T4.1, T4.4).
- Produces (contract): `TrajectoryChunk.teacher_action: Tree | None = None`, `TrajectoryChunk.has_teacher: Tensor | None = None` (payload/`_apply` carry them, `validate_slot_structure` rejects `has_teacher` on a non-ACT slot); `WeightPayload.teacher_active: bool = False`; `RolloutLoop(..., teachers: Mapping[str, BotSpec] | None = None)`, `rollout_worker_process(..., teachers=None)`; `KickstartLoss.compute_labels(student_dist, teacher_actions, has_teacher, is_act, reduction) -> Tensor`.
- Produces (additions, amendments A8/A9): `KickstartLoss(teacher: PolicyModel | None, ...)` (None = scripted teacher); `BaseAlgorithm.teacher_active` (default False) and `APPO.teacher_active`; `WeightPayload.from_model(agent_id, policy_version, model, teacher_active=False)`; `BufferSpec.teacher: bool = False`; `RolloutBuffer.write_act(..., reward, teacher_action=None)`; train metric `kickstart_label_frac` (labeled share of ACT slots, scripted teachers only); `_worker_main(..., teachers=None)`.

- [ ] **Step 1: Test kit and the failing tests**

Append to `tests/game_helpers.py` (import `ScriptedBot` from `colosseum.players` at the top if T1.2 did not):

```python
# ---------------------------------------------------------------------------
# SP3 (T4.5): a scripted kickstart teacher for DAgger tests
# ---------------------------------------------------------------------------


class LabelBot(ScriptedBot):
    """Always plays ``action`` (a discrete index); ``fail="raise"`` raises in ``act``.

    Every call is appended to the class-level ``LabelBot.calls`` (tests clear it): ``("init", id)``,
    ``("reset", id, role, seat, layout, first rng draw)``, ``("act", id, obs, info)``.
    """

    calls: list[tuple] = []

    def __init__(self, action: int = 1, fail: str = "") -> None:
        super().__init__()
        self.action, self.fail = int(action), fail
        LabelBot.calls.append(("init", id(self)))

    def reset(self, *, role, seat, layout, rng) -> None:
        LabelBot.calls.append(("reset", id(self), role, seat, layout, float(rng.random())))

    def act(self, obs, mask, info):
        LabelBot.calls.append(("act", id(self), np.array(obs, copy=True), info))
        if self.fail == "raise":
            raise RuntimeError("teacher bot failure")
        return np.int64(self.action)
```

In `tests/unit/test_sp3_learner_factory.py` replace `test_a_scripted_teacher_is_not_supported_yet` with:

```python
def test_a_scripted_agent_is_a_dagger_teacher():
    from colosseum.players.registry import BotSpec

    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "bot", "lambda": 0.7}}, "bot": BOT})
    teacher = resolve_teacher(cfg, "agent_0", env_spec(cfg))
    assert teacher.kind == "scripted" and teacher.frozen is None and teacher.lambda_ == 0.7
    assert teacher.bot == BotSpec("colosseum.players.RandomBot", {})
    algo = _build(cfg, "agent_0", teacher)
    assert algo._kickstart.teacher is None and algo.teacher_active


def test_a_scripted_teacher_must_play_every_role_of_its_student():
    cfg = make_test_config("asymmetric", agents={
        "hunter": {"roles": ["hunter"]},
        "prey": {"roles": ["prey"], "kickstart": {"teacher": "chaser"}},
        "chaser": {**BOT, "roles": ["hunter"]},
    })
    with pytest.raises(ConfigError, match="every role"):
        resolve_teacher(cfg, "prey", env_spec(cfg))
```

and add `test_launcher_passes_scripted_teachers_to_the_workers` (same `FakeProcess` pattern as `test_launcher_passes_numpy_teacher_specs_and_the_spec_to_learners`, recording the worker's kwargs instead of stopping at it):

```python
def test_launcher_passes_scripted_teachers_to_the_workers(tmp_path, monkeypatch, restore_global_rng):
    import colosseum.launcher as launcher_module
    from colosseum.players.registry import BotSpec

    cfg = make_test_config("turns", agents={"agent_0": {"kickstart": {"teacher": "bot"}}, "bot": BOT},
                           matchmaking={"anchors": []})
    started: dict[str, dict] = {}

    class _Stop(Exception):
        pass

    class FakeProcess:
        exitcode = 0

        def __init__(self, target, kwargs, name=None, daemon=None):
            self.target, self.kwargs = target, kwargs

        def start(self):
            role = "learner" if self.target is launcher_module._learner_target else "worker"
            started[role] = self.kwargs
            if role == "worker":
                raise _Stop

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg, make_test_run_dir(cfg, tmp_path)).launch()
    assert started["worker"]["teachers"] == {"agent_0": BotSpec("colosseum.players.RandomBot", {})}
    assert started["learner"]["teacher"].kind == "scripted"
    assert_no_tensors(started["worker"]["teachers"], "worker teachers")
```

Create `tests/unit/test_sp3_dagger_chunks.py`:

```python
"""DAgger labels in chunk v2 (spec block 6): buffers write them on ACT slots only, payloads and gRPC
validation carry them with their dtypes, the slot rules reject a label elsewhere; WeightPayload's
teacher flag (T4.5)."""
from __future__ import annotations

import pickle

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.ipc import assert_no_tensors
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import SLOT_BOOT, TrajectoryChunk, WeightPayload, validate_slot_structure
from colosseum.envs.game import RoleSpec
from colosseum.transport.serialization import validate_chunk_payload
from colosseum.worker.buffers import BufferSpec, RolloutBuffer
from game_helpers import chunk_v2_payload, make_test_model

OBS = gymnasium.spaces.Box(-1.0, 1.0, (3,), np.float32)
ACT = gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(4),
                             "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)})


def buffer(teacher: bool, slots: int = 4) -> RolloutBuffer:
    return RolloutBuffer(slots, BufferSpec(ObsSpec.from_space(OBS), ActionSpec.from_space(ACT), None, teacher=teacher))


def act(buf: RolloutBuffer, label=None) -> None:
    action = {"move": np.int64(1), "aim": np.zeros(2, np.float32)}
    buf.write_act(np.zeros(3, np.float32), None, None, action, -1.0, None, 0.0, teacher_action=label)


def test_a_teacher_buffer_labels_act_slots_only_and_keeps_dtypes():
    buf = buffer(teacher=True)
    buf.begin(None, 0)
    act(buf, {"move": np.int64(3), "aim": np.array([0.5, -0.5], np.float32)})
    act(buf)                                       # the teacher was not asked (e.g. its flag is off)
    buf.mark_terminal()
    buf.write_pad()
    buf.write_pad()
    chunk = buf.build_chunk("a")
    assert chunk.has_teacher.tolist() == [True, False, False, False]
    assert chunk.teacher_action["move"].dtype == torch.int64 and chunk.teacher_action["aim"].dtype == torch.float32
    assert chunk.teacher_action["move"].tolist() == [3, 0, 0, 0]
    assert chunk.teacher_action["aim"][0].tolist() == [0.5, -0.5] and not chunk.teacher_action["aim"][1:].any()
    validate_slot_structure(chunk)


def test_buffers_without_a_teacher_have_no_label_fields():
    buf = buffer(teacher=False, slots=2)
    buf.begin(None, 0)
    with pytest.raises(ValueError, match="teacher"):
        act(buf, {"move": np.int64(3), "aim": np.zeros(2, np.float32)})
    act(buf)
    buf.write_boot(np.zeros(3, np.float32), None, reset_after=False)
    chunk = buf.build_chunk("a")
    assert chunk.teacher_action is None and chunk.has_teacher is None


def test_payload_round_trip_carries_the_labels():
    buf = buffer(teacher=True, slots=2)
    buf.begin(None, 3)
    act(buf, {"move": np.int64(2), "aim": np.array([0.25, 0.0], np.float32)})
    buf.write_boot(np.zeros(3, np.float32), None, reset_after=False)
    payload = buf.build_chunk("a").to_payload()
    assert_no_tensors(payload)
    payload = pickle.loads(pickle.dumps(payload))
    assert payload["teacher_action"]["move"].dtype == np.int64 and payload["has_teacher"].dtype == np.bool_
    validate_chunk_payload(payload)
    back = TrajectoryChunk.from_payload(payload)
    assert back.has_teacher.tolist() == [True, False] and back.teacher_action["move"].tolist() == [2, 0]
    moved = back.to("cpu")
    assert torch.equal(moved.has_teacher, back.has_teacher)


def test_payloads_without_the_fields_still_load():
    chunk = TrajectoryChunk.from_payload(chunk_v2_payload(pattern="AAB"))
    assert chunk.teacher_action is None and chunk.has_teacher is None
    validate_chunk_payload(chunk.to_payload())


def test_a_label_on_a_non_act_slot_breaks_the_slot_rules():
    payload = chunk_v2_payload(pattern="AAB")
    payload["teacher_action"] = np.zeros(3, np.int64)
    payload["has_teacher"] = np.array([True, False, True])
    assert payload["kind"][2] == SLOT_BOOT
    with pytest.raises(ValueError, match="teacher label on a non-ACT slot"):
        validate_slot_structure(TrajectoryChunk.from_payload(payload))
    with pytest.raises(ValueError, match="teacher label on a non-ACT slot"):
        validate_chunk_payload(payload)
    half = chunk_v2_payload(pattern="AAB")
    half["has_teacher"] = np.array([True, False, False])
    with pytest.raises(ValueError, match="come together"):
        validate_chunk_payload(half)


def test_weight_payload_carries_the_teacher_flag():
    model = make_test_model(RoleSpec(OBS, ACT))
    assert WeightPayload("a", 0).teacher_active is False
    assert WeightPayload.from_model("a", 1, model).teacher_active is False
    payload = WeightPayload.from_model("a", 1, model, teacher_active=True)
    assert payload.teacher_active is True
    assert_no_tensors(payload)
```

Create `tests/contract/test_sp3_dagger_worker.py`:

```python
"""Scripted kickstart teacher on the worker (DAgger, spec block 6): labels in the ACT slots of the
collecting seats, one teacher instance per (student, env, seat), resets at episode starts, infos,
legality, the teacher flag, and the learner still reproduces the worker's log-probs (T4.5)."""
from __future__ import annotations

from collections import defaultdict

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.errors import PlayerError
from colosseum.core.types import SLOT_ACT, SeatAssignment, TrajectoryChunk, WeightPayload
from colosseum.envs.game import GameSpec, MultiAgentEnv, RoleSpec, StepResult
from colosseum.players.registry import BotSpec
from game_harness import Collected, GameFactory, learner_eval, lineup, make_loop, run_steps, run_until_chunks
from game_helpers import SCRIPT_OBS_SPACE, LabelBot, Tick, make_test_model

ROLE = RoleSpec(SCRIPT_OBS_SPACE, gymnasium.spaces.Discrete(3))
TEACHER = {"a": BotSpec("game_helpers.LabelBot", {"action": 2})}
THREE_MOVES = [Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(acting={0, 1}), Tick(over=True)]
SOLO = [Tick(acting={0}), Tick(acting={0}), Tick(acting={0}), Tick(over=True)]


@pytest.fixture(autouse=True)
def clear_label_bot():
    LabelBot.calls.clear()
    yield
    LabelBot.calls.clear()


def factory():
    torch.manual_seed(0)
    return make_test_model(ROLE)


def calls(kind: str) -> list[tuple]:
    return [c for c in LabelBot.calls if c[0] == kind]


class InfoGame(MultiAgentEnv):
    """Solo, 4 steps; ``infos[0] = {"t": t}``."""

    def __init__(self) -> None:
        self.spec = GameSpec.solo(SCRIPT_OBS_SPACE, gymnasium.spaces.Discrete(3))
        self.t = 0

    def _result(self, **kwargs) -> StepResult:
        return StepResult(acting={0}, obs={0: np.full(5, self.t, np.float32)}, infos={0: {"t": self.t}}, **kwargs)

    def reset(self, seed, layout):
        self.t = 0
        return self._result()

    def step(self, actions):
        self.t += 1
        if self.t >= 4:
            return StepResult(acting=set(), obs={}, rewards={0: 1.0}, episode_over=True)
        return self._result(rewards={0: 0.0})


def test_act_slots_of_collecting_seats_carry_the_teachers_action():
    loop, col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")],
                          chunk_length=8, teachers=TEACHER)
    for chunk in run_until_chunks(loop, col, 4):
        chunk = TrajectoryChunk.from_payload(chunk.to_payload())
        is_act = chunk.kind == SLOT_ACT
        assert torch.equal(chunk.has_teacher, is_act)
        assert (chunk.teacher_action[is_act] == 2).all() and (chunk.teacher_action[~is_act] == 0).all()
        assert chunk.teacher_action.dtype == chunk.actions.dtype
    loop.close()


def test_only_collecting_seats_ask_their_teacher():
    loop, _col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory},
                           [lineup("2p", "a", SeatAssignment("a", collect=False))], chunk_length=8, teachers=TEACHER)
    run_steps(loop, 12)
    acts = calls("act")
    assert acts and {int(obs[3]) for _kind, _id, obs, _info in acts} == {0}     # obs[3] is the seat
    loop.close()


def test_one_teacher_per_student_env_and_seat_reset_at_every_episode_start():
    loop, _col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")] * 2,
                           chunk_length=8, teachers=TEACHER)
    run_steps(loop, 9)                             # 3 episodes of 3 env steps in each env
    assert len(calls("init")) == 4                 # 2 envs x 2 seats
    episodes_by_bot: dict[int, set] = defaultdict(set)
    for _kind, bot, obs, _info in calls("act"):
        episodes_by_bot[bot].add((int(obs[0]), int(obs[1])))    # (env tag, episode)
    resets = calls("reset")
    for bot, episodes in episodes_by_bot.items():
        assert len(episodes) == 3 and len({e for e, _ in episodes}) == 1
        assert len([r for r in resets if r[1] == bot]) == 3     # once per episode it collected in
    assert {(r[2], r[4]) for r in resets} == {("player", "2p")} and {r[3] for r in resets} == {0, 1}
    loop.close()


def test_teacher_rng_is_reproducible_with_the_seed():
    def draws(seed: int) -> list[float]:
        LabelBot.calls.clear()
        loop, _col = make_loop(GameFactory((SOLO, 1)), {"a": factory}, [lineup("solo", "a")], chunk_length=8,
                               teachers=TEACHER, seed=seed)
        run_steps(loop, 8)
        loop.close()
        return [r[5] for r in calls("reset")]

    assert draws(3) == draws(3) and draws(3) != draws(4)


def test_the_teacher_sees_the_seats_infos():
    loop, _col = make_loop(InfoGame, {"a": factory}, [lineup("solo", "a")], chunk_length=8, teachers=TEACHER)
    run_steps(loop, 4)
    assert [info for _kind, _id, _obs, info in calls("act")] == [{"t": 0}, {"t": 1}, {"t": 2}, {"t": 3}]
    loop.close()


@pytest.mark.parametrize("bot, message", [
    (BotSpec("game_helpers.LabelBot", {"action": 2}), "kickstart teacher of agent 'a'"),
    (BotSpec("game_helpers.LabelBot", {"action": 1, "fail": "raise"}), "act raised RuntimeError: teacher bot failure"),
])
def test_an_illegal_action_or_a_failure_of_the_teacher_is_a_player_error(bot, message):
    masked = GameFactory((SOLO, 1), mask_fn=lambda k, t, seat: np.array([True, True, False]))
    loop, _col = make_loop(masked, {"a": factory}, [lineup("solo", "a")], chunk_length=8, teachers={"a": bot})
    with pytest.raises(PlayerError, match="worker 0, env 0, seat 0, episode step 0, layout solo") as info:
        run_steps(loop, 1)
    assert message in str(info.value)
    loop.close()


def test_the_teacher_flag_stops_the_queries():
    col = Collected()
    loop, col = make_loop(GameFactory((SOLO, 1)), {"a": factory}, [lineup("solo", "a")], chunk_length=4,
                          teachers=TEACHER, collected=col)
    run_steps(loop, 6)
    asked = len(calls("act"))
    assert asked == 6
    col.weights["a"] = [WeightPayload.from_model("a", 1, factory(), teacher_active=False)]
    run_steps(loop, 1)                              # this step still asks; the sync after it reads the flag
    run_steps(loop, 8)
    assert len(calls("act")) == asked + 1
    last = TrajectoryChunk.from_payload(col.chunks[-1].to_payload())
    assert last.policy_version == 1 and not last.has_teacher.any()
    loop.close()


def test_the_learner_reproduces_the_worker_with_labels_in_the_chunks():
    loop, col = make_loop(GameFactory((THREE_MOVES, 2)), {"a": factory}, [lineup("2p", "a", "a")],
                          chunk_length=8, teachers=TEACHER)
    chunks = run_until_chunks(loop, col, 3)
    view = learner_eval(factory(), chunks, ROLE)
    assert torch.allclose(view.log_probs[view.is_act], view.worker_log_probs[view.is_act], atol=1e-5)
    loop.close()
```

(`make_loop` passes extra keyword arguments such as `teachers` to `RolloutLoop`; `GameFactory(..., mask_fn=...)` passes `mask_fn` to every `TickGame`.)

Create `tests/unit/test_sp3_dagger_loss.py`:

```python
"""The DAgger loss (spec block 6): lambda * mean(-log pi(a_teacher)) over labeled ACT slots, joint for one
decider and mean over valid deciders with Units; unlabeled chunks add nothing; the teacher flag; a
student learns the teacher's action (T4.5)."""
from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.kickstart import KickstartLoss
from colosseum.bc.offline_bc import per_sample_nll
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_map, tree_stack
from colosseum.core.types import SLOT_ACT, TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.learner.learner import _push_weights
from colosseum.networks.state import cat_batch
from game_helpers import NumpyOnlyQueue, chunk_v2_payload, learner_role, make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
UNITS = RoleSpec(OBS, Units(4, gymnasium.spaces.Discrete(3)))
ROLE3 = learner_role(3)


def labeled(chunk: TrajectoryChunk, label_tree, every: int = 1) -> TrajectoryChunk:
    """``chunk`` with the teacher action ``label_tree`` (one row) on every ``every``-th ACT slot."""
    S = chunk.num_slots
    is_act = chunk.kind == SLOT_ACT
    picked = is_act & (torch.arange(S) % every == 0)
    rows = tree_map(lambda leaf: torch.as_tensor(leaf).expand(S, *torch.as_tensor(leaf).shape).clone(), label_tree)
    zeros = tree_map(torch.zeros_like, chunk.actions)
    actions = tree_map(lambda t, z: torch.where(picked.reshape(-1, *([1] * (t.dim() - 1))), t.to(z.dtype), z),
                       rows, zeros)
    payload = chunk.to_payload()
    payload["teacher_action"] = tree_map(lambda t: t.numpy(), actions)
    payload["has_teacher"] = picked.numpy()
    return TrajectoryChunk.from_payload(payload)


def student_view(model, chunks):
    S, B = chunks[0].num_slots, len(chunks)
    stack = lambda get: tree_stack([get(c) for c in chunks], axis=1)   # noqa: E731
    obs, reset, masks = stack(lambda c: c.obs), stack(lambda c: c.reset_after), stack(lambda c: c.action_masks)
    dist = model.unroll(obs, cat_batch([c.initial_state for c in chunks]), reset.bool(), masks, with_value=False).dist
    flat = lambda tree: tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), tree)   # noqa: E731
    is_act = stack(lambda c: c.kind) == SLOT_ACT
    return dist, flat(stack(lambda c: c.teacher_action)), stack(lambda c: c.has_teacher), is_act


def test_compute_labels_is_the_joint_nll_over_labeled_act_slots():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = [labeled(synthetic_chunk(model, ROLE3, p, seed=i), np.int64(2), every=2)
              for i, p in enumerate(["AAAATP", "AAAAAB"])]
    dist, actions, has, is_act = student_view(model, chunks)
    kick = KickstartLoss(None, initial_lambda=0.5)
    loss = kick.compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has, is_act=is_act,
                               reduction="sum")
    rows = (has & is_act).reshape(-1)
    expected = 0.5 * (-dist.log_prob(actions))[rows].mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-5) and float(loss) > 0


def test_with_units_it_is_the_mean_nll_over_valid_deciders_as_in_bc():
    torch.manual_seed(0)
    model = make_test_model(UNITS)
    label = np.array([1, 2, 0, 1], np.int64)
    chunks = [labeled(synthetic_chunk(model, UNITS, "AATP", seed=i, random_units=False), label) for i in range(2)]
    dist, actions, has, is_act = student_view(model, chunks)
    loss = KickstartLoss(None).compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has,
                                              is_act=is_act, reduction="mean_valid")
    nll, _valid = per_sample_nll(dist, actions, 4)
    rows = (has & is_act).reshape(-1)
    assert float(loss) == pytest.approx(float(nll[rows].mean()), rel=1e-5)


def test_no_label_gives_zero_and_lambda_zero_gives_zero():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunk = labeled(synthetic_chunk(model, ROLE3, "AAAB"), np.int64(2), every=100)   # only slot 0 labeled
    dist, actions, has, is_act = student_view(model, [chunk])
    none = torch.zeros_like(has)
    assert float(KickstartLoss(None).compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=none,
                                                    is_act=is_act, reduction="sum")) == 0.0
    spent = KickstartLoss(None, initial_lambda=1.0, decay_steps=1)
    spent.step()
    assert float(spent.compute_labels(student_dist=dist, teacher_actions=actions, has_teacher=has, is_act=is_act,
                                      reduction="sum")) == 0.0


def test_chunks_without_labels_add_nothing_and_stay_out_of_the_denominator():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    with_labels = labeled(synthetic_chunk(model, ROLE3, "AAAAAB", seed=1), np.int64(2))
    without = synthetic_chunk(model, ROLE3, "AAAAAB", seed=2)                           # e.g. a parked buffer
    algo = APPO(model, AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space),
                kickstart=KickstartLoss(None, initial_lambda=1.0))
    alone = algo.compute_loss([with_labels])
    mixed = algo.compute_loss([with_labels, without])
    assert float(mixed["kickstart_loss"]) == pytest.approx(float(alone["kickstart_loss"]), rel=1e-5)
    assert float(mixed["kickstart_label_frac"]) == pytest.approx(0.5)


def test_appo_reports_labels_and_switches_the_teacher_flag_off_at_lambda_zero():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = [labeled(synthetic_chunk(model, ROLE3, "AAAAAB", seed=s), np.int64(2)) for s in range(2)]
    algo = APPO(model, AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space),
                kickstart=KickstartLoss(None, initial_lambda=0.5, decay_steps=2))
    queue = NumpyOnlyQueue(maxsize=1)
    _push_weights(algo, "a", [queue])
    assert algo.teacher_active and queue.get_nowait().teacher_active is True
    metrics = algo.train_step(chunks)
    assert metrics["kickstart_loss"] > 0 and metrics["kickstart_label_frac"] == pytest.approx(1.0)
    algo.train_step(chunks)                                              # lambda decays to 0
    assert not algo.teacher_active
    _push_weights(algo, "a", [queue])
    assert queue.get_nowait().teacher_active is False
    plain = APPO(make_test_model(ROLE3), AlgorithmConfig(), ActionSpec.from_space(ROLE3.action_space))
    assert not plain.teacher_active and "kickstart_label_frac" not in plain.train_step(chunks)


def test_a_student_learns_the_teachers_action():
    torch.manual_seed(0)
    model = make_test_model(ROLE3)
    chunks = []
    for s in range(4):
        payload = chunk_v2_payload(pattern="A" * 15 + "B", version=s)
        payload["reward"][:] = 0.0
        chunks.append(labeled(TrajectoryChunk.from_payload(payload), np.int64(2)))
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2, lr_schedule="constant", entropy_coeff=0.0),
                ActionSpec.from_space(ROLE3.action_space), kickstart=KickstartLoss(None, initial_lambda=5.0,
                                                                                 decay_steps=10**6))
    for _ in range(60):
        algo.train_step(chunks)
    with torch.no_grad():
        obs = torch.randn(64, 4)
        probs = model.step(obs, None, torch.ones(64, 3, dtype=torch.bool)).dist.log_prob(torch.full((64,), 2)).exp()
    assert float(probs.mean()) > 0.9
```

Create `tests/integration/test_sp3_dagger_run.py`:

```python
"""`colosseum train` with a scripted kickstart teacher (DAgger): the worker labels the student's
decisions and the learner's label loss is in the train metrics (spec block 6, T4.5)."""
from __future__ import annotations

from pathlib import Path

from cli_runner import run_train
from game_helpers import write_test_config


def test_a_scripted_teacher_labels_the_students_decisions(tmp_path: Path):
    config = write_test_config(
        tmp_path / "dagger.yaml", "turns",
        agents={"agent_0": {}, "teacher_bot": {"kind": "scripted", "class": "game_helpers.LabelBot",
                                               "kwargs": {"action": 1}}},
        matchmaking={"anchors": []},
        kickstart={"teacher": "teacher_bot", "lambda": 1.0, "decay_steps": 100_000},
    )
    run = run_train(config, tmp_path, name="dagger")
    assert run.returncode == 0, run.stderr[-3000:]
    train = [r for r in run.records("train") if r["agent"] == "agent_0"]
    assert train and max(r["kickstart_loss"] for r in train) > 0
    assert max(r["kickstart_label_frac"] for r in train) > 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_dagger_chunks.py tests/contract/test_sp3_dagger_worker.py tests/unit/test_sp3_dagger_loss.py tests/unit/test_sp3_learner_factory.py -q`
Expected: failures: `TypeError: BufferSpec.__init__() got an unexpected keyword argument 'teacher'`, `TypeError: RolloutLoop.__init__() got an unexpected keyword argument 'teachers'`, `AttributeError: 'KickstartLoss' object has no attribute 'compute_labels'` (and `KickstartLoss(None)` failing on `None.eval()`), `ConfigError: ... scripted kickstart teachers are not supported yet`.

- [ ] **Step 3: Chunk and weight types; payload validation**

`src/colosseum/core/types.py`:
- `TrajectoryChunk`: add after `behavior_unit_logp`:
  ```python
      teacher_action: Tree | None = None   # [S, ...] scripted kickstart teacher's action (DAgger), action-tree layout
      has_teacher: torch.Tensor | None = None   # [S] bool: the slot carries a teacher label (ACT slots only)
  ```
  and to the class docstring: "``teacher_action`` / ``has_teacher`` (both or neither) exist only for an agent with a scripted kickstart teacher: the teacher's action on ACT slots it labelled, zeros elsewhere."
- `_apply`: add `teacher_action=_map_optional(fn, self.teacher_action), has_teacher=None if self.has_teacher is None else fn(self.has_teacher),`.
- `to_payload`: add
  ```python
              "teacher_action": None if self.teacher_action is None else tree_to_numpy(self.teacher_action),
              "has_teacher": None if self.has_teacher is None else tensor_to_numpy(self.has_teacher),
  ```
- `from_payload`: read `teacher, has = payload.get("teacher_action"), payload.get("has_teacher")` (payloads from before SP3 lack them) and pass `teacher_action=None if teacher is None else tree_to_torch(teacher), has_teacher=None if has is None else numpy_to_tensor(has)`.
- `_SLOT_RULES`: append `"a teacher label on a non-ACT slot"`; in `validate_slot_structure`, after the `kind.size == 0` check:
  ```python
      has_teacher = None if chunk.has_teacher is None else chunk.has_teacher.cpu().numpy().astype(bool)
      if has_teacher is not None and has_teacher.shape != kind.shape:
          raise ValueError(f"{where}): has_teacher must have the shape {kind.shape} of kind, got {has_teacher.shape}")
  ```
  and append to the `broken` tuple (last, matching the new last rule):
  ```python
          has_teacher & ~act if has_teacher is not None else np.zeros(kind.size, dtype=bool),   # a label off ACT
  ```
  Docstring: "A teacher label (``has_teacher``) only on ACT slots."
- `WeightPayload`: add the field `teacher_active: bool = False` (docstring: "True while a scripted kickstart teacher has lambda > 0: the worker keeps asking it (DAgger)") and the parameter `teacher_active: bool = False` to `from_model`, passed through.

`src/colosseum/transport/serialization.py::validate_chunk_payload`, after the `behavior_unit_logp` block:

```python
    teacher_action, has_teacher = payload.get("teacher_action"), payload.get("has_teacher")
    if (teacher_action is None) != (has_teacher is None):
        raise ValueError("chunk payload fields 'teacher_action' and 'has_teacher' come together")
    if teacher_action is not None:
        trees.append(("teacher_action", teacher_action))
        if not isinstance(has_teacher, np.ndarray) or has_teacher.ndim != 1 or has_teacher.shape[0] != num_slots:
            raise ValueError(f"chunk payload field 'has_teacher' must be a 1-D numpy array [S={num_slots}]")
```

- [ ] **Step 4: Buffers**

`src/colosseum/worker/buffers.py`:
- `BufferSpec`: add `teacher: bool = False` (docstring: "``teacher``: the agent has a scripted kickstart teacher; its chunks carry ``teacher_action`` / ``has_teacher``").
- `RolloutBuffer.__init__`: after `self._unit_logp = ...`:
  ```python
          self._teacher_actions = spec.action.allocate_actions((S,)) if spec.teacher else None
          self._has_teacher = np.zeros(S, dtype=np.bool_) if spec.teacher else None
  ```
- `write_act(..., reward: float, teacher_action: Tree | None = None)`: before the cursor checks:
  ```python
          if teacher_action is not None and self._teacher_actions is None:
              raise ValueError("teacher_action given, but this agent's buffers carry no teacher labels "
                               "(BufferSpec.teacher)")
  ```
  and before `self._cursor += 1`:
  ```python
          if self._teacher_actions is not None:
              put_row(self._teacher_actions, i, self._zero_action if teacher_action is None else teacher_action)
              self._has_teacher[i] = teacher_action is not None
  ```
  (docstring: "``teacher_action``: the scripted kickstart teacher's action for this decision, if it was asked").
- `_write_non_act`: before `self._cursor += 1`:
  ```python
          if self._teacher_actions is not None:
              put_row(self._teacher_actions, i, self._zero_action)
              self._has_teacher[i] = False
  ```
- `build_chunk`: add
  ```python
              teacher_action=None if self._teacher_actions is None else _to_torch(self._teacher_actions),
              has_teacher=None if self._has_teacher is None else torch.from_numpy(self._has_teacher.copy()),
  ```
- module docstring rule: "- a teacher label (``has_teacher``) only on ACT slots; BOOT and PAD carry zero labels".

- [ ] **Step 5: Worker: teachers in `RolloutLoop`**

`src/colosseum/worker/rollout_loop.py`:
- imports: `from typing import Any, Literal`; `from colosseum.core.errors import PlayerError`; `from colosseum.players.registry import BotSpec, make_bot`; `from colosseum.players.scripted import ScriptedBot, bot_rng, check_bot_action`.
- module docstring, new bullet: "- **Scripted kickstart teachers (DAgger).** For an agent in ``teachers`` every decision of a seat that collects for it is also given to a teacher instance of that (agent, env, seat) with the same ``obs``, ``mask`` and ``info``; its checked action goes into the ACT slot (``teacher_action``, ``has_teacher``). An instance is created at its first use and reset at the first decision of every episode in which the agent collects at that seat (lazily; ``on_episode_start`` records the layout and seed). The learner's ``WeightPayload.teacher_active`` switches the queries off (lambda 0)."
- a small record type next to `_SeatTrack`:
  ```python
  @dataclass
  class _TeacherBot:
      """A scripted kickstart teacher instance of one (student agent, env, seat)."""

      bot: ScriptedBot
      episode: int = -1          # the env's episode counter at its last reset
  ```
- `__init__`: new parameter `teachers: Mapping[str, BotSpec] | None = None` (after `fixed_players`); right after `self._weight_sync_interval = weight_sync_interval` (before any model is synced):
  ```python
          self._teacher_specs = dict(teachers or {})
          unknown = sorted(set(self._teacher_specs) - set(self._agent_ids))
          if unknown:
              raise ValueError(f"worker {worker_id}: kickstart teachers for unknown agents {unknown}")
          self._teacher_active = {aid: True for aid in self._teacher_specs}   # until the learner says otherwise
          self._teacher_bots: dict[tuple[str, int, int], _TeacherBot] = {}
          # Per env: the current episode's layout, seed, counter and step (teacher resets and error context).
          self._episode_layout: list[str | None] = [None] * num_envs
          self._episode_seed: list[int | None] = [None] * num_envs
          self._episode_count = [0] * num_envs
          self._episode_step = [0] * num_envs
  ```
  the `BufferSpec(...)` of every agent gets `teacher=aid in self._teacher_specs`, and `self._spec = spec` is set right after `spec = vec_env.spec`.
- `sync_weights`: inside `if payload is not None:` add
  ```python
                  if aid in self._teacher_active:
                      self._teacher_active[aid] = bool(payload.teacher_active)
  ```
- new observer hook (merge into T1.4's `on_episode_start` if it already exists):
  ```python
      def on_episode_start(self, env: int, layout: str, episode_seed: int | None) -> None:
          """MatchRunner hook after every env reset: the episode's layout and seed (teacher resets)."""
          self._episode_layout[env] = layout
          self._episode_seed[env] = episode_seed
          self._episode_count[env] += 1
          self._episode_step[env] = 0
  ```
- `on_rewards`: first statement `self._episode_step[env] += 1` (called once per env step).
- `on_act`: right after the `if track is None: return`, add `label = self._teacher_label(env, seat, track.agent_id, record)` and pass `teacher_action=label` to `buf.write_act(...)`.
- new helper:
  ```python
      def _teacher_label(self, env: int, seat: int, agent_id: str, record: ActRecord) -> Any:
          """The scripted kickstart teacher's checked action for a collecting seat's decision, or None (no teacher,
          or the learner switched it off: ``WeightPayload.teacher_active``). PlayerError with context on an
          illegal action or a failure inside the bot."""
          spec = self._teacher_specs.get(agent_id)
          if spec is None or not self._teacher_active[agent_id]:
              return None
          layout = self._episode_layout[env] or self._runner.lineup(env).layout
          role = self._spec.role_of(layout, seat)
          where = (f"worker {self.worker_id}, env {env}, seat {seat}, episode step {self._episode_step[env]}, "
                   f"layout {layout}: kickstart teacher of agent {agent_id!r}")
          key = (agent_id, env, seat)
          teacher = self._teacher_bots.get(key)
          if teacher is None:
              teacher = self._teacher_bots[key] = _TeacherBot(make_bot(spec, self._spec))
          if teacher.episode != self._episode_count[env]:
              try:
                  teacher.bot.reset(role=role, seat=seat, layout=layout,
                                    rng=bot_rng(self._episode_seed[env], seat, f"{agent_id}/teacher"))
              except Exception as e:  # noqa: BLE001 - user code
                  raise PlayerError(f"{where}: reset raised {type(e).__name__}: {e}") from e
              teacher.episode = self._episode_count[env]
          try:
              action = teacher.bot.act(record.obs, record.mask, record.info)
          except Exception as e:  # noqa: BLE001 - user code
              raise PlayerError(f"{where}: act raised {type(e).__name__}: {e}") from e
          checked = check_bot_action(self._spec.roles[role], action, record.mask, where)
          return action if checked is None else checked      # T1.2 returns the action cast to the role's dtypes
  ```

`src/colosseum/worker/rollout_worker.py::rollout_worker_process`: new parameter `teachers: Mapping[str, BotSpec] | None = None` passed to `RolloutLoop(..., teachers=teachers)` (docstring: "``teachers``: scripted kickstart teachers by student agent (``BotSpec``, never instances)").

`src/colosseum/worker/match_runner.py` (amendment A1): the `ActRecord(...)` built in `_infer` for neural groups must carry `info=env_state.result.infos.get(seat)` (as for scripted seats); add it if T1.3 did not.

- [ ] **Step 6: Kickstart, APPO, learner**

`src/colosseum/bc/kickstart.py`:
- `__init__(self, teacher: PolicyModel | None, ...)`: `teacher=None` is a scripted teacher (labels only). Replace the teacher setup with
  ```python
          self._teacher = teacher
          self._teacher_stateful = False
          if teacher is not None:
              teacher.eval()
              for p in teacher.parameters():
                  p.requires_grad_(False)
              self._teacher_stateful = bool(teacher.is_stateful)
  ```
  (`teacher` property type `PolicyModel | None`; `to()` moves the teacher only if there is one.)
- `compute`: first statement `if self._teacher is None: raise RuntimeError("a scripted kickstart teacher has no model: the loss comes from chunk labels (compute_labels)")`.
- new method:
  ```python
      def compute_labels(
          self,
          *,
          student_dist: Distribution,
          teacher_actions: Tree,
          has_teacher: Tensor,
          is_act: Tensor,
          reduction: Literal["mean_valid", "sum"],
      ) -> Tensor:
          """DAgger loss ``lambda * mean(-log pi(a_teacher))`` over the ACT slots that carry a teacher label.

          Args:
              student_dist: the student's masked distribution over ``S*B`` time-major rows.
              teacher_actions: action tree, leaves ``[S*B, ...]``; unlabeled rows hold any legal action.
              has_teacher: ``[S, B]`` bool, True on labeled slots.
              is_act: ``[S, B]`` bool.
              reduction: ``"sum"`` = the joint log-prob (one decider); ``"mean_valid"`` = the mean of
                  ``unit_log_prob`` over the valid deciders (``Units``, as in BC).
          Unlabeled slots add nothing and are not in the denominator; no labeled slot gives 0.
          """
          lam = self.current_lambda
          labeled = (has_teacher.bool() & is_act).reshape(-1)
          if lam <= 0:
              return torch.zeros((), device=is_act.device)
          if reduction == "sum":
              nll = -student_dist.log_prob(teacher_actions).float()
          elif reduction == "mean_valid":
              unit_lp = student_dist.unit_log_prob(teacher_actions).float()
              valid = student_dist.unit_valid(teacher_actions)
              total = torch.where(valid, unit_lp, torch.zeros_like(unit_lp)).sum(-1)
              nll = -total / valid.sum(-1).clamp(min=1).to(total.dtype)
          else:
              raise ValueError(f"reduction must be 'sum' or 'mean_valid', got {reduction!r}")
          mean = torch.where(labeled, nll, torch.zeros_like(nll)).sum() / labeled.sum().clamp(min=1)
          return lam * mean
  ```
- module docstring: add "A scripted teacher (``teacher=None``, DAgger) has no model: the worker writes its actions into the chunks (``teacher_action``, ``has_teacher``) and ``compute_labels`` gives ``lambda * mean(-log pi(a_teacher))`` over the labeled ACT slots."

`src/colosseum/algorithms/base.py`, in `BaseAlgorithm`:

```python
    @property
    def teacher_active(self) -> bool:
        """True while a scripted kickstart teacher should label the workers' decisions (DAgger); the learner
        sends it with the weights (``WeightPayload.teacher_active``). Default: False."""
        return False
```

`src/colosseum/algorithms/appo.py`:
- `__init__`: the kickstart block becomes
  ```python
          if kickstart is not None:
              if kickstart.teacher is not None:
                  check_teacher_state_layout(self._model, kickstart.teacher)
              kickstart.to(device)
  ```
- property:
  ```python
      @property
      def teacher_active(self) -> bool:
          """A scripted kickstart teacher with lambda > 0: the workers keep labelling decisions."""
          k = self._kickstart
          return k is not None and k.teacher is None and k.current_lambda > 0
  ```
- `_TIME_MAJOR_KEYS`: add `"teacher_action", "has_teacher"`.
- `_prepare_batch`: `teacher_action, has_teacher = self._stack_labels(chunks)` and add `"teacher_action": teacher_action, "has_teacher": has_teacher` to `batch_cpu`; new helpers:
  ```python
      @staticmethod
      def _stack_labels(chunks: list[TrajectoryChunk]) -> tuple[Any, Tensor | None]:
          """Teacher labels ``[S, B, ...]`` and ``has_teacher [S, B]`` (None if no chunk has labels). Chunks
          without labels (parked buffers, the delay of the teacher flag) get zeros and False."""
          template = next((c for c in chunks if c.has_teacher is not None), None)
          if template is None:
              return None, None
          actions = [c.teacher_action if c.has_teacher is not None
                     else tree_map(torch.zeros_like, template.teacher_action) for c in chunks]
          flags = [c.has_teacher if c.has_teacher is not None else torch.zeros_like(template.has_teacher)
                   for c in chunks]
          return tree_stack(actions, axis=1), torch.stack(flags, dim=1)

      def _label_actions(self, batch: dict) -> Any:
          """Teacher actions ``[S*B, ...]`` where a slot is labeled, else the recorded action (always legal under
          the slot's mask, so every log-prob is finite)."""
          S, B = batch["kind"].shape
          labeled = batch["has_teacher"].bool().reshape(S * B)
          teacher = tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), batch["teacher_action"])
          return tree_map(lambda t, a: torch.where(labeled.reshape(-1, *([1] * (t.dim() - 1))), t, a),
                          teacher, self._flat_actions(batch))
  ```
- `_loss_from_batch`: the kickstart block (after T4.3) becomes
  ```python
          kickstart_loss = self._zero_loss
          label_frac = self._zero_loss
          kick = self._kickstart
          if kick is not None and kick.current_lambda > 0 and not warmup:
              if kick.teacher is None:                         # scripted teacher (DAgger): labels in the chunks
                  if batch["has_teacher"] is not None:
                      label_frac = (batch["has_teacher"].bool() & is_act).sum().float() / n_act
                      with self._autocast():
                          kickstart_loss = kick.compute_labels(
                              student_dist=out.dist, teacher_actions=self._label_actions(batch),
                              has_teacher=batch["has_teacher"], is_act=is_act,
                              reduction="sum" if K == 1 else "mean_valid",
                          )
              else:
                  with self._autocast():
                      kickstart_loss = kick.compute(
                          student_dist=out.dist, obs=batch["obs"], reset_after=batch["reset_after"],
                          state0=batch["initial_state"], action_mask=batch["action_masks"],
                          actions=self._flat_actions(batch), is_act=is_act, reduction=self._entropy_reduction,
                      )
              total_loss = total_loss + kickstart_loss
  ```
  and where the result gets `kickstart_loss` / `kickstart_lambda`, add for scripted teachers (every minibatch, so the step mean is right):
  ```python
              if kick.teacher is None:
                  result["kickstart_label_frac"] = label_frac.detach()
  ```
  (rename the local `self._kickstart` uses in that block to `kick`). Module docstring bullet: "- Kickstart from a scripted teacher (``KickstartLoss(None)``, DAgger): ``lambda * mean(-log pi(a_teacher))`` over the labeled ACT slots, joint for one decider and the mean over valid deciders with ``Units``; metric ``kickstart_label_frac``."

`src/colosseum/learner/learner.py::_push_weights`:

```python
    payload = WeightPayload.from_model(agent_id, algorithm.policy_version, algorithm.model,
                                       teacher_active=bool(getattr(algorithm, "teacher_active", False)))
```

(`getattr`: duck-typed test algorithms have no such property.) Module docstring step 3: "publishes new weights (numpy ``WeightPayload``, with the scripted teacher's ``teacher_active`` flag) ...".

- [ ] **Step 7: Factory and launcher**

`src/colosseum/learner/factory.py`: replace `resolve_teacher` (T4.1 + T4.4) with the final form, which also resolves scripted agents:

```python
def _check_teacher_roles(config: ColosseumConfig, spec: GameSpec, agent_id: str, teacher_roles: Sequence[str],
                         where: str) -> None:
    from colosseum.players.registry import resolve_player_roles

    student_roles = resolve_player_roles(config, spec)[agent_id]
    missing = [r for r in student_roles if r not in teacher_roles]
    if missing:
        raise ConfigError(f"{where}: the teacher plays roles {list(teacher_roles)}; a teacher must play every role "
                          f"of its student ({list(student_roles)}), missing {missing}")


def resolve_teacher(config: ColosseumConfig, agent_id: str, spec: GameSpec) -> TeacherSpec | None:
    """The kickstart teacher of trainable ``agent_id`` (effective ``kickstart`` section), or None.

    ``teacher`` is the name of a frozen agent (neural, its own architecture) or of a scripted agent
    (DAgger: ``BotSpec``), a ``.pt`` path (the student's architecture, SP2) or a checkpoint dir
    (architecture and roles from its ``meta.json``). The teacher must play every role of its student;
    the name of a trainable agent is a ConfigError with a hint. Main process only.
    """
    from colosseum.players.registry import resolve_player_roles

    ks = config.get_agent_config(agent_id).kickstart
    if ks.teacher is None:
        return None
    ref = ks.teacher
    where = f"agent {agent_id!r}: kickstart.teacher={ref!r}"
    settings = {"lambda_": float(ks.lambda_), "decay_steps": int(ks.decay_steps), "kl": ks.kl}
    if ref in agent_ids(config):
        kind = config.agent_kind(ref)
        if kind == "trainable":
            raise ConfigError(
                f"{where} names a trainable agent; a teacher is fixed: to learn from a snapshot, declare it as a "
                f"frozen agent with path (agents.<name>: {{kind: frozen, path: <checkpoint dir>}}) and name that agent"
            )
        entry = config.agent_entry(ref)
        if kind == "scripted":
            _check_teacher_roles(config, spec, agent_id, resolve_player_roles(config, spec)[ref], where)
            return TeacherSpec(kind="scripted", source=f"scripted agent {ref!r}",
                               bot=BotSpec(entry.class_path, dict(entry.kwargs)), **settings)
        frozen = _load_frozen(config, ref, entry.path, spec, where)
        source = f"frozen agent {ref!r} ({entry.path})"
    else:
        path = Path(ref)
        if path.is_file() and path.suffix == ".pt":
            frozen = _student_pt(config, agent_id, ref, spec, where)
        elif path.is_dir():
            frozen = _load_frozen(config, agent_id, ref, spec, where)
        else:
            raise ConfigError(f"{where} is neither an agent of the config ({agent_ids(config)}) nor an existing .pt "
                              f"file or checkpoint dir")
        source = ref
    _check_teacher_roles(config, spec, agent_id, frozen.roles, where)
    return TeacherSpec(kind="neural", source=source, frozen=frozen, **settings)
```

(add `from collections.abc import Sequence`), and in `build_kickstart` replace the scripted `raise` with:

```python
    if teacher.kind == "scripted":                 # DAgger: the workers write the labels into the chunks
        return KickstartLoss(None, initial_lambda=teacher.lambda_, decay_steps=teacher.decay_steps)
```

`src/colosseum/launcher.py`:
- `_worker_main(..., teachers: Mapping[str, BotSpec] | None = None)` passes `teachers=teachers` to `rollout_worker_process` (type-only import of `BotSpec`);
- `_start_children`, worker `kwargs=dict(...)`: add
  ```python
                      teachers={aid: t.bot for aid, t in self._teachers.items() if t is not None and t.kind == "scripted"},
  ```

- [ ] **Step 8: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_dagger_chunks.py tests/contract/test_sp3_dagger_worker.py tests/unit/test_sp3_dagger_loss.py tests/unit/test_sp3_learner_factory.py tests/unit/test_chunk_v2_types.py tests/unit/test_slot_buffers.py tests/unit/test_sp2_serialization.py tests/unit/test_kickstart_v2.py tests/contract/test_learner_reproduces_worker.py tests/contract/test_rollout_loop_lineups.py tests/integration/test_sp3_dagger_run.py -q`
Expected: all pass (the integration run takes about 10 s).

- [ ] **Step 9: Full fast suite and lint**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` then `.venv/bin/ruff check .`
Expected: all pass, zero warnings; ruff clean.

- [ ] **Step 10: Commit**

```bash
git add src/colosseum/core/types.py src/colosseum/transport/serialization.py src/colosseum/worker \
    src/colosseum/bc/kickstart.py src/colosseum/algorithms src/colosseum/learner src/colosseum/launcher.py \
    tests/game_helpers.py tests/unit/test_sp3_dagger_chunks.py tests/contract/test_sp3_dagger_worker.py \
    tests/unit/test_sp3_dagger_loss.py tests/unit/test_sp3_learner_factory.py tests/integration/test_sp3_dagger_run.py
git commit -m "feat: scripted kickstart teacher (DAgger): worker labels in chunks, label loss, teacher flag"
git push origin sp3-league
```

---

## Contract notes

**Assumptions about other parts (relied on, not re-specified here).**

- **T0.1:** `tests/game_helpers.py` exports `SP2_CHECKPOINT` (the SP2 checkpoint dir `.../run/checkpoints/agent_0/ckpt_v3`) and `SP2_TTT_TINY` (the SP2-format config it was generated with); T4.2 loads that checkpoint as `init`.
- **T1.1:** `TrainableAgent.matchmaking/init/kickstart` exist as free dicts; `_validate_agent_overrides` validates `get_agent_config` of the trainable agents only; `agent_kind`/`agent_entry` handle the implicit `agent_0`; `--set agents.<id>.<section>.<key>=...` walks the discriminated agent union down to the free dicts.
- **T1.2:** `resolve_player_roles(config, spec)` returns every agent (implicit `agent_0` included); `make_bot(BotSpec, spec)` sets `game_spec`; `check_bot_action(role, action, mask, where)` raises `PlayerError(f"{where}: ...")`; `bot_rng`, `check_bot_action`, `ScriptedBot` live in `colosseum.players.scripted`; see A3 for `load_frozen` / `build_frozen_model`.
- **T1.3:** `SeatAssignment.source` / `SeatResult.source`; `MatchRunner._resolve` keeps `source`; the optional observer hook `on_episode_start(env, layout, episode_seed)` runs after every env reset (the initial one inside `MatchRunner.__init__` included); see A1 for `ActRecord.info`.
- **T1.4:** `Coordinator(config, spec, player_roles, checkpoint_dir, env_steps)`; the launcher's `RunSetup` carries every agent's roles (called `setup.player_roles` here); `RolloutLoop(..., fixed_players=...)`; `game_helpers.make_coordinator` passes player roles; `_resolve_lineups` keeps `source` (T3.3 Step 4 fixes it otherwise).
- **T1.5:** `validate_config` returns a `ValidationReport` (called `report` in its body) and computes the player roles; `colosseum validate` prints `report.lines` (T3.3 Step 6 adds it otherwise); scripted agents are imported and played, frozen agents loaded.
- **T2.1:** `core.config.LEGACY_OVERRIDE_KEYS: set[str]` (dotted keys `apply_overrides` accepts without the schema walk; `{"checkpoint.pool_size"}`), extended here by T3.1 and T4.1; `checkpoint.pool_size` is still accepted from `cli_runner.TINY`.
- **T2.3:** `colosseum.league.pfsp` (`PlayerKey`, `pfsp_weight(score, weighting, exponent)` with `hard`/`balanced`/`uniform` and the 1e-6 floor, `PfspStats(halflife_by_agent)` with `score(layout, owner, player)`); the coordinator holds its `PfspStats` (called `self._pfsp` here) and updates it in `report_match_result`; `ratings_snapshot()` lists scripted/frozen agents; T2.3 created `src/colosseum/league/__init__.py`.

**Contract details pinned down by this part.**

- `MatchmakingConfig.anchors` dict values and every `OpponentShares` field are `ScheduleField` (`Annotated[ScheduleValue, BeforeValidator(parse_schedule)]`): non-negative finite numbers, schedule keys non-negative integers (strings like `"1e6"` / `"2_000_000"` accepted), points stored sorted; `model_dump(mode="json")` round-trips through YAML.
- `effective_mix(context, owner, layout, team, env_steps)` returns the shares for opposing team index `team` with `rivals` folded into `latest` when the owner cannot play the team and the shares of never-fillable categories spread proportionally; snapshots count as available (A15).
- Played-share metric: the agent's own `episodes` record gets `opponents: {layout: {"teams": n, "categories": {latest, snapshots, rivals, anchors, fallback: share}, "anchors": {id: share}}}`; WandB sees `episodes/<agent>/opponents/...`. No new reserved agent ids (A11).
- Label loss reduction: APPO calls `compute_labels(..., reduction="sum")` with one decider (joint log-prob) and `"mean_valid"` with `K > 1` (spec: "as in BC"), whatever `algorithm.entropy_reduction` is.
- A scripted teacher ignores `kickstart.kl` (spec: `kl` only for neural teachers).
- `critic_warmup_steps` is not part of what a resume overrides: a resumed agent keeps its configured `critic_warmup_steps` and continues the saved counter; only `init.from` is ignored on resume.
- Live example configs (`configs/examples/*.yaml`) keep their SP2 matchmaking knobs in this part (translated at load); T6.3 rewrites `team_tag.yaml`, T6.6 may rewrite the rest.

**Proposed amendments to the overview contract.**

- **A1.** `ActRecord.info` is set for every acting seat, neural seats included (`result.infos.get(seat)`), not only for scripted ones. Reason: the DAgger teacher of a neural student seat receives that seat's `info` (spec block 6: "с теми же obs, mask, info"). T4.5 adds it in `MatchRunner._infer` if T1.3 did not.
- **A2.** The overview's test-kit list should name T0.1's `SP2_CHECKPOINT` and `SP2_TTT_TINY` (as part 01 defines them). Reason: T4.2's criterion test "SP2 checkpoint loads as `init`" uses them; the overview's contract does not list them.
- **A3.** `.pt` teachers: `learner.factory` builds their `FrozenSpec` itself (the student's effective networks and roles, `read_weights_file`); `load_frozen(config, agent_id, path, spec)` is used for checkpoint dirs (any `agent_id`, trainable included: architecture and roles from `meta.json`) and frozen agents; `build_frozen_model(config, frozen, spec)` uses only `frozen` and `spec` (the passed `config` may be a `get_agent_config` result with `agents == {}`) and raises ConfigError (via `check_model_state`) when the weights do not fit. Reason: the contract only says `load_frozen` is "also used for teacher/init paths"; this pins the `.pt` semantics to SP2's ("сеть ученика") without depending on how T1.2 treats trainable ids.
- **A4.** `MatchmakerContext` is a dataclass with the contract's attributes plus constructor fields `snapshots_fn`, `pfsp_fn`, `env_steps_fn`; `MatchmakerContext.from_config(config, spec, player_roles, *, rng=None, snapshots_fn=..., pfsp_fn=None, env_steps_fn=...)`; `fixed` property; `anchors_at(owner, env_steps)`. Also `league.base.load_matchmaker_class(path)` and `league.lineups.playable_layouts(spec, config, roles)`. Reason: the contract fixes only the read API; the coordinator, `validate` and tests need one way to build a context.
- **A5.** The overview contract should list `core.config.LEGACY_OVERRIDE_KEYS: set[str]` (part 01, T2.1: `apply_overrides` skips the schema walk for these keys), extended by T3.1 with `matchmaking.mode|self_play_ratio|latest_prob|pfsp_exponent` and by T4.1 with `training.kickstart_*`. Reason: `--set` edits the raw input, which the translation then sees; SP2 integration tests pass `--set matchmaking.mode=...` and must stay green.
- **A6.** Per-agent `matchmaking` overrides merge with `core.config.merge_matchmaking` (`opponents` and `pfsp` per key; a schedule and every other key — `anchors`, `layouts` — replaced whole), not `deep_merge`. Reason: a deep merge would merge the breakpoints of a global and an agent schedule into a third schedule and union anchor maps, which nobody wrote.
- **A7.** Anchor-name checks (exists; scripted or frozen) live in `validate_matchmaking`, not in the config model. Reason: derived per-agent configs (`get_agent_config`) have `agents == {}` and would fail a model-level check.
- **A8.** `KickstartLoss(teacher: PolicyModel | None, ...)` — `None` = scripted teacher (labels only; `compute` raises); `BaseAlgorithm.teacher_active` property (default False; APPO: scripted teacher with lambda > 0), read by `learner._push_weights` through `getattr`; `WeightPayload.from_model(agent_id, policy_version, model, teacher_active=False)`; train metric `kickstart_label_frac`. Reason: the contract names the flag and `compute_labels` but not how the learner knows the flag's value.
- **A9.** `BufferSpec.teacher: bool = False`; `RolloutBuffer.write_act(..., reward, teacher_action=None)`; chunks of an agent with a scripted teacher always carry both fields (zeros/False where nothing was labelled). Reason: the buffer layer is where the labels are written; the contract lists only the chunk fields.
- **A10.** Construction helpers: `learner.factory.agent_ids`, `build_teacher_model`, `build_kickstart`, `check_teacher_compat`, `apply_init`; `bc.kickstart.check_teacher_state_layout` (replaces APPO's private `_check_teacher_state_layout`, so `validate` and APPO share the rule); `_learner_main(..., spec=None, teacher=None, init_state=None)`; `_worker_main(..., teachers=None)`; `Launcher._build_coordinator(setup)`, `_resolve_init(spec, resume_states)`, `_teachers`, `_init_states`. Reason: needed by the launcher/learner split of resolution (main process) and construction (learner process).
- **A11.** Played shares go into the owner's own `episodes` record (`opponents` field), attributed by `metrics.aggregator.opponent_draws(result)`: the owner team is the team whose seats carry `owner`; results whose owner team has latest seats of several agents (`teammates: mixed`) are skipped (teammates are drawn independently of opponents, so the shares stay unbiased); the anchor of an `anchors` team is its most frequent `fixed` agent. Reason: spec block 5 leaves the keys to the plan; this needs no new reserved agent ids.
- **A12.** Task ownership around the SP2 matchmaker: T3.1 keeps `LineupMatchmaker` alive for one task (SP2 numbers read back from `opponents` at env step 0; only the `make_mm` helper of `test_sp2_matchmaker.py` changes); T3.2 switches the coordinator to `MixtureMatchmaker` (deleting `coordinator/matchmaker.py` forces it, and `check_lineup` runs there already), builds the coordinator's `PfspStats` with the agents' `pfsp.halflife_games` (T2.3 predates the field) and deletes `tests/unit/test_sp2_matchmaker.py` after porting its guarantees; T3.3 adds `matchmaker_class`, `on_result`, the launcher's env-step wiring, metrics and the `validate` printout. Reason: the overview gives the coordinator integration to T3.3, but the green-suite rule needs a working coordinator after T3.1 and T3.2.
- **A13.** APPO trainer-state key `critic_warmup_done` (absent in older trainer states → 0: the warm-up restarts, consistent with a resume without trainer state, which also restarts with a fresh optimizer); `APPO.critic_warming_up` property; train metric `critic_warmup`; `critic_warmup_steps > 0` with `algorithm.value_loss_coeff == 0` is a ConfigError (`validate`) / ValueError (APPO). Reason: the contract only names the constructor argument.
- **A14.** `colosseum/league/__init__.py` exports lazily (PEP 562). Reason: `core.config` imports `league.schedule` (contract: "`ScheduleValue` defined in `colosseum.league.schedule`, re-used here"), and `league.base`/`league.mixture` import `core.config`; eager exports would be an import cycle (and would pull torch into `core.config`).
- **A15.** `effective_mix` is the structural mix (snapshots counted as available; anchors by positive weight at that step); the runtime draw uses the snapshots that exist. Reason: the same function serves `validate` (no snapshots yet) and the statistical tests.
- **A16.** A scripted teacher instance is reset lazily, at the student's first decision of each episode at that seat (episodes counted from `on_episode_start`), with `bot_rng(episode_seed, seat, f"{agent}/teacher")`. Reason: `on_episode_start` also runs inside `MatchRunner.__init__`, before `RolloutLoop` has its runner and lineups; the observable behavior is the spec's ("reset at the start of every episode in which the student collects at that seat"), and the teacher's random stream differs from a bot playing at the same seat.

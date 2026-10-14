# SP3 Plan — Part 03: `record`, BC data, distributed guards, demo bots, pipeline, team_tag, APPO defaults, docs, acceptance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** Spec blocks 7–11 and the acceptance of section 3:
- **T5.1** `colosseum record` (block 7): any player kind, lineups without / with `--against`, per-seat contiguous episodes per role in the existing BC format, `record.json` with the eval summary;
- **T5.2** `colosseum bc` with several `--data` and `record` output directories (block 7; criterion 2 «данные `record` проходят через `bc`»);
- **T6.1** distributed guards (block 8; criterion 9);
- **T6.2** demo bots (`tic_tac_toe`, `unit_harvest`), the fast pipeline smoke on tic-tac-toe, the fast learning tests (scripted-teacher kickstart, `RandomBot` anchor) (block 2, section 6, criterion 3);
- **T6.3** team_tag with a `RandomBot` anchor: config v3, anchor-share measurement, single-run slow test, the 6-seed check, the draw-penalty fallback (block 9; criterion 4);
- **T6.4** the slow pipeline test on `unit_harvest` with measured thresholds, the "pipeline vs scratch" measurement, the slow-suite pre-check (criterion 3, 10);
- **T6.5** APPO `Units` defaults: multi-seed measurement and the pre-fixed rule (block 10; criterion 7);
- **T6.6** documentation, example configs in the v3 form, the throughput measurement against `main` (block 11; criteria 5, 8);
- **T6.7** acceptance against section 3, the report, the stop for the owner (section 7).

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's. Additions, assumptions about other parts and proposed amendments are listed in `## Contract notes` at the end; amendments accepted into the overview win over this part.

**Execution order:** T5.1 → T5.2 → T6.1 → T6.2 → T6.3 → T6.4 → T6.5 → T6.6 → T6.7 (the index order). T6.3, T6.4 and T6.5 are long measurement tasks (about 1 h, 1 h and 2 h of machine time). They never run at the same time as each other or as any other CPU-heavy job (policy lag, and with it every learning number, depends on the load); before each measurement command check `uptime` (1-minute load average below 1.0).

**Conventions used in every task.**
- Run every command from the repository root with `.venv/bin/python` / `.venv/bin/ruff` / `.venv/bin/colosseum`.
- "Full fast suite" = `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, **zero warnings**) followed by `.venv/bin/ruff check .`.
- Every task ends with a commit on `sp3-league` (conventional prefix, no attribution or co-author lines) and `git push origin sp3-league`.
- New test files start with `test_sp3_`; new support modules under `tests/` get unique basenames and are added to `[tool.ruff.lint.isort] known-first-party` in `pyproject.toml` in the task that creates them.
- Integration runs start processes through `tests/cli_runner.py`; at most 2 worker processes per test; files only under `tmp_path`.
- Long commands (measurements, the slow suite) run in the background with output to a log file under the task's scratch directory (`nohup ... > log 2>&1 &`, or the tool's background mode); poll the log, never block a tool call for more than 10 minutes.

**Cross-task notes.**
- **Player resolution is shared with `eval -a` (Contract notes N1, N2).** T1.5 (part 01) resolves `-a name` / `-a name=path` inside `eval.evaluate` and lets `play_lineups` take `ScriptedBot` prototypes and `ScriptedPlayer`s. T5.1 extracts that loop body into `colosseum.eval.load_player` (behaviour and messages unchanged for `eval`) so `record --player/--against` resolves players by exactly the same rules (spec block 7: «правила `eval -a`»).
- **SP2 test helpers stay.** Spec block 2 lets the plan decide where `random_model` and `ScriptedPolicy` (`tests/learning/demo_learning.py`) move to `RandomBot` and the demo bots. Decision: the seven SP2 slow tests and `scripts/units_experiment.py` keep them (they are exact equivalents of `RandomBot` / `HarvestBot`, and changing them would move the baselines behind SP2's thresholds and SP2's units data); every new SP3 test uses `RandomBot` and the demo bots through `ScriptedPlayer`.
- **Shared kits.** `tests/pipeline_kit.py` (T6.2) drives the CLI pipeline for the fast smoke (T6.2), the slow pipeline test (T6.4) and `scripts/pipeline_vs_scratch.py` (T6.4). `tests/learning/sp3_bandits.py` (T6.2) holds the fast-learning games and the teacher bot.
- **New example config** `configs/examples/unit_harvest_league.yaml` (T6.2): the README pipeline and the slow pipeline test start from it; its budget is calibrated in T6.4.
- **Measurements and rulings.** The decision rules are fixed in this plan before any number is seen. Implementers run the measurements and report the raw numbers; the values they produce (thresholds, the anchors share, defaults) are recorded by the controller as rulings ("что — почему — цена ошибки") in the SDD ledger, and T6.7 copies every ruling into the report.
- **Reference machine:** WSL2, 8 cores, about 11 GB RAM, no GPU (the SP1/SP2 machine). Budgets are sized for it.

---

### Task T5.1: `colosseum record`

Spec block 7 («`record`»), section 6 (contract test «`record` → `bc`», finished in T5.2). The player plays on the eval engine (`play_lineups` on `MatchRunner`); an observer collects every decision of the recorded seats; each (env, seat) buffer is flushed whole at the episode end into the writer of the seat's role.

**Files:**
- Create: `src/colosseum/record.py`
- Modify: `src/colosseum/eval.py` (`play_lineups(..., observer=None)`; `_Collector` forwards to it)
- Modify: `src/colosseum/cli.py` (the `record` command)
- Test: `tests/integration/test_sp3_record.py`

**Interfaces:**
- Consumes:
  - T1.5's `evaluate(config, agents: Mapping[str, str | None], ...)` (its per-player loop: `config.agent_entry(name)` — `ConfigError` "Unknown agent ..." for an unknown name —, a trainable name without a path → `ConfigError` "... is a trainable agent, whose weights are not in the config; use -a <name>=<checkpoint dir or .pt>", a scripted agent → `ScriptedPlayer(functools.partial(make_bot, BotSpec(...), spec))` with its roles from `resolve_player_roles`, a frozen agent → `load_eval_model(config, name, entry.path, ...)`); `play_lineups(*, env_fn, models: Mapping[str, PolicyModel | ScriptedBot | ScriptedPlayer], ...)` (T1.5); `schedule_lineups`, `summarize`, `EvalReport`, `load_eval_model` (SP2).
  - `BotSpec`, `make_bot`, `resolve_player_roles` (T1.2).
  - `ActRecord.agent_id/obs/action/mask` (T1.3: scripted seats too), `EpisodeEnd.result.seats[*].seat/.role`, the five `MatchObserver` methods and the optional `on_episode_start` (T1.3).
  - `ObsSpec.allocate`, `ActionSpec.allocate_actions` / `full_mask` / `has_masks`, `colosseum.worker.buffers.put_row`, `colosseum.utils.fs.write_text_atomic`, `env_spec`, `make_env`, `validate_config` (T1.5: returns a `ValidationReport`).
  - `ConfigError`, `PlayerError` (T1.2); `colosseum.players.RandomBot` (T1.2).
  - Test support: `make_test_config`, `write_test_config`, `agent_role_of`, `RandomPolicy`, `TurnTakingGame`, toy games `turns`, `asymmetric`, `units`.
- Produces:
  - `colosseum.record.record(config, player, against, *, layouts, num_matches, output, num_envs=8, seed=None, deterministic=False) -> dict` (contract; `player` / `against` items are `"name"` or `"name=path"`), the content of `record.json`.
  - `colosseum.record.RECORD_FILE = "record.json"`, `RECORD_FORMAT = 1`, `DECISIONS_PER_FILE = 100_000`, `RoleWriter`, `RecordObserver`.
  - `colosseum.eval.load_player(config, name, path, *, spec, validated=None, option="-a") -> tuple[PolicyModel | ScriptedPlayer, list[str]]`: T1.5's per-player resolution as a function (`path=None` = the scripted or frozen agent `name` of the config); `option` is the flag named in the trainable-agent error (`-a`, `--player`, `--against`); `evaluate` calls it.
  - `colosseum.eval.play_lineups(..., observer: MatchObserver | None = None)`: the observer receives every callback of every episode except `on_episode_end` of discarded episodes; for those it gets the optional `on_episode_discarded(env)` (called only if defined).
  - CLI `colosseum record -c CFG --player P [--against A ...] [--layout L ...] [--num-matches N] --output DIR [--num-envs E] [--seed S] [--deterministic]`.
  - `record.json` keys: `format`, `game`, `env_kwargs`, `player`, `player_source`, `against`, `against_sources`, `layouts`, `num_matches`, `matches`, `seed`, `deterministic`, `seat_episodes`, `decisions`, `roles` (`{role: {"seat_episodes", "decisions", "files"}}`, only roles with data), `summary` (`EvalReport.to_dict()`).

- [ ] **Step 1: Check that the new names are free**

```bash
find tests -name "test_sp3_record.py"; ls src/colosseum/record.py 2>/dev/null; grep -n "def record_cmd\|\"record\"" src/colosseum/cli.py
grep -n "def load_player\|def evaluate\|agent_entry\|ScriptedPlayer" src/colosseum/eval.py | head
```

Expected: the first three print nothing; the last shows T1.5's `evaluate` with its per-player loop (`agent_entry`, `ScriptedPlayer`) and no `load_player` yet.

- [ ] **Step 2: Write the failing tests**

Create `tests/integration/test_sp3_record.py`:

```python
"""``colosseum record`` (spec block 7, T5.1): lineups with and without --against, per-seat contiguous
seat-episodes per role in the BC format, record.json, discarded episodes, errors."""
from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import play_lineups
from colosseum.record import RECORD_FILE, record
from game_helpers import RandomPolicy, TurnTakingGame, agent_role_of, make_test_config, write_test_config

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
ASYM_AGENTS = {
    "hunter": {"roles": ["hunter"]},
    "prey": {"roles": ["prey"]},
    "preybot": {**RANDOM_BOT, "roles": ["prey"]},
    "hunterbot": {**RANDOM_BOT, "roles": ["hunter"]},
}


def _turns(**agents):
    """TurnTakingGame (6 moves, seats alternate: 3 decisions per seat and episode) with RandomBot agents."""
    return make_test_config("turns", agents=agents or {"bot": RANDOM_BOT})


def _part(directory) -> dict:
    files = sorted(directory.glob("part-*.pt"))
    assert [f.name for f in files] == ["part-00000.pt"], files
    return torch.load(files[0], weights_only=True)


def _seat_episodes(data) -> list[torch.Tensor]:
    """Observations split after every ``done`` (one piece per seat-episode)."""
    dones = data["dones"]
    assert bool(dones[-1]), "a part file ends with a finished seat-episode"
    ends = torch.nonzero(dones).flatten().tolist()
    starts = [0] + [e + 1 for e in ends[:-1]]
    return [data["observations"][s:e + 1] for s, e in zip(starts, ends, strict=True)]


def test_a_bot_against_itself_records_every_seat_contiguously(tmp_path):
    out = tmp_path / "data"
    content = record(_turns(), "bot", [], layouts=None, num_matches=10, output=out, num_envs=4, seed=0)
    data = _part(out / "player")
    # 10 matches x 2 seats x 3 decisions; with 4 envs for 10 lineups some envs play extra
    # episodes after the last lineup started: the eval engine discards them, so does record
    assert data["observations"].shape == (60, 3) and data["observations"].dtype == torch.float32
    assert data["actions"].shape == (60,) and data["actions"].dtype == torch.int64
    assert data["action_masks"].shape == (60, 3) and data["action_masks"].dtype == torch.bool
    assert bool(data["action_masks"][torch.arange(60), data["actions"]].all())     # every action legal
    episodes = _seat_episodes(data)
    assert len(episodes) == 20
    for obs in episodes:                    # obs = [t / length, seat, 1]: one seat, time increasing
        assert obs.shape[0] == 3
        assert bool((obs[:, 1] == obs[0, 1]).all())
        assert bool((obs[1:, 0] > obs[:-1, 0]).all())
    assert content["player"] == "bot" and content["player_source"] is None and content["against"] == []
    assert (content["matches"], content["seat_episodes"], content["decisions"]) == (10, 20, 60)
    assert content["roles"] == {"player": {"seat_episodes": 20, "decisions": 60, "files": ["part-00000.pt"]}}
    assert content["layouts"] == ["2p"] and content["summary"]["layouts"]["2p"]["n"] == 10
    saved = json.loads((out / RECORD_FILE).read_text())
    assert {k: saved[k] for k in ("format", "player", "against", "decisions", "roles")} == {
        "format": 1, "player": "bot", "against": [], "decisions": 60, "roles": content["roles"]}


def test_against_records_only_the_players_seats(tmp_path):
    out = tmp_path / "data"
    content = record(_turns(bot=RANDOM_BOT, other=RANDOM_BOT), "bot", ["other"], layouts=None, num_matches=10,
                     output=out, num_envs=4, seed=0)
    data = _part(out / "player")
    assert data["observations"].shape[0] == 30 and int(data["dones"].sum()) == 10
    assert {int(obs[0, 1]) for obs in _seat_episodes(data)} == {0, 1}     # the pair rotates over the sides
    assert content["against"] == ["other"] and content["seat_episodes"] == 10
    (pair,) = [r for r in content["summary"]["layouts"]["2p"]["pairs"] if r["agent_a"] == "bot"]
    assert pair["agent_b"] == "other" and pair["n"] == 10


def test_a_player_without_every_role_of_a_layout_needs_against(tmp_path):
    config = make_test_config("asymmetric", agents=ASYM_AGENTS)
    with pytest.raises(ConfigError, match="--against"):
        record(config, "preybot", [], layouts=None, num_matches=2, output=tmp_path / "a", seed=0)
    with pytest.raises(ConfigError, match="--against"):
        record(config, "preybot", [], layouts=["1v2"], num_matches=2, output=tmp_path / "b", seed=0)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    content = record(config, "preybot", ["hunterbot"], layouts=None, num_matches=4, output=tmp_path / "c", seed=0)
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["prey", RECORD_FILE]
    data = _part(tmp_path / "c" / "prey")
    assert data["observations"].shape == (40, 3)            # 4 matches x 2 prey seats x 5 steps
    assert int(data["dones"].sum()) == 8
    assert "action_masks" not in data                       # the prey role has nothing to mask
    assert content["roles"]["prey"]["seat_episodes"] == 8 and "hunter" not in content["roles"]


def test_units_actions_and_uint8_observations_load_into_the_bc_trainer(tmp_path):
    config = make_test_config("units", agents={"bot": RANDOM_BOT})
    out = tmp_path / "data"
    record(config, "bot", [], layouts=None, num_matches=3, output=out, num_envs=2, seed=0)
    data = _part(out / "player")
    assert data["observations"]["grid"].dtype == torch.uint8          # the role's dtypes are kept
    assert data["observations"]["entity_mask"].dtype == torch.int8
    assert data["actions"]["base"].shape == (18,)                     # 3 episodes x 6 steps
    assert data["actions"]["units"]["move"].shape == (18, 4)          # U = 4, ActionSpec.allocate_actions layout
    assert data["action_masks"]["units"]["action"].dtype == torch.bool
    _roles, role = agent_role_of(config, "agent_0")
    model = build_model(config.get_agent_config("agent_0"), role)
    trainer = OfflineBCTrainer(model, ActionSpec.from_space(role.action_space),
                               ObsSpec.from_space(role.observation_space))
    assert trainer.load_data(out / "player") == 18                    # shapes, dtypes, legality under the masks


def test_a_frozen_agent_by_name_and_a_pt_as_name_equals_path(tmp_path):
    base = make_test_config("turns")
    _roles, role = agent_role_of(base, "agent_0")
    torch.manual_seed(0)
    pt = tmp_path / "net.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    config = make_test_config("turns", agents={"fz": {"kind": "frozen", "path": str(pt)}})
    by_name = record(config, "fz", [], layouts=None, num_matches=2, output=tmp_path / "a", seed=0)
    by_path = record(config, f"net={pt}", [], layouts=None, num_matches=2, output=tmp_path / "b", seed=0)
    assert by_name["decisions"] == by_path["decisions"] == 12
    assert by_path["player"] == "net" and by_path["player_source"] == str(pt)


def test_bad_players_and_a_non_empty_output_are_config_errors(tmp_path):
    config = _turns(bot=RANDOM_BOT)
    with pytest.raises(ConfigError, match=r"trainable agent.*--player agent_0=<checkpoint dir or \.pt>"):
        record(config, "agent_0", [], layouts=None, num_matches=1, output=tmp_path / "a")
    with pytest.raises(ConfigError, match="more than once"):
        record(config, "bot", ["bot"], layouts=None, num_matches=1, output=tmp_path / "b")
    with pytest.raises(ConfigError, match="nobody"):
        record(config, "nobody", [], layouts=None, num_matches=1, output=tmp_path / "c")
    with pytest.raises(ConfigError, match="not layouts of the game"):
        record(config, "bot", [], layouts=["9p"], num_matches=1, output=tmp_path / "d")
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "old.txt").write_text("x")
    with pytest.raises(ConfigError, match="not empty"):
        record(config, "bot", [], layouts=None, num_matches=1, output=tmp_path / "e")


def test_a_seed_makes_the_recording_reproducible(tmp_path):
    for name in ("a", "b"):
        record(_turns(), "bot", [], layouts=None, num_matches=6, output=tmp_path / name, num_envs=3, seed=7)
    a, b = _part(tmp_path / "a" / "player"), _part(tmp_path / "b" / "player")
    assert all(torch.equal(a[k], b[k]) for k in ("observations", "actions", "action_masks", "dones"))


class _CountingObserver:
    def __init__(self) -> None:
        self.acts = self.ends = self.discarded = 0

    def on_act(self, env, seat, record) -> None:
        self.acts += 1

    def on_rewards(self, env, rewards) -> None:
        pass

    def on_terminated(self, env, seats) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        pass

    def on_episode_end(self, env, end) -> None:
        self.ends += 1

    def on_episode_discarded(self, env) -> None:
        self.discarded += 1


def test_play_lineups_observer_sees_scheduled_episodes_and_the_discarded_rest():
    role = TurnTakingGame().spec.roles["player"]
    observer = _CountingObserver()
    lineups = [Lineup("2p", [SeatAssignment("r"), SeatAssignment("r")]) for _ in range(3)]
    results = play_lineups(env_fn=TurnTakingGame, models={"r": RandomPolicy(role)}, lineups=lineups, num_envs=2,
                           seed=0, observer=observer)
    # Both envs end their first episode at step 6: env 0 gets the third lineup, env 1 has none left
    # and plays an unscheduled episode that ends together with env 0's at step 12.
    assert len(results) == 3
    assert (observer.ends, observer.discarded, observer.acts) == (3, 1, 24)


def test_record_cli(tmp_path, restore_root_logging):
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    out = tmp_path / "data"
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "bot", "-n", "4", "-o", str(out),
                                       "--seed", "0", "--num-envs", "2"])
    assert result.exit_code == 0, result.output
    assert "Recorded 24 decisions" in result.output and "player: 8 seat-episodes" in result.output
    assert (out / "player" / "part-00000.pt").is_file() and (out / RECORD_FILE).is_file()
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "nobody", "-o", str(tmp_path / "x")])
    assert result.exit_code == 1 and result.stderr.startswith("Config error:")
    result = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "bot", "-o", str(tmp_path / "y"),
                                       "--num-matches", "0"])
    assert result.exit_code == 2
```

- [ ] **Step 3: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/integration/test_sp3_record.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.record'`.

- [ ] **Step 4: `load_player` and an observer for `play_lineups`**

In `src/colosseum/eval.py`, extract the body of T1.5's per-player loop in `evaluate` into a function (add `import functools` and `from colosseum.worker.match_runner import ScriptedPlayer` if T1.5 did not import them at module level):

```python
def load_player(config: ColosseumConfig, name: str, path: str | None, *, spec: GameSpec,
                validated: set[str] | None = None,
                option: str = "-a") -> tuple[PolicyModel | ScriptedPlayer, list[str]]:
    """One player of ``eval -a`` / ``record --player`` / ``record --against``: ``path=None`` is the scripted
    or frozen agent ``name`` of the config (a trainable agent needs a path: ConfigError naming ``option``),
    otherwise a checkpoint dir or a ``.pt`` (``load_eval_model``). Returns ``(player, roles)``."""
    from colosseum.players.registry import BotSpec, make_bot, resolve_player_roles

    if path is None:
        entry = config.agent_entry(name)          # ConfigError for an unknown name
        if entry.kind == "trainable":
            raise ConfigError(f"{option} {name}: '{name}' is a trainable agent, whose weights are not in the config; "
                              f"use {option} {name}=<checkpoint dir or .pt>")
        if entry.kind == "scripted":
            bot = BotSpec(entry.class_path, dict(entry.kwargs))
            make_bot(bot, spec)                    # a bad class or kwargs fail here, as a ConfigError
            roles = list(resolve_player_roles(config, spec)[name])
            return ScriptedPlayer(functools.partial(make_bot, bot, spec)), roles
        path = entry.path
        p = Path(path)
        if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
            raise ConfigError(f"agents.{name}.path={path!r}: expected a checkpoint dir or a .pt file")
    return load_eval_model(config, name, path, spec=spec, validated=validated)
```

and make the loop of `evaluate` read `models[name], players[name] = load_player(config, name, path, spec=spec, validated=validated)` (drop its now-unused locals). If T1.5's loop differs in detail (e.g. caches `resolve_player_roles`), move its body as it is and only add the `option` parameter to the trainable-agent message; T1.5's tests (`tests/unit/test_sp3_eval_players.py`) must pass unchanged.

Then replace the class `_Collector` with the version below (keep whatever T1.5 added to it, e.g. scripted-player handling, unchanged; the new parts are `inner`, the forwarding and `on_episode_start`):

```python
class _Collector:
    """``MatchObserver`` that keeps the result of every scheduled lineup and feeds the next one.

    ``inner`` (optional) receives every callback; ``on_episode_end`` only for scheduled episodes,
    and for the extra episodes of envs left without a lineup the optional ``on_episode_discarded(env)``.
    """

    def __init__(self, pending: deque[Lineup], scheduled: list[bool], inner: MatchObserver | None = None) -> None:
        self.runner: MatchRunner | None = None
        self.results: list[MatchResult] = []
        self._pending = pending
        self._scheduled = scheduled
        self._inner = inner

    def on_act(self, env, seat, record) -> None:
        if self._inner is not None:
            self._inner.on_act(env, seat, record)

    def on_rewards(self, env, rewards) -> None:
        if self._inner is not None:
            self._inner.on_rewards(env, rewards)

    def on_terminated(self, env, seats) -> None:
        if self._inner is not None:
            self._inner.on_terminated(env, seats)

    def on_lineup_applied(self, env, old, new) -> None:
        if self._inner is not None:
            self._inner.on_lineup_applied(env, old, new)

    def on_episode_start(self, env: int, layout: str, episode_seed: int | None) -> None:
        hook = getattr(self._inner, "on_episode_start", None)
        if hook is not None:
            hook(env, layout, episode_seed)

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        if self._scheduled[env]:
            self.results.append(end.result)
            if self._inner is not None:
                self._inner.on_episode_end(env, end)
        else:
            discard = getattr(self._inner, "on_episode_discarded", None)
            if discard is not None:
                discard(env)
        if self._pending:
            self.runner.set_next_lineup(env, self._pending.popleft())  # applied at this episode end
            self._scheduled[env] = True
        else:
            self._scheduled[env] = False
```

In `play_lineups`, add the keyword parameter `observer: MatchObserver | None = None` after `max_idle_steps`, document it in the docstring ("``observer`` (optional) sees every decision and episode of the scheduled lineups; see ``_Collector``"), and build the collector with `collector = _Collector(pending, [True] * n, inner=observer)`. Import `MatchObserver` from `colosseum.worker.match_runner` next to the existing `MatchRunner` / `EpisodeEnd` imports.

- [ ] **Step 5: `src/colosseum/record.py`**

```python
"""``colosseum record``: a player's decisions as behavioural-cloning data (spec block 7).

The player (a scripted or frozen agent of the config, or ``name=path`` for a checkpoint dir or a
``.pt``) plays on the eval engine (``play_lineups`` on ``MatchRunner``):
- without opponents it takes every seat of every layout it can fill, and every seat is recorded;
  a layout with a role the player does not play needs ``--against``;
- every opponent (``--against``) forms a pair with the player, scheduled like ``colosseum eval``
  (``schedule_lineups``: the orientation rotates per match, ``num_matches`` per pair and layout);
  only the player's seats are recorded.

Each (env, seat) collects its decisions in its own buffer; at the episode end the buffer goes,
whole, to the writer of the seat's role, so a seat-episode is contiguous in its file and its last
decision has ``dones = True``. Episodes the eval engine discards (extra episodes of envs left
without a scheduled lineup) are dropped. A writer produces ``<output>/<role>/part-NNNNN.pt`` in
the format of ``colosseum.bc.offline_bc`` (``observations``, ``actions``, ``action_masks`` when the
role has masks, ``dones``; the dtypes of the role's spaces) and ``<output>/record.json`` describes
the run, with the eval summary of the played matches.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec, make_env
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import Tree, tree_map
from colosseum.core.types import Lineup
from colosseum.envs.game import GameSpec, RoleSpec
from colosseum.eval import load_player, play_lineups, schedule_lineups, summarize
from colosseum.utils.fs import write_text_atomic
from colosseum.worker.buffers import put_row
from colosseum.worker.match_runner import ActRecord, EpisodeEnd

logger = logging.getLogger(__name__)

__all__ = ["DECISIONS_PER_FILE", "RECORD_FILE", "RECORD_FORMAT", "RecordObserver", "RoleWriter", "record"]

RECORD_FILE = "record.json"
RECORD_FORMAT = 1
DECISIONS_PER_FILE = 100_000      # a part file is written once this many decisions are pending

Decision = tuple[Tree, Tree, "Tree | None"]     # (observation, action, normalized mask) of one decision


class RoleWriter:
    """Collects whole seat-episodes of one role and writes them as BC part files."""

    def __init__(self, directory: Path, role: RoleSpec, decisions_per_file: int = DECISIONS_PER_FILE) -> None:
        self.directory = Path(directory)
        self._obs_spec = ObsSpec.from_space(role.observation_space)
        self._action_spec = ActionSpec.from_space(role.action_space)
        self._decisions_per_file = int(decisions_per_file)
        self._pending: list[list[Decision]] = []
        self._pending_decisions = 0
        self.files: list[str] = []
        self.seat_episodes = 0
        self.decisions = 0

    def add_episode(self, decisions: Sequence[Decision]) -> None:
        """One seat-episode, in order; empty episodes (a seat that never acted) are skipped."""
        if not decisions:
            return
        self._pending.append(list(decisions))
        self._pending_decisions += len(decisions)
        self.seat_episodes += 1
        self.decisions += len(decisions)
        if self._pending_decisions >= self._decisions_per_file:
            self.flush()

    def flush(self) -> None:
        """Write the pending seat-episodes as the next ``part-NNNNN.pt`` (atomic rename)."""
        n = self._pending_decisions
        if n == 0:
            return
        obs = self._obs_spec.allocate((n,))
        actions = self._action_spec.allocate_actions((n,))
        masks = self._action_spec.full_mask((n,)) if self._action_spec.has_masks else None
        dones = np.zeros(n, dtype=bool)
        i = 0
        for episode in self._pending:
            for o, a, m in episode:
                put_row(obs, i, o)
                put_row(actions, i, a)
                if masks is not None and m is not None:
                    put_row(masks, i, m)
                i += 1
            dones[i - 1] = True
        data: dict[str, Any] = {
            "observations": tree_map(torch.from_numpy, obs),
            "actions": tree_map(torch.from_numpy, actions),
            "dones": torch.from_numpy(dones),
        }
        if masks is not None:
            data["action_masks"] = tree_map(torch.from_numpy, masks)
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"part-{len(self.files):05d}.pt"
        tmp = self.directory / f".{name}.tmp"
        torch.save(data, tmp)
        os.replace(tmp, self.directory / name)
        self.files.append(name)
        self._pending.clear()
        self._pending_decisions = 0


class RecordObserver:
    """``MatchObserver`` for ``play_lineups``: buffers the decisions of ``player``'s seats per
    (env, seat) and hands every finished seat-episode to the writer of its role."""

    def __init__(self, writers: Mapping[str, RoleWriter], player: str) -> None:
        self._writers = writers
        self._player = player
        self._buffers: dict[tuple[int, int], list[Decision]] = {}

    def on_act(self, env: int, seat: int, record: ActRecord) -> None:
        if record.agent_id == self._player:
            self._buffers.setdefault((env, seat), []).append((record.obs, record.action, record.mask))

    def on_rewards(self, env: int, rewards: dict[int, float]) -> None:
        pass

    def on_terminated(self, env: int, seats: list[int]) -> None:
        pass

    def on_lineup_applied(self, env, old, new) -> None:
        pass

    def on_episode_end(self, env: int, end: EpisodeEnd) -> None:
        roles = {s.seat: s.role for s in end.result.seats}
        for key in sorted(k for k in self._buffers if k[0] == env):
            self._writers[roles[key[1]]].add_episode(self._buffers.pop(key))

    def on_episode_discarded(self, env: int) -> None:
        for key in [k for k in self._buffers if k[0] == env]:
            del self._buffers[key]


def _parse_player(text: str) -> tuple[str, str | None]:
    """``name`` (a scripted or frozen agent of the config) or ``name=path``."""
    name, sep, path = text.partition("=")
    if not name or (sep and not path):
        raise ConfigError(f"player {text!r}: expected an agent name of the config or name=path")
    return name, (path if sep else None)


def _schedule(spec: GameSpec, player: str, roles: Mapping[str, Sequence[str]], opponents: Sequence[str],
              layouts: Sequence[str] | None, num_matches: int) -> list[Lineup]:
    explicit = bool(layouts)
    chosen = list(dict.fromkeys(layouts)) if layouts else list(spec.layouts)
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    lineups: list[Lineup] = []
    for layout in chosen:
        if not opponents:
            missing = sorted({seat.role for seat in spec.layouts[layout]} - set(roles[player]))
            if missing:
                if explicit:
                    raise ConfigError(f"layout {layout!r}: player {player!r} does not play the roles {missing}; "
                                      f"add --against <player> for those seats")
                continue
            lineups.extend(schedule_lineups(spec, layout, {player: roles[player]}, num_matches))
            continue
        found: list[Lineup] = []
        for opponent in opponents:
            pair = schedule_lineups(spec, layout, {player: roles[player], opponent: roles[opponent]}, num_matches)
            found.extend(lineup for lineup in pair if any(seat.agent_id == player for seat in lineup.seats))
        if not found and explicit:
            raise ConfigError(f"layout {layout!r}: {player!r} and {list(opponents)} cannot fill its seats together")
        lineups.extend(found)
    if not lineups:
        hint = ("add --against <player> for the roles it does not play" if not opponents
                else "check the players' roles (agents.<id>.roles)")
        raise ConfigError(f"player {player!r} (roles {list(roles[player])}) cannot fill any layout of the game "
                          f"{sorted(spec.layouts)}: {hint}")
    return lineups


def record(config: ColosseumConfig, player: str, against: Sequence[str], *, layouts: Sequence[str] | None,
           num_matches: int, output: str | Path, num_envs: int = 8, seed: int | None = None,
           deterministic: bool = False) -> dict:
    """Record ``player``'s decisions into ``output`` (module docstring); returns the ``record.json`` content."""
    if num_matches < 1:
        raise ConfigError(f"num_matches must be >= 1, got {num_matches}")
    spec = env_spec(config)
    out = Path(output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConfigError(f"--output {out}: exists and is not empty; choose a new directory")
    p_name, p_path = _parse_player(player)
    opponents = [_parse_player(text) for text in against]
    names = [p_name] + [name for name, _ in opponents]
    reused = sorted({name for name in names if names.count(name) > 1})
    if reused:
        raise ConfigError(f"player names {reused} are used more than once; without --against the player takes "
                          f"every seat (a bot against itself)")
    models: dict[str, Any] = {}
    roles: dict[str, list[str]] = {}
    validated: set[str] = set()
    for option, (name, path) in [("--player", (p_name, p_path)), *(("--against", o) for o in opponents)]:
        models[name], roles[name] = load_player(config, name, path, spec=spec, validated=validated, option=option)
    lineups = _schedule(spec, p_name, roles, [name for name, _ in opponents], layouts, num_matches)
    out.mkdir(parents=True, exist_ok=True)
    writers = {role: RoleWriter(out / role, spec.roles[role]) for role in roles[p_name]}
    results = play_lineups(env_fn=lambda: make_env(config), models=models, lineups=lineups, num_envs=num_envs,
                           seed=seed, deterministic=deterministic, max_idle_steps=config.env.max_idle_steps,
                           observer=RecordObserver(writers, p_name))
    for writer in writers.values():
        writer.flush()
    report = summarize(spec, results, agents=names, num_matches=num_matches, deterministic=deterministic)
    played = {lineup.layout for lineup in lineups}
    content = {
        "format": RECORD_FORMAT,
        "game": config.env.env_class,
        "env_kwargs": config.env.kwargs,
        "player": p_name,
        "player_source": p_path,
        "against": [name for name, _ in opponents],
        "against_sources": {name: path for name, path in opponents},
        "layouts": [name for name in spec.layouts if name in played],
        "num_matches": num_matches,
        "matches": len(results),
        "seed": seed,
        "deterministic": deterministic,
        "seat_episodes": sum(w.seat_episodes for w in writers.values()),
        "decisions": sum(w.decisions for w in writers.values()),
        "roles": {role: {"seat_episodes": w.seat_episodes, "decisions": w.decisions, "files": list(w.files)}
                  for role, w in writers.items() if w.decisions},
        "summary": report.to_dict(),
    }
    write_text_atomic(out / RECORD_FILE, json.dumps(content, indent=2, default=str) + "\n")
    logger.info("Recorded %d decisions of %s (%d seat-episodes, %d matches) into %s", content["decisions"], p_name,
                content["seat_episodes"], content["matches"], out)
    return content
```

Notes for the implementer:
- `ActRecord.obs` is already a copy cast to the role's dtypes, `ActRecord.action` the numpy tree sent to the env; `put_row` into the `allocate_actions` layout makes a scripted bot's action (a Python `int`, a list, a 0-d array) the same layout as a network's. If T1.3 made `ActRecord.mask` `None` for a role with masks, the writer leaves the full mask in that row (`full_mask` default), which is what the tracker normalizes to.
- `env_kwargs` goes into JSON with `default=str`, so a non-JSON value cannot break the write.

- [ ] **Step 6: The `record` command**

In `src/colosseum/cli.py`, add after the `eval` command:

```python
@main.command("record")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Config YAML: its env is used for every match; its scripted/frozen agents can be named")
@click.option("--player", "-p", required=True,
              help="Who is recorded: a scripted or frozen agent of the config, or name=path (checkpoint dir or .pt)")
@click.option("--against", "against", multiple=True,
              help="Opponent (same forms as --player; repeatable): each one plays the player in eval's pair "
                   "rotation and only the player's seats are recorded. Without --against the player takes every "
                   "seat and every seat is recorded.")
@click.option("--layout", "layouts", multiple=True,
              help="Layout to play (repeatable). Default: every layout the players can fill.")
@click.option("--num-matches", "-n", default=100, type=click.IntRange(min=1), show_default=True,
              help="Matches per layout (with --against: per opponent and layout)")
@click.option("--output", "-o", required=True, type=click.Path(file_okay=False),
              help="New or empty directory: <output>/<role>/part-NNNNN.pt (BC data) and record.json")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Parallel environments")
@click.option("--seed", default=None, type=int, help="Seed for env resets, bots and sampling")
@click.option("--deterministic", is_flag=True, default=False,
              help="Neural players act greedily (scripted bots are unaffected)")
def record_cmd(config: str, player: str, against: tuple[str, ...], layouts: tuple[str, ...], num_matches: int,
               output: str, num_envs: int, seed: int | None, deterministic: bool) -> None:
    """Record a player's decisions as behavioural-cloning data (read by 'colosseum bc --data <output>').

    Exit code: 0 done, 1 config error (unknown player, a layout the player cannot fill without
    --against, a non-empty output directory) or a scripted player's illegal action, 2 bad
    command-line arguments, 130 SIGINT, 143 SIGTERM.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with _config_errors():
        from colosseum.core.config import load_config
        from colosseum.core.errors import PlayerError
        from colosseum.core.registry import validate_config
        from colosseum.eval import EvalReport
        from colosseum.record import RECORD_FILE, record

        cfg = load_config(config)
        validate_config(cfg)
        try:
            content = record(cfg, player, list(against), layouts=list(layouts) or None, num_matches=num_matches,
                             output=output, num_envs=num_envs, seed=seed, deterministic=deterministic)
        except PlayerError as e:
            click.echo(f"Player error: {e}", err=True)
            sys.exit(1)
    summary = content["summary"]
    report = EvalReport(agents=summary["agents"], num_matches=summary["num_matches"],
                        deterministic=summary["deterministic"], layouts=summary["layouts"])
    click.echo("\n" + report.text())
    click.echo(f"Recorded {content['decisions']} decisions ({content['seat_episodes']} seat-episodes of "
               f"{content['matches']} matches) into {output}:")
    for role, cell in content["roles"].items():
        click.echo(f"  {role}: {cell['seat_episodes']} seat-episodes, {cell['decisions']} decisions, "
                   f"{len(cell['files'])} file(s)")
    click.echo(f"Description: {output}/{RECORD_FILE}")
```

If T1.x already added `PlayerError` to `_config_errors`, drop the inner `try`/`except` (the outer one prints it).

- [ ] **Step 7: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/integration/test_sp3_record.py tests/unit/test_sp3_eval_players.py tests/integration/test_sp2_eval_cli.py tests/contract/test_sp2_eval_engine.py -q`

Expected: all pass (the T1.5 and SP2 eval tests guard the `load_player` extraction and the `_Collector` change).

- [ ] **Step 8: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/record.py src/colosseum/eval.py src/colosseum/cli.py tests/integration/test_sp3_record.py
git commit -m "feat: colosseum record writes a player's decisions as BC data per role"
git push origin sp3-league
```

---
### Task T5.2: `bc` with several `--data` and record dirs

Spec block 7 («`colosseum bc`»), criterion 2 («данные `record` проходят через `bc`»). `--data` becomes repeatable; a directory with `record.json` contributes the folders of the agent's roles; every other path keeps the SP2 behaviour (a `.pt` file, or every `*.pt` of a directory).

**Files:**
- Modify: `src/colosseum/bc/offline_bc.py` (`bc_data_sources`)
- Modify: `src/colosseum/cli.py` (`bc`: repeatable `--data`, decisions count in the final message)
- Test: `tests/integration/test_sp3_bc_record.py`

**Interfaces:**
- Consumes: `colosseum.record.record`, `RECORD_FILE` (T5.1); `OfflineBCTrainer.load_data` (SP2); `DataError`; the `bc` command's existing agent and role resolution (SP2, adapted by T1.x).
- Produces:
  - `colosseum.bc.offline_bc.bc_data_sources(paths: Sequence[str | Path], roles: Sequence[str]) -> list[Path]`: a `record` output dir (it holds `record.json`) becomes its role folders, in the order of `roles`, roles without data skipped; `DataError` when such a dir has none of `roles` or an unreadable `record.json`; any other path is returned unchanged.
  - CLI `colosseum bc ... --data D [--data D2 ...]`; the final line reads `BC training complete (<agent>): <N> decisions, final-epoch NLL=...` (the SP2 substrings `(<agent>)` and `final-epoch NLL` stay).

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_sp3_bc_record.py`:

```python
"""``colosseum bc`` reads ``record`` output dirs and several --data paths (spec block 7, T5.2)."""
from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner

from colosseum.bc.offline_bc import bc_data_sources
from colosseum.cli import main
from colosseum.core.errors import DataError
from colosseum.core.registry import build_model
from colosseum.record import RECORD_FILE, record
from game_helpers import agent_role_of, make_test_config, write_test_config

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
ASYM_AGENTS = {
    "hunter": {"roles": ["hunter"]},
    "prey": {"roles": ["prey"]},
    "preybot": {**RANDOM_BOT, "roles": ["prey"]},
    "hunterbot": {**RANDOM_BOT, "roles": ["hunter"]},
}

pytestmark = pytest.mark.usefixtures("restore_root_logging")


def bc(*args):
    return CliRunner().invoke(main, ["bc", *map(str, args)])


def test_sources_expand_record_dirs_into_the_agents_role_folders(tmp_path):
    rec = tmp_path / "rec"
    for role in ("hunter", "prey"):
        (rec / role).mkdir(parents=True)
    (rec / RECORD_FILE).write_text(json.dumps({"roles": {"hunter": {}, "prey": {}}}))
    plain = tmp_path / "plain"
    plain.mkdir()
    one = tmp_path / "one.pt"
    one.write_bytes(b"")
    assert bc_data_sources([rec, plain, one], ["prey"]) == [rec / "prey", plain, one]
    assert bc_data_sources([str(rec)], ["prey", "hunter"]) == [rec / "prey", rec / "hunter"]
    with pytest.raises(DataError, match="the agent plays"):
        bc_data_sources([rec], ["scout"])
    (rec / RECORD_FILE).write_text("{not json")
    with pytest.raises(DataError, match=RECORD_FILE):
        bc_data_sources([rec], ["prey"])


def test_bc_trains_on_a_record_dir(tmp_path):
    cfg = make_test_config("turns", agents={"bot": RANDOM_BOT})
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    record(cfg, "bot", [], layouts=None, num_matches=10, output=tmp_path / "rec", num_envs=4, seed=0)
    out = tmp_path / "bc.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", out, "--epochs", 2, "--batch-size", 16)
    assert result.exit_code == 0, result.output
    assert "(agent_0): 60 decisions, final-epoch NLL" in result.output
    _roles, role = agent_role_of(cfg, "agent_0")
    build_model(cfg.get_agent_config("agent_0"), role).load_state_dict(torch.load(out, weights_only=True))


def test_bc_takes_several_data_paths(tmp_path):
    cfg = make_test_config("turns", agents={"bot": RANDOM_BOT})
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    for seed in (0, 1):
        record(cfg, "bot", [], layouts=None, num_matches=10, output=tmp_path / f"rec{seed}", num_envs=4, seed=seed)
    part = tmp_path / "rec1" / "player" / "part-00000.pt"
    result = bc("-c", cfg_path, "-d", tmp_path / "rec0", "-d", part, "-o", tmp_path / "bc.pt", "--epochs", 1)
    assert result.exit_code == 0, result.output
    assert "(agent_0): 120 decisions" in result.output


def test_bc_reads_only_the_agents_roles_of_a_record(tmp_path):
    cfg = make_test_config("asymmetric", agents=ASYM_AGENTS)
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "asymmetric", agents=ASYM_AGENTS)
    record(cfg, "preybot", ["hunterbot"], layouts=None, num_matches=4, output=tmp_path / "rec", seed=0)
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", tmp_path / "prey.pt", "--agent", "prey", "--epochs", 1)
    assert result.exit_code == 0, result.output
    assert "(prey): 40 decisions" in result.output
    result = bc("-c", cfg_path, "-d", tmp_path / "rec", "-o", tmp_path / "hunter.pt", "--agent", "hunter")
    assert result.exit_code == 1
    assert result.stderr.startswith("Config error:") and "prey" in result.stderr
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/integration/test_sp3_bc_record.py -q`

Expected: collection error `ImportError: cannot import name 'bc_data_sources'`.

- [ ] **Step 3: `bc_data_sources`**

In `src/colosseum/bc/offline_bc.py` add `import json` and `from collections.abc import Sequence` to the imports, extend the module docstring's data-format paragraph with "A ``colosseum record`` output directory (``record.json`` plus one folder of part files per role) is read through ``bc_data_sources``.", and add after `per_sample_nll`:

```python
def bc_data_sources(paths: Sequence[str | Path], roles: Sequence[str]) -> list[Path]:
    """Expand ``colosseum bc --data`` paths for an agent playing ``roles``.

    A ``colosseum record`` output directory (it contains ``record.json``) becomes its folders of
    ``roles`` that hold data, in the order of ``roles``; any other path (a ``.pt`` file or a
    directory of ``.pt`` files) is returned unchanged. ``DataError`` if a record directory has an
    unreadable ``record.json`` or no data for any of ``roles``.
    """
    from colosseum.record import RECORD_FILE

    sources: list[Path] = []
    for raw in paths:
        path = Path(raw)
        meta = path / RECORD_FILE
        if not (path.is_dir() and meta.is_file()):
            sources.append(path)
            continue
        try:
            recorded = json.loads(meta.read_text())["roles"]
            recorded_roles = set(recorded)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise DataError(f"{meta}: not a readable {RECORD_FILE} ({type(e).__name__}: {e})") from e
        found = [path / role for role in roles if role in recorded_roles and (path / role).is_dir()]
        if not found:
            raise DataError(f"{path}: the record has data for the roles {sorted(recorded_roles)}, the agent plays "
                            f"{list(roles)}; record one of the agent's roles or choose another --agent")
        sources.extend(found)
    return sources
```

- [ ] **Step 4: Repeatable `--data` in the `bc` command**

In `src/colosseum/cli.py`, in the `bc` command:
- replace the `--data` option with

```python
@click.option("--data", "-d", "data", required=True, multiple=True, type=click.Path(exists=True),
              help="BC data, repeatable: a .pt file, a directory of .pt files (keys: observations, actions, "
                   "optional action_masks, dones; trees in the agent's spaces), or a 'colosseum record' output "
                   "directory (its record.json selects the folders of the agent's roles)")
```

- change the parameter to `data: tuple[str, ...]`;
- keep the agent's role list of the existing role resolution in a variable `roles` (today: `role = agent_role_spec(spec, resolve_agent_roles(cfg, spec)[agent])` becomes `roles = resolve_agent_roles(cfg, spec)[agent]` and `role = agent_role_spec(spec, roles)`; if T1.x replaced `resolve_agent_roles`, use its successor the same way);
- replace the data-loading block with

```python
    with _config_errors():  # unreadable or malformed data, actions that do not fit the policy (DataError)
        from colosseum.bc.offline_bc import bc_data_sources

        for source in bc_data_sources(data, roles):
            trainer.load_data(source)
        metrics = trainer.train(num_epochs=epochs, batch_size=batch_size)
```

- and the first message line with

```python
    message = (f"BC training complete ({agent}): {int(metrics['num_samples'])} decisions, "
               f"final-epoch NLL={metrics['bc_loss']:.4f}")
```

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/integration/test_sp3_bc_record.py tests/integration/test_sp2_bc_cli.py tests/unit/test_sp2_bc_trainer.py tests/learning/test_sp2_bc_learns.py -q`

Expected: all pass (the SP2 BC tests still see `(agent_0)` and `final-epoch NLL`).

- [ ] **Step 6: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/bc/offline_bc.py src/colosseum/cli.py tests/integration/test_sp3_bc_record.py
git commit -m "feat: colosseum bc takes several --data paths and record output dirs"
git push origin sp3-league
```

---

### Task T6.1: Distributed-mode guards

Spec block 8, criterion 9. `run-learner` / `run-workers` keep the SP2 scope: settings that cannot be reduced to it are a `ConfigError` naming SP5; any opponent mix other than "latest only" is reduced to it with one warning; every decision looks at the values of the final config; the distributed learner's snapshot storage uses the local retention rules.

**Files:**
- Modify: `src/colosseum/distributed.py` (`check_distributed_scope`, `distributed_checkpoint_manager`; `distributed_setup` calls the check first; `run_distributed_learner` builds its `CheckpointManager` through the helper; the module docstring's scope paragraph)
- Test: `tests/unit/test_sp3_distributed_guards.py`

**Interfaces:**
- Consumes: `ColosseumConfig.fixed_agent_ids()`, `get_trainable_agent_ids()`, `get_agent_config(aid)` (T1.1; deep-merged `matchmaking`, `init`, `kickstart` per T3.1/T4.1), `MatchmakingConfig.opponents` / `matchmaker_class` (T3.1), `InitConfig.from_` / `critic_warmup_steps`, `KickstartConfig.teacher` (T4.1), `CheckpointManager(base_dir, keep_last, keep_every, interval, on_evict)` (T2.1), the SP2 fixture `tests/fixtures/sp2/configs/tic_tac_toe.yaml` (T0.1; Contract note N7).
- Produces:
  - `colosseum.distributed.check_distributed_scope(config: ColosseumConfig) -> None` (ConfigError listing every refused setting; one WARNING from logger `colosseum.distributed` containing `reduced to latest only` when some trainable agent's mix is not latest-only);
  - `colosseum.distributed.distributed_checkpoint_manager(config: ColosseumConfig, base_dir: str | Path) -> CheckpointManager`.

**What is refused (spec block 8, by value):**

| Setting | Refused when |
|---|---|
| scripted / frozen agents | `config.fixed_agent_ids()` is not empty (this also covers every non-empty anchor list: anchors can only name fixed agents) |
| `init` | an agent's effective `init.from_` is not `None`, or `init.critic_warmup_steps > 0` (`strict` alone is harmless) |
| `matchmaking.matchmaker_class` | not `None` |
| per-agent `matchmaking` | an agent's effective `matchmaking` differs from the global section |
| `kickstart` | an agent's effective `kickstart` differs from the global section (a teacher per agent), or the global `teacher` is set and is not a `.pt` path (a fixed agent's name or a checkpoint dir) |

The SP2 forms stay accepted: `training.kickstart_*` (translated to the top-level `kickstart` by T4.1) and an equivalent top-level `kickstart` with a `.pt` teacher.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_distributed_guards.py`:

```python
"""Distributed roles keep the SP2 scope (spec block 8, T6.1): SP3 settings are ConfigErrors naming
SP5, any opponent mix is reduced to latest only with one warning, decisions by values, retention."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager, distributed_setup
from game_helpers import agent_role_of, make_test_config, write_test_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SP2_TTT = REPO_ROOT / "tests" / "fixtures" / "sp2" / "configs" / "tic_tac_toe.yaml"
RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}
LATEST_ONLY = {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}


class _ExitedProcess:
    """mp.Process stand-in (never reached when the scope check refuses the config)."""

    exitcode = 0

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


def _reductions(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "colosseum.distributed" and r.levelno == logging.WARNING
            and "reduced to latest only" in r.getMessage()]


def _teacher_pt(tmp_path) -> Path:
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / "teacher.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


@pytest.mark.parametrize(("sections", "needle"), [
    ({"agents": {"bot": RANDOM_BOT}}, "bot"),
    ({"agents": {"old": {"kind": "frozen", "path": "missing.pt"}}}, "old"),
    ({"init": {"from": "bc.pt"}}, "init.from"),
    ({"init": {"critic_warmup_steps": 5}}, "critic_warmup_steps"),
    ({"matchmaking": {"matchmaker_class": "my_game.league.Mine"}}, "matchmaker_class"),
    ({"agents": {"agent_0": {"matchmaking": {"teammates": "mixed"}}}}, "agents.agent_0.matchmaking"),
    ({"agents": {"agent_0": {"kickstart": {"lambda": 0.5}}}}, "agents.agent_0.kickstart"),
    ({"kickstart": {"teacher": "runs/x/checkpoints/agent_0/ckpt_v10"}}, "kickstart.teacher"),
], ids=["scripted", "frozen", "init-from", "critic-warmup", "matchmaker-class", "agent-matchmaking",
        "agent-kickstart", "teacher-dir"])
def test_sp3_settings_are_refused_with_a_pointer_to_sp5(sections, needle):
    config = make_test_config("turns", **sections)
    with pytest.raises(ConfigError, match="SP5") as info:
        check_distributed_scope(config)
    assert needle in str(info.value)


def test_values_decide_not_the_presence_of_a_key():
    config = make_test_config(
        "turns", init={"strict": False},
        agents={"agent_0": {"matchmaking": {"teammates": "self"}, "kickstart": {"lambda": 1.0},
                            "init": {"strict": False, "critic_warmup_steps": 0}}})
    check_distributed_scope(config)          # every value equals the global one: nothing to refuse


@pytest.mark.parametrize(("opponents", "warned"), [
    (LATEST_ONLY, False),
    ({**LATEST_ONLY, "snapshots": {0: 0.0, 100000: 0.0}}, False),       # a schedule of zeros is latest only
    ({**LATEST_ONLY, "snapshots": {0: 0.0, 100000: 0.2}}, True),
    ({"latest": 0.7, "snapshots": 0.2, "rivals": 0.0, "anchors": 0.1}, True),   # the SP3 default
], ids=["latest-only", "zero-schedule", "schedule", "default"])
def test_any_mix_but_latest_only_is_reduced_with_one_warning(opponents, warned, caplog):
    config = make_test_config("turns", matchmaking={"opponents": opponents})
    with caplog.at_level(logging.WARNING, logger="colosseum.distributed"):
        distributed_setup(config, ["agent_0"])
    assert len(_reductions(caplog)) == (1 if warned else 0)


def test_sp2_configs_resolved_configs_and_the_sp2_kickstart_form_start(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="colosseum.distributed"):
        distributed_setup(load_config(SP2_TTT), ["agent_0"])         # mode self_play, latest_prob 0.8
        assert len(_reductions(caplog)) == 1
        caplog.clear()
        resolved = write_test_config(tmp_path / "resolved.yaml", "turns")   # every field written explicitly
        distributed_setup(load_config(resolved), ["agent_0"])
        assert len(_reductions(caplog)) == 1
    raw = yaml.safe_load(resolved.read_text())
    raw.pop("kickstart", None)
    raw["training"]["kickstart_teacher"] = str(_teacher_pt(tmp_path))     # SP2 form, translated by T4.1
    sp2_form = tmp_path / "sp2_kickstart.yaml"
    sp2_form.write_text(yaml.safe_dump(raw, sort_keys=False))
    distributed_setup(load_config(sp2_form), ["agent_0"])
    top_level = make_test_config("turns", kickstart={"teacher": str(_teacher_pt(tmp_path))})
    distributed_setup(top_level, ["agent_0"])


def test_the_distributed_learner_keeps_snapshots_like_the_local_one(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints")
    state = {"w": np.zeros(2, np.float32)}
    for version in range(10, 101, 10):
        manager.save("agent_0", version, state, trainer_state={"step": version})
    manager.save("agent_0", 105, state, trainer_state={"step": 105}, meta_extra={"final": True})
    agent_dir = tmp_path / "checkpoints" / "agent_0"
    kept = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
    assert kept == [30, 60, 90, 100, 105]          # every 3rd interval, the newest 2, the final one
    with_state = sorted(int(p.parent.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*/trainer_state.pt"))
    assert with_state == [100, 105]                # trainer_state.pt only inside the keep_last window


def test_entry_points_refuse_before_a_run_dir_exists(tmp_path, monkeypatch, restore_root_logging):
    import colosseum.distributed as distributed

    path = write_test_config(tmp_path / "cfg.yaml", "turns", agents={"bot": RANDOM_BOT})
    overrides = {"run.dir": str(tmp_path / "runs")}
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_learner(str(path), "agent_0", 0, "localhost:1", overrides=overrides)
    monkeypatch.setattr(distributed.mp, "Process", _ExitedProcess)
    with pytest.raises(ConfigError, match="SP5"):
        distributed.run_distributed_workers(str(path), "localhost:1", {"agent_0": "localhost:2"}, overrides)
    assert not (tmp_path / "runs").exists()
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_distributed_guards.py -q`

Expected: collection error `ImportError: cannot import name 'check_distributed_scope'`.

- [ ] **Step 3: The scope check and the retention helper**

In `src/colosseum/distributed.py`:
- add `from pathlib import Path` to the imports;
- in the module docstring, replace the paragraph starting "Scope (SP2, spec block 10)" with:

```text
Scope (SP2 spec block 10, kept by SP3 spec block 8): self-play on the latest weights, without a
coordinator, for games where every agent of the run plays every role of the enabled layouts. Each
worker env gets a fixed lineup: a layout drawn by ``matchmaking.layouts`` and every seat the latest
weights of one agent (agents in rotation over the envs). ``check_distributed_scope`` refuses, by the
values of the final config, everything SP3 added that cannot be reduced to this (scripted and frozen
agents, ``init``, a custom matchmaker, per-agent matchmaking or kickstart, a teacher other than one
global ``.pt``) with a ConfigError pointing to SP5; any opponent mix other than "latest only" is
reduced to it with one warning (default and resolved configs write every share explicitly).
Layouts are drawn once per worker env (ruling PR-3) from an RNG seeded by ``training.seed +
worker_id``, and the agent rotation starts at env 0 of worker 0, so every worker machine of a run
starts with the same layout mix and rotation.
```

- add after `DistributedSetup`:

```python
_SP5 = ("distributed mode (run-learner / run-workers) keeps the SP2 scope until SP5 (a hub and a league "
        "across machines); train this config with 'colosseum train' on one machine, or remove")


def _schedule_values(value) -> list[float]:
    """Every value of a share: a number, or the points of a piecewise-linear schedule."""
    return [float(v) for v in value.values()] if isinstance(value, dict) else [float(value)]


def _latest_only(opponents) -> bool:
    others = [v for name in ("snapshots", "rivals", "anchors") for v in _schedule_values(getattr(opponents, name))]
    return all(v == 0 for v in others) and all(v > 0 for v in _schedule_values(opponents.latest))


def check_distributed_scope(config: ColosseumConfig) -> None:
    """Spec block 8: refuse SP3 settings distributed mode cannot honour (ConfigError naming SP5) and
    warn once when some trainable agent's opponent mix is reduced to "latest only". Decides by the
    values of the final config, not by which keys the YAML wrote."""
    problems: list[str] = []
    fixed = config.fixed_agent_ids()
    if fixed:
        problems.append(f"scripted/frozen agents {fixed} (anchors, fixed opponents, teachers)")
    if config.matchmaking.matchmaker_class is not None:
        problems.append(f"matchmaking.matchmaker_class {config.matchmaking.matchmaker_class!r}")
    trainable = config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in trainable}
    for aid, acfg in agent_configs.items():
        if acfg.matchmaking != config.matchmaking:
            problems.append(f"agents.{aid}.matchmaking (opponent selection per agent)")
        if acfg.init.from_ is not None:
            problems.append(f"init.from {acfg.init.from_!r} of agent {aid!r}")
        if acfg.init.critic_warmup_steps > 0:
            problems.append(f"init.critic_warmup_steps={acfg.init.critic_warmup_steps} of agent {aid!r}")
        if acfg.kickstart != config.kickstart:
            problems.append(f"agents.{aid}.kickstart (a kickstart teacher per agent)")
    teacher = config.kickstart.teacher
    if teacher is not None and Path(teacher).suffix != ".pt":
        problems.append(f"kickstart.teacher {teacher!r} (distributed mode takes one global .pt teacher with the "
                        f"student's architecture)")
    if problems:
        raise ConfigError(f"{_SP5}: " + "; ".join(problems))
    reduced = {aid: acfg.matchmaking.opponents.model_dump(mode="json")
               for aid, acfg in agent_configs.items() if not _latest_only(acfg.matchmaking.opponents)}
    if reduced:
        logger.warning(f"distributed mode plays every seat with the agents' latest weights (SP2 scope; leagues "
                       f"across machines come with SP5): matchmaking.opponents {reduced} reduced to latest only")


def distributed_checkpoint_manager(config: ColosseumConfig, base_dir: str | Path):
    """The distributed learner's snapshot storage: the local retention (``keep_last``, every
    ``keep_every``-th, the final snapshot; ``trainer_state.pt`` only in the ``keep_last`` window).
    Nothing listens for evictions: distributed workers play latest weights only."""
    from colosseum.coordinator.checkpoint_manager import CheckpointManager

    ckpt = config.checkpoint
    return CheckpointManager(base_dir=base_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every,
                             interval=ckpt.interval)
```

- in `distributed_setup`, call `check_distributed_scope(config)` as the first statement (before `validate_config(config)`: a refused setting is reported without loading files the refused setting names);
- in `run_distributed_learner`, replace the `CheckpointManager(...)` construction (T2.1 left it with `keep_last`) and its local import by `coordinator_ckpt = distributed_checkpoint_manager(config, run_dir.checkpoints)`.

If `get_agent_config` normalizes the matchmaking or kickstart section (e.g. resolves `anchors`) so that an agent without an override no longer compares equal to the global section, compare the two normalized forms instead (normalize the global section the same way), and say so in the commit body: the rule is "an agent's effective value differs from the global one".

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_distributed_guards.py tests/unit/test_sp2_distributed_roles.py tests/integration/test_sp2_distributed_e2e.py tests/integration/test_sp2_grpc.py -q`

Expected: all pass (the SP2 distributed tests stay green; their default configs only log the reduction warning).

- [ ] **Step 5: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/distributed.py tests/unit/test_sp3_distributed_guards.py
git commit -m "feat: distributed roles refuse SP3-only settings and reduce opponent mixes to latest"
git push origin sp3-league
```

---
### Task T6.2: Demo bots, fast pipeline smoke, fast learning tests

Spec block 2 («Боты демо-игр»), section 6 («Быстрые учится», «Смоук конвейера»), criterion 3 (the fast smoke). Bots: `unit_harvest` (the port of `scripted_action`, with builds only when the mask allows them) and `tic_tac_toe` (win, block, center, random). The pipeline kit drives the real CLI; the smoke runs it on tic-tac-toe with short budgets and no thresholds. The two fast learning tests drive the real `RolloutLoop` and APPO in-process (SP2's `train_in_process`, extended).

Measured while writing this plan (pure numpy, outside the framework, 1000 / 100 games): the tic-tac-toe bot against a uniformly random player wins 0.894, draws 0.092, loses 0.014; the harvest bot wins 1.000 and never needs a masked action. The bot tests use bounds with a margin (≥ 0.80 wins and ≤ 0.05 losses; ≥ 0.95 wins).

**Files:**
- Create: `examples/tic_tac_toe/bots.py`, `examples/unit_harvest/bots.py`
- Create: `configs/examples/unit_harvest_league.yaml`
- Create: `tests/pipeline_kit.py` (support module), `tests/learning/sp3_bandits.py` (support module)
- Modify: `tests/learning/demo_learning.py` (`AgentSetup.algorithm_fn`; `train_in_process(fixed_players=, teachers=, on_chunk=)`)
- Modify: `pyproject.toml` (`known-first-party` += `pipeline_kit`, `sp3_bandits`)
- Test: `tests/unit/test_sp3_demo_bots.py`, `tests/integration/test_sp3_pipeline_smoke.py`, `tests/learning/test_sp3_fast_learning.py`

**Interfaces:**
- Consumes: `ScriptedBot`, `RandomBot` (T1.2); `BotSpec`, `FixedPlayers`, `make_bot` (T1.2); `ScriptedPlayer` (T1.3); `play_lineups` with `ScriptedPlayer` values (T1.5); `RolloutLoop(..., fixed_players=, teachers=)` (T1.4, T4.5); `WeightPayload.teacher_active`, `TrajectoryChunk.has_teacher` (T4.5); `FIXED_NETWORK_ID` (T1.3); `resolve_teacher`, `build_algorithm` (T4.1–T4.5); the CLI commands `record`, `bc` (T5.1, T5.2), `train`, `eval -a name` (T1.5); `cli_runner.run_in_session`, `run_train`, `TrainRun`; `examples.tic_tac_toe.game.WINNING_LINES`, `examples.unit_harvest.game.scripted_action`.
- Produces:
  - `examples.tic_tac_toe.bots.TicTacToeBot`, `examples.unit_harvest.bots.HarvestBot` (ScriptedBot subclasses, no kwargs);
  - `configs/examples/unit_harvest_league.yaml` (agents `main`, `greedy` = `HarvestBot`, `random` = `RandomBot`; explicit anchors `{greedy: 1.0, random: 1.0}`);
  - `pipeline_kit`: `MAIN`, `RANDOM`, `BC_NET`, `TRAINED`, `SMOKE_SETTINGS`, `PipelineGame`, `PipelineSettings`, `TIC_TAC_TOE`, `UNIT_HARVEST`, `TIC_TAC_TOE_SMOKE`, `UNIT_HARVEST_SETTINGS`, `pipeline_config`, `write_config`, `CliResult`, `cli`, `record`, `bc`, `train`, `newest_checkpoint`, `evaluate`, `PipelineRun`, `run_pipeline`;
  - `sp3_bandits`: `SilentBandit`, `DuelBandit`, `OffsetTeacher`, `OFFSET_TARGETS`;
  - `demo_learning.AgentSetup(..., algorithm_fn=None)`, `train_in_process(..., fixed_players=None, teachers=None, on_chunk=None)`.

- [ ] **Step 1: Check that the new names are free**

```bash
find tests -name "pipeline_kit.py" -o -name "sp3_bandits.py" -o -name "test_sp3_demo_bots.py" -o -name "test_sp3_pipeline_smoke.py" -o -name "test_sp3_fast_learning.py"
ls examples/tic_tac_toe/bots.py examples/unit_harvest/bots.py configs/examples/unit_harvest_league.yaml 2>/dev/null
```

Expected: nothing printed.

- [ ] **Step 2: Write the failing bot tests**

Create `tests/unit/test_sp3_demo_bots.py`:

```python
"""Demo-game bots (spec block 2, T6.2): tic-tac-toe wins, blocks, takes the center, else plays a
random legal cell; the unit_harvest bot is scripted_action with legal builds; both beat RandomBot
on the real MatchRunner (legality checked by the framework on every decision)."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.types import Lineup, MatchResult, SeatAssignment
from colosseum.eval import play_lineups
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from examples.tic_tac_toe.bots import TicTacToeBot
from examples.tic_tac_toe.game import TicTacToeGame
from examples.unit_harvest.bots import HarvestBot
from examples.unit_harvest.game import UnitHarvestGame, scripted_action


def _board(own: list[int], opp: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(obs, mask)`` of a tic-tac-toe position from the mover's side."""
    board = np.zeros(9, np.int8)
    board[own] = 1
    board[opp] = 2
    obs = np.stack([(board == 1), (board == 2), (board == 0)]).reshape(3, 3, 3).astype(np.float32)
    return obs, board == 0


def _ttt_bot(seed: int = 0) -> TicTacToeBot:
    bot = TicTacToeBot()
    bot.game_spec = TicTacToeGame.spec
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(seed))
    return bot


def test_tic_tac_toe_bot_wins_then_blocks_then_takes_the_center():
    assert _ttt_bot().act(*_board([0, 1], [3, 4]), None) == 2        # its own line beats blocking 3-4-5
    assert _ttt_bot().act(*_board([0, 8], [3, 4]), None) == 5        # block 3-4-5 (0-4-8 is taken)
    assert _ttt_bot().act(*_board([], []), None) == 4                # the center
    obs, mask = _board([4], [0])
    picks = {_ttt_bot(seed).act(obs, mask, None) for seed in range(20)}
    assert picks <= set(np.flatnonzero(mask).tolist()) and len(picks) > 1   # else a random legal cell


def test_harvest_bot_builds_only_when_the_mask_allows_it():
    env = UnitHarvestGame()
    bot = HarvestBot()
    bot.game_spec = env.spec
    bot.reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0))
    result = env.reset(0, "2p")
    obs = {key: np.array(value, copy=True) for key, value in result.obs[0].items()}
    obs["base"][0] = 0.5                       # the observation claims a stock of 5 ...
    mask = result.action_masks[0]
    assert not mask["base"][1]                 # ... but the env forbids building (stock 0)
    assert scripted_action(obs)["base"] == 1
    action = bot.act(obs, mask, None)
    assert action["base"] == 0
    np.testing.assert_array_equal(action["workers"], scripted_action(obs)["workers"])


def _wdl(results: list[MatchResult], agent: str) -> tuple[float, float, float]:
    w = d = losses = 0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        (mine,) = {s.team for s in r.seats if s.agent_id == agent}
        (other,) = set(ranks) - {mine}
        w += ranks[mine] < ranks[other]
        d += ranks[mine] == ranks[other]
        losses += ranks[mine] > ranks[other]
    n = len(results)
    return w / n, d / n, losses / n


@pytest.mark.parametrize(("env_cls", "bot_class", "matches", "min_win", "max_loss"), [
    (TicTacToeGame, "examples.tic_tac_toe.bots.TicTacToeBot", 200, 0.80, 0.05),
    (UnitHarvestGame, "examples.unit_harvest.bots.HarvestBot", 40, 0.95, 0.0),
], ids=["tic_tac_toe", "unit_harvest"])
def test_demo_bots_beat_random_bot_on_the_match_runner(env_cls, bot_class, matches, min_win, max_loss):
    spec = env_cls().spec
    players = {"bot": ScriptedPlayer(functools.partial(make_bot, BotSpec(bot_class, {}), spec)),
               "random": ScriptedPlayer(functools.partial(make_bot, BotSpec("colosseum.players.RandomBot", {}), spec))}
    seats = [SeatAssignment("bot"), SeatAssignment("random")]
    lineups = [Lineup("2p", seats if m % 2 == 0 else seats[::-1]) for m in range(matches)]
    win, _draw, loss = _wdl(play_lineups(env_fn=env_cls, models=players, lineups=lineups, num_envs=8, seed=0), "bot")
    assert win >= min_win and loss <= max_loss, (win, loss)
```

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_demo_bots.py -q`

Expected: collection error `ModuleNotFoundError: No module named 'examples.tic_tac_toe.bots'`.

- [ ] **Step 3: The bots**

Create `examples/tic_tac_toe/bots.py`:

```python
"""A scripted tic-tac-toe player (SP3 spec block 2): complete an own line if possible, else block the
opponent's, else take the center, else a random legal cell (from the episode's ``rng``)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot
from examples.tic_tac_toe.game import WINNING_LINES


class TicTacToeBot(ScriptedBot):
    def __init__(self) -> None:
        super().__init__()
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        own = np.asarray(obs[0]).reshape(9) > 0.5
        opp = np.asarray(obs[1]).reshape(9) > 0.5
        legal = (np.asarray(mask, dtype=bool).reshape(9) if mask is not None
                 else np.asarray(obs[2]).reshape(9) > 0.5)
        for marks in (own, opp):        # win first, then block
            for line in WINNING_LINES:
                if sum(bool(marks[c]) for c in line) == 2:
                    free = [c for c in line if legal[c]]
                    if free:
                        return int(free[0])
        if legal[4]:
            return 4
        return int(self._rng.choice(np.flatnonzero(legal)))
```

Create `examples/unit_harvest/bots.py`:

```python
"""A scripted unit_harvest player (SP3 spec block 2): the port of ``game.scripted_action`` (carriers
walk home, the other workers walk to the nearest resource) whose base builds whenever the action
mask allows it (``scripted_action`` alone would also try when every unit slot is used)."""
from __future__ import annotations

from typing import Any

import numpy as np

from colosseum.players import ScriptedBot
from examples.unit_harvest.game import scripted_action


class HarvestBot(ScriptedBot):
    def act(self, obs: Any, mask: Any, info: Any) -> dict:
        action = scripted_action(obs)
        can_build = True if mask is None else bool(np.asarray(mask["base"])[1])
        return {"base": 1 if can_build else 0, "workers": np.asarray(action["workers"], dtype=np.int64)}
```

(If `ScriptedBot.__init__` takes no arguments or does not exist, `super().__init__()` is still valid: it is `object.__init__`.)

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_demo_bots.py -q`

Expected: 4 passed.

- [ ] **Step 4: The league example config**

Create `configs/examples/unit_harvest_league.yaml` (the env, networks, algorithm, rollout and learner sections are copied from `configs/examples/unit_harvest.yaml`; `training.total_timesteps` is recalibrated in T6.4):

```yaml
# Unit harvest league (SP3): the trainable agent `main` plays its latest weights, PFSP snapshots and two
# scripted anchors, the game's bot `greedy` and `random`. The competition pipeline (README, docs/LEAGUE_GUIDE.md):
#   colosseum record -c configs/examples/unit_harvest_league.yaml --player greedy --num-matches 300 \
#     --output data/greedy --seed 0
#   colosseum bc -c configs/examples/unit_harvest_league.yaml --agent main --data data/greedy --output bc.pt --epochs 5
#   colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-league \
#     --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 \
#     --set agents.main.kickstart.teacher=greedy
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
  total_timesteps: 160000   # global env steps (T6.4 calibration: about 2 minutes on 8 cores)
  seed: null

matchmaking:
  opponents: {latest: 0.5, snapshots: 0.2, rivals: 0.0, anchors: 0.3}
  anchors: {greedy: 1.0, random: 1.0}

checkpoint:
  interval: 100           # train steps between snapshots
  keep_last: 10
  save_optimizer: true

metrics:
  use_wandb: false
  log_interval: 10
  console_interval_sec: 10.0

agents:
  main:                     # trainable; the pipeline sets init.from / kickstart.teacher with --set
    init: {from: null, critic_warmup_steps: 0}
    kickstart: {teacher: null}
  greedy:
    kind: scripted
    class: examples.unit_harvest.bots.HarvestBot
  random:
    kind: scripted
    class: colosseum.players.RandomBot
```

Run: `.venv/bin/colosseum validate -c configs/examples/unit_harvest_league.yaml` and `.venv/bin/python -m pytest tests/unit/test_example_configs.py -q`

Expected: `Config is valid.` (the printed mix lists `greedy` and `random` as anchors); the example-config tests pass, including the new file.

- [ ] **Step 5: Bandits and the teacher bot**

Create `tests/learning/sp3_bandits.py`:

```python
"""One-step-decision games for the fast SP3 learning tests (spec section 6, «Быстрые учится»)."""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult
from colosseum.players import ScriptedBot

CONTEXTS = 4
OBS_SPACE = gymnasium.spaces.Box(0.0, 1.0, (CONTEXTS,), np.float32)
ACT_SPACE = gymnasium.spaces.Discrete(CONTEXTS)
OFFSET_TARGETS = torch.tensor([(c + 1) % CONTEXTS for c in range(CONTEXTS)])   # OffsetTeacher's action per context


def _one_hot(c: int) -> np.ndarray:
    return np.eye(CONTEXTS, dtype=np.float32)[int(c)]


class SilentBandit(MultiAgentEnv):
    """Solo, ``LENGTH`` decisions per episode on a one-hot context; the reward is always 0, so only a
    kickstart teacher can move the policy."""

    LENGTH = 4
    spec = GameSpec.solo(OBS_SPACE, ACT_SPACE)

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._t = 0
        self._ctx = 0

    def _turn(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0}, obs={0: _one_hot(self._ctx)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._ctx = int(self._rng.integers(CONTEXTS))
        return self._turn({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards={0: 0.0}, episode_over=True)
        self._ctx = int(self._rng.integers(CONTEXTS))
        return self._turn({0: 0.0})


class DuelBandit(MultiAgentEnv):
    """Two seats act simultaneously for ``LENGTH`` steps; each sees its own one-hot context and scores 1
    when it picks it. The default outcome (team score = return) decides the match; a uniformly random
    player scores ``LENGTH / 4`` on average, a perfect one ``LENGTH``."""

    LENGTH = 4
    spec = GameSpec.symmetric(2, OBS_SPACE, ACT_SPACE)

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        self._t = 0
        self._ctx = np.zeros(2, dtype=np.int64)

    def _turn(self, rewards: dict[int, float]) -> StepResult:
        return StepResult(acting={0, 1}, obs={s: _one_hot(self._ctx[s]) for s in (0, 1)}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._ctx = self._rng.integers(CONTEXTS, size=2)
        return self._turn({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        rewards = {s: float(int(actions[s]) == int(self._ctx[s])) for s in (0, 1)}
        self._t += 1
        if self._t >= self.LENGTH:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        self._ctx = self._rng.integers(CONTEXTS, size=2)
        return self._turn(rewards)


class OffsetTeacher(ScriptedBot):
    """Plays ``(context + 1) % 4``: an action no reward of ``SilentBandit`` points to."""

    def act(self, obs: Any, mask: Any, info: Any) -> int:
        return int((int(np.argmax(obs)) + 1) % CONTEXTS)
```

- [ ] **Step 6: Extend `train_in_process`**

In `tests/learning/demo_learning.py`:
- add the imports `from colosseum.players.registry import BotSpec, FixedPlayers`;
- add the field `algorithm_fn: Callable[[], Any] | None = None` (comment: "SP3: builds the algorithm instead of ``APPO(model_fn(), config, ...)``, e.g. ``learner.factory.build_algorithm`` with a kickstart teacher") as the last field of `AgentSetup`;
- replace `train_in_process` with:

```python
def train_in_process(*, env_fn: Callable[[], MultiAgentEnv], agents: Mapping[str, AgentSetup],
                     lineups: Sequence[Lineup], chunk_length: int = 16, batch_chunks: int = 4,
                     max_updates: int = 300, solved: Callable[[dict[str, PolicyModel]], bool],
                     check_every: int = 10, seed: int = 0,
                     last_metrics: dict[str, dict[str, float]] | None = None,
                     fixed_players: FixedPlayers | None = None,
                     teachers: Mapping[str, BotSpec] | None = None,
                     on_chunk: Callable[[TrajectoryChunk], None] | None = None) -> int:
    """Collect with one ``RolloutLoop`` (one env per lineup) and train one algorithm per agent.
    Every update trains every agent on exactly ``batch_chunks`` of its chunks. Returns the number
    of updates after which ``solved(models)`` first held (checked every ``check_every``), or -1.
    ``last_metrics`` (if given) receives each agent's train metrics of the last update.

    SP3: ``fixed_players`` seat scripted / frozen agents (their seats in ``lineups`` use
    ``FIXED_NETWORK_ID`` and never collect); ``teachers`` are scripted kickstart teachers per
    trainable agent (the agent's weight payloads carry ``teacher_active=True``); ``on_chunk`` sees
    every chunk after the payload round trip."""
    torch.manual_seed(seed)
    algos = {a: s.algorithm_fn() if s.algorithm_fn is not None else APPO(s.model_fn(), s.config, s.action_spec,
                                                                           device="cpu")
             for a, s in agents.items()}
    teachers = dict(teachers or {})

    def payload(agent_id: str) -> WeightPayload:
        algo = algos[agent_id]
        weights = WeightPayload.from_model(agent_id, algo.policy_version, algo.model)
        weights.teacher_active = agent_id in teachers
        return weights

    pending: dict[str, list[TrajectoryChunk]] = {a: [] for a in agents}
    latest = {a: payload(a) for a in agents}

    def send(chunk_out) -> None:
        chunk = TrajectoryChunk.from_payload(chunk_out.to_payload())
        if on_chunk is not None:
            on_chunk(chunk)
        pending[chunk.agent_id].append(chunk)

    io = LoopIO(send_chunk=send, poll_weights=lambda agent_id: latest[agent_id])
    loop = RolloutLoop(worker_id=0, env_fn=env_fn, num_envs=len(lineups), chunk_length=chunk_length,
                       agent_ids=list(agents), agent_roles={a: s.roles for a, s in agents.items()},
                       model_factories={a: s.model_fn for a, s in agents.items()}, io=io, lineups=list(lineups),
                       weight_sync_interval=0.0, seed=seed, fixed_players=fixed_players,
                       teachers=teachers or None)
    try:
        for update in range(1, max_updates + 1):
            while any(len(chunks) < batch_chunks for chunks in pending.values()):
                loop.step()
            for agent_id, algo in algos.items():
                batch = pending[agent_id][:batch_chunks]
                del pending[agent_id][:batch_chunks]
                algo.set_progress(update / max_updates)
                metrics = algo.train_step(batch)
                if last_metrics is not None:
                    last_metrics[agent_id] = metrics
                latest[agent_id] = payload(agent_id)
            loop.sync_weights()
            if update % check_every == 0 and solved({a: algo.model for a, algo in algos.items()}):
                return update
    finally:
        loop.close()
    return -1
```

Update the module docstring's first bullet: "``train_in_process``: the real ``RolloutLoop`` and one algorithm per agent (APPO, or ``AgentSetup.algorithm_fn``) in one process, with optional scripted / frozen players and scripted kickstart teachers (SP3), every chunk through ``to_payload``/``from_payload`` (fast tests);".

- [ ] **Step 7: Write the fast learning tests**

Create `tests/learning/test_sp3_fast_learning.py`:

```python
"""Fast SP3 learning checks (spec section 6, «Быстрые учится»): a scripted kickstart teacher (DAgger
labels in the chunk) moves the student to its action on a bandit whose reward is always 0, and a
RandomBot anchor is beaten while only the latest seats collect data."""
from __future__ import annotations

import functools
import time

import pytest
import torch

from colosseum.core.config import AlgorithmConfig, ColosseumConfig
from colosseum.core.registry import build_model, env_spec
from colosseum.core.specs import ActionSpec
from colosseum.core.types import FIXED_NETWORK_ID, Lineup, SeatAssignment
from colosseum.learner.factory import build_algorithm, resolve_teacher
from colosseum.players.registry import BotSpec, FixedPlayers, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from demo_learning import AgentSetup, GreedyPolicy, play, train_in_process, win_rate
from game_learning_envs import make_mlp_model
from sp3_bandits import OFFSET_TARGETS, DuelBandit, SilentBandit

pytestmark = pytest.mark.usefixtures("restore_global_rng")
RANDOM_BOT = BotSpec("colosseum.players.RandomBot", {})


def test_scripted_kickstart_teacher_moves_the_student_to_its_action():
    config = ColosseumConfig.model_validate({
        "env": {"env_class": "sp3_bandits.SilentBandit", "kwargs": {}},
        "networks": {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}},
        "algorithm": {"learning_rate": 3.0e-3, "lr_schedule": "constant", "entropy_coeff": 0.0},
        "kickstart": {"teacher": "teacher", "lambda": 1.0, "decay_steps": 1_000_000},
        "matchmaking": {"anchors": []},
        "agents": {"agent_0": {}, "teacher": {"kind": "scripted", "class": "sp3_bandits.OffsetTeacher"}},
    })
    spec = env_spec(config)
    role = spec.roles["player"]
    agent_config = config.get_agent_config("agent_0")
    teacher = resolve_teacher(config, "agent_0", spec)
    assert teacher is not None and teacher.kind == "scripted" and teacher.bot is not None
    setup = AgentSetup(
        roles=["player"], model_fn=functools.partial(build_model, agent_config, role),
        config=agent_config.algorithm, action_spec=ActionSpec.from_space(role.action_space),
        algorithm_fn=functools.partial(build_algorithm, agent_config, role, spec, device="cpu", teacher=teacher))
    labelled: list[bool] = []

    def solved(models) -> bool:
        model = models["agent_0"]
        with torch.no_grad():
            probs = model.step(torch.eye(4), model.initial_state(4)).dist.log_prob(OFFSET_TARGETS).exp()
        return bool((probs >= 0.9).all())

    start = time.monotonic()
    updates = train_in_process(
        env_fn=SilentBandit, agents={"agent_0": setup},
        lineups=[Lineup("solo", [SeatAssignment("agent_0")]) for _ in range(8)], max_updates=300, solved=solved,
        teachers={"agent_0": teacher.bot},
        on_chunk=lambda c: labelled.append(c.has_teacher is not None and bool(c.has_teacher.any())))
    assert any(labelled), "no chunk carried teacher labels"
    assert updates != -1, "the student did not follow the scripted teacher within 300 updates (the reward is 0)"
    assert time.monotonic() - start < 60


def test_random_bot_anchor_is_beaten_while_only_latest_seats_collect():
    spec = DuelBandit.spec
    action_spec = ActionSpec.from_space(spec.roles["player"].action_space)
    config = AlgorithmConfig(learning_rate=3e-3, lr_schedule="constant", entropy_coeff=0.003)
    setup = AgentSetup(["player"], lambda: make_mlp_model(obs_dim=4, num_actions=4, hidden=32), config, action_spec)
    fixed = FixedPlayers(bots={"random": RANDOM_BOT}, frozen={}, roles={"random": ("player",)})
    seats = [SeatAssignment("agent_0"), SeatAssignment("random", FIXED_NETWORK_ID, collect=False)]
    lineups = [Lineup("2p", seats if e % 2 == 0 else seats[::-1]) for e in range(8)]
    random_player = ScriptedPlayer(functools.partial(make_bot, RANDOM_BOT, spec))
    collected: set[str] = set()

    def solved(models) -> bool:
        greedy = GreedyPolicy(models["agent_0"], action_spec)
        eval_seats = [SeatAssignment("g"), SeatAssignment("random")]
        results = play(DuelBandit, {"g": greedy, "random": random_player},
                       [Lineup("2p", eval_seats if m % 2 == 0 else eval_seats[::-1]) for m in range(64)])
        return win_rate(results, "g") >= 0.9

    start = time.monotonic()
    updates = train_in_process(env_fn=DuelBandit, agents={"agent_0": setup}, lineups=lineups, max_updates=300,
                               solved=solved, fixed_players=fixed, on_chunk=lambda c: collected.add(c.agent_id))
    assert collected == {"agent_0"}, collected          # the anchor's seats never produce chunks
    assert updates != -1, "RandomBot not beaten in 90% of duels within 300 updates"
    assert time.monotonic() - start < 60
```

Run: `.venv/bin/python -m pytest tests/learning/test_sp3_fast_learning.py tests/learning/test_sp2_fast_learning.py tests/learning/test_sp2_units_coop_learning.py -q --durations=5`

Expected: all pass (the SP2 fast tests guard the `train_in_process` change), each SP3 test well under 20 s. If the teacher test does not converge, check first that `has_teacher` labels arrive (`labelled`) and that the teacher loss reaches APPO (T4.5); do not raise the learning rate to hide a wiring bug.

- [ ] **Step 8: The pipeline kit**

Create `tests/pipeline_kit.py`:

```python
"""The SP3 competition pipeline as test helpers: ``record`` -> ``bc`` -> ``train`` (``init`` with
critic warm-up, kickstart, anchors) -> ``eval`` (spec section 3, criterion 3).

Used by the fast tic-tac-toe smoke (``tests/integration/test_sp3_pipeline_smoke.py``), the slow
unit_harvest test (``tests/learning/test_sp3_pipeline_slow.py``) and
``scripts/pipeline_vs_scratch.py``. Every step is the real CLI: in a subprocess through
``cli_runner.run_in_session`` (default) or in-process through click's ``CliRunner``
(``in_process=True``); ``train`` always runs in a subprocess (it starts processes).

The pipeline config is an example config with these agents (``pipeline_config``):
- ``main``: the trainable agent (``init`` and ``kickstart`` are added for the train step);
- ``PipelineGame.bot``: the game's scripted bot (recorded, an anchor, possibly the DAgger teacher);
- ``random``: ``colosseum.players.RandomBot``;
- ``bc_net``: the frozen BC weights, once they exist.
SP2 matchmaking knobs are dropped and ``pool_size`` becomes ``keep_last``, so old and new knobs
never meet (that would be a ConfigError).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cli_runner import REPO_ROOT, TrainRun, run_in_session, run_train

MAIN = "main"
RANDOM = "random"
BC_NET = "bc_net"
TRAINED = "trained"
RANDOM_BOT_CLASS = "colosseum.players.RandomBot"
SP2_MATCHMAKING_KNOBS = ("mode", "self_play_ratio", "latest_prob", "pfsp_exponent")
_CKPT_RE = re.compile(r"ckpt_v(\d+)")

# Short training of the fast smoke (cli_runner.TINY, with keep_last instead of SP2's pool_size).
SMOKE_SETTINGS: dict[str, Any] = {
    "training.total_timesteps": 3000, "rollout.num_workers": 1, "rollout.envs_per_worker": 8,
    "rollout.chunk_length": 16, "rollout.weight_sync_interval_sec": 0.5, "rollout.match_refresh_interval_sec": 1.0,
    "learner.batch_chunks": 2, "learner.queue_size": 16, "checkpoint.interval": 20, "checkpoint.keep_last": 5,
    "metrics.log_interval": 1, "metrics.console_interval_sec": 1.0,
}


@dataclass(frozen=True)
class PipelineGame:
    config: str                                     # configs/examples/<config>.yaml
    bot: str                                        # agent id of the game's bot in the pipeline config
    bot_class: str                                  # its ScriptedBot (added when the config has no such agent)
    layout: str                                     # the layout whose eval rows are returned
    opponents: Mapping[str, float] | None = None    # matchmaking.opponents (None: the config's own)


@dataclass(frozen=True)
class PipelineSettings:
    record_matches: int
    bc_epochs: int
    critic_warmup_steps: int
    kickstart_teacher: str                          # an agent id (scripted: DAgger labels; frozen: KL)
    kickstart_lambda: float
    kickstart_decay_steps: int
    eval_matches: int
    train_sets: Mapping[str, Any] = field(default_factory=dict)   # --set of the train step
    init_from: str = BC_NET                         # BC_NET (the frozen agent's name) or "path" (bc.pt)


TIC_TAC_TOE = PipelineGame("tic_tac_toe", "ttt_bot", "examples.tic_tac_toe.bots.TicTacToeBot", "2p",
                           opponents={"latest": 0.4, "snapshots": 0.1, "rivals": 0.0, "anchors": 0.5})
UNIT_HARVEST = PipelineGame("unit_harvest_league", "greedy", "examples.unit_harvest.bots.HarvestBot", "2p")

TIC_TAC_TOE_SMOKE = PipelineSettings(record_matches=20, bc_epochs=2, critic_warmup_steps=2, kickstart_teacher=BC_NET,
                                     kickstart_lambda=1.0, kickstart_decay_steps=50, eval_matches=4,
                                     train_sets=SMOKE_SETTINGS)
# T6.4 calibration (ruling): budgets of the slow unit_harvest pipeline test.
UNIT_HARVEST_SETTINGS = PipelineSettings(record_matches=300, bc_epochs=5, critic_warmup_steps=30,
                                         kickstart_teacher="greedy", kickstart_lambda=1.0, kickstart_decay_steps=300,
                                         eval_matches=60, train_sets={"rollout.num_workers": 2}, init_from="path")


def pipeline_config(game: PipelineGame, *, bc_path: Path | None = None, init: Mapping[str, Any] | None = None,
                    kickstart: Mapping[str, Any] | None = None) -> dict:
    """The raw pipeline config (module docstring)."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "examples" / f"{game.config}.yaml").read_text())
    agents = {name: dict(entry or {}) for name, entry in (raw.get("agents") or {}).items()}
    agents.setdefault(MAIN, {})
    agents.setdefault(game.bot, {"kind": "scripted", "class": game.bot_class})
    agents.setdefault(RANDOM, {"kind": "scripted", "class": RANDOM_BOT_CLASS})
    if bc_path is not None:
        agents[BC_NET] = {"kind": "frozen", "path": str(bc_path)}
    if init is not None:
        agents[MAIN]["init"] = dict(init)
    if kickstart is not None:
        agents[MAIN]["kickstart"] = dict(kickstart)
    raw["agents"] = agents
    matchmaking = {k: v for k, v in (raw.get("matchmaking") or {}).items() if k not in SP2_MATCHMAKING_KNOBS}
    if game.opponents is not None:
        matchmaking["opponents"] = dict(game.opponents)
    raw["matchmaking"] = matchmaking
    checkpoint = dict(raw.get("checkpoint") or {})
    if "pool_size" in checkpoint:
        checkpoint["keep_last"] = checkpoint.pop("pool_size")
    raw["checkpoint"] = checkpoint
    return raw


def write_config(path: Path, raw: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(raw), sort_keys=False))
    return path


@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str


def cli(args: Sequence[Any], *, in_process: bool = False, timeout: float = 900.0) -> CliResult:
    """``colosseum <args>``: in a subprocess (own session, killed on exit) or in-process."""
    args = [str(a) for a in args]
    if in_process:
        from click.testing import CliRunner

        from colosseum.cli import main

        result = CliRunner().invoke(main, args)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return CliResult(result.exit_code, result.stdout, result.stderr)
    proc = run_in_session([sys.executable, "-m", "colosseum", *args], timeout)
    return CliResult(proc.returncode, proc.stdout, proc.stderr)


def _ok(result: CliResult, what: str) -> CliResult:
    assert result.code == 0, (f"{what} failed with exit code {result.code}:\n{result.stdout[-2000:]}\n"
                              f"{result.stderr[-3000:]}")
    return result


def record(config: Path, player: str, out: Path, *, num_matches: int, seed: int, in_process: bool = False) -> dict:
    _ok(cli(["record", "-c", config, "--player", player, "--num-matches", num_matches, "--output", out,
             "--seed", seed], in_process=in_process), "record")
    return json.loads((Path(out) / "record.json").read_text())


def bc(config: Path, data: Path, out: Path, *, epochs: int, agent: str = MAIN, in_process: bool = False) -> Path:
    _ok(cli(["bc", "-c", config, "--agent", agent, "--data", data, "--output", out, "--epochs", epochs],
            in_process=in_process), "bc")
    return Path(out)


def train(config: Path, workdir: Path, name: str, sets: Mapping[str, Any], *, timeout: float = 900.0) -> TrainRun:
    """``colosseum train`` into ``<workdir>/runs/<name>`` with the config's own sizes plus ``sets``."""
    run = run_train(config, workdir, name, dict(sets), timeout=timeout, tiny=False)
    assert run.returncode == 0, run.stderr[-3000:]
    return run


def newest_checkpoint(root: Path, agent_id: str) -> Path:
    agent_dir = Path(root) / "checkpoints" / agent_id
    versions = {int(m.group(1)): d for d in agent_dir.iterdir()
                if d.is_dir() and (m := _CKPT_RE.fullmatch(d.name))} if agent_dir.is_dir() else {}
    assert versions, f"no checkpoints of {agent_id} in {root}"
    return versions[max(versions)]


def evaluate(config: Path, checkpoint: Path, opponents: Sequence[str], out: Path, *, layout: str, num_matches: int,
             seed: int = 12345, in_process: bool = False) -> dict[str, dict]:
    """``colosseum eval -a trained=<checkpoint> -a <opponent> ... --deterministic`` (opponents by name);
    returns the ``wdl`` rows "trained vs <opponent>" of ``layout``, keyed by opponent."""
    args: list[Any] = ["eval", "-c", config, "-a", f"{TRAINED}={checkpoint}"]
    for name in opponents:
        args += ["-a", name]
    args += ["--layout", layout, "--num-matches", num_matches, "--seed", seed, "--deterministic", "--output", out]
    _ok(cli(args, in_process=in_process), "eval")
    report = json.loads(Path(out).read_text())
    return {row["agent_b"]: row for row in report["layouts"][layout]["pairs"] if row["agent_a"] == TRAINED}


@dataclass
class PipelineRun:
    record: dict | None              # record.json (None when bc_path was given)
    bc_path: Path
    run: TrainRun
    checkpoint: Path
    pairs: dict[str, dict]           # opponent -> eval row "trained vs opponent"
    seconds: dict[str, float]


def run_pipeline(game: PipelineGame, workdir: Path, *, seed: int, settings: PipelineSettings, warm_start: bool = True,
                 bc_path: Path | None = None, name: str = "pipeline", in_process: bool = False) -> PipelineRun:
    """record -> bc (both skipped when ``bc_path`` is given) -> train -> eval against the bot, RandomBot
    and the frozen BC net. ``warm_start=False`` trains in the same league without ``init`` and
    kickstart (the "pipeline vs scratch" comparison)."""
    workdir = Path(workdir)
    seconds: dict[str, float] = {}
    record_json = None
    if bc_path is None:
        record_cfg = write_config(workdir / "record.yaml", pipeline_config(game))
        start = time.monotonic()
        record_json = record(record_cfg, game.bot, workdir / "data", num_matches=settings.record_matches, seed=seed,
                             in_process=in_process)
        seconds["record"] = time.monotonic() - start
        start = time.monotonic()
        bc_path = bc(record_cfg, workdir / "data", workdir / "bc.pt", epochs=settings.bc_epochs, in_process=in_process)
        seconds["bc"] = time.monotonic() - start
    init = kickstart = None
    if warm_start:
        init = {"from": BC_NET if settings.init_from == BC_NET else str(bc_path),
                "critic_warmup_steps": settings.critic_warmup_steps}
        kickstart = {"teacher": settings.kickstart_teacher, "lambda": settings.kickstart_lambda,
                     "decay_steps": settings.kickstart_decay_steps}
    train_cfg = write_config(workdir / f"{name}.yaml",
                             pipeline_config(game, bc_path=bc_path, init=init, kickstart=kickstart))
    start = time.monotonic()
    run = train(train_cfg, workdir, name, {"training.seed": seed, **settings.train_sets})
    seconds["train"] = time.monotonic() - start
    checkpoint = newest_checkpoint(run.root, MAIN)
    start = time.monotonic()
    pairs = evaluate(train_cfg, checkpoint, [game.bot, RANDOM, BC_NET], workdir / f"{name}-eval.json",
                     layout=game.layout, num_matches=settings.eval_matches, in_process=in_process)
    seconds["eval"] = time.monotonic() - start
    return PipelineRun(record_json, Path(bc_path), run, checkpoint, pairs, seconds)
```

Add `"pipeline_kit"` and `"sp3_bandits"` to `known-first-party` in `pyproject.toml` (alphabetical order of the list).

- [ ] **Step 9: The fast pipeline smoke**

Create `tests/integration/test_sp3_pipeline_smoke.py`:

```python
"""Fast smoke of the SP3 competition pipeline on tic-tac-toe through the CLI (spec section 3,
criterion 3; section 6): record the game's bot -> bc -> train with init from the frozen BC net
(critic warm-up), a neural kickstart teacher and scripted / frozen anchors -> eval against the bot,
RandomBot and the frozen BC net. Short budgets, no thresholds."""
from __future__ import annotations

import pytest

import pipeline_kit as kit

pytestmark = pytest.mark.usefixtures("restore_root_logging")


def test_tic_tac_toe_pipeline_smoke(tmp_path):
    game = kit.TIC_TAC_TOE
    result = kit.run_pipeline(game, tmp_path, seed=0, settings=kit.TIC_TAC_TOE_SMOKE, in_process=True)
    assert result.record["roles"]["player"]["decisions"] > 0
    assert (tmp_path / "data" / "player" / "part-00000.pt").is_file() and result.bc_path.is_file()
    assert result.run.records("train"), "the learner never trained"
    elo = result.run.ratings()["layouts"]["2p"]["elo"]
    assert {game.bot, kit.RANDOM, kit.BC_NET} & set(elo), f"no anchor played in training: {sorted(elo)}"
    assert set(result.pairs) == {game.bot, kit.RANDOM, kit.BC_NET}
    assert all(row["n"] == 4 for row in result.pairs.values())
```

Run: `.venv/bin/python -m pytest tests/integration/test_sp3_pipeline_smoke.py -q --durations=1`

Expected: 1 passed. Note the duration: if it exceeds 20 s, lower `SMOKE_SETTINGS["training.total_timesteps"]` to 2000 (and say so in the commit body); if it is still above 20 s, report to the controller (the spec requires the smoke in the fast suite).

- [ ] **Step 10: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 11: Commit and push**

```bash
git add examples/tic_tac_toe/bots.py examples/unit_harvest/bots.py configs/examples/unit_harvest_league.yaml \
  tests/pipeline_kit.py tests/learning/sp3_bandits.py tests/learning/demo_learning.py pyproject.toml \
  tests/unit/test_sp3_demo_bots.py tests/integration/test_sp3_pipeline_smoke.py tests/learning/test_sp3_fast_learning.py
git commit -m "feat: demo bots, the pipeline kit with a tic-tac-toe smoke, and fast SP3 learning tests"
git push origin sp3-league
```

---
### Task T6.3: team_tag with a `RandomBot` anchor, single-run slow test

Spec block 9, criterion 4. The brainstorm measurement showed that more diverse past opponents do not cure the passive, draw-seeking policy (one failure out of six seeds in each variant); the cure tried here is an anchor against which aggression pays: a team of uniformly random legal players. The example config moves to the v3 form with a scripted agent `random` (the trainable `agent_0` is added implicitly, spec block 1); the anchors share is measured with a rule fixed below; the slow test becomes a single run; the fallback is a draw penalty inside the env.

**Files:**
- Modify: `configs/examples/team_tag.yaml` (v3 form, `random` agent, anchors share)
- Create: `scripts/team_tag_anchors.py` (measurement driver; also used by T6.7)
- Modify: `tests/learning/test_demo_learning_slow.py` (team_tag: one run; the best-of-two helper and `TEAM_TAG_RETRY_SEED_OFFSET` removed)
- Test: `tests/unit/test_sp3_team_tag_anchors.py` (the decision rule and the config shape)
- Create: `docs/benchmarks/team-tag-anchors.jsonl` (raw rows of the measurement)
- Modify (only on the fallback path, Step 8): `examples/team_tag/game.py` (`draw_reward`), `tests/contract/test_demo_team_tag.py`

**Interfaces:**
- Consumes: `demo_learning.train_example`, `greedy_agent`, `play`, `random_model`, `team_lineups`, `team_of`; `env_spec`; the v3 `matchmaking.opponents` / `anchors` (T3.1); `ScriptedAgent` with `class: colosseum.players.RandomBot` (T1.1, T1.2).
- Produces: `scripts/team_tag_anchors.py` with `choose_anchor_share(rows, *, draw_reward=None, threshold=0.80, tie=0.02, min_seeds=6) -> tuple[float | None, list[str]]`, `choose_draw_reward(rows, *, anchors, threshold=0.80, tie=0.02, min_seeds=6) -> tuple[float | None, list[str]]`, `wdl(results, agent)`, CLI `--anchors A ... --seeds S ... [--draw-reward D ...] --jsonl PATH --run-dir DIR`.

**Decision rule (fixed before the measurement).** For each anchors share `a` (with `snapshots` 0.2, `latest` 0.8 − a, `rivals` 0), six seeds: a share *passes* if its minimum win rate over the six seeds is ≥ 0.80. Among passing shares choose the one with the highest minimum; a smaller share whose minimum is within 0.02 of the best minimum wins the tie (fewer Python bot calls, closer to the default mix). No passing share → extend the grid once with `a = 0.5` (six more runs); still none → the fallback (Step 8).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_team_tag_anchors.py`:

```python
"""team_tag anchors (spec block 9, T6.3): the pre-fixed decision rule of scripts/team_tag_anchors.py
and the v3 shape of configs/examples/team_tag.yaml. No training here."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from colosseum.core.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "team_tag_anchors.py"
_spec = importlib.util.spec_from_file_location("team_tag_anchors", _SCRIPT)
tta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta)


def _rows(wins: dict[float, list[float]], draw_reward=None) -> list[dict]:
    return [{"anchors": a, "draw_reward": draw_reward, "seed": s, "win": w}
            for a, ws in wins.items() for s, w in enumerate(ws)]


def test_the_highest_minimum_among_passing_shares_wins():
    rows = _rows({0.1: [0.9, 0.7, 0.9, 0.9, 0.9, 0.9], 0.2: [0.85, 0.82, 0.9, 0.9, 0.9, 0.9],
                  0.3: [0.88, 0.87, 0.9, 0.9, 0.9, 0.9]})
    choice, detail = tta.choose_anchor_share(rows)
    assert choice == 0.3
    assert any(line.startswith("anchors 0.1: min 0.700") for line in detail)


def test_a_smaller_share_within_the_tie_margin_wins():
    rows = _rows({0.2: [0.85] * 6, 0.3: [0.86] * 6})
    assert tta.choose_anchor_share(rows)[0] == 0.2


def test_no_passing_share_and_too_few_seeds_give_none():
    assert tta.choose_anchor_share(_rows({0.1: [0.9, 0.9, 0.9, 0.9, 0.9, 0.79]}))[0] is None
    assert tta.choose_anchor_share(_rows({0.1: [0.95] * 5}))[0] is None              # needs 6 seeds
    assert tta.choose_anchor_share(_rows({0.1: [0.95] * 6}, draw_reward=-0.5))[0] is None   # other draw reward


def test_the_draw_reward_fallback_prefers_the_mildest_passing_penalty():
    rows = ([{"anchors": 0.3, "draw_reward": d, "seed": s, "win": w} for s in range(6)
             for d, w in ((-0.25, 0.86), (-0.5, 0.87))]
            + [{"anchors": 0.1, "draw_reward": -0.25, "seed": s, "win": 0.99} for s in range(6)])
    assert tta.choose_draw_reward(rows, anchors=0.3)[0] == -0.25


def test_team_tag_example_is_v3_with_a_random_bot_anchor():
    path = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
    raw = yaml.safe_load(path.read_text())
    assert not set(raw["matchmaking"]) & {"mode", "self_play_ratio", "latest_prob", "pfsp_exponent"}
    assert "pool_size" not in raw["checkpoint"]
    config = load_config(path)
    assert config.get_trainable_agent_ids() == ["agent_0"]          # implicit, the agents section has only the bot
    assert config.fixed_agent_ids() == ["random"]
    assert config.agent_entry("random").class_path == "colosseum.players.RandomBot"
    assert config.matchmaking.anchors == ["random"]
    shares = config.matchmaking.opponents
    assert shares.anchors > 0 and shares.snapshots == 0.2 and shares.rivals == 0.0
    assert abs(shares.latest + shares.anchors - 0.8) < 1e-9
```

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_team_tag_anchors.py -q`

Expected: collection error (`scripts/team_tag_anchors.py` does not exist).

- [ ] **Step 2: The measurement driver**

Create `scripts/team_tag_anchors.py`:

```python
"""team_tag with RandomBot anchors (SP3 spec block 9; T6.3 calibration, T6.7 acceptance).

For every (anchors share, draw reward, seed) the script trains ``configs/examples/team_tag.yaml``
with ``colosseum train`` (2 workers, the config's budget), then plays the newest checkpoint greedily
as a whole team against a team of uniformly random legal players (200 matches, the trained team is
team ``m % 2``), exactly like the slow test. The shares are set with ``--set``:
``matchmaking.opponents.anchors = a`` and ``matchmaking.opponents.latest = 0.8 - a`` (``snapshots``
stays 0.2, ``rivals`` 0). ``--draw-reward`` (the spec's fallback) sets the env's ``draw_reward``.

One JSON line per run is appended to ``--jsonl`` (win / draw / loss rates, mean episode length,
training seconds, env steps per second); finished (anchors, draw_reward, seed) rows are skipped on a
rerun, so an interrupted measurement continues where it stopped.

Decision rules (T6.3, fixed before the measurement):
- ``choose_anchor_share``: a share passes if its minimum win rate over ``min_seeds`` seeds is
  >= ``threshold`` (0.80); the passing share with the highest minimum is chosen, and a smaller
  share within ``tie`` (0.02) of that minimum wins the tie;
- ``choose_draw_reward`` (fallback): the same rule over draw rewards at one anchors share; the
  mildest penalty (closest to 0) wins the tie.

Usage (the evaluation imports the test kit, so ``tests/`` and ``tests/learning`` go on ``sys.path``)::

    .venv/bin/python scripts/team_tag_anchors.py --anchors 0.1 0.2 0.3 --seeds 0 1 2 3 4 5 \\
        --jsonl docs/benchmarks/team-tag-anchors.jsonl --run-dir /tmp/colosseum-team-tag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CONFIG = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
THRESHOLD = 0.80
TIE = 0.02
MIN_SEEDS = 6
LATEST_PLUS_ANCHORS = 0.8
EVAL_MATCHES = 200


def wdl(results, agent: str) -> tuple[float, float, float]:
    """Win / draw / loss rates of ``agent``'s team (two-team matches)."""
    from demo_learning import team_of

    w = d = losses = 0
    for r in results:
        ranks = {t.team: t.rank for t in r.teams}
        mine = team_of(r, agent)
        (other,) = set(ranks) - {mine}
        if ranks[mine] < ranks[other]:
            w += 1
        elif ranks[mine] == ranks[other]:
            d += 1
        else:
            losses += 1
    n = len(results)
    return w / n, d / n, losses / n


def _choose(groups: dict[float, list[float]], *, threshold: float, tie: float, min_seeds: int,
            prefer_small: bool) -> float | None:
    passing = {key: min(wins) for key, wins in groups.items() if len(wins) >= min_seeds and min(wins) >= threshold}
    if not passing:
        return None
    best = max(passing.values())
    near = [key for key, low in passing.items() if low >= best - tie]
    return min(near) if prefer_small else max(near)


def _detail(name: str, groups: dict[float, list[float]]) -> list[str]:
    return [f"{name} {key}: min {min(v):.3f}, mean {sum(v) / len(v):.3f}, n {len(v)}"
            for key, v in sorted(groups.items())]


def choose_anchor_share(rows: Sequence[dict], *, draw_reward: float | None = None, threshold: float = THRESHOLD,
                        tie: float = TIE, min_seeds: int = MIN_SEEDS) -> tuple[float | None, list[str]]:
    """The T6.3 rule over anchors shares (module docstring); rows of other draw rewards are ignored."""
    groups: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("draw_reward") == draw_reward and "win" in r:
            groups[float(r["anchors"])].append(float(r["win"]))
    return (_choose(groups, threshold=threshold, tie=tie, min_seeds=min_seeds, prefer_small=True),
            _detail("anchors", groups))


def choose_draw_reward(rows: Sequence[dict], *, anchors: float, threshold: float = THRESHOLD, tie: float = TIE,
                       min_seeds: int = MIN_SEEDS) -> tuple[float | None, list[str]]:
    """The fallback rule over draw rewards at one anchors share; the mildest penalty wins the tie."""
    groups: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("draw_reward") is not None and float(r["anchors"]) == anchors and "win" in r:
            groups[float(r["draw_reward"])].append(float(r["win"]))
    return (_choose(groups, threshold=threshold, tie=tie, min_seeds=min_seeds, prefer_small=False),
            _detail("draw_reward", groups))


def train_and_eval(anchors: float, draw_reward: float | None, seed: int, run_dir: Path) -> dict:
    import yaml

    from colosseum.core.registry import env_spec
    from demo_learning import greedy_agent, play, random_model, team_lineups, train_example

    sets: dict = {"rollout.num_workers": 2, "training.seed": seed, "matchmaking.opponents.anchors": anchors,
                  "matchmaking.opponents.latest": round(LATEST_PLUS_ANCHORS - anchors, 6)}
    if draw_reward is not None:
        kwargs = dict(yaml.safe_load(CONFIG.read_text())["env"]["kwargs"])
        kwargs["draw_reward"] = draw_reward
        sets["env.kwargs"] = json.dumps(kwargs)
    parent = run_dir / f"a{anchors}-d{draw_reward}-s{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    run = train_example("team_tag", parent, sets, timeout=1800)
    trained, role = greedy_agent(run, "agent_0")
    spec = env_spec(run.config)
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   team_lineups(spec.teams("2v2"), "2v2", "trained", "random", EVAL_MATCHES))
    w, d, losses = wdl(results, "trained")
    return {"anchors": anchors, "draw_reward": draw_reward, "seed": seed, "win": w, "draw": d, "loss": losses,
            "mean_len": sum(r.episode_length for r in results) / len(results), "train_sec": round(run.elapsed, 1),
            "env_steps_per_s": round(run.env_steps_per_sec())}


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["anchors"], r["draw_reward"], r["seed"]) for r in rows if "win" in r}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchors", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--draw-reward", type=float, nargs="+", default=None)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/colosseum-team-tag"))
    args = parser.parse_args()
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = _done(args.jsonl)
    failures = 0
    for draw_reward in args.draw_reward or [None]:
        for anchors in args.anchors:
            for seed in args.seeds:
                if (anchors, draw_reward, seed) in done:
                    continue
                start = time.monotonic()
                try:
                    row = train_and_eval(anchors, draw_reward, seed, args.run_dir)
                except Exception as e:  # noqa: BLE001 - keep measuring; the row records the failure
                    row = {"anchors": anchors, "draw_reward": draw_reward, "seed": seed,
                           "error": f"{type(e).__name__}: {str(e)[-1500:]}"}
                    failures += 1
                row["wall_sec"] = round(time.monotonic() - start, 1)
                with args.jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    for draw_reward in args.draw_reward or [None]:
        choice, detail = choose_anchor_share(rows, draw_reward=draw_reward)
        print(f"draw_reward {draw_reward}: anchors share by the rule: {choice}")
        for line in detail:
            print("  " + line)
    if args.draw_reward:
        for anchors in args.anchors:
            choice, detail = choose_draw_reward(rows, anchors=anchors)
            print(f"anchors {anchors}: draw reward by the rule: {choice}")
            for line in detail:
                print("  " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: `team_tag.yaml` in the v3 form (provisional share 0.2)**

In `configs/examples/team_tag.yaml`, replace the `matchmaking:` and `checkpoint:` sections and add an `agents:` section at the end:

```yaml
matchmaking:
  teammates: self           # homogeneous teams
  # SP3 (spec block 9): a team of uniformly random legal players as an anchor; against it aggression
  # pays, which the passive draw-seeking self-play policy lacks. Share: T6.3 ruling (measurement in
  # docs/benchmarks.md, «team_tag: доля якоря»).
  opponents: {latest: 0.6, snapshots: 0.2, rivals: 0.0, anchors: 0.2}
  anchors: [random]

checkpoint:
  interval: 100           # train steps between snapshots
  keep_last: 10
  save_optimizer: true
```

```yaml
agents:                   # only a scripted agent: the trainable agent_0 is added implicitly
  random:
    kind: scripted
    class: colosseum.players.RandomBot
```

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_team_tag_anchors.py tests/unit/test_example_configs.py tests/integration/test_demo_roles_smoke.py -q`

Expected: all pass (the smoke trains team_tag for 3000 steps with the anchor).

- [ ] **Step 4: Run the measurement**

Preconditions: nothing else CPU-heavy runs; `uptime` load below 1.0.

```bash
mkdir -p /tmp/colosseum-team-tag
nohup .venv/bin/python scripts/team_tag_anchors.py --anchors 0.1 0.2 0.3 --seeds 0 1 2 3 4 5 \
  --jsonl docs/benchmarks/team-tag-anchors.jsonl --run-dir /tmp/colosseum-team-tag \
  > /tmp/colosseum-team-tag/driver.log 2>&1 &
```

18 runs at about 3 minutes each (training about 160 s, evaluation a few seconds): about one hour. Poll `tail -3 /tmp/colosseum-team-tag/driver.log` every few minutes. A row with `"error"` is investigated (its run log under `/tmp/colosseum-team-tag/a<A>-d<D>-s<S>/runs/team_tag/logs/`), fixed if it is a bug of this task, and rerun (delete the error line from the jsonl first).

Record for the report: every row (win / draw / loss, mean length, training seconds, env steps per second); the rule's output (the last lines of the log).

- [ ] **Step 5: Apply the rule**

- **A share passed:** set it in `team_tag.yaml` (`anchors: <a>`, `latest: <0.8 - a>`). If the chosen share is not 0.2, rerun Step 3's command. The controller records the ruling: "team_tag: `opponents.anchors = <a>` — the T6.3 rule over 18 runs (minimum win rates per share: …) — cost if wrong: a passive run at acceptance (T6.7 runs 6 fresh seeds)".
- **No share passed:** extend the grid once: the Step 4 command with `--anchors 0.5` (same jsonl; 6 runs, 20 minutes), then apply the rule to all four shares.
- **Still none:** go to Step 8 (fallback).

Also record the mean training time per run: if it exceeds 200 s (the anchor's Python bot calls slow the workers), the controller decides by ruling between keeping the budget (the slow test then takes longer than SP2's 3 minutes) and lowering `training.total_timesteps` (which needs a confirmation run of the rule's share on 6 seeds).

- [ ] **Step 6: The slow test becomes a single run**

In `tests/learning/test_demo_learning_slow.py`, delete `TEAM_TAG_RETRY_SEED_OFFSET`, `_team_tag_win_rate` and `test_team_tag_team_beats_random_team_80_percent`, and add in their place:

```python
def test_team_tag_team_beats_random_team_80_percent(tmp_path):
    """One run (SP3 spec block 9, criterion 4): the example config trains against a RandomBot anchor
    (share: T6.3 ruling), which removes SP2's passive, draw-seeking failure mode, so the SP2
    best-of-two retry is gone. The threshold is the spec's."""
    run = train_example("team_tag", tmp_path, TWO_WORKERS)
    trained, role = greedy_agent(run, "agent_0")
    spec = env_spec(run.config)
    results = play(run.env_fn(), {"trained": trained, "random": random_model(role)},
                   team_lineups(spec.teams("2v2"), "2v2", "trained", "random", 200))
    rate = win_rate(results, "trained")
    draws = sum(len({t.rank for t in r.teams}) == 1 for r in results) / len(results)
    mean_len = sum(r.episode_length for r in results) / len(results)
    _report("team_tag", run, win_rate=rate, draws=draws, mean_len=mean_len)
    assert rate >= 0.80
```

Update the module docstring: replace "Thresholds are the spec's; changing one needs a ruling with measurements (T8.3 Step 9), never a silent edit." with "Thresholds are the spec's; changing one needs a ruling with measurements (SP2 T8.3 Step 9, SP3 T6.3/T6.4), never a silent edit. team_tag trains once since SP3 (RandomBot anchor, spec block 9)."

Run: `.venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -k team_tag -v -s 2>&1 | grep -E "^\[|PASSED|FAILED"`

Expected: PASSED with the printed win rate ≥ 0.80 (one run, about 3 minutes). A failure here, after the rule passed on six seeds, is reported to the controller with the printed numbers (it decides: rerun once as a check of noise, or more seeds).

- [ ] **Step 7: Benchmarks section**

Append to `docs/benchmarks.md`:

```markdown
## team_tag: доля якоря `RandomBot` (SP3, блок 9)

`scripts/team_tag_anchors.py`: `configs/examples/team_tag.yaml` (v3, scripted-агент `random`), для каждой доли якоря `a` — `opponents = {latest: 0.8 − a, snapshots: 0.2, rivals: 0, anchors: a}`, 6 сидов, бюджет конфига (400 000 шагов сред), 2 воркера. Оценка — как в медленном тесте: greedy-команда против команды случайных игроков, 200 партий. Правило выбора (T6.3, задано до замера): доля проходит, если минимум win-rate по 6 сидам ≥ 0.80; из прошедших — с наибольшим минимумом, при разнице минимумов ≤ 0.02 — меньшая доля. Сырые данные: [`benchmarks/team-tag-anchors.jsonl`](benchmarks/team-tag-anchors.jsonl). Дата: <date>, машина: <nproc, RAM>.

| anchors | seed | win | draw | loss | средняя длина | обучение, с |
|---|---|---|---|---|---|---|
<one row per run, from the jsonl>

Итог по долям: <the rule's detail lines: min / mean per share>. Выбрана доля **<a>** (ruling T6.3). <one sentence on draws and episode length compared with the brainstorm's passive failures: 30–35 % draws, episodes at the 30-step limit>.
```

Fill every `<...>` from the jsonl and the log (the date with `date +%F`, the machine with `nproc` and `free -g`).

- [ ] **Step 8: Fallback — a draw penalty in the env (only if Step 5 found no passing share)**

1. In `examples/team_tag/game.py`, add the keyword `draw_reward: float = 0.0` to `TeamTagGame.__init__` (store it as `self.draw_reward`; docstring: "``draw_reward``: the reward of every seat when the match ends in a draw (0 = SP2's rule); a negative value makes waiting for the time limit cost something (SP3 spec block 9 fallback)"), and in `step` replace the `final = ...` line with

```python
        if active[0] == active[1]:
            final = {0: self.draw_reward, 1: self.draw_reward}
        else:
            final = {0: float(np.sign(active[0] - active[1])), 1: float(np.sign(active[1] - active[0]))}
```

2. Add to `tests/contract/test_demo_team_tag.py`:

```python
def test_draw_reward_is_paid_to_every_seat_of_a_drawn_match():
    env = TeamTagGame(size=6, view_radius=3, max_steps=1, draw_reward=-0.25)
    env.reset(0, "2v2")
    result = env.step({seat: 0 for seat in range(4)})          # everybody stays: a draw at the step limit
    assert result.episode_over and result.rewards == {seat: -0.25 for seat in range(4)}
```

3. Measure `--draw-reward -0.25 -0.5` at the share with the highest minimum of Step 4/5 (`--anchors <that share>`, same jsonl, 12 runs), apply `choose_draw_reward`, put the chosen value into `team_tag.yaml`'s `env.kwargs` (`draw_reward: <d>`), and rerun Step 6. The controller records the ruling with all numbers. If no draw reward passes either, stop and report to the controller (the owner decides).

- [ ] **Step 9: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 10: Commit and push**

```bash
git add configs/examples/team_tag.yaml scripts/team_tag_anchors.py tests/unit/test_sp3_team_tag_anchors.py \
  tests/learning/test_demo_learning_slow.py docs/benchmarks.md docs/benchmarks/team-tag-anchors.jsonl
# fallback path only: git add examples/team_tag/game.py tests/contract/test_demo_team_tag.py
git commit -m "feat: team_tag trains against a RandomBot anchor; its slow test is a single run"
git push origin sp3-league
```

---
### Task T6.4: Slow pipeline test on `unit_harvest`, thresholds, pipeline vs scratch

Spec section 3, criterion 3 (the main criterion) and criterion 10 (the slow suite): the whole competition pipeline on `unit_harvest` through the CLI — `record` of the game's bot → `bc` → `train` with `init` from the BC weights (critic warm-up), a scripted kickstart teacher (DAgger, the bot itself) and anchors → `eval` of the trained agent against the bot, against `RandomBot` and against the frozen `bc.pt`. Thresholds are measured here and fixed as rulings. The "pipeline vs training from scratch on an equal budget" measurement goes into the report, not into an assertion.

Background from SP2 (docs/benchmarks.md): at K = 8 a policy trained from scratch for 260–340k env steps beats a random player 100 % and *ties* the scripted bot (`share_vs_scripted` exactly 0.5 in six runs: both reach the same ceiling). So "vs the bot" is measured by the eval `score` (win = 1, draw = 0.5), and its threshold is expected near 0.5, not above it.

**Files:**
- Create: `tests/learning/test_sp3_pipeline_slow.py`
- Create: `scripts/pipeline_vs_scratch.py`
- Possibly modify (calibration, Step 3): `tests/pipeline_kit.py` (`UNIT_HARVEST_SETTINGS`), `configs/examples/unit_harvest_league.yaml` (`training.total_timesteps`)
- Create: `docs/benchmarks/pipeline-vs-scratch.json`; modify `docs/benchmarks.md`

**Interfaces:**
- Consumes: `pipeline_kit.run_pipeline`, `UNIT_HARVEST`, `UNIT_HARVEST_SETTINGS`, `PipelineRun`, `RANDOM`, `BC_NET` (T6.2); `demo_learning.LEARNING_SEED`; the CLI `record`, `bc`, `train`, `eval` (T5.1, T5.2, T1.5).
- Produces: the slow test `test_unit_harvest_pipeline_beats_random_and_holds_the_bot_and_bc` with constants `VS_RANDOM_MIN_WIN`, `VS_BOT_MIN_SCORE`, `VS_BC_MIN_SCORE`; `scripts/pipeline_vs_scratch.py` (`--seeds`, `--json`, `--work-dir`).

**Threshold rule (fixed before the measurement).** Three calibration runs of the test (seeds 0, 1, 2). For each metric `x` ∈ {win rate vs `random`, score vs `greedy`, score vs `bc_net`}: threshold = max(floor, ⌊(min over the 3 runs − 0.05) / 0.05⌋ · 0.05). Floors: win rate vs `random` 0.80 (SP2's bar for this game), score vs the bot 0.35, score vs `bc_net` 0.45 (RL must not fall clearly below its own warm start). If any calibration run is below a floor, the pipeline is not working: tune `UNIT_HARVEST_SETTINGS` / the league config within the time budget (each change is a ruling with the numbers) and recalibrate; if that fails, stop and report to the controller.

- [ ] **Step 1: Write the slow test (floors as thresholds) and the comparison script**

Create `tests/learning/test_sp3_pipeline_slow.py`:

```python
"""Slow SP3 pipeline test on unit_harvest (spec section 3, criterion 3): record the game's bot ->
bc -> train from the BC weights (critic warm-up) with the bot as a DAgger kickstart teacher and
scripted anchors -> eval against the bot, RandomBot and the frozen BC net, all through the CLI.
Thresholds: T6.4 rule (max(floor, min over three calibration runs - 0.05, rounded down to 0.05));
changing one needs a ruling with measurements."""
from __future__ import annotations

import json

import pytest

import pipeline_kit as kit
from demo_learning import LEARNING_SEED

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1500), pytest.mark.usefixtures("restore_global_rng")]

# T6.4 ruling (calibration seeds 0, 1, 2): <filled in Step 4>
VS_RANDOM_MIN_WIN = 0.80      # floor: SP2's bar for unit_harvest
VS_BOT_MIN_SCORE = 0.35       # floor (score = wins + draws / 2; SP2: trained policies tie the bot at K = 8)
VS_BC_MIN_SCORE = 0.45        # floor: RL must not end clearly below its own warm start


def _cell(row: dict) -> str:
    return f"W/D/L {row['wins']}/{row['draws']}/{row['losses']} score {row['score']:.3f} win {row['win_rate']:.3f}"


def test_unit_harvest_pipeline_beats_random_and_holds_the_bot_and_bc(tmp_path):
    game, settings = kit.UNIT_HARVEST, kit.UNIT_HARVEST_SETTINGS
    result = kit.run_pipeline(game, tmp_path, seed=LEARNING_SEED, settings=settings)
    vs_random, vs_bot, vs_bc = (result.pairs[name] for name in (kit.RANDOM, game.bot, kit.BC_NET))
    seconds = json.dumps({k: round(v) for k, v in result.seconds.items()})
    print(f"[unit_harvest pipeline] seed {LEARNING_SEED}; seconds {seconds}; recorded {result.record['decisions']} "
          f"decisions; env steps/s {_env_steps_per_sec(result):.0f}")
    print(f"[unit_harvest pipeline] vs random: {_cell(vs_random)}; vs {game.bot}: {_cell(vs_bot)}; "
          f"vs {kit.BC_NET}: {_cell(vs_bc)}")
    assert vs_random["win_rate"] >= VS_RANDOM_MIN_WIN
    assert vs_bot["score"] >= VS_BOT_MIN_SCORE
    assert vs_bc["score"] >= VS_BC_MIN_SCORE


def _env_steps_per_sec(result: kit.PipelineRun) -> float:
    system = result.run.records("system")
    if len(system) < 2 or system[-1]["ts"] <= system[0]["ts"]:
        return float("nan")
    return (system[-1]["env_steps"] - system[0]["env_steps"]) / (system[-1]["ts"] - system[0]["ts"])
```

Create `scripts/pipeline_vs_scratch.py`:

```python
"""Pipeline vs training from scratch on an equal env-step budget (SP3 spec section 3, criterion 3;
reported, not asserted).

For every seed: the full pipeline of the slow test (``tests/pipeline_kit.py``: record the
unit_harvest bot -> bc -> train with init + critic warm-up + the bot as DAgger teacher + anchors ->
eval), then the same training budget and league from scratch (no ``init``, no kickstart; the frozen
BC net stays in the config so both leagues are identical). Both trained agents are evaluated
against the bot, RandomBot and the BC net (``colosseum eval --deterministic``).

Usage::

    .venv/bin/python scripts/pipeline_vs_scratch.py --seeds 0 1 2 --json docs/benchmarks/pipeline-vs-scratch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tests", REPO_ROOT / "tests" / "learning"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _row(variant: str, seed: int, run, game) -> dict:
    import pipeline_kit as kit

    row = {"variant": variant, "seed": seed, **{f"{k}_s": round(v, 1) for k, v in run.seconds.items()}}
    for name, key in ((game.bot, "bot"), (kit.RANDOM, "random"), (kit.BC_NET, "bc")):
        cell = run.pairs[name]
        row.update({f"{key}_score": cell["score"], f"{key}_win": cell["win_rate"],
                    f"{key}_draw": cell["draws"] / cell["n"]})
    return row


def main() -> int:
    import pipeline_kit as kit

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/colosseum-pipeline-vs-scratch"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    game, settings = kit.UNIT_HARVEST, kit.UNIT_HARVEST_SETTINGS
    rows = []
    for seed in args.seeds:
        workdir = args.work_dir / f"s{seed}"
        workdir.mkdir(parents=True, exist_ok=True)
        pipeline = kit.run_pipeline(game, workdir, seed=seed, settings=settings, name="pipeline")
        rows.append(_row("pipeline", seed, pipeline, game))
        print(json.dumps(rows[-1]), flush=True)
        scratch = kit.run_pipeline(game, workdir, seed=seed, settings=settings, warm_start=False,
                                   bc_path=pipeline.bc_path, name="scratch")
        rows.append(_row("scratch", seed, scratch, game))
        print(json.dumps(rows[-1]), flush=True)
    cols = ["variant", "seed", "bot_score", "bot_win", "bot_draw", "random_win", "bc_score", "record_s", "bc_s",
            "train_s", "eval_s"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f"{row[c]:.3f}" if isinstance(row.get(c), float) else str(row.get(c, "-"))
                                for c in cols) + " |")
    for variant in ("pipeline", "scratch"):
        scores = [r["bot_score"] for r in rows if r["variant"] == variant]
        print(f"{variant}: mean score vs the bot {sum(scores) / len(scores):.3f} over {len(scores)} seeds")
    if args.json:
        result = {"seeds": args.seeds, "settings": repr(settings), "rows": rows}
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `.venv/bin/python -m pytest tests/learning/test_sp3_pipeline_slow.py --collect-only -q` and `.venv/bin/python -m pytest -m "not gpu and not slow" --collect-only -q | tail -1`

Expected: the slow test is collected; the fast-suite count did not change (the test is `slow`).

- [ ] **Step 2: First run (seed 0) and the time budget**

Preconditions: nothing else CPU-heavy runs; `uptime` load below 1.0.

```bash
COLOSSEUM_LEARNING_SEED=0 .venv/bin/python -m pytest tests/learning/test_sp3_pipeline_slow.py -m slow -v -s \
  2>&1 | tee /tmp/sp3-pipeline-s0.txt | grep -E "^\[|PASSED|FAILED|Error"
```

Record: the printed seconds per step, the three eval rows, env steps per second. Budget target (the SP2 rule for slow learning tests): training at most about 3 minutes, the whole test at most about 5 minutes. If the training step is longer, lower `training.total_timesteps` in `configs/examples/unit_harvest_league.yaml`; if record or bc dominate, lower `record_matches` / `bc_epochs` in `UNIT_HARVEST_SETTINGS` (each change is reported to the controller, who records the ruling with the numbers).

- [ ] **Step 3: Calibration runs (seeds 1 and 2)**

```bash
for s in 1 2; do
  COLOSSEUM_LEARNING_SEED=$s .venv/bin/python -m pytest tests/learning/test_sp3_pipeline_slow.py -m slow -v -s \
    2>&1 | tee /tmp/sp3-pipeline-s$s.txt | grep -E "^\[|PASSED|FAILED|Error"
done
```

Record the same numbers. A run below a floor triggers the tuning described above the steps (then all three calibration runs are repeated with the final settings).

- [ ] **Step 4: Fix the thresholds by the rule**

Compute, from the three runs, `threshold = max(floor, floor_to_0.05(min - 0.05))` for each metric and replace the three constants and the comment line `# T6.4 ruling (calibration seeds 0, 1, 2): ...` with the numbers (for example `# T6.4 ruling (calibration seeds 0, 1, 2): vs random win 1.000 / 1.000 / 0.983, vs greedy score 0.50 / 0.48 / 0.52, vs bc_net score 0.61 / 0.55 / 0.58`). The controller records the ruling: "unit_harvest pipeline thresholds — the T6.4 rule over three calibration runs (values) — cost if wrong: a flaky slow test (too tight) or a test that misses a broken warm start (too loose)".

Rerun once with seed 0 (Step 2 command): PASSED.

- [ ] **Step 5: Pipeline vs scratch**

```bash
mkdir -p /tmp/colosseum-pipeline-vs-scratch
nohup .venv/bin/python scripts/pipeline_vs_scratch.py --seeds 0 1 2 --json docs/benchmarks/pipeline-vs-scratch.json \
  > /tmp/colosseum-pipeline-vs-scratch/driver.log 2>&1 &
```

Six trainings (about 15–20 minutes). Then append to `docs/benchmarks.md`:

```markdown
## Конвейер против обучения с нуля (SP3, критерий 3)

`scripts/pipeline_vs_scratch.py`: `unit_harvest` (K = 8), лига `configs/examples/unit_harvest_league.yaml` (якоря `greedy` и `random`), бюджет обучения <N> шагов сред на прогон, 2 воркера, сиды 0, 1, 2. «Конвейер» — `record` бота (<M> партий) → `bc` (<E> эпох) → `train` с `init` (прогрев критика <W> шагов обучения), kickstart от бота (DAgger, λ = 1, затухание <D> шагов) и якорями; «с нуля» — та же лига и тот же бюджет без `init` и kickstart. Оценка — `colosseum eval --deterministic`, <K> партий на пару; `score` = победы + ничьи / 2. Сырые данные: [`benchmarks/pipeline-vs-scratch.json`](benchmarks/pipeline-vs-scratch.json). Дата: <date>.

| вариант | сид | score против бота | win против бота | ничьи с ботом | win против random | score против bc_net | record, с | bc, с | обучение, с |
|---|---|---|---|---|---|---|---|---|---|
<six rows from the script's table>

Вывод: <mean score vs the bot, pipeline vs scratch; the extra wall time of record + bc; one sentence: does the warm start pay off on this game and budget>.
```

Fill every `<...>` from the script output and `UNIT_HARVEST_SETTINGS`. This result is reported (and copied into the acceptance report), not asserted.

- [ ] **Step 6: Slow-suite pre-check (criterion 10)**

The new default opponent mix (snapshots of other agents in asymmetric games, PFSP over snapshots) can move the SP2 slow tests (spec section 8, risks). Run the whole slow suite once before the acceptance:

```bash
nohup .venv/bin/python -m pytest -m slow -v -s > /tmp/sp3-slow-precheck.txt 2>&1 &
# later:
grep -E "^\[|PASSED|FAILED" /tmp/sp3-slow-precheck.txt; tail -3 /tmp/sp3-slow-precheck.txt
```

Expected: every slow test passes (the 7 SP2 learning tests, team_tag as a single run, the 2 torch.compile checks, the new pipeline test). A failing SP2 learning test is reported to the controller with its printed numbers; the controller decides by ruling with measurements (rerun to check noise; a change of that example config, e.g. its `opponents`, measured on 3 seeds) — thresholds of SP2 tests are never lowered silently.

- [ ] **Step 7: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`.

- [ ] **Step 8: Commit and push**

```bash
git add tests/learning/test_sp3_pipeline_slow.py scripts/pipeline_vs_scratch.py tests/pipeline_kit.py \
  configs/examples/unit_harvest_league.yaml docs/benchmarks.md docs/benchmarks/pipeline-vs-scratch.json
git commit -m "test: slow unit_harvest pipeline test with measured thresholds; pipeline vs scratch measurement"
git push origin sp3-league
```

---

### Task T6.5: APPO `Units` defaults: multi-seed measurement and the pre-fixed rule

Spec block 10, criterion 7. SP2 chose `unit_trace: auto = joint` on one seed and left `ratio_mode: auto = per_unit` with `Units` as an open owner question (docs/benchmarks.md, «Эксперимент по юнитам»). SP3 measures the two questions on three seeds and changes a default only by the rule fixed in the spec.

**Grid** (`unit_harvest`, `scripts/units_experiment.py`, 340 000 env steps per run as in SP2, 2 workers, runs one at a time):

| cell | `ratio_mode` | `unit_trace` | role |
|---|---|---|---|
| A | `per_unit` | `auto` (= `joint`) | the current default (baseline of both questions) |
| B | `joint` | `auto` | the `ratio_mode` alternative |
| C | `per_unit` | `none` | the `unit_trace` alternative |

× K ∈ {8, 128} × seeds {0, 1, 2} = 18 runs. SP2 timings: K = 8 about 225 s, K = 128 about 490 s per run, evaluation about 20 s: about 2 hours in total.

**Rule (spec block 10, fixed in advance), per question:** the default changes to the alternative only if both hold:
1. at K = 128, `mean(alt) − mean(base) > 2 · SE`, where `SE = sqrt(var(alt)/n_alt + var(base)/n_base)` with the sample variances (ddof = 1) of `share_vs_scripted` over the seeds (`n` = 3; unpaired: async training does not pair seeds);
2. at K = 8, `mean(alt) ≥ mean(base) − 0.03`.

Otherwise the default stays. If both questions switch, both changes apply (the combination `joint` + `none` is then measured once on the three K = 128 seeds and reported, not ruled on).

**Files:**
- Modify: `scripts/units_experiment.py` (`--cells`; `unit_trace` value `auto`; `decide_defaults`; the JSON gets `decisions`)
- Test: `tests/unit/test_sp3_units_defaults_rule.py`
- Create: `docs/benchmarks/units-defaults-sp3.json`; modify `docs/benchmarks.md`
- Only if the rule switches a default: `src/colosseum/algorithms/appo.py` (`resolve_modes`, module docstring), `src/colosseum/core/config.py` (`AlgorithmConfig.ratio_mode` / `unit_trace` descriptions), `tests/unit/test_appo_v2.py::test_modes_resolve_auto_and_collapse_for_one_decider`, `configs/examples/unit_harvest.yaml` (comments), `README.md` (algorithm row)

**Interfaces:**
- Consumes: `scripts/units_experiment.py` (`train`, `diagnostics`, `evaluate`, `recommend`, `K_KWARGS`); `demo_learning.TrainedRun`; `resolve_modes` (SP2).
- Produces: `units_experiment.SP3_CELLS`, `units_experiment.compare(rows, baseline, alternative) -> dict`, `units_experiment.decide_defaults(rows) -> dict[str, dict]`, CLI `--cells RATIO:TRACE ...`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_units_defaults_rule.py`:

```python
"""The SP3 rule for the APPO Units defaults (spec block 10, T6.5): scripts/units_experiment.py's
decide_defaults on synthetic rows. No processes, no training."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "units_experiment.py"
_spec = importlib.util.spec_from_file_location("units_experiment_sp3", _SCRIPT)
ue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ue)


def _rows(cells: dict[tuple[int, str, str], list[float]]) -> list[dict]:
    return [{"k": k, "ratio_mode": r, "unit_trace": t, "seed": s, "share_vs_scripted": v}
            for (k, r, t), values in cells.items() for s, v in enumerate(values)]


BASE8 = {(8, "per_unit", "auto"): [0.50, 0.50, 0.50], (8, "joint", "auto"): [0.50, 0.50, 0.50],
         (8, "per_unit", "none"): [0.50, 0.50, 0.50]}


def test_a_clear_win_at_k128_that_holds_at_k8_switches():
    rows = _rows({**BASE8, (128, "per_unit", "auto"): [0.20, 0.18, 0.22], (128, "joint", "auto"): [0.30, 0.29, 0.31],
                  (128, "per_unit", "none"): [0.21, 0.19, 0.20]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is True and decisions["ratio_mode"]["choice"] == "joint"
    assert decisions["unit_trace"]["switch"] is False and decisions["unit_trace"]["choice"] == "joint"
    assert decisions["ratio_mode"]["diff_k128"] == pytest.approx(0.10)


def test_a_win_within_two_standard_errors_does_not_switch():
    rows = _rows({**BASE8, (128, "per_unit", "auto"): [0.10, 0.30, 0.20], (128, "joint", "auto"): [0.15, 0.35, 0.25],
                  (128, "per_unit", "none"): [0.20, 0.20, 0.20]})
    assert ue.decide_defaults(rows)["ratio_mode"]["switch"] is False


def test_losing_more_than_003_at_k8_blocks_the_switch():
    rows = _rows({(8, "per_unit", "auto"): [0.50, 0.50, 0.50], (8, "joint", "auto"): [0.46, 0.46, 0.46],
                  (8, "per_unit", "none"): [0.48, 0.48, 0.48],
                  (128, "per_unit", "auto"): [0.20, 0.18, 0.22], (128, "joint", "auto"): [0.30, 0.29, 0.31],
                  (128, "per_unit", "none"): [0.30, 0.29, 0.31]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is False                     # 0.04 worse at K = 8
    assert decisions["unit_trace"]["switch"] is True and decisions["unit_trace"]["choice"] == "none"   # 0.02 worse


def test_missing_cells_give_no_decision():
    rows = _rows({(128, "per_unit", "auto"): [0.2, 0.2, 0.2], (128, "joint", "auto"): [0.3]})
    decisions = ue.decide_defaults(rows)
    assert decisions["ratio_mode"]["switch"] is None and "K=8" in decisions["ratio_mode"]["note"]
    assert decisions["unit_trace"]["switch"] is None


def test_cells_parse_ratio_and_trace():
    assert ue.parse_cells(["joint:auto", "per_unit:none"]) == [("joint", "auto"), ("per_unit", "none")]
    with pytest.raises(ValueError, match="ratio_mode:unit_trace"):
        ue.parse_cells(["joint"])
```

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_units_defaults_rule.py tests/unit/test_units_experiment_rule.py -q`

Expected: the new tests fail with `AttributeError: module 'units_experiment_sp3' has no attribute 'decide_defaults'`; the SP2 rule tests pass.

- [ ] **Step 2: Extend `scripts/units_experiment.py`**

1. Docstring: add a paragraph after the SP2 rule paragraph:

```text
SP3 (spec block 10): ``--cells`` replaces the ratio x trace product by explicit cells. The SP3 grid
is ``--cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2`` (the current default and the two
alternatives); ``decide_defaults`` applies the SP3 rule: a default changes only if at K=128 the
alternative's mean ``share_vs_scripted`` beats the default's by more than 2 standard errors of the
difference (unpaired, sample variances) and at K=8 it is at most 0.03 worse.
```

and the usage line `.venv/bin/python scripts/units_experiment.py --cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2 --steps 340000 --parallel 1 --json docs/benchmarks/units-defaults-sp3.json`.

2. Constants: `UNIT_TRACES = ("auto", "joint", "geo_mean", "none")` (SP2's `recommend` keeps iterating its own traces: change its loops to `SP2_TRACES = ("joint", "geo_mean", "none")`), plus:

```python
SP2_TRACES = ("joint", "geo_mean", "none")
SP3_CELLS = (("per_unit", "auto"), ("joint", "auto"), ("per_unit", "none"))
SE_FACTOR = 2.0          # SP3 rule: the K=128 gain must exceed 2 standard errors of the difference
K8_TOLERANCE = 0.03      # SP3 rule: at K=8 the alternative may be at most this much worse
```

In `recommend`, replace both uses of `UNIT_TRACES` with `SP2_TRACES` (its tests keep passing).

3. Add after `recommend`:

```python
def parse_cells(values: list[str]) -> list[tuple[str, str]]:
    cells = []
    for value in values:
        ratio, sep, trace = value.partition(":")
        if not sep or ratio not in RATIO_MODES or trace not in UNIT_TRACES:
            raise ValueError(f"cell {value!r}: expected ratio_mode:unit_trace with ratio_mode in {RATIO_MODES} "
                             f"and unit_trace in {UNIT_TRACES}")
        cells.append((ratio, trace))
    return cells


def _shares(rows: list[dict], k: int, cell: tuple[str, str]) -> list[float]:
    return [float(r["share_vs_scripted"]) for r in rows if "share_vs_scripted" in r and r["k"] == k
            and (r["ratio_mode"], r["unit_trace"]) == cell]


def _mean_var(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def compare(rows: list[dict], baseline: tuple[str, str], alternative: tuple[str, str]) -> dict:
    """The SP3 rule for one question: switch to ``alternative`` only if at K=128 its mean share beats
    ``baseline`` by more than ``SE_FACTOR`` standard errors of the difference and at K=8 it is at most
    ``K8_TOLERANCE`` worse. ``switch`` is None (with a ``note``) when a cell lacks the data."""
    a128, b128 = _shares(rows, 128, alternative), _shares(rows, 128, baseline)
    a8, b8 = _shares(rows, 8, alternative), _shares(rows, 8, baseline)
    missing = [f"K=128 {c} (need >= 2 seeds)" for c, v in ((alternative, a128), (baseline, b128)) if len(v) < 2]
    missing += [f"K=8 {c}" for c, v in ((alternative, a8), (baseline, b8)) if not v]
    out: dict = {"baseline": list(baseline), "alternative": list(alternative)}
    if missing:
        return {**out, "switch": None, "note": "no decision: missing " + ", ".join(missing)}
    mean_a, var_a = _mean_var(a128)
    mean_b, var_b = _mean_var(b128)
    se = math.sqrt(var_a / len(a128) + var_b / len(b128))
    diff128 = mean_a - mean_b
    diff8 = sum(a8) / len(a8) - sum(b8) / len(b8)
    switch = diff128 > SE_FACTOR * se and diff8 >= -K8_TOLERANCE
    return {**out, "switch": switch, "diff_k128": diff128, "se_k128": se, "diff_k8": diff8,
            "mean_k128": {"alternative": mean_a, "baseline": mean_b}}


def decide_defaults(rows: list[dict]) -> dict[str, dict]:
    """Both SP3 questions; ``choice`` is the resolved value with Units (``auto`` stays when no switch)."""
    ratio = compare(rows, ("per_unit", "auto"), ("joint", "auto"))
    trace = compare(rows, ("per_unit", "auto"), ("per_unit", "none"))
    ratio["choice"] = "joint" if ratio["switch"] else "per_unit"
    trace["choice"] = "none" if trace["switch"] else "joint"
    return {"ratio_mode": ratio, "unit_trace": trace}
```

4. In `main`: add `parser.add_argument("--cells", nargs="+", default=None, help="explicit ratio_mode:unit_trace cells (SP3 grid: per_unit:auto joint:auto per_unit:none)")`; build the grid as `[(k, r, t, s) for k in args.k for (r, t) in parse_cells(args.cells) for s in args.seeds]` when `--cells` is given, else the SP2 product; make `--unit-traces` choices `UNIT_TRACES`. After the SP2 recommendation, when `--cells` is given, print and store the SP3 decisions:

```python
    decisions = decide_defaults(rows) if args.cells else None
    if decisions:
        for question, d in decisions.items():
            if d["switch"] is None:
                print(f"{question}: {d['note']}")
            else:
                print(f"{question}: {'SWITCH to' if d['switch'] else 'keep'} {d['choice']} "
                      f"(K=128 diff {d['diff_k128']:+.3f}, 2*SE {SE_FACTOR * d['se_k128']:.3f}; "
                      f"K=8 diff {d['diff_k8']:+.3f})")
```

and add `"decisions": decisions` and `"cells": args.cells` to the JSON result.

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_units_defaults_rule.py tests/unit/test_units_experiment_rule.py -q`

Expected: all pass.

- [ ] **Step 3: Run the measurement**

Preconditions: nothing else CPU-heavy runs; `uptime` load below 1.0.

```bash
mkdir -p /tmp/colosseum-units-sp3
nohup .venv/bin/python scripts/units_experiment.py --cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2 \
  --steps 340000 --parallel 1 --run-dir /tmp/colosseum-units-sp3 --json docs/benchmarks/units-defaults-sp3.json \
  > /tmp/colosseum-units-sp3/driver.log 2>&1 &
```

About 2 hours. Poll `tail -5 /tmp/colosseum-units-sp3/driver.log`. A run with a non-zero exit is investigated through its `<name>.out` and rerun (the script reruns the whole grid; for a single rerun call `train(...)` on that cell from a Python shell and rebuild the JSON by rerunning with `--cells <that cell> --seeds <that seed>` into a separate JSON, then merge the rows by hand into the main JSON and recompute `decide_defaults` from it — say so in the report).

Record: the full table (all diagnostics columns), the two decisions with their numbers.

- [ ] **Step 4: Apply the rule**

- **No switch (both "keep"):** no code change. The ruling: "`ratio_mode: auto` = `per_unit` and `unit_trace: auto` = `joint` with `Units` stay — the SP3 rule on 3 seeds (numbers) — cost if wrong: a worse default on many-unit games, an explicit setting overrides it".
- **`ratio_mode` switches to `joint`:** in `src/colosseum/algorithms/appo.py::resolve_modes` replace

```python
    ratio: RatioMode = config.ratio_mode if config.ratio_mode != "auto" else (
        "per_unit" if action_spec.has_units else "joint")
```

with

```python
    ratio: RatioMode = config.ratio_mode if config.ratio_mode != "auto" else "joint"   # SP3 rule (docs/benchmarks.md)
```

  and in `tests/unit/test_appo_v2.py::test_modes_resolve_auto_and_collapse_for_one_decider` the first assertion becomes `assert resolve_modes(AlgorithmConfig(), units) == ("joint", "joint", "sum")` with the comment `# SP3 ruling (T6.5): ratio_mode auto = joint also with Units`.
- **`unit_trace` switches to `none`:** replace `trace: UnitTrace = config.unit_trace if config.unit_trace != "auto" else "joint"` with `trace: UnitTrace = config.unit_trace if config.unit_trace != "auto" else "none"   # SP3 rule with Units (K > 1); K == 1 returns earlier` and update the same test's first assertion to `("per_unit", "none", "mean_valid")` (or `("joint", "none", "sum")` if both switch).
- For any switch also update: the module docstring of `appo.py` (the `auto` sentences), `resolve_modes`' docstring, the `AlgorithmConfig.ratio_mode` / `unit_trace` descriptions in `core/config.py`, the comments in `configs/examples/unit_harvest.yaml` (`ratio_mode: "auto"  # ...`), the README configuration-reference row `algorithm`, and the `docs/ENV_GUIDE.md` sentence on `ratio_mode` if it names the default (`grep -n "ratio_mode\|unit_trace" README.md docs/ENV_GUIDE.md`). If both switch, run the combination once: `--cells joint:none --k 128 --seeds 0 1 2 --json docs/benchmarks/units-defaults-sp3-combo.json` (reported).

- [ ] **Step 5: Benchmarks section**

Append to `docs/benchmarks.md`:

```markdown
## Дефолты APPO при `Units` (SP3, блок 10)

`scripts/units_experiment.py --cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2 --steps 340000 --parallel 1`: `unit_harvest`, K = 8 и K = 128 (параметры как в SP2), 340 000 шагов сред на прогон, 2 воркера, прогоны по одному. Ячейки: текущий дефолт (`per_unit`, `unit_trace` `auto` = `joint`) и две альтернативы: `ratio_mode: joint` и `unit_trace: none`. Метрика — `share_vs_scripted` (100 партий против `scripted_action`). Сырые данные: [`benchmarks/units-defaults-sp3.json`](benchmarks/units-defaults-sp3.json). Дата: <date>; время: <hours>.

Правило (спека SP3, блок 10, задано до замера): дефолт меняется на альтернативу, только если при K = 128 её среднее по 3 сидам лучше дефолта больше чем на 2 стандартные ошибки разности (SE = sqrt(s²_alt/3 + s²_base/3), выборочные дисперсии) и при K = 8 она хуже не больше чем на 0.03.

| k | ratio_mode | unit_trace | seed | win_vs_random | share_vs_scripted | clip_fraction | clip_fraction_joint | ess | log_rho_abs_p95 | env_steps_per_s | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
<18 rows from the script's table>

| вопрос | среднее K=128 (дефолт → альтернатива) | разность | 2·SE | разность K=8 | решение |
|---|---|---|---|---|---|
| `ratio_mode` (`per_unit` → `joint`) | <…> | <…> | <…> | <…> | <оставить / сменить> |
| `unit_trace` (`joint` → `none`) | <…> | <…> | <…> | <…> | <оставить / сменить> |

Решение (ruling T6.5): <the defaults after the rule, one sentence each; the SP2 open questions about ratio_mode and unit_trace are closed by it>.
```

Fill every `<...>` from the log and the JSON.

- [ ] **Step 6: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`. If a default switched, also run `.venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -k unit_harvest -v -s` (the only slow learning test with `Units`): PASSED.

- [ ] **Step 7: Commit and push**

```bash
git add scripts/units_experiment.py tests/unit/test_sp3_units_defaults_rule.py docs/benchmarks.md \
  docs/benchmarks/units-defaults-sp3.json
# only if a default switched: git add src/colosseum/algorithms/appo.py src/colosseum/core/config.py \
#   tests/unit/test_appo_v2.py configs/examples/unit_harvest.yaml README.md docs/ENV_GUIDE.md
git commit -m "perf: multi-seed measurement of the APPO Units defaults with the SP3 rule"
git push origin sp3-league
```

---
### Task T6.6: Docs, example configs in the v3 form, benchmark config, throughput against `main`

Spec block 11 and criteria 5 and 8: `docs/LEAGUE_GUIDE.md` (new, Russian) with the recipes of block 11; the `infos` section in `docs/ENV_GUIDE.md`; README (pipeline quick start, reference, status) and CLAUDE.md (status, roadmap, open items) describing the real state; the pinned benchmark workload checked; throughput of the branch measured against `main` in the same conditions (criterion 3.5). The example configs move to the v3 form (behaviour identical to the translation of their SP2 form), so the quick start prints no translation warnings; the SP2 copies under `tests/fixtures/sp2/configs/` keep the old form.

**Files:**
- Create: `docs/LEAGUE_GUIDE.md`
- Modify: `docs/ENV_GUIDE.md`, `README.md`, `CLAUDE.md`, `docs/benchmarks.md`
- Modify: `configs/examples/{chase,coin_grid,coop_buttons,predator_prey,space_miners,tic_tac_toe,tic_tac_toe_attention,tic_tac_toe_multi,tron,unit_harvest}.yaml` (v3 matchmaking and checkpoint sections)
- Modify (if still in the SP2 form): `tests/cli_runner.py` (`TINY`: `checkpoint.pool_size` → `checkpoint.keep_last`), `tests/game_helpers.py` (`make_test_config`: `pool_size` → `keep_last`)
- Modify: `scripts/bench_throughput.py` (`_make_config`: explicit latest-only opponents), `tests/unit/test_bench_throughput.py`
- Create: `tests/unit/test_sp3_example_configs_v3.py`
- Create: `docs/benchmarks/sp3-throughput/{main-1,branch-1,main-2,branch-2}.json`

**Interfaces:**
- Consumes: everything SP3 built (names from the overview contract), the T6.2–T6.5 results and rulings, `SP2_MATCHMAKING_KNOBS` (T3.1), the SP2 config copies (T0.1; Contract note N7).
- Produces: documentation only, plus the v3 example configs and the throughput numbers of criterion 3.5.

- [ ] **Step 1: Example configs in the v3 form**

Replace in each example config the `matchmaking:` section by the translation of its SP2 knobs (spec block 5 formula: `spr = 1` for `mode: self_play`, else `self_play_ratio`; `latest = spr · latest_prob`, `snapshots = spr · (1 − latest_prob)`, `rivals = 1 − spr`, `anchors = 0`; `pfsp = {weighting: hard, exponent: pfsp_exponent}`, SP2 default exponent 1.0) and rename `pool_size` to `keep_last` in `checkpoint:` (comments kept; `keep_every` stays at its default, as in the translation):

| config | `matchmaking` in the v3 form (other keys of the section unchanged) | `checkpoint.keep_last` |
|---|---|---|
| `chase.yaml` | `opponents: {latest: 0.5, snapshots: 0.5, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 5 |
| `coin_grid.yaml` | `opponents: {latest: 0.5, snapshots: 0.5, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 10 |
| `coop_buttons.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}`, `teammates: mixed`, `teammate_self_prob: 0.5` | 10 |
| `predator_prey.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 10 |
| `space_miners.yaml` | `opponents: {latest: 0.5, snapshots: 0.5, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 10 |
| `tic_tac_toe.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}`, `shuffle_seats: true` | 10 |
| `tic_tac_toe_attention.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 10 |
| `tic_tac_toe_multi.yaml` | `opponents: {latest: 0.0, snapshots: 0.0, rivals: 1.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}`, `shuffle_seats: true` | 5 |
| `tron.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}`, `layouts` unchanged | 10 |
| `unit_harvest.yaml` | `opponents: {latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`, `pfsp: {weighting: hard, exponent: 1.0}` | 10 |

Put one comment line above each `opponents:` line: `# SP2 form: mode self_play, latest_prob 0.8 (translated exactly; see docs/LEAGUE_GUIDE.md)` with that config's old values. `team_tag.yaml` is already v3 (T6.3); `unit_harvest_league.yaml` is new.

Check the test-support defaults: `grep -n "pool_size" tests/cli_runner.py tests/game_helpers.py`. Any hit is renamed to `keep_last` (same value) — otherwise a test config that also sets `checkpoint.keep_last` would combine the old and the new knob (a ConfigError).

Create `tests/unit/test_sp3_example_configs_v3.py`:

```python
"""Example configs in the SP3 form (T6.6): no SP2 knob, no translation warning, and the matchmaking
and checkpoint sections equal the translation of their SP2 copies (tests/fixtures/sp2/configs)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from colosseum.core.config import SP2_MATCHMAKING_KNOBS, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "configs" / "examples"
SP2_COPIES = REPO_ROOT / "tests" / "fixtures" / "sp2" / "configs"
CHANGED_ON_PURPOSE = {"team_tag.yaml"}          # T6.3: RandomBot anchor (spec block 9)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_example_configs_use_no_sp2_knob(path, caplog):
    raw = yaml.safe_load(path.read_text())
    assert not set(raw.get("matchmaking") or {}) & set(SP2_MATCHMAKING_KNOBS)
    assert "pool_size" not in (raw.get("checkpoint") or {})
    assert not [key for key in (raw.get("training") or {}) if key.startswith("kickstart_")]
    with caplog.at_level(logging.WARNING):
        load_config(path)
    assert [r.getMessage() for r in caplog.records if r.name.startswith("colosseum")] == []


@pytest.mark.parametrize("name", sorted(p.name for p in SP2_COPIES.glob("*.yaml") if p.name not in CHANGED_ON_PURPOSE))
def test_example_configs_behave_like_the_translation_of_their_sp2_copies(name):
    live, old = load_config(EXAMPLES / name), load_config(SP2_COPIES / name)
    assert live.matchmaking == old.matchmaking
    assert live.checkpoint == old.checkpoint
```

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_example_configs_v3.py tests/unit/test_example_configs.py -q`

Expected: all pass. A mismatch in the second test means the table above disagrees with T3.1's translation (for example in the PFSP exponent or `keep_every`): follow the translation (the code is the authority for "behaviour identical") and say so in the commit body.

- [ ] **Step 2: The pinned benchmark workload**

In `scripts/bench_throughput.py::_make_config`, the `matchmaking` section becomes explicitly latest-only (the same work as `main`'s SP2 workload: there every seat played the latest weights because no checkpoint was ever written):

```python
        matchmaking={
            "opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0},
            "layouts": {LAYOUT: 1.0},
            "shuffle_seats": True,  # one agent, every seat latest+collect: shuffling is a no-op
        },
```

(keep whatever `checkpoint` keys T2.1 set: `interval` 10**9, `keep_last`, `keep_every: 0`, `save_optimizer`; if it still says `pool_size`, rename it). Add to `tests/unit/test_bench_throughput.py`:

```python
def test_benchmark_workload_is_latest_only_self_play_without_fixed_players(tmp_path):
    """Criterion 3.5 measures the SP2 workload: no scripted / frozen players, every seat latest."""
    cfg = bench._make_config(1, str(tmp_path))
    assert cfg.get_trainable_agent_ids() == ["agent_0"] and cfg.fixed_agent_ids() == []
    opponents = cfg.matchmaking.opponents
    assert (opponents.latest, opponents.snapshots, opponents.rivals, opponents.anchors) == (1.0, 0.0, 0.0, 0.0)
    assert cfg.init.from_ is None and cfg.kickstart.teacher is None
    assert cfg.checkpoint.interval >= 10**9                       # no snapshot is ever written
```

Run: `.venv/bin/python -m pytest tests/unit/test_bench_throughput.py -q`

Expected: all pass.

- [ ] **Step 3: Throughput: `main` and the branch in the same conditions (criterion 3.5)**

Preconditions: nothing else running (`uptime` 1-minute load below 1.0; no other benchmark, measurement or test run), the branch committed. Runs alternate main / branch twice, so slow drift of the machine affects both equally.

```bash
REPO=$(pwd); B=/tmp/sp3-bench; mkdir -p $B
git worktree add --detach /tmp/colosseum-main main
(cd /tmp/colosseum-main && PYTHONPATH=/tmp/colosseum-main/src $REPO/.venv/bin/python -c "import colosseum; print(colosseum.__file__)")
```

Expected: the printed path is `/tmp/colosseum-main/src/colosseum/__init__.py` (the `PYTHONPATH` entry precedes the editable install of the branch; spawned children inherit it).

```bash
for i in 1 2; do
  uptime
  (cd /tmp/colosseum-main && PYTHONPATH=/tmp/colosseum-main/src $REPO/.venv/bin/python scripts/bench_throughput.py \
     --workers 1 2 4 --duration 60 --warmup 15 --json $B/main-$i.json)
  uptime
  .venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15 --json $B/branch-$i.json
done
git worktree remove /tmp/colosseum-main
mkdir -p docs/benchmarks/sp3-throughput && cp $B/*.json docs/benchmarks/sp3-throughput/
```

About 20 minutes. Compute per worker count w ∈ {1, 2, 4}: `main_w` = mean env steps/s of `main-1` and `main-2`, `branch_w` = the same for the branch. Criterion 3.5 holds if `branch_w ≥ 0.95 · main_w` for every w and `branch_1 < branch_2 < branch_4`. If it fails: rerun the pair once (noise check); if it still fails, report to the controller with the numbers (a performance fix is a separate task with a re-measurement; the criterion is never relaxed silently).

Append to `docs/benchmarks.md`:

```markdown
## После SP3

Та же нагрузка (`scripts/bench_throughput.py`, `_make_config`: крестики-нолики, self-play, только latest, без скриптовых и frozen игроков, без чекпоинтов), та же команда. `main` (SP2, коммит <sha>) и ветка `sp3-league` (коммит <sha>) замерены в один день на одной машине по очереди: main → ветка → main → ветка (`main` — из `git worktree` с `PYTHONPATH=<worktree>/src`, тот же `.venv`). Дата: <date>. Машина: <…>. Load average перед прогонами: <…>. Сырые данные: [`benchmarks/sp3-throughput/`](benchmarks/sp3-throughput/).

| Воркеры | main, шаги сред/с (прогон 1 / 2 / среднее) | ветка, шаги сред/с (1 / 2 / среднее) | ветка / main | апдейты/с ветки | узкое место |
|---|---|---|---|---|---|
| 1 | <…> | <…> | <…> % | <…> | <…> |
| 2 | <…> | <…> | <…> % | <…> | <…> |
| 4 | <…> | <…> | <…> % | <…> | <…> |

Критерий 3.5 спеки SP3: ветка не хуже `main` более чем на 5 % при 1 / 2 / 4 воркерах — **<да/нет>**; рост 1 → 2 → 4 монотонный — **<да/нет>**. Для справки, SP2 при приёмке: 4848 / 9298 / 13702.
```

- [ ] **Step 4: `docs/LEAGUE_GUIDE.md`**

Create `docs/LEAGUE_GUIDE.md` with the text below. Two places are filled from the running code, not invented: the `validate` printout of section 13 (run the command shown there and paste its output verbatim) and the metric keys of section 14 (from T3.3: `grep -n "opponent\|share" src/colosseum/metrics/aggregator.py src/colosseum/metrics/hub.py`). Every YAML block of the guide must pass `validate` when inserted into the named example config: check sections 3, 5, 9 and 10 that way (a scratch copy under `/tmp`).

````markdown
# Лига, игроки и warm start

Как задать, с кем играет обучаемый агент, и как начать обучение не с нуля. Всё настраивается в одном конфиге. `colosseum validate -c cfg.yaml` печатает итоговую смесь соперников каждого агента и отчёт `init` — проверяйте им каждый рецепт (раздел 13).

## 1. Понятия

- **Агент** — запись в `agents:`. Поле `kind`:
  - `trainable` (по умолчанию) — обучаемая линия со своим лёрнером;
  - `scripted` — бот на правилах, подкласс `colosseum.players.ScriptedBot` (раздел 3);
  - `frozen` — фиксированные веса из файла: чекпоинт другого запуска, прошлый сабмит, BC-сеть (раздел 4).
- Если в конфиге нет ни одного обучаемого агента (нет секции `agents` или в ней только scripted и frozen), неявно добавляется обучаемый `agent_0` с глобальными настройками. Добавление бота не заставляет переписывать минимальный конфиг.
- **Игрок** — тот, кто сидит за местом: latest обучаемого агента, его снимок `ckpt_v<N>` или scripted / frozen агент. В конфиге и CLI игроков называют только по имени агента; снимки текущего запуска выбирает матчмейкер, а конкретный снимок другого запуска подключается как frozen-агент с `path`.
- **Владелец данных** — обучаемый агент, для которого собирается среда (обучаемые агенты по очереди, в порядке конфига). Данные собирают только места с latest обучаемых агентов; снимки, scripted и frozen агенты не собирают никогда.
- **Якоря** — scripted и frozen агенты, которые играют соперниками (`matchmaking.anchors`).

## 2. Один агент в self-play

Минимальный конфиг (без `agents` и без `matchmaking`) — это один `agent_0` и смесь по умолчанию:

```yaml
matchmaking:
  opponents: {latest: 0.7, snapshots: 0.2, rivals: 0.0, anchors: 0.1}
  pfsp: {weighting: hard, exponent: 2.0, halflife_games: 200}
```

Для каждой команды соперников независимо разыгрывается категория её ядра:
- `latest` — текущие веса владельца;
- `snapshots` — хранимые снимки (свои; в асимметричной игре — и снимки соперника, раздел 6), по PFSP;
- `rivals` — latest других обучаемых агентов (арена, раздел 5);
- `anchors` — якоря (раздел 3).

Пустая сейчас категория (снимков ещё нет, якорей нет) отдаёт свою долю остальным пропорционально. Ядро садится на одно место команды; остальные места команды заполняются по `teammates` (`self` — ядру). PFSP: для каждой пары «latest владельца против игрока X» в каждом варианте партии ведётся EMA счёта x (победа 1, ничья 0.5; `halflife_games` — полураспад в партиях; до первой игры 0.5), вес кандидата — `hard` (1 − x)^p, `balanced` x(1 − x) или `uniform`.

**Конфиги SP2** работают без правок: `mode`, `self_play_ratio`, `latest_prob`, `pfsp_exponent` переводятся (`spr` = 1 при `mode: self_play`, иначе `self_play_ratio`; `latest = spr · latest_prob`, `snapshots = spr · (1 − latest_prob)`, `rivals = 1 − spr`, `anchors = 0`, `pfsp = {weighting: hard, exponent: pfsp_exponent}`) с одним предупреждением, которое показывает получившуюся секцию. Старые ручки вместе с `opponents` или `pfsp` — ошибка конфига. Так же переводятся `checkpoint.pool_size` → `keep_last` и `training.kickstart_*` → секция `kickstart`.

## 3. Скриптовый бот и якоря

```python
# my_game/bots.py
import numpy as np
from colosseum.players import ScriptedBot

class Greedy(ScriptedBot):
    def __init__(self, aggr: float = 0.5):            # kwargs из конфига
        self.aggr = aggr

    def reset(self, *, role, seat, layout, rng):     # в начале каждого эпизода, где бот сидит за этим местом
        self.rng = rng                               # засеян из сида эпизода и номера места

    def act(self, obs, mask, info):                   # numpy-деревья; info = StepResult.infos[seat] или None
        legal = np.flatnonzero(mask)
        return int(legal[0] if self.rng.random() < self.aggr else self.rng.choice(legal))
```

- Экземпляр создаётся на тройку (агент, среда, место) при первом появлении бота за этим местом и живёт между эпизодами; память эпизода бот держит в `self`. `self.game_spec` (`GameSpec`) выставляет фреймворк до первого `reset`.
- Действие проходит ту же проверку легальности, что и действие сети. Нелегальное действие или исключение в боте — `PlayerError` с контекстом «воркер, среда, место, шаг эпизода, вариант, агент».
- `info` — `StepResult.infos[seat]` среды: туда среда кладёт «сырое» состояние для ботов (`docs/ENV_GUIDE.md`, раздел 10).
- Встроенный `colosseum.players.RandomBot` — случайное легальное действие.

```yaml
agents:
  main: {}                                   # kind: trainable
  greedy:
    kind: scripted
    class: my_game.bots.Greedy
    kwargs: {aggr: 0.7}
    roles: [player]                          # по умолчанию — все роли игры (пространства ролей могут различаться)
  random:
    kind: scripted
    class: colosseum.players.RandomBot

matchmaking:
  opponents: {latest: 0.6, snapshots: 0.2, rivals: 0.0, anchors: 0.2}
  anchors: {greedy: 2.0, random: 1.0}        # веса внутри категории anchors
```

`anchors: null` (по умолчанию) — все scripted и frozen агенты конфига с равными весами; `anchors: []` — никого (например, бот нужен только как учитель kickstart); список имён — равные веса. Якорь садится ядром команды соперников; места его команды, роли которых он играет, тоже получают его (`teammates: self`). Пример: `configs/examples/team_tag.yaml` — якорь `RandomBot` против пассивной политики «тянуть на ничью»; `configs/examples/unit_harvest_league.yaml` — бот игры и `RandomBot`.

## 4. Замороженный агент (прошлый сабмит)

```yaml
agents:
  main: {}
  prev_sub:
    kind: frozen
    path: runs/sub12/checkpoints/main/ckpt_v9000   # папка чекпоинта: роли, сигнатура и networks из meta.json
  bc_net:
    kind: frozen
    path: bc.pt                                    # .pt: сеть — глобальная networks с этим override
    networks: {kwargs: {hidden: 128}}
    roles: [player]                                # для .pt; по умолчанию все роли с одинаковыми пространствами
```

Frozen-агент может быть якорем, источником `init.from` (раздел 8), учителем kickstart и игроком в `eval` / `record` по имени. Архитектура своя (не обязана совпадать с обучаемым агентом). Для папки чекпоинта `roles` и `networks` указывать нельзя. Чекпоинты SP2 подходят: их `meta.json` содержит сеть, роли и сигнатуру.

## 5. Лига из нескольких агентов

```yaml
agents:
  small: {networks: {kwargs: {hidden: 64}}}
  large: {networks: {kwargs: {hidden: 256}}, algorithm: {learning_rate: 1.0e-4}}

matchmaking:
  opponents: {latest: 0.4, snapshots: 0.2, rivals: 0.3, anchors: 0.1}
```

`rivals` — latest другого обучаемого агента, выбранного по PFSP; эти места тоже собирают данные (арена: один шаг среды кормит двух лёрнеров). `snapshots` — снимки всех агентов, играющих роли команды соперников. Пример: `configs/examples/tic_tac_toe_multi.yaml`.

## 6. Асимметричная игра

В `predator_prey` агент `hunter` играет роль охотника, `prey` — жертвы (`agents.<id>.roles`). Для владельца-охотника команда жертв играется только агентами-жертвами, поэтому:
- `latest` — latest жертвы (выбор по PFSP, если жертв несколько), а доля `rivals` прибавляется к `latest`;
- `snapshots` — снимки жертвы: охотник встречает и прошлые версии соперника;
- `anchors` — якоря с ролью жертвы, например `prey_bot: {kind: scripted, class: ..., roles: [prey]}`.

Проверка конфига требует, чтобы каждую роль каждого варианта играл хотя бы один обучаемый агент или якорь владельца.

## 7. Кооператив со скриптовым напарником

- **Роль, которую агент не играет.** Если в команде есть место роли, которой нет у ядра (например, `pilot` и `gunner`, а обучаемый агент играет только `pilot`), место получает latest обучаемого агента с этой ролью, а если такого нет — якорь владельца с этой ролью (по весам якорей):

```yaml
agents:
  pilot: {roles: [pilot]}
  gunner_bot: {kind: scripted, class: my_game.bots.Gunner, roles: [gunner]}
```

- **Смешанные напарники.** С `teammates: mixed` каждое место роли ядра достаётся ядру с вероятностью `teammate_self_prob`, иначе — равномерно одному из кандидатов: latest других обучаемых агентов с этой ролью, снимки агента ядра, якоря владельца с этой ролью. Так обучаемый агент учится играть и с ботом-напарником. Пример формы: `configs/examples/coop_buttons.yaml`.

## 8. Конвейер: `record` → `bc` → `init` → kickstart

```bash
# 1. Бот играет сам с собой; все его решения — данные BC (папка на роль + record.json)
colosseum record -c configs/examples/unit_harvest_league.yaml --player greedy --num-matches 300 --output data/greedy --seed 0
# 2. BC-сеть агента main
colosseum bc -c configs/examples/unit_harvest_league.yaml --agent main --data data/greedy --output bc.pt --epochs 5
# 3. RL с BC-весов: прогрев критика, kickstart от бота (DAgger), лига с якорями
colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-league \
  --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
```

- **`record`.** `--player` / `--against` — имя scripted или frozen агента либо `name=path` (папка чекпоинта или `.pt`). Без `--against` игрок занимает все места, и записываются все места; если он не играет какую-то роль варианта — ошибка с подсказкой «добавьте `--against`». С `--against` каждый соперник даёт пару с игроком (ротация как в `eval`, `--num-matches` на пару и вариант), записываются только места игрока. Эпизод места лежит в файле непрерывно, `dones` корректны; `record.json` описывает запись и содержит сводку исходов.
- **`bc`.** `--data` можно повторять; для папки с `record.json` берутся подпапки ролей агента.
- **`init`** (секция агента или глобальная): `from` — `.pt`, папка чекпоинта, папка запуска (последний чекпоинт агента с тем же id) или имя frozen-агента; грузятся только веса (версия политики 0, новый оптимизатор). `strict: false` грузит тензоры с совпавшими именем и формой и перечисляет остальные в логе и в выводе `validate`. При `training.resume_from` resume важнее: восстановленный агент игнорирует `init` (запись в лог).
- **Прогрев критика** `critic_warmup_steps: N`: первые N шагов обучения обновляется только value-путь (value-голова и `critic_encoder`), лоссы политики, энтропии и kickstart выключены, статистика нормализаторов не обновляется — политика на воркерах побитово остаётся BC-политикой, пока критик догоняет.
- **Kickstart** (`kickstart: {teacher, lambda, decay_steps, kl}`): `teacher` — имя frozen или scripted агента либо путь (`.pt` или папка чекпоинта). Нейросетевой учитель — KL по решающим (`kl: forward | reverse`), архитектура своя; рекуррентный учитель — только с той же раскладкой состояния, что у ученика. Скриптовый учитель — DAgger: на каждом ходе собирающего места воркер спрашивает бота, его действие идёт в чанк, лёрнер добавляет `λ · mean(−log π(a_учителя))`. λ затухает линейно за `decay_steps` шагов обучения, начиная после прогрева. Имя обучаемого агента учителем быть не может: нужный снимок подключается как frozen-агент с `path`.

Сравнение конвейера с обучением с нуля на равном бюджете — `docs/benchmarks.md`, «Конвейер против обучения с нуля».

## 9. Расписания долей

Любая доля `opponents` и любой вес якоря — число или кусочно-линейное расписание по глобальным env-шагам запуска (до первой точки и после последней значение постоянное):

```yaml
matchmaking:
  opponents:
    latest: {0: 0.8, 2_000_000: 0.5}
    snapshots: {0: 0.0, 2_000_000: 0.3}
    rivals: 0.0
    anchors: {0: 0.2, 2_000_000: 0.2}
  anchors:
    greedy: {0: 1.0, 1_000_000: 0.2}    # бот важен в начале
    random: 1.0
```

Проверка конфига требует, чтобы в каждой точке расписания у каждой команды соперников была хотя бы одна структурно непустая категория с положительной долей.

## 10. Override на агента

`agents.<id>.matchmaking` (deep-merge на глобальную секцию) переопределяет `opponents`, `anchors`, `pfsp`, `layouts`, `teammates`, `teammate_self_prob`; `shuffle_seats` и `matchmaker_class` — только глобальные. Так же устроены `init` и `kickstart`: глобальная секция — дефолт для всех обучаемых агентов, `agents.<id>.init` / `.kickstart` переопределяют.

```yaml
agents:
  main: {}
  explorer:
    matchmaking:
      opponents: {latest: 0.3, snapshots: 0.6, rivals: 0.1, anchors: 0.0}
      pfsp: {weighting: balanced}
```

## 11. Свой матчмейкер

```python
# my_game/league.py
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.league import BaseMatchmaker, MatchmakerContext


class BotFirst(BaseMatchmaker):
    """Первые 500 000 шагов — только против бота greedy, потом — против последнего снимка владельца."""

    def __init__(self, context: MatchmakerContext) -> None:
        super().__init__(context)
        self.ctx = context

    def lineup_for(self, owner: str) -> Lineup:
        me = SeatAssignment(owner, LATEST_NETWORK_ID, collect=True, source=SOURCE_OWNER)
        snapshots = self.ctx.snapshots(owner)
        if self.ctx.env_steps() < 500_000 or not snapshots:
            other = SeatAssignment("greedy", FIXED_NETWORK_ID, collect=False, source="anchors")
        else:
            other = SeatAssignment(owner, snapshots[-1], collect=False, source="snapshots")
        return Lineup("2p", [me, other] if self.ctx.rng.random() < 0.5 else [other, me])
```

```yaml
matchmaking:
  matchmaker_class: my_game.league.BotFirst
```

`MatchmakerContext` даёт `spec`, агентов с видом и ролями (`agents`), обучаемых агентов (`trainable`), итоговый `matchmaking` каждого агента, `snapshots(agent)`, `pfsp_score(layout, owner, (agent, network))`, `env_steps()`, `anchors(owner)` и общий `rng`. `on_result(result)` получает каждый `MatchResult`. Фреймворк проверяет каждый состав (вариант, число мест, роли, существование игроков и снимков, `collect` только у latest обучаемых); нарушение — ошибка с именем класса и составом.

## 12. Хранение снимков

```yaml
checkpoint:
  interval: 1000        # снимок каждые N шагов обучения
  keep_last: 20         # последние 20 (бывший pool_size)
  keep_every: 10        # и каждый 10-й навсегда (версия кратна interval · keep_every); 0 = выкл.
  save_optimizer: true
```

Финальный снимок не удаляется никогда. `trainer_state.pt` есть только у снимков в окне `keep_last` (resume нужен только с последних). Удалённый снимок пропадает из кандидатов матчмейкера и PFSP-статистики, воркеры выгружают его модель. Resume из папки запуска (`training.resume_from: runs/<run>`) переносит хранимые снимки в новый запуск (hardlink или копия), так что пул соперников переживает resume; из папки чекпоинта или `.pt` пул начинается пустым. Лучшие снимки по рейтингу (top-k) — в SP4; до тех пор держите их через `keep_every` или как frozen-агентов.

## 13. Что печатает `validate`

```bash
colosseum validate -c configs/examples/unit_harvest_league.yaml
```

<paste the real output of this command here, verbatim, in a ```text block>

Для каждого обучаемого агента и варианта партии — доли категорий после перераспределения пустых в начальной и конечной точке расписаний и список якорей с весами; для `init` — откуда берутся веса и (при `strict: false`) какие тензоры не загрузились.

## 14. Метрики и рейтинги

- Scripted и frozen агенты — отдельные сущности в ELO, матрице win-rate и `role_win_rates` (`ratings.json`): «main против greedy» видно сразу. Снимки засчитываются своему агенту (рейтинги снимков — SP4).
- PFSP-таблица (счёт и число игр latest владельца против каждого игрока) — в `ratings.json`, по вариантам.
- Доли **сыгранных** эпизодов по категориям и якорям, по агентам и вариантам — в `metrics.jsonl` и WandB: <the record kind and the key names from T3.3, with one example line from a run's metrics.jsonl>. Сыгранные доли могут отличаться от заданных: состав применяется в конце эпизода, а обновление составов заменяет ещё не сыгранные.

## 15. Ограничения

- Распределённый режим (`run-learner` / `run-workers`) — только self-play на latest (объём SP2): scripted и frozen агенты, `init`, прогрев критика, свой матчмейкер, override `matchmaking` и `kickstart` на агента там — ошибка конфига с пометкой SP5; любая смесь соперников сводится к «только latest» с предупреждением.
- Top-k снимков, рейтинги снимков, OpenSkill / Bradley–Terry, `colosseum tournament` — SP4.
- Рекуррентный учитель kickstart с другой раскладкой состояния, батчевый API скриптовых ботов, эксплойтеры — позже, по необходимости (свой матчмейкер закрывает экзотику).
````

- [ ] **Step 5: `docs/ENV_GUIDE.md`: `infos` for scripted bots**

Append after section 9:

````markdown
## 10. `infos` для скриптовых ботов

Скриптовый бот (`colosseum.players.ScriptedBot`, `docs/LEAGUE_GUIDE.md`) получает в `act(obs, mask, info)` наблюдение и маску своего места и `info = StepResult.infos.get(seat)` последнего шага (`None`, если среда ничего не положила). Через `infos` среда отдаёт боту «сырое» состояние, которое нейросети не нужно: координаты, ссылки на объекты движка, словари.

```python
return StepResult(acting={0, 1}, obs=..., action_masks=..., rewards=...,
                  infos={s: {"enemy_base": self._bases[1 - s].copy()} for s in (0, 1)})
```

- Ключ — номер места, значение — что угодно (обычно `dict`). Фреймворк его не проверяет и не копирует; в чанки и BC-данные он не попадает. Читают его только места со скриптовыми ботами (игроки и учителя DAgger), в обучении, `eval` и `record` одинаково.
- `rollout.vec_env: sync`: объект передаётся по ссылке, цены нет. Не меняйте его после `step`: бот может держать ссылку.
- `rollout.vec_env: subprocess`: `StepResult` целиком пиклится из процесса среды в воркер на каждом шаге — вместе с `infos`, даже если ботов в партии нет. Цена — размер `infos`: крупные объекты (вся карта, история) на каждом шаге заметно замедляют воркер. Кладите туда только нужное ботам и компактно (numpy-массивы вместо списков словарей) или включайте их флагом среды в `env.kwargs`, который ставится только в конфигах с ботами.
````

Also add a row to the guide's first table (or the paragraph that lists what a game provides) pointing to section 10, if the guide has such a list (`sed -n 1,20p docs/ENV_GUIDE.md`).

- [ ] **Step 6: README**

Edit `README.md` as follows (English; keep every other section).

1. The status blockquote under the title becomes:

```markdown
> **Status (SP3 of 6).** Single-machine training is the supported mode for every game structure: solo, 1v1 with
> turn-based or simultaneous moves, one bot controlling many units, team vs team, free-for-all with elimination and
> 2–4 players in one run, asymmetric roles, and cooperative games. Each structure has a demo game that is trained
> end-to-end in the test suite. SP3 adds scripted and frozen players, a configurable league (latest weights, PFSP
> over stored snapshots, other agents, scripted/frozen anchors; shares, schedules, per-agent overrides, a custom
> matchmaker), snapshot retention, per-agent warm start (`init`, critic warm-up, kickstart from a neural or scripted
> teacher) and `colosseum record`. Read [Status and limitations](#status-and-limitations) before relying on anything else.
```

2. After the "Other game structures" block of the Quick start, add:

````markdown
### Competition pipeline: bot → BC → league

The game's scripted bot plays itself, its decisions become BC data, the BC weights start RL with a critic warm-up,
the bot itself keeps teaching (DAgger kickstart) and plays as an anchor:

```bash
colosseum record -c configs/examples/unit_harvest_league.yaml --player greedy --num-matches 300 \
  --output data/greedy --seed 0
colosseum bc -c configs/examples/unit_harvest_league.yaml --agent main --data data/greedy --output bc.pt --epochs 5
colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-league \
  --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
NEW=$(ls -d runs/harvest-league/checkpoints/main/ckpt_v* | sort -V | tail -1)
colosseum eval -c configs/examples/unit_harvest_league.yaml -a trained=$NEW -a greedy -a random -a bc=bc.pt \
  --num-matches 100 --deterministic
```

On an 8-core CPU recording takes about <R> s, BC about <B> s and training about <T> s. The trained agent beats
`random` in <W>% of games and scores <S> against `greedy` (win = 1, draw = 0.5; on this game a good policy mostly
draws with the bot). Recipes for leagues, anchors, asymmetric games and custom matchmakers:
[`docs/LEAGUE_GUIDE.md`](docs/LEAGUE_GUIDE.md) (Russian). `data/` and `bc.pt` are yours to delete.
````

   Fill `<R>`, `<B>`, `<T>`, `<W>`, `<S>` by running exactly these commands once from the repo root (fresh `run.name`, then `rm -rf data bc.pt runs/harvest-league`); add `data/` and `bc.pt` to `.gitignore` if they are not covered.

3. Configuration reference table — replace the rows `training`, `matchmaking`, `checkpoint`, `agents` and add `init` and `kickstart`:

```markdown
| `training` | `total_timesteps` 10,000,000 (global env steps), `seed`, `resume_from` (SP2's `kickstart_*` keys are translated into `kickstart` with one warning) |
| `matchmaking` | `opponents` (`latest` 0.7, `snapshots` 0.2, `rivals` 0, `anchors` 0.1; each a number or a schedule `{env_step: share}`), `anchors` (null = every scripted and frozen agent; a list of names or `{name: weight-or-schedule}`), `pfsp` (`weighting` `hard` / `balanced` / `uniform`, `exponent` 2.0, `halflife_games` 200), `layouts` (`{layout: weight}`; empty = every layout equally), `teammates` (`self` / `mixed`), `teammate_self_prob` 0.5, `shuffle_seats` (true), `matchmaker_class` (null; a `colosseum.league.BaseMatchmaker` subclass). SP2's `mode`, `self_play_ratio`, `latest_prob`, `pfsp_exponent` are translated with one warning |
| `checkpoint` | `interval` 1000 (train steps), `keep_last` 20, `keep_every` 10 (every N-th snapshot kept for good; 0 = off), `save_optimizer` (true); the final snapshot is never deleted; SP2's `pool_size` = `keep_last` |
| `init` | `from` (null; `.pt`, checkpoint dir, run dir or a frozen agent's name), `strict` (true; false loads tensors with matching names and shapes), `critic_warmup_steps` 0 (train steps that update only the value path) |
| `kickstart` | `teacher` (null; a frozen or scripted agent's name, a `.pt` or a checkpoint dir), `lambda` 1.0, `decay_steps` 50000 (train steps after the warm-up), `kl` (`forward` / `reverse`, neural teachers) |
| `agents` | `{agent_id: {kind: trainable (default) / scripted / frozen, ...}}`: trainable — `roles`, partial overrides `networks`, `algorithm`, `learner`, `matchmaking`, `init`, `kickstart` (deep-merged onto the global sections); scripted — `class`, `kwargs`, `roles`; frozen — `path` (plus `networks` and `roles` for a `.pt`) |
```

4. The paragraphs **Agents and roles** and **Matchmaking** are replaced by:

```markdown
**Agents, players and roles.** Every key of `agents` is an agent: `trainable` (the default kind) has its own learner;
`scripted` is a `colosseum.players.ScriptedBot` subclass (`colosseum.players.RandomBot` is built in); `frozen` plays
fixed weights from a checkpoint dir or a `.pt`. Without any trainable agent the config gets an implicit `agent_0` with
the global settings. `roles` lists the roles an agent plays (omitted = every role; a trainable or frozen agent's roles
must share their spaces). Only seats with a trainable agent's latest weights collect data. An agent id may contain
letters, digits, `_` and `-` (not starting with `-`), no `.`, and cannot be `ratings`, `system`, `episodes` or `train`.

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index. For each match the built-in
matchmaker picks a layout (`layouts`, only layouts with a seat for the owner's roles), the owner's team (its latest
weights on one seat of its role), and for every other team independently a core category by the `opponents` shares:
the owner's latest weights (or, in an asymmetric game, the latest weights of an agent that plays that team), a stored
snapshot of an agent that plays the team (PFSP), another trainable agent's latest weights (`rivals`, PFSP) or an anchor
(by weight). Shares of categories that are empty right now go to the others. The team's other seats follow
`teammates`. `docs/LEAGUE_GUIDE.md` has the full rules and recipes; `colosseum validate` prints each agent's effective
mix.
```

5. In **Ratings**, append: "Scripted and frozen agents are entities of their own in the ELO and win-rate tables; snapshots count for their agent. `ratings.json` also holds the PFSP table per layout, and `metrics.jsonl` the shares of played episodes per opponent category and anchor."

6. The section **Behavioural cloning and kickstarting** becomes:

````markdown
## Recording, behavioural cloning and warm start

```bash
colosseum record -c cfg.yaml --player <scripted|frozen agent | name=path> [--against <player> ...] [--layout L ...] \
  --num-matches N --output data/x [--num-envs E] [--seed S] [--deterministic]
colosseum bc -c cfg.yaml --agent main --data data/x [--data more.pt ...] --output bc.pt --epochs 20
```

`record` plays the player on the eval engine and writes its decisions per role (`<output>/<role>/part-NNNNN.pt`,
every seat-episode contiguous) plus `record.json`. Without `--against` the player takes every seat; with it, each
opponent forms a pair with the player as in `eval`, and only the player's seats are recorded. `bc` reads `.pt` files,
directories of them and `record` output directories (the folders of the agent's roles), `--data` repeatable; other
options: `--batch-size` (256), `--lr` (1e-3), `--seq-len` (default `bc.seq_len`). The loss is `-log_prob` of the
recorded action (the mean over valid deciders for actions with units); masks are applied, and an expert action the
mask forbids is a data error.

Warm start is per agent (`init`, `kickstart`; global sections are the defaults): `init.from` loads weights only
(strict, or partial with `strict: false`), `critic_warmup_steps` first trains the value path alone while the workers
keep the initial policy bit for bit, and `kickstart` adds `lambda * KL(teacher‖student)` per decider (a neural teacher
of any architecture) or `lambda * -log pi(teacher's action)` (a scripted teacher, DAgger labels written by the
workers), decaying linearly over `decay_steps` after the warm-up. `training.resume_from` takes precedence over `init`.
````

7. In **Evaluation**: the command line becomes `colosseum eval -c cfg.yaml -a A=<ckpt-dir|.pt> [-a <scripted or frozen agent>] ...`; add after the first paragraph: "A bare name (`-a greedy`) plays a scripted or frozen agent of the config; a trainable agent needs `name=path`." and change the in-process sentence to "...plays any `PolicyModel` instances and scripted players (`colosseum.worker.match_runner.ScriptedPlayer`) in explicit lineups (the learning tests use it with `RandomBot`)."

8. In **Distributed mode (limited)**, append to the limitations paragraph: "SP3 features are local only: scripted and frozen agents, `init`, critic warm-up, a custom matchmaker, per-agent `matchmaking` or `kickstart` are refused with a config error pointing to SP5, and any opponent mix is reduced to latest weights with one warning."

9. **Status and limitations:** the SP3 row becomes `| ✔ | SP3 Players, league, warm start | scripted and frozen players, opponent mixes with PFSP over snapshots and anchors, snapshot retention, per-agent init / critic warm-up / kickstart (neural or DAgger), `colosseum record` (implemented on `sp3-league`; acceptance report in docs/superpowers/reports) |`; the SP4 row gains "top-k snapshot storage"; the SP5 row gains "distributed league, scripted/frozen players and SP3 warm start across machines". The **Players** limitation bullet becomes: "**Players:** external players are frozen agents (`.pt` or checkpoint dir); no top-k snapshot storage or snapshot ratings yet (SP4); online ELO is a progress indicator, not a selection-grade rating; scripted bots run one Python call per seat (no batched bot API)."

10. **Tests:** the fast-suite count and the slow-suite description from `.venv/bin/python -m pytest -m "not gpu and not slow" --collect-only -q | tail -1` and `-m slow` (the slow suite: the 7 demo-game learning tests, the unit_harvest pipeline test and 2 torch.compile checks; its duration from T6.4 Step 6).

11. **Project structure:** add `players/` (scripted bots, fixed-player registry), `league/` (matchmaker interface, mixture matchmaker, PFSP, schedules), `record.py`, `learner/factory.py`; remove `coordinator/agent_pool.py` and `coordinator/matchmaker.py` if listed.

- [ ] **Step 7: CLAUDE.md**

Edit `CLAUDE.md` (English):

1. Every "(target; see Roadmap — SP3 ...)" parenthetical that SP3 implemented now describes the implementation (`grep -n "SP3" CLAUDE.md`): line about Online BC (kickstarting) — "a teacher per agent: `kickstart` section per agent, any frozen/scripted teacher"; Phase 2 Online BC — "scripted teachers (DAgger) exist since SP3"; PFSP priority — "`matchmaking.pfsp` `hard` / `balanced` / `uniform`, over latest weights and stored snapshots of every agent that plays the team"; FrozenCheckpoint / ScriptedAgent — implemented (`agents.<id>.kind`); FIFO pool — "`keep_last` + `keep_every` + final; top-k is SP4"; Mode B — implemented as the scripted kickstart teacher. Keep the remaining SP4/SP5 targets as targets.
2. **Implementation Status:** "State after SP3 (players, league, warm start; developed on branch `sp3-league`, ...)" with the test counts from Step 6.10; the SP3 records (spec, plan dir, acceptance report path, review dir) in the same form as the SP2 line.
3. **Commands:** add `colosseum record -c cfg.yaml --player P [--against A ...] --num-matches N --output DIR [--layout L ...] [--seed S] [--deterministic]`; `bc` with repeatable `--data` (record dirs); `eval -a name` for scripted/frozen agents; the measurement scripts `scripts/team_tag_anchors.py`, `scripts/pipeline_vs_scratch.py`, `scripts/units_experiment.py --cells`.
4. **Works (single machine):** add bullets **Players** (kinds, implicit `agent_0`, `ScriptedBot`/`RandomBot`, `fixed` network id, `infos`, `PlayerError`, frozen of any architecture, never collecting), **League** (v3 mix, schedules, overrides, SP2 knob translation, `BaseMatchmaker`, lineup checks, PFSP statistics per player with half-life, fixed agents in ratings, played shares in metrics, validate printout), **Snapshot storage** (`keep_last`/`keep_every`/final, `trainer_state` window, evictions to workers, pool carried over a run-dir resume), **Warm start** (per-agent `init` strict/partial with report, critic warm-up, kickstart neural any architecture / scripted DAgger with `teacher_action`, resume precedence), **record / BC** (record, record dirs in bc); update the Demo games bullet (bots, `unit_harvest_league.yaml`, team_tag single run with its anchor share) and the Throughput bullet (the T6.6 numbers); the Algorithm bullet states the `Units` defaults after T6.5.
5. **Partial:** the distributed bullet gains the SP3 guard sentence (as in README).
6. **Not implemented:** remove the SP3 line; add "top-k snapshot storage, snapshot ratings (SP4); distributed league with scripted/frozen players and SP3 warm start (SP5); recurrent kickstart teacher with another state layout, batched scripted-bot API, exploiters, preset mixes, value pre-training on BC data, a one-command phase pipeline (later, as needed)".
7. **Roadmap:** `**SP3 Players, league, warm start** (done)`; the SP3 row text describes what was built; SP4 row gains "top-k snapshot storage".
8. **Open items:** add «Parked during SP3» with every deferred item from the SDD ledger (`grep -n "Deferred\|Parked\|SP4\|SP5\|SP6" .superpowers/sdd/*/progress.md`), each with its addressee (SP4 / SP5 / SP6 / minor), and remove the SP2 items SP3 closed: the team_tag best-of-two item, the `ratio_mode` / `unit_trace` open questions (closed by T6.5's ruling), the `agent_pool.py` legacy minor, the owner-rotation note (now in the registry).
9. **Next step:** SP4 and SP5 (may run in parallel), each starting with a brainstorm and a spec; list their design inputs (the SP4/SP5 rows, «Parked during SP3», SP3 spec section 2). Keep "GPU checks still pending" if still true.

- [ ] **Step 8: Verify the docs against the code**

```bash
.venv/bin/colosseum record --help; .venv/bin/colosseum bc --help; .venv/bin/colosseum eval --help
grep -n "colosseum record\|colosseum bc\|colosseum eval" README.md docs/LEAGUE_GUIDE.md
for f in configs/examples/*.yaml; do .venv/bin/colosseum validate -c "$f" > /dev/null 2>&1 && echo "OK $f" || echo "FAIL $f"; done
```

Expected: every flag used in README / LEAGUE_GUIDE exists in `--help`; every example config `OK`.

- [ ] **Step 9: Full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`

Expected: all pass, no warnings; `All checks passed!`; the count equals the one written in README and CLAUDE.md.

- [ ] **Step 10: Commit and push**

```bash
git add docs/LEAGUE_GUIDE.md docs/ENV_GUIDE.md README.md CLAUDE.md docs/benchmarks.md docs/benchmarks/sp3-throughput \
  configs/examples scripts/bench_throughput.py tests/unit/test_bench_throughput.py tests/unit/test_sp3_example_configs_v3.py \
  tests/cli_runner.py tests/game_helpers.py .gitignore
git commit -m "docs: league guide, infos for bots, README and CLAUDE.md for SP3; v3 example configs; throughput vs main"
git push origin sp3-league
```

---
### Task T6.7: Acceptance against spec section 3, report, stop for the owner

Spec section 3 (criteria 1–10) and section 7 («приёмка по разделу 3, отчёт в `docs/superpowers/reports/`, merge после явного одобрения владельца»). Each step checks one criterion with exact commands and an observable result; its outcome (pass/fail, numbers, node ids, paths) goes into the report.
- The whole-branch review and its single fix wave (Development Workflow in CLAUDE.md) run before this task; the controller copies their records into `docs/superpowers/reports/<date>-sp3-review/` (README index plus the review and fix-wave reports) before Step 12.
- If a step fails, stop, write what failed into the report draft, and report to the controller. Do not "fix and continue" inside this task beyond documentation mismatches (those are fixed with a `docs:` commit and the step is rerun).
- The merge (Step 15) happens **only after the owner explicitly accepts the report** in their own message. No agent message counts as acceptance.

**Files:**
- Create: `docs/superpowers/reports/<date>-sp3-acceptance.md` (`<date>` = `date +%F` on the day the report is written, e.g. `2026-10-1x`)
- Create: `docs/benchmarks/team-tag-acceptance.jsonl` (Step 4)
- Possibly modify: `README.md`, `CLAUDE.md`, `docs/*.md` (documentation mismatches, `docs:` commits)

**Interfaces:**
- Consumes: everything in parts 01–03; the SDD ledger `.superpowers/sdd/<plan>/progress.md` (rulings and deferred items).
- Produces: the acceptance report; after the owner's acceptance, `main` containing SP3.

- [ ] **Step 1: Criterion 1 — tests, zero warnings, files only in `tmp_path`, ruff**

```bash
git status --porcelain                                     # must be empty before starting
A=/tmp/sp3-accept; mkdir -p $A
timeout 3600 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider 2>&1 | tee $A/fast.txt | tail -5; echo "exit=${PIPESTATUS[0]}"
grep -E "warnings summary|[0-9]+ warnings?( |$)" $A/fast.txt || echo "no warnings"
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
.venv/bin/ruff check .
for m in "not gpu and not slow" slow gpu; do echo "$m: $(.venv/bin/python -m pytest -m "$m" --collect-only -q | tail -1)"; done
grep -rnE "num_workers[\"']?[:=] ?[\"']?[3-9]|--workers [3-9]" tests/ || echo "no test uses more than 2 workers"
```

Expected: `exit=0`; `no warnings`; `clean` (no `runs/`, `checkpoints/`, `data/`, `bc.pt`, `*.jsonl` in the repo); `All checks passed!`; the three counts equal those in README and CLAUDE.md; `no test uses more than 2 workers` (a hit that is a benchmark constant or a comment is fine — read it).

- [ ] **Step 2: Criterion 2 — contract tests**

```bash
.venv/bin/python -m pytest $(ls tests/*/test_sp3_*.py) tests/contract -v -rA -m "not slow" 2>&1 | tee $A/contract.txt | tail -5
```

Expected: everything passes. For each item of criterion 2 find the passing test(s) in `$A/contract.txt` and list their node ids in the report:

| Item (spec criterion 2 / section 6) | Task that wrote the test |
|---|---|
| a scripted bot at a seat gets its seat's `infos`; `reset` with `rng`; an illegal action and an exception in the bot are errors with context | T1.3 |
| a frozen agent of another architecture plays and never collects; `collect` only on latest trainable seats | T1.3, T1.4 |
| the worker unloads evicted snapshots | T2.2 |
| actual category shares converge to the configured ones, incl. redistribution of empty categories, schedules and per-agent overrides | T3.2 |
| an asymmetric game seats a snapshot of the other agent | T3.2 |
| lineups of a custom matchmaker class are checked | T3.2 |
| snapshot retention (`keep_last` / `keep_every` / final; `trainer_state.pt` only inside `keep_last`; the pool carried over a run-dir resume) | T2.1 |
| `init` strict and partial | T4.2 |
| critic warm-up keeps the workers' policy bit for bit | T4.3 |
| the scripted teacher's action lies in the chunk and gives a loss (reproduced by the learner) | T4.5 |
| `record` data go through `bc` | T5.2 (`test_sp3_bc_record.py`) |
| SP2 compatibility (criterion 6) | T0.1 (Step 6) |

An item without a passing test fails this criterion.

- [ ] **Step 3: Criteria 3 and 10 — the slow suite (pipeline, team_tag, SP2 learning tests)**

```bash
nproc; uptime
.venv/bin/python -m pytest tests/learning tests/integration -m "not slow" -k "sp3" -v --durations=0 2>&1 | tail -12
nohup .venv/bin/python -m pytest -m slow -v -s -p no:cacheprovider > $A/slow.txt 2>&1 &
# when it finished (about 20-25 minutes):
grep -E "^\[|PASSED|FAILED" $A/slow.txt; tail -3 $A/slow.txt
```

Expected: the fast SP3 learning tests and the tic-tac-toe pipeline smoke pass, each under 20 s; `nproc` = 8; every slow test PASSES: the 7 demo-game learning tests (team_tag as one run), the unit_harvest pipeline test (printed values at or above its T6.4 thresholds) and the 2 torch.compile checks. Compare the printed values with T6.3/T6.4 and SP2: a value much lower than measured there means something changed since; investigate before going on. Copy the «Конвейер против обучения с нуля» table of `docs/benchmarks.md` (T6.4) into the report under criterion 3.

- [ ] **Step 4: Criterion 4 — team_tag on 6 fresh seeds**

Preconditions: the slow suite of Step 3 has finished; `uptime` load below 1.0.

```bash
nohup .venv/bin/python scripts/team_tag_anchors.py --anchors <the T6.3 share> --seeds 10 11 12 13 14 15 \
  --jsonl docs/benchmarks/team-tag-acceptance.jsonl --run-dir /tmp/sp3-team-tag-accept > $A/team_tag.txt 2>&1 &
# about 20 minutes, then:
tail -8 $A/team_tag.txt
```

(Add `--draw-reward <d>` only if T6.3 took the fallback path, with that ruling's value.) Seeds 10–15 are new: the share was chosen on seeds 0–5. Expected: every row has `win` ≥ 0.80 (the rule's output names the share). A seed below 0.80 fails criterion 4: stop and report (the fallback decision belongs to the controller and the owner, with these numbers).

- [ ] **Step 5: Criterion 5 — speed**

```bash
ls docs/benchmarks/sp3-throughput/
git log --oneline -1 -- docs/benchmarks/sp3-throughput
git diff --stat $(git log --format=%h -1 -- docs/benchmarks/sp3-throughput)..HEAD -- src/colosseum/worker src/colosseum/learner \
  src/colosseum/algorithms src/colosseum/envs src/colosseum/networks src/colosseum/core src/colosseum/coordinator src/colosseum/league
sed -n '/## После SP3/,/^## [^П]/p' docs/benchmarks.md | head -20
```

Expected: the «После SP3» table shows branch ≥ 95 % of `main` at 1 / 2 / 4 workers and a monotonic 1 → 2 → 4. If the `git diff --stat` lists changes to those directories since the measurement (e.g. the review fix wave), rerun T6.6 Step 3 (both `main` and the branch, alternating), update the table and the JSON (`docs:` commit) and use the new numbers.

- [ ] **Step 6: Criterion 6 — SP2 compatibility**

```bash
.venv/bin/python -m pytest $(grep -rlE "fixtures.{0,6}sp2" tests --include="test_*.py") -v 2>&1 | tail -15
for f in tests/fixtures/sp2/configs/*.yaml; do
  .venv/bin/colosseum validate -c "$f" > $A/validate-$(basename $f).txt 2>&1 && echo "OK $f" || echo "FAIL $f"
done
grep -l "translat\|deprecat" $A/validate-*.txt | head -3
```

Expected: the T0.1 compatibility tests pass (every SP2 config copy validates; the SP2 checkpoint fixture loads as a frozen agent, as `init` and for resume; old knobs are translated, also partially given ones; a resolved SP3 config reloads without warnings or errors); every copy `OK`; warnings about old knobs appear (allowed, criterion 6). List the node ids in the report.

- [ ] **Step 7: Criterion 7 — APPO defaults measurement**

```bash
sed -n '/## Дефолты APPO при `Units`/,/^## [^Д]/p' docs/benchmarks.md | head -40
.venv/bin/python -c "import json; d = json.load(open('docs/benchmarks/units-defaults-sp3.json')); print(len(d['rows']), json.dumps({q: (v['switch'], v['choice']) for q, v in d['decisions'].items()}))"
.venv/bin/python -m pytest tests/unit/test_appo_v2.py -k "modes_resolve" tests/unit/test_sp3_units_defaults_rule.py -q
```

Expected: the section has the 18-row table and the decision table; the JSON has 18 rows and two decisions; the code default (`resolve_modes`, pinned by `test_modes_resolve_auto_and_collapse_for_one_decider`) matches the ruling.

- [ ] **Step 8: Criterion 8 — documentation**

In a fresh shell (`env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc`), from the repo root, run the README "Competition pipeline" block exactly as written (after `source .venv/bin/activate`), then the README Quick start's tic-tac-toe commands:

```bash
source .venv/bin/activate
colosseum record -c configs/examples/unit_harvest_league.yaml --player greedy --num-matches 300 \
  --output data/greedy --seed 0
colosseum bc -c configs/examples/unit_harvest_league.yaml --agent main --data data/greedy --output bc.pt --epochs 5
colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-league \
  --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
NEW=$(ls -d runs/harvest-league/checkpoints/main/ckpt_v* | sort -V | tail -1)
colosseum eval -c configs/examples/unit_harvest_league.yaml -a trained=$NEW -a greedy -a random -a bc=bc.pt \
  --num-matches 100 --deterministic
colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
```

Expected: every command exits 0; no translation warning is printed for the example configs; the times and results agree with the README's numbers (T6.6); the `eval` report has the rows trained vs greedy / random / bc. Then:
- compare every flag in README and `docs/LEAGUE_GUIDE.md` with `colosseum <command> --help`;
- check that `docs/LEAGUE_GUIDE.md` has every recipe of spec block 11 (one agent in self-play; scripted anchors; a league of several agents; asymmetry; cooperative with a scripted teammate; `record` → `bc` → `init` → kickstart; schedules; a custom matchmaker; reading the `validate` mix): `grep -n "^## " docs/LEAGUE_GUIDE.md`;
- check that `docs/ENV_GUIDE.md` has the section «`infos` для скриптовых ботов» with the `subprocess` cost;
- check that README "Status and limitations" and CLAUDE.md "Implementation Status" / "Roadmap" / open items describe what Steps 1–7 observed.

Clean up: `rm -rf runs data bc.pt`.

- [ ] **Step 9: Criterion 9 — distributed mode**

```bash
.venv/bin/python -m pytest tests -v -k "distributed or grpc" 2>&1 | tail -15
rm -rf /tmp/sp3-acceptance
.venv/bin/colosseum serve-weight-store --port 50051 & WS=$!
.venv/bin/colosseum run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0 --traj-port 50052 \
  --weight-store localhost:50051 --set run.dir=/tmp/sp3-acceptance --set training.total_timesteps=20000 & LR=$!
sleep 10
.venv/bin/colosseum run-workers -c configs/examples/tic_tac_toe.yaml --weight-store localhost:50051 -l agent_0=localhost:50052 \
  --set run.dir=/tmp/sp3-acceptance --set training.total_timesteps=20000; echo "workers exit=$?"
kill $LR $WS; wait
ls /tmp/sp3-acceptance; ls /tmp/sp3-acceptance/*/checkpoints/agent_0 2>/dev/null | tail -3
.venv/bin/colosseum run-workers -c configs/examples/team_tag.yaml --weight-store localhost:50051 -l agent_0=localhost:50052 \
  --set run.dir=/tmp/sp3-acceptance; echo "scripted agent exit=$?"
.venv/bin/colosseum run-learner -c configs/examples/unit_harvest_league.yaml --agent main --traj-port 50052 \
  --weight-store localhost:50051 --set run.dir=/tmp/sp3-acceptance; echo "league learner exit=$?"
rm -rf /tmp/sp3-acceptance
```

The weight store is already stopped for the last two commands on purpose: the scope check happens before any connection.

Expected: the distributed tests pass (SP2 and `test_sp3_distributed_guards.py`); `workers exit=0`; the learner logged one "reduced to latest only" warning (the example config translates to latest 0.8 / snapshots 0.2) and its run dir has checkpoints kept by `keep_last`/`keep_every`; the last two commands fail fast with `exit=1` and a `Config error:` line naming SP5 and the scripted agents.

- [ ] **Step 10: Hygiene**

```bash
git status --porcelain
git status --porcelain --ignored | grep -vE "\.venv/|__pycache__|\.egg-info|\.ruff_cache|\.pytest_cache|\.superpowers/" || echo "clean"
ps -eo pid,args | grep -E "colosseum|multiprocessing|bench_throughput|team_tag_anchors|units_experiment" | grep -v grep || echo "no leftover processes"
git worktree list
```

Expected: an empty status (except the report being written), `clean`, `no leftover processes`, only the main worktree.

- [ ] **Step 11: Collect the rulings and the deferred items**

```bash
ls .superpowers/sdd/
grep -n "Ruling:" .superpowers/sdd/*/progress.md | wc -l
grep -n "Ruling:" .superpowers/sdd/*/progress.md > $A/rulings.txt
grep -nE "Deferred|deferred|Minor|minor" .superpowers/sdd/*/progress.md > $A/deferred.txt
```

Every ruling goes into the report verbatim, in ledger order ("что — почему — цена ошибки"), grouped as: plan-stage rulings (the overview's contract amendments A1…), pre-flight rulings, task rulings (T0.1 … T6.6, including the measured values: T6.3 anchors share, T6.4 thresholds and budgets, T6.5 defaults), review and fix-wave rulings. Every deferred minor goes into the appendix verbatim; the open items section lists those addressed to later sub-projects.

- [ ] **Step 12: Write the report**

Create `docs/superpowers/reports/<date>-sp3-acceptance.md` (Russian) with the structure of the SP2 report (`docs/superpowers/reports/2026-10-09-sp2-acceptance.md`):

```markdown
# SP3: приёмка по спеке §3 (T6.7)

- Ветка `sp3-league`. Начало приёмки: HEAD `<sha>`. Правки документации во время приёмки: <shas или «нет»>. Отчёт — следующий коммит.
- Дата: <date>.
- Машина: <nproc> ядер, <RAM> ГБ, без GPU. Python <версия>, torch <версия>, `.venv`.
- Записи финального ревью и волны исправлений: [`<date>-sp3-review/`](<date>-sp3-review/README.md).
- Шаг 15 (merge) не выполнялся: ждёт явного одобрения владельца.

## Итог

<ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ / список невыполненных; одна фраза о вопросах владельцу>

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | <PASS/FAIL: N passed, M deselected, время, предупреждений 0; ruff; дерево чистое; счёт N / slow / gpu совпадает с README и CLAUDE.md> |
| 3.2 | Контрактные тесты | <число тестов; у каждого пункта есть зелёный тест> |
| 3.3 | Конвейер | <медленный тест: значения против порогов T6.4; смоук: длительность; конвейер против обучения с нуля: средний score против бота> |
| 3.4 | team_tag с одного прогона | <медленный тест одним прогоном: win-rate; 6 новых сидов: минимум и все значения; доля якоря> |
| 3.5 | Скорость | <шаги сред/с ветки и main при 1/2/4 воркерах, доля, монотонность> |
| 3.6 | Совместимость с SP2 | <конфиги-копии: N/N OK; чекпоинт SP2 как frozen, init, resume: node ids> |
| 3.7 | Дефолты APPO | <решения по двум вопросам с числами; дефолт в коде> |
| 3.8 | Документация | <LEAGUE_GUIDE (рецепты), ENV_GUIDE (infos), README дословно, CLAUDE.md> |
| 3.9 | Распределённый режим | <тесты; localhost smoke; отказ с SP5 для scripted-агентов и лиги> |
| 3.10 | Slow-набор | <все N тестов, время> |

## §3.1 … §3.10
<по разделу на критерий: команда, вывод, node ids, числа>

## Замеры
### team_tag: доля якоря (T6.3) и проверка приёмки (шаг 4)
### Конвейер на unit_harvest: калибровка порогов (T6.4) и конвейер против обучения с нуля
### Дефолты APPO при `Units` (T6.5)
### Скорость (T6.6)

## Отклонения и замечания
<каждое отклонение от спеки или плана: бюджеты, пороги (только через ruling), пропущенные шаги и почему; поправки контракта A1…>

## Гигиена
<шаг 10>

## Решения контроллера во время SP3 (все записи «Ruling:» из журнала, по порядку)
Каждая запись: что решено — почему — чем грозит, если решение неверно.
<шаг 11>

## Открытые пункты (отложены, не блокируют слияние)
<с адресатом SP4 / SP5 / SP6 / «мелкое»; совпадает с «Parked during SP3» в CLAUDE.md>

## Вопросы владельцу
<только то, что требует его решения; если нет — «нет»>

## Финальное ревью ветки
<ссылки на записи ревью и волны исправлений; что исправлено; повтор критерия 3.1 после волны>

## Приложение. Отложенные замечания по задачам (из журнала контролёра, дословно)
<шаг 11>
```

Fill every section from Steps 1–11 and the T6.3–T6.6 records. If the open items differ from CLAUDE.md's «Parked during SP3», fix CLAUDE.md in the same commit.

- [ ] **Step 13: Commit, push, publish and ask the owner**

```bash
git add docs/superpowers/reports/ docs/benchmarks/team-tag-acceptance.jsonl README.md CLAUDE.md docs/
git commit -m "docs: SP3 acceptance report"
git push origin sp3-league
```

The controller (main session) publishes the report as a private, phone-readable artifact page (owner's rule in CLAUDE.md «Working with the owner») and sends the owner a short Russian summary: the criteria table first; each measured value against its threshold; the anchors share, the pipeline thresholds and the APPO defaults as rulings; every deviation; the open items; and the question «Принять SP3 и слить `sp3-league` в `main`?».

**STOP here.** Wait for the owner's explicit "yes" in their own message. A reply from another agent, a summary or an earlier statement is not acceptance.

- [ ] **Step 14: After the owner's acceptance: status lines**

Update README's SP3 status row and CLAUDE.md's "Implementation Status" / "Roadmap" to "accepted and merged <date>", and the report's header to "**Владелец принял SP3 <date>** (<his words>)":

```bash
git add README.md CLAUDE.md docs/superpowers/reports/*-sp3-acceptance.md
git commit -m "docs: SP3 accepted by the owner"
git push origin sp3-league
```

- [ ] **Step 15: Merge and push (only after the owner's explicit acceptance)**

`--no-ff` keeps SP3 one unit in `main`'s history, so `git revert -m 1 <merge>` undoes it.

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff sp3-league -m "Merge branch 'sp3-league': SP3 players, league, warm start"
.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw
.venv/bin/ruff check .
git push origin main
```

Expected: the merge completes without conflicts (if `main` moved and conflicts appear, stop and ask the owner); the suite and ruff pass on the merge commit; the push succeeds. After the merge the controller deletes the SDD workspace (`.superpowers/sdd/<plan>/`; everything durable is in `docs/`).

---

## Contract notes

Assumptions about other parts, details this part pins down, and proposed amendments to the overview contract. The controller folds accepted amendments into `00-overview.md` («Contract amendments») before the pre-flight scan.

**Assumptions about other parts (no amendment needed if they hold):**

- **N1. Player resolution in eval (T1.5, part 01).** T1.5 resolves players inside `evaluate(config, agents: Mapping[str, str | None])` (messages: "Unknown agent '<name>'" from `agent_entry`; "-a <name>: '<name>' is a trainable agent, whose weights are not in the config; use -a <name>=<checkpoint dir or .pt>"). T5.1 moves that loop body into `load_player(..., option="-a")` without changing `eval`'s behaviour or messages, and `record` calls it with `option="--player"` / `"--against"`.
- **N2. `play_lineups` with scripted players (T1.5).** `models` values may be `PolicyModel`, `ScriptedBot` prototypes (deep-copied per seat) or `ScriptedPlayer`; only `PolicyModel`s are switched to eval mode. T6.2's tests and T5.1 pass `ScriptedPlayer`s.
- **N3. `MatchObserver.on_episode_start` (T1.3)** is optional; the eval collector forwards it (T5.1) only if the inner observer defines it.
- **N4. `ActRecord` for scripted seats (T1.3):** `obs` is the role-dtype numpy copy, `action` the numpy tree sent to the env (any layout `put_row` accepts), `mask` the tracker's normalized mask. `record` normalizes actions into the `ActionSpec.allocate_actions` layout itself, so a bot returning a Python `int` or a list is fine.
- **N5. `--set agents.<id>.<section>.<key>=...` works for every kind (T1.1, spec block 1)**, including nested keys of a trainable agent's `init` / `kickstart` (`agents.main.init.from=bc.pt`); the README pipeline and LEAGUE_GUIDE depend on it. `unit_harvest_league.yaml` writes `init: {from: null, critic_warmup_steps: 0}` and `kickstart: {teacher: null}` for `main` so the keys exist.
- **N6. T2.1 / T3.1 updated the test-support defaults and the benchmark config** (`cli_runner.TINY`, `game_helpers.make_test_config`, `bench_throughput._make_config`) away from `pool_size` / SP2 matchmaking knobs, because those are model-validated dicts, not YAML input. T6.6 Step 1–2 checks it and fixes leftovers. `pipeline_kit.SMOKE_SETTINGS` does not use `TINY` for that reason.
- **N7. The SP2 fixture layout (T0.1):** SP2 config copies at `tests/fixtures/sp2/configs/<same file name>.yaml` (used by T6.1's test and T6.6's v3 comparison). If T0.1 chose another directory, adjust the two path constants.
- **N8. Translation warnings go through `logging` under a `colosseum.*` logger** (overview, global constraints); T6.6's test filters `caplog` records by the `colosseum` prefix.
- **N9. `get_agent_config(aid)` of an agent without overrides returns sections equal (`==`) to the global ones**, so T6.1 can compare effective `matchmaking` / `kickstart` by value. If T3.1/T4.1 normalize those sections inside `get_agent_config`, T6.1 compares normalized forms (stated in the task).
- **N10. `TeacherSpec.bot` is a `BotSpec`, and `build_algorithm(..., teacher=TeacherSpec(kind="scripted"))` returns an algorithm whose kickstart term uses the chunk's `teacher_action` / `has_teacher` (T4.5)**; `RolloutLoop(teachers={agent: BotSpec})` writes the labels only while the agent's `WeightPayload.teacher_active` is true. T6.2's fast test sets `teacher_active` by assigning the field after `WeightPayload.from_model` (the dataclass is mutable).

**Details this part pins down (new names, all additive):**

- **D1. `colosseum.record`:** `RECORD_FILE = "record.json"`, `RECORD_FORMAT = 1`, `DECISIONS_PER_FILE = 100_000`, `RoleWriter(directory, role, decisions_per_file=DECISIONS_PER_FILE)` with `add_episode(decisions)` / `flush()` / `files` / `seat_episodes` / `decisions`, `RecordObserver(writers, player)`; `record.json` keys as listed in T5.1. Without `--against` and without `--layout`, layouts the player cannot fill alone are skipped and a `ConfigError` ("add --against") is raised only if none is left or an explicit `--layout` cannot be filled — the spec's sentence read for the default-layout case.
- **D2. `play_lineups(..., observer: MatchObserver | None = None)`** and the optional observer hook `on_episode_discarded(env)` (eval engine only; `MatchRunner` never calls it).
- **D3. `colosseum.bc.offline_bc.bc_data_sources(paths, roles) -> list[Path]`**; the `bc` CLI's final line gains the decision count.
- **D4. `colosseum.distributed.check_distributed_scope(config) -> None`** and **`distributed_checkpoint_manager(config, base_dir) -> CheckpointManager`**; `distributed_setup` calls the check before `validate_config`.
- **D5. Test and script support:** `tests/pipeline_kit.py`, `tests/learning/sp3_bandits.py`, `demo_learning.AgentSetup.algorithm_fn`, `train_in_process(fixed_players=, teachers=, on_chunk=)`, `scripts/team_tag_anchors.py`, `scripts/pipeline_vs_scratch.py`, `scripts/units_experiment.py --cells` / `decide_defaults`.
- **D6. New example config** `configs/examples/unit_harvest_league.yaml` (agents `main`, `greedy`, `random`), and the v3 form of every example config (T6.6; behaviour equal to the translation of the SP2 form).
- **D7. `TeamTagGame(draw_reward=0.0)`** exists only if T6.3 takes the fallback path (a ruling).

**Proposed amendments to `00-overview.md`:**

- **A-P3.1** — File map: add `src/colosseum/eval.py` to the files T5.1 modifies (`load_player` extracted from `evaluate`; `play_lineups(observer=...)`); the overview lists only T1.2 and T1.5. Reason: `record` must see every decision of the scheduled episodes and drop the discarded ones; the eval engine's collector is the only place that knows which episode is scheduled.
- **A-P3.2** — Interface contract, `colosseum.eval`: add `load_player(config, name, path, *, spec, validated=None, option="-a") -> tuple[PolicyModel | ScriptedPlayer, list[str]]` (produced by T5.1 as an extraction of T1.5's `evaluate` loop; `evaluate` calls it) and `play_lineups(..., observer=None)` (T5.1). Reason: `eval -a` and `record --player/--against` must resolve players by the same rules (spec block 7: «правила `eval -a`»); one function instead of two copies.
- **A-P3.3** — File map: add `configs/examples/unit_harvest_league.yaml` (T6.2), `tests/pipeline_kit.py` (T6.2), `tests/learning/sp3_bandits.py` (T6.2), `scripts/team_tag_anchors.py` (T6.3), `scripts/pipeline_vs_scratch.py` (T6.4), `docs/benchmarks/{team-tag-anchors.jsonl,team-tag-acceptance.jsonl,pipeline-vs-scratch.json,units-defaults-sp3.json,sp3-throughput/}`. Reason: completeness of the "Create" list.
- **A-P3.4** — Interface contract, `colosseum.record`: add `RECORD_FILE` (imported by `bc_data_sources`, T5.2) and `bc_data_sources` under a `colosseum.bc.offline_bc` entry (T5.2). Reason: T5.2 depends on T5.1's constant.
- **A-P3.5** — Interface contract, `colosseum.distributed` (T6.1): `check_distributed_scope(config)`, `distributed_checkpoint_manager(config, base_dir)`. Reason: T4.1 also edits `distributed.py`; pinning the names avoids a second `CheckpointManager` construction path.
- **A-P3.6** — Cross-part note: test-support defaults (`cli_runner.TINY`, `make_test_config`, `bench_throughput._make_config`) must be in the v3 form after T2.1 / T3.1 (N6). Reason: once example or test configs use `keep_last` / `opponents`, an old knob injected by a shared default turns into "old + new together" = ConfigError.
- **A-P3.7** — Task index: T6.5 depends on T4.5 (the final training code) and on T6.4 for scheduling only (measurements never overlap), not on T6.1 (it uses nothing from the distributed guards). Execution order unchanged. Reason: accurate dependencies for the controller's scheduling.

# SP4 Plan — Part 1: Compatibility fixture, match log, ratings, metrics and league state

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following parts of `docs/superpowers/specs/2026-10-16-sp4-selection-design.md` (Russian; read the spec first, it is the binding authority):
- **Section 6, "Совместимость (первая задача)", and §7 P0:** the SP3 run-dir fixture made by unchanged code, copies of the SP3 example configs, the compatibility baseline and the result-flow contract tests on current behaviour (T0.1).
- **Block 1 (match log):** the new fields of `MatchResult` / `SeatResult` / `Lineup`, per-seat weights versions, explicit lineup seeds, deterministic uids, counted result drops, the results queue of 10 000, the `match_log` config section (T1.1); rating entities, team-pair sufficient statistics and the SQLite journal library — schema, append, compaction into `counts`, disk guard, resume import, read-only reader (T1.2); the journal writer thread in training, the compaction cadence, the shutdown flush, the resume import, the `system` journal fields and `run_id` (T1.3).
- **Block 2 (ratings):** the Bradley–Terry library — fit, side bias by team index, role groups and components, anchors, Elo units, Laplace and bootstrap uncertainty, the G² transitivity test, `score` layouts (T2.1); live ratings — the refit thread, the `bt` section of `ratings.json`, the compact `ratings` record, console, WandB, the `ratings` config section, one BLAS thread in the training main process (T2.2).
- **Block 7 (metrics and league state):** `rho_trunc_mass` / `c_trunc_mass`, monotonic loss counters, the device line of `colosseum bc` (T3.1); `league_state.json` (T3.2). Item 7.3 (compact `ratings` record) is in T2.2.

Not in this part: snapshot retention (`keep_top_k`, `pinned`, `max_snapshots`: T4.1), the tournament (T5.x), pulse (T6.1), report and dashboard (T7.x), distributed guards and docs (T8.x). Until T4.1, `checkpoint.keep_top_k` does not exist, so the live ratings run with `top_k = 0`.

**What the part delivers.** After T3.2:
- every training match lands in `runs/<name>/matches.db` exactly once or is counted in `dropped_match_results`, with the weights versions of every seat at the episode's start and end;
- the journal stays within its detail window (old training matches folded into `counts`), switches to counts-only when the disk is low, and a run-dir resume continues it (the parent's detail folded into counts, its tournament and pulse matches copied);
- a Bradley–Terry fit of snapshots with intervals runs every `ratings.refit_interval_sec` in a background thread; `ratings.json` carries a `bt` section, `metrics.jsonl` a compact `ratings` record, the console a `bt <rating> ± <ci>` field;
- `rho_trunc_mass` / `c_trunc_mass` replace `rho_clip_frac` / `c_clip_frac`; `dropped_reward_episodes` and `dropped_match_results` are monotonic per run;
- ratings and PFSP statistics survive a run-dir resume through `league_state.json`.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`).
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files have basenames that start with `test_sp4_` (unique across `tests/`); shared test support goes into `tests/game_helpers.py`.
- Tests that drive a `RolloutLoop` through `game_harness` (`make_loop`, `GameFactory`, `Collected`) live in `tests/contract/` (`game_harness` is importable only from there).
- Every file a test writes is under `tmp_path`; fixtures under `tests/fixtures/` are read-only (copy before writing). At most 2 worker processes per test; processes through `tests/cli_runner.py`.
- Messages go through `logging` (never `warnings.warn`), so `-rw` stays empty. Durations and cadences use `time.monotonic()`; data is ordered by counters and row ids, never by wall-clock `ts`.
- Commit messages use conventional prefixes and carry no attribution lines; push after every task (`git push`, branch `sp4-selection`).

**Cross-task notes.**
- **T0.1 runs on unchanged code** (the branch has only docs commits before it). Its fixture is the compatibility baseline every later task keeps green. Later tasks extend its tests: T1.3 (the SP3 run-dir resume starts an empty journal with an INFO line), T3.2 (it starts an empty league state with an INFO line); in other parts T4.1 (SP3 snapshots without `run_id` are imported, not final), T5.4 (SP3 snapshots as tournament candidates and frozen players).
- **The new package `colosseum.arena`** is created by T1.1 (`__init__.py` and `entities.py`: `MatchRunner` needs `snapshot_version`), extended by T1.2 (`comparisons.py`, `matchlog.py`), T1.3 (`MatchLogWriter` in `matchlog.py`), T2.1 (`bt.py`) and T2.2 (`live.py`). None of its modules imports the launcher, the coordinator or `multiprocessing`; they import `colosseum.core.*` only.
- **Import rule for `colosseum.arena`:** `entities` imports `colosseum.core.types`; `comparisons` imports `entities` and `colosseum.core.outcomes`; `matchlog` imports `comparisons`; `bt` imports `comparisons` and numpy; `live` imports `bt`. `MatchRunner` imports `colosseum.arena.entities` only.
- **`run_id`** is created by T1.1 (uids need it): `run.json` in the run dir, written once by `Launcher.launch` (`RunDir.ensure_run_id()`), never by `RunDir.create` (an SP2 test pins the files `create` makes). T7.1 adds `run_state.json` next to it; `run.json` stays the immutable identity of the run.
- **Thread rule (overview, global constraints):** the journal writer and the live-rating thread own their SQLite connections and data. They talk to the main loop only through the writer's bounded queue, `note_evicted`, the shared `ComparisonTable` (internally locked) and immutable published objects (`WriterStats`, `RatingsSnapshot`). They never call `Coordinator`, `CheckpointManager`, `PfspStats` or `RatingBook` methods.
- **Provisional constants.** `colosseum.arena.bt.PRIOR_SIGMA = 4.0` (logits) and `SMOOTHNESS = 0.5` are placeholders; T8.2 fixes them by cross-validation (controller rulings). Tests of this part read the module constants, never literals, so T8.2 can change them.
- **Shared helpers this part adds to `tests/game_helpers.py`:** `SP3_FIXTURES`, `SP3_CONFIGS`, `SP3_CONFIG_NAMES`, `SP3_TTT_TINY`, `SP3_RUN`, `sp3_manifest()`, `copy_sp3_run(tmp_path)` (T0.1); `make_match_result(...)`, `synthetic_journal_results(...)` (T1.2); `synthetic_bt_matches(...)`, `bt_table(...)` (T2.1).

**Verified while writing this part.** The library code of T1.2, T1.3 (writer), T2.1 and T2.2 (refit) was run in a scratch package against stand-in record types with the new fields: tables built from the journal equal the in-memory tables exactly before compaction and to float rounding after it; fits from both agree; the size estimate of a detailed match (`estimated_size`) is within 7 % of SQLite's real page use for 1v1, 2v2 and 4-team FFA matches; Laplace intervals cover the true anchored differences 0.945–0.991 on average over 15 runs × 20 replications in the six synthetic scenarios of T2.1 (run-to-run sd 0.007–0.021); one live refit of 100 / 500 / 1000 / 2000 entities (20 comparisons each, 1000 Laplace samples) took 0.21 / 0.72 / 1.14 / 1.62 s on this machine. The SP3 fixture generator of T0.1 was run once on the unchanged branch: about 7 s, five snapshots (two `keep_every` snapshots without `trainer_state.pt`), about 300 KB.

---

### Task T0.1: SP3 run-dir fixture, compatibility baseline, result-flow tests on current behaviour

Spec section 6 ("Совместимость (первая задача)"), §7 P0 and criterion 3.8. Before any SP4 change, unchanged SP3 code produces (a) byte copies of every `configs/examples/*.yaml` (T8.4 edits `unit_harvest_league.yaml`; the copies keep the SP3 form) and (b) a small SP3 run dir: a tiny tic-tac-toe run with a trainable agent `main` and a `RandomBot` anchor `random`, with several snapshots (some `keep_every` snapshots without `trainer_state.pt`), `metrics.jsonl`, `ratings.json`, and no `matches.db` / `league_state.json`. A generator script is committed next to it (as `tests/fixtures/sp2/make_sp2_checkpoint.py`); its `manifest.json` records the snapshot versions, because training is asynchronous and versions differ between generations. The baseline tests pass on SP3 code now and stay green through SP4. The result-flow tests pin what SP4 must keep while it reworks the flow: every finished episode is reported exactly once with a unique match id, the launcher feeds every drained result to the coordinator and the hub once, and a full results queue never blocks a worker.

**Files:**
- Create: `tests/fixtures/sp3/configs/{chase,coin_grid,coop_buttons,predator_prey,space_miners,team_tag,tic_tac_toe,tic_tac_toe_attention,tic_tac_toe_multi,tron,unit_harvest,unit_harvest_league}.yaml` (byte copies of `configs/examples/*.yaml`)
- Create: `tests/fixtures/sp3/sp3_ttt_tiny.yaml` (the SP3-form config of the run fixture)
- Create: `tests/fixtures/sp3/make_sp3_run.py` (generator, run once now; committed with its output)
- Create (generated): `tests/fixtures/sp3/run/{config.resolved.yaml,metrics.jsonl,ratings.json,checkpoints/main/ckpt_v*/...}`, `tests/fixtures/sp3/manifest.json`
- Modify: `.gitignore` (the repo ignores `checkpoints/` and `*.pt`; the fixture must be committed)
- Modify: `tests/game_helpers.py` (fixture paths, `sp3_manifest`, `copy_sp3_run`)
- Test: `tests/unit/test_sp4_sp3_compat.py`, `tests/integration/test_sp4_sp3_resume.py`, `tests/contract/test_sp4_result_flow.py`

**Interfaces:**
- Consumes (SP3, unchanged): `colosseum train` / `validate` / `eval` (CLI), `load_checkpoint_dir`, `check_model_state`, `colosseum.core.registry.{build_network, env_spec}`, `colosseum.core.config.{NetworkConfig, load_config}`, `MatchResult` / `SeatResult` / `TeamResult` / `Lineup` / `SeatAssignment`, `RolloutLoop` via `game_harness.make_loop`, `rollout_worker_process`, `Launcher._drain_results`.
- Produces (test support, `tests/game_helpers.py`):
  - `SP3_FIXTURES: Path`, `SP3_CONFIGS: Path`, `SP3_CONFIG_NAMES: tuple[str, ...]`, `SP3_TTT_TINY: Path`, `SP3_RUN: Path`;
  - `sp3_manifest() -> dict` (`agent`, `anchor`, `layout`, `versions`, `final_version`, `with_trainer_state`, `env_steps`);
  - `copy_sp3_run(tmp_path) -> Path` (a writable copy of `SP3_RUN`).

- [ ] **Step 1: Copy the SP3 example configs**

Run (on the unchanged tree; `git status` must show only plan files):
```bash
mkdir -p tests/fixtures/sp3/configs
cp configs/examples/*.yaml tests/fixtures/sp3/configs/
ls tests/fixtures/sp3/configs | wc -l   # 12
```

- [ ] **Step 2: Write the fixture config and the generator**

Create `tests/fixtures/sp3/sp3_ttt_tiny.yaml`:
```yaml
# SP3-form config of the SP3 run-dir fixture (tests/fixtures/sp3/run), SP4 task T0.1: tic-tac-toe with a tiny
# MLP, a trainable agent 'main' and a RandomBot anchor 'random'. keep_last 3 + keep_every 2 at interval 20 leave
# snapshots with and without trainer_state.pt.
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
  kwargs: {hidden: 16, latent: 8}
algorithm:
  learning_rate: 1.0e-3
  lr_schedule: "constant"
rollout:
  chunk_length: 16
  num_workers: 1
  envs_per_worker: 4
  weight_sync_interval_sec: 0.5
  match_refresh_interval_sec: 1.0
learner:
  device: "cpu"
  queue_size: 16
  batch_chunks: 2
training:
  total_timesteps: 6000
  seed: 0
matchmaking:
  opponents: {latest: 0.5, snapshots: 0.3, rivals: 0.0, anchors: 0.2}
  anchors: [random]
checkpoint:
  interval: 20
  keep_last: 3
  keep_every: 2
  save_optimizer: true
metrics:
  use_wandb: false
  log_interval: 5
  console_interval_sec: 1.0
agents:
  main: {}
  random:
    kind: scripted
    class: "colosseum.players.RandomBot"
```

Create `tests/fixtures/sp3/make_sp3_run.py`:
```python
"""Generator of the SP3 run-dir fixture (SP4 task T0.1). Run ONCE on unchanged SP3 code; the output is committed.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp3/make_sp3_run.py

Trains ``sp3_ttt_tiny.yaml`` with ``colosseum train`` in a temporary directory and copies what an SP3 run dir
holds into ``tests/fixtures/sp3/run/``: ``config.resolved.yaml`` (``run.dir`` rewritten to ``runs``),
``metrics.jsonl``, ``ratings.json`` and ``checkpoints/`` (``logs/`` is left out: it holds machine paths). It then
writes ``manifest.json`` with what tests need to know about the run: training is asynchronous, so the snapshot
versions differ between generations. It is a record of how the fixture was made; SP4 tasks never re-run it (the
code after T1.1 makes SP4 run dirs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG = HERE / "sp3_ttt_tiny.yaml"
AGENT = "main"
ANCHOR = "random"


def main() -> None:
    dst = HERE / "run"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(CONFIG),
               "--set", f"run.dir={tmp}", "--set", "run.name=sp3_run"]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        src = Path(tmp) / "sp3_run"
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        for name in ("metrics.jsonl", "ratings.json"):
            shutil.copy2(src / name, dst / name)
        resolved = yaml.safe_load((src / "config.resolved.yaml").read_text())
        resolved["run"]["dir"] = "runs"
        (dst / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        shutil.copytree(src / "checkpoints", dst / "checkpoints")
    assert not (dst / "matches.db").exists() and not (dst / "league_state.json").exists()
    agent_dir = dst / "checkpoints" / AGENT
    metas = {p.name: json.loads((p / "meta.json").read_text()) for p in sorted(agent_dir.iterdir())}
    finals = [m["policy_version"] for m in metas.values() if m["final"]]
    assert len(finals) == 1, finals
    manifest = {
        "agent": AGENT,
        "anchor": ANCHOR,
        "layout": "2p",
        "versions": sorted(m["policy_version"] for m in metas.values()),
        "final_version": finals[0],
        "with_trainer_state": sorted(m["policy_version"] for name, m in metas.items()
                                     if (agent_dir / name / "trainer_state.pt").exists()),
        "env_steps": max(int(m["env_steps"]) for m in metas.values()),
    }
    assert len(manifest["versions"]) >= 3 and len(manifest["with_trainer_state"]) < len(manifest["versions"])
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {dst}: {manifest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the run dir and let git see it**

Append to `.gitignore` (after the SP2 fixture lines):
```
# SP3 run-dir fixture (SP4 T0.1): committed SP3 snapshots inside an SP3-shaped run dir
!/tests/fixtures/sp3/run/checkpoints/
!/tests/fixtures/sp3/run/checkpoints/**
```
Run:
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp3/make_sp3_run.py
git status --porcelain -uall tests/fixtures/sp3 .gitignore
du -sh tests/fixtures/sp3
```
Expected: about 7 s; `wrote .../tests/fixtures/sp3/run: {...}` with 4–6 versions (e.g. `[40, 80, 100, 120, 137]`, two of them without trainer state) and `env_steps` a little above 6000; `git status` lists the 12 config copies, `sp3_ttt_tiny.yaml`, the generator, `manifest.json`, `run/config.resolved.yaml`, `run/metrics.jsonl`, `run/ratings.json` and every file under `run/checkpoints/main/` (about 300 KB in all). If the assertion in the generator fails (fewer than three snapshots), raise `training.total_timesteps` in `sp3_ttt_tiny.yaml` by 2000 and run it again.

- [ ] **Step 4: Add the fixture paths to `tests/game_helpers.py`**

Add `import json` to the module imports, then append after the SP2 fixture block (`copy_sp2_run`):
```python
# ---------------------------------------------------------------------------
# SP4 (T0.1): SP3 compatibility fixtures (read-only inputs; copy before writing)
# ---------------------------------------------------------------------------


SP3_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sp3"
SP3_CONFIGS = SP3_FIXTURES / "configs"
SP3_CONFIG_NAMES = ("chase", "coin_grid", "coop_buttons", "predator_prey", "space_miners", "team_tag",
                    "tic_tac_toe", "tic_tac_toe_attention", "tic_tac_toe_multi", "tron", "unit_harvest",
                    "unit_harvest_league")
SP3_TTT_TINY = SP3_FIXTURES / "sp3_ttt_tiny.yaml"
SP3_RUN = SP3_FIXTURES / "run"


def sp3_manifest() -> dict:
    """What the SP3 run fixture holds (``tests/fixtures/sp3/manifest.json``, written by its generator):
    ``agent``, ``anchor``, ``layout``, ``versions``, ``final_version``, ``with_trainer_state``, ``env_steps``."""
    return json.loads((SP3_FIXTURES / "manifest.json").read_text())


def copy_sp3_run(tmp_path) -> Path:
    """A writable copy of the SP3 run fixture under ``tmp_path`` (tests never write into tests/fixtures)."""
    dst = Path(tmp_path) / "sp3_run"
    shutil.copytree(SP3_RUN, dst)
    return dst
```

- [ ] **Step 5: Write the baseline tests**

Create `tests/unit/test_sp4_sp3_compat.py`:
```python
"""SP3 compatibility baseline (SP4 T0.1, spec section 6 and criterion 3.8): the SP3 example configs (copies)
validate and load silently, the SP3 run fixture is an SP3 run dir whose snapshots load read-only and play in eval
(by path and as a frozen agent), and match records are constructible without the SP4 fields.

Created on SP3 code; it must stay green through SP4. Later tasks extend the SP3 baseline elsewhere: T1.3 and T3.2
(`test_sp4_sp3_resume.py`), T4.1 (imported SP3 snapshots are not final), T5.4 (SP3 snapshots in a tournament).
"""
from __future__ import annotations

import dataclasses
import json
import logging

import pytest
import yaml
from click.testing import CliRunner

from cli_runner import REPO_ROOT
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.registry import build_network, env_spec
from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from game_helpers import SP3_CONFIG_NAMES, SP3_CONFIGS, SP3_RUN, SP3_TTT_TINY, sp3_manifest

SP3_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks",
                 "policy_version", "role_signature", "roles", "timestamp"}


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_the_fixture_has_a_copy_of_every_sp3_example_config():
    assert sorted(p.stem for p in SP3_CONFIGS.glob("*.yaml")) == sorted(SP3_CONFIG_NAMES)


@pytest.mark.parametrize("name", SP3_CONFIG_NAMES)
def test_sp3_config_copies_validate_without_edits(name, monkeypatch):
    if name == "space_miners":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(SP3_CONFIGS / f"{name}.yaml")])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("Config is valid."), result.output


@pytest.mark.parametrize("name", SP3_CONFIG_NAMES)
def test_sp3_config_copies_load_without_warnings(name, caplog):
    with caplog.at_level(logging.WARNING):
        load_config(SP3_CONFIGS / f"{name}.yaml")
    assert [r.getMessage() for r in caplog.records if r.name.startswith("colosseum")] == []


def test_the_run_fixture_is_an_sp3_run_dir():
    manifest = sp3_manifest()
    assert sorted(p.name for p in SP3_RUN.iterdir()) == ["checkpoints", "config.resolved.yaml", "metrics.jsonl",
                                                       "ratings.json"]
    agent_dir = SP3_RUN / "checkpoints" / manifest["agent"]
    versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.iterdir())
    assert versions == manifest["versions"] and len(versions) >= 3
    assert manifest["final_version"] == max(versions)
    without_trainer = set(versions) - set(manifest["with_trainer_state"])
    assert without_trainer, "the fixture needs keep_every snapshots without trainer_state.pt"
    for version in versions:
        meta = json.loads((agent_dir / f"ckpt_v{version}" / "meta.json").read_text())
        assert set(meta) == SP3_META_KEYS                     # no run_id / pinned (SP4 adds them in T4.1)
        assert meta["final"] is (version == manifest["final_version"])
    ratings = json.loads((SP3_RUN / "ratings.json").read_text())
    assert set(ratings) == {"env_steps", "layouts"} and manifest["layout"] in ratings["layouts"]
    assert manifest["anchor"] in ratings["layouts"][manifest["layout"]]["elo"]
    kinds = {json.loads(line)["kind"] for line in (SP3_RUN / "metrics.jsonl").read_text().splitlines() if line}
    assert kinds == {"train", "episodes", "ratings", "system"}
    resolved = load_config(SP3_RUN / "config.resolved.yaml")
    assert resolved.get_trainable_agent_ids() == [manifest["agent"]]
    assert resolved.fixed_agent_ids() == [manifest["anchor"]]


def test_every_sp3_snapshot_loads_read_only_with_its_architecture():
    manifest = sp3_manifest()
    before = _listing(SP3_RUN)
    spec = env_spec(load_config(SP3_TTT_TINY))
    for version in manifest["versions"]:
        ckpt = load_checkpoint_dir(SP3_RUN / "checkpoints" / manifest["agent"] / f"ckpt_v{version}")
        assert ckpt["meta"]["policy_version"] == version and ckpt["roles"] == ["player"]
        model = build_network(NetworkConfig.model_validate(ckpt["meta"]["networks"]), spec.roles["player"])
        check_model_state(model, ckpt["model_state"], f"ckpt_v{version}")
    assert _listing(SP3_RUN) == before


def test_eval_plays_sp3_snapshots_by_path_and_as_a_frozen_agent(tmp_path):
    manifest = sp3_manifest()
    before = _listing(SP3_RUN)
    agent_dir = SP3_RUN / "checkpoints" / manifest["agent"]
    first, last = agent_dir / f"ckpt_v{manifest['versions'][0]}", agent_dir / f"ckpt_v{manifest['final_version']}"
    out = tmp_path / "paths.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(SP3_TTT_TINY), "-a", f"old={first}", "-a", f"new={last}",
                                       "-n", "2", "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["agents"] == ["old", "new"]
    data = yaml.safe_load(SP3_TTT_TINY.read_text())
    data["agents"]["old"] = {"kind": "frozen", "path": str(first)}
    cfg = tmp_path / "with_frozen.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    out = tmp_path / "frozen.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "old", "-a", "random", "-n", "2",
                                       "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["layouts"]["2p"]["n"] == 2
    assert _listing(SP3_RUN) == before


def test_match_records_are_constructible_without_the_sp4_fields():
    seats = [SeatResult(0, "player", 0, "a", "latest", 1.0), SeatResult(1, "player", 1, "b", "fixed", -1.0, None, "")]
    result = MatchResult("m0", "2p", "wdl", seats, [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 9)
    assert (result.match_id, result.layout, result.episode_length) == ("m0", "2p", 9)
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "fixed", False, "anchors")])
    assert lineup.seats[1].source == "anchors"
    # SP3 positional order is kept: new fields only ever come after the SP3 ones.
    assert [f.name for f in dataclasses.fields(MatchResult)][:6] == [
        "match_id", "layout", "outcome_kind", "seats", "teams", "episode_length"]
    assert [f.name for f in dataclasses.fields(SeatResult)][:8] == [
        "seat", "role", "team", "agent_id", "network_id", "reward", "eliminated_step", "source"]
    assert [f.name for f in dataclasses.fields(Lineup)][:2] == ["layout", "seats"]
```

Create `tests/integration/test_sp4_sp3_resume.py`:
```python
"""SP3 compatibility baseline (SP4 T0.1): `colosseum train` resumes from the SP3 run fixture (a copy) and carries
its snapshot pool over; the source is only read. T1.3 adds the empty journal, T3.2 the empty league state."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP3_TTT_TINY, copy_sp3_run, sp3_manifest


def _versions(agent_dir) -> list[int]:
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_training_resumes_from_the_sp3_run_dir(tmp_path):
    manifest = sp3_manifest()
    source = copy_sp3_run(tmp_path)
    before = _listing(source)
    budget = manifest["env_steps"] + 2000          # a resume needs a budget above the checkpoint's env_steps
    run = run_train(SP3_TTT_TINY, tmp_path, name="resumed",
                    overrides={"training.resume_from": str(source), "training.total_timesteps": str(budget)})
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert f"Resume [{manifest['agent']}]:" in log and f"(policy_version {manifest['final_version']})" in log
    assert f"env-step counter continues from {manifest['env_steps']}" in log
    assert "snapshot pool carried over" in log
    versions = _versions(run.root / "checkpoints" / manifest["agent"])
    assert min(versions) <= manifest["final_version"] < max(versions)   # imported snapshots + new ones
    assert _listing(source) == before                                     # the source is untouched
```

Create `tests/contract/test_sp4_result_flow.py`:
```python
"""Result flow on current (SP3) behaviour (SP4 T0.1, spec section 6 and §7 P0), the baseline the SP4 journal must
keep: every finished episode is reported exactly once with a unique match id, the launcher feeds each drained
result to the coordinator and the hub once, and a full results queue never blocks a worker. T1.1 extends this
file with the SP4 record fields, versions, uids and the counted drops."""
from __future__ import annotations

import queue
import threading

import pytest
import torch

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult, WorkerCommand
from colosseum.launcher import Launcher
from colosseum.worker.rollout_worker import rollout_worker_process
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import NumpyOnlyQueue, Tick, TickGame, make_test_config, make_test_model, make_test_run_dir

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]
SOLO = [Tick(acting={0}), Tick(acting={0}, rewards={0: 1.0}), Tick(over=True, rewards={0: 1.0})]
SOLO_ROLE = TickGame(SOLO, 1).spec.roles["player"]


def _solo_env() -> TickGame:
    return TickGame(SOLO, 1)


def _solo_model():
    return make_test_model(SOLO_ROLE)


def _episode(length: int) -> list[Tick]:
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_every_finished_episode_is_reported_once_with_a_unique_match_id():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2), (_episode(3), 2), (_episode(5), 2)),
                          {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "b"), lineup("2p", "b", "a"), lineup("2p", "a", "a")])
    run_steps(loop, 60)
    assert len(col.results) == loop.stats["episodes"] > 0
    ids = [r.match_id for r in col.results]
    assert len(set(ids)) == len(ids)
    for env in range(3):
        episodes = [int(m.rsplit("_ep", 1)[1]) for m in ids if m.startswith(f"w0_e{env}_")]
        assert episodes == list(range(len(episodes)))          # consecutive per env, none skipped or repeated


def test_lineup_changes_keep_one_result_per_episode_with_the_lineup_that_played_it():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", "a")])
    for i in range(40):
        if i % 5 == 0:
            col.commands.append(WorkerCommand(lineups=[lineup("2p", *("ab" if i % 10 == 0 else "ba"))]))
        loop.step()
    assert len(col.results) == loop.stats["episodes"]
    assert {tuple(s.agent_id for s in r.seats) for r in col.results} <= {("a", "a"), ("a", "b"), ("b", "a")}


def _result(i: int) -> MatchResult:
    return MatchResult(f"w0_e0_ep{i}", "2p", "wdl",
                       [SeatResult(0, "player", 0, "agent_0", "latest", 1.0),
                        SeatResult(1, "player", 1, "agent_0", "latest", -1.0)],
                       [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 4)


class _Recorder:
    def __init__(self) -> None:
        self.results: list[MatchResult] = []

    def report_match_result(self, result: MatchResult) -> None:
        self.results.append(result)

    def on_match_result(self, result: MatchResult) -> None:
        self.results.append(result)


def test_the_launcher_feeds_each_drained_result_to_the_coordinator_and_the_hub_once(tmp_path):
    config = make_test_config("turns")
    launcher = Launcher(config, make_test_run_dir(config, tmp_path))
    results_queue: queue.Queue = queue.Queue()
    sent = [_result(i) for i in range(5)]
    for item in sent:
        results_queue.put(item)
    coordinator, hub = _Recorder(), _Recorder()
    launcher._results_queue, launcher._coordinator, launcher._hub = results_queue, coordinator, hub
    launcher._drain_results()
    launcher._drain_results()                                   # nothing new: nothing fed twice
    assert [r.match_id for r in coordinator.results] == [r.match_id for r in sent]
    assert [r.match_id for r in hub.results] == [r.match_id for r in sent]


def test_a_full_results_queue_never_blocks_the_worker(restore_torch_threads):
    results = NumpyOnlyQueue(maxsize=1)
    rollout_worker_process(
        worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"], agent_roles={"a": ["player"]},
        model_factories={"a": _solo_model}, trajectory_queues={"a": NumpyOnlyQueue()},
        weight_queues={"a": NumpyOnlyQueue()}, stop_event=threading.Event(), max_env_steps=40,
        lineups=[Lineup("solo", [SeatAssignment("a")])] * 2, results_queue=results, seed=0,
    )
    assert results.qsize() == 1                                  # the first result; the others were dropped
```

- [ ] **Step 6: Run the baseline tests (they pass on SP3 code)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_sp3_compat.py tests/integration/test_sp4_sp3_resume.py tests/contract/test_sp4_result_flow.py -q -rw`
Expected: all pass (`space_miners` validate is skipped without Box2D); this is a baseline, not a failing-first test. If anything fails, the fixture (not the code) is wrong: fix the fixture or the test, never the code.

- [ ] **Step 7: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 8: Commit and push**

```bash
git add .gitignore tests/fixtures/sp3 tests/game_helpers.py tests/unit/test_sp4_sp3_compat.py \
        tests/integration/test_sp4_sp3_resume.py tests/contract/test_sp4_result_flow.py
git commit -m "test: SP3 compatibility fixtures (example config copies, SP3 run dir) and result-flow baseline"
git push
```

---

### Task T1.1: Match record fields, per-seat versions, explicit lineup seeds, counted result drops, `match_log` config

Spec block 1 ("Запись матча", "Запись", "Конфиг"). `MatchResult` gains `uid`, `source`, `seed`, `env_steps`, `block`; `SeatResult` gains `version_start` / `version_end`; `Lineup` gains an explicit `seed`, `block` and `uid` (tournament, pulse). `MatchRunner` reads every seat's weights version when the episode starts (right after the env reset) and when it ends (`ckpt_v<N>` → N, `fixed` → None, `latest` → a version function; `RolloutLoop` passes its `policy_versions`, which change mid-episode when weights are synced), uses a lineup's explicit seed instead of the derived one, and makes the uid (`<run id>/train/w<W>_e<E>_ep<K>` in training). The results queue grows to 10 000; a full queue still never blocks a worker, but every dropped result is counted (`dropped_results`, cumulative per worker; the first drop is a WARNING), and a worker sends a final stats report and flushes its results and stats queues on exit, so that the main process can account for every result. The main loop stamps each received result with the global env-step counter. The run id (`run.json`) is created here because the uids need it. The `match_log` config section is added (used from T1.3). The new package `colosseum.arena` starts with `entities.py` (rating versions and entity keys; `MatchRunner` needs `snapshot_version`).

**Files:**
- Create: `src/colosseum/arena/__init__.py`, `src/colosseum/arena/entities.py`
- Modify: `src/colosseum/core/types.py` (`MATCH_SOURCES`, `Lineup`, `SeatResult`, `MatchResult`)
- Modify: `src/colosseum/worker/match_runner.py` (`version_fn`, `source`, `uid_prefix`, explicit seeds, versions, uids)
- Modify: `src/colosseum/worker/rollout_loop.py` (`run_id`, `version_fn`, `dropped_results`)
- Modify: `src/colosseum/worker/rollout_worker.py` (`run_id`, counted drops, final stats, queue flush)
- Modify: `src/colosseum/launcher.py` (`_RESULTS_QUEUE_SIZE`, `run_id`, `env_steps` stamp, lineup fields kept)
- Modify: `src/colosseum/core/run_dir.py` (`RUN_INFO_FILE`, `read_run_id`, `RunDir.run_id`, `ensure_run_id`)
- Modify: `src/colosseum/core/config.py` (`MatchLogConfig`, `ColosseumConfig.match_log`)
- Modify: `src/colosseum/eval.py` (`play_lineups` keeps a lineup's explicit fields)
- Modify: `scripts/bench_throughput.py` (`_make_config`: `match_log`)
- Modify (tests pinned to the old behaviour): `tests/contract/test_rollout_loop_lineups.py` (stats keys), `tests/unit/test_sp2_launcher_lifecycle.py` (a fake result string)
- Modify (extend the T0.1 baseline): `tests/contract/test_sp4_result_flow.py`
- Test: `tests/unit/test_sp4_match_records.py`

**Interfaces:**
- Consumes: `MatchRunner`, `RolloutLoop`, `rollout_worker_process`, `Launcher._drain_results` (SP3); `colosseum.utils.process.flush_queue` (`utils/process.py:126`: closes a queue and waits for its feeder thread while the parent lives).
- Produces (contract T1.1, plus additions marked *):
  - `MATCH_SOURCES`; `Lineup.seed/block/uid`; `SeatResult.version_start/version_end`; `MatchResult.uid/source/seed/env_steps/block`;
  - `MatchRunner(..., version_fn=None, source="train", uid_prefix="")`;
  - *`RolloutLoop(..., run_id=None)`, *`rollout_worker_process(..., run_id=None)`, *`_worker_main(..., run_id=None)`;
  - *`LoopIO.report_result` may return `False` (dropped; counted in *`RolloutLoop.stats["dropped_results"]`);
  - *`report_worker_stats(q, worker_id, stats, timeout=None)` (`timeout`: the final report waits for room);
  - `MatchLogConfig` (`enabled`, `detail_window_mb`, `min_free_disk_mb`, `compact_interval_sec`), `ColosseumConfig.match_log`;
  - `colosseum.arena.entities`: `EntityKey`, `snapshot_version`, `rating_version`, `entity_key`, `lineage`, *`entity_version`;
  - *`colosseum.core.run_dir.RUN_INFO_FILE = "run.json"`, *`read_run_id(root) -> str | None`, `RunDir.run_id` (read only), *`RunDir.run_info_path`, *`RunDir.ensure_run_id() -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_match_records.py`:
```python
"""Match records of SP4 (T1.1, spec block 1): the new optional fields, rating entities, per-seat weights versions
read by MatchRunner at the episode's start and end, explicit lineup seeds / blocks / uids, the uid prefix, the
run id file and the match_log config section."""
from __future__ import annotations

import json

import numpy as np
import pytest

from colosseum.arena.entities import entity_key, entity_version, lineage, rating_version, snapshot_version
from colosseum.core.config import ColosseumConfig, MatchLogConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RUN_INFO_FILE, RunDir, read_run_id
from colosseum.core.types import MATCH_SOURCES, Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.envs.vector import VectorEnv
from colosseum.eval import play_lineups
from colosseum.worker.match_runner import MatchRunner
from game_helpers import DictModelPool, RecordingObserver, Tick, TickGame, make_test_config, make_test_model

TURNS = [Tick(acting={0}), Tick(acting={1}), Tick(acting={0}), Tick(over=True, rewards={0: 1.0, 1: -1.0})]
ROLE2 = TickGame(TURNS, 2).spec.roles["player"]


def test_new_record_fields_have_defaults():
    seat = SeatResult(0, "player", 0, "a", "latest", 1.0)
    assert (seat.version_start, seat.version_end) == (None, None)
    result = MatchResult("m", "2p", "wdl", [seat], [TeamResult(0, 1.0, 1.0)], 3)
    assert (result.uid, result.source, result.seed, result.env_steps, result.block) == ("", "train", None, None, None)
    lineup = Lineup("2p", [SeatAssignment("a")])
    assert (lineup.seed, lineup.block, lineup.uid) == (None, None, None)
    assert MATCH_SOURCES == ("train", "tournament", "pulse")


def test_entities_versions_buckets_and_lineages():
    assert snapshot_version("ckpt_v1200") == 1200
    assert snapshot_version("latest") is None and snapshot_version("fixed") is None
    assert rating_version("ckpt_v1250", 99, 100) == 1250                 # a snapshot counts for its own version
    assert rating_version("latest", 1250, 100) == 1200                   # latest counts for its bucket
    assert rating_version("latest", 99, 100) == 0
    assert rating_version("latest", None, 100) is None
    assert rating_version("fixed", 7, 100) is None
    with pytest.raises(ValueError, match="network id"):
        rating_version("ckpt_x", 1, 100)
    assert entity_key("main", 1200) == "main@v1200" and entity_key("random", None) == "random"
    assert lineage("main@v1200") == "main" and lineage("b/main@v3") == "b/main" and lineage("greedy") == "greedy"
    assert entity_version("main@v1200") == 1200 and entity_version("prev_sub#ab12cd34") is None


def _runner(lineups, *, versions=None, seed=0, num_envs=1, **kwargs):
    games: list[TickGame] = []

    def env_fn():
        games.append(TickGame(TURNS, 2))
        return games[-1]

    model = make_test_model(ROLE2)
    pool = DictModelPool({("a", "latest"): model, ("b", "latest"): model, ("b", "ckpt_v7"): model,
                          ("bot", "fixed"): model})
    observer = RecordingObserver()
    runner = MatchRunner(vec_env=VectorEnv(env_fn, num_envs), lineups=lineups, models=pool, observer=observer,
                         seed=seed, version_fn=None if versions is None else versions.get, **kwargs)
    return runner, observer, games


def _results(observer):
    return [e[2].result for e in observer.events if e[0] == "end"]


def test_seat_versions_at_the_episode_start_and_end_follow_the_version_function():
    versions = {"a": 3}
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "ckpt_v7", collect=False)])
    runner, observer, _ = _runner([lineup], versions=versions)
    runner.step()
    versions["a"] = 5                             # new weights arrive in the middle of the episode
    runner.step()
    runner.step()                                 # episode 0 ends (3 steps)
    for _ in range(3):
        runner.step()                             # episode 1 starts and ends with version 5
    first, second = _results(observer)
    assert [(s.version_start, s.version_end) for s in first.seats] == [(3, 5), (7, 7)]
    assert [(s.version_start, s.version_end) for s in second.seats] == [(5, 5), (7, 7)]


def test_fixed_seats_and_runners_without_a_version_source_have_no_versions():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", "fixed", collect=False)])
    runner, observer, _ = _runner([lineup])
    for _ in range(3):
        runner.step()
    (result,) = _results(observer)
    assert [(s.version_start, s.version_end) for s in result.seats] == [(None, None), (None, None)]


def test_derived_seeds_uids_and_sources_of_runner_made_results():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])
    runner, observer, games = _runner([lineup], seed=11, uid_prefix="RUN/train/", match_id_prefix="w2_e",
                                      source="pulse")
    for _ in range(3):
        runner.step()
    (result,) = _results(observer)
    expected = int(np.random.SeedSequence([11, 0, 0]).generate_state(1)[0])
    assert games[0].log[0][2] == expected and result.seed == expected
    assert result.match_id == "w2_e0_ep0" and result.uid == "RUN/train/w2_e0_ep0"
    assert result.source == "pulse" and result.block is None and result.env_steps is None
    runner_without_seed, observer2, _ = _runner([lineup], seed=None)
    for _ in range(3):
        runner_without_seed.step()
    assert _results(observer2)[0].seed is None and _results(observer2)[0].uid == "m0_ep0"


def test_explicit_lineup_seeds_blocks_and_uids_reach_the_env_and_the_result():
    first = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")], seed=1234, block="A/0/7", uid="T1/A/0/7/r0")
    second = Lineup("2p", [SeatAssignment("b"), SeatAssignment("a")], seed=1234, block="A/0/7", uid="T1/A/0/7/r1")
    runner, observer, games = _runner([first], seed=5)
    runner.set_next_lineup(0, second)
    for _ in range(6):
        runner.step()
    one, two = _results(observer)[:2]
    assert [log[2] for log in games[0].log if log[3] is None][:2] == [1234, 1234]   # both resets
    assert (one.seed, one.block, one.uid) == (1234, "A/0/7", "T1/A/0/7/r0")
    assert (two.seed, two.block, two.uid) == (1234, "A/0/7", "T1/A/0/7/r1")
    assert [s.agent_id for s in two.seats] == ["b", "a"]


def test_a_missing_snapshot_keeps_the_lineup_fields():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v99", collect=False)], seed=9,
                    block="B", uid="U")
    runner, observer, _ = _runner([lineup])
    assert (runner.lineup(0).seed, runner.lineup(0).block, runner.lineup(0).uid) == (9, "B", "U")
    assert runner.lineup(0).seats[1].network_id == "latest"          # SP1 fallback


def test_an_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])], source="eval")


def test_play_lineups_keeps_explicit_seeds():
    model = make_test_model(ROLE2)
    games: list[TickGame] = []

    def env_fn():
        games.append(TickGame(TURNS, 2))
        return games[-1]

    results = play_lineups(env_fn=env_fn, models={"a": model, "b": model}, num_envs=1, seed=0,
                           lineups=[Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")], seed=77, uid="x")])
    assert results[0].seed == 77 and results[0].uid == "x" and games[0].log[0][2] == 77


def test_run_id_is_written_once_and_read_without_writing(tmp_path):
    config = make_test_config("turns")
    config.run.dir, config.run.name = str(tmp_path / "runs"), "r"
    run = RunDir.create(config)
    assert run.run_id is None and not (run.root / RUN_INFO_FILE).exists()
    run_id = run.ensure_run_id()
    assert len(run_id) == 32 and run.ensure_run_id() == run_id == read_run_id(run.root)
    assert json.loads((run.root / RUN_INFO_FILE).read_text()) == {"run_id": run_id}
    assert read_run_id(tmp_path / "nowhere") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / RUN_INFO_FILE).write_text("{")
    with pytest.raises(ConfigError, match="run.json"):
        read_run_id(tmp_path / "bad")


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def test_match_log_config_defaults_validation_and_overrides(tmp_path):
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.match_log == MatchLogConfig(enabled=True, detail_window_mb=512.0, min_free_disk_mb=2048.0,
                                           compact_interval_sec=60.0)
    for bad in ({"detail_window_mb": 0}, {"min_free_disk_mb": -1}, {"compact_interval_sec": 0}, {"window": 1}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(match_log=bad))
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(_data()))
    assert load_config(path, {"match_log.detail_window_mb": 64}).match_log.detail_window_mb == 64.0
    with pytest.raises(ConfigError):                       # global only: no per-agent override
        load_config(path, {"agents.a.match_log.enabled": False})
    assert ColosseumConfig.model_validate(_data(agents={"a": {}})).get_agent_config("a").match_log.enabled
```

Extend `tests/contract/test_sp4_result_flow.py` (T0.1):
1. Replace the module docstring by:
```python
"""Result flow (SP4 T0.1 on SP3 behaviour, extended by T1.1; spec section 6, §7 P0 and block 1): every finished
episode is reported exactly once with a unique match id and uid, with the weights versions of every seat at the
episode's start and end; the launcher stamps env_steps and feeds each drained result to the coordinator and the
hub once; a full results queue never blocks a worker, and every drop is counted (a WARNING on the first)."""
```
2. Imports: add `import logging` (before `import queue`); replace the `colosseum.core.types` import by
```python
from colosseum.core.types import (
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
    WeightPayload,
    WorkerCommand,
    state_dict_to_numpy,
)
```
   and `from game_harness import GameFactory, lineup, make_loop, run_steps` by `from game_harness import Collected, GameFactory, lineup, make_loop, run_steps`.
3. In `test_the_launcher_feeds_each_drained_result_to_the_coordinator_and_the_hub_once`, set `sent[4].env_steps = 77` (a pulse result arrives with its snapshot's env_steps) before the results are queued, add `launcher._env_step_counter.add(1234)` right before the first `launcher._drain_results()`, and append
```python
    assert [r.env_steps for r in coordinator.results] == [1234] * 4 + [77]     # stamped at arrival when unset
```
4. Replace `test_a_full_results_queue_never_blocks_the_worker` by the following three tests:
```python
def test_a_full_results_queue_never_blocks_the_worker_and_every_drop_is_counted(restore_torch_threads, caplog):
    results, stats = NumpyOnlyQueue(maxsize=1), NumpyOnlyQueue()
    with caplog.at_level(logging.WARNING, logger="colosseum.worker.rollout_worker"):
        rollout_worker_process(
            worker_id=0, env_fn=_solo_env, num_envs=2, chunk_length=3, agent_ids=["a"],
            agent_roles={"a": ["player"]}, model_factories={"a": _solo_model},
            trajectory_queues={"a": NumpyOnlyQueue()}, weight_queues={"a": NumpyOnlyQueue()},
            stop_event=threading.Event(), max_env_steps=40, lineups=[Lineup("solo", [SeatAssignment("a")])] * 2,
            results_queue=results, stats_queue=stats, stats_interval_sec=3600.0, seed=0, run_id="RUN",
        )
    assert results.qsize() == 1                                  # the first result; the others were dropped
    reports = [stats.get_nowait() for _ in range(stats.qsize())]
    final = reports[-1]                                          # the final report on exit
    assert final["episodes"] > 1 and final["dropped_results"] == final["episodes"] - 1
    assert results.get_nowait().uid == "RUN/train/w0_e0_ep0"
    drops = [r for r in caplog.records if "results queue is full" in r.getMessage()]
    assert len(drops) == 1 and drops[0].levelno == logging.WARNING   # the first drop only


def test_latest_seat_versions_follow_weight_syncs_in_the_middle_of_an_episode():
    model = make_test_model(ROLE2)
    weights = state_dict_to_numpy(model.state_dict())
    collected = Collected(weights={"a": [WeightPayload("a", v, weights) for v in range(1, 30)]})
    # weight_sync_interval 0: one payload at construction (version 1), then one after every step; episodes take
    # two steps, so episode 0 starts at version 1 and ends at version 2 (the sync after its first step)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model, "b": lambda: model},
                          [lineup("2p", "a", SeatAssignment("b", collect=False))], collected=collected,
                          run_id="RUN")
    run_steps(loop, 4)
    first, second = col.results[:2]
    assert [(s.version_start, s.version_end) for s in first.seats] == [(1, 2), (0, 0)]   # b: never synced
    assert [(s.version_start, s.version_end) for s in second.seats] == [(2, 4), (0, 0)]
    assert (first.uid, second.uid) == ("RUN/train/w0_e0_ep0", "RUN/train/w0_e0_ep1")
    assert loop.stats["dropped_results"] == 0


def test_a_report_result_that_returns_false_is_counted_as_dropped():
    model = make_test_model(ROLE2)
    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": lambda: model}, [lineup("2p", "a", "a")])
    delivered: list[MatchResult] = []
    loop._io.report_result = lambda r: delivered.append(r) or len(delivered) % 2 == 1   # every second one fails
    run_steps(loop, 12)                                           # six episodes of two steps
    assert loop.stats["episodes"] == len(delivered) == 6 and loop.stats["dropped_results"] == 3
```

In `tests/contract/test_rollout_loop_lineups.py` (`test_results_are_reported_per_episode_and_stats_have_the_contract_keys`, the `set(loop.stats) == {...}` assertion) add `"dropped_results"` after `"dropped_reward_episodes"`.

In `tests/unit/test_sp2_launcher_lifecycle.py::test_launcher_closes_metrics_file_when_final_drain_fails` the results queue holds the string `"a result"`; `_drain_results` now stamps `env_steps` on every result, so replace `results.put("a result")` by `results.put(MatchResult("w0_e0_ep0", "2p", "wdl", [], [], 1))   # SP4: results are stamped with env_steps` and add `from colosseum.core.types import MatchResult` to the imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_match_records.py tests/contract/test_sp4_result_flow.py tests/contract/test_rollout_loop_lineups.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena'`; after the package exists, failures such as `TypeError: SeatResult.__init__() ... version_start`, `TypeError: MatchRunner.__init__() got an unexpected keyword argument 'version_fn'`, `ImportError: cannot import name 'MatchLogConfig'`.

- [ ] **Step 3: Create the `colosseum.arena` package and `entities.py`**

Create `src/colosseum/arena/__init__.py`:
```python
"""Selection and observability libraries of SP4 (spec block 8: no process queues, no launcher).

- ``entities``: rating entities (``agent@v<N>``), version buckets, lineages;
- ``comparisons``: team-pair sufficient statistics of match results (``ComparisonTable``);
- ``matchlog``: the SQLite match journal (``MatchLog``, ``MatchLogReader``, ``MatchLogWriter``);
- ``bt``: Bradley-Terry fit, Laplace and bootstrap uncertainty, role groups, anchors, transitivity test;
- ``live``: the live refit loop of the training main process and the ``bt`` section of ``ratings.json``.

Later SP4 tasks add ``retention``, ``tournament``, ``pulse`` and ``report``. Callers import submodules; this
package imports nothing eagerly.
"""
```

Create `src/colosseum/arena/entities.py`:
```python
"""Rating entities (SP4 spec block 2).

An entity is what a Bradley-Terry rating is fitted for:

- a trainable agent's seat counts for ``<agent>@v<B>``, where ``B`` is the seat's rating version: ``N`` for a
  snapshot ``ckpt_v<N>``, and for the latest weights the version bucket
  ``floor(version_start / checkpoint.interval) * checkpoint.interval`` (the snapshot saved at the start of
  that stretch of training; bucket ``v0`` has no snapshot: it is a rating entity, not a snapshot);
- a scripted or frozen agent (network ``fixed``) is one entity named by its agent id;
- a seat whose rating version is unknown (``latest`` without ``version_start``: eval, tournament players whose
  agent id already is the entity name) is the entity named by its agent id.

The lineage of an entity is its name without the ``@v<N>`` suffix (``main@v1200`` -> ``main``,
``b/main@v3`` -> ``b/main``); averages over a field are taken per lineage, not per snapshot.
"""

from __future__ import annotations

import re

from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID

EntityKey = str

_SNAPSHOT_RE = re.compile(r"ckpt_v(\d+)")
_VERSION_SUFFIX_RE = re.compile(r"@v(\d+)$")


def snapshot_version(network_id: str) -> int | None:
    """``"ckpt_v1200"`` -> 1200; any other network id (``latest``, ``fixed``) -> None."""
    match = _SNAPSHOT_RE.fullmatch(network_id)
    return None if match is None else int(match.group(1))


def rating_version(network_id: str, version_start: int | None, interval: int) -> int | None:
    """The version a seat's match counts for (module docstring): ``ckpt_v<N>`` -> N; ``latest`` -> the
    bucket of ``version_start`` (None without it); ``fixed`` -> None. ``interval`` >= 1."""
    if network_id == FIXED_NETWORK_ID:
        return None
    version = snapshot_version(network_id)
    if version is not None:
        return version
    if network_id != LATEST_NETWORK_ID:
        raise ValueError(f"unknown network id {network_id!r} (expected 'latest', 'fixed' or 'ckpt_v<N>')")
    if version_start is None:
        return None
    if interval < 1:
        raise ValueError(f"checkpoint interval must be >= 1, got {interval}")
    return (int(version_start) // int(interval)) * int(interval)


def entity_key(agent_id: str, version: int | None) -> EntityKey:
    """``("main", 1200)`` -> ``"main@v1200"``; ``(agent, None)`` -> ``agent``."""
    return agent_id if version is None else f"{agent_id}@v{int(version)}"


def lineage(entity: EntityKey) -> str:
    """The entity without its ``@v<N>`` suffix (module docstring)."""
    return _VERSION_SUFFIX_RE.sub("", entity)


def entity_version(entity: EntityKey) -> int | None:
    """``"main@v1200"`` -> 1200; an entity without a version suffix -> None."""
    match = _VERSION_SUFFIX_RE.search(entity)
    return None if match is None else int(match.group(1))
```

- [ ] **Step 4: Record fields (`src/colosseum/core/types.py`)**

After `OPPONENT_CATEGORIES = (...)` add:
```python
# Where a match record comes from (SP4 spec block 1): training workers, a tournament, the pulse process.
MATCH_SOURCES = ("train", "tournament", "pulse")
```
Replace `Lineup`, `SeatResult` and `MatchResult` (keep every existing field and its order; the new ones go last):
```python
@dataclass
class Lineup:
    """One match composition: a layout of the game and one assignment per seat of it.

    Optional (SP4): ``seed`` is the explicit episode seed (tournament, pulse; None = ``MatchRunner``'s derived
    seed), ``block`` the design block copied into ``MatchResult.block``, ``uid`` a deterministic match uid
    (None = the runner makes one). A lineup with a ``uid`` is meant to be played once.
    """

    layout: str
    seats: list[SeatAssignment]
    seed: int | None = None
    block: str | None = None
    uid: str | None = None


@dataclass
class SeatResult:
    """One seat of a finished match. ``reward`` is the undiscounted episode return; ``source`` is the
    seat's ``SeatAssignment.source``. ``version_start`` / ``version_end`` (SP4): the seat's weights version at the
    episode's start and end (``latest``: the worker's version of the agent, which can change mid-episode;
    ``ckpt_v<N>``: N; ``fixed``: None; None also when the runner has no version source)."""

    seat: int
    role: str
    team: int
    agent_id: str
    network_id: str
    reward: float
    eliminated_step: int | None = None
    source: str = ""
    version_start: int | None = None
    version_end: int | None = None
```
(`TeamResult` is unchanged.)
```python
@dataclass
class MatchResult:
    """A finished match: one SeatResult per occupied seat, one TeamResult per team.

    SP4 (spec block 1): ``uid`` is globally unique (``<run id>/train/w<W>_e<E>_ep<K>`` in training, the lineup's
    uid in a tournament; ``match_id`` is unique only within one runner); ``source`` is one of ``MATCH_SOURCES``;
    ``seed`` the episode seed when known; ``env_steps`` the global env-step counter when the result was received
    (stamped by the training main process; a pulse match carries its snapshot's); ``block`` the tournament / pulse
    design block (None in training).
    """

    match_id: str
    layout: str
    outcome_kind: Literal["score", "wdl", "rank"]
    seats: list[SeatResult]
    teams: list[TeamResult]
    episode_length: int
    uid: str = ""
    source: str = "train"
    seed: int | None = None
    env_steps: int | None = None
    block: str | None = None
```

- [ ] **Step 5: `MatchRunner` (`src/colosseum/worker/match_runner.py`)**

1. Append to the module docstring (after the `context` paragraph):
```
Match records (SP4 spec block 1). Every ``SeatResult`` carries the seat's weights version at the episode's start
(read right after the env reset) and at its end (read when the result is made): ``ckpt_v<N>`` -> N, ``fixed`` ->
None, ``latest`` -> ``version_fn(agent_id)`` (the worker's current version; None without ``version_fn``). A lineup's
explicit ``seed`` replaces the derived episode seed (``SeedSequence([seed, env, episode])``); the episode seed (when
there is one) goes into ``MatchResult.seed``, the lineup's ``block`` into ``MatchResult.block``, and the uid is the
lineup's ``uid`` or ``uid_prefix + match_id``. ``source`` (one of ``MATCH_SOURCES``) is copied into every result.
```
2. Imports: `from dataclasses import dataclass, replace`; `from colosseum.arena.entities import snapshot_version` (before `from colosseum.core.errors import PlayerError`); add `MATCH_SOURCES` to the `colosseum.core.types` import.
3. `_EnvState.__init__`: after `self.episode_index = 0` add
```python
        self.seed: int | None = None                 # the current episode's seed
        self.version_start: dict[int, int | None] = {}   # weights version per seat at the episode start
```
4. `MatchRunner.__init__`: new keyword arguments after `match_id_prefix: str = "m",`:
```python
        version_fn: Callable[[str], int | None] | None = None,
        source: str = "train",
        uid_prefix: str = "",
```
   first statement of the body:
```python
        if source not in MATCH_SOURCES:
            raise ValueError(f"MatchRunner: source must be one of {MATCH_SOURCES}, got {source!r}")
```
   and after `self._prefix = match_id_prefix`:
```python
        self._version_fn = version_fn
        self._source = source
        self._uid_prefix = uid_prefix
```
5. `_resolve`: its last line `return Lineup(layout=lineup.layout, seats=seats)` becomes `return replace(lineup, seats=seats)   # seed, block and uid stay`. Add after `_resolve`:
```python
    def _seat_version(self, assignment: SeatAssignment) -> int | None:
        """The seat's weights version now: ``ckpt_v<N>`` -> N, ``fixed`` -> None, ``latest`` -> ``version_fn``."""
        net = assignment.network_id
        if net == FIXED_NETWORK_ID:
            return None
        if net == LATEST_NETWORK_ID:
            if self._version_fn is None:
                return None
            version = self._version_fn(assignment.agent_id)
            return None if version is None else int(version)
        return snapshot_version(net)
```
6. `_match_result`: the `SeatResult(...)` gets `version_start=env_state.version_start.get(s), version_end=self._seat_version(a),` after `source=a.source,`; the return becomes
```python
        lineup = env_state.lineup
        match_id = f"{self._prefix}{e}_ep{env_state.episode_index}"
        return MatchResult(
            match_id=match_id,
            layout=layout,
            outcome_kind=self.spec.outcome_kind(layout),
            seats=seats,
            teams=[TeamResult(team=t, rank=float(ranks[t]), score=float(scores[t])) for t in sorted(ranks)],
            episode_length=tracker.episode_step,
            uid=lineup.uid if lineup.uid is not None else f"{self._uid_prefix}{match_id}",
            source=self._source,
            seed=env_state.seed,
            block=lineup.block,
        )
```
   (`_match_result` runs before the staged lineup is applied in `_after_step`, so `env_state.lineup` is the episode's lineup.)
7. `_reset_envs`: replace the `requests = {...}` comprehension by
```python
        requests = {}
        for e in envs:
            lineup = self._envs[e].lineup
            seed = (int(lineup.seed) if lineup.seed is not None
                    else self._episode_seed(e, self._envs[e].episode_index))
            requests[e] = (seed, lineup.layout)
```
   and in the loop over `envs`, after `env_state.result = results[e]`, add
```python
            env_state.seed = requests[e][0]
            env_state.version_start = {s: self._seat_version(a) for s, a in enumerate(env_state.lineup.seats)}
```
   (bots keep being reset with `requests[e][0]`, so an explicit seed also seeds the scripted seats, as in SP3.)

- [ ] **Step 6: `RolloutLoop` (`src/colosseum/worker/rollout_loop.py`)**

1. Module docstring, after the line "- A chunk's ``policy_version`` is the agent's version at its first slot.":
```
- **Match results** (SP4 spec block 1) carry each latest seat's policy version at the episode's start and end
  (``MatchRunner(version_fn=...)`` reads ``policy_versions``, which change when weights are synced, also
  mid-episode) and the uid ``<run_id>/train/w<W>_e<E>_ep<K>`` (``match_id`` without a run id). A result
  ``LoopIO.report_result`` could not deliver (it returns False) is counted in ``stats["dropped_results"]``.
```
2. `LoopIO.report_result`:
```python
    # Returns False when the result was dropped (counted in stats["dropped_results"]); None or True: delivered.
    report_result: Callable[[MatchResult], bool | None] | None = None
```
3. `RolloutLoop.__init__`: keyword `run_id: str | None = None` after `teachers`; after `self._dropped_reward_warned = False` add `self._dropped_results = 0`; the `MatchRunner(...)` call gets
```python
                version_fn=self._policy_versions.get, source="train",
                uid_prefix=f"{run_id}/train/" if run_id else "",
```
   (`self._policy_versions` is filled by `sync_weights()` before the runner is built, so the initial resets read real versions.)
4. `stats`: add `"dropped_results": self._dropped_results,` after `"dropped_reward_episodes"`.
5. `on_episode_end`: replace
```python
        if self._io.report_result is not None:
            self._io.report_result(end.result)
```
   by
```python
        if self._io.report_result is not None and self._io.report_result(end.result) is False:
            self._dropped_results += 1
```

- [ ] **Step 7: The worker process (`src/colosseum/worker/rollout_worker.py`)**

1. Import `from colosseum.utils.process import SHUTDOWN_GRACE_SEC, flush_queue`.
2. Replace `report_worker_stats` by
```python
def report_worker_stats(q, worker_id: int, stats: dict, timeout: float | None = None) -> None:
    """Best-effort worker stats for the main process (system metrics). Never blocks, unless ``timeout`` (seconds)
    is given: the final report on exit waits that long for room in the queue."""
    item = {"kind": "worker_stats", "worker_id": int(worker_id), **{k: int(v) for k, v in stats.items()}}
    try:
        if timeout is None:
            q.put_nowait(item)
        else:
            q.put(item, timeout=timeout)
    except queue.Full:
        pass
```
3. `rollout_worker_process`: keyword `run_id: str | None = None` after `teachers`, passed to `RolloutLoop(..., run_id=run_id)`. Docstring: "results are put non-blocking on ``results_queue`` (a full queue drops the result: counted in the stats as ``dropped_results``, the first drop is a WARNING);" and, at its end, "``run_id`` prefixes the match uids (``<run_id>/train/...``). On exit a final stats report is sent, and the results and stats queues are flushed (``flush_queue``) so that neither the last results nor the final counters are lost."
4. Replace the `report_result` closure by
```python
    drop_warned = [False]

    def report_result(result: MatchResult) -> bool:
        try:
            results_queue.put_nowait(result)
            return True
        except queue.Full:
            if not drop_warned[0]:
                drop_warned[0] = True
                logger.warning(f"Worker {worker_id}: the results queue is full; dropped match result "
                               f"{result.uid or result.match_id} (counted in dropped_match_results of the system "
                               f"metrics; further drops are only counted)")
            return False
```
5. Replace the final `try/finally` block by
```python
    try:
        loop.run(should_stop=should_stop, max_env_steps=max_env_steps)
    finally:
        if counter is not None:
            counter.flush()
        try:
            if stats_queue is not None:   # the final counters (dropped results, dropped reward episodes)
                report_worker_stats(stats_queue, worker_id, loop.stats, timeout=1.0)
            loop.close()
        finally:
            # The results and the final stats must reach the main process (it drains them until this process
            # exits); then detach every feeder thread, so undrained items (e.g. chunks a stopped learner never
            # consumed) cannot block exit.
            for q in (results_queue, stats_queue):
                if q is not None:
                    flush_queue(q, warn_after=SHUTDOWN_GRACE_SEC)
            for q in [*trajectory_queues.values(), results_queue, stats_queue]:
                if q is not None and hasattr(q, "cancel_join_thread"):
                    q.cancel_join_thread()
```
   `flush_queue` waits only while the parent lives (the launcher drains results and metrics during its shutdown, and terminates stragglers after `SHUTDOWN_GRACE_SEC`); objects without `join_thread` (test queues, the distributed role's adapters) are skipped.

- [ ] **Step 8: The launcher, the run id, eval and the config**

`src/colosseum/launcher.py`:
1. `_RESULTS_QUEUE_SIZE = 1000` becomes
```python
# Match results from all workers (SP4 spec block 1): a full queue drops results, and workers count the drops.
_RESULTS_QUEUE_SIZE = 10_000
```
2. `_worker_main`: keyword `run_id: str | None = None` after `teachers`, passed as `run_id=run_id` to `rollout_worker_process`; docstring: "...its scripted kickstart teachers (DAgger); ``run_id`` prefixes the match uids."
3. `_resolve_lineups`: `resolved.append(Lineup(layout=lineup.layout, seats=seats))` becomes `resolved.append(replace(lineup, seats=seats))   # seed, block and uid stay`.
4. `Launcher.__init__`: after `self._metrics_queue: mp.Queue | None = None` add
```python
        # Set by launch(): this run's id (run.json; match uids and the journal's runs table).
        self._run_id: str | None = None
```
5. `launch()`: replace `logger.info("Colosseum: Starting training pipeline")` by
```python
        self._run_id = self._run_dir.ensure_run_id()
        logger.info(f"Colosseum: Starting training pipeline (run id {self._run_id})")
```
6. `_start_children`: the worker `kwargs` get `run_id=self._run_id,` after `metrics_queue=metrics_queue,`.
7. `_drain_results`:
```python
    def _drain_results(self) -> None:
        """Feed queued match results to the coordinator (ratings, PFSP) and the hub; a result without
        ``env_steps`` is stamped with the global env-step counter at its arrival (spec block 1; a pulse result
        arrives with its snapshot's, T6.1)."""
        if self._results_queue is None:
            return
        workers = lambda: [n for n in self._supervisor.names if n.startswith("worker-")]  # noqa: E731
        results = self._drain(self._results_queue, "results", workers)
        if not results:
            return
        env_steps = int(self._env_step_counter.value)
        for result in results:
            if result.env_steps is None:
                result.env_steps = env_steps
            self._coordinator.report_match_result(result)
            self._hub.on_match_result(result)
```

`src/colosseum/core/run_dir.py`: add `run.json` to the layout in the module docstring (`run.json                 the run's id ({"run_id": <uuid4 hex>}), written once by the launcher (SP4)`), `import json`, `import uuid`, and
```python
RUN_INFO_FILE = "run.json"


def read_run_id(root: str | Path) -> str | None:
    """The run id stored in ``<root>/run.json``; None when the file is absent (an SP3 run dir, or a run that never
    started). Never writes. An unreadable or malformed file is a ConfigError naming it."""
    path = Path(root) / RUN_INFO_FILE
    if not path.is_file():
        return None
    try:
        run_id = json.loads(path.read_text())["run_id"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ConfigError(f"Malformed {path}: {e!r}; it must be {{\"run_id\": \"<id>\"}}") from e
    if not isinstance(run_id, str) or not run_id:
        raise ConfigError(f"Malformed {path}: run_id must be a non-empty string, got {run_id!r}")
    return run_id
```
and in `RunDir`, after `resolved_config_path`:
```python
    @property
    def run_info_path(self) -> Path:
        return self.root / RUN_INFO_FILE

    @property
    def run_id(self) -> str | None:
        """The stored run id (``read_run_id``); None before ``ensure_run_id``."""
        return read_run_id(self.root)

    def ensure_run_id(self) -> str:
        """The run's id, created on the first call (uuid4 hex, written atomically to ``run.json``)."""
        existing = read_run_id(self.root)
        if existing is not None:
            return existing
        run_id = uuid.uuid4().hex
        write_text_atomic(self.run_info_path, json.dumps({"run_id": run_id}) + "\n")
        return run_id
```
`RunDir.create` does not write `run.json` (`tests/unit/test_sp2_run_dir.py:127` pins what `create` leaves behind).

`src/colosseum/eval.py` (`play_lineups`): replace the copy of the lineups by
```python
    lineups = [dataclasses.replace(lineup, seats=[dataclasses.replace(seat, collect=False) for seat in lineup.seats])
               for lineup in lineups]
```
(SP2/SP3 eval lineups carry no seed: their seeding is unchanged.)

`src/colosseum/core/config.py`: module docstring gains the line `SP4: the global ``match_log`` section.`; before `class RunConfig` add
```python
class MatchLogConfig(StrictModel):
    """The match log ``runs/<name>/matches.db`` (SP4 spec block 1). Global only (no per-agent override)."""

    enabled: bool = Field(
        default=True,
        description="Journal every training match (false: only the online ELO of SP2; snapshot ratings, top-k "
                    "snapshots and evaluation need the journal).",
    )
    detail_window_mb: float = Field(
        default=512.0, gt=0,
        description="Detailed training matches are kept within this size (an estimate, about 1M matches at "
                    "512); older ones are compacted into counts, which keep the ratings exact.",
    )
    min_free_disk_mb: float = Field(
        default=2048.0, ge=0,
        description="When the journal's disk has less free space, new training matches go to the counts only "
                    "(the detail resumes when space is back).",
    )
    compact_interval_sec: float = Field(default=60.0, gt=0, description="Seconds between compaction passes.")
```
and in `ColosseumConfig`, after `metrics: MetricsConfig = ...`: `match_log: MatchLogConfig = Field(default_factory=MatchLogConfig)`. (`TrainableAgent` has no `match_log` field, so a per-agent override is a ConfigError; `get_agent_config` carries the global section along.)

`scripts/bench_throughput.py::_make_config`, after the `metrics=...` argument:
```python
        # SP4 criterion 3.6: the journal is on while throughput is measured.
        match_log={"enabled": True, "detail_window_mb": 512.0, "min_free_disk_mb": 2048.0,
                   "compact_interval_sec": 60.0},
```

- [ ] **Step 9: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_match_records.py tests/contract/test_sp4_result_flow.py tests/contract/test_rollout_loop_lineups.py tests/unit/test_match_runner.py tests/unit/test_sp3_match_runner_players.py tests/unit/test_rollout_worker_v2.py tests/unit/test_bench_throughput.py tests/unit/test_sp2_launcher_lifecycle.py -q -rw`
Expected: all pass.

- [ ] **Step 10: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings. (The suite also exercises the worker's exit flush through every `colosseum train` integration test.)

- [ ] **Step 11: Commit and push**

```bash
git add src/colosseum/arena src/colosseum/core/types.py src/colosseum/core/run_dir.py src/colosseum/core/config.py \
        src/colosseum/worker src/colosseum/launcher.py src/colosseum/eval.py scripts/bench_throughput.py \
        tests/unit/test_sp4_match_records.py tests/contract/test_sp4_result_flow.py \
        tests/contract/test_rollout_loop_lineups.py tests/unit/test_sp2_launcher_lifecycle.py
git commit -m "feat: match record fields, per-seat weights versions, explicit lineup seeds, counted result drops, match_log config"
git push
```

---

### Task T1.2: `arena.comparisons` and `arena.matchlog` (library: schema, append, counts, compaction, disk guard, import, reader)

Spec block 1 ("Хранилище", "Своевременная чистка", "Resume из папки запуска", "API для чтения") and block 2 ("Модель": the unit of comparison). A pure library: no launcher, no queues. `colosseum.arena.comparisons` turns match results into team-pair sufficient statistics (`TeamKey` = team index + sorted `(entity, role)` members; per pair `n`, `Σw`, `Σw·y` with `w = 1/(T−1)` and `y` from ranks; one-team layouts keep `n`, sum and sum of squares of the team score). `colosseum.arena.matchlog.MatchLog` is the SQLite journal: WAL, `auto_vacuum = INCREMENTAL`, `synchronous = NORMAL`, `journal_size_limit`, `busy_timeout`; an `entities` id table; `matches` / `seats` / `teams` for detailed matches, `counts` for compacted ones; one transaction per appended batch; duplicates by uid ignored; compaction of the oldest detailed training matches into `counts` in batches of ≤ 10 000 until the estimated detail size is within the window (matches of evicted entities first), then `incremental_vacuum` and `wal_checkpoint(TRUNCATE)`; tournament and pulse matches are never compacted; the disk guard sends new training matches straight to `counts`; `import_parent` folds a parent journal's detailed training matches into `counts` and copies its counts, tournament and pulse matches; `MatchLogReader` reads read-only (missing file = empty journal). The fits that consume the tables are T2.1's: that task adds the test "fits from the journal equal fits from the in-memory table before and after compaction"; here the inputs (the tables) are compared.

**Files:**
- Create: `src/colosseum/arena/comparisons.py`, `src/colosseum/arena/matchlog.py`
- Modify: `tests/game_helpers.py` (`make_match_result`, `synthetic_journal_results`)
- Test: `tests/unit/test_sp4_comparisons.py`, `tests/unit/test_sp4_matchlog.py`

**Interfaces:**
- Consumes: `colosseum.arena.entities` (T1.1), `MatchResult` / `SeatResult` / `TeamResult` / `MATCH_SOURCES` (T1.1), `colosseum.core.outcomes.pairwise_rank_score` (`core/outcomes.py:67`).
- Produces (contract T1.2, plus additions marked *):
  - `colosseum.arena.comparisons`: `TeamKey(team, members)` (frozen, ordered; *`TeamKey.of(team, members)`, *`entities()`, *`encode()` / *`decode()`), `PairStat(n, sw, swy)` (*`add`), *`ScoreStat(n, total, total_sq)` (the contract's "ScoreStat … n, sum, sumsq"), `ComparisonTable` (`add_result`, `add`, *`add_score`, `merge`, *`copy`, *`version`, `layouts`, *`sources`, `pairs`, `scores`, `entity_games`, *`as_dict`), `match_pairs(result, rating_versions)`, *`match_score`, *`team_keys`, *`pairs_of(keys, ranks)`, *`score_of(keys, scores)`, *`coefficients(a, b)`, *`Pair`;
  - `colosseum.arena.matchlog`: `MATCH_LOG_FILE`, `SCHEMA_VERSION`, `RunInfo`, `MatchLog(path, run, *, disk_free_fn, min_free_disk_mb)` with `append(results, *, tournament=None)` (*keyword `tournament`), `compact(*, window_bytes, max_batch=10_000, evicted=(), should_stop=...)` (*keyword `should_stop`), `import_parent`, `comparison_table`, `sizes`, `close`, `open_readonly`, *`rating_versions(result)`, *`degraded`; `MatchLogReader(path)` with `matches(*, agent, version, layout, source, tournament, outcome, env_steps, limit)` (*`version`, *`outcome`, *`env_steps`), `comparison_table`, `blocks`, `runs`, `sizes`, *`close`; *`AppendStats`, *`CompactStats`, *`ImportStats`, *`LogSizes`, *`MatchRow`, *`SeatRow`, *`MatchLogError`, *`estimated_size`, *`default_disk_free`;
  - test support: *`make_match_result(seats, ranks=None, *, layout, uid, source, scores, ...)`, *`synthetic_journal_results(n, *, seed, prefix, source, layout, interval, first)`.

- [ ] **Step 1: Add the record builders to `tests/game_helpers.py`**

Append:
```python
# ---------------------------------------------------------------------------
# SP4 (T1.2): match records for journal and rating tests
# ---------------------------------------------------------------------------


def make_match_result(seats: Sequence[tuple], ranks: Sequence[float] | None = None, *, layout: str = "2p",
                      uid: str = "u0", source: str = "train", scores: Sequence[float] | None = None,
                      match_id: str | None = None, env_steps: int | None = None, seed: int | None = None,
                      block: str | None = None, length: int = 5, outcome_kind: str | None = None):
    """A ``MatchResult`` from compact seat tuples ``(team, agent_id, network_id[, role[, version_start[,
    version_end]]])`` in seat order. ``ranks`` / ``scores``: one per team (default: team 0 best, then 1, ...; scores
    0); the outcome kind follows the number of teams unless given."""
    from colosseum.core.types import MatchResult, SeatResult, TeamResult

    seat_results = []
    for i, spec in enumerate(seats):
        team, agent, network, *rest = spec
        role = rest[0] if len(rest) > 0 else "player"
        start = rest[1] if len(rest) > 1 else None
        end = rest[2] if len(rest) > 2 else start
        seat_results.append(SeatResult(seat=i, role=role, team=int(team), agent_id=agent, network_id=network,
                                       reward=0.0, version_start=start, version_end=end))
    num_teams = 1 + max(s.team for s in seat_results)
    ranks = list(ranks) if ranks is not None else [float(t + 1) for t in range(num_teams)]
    scores = list(scores) if scores is not None else [0.0] * num_teams
    kind = outcome_kind or ("score" if num_teams == 1 else "wdl" if num_teams == 2 else "rank")
    return MatchResult(match_id=match_id or uid, layout=layout, outcome_kind=kind, seats=seat_results,
                       teams=[TeamResult(t, float(ranks[t]), float(scores[t])) for t in range(num_teams)],
                       episode_length=length, uid=uid, source=source, seed=seed, env_steps=env_steps, block=block)


def synthetic_journal_results(n: int, *, seed: int = 0, prefix: str = "R/train/", source: str = "train",
                              layout: str = "2p", interval: int = 10, first: int = 0) -> list:
    """``n`` tic-tac-toe-like 1v1 results for journal tests: agent ``main``'s latest weights (``version_start`` in
    ``[0, 3 * interval)``, i.e. buckets v0 / v10 / v20 at the default interval) against ``main@ckpt_v10``,
    ``main@ckpt_v20``, the fixed agents ``random`` / ``greedy`` or latest itself, on random sides, with draws.
    uids ``<prefix>w0_e<k % 4>_ep<k>`` for ``k = first .. first + n - 1``; ``env_steps = 10 * k``."""
    rng = np.random.default_rng(seed)
    opponents = [("main", "ckpt_v10"), ("main", "ckpt_v20"), ("random", "fixed"), ("greedy", "fixed"),
                 ("main", "latest")]
    out = []
    for k in range(first, first + n):
        version = int(rng.integers(0, 3 * interval))
        agent, network = opponents[int(rng.integers(len(opponents)))]
        own = (0, "main", "latest", "player", version, version + int(rng.integers(0, 2)))
        other = (1, agent, network, "player", version if network == "latest" else None,
                 version if network == "latest" else None)
        seats = [own, other] if rng.random() < 0.5 else [(0, *other[1:]), (1, *own[1:])]
        u = rng.random()
        ranks = [1.0, 1.0] if u < 0.1 else ([1.0, 2.0] if u < 0.55 else [2.0, 1.0])
        out.append(make_match_result(seats, ranks, layout=layout, uid=f"{prefix}w0_e{k % 4}_ep{k}",
                                     match_id=f"w0_e{k % 4}_ep{k}", source=source, env_steps=10 * k,
                                     seed=int(rng.integers(2**31)), length=int(rng.integers(5, 10))))
    return out
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp4_comparisons.py`:
```python
"""Team-pair sufficient statistics (SP4 T1.2, spec blocks 1-2): team keys with team index and roles, weights
1/(T-1), outcomes from ranks, canonical order, self-play excluded from games, score layouts, copy / merge, and a
thread-safety smoke test."""
from __future__ import annotations

import threading

import pytest

from colosseum.arena.comparisons import (
    ComparisonTable,
    PairStat,
    ScoreStat,
    TeamKey,
    coefficients,
    match_pairs,
    match_score,
    team_keys,
)
from game_helpers import make_match_result


def test_team_keys_carry_the_team_index_entities_and_roles():
    result = make_match_result([(0, "main", "latest", "hunter", 1250), (1, "main", "ckpt_v700", "prey"),
                                (1, "random", "fixed", "prey")], layout="1v2")
    keys = team_keys(result, [1200, 700, None])
    assert keys == {0: TeamKey(0, (("main@v1200", "hunter"),)),
                    1: TeamKey(1, (("main@v700", "prey"), ("random", "prey")))}
    assert TeamKey.decode(keys[1].encode()) == keys[1]
    with pytest.raises(ValueError, match="rating versions"):
        team_keys(result, [1])


def test_pairs_weights_and_outcomes_from_ranks():
    duel = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")], [2.0, 1.0])
    ((a, b, w, y),) = match_pairs(duel, [None, None])
    assert (a.team, b.team, w, y) == (0, 1, 1.0, 0.0)
    draw = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")], [1.0, 1.0])
    assert match_pairs(draw, [None, None])[0][3] == 0.5
    ffa = make_match_result([(t, f"p{t}", "fixed") for t in range(4)], [2.0, 1.0, 4.0, 2.0], layout="4p")
    pairs = match_pairs(ffa, [None] * 4)
    assert len(pairs) == 6 and all(w == pytest.approx(1 / 3) for *_, w, _ in pairs)
    outcome = {(a.team, b.team): y for a, b, _, y in pairs}
    assert outcome == {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 0.5, (1, 2): 1.0, (1, 3): 1.0, (2, 3): 0.0}
    solo = make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[7.5], layout="coop")
    assert match_pairs(solo, [None, None]) == []
    key, score = match_score(solo, [None, None])
    assert score == 7.5 and key.entities() == ["a", "b"]


def test_coefficients_cancel_an_entity_on_both_sides():
    a = TeamKey.of(0, [("x", "p"), ("y", "p")])
    b = TeamKey.of(1, [("x", "p"), ("z", "p")])
    assert coefficients(a, b) == {"y": 0.5, "z": -0.5}
    assert coefficients(TeamKey.of(0, [("x", "p")]), TeamKey.of(1, [("x", "p")])) == {}


def test_the_table_stores_pairs_in_canonical_order_and_sums_them():
    table = ComparisonTable()
    a, b = TeamKey.of(0, [("x", "p")]), TeamKey.of(1, [("y", "p")])
    table.add("2p", "train", b, a, PairStat(3, 3.0, 1.0))           # given as (b, a): stored as (a, b)
    table.add("2p", "train", a, b, PairStat(2, 2.0, 2.0))
    table.add("2p", "pulse", a, b, PairStat(1, 1.0, 0.0))
    assert table.pairs("2p", ["train"]) == [(a, b, PairStat(5, 5.0, 4.0))]
    assert table.pairs("2p", ["train", "pulse"]) == [(a, b, PairStat(6, 6.0, 4.0))]
    assert table.pairs("2p", ["tournament"]) == [] and table.layouts() == ["2p"]
    assert table.sources("2p") == ["pulse", "train"]
    with pytest.raises(ValueError, match="team indices"):
        table.add("2p", "train", a, a, PairStat(1, 1.0, 1.0))


def test_self_play_counts_for_no_entity_and_games_count_matches():
    table = ComparisonTable()
    for ranks in ([1.0, 2.0], [2.0, 1.0]):
        table.add_result(make_match_result([(0, "main", "latest", "player", 3), (1, "main", "latest", "player", 4)],
                                           ranks), [0, 0])          # one bucket against itself: only the side bias
    table.add_result(make_match_result([(0, "main", "latest", "player", 15), (1, "random", "fixed")]), [10, None])
    assert table.entity_games("2p", ["train"]) == {"main@v10": 1, "random": 1}
    ((a, b, stat), _) = sorted(table.pairs("2p", ["train"]), key=lambda r: r[2].n, reverse=True)
    assert a.entities() == b.entities() == ["main@v0"] and stat == PairStat(2, 2.0, 1.0)


def test_score_layouts_keep_n_sum_and_sum_of_squares():
    table = ComparisonTable()
    for score in (1.0, 3.0):
        table.add_result(make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[score], layout="coop"),
                         [None, None])
    ((key, stat),) = table.scores("coop", ["train"])
    assert key.entities() == ["a", "b"] and stat == ScoreStat(2, 4.0, 10.0)
    assert table.pairs("coop", ["train"]) == [] and table.entity_games("coop", ["train"]) == {"a": 2, "b": 2}


def test_copy_is_independent_and_merge_adds_everything():
    table = ComparisonTable()
    table.add_result(make_match_result([(0, "a", "fixed"), (1, "b", "fixed")]), [None, None])
    copy = table.copy()
    table.add_result(make_match_result([(0, "a", "fixed"), (1, "b", "fixed")]), [None, None])
    assert copy.pairs("2p", ["train"])[0][2].n == 1 and table.pairs("2p", ["train"])[0][2].n == 2
    version = copy.version
    copy.merge(table)
    assert copy.pairs("2p", ["train"])[0][2] == PairStat(3, 3.0, 3.0) and copy.version > version


def test_concurrent_writers_and_copies_lose_nothing():
    table = ComparisonTable()
    result = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")])

    def writer():
        for _ in range(2000):
            table.add_result(result, [None, None])

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        table.copy()                                    # a reader thread copying while the writers write
    for thread in threads:
        thread.join()
    assert table.pairs("2p", ["train"])[0][2].n == 4000
```

Create `tests/unit/test_sp4_matchlog.py`:
```python
"""The match log library (SP4 T1.2, spec block 1 and section 6 "журнал"): schema and SQLite settings, append in
one transaction, duplicates, the comparison table from detail and counts, compaction within the window (tables
unchanged, team index kept, tournament / pulse untouched, evicted entities first), the disk guard, the resume
import, the read-only reader while a writer writes, two writers, corrupt files and other schema versions."""
from __future__ import annotations

import logging
import sqlite3
import threading

import pytest

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    SCHEMA_VERSION,
    MatchLog,
    MatchLogError,
    MatchLogReader,
    RunInfo,
    estimated_size,
)
from game_helpers import make_match_result, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)
FREE = 10**12


def _log(tmp_path, run=RUN, name=MATCH_LOG_FILE, free=lambda _: FREE, **kwargs) -> MatchLog:
    return MatchLog(tmp_path / name, run, disk_free_fn=free, **kwargs)


def _memory_table(log: MatchLog, results) -> ComparisonTable:
    table = ComparisonTable()
    for result in results:
        table.add_result(result, log.rating_versions(result))
    return table


def _close(a: dict, b: dict, tol: float = 1e-9) -> bool:
    """Two ``ComparisonTable.as_dict()`` results agree: same keys, counts equal, sums within ``tol``."""
    if a.keys() != b.keys():
        return False
    for key in a:
        for part in ("pairs", "scores"):
            if a[key][part].keys() != b[key][part].keys():
                return False
            for cell, (n, s1, s2) in a[key][part].items():
                m, t1, t2 = b[key][part][cell]
                if n != m or abs(s1 - t1) > tol * max(1.0, abs(t1)) or abs(s2 - t2) > tol * max(1.0, abs(t2)):
                    return False
    return True


def _sql(path, query, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query, args).fetchall()
    finally:
        conn.close()


def test_schema_tables_indexes_and_sqlite_settings(tmp_path):
    log = _log(tmp_path)
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    tables = {r[0] for r in _sql(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"meta", "entities", "runs", "matches", "seats", "teams", "counts"}
    indexes = {r[0] for r in _sql(path, "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE "
                                        "'sqlite_autoindex%'")}
    assert {"matches_by_source", "seats_by_agent", "seats_by_entity"} <= indexes
    assert _sql(path, "PRAGMA journal_mode") == [("wal",)]
    assert _sql(path, "PRAGMA auto_vacuum") == [(2,)]                 # INCREMENTAL
    assert _sql(path, "SELECT value FROM meta WHERE key = 'schema_version'") == [(str(SCHEMA_VERSION),)]
    assert _sql(path, "SELECT run_id, checkpoint_interval FROM runs") == [("R", 10)]


def test_append_writes_detail_with_rating_versions_and_ignores_duplicates(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(50)
    stats = log.append(results)
    assert (stats.detailed, stats.duplicates, stats.counted) == (50, 0, 0) and all(stats.accepted)
    again = log.append(results[:5] + synthetic_journal_results(1, first=50))
    assert (again.detailed, again.duplicates) == (1, 5) and again.accepted == [False] * 5 + [True]
    sizes = log.sizes()
    assert sizes.detail_matches == 51 and sizes.detail_bytes == sum(estimated_size(r) for r in results) + \
        estimated_size(synthetic_journal_results(1, first=50)[0])
    row = log.open_readonly(log.path).matches(limit=1)[0]                         # newest first
    assert row.uid == "R/train/w0_e2_ep50" and row.worker == 0 and row.env_steps == 500
    main = next(s for s in row.seats if s.agent_id == "main" and s.network_id == "latest")
    assert main.rating_version == (main.version_start // 10) * 10 and main.entity == f"main@v{main.rating_version}"
    log.close()


def test_append_is_one_transaction(tmp_path):
    log = _log(tmp_path)
    bad = synthetic_journal_results(3)
    bad[2].source = "eval"
    with pytest.raises(ValueError, match="source"):
        log.append(bad)
    bad[2].source, bad[1].uid = "train", ""
    with pytest.raises(ValueError, match="uid"):
        log.append(bad)
    assert log.sizes().detail_matches == 0 and log.comparison_table().layouts() == []
    log.close()


def test_the_table_from_the_journal_equals_the_in_memory_table(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(300) + [
        make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[2.0], layout="coop", uid="coop1"),
        make_match_result([(t, f"p{t}", "fixed") for t in range(3)], [3.0, 1.0, 2.0], layout="3p", uid="ffa1")]
    log.append(results)
    assert log.comparison_table(("train",)).as_dict() == _memory_table(log, results).as_dict()
    log.close()


def test_compaction_keeps_the_window_and_the_table(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(4000)
    for i in range(0, len(results), 500):
        log.append(results[i:i + 500])
    window = log.sizes().detail_bytes // 5
    before = log.comparison_table(("train",)).as_dict()
    stats = log.compact(window_bytes=window, max_batch=700)
    sizes = log.sizes()
    assert stats.folded > 0 and stats.batches >= stats.folded // 700 and not stats.interrupted
    assert sizes.detail_bytes <= window and sizes.detail_bytes > window - 2 * estimated_size(results[0])
    assert sizes.counts_rows > 0 and sizes.wal_bytes == 0               # wal_checkpoint(TRUNCATE)
    after = log.comparison_table(("train",)).as_dict()
    assert _close(after, before) and _close(after, _memory_table(log, results).as_dict())
    oldest = _sql(log.path, "SELECT MIN(id) FROM matches")[0][0]
    assert oldest > stats.folded - 1                                    # the oldest matches were folded
    log.close()


def test_compaction_gives_space_back_and_never_touches_tournament_or_pulse(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(200, source="pulse", prefix="R/pulse/"))
    log.append(synthetic_journal_results(100, source="tournament", prefix="T1/"), tournament="T1")
    for i in range(10):
        log.append(synthetic_journal_results(1000, first=1000 * i))
    file_before = log.sizes().file_bytes
    log.compact(window_bytes=20_000)
    sizes = log.sizes()
    assert sizes.other_matches == 300 and sizes.file_bytes < file_before / 2
    assert sizes.detail_bytes <= 20_000 and sizes.detail_matches > 0      # the window is kept, not emptied
    reader = MatchLogReader(log.path)
    assert len(reader.matches(source="pulse", limit=1000)) == 200
    assert len(reader.matches(tournament="T1", limit=1000)) == 100
    assert reader.comparison_table(("tournament",), tournament="T1").layouts() == ["2p"]
    reader.close()
    log.close()


def test_matches_of_evicted_entities_are_folded_first(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(600)
    log.append(results)
    before = log.comparison_table().as_dict()
    stats = log.compact(window_bytes=10**9, evicted=["main@v10", "nobody@v5"])
    with_v10 = sum(any(s.network_id == "ckpt_v10" or (s.network_id == "latest" and 10 <= (s.version_start or -1) < 20)
                       for s in r.seats) for r in results)
    assert stats.evicted_folded == stats.folded == with_v10 > 0
    reader = MatchLogReader(log.path)
    assert reader.matches(agent="main", version=10, limit=10) == []
    assert len(reader.matches(agent="main", version=20, limit=10)) == 10
    assert _close(reader.comparison_table().as_dict(), before)
    reader.close()
    log.close()


def test_compaction_stops_between_batches(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(1000))
    calls = []
    stats = log.compact(window_bytes=0, max_batch=100, should_stop=lambda: calls.append(1) or len(calls) > 3)
    assert stats.interrupted and stats.batches == 3 and stats.folded == 300
    log.close()


def test_the_disk_guard_switches_to_counts_and_back(tmp_path, caplog):
    free = [FREE]
    log = _log(tmp_path, free=lambda _: free[0], min_free_disk_mb=100)
    results = synthetic_journal_results(300)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.matchlog"):
        assert log.append(results[:100]).detailed == 100
        free[0] = 50 * 2**20
        low = log.append(results[100:200])
        log.append(results[200:250])
        assert (low.counted, low.detailed, low.degraded) == (100, 0, True) and log.sizes().degraded
        tournament = log.append(synthetic_journal_results(10, source="tournament", prefix="T/"), tournament="T")
        assert tournament.detailed == 10                                # a tournament writes as usual
        free[0] = FREE
        assert log.append(results[250:]).detailed == 50 and not log.sizes().degraded
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    infos = [r for r in caplog.records if r.levelno == logging.INFO and "resumes" in r.getMessage()]
    assert len(warnings) == 1 and "min_free_disk_mb" in warnings[0].getMessage() and len(infos) == 1
    assert log.sizes().detail_matches == 150
    assert _close(log.comparison_table(("train",)).as_dict(), _memory_table(log, results).as_dict())
    log.close()


def test_import_parent_folds_its_detail_and_copies_counts_tournament_and_pulse(tmp_path):
    parent = _log(tmp_path, name="parent.db")
    train = synthetic_journal_results(800)
    parent.append(train)
    parent.compact(window_bytes=parent.sizes().detail_bytes // 2)       # half detail, half counts
    parent.append(synthetic_journal_results(40, source="pulse", prefix="R/pulse/"))
    parent.append(synthetic_journal_results(20, source="tournament", prefix="T1/"), tournament="T1")
    expected = parent.comparison_table(("train", "pulse")).as_dict()
    parent.close()
    child_run = RunInfo("C", "child", "R", "ttt", "h", 10)
    child = _log(tmp_path, run=child_run, name="child.db")
    stats = child.import_parent(tmp_path / "parent.db")
    assert stats.copied == 60 and stats.folded > 0 and stats.counts_rows > 0
    sizes = child.sizes()
    assert sizes.detail_matches == 0 and sizes.other_matches == 60      # no training detail copied
    assert _close(child.comparison_table(("train", "pulse")).as_dict(), expected)
    reader = MatchLogReader(tmp_path / "child.db")
    assert [r.run_id for r in reader.runs()] == ["C", "R"] and reader.runs()[0].parent_run_id == "R"
    assert len(reader.matches(tournament="T1", limit=100)) == 20
    reader.close()
    child.close()


def test_the_reader_reads_while_a_writer_writes(tmp_path):
    log = _log(tmp_path)
    log.append(synthetic_journal_results(10))
    reader = MatchLogReader(log.path)
    errors: list[BaseException] = []

    def write():
        try:
            writer = _log(tmp_path)
            for i in range(20):
                writer.append(synthetic_journal_results(100, first=10 + 100 * i))
            writer.close()
        except BaseException as e:   # noqa: BLE001 - reported below
            errors.append(e)

    thread = threading.Thread(target=write)
    thread.start()
    seen = []
    while thread.is_alive():
        seen.append(sum(s.n for _, _, s in reader.comparison_table(("train",)).pairs("2p", ["train"])))
    thread.join()
    assert not errors and seen == sorted(seen)                         # monotone: every read is consistent
    assert reader.sizes().detail_matches == 2010
    reader.close()
    log.close()


def test_two_writers_lose_no_record(tmp_path):
    training = _log(tmp_path)
    errors: list[BaseException] = []

    def tournament():
        try:
            writer = MatchLog(tmp_path / MATCH_LOG_FILE, None, disk_free_fn=lambda _: FREE)
            for i in range(30):                                     # one block per transaction
                block = synthetic_journal_results(10, source="tournament", prefix=f"T/{i}/")
                for r in block:
                    r.block = f"b{i}"
                writer.append(block, tournament="T")
            writer.close()
        except BaseException as e:   # noqa: BLE001 - reported below
            errors.append(e)

    thread = threading.Thread(target=tournament)
    thread.start()
    for i in range(30):
        training.append(synthetic_journal_results(100, first=100 * i))
    thread.join()
    assert not errors
    sizes = training.sizes()
    assert (sizes.detail_matches, sizes.other_matches) == (3000, 300)
    reader = MatchLogReader(training.path)
    assert reader.blocks("T") == {f"b{i}" for i in range(30)} and reader.blocks("other") == set()
    reader.close()
    training.close()


def test_matches_filters(tmp_path):
    log = _log(tmp_path)
    results = synthetic_journal_results(200)
    log.append(results)
    reader = MatchLogReader(log.path)
    rows = reader.matches(limit=5)
    assert [r.uid for r in rows] == [r.uid for r in results[::-1][:5]]                   # newest first
    assert all(r.env_steps <= 500 for r in reader.matches(env_steps=(0, 500), limit=1000))
    assert len(reader.matches(env_steps=(0, 500), limit=1000)) == 51
    wins = reader.matches(agent="random", outcome="win", limit=1000)
    assert wins and all(any(s.agent_id == "random" for s in r.seats) for r in wins)
    random_team = [next(s.team for s in r.seats if s.agent_id == "random") for r in wins]
    assert all(dict((t, rank) for t, rank, _ in r.teams)[team] == 1.0 and
               sorted(rank for _, rank, _ in r.teams) == [1.0, 2.0] for r, team in zip(wins, random_team))
    assert reader.matches(agent="nobody") == [] and reader.matches(layout="9p") == []
    with pytest.raises(ValueError, match="agent"):
        reader.matches(outcome="win")
    reader.close()
    log.close()


def test_a_missing_journal_reads_as_empty_and_other_schemas_are_refused(tmp_path):
    empty = MatchLogReader(tmp_path / "missing.db")
    assert empty.comparison_table().layouts() == [] and empty.matches() == [] and empty.runs() == []
    empty.close()
    log = _log(tmp_path)
    log.close()
    conn = sqlite3.connect(tmp_path / MATCH_LOG_FILE)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(MatchLogError, match="schema version"):
        _log(tmp_path)
    with pytest.raises(MatchLogError, match="schema version"):
        MatchLogReader(tmp_path / MATCH_LOG_FILE)


def test_a_corrupt_file_is_moved_away_and_a_new_journal_starts(tmp_path, caplog):
    (tmp_path / MATCH_LOG_FILE).write_bytes(b"this is not an sqlite database" * 100)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        log = _log(tmp_path)
    assert (tmp_path / f"{MATCH_LOG_FILE}.corrupt-1").exists()
    assert any("not a readable SQLite database" in r.getMessage() for r in caplog.records)
    assert log.append(synthetic_journal_results(3)).detailed == 3
    log.close()


def test_the_size_estimate_tracks_the_pages_sqlite_uses(tmp_path):
    """The window is enforced on ``estimated_size``; it must stay close to the real page use (calibrated within
    7 % on 1v1, 2v2 and FFA matches; the band here is 25 %)."""
    log = _log(tmp_path)
    for i in range(6):
        log.append(synthetic_journal_results(1000, first=1000 * i, prefix="0123456789abcdef0123456789abcdef/train/"))
    log.close()
    path = tmp_path / MATCH_LOG_FILE
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    pages, free, size = (conn.execute(f"PRAGMA {p}").fetchone()[0] for p in ("page_count", "freelist_count",
                                                                            "page_size"))
    estimate = int(conn.execute("SELECT value FROM meta WHERE key = 'detail_train_bytes'").fetchone()[0])
    conn.close()
    assert 0.75 < (pages - free) * size / estimate < 1.25
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_comparisons.py tests/unit/test_sp4_matchlog.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.arena.comparisons'` / `'colosseum.arena.matchlog'`.

- [ ] **Step 4: Write `src/colosseum/arena/comparisons.py`**

```python
"""Team-pair sufficient statistics of match results (SP4 spec blocks 1-2).

The unit of the Bradley-Terry ratings is a pair of teams of one match with ``T >= 2`` teams:
weight ``1 / (T - 1)``, outcome 1 / 0.5 / 0 from the team ranks (``pairwise_rank_score``, the rule of SP2's
``member_pairs`` at team level). A team is a :class:`TeamKey`: its team index in the layout (the index also
fixes its roles, so the side bias ``b[layout, team]`` of ``colosseum.arena.bt`` is keyed by it) and its
members as sorted ``(entity, role)`` pairs. For every ``(layout, source)`` a :class:`ComparisonTable` keeps,
per ordered team pair ``(a, b)`` with ``a.team < b.team``, the sums ``n`` (matches), ``sw`` (weights) and
``swy`` (weight * outcome of ``a``): exactly what the fit needs (point estimate and Hessian), so the journal's
compacted ``counts`` rows and its detailed matches give the same table.

One-team layouts (``score``) have no pairs: their team composition keeps ``n``, the sum and the sum of squares
of the team score (:class:`ScoreStat`).

A pair whose two teams have the same members (an entity against itself, e.g. latest against latest of one
version bucket) carries information about the side bias only: no entity's net coefficient is non-zero, so it
counts for no entity's games (``entity_games``).

The table is thread-safe: the journal writer thread adds to it, the live-rating thread takes ``copy()``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from colosseum.arena.entities import EntityKey, entity_key
from colosseum.core.outcomes import pairwise_rank_score
from colosseum.core.types import MatchResult

Pair = tuple["TeamKey", "TeamKey", float, float]   # (a, b, weight, outcome of a)

_COEF_EPS = 1e-12


@dataclass(frozen=True, order=True)
class TeamKey:
    """A team of a match as a rating unit: team index in the layout and sorted ``(entity, role)`` members."""

    team: int
    members: tuple[tuple[EntityKey, str], ...]

    def entities(self) -> list[EntityKey]:
        """Member entities (with repeats: two seats of one entity count twice)."""
        return [entity for entity, _ in self.members]

    def encode(self) -> str:
        """Canonical text form (reports, tests): ``[team, [[entity, role], ...]]`` as compact JSON."""
        return json.dumps([self.team, [list(m) for m in self.members]], separators=(",", ":"))

    @classmethod
    def decode(cls, text: str) -> TeamKey:
        team, members = json.loads(text)
        return cls(int(team), tuple((str(e), str(r)) for e, r in members))

    @classmethod
    def of(cls, team: int, members: Iterable[tuple[EntityKey, str]]) -> TeamKey:
        return cls(int(team), tuple(sorted((str(e), str(r)) for e, r in members)))


@dataclass
class PairStat:
    """Sufficient statistics of one ordered team pair: matches, sum of weights, sum of weight * outcome."""

    n: int = 0
    sw: float = 0.0
    swy: float = 0.0

    def add(self, other: PairStat) -> None:
        self.n += int(other.n)
        self.sw += float(other.sw)
        self.swy += float(other.swy)


@dataclass
class ScoreStat:
    """Team-score statistics of one composition of a one-team layout: matches, sum, sum of squares."""

    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, other: ScoreStat) -> None:
        self.n += int(other.n)
        self.total += float(other.total)
        self.total_sq += float(other.total_sq)


def team_keys(result: MatchResult, rating_versions: Sequence[int | None]) -> dict[int, TeamKey]:
    """``{team index: TeamKey}`` of a result; ``rating_versions`` has one entry per ``result.seats`` entry."""
    if len(rating_versions) != len(result.seats):
        raise ValueError(f"match {result.uid or result.match_id!r}: {len(rating_versions)} rating versions for "
                         f"{len(result.seats)} seats")
    members: dict[int, list[tuple[EntityKey, str]]] = {}
    for seat, version in zip(result.seats, rating_versions, strict=True):
        members.setdefault(int(seat.team), []).append((entity_key(seat.agent_id, version), seat.role))
    return {team: TeamKey.of(team, m) for team, m in sorted(members.items())}


def pairs_of(keys: Mapping[int, TeamKey], ranks: Mapping[int, float]) -> list[Pair]:
    """``(a, b, weight, outcome_a)`` of every pair of teams (``a.team < b.team``) that has members and a rank;
    weight ``1 / (T - 1)`` with ``T`` such teams; ``[]`` for fewer than two. The one rule for results and for
    journal rows (``colosseum.arena.matchlog`` folds detailed matches with it)."""
    teams = [t for t in sorted(keys) if t in ranks]
    if len(teams) < 2:
        return []
    weight = 1.0 / (len(teams) - 1)
    return [(keys[ta], keys[tb], weight, pairwise_rank_score(float(ranks[ta]), float(ranks[tb])))
            for i, ta in enumerate(teams) for tb in teams[i + 1:]]


def score_of(keys: Mapping[int, TeamKey], scores: Mapping[int, float]) -> tuple[TeamKey, float] | None:
    """``(team, team score)`` when exactly one team has members and a score (a one-team layout), else None."""
    if len(keys) != 1 or len(scores) != 1:
        return None
    ((team, key),) = keys.items()
    return (key, float(scores[team])) if team in scores else None


def match_pairs(result: MatchResult, rating_versions: Sequence[int | None]) -> list[Pair]:
    """``(a, b, weight, outcome_a)`` for every pair of teams of ``result`` (``a.team < b.team``); ``[]`` for a
    one-team layout (``pairs_of``)."""
    return pairs_of(team_keys(result, rating_versions), {int(t.team): float(t.rank) for t in result.teams})


def match_score(result: MatchResult, rating_versions: Sequence[int | None]) -> tuple[TeamKey, float] | None:
    """``(team, team score)`` of a one-team result, else None (``score_of``)."""
    return score_of(team_keys(result, rating_versions), {int(t.team): float(t.score) for t in result.teams})


def coefficients(a: TeamKey, b: TeamKey) -> dict[EntityKey, float]:
    """Net coefficient of every entity in ``S_a - S_b`` (team strength = mean of its members): non-zero
    entries only. An entity that sits on both teams in the same share cancels."""
    out: dict[EntityKey, float] = {}
    for key, sign in ((a, 1.0), (b, -1.0)):
        share = sign / len(key.members)
        for entity, _ in key.members:
            out[entity] = out.get(entity, 0.0) + share
    return {e: c for e, c in out.items() if abs(c) > _COEF_EPS}


def _pair_order(item) -> tuple:
    (a, b), _ = item
    return a.team, a.members, b.team, b.members


@dataclass
class _Cell:
    pairs: dict[tuple[TeamKey, TeamKey], PairStat] = field(default_factory=dict)
    scores: dict[TeamKey, ScoreStat] = field(default_factory=dict)


class ComparisonTable:
    """Sufficient statistics of team pairs per ``(layout, source)``, plus score sums for one-team layouts
    (module docstring). Every method holds the table's lock; ``version`` counts mutations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cells: dict[tuple[str, str], _Cell] = {}
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _cell(self, layout: str, source: str) -> _Cell:
        return self._cells.setdefault((str(layout), str(source)), _Cell())

    def add_result(self, result: MatchResult, rating_versions: Sequence[int | None]) -> None:
        """Add one match (pairs, or the team score of a one-team layout) under ``(result.layout, result.source)``."""
        pairs = match_pairs(result, rating_versions)
        with self._lock:
            if pairs:
                for a, b, w, y in pairs:
                    self._add_pair(result.layout, result.source, a, b, PairStat(1, w, w * y))
            else:
                score = match_score(result, rating_versions)
                if score is None:
                    return
                key, value = score
                self._add_score(result.layout, result.source, key, ScoreStat(1, value, value * value))
            self._version += 1

    def _add_pair(self, layout: str, source: str, a: TeamKey, b: TeamKey, stat: PairStat) -> None:
        if a.team > b.team:      # canonical order: the lower team index first
            a, b, stat = b, a, PairStat(stat.n, stat.sw, stat.sw - stat.swy)
        if a.team == b.team:
            raise ValueError(f"a team pair needs two different team indices, got {a.team} twice")
        cell = self._cell(layout, source)
        cell.pairs.setdefault((a, b), PairStat()).add(stat)

    def _add_score(self, layout: str, source: str, key: TeamKey, stat: ScoreStat) -> None:
        self._cell(layout, source).scores.setdefault(key, ScoreStat()).add(stat)

    def add(self, layout: str, source: str, a: TeamKey, b: TeamKey, stat: PairStat) -> None:
        """Add aggregated pair statistics (a journal ``counts`` row); the pair is stored in canonical order."""
        with self._lock:
            self._add_pair(layout, source, a, b, stat)
            self._version += 1

    def add_score(self, layout: str, source: str, key: TeamKey, stat: ScoreStat) -> None:
        """Add aggregated score statistics of a one-team composition."""
        with self._lock:
            self._add_score(layout, source, key, stat)
            self._version += 1

    def merge(self, other: ComparisonTable) -> None:
        """Add every statistic of ``other``."""
        snapshot = other.copy()
        with self._lock:
            for (layout, source), cell in snapshot._cells.items():
                for (a, b), stat in cell.pairs.items():
                    self._add_pair(layout, source, a, b, stat)
                for key, stat in cell.scores.items():
                    self._add_score(layout, source, key, stat)
            self._version += 1

    def copy(self) -> ComparisonTable:
        """An independent copy (taken under the lock)."""
        out = ComparisonTable()
        with self._lock:
            for key, cell in self._cells.items():
                out._cells[key] = _Cell(
                    pairs={k: PairStat(s.n, s.sw, s.swy) for k, s in cell.pairs.items()},
                    scores={k: ScoreStat(s.n, s.total, s.total_sq) for k, s in cell.scores.items()},
                )
            out._version = self._version
        return out

    def layouts(self) -> list[str]:
        with self._lock:
            return sorted({layout for layout, _ in self._cells})

    def sources(self, layout: str) -> list[str]:
        with self._lock:
            return sorted(source for lay, source in self._cells if lay == layout)

    def pairs(self, layout: str, sources: Collection[str]) -> list[tuple[TeamKey, TeamKey, PairStat]]:
        """Pair statistics of ``layout`` summed over ``sources`` (sorted by key)."""
        merged: dict[tuple[TeamKey, TeamKey], PairStat] = {}
        with self._lock:
            for source in sources:
                cell = self._cells.get((layout, source))
                if cell is None:
                    continue
                for key, stat in cell.pairs.items():
                    merged.setdefault(key, PairStat()).add(stat)
        return [(a, b, stat) for (a, b), stat in sorted(merged.items(), key=_pair_order)]

    def scores(self, layout: str, sources: Collection[str]) -> list[tuple[TeamKey, ScoreStat]]:
        """Score statistics of ``layout`` summed over ``sources`` (sorted by key)."""
        merged: dict[TeamKey, ScoreStat] = {}
        with self._lock:
            for source in sources:
                cell = self._cells.get((layout, source))
                if cell is None:
                    continue
                for key, stat in cell.scores.items():
                    merged.setdefault(key, ScoreStat()).add(stat)
        return sorted(merged.items(), key=lambda kv: (kv[0].team, kv[0].members))

    def entity_games(self, layout: str, sources: Collection[str]) -> dict[EntityKey, int]:
        """Matches per entity with a non-zero net coefficient (pairs) or a seat (scores); self-play excluded."""
        games: dict[EntityKey, int] = {}
        for a, b, stat in self.pairs(layout, sources):
            for entity in coefficients(a, b):
                games[entity] = games.get(entity, 0) + stat.n
        for key, stat in self.scores(layout, sources):
            for entity in set(key.entities()):
                games[entity] = games.get(entity, 0) + stat.n
        return games

    def as_dict(self) -> dict:
        """Plain nested form for tests and debugging: ``{(layout, source): {"pairs": {(a, b): (n, sw, swy)},
        "scores": {key: (n, total, total_sq)}}}``."""
        with self._lock:
            return {key: {"pairs": {(a.encode(), b.encode()): (s.n, s.sw, s.swy) for (a, b), s in cell.pairs.items()},
                          "scores": {k.encode(): (s.n, s.total, s.total_sq) for k, s in cell.scores.items()}}
                    for key, cell in sorted(self._cells.items())}
```

Notes: `pairs_of` / `score_of` are the single rule shared by results (`match_pairs`, `add_result`) and journal rows (`matchlog` folds detailed matches with them), so a table built from results and one built from the journal agree by construction. `pairs()` sorts with a key on `(team, members)` tuples (tuples of strings compare in C), not on the dataclasses (dataclass comparisons dominated the refit time at 20 000 rows in the profile).

- [ ] **Step 5: Write `src/colosseum/arena/matchlog.py`**

```python
"""The match log: a bounded SQLite journal of finished matches (SP4 spec block 1).

``runs/<name>/matches.db`` holds every training, tournament and pulse match of a run. Tables:

- ``meta`` (schema version; the running size estimate and count of the detailed training matches), ``entities``
  (numeric ids of agents, networks, roles, sources, layouts, rating entities, opponent categories and
  tournaments), ``runs`` (this run and the runs it was resumed from);
- ``matches`` / ``seats`` / ``teams``: detailed matches (``uid`` unique; per seat the weights versions at the
  episode's start and end and the rating version, see ``colosseum.arena.entities``);
- ``counts``: compacted matches as team-pair sufficient statistics (``n``, ``sum w``, ``sum w*y``) and, for
  one-team layouts, team-score sums (``n``, ``sum``, ``sum of squares``) per ``(layout, source, team keys)``.
  ``comparison_table`` reads detailed matches and counts into one ``ComparisonTable``, the same before and after
  a compaction (up to float rounding of the sums).

Bounded by construction. Tournament and pulse matches stay detailed. Detailed training matches are kept within
``window_bytes``: every match row records its estimated size (``estimated_size``), ``meta`` keeps the running sum,
and ``compact`` folds the oldest into ``counts`` in batches of at most ``max_batch`` matches (one transaction each),
then runs ``PRAGMA incremental_vacuum`` and ``wal_checkpoint(TRUNCATE)``. Training matches of evicted entities are
folded first. When the journal's disk has less than ``min_free_disk_mb`` free, new training matches go straight to
``counts`` (``degraded``; a WARNING on the switch, an INFO when detail recording resumes); a tournament writes as
usual.

SQLite rules: WAL; ``auto_vacuum = INCREMENTAL`` (set before the first table); ``synchronous = NORMAL``;
``journal_size_limit``; ``busy_timeout``; one connection per thread or process; a writer's transaction is one
batch, so two writers (training and a tournament) can share the file. ``MatchLogReader`` opens read-only and every
query is its own transaction. A file that is not an SQLite database is renamed (``matches.db.corrupt-<k>``) and a
new journal starts (ERROR); a journal of another schema version is a ``MatchLogError``.

Order is by row id (insertion order), never by the informational ``ts``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from colosseum.arena.comparisons import (
    ComparisonTable,
    PairStat,
    ScoreStat,
    TeamKey,
    pairs_of,
    score_of,
    team_keys,
)
from colosseum.arena.entities import EntityKey, entity_key, rating_version
from colosseum.core.types import MATCH_SOURCES, MatchResult

logger = logging.getLogger(__name__)

MATCH_LOG_FILE = "matches.db"
SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 10_000
JOURNAL_SIZE_LIMIT = 64 * 1024 * 1024
READ_BATCH = 10_000
MAX_BATCH = 10_000
# Estimated bytes of one detailed match: the rows' payload (the uid twice: its UNIQUE index repeats it) times an
# overhead factor for B-tree pages and the indexes. Calibrated on 1v1, 2v2 and 4-team FFA matches: the estimate
# is within 7 % of the pages SQLite really uses (T1.2).
MATCH_ROW_BYTES = 48
SEAT_ROW_BYTES = 40
TEAM_ROW_BYTES = 20
INDEX_OVERHEAD = 1.2
_PAIR, _SCORE = 0, 1
_WORKER_RE = re.compile(r"^w(\d+)_e")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, "
    "UNIQUE (kind, name))",
    "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, name TEXT NOT NULL, "
    "parent_run_id TEXT, game TEXT NOT NULL, config_hash TEXT NOT NULL, checkpoint_interval INTEGER NOT NULL, "
    "created REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS matches (id INTEGER PRIMARY KEY, uid TEXT NOT NULL UNIQUE, run INTEGER, "
    "source INTEGER NOT NULL, layout INTEGER NOT NULL, outcome_kind TEXT NOT NULL, length INTEGER NOT NULL, "
    "seed INTEGER, env_steps INTEGER, block TEXT, tournament INTEGER, worker INTEGER, ts REAL NOT NULL, "
    "replay TEXT, size INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS matches_by_source ON matches (source, id)",
    "CREATE INDEX IF NOT EXISTS matches_by_tournament ON matches (tournament, block) WHERE tournament IS NOT NULL",
    "CREATE TABLE IF NOT EXISTS seats (match INTEGER NOT NULL, seat INTEGER NOT NULL, team INTEGER NOT NULL, "
    "role INTEGER NOT NULL, agent INTEGER NOT NULL, network INTEGER NOT NULL, entity INTEGER NOT NULL, "
    "version_start INTEGER, version_end INTEGER, rating_version INTEGER, reward REAL NOT NULL, "
    "eliminated_step INTEGER, category INTEGER NOT NULL, PRIMARY KEY (match, seat)) WITHOUT ROWID",
    "CREATE INDEX IF NOT EXISTS seats_by_agent ON seats (agent, rating_version, match)",
    "CREATE INDEX IF NOT EXISTS seats_by_entity ON seats (entity, match)",
    "CREATE TABLE IF NOT EXISTS teams (match INTEGER NOT NULL, team INTEGER NOT NULL, rank REAL NOT NULL, "
    "score REAL NOT NULL, PRIMARY KEY (match, team)) WITHOUT ROWID",
    "CREATE TABLE IF NOT EXISTS counts (id INTEGER PRIMARY KEY, kind INTEGER NOT NULL, layout INTEGER NOT NULL, "
    "source INTEGER NOT NULL, key_a TEXT NOT NULL, key_b TEXT NOT NULL, n INTEGER NOT NULL, s1 REAL NOT NULL, "
    "s2 REAL NOT NULL, UNIQUE (kind, layout, source, key_a, key_b))",
)
_UPSERT_COUNT = (
    "INSERT INTO counts (kind, layout, source, key_a, key_b, n, s1, s2) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (kind, layout, source, key_a, key_b) DO UPDATE SET n = n + excluded.n, s1 = s1 + excluded.s1, "
    "s2 = s2 + excluded.s2"
)
_MATCH_COLUMNS = ("uid, run, source, layout, outcome_kind, length, seed, env_steps, block, tournament, worker, "
                  "ts, replay, size")
_SEAT_COLUMNS = ("seat, team, role, agent, network, entity, version_start, version_end, rating_version, reward, "
                 "eliminated_step, category")


class MatchLogError(Exception):
    """A journal this code cannot use (another schema version, not a match log)."""


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    name: str
    parent_run_id: str | None
    game: str
    config_hash: str
    checkpoint_interval: int


@dataclass
class AppendStats:
    detailed: int = 0
    counted: int = 0            # folded straight into counts (disk guard)
    duplicates: int = 0         # uid already in the journal: ignored
    accepted: list[bool] = field(default_factory=list)   # per result: journaled (detailed or counted)
    degraded: bool = False


@dataclass
class CompactStats:
    folded: int = 0             # detailed training matches folded into counts
    evicted_folded: int = 0     # of which because they had an evicted entity
    batches: int = 0
    interrupted: bool = False


@dataclass
class ImportStats:
    counts_rows: int = 0
    folded: int = 0             # the parent's detailed training matches, folded into counts
    copied: int = 0             # the parent's tournament and pulse matches, copied in detail


@dataclass(frozen=True)
class LogSizes:
    file_bytes: int
    wal_bytes: int
    detail_bytes: int           # estimate of the detailed training matches
    detail_matches: int         # detailed training matches
    other_matches: int          # detailed tournament and pulse matches
    counts_rows: int
    free_disk_bytes: int
    degraded: bool


@dataclass(frozen=True)
class SeatRow:
    seat: int
    team: int
    role: str
    agent_id: str
    network_id: str
    entity: EntityKey
    version_start: int | None
    version_end: int | None
    rating_version: int | None
    reward: float
    eliminated_step: int | None
    category: str


@dataclass(frozen=True)
class MatchRow:
    id: int
    uid: str
    run_id: str | None
    source: str
    layout: str
    outcome_kind: str
    length: int
    seed: int | None
    env_steps: int | None
    block: str | None
    tournament: str | None
    worker: int | None
    ts: float
    seats: tuple[SeatRow, ...]
    teams: tuple[tuple[int, float, float], ...]     # (team, rank, score)


def default_disk_free(directory: Path) -> int:
    """Free bytes on the disk of ``directory``."""
    return int(shutil.disk_usage(directory).free)


def estimated_size(result: MatchResult) -> int:
    """Estimated bytes of a detailed match (module docstring)."""
    payload = MATCH_ROW_BYTES + 2 * len(result.uid) + len(result.block or "")
    payload += SEAT_ROW_BYTES * len(result.seats) + TEAM_ROW_BYTES * len(result.teams)
    return int(payload * INDEX_OVERHEAD)


def _connect(path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None,
                               timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def _is_corrupt(error: sqlite3.DatabaseError) -> bool:
    """A file that is not (or no longer) an SQLite database; other errors (locked, I/O, permissions) are not."""
    text = str(error).lower()
    return "not a database" in text or "malformed" in text


def _worker_of(match_id: str) -> int | None:
    match = _WORKER_RE.match(match_id or "")
    return None if match is None else int(match.group(1))


def _add_detail(table: ComparisonTable, layout: str, source: str, keys: Mapping[int, TeamKey],
                ranks: Mapping[int, float], scores: Mapping[int, float]) -> None:
    pairs = pairs_of(keys, ranks)
    if pairs:
        for a, b, w, y in pairs:
            table.add(layout, source, a, b, PairStat(1, w, w * y))
        return
    score = score_of(keys, scores)
    if score is not None:
        key, value = score
        table.add_score(layout, source, key, ScoreStat(1, value, value * value))


def _fold_into(acc: dict, layout: str, source: str, keys: Mapping[int, TeamKey], ranks: Mapping[int, float],
               scores: Mapping[int, float]) -> None:
    """Add one match to ``acc`` = ``{(kind, layout, source, TeamKey a, TeamKey b or None): [n, s1, s2]}``."""
    pairs = pairs_of(keys, ranks)
    if pairs:
        for a, b, w, y in pairs:
            cell = acc.setdefault((_PAIR, layout, source, a, b), [0, 0.0, 0.0])
            cell[0] += 1
            cell[1] += w
            cell[2] += w * y
        return
    score = score_of(keys, scores)
    if score is not None:
        key, value = score
        cell = acc.setdefault((_SCORE, layout, source, key, None), [0, 0.0, 0.0])
        cell[0] += 1
        cell[1] += value
        cell[2] += value * value


class _Store:
    """Reading code shared by the writer and the reader (one connection, an id <-> name cache)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ids: dict[tuple[str, str], int] = {}
        self._names: dict[int, str] = {}

    def _lookup(self, kind: str, name: str) -> int | None:
        key = (kind, name)
        found = self._ids.get(key)
        if found is not None:
            return found
        row = self._conn.execute("SELECT id FROM entities WHERE kind = ? AND name = ?", key).fetchone()
        if row is None:
            return None
        self._ids[key] = int(row[0])
        self._names[int(row[0])] = name
        return int(row[0])

    def _name(self, entity_id: int | None) -> str | None:
        if entity_id is None:
            return None
        name = self._names.get(entity_id)
        if name is None:
            row = self._conn.execute("SELECT name FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                raise MatchLogError(f"entity id {entity_id} is not in the entities table")
            name = self._names[entity_id] = str(row[0])
        return name

    def _meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def _decode_key(self, text: str) -> TeamKey:
        """A ``counts`` key ``"<team>|<entity id>:<role id>,..."`` back to a ``TeamKey``."""
        team, _, members = text.partition("|")
        pairs = []
        for item in members.split(","):
            entity, _, role = item.partition(":")
            pairs.append((self._name(int(entity)), self._name(int(role))))
        return TeamKey.of(int(team), pairs)

    def _match_filter(self, sources: Collection[str], tournament: str | None) -> tuple[str, list]:
        ids = [i for s in sources if (i := self._lookup("source", s)) is not None]
        if not ids:
            return "0", []
        clause = f"source IN ({','.join('?' * len(ids))})"
        args: list = list(ids)
        if tournament is None:
            clause += " AND tournament IS NULL"
        else:
            tid = self._lookup("tournament", tournament)
            if tid is None:
                return "0", []
            clause += " AND tournament = ?"
            args.append(tid)
        return clause, args

    def _iter_detail(self, where: str, args: Sequence) -> Iterator[tuple[int, str, str, dict[int, TeamKey],
                                                                         dict[int, float], dict[int, float]]]:
        """``(match id, layout, source, team keys, ranks, scores)`` of the detailed matches that ``where`` (on
        ``matches``) selects, in id order, read in batches of ``READ_BATCH`` (each query is one transaction)."""
        last = 0
        while True:
            heads = self._conn.execute(
                f"SELECT id, layout, source FROM matches WHERE id > ? AND ({where}) ORDER BY id LIMIT ?",
                (last, *args, READ_BATCH)).fetchall()
            if not heads:
                return
            lo, hi = heads[0][0], heads[-1][0]
            members: dict[int, dict[int, list[tuple[str, str]]]] = {}
            for match, team, entity, role in self._conn.execute(
                    "SELECT match, team, entity, role FROM seats WHERE match BETWEEN ? AND ? ORDER BY match, seat",
                    (lo, hi)):
                members.setdefault(match, {}).setdefault(team, []).append((self._name(entity), self._name(role)))
            ranks: dict[int, dict[int, float]] = {}
            scores: dict[int, dict[int, float]] = {}
            for match, team, rank, score in self._conn.execute(
                    "SELECT match, team, rank, score FROM teams WHERE match BETWEEN ? AND ?", (lo, hi)):
                ranks.setdefault(match, {})[team] = float(rank)
                scores.setdefault(match, {})[team] = float(score)
            for match, layout, source in heads:
                keys = {t: TeamKey.of(t, m) for t, m in sorted(members.get(match, {}).items())}
                yield match, self._name(layout), self._name(source), keys, ranks.get(match, {}), scores.get(match, {})
            last = hi

    def _table(self, sources: Collection[str], tournament: str | None) -> ComparisonTable:
        table = ComparisonTable()
        src_ids = [i for s in sources if (i := self._lookup("source", s)) is not None]
        if tournament is None and src_ids:
            for kind, layout, source, key_a, key_b, n, s1, s2 in self._conn.execute(
                    f"SELECT kind, layout, source, key_a, key_b, n, s1, s2 FROM counts "
                    f"WHERE source IN ({','.join('?' * len(src_ids))}) ORDER BY id", src_ids):
                if kind == _PAIR:
                    table.add(self._name(layout), self._name(source), self._decode_key(key_a),
                              self._decode_key(key_b), PairStat(int(n), float(s1), float(s2)))
                else:
                    table.add_score(self._name(layout), self._name(source), self._decode_key(key_a),
                                    ScoreStat(int(n), float(s1), float(s2)))
        where, args = self._match_filter(sources, tournament)
        for _, layout, source, keys, ranks, scores in self._iter_detail(where, args):
            _add_detail(table, layout, source, keys, ranks, scores)
        return table

    def _sizes(self, path: Path, free: int, degraded: bool) -> LogSizes:
        wal = Path(f"{path}-wal")
        total = int(self._conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        detail_matches = int(self._meta("detail_train_matches", "0"))
        return LogSizes(
            file_bytes=path.stat().st_size if path.exists() else 0,
            wal_bytes=wal.stat().st_size if wal.exists() else 0,
            detail_bytes=int(self._meta("detail_train_bytes", "0")),
            detail_matches=detail_matches,
            other_matches=total - detail_matches,
            counts_rows=int(self._conn.execute("SELECT COUNT(*) FROM counts").fetchone()[0]),
            free_disk_bytes=int(free),
            degraded=degraded,
        )


class MatchLog(_Store):
    """A writer connection to a run's journal (module docstring).

    ``run``: the training run this connection writes for (registered in ``runs``; its ``checkpoint_interval`` is
    the width of the version buckets); None for a writer that is not a training run (a tournament): its matches
    have no run, and their rating versions come from the network ids only. ``disk_free_fn(directory) -> bytes``
    is the disk guard's probe (tests pass their own).
    """

    def __init__(self, path: str | Path, run: RunInfo | None, *,
                 disk_free_fn: Callable[[Path], int] = default_disk_free, min_free_disk_mb: float = 2048.0) -> None:
        self.path = Path(path)
        self.run = run
        self._disk_free_fn = disk_free_fn
        self._min_free = int(float(min_free_disk_mb) * 1024 * 1024)
        self._degraded = False
        self._interval = int(run.checkpoint_interval) if run is not None else 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(self._open())
        self._run_row = self._register_run(run) if run is not None else None

    # ---- opening ---------------------------------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        try:
            return self._open_once()
        except sqlite3.DatabaseError as e:
            if not _is_corrupt(e):
                raise
            broken = self._quarantine()
            logger.error(f"Match log {self.path} is not a readable SQLite database ({e}); moved it to {broken} and "
                         f"started a new journal")
            return self._open_once()

    def _open_once(self) -> sqlite3.Connection:
        new = not self.path.exists() or self.path.stat().st_size == 0
        conn = _connect(self.path, readonly=False)
        try:
            if new:
                conn.execute("PRAGMA auto_vacuum = INCREMENTAL")    # only takes effect before the first table
            conn.execute("PRAGMA journal_mode = WAL").fetchall()
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(f"PRAGMA journal_size_limit = {JOURNAL_SIZE_LIMIT}").fetchall()
            self._create_schema(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _quarantine(self) -> Path:
        k = 1
        while Path(f"{self.path}.corrupt-{k}").exists():
            k += 1
        target = Path(f"{self.path}.corrupt-{k}")
        os.replace(self.path, target)
        for suffix in ("-wal", "-shm"):
            side = Path(f"{self.path}{suffix}")
            if side.exists():
                os.replace(side, Path(f"{target}{suffix}"))
        return target

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _SCHEMA:
                conn.execute(statement)
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                                 [("schema_version", str(SCHEMA_VERSION)), ("detail_train_bytes", "0"),
                                  ("detail_train_matches", "0")])
            elif int(row[0]) != SCHEMA_VERSION:
                raise MatchLogError(f"match log {self.path} has schema version {row[0]}; this Colosseum reads "
                                    f"version {SCHEMA_VERSION} (move the file away to start a new journal)")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def _register_run(self, run: RunInfo) -> int:
        def register() -> int:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                "created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run.run_id, run.name, run.parent_run_id, run.game, run.config_hash, int(run.checkpoint_interval),
                 time.time()))
            return int(self._conn.execute("SELECT id FROM runs WHERE run_id = ?", (run.run_id,)).fetchone()[0])
        return self._transaction(register)

    @classmethod
    def open_readonly(cls, path: str | Path) -> MatchLogReader:
        return MatchLogReader(path)

    # ---- writing helpers ---------------------------------------------------------------------------------------

    def _transaction(self, body: Callable[[], object]):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            out = body()
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            self._ids.clear()          # ids inserted by the rolled-back transaction are gone
            self._names.clear()
            raise
        return out

    def _id(self, kind: str, name: str) -> int:
        found = self._lookup(kind, name)
        if found is not None:
            return found
        self._conn.execute("INSERT OR IGNORE INTO entities (kind, name) VALUES (?, ?)", (kind, name))
        found = self._lookup(kind, name)
        assert found is not None
        return found

    def _encode_key(self, key: TeamKey) -> str:
        return f"{key.team}|" + ",".join(f"{self._id('entity', e)}:{self._id('role', r)}" for e, r in key.members)

    def _write_counts(self, acc: Mapping) -> None:
        """Upsert ``{(kind, layout, source, TeamKey a, TeamKey b or None): [n, s1, s2]}`` into ``counts``."""
        self._conn.executemany(_UPSERT_COUNT, [
            (kind, self._id("layout", layout), self._id("source", source), self._encode_key(a),
             "" if b is None else self._encode_key(b), n, s1, s2)
            for (kind, layout, source, a, b), (n, s1, s2) in acc.items()])

    def _bump_detail(self, nbytes: int, nmatches: int) -> None:
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key = 'detail_train_bytes'",
                           (int(nbytes),))
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + ? WHERE key = 'detail_train_matches'",
                           (int(nmatches),))

    def rating_versions(self, result: MatchResult) -> list[int | None]:
        """The rating version of every seat of ``result`` (``entities.rating_version`` with this run's
        ``checkpoint.interval``)."""
        return [rating_version(s.network_id, s.version_start, self._interval) for s in result.seats]

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _check_disk(self) -> bool:
        free = self._disk_free_fn(self.path.parent)
        low = free < self._min_free
        if low and not self._degraded:
            logger.warning(f"Match log: {free / 2**20:.0f} MB free on the journal's disk, below "
                           f"match_log.min_free_disk_mb ({self._min_free / 2**20:.0f}); new training matches go to "
                           f"the compacted counts only until space is back")
        elif not low and self._degraded:
            logger.info(f"Match log: {free / 2**20:.0f} MB free again; detailed recording of training matches "
                        f"resumes")
        self._degraded = low
        return low

    # ---- appending ---------------------------------------------------------------------------------------------

    def append(self, results: Sequence[MatchResult], *, tournament: str | None = None) -> AppendStats:
        """Journal ``results`` in one transaction (module docstring); ``tournament``: the id of the tournament
        they belong to. A uid already present is a duplicate (ignored, counted). ValueError (nothing written)
        for an unknown source or an empty uid."""
        stats = AppendStats()
        if not results:
            return stats
        for result in results:
            if result.source not in MATCH_SOURCES:
                raise ValueError(f"match {result.uid or result.match_id!r}: source {result.source!r} is not one of "
                                 f"{MATCH_SOURCES}")
            if not result.uid:
                raise ValueError(f"match {result.match_id!r} has no uid")
        degraded = tournament is None and self._check_disk()
        stats.degraded = self._degraded

        def body() -> None:
            tid = self._id("tournament", tournament) if tournament is not None else None
            train_bytes = train_matches = 0
            for result in results:
                versions = self.rating_versions(result)
                is_train = result.source == "train" and tournament is None
                if degraded and is_train:
                    acc: dict = {}
                    keys = team_keys(result, versions)
                    _fold_into(acc, result.layout, result.source, keys, {t.team: t.rank for t in result.teams},
                               {t.team: t.score for t in result.teams})
                    self._write_counts(acc)
                    stats.counted += 1
                    stats.accepted.append(True)
                    continue
                size = estimated_size(result)
                cur = self._conn.execute(
                    f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    f"NULL, ?)",
                    (result.uid, self._run_row, self._id("source", result.source), self._id("layout", result.layout),
                     result.outcome_kind, int(result.episode_length), result.seed, result.env_steps, result.block,
                     tid, _worker_of(result.match_id), time.time(), size))
                if cur.rowcount == 0:
                    stats.duplicates += 1
                    stats.accepted.append(False)
                    continue
                match = int(cur.lastrowid)
                self._conn.executemany(
                    f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(match, s.seat, s.team, self._id("role", s.role), self._id("agent", s.agent_id),
                      self._id("network", s.network_id), self._id("entity", entity_key(s.agent_id, v)),
                      s.version_start, s.version_end, v, float(s.reward), s.eliminated_step,
                      self._id("category", s.source)) for s, v in zip(result.seats, versions, strict=True)])
                self._conn.executemany("INSERT INTO teams (match, team, rank, score) VALUES (?, ?, ?, ?)",
                                       [(match, t.team, float(t.rank), float(t.score)) for t in result.teams])
                if is_train:
                    train_bytes += size
                    train_matches += 1
                stats.detailed += 1
                stats.accepted.append(True)
            if train_matches:
                self._bump_detail(train_bytes, train_matches)

        self._transaction(body)
        return stats

    # ---- compaction --------------------------------------------------------------------------------------------

    def _fold_matches(self, ids: Sequence[int]) -> None:
        """Fold the detailed matches ``ids`` into counts and delete them, in one transaction."""
        marks = ",".join("?" * len(ids))

        def body() -> None:
            acc: dict = {}
            for _, layout, source, keys, ranks, scores in self._iter_detail(f"id IN ({marks})", ids):
                _fold_into(acc, layout, source, keys, ranks, scores)
            self._write_counts(acc)
            train = self._lookup("source", "train")
            nbytes, nmatches = self._conn.execute(
                f"SELECT COALESCE(SUM(size), 0), COUNT(*) FROM matches WHERE id IN ({marks}) AND source = ? "
                f"AND tournament IS NULL", (*ids, train)).fetchone()
            for table, column in (("seats", "match"), ("teams", "match"), ("matches", "id")):
                self._conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", ids)
            self._bump_detail(-int(nbytes), -int(nmatches))

        self._transaction(body)

    def compact(self, *, window_bytes: int, max_batch: int = MAX_BATCH, evicted: Collection[EntityKey] = (),
                should_stop: Callable[[], bool] = lambda: False) -> CompactStats:
        """Fold the training matches of ``evicted`` entities, then the oldest detailed training matches beyond
        ``window_bytes``, into counts, in batches of at most ``max_batch`` (<= 10 000) matches; then give the
        space back. Stops between batches once ``should_stop()`` (``CompactStats.interrupted``)."""
        stats = CompactStats()
        train = self._lookup("source", "train")
        if train is None:
            return stats
        max_batch = max(1, min(int(max_batch), MAX_BATCH))
        evicted_ids = [i for e in sorted(set(evicted)) if (i := self._lookup("entity", e)) is not None]
        for k in range(0, len(evicted_ids), 500):
            chunk = evicted_ids[k:k + 500]
            marks = ",".join("?" * len(chunk))
            while not stats.interrupted:
                if should_stop():
                    stats.interrupted = True
                    break
                ids = [int(r[0]) for r in self._conn.execute(
                    f"SELECT DISTINCT s.match FROM seats s JOIN matches m ON m.id = s.match "
                    f"WHERE s.entity IN ({marks}) AND m.source = ? AND m.tournament IS NULL ORDER BY s.match LIMIT ?",
                    (*chunk, train, max_batch))]
                if not ids:
                    break
                self._fold_matches(ids)
                stats.folded += len(ids)
                stats.evicted_folded += len(ids)
                stats.batches += 1
        while not stats.interrupted:
            nbytes = int(self._meta("detail_train_bytes", "0"))
            nmatches = int(self._meta("detail_train_matches", "0"))
            if nbytes <= window_bytes or nmatches <= 0:
                break
            if should_stop():
                stats.interrupted = True
                break
            needed = max(1, math.ceil((nbytes - window_bytes) / (nbytes / nmatches)))
            ids = [int(r[0]) for r in self._conn.execute(
                "SELECT id FROM matches WHERE source = ? AND tournament IS NULL ORDER BY id LIMIT ?",
                (train, min(max_batch, needed)))]
            if not ids:
                break
            self._fold_matches(ids)
            stats.folded += len(ids)
            stats.batches += 1
        if stats.folded:
            self._conn.execute("PRAGMA incremental_vacuum").fetchall()    # one step per fetched row
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        return stats

    # ---- resume ------------------------------------------------------------------------------------------------

    def import_parent(self, parent_path: str | Path) -> ImportStats:
        """Run-dir resume (spec block 1): copy the parent journal's ``runs`` rows, counts, tournament and pulse
        matches; fold its detailed training matches into this journal's counts (their detail stays in the
        parent's journal). Batches of ``READ_BATCH`` per transaction. Call it once, on a new journal."""
        stats = ImportStats()
        parent = MatchLogReader(parent_path)
        try:
            runs = parent._conn.execute("SELECT run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                                        "created FROM runs ORDER BY id").fetchall()
            self._transaction(lambda: self._conn.executemany(
                "INSERT OR IGNORE INTO runs (run_id, name, parent_run_id, game, config_hash, checkpoint_interval, "
                "created) VALUES (?, ?, ?, ?, ?, ?, ?)", runs))
            last = 0
            while True:
                rows = parent._conn.execute(
                    "SELECT id, kind, layout, source, key_a, key_b, n, s1, s2 FROM counts WHERE id > ? ORDER BY id "
                    "LIMIT ?", (last, READ_BATCH)).fetchall()
                if not rows:
                    break
                acc = {(kind, parent._name(layout), parent._name(source), parent._decode_key(key_a),
                        parent._decode_key(key_b) if key_b else None): (n, s1, s2)
                       for _, kind, layout, source, key_a, key_b, n, s1, s2 in rows}
                self._transaction(lambda acc=acc: self._write_counts(acc))
                stats.counts_rows += len(rows)
                last = rows[-1][0]
            train = parent._lookup("source", "train")
            if train is not None:
                acc, pending = {}, 0
                for _, layout, source, keys, ranks, scores in parent._iter_detail(
                        "source = ? AND tournament IS NULL", (train,)):
                    _fold_into(acc, layout, source, keys, ranks, scores)
                    pending += 1
                    if pending >= READ_BATCH:
                        self._transaction(lambda acc=acc: self._write_counts(acc))
                        stats.folded += pending
                        acc, pending = {}, 0
                if pending:
                    self._transaction(lambda acc=acc: self._write_counts(acc))
                    stats.folded += pending
            stats.copied = self._copy_detail(parent, train)
        finally:
            parent.close()
        return stats

    def _copy_detail(self, parent: MatchLogReader, train: int | None) -> int:
        """Copy the parent's detailed tournament and pulse matches; returns how many were new here."""
        where = "1" if train is None else "(source != ? OR tournament IS NOT NULL)"
        args: tuple = () if train is None else (train,)
        run_ids = {r[0]: r[1] for r in parent._conn.execute("SELECT id, run_id FROM runs")}
        copied = last = 0
        while True:
            heads = parent._conn.execute(
                f"SELECT id, {_MATCH_COLUMNS} FROM matches WHERE id > ? AND {where} ORDER BY id LIMIT ?",
                (last, *args, READ_BATCH)).fetchall()
            if not heads:
                return copied
            last = heads[-1][0]

            def body(heads=heads) -> int:
                n = 0
                for (pid, uid, run, source, layout, kind, length, seed, env_steps, block, tour, worker, ts, replay,
                     size) in heads:
                    run_row = None
                    if run in run_ids:
                        row = self._conn.execute("SELECT id FROM runs WHERE run_id = ?", (run_ids[run],)).fetchone()
                        run_row = None if row is None else int(row[0])
                    tid = None if tour is None else self._id("tournament", parent._name(tour))
                    cur = self._conn.execute(
                        f"INSERT OR IGNORE INTO matches ({_MATCH_COLUMNS}) "
                        f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (uid, run_row, self._id("source", parent._name(source)),
                         self._id("layout", parent._name(layout)), kind, length, seed, env_steps, block, tid, worker,
                         ts, replay, size))
                    if cur.rowcount == 0:
                        continue
                    match = int(cur.lastrowid)
                    seats = parent._conn.execute(f"SELECT {_SEAT_COLUMNS} FROM seats WHERE match = ?",
                                                 (pid,)).fetchall()
                    self._conn.executemany(
                        f"INSERT INTO seats (match, {_SEAT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [(match, seat, team, self._id("role", parent._name(role)),
                          self._id("agent", parent._name(agent)), self._id("network", parent._name(network)),
                          self._id("entity", parent._name(entity)), vs, ve, rv, reward, elim,
                          self._id("category", parent._name(category)))
                         for seat, team, role, agent, network, entity, vs, ve, rv, reward, elim, category in seats])
                    self._conn.executemany(
                        "INSERT INTO teams (match, team, rank, score) VALUES (?, ?, ?, ?)",
                        [(match, *row) for row in parent._conn.execute(
                            "SELECT team, rank, score FROM teams WHERE match = ?", (pid,))])
                    n += 1
                return n

            copied += self._transaction(body)

    # ---- reading -----------------------------------------------------------------------------------------------

    def comparison_table(self, sources: Collection[str] = ("train", "pulse"),
                         tournament: str | None = None) -> ComparisonTable:
        """Detailed matches and counts of ``sources`` as one table; ``tournament``: only that tournament's
        detailed matches (counts never hold tournament matches); None: only matches outside tournaments."""
        return self._table(sources, tournament)

    def sizes(self) -> LogSizes:
        return self._sizes(self.path, self._disk_free_fn(self.path.parent), self._degraded)

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass


class MatchLogReader(_Store):
    """Read-only access (report, dashboard, tournament, SP5's hub). Every query is its own transaction; a missing
    file reads as an empty journal. The connection may be used from another thread than the one that opened it
    (one thread at a time)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            for statement in _SCHEMA:
                conn.execute(statement)
            super().__init__(conn)
            return
        super().__init__(_connect(self.path, readonly=True))
        try:
            version = self._meta("schema_version")
        except sqlite3.DatabaseError as e:
            self._conn.close()
            raise MatchLogError(f"{self.path} is not a match log ({e})") from e
        if version is None or int(version) != SCHEMA_VERSION:
            self._conn.close()
            raise MatchLogError(f"{self.path} is a match log of schema version {version}, not {SCHEMA_VERSION}")

    def comparison_table(self, sources: Collection[str] = ("train", "pulse"),
                         tournament: str | None = None) -> ComparisonTable:
        """As ``MatchLog.comparison_table``."""
        return self._table(sources, tournament)

    def blocks(self, tournament: str) -> set[str]:
        """Block ids of ``tournament`` in the journal (blocks are written whole, one transaction each)."""
        tid = self._lookup("tournament", tournament)
        if tid is None:
            return set()
        return {str(r[0]) for r in self._conn.execute(
            "SELECT DISTINCT block FROM matches WHERE tournament = ? AND block IS NOT NULL", (tid,))}

    def runs(self) -> list[RunInfo]:
        """The ``runs`` rows in insertion order (this run first, then the imported parents)."""
        return [RunInfo(run_id=r[0], name=r[1], parent_run_id=r[2], game=r[3], config_hash=r[4],
                        checkpoint_interval=int(r[5]))
                for r in self._conn.execute("SELECT run_id, name, parent_run_id, game, config_hash, "
                                            "checkpoint_interval FROM runs ORDER BY id")]

    def sizes(self) -> LogSizes:
        free = default_disk_free(self.path.parent) if self.path.parent.exists() else 0
        return self._sizes(self.path, free, False)

    def matches(self, *, agent: str | None = None, version: int | None = None, layout: str | None = None,
                source: str | None = None, tournament: str | None = None, outcome: str | None = None,
                env_steps: tuple[int, int] | None = None, limit: int = 1000) -> list[MatchRow]:
        """Detailed matches, newest first. ``agent`` (with ``version``: a rating version) selects matches with such
        a seat; ``outcome`` (``win`` / ``draw`` / ``loss``; needs ``agent``) is that agent's team result against the
        best other team; ``env_steps`` is an inclusive range."""
        if outcome is not None and agent is None:
            raise ValueError("an outcome filter needs an agent")
        clauses: list[str] = []
        args: list = []
        for kind, name in (("layout", layout), ("source", source), ("tournament", tournament)):
            if name is not None:
                found = self._lookup(kind, name)
                if found is None:
                    return []
                clauses.append(f"m.{kind} = ?")
                args.append(found)
        if env_steps is not None:
            clauses.append("m.env_steps BETWEEN ? AND ?")
            args.extend(int(x) for x in env_steps)
        if agent is not None:
            agent_id = self._lookup("agent", agent)
            if agent_id is None:
                return []
            sub = "SELECT match FROM seats WHERE agent = ?"
            args.append(agent_id)
            if version is not None:
                sub += " AND rating_version = ?"
                args.append(int(version))
            clauses.append(f"m.id IN ({sub})")
        where = " AND ".join(clauses) or "1"
        out: list[MatchRow] = []
        last: int | None = None
        while len(out) < limit:
            page = self._conn.execute(
                f"SELECT m.id, m.uid, r.run_id, m.source, m.layout, m.outcome_kind, m.length, m.seed, m.env_steps, "
                f"m.block, m.tournament, m.worker, m.ts FROM matches m LEFT JOIN runs r ON r.id = m.run "
                f"WHERE {where}{'' if last is None else ' AND m.id < ?'} ORDER BY m.id DESC LIMIT ?",
                (*args, *(() if last is None else (last,)), max(limit, 100))).fetchall()
            if not page:
                break
            last = page[-1][0]
            for row in page:
                match = self._row(row)
                if outcome is not None and _outcome_for(match, agent) != outcome:
                    continue
                out.append(match)
                if len(out) >= limit:
                    break
        return out

    def _row(self, row) -> MatchRow:
        mid = row[0]
        seats = tuple(SeatRow(
            seat=s[0], team=s[1], role=self._name(s[2]), agent_id=self._name(s[3]), network_id=self._name(s[4]),
            entity=self._name(s[5]), version_start=s[6], version_end=s[7], rating_version=s[8], reward=float(s[9]),
            eliminated_step=s[10], category=self._name(s[11]))
            for s in self._conn.execute(f"SELECT {_SEAT_COLUMNS} FROM seats WHERE match = ? ORDER BY seat", (mid,)))
        teams = tuple((int(t[0]), float(t[1]), float(t[2])) for t in self._conn.execute(
            "SELECT team, rank, score FROM teams WHERE match = ? ORDER BY team", (mid,)))
        return MatchRow(id=mid, uid=row[1], run_id=row[2], source=self._name(row[3]), layout=self._name(row[4]),
                        outcome_kind=row[5], length=int(row[6]), seed=row[7], env_steps=row[8], block=row[9],
                        tournament=self._name(row[10]), worker=row[11], ts=float(row[12]), seats=seats, teams=teams)

    def close(self) -> None:
        self._conn.close()


def _outcome_for(match: MatchRow, agent: str) -> str | None:
    ranks = {team: rank for team, rank, _ in match.teams}
    mine = sorted({s.team for s in match.seats if s.agent_id == agent})
    if not mine or len(ranks) < 2:
        return None
    own = ranks[mine[0]]
    best_other = min(r for t, r in ranks.items() if t != mine[0])
    return "win" if own < best_other else ("draw" if own == best_other else "loss")
```

Implementation notes (checked while writing the part):
- `PRAGMA incremental_vacuum` frees one page per fetched result row: without `.fetchall()` it does almost nothing. The checkpoint after it is what shrinks the file in WAL mode.
- `auto_vacuum` is honoured only before the first table exists, hence the `new` check; the test asserts `PRAGMA auto_vacuum == 2`.
- `counts` keys are `"<team>|<entity id>:<role id>,..."` (ids of the `entities` table): about 95 bytes per row on 2v2 data, against about 240 with names; `import_parent` decodes the parent's keys with the parent's ids and re-encodes them with this journal's.
- Every transaction goes through `_transaction`, which clears the id cache on rollback (ids inserted by a rolled-back transaction do not exist).
- A uid already present is ignored (`INSERT OR IGNORE` + `rowcount == 0`) and reported in `AppendStats.accepted`, so a caller that feeds an in-memory table (T1.3) adds only journaled results. Training matches written under the disk guard go straight to `counts`; their uids are not kept (duplicates cannot be detected for them; training uids are unique by construction).
- `MatchLogReader` opens with `check_same_thread=False` (the dashboard's request threads, T7.3, use it one at a time); `MatchLog` keeps SQLite's default (one owner thread).

- [ ] **Step 6: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_comparisons.py tests/unit/test_sp4_matchlog.py -q -rw`
Expected: all pass (about 2 s).

- [ ] **Step 7: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/arena/comparisons.py src/colosseum/arena/matchlog.py tests/game_helpers.py \
        tests/unit/test_sp4_comparisons.py tests/unit/test_sp4_matchlog.py
git commit -m "feat: team-pair statistics and the SQLite match log (counts, compaction, disk guard, resume import, reader)"
git push
```

---

### Task T1.3: The journal in training: writer thread, bounded queue, shutdown flush, compaction cadence, resume import, `system` journal fields

Spec block 1 ("Запись": the writer thread, the bounded queue of 100 000, the shutdown flush; "Своевременная чистка"; "Resume из папки запуска") and block 7.2 (a monotonic `dropped_match_results`). The training main process opens the journal once at start-up (schema and file problems are a `ConfigError` before any child starts), imports the resumed run's journal on a run-dir resume (an SP3 run dir has none: INFO, the journal starts empty), reads the in-memory `ComparisonTable` of the live ratings from it once, and starts `MatchLogWriter`: a thread with its own connection. The main loop submits every drained result (never blocking; a full queue drops and counts); the thread appends in batches, feeds the table with what it journaled, compacts every `match_log.compact_interval_sec` (evicted snapshots' matches first, through a coordinator evict listener), and flushes its queue on stop within `SHUTDOWN_GRACE_SEC`. A journal error is an ERROR and a count, never a crash. The `system` record gains `dropped_match_results` (workers' cumulative drops, summed over every worker ever seen, plus the writer's) and `match_log` (the writer's stats and the journal sizes); the main log ends with the run's result accounting.

**Files:**
- Modify: `src/colosseum/arena/matchlog.py` (`RUN_SOURCES`, `MatchLogWriter`, `WriterStats`)
- Modify: `src/colosseum/launcher.py` (`_open_match_log`, `_on_snapshot_evicted`, `_match_log_stats`, `_stop_match_log`, submit in `_drain_results`, wiring in `launch` / `_monitor_loop` / `_finish_metrics`)
- Modify: `src/colosseum/core/run_dir.py` (`matches_path`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`add_evict_listener`)
- Modify: `src/colosseum/metrics/aggregator.py` (`CUMULATIVE_WORKER_KEYS`, `SystemStats.cumulative`, `dropped_match_results`, `match_log`)
- Modify: `src/colosseum/metrics/hub.py` (`maybe_tick` / `close(match_log=...)`, `system_cumulative`)
- Modify: `src/colosseum/metrics/jsonl.py` (`REQUIRED_KEYS["system"]` gains `dropped_match_results`)
- Modify (extend the T0.1 baseline): `tests/integration/test_sp4_sp3_resume.py`
- Test: `tests/unit/test_sp4_matchlog_writer.py`, `tests/integration/test_sp4_journal_runs.py`

**Interfaces:**
- Consumes: `MatchLog`, `MatchLogReader`, `RunInfo`, `ComparisonTable` (T1.2); `RunDir.ensure_run_id`, `read_run_id`, `MatchResult.env_steps` stamped by `_drain_results`, worker stats `dropped_results` (T1.1); `classify_resume_source` / `RESUME_RUN_DIR` (`coordinator/checkpoint_manager.py`), `config_hash`, `SHUTDOWN_GRACE_SEC`.
- Produces (contract T1.3, plus additions marked *):
  - `MatchLogWriter(log_factory, *, queue_size=100_000, compact_interval_sec, window_bytes, table=None, clock=time.monotonic)` (*`clock`) with `start`, `submit -> int`, `note_evicted`, `stop(timeout) -> WriterStats`, `stats`; `WriterStats` (`written`, `dropped`, `duplicates`, *`write_errors`, `compactions`, *`folded`, `degraded`, `sizes`, *`pending`; *`as_dict()`); *`RUN_SOURCES = ("train", "pulse")`, *`WRITER_QUEUE_SIZE`, *`WRITE_BATCH`, *`UNTOUCHED_WARNING_FLOOR`;
  - `RunDir.matches_path`;
  - *`Coordinator.add_evict_listener(listener)` (`listener(agent_id, checkpoint_id)` after the PFSP forget, on the main loop);
  - *`colosseum.metrics.aggregator.CUMULATIVE_WORKER_KEYS`, *`SystemStats.cumulative(key)`, *`SystemStats.snapshot(env_steps, queue_depths, match_log=None)`; `system` record keys *`dropped_match_results` (required) and *`match_log` (`WriterStats.as_dict()` or null);
  - *`MetricsHub.maybe_tick(..., match_log=None)`, *`MetricsHub.close(..., match_log=None)`, *`MetricsHub.system_cumulative(key)`;
  - *`Launcher._comparison_table` (the table the writer feeds; T2.2's `LiveRatings` input), *`Launcher._match_log_writer`, *`Launcher._results_received`;
  - main-log line *`Match results: <R> received, <W> journaled, <K> duplicates, <D> dropped by the journal writer, <Dw> dropped by workers`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_matchlog_writer.py`:
```python
"""The journal writer thread (SP4 T1.3, spec block 1 "Запись"): it journals what is submitted and feeds the
in-memory table, counts drops (queue full, write errors, left at stop) with one WARNING / ERROR each, compacts on
its cadence with the evicted entities, never raises into the training process, and stops within its timeout. Also
the run-level counters of the system record and the coordinator's evict listener."""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from functools import partial

import numpy as np

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.matchlog import (
    MATCH_LOG_FILE,
    WRITE_BATCH,
    AppendStats,
    CompactStats,
    LogSizes,
    MatchLog,
    MatchLogWriter,
    RunInfo,
)
from colosseum.metrics.aggregator import SystemStats
from game_helpers import make_coordinator, make_test_config, synthetic_journal_results

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="ttt", config_hash="h", checkpoint_interval=10)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeLog:
    """A MatchLog stand-in: records calls; ``fail_appends`` failing appends first; ``delay`` per append."""

    def __init__(self, fail_appends: int = 0, delay: float = 0.0, file_bytes: int = 0) -> None:
        self.appended: list = []
        self.compactions: list[tuple[int, frozenset]] = []
        self.fail_appends = fail_appends
        self.delay = delay
        self.file_bytes = file_bytes
        self.closed = False

    def append(self, results):
        time.sleep(self.delay)
        if self.fail_appends:
            self.fail_appends -= 1
            raise sqlite3.OperationalError("disk I/O error")
        self.appended.extend(results)
        return AppendStats(detailed=len(results), accepted=[True] * len(results))

    def rating_versions(self, result):
        return [None] * len(result.seats)

    def compact(self, *, window_bytes, evicted=(), should_stop=lambda: False, **_):
        self.compactions.append((window_bytes, frozenset(evicted)))
        return CompactStats()

    def sizes(self):
        return LogSizes(file_bytes=self.file_bytes, wal_bytes=0, detail_bytes=0, detail_matches=0, other_matches=0,
                        counts_rows=0, free_disk_bytes=10**12, degraded=False)

    def close(self):
        self.closed = True


def test_the_writer_journals_everything_and_feeds_the_table(tmp_path):
    table = ComparisonTable()
    factory = partial(MatchLog, tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    writer = MatchLogWriter(factory, compact_interval_sec=3600.0, window_bytes=10**9, table=table)
    writer.start()
    results = synthetic_journal_results(3000)
    for i in range(0, 3000, 250):
        assert writer.submit(results[i:i + 250]) == 0
    stats = writer.stop(timeout=7.0)
    assert (stats.written, stats.dropped, stats.write_errors, stats.pending) == (3000, 0, 0, 0)
    assert sum(s.n for _, _, s in table.pairs("2p", ["train"])) == 3000
    assert stats.sizes is not None and stats.sizes.detail_matches == 3000
    check = MatchLog(tmp_path / MATCH_LOG_FILE, RUN)
    assert check.comparison_table(("train",)).as_dict() == table.as_dict()
    check.close()


def test_a_full_writer_queue_drops_counts_and_warns_once(caplog):
    log = FakeLog()
    writer = MatchLogWriter(lambda: log, queue_size=10, compact_interval_sec=3600.0, window_bytes=1)
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        assert writer.submit(synthetic_journal_results(25)) == 15        # not started yet: the queue fills up
        assert writer.submit(synthetic_journal_results(5, first=25)) == 5
        writer.start()
        stats = writer.stop(timeout=5.0)
    assert (stats.written, stats.dropped) == (10, 20) and len(log.appended) == 10
    full = [r for r in caplog.records if "queue is full" in r.getMessage()]
    assert len(full) == 1 and full[0].levelno == logging.WARNING


def test_stop_counts_what_it_could_not_write_in_time(caplog):
    log = FakeLog(delay=0.3)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    results = synthetic_journal_results(8 * WRITE_BATCH)
    writer.submit(results)
    writer.start()
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        started = time.monotonic()
        stats = writer.stop(timeout=1.0)
        took = time.monotonic() - started
    assert took < 2.0
    assert stats.dropped > 0 and stats.written + stats.dropped == len(results)
    assert any("stopped with" in r.getMessage() for r in caplog.records)


def test_a_write_error_is_logged_and_counted_and_the_writer_goes_on(caplog):
    log = FakeLog(fail_appends=1)
    writer = MatchLogWriter(lambda: log, compact_interval_sec=3600.0, window_bytes=1)
    writer.start()
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        writer.submit(synthetic_journal_results(7))
        assert _wait(lambda: writer.stats.write_errors == 1)
        writer.submit(synthetic_journal_results(5, first=7))
        stats = writer.stop(timeout=5.0)
    assert (stats.written, stats.dropped, stats.write_errors) == (5, 7, 1)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None and "training goes on" in errors[0].getMessage()


def test_compaction_runs_on_its_cadence_with_the_evicted_entities():
    log = FakeLog()
    now = [0.0]
    writer = MatchLogWriter(lambda: log, compact_interval_sec=60.0, window_bytes=4096, clock=lambda: now[0])
    writer.start()
    writer.note_evicted(["main@v10"])
    time.sleep(0.3)
    assert log.compactions == []                        # the cadence has not elapsed
    now[0] = 61.0
    assert _wait(lambda: len(log.compactions) == 1)
    assert log.compactions[0] == (4096, frozenset({"main@v10"}))
    now[0] = 122.0
    assert _wait(lambda: len(log.compactions) == 2) and log.compactions[1] == (4096, frozenset())
    assert writer.stop(timeout=5.0).compactions == 2 and log.closed


def test_untouched_parts_beyond_the_window_warn_once(caplog):
    log = FakeLog(file_bytes=200 * 2**20)
    now = [0.0]
    writer = MatchLogWriter(lambda: log, compact_interval_sec=1.0, window_bytes=2**20, clock=lambda: now[0])
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.matchlog"):
        writer.start()
        for step in (2.0, 4.0):
            now[0] = step
            assert _wait(lambda step=step: len(log.compactions) == int(step // 2))
        writer.stop(timeout=5.0)
    assert len([r for r in caplog.records if "compaction does not touch" in r.getMessage()]) == 1


def test_a_journal_that_cannot_be_opened_never_stops_training(caplog):
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    writer = MatchLogWriter(broken, compact_interval_sec=3600.0, window_bytes=1)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.matchlog"):
        writer.start()
        writer.submit(synthetic_journal_results(4))
        assert _wait(lambda: writer.stats.dropped == 4)
        stats = writer.stop(timeout=5.0)
    assert stats.written == 0 and stats.dropped == 4
    assert any("opening the journal failed" in r.getMessage() for r in caplog.records)


def test_run_level_counters_never_decrease_and_include_the_writer():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_worker_stats({"worker_id": 0, "dropped_results": 3})
    stats.on_worker_stats({"worker_id": 1, "dropped_results": 2})
    clock[0] = 1.0
    assert stats.snapshot(0, {}, match_log={"dropped": 4})["dropped_match_results"] == 9
    clock[0] = 60.0                                      # both workers fell silent: their counts stay
    snap = stats.snapshot(0, {})
    assert snap["workers_reporting"] == 0 and snap["dropped_match_results"] == 5 and snap["match_log"] is None
    stats.on_worker_stats({"worker_id": 0, "dropped_results": 1})   # a stale lower report never decreases it
    assert stats.cumulative("dropped_results") == 5


def test_the_coordinator_tells_evict_listeners(tmp_path):
    config = make_test_config("solo", checkpoint={"interval": 10, "keep_last": 1, "keep_every": 0})
    coordinator = make_coordinator(config, tmp_path / "ckpt")
    heard: list[tuple[str, str]] = []
    coordinator.add_evict_listener(lambda agent, ckpt: heard.append((agent, ckpt)))
    state = {"w": np.zeros(2, np.float32)}
    for version in (10, 20, 30):
        coordinator.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": state})
    assert heard == [("agent_0", "ckpt_v10"), ("agent_0", "ckpt_v20")]


def test_stop_without_start_and_twice_is_harmless():
    writer = MatchLogWriter(FakeLog, compact_interval_sec=1.0, window_bytes=1)
    assert writer.stop(timeout=1.0).written == 0
    thread_count = threading.active_count()
    writer2 = MatchLogWriter(FakeLog, compact_interval_sec=1.0, window_bytes=1)
    writer2.start()
    writer2.stop(timeout=5.0)
    writer2.stop(timeout=5.0)
    assert threading.active_count() <= thread_count
```

Create `tests/integration/test_sp4_journal_runs.py`:
```python
"""The match log in real training runs (SP4 T1.3, spec block 1 and section 6): every match a worker finished is
journaled exactly once or counted as dropped, seats carry their weights and rating versions, uids carry the run
id, the system record reports the journal, a run-dir resume continues the journal (the parent's detail folded into
counts), and match_log.enabled: false writes no journal."""
from __future__ import annotations

import ast
import json
import re
import sqlite3

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.config import load_config

BUDGET = 8000          # longer than one worker-stats interval (2 s)


def _worker_stats(run) -> dict:
    line = next(line for line in run.log("worker-0").splitlines() if "Worker 0: finished." in line)
    return ast.literal_eval(line.split("finished. ", 1)[1])


def _accounting(run) -> dict[str, int]:
    m = re.search(r"Match results: (\d+) received, (\d+) journaled, (\d+) duplicates, (\d+) dropped by the journal "
                  r"writer, (\d+) dropped by workers", run.log("main"))
    assert m, run.log("main")[-3000:]
    return dict(zip(("received", "journaled", "duplicates", "writer", "workers"), map(int, m.groups()), strict=True))


def _journal_matches(path) -> int:
    conn = sqlite3.connect(path)
    try:
        detailed = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        counted = conn.execute("SELECT COALESCE(SUM(n), 0) FROM counts WHERE kind = 0").fetchone()[0]
    finally:
        conn.close()
    return detailed + counted                          # tic-tac-toe: one team pair per match


def test_every_match_is_journaled_once_or_counted(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="journal",
                    overrides={"training.total_timesteps": str(BUDGET), "match_log.compact_interval_sec": "1",
                               "match_log.detail_window_mb": "0.05"})
    assert run.returncode == 0, run.stderr[-3000:]
    worker, accounting = _worker_stats(run), _accounting(run)
    assert worker["episodes"] > 100
    assert worker["episodes"] == accounting["received"] + worker["dropped_results"]
    assert accounting["workers"] == worker["dropped_results"] and accounting["duplicates"] == 0
    assert accounting["received"] == accounting["journaled"] + accounting["writer"]
    assert _journal_matches(run.root / MATCH_LOG_FILE) == accounting["journaled"]
    system = run.records("system")[-1]
    assert system["dropped_match_results"] == accounting["writer"] + accounting["workers"]
    assert system["match_log"]["written"] == accounting["journaled"] and system["match_log"]["compactions"] >= 1
    # a 50 KB window: compaction folded matches into counts (the final record also counts the matches written after
    # the last pass; the disk bound itself is criterion 3.5, a slow test of part 3)
    assert system["match_log"]["folded"] > 0 and system["match_log"]["counts_rows"] > 0


def test_journaled_seats_carry_versions_and_uids_the_run_id(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="versions", overrides={"training.total_timesteps": str(BUDGET)})
    assert run.returncode == 0, run.stderr[-3000:]
    run_id = json.loads((run.root / "run.json").read_text())["run_id"]
    reader = MatchLogReader(run.root / MATCH_LOG_FILE)
    rows = reader.matches(limit=100_000)
    assert rows and all(r.uid.startswith(f"{run_id}/train/w0_e") and r.source == "train" for r in rows)
    steps = [r.env_steps for r in sorted(rows, key=lambda r: r.id)]
    assert steps == sorted(steps) and steps[-1] > 0                   # stamped on arrival, in arrival order
    interval = load_config(run.root / "config.resolved.yaml").checkpoint.interval   # 20 (cli_runner.TINY)
    latest = [s for r in rows for s in r.seats if s.network_id == "latest"]
    assert latest and all(s.version_start is not None and s.version_end >= s.version_start for s in latest)
    assert all(s.rating_version == (s.version_start // interval) * interval for s in latest)
    snapshots = [s for r in rows for s in r.seats if s.network_id.startswith("ckpt_v")]
    assert all(s.rating_version == int(s.network_id.removeprefix("ckpt_v")) == s.version_start for s in snapshots)
    assert [i.run_id for i in reader.runs()] == [run_id] and reader.runs()[0].checkpoint_interval == interval
    reader.close()


def test_a_run_dir_resume_continues_the_journal(tmp_path):
    first = run_train(TTT_CONFIG, tmp_path, name="first", overrides={"training.total_timesteps": "4000"})
    assert first.returncode == 0, first.stderr[-3000:]
    parent_id = json.loads((first.root / "run.json").read_text())["run_id"]
    parent = MatchLogReader(first.root / MATCH_LOG_FILE)
    parent_total = sum(s.n for _, _, s in parent.comparison_table().pairs("2p", ["train"]))
    parent.close()
    second = run_train(TTT_CONFIG, tmp_path, name="second",
                       overrides={"training.resume_from": str(first.root), "training.total_timesteps": "8000"})
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"Match log: imported {first.root / MATCH_LOG_FILE}" in second.log("main")
    child = MatchLogReader(second.root / MATCH_LOG_FILE)
    runs = child.runs()
    assert [r.run_id for r in runs][1:] == [parent_id] and runs[0].parent_run_id == parent_id
    child_total = sum(s.n for _, _, s in child.comparison_table().pairs("2p", ["train"]))
    assert child_total > parent_total and child.sizes().counts_rows > 0
    assert all(r.uid.startswith(runs[0].run_id) for r in child.matches(limit=100_000))   # no parent detail copied
    child.close()


def test_a_disabled_journal_writes_nothing(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="nolog", overrides={"match_log.enabled": "false"})
    assert run.returncode == 0, run.stderr[-3000:]
    assert not (run.root / MATCH_LOG_FILE).exists() and "Match log: off" in run.log("main")
    assert run.records("system")[-1]["match_log"] is None
```

Extend `tests/integration/test_sp4_sp3_resume.py` (T0.1): the docstring's last sentence becomes "T1.3: the new run starts an empty journal (INFO); T3.2: an empty league state (INFO).", add the import `from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader`, and append to the test:
```python
    assert f"has no {MATCH_LOG_FILE} (an SP3 run dir); the journal starts empty" in log        # T1.3
    reader = MatchLogReader(run.root / MATCH_LOG_FILE)
    (only,) = reader.runs()
    assert only.parent_run_id is None                                     # an SP3 run dir has no run id
    assert reader.matches(limit=1) and all(r.run_id == only.run_id for r in reader.matches(limit=1000))
    reader.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_matchlog_writer.py tests/integration/test_sp4_journal_runs.py tests/integration/test_sp4_sp3_resume.py -q`
Expected: `ImportError: cannot import name 'WRITE_BATCH' from 'colosseum.arena.matchlog'` (collection) and, in the integration tests, no `matches.db` / no `Match results:` line.

- [ ] **Step 3: The writer (`src/colosseum/arena/matchlog.py`)**

1. Append to the module docstring:
```
``MatchLogWriter`` (training main process, SP4 T1.3) is a thread with its own ``MatchLog`` connection: the main
loop submits results to its bounded queue (a full queue drops and counts them), the thread appends them in batches,
feeds the in-memory ``ComparisonTable`` of the live ratings with what it journaled, compacts every
``compact_interval_sec``, and flushes its queue on ``stop``. A write error is an ERROR in the log and a count,
never an exception in the training process.
```
2. Imports: add `import queue`, `import threading`, and `replace` to `from dataclasses import dataclass, field, replace`.
3. After `SCHEMA_VERSION = 1`:
```python
# The sources of a run's own ratings (live ratings, top-k, the report's ratings): tournaments rate only their own
# matches (spec block 2).
RUN_SOURCES = ("train", "pulse")
```
   and use it as the default of both `comparison_table(self, sources: Collection[str] = RUN_SOURCES, ...)` (`MatchLog` and `MatchLogReader`).
4. Append to the end of the module:
```python
# ---------------------------------------------------------------------------------------------------------------
# The writer thread of the training main process (T1.3)
# ---------------------------------------------------------------------------------------------------------------

WRITER_QUEUE_SIZE = 100_000
WRITE_BATCH = 2_000
# The parts of the file compaction does not touch (counts, tournament and pulse matches) may grow past the window:
# one WARNING when they exceed max(window, this many bytes).
UNTOUCHED_WARNING_FLOOR = 64 * 1024 * 1024


@dataclass(frozen=True)
class WriterStats:
    """What the writer did so far (an immutable snapshot; the main loop reads it for the ``system`` record)."""

    written: int = 0            # journaled results (detailed or counted)
    dropped: int = 0            # lost: writer queue full, write errors, left in the queue at stop, journal unusable
    duplicates: int = 0         # uids already in the journal
    write_errors: int = 0       # failed appends and compaction passes
    compactions: int = 0
    folded: int = 0             # training matches folded into counts by compaction
    degraded: bool = False      # the disk guard is on (counts only)
    sizes: LogSizes | None = None   # after the last compaction pass (or the start)
    pending: int = 0            # results waiting in the queue

    def as_dict(self) -> dict:
        """JSON-ready form for the ``system`` record (``match_log``)."""
        out = {"written": self.written, "dropped": self.dropped, "duplicates": self.duplicates,
               "write_errors": self.write_errors, "compactions": self.compactions, "folded": self.folded,
               "degraded": self.degraded, "pending": self.pending}
        if self.sizes is not None:
            out.update({"file_bytes": self.sizes.file_bytes, "wal_bytes": self.sizes.wal_bytes,
                        "detail_bytes": self.sizes.detail_bytes, "detail_matches": self.sizes.detail_matches,
                        "other_matches": self.sizes.other_matches, "counts_rows": self.sizes.counts_rows,
                        "free_disk_bytes": self.sizes.free_disk_bytes})
        return out


class MatchLogWriter:
    """Journal writer thread (module docstring). ``log_factory()`` opens the thread's own ``MatchLog``;
    ``table`` (optional) gets every journaled result; ``clock`` measures the compaction cadence (monotonic)."""

    def __init__(self, log_factory: Callable[[], MatchLog], *, queue_size: int = WRITER_QUEUE_SIZE,
                 compact_interval_sec: float, window_bytes: int, table: ComparisonTable | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory = log_factory
        self._queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self._interval = float(compact_interval_sec)
        self._window = int(window_bytes)
        self._untouched_limit = max(int(window_bytes), UNTOUCHED_WARNING_FLOOR)
        self._table = table
        self._clock = clock
        self._lock = threading.Lock()
        self._stats = WriterStats()
        self._evicted: set[EntityKey] = set()
        self._stop = threading.Event()
        self._deadline = math.inf
        self._thread: threading.Thread | None = None
        self._warned_drop = self._warned_error = self._warned_untouched = False

    @property
    def stats(self) -> WriterStats:
        with self._lock:
            return replace(self._stats, pending=self._queue.qsize())

    def _bump(self, **delta: int) -> None:
        with self._lock:
            self._stats = replace(self._stats, **{k: getattr(self._stats, k) + v for k, v in delta.items()})

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="match-log-writer", daemon=True)
        self._thread.start()

    def submit(self, results: Sequence[MatchResult]) -> int:
        """Queue ``results`` without blocking; returns how many were dropped (queue full; counted, first WARNING)."""
        dropped = 0
        for result in results:
            try:
                self._queue.put_nowait(result)
            except queue.Full:
                dropped += 1
        if dropped:
            self._bump(dropped=dropped)
            if not self._warned_drop:
                self._warned_drop = True
                logger.warning(f"Match log writer queue is full ({self._queue.maxsize} results); dropped {dropped} "
                               f"match results (counted in dropped_match_results; further drops are only counted)")
        return dropped

    def note_evicted(self, entities: Collection[EntityKey]) -> None:
        """Entities of evicted snapshots: their training matches are folded at the next compaction pass."""
        with self._lock:
            self._evicted.update(entities)

    def stop(self, timeout: float) -> WriterStats:
        """Stop compacting, write what is queued, close the journal; within ``timeout`` seconds. What could not be
        written in time is counted as dropped (one WARNING)."""
        self._deadline = self._clock() + max(0.0, float(timeout) - 0.5)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        left = 0
        while True:
            try:
                self._queue.get_nowait()
                left += 1
            except queue.Empty:
                break
        if left:
            self._bump(dropped=left)
            logger.warning(f"Match log writer stopped with {left} results not written (counted in "
                           f"dropped_match_results)")
        return self.stats

    # ---- the thread --------------------------------------------------------------------------------------------

    def _take(self, timeout: float) -> list[MatchResult]:
        try:
            batch = [self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()]
        except queue.Empty:
            return []
        while len(batch) < WRITE_BATCH:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _write(self, log: MatchLog, batch: list[MatchResult]) -> None:
        try:
            stats = log.append(batch)
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            self._bump(dropped=len(batch), write_errors=1)
            logger.error(f"Match log: writing {len(batch)} results failed ({type(e).__name__}: {e}); they are counted "
                         f"in dropped_match_results, training goes on", exc_info=not self._warned_error)
            self._warned_error = True
            return
        if self._table is not None:
            for result, accepted in zip(batch, stats.accepted, strict=True):
                if accepted:
                    self._table.add_result(result, log.rating_versions(result))
        self._bump(written=stats.detailed + stats.counted, duplicates=stats.duplicates)
        with self._lock:
            self._stats = replace(self._stats, degraded=stats.degraded)

    def _compact(self, log: MatchLog) -> None:
        with self._lock:
            evicted, self._evicted = self._evicted, set()
        try:
            result = log.compact(window_bytes=self._window, evicted=evicted, should_stop=self._stop.is_set)
            sizes = log.sizes()
        except Exception as e:  # noqa: BLE001 - a journal problem must never stop training
            self._bump(write_errors=1)
            with self._lock:
                self._evicted |= evicted
            logger.error(f"Match log: compaction failed ({type(e).__name__}: {e})", exc_info=not self._warned_error)
            self._warned_error = True
            return
        if result.interrupted:
            with self._lock:
                self._evicted |= evicted          # the evicted entities not folded yet stay pending
        untouched = sizes.file_bytes - sizes.detail_bytes
        if untouched > self._untouched_limit and not self._warned_untouched:
            self._warned_untouched = True
            logger.warning(f"Match log: the parts compaction does not touch (counts, tournament and pulse matches) "
                           f"take {untouched / 2**20:.0f} MB, more than the detail window; the journal grows past "
                           f"match_log.detail_window_mb (docs/ARENA_GUIDE.md)")
        self._bump(compactions=1, folded=result.folded)
        with self._lock:
            self._stats = replace(self._stats, sizes=sizes)

    def _run(self) -> None:
        try:
            log = self._factory()
        except Exception as e:  # noqa: BLE001 - reported; results are only counted from now on
            logger.error(f"Match log: opening the journal failed ({type(e).__name__}: {e}); match results are "
                         f"counted as dropped", exc_info=True)
            while not self._stop.is_set():
                batch = self._take(0.2)
                if batch:
                    self._bump(dropped=len(batch), write_errors=1)
            return
        try:
            with self._lock:
                self._stats = replace(self._stats, sizes=log.sizes())
            last_compact = self._clock()
            while not self._stop.is_set():
                batch = self._take(0.2)
                if batch:
                    self._write(log, batch)
                if self._clock() - last_compact >= self._interval:
                    self._compact(log)
                    last_compact = self._clock()
            while self._clock() < self._deadline:          # stop(): flush the queue, no more compaction
                batch = self._take(0.0)
                if not batch:
                    break
                self._write(log, batch)
            with self._lock:
                self._stats = replace(self._stats, sizes=log.sizes())
        finally:
            log.close()
```
   Thread rules (overview, global constraints): the writer owns its connection (opened in the thread by `log_factory`) and touches nothing of the main loop but its queue, its stats (under its lock) and the `ComparisonTable` (internally locked). `stop` sets a deadline `timeout - 0.5 s` for the flush, so `join(timeout)` returns in time; whatever is still queued afterwards is counted as dropped.

- [ ] **Step 4: Run directory and coordinator**

`src/colosseum/core/run_dir.py`: layout line `matches.db               the match log (SP4: colosseum.arena.matchlog)` after `ratings.json`, and after `run_info_path`:
```python
    @property
    def matches_path(self) -> Path:
        from colosseum.arena.matchlog import MATCH_LOG_FILE

        return self.root / MATCH_LOG_FILE
```
(a local import: `colosseum.arena.matchlog` imports `colosseum.core.types`, which `run_dir` must not pull in at import time.)

`src/colosseum/coordinator/coordinator.py`: in `__init__`, after `self._evicted = {}`:
```python
        self._evict_listeners: list[Callable[[str, str], None]] = []   # e.g. the journal writer (SP4)
```
before `_on_evict`:
```python
    def add_evict_listener(self, listener: Callable[[str, str], None]) -> None:
        """Also call ``listener(agent_id, checkpoint_id)`` for every evicted snapshot (main loop; SP4: the journal
        writer folds the snapshot's matches)."""
        self._evict_listeners.append(listener)
```
and in `_on_evict`, right after `self._pfsp.forget((agent_id, checkpoint_id))` (before the `_buffer_evictions` early return):
```python
        for listener in self._evict_listeners:
            listener(agent_id, checkpoint_id)
```
(its docstring: "..., its PFSP statistics are dropped, the evict listeners are told, and workers are told to unload it ..."). The retention index is updated before `on_evict` (SP3 ruling #24), so a listener sees the store without the snapshot.

- [ ] **Step 5: System metrics**

`src/colosseum/metrics/jsonl.py`: `REQUIRED_KEYS["system"]` gains `"dropped_match_results"` (after `"dropped_reward_episodes"`).

`src/colosseum/metrics/aggregator.py`: before `class SystemStats`:
```python
# Worker stats that are cumulative per worker: the run-level value is the sum over every worker ever seen of the
# largest value it reported, so it never decreases (SP4 spec block 7.2).
CUMULATIVE_WORKER_KEYS = ("dropped_results",)
```
`SystemStats` docstring, last sentence becomes: "Per-worker stats older than ``worker_timeout_sec`` (a dead or stuck worker) are forgotten, except the ``CUMULATIVE_WORKER_KEYS`` counters, which are kept for the whole run. ``dropped_match_results`` = the workers' dropped results + the journal writer's (``match_log["dropped"]``)." In `__init__`: `self._cumulative: dict[str, dict[int, int]] = {key: {} for key in CUMULATIVE_WORKER_KEYS}`. Replace `on_worker_stats` and add `cumulative`:
```python
    def on_worker_stats(self, stats: dict) -> None:
        worker = int(stats["worker_id"])
        self._workers[worker] = (self._clock(), dict(stats))
        for key, per_worker in self._cumulative.items():
            if key in stats:
                per_worker[worker] = max(per_worker.get(worker, 0), int(stats[key]))

    def cumulative(self, key: str) -> int:
        """A ``CUMULATIVE_WORKER_KEYS`` counter for the whole run."""
        return int(sum(self._cumulative[key].values()))
```
`snapshot` gets a keyword `match_log: dict[str, Any] | None = None` (docstring: "``match_log``: the journal writer's ``WriterStats.as_dict()`` (None: no journal).") and two keys before `"workers_reporting"`:
```python
            "dropped_match_results": self.cumulative("dropped_results") + int((match_log or {}).get("dropped", 0)),
            "match_log": None if match_log is None else dict(match_log),
```

`src/colosseum/metrics/hub.py`: `maybe_tick(self, *, env_steps, ratings, queue_depths, force=False, match_log: dict | None = None)` (docstring: "``match_log`` the journal writer's ``WriterStats.as_dict()`` (None: no journal), written into the ``system`` record") passes it on: `system = self._system.snapshot(env_steps, queue_depths, match_log=match_log)`; `close(..., match_log: dict | None = None)` passes it to its forced `maybe_tick`; after `on_worker_stats` add
```python
    def system_cumulative(self, key: str) -> int:
        """A run-level worker counter (``SystemStats.cumulative``)."""
        return self._system.cumulative(key)
```
WandB rows get `system/match_log/<number>` columns through the existing `flatten` (bools and None are dropped).

- [ ] **Step 6: The launcher (`src/colosseum/launcher.py`)**

1. `Launcher.__init__`, after `self._run_id`:
```python
        # Set by launch() when match_log.enabled: the journal writer thread and the in-memory comparison table it
        # feeds (the live ratings' input, T2.2); results received from the workers so far.
        self._match_log_writer = None
        self._comparison_table = None
        self._match_log_stopped = False
        self._results_received = 0
```
2. `launch()`: right after `self._open_metrics(resume_states)` add `self._open_match_log(coordinator)` (inside the same `try`, before any child starts: a `ConfigError` leaves no child behind, and the outer `finally` stops the writer).
3. `_monitor_loop`: the `maybe_tick` call gets `match_log=self._match_log_stats()`.
4. `_drain_results`: after the loop over the results:
```python
        self._results_received += len(results)
        if self._match_log_writer is not None:
            self._match_log_writer.submit(results)
```
5. New methods after `_drain_results`:
```python
    def _open_match_log(self, coordinator: Coordinator) -> None:
        """The match log (spec block 1). The journal is opened here first, so a schema or file problem is a
        ConfigError before any child starts; a run-dir resume imports the resumed run's journal (an SP3 run dir has
        none: INFO, the journal starts empty); the in-memory comparison table of the live ratings is read from it
        once; then the writer thread starts with its own connection. Evicted snapshots reach the writer through the
        coordinator's evict listener."""
        import sqlite3
        from functools import partial
        from pathlib import Path

        from colosseum.arena.matchlog import (
            MATCH_LOG_FILE,
            RUN_SOURCES,
            MatchLog,
            MatchLogError,
            MatchLogWriter,
            RunInfo,
        )
        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source
        from colosseum.core.config import config_hash
        from colosseum.core.errors import ConfigError
        from colosseum.core.run_dir import read_run_id

        cfg = self._config
        if not cfg.match_log.enabled:
            logger.info("Match log: off (match_log.enabled is false)")
            return
        resume_from = cfg.training.resume_from
        parent = (Path(resume_from) if resume_from and classify_resume_source(resume_from) == RESUME_RUN_DIR
                  else None)
        run = RunInfo(run_id=self._run_id, name=self._run_dir.run_name or self._run_dir.root.name,
                      parent_run_id=read_run_id(parent) if parent is not None else None, game=cfg.env.env_class,
                      config_hash=config_hash(cfg), checkpoint_interval=cfg.checkpoint.interval)
        path = self._run_dir.matches_path
        factory = partial(MatchLog, path, run, min_free_disk_mb=cfg.match_log.min_free_disk_mb)
        try:
            log = factory()
            try:
                if parent is not None:
                    parent_log = parent / MATCH_LOG_FILE
                    if parent_log.exists():
                        stats = log.import_parent(parent_log)
                        logger.info(f"Match log: imported {parent_log}: {stats.counts_rows} counts rows, "
                                    f"{stats.folded} detailed training matches folded into counts, "
                                    f"{stats.copied} tournament / pulse matches copied")
                    else:
                        logger.info(f"Match log: the resumed run dir {parent} has no {MATCH_LOG_FILE} (an SP3 run "
                                    f"dir); the journal starts empty")
                table = log.comparison_table(RUN_SOURCES)
            finally:
                log.close()
        except MatchLogError as e:
            raise ConfigError(f"{e}") from e
        except sqlite3.Error as e:
            raise ConfigError(f"Match log {path} cannot be opened or imported ({type(e).__name__}: {e}); move it "
                              f"away or set match_log.enabled: false") from e
        self._comparison_table = table
        self._match_log_writer = MatchLogWriter(
            factory, compact_interval_sec=cfg.match_log.compact_interval_sec,
            window_bytes=int(cfg.match_log.detail_window_mb * 1024 * 1024), table=table)
        self._match_log_writer.start()
        coordinator.add_evict_listener(self._on_snapshot_evicted)
        mlc = cfg.match_log
        logger.info(f"Match log: {path} (detail window {mlc.detail_window_mb:g} MB, min free disk "
                    f"{mlc.min_free_disk_mb:g} MB, compaction every {mlc.compact_interval_sec:g} s)")

    def _on_snapshot_evicted(self, agent_id: str, checkpoint_id: str) -> None:
        """Coordinator evict listener: the journal folds the evicted snapshot's training matches."""
        from colosseum.arena.entities import entity_key, snapshot_version

        version = snapshot_version(checkpoint_id)
        if self._match_log_writer is not None and version is not None:
            self._match_log_writer.note_evicted([entity_key(agent_id, version)])

    def _match_log_stats(self) -> dict | None:
        """The journal writer's stats for the ``system`` record (None without a journal)."""
        return None if self._match_log_writer is None else self._match_log_writer.stats.as_dict()

    def _stop_match_log(self) -> None:
        """Flush and stop the journal writer (within SHUTDOWN_GRACE_SEC); log the run's result accounting."""
        if self._match_log_writer is None or self._match_log_stopped:
            return
        self._match_log_stopped = True
        stats = self._match_log_writer.stop(SHUTDOWN_GRACE_SEC)
        worker_drops = self._hub.system_cumulative("dropped_results") if self._hub is not None else 0
        logger.info(f"Match results: {self._results_received} received, {stats.written} journaled, "
                    f"{stats.duplicates} duplicates, {stats.dropped} dropped by the journal writer, {worker_drops} "
                    f"dropped by workers")
```
6. `_finish_metrics`: inside `if self._hub is not None:` call `self._stop_match_log()` after `self._drain_metrics()`, and pass `match_log=self._match_log_stats()` to `self._hub.close(...)`; at the start of its `finally:` block add
```python
            try:
                self._stop_match_log()            # also when the final drains failed
            except Exception:  # noqa: BLE001 - must not replace the original error
                logger.exception("Stopping the match log writer failed")
```
   so the writer is stopped (once: `_match_log_stopped`) on every path. The writer flushes after the children are gone and the final drains are done; `stop(SHUTDOWN_GRACE_SEC)` bounds it (it normally takes milliseconds: the thread writes continuously).

- [ ] **Step 7: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_matchlog_writer.py tests/integration/test_sp4_journal_runs.py tests/integration/test_sp4_sp3_resume.py tests/unit/test_sp2_metrics.py tests/integration/test_sp2_metrics_outputs.py tests/unit/test_sp2_launcher_lifecycle.py tests/unit/test_sp2_coordinator.py -q -rw`
Expected: all pass (the four journal runs take about 30 s together).

- [ ] **Step 8: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings. Every `colosseum train` integration test now writes a `matches.db`; none of them pins the run dir's file list (`tests/unit/test_sp2_run_dir.py:127` checks what `RunDir.create` leaves, before any launch).

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/arena/matchlog.py src/colosseum/launcher.py src/colosseum/core/run_dir.py \
        src/colosseum/coordinator/coordinator.py src/colosseum/metrics tests/unit/test_sp4_matchlog_writer.py \
        tests/integration/test_sp4_journal_runs.py tests/integration/test_sp4_sp3_resume.py
git commit -m "feat: match log in training (writer thread, compaction cadence, resume import, system journal fields)"
git push
```

---

### Task T2.1: `arena.bt`: fit, bias by team index, groups and components, anchor, Laplace, bootstrap, G², score layouts

Spec block 2 ("Сущности", "Модель", "Подгонка"), criterion 3.2 and section 6 ("Bradley–Terry"). A pure numpy library. Per layout: team strength = mean of members' strengths + a bias by team index (sum 0); logistic likelihood with fractional outcomes over the `ComparisonTable`'s team pairs; a Gaussian prior on strengths and bias, and a random-walk smoothness between consecutive versions of one lineage; MAP by Newton's method with backtracking (dense Hessian up to 2000 parameters, preconditioned conjugate gradients above), warm-started. Cells = (role group, component), each with its anchor by the spec's rule chain (configured agent → first `RandomBot` → mean of the fixed agents that played → `v0` of the first trainable agent → first entity) and a note when a configured or default anchor did not play there. Ratings in Elo units. Laplace uncertainty (normal intervals; `P(best)`, `P(top-k)` per lineage and per cell, and the lineage-contrast lower bound from samples of `N(θ̂, H⁻¹)`); bootstrap over `BootstrapData` (compacted rows with Poisson outcomes, detailed matches with Poisson(1) weights, blocks resampled within strata: the data interface shared with the tournament's statistics, T5.3); the G² transitivity test with its own χ² survival function; mean-score summaries for one-team layouts. This task also adds the test promised by T1.2: fits from the journal equal fits from the in-memory table before and after compaction.

**Files:**
- Create: `src/colosseum/arena/bt.py`
- Modify: `tests/game_helpers.py` (`synthetic_bt_matches`, `bt_table`)
- Test: `tests/unit/test_sp4_bt.py`

**Interfaces:**
- Consumes: `ComparisonTable`, `PairStat`, `TeamKey`, `coefficients`, `match_pairs` (T1.2); `entity_version`, `lineage` (T1.1); `MatchLog` (T1.2, in the journal test).
- Produces (contract T2.1, plus additions marked *):
  - `BTOptions(prior_sigma=PRIOR_SIGMA, smoothness=SMOOTHNESS, anchor=None, max_iter=50, tol=1e-8, dense_limit=DENSE_LIMIT)` (*defaults from the module constants, *`dense_limit`), *`DEFAULT_OPTIONS`;
  - `BTGroup(layout, group, component, entities, anchor, *anchor_members, *rule, *note)`;
  - `BTFit(layout, groups, rating, se, bias, *bias_se, games, iterations, converged, *approximate, ...)` with *`group_of(entity)`;
  - `fit(table, layout, *, sources, options=DEFAULT_OPTIONS, fixed_agents=(), random_bot=None, *agent_order=(), warm=None)`;
  - `laplace(fit_, *, samples=1000, rng, top_k=0, quantile=0.1)`, `bootstrap(fit_, data, *, samples, rng, top_k=0, quantile=0.1)` → `Uncertainty(ci95, lcb, p_best_lineage, p_best_group, p_top_k, *p_top_k_group, *contrast_samples)`;
  - *`BootstrapData(counts, matches, blocks)`, *`Contribution` — the interface T5.3 builds for paired stratified bootstrap (`blocks = {stratum: {key: [contributions]}}`, the same key for every candidate's block on one seed);
  - `transitivity_test(table, fit_, *, sources, min_games=20)` → *`TransitivityResult(g2, dof, p_value, flagged, pairs_tested, worst_pairs)`; *`chi2_sf(x, dof)`;
  - `score_summary(table, layout, *, sources)` → *`ScoreSummary(layout, entities, compositions, homogeneous)`, *`ScoreCell(n, mean, ci_low, ci_high)`;
  - constants *`ELO_PER_LOGIT`, *`Z_95`, *`PRIOR_SIGMA = 4.0`, *`SMOOTHNESS = 0.5`, *`DENSE_LIMIT = 2000`, *`MEAN_OF_FIXED = "mean-of-fixed"`, *`TRANSITIVITY_P = 0.01`;
  - test support: *`synthetic_bt_matches(strengths, *, n_matches, rng, team_size, num_teams, bias, draw, roles, layout, choose, source, uid_prefix)`, *`bt_table(results, interval=1)`.

- [ ] **Step 1: Add the synthetic-data helpers to `tests/game_helpers.py`**

Append:
```python
# ---------------------------------------------------------------------------
# SP4 (T2.1): synthetic matches of known strengths for the Bradley-Terry tests
# ---------------------------------------------------------------------------


def _entity_seat(entity: str) -> tuple[str, str]:
    """``"main@v40"`` -> ``("main", "ckpt_v40")`` (a snapshot seat); any other name -> a fixed seat."""
    if "@v" in entity:
        agent, version = entity.rsplit("@v", 1)
        return agent, f"ckpt_v{version}"
    return entity, "fixed"


def synthetic_bt_matches(strengths: Mapping[str, float], *, n_matches: int, rng: np.random.Generator,
                         team_size: int = 1, num_teams: int = 2, bias: Sequence[float] | None = None,
                         draw: float = 0.0, roles: Sequence[str] | None = None, layout: str = "2p",
                         choose: Callable[[np.random.Generator], list[list[str]]] | None = None,
                         source: str = "train", uid_prefix: str = "u") -> list:
    """Match results drawn from a known Bradley-Terry model (logits). Entities ``agent@v<N>`` sit on snapshot seats
    (rating version N), other names on fixed seats. Teams: ``choose(rng)`` or ``num_teams * team_size`` distinct
    entities drawn uniformly. Team strength = mean member strength + ``bias[team index]``. Two teams: team 0 wins with
    ``sigmoid(S_0 - S_1)``, with draws of probability ``draw * 2 * min(p, 1 - p)`` arranged so the expected outcome
    stays ``p``; more teams: ranks from ``S + Gumbel noise`` (every pairwise probability is the logistic one).
    ``roles[t]`` is the role of team t's seats (default ``"player"``)."""
    from colosseum.core.types import MatchResult, SeatResult, TeamResult

    names = sorted(strengths)
    bias = list(bias) if bias is not None else [0.0] * num_teams
    roles = list(roles) if roles is not None else ["player"] * num_teams
    out = []
    for m in range(n_matches):
        if choose is not None:
            teams = choose(rng)
        else:
            picked = rng.choice(len(names), size=team_size * num_teams, replace=False)
            teams = [[names[i] for i in picked[t * team_size:(t + 1) * team_size]] for t in range(num_teams)]
        strength = [float(np.mean([strengths[e] for e in team])) + bias[t] for t, team in enumerate(teams)]
        if len(teams) == 2:
            p = 1.0 / (1.0 + math.exp(-(strength[0] - strength[1])))
            d = draw * 2.0 * min(p, 1.0 - p)
            if rng.random() < d:
                ranks = [1.0, 1.0]
            else:
                ranks = [1.0, 2.0] if rng.random() < (p - d / 2.0) / (1.0 - d) else [2.0, 1.0]
        else:
            order = np.argsort(-(np.asarray(strength) + rng.gumbel(size=len(teams))))
            ranks = [0.0] * len(teams)
            for place, team in enumerate(order):
                ranks[int(team)] = float(place + 1)
        seats = []
        for t, team in enumerate(teams):
            for entity in team:
                agent, network = _entity_seat(entity)
                seats.append(SeatResult(seat=len(seats), role=roles[t], team=t, agent_id=agent, network_id=network,
                                        reward=0.0))
        out.append(MatchResult(match_id=f"m{m}", layout=layout, outcome_kind="wdl" if len(teams) == 2 else "rank",
                               seats=seats, teams=[TeamResult(t, ranks[t], 0.0) for t in range(len(teams))],
                               episode_length=1, uid=f"{uid_prefix}{m}", source=source))
    return out


def bt_table(results: Sequence, interval: int = 1):
    """A ``ComparisonTable`` of ``results`` (rating versions with checkpoint interval ``interval``)."""
    from colosseum.arena.comparisons import ComparisonTable
    from colosseum.arena.entities import rating_version

    table = ComparisonTable()
    for result in results:
        table.add_result(result, [rating_version(s.network_id, s.version_start, interval) for s in result.seats])
    return table
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp4_bt.py`:
```python
"""Bradley-Terry ratings (SP4 T2.1, spec block 2, criterion 3.2 and section 6 "Bradley–Terry"): coverage of the
true anchored differences on synthetic data (1v1 with a side and draws, teams, FFA, asymmetric roles with and
without a bridge, a disconnected graph), the side bias, self-play, anchors, smoothness, finite estimates, the G2
test, score layouts, the iterative path, warm starts, Laplace and bootstrap uncertainty, and fits from the journal
before and after compaction. Every test reads the module's prior and smoothness constants (T8.2 may change them)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from colosseum.arena.bt import (
    ELO_PER_LOGIT,
    MEAN_OF_FIXED,
    Z_95,
    BootstrapData,
    BTOptions,
    bootstrap,
    chi2_sf,
    fit,
    laplace,
    score_summary,
    transitivity_test,
)
from colosseum.arena.comparisons import ComparisonTable, PairStat, match_pairs
from colosseum.arena.matchlog import MatchLog, RunInfo
from game_helpers import bt_table, make_match_result, synthetic_bt_matches, synthetic_journal_results

TRAIN = ("train",)


# ---------------------------------------------------------------------------
# coverage scenarios: each returns (true strengths in logits, results, fit keyword arguments)
# ---------------------------------------------------------------------------


def _one_v_one(rng):
    strengths = {"random": -2.0, "greedy": 0.5, "frozen": 1.0}
    strengths.update({f"main@v{v}": -1.5 + 0.25 * k for k, v in enumerate(range(0, 1300, 100))})
    results = synthetic_bt_matches(strengths, n_matches=140 * len(strengths), rng=rng, bias=[0.4, -0.4], draw=0.2)
    return strengths, results, dict(fixed_agents=("random", "greedy", "frozen"), random_bot="random",
                                    agent_order=("main", "random", "greedy", "frozen"))


def _teams(rng):
    strengths = {f"a{i}": (i - 7) * 0.3 for i in range(16)}
    strengths["random"] = -2.5
    return strengths, synthetic_bt_matches(strengths, n_matches=1700, rng=rng, team_size=2, bias=[0.2, -0.2]), dict(
        fixed_agents=("random",), random_bot="random")


def _ffa(rng):
    strengths = {f"a{i}": (i - 6) * 0.3 for i in range(12)}
    strengths["random"] = -2.0
    results = synthetic_bt_matches(strengths, n_matches=900, rng=rng, num_teams=4, bias=[0.3, 0.1, -0.1, -0.3],
                                   layout="4p")
    return strengths, results, dict(fixed_agents=("random",), random_bot="random")


def _asymmetric(bridge: bool):
    def scenario(rng):
        hunters = {f"h{i}": (i - 2) * 0.5 for i in range(5)}
        prey = {f"p{i}": (i - 2) * 0.5 for i in range(5)}
        strengths = {**hunters, **prey, **({"random": -1.5} if bridge else {})}
        h_pool = sorted(hunters) + (["random"] if bridge else [])
        p_pool = sorted(prey) + (["random"] if bridge else [])

        def choose(rng):
            while True:
                h, p = h_pool[rng.integers(len(h_pool))], p_pool[rng.integers(len(p_pool))]
                if h != p:
                    return [[h], [p]]

        results = synthetic_bt_matches(strengths, n_matches=1800, rng=rng, bias=[0.7, -0.7],
                                       roles=["hunter", "prey"], layout="1v1", choose=choose)
        kwargs = dict(fixed_agents=("random",), random_bot="random") if bridge else {}
        return strengths, results, kwargs
    return scenario


def _disconnected(rng):
    strengths = {f"a{i}": i * 0.4 for i in range(4)} | {f"b{i}": 3.0 + i * 0.4 for i in range(4)}

    def choose(rng):
        side = "a" if rng.random() < 0.5 else "b"
        i, j = rng.choice(4, 2, replace=False)
        return [[f"{side}{i}"], [f"{side}{j}"]]

    return strengths, synthetic_bt_matches(strengths, n_matches=1200, rng=rng, choose=choose), {}


SCENARIOS = {"1v1_side_draws": _one_v_one, "2v2_teams": _teams, "ffa_4": _ffa,
             "asymmetric_no_bridge": _asymmetric(False), "asymmetric_bridge": _asymmetric(True),
             "disconnected": _disconnected}
REPLICATIONS = 20
MIN_COVERAGE = 0.85


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_true_anchored_differences_fall_in_the_95_percent_intervals(name):
    """Criterion 3.2. Over 20 replications (seeds 0..19) the true difference of every non-anchor entity to its
    cell's anchor (same scale) must fall into its Laplace 95 % interval at least 85 % of the time (nominal 0.95,
    tolerance 0.10). Every entity has at least 200 comparisons; prior and smoothness are the module's. Calibration
    (T2.1, 15 independent runs of 20 replications per scenario, seeds 1000k..1000k+19): mean coverage 0.945-0.990,
    run-to-run sd at most 0.030, so 0.85 lies at least 3.4 sd below every scenario's mean: a correct implementation
    fails with probability below 1e-3 (normal approximation, per scenario)."""
    hits = total = 0
    for seed in range(REPLICATIONS):
        strengths, results, kwargs = SCENARIOS[name](np.random.default_rng(seed))
        fitted = fit(bt_table(results), results[0].layout, sources=TRAIN, **kwargs)
        assert fitted.converged
        assert min(fitted.games.values()) >= 200
        for group in fitted.groups:
            anchor = np.mean([strengths[a] for a in group.anchor_members])
            for entity in group.entities:
                if group.anchor_members == [entity]:
                    continue
                true = (strengths[entity] - anchor) * ELO_PER_LOGIT
                half = Z_95 * fitted.se[entity]
                hits += fitted.rating[entity] - half <= true <= fitted.rating[entity] + half
                total += 1
    assert hits / total >= MIN_COVERAGE, (hits, total)


def test_groups_without_a_bridge_are_not_compared_and_a_bridge_joins_them():
    _, results, _ = _asymmetric(False)(np.random.default_rng(0))
    alone = fit(bt_table(results), "1v1", sources=TRAIN)
    assert [(g.group, g.component) for g in alone.groups] == [("hunter", 0), ("prey", 0)]
    assert {g.group: g.anchor for g in alone.groups} == {"hunter": "h0", "prey": "p0"}   # each group its own zero
    assert alone.rating["h0"] == alone.rating["p0"] == 0.0
    _, results, kwargs = _asymmetric(True)(np.random.default_rng(0))
    bridged = fit(bt_table(results), "1v1", sources=TRAIN, **kwargs)
    ((group,),) = [bridged.groups]
    assert group.group == "hunter+prey" and group.anchor == "random" and len(group.entities) == 11


def test_a_disconnected_graph_gets_an_anchor_per_component():
    _, results, _ = _disconnected(np.random.default_rng(0))
    fitted = fit(bt_table(results), "2p", sources=TRAIN)
    assert [(g.component, g.anchor, g.entities) for g in fitted.groups] == [
        (0, "a0", ["a0", "a1", "a2", "a3"]), (1, "b0", ["b0", "b1", "b2", "b3"])]
    assert fitted.rating["a0"] == fitted.rating["b0"] == 0.0


def test_the_side_bias_is_recovered_and_sums_to_zero():
    """1v1 without draws, true bias +-0.4 logits; the estimate is within 4 standard errors (a correct
    implementation fails with probability 6e-5)."""
    strengths = {f"a{i}": (i - 4) * 0.3 for i in range(9)}
    results = synthetic_bt_matches(strengths, n_matches=6000, rng=np.random.default_rng(3), bias=[0.4, -0.4])
    fitted = fit(bt_table(results), "2p", sources=TRAIN)
    assert fitted.bias[0] == pytest.approx(-fitted.bias[1])
    assert abs(fitted.bias[0] - 0.4 * ELO_PER_LOGIT) < 4 * fitted.bias_se[0]


def test_self_play_informs_only_the_bias_and_counts_for_no_games():
    rng = np.random.default_rng(5)
    results = []
    for k in range(3000):                       # latest of one bucket against itself: the first side wins 70 %
        first = rng.random() < 0.7
        results.append(make_match_result([(0, "main", "latest", "player", 3), (1, "main", "latest", "player", 4)],
                                         [1.0, 2.0] if first else [2.0, 1.0], uid=f"s{k}"))
    for k in range(400):
        results.append(make_match_result([(0, "main", "latest", "player", 2), (1, "random", "fixed")],
                                         [1.0, 2.0] if rng.random() < 0.6 else [2.0, 1.0], uid=f"r{k}"))
    fitted = fit(bt_table(results, interval=10), "2p", sources=TRAIN, random_bot="random", fixed_agents=("random",))
    assert fitted.games == {"main@v0": 400, "random": 400}
    true_bias = 0.5 * math.log(0.7 / 0.3) * ELO_PER_LOGIT        # b0 - b1 = logit(0.7)
    assert abs(fitted.bias[0] - true_bias) < 4 * fitted.bias_se[0]


def _duels(pairs: dict[tuple[str, str], tuple[int, int]]) -> ComparisonTable:
    """A table from ``{(a, b): (wins of a, wins of b)}``, every match on both sides equally."""
    table = ComparisonTable()
    k = 0
    for (a, b), (wins_a, wins_b) in pairs.items():
        for winner, count in ((a, wins_a), (b, wins_b)):
            for i in range(count):
                first, second = (a, b) if i % 2 == 0 else (b, a)
                table.add_result(make_match_result([(0, *_seat(first)), (1, *_seat(second))],
                                                   [1.0, 2.0] if winner == first else [2.0, 1.0], uid=f"d{k}"),
                                 [_version(first), _version(second)])
                k += 1
    return table


def _seat(entity: str) -> tuple[str, str]:
    if "@v" in entity:
        agent, version = entity.rsplit("@v", 1)
        return agent, f"ckpt_v{version}"
    return entity, "fixed"


def _version(entity: str) -> int | None:
    return int(entity.rsplit("@v", 1)[1]) if "@v" in entity else None


def test_anchor_rules():
    table = _duels({("main@v0", "random"): (30, 30), ("main@v100", "greedy"): (40, 20), ("random", "greedy"): (20, 40),
                    ("other@v0", "main@v100"): (10, 50)})
    order = ("main", "other", "random", "greedy")
    by_rule = {}
    for options, kwargs in (
            (BTOptions(anchor="greedy"), {}),
            (BTOptions(anchor="main"), {}),
            (BTOptions(), dict(random_bot="random")),
            (BTOptions(), dict(fixed_agents=("random", "greedy"))),
            (BTOptions(), dict(agent_order=order)),
            (BTOptions(), {}),
            (BTOptions(anchor="nobody"), dict(random_bot="random")),
    ):
        fitted = fit(table, "2p", sources=TRAIN, options=options, **kwargs)
        (group,) = fitted.groups
        by_rule[(options.anchor, tuple(sorted(kwargs)))] = (group.anchor, group.rule, group.note)
        members = group.anchor_members
        assert np.mean([fitted.rating[m] for m in members]) == pytest.approx(0.0, abs=1e-9)
    assert by_rule[("greedy", ())] == ("greedy", "config", None)
    assert by_rule[("main", ())] == ("main@v0", "config", None)                 # a trainable agent: its v0
    assert by_rule[(None, ("random_bot",))] == ("random", "random_bot", None)
    assert by_rule[(None, ("fixed_agents",))] == (MEAN_OF_FIXED, "mean_of_fixed", None)
    assert by_rule[(None, ("agent_order",))] == ("main@v0", "first_trainable", None)
    assert by_rule[(None, ())] == ("greedy", "first", None)
    assert by_rule[("nobody", ("random_bot",))] == ("random", "random_bot", "ratings.anchor 'nobody' did not play here")


def test_a_random_bot_that_did_not_play_is_replaced_with_a_note():
    table = _duels({("main@v0", "greedy"): (30, 30), ("main@v100", "greedy"): (40, 20)})
    (group,) = fit(table, "2p", sources=TRAIN, random_bot="random", fixed_agents=("random", "greedy")).groups
    assert (group.anchor, group.rule, group.note) == (MEAN_OF_FIXED, "mean_of_fixed",
                                                      "RandomBot agent 'random' did not play here")
    assert group.anchor_members == ["greedy"]


def test_smoothness_pulls_a_rarely_played_version_towards_its_neighbours():
    pairs = {(f"main@v{v}", "random"): (50 + 2 * k, 50 - 2 * k) for k, v in enumerate((0, 10, 20, 40, 50))}
    pairs[("main@v30", "random")] = (0, 3)                      # three games, all lost
    table = _duels(pairs)
    smooth = fit(table, "2p", sources=TRAIN, random_bot="random")
    rough = fit(table, "2p", sources=TRAIN, random_bot="random", options=BTOptions(smoothness=0.0))
    neighbours = (smooth.rating["main@v20"] + smooth.rating["main@v40"]) / 2
    assert abs(smooth.rating["main@v30"] - neighbours) < abs(rough.rating["main@v30"] - neighbours)
    assert smooth.rating["main@v30"] > rough.rating["main@v30"]


def test_an_entity_that_always_loses_or_rarely_plays_gets_a_finite_rating():
    table = _duels({("x", "loser"): (40, 0), ("y", "loser"): (40, 0), ("x", "y"): (20, 20), ("x", "rare"): (1, 0)})
    fitted = fit(table, "2p", sources=TRAIN, fixed_agents=("x", "y", "loser", "rare"))
    for entity in ("loser", "rare"):
        assert math.isfinite(fitted.rating[entity]) and math.isfinite(fitted.se[entity]) and fitted.se[entity] > 0
    assert fitted.rating["loser"] < fitted.rating["x"] - 300


def test_the_transitivity_test_is_quiet_on_transitive_data_with_rare_pairs_and_flags_a_cycle():
    """Transitive: 6 entities with about 100 matches per pair and side, plus 10 entities with a few matches each
    (pairs below min_games are left out); under the null the p-value is uniform, the assertion p > 1e-3 fails with
    probability 1e-3 (seed 0 was checked). A rock-paper-scissors cycle (80 % each way, 300 matches per pair) is
    flagged."""
    rng = np.random.default_rng(0)
    strengths = {f"a{i}": i * 0.3 for i in range(6)}

    def frequent(rng):
        i, j = rng.choice(6, 2, replace=False)
        return [[f"a{i}"], [f"a{j}"]]

    results = synthetic_bt_matches(strengths, n_matches=3000, rng=rng, choose=frequent, bias=[0.3, -0.3])
    rare = {f"r{i}": float(rng.normal()) for i in range(10)}
    results += synthetic_bt_matches({**strengths, **rare}, n_matches=100, rng=rng, uid_prefix="rare",
                                    choose=lambda rng: [[f"r{rng.integers(10)}"], [f"a{rng.integers(6)}"]])
    table = bt_table(results)
    transitive = transitivity_test(table, fit(table, "2p", sources=TRAIN), sources=TRAIN)
    assert transitive.pairs_tested == 30 and transitive.dof > 0
    assert transitive.p_value > 1e-3 and transitive.flagged == (transitive.p_value < 0.01)
    cycle = _duels({("A", "B"): (240, 60), ("B", "C"): (240, 60), ("C", "A"): (240, 60)})
    result = transitivity_test(cycle, fit(cycle, "2p", sources=TRAIN), sources=TRAIN)
    assert result.flagged and result.p_value < 1e-10 and len(result.worst_pairs) == 5
    few = transitivity_test(cycle, fit(cycle, "2p", sources=TRAIN), sources=TRAIN, min_games=10_000)
    assert (few.p_value, few.flagged, few.pairs_tested) == (None, False, 0)


def test_chi2_survival_function():
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(18.307038053275146, 10) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(124.34211340400407, 100) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(0.0, 3) == 1.0 and chi2_sf(1e4, 3) < 1e-300 + 1e-12


def test_score_layouts_summarize_entities_and_compositions():
    table = ComparisonTable()
    for k, (members, score) in enumerate([(("a", "a"), 1.0), (("a", "a"), 3.0), (("a", "b"), 5.0), (("a", "b"), 7.0)]):
        table.add_result(make_match_result([(0, m, "fixed") for m in members], scores=[score], layout="coop",
                                           uid=f"c{k}"), [None, None])
    summary = score_summary(table, "coop", sources=TRAIN)
    assert summary.compositions["a+a"].mean == 2.0 and summary.homogeneous == {"a+a": True, "a+b": False}
    assert (summary.entities["a"].n, summary.entities["a"].mean) == (4, 4.0)
    assert (summary.entities["b"].n, summary.entities["b"].mean) == (2, 6.0)
    half = Z_95 * math.sqrt(2.0 / 2)                           # sd of (5, 7) = sqrt(2), n = 2
    assert summary.compositions["a+b"].ci_low == pytest.approx(6.0 - half)
    assert fit(table, "coop", sources=TRAIN).groups == []      # no pairs: no Bradley-Terry fit


def test_the_iterative_path_agrees_with_the_dense_one():
    strengths, results, kwargs = _one_v_one(np.random.default_rng(1))
    table = bt_table(results)
    dense = fit(table, "2p", sources=TRAIN, **kwargs)
    iterative = fit(table, "2p", sources=TRAIN, options=BTOptions(dense_limit=5), **kwargs)
    assert not dense.approximate and iterative.approximate and iterative.converged
    assert max(abs(dense.rating[e] - iterative.rating[e]) for e in dense.rating) < 1e-6
    assert all(0.8 < iterative.se[e] / dense.se[e] < 1.25 for e in dense.se if dense.se[e] > 0)
    assert len(laplace(iterative, samples=200, rng=np.random.default_rng(0)).lcb) == len(strengths)


def test_a_warm_start_converges_faster_to_the_same_fit():
    _, results, kwargs = _one_v_one(np.random.default_rng(2))
    table = bt_table(results)
    cold = fit(table, "2p", sources=TRAIN, **kwargs)
    warm = fit(table, "2p", sources=TRAIN, warm=cold, **kwargs)
    assert warm.iterations < cold.iterations
    assert max(abs(cold.rating[e] - warm.rating[e]) for e in cold.rating) < 1e-6


def test_an_empty_table_gives_an_empty_fit():
    empty = fit(ComparisonTable(), "2p", sources=TRAIN)
    assert (empty.groups, empty.rating, empty.converged) == ([], {}, True)
    assert laplace(empty, rng=np.random.default_rng(0)).ci95 == {}
    assert transitivity_test(ComparisonTable(), empty, sources=TRAIN).p_value is None


def test_laplace_probabilities_lower_bounds_and_determinism():
    _, results, kwargs = _one_v_one(np.random.default_rng(4))
    fitted = fit(bt_table(results), "2p", sources=TRAIN, **kwargs)
    first = laplace(fitted, samples=2000, rng=np.random.default_rng(7), top_k=3, quantile=0.1)
    again = laplace(fitted, samples=2000, rng=np.random.default_rng(7), top_k=3, quantile=0.1)
    assert first.lcb == again.lcb
    assert sum(first.p_best_group.values()) == pytest.approx(1.0)
    mains = [e for e in fitted.rating if e.startswith("main@")]
    assert sum(first.p_best_lineage[e] for e in mains) == pytest.approx(1.0)
    assert sum(first.p_top_k[e] for e in mains) == pytest.approx(3.0)
    assert sum(first.p_top_k_group.values()) == pytest.approx(3.0)
    mean_main = np.mean([fitted.rating[e] for e in mains])
    for entity in mains:                                       # the 10 % quantile lies below the mean contrast
        assert first.lcb[entity] < fitted.rating[entity] - mean_main
        assert first.contrast_samples[entity].shape == (2000,)
    lo, hi = first.ci95["main@v1200"]
    assert (hi - lo) / 2 == pytest.approx(Z_95 * fitted.se["main@v1200"])


def test_bootstrap_by_matches_counts_and_blocks():
    strengths, results, kwargs = _one_v_one(np.random.default_rng(6))
    table = bt_table(results)
    fitted = fit(table, "2p", sources=TRAIN, **kwargs)
    fast = laplace(fitted, samples=1000, rng=np.random.default_rng(0))
    contributions = [match_pairs(r, [None if s.network_id == "fixed" else int(s.network_id[6:]) for s in r.seats])
                     for r in results]
    by_matches = bootstrap(fitted, BootstrapData(matches=contributions), samples=60, rng=np.random.default_rng(1))
    by_counts = bootstrap(fitted, BootstrapData(counts=table.pairs("2p", TRAIN)), samples=60,
                          rng=np.random.default_rng(1))
    blocks = {"field": {str(k): [c for match in contributions[k * 8:(k + 1) * 8] for c in match]
                        for k in range(len(contributions) // 8)}}
    by_blocks = bootstrap(fitted, BootstrapData(blocks=blocks), samples=60, rng=np.random.default_rng(1))
    for uncertainty in (by_matches, by_counts, by_blocks):
        for entity in ("main@v0", "main@v1200", "greedy"):
            lo, hi = uncertainty.ci95[entity]
            flo, fhi = fast.ci95[entity]
            assert 0.5 < (hi - lo) / (fhi - flo) < 2.0          # same order of magnitude as Laplace
            assert lo < fitted.rating[entity] < hi
    stranger = BootstrapData(counts=_duels({("p", "q"): (1, 1)}).pairs("2p", TRAIN))
    with pytest.raises(ValueError, match="does not have"):
        bootstrap(fitted, stranger, samples=2, rng=np.random.default_rng(0))


def test_fits_from_the_journal_match_the_in_memory_table_before_and_after_compaction(tmp_path):
    """Spec section 6 ("сжатие не меняет рейтинги"): the journal's table gives the fit of the in-memory table, and
    so does it after most matches were folded into counts (the team index, hence the bias, is kept)."""
    run = RunInfo("R", "run", None, "ttt", "h", 10)
    log = MatchLog(tmp_path / "matches.db", run, disk_free_fn=lambda _: 10**12)
    results = synthetic_journal_results(3000)
    log.append(results)
    memory = ComparisonTable()
    for result in results:
        memory.add_result(result, log.rating_versions(result))
    kwargs = dict(sources=TRAIN, fixed_agents=("random", "greedy"), random_bot="random", agent_order=("main",))
    reference = fit(memory, "2p", **kwargs)
    before = fit(log.comparison_table(TRAIN), "2p", **kwargs)
    log.compact(window_bytes=log.sizes().detail_bytes // 10)
    after = fit(log.comparison_table(TRAIN), "2p", **kwargs)
    log.close()
    for other in (before, after):
        assert max(abs(other.rating[e] - reference.rating[e]) for e in reference.rating) < 1e-6
        assert max(abs(other.se[e] - reference.se[e]) for e in reference.se) < 1e-6
        assert other.bias[0] == pytest.approx(reference.bias[0], abs=1e-6)
    assert PairStat(1, 1.0, 1.0).n == 1
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_bt.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena.bt'`.

- [ ] **Step 4: Write `src/colosseum/arena/bt.py`**

```python
"""Bradley-Terry ratings of entities from team-pair statistics (SP4 spec block 2).

Model, per layout. Team strength = mean of its members' strengths ``theta`` + a bias ``b[team index]`` (the side
and its roles; ``sum_t b_t = 0``, parametrized as ``b = C beta`` with ``C = [I; -1]``). ``P(a beats b) =
sigmoid(S_a - S_b)``; a draw counts 0.5 (logistic likelihood with fractional outcomes). Each team pair of a match
is one comparison with weight ``1 / (T - 1)`` (``colosseum.arena.comparisons``). Priors: ``theta_e ~ N(0,
prior_sigma^2)`` (finite estimates for an entity that always loses), ``beta ~ N(0, prior_sigma^2)``, and a
random-walk smoothness ``-(smoothness / 2) (theta_{v_k+1} - theta_{v_k})^2`` between consecutive versions of one
lineage. MAP by Newton's method with backtracking on the log posterior: a dense Hessian up to
``BTOptions.dense_limit`` parameters, conjugate gradients with a Jacobi preconditioner above; warm start from a
previous fit.

Groups and components. Entities that played the same roles of the layout (in the data) form a role group; groups
merge when an entity played roles of both (a bridge). Entities linked by comparisons form a component. Ratings are
comparable only within a cell = (role group, component); every cell has its own anchor (rating 0), chosen by
``_anchor``: ``options.anchor`` (an agent of the config: its entity, or for a trainable agent its lowest version)
if it played in the cell; else the config's first ``RandomBot`` agent; else the mean of the scripted / frozen agents
that played in the cell; else the lowest version (``v0``) of the first trainable agent (config order) of the cell;
else the first entity by name. A configured or default anchor that did not play in a cell is noted in
``BTGroup.note`` (callers log each note once).

Scale: Elo units, ``ELO_PER_LOGIT = 400 / ln 10`` per logit.

Uncertainty. ``laplace``: normal intervals from the inverse Hessian at the MAP; ``P(best)``, ``P(top-k)`` and the
lower bound from samples of ``N(theta_hat, H^-1)`` (a diagonal approximation on the iterative path).
``bootstrap``: refits on resampled data (``BootstrapData``), warm-started at the fit, percentile intervals. Both
report ``P(best)`` within the entity's lineage and within its cell, ``P(top-k)`` likewise, and ``lcb``, the
``quantile`` of the entity's lineage contrast ``theta_e - mean(theta of its lineage in the cell)`` (spec block 4:
the anchor's uncertainty does not reorder the snapshots of one agent).

``PRIOR_SIGMA`` and ``SMOOTHNESS`` are provisional; task T8.2 fixes them by cross-validation (controller rulings).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from colosseum.arena.comparisons import ComparisonTable, PairStat, TeamKey, coefficients
from colosseum.arena.entities import EntityKey, entity_version, lineage

ELO_PER_LOGIT = 400.0 / math.log(10.0)
Z_95 = 1.959963984540054
PRIOR_SIGMA = 4.0          # logits (~695 Elo); provisional (T8.2)
SMOOTHNESS = 0.5           # precision of one version step (sd ~1.41 logits); provisional (T8.2)
DENSE_LIMIT = 2000         # parameters; above it Newton steps use conjugate gradients
MEAN_OF_FIXED = "mean-of-fixed"
TRANSITIVITY_P = 0.01      # the G2 test flags p < 0.01 (spec block 2)


@dataclass(frozen=True)
class BTOptions:
    prior_sigma: float = PRIOR_SIGMA
    smoothness: float = SMOOTHNESS
    anchor: str | None = None
    max_iter: int = 50
    tol: float = 1e-8
    dense_limit: int = DENSE_LIMIT


DEFAULT_OPTIONS = BTOptions()


@dataclass
class BTGroup:
    """One cell of a layout: the entities of one role group and one component, and their anchor."""

    layout: str
    group: str                       # '+'-joined sorted role names of the role group ("player", "hunter+prey")
    component: int
    entities: list[EntityKey]
    anchor: EntityKey | str          # an entity, or MEAN_OF_FIXED
    anchor_members: list[EntityKey]  # the entities whose mean is 0 (one for an entity anchor)
    rule: str                        # "config" | "random_bot" | "mean_of_fixed" | "first_trainable" | "first"
    note: str | None = None          # why a configured / default anchor was replaced


@dataclass
class _Design:
    """Sparse design of a fit: rows = team pairs, columns = entities then the free bias parameters."""

    entities: list[EntityKey]
    index: dict[EntityKey, int]
    num_teams: int
    keys: list[tuple[TeamKey, TeamKey]]
    row_of: dict[tuple[TeamKey, TeamKey], int]
    nz_row: np.ndarray
    nz_col: np.ndarray
    nz_val: np.ndarray
    sw: np.ndarray
    swy: np.ndarray
    prior: np.ndarray | None         # dense prior precision (None above the dense limit)
    prior_diag: np.ndarray
    chains: list[tuple[int, int]]    # smoothness edges (column pairs)

    @property
    def num_params(self) -> int:
        return len(self.entities) + max(self.num_teams - 1, 0)


@dataclass
class BTFit:
    layout: str
    groups: list[BTGroup]
    rating: dict[EntityKey, float]      # Elo units, anchor = 0 in its cell
    se: dict[EntityKey, float]          # Laplace standard errors of the anchored rating (Elo units)
    bias: dict[int, float]              # team index -> Elo units; sums to 0 over the layout's teams
    bias_se: dict[int, float]
    games: dict[EntityKey, int]         # matches with a non-zero coefficient (self-play excluded)
    iterations: int
    converged: bool
    approximate: bool = False           # iterative path: standard errors from the Hessian's diagonal
    params: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)   # MAP, logits
    design: _Design | None = field(default=None, repr=False)
    options: BTOptions = field(default=DEFAULT_OPTIONS, repr=False)
    hessian: np.ndarray | None = field(default=None, repr=False)   # dense negative Hessian at the MAP

    def group_of(self, entity: EntityKey) -> BTGroup:
        for group in self.groups:
            if entity in group.entities:
                return group
        raise KeyError(entity)


@dataclass
class Uncertainty:
    ci95: dict[EntityKey, tuple[float, float]]
    lcb: dict[EntityKey, float]                   # quantile of the lineage contrast (Elo units)
    p_best_lineage: dict[EntityKey, float]
    p_best_group: dict[EntityKey, float]          # within the cell
    p_top_k: dict[EntityKey, float]               # within the lineage (only with top_k > 0)
    p_top_k_group: dict[EntityKey, float]         # within the cell (only with top_k > 0)
    contrast_samples: dict[EntityKey, np.ndarray] = field(default_factory=dict, repr=False)   # Elo, float32


Contribution = tuple[TeamKey, TeamKey, float, float]   # (a, b, weight, outcome of a), as comparisons.match_pairs


@dataclass
class BootstrapData:
    """What bootstrap replicates resample, a decomposition of the fitted table (shared with the tournament's
    statistics, T5.3):

    - ``counts``: aggregated rows ``(a, b, PairStat)`` (compacted training matches): wins ~ Poisson(swy / w) and
      losses ~ Poisson(n - swy / w) with ``w = sw / n``: exact without draws, a draw's variance is overstated;
    - ``matches``: per detailed match the list of its contributions; every match gets a Poisson(1) weight, so the
      pairs of one FFA match move together;
    - ``blocks``: ``{stratum: {key: [contributions]}}`` (tournament and pulse blocks: a stratum is a field member
      or composition, a key a seed index shared by every candidate); per replicate and stratum the keys are drawn
      with replacement, the same draw for every contribution of a key (paired, stratified).
    """

    counts: list[tuple[TeamKey, TeamKey, PairStat]] = field(default_factory=list)
    matches: list[list[Contribution]] = field(default_factory=list)
    blocks: dict[str, dict[str, list[Contribution]]] = field(default_factory=dict)


@dataclass
class TransitivityResult:
    g2: float
    dof: int
    p_value: float | None            # None: not testable (no pair with min_games, or dof <= 0)
    flagged: bool                    # p_value < TRANSITIVITY_P
    pairs_tested: int
    worst_pairs: list[tuple[TeamKey, TeamKey, float]]   # (a, b, signed deviance residual), largest |r| first


@dataclass
class ScoreCell:
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class ScoreSummary:
    layout: str
    entities: dict[EntityKey, ScoreCell]
    compositions: dict[str, ScoreCell]          # "a+a+b": sorted member entities
    homogeneous: dict[str, bool]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _UnionFind:
    """Union-find whose root is the smallest element (deterministic labels)."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _build_design(rows: Sequence[tuple[TeamKey, TeamKey, PairStat]], options: BTOptions) -> _Design | None:
    coefs = [coefficients(a, b) for a, b, _ in rows]
    entities = sorted({e for c in coefs for e in c})
    if not entities:
        return None
    index = {e: i for i, e in enumerate(entities)}
    num_teams = 1 + max(max(a.team, b.team) for a, b, _ in rows)
    E, F = len(entities), max(num_teams - 1, 0)

    def bias_coef(team: int) -> dict[int, float]:      # b = C beta: team t < T-1 -> beta_t, team T-1 -> -sum(beta)
        if team < num_teams - 1:
            return {E + team: 1.0}
        return {E + j: -1.0 for j in range(F)}

    nz_row, nz_col, nz_val = [], [], []
    for r, ((a, b, _), coef) in enumerate(zip(rows, coefs, strict=True)):
        entry: dict[int, float] = {index[e]: c for e, c in coef.items()}
        for col, val in bias_coef(a.team).items():
            entry[col] = entry.get(col, 0.0) + val
        for col, val in bias_coef(b.team).items():
            entry[col] = entry.get(col, 0.0) - val
        for col, val in entry.items():
            if val != 0.0:
                nz_row.append(r)
                nz_col.append(col)
                nz_val.append(val)
    by_lineage: dict[str, list[tuple[int, EntityKey]]] = {}
    for e in entities:
        version = entity_version(e)
        if version is not None:
            by_lineage.setdefault(lineage(e), []).append((version, e))
    chains: list[tuple[int, int]] = []
    for versions in by_lineage.values():
        versions.sort()
        chains.extend((index[x], index[y]) for (_, x), (_, y) in zip(versions, versions[1:], strict=False))
    precision = 1.0 / options.prior_sigma ** 2
    prior_diag = np.full(E + F, precision)
    for i, j in chains:
        prior_diag[i] += options.smoothness
        prior_diag[j] += options.smoothness
    prior = None
    if E + F <= options.dense_limit:
        prior = np.diag(np.full(E + F, precision))
        for i, j in chains:
            prior[i, i] += options.smoothness
            prior[j, j] += options.smoothness
            prior[i, j] -= options.smoothness
            prior[j, i] -= options.smoothness
    keys = [(a, b) for a, b, _ in rows]
    return _Design(
        entities=entities, index=index, num_teams=num_teams, keys=keys, row_of={k: r for r, k in enumerate(keys)},
        nz_row=np.asarray(nz_row, np.int64), nz_col=np.asarray(nz_col, np.int64), nz_val=np.asarray(nz_val),
        sw=np.asarray([s.sw for _, _, s in rows], np.float64), swy=np.asarray([s.swy for _, _, s in rows], np.float64),
        prior=prior, prior_diag=prior_diag, chains=chains,
    )


def _eta(d: _Design, params: np.ndarray) -> np.ndarray:
    return np.bincount(d.nz_row, weights=d.nz_val * params[d.nz_col], minlength=len(d.keys))


def _prior_times(d: _Design, v: np.ndarray, options: BTOptions) -> np.ndarray:
    if d.prior is not None:
        return d.prior @ v
    out = v / options.prior_sigma ** 2
    for i, j in d.chains:
        diff = options.smoothness * (v[i] - v[j])
        out[i] += diff
        out[j] -= diff
    return out


def _log_post(d: _Design, params: np.ndarray, sw: np.ndarray, swy: np.ndarray, options: BTOptions) -> float:
    eta = _eta(d, params)
    return float(np.sum(swy * eta - sw * np.logaddexp(0.0, eta))) - 0.5 * float(params @ _prior_times(d, params,
                                                                                                       options))


def _gradient(d: _Design, params: np.ndarray, sw: np.ndarray, swy: np.ndarray,
              options: BTOptions) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of the log posterior and the per-row curvature weights ``sw p (1 - p)``."""
    p = _sigmoid(_eta(d, params))
    grad = np.bincount(d.nz_col, weights=d.nz_val * (swy - sw * p)[d.nz_row], minlength=d.num_params)
    return grad - _prior_times(d, params, options), sw * p * (1.0 - p)


def _dense_hessian(d: _Design, w: np.ndarray) -> np.ndarray:
    """Negative Hessian of the log posterior: ``sum_r w_r z_r z_r^T`` + the prior precision."""
    H = d.prior.copy()
    order = np.argsort(d.nz_row, kind="stable")
    rows, cols, vals = d.nz_row[order], d.nz_col[order], d.nz_val[order]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    sizes = ends - starts
    for k in np.unique(sizes):                    # rows grouped by their number of non-zeros
        sel = starts[sizes == k]
        idx = sel[:, None] + np.arange(k)[None, :]
        c, v = cols[idx], vals[idx]
        np.add.at(H, (np.repeat(c[:, :, None], k, 2), np.repeat(c[:, None, :], k, 1)),
                  w[rows[sel]][:, None, None] * v[:, :, None] * v[:, None, :])
    return H


def _hess_times(d: _Design, w: np.ndarray, v: np.ndarray, options: BTOptions) -> np.ndarray:
    out = np.bincount(d.nz_col, weights=d.nz_val * (w * _eta(d, v))[d.nz_row], minlength=d.num_params)
    return out + _prior_times(d, v, options)


def _hess_diag(d: _Design, w: np.ndarray) -> np.ndarray:
    return np.bincount(d.nz_col, weights=d.nz_val ** 2 * w[d.nz_row], minlength=d.num_params) + d.prior_diag


def _cg(apply: Callable[[np.ndarray], np.ndarray], b: np.ndarray, diag: np.ndarray, tol: float = 1e-10,
        max_iter: int = 500) -> np.ndarray:
    """Preconditioned conjugate gradients for ``apply(x) = b`` (``apply`` symmetric positive definite)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    bnorm = max(float(np.linalg.norm(b)), 1e-300)
    for _ in range(max_iter):
        if float(np.linalg.norm(r)) <= tol * bnorm:
            break
        ap = apply(p)
        alpha = rz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        z = r / diag
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _newton(d: _Design, sw: np.ndarray, swy: np.ndarray, start: np.ndarray,
            options: BTOptions) -> tuple[np.ndarray, int, bool]:
    params = start.copy()
    value = _log_post(d, params, sw, swy, options)
    for iteration in range(1, options.max_iter + 1):
        grad, w = _gradient(d, params, sw, swy, options)
        if d.prior is not None:
            step = np.linalg.solve(_dense_hessian(d, w), grad)
        else:
            step = _cg(lambda v, w=w: _hess_times(d, w, v, options), grad, _hess_diag(d, w))
        t = 1.0
        while True:                               # backtracking: the log posterior never decreases
            candidate = params + t * step
            new_value = _log_post(d, candidate, sw, swy, options)
            if new_value >= value - 1e-12 * max(1.0, abs(value)) or t < 1e-6:
                break
            t *= 0.5
        params, value = candidate, new_value
        if float(np.max(np.abs(t * step))) < options.tol:
            return params, iteration, True
    return params, options.max_iter, False


# ---------------------------------------------------------------------------
# groups, components, anchors
# ---------------------------------------------------------------------------


def _cells(rows: Sequence[tuple[TeamKey, TeamKey, PairStat]],
           entities: Sequence[EntityKey]) -> list[tuple[str, int, list[EntityKey]]]:
    """``(group label, component, entities)`` of every cell (module docstring), sorted."""
    known = set(entities)
    roles, links = _UnionFind(), _UnionFind()
    entity_roles: dict[EntityKey, set[str]] = {}
    for a, b, _ in rows:
        for key in (a, b):
            for entity, role in key.members:
                if entity in known:
                    entity_roles.setdefault(entity, set()).add(role)
        active = list(coefficients(a, b))
        for entity in active[1:]:
            links.union(active[0], entity)
    for played in entity_roles.values():
        first = min(played)
        for role in played:
            roles.union(first, role)
    group_roles: dict[str, set[str]] = {}
    for played in entity_roles.values():
        group_roles.setdefault(roles.find(min(played)), set()).update(played)
    labels = {root: "+".join(sorted(played)) for root, played in group_roles.items()}
    component_ids: dict[EntityKey, int] = {}
    for entity in sorted(entities):
        component_ids.setdefault(links.find(entity), len(component_ids))
    cells: dict[tuple[str, int], list[EntityKey]] = {}
    for entity in entities:
        label = labels[roles.find(min(entity_roles[entity]))]
        cells.setdefault((label, component_ids[links.find(entity)]), []).append(entity)
    return [(label, comp, sorted(members)) for (label, comp), members in sorted(cells.items())]


def _lowest_version(agent: str, entities: Iterable[EntityKey]) -> EntityKey | None:
    versions = [(v, e) for e in entities if lineage(e) == agent and (v := entity_version(e)) is not None]
    return min(versions)[1] if versions else None


def _anchor(cell: Sequence[EntityKey], options: BTOptions, fixed_agents: Collection[str], random_bot: str | None,
            agent_order: Sequence[str]) -> tuple[str, list[EntityKey], str, str | None]:
    """``(anchor, anchor members, rule, note)`` of one cell (module docstring)."""
    present = set(cell)
    notes: list[str] = []
    if options.anchor is not None:
        name = options.anchor
        if name in present:
            return name, [name], "config", None
        lowest = _lowest_version(name, cell)
        if lowest is not None:
            return lowest, [lowest], "config", None
        notes.append(f"ratings.anchor {name!r} did not play here")
    if random_bot is not None:
        if random_bot in present:
            return random_bot, [random_bot], "random_bot", "; ".join(notes) or None
        notes.append(f"RandomBot agent {random_bot!r} did not play here")
    note = "; ".join(notes) or None
    fixed = [f for f in fixed_agents if f in present]
    if fixed:
        return MEAN_OF_FIXED, fixed, "mean_of_fixed", note
    for agent in agent_order:
        if agent in fixed_agents:
            continue
        lowest = _lowest_version(agent, cell)
        if lowest is not None:
            return lowest, [lowest], "first_trainable", note
    first = sorted(cell)[0]
    return first, [first], "first", note


def _contrast(design: _Design, group: BTGroup) -> np.ndarray:
    c = np.zeros(design.num_params)
    for entity in group.anchor_members:
        c[design.index[entity]] += 1.0 / len(group.anchor_members)
    return c


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def fit(table: ComparisonTable, layout: str, *, sources: Collection[str], options: BTOptions = DEFAULT_OPTIONS,
        fixed_agents: Collection[str] = (), random_bot: str | None = None, agent_order: Sequence[str] = (),
        warm: BTFit | None = None) -> BTFit:
    """MAP Bradley-Terry fit of one layout over ``sources`` (module docstring); an empty fit without comparisons.

    ``fixed_agents``: the config's scripted / frozen agents; ``random_bot``: the config's first ``RandomBot`` agent;
    ``agent_order``: every agent in config order (the "first trainable agent" anchor rule)."""
    rows = table.pairs(layout, sources)
    design = _build_design(rows, options) if rows else None
    if design is None:
        return BTFit(layout=layout, groups=[], rating={}, se={}, bias={}, bias_se={}, games={}, iterations=0,
                     converged=True, options=options)
    E, T = len(design.entities), design.num_teams
    start = np.zeros(design.num_params)
    if warm is not None and warm.design is not None:
        for entity, i in design.index.items():
            j = warm.design.index.get(entity)
            if j is not None:
                start[i] = warm.params[j]
        if warm.design.num_teams == T:
            start[E:] = warm.params[len(warm.design.entities):]
    params, iterations, converged = _newton(design, design.sw, design.swy, start, options)
    _, w = _gradient(design, params, design.sw, design.swy, options)
    hessian = _dense_hessian(design, w) if design.prior is not None else None
    groups = []
    for label, component, members in _cells(rows, design.entities):
        anchor, anchor_members, rule, note = _anchor(members, options, fixed_agents, random_bot, agent_order)
        groups.append(BTGroup(layout, label, component, members, anchor, anchor_members, rule, note))
    if hessian is not None:
        cov = np.linalg.inv(hessian)
        cov_diag = np.diag(cov)
    else:
        diag_inv = 1.0 / _hess_diag(design, w)
    rating, se = {}, {}
    for group in groups:
        c = _contrast(design, group)
        cols = np.array([design.index[e] for e in group.entities])
        base = float(c @ params)
        if hessian is not None:
            cov_c = cov @ c
            var = cov_diag[cols] - 2.0 * cov_c[cols] + float(c @ cov_c)
        else:
            var = diag_inv[cols] - 2.0 * c[cols] * diag_inv[cols] + float(np.sum(c * c * diag_inv))
        for k, entity in enumerate(group.entities):
            rating[entity] = (float(params[cols[k]]) - base) * ELO_PER_LOGIT
            se[entity] = math.sqrt(max(float(var[k]), 0.0)) * ELO_PER_LOGIT
    C = np.vstack([np.eye(T - 1), -np.ones((1, T - 1))])         # T >= 2: every row pairs two teams
    bias_var = (np.einsum("ti,ij,tj->t", C, cov[E:, E:], C) if hessian is not None
                else (C * C) @ diag_inv[E:])
    bias_logits = C @ params[E:]
    games = {e: g for e, g in table.entity_games(layout, sources).items() if e in design.index}
    return BTFit(
        layout=layout, groups=groups, rating=rating, se=se,
        bias={t: float(bias_logits[t]) * ELO_PER_LOGIT for t in range(T)},
        bias_se={t: math.sqrt(max(float(bias_var[t]), 0.0)) * ELO_PER_LOGIT for t in range(T)},
        games=games, iterations=iterations, converged=converged, approximate=hessian is None, params=params,
        design=design, options=options, hessian=hessian,
    )


def _summaries(fit_: BTFit, samples: np.ndarray, *, top_k: int, quantile: float,
               ci_from_samples: bool) -> Uncertainty:
    """Probabilities, intervals and lower bounds from parameter samples ``[S, P]`` (logits)."""
    design = fit_.design
    ci: dict = {}
    lcb: dict = {}
    pbl: dict = {}
    pbg: dict = {}
    ptk: dict = {}
    ptkg: dict = {}
    contrasts: dict = {}
    for group in fit_.groups:
        cols = np.array([design.index[e] for e in group.entities])
        anchored = (samples[:, cols] - (samples @ _contrast(design, group))[:, None]) * ELO_PER_LOGIT   # [S, n]
        best = np.argmax(anchored, axis=1)
        ranks = np.argsort(np.argsort(-anchored, axis=1), axis=1) if top_k > 0 else None
        for j, entity in enumerate(group.entities):
            pbg[entity] = float(np.mean(best == j))
            if ci_from_samples:
                lo, hi = np.quantile(anchored[:, j], [0.025, 0.975])
                ci[entity] = (float(lo), float(hi))
            else:
                half = Z_95 * fit_.se[entity]
                ci[entity] = (fit_.rating[entity] - half, fit_.rating[entity] + half)
            if ranks is not None:
                ptkg[entity] = float(np.mean(ranks[:, j] < top_k))
        by_lineage: dict[str, list[int]] = {}
        for j, entity in enumerate(group.entities):
            by_lineage.setdefault(lineage(entity), []).append(j)
        for columns in by_lineage.values():
            sub = anchored[:, columns]
            contrast = sub - sub.mean(axis=1, keepdims=True)
            best_l = np.argmax(sub, axis=1)
            ranks_l = np.argsort(np.argsort(-sub, axis=1), axis=1) if top_k > 0 else None
            for k, j in enumerate(columns):
                entity = group.entities[j]
                pbl[entity] = float(np.mean(best_l == k))
                lcb[entity] = float(np.quantile(contrast[:, k], quantile))
                contrasts[entity] = contrast[:, k].astype(np.float32)
                if ranks_l is not None:
                    ptk[entity] = float(np.mean(ranks_l[:, k] < top_k))
    return Uncertainty(ci95=ci, lcb=lcb, p_best_lineage=pbl, p_best_group=pbg, p_top_k=ptk, p_top_k_group=ptkg,
                       contrast_samples=contrasts)


def _empty_uncertainty() -> Uncertainty:
    return Uncertainty({}, {}, {}, {}, {}, {})


def laplace(fit_: BTFit, *, samples: int = 1000, rng: np.random.Generator, top_k: int = 0,
            quantile: float = 0.1) -> Uncertainty:
    """Fast uncertainty (module docstring): normal intervals from the fit's standard errors; probabilities and
    ``lcb`` from ``samples`` draws of ``N(theta_hat, H^-1)``."""
    if fit_.design is None:
        return _empty_uncertainty()
    z = rng.standard_normal((samples, fit_.design.num_params))
    if fit_.hessian is not None:
        L = np.linalg.cholesky(fit_.hessian)
        delta = np.linalg.solve(L.T, z.T).T              # cov(delta) = (L L^T)^-1 = H^-1
    else:
        _, w = _gradient(fit_.design, fit_.params, fit_.design.sw, fit_.design.swy, fit_.options)
        delta = z / np.sqrt(_hess_diag(fit_.design, w))
    return _summaries(fit_, fit_.params[None, :] + delta, top_k=top_k, quantile=quantile, ci_from_samples=False)


def _contribution_arrays(design: _Design, items: Iterable[tuple[int, TeamKey, TeamKey, float, float]]):
    unit, row, w, wy = [], [], [], []
    for u, a, b, weight, outcome in items:
        if a.team > b.team:
            a, b, outcome = b, a, 1.0 - outcome
        r = design.row_of.get((a, b))
        if r is None:
            raise ValueError(f"bootstrap data has a team pair the fit does not have: {a} vs {b}")
        unit.append(u)
        row.append(r)
        w.append(weight)
        wy.append(weight * outcome)
    return (np.asarray(unit, np.int64), np.asarray(row, np.int64), np.asarray(w, np.float64),
            np.asarray(wy, np.float64))


def bootstrap(fit_: BTFit, data: BootstrapData, *, samples: int, rng: np.random.Generator, top_k: int = 0,
              quantile: float = 0.1) -> Uncertainty:
    """Uncertainty from ``samples`` refits on resampled ``data`` (``BootstrapData``), each warm-started at the fit;
    percentile intervals. The cells and anchors are those of ``fit_``; ``data`` must decompose the table it was fit
    on (a team pair the fit does not know is a ValueError)."""
    design = fit_.design
    if design is None:
        return _empty_uncertainty()
    c_rows, c_w, c_wins, c_losses = [], [], [], []
    for a, b, stat in data.counts:
        if stat.n <= 0:
            continue
        w = stat.sw / stat.n
        wins = stat.swy / w if w > 0 else 0.0           # in matches: a draw counts half
        if a.team > b.team:                              # canonical order: the outcome becomes b's
            a, b, wins = b, a, stat.n - wins
        r = design.row_of.get((a, b))
        if r is None:
            raise ValueError(f"bootstrap data has a team pair the fit does not have: {a} vs {b}")
        wins = min(max(wins, 0.0), float(stat.n))
        c_rows.append(r)
        c_w.append(w)
        c_wins.append(wins)
        c_losses.append(stat.n - wins)
    c_rows_a, c_w_a = np.asarray(c_rows, np.int64), np.asarray(c_w, np.float64)
    c_wins_a, c_losses_a = np.asarray(c_wins, np.float64), np.asarray(c_losses, np.float64)
    m_unit, m_row, m_w, m_wy = _contribution_arrays(
        design, ((m, a, b, w, y) for m, contribs in enumerate(data.matches) for a, b, w, y in contribs))
    strata: list[tuple[int, int]] = []          # (first unit, number of keys) per stratum
    units: list[list[Contribution]] = []
    for _, by_key in sorted(data.blocks.items()):
        keys = sorted(by_key)
        strata.append((len(units), len(keys)))
        units.extend(by_key[k] for k in keys)
    b_unit, b_row, b_w, b_wy = _contribution_arrays(
        design, ((u, a, b, w, y) for u, contribs in enumerate(units) for a, b, w, y in contribs))
    R = len(design.keys)
    replicates = np.empty((samples, design.num_params))
    for s in range(samples):
        sw, swy = np.zeros(R), np.zeros(R)
        if len(c_rows_a):
            wins = rng.poisson(c_wins_a)
            losses = rng.poisson(c_losses_a)
            np.add.at(sw, c_rows_a, c_w_a * (wins + losses))
            np.add.at(swy, c_rows_a, c_w_a * wins)
        if len(m_row):
            weight = rng.poisson(1.0, size=len(data.matches))[m_unit]
            np.add.at(sw, m_row, m_w * weight)
            np.add.at(swy, m_row, m_wy * weight)
        if len(b_row):
            unit_weight = np.zeros(len(units))
            for first, k in strata:
                unit_weight[first:first + k] = rng.multinomial(k, np.full(k, 1.0 / k))
            weight = unit_weight[b_unit]
            np.add.at(sw, b_row, b_w * weight)
            np.add.at(swy, b_row, b_wy * weight)
        replicates[s], _, _ = _newton(design, sw, swy, fit_.params, fit_.options)
    return _summaries(fit_, replicates, top_k=top_k, quantile=quantile, ci_from_samples=True)


def chi2_sf(x: float, dof: int) -> float:
    """Survival function of chi^2(dof) at ``x``: the regularized upper incomplete gamma ``Q(dof / 2, x / 2)``
    (series below ``a + 1``, Lentz's continued fraction above; Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    a, z = 0.5 * dof, 0.5 * x
    log_front = -z + a * math.log(z) - math.lgamma(a)
    if z < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_front))
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(log_front) * h)


def transitivity_test(table: ComparisonTable, fit_: BTFit, *, sources: Collection[str],
                      min_games: int = 20) -> TransitivityResult:
    """Deviance ``G2 = 2 (LL_pairs - LL_BT)`` over the team pairs with at least ``min_games`` matches, against
    ``chi^2`` with ``pairs - (entities - cells) - (T - 1)`` degrees of freedom; flagged at p < 0.01 (spec block 2).
    With fractional weights (FFA) and draws the reference distribution is approximate."""
    design = fit_.design
    if design is None:
        return TransitivityResult(0.0, 0, None, False, 0, [])
    eta = _eta(design, fit_.params)
    g2 = 0.0
    residuals: list[tuple[TeamKey, TeamKey, float]] = []
    involved: set[EntityKey] = set()
    tested = 0
    for a, b, stat in table.pairs(fit_.layout, sources):
        r = design.row_of.get((a, b))
        if stat.n < min_games or stat.sw <= 0 or r is None:
            continue
        tested += 1
        involved.update(coefficients(a, b))
        p = min(max(float(_sigmoid(eta[r])), 1e-12), 1.0 - 1e-12)
        y, n = stat.swy, stat.sw
        dev = 0.0
        if y > 0:
            dev += y * math.log((y / n) / p)
        if n - y > 0:
            dev += (n - y) * math.log(((n - y) / n) / (1.0 - p))
        dev = max(2.0 * dev, 0.0)
        g2 += dev
        residuals.append((a, b, math.copysign(math.sqrt(dev), y / n - p)))
    cells = {(g.group, g.component) for g in fit_.groups if set(g.entities) & involved}
    dof = tested - (len(involved) - len(cells)) - (design.num_teams - 1)
    p_value = chi2_sf(g2, dof) if tested and dof > 0 else None
    residuals.sort(key=lambda item: -abs(item[2]))
    return TransitivityResult(g2=g2, dof=dof, p_value=p_value,
                              flagged=p_value is not None and p_value < TRANSITIVITY_P, pairs_tested=tested,
                              worst_pairs=residuals[:5])


def _score_cell(n: int, total: float, total_sq: float) -> ScoreCell:
    mean = total / n
    var = (total_sq - n * mean * mean) / (n - 1) if n >= 2 else 0.0
    half = Z_95 * math.sqrt(max(var, 0.0) / n) if n >= 2 else 0.0
    return ScoreCell(n=n, mean=mean, ci_low=mean - half, ci_high=mean + half)


def score_summary(table: ComparisonTable, layout: str, *, sources: Collection[str]) -> ScoreSummary:
    """One-team layouts (spec block 2): mean team score with a 95 % normal CI per entity (every match it sat in) and
    per team composition (``"a+a+b"``: sorted member entities)."""
    per_entity: dict[EntityKey, list[float]] = {}
    per_comp: dict[str, list[float]] = {}
    homogeneous: dict[str, bool] = {}
    for key, stat in table.scores(layout, sources):
        members = sorted(key.entities())
        comp = "+".join(members)
        homogeneous[comp] = len(set(members)) == 1
        for cell in [per_comp.setdefault(comp, [0, 0.0, 0.0])] + [per_entity.setdefault(e, [0, 0.0, 0.0])
                                                                  for e in sorted(set(members))]:
            cell[0] += stat.n
            cell[1] += stat.total
            cell[2] += stat.total_sq
    return ScoreSummary(layout=layout,
                        entities={e: _score_cell(*v) for e, v in sorted(per_entity.items())},
                        compositions={c: _score_cell(*v) for c, v in sorted(per_comp.items())},
                        homogeneous=homogeneous)
```

How the pieces fit (for the reviewer):
- **Design.** Each table row (team pair `a`, `b`) becomes a sparse row: the net member coefficients `coefficients(a, b)` (an entity on both teams cancels; self-play rows have only bias entries) plus the bias entries `C[a.team] − C[b.team]`. The log posterior is `Σ_r [swy_r η_r − sw_r log(1 + e^{η_r})] − ½ θᵀPθ`; gradient and Hessian come from `np.bincount` / `np.add.at` over the non-zeros.
- **Uncertainty of an anchored rating** is `uᵀ Σ u` with `u = e_entity − c` (`c` the anchor's contrast vector: one-hot, or `1/|F|` over the fixed agents), vectorized per cell as `Σ_ee − 2 (Σc)_e + cᵀΣc`.
- **Laplace samples** solve `Lᵀ δ = z` with `H = L Lᵀ` (`cov δ = H⁻¹`), no explicit inverse; the iterative path samples with `1 / diag(H)`.
- **Bootstrap counts rows.** A compacted row keeps `n`, `Σw`, `Σw·y`; per replicate wins ~ Poisson(Σw·y / w) and losses ~ Poisson(n − Σw·y / w) with `w = Σw / n` (exact without draws). This is the spec's "Poisson weights by rows" for compacted data, under which the correlation of the pairs of one FFA match is lost (spec block 1); detailed matches keep it (one Poisson weight per match).
- **G² degrees of freedom:** `pairs − (entities − cells) − (T − 1)` over the pairs with `≥ min_games` matches; `p_value` None when that is not positive.
- **Determinism.** Union-find roots are the smallest element (labels independent of set order and of `PYTHONHASHSEED`); `ComparisonTable.pairs` is sorted, so the design is the same on every run.

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_bt.py -q -rw`
Expected: all pass (about 7 s; the six coverage tests take most of it).

- [ ] **Step 6: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/arena/bt.py tests/game_helpers.py tests/unit/test_sp4_bt.py
git commit -m "feat: Bradley-Terry ratings of snapshots (side bias, role groups, anchors, Laplace, bootstrap, G2 test)"
git push
```

---

### Task T2.2: Live ratings: refit thread, `ratings.json` `bt`, compact `ratings` record, console and WandB, `ratings` config, BLAS threads

Spec block 2 ("Живой рейтинг", "Конфиг") and block 7.3. A `LiveRatings` thread in the training main process refits Bradley–Terry every `ratings.refit_interval_sec` from a copy of the in-memory `ComparisonTable` the journal writer feeds (T1.3), with Laplace uncertainty, the G² test and score summaries, and publishes an immutable `RatingsSnapshot`; it never touches the coordinator, the storage or the hub. At each metrics tick the hub writes: `ratings.json = {"env_steps", "layouts", "bt"}` (full PFSP tables, `bt_section`); a compact `ratings` record (each layout's PFSP table replaced by `pfsp_summary`; `bt` = `bt_summary` per trainable agent); the console field `bt <rating> ± <ci>`; numeric WandB rows `ratings/bt/...`. The ratings snapshot of the coordinator is built only at a tick (the hub takes a callable). At the end of the run the thread stops and, if the last refit was fast, refits once more on the final data. The `ratings` config section is added (`anchor` checked by `validate`, which also prints the journal and rating settings). `colosseum train` loads numpy's BLAS with one thread in the main process only. The refit cost is measured for T8.2.

**Files:**
- Create: `src/colosseum/arena/live.py`
- Modify: `src/colosseum/core/config.py` (`RatingsConfig`, `ColosseumConfig.ratings`)
- Modify: `src/colosseum/core/validation.py` (`_check_ratings`)
- Modify: `src/colosseum/league/pfsp.py` (`pfsp_summary`, `compact_ratings`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`ratings_snapshot(compact=...)`)
- Modify: `src/colosseum/metrics/hub.py` (callable `ratings`, `bt`, compact record, `ratings.json` `bt`, console, WandB)
- Modify: `src/colosseum/metrics/console.py` (`format_line(bt=...)`)
- Modify: `src/colosseum/launcher.py` (`_open_live_ratings`, `_live_snapshot`, `_finish_live_ratings`, tick and close wiring)
- Modify: `src/colosseum/utils/process.py` (`BLAS_THREAD_VARS`, `limit_main_blas_threads`), `src/colosseum/cli.py` (`train` calls it first)
- Modify: `scripts/bench_throughput.py` (`_make_config`: `ratings`)
- Modify (the SP2 tests pinning the ratings keys, spec §8): `tests/integration/test_sp2_metrics_outputs.py`, `tests/integration/test_sp2_game_runs.py`
- Test: `tests/unit/test_sp4_live_ratings.py`, `tests/integration/test_sp4_live_ratings_run.py`

**Interfaces:**
- Consumes: `fit`, `laplace`, `transitivity_test`, `score_summary`, `BTOptions`, `BTFit`, `Uncertainty` (T2.1); `ComparisonTable.copy` / `version`, `RUN_SOURCES` (T1.2, T1.3); `Launcher._comparison_table` (T1.3); `PfspStats.snapshot()` shapes (`league/pfsp.py:103`).
- Produces (contract T2.2, plus additions marked *):
  - `RatingsSnapshot(version, env_steps, fits, uncertainty, scores, *transitivity, *refit_sec)`;
  - `LiveRatings(table, *, interval_sec, options=DEFAULT_OPTIONS, fixed_agents=(), random_bot=None, *agent_order=(), top_k=0, quantile=0.1, env_steps_fn, *samples=LAPLACE_SAMPLES, *seed=0)` with `start`, `stop(timeout)`, `latest()`, *`refit_now()`, *`last_refit_sec`;
  - `bt_section(snapshot)`, `bt_summary(snapshot, agents)`, *`console_bt(agent_summary)`, *`compute_snapshot(table, *, version, env_steps, ...)`, *`random_bot_agent(config)`, *`LAPLACE_SAMPLES = 1000`, *`FINAL_REFIT_MAX_SEC = 2.0`;
  - `RatingsConfig(anchor=None, refit_interval_sec=60.0, bootstrap_samples=200)`, `ColosseumConfig.ratings`;
  - `Coordinator.ratings_snapshot(*, compact=False)`; *`colosseum.league.pfsp.pfsp_summary(table)`, *`compact_ratings(layouts)`;
  - *`MetricsHub.maybe_tick(..., ratings: dict | Callable[[], dict], ..., bt=None)`, *`MetricsHub.close(..., bt=None)`; *`ConsoleReporter.format_line(..., bt=None)`;
  - *`colosseum.utils.process.BLAS_THREAD_VARS`, *`limit_main_blas_threads() -> bool`.
  - `ratings.json` = `{"env_steps", "layouts", "bt"}`; `bt[layout][<role group>]` = rows `{entity, lineage, version, rating, se, ci_low, ci_high, lcb, games, p_best, p_best_lineage, [p_top_k], component, anchor}` sorted by rating; per-layout extras under keys starting with `@` (`@bias`, `@anchors`, `@transitivity`, `@fit`, `@score`). The `ratings` record of `metrics.jsonl` = `{"env_steps", "layouts" (with pfsp_summary), "bt"?}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_live_ratings.py`:
```python
"""Live ratings (SP4 T2.2, spec blocks 2 and 7.3): the refit thread publishes snapshots and skips an unchanged
table, survives a failing refit, logs anchor notes once; the bt section of ratings.json, the compact ratings record,
the console and WandB fields; the ratings config section and its validate checks; one BLAS thread in the training
main process."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from click.testing import CliRunner

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.live import (
    LiveRatings,
    bt_section,
    bt_summary,
    compute_snapshot,
    console_bt,
    random_bot_agent,
)
from colosseum.core.config import ColosseumConfig, RatingsConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.league.pfsp import compact_ratings, pfsp_summary
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from game_helpers import (
    bt_table,
    make_coordinator,
    make_match_result,
    make_test_config,
    scripted_agent,
    synthetic_bt_matches,
)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _league_table(n: int = 600, seed: int = 0) -> ComparisonTable:
    strengths = {"random": -1.0, "greedy": 0.3, "main@v0": -0.5, "main@v100": 0.2, "main@v200": 0.6}
    return bt_table(synthetic_bt_matches(strengths, n_matches=n, rng=np.random.default_rng(seed), bias=[0.2, -0.2]),
                    interval=100)


def _snapshot(table: ComparisonTable):
    return compute_snapshot(table, version=1, env_steps=1234, fixed_agents=("random", "greedy"), random_bot="random",
                            agent_order=("main", "random", "greedy"), top_k=2, rng=np.random.default_rng(0),
                            samples=300)


def test_the_thread_publishes_snapshots_and_skips_an_unchanged_table():
    table = _league_table()
    steps = [500]
    live = LiveRatings(table, interval_sec=0.05, fixed_agents=("random", "greedy"), random_bot="random",
                       env_steps_fn=lambda: steps[0], samples=200)
    assert live.latest() is None and live.last_refit_sec is None
    live.start()
    try:
        assert _wait(lambda: live.latest() is not None)
        first = live.latest()
        assert first.version == 1 and first.env_steps == 500 and set(first.fits) == {"2p"}
        time.sleep(0.3)
        assert live.latest() is first                      # unchanged table: no new refit
        steps[0] = 900
        table.add_result(make_match_result([(0, "main", "ckpt_v200"), (1, "random", "fixed")]), [200, None])
        assert _wait(lambda: live.latest().version == 2)
        assert live.latest().env_steps == 900 and live.last_refit_sec >= 0.0
    finally:
        live.stop(timeout=5.0)


def test_a_failing_refit_is_logged_once_and_the_thread_goes_on(caplog, monkeypatch):
    table = _league_table()
    calls = []
    original = table.copy

    def flaky_copy():
        calls.append(1)
        if len(calls) <= 3:
            raise RuntimeError("table broken")
        return original()

    monkeypatch.setattr(table, "copy", flaky_copy)
    live = LiveRatings(table, interval_sec=0.02, env_steps_fn=lambda: 0, samples=100)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.live"):
        live.start()
        try:
            assert _wait(lambda: live.latest() is not None)
        finally:
            live.stop(timeout=5.0)
    errors = [r for r in caplog.records if "refit failed" in r.getMessage()]
    assert len(errors) == 3 and errors[0].exc_info and not errors[1].exc_info   # a traceback the first time only


def test_anchor_notes_are_logged_once(caplog):
    live = LiveRatings(_league_table(), interval_sec=3600.0, random_bot="nobot", env_steps_fn=lambda: 0, samples=50)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.live"):
        live.refit_now()
        live._table.add_result(make_match_result([(0, "main", "ckpt_v0"), (1, "random", "fixed")]), [0, None])
        live.refit_now()
    notes = [r.getMessage() for r in caplog.records if "did not play here" in r.getMessage()]
    assert len(notes) == 1 and "RandomBot agent 'nobot'" in notes[0]


def test_the_bt_section_of_ratings_json():
    table = _league_table()
    table.add_result(make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[3.0], layout="coop"),
                     [None, None])
    section = bt_section(_snapshot(table))
    assert set(section) == {"2p", "coop"}
    rows = section["2p"]["player"]
    assert [r["rating"] for r in rows] == sorted((r["rating"] for r in rows), reverse=True)
    assert {r["entity"] for r in rows} == {"random", "greedy", "main@v0", "main@v100", "main@v200"}
    (anchor,) = [r for r in rows if r["anchor"]]
    assert anchor["entity"] == "random" and anchor["rating"] == 0.0
    main = next(r for r in rows if r["entity"] == "main@v200")
    assert set(main) >= {"lineage", "version", "se", "ci_low", "ci_high", "lcb", "games", "p_best", "p_best_lineage",
                         "p_top_k", "component"}
    assert main["lineage"] == "main" and main["version"] == 200 and main["ci_low"] < main["rating"] < main["ci_high"]
    assert set(section["2p"]["@bias"]) == {"0", "1"} and section["2p"]["@anchors"]["player/0"]["rule"] == "random_bot"
    assert "p_value" in section["2p"]["@transitivity"] and section["2p"]["@fit"]["converged"]
    assert section["coop"]["@score"]["compositions"][0]["composition"] == "a+b"
    json.dumps(section)                                    # JSON-ready
    assert bt_section(None) == {}


def test_the_compact_bt_summary_and_the_console_field():
    summary = bt_summary(_snapshot(_league_table()), ["main", "absent"])
    assert set(summary) == {"main"}
    cells = summary["main"]["2p"]
    assert cells["latest"]["entity"] == "main@v200"
    assert cells["best"]["rating"] == max(cells["best"]["rating"], cells["latest"]["rating"])
    rating, half = console_bt(summary["main"])
    assert rating == cells["latest"]["rating"] and half == pytest.approx(
        (cells["latest"]["ci_high"] - cells["latest"]["ci_low"]) / 2)
    assert console_bt({}) is None and bt_summary(None, ["main"]) == {}


def test_pfsp_tables_are_summarized_in_the_records():
    table = {"a": {"a@ckpt_v1": {"score": 0.25, "games": 4.0}, "rnd@fixed": {"score": 0.75, "games": 2.0}},
             "b": {}}
    assert pfsp_summary(table) == {"a": {"players": 2, "min_score": 0.25, "max_score": 0.75},
                                   "b": {"players": 0, "min_score": None, "max_score": None}}
    compact = compact_ratings({"2p": {"elo": {"a": 1200.0}, "pfsp": table}})
    assert compact == {"2p": {"elo": {"a": 1200.0}, "pfsp_summary": pfsp_summary(table)}}


def test_the_coordinator_gives_the_compact_snapshot(tmp_path):
    coordinator = make_coordinator(make_test_config("turns"), tmp_path / "ckpt")
    full, compact = coordinator.ratings_snapshot(), coordinator.ratings_snapshot(compact=True)
    assert "pfsp" in full["2p"] and "pfsp" not in compact["2p"] and "pfsp_summary" in compact["2p"]


def test_the_hub_writes_compact_records_bt_in_ratings_json_console_and_wandb(tmp_path, caplog):
    rows = []

    class FakeWandB:
        def log_global(self, row, step):
            rows.append(row)

        def log_train(self, agent_id, values, step):
            pass

    clock = [0.0]
    built = []
    full = {"2p": {"elo": {"main": 1200.0}, "win_rates": {}, "games": {}, "wr_vs_past": {}, "past_games": {},
                   "scores": {}, "cross_play": {}, "role_win_rates": {},
                   "pfsp": {"main": {"random@fixed": {"score": 0.4, "games": 3.0}}}}}

    def ratings():
        built.append(1)
        return full

    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["main"], total_timesteps=100, log_interval=1, console_interval_sec=10.0,
                     wandb_logger=FakeWandB(), clock=lambda: clock[0])
    snapshot = _snapshot(_league_table())
    for _ in range(5):
        hub.maybe_tick(env_steps=10, ratings=ratings, queue_depths={}, bt=snapshot)    # not due yet
    assert built == []                                     # built only at a tick (spec block 7.3)
    clock[0] = 11.0
    with caplog.at_level(logging.INFO, logger="colosseum.progress"):
        hub.close(env_steps=20, ratings=ratings, queue_depths={}, bt=snapshot)
    assert built == [1]
    record = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
              if '"ratings"' in line][-1]
    assert "pfsp" not in record["layouts"]["2p"] and record["layouts"]["2p"]["pfsp_summary"]["main"]["players"] == 1
    assert record["bt"]["main"]["2p"]["latest"]["entity"] == "main@v200"
    written = json.loads((tmp_path / "ratings.json").read_text())
    assert set(written) == {"env_steps", "layouts", "bt"} and "pfsp" in written["layouts"]["2p"]
    assert written["bt"]["2p"]["player"][0]["rating"] >= written["bt"]["2p"]["player"][-1]["rating"]
    assert "| bt " in caplog.text and " ± " in caplog.text
    assert any(key.startswith("ratings/bt/main/2p/latest/rating") for key in rows[-1])
    assert not any("/pfsp" in key for key in rows[-1])


def test_without_a_snapshot_the_record_has_no_bt_and_ratings_json_an_empty_one(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json", agent_ids=["a"],
                     total_timesteps=10, log_interval=1, console_interval_sec=0.0)
    hub.close(env_steps=0, ratings={}, queue_depths={})
    record = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines() if '"ratings"' in line][0]
    assert "bt" not in record and json.loads((tmp_path / "r.json").read_text())["bt"] == {}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def test_the_ratings_section_its_validation_and_the_random_bot_rule():
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.ratings == RatingsConfig(anchor=None, refit_interval_sec=60.0, bootstrap_samples=200)
    for bad in ({"refit_interval_sec": 0}, {"bootstrap_samples": 10}, {"anchors": "x"}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(ratings=bad))
    league = make_test_config("turns", agents={"main": {}, "greedy": scripted_agent("game_helpers.ConstantBot"),
                                               "rnd": scripted_agent()})
    assert random_bot_agent(league) == "rnd"
    report = validate_config(league)
    assert any("anchor 'rnd' (the first RandomBot agent)" in line for line in report.lines)
    assert any(line.startswith("match log: on") for line in report.lines)
    with pytest.raises(ConfigError, match="ratings.anchor 'nobody' is not an agent"):
        validate_config(make_test_config("turns", ratings={"anchor": "nobody"}))


def test_train_limits_the_main_process_blas_before_numpy(monkeypatch):
    import colosseum.launcher
    import colosseum.utils.process as process

    calls = []
    monkeypatch.setattr(process, "limit_main_blas_threads", lambda: calls.append("blas") or False)
    monkeypatch.setattr(colosseum.launcher, "run_training", lambda *a, **k: calls.append("train") or 0)
    from colosseum.cli import main

    result = CliRunner().invoke(main, ["train", "-c", os.devnull])
    assert result.exit_code == 0 and calls == ["blas", "train"]


_BLAS_PROBE = r"""
import ctypes, glob, multiprocessing as mp, os, sys
from colosseum.utils.process import limit_main_blas_threads

def threads():
    import numpy
    libs = glob.glob(os.path.join(os.path.dirname(numpy.__file__), "..", "numpy.libs", "*openblas*.so*"))
    if not libs:
        return None
    lib = ctypes.CDLL(libs[0])
    for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
        if hasattr(lib, name):
            return getattr(lib, name)()
    return None

def child(q):
    q.put(threads())

if __name__ == "__main__":
    before = dict(os.environ)
    assert limit_main_blas_threads() is True
    assert dict(os.environ) == before, "the environment must be restored"
    assert limit_main_blas_threads() is False          # numpy is imported now
    mp.set_start_method("spawn")
    q = mp.Queue()
    p = mp.Process(target=child, args=(q,))
    p.start()
    print(threads(), q.get())
    p.join()
"""


def test_the_blas_limit_applies_to_this_process_only(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_BLAS_PROBE)
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    main_threads, child_threads = out.stdout.split()
    if main_threads == "None":
        pytest.skip("numpy does not use a bundled OpenBLAS here")
    assert int(main_threads) == 1
    if (os.cpu_count() or 1) > 1:
        assert int(child_threads) > 1                     # spawned children keep the default
```

Create `tests/integration/test_sp4_live_ratings_run.py`:
```python
"""Live ratings in a real training run (SP4 T2.2, spec block 2): ratings.json gets the bt section (the RandomBot
anchor at 0, snapshot buckets of the trainable agent), the ratings records get the compact bt summary, the console
shows `bt <rating> ± <ci>`."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP3_TTT_TINY


def test_a_run_publishes_live_ratings(tmp_path):
    run = run_train(SP3_TTT_TINY, tmp_path, name="live", tiny=False,
                    overrides={"ratings.refit_interval_sec": "1", "training.total_timesteps": "6000"})
    assert run.returncode == 0, run.stderr[-3000:]
    section = run.ratings()["bt"]["2p"]
    rows = section["player"]
    by_entity = {r["entity"]: r for r in rows}
    assert by_entity["random"]["anchor"] and by_entity["random"]["rating"] == 0.0
    assert section["@anchors"]["player/0"]["rule"] == "random_bot"
    buckets = sorted(r["version"] for r in rows if r["lineage"] == "main")
    assert buckets and buckets[0] == 0 and all(v % 20 == 0 for v in buckets)       # checkpoint.interval 20
    assert all(r["ci_low"] <= r["rating"] <= r["ci_high"] for r in rows)
    records = [r for r in run.records("ratings") if "bt" in r]
    assert records and records[-1]["bt"]["main"]["2p"]["latest"]["entity"] == f"main@v{buckets[-1]}"
    assert "| bt " in run.stderr and "Live ratings: Bradley-Terry refit every 1 s" in run.log("main")
```

In `tests/integration/test_sp2_metrics_outputs.py` (spec §8: "Обновляется тест SP2, закрепляющий ключи") replace `assert set(ratings) == {"env_steps", "layouts"}` by
```python
    assert set(ratings) == {"env_steps", "layouts", "bt"}               # SP4: the live Bradley-Terry section
```
and after the `last_ratings_record` key assertion add
```python
    # SP4 (spec block 7.3): the records carry a PFSP summary; the full table stays in ratings.json
    assert "pfsp" not in last_ratings_record["layouts"]["2p"] and "pfsp_summary" in last_ratings_record["layouts"]["2p"]
```

In `tests/integration/test_sp2_game_runs.py::test_self_play_reaches_the_budget_and_checkpoints_carry_roles` replace `assert set(ratings) == {"env_steps", "layouts"} and ratings["layouts"]` by `assert set(ratings) == {"env_steps", "layouts", "bt"} and ratings["layouts"]   # SP4: "bt"`. (No other test reads the `pfsp` key of a `ratings` record: `grep -rn pfsp tests/integration tests/learning tests/contract scripts`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_live_ratings.py tests/integration/test_sp4_live_ratings_run.py tests/integration/test_sp2_metrics_outputs.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.arena.live'`; the SP2 output test fails on the missing `bt` key.

- [ ] **Step 3: Write `src/colosseum/arena/live.py`**

```python
"""Live snapshot ratings of a training run (SP4 spec block 2, "Живой рейтинг").

``LiveRatings`` is a thread of the training main process. Every ``interval_sec`` (monotonic) it takes a copy of the
in-memory ``ComparisonTable`` that the journal writer feeds (``table.copy()``; nothing else is shared), fits every
layout of the run's sources (``RUN_SOURCES``: train and pulse; ``colosseum.arena.bt.fit`` warm-started from the
previous fit), computes Laplace uncertainty (``LAPLACE_SAMPLES`` draws) and the G2 test, summarizes one-team
layouts, and publishes an immutable ``RatingsSnapshot`` (``latest()``). A refit is skipped while the table has not
changed. It never touches the coordinator, the storage or the metrics hub; the main loop reads ``latest()`` at its
metrics tick. A failed refit is an ERROR (with traceback, once) and the thread goes on. Anchor notes (a configured
or default anchor that did not play in a cell) are logged once each (INFO).

``bt_section(snapshot)`` is the ``bt`` key of ``ratings.json``: ``{layout: {group: [entity rows]}}`` sorted by
rating; keys starting with ``@`` (impossible in role names) carry per-layout extras: ``@bias`` (team index ->
Elo), ``@anchors`` (``"<group>/<component>"`` -> anchor and rule), ``@transitivity`` and, for one-team layouts,
``@score`` (entities and compositions). ``bt_summary(snapshot, agents)`` is the compact form for the ``ratings``
record of ``metrics.jsonl``: per agent and layout its latest version bucket and its best entity.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from colosseum.arena.bt import (
    DEFAULT_OPTIONS,
    BTFit,
    BTOptions,
    ScoreSummary,
    TransitivityResult,
    Uncertainty,
    fit,
    laplace,
    score_summary,
    transitivity_test,
)
from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.entities import entity_version, lineage
from colosseum.arena.matchlog import RUN_SOURCES

logger = logging.getLogger(__name__)

LAPLACE_SAMPLES = 1000
# The final refit at the end of a run runs only when the last refit took at most this long (seconds).
FINAL_REFIT_MAX_SEC = 2.0


@dataclass(frozen=True)
class RatingsSnapshot:
    version: int                                   # increments per published refit
    env_steps: int                                 # the global env-step counter when the refit started
    fits: Mapping[str, BTFit]                      # per layout with team pairs
    uncertainty: Mapping[str, Uncertainty]
    scores: Mapping[str, ScoreSummary]             # per one-team layout
    transitivity: Mapping[str, TransitivityResult] = field(default_factory=dict)
    refit_sec: float = 0.0                         # how long the refit took (monotonic)


def random_bot_agent(config) -> str | None:
    """The config's first scripted agent whose class is ``colosseum.players.RandomBot`` (config order), or None."""
    from colosseum.core.registry import import_class
    from colosseum.players import RandomBot

    for agent_id in config.fixed_agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind != "scripted":
            continue
        try:
            if import_class(entry.class_path) is RandomBot:
                return agent_id
        except (ImportError, AttributeError, ValueError, TypeError):
            continue
    return None


def compute_snapshot(table: ComparisonTable, *, version: int, env_steps: int, options: BTOptions = DEFAULT_OPTIONS,
                     fixed_agents: Collection[str] = (), random_bot: str | None = None,
                     agent_order: Sequence[str] = (), top_k: int = 0, quantile: float = 0.1,
                     rng: np.random.Generator, previous: RatingsSnapshot | None = None,
                     samples: int = LAPLACE_SAMPLES, sources: Collection[str] = RUN_SOURCES) -> RatingsSnapshot:
    """One refit of every layout of ``table`` (module docstring)."""
    started = time.monotonic()
    fits: dict[str, BTFit] = {}
    uncertainty: dict[str, Uncertainty] = {}
    scores: dict[str, ScoreSummary] = {}
    transitivity: dict[str, TransitivityResult] = {}
    for layout in table.layouts():
        if table.pairs(layout, sources):
            warm = previous.fits.get(layout) if previous is not None else None
            fitted = fit(table, layout, sources=sources, options=options, fixed_agents=fixed_agents,
                         random_bot=random_bot, agent_order=agent_order, warm=warm)
            fits[layout] = fitted
            uncertainty[layout] = laplace(fitted, samples=samples, rng=rng, top_k=top_k, quantile=quantile)
            transitivity[layout] = transitivity_test(table, fitted, sources=sources)
        if table.scores(layout, sources):
            scores[layout] = score_summary(table, layout, sources=sources)
    return RatingsSnapshot(version=version, env_steps=int(env_steps), fits=fits, uncertainty=uncertainty,
                           scores=scores, transitivity=transitivity, refit_sec=time.monotonic() - started)


class LiveRatings:
    """The refit thread (module docstring)."""

    def __init__(self, table: ComparisonTable, *, interval_sec: float, options: BTOptions = DEFAULT_OPTIONS,
                 fixed_agents: Collection[str] = (), random_bot: str | None = None, agent_order: Sequence[str] = (),
                 top_k: int = 0, quantile: float = 0.1, env_steps_fn: Callable[[], int],
                 samples: int = LAPLACE_SAMPLES, seed: int = 0) -> None:
        self._table = table
        self._interval = float(interval_sec)
        self._kwargs = dict(options=options, fixed_agents=tuple(fixed_agents), random_bot=random_bot,
                            agent_order=tuple(agent_order), top_k=int(top_k), quantile=float(quantile),
                            samples=int(samples))
        self._env_steps_fn = env_steps_fn
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()          # guards _latest
        self._refit_lock = threading.Lock()    # one refit at a time (the thread, or refit_now from the main loop)
        self._latest: RatingsSnapshot | None = None
        self._table_version = -1
        self._notes: set[str] = set()
        self._failed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def latest(self) -> RatingsSnapshot | None:
        """The newest published snapshot (immutable), or None before the first refit."""
        with self._lock:
            return self._latest

    @property
    def last_refit_sec(self) -> float | None:
        snapshot = self.latest()
        return None if snapshot is None else snapshot.refit_sec

    def refit_now(self) -> RatingsSnapshot | None:
        """Refit now unless the table is unchanged since the last refit; returns the latest snapshot."""
        with self._refit_lock:
            previous = self.latest()
            version = self._table.version
            if previous is not None and version == self._table_version:
                return previous
            snapshot = compute_snapshot(self._table.copy(), version=1 if previous is None else previous.version + 1,
                                        env_steps=int(self._env_steps_fn()), rng=self._rng, previous=previous,
                                        **self._kwargs)
            for fitted in snapshot.fits.values():
                for group in fitted.groups:
                    key = f"{fitted.layout}/{group.group}/{group.component}: {group.note}"
                    if group.note and key not in self._notes:
                        self._notes.add(key)
                        logger.info(f"Ratings, layout {fitted.layout}, group {group.group}: {group.note}; the anchor "
                                    f"is {group.anchor} ({group.rule})")
            with self._lock:
                self._latest = snapshot
                self._table_version = version
            return snapshot

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-ratings", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.refit_now()
            except Exception as e:  # noqa: BLE001 - ratings must never stop training
                logger.error(f"Live ratings: a refit failed ({type(e).__name__}: {e}); training goes on",
                             exc_info=not self._failed)
                self._failed = True

    def stop(self, timeout: float) -> None:
        """Stop the thread (a refit in progress finishes first, up to ``timeout`` seconds)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)


# ---------------------------------------------------------------------------
# ratings.json and metrics.jsonl forms
# ---------------------------------------------------------------------------


def _entity_rows(fitted: BTFit, unc: Uncertainty) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for group in fitted.groups:
        for entity in group.entities:
            lo, hi = unc.ci95.get(entity, (fitted.rating[entity], fitted.rating[entity]))
            row = {"entity": entity, "lineage": lineage(entity), "version": entity_version(entity),
                   "rating": fitted.rating[entity], "se": fitted.se[entity], "ci_low": lo, "ci_high": hi,
                   "lcb": unc.lcb.get(entity), "games": int(fitted.games.get(entity, 0)),
                   "p_best": unc.p_best_group.get(entity), "p_best_lineage": unc.p_best_lineage.get(entity),
                   "component": group.component, "anchor": entity in group.anchor_members}
            if unc.p_top_k:
                row["p_top_k"] = unc.p_top_k.get(entity)
            rows.setdefault(group.group, []).append(row)
    for group_rows in rows.values():
        group_rows.sort(key=lambda r: (-r["rating"], r["entity"]))
    return rows


def bt_section(snapshot: RatingsSnapshot | None) -> dict:
    """The ``bt`` key of ``ratings.json`` (module docstring); ``{}`` without a snapshot."""
    if snapshot is None:
        return {}
    out: dict[str, dict] = {}
    for layout, fitted in snapshot.fits.items():
        section: dict = dict(_entity_rows(fitted, snapshot.uncertainty[layout]))
        section["@bias"] = {str(t): b for t, b in fitted.bias.items()}
        section["@anchors"] = {f"{g.group}/{g.component}": {"anchor": g.anchor, "members": g.anchor_members,
                                                            "rule": g.rule} for g in fitted.groups}
        test = snapshot.transitivity.get(layout)
        if test is not None:
            section["@transitivity"] = {"g2": test.g2, "dof": test.dof, "p_value": test.p_value,
                                        "flagged": test.flagged, "pairs_tested": test.pairs_tested,
                                        "worst_pairs": [{"a": a.encode(), "b": b.encode(), "residual": r}
                                                        for a, b, r in test.worst_pairs]}
        section["@fit"] = {"converged": fitted.converged, "iterations": fitted.iterations,
                           "approximate": fitted.approximate}
        out[layout] = section
    for layout, summary in snapshot.scores.items():
        out.setdefault(layout, {})["@score"] = {
            "entities": [{"entity": e, "n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high}
                         for e, c in summary.entities.items()],
            "compositions": [{"composition": k, "n": c.n, "mean": c.mean, "ci_low": c.ci_low, "ci_high": c.ci_high,
                              "homogeneous": summary.homogeneous[k]} for k, c in summary.compositions.items()],
        }
    return out


def _cell(entity: str, fitted: BTFit, unc: Uncertainty) -> dict:
    lo, hi = unc.ci95.get(entity, (fitted.rating[entity], fitted.rating[entity]))
    return {"entity": entity, "rating": fitted.rating[entity], "ci_low": lo, "ci_high": hi,
            "lcb": unc.lcb.get(entity), "games": int(fitted.games.get(entity, 0))}


def bt_summary(snapshot: RatingsSnapshot | None, agents: Sequence[str]) -> dict:
    """``{agent: {layout: {"latest": cell, "best": cell}}}`` for the ``ratings`` record: ``latest`` is the agent's
    newest version bucket, ``best`` its entity with the highest rating (Bradley-Terry layouts), or the same from the
    mean scores of one-team layouts (cells with ``mean``, ``ci_low``, ``ci_high``, ``n``)."""
    if snapshot is None:
        return {}
    out: dict[str, dict] = {}
    for agent in agents:
        per_layout: dict[str, dict] = {}
        for layout, fitted in snapshot.fits.items():
            mine = [e for e in fitted.rating if lineage(e) == agent and entity_version(e) is not None]
            if not mine:
                continue
            unc = snapshot.uncertainty[layout]
            newest = max(mine, key=lambda e: entity_version(e))
            best = max(mine, key=lambda e: (fitted.rating[e], entity_version(e)))
            per_layout[layout] = {"latest": _cell(newest, fitted, unc), "best": _cell(best, fitted, unc)}
        for layout, summary in snapshot.scores.items():
            mine = [e for e in summary.entities if lineage(e) == agent and entity_version(e) is not None]
            if not mine:
                continue
            newest = max(mine, key=lambda e: entity_version(e))
            best = max(mine, key=lambda e: (summary.entities[e].mean, entity_version(e)))
            per_layout[layout] = {
                key: {"entity": e, "mean": summary.entities[e].mean, "ci_low": summary.entities[e].ci_low,
                      "ci_high": summary.entities[e].ci_high, "n": summary.entities[e].n}
                for key, e in (("latest", newest), ("best", best))}
        if per_layout:
            out[agent] = per_layout
    return out


def console_bt(summary: Mapping[str, Mapping]) -> tuple[float, float] | None:
    """``(rating, half 95 % CI)`` of an agent's latest bucket for the console (``bt_summary`` of one agent): the
    Bradley-Terry layout where that bucket has the most games; None without one."""
    best = None
    for cells in summary.values():
        latest = cells.get("latest", {})
        if "rating" not in latest:
            continue
        if best is None or latest["games"] > best["games"]:
            best = latest
    if best is None:
        return None
    return best["rating"], (best["ci_high"] - best["ci_low"]) / 2.0
```

- [ ] **Step 4: Config, validation, PFSP summary, coordinator**

`src/colosseum/core/config.py`: before `class RunConfig` add
```python
class RatingsConfig(StrictModel):
    """Snapshot ratings (SP4 spec block 2): Bradley-Terry over the match log. Global only."""

    anchor: str | None = Field(
        default=None,
        description="Agent of the config whose rating is 0 in every role group and component it plays in (a "
                    "trainable agent: its first version bucket). null: the first RandomBot agent, else the mean of the "
                    "scripted / frozen agents, else v0 of the first trainable agent.",
    )
    refit_interval_sec: float = Field(default=60.0, gt=0, description="Seconds between live refits during training.")
    bootstrap_samples: int = Field(
        default=200, ge=20, description="Bootstrap replicates for reports, tournaments and --snapshots top:N.")
```
and in `ColosseumConfig`, after `match_log`: `ratings: RatingsConfig = Field(default_factory=RatingsConfig)`; the module docstring line becomes `SP4: the global ``match_log`` and ``ratings`` sections.`

`src/colosseum/core/validation.py`: before `validate_config` add
```python
def _check_ratings(config: ColosseumConfig, report: ValidationReport) -> None:
    """``ratings.anchor`` names an agent of the config (SP4 spec block 2); report the journal and rating
    settings."""
    from colosseum.arena.live import random_bot_agent

    anchor = config.ratings.anchor
    if anchor is not None and anchor not in config.agent_ids():
        raise ConfigError(f"ratings.anchor {anchor!r} is not an agent of the config (agents: {config.agent_ids()}); "
                          f"name one of them or remove ratings.anchor")
    log = config.match_log
    if not log.enabled:
        report.lines.append("match log: off (no snapshot ratings)")
        return
    report.lines.append(f"match log: on (detail window {log.detail_window_mb:g} MB, min free disk "
                        f"{log.min_free_disk_mb:g} MB, compaction every {log.compact_interval_sec:g} s)")
    if anchor is not None:
        rule = f"anchor {anchor!r}"
    elif (random_bot := random_bot_agent(config)) is not None:
        rule = f"anchor {random_bot!r} (the first RandomBot agent)"
    elif config.fixed_agent_ids():
        rule = f"anchor: the mean of the fixed agents that played ({', '.join(config.fixed_agent_ids())})"
    else:
        rule = "anchor: v0 of the first trainable agent"
    ratings = config.ratings
    report.lines.append(f"ratings: Bradley-Terry over train and pulse matches, refit every "
                        f"{ratings.refit_interval_sec:g} s, {rule}; bootstrap {ratings.bootstrap_samples}")
```
and in `validate_config`, in the `if train:` block, after `report.lines.extend(describe_mix(config, spec))` add `_check_ratings(config, report)` (the "play" scope does not check it: `record` / `bc` / `eval` write no ratings).

`src/colosseum/league/pfsp.py`: before `pfsp_weight` add
```python
def pfsp_summary(table: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, dict]:
    """The compact form of one layout's PFSP table (``snapshot()[layout]``) for the ``ratings`` records of
    metrics.jsonl (SP4 spec block 7.3): per owner the number of players and their smallest / largest score."""
    out = {}
    for owner, players in table.items():
        scores = [float(cell["score"]) for cell in players.values()]
        out[owner] = {"players": len(scores), "min_score": min(scores) if scores else None,
                      "max_score": max(scores) if scores else None}
    return out


def compact_ratings(layouts: Mapping[str, Mapping]) -> dict[str, dict]:
    """``Coordinator.ratings_snapshot()`` with each layout's ``pfsp`` table replaced by ``pfsp_summary`` (the full
    table stays in ratings.json)."""
    out = {}
    for layout, table in layouts.items():
        compact = {k: v for k, v in table.items() if k != "pfsp"}
        if "pfsp" in table:
            compact["pfsp_summary"] = pfsp_summary(table["pfsp"])
        out[layout] = compact
    return out
```

`src/colosseum/coordinator/coordinator.py`: import `compact_ratings` next to `PfspStats` and replace `ratings_snapshot`:
```python
    def ratings_snapshot(self, *, compact: bool = False) -> dict:
        """``RatingBook.snapshot()`` with each layout's PFSP table under ``"pfsp"`` (JSON-serializable); with
        ``compact`` the table is replaced by its ``pfsp_summary`` (the form of the metrics.jsonl records)."""
        snap = self._ratings.snapshot()
        pfsp = self._pfsp.snapshot()
        for layout, table in snap.items():
            table["pfsp"] = pfsp.get(layout, {})
        return compact_ratings(snap) if compact else snap
```

`scripts/bench_throughput.py::_make_config`, after `match_log=...`: `ratings={"anchor": None, "refit_interval_sec": 60.0, "bootstrap_samples": 200},`.

- [ ] **Step 5: Hub and console**

`src/colosseum/metrics/console.py` — `format_line` gets a keyword `bt: tuple[float, float] | None = None` (docstring: "``bt``: the agent's live Bradley-Terry rating of its latest version bucket and the half width of its 95 % interval (Elo units, SP4).") and, after the `wr_arena` part:
```python
        if bt is not None:
            parts.append(f"bt {bt[0]:.0f} ± {bt[1]:.0f}")
```

`src/colosseum/metrics/hub.py`:
1. Imports: `from colosseum.arena.live import RatingsSnapshot, bt_section, bt_summary, console_bt` and `from colosseum.league.pfsp import compact_ratings`.
2. `MetricsHub` docstring:
```
    Every ``console_interval_sec`` it also writes ``episodes``, ``system`` and ``ratings``
    records, rewrites ``ratings.json`` and prints one console line per agent (ratings aggregated over layouts).
    SP4 (spec block 7.3): the ``ratings`` record is ``{"env_steps", "layouts", "bt"?}`` with each layout's PFSP
    table replaced by its ``pfsp_summary`` and ``bt`` the compact live Bradley-Terry summary (``bt_summary``; absent
    before the first refit); ``ratings.json`` is ``{"env_steps", "layouts", "bt"}`` with the full PFSP tables and
    ``bt_section``. The ratings snapshot is built only at a tick (``ratings`` may be a callable).
    ``initial_env_steps`` / ``initial_train_steps`` are the counters a resumed run continues from (rate baselines,
    see ``SystemStats``). ``agent_ids`` are the trainable agents: one console line each, and the only opponents of
    the console's arena win rate (ruling P11).
```
3. `maybe_tick(self, *, env_steps: int, ratings: dict | Callable[[], dict], queue_depths: dict[str, int], force: bool = False, match_log: dict | None = None, bt: RatingsSnapshot | None = None) -> bool` (docstring: "``ratings`` is ``Coordinator.ratings_snapshot()`` (``{layout: {elo, win_rates, ..., pfsp}}``) or a callable that returns it, called only when the tick is due; ``match_log`` ...; ``bt`` the newest live ratings snapshot (None: none yet)."). Replace the two lines that write the `ratings` record and `ratings.json`, and the WandB ratings row, by
```python
        if callable(ratings):
            ratings = ratings()
        summary = bt_summary(bt, self._agent_ids)
        record = {"env_steps": int(env_steps), "layouts": compact_ratings(ratings)}
        if bt is not None:
            record["bt"] = summary
        self._writer.write("ratings", **record)
        write_json_atomic(self._ratings_path, {"env_steps": int(env_steps), "layouts": ratings,
                                               "bt": bt_section(bt)})
        if self._wandb is not None:
            row = flatten("system", {k: v for k, v in system.items() if k != "env_steps"})
            row.update(flatten("ratings", _ratings_row(ratings)))
            row.update(flatten("ratings/bt", summary))      # numbers only: entity names are dropped
```
   (the rest of the WandB block is unchanged); `self._report_console(int(env_steps), system, ratings, summary)`.
4. `_report_console(self, env_steps, system, ratings, bt: dict)` passes `bt=console_bt(bt.get(agent_id, {}))` to `format_line`.
5. `close(self, *, env_steps, ratings: dict | Callable[[], dict], queue_depths, match_log=None, bt: RatingsSnapshot | None = None)` passes `bt=bt` to its forced `maybe_tick`.

- [ ] **Step 6: Launcher and the BLAS limit**

`src/colosseum/launcher.py`:
1. `Launcher.__init__`, after `self._results_received = 0`:
```python
        # Set by launch() with the journal: the live Bradley-Terry refit thread (T2.2).
        self._live_ratings = None
```
2. `launch()`: after `self._open_match_log(coordinator)` add `self._open_live_ratings()`.
3. `_monitor_loop`: the tick becomes
```python
            self._hub.maybe_tick(env_steps=env_steps, ratings=coordinator.ratings_snapshot,
                                 queue_depths=queue_depths(self._trajectory_queues),
                                 match_log=self._match_log_stats(), bt=self._live_snapshot())
```
   (the method itself, not its result: the snapshot is built only when a tick is due — the SP2 open item "the main process builds a ratings snapshot on every monitor pass").
4. New methods before `_on_snapshot_evicted`:
```python
    def _open_live_ratings(self) -> None:
        """The live Bradley-Terry refit thread over the writer's in-memory table (spec block 2); none without the
        journal. ``top_k`` is 0 until snapshot retention by rating exists (T4.1)."""
        if self._comparison_table is None:
            return
        from colosseum.arena.bt import BTOptions
        from colosseum.arena.live import LiveRatings, random_bot_agent

        cfg = self._config
        self._live_ratings = LiveRatings(
            self._comparison_table, interval_sec=cfg.ratings.refit_interval_sec,
            options=BTOptions(anchor=cfg.ratings.anchor), fixed_agents=cfg.fixed_agent_ids(),
            random_bot=random_bot_agent(cfg), agent_order=cfg.agent_ids(), top_k=0, quantile=0.1,
            env_steps_fn=lambda: int(self._env_step_counter.value), seed=cfg.training.seed or 0)
        self._live_ratings.start()
        logger.info(f"Live ratings: Bradley-Terry refit every {cfg.ratings.refit_interval_sec:g} s")

    def _live_snapshot(self):
        return None if self._live_ratings is None else self._live_ratings.latest()

    def _finish_live_ratings(self):
        """Stop the refit thread; refit once more on the final data unless the last refit was slow
        (``FINAL_REFIT_MAX_SEC``). Returns the newest snapshot (None without live ratings)."""
        if self._live_ratings is None:
            return None
        from colosseum.arena.live import FINAL_REFIT_MAX_SEC

        self._live_ratings.stop(SHUTDOWN_GRACE_SEC)
        last = self._live_ratings.last_refit_sec
        if last is None or last <= FINAL_REFIT_MAX_SEC:
            try:
                return self._live_ratings.refit_now()
            except Exception:  # noqa: BLE001 - the final records are still written
                logger.exception("Live ratings: the final refit failed; ratings.json keeps the last snapshot")
        else:
            logger.info(f"Live ratings: the last refit took {last:.1f} s; no final refit (ratings.json keeps the "
                        f"snapshot of env step {self._live_ratings.latest().env_steps})")
        return self._live_ratings.latest()
```
5. `_finish_metrics`: after `self._stop_match_log()` add `final_bt = self._finish_live_ratings()`; the `close` call gets `ratings=self._coordinator.ratings_snapshot` and `bt=final_bt`; in the `finally` block's `try:` add, after `self._stop_match_log()`,
```python
                if self._live_ratings is not None:
                    self._live_ratings.stop(SHUTDOWN_GRACE_SEC)
```
   and the `logger.exception` message becomes `"Stopping the match log writer or the live ratings failed"`.

`src/colosseum/utils/process.py`, before `parent_alive`:
```python
# Thread-count variables of the BLAS libraries numpy may load (OpenBLAS, MKL, BLIS, Accelerate, OpenMP).
BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_main_blas_threads() -> bool:
    """Load numpy's BLAS with one thread in this process only (SP4 spec block 2: the live-rating refit of the
    training main process must not compete with the workers for cores).

    BLAS libraries read their thread count when they are loaded, i.e. on ``import numpy``: the variables are set,
    numpy is imported, and the environment is restored at once, so spawned children (workers, learners) start with
    the original environment. Returns False (and changes nothing) when numpy is already imported. OpenBLAS (the
    numpy wheels) and OpenMP read the count at load time; a BLAS that reads it later is not limited."""
    if "numpy" in sys.modules:
        return False
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARS}
    os.environ.update({name: "1" for name in BLAS_THREAD_VARS})
    try:
        import numpy  # noqa: F401 - loads the BLAS with the variables above
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return True
```
(`colosseum.utils.process` must stay free of numpy imports: the CLI imports it before any command runs.)

`src/colosseum/cli.py`, `train`: the body becomes
```python
    from colosseum.utils.process import limit_main_blas_threads

    limit_main_blas_threads()   # first: numpy is imported below; this process's BLAS gets one thread (SP4)
    with _config_errors():
        from colosseum.launcher import run_training  # imports torch: about a second

        code = run_training(config, overrides=_parse_overrides(overrides) or None)
    sys.exit(code)
```
Before the call nothing has imported numpy (checked: `import colosseum.cli` and `colosseum.utils.process` leave `numpy` out of `sys.modules`); spawned children inherit the restored environment (the probe in the test checks the child's OpenBLAS thread count).

- [ ] **Step 7: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_live_ratings.py tests/integration/test_sp4_live_ratings_run.py tests/integration/test_sp2_metrics_outputs.py tests/integration/test_sp2_game_runs.py tests/unit/test_sp2_metrics.py tests/unit/test_sp3_pfsp_stats.py tests/unit/test_sp3_share_metrics.py tests/unit/test_sp2_launcher_lifecycle.py tests/unit/test_wandb_logger.py tests/unit/test_bench_throughput.py -q -rw`
Expected: all pass.

- [ ] **Step 8: Measure the refit cost (for T8.2)**

Run (prints one line per size; record the numbers in the task report):
```bash
env -u PYTHONPATH .venv/bin/python - <<'EOF'
import sys, time
sys.path.insert(0, "tests")
import numpy as np
from colosseum.arena.live import compute_snapshot
from game_helpers import bt_table, synthetic_bt_matches
for E in (100, 500, 1000, 2000):
    strengths = {f"main@v{10 * k}": 3.0 * k / E for k in range(E)} | {"random": -1.0}
    def choose(rng, E=E):
        i, j = rng.integers(E), rng.integers(E + 1)
        a, b = f"main@v{10 * i}", ("random" if j == E else f"main@v{10 * j}")
        return [[a], [b if b != a else "random"]]
    table = bt_table(synthetic_bt_matches(strengths, n_matches=20 * E, rng=np.random.default_rng(1), choose=choose))
    started = time.monotonic()
    compute_snapshot(table, version=1, env_steps=0, fixed_agents=("random",), random_bot="random",
                     agent_order=("main",), top_k=5, rng=np.random.default_rng(0))
    print(f"{E} entities, {len(table.pairs('2p', ['train']))} pair rows: refit {time.monotonic() - started:.2f} s")
EOF
```
Expected on the owner's machine (measured while writing the plan): about 0.2 / 0.7 / 1.1 / 1.6 s (2000 entities take the iterative path). The live refit runs in its own thread every 60 s; BLAS is single-threaded in the main process.

- [ ] **Step 9: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 10: Commit and push**

```bash
git add src/colosseum/arena/live.py src/colosseum/core/config.py src/colosseum/core/validation.py \
        src/colosseum/league/pfsp.py src/colosseum/coordinator/coordinator.py src/colosseum/metrics \
        src/colosseum/launcher.py src/colosseum/utils/process.py src/colosseum/cli.py scripts/bench_throughput.py \
        tests/unit/test_sp4_live_ratings.py tests/integration/test_sp4_live_ratings_run.py \
        tests/integration/test_sp2_metrics_outputs.py tests/integration/test_sp2_game_runs.py
git commit -m "feat: live snapshot ratings (refit thread, bt in ratings.json, compact ratings records, console, WandB)"
git push
```

---

### Task T3.1: `rho_trunc_mass` / `c_trunc_mass`, monotonic loss counters, `bc` device log

Spec block 7.1, 7.2, 7.5 (owner's answer to Q2 of the GPU checks). The APPO `train` metrics `rho_clip_frac` / `c_clip_frac` (the share of ρ above a bar, which counted float noise above ρ̄ = 1: 0.41–0.52 in every real run) are replaced by the truncated importance-weight mass over ACT slots, `rho_trunc_mass = Σ max(ρ − ρ̄, 0) / Σ ρ` and `c_trunc_mass = Σ max(ρ − c̄, 0) / Σ ρ`, with the same ρ (`unit_trace`) and the same bars as V-trace (with `unit_trace: none` V-trace runs with ρ ≡ 1 and bars 1, so the mass is exactly 0). The old names are removed, not aliased (spec ruling 9); every reference is updated: README's metric table, the APPO tests, `docs/GPU_CHECKS.md`, `docs/benchmarks.md` (its recorded tables keep their historical `c_clip_frac` column, explained in the text), `scripts/units_experiment.py`, `scripts/measure_amp_logratio.py`. `dropped_reward_episodes` becomes a run-level counter like `dropped_match_results` (T1.3: per worker the largest reported value, summed over every worker ever seen). `colosseum bc` logs its device. This touches the training math, so the task runs the slow suite once.

**Files:**
- Modify: `src/colosseum/algorithms/appo.py` (`_diagnostics`: the bars as arguments, the two masses)
- Modify: `src/colosseum/metrics/aggregator.py` (`CUMULATIVE_WORKER_KEYS` gains `dropped_reward_episodes`)
- Modify: `src/colosseum/bc/offline_bc.py` (`OfflineBCTrainer.__init__` logs the device)
- Modify: `scripts/units_experiment.py`, `scripts/measure_amp_logratio.py`
- Modify (docs): `README.md` (`train` and `system` rows of the metrics table), `docs/ENV_GUIDE.md` (`dropped_reward_episodes`), `docs/GPU_CHECKS.md`, `docs/benchmarks.md`
- Modify (tests pinned to the old names): `tests/unit/test_appo_v2.py`
- Test: `tests/unit/test_sp4_metric_changes.py`

**Interfaces:**
- Consumes: `APPO._loss_from_batch` (`algorithms/appo.py:354`; the effective `rho_bar` / `c_bar` of the V-trace call, `appo.py:387-393`), `SystemStats.cumulative` and `CUMULATIVE_WORKER_KEYS` (T1.3).
- Produces: `train` metrics `rho_trunc_mass`, `c_trunc_mass` (and no `rho_clip_frac`, `c_clip_frac`); *`APPO._diagnostics(..., rho_bar, c_bar)`; `system.dropped_reward_episodes` monotonic per run; the INFO line `BC trainer: device <device>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_metric_changes.py`:
```python
"""Metric changes of SP4 block 7 (T3.1): the truncated importance-weight mass replaces the clip fractions (exactly
0 with unit_trace none, 0 up to float rounding on-policy, the formula off-policy), the run-level loss counters never
decrease, and `colosseum bc` logs its device."""
from __future__ import annotations

import copy
import logging

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.core.config import AlgorithmConfig
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import SLOT_ACT
from colosseum.envs.game import RoleSpec
from colosseum.metrics.aggregator import SystemStats
from game_helpers import make_test_model, synthetic_chunk

OBS = gymnasium.spaces.Box(-1.0, 1.0, (5,), np.float32)
DISCRETE = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
PATTERNS = ["AAATAB", "ATAATP", "AARAAB", "AAAAAB"]


def _chunks(model, noise, seed=0):
    return [synthetic_chunk(model, DISCRETE, p, seed=seed + i, logp_noise=noise) for i, p in enumerate(PATTERNS)]


def _losses(model, chunks, **cfg) -> dict[str, float]:
    algo = APPO(model, AlgorithmConfig(**cfg), ActionSpec.from_space(DISCRETE.action_space), device="cpu")
    with torch.no_grad():
        return {k: float(v) for k, v in algo.compute_loss(chunks).items()}


def test_the_clip_fractions_are_gone():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.3))
    assert {"rho_trunc_mass", "c_trunc_mass"} <= set(losses)
    assert not {"rho_clip_frac", "c_clip_frac"} & set(losses)


def test_on_policy_the_mass_is_zero_up_to_float_rounding_even_at_bars_of_one():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.0), vtrace_rho_bar=1.0, vtrace_c_bar=1.0)
    assert 0.0 <= losses["rho_trunc_mass"] < 1e-6 and 0.0 <= losses["c_trunc_mass"] < 1e-6


def test_off_policy_the_mass_is_the_truncated_share_of_the_weights():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    chunks = _chunks(model, 0.5)
    algo = APPO(copy.deepcopy(model), AlgorithmConfig(vtrace_rho_bar=1.0, vtrace_c_bar=0.8),
                ActionSpec.from_space(DISCRETE.action_space), device="cpu")
    lp, _, _ = algo.evaluate_chunks(chunks)
    act = torch.stack([c.kind for c in chunks], dim=1).reshape(-1) == SLOT_ACT
    behavior = torch.stack([c.behavior_logp for c in chunks], dim=1).reshape(-1)
    rho = torch.exp(lp - behavior)[act]
    with torch.no_grad():
        losses = {k: float(v) for k, v in algo.compute_loss(chunks).items()}
    assert losses["rho_trunc_mass"] == pytest.approx(float((rho - 1.0).clamp(min=0).sum() / rho.sum()), rel=1e-4)
    assert losses["c_trunc_mass"] == pytest.approx(float((rho - 0.8).clamp(min=0).sum() / rho.sum()), rel=1e-4)
    assert 0.0 < losses["rho_trunc_mass"] < losses["c_trunc_mass"] < 1.0


def test_with_unit_trace_none_the_mass_is_exactly_zero():
    torch.manual_seed(0)
    model = make_test_model(DISCRETE)
    losses = _losses(model, _chunks(model, 0.5), unit_trace="none", vtrace_rho_bar=0.5, vtrace_c_bar=0.5)
    assert losses["rho_trunc_mass"] == losses["c_trunc_mass"] == 0.0     # rho = c = 1 and V-trace uses bars of 1


def test_run_level_loss_counters_never_decrease():
    clock = [0.0]
    stats = SystemStats(clock=lambda: clock[0])
    stats.on_worker_stats({"worker_id": 0, "dropped_reward_episodes": 3, "dropped_results": 1})
    stats.on_worker_stats({"worker_id": 1, "dropped_reward_episodes": 4})
    clock[0] = 1.0
    first = stats.snapshot(10, {})
    assert (first["dropped_reward_episodes"], first["dropped_match_results"]) == (7, 1)
    clock[0] = 100.0                                    # every worker fell silent (stalled, or the run is ending)
    later = stats.snapshot(20, {})
    assert later["workers_reporting"] == 0 and later["dropped_reward_episodes"] == 7
    stats.on_worker_stats({"worker_id": 2, "dropped_reward_episodes": 1})   # a new worker adds to the total
    assert stats.snapshot(30, {})["dropped_reward_episodes"] == 8


def test_bc_logs_its_device(caplog):
    role = RoleSpec(OBS, gymnasium.spaces.Discrete(4))
    with caplog.at_level(logging.INFO, logger="colosseum.bc.offline_bc"):
        OfflineBCTrainer(make_test_model(role), ActionSpec.from_space(role.action_space), ObsSpec.from_space(OBS),
                         device="cpu")
    assert any(r.getMessage() == "BC trainer: device cpu" for r in caplog.records)
```

In `tests/unit/test_appo_v2.py`:
1. `DIAGNOSTICS`: replace `"rho_clip_frac", "c_clip_frac",` by `"rho_trunc_mass", "c_trunc_mass",`.
2. `test_zero_lag_diagnostics_are_exact`: `assert losses["rho_clip_frac"] == losses["c_clip_frac"] == 0.0` becomes `assert losses["rho_trunc_mass"] == losses["c_trunc_mass"] == 0.0` (bars 1.5: no ρ reaches them); in its docstring "a rho of 1 + 1e-7 is clipped by a bar of exactly 1" becomes "a rho of 1 + 1e-7 is clipped by a bar of exactly 1 (the truncated mass at bars of 1 is tested in test_sp4_metric_changes.py)".
3. `test_log_rho_quantiles_and_ess_are_taken_over_act_slots`: `assert losses["rho_clip_frac"] == pytest.approx(float((rho > 1.0).float().mean()))` becomes `assert losses["rho_trunc_mass"] == pytest.approx(float((rho - 1.0).clamp(min=0).sum() / rho.sum()), rel=1e-4)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_metric_changes.py tests/unit/test_appo_v2.py -q`
Expected: `KeyError: 'rho_trunc_mass'`, the counter test (`dropped_reward_episodes` reads 0 once the workers are silent) and the BC device test fail.

- [ ] **Step 3: APPO (`src/colosseum/algorithms/appo.py`)**

1. In `_loss_from_batch`, the `_diagnostics(...)` call passes the bars V-trace used:
```python
            **self._diagnostics(batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                                trace_log_rho, vt.vs, rho_bar, c_bar),
```
   (`rho_bar`, `c_bar` are set in the `with torch.no_grad():` block above: the config's bars, or 1 and 1 for `unit_trace: none`.)
2. `_diagnostics` gets the two parameters and a docstring:
```python
    def _diagnostics(self, batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                     trace_log_rho, vs, rho_bar: float, c_bar: float) -> dict[str, Tensor]:
        """Training diagnostics over ACT slots. ``rho_bar`` / ``c_bar`` are the bars V-trace applied (1 and 1 with
        ``unit_trace: none``, where every rho is 1): ``rho_trunc_mass = sum(max(rho - rho_bar, 0)) / sum(rho)`` and
        ``c_trunc_mass`` (with ``c_bar``) are the share of the importance-weight mass V-trace cut off, 0 on-policy up
        to float rounding (SP4 spec block 7.1; they replace the fractions ``rho_clip_frac`` / ``c_clip_frac``,
        which counted float noise above a bar of 1)."""
```
3. After `sum_rho2 = (rho * rho).sum()` add
```python
        zeros_rho = torch.zeros_like(rho)
        mass = sum_rho.clamp(min=1e-12)
```
   and replace the two `*_clip_frac` entries of the returned dict by
```python
            "rho_trunc_mass": torch.where(is_act, (rho - rho_bar).clamp(min=0.0), zeros_rho).sum() / mass,
            "c_trunc_mass": torch.where(is_act, (rho - c_bar).clamp(min=0.0), zeros_rho).sum() / mass,
```
   `rho` there is `exp(clamp(trace_log_rho, -20, 20))` on ACT slots and 0 elsewhere — the ρ V-trace clamps (`algorithms/vtrace.py:57-60`).

- [ ] **Step 4: Counters, BC, scripts**

`src/colosseum/metrics/aggregator.py`: `CUMULATIVE_WORKER_KEYS = ("dropped_results", "dropped_reward_episodes")`, and in `SystemStats.snapshot` the `dropped_reward_episodes` entry becomes `"dropped_reward_episodes": self.cumulative("dropped_reward_episodes"),` (it summed only the workers that reported in the last 6 s: it could decrease and read 0 at shutdown, SP2 ruling #41 / N-1). `test_sp2_metrics.py::test_system_stats_rates_and_resume_baselines` stays green (7 = 3 + 4).

`src/colosseum/bc/offline_bc.py`, `OfflineBCTrainer.__init__`, right after `self._device = torch.device(device)`:
```python
        logger.info("BC trainer: device %s", self._device)
```
(`colosseum bc` configures INFO logging to stderr, so the line is printed; `OfflineBCTrainer` used as a library logs it too.)

`scripts/units_experiment.py`: in `DIAG_KEYS` `"c_clip_frac"` becomes `"c_trunc_mass"`; in the module docstring the list ends "``log_rho_abs_p95``, ``log_rho_joint_abs_mean``, ``c_trunc_mass``, ``deciders_valid_mean`` (``c_clip_frac`` before SP4: the tables of ``docs/benchmarks.md`` recorded then keep that column)."

`scripts/measure_amp_logratio.py`: `DIAGNOSTICS = ("log_rho_abs_p95", "clip_fraction", "rho_trunc_mass")`; in the docstring "``clip_fraction``, ``rho_clip_frac``." becomes "``clip_fraction``, ``rho_trunc_mass`` (``rho_clip_frac`` before SP4)."

- [ ] **Step 5: Docs**

`README.md`, metrics table:
- `train` row: `` `rho_mean`, `rho_clip_frac`, `c_clip_frac`, `ess`, `` becomes `` `rho_mean`, `rho_trunc_mass` / `c_trunc_mass` (the share of the importance-weight mass V-trace cuts off at ρ̄ / c̄: `Σ max(ρ − ρ̄, 0) / Σ ρ` over ACT slots, 0 on-policy), `ess`, ``;
- `system` row: the `dropped_reward_episodes` explanation becomes "(seat-episodes whose rewards were dropped because the seat never acted; a run-level counter: the largest count of every worker that ever reported, summed, so it never decreases; a worker logs a WARNING on its first drop, see ENV_GUIDE), `dropped_match_results` (match results lost on the way to the journal: by workers on a full results queue and by the journal writer; never decreases), `match_log` (the journal writer's counters and the journal sizes, or null without a journal), `workers_reporting`".

`docs/ENV_GUIDE.md` (§ on rewards of waiting seats), replace the sentences from "Поле `dropped_reward_episodes`" to "надёжный сигнал — WARNING в логе воркера." by
```
  потери только считаются. Поле `dropped_reward_episodes` записи `system` в `metrics.jsonl` и в WandB
  (`system/dropped_reward_episodes`) — счётчик на весь запуск (эпизоды-места с потерянной ненулевой
  наградой): наибольшее значение каждого воркера, когда-либо приславшего статистику, в сумме по
  воркерам. Он не уменьшается, в том числе когда воркер подвисает, и в последней записи равен итогу
  запуска (SP4).
```

`docs/GPU_CHECKS.md` (the paragraph "Практически: ..."): the last sentence becomes "Доля `rho_clip_frac` (до SP4) на GPU при нулевом лаге читалась 0.13–0.54 и в fp32 тоже: она считала любой ρ > ρ̄ = 1, то есть и численный шум. С SP4 вместо неё — срезанная масса `rho_trunc_mass` / `c_trunc_mass` (`Σ max(ρ − ρ̄, 0) / Σ ρ` по ACT-слотам): при нулевом лаге она равна 0 с точностью до погрешности float (~1e-7) и растёт с тем, сколько V-trace реально срезает."

`docs/benchmarks.md` ("Как читать таблицу" of the units experiment): the first bullet becomes "- `rho_clip_frac` / `c_clip_frac` (метрики до SP4; колонка таблицы записана тогда) считали долю ρ > 1.0. Сюда попадали и отношения, отличающиеся от 1 на ~1e-7 из-за погрешности float (≈ 5 % даже при нулевом лаге). Поэтому `c_clip_frac` ≈ 0.4–0.5 у `joint`/`geo_mean` — не доля «настоящих» обрезок; у `none` ρ = c = 1, и колонка равна 0. С SP4 их заменили `rho_trunc_mass` / `c_trunc_mass` — срезанная масса весов `Σ max(ρ − ρ̄, 0) / Σ ρ` (0 при нулевом лаге); `scripts/units_experiment.py` пишет `c_trunc_mass`." The tables and `docs/benchmarks/*.json` stay as recorded (measurements are history).

Check that no live reference is left:
```bash
grep -rn "rho_clip_frac\|c_clip_frac" --include=*.py src tests; grep -n "rho_clip_frac\|c_clip_frac" README.md docs/ENV_GUIDE.md docs/LEAGUE_GUIDE.md
```
Expected: only `src/colosseum/algorithms/appo.py` (the `_diagnostics` docstring naming the replaced metrics) and `tests/unit/test_sp4_metric_changes.py` (which asserts they are gone). The SP3 fixture's `metrics.jsonl` keeps the old names (it is SP3 output; T7.1's report must read both). `CLAUDE.md` is updated by T8.4; `docs/benchmarks.md`, `docs/GPU_CHECKS.md` and the two scripts' docstrings name the old metrics as history; specs, plans and reports are frozen.

- [ ] **Step 6: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_metric_changes.py tests/unit/test_appo_v2.py tests/unit/test_vtrace_slots.py tests/contract/test_learner_reproduces_worker.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_bc_trainer.py tests/unit/test_units_experiment_rule.py -q -rw`
Expected: all pass.

- [ ] **Step 7: Full fast suite + ruff; GPU tests; slow suite once**

Run the full fast suite and ruff (green, zero warnings), then `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -q -W error` (AMP diagnostics go through `_diagnostics`; CUDA is present on this machine) and the slow suite once: `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v` (about 7 min here; the training math changed only in diagnostics, so every SP2/SP3 threshold must hold unchanged). Record the slow-suite durations in the task report.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/algorithms/appo.py src/colosseum/metrics/aggregator.py src/colosseum/bc/offline_bc.py \
        scripts/units_experiment.py scripts/measure_amp_logratio.py README.md docs/ENV_GUIDE.md docs/GPU_CHECKS.md \
        docs/benchmarks.md tests/unit/test_appo_v2.py tests/unit/test_sp4_metric_changes.py
git commit -m "feat: truncated importance-weight mass replaces the clip fractions; run-level loss counters; bc device log"
git push
```

---

### Task T3.2: `league_state.json` (save, load on a run-dir resume)

Spec block 7.4 and criterion 3.2 ("resume из папки запуска продолжает журнал, рейтинги и статистику PFSP"). `RatingBook` (every table of every layout: ELO, win-rate sums, the `wr_vs_past` window, score trackers, cross-play, role win rates) and `PfspStats` (the EMA table and the forgotten snapshots) get `state_dict` / `load_state_dict`; entries of agents that are not in the new config are dropped with one INFO line, new agents start empty, a layout the game no longer has is skipped. The coordinator writes `runs/<name>/league_state.json` atomically (`write_json_atomic`) at every metrics tick and at the end of the run; a run-dir resume loads the resumed run's file before the snapshot pool is imported (so evictions during the import forget their PFSP entries, SP3 ruling #50); an SP3 run dir has none (INFO: empty statistics). A resume from a checkpoint dir or a `.pt` keeps SP3 behaviour (empty). A custom matchmaker's own state (`on_result`) is not saved (documented by T8.4).

**Files:**
- Modify: `src/colosseum/coordinator/ratings.py` (`_LayoutRatings.state_dict` / `load_state_dict`, `RatingBook.state_dict` / `load_state_dict`, `logger`)
- Modify: `src/colosseum/league/pfsp.py` (`PfspStats.state_dict` / `load_state_dict`, `logger`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`LEAGUE_STATE_FILE`, `LEAGUE_STATE_VERSION`, `save_league_state`, `load_league_state`)
- Modify: `src/colosseum/core/run_dir.py` (`league_state_path`)
- Modify: `src/colosseum/launcher.py` (`_load_league_state`, `_save_league_state`, wiring)
- Modify (extend the T0.1 baseline): `tests/integration/test_sp4_sp3_resume.py`
- Test: `tests/unit/test_sp4_league_state.py`, `tests/integration/test_sp4_league_state_resume.py`

**Interfaces:**
- Consumes: `RatingBook`, `PfspStats` internals (`coordinator/ratings.py`, `league/pfsp.py:50-111`), `write_json_atomic` (`metrics/jsonl.py:79`), `classify_resume_source` / `RESUME_RUN_DIR`, the hub tick's return value (`MetricsHub.maybe_tick -> bool`).
- Produces (contract T3.2, plus additions marked *):
  - `RatingBook.state_dict() -> dict`, `RatingBook.load_state_dict(state, *, agents) -> list[str]` (*returns the dropped agents);
  - `PfspStats.state_dict() -> dict` (`prior`, `table`, `forgotten`), `PfspStats.load_state_dict(state, *, agents) -> list[str]`;
  - `LEAGUE_STATE_FILE = "league_state.json"` (in `colosseum.coordinator.coordinator`), *`LEAGUE_STATE_VERSION = 1`;
  - `Coordinator.save_league_state(path, *, env_steps)` (*keyword `env_steps`), `Coordinator.load_league_state(path) -> bool`;
  - `RunDir.league_state_path`;
  - *`Launcher._load_league_state(coordinator)`, *`Launcher._save_league_state()`.
  - `league_state.json` = `{"version": 1, "env_steps", "ratings": {"layouts": {...}}, "pfsp": {"prior", "table", "forgotten"}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp4_league_state.py`:
```python
"""League state (SP4 T3.2, spec block 7.4): RatingBook and PfspStats save and restore every table (forgotten
snapshots included), entries of agents that left the config are dropped with INFO, the coordinator writes and reads
league_state.json, and an unreadable file is a ConfigError."""
from __future__ import annotations

import json
import logging

import gymnasium
import numpy as np
import pytest

from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE
from colosseum.coordinator.ratings import RatingBook
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.league.pfsp import PfspStats
from game_helpers import make_coordinator, make_match_result, make_test_config, scripted_agent

SPEC = GameSpec(
    roles={"player": RoleSpec(gymnasium.spaces.Box(0, 1, (2,), np.float32), gymnasium.spaces.Discrete(3))},
    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
             "coop": (SeatSpec("player", 0), SeatSpec("player", 0))},
)


def _results():
    rng = np.random.default_rng(0)
    out = []
    for k in range(300):
        other = ("b", "latest") if k % 3 else ("a", "ckpt_v10")
        if k % 5 == 0:
            other = ("rnd", "fixed")
        ranks = [1.0, 2.0] if rng.random() < 0.6 else ([2.0, 1.0] if rng.random() < 0.8 else [1.0, 1.0])
        out.append(make_match_result([(0, "a", "latest"), (1, *other)], ranks, uid=f"m{k}"))
    for k in range(20):
        out.append(make_match_result([(0, "a", "latest"), (0, "b", "latest")], scores=[float(k % 4)], layout="coop",
                                     uid=f"c{k}"))
    return out


def test_the_rating_book_round_trips():
    book = RatingBook(SPEC, ["a", "b", "rnd"], past_window=50)
    for result in _results():
        book.update(result)
    state = json.loads(json.dumps(book.state_dict()))       # through JSON, as league_state.json
    again = RatingBook(SPEC, ["a", "b", "rnd"], past_window=50)
    assert again.load_state_dict(state, agents=["a", "b", "rnd"]) == []
    assert again.snapshot() == book.snapshot()
    for result in _results()[:60]:                          # the tables go on from the restored state
        book.update(result)
        again.update(result)
    assert again.snapshot() == book.snapshot()


def test_agents_that_left_the_config_are_dropped_with_info(caplog):
    book = RatingBook(SPEC, ["a", "b", "rnd"])
    for result in _results():
        book.update(result)
    fresh = RatingBook(SPEC, ["a", "newcomer"])
    with caplog.at_level(logging.INFO, logger="colosseum.coordinator.ratings"):
        assert fresh.load_state_dict(book.state_dict(), agents=["a", "newcomer"]) == ["b", "rnd"]
    snap = fresh.snapshot()["2p"]
    assert set(snap["games"]["a"]) == {"newcomer"} and snap["games"]["a"]["newcomer"] == 0
    assert snap["elo"]["newcomer"] == 1200.0 and snap["wr_vs_past"]["a"] is not None
    assert fresh.snapshot()["coop"]["cross_play"] == {}      # "a+b" had a dropped member
    assert any("dropped: ['b', 'rnd']" in r.getMessage() for r in caplog.records)
    other_game = {"layouts": {"9p": book.state_dict()["layouts"]["2p"]}}
    with caplog.at_level(logging.INFO, logger="colosseum.coordinator.ratings"):
        fresh.load_state_dict(other_game, agents=["a"])
    assert any("layouts this game does not have" in r.getMessage() for r in caplog.records)


def test_pfsp_statistics_round_trip_with_the_forgotten_snapshots(caplog):
    stats = PfspStats({"a": 20.0, "b": 20.0})
    for result in _results():
        stats.update(result)
    stats.forget(("a", "ckpt_v10"))
    state = json.loads(json.dumps(stats.state_dict()))
    again = PfspStats({"a": 20.0, "b": 20.0})
    assert again.load_state_dict(state, agents=["a", "b", "rnd"]) == []
    assert again.snapshot() == stats.snapshot()
    late = make_match_result([(0, "a", "latest"), (1, "a", "ckpt_v10")], uid="late")
    again.update(late)                                      # a forgotten snapshot never comes back (SP3 #50)
    assert again.games("2p", "a", ("a", "ckpt_v10")) == 0.0
    only_a = PfspStats({"a": 20.0})
    with caplog.at_level(logging.INFO, logger="colosseum.league.pfsp"):
        assert only_a.load_state_dict(state, agents=["a", "b"]) == ["b", "rnd"]
    assert set(only_a.snapshot()["2p"]) == {"a"} and "rnd@fixed" not in only_a.snapshot()["2p"]["a"]


def test_a_snapshot_forgotten_before_the_load_stays_forgotten():
    stats = PfspStats({"a": 20.0})
    for result in _results():
        stats.update(result)
    state = stats.state_dict()
    again = PfspStats({"a": 20.0})
    again.forget(("a", "ckpt_v10"))                         # evicted by the import of the pool, before the load
    again.load_state_dict(state, agents=["a", "b", "rnd"])
    assert "a@ckpt_v10" not in again.snapshot()["2p"]["a"]


def test_the_coordinator_saves_and_loads_league_state(tmp_path):
    config = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    coordinator = make_coordinator(config, tmp_path / "ckpt")
    for k in range(40):
        coordinator.report_match_result(make_match_result([(0, "agent_0", "latest"), (1, "rnd", "fixed")],
                                                          [1.0, 2.0] if k % 3 else [2.0, 1.0], uid=f"m{k}"))
    path = tmp_path / LEAGUE_STATE_FILE
    coordinator.save_league_state(path, env_steps=1234)
    saved = json.loads(path.read_text())
    assert (saved["version"], saved["env_steps"]) == (1, 1234) and set(saved) == {"version", "env_steps", "ratings",
                                                                                   "pfsp"}
    fresh = make_coordinator(config, tmp_path / "ckpt2")
    assert fresh.load_league_state(tmp_path / "missing.json") is False
    assert fresh.load_league_state(path) is True
    assert fresh.ratings_snapshot() == coordinator.ratings_snapshot()
    path.write_text("{")
    with pytest.raises(ConfigError, match="league state"):
        fresh.load_league_state(path)
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ConfigError, match="version 99"):
        fresh.load_league_state(path)
```

Create `tests/integration/test_sp4_league_state_resume.py`:
```python
"""League state across a run-dir resume (SP4 T3.2, spec block 7.4 and criterion 3.2): league_state.json is written
by a run and restored by a run-dir resume (the ratings continue), while a resume from a checkpoint dir starts with
empty statistics as in SP3."""
from __future__ import annotations

import json

from cli_runner import newest_checkpoint, run_train
from game_helpers import SP3_TTT_TINY


def _games(ratings: dict) -> int:
    return ratings["layouts"]["2p"]["games"]["main"]["random"]


def test_a_run_dir_resume_continues_the_league_state(tmp_path):
    first = run_train(SP3_TTT_TINY, tmp_path, name="first", tiny=False)
    assert first.returncode == 0, first.stderr[-3000:]
    state = json.loads((first.root / "league_state.json").read_text())
    assert state["version"] == 1 and "main" in state["pfsp"]["table"]["2p"]
    before = _games(first.ratings())
    assert before > 0
    budget = state["env_steps"] + 3000
    second = run_train(SP3_TTT_TINY, tmp_path, name="second", tiny=False,
                       overrides={"training.resume_from": str(first.root), "training.total_timesteps": str(budget)})
    assert second.returncode == 0, second.stderr[-3000:]
    assert "league state (ratings, PFSP statistics) restored from" in second.log("main")
    assert _games(second.ratings()) > before                # counted on from the restored tables
    third = run_train(SP3_TTT_TINY, tmp_path, name="third", tiny=False,
                      overrides={"training.resume_from": str(newest_checkpoint(first.root, "main")),
                                 "training.total_timesteps": str(budget)})
    assert third.returncode == 0, third.stderr[-3000:]
    assert "league state" not in third.log("main")          # an explicit checkpoint resume: SP3 behaviour
    assert _games(third.ratings()) < before                 # its own 3000 steps only (the first run had 6000)
```

Extend `tests/integration/test_sp4_sp3_resume.py`: append to the test
```python
    assert "has no league_state.json (an SP3 run dir); ratings and PFSP statistics start empty" in log   # T3.2
    assert (run.root / "league_state.json").is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_league_state.py tests/integration/test_sp4_league_state_resume.py tests/integration/test_sp4_sp3_resume.py -q`
Expected: `ImportError: cannot import name 'LEAGUE_STATE_FILE'` (collection); the integration tests find no `league_state.json`.

- [ ] **Step 3: `RatingBook` (`src/colosseum/coordinator/ratings.py`)**

1. Module docstring, append: "SP4 (spec block 7.4): ``RatingBook.state_dict()`` / ``load_state_dict(state, agents=...)`` save and restore every table (``league_state.json``), so a run-dir resume continues them; entries of agents that are not in the new config are dropped (one INFO line), new agents start empty."
2. Imports: `import logging`; `from collections.abc import Collection, Iterable, Mapping, Sequence`; after the imports `logger = logging.getLogger(__name__)`.
3. `_LayoutRatings`, after `role_win_rates`:
```python
    def state_dict(self) -> dict[str, Any]:
        wr = self.win_rates
        return {
            "outcome_kind": self.outcome_kind,
            "elo": dict(self.elo._ratings),
            "win_rates": {"scores": wr._scores, "weights": wr._weights, "games": wr._games},
            "past": {a: [list(entry) for entry in entries] for a, entries in self.past._scores.items()},
            "scores": {a: list(stats) for a, stats in self.scores._stats.items()},
            "cross_play": {key: list(cell) for key, cell in self.cross_play._cells.items()},
            "role_scores": self.role_scores,
        }

    def load_state_dict(self, state: Mapping[str, Any], known: Collection[str], dropped: set[str]) -> None:
        """Restore ``state_dict`` entries whose agents are all ``known``; the others' agents go into ``dropped``."""
        def ok(*agents: str) -> bool:
            unknown = [a for a in agents if a not in known]
            dropped.update(unknown)
            return not unknown

        self.elo._ratings = {a: float(r) for a, r in state["elo"].items() if ok(a)}
        tracker = WinRateTracker()
        for part, target in (("scores", tracker._scores), ("weights", tracker._weights), ("games", tracker._games)):
            for a, row in state["win_rates"][part].items():
                if ok(a):
                    target[a] = {b: (int(v) if part == "games" else float(v)) for b, v in row.items() if ok(b)}
        self.win_rates = tracker
        self.past._scores = {a: deque(((float(s), float(w)) for s, w in entries), maxlen=self.past._window)
                             for a, entries in state["past"].items() if ok(a)}
        self.scores._stats = {a: (int(v[0]), float(v[1]), float(v[2]), float(v[3]))
                              for a, v in state["scores"].items() if ok(a)}
        self.cross_play._cells = {key: [int(c[0]), float(c[1])] for key, c in state["cross_play"].items()
                                  if ok(*key.split("+"))}
        self.role_scores = {role: {a: {b: [float(c[0]), float(c[1])] for b, c in row.items() if ok(b)}
                                   for a, row in table.items() if ok(a)}
                            for role, table in state["role_scores"].items()}
```
4. `RatingBook`, before `snapshot`:
```python
    def state_dict(self) -> dict[str, Any]:
        """Every table of every layout, JSON-ready (``league_state.json``)."""
        return {"layouts": {layout: book.state_dict() for layout, book in self._books.items()}}

    def load_state_dict(self, state: Mapping[str, Any], *, agents: Collection[str]) -> list[str]:
        """Restore ``state_dict`` output. Entries of agents not in ``agents`` are dropped and a layout the game no
        longer has (or of another outcome kind) is skipped, each with one INFO line; returns the dropped agents."""
        dropped: set[str] = set()
        skipped = []
        for layout, saved in state.get("layouts", {}).items():
            book = self._books.get(layout)
            if book is None or saved.get("outcome_kind") != book.outcome_kind:
                skipped.append(layout)
                continue
            book.load_state_dict(saved, set(agents), dropped)
        if dropped:
            logger.info(f"League state: ratings of agents not in this config are dropped: {sorted(dropped)}")
        if skipped:
            logger.info(f"League state: ratings of layouts this game does not have are dropped: {sorted(skipped)}")
        return sorted(dropped)
```

- [ ] **Step 4: `PfspStats` (`src/colosseum/league/pfsp.py`)**

1. Module docstring: "A resumed run starts empty (as SP2 ratings)." becomes "A run-dir resume restores the statistics from ``league_state.json`` (SP4, ``state_dict`` / ``load_state_dict``); any other resume starts empty."
2. Imports: `import logging`, `from collections.abc import Collection, Mapping`, `from typing import Any`; `logger = logging.getLogger(__name__)` before `PlayerKey`.
3. `PfspStats`, before `snapshot`:
```python
    def state_dict(self) -> dict[str, Any]:
        """The statistics and the forgotten snapshots, JSON-ready (``league_state.json``)."""
        return {
            "prior": self._prior,
            "table": {layout: {owner: [[p[0], p[1], cell[0], cell[1]] for p, cell in sorted(players.items())]
                               for owner, players in owners.items()}
                      for layout, owners in self._table.items()},
            "forgotten": sorted([list(p) for p in self._forgotten]),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, agents: Collection[str]) -> list[str]:
        """Restore ``state_dict`` output: owners that are not tracked here and players of agents not in ``agents``
        are dropped (one INFO line); forgotten snapshots stay forgotten (also those forgotten here before the
        load). Returns the dropped agents."""
        known = set(agents)
        dropped: set[str] = set()
        self._forgotten |= {(str(p[0]), str(p[1])) for p in state.get("forgotten", ()) if p[0] in known}
        table: dict[str, dict[str, dict[PlayerKey, list[float]]]] = {}
        for layout, owners in state.get("table", {}).items():
            for owner, rows in owners.items():
                if owner not in self._halflife:
                    dropped.add(owner)
                    continue
                for agent, network, score, games in rows:
                    player = (str(agent), str(network))
                    if agent not in known:
                        dropped.add(agent)
                    elif player not in self._forgotten:
                        table.setdefault(layout, {}).setdefault(owner, {})[player] = [float(score), float(games)]
        self._table = table
        if dropped:
            logger.info(f"League state: PFSP statistics of agents not in this config are dropped: {sorted(dropped)}")
        return sorted(dropped)
```

- [ ] **Step 5: Coordinator, run dir, launcher**

`src/colosseum/coordinator/coordinator.py`, after `logger = ...`:
```python
# The league state of a run (SP4 spec block 7.4): ratings and PFSP statistics, restored by a run-dir resume.
LEAGUE_STATE_FILE = "league_state.json"
LEAGUE_STATE_VERSION = 1
```
and before `save_checkpoint_payload`:
```python
    def save_league_state(self, path: str | Path, *, env_steps: int) -> None:
        """Write the ratings (``RatingBook``) and the PFSP statistics (with the forgotten snapshots) atomically to
        ``path`` (spec block 7.4). A custom matchmaker's own state (``on_result``) is not saved."""
        from colosseum.metrics.jsonl import write_json_atomic

        write_json_atomic(path, {"version": LEAGUE_STATE_VERSION, "env_steps": int(env_steps),
                                 "ratings": self._ratings.state_dict(), "pfsp": self._pfsp.state_dict()})

    def load_league_state(self, path: str | Path) -> bool:
        """Restore a saved league state; False when ``path`` does not exist (an SP3 run dir). Entries of agents
        that are not in this config are dropped (INFO). An unreadable file is a ConfigError."""
        import json

        path = Path(path)
        if not path.is_file():
            return False
        try:
            state = json.loads(path.read_text())
            if state.get("version") != LEAGUE_STATE_VERSION:
                raise ValueError(f"version {state.get('version')!r}, expected {LEAGUE_STATE_VERSION}")
            players = list(self._player_roles)
            self._ratings.load_state_dict(state["ratings"], agents=players)
            self._pfsp.load_state_dict(state["pfsp"], agents=players)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            raise ConfigError(f"Cannot read the league state {path} ({type(e).__name__}: {e}); move it away to "
                              f"resume with empty ratings and PFSP statistics") from e
        return True
```

`src/colosseum/core/run_dir.py`: layout line `league_state.json        ratings and PFSP statistics, for a run-dir resume (SP4)` after `matches.db`, and before `matches_path`:
```python
    @property
    def league_state_path(self) -> Path:
        from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE

        return self.root / LEAGUE_STATE_FILE
```

`src/colosseum/launcher.py`:
1. `Launcher.__init__`, after `self._live_ratings = None`: `self._league_state_error_logged = False`.
2. `launch()`: between `resume_states = self._resolve_resume(...)` and `self._import_snapshot_pool(coordinator)` add
```python
        self._load_league_state(coordinator)          # before the import: evictions there forget PFSP entries
```
3. `_monitor_loop`: the tick becomes
```python
            if self._hub.maybe_tick(env_steps=env_steps, ratings=coordinator.ratings_snapshot,
                                    queue_depths=queue_depths(self._trajectory_queues),
                                    match_log=self._match_log_stats(), bt=self._live_snapshot()):
                self._save_league_state()
```
4. `_finish_metrics`: after the `self._hub.close(...)` call add `self._save_league_state()`.
5. New methods before `_import_snapshot_pool`:
```python
    def _load_league_state(self, coordinator: Coordinator) -> None:
        """A run-dir resume restores the resumed run's ratings and PFSP statistics (spec block 7.4); an SP3 run dir
        has no league_state.json (INFO: they start empty). A resume from a checkpoint dir or a .pt starts empty, as
        in SP3."""
        from pathlib import Path

        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source
        from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE

        resume_from = self._config.training.resume_from
        if not resume_from or classify_resume_source(resume_from) != RESUME_RUN_DIR:
            return
        path = Path(resume_from) / LEAGUE_STATE_FILE
        if coordinator.load_league_state(path):
            logger.info(f"Resume: league state (ratings, PFSP statistics) restored from {path}")
        else:
            logger.info(f"Resume: {resume_from} has no {LEAGUE_STATE_FILE} (an SP3 run dir); ratings and PFSP "
                        f"statistics start empty")

    def _save_league_state(self) -> None:
        """Write league_state.json (each metrics tick and at the end); a failure is an ERROR, never a crash."""
        if self._coordinator is None:
            return
        try:
            self._coordinator.save_league_state(self._run_dir.league_state_path,
                                                env_steps=int(self._env_step_counter.value))
        except Exception as e:  # noqa: BLE001 - the league state must never stop training
            logger.error(f"Writing {self._run_dir.league_state_path} failed ({type(e).__name__}: {e})",
                         exc_info=not self._league_state_error_logged)
            self._league_state_error_logged = True
```

- [ ] **Step 6: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_league_state.py tests/integration/test_sp4_league_state_resume.py tests/integration/test_sp4_sp3_resume.py tests/unit/test_sp2_ratings.py tests/unit/test_sp3_pfsp_stats.py tests/unit/test_sp2_coordinator.py tests/unit/test_sp3_coordinator_league.py tests/integration/test_sp2_league_runs.py -q -rw`
Expected: all pass.

- [ ] **Step 7: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/coordinator src/colosseum/league/pfsp.py src/colosseum/core/run_dir.py src/colosseum/launcher.py \
        tests/unit/test_sp4_league_state.py tests/integration/test_sp4_league_state_resume.py \
        tests/integration/test_sp4_sp3_resume.py
git commit -m "feat: league_state.json: ratings and PFSP statistics survive a run-dir resume"
git push
```

---

## Contract notes

**Dry run.** While writing this part, every task's code and tests were applied in order to a scratch copy of the branch (`sp4-selection` at `0b1422f`, docs only before T0.1), as an implementer would: each task's own tests passed, and the full fast suite passed with zero warnings after T1.1 (1737 tests), T1.3 (1775), T2.2 (1811, once the `test_sp2_game_runs.py` key check found by that run was updated) and T3.2 (1823), with `ruff check .` clean; the slow suite after T3.2: 10 passed in 408 s, every SP2/SP3 threshold unchanged. Findings of the dry run that are built into the tasks: two SP2 tests pin the `ratings.json` keys (`test_sp2_metrics_outputs.py`, `test_sp2_game_runs.py`, both updated in T2.2); `test_sp2_launcher_lifecycle.py` queues a string as a fake result (T1.1 makes it a `MatchResult`, because results are now stamped); `tests/conftest.py` fails the suite if anything is written into the repo root; `PRAGMA incremental_vacuum` needs `.fetchall()`; the per-match size estimate needed an index factor of 1.2 and the uid counted twice; `LogRecord.exc_info` is `False`, not `None`, when `exc_info=False`.

### Additions and deviations versus `00-overview.md`

1. **`colosseum.arena` and `entities.py` are created by T1.1, not T1.2** — `MatchRunner` needs `snapshot_version` in T1.1. T1.2 adds `comparisons.py` and `matchlog.py`.
2. **`run_id` is created by T1.1, stored in `run.json`** (`{"run_id": "<uuid4 hex>"}`, `RUN_INFO_FILE`), written once by `Launcher.launch` through `RunDir.ensure_run_id()`; `RunDir.run_id` is a read-only property, `read_run_id(root)` never writes (an SP3 parent has none: `parent_run_id = None`). `RunDir.create` does not write it (`test_sp2_run_dir.py:127` pins what `create` leaves). Proposed for T7.1: keep `run.json` as the immutable identity and put the mutable status into `run_state.json` (no `run_id` there).
3. **`MatchLog(path, run: RunInfo | None, ...)`** — `None` for a writer that is not a training run (a tournament): its matches have no `runs` row and their rating versions come from network ids only (bucket width 1).
4. **`MatchLog.append(results, *, tournament=None)`** and a `matches.tournament` column (an id of the `entities` table) — the tournament id must be stored to read "this tournament's matches only" (spec block 2); `comparison_table(sources, tournament=X)` reads only that tournament's detailed matches, `tournament=None` only matches outside tournaments; `counts` never hold tournament matches. T5.4 appends one whole block per call.
5. **`MatchLog.compact(..., should_stop=...)`** — the writer interrupts compaction between batches at stop (spec block 1); **`MatchLog.rating_versions(result)`** and **`AppendStats.accepted`** — the writer feeds the in-memory table only with journaled results (duplicates are not added twice).
6. **`ScoreStat(n, total, total_sq)`** — the overview's "n, sum, sumsq" with names that do not shadow builtins.
7. **`MatchLogReader.matches(..., version, outcome, env_steps)`** — the spec's filters (agent, version, layout, source, outcome, env-step range); `MatchRow` / `SeatRow` are the row types; `MatchLogReader.close()`; a missing file reads as an empty journal.
8. **`counts` keys are id-encoded** (`"<team>|<entity id>:<role id>,..."`), not names — about 95 bytes per row instead of about 240 (measured on 2v2 data); `import_parent` re-encodes the parent's keys.
9. **`bt` additions:** `BTFit.bias_se`, `BTFit.approximate`, `BTFit.group_of`; `BTGroup.anchor_members`, `rule`, `note`; `fit(..., agent_order=())` (the "first trainable agent" rule needs the config order); `Uncertainty.p_top_k_group` (the spec asks for `P(top-k)` within the lineage and within the group: `p_top_k` is the lineage's) and `Uncertainty.contrast_samples` (Elo, float32: T4.1 averages a snapshot's score over layouts on each sample, spec block 4); `BTOptions` defaults are the module constants and gain `dense_limit`; `DEFAULT_OPTIONS`; `chi2_sf`; `TransitivityResult`, `ScoreSummary`, `ScoreCell` spelled out.
10. **`BootstrapData(counts, matches, blocks)` and `Contribution`** — the block data interface shared with T5.3: `blocks = {stratum: {key: [contributions]}}`; per replicate and stratum the keys are drawn with replacement, the same draw for every contribution of a key. For the paired stratified tournament bootstrap a stratum is a field member (or FFA composition) and a key a seed index shared by every candidate's block on that seed, so one key holds the contributions of all candidates' blocks with that seed. T5.3 may build the replicates itself for candidate scores, but BT intervals of a tournament go through `bt.bootstrap` with this structure.
11. **`live` additions:** `LiveRatings(..., agent_order, samples, seed)`, `refit_now()`, `last_refit_sec`; `RatingsSnapshot.transitivity`, `refit_sec`; `compute_snapshot`, `random_bot_agent(config)`, `console_bt`, `LAPLACE_SAMPLES`, `FINAL_REFIT_MAX_SEC`. The main loop learns that a new snapshot was published by comparing `latest().version` (T4.1's retention after each refit).
12. **`ratings.json` `bt` layout:** `bt[layout][<role group label>]` = entity rows; per-layout extras under keys that start with `@` (`@bias`, `@anchors`, `@transitivity`, `@fit`, `@score`), which a role name cannot (role names may start with `_` but not with `@`: `envs/game.py` `_NAME_RE`). The group label is the `+`-joined sorted role names of the merged role group (`"player"`, `"hunter"`, `"hunter+prey"`), not the overview's example `"default"`.
13. **The hub compacts** the full ratings snapshot itself (`league.pfsp.compact_ratings`, also behind `Coordinator.ratings_snapshot(compact=True)`): `ratings.json` needs the full table and the record the compact one, from one snapshot built per tick. `MetricsHub.maybe_tick` / `close` take `ratings` as a dict or a callable and gain `match_log=` and `bt=`; `ConsoleReporter.format_line(..., bt=None)`.
14. **`Coordinator.add_evict_listener(listener)`** (T1.3) — how evictions reach the journal writer (`note_evicted`) without the thread calling coordinator methods; T4.1 keeps calling `on_evict` on the main loop, so listeners keep working.
15. **`Coordinator.save_league_state(path, *, env_steps)`** — the file records the env-step counter it was written at.
16. **`system` record:** new required key `dropped_match_results` and the key `match_log` (`WriterStats.as_dict()` or null). `SystemStats.cumulative(key)` and `CUMULATIVE_WORKER_KEYS` (`dropped_results`, from T3.1 also `dropped_reward_episodes`). The journal sizes in `match_log` are refreshed at each compaction pass (and at start and stop), so they lag by up to `compact_interval_sec`.
17. **Worker side (T1.1):** `LoopIO.report_result` returns False for a dropped result; `RolloutLoop.stats["dropped_results"]`; `report_worker_stats(..., timeout=None)`; on exit a final stats report (blocking up to 1 s) and `flush_queue` of the results and stats queues — without the flush, the last results and the final counters could be lost silently at exit (`cancel_join_thread`), and "exactly once or counted" would not hold.
18. **`RUN_SOURCES = ("train", "pulse")`** lives in `arena.matchlog` (the default sources of `comparison_table`; the live ratings use it).
19. **`MatchLogConfig` / `RatingsConfig`** are exactly the overview's; `ratings.anchor` is checked by `validate_config` in the "train" scope (`colosseum validate`, `train`), which also prints the journal and rating settings.

### Proposed rulings (what — why — cost if wrong)

1. **Coverage band of the Bradley–Terry test: at least 0.85 over 20 replications per scenario** (nominal 0.95, tolerance 0.10) — measured run-to-run sd up to 0.030 (asymmetric roles with a bridge), so 0.85 keeps a correct implementation's false-failure probability below 1e-3 in every scenario; 0.87 did not — cost: an implementation whose intervals cover 0.86–0.90 passes; T8.2 may tighten the band after fixing prior and smoothness.
2. **Self-play, generalized:** an entity gets no game from a pair in which its net coefficient is zero (`coefficients`), which covers both "an entity against itself" and partial overlaps such as `{x, y}` vs `{x, z}` (x cancels) — the spec's rule is a special case of this — cost: none known.
3. **A configured anchor that names a trainable agent** is its lowest version in the cell (normally `v0`) — the spec says "имя агента конфига" without saying which entity of a trainable agent — cost: another zero point than the owner expects.
4. **Last anchor fallback: the first entity by name** (cells with no fixed agent and no trainable lineage, e.g. tournament-only names) — the spec's chain ends at `v0` of the first trainable agent — cost: an arbitrary zero for such cells; tournaments pass their own anchor (T5.3).
5. **Priors:** the free bias parameters get the same Gaussian prior as strengths; the smoothness chain spans every version of a lineage regardless of components (components are computed from data only) — needed for finite, identifiable estimates without a bridge — cost: the bias of an unbridged asymmetric layout is prior-driven (it is not interpretable there anyway).
6. **G²:** pairs are team-pair rows with at least `min_games` matches; `dof = pairs − (entities − cells) − (T − 1)`; with fractional FFA weights and draws the χ² reference is approximate — the spec leaves the parameter count open — cost: approximate p-values in FFA.
7. **Bootstrap of compacted rows:** wins ~ Poisson(Σw·y / w), losses ~ Poisson(n − Σw·y / w) — the outcome multiset of a compacted row is not stored — cost: intervals slightly too wide with many draws on compacted data (detailed matches and blocks are resampled exactly).
8. **"Faster than expected" growth of the untouched parts** (spec block 1) = file size minus the detail estimate above `max(detail window, 64 MiB)`: one WARNING — cost: slower growth stays silent (visible in the `system` sizes).
9. **Disk guard:** training matches folded straight into `counts` keep no uid, so a duplicate cannot be detected while the guard is on — training uids are unique by construction — cost: none in practice.
10. **Journal quarantine** only for "not a database" / "malformed" errors; any other SQLite error while opening at start-up is a `ConfigError` with the hint to move the file or disable the journal — cost: a locked or unreadable file stops the start instead of being replaced.
11. **Exit accounting:** workers flush their results and stats queues on exit (`flush_queue`, waiting while the parent lives); the writer flushes within `SHUTDOWN_GRACE_SEC` after the children are gone, so the main process may take up to that much longer to exit in the worst case (normally milliseconds) — the 10 s rule concerns children — cost: a slower exit when the journal's disk is slow.
12. **Live refit cadence:** the first refit runs one interval after the start; a refit is skipped while the table is unchanged; at the end one final refit runs if the last refit took at most 2 s — cost: short runs with a long interval publish `bt` only from the final refit.
13. **Without the journal** (`match_log.enabled: false`) there are no live ratings: `ratings.json` has `"bt": {}` and the records no `bt` key — cost: none (the spec makes ratings functions of the journal).
14. **`bt_summary` "best"** = the agent's entity with the highest rating (ties: the newer version), not the highest lower bound — the record shows ratings with intervals; top-k uses LCB (T4.1) — cost: a noisy best in the compact record.
15. **League state** is loaded before the snapshot pool import, and forgotten snapshots are unioned on load — evictions during the import must still forget their PFSP entries (SP3 ruling #50) — cost: none. A malformed `league_state.json` at resume is a `ConfigError` with a "move it away" hint (the file is written atomically, so this means tampering) — cost: a resume stops instead of starting empty.
16. **BLAS limit:** the thread variables are set around `import numpy` in `colosseum train` and restored at once — OpenBLAS (numpy's wheels) and OpenMP read them at load; a BLAS that reads them later (some MKL builds) is not limited — cost: on such a build the refit may use several cores.
17. **`docs/benchmarks.md` keeps its recorded tables** (column `c_clip_frac`, `docs/benchmarks/*.json`) and explains the replacement in its text — measurements are history — cost: none.
18. **Result stamping:** `_drain_results` stamps `env_steps` only on results that have none (a pulse result arrives with its snapshot's, T6.1) — cost: none.

### Assumptions about other parts

1. **T4.1** sets `LiveRatings(top_k=checkpoint.keep_top_k, quantile=checkpoint.top_k_quantile)` (T2.2 passes `top_k=0`), reads `Uncertainty.lcb` / `contrast_samples` per layout and `BTFit.games` (self-play excluded) for `top_k_min_games`, and runs retention on the main loop when `LiveRatings.latest().version` changes; its evictions keep reaching the journal through `Coordinator.add_evict_listener`.
2. **T5.x** writes tournament matches with `MatchLog(path, None)` and `append(block_results, tournament=<id>)` (one whole block per call), reads them with `MatchLogReader.comparison_table(("tournament",), tournament=<id>)` and `blocks(<id>)`, and builds `BootstrapData.blocks` as in note 10. Tournament seats are named by `agent_id` = the tournament player name and network `latest` without `version_start`, so `entity_key` gives the name itself.
3. **T6.1** sends pulse results through the training results queue with `source="pulse"`, `block` set and `env_steps` from the snapshot's `meta.json` (kept by ruling 18); the writer journals them in detail and adds them to the live table (`RUN_SOURCES`).
4. **T7.1** reads `ratings.json["bt"]` in the format of note 12, the `system` keys of note 16, and `MatchLogReader` for matches; `metrics.jsonl` of SP3 runs (the T0.1 fixture) carries `rho_clip_frac` / `c_clip_frac` instead of the new names.
5. **T8.1** treats `match_log` and `ratings` as no-ops in the distributed roles (they produce no results, spec block 8); `scripts/bench_throughput.py::_make_config` already sets both sections (T1.1, T2.2).
6. **T8.2** fixes `bt.PRIOR_SIGMA` / `bt.SMOOTHNESS` (tests read the constants) and may revisit ruling 1; the refit cost numbers are in T2.2's report (step 8).

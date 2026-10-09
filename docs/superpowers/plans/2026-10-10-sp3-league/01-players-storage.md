# SP3 Plan — Part 1: Compatibility fixtures, agents and players, player execution, snapshot storage, PFSP statistics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following spec blocks of `docs/superpowers/specs/2026-10-10-sp3-league-design.md` (read the spec first; it is the binding authority):
- **Section 6, "Совместимость (первая задача)":** the SP2 fixtures (config copies, a checkpoint made by SP2 code) and the baseline tests (T0.1).
- **Block 1 (agents and players in the config):** `kind` (`trainable` / `scripted` / `frozen`), the implicit trainable `agent_0`, the reserved network id `"fixed"`, the agent registry that replaces `coordinator/agent_pool.py`, `--set` for every kind, `validate` of scripted and frozen agents (T1.1, T1.4, T1.5). The checks of player names in `anchors`, `kickstart.teacher` and `init.from` belong to T3.3 / T4.x (those fields do not exist before).
- **Block 2 (scripted bots):** `ScriptedBot`, `RandomBot`, instances per (agent, env, seat), `rng` per episode and seat, the legality gate and `PlayerError` (T1.2, T1.3). Demo-game bots are T6.2.
- **Block 3 (player execution):** the player pool of `MatchRunner` (neural players batched, scripted bots one by one, `infos`), `collect` only on latest seats, frozen agents on the worker, `eval -a name`, `play_lineups` with bots (T1.3–T1.5); snapshot eviction on workers (T2.2).
- **Block 4 (snapshot storage):** `keep_last` / `keep_every` / final, `trainer_state.pt` only in the `keep_last` window, the run-dir resume import of the pool, the eviction chain storage → coordinator → workers (T2.1, T2.2).
- **Block 5, statistics and ratings part:** PFSP statistics per player (`MemberPair` with network ids, EMA with `halflife_games`, `forget`), scripted and frozen agents as rating entities, the PFSP table in `ratings.json`, and the `anchor` opponent type of the episode aggregator (T2.3). The matchmaker itself (categories, shares, schedules, `BaseMatchmaker`) is Part 2 (T3.1–T3.3).

**What the part delivers.** After T2.3: configs declare scripted and frozen agents next to trainable ones; every match core (training worker, eval) seats them correctly (bots see `obs`, `mask`, `infos[seat]`; frozen agents run their own architecture; neither ever collects); `colosseum eval -a greedy` works by name; `colosseum validate` plays every bot and loads every frozen agent; snapshots are stored by `keep_last` + `keep_every` + final, carried over a run-dir resume and unloaded by workers once unused; the coordinator keeps PFSP statistics per player (latest of another agent, any snapshot, any anchor) and rates scripted and frozen agents. The SP2 matchmaker (`LineupMatchmaker`) still builds the lineups until T3.2, so in T1.x/T2.x tests fixed players are seated through explicit lineups.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Run every command from the repository root with `.venv/bin/python` / `.venv/bin/ruff`.
- "Full fast suite" means `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `.venv/bin/ruff check .`.
- New test files have basenames that start with `test_sp3_` (unique across `tests/`); shared test support goes into `tests/game_helpers.py`.
- Tests that drive a `RolloutLoop` through `game_harness` (`make_loop`, `GameFactory`) live in `tests/contract/`: `game_harness` is importable only from there (pytest's prepend import mode puts the test file's own dir on `sys.path`).
- Commit messages use conventional prefixes and carry no attribution lines; push after every task (`git push origin sp3-league`).
- Translation and deprecation messages go through `logging` (never `warnings.warn`), so `-rw` stays empty.

**Cross-task notes.**
- **T0.1 runs on unchanged SP2 code** and creates the compatibility baseline. Its tests are extended by later tasks: T1.2 (the SP2 checkpoint as a frozen agent: `load_fixed_players`), T1.5 (the SP2 checkpoint as a frozen agent by name in `eval` and in `validate`), T2.1 (a run-dir resume of the SP2 run imports `ckpt_v3` into the new run's pool), and in other parts T3.1 (SP2 matchmaking knobs of the config copies translate with one warning each), T4.1 (`training.kickstart_*` translate), T4.2 (the SP2 checkpoint as `init`, from its checkpoint dir and from its run dir), T4.4 (the SP2 checkpoint as a kickstart teacher).
- **The SP2 matchmaker stays until T3.2.** It is built from the trainable agents only (`Coordinator` passes it the trainable roles), so declaring a scripted or frozen agent never changes SP2 lineups before T3.2. `validate_matchmaking` of SP2 still requires every role to be played by a trainable agent until T3.2 replaces it.
- **Import cycles.** `colosseum.players.scripted` imports `colosseum.core.validation` (for `random_legal_action`) at module level; therefore `colosseum.core.validation` imports `colosseum.players.*` only inside functions.
- **Fixtures are read-only.** Tests never write under `tests/fixtures/`: resume tests copy the SP2 run dir into `tmp_path` first (`copy_sp2_run`); eval and `load_checkpoint_dir` read it in place (they never write).
- **Shared helpers this part adds to `tests/game_helpers.py`:** `SP2_FIXTURES`, `SP2_CONFIGS`, `SP2_CONFIG_NAMES`, `SP2_TTT_TINY`, `SP2_RUN`, `SP2_CHECKPOINT`, `SP2_CHECKPOINT_VERSION`, `SP2_CHECKPOINT_ENV_STEPS`, `copy_sp2_run` (T0.1); `RecordingBot`, `ConstantBot`, `scripted_agent`, `frozen_agent`, `scripted_player` (T1.2); `TickGame(..., infos=True)` (T1.3); `make_coordinator` builds the coordinator from `resolve_player_roles` (T1.4); `make_test_config` uses `keep_last`/`keep_every` (T2.1).

---

### Task T0.1: SP2 compatibility fixtures and baseline tests

Spec section 6 ("Совместимость (первая задача)") and criterion 3.6. Before any SP3 change, SP2 code produces (a) byte copies of every `configs/examples/*.yaml` (the live examples change during SP3, e.g. `team_tag.yaml` in T6.3 and `pool_size` → `keep_last` in T2.1) and (b) a small SP2 checkpoint (`model.pt`, `trainer_state.pt`, `meta.json`) inside an SP2-shaped run dir, made through SP2's own learner payload and checkpoint helpers. The baseline tests pass on SP2 code now and must stay green through SP3.

**Files:**
- Create: `tests/fixtures/sp2/configs/{chase,coin_grid,coop_buttons,predator_prey,space_miners,team_tag,tic_tac_toe,tic_tac_toe_attention,tic_tac_toe_multi,tron,unit_harvest}.yaml` (byte copies of `configs/examples/*.yaml`)
- Create: `tests/fixtures/sp2/sp2_ttt_tiny.yaml` (the SP2-format config of the checkpoint fixture)
- Create: `tests/fixtures/sp2/make_sp2_checkpoint.py` (generator, run once now; committed with its output)
- Create (generated): `tests/fixtures/sp2/run/checkpoints/agent_0/ckpt_v3/{model.pt,trainer_state.pt,meta.json}`
- Modify: `.gitignore` (the repo ignores `checkpoints/` and `*.pt`; the fixture must be committed)
- Modify: `tests/game_helpers.py` (fixture paths, `copy_sp2_run`)
- Test: `tests/unit/test_sp3_sp2_compat.py`, `tests/integration/test_sp3_sp2_resume.py`

**Interfaces:**
- Consumes (SP2, unchanged): `colosseum.learner.learner.make_checkpoint_payload`, `colosseum.coordinator.checkpoint_manager.CheckpointManager(base_dir).save(...)`, `load_checkpoint_dir`, `colosseum.algorithms.appo.APPO`, `colosseum.core.registry.{build_model, env_spec}`, `colosseum.core.roles.{resolve_agent_roles, agent_role_spec, role_signature}`, `colosseum.core.config.{load_config, config_hash}`, `game_helpers.synthetic_chunk`, `cli_runner.run_train`, `colosseum.launcher.{Launcher, setup_run}`.
- Produces (test support, `tests/game_helpers.py`):
  - `SP2_FIXTURES: Path`, `SP2_CONFIGS: Path`, `SP2_CONFIG_NAMES: tuple[str, ...]`, `SP2_TTT_TINY: Path`, `SP2_RUN: Path`, `SP2_CHECKPOINT: Path`, `SP2_CHECKPOINT_VERSION = 3`, `SP2_CHECKPOINT_ENV_STEPS = 1536`;
  - `copy_sp2_run(tmp_path) -> Path` (a writable copy of `SP2_RUN`).

- [ ] **Step 1: Copy the SP2 example configs**

Run (on the unchanged SP2 tree; `git status` must be clean apart from this part's plan files):
```bash
mkdir -p tests/fixtures/sp2/configs
cp configs/examples/*.yaml tests/fixtures/sp2/configs/
ls tests/fixtures/sp2/configs | wc -l   # 11
```

- [ ] **Step 2: Write the fixture config and the generator**

Create `tests/fixtures/sp2/sp2_ttt_tiny.yaml`:
```yaml
# SP2-format config of the SP2 checkpoint fixture (tests/fixtures/sp2/run): tic-tac-toe with a tiny MLP.
# Written for SP2 and kept in its SP2 form (old knobs: matchmaking.mode / latest_prob, checkpoint.pool_size).
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
  total_timesteps: 3000
  seed: 0
matchmaking:
  mode: self_play
  latest_prob: 0.8
checkpoint:
  interval: 20
  pool_size: 5
  save_optimizer: true
metrics:
  use_wandb: false
  log_interval: 1
  console_interval_sec: 1.0
```

Create `tests/fixtures/sp2/make_sp2_checkpoint.py`:
```python
"""Generator of the SP2 checkpoint fixture (SP3 task T0.1). Run ONCE on SP2 code; the output is committed.

    .venv/bin/python tests/fixtures/sp2/make_sp2_checkpoint.py

Writes ``tests/fixtures/sp2/run/checkpoints/agent_0/ckpt_v3/`` (``model.pt``, ``trainer_state.pt``,
``meta.json``) for ``sp2_ttt_tiny.yaml``: three APPO updates on synthetic chunks, then the payload and
``meta.json`` exactly as SP2's launcher saves them (``learner.make_checkpoint_payload`` +
``Coordinator.save_checkpoint_payload`` + ``Launcher._save_checkpoint``'s meta). It uses only APIs that
existed in SP2 and is a record of how the fixture was made; SP3 tasks never re-run it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch  # noqa: E402

from colosseum.algorithms.appo import APPO  # noqa: E402
from colosseum.coordinator.checkpoint_manager import CheckpointManager  # noqa: E402
from colosseum.core.config import config_hash, load_config  # noqa: E402
from colosseum.core.registry import build_model, env_spec  # noqa: E402
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature  # noqa: E402
from colosseum.core.specs import ActionSpec  # noqa: E402
from colosseum.learner.learner import make_checkpoint_payload  # noqa: E402
from game_helpers import synthetic_chunk  # noqa: E402

AGENT = "agent_0"
ENV_STEPS = 1536


def main() -> None:
    torch.manual_seed(0)
    config = load_config(HERE / "sp2_ttt_tiny.yaml")
    spec = env_spec(config)
    roles = resolve_agent_roles(config, spec)[AGENT]
    role = agent_role_spec(spec, roles)
    agent_config = config.get_agent_config(AGENT)
    model = build_model(agent_config, role)
    algo = APPO(model, agent_config.algorithm, ActionSpec.from_space(role.action_space), device="cpu")
    for step in range(3):
        chunks = [synthetic_chunk(model, role, "AAATAAAB", seed=10 * step + i, agent_id=AGENT) for i in range(2)]
        algo.train_step(chunks)
    payload = make_checkpoint_payload(AGENT, algo, final=True)
    shutil.rmtree(HERE / "run", ignore_errors=True)
    meta = {
        "final": bool(payload["final"]),
        "networks": agent_config.networks.model_dump(mode="json", by_alias=True),
        "config_hash": config_hash(config),
        "env_steps": ENV_STEPS,
        "roles": list(roles),
        "role_signature": role_signature(role),
    }
    out = HERE / "run" / "checkpoints"
    ckpt_id = CheckpointManager(out).save(
        agent_id=AGENT, policy_version=int(payload["policy_version"]), model_state=payload["model_state"],
        trainer_state=payload["trainer_state_bytes"], meta_extra=meta,
    )
    print(f"wrote {out / AGENT / ckpt_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the checkpoint and let git see it**

Append to `.gitignore`:
```
# SP2 compatibility fixtures (SP3 T0.1): a committed SP2 checkpoint inside an SP2-shaped run dir
!/tests/fixtures/sp2/run/checkpoints/
!/tests/fixtures/sp2/run/checkpoints/**
```
Run:
```bash
OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp2/make_sp2_checkpoint.py
git status --porcelain -uall tests/fixtures .gitignore
```
Expected: `wrote .../tests/fixtures/sp2/run/checkpoints/agent_0/ckpt_v3`; `git status` lists the 11 config copies, `sp2_ttt_tiny.yaml`, the generator and the three files of `ckpt_v3` (about 25 KB together; `meta.json` has `policy_version: 3`, `env_steps: 1536`, `final: true`, `roles: [player]`).

- [ ] **Step 4: Add the fixture paths to `tests/game_helpers.py`**

Add `import shutil` and `from pathlib import Path` to the module imports (and drop the function-local `from pathlib import Path` in `write_test_config`), then append:
```python
# ---------------------------------------------------------------------------
# SP3 (T0.1): SP2 compatibility fixtures (read-only inputs; copy before writing)
# ---------------------------------------------------------------------------


SP2_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sp2"
SP2_CONFIGS = SP2_FIXTURES / "configs"
SP2_CONFIG_NAMES = ("chase", "coin_grid", "coop_buttons", "predator_prey", "space_miners", "team_tag",
                    "tic_tac_toe", "tic_tac_toe_attention", "tic_tac_toe_multi", "tron", "unit_harvest")
SP2_TTT_TINY = SP2_FIXTURES / "sp2_ttt_tiny.yaml"
SP2_RUN = SP2_FIXTURES / "run"
SP2_CHECKPOINT = SP2_RUN / "checkpoints" / "agent_0" / "ckpt_v3"
SP2_CHECKPOINT_VERSION = 3
SP2_CHECKPOINT_ENV_STEPS = 1536


def copy_sp2_run(tmp_path) -> Path:
    """A writable copy of the SP2 fixture run dir under ``tmp_path`` (tests never write into tests/fixtures)."""
    dst = Path(tmp_path) / "sp2_run"
    shutil.copytree(SP2_RUN, dst)
    return dst
```

- [ ] **Step 5: Write the baseline tests**

Create `tests/unit/test_sp3_sp2_compat.py`:
```python
"""SP2 compatibility baseline (SP3 T0.1): the SP2 example configs (copies) validate and re-read cleanly,
and the SP2 checkpoint fixture loads for eval and resume.

Created on SP2 code; it must stay green through SP3. Later tasks add their own compatibility checks:
T1.2 / T1.5 (the checkpoint as a frozen agent), T2.1 (run-dir resume imports the pool), T3.1 (SP2
matchmaking knobs), T4.1 (training.kickstart_*), T4.2 (the checkpoint as init), T4.4 (as a teacher).
"""
from __future__ import annotations

import json
import logging

import pytest
import yaml
from click.testing import CliRunner

from cli_runner import REPO_ROOT
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import load_checkpoint_dir
from colosseum.core.config import load_config
from colosseum.core.registry import env_spec
from colosseum.core.roles import role_signature
from colosseum.launcher import Launcher, setup_run
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_CHECKPOINT_ENV_STEPS,
    SP2_CHECKPOINT_VERSION,
    SP2_CONFIG_NAMES,
    SP2_CONFIGS,
    SP2_TTT_TINY,
    make_test_run_dir,
)

SP2_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks",
                 "policy_version", "role_signature", "roles", "timestamp"}


def test_the_fixture_has_a_copy_of_every_sp2_example_config():
    assert sorted(p.stem for p in SP2_CONFIGS.glob("*.yaml")) == sorted(SP2_CONFIG_NAMES)


@pytest.mark.parametrize("name", SP2_CONFIG_NAMES)
def test_sp2_config_copies_validate_without_edits(name, monkeypatch):
    if name == "space_miners":
        # importorskip also silences Box2D's SWIG import-time DeprecationWarnings.
        pytest.importorskip("Box2D", reason="space_miners needs Box2D (pip install -e '.[examples]')")
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["validate", "-c", str(SP2_CONFIGS / f"{name}.yaml")])
    assert result.exit_code == 0, result.output
    assert "Config is valid." in result.output


@pytest.mark.parametrize("name", SP2_CONFIG_NAMES)
def test_the_resolved_form_of_an_sp2_config_rereads_without_warnings(name, tmp_path, caplog):
    """What a run writes as config.resolved.yaml (model_dump by alias) loads again silently and equal."""
    config = load_config(SP2_CONFIGS / f"{name}.yaml")
    resolved = tmp_path / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        again = load_config(resolved)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert again == config


def test_the_checkpoint_fixture_is_an_sp2_checkpoint_of_the_tiny_config():
    assert sorted(p.name for p in SP2_CHECKPOINT.iterdir()) == ["meta.json", "model.pt", "trainer_state.pt"]
    meta = load_checkpoint_dir(SP2_CHECKPOINT)["meta"]
    assert set(meta) == SP2_META_KEYS
    assert meta["policy_version"] == SP2_CHECKPOINT_VERSION and meta["env_steps"] == SP2_CHECKPOINT_ENV_STEPS
    assert meta["final"] is True and meta["roles"] == ["player"]
    spec = env_spec(load_config(SP2_TTT_TINY))
    assert meta["role_signature"] == role_signature(spec.roles["player"])


def test_eval_plays_the_sp2_checkpoint_dir_and_leaves_it_untouched(tmp_path):
    before = sorted(p.name for p in SP2_CHECKPOINT.parent.iterdir())
    out = tmp_path / "result.json"
    result = CliRunner().invoke(main, ["eval", "-c", str(SP2_TTT_TINY), "-a", f"old={SP2_CHECKPOINT}", "-n", "2",
                                       "--num-envs", "2", "--seed", "0", "-o", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["agents"] == ["old"] and report["layouts"]["2p"]["n"] == 2
    assert sorted(p.name for p in SP2_CHECKPOINT.parent.iterdir()) == before


def test_resume_from_the_sp2_checkpoint_dir_restores_version_trainer_state_and_env_steps(tmp_path):
    config = load_config(SP2_TTT_TINY, {"training.resume_from": str(SP2_CHECKPOINT)})
    launcher = Launcher(config, make_test_run_dir(config, tmp_path, name="resumed"))
    setup = setup_run(config, validate=False)
    states = launcher._resolve_resume(setup.agent_configs, setup.role_specs)
    assert states["agent_0"]["policy_version"] == SP2_CHECKPOINT_VERSION
    assert states["agent_0"]["trainer_state"] is not None
    assert launcher.env_steps_done == SP2_CHECKPOINT_ENV_STEPS
```

Create `tests/integration/test_sp3_sp2_resume.py`:
```python
"""SP2 compatibility baseline (SP3 T0.1): `colosseum train` resumes from the SP2 fixture run dir."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP2_CHECKPOINT_ENV_STEPS, SP2_CHECKPOINT_VERSION, SP2_TTT_TINY, copy_sp2_run


def _versions(agent_dir) -> list[int]:
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def test_training_resumes_from_the_sp2_run_dir(tmp_path):
    source = copy_sp2_run(tmp_path)
    # tiny=False: the fixture config is already tiny, and cli_runner.TINY's new-style keys (checkpoint.keep_last
    # from T2.1 on) next to the config's SP2 knobs (pool_size) would be "old + new together", a ConfigError.
    run = run_train(SP2_TTT_TINY, tmp_path, name="resumed", overrides={"training.resume_from": str(source)},
                    tiny=False)
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert "Resume [agent_0]:" in log and f"(policy_version {SP2_CHECKPOINT_VERSION})" in log
    assert f"env-step counter continues from {SP2_CHECKPOINT_ENV_STEPS}" in log
    versions = _versions(run.root / "checkpoints" / "agent_0")
    assert versions and min(versions) > SP2_CHECKPOINT_VERSION       # T2.1: the imported ckpt_v3 joins the pool
    assert _versions(source / "checkpoints" / "agent_0") == [SP2_CHECKPOINT_VERSION]  # the source is untouched
```

- [ ] **Step 6: Run the baseline tests (they pass on SP2 code)**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_sp2_compat.py tests/integration/test_sp3_sp2_resume.py -q -rw`
Expected: all pass (`space_miners` validate is skipped without Box2D); this is a baseline, not a failing-first test. If anything fails, the fixture (not the code) is wrong: fix the fixture.

- [ ] **Step 7: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings; ruff clean (the generator's `E402` imports carry `noqa`).

- [ ] **Step 8: Commit and push**

```bash
git add .gitignore tests/fixtures/sp2 tests/game_helpers.py tests/unit/test_sp3_sp2_compat.py \
        tests/integration/test_sp3_sp2_resume.py
git commit -m "test: SP2 compatibility fixtures (example config copies, SP2 checkpoint) and baseline tests"
git push origin sp3-league
```

---

### Task T1.1: Agent kinds in the config (`kind`, implicit `agent_0`)

Spec block 1. `agents.<id>` becomes a discriminated union on `kind` (`trainable` by default, `scripted`, `frozen`) with per-kind fields and `extra="forbid"`; an unknown field is a `ConfigError` naming the kind's fields (and, for a trainable entry with `class` / `path`, the kind it probably meant). Without any trainable agent, an implicit trainable `agent_0` with the global settings exists. `--set agents.<id>.<field>=...` works for every kind. `matchmaking`, `init` and `kickstart` exist on trainable entries but are rejected ("not supported yet") until T3.1 / T4.1 give them meaning, so no setting is ever silently ignored.

**Files:**
- Modify: `src/colosseum/core/config.py`
- Test: `tests/unit/test_sp3_agent_kinds.py`
- No other file changes: `scripts/bench_throughput.py::_make_config` has no `agents` section (the implicit `agent_0` is unchanged), and every SP2 caller uses `get_trainable_agent_ids` / `get_agent_config` / `agent_roles`, whose results for SP2 configs do not change.

**Interfaces:**
- Consumes: `StrictModel`, `deep_merge`, `check_agent_id`, `ConfigError`.
- Produces (contract T1.1, plus additions marked *):
  - `AgentKind = Literal["trainable", "scripted", "frozen"]`;
  - `TrainableAgent` (`kind`, `roles`, `networks`, `algorithm`, `learner`, `matchmaking`, `init`, `kickstart`), `AgentOverride = TrainableAgent`, `ScriptedAgent` (`kind`, `class_path` alias `class`, `kwargs`, `roles`), `FrozenAgent` (`kind`, `path`, `networks`, `roles`), `AgentEntry` (the annotated union);
  - `ColosseumConfig.agents: dict[str, AgentEntry]`; methods `get_trainable_agent_ids()`, `fixed_agent_ids()`, `agent_entry(id)`, `agent_kind(id)`, `get_agent_config(id)` (trainable only), `agent_roles(id)` (any kind), *`agent_ids()` (every agent in config order, the implicit `agent_0` first);
  - *`IMPLICIT_AGENT_ID = "agent_0"`;
  - `StrictModel.model_config` gains `populate_by_name=True`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_agent_kinds.py`:
```python
"""Agent kinds in the config (SP3 T1.1, spec block 1): trainable / scripted / frozen, implicit agent_0, --set."""
from __future__ import annotations

import pytest
import yaml

from colosseum.core.config import (
    AgentOverride,
    ColosseumConfig,
    FrozenAgent,
    ScriptedAgent,
    TrainableAgent,
    apply_overrides,
    load_config,
)
from colosseum.core.errors import ConfigError

RANDOM_BOT = {"kind": "scripted", "class": "colosseum.players.RandomBot"}


def _data(**sections) -> dict:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}}
    data.update(sections)
    return data


def _write(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


MIXED = {
    "main": None,
    "beta": {"algorithm": {"learning_rate": 1.0e-4}},
    "greedy": {"kind": "scripted", "class": "my_game.bots.Greedy", "kwargs": {"aggr": 0.7}},
    "prev": {"kind": "frozen", "path": "runs/sub12/checkpoints/main/ckpt_v9000"},
    "bc_net": {"kind": "frozen", "path": "runs/bc/main.pt", "networks": {"kwargs": {"hidden": 32}},
               "roles": ["player"]},
}


def test_entries_default_to_trainable_and_keep_their_kind():
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    assert cfg.agent_ids() == ["main", "beta", "greedy", "prev", "bc_net"]
    assert cfg.get_trainable_agent_ids() == ["main", "beta"]
    assert cfg.fixed_agent_ids() == ["greedy", "prev", "bc_net"]
    assert [cfg.agent_kind(a) for a in cfg.agent_ids()] == ["trainable", "trainable", "scripted", "frozen", "frozen"]
    assert cfg.agent_entry("main") == TrainableAgent() and AgentOverride is TrainableAgent
    greedy = cfg.agent_entry("greedy")
    assert isinstance(greedy, ScriptedAgent)
    assert greedy.class_path == "my_game.bots.Greedy" and greedy.kwargs == {"aggr": 0.7} and greedy.roles is None
    bc_net = cfg.agent_entry("bc_net")
    assert isinstance(bc_net, FrozenAgent) and bc_net.networks == {"kwargs": {"hidden": 32}}
    assert cfg.agent_roles("greedy") is None and cfg.agent_roles("bc_net") == ["player"]
    assert cfg.get_agent_config("beta").algorithm.learning_rate == 1.0e-4


def test_the_resolved_dump_round_trips_with_aliases(tmp_path):
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    path = _write(tmp_path, cfg.model_dump(mode="json", by_alias=True), "resolved.yaml")
    dumped = yaml.safe_load(path.read_text())
    assert dumped["agents"]["greedy"]["class"] == "my_game.bots.Greedy"
    assert dumped["agents"]["main"]["kind"] == "trainable"
    assert load_config(path) == cfg
    assert ColosseumConfig.model_validate(cfg.model_dump()) == cfg   # field names validate too


@pytest.mark.parametrize("agents", [{}, {"rnd": RANDOM_BOT}, {"old": {"kind": "frozen", "path": "x.pt"}}])
def test_an_implicit_trainable_agent_0_exists_without_trainable_agents(agents):
    cfg = ColosseumConfig.model_validate(_data(agents=agents))
    assert cfg.get_trainable_agent_ids() == ["agent_0"]
    assert cfg.agent_ids() == ["agent_0", *agents]
    assert cfg.agent_kind("agent_0") == "trainable" and cfg.agent_entry("agent_0") == TrainableAgent()
    assert cfg.get_agent_config("agent_0").agents == {}
    assert cfg.fixed_agent_ids() == list(agents)


def test_agent_0_is_not_implicit_when_a_trainable_agent_exists():
    cfg = ColosseumConfig.model_validate(_data(agents={"main": {}, "rnd": RANDOM_BOT}))
    with pytest.raises(ConfigError, match="Unknown agent 'agent_0'"):
        cfg.agent_entry("agent_0")


def test_a_fixed_agent_named_agent_0_without_trainable_agents_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="implicit trainable agent"):
        load_config(_write(tmp_path, _data(agents={"agent_0": RANDOM_BOT})))


@pytest.mark.parametrize("entry, words", [
    ({"kind": "scripted", "class": "a.B", "path": "x.pt"}, ["path", "scripted", "kwargs"]),
    ({"kind": "frozen", "path": "x.pt", "class": "a.B"}, ["class", "frozen", "networks"]),
    ({"class": "a.B"}, ["class", "trainable", "kind: scripted"]),
    ({"path": "x.pt"}, ["path", "trainable", "kind: frozen"]),
    ({"algoritm": {}}, ["algoritm", "trainable"]),
    ({"kind": "bot"}, ["bot"]),
    ({"kind": "scripted"}, ["class"]),
    ({"kind": "scripted", "class": "NoDot"}, ["dotted"]),
    ({"kind": "frozen"}, ["path"]),
    ({"kind": "scripted", "class": "a.B", "roles": []}, ["empty"]),
    ({"kind": "frozen", "path": "x.pt", "roles": ["p", "p"]}, ["more than once"]),
    ({"matchmaking": {"anchors": []}}, ["not supported yet"]),
    ({"init": {"from": "bc.pt"}}, ["not supported yet"]),
    ({"kickstart": {"teacher": "x"}}, ["not supported yet"]),
])
def test_bad_agent_entries_are_config_errors_with_a_hint(tmp_path, entry, words):
    with pytest.raises(ConfigError) as info:
        load_config(_write(tmp_path, _data(agents={"main": {}, "x": entry})))
    for word in words:
        assert word in str(info.value), str(info.value)


def test_get_agent_config_is_for_trainable_agents_only():
    cfg = ColosseumConfig.model_validate(_data(agents=MIXED))
    with pytest.raises(ConfigError, match="'greedy' is a scripted agent"):
        cfg.get_agent_config("greedy")
    with pytest.raises(ConfigError, match="'prev' is a frozen agent"):
        cfg.get_agent_config("prev")
    with pytest.raises(ConfigError, match="Unknown agent"):
        cfg.agent_kind("nobody")


def test_set_reaches_the_fields_of_every_kind(tmp_path):
    path = _write(tmp_path, _data(agents={"main": {}, "greedy": {"kind": "scripted", "class": "a.B"}}))
    cfg = load_config(path, {
        "agents.greedy.kwargs.aggr": 0.9, "agents.greedy.class": "c.D", "agents.greedy.roles": ["player"],
        "agents.prev.kind": "frozen", "agents.prev.path": "old.pt", "agents.prev.networks.kwargs.hidden": 32,
        "agents.main.algorithm.learning_rate": 1.0e-4,
    })
    greedy = cfg.agent_entry("greedy")
    assert greedy.kwargs == {"aggr": 0.9} and greedy.class_path == "c.D" and greedy.roles == ["player"]
    assert cfg.agent_kind("prev") == "frozen" and cfg.agent_entry("prev").networks == {"kwargs": {"hidden": 32}}
    assert cfg.get_agent_config("main").algorithm.learning_rate == 1.0e-4
    raw = yaml.safe_load(path.read_text())
    for key in ("agents.greedy.clas", "agents.greedy.kind.x", "agents.main.trainin.x"):
        with pytest.raises(ConfigError, match="agents"):
            apply_overrides(raw, {key: 1})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_agent_kinds.py -q`
Expected: collection error `ImportError: cannot import name 'FrozenAgent' from 'colosseum.core.config'`.

- [ ] **Step 3: Write the implementation**

In `src/colosseum/core/config.py`:

1. Imports: `from typing import Annotated, Any, Literal` and add `ValidationInfo` to the pydantic import.

2. `StrictModel`:
```python
class StrictModel(BaseModel):
    """Base for every config model: unknown keys are errors, not silently ignored (R5-07). Fields with an
    alias (``class``, ``from``, ``lambda``) accept their field name too (a ``model_dump()`` without aliases
    validates again)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
```

3. Replace the whole "Per-agent config overrides" section (the `AgentOverride` class and `_AGENT_SECTIONS`) by:
```python
# ---------------------------------------------------------------------------
# Agents (SP3 spec block 1): trainable, scripted and frozen
# ---------------------------------------------------------------------------


AgentKind = Literal["trainable", "scripted", "frozen"]

# Without any trainable agent in ``agents`` this trainable agent (global settings) exists implicitly.
IMPLICIT_AGENT_ID = "agent_0"


def _check_roles_list(roles: list[str] | None) -> list[str] | None:
    if roles is not None:
        if not roles:
            raise ValueError("roles must not be empty (omit it to play every role)")
        duplicates = sorted({r for r in roles if roles.count(r) > 1})
        if duplicates:
            raise ValueError(f"roles lists {duplicates} more than once")
    return roles


class _AgentEntry(StrictModel):
    """Base of the three agent kinds: an unknown field is named together with the kind's fields."""

    @model_validator(mode="before")
    @classmethod
    def _name_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fields = cls.model_fields
        allowed = set(fields) | {f.alias for f in fields.values() if f.alias}
        unknown = sorted(str(key) for key in data if key not in allowed)
        if unknown:
            kind = data.get("kind", "trainable")
            names = sorted(f.alias or name for name, f in fields.items())
            hint = ""
            if kind == "trainable" and {"class", "kwargs"} & set(unknown):
                hint = "; a rule-based bot needs kind: scripted"
            elif kind == "trainable" and "path" in unknown:
                hint = "; fixed weights from a file need kind: frozen"
            raise ValueError(f"unknown field(s) {unknown} for a {kind} agent (its fields: {names}){hint}")
        return data


class TrainableAgent(_AgentEntry):
    """A trainable agent (``kind: trainable``, the default): partial per-agent overrides.

    The section overrides are deep-merged onto the global sections before validation (R5-08), so an
    override that sets only ``learning_rate`` keeps every other global algorithm value.

    Caveat for ``networks``: the merge is key by key, so an override that switches to
    ``model_class`` still inherits the global ``encoder_class`` / ``policy_class`` /
    ``value_class`` (and ``core``) unless it sets them to ``null``, and an override that
    swaps only ``encoder_class`` still inherits the global ``networks.kwargs`` (set
    ``kwargs`` explicitly if the new classes take different arguments).
    """

    kind: Literal["trainable"] = "trainable"
    roles: list[str] | None = Field(
        default=None,
        description="Roles this agent plays (all must have the same spaces). Omitted = every role of the game, "
                    "which then must all have the same spaces.",
    )
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    matchmaking: dict[str, Any] | None = Field(default=None, description="Per-agent matchmaking override.")
    init: dict[str, Any] | None = Field(default=None, description="Per-agent warm start (init) override.")
    kickstart: dict[str, Any] | None = Field(default=None, description="Per-agent kickstart override.")

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)

    @field_validator("matchmaking", "init", "kickstart")
    @classmethod
    def _not_supported_yet(cls, value: dict[str, Any] | None, info: ValidationInfo) -> dict[str, Any] | None:
        if value is not None:
            raise ValueError(f"per-agent '{info.field_name}' sections are not supported yet; remove "
                             f"agents.<id>.{info.field_name}")
        return value


# SP2 name of the per-agent override model.
AgentOverride = TrainableAgent


class ScriptedAgent(_AgentEntry):
    """A rule-based bot (``kind: scripted``): a ``colosseum.players.ScriptedBot`` subclass built with ``kwargs``."""

    kind: Literal["scripted"]
    class_path: str = Field(..., alias="class",
                            description="Dotted path to a colosseum.players.ScriptedBot subclass.")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Constructor kwargs of the bot.")
    roles: list[str] | None = Field(
        default=None, description="Roles the bot plays (their spaces may differ). Omitted = every role of the game.",
    )

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, value: str) -> str:
        if "." not in value:
            raise ValueError(f"class must be a dotted path 'package.module.Class', got {value!r}")
        return value

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)


class FrozenAgent(_AgentEntry):
    """Fixed weights from a file (``kind: frozen``): a checkpoint dir (roles and networks from its
    ``meta.json``) or a ``.pt`` state_dict (``networks``: partial override onto the global networks;
    ``roles``: default every role of the game, which then must share one set of spaces)."""

    kind: Literal["frozen"]
    path: str = Field(..., min_length=1, description="Checkpoint dir or .pt state_dict.")
    networks: dict[str, Any] | None = Field(default=None, description="Only with a .pt path.")
    roles: list[str] | None = Field(default=None, description="Only with a .pt path.")

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)


AgentEntry = Annotated[TrainableAgent | ScriptedAgent | FrozenAgent, Field(discriminator="kind")]

_AGENT_SECTIONS = ("networks", "algorithm", "learner")
```

4. In `ColosseumConfig`, replace the `agents` field, its two field validators and `_validate_agent_overrides`, `_require_known_agent`, `get_agent_config`, `agent_roles`, `get_trainable_agent_ids` by:
```python
    agents: dict[str, AgentEntry] = Field(
        default_factory=dict,
        description="Agents by id. kind: trainable (default; partial overrides of networks / algorithm / "
                    "learner and roles), scripted (a ScriptedBot class with kwargs and roles) or frozen (fixed "
                    "weights from a checkpoint dir or a .pt). Without a trainable agent an implicit trainable "
                    "'agent_0' with the global settings exists.",
    )

    @field_validator("agents", mode="before")
    @classmethod
    def _default_kind(cls, value: Any) -> Any:
        """``null`` = a trainable agent with no overrides; an entry without ``kind`` is trainable."""
        if not isinstance(value, dict):
            return value
        out = {}
        for agent_id, entry in value.items():
            if entry is None:
                entry = {}
            if isinstance(entry, dict) and "kind" not in entry:
                entry = {**entry, "kind": "trainable"}
            out[agent_id] = entry
        return out

    @field_validator("agents")
    @classmethod
    def _check_agent_ids(cls, agents: dict[str, Any]) -> dict[str, Any]:
        for agent_id in agents:
            try:
                check_agent_id(agent_id)
            except ConfigError as e:
                raise ValueError(str(e)) from e
        return agents

    @model_validator(mode="after")
    def _validate_agent_overrides(self) -> ColosseumConfig:
        if self._has_implicit_agent() and IMPLICIT_AGENT_ID in self.agents:
            raise ValueError(
                f"agents.{IMPLICIT_AGENT_ID} is a {self.agents[IMPLICIT_AGENT_ID].kind} agent, but the config has no "
                f"trainable agent, so '{IMPLICIT_AGENT_ID}' is the implicit trainable agent; rename the "
                f"{self.agents[IMPLICIT_AGENT_ID].kind} agent or add a trainable agent"
            )
        for agent_id, entry in self.agents.items():
            if entry.kind != "trainable":
                continue
            try:
                self.get_agent_config(agent_id)
            except ValidationError as e:
                raise ValueError(f"agents.{agent_id}: invalid override:\n{e}") from None
        return self

    def _has_implicit_agent(self) -> bool:
        return not any(entry.kind == "trainable" for entry in self.agents.values())

    def agent_ids(self) -> list[str]:
        """Every agent in config order; the implicit trainable ``agent_0`` (if any) comes first."""
        ids = list(self.agents)
        return [IMPLICIT_AGENT_ID, *ids] if self._has_implicit_agent() else ids

    def _require_known_agent(self, agent_id: str) -> None:
        """Raise ConfigError unless ``agent_id`` is a configured agent or the implicit ``agent_0``."""
        if agent_id in self.agents or (agent_id == IMPLICIT_AGENT_ID and self._has_implicit_agent()):
            return
        if not self.agents:
            raise ConfigError(f"Unknown agent '{agent_id}': without an 'agents' section the only agent is "
                              f"'{IMPLICIT_AGENT_ID}'")
        raise ConfigError(f"Unknown agent '{agent_id}'. Known agents: {self.agent_ids()}")

    def agent_entry(self, agent_id: str) -> TrainableAgent | ScriptedAgent | FrozenAgent:
        """The agent's config entry (the implicit ``agent_0`` is ``TrainableAgent()``)."""
        self._require_known_agent(agent_id)
        entry = self.agents.get(agent_id)
        return entry if entry is not None else TrainableAgent()

    def agent_kind(self, agent_id: str) -> AgentKind:
        return self.agent_entry(agent_id).kind

    def get_agent_config(self, agent_id: str) -> ColosseumConfig:
        """Effective config of one TRAINABLE agent: global sections deep-merged with its overrides."""
        entry = self.agent_entry(agent_id)
        if entry.kind != "trainable":
            raise ConfigError(
                f"agent '{agent_id}' is a {entry.kind} agent: only trainable agents have a training config "
                f"(networks / algorithm / learner)"
            )
        data = self.model_dump(by_alias=True)
        for section in _AGENT_SECTIONS:
            part = getattr(entry, section)
            if part:
                data[section] = deep_merge(data[section], part)
        data["agents"] = {}
        return ColosseumConfig.model_validate(data)

    def agent_roles(self, agent_id: str) -> list[str] | None:
        """``agents.<id>.roles`` of any kind (None when omitted).

        Call it on the top-level config, not on a ``get_agent_config()`` result (which has ``agents == {}``).
        """
        roles = self.agent_entry(agent_id).roles
        return None if roles is None else list(roles)

    def get_trainable_agent_ids(self) -> list[str]:
        """Trainable agent ids in config order (owner rotation order); ``["agent_0"]`` without any."""
        ids = [agent_id for agent_id, entry in self.agents.items() if entry.kind == "trainable"]
        return ids or [IMPLICIT_AGENT_ID]

    def fixed_agent_ids(self) -> list[str]:
        """Scripted and frozen agent ids in config order."""
        return [agent_id for agent_id, entry in self.agents.items() if entry.kind != "trainable"]
```

5. Replace `_check_override_path` (and add `_model_members` above it):
```python
def _model_members(tp: Any) -> list[type[BaseModel]] | None:
    """The config models an annotation stands for: a model, an optional model or a union of models
    (``Annotated`` discriminated unions included, e.g. agent entries); None for anything else."""
    if typing.get_origin(tp) is typing.Annotated:
        tp = typing.get_args(tp)[0]
    tp = _unwrap_optional(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return [tp]
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if members and all(isinstance(m, type) and issubclass(m, BaseModel) for m in members):
            return members
    return None


def _check_override_path(parts: list[str]) -> None:
    """Walk the schema of ColosseumConfig along ``parts``; raise ConfigError on an unknown key.

    At a union of models (an agent entry) a key is valid if one of the members has it.
    """
    tp: Any = ColosseumConfig
    for i, part in enumerate(parts):
        where = ".".join(parts[: i + 1])
        members = _model_members(tp)
        if members is not None:
            found = None
            for model in members:
                name = next((n for n, f in model.model_fields.items() if part in (n, f.alias)), None)
                if name is not None:
                    found = model.model_fields[name].annotation
                    break
            if found is None:
                valid = sorted({f.alias or n for model in members for n, f in model.model_fields.items()})
                raise ConfigError(f"Unknown config key '{where}' (valid keys here: {valid})")
            tp = found
            continue
        tp = _unwrap_optional(tp)
        if typing.get_origin(tp) is dict:
            tp = typing.get_args(tp)[1]
        elif tp is Any:
            return  # free-form dict (env.kwargs, agent override bodies): checked at validation
        else:
            raise ConfigError(f"Cannot set '{'.'.join(parts)}': '{'.'.join(parts[:i])}' is not a section")
```

6. Module docstring: append a sentence "SP3: ``agents.<id>.kind`` (trainable / scripted / frozen) and the implicit trainable ``agent_0``."

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_agent_kinds.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings (the SP2 config tests `test_config_v2.py`, `test_sp2_config_overrides.py` and the T0.1 baseline stay green unchanged).

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/core/config.py tests/unit/test_sp3_agent_kinds.py
git commit -m "feat: agent kinds in the config (trainable, scripted, frozen) with an implicit agent_0"
git push origin sp3-league
```

---

### Task T1.2: `colosseum.players`: `ScriptedBot`, `RandomBot`, fixed-player loading, shared frozen loader

Spec blocks 1–2 and 3 (frozen agents). The player model outside the worker: the `ScriptedBot` base class (`reset` / `act`), the built-in `RandomBot` (a random legal action), the bot RNG per episode seed / seat / agent, the single legality check of a bot's action (`check_bot_action`, `PlayerError`), the picklable descriptions that travel to workers (`BotSpec`, `FrozenSpec`, `FixedPlayers`), the roles of every agent kind, and one frozen-weights loader shared by frozen agents, `eval` and (later) kickstart teachers and `init`. `eval.load_eval_model` is rewritten on top of it (same behaviour and messages).

**Files:**
- Create: `src/colosseum/players/__init__.py`, `src/colosseum/players/scripted.py`, `src/colosseum/players/registry.py`
- Modify: `src/colosseum/core/errors.py` (`PlayerError`)
- Modify: `src/colosseum/core/registry.py` (`build_network`; `build_model` delegates to it)
- Modify: `src/colosseum/coordinator/checkpoint_manager.py` (`read_checkpoint_meta`)
- Modify: `src/colosseum/eval.py` (`load_eval_model` delegates to `load_frozen` + `build_frozen_model`; `_agent_model_config`, `_pt_roles`, `_architecture_key` move into `players/registry.py`)
- Modify: `tests/game_helpers.py` (`RecordingBot`, `ConstantBot`, `scripted_agent`, `frozen_agent`, `scripted_player`)
- Test: `tests/unit/test_sp3_players.py`
- Existing tests that pin `load_eval_model` behaviour and must stay green unchanged: `tests/integration/test_sp2_eval_cli.py` (messages "not an SP2 checkpoint", "not roles of the game", "role signature", "networks", "different spaces", "do not match", "TypeError", the prefixes "Checkpoint <dir>: meta.json networks: agent 'b': model.step failed" and "<pt>: networks: agent 'a'", and the monkeypatch of `colosseum.eval._check_model`).

**Interfaces:**
- Consumes: `ColosseumConfig.{agent_ids, agent_entry, agent_kind, get_agent_config, get_trainable_agent_ids, fixed_agent_ids, networks}`, `NetworkConfig`, `deep_merge` (T1.1); `core.validation.random_legal_action`; `ActionSpec.{allocate_actions, first_illegal_action}`; `checkpoint_manager.{load_checkpoint_dir, read_weights_file, check_model_state}`; `core.roles.{resolve_agent_roles, agent_role_spec, role_signature}`; `core.registry.import_class`.
- Produces (contract `colosseum.core.errors`, `colosseum.players`, plus additions marked *):
  - `PlayerError(ColosseumError)`;
  - `ScriptedBot` (`game_spec`, `reset(*, role, seat, layout, rng)`, abstract `act(obs, mask, info)`), `RandomBot`;
  - `bot_rng(episode_seed, seat, agent_id) -> np.random.Generator`;
  - `check_bot_action(role, action, mask, where, *, action_spec=None) -> Tree` (*returns the action cast to the role's dtypes; *optional precomputed `action_spec`);
  - `BotSpec(class_path, kwargs)`, `FrozenSpec(agent_id, roles, networks, model_state, source, *networks_source="")`, `FixedPlayers(bots, frozen, roles)`;
  - `resolve_player_roles(config, spec)`, `load_fixed_players(config, spec)`, `load_frozen(config, agent_id, path, spec)`, `build_frozen_model(config: ColosseumConfig | None, frozen, spec)`, `make_bot(bot, spec)`;
  - *`colosseum.core.registry.build_network(networks: NetworkConfig, role: RoleSpec) -> PolicyModel`;
  - *`colosseum.coordinator.checkpoint_manager.read_checkpoint_meta(ckpt_dir) -> dict` (validated `meta.json`; ConfigError naming the dir; never writes);
  - test support: `RecordingBot`, `ConstantBot`, `scripted_agent(class_path=..., roles=None, **kwargs) -> dict`, `frozen_agent(path, roles=None, networks=None) -> dict`, `scripted_player(cls, spec, **kwargs) -> ScriptedPlayer`-like factory holder (see Step 3; it returns a `functools.partial` until T1.3 adds `ScriptedPlayer`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_players.py`:
```python
"""colosseum.players (SP3 T1.2, spec blocks 1-3): ScriptedBot, RandomBot, bot RNG, the legality gate,
roles of every agent kind, fixed-player loading and the shared frozen loader."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.errors import ColosseumError, ConfigError, PlayerError
from colosseum.core.registry import build_model, env_spec
from colosseum.core.roles import role_signature
from colosseum.envs.contract import EpisodeTracker
from colosseum.players import RandomBot, ScriptedBot
from colosseum.players.registry import (
    BotSpec,
    FixedPlayers,
    build_frozen_model,
    load_fixed_players,
    load_frozen,
    make_bot,
    resolve_player_roles,
)
from colosseum.players.scripted import bot_rng, check_bot_action
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    ConstantBot,
    TurnTakingGame,
    UnitsGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
)

WIDE = {"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none", "hidden": 32}}


def numpy_state(model) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def save_checkpoint(base_dir, agent_id, config, trainable="agent_0", networks=None, signature=None):
    """A signed SP2-style checkpoint dir with ``trainable``'s roles in the game of ``config``; the model is
    built from ``networks`` (a raw networks dict, default WIDE), which also goes into meta.json."""
    roles, role = agent_role_of(config, trainable)
    nets = WIDE if networks is None else networks
    model = build_model(config.model_copy(update={"networks": NetworkConfig.model_validate(nets)}), role)
    ckpt_id = CheckpointManager(base_dir).save(agent_id, 2, numpy_state(model), meta_extra={
        "networks": nets, "roles": roles, "role_signature": signature or role_signature(role)})
    return base_dir / agent_id / ckpt_id


def test_player_error_is_a_colosseum_error():
    assert issubclass(PlayerError, ColosseumError) and not issubclass(PlayerError, ConfigError)


def test_scripted_bot_is_abstract_and_reset_is_a_no_op():
    with pytest.raises(TypeError):
        ScriptedBot()

    class Echo(ScriptedBot):
        def act(self, obs, mask, info):
            return 0

    assert Echo().reset(role="player", seat=0, layout="2p", rng=np.random.default_rng(0)) is None


def test_bot_rng_depends_on_episode_seed_seat_and_agent():
    draw = bot_rng(123, 0, "rnd").random(3)
    assert np.array_equal(draw, bot_rng(123, 0, "rnd").random(3))
    for other in (bot_rng(123, 1, "rnd"), bot_rng(124, 0, "rnd"), bot_rng(123, 0, "greedy")):
        assert not np.array_equal(draw, other.random(3))
    assert not np.array_equal(bot_rng(None, 0, "rnd").random(3), bot_rng(None, 0, "rnd").random(3))


@pytest.mark.parametrize("game_cls", [TurnTakingGame, UnitsGame])
def test_random_bot_plays_legal_actions_that_pass_the_gate(game_cls):
    game = game_cls()
    layout = next(iter(game.spec.layouts))
    bots = {}
    for seat, seat_spec in enumerate(game.spec.layouts[layout]):
        bots[seat] = make_bot(BotSpec("colosseum.players.RandomBot", {}), game.spec)
        assert isinstance(bots[seat], RandomBot)
        bots[seat].reset(role=seat_spec.role, seat=seat, layout=layout, rng=bot_rng(7, seat, "rnd"))
    tracker = EpisodeTracker(game.spec)
    result = game.reset(0, layout)
    masks = tracker.on_reset(layout, result)
    chosen = []
    while not result.episode_over:
        actions = {}
        for seat in sorted(result.acting):
            role = game.spec.roles[game.spec.role_of(layout, seat)]
            actions[seat] = check_bot_action(role, bots[seat].act(result.obs[seat], masks[seat], None), masks[seat],
                                             "test")
            chosen.append(actions[seat])
        result = game.step(actions)
        masks = tracker.on_step(actions, result)
    assert chosen
    if game_cls is TurnTakingGame:
        assert all(int(a) in (0, 1) for a in chosen)   # action 2 is always masked


def test_check_bot_action_casts_and_rejects_bad_actions():
    turns = TurnTakingGame().spec.roles["player"]
    mask = np.array([True, False, True])
    out = check_bot_action(turns, 2, mask, "w")
    assert isinstance(out, np.ndarray) and out.dtype == np.int64 and out.shape == () and int(out) == 2
    with pytest.raises(PlayerError, match=r"^w: illegal action: action 1 at <root> is illegal"):
        check_bot_action(turns, 1, mask, "w")
    with pytest.raises(PlayerError, match="not in the role's action space"):
        check_bot_action(turns, 5, None, "w")
    with pytest.raises(PlayerError, match="structure"):
        check_bot_action(turns, {"move": 1}, None, "w")
    with pytest.raises(PlayerError, match="integer"):
        check_bot_action(turns, 1.5, None, "w")
    with pytest.raises(PlayerError, match="shape"):
        check_bot_action(turns, [1, 2], None, "w")
    with pytest.raises(PlayerError, match="None"):
        check_bot_action(turns, None, None, "w")

    units = UnitsGame()
    role = units.spec.roles["player"]
    masks = EpisodeTracker(units.spec).on_reset("solo", units.reset(0, "solo"))   # only unit 0 exists
    legal = {"units": {"target": [0, 0, 0, 0], "move": [3, 0, 0, 0]}, "base": 1}   # key order does not matter
    cast = check_bot_action(role, legal, masks[0], "w")
    assert list(cast) == ["base", "units"] and cast["units"]["move"].dtype == np.int64
    illegal = {"base": 1, "units": {"move": [3, 0, 0, 0], "target": [3, 0, 0, 0]}}
    with pytest.raises(PlayerError, match="unit 0 component 'target'"):
        check_bot_action(role, illegal, masks[0], "w")


def test_make_bot_imports_constructs_and_sets_the_game_spec():
    spec = TurnTakingGame().spec
    bot = make_bot(BotSpec("game_helpers.ConstantBot", {"value": 1}), spec)
    assert isinstance(bot, ConstantBot) and bot.game_spec is spec and bot.value == 1
    with pytest.raises(ConfigError, match="cannot be imported"):
        make_bot(BotSpec("no_such_module.Bot", {}), spec)
    with pytest.raises(ConfigError, match="must subclass"):
        make_bot(BotSpec("game_helpers.TurnTakingGame", {}), spec)
    with pytest.raises(ConfigError, match="kwargs"):
        make_bot(BotSpec("game_helpers.ConstantBot", {"nope": 1}), spec)
    assert pickle.loads(pickle.dumps(BotSpec("a.B", {"x": 1}))) == BotSpec("a.B", {"x": 1})


def test_resolve_player_roles_covers_every_kind_in_config_order(tmp_path):
    base = make_test_config("asymmetric")
    old_hunter = save_checkpoint(tmp_path / "ck", "hunter", base, trainable="hunter",
                                 networks=base.networks.model_dump(mode="json", by_alias=True))
    _roles, prey_role = agent_role_of(base, "prey")
    prey_pt = tmp_path / "prey.pt"
    torch.save(build_model(base.get_agent_config("prey"), prey_role).state_dict(), prey_pt)
    cfg = make_test_config("asymmetric", agents={
        "rnd": scripted_agent(), "hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]},
        "prey_bot": scripted_agent(roles=["prey"]), "old_hunter": frozen_agent(str(old_hunter)),
        "old_prey": frozen_agent(str(prey_pt), roles=["prey"]),
    })
    spec = env_spec(cfg)
    assert resolve_player_roles(cfg, spec) == {
        "rnd": ["hunter", "prey"], "hunter": ["hunter"], "prey": ["prey"], "prey_bot": ["prey"],
        "old_hunter": ["hunter"], "old_prey": ["prey"],
    }
    for agents, message in (
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": frozen_agent(str(prey_pt))}, "different spaces"),
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": scripted_agent(roles=["wolf"])}, "unknown roles"),
        ({"h": {"roles": ["hunter"]}, "p": {"roles": ["prey"]}, "z": frozen_agent(str(tmp_path / "nope"))},
         "expected a checkpoint dir"),
    ):
        with pytest.raises(ConfigError, match=message):
            resolve_player_roles(make_test_config("asymmetric", agents=agents), spec)


def test_load_fixed_players_loads_bots_and_frozen_weights_of_their_own_architecture(tmp_path):
    base = make_test_config("turns")
    wide_dir = save_checkpoint(tmp_path / "ck", "wide_agent", base)
    _roles, role = agent_role_of(base, "agent_0")
    mid_pt = tmp_path / "mid.pt"
    mid_cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": 24}})
    torch.save(build_model(mid_cfg.get_agent_config("agent_0"), role).state_dict(), mid_pt)
    cfg = make_test_config("turns", agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(str(wide_dir)),
        "mid": frozen_agent(str(mid_pt), networks={"kwargs": {"hidden": 24}}),
    })
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    assert isinstance(fixed, FixedPlayers)
    assert fixed.bots == {"rnd": BotSpec("colosseum.players.RandomBot", {})}
    assert list(fixed.frozen) == ["wide", "mid"]
    assert fixed.roles == {"rnd": ("player",), "wide": ("player",), "mid": ("player",)}
    wide = fixed.frozen["wide"]
    assert wide.agent_id == "wide" and wide.roles == ("player",) and wide.source == str(wide_dir)
    assert wide.networks["kwargs"]["hidden"] == 32
    assert wide.networks_source == f"Checkpoint {wide_dir}: meta.json networks"
    assert all(isinstance(v, np.ndarray) for v in wide.model_state.values())
    model = build_frozen_model(None, wide, spec)          # the architecture comes from the spec alone
    assert not model.training
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), wide.model_state[key])
    mid = build_frozen_model(cfg, fixed.frozen["mid"], spec)
    assert fixed.frozen["mid"].networks_source == f"{mid_pt}: networks"
    assert sum(p.numel() for p in mid.parameters()) < sum(p.numel() for p in model.parameters())
    again = pickle.loads(pickle.dumps(fixed))             # crosses process boundaries as Process kwargs
    assert again.bots == fixed.bots and list(again.frozen) == ["wide", "mid"]


def test_frozen_agent_errors_name_the_agent_or_the_path(tmp_path):
    base = make_test_config("turns")
    good = save_checkpoint(tmp_path / "ck", "wide_agent", base)
    other = save_checkpoint(tmp_path / "ck2", "other_agent", base, signature="obs=other-game")
    spec = env_spec(base)
    with pytest.raises(ConfigError, match="meta.json gives"):
        load_frozen(make_test_config("turns", agents={"agent_0": {}, "w": frozen_agent(str(good), roles=["player"])}),
                    "w", str(good), spec)
    with pytest.raises(ConfigError, match="role signature"):
        load_frozen(base, "x", str(other), spec)
    with pytest.raises(ConfigError, match="scripted agent"):
        load_frozen(make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()}), "rnd", str(good), spec)
    narrow_pt = tmp_path / "narrow.pt"
    torch.save(torch.load(good / "model.pt", weights_only=True), narrow_pt)   # wide weights, default networks
    frozen = load_frozen(base, "x", str(narrow_pt), spec)
    with pytest.raises(ConfigError, match="do not match"):
        build_frozen_model(base, frozen, spec)
    meta = json.loads((good / "meta.json").read_text())
    (good / "meta.json").write_text(json.dumps({**meta, "networks": {"bogus_key": 1}}))
    with pytest.raises(ConfigError, match="invalid meta.json networks"):
        load_frozen(base, "x", str(good), spec)


def test_the_sp2_checkpoint_fixture_loads_as_a_frozen_agent():
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT)})
    spec = env_spec(cfg)
    assert cfg.get_trainable_agent_ids() == ["agent_0"] and cfg.fixed_agent_ids() == ["old"]
    fixed = load_fixed_players(cfg, spec)
    assert fixed.roles == {"old": ("player",)}
    model = build_frozen_model(cfg, fixed.frozen["old"], spec)
    obs = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    out = model.step(obs, model.initial_state(1), torch.ones((1, 9), dtype=torch.bool))
    assert tuple(out.dist.sample().shape) == (1,)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_players.py -q`
Expected: collection error `ImportError: cannot import name 'PlayerError' from 'colosseum.core.errors'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/core/errors.py`, append:
```python
class PlayerError(ColosseumError):
    """A scripted player broke the rules (illegal action, exception, wrong structure).

    The message carries the SP2 context "worker W, env E, seat P, episode step K, layout L" plus
    ``agent '<id>'``."""
```

`src/colosseum/core/registry.py`: add `NetworkConfig` to the `TYPE_CHECKING` import from `colosseum.core.config`; rename the body of `build_model` into `build_network(net, role)` and let `build_model` delegate:
```python
def build_model(agent_config: ColosseumConfig, role: RoleSpec) -> PolicyModel:
    """Build the agent's ``PolicyModel`` for ``role``'s spaces from ``agent_config.networks`` (``build_network``)."""
    return build_network(agent_config.networks, role)


def build_network(net: NetworkConfig, role: RoleSpec) -> PolicyModel:
    """Build a ``PolicyModel`` for ``role``'s spaces from a ``networks`` section.

    - ``model_class``: ``cls(**inject, **networks.kwargs)``; it must be a ``PolicyModel``.
    - otherwise a ``ComposedModel``: ``encoder(**inject, **kwargs)``;
      ``core(input_dim=encoder.latent_dim, **core.kwargs)`` (``NoCore`` when ``core`` is null);
      ``critic_encoder(**inject, **kwargs)`` if set (its role must declare a global state);
      ``policy(in_dim=core.output_dim, **inject, **kwargs)``;
      ``value(in_dim=core.output_dim + critic.output_dim, **kwargs)``. ``in_dim`` is passed only to
      heads that name it.
    """
    # body of the former build_model, starting at "from colosseum.networks.base import BaseCriticEncoder",
    # with "net = agent_config.networks" removed (``net`` is the parameter)
```

`src/colosseum/coordinator/checkpoint_manager.py`, add after `load_checkpoint_dir`:
```python
def read_checkpoint_meta(ckpt_dir: str | Path) -> dict:
    """The validated ``meta.json`` of one checkpoint dir, read without touching the dir.

    A missing dir or an invalid ``meta.json`` raises ConfigError naming the dir.
    """
    path = Path(ckpt_dir)
    try:
        if not path.is_dir():
            raise ValueError("not a directory")
        return _parse_meta(path)
    except ValueError as e:
        raise ConfigError(f"Malformed checkpoint {path}: {e}") from e
```

Create `src/colosseum/players/__init__.py`:
```python
"""Players other than the latest weights of trainable agents (SP3 spec blocks 1-3).

``ScriptedBot`` is the base class of rule-based bots; ``RandomBot`` plays a uniformly random legal
action. ``colosseum.players.registry`` turns the config's scripted and frozen agents into picklable
specs (``BotSpec``, ``FrozenSpec``) and builds them.
"""

from colosseum.players.scripted import RandomBot, ScriptedBot

__all__ = ["RandomBot", "ScriptedBot"]
```

Create `src/colosseum/players/scripted.py`:
```python
"""Scripted bots (SP3 spec block 2).

A bot is a ``ScriptedBot`` subclass built from ``agents.<id>.class`` and ``kwargs``. The framework sets
``game_spec`` right after construction, calls ``reset`` at the start of every episode in which the bot
sits at a seat (``rng`` from :func:`bot_rng`), and ``act`` on every turn of that seat with numpy trees:
the observation (cast to the role's dtypes), the normalized action mask (or None) and
``StepResult.infos.get(seat)`` of the latest step (or None). A bot instance belongs to one
(agent, env, seat) and lives across episodes; per-episode memory goes into ``self``.

Every action passes :func:`check_bot_action`, the same legality gate as recorded actions
(``ActionSpec.first_illegal_action``, which applies ``units_component_valid`` to ``Units``).
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from colosseum.core.errors import PlayerError
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import Tree, tree_map, tree_to_torch
from colosseum.core.validation import random_legal_action
from colosseum.envs.game import GameSpec, RoleSpec

__all__ = ["RandomBot", "ScriptedBot", "bot_rng", "check_bot_action"]


class ScriptedBot(ABC):
    """Base class of rule-based bots (module docstring)."""

    game_spec: GameSpec  # set by the framework right after construction, before the first reset

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        """Start of an episode in which this bot sits at ``seat`` (role ``role``) of ``layout``."""
        return None

    @abstractmethod
    def act(self, obs: Tree, mask: Tree | None, info: Any) -> Tree:
        """An action tree in the role's action space for the current observation."""


class RandomBot(ScriptedBot):
    """A uniformly random legal action (``core.validation.random_legal_action``) from the episode RNG."""

    def __init__(self) -> None:
        self._role: RoleSpec | None = None
        self._rng = np.random.default_rng()

    def reset(self, *, role: str, seat: int, layout: str, rng: np.random.Generator) -> None:
        self._role = self.game_spec.roles[role]
        self._rng = rng

    def act(self, obs: Tree, mask: Tree | None, info: Any) -> Tree:
        if self._role is None:
            raise RuntimeError("RandomBot.act before reset")
        return random_legal_action(self._role, mask, self._rng)


def bot_rng(episode_seed: int | None, seat: int, agent_id: str) -> np.random.Generator:
    """The RNG a bot gets at ``reset``: ``SeedSequence([episode_seed, seat, crc32(agent_id)])``, so bots are
    reproducible under ``training.seed``; fresh entropy when ``episode_seed`` is None."""
    if episode_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence(
        [int(episode_seed) % 2**64, int(seat), zlib.crc32(agent_id.encode("utf-8"))]))


def check_bot_action(role: RoleSpec, action: Tree, mask: Tree | None, where: str, *,
                     action_spec: ActionSpec | None = None) -> Tree:
    """The bot's ``action`` as numpy leaves with the role's dtypes, or ``PlayerError(f"{where}: ...")``.

    Checks, in order: the tree structure of the role's action space (dict key order does not matter),
    each component's kind (integer components take integers) and shape, ``action_space.contains``
    (ranges and bounds), and under a normalized ``mask`` ``ActionSpec.first_illegal_action``.
    """
    spec = action_spec if action_spec is not None else ActionSpec.from_space(role.action_space)

    def cast(zero: np.ndarray, leaf: Any) -> np.ndarray:
        if leaf is None:
            raise PlayerError(f"{where}: the action has None where the role's action space expects a value")
        arr = np.asarray(leaf)
        if np.issubdtype(zero.dtype, np.integer) and arr.dtype.kind not in "iub":
            raise PlayerError(f"{where}: an integer action component got {arr.dtype} values ({arr!r})")
        if np.issubdtype(zero.dtype, np.floating) and arr.dtype.kind not in "iuf":
            raise PlayerError(f"{where}: a float action component got {arr.dtype} values ({arr!r})")
        if arr.shape != zero.shape:
            raise PlayerError(f"{where}: an action component has shape {arr.shape}, the role's action space "
                              f"expects {zero.shape}")
        return arr.astype(zero.dtype, copy=True)

    try:
        out = tree_map(cast, spec.allocate_actions(()), action)
    except ValueError as e:  # tree_map: dict keys or leaf/dict structure differ
        raise PlayerError(f"{where}: the action does not have the structure of the role's action space "
                          f"{role.action_space} ({e})") from e
    if not role.action_space.contains(out):
        raise PlayerError(f"{where}: action {out!r} is not in the role's action space {role.action_space} "
                          f"(out of range or bounds)")
    if mask is not None:
        found = spec.first_illegal_action(tree_to_torch(tree_map(lambda leaf: leaf[None], out)),
                                          tree_to_torch(tree_map(lambda m: np.asarray(m)[None], mask)))
        if found is not None:
            raise PlayerError(f"{where}: illegal action: {found[1]}")
    return out
```

Create `src/colosseum/players/registry.py`:
```python
"""Scripted and frozen agents of a config as picklable specs, their roles, and their construction
(SP3 spec blocks 1 and 3).

Nothing here holds torch tensors or bot instances: ``BotSpec`` (class path + kwargs) and ``FrozenSpec``
(numpy weights + the networks section) cross process boundaries; every process builds its own bots
(``make_bot``) and frozen models (``build_frozen_model``). ``load_frozen`` is the one loader of fixed
weights (frozen agents, ``eval -a``, and later kickstart teachers and ``init``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from colosseum.coordinator.checkpoint_manager import (
    check_model_state,
    load_checkpoint_dir,
    read_checkpoint_meta,
    read_weights_file,
)
from colosseum.core.config import ColosseumConfig, FrozenAgent, NetworkConfig, deep_merge
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_network, import_class
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.core.types import state_dict_from_numpy
from colosseum.envs.game import GameSpec
from colosseum.networks.model import PolicyModel
from colosseum.players.scripted import ScriptedBot

__all__ = ["BotSpec", "FixedPlayers", "FrozenSpec", "build_frozen_model", "load_fixed_players", "load_frozen",
           "make_bot", "resolve_player_roles"]


@dataclass(frozen=True)
class BotSpec:
    """A scripted bot as data: its class path and constructor kwargs (picklable)."""

    class_path: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenSpec:
    """Fixed weights with everything needed to build their model in any process.

    ``networks`` is a ``NetworkConfig`` dump (by alias); ``source`` the path the weights came from;
    ``networks_source`` says where ``networks`` came from (prefix of architecture errors).
    """

    agent_id: str
    roles: tuple[str, ...]
    networks: dict[str, Any]
    model_state: dict[str, np.ndarray]
    source: str
    networks_source: str = ""


@dataclass(frozen=True)
class FixedPlayers:
    """Every scripted (``bots``) and frozen (``frozen``) agent of a config, and their roles."""

    bots: dict[str, BotSpec]
    frozen: dict[str, FrozenSpec]
    roles: dict[str, tuple[str, ...]]


def make_bot(bot: BotSpec, spec: GameSpec) -> ScriptedBot:
    """Import and construct a bot, then set its ``game_spec``; ConfigError for a bad class or kwargs."""
    try:
        cls = import_class(bot.class_path)
    except Exception as e:  # noqa: BLE001 - any import failure is a config problem
        raise ConfigError(f"scripted bot class {bot.class_path!r} cannot be imported ({type(e).__name__}: {e}); "
                          f"use a dotted path 'package.module.Class' importable from the working directory") from e
    if not issubclass(cls, ScriptedBot):
        raise ConfigError(f"scripted bot class {bot.class_path!r} must subclass colosseum.players.ScriptedBot, "
                          f"got {cls.__name__}")
    try:
        instance = cls(**bot.kwargs)
    except Exception as e:  # noqa: BLE001 - a constructor failure from config kwargs
        raise ConfigError(f"scripted bot {bot.class_path!r}: constructing it with kwargs {bot.kwargs} failed "
                          f"({type(e).__name__}: {e})") from e
    instance.game_spec = spec
    return instance


def _check_role_names(spec: GameSpec, agent_id: str, roles: list[str]) -> None:
    unknown = [r for r in roles if r not in spec.roles]
    if unknown:
        raise ConfigError(f"agents.{agent_id}.roles: unknown roles {unknown}; the game has roles {list(spec.roles)}")


def _bad_path(agent_id: str, path: str) -> str:
    return (f"agents.{agent_id}.path={path!r}: expected a checkpoint dir (with model.pt and meta.json) or a "
            f".pt state_dict file")


def _pt_roles(config: ColosseumConfig, spec: GameSpec, name: str, path: Path, explicit: list[str] | None) -> list[str]:
    """Roles of a ``.pt`` player: a trainable agent's roles; else ``explicit`` (one set of spaces); else every
    role of the game, which then must share one set of spaces."""
    if name in config.get_trainable_agent_ids():
        return resolve_agent_roles(config, spec)[name]
    if explicit is not None:
        _check_role_names(spec, name, explicit)
        if len({role_signature(spec.roles[r]) for r in explicit}) > 1:
            raise ConfigError(f"agents.{name}.roles: roles {explicit} have different spaces; one network serves all "
                              f"roles of an agent, so list roles with the same spaces")
        return list(explicit)
    roles = list(spec.roles)
    if len({role_signature(spec.roles[r]) for r in roles}) > 1:
        raise ConfigError(
            f"{path}: agent {name!r}: a .pt file plays every role of the game unless agents.{name}.roles says "
            f"otherwise, but the roles {roles} have different spaces; set agents.{name}.roles (kind: frozen), "
            f"name the agent after a configured agent, or use a checkpoint dir"
        )
    return roles


def _checkpoint_roles(spec: GameSpec, path: Path, meta: dict) -> list[str]:
    roles, signature = meta.get("roles"), meta.get("role_signature")
    if roles is None or signature is None:
        raise ConfigError(f"Checkpoint {path}: meta.json has no roles/role_signature (not an SP2 checkpoint)")
    unknown = sorted(set(roles) - set(spec.roles))
    if unknown:
        raise ConfigError(f"Checkpoint {path}: roles {unknown} are not roles of the game {sorted(spec.roles)}")
    for role in roles:  # every role: one network serves them all
        expected = role_signature(spec.roles[role])
        if signature != expected:
            raise ConfigError(
                f"Checkpoint {path}: role signature {signature!r} does not match the game's spaces for role "
                f"{role!r} ({expected!r}); it was trained on a different game or game version"
            )
    return list(roles)


def resolve_player_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]:
    """``{agent_id: roles}`` for EVERY agent in config order (``ColosseumConfig.agent_ids``).

    Trainable: ``core.roles.resolve_agent_roles``. Scripted: ``roles`` or every role of the game (their
    spaces may differ). Frozen: a checkpoint dir's ``meta.json`` roles (checked against the game), or for a
    ``.pt`` the ``roles`` field / every role with one set of spaces. ConfigError with a hint otherwise.
    """
    trainable = resolve_agent_roles(config, spec)
    out: dict[str, list[str]] = {}
    for agent_id in config.agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind == "trainable":
            out[agent_id] = list(trainable[agent_id])
        elif entry.kind == "scripted":
            if entry.roles is not None:
                _check_role_names(spec, agent_id, entry.roles)
            out[agent_id] = list(entry.roles) if entry.roles is not None else list(spec.roles)
        else:
            path = Path(entry.path)
            if path.is_dir():
                out[agent_id] = _checkpoint_roles(spec, path, read_checkpoint_meta(path))
            elif path.is_file() and path.suffix == ".pt":
                out[agent_id] = _pt_roles(config, spec, agent_id, path, entry.roles)
            else:
                raise ConfigError(_bad_path(agent_id, entry.path))
    return out


def load_frozen(config: ColosseumConfig, agent_id: str, path: str | Path, spec: GameSpec) -> FrozenSpec:
    """Read fixed weights for the player ``agent_id`` from a checkpoint dir or a ``.pt`` (read-only).

    - Checkpoint dir (``load_checkpoint_dir``): roles and role signature from ``meta.json`` (required; checked
      against the game), networks from ``meta.json`` (else the agent's / global networks). A frozen agent
      with a checkpoint dir must not set ``networks`` or ``roles``.
    - ``.pt``: roles by ``_pt_roles``; networks: a trainable agent's effective networks, a frozen agent's
      ``networks`` override deep-merged onto the global networks, else the global networks.

    ConfigError naming the path; a path that is neither raises FileNotFoundError (callers check first).
    """
    p = Path(path)
    entry = config.agent_entry(agent_id) if agent_id in config.agent_ids() else None
    if entry is not None and entry.kind == "scripted":
        raise ConfigError(f"agent {agent_id!r} is a scripted agent: it has no weights to load from {p}")
    frozen_entry = entry if isinstance(entry, FrozenAgent) else None
    base = config.get_agent_config(agent_id).networks if entry is not None and entry.kind == "trainable" \
        else config.networks
    if p.is_dir():
        if frozen_entry is not None and (frozen_entry.networks is not None or frozen_entry.roles is not None):
            raise ConfigError(f"agents.{agent_id}: {p} is a checkpoint dir, whose meta.json gives the roles and "
                              f"networks; remove agents.{agent_id}.networks / roles (they are for .pt files)")
        loaded = load_checkpoint_dir(p)
        roles = _checkpoint_roles(spec, p, loaded["meta"])
        raw = loaded["meta"].get("networks")
        if raw is not None:
            try:
                networks = NetworkConfig.model_validate(raw)
            except ValidationError as e:
                raise ConfigError(f"Checkpoint {p}: invalid meta.json networks:\n{e}") from e
            networks_source = f"Checkpoint {p}: meta.json networks"
        else:
            networks, networks_source = base, f"Checkpoint {p}: networks of the config"
        model_state = loaded["model_state"]
    elif p.is_file() and p.suffix == ".pt":
        roles = _pt_roles(config, spec, agent_id, p, frozen_entry.roles if frozen_entry is not None else None)
        networks = base
        if frozen_entry is not None and frozen_entry.networks:
            try:
                networks = NetworkConfig.model_validate(
                    deep_merge(config.networks.model_dump(by_alias=True), frozen_entry.networks))
            except ValidationError as e:
                raise ConfigError(f"agents.{agent_id}.networks: invalid override:\n{e}") from e
        networks_source = f"{p}: networks"
        try:
            model_state = read_weights_file(p)
        except ValueError as e:
            raise ConfigError(f"{p}: {e}") from e
    else:
        raise FileNotFoundError(f"{p}: expected a checkpoint directory or a .pt file")
    return FrozenSpec(agent_id=agent_id, roles=tuple(roles), networks=networks.model_dump(mode="json", by_alias=True),
                      model_state=model_state, source=str(p), networks_source=networks_source)


def build_frozen_model(config: ColosseumConfig | None, frozen: FrozenSpec, spec: GameSpec) -> PolicyModel:
    """The frozen player's model, weights loaded, in eval mode. ``config`` is not needed (``frozen`` holds the
    networks) and may be None, e.g. in a worker process. ConfigError naming ``frozen.source`` when the model
    cannot be built or the weights do not fit it."""
    role = agent_role_spec(spec, list(frozen.roles))
    try:
        model = build_network(NetworkConfig.model_validate(frozen.networks), role)
    except ConfigError as e:
        raise ConfigError(f"{frozen.source}: {e}") from e
    except Exception as e:  # noqa: BLE001 - a networks section is user data: any constructor failure
        raise ConfigError(f"{frozen.source}: cannot build the agent's model ({type(e).__name__}: {e})") from e
    check_model_state(model, frozen.model_state, frozen.source)
    model.load_state_dict(state_dict_from_numpy(frozen.model_state))
    model.eval()
    return model


def load_fixed_players(config: ColosseumConfig, spec: GameSpec) -> FixedPlayers:
    """Every scripted and frozen agent of ``config`` (main process; before any child starts).

    Scripted bots are imported and constructed once (a bad class or kwargs fail here); frozen weights are
    read with ``load_frozen`` (roles and signatures checked). Models are built where they run.
    """
    roles = resolve_player_roles(config, spec)
    bots: dict[str, BotSpec] = {}
    frozen: dict[str, FrozenSpec] = {}
    for agent_id in config.fixed_agent_ids():
        entry = config.agent_entry(agent_id)
        if entry.kind == "scripted":
            bot = BotSpec(entry.class_path, dict(entry.kwargs))
            make_bot(bot, spec)
            bots[agent_id] = bot
        else:
            frozen[agent_id] = load_frozen(config, agent_id, entry.path, spec)
    return FixedPlayers(bots=bots, frozen=frozen, roles={a: tuple(roles[a]) for a in config.fixed_agent_ids()})
```

`src/colosseum/eval.py`: delete `_agent_model_config`, `_pt_roles` and `_architecture_key`, and replace the body of `load_eval_model` (signature and docstring unchanged, except "Problems raise ConfigError naming the path" stays):
```python
    from colosseum.core.registry import env_spec
    from colosseum.players.registry import build_frozen_model, load_frozen

    spec = spec if spec is not None else env_spec(config)
    frozen = load_frozen(config, name, path, spec)
    model = build_frozen_model(config, frozen, spec)
    role = agent_role_spec(spec, list(frozen.roles))
    key = json.dumps([frozen.networks, list(frozen.roles)], sort_keys=True)
    if validated is None or key not in validated:
        try:
            _check_model(model, role, None, f"agent {name!r}")
        except ConfigError as e:
            raise ConfigError(f"{frozen.networks_source}: {e}") from e
        if validated is not None:
            validated.add(key)
    return model, list(frozen.roles)
```
Then `.venv/bin/ruff check --fix src/colosseum/eval.py` drops the imports that became unused (`check_model_state`, `load_checkpoint_dir`, `read_weights_file`, `NetworkConfig`, `ValidationError`, `resolve_agent_roles`, `role_signature`, `state_dict_from_numpy`); `_check_model` stays imported in `colosseum.eval` (a test monkeypatches it there).

`tests/game_helpers.py`: add imports `import functools` and `from typing import ClassVar` (if missing), `from colosseum.core.validation import random_legal_action`, `from colosseum.players import ScriptedBot`, then append:
```python
# ---------------------------------------------------------------------------
# SP3 (T1.2): scripted bots and config entries for fixed players
# ---------------------------------------------------------------------------


class RecordingBot(ScriptedBot):
    """Remembers every ``reset`` / ``act`` argument and plays a random legal action from its episode RNG.

    ``RecordingBot.instances`` lists every bot constructed in this process, in order (tests clear it).
    Each reset entry holds the first draw of the episode RNG (``draw``), so tests can compare RNGs.
    """

    instances: ClassVar[list[RecordingBot]] = []

    def __init__(self) -> None:
        self.resets: list[dict] = []
        self.acts: list[dict] = []
        self._role: RoleSpec | None = None
        self._rng: np.random.Generator | None = None
        RecordingBot.instances.append(self)

    def reset(self, *, role, seat, layout, rng) -> None:
        self.resets.append({"role": role, "seat": seat, "layout": layout, "draw": float(rng.random())})
        self._role = self.game_spec.roles[role]
        self._rng = rng

    def act(self, obs, mask, info):
        self.acts.append({"obs": obs, "mask": mask, "info": info})
        return random_legal_action(self._role, mask, self._rng)


class ConstantBot(ScriptedBot):
    """Always the action ``value`` of a ``Discrete`` action space (DAgger and gate tests)."""

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def act(self, obs, mask, info):
        return np.int64(self.value)


def scripted_agent(class_path: str = "colosseum.players.RandomBot", roles: list[str] | None = None,
                   **kwargs) -> dict:
    """An ``agents.<id>`` entry of a scripted agent (raw config dict)."""
    entry: dict = {"kind": "scripted", "class": class_path, "kwargs": dict(kwargs)}
    if roles is not None:
        entry["roles"] = list(roles)
    return entry


def frozen_agent(path, roles: list[str] | None = None, networks: dict | None = None) -> dict:
    """An ``agents.<id>`` entry of a frozen agent (raw config dict)."""
    entry: dict = {"kind": "frozen", "path": str(path)}
    if roles is not None:
        entry["roles"] = list(roles)
    if networks is not None:
        entry["networks"] = dict(networks)
    return entry


def _bot_with_spec(cls, spec: GameSpec, kwargs: dict) -> ScriptedBot:
    bot = cls(**kwargs)
    bot.game_spec = spec
    return bot


def scripted_player(cls, spec: GameSpec, **kwargs):
    """A bot factory for a player pool (``ScriptedPlayer(factory)`` from T1.3): ``cls(**kwargs)`` with ``spec``."""
    return functools.partial(_bot_with_spec, cls, spec, kwargs)
```
(T1.3 wraps it: `scripted_player` then returns `ScriptedPlayer(functools.partial(...))`; see T1.3 Step 3.)

- [ ] **Step 4: Run the new tests and the eval tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_players.py tests/integration/test_sp2_eval_cli.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/players src/colosseum/core/errors.py src/colosseum/core/registry.py \
        src/colosseum/coordinator/checkpoint_manager.py src/colosseum/eval.py tests/game_helpers.py \
        tests/unit/test_sp3_players.py
git commit -m "feat: colosseum.players (ScriptedBot, RandomBot, legality gate, fixed-player specs, shared frozen loader)"
git push origin sp3-league
```

---

### Task T1.3: `MatchRunner` player pool (scripted seats, `infos`, `fixed`, `source`)

Spec blocks 2–3. The match core serves a player pool: `(agent_id, network_id)` → a neural player (`PolicyModel`; latest, snapshots and frozen agents under `"fixed"`) or a scripted one (`ScriptedPlayer`, a factory of bot instances). Neural players are batched per group as in SP2; scripted seats act one by one, see their observation (cast to the role's dtypes), the normalized mask and `infos[seat]` of the latest `StepResult`, and pass the legality gate. Bot instances exist per (agent, env, seat), are created lazily and reset after every env reset where they sit, with `bot_rng(episode seed, seat, agent)`. `on_act` also fires for scripted seats (log-prob 0, no unit log-probs, no state). `collect=True` is allowed only on `"latest"` seats. `SeatAssignment.source` (the opponent category of a team, set by the matchmaker from T3.2) travels into `SeatResult.source`.

**Files:**
- Modify: `src/colosseum/core/types.py` (`FIXED_NETWORK_ID`, `SOURCE_OWNER`, `OPPONENT_CATEGORIES`, `SeatAssignment.source`, `SeatResult.source`)
- Modify (rewrite): `src/colosseum/worker/match_runner.py`
- Modify: `tests/game_helpers.py` (`TickGame(..., infos=False)`; `scripted_player` returns a `ScriptedPlayer`)
- Test: `tests/unit/test_sp3_match_runner_players.py`
- Existing tests that must stay green unchanged: `tests/unit/test_match_runner.py` (the "no model for agent 'zzz'" and "env 0, seat 1: ... collect" messages), `tests/contract/test_rollout_loop_lineups.py`, `tests/contract/test_sp2_eval_engine.py`.

**Interfaces:**
- Consumes: `ScriptedBot`, `bot_rng`, `check_bot_action`, `PlayerError` (T1.2).
- Produces (contract T1.3):
  - `colosseum.core.types`: `FIXED_NETWORK_ID = "fixed"`, `SOURCE_OWNER = "owner"`, `OPPONENT_CATEGORIES = ("latest", "snapshots", "rivals", "anchors", "fallback")`; `SeatAssignment(agent_id, network_id="latest", collect=True, source="")`; `SeatResult(..., eliminated_step=None, source="")`;
  - `colosseum.worker.match_runner`: `ScriptedPlayer(factory)`, `PlayerPool` (`ModelPool = PlayerPool`), `ActRecord.info`, the optional observer hook `on_episode_start(env, layout, episode_seed)`, `MatchRunner.next_lineup(env)`; `MatchRunner(..., models: PlayerPool, ...)` keeps its keyword name `models`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_match_runner_players.py`:
```python
"""MatchRunner player pool (SP3 T1.3, spec blocks 2-3): scripted seats with infos, bot RNG and lifetime,
the legality gate with context, frozen players under 'fixed', collect rules, source, on_episode_start."""
from __future__ import annotations

import functools

import numpy as np
import pytest

from colosseum.core.errors import PlayerError
from colosseum.core.types import FIXED_NETWORK_ID, OPPONENT_CATEGORIES, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.envs.vector import VectorEnv
from colosseum.players import ScriptedBot
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import MatchRunner, ScriptedPlayer
from game_helpers import (
    DictModelPool,
    RecordingBot,
    RecordingObserver,
    Tick,
    TickGame,
    UnitsGame,
    make_test_model,
    scripted_player,
)

TURNS = [Tick(acting={0}), Tick(acting={1}), Tick(acting={0}), Tick(over=True, rewards={0: 1.0, 1: -1.0})]
MASK = np.array([True, False, True])


class IllegalBot(ScriptedBot):
    def act(self, obs, mask, info):
        return np.int64(1)                    # masked by MASK


class CrashingBot(ScriptedBot):
    def __init__(self, where: str = "act") -> None:
        self.where = where

    def reset(self, *, role, seat, layout, rng):
        if self.where == "reset":
            raise RuntimeError("reset bug")

    def act(self, obs, mask, info):
        raise RuntimeError("act bug")


class WrongShapeBot(ScriptedBot):
    def act(self, obs, mask, info):
        return {"x": 0}


class UnitsTargetBot(ScriptedBot):
    """Unit 0 moves 3 and targets unit 3, which does not exist at step 0 of UnitsGame."""

    def act(self, obs, mask, info):
        return {"base": 0, "units": {"move": np.array([3, 0, 0, 0]), "target": np.array([3, 0, 0, 0])}}


class EpisodeStartObserver(RecordingObserver):
    def on_episode_start(self, env, layout, episode_seed):
        self.events.append(("start", env, layout, episode_seed))


def _role(num_seats=2, **kwargs):
    return next(iter(TickGame([Tick(acting={0})], num_seats, **kwargs).spec.roles.values()))


def _runner(lineups, players, *, script=TURNS, num_seats=2, num_envs=1, seed=0, observer=None, **game_kwargs):
    games = []

    def env_fn():
        games.append(TickGame(script, num_seats, **game_kwargs))
        return games[-1]

    vec = VectorEnv(env_fn, num_envs)
    observer = observer if observer is not None else RecordingObserver()
    runner = MatchRunner(vec_env=vec, lineups=lineups, models=DictModelPool(players), observer=observer,
                         seed=seed, context="worker 0, ", match_id_prefix="w0_e")
    return runner, observer, games


def _spec(num_seats=2, **kwargs):
    return TickGame([Tick(acting={0})], num_seats, **kwargs).spec


def _bot_lineup(*seats):
    return Lineup("2p", [SeatAssignment(a, FIXED_NETWORK_ID, False) if a.startswith("bot") else SeatAssignment(a)
                         for a in seats])


@pytest.fixture(autouse=True)
def _clear_recording_bots():
    RecordingBot.instances.clear()
    yield
    RecordingBot.instances.clear()


def test_types_name_the_fixed_network_and_the_categories():
    assert FIXED_NETWORK_ID == "fixed" and SOURCE_OWNER == "owner"
    assert OPPONENT_CATEGORIES == ("latest", "snapshots", "rivals", "anchors", "fallback")
    assert SeatAssignment("a").source == ""


def test_a_scripted_seat_gets_obs_mask_and_infos_and_is_reset_every_episode():
    spec = _spec(mask_fn=lambda k, t, s: MASK, infos=True)
    recording = BotSpec("game_helpers.RecordingBot", {})
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): ScriptedPlayer(functools.partial(make_bot, recording, spec))}
    runner, obs, _games = _runner([_bot_lineup("a", "bot")], players, mask_fn=lambda k, t, s: MASK, infos=True)
    (bot,) = RecordingBot.instances                      # created at the first reset of its (agent, env, seat)
    assert [r["seat"] for r in bot.resets] == [1] and bot.resets[0]["role"] == "player"
    assert bot.resets[0]["layout"] == "2p"
    for _ in range(3):
        runner.step()
    assert len(RecordingBot.instances) == 1               # kept between episodes
    assert len(bot.resets) == 2                           # reset again after the episode end
    (act,) = bot.acts
    assert act["obs"].dtype == np.float32 and act["obs"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert act["mask"].tolist() == MASK.tolist()
    assert act["info"] == {"k": 0, "t": 1, "seat": 1}
    bot_records = [e[3] for e in obs.events if e[0] == "act" and e[2] == 1]
    assert len(bot_records) == 1
    record = bot_records[0]
    assert (record.agent_id, record.network_id) == ("bot", FIXED_NETWORK_ID)
    assert record.log_prob == 0.0 and record.unit_log_probs is None and record.pre_state is None
    assert record.info == {"k": 0, "t": 1, "seat": 1} and int(record.action) in (0, 2)
    result = next(e[2].result for e in obs.events if e[0] == "end")
    assert [s.network_id for s in result.seats] == ["latest", FIXED_NETWORK_ID]


def test_bot_rngs_follow_the_episode_seed_seat_and_agent():
    def draws(seed):
        RecordingBot.instances.clear()
        spec = _spec()
        factory = ScriptedPlayer(functools.partial(make_bot, BotSpec("game_helpers.RecordingBot", {}), spec))
        _runner([_bot_lineup("bot_a", "bot_a"), _bot_lineup("bot_a", "bot_b")],
                {("bot_a", FIXED_NETWORK_ID): factory, ("bot_b", FIXED_NETWORK_ID): factory}, num_envs=2, seed=seed)
        return [b.resets[0]["draw"] for b in RecordingBot.instances]

    first, again, other = draws(7), draws(7), draws(8)
    assert first == again and first != other
    assert len(set(first)) == 4                           # env, seat and agent all change the stream


def test_bot_instances_are_per_agent_env_and_seat_and_survive_lineup_changes():
    spec = _spec()
    factory = ScriptedPlayer(functools.partial(make_bot, BotSpec("game_helpers.RecordingBot", {}), spec))
    players = {("a", "latest"): make_test_model(_role()), ("bot", FIXED_NETWORK_ID): factory}
    runner, _obs, _games = _runner([_bot_lineup("a", "bot")] * 2, players, num_envs=2)
    assert len(RecordingBot.instances) == 2               # one per env
    first_env0 = RecordingBot.instances[0]
    runner.set_next_lineup(0, _bot_lineup("bot", "a"))
    assert runner.next_lineup(0) == _bot_lineup("bot", "a") and runner.next_lineup(1) is None
    for _ in range(3):
        runner.step()                                     # episode end: env 0 moves the bot to seat 0
    assert len(RecordingBot.instances) == 3               # a new (bot, env 0, seat 0) instance
    runner.set_next_lineup(0, _bot_lineup("a", "bot"))
    for _ in range(3):
        runner.step()
    assert len(RecordingBot.instances) == 3 and len(first_env0.resets) == 2   # seat 1 of env 0 is back


@pytest.mark.parametrize("bot_cls, kwargs, message", [
    (IllegalBot, {}, r"^worker 0, env 0, seat 1, episode step 1, layout 2p: agent 'bot': illegal action: "
                     r"action 1 at <root> is illegal"),
    (CrashingBot, {}, r"^worker 0, env 0, seat 1, episode step 1, layout 2p: agent 'bot': act raised "
                      r"RuntimeError: act bug"),
    (WrongShapeBot, {}, r"agent 'bot': the action does not have the structure"),
])
def test_a_bot_that_breaks_the_rules_is_a_player_error_with_context(bot_cls, kwargs, message):
    spec = _spec(mask_fn=lambda k, t, s: MASK)
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(bot_cls, spec, **kwargs)}
    runner, _obs, _games = _runner([_bot_lineup("a", "bot")], players, mask_fn=lambda k, t, s: MASK)
    runner.step()                                          # seat 0 acts
    with pytest.raises(PlayerError, match=message) as info:
        runner.step()                                      # seat 1 (the bot) acts at episode step 1
    if bot_cls is CrashingBot:
        assert isinstance(info.value.__cause__, RuntimeError)


def test_a_bot_whose_reset_raises_fails_at_the_episode_start():
    spec = _spec()
    with pytest.raises(PlayerError, match=r"env 0, seat 1, episode step 0, layout 2p: agent 'bot': reset raised "
                                          r"RuntimeError: reset bug"):
        _runner([_bot_lineup("a", "bot")], {("a", "latest"): make_test_model(_role()),
                                            ("bot", FIXED_NETWORK_ID): scripted_player(CrashingBot, spec,
                                                                                       where="reset")})


def test_units_actions_of_bots_pass_the_same_gate():
    def run(bot_cls_or_spec):
        vec = VectorEnv(lambda: UnitsGame(), 1)
        player = (ScriptedPlayer(functools.partial(make_bot, bot_cls_or_spec, vec.spec))
                  if isinstance(bot_cls_or_spec, BotSpec) else scripted_player(bot_cls_or_spec, vec.spec))
        runner = MatchRunner(vec_env=vec, lineups=[Lineup("solo", [SeatAssignment("bot", FIXED_NETWORK_ID, False)])],
                             models=DictModelPool({("bot", FIXED_NETWORK_ID): player}), seed=0, context="worker 2, ")
        try:
            for _ in range(12):                            # two UnitsGame episodes
                runner.step()
        finally:
            runner.close()
        return runner.episodes_finished

    assert run(BotSpec("colosseum.players.RandomBot", {})) == 2
    with pytest.raises(PlayerError, match=r"worker 2, env 0, seat 0, episode step 0, layout solo: agent 'bot': "
                                          r"illegal action: unit 0 component 'target'"):
        run(UnitsTargetBot)


def test_only_latest_seats_may_collect_and_fixed_seats_need_their_player():
    spec = _spec()
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(RecordingBot, spec)}
    with pytest.raises(ValueError, match=r"env 0, seat 1: agent 'bot' plays network 'fixed' with collect=True"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", FIXED_NETWORK_ID, True)])], players)
    with pytest.raises(ValueError, match=r"no model for agent 'bot' \(network 'latest'\)"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", collect=False)])], players)
    with pytest.raises(ValueError, match=r"no model for agent 'ghost' \(network 'fixed'\)"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("ghost", FIXED_NETWORK_ID, False)])], players)


def test_a_frozen_player_of_another_architecture_is_batched_under_fixed():
    small, wide = make_test_model(_role()), make_test_model(_role(), hidden=32)
    players = {("a", "latest"): small, ("old", FIXED_NETWORK_ID): wide}
    lineups = [Lineup("2p", [SeatAssignment("a"), SeatAssignment("old", FIXED_NETWORK_ID, False)]),
               Lineup("2p", [SeatAssignment("old", FIXED_NETWORK_ID, False), SeatAssignment("a")])]
    runner, obs, _games = _runner(lineups, players, num_envs=2,
                                  script=[Tick(acting={0, 1}), Tick(over=True, rewards={0: 1.0, 1: -1.0})])
    runner.step()
    acts = [(e[1], e[2], e[3].agent_id, e[3].network_id) for e in obs.events if e[0] == "act"]
    assert acts == [(0, 0, "a", "latest"), (0, 1, "old", "fixed"), (1, 0, "old", "fixed"), (1, 1, "a", "latest")]
    assert all(e[3].pre_state is None for e in obs.events if e[0] == "act")   # stateless models
    results = [e[2].result for e in obs.events if e[0] == "end"]
    assert [[s.network_id for s in r.seats] for r in results] == [["latest", "fixed"], ["fixed", "latest"]]


def test_source_travels_from_the_lineup_into_the_result():
    spec = _spec()
    players = {("a", "latest"): make_test_model(_role()),
               ("bot", FIXED_NETWORK_ID): scripted_player(RecordingBot, spec)}
    lineup = Lineup("2p", [SeatAssignment("a", source=SOURCE_OWNER),
                           SeatAssignment("bot", FIXED_NETWORK_ID, False, source="anchors")])
    runner, obs, _games = _runner([lineup], players)
    assert [s.source for s in runner.lineup(0).seats] == [SOURCE_OWNER, "anchors"]
    for _ in range(3):
        runner.step()
    result = next(e[2].result for e in obs.events if e[0] == "end")
    assert [s.source for s in result.seats] == [SOURCE_OWNER, "anchors"]


def test_on_episode_start_is_called_after_every_reset_with_the_episode_seed():
    players = {("a", "latest"): make_test_model(_role())}
    observer = EpisodeStartObserver()
    runner, _obs, games = _runner([Lineup("2p", [SeatAssignment("a")] * 2)], players, observer=observer, seed=3)
    for _ in range(3):
        runner.step()
    starts = [e for e in observer.events if e[0] == "start"]
    reset_seeds = [entry[2] for entry in games[0].log if entry[3] is None]
    assert [(e[1], e[2]) for e in starts] == [(0, "2p"), (0, "2p")]
    assert [e[3] for e in starts] == reset_seeds and None not in reset_seeds
    assert observer.kinds().index("start") == 0           # before the first act
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_match_runner_players.py -q`
Expected: collection error `ImportError: cannot import name 'FIXED_NETWORK_ID' from 'colosseum.core.types'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/core/types.py`: below `LATEST_NETWORK_ID` add
```python
# Network id of the seats of scripted and frozen agents (one fixed player per agent; never collects).
FIXED_NETWORK_ID = "fixed"
# SeatAssignment.source of the data owner's team; the core of every opposing team carries its category.
SOURCE_OWNER = "owner"
# Opponent categories of the built-in matchmaker (SP3 spec block 5); "fallback" marks the runtime fallback.
OPPONENT_CATEGORIES = ("latest", "snapshots", "rivals", "anchors", "fallback")
```
and replace `SeatAssignment` and `SeatResult` by:
```python
@dataclass
class SeatAssignment:
    """Who plays one seat: an agent's latest weights, a snapshot (``ckpt_v<N>``) or a scripted / frozen agent
    (``FIXED_NETWORK_ID``), and whether the seat collects (only ``latest`` seats may).

    ``source``: "" (eval, tests), ``SOURCE_OWNER`` for the data owner's team, or one of
    ``OPPONENT_CATEGORIES`` for an opposing team (every seat of a team carries its team's value).
    """

    agent_id: str
    network_id: str = LATEST_NETWORK_ID
    collect: bool = True
    source: str = ""
```
```python
@dataclass
class SeatResult:
    """One seat of a finished match. ``reward`` is the undiscounted episode return; ``source`` is the
    seat's ``SeatAssignment.source``."""

    seat: int
    role: str
    team: int
    agent_id: str
    network_id: str
    reward: float
    eliminated_step: int | None = None
    source: str = ""
```

Replace `src/colosseum/worker/match_runner.py` with:
```python
"""MatchRunner: the match core shared by training (RolloutLoop), eval and record (SP2 spec block 5,
SP3 spec block 3).

It owns a vector env, one :class:`Lineup` per env, one ``EpisodeTracker`` per env, the model state of
every neural seat and the scripted bots. Each :meth:`MatchRunner.step`:

1. acts for every acting seat of every env. Neural seats are inferred in batches grouped by
   ``(agent_id, network_id)`` (policy only, ``networks.model.act``). Scripted seats (the pool gives a
   :class:`ScriptedPlayer`) call their bot's ``act(obs, mask, info)`` one by one; the action passes
   ``check_bot_action`` (the single legality gate). Then ``observer.on_act`` per acting seat in
   ``(env, seat)`` order (a scripted seat's record has ``log_prob`` 0, no unit log-probs, no state);
2. steps every env once (``vec_env.step``; an env without acting seats gets ``{}``);
3. per env in index order: ``EpisodeTracker.on_step`` (contract checks, normalized masks)
   -> ``on_rewards`` (every reward of the step, including the elimination step) ->
   ``on_terminated`` (if any seat was eliminated) -> if the episode is over:
   ``on_episode_end`` (with the :class:`MatchResult`) -> apply the next lineup
   (``on_lineup_applied``) -> reset the env's model states;
4. resets all finished envs in one ``vec_env.reset`` with per-episode seeds; after each reset the
   bots of the env's scripted seats are reset (``bot_rng(episode seed, seat, agent)``), then
   ``observer.on_episode_start(env, layout, episode_seed)`` is called if the observer defines it.

Players. ``models.get(agent_id, network_id)`` gives a ``PolicyModel`` or a ``ScriptedPlayer``: an
agent's latest weights under ``"latest"``, its snapshots under ``"ckpt_v<N>"``, scripted and frozen
agents under ``FIXED_NETWORK_ID``. A bot instance exists per ``(agent, env, seat)``: created by the
player's factory at the first reset with the agent at that seat and kept between episodes (also across
lineup changes). Bots get the observation cast to the role's dtypes, the normalized mask and
``StepResult.infos.get(seat)`` of the latest result (``None`` without one; MatchRunner already keeps
that result, so infos cost nothing for envs without scripted seats). A bot's exception or illegal action
is a ``PlayerError`` with the context "worker W, env E, seat P, episode step K, layout L: agent 'X'".

Seat returns, elimination steps, the episode length and the team ranks/scores of the
:class:`MatchResult` come from the env's ``EpisodeTracker`` (one source of truth; the outcome is
resolved with ``resolve_outcome``: default team score = mean of the team's seat returns).
``SeatAssignment.source`` is copied into ``SeatResult.source``.

A lineup naming a snapshot the pool cannot provide is seated as the agent's latest weights with
``collect=True`` (SP1 rule; one warning per (agent, network)); latest and fixed seats must be in the
pool. Only ``latest`` seats may collect: any other seat with ``collect=True`` is a ``ValueError``.

``context`` is a prefix ending with ``", "`` (e.g. ``"worker 3, "``); env contract errors
read ``"worker 3, env 1, seat 2, episode step 7, layout 4p: ..."``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from colosseum.core.errors import PlayerError
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import Tree, tree_map, tree_to_numpy, tree_to_torch
from colosseum.core.types import (
    FIXED_NETWORK_ID,
    LATEST_NETWORK_ID,
    Lineup,
    MatchResult,
    SeatAssignment,
    SeatResult,
    TeamResult,
)
from colosseum.envs.contract import EpisodeTracker
from colosseum.envs.game import GameSpec, StepResult
from colosseum.envs.vector import SubprocessVectorEnv, VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
from colosseum.players.scripted import ScriptedBot, bot_rng, check_bot_action
from colosseum.worker.buffers import put_row

logger = logging.getLogger(__name__)

__all__ = ["ActRecord", "EpisodeEnd", "MatchObserver", "MatchRunner", "ModelPool", "PlayerPool", "ScriptedPlayer"]


@dataclass(frozen=True)
class ScriptedPlayer:
    """A scripted player in a pool: ``factory()`` builds one bot instance (per agent, env and seat)."""

    factory: Callable[[], ScriptedBot]


class PlayerPool(Protocol):
    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None: ...


ModelPool = PlayerPool  # SP2 name


@dataclass
class ActRecord:
    """One decision of one seat, as the observer sees it (numpy only, plus the model state).

    Scripted seats: ``log_prob`` 0.0, ``unit_log_probs`` None, ``pre_state`` None; ``info`` is the
    ``infos`` entry the bot saw (None for neural seats)."""

    agent_id: str
    network_id: str
    obs: Tree                        # numpy, cast to the role's observation dtypes
    global_state: Tree | None        # only when the seat's role declares global_state_space
    mask: Tree | None                # normalized mask (EpisodeTracker)
    action: Tree                     # numpy, as sent to the env
    log_prob: float
    unit_log_probs: np.ndarray | None   # [K] float32 when the action has K > 1 deciders
    pre_state: State                 # model state before this act, leaves [1, ...]
    info: Any = None


@dataclass
class EpisodeEnd:
    """How an env's episode ended. ``live_seats`` excludes every eliminated seat (also this step's);
    ``final_obs`` / ``final_global_state`` (truncation only) hold the live seats only."""

    truncated: bool
    live_seats: list[int]
    final_obs: dict[int, Tree] | None
    final_global_state: dict[int, Tree] | None
    result: MatchResult


class MatchObserver(Protocol):
    """Observer of a MatchRunner. Optional extra method: ``on_episode_start(env, layout, episode_seed)``,
    called after every env reset (after the scripted bots were reset)."""

    def on_act(self, env: int, seat: int, record: ActRecord) -> None: ...
    def on_rewards(self, env: int, rewards: dict[int, float]) -> None: ...
    def on_terminated(self, env: int, seats: list[int]) -> None: ...
    def on_episode_end(self, env: int, end: EpisodeEnd) -> None: ...
    def on_lineup_applied(self, env: int, old: Lineup, new: Lineup) -> None: ...


@dataclass
class _RoleInfo:
    obs: ObsSpec
    action: ActionSpec
    has_global_state: bool


class _EnvState:
    """Per-env bookkeeping of the runner."""

    def __init__(self, tracker: EpisodeTracker, lineup: Lineup) -> None:
        self.tracker = tracker
        self.lineup = lineup
        self.next_lineup: Lineup | None = None
        self.result: StepResult | None = None      # the latest result (its acting seats act next)
        self.masks: dict[int, Tree | None] = {}     # normalized masks of those acting seats
        self.states: dict[int, State] = {}
        self.scripted: dict[int, ScriptedPlayer] = {}   # seats of the current lineup played by bots
        self.episode_index = 0


class MatchRunner:
    """Runs matches on a vector env for given lineups and a player pool (module docstring)."""

    def __init__(
        self,
        *,
        vec_env: VectorEnv | SubprocessVectorEnv,
        lineups: Sequence[Lineup],
        models: PlayerPool,
        observer: MatchObserver | None = None,
        seed: int | None = None,
        max_idle_steps: int = 1000,
        deterministic: bool = False,
        context: str = "",
        match_id_prefix: str = "m",
    ) -> None:
        self._vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.spec: GameSpec = vec_env.spec
        if len(lineups) != self.num_envs:
            raise ValueError(f"need one lineup per env: {len(lineups)} lineups for {self.num_envs} envs")
        self._players = models
        self._observer = observer
        self._on_episode_start = getattr(observer, "on_episode_start", None)
        self._seed = seed
        self._deterministic = deterministic
        self._context = context
        self._prefix = match_id_prefix
        self._episodes_finished = 0
        self._warned_missing: set[tuple[str, str]] = set()
        self._bots: dict[tuple[str, int, int], ScriptedBot] = {}
        self._roles = {
            name: _RoleInfo(
                obs=ObsSpec.from_space(role.observation_space),
                action=ActionSpec.from_space(role.action_space),
                has_global_state=role.global_state_space is not None,
            )
            for name, role in self.spec.roles.items()
        }
        self._envs: list[_EnvState] = []
        for e, lineup in enumerate(lineups):
            tracker = EpisodeTracker(self.spec, max_idle_steps=max_idle_steps, context=f"{context}env {e}")
            env_state = _EnvState(tracker, self._resolve(lineup, e))
            self._init_model_states(env_state)
            self._envs.append(env_state)
        self._reset_envs(list(range(self.num_envs)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lineup(self, env: int) -> Lineup:
        """Env ``env``'s current lineup (after the missing-network fallback)."""
        return self._envs[env].lineup

    def next_lineup(self, env: int) -> Lineup | None:
        """The lineup staged for env ``env`` (applied at its next episode end), if any."""
        return self._envs[env].next_lineup

    def set_next_lineup(self, env: int, lineup: Lineup) -> None:
        """Stage ``lineup`` for env ``env``; it is applied at the env's next episode end.

        Layout, seat count, the players of latest and fixed seats (and every snapshot seat's agent's latest
        model) are checked now; the missing-snapshot fallback is applied at the episode end (a snapshot may
        be loaded in between).
        """
        self._check_lineup(lineup, env)
        self._envs[env].next_lineup = lineup

    @property
    def episodes_finished(self) -> int:
        return self._episodes_finished

    def step(self) -> int:
        """One step of every env (see the module docstring). Returns the env steps taken."""
        actions = self._infer()
        results = self._vec_env.step(actions)
        finished = [e for e in range(self.num_envs) if self._after_step(e, actions[e], results[e])]
        if finished:
            self._reset_envs(finished)
        return self.num_envs

    def close(self) -> None:
        self._vec_env.close()

    # ------------------------------------------------------------------
    # Lineups and players
    # ------------------------------------------------------------------

    def _check_lineup(self, lineup: Lineup, env: int) -> None:
        if lineup.layout not in self.spec.layouts:
            raise ValueError(
                f"{self._context}env {env}: lineup layout {lineup.layout!r} is not one of "
                f"{sorted(self.spec.layouts)}"
            )
        size = self.spec.layout_size(lineup.layout)
        if len(lineup.seats) != size:
            raise ValueError(
                f"{self._context}env {env}: lineup for layout {lineup.layout!r} has "
                f"{len(lineup.seats)} seats, the layout has {size}"
            )
        for seat, assignment in enumerate(lineup.seats):
            net = assignment.network_id
            needed = net if net in (LATEST_NETWORK_ID, FIXED_NETWORK_ID) else LATEST_NETWORK_ID
            if self._players.get(assignment.agent_id, needed) is None:
                raise ValueError(
                    f"{self._context}env {env}: the player pool has no model for agent {assignment.agent_id!r} "
                    f"(network {needed!r})"
                )
            if assignment.collect and net != LATEST_NETWORK_ID:
                raise ValueError(
                    f"{self._context}env {env}, seat {seat}: agent {assignment.agent_id!r} plays network "
                    f"{net!r} with collect=True; only {LATEST_NETWORK_ID!r} seats collect"
                )

    def _resolve(self, lineup: Lineup, env: int) -> Lineup:
        """Validate ``lineup`` and replace snapshots the pool cannot provide by latest + collect."""
        self._check_lineup(lineup, env)
        seats = []
        for assignment in lineup.seats:
            aid, net = assignment.agent_id, assignment.network_id
            if self._players.get(aid, net) is not None:
                seats.append(SeatAssignment(aid, net, assignment.collect, assignment.source))
                continue
            if (aid, net) not in self._warned_missing:
                self._warned_missing.add((aid, net))
                logger.warning(
                    f"{self._context}network {net!r} of {aid!r} is not loaded; "
                    f"seating {LATEST_NETWORK_ID!r} (collecting) instead"
                )
            seats.append(SeatAssignment(aid, LATEST_NETWORK_ID, True, assignment.source))
        return Lineup(layout=lineup.layout, seats=seats)

    def _player(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer:
        player = self._players.get(agent_id, network_id)
        if player is None:
            raise RuntimeError(f"{self._context}player ({agent_id!r}, {network_id!r}) disappeared from the pool")
        return player

    def _where(self, e: int, seat: int, agent_id: str) -> str:
        tracker = self._envs[e].tracker
        return (f"{self._context}env {e}, seat {seat}, episode step {tracker.episode_step}, "
                f"layout {tracker.layout}: agent {agent_id!r}")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _infer(self) -> dict[int, dict[int, Tree]]:
        """Batched inference for neural seats, ``act`` for scripted ones; ``on_act`` in (env, seat) order."""
        groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        bot_seats: list[tuple[int, int]] = []
        for e, env_state in enumerate(self._envs):
            scripted = env_state.scripted
            for seat in sorted(env_state.result.acting):
                if scripted and seat in scripted:
                    bot_seats.append((e, seat))
                    continue
                a = env_state.lineup.seats[seat]
                groups[(a.agent_id, a.network_id)].append((e, seat))

        actions: dict[int, dict[int, Tree]] = {e: {} for e in range(self.num_envs)}
        records: dict[tuple[int, int], ActRecord] = {}
        for (aid, net), seats in groups.items():
            model = self._player(aid, net)
            # every seat of a group has the same spaces: an agent's roles share them (core.roles)
            first_env, first_seat = seats[0]
            info = self._roles[self.spec.role_of(self._envs[first_env].tracker.layout, first_seat)]
            n = len(seats)
            obs_batch = info.obs.allocate((n,))
            mask_batch = info.action.full_mask((n,)) if info.action.has_masks else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                put_row(obs_batch, j, env_state.result.obs[seat])
                if mask_batch is not None:
                    put_row(mask_batch, j, env_state.masks[seat])
            # the records' copies are taken before inference: a model may transform its input in place
            obs_rows = [tree_map(lambda leaf, j=j: leaf[j].copy(), obs_batch) for j in range(n)]
            state_batch = cat_batch([self._envs[e].states[seat] for e, seat in seats])
            out = act(model, tree_to_torch(obs_batch), state_batch,
                      None if mask_batch is None else tree_to_torch(mask_batch),
                      deterministic=self._deterministic)
            act_np = tree_to_numpy(out.actions)
            log_probs = out.log_probs.float().cpu().numpy()
            unit_lps = out.unit_log_probs.float().cpu().numpy() if info.action.num_deciders > 1 else None
            for j, (e, seat) in enumerate(seats):
                env_state = self._envs[e]
                action = tree_map(lambda leaf, j=j: leaf[j].copy(), act_np)
                actions[e][seat] = action
                gs = env_state.result.global_state
                records[(e, seat)] = ActRecord(
                    agent_id=aid,
                    network_id=net,
                    obs=obs_rows[j],
                    global_state=gs[seat] if info.has_global_state and gs is not None else None,
                    mask=env_state.masks.get(seat),
                    action=action,
                    log_prob=float(log_probs[j]),
                    unit_log_probs=None if unit_lps is None else unit_lps[j].copy(),
                    pre_state=env_state.states[seat],
                )
                env_state.states[seat] = None if out.state is None else slice_batch(out.state, j)
        for e, seat in bot_seats:
            actions[e][seat], records[(e, seat)] = self._bot_act(e, seat)
        if self._observer is not None:
            for key in sorted(records):
                self._observer.on_act(key[0], key[1], records[key])
        return actions

    def _bot_act(self, e: int, seat: int) -> tuple[Tree, ActRecord]:
        """One scripted decision: the bot sees obs (role dtypes), the normalized mask and infos[seat]."""
        env_state = self._envs[e]
        a = env_state.lineup.seats[seat]
        role_name = self.spec.role_of(env_state.tracker.layout, seat)
        info = self._roles[role_name]
        obs_batch = info.obs.allocate((1,))
        put_row(obs_batch, 0, env_state.result.obs[seat])
        obs = tree_map(lambda leaf: leaf[0].copy(), obs_batch)
        mask = env_state.masks.get(seat)
        infos = env_state.result.infos
        seat_info = infos.get(seat) if infos else None
        where = self._where(e, seat, a.agent_id)
        bot = self._bots[(a.agent_id, e, seat)]
        try:
            raw = bot.act(obs, mask, seat_info)
        except Exception as exc:  # noqa: BLE001 - any bot failure is reported with its context
            raise PlayerError(f"{where}: act raised {type(exc).__name__}: {exc}") from exc
        action = check_bot_action(self.spec.roles[role_name], raw, mask, where, action_spec=info.action)
        gs = env_state.result.global_state
        record = ActRecord(
            agent_id=a.agent_id, network_id=a.network_id, obs=obs,
            global_state=gs[seat] if info.has_global_state and gs is not None else None,
            mask=mask, action=action, log_prob=0.0, unit_log_probs=None, pre_state=None, info=seat_info,
        )
        return action, record

    def _after_step(self, e: int, actions: dict[int, Tree], result: StepResult) -> bool:
        """Process env ``e``'s step result; True if its episode ended."""
        env_state = self._envs[e]
        tracker = env_state.tracker
        env_state.masks = tracker.on_step(actions, result)
        obs = self._observer
        if obs is not None:
            obs.on_rewards(e, {int(s): float(r) for s, r in result.rewards.items()})
            if result.terminated:
                obs.on_terminated(e, sorted(int(s) for s in result.terminated))
        if not result.episode_over:
            env_state.result = result
            return False

        match_result = self._match_result(e)
        self._episodes_finished += 1
        if obs is not None:
            live = tracker.live_seats()
            truncated = bool(result.truncated)
            gs = result.global_state
            obs.on_episode_end(e, EpisodeEnd(
                truncated=truncated,
                live_seats=live,
                final_obs={s: result.final_obs[s] for s in live} if truncated else None,
                final_global_state=(
                    {s: gs[s] for s in live if s in gs} if truncated and gs is not None else None
                ),
                result=match_result,
            ))
        if env_state.next_lineup is not None:
            old, new = env_state.lineup, self._resolve(env_state.next_lineup, e)
            env_state.lineup, env_state.next_lineup = new, None
            if obs is not None:
                obs.on_lineup_applied(e, old, new)
        env_state.episode_index += 1
        self._init_model_states(env_state)
        return True

    def _match_result(self, e: int) -> MatchResult:
        """The finished episode's result from the env's tracker (returns, eliminations, teams)."""
        env_state = self._envs[e]
        tracker = env_state.tracker
        layout = tracker.layout
        ranks, scores = tracker.team_result()
        returns = tracker.seat_returns()
        seat_specs = self.spec.layouts[layout]
        seats = [
            SeatResult(
                seat=s,
                role=seat_specs[s].role,
                team=seat_specs[s].team,
                agent_id=a.agent_id,
                network_id=a.network_id,
                reward=float(returns[s]),
                eliminated_step=tracker.eliminated_step(s),
                source=a.source,
            )
            for s, a in enumerate(env_state.lineup.seats)
        ]
        return MatchResult(
            match_id=f"{self._prefix}{e}_ep{env_state.episode_index}",
            layout=layout,
            outcome_kind=self.spec.outcome_kind(layout),
            seats=seats,
            teams=[TeamResult(team=t, rank=float(ranks[t]), score=float(scores[t])) for t in sorted(ranks)],
            episode_length=tracker.episode_step,
        )

    def _init_model_states(self, env_state: _EnvState) -> None:
        """Initial model states of every neural seat and the scripted seats of the env's lineup (episode start)."""
        states: dict[int, State] = {}
        scripted: dict[int, ScriptedPlayer] = {}
        for s, a in enumerate(env_state.lineup.seats):
            player = self._player(a.agent_id, a.network_id)
            if isinstance(player, ScriptedPlayer):
                scripted[s] = player
                states[s] = None
            else:
                states[s] = player.initial_state(1)
        env_state.states = states
        env_state.scripted = scripted

    def _episode_seed(self, e: int, k: int) -> int | None:
        if self._seed is None:
            return None
        return int(np.random.SeedSequence([int(self._seed), e, k]).generate_state(1)[0])

    def _reset_bots(self, e: int, episode_seed: int | None) -> None:
        """Create (lazily) and reset the bot of every scripted seat of env ``e``'s lineup."""
        env_state = self._envs[e]
        layout = env_state.lineup.layout
        for seat, player in env_state.scripted.items():
            aid = env_state.lineup.seats[seat].agent_id
            key = (aid, e, seat)
            bot = self._bots.get(key)
            if bot is None:
                try:
                    bot = player.factory()
                except Exception as exc:  # noqa: BLE001 - reported with its context
                    raise PlayerError(f"{self._where(e, seat, aid)}: creating the bot failed "
                                      f"({type(exc).__name__}: {exc})") from exc
                self._bots[key] = bot
            try:
                bot.reset(role=self.spec.role_of(layout, seat), seat=seat, layout=layout,
                          rng=bot_rng(episode_seed, seat, aid))
            except Exception as exc:  # noqa: BLE001 - reported with its context
                raise PlayerError(f"{self._where(e, seat, aid)}: reset raised {type(exc).__name__}: {exc}") from exc

    def _reset_envs(self, envs: list[int]) -> None:
        requests = {
            e: (self._episode_seed(e, self._envs[e].episode_index), self._envs[e].lineup.layout) for e in envs
        }
        results = self._vec_env.reset(requests)
        for e in envs:
            env_state = self._envs[e]
            env_state.masks = env_state.tracker.on_reset(env_state.lineup.layout, results[e])
            env_state.result = results[e]
            if env_state.scripted:
                self._reset_bots(e, requests[e][0])
            if self._on_episode_start is not None:
                self._on_episode_start(e, env_state.lineup.layout, requests[e][0])
```

`tests/game_helpers.py`:
1. `TickGame.__init__` gets a keyword `infos: bool = False` (stored as `self.infos_enabled`); in `_result`, after the global-state block:
```python
        if self.infos_enabled:
            res.infos = {s: {"k": self.k, "t": self.t, "seat": s} for s in acting}
```
   and add to the class docstring: "With ``infos`` every acting seat gets ``infos[seat] = {"k", "t", "seat"}``."
2. `scripted_player` now returns a `ScriptedPlayer` (import it from `colosseum.worker.match_runner`):
```python
def scripted_player(cls, spec: GameSpec, **kwargs):
    """A ``ScriptedPlayer`` for a player pool: each instance is ``cls(**kwargs)`` with ``game_spec = spec``."""
    return ScriptedPlayer(functools.partial(_bot_with_spec, cls, spec, kwargs))
```

- [ ] **Step 4: Run the new tests and the SP2 match-runner tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_match_runner_players.py tests/unit/test_match_runner.py tests/contract/test_rollout_loop_lineups.py tests/contract/test_sp2_eval_engine.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/core/types.py src/colosseum/worker/match_runner.py tests/game_helpers.py \
        tests/unit/test_sp3_match_runner_players.py
git commit -m "feat: MatchRunner player pool with scripted seats (infos, bot RNG, legality gate) and fixed players"
git push origin sp3-league
```

---

### Task T1.4: Fixed players in the worker, launcher and coordinator (`AgentPool` removed)

Spec blocks 1 and 3. The launcher resolves the roles of every agent and loads the fixed players once (main process, before any child starts); workers get them as numpy/`BotSpec` data (`FixedPlayers`, a `rollout_worker_process` argument) and build their own bots and frozen models (frozen agents: one model per agent with its own architecture). The rollout loop serves them under `"fixed"`; they never collect. Lineup resolution in the launcher keeps fixed seats and every seat's `source`. The coordinator takes the roles of every player, rotates data ownership over the trainable agents only (config order, SP2 observation T5.3) and no longer has the dead `AgentPool`.

**Files:**
- Modify: `src/colosseum/worker/rollout_loop.py` (`fixed_players`; `get` serves `"fixed"`)
- Modify: `src/colosseum/worker/rollout_worker.py` (`fixed_players` pass-through)
- Modify: `src/colosseum/launcher.py` (`RunSetup.player_roles` / `.fixed`, `setup_run`, `_worker_main`, `_start_children`, `launch`, `_resolve_lineups`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`player_roles`, `env_steps`, rotation over trainable agents, `AgentPool` gone)
- Delete: `src/colosseum/coordinator/agent_pool.py`
- Modify: `tests/game_helpers.py` (`make_coordinator` uses `resolve_player_roles`)
- Test: `tests/contract/test_sp3_fixed_players_wiring.py`, `tests/integration/test_sp3_fixed_players_worker.py`
- Existing tests that keep passing unchanged: `tests/unit/test_sp2_coordinator.py` (its "no roles" case still misses an agent), `tests/unit/test_sp2_launcher_checkpoints.py` (`_worker_main` without `fixed_players`), `tests/integration/test_sp2_pipelines.py`.
- `scripts/bench_throughput.py`: no change (no fixed players; the worker path without them is unchanged).

**Interfaces:**
- Consumes: `FixedPlayers`, `BotSpec`, `FrozenSpec`, `resolve_player_roles`, `load_fixed_players`, `build_frozen_model`, `make_bot` (T1.2); `ScriptedPlayer`, `FIXED_NETWORK_ID`, `SeatAssignment.source` (T1.3).
- Produces (contract T1.4, plus additions marked *):
  - `RolloutLoop(..., fixed_players: FixedPlayers | None = None)`; `rollout_worker_process(..., fixed_players: FixedPlayers | None = None)`;
  - `Coordinator(config, spec, player_roles, checkpoint_dir, env_steps=lambda: 0)`; *`Coordinator.player_roles` (every agent), `Coordinator.agent_roles` (trainable agents, unchanged meaning); the callable is kept as `Coordinator._env_steps` for T3.x;
  - *`RunSetup.player_roles: dict[str, list[str]]`, *`RunSetup.fixed: FixedPlayers`;
  - *`_worker_main(..., fixed_players: FixedPlayers | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/contract/test_sp3_fixed_players_wiring.py`:
```python
"""Fixed players in the rollout loop, the launcher and the coordinator (SP3 T1.4, spec blocks 1 and 3)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import NetworkConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, build_network, env_spec
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, SeatAssignment, state_dict_to_numpy
from colosseum.launcher import _resolve_lineups, setup_run
from colosseum.players.registry import BotSpec, FixedPlayers, FrozenSpec
from colosseum.worker.match_runner import ScriptedPlayer
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import (
    RecordingBot,
    Tick,
    TickGame,
    agent_role_of,
    frozen_agent,
    make_coordinator,
    make_test_config,
    make_test_model,
    scripted_agent,
)

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]
WIDE = NetworkConfig.model_validate({"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none",
                                                                                             "hidden": 32}})


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _wide_frozen() -> FrozenSpec:
    model = build_network(WIDE, ROLE2)
    return FrozenSpec(agent_id="old", roles=("player",), networks=WIDE.model_dump(mode="json", by_alias=True),
                      model_state=state_dict_to_numpy(model.state_dict()), source="in-test")


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def test_the_rollout_loop_serves_fixed_players_that_never_collect():
    RecordingBot.instances.clear()
    fixed = FixedPlayers(bots={"bot": BotSpec("game_helpers.RecordingBot", {})}, frozen={"old": _wide_frozen()},
                         roles={"bot": ("player",), "old": ("player",)})
    loop, col = make_loop(
        GameFactory((_episode(3), 2)), {"a": lambda: make_test_model(ROLE2)},
        [lineup("2p", "a", SeatAssignment("bot", FIXED_NETWORK_ID, False)),
         lineup("2p", SeatAssignment("old", FIXED_NETWORK_ID, False), "a")],
        fixed_players=fixed,
    )
    try:
        assert isinstance(loop.get("bot", FIXED_NETWORK_ID), ScriptedPlayer)
        old = loop.get("old", FIXED_NETWORK_ID)
        assert not old.training
        assert sum(p.numel() for p in old.parameters()) > sum(p.numel() for p in loop.get("a", "latest").parameters())
        assert loop.get("bot", LATEST_NETWORK_ID) is None and loop.get("nobody", FIXED_NETWORK_ID) is None
        run_steps(loop, 12)                                   # 4 episodes per env
    finally:
        loop.close()
    assert col.chunks and {c.agent_id for c in col.chunks} == {"a"}
    assert {(s.agent_id, s.network_id) for r in col.results for s in r.seats} == {
        ("a", "latest"), ("bot", "fixed"), ("old", "fixed")}
    assert loop.stats["recorded_transitions"] == 3 * 4 * 2   # only the latest seats of "a" record
    assert len(RecordingBot.instances) == 1 and len(RecordingBot.instances[0].acts) == 3 * 4
    RecordingBot.instances.clear()


def test_resolve_lineups_keeps_fixed_seats_and_sources(tmp_path):
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    anchored = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                             SeatAssignment("rnd", FIXED_NETWORK_ID, False, source="anchors")])
    snapshot = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                             SeatAssignment("agent_0", "ckpt_v10", False, source="snapshots")])
    new_ckpts, resolved = _resolve_lineups([anchored, snapshot], coord, ["agent_0"])
    assert resolved == [anchored, snapshot]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"] and "rnd" not in new_ckpts
    missing = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                            SeatAssignment("agent_0", "ckpt_v99", False, source="snapshots")])
    _new, (fallback,) = _resolve_lineups([missing], coord, ["agent_0"])
    assert fallback.seats[1] == SeatAssignment("agent_0", LATEST_NETWORK_ID, True, "snapshots")


def test_the_coordinator_rotates_owners_over_trainable_agents_in_config_order(tmp_path):
    cfg = make_test_config("turns", agents={"rnd": scripted_agent(), "a": {}, "b": {}},
                           matchmaking={"mode": "self_play", "latest_prob": 1.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert coord.player_roles == {"rnd": ["player"], "a": ["player"], "b": ["player"]}
    assert coord.agent_roles == {"a": ["player"], "b": ["player"]}
    lineups = coord.generate_lineups(4, env_offset=0)
    assert [next(s.agent_id for s in lu.seats if s.collect) for lu in lineups] == ["a", "b", "a", "b"]
    assert all(s.agent_id != "rnd" for lu in lineups for s in lu.seats)   # SP2 knobs: no anchors
    with pytest.raises(ConfigError, match=r"no roles resolved for agents \['rnd'\]"):
        Coordinator(cfg, env_spec(cfg), {"a": ["player"], "b": ["player"]}, tmp_path / "ckpt2")
    assert not hasattr(coord, "agent_pool")
    assert importlib.util.find_spec("colosseum.coordinator.agent_pool") is None


def test_setup_run_resolves_every_player_and_loads_fixed_players(tmp_path):
    base = make_test_config("turns")
    _roles, role = agent_role_of(base, "agent_0")
    pt = tmp_path / "old.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    setup = setup_run(cfg, validate=False)
    assert setup.player_roles == {"agent_0": ["player"], "rnd": ["player"], "old": ["player"]}
    assert setup.agent_roles == {"agent_0": ["player"]} and list(setup.agent_configs) == ["agent_0"]
    assert setup.fixed.bots == {"rnd": BotSpec("colosseum.players.RandomBot", {})}
    assert list(setup.fixed.frozen) == ["old"] and setup.fixed.roles == {"rnd": ("player",), "old": ("player",)}
    bad = make_test_config("turns", agents={"agent_0": {}, "old": frozen_agent(tmp_path / "missing.pt")})
    with pytest.raises(ConfigError, match="expected a checkpoint dir"):
        setup_run(bad, validate=False)


def test_worker_main_passes_fixed_players_to_the_rollout_worker(monkeypatch):
    import colosseum.worker.rollout_worker as rollout_worker
    from colosseum.launcher import _worker_main

    recorded: dict = {}
    monkeypatch.setattr(rollout_worker, "rollout_worker_process", lambda **kwargs: recorded.update(kwargs))
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    setup = setup_run(cfg, validate=False)
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("rnd", FIXED_NETWORK_ID, False)])
               for _ in range(cfg.rollout.envs_per_worker)]
    _worker_main(worker_id=0, config=cfg, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
                 agent_configs=setup.agent_configs, role_specs=setup.role_specs, trajectory_queues={},
                 weight_queues={}, stop_event=None, lineups=lineups, fixed_players=setup.fixed)
    assert recorded["fixed_players"] is setup.fixed and recorded["lineups"] == lineups
```

Create `tests/integration/test_sp3_fixed_players_worker.py`:
```python
"""A spawned worker process builds and seats scripted and frozen players from FixedPlayers (SP3 T1.4)."""
from __future__ import annotations

import multiprocessing as mp

import torch

from colosseum.core.registry import build_model
from colosseum.core.types import FIXED_NETWORK_ID, Lineup, SeatAssignment, TrajectoryChunk
from colosseum.launcher import _worker_target, setup_run
from game_helpers import agent_role_of, frozen_agent, make_test_config, scripted_agent

WIDE_KWARGS = {"core": "none", "hidden": 32}


def test_a_spawned_worker_seats_scripted_and_frozen_players_that_never_collect(tmp_path):
    base = make_test_config("turns", networks={"kwargs": WIDE_KWARGS})
    _roles, role = agent_role_of(base, "agent_0")
    pt = tmp_path / "wide.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    config = make_test_config("turns", rollout={"envs_per_worker": 2, "chunk_length": 4}, agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(pt, networks={"kwargs": WIDE_KWARGS})})
    setup = setup_run(config, validate=False)
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("rnd", FIXED_NETWORK_ID, False)]),
               Lineup("2p", [SeatAssignment("wide", FIXED_NETWORK_ID, False), SeatAssignment("agent_0")])]
    ctx = mp.get_context("spawn")
    trajectories, weights, results, stop = ctx.Queue(maxsize=64), ctx.Queue(maxsize=1), ctx.Queue(maxsize=100), \
        ctx.Event()
    proc = ctx.Process(target=_worker_target, daemon=True, kwargs=dict(
        worker_id=0, config=config, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
        agent_configs=setup.agent_configs, role_specs=setup.role_specs,
        trajectory_queues={"agent_0": trajectories}, weight_queues={"agent_0": weights}, stop_event=stop,
        lineups=lineups, results_queue=results, fixed_players=setup.fixed,
    ))
    proc.start()
    try:
        got = [results.get(timeout=60) for _ in range(4)]
        chunks = [TrajectoryChunk.from_payload(trajectories.get(timeout=60)) for _ in range(2)]
    finally:
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    assert {(s.agent_id, s.network_id) for r in got for s in r.seats} == {
        ("agent_0", "latest"), ("rnd", FIXED_NETWORK_ID), ("wide", FIXED_NETWORK_ID)}
    assert all(c.agent_id == "agent_0" for c in chunks)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/contract/test_sp3_fixed_players_wiring.py tests/integration/test_sp3_fixed_players_worker.py -q`
Expected: failures — `TypeError: RolloutLoop.__init__() got an unexpected keyword argument 'fixed_players'`, `AttributeError: 'RunSetup' object has no attribute 'player_roles'`, `AttributeError: 'Coordinator' object has no attribute 'player_roles'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/worker/rollout_loop.py`:
1. Imports: `import functools`; `from colosseum.core.types import FIXED_NETWORK_ID` (next to `LATEST_NETWORK_ID`); `from colosseum.players.registry import FixedPlayers, build_frozen_model, make_bot`; `from colosseum.worker.match_runner import ActRecord, EpisodeEnd, MatchRunner, ScriptedPlayer`.
2. Module docstring: replace the sentence "``RolloutLoop`` is the :class:`MatchRunner`'s model pool (each agent's latest model plus frozen checkpoints) and its observer." by "``RolloutLoop`` is the :class:`MatchRunner`'s player pool (each trainable agent's latest model and loaded snapshots, plus the fixed players: scripted bots and frozen models of ``FixedPlayers``, served under ``FIXED_NETWORK_ID``) and its observer. Fixed players never collect."
3. `__init__` gets the keyword parameter `fixed_players: FixedPlayers | None = None` (after `max_idle_steps`); right after `spec = vec_env.spec` insert:
```python
            # Fixed players (spec block 3): one bot factory per scripted agent and one model per frozen agent
            # (its own architecture, weights from the main process), served under FIXED_NETWORK_ID.
            self._fixed: dict[str, PolicyModel | ScriptedPlayer] = {}
            if fixed_players is not None:
                for aid, bot in fixed_players.bots.items():
                    self._fixed[aid] = ScriptedPlayer(functools.partial(make_bot, bot, spec))
                for aid, frozen in fixed_players.frozen.items():
                    self._fixed[aid] = build_frozen_model(None, frozen, spec)
```
4. Replace the `# ModelPool` section header by `# PlayerPool` and `get` by:
```python
    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None:
        if network_id == FIXED_NETWORK_ID:
            return self._fixed.get(agent_id)
        return self._models.get(agent_id, {}).get(network_id)
```

`src/colosseum/worker/rollout_worker.py`: import `from colosseum.players.registry import FixedPlayers`; add the keyword parameter `fixed_players: FixedPlayers | None = None` (after `max_idle_steps`), document it in the docstring ("``fixed_players``: scripted and frozen agents, built inside this process"), pass `fixed_players=fixed_players` to `RolloutLoop(...)`, and extend the start log line with `f"fixed players={sorted((fixed_players.bots | fixed_players.frozen) if fixed_players else {})}"`.

`src/colosseum/coordinator/coordinator.py`:
1. Delete the `AgentPool` import, the `agent_pool` property and `src/colosseum/coordinator/agent_pool.py`. Import `Callable` from `collections.abc`.
2. Module docstring: "- the players (trainable, scripted, frozen agents) and their roles; owner rotation over the trainable agents in config order and one ``Lineup`` per env (``LineupMatchmaker`` until T3.2);".
3. Replace `__init__` and add the properties:
```python
    def __init__(self, config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]],
                 checkpoint_dir: str | Path, env_steps: Callable[[], int] = lambda: 0) -> None:
        """``player_roles``: roles of EVERY agent (``players.registry.resolve_player_roles``); ``env_steps``:
        the run's global env-step counter (share schedules, SP3 block 5)."""
        self._config = config
        self._spec = spec
        trainable = config.get_trainable_agent_ids()
        players = [*trainable, *config.fixed_agent_ids()]
        missing = [a for a in players if a not in player_roles]
        if missing:
            raise ConfigError(f"Coordinator: no roles resolved for agents {missing}")
        self._trainable = list(trainable)
        self._player_roles = {a: list(player_roles[a]) for a in players}
        self._agent_roles = {a: list(player_roles[a]) for a in trainable}
        self._env_steps = env_steps
        # One RNG for matchmaking and seat permutations: runs with the same seed get the same schedule.
        self._rng = random.Random(config.training.seed)
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
    def player_roles(self) -> dict[str, list[str]]:
        """Roles of every agent (trainable, scripted, frozen) in config order."""
        return {a: list(r) for a, r in self._player_roles.items()}

    @property
    def trainable_agents(self) -> list[str]:
        """Trainable agent ids in config order (the owner rotation order)."""
        return list(self._trainable)
```
4. `generate_lineups`: replace `agents = [a.agent_id for a in self._agent_pool.list_trainable()]` by `agents = self._trainable`.

`src/colosseum/launcher.py`:
1. Imports: `from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, SeatAssignment`; `from colosseum.players.registry import FixedPlayers`.
2. `RunSetup`: docstring "... every agent's roles (``player_roles``), the trainable agents' roles, effective configs and role specs, and the fixed players (loaded once in the main process)."; add the fields `player_roles: dict[str, list[str]]` and `fixed: FixedPlayers` after `role_specs`.
3. `setup_run`:
```python
def setup_run(config: ColosseumConfig, validate: bool = True) -> RunSetup:
    """``RunSetup`` of ``config``; ``validate`` first runs ``validate_run_config``."""
    from colosseum.core.registry import env_spec
    from colosseum.core.roles import agent_role_spec
    from colosseum.players.registry import load_fixed_players, resolve_player_roles

    if validate:
        validate_run_config(config)
    spec = env_spec(config)
    player_roles = resolve_player_roles(config, spec)
    agent_ids = config.get_trainable_agent_ids()
    return RunSetup(
        spec=spec,
        agent_roles={aid: list(player_roles[aid]) for aid in agent_ids},
        agent_configs={aid: config.get_agent_config(aid) for aid in agent_ids},
        role_specs={aid: agent_role_spec(spec, player_roles[aid]) for aid in agent_ids},
        player_roles=player_roles,
        fixed=load_fixed_players(config, spec),
    )
```
4. `_worker_main`: add the keyword parameter `fixed_players: FixedPlayers | None = None` (after `metrics_queue`), mention it in the docstring, and pass `fixed_players=fixed_players` to `rollout_worker_process`.
5. `_start_children`: add `fixed_players=setup.fixed` to the worker `kwargs`.
6. `launch`: after the "Roles" log line add `logger.info(f"  Fixed players: {cfg.fixed_agent_ids() or 'none'}")`, and build the coordinator with every player's roles and the global counter:
```python
        coordinator = Coordinator(cfg, setup.spec, setup.player_roles, checkpoint_dir=self._run_dir.checkpoints,
                                  env_steps=lambda: int(self._env_step_counter.value))
```
7. `_resolve_lineups`: docstring gains "Latest and fixed seats (scripted / frozen agents, ``FIXED_NETWORK_ID``) pass unchanged; every seat keeps its ``source``."; the loop body becomes:
```python
        for seat in lineup.seats:
            if seat.network_id in (LATEST_NETWORK_ID, FIXED_NETWORK_ID):
                seats.append(SeatAssignment(seat.agent_id, seat.network_id, seat.collect, seat.source))
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
                seats.append(SeatAssignment(seat.agent_id, ckpt_id, seat.collect, seat.source))
            else:
                seats.append(SeatAssignment(seat.agent_id, LATEST_NETWORK_ID, True, seat.source))
```

`tests/game_helpers.py`, `make_coordinator`:
```python
def make_coordinator(config, checkpoint_dir):
    """``Coordinator`` for ``config`` with the env's spec and the roles of every player."""
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.registry import env_spec
    from colosseum.players.registry import resolve_player_roles

    spec = env_spec(config)
    return Coordinator(config, spec, resolve_player_roles(config, spec), checkpoint_dir)
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest tests/contract/test_sp3_fixed_players_wiring.py tests/integration/test_sp3_fixed_players_worker.py tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_launcher_checkpoints.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings (`grep -rn agent_pool src tests` finds nothing).

- [ ] **Step 6: Commit and push**

```bash
git add -A src/colosseum/worker src/colosseum/launcher.py src/colosseum/coordinator tests/game_helpers.py \
        tests/contract/test_sp3_fixed_players_wiring.py tests/integration/test_sp3_fixed_players_worker.py
git commit -m "feat: fixed players reach workers; coordinator takes every player's roles (AgentPool removed)"
git push origin sp3-league
```

---

### Task T1.5: `eval -a name`, `play_lineups` with bots, `validate` of fixed agents

Spec blocks 1 (`validate`), 3 and 7 (eval part). `colosseum eval -a greedy` takes a scripted or frozen agent of the config by name (a trainable name without a path is a `ConfigError` with the `name=path` hint; an unknown name is a usage error); `-a name=path` stays as in SP2. The in-process `play_lineups` accepts `ScriptedBot` prototypes (every (agent, env, seat) plays a deep copy) and `ScriptedPlayer`s next to `PolicyModel`s. `validate_config` returns a `ValidationReport`; it imports every scripted agent's class and plays it for a few steps in each enabled layout where it has a seat (every action through the legality gate), and loads every frozen agent (role signature, architecture check once per architecture). `colosseum validate` prints every agent with its kind plus the report lines. The checks of `anchors`, `kickstart.teacher` and `init.from` names come with those fields (T3.3, T4.2, T4.4).

**Files:**
- Modify: `src/colosseum/eval.py` (`play_lineups`, `evaluate`)
- Modify: `src/colosseum/cli.py` (`_parse_agent_spec`, `eval_cmd`, `validate_cmd`)
- Modify: `src/colosseum/core/validation.py` (`ValidationReport`, `_check_frozen_players`, `_check_scripted_players`, `validate_config` returns the report)
- Modify: `src/colosseum/core/registry.py` (`validate_config` returns the report)
- Test: `tests/unit/test_sp3_eval_players.py`, `tests/unit/test_sp3_validate_players.py`
- Existing tests that keep passing unchanged: `tests/integration/test_sp2_eval_cli.py` (`-a no-equals-sign` is an unknown name: exit 2), `tests/unit/test_sp2_validate.py` ("OK: agent 'hunter'" stays a substring), `tests/contract/test_sp2_eval_engine.py`, `tests/unit/test_sp2_launcher_lifecycle.py` (`count_validations` returns the real result).

**Interfaces:**
- Consumes: `ScriptedBot`, `bot_rng`, `check_bot_action`, `PlayerError`, `BotSpec`, `make_bot`, `load_fixed_players`, `resolve_player_roles`, `build_frozen_model` (T1.2); `ScriptedPlayer` (T1.3); `ColosseumConfig.{agent_ids, agent_kind, agent_entry}` (T1.1).
- Produces (contract T1.5, plus additions marked *):
  - `colosseum.core.validation.ValidationReport(lines: list[str])`; `validate_config(config) -> ValidationReport` (also through `colosseum.core.registry.validate_config`);
  - `play_lineups(*, env_fn, models: Mapping[str, PolicyModel | ScriptedBot | ScriptedPlayer], ...)`;
  - *`evaluate(config, agents: Mapping[str, str | None], ...)` (`None` = the scripted or frozen agent of that name);
  - *`cli._parse_agent_spec(spec) -> tuple[str, str | None]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_eval_players.py`:
```python
"""eval with players (SP3 T1.5, spec blocks 3 and 7): play_lineups with bots, -a <name> of fixed agents."""
from __future__ import annotations

import functools
import json

import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.eval import evaluate, play_lineups, schedule_lineups
from colosseum.players import ScriptedBot
from colosseum.players.registry import BotSpec, make_bot
from colosseum.worker.match_runner import ScriptedPlayer
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    TurnTakingGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    make_test_model,
    scripted_agent,
    write_test_config,
)


class FirstLegalBot(ScriptedBot):
    def __init__(self) -> None:
        self.calls = 0

    def act(self, obs, mask, info):
        self.calls += 1
        return np.int64(np.flatnonzero(mask)[0] if mask is not None else 0)


def _pt(tmp_path):
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / "net.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


def invoke(*args):
    return CliRunner().invoke(main, ["eval", *map(str, args)])


def test_play_lineups_takes_bot_prototypes_and_scripted_players():
    spec = TurnTakingGame().spec
    model = make_test_model(spec.roles["player"])
    model.train()
    proto = FirstLegalBot()
    lineups = schedule_lineups(spec, "2p", {"net": ["player"], "bot": ["player"]}, 4)
    results = play_lineups(env_fn=TurnTakingGame, models={"net": model, "bot": proto}, lineups=lineups, num_envs=2,
                           seed=0)
    assert len(results) == 4 and proto.calls == 0          # every seat plays a deep copy of the prototype
    assert {s.agent_id for r in results for s in r.seats} == {"net", "bot"}
    assert model.training                                   # train/eval flags are restored
    rnd = ScriptedPlayer(functools.partial(make_bot, BotSpec("colosseum.players.RandomBot", {}), spec))
    assert len(play_lineups(env_fn=TurnTakingGame, models={"net": model, "bot": rnd}, lineups=lineups,
                            num_envs=2, seed=0)) == 4


def test_evaluate_takes_scripted_and_frozen_agents_by_name(tmp_path):
    pt = _pt(tmp_path)
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    report = evaluate(cfg, {"rnd": None, "old": None, "net": str(pt)}, layouts=None, num_matches=2, seed=0,
                      num_envs=2)
    pairs = {(r["agent_a"], r["agent_b"]) for r in report.layouts["2p"]["pairs"]}
    assert {("rnd", "old"), ("rnd", "net"), ("old", "net")} <= pairs
    with pytest.raises(ConfigError, match=r"trainable agent.*-a agent_0=<checkpoint dir or \.pt>"):
        evaluate(cfg, {"agent_0": None}, layouts=None, num_matches=1)
    with pytest.raises(ConfigError, match="Unknown agent 'ghost'"):
        evaluate(cfg, {"ghost": None}, layouts=None, num_matches=1)


def test_eval_cli_resolves_names_of_scripted_and_frozen_agents(tmp_path):
    pt = _pt(tmp_path)
    cfg_path = write_test_config(tmp_path / "cfg.yaml", "turns",
                                 agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", "rnd", "-a", "old", "-a", f"net={pt}", "-n", 2, "--num-envs", 2,
                    "--seed", 0, "-o", out)
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["agents"] == ["rnd", "old", "net"]
    result = invoke("-c", cfg_path, "-a", "agent_0", "-a", "rnd")
    assert result.exit_code == 1 and "-a agent_0=" in result.stderr, result.output
    result = invoke("-c", cfg_path, "-a", "nobody")
    assert result.exit_code == 2 and "nobody" in result.output
    assert invoke("-c", cfg_path, "-a", "rnd=").exit_code == 2


def test_eval_cli_plays_the_sp2_checkpoint_as_a_frozen_agent_by_name(tmp_path):
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT),
                                     "agents.rnd.kind": "scripted", "agents.rnd.class": "colosseum.players.RandomBot"})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), sort_keys=False))
    out = tmp_path / "r.json"
    result = invoke("-c", cfg_path, "-a", "old", "-a", "rnd", "-n", 2, "--num-envs", 2, "--seed", 0, "-o", out)
    assert result.exit_code == 0, result.output
    rows = json.loads(out.read_text())["layouts"]["2p"]["pairs"]
    assert {(r["agent_a"], r["agent_b"]) for r in rows} == {("old", "rnd"), ("rnd", "old")}
```

Create `tests/unit/test_sp3_validate_players.py`:
```python
"""validate of fixed agents (SP3 T1.5, spec block 1): bots are played under the legality gate, frozen agents
are loaded and checked; `colosseum validate` lists every agent with its kind."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, validate_config
from colosseum.core.roles import role_signature
from colosseum.core.validation import ValidationReport
from colosseum.players import ScriptedBot
from game_helpers import (
    SP2_CHECKPOINT,
    SP2_TTT_TINY,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
    write_test_config,
)


class IllegalTurnBot(ScriptedBot):
    def act(self, obs, mask, info):
        return np.int64(2)                     # TurnTakingGame always masks action 2


class CrashingTurnBot(ScriptedBot):
    def act(self, obs, mask, info):
        raise RuntimeError("bot bug")


def _model_state(config):
    _roles, role = agent_role_of(config, "agent_0")
    model = build_model(config.get_agent_config("agent_0"), role)
    return {k: v.detach().numpy() for k, v in model.state_dict().items()}


def _pt(tmp_path, hidden=16):
    cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": hidden}})
    path = tmp_path / f"net{hidden}.pt"
    torch.save({k: torch.from_numpy(v) for k, v in _model_state(cfg).items()}, path)
    return path


def _turns(**agents):
    return make_test_config("turns", agents={"agent_0": {}, **agents})


def test_validate_plays_scripted_agents_and_loads_frozen_ones(tmp_path):
    report = validate_config(_turns(rnd=scripted_agent(), old=frozen_agent(_pt(tmp_path))))
    assert isinstance(report, ValidationReport)
    assert any(line.startswith("agent 'rnd' (scripted colosseum.players.RandomBot): roles ['player']; played ")
               for line in report.lines), report.lines
    assert any(line.startswith("agent 'old' (frozen): ") for line in report.lines), report.lines


@pytest.mark.parametrize("class_path, message", [
    ("test_sp3_validate_players.IllegalTurnBot",
     r"validate, seat 0, episode step 0, layout 2p: agent 'bad': illegal action: action 2"),
    ("test_sp3_validate_players.CrashingTurnBot", r"agent 'bad': act raised RuntimeError: bot bug"),
    ("game_helpers.TurnTakingGame", "must subclass colosseum.players.ScriptedBot"),
    ("no_such_module.Bot", "cannot be imported"),
])
def test_a_broken_scripted_agent_fails_validate(class_path, message):
    with pytest.raises(ConfigError, match=message):
        validate_config(_turns(bad=scripted_agent(class_path)))


def test_broken_frozen_agents_fail_validate(tmp_path):
    cfg = make_test_config("turns")
    roles, role = agent_role_of(cfg, "agent_0")
    other = tmp_path / "other"
    CheckpointManager(other).save("x", 1, _model_state(cfg), meta_extra={"roles": roles, "role_signature": "nope"})
    with pytest.raises(ConfigError, match="role signature"):
        validate_config(_turns(old=frozen_agent(other / "x" / "ckpt_v1")))
    signed = tmp_path / "signed"
    CheckpointManager(signed).save("x", 1, _model_state(cfg),
                                   meta_extra={"roles": roles, "role_signature": role_signature(role)})
    with pytest.raises(ConfigError, match="meta.json gives"):
        validate_config(_turns(old=frozen_agent(signed / "x" / "ckpt_v1", roles=["player"])))
    with pytest.raises(ConfigError, match="do not match"):
        validate_config(_turns(old=frozen_agent(_pt(tmp_path, hidden=32))))     # wide weights, default networks
    meta = json.loads((signed / "x" / "ckpt_v1" / "meta.json").read_text())
    assert "networks" not in meta                                               # the config's networks are used
    validate_config(_turns(old=frozen_agent(signed / "x" / "ckpt_v1")))


def test_the_sp2_checkpoint_validates_as_a_frozen_agent():
    cfg = load_config(SP2_TTT_TINY, {"agents.old.kind": "frozen", "agents.old.path": str(SP2_CHECKPOINT)})
    report = validate_config(cfg)
    assert any(line.startswith(f"agent 'old' (frozen): {SP2_CHECKPOINT}") for line in report.lines), report.lines


def test_cli_validate_lists_every_agent_with_its_kind(tmp_path):
    path = write_test_config(tmp_path / "cfg.yaml", "turns",
                             agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(_pt(tmp_path))})
    result = CliRunner().invoke(main, ["validate", "-c", str(path)])
    assert result.exit_code == 0, result.output
    for line in ("OK: agent 'agent_0' (trainable)", "OK: agent 'rnd' (scripted)", "OK: agent 'old' (frozen)",
                 "agent 'rnd' (scripted colosseum.players.RandomBot)", "Config is valid."):
        assert line in result.output, result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_eval_players.py tests/unit/test_sp3_validate_players.py -q`
Expected: collection error `ImportError: cannot import name 'ValidationReport' from 'colosseum.core.validation'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/core/validation.py`:
1. Imports: `import json`, `from dataclasses import dataclass, field`; `from colosseum.core.errors import ConfigError, EnvContractError, PlayerError`. (`colosseum.players.*` is imported inside the functions below: `colosseum.players.scripted` imports this module.)
2. Add after the constants:
```python
@dataclass
class ValidationReport:
    """What ``colosseum validate`` prints after the per-agent OK lines: the effective behaviour that is not
    visible in the config (fixed players; later the opponent mix and init reports)."""

    lines: list[str] = field(default_factory=list)
```
3. Add the two checks (before `validate_config`):
```python
def _check_frozen_players(config: ColosseumConfig, spec: GameSpec, fixed: Any, samples: dict[str, tuple],
                          report: ValidationReport) -> None:
    """Every frozen agent: its model is built and its weights loaded (``build_frozen_model``); each distinct
    architecture gets the model check of trainable agents once."""
    from colosseum.core.roles import agent_role_spec
    from colosseum.players.registry import build_frozen_model

    checked: set[str] = set()
    for aid, frozen in fixed.frozen.items():
        model = build_frozen_model(config, frozen, spec)
        key = json.dumps([frozen.networks, list(frozen.roles)], sort_keys=True)
        if key not in checked:
            sample = next((samples[r] for r in frozen.roles if r in samples), None)
            try:
                _check_model(model, agent_role_spec(spec, list(frozen.roles)), sample, f"agent {aid!r}")
            except ConfigError as e:
                raise ConfigError(f"{frozen.networks_source}: {e}") from e
            checked.add(key)
        report.lines.append(f"agent '{aid}' (frozen): {frozen.source}, roles {list(frozen.roles)}")


def _check_scripted_players(config: ColosseumConfig, spec: GameSpec, fixed: Any, layouts: Sequence[str],
                            report: ValidationReport) -> None:
    """Play every scripted agent in each enabled layout where it has a seat: one bot per such seat, other
    seats random legal, up to ``VALIDATE_STEPS`` steps under the contract checks; every bot action goes
    through ``check_bot_action``. A bot failure is a ConfigError naming the agent (SP2 context format)."""
    from colosseum.envs.contract import EpisodeTracker
    from colosseum.players.registry import make_bot
    from colosseum.players.scripted import bot_rng, check_bot_action

    for aid, bot_spec in fixed.bots.items():
        roles = set(fixed.roles[aid])
        played = [name for name in layouts if any(s.role in roles for s in spec.layouts[name])]
        decisions = 0
        rng = np.random.default_rng(0)
        env = make_env(config)
        try:
            for layout in played:
                tracker = EpisodeTracker(spec, max_idle_steps=config.env.max_idle_steps, context="validate")
                try:
                    result = env.reset(seed=0, layout=layout)
                except EnvContractError:
                    raise
                except Exception as e:
                    raise ConfigError(f"env.reset(seed=0, layout={layout!r}) of {config.env.env_class!r} failed: "
                                      f"{type(e).__name__}: {e}") from e
                masks = tracker.on_reset(layout, result)
                bots = {}
                for seat, seat_spec in enumerate(spec.layouts[layout]):
                    if seat_spec.role not in roles:
                        continue
                    bots[seat] = make_bot(bot_spec, spec)
                    try:
                        bots[seat].reset(role=seat_spec.role, seat=seat, layout=layout, rng=bot_rng(0, seat, aid))
                    except Exception as e:
                        raise ConfigError(f"validate, seat {seat}, episode step 0, layout {layout}: agent {aid!r}: "
                                          f"reset raised {type(e).__name__}: {e}") from e
                for step in range(VALIDATE_STEPS):
                    if result.episode_over:
                        break
                    actions = {}
                    for seat in sorted(result.acting):
                        role = spec.roles[spec.role_of(layout, seat)]
                        if seat not in bots:
                            actions[seat] = random_legal_action(role, masks[seat], rng)
                            continue
                        where = f"validate, seat {seat}, episode step {step}, layout {layout}: agent {aid!r}"
                        obs = _as_spec(ObsSpec.from_space(role.observation_space), result.obs[seat])
                        try:
                            raw = bots[seat].act(obs, masks[seat], (result.infos or {}).get(seat))
                        except Exception as e:
                            raise ConfigError(f"{where}: act raised {type(e).__name__}: {e}") from e
                        try:
                            actions[seat] = check_bot_action(role, raw, masks[seat], where)
                        except PlayerError as e:
                            raise ConfigError(str(e)) from e
                        decisions += 1
                    try:
                        result = env.step(actions)
                    except EnvContractError:
                        raise
                    except Exception as e:
                        raise ConfigError(f"env.step of {config.env.env_class!r} failed in layout {layout!r} at "
                                          f"episode step {step + 1}: {type(e).__name__}: {e}") from e
                    masks = tracker.on_step(actions, result)
        finally:
            env.close()
        what = (f"played {decisions} decisions in layouts {played}" if played
                else f"has no seat in the enabled layouts {list(layouts)}")
        report.lines.append(f"agent '{aid}' (scripted {bot_spec.class_path}): roles {sorted(roles)}; {what}")
```
4. `validate_config` returns the report; replace its body from `spec = _game_spec(config)` on:
```python
    from colosseum.coordinator.matchmaker import enabled_layouts, validate_matchmaking
    from colosseum.core.roles import agent_role_spec, resolve_agent_roles
    from colosseum.players.registry import load_fixed_players

    report = ValidationReport()
    spec = _game_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    validate_matchmaking(spec, agent_roles, config.matchmaking)
    fixed = load_fixed_players(config, spec)   # roles, classes, paths and role signatures of fixed agents
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_roles}
    role_specs = {aid: agent_role_spec(spec, roles) for aid, roles in agent_roles.items()}
    for aid, acfg in agent_configs.items():
        if acfg.networks.critic_encoder_class and role_specs[aid].global_state_space is None:
            raise ConfigError(
                f"agent {aid!r}: networks.critic_encoder_class is set, but its roles {agent_roles[aid]} declare "
                f"no global_state_space; remove critic_encoder_class or give the roles a global_state_space"
            )
    layouts = list(enabled_layouts(spec, config.matchmaking))
    samples = _exercise_env(config, spec, layouts)
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
    _check_frozen_players(config, spec, fixed, samples, report)
    _check_scripted_players(config, spec, fixed, layouts, report)
    return report
```
   and add to its docstring: "- every frozen agent: weights, role signature and its architecture (once per architecture); every scripted agent: imported, constructed and played in each enabled layout where it has a seat, every action through the legality gate. Returns a ``ValidationReport``."

`src/colosseum/core/registry.py`, `validate_config`:
```python
def validate_config(config: ColosseumConfig) -> ValidationReport:
    """Every check that can run before a run starts; see ``colosseum.core.validation``. Returns the report
    ``colosseum validate`` prints. Callers use this name (``registry.validate_config``), so tests can
    monkeypatch it here."""
    from colosseum.core.validation import validate_config as _validate_config

    return _validate_config(config)
```
(add `from colosseum.core.validation import ValidationReport` under `TYPE_CHECKING`).

`src/colosseum/eval.py`:
1. Imports: `import copy`, `import functools`; `from colosseum.players.scripted import ScriptedBot`; `from colosseum.worker.match_runner import EpisodeEnd, MatchRunner, ScriptedPlayer`.
2. Module docstring, "Engine" paragraph: "``models`` maps every ``agent_id`` of the lineups to a ``PolicyModel``, a ``ScriptedPlayer`` or a ``ScriptedBot`` instance (a prototype: every (agent, env, seat) plays a deep copy with ``game_spec`` set)."
3. `_FixedModels` docstring: "``PlayerPool`` that serves ``models[agent_id]`` (a model or a ``ScriptedPlayer``) for any network id."
4. Add:
```python
def _copy_bot(prototype: ScriptedBot, spec: GameSpec) -> ScriptedBot:
    bot = copy.deepcopy(prototype)
    bot.game_spec = spec
    return bot
```
5. `play_lineups`: `models: Mapping[str, PolicyModel | ScriptedBot | ScriptedPlayer]`; docstring adds "Scripted players need no eval mode; a ``ScriptedBot`` instance is never played itself (deep copies are)."; the body becomes:
```python
    lineups = list(lineups)
    if not lineups:
        return []
    unknown = sorted({seat.agent_id for lineup in lineups for seat in lineup.seats} - set(models))
    if unknown:
        raise ValueError(f"play_lineups: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_lineups: num_envs must be >= 1, got {num_envs}")
    neural = [m for m in models.values() if isinstance(m, PolicyModel)]
    rng = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()
    was_training = [(m, m.training) for model in neural for m in model.modules()]
    try:
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            for model in neural:
                model.eval()
            n = min(num_envs, len(lineups))
            pending = deque(lineups[n:])
            collector = _Collector(pending, [True] * n)
            vec_env = VectorEnv(env_fn, n)
            try:
                pool = {name: ScriptedPlayer(functools.partial(_copy_bot, player, vec_env.spec))
                        if isinstance(player, ScriptedBot) else player for name, player in models.items()}
                runner = MatchRunner(vec_env=vec_env, lineups=lineups[:n], models=_FixedModels(pool),
                                     observer=collector, seed=seed, max_idle_steps=max_idle_steps,
                                     deterministic=deterministic, context="eval, ", match_id_prefix="eval")
            except BaseException:
                vec_env.close()
                raise
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
```
6. `evaluate`:
```python
def evaluate(config: ColosseumConfig, agents: Mapping[str, str | None], *, layouts: Sequence[str] | None,
             num_matches: int, seed: int | None = None, deterministic: bool = False,
             num_envs: int = 8) -> EvalReport:
    """Load ``agents`` (name -> checkpoint dir or ``.pt``; ``None`` = the scripted or frozen agent of that name
    in the config), schedule, play and summarize."""
    from colosseum.core.registry import env_spec, make_env
    from colosseum.players.registry import BotSpec, make_bot, resolve_player_roles

    spec = env_spec(config)
    models: dict[str, PolicyModel | ScriptedPlayer] = {}
    players: dict[str, list[str]] = {}
    validated: set[str] = set()  # each distinct architecture is checked once
    config_roles: dict[str, list[str]] | None = None
    for name, path in agents.items():
        if path is None:
            entry = config.agent_entry(name)          # ConfigError for an unknown name
            if entry.kind == "trainable":
                raise ConfigError(f"-a {name}: '{name}' is a trainable agent, whose weights are not in the config; "
                                  f"use -a {name}=<checkpoint dir or .pt>")
            if entry.kind == "scripted":
                if config_roles is None:
                    config_roles = resolve_player_roles(config, spec)
                bot = BotSpec(entry.class_path, dict(entry.kwargs))
                make_bot(bot, spec)                    # a bad class or kwargs fail here, as a ConfigError
                models[name] = ScriptedPlayer(functools.partial(make_bot, bot, spec))
                players[name] = list(config_roles[name])
                continue
            path = entry.path
            p = Path(path)
            if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
                raise ConfigError(f"agents.{name}.path={path!r}: expected a checkpoint dir or a .pt file")
        models[name], players[name] = load_eval_model(config, name, path, spec=spec, validated=validated)
    chosen = list(dict.fromkeys(layouts)) if layouts else default_layouts(spec, players)
    # ... the rest of the SP2 body unchanged (unknown layouts, schedule, play_lineups, summarize) ...
```

`src/colosseum/cli.py`:
1. `_parse_agent_spec`:
```python
def _parse_agent_spec(spec: str) -> tuple[str, str | None]:
    """``name=path`` -> (name, path); ``name`` -> (name, None): a scripted or frozen agent of the config."""
    name, sep, path = spec.partition("=")
    if not name or (sep and not path):
        raise click.BadParameter(f"expected name or name=path, got {spec!r}", param_hint="'--agent'")
    return name, (path if sep else None)
```
2. `eval_cmd`: the `--agent` help becomes "name=path or name. path is a checkpoint dir (architecture and roles from its meta.json) or a .pt state_dict (architecture and roles of agents.<name>, else the global networks playing every role). A bare name is a scripted or frozen agent of the config. Repeatable."; replace the path loop by:
```python
        for name, path in specs:
            if path is None:
                if name not in cfg.agent_ids():
                    raise click.BadParameter(f"{name!r} is not an agent of the config; use name=path for a "
                                             f"checkpoint dir or a .pt file", param_hint="'--agent'")
                continue
            p = Path(path)
            if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
                raise click.BadParameter(f"{p}: expected a checkpoint directory or a .pt file",
                                         param_hint="'--agent'")
```
3. `validate_cmd`:
```python
        cfg = load_config(config, _parse_overrides(overrides) or None)
        report = validate_config(cfg)
        for aid in cfg.agent_ids():
            click.echo(f"  OK: agent '{aid}' ({cfg.agent_kind(aid)})")
        for line in report.lines:
            click.echo(f"  {line}")
    click.echo("Config is valid.")
```

- [ ] **Step 4: Run the new tests and the SP2 eval/validate tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_eval_players.py tests/unit/test_sp3_validate_players.py tests/integration/test_sp2_eval_cli.py tests/unit/test_sp2_validate.py tests/contract/test_sp2_eval_engine.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/eval.py src/colosseum/cli.py src/colosseum/core/validation.py src/colosseum/core/registry.py \
        tests/unit/test_sp3_eval_players.py tests/unit/test_sp3_validate_players.py
git commit -m "feat: eval -a <name> for scripted and frozen agents, play_lineups with bots, validate of fixed agents"
git push origin sp3-league
```

---

### Task T2.1: Snapshot retention (`keep_last`/`keep_every`/final) and run-dir resume import

Spec block 4. The FIFO pool of SP2 becomes three retention rules: the newest `keep_last` snapshots (they alone keep `trainer_state.pt`), every snapshot whose version is a multiple of `interval * keep_every` (`model.pt` + `meta.json` only), and every final snapshot (never deleted). Every eviction is reported (`on_evict`) so that T2.2 can tell workers and T2.3 can drop PFSP statistics. `checkpoint.pool_size` (SP2) is read as `keep_last` with one warning; both together are a `ConfigError`. A resume from a run dir carries the stored snapshots of the old run (`model.pt` + `meta.json`, hard-linked where possible, else copied; the source is only read) into the new run's store before any worker starts; a resume from a checkpoint dir or a `.pt` starts with an empty pool. The distributed learner uses the same `CheckpointManager` and therefore the same rules (spec block 8).

**Files:**
- Modify: `src/colosseum/core/config.py` (`CheckpointConfig`, `pool_size` translation, `LEGACY_OVERRIDE_KEYS`, `logger`)
- Modify: `src/colosseum/coordinator/checkpoint_manager.py` (retention, `on_evict`, `import_snapshots`, `_link_or_copy`)
- Modify: `src/colosseum/coordinator/coordinator.py` (retention from the config, `_on_evict`, `import_snapshots`)
- Modify: `src/colosseum/launcher.py` (`_import_snapshot_pool` in `launch`)
- Modify: `src/colosseum/distributed.py` (`CheckpointManager(..., keep_last, keep_every, interval)`)
- Modify: `configs/examples/*.yaml` (`pool_size:` → `keep_last:`; the SP2 copies in `tests/fixtures/sp2/configs/` stay untouched)
- Modify: `scripts/bench_throughput.py::_make_config` (`"keep_last": 10, "keep_every": 0`)
- Modify: `tests/game_helpers.py` (`make_test_config`: `"checkpoint": {"interval": 20, "keep_last": 5, "keep_every": 0}`), `tests/cli_runner.py` (`TINY`: `"checkpoint.keep_last": "5"`)
- Modify (tests whose `pool_size` would now meet `keep_last` from the defaults, or that read the removed field): `tests/unit/test_sp2_checkpoints.py` (every `pool_size=` → `keep_last=`; `checkpoint={"pool_size": 2}` → `checkpoint={"keep_last": 2}`), `tests/integration/test_sp2_pipelines.py` (`checkpoint={"interval": 40, "pool_size": 5}` → `{"interval": 40, "keep_last": 5, "keep_every": 0}`; `CheckpointManager(run.checkpoints, pool_size=5)` → `keep_last=5`), `tests/unit/test_config_v2.py` (`test_defaults`: the checkpoint assertion becomes `assert (cfg.checkpoint.interval, cfg.checkpoint.keep_last, cfg.checkpoint.save_optimizer) == (1000, 20, True)` followed by `assert cfg.checkpoint.keep_every == 10`)
- Modify (extend the T0.1 baseline): `tests/integration/test_sp3_sp2_resume.py`
- Modify (SP2 tests that assumed an empty pool after a run-dir resume; Step 3 shows the edits): `tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr`, `tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature`
- Test: `tests/unit/test_sp3_snapshot_retention.py`

**Interfaces:**
- Consumes: `Coordinator(config, spec, player_roles, checkpoint_dir, env_steps)` and `Coordinator._trainable` (T1.4); `classify_resume_source`, `RESUME_RUN_DIR`, `_read_agent_dir`.
- Produces (contract T2.1, plus additions marked *):
  - `CheckpointConfig(interval=1000, keep_last=20, keep_every=10, save_optimizer=True)`; raw `pool_size` → `keep_last` with one warning;
  - `CheckpointManager(base_dir, keep_last=20, keep_every=0, interval=1, on_evict=None)`; `import_snapshots(src_checkpoints_dir, agent_id, *expected_signature=None) -> list[str]`;
  - `Coordinator.import_snapshots(run_dir)`;
  - *`colosseum.core.config.LEGACY_OVERRIDE_KEYS: set[str]` (`{"checkpoint.pool_size"}`): dotted keys that `--set` accepts although they are not in the schema (translated old knobs; T3.1 / T4.1 add theirs);
  - *`Launcher._import_snapshot_pool(coordinator)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_snapshot_retention.py`:
```python
"""Snapshot storage (SP3 T2.1, spec block 4): keep_last / keep_every / final, trainer_state only in the
keep_last window, eviction callbacks, the pool_size translation and the run-dir resume import."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pytest
import yaml

from colosseum.coordinator import checkpoint_manager as cm_module
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import CheckpointConfig, ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model
from colosseum.launcher import Launcher
from game_helpers import agent_role_of, make_coordinator, make_test_config, make_test_run_dir


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2,), value, np.float32)}


def ids(manager, agent_id="a") -> list[str]:
    return [c.checkpoint_id for c in manager.list_checkpoints(agent_id)]


def model_state(config, agent_id="agent_0") -> dict[str, np.ndarray]:
    _roles, role = agent_role_of(config, agent_id)
    return {k: v.detach().numpy().copy() for k, v in build_model(config.get_agent_config(agent_id), role)
            .state_dict().items()}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def _write(tmp_path, data) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_keep_last_keep_every_and_the_trainer_state_window(tmp_path):
    evicted = []
    mgr = CheckpointManager(tmp_path, keep_last=3, keep_every=2, interval=10,
                            on_evict=lambda agent, ckpt: evicted.append((agent, ckpt)))
    for version in range(10, 101, 10):
        mgr.save("a", version, sd(version), trainer_state=b"opt")
    assert ids(mgr) == ["ckpt_v20", "ckpt_v40", "ckpt_v60", "ckpt_v80", "ckpt_v90", "ckpt_v100"]
    assert evicted == [("a", "ckpt_v10"), ("a", "ckpt_v30"), ("a", "ckpt_v50"), ("a", "ckpt_v70")]
    agent_dir = tmp_path / "a"
    assert sorted(p.parent.name for p in agent_dir.glob("ckpt_v*/trainer_state.pt")) == [
        "ckpt_v100", "ckpt_v80", "ckpt_v90"]
    assert sorted(p.name for p in agent_dir.iterdir()) == sorted(ids(mgr))   # no leftovers
    assert ids(CheckpointManager(tmp_path, keep_last=3, keep_every=2, interval=10)) == ids(mgr)   # a scan agrees


def test_a_final_snapshot_is_never_evicted_and_keep_every_0_is_fifo(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=2, keep_every=0, interval=10)
    mgr.save("a", 7, sd(7), trainer_state=b"opt", meta_extra={"final": True})
    for version in (10, 20, 30):
        mgr.save("a", version, sd(version), trainer_state=b"opt")
    assert ids(mgr) == ["ckpt_v7", "ckpt_v20", "ckpt_v30"]
    assert not (tmp_path / "a" / "ckpt_v7" / "trainer_state.pt").exists()
    assert json.loads((tmp_path / "a" / "ckpt_v7" / "meta.json").read_text())["final"] is True
    assert float(mgr.load_model("a", "ckpt_v7")["w"][0]) == 7.0


def test_the_snapshot_just_saved_is_kept_even_when_it_is_older(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=1)
    mgr.save("a", 20, sd(20))
    mgr.save("a", 5, sd(5))
    assert ids(mgr) == ["ckpt_v5", "ckpt_v20"]


def test_import_snapshots_links_model_and_meta_reads_the_source_only_and_applies_retention(tmp_path):
    old_dir = tmp_path / "old" / "checkpoints"
    old = CheckpointManager(old_dir, keep_last=10)
    for version in (10, 20, 30, 40):
        old.save("a", version, sd(version), trainer_state=b"opt", meta_extra={"role_signature": "sig"})
    leftover = old_dir / "a" / ".tmp-ckpt_v50-0123abcd"            # a stale write dir of the old run
    leftover.mkdir()
    stale = time.time() - cm_module.STALE_TMP_AGE_SEC - 60
    os.utime(leftover, (stale, stale))
    before = sorted(str(p.relative_to(tmp_path / "old")) for p in (tmp_path / "old").rglob("*"))
    evicted = []
    new = CheckpointManager(tmp_path / "new" / "checkpoints", keep_last=2, keep_every=2, interval=10,
                            on_evict=lambda agent, ckpt: evicted.append((agent, ckpt)))
    assert new.import_snapshots(old_dir, "a", expected_signature="sig") == ["ckpt_v20", "ckpt_v30", "ckpt_v40"]
    assert evicted == [("a", "ckpt_v10")]
    for info in new.list_checkpoints("a"):
        assert sorted(p.name for p in info.path.iterdir()) == ["meta.json", "model.pt"]
        src = old_dir / "a" / info.checkpoint_id
        assert (info.path / "model.pt").read_bytes() == (src / "model.pt").read_bytes()
    assert sorted(str(p.relative_to(tmp_path / "old")) for p in (tmp_path / "old").rglob("*")) == before
    assert float(new.load_model("a", "ckpt_v40")["w"][0]) == 40.0
    new.save("a", 50, sd(50), trainer_state=b"opt")                    # retention goes on across saves
    assert ids(new) == ["ckpt_v20", "ckpt_v40", "ckpt_v50"]
    assert new.import_snapshots(tmp_path / "old" / "checkpoints", "nobody") == []


def test_import_snapshots_checks_the_role_signature(tmp_path):
    old = CheckpointManager(tmp_path / "old")
    old.save("a", 10, sd(10), meta_extra={"role_signature": "other-game"})
    new = CheckpointManager(tmp_path / "new")
    with pytest.raises(ConfigError, match="role signature") as info:
        new.import_snapshots(tmp_path / "old", "a", expected_signature="sig")
    assert "ckpt_v10" in str(info.value) and ids(new) == []


def test_checkpoint_config_defaults_and_the_pool_size_translation(tmp_path, caplog):
    cfg = ColosseumConfig.model_validate(_data())
    assert (cfg.checkpoint.keep_last, cfg.checkpoint.keep_every) == (20, 10)
    with caplog.at_level(logging.WARNING, logger="colosseum.core.config"):
        old = load_config(_write(tmp_path, _data(checkpoint={"interval": 50, "pool_size": 5})))
    assert (old.checkpoint.keep_last, old.checkpoint.keep_every, old.checkpoint.interval) == (5, 10, 50)
    assert len([r for r in caplog.records if "pool_size" in r.getMessage()]) == 1
    assert "pool_size" not in old.model_dump(mode="json", by_alias=True)["checkpoint"]
    assert load_config(_write(tmp_path, _data()), {"checkpoint.pool_size": 7}).checkpoint.keep_last == 7
    with pytest.raises(ConfigError, match="both"):
        load_config(_write(tmp_path, _data(checkpoint={"pool_size": 5, "keep_last": 3})))
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, _data(checkpoint={"keep_every": -1})))
    with pytest.raises(ValueError):
        CheckpointConfig(pool_size=0)


def test_the_coordinator_stores_snapshots_by_the_configured_rules(tmp_path):
    cfg = make_test_config("solo", checkpoint={"interval": 10, "keep_last": 2, "keep_every": 3})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    for version in range(10, 71, 10):
        coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": sd(version),
                                       "trainer_state_bytes": b"opt"})
    assert ids(coord.checkpoint_manager, "agent_0") == ["ckpt_v30", "ckpt_v60", "ckpt_v70"]


def test_a_run_dir_resume_imports_the_pool_and_a_checkpoint_dir_resume_does_not(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 5})
    old = make_coordinator(cfg, tmp_path / "old" / "checkpoints")
    for aid in ("a", "b"):
        for version in (20, 40):
            old.save_checkpoint_payload({"agent_id": aid, "policy_version": version,
                                         "model_state": model_state(cfg, aid), "trainer_state_bytes": b"opt"},
                                        meta_extra={"env_steps": version})
    resumed = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 5},
                               training={"resume_from": str(tmp_path / "old")})
    launcher = Launcher(resumed, make_test_run_dir(resumed, tmp_path, name="new"))
    coord = make_coordinator(resumed, launcher._run_dir.checkpoints)
    launcher._import_snapshot_pool(coord)
    assert ids(coord.checkpoint_manager, "a") == ids(coord.checkpoint_manager, "b") == ["ckpt_v20", "ckpt_v40"]
    from_ckpt = make_test_config("turns", agents={"a": {}, "b": {}},
                                 training={"resume_from": str(tmp_path / "old" / "checkpoints" / "a" / "ckpt_v40")})
    launcher2 = Launcher(from_ckpt, make_test_run_dir(from_ckpt, tmp_path, name="new2"))
    coord2 = make_coordinator(from_ckpt, launcher2._run_dir.checkpoints)
    launcher2._import_snapshot_pool(coord2)
    assert ids(coord2.checkpoint_manager, "a") == []
```

Extend `tests/integration/test_sp3_sp2_resume.py` (T0.1): replace the two lines
```python
    versions = _versions(run.root / "checkpoints" / "agent_0")
    assert versions and min(versions) > SP2_CHECKPOINT_VERSION       # T2.1: the imported ckpt_v3 joins the pool
```
by
```python
    agent_dir = run.root / "checkpoints" / "agent_0"
    versions = _versions(agent_dir)
    assert versions[0] == SP2_CHECKPOINT_VERSION and len(versions) >= 2   # the SP2 snapshot joined the new pool
    imported = agent_dir / f"ckpt_v{SP2_CHECKPOINT_VERSION}"
    assert sorted(p.name for p in imported.iterdir()) == ["meta.json", "model.pt"]   # final: kept; no trainer state
    assert "snapshot pool carried over" in log
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_snapshot_retention.py -q`
Expected: failures — `TypeError: CheckpointManager.__init__() got an unexpected keyword argument 'keep_last'` and `AttributeError: 'CheckpointConfig' object has no attribute 'keep_last'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/core/config.py`:
1. `import logging` and `logger = logging.getLogger(__name__)` (after the imports).
2. Replace `CheckpointConfig`:
```python
class CheckpointConfig(StrictModel):
    """Snapshot storage (SP3 spec block 4). Snapshots live in the run dir (``<run>/checkpoints/``).

    Kept: the newest ``keep_last`` (with ``trainer_state.pt``), every snapshot whose version is a multiple of
    ``interval * keep_every`` (``model.pt`` + ``meta.json``), and every final snapshot. ``pool_size`` (SP2) is
    read as ``keep_last`` with a warning.
    """

    interval: int = Field(
        default=1000, ge=1,
        description="Save a checkpoint every N TRAINING steps (optimizer updates / policy versions), not env "
                    "steps (was self_play.checkpoint_interval).",
    )
    keep_last: int = Field(
        default=20, ge=1,
        description="The newest N snapshots per agent are kept, with trainer_state.pt (was pool_size, a FIFO pool).",
    )
    keep_every: int = Field(
        default=10, ge=0,
        description="Also keep every snapshot whose version is a multiple of interval * keep_every (model.pt and "
                    "meta.json only); 0 = off.",
    )
    save_optimizer: bool = Field(
        default=True,
        description="Whether checkpoints include the trainer state (optimizer, LR progress, AMP scaler, "
                    "kickstart, counters) as trainer_state.pt. Without it a resume restores weights and "
                    "policy_version only.",
    )

    @model_validator(mode="before")
    @classmethod
    def _translate_pool_size(cls, data: Any) -> Any:
        """SP2's ``pool_size`` is ``keep_last`` (one warning); both together are an error."""
        if not isinstance(data, dict) or "pool_size" not in data:
            return data
        if "keep_last" in data:
            raise ValueError("checkpoint.pool_size (SP2) and checkpoint.keep_last are both set; keep only keep_last "
                             "(pool_size is its old name)")
        data = dict(data)
        data["keep_last"] = data.pop("pool_size")
        logger.warning(f"checkpoint.pool_size is the SP2 name of checkpoint.keep_last; read as keep_last: "
                       f"{data['keep_last']} (rename it in the config)")
        return data
```
3. Before `apply_overrides`, add:
```python
# Old (SP2) knobs that ``--set`` accepts although they are not in the schema: the config models translate
# them (one warning each) and never store them.
LEGACY_OVERRIDE_KEYS: set[str] = {"checkpoint.pool_size"}
```
   and in `apply_overrides` replace `_check_override_path(parts)` by
```python
        if key not in LEGACY_OVERRIDE_KEYS:
            _check_override_path(parts)
```
4. Module docstring: "SP3: ``checkpoint.keep_last`` / ``keep_every`` (``pool_size`` is read as ``keep_last``)."

`src/colosseum/coordinator/checkpoint_manager.py`:
1. Imports: `from collections.abc import Callable`.
2. Module docstring: replace "Checkpoint storage: atomic per-agent checkpoints, FIFO pool, resume resolution." by "Checkpoint storage: atomic per-agent snapshots, retention (``keep_last`` / ``keep_every`` / final), the run-dir pool import, resume resolution." and add a paragraph:
```
Retention (SP3 spec block 4), after every save and import: the newest ``keep_last`` snapshots, every
snapshot whose version is a multiple of ``interval * keep_every`` (``keep_every > 0``), every final
snapshot (``meta.json`` ``final: true``) and the snapshot just saved are kept; every other one is evicted
(``on_evict(agent_id, checkpoint_id)`` is called after each). Only the newest ``keep_last`` keep
``trainer_state.pt``: resume only ever needs the latest snapshots.
```
3. Replace the class header, `__init__` and the retention part of `save`, and add `_retain` and `import_snapshots`:
```python
class CheckpointManager:
    """Saves snapshots atomically and applies the retention rules per agent (module docstring)."""

    def __init__(self, base_dir: str | Path, keep_last: int = 20, keep_every: int = 0, interval: int = 1,
                 on_evict: Callable[[str, str], None] | None = None) -> None:
        if keep_last < 1 or keep_every < 0 or interval < 1:
            raise ValueError(f"CheckpointManager: need keep_last >= 1, keep_every >= 0, interval >= 1; got "
                             f"{keep_last}, {keep_every}, {interval}")
        self._base_dir = Path(base_dir)
        self._keep_last = int(keep_last)
        self._keep_every = int(keep_every)
        self._interval = int(interval)
        self._on_evict = on_evict
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan()
```
   In `save`, replace the docstring's first line by "Write ``ckpt_v<policy_version>`` atomically, then apply the retention rules (this snapshot is always kept)." and replace everything after the `try/except` block by:
```python
        entries = [c for c in self._index.get(agent_id, []) if c.checkpoint_id != checkpoint_id]
        entries.append(CheckpointInfo(checkpoint_id, agent_id, int(policy_version), final_dir, timestamp, meta))
        entries.sort(key=lambda c: c.policy_version)
        self._index[agent_id] = self._retain(agent_id, entries, protect=checkpoint_id)
        logger.info(f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(self._index[agent_id])}; "
                    f"keep_last {self._keep_last}, keep_every {self._keep_every})")
        return checkpoint_id

    def _retain(self, agent_id: str, entries: list[CheckpointInfo], protect: str | None = None) -> list[CheckpointInfo]:
        """Apply the retention rules to ``entries`` (sorted by version); returns the kept ones."""
        newest = {c.checkpoint_id for c in entries[-self._keep_last:]}
        period = self._interval * self._keep_every
        keep = set(newest) | ({protect} if protect is not None else set())
        for c in entries:
            if (period > 0 and c.policy_version % period == 0) or c.meta.get("final") is True:
                keep.add(c.checkpoint_id)
        kept: list[CheckpointInfo] = []
        for c in entries:
            if c.checkpoint_id not in keep:
                self._evict(agent_id, c.checkpoint_id)
                logger.debug(f"Evicted checkpoint {c.checkpoint_id} of {agent_id}")
                if self._on_evict is not None:
                    self._on_evict(agent_id, c.checkpoint_id)
                continue
            kept.append(c)
            if c.checkpoint_id not in newest:
                (c.path / TRAINER_FILE).unlink(missing_ok=True)
        return kept

    def import_snapshots(self, src_checkpoints_dir: str | Path, agent_id: str,
                         expected_signature: str | None = None) -> list[str]:
        """Carry a previous run's snapshots of ``agent_id`` (``<src>/<agent_id>/ckpt_v*``) into this store.

        ``model.pt`` and ``meta.json`` are hard-linked (copied where a link fails), ``trainer_state.pt`` never;
        then the retention rules apply. The source is only read (no tmp cleanup), a malformed snapshot there
        is a ConfigError (as a strict resume). Ids already in this store are skipped. With
        ``expected_signature`` every source snapshot must carry that ``role_signature``. Returns the ids of the
        agent's snapshots after retention.
        """
        src_agent = Path(src_checkpoints_dir) / check_agent_id(agent_id)
        infos = _read_agent_dir(src_agent, strict=True)
        if expected_signature is not None:
            for info in infos:
                signature = info.meta.get("role_signature")
                if signature != expected_signature:
                    raise ConfigError(
                        f"Snapshot {info.path}: role signature {signature!r} differs from the agent's "
                        f"{expected_signature!r}; the snapshot pool of the resumed run cannot be carried over "
                        f"(resume from a checkpoint dir to start with an empty pool)"
                    )
        entries = list(self._index.get(agent_id, []))
        present = {c.checkpoint_id for c in entries}
        agent_dir = self._agent_dir(agent_id)
        for info in infos:
            if info.checkpoint_id in present:
                continue
            agent_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = agent_dir / f"{_TMP_PREFIX}{info.checkpoint_id}-{uuid.uuid4().hex[:8]}"
            tmp_dir.mkdir()
            try:
                for name in (MODEL_FILE, META_FILE):
                    _link_or_copy(info.path / name, tmp_dir / name)
                os.replace(tmp_dir, agent_dir / info.checkpoint_id)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
            entries.append(CheckpointInfo(info.checkpoint_id, agent_id, info.policy_version,
                                          agent_dir / info.checkpoint_id, info.timestamp, dict(info.meta)))
        entries.sort(key=lambda c: c.policy_version)
        self._index[agent_id] = self._retain(agent_id, entries)
        kept = [c.checkpoint_id for c in self._index[agent_id]]
        if infos:
            logger.info(f"Imported {len(infos)} snapshots of {agent_id} from {src_agent}; pool now {kept}")
        return kept
```
   Add after `_clean_tmp_dirs`:
```python
def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst``; copy when links are impossible (another file system, no support)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
```

`src/colosseum/coordinator/coordinator.py`:
1. Build the manager from the config and add the eviction hook and the import:
```python
        ckpt = config.checkpoint
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every, interval=ckpt.interval,
            on_evict=self._on_evict,
        )
```
```python
    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Storage evicted a snapshot (spec block 4): the matchmaker's candidates come from the store, so it is
        gone from new lineups."""
        logger.debug(f"Snapshot {checkpoint_id} of {agent_id} evicted")

    def import_snapshots(self, run_dir: str | Path) -> None:
        """Run-dir resume (spec block 4): carry the stored snapshots of every trainable agent of ``run_dir``
        into this run's store; a snapshot with another role signature is a ConfigError."""
        for aid in self._trainable:
            kept = self._checkpoint_manager.import_snapshots(Path(run_dir) / "checkpoints", aid,
                                                             expected_signature=self._role_signatures[aid])
            logger.info(f"Resume [{aid}]: snapshot pool carried over from {run_dir}: {kept}")
```
   (`_role_signatures` must be assigned before the manager is constructed is not required: `_on_evict` does not use it; keep the existing order.)

`src/colosseum/launcher.py`: in `launch`, right after `resume_states = self._resolve_resume(...)`, add `self._import_snapshot_pool(coordinator)`; add the method:
```python
    def _import_snapshot_pool(self, coordinator: Coordinator) -> None:
        """A resume from a run dir carries its snapshot pool into this run (spec block 4), before the first
        lineups are drawn; a resume from a checkpoint dir or a .pt starts with an empty pool."""
        from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source

        resume_from = self._config.training.resume_from
        if resume_from and classify_resume_source(resume_from) == RESUME_RUN_DIR:
            coordinator.import_snapshots(resume_from)
```

`src/colosseum/distributed.py`:
```python
    ckpt_cfg = config.checkpoint
    coordinator_ckpt = CheckpointManager(base_dir=run_dir.checkpoints, keep_last=ckpt_cfg.keep_last,
                                         keep_every=ckpt_cfg.keep_every, interval=ckpt_cfg.interval)
```

Configs and support files (exact edits):
```bash
sed -i 's/^  pool_size:/  keep_last:/' configs/examples/*.yaml
grep -n "pool_size" configs/examples/*.yaml          # nothing
```
- `scripts/bench_throughput.py::_make_config`: `checkpoint={"interval": 10**9, "keep_last": 10, "keep_every": 0, "save_optimizer": True}` (comment unchanged).
- `tests/game_helpers.py::make_test_config`: `"checkpoint": {"interval": 20, "keep_last": 5, "keep_every": 0},`.
- `tests/cli_runner.py`: `"checkpoint.keep_last": "5",` instead of `"checkpoint.pool_size": "5",`.
- `tests/unit/test_sp2_checkpoints.py`: `sed -i 's/pool_size=/keep_last=/g; s/"pool_size": 2/"keep_last": 2/' tests/unit/test_sp2_checkpoints.py`.
- `tests/integration/test_sp2_pipelines.py`: `checkpoint={"interval": 40, "keep_last": 5, "keep_every": 0},` and `CheckpointManager(run.checkpoints, keep_last=5)`.
- `tests/unit/test_config_v2.py::test_defaults`: the checkpoint assertion as listed under Files.
- `tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr`: the resumed run now also holds the first run's stored snapshots, so replace
```python
    assert versions and min(versions) > first_final, (first_final, versions)
```
  by
```python
    assert [v for v in versions if v > first_final], (first_final, versions)
    # SP3 T2.1: a run-dir resume carries the first run's stored snapshots (model.pt + meta.json) into the pool
    carried = [v for v in versions if v <= first_final]
    assert first_final in carried and set(carried) <= set(checkpoint_versions(first, "agent_0")), (carried, versions)
    assert not any((second.root / "checkpoints" / "agent_0" / f"ckpt_v{v}" / "trainer_state.pt").is_file()
                   for v in carried)
```
- `tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature`: replace
```python
    assert min(m["policy_version"] for m in metas(second, "agent_0")) > version
```
  by
```python
    resumed = [m["policy_version"] for m in metas(second, "agent_0")]
    # SP3 T2.1: the first run's stored snapshots are carried into the new pool (its final one is never evicted)
    assert version in resumed and max(resumed) > version, (version, resumed)
```

- [ ] **Step 4: Run the new tests and the touched SP2 tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_snapshot_retention.py tests/unit/test_sp2_checkpoints.py tests/unit/test_config_v2.py tests/unit/test_bench_throughput.py tests/integration/test_sp3_sp2_resume.py "tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr" "tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature" -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings (`grep -rn "pool_size" src scripts configs` finds only the translation in `core/config.py`).

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/core/config.py src/colosseum/coordinator src/colosseum/launcher.py src/colosseum/distributed.py \
        configs/examples scripts/bench_throughput.py tests/game_helpers.py tests/cli_runner.py \
        tests/unit/test_sp2_checkpoints.py tests/integration/test_sp2_pipelines.py tests/unit/test_config_v2.py \
        tests/integration/test_sp3_sp2_resume.py tests/integration/test_sp2_league_runs.py \
        tests/integration/test_sp2_game_runs.py \
        tests/unit/test_sp3_snapshot_retention.py
git commit -m "feat: snapshot retention (keep_last, keep_every, final) and the run-dir resume pool import"
git push origin sp3-league
```

---

### Task T2.2: Snapshot eviction on workers

Spec blocks 3 and 4 ("Удаление снимка"): storage reports every eviction to the coordinator; the coordinator's candidates come from the store, so an evicted snapshot never enters a new lineup; the launcher sends the evicted ids to every worker in `WorkerCommand.evict` (retrying a worker whose command queue was full, like checkpoint deltas); a worker unloads such a snapshot as soon as neither the current nor the staged lineup of any of its envs uses it. A lineup that still names an unloaded snapshot falls back to the agent's latest weights with `collect=True` and a warning (SP1 rule, already in `MatchRunner`). PFSP statistics of evicted snapshots are dropped by T2.3.

**Files:**
- Modify: `src/colosseum/core/types.py` (`WorkerCommand.evict`)
- Modify: `src/colosseum/worker/rollout_worker.py` (`_drain_commands` merges `evict`)
- Modify: `src/colosseum/worker/rollout_loop.py` (pending evictions, `_unload_unused`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`_on_evict` records, `take_evictions`)
- Modify: `src/colosseum/launcher.py` (`_worker_evictions`, `_refresh_worker_matches(..., worker_evictions=None)`)
- Test: `tests/contract/test_sp3_snapshot_eviction.py`
- Existing tests that keep passing unchanged: `tests/unit/test_rollout_worker_v2.py` (`_drain_commands` merge), `tests/unit/test_sp2_launcher_checkpoints.py` (`_refresh_worker_matches` with four positional arguments), `tests/unit/test_sp2_launcher_lifecycle.py` (the wrapped refresh passes `*args` through), `tests/contract/test_rollout_loop_lineups.py` (the exact `stats` key set: no new stats keys).

**Interfaces:**
- Consumes: `CheckpointManager(on_evict=...)`, `Coordinator._on_evict` (T2.1); `MatchRunner.next_lineup`, `MatchRunner.lineup` (T1.3).
- Produces (contract T2.2, plus additions marked *):
  - `WorkerCommand(lineups, new_checkpoints={}, evict={})`;
  - `Coordinator.take_evictions() -> dict[str, list[str]]` (evicted since the last call, in eviction order);
  - *`Launcher._refresh_worker_matches(coordinator, agent_ids, command_queues, worker_sent_ckpts, worker_evictions=None)` (`worker_evictions[w]`: `{agent: set(ids)}` still to deliver to worker `w`; `None` = no retry state).

- [ ] **Step 1: Write the failing tests**

Create `tests/contract/test_sp3_snapshot_eviction.py`:
```python
"""Snapshot eviction (SP3 T2.2, spec blocks 3-4): coordinator -> launcher -> worker, unload when unused."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue

import numpy as np

from colosseum.core.types import SeatAssignment, WorkerCommand, state_dict_to_numpy
from colosseum.launcher import Launcher
from colosseum.worker.rollout_worker import _drain_commands
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import Tick, TickGame, make_coordinator, make_test_config, make_test_model

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def test_the_coordinator_hands_out_each_eviction_once(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 2, "keep_every": 0},
                           matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    for version in (10, 20, 30, 40):
        coord.checkpoint_manager.save("agent_0", version, sd(version))
    assert coord.take_evictions() == {"agent_0": ["ckpt_v10", "ckpt_v20"]}
    assert coord.take_evictions() == {}
    used = {s.network_id for lu in coord.generate_lineups(32, 0) for s in lu.seats}
    assert used <= {"latest", "ckpt_v30", "ckpt_v40"}          # evicted snapshots never enter new lineups


def test_drain_commands_merges_evictions():
    q = queue.Queue()
    q.put(WorkerCommand(lineups=[None], evict={"a": ["ckpt_v1"]}))
    q.put(WorkerCommand(lineups=[None], evict={"a": ["ckpt_v2", "ckpt_v1"], "b": ["ckpt_v3"]}))
    cmd = _drain_commands(q)
    assert cmd.evict == {"a": ["ckpt_v1", "ckpt_v2"], "b": ["ckpt_v3"]}
    assert WorkerCommand(lineups=[]).evict == {}


def test_the_worker_unloads_an_evicted_snapshot_once_no_lineup_uses_it(caplog):
    created = []

    def factory():
        created.append(make_test_model(ROLE2))
        return created[-1]

    loop, col = make_loop(GameFactory((_episode(2), 2)), {"a": factory},
                          [lineup("2p", "a", "a"), lineup("2p", "a", "a")])
    ckpt = state_dict_to_numpy(make_test_model(ROLE2).state_dict())

    def snap(ckpt_id):
        return SeatAssignment("a", ckpt_id, collect=False)

    try:
        col.commands.append(WorkerCommand(
            lineups=[lineup("2p", "a", snap("ckpt_v1")), lineup("2p", "a", snap("ckpt_v2"))],
            new_checkpoints={"a": {"ckpt_v1": ckpt, "ckpt_v2": ckpt}}))
        run_steps(loop, 2)              # episode 0 of both envs ends at step 2: the snapshots are seated
        assert loop.get("a", "ckpt_v1") is not None and loop.get("a", "ckpt_v2") is not None
        col.commands.append(WorkerCommand(lineups=[None, lineup("2p", "a", "a")],
                                          evict={"a": ["ckpt_v1", "ckpt_v2", "ckpt_v9"]}))
        run_steps(loop, 1)              # polled; env 0 plays ckpt_v1, env 1 plays ckpt_v2 (latest-only staged)
        assert loop.get("a", "ckpt_v1") is not None and loop.get("a", "ckpt_v2") is not None
        run_steps(loop, 1)              # episode 1 ends: env 1 applies (a, a) -> ckpt_v2 is unused
        assert loop.get("a", "ckpt_v2") is None and loop.get("a", "ckpt_v1") is not None
        col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", "a"), None]))
        run_steps(loop, 2)
        assert loop.get("a", "ckpt_v1") is None and len(created) == 3
        col.commands.append(WorkerCommand(lineups=[lineup("2p", "a", snap("ckpt_v1")), None]))
        with caplog.at_level(logging.WARNING):
            run_steps(loop, 4)          # a lineup naming an unloaded snapshot: latest, collecting (SP1 rule)
    finally:
        loop.close()
    assert "network 'ckpt_v1' of 'a' is not loaded" in caplog.text
    by_episode = [(r.match_id, [s.network_id for s in r.seats]) for r in col.results]
    assert ("w0_e0_ep1", ["latest", "ckpt_v1"]) in by_episode and ("w0_e1_ep1", ["latest", "ckpt_v2"]) in by_episode
    assert ("w0_e0_ep4", ["latest", "latest"]) in by_episode


def test_the_launcher_sends_evictions_to_every_worker_and_retries_a_full_queue(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0},
                           matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    coord.checkpoint_manager.save("agent_0", 20, sd(20))           # evicts ckpt_v10
    launcher = Launcher.__new__(Launcher)                          # _refresh_worker_matches needs only _config
    launcher._config = cfg
    ctx = mp.get_context("spawn")
    q0, q1 = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
    q1.put("occupied")
    sent = [{"agent_0": {"ckpt_v10"}}, {"agent_0": {"ckpt_v10"}}]
    pending: list[dict[str, set[str]]] = [{}, {}]
    launcher._refresh_worker_matches(coord, ["agent_0"], [q0, q1], sent, pending)
    first = q0.get(timeout=5)
    assert first.evict == {"agent_0": ["ckpt_v10"]} and sent[0]["agent_0"] == {"ckpt_v20"}
    assert all(s.network_id != "ckpt_v10" for lu in first.lineups for s in lu.seats)
    assert q1.get(timeout=5) == "occupied" and pending[1] == {"agent_0": {"ckpt_v10"}}   # kept for the retry
    launcher._refresh_worker_matches(coord, ["agent_0"], [q0, q1], sent, pending)
    assert q1.get(timeout=5).evict == {"agent_0": ["ckpt_v10"]} and pending[1] == {"agent_0": set()}
    assert q0.get(timeout=5).evict == {}
```
(With `GameFactory((_episode(2), 2))` an episode takes 2 env steps: from the reset tick to the acting tick, then to the over tick. A command is polled at the start of a step; a staged lineup is applied at the episode end of that env.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/contract/test_sp3_snapshot_eviction.py -q`
Expected: failures — `AttributeError: 'Coordinator' object has no attribute 'take_evictions'`, `TypeError: WorkerCommand.__init__() got an unexpected keyword argument 'evict'`.

- [ ] **Step 3: Write the implementation**

`src/colosseum/core/types.py`, `WorkerCommand`:
```python
@dataclass
class WorkerCommand:
    """Runtime update from the coordinator to one worker.

    ``lineups[e]`` replaces env ``e``'s lineup at its next episode end (``None`` = keep).
    ``new_checkpoints`` (``{agent_id: {checkpoint_id: numpy state_dict}}``) carries only the
    checkpoints the worker does not have yet; they are loaded at once. ``evict``
    (``{agent_id: [checkpoint_id, ...]}``) lists snapshots the storage deleted: the worker unloads each one
    as soon as no current or staged lineup of its envs uses it; ids it does not hold are ignored.
    """

    lineups: list[Lineup | None]
    new_checkpoints: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    evict: dict[str, list[str]] = field(default_factory=dict)
```

`src/colosseum/worker/rollout_worker.py`, `_drain_commands` (docstring: "... and ``evict`` lists of all drained commands are merged (union, first-seen order)."):
```python
    latest: WorkerCommand | None = None
    merged: dict[str, dict[str, Any]] = {}
    evict: dict[str, list[str]] = {}
    lineups: list[Lineup | None] = []
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break
        for aid, ckpts in cmd.new_checkpoints.items():
            merged.setdefault(aid, {}).update(ckpts)
        for aid, ckpt_ids in cmd.evict.items():
            bucket = evict.setdefault(aid, [])
            bucket.extend(c for c in ckpt_ids if c not in bucket)
        if len(cmd.lineups) > len(lineups):
            lineups.extend([None] * (len(cmd.lineups) - len(lineups)))
        for e, lineup in enumerate(cmd.lineups):
            if lineup is not None:
                lineups[e] = lineup
        latest = cmd
    if latest is None:
        return None
    return WorkerCommand(lineups=lineups, new_checkpoints=merged, evict=evict)
```

`src/colosseum/worker/rollout_loop.py`:
1. Module docstring, new bullet: "- Snapshot eviction (spec block 3): ids in ``WorkerCommand.evict`` are unloaded as soon as no env's current or staged lineup uses them; a later lineup naming one gets the agent's latest weights (``MatchRunner``'s SP1 fallback)."
2. In `__init__`, next to `self._policy_versions`: `self._pending_evict: set[tuple[str, str]] = set()`.
3. `_poll_command`, after staging the lineups:
```python
        for aid, ckpt_ids in cmd.evict.items():
            loaded = self._models.get(aid, {})
            self._pending_evict.update((aid, c) for c in ckpt_ids if c in loaded and c != LATEST_NETWORK_ID)
        if self._pending_evict:
            self._unload_unused()
```
4. At the end of `on_lineup_applied`:
```python
        if self._pending_evict:
            self._unload_unused()
```
5. New helper:
```python
    def _unload_unused(self) -> None:
        """Unload pending evicted snapshots that no env's current or staged lineup uses."""
        used: set[tuple[str, str]] = set()
        for e in range(self._runner.num_envs):
            for lu in (self._runner.lineup(e), self._runner.next_lineup(e)):
                if lu is not None:
                    used.update((s.agent_id, s.network_id) for s in lu.seats)
        for aid, ckpt_id in sorted(self._pending_evict - used):
            del self._models[aid][ckpt_id]
            self._pending_evict.discard((aid, ckpt_id))
            logger.info(f"Worker {self.worker_id}: agent {aid}: unloaded evicted snapshot {ckpt_id}")
```

`src/colosseum/coordinator/coordinator.py`:
1. In `__init__`, before the `CheckpointManager` is built: `self._evicted: dict[str, list[str]] = {}`.
2. Replace `_on_evict` and add `take_evictions`:
```python
    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Storage evicted a snapshot (spec block 4): new lineups no longer draw it (the matchmaker's
        candidates come from the store), and workers are told to unload it (``take_evictions``)."""
        self._evicted.setdefault(agent_id, []).append(checkpoint_id)

    def take_evictions(self) -> dict[str, list[str]]:
        """Snapshots evicted since the previous call, per agent in eviction order (each id once)."""
        evicted, self._evicted = self._evicted, {}
        return evicted
```

`src/colosseum/launcher.py`:
1. `__init__`: `self._worker_evictions: list[dict[str, set[str]]] = []`.
2. `launch`: right after `worker_sent_ckpts = [...]`: `self._worker_evictions = [{} for _ in range(cfg.rollout.num_workers)]`.
3. `_monitor_loop`: the refresh call becomes
   `self._refresh_worker_matches(coordinator, agent_ids, command_queues, worker_sent_ckpts, self._worker_evictions)`.
4. Replace `_refresh_worker_matches`:
```python
    def _refresh_worker_matches(
        self,
        coordinator: Coordinator,
        agent_ids: list[str],
        command_queues: list[mp.Queue],
        worker_sent_ckpts: list[dict[str, set]],
        worker_evictions: list[dict[str, set[str]]] | None = None,
    ) -> None:
        """Advance the owner rotation and send every worker fresh lineups.

        Lineups for worker ``w`` are generated at global env offset ``w * envs_per_worker``. Each
        command carries only checkpoints that the worker does not have yet, and the snapshots evicted
        since they were last delivered to it (``worker_evictions[w]``). A delivery counts only after its
        command was put successfully; a full command queue means the worker skips this round and gets the
        deltas and evictions with the next refresh.
        """
        from colosseum.core.types import WorkerCommand

        num_envs = self._config.rollout.envs_per_worker
        if worker_evictions is None:
            worker_evictions = [{} for _ in command_queues]
        for aid, ckpt_ids in coordinator.take_evictions().items():
            for pending in worker_evictions:
                pending.setdefault(aid, set()).update(ckpt_ids)
        coordinator.next_round()
        for worker_id, cq in enumerate(command_queues):
            sent = worker_sent_ckpts[worker_id]
            pending = worker_evictions[worker_id]
            lineups = coordinator.generate_lineups(num_envs, env_offset=worker_id * num_envs)
            new_ckpts, lineups = _resolve_lineups(lineups, coordinator, agent_ids, already_sent=sent)
            evict = {aid: sorted(ids) for aid, ids in pending.items() if ids}
            cmd = WorkerCommand(lineups=list(lineups), new_checkpoints={aid: c for aid, c in new_ckpts.items() if c},
                                evict=evict)
            try:
                cq.put_nowait(cmd)
            except queue.Full:
                logger.debug(f"worker-{worker_id} has not consumed its previous command; skipping this refresh")
                continue
            for aid, ckpts in new_ckpts.items():
                sent.setdefault(aid, set()).update(ckpts)
            for aid, ids in evict.items():
                sent.get(aid, set()).difference_update(ids)
                pending[aid].difference_update(ids)
```

- [ ] **Step 4: Run the new tests and the SP2 worker/launcher tests**

Run: `.venv/bin/python -m pytest tests/contract/test_sp3_snapshot_eviction.py tests/unit/test_rollout_worker_v2.py tests/unit/test_sp2_launcher_checkpoints.py tests/contract/test_rollout_loop_lineups.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/core/types.py src/colosseum/worker src/colosseum/coordinator/coordinator.py \
        src/colosseum/launcher.py tests/contract/test_sp3_snapshot_eviction.py
git commit -m "feat: evicted snapshots reach workers (WorkerCommand.evict) and are unloaded once unused"
git push origin sp3-league
```

---

### Task T2.3: PFSP statistics per player; fixed agents in ratings; `anchor` opponent type

Spec block 5 ("PFSP-статистика", "Рейтинги и метрики"). The coordinator keeps, per layout, an EMA of the score of every trainable agent O's latest weights against each player X it met — the latest of another agent, any snapshot (O's own included) or a scripted / frozen agent — from the member pairs of the SP2 ratings, which now carry the network ids of both sides; only pairs with O@latest on one side and anything but O@latest on the other count. The step is `1 − 2^(−w / halflife_games)` (w = the pair's weight), the prior 0.5; statistics of evicted snapshots are dropped; a resumed run starts empty. `pfsp_weight` gives the candidate weight (`hard` (1 − x)^p, `balanced` x(1 − x), `uniform` 1, floor 1e-6) for the matchmaker of T3.2. Scripted and frozen agents become rating entities (ELO, win-rate matrix, `role_win_rates`); snapshots still count for their agent. The PFSP table goes into `ratings.json` / the `ratings` records (not into WandB columns). The episode aggregator gets the opponent type `anchor` (an opposing seat played by a scripted or frozen agent).

The halflife per agent comes from `matchmaking.pfsp.halflife_games` from T3.1 on; until then (and as its default) it is `DEFAULT_HALFLIFE_GAMES = 200`. T3.3 builds `PfspStats` from each trainable agent's effective `pfsp.halflife_games`.

**Files:**
- Create: `src/colosseum/league/__init__.py` (package docstring only; T3.2 adds the exports), `src/colosseum/league/pfsp.py`
- Modify: `src/colosseum/coordinator/ratings.py` (`MemberPair.net_a` / `net_b`, `_classify`, `member_pairs`, docstring)
- Modify: `src/colosseum/coordinator/coordinator.py` (ratings over every player, `PfspStats`, `forget` on eviction, `pfsp` property, `ratings_snapshot` with `"pfsp"`)
- Modify: `src/colosseum/metrics/aggregator.py` (`OPPONENT_TYPES`, `opponent_type`)
- Modify: `src/colosseum/metrics/hub.py` (WandB rows without the PFSP tables)
- Modify (tests pinned to the old shapes): `tests/unit/test_sp2_ratings.py` (four `MemberPair(...)` literals gain the network ids), `tests/unit/test_sp2_metrics.py` (four W/D/L dict literals gain `"anchor": [0, 0, 0]`), `tests/integration/test_sp2_metrics_outputs.py` (the W/D/L key set gains `"anchor"`; `ratings.json` layouts carry `"pfsp"`)
- Test: `tests/unit/test_sp3_pfsp_stats.py`

**Interfaces:**
- Consumes: `member_pairs`, `RatingBook` (SP2); `Coordinator._on_evict` (T2.2); `FIXED_NETWORK_ID`, `LATEST_NETWORK_ID` (T1.3).
- Produces (contract `colosseum.league.pfsp`, plus additions marked *):
  - `PlayerKey = tuple[str, str]`; `pfsp_weight(score, weighting, exponent) -> float`;
  - `PfspStats(halflife_by_agent, prior=0.5)` with `update(result)`, `score(layout, owner, player)`, `games(layout, owner, player)`, `forget(player)`, `snapshot()` (`{layout: {owner: {"agent@net": {"score", "games"}}}}`);
  - *`DEFAULT_HALFLIFE_GAMES = 200.0`, *`PFSP_MIN_WEIGHT = 1e-6`, *`PFSP_WEIGHTINGS = ("hard", "balanced", "uniform")`, *`player_name(player) -> str` (`"agent@net"`);
  - *`MemberPair(kind, a, b, score_a, weight, role_a="", role_b="", net_a="", net_b="")`;
  - `Coordinator.ratings_snapshot()` adds `"pfsp"` per layout; *`Coordinator.pfsp` (the `PfspStats`, read by T3.2's `MatchmakerContext.pfsp_score`);
  - *`colosseum.metrics.aggregator.OPPONENT_TYPES = ("latest", "past", "arena", "anchor")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp3_pfsp_stats.py`:
```python
"""PFSP statistics per player, fixed agents as rating entities, the anchor opponent type (SP3 T2.3, block 5)."""
from __future__ import annotations

import json

import gymnasium
import numpy as np
import pytest

from colosseum.coordinator.ratings import MemberPair, RatingBook, member_pairs
from colosseum.core.types import FIXED_NETWORK_ID, MatchResult, SeatResult, TeamResult
from colosseum.envs.game import GameSpec
from colosseum.league.pfsp import (
    DEFAULT_HALFLIFE_GAMES,
    PFSP_MIN_WEIGHT,
    PfspStats,
    pfsp_weight,
    player_name,
)
from colosseum.metrics.aggregator import OPPONENT_TYPES, EpisodeAggregator, opponent_type
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from game_helpers import make_coordinator, make_test_config, scripted_agent

SPEC = GameSpec.symmetric([2], gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32), gymnasium.spaces.Discrete(3))


def seat(i, team, agent, network="latest", role="player") -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=0.0)


def duel(a, net_a, b, net_b, rank_a, rank_b, layout="2p") -> MatchResult:
    return MatchResult(match_id="m", layout=layout, outcome_kind="wdl",
                       seats=[seat(0, 0, a, net_a), seat(1, 1, b, net_b)],
                       teams=[TeamResult(0, float(rank_a), 0.0), TeamResult(1, float(rank_b), 0.0)], episode_length=3)


def step(weight, halflife=DEFAULT_HALFLIFE_GAMES) -> float:
    return 1.0 - 2.0 ** (-weight / halflife)


def test_pfsp_weights():
    assert pfsp_weight(0.25, "hard", 2.0) == pytest.approx(0.5625)
    assert pfsp_weight(0.25, "balanced", 7.0) == pytest.approx(0.1875)
    assert pfsp_weight(0.9, "uniform", 2.0) == 1.0
    assert pfsp_weight(1.0, "hard", 2.0) == PFSP_MIN_WEIGHT == 1e-6
    assert pfsp_weight(0.0, "balanced", 1.0) == PFSP_MIN_WEIGHT
    with pytest.raises(ValueError, match="weighting"):
        pfsp_weight(0.5, "steep", 1.0)


def test_the_ema_starts_at_the_prior_and_steps_by_the_halflife():
    stats = PfspStats({"a": 2.0})
    past = ("a", "ckpt_v3")
    assert stats.score("2p", "a", past) == 0.5 and stats.games("2p", "a", past) == 0.0
    stats.update(duel("a", "latest", "a", "ckpt_v3", 1, 2))     # a@latest beats its own snapshot
    assert stats.score("2p", "a", past) == pytest.approx(0.5 + step(1.0, 2.0) * 0.5)
    stats.update(duel("a", "ckpt_v3", "a", "latest", 1, 2))     # and loses from the other side
    x = 0.5 + step(1.0, 2.0) * 0.5
    assert stats.score("2p", "a", past) == pytest.approx(x + step(1.0, 2.0) * (0.0 - x))
    assert stats.games("2p", "a", past) == 2.0
    with pytest.raises(ValueError, match="halflife"):
        PfspStats({"a": 0.0})


def test_only_pairs_with_an_owner_at_latest_count_and_networks_are_told_apart():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES, "b": DEFAULT_HALFLIFE_GAMES})
    stats.update(duel("a", "latest", "b", "latest", 1, 2))           # both are owners
    stats.update(duel("a", "latest", "b", "ckpt_v4", 2, 1))          # a against b's snapshot
    stats.update(duel("a", "ckpt_v1", "b", "ckpt_v4", 1, 2))         # nobody at latest: ignored
    stats.update(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 1))  # a draw against an anchor
    stats.update(duel("a", "latest", "a", "latest", 1, 2))           # O@latest against O@latest: no pair
    snap = stats.snapshot()
    assert set(snap) == {"2p"} and set(snap["2p"]) == {"a", "b"}     # fixed agents are never owners
    assert set(snap["2p"]["a"]) == {"b@latest", "b@ckpt_v4", "rnd@fixed"}
    assert set(snap["2p"]["b"]) == {"a@latest"}
    assert snap["2p"]["a"]["rnd@fixed"] == {"score": pytest.approx(0.5), "games": 1.0}
    assert stats.score("2p", "a", ("b", "latest")) > 0.5 > stats.score("2p", "b", ("a", "latest"))
    assert stats.score("2p", "a", ("b", "ckpt_v4")) < 0.5
    assert stats.score("4p", "a", ("b", "latest")) == 0.5                # per layout
    assert player_name(("rnd", FIXED_NETWORK_ID)) == "rnd@fixed"


def test_team_pairs_split_their_weight_and_owners_outside_the_map_are_not_tracked():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES})
    two_v_two = MatchResult(match_id="m", layout="2v2", outcome_kind="wdl",
                            seats=[seat(0, 0, "a"), seat(1, 0, "a"),
                                   seat(2, 1, "b", "ckpt_v1"), seat(3, 1, "b", "ckpt_v1")],
                            teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    stats.update(two_v_two)                                           # four member pairs of weight 1/4
    assert stats.games("2v2", "a", ("b", "ckpt_v1")) == pytest.approx(1.0)
    expected = 0.5
    for _ in range(4):
        expected += step(0.25) * (1.0 - expected)
    assert stats.score("2v2", "a", ("b", "ckpt_v1")) == pytest.approx(expected)
    stats.update(duel("b", "latest", "a", "ckpt_v2", 1, 2))           # b is not in the halflife map
    assert "2p" not in stats.snapshot()


def test_forget_drops_a_player_in_every_layout_and_for_every_owner():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES, "b": DEFAULT_HALFLIFE_GAMES})
    stats.update(duel("a", "latest", "b", "ckpt_v4", 1, 2))
    stats.update(duel("a", "latest", "b", "ckpt_v4", 1, 2, layout="4p"))
    stats.update(duel("a", "latest", "b", "latest", 1, 2))
    stats.forget(("b", "ckpt_v4"))
    assert stats.games("2p", "a", ("b", "ckpt_v4")) == 0.0 and stats.games("4p", "a", ("b", "ckpt_v4")) == 0.0
    assert stats.games("2p", "a", ("b", "latest")) == 1.0


def test_member_pairs_carry_the_network_ids_of_both_sides():
    assert member_pairs(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2)) == [
        MemberPair("cross", "a", "rnd", 1.0, 1.0, "player", "player", "latest", "fixed")]
    assert member_pairs(duel("a", "ckpt_v3", "a", "latest", 1, 2)) == [
        MemberPair("past", "a", "a", 0.0, 1.0, "player", "player", "latest", "ckpt_v3")]


def test_scripted_and_frozen_agents_are_rating_entities():
    book = RatingBook(SPEC, ["a", "rnd"])
    book.update(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2))
    snap = book.snapshot()["2p"]
    assert snap["elo"]["a"] > 1200.0 > snap["elo"]["rnd"]
    assert snap["win_rates"]["a"]["rnd"] == 1.0 and snap["games"]["rnd"]["a"] == 1
    assert snap["wr_vs_past"]["rnd"] is None
    assert snap["role_win_rates"]["player"]["rnd"]["a"] == 0.0


def test_the_coordinator_rates_fixed_agents_and_publishes_the_pfsp_table(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "rnd": scripted_agent()})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.report_match_result(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2))
    snap = coord.ratings_snapshot()["2p"]
    assert set(snap["elo"]) == {"a", "rnd"}
    assert snap["pfsp"] == {"a": {"rnd@fixed": {"score": pytest.approx(0.5 + step(1.0) * 0.5), "games": 1.0}}}
    assert coord.pfsp.score("2p", "a", ("rnd", FIXED_NETWORK_ID)) > 0.5
    json.dumps(coord.ratings_snapshot())                             # ratings.json stays JSON


def test_an_evicted_snapshot_is_forgotten_by_pfsp(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    state = {"w": np.zeros((2, 3), np.float32)}
    coord.checkpoint_manager.save("agent_0", 10, state)
    coord.report_match_result(duel("agent_0", "latest", "agent_0", "ckpt_v10", 1, 2))
    assert coord.pfsp.games("2p", "agent_0", ("agent_0", "ckpt_v10")) == 1.0
    coord.checkpoint_manager.save("agent_0", 20, state)               # evicts ckpt_v10
    assert coord.pfsp.games("2p", "agent_0", ("agent_0", "ckpt_v10")) == 0.0
    assert "agent_0@ckpt_v10" not in json.dumps(coord.ratings_snapshot())


def test_an_opponent_played_by_a_fixed_agent_is_an_anchor():
    assert OPPONENT_TYPES == ("latest", "past", "arena", "anchor")
    result = duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2)
    assert opponent_type(result, result.seats[0]) == "anchor"
    team = MatchResult(match_id="m", layout="2v2", outcome_kind="wdl",
                       seats=[seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "b"), seat(3, 1, "rnd", FIXED_NETWORK_ID)],
                       teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    assert opponent_type(team, team.seats[0]) == "anchor" and opponent_type(team, team.seats[2]) == "arena"
    agg = EpisodeAggregator()
    agg.add(result)
    out = agg.flush()
    assert set(out) == {"a"}                                          # fixed seats are not aggregated
    assert out["a"]["wdl"] == {"latest": [0, 0, 0], "past": [0, 0, 0], "arena": [0, 0, 0], "anchor": [1, 0, 0]}


def test_wandb_rows_leave_out_the_pfsp_tables(tmp_path):
    class Rows:
        def __init__(self) -> None:
            self.rows: list[dict] = []

        def log_global(self, row, step):
            self.rows.append(row)

        def log_train(self, agent_id, values, step):
            pass

    rows = Rows()
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    hub = MetricsHub(writer=writer, ratings_path=tmp_path / "ratings.json", agent_ids=["a"], total_timesteps=10,
                     log_interval=1, console_interval_sec=0.0, wandb_logger=rows)
    ratings = {"2p": {"elo": {"a": 1210.0, "rnd": 1190.0}, "win_rates": {}, "games": {}, "wr_vs_past": {"a": None},
                      "past_games": {"a": 0}, "scores": {}, "cross_play": {}, "role_win_rates": {},
                      "pfsp": {"a": {"rnd@fixed": {"score": 0.6, "games": 1.0}}}}}
    try:
        hub.maybe_tick(env_steps=5, ratings=ratings, queue_depths={}, force=True)
    finally:
        writer.close()
    assert rows.rows[0]["ratings/2p/elo/rnd"] == 1190.0
    assert not any("/pfsp/" in key for key in rows.rows[0])
    written = json.loads((tmp_path / "ratings.json").read_text())
    assert written["layouts"]["2p"]["pfsp"]["a"]["rnd@fixed"]["games"] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_pfsp_stats.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.league'`.

- [ ] **Step 3: Write the implementation**

Create `src/colosseum/league/__init__.py`:
```python
"""League: opponent selection (matchmaker interface and the built-in mixture matchmaker, SP3 T3.x) and the
PFSP statistics per player (``colosseum.league.pfsp``)."""
```

Create `src/colosseum/league/pfsp.py`:
```python
"""PFSP statistics per player (SP3 spec block 5).

For every layout and every trainable agent O (the owner) an EMA of O@latest's score against each player X
it met: the latest weights of another agent, a snapshot of any agent (O's own included) or a scripted /
frozen agent (network ``"fixed"``). Source: the member pairs of a result (``coordinator.ratings.member_pairs``:
seats of different teams, score by team ranks, weight ``1 / (T - 1)`` split between the counted pairs, a
draw = 0.5). Only pairs with O@latest on one side and anything but O@latest on the other count. The EMA step
of a pair of weight ``w`` is ``1 - 2 ** (-w / halflife_games)``; before the first game the score is the prior
(0.5). Statistics of an evicted snapshot are dropped (``forget``); a resumed run starts empty (as SP2 ratings).

``pfsp_weight`` turns a score into a candidate weight for the matchmaker: ``hard`` ``(1 - x) ** p`` (focus on
the players O loses to), ``balanced`` ``x * (1 - x)`` (focus on even players), ``uniform`` 1; floor 1e-6.
"""

from __future__ import annotations

from collections.abc import Mapping

from colosseum.coordinator.ratings import member_pairs
from colosseum.core.types import LATEST_NETWORK_ID, MatchResult

PlayerKey = tuple[str, str]  # (agent_id, network_id)

DEFAULT_HALFLIFE_GAMES = 200.0
PFSP_MIN_WEIGHT = 1e-6
PFSP_WEIGHTINGS = ("hard", "balanced", "uniform")


def player_name(player: PlayerKey) -> str:
    """``"agent@network"``: the key of a player in snapshots (``ratings.json``)."""
    return f"{player[0]}@{player[1]}"


def pfsp_weight(score: float, weighting: str, exponent: float) -> float:
    """Candidate weight of a player against whom the owner's score is ``score`` (module docstring)."""
    x = min(1.0, max(0.0, float(score)))
    if weighting == "hard":
        weight = (1.0 - x) ** float(exponent)
    elif weighting == "balanced":
        weight = x * (1.0 - x)
    elif weighting == "uniform":
        weight = 1.0
    else:
        raise ValueError(f"unknown PFSP weighting {weighting!r}; expected one of {PFSP_WEIGHTINGS}")
    return max(weight, PFSP_MIN_WEIGHT)


class PfspStats:
    """Per layout, per owner, per player: ``[score EMA, games]`` (module docstring).

    ``halflife_by_agent`` maps every owner to track (the trainable agents) to its ``halflife_games``; pairs of
    other owners are ignored.
    """

    def __init__(self, halflife_by_agent: Mapping[str, float], prior: float = 0.5) -> None:
        for agent_id, halflife in halflife_by_agent.items():
            if not float(halflife) > 0.0:
                raise ValueError(f"PFSP halflife_games of {agent_id!r} must be > 0, got {halflife}")
        self._halflife = {agent_id: float(h) for agent_id, h in halflife_by_agent.items()}
        self._prior = float(prior)
        self._table: dict[str, dict[str, dict[PlayerKey, list[float]]]] = {}

    def update(self, result: MatchResult) -> None:
        """Fold one finished match into the statistics of its layout."""
        for pair in member_pairs(result):
            self._record(result.layout, pair.a, pair.net_a, (pair.b, pair.net_b), pair.score_a, pair.weight)
            self._record(result.layout, pair.b, pair.net_b, (pair.a, pair.net_a), 1.0 - pair.score_a, pair.weight)

    def _record(self, layout: str, owner: str, network: str, player: PlayerKey, score: float, weight: float) -> None:
        if network != LATEST_NETWORK_ID or player == (owner, LATEST_NETWORK_ID):
            return
        halflife = self._halflife.get(owner)
        if halflife is None:
            return
        cell = self._table.setdefault(layout, {}).setdefault(owner, {}).setdefault(player, [self._prior, 0.0])
        cell[0] += (1.0 - 2.0 ** (-float(weight) / halflife)) * (float(score) - cell[0])
        cell[1] += float(weight)

    def _cell(self, layout: str, owner: str, player: PlayerKey) -> list[float] | None:
        return self._table.get(layout, {}).get(owner, {}).get(tuple(player))

    def score(self, layout: str, owner: str, player: PlayerKey) -> float:
        """The owner's score EMA against ``player`` in ``layout`` (the prior before the first game)."""
        cell = self._cell(layout, owner, player)
        return self._prior if cell is None else cell[0]

    def games(self, layout: str, owner: str, player: PlayerKey) -> float:
        """Summed pair weights behind ``score`` (fractional for team and FFA layouts)."""
        cell = self._cell(layout, owner, player)
        return 0.0 if cell is None else cell[1]

    def forget(self, player: PlayerKey) -> None:
        """Drop ``player`` everywhere (an evicted snapshot)."""
        key = tuple(player)
        for owners in self._table.values():
            for players in owners.values():
                players.pop(key, None)

    def snapshot(self) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
        """``{layout: {owner: {"agent@net": {"score", "games"}}}}`` (JSON-ready; players sorted)."""
        return {
            layout: {
                owner: {player_name(p): {"score": cell[0], "games": cell[1]} for p, cell in sorted(players.items())}
                for owner, players in owners.items()
            }
            for layout, owners in self._table.items()
        }
```

`src/colosseum/coordinator/ratings.py`:
1. Module docstring, replace the "Rating entities follow SP1's code ..." sentence by: "Rating entities: ELO and the win-rate matrix are keyed by the base ``agent_id`` (SP2): a snapshot of X playing Y counts as X vs Y (snapshot ratings are SP4); scripted and frozen agents (network ``"fixed"``) are entities of their own (SP3). "Latest of X vs a snapshot of X" feeds ``wr_vs_past``; two seats of the same agent that are both latest (or both not latest) carry no signal and are skipped. Member pairs carry the network ids of both sides (PFSP statistics per player, ``colosseum.league.pfsp``)."
2. `MemberPair`: docstring adds "``net_a`` / ``net_b``: the network ids of the two sides."; add the fields `net_a: str = ""` and `net_b: str = ""` after `role_b`.
3. `_classify` and `member_pairs`:
```python
def _classify(sa: SeatResult, sb: SeatResult, score: float) -> tuple | None:
    if sa.agent_id != sb.agent_id:
        return "cross", sa.agent_id, sb.agent_id, score, sa.role, sb.role, sa.network_id, sb.network_id
    a_latest = sa.network_id == LATEST_NETWORK_ID
    b_latest = sb.network_id == LATEST_NETWORK_ID
    if a_latest and not b_latest:
        return "past", sa.agent_id, sa.agent_id, score, sa.role, sb.role, sa.network_id, sb.network_id
    if b_latest and not a_latest:
        return "past", sb.agent_id, sb.agent_id, 1.0 - score, sb.role, sa.role, sb.network_id, sa.network_id
    return None
```
   and in `member_pairs` the loop body becomes
```python
            for kind, a, b, s, role_a, role_b, net_a, net_b in counted:
                pairs.append(MemberPair(kind, a, b, s, share / len(counted), role_a, role_b, net_a, net_b))
```

`src/colosseum/coordinator/coordinator.py`:
1. Imports: `from colosseum.league.pfsp import DEFAULT_HALFLIFE_GAMES, PfspStats`.
2. In `__init__`, before the `CheckpointManager` is built, add
   `self._pfsp = PfspStats({a: DEFAULT_HALFLIFE_GAMES for a in trainable})`, and replace `self._ratings = RatingBook(spec, trainable)` by `self._ratings = RatingBook(spec, players)` (every player is a rating entity).
3. `_on_evict` also drops the snapshot's statistics:
```python
    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """Storage evicted a snapshot (spec block 4): new lineups no longer draw it (the matchmaker's
        candidates come from the store), its PFSP statistics are dropped, and workers are told to unload it
        (``take_evictions``)."""
        self._evicted.setdefault(agent_id, []).append(checkpoint_id)
        self._pfsp.forget((agent_id, checkpoint_id))
```
4. Property, results and snapshot:
```python
    @property
    def pfsp(self) -> PfspStats:
        """PFSP statistics per player (read by the matchmaker)."""
        return self._pfsp

    def report_match_result(self, result: MatchResult) -> None:
        """Keep the result; update the ratings and the PFSP statistics of its layout."""
        self._match_results.append(result)
        self._ratings.update(result)
        self._pfsp.update(result)

    def ratings_snapshot(self) -> dict:
        """``RatingBook.snapshot()`` with each layout's PFSP table under ``"pfsp"`` (JSON-serializable)."""
        snap = self._ratings.snapshot()
        pfsp = self._pfsp.snapshot()
        for layout, table in snap.items():
            table["pfsp"] = pfsp.get(layout, {})
        return snap
```

`src/colosseum/metrics/aggregator.py`:
```python
OPPONENT_TYPES = ("latest", "past", "arena", "anchor")
```
```python
def opponent_type(result: MatchResult, seat: SeatResult) -> str | None:
    """Kind of opposition a seat met; teammates are not opponents.

    'anchor' (a seat of another team is played by a scripted or frozen agent), 'arena' (a seat of another
    team plays another agent), 'past' (one plays a snapshot of the seat's agent), 'latest' (all play the
    agent's latest weights), or None (no other team: solo or cooperative layouts).
    """
    opponents = [s for s in result.seats if s.team != seat.team]
    if not opponents:
        return None
    if any(s.network_id == FIXED_NETWORK_ID for s in opponents):
        return "anchor"
    if any(s.agent_id != seat.agent_id for s in opponents):
        return "arena"
    if any(s.network_id != LATEST_NETWORK_ID for s in opponents):
        return "past"
    return "latest"
```
(import `FIXED_NETWORK_ID` next to `LATEST_NETWORK_ID`.)

`src/colosseum/metrics/hub.py`: add
```python
def _ratings_row(ratings: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ratings for WandB without the PFSP tables: one column per (owner, snapshot) would grow without bound;
    they stay in metrics.jsonl and ratings.json."""
    return {layout: {k: v for k, v in table.items() if k != "pfsp"} for layout, table in ratings.items()}
```
and in `maybe_tick` replace `row.update(flatten("ratings", ratings))` by `row.update(flatten("ratings", _ratings_row(ratings)))`.

Test edits:
- `tests/unit/test_sp2_ratings.py`: `MemberPair("cross", "a", "b", 1.0, 1.0, "player", "player")` → `MemberPair("cross", "a", "b", 1.0, 1.0, "player", "player", "latest", "latest")`; the two `past` literals of `test_latest_vs_own_checkpoint_is_a_past_pair_from_the_latest_side` gain `"latest", "ckpt_v3"` (wrap the `latest_first` assertion after `== [` to stay within 120 columns); in `test_rating_book_decisive_past_and_mixed_team_pairs` the three literals become `MemberPair("cross", "a", "b", 1.0, third, "player", "player", "latest", "latest")`, `MemberPair("past", "a", "a", 0.0, third, "player", "player", "latest", "ckpt_v1")`, `MemberPair("cross", "a", "b", 1.0, third, "player", "player", "ckpt_v1", "latest")`.
- `tests/unit/test_sp2_metrics.py` (`test_episode_aggregator_wdl_from_team_ranks_and_layout_breakdown`): every full W/D/L dict literal gets `"anchor": [0, 0, 0]` as its last key — `{"latest": [1, 2, 3], "past": [1, 0, 0], "arena": [1, 0, 0], "anchor": [0, 0, 0]}`, `{"latest": [1, 0, 3], "past": [0, 0, 0], "arena": [0, 0, 0], "anchor": [0, 0, 0]}`, `{"latest": [0, 2, 0], "past": [1, 0, 0], "arena": [1, 0, 0], "anchor": [0, 0, 0]}`, `{"latest": [0, 0, 0], "past": [0, 0, 0], "arena": [0, 0, 0], "anchor": [0, 0, 0]}`.
- `tests/integration/test_sp2_metrics_outputs.py`: `set(episodes[0]["wdl"]) == {"latest", "past", "arena", "anchor"}` and, after the `ratings` key check, `assert "pfsp" in ratings["layouts"]["2p"]`.

- [ ] **Step 4: Run the new tests and the touched SP2 tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sp3_pfsp_stats.py tests/unit/test_sp2_ratings.py tests/unit/test_sp2_metrics.py tests/unit/test_sp2_coordinator.py tests/integration/test_sp2_metrics_outputs.py -q`
Expected: all pass.

- [ ] **Step 5: Full fast suite + ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` and `.venv/bin/ruff check .`
Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/league src/colosseum/coordinator src/colosseum/metrics tests/unit/test_sp3_pfsp_stats.py \
        tests/unit/test_sp2_ratings.py tests/unit/test_sp2_metrics.py tests/integration/test_sp2_metrics_outputs.py
git commit -m "feat: PFSP statistics per player, scripted and frozen agents as rating entities, anchor opponent type"
git push origin sp3-league
```

---

## Contract notes

**Dry run.** While writing this part, every task's code and tests were applied in order to a scratch copy of the SP2 tree (commit `fc6b621`) with the shown code: each task's focused tests passed, and the full fast suite passed with zero warnings after T1.5 (1340 tests) and after T2.3 (1363 tests), with `ruff check .` clean. Findings of the dry run that are built into the tasks: SP2 pins the message "without an 'agents' section the only agent is 'agent_0'" (kept in T1.1); `game_harness` is importable only from `tests/contract/` (the tests that drive a `RolloutLoop` live there); two SP2 integration tests assumed an empty snapshot pool after a run-dir resume (adapted in T2.1); `cli_runner.TINY` cannot be applied to an SP2 config copy that still has `pool_size` (T0.1's resume test uses `tiny=False`).

### Assumptions about other parts
1. **T3.1 / T4.1 lift T1.1's guard.** `TrainableAgent.matchmaking` / `init` / `kickstart` are rejected ("not supported yet") until T3.1 (matchmaking) and T4.1 (init, kickstart) validate them and add them to `_AGENT_SECTIONS`. Both tasks also add their SP2 knob paths to `LEGACY_OVERRIDE_KEYS` (`matchmaking.mode`, `matchmaking.self_play_ratio`, `matchmaking.latest_prob`, `matchmaking.pfsp_exponent`, `training.kickstart_teacher`, `training.kickstart_lambda`, `training.kickstart_decay_steps`, `training.kickstart_kl`), so `--set` of an old knob keeps working as for `checkpoint.pool_size`.
2. **Old knob + new key is an error also across `--set`.** A test that trains an SP2 config copy must not add new-style keys whose old names the copy contains: `cli_runner.TINY` sets `checkpoint.keep_last` from T2.1 on, so such tests call `run_train(..., tiny=False)` (T0.1 does; the fixture config is tiny already). T3.1 / T4.1 keep `TINY` free of matchmaking / kickstart keys.
3. **T3.2 (matchmaker).** Seats fixed players as `SeatAssignment(agent, FIXED_NETWORK_ID, collect=False, source=...)` and sets `source` on every seat (`SOURCE_OWNER`, or the opposing team core's category). It reads from the coordinator: `_trainable` (rotation order), `_player_roles`, `_env_steps`, `pfsp` (`MatchmakerContext.pfsp_score` = `coordinator.pfsp.score`), and snapshot candidates from `checkpoint_manager.list_checkpoints` (evicted ones disappear by construction; `take_evictions` stays the launcher's). T3.2 replaces the SP2 call `validate_matchmaking(spec, agent_roles, config.matchmaking)` in `core/validation.validate_config` (T1.5 keeps it) and in `distributed.py`.
4. **T3.3.** Builds `PfspStats({aid: <effective matchmaking.pfsp.halflife_games of aid>})` instead of T2.3's `DEFAULT_HALFLIFE_GAMES` map; share metrics read `SeatResult.source`; the mix printout appends to `ValidationReport.lines`.
5. **T4.x (init, teachers).** Paths and frozen names go through `load_frozen(config, agent_id, path, spec)` + `build_frozen_model(...)`. For a `.pt` that must have the student's architecture, pass the STUDENT's (trainable) id as `agent_id`: its effective networks and roles are used; for a frozen agent name, pass that agent's id and `entry.path`. `FrozenSpec.networks_source` prefixes architecture errors, `FrozenSpec.source` names the weights.
6. **T4.5 (DAgger).** Scripted teachers are `BotSpec`s; the worker builds them with `make_bot`, resets them with `bot_rng`, and stores the action returned by `check_bot_action` (already cast to the role's dtypes) as `teacher_action`.
7. **T5.1 (`record`).** Uses `MatchRunner` with `ActRecord.info` and the optional `on_episode_start(env, layout, episode_seed)` hook; player pools as in `evaluate` (`ScriptedPlayer` for scripted names, `load_eval_model` for frozen names and paths). In eval / record pools (`eval._FixedModels`) every seat runs under network `"latest"` with `collect=False`, so `SeatResult.network_id` is `"latest"` there also for scripted seats.
8. **T6.1 / T6.2.** Distributed guards use `config.fixed_agent_ids()` and `CheckpointManager` already applies `keep_last` / `keep_every` in `distributed.py` (T2.1). Demo bots subclass `colosseum.players.ScriptedBot`.

### Contract details pinned by this part
- `ColosseumConfig.agent_ids()` (every agent in config order, the implicit `agent_0` first) and `IMPLICIT_AGENT_ID`; unknown fields of an agent entry are a `ConfigError` naming the kind's fields; a scripted / frozen agent named `agent_0` without any trainable agent is a `ConfigError`.
- `check_bot_action` checks, in order: structure (dict key order free), component kind and shape, `action_space.contains`, `first_illegal_action` under the normalized mask. `PlayerError` messages: `"<context>: agent '<id>': act raised <Type>: <msg>"`, `"...: reset raised ..."`, `"...: creating the bot failed (...)"`, `"...: illegal action: <reason>"`; the context is `"<runner context>env E, seat P, episode step K, layout L"`. In `validate` the same failures are `ConfigError`s with the prefix `"validate, seat P, episode step K, layout L: agent '<id>'"`.
- Bot instances: one per `(agent_id, env, seat)`, created by the player factory at the first reset with that agent at that seat, reset after every env reset where it sits, kept across lineup changes; `bot_rng(episode_seed, seat, agent_id)`; eval prototypes are deep-copied per instance.
- `MatchRunner._check_lineup` requires the pool entry `(agent, "latest")` for latest and snapshot seats and `(agent, "fixed")` for fixed seats; its error keeps the SP2 words "no model for agent '<id>'" plus `(network '<net>')`.
- `CheckpointManager` retention also keeps the snapshot just saved (SP2 invariant); scans never evict; `import_snapshots` skips ids already present and never copies `trainer_state.pt`; the launcher imports the pool only for a run-dir `training.resume_from`, after `_resolve_resume` and before the first lineups.
- `Coordinator.take_evictions()` returns each evicted id once; the launcher keeps per-worker pending evictions (`Launcher._worker_evictions`) until a command carrying them was put; a worker ignores ids it does not hold.
- `PfspStats` ignores owners missing from `halflife_by_agent`; `forget` keeps empty owner tables; `snapshot()` sorts players. `ratings.json` / `ratings` records carry `"pfsp"` per layout; WandB rows do not.
- `ValidationReport.lines` (T1.5): `"agent '<id>' (scripted <class>): roles [...]; played N decisions in layouts [...]"` (or `"has no seat in the enabled layouts [...]"`) and `"agent '<id>' (frozen): <path>, roles [...]"`; `colosseum validate` prints `"  OK: agent '<id>' (<kind>)"` per agent, then the report lines.

### Proposed amendments to `00-overview.md`
1. **`build_frozen_model(config: ColosseumConfig | None, frozen, spec)`** — `config` optional: workers have no config (only `FixedPlayers`), and `FrozenSpec.networks` is complete.
2. **`check_bot_action(...) -> Tree`** (the action cast to the role's dtypes) with a keyword-only `action_spec: ActionSpec | None = None` — env, record and DAgger need the cast action; the runner passes its cached `ActionSpec` instead of rebuilding one per decision.
3. **`FrozenSpec.networks_source: str = ""`** (new last field) — keeps SP2 eval's pinned error prefixes ("Checkpoint <dir>: meta.json networks: ...", "<pt>: networks: ...").
4. **`ColosseumConfig.agent_ids()` and `IMPLICIT_AGENT_ID`** — `resolve_player_roles` ("every agent, config order") needs one definition that includes the implicit `agent_0`.
5. **`CheckpointManager.import_snapshots(src, agent_id, expected_signature=None)`** — carried snapshots obey the strict-resume role-signature rule.
6. **`Coordinator.pfsp`, `Coordinator.player_roles`, `Coordinator.trainable_agents`** (properties) — T3.2's context reads PFSP and players without touching private fields.
7. **`colosseum.core.config.LEGACY_OVERRIDE_KEYS`** — `--set` of a translated old knob (e.g. `checkpoint.pool_size`) passes the schema walk; T3.1 / T4.1 extend it.
8. **`TrainableAgent.matchmaking` / `init` / `kickstart` are rejected until T3.1 / T4.1** — a free dict that is silently ignored would break "no silent fallbacks".
9. **`MemberPair.net_a` / `net_b`, `OPPONENT_TYPES` gains `"anchor"`, `PFSP_MIN_WEIGHT`, `DEFAULT_HALFLIFE_GAMES`, `PFSP_WEIGHTINGS`, `player_name`** — PFSP needs the network ids of member pairs; the rest are shared constants for T3.x.
10. **`src/colosseum/league/__init__.py` is created by T2.3** (docstring only); T3.2 adds the exports listed in the overview.
11. **`evaluate(config, agents: Mapping[str, str | None], ...)`** and **`cli._parse_agent_spec(spec) -> tuple[str, str | None]`** — `None` = a scripted / frozen agent of the config by name (`eval -a greedy`).
12. **`RunSetup.player_roles`, `RunSetup.fixed`, `_worker_main(..., fixed_players=None)`** — how `FixedPlayers` reach workers.
13. **`Launcher._refresh_worker_matches(..., worker_evictions=None)`** and **`Launcher._import_snapshot_pool(coordinator)`** — retry state for evictions; the run-dir pool import.
14. **`colosseum.core.registry.build_network(networks, role)`** and **`colosseum.coordinator.checkpoint_manager.read_checkpoint_meta(ckpt_dir)`** — building a model from a networks section without a whole config; reading roles of a frozen checkpoint dir without loading its weights.
15. **Test file locations:** `tests/contract/test_sp3_fixed_players_wiring.py` and `tests/contract/test_sp3_snapshot_eviction.py` (not `tests/unit/`), because they use `game_harness`.

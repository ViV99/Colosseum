# SP4 Plan — Part 2: Snapshot retention, `colosseum pin`, the tournament (players, engine, schedules, statistics, modes, CLI), pulse

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following blocks of `docs/superpowers/specs/2026-10-16-sp4-selection-design.md` (Russian, the binding authority; read it first):
- **Block 4 (snapshot retention):** the pure retention function (always-kept set, `keep_every`, top-k by LCB in the lineage-contrast scale, protection of new snapshots, the `max_snapshots` cap and its eviction order), `run_id` / `pinned` in `meta.json`, finals of imported runs become ordinary, retention after every live refit, the `validate` printout, the distributed learner's FIFO cap (T4.1); `colosseum pin` (T4.2).
- **Block 3 (tournament and selection):** tournament players and entity names, the shared player loading with `eval` / `record` (SP3 dedup items), the `(agent, network)` player pool, the worker-process engine (T5.1); blocks, admissible rotations, FFA compositions, `score` layouts, seed domains A / B / P, role groups (T5.2); candidate scores, the paired stratified bootstrap, `confirmed`, repeated games, halving, select, several layouts, the report schema (T5.3); the `colosseum tournament` CLI, the tournament dir, resume by whole blocks, snapshot links, the env check, journal writes, `--pin` (T5.4).
- **Block 6 (pulse):** the `evaluation` section, the `pulse` child process, snapshot requests with a bounded backlog, pulse matches in the journal (T6.1).
- **Section 3 criteria** these tasks make testable: 3.2 (retention, tournament, pulse items), 3.8 (SP3 checkpoints in tournaments), parts of 3.3 / 3.4 (the machinery the T8.x measurements and slow tests use).
- **§8 rule changes** owned here: only the final snapshot of the CURRENT run is protected; the `max_snapshots` cap (default 100); `run_id` and `pinned` in `meta.json`; `agent@v<N>` references in the CLI (`tournament --field`, `pin`); `eval` / `record` no longer apply a frozen entry's `.pt` settings to `-a name=path`; `MatchRunner` takes an explicit seed per lineup (the field arrives with T1.1; this part makes `play_lineups` keep it).

**What the part delivers.** After T6.1:
- snapshots are kept by `keep_last` + the current run's final + pinned + `keep_every` + top-k by the live BT lower bound (with protection of new snapshots), capped by `max_snapshots`; retention reruns after every live refit; `colosseum pin` pins and unpins snapshots, also while the run trains;
- `colosseum tournament` compares run snapshots (several `--run` with aliases), `-a` checkpoints and `.pt` files of any architecture and the config's scripted / frozen agents in `gauntlet`, `round-robin`, `halving` and `select` modes on common seeds with seat rotations, in local worker processes, writing whole blocks to the journal, resuming interrupted tournaments, and reporting candidate scores with intervals, the worst opponent, BT ratings, `confirmed` and the indistinguishable finalists (`report.json`, `report.txt`; `report.html` once Part 3's renderer exists);
- with `evaluation.enabled` a `pulse` child evaluates every k-th new snapshot against a fixed field on the same seeds and writes `source = pulse` matches to the journal.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's. Additions are marked `*` in each task's **Interfaces**; every addition, deviation and requirement on Parts 1 and 3 is listed in `## Contract notes` at the end. Accepted contract amendments in the overview win over this part.

**Conventions used in every task.**
- Run every command from the repository root, always as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` / `env -u PYTHONPATH .venv/bin/colosseum ...` (the owner's `~/.bashrc` sets a ROS `PYTHONPATH` that breaks pytest).
- "Full fast suite" = `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, **zero warnings**) followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp4_`; shared test support goes into `tests/game_helpers.py` (no new support modules in this part). Class names in `game_helpers.py` are unique there.
- At most **2 worker processes per test**; tournament workers and the `pulse` process count (a training run with 1 rollout worker + pulse = 2).
- Files only under `tmp_path`; fixtures under `tests/fixtures/` are copied before anything writes into them (`copy_sp3_run`, T0.1).
- CLI integration tests of `train` start processes through `tests/cli_runner.py`; `tournament` / `pin` / `eval` run in-process with `click.testing.CliRunner` (their worker children are spawned and supervised by the command itself).
- Every task ends with a commit on `sp4-selection` (conventional prefix, no attribution or co-author lines) and `git push`.
- Durations and timeouts: `time.monotonic()`; data order by counters (versions, block indices, row ids), never by `ts`.

**Cross-task notes.**
- **Part 1 through the contract only.** This part uses `colosseum.core.types` (T1.1: `Lineup.seed/block/uid`, `MatchResult.uid/source/seed/env_steps/block`, `SeatResult.version_start/version_end`), `MatchRunner(version_fn, source, uid_prefix)` (T1.1), `arena.entities`, `arena.comparisons`, `arena.matchlog` (T1.2, T1.3), `arena.bt` (T2.1), `arena.live` (T2.2), `RunDir.run_id` / `matches_path` (T1.3), and a handful of names the contract lacks; those are requirements **R1–R12** in the Contract notes. If a requirement was met under another name, the implementer uses that name and reports it.
- **Provisional constants.** Tournament defaults (`DEFAULT_BLOCKS = 50`, `DEFAULT_HALVING_BLOCKS = 8`, `DEFAULT_CONFIRM_BLOCKS = 100`, `DEFAULT_FINALISTS = 3`) and `PAIRED_BOOTSTRAP_SAMPLES = 2000` are provisional; T8.2 measures them on the bot ladder and the controller fixes them as rulings. Their docstrings say so.
- **Import direction.** `colosseum.arena.tournament.*` imports `colosseum.eval` (`play_lineups`, `wilson_interval`, the shared loaders); `colosseum.eval` never imports `colosseum.arena`. `colosseum.arena.pulse` imports `arena.tournament.schedule` and `colosseum.eval`. `colosseum.arena.retention` imports nothing from torch and nothing from `arena.bt` at module level (duck-typed fits), so the distributed learner and SP5's hub can call it.
- **Seats of tournament players.** A run snapshot plays as `SeatAssignment(agent_id=<[alias/]agent>, network_id="ckpt_v<N>")`, every other player as `(<name>, "fixed")`. The journal therefore computes exactly the tournament entity names of spec block 2 (`main@v1200`, `b/main@v3`, `greedy`, `prev#1a2b3c4d`) from `entity_key(agent_id, rating_version)`, and `SeatResult.version_start == N` for snapshots (T1.1's rule "`ckpt_vN`: N").
- **Block ids.** `"<prefix>/<stage>/<layout>/<candidate>/<composition>/<index>"`; a match uid is `"<block id>/r<rotation>"`. The prefix is the tournament name (a path component, no `/`) or `pulse-<run id 12 hex>`. Part 1 keys `MatchLog.comparison_table(tournament=T)` and `MatchLogReader.blocks(T)` on that prefix (R4).
- **Shared helpers this part adds to `tests/game_helpers.py`:** `CrashBot`, `PlayerErrorBot`, `write_fake_run` (T5.1; `biases=` and `biased_model` in T5.4), `SyntheticSource`, `SyntheticPlayer`, `synthetic_players` (T5.2), `FakeArena` (T5.3), `PulseFailBot` (T6.1).

---

### Task T4.1: Snapshot retention (`arena.retention`), `CheckpointManager` integration, top-k after refits, the cap, `validate` printout, distributed FIFO cap

Spec block 4 (after the review fixes, §8 «Уточнения» 7, 12, 13) and §8 «Изменения правил»: the keep/evict decision becomes the pure function `arena.retention.retain(entries, scores, policy)`. `CheckpointManager` builds the entries from its index (re-reading `pinned` from disk, so `colosseum pin` works while a run trains), asks a scores provider for the snapshots' live-BT lower bounds when `keep_top_k > 0`, applies the decision (index first, then `on_evict`: SP3 ruling #24) and logs the decision's warnings once. `meta.json` of every new snapshot gets `run_id` and `pinned: false`; the final snapshot is protected only when its `run_id` is the current run's, so imported finals (and SP3 snapshots, which have no `run_id`) become ordinary. A manager built without a `run_id` (tests, old callers) keeps the SP3 rule (every final protected), which keeps every SP3 storage test valid.

After every live refit (a new `LiveRatings.latest().version`) the main loop reruns retention for every trainable agent (only with `keep_top_k > 0`, where scores matter); evictions reach the coordinator (PFSP `forget`, worker unloads: SP3 #50, #24) and the journal (`MatchLogWriter.note_evicted`: matches of the evicted entity are compacted on the next pass). The distributed learner gets the cap with FIFO order (it has no ratings) and refuses `keep_top_k > 0` (SP5).

The order of the rules is the spec's:

1. always: newest `keep_last`; the current run's final; pinned; the snapshot just saved;
2. `keep_every`: version multiple of `interval * keep_every`;
3. top-k (`keep_top_k > 0`): the `keep_top_k` rated snapshots with the highest LCB (quantile `top_k_quantile` of the live Laplace distribution, layout-weighted on every draw, lineage-contrast scale);
4. new (`keep_top_k > 0`): outside `keep_last` with fewer than `top_k_min_games` rated games (self-play excluded; a bucket `v<B>` counts latest games of its bucket against other entities: `BTFit.games` already does this, T2.1);

then the cap `max_snapshots`: keep_every-only (FIFO when `keep_top_k == 0`, else unrated first, then ascending LCB), then new-only (oldest first), then top-k (ascending LCB); rule 1 alone above the cap → WARNING once, rule 1 stays.

**Files:**
- Create: `src/colosseum/arena/retention.py`
- Modify: `src/colosseum/core/config.py` (`CheckpointConfig`: `keep_top_k`, `top_k_quantile`, `top_k_min_games`, `max_snapshots` and their checks; `ColosseumConfig._check_sp4_dependencies`)
- Modify: `src/colosseum/coordinator/checkpoint_manager.py` (`retain` integration, `run_id` / `pinned`, `pin`, `pinned`, `reapply_retention`, `set_pinned`, `read_snapshots`, `link_or_copy`)
- Modify: `src/colosseum/coordinator/coordinator.py` (`run_id`, scores provider, `set_ratings_source`, `add_evict_listener`)
- Modify: `src/colosseum/launcher.py` (`_build_coordinator` passes the run id; `_wire_retention`; `_reapply_retention` in the monitor loop)
- Modify: `src/colosseum/core/validation.py` (retention printout; the pinned-count WARNING)
- Modify: `src/colosseum/distributed.py` (`distributed_checkpoint_manager(..., run_id=None)` with `max_snapshots`; `keep_top_k > 0` refused)
- Modify: `scripts/bench_throughput.py::_make_config` (the four new checkpoint keys, explicit)
- Modify (the final-snapshot rule change, Step 3): `tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr`, `tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature`, `tests/integration/test_sp3_sp2_resume.py::test_training_resumes_from_the_sp2_run_dir`, and any T0.1 test of Part 1 that asserts an imported final snapshot survives a run-dir resume (Step 1 greps for it)
- Test: `tests/unit/test_sp4_retention.py`, `tests/unit/test_sp4_snapshot_store.py`, `tests/integration/test_sp4_retention_runs.py`

**Interfaces:**
- Consumes: `RunDir.run_id` (T1.3, R1); `LiveRatings.latest() -> RatingsSnapshot | None`, `RatingsSnapshot.version / fits / uncertainty / scores` (T2.2); `BTFit.rating / games` and `Uncertainty.lcb` (T2.1) plus `Uncertainty.draws` and `ScoreSummary.by_entity` (R5, R6); `MatchLogWriter.note_evicted(entities)` (T1.3); `entities.entity_key`, `entities.snapshot_version` (T1.2); the launcher attributes `self._match_log_writer` (T1.3) and `self._live_ratings` (T2.2) (R2); `match_log.enabled` (T1.1).
- Produces (contract T4.1, additions marked *):
  - `arena.retention`: `SnapshotInfo`, `SnapshotScore`, `RetentionPolicy`, `RetentionDecision`, `retain(entries, scores, policy, *, protect=None)`; *`combine_scores(agent_id, *, fits, uncertainty, scores, layout_weights, quantile) -> dict[str, SnapshotScore]`; *`snapshot_scores(snapshot, agent_id, *, layout_weights, quantile)`; *`describe_retention(checkpoint) -> str`;
  - `CheckpointConfig.keep_top_k / top_k_quantile / top_k_min_games / max_snapshots`;
  - `CheckpointManager(..., run_id=None, keep_top_k=0, top_k_min_games=200, max_snapshots=None, scores_fn=None)`; `pin(agent_id, checkpoint_id, pinned=True)`, `pinned(agent_id)`, `reapply_retention(agent_id) -> list[str]`;
  - *`checkpoint_manager.set_pinned(ckpt_dir, pinned) -> dict` (atomic `meta.json` rewrite; read-only callers' tool: `colosseum pin`, `tournament --pin`); *`read_snapshots(agent_dir) -> list[CheckpointInfo]` (read-only listing, never cleans); *`link_or_copy(src, dst)` (the former `_link_or_copy`, now public; the old name stays an alias);
  - *`Coordinator(..., run_id=None)`; *`Coordinator.set_ratings_source(fn)`; *`Coordinator.add_evict_listener(fn)`; *`Coordinator.snapshot_scores(agent_id)`;
  - *`Launcher._wire_retention()`, *`Launcher._reapply_retention()`;
  - *`distributed_checkpoint_manager(config, base_dir, run_id=None)`.

- [ ] **Step 1: Check the ground and the names**

```bash
ls src/colosseum/arena/ && grep -n "class LiveRatings\|def latest\|class RatingsSnapshot" src/colosseum/arena/live.py
grep -n "class Uncertainty\|draws\|class ScoreSummary\|by_entity" src/colosseum/arena/bt.py
grep -n "_match_log_writer\|_live_ratings\|def note_evicted" src/colosseum/launcher.py src/colosseum/arena/matchlog.py
grep -n "def run_id\|run_id" src/colosseum/core/run_dir.py
grep -n "keep_top_k\|max_snapshots" src/colosseum/core/config.py scripts/bench_throughput.py
grep -rn "final" tests/integration/test_sp4_*.py tests/unit/test_sp4_*.py 2>/dev/null | grep -i "import\|carried\|resume" | head
```

Expected: `arena/` has `entities.py`, `comparisons.py`, `matchlog.py`, `bt.py`, `live.py`; `LiveRatings.latest`, `RatingsSnapshot`, `Uncertainty`, `ScoreSummary`, `note_evicted` exist; `RunDir.run_id` exists. If `Uncertainty.draws`, `ScoreSummary.by_entity`, `Launcher._match_log_writer` or `Launcher._live_ratings` carry other names, use those names everywhere below and report them (R2, R5, R6). If an earlier task added `keep_top_k` with a "not supported yet" rejection, this task removes that rejection (overview cross-part note). The last grep lists T0.1/T1.x tests that assert an imported final survives a resume: Step 3 updates them like the SP2 ones.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_sp4_retention.py`:

```python
"""Snapshot retention, the pure function (SP4 T4.1, spec block 4): table tests of every rule, the cap order,
and the combination of per-layout ratings into one score per snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from colosseum.arena.retention import (
    RetentionPolicy,
    SnapshotInfo,
    SnapshotScore,
    combine_scores,
    describe_retention,
    retain,
)
from colosseum.core.config import CheckpointConfig

RUN = "run-now"


def snaps(*versions, final=(), pinned=(), run_id=RUN) -> list[SnapshotInfo]:
    return [SnapshotInfo(f"ckpt_v{v}", v, v in final, run_id, v in pinned) for v in versions]


def policy(**kw) -> RetentionPolicy:
    base = dict(keep_last=2, keep_every_period=0, keep_top_k=0, top_k_min_games=0, max_snapshots=None,
                current_run_id=RUN)
    return RetentionPolicy(**{**base, **kw})


def kept(decision) -> list[int]:
    return [int(c.removeprefix("ckpt_v")) for c in decision.keep]


def scored(**lcbs) -> dict[str, SnapshotScore]:
    """``scored(v10=(lcb, games))``."""
    return {f"ckpt_{k}": SnapshotScore(lcb, games) for k, (lcb, games) in lcbs.items()}


def test_keep_last_keep_every_and_the_snapshot_just_saved():
    d = retain(snaps(10, 20, 30, 40, 50, 60), {}, policy(keep_last=2, keep_every_period=20), protect="ckpt_v10")
    assert kept(d) == [10, 20, 40, 50, 60]
    assert d.evict == ["ckpt_v30"] and d.reasons["ckpt_v30"].startswith("evicted")
    assert "just saved" in d.reasons["ckpt_v10"] and "keep_every" in d.reasons["ckpt_v20"]
    assert d.warnings == []


def test_only_the_final_snapshot_of_the_current_run_is_protected():
    entries = [SnapshotInfo("ckpt_v7", 7, True, "parent", False), SnapshotInfo("ckpt_v9", 9, True, None, False),
               SnapshotInfo("ckpt_v11", 11, True, RUN, False), *snaps(20, 30)]
    assert kept(retain(entries, {}, policy())) == [11, 20, 30]
    # a manager without a run id (tests, SP3 callers) keeps the SP3 rule: every final snapshot is protected
    assert kept(retain(entries, {}, policy(current_run_id=None))) == [7, 9, 11, 20, 30]


def test_pinned_snapshots_are_never_evicted_not_even_by_the_cap():
    d = retain(snaps(10, 20, 30, 40, pinned=(10, 20)), {}, policy(keep_last=2, max_snapshots=3))
    assert kept(d) == [10, 20, 30, 40]
    assert len(d.warnings) == 1 and "max_snapshots=3" in d.warnings[0]


def test_top_k_by_lcb_among_all_rated_snapshots_ties_go_to_the_newer():
    scores = scored(v10=(5.0, 500), v20=(9.0, 500), v30=(7.0, 500), v40=(7.0, 500), v50=(1.0, 500), v60=(0.0, 500))
    d = retain(snaps(10, 20, 30, 40, 50, 60), scores, policy(keep_last=1, keep_top_k=2, top_k_min_games=100))
    assert kept(d) == [20, 40, 60]
    assert "top_k" in d.reasons["ckpt_v20"]


def test_new_snapshots_are_protected_until_they_have_enough_rated_games():
    scores = scored(v10=(9.0, 500), v20=(1.0, 50), v30=(0.5, 500))
    d = retain(snaps(10, 20, 30, 40, 50), scores, policy(keep_last=2, keep_top_k=1, top_k_min_games=100))
    assert kept(d) == [10, 20, 40, 50]          # 20: 50 < 100 games; 30: rated, not top-1
    assert "new" in d.reasons["ckpt_v20"]
    unrated = retain(snaps(10, 20, 30), {}, policy(keep_last=1, keep_top_k=1, top_k_min_games=1))
    assert kept(unrated) == [10, 20, 30]        # no scores: everything is new
    off = retain(snaps(10, 20, 30), {}, policy(keep_last=1, keep_top_k=1, top_k_min_games=0))
    assert kept(off) == [30]


def test_the_cap_without_top_k_evicts_keep_every_snapshots_oldest_first():
    d = retain(snaps(10, 20, 30, 40, 50, 60), {}, policy(keep_last=2, keep_every_period=10, max_snapshots=4))
    assert kept(d) == [30, 40, 50, 60]
    assert d.reasons["ckpt_v10"] == "evicted: over max_snapshots=4 (keep_every)"


def test_the_cap_with_top_k_order_keep_every_new_top_k():
    entries = snaps(10, 20, 30, 40, 50, 60, 70)
    scores = scored(v10=(3.0, 500), v20=(8.0, 500), v30=(1.0, 10), v40=(2.0, 500), v50=(None, 500))
    pol = policy(keep_last=1, keep_every_period=10, keep_top_k=2, top_k_min_games=100)
    # top-k: 20 (8.0), 10 (3.0); new (< 100 games, outside keep_last): 30, 60 (no score); keep_every-only: 40, 50.
    # The cap evicts the unrated keep_every-only 50, then 40 (by lcb), then the new 30, 60 (oldest first), then
    # the top-k 10 (lower lcb) and 20; 70 (keep_last) stays.
    order = []
    for cap in (6, 5, 4, 3, 2, 1):
        d = retain(entries, scores, policy(**{**pol.__dict__, "max_snapshots": cap}))
        order.append(kept(d))
    assert order == [[10, 20, 30, 40, 60, 70], [10, 20, 30, 60, 70], [10, 20, 60, 70], [10, 20, 70], [20, 70], [70]]


def test_bad_policies_and_duplicate_ids_are_value_errors():
    with pytest.raises(ValueError):
        retain(snaps(1), {}, policy(keep_last=0))
    with pytest.raises(ValueError):
        retain(snaps(1), {}, policy(max_snapshots=0))
    with pytest.raises(ValueError, match="duplicate"):
        retain(snaps(1, 1), {}, policy())


def test_entries_in_any_order_give_ids_by_version():
    d = retain(list(reversed(snaps(10, 20, 30))), {}, policy(keep_last=2))
    assert d.keep == ["ckpt_v20", "ckpt_v30"] and d.evict == ["ckpt_v10"]


@dataclass
class _Fit:            # the BTFit fields combine_scores reads
    rating: dict
    games: dict


@dataclass
class _Unc:            # the Uncertainty fields combine_scores reads
    lcb: dict
    draws: dict = field(default_factory=dict)


@dataclass
class _Scores:         # the ScoreSummary field combine_scores reads
    by_entity: dict


def test_combine_scores_weights_the_draws_of_every_layout():
    rng = np.random.default_rng(0)
    a, b = rng.normal(100.0, 10.0, 4000), rng.normal(0.0, 10.0, 4000)
    fits = {"2p": _Fit({"main@v10": 100.0, "main@v20": 1.0, "greedy": 0.0}, {"main@v10": 300, "main@v20": 7}),
            "4p": _Fit({"main@v10": 0.0}, {"main@v10": 50})}
    unc = {"2p": _Unc({"main@v10": 87.0, "main@v20": -5.0}, {"main@v10": a}), "4p": _Unc({"main@v10": -13.0},
                                                                                       {"main@v10": b})}
    out = combine_scores("main", fits=fits, uncertainty=unc, scores={}, layout_weights={"2p": 3.0, "4p": 1.0},
                         quantile=0.1)
    assert set(out) == {"ckpt_v10", "ckpt_v20"} and "greedy" not in out
    assert out["ckpt_v10"].games == 350 and out["ckpt_v20"] == SnapshotScore(-5.0, 7)   # one layout: its lcb
    expected = np.quantile(0.75 * a + 0.25 * b, 0.1)
    assert out["ckpt_v10"].lcb == pytest.approx(expected)
    other = combine_scores("other", fits=fits, uncertainty=unc, scores={}, layout_weights={}, quantile=0.1)
    assert other == {}


def test_combine_scores_of_score_layouts_uses_mean_and_standard_error():
    scores = {"coop2": _Scores({"main@v10": (100, 2.0, 0.1), "helper": (5, 1.0, 1.0)})}
    out = combine_scores("main", fits={}, uncertainty={}, scores=scores, layout_weights={"coop2": 1.0},
                         quantile=0.1)
    assert out["ckpt_v10"].games == 100
    assert out["ckpt_v10"].lcb == pytest.approx(2.0 - 1.2815515655446004 * 0.1)


def test_describe_retention_names_every_rule():
    text = describe_retention(CheckpointConfig(interval=1000, keep_last=20, keep_every=10))
    assert "newest 20" in text and "every 10000th version" in text and "top-k off" in text and "at most 100" in text
    top = describe_retention(CheckpointConfig(keep_top_k=5, top_k_min_games=200, max_snapshots=None))
    assert "top 5" in top and "200 rated games" in top and "no cap" in top
```

Create `tests/unit/test_sp4_snapshot_store.py`:

```python
"""Snapshot storage with SP4 retention (T4.1, spec block 4): run_id / pinned in meta.json, the current run's
final, pins written on disk while a run trains, top-k through the scores provider, reapply after a refit,
the cap, config checks, the validate printout, the coordinator wiring, the distributed FIFO cap."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pytest
import yaml

from colosseum.arena.retention import SnapshotScore
from colosseum.coordinator.checkpoint_manager import CheckpointManager, read_snapshots, set_pinned
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.distributed import check_distributed_scope, distributed_checkpoint_manager
from game_helpers import make_coordinator, make_test_config


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2,), value, np.float32)}


def ids(manager, agent_id="a") -> list[str]:
    return [c.checkpoint_id for c in manager.list_checkpoints(agent_id)]


def meta(manager, checkpoint_id, agent_id="a") -> dict:
    return json.loads((manager.base_dir / agent_id / checkpoint_id / "meta.json").read_text())


def test_new_snapshots_carry_run_id_and_pinned_and_imported_finals_are_ordinary(tmp_path):
    old = CheckpointManager(tmp_path / "old", keep_last=5, run_id="r1")
    old.save("a", 7, sd(7), meta_extra={"final": True})
    old.save("a", 10, sd(10))
    assert meta(old, "ckpt_v7")["run_id"] == "r1" and meta(old, "ckpt_v7")["pinned"] is False
    new = CheckpointManager(tmp_path / "new", keep_last=1, run_id="r2")
    assert new.import_snapshots(tmp_path / "old", "a") == ["ckpt_v10"]       # r1's final is not protected in r2
    new.save("a", 20, sd(20), meta_extra={"final": True})
    new.save("a", 30, sd(30))
    assert ids(new) == ["ckpt_v20", "ckpt_v30"]                              # r2's own final is
    assert meta(new, "ckpt_v20")["run_id"] == "r2"
    legacy = CheckpointManager(tmp_path / "legacy", keep_last=1)             # no run id: the SP3 rule
    legacy.save("a", 7, sd(7), meta_extra={"final": True})
    legacy.save("a", 8, sd(8))
    assert ids(legacy) == ["ckpt_v7", "ckpt_v8"] and "run_id" not in meta(legacy, "ckpt_v7")


def test_a_pin_written_on_disk_by_another_process_is_honoured(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=1, run_id="r")
    mgr.save("a", 10, sd(10))
    set_pinned(tmp_path / "a" / "ckpt_v10", True)                            # what `colosseum pin` does
    mgr.save("a", 20, sd(20))
    mgr.save("a", 30, sd(30))
    assert ids(mgr) == ["ckpt_v10", "ckpt_v30"] and mgr.pinned("a") == ["ckpt_v10"]
    mgr.pin("a", "ckpt_v10", pinned=False)
    assert meta(mgr, "ckpt_v10")["pinned"] is False
    mgr.save("a", 40, sd(40))
    assert ids(mgr) == ["ckpt_v40"]
    with pytest.raises(ConfigError, match="ckpt_v99"):
        mgr.pin("a", "ckpt_v99")


def test_set_pinned_replaces_the_file_so_a_hard_linked_source_is_untouched(tmp_path):
    old = CheckpointManager(tmp_path / "old", keep_last=5, run_id="r1")
    old.save("a", 10, sd(10))
    new = CheckpointManager(tmp_path / "new", keep_last=5, run_id="r2")
    new.import_snapshots(tmp_path / "old", "a")
    new.pin("a", "ckpt_v10")
    assert meta(new, "ckpt_v10")["pinned"] is True and meta(old, "ckpt_v10")["pinned"] is False
    again = CheckpointManager(tmp_path / "third", keep_last=1, run_id="r3")
    again.import_snapshots(tmp_path / "new", "a")                            # the pin survives a resume
    again.save("a", 20, sd(20))
    again.save("a", 30, sd(30))
    assert ids(again) == ["ckpt_v10", "ckpt_v30"]


def test_top_k_uses_the_scores_provider_and_reapply_evicts_after_a_refit(tmp_path):
    scores = {"ckpt_v10": SnapshotScore(1.0, 500), "ckpt_v20": SnapshotScore(5.0, 500),
              "ckpt_v30": SnapshotScore(3.0, 500)}
    evicted = []
    mgr = CheckpointManager(tmp_path, keep_last=1, keep_top_k=1, top_k_min_games=100, run_id="r",
                            scores_fn=lambda agent: dict(scores),
                            on_evict=lambda a, c: evicted.append((a, c, ids(mgr))))
    for version in (10, 20, 30, 40):
        mgr.save("a", version, sd(version))
    assert ids(mgr) == ["ckpt_v20", "ckpt_v40"]
    scores.update({"ckpt_v20": SnapshotScore(0.0, 500), "ckpt_v40": SnapshotScore(9.0, 500)})
    evicted.clear()
    assert mgr.reapply_retention("a") == ["ckpt_v20"]
    assert ids(mgr) == ["ckpt_v40"]
    assert evicted == [("a", "ckpt_v20", ["ckpt_v40"])]                      # the index is updated first (SP3 #24)
    assert mgr.reapply_retention("a") == [] and mgr.reapply_retention("nobody") == []


def test_max_snapshots_caps_keep_every_snapshots_oldest_first_and_warns_once(tmp_path, caplog):
    mgr = CheckpointManager(tmp_path, keep_last=2, keep_every=1, interval=10, max_snapshots=3, run_id="r")
    for version in range(10, 61, 10):
        mgr.save("a", version, sd(version))
    assert ids(mgr) == ["ckpt_v40", "ckpt_v50", "ckpt_v60"]
    pinned = CheckpointManager(tmp_path / "p", keep_last=2, max_snapshots=2, run_id="r")
    with caplog.at_level(logging.WARNING, logger="colosseum.coordinator.checkpoint_manager"):
        for version in (10, 20, 30, 40):
            pinned.save("a", version, sd(version))
            pinned.pin("a", f"ckpt_v{version}")
    assert ids(pinned) == ["ckpt_v10", "ckpt_v20", "ckpt_v30", "ckpt_v40"]
    assert len([r for r in caplog.records if "exceed checkpoint.max_snapshots" in r.getMessage()]) == 1


def test_read_snapshots_lists_without_cleaning(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last=5)
    mgr.save("a", 10, sd(10))
    leftover = tmp_path / "a" / ".tmp-ckpt_v20-0123abcd"
    leftover.mkdir()
    assert [c.checkpoint_id for c in read_snapshots(tmp_path / "a")] == ["ckpt_v10"]
    assert leftover.is_dir() and read_snapshots(tmp_path / "missing") == []


def _write(tmp_path, **sections) -> str:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_checkpoint_config_defaults_and_checks(tmp_path):
    cfg = load_config(_write(tmp_path))
    c = cfg.checkpoint
    assert (c.keep_top_k, c.top_k_quantile, c.top_k_min_games, c.max_snapshots) == (0, 0.1, 200, 100)
    assert load_config(_write(tmp_path, checkpoint={"max_snapshots": None})).checkpoint.max_snapshots is None
    with pytest.raises(ConfigError, match="max_snapshots"):
        load_config(_write(tmp_path, checkpoint={"keep_last": 20, "max_snapshots": 5}))
    with pytest.raises(ConfigError, match="match_log"):
        load_config(_write(tmp_path, checkpoint={"keep_top_k": 3}, match_log={"enabled": False}))
    for bad in ({"top_k_quantile": 0.5}, {"top_k_quantile": 0.0}, {"keep_top_k": -1}, {"top_k_min_games": -1},
                {"max_snapshots": 0}):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, checkpoint=bad))
    assert load_config(_write(tmp_path, checkpoint={"keep_top_k": 3})).checkpoint.keep_top_k == 3   # log on
    default = ColosseumConfig.model_validate(make_test_config("turns").model_dump(by_alias=True))
    assert default.checkpoint.max_snapshots == 100


def test_validate_prints_the_retention_rule_and_warns_about_pins_over_the_cap(tmp_path, caplog):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 2, "max_snapshots": 3})
    lines = validate_config(cfg).lines
    assert sum(1 for line in lines if line.startswith("agent 'a': snapshots") or
               line.startswith("agent 'b': snapshots")) == 2
    old = tmp_path / "old"
    mgr = CheckpointManager(old / "checkpoints", keep_last=5, run_id="r1")
    for version in (10, 20):
        mgr.save("a", version, sd(version))
        mgr.pin("a", f"ckpt_v{version}")
    resumed = make_test_config("turns", agents={"a": {}, "b": {}}, checkpoint={"keep_last": 2, "max_snapshots": 3},
                               training={"resume_from": str(old)})
    with caplog.at_level(logging.WARNING, logger="colosseum.core.validation"):
        warnings = [line for line in validate_config(resumed).lines if line.startswith("WARNING")]
    assert len(warnings) == 1 and "agent 'a'" in warnings[0] and "2 pinned" in warnings[0]
    assert any("2 pinned" in r.getMessage() for r in caplog.records)


@dataclass
class _Fit:
    rating: dict
    games: dict


@dataclass
class _Unc:
    lcb: dict
    draws: dict = field(default_factory=dict)


@dataclass
class _Snapshot:       # the RatingsSnapshot fields snapshot_scores reads
    version: int
    fits: dict
    uncertainty: dict
    scores: dict = field(default_factory=dict)
    env_steps: int = 0


def test_the_coordinator_feeds_top_k_from_the_live_ratings_and_tells_listeners(tmp_path):
    cfg = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 1, "keep_top_k": 1,
                                                "top_k_min_games": 0, "keep_every": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    seen = []
    coord.add_evict_listener(lambda agent, ckpt: seen.append((agent, ckpt)))
    snap = _Snapshot(1, {"2p": _Fit({"agent_0@v10": 5.0, "agent_0@v20": 1.0}, {"agent_0@v10": 9, "agent_0@v20": 9})},
                     {"2p": _Unc({"agent_0@v10": 4.0, "agent_0@v20": 0.0})})
    coord.set_ratings_source(lambda: snap)
    assert coord.snapshot_scores("agent_0") == {"ckpt_v10": SnapshotScore(4.0, 9), "ckpt_v20": SnapshotScore(0.0, 9)}
    for version in (10, 20, 30):
        coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": version, "model_state": sd(version),
                                       "trainer_state_bytes": b"opt"})
    assert ids(coord.checkpoint_manager, "agent_0") == ["ckpt_v10", "ckpt_v30"]
    assert seen == [("agent_0", "ckpt_v20")]
    coord.set_ratings_source(lambda: None)
    assert coord.snapshot_scores("agent_0") == {}


def test_the_distributed_learner_caps_its_pool_fifo_and_refuses_top_k(tmp_path):
    config = make_test_config("turns", checkpoint={"interval": 10, "keep_last": 1, "keep_every": 1,
                                                   "max_snapshots": 3})
    manager = distributed_checkpoint_manager(config, tmp_path / "checkpoints", run_id="r")
    for version in range(10, 71, 10):
        manager.save("agent_0", version, sd(version))
    assert ids(manager, "agent_0") == ["ckpt_v50", "ckpt_v60", "ckpt_v70"]
    with pytest.raises(ConfigError, match="keep_top_k"):
        check_distributed_scope(make_test_config("turns", checkpoint={"keep_top_k": 2}))
```

Note on the validate test: `training.resume_from` is the run dir `old`, where only agent `a` has snapshots; `validate_config` decides resume sources from the file system only (`learner.factory.resume_source_of`), so the snapshots need no role signature. If a Part 1 task made `validate_config` read resumed checkpoints, save them with `meta_extra=` the coordinator's roles / signature (`make_coordinator(...).save_checkpoint_payload`).

Create `tests/integration/test_sp4_retention_runs.py`:

```python
"""Retention in a real run (SP4 T4.1): top-k with live ratings and the cap; every snapshot names its run."""
from __future__ import annotations

import json

from cli_runner import TTT_CONFIG, run_train
from colosseum.core.run_dir import RunDir


def test_a_top_k_run_stays_under_max_snapshots_and_writes_run_id(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="topk", overrides={
        "checkpoint.interval": "5", "checkpoint.keep_last": "2", "checkpoint.keep_every": "0",
        "checkpoint.keep_top_k": "2", "checkpoint.top_k_min_games": "0", "checkpoint.max_snapshots": "4",
        "ratings.refit_interval_sec": "1.0"})
    assert run.returncode == 0, run.stderr[-3000:]
    agent_dir = run.root / "checkpoints" / "agent_0"
    snapshots = sorted(agent_dir.glob("ckpt_v*"))
    assert 1 <= len(snapshots) <= 4, [p.name for p in snapshots]
    run_id = RunDir.open(run.root).run_id
    metas = [json.loads((p / "meta.json").read_text()) for p in snapshots]
    assert all(m["run_id"] == run_id and m["pinned"] is False for m in metas)
    log = run.log("main")
    assert "Traceback" not in log and "ERROR" not in log
```

- [ ] **Step 3: Update the SP2/SP3 tests that assumed every final survives a run-dir resume**

The rule change (spec §8: only the current run's final is protected) makes the first run's final an ordinary snapshot in the resumed run. Edit:

`tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr` — replace
```python
    carried = [v for v in versions if v <= first_final]
    assert first_final in carried and set(carried) <= set(checkpoint_versions(first, "agent_0")), (carried, versions)
```
by
```python
    carried = [v for v in versions if v <= first_final]
    assert set(carried) <= set(checkpoint_versions(first, "agent_0")), (carried, versions)
    # SP4 T4.1: the first run's final is an ordinary snapshot here; only this run's final is protected
    second_final = json.loads((second.root / "checkpoints" / "agent_0" / f"ckpt_v{versions[-1]}" / "meta.json")
                              .read_text())
    assert second_final["final"] is True and second_final["run_id"] != json.loads(
        (first.root / "checkpoints" / "agent_0" / f"ckpt_v{first_final}" / "meta.json").read_text())["run_id"]
```
(`json` is already imported there; if not, add `import json`.)

`tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature` — replace
```python
    resumed = [m["policy_version"] for m in metas(second, "agent_0")]
    # SP3 T2.1: the first run's stored snapshots are carried into the new pool (its final one is never evicted)
    assert version in resumed and max(resumed) > version, (version, resumed)
```
by
```python
    resumed = [m["policy_version"] for m in metas(second, "agent_0")]
    # SP3 T2.1: the first run's stored snapshots are carried into the new pool; SP4 T4.1: its final one is an
    # ordinary snapshot there (only the current run's final is protected), so it may have been evicted
    first_versions = {m["policy_version"] for m in metas(first, "agent_0")}
    assert max(resumed) > version and {v for v in resumed if v <= version} <= first_versions, (version, resumed)
    assert "snapshot pool carried over" in second.log("main")
```

`tests/integration/test_sp3_sp2_resume.py::test_training_resumes_from_the_sp2_run_dir` — replace
```python
    versions = _versions(agent_dir)
    assert versions[0] == SP2_CHECKPOINT_VERSION and len(versions) >= 2   # the SP2 snapshot joined the new pool
    imported = agent_dir / f"ckpt_v{SP2_CHECKPOINT_VERSION}"
    assert sorted(p.name for p in imported.iterdir()) == ["meta.json", "model.pt"]   # final: kept; no trainer state
```
by
```python
    versions = _versions(agent_dir)
    assert versions and max(versions) > SP2_CHECKPOINT_VERSION
    # the SP2 snapshot joined the new pool (the log), as an ordinary snapshot: it has no run_id, so SP4 T4.1
    # treats it as imported and retention may evict it like any other
    assert f"Imported 1 snapshots of agent_0" in log
    imported = agent_dir / f"ckpt_v{SP2_CHECKPOINT_VERSION}"
    if imported.exists():
        assert sorted(p.name for p in imported.iterdir()) == ["meta.json", "model.pt"]   # no trainer state
```

Apply the same reasoning to every test the Step 1 grep found (assert "carried, not necessarily kept").

- [ ] **Step 4: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_retention.py tests/unit/test_sp4_snapshot_store.py -q`
Expected: collection errors — `ModuleNotFoundError: No module named 'colosseum.arena.retention'`, `ImportError: cannot import name 'read_snapshots'`.

- [ ] **Step 5: Write `src/colosseum/arena/retention.py`**

```python
"""Snapshot retention (SP4 spec block 4): a pure function "entries + scores + policy -> what to keep".

Per agent, ``retain`` builds the kept set in this order:

1. **always** (never evicted, not even by the cap): the newest ``keep_last``; the final snapshot of the
   CURRENT run (``final`` and ``run_id == policy.current_run_id``; finals imported from parent runs and SP3
   snapshots without ``run_id`` are ordinary; with ``current_run_id=None``, e.g. a manager built without a
   run id, every final is protected as in SP3); every pinned snapshot; the snapshot just saved (``protect``);
2. **keep_every**: version is a multiple of ``keep_every_period`` (``interval * keep_every``; 0 = off);
3. **top-k** (``keep_top_k > 0``): the ``keep_top_k`` rated snapshots with the highest ``lcb`` (ties: the
   newer first), chosen among ALL rated entries (a snapshot of rule 1 may hold a top-k place);
4. **new** (only with ``keep_top_k > 0``): a snapshot outside the ``keep_last`` window with fewer than
   ``top_k_min_games`` rated games (``SnapshotScore.games``; no score = 0 games) is kept until it has them.

Everything else is evicted. Then the cap: while the kept set is larger than ``max_snapshots`` (``None`` =
no cap), unprotected snapshots (not rule 1) are evicted in this order:

a. keep_every-only snapshots (rule 2, not 3 or 4): oldest first when ``keep_top_k == 0`` (FIFO, no ratings:
   also the distributed learner), else the unrated ones first (oldest first), then by ascending ``lcb``
   (ties: the older first);
b. new-only snapshots (rule 4, not 3): oldest first;
c. top-k snapshots: ascending ``lcb`` (ties: the older first).

If rule 1 alone exceeds the cap, every other snapshot is evicted, rule 1 stays and ``warnings`` says so
(the caller logs it once per agent). ``reasons`` names, per checkpoint id, why it was kept or evicted.
The function never touches the file system; ``CheckpointManager`` applies the decision (index first, then
``on_evict``), and SP5's storage can call it the same way.

``combine_scores`` / ``snapshot_scores`` turn per-layout BT results (live Laplace or bootstrap) into one
``SnapshotScore`` per snapshot of an agent (spec block 4: the BT rating over the agent's competitive layouts,
averaged with the ``matchmaking.layouts`` weights on every draw; lower bounds in the lineage-contrast scale;
games with ``score`` layouts only use the mean score).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

KEEP_LAST, FINAL, PINNED, SAVED = "keep_last", "final", "pinned", "just saved"
KEEP_EVERY, TOP_K, NEW = "keep_every", "top_k", "new"


@dataclass(frozen=True)
class SnapshotInfo:
    checkpoint_id: str
    version: int
    final: bool
    run_id: str | None
    pinned: bool


@dataclass(frozen=True)
class SnapshotScore:
    lcb: float | None    # None = not rated yet
    games: int


@dataclass(frozen=True)
class RetentionPolicy:
    keep_last: int
    keep_every_period: int    # interval * keep_every; 0 = off
    keep_top_k: int
    top_k_min_games: int
    max_snapshots: int | None
    current_run_id: str | None


@dataclass(frozen=True)
class RetentionDecision:
    keep: list[str]
    evict: list[str]
    reasons: dict[str, str]   # checkpoint_id -> why kept / evicted (validate, logs)
    warnings: list[str] = field(default_factory=list)


def _lcb(scores: Mapping[str, SnapshotScore], checkpoint_id: str) -> float | None:
    score = scores.get(checkpoint_id)
    return None if score is None else score.lcb


def retain(entries: Sequence[SnapshotInfo], scores: Mapping[str, SnapshotScore], policy: RetentionPolicy,
           *, protect: str | None = None) -> RetentionDecision:
    """Apply the rules of the module docstring to ``entries`` (any order; ids unique)."""
    if policy.keep_last < 1 or policy.keep_every_period < 0 or policy.keep_top_k < 0 \
            or policy.top_k_min_games < 0 or (policy.max_snapshots is not None and policy.max_snapshots < 1):
        raise ValueError(f"retain: invalid policy {policy}")
    ordered = sorted(entries, key=lambda e: e.version)
    ids = [e.checkpoint_id for e in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError(f"retain: duplicate checkpoint ids in {ids}")
    version = {e.checkpoint_id: e.version for e in ordered}
    newest = {e.checkpoint_id for e in ordered[-policy.keep_last:]}

    always: dict[str, list[str]] = {}
    for e in ordered:
        why = []
        if e.checkpoint_id in newest:
            why.append(KEEP_LAST)
        if e.final and (policy.current_run_id is None or e.run_id == policy.current_run_id):
            why.append(FINAL)
        if e.pinned:
            why.append(PINNED)
        if e.checkpoint_id == protect:
            why.append(SAVED)
        if why:
            always[e.checkpoint_id] = why

    period = policy.keep_every_period
    every = {e.checkpoint_id for e in ordered if period > 0 and e.version % period == 0}
    top: set[str] = set()
    new: set[str] = set()
    if policy.keep_top_k > 0:
        rated = [e for e in ordered if _lcb(scores, e.checkpoint_id) is not None]
        rated.sort(key=lambda e: (-_lcb(scores, e.checkpoint_id), -e.version))
        top = {e.checkpoint_id for e in rated[:policy.keep_top_k]}
        for e in ordered:
            score = scores.get(e.checkpoint_id)
            games = score.games if score is not None else 0
            if e.checkpoint_id not in newest and games < policy.top_k_min_games:
                new.add(e.checkpoint_id)

    reasons: dict[str, str] = {}
    keep: set[str] = set()
    for cid in ids:
        rules = list(always.get(cid, []))
        rules += [name for name, members in ((KEEP_EVERY, every), (TOP_K, top), (NEW, new)) if cid in members]
        if rules:
            keep.add(cid)
            reasons[cid] = "kept: " + ", ".join(rules)
        else:
            reasons[cid] = "evicted: no retention rule"

    warnings: list[str] = []
    cap = policy.max_snapshots
    if cap is not None and len(keep) > cap:
        if len(always) > cap:
            warnings.append(f"{len(always)} protected snapshots (keep_last, final of this run, pinned, just saved) "
                            f"exceed checkpoint.max_snapshots={cap}; they are all kept")
        unprotected = [cid for cid in ids if cid in keep and cid not in always]
        every_only = [cid for cid in unprotected if cid not in top and cid not in new]
        new_only = [cid for cid in unprotected if cid in new and cid not in top]
        top_only = [cid for cid in unprotected if cid in top]
        if policy.keep_top_k == 0:
            stage_a = sorted(every_only, key=lambda c: version[c])
        else:
            unrated = sorted((c for c in every_only if _lcb(scores, c) is None), key=lambda c: version[c])
            rated_a = sorted((c for c in every_only if _lcb(scores, c) is not None),
                             key=lambda c: (_lcb(scores, c), version[c]))
            stage_a = unrated + rated_a
        stage_b = sorted(new_only, key=lambda c: version[c])
        stage_c = sorted(top_only, key=lambda c: (_lcb(scores, c), version[c]))
        for stage, label in ((stage_a, KEEP_EVERY), (stage_b, NEW), (stage_c, TOP_K)):
            for cid in stage:
                if len(keep) <= cap:
                    break
                keep.discard(cid)
                reasons[cid] = f"evicted: over max_snapshots={cap} ({label})"
    return RetentionDecision(keep=[c for c in ids if c in keep], evict=[c for c in ids if c not in keep],
                             reasons=reasons, warnings=warnings)


def _version_of(entity: str, agent_id: str) -> int | None:
    """``N`` of the entity key ``<agent_id>@v<N>``, else None (other agents, fixed players)."""
    prefix = f"{agent_id}@v"
    rest = entity[len(prefix):] if entity.startswith(prefix) else ""
    return int(rest) if rest.isdigit() else None


def _weights(layouts: Sequence[str], layout_weights: Mapping[str, float]) -> np.ndarray:
    w = np.array([float(layout_weights.get(name, 0.0)) for name in layouts])
    return w if w.sum() > 0 else np.ones(len(layouts))


def combine_scores(agent_id: str, *, fits: Mapping[str, Any], uncertainty: Mapping[str, Any],
                   scores: Mapping[str, Any], layout_weights: Mapping[str, float],
                   quantile: float) -> dict[str, SnapshotScore]:
    """One ``SnapshotScore`` per snapshot entity ``<agent_id>@v<N>``, keyed ``ckpt_v<N>`` (module docstring).

    - Pairwise layouts (``fits``: layout -> ``BTFit``; ``uncertainty``: layout -> ``Uncertainty``): an entity
      rated in one layout gets that layout's ``Uncertainty.lcb``; rated in several, the ``quantile`` of the
      layout-weighted mean of its ``Uncertainty.draws`` (the same draw index across layouts; layouts are
      independent fits). ``games``: ``BTFit.games`` summed over the layouts (self-play excluded there).
    - Only when no fit names the agent's snapshots: ``score`` layouts (``scores``: layout -> ``ScoreSummary``
      with ``by_entity[entity] = (n, mean, se)``): the weighted mean and its standard error,
      ``lcb = mean + z_q * se``, ``games`` = matches.
    Weights: ``layout_weights`` (``matchmaking.layouts``; layouts without a weight count 0, all 0 -> equal).
    An entity without an interval yet (rated, but not in ``uncertainty``) has ``lcb=None``.
    """
    per_entity: dict[str, list[str]] = {}
    for layout, fit in fits.items():
        for entity in fit.rating:
            if _version_of(entity, agent_id) is not None:
                per_entity.setdefault(entity, []).append(layout)
    out: dict[str, SnapshotScore] = {}
    if per_entity:
        for entity, layouts in per_entity.items():
            games = sum(int(fits[name].games.get(entity, 0)) for name in layouts)
            unc = [uncertainty.get(name) for name in layouts]
            if any(u is None or entity not in u.lcb for u in unc):
                lcb = None
            elif len(layouts) == 1:
                lcb = float(unc[0].lcb[entity])
            else:
                w = _weights(layouts, layout_weights)
                draws = [np.asarray(u.draws[entity], dtype=np.float64) for u in unc]
                n = min(len(d) for d in draws)
                mixed = sum(wi * d[:n] for wi, d in zip(w, draws, strict=True)) / w.sum()
                lcb = float(np.quantile(mixed, quantile))
            out[f"ckpt_v{_version_of(entity, agent_id)}"] = SnapshotScore(lcb, games)
        return out
    z = NormalDist().inv_cdf(quantile)
    rows: dict[str, list[tuple[str, int, float, float]]] = {}
    for layout, summary in scores.items():
        for entity, (n, mean, se) in summary.by_entity.items():
            if _version_of(entity, agent_id) is not None:
                rows.setdefault(entity, []).append((layout, int(n), float(mean), float(se)))
    for entity, items in rows.items():
        w = _weights([layout for layout, *_ in items], layout_weights)
        mean = float(sum(wi * m for wi, (_l, _n, m, _s) in zip(w, items, strict=True)) / w.sum())
        se = math.sqrt(sum((wi * s) ** 2 for wi, (_l, _n, _m, s) in zip(w, items, strict=True))) / w.sum()
        out[f"ckpt_v{_version_of(entity, agent_id)}"] = SnapshotScore(mean + z * se, sum(n for _l, n, *_ in items))
    return out


def snapshot_scores(snapshot: Any, agent_id: str, *, layout_weights: Mapping[str, float],
                    quantile: float) -> dict[str, SnapshotScore]:
    """``combine_scores`` of a ``RatingsSnapshot`` (T2.2): ``fits``, ``uncertainty`` and ``scores`` per layout."""
    return combine_scores(agent_id, fits=snapshot.fits, uncertainty=snapshot.uncertainty,
                          scores=getattr(snapshot, "scores", {}) or {}, layout_weights=layout_weights,
                          quantile=quantile)


def describe_retention(checkpoint: Any) -> str:
    """The effective retention rule of a ``CheckpointConfig`` in one line (``validate``, spec block 4)."""
    parts = [f"newest {checkpoint.keep_last} (keep_last, with trainer state)"]
    if checkpoint.keep_every > 0:
        parts.append(f"every {checkpoint.interval * checkpoint.keep_every}th version (keep_every "
                     f"{checkpoint.keep_every})")
    parts += ["the final one of this run", "pinned"]
    if checkpoint.keep_top_k > 0:
        parts.append(f"top {checkpoint.keep_top_k} by the rating's lower bound (quantile {checkpoint.top_k_quantile});"
                     f" a snapshot with fewer than {checkpoint.top_k_min_games} rated games is kept until it has them")
        order = ("keep_every snapshots by lowest lower bound (unrated first), then new ones oldest first, then "
                 "top-k by lowest lower bound")
    else:
        parts.append("top-k off")
        order = "keep_every snapshots oldest first"
    cap = checkpoint.max_snapshots
    parts.append("no cap" if cap is None else f"at most {cap} per agent (over the cap: {order})")
    return "snapshots: " + "; ".join(parts)
```

- [ ] **Step 6: Config keys and checks**

`src/colosseum/core/config.py`, `CheckpointConfig`: update the class docstring (add "SP4: ``keep_top_k`` top-k by the live rating's lower bound with protection of new snapshots; ``max_snapshots`` caps the pool (spec block 4).") and add after `keep_every`:

```python
    keep_top_k: int = Field(
        default=0, ge=0,
        description="Also keep the k snapshots with the highest lower bound of their live BT rating (needs "
                    "match_log.enabled); 0 = off.",
    )
    top_k_quantile: float = Field(
        default=0.1, gt=0.0, lt=0.5,
        description="Quantile of the rating distribution used as the lower bound for keep_top_k.",
    )
    top_k_min_games: int = Field(
        default=200, ge=0,
        description="With keep_top_k > 0, a snapshot outside keep_last with fewer rated games is never evicted.",
    )
    max_snapshots: int | None = Field(
        default=100, ge=1,
        description="At most this many snapshots per agent (keep_last, this run's final and pinned ones are never "
                    "evicted); null = no cap.",
    )
```

and an after-validator (below `_translate_pool_size`):

```python
    @model_validator(mode="after")
    def _check_cap(self) -> CheckpointConfig:
        if self.max_snapshots is not None and self.max_snapshots < self.keep_last:
            raise ValueError(f"checkpoint.max_snapshots ({self.max_snapshots}) is smaller than checkpoint.keep_last "
                             f"({self.keep_last}): the newest keep_last snapshots are never evicted; raise "
                             f"max_snapshots or lower keep_last")
        return self
```

`ColosseumConfig`: add (after `_validate_agent_overrides`; T6.1 extends the same method):

```python
    @model_validator(mode="after")
    def _check_sp4_dependencies(self) -> ColosseumConfig:
        """SP4 features that read the match log (spec blocks 1, 4, 6)."""
        if self.checkpoint.keep_top_k > 0 and not self.match_log.enabled:
            raise ValueError(f"checkpoint.keep_top_k={self.checkpoint.keep_top_k} keeps snapshots by their rating, "
                             f"which comes from the match log; set match_log.enabled: true or keep_top_k: 0")
        return self
```

(`load_config` turns the `ValidationError` into a `ConfigError`, as for every model check.)

`scripts/bench_throughput.py::_make_config`: the checkpoint section becomes
```python
        checkpoint={
            "interval": 10**9,  # no checkpoints: every seat collects
            "keep_last": 10,
            "keep_every": 0,
            "keep_top_k": 0,
            "top_k_quantile": 0.1,
            "top_k_min_games": 200,
            "max_snapshots": 100,
            "save_optimizer": True,
        },
```

- [ ] **Step 7: `CheckpointManager`**

`src/colosseum/coordinator/checkpoint_manager.py`:

1. Module docstring: replace the "Retention (SP3 spec block 4) ..." paragraph by:
```
Retention (SP4 spec block 4; the decision is ``colosseum.arena.retention.retain``), after every save, import
and ``reapply_retention``: the newest ``keep_last``, the final snapshot of the current run (``run_id``; with no
run id every final, as in SP3), pinned snapshots (``meta.json`` ``pinned``, re-read from disk before every
decision, so ``colosseum pin`` works while a run trains), the snapshot just saved, every ``keep_every``-th
version, and with ``keep_top_k > 0`` the top-k by the scores provider's lower bound plus new snapshots with
fewer than ``top_k_min_games`` rated games; then the ``max_snapshots`` cap. Evicted ids leave the index first,
then ``on_evict(agent_id, checkpoint_id)`` is called for each (a raising callback never leaves deleted dirs in
the index). Only the newest ``keep_last`` keep ``trainer_state.pt``. Every new ``meta.json`` gets ``pinned``
(false) and, with a run id, ``run_id``.
```
2. Imports: `from collections.abc import Callable, Mapping`; `from colosseum.arena.retention import RetentionPolicy, SnapshotInfo, SnapshotScore, retain`; `from colosseum.utils.fs import write_text_atomic`.
3. Rename `_link_or_copy` to `link_or_copy` (public; docstring unchanged) and keep `_link_or_copy = link_or_copy` below it.
4. Add after `_read_agent_dir`:
```python
def read_snapshots(agent_dir: str | Path) -> list[CheckpointInfo]:
    """Valid snapshots of one agent dir, by version, read without touching it (no tmp cleanup, malformed ones
    skipped with a warning). For read-only callers on a live run (``colosseum pin``, the tournament)."""
    return _read_agent_dir(Path(agent_dir))


def set_pinned(ckpt_dir: str | Path, pinned: bool) -> dict:
    """Set ``pinned`` in ``ckpt_dir/meta.json`` by an atomic rewrite (temp file + ``os.replace``: a hard-linked
    copy of the file in another run dir keeps its old content); returns the new meta. ConfigError for a
    missing dir or an invalid ``meta.json``."""
    path = Path(ckpt_dir)
    meta = read_checkpoint_meta(path)
    meta["pinned"] = bool(pinned)
    write_text_atomic(path / META_FILE, json.dumps(meta, indent=2, sort_keys=True))
    return meta
```
   (`read_checkpoint_meta` is defined further down; move `read_snapshots` / `set_pinned` below it, or keep them
   above: Python resolves the name at call time.)
5. `__init__`:
```python
    def __init__(self, base_dir: str | Path, keep_last: int = 20, keep_every: int = 0, interval: int = 1,
                 on_evict: Callable[[str, str], None] | None = None, *, run_id: str | None = None,
                 keep_top_k: int = 0, top_k_min_games: int = 200, max_snapshots: int | None = None,
                 scores_fn: Callable[[str], Mapping[str, SnapshotScore]] | None = None) -> None:
        if keep_last < 1 or keep_every < 0 or interval < 1 or keep_top_k < 0 or top_k_min_games < 0 \
                or (max_snapshots is not None and max_snapshots < keep_last):
            raise ValueError(f"CheckpointManager: need keep_last >= 1, keep_every >= 0, interval >= 1, keep_top_k >= 0, "
                             f"top_k_min_games >= 0 and max_snapshots >= keep_last; got {keep_last}, {keep_every}, "
                             f"{interval}, {keep_top_k}, {top_k_min_games}, {max_snapshots}")
        self._base_dir = Path(base_dir)
        self._keep_last = int(keep_last)
        self._keep_every = int(keep_every)
        self._interval = int(interval)
        self._on_evict = on_evict
        self._run_id = run_id
        self._keep_top_k = int(keep_top_k)
        self._top_k_min_games = int(top_k_min_games)
        self._max_snapshots = max_snapshots
        self._scores_fn = scores_fn
        self._warned: set[str] = set()       # agents whose cap overflow was logged
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[CheckpointInfo]] = {}
        self._scan()
```
   plus a property `run_id -> str | None`.
6. `save`: build the meta as
```python
        meta = {
            "agent_id": agent_id,
            "checkpoint_id": checkpoint_id,
            "policy_version": int(policy_version),
            "timestamp": timestamp,
            **(meta_extra or {}),
        }
        previous = next((c for c in self._index.get(agent_id, []) if c.checkpoint_id == checkpoint_id), None)
        meta.setdefault("pinned", bool(previous is not None and previous.meta.get("pinned") is True))
        if self._run_id is not None:
            meta.setdefault("run_id", self._run_id)
```
   and change the log line to `f"Saved checkpoint {checkpoint_id} of {agent_id} (pool {len(self._index[agent_id])})"`.
7. Replace `_retain` by:
```python
    def _policy(self) -> RetentionPolicy:
        return RetentionPolicy(keep_last=self._keep_last, keep_every_period=self._interval * self._keep_every,
                               keep_top_k=self._keep_top_k, top_k_min_games=self._top_k_min_games,
                               max_snapshots=self._max_snapshots, current_run_id=self._run_id)

    @staticmethod
    def _refresh_pins(entries: list[CheckpointInfo]) -> None:
        """Re-read ``pinned`` from every entry's ``meta.json`` (another process may have pinned it); an unreadable
        file keeps the indexed value."""
        for c in entries:
            try:
                disk = json.loads((c.path / META_FILE).read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(disk, dict):
                c.meta["pinned"] = disk.get("pinned") is True

    def _retain(self, agent_id: str, entries: list[CheckpointInfo],
                protect: str | None = None) -> tuple[list[CheckpointInfo], list[str]]:
        """Apply ``arena.retention.retain`` to ``entries`` (sorted by version); returns the kept ones and the ids
        evicted (the caller updates the index first, then calls ``_notify_evicted``)."""
        self._refresh_pins(entries)
        scores = self._scores_fn(agent_id) if self._scores_fn is not None and self._keep_top_k > 0 else {}
        infos = [SnapshotInfo(c.checkpoint_id, c.policy_version, c.meta.get("final") is True, c.meta.get("run_id"),
                              c.meta.get("pinned") is True) for c in entries]
        decision = retain(infos, scores, self._policy(), protect=protect)
        if decision.warnings and agent_id not in self._warned:      # once per agent (spec block 4)
            self._warned.add(agent_id)
            for warning in decision.warnings:
                logger.warning(f"Snapshots of {agent_id}: {warning}")
        keep = set(decision.keep)
        newest = {c.checkpoint_id for c in entries[-self._keep_last:]}
        kept: list[CheckpointInfo] = []
        evicted: list[str] = []
        for c in entries:
            if c.checkpoint_id not in keep:
                self._evict(agent_id, c.checkpoint_id)
                logger.debug(f"Evicted checkpoint {c.checkpoint_id} of {agent_id}: {decision.reasons[c.checkpoint_id]}")
                evicted.append(c.checkpoint_id)
                continue
            kept.append(c)
            if c.checkpoint_id not in newest:
                (c.path / TRAINER_FILE).unlink(missing_ok=True)
        return kept, evicted

    def reapply_retention(self, agent_id: str) -> list[str]:
        """Rerun retention for ``agent_id`` without a save (new ratings, new pins); returns the evicted ids, after
        ``on_evict`` ran for each. Main loop only (spec block 4)."""
        entries = list(self._index.get(agent_id, []))
        if not entries:
            return []
        self._index[agent_id], evicted = self._retain(agent_id, entries)
        if evicted:
            logger.info(f"Retention of {agent_id}: evicted {evicted} (pool {len(self._index[agent_id])})")
        self._notify_evicted(agent_id, evicted)
        return evicted

    def pin(self, agent_id: str, checkpoint_id: str, pinned: bool = True) -> None:
        """Pin (or unpin) a stored snapshot: ``meta.json`` is rewritten atomically. A pinned snapshot is never
        evicted; an unpinned one becomes evictable at the next retention pass. ConfigError for an unknown id."""
        entry = next((c for c in self._index.get(agent_id, []) if c.checkpoint_id == checkpoint_id), None)
        if entry is None:
            stored = [c.checkpoint_id for c in self._index.get(agent_id, [])]
            raise ConfigError(f"agent {agent_id!r} has no stored snapshot {checkpoint_id!r} (stored: {stored[-10:]})")
        entry.meta.update(set_pinned(entry.path, pinned))

    def pinned(self, agent_id: str) -> list[str]:
        """Ids of the pinned snapshots of ``agent_id`` (as on disk now), by version."""
        entries = self._index.get(agent_id, [])
        self._refresh_pins(entries)
        return [c.checkpoint_id for c in entries if c.meta.get("pinned") is True]
```
   `import_snapshots`: unchanged except that `info.meta` (copied with `dict(...)`) already carries the source's `pinned` and `run_id`; `_retain` re-reads pins from the linked files.

- [ ] **Step 8: Coordinator and launcher wiring**

`src/colosseum/coordinator/coordinator.py`:
1. Imports: `from colosseum.arena.retention import SnapshotScore, snapshot_scores` and `from colosseum.league.lineups import enabled_layouts`.
2. `__init__(self, config, spec, player_roles, checkpoint_dir, env_steps=lambda: 0, run_id: str | None = None)`; before building the manager:
```python
        self._ratings_source: Callable[[], object | None] | None = None
        self._evict_listeners: list[Callable[[str, str], None]] = []
        self._checkpoint_manager = CheckpointManager(
            base_dir=checkpoint_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every, interval=ckpt.interval,
            on_evict=self._on_evict, run_id=run_id, keep_top_k=ckpt.keep_top_k,
            top_k_min_games=ckpt.top_k_min_games, max_snapshots=ckpt.max_snapshots, scores_fn=self.snapshot_scores,
        )
```
3. `_on_evict`: after the buffering block (and also when nothing is buffered), call every listener:
```python
    def _on_evict(self, agent_id: str, checkpoint_id: str) -> None:
        """... (docstring as before, plus:) Listeners (``add_evict_listener``: the journal) are told last."""
        logger.debug(f"Snapshot {checkpoint_id} of {agent_id} evicted")
        self._pfsp.forget((agent_id, checkpoint_id))
        if self._buffer_evictions:
            bucket = self._evicted.setdefault(agent_id, [])
            if checkpoint_id not in bucket:
                bucket.append(checkpoint_id)
        for listener in self._evict_listeners:
            listener(agent_id, checkpoint_id)
```
4. New methods:
```python
    def set_ratings_source(self, fn: Callable[[], object | None]) -> None:
        """Where top-k retention reads ratings from: a callable returning the latest immutable
        ``RatingsSnapshot`` (``LiveRatings.latest``) or None. Main loop only."""
        self._ratings_source = fn

    def add_evict_listener(self, fn: Callable[[str, str], None]) -> None:
        """``fn(agent_id, checkpoint_id)`` after every eviction (after PFSP forgot it)."""
        self._evict_listeners.append(fn)

    def snapshot_scores(self, agent_id: str) -> dict[str, SnapshotScore]:
        """Live-rating scores of ``agent_id``'s snapshots (``arena.retention.snapshot_scores``): layout weights from
        the agent's ``matchmaking.layouts``, quantile ``checkpoint.top_k_quantile``; ``{}`` before the first refit."""
        snapshot = self._ratings_source() if self._ratings_source is not None else None
        if snapshot is None:
            return {}
        weights = enabled_layouts(self._spec, self._config.get_agent_config(agent_id).matchmaking)
        return snapshot_scores(snapshot, agent_id, layout_weights=weights,
                               quantile=self._config.checkpoint.top_k_quantile)
```

`src/colosseum/launcher.py`:
1. `_build_coordinator`: `Coordinator(..., env_steps=..., run_id=self._run_dir.run_id)`.
2. `__init__`: `self._retention_version: int | None = None`.
3. New methods:
```python
    def _wire_retention(self) -> None:
        """T4.1: top-k retention reads the live ratings; every eviction reaches the journal, whose next compaction
        pass folds the evicted entity's detailed matches into counts (spec blocks 1, 4)."""
        from colosseum.arena.entities import entity_key, snapshot_version

        live = getattr(self, "_live_ratings", None)
        if live is not None:
            self._coordinator.set_ratings_source(live.latest)

        def note(agent_id: str, checkpoint_id: str) -> None:
            writer = getattr(self, "_match_log_writer", None)
            if writer is not None:
                writer.note_evicted([entity_key(agent_id, snapshot_version(checkpoint_id))])

        self._coordinator.add_evict_listener(note)

    def _reapply_retention(self) -> None:
        """After every live refit (a new ``RatingsSnapshot.version``), rerun retention for every trainable agent;
        only with ``keep_top_k > 0``, where scores decide (spec block 4). Main loop only."""
        live = getattr(self, "_live_ratings", None)
        if live is None or self._config.checkpoint.keep_top_k == 0 or self._coordinator is None:
            return
        snapshot = live.latest()
        if snapshot is None or snapshot.version == self._retention_version:
            return
        self._retention_version = snapshot.version
        for aid in self._agent_ids:
            self._coordinator.checkpoint_manager.reapply_retention(aid)
```
4. `launch`: call `self._wire_retention()` right after T1.3 / T2.2 create the journal writer and the live ratings (both exist before `_start_children`; if T2.2 builds `self._live_ratings` inside `_open_metrics`, call `_wire_retention()` right after `self._open_metrics(resume_states)`).
5. `_monitor_loop`: call `self._reapply_retention()` after `self._drain_all_checkpoints()` (both run on the main loop; the refit thread never touches the coordinator).

- [ ] **Step 9: `validate` printout and the distributed guard**

`src/colosseum/core/validation.py`, in `validate_config` (train scope only, after `report.lines.extend(describe_mix(config, spec))`):

```python
        _describe_retention(config, report)
```
and the helper (module level, `logger = logging.getLogger(__name__)` added at the top with `import logging`):

```python
def _describe_retention(config: ColosseumConfig, report: ValidationReport) -> None:
    """The effective retention rule per trainable agent (spec block 4), and a WARNING when the pinned snapshots a
    run-dir resume carries over plus ``keep_last`` exceed ``max_snapshots``."""
    from pathlib import Path

    from colosseum.arena.retention import describe_retention
    from colosseum.coordinator.checkpoint_manager import RESUME_RUN_DIR, classify_resume_source, read_snapshots

    rule = describe_retention(config.checkpoint)
    source = config.training.resume_from
    try:
        run_dir = Path(source) if source and classify_resume_source(source) == RESUME_RUN_DIR else None
    except ConfigError:
        run_dir = None          # a bad resume source is reported by the launcher, as before (SP3)
    cap = config.checkpoint.max_snapshots
    for aid in config.get_trainable_agent_ids():
        report.lines.append(f"agent '{aid}': {rule}")
        if run_dir is None or cap is None:
            continue
        pinned = sum(1 for c in read_snapshots(run_dir / "checkpoints" / aid) if c.meta.get("pinned") is True)
        if config.checkpoint.keep_last + pinned > cap:
            message = (f"agent '{aid}': keep_last {config.checkpoint.keep_last} + {pinned} pinned snapshots carried "
                       f"over from {run_dir} exceed checkpoint.max_snapshots {cap}: the protected snapshots are all "
                       f"kept and nothing else fits; unpin some (colosseum pin --remove) or raise max_snapshots")
            logger.warning(message)
            report.lines.append(f"WARNING: {message}")
```

`src/colosseum/distributed.py`:
1. `check_distributed_scope`: add to `problems` (before the `raise`):
```python
    if config.checkpoint.keep_top_k > 0:
        problems.append(f"checkpoint.keep_top_k={config.checkpoint.keep_top_k} (snapshot ratings need the match "
                        f"log, which the distributed roles do not write)")
```
2. `distributed_checkpoint_manager(config, base_dir, run_id: str | None = None)`:
```python
    ckpt = config.checkpoint
    return CheckpointManager(base_dir=base_dir, keep_last=ckpt.keep_last, keep_every=ckpt.keep_every,
                             interval=ckpt.interval, run_id=run_id, max_snapshots=ckpt.max_snapshots)
```
   and its docstring: "... the final snapshot of this learner's run (``run_id``) ...; ``max_snapshots`` with FIFO order (no ratings: spec block 8)."
3. `run_distributed_learner`: `coordinator_ckpt = distributed_checkpoint_manager(config, run_dir.checkpoints, run_id=run_dir.run_id)`.

- [ ] **Step 10: Run the new and the touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_retention.py tests/unit/test_sp4_snapshot_store.py tests/integration/test_sp4_retention_runs.py tests/unit/test_sp3_snapshot_retention.py tests/unit/test_sp2_checkpoints.py tests/unit/test_sp3_distributed_guards.py tests/unit/test_bench_throughput.py tests/integration/test_sp3_sp2_resume.py "tests/integration/test_sp2_league_runs.py::test_resume_continues_versions_env_steps_and_lr" "tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature" -q`
Expected: all pass.

- [ ] **Step 11: Full fast suite + ruff**

Expected: green, zero warnings. `grep -rn "_link_or_copy" src` shows only the alias line.

- [ ] **Step 12: Commit and push**

```bash
git add src/colosseum/arena/retention.py src/colosseum/core/config.py src/colosseum/coordinator \
        src/colosseum/launcher.py src/colosseum/core/validation.py src/colosseum/distributed.py \
        scripts/bench_throughput.py tests/unit/test_sp4_retention.py tests/unit/test_sp4_snapshot_store.py \
        tests/integration/test_sp4_retention_runs.py tests/integration/test_sp2_league_runs.py \
        tests/integration/test_sp2_game_runs.py tests/integration/test_sp3_sp2_resume.py
git commit -m "feat: snapshot retention by rating (top-k, pinned, max_snapshots, current-run final)"
git push
```

---
### Task T4.2: `colosseum pin`

Spec block 3 («Закрепление») and block 4: `colosseum pin runs/x main@v1200 [--remove]` pins (unpins) a stored snapshot by rewriting its `meta.json` atomically (`set_pinned`, T4.1); `colosseum pin runs/x --list` lists every agent's stored snapshots with the pinned ones marked. The command never builds a `CheckpointManager` on the run dir (its scan cleans tmp dirs of a live writer; SP3 rule "read-only on user paths"): it uses `read_snapshots` and `set_pinned`. A training run of that dir sees the pin at its next retention pass (`_refresh_pins`, T4.1). `tournament --pin` (T5.4) calls the same library function.

**Files:**
- Modify: `src/colosseum/coordinator/checkpoint_manager.py` (`SNAPSHOT_REF_RE`, `parse_snapshot_ref`, `pin_snapshot`, `list_snapshots`)
- Modify: `src/colosseum/cli.py` (the `pin` command)
- Test: `tests/integration/test_sp4_pin_cli.py`

**Interfaces:**
- Consumes: `set_pinned`, `read_snapshots` (T4.1); `CheckpointInfo`.
- Produces:
  - CLI `colosseum pin RUN [AGENT@vN] [--remove] [--list]`; exit codes 0 / 1 (config error: not a run dir, unknown agent or snapshot, a malformed `meta.json`) / 2 (bad arguments).
  - *`checkpoint_manager.parse_snapshot_ref(text) -> tuple[str, int]` (`"main@v1200"` → `("main", 1200)`; ConfigError otherwise);
  - *`checkpoint_manager.pin_snapshot(run_dir, agent_id, version, *, pinned=True) -> Path` (the snapshot dir);
  - *`checkpoint_manager.list_snapshots(run_dir) -> dict[str, list[CheckpointInfo]]` (agents in name order).

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_sp4_pin_cli.py`:

```python
"""``colosseum pin`` (SP4 T4.2, spec blocks 3 and 4): pin / unpin / list, errors with hints, and a pin that a
running training honours."""
from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from cli_runner import TTT_CONFIG, training_process, wait_for
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import CheckpointManager, parse_snapshot_ref, pin_snapshot
from colosseum.core.errors import ConfigError


def invoke(*args):
    return CliRunner().invoke(main, ["pin", *map(str, args)])


def make_run(tmp_path):
    root = tmp_path / "run"
    mgr = CheckpointManager(root / "checkpoints", keep_last=10, run_id="r")
    for aid in ("main", "rival"):
        for version in (10, 20, 30):
            mgr.save(aid, version, {"w": np.zeros(2, np.float32)})
    return root


def pinned_flag(root, aid, version) -> bool:
    return json.loads((root / "checkpoints" / aid / f"ckpt_v{version}" / "meta.json").read_text())["pinned"]


def test_pin_unpin_and_list(tmp_path):
    root = make_run(tmp_path)
    result = invoke(root, "main@v20")
    assert result.exit_code == 0, result.output
    assert "Pinned main@v20" in result.output and pinned_flag(root, "main", 20) is True
    assert invoke(root, "main@v20").exit_code == 0                      # idempotent
    listing = invoke(root, "--list")
    assert listing.exit_code == 0
    assert "main: ckpt_v10, ckpt_v20 (pinned), ckpt_v30" in listing.output and "rival: ckpt_v10" in listing.output
    result = invoke(root, "main@v20", "--remove")
    assert result.exit_code == 0 and "Unpinned main@v20" in result.output and pinned_flag(root, "main", 20) is False


def test_errors_name_the_problem_and_a_hint(tmp_path):
    root = make_run(tmp_path)
    result = invoke(root, "main@v25")
    assert result.exit_code == 1 and "ckpt_v25" in result.output and "ckpt_v30" in result.output
    assert "--list" in result.output
    result = invoke(root, "nobody@v10")
    assert result.exit_code == 1 and "main" in result.output and "rival" in result.output
    result = invoke(tmp_path / "missing", "main@v10")
    assert result.exit_code != 0
    result = invoke(tmp_path, "main@v10")
    assert result.exit_code == 1 and "not a run dir" in result.output
    assert invoke(root, "main-v10").exit_code == 2
    assert invoke(root).exit_code == 2                                   # neither a snapshot nor --list
    assert invoke(root, "main@v10", "--list").exit_code == 2
    assert parse_snapshot_ref("b_1@v7") == ("b_1", 7)
    with pytest.raises(ConfigError, match="agent@v"):
        parse_snapshot_ref("main@7")
    with pytest.raises(ConfigError):
        pin_snapshot(root, "main", 99)


def test_a_running_training_keeps_a_snapshot_pinned_from_outside(tmp_path):
    overrides = {"training.total_timesteps": "40000", "checkpoint.interval": "5", "checkpoint.keep_last": "2",
                 "checkpoint.keep_every": "0"}
    with training_process(TTT_CONFIG, tmp_path, name="live", overrides=overrides) as (proc, root):
        agent_dir = root / "checkpoints" / "agent_0"
        assert wait_for(lambda: len(list(agent_dir.glob("ckpt_v*"))) >= 1 or proc.poll() is not None, 120)
        first = min(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
        result = invoke(root, f"agent_0@v{first}")
        if result.exit_code != 0:          # evicted between the glob and the pin: take the newest
            first = max(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
            result = invoke(root, f"agent_0@v{first}")
        assert result.exit_code == 0, result.output
        assert wait_for(lambda: max(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))
                        >= first + 20 or proc.poll() is not None, 120)
        assert (agent_dir / f"ckpt_v{first}").is_dir() and pinned_flag(root, "agent_0", first) is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_pin_cli.py -q`
Expected: `ImportError: cannot import name 'parse_snapshot_ref'`.

- [ ] **Step 3: Implementation**

`src/colosseum/coordinator/checkpoint_manager.py` (after `set_pinned`):

```python
SNAPSHOT_REF_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_-]*)@v(\d+)")


def parse_snapshot_ref(text: str) -> tuple[str, int]:
    """``"main@v1200"`` -> ``("main", 1200)``: a snapshot reference of the CLI (spec §8: ``agent@v<N>`` in the CLI
    only; configs name agents)."""
    match = SNAPSHOT_REF_RE.fullmatch(text)
    if match is None:
        raise ConfigError(f"{text!r} is not a snapshot reference; write agent@v<version>, e.g. main@v1200")
    return match.group(1), int(match.group(2))


def _run_checkpoints(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    ckpts = root / "checkpoints"
    if not ckpts.is_dir():
        raise ConfigError(f"{root} is not a run dir (no checkpoints/ inside); pass runs/<name>")
    return ckpts


def list_snapshots(run_dir: str | Path) -> dict[str, list[CheckpointInfo]]:
    """Stored snapshots of every agent of a run dir (read-only), agents by name, snapshots by version."""
    ckpts = _run_checkpoints(run_dir)
    out = {}
    for agent_dir in sorted(p for p in ckpts.iterdir() if p.is_dir() and not p.name.startswith(".")):
        infos = read_snapshots(agent_dir)
        if infos:
            out[agent_dir.name] = infos
    return out


def pin_snapshot(run_dir: str | Path, agent_id: str, version: int, *, pinned: bool = True) -> Path:
    """Pin (unpin) snapshot ``ckpt_v<version>`` of ``agent_id`` in ``run_dir`` (``set_pinned``; safe while the run
    trains: its retention re-reads pins). ConfigError naming the stored agents or snapshots otherwise."""
    snapshots = list_snapshots(run_dir)
    if agent_id not in snapshots:
        raise ConfigError(f"{run_dir}: no snapshots of agent {agent_id!r}; agents with snapshots: {sorted(snapshots)}")
    by_id = {c.checkpoint_id: c for c in snapshots[agent_id]}
    checkpoint_id = f"ckpt_v{int(version)}"
    if checkpoint_id not in by_id:
        stored = list(by_id)
        raise ConfigError(f"{run_dir}: agent {agent_id!r} has no stored snapshot {checkpoint_id} (newest stored: "
                          f"{stored[-10:]}); see colosseum pin {run_dir} --list")
    set_pinned(by_id[checkpoint_id].path, pinned)
    return by_id[checkpoint_id].path
```

`src/colosseum/cli.py` (after the `bc` command):

```python
@main.command("pin")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("snapshot", required=False)
@click.option("--remove", is_flag=True, default=False, help="Unpin instead of pin")
@click.option("--list", "list_", is_flag=True, default=False, help="List every agent's stored snapshots")
def pin_cmd(run_dir: str, snapshot: str | None, remove: bool, list_: bool) -> None:
    """Pin a stored snapshot (AGENT@vN) of a run dir so retention never evicts it, unpin it (--remove) or list
    the snapshots (--list). Safe while the run trains: its next retention pass sees the pin.

    Exit code: 0 done, 1 config error (not a run dir, unknown agent or snapshot), 2 bad arguments.
    """
    if list_ == (snapshot is not None):
        raise click.UsageError("give either a snapshot (AGENT@vN) or --list")
    if list_ and remove:
        raise click.UsageError("--remove needs a snapshot")
    with _config_errors():
        from colosseum.coordinator.checkpoint_manager import list_snapshots, parse_snapshot_ref, pin_snapshot

        if list_:
            for agent_id, infos in list_snapshots(run_dir).items():
                names = [f"{c.checkpoint_id} (pinned)" if c.meta.get("pinned") is True else c.checkpoint_id
                         for c in infos]
                click.echo(f"{agent_id}: {', '.join(names)}")
            return
        try:
            agent_id, version = parse_snapshot_ref(snapshot)
        except Exception as e:  # noqa: BLE001 - a malformed reference is a usage error
            raise click.BadParameter(str(e), param_hint="'SNAPSHOT'") from e
        path = pin_snapshot(run_dir, agent_id, version, pinned=not remove)
    click.echo(f"{'Unpinned' if remove else 'Pinned'} {agent_id}@v{version} ({path})")
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_pin_cli.py -q`
Expected: all pass (the live test takes about 15 s).

- [ ] **Step 5: Full fast suite + ruff**

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/coordinator/checkpoint_manager.py src/colosseum/cli.py tests/integration/test_sp4_pin_cli.py
git commit -m "feat: colosseum pin (pin, unpin and list stored snapshots of a run)"
git push
```

---
### Task T5.1: Tournament players and engine; shared player loading with `eval` / `record`

Spec block 3 («Участники», «Исполнение», the last bullet of «Исполнение»: shared code with `eval` and the SP3 dedup items) and §8 («`eval` / `record` больше не накладывают `.pt`-настройки frozen-записи на `-a имя=путь`», «`MatchRunner` принимает явный seed на состав»).

Three pieces:
1. **Shared loading in `colosseum.eval`** (dedup of SP3 appendix items): one `parse_agent_spec` for `eval -a`, `record --player/--against` and `tournament -a` (replaces `cli._parse_agent_spec` and `record.parse_player`); one `balanced_match_count` (the "odd count rounded up on competitive layouts" predicate of `cli eval` and `record`); `load_player(..., fixed=)` reuses the `FixedPlayers` that `validate_config` already loaded (frozen agents were loaded twice); `load_frozen(..., use_entry=False)` for user paths (a frozen entry's `.pt` `networks` / `roles` no longer apply to `-a name=path`). `play_lineups` keeps a lineup's `seed` / `block` / `uid` (T1.1's fields; its copies used to drop them), passes `source` to `MatchRunner`, and takes an `(agent, network)`-aware `pool` instead of `models`. `MatchRunner(snapshot_fallback=False)` serves `ckpt_v<N>` seats from the pool itself (no "agent's latest" needed, no silent latest fallback).
2. **`arena.tournament.players`:** `TournamentPlayer` (picklable data) for run snapshots, `-a` checkpoints / `.pt` files and the config's scripted / frozen agents, with the entity names of spec block 2; snapshot selection `all | last:N | top:N | every:K`; `resolve_players` (candidates and field by the rules in its docstring).
3. **`arena.tournament.engine`:** `play_blocks` over spawn worker processes with a lazy LRU model pool.

`arena/tournament/schedule.py` is created here with only `Block`, `STAGES` and `stage_seed` (the engine and the players' tests need the type); T5.2 adds the schedule builders to the same module.

**Files:**
- Modify: `src/colosseum/eval.py` (`parse_agent_spec`, `balanced_match_count`, `load_frozen_player`, `load_eval_model(use_entry=)`, `load_player(fixed=)`, `evaluate(fixed=)`, `play_lineups(pool=, source=)`)
- Modify: `src/colosseum/record.py` (`parse_player` removed: `eval.parse_agent_spec`; `balanced_match_count`; `record(fixed=)`)
- Modify: `src/colosseum/cli.py` (`_parse_agent_spec` removed; `eval` / `record` pass `report.fixed`)
- Modify: `src/colosseum/players/registry.py` (`load_frozen(..., use_entry=True)`; `checkpoint_roles` public)
- Modify: `src/colosseum/core/validation.py` (`ValidationReport.fixed`)
- Modify: `src/colosseum/worker/match_runner.py` (`snapshot_fallback`)
- Create: `src/colosseum/arena/tournament/__init__.py`, `src/colosseum/arena/tournament/players.py`, `src/colosseum/arena/tournament/engine.py`, `src/colosseum/arena/tournament/schedule.py` (`Block`, `STAGES`, `stage_seed` only)
- Modify: `tests/game_helpers.py` (`import os`; `CrashBot`, `PlayerErrorBot`, `write_fake_run`)
- Test: `tests/unit/test_sp4_eval_dedup.py`, `tests/unit/test_sp4_tournament_players.py`, `tests/integration/test_sp4_tournament_engine.py`

**Interfaces:**
- Consumes: `Lineup.seed/block/uid`, `MatchResult.uid/source/seed/block`, `SeatResult.version_start`, `MatchRunner(source=...)` (T1.1); `MatchLog`, `RunInfo`, `MATCH_LOG_FILE`, `MatchLog.open_readonly(...).comparison_table`, `MatchLogReader.close` (T1.2, R3); `bt.fit`, `bt.bootstrap`, `bt.score_summary` (T2.1), `bt.default_options`, `bt.BootstrapData.from_table` (R7, R8); `retention.combine_scores` (T4.1); `checkpoint_manager.read_snapshots`, `read_weights_file`, `read_checkpoint_meta`; `players.registry.load_frozen`, `build_frozen_model`, `make_bot`, `FixedPlayers`, `FrozenSpec`, `BotSpec`; `utils.process.run_child`, `start_process`, `ProcessSupervisor`; `core.threads.configure_torch_threads`.
- Produces (contract T5.1, additions marked *):
  - `players.TournamentPlayer` (`name`, `display`, `kind`, `agent_id`*, `network_id`*, `roles`, `lineage`, `source`*, `version`*, `run`*, `run_agent`*); *`PlayerSource(kind, origin, weights=None, networks=None, bot=None)`; *`RunSelection(path, alias=None, agent=None, snapshots="all")`; *`PlayerRequest(user, runs, field)`; `resolve_candidates` / `resolve_field` are one function here: *`resolve_players(config, spec, request, *, fixed, link=None, rng=None, allow_empty_field=False) -> tuple[list[TournamentPlayer], list[TournamentPlayer]]`; *`entity_hash`, `user_player`, `config_player`, `snapshot_player`, `parse_snapshots`, `select_snapshots`, `top_snapshot_lcb`, `player_to_dict`, `player_from_dict`;
  - `engine.play_blocks(config, spec, players, blocks, *, num_workers, num_envs, deterministic, on_block, log_dir=None, source="tournament") -> None`; *`engine.LazyPlayerPool`, `engine.make_tasks`, `engine.TournamentError`, `MODEL_CACHE_SIZE`, `TASK_MATCHES`;
  - `schedule.Block`, `schedule.STAGES`, `schedule.stage_seed(seed, stage, index)` (moved from T5.2's list to here);
  - *`eval.parse_agent_spec(text, *, option="-a")`, *`eval.balanced_match_count(spec, layouts, num_matches)`, *`eval.load_frozen_player(config, name, path, *, spec, validated=None, use_entry=True) -> (model, FrozenSpec)`; `eval.load_eval_model(..., use_entry=False)`; `eval.load_player(..., fixed=None)`; `eval.evaluate(..., fixed=None)`; `eval.play_lineups(*, env_fn, models=None, lineups, ..., pool=None, source="train")`;
  - *`players.registry.load_frozen(..., use_entry=True)`, *`players.registry.checkpoint_roles(spec, path, meta)` (the former `_checkpoint_roles`);
  - *`ValidationReport.fixed: FixedPlayers | None`;
  - *`MatchRunner(..., snapshot_fallback=True)`;
  - *`record.record(..., fixed=None)`.

- [ ] **Step 1: Check the ground**

```bash
grep -n "seed: int | None = None\|block: str | None\|uid: str" src/colosseum/core/types.py
grep -n "source: str\|uid_prefix\|version_fn" src/colosseum/worker/match_runner.py | head
grep -n "def open_readonly\|def comparison_table\|def close\|def default_options\|class BootstrapData\|def from_table" \
     src/colosseum/arena/matchlog.py src/colosseum/arena/bt.py
grep -rn "parse_player\|_parse_agent_spec" src tests scripts
```

Expected: T1.1's `Lineup` / `MatchResult` fields and `MatchRunner(source=..., uid_prefix=..., version_fn=...)` exist; the Part 1 names exist (else adapt and report: R3, R7, R8); `parse_player` / `_parse_agent_spec` are used only in `cli.py` and `record.py` (if a test imports them, switch it to `eval.parse_agent_spec`).

- [ ] **Step 2: Write the failing tests**

Add to `tests/game_helpers.py` (`import os` at the top with the other imports; the classes and the helper at the end, under a new section header "SP4 (T5.1): tournament support"):

```python
class CrashBot(ScriptedBot):
    """Kills its own process (``os._exit(3)``) at the first ``act``: a tournament worker dying mid-task."""

    def act(self, obs, mask, info):
        os._exit(3)


class PlayerErrorBot(ScriptedBot):
    """Raises at the first ``act``: the match core turns that into a PlayerError with its context."""

    def act(self, obs, mask, info):
        raise RuntimeError("bot failure on purpose")


def write_fake_run(root, game: str = "turns", *, agent: str = "main", versions=(10, 20, 30),
                   agents: dict | None = None, **sections):
    """A run dir as ``colosseum train`` leaves it, without training: ``config.resolved.yaml`` of
    ``make_test_config(game, agents=agents or {agent: {}}, **sections)`` and ``checkpoints/<agent>/ckpt_v<N>`` for
    every version (``write_ckpt_dir``; seed = position, so the weights differ). Returns ``(root, config)``."""
    import yaml

    root = Path(root)
    config = make_test_config(game, agents=agents if agents is not None else {agent: {}}, **sections)
    (root / "checkpoints" / agent).mkdir(parents=True)
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json", by_alias=True), sort_keys=False))
    for i, version in enumerate(versions):
        write_ckpt_dir(root / "checkpoints" / agent, config, agent, policy_version=version, seed=i)
    return root, config
```

Create `tests/unit/test_sp4_eval_dedup.py`:

```python
"""Shared player loading of eval / record / tournament (SP4 T5.1; SP3 appendix dedup items): one -a parser, one
rounding rule, frozen agents loaded once, no .pt overlay of a frozen entry on a user path, play_lineups with
explicit seeds, a pool and a source."""
from __future__ import annotations

import pytest
import torch
from click.testing import CliRunner

import colosseum.players.registry as registry
from colosseum.cli import main
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, make_env
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.eval import balanced_match_count, parse_agent_spec, play_lineups
from colosseum.players import RandomBot
from colosseum.worker.match_runner import MatchRunner
from game_helpers import (
    DictModelPool,
    EliminationFFA,
    TurnTakingGame,
    agent_role_of,
    frozen_agent,
    make_test_config,
    scripted_agent,
    scripted_player,
    write_test_config,
)


def _pt(tmp_path, name, hidden=16, seed=0):
    cfg = make_test_config("turns", networks={"kwargs": {"core": "none", "hidden": hidden}})
    torch.manual_seed(seed)
    _roles, role = agent_role_of(cfg, "agent_0")
    path = tmp_path / f"{name}.pt"
    torch.save(build_model(cfg.get_agent_config("agent_0"), role).state_dict(), path)
    return path


def test_parse_agent_spec_and_the_rounding_rule():
    assert parse_agent_spec("greedy") == ("greedy", None)
    assert parse_agent_spec("a=runs/x/ckpt_v3") == ("a", "runs/x/ckpt_v3")
    for bad in ("", "=x", "a="):
        with pytest.raises(ConfigError, match="expected an agent name"):
            parse_agent_spec(bad, option="--against")
    spec = EliminationFFA(4).spec
    assert balanced_match_count(spec, None, 3) == 4 and balanced_match_count(spec, ["2p"], 4) == 4
    solo = TurnTakingGame().spec
    assert balanced_match_count(solo, ["2p"], 5) == 6


def test_eval_loads_a_frozen_agent_once_and_record_uses_the_shared_parser(tmp_path, monkeypatch):
    pt = _pt(tmp_path, "old")
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns",
                            agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    calls = []
    original = registry.load_frozen
    monkeypatch.setattr(registry, "load_frozen", lambda *a, **k: calls.append(a[1]) or original(*a, **k))
    result = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "old", "-a", "rnd", "-n", "2", "--num-envs", "2"])
    assert result.exit_code == 0, result.output
    assert calls == ["old"]
    bad = CliRunner().invoke(main, ["record", "-c", str(cfg), "--player", "=x", "-o", str(tmp_path / "rec")])
    assert bad.exit_code == 1 and "expected an agent name" in bad.output


def test_a_user_path_ignores_the_frozen_entry_of_the_same_name(tmp_path):
    wide, narrow = _pt(tmp_path, "wide", hidden=32), _pt(tmp_path, "narrow", hidden=16)
    cfg = write_test_config(tmp_path / "cfg.yaml", "turns", agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(wide, networks={"kwargs": {"hidden": 32}})})
    by_name = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", "wide", "-a", "rnd", "-n", "2",
                                        "--num-envs", "2"])
    assert by_name.exit_code == 0, by_name.output                       # the entry's networks for its own path
    by_path = CliRunner().invoke(main, ["eval", "-c", str(cfg), "-a", f"wide={narrow}", "-a", "rnd", "-n", "2",
                                        "--num-envs", "2"])
    assert by_path.exit_code == 0, by_path.output                       # SP4: global networks for a user path


def test_play_lineups_keeps_explicit_seeds_uids_and_the_source():
    lineups = [Lineup("2p", [SeatAssignment("r1", "fixed"), SeatAssignment("r2", "fixed")], seed=1234 + i % 2,
                      block=f"t/A/2p/x/-/{i % 2}", uid=f"t/A/2p/x/-/{i % 2}/r{i // 2}") for i in range(4)]
    models = {"r1": RandomBot(), "r2": RandomBot()}
    results = play_lineups(env_fn=TurnTakingGame, models=models, lineups=lineups, num_envs=2, seed=0,
                           source="tournament")
    assert sorted(r.uid for r in results) == sorted(lu.uid for lu in lineups)
    by_uid = {r.uid: r for r in results}
    assert all(by_uid[lu.uid].seed == lu.seed and by_uid[lu.uid].block == lu.block for lu in lineups)
    assert {r.source for r in results} == {"tournament"}
    # the same seed gives the same bot games (bots are seeded from the episode seed)
    assert [t.rank for t in by_uid[lineups[0].uid].teams] == [t.rank for t in by_uid[lineups[2].uid].teams]
    assert [s.reward for s in by_uid[lineups[0].uid].seats] == [s.reward for s in by_uid[lineups[2].uid].seats]
    with pytest.raises(ValueError, match="exactly one"):
        play_lineups(env_fn=TurnTakingGame, lineups=lineups, num_envs=2)


def test_play_lineups_with_a_pool_serves_snapshots_by_network_id():
    cfg = make_test_config("turns")
    _roles, role = agent_role_of(cfg, "agent_0")
    torch.manual_seed(0)
    v5, v9 = build_model(cfg.get_agent_config("agent_0"), role), build_model(cfg.get_agent_config("agent_0"), role)
    pool = DictModelPool({("main", "ckpt_v5"): v5.eval(), ("main", "ckpt_v9"): v9.eval(),
                          ("rnd", "fixed"): scripted_player(RandomBot, env_spec(cfg))})
    lineups = [Lineup("2p", [SeatAssignment("main", "ckpt_v5"), SeatAssignment("main", "ckpt_v9")], seed=7,
                      uid="u0", block="b0"),
               Lineup("2p", [SeatAssignment("main", "ckpt_v9"), SeatAssignment("rnd", "fixed")], seed=8, uid="u1",
                      block="b1")]
    results = play_lineups(env_fn=lambda: make_env(cfg), pool=pool, lineups=lineups, num_envs=2, seed=0,
                           source="tournament")
    versions = {r.uid: [s.version_start for s in r.seats] for r in results}
    assert versions == {"u0": [5, 9], "u1": [9, None]}


def test_match_runner_without_the_snapshot_fallback_needs_the_snapshot_itself():
    from colosseum.envs.vector import VectorEnv

    pool = DictModelPool({("rnd", "fixed"): scripted_player(RandomBot, TurnTakingGame().spec)})
    lineup = Lineup("2p", [SeatAssignment("main", "ckpt_v5", collect=False), SeatAssignment("rnd", "fixed", False)])
    vec_env = VectorEnv(TurnTakingGame, 1)
    try:
        with pytest.raises(ValueError, match="ckpt_v5"):
            MatchRunner(vec_env=vec_env, lineups=[lineup], models=pool, snapshot_fallback=False)
    finally:
        vec_env.close()
```

Create `tests/unit/test_sp4_tournament_players.py`:

```python
"""Tournament players (SP4 T5.1, spec blocks 2 and 3): entity names, kinds, snapshot selection (all, last, every,
top by the run journal's lower bound), candidates vs field, the lazy LRU pool, (de)serialization."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
from colosseum.arena.tournament.engine import LazyPlayerPool
from colosseum.arena.tournament.players import (
    PlayerRequest,
    RunSelection,
    config_player,
    entity_hash,
    parse_snapshots,
    player_from_dict,
    player_to_dict,
    resolve_players,
    select_snapshots,
    user_player,
)
from colosseum.coordinator.checkpoint_manager import read_snapshots
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.types import MatchResult, SeatResult, TeamResult
from colosseum.players.registry import load_fixed_players
from colosseum.worker.match_runner import ScriptedPlayer
from game_helpers import make_test_config, scripted_agent, write_ckpt_dir, write_fake_run


def _cfg(**agents):
    return make_test_config("turns", agents={"main": {}, "rnd": scripted_agent(), **agents})


def test_user_and_config_players_have_the_entity_names_of_block_2(tmp_path):
    cfg = _cfg()
    spec = env_spec(cfg)
    ckpt = write_ckpt_dir(tmp_path / "x", cfg, "main", policy_version=5)
    p = user_player(cfg, spec, "prev", str(ckpt), validated=set())
    assert p.name == f"prev#{entity_hash(ckpt)}" and len(entity_hash(ckpt)) == 8 and p.display == "prev"
    assert (p.agent_id, p.network_id, p.kind, p.lineage) == (p.name, "fixed", "checkpoint", p.name)
    assert p.source.weights == str(ckpt / "model.pt") and p.roles == ("player",)
    bot = config_player(cfg, spec, "rnd", load_fixed_players(cfg, spec))
    assert (bot.name, bot.agent_id, bot.network_id, bot.kind) == ("rnd", "rnd", "fixed", "scripted")
    assert bot.source.bot.class_path == "colosseum.players.RandomBot"
    with pytest.raises(ConfigError, match="trainable"):
        config_player(cfg, spec, "main", load_fixed_players(cfg, spec))
    assert player_from_dict(player_to_dict(p)) == p and player_from_dict(player_to_dict(bot)) == bot


def test_snapshot_selection_rules(tmp_path):
    root, _cfg_ = write_fake_run(tmp_path / "run", versions=(10, 20, 30, 40, 50))
    infos = read_snapshots(root / "checkpoints" / "main")
    pick = lambda text, lcb=None: [c.policy_version for c in select_snapshots(infos, *parse_snapshots(text), lcb=lcb)]  # noqa: E731
    assert pick("all") == [10, 20, 30, 40, 50] and pick("last:2") == [40, 50] and pick("every:2") == [10, 30, 50]
    assert pick("top:2", {"ckpt_v20": 3.0, "ckpt_v40": 9.0, "ckpt_v10": 1.0}) == [20, 40]
    for bad in ("last", "last:0", "top:x", "first:2"):
        with pytest.raises(ConfigError, match="--snapshots"):
            parse_snapshots(bad)


def _match(i, version, wins) -> MatchResult:
    """A 1v1 training match of main@v<version> against rnd; sides alternate."""
    side = i % 2
    seats = [None, None]
    seats[side] = SeatResult(side, "player", side, "main", "latest", 0.0, version_start=version, version_end=version)
    seats[1 - side] = SeatResult(1 - side, "player", 1 - side, "rnd", "fixed", 0.0)
    ranks = {side: 1.0 if wins else 2.0, 1 - side: 2.0 if wins else 1.0}
    return MatchResult(match_id=f"m{i}", layout="2p", outcome_kind="wdl", seats=seats,
                       teams=[TeamResult(t, ranks[t], 0.0) for t in (0, 1)], episode_length=6, uid=f"u{version}-{i}",
                       source="train", env_steps=i)


def test_top_snapshots_come_from_the_run_journal_and_need_it(tmp_path):
    root, cfg = write_fake_run(tmp_path / "run", versions=(10, 20, 30), agents={"main": {}, "rnd": scripted_agent()},
                               checkpoint={"interval": 10})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    request = PlayerRequest(user=(), runs=(RunSelection(root, snapshots="top:1"),), field=None)
    with pytest.raises(ConfigError, match="match log"):
        resolve_players(cfg, spec, request, fixed=fixed, rng=np.random.default_rng(0))
    rng = np.random.default_rng(0)
    log = MatchLog(root / MATCH_LOG_FILE, RunInfo("r", "run", None, cfg.env.env_class, "h", 10))
    rates = {10: 0.2, 20: 0.85, 30: 0.5}
    log.append([_match(i, v, rng.random() < rate) for v, rate in rates.items() for i in range(300)])
    log.close()
    candidates, field = resolve_players(cfg, spec, request, fixed=fixed, rng=np.random.default_rng(1))
    assert [p.name for p in candidates] == ["main@v20"] and [p.name for p in field] == ["rnd"]


def test_candidates_and_field_aliases_and_explicit_field_members(tmp_path):
    first, cfg = write_fake_run(tmp_path / "a", versions=(10, 20), agents={"main": {}, "rnd": scripted_agent()})
    second, _ = write_fake_run(tmp_path / "b", versions=(5,), agents={"main": {}, "rnd": scripted_agent()})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    ckpt = write_ckpt_dir(tmp_path / "x", cfg, "main", policy_version=7)
    request = PlayerRequest(user=(("prev", str(ckpt)), ("rnd", None)),
                            runs=(RunSelection(first), RunSelection(second, alias="old")),
                            field=("rnd", "main@v10", "old/main@v5"))
    candidates, field = resolve_players(cfg, spec, request, fixed=fixed)
    assert [p.name for p in candidates] == [f"prev#{entity_hash(ckpt)}", "main@v20"]
    assert [p.name for p in field] == ["rnd", "main@v10", "old/main@v5"]
    old = field[2]
    assert (old.agent_id, old.network_id, old.lineage, old.version, old.run_agent) == (
        "old/main", "ckpt_v5", "old/main", 5, "main")
    default = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first),), field=None), fixed=fixed)
    assert [p.name for p in default[1]] == ["rnd"]
    with pytest.raises(ConfigError, match="nobody"):
        resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first),), field=("nobody",)), fixed=fixed)
    with pytest.raises(ConfigError, match="field is empty"):
        resolve_players(make_test_config("turns", agents={"main": {}}), spec,
                        PlayerRequest(user=(), runs=(RunSelection(first),), field=None), fixed=fixed)
    with pytest.raises(ConfigError, match="alias"):
        resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(first), RunSelection(second, alias="a/b")),
                                                 field=None), fixed=fixed)


def test_the_lazy_pool_loads_on_first_use_and_keeps_reserved_models(tmp_path):
    root, cfg = write_fake_run(tmp_path / "run", versions=(1, 2, 3), agents={"main": {}, "rnd": scripted_agent()})
    spec = env_spec(cfg)
    fixed = load_fixed_players(cfg, spec)
    candidates, field = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(root),), field=None),
                                        fixed=fixed)
    pool = LazyPlayerPool([*candidates, *field], spec, capacity=1)
    assert pool.loads == 0 and isinstance(pool.get("rnd", "fixed"), ScriptedPlayer)
    pool.reserve({("main", "ckpt_v1"), ("main", "ckpt_v2")})
    assert pool.loads == 2 and set(pool.cached()) == {("main", "ckpt_v1"), ("main", "ckpt_v2")}
    assert not pool.get("main", "ckpt_v1").training                     # eval mode
    pool.get("main", "ckpt_v1")
    assert pool.loads == 2                                               # cached
    pool.reserve({("main", "ckpt_v3")})
    assert pool.cached() == [("main", "ckpt_v3")]                         # capacity 1: the others leave
    assert pool.get("nobody", "fixed") is None
    with pytest.raises(ValueError, match="nobody"):
        pool.reserve({("nobody", "fixed")})
```

Create `tests/integration/test_sp4_tournament_engine.py`:

```python
"""The tournament engine (SP4 T5.1, spec block 3): whole blocks in two spawn workers, explicit seeds, results
back in the main process, errors with their context, a dead worker named with its log."""
from __future__ import annotations

import pytest

from colosseum.arena.tournament.engine import TournamentError, make_tasks, play_blocks
from colosseum.arena.tournament.players import PlayerRequest, RunSelection, resolve_players
from colosseum.arena.tournament.schedule import Block, stage_seed
from colosseum.core.errors import PlayerError
from colosseum.core.registry import env_spec
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.players.registry import load_fixed_players
from game_helpers import scripted_agent, write_fake_run


def _block(index, a, b, seed_index=None, prefix="t") -> Block:
    """A 1v1 block: ``a`` vs ``b`` on both sides (players as (agent_id, network_id))."""
    bid = f"{prefix}/A/2p/{a[0]}{a[1]}/{b[0]}/{index}"
    seed = stage_seed(0, "A", index if seed_index is None else seed_index)
    lineups = tuple(Lineup("2p", [SeatAssignment(*first, collect=False), SeatAssignment(*second, collect=False)],
                           seed=seed, block=bid, uid=f"{bid}/r{r}")
                    for r, (first, second) in enumerate([(a, b), (b, a)]))
    return Block(id=bid, stage="A", layout="2p", candidate=f"{a[0]}@{a[1]}", composition=(b[0],), index=index,
                 seed=seed, lineups=lineups, rotations=(0, 1))


def _setup(tmp_path, **bots):
    root, cfg = write_fake_run(tmp_path / "run", versions=(5, 9),
                               agents={"main": {}, "rnd": scripted_agent(), **bots})
    spec = env_spec(cfg)
    candidates, field = resolve_players(cfg, spec, PlayerRequest(user=(), runs=(RunSelection(root),),
                                                                 field=("rnd", *bots)),
                                        fixed=load_fixed_players(cfg, spec))
    return cfg, spec, {p.name: p for p in [*candidates, *field]}


def test_blocks_are_played_whole_by_two_workers(tmp_path):
    cfg, spec, players = _setup(tmp_path)
    blocks = [_block(k, ("main", "ckpt_v5"), ("rnd", "fixed")) for k in range(6)]
    blocks += [_block(k, ("main", "ckpt_v9"), ("rnd", "fixed")) for k in range(6)]
    blocks.append(_block(99, ("main", "ckpt_v9"), ("rnd", "fixed"), seed_index=0))   # same seed as block 0 of v9
    seen: dict[str, list] = {}
    play_blocks(cfg, spec, players, blocks, num_workers=2, num_envs=2, deterministic=True,
                on_block=lambda block_id, results: seen.setdefault(block_id, []).extend(results),
                log_dir=tmp_path / "logs")
    assert set(seen) == {b.id for b in blocks}
    for b in blocks:
        results = seen[b.id]
        assert [r.uid for r in results] == sorted(lu.uid for lu in b.lineups)
        assert {r.seed for r in results} == {b.seed} and {r.source for r in results} == {"tournament"}
        assert {r.block for r in results} == {b.id} and all(r.env_steps is None for r in results)
        snapshot_seats = [s for r in results for s in r.seats if s.agent_id == "main"]
        version = int(b.lineups[0].seats[0].network_id.removeprefix("ckpt_v"))
        assert {s.version_start for s in snapshot_seats} == {version}
    same = [seen[blocks[6].id], seen[blocks[-1].id]]                      # common seed, deterministic policy
    assert [[t.rank for t in r.teams] for r in same[0]] == [[t.rank for t in r.teams] for r in same[1]]
    assert (tmp_path / "logs" / "tournament-worker-0.log").is_file()
    assert [len(t) for t in make_tasks(blocks, num_envs=2)] == [4, 4, 4, 1]   # >= 4 * 2 matches per task


def test_a_bot_error_in_a_worker_is_a_player_error_in_the_main_process(tmp_path):
    cfg, spec, players = _setup(tmp_path, bad=scripted_agent("game_helpers.PlayerErrorBot"))
    blocks = [_block(0, ("main", "ckpt_v5"), ("bad", "fixed"))]
    with pytest.raises(PlayerError, match="bad"):
        play_blocks(cfg, spec, players, blocks, num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None, log_dir=tmp_path / "logs")


def test_a_dead_worker_names_its_log(tmp_path):
    cfg, spec, players = _setup(tmp_path, crash=scripted_agent("game_helpers.CrashBot"))
    blocks = [_block(0, ("main", "ckpt_v5"), ("crash", "fixed"))]
    with pytest.raises(TournamentError, match="tournament-worker-0"):
        play_blocks(cfg, spec, players, blocks, num_workers=1, num_envs=1, deterministic=True,
                    on_block=lambda *_: None, log_dir=tmp_path / "logs")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_eval_dedup.py tests/unit/test_sp4_tournament_players.py tests/integration/test_sp4_tournament_engine.py -q`
Expected: `ImportError: cannot import name 'balanced_match_count'` / `No module named 'colosseum.arena.tournament'`.

- [ ] **Step 4: Shared loading (`players/registry.py`, `eval.py`, `record.py`, `cli.py`, `validation.py`)**

`src/colosseum/players/registry.py`:
1. Rename `_checkpoint_roles` to `checkpoint_roles` (public; update its two call sites) and add it to `__all__`.
2. `load_frozen(config, agent_id, path, spec, *, use_entry: bool = True)`: after `entry = ...` add
```python
    if not use_entry and isinstance(entry, FrozenAgent):
        entry = None    # a user path (eval/record/tournament -a name=path): the frozen entry's .pt settings do not apply
```
   and extend the docstring: "``use_entry=False`` (a path given on the command line): a frozen entry of the same name is ignored (SP4 §8); a trainable entry's architecture still applies to a ``.pt``."

`src/colosseum/core/validation.py`: `ValidationReport` gains `fixed: Any = None  # the FixedPlayers validate loaded (reused by eval / record / tournament)`; in `validate_config` set `report.fixed = fixed` right after `fixed = load_fixed_players(...)`.

`src/colosseum/eval.py`:
1. Imports: `from colosseum.worker.match_runner import EpisodeEnd, MatchObserver, MatchRunner, PlayerPool, ScriptedPlayer`.
2. After `SCORE_CI_METHOD`:
```python
def parse_agent_spec(text: str, *, option: str = "-a") -> tuple[str, str | None]:
    """``name`` (a scripted or frozen agent of the config) or ``name=path``: the player syntax of ``eval -a``,
    ``record --player`` / ``--against`` and ``tournament -a``. ConfigError otherwise."""
    name, sep, path = text.partition("=")
    if not name or (sep and not path):
        raise ConfigError(f"{option} {text!r}: expected an agent name of the config or name=path")
    return name, (path if sep else None)


def balanced_match_count(spec: GameSpec, layouts: Sequence[str] | None, num_matches: int) -> int:
    """``num_matches`` rounded up to even when one of ``layouts`` (default: every layout of the game) has two or
    more teams, so both players of a pair play every side equally often (SP3 ruling #32). Callers that play
    pairs use it and tell the user when it changed the count."""
    chosen = list(layouts) if layouts else list(spec.layouts)
    competitive = any(spec.num_teams(name) >= 2 for name in chosen if name in spec.layouts)
    return num_matches + 1 if competitive and num_matches % 2 else num_matches
```
3. `play_lineups`: new signature and copies:
```python
def play_lineups(
    *,
    env_fn: Callable[[], MultiAgentEnv],
    models: Mapping[str, PolicyModel | ScriptedBot | ScriptedPlayer] | None = None,
    lineups: Sequence[Lineup],
    num_envs: int = 8,
    seed: int | None = None,
    deterministic: bool = False,
    max_idle_steps: int = 1000,
    observer: MatchObserver | None = None,
    pool: PlayerPool | None = None,
    source: str = "train",
) -> list[MatchResult]:
```
   docstring additions: "Exactly one of ``models`` (``agent_id`` -> player for any network id) and ``pool`` (a ``PlayerPool`` keyed by ``(agent_id, network_id)``: the tournament's; its models must already be in eval mode, ``play_lineups`` does not touch them). A lineup's ``seed`` / ``block`` / ``uid`` are kept (``MatchRunner`` resets with the lineup's seed and copies block and uid into the result); ``source`` is the results' ``source``. Snapshot seats are served by the pool itself (``snapshot_fallback=False``)." Body changes:
```python
    if (models is None) == (pool is None):
        raise ValueError("play_lineups: pass exactly one of models and pool")
    lineups = [dataclasses.replace(lineup, seats=[dataclasses.replace(seat, collect=False) for seat in lineup.seats])
               for lineup in lineups]
    if not lineups:
        return []
    if models is not None:
        unknown = sorted({seat.agent_id for lineup in lineups for seat in lineup.seats} - set(models))
        if unknown:
            raise ValueError(f"play_lineups: lineups use unknown agents {unknown}")
    if num_envs < 1:
        raise ValueError(f"play_lineups: num_envs must be >= 1, got {num_envs}")
    neural = [m for m in models.values() if isinstance(m, PolicyModel)] if models is not None else []
```
   and inside the `try` that builds the runner:
```python
                if models is not None:
                    players = {name: ScriptedPlayer(functools.partial(_copy_bot, player, vec_env.spec))
                               if isinstance(player, ScriptedBot) else player for name, player in models.items()}
                    served: PlayerPool = _FixedModels(players)
                else:
                    served = pool
                runner = MatchRunner(vec_env=vec_env, lineups=lineups[:n], models=served, observer=collector,
                                     seed=seed, max_idle_steps=max_idle_steps, deterministic=deterministic,
                                     context="eval, ", match_id_prefix="eval", source=source,
                                     snapshot_fallback=False)
```
   (If T1.1 already copies lineups with `dataclasses.replace`, keep its form.)
4. Loading:
```python
def load_frozen_player(config: ColosseumConfig, name: str, path: str | Path, *, spec: GameSpec,
                       validated: set[str] | None = None, use_entry: bool = True) -> tuple[PolicyModel, Any]:
    """``(model in eval mode, FrozenSpec)`` of a checkpoint dir or ``.pt`` (``players.registry.load_frozen`` +
    ``build_frozen_model``), its architecture checked once per distinct (networks, roles) as in
    ``validate_config`` (keys in ``validated`` are skipped, new ones added; None: always check)."""
    from colosseum.players.registry import build_frozen_model, load_frozen

    frozen = load_frozen(config, name, path, spec, use_entry=use_entry)
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
    return model, frozen
```
   `load_eval_model(config, name, path, *, spec=None, validated=None, use_entry=False)`: keep the docstring (add "``use_entry`` as in ``load_frozen``: False for paths given on the command line") and make the body
```python
    from colosseum.core.registry import env_spec

    spec = spec if spec is not None else env_spec(config)
    model, frozen = load_frozen_player(config, name, path, spec=spec, validated=validated, use_entry=use_entry)
    return model, list(frozen.roles)
```
   `load_player(config, name, path, *, spec, validated=None, option="-a", fixed=None)`:
```python
    from colosseum.players.registry import BotSpec, build_frozen_model, make_bot, resolve_player_roles

    if path is None:
        entry = config.agent_entry(name)          # ConfigError for an unknown name
        if entry.kind == "trainable":
            raise ConfigError(f"{option} {name}: '{name}' is a trainable agent, whose weights are not in the config; "
                              f"use {option} {name}=<checkpoint dir or .pt>")
        if fixed is not None and name in fixed.bots:      # validate_config built and played it already
            return ScriptedPlayer(functools.partial(make_bot, fixed.bots[name], spec)), list(fixed.roles[name])
        if fixed is not None and name in fixed.frozen:    # loaded and checked by validate_config: not again
            frozen = fixed.frozen[name]
            return build_frozen_model(config, frozen, spec), list(frozen.roles)
        if entry.kind == "scripted":
            bot = BotSpec(entry.class_path, dict(entry.kwargs))
            make_bot(bot, spec)                    # a bad class or kwargs fail here, as a ConfigError
            roles = list(resolve_player_roles(config, spec, [name])[name])   # other fixed agents are not read
            return ScriptedPlayer(functools.partial(make_bot, bot, spec)), roles
        p = Path(entry.path)
        if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
            raise ConfigError(f"agents.{name}.path={entry.path!r}: expected a checkpoint dir or a .pt file")
        return load_eval_model(config, name, entry.path, spec=spec, validated=validated, use_entry=True)
    p = Path(path)
    if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
        raise ConfigError(f"{option} {name}={path}: expected a checkpoint dir or a .pt file")
    return load_eval_model(config, name, path, spec=spec, validated=validated, use_entry=False)
```
   `evaluate(..., num_envs=8, fixed=None)` passes `fixed=fixed` to `load_player`.

`src/colosseum/record.py`: remove `parse_player` (and from `__all__`); import `balanced_match_count, parse_agent_spec` from `colosseum.eval`; `record(..., decisions_per_file=DECISIONS_PER_FILE, fixed=None)`; in `record`:
```python
    p_name, p_path = parse_agent_spec(player, option="--player")
    opponents = [parse_agent_spec(text, option="--against") for text in against]
    ...
        models[name], roles[name] = load_player(config, name, path, spec=spec, validated=validated, option=option,
                                                fixed=fixed)
    if opponents:
        rounded = balanced_match_count(spec, layouts, num_matches)
        if rounded != num_matches:
            logger.info("num_matches %d is odd; rounded up, using %d per opponent and layout so the player plays "
                        "every side equally often", num_matches, rounded)
            num_matches = rounded
```
   (replacing the old inline predicate; the docstring sentence about rounding stays.)

`src/colosseum/cli.py`:
1. Remove `_parse_agent_spec`.
2. `eval_cmd` body inside `_config_errors()`:
```python
        from colosseum.core.config import load_config
        from colosseum.core.errors import ConfigError
        from colosseum.core.registry import env_spec, validate_config
        from colosseum.eval import balanced_match_count, evaluate, parse_agent_spec

        cfg = load_config(config, _parse_overrides(overrides) or None)
        try:
            specs = [parse_agent_spec(spec) for spec in agents]
        except ConfigError as e:
            raise click.BadParameter(str(e), param_hint="'--agent'") from e
        report_v = validate_config(cfg, scope="play", players=[name for name, path in specs if path is None])
        ...  (duplicate-name and path checks unchanged)
        spec = env_spec(cfg)
        if len(specs) > 1:
            rounded = balanced_match_count(spec, list(layouts) or None, num_matches)
            if rounded != num_matches:
                click.echo(f"Note: --num-matches {num_matches} is odd; using {rounded} per pair so every "
                           f"agent plays every side equally often.", err=True)
                num_matches = rounded
        report = evaluate(cfg, dict(specs), layouts=list(layouts) or None, num_matches=num_matches, seed=seed,
                          deterministic=deterministic, num_envs=num_envs, fixed=report_v.fixed)
```
3. `record_cmd`:
```python
        from colosseum.eval import EvalReport, parse_agent_spec
        from colosseum.record import RECORD_FILE, record

        cfg = load_config(config, _parse_overrides(overrides) or None)
        named = [parse_agent_spec(text, option=opt) for opt, text in (("--player", player),
                                                                       *(("--against", a) for a in against))]
        report_v = validate_config(cfg, scope="play", players=[name for name, path in named if path is None])
        content = record(cfg, player, list(against), layouts=list(layouts) or None, num_matches=num_matches,
                         output=output, num_envs=num_envs, seed=seed, deterministic=deterministic,
                         fixed=report_v.fixed)
```

- [ ] **Step 5: `MatchRunner(snapshot_fallback=...)`**

`src/colosseum/worker/match_runner.py`:
1. `__init__` gains `snapshot_fallback: bool = True` (after `match_id_prefix` and T1.1's new arguments); store `self._snapshot_fallback = bool(snapshot_fallback)`.
2. `_check_lineup`: `needed = net if net in (LATEST_NETWORK_ID, FIXED_NETWORK_ID) or not self._snapshot_fallback else LATEST_NETWORK_ID`.
3. `_resolve`: right after `self._check_lineup(lineup, env)`:
```python
        if not self._snapshot_fallback:   # the pool serves every seat itself (eval, tournament, pulse)
            return dataclasses.replace(lineup, seats=[dataclasses.replace(seat) for seat in lineup.seats])
```
   (`import dataclasses` at the top.)
4. Module docstring, the fallback paragraph: append "With ``snapshot_fallback=False`` (``play_lineups``) a snapshot seat must be in the pool itself (ValueError otherwise) and is never replaced."

- [ ] **Step 6: `arena/tournament/__init__.py` and `schedule.py` (first part)**

`src/colosseum/arena/tournament/__init__.py`:
```python
"""``colosseum tournament`` libraries (SP4 spec block 3): ``players`` (who plays, entity names), ``schedule``
(blocks, rotations, compositions, seed domains), ``engine`` (worker processes), ``stats`` (scores, the paired
stratified bootstrap, ``confirmed``), ``modes`` (gauntlet / round-robin / halving / select, the report) and ``run``
(the tournament dir, resume, the journal). Nothing here touches the training launcher; SP5's hub can call it."""
```

`src/colosseum/arena/tournament/schedule.py` (T5.2 extends the docstring and adds the builders):
```python
"""Tournament schedules (SP4 spec block 3). T5.1: the ``Block`` type and the seed domains; T5.2 adds the builders.

Seeds: ``stage_seed(seed, stage, index)`` puts the stage's domain code in the top two bits of a 32-bit seed and
a keyed bijection of ``index`` in the low 30 bits, so the domains A (screening), B (confirmation) and P (pulse)
never share a seed and no seed repeats within a domain (indices < 2**30). Every candidate gets the same seed
for the same (stage, index): common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colosseum.core.types import Lineup

STAGES = ("A", "B", "P")
_STAGE_CODE = {"A": 1, "B": 2, "P": 3}
_LOW_BITS = 30
_LOW_MASK = (1 << _LOW_BITS) - 1


@dataclass(frozen=True)
class Block:
    """One seed, one composition and every admissible rotation (T5.2): the unit of scheduling, resume and
    bootstrap. ``lineups[i]`` is rotation ``rotations[i]``; every lineup carries ``seed``, ``block`` (= ``id``)
    and ``uid`` (= ``id + "/r<rotation>"``)."""

    id: str
    stage: str
    layout: str
    candidate: str
    composition: tuple[str, ...]
    index: int
    seed: int
    lineups: tuple[Lineup, ...]
    rotations: tuple[int, ...]

    @property
    def stratum(self) -> str:
        """The composition as one key: field members joined by ``+`` (``-`` for the homogeneous team)."""
        return "+".join(self.composition) or "-"


def stage_seed(seed: int, stage: str, index: int) -> int:
    """The episode seed of block ``index`` of ``stage`` (module docstring: disjoint domains, no repeats)."""
    if stage not in _STAGE_CODE:
        raise ValueError(f"stage_seed: stage must be one of {STAGES}, got {stage!r}")
    if not 0 <= int(index) <= _LOW_MASK:
        raise ValueError(f"stage_seed: index must be in [0, 2**30), got {index}")
    code = _STAGE_CODE[stage]
    a, b, c = (int(x) for x in np.random.SeedSequence([int(seed) % 2**63, code]).generate_state(3, np.uint32))
    x = (int(index) * (a | 1) + b) & _LOW_MASK      # odd multiplier + offset: a bijection mod 2**30
    x ^= x >> 15                                     # xorshift: a bijection
    x = (x * (c | 1)) & _LOW_MASK
    x ^= x >> 13
    return (code << _LOW_BITS) | x
```

- [ ] **Step 7: `arena/tournament/players.py`**

```python
"""Tournament players (SP4 spec blocks 2 and 3): who can play, under which entity name, built how.

Kinds and names (the name is the entity key the journal computes from the seat, spec block 2):
- ``snapshot``: ``ckpt_v<N>`` of a run (``--run [alias=]runs/x --agent A``): name ``A@v<N>`` for the first
  ``--run``, ``<alias>/A@v<N>`` for the others (alias: the given one, else the run dir's name); seat
  ``(agent_id="[alias/]A", network_id="ckpt_v<N>")``, lineage ``[alias/]A``; architecture and roles from its
  ``meta.json`` (``networks`` absent: the run config's agent); every distinct architecture is checked once with
  that snapshot's weights (``eval``'s model check);
- ``checkpoint`` / ``weights``: ``-a name=path`` (a checkpoint dir / a ``.pt``): name ``name#<hash8>`` (the first
  8 hex digits of sha256 of the resolved path), display ``name``; seat ``(name#hash8, "fixed")``; loaded like
  ``eval -a name=path`` (``eval.load_frozen_player``, ``use_entry=False``);
- ``scripted`` / ``frozen``: an agent of the config (``-a name``, a ``--field`` name, the default field): name
  ``name``, seat ``(name, "fixed")``.

Candidates and field (``resolve_players``; proposed rulings, Contract notes D3):
- the field is ``--field`` (names in order), else every scripted / frozen agent of the config that is not an
  ``-a name`` candidate; a ``--field`` name is, in this order, an ``-a`` player's name, a scripted / frozen agent
  of the config, or a snapshot reference ``[alias/]agent@v<N>`` of a ``--run`` (any stored snapshot, not only
  the selected ones);
- candidates are every ``-a`` player and every selected run snapshot that is not in the field;
- an empty candidate list, an empty field (except round-robin), a duplicate name, an unknown ``--field`` name,
  two runs with one alias, an alias that is not a path component: ConfigError.

Snapshot selection (``--snapshots``): ``all``; ``last:N`` (newest N); ``every:K`` (every K-th stored snapshot,
counting from the newest, which is always in); ``top:N`` (the N best by the lower bound of the BT rating from
the run's journal, ``source`` train and pulse, bootstrap; spec block 3). ``top:N`` needs ``runs/x/matches.db``
(an SP3 run dir has none: ConfigError).

``TournamentPlayer`` is picklable data (no models, no bot instances); ``player_to_dict`` / ``player_from_dict``
store it in ``tournament.yaml``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from colosseum.coordinator.checkpoint_manager import (
    MODEL_FILE,
    CheckpointInfo,
    read_checkpoint_meta,
    read_snapshots,
    read_weights_file,
)
from colosseum.core.config import ColosseumConfig, NetworkConfig, check_agent_id, check_path_component, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.roles import agent_role_spec
from colosseum.core.types import FIXED_NETWORK_ID
from colosseum.envs.game import GameSpec
from colosseum.players.registry import BotSpec, FixedPlayers, FrozenSpec, build_frozen_model, checkpoint_roles

PlayerKind = Literal["snapshot", "checkpoint", "weights", "scripted", "frozen"]
ENTITY_HASH_LEN = 8
_SNAPSHOT_RE = re.compile(r"(?:(?P<alias>[^/@]+)/)?(?P<agent>[A-Za-z0-9_][A-Za-z0-9_-]*)@v(?P<version>\d+)")
_SELECTION_RE = re.compile(r"(last|top|every):([1-9]\d*)")


@dataclass(frozen=True)
class PlayerSource:
    """How a worker builds the player: ``model`` (``weights`` file + ``networks`` dump) or ``bot`` (``BotSpec``).
    ``origin`` names it in messages and reports."""

    kind: Literal["model", "bot"]
    origin: str
    weights: str | None = None
    networks: dict[str, Any] | None = None
    bot: BotSpec | None = None


@dataclass(frozen=True)
class TournamentPlayer:
    name: str
    display: str
    kind: PlayerKind
    agent_id: str
    network_id: str
    roles: tuple[str, ...]
    lineage: str
    source: PlayerSource
    version: int | None = None
    run: str | None = None          # the --run dir of a snapshot (pin target)
    run_agent: str | None = None    # its agent id inside that run


@dataclass(frozen=True)
class RunSelection:
    path: Path
    alias: str | None = None
    agent: str | None = None
    snapshots: str = "all"


@dataclass(frozen=True)
class PlayerRequest:
    user: tuple[tuple[str, str | None], ...]     # parsed -a specs, in command-line order
    runs: tuple[RunSelection, ...]
    field: tuple[str, ...] | None                # --field names; None = the default field


def entity_hash(path: str | Path) -> str:
    """First ``ENTITY_HASH_LEN`` hex digits of sha256 of the resolved path (the ``#<hash>`` of ``-a name=path``)."""
    return hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:ENTITY_HASH_LEN]


def _weights_file(path: Path) -> Path:
    return path / MODEL_FILE if path.is_dir() else path


def user_player(config: ColosseumConfig, spec: GameSpec, name: str, path: str, *,
                validated: set[str]) -> TournamentPlayer:
    """``-a name=path`` (module docstring); the weights are loaded and the architecture checked here once."""
    from colosseum.eval import load_frozen_player

    check_agent_id(name)
    p = Path(path)
    if not (p.is_dir() or (p.is_file() and p.suffix == ".pt")):
        raise ConfigError(f"-a {name}={path}: expected a checkpoint dir or a .pt file")
    _model, frozen = load_frozen_player(config, name, p, spec=spec, validated=validated, use_entry=False)
    key = f"{name}#{entity_hash(p)}"
    return TournamentPlayer(
        name=key, display=name, kind="checkpoint" if p.is_dir() else "weights", agent_id=key,
        network_id=FIXED_NETWORK_ID, roles=tuple(frozen.roles), lineage=key,
        source=PlayerSource("model", origin=str(p), weights=str(_weights_file(p)), networks=dict(frozen.networks)))


def config_player(config: ColosseumConfig, spec: GameSpec, name: str, fixed: FixedPlayers) -> TournamentPlayer:
    """A scripted or frozen agent of the config (``fixed``: what ``validate_config`` loaded)."""
    entry = config.agent_entry(name)                  # ConfigError for an unknown name
    if entry.kind == "trainable":
        raise ConfigError(f"'{name}' is a trainable agent of the config, whose weights are not in the config; "
                          f"name a checkpoint with -a {name}=<dir or .pt> or a snapshot with --run ... / "
                          f"--field {name}@v<N>")
    if name in fixed.bots:
        return TournamentPlayer(name=name, display=name, kind="scripted", agent_id=name, network_id=FIXED_NETWORK_ID,
                                roles=tuple(fixed.roles[name]), lineage=name,
                                source=PlayerSource("bot", origin=f"agents.{name}", bot=fixed.bots[name]))
    if name not in fixed.frozen:
        raise ConfigError(f"agent {name!r} was not loaded (validate_config scope 'play' with players={name!r})")
    frozen = fixed.frozen[name]
    return TournamentPlayer(name=name, display=name, kind="frozen", agent_id=name, network_id=FIXED_NETWORK_ID,
                            roles=tuple(frozen.roles), lineage=name,
                            source=PlayerSource("model", origin=frozen.source,
                                                weights=str(_weights_file(Path(frozen.source))),
                                                networks=dict(frozen.networks)))


def snapshot_player(ckpt_dir: str | Path, *, agent: str, prefix: str, spec: GameSpec, run_config: ColosseumConfig,
                    run: str | Path, validated: set[str]) -> TournamentPlayer:
    """One run snapshot (module docstring). ``prefix`` is ``""`` or ``"<alias>/"``; ``ckpt_dir`` may be a linked
    copy (T5.4) whose name is ``ckpt_v<N>``."""
    path = Path(ckpt_dir)
    meta = read_checkpoint_meta(path)
    version = int(meta["policy_version"])
    roles = checkpoint_roles(spec, path, meta)
    raw = meta.get("networks")
    try:
        networks = NetworkConfig.model_validate(raw) if raw is not None else \
            run_config.get_agent_config(agent).networks
    except ValidationError as e:
        raise ConfigError(f"Checkpoint {path}: invalid meta.json networks:\n{e}") from e
    dump = networks.model_dump(mode="json", by_alias=True)
    key = json_key(dump, roles)
    if key not in validated:          # one model check per architecture, with this snapshot's weights
        from colosseum.core.validation import _check_model

        frozen = FrozenSpec(agent_id=agent, roles=tuple(roles), networks=dump,
                            model_state=read_weights_file(path / MODEL_FILE), source=str(path),
                            networks_source=f"Checkpoint {path}: networks")
        model = build_frozen_model(None, frozen, spec)      # also checks the weights against the architecture
        _check_model(model, agent_role_spec(spec, roles), None, f"snapshot {path}")
        validated.add(key)
    agent_id = f"{prefix}{agent}"
    return TournamentPlayer(
        name=f"{agent_id}@v{version}", display=f"{agent_id}@v{version}", kind="snapshot", agent_id=agent_id,
        network_id=f"ckpt_v{version}", roles=tuple(roles), lineage=agent_id,
        source=PlayerSource("model", origin=str(path), weights=str(path / MODEL_FILE), networks=dump),
        version=version, run=str(run), run_agent=agent)


def json_key(networks: Mapping[str, Any], roles: Sequence[str]) -> str:
    import json

    return json.dumps([networks, list(roles)], sort_keys=True)


def parse_snapshots(text: str) -> tuple[str, int]:
    """``all`` -> ("all", 0); ``last:N`` / ``top:N`` / ``every:K`` -> (kind, n). ConfigError otherwise."""
    if text == "all":
        return "all", 0
    match = _SELECTION_RE.fullmatch(text)
    if match is None:
        raise ConfigError(f"--snapshots {text!r}: expected all, last:N, top:N or every:K (N, K >= 1)")
    return match.group(1), int(match.group(2))


def select_snapshots(infos: Sequence[CheckpointInfo], kind: str, n: int, *,
                     lcb: Mapping[str, float] | None = None) -> list[CheckpointInfo]:
    """The selected snapshots, by version (module docstring). ``lcb``: checkpoint id -> lower bound (``top``)."""
    ordered = sorted(infos, key=lambda c: c.policy_version)
    if kind == "all":
        return ordered
    if kind == "last":
        return ordered[-n:]
    if kind == "every":
        picked = {c.checkpoint_id for c in ordered[::-1][::n]}
        return [c for c in ordered if c.checkpoint_id in picked]
    rated = [c for c in ordered if lcb is not None and c.checkpoint_id in lcb]
    rated.sort(key=lambda c: (-lcb[c.checkpoint_id], -c.policy_version))
    best = {c.checkpoint_id for c in rated[:n]}
    return [c for c in ordered if c.checkpoint_id in best]


def _random_bot(config: ColosseumConfig) -> str | None:
    """The first scripted agent of ``config`` whose class is ``colosseum.players.RandomBot`` (BT anchor rule)."""
    from colosseum.core.registry import import_class
    from colosseum.players import RandomBot

    for name in config.fixed_agent_ids():
        entry = config.agent_entry(name)
        if entry.kind == "scripted":
            try:
                if import_class(entry.class_path) is RandomBot:
                    return name
            except Exception:  # noqa: BLE001 - validate reports bad classes; here it only means "not RandomBot"
                continue
    return None


def top_snapshot_lcb(run: Path, agent: str, run_config: ColosseumConfig, spec: GameSpec, *,
                     rng: np.random.Generator) -> dict[str, float]:
    """``{checkpoint id: lower bound}`` of ``agent``'s rated snapshots from the run's journal: BT per competitive
    layout (``source`` train and pulse, ``bt.bootstrap`` rows), combined with the agent's layout weights
    (``retention.combine_scores``, quantile ``checkpoint.top_k_quantile``); ``score``-only games use the mean
    score. ConfigError without a journal."""
    from colosseum.arena import bt
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
    from colosseum.arena.retention import combine_scores
    from colosseum.league.lineups import enabled_layouts

    db = run / MATCH_LOG_FILE
    if not db.is_file():
        raise ConfigError(f"--snapshots top:N needs the run's match log {db}; this run has none (an SP3 run dir, or "
                          f"match_log.enabled was false): use all, last:N or every:K")
    sources = ("train", "pulse")
    reader = MatchLog.open_readonly(db)
    try:
        table = reader.comparison_table(sources=sources)
    finally:
        reader.close()
    options = bt.default_options(anchor=run_config.ratings.anchor)
    fixed_agents, random_bot = run_config.fixed_agent_ids(), _random_bot(run_config)
    fits, uncertainty, scores = {}, {}, {}
    for layout in table.layouts():
        if layout not in spec.layouts:
            continue
        if spec.num_teams(layout) == 1:
            scores[layout] = bt.score_summary(table, layout, sources=sources)
            continue
        fit = bt.fit(table, layout, sources=sources, options=options, fixed_agents=fixed_agents, random_bot=random_bot)
        fits[layout] = fit
        uncertainty[layout] = bt.bootstrap(fit, bt.BootstrapData.from_table(table, layout, sources=sources),
                                           samples=run_config.ratings.bootstrap_samples, rng=rng,
                                           quantile=run_config.checkpoint.top_k_quantile)
    weights = enabled_layouts(spec, run_config.get_agent_config(agent).matchmaking)
    combined = combine_scores(agent, fits=fits, uncertainty=uncertainty, scores=scores, layout_weights=weights,
                              quantile=run_config.checkpoint.top_k_quantile)
    return {cid: s.lcb for cid, s in combined.items() if s.lcb is not None}


def load_run_config(run: Path) -> ColosseumConfig:
    path = run / "config.resolved.yaml"
    if not path.is_file():
        raise ConfigError(f"--run {run}: no config.resolved.yaml; pass a run dir written by colosseum train")
    return load_config(str(path))


def run_agent(run: Path, run_config: ColosseumConfig, agent: str | None) -> str:
    """``--agent`` of a run, default its only trainable agent; ConfigError naming the choices."""
    trainable = run_config.get_trainable_agent_ids()
    if agent is None:
        if len(trainable) != 1:
            raise ConfigError(f"--run {run} has trainable agents {trainable}; choose one with --agent")
        return trainable[0]
    if agent not in trainable:
        raise ConfigError(f"--run {run} --agent {agent!r}: not a trainable agent of that run ({trainable})")
    return agent


LinkFn = Callable[[int, RunSelection, CheckpointInfo], Path]


def resolve_players(config: ColosseumConfig, spec: GameSpec, request: PlayerRequest, *, fixed: FixedPlayers,
                    link: LinkFn | None = None, rng: np.random.Generator | None = None,
                    allow_empty_field: bool = False) -> tuple[list[TournamentPlayer], list[TournamentPlayer]]:
    """``(candidates, field)`` by the rules of the module docstring. ``link(run_index, selection, info)`` returns
    the dir a snapshot is played from (T5.4: a hard link in the tournament dir); default the run's own dir."""
    rng = rng if rng is not None else np.random.default_rng(0)
    validated: set[str] = set()
    link = link if link is not None else (lambda _i, _sel, info: info.path)
    users: dict[str, TournamentPlayer] = {}
    for name, path in request.user:
        users[name] = config_player(config, spec, name, fixed) if path is None else \
            user_player(config, spec, name, path, validated=validated)
    runs = []        # (index, selection, prefix, agent, run_config, stored infos)
    aliases: set[str] = set()
    for i, sel in enumerate(request.runs):
        run = Path(sel.path)
        run_config = load_run_config(run)
        agent = run_agent(run, run_config, sel.agent)
        alias = sel.alias if sel.alias is not None else run.name
        check_path_component(alias, "--run alias")
        if alias in aliases:
            raise ConfigError(f"--run alias {alias!r} is used twice; give the runs different aliases (alias=runs/x)")
        aliases.add(alias)
        infos = read_snapshots(run / "checkpoints" / agent)
        if not infos:
            raise ConfigError(f"--run {run} --agent {agent}: no stored snapshots in {run / 'checkpoints' / agent}")
        runs.append((i, sel, "" if i == 0 else f"{alias}/", alias, agent, run_config, infos))

    def snapshot(i, sel, prefix, agent, run_config, info) -> TournamentPlayer:
        return snapshot_player(link(i, sel, info), agent=agent, prefix=prefix, spec=spec, run_config=run_config,
                               run=sel.path, validated=validated)

    selected: list[TournamentPlayer] = []
    for i, sel, prefix, _alias, agent, run_config, infos in runs:
        kind, n = parse_snapshots(sel.snapshots)
        lcb = top_snapshot_lcb(Path(sel.path), agent, run_config, spec, rng=rng) if kind == "top" else None
        chosen = select_snapshots(infos, kind, n, lcb=lcb)
        if kind == "top" and len(chosen) < n:
            import logging

            logging.getLogger(__name__).info("--run %s --snapshots top:%d: only %d snapshots are rated", sel.path, n,
                                             len(chosen))
        selected.extend(snapshot(i, sel, prefix, agent, run_config, info) for info in chosen)

    field_players: list[TournamentPlayer] = []
    if request.field is None:
        names = [n for n in config.fixed_agent_ids() if not (n in users and users[n].kind in ("scripted", "frozen"))]
        field_players = [config_player(config, spec, n, fixed) for n in names]
    else:
        by_alias = {("" if i == 0 else alias): (i, sel, prefix, agent, rc, infos)
                    for i, sel, prefix, alias, agent, rc, infos in runs}
        for name in request.field:
            if name in users:
                field_players.append(users[name])
                continue
            if name in config.agent_ids():
                field_players.append(config_player(config, spec, name, fixed))
                continue
            match = _SNAPSHOT_RE.fullmatch(name)
            target = by_alias.get(match.group("alias") or "") if match is not None else None
            if target is None or target[3] != match.group("agent"):
                raise ConfigError(
                    f"--field {name!r} is not an -a player {sorted(users)}, an agent of the config "
                    f"{config.agent_ids()} or a snapshot [alias/]agent@v<N> of a --run "
                    f"({[f'{a}/{t[3]}@v<N>' if a else f'{t[3]}@v<N>' for a, t in by_alias.items()]})")
            i, sel, prefix, agent, rc, infos = target
            info = next((c for c in infos if c.policy_version == int(match.group("version"))), None)
            if info is None:
                raise ConfigError(f"--field {name!r}: {sel.path} has no stored snapshot ckpt_v{match.group('version')} "
                                  f"of {agent} (newest: {[c.checkpoint_id for c in infos][-5:]})")
            field_players.append(snapshot(i, sel, prefix, agent, rc, info))
    in_field = {p.name for p in field_players} | {p.display for p in field_players}
    candidates = [p for p in [*users.values(), *selected] if p.name not in in_field and p.display not in in_field]
    names = [p.name for p in [*candidates, *field_players]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"players {duplicates} appear more than once (as candidates and in --field?)")
    if not candidates:
        raise ConfigError("no candidates: give -a players or --run snapshots that are not in --field")
    if not field_players and not allow_empty_field:
        raise ConfigError("the field is empty: name opponents with --field (agents of the config, -a players or "
                          "snapshots), or add scripted / frozen agents to the config")
    return candidates, field_players


def player_to_dict(player: TournamentPlayer) -> dict[str, Any]:
    data = asdict(player)
    data["roles"] = list(player.roles)
    return data


def player_from_dict(data: Mapping[str, Any]) -> TournamentPlayer:
    src = dict(data["source"])
    bot = src.get("bot")
    source = PlayerSource(kind=src["kind"], origin=src["origin"], weights=src.get("weights"),
                          networks=src.get("networks"),
                          bot=BotSpec(bot["class_path"], dict(bot.get("kwargs") or {})) if bot else None)
    return TournamentPlayer(name=data["name"], display=data["display"], kind=data["kind"], agent_id=data["agent_id"],
                            network_id=data["network_id"], roles=tuple(data["roles"]), lineage=data["lineage"],
                            source=source, version=data.get("version"), run=data.get("run"),
                            run_agent=data.get("run_agent"))
```

Notes for the implementer:
- `_SNAPSHOT_RE`'s alias group excludes `/` and `@`; `check_path_component` already forbids them in aliases (a ConfigError "alias" for `a/b`).
- The `-a name` (no path) of a scripted / frozen agent is a candidate unless named in `--field`; the default field leaves it out (it is a candidate).

- [ ] **Step 8: `arena/tournament/engine.py`**

```python
"""Tournament engine (SP4 spec block 3): whole blocks played in local worker processes on ``MatchRunner``.

``play_blocks`` splits the blocks into tasks of whole blocks (``make_tasks``: at least ``TASK_MATCHES *
num_envs`` matches each, so env batches are rarely half empty), starts ``min(num_workers, tasks)`` spawn
processes and feeds them the tasks through a queue. A worker (``run_child``: log ``<log_dir>/tournament-worker-
<i>.log``, SIGINT ignored, dies with the parent) plays a task with ``eval.play_lineups`` on a ``LazyPlayerPool``
(``source`` = ``tournament``; torch sampling seeded with the task's first block seed) and sends the results
back grouped by block. The main process calls ``on_block(block_id, results)`` once per block of a finished
task, in task order, results sorted by uid: a block is never reported in part. A worker's ConfigError,
EnvContractError or PlayerError is re-raised in the main process with its message; a worker that dies
otherwise raises ``TournamentError`` naming its log. On every exit (also a KeyboardInterrupt in the main
process) the workers are stopped: ``stop_event``, ``STOP_GRACE_SEC``, then terminate / kill.

Results carry ``source``, the lineup's ``seed``, ``block`` and ``uid`` (T1.1); a seat's ``version_start`` /
``version_end`` is ``N`` for ``ckpt_v<N>`` and None for fixed players; ``env_steps`` is None.

``LazyPlayerPool`` is a ``PlayerPool`` keyed by ``(agent_id, network_id)``: scripted players are
``ScriptedPlayer`` factories; neural players are built on first use from their weights file
(``build_frozen_model``: eval mode) and kept in an LRU cache of ``MODEL_CACHE_SIZE`` models; ``reserve`` loads the
models of one task and protects them until the next ``reserve`` (no model is reloaded inside a task).
"""

from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import queue
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from colosseum.arena.tournament.players import TournamentPlayer
from colosseum.arena.tournament.schedule import Block
from colosseum.coordinator.checkpoint_manager import read_weights_file
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec
from colosseum.networks.model import PolicyModel
from colosseum.players.registry import FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import ProcessSupervisor, run_child, start_process
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)

MODEL_CACHE_SIZE = 16      # neural players kept per worker (LRU); a task's own players are never evicted
TASK_MATCHES = 4           # a task holds whole blocks with >= TASK_MATCHES * num_envs matches (or the rest)
STOP_GRACE_SEC = 2.0
_ERRORS: dict[str, type[Exception]] = {"ConfigError": ConfigError, "EnvContractError": EnvContractError,
                                       "PlayerError": PlayerError}


class TournamentError(RuntimeError):
    """A tournament worker died or broke the protocol (the message names its log)."""


def build_player_model(player: TournamentPlayer, spec: GameSpec) -> PolicyModel:
    """The neural player's model with its weights, in eval mode; ConfigError naming its origin."""
    try:
        state = read_weights_file(player.source.weights)
    except ValueError as e:
        raise ConfigError(f"{player.source.origin}: {e}") from e
    frozen = FrozenSpec(agent_id=player.agent_id, roles=tuple(player.roles), networks=dict(player.source.networks),
                        model_state=state, source=player.source.origin, networks_source=player.source.origin)
    return build_frozen_model(None, frozen, spec)


class LazyPlayerPool:
    """``PlayerPool`` of tournament players by ``(agent_id, network_id)`` (module docstring)."""

    def __init__(self, players: Iterable[TournamentPlayer], spec: GameSpec, capacity: int = MODEL_CACHE_SIZE) -> None:
        self._spec = spec
        self._players = {(p.agent_id, p.network_id): p for p in players}
        self._capacity = int(capacity)
        self._models: OrderedDict[tuple[str, str], PolicyModel] = OrderedDict()
        self._bots: dict[tuple[str, str], ScriptedPlayer] = {}
        self._reserved: set[tuple[str, str]] = set()
        self.loads = 0

    def reserve(self, keys: Iterable[tuple[str, str]]) -> None:
        keys = set(keys)
        unknown = sorted(k for k in keys if k not in self._players)
        if unknown:
            raise ValueError(f"LazyPlayerPool: unknown players {unknown}")
        self._reserved = {k for k in keys if self._players[k].source.kind == "model"}
        for key in sorted(self._reserved):
            self._model(key)
        self._trim()

    def get(self, agent_id: str, network_id: str) -> PolicyModel | ScriptedPlayer | None:
        key = (agent_id, network_id)
        player = self._players.get(key)
        if player is None:
            return None
        if player.source.kind == "bot":
            if key not in self._bots:
                self._bots[key] = ScriptedPlayer(functools.partial(make_bot, player.source.bot, self._spec))
            return self._bots[key]
        return self._model(key)

    def cached(self) -> list[tuple[str, str]]:
        return list(self._models)

    def _model(self, key: tuple[str, str]) -> PolicyModel:
        model = self._models.get(key)
        if model is not None:
            self._models.move_to_end(key)
            return model
        model = build_player_model(self._players[key], self._spec)
        self.loads += 1
        self._models[key] = model
        self._trim()
        return model

    def _trim(self) -> None:
        limit = max(self._capacity, len(self._reserved))
        for key in list(self._models):
            if len(self._models) <= limit:
                break
            if key not in self._reserved:
                del self._models[key]


def make_tasks(blocks: Sequence[Block], num_envs: int) -> list[list[Block]]:
    """Consecutive whole blocks, each task with at least ``TASK_MATCHES * num_envs`` matches (the last: the rest)."""
    target = max(1, TASK_MATCHES * int(num_envs))
    tasks: list[list[Block]] = []
    current: list[Block] = []
    matches = 0
    for block in blocks:
        current.append(block)
        matches += len(block.lineups)
        if matches >= target:
            tasks.append(current)
            current, matches = [], 0
    if current:
        tasks.append(current)
    return tasks


def _put(q: Any, item: Any, stop_event: Any) -> bool:
    """Blocking put that gives up when ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def _worker_target(*, worker_id: int, log_dir: str | None = None, **kwargs: Any) -> None:
    run_child(f"tournament-worker-{worker_id}", log_dir, _worker_main, worker_id=worker_id, **kwargs)


def _worker_main(*, worker_id: int, config: ColosseumConfig, players: list[TournamentPlayer], tasks: Any, results: Any,
                 stop_event: Any, num_envs: int, deterministic: bool, source: str) -> None:
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.eval import play_lineups

    configure_torch_threads(1)
    spec = env_spec(config)
    pool = LazyPlayerPool(players, spec)
    while not stop_event.is_set():
        try:
            item = tasks.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            return
        index, blocks = item
        try:
            pool.reserve({(s.agent_id, s.network_id) for b in blocks for lu in b.lineups for s in lu.seats})
            played = play_lineups(env_fn=lambda: make_env(config), pool=pool,
                                  lineups=[lu for b in blocks for lu in b.lineups], num_envs=num_envs,
                                  seed=blocks[0].seed, deterministic=deterministic,
                                  max_idle_steps=config.env.max_idle_steps, source=source)
        except (ConfigError, EnvContractError, PlayerError) as e:
            _put(results, ("error", index, type(e).__name__, str(e)), stop_event)
            raise
        by_block: dict[str, list[MatchResult]] = {}
        for result in played:
            by_block.setdefault(result.block, []).append(result)
        logger.info(f"tournament worker {worker_id}: task {index} done ({len(played)} matches)")
        if not _put(results, ("done", index, by_block), stop_event):
            return


def _error_after_failure(results: Any, failure: Any) -> Exception:
    """The error a dead worker sent before dying (a typed one), else a TournamentError naming its log."""
    try:
        while True:
            msg = results.get(timeout=1.0)
            if msg[0] == "error":
                return _ERRORS[msg[2]](msg[3])
    except queue.Empty:
        return TournamentError(f"tournament {failure.message()}")


def play_blocks(config: ColosseumConfig, spec: GameSpec, players: Mapping[str, TournamentPlayer] | Sequence,
                blocks: Sequence[Block], *, num_workers: int, num_envs: int, deterministic: bool,
                on_block: Callable[[str, list[MatchResult]], None], log_dir: str | Path | None = None,
                source: str = "tournament") -> None:
    """Play ``blocks`` (module docstring)."""
    blocks = list(blocks)
    if not blocks:
        return
    if num_workers < 1 or num_envs < 1:
        raise ValueError(f"play_blocks: need num_workers >= 1 and num_envs >= 1, got {num_workers}, {num_envs}")
    player_list = list(players.values()) if isinstance(players, Mapping) else list(players)
    tasks = make_tasks(blocks, num_envs)
    ctx = mp.get_context("spawn")
    task_q, result_q, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    supervisor = ProcessSupervisor(stop, log_dir=log_dir)
    try:
        for index, task in enumerate(tasks):
            task_q.put((index, task))
        workers = min(num_workers, len(tasks))
        for _ in range(workers):
            task_q.put(None)
        for w in range(workers):
            proc = ctx.Process(target=_worker_target, name=f"tournament-worker-{w}", daemon=True, kwargs=dict(
                worker_id=w, log_dir=None if log_dir is None else str(log_dir), config=config, players=player_list,
                tasks=task_q, results=result_q, stop_event=stop, num_envs=num_envs, deterministic=deterministic,
                source=source))
            start_process(proc)
            supervisor.add(f"tournament-worker-{w}", proc)
        pending = set(range(len(tasks)))
        while pending:
            try:
                msg = result_q.get(timeout=0.2)
            except queue.Empty:
                failure = supervisor.first_failure(nonzero_only=True)
                if failure is not None:
                    raise _error_after_failure(result_q, failure) from None
                if not supervisor.alive():
                    raise TournamentError(f"tournament workers exited with {len(pending)} tasks unplayed") from None
                continue
            if msg[0] == "error":
                raise _ERRORS[msg[2]](msg[3])
            _kind, index, by_block = msg
            for block in tasks[index]:
                got = sorted(by_block.get(block.id, []), key=lambda r: r.uid)
                if len(got) != len(block.lineups):
                    raise TournamentError(f"block {block.id}: {len(got)} results for {len(block.lineups)} matches")
                on_block(block.id, got)
            pending.discard(index)
    finally:
        stop.set()
        supervisor.wait_all(STOP_GRACE_SEC)
        supervisor.kill_remaining()
        for q in (task_q, result_q):
            q.cancel_join_thread()
            q.close()
```

- [ ] **Step 9: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_eval_dedup.py tests/unit/test_sp4_tournament_players.py tests/integration/test_sp4_tournament_engine.py tests/integration/test_sp2_eval_cli.py tests/unit/test_sp3_eval_players.py tests/integration/test_sp3_record.py tests/unit/test_sp3_players.py tests/contract/test_sp2_eval_engine.py tests/unit/test_sp2_eval_schedule.py -q`
Expected: all pass. If an SP3 test asserted the removed `.pt` overlay on `-a name=path`, change it to the new rule (the entry applies only to the agent's own `path`) and say so in the task report.

- [ ] **Step 10: Full fast suite + ruff**

- [ ] **Step 11: Commit and push**

```bash
git add src/colosseum/eval.py src/colosseum/record.py src/colosseum/cli.py src/colosseum/players/registry.py \
        src/colosseum/core/validation.py src/colosseum/worker/match_runner.py src/colosseum/arena/tournament \
        tests/game_helpers.py tests/unit/test_sp4_eval_dedup.py tests/unit/test_sp4_tournament_players.py \
        tests/integration/test_sp4_tournament_engine.py
git commit -m "feat: tournament players and worker engine; shared player loading for eval, record and tournament"
git push
```

---
### Task T5.2: Tournament schedules (blocks, admissible rotations, FFA compositions, `score` layouts, seed domains, role groups)

Spec block 3 («Дизайн матчей»): a **block** is one seed, one composition (candidate + field members by team) and every admissible rotation of the team assignment; it is the unit of scheduling, resume and bootstrap. Seeds come from `stage_seed` (T5.1) with the domains A (screening), B (confirmation) and P (pulse) disjoint by construction; every candidate gets the same seed for the same (stage, index). FFA gauntlets enumerate compositions of the field so every member meets the candidate equally often; asymmetric roles keep only admissible rotations; `score` layouts schedule the candidate's homogeneous team and cross-play compositions with one field member. Role groups (spec block 2) decide which candidates share a table; overlapping groups are a ConfigError.

**Files:**
- Modify: `src/colosseum/arena/tournament/schedule.py` (the whole module below; `Block`, `STAGES`, `stage_seed` unchanged from T5.1)
- Modify: `tests/game_helpers.py` (`SyntheticSource`, `SyntheticPlayer`, `synthetic_players`)
- Test: `tests/unit/test_sp4_tournament_schedule.py`

**Interfaces:**
- Consumes: `Lineup(layout, seats, seed=, block=, uid=)`, `SeatAssignment` (T1.1); `GameSpec.teams / num_teams / layouts`; `Block`, `stage_seed` (T5.1).
- Produces (contract T5.2, additions marked *): `gauntlet_blocks(spec, layout, candidates, field, players, *, blocks, seed, stage, first_index=0*, prefix="t"*)`, `round_robin_blocks(spec, layout, participants, players, *, blocks, seed, stage="A", first_index=0, prefix="t")`, `score_blocks(...)` (same keywords as `gauntlet_blocks`); *`compositions(field, slots)`, *`role_group(spec, layout, roles)`, *`group_label(group)`, *`check_role_groups(spec, layout, names, players) -> dict[str, list[str]]`, *`SeatPlayer` (protocol: `agent_id`, `network_id`, `roles`). Contract deviation: the builders take `players: Mapping[name, SeatPlayer]` instead of `roles` (they need the seat ids too; Contract notes C4).

- [ ] **Step 1: Write the failing tests**

Add to `tests/game_helpers.py` (section "SP4 (T5.2): tournament stand-ins"):

```python
@dataclass(frozen=True)
class SyntheticSource:
    origin: str = "synthetic"


@dataclass(frozen=True)
class SyntheticPlayer:
    """A tournament player stand-in with the fields ``schedule``, ``stats`` and ``modes`` read."""

    agent_id: str
    network_id: str = "fixed"
    roles: tuple[str, ...] = ("player",)
    display: str = ""
    kind: str = "scripted"
    lineage: str = ""
    source: SyntheticSource = field(default_factory=SyntheticSource)


def synthetic_players(names, roles=None) -> dict:
    """``"main@v3"`` -> a snapshot of ``main`` (``ckpt_v3``, lineage ``main``); any other name -> a scripted player
    (lineage = name). ``roles``: name -> roles (default ``("player",)``)."""
    out = {}
    for name in names:
        agent, _, version = name.partition("@v")
        out[name] = SyntheticPlayer(agent, f"ckpt_v{version}" if version else "fixed",
                                    tuple((roles or {}).get(name, ("player",))), display=name,
                                    kind="snapshot" if version else "scripted", lineage=agent)
    return out
```

Create `tests/unit/test_sp4_tournament_schedule.py`:

```python
"""Tournament schedules (SP4 T5.2, spec block 3): blocks, admissible rotations, FFA compositions, score layouts,
seed domains, role groups."""
from __future__ import annotations

import collections

import pytest

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_tournament_schedule.py -q`
Expected: `ImportError: cannot import name 'check_role_groups' from 'colosseum.arena.tournament.schedule'`.

- [ ] **Step 3: Write `src/colosseum/arena/tournament/schedule.py`**

```python
"""Tournament schedules (SP4 spec block 3): blocks, admissible team rotations, FFA compositions, ``score``
layouts, seed domains.

A **block** is one seed (``stage_seed(seed, stage, index)``), one composition and every admissible rotation:
- two or more teams (``gauntlet_blocks``, ``round_robin_blocks``): the entity list ``[candidate, *composition]``
  is assigned to the teams by cyclic shifts; shift ``r`` gives team ``t`` the entity ``E[(t - r) % T]`` (so the
  candidate sits in team ``r``). A shift is admissible when every team's roles are roles of its entity; a
  block holds every admissible shift (1v1: two matches with sides swapped; T symmetric teams: T matches;
  asymmetric roles: only the admissible shifts, sometimes one). A composition without an admissible shift is
  not scheduled.
- one team (``score_blocks``): the candidate's homogeneous team (one match; composition ``()``) and, per field
  member ``f``, the composition ``(f,)``: ``f`` takes one seat, the candidate the others, and the block holds
  one match per seat ``f`` can take (rotation = that seat), like SP2's cross-play.

Gauntlet compositions (spec block 3): the candidate takes one team, field members the other ``T - 1``: every
combination of ``T - 1`` field members when ``F >= T - 1``, else every multiset of size ``T - 1``; so every field
member meets the candidate equally often (symmetric roles). ``blocks`` is the number of blocks per
composition: indices ``first_index .. first_index + blocks - 1``.

Seeds: ``stage_seed(seed, stage, index)`` puts the stage's domain code in the top two bits of a 32-bit seed and
a keyed bijection of ``index`` in the low 30 bits, so the domains A (screening), B (confirmation) and P (pulse)
never share a seed and no seed repeats within a domain (indices < 2**30). Every candidate gets the same seed
for the same (stage, index): common random numbers.

Ids: ``Block.id = "<prefix>/<stage>/<layout>/<candidate>/<composition or ->/<index>"`` and every match's
``Lineup.uid = "<block id>/r<rotation>"``; ``prefix`` is the tournament name (or ``pulse``) and contains no
``/``. Every seat plays ``collect=False``.

Role groups: a candidate's group in a layout is the set of the layout's roles it plays. Candidates are ranked
within a group; groups that overlap without being equal (e.g. ``{hunter}`` and ``{hunter, prey}``) cannot share
a table: ``check_role_groups`` raises ConfigError listing them.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from colosseum.core.errors import ConfigError
from colosseum.core.types import Lineup, SeatAssignment
from colosseum.envs.game import GameSpec

STAGES = ("A", "B", "P")
_STAGE_CODE = {"A": 1, "B": 2, "P": 3}
_LOW_BITS = 30
_LOW_MASK = (1 << _LOW_BITS) - 1


class SeatPlayer(Protocol):
    """What a schedule needs of a player (``TournamentPlayer`` has it)."""

    @property
    def agent_id(self) -> str: ...
    @property
    def network_id(self) -> str: ...
    @property
    def roles(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Block:
    id: str
    stage: str
    layout: str
    candidate: str
    composition: tuple[str, ...]
    index: int
    seed: int
    lineups: tuple[Lineup, ...]
    rotations: tuple[int, ...]

    @property
    def stratum(self) -> str:
        """The composition as one key: field members joined by ``+`` (``-`` for the homogeneous team)."""
        return "+".join(self.composition) or "-"


def stage_seed(seed: int, stage: str, index: int) -> int:
    """The episode seed of block ``index`` of ``stage`` (module docstring: disjoint domains, no repeats)."""
    if stage not in _STAGE_CODE:
        raise ValueError(f"stage_seed: stage must be one of {STAGES}, got {stage!r}")
    if not 0 <= int(index) <= _LOW_MASK:
        raise ValueError(f"stage_seed: index must be in [0, 2**30), got {index}")
    code = _STAGE_CODE[stage]
    a, b, c = (int(x) for x in np.random.SeedSequence([int(seed) % 2**63, code]).generate_state(3, np.uint32))
    x = (int(index) * (a | 1) + b) & _LOW_MASK      # odd multiplier + offset: a bijection mod 2**30
    x ^= x >> 15
    x = (x * (c | 1)) & _LOW_MASK
    x ^= x >> 13
    return (code << _LOW_BITS) | x


def _seat(player: SeatPlayer) -> SeatAssignment:
    return SeatAssignment(agent_id=player.agent_id, network_id=player.network_id, collect=False)


def _team_roles(spec: GameSpec, layout: str) -> list[set[str]]:
    seats = spec.layouts[layout]
    return [{seats[s].role for s in members} for members in spec.teams(layout)]


def role_group(spec: GameSpec, layout: str, roles: Sequence[str]) -> frozenset[str]:
    """The roles of ``layout`` that a player with ``roles`` plays (its role group in that layout)."""
    return frozenset({seat.role for seat in spec.layouts[layout]} & set(roles))


def group_label(group: frozenset[str]) -> str:
    return "+".join(sorted(group))


def check_role_groups(spec: GameSpec, layout: str, names: Sequence[str],
                      players: Mapping[str, SeatPlayer]) -> dict[str, list[str]]:
    """``{group label: candidates}`` of ``layout``; ConfigError when two groups overlap without being equal."""
    groups: dict[frozenset[str], list[str]] = {}
    for name in names:
        groups.setdefault(role_group(spec, layout, players[name].roles), []).append(name)
    empty = groups.pop(frozenset(), None)
    if empty:
        raise ConfigError(f"layout {layout!r}: candidates {empty} play none of its roles "
                          f"{sorted({s.role for s in spec.layouts[layout]})}; choose layouts with --layout")
    keys = list(groups)
    for g1, g2 in itertools.combinations(keys, 2):
        if g1 & g2:
            listing = {group_label(g): groups[g] for g in keys}
            raise ConfigError(f"layout {layout!r}: candidates of overlapping role groups cannot be ranked in one "
                              f"table: {listing}; give every candidate the same roles or disjoint roles "
                              f"(e.g. hunters and prey)")
    return {group_label(g): members for g, members in groups.items()}


def _rotation_lineup(spec: GameSpec, layout: str, entities: Sequence[str], shift: int,
                     players: Mapping[str, SeatPlayer]) -> Lineup | None:
    teams = spec.teams(layout)
    roles = _team_roles(spec, layout)
    n = len(teams)
    seats: list[SeatAssignment | None] = [None] * spec.layout_size(layout)
    for t, members in enumerate(teams):
        name = entities[(t - shift) % n]
        if not roles[t] <= set(players[name].roles):
            return None
        for s in members:
            seats[s] = _seat(players[name])
    return Lineup(layout=layout, seats=seats)  # type: ignore[arg-type]


def _block(prefix: str, stage: str, layout: str, candidate: str, composition: tuple[str, ...], index: int,
           seed: int, lineups: list[tuple[int, Lineup]]) -> Block:
    block_id = f"{prefix}/{stage}/{layout}/{candidate}/{'+'.join(composition) or '-'}/{index}"
    s = stage_seed(seed, stage, index)
    stamped = tuple(Lineup(layout=lu.layout, seats=lu.seats, seed=s, block=block_id, uid=f"{block_id}/r{r}")
                    for r, lu in lineups)
    return Block(id=block_id, stage=stage, layout=layout, candidate=candidate, composition=composition,
                 index=index, seed=s, lineups=stamped, rotations=tuple(r for r, _ in lineups))


def _check_common(spec: GameSpec, layout: str, stage: str, blocks: int, first_index: int, prefix: str) -> None:
    if layout not in spec.layouts:
        raise ConfigError(f"layout {layout!r} is not a layout of the game {sorted(spec.layouts)}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if blocks < 1 or first_index < 0:
        raise ValueError(f"need blocks >= 1 and first_index >= 0, got {blocks}, {first_index}")
    if not prefix or "/" in prefix:
        raise ValueError(f"block id prefix must be non-empty without '/', got {prefix!r}")


def compositions(field: Sequence[str], slots: int) -> list[tuple[str, ...]]:
    """Gauntlet compositions of ``slots`` field members: combinations when ``len(field) >= slots``, else
    multisets (module docstring)."""
    if slots == 0:
        return [()]
    if len(field) >= slots:
        return list(itertools.combinations(field, slots))
    return list(itertools.combinations_with_replacement(field, slots))


def gauntlet_blocks(spec: GameSpec, layout: str, candidates: Sequence[str], field: Sequence[str],
                    players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str,
                    first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Blocks of every candidate against the field in a layout of two or more teams (module docstring).

    ConfigError when a candidate has no admissible composition (the message lists the teams' roles), when
    the field is empty, or for a one-team layout (use ``score_blocks``)."""
    _check_common(spec, layout, stage, blocks, first_index, prefix)
    n_teams = spec.num_teams(layout)
    if n_teams < 2:
        raise ValueError(f"gauntlet_blocks: layout {layout!r} has one team; use score_blocks")
    if not field:
        raise ConfigError("the field is empty: name opponents with --field (agents of the config, -a players or "
                          "snapshots), or add scripted / frozen agents to the config")
    out: list[Block] = []
    comps = compositions(list(field), n_teams - 1)
    for cand in candidates:
        found = False
        for comp in comps:
            entities = [cand, *comp]
            rotations = [(r, lu) for r in range(n_teams)
                         if (lu := _rotation_lineup(spec, layout, entities, r, players)) is not None]
            if not rotations:
                continue
            found = True
            for k in range(first_index, first_index + blocks):
                out.append(_block(prefix, stage, layout, cand, tuple(comp), k, seed, rotations))
        if not found:
            raise ConfigError(
                f"layout {layout!r}: candidate {cand!r} (roles {list(players[cand].roles)}) and the field "
                f"{list(field)} cannot fill its teams (roles per team: "
                f"{[sorted(r) for r in _team_roles(spec, layout)]}); check the roles or choose --layout")
    return out


def round_robin_blocks(spec: GameSpec, layout: str, participants: Sequence[str],
                       players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str = "A",
                       first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Every pair (a, b) of ``participants`` in a two-team layout: blocks with ``candidate=a``,
    ``composition=(b,)`` and both admissible sides. A pair without an admissible side is skipped; ConfigError
    when no pair can play, or for a layout with other than two teams."""
    _check_common(spec, layout, stage, blocks, first_index, prefix)
    if spec.num_teams(layout) != 2:
        raise ConfigError(f"--mode round-robin needs two-team layouts; {layout!r} has {spec.num_teams(layout)} "
                          f"teams (use gauntlet, halving or select)")
    out: list[Block] = []
    for a, b in itertools.combinations(participants, 2):
        rotations = [(r, lu) for r in range(2)
                     if (lu := _rotation_lineup(spec, layout, [a, b], r, players)) is not None]
        for k in range(first_index, first_index + blocks if rotations else first_index):
            out.append(_block(prefix, stage, layout, a, (b,), k, seed, rotations))
    if not out:
        raise ConfigError(f"layout {layout!r}: no pair of {list(participants)} can fill its two teams")
    return out


def score_blocks(spec: GameSpec, layout: str, candidates: Sequence[str], field: Sequence[str],
                 players: Mapping[str, SeatPlayer], *, blocks: int, seed: int, stage: str,
                 first_index: int = 0, prefix: str = "t") -> list[Block]:
    """Blocks of a one-team layout (module docstring): per candidate its homogeneous team (if it plays every
    role) and, per field member, the cross-play composition. ConfigError when a candidate has no block."""
    _check_common(spec, layout, stage, blocks, first_index, prefix)
    if spec.num_teams(layout) != 1:
        raise ValueError(f"score_blocks: layout {layout!r} has {spec.num_teams(layout)} teams")
    seat_roles = [seat.role for seat in spec.layouts[layout]]
    out: list[Block] = []
    for cand in candidates:
        own = set(players[cand].roles)
        plans: list[tuple[tuple[str, ...], list[tuple[int, Lineup]]]] = []
        if set(seat_roles) <= own:
            plans.append(((), [(0, Lineup(layout, [_seat(players[cand]) for _ in seat_roles]))]))
        if len(seat_roles) >= 2:
            for f in field:
                rotations = []
                for s, role in enumerate(seat_roles):
                    others = [r for i, r in enumerate(seat_roles) if i != s]
                    if role in players[f].roles and set(others) <= own:
                        seats = [_seat(players[f]) if i == s else _seat(players[cand]) for i in range(len(seat_roles))]
                        rotations.append((s, Lineup(layout, seats)))
                if rotations:
                    plans.append(((f,), rotations))
        if not plans:
            raise ConfigError(f"layout {layout!r}: candidate {cand!r} (roles {sorted(own)}) cannot play its seats "
                              f"(roles {seat_roles}) alone or with a field member; choose --layout")
        for comp, rotations in plans:
            for k in range(first_index, first_index + blocks):
                out.append(_block(prefix, stage, layout, cand, comp, k, seed, rotations))
    return out
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_tournament_schedule.py tests/integration/test_sp4_tournament_engine.py -q`
Expected: all pass (the 50 000-seed domain test takes well under a second).

- [ ] **Step 5: Full fast suite + ruff**

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/arena/tournament/schedule.py tests/game_helpers.py tests/unit/test_sp4_tournament_schedule.py
git commit -m "feat: tournament schedules (blocks with admissible rotations, FFA compositions, score layouts, seed domains)"
git push
```

---
### Task T5.3: Tournament statistics and modes (candidate scores, paired stratified bootstrap, `confirmed`, repeats, halving, select, several layouts, the report)

Spec block 3 («Режимы», «Результат кандидата», «Раскладки score», «Итог select», «Отчёт»). `stats` turns played blocks into candidate tables (mean pairwise score against every field team, equal lineage weights, worst opponent; mean team score in `score` layouts), bootstraps them pairwise and stratified (block indices redrawn within each composition, the same draw for every candidate: common seeds), decides `confirmed` and flags repeated deterministic games. `modes` runs `gauntlet`, `round-robin`, `halving` and `select` on a results source (`play(blocks)`: the tournament runner in T5.4, a fake in the tests), combines several layouts with the `matchmaking.layouts` weights on the same replicates, adds the tournament's own BT ratings (T2.1) per pairwise layout and builds the report dict (schema below) and its text.

**Report schema** (`TournamentReport.to_dict()`, written as `report.json`; Part 3's renderer reads it, `format` 1):

```text
{
  "format": 1, "kind": "tournament", "name": str, "mode": "gauntlet" | "round-robin" | "halving" | "select",
  "seed": int, "deterministic": bool, "ci_level": 0.95, "bootstrap_samples": int,
  "settings": {"blocks", "halving_blocks", "confirm_blocks", "finalists"},
  "candidates": [PlayerRow], "field": [PlayerRow],
  "layouts": [{"layout", "outcome_kind": "wdl"|"rank"|"score", "metric": "pairwise"|"score", "weight": float}],
  "tables": [Table],            # per (stage, layout, group); "layout": "*" = the weighted combination
  "rounds": [Round],            # halving / select stage A
  "selection": {group: Selection} | null,
  "bt": {layout: BTSection | null},   # null for score layouts and when no BT was computed
  "games": {"matches": int, "blocks": int, "by_stage": {stage: {"blocks": int, "matches": int}}},
  "flags": {"repeated_pairs": [[layout, candidate, column], ...], "non_transitive": [layout, ...]},
  "elapsed_sec": float          # informational (monotonic clock)
}
PlayerRow = {"name", "display", "kind", "lineage", "roles": [...], "source": str}
Table (per layout) = {"stage": "A"|"B", "layout": str, "group": str, "metric": "pairwise"|"score",
         "columns": [column names; "(self)" = the homogeneous team in score layouts], "column_weights": [float],
         "rows": [{"name", "score", "ci95": [lo, hi], "worst": {"opponent", "score"} | null, "games": int,
                   "blocks": int, "repeated": [columns], "bt": {"rating", "ci95", "p_best"} | null}],  # by score
         "matrix": {candidate: {column: {"score", "ci95", "games", "blocks", "repeated": bool}}}}
Table (combined, layout "*") = {"stage", "layout": "*", "group", "rows": [{"name", "score", "ci95"}]}
Table (round-robin) = {"stage": "A", "layout", "group": "*", "metric": "pairwise", "columns": [participants],
         "rows": [{"name", "score", "opponents"}], "matrix": {a: {b: {"score", "ci95", "blocks"}}}}
Round = {"group", "round", "stage": "A", "blocks_per_composition", "first_index", "active": [ranked names],
         "scores": {name: score}, "eliminated": [names]}
Selection = {"winner", "runner_up" | null, "confirmed": bool, "diff": float | null, "diff_ci95": [lo, hi] | null,
             "indistinguishable": [names], "repeated_columns": [[layout, column], ...], "note": str}
BTSection = {"anchor": [{"group", "component", "anchor"}], "entities": [{"name", "rating", "ci95", "p_best",
             "games"}], "transitivity": {"p_value": float | null, "flag": bool, "worst_pairs": [...]},
             "converged": bool}
```

Ratings are in Elo units (T2.1); scores in [0, 1] for pairwise layouts, team-score units for `score` layouts.

**Files:**
- Create: `src/colosseum/arena/tournament/stats.py`, `src/colosseum/arena/tournament/modes.py`
- Modify: `tests/game_helpers.py` (`FakeArena`)
- Test: `tests/unit/test_sp4_tournament_stats.py`, `tests/unit/test_sp4_tournament_modes.py`

**Interfaces:**
- Consumes: `schedule.*` (T5.2); `MatchResult.uid / block / teams / seats` (T1.1); `core.outcomes.pairwise_rank_score`; `eval.wilson_interval`; `arena.bt.fit`, `bt.bootstrap`, `bt.transitivity_test`, `BTFit.rating / games / groups / converged`, `Uncertainty.ci95 / p_best_group` (T2.1), `bt.default_options`, `bt.BootstrapData.from_blocks` (R7, R8); `arena.comparisons.ComparisonTable.add_result` (T1.2); `arena.entities.snapshot_version` (T1.2).
- Produces (contract T5.3; the contract names `candidate_scores`, `paired_bootstrap`, `confirmed`, `repeated_pairs`, `modes.run`, `TournamentReport.to_dict()/.text()` map as follows, additions marked *):
  - `stats.candidate_scores(blocks, results_by_block, *, names, lineage, group_of, metric_of, deterministic=False) -> dict[(stage, layout, group), CandidateTable]` (deviation: blocks and lookup maps instead of `candidates, field, lineage`; Contract notes C5);
  - `stats.CandidateTable` with `cells()`, `scores()`, `weights`, `bootstrap_cells(samples, rng)` and `bootstrap_scores(samples, rng)` (= the contract's `paired_bootstrap`), `repeated` (= `repeated_pairs`), `blocks_of(column)`;
  - `stats.confirmed(diff_replicates, *, level=0.95, conflict=False) -> bool` (deviation: takes the difference replicates; `select_winner` computes them); *`stats.select_winner(combined, tables, *, level=0.95) -> Selection`, *`stats.combine(tables, weights, *, samples, rng) -> Combined`, *`stats.repeated_conflict`, *`stats.cell_interval`, *`stats.block_values`, *`stats.quantiles`, *`SELF`, *`PAIRED_BOOTSTRAP_SAMPLES`;
  - `modes.run(mode, *, spec, players, candidates, field, settings, play, bt_fn=None) -> TournamentReport`; *`modes.TournamentSettings`, *`modes.MODES`, *`modes.DEFAULT_BLOCKS / DEFAULT_HALVING_BLOCKS / DEFAULT_CONFIRM_BLOCKS / DEFAULT_FINALISTS`, *`modes.bt_section(layout, results, blocks, *, players, anchor=None, samples=200, seed=0) -> dict`, *`modes.report_text(data)`, *`modes.REPORT_FORMAT`.

Proposed rulings in this task (Contract notes D5–D9): repeated deterministic pairs count as one block by (a) a per-cell interval of one observation and (b) never letting `confirmed` rest on a column where winner and runner-up differ and either repeats; stage A tables of `halving` / `select` show the finalists over the blocks they all played (eliminated candidates appear in `rounds` with their scores); `round-robin` only on two-team layouts; the stage A table rows of `select` carry no selection.

- [ ] **Step 1: Write the failing tests**

Add to `tests/game_helpers.py` (section "SP4 (T5.3): a results source for tournament modes"; it needs `from colosseum.core.types import MatchResult, SeatResult, TeamResult` at the top):

```python
class FakeArena:
    """A ``play`` function for the tournament modes and statistics without envs: every match is a Bradley-Terry
    draw on ``strengths`` (a team's strength is its player's; ranks by ``strength + Gumbel``, so
    ``P(a beats b) = sigmoid(s_a - s_b)``); a one-team layout scores ``mean strength + N(0, noise)``. Results are
    real ``MatchResult``s (uid, block, seed, source "tournament"); ``length`` is the episode length (int or
    callable of the rng). ``calls`` lists the block ids of every call."""

    def __init__(self, spec, players, strengths, *, seed=0, noise=0.1, length=6):
        self.spec, self.strengths, self.noise, self.length = spec, strengths, noise, length
        self.rng = np.random.default_rng(seed)
        self.names = {(p.agent_id, p.network_id): n for n, p in players.items()}
        self.calls: list[list[str]] = []

    def _length(self) -> int:
        return int(self.length(self.rng)) if callable(self.length) else int(self.length)

    def __call__(self, blocks):
        self.calls.append([b.id for b in blocks])
        out = {}
        for b in blocks:
            seat_specs = self.spec.layouts[b.layout]
            rows = []
            for lu in b.lineups:
                names = {seat_specs[s].team: self.names[(x.agent_id, x.network_id)] for s, x in enumerate(lu.seats)}
                if len(names) == 1:
                    value = float(np.mean([self.strengths[self.names[(x.agent_id, x.network_id)]]
                                           for x in lu.seats])) + float(self.rng.normal(0.0, self.noise))
                    teams = [TeamResult(0, 1.0, value)]
                else:
                    perf = {t: self.strengths[n] + float(self.rng.gumbel()) for t, n in names.items()}
                    order = sorted(perf, key=lambda t: -perf[t])
                    teams = [TeamResult(t, float(order.index(t) + 1), 0.0) for t in sorted(names)]
                seats = [SeatResult(s, seat_specs[s].role, seat_specs[s].team, x.agent_id, x.network_id, 0.0)
                         for s, x in enumerate(lu.seats)]
                rows.append(MatchResult(match_id=lu.uid, layout=b.layout, outcome_kind=self.spec.outcome_kind(b.layout),
                                        seats=seats, teams=teams, episode_length=self._length(), uid=lu.uid,
                                        source="tournament", seed=lu.seed, block=b.id))
            out[b.id] = rows
        return out
```

Create `tests/unit/test_sp4_tournament_stats.py`:

```python
"""Tournament statistics (SP4 T5.3, spec block 3): candidate scores, lineage weights, the paired stratified
bootstrap, confirmed, repeated games."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.arena.tournament.schedule import gauntlet_blocks, score_blocks
from colosseum.arena.tournament.stats import SELF, candidate_scores, combine, select_winner
from game_helpers import CoopGame, EliminationFFA, FakeArena, TurnTakingGame, synthetic_players

PLAYERS = synthetic_players(["main@v10", "main@v20", "other@v5", "greedy", "random", "main@v1", "main@v2"])
NAMES = {(p.agent_id, p.network_id): n for n, p in PLAYERS.items()}
LINEAGE = {n: p.lineage for n, p in PLAYERS.items()}


def tables_for(blocks, results, *, deterministic=False):
    return candidate_scores(blocks, results, names=NAMES, lineage=LINEAGE,
                            group_of={(b.layout, b.candidate): "player" for b in blocks},
                            metric_of={"2p": "pairwise", "4p": "pairwise", "coop2": "score"},
                            deterministic=deterministic)


def test_scores_are_mean_pairwise_scores_with_equal_lineage_weights():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy", "main@v1", "main@v2"], PLAYERS, blocks=2, seed=0,
                             stage="A")
    strengths = {"main@v10": 0.0, "greedy": -100.0, "main@v1": 100.0, "main@v2": 100.0}
    (table,) = tables_for(blocks, FakeArena(spec, PLAYERS, strengths)(blocks)).values()
    assert table.columns == ["greedy", "main@v1", "main@v2"]
    assert table.cells().tolist() == [[1.0, 0.0, 0.0]]
    assert table.weights.tolist() == [0.5, 0.25, 0.25]     # lineages greedy and main: the snapshots share a weight
    assert table.scores().tolist() == [0.5]
    assert table.games["main@v10"] == {"greedy": 4, "main@v1": 4, "main@v2": 4} and table.blocks_of("greedy") == 2


def test_ffa_pairwise_scores_count_every_opposing_team():
    spec = EliminationFFA(4).spec
    blocks = gauntlet_blocks(spec, "4p", ["main@v10"], ["greedy", "random"], PLAYERS, blocks=1, seed=0, stage="A")
    arena = FakeArena(spec, PLAYERS, {"main@v10": 50.0, "greedy": 0.0, "random": -50.0})
    (table,) = tables_for(blocks, arena(blocks)).values()
    assert table.cells().tolist() == [[1.0, 1.0]] and table.scores().tolist() == [1.0]


def test_paired_bootstrap_confirms_only_a_real_gap():
    spec = TurnTakingGame().spec
    strengths = {"main@v10": 1.0, "main@v20": 0.0, "other@v5": 0.0, "greedy": 0.0, "random": -0.5}
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "main@v20", "other@v5"], ["greedy", "random"], PLAYERS,
                             blocks=150, seed=0, stage="B")
    tables = list(tables_for(blocks, FakeArena(spec, PLAYERS, strengths, seed=1)(blocks)).values())
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=1000, rng=np.random.default_rng(2)), tables)
    assert sel.winner == "main@v10" and sel.confirmed and sel.diff_ci95[0] > 0
    tables = list(tables_for(blocks, FakeArena(spec, PLAYERS, {**strengths, "main@v10": 0.0}, seed=3)(blocks)).values())
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=1000, rng=np.random.default_rng(4)), tables)
    assert not sel.confirmed and sel.runner_up in sel.indistinguishable


def test_the_bootstrap_interval_covers_the_true_score():
    """Coverage of the per-candidate 95 % interval over 200 synthetic gauntlets of 60 blocks per composition; the
    true score is the BT expectation (Gumbel performance: P(win) = sigmoid(difference)). Measured coverage of this
    percentile bootstrap: 0.936 over 1500 runs; with n = 200 a correct implementation gives >= 176 hits except
    with probability < 1e-3 (binomial, p = 0.936, z = -3.1). The seeds are fixed, so the test is deterministic."""
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy", "random"], PLAYERS, blocks=60, seed=0, stage="A")
    strengths = {"main@v10": 0.5, "greedy": 0.0, "random": -1.0}
    truth = 0.5 / (1 + np.exp(-0.5)) + 0.5 / (1 + np.exp(-1.5))
    arena = FakeArena(spec, PLAYERS, strengths, seed=5)
    rng = np.random.default_rng(6)
    hits = 0
    for _ in range(200):
        (table,) = tables_for(blocks, arena(blocks)).values()
        lo, hi = np.quantile(table.bootstrap_scores(400, rng)[:, 0], [0.025, 0.975])
        hits += lo <= truth <= hi
    assert hits >= 176


def test_repeated_deterministic_games_are_flagged_and_never_confirm():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "main@v20"], ["greedy"], PLAYERS, blocks=5, seed=0, stage="B")
    results = FakeArena(spec, PLAYERS, {"main@v10": 100.0, "main@v20": -100.0, "greedy": 0.0})(blocks)
    tables = list(tables_for(blocks, results, deterministic=True).values())
    assert tables[0].repeated == {("main@v10", "greedy"), ("main@v20", "greedy")}
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=200, rng=np.random.default_rng(1)), tables)
    assert sel.winner == "main@v10" and not sel.confirmed and sel.repeated_columns == [("2p", "greedy")]
    assert not list(tables_for(blocks, results, deterministic=False).values())[0].repeated


def test_score_layouts_average_team_scores_over_the_homogeneous_team_and_the_field():
    spec = CoopGame(size=2).spec
    blocks = score_blocks(spec, "coop2", ["main@v10"], ["greedy"], PLAYERS, blocks=3, seed=0, stage="A")
    arena = FakeArena(spec, PLAYERS, {"main@v10": 4.0, "greedy": 0.0}, noise=0.0)
    (table,) = tables_for(blocks, arena(blocks)).values()
    assert table.columns == [SELF, "greedy"] and table.cells().tolist() == [[4.0, 2.0]]
    assert table.scores().tolist() == [3.0]


def test_tables_need_the_same_blocks_for_every_candidate():
    spec = TurnTakingGame().spec
    a = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=2, seed=0, stage="A")
    b = gauntlet_blocks(spec, "2p", ["main@v20"], ["greedy"], PLAYERS, blocks=3, seed=0, stage="A")
    results = FakeArena(spec, PLAYERS, {"main@v10": 0, "main@v20": 0, "greedy": 0})(a + b)
    with pytest.raises(ValueError, match="same blocks"):
        tables_for(a + b, results)
```

Create `tests/unit/test_sp4_tournament_modes.py`:

```python
"""Tournament modes on a substituted results source (SP4 T5.3, spec block 3): gauntlet, round-robin, halving,
select, several layouts, role groups, the report schema."""
from __future__ import annotations

import json

import pytest

from colosseum.arena.tournament.modes import TournamentSettings, run
from colosseum.core.errors import ConfigError
from colosseum.envs.game import GameSpec, SeatSpec
from game_helpers import AsymmetricGame, CoopGame, FakeArena, TurnTakingGame, synthetic_players

LADDER = {"main@v1": -1.0, "main@v2": -0.5, "main@v3": 0.0, "main@v4": 0.5, "main@v5": 2.0,
          "greedy": 0.0, "random": -1.0}
CANDIDATES = [n for n in LADDER if "@v" in n]


def settings(**kw):
    base = dict(name="t", layouts=["2p"], layout_weights={"2p": 1.0}, blocks=40, halving_blocks=4,
                confirm_blocks=200, finalists=2, seed=3, bootstrap_samples=400)
    return TournamentSettings(**{**base, **kw})


def play(mode, spec, strengths, candidates, field, arena=None, players=None, roles=None, **kw):
    players = players or synthetic_players(strengths, roles)
    arena = arena or FakeArena(spec, players, strengths)
    return run(mode, spec=spec, players=players, candidates=candidates, field=field, settings=settings(**kw),
               play=arena).to_dict()


def test_select_halves_then_confirms_the_strongest_on_fresh_seeds():
    spec = TurnTakingGame().spec
    players = synthetic_players(LADDER)
    arena = FakeArena(spec, players, LADDER)
    report = play("select", spec, LADDER, CANDIDATES, ["greedy", "random"], arena=arena, players=players)
    rounds = report["rounds"]
    assert [r["blocks_per_composition"] for r in rounds] == [4, 8]
    assert [len(r["active"]) for r in rounds] == [5, 3] and [len(r["eliminated"]) for r in rounds] == [2, 1]
    sel = report["selection"]["player"]
    assert sel["winner"] == "main@v5" and sel["confirmed"] is True and sel["diff_ci95"][0] > 0
    assert "biased up" in sel["note"]
    stage_a = {i for call in arena.calls for i in call if "/A/" in i}
    stage_b = {i for call in arena.calls for i in call if "/B/" in i}
    assert stage_a and stage_b and not stage_a & stage_b
    tables = {(t["stage"], t["layout"]): t for t in report["tables"]}
    assert set(tables) == {("A", "2p"), ("B", "2p")}
    b_rows = tables[("B", "2p")]["rows"]
    assert b_rows[0]["name"] == "main@v5" and len(b_rows) == 2
    assert report["games"]["by_stage"]["B"]["blocks"] == 2 * 2 * 200
    json.dumps(report)


def test_select_does_not_confirm_indistinguishable_finalists():
    spec = TurnTakingGame().spec
    flat = {"main@v1": 0.0, "main@v2": 0.0, "main@v3": 0.0, "greedy": 0.0}
    report = play("select", spec, flat, ["main@v1", "main@v2", "main@v3"], ["greedy"], confirm_blocks=60)
    sel = report["selection"]["player"]
    assert sel["confirmed"] is False and sel["runner_up"] in sel["indistinguishable"]


def test_halving_accumulates_rounds_and_stops_at_the_finalists():
    spec = TurnTakingGame().spec
    report = play("halving", spec, LADDER, CANDIDATES, ["greedy"], finalists=1)
    assert [len(r["active"]) for r in report["rounds"]] == [5, 3, 2]
    assert [r["first_index"] for r in report["rounds"]] == [0, 4, 12]
    assert report["selection"] is None
    (table,) = [t for t in report["tables"] if t["stage"] == "A"]
    assert [r["name"] for r in table["rows"]] == ["main@v5"] and table["rows"][0]["blocks"] == 4 + 8 + 16
    with pytest.raises(ConfigError, match="mode"):
        play("swiss", spec, LADDER, CANDIDATES, ["greedy"])


def test_gauntlet_tables_worst_opponent_and_matrix():
    spec = TurnTakingGame().spec
    report = play("gauntlet", spec, LADDER, ["main@v5", "main@v1"], ["greedy", "random"], blocks=30)
    (table,) = report["tables"]
    top = table["rows"][0]
    assert top["name"] == "main@v5" and top["worst"]["opponent"] == "greedy"
    cell = table["matrix"]["main@v5"]["random"]
    assert (cell["blocks"], cell["games"]) == (30, 60) and len(cell["ci95"]) == 2


def test_several_layouts_are_combined_with_the_layout_weights():
    spec = GameSpec(roles=TurnTakingGame().spec.roles,
                    layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1)),
                             "2v": (SeatSpec("player", 0), SeatSpec("player", 1))})
    report = play("gauntlet", spec, LADDER, ["main@v5", "main@v1"], ["greedy"], layouts=["2p", "2v"],
                  layout_weights={"2p": 3.0, "2v": 1.0}, blocks=20)
    (combined,) = [t for t in report["tables"] if t["layout"] == "*"]
    per = {t["layout"]: {r["name"]: r["score"] for r in t["rows"]} for t in report["tables"] if t["layout"] != "*"}
    got = {r["name"]: r["score"] for r in combined["rows"]}["main@v5"]
    assert got == pytest.approx(0.75 * per["2p"]["main@v5"] + 0.25 * per["2v"]["main@v5"])


def test_asymmetric_roles_select_one_winner_per_role_group():
    spec = AsymmetricGame().spec
    strengths = {"h@v1": 1.0, "h@v2": -1.0, "y@v1": 1.0, "y@v2": -1.0, "bot": 0.0}
    roles = {"h@v1": ("hunter",), "h@v2": ("hunter",), "y@v1": ("prey",), "y@v2": ("prey",), "bot": ("hunter", "prey")}
    report = play("select", spec, strengths, ["h@v1", "h@v2", "y@v1", "y@v2"], ["bot"], roles=roles,
                  layouts=["1v2"], layout_weights={"1v2": 1.0})
    assert {g: s["winner"] for g, s in report["selection"].items()} == {"hunter": "h@v1", "prey": "y@v1"}


def test_score_layouts_and_the_mixed_kind_rule():
    spec = CoopGame(size=2).spec
    strengths = {"main@v1": 1.0, "main@v2": 3.0, "helper": 0.0}
    report = play("select", spec, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2"],
                  layout_weights={"coop2": 1.0}, confirm_blocks=30)
    assert report["selection"]["player"]["winner"] == "main@v2" and report["bt"] == {"coop2": None}
    table = next(t for t in report["tables"] if t["stage"] == "B")
    assert table["metric"] == "score" and table["columns"] == ["(self)", "helper"]
    mixed = GameSpec(roles=spec.roles, layouts={"coop2": spec.layouts["coop2"],
                                                "2p": (SeatSpec("player", 0), SeatSpec("player", 1))})
    with pytest.raises(ConfigError, match="score"):
        play("select", mixed, strengths, ["main@v1", "main@v2"], ["helper"], layouts=["coop2", "2p"],
             layout_weights={"coop2": 1.0, "2p": 1.0})


def test_round_robin_matrix_is_antisymmetric():
    spec = TurnTakingGame().spec
    report = play("round-robin", spec, LADDER, ["main@v5", "main@v1"], ["greedy"], blocks=20)
    (table,) = report["tables"]
    m = table["matrix"]
    assert m["main@v5"]["greedy"]["score"] == pytest.approx(1 - m["greedy"]["main@v5"]["score"])
    assert table["rows"][0]["name"] == "main@v5"


def test_the_report_text_names_tables_and_the_selection():
    spec = TurnTakingGame().spec
    players = synthetic_players(LADDER)
    rep = run("select", spec=spec, players=players, candidates=CANDIDATES, field=["greedy"], settings=settings(),
              play=FakeArena(spec, players, LADDER))
    text = rep.text()
    assert "stage B" in text and "winner main@v5 (confirmed" in text and "halving player round 0" in text


def test_bt_section_rates_the_tournament_matches_with_block_intervals():
    """The tournament's own BT ratings (T2.1 library): order follows the true strengths on 120 blocks per
    composition (BT differences of 1.5 logits between neighbours: a reversal at this size is far below 1e-3)."""
    from colosseum.arena.tournament.modes import bt_section
    from colosseum.arena.tournament.schedule import gauntlet_blocks

    spec = TurnTakingGame().spec
    strengths = {"main@v1": -1.5, "main@v2": 0.0, "main@v3": 1.5, "greedy": 0.0}
    players = synthetic_players(strengths)
    blocks = gauntlet_blocks(spec, "2p", ["main@v1", "main@v2", "main@v3"], ["greedy"], players, blocks=120, seed=0,
                             stage="A")
    played = FakeArena(spec, players, strengths)(blocks)
    results = [r for b in blocks for r in played[b.id]]
    section = bt_section("2p", results, blocks, players=players, samples=50, seed=0)
    names = [e["name"] for e in section["entities"]]
    assert names.index("main@v3") < names.index("main@v2") < names.index("main@v1")
    for e in section["entities"]:
        assert e["ci95"][0] <= e["rating"] <= e["ci95"][1] and 0.0 <= e["p_best"] <= 1.0
    assert section["transitivity"]["flag"] is False and section["converged"] is True
    json.dumps(section)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_tournament_stats.py tests/unit/test_sp4_tournament_modes.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.arena.tournament.stats'`.

- [ ] **Step 3: Write `src/colosseum/arena/tournament/stats.py`**

```python
"""Tournament statistics (SP4 spec block 3): candidate scores, the paired stratified bootstrap, ``confirmed``,
repeated deterministic games.

Tables. ``candidate_scores`` turns played blocks into one ``CandidateTable`` per (stage, layout, role group):
- **columns** are the opponents the score is averaged over: the field members (pairwise layouts) or
  ``SELF`` (the candidate's homogeneous team) plus the field members it was paired with (one-team ``score``
  layouts);
- **strata** are the compositions (``Block.stratum``) and **indices** the block indices of each stratum;
  every candidate of a table has played the same (stratum, index) blocks (the same seeds);
- ``values[c, s, k, col]`` is the block's mean for that column: pairwise layouts average the candidate's
  pairwise score against the column's team (1 / 0.5 / 0 by team ranks, over the block's rotations and over
  every team that member holds); ``score`` layouts average the team score over the block's matches.

Candidate score: ``S_c = sum_col w_col * cell(c, col)`` with ``cell`` the mean of the column's block values
over every block that holds the column and ``w_col`` equal per lineage (``1 / lineages``), split equally
between the columns of a lineage (snapshots of one run weigh as one opponent). The worst opponent is the
column with the smallest cell.

Paired stratified bootstrap: a replicate redraws, with replacement, the block indices within every stratum,
the SAME draw for every candidate (common seeds), and recomputes every cell and score; several layouts are
redrawn independently and combined per replicate with the layout weights. Intervals are the 2.5 % / 97.5 %
quantiles of the replicates. ``confirmed``: the 95 % interval of ``S_winner - S_runner_up`` lies above 0, and
no column where the two differ is a repeated pair of either (below).

Repeated games (``--deterministic`` only): a (candidate, column) pair whose every block gave the same
outcomes and lengths (two or more blocks) is flagged. It counts as one block: its per-cell interval is the
Wilson interval of one observation (pairwise) or the point itself (score), and a lead that rests on such a
column is never ``confirmed`` (spec block 3: «Такие пары в ДИ и confirmed считаются одним блоком»).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from colosseum.core.outcomes import pairwise_rank_score
from colosseum.core.types import MatchResult
from colosseum.eval import wilson_interval

SELF = "(self)"
PAIRED_BOOTSTRAP_SAMPLES = 2000   # provisional; T8.2 confirms or changes it (ruling)
_CHUNK = 250                      # replicates per vectorized chunk (memory bound)


@dataclass
class CandidateTable:
    stage: str
    layout: str
    metric: str                          # "pairwise" | "score"
    group: str
    candidates: list[str]
    columns: list[str]
    lineage: dict[str, str]              # column -> lineage
    strata: list[str]
    indices: dict[str, list[int]]        # stratum -> block indices (sorted)
    values: np.ndarray                   # [C, S, Kmax, L] block means; 0 where absent
    present: np.ndarray                  # [S, L] bool: the stratum holds the column
    games: dict[str, dict[str, int]]     # candidate -> column -> comparisons (pairwise) or matches (score)
    repeated: set[tuple[str, str]] = field(default_factory=set)

    @property
    def weights(self) -> np.ndarray:
        """Column weights: equal per lineage, split within a lineage (module docstring); sums to 1."""
        lineages = sorted({self.lineage[c] for c in self.columns})
        per = {lin: sum(1 for c in self.columns if self.lineage[c] == lin) for lin in lineages}
        return np.array([1.0 / (len(lineages) * per[self.lineage[c]]) for c in self.columns])

    def blocks_of(self, column: str) -> int:
        j = self.columns.index(column)
        return int(sum(len(self.indices[s]) for i, s in enumerate(self.strata) if self.present[i, j]))

    def cells(self) -> np.ndarray:
        """Point estimates ``[C, L]``."""
        k_mask = self._k_mask()                                       # [S, Kmax]
        den = (k_mask.sum(axis=1)[:, None] * self.present).sum(axis=0)  # [L]
        num = self.values.sum(axis=2).sum(axis=1)                     # [C, L] (absent entries are 0)
        return num / np.maximum(den, 1)

    def scores(self) -> np.ndarray:
        """Candidate scores ``[C]``."""
        return self.cells() @ self.weights

    def _k_mask(self) -> np.ndarray:
        kmax = self.values.shape[2]
        return np.array([[k < len(self.indices[s]) for k in range(kmax)] for s in self.strata], dtype=bool)

    def bootstrap_cells(self, samples: int, rng: np.random.Generator) -> np.ndarray:
        """Paired stratified bootstrap replicates of every cell, ``[samples, C, L]`` (module docstring)."""
        kmax = self.values.shape[2]
        sizes = [len(self.indices[s]) for s in self.strata]
        den = (np.array(sizes)[:, None] * self.present).sum(axis=0)   # [L]: constant over replicates
        out = np.empty((samples, len(self.candidates), len(self.columns)))
        values = self.values.astype(np.float64)
        for start in range(0, samples, _CHUNK):
            r = min(_CHUNK, samples - start)
            counts = np.zeros((r, len(self.strata), kmax))
            for i, size in enumerate(sizes):
                counts[:, i, :size] = rng.multinomial(size, np.full(size, 1.0 / size), size=r)
            out[start:start + r] = np.einsum("rsk,cskl->rcl", counts, values) / np.maximum(den, 1)
        return out

    def bootstrap_scores(self, samples: int, rng: np.random.Generator) -> np.ndarray:
        """``[samples, C]`` candidate-score replicates."""
        return self.bootstrap_cells(samples, rng) @ self.weights


def _rotation(uid: str) -> int:
    return int(uid.rsplit("/r", 1)[1])


def _team_names(result: MatchResult, names: Mapping[tuple[str, str], str]) -> dict[int, str]:
    teams: dict[int, set[str]] = defaultdict(set)
    for seat in result.seats:
        teams[seat.team].add(names[(seat.agent_id, seat.network_id)])
    out = {}
    for team, members in teams.items():
        if len(members) != 1:
            raise ValueError(f"match {result.uid}: team {team} mixes players {sorted(members)}")
        out[team] = next(iter(members))
    return out


def block_values(block, results: Sequence[MatchResult], names: Mapping[tuple[str, str], str],
                 metric: str) -> dict[str, tuple[float, int, tuple]]:
    """``{column: (block mean, comparisons or matches, signature)}`` of one played block (module docstring);
    the signature (sorted ``(rotation, value, length)``) detects repeated games."""
    if len(results) != len(block.lineups):
        raise ValueError(f"block {block.id}: {len(results)} results for {len(block.lineups)} matches")
    acc: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for result in results:
        rotation = _rotation(result.uid)
        if metric == "score":
            column = block.composition[0] if block.composition else SELF
            acc[column].append((rotation, float(result.teams[0].score), int(result.episode_length)))
            continue
        teams = _team_names(result, names)
        ranks = {t.team: t.rank for t in result.teams}
        own = [t for t, name in teams.items() if name == block.candidate]
        if len(own) != 1:
            raise ValueError(f"match {result.uid}: candidate {block.candidate!r} holds {len(own)} teams")
        for team, name in teams.items():
            if team != own[0]:
                acc[name].append((rotation, pairwise_rank_score(ranks[own[0]], ranks[team]),
                                  int(result.episode_length)))
    return {col: (float(np.mean([v for _r, v, _l in rows])), len(rows), tuple(sorted(rows)))
            for col, rows in acc.items()}


def candidate_scores(blocks: Sequence, results_by_block: Mapping[str, Sequence[MatchResult]], *,
                     names: Mapping[tuple[str, str], str], lineage: Mapping[str, str],
                     group_of: Mapping[tuple[str, str], str], metric_of: Mapping[str, str],
                     deterministic: bool = False) -> dict[tuple[str, str, str], CandidateTable]:
    """One ``CandidateTable`` per (stage, layout, group) of ``blocks`` (all played: ``results_by_block`` holds
    every block id). ``names``: (agent_id, network_id) -> player name; ``lineage``: player name -> lineage;
    ``group_of``: (layout, candidate) -> group label; ``metric_of``: layout -> "pairwise" | "score".
    ValueError when the candidates of a table did not play the same blocks."""
    raw: dict[tuple[str, str, str], dict[str, dict[tuple[str, int], dict[str, tuple]]]] = \
        defaultdict(lambda: defaultdict(dict))
    for block in blocks:
        key = (block.stage, block.layout, group_of[(block.layout, block.candidate)])
        raw[key][block.candidate][(block.stratum, block.index)] = block_values(
            block, results_by_block[block.id], names, metric_of[block.layout])
    tables = {}
    for (stage, layout, group), per_candidate in raw.items():
        candidates = list(per_candidate)
        cells = {c: set(per_candidate[c]) for c in candidates}
        first = cells[candidates[0]]
        if any(cells[c] != first for c in candidates):
            raise ValueError(f"stage {stage}, layout {layout}, group {group}: the candidates did not play the same "
                             f"blocks")
        strata = sorted({s for s, _k in first})
        indices = {s: sorted(k for s2, k in first if s2 == s) for s in strata}
        columns = sorted({col for c in candidates for cell in per_candidate[c].values() for col in cell},
                         key=lambda col: (col != SELF, col))
        kmax = max(len(v) for v in indices.values())
        values = np.zeros((len(candidates), len(strata), kmax, len(columns)))
        present = np.zeros((len(strata), len(columns)), dtype=bool)
        games = {c: dict.fromkeys(columns, 0) for c in candidates}
        sigs: dict[tuple[str, str], set] = defaultdict(set)
        nblocks: dict[tuple[str, str], int] = defaultdict(int)
        for ci, c in enumerate(candidates):
            for si, s in enumerate(strata):
                for ki, k in enumerate(indices[s]):
                    for col, (mean, n, sig) in per_candidate[c][(s, k)].items():
                        j = columns.index(col)
                        values[ci, si, ki, j] = mean
                        present[si, j] = True
                        games[c][col] += n
                        sigs[(c, col)].add(sig)
                        nblocks[(c, col)] += 1
        repeated = {pair for pair, found in sigs.items() if deterministic and nblocks[pair] >= 2 and len(found) == 1}
        tables[(stage, layout, group)] = CandidateTable(
            stage=stage, layout=layout, metric=metric_of[layout], group=group, candidates=candidates,
            columns=columns, lineage={col: (SELF if col == SELF else lineage[col]) for col in columns},
            strata=strata, indices=indices, values=values, present=present, games=games, repeated=repeated)
    return tables


def quantiles(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def cell_interval(table: CandidateTable, candidate: str, column: str, replicates: np.ndarray,
                  level: float = 0.95) -> tuple[float, float]:
    """95 % interval of one cell from its bootstrap replicates ``[R, C, L]``; a repeated pair counts as one
    block (Wilson of one observation for pairwise, the point for score)."""
    ci, j = table.candidates.index(candidate), table.columns.index(column)
    value = float(table.cells()[ci, j])
    if (candidate, column) in table.repeated:
        return wilson_interval(value, 1) if table.metric == "pairwise" else (value, value)
    return quantiles(replicates[:, ci, j], level)


def repeated_conflict(tables: Sequence[CandidateTable], a: str, b: str) -> list[tuple[str, str]]:
    """``(layout, column)`` pairs where ``a`` and ``b`` differ and either one's games repeat."""
    out = []
    for t in tables:
        if a not in t.candidates or b not in t.candidates:
            continue
        cells = t.cells()
        ia, ib = t.candidates.index(a), t.candidates.index(b)
        for j, col in enumerate(t.columns):
            if ((a, col) in t.repeated or (b, col) in t.repeated) and not np.isclose(cells[ia, j], cells[ib, j]):
                out.append((t.layout, col))
    return out


@dataclass
class Combined:
    """Candidate scores of one (stage, group) over its layouts, with per-replicate combination."""

    candidates: list[str]
    point: np.ndarray        # [C]
    replicates: np.ndarray   # [R, C]


def combine(tables: Sequence[CandidateTable], weights: Mapping[str, float], *, samples: int,
            rng: np.random.Generator) -> Combined:
    """Layout-weighted scores of the candidates common to every table, per replicate (module docstring)."""
    candidates = [c for c in tables[0].candidates if all(c in t.candidates for t in tables)]
    total = sum(weights[t.layout] for t in tables)
    point = np.zeros(len(candidates))
    reps = np.zeros((samples, len(candidates)))
    for t in tables:
        idx = [t.candidates.index(c) for c in candidates]
        w = weights[t.layout] / total
        point += w * t.scores()[idx]
        reps += w * t.bootstrap_scores(samples, rng)[:, idx]
    return Combined(candidates=candidates, point=point, replicates=reps)


def confirmed(diff_replicates: np.ndarray, *, level: float = 0.95, conflict: bool = False) -> bool:
    """The ``level`` interval of ``S_winner - S_runner_up`` lies above 0 and no repeated column decides it."""
    return (not conflict) and quantiles(diff_replicates, level)[0] > 0


@dataclass
class Selection:
    winner: str
    runner_up: str | None
    confirmed: bool
    diff: float | None
    diff_ci95: tuple[float, float] | None
    indistinguishable: list[str]
    repeated_columns: list[tuple[str, str]]


def select_winner(combined: Combined, tables: Sequence[CandidateTable], *, level: float = 0.95) -> Selection:
    """The winner by point score (ties: input order), ``confirmed`` against the runner-up and every finalist
    whose difference interval contains 0 (``indistinguishable``)."""
    order = sorted(range(len(combined.candidates)), key=lambda i: (-combined.point[i], i))
    w = order[0]
    winner = combined.candidates[w]
    if len(order) == 1:
        return Selection(winner, None, True, None, None, [], [])
    u = order[1]
    runner = combined.candidates[u]
    diff = combined.replicates[:, w] - combined.replicates[:, u]
    conflict = repeated_conflict(tables, winner, runner)
    indistinct = []
    for i in order[1:]:
        d = combined.replicates[:, w] - combined.replicates[:, i]
        name = combined.candidates[i]
        if quantiles(d, level)[0] <= 0 or repeated_conflict(tables, winner, name):
            indistinct.append(name)
    return Selection(winner=winner, runner_up=runner, confirmed=confirmed(diff, level=level, conflict=bool(conflict)),
                     diff=float(combined.point[w] - combined.point[u]), diff_ci95=quantiles(diff, level),
                     indistinguishable=indistinct, repeated_columns=conflict)
```

- [ ] **Step 4: Write `src/colosseum/arena/tournament/modes.py`**

```python
"""Tournament modes (SP4 spec block 3): gauntlet, round-robin, halving, select; the report.

Every mode builds blocks (``schedule``), asks ``play(blocks)`` for their results (the tournament runner: the
journal for blocks already written whole, worker processes for the rest; tests pass a fake) and turns them
into tables (``stats``):
- ``gauntlet``: every candidate plays ``blocks`` blocks per composition on stage A; tables per layout and role
  group, plus a layout-weighted combination per group when there are several layouts;
- ``round-robin``: every pair of participants (candidates and field) plays ``blocks`` blocks per pair on stage
  A (two-team layouts only); a score matrix with intervals;
- ``halving``: successive halving on stage A, per role group. Round ``r`` plays ``halving_blocks * 2**r`` new
  blocks per composition for the active candidates (indices continue, so every round has fresh seeds and
  every active candidate the same ones); ranking by the combined score over all blocks played so far; the
  lower ``floor(n / 2)`` leave, but never below ``finalists``; rounds end when ``finalists`` are left (one round
  when there are no more than ``finalists`` to begin with);
- ``select``: ``halving`` (stage A) to ``finalists``, then a gauntlet of the finalists with ``confirm_blocks``
  per composition on fresh seeds (stage B). The winner per group is the best stage B score; ``confirmed`` by
  the paired stratified bootstrap of stage B only (``stats.select_winner``). Stage A games never enter stage B
  numbers (winner's curse); the winner's own stage B score is still biased up by the selection among the
  finalists (the report says so), the interval of the difference is not.

Several layouts: per-layout tables, and a combination with ``layout_weights`` (``matchmaking.layouts``,
default equal) computed on the same bootstrap replicates. ``score`` layouts (one team) use the mean team score;
``halving`` and ``select`` refuse to mix them with pairwise layouts (ConfigError: choose ``--layout``).
The BT rating of every pairwise layout is fitted on the tournament's own matches (``bt_section``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum.arena.tournament.schedule import (
    Block,
    check_role_groups,
    gauntlet_blocks,
    round_robin_blocks,
    score_blocks,
)
from colosseum.arena.tournament.stats import (
    PAIRED_BOOTSTRAP_SAMPLES,
    SELF,
    CandidateTable,
    block_values,
    candidate_scores,
    cell_interval,
    combine,
    quantiles,
    select_winner,
)
from colosseum.core.errors import ConfigError
from colosseum.core.types import MatchResult
from colosseum.envs.game import GameSpec

MODES = ("gauntlet", "round-robin", "halving", "select")
REPORT_FORMAT = 1
# Provisional defaults; T8.2 measures them on the bot ladder and the controller fixes them as rulings.
DEFAULT_BLOCKS = 50            # gauntlet / round-robin: blocks per composition (pair)
DEFAULT_HALVING_BLOCKS = 8     # halving round 0: blocks per composition
DEFAULT_CONFIRM_BLOCKS = 100   # select stage B: blocks per composition
DEFAULT_FINALISTS = 3
WINNER_BIAS_NOTE = ("the winner's stage B score is biased up by the selection among the finalists; the interval "
                    "of its difference to the runner-up is not")

PlayFn = Callable[[Sequence[Block]], Mapping[str, Sequence[MatchResult]]]
BTSectionFn = Callable[[str, Sequence[MatchResult], Sequence[Block]], "dict | None"]


@dataclass(frozen=True)
class TournamentSettings:
    name: str                              # the tournament id: block-id prefix and journal key
    layouts: list[str]
    layout_weights: dict[str, float]
    blocks: int = DEFAULT_BLOCKS
    halving_blocks: int = DEFAULT_HALVING_BLOCKS
    confirm_blocks: int = DEFAULT_CONFIRM_BLOCKS
    finalists: int = DEFAULT_FINALISTS
    seed: int = 0
    deterministic: bool = False
    bootstrap_samples: int = PAIRED_BOOTSTRAP_SAMPLES
    level: float = 0.95


@dataclass
class TournamentReport:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def text(self) -> str:
        return report_text(self.data)


def _player_row(players: Mapping[str, Any], name: str) -> dict[str, Any]:
    p = players[name]
    return {"name": name, "display": p.display, "kind": p.kind, "lineage": p.lineage, "roles": list(p.roles),
            "source": p.source.origin}


class _Context:
    def __init__(self, spec: GameSpec, players: Mapping[str, Any], candidates: Sequence[str], field_: Sequence[str],
                 settings: TournamentSettings, play: PlayFn) -> None:
        self.spec, self.players, self.settings, self.play = spec, players, settings, play
        self.candidates, self.field = list(candidates), list(field_)
        self.names = {(p.agent_id, p.network_id): n for n, p in players.items()}
        self.lineage = {n: p.lineage for n, p in players.items()}
        self.metric_of = {L: ("score" if spec.num_teams(L) == 1 else "pairwise") for L in settings.layouts}
        self.blocks: list[Block] = []
        self.results: dict[str, list[MatchResult]] = {}
        self.rng = np.random.default_rng(np.random.SeedSequence([int(settings.seed) % 2**63, 0x7EB0]))

    def run(self, blocks: Sequence[Block]) -> None:
        blocks = [b for b in blocks if b.id not in self.results]
        if not blocks:
            return
        got = self.play(blocks)
        for b in blocks:
            if b.id not in got:
                raise ValueError(f"tournament: the results source returned nothing for block {b.id}")
            self.results[b.id] = list(got[b.id])
        self.blocks.extend(blocks)

    def schedule(self, layout: str, candidates: Sequence[str], *, blocks: int, stage: str,
                 first_index: int = 0) -> list[Block]:
        common = dict(blocks=blocks, seed=self.settings.seed, stage=stage, first_index=first_index,
                      prefix=self.settings.name)
        if self.metric_of[layout] == "score":
            return score_blocks(self.spec, layout, candidates, self.field, self.players, **common)
        return gauntlet_blocks(self.spec, layout, candidates, self.field, self.players, **common)

    def tables(self, blocks: Sequence[Block], groups: Mapping[tuple[str, str], str]) -> dict:
        return candidate_scores(blocks, self.results, names=self.names, lineage=self.lineage, group_of=groups,
                                metric_of=self.metric_of, deterministic=self.settings.deterministic)


def _groups(ctx: _Context) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    """``{group: candidates}`` over all layouts and ``{(layout, candidate): group}``; a candidate must be in one
    group label in every layout (ConfigError otherwise)."""
    group_of: dict[tuple[str, str], str] = {}
    members: dict[str, list[str]] = {}
    for layout in ctx.settings.layouts:
        for label, names in check_role_groups(ctx.spec, layout, ctx.candidates, ctx.players).items():
            for name in names:
                group_of[(layout, name)] = label
    for name in ctx.candidates:
        labels = {group_of[(L, name)] for L in ctx.settings.layouts}
        if len(labels) > 1:
            raise ConfigError(f"candidate {name!r} plays different role groups in the layouts {ctx.settings.layouts} "
                              f"({sorted(labels)}); choose layouts with --layout")
        members.setdefault(labels.pop(), []).append(name)
    return members, group_of


def _table_dict(ctx: _Context, table: CandidateTable) -> dict[str, Any]:
    reps = table.bootstrap_cells(ctx.settings.bootstrap_samples, ctx.rng)
    score_reps = reps @ table.weights
    cells, scores = table.cells(), table.scores()
    rows = []
    for i, c in enumerate(table.candidates):
        cols = {col: float(cells[i, j]) for j, col in enumerate(table.columns)}
        opponents = {col: v for col, v in cols.items() if col != SELF}
        worst = min(opponents, key=lambda col: (opponents[col], col)) if opponents else None
        rows.append({
            "name": c, "score": float(scores[i]), "ci95": list(quantiles(score_reps[:, i], ctx.settings.level)),
            "worst": None if worst is None else {"opponent": worst, "score": opponents[worst]},
            "games": int(sum(table.games[c].values())),
            "blocks": int(sum(len(v) for v in table.indices.values())),
            "repeated": sorted(col for (cand, col) in table.repeated if cand == c),
        })
    rows.sort(key=lambda r: (-r["score"], table.candidates.index(r["name"])))
    matrix = {c: {col: {"score": float(cells[i, j]),
                        "ci95": list(cell_interval(table, c, col, reps, ctx.settings.level)),
                        "games": int(table.games[c][col]), "blocks": table.blocks_of(col),
                        "repeated": (c, col) in table.repeated}
                  for j, col in enumerate(table.columns)}
              for i, c in enumerate(table.candidates)}
    return {"stage": table.stage, "layout": table.layout, "group": table.group, "metric": table.metric,
            "columns": list(table.columns), "column_weights": [float(w) for w in table.weights],
            "rows": rows, "matrix": matrix}


def _combined_dict(stage: str, group: str, combined, level: float) -> dict[str, Any]:
    rows = [{"name": c, "score": float(combined.point[i]), "ci95": list(quantiles(combined.replicates[:, i], level))}
            for i, c in enumerate(combined.candidates)]
    rows.sort(key=lambda r: -r["score"])
    return {"stage": stage, "layout": "*", "group": group, "rows": rows}


def _stage_tables(ctx: _Context, stage: str, names: Sequence[str], group: str, group_of) -> list[CandidateTable]:
    blocks = [b for b in ctx.blocks if b.stage == stage and b.candidate in names]
    tables = ctx.tables(blocks, group_of)
    return [tables[(stage, L, group)] for L in ctx.settings.layouts if (stage, L, group) in tables]


def _halving(ctx: _Context, group: str, names: list[str], group_of, rounds: list[dict]) -> list[str]:
    s = ctx.settings
    active, offset, size, r = list(names), 0, s.halving_blocks, 0
    while True:
        for layout in s.layouts:
            ctx.run(ctx.schedule(layout, active, blocks=size, stage="A", first_index=offset))
        tables = _stage_tables(ctx, "A", active, group, group_of)
        point = combine(tables, s.layout_weights, samples=1, rng=ctx.rng).point
        order = sorted(range(len(active)), key=lambda i: (-point[i], names.index(active[i])))
        ranked = [active[i] for i in order]
        keep = len(active) if len(active) <= s.finalists else max(s.finalists, len(active) - len(active) // 2)
        rounds.append({"group": group, "round": r, "stage": "A", "blocks_per_composition": size,
                       "first_index": offset, "active": list(ranked),
                       "scores": {active[i]: float(point[i]) for i in order},
                       "eliminated": ranked[keep:]})
        active = ranked[:keep]
        offset, size, r = offset + size, size * 2, r + 1
        if len(active) <= s.finalists:
            return active


def run(mode: str, *, spec: GameSpec, players: Mapping[str, Any], candidates: Sequence[str], field: Sequence[str],
        settings: TournamentSettings, play: PlayFn, bt_fn: BTSectionFn | None = None) -> TournamentReport:
    """Play ``mode`` (module docstring) and build the report."""
    if mode not in MODES:
        raise ConfigError(f"--mode {mode!r}: expected one of {list(MODES)}")
    started = time.monotonic()
    ctx = _Context(spec, players, candidates, field, settings, play)
    metrics = set(ctx.metric_of.values())
    if mode in ("halving", "select") and len(metrics) > 1:
        raise ConfigError(f"--mode {mode}: the layouts {settings.layouts} mix score (one team) and win/rank layouts, "
                          f"whose results cannot be averaged; choose layouts of one kind with --layout")
    tables: list[dict] = []
    rounds: list[dict] = []
    selection: dict[str, dict] = {}
    if mode == "round-robin":
        participants = list(dict.fromkeys([*candidates, *field]))
        for layout in settings.layouts:
            ctx.run(round_robin_blocks(spec, layout, participants, players, blocks=settings.blocks,
                                       seed=settings.seed, stage="A", prefix=settings.name))
        tables.extend(_round_robin_tables(ctx, participants))
    else:
        groups, group_of = _groups(ctx)
        for group, names in groups.items():
            if mode == "gauntlet":
                for layout in settings.layouts:
                    ctx.run(ctx.schedule(layout, names, blocks=settings.blocks, stage="A"))
                stages = [("A", names)]
            else:
                finalists = _halving(ctx, group, names, group_of, rounds)
                stages = [("A", finalists)]        # stage A table: the blocks every finalist played
                if mode == "select":
                    for layout in settings.layouts:
                        ctx.run(ctx.schedule(layout, finalists, blocks=settings.confirm_blocks, stage="B"))
                    stages.append(("B", finalists))
            for stage, members in stages:
                stage_tables = _stage_tables(ctx, stage, members, group, group_of)
                tables.extend(_table_dict(ctx, t) for t in stage_tables)
                if len(stage_tables) == 1 and not (mode == "select" and stage == "B"):
                    continue
                comb = combine(stage_tables, settings.layout_weights, samples=settings.bootstrap_samples, rng=ctx.rng)
                if len(stage_tables) > 1:
                    tables.append(_combined_dict(stage, group, comb, settings.level))
                if mode == "select" and stage == "B":
                    sel = select_winner(comb, stage_tables, level=settings.level)
                    selection[group] = {
                        "winner": sel.winner, "runner_up": sel.runner_up, "confirmed": sel.confirmed,
                        "diff": sel.diff, "diff_ci95": None if sel.diff_ci95 is None else list(sel.diff_ci95),
                        "indistinguishable": sel.indistinguishable,
                        "repeated_columns": [list(x) for x in sel.repeated_columns], "note": WINNER_BIAS_NOTE}
    bt: dict[str, dict | None] = {}
    for layout in settings.layouts:
        if bt_fn is not None and ctx.metric_of[layout] == "pairwise":
            layout_blocks = [b for b in ctx.blocks if b.layout == layout]
            results = [r for b in layout_blocks for r in ctx.results[b.id]]
            bt[layout] = bt_fn(layout, results, layout_blocks)
        else:
            bt[layout] = None
    for t in tables:                      # the BT columns of every per-layout table row (spec block 3)
        section = bt.get(t["layout"])
        by_name = {e["name"]: e for e in section["entities"]} if section else {}
        for row in t["rows"]:
            e = by_name.get(row["name"])
            row["bt"] = None if e is None else {"rating": e["rating"], "ci95": e["ci95"], "p_best": e["p_best"]}
    data = {
        "format": REPORT_FORMAT, "kind": "tournament", "name": settings.name, "mode": mode,
        "seed": settings.seed, "deterministic": settings.deterministic, "ci_level": settings.level,
        "bootstrap_samples": settings.bootstrap_samples,
        "settings": {"blocks": settings.blocks, "halving_blocks": settings.halving_blocks,
                     "confirm_blocks": settings.confirm_blocks, "finalists": settings.finalists},
        "candidates": [_player_row(players, n) for n in candidates],
        "field": [_player_row(players, n) for n in field],
        "layouts": [{"layout": L, "outcome_kind": spec.outcome_kind(L), "metric": ctx.metric_of[L],
                     "weight": float(settings.layout_weights[L])} for L in settings.layouts],
        "tables": tables, "rounds": rounds, "selection": selection or None, "bt": bt,
        "games": {"matches": sum(len(v) for v in ctx.results.values()), "blocks": len(ctx.results),
                  "by_stage": {st: {"blocks": sum(1 for b in ctx.blocks if b.stage == st),
                                    "matches": sum(len(ctx.results[b.id]) for b in ctx.blocks if b.stage == st)}
                               for st in sorted({b.stage for b in ctx.blocks})}},
        "flags": {"repeated_pairs": sorted({(t["layout"], r["name"], col) for t in tables if "matrix" in t
                                            for r in t["rows"] for col in r.get("repeated", [])}),
                  "non_transitive": sorted(L for L, sec in bt.items()
                                           if sec and sec.get("transitivity", {}).get("flag"))},
        "elapsed_sec": time.monotonic() - started,
    }
    data["flags"]["repeated_pairs"] = [list(x) for x in data["flags"]["repeated_pairs"]]
    return TournamentReport(data)


def _round_robin_tables(ctx: _Context, participants: Sequence[str]) -> list[dict]:
    """Pairwise matrix per layout: ``cells[a][b]`` = a's mean pairwise score against b (1 - b's against a), with a
    bootstrap interval over the pair's blocks."""
    out = []
    for layout in ctx.settings.layouts:
        per_pair: dict[tuple[str, str], list[float]] = {}
        for b in ctx.blocks:
            if b.layout != layout:
                continue
            values = block_values(b, ctx.results[b.id], ctx.names, "pairwise")
            per_pair.setdefault((b.candidate, b.composition[0]), []).append(values[b.composition[0]][0])
        matrix: dict[str, dict[str, dict]] = {a: {} for a in participants}
        for (a, b), vals in per_pair.items():
            x = np.asarray(vals)
            idx = ctx.rng.integers(len(x), size=(ctx.settings.bootstrap_samples, len(x)))
            lo, hi = quantiles(x[idx].mean(axis=1), ctx.settings.level)
            matrix[a][b] = {"score": float(x.mean()), "ci95": [lo, hi], "blocks": len(x)}
            matrix[b][a] = {"score": float(1 - x.mean()), "ci95": [1 - hi, 1 - lo], "blocks": len(x)}
        rows = []
        for a in participants:
            row = matrix[a]
            rows.append({"name": a, "score": float(np.mean([c["score"] for c in row.values()])) if row else 0.0,
                         "opponents": len(row)})
        rows.sort(key=lambda r: -r["score"])
        out.append({"stage": "A", "layout": layout, "group": "*", "metric": "pairwise",
                    "columns": list(participants), "rows": rows, "matrix": matrix})
    return out


def report_text(data: Mapping[str, Any]) -> str:
    """Plain-text report (``report.txt`` and the console)."""
    lines = [f"Tournament {data['name']} ({data['mode']}, seed {data['seed']}"
             f"{', deterministic' if data['deterministic'] else ''}): {data['games']['matches']} matches in "
             f"{data['games']['blocks']} blocks"]
    for t in data["tables"]:
        title = "all layouts (weighted)" if t["layout"] == "*" else f"layout {t['layout']}"
        lines.append(f"== stage {t['stage']}, {title}, group {t['group']}")
        for r in t["rows"]:
            ci = r.get("ci95")
            text = f"  {r['name']:<28} {r['score']:.3f}"
            if ci:
                text += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
            if r.get("worst"):
                text += f"  worst {r['worst']['opponent']} {r['worst']['score']:.3f}"
            if "games" in r:
                text += f"  games {r['games']}"
            if r.get("repeated"):
                text += f"  REPEATED vs {', '.join(r['repeated'])}"
            lines.append(text)
    for r in data["rounds"]:
        lines.append(f"-- halving {r['group']} round {r['round']}: {r['blocks_per_composition']} blocks/composition, "
                     f"out: {', '.join(r['eliminated']) or '-'}")
    for group, sel in (data["selection"] or {}).items():
        state = "confirmed" if sel["confirmed"] else "NOT confirmed"
        line = f"** {group}: winner {sel['winner']} ({state}"
        if sel["diff_ci95"] is not None:
            lo, hi = sel["diff_ci95"]
            line += f"; vs {sel['runner_up']}: {sel['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        lines.append(line + ")")
        if sel["indistinguishable"]:
            lines.append(f"   indistinguishable from the winner: {', '.join(sel['indistinguishable'])}")
        lines.append(f"   note: {sel['note']}")
    for layout, sec in data["bt"].items():
        if sec:
            anchors = ", ".join(f"{a['group']}: {a['anchor']}" for a in sec["anchor"])
            lines.append(f"== BT ratings, layout {layout} (anchor {anchors})")
            for e in sec["entities"]:
                lines.append(f"  {e['name']:<28} {e['rating']:+8.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]  "
                             f"P(best) {e['p_best']:.2f}  games {e['games']}")
            if sec["transitivity"]["flag"]:
                lines.append(f"  possible cycles (G2 p={sec['transitivity']['p_value']:.3g}): "
                             f"{sec['transitivity']['worst_pairs']}")
    if data["flags"]["repeated_pairs"]:
        lines.append("Games repeat (not enough randomness): " +
                     "; ".join(f"{L}: {c} vs {f}" for L, c, f in data["flags"]["repeated_pairs"]))
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_random_bot(class_path: str) -> bool:
    from colosseum.core.registry import import_class
    from colosseum.players import RandomBot

    try:
        return import_class(class_path) is RandomBot
    except Exception:  # noqa: BLE001 - not importable here: not the RandomBot
        return False


def bt_section(layout: str, results: Sequence[MatchResult], blocks: Sequence[Block], *, players: Mapping[str, Any],
               anchor: str | None = None, samples: int = 200, seed: int = 0) -> dict[str, Any]:
    """BT ratings of one pairwise layout from the tournament's own matches (spec block 3): ``arena.bt.fit`` on a
    ``ComparisonTable`` of ``results`` (``source`` tournament), intervals by ``bt.bootstrap`` over whole blocks
    (strata: stage and composition), ``P(best)`` within the role group, the G2 non-transitivity flag
    (p < 0.01). The anchor rule is the run's (``ratings.anchor``, else the first ``RandomBot``, else the mean of
    the fixed players); every non-snapshot participant counts as fixed."""
    from colosseum.arena import bt
    from colosseum.arena.comparisons import ComparisonTable
    from colosseum.arena.entities import snapshot_version

    table = ComparisonTable()
    for result in results:
        table.add_result(result, [snapshot_version(seat.network_id) for seat in result.seats])
    fixed = [name for name, p in players.items() if p.kind != "snapshot"]
    random_bot = next((name for name, p in players.items() if p.kind == "scripted"
                       and getattr(p.source, "bot", None) is not None and _is_random_bot(p.source.bot.class_path)),
                      None)
    fit = bt.fit(table, layout, sources=("tournament",), options=bt.default_options(anchor=anchor),
                 fixed_agents=fixed, random_bot=random_bot)
    by_block: dict[str, list[MatchResult]] = {}
    for result in results:
        by_block.setdefault(result.block, []).append(result)
    strata = {b.id: f"{b.stage}/{b.stratum}" for b in blocks}
    unc = bt.bootstrap(fit, bt.BootstrapData.from_blocks(by_block, strata=strata), samples=samples,
                       rng=np.random.default_rng(np.random.SeedSequence([int(seed) % 2**63, 0xB7])))
    tt = bt.transitivity_test(table, fit, sources=("tournament",))
    p_value = None if tt.p_value is None else float(tt.p_value)
    entities = sorted((e for e in fit.rating if e in unc.ci95), key=lambda e: (-fit.rating[e], e))
    return {
        "anchor": [{"group": g.group, "component": int(g.component), "anchor": g.anchor} for g in fit.groups],
        "entities": [{"name": e, "rating": float(fit.rating[e]), "ci95": [float(x) for x in unc.ci95[e]],
                      "p_best": float(unc.p_best_group.get(e, 0.0)), "games": int(fit.games.get(e, 0))}
                     for e in entities],
        "transitivity": {"p_value": p_value, "flag": p_value is not None and p_value < 0.01,
                         "worst_pairs": _jsonable(tt.worst_pairs)},
        "converged": bool(fit.converged),
    }
```

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_tournament_stats.py tests/unit/test_sp4_tournament_modes.py -q`
Expected: all pass in a few seconds (the coverage test is the slowest, under 1 s).

- [ ] **Step 6: Full fast suite + ruff**

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/arena/tournament/stats.py src/colosseum/arena/tournament/modes.py tests/game_helpers.py \
        tests/unit/test_sp4_tournament_stats.py tests/unit/test_sp4_tournament_modes.py
git commit -m "feat: tournament statistics and modes (paired stratified bootstrap, halving, select, confirmed)"
git push
```

---
### Task T5.4: `colosseum tournament` (CLI, several `--run`, the tournament dir, resume by whole blocks, snapshot links, the env check, journal writes, reports, `--pin`)

Spec block 3 («Исполнение», «Куда пишет», «Закрепление», the example command) and §3.2 («турнир: ... продолжение прерванного турнира пропускает только целые блоки; смена настроек под тем же `--name` — `ConfigError`; снимки турнира переживают вытеснение обучением; несовпадение `env` с запуском — `ConfigError`; несколько `--run` с алиасами; frozen и скриптовые игроки в роли тиммейтов»), §3.8 (SP3 checkpoints play as run snapshots and as frozen agents).

`arena.tournament.run` wires everything: config (`-c`, else the first run's `config.resolved.yaml`; the env section must equal every `--run`'s), play-scope validation, the tournament dir and its `tournament.yaml`, player resolution with snapshot hard links into `snapshots/`, the journal (`MatchLog` writer on the run's `matches.db` next to a possibly training run, one transaction per whole block), resume (blocks written whole are reused from the journal), `modes.run`, the reports, and `--pin` (only a **confirmed** `select` winner that is a run snapshot is pinned). The CLI groups each `--agent` / `--snapshots` with the `--run` before it.

**Files:**
- Create: `src/colosseum/arena/tournament/run.py`
- Modify: `src/colosseum/cli.py` (the `tournament` command)
- Modify: `src/colosseum/core/run_dir.py` (`tournaments_dir`, if T7.1 has not added it yet)
- Modify: `tests/game_helpers.py` (`write_fake_run(..., biases=...)`)
- Test: `tests/integration/test_sp4_tournament_cli.py`

**Interfaces:**
- Consumes: `players.resolve_players`, `PlayerRequest`, `RunSelection`, `player_to_dict` / `player_from_dict`, `load_run_config` (T5.1); `engine.play_blocks`, `TournamentError` (T5.1); `schedule.stage_seed` (T5.1); `modes.run`, `bt_section`, `TournamentSettings`, `DEFAULT_*` (T5.3); `MatchLog(path, run, *, min_free_disk_mb)`, `MatchLog.append`, `MatchLog.close`, `MatchLog.open_readonly(...).blocks(name)` / `.runs()` / `.close()`, `RunInfo`, `MATCH_LOG_FILE` (T1.2), `MatchLogReader.match_results(source=, tournament=)` (R4); `RunDir.open(root).run_id` (R1); `pin_snapshot` (T4.2); `link_or_copy` (T4.1); `validate_config(scope="play", players=...)` with `ValidationReport.fixed` (T5.1); `enabled_layouts`; Part 3's `render_tournament_html` (R12, optional until T7.2).
- Produces:
  - CLI `colosseum tournament [-c CFG] [-a NAME[=PATH] ...] [--run [ALIAS=]DIR [--agent A] [--snapshots all|last:N|top:N|every:K] ...] [--field a,b,...] [--mode gauntlet|round-robin|halving|select] [--layout L ...] [--blocks N] [--halving-blocks N] [--confirm-blocks N] [--finalists N] [--num-workers W] [--num-envs E] [--deterministic] [--seed S] [--name NAME] [--output DIR] [--pin] [--set k=v ...]`; exit codes 0 / 1 (config, player or env error; a worker died) / 2 (bad arguments) / 130 (SIGINT);
  - *`run.TournamentOptions`, *`run.run_tournament(options) -> TournamentOutcome`, *`run.TournamentDir(path)` with `yaml_path`, `snapshots`, `logs`, `report_paths`, `load()`, `write(data)`, `check(stored, request)`, `write_reports(report)`; *`run.BlockRunner`; *`run.diff_fields(a, b)`; *`run.TOURNAMENT_FILE`, *`run.TOURNAMENT_FORMAT`;
  - *`RunDir.tournaments_dir` (`runs/<name>/tournaments`);
  - tournament dir layout: `tournament.yaml`, `snapshots/<i>-<run name>/<agent>/ckpt_v<N>/{model.pt, meta.json}`, `logs/{tournament,tournament-worker-<i>}.log`, `report.json`, `report.txt`, `report.html` (once Part 3's renderer exists).

Proposed rulings (Contract notes D10–D14): `--output DIR` mirrors a run dir (`DIR/matches.db`, `DIR/tournaments/<name>/`); `--output` together with `--run` is a usage error; on resume the stored players are reused (not re-resolved); `--pin` pins only a confirmed winner, and only a run snapshot; `num_workers` / `num_envs` / `--pin` are not part of the settings check; layout weights come from the first `--run` agent's effective `matchmaking.layouts` (else the config's), equal when a chosen layout has no weight (INFO).

- [ ] **Step 1: Write the failing tests**

`tests/game_helpers.py`: extend `write_fake_run` with `biases: Mapping[int, Sequence[float]] | None = None` — for a version in `biases`, the snapshot's policy ignores the observation and plays by fixed logits:

```python
def biased_model(config, agent_id: str, logits, seed: int = 0):
    """A ``GameTestModel`` of trainable ``agent_id`` whose policy head has zero weights and bias ``logits``: it
    plays one fixed distribution whatever it sees (``[20, 0, 0]``: always action 0)."""
    from colosseum.core.registry import build_model

    _roles, role = agent_role_of(config, agent_id)
    torch.manual_seed(seed)
    model = build_model(config.get_agent_config(agent_id), role)
    head = model.inner.policy.heads["0"]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.as_tensor(list(logits), dtype=head.bias.dtype))
    return model
```
and in `write_fake_run` the loop becomes
```python
    for i, version in enumerate(versions):
        model = biased_model(config, agent, biases[version], seed=i) if biases and version in biases else None
        write_ckpt_dir(root / "checkpoints" / agent, config, agent, model=model, policy_version=version, seed=i)
```
(`biases` keyword after `agents`, before `**sections`; update the docstring.)

Create `tests/integration/test_sp4_tournament_cli.py`:

```python
"""``colosseum tournament`` (SP4 T5.4, spec block 3): select on a run with --pin, the journal and the reports,
resume by whole blocks, the settings check, snapshot links that survive eviction, the env check, several --run
with aliases, --output without a run, scripted teammates in score layouts, SP3 run dirs and checkpoints, usage
errors. Every invocation runs in-process (CliRunner) and spawns at most two workers."""
from __future__ import annotations

import json
import shutil

import pytest
from click.testing import CliRunner

import colosseum.arena.tournament.run as run_module
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from colosseum.arena.tournament.players import entity_hash
from colosseum.cli import main
from game_helpers import copy_sp3_run, scripted_agent, write_fake_run, write_test_config

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}
# TurnTakingGame pays the acting seat's action to the other seat: always 0 wins, always 1 gives everything away
STRONG, UNIFORM, WEAK = [20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-20.0, 20.0, 0.0]


def invoke(*args):
    return CliRunner().invoke(main, ["tournament", *map(str, args)])


def fake_run(tmp_path, name="run", versions=(10, 20, 30), game="turns", **kw):
    return write_fake_run(tmp_path / name, game, agents=AGENTS, versions=versions,
                          biases={10: WEAK, 20: UNIFORM, 30: STRONG} if game == "turns" else None, **kw)


def report(root, name) -> dict:
    return json.loads((root / "tournaments" / name / "report.json").read_text())


def journal(db, name):
    reader = MatchLog.open_readonly(db)
    try:
        return reader.blocks(name), reader.match_results(source="tournament", tournament=name)
    finally:
        reader.close()


def test_select_on_a_run_writes_the_journal_and_reports_and_pins_the_confirmed_winner(tmp_path):
    root, _cfg = fake_run(tmp_path)
    result = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                    "--confirm-blocks", 16, "--finalists", 2, "--num-workers", 2, "--num-envs", 2, "--seed", 3,
                    "--name", "sel", "--pin")
    assert result.exit_code == 0, result.output
    data = report(root, "sel")
    sel = data["selection"]["player"]
    assert sel["winner"] == "main@v30" and sel["confirmed"] is True
    assert json.loads((root / "checkpoints" / "main" / "ckpt_v30" / "meta.json").read_text())["pinned"] is True
    assert "Pinned main@v30" in result.output
    tdir = root / "tournaments" / "sel"
    for name in ("tournament.yaml", "report.json", "report.txt"):
        assert (tdir / name).is_file()
    assert (tdir / "logs" / "tournament.log").is_file()
    blocks, results = journal(root / MATCH_LOG_FILE, "sel")
    assert len(blocks) == data["games"]["blocks"] and len(results) == data["games"]["matches"]
    assert len({r.uid for r in results}) == len(results) and {r.source for r in results} == {"tournament"}
    assert all(r.uid.startswith("sel/") and r.env_steps is None for r in results)
    again = invoke("--run", root, "--mode", "select", "--field", "rnd,ones", "--halving-blocks", 2,
                   "--confirm-blocks", 16, "--finalists", 2, "--num-workers", 1, "--num-envs", 2, "--seed", 3,
                   "--name", "sel")
    assert again.exit_code == 0, again.output
    assert "0 played" in again.output                                   # every block came from the journal
    assert journal(root / MATCH_LOG_FILE, "sel")[0] == blocks


def test_an_interrupted_tournament_resumes_with_whole_blocks_only(tmp_path, monkeypatch):
    root, _cfg = fake_run(tmp_path)
    args = ["--run", root, "--snapshots", "last:2", "--field", "rnd", "--blocks", 3, "--num-envs", 2, "--name", "g"]
    original = run_module.BlockRunner._store
    stored = []

    def failing(self, block_id, results):
        if len(stored) == 2:
            raise RuntimeError("interrupted on purpose")
        stored.append(block_id)
        original(self, block_id, results)

    monkeypatch.setattr(run_module.BlockRunner, "_store", failing)
    first = invoke(*args)
    assert first.exit_code != 0 and isinstance(first.exception, RuntimeError)
    assert len(journal(root / MATCH_LOG_FILE, "g")[0]) == 2
    monkeypatch.setattr(run_module.BlockRunner, "_store", original)
    second = invoke(*args)
    assert second.exit_code == 0, second.output
    assert "4 played, 2 from the journal" in second.output              # 2 candidates x 1 field x 3 blocks = 6
    blocks, results = journal(root / MATCH_LOG_FILE, "g")
    assert len(blocks) == 6 and len(results) == 12
    changed = invoke(*args[:-2], "--blocks", 4, "--name", "g")
    assert changed.exit_code == 1 and "blocks" in changed.output and "tournament.yaml" in changed.output


def test_tournament_snapshots_survive_eviction_by_the_run(tmp_path):
    root, _cfg = fake_run(tmp_path)
    args = ["--run", root, "--field", "rnd", "--blocks", 1, "--num-envs", 2, "--name", "links"]
    assert invoke(*args).exit_code == 0
    linked = root / "tournaments" / "links" / "snapshots" / f"0-{root.name}" / "main" / "ckpt_v10" / "model.pt"
    assert linked.is_file() and linked.stat().st_nlink >= 2              # a hard link (same file system)
    shutil.rmtree(root / "checkpoints" / "main" / "ckpt_v10")             # the run evicts it
    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert linked.is_file() and "0 played" in again.output


def test_the_env_of_the_tournament_config_must_be_the_runs(tmp_path):
    root, _cfg = fake_run(tmp_path)
    other = write_test_config(tmp_path / "other.yaml", "turns", env={"kwargs": {"length": 8}}, agents=AGENTS)
    result = invoke("-c", other, "--run", root, "--field", "rnd", "--blocks", 1)
    assert result.exit_code == 1 and "env" in result.output and str(root) in result.output


def test_several_runs_with_aliases_and_snapshot_references_in_the_field(tmp_path):
    first, _ = fake_run(tmp_path, "a", versions=(10, 30))
    second, _ = fake_run(tmp_path, "b", versions=(5,))
    result = invoke("--run", first, "--snapshots", "last:1", "--run", f"old={second}", "--field", "rnd,main@v10",
                    "--blocks", 1, "--num-envs", 2, "--name", "multi")
    assert result.exit_code == 0, result.output
    data = report(first, "multi")
    assert [c["name"] for c in data["candidates"]] == ["main@v30", "old/main@v5"]
    assert [f["name"] for f in data["field"]] == ["rnd", "main@v10"]
    assert (first / MATCH_LOG_FILE).is_file() and not (second / MATCH_LOG_FILE).exists()


def test_output_dir_without_a_run_and_scripted_teammates_in_a_score_layout(tmp_path):
    root, cfg_ = fake_run(tmp_path, game="coop", versions=(1,))
    cfg = write_test_config(tmp_path / "coop.yaml", "coop", agents=AGENTS)
    ckpt = root / "checkpoints" / "main" / "ckpt_v1"
    out = tmp_path / "out"
    result = invoke("-c", cfg, "-a", f"prev={ckpt}", "--field", "ones", "--blocks", 2, "--num-envs", 2, "--output", out,
                    "--name", "coop")
    assert result.exit_code == 0, result.output
    data = json.loads((out / "tournaments" / "coop" / "report.json").read_text())
    (table,) = data["tables"]
    assert table["metric"] == "score" and table["columns"] == ["(self)", "ones"]
    assert table["rows"][0]["name"] == f"prev#{entity_hash(ckpt)}"
    assert (out / MATCH_LOG_FILE).is_file()


def test_sp3_run_dirs_play_as_runs_and_their_checkpoints_as_frozen_agents(tmp_path):
    sp3 = copy_sp3_run(tmp_path)
    agent = next(p.name for p in (sp3 / "checkpoints").iterdir() if p.is_dir())
    ckpt = sorted((sp3 / "checkpoints" / agent).glob("ckpt_v*"))[0]
    top = invoke("--run", sp3, "--snapshots", "top:1", "-a", f"old={ckpt}", "--field", "old", "--name", "top")
    assert top.exit_code == 1 and "match log" in top.output              # before this test writes a journal there
    result = invoke("--run", sp3, "--snapshots", "last:2", "-a", f"old={ckpt}", "--field", "old", "--blocks", 1,
                    "--num-envs", 2, "--name", "sp3")
    assert result.exit_code == 0, result.output
    data = report(sp3, "sp3")
    assert all(c["name"].startswith(f"{agent}@v") for c in data["candidates"]) and len(data["candidates"]) == 2
    assert data["field"][0]["name"].startswith("old#")


@pytest.mark.parametrize("args, message", [
    (["--agent", "main", "--run", "RUN"], "--agent"),
    (["--run", "RUN", "--field", "rnd", "--pin"], "--pin"),
    (["--run", "RUN", "--field", "rnd", "--output", "OUT"], "--output"),
    (["-a", "x=nowhere.pt"], "--run"),
    (["--run", "RUN", "--snapshots", "all", "--snapshots", "last:1"], "twice"),
])
def test_usage_errors_exit_with_2(tmp_path, args, message):
    root, _cfg = fake_run(tmp_path)
    args = [str(root) if a == "RUN" else str(tmp_path / "out") if a == "OUT" else a for a in args]
    result = invoke(*args)
    assert result.exit_code == 2 and message in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_tournament_cli.py -q`
Expected: failures — `No such command 'tournament'` (exit 2 everywhere) and `ModuleNotFoundError: colosseum.arena.tournament.run`.

- [ ] **Step 3: Write `src/colosseum/arena/tournament/run.py`**

```python
"""``colosseum tournament`` (SP4 spec block 3): the tournament dir, the settings check and resume, snapshot
links, the journal, the reports, ``--pin``.

Where things go:
- with ``--run runs/x`` (several: the first one): matches into ``runs/x/matches.db`` (the run's journal in WAL
  mode: the run may be training meanwhile), the tournament dir ``runs/x/tournaments/<name>/``;
- without ``--run``: ``--output DIR`` laid out like a run dir: ``DIR/matches.db``, ``DIR/tournaments/<name>/``.

The tournament dir holds ``tournament.yaml`` (format, name, id, the request, the resolved players, layouts,
layout weights and the seed domains), ``snapshots/`` (hard links, else copies, of every run snapshot the
tournament plays, made before the first match: a snapshot the training evicts meanwhile still plays),
``logs/`` (``tournament.log`` and the workers' logs) and the reports (``report.json``, ``report.txt``,
``report.html``).

Resume: a start with an existing ``--name`` compares its request (mode, sizes, seed, layouts, the players as
given on the command line, the ``--set`` overrides and the env section of the config; not ``--num-workers`` /
``--num-envs`` / ``--pin``) with ``tournament.yaml``: a difference is a ConfigError listing the fields. The stored
players are reused (not re-resolved: ``top:N`` of a training run moves), and every block already in the journal
(written whole, one transaction per block) is taken from it instead of being played again. Match uids are
deterministic (``schedule``), so the journal never holds a match twice.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from colosseum.arena.tournament import modes
from colosseum.arena.tournament.engine import play_blocks
from colosseum.arena.tournament.players import (
    PlayerRequest,
    RunSelection,
    TournamentPlayer,
    load_run_config,
    player_from_dict,
    player_to_dict,
    resolve_players,
    run_agent,
)
from colosseum.arena.tournament.schedule import STAGES, Block, stage_seed
from colosseum.coordinator.checkpoint_manager import CheckpointInfo, link_or_copy, pin_snapshot
from colosseum.core.config import ColosseumConfig, check_path_component, config_hash, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.types import MatchResult
from colosseum.utils.fs import write_text_atomic

logger = logging.getLogger(__name__)

TOURNAMENT_FILE = "tournament.yaml"
TOURNAMENT_FORMAT = 1
REPORT_JSON, REPORT_TXT, REPORT_HTML = "report.json", "report.txt", "report.html"


@dataclass(frozen=True)
class TournamentOptions:
    """The command line, parsed (``cli.tournament``)."""

    mode: str = "gauntlet"
    candidates: tuple[tuple[str, str | None], ...] = ()   # parsed -a specs
    runs: tuple[RunSelection, ...] = ()
    field: tuple[str, ...] | None = None
    layouts: tuple[str, ...] = ()
    blocks: int = modes.DEFAULT_BLOCKS
    halving_blocks: int = modes.DEFAULT_HALVING_BLOCKS
    confirm_blocks: int = modes.DEFAULT_CONFIRM_BLOCKS
    finalists: int = modes.DEFAULT_FINALISTS
    seed: int = 0
    deterministic: bool = False
    num_workers: int = 1
    num_envs: int = 8
    name: str | None = None
    output: str | None = None
    pin: bool = False
    config_path: str | None = None
    overrides: dict | None = None                          # parsed --set


@dataclass
class TournamentOutcome:
    report: modes.TournamentReport
    directory: Path
    played_blocks: int
    reused_blocks: int
    pinned: list[str]
    written: list[Path]


def diff_fields(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two JSON-like trees (lists compared whole)."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            out.extend(diff_fields(a.get(key), b.get(key), f"{prefix}{key}."))
        return out
    return [] if a == b else [prefix.rstrip(".") or "<root>"]


class TournamentDir:
    """``runs/x/tournaments/<name>/`` (module docstring)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def yaml_path(self) -> Path:
        return self.path / TOURNAMENT_FILE

    @property
    def snapshots(self) -> Path:
        return self.path / "snapshots"

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    def load(self) -> dict | None:
        if not self.yaml_path.is_file():
            return None
        data = yaml.safe_load(self.yaml_path.read_text()) or {}
        if data.get("format") != TOURNAMENT_FORMAT:
            raise ConfigError(f"{self.yaml_path}: format {data.get('format')!r}, expected {TOURNAMENT_FORMAT}")
        return data

    def write(self, data: Mapping[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.yaml_path, yaml.safe_dump(dict(data), sort_keys=False))

    def check(self, stored: Mapping[str, Any], request: Mapping[str, Any]) -> None:
        """ConfigError naming every request field that differs from ``tournament.yaml`` (spec block 3)."""
        fields = diff_fields(stored.get("request", {}), dict(request))
        if fields:
            raise ConfigError(f"tournament {stored.get('name')!r} exists ({self.yaml_path}) with other settings: "
                              f"{', '.join(fields)} differ; rerun with the same settings to continue it, or choose "
                              f"another --name")

    def link(self, run_index: int, selection: RunSelection, info: CheckpointInfo) -> Path:
        """The snapshot's hard-linked (else copied) ``model.pt`` + ``meta.json`` in ``snapshots/``."""
        dst = self.snapshots / f"{run_index}-{Path(selection.path).name}" / info.agent_id / info.checkpoint_id
        if (dst / "model.pt").is_file() and (dst / "meta.json").is_file():
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{info.checkpoint_id}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        for name in ("model.pt", "meta.json"):
            link_or_copy(info.path / name, tmp / name)
        os.replace(tmp, dst)
        return dst

    def write_reports(self, report: modes.TournamentReport) -> list[Path]:
        """``report.json``, ``report.txt`` and, once Part 3's renderer exists, ``report.html``."""
        import json

        data = report.to_dict()
        written = [write_text_atomic(self.path / REPORT_JSON, json.dumps(data, indent=2, sort_keys=False) + "\n"),
                   write_text_atomic(self.path / REPORT_TXT, report.text() + "\n")]
        try:   # T7.2 provides the renderer; until then the HTML report is skipped (Contract notes R12)
            from colosseum.arena.report.render import render_tournament_html
        except ImportError:
            logger.info("report.html is not written yet: colosseum.arena.report.render arrives with SP4 T7.2")
            return written
        written.append(write_text_atomic(self.path / REPORT_HTML, render_tournament_html(data)))
        return written


class BlockRunner:
    """The ``play`` of ``modes.run``: blocks already in the journal come from it; the others are played by
    ``engine.play_blocks`` and every finished block is appended to the journal in one transaction (``_store``)."""

    def __init__(self, *, config: ColosseumConfig, spec, players: Mapping[str, TournamentPlayer], log,
                 done: Mapping[str, list[MatchResult]], num_workers: int, num_envs: int, deterministic: bool,
                 log_dir: Path) -> None:
        self._config, self._spec, self._players, self._log = config, spec, players, log
        self._done = {k: list(v) for k, v in done.items()}
        self._num_workers, self._num_envs, self._deterministic, self._log_dir = (num_workers, num_envs,
                                                                                 deterministic, log_dir)
        self.played = 0
        self.reused = 0

    def _store(self, block_id: str, results: list[MatchResult]) -> None:
        self._log.append(results)            # one transaction: a block is in the journal whole or not at all
        self._done[block_id] = list(results)
        self.played += 1

    def __call__(self, blocks: Sequence[Block]) -> dict[str, list[MatchResult]]:
        out: dict[str, list[MatchResult]] = {}
        todo: list[Block] = []
        for block in blocks:
            got = self._done.get(block.id)
            if got is None:
                todo.append(block)
                continue
            if sorted(r.uid for r in got) != sorted(lu.uid for lu in block.lineups):
                raise ConfigError(f"journal block {block.id} has matches {sorted(r.uid for r in got)}, the schedule "
                                  f"{sorted(lu.uid for lu in block.lineups)}: the tournament's settings changed; "
                                  f"choose another --name")
            out[block.id] = got
            self.reused += 1
        if todo:
            play_blocks(self._config, self._spec, self._players, todo, num_workers=self._num_workers,
                        num_envs=self._num_envs, deterministic=self._deterministic, on_block=self._store,
                        log_dir=self._log_dir, source="tournament")
            out.update({b.id: self._done[b.id] for b in todo})
        return out


def _request(options: TournamentOptions, config: ColosseumConfig) -> dict[str, Any]:
    return {
        "mode": options.mode,
        "candidates": [f"{n}={p}" if p is not None else n for n, p in options.candidates],
        "runs": [{"path": str(Path(r.path).resolve()), "alias": r.alias, "agent": r.agent, "snapshots": r.snapshots}
                 for r in options.runs],
        "field": None if options.field is None else list(options.field),
        "layouts": list(options.layouts),
        "blocks": options.blocks, "halving_blocks": options.halving_blocks,
        "confirm_blocks": options.confirm_blocks, "finalists": options.finalists,
        "seed": options.seed, "deterministic": options.deterministic,
        "overrides": dict(sorted((options.overrides or {}).items())),
        "env": config.env.model_dump(mode="json", by_alias=True),
    }


def _config(options: TournamentOptions) -> ColosseumConfig:
    if options.config_path is not None:
        config = load_config(options.config_path, options.overrides or None)
    elif options.runs:
        config = load_config(Path(options.runs[0].path) / "config.resolved.yaml", options.overrides or None)
    else:
        raise ConfigError("give -c CONFIG or --run DIR (the run's config.resolved.yaml is used then)")
    own = config.env.model_dump(mode="json", by_alias=True)
    for sel in options.runs:
        theirs = load_run_config(Path(sel.path)).env.model_dump(mode="json", by_alias=True)
        if theirs != own:
            raise ConfigError(f"--run {sel.path}: its env section differs from the tournament config's "
                              f"({', '.join(diff_fields(theirs, own))}); snapshots play only in the game they were "
                              f"trained on: drop -c (the run's config is used) or fix the config's env")
    return config


def _layouts(options: TournamentOptions, config: ColosseumConfig, spec) -> tuple[list[str], dict[str, float]]:
    from colosseum.league.lineups import enabled_layouts

    if options.runs:
        first = Path(options.runs[0].path)
        run_config = load_run_config(first)
        matchmaking = run_config.get_agent_config(run_agent(first, run_config, options.runs[0].agent)).matchmaking
    else:
        matchmaking = config.matchmaking
    weights = enabled_layouts(spec, matchmaking)
    chosen = list(dict.fromkeys(options.layouts)) if options.layouts else [n for n in weights if n in spec.layouts]
    unknown = sorted(set(chosen) - set(spec.layouts))
    if unknown:
        raise ConfigError(f"--layout {unknown}: not layouts of the game {sorted(spec.layouts)}")
    if all(name in weights for name in chosen):
        return chosen, {name: float(weights[name]) for name in chosen}
    logger.info("layouts %s: some have no weight in matchmaking.layouts; weighting them equally", chosen)
    return chosen, dict.fromkeys(chosen, 1.0)


def _journal_run(options: TournamentOptions, config: ColosseumConfig, db: Path, tournament_id: str):
    from colosseum.arena.matchlog import MatchLog, RunInfo
    from colosseum.core.run_dir import RunDir

    if not options.runs:
        return RunInfo(run_id=tournament_id, name=f"tournament-{options.name}", parent_run_id=None,
                       game=config.env.env_class, config_hash=config_hash(config),
                       checkpoint_interval=config.checkpoint.interval)
    first = Path(options.runs[0].path)
    run_id = RunDir.open(first).run_id
    if db.is_file():
        reader = MatchLog.open_readonly(db)
        try:
            known = {r.run_id: r for r in reader.runs()}
        finally:
            reader.close()
        if run_id in known:
            return known[run_id]
    run_config = load_run_config(first)
    return RunInfo(run_id=run_id, name=first.name, parent_run_id=None, game=run_config.env.env_class,
                   config_hash=config_hash(run_config), checkpoint_interval=run_config.checkpoint.interval)


def _file_logging(path: Path) -> logging.Handler:
    """``tournament.log`` for the ``colosseum`` loggers of this command (removed by the caller)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] tournament %(name)s: %(message)s"))
    root = logging.getLogger("colosseum")
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler


def run_tournament(options: TournamentOptions) -> TournamentOutcome:
    """Run (or continue) one tournament (module docstring)."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
    from colosseum.core.registry import env_spec, validate_config

    config = _config(options)
    spec = env_spec(config)
    name = options.name or f"{options.mode}-{datetime.now():%Y%m%d-%H%M%S}"
    check_path_component(name, "tournament name")
    options = dataclasses.replace(options, name=name)
    if options.runs:
        root = Path(options.runs[0].path)
    elif options.output is not None:
        root = Path(options.output)
    else:
        raise ConfigError("--output DIR is required without --run")
    tdir = TournamentDir(root / "tournaments" / name)
    db = root / MATCH_LOG_FILE
    handler = _file_logging(tdir.logs / "tournament.log")
    try:
        request = _request(options, config)
        stored = tdir.load()
        if stored is not None:
            tdir.check(stored, request)
            players = {d["name"]: player_from_dict(d) for d in stored["players"]}
            candidates, field_names = list(stored["candidates"]), list(stored["field"])
            layouts, weights = list(stored["layouts"]), dict(stored["layout_weights"])
            missing = [p.name for p in players.values()
                       if p.source.kind == "model" and not Path(p.source.weights).is_file()]
            if missing:
                raise ConfigError(f"tournament {name!r}: the weights of {missing} are gone (snapshots/ of "
                                  f"{tdir.path} or the -a paths); they cannot be played again")
            logger.info("Continuing tournament %s (%s)", name, tdir.path)
        else:
            if tdir.path.exists() and any(p.name != "logs" for p in tdir.path.iterdir()):
                raise ConfigError(f"{tdir.path} exists without {TOURNAMENT_FILE}; remove it or choose another --name")
            fixed_names = [n for n in config.fixed_agent_ids()
                           if options.field is None or n in options.field or n in dict(options.candidates)]
            report_v = validate_config(config, scope="play", players=fixed_names)
            request_players = PlayerRequest(user=tuple(options.candidates), runs=tuple(options.runs),
                                            field=options.field)
            cand, fld = resolve_players(config, spec, request_players, fixed=report_v.fixed, link=tdir.link,
                                        rng=np.random.default_rng(options.seed),
                                        allow_empty_field=options.mode == "round-robin")
            players = {p.name: p for p in [*cand, *fld]}
            candidates, field_names = [p.name for p in cand], [p.name for p in fld]
            layouts, weights = _layouts(options, config, spec)
            tdir.write({
                "format": TOURNAMENT_FORMAT, "name": name, "id": uuid.uuid4().hex,
                "created": datetime.now().isoformat(timespec="seconds"),   # informational
                "request": request,
                "players": [player_to_dict(p) for p in players.values()],
                "candidates": candidates, "field": field_names,
                "layouts": layouts, "layout_weights": weights,
                "seeds": {"seed": options.seed,
                          "domains": {stage: [stage_seed(options.seed, stage, k) for k in range(3)]
                                      for stage in STAGES}},
            })
            stored = tdir.load()
        log = MatchLog(db, _journal_run(options, config, db, stored["id"]),
                       min_free_disk_mb=config.match_log.min_free_disk_mb)
        try:
            reader = MatchLog.open_readonly(db)
            try:
                whole = reader.blocks(name)
                done: dict[str, list[MatchResult]] = {}
                for result in reader.match_results(source="tournament", tournament=name):
                    if result.block in whole:
                        done.setdefault(result.block, []).append(result)
            finally:
                reader.close()
            runner = BlockRunner(config=config, spec=spec, players=players, log=log, done=done,
                                 num_workers=options.num_workers, num_envs=options.num_envs,
                                 deterministic=options.deterministic, log_dir=tdir.logs)
            settings = modes.TournamentSettings(
                name=name, layouts=layouts, layout_weights=weights, blocks=options.blocks,
                halving_blocks=options.halving_blocks, confirm_blocks=options.confirm_blocks,
                finalists=options.finalists, seed=options.seed, deterministic=options.deterministic)
            bt_fn = functools.partial(modes.bt_section, players=players, anchor=config.ratings.anchor,
                                      samples=config.ratings.bootstrap_samples, seed=options.seed)
            report = modes.run(options.mode, spec=spec, players=players, candidates=candidates, field=field_names,
                               settings=settings, play=runner, bt_fn=bt_fn)
        finally:
            log.close()
        written = tdir.write_reports(report)
        pinned = _pin(report, players) if options.pin else []
        logger.info("Tournament %s: %d blocks played, %d from the journal", name, runner.played, runner.reused)
        return TournamentOutcome(report, tdir.path, runner.played, runner.reused, pinned, written)
    finally:
        logging.getLogger("colosseum").removeHandler(handler)
        handler.close()


def _pin(report: modes.TournamentReport, players: Mapping[str, TournamentPlayer]) -> list[str]:
    """Pin every confirmed ``select`` winner that is a run snapshot (spec block 3); the others are reported."""
    pinned = []
    for group, sel in (report.to_dict().get("selection") or {}).items():
        winner = players[sel["winner"]]
        if not sel["confirmed"]:
            logger.warning("--pin: group %s: the winner %s is not confirmed; nothing pinned", group, winner.name)
            continue
        if winner.kind != "snapshot":
            logger.warning("--pin: group %s: the winner %s is not a run snapshot; nothing pinned", group, winner.name)
            continue
        pin_snapshot(winner.run, winner.run_agent, winner.version)
        pinned.append(winner.name)
    return pinned
```

Notes for the implementer:
- `MatchLog(db, run, min_free_disk_mb=...)` on the run's live journal: Part 1 guarantees a second writer works (WAL, `busy_timeout`) and that registering an existing `run_id` is idempotent (R4). Tournament matches are written even under the disk guard (spec block 1, item 4; Part 1).
- `validate_config` runs only when the tournament is new (a resumed one reuses its stored, already validated players).

- [ ] **Step 4: The CLI command**

`src/colosseum/cli.py` (after `record_cmd`):

```python
class _TournamentCommand(click.Command):
    """Keeps the raw arguments: every ``--agent`` / ``--snapshots`` belongs to the ``--run`` before it."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta["tournament_args"] = list(args)
        return super().parse_args(ctx, args)


def _group_runs(args: list[str]) -> list[dict]:
    """``[{"run", "agent", "snapshots"}]`` in command-line order (``--opt value`` and ``--opt=value``)."""
    runs: list[dict] = []
    i = 0
    while i < len(args):
        token, key, value = args[i], None, None
        for opt in ("--run", "--agent", "--snapshots"):
            if token == opt:
                key, value = opt, args[i + 1] if i + 1 < len(args) else None
                i += 1
                break
            if token.startswith(opt + "="):
                key, value = opt, token[len(opt) + 1:]
                break
        i += 1
        if key is None:
            continue
        if key == "--run":
            runs.append({"run": value, "agent": None, "snapshots": None})
            continue
        if not runs:
            raise click.UsageError(f"{key} belongs to a --run: write it after the --run it describes")
        slot = key.removeprefix("--")
        if runs[-1][slot] is not None:
            raise click.UsageError(f"--run {runs[-1]['run']}: {key} given twice")
        runs[-1][slot] = value
    return runs


def _run_selection(item: dict):
    from pathlib import Path

    from colosseum.arena.tournament.players import RunSelection

    text = item["run"]
    alias, sep, path = text.partition("=")
    if not sep or not alias or "/" in alias or not path:
        alias, path = None, text
    p = Path(path)
    if not p.is_dir():
        raise click.BadParameter(f"{path}: not a directory", param_hint="'--run'")
    return RunSelection(path=p, alias=alias, agent=item["agent"], snapshots=item["snapshots"] or "all")


@main.command("tournament", cls=_TournamentCommand)
@click.option("--config", "-c", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Config YAML (its env and agents); default: the first --run's config.resolved.yaml")
@click.option("-a", "--candidate", "candidates", multiple=True,
              help="Candidate: name=path (a checkpoint dir or .pt of any architecture) or a scripted / frozen agent "
                   "of the config. Repeatable.")
@click.option("--run", "runs", multiple=True,
              help="[alias=]runs/x: candidates are its stored snapshots (repeatable; --agent / --snapshots after a "
                   "--run apply to it). Matches go to the first run's matches.db, reports to its tournaments/<name>/.")
@click.option("--agent", "run_agents", multiple=True, help="The --run's agent (default: its only trainable agent)")
@click.option("--snapshots", "run_snapshots", multiple=True,
              help="The --run's snapshots: all (default), last:N, top:N (by the run journal's rating lower bound), "
                   "every:K")
@click.option("--field", default=None,
              help="Comma-separated opponents: agents of the config, -a names or snapshots [alias/]agent@vN. "
                   "Default: every scripted / frozen agent of the config.")
@click.option("--mode", default="gauntlet", show_default=True,
              type=click.Choice(["gauntlet", "round-robin", "halving", "select"]))
@click.option("--layout", "layouts", multiple=True, help="Layout (repeatable); default: the enabled layouts")
@click.option("--blocks", default=None, type=click.IntRange(min=1),
              help="gauntlet / round-robin: blocks per composition (default 50, provisional)")
@click.option("--halving-blocks", default=None, type=click.IntRange(min=1),
              help="halving / select: blocks per composition of round 0 (default 8, provisional)")
@click.option("--confirm-blocks", default=None, type=click.IntRange(min=1),
              help="select: blocks per composition of the confirmation stage (default 100, provisional)")
@click.option("--finalists", default=None, type=click.IntRange(min=1), help="halving / select (default 3)")
@click.option("--num-workers", default=1, type=click.IntRange(min=1), show_default=True, help="Worker processes")
@click.option("--num-envs", default=8, type=click.IntRange(min=1), show_default=True, help="Envs per worker")
@click.option("--deterministic", is_flag=True, default=False, help="Greedy policies (default: stochastic)")
@click.option("--seed", default=0, type=int, show_default=True, help="Seed of the block seeds (domains A, B)")
@click.option("--name", default=None, help="Tournament name (default <mode>-<YYYYmmdd-HHMMSS>); the same name "
                                           "continues an interrupted tournament")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False),
              help="Without --run: the directory for matches.db and tournaments/<name>/")
@click.option("--pin", is_flag=True, default=False, help="select with --run: pin the confirmed winner snapshot")
@click.option("--set", "overrides", multiple=True, help="Override config values." + _SET_HELP_YAML)
def tournament_cmd(config: str | None, candidates: tuple[str, ...], runs: tuple[str, ...], run_agents: tuple[str, ...],
                   run_snapshots: tuple[str, ...], field: str | None, mode: str, layouts: tuple[str, ...],
                   blocks: int | None, halving_blocks: int | None, confirm_blocks: int | None, finalists: int | None,
                   num_workers: int, num_envs: int, deterministic: bool, seed: int, name: str | None,
                   output: str | None, pin: bool, overrides: tuple[str, ...]) -> None:
    """Compare candidates (run snapshots, checkpoints of any architecture, scripted / frozen agents) on common
    seeds with seat rotations, and select the best (--mode select: screening, then confirmation on fresh seeds).

    The config is checked in the "play" scope (like eval). Exit code: 0 done, 1 config, player or env error or a
    dead worker, 2 bad arguments, 130 SIGINT (finished blocks are kept: rerun with the same --name to continue).
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    grouped = _group_runs(click.get_current_context().meta.get("tournament_args", []))
    if len(grouped) != len(runs):
        raise click.UsageError("could not match --agent / --snapshots to their --run")
    if pin and (not runs or mode != "select"):
        raise click.UsageError("--pin needs --mode select and --run (it pins the winner in the run's storage)")
    if output is not None and runs:
        raise click.UsageError("--output is for tournaments without --run (with --run the reports go to "
                               "runs/x/tournaments/<name>/)")
    if not runs and config is None:
        raise click.UsageError("give -c CONFIG or --run DIR")
    with _config_errors():
        from colosseum.arena.tournament import modes
        from colosseum.arena.tournament.engine import TournamentError
        from colosseum.arena.tournament.run import TournamentOptions, run_tournament
        from colosseum.core.errors import ConfigError
        from colosseum.eval import parse_agent_spec

        try:
            parsed = tuple(parse_agent_spec(text) for text in candidates)
        except ConfigError as e:
            raise click.BadParameter(str(e), param_hint="'-a'") from e
        options = TournamentOptions(
            mode=mode, candidates=parsed, runs=tuple(_run_selection(item) for item in grouped),
            field=None if field is None else tuple(n.strip() for n in field.split(",") if n.strip()),
            layouts=tuple(layouts),
            blocks=blocks or modes.DEFAULT_BLOCKS, halving_blocks=halving_blocks or modes.DEFAULT_HALVING_BLOCKS,
            confirm_blocks=confirm_blocks or modes.DEFAULT_CONFIRM_BLOCKS,
            finalists=finalists or modes.DEFAULT_FINALISTS, seed=seed, deterministic=deterministic,
            num_workers=num_workers, num_envs=num_envs, name=name, output=output, pin=pin, config_path=config,
            overrides=_parse_overrides(overrides))
        try:
            outcome = run_tournament(options)
        except TournamentError as e:
            click.echo(f"Tournament error: {e}", err=True)
            sys.exit(1)
    click.echo("\n" + outcome.report.text())
    click.echo(f"Blocks: {outcome.played_blocks} played, {outcome.reused_blocks} from the journal")
    for path in outcome.written:
        click.echo(f"Written: {path}")
    for winner in outcome.pinned:
        click.echo(f"Pinned {winner}")
```

`src/colosseum/core/run_dir.py`: add (if T7.1 did not) the property `tournaments_dir -> self.root / "tournaments"` and the module docstring line `tournaments/<name>/  colosseum tournament dirs (SP4 T5.4)`.

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp4_tournament_cli.py -q`
Expected: all pass (about a minute: every invocation spawns its workers).

- [ ] **Step 6: Full fast suite + ruff**

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/arena/tournament/run.py src/colosseum/cli.py src/colosseum/core/run_dir.py tests/game_helpers.py \
        tests/integration/test_sp4_tournament_cli.py
git commit -m "feat: colosseum tournament (tournament dir, resume by whole blocks, snapshot links, journal, --pin)"
git push
```

---
### Task T6.1: Pulse (`evaluation` section, the `pulse` child, snapshot requests with a bounded backlog, pulse matches in the journal)

Spec block 6 and §3.2 («оценка во время обучения: процесс получает каждый новый снимок, его матчи попадают в журнал с источником `pulse`, процесс живёт и умирает вместе с запуском»; §6: «`pulse` получает каждый новый снимок; его матчи в журнале с `source = pulse` и одинаковыми seed для разных снимков; переполнение очереди снимков пропускает старые; процесс умирает с запуском, а его падение останавливает запуск»).

The launcher starts one `pulse` child (worker lifecycle: `run_child` → `logs/pulse.log`, SIGINT ignored, dies with the parent; a crash stops the run with a pointer to its log). After every snapshot save the main loop offers the snapshot to a `PulseFeeder` (numpy state dict, as for workers): every `evaluation.every`-th snapshot of every evaluated agent, and the final one always. The feeder keeps at most `max_backlog` waiting requests (the oldest is dropped: INFO and the `pulse_skipped_snapshots` counter of the `system` record) and hands the oldest one to the pulse process whenever its one-slot request queue is free. The pulse plays the gauntlet blocks of spec block 3 on the seed domain `P` — the same seeds for every snapshot — and sends each snapshot's results in one message (blocking put that gives up only at the stop). The main loop submits them to the journal writer only (live BT ratings see them; SP2 online ELO, win rates, PFSP and the matchmaker stay training-only), and `runs/<name>/pulse_state.json` records which snapshots were offered, evaluated, skipped and left unevaluated (the final snapshot usually is: the run stops first; spec block 6 «Финальный снимок оценивается, только если запуск дал на это время»).

**Files:**
- Create: `src/colosseum/arena/pulse.py`
- Modify: `src/colosseum/core/config.py` (`EvaluationConfig`; `ColosseumConfig.evaluation`; `_check_sp4_dependencies`)
- Modify: `src/colosseum/core/validation.py` (`_check_evaluation`: field, agents, layouts, role coverage, printout)
- Modify: `src/colosseum/launcher.py` (feeder, queues, the `pulse` child, offers after saves, the pump in the monitor loop and the shutdown poll, `pulse_state.json`, system values)
- Modify: `src/colosseum/core/run_dir.py` (`pulse_state_path`)
- Modify: `scripts/bench_throughput.py::_make_config` (the `evaluation` section, explicit and off)
- Modify: `tests/game_helpers.py` (`PulseFailBot`)
- Test: `tests/unit/test_sp4_pulse.py`, `tests/integration/test_sp4_pulse_runs.py`

**Interfaces:**
- Consumes: `schedule.gauntlet_blocks` / `score_blocks` (T5.2, stage `P`); `engine.make_tasks`, `LazyPlayerPool`-free pool (here: `_PulsePool`), `players.config_player`, `TournamentPlayer`, `PlayerSource` (T5.1); `eval.play_lineups(pool=, source="pulse")` (T5.1); `FixedPlayers`, `FrozenSpec`, `build_frozen_model`, `make_bot`; `MatchLogWriter.submit` (T1.3) via `Launcher._match_log_writer` (R2); `RunDir.run_id` (R1); `MetricsHub.set_system_value` (R10); `utils.process.run_child`, `start_process`; `core.ipc.release_command_queues`; `league.lineups.playable_layouts`.
- Produces (contract T6.1, additions marked *):
  - `EvaluationConfig(enabled=False, agents=None, field=None, layouts=None, every=1, blocks=25, num_envs=8, deterministic=False, max_backlog=4)`, `ColosseumConfig.evaluation`;
  - `pulse.PulseRequest(agent_id, checkpoint_id, version, env_steps, state_dict, final=False*)`;
  - `pulse.pulse_main(config, *, requests, results, stop_event, field_specs, field*, agents*, run_id*)`; *`pulse.pulse_target(*, log_dir, **kwargs)`;
  - *`pulse.PulseAgent(roles, networks, layouts)`, *`pulse.PulsePlan(agents, field)`, *`pulse.pulse_plan(config, spec, player_roles) -> PulsePlan`, *`pulse.describe_pulse(plan, config, spec, player_roles) -> str`, *`pulse.pulse_blocks(...)`, *`pulse.PulseResults`, *`pulse.PulseFeeder`, *`PULSE_STATE_FILE = "pulse_state.json"`;
  - *`RunDir.pulse_state_path`;
  - `system` record values (through R10): `pulse_skipped_snapshots`, `pulse_evaluated_snapshots`, `pulse_pending_snapshots`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/game_helpers.py` (section "SP4 (T6.1): pulse"; `from colosseum.utils.logging import ENV_PROCESS_NAME` at the top):

```python
class PulseFailBot(ScriptedBot):
    """Random legal play, except in the process named ``pulse`` (``COLOSSEUM_PROCESS_NAME``), where ``act`` raises:
    ``validate`` and the rollout workers play it normally, the pulse process crashes on it."""

    def reset(self, *, role, seat, layout, rng) -> None:
        self._role, self._rng = self.game_spec.roles[role], rng

    def act(self, obs, mask, info):
        if os.environ.get(ENV_PROCESS_NAME) == "pulse":
            raise RuntimeError("pulse failure on purpose")
        return random_legal_action(self._role, mask, self._rng)
```

Create `tests/unit/test_sp4_pulse.py`:

```python
"""Pulse (SP4 T6.1, spec block 6): the evaluation section and its checks, the feeder's choice and backlog, and the
pulse process on a queue pair (one spawn child)."""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue

import numpy as np
import pytest
import yaml

from colosseum.arena.pulse import PulseFeeder, PulseRequest, PulseResults, pulse_plan, pulse_target
from colosseum.core.config import load_config
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, env_spec, validate_config
from colosseum.players.registry import load_fixed_players, resolve_player_roles
from game_helpers import agent_role_of, make_test_config, scripted_agent

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}


def _cfg(**evaluation):
    return make_test_config("turns", agents=AGENTS, evaluation={"enabled": True, **evaluation})


def _write(tmp_path, **sections) -> str:
    data = {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, "agents": AGENTS, **sections}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(path)


def test_the_evaluation_section_defaults_and_config_checks(tmp_path):
    ev = load_config(_write(tmp_path)).evaluation
    assert (ev.enabled, ev.agents, ev.field, ev.layouts, ev.every, ev.blocks, ev.num_envs, ev.deterministic,
            ev.max_backlog) == (False, None, None, None, 1, 25, 8, False, 4)
    with pytest.raises(ConfigError, match="match_log"):
        load_config(_write(tmp_path, evaluation={"enabled": True}, match_log={"enabled": False}))
    with pytest.raises(ConfigError, match="field"):
        load_config(_write(tmp_path, evaluation={"enabled": True, "field": []}))
    for bad in ({"every": 0}, {"blocks": 0}, {"num_envs": 0}, {"max_backlog": 0}):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, evaluation=bad))


def test_validate_checks_the_field_agents_layouts_and_prints_the_plan():
    lines = validate_config(_cfg(blocks=5)).lines
    (line,) = [x for x in lines if x.startswith("pulse:")]
    assert "agents ['main']" in line and "field ['rnd', 'ones']" in line and "20 matches per snapshot" in line
    for evaluation, message in (({"field": ["nobody"]}, "nobody"), ({"field": ["main"]}, "scripted or frozen"),
                                ({"agents": ["rnd"]}, "trainable"), ({"layouts": ["9p"]}, "9p")):
        with pytest.raises(ConfigError, match=message):
            validate_config(_cfg(**evaluation))
    assert not [x for x in validate_config(make_test_config("turns", agents=AGENTS)).lines if x.startswith("pulse:")]


def test_role_coverage_is_checked_per_agent_and_layout():
    cfg = make_test_config("asymmetric", agents={"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]},
                                                 "preybot": {**scripted_agent(), "roles": ["prey"]}},
                           evaluation={"enabled": True, "agents": ["prey"]})
    with pytest.raises(ConfigError, match="prey"):
        pulse_plan(cfg, env_spec(cfg), resolve_player_roles(cfg, env_spec(cfg)))


def _request(version, agent="main", final=False, env_steps=None):
    return PulseRequest(agent, f"ckpt_v{version}", version, env_steps if env_steps is not None else 10 * version,
                        {"w": np.zeros(2, np.float32)}, final=final)


def test_the_feeder_offers_every_kth_snapshot_and_the_final_one_and_drops_the_oldest(caplog):
    feeder = PulseFeeder(["main"], every=2, max_backlog=2)
    for version in range(1, 8):
        feeder.offer("main", f"ckpt_v{version}", version, 10 * version, {"w": np.zeros(1)}, final=False)
    feeder.offer("other", "ckpt_v1", 1, 10, {}, final=False)                 # not evaluated
    with caplog.at_level(logging.INFO, logger="colosseum.arena.pulse"):
        feeder.offer("main", "ckpt_v8", 8, 80, {}, final=True)                # 8th: even, and final
        feeder.offer("main", "ckpt_v9", 9, 90, {}, final=True)                # odd, but final
    q = queue.Queue(maxsize=1)
    assert feeder.pump(q) == 1 and q.get_nowait().version in (6, 8)
    state = feeder.state()
    assert state["offered"]["main"] == [2, 4, 6, 8, 9]
    assert state["skipped"]["main"] == [2, 4, 6] or state["skipped"]["main"] == [2, 4]
    assert feeder.skipped == len(state["skipped"]["main"])
    assert any("skipped" in r.getMessage() for r in caplog.records)
    assert "other" not in state["offered"]


def test_the_feeder_bookkeeping_adds_up():
    feeder = PulseFeeder(["main"], every=1, max_backlog=1)
    q = queue.Queue(maxsize=1)
    for version in (1, 2, 3):
        feeder.offer("main", f"ckpt_v{version}", version, version, {}, final=version == 3)
        feeder.pump(q)
    taken = q.get_nowait()
    feeder.done(PulseResults(taken.agent_id, taken.checkpoint_id, taken.version, [], True))
    feeder.pump(q)
    state = feeder.state()
    offered, evaluated = set(state["offered"]["main"]), set(state["evaluated"]["main"])
    rest = set(state["skipped"]["main"]) | set(state["unevaluated"]["main"])
    assert evaluated == {1} and offered == evaluated | rest and not evaluated & rest
    assert state["unevaluated"]["main"] == [3] and feeder.unevaluated_final() == [("main", "ckpt_v3")]


def test_the_pulse_process_plays_every_snapshot_on_the_same_seeds(tmp_path):
    cfg = _cfg(blocks=3, num_envs=2)
    spec = env_spec(cfg)
    plan = pulse_plan(cfg, spec, resolve_player_roles(cfg, spec))
    fixed = load_fixed_players(cfg, spec, plan.field)
    _roles, role = agent_role_of(cfg, "main")
    ctx = mp.get_context("spawn")
    requests, results, stop = ctx.Queue(maxsize=1), ctx.Queue(), ctx.Event()
    proc = ctx.Process(target=pulse_target, daemon=True, kwargs=dict(
        log_dir=str(tmp_path / "logs"), config=cfg, requests=requests, results=results, stop_event=stop,
        field_specs=fixed, field=plan.field, agents=plan.agents, run_id="0123456789abcdef"))
    proc.start()
    try:
        got = []
        for version in (5, 10):
            state = {k: v.detach().numpy().copy() for k, v in
                     build_model(cfg.get_agent_config("main"), role).state_dict().items()}
            requests.put(PulseRequest("main", f"ckpt_v{version}", version, 100 * version, state))
            got.append(results.get(timeout=60))
        requests.put(None)
        proc.join(30)
        assert proc.exitcode == 0
    finally:
        stop.set()
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        for q in (requests, results):
            q.cancel_join_thread()
            q.close()
    assert [m.version for m in got] == [5, 10] and all(m.complete for m in got)
    seeds = [sorted({r.seed for r in m.results}) for m in got]
    assert seeds[0] == seeds[1] and len(seeds[0]) == 3                     # 3 blocks: the same seeds for both
    for m in got:
        assert {r.source for r in m.results} == {"pulse"} and {r.env_steps for r in m.results} == {100 * m.version}
        assert all(r.uid.startswith(f"pulse-0123456789ab/P/2p/main@v{m.version}/") for r in m.results)
        assert len(m.results) == 2 * 3 * 2                                 # 2 field members x 3 blocks x 2 sides
    assert (tmp_path / "logs" / "pulse.log").is_file()
```

Create `tests/integration/test_sp4_pulse_runs.py`:

```python
"""Pulse in real runs (SP4 T6.1): snapshots evaluated into the journal on the same seeds; a pulse crash stops the
run with a pointer to its log; the pulse dies with the run. One rollout worker + pulse = two worker processes."""
from __future__ import annotations

import json
import os
import re
import signal

from cli_runner import read_records, run_train, training_process, wait_for
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from game_helpers import scripted_agent, write_test_config

AGENTS = {"main": {}, "rnd": scripted_agent()}
PULSE = {"evaluation.enabled": "true", "evaluation.every": "1", "evaluation.blocks": "2", "evaluation.num_envs": "2",
         "evaluation.max_backlog": "2", "training.total_timesteps": "4000", "checkpoint.interval": "10"}


def test_pulse_evaluates_snapshots_into_the_journal_on_the_same_seeds(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns", agents=AGENTS)
    run = run_train(cfg, tmp_path, name="pulse", overrides=PULSE)
    assert run.returncode == 0, run.stderr[-3000:]
    state = json.loads((run.root / "pulse_state.json").read_text())
    offered, evaluated = state["offered"]["main"], state["evaluated"]["main"]
    rest = set(state["skipped"]["main"]) | set(state["unevaluated"]["main"])
    assert evaluated and set(offered) == set(evaluated) | rest
    reader = MatchLog.open_readonly(run.root / MATCH_LOG_FILE)
    try:
        results = reader.match_results(source="pulse")
    finally:
        reader.close()
    by_version: dict[int, set] = {}
    for r in results:
        version = next(s.version_start for s in r.seats if s.agent_id == "main")
        by_version.setdefault(version, set()).add(r.seed)
        meta = run.root / "checkpoints" / "main" / f"ckpt_v{version}" / "meta.json"
        if meta.is_file():
            assert r.env_steps == json.loads(meta.read_text())["env_steps"]
    assert set(by_version) == set(evaluated)
    assert len({frozenset(seeds) for seeds in by_version.values()}) == 1    # identical seeds for every snapshot
    assert (run.root / "logs" / "pulse.log").is_file()
    system = read_records(run.root, "system")
    assert system and "pulse_skipped_snapshots" in system[-1]


def test_a_pulse_crash_stops_the_run_with_a_pointer_to_its_log(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns",
                            agents={"main": {}, "bad": scripted_agent("game_helpers.PulseFailBot")})
    run = run_train(cfg, tmp_path, name="crash", overrides={**PULSE, "training.total_timesteps": "200000"})
    assert run.returncode == 1
    log = run.log("main")
    assert "pulse died" in log and "pulse.log" in log
    assert "pulse failure on purpose" in run.log("pulse")


def test_the_pulse_dies_with_the_run(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns", agents=AGENTS)
    with training_process(cfg, tmp_path, name="orphan", overrides={**PULSE, "training.total_timesteps": "10000000"}) \
            as (proc, root):
        log = root / "logs" / "pulse.log"
        assert wait_for(lambda: log.is_file() and "started (pid" in log.read_text(), 120)
        pid = int(re.search(r"started \(pid (\d+)\)", log.read_text()).group(1))
        os.kill(proc.pid, signal.SIGKILL)                                   # the main process only

        def gone() -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            return False

        assert wait_for(gone, 10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_pulse.py tests/integration/test_sp4_pulse_runs.py -q`
Expected: `ModuleNotFoundError: No module named 'colosseum.arena.pulse'`.

- [ ] **Step 3: `EvaluationConfig`**

`src/colosseum/core/config.py` (after `CheckpointConfig`):

```python
class EvaluationConfig(StrictModel):
    """Pulse: every new snapshot against a fixed field during training (SP4 spec block 6). Global only."""

    enabled: bool = Field(default=False, description="Start the pulse process (needs match_log.enabled).")
    agents: list[str] | None = Field(default=None, description="Trainable agents to evaluate; null = all.")
    field: list[str] | None = Field(
        default=None, description="Scripted / frozen agents of the config to play against; null = all of them.")
    layouts: list[str] | None = Field(
        default=None, description="Layouts; null = every enabled layout with a seat of the agent's roles.")
    every: int = Field(default=1, ge=1, description="Evaluate every k-th saved snapshot (the final one always).")
    blocks: int = Field(default=25, ge=1, description="Blocks per field member (or composition); 1v1: 2 matches each.")
    num_envs: int = Field(default=8, ge=1, description="Envs of the pulse process.")
    deterministic: bool = Field(default=False, description="Greedy snapshot policies.")
    max_backlog: int = Field(default=4, ge=1, description="Snapshots waiting for evaluation; the oldest are skipped.")
```

`ColosseumConfig`: `evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)` (after `checkpoint`), and extend `_check_sp4_dependencies`:

```python
        ev = self.evaluation
        if ev.enabled and not self.match_log.enabled:
            raise ValueError("evaluation.enabled: pulse matches go to the match log; set match_log.enabled: true or "
                             "evaluation.enabled: false")
        if ev.enabled and ev.field is not None and not ev.field:
            raise ValueError("evaluation.field is empty: name scripted / frozen agents of the config, or null for all")
```

`src/colosseum/core/run_dir.py`: `pulse_state_path -> self.root / "pulse_state.json"` (and the docstring line `pulse_state.json  pulse bookkeeping (SP4 T6.1)`).

`scripts/bench_throughput.py::_make_config`: add
```python
        evaluation={"enabled": False, "agents": None, "field": None, "layouts": None, "every": 1, "blocks": 25,
                    "num_envs": 8, "deterministic": False, "max_backlog": 4},   # criterion 3.6: pulse off
```

- [ ] **Step 4: Write `src/colosseum/arena/pulse.py`**

```python
"""Pulse: evaluating new snapshots against a fixed field during training (SP4 spec block 6).

Main process (``PulseFeeder``): after every snapshot save the launcher offers the snapshot (numpy state dict, as
for workers). Agents of ``evaluation.agents`` (null: every trainable agent); every ``evaluation.every``-th saved
snapshot of each, and the final one always. Requests wait in a backlog of ``max_backlog``; when it is full the
oldest waiting request is dropped (INFO, ``skipped``: newer snapshots matter more). ``pump`` hands the oldest
waiting request to the one-slot request queue whenever the pulse process has taken the previous one; ``done``
records an evaluated snapshot. ``state()`` (``runs/<name>/pulse_state.json``) lists per agent the versions
offered, evaluated, skipped and unevaluated (waiting or in flight when the run stopped).

Child process (``pulse_target`` -> ``pulse_main``; started like a worker: ``run_child`` writes ``logs/pulse.log``,
SIGINT is ignored, it dies with the parent, a crash stops the run): one request at a time, the snapshot's model
built with the agent's networks and roles, the gauntlet blocks of spec block 3 (``schedule.gauntlet_blocks`` /
``score_blocks``, stage ``P``: the same seeds for every snapshot; seed ``training.seed`` or 0) against
``evaluation.field`` in ``evaluation.layouts`` (null: the enabled layouts with a seat of the agent's roles),
``evaluation.blocks`` blocks per composition, ``num_envs`` envs, greedy with ``deterministic``. Its seats never
collect. Every result has ``source = pulse`` and the snapshot's ``env_steps``. One snapshot's results go to the
main process in one ``PulseResults`` message with a blocking put that gives up only at the stop. Block ids:
``pulse-<run id[:12]>/P/<layout>/<agent>@v<N>/<composition>/<k>``.

The main process gives pulse results to the journal writer only (the live BT ratings read the journal's table);
SP2/SP3 online ELO, win rates, PFSP and the matchmaker stay training-only.
"""

from __future__ import annotations

import functools
import logging
import queue
from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from colosseum.arena.tournament.engine import make_tasks
from colosseum.arena.tournament.players import PlayerSource, TournamentPlayer, config_player
from colosseum.arena.tournament.schedule import Block, gauntlet_blocks, score_blocks
from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError
from colosseum.core.types import FIXED_NETWORK_ID, MatchResult
from colosseum.envs.game import GameSpec
from colosseum.players.registry import FixedPlayers, FrozenSpec, build_frozen_model, make_bot
from colosseum.utils.process import run_child
from colosseum.worker.match_runner import ScriptedPlayer

logger = logging.getLogger(__name__)

PULSE_STATE_FILE = "pulse_state.json"


@dataclass
class PulseRequest:
    """Main -> pulse (numpy only)."""

    agent_id: str
    checkpoint_id: str
    version: int
    env_steps: int
    state_dict: dict[str, np.ndarray]
    final: bool = False


@dataclass
class PulseResults:
    """Pulse -> main: every match of one snapshot (``complete``: all its blocks were played)."""

    agent_id: str
    checkpoint_id: str
    version: int
    results: list[MatchResult]
    complete: bool


@dataclass(frozen=True)
class PulseAgent:
    roles: tuple[str, ...]
    networks: dict[str, Any]
    layouts: tuple[str, ...]


@dataclass(frozen=True)
class PulsePlan:
    agents: dict[str, PulseAgent]
    field: list[str]


@dataclass(frozen=True)
class _Seat:     # what a schedule needs of a player (validation only)
    agent_id: str
    network_id: str
    roles: tuple[str, ...]


def pulse_blocks(spec: GameSpec, layouts: Sequence[str], candidate: str, field: Sequence[str],
                 players: Mapping[str, Any], *, blocks: int, seed: int, prefix: str) -> list[Block]:
    """The pulse gauntlet of one snapshot: stage P blocks of every layout (spec blocks 3 and 6)."""
    out: list[Block] = []
    for layout in layouts:
        build = score_blocks if spec.num_teams(layout) == 1 else gauntlet_blocks
        out.extend(build(spec, layout, [candidate], list(field), players, blocks=blocks, seed=seed, stage="P",
                         prefix=prefix))
    return out


def pulse_plan(config: ColosseumConfig, spec: GameSpec, player_roles: Mapping[str, Sequence[str]]) -> PulsePlan:
    """The evaluated agents (roles, networks, layouts) and the field, checked (spec block 6: the field names exist
    and are scripted / frozen agents of the config, the agents are trainable, the roles cover the layouts).
    ConfigError otherwise."""
    from colosseum.league.lineups import playable_layouts

    ev = config.evaluation
    trainable = config.get_trainable_agent_ids()
    agents = list(ev.agents) if ev.agents is not None else trainable
    for name in agents:
        if name not in trainable:
            raise ConfigError(f"evaluation.agents: {name!r} is not a trainable agent of the config ({trainable})")
    fixed = config.fixed_agent_ids()
    field = list(ev.field) if ev.field is not None else fixed
    for name in field:
        if name not in fixed:
            raise ConfigError(f"evaluation.field: {name!r} is not a scripted or frozen agent of the config ({fixed}); "
                              f"the config names agents only (snapshots are named on the command line: SP3 ruling 9)")
    if not field:
        raise ConfigError("evaluation.field is empty: the config has no scripted / frozen agents; add one "
                          "(e.g. kind: scripted, class: colosseum.players.RandomBot) or disable evaluation")
    out: dict[str, PulseAgent] = {}
    for name in agents:
        acfg = config.get_agent_config(name)
        roles = tuple(player_roles[name])
        if ev.layouts is not None:
            unknown = sorted(set(ev.layouts) - set(spec.layouts))
            if unknown:
                raise ConfigError(f"evaluation.layouts {unknown}: not layouts of the game {sorted(spec.layouts)}")
            layouts = [n for n in ev.layouts if any(s.role in roles for s in spec.layouts[n])]
        else:
            layouts = playable_layouts(spec, acfg.matchmaking, roles)
        if not layouts:
            raise ConfigError(f"evaluation: agent {name!r} (roles {list(roles)}) has a seat in none of the layouts "
                              f"{ev.layouts or sorted(spec.layouts)}")
        seats = {f"{name}@v0": _Seat(name, "ckpt_v0", roles),
                 **{f: _Seat(f, FIXED_NETWORK_ID, tuple(player_roles[f])) for f in field}}
        try:
            pulse_blocks(spec, layouts, f"{name}@v0", field, seats, blocks=1, seed=0, prefix="pulse")
        except ConfigError as e:
            raise ConfigError(f"evaluation: agent {name!r}: {e}") from e
        out[name] = PulseAgent(roles=roles, networks=acfg.networks.model_dump(mode="json", by_alias=True),
                               layouts=tuple(layouts))
    return PulsePlan(agents=out, field=field)


def describe_pulse(plan: PulsePlan, config: ColosseumConfig, spec: GameSpec, player_roles) -> str:
    """One ``validate`` line with the effective pulse settings and the matches per snapshot."""
    ev = config.evaluation
    matches = {}
    for name, agent in plan.agents.items():
        seats = {f"{name}@v0": _Seat(name, "ckpt_v0", agent.roles),
                 **{f: _Seat(f, FIXED_NETWORK_ID, tuple(player_roles[f])) for f in plan.field}}
        blocks = pulse_blocks(spec, agent.layouts, f"{name}@v0", plan.field, seats, blocks=ev.blocks, seed=0,
                              prefix="pulse")
        matches[name] = sum(len(b.lineups) for b in blocks)
    per = ", ".join(f"{matches[n]} matches per snapshot of '{n}'" for n in plan.agents)
    return (f"pulse: agents {list(plan.agents)}, every {ev.every} saved snapshot(s) (the final one always); field "
            f"{plan.field}; layouts { {n: list(a.layouts) for n, a in plan.agents.items()} }; {ev.blocks} blocks per "
            f"composition: {per}; {ev.num_envs} envs, {'greedy' if ev.deterministic else 'stochastic'}; backlog "
            f"{ev.max_backlog}")


class PulseFeeder:
    """Main-process side (module docstring). Main loop only."""

    def __init__(self, agents: Collection[str], every: int, max_backlog: int) -> None:
        self._agents = set(agents)
        self._every = int(every)
        self._max_backlog = int(max_backlog)
        self._saved: dict[str, int] = defaultdict(int)
        self._waiting: deque[PulseRequest] = deque()
        self._in_flight: dict[tuple[str, str], PulseRequest] = {}
        self._offered: dict[str, list[int]] = defaultdict(list)
        self._evaluated: dict[str, list[int]] = defaultdict(list)
        self._skipped: dict[str, list[int]] = defaultdict(list)
        self._final: dict[str, str] = {}
        self.skipped = 0
        self.evaluated = 0
        self.changed = False

    def offer(self, agent_id: str, checkpoint_id: str, version: int, env_steps: int,
              state_dict: Mapping[str, np.ndarray], *, final: bool = False) -> bool:
        """A saved snapshot; True when it was queued for evaluation."""
        if agent_id not in self._agents:
            return False
        self._saved[agent_id] += 1
        if not final and self._saved[agent_id] % self._every != 0:
            return False
        if final:
            self._final[agent_id] = checkpoint_id
        if len(self._waiting) >= self._max_backlog:
            old = self._waiting.popleft()
            self._skipped[old.agent_id].append(old.version)
            self.skipped += 1
            logger.info(f"pulse: snapshot {old.checkpoint_id} of {old.agent_id} skipped (backlog of "
                        f"{self._max_backlog} full; newer snapshots go first)")
        self._waiting.append(PulseRequest(agent_id, checkpoint_id, int(version), int(env_steps), dict(state_dict),
                                          final=bool(final)))
        self._offered[agent_id].append(int(version))
        self.changed = True
        return True

    def pump(self, requests: Any) -> int:
        """Move waiting requests into ``requests`` while it takes them (non-blocking); returns how many moved."""
        moved = 0
        while self._waiting:
            try:
                requests.put_nowait(self._waiting[0])
            except queue.Full:
                break
            request = self._waiting.popleft()
            self._in_flight[(request.agent_id, request.checkpoint_id)] = request
            moved += 1
        return moved

    def done(self, message: PulseResults) -> None:
        self._in_flight.pop((message.agent_id, message.checkpoint_id), None)
        self._evaluated[message.agent_id].append(int(message.version))
        self.evaluated += 1
        self.changed = True

    @property
    def pending(self) -> int:
        return len(self._waiting) + len(self._in_flight)

    def _unevaluated(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for request in [*self._in_flight.values(), *self._waiting]:
            out[request.agent_id].append(request.version)
        return {a: sorted(v) for a, v in out.items()}

    def unevaluated_final(self) -> list[tuple[str, str]]:
        """``(agent, checkpoint id)`` of final snapshots still waiting or in flight."""
        left = {(r.agent_id, r.checkpoint_id) for r in [*self._in_flight.values(), *self._waiting]}
        return [(a, c) for a, c in self._final.items() if (a, c) in left]

    def state(self) -> dict[str, Any]:
        unevaluated = self._unevaluated()
        agents = sorted(self._agents)
        return {"offered": {a: list(self._offered.get(a, [])) for a in agents if a in self._offered},
                "evaluated": {a: sorted(self._evaluated.get(a, [])) for a in agents if a in self._offered},
                "skipped": {a: sorted(self._skipped.get(a, [])) for a in agents if a in self._offered},
                "unevaluated": {a: unevaluated.get(a, []) for a in agents if a in self._offered}}


class _PulsePool:
    """``PlayerPool``: the current snapshot under ``(agent, ckpt_v<N>)``, the field under ``(name, "fixed")``."""

    def __init__(self, spec: GameSpec, field_specs: FixedPlayers) -> None:
        self._fixed: dict[tuple[str, str], Any] = {}
        for name, bot in field_specs.bots.items():
            self._fixed[(name, FIXED_NETWORK_ID)] = ScriptedPlayer(functools.partial(make_bot, bot, spec))
        for name, frozen in field_specs.frozen.items():
            self._fixed[(name, FIXED_NETWORK_ID)] = build_frozen_model(None, frozen, spec)
        self._candidate: tuple[tuple[str, str], Any] | None = None

    def set_candidate(self, key: tuple[str, str], model: Any) -> None:
        self._candidate = (key, model)

    def get(self, agent_id: str, network_id: str) -> Any:
        if self._candidate is not None and self._candidate[0] == (agent_id, network_id):
            return self._candidate[1]
        return self._fixed.get((agent_id, network_id))


def _put(q: Any, item: Any, stop_event: Any) -> bool:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False


def pulse_target(*, log_dir: str | None = None, **kwargs: Any) -> None:
    """Process entry point (``mp.Process`` target): logging and the child signal policy first."""
    run_child("pulse", log_dir, pulse_main, **kwargs)


def pulse_main(config: ColosseumConfig, *, requests: Any, results: Any, stop_event: Any, field_specs: FixedPlayers,
               field: Sequence[str], agents: Mapping[str, PulseAgent], run_id: str) -> None:
    """The pulse process body (module docstring); returns on ``None`` or at the stop."""
    from colosseum.core.registry import env_spec, make_env
    from colosseum.core.threads import configure_torch_threads
    from colosseum.eval import play_lineups

    configure_torch_threads(1)
    spec = env_spec(config)
    ev = config.evaluation
    seed = config.training.seed if config.training.seed is not None else 0
    prefix = f"pulse-{run_id[:12]}"
    pool = _PulsePool(spec, field_specs)
    field_players = {name: config_player(config, spec, name, field_specs) for name in field}
    logger.info(f"pulse: agents {list(agents)}, field {list(field)}, {ev.blocks} blocks per composition")
    while not stop_event.is_set():
        try:
            request = requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if request is None:
            return
        agent = agents[request.agent_id]
        name = f"{request.agent_id}@v{request.version}"
        origin = f"pulse snapshot {request.checkpoint_id} of {request.agent_id}"
        model = build_frozen_model(None, FrozenSpec(agent_id=request.agent_id, roles=agent.roles,
                                                    networks=dict(agent.networks), model_state=request.state_dict,
                                                    source=origin, networks_source=origin), spec)
        pool.set_candidate((request.agent_id, request.checkpoint_id), model)
        candidate = TournamentPlayer(name=name, display=name, kind="snapshot", agent_id=request.agent_id,
                                     network_id=request.checkpoint_id, roles=agent.roles, lineage=request.agent_id,
                                     source=PlayerSource("model", origin=origin, networks=dict(agent.networks)),
                                     version=request.version)
        blocks = pulse_blocks(spec, agent.layouts, name, field, {name: candidate, **field_players}, blocks=ev.blocks,
                              seed=seed, prefix=prefix)
        played: list[MatchResult] = []
        for task in make_tasks(blocks, ev.num_envs):
            if stop_event.is_set():
                return                                   # stopping: this snapshot stays unevaluated
            got = play_lineups(env_fn=lambda: make_env(config), pool=pool,
                               lineups=[lu for b in task for lu in b.lineups], num_envs=ev.num_envs,
                               seed=task[0].seed, deterministic=ev.deterministic,
                               max_idle_steps=config.env.max_idle_steps, source="pulse")
            for result in got:
                result.env_steps = request.env_steps
            played.extend(got)
        logger.info(f"pulse: {name} evaluated ({len(played)} matches)")
        if not _put(results, PulseResults(request.agent_id, request.checkpoint_id, request.version, played, True),
                    stop_event):
            return
```

- [ ] **Step 5: Validation**

`src/colosseum/core/validation.py`, in `validate_config` (train scope, after `_describe_retention(config, report)`):

```python
        _check_evaluation(config, spec, player_roles, report)
```
```python
def _check_evaluation(config: ColosseumConfig, spec: GameSpec, player_roles: dict[str, list[str]],
                      report: ValidationReport) -> None:
    """Pulse (spec block 6): the field, the agents and the role coverage (``arena.pulse.pulse_plan``), and the
    effective settings in one line."""
    if not config.evaluation.enabled:
        return
    from colosseum.arena.pulse import describe_pulse, pulse_plan

    plan = pulse_plan(config, spec, player_roles)
    report.lines.append(describe_pulse(plan, config, spec, player_roles))
```

- [ ] **Step 6: Launcher wiring**

`src/colosseum/launcher.py`:
1. Constants: `_PULSE_RESULTS_QUEUE_SIZE = 64` (one message per snapshot).
2. `__init__`: `self._pulse_feeder = None`, `self._pulse_plan = None`, `self._pulse_requests: mp.Queue | None = None`, `self._pulse_results: mp.Queue | None = None`.
3. `launch()`, after `self._checkpoint_meta = ...`:
```python
        if cfg.evaluation.enabled:
            from colosseum.arena.pulse import PulseFeeder, pulse_plan

            self._pulse_plan = pulse_plan(cfg, setup.spec, setup.player_roles)
            self._pulse_feeder = PulseFeeder(self._pulse_plan.agents, cfg.evaluation.every,
                                             cfg.evaluation.max_backlog)
            self._pulse_requests = mp.Queue(maxsize=1)
            self._pulse_results = mp.Queue(maxsize=_PULSE_RESULTS_QUEUE_SIZE)
```
   and after `self._all_queues = [...]` add `self._all_queues += [q for q in (self._pulse_requests, self._pulse_results) if q is not None]`.
4. `_start_children`, after the worker loop:
```python
        if self._pulse_feeder is not None:
            from colosseum.arena.pulse import pulse_target
            from colosseum.players.registry import FixedPlayers

            fixed, names = setup.fixed, self._pulse_plan.field
            field_specs = FixedPlayers(bots={n: fixed.bots[n] for n in names if n in fixed.bots},
                                       frozen={n: fixed.frozen[n] for n in names if n in fixed.frozen},
                                       roles={n: fixed.roles[n] for n in names})
            pulse_proc = mp.Process(target=pulse_target, name="pulse", daemon=True, kwargs=dict(
                log_dir=str(self._run_dir.logs), config=cfg, requests=self._pulse_requests,
                results=self._pulse_results, stop_event=self._stop_event, field_specs=field_specs, field=list(names),
                agents=self._pulse_plan.agents, run_id=self._run_dir.run_id))
            start_process(pulse_proc)
            self._supervisor.add("pulse", pulse_proc)
            logger.info(f"Pulse started for {list(self._pulse_plan.agents)} against {names}")
```
5. `_save_checkpoint`:
```python
    def _save_checkpoint(self, payload: dict) -> None:
        aid = payload["agent_id"]
        env_steps = int(self._env_step_counter.value)
        meta = {**self._checkpoint_meta.get(aid, {}), "env_steps": env_steps}
        checkpoint_id = self._coordinator.save_checkpoint_payload(payload, meta_extra=meta)
        if self._pulse_feeder is not None:
            self._pulse_feeder.offer(aid, checkpoint_id, int(payload["policy_version"]), env_steps,
                                     payload["model_state"], final=bool(payload.get("final", False)))
```
6. New methods:
```python
    def _main_produced_queues(self) -> list:
        """Queues this process writes to and must release itself (``release_command_queues``)."""
        return [*self._command_queues, *([self._pulse_requests] if self._pulse_requests is not None else [])]

    def _pump_pulse(self) -> None:
        """Hand waiting snapshots to the pulse process; journal its results; keep pulse_state.json current."""
        feeder = self._pulse_feeder
        if feeder is None or self._pulse_requests is None:
            return
        feeder.pump(self._pulse_requests)
        writer = getattr(self, "_match_log_writer", None)
        for message in self._drain(self._pulse_results, "pulse-results", lambda: ["pulse"]):
            if writer is not None:
                writer.submit(message.results)     # dropped results are counted by the writer (T1.3)
            feeder.done(message)
        if self._hub is not None:          # R10; if Part 1 has no such hook, add it to MetricsHub here
            self._hub.set_system_value("pulse_skipped_snapshots", feeder.skipped)
            self._hub.set_system_value("pulse_evaluated_snapshots", feeder.evaluated)
            self._hub.set_system_value("pulse_pending_snapshots", feeder.pending)
        if feeder.changed:
            self._write_pulse_state()

    def _write_pulse_state(self) -> None:
        from colosseum.metrics.jsonl import write_json_atomic

        write_json_atomic(self._run_dir.pulse_state_path, self._pulse_feeder.state())
        self._pulse_feeder.changed = False

    def _finish_pulse(self) -> None:
        """At the end: the final pulse state, and an INFO per final snapshot the run gave no time to evaluate."""
        if self._pulse_feeder is None:
            return
        for agent_id, checkpoint_id in self._pulse_feeder.unevaluated_final():
            logger.info(f"pulse: the final snapshot {checkpoint_id} of {agent_id} was not evaluated (the run "
                        f"stopped first); evaluate it with colosseum tournament")
        self._write_pulse_state()
```
7. Call `self._pump_pulse()` in `_monitor_loop` right after `self._drain_results()`; in `_shutdown`'s `poll`, add `self._pump_pulse` to the drains tuple; in `_finish_metrics`, call `self._pump_pulse()` after `self._drain_results()` and `self._finish_pulse()` before `self._hub.close(...)`.
8. `_shutdown`: `produced_here = {id(q) for q in self._main_produced_queues()}`; `_release_queues`: `produced_here = {id(q) for q in self._main_produced_queues()}` and `release_command_queues(self._main_produced_queues(), _QUEUE_RELEASE_SEC)`; at its end also `self._pulse_requests = self._pulse_results = None`.

The pulse process is in the supervisor, so `first_failure` stops the run when it dies ("pulse died (exit 1), see <run>/logs/pulse.log"), `wait_all` / `kill_remaining` stop it with the rest, and `_exit_code_after_shutdown` treats it like a worker.

- [ ] **Step 7: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp4_pulse.py tests/integration/test_sp4_pulse_runs.py tests/unit/test_bench_throughput.py tests/unit/test_sp2_launcher_lifecycle.py -q`
Expected: all pass (the integration file takes about 40 s).

- [ ] **Step 8: Full fast suite + ruff**

The pulse touches the launcher's process lifecycle: also run the slow suite once (`env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v`, about 7 min) and report its result.

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/arena/pulse.py src/colosseum/core/config.py src/colosseum/core/validation.py \
        src/colosseum/launcher.py src/colosseum/core/run_dir.py scripts/bench_throughput.py tests/game_helpers.py \
        tests/unit/test_sp4_pulse.py tests/integration/test_sp4_pulse_runs.py
git commit -m "feat: pulse (evaluate new snapshots against a fixed field during training)"
git push
```

---
## Contract notes

Requirements on other parts, additions and deviations against the overview contract, and proposed rulings ("what — why — cost if wrong"). The controller folds accepted items into `00-overview.md` («Contract amendments») before execution.

### Requirements on Part 1 (and one on Part 3)

- **R1 — `RunDir.run_id` is get-or-create** (T1.3). `RunDir.open(root).run_id` returns the run's stored id and, for a run dir without one (an SP3 run dir used by `tournament --run`, a fake run in tests), creates and stores one. Used by T4.1 (the manager's `run_id`, so only this run's final is protected), T5.4 (the journal's `RunInfo` of a `--run`), T6.1 (the pulse block prefix) and the T4.1 integration test.
- **R2 — launcher attributes and shutdown order** (T1.3, T2.2): `Launcher._match_log_writer: MatchLogWriter | None` and `Launcher._live_ratings: LiveRatings | None`, both created in `launch()` before `_start_children`. The writer is stopped after `_finish_metrics` has drained the last results (T6.1's pulse results arrive during the shutdown poll and are submitted to it). Other names are fine if reported; this part reads them with `getattr(..., None)`.
- **R3 — `MatchLogReader.close()`** (T1.2); a context manager in addition is welcome.
- **R4 — tournaments in the journal** (T1.2):
  - the tournament id is the block-id prefix before the first `/` (block ids `"<tournament>/<stage>/<layout>/<candidate>/<composition>/<index>"`, uids `"<block>/r<rotation>"`; tournament names are path components, so they contain no `/`); `MatchLog.comparison_table(tournament=T)` and `MatchLogReader.blocks(T)` select `source == "tournament"` matches whose `block` starts with `T + "/"`; `blocks(T)` returns full block ids;
  - `MatchLogReader.match_results(*, source: str, tournament: str | None = None) -> list[MatchResult]` rebuilds complete `MatchResult`s (uid, block, seed, env_steps, seats with `version_start` / `version_end`, teams, length) of detailed matches (tournament and pulse matches are never compacted); T5.4 (resume) and the T6.1 tests use it;
  - `MatchLog(path, run)` with a `run_id` already in `runs` is idempotent, and a second writer process (a tournament next to a training run) works (WAL, `busy_timeout`); tournament matches are written even under the disk guard (spec block 1, item 4).
- **R5 — `Uncertainty.draws: Mapping[EntityKey, np.ndarray]`** (T2.1): the Laplace / bootstrap draws in the lineage-contrast scale (the scale of `lcb`), one array per entity, the same length within a fit. `retention.combine_scores` averages layouts per draw (spec block 4: «со средним по весам matchmaking.layouts, вычисленным на каждой выборке»); with one layout only `lcb` is read.
- **R6 — `ScoreSummary.by_entity: Mapping[EntityKey, tuple[int, float, float]]`** (T2.1): `(games, mean, standard error)` of every entity's homogeneous-team score in a `score` layout (top-k and `top:N` in games with `score` layouts only).
- **R7 — `bt.BootstrapData.from_table(table, layout, *, sources)`** (T2.1): Poisson row weights over a `ComparisonTable` (`top:N`).
- **R8 — `bt.BootstrapData.from_blocks(results_by_block, *, strata=None)`** (T2.1): whole blocks resampled, within `strata[block_id]` when given (the tournament's BT intervals, spec block 2 «по блокам seed + ротации»).
- **R9 — `bt.default_options(anchor: str | None = None) -> BTOptions`** (T2.1): the module's prior / smoothness constants (T8.2 fixes them) with the given anchor.
- **R10 — `MetricsHub.set_system_value(key, value)`** (T1.3 / T3.1): a value merged into every later `system` record (T6.1: `pulse_skipped_snapshots`, `pulse_evaluated_snapshots`, `pulse_pending_snapshots`). If Part 1 adds its own system fields another way, T6.1 adds this small hook itself.
- **R11 — `TransitivityResult.p_value: float | None`** (None when no pair has enough games) and JSON-convertible `worst_pairs` (T2.1).
- **R12 (Part 3, T7.2) — `colosseum.arena.report.render.render_tournament_html(report: dict) -> str`**, rendering the report schema of T5.3. Until it exists `TournamentDir.write_reports` writes `report.json` / `report.txt` and logs one INFO; T7.2 removes the `ImportError` guard (the TODO hook) so a missing renderer becomes an error again. Part 3's run report reads tournaments from `runs/x/tournaments/<name>/report.json` (same schema).

### Additions to the contract (all marked `*` in the tasks)

- **C1 (T4.1)** `arena.retention.combine_scores(agent_id, *, fits, uncertainty, scores, layout_weights, quantile)`, `snapshot_scores(snapshot, agent_id, *, layout_weights, quantile)`, `describe_retention(checkpoint)`. `retain` and the four dataclasses are the contract's.
- **C2 (T4.1, T4.2)** `checkpoint_manager.set_pinned(ckpt_dir, pinned)`, `read_snapshots(agent_dir)`, `link_or_copy(src, dst)` (public; `_link_or_copy` stays an alias), `parse_snapshot_ref(text)`, `pin_snapshot(run_dir, agent_id, version, *, pinned=True)`, `list_snapshots(run_dir)`, `SNAPSHOT_REF_RE`. The new `CheckpointManager` arguments are keyword-only; a manager with `run_id=None` keeps the SP3 rule (every final protected).
- **C3 (T4.1)** `Coordinator(..., run_id=None)`, `set_ratings_source(fn)`, `add_evict_listener(fn)`, `snapshot_scores(agent_id)`; the manager's `scores_fn(agent_id)` keeps the contract signature and returns scores for every rated version of the agent (`ckpt_v<N>` keys), the manager picks what it holds. `Launcher._wire_retention()`, `_reapply_retention()`. `distributed_checkpoint_manager(config, base_dir, run_id=None)`.
- **C4 (T5.1, T5.2)** `schedule.Block`, `STAGES` and `stage_seed` are created in T5.1 (the engine and its tests need the type), the builders in T5.2. Deviation: the builders take `players: Mapping[name, SeatPlayer]` (agent id, network id, roles) instead of `roles` — a lineup needs the seat ids; plus `first_index` (halving rounds) and `prefix` (tournament name / pulse); new helpers `compositions`, `role_group`, `group_label`, `check_role_groups`, `SeatPlayer`.
- **C5 (T5.3)** Deviation: `stats.candidate_scores(blocks, results_by_block, *, names, lineage, group_of, metric_of, deterministic)` returns tables per (stage, layout, group); `paired_bootstrap` is `CandidateTable.bootstrap_cells / bootstrap_scores`; `confirmed(diff_replicates, *, level, conflict)` takes the difference replicates (`select_winner` builds them over several layouts); `repeated_pairs` is `CandidateTable.repeated` (+ `repeated_conflict`). Additions: `combine`, `select_winner`, `Selection`, `Combined`, `cell_interval`, `block_values`, `quantiles`, `SELF`, `PAIRED_BOOTSTRAP_SAMPLES`.
- **C6 (T5.3)** `modes.run(mode, *, spec, players, candidates, field, settings, play, bt_fn=None)`, `TournamentSettings`, `bt_section`, `report_text`, `MODES`, `REPORT_FORMAT`, `DEFAULT_BLOCKS / DEFAULT_HALVING_BLOCKS / DEFAULT_CONFIRM_BLOCKS / DEFAULT_FINALISTS`, and the report schema in T5.3 (binding for Part 3).
- **C7 (T5.1)** `TournamentPlayer` gains `agent_id`, `network_id`, `source` (`PlayerSource`), `version`, `run`, `run_agent`; `resolve_players(config, spec, request, *, fixed, link=None, rng=None, allow_empty_field=False)` replaces the contract's `resolve_candidates` / `resolve_field` (one pass decides both: a name is in one list only); `RunSelection`, `PlayerRequest`, `entity_hash`, `user_player`, `config_player`, `snapshot_player`, `parse_snapshots`, `select_snapshots`, `top_snapshot_lcb`, `load_run_config`, `run_agent`, `player_to_dict` / `player_from_dict`.
- **C8 (T5.1)** `engine.play_blocks(..., log_dir=None, source="tournament")`, `LazyPlayerPool`, `make_tasks`, `TournamentError`, `MODEL_CACHE_SIZE`, `TASK_MATCHES`.
- **C9 (T5.1)** `eval.parse_agent_spec`, `balanced_match_count`, `load_frozen_player`; `load_eval_model(..., use_entry=False)`; `load_player(..., fixed=None)`; `evaluate(..., fixed=None)`; `play_lineups(*, env_fn, models=None, lineups, ..., pool=None, source="train")`; `record.record(..., fixed=None)` (`record.parse_player` removed); `players.registry.load_frozen(..., use_entry=True)`, `checkpoint_roles` (public); `ValidationReport.fixed`; `MatchRunner(..., snapshot_fallback=True)`.
- **C10 (T6.1)** `pulse_main(config, *, requests, results, stop_event, field_specs, field, agents, run_id)` (the contract's `log_dir` moves to `pulse_target`); `PulseRequest.final`; `PulseAgent`, `PulsePlan`, `pulse_plan`, `describe_pulse`, `pulse_blocks`, `PulseResults`, `PulseFeeder`, `PULSE_STATE_FILE`; `RunDir.pulse_state_path`; `runs/<name>/pulse_state.json` (`offered` / `evaluated` / `skipped` / `unevaluated` versions per agent; Part 3 marks unevaluated snapshots from it).
- **C11 (T5.4)** `RunDir.tournaments_dir` (added here if T7.1 has not); `run.TournamentOptions`, `run_tournament`, `TournamentDir`, `BlockRunner`, `diff_fields`, `TOURNAMENT_FILE`, `TOURNAMENT_FORMAT`; the tournament dir layout listed in T5.4.
- **C12 (T4.1)** `check_distributed_scope` refuses `checkpoint.keep_top_k > 0` here (the key becomes usable in T4.1); T8.1 adds `evaluation.enabled` and the bench config check.

### Deviations from the brief / overview

- **V1 — pulse results travel on their own queue** (`Launcher._pulse_results`, one message per snapshot), not on the workers' results queue. Why: the workers' queue is `put_nowait` with counted drops, the pulse needs a blocking put; and `_drain_results` feeds the coordinator (online ELO, PFSP, matchmaker) and the episodes aggregator, which pulse matches must not touch (spec block 2: the online ELO stays as it is). Cost if wrong: one more queue in the launcher's release lists.
- **V2 — task dependencies:** T5.1 also uses T2.1 (`bt`) and T4.1 (`combine_scores`) for `--snapshots top:N`; the table order already runs both first. T5.4 uses T7.1's `RunDir.tournaments_dir` before T7.1 exists, so T5.4 adds it.
- **V3 — size:** T5.1 is the largest task (shared loading + players + engine). If the controller prefers, split it at the step boundary: T5.1a = Steps 4–5 (shared loading, `MatchRunner(snapshot_fallback)`, `test_sp4_eval_dedup.py`), T5.1b = Steps 6–8 (`schedule.Block`, players, engine and their tests).

### Proposed rulings

- **D1 (T4.1)** Top-k is chosen among all rated snapshots, including those already kept by `keep_last` / pins — why: "the k best" should not depend on the other rules; cost: when the best are also the newest, fewer distinct snapshots stay (the cap is the real disk bound).
- **D2 (T4.1)** Over the cap with top-k on, unrated keep_every-only snapshots are evicted before rated ones (ascending LCB) — why: no evidence keeps them (a snapshot with fewer than `top_k_min_games` games is "new" and protected anyway); cost: rare loss of an old snapshot that had games but no interval yet.
- **D3 (T4.1)** A `CheckpointManager` without a run id protects every final (the SP3 rule) — why: SP3 storage tests and direct users keep working; the launcher and the distributed learner always pass the run id; cost: none in runs.
- **D4 (T4.1)** The "protected snapshots exceed max_snapshots" WARNING is logged once per agent and process — why: retention runs after every save and refit; cost: a later, larger overflow is not re-announced (validate warns up front).
- **D5 (T5.3)** Repeated deterministic pairs (checked only with `--deterministic`) count as one block: the cell's interval is the Wilson interval of one observation (pairwise) or the point (score), and `confirmed` is false whenever the winner and the runner-up differ in a column where either repeats — why: spec block 3 «в ДИ и confirmed считаются одним блоком»; a bootstrap of identical blocks would otherwise claim certainty; cost: deterministic tournaments of deterministic games rarely confirm anything (the report says "not enough randomness").
- **D6 (T5.3)** Halving keeps `max(finalists, n - floor(n/2))`, stops when at most `finalists` are left (one round when there are no more to begin with), breaks ties by input order; the stage A table shows the finalists over the blocks they all played, eliminated candidates appear only in `rounds` with their scores — why: tables need common blocks for paired statistics; cost: the report has no intervals for eliminated candidates.
- **D7 (T5.3)** `round-robin` runs only on two-team layouts (a ConfigError names the others) — why: "all pairs" has no natural seat assignment in FFA or score layouts; cost: FFA matrices need `gauntlet` with each participant as candidate.
- **D8 (T5.2, T5.3)** `score` layouts with a field: the field member takes one seat and the candidate the others (one match per seat the member can take); the candidate's score weights its homogeneous team as one more "lineage" — why: the spec's «по одному месту, ротация мест, как cross-play SP2»; cost: in teams of 3+ the mix "one helper, rest candidate" is the only cross-play composition.
- **D9 (T5.3)** The tournament's BT ratings are fitted per pairwise layout on all its stages' matches; every non-snapshot participant counts as a fixed agent for the anchor rule — why: one rating table per layout, the run's anchor rule; cost: eliminated candidates' ratings rest on stage A games only (their `games` column shows it).
- **D10 (T5.4)** `--output DIR` is laid out like a run dir (`DIR/matches.db`, `DIR/tournaments/<name>/`), so one DIR holds several tournaments; `--output` with `--run` is a usage error — why: the same layout for reports and dashboards (Part 3); cost: one directory level more than "everything in DIR".
- **D11 (T5.4)** Resume reuses the stored players (not re-resolved) and compares the request without `--num-workers`, `--num-envs`, `--pin` — why: `top:N` of a training run moves; execution settings do not change results by block; cost: stochastic sampling differs by worker split (blocks stay valid).
- **D12 (T5.4)** `--pin` pins only a confirmed `select` winner that is a run snapshot (WARNING otherwise) — why: «тихого выбора нет»; cost: the owner pins an unconfirmed winner by hand (`colosseum pin`).
- **D13 (T5.4)** Layout weights: the first `--run` agent's effective `matchmaking.layouts` (else the config's); equal weights with an INFO when a chosen layout has none — why: spec block 3 «с весами `matchmaking.layouts` агента»; cost: a mixed candidate set is weighted by one run's taste.
- **D14 (T5.1)** Candidates vs field: an explicit `--field` name is only in the field; the default field is every scripted / frozen agent of the config not given as `-a name`; the first `--run`'s names have no prefix even with an alias; the default alias is the run dir's name; two equal aliases are an error — why: spec block 3's naming (`main@v1200`, `<alias>/main@v1200`) and one list per name; cost: a user wanting a config bot as both candidate and opponent must copy it.
- **D15 (T5.1)** `--snapshots every:K` takes every K-th stored snapshot counting from the newest (always included) — why: stored versions are irregular (top-k, `keep_every`); cost: not "versions divisible by K".
- **D16 (T5.4, T6.1)** `tournament --seed` defaults to 0 (resumable, reproducible); pulse seeds use `training.seed` (0 when null) — why: every snapshot must meet the same seeds; cost: the pulse seed set does not change between runs unless the seed does.
- **D17 (T5.1)** Seeds are 32-bit: the stage code in the top two bits, a keyed bijection of the index in the low 30 bits (indices < 2^30) — why: "never colliding by construction" for A / B / P and no repeats within a domain; cost: seeds are >= 2^30 (fine for numpy / gymnasium seeding).
- **D18 (T5.1)** A worker task holds whole blocks with at least `4 * num_envs` matches; its torch sampling seed is the first block's seed — why: full env batches without splitting blocks; cost: stochastic sampling is reproducible only for the same `num_envs` / `num_workers` split.
- **D19 (T6.1)** The pulse is offered the final snapshot regardless of `every`, its results go to the journal only, one message per snapshot, and `pulse_state.json` records the bookkeeping — why: the final snapshot matters most; no SP2 rating changes; cost: the final is usually unevaluated in local runs (the stop comes first; logged and recorded).
- **D20 (T5.1)** `-a <frozen name>=<path>` ignores the frozen entry (spec §8), but `-a <trainable name>=<.pt>` still uses the trainable agent's architecture and roles (the SP2 eval rule) — why: only the frozen overlay was a surprise; cost: none known.
- **D21 (T4.2)** `colosseum pin` is idempotent (pinning a pinned snapshot, unpinning an unpinned one), never builds a `CheckpointManager` on the run dir, and a training run sees the change at its next retention pass — why: safe on live runs; cost: a pin written between the trainer's pin refresh and its eviction of that snapshot is lost (the CLI's success message comes before; the window is one retention pass).
